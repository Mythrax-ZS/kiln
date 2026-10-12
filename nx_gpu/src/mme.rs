//! The macro engine (MME) of the Maxwell 3D class: a small processor that runs the macros a driver
//! uploads, and turns a call with a few parameters into many method writes.
//!
//! The instruction set is that of the Fermi macro engine, which the Maxwell one is (docs/ASSUMPTIONS.md
//! A-077). It is implemented from Mesa's description of it (`src/nouveau/mme`: `mme_fermi.c` for the
//! encoding and `mme_fermi_sim.c` for the behavior, MIT licence); the execution loop, with its delay
//! slots and the way an exit is delayed, follows that simulator.
//!
//! A macro has eight registers, `r0` always zero, and a queue of parameters: the words the program
//! sent with the call. `r1` starts as the first parameter. Each instruction computes a value with
//! an ALU operation and then *assigns* it: into a register, as the address of the method the macro
//! is writing, or as data of a write to that method (`emit`). A branch has a delay slot, and an
//! instruction that ends the macro ends it one instruction later.

use std::fmt;

/// How many instructions a macro may run before the interpreter gives up (a macro that loops
/// forever would otherwise hang the run).
pub const STEP_LIMIT: u64 = 1 << 22;

/// What a macro reads and writes: the state of the engine it belongs to.
pub trait Host {
    /// The value of the state register of the method at `method` (a byte offset).
    fn state(&mut self, method: u32) -> u32;
    /// A write of `data` to the method at `method` (a byte offset).
    fn emit(&mut self, method: u32, data: u32) -> Result<(), String>;
}

/// What stopped a macro from finishing.
#[derive(Debug, PartialEq, Eq)]
pub enum Fault {
    /// The macro asked for a parameter the call did not have.
    TooFewParameters,
    /// The macro ended and parameters were left over.
    TooManyParameters(usize),
    /// An instruction this code does not know.
    Unknown(String),
    /// The macro ran past the code that is loaded, or branched outside it.
    OutOfCode(i64),
    /// More than [`STEP_LIMIT`] instructions.
    StepLimit,
    /// A write the macro made failed.
    Write(String),
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::TooFewParameters => {
                write!(f, "the macro read more parameters than it was called with")
            }
            Fault::TooManyParameters(n) => write!(f, "the macro ended with {n} parameters unread"),
            Fault::Unknown(s) => write!(f, "{s}"),
            Fault::OutOfCode(ip) => {
                write!(f, "the macro went to instruction {ip}, outside its code")
            }
            Fault::StepLimit => write!(f, "the macro ran for more than {STEP_LIMIT} instructions"),
            Fault::Write(s) => write!(f, "{s}"),
        }
    }
}

/// What a macro did, for the counts.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Instructions executed.
    pub steps: u64,
    /// Method writes made.
    pub emitted: u64,
}

// Operations (bits 0 to 2 of an instruction).
const OP_ALU_REG: u32 = 0;
const OP_ADD_IMM: u32 = 1;
const OP_MERGE: u32 = 2;
const OP_BFE_LSL_IMM: u32 = 3;
const OP_BFE_LSL_REG: u32 = 4;
const OP_STATE: u32 = 5;
const OP_BRANCH: u32 = 7;

// ALU operations (bits 17 to 21 of an ALU instruction).
const ALU_ADD: u32 = 0;
const ALU_ADDC: u32 = 1;
const ALU_SUB: u32 = 2;
const ALU_SUBB: u32 = 3;
const ALU_XOR: u32 = 8;
const ALU_OR: u32 = 9;
const ALU_AND: u32 = 10;
const ALU_AND_NOT: u32 = 11;
const ALU_NAND: u32 = 12;

// Assignments (bits 4 to 6).
const ASSIGN_LOAD: u32 = 0;
const ASSIGN_MOVE: u32 = 1;
const ASSIGN_MOVE_SET_ADDRESS: u32 = 2;
const ASSIGN_LOAD_EMIT: u32 = 3;
const ASSIGN_MOVE_EMIT: u32 = 4;
const ASSIGN_LOAD_SET_ADDRESS: u32 = 5;
const ASSIGN_MOVE_SET_ADDRESS_LOAD_EMIT: u32 = 6;
const ASSIGN_MOVE_SET_ADDRESS_LOAD_EMIT_HIGH: u32 = 7;

fn bits(word: u32, lo: u32, hi: u32) -> u32 {
    (word >> lo) & (u32::MAX >> (31 - (hi - lo)))
}

/// The 18-bit signed immediate (bits 14 to 31).
fn immediate(word: u32) -> i32 {
    (word as i32) >> 14
}

fn mask(size: u32) -> u32 {
    if size >= 32 {
        u32::MAX
    } else {
        (1 << size) - 1
    }
}

/// Extract `size` bits of `value` from `src_bit` and shift them to `dst_bit`.
fn bfe_lsl(value: u32, src_bit: u32, dst_bit: u32, size: u32) -> u32 {
    if dst_bit > 31 || src_bit > 31 {
        return 0;
    }
    ((value >> src_bit) & mask(size)).wrapping_shl(dst_bit)
}

struct Machine<'a> {
    params: &'a [u32],
    next_param: usize,
    regs: [u32; 8],
    carry: u32,
    /// The method writes go to: the byte address and how far it moves after each write.
    method: Option<(u32, u32)>,
    emitted: u64,
}

impl Machine<'_> {
    fn reg(&self, r: u32) -> u32 {
        self.regs[r as usize]
    }

    fn set_reg(&mut self, r: u32, v: u32) {
        if r != 0 {
            self.regs[r as usize] = v;
        }
    }

    fn param(&mut self) -> Result<u32, Fault> {
        let v = *self
            .params
            .get(self.next_param)
            .ok_or(Fault::TooFewParameters)?;
        self.next_param += 1;
        Ok(v)
    }

    fn set_method(&mut self, v: u32) {
        self.method = Some(((v & 0xFFF) << 2, ((v >> 12) & 0xF) * 4));
    }

    fn emit(&mut self, host: &mut dyn Host, data: u32) -> Result<(), Fault> {
        // A write before any method was set is dropped; what the engine does is not known.
        if let Some((address, step)) = self.method {
            host.emit(address, data).map_err(Fault::Write)?;
            self.emitted += 1;
            self.method = Some((address.wrapping_add(step), step));
        }
        Ok(())
    }

    /// The value an operation computes.
    fn compute(&mut self, host: &mut dyn Host, word: u32) -> Result<u32, Fault> {
        let op = bits(word, 0, 2);
        let x = self.reg(bits(word, 11, 13));
        let y = self.reg(bits(word, 14, 16));
        Ok(match op {
            OP_ALU_REG => match bits(word, 17, 21) {
                ALU_ADD => {
                    let r = x.wrapping_add(y);
                    self.carry = u32::from(r < x);
                    r
                }
                ALU_ADDC => {
                    let r = x.wrapping_add(y).wrapping_add(self.carry);
                    self.carry = u32::from(r < x);
                    r
                }
                ALU_SUB => {
                    let r = x.wrapping_sub(y);
                    self.carry = u32::from(r > x);
                    r
                }
                ALU_SUBB => {
                    let r = x.wrapping_sub(y).wrapping_sub(self.carry);
                    self.carry = u32::from(r > x);
                    r
                }
                ALU_XOR => x ^ y,
                ALU_OR => x | y,
                ALU_AND => x & y,
                ALU_AND_NOT => x & !y,
                ALU_NAND => !(x & y),
                other => return Err(Fault::Unknown(format!("the macro ALU operation {other}"))),
            },
            OP_ADD_IMM => x.wrapping_add(immediate(word) as u32),
            OP_MERGE => {
                let (src_bit, size, dst_bit) =
                    (bits(word, 17, 21), bits(word, 22, 26), bits(word, 27, 31));
                (x & !(mask(size).wrapping_shl(dst_bit)))
                    | ((y >> src_bit) & mask(size)).wrapping_shl(dst_bit)
            }
            OP_BFE_LSL_IMM => bfe_lsl(y, x, bits(word, 27, 31), bits(word, 22, 26)),
            OP_BFE_LSL_REG => bfe_lsl(y, bits(word, 17, 21), x, bits(word, 22, 26)),
            OP_STATE => {
                let index = x.wrapping_add(immediate(word) as u32);
                host.state(index.wrapping_mul(4))
            }
            other => return Err(Fault::Unknown(format!("the macro operation {other}"))),
        })
    }

    /// Execute one instruction that is not a branch.
    fn assign(&mut self, host: &mut dyn Host, word: u32) -> Result<(), Fault> {
        let value = self.compute(host, word)?;
        let dst = bits(word, 8, 10);
        match bits(word, 4, 6) {
            ASSIGN_LOAD => {
                let p = self.param()?;
                self.set_reg(dst, p);
            }
            ASSIGN_MOVE => self.set_reg(dst, value),
            ASSIGN_MOVE_SET_ADDRESS => {
                self.set_reg(dst, value);
                self.set_method(value);
            }
            ASSIGN_LOAD_EMIT => {
                let p = self.param()?;
                self.set_reg(dst, p);
                self.emit(host, value)?;
            }
            ASSIGN_MOVE_EMIT => {
                self.set_reg(dst, value);
                self.emit(host, value)?;
            }
            // The register takes the parameter and the method address the computed value. (Mesa's
            // simulator stores the computed value here; the name and the other emulators agree
            // with this; docs/ASSUMPTIONS.md A-077.)
            ASSIGN_LOAD_SET_ADDRESS => {
                let p = self.param()?;
                self.set_reg(dst, p);
                self.set_method(value);
            }
            ASSIGN_MOVE_SET_ADDRESS_LOAD_EMIT => {
                self.set_reg(dst, value);
                self.set_method(value);
                let p = self.param()?;
                self.emit(host, p)?;
            }
            ASSIGN_MOVE_SET_ADDRESS_LOAD_EMIT_HIGH => {
                self.set_reg(dst, value);
                self.set_method(value);
                self.emit(host, (value >> 12) & 0x3F)?;
            }
            _ => unreachable!("three bits"),
        }
        Ok(())
    }
}

const ALU_NAMES: [&str; 13] = [
    "add", "addc", "sub", "subb", "?4", "?5", "?6", "?7", "xor", "or", "and", "and_not", "nand",
];

const ASSIGN_NAMES: [&str; 8] = [
    "load",
    "move",
    "move_set_maddr",
    "load_emit",
    "move_emit",
    "load_set_maddr",
    "move_set_maddr_load_emit",
    "move_set_maddr_load_emit_high",
];

/// One instruction as text: the operation, then (unless it is a branch) what is done with the
/// result, then the exit mark.
pub fn disassemble(word: u32) -> String {
    let r = |at: u32| format!("r{}", bits(word, at, at + 2));
    let imm = immediate(word);
    let op = match bits(word, 0, 2) {
        OP_ALU_REG => {
            let alu = bits(word, 17, 21) as usize;
            format!(
                "{} {} {}",
                ALU_NAMES.get(alu).copied().unwrap_or("?"),
                r(11),
                r(14)
            )
        }
        OP_ADD_IMM => format!("add_imm {} {imm}", r(11)),
        OP_MERGE => format!(
            "merge {} {} (dst bit {}, size {}, src bit {})",
            r(11),
            r(14),
            bits(word, 27, 31),
            bits(word, 22, 26),
            bits(word, 17, 21)
        ),
        OP_BFE_LSL_IMM => format!(
            "bfe_lsl_imm {} {} (dst bit {}, size {})",
            r(11),
            r(14),
            bits(word, 27, 31),
            bits(word, 22, 26)
        ),
        OP_BFE_LSL_REG => format!(
            "bfe_lsl_reg {} {} (src bit {}, size {})",
            r(11),
            r(14),
            bits(word, 17, 21),
            bits(word, 22, 26)
        ),
        OP_STATE => format!("state {} {imm}", r(11)),
        OP_BRANCH => format!(
            "{} {} {imm:+}{}",
            if bits(word, 4, 4) != 0 { "bnz" } else { "bz" },
            r(11),
            if bits(word, 5, 5) != 0 {
                " no_delay"
            } else {
                ""
            }
        ),
        other => format!("op{other}"),
    };
    let assign = if bits(word, 0, 2) == OP_BRANCH {
        String::new()
    } else {
        format!("; {} {}", ASSIGN_NAMES[bits(word, 4, 6) as usize], r(8))
    };
    let end = if bits(word, 7, 7) != 0 {
        " ; end_next"
    } else {
        ""
    };
    format!("{op}{assign}{end}")
}

/// The instructions of `code` from `start`, `count` of them, one per line with their index.
pub fn listing(code: &[u32], start: usize, count: usize) -> String {
    code.iter()
        .enumerate()
        .skip(start)
        .take(count)
        .map(|(i, w)| format!("    {i:>4}: {w:08x}  {}\n", disassemble(*w)))
        .collect()
}

/// Run the macro that starts at instruction `start` of `code` with `params` as its parameters.
pub fn run(
    code: &[u32],
    start: usize,
    params: &[u32],
    host: &mut dyn Host,
) -> Result<Summary, Fault> {
    let mut m = Machine {
        params,
        next_param: 0,
        regs: [0; 8],
        carry: 0,
        method: None,
        emitted: 0,
    };
    // The first parameter is already in r1 when the macro starts.
    let first = m.param().unwrap_or(0);
    m.set_reg(1, first);

    let fetch = |ip: i64| -> Result<u32, Fault> {
        usize::try_from(ip)
            .ok()
            .and_then(|i| code.get(i).copied())
            .ok_or(Fault::OutOfCode(ip))
    };
    let mut ip = start as i64;
    let mut next_ip = ip;
    let mut steps = 0u64;
    // The instruction about to run is the delay slot of a branch.
    let mut delay_slot = false;
    // An exit flag on a branch with a delay slot is ignored, and so is the one of its slot.
    let mut ignore_next_exit = false;
    let mut exit = false;
    while !exit {
        let word = fetch(ip)?;
        steps += 1;
        if steps > STEP_LIMIT {
            return Err(Fault::StepLimit);
        }
        let is_branch = bits(word, 0, 2) == OP_BRANCH;
        if is_branch && delay_slot {
            return Err(Fault::Unknown(
                "a macro branch in the delay slot of a branch".to_string(),
            ));
        }
        if !delay_slot {
            next_ip = ip + 1;
        }
        step(&mut m, host, word, ip, &mut next_ip)?;
        delay_slot = is_branch && bits(word, 5, 5) == 0;
        ip = if delay_slot { ip + 1 } else { next_ip };
        let ends = bits(word, 7, 7) != 0;
        if ends && delay_slot {
            ignore_next_exit = true;
            continue;
        }
        exit = ends && !ignore_next_exit;
        ignore_next_exit = false;
    }
    // The instruction after the one that ended the macro still runs (its delay slot).
    let word = fetch(ip)?;
    steps += 1;
    let mut after = ip + 1;
    step(&mut m, host, word, ip, &mut after)?;
    let left = m.params.len() - m.next_param;
    if left > 0 {
        return Err(Fault::TooManyParameters(left));
    }
    Ok(Summary {
        steps,
        emitted: m.emitted,
    })
}

/// Execute the instruction `word` at `ip`; a branch that is taken sets `next_ip`.
fn step(
    m: &mut Machine<'_>,
    host: &mut dyn Host,
    word: u32,
    ip: i64,
    next_ip: &mut i64,
) -> Result<(), Fault> {
    if bits(word, 0, 2) == OP_BRANCH {
        let value = m.reg(bits(word, 11, 13));
        let not_zero = bits(word, 4, 4) != 0;
        if (value != 0) == not_zero {
            *next_ip = ip + i64::from(immediate(word));
        } else if bits(word, 5, 5) == 0 {
            // Not taken, with a delay slot: the slot runs and then the macro goes on after it.
            // (Mesa's simulator would run the slot a second time here.)
            *next_ip = ip + 2;
        }
        Ok(())
    } else {
        m.assign(host, word)
    }
}

/// A small assembler, for tests and for reading a macro in a trace.
#[cfg(test)]
pub mod asm {
    use super::*;

    pub const END_NEXT: u32 = 1 << 7;

    // The operations the tests of other modules use.
    pub const ADD: u32 = ALU_ADD;
    pub const OR: u32 = ALU_OR;
    pub const LOAD: u32 = ASSIGN_LOAD;
    pub const MOVE: u32 = ASSIGN_MOVE;
    pub const MOVE_EMIT: u32 = ASSIGN_MOVE_EMIT;
    pub const MOVE_SET_ADDRESS: u32 = ASSIGN_MOVE_SET_ADDRESS;

    /// The method-address word that selects `method` (a byte offset) and steps `inc` words.
    pub fn address(method: u32, inc: u32) -> u32 {
        (method >> 2) | inc << 12
    }

    fn reg(r: u32, at: u32) -> u32 {
        (r & 7) << at
    }

    fn imm(v: i32) -> u32 {
        ((v as u32) & 0x3FFFF) << 14
    }

    pub fn alu(op: u32, assign: u32, dst: u32, a: u32, b: u32) -> u32 {
        OP_ALU_REG | assign << 4 | reg(dst, 8) | reg(a, 11) | reg(b, 14) | op << 17
    }

    pub fn add_imm(assign: u32, dst: u32, a: u32, v: i32) -> u32 {
        OP_ADD_IMM | assign << 4 | reg(dst, 8) | reg(a, 11) | imm(v)
    }

    pub fn state(assign: u32, dst: u32, a: u32, v: i32) -> u32 {
        OP_STATE | assign << 4 | reg(dst, 8) | reg(a, 11) | imm(v)
    }

    /// A bit-field instruction: `regs` are (dst, a, b), `bits` are (source bit, size, destination bit).
    fn field(op: u32, assign: u32, regs: [u32; 3], bits: [u32; 3]) -> u32 {
        op | assign << 4
            | reg(regs[0], 8)
            | reg(regs[1], 11)
            | reg(regs[2], 14)
            | bits[0] << 17
            | bits[1] << 22
            | bits[2] << 27
    }

    pub fn merge(
        assign: u32,
        dst: u32,
        a: u32,
        b: u32,
        dst_bit: u32,
        size: u32,
        src_bit: u32,
    ) -> u32 {
        field(OP_MERGE, assign, [dst, a, b], [src_bit, size, dst_bit])
    }

    pub fn bfe_lsl_imm(assign: u32, dst: u32, a: u32, b: u32, dst_bit: u32, size: u32) -> u32 {
        field(OP_BFE_LSL_IMM, assign, [dst, a, b], [0, size, dst_bit])
    }

    pub fn bfe_lsl_reg(assign: u32, dst: u32, a: u32, b: u32, src_bit: u32, size: u32) -> u32 {
        field(OP_BFE_LSL_REG, assign, [dst, a, b], [src_bit, size, 0])
    }

    /// A branch on `r` being (non)zero by `offset` instructions from itself.
    pub fn branch(not_zero: bool, no_delay: bool, r: u32, offset: i32) -> u32 {
        OP_BRANCH | u32::from(not_zero) << 4 | u32::from(no_delay) << 5 | reg(r, 11) | imm(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::asm::*;
    use super::*;
    use std::collections::BTreeMap;

    /// A host with a few state registers that records the writes.
    #[derive(Default)]
    struct Recorder {
        state: BTreeMap<u32, u32>,
        writes: Vec<(u32, u32)>,
    }

    impl Host for Recorder {
        fn state(&mut self, method: u32) -> u32 {
            self.state.get(&method).copied().unwrap_or(0)
        }

        fn emit(&mut self, method: u32, data: u32) -> Result<(), String> {
            self.state.insert(method, data);
            self.writes.push((method, data));
            Ok(())
        }
    }

    fn exec(code: &[u32], params: &[u32]) -> (Result<Summary, Fault>, Recorder) {
        let mut host = Recorder::default();
        let r = run(code, 0, params, &mut host);
        (r, host)
    }

    #[test]
    fn the_first_parameter_is_in_r1_and_a_macro_writes_what_it_computes() {
        // r2 = r1 + 5; set the method to 0x1618 from the immediate; emit r2 there.
        let code = [
            add_imm(ASSIGN_MOVE, 2, 1, 5),
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 3, 0, address(0x1618, 0) as i32),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 4, 2, 0) | END_NEXT,
            // The instruction after the one that ends the macro still runs.
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 4, 2, 0),
        ];
        let (r, host) = exec(&code, &[10]);
        assert_eq!(r.unwrap().emitted, 2);
        assert_eq!(host.writes, vec![(0x1618, 15), (0x1618, 15)]);
    }

    #[test]
    fn the_method_address_steps_after_each_write_by_its_increment() {
        let code = [
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 2, 0, address(0x1000, 1) as i32),
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 7),
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 8),
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 9) | END_NEXT,
            add_imm(ASSIGN_MOVE, 3, 0, 0),
        ];
        let (r, host) = exec(&code, &[0]);
        r.unwrap();
        assert_eq!(host.writes, vec![(0x1000, 7), (0x1004, 8), (0x1008, 9)]);
    }

    #[test]
    fn loading_takes_parameters_in_order_and_the_load_forms_emit() {
        let code = [
            // r2 = param 1 (r1 holds param 0)
            alu(ALU_ADD, ASSIGN_LOAD, 2, 0, 0),
            // method = 0x2000, then write r1 + 0 while loading r3 = param 2
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 4, 0, address(0x2000, 0) as i32),
            alu(ALU_ADD, ASSIGN_LOAD_EMIT, 3, 1, 0),
            // write the next parameter (3) without touching registers: set the method again
            // from r4 and load-emit
            alu(ALU_ADD, ASSIGN_MOVE_SET_ADDRESS_LOAD_EMIT, 5, 4, 0) | END_NEXT,
            alu(ALU_ADD, ASSIGN_MOVE, 6, 2, 3),
        ];
        let (r, host) = exec(&code, &[100, 20, 3, 4]);
        // Four parameters were all read: 100 (r1), 20 (r2), 3 (r3), 4 (the emit).
        r.unwrap();
        assert_eq!(host.writes, vec![(0x2000, 100), (0x2000, 4)]);
    }

    #[test]
    fn asking_for_a_parameter_that_is_not_there_or_leaving_one_is_a_fault() {
        let code = [
            alu(ALU_ADD, ASSIGN_LOAD, 2, 0, 0) | END_NEXT,
            alu(ALU_ADD, ASSIGN_MOVE, 3, 0, 0),
        ];
        assert_eq!(exec(&code, &[1]).0.unwrap_err(), Fault::TooFewParameters);
        assert_eq!(
            exec(&code, &[1, 2, 3]).0.unwrap_err(),
            Fault::TooManyParameters(1)
        );
        assert!(exec(&code, &[1, 2]).0.is_ok());
    }

    #[test]
    fn the_alu_does_what_its_names_say_and_carries() {
        // r1 = 0xFFFF_FFFF (parameter). r2 = r1 + 1 carries; r3 = 0 + 0 + carry = 1.
        let code = [
            alu(ALU_ADD, ASSIGN_MOVE, 2, 1, 1),
            alu(ALU_ADDC, ASSIGN_MOVE, 3, 0, 0),
            // subtraction borrows: 0 - 1
            alu(ALU_SUB, ASSIGN_MOVE, 4, 0, 1),
            alu(ALU_SUBB, ASSIGN_MOVE, 5, 0, 0),
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 7, 0, address(0x100, 1) as i32),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 2, 0),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 3, 0),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 4, 0),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 5, 0) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let (r, host) = exec(&code, &[0xFFFF_FFFF]);
        r.unwrap();
        let sent: Vec<u32> = host.writes.iter().map(|w| w.1).collect();
        assert_eq!(sent, vec![0xFFFF_FFFE, 1, 1, 0xFFFF_FFFF], "{sent:x?}");
    }

    #[test]
    fn the_logic_operations_and_the_bit_fields() {
        let code = [
            // r1 = 0b1100 (parameter); r2 = 0b1010 through add
            add_imm(ASSIGN_MOVE, 2, 0, 0b1010),
            alu(ALU_XOR, ASSIGN_MOVE, 3, 1, 2),     // 0110
            alu(ALU_AND_NOT, ASSIGN_MOVE, 4, 1, 2), // 0100
            alu(ALU_NAND, ASSIGN_MOVE, 5, 1, 2),    // !(1000)
            // merge: r6 = (r1 with bits 4..8 replaced by r2 bits 1..5)
            merge(ASSIGN_MOVE, 6, 1, 2, 4, 4, 1),
            // bfe_lsl_imm: take 3 bits of r2 from the bit r0 (=0) and put them at bit 8
            bfe_lsl_imm(ASSIGN_MOVE, 7, 0, 2, 8, 3),
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 1, 0, address(0x300, 1) as i32),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 3, 0),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 4, 0),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 5, 0),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 6, 0),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 7, 0) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let (r, host) = exec(&code, &[0b1100]);
        r.unwrap();
        let sent: Vec<u32> = host.writes.iter().map(|w| w.1).collect();
        assert_eq!(
            sent,
            vec![
                0b0110,
                0b0100,
                !0b1000u32,
                0b1100 | (0b0101 << 4),
                0b010 << 8
            ]
        );
    }

    #[test]
    fn bfe_lsl_with_a_register_for_the_destination_bit() {
        // r1 = parameter 4; r2 = 0xF0; take 4 bits from bit 4 of r2 and put them at bit r1.
        let code = [
            add_imm(ASSIGN_MOVE, 2, 0, 0xF0),
            bfe_lsl_reg(ASSIGN_MOVE, 3, 1, 2, 4, 4),
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 4, 0, address(0x40, 0) as i32),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 3, 0) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let (r, host) = exec(&code, &[4]);
        r.unwrap();
        assert_eq!(host.writes, vec![(0x40, 0xF << 4)]);
    }

    #[test]
    fn state_reads_the_register_of_a_method() {
        // Write 77 to method 0x500 through the macro, then read it back with STATE (the index is
        // the method divided by four) and write it to 0x600.
        let code = [
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 2, 0, address(0x500, 0) as i32),
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 77),
            state(ASSIGN_MOVE, 4, 0, 0x500 / 4),
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 5, 0, address(0x600, 0) as i32),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 4, 0) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let (r, host) = exec(&code, &[0]);
        r.unwrap();
        assert_eq!(host.writes, vec![(0x500, 77), (0x600, 77)]);
    }

    #[test]
    fn a_branch_has_a_delay_slot_unless_it_says_otherwise() {
        // r1 = 1. The branch is taken (r1 != 0) over two instructions; the delay slot runs.
        let code = [
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 2, 0, address(0x80, 1) as i32),
            branch(true, false, 1, 4),
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 1), // delay slot: runs
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 2), // skipped
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 3), // skipped
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 4) | END_NEXT, // target
            add_imm(ASSIGN_MOVE, 3, 0, 0),
        ];
        let (r, host) = exec(&code, &[1]);
        r.unwrap();
        let sent: Vec<u32> = host.writes.iter().map(|w| w.1).collect();
        assert_eq!(sent, vec![1, 4]);
        // With no delay the slot is skipped too; not taken falls through.
        let mut code2 = code;
        code2[1] = branch(true, true, 1, 4);
        let (r, host) = exec(&code2, &[1]);
        r.unwrap();
        let sent: Vec<u32> = host.writes.iter().map(|w| w.1).collect();
        assert_eq!(sent, vec![4]);
        let (r, host) = exec(&code, &[0]);
        r.unwrap();
        let sent: Vec<u32> = host.writes.iter().map(|w| w.1).collect();
        assert_eq!(sent, vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_loop_counts_down_with_a_backward_branch() {
        // r1 = 3 (the count). Each trip writes the count and subtracts one; a branch back while
        // it is not zero.
        let code = [
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 2, 0, address(0x10, 0) as i32),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 1, 0),
            add_imm(ASSIGN_MOVE, 1, 1, -1),
            branch(true, true, 1, -2),
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let (r, host) = exec(&code, &[3]);
        r.unwrap();
        let sent: Vec<u32> = host.writes.iter().map(|w| w.1).collect();
        assert_eq!(sent, vec![3, 2, 1]);
    }

    #[test]
    fn an_exit_flag_in_a_delay_slot_ends_the_macro_after_the_next_instruction() {
        // A branch that is always taken, with the exit flag on its delay slot: the slot runs,
        // then the instruction at the target, and the macro is over.
        let code = [
            add_imm(MOVE_SET_ADDRESS, 2, 0, address(0x20, 1) as i32),
            add_imm(MOVE_EMIT, 3, 0, 6),            // 1: the target
            branch(false, false, 0, -1),            // 2: back to 1 (r0 is zero)
            add_imm(MOVE_EMIT, 3, 0, 7) | END_NEXT, // 3: the slot
            add_imm(MOVE_EMIT, 3, 0, 8),            // 4: never run
        ];
        let (r, host) = exec(&code, &[0]);
        r.unwrap();
        let sent: Vec<u32> = host.writes.iter().map(|w| w.1).collect();
        assert_eq!(sent, vec![6, 7, 6]);
    }

    #[test]
    fn an_exit_flag_on_a_delayed_branch_and_in_its_slot_are_both_ignored() {
        // A loop of ten trips whose branch carries the exit flag: it still runs ten trips, with
        // the write in the slot each time, and leaves by the exit flag after it.
        let code = [
            add_imm(MOVE_SET_ADDRESS, 2, 0, address(0x30, 0) as i32),
            add_imm(MOVE, 1, 0, 10),                // r1 = 10
            add_imm(MOVE, 1, 1, -1),                // 2: r1 -= 1
            branch(true, false, 1, -1) | END_NEXT,  // 3: while r1 != 0, back to 2
            add_imm(MOVE_EMIT, 3, 0, 1) | END_NEXT, // 4: the slot, writes 1 each trip
            add_imm(MOVE, 0, 0, 0) | END_NEXT,      // 5
            add_imm(MOVE, 0, 0, 0),
        ];
        let mut host = Recorder::default();
        run(&code, 0, &[0], &mut host).unwrap();
        assert_eq!(host.writes.len(), 10, "{:?}", host.writes);
    }

    #[test]
    fn a_branch_in_a_delay_slot_is_a_fault() {
        let code = [
            branch(false, false, 0, 2),
            branch(false, false, 0, 2),
            add_imm(MOVE, 3, 0, 5) | END_NEXT,
            add_imm(MOVE, 3, 0, 5),
            add_imm(MOVE, 3, 0, 5),
        ];
        let e = exec(&code, &[0]).0.unwrap_err();
        assert!(
            matches!(&e, Fault::Unknown(s) if s.contains("delay slot")),
            "{e:?}"
        );
    }

    #[test]
    fn a_runaway_or_stray_macro_is_stopped() {
        // A branch to itself without delay never ends.
        let spin = [branch(false, true, 0, 0), alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0)];
        assert_eq!(exec(&spin, &[0]).0.unwrap_err(), Fault::StepLimit);
        // Running off the end of the code.
        let off = [add_imm(ASSIGN_MOVE, 2, 0, 1)];
        assert!(matches!(
            exec(&off, &[0]).0.unwrap_err(),
            Fault::OutOfCode(1)
        ));
        // The operation nobody knows.
        let unknown = [6u32 | END_NEXT, 0];
        assert!(matches!(
            exec(&unknown, &[0]).0.unwrap_err(),
            Fault::Unknown(_)
        ));
        // A failing write is reported with its text.
        struct Refuses;
        impl Host for Refuses {
            fn state(&mut self, _: u32) -> u32 {
                0
            }
            fn emit(&mut self, _: u32, _: u32) -> Result<(), String> {
                Err("no".to_string())
            }
        }
        let code = [
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 2, 0, address(0x10, 0) as i32),
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 1) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        assert_eq!(
            run(&code, 0, &[0], &mut Refuses).unwrap_err(),
            Fault::Write("no".into())
        );
    }

    #[test]
    fn a_macro_starts_where_the_start_table_says() {
        let code = [
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
            add_imm(ASSIGN_MOVE_SET_ADDRESS, 2, 0, address(0x70, 0) as i32),
            add_imm(ASSIGN_MOVE_EMIT, 3, 0, 9) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let mut host = Recorder::default();
        run(&code, 2, &[0], &mut host).unwrap();
        assert_eq!(host.writes, vec![(0x70, 9)]);
    }

    #[test]
    fn the_high_form_sends_the_bits_above_the_method_number() {
        // MOVE_SET_MADDR_LOAD_EMIT_HIGH: sets the method from the value and writes bits 12..17
        // of the value to it.
        let value = (0x1618 >> 2) | (0x2A << 12);
        let code = [
            add_imm(ASSIGN_MOVE_SET_ADDRESS_LOAD_EMIT_HIGH, 2, 0, value) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let mut host = Recorder::default();
        run(&code, 0, &[0], &mut host).unwrap();
        assert_eq!(host.writes, vec![(0x1618, 0x2A)]);
    }

    #[test]
    fn the_load_and_set_address_form_loads_the_register_and_sets_the_address() {
        let code = [
            add_imm(ASSIGN_LOAD_SET_ADDRESS, 2, 0, address(0x90, 0) as i32),
            alu(ALU_OR, ASSIGN_MOVE_EMIT, 0, 2, 0) | END_NEXT,
            alu(ALU_OR, ASSIGN_MOVE, 0, 0, 0),
        ];
        let mut host = Recorder::default();
        run(&code, 0, &[0, 55], &mut host).unwrap();
        assert_eq!(host.writes, vec![(0x90, 55)], "r2 took the parameter");
    }
}
