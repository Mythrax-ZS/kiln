//! The macro engine of the Turing and later 3D and compute classes, the Switch 2's Ampere ones
//! among them ("MME64"; docs/ASSUMPTIONS.md A-208). It does what the Maxwell one ([`crate::mme`])
//! does, turning a call with a few parameters into method writes, with another instruction set.
//!
//! An instruction is 96 bits, uploaded as three words, the first the highest. It has two ALUs and
//! two outputs, which run together: both ALUs read their operands before either writes its
//! destination, then each output may set the method the macro writes to (with an increment applied
//! after each write) and emit a value to it. There are 24 registers; the other register numbers
//! name zero, the immediates of the instruction and the two parameters it loads. A predicate may
//! turn each of the four parts on or off. An instruction marked as the end ends the macro after the
//! next instruction that runs (a delay slot), and a branch to the offset -4096 stops it at once.
//!
//! The semantics are Mesa's (`src/nouveau/mme/mme_tu104_sim.c`, MIT licence), with what the
//! Switch 2's NVN2 needs on top: a loop of zero passes skips its body, only a taken branch with
//! bit 14 of its immediate keeps an end mark from ending the macro, and a parameter read past those
//! of the call is 0 (NVN2's macros fall through into the next and read more than they are
//! given; A-237).

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

/// Words of the data RAM, the memory macros read and write with `DREAD` and `DWRITE`.
pub const DATA_RAM_WORDS: usize = 0xC00;
/// Instructions the instruction RAM holds.
pub const INSTRUCTIONS: usize = 0x1000;
/// Entries of the table of where each macro starts.
pub const START_ENTRIES: usize = 0x80;
/// How many instructions a macro may run before the interpreter gives up.
pub const STEP_LIMIT: u64 = 1 << 22;

/// The methods (word indices) that load the code and the start table.
pub mod method {
    pub const INSTRUCTION_RAM_POINTER: u32 = 0x45;
    pub const INSTRUCTION_RAM: u32 = 0x46;
    pub const START_ADDRESS_RAM_POINTER: u32 = 0x47;
    pub const START_ADDRESS_RAM: u32 = 0x48;
    /// `SET_MME_MEM_ADDRESS_A`/`_B`: the GPU address the DMA reads below start at.
    pub const MEM_ADDRESS_A: u32 = 0x154;
    pub const MEM_ADDRESS_B: u32 = 0x155;
    /// `SET_MME_DATA_RAM_ADDRESS`: where in the data RAM `MME_DMA_READ` puts its words.
    pub const DATA_RAM_ADDRESS: u32 = 0x156;
    /// `MME_DMA_READ(count)`: words from memory into the data RAM.
    pub const DMA_READ: u32 = 0x157;
    /// `MME_DMA_READ_FIFOED(count)`: words from memory onto the parameters of the running macro.
    pub const DMA_READ_FIFOED: u32 = 0x158;
    /// The first method that calls a macro: `0xE00 + 2 j` starts macro `j`, the odd one after it
    /// gives it more parameters.
    pub const CALL_FIRST: u32 = 0xE00;
}

/// What a macro reads and writes besides its registers.
pub trait Host {
    /// The value of the method `method` (a word index) of the engine.
    fn state(&mut self, method: u32) -> u32;
    /// A write of `data` to the method `method` (a word index). A write may add words to the
    /// parameters still to be read (`MME_DMA_READ_FIFOED`).
    fn emit(&mut self, method: u32, data: u32, params: &mut VecDeque<u32>) -> Result<(), String>;
    /// The data RAM of the channel the macro runs for.
    fn data_ram(&mut self) -> &mut [u32];
    /// Whether the macro's reads of the data RAM are to be printed (`NX_TRACE_MACRO`).
    fn traced(&self) -> bool {
        false
    }
}

/// What stopped a macro from finishing.
#[derive(Debug, PartialEq, Eq)]
pub enum Fault {
    /// An instruction or an operand this code does not know.
    Unknown(String),
    /// The macro ran past the instruction RAM.
    OutOfCode(usize),
    /// More than [`STEP_LIMIT`] instructions.
    StepLimit,
    /// A write the macro made failed.
    Write(String),
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Unknown(s) | Fault::Write(s) => write!(f, "{s}"),
            Fault::OutOfCode(ip) => {
                write!(f, "the macro went to instruction {ip:#x}, outside its code")
            }
            Fault::StepLimit => write!(f, "the macro ran for more than {STEP_LIMIT} instructions"),
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
    /// Parameters read past the end of those the call had (each read as 0).
    pub short_reads: u64,
}

// Register numbers that are not registers.
const R_ZERO: u8 = 24;
const R_IMM: u8 = 25;
const R_IMM_OTHER: u8 = 26;
const R_IMM32: u8 = 27;
const R_LOAD0: u8 = 28;
const R_LOAD1: u8 = 29;

// What an output takes its method or its data from.
const O_NONE: u8 = 0;
const O_ALU0: u8 = 1;
const O_ALU1: u8 = 2;
const O_LOAD0: u8 = 3;
const O_LOAD1: u8 = 4;
const O_IMM0: u8 = 5;
const O_IMM1: u8 = 6;
const O_IMM0_HIGH: u8 = 8;
const O_IMM1_HIGH: u8 = 9;
const O_IMM32: u8 = 10;

// ALU operations.
const ADD: u8 = 0;
const ADDC: u8 = 1;
const SUB: u8 = 2;
const SUBB: u8 = 3;
const MUL: u8 = 4;
const MULH: u8 = 5;
const MULU: u8 = 6;
const EXTENDED: u8 = 7;
const CLZ: u8 = 8;
const SLL: u8 = 9;
const SRL: u8 = 10;
const SRA: u8 = 11;
const AND: u8 = 12;
const NAND: u8 = 13;
const OR: u8 = 14;
const XOR: u8 = 15;
const MERGE: u8 = 16;
const SLT: u8 = 17;
const SLTU: u8 = 18;
const SLE: u8 = 19;
const SLEU: u8 = 20;
const SEQ: u8 = 21;
const STATE: u8 = 22;
const LOOP: u8 = 23;
const JAL: u8 = 24;
const BLT: u8 = 25;
const BLTU: u8 = 26;
const BLE: u8 = 27;
const BLEU: u8 = 28;
const BEQ: u8 = 29;
const DREAD: u8 = 30;
const DWRITE: u8 = 31;

const OP_NAMES: [&str; 32] = [
    "add", "addc", "sub", "subb", "mul", "mulh", "mulu", "extended", "clz", "sll", "srl", "sra",
    "and", "nand", "or", "xor", "merge", "slt", "sltu", "sle", "sleu", "seq", "state", "loop",
    "jal", "blt", "bltu", "ble", "bleu", "beq", "dread", "dwrite",
];

/// For each predicate mode, when each of the four parts (ALU 0, ALU 1, output 0, output 1) runs:
/// `T` when the predicate register is zero, `F` when it is not, `U` either way (Mesa's letters; a
/// fill macro of NVN2 writes the odd word left over its loop of pairs with an `FFFF` part on the
/// count's lowest bit).
const PREDICATES: [&[u8; 4]; 16] = [
    b"UUUU", b"TTTT", b"FFFF", b"TTUU", b"FFUU", b"TFUU", b"TUUU", b"FUUU", b"UUTT", b"UUTF",
    b"UUTU", b"UUFT", b"UUFF", b"UUFU", b"UUUT", b"UUUF",
];

/// The offset of a branch that stops the macro instead of going anywhere.
const STOP_OFFSET: i32 = -4096;

/// One ALU of an instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Alu {
    op: u8,
    dst: u8,
    a: u8,
    b: u8,
}

/// One output of an instruction: where it takes the method from and the data it emits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Out {
    method: u8,
    emit: u8,
}

/// A decoded instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Instruction {
    end: bool,
    predicate_mode: u8,
    predicate: u8,
    alu: [Alu; 2],
    imm: [u16; 2],
    out: [Out; 2],
    /// Which of the two parameters the instruction loads before it runs.
    loads: [bool; 2],
    /// The parts that run when the predicate register is non-zero and when it is zero, as bits
    /// (ALU 0, ALU 1, output 0, output 1).
    when_set: u8,
    when_clear: u8,
}

impl Instruction {
    /// Decode the three words of an instruction, in the order they are uploaded.
    fn decode(words: [u32; 3]) -> Instruction {
        let bits = u128::from(words[2]) | u128::from(words[1]) << 32 | u128::from(words[0]) << 64;
        let field = |lo: u32, len: u32| ((bits >> lo) & ((1u128 << len) - 1)) as u32;
        let alu = |at: u32| Alu {
            op: field(at, 5) as u8,
            dst: field(at + 5, 5) as u8,
            a: field(at + 10, 5) as u8,
            b: field(at + 15, 5) as u8,
        };
        let mut i = Instruction {
            end: field(0, 1) != 0,
            predicate_mode: field(1, 4) as u8,
            predicate: field(5, 5) as u8,
            alu: [alu(10), alu(46)],
            imm: [field(30, 16) as u16, field(66, 16) as u16],
            out: [
                Out {
                    method: field(82, 3) as u8,
                    emit: field(85, 4) as u8,
                },
                Out {
                    method: field(89, 3) as u8,
                    emit: field(92, 4) as u8,
                },
            ],
            ..Instruction::default()
        };
        for (n, (reg, out)) in [(R_LOAD0, O_LOAD0), (R_LOAD1, O_LOAD1)]
            .into_iter()
            .enumerate()
        {
            i.loads[n] = i.predicate == reg
                || i.alu.iter().any(|a| a.a == reg || a.b == reg)
                || i.out.iter().any(|o| o.method == out || o.emit == out);
        }
        let pattern = PREDICATES[usize::from(i.predicate_mode)];
        for (part, letter) in pattern.iter().enumerate() {
            if *letter != b'T' {
                i.when_set |= 1 << part;
            }
            if *letter != b'F' {
                i.when_clear |= 1 << part;
            }
        }
        i
    }

    fn imm32(&self) -> u32 {
        u32::from(self.imm[0]) << 16 | u32::from(self.imm[1])
    }
}

/// The macro code of a channel: the instruction RAM, decoded, and where each macro starts. The
/// channel's 3D and compute engines share it (A-208).
#[derive(Clone, Debug)]
pub struct Program {
    words: Vec<u32>,
    decoded: Vec<Instruction>,
    starts: [u32; START_ENTRIES],
    word_pointer: usize,
    start_pointer: usize,
}

impl Default for Program {
    fn default() -> Program {
        Program {
            words: vec![0; INSTRUCTIONS * 3],
            decoded: vec![Instruction::decode([0; 3]); INSTRUCTIONS],
            starts: [0; START_ENTRIES],
            word_pointer: 0,
            start_pointer: 0,
        }
    }
}

impl Program {
    /// A write to one of the four methods that load the code; `false` for another method.
    pub fn load(&mut self, method: u32, data: u32) -> bool {
        match method {
            method::INSTRUCTION_RAM_POINTER => self.word_pointer = data as usize * 3,
            method::INSTRUCTION_RAM => {
                let at = self.word_pointer;
                if let Some(word) = self.words.get_mut(at) {
                    *word = data;
                    let index = at / 3;
                    let w = &self.words[index * 3..index * 3 + 3];
                    self.decoded[index] = Instruction::decode([w[0], w[1], w[2]]);
                }
                self.word_pointer += 1;
            }
            method::START_ADDRESS_RAM_POINTER => self.start_pointer = data as usize % START_ENTRIES,
            method::START_ADDRESS_RAM => {
                self.starts[self.start_pointer] = data;
                self.start_pointer = (self.start_pointer + 1) % START_ENTRIES;
            }
            _ => return false,
        }
        true
    }

    /// Where macro `entry` starts.
    pub fn start(&self, entry: u32) -> u32 {
        self.starts[entry as usize % START_ENTRIES]
    }

    /// The instructions from `start`, `count` of them, one per line, for a trace.
    pub fn listing(&self, start: usize, count: usize) -> String {
        let mut lines = Vec::new();
        for ip in start..(start + count).min(INSTRUCTIONS) {
            let i = &self.decoded[ip];
            let alu = |a: &Alu, n: usize| {
                format!(
                    "{} r{} = r{}, r{} (imm {:#x})",
                    OP_NAMES[usize::from(a.op & 31)],
                    a.dst,
                    a.a,
                    a.b,
                    i.imm[n]
                )
            };
            lines.push(format!(
                "{ip:#05x}: {}pred {}:r{} | {} | {} | out {}/{} {}/{}",
                if i.end { "end " } else { "" },
                i.predicate_mode,
                i.predicate,
                alu(&i.alu[0], 0),
                alu(&i.alu[1], 1),
                i.out[0].method,
                i.out[0].emit,
                i.out[1].method,
                i.out[1].emit,
            ));
        }
        lines.join("\n")
    }
}

/// The macro code and the data RAM of each channel of a process, as the GPU keeps them: a
/// channel's 3D and compute engines share its code, and no channel sees another's (A-208).
#[derive(Debug, Default)]
pub struct Store {
    programs: std::collections::HashMap<u32, Arc<Program>>,
    data: std::collections::HashMap<u32, Box<[u32]>>,
}

impl Store {
    /// A write of `channel` to a method that loads code, applied to its program; `false` for
    /// another method.
    pub fn load(&mut self, channel: u32, method: u32, data: u32) -> bool {
        if !(method::INSTRUCTION_RAM_POINTER..=method::START_ADDRESS_RAM).contains(&method) {
            return false;
        }
        Arc::make_mut(self.programs.entry(channel).or_default()).load(method, data)
    }

    /// The code of `channel` (empty until it loads some).
    pub fn program(&mut self, channel: u32) -> Arc<Program> {
        Arc::clone(self.programs.entry(channel).or_default())
    }

    /// The data RAM of `channel`, made the first time.
    pub fn data_ram(&mut self, channel: u32) -> &mut [u32] {
        self.data
            .entry(channel)
            .or_insert_with(|| vec![0; DATA_RAM_WORDS].into_boxed_slice())
    }
}

/// Run the macro that starts at instruction `start`, with `params` as its parameters.
pub fn run(
    program: &Program,
    start: u32,
    params: Vec<u32>,
    host: &mut dyn Host,
) -> Result<Summary, Fault> {
    let mut params: VecDeque<u32> = params.into();
    let mut regs = [0u32; 24];
    let mut carry = 0u32;
    let (mut method, mut increment) = (0u32, 0u32);
    let (mut loop_count, mut loop_start, mut loop_end) = (0u32, 0usize, 0usize);
    let mut summary = Summary::default();
    let mut ip = start as usize;
    let mut ending = false;
    loop {
        if summary.steps >= STEP_LIMIT {
            return Err(Fault::StepLimit);
        }
        summary.steps += 1;
        let i = *program.decoded.get(ip).ok_or(Fault::OutOfCode(ip))?;
        let mut load = [0u32; 2];
        for (n, slot) in load.iter_mut().enumerate() {
            if i.loads[n] {
                *slot = match params.pop_front() {
                    Some(v) => v,
                    None => {
                        summary.short_reads += 1;
                        0
                    }
                };
            }
        }
        let read = |reg: u8, lane: usize, regs: &[u32; 24]| -> Result<u32, Fault> {
            Ok(match reg {
                0..=23 => regs[usize::from(reg)],
                R_ZERO => 0,
                R_IMM => i.imm[lane] as i16 as i32 as u32,
                R_IMM_OTHER => i.imm[1 - lane] as i16 as i32 as u32,
                R_IMM32 => i.imm32(),
                R_LOAD0 => load[0],
                R_LOAD1 => load[1],
                other => return Err(Fault::Unknown(format!("register {other} at {ip:#x}"))),
            })
        };
        let enabled = if i.predicate_mode == 0 {
            0xF
        } else if read(i.predicate, 0, &regs)? != 0 {
            i.when_set
        } else {
            i.when_clear
        };
        // Both ALUs read their operands before either writes.
        let mut operands = [(0u32, 0u32); 2];
        for (lane, operand) in operands.iter_mut().enumerate() {
            if enabled & (1 << lane) != 0 {
                *operand = (
                    read(i.alu[lane].a, lane, &regs)?,
                    read(i.alu[lane].b, lane, &regs)?,
                );
            }
        }
        let mut next = ip + 1;
        let mut keeps_going = false;
        let mut stopped = false;
        let mut results = [0u32; 2];
        for lane in 0..2 {
            if enabled & (1 << lane) == 0 {
                continue;
            }
            let (x, y) = operands[lane];
            let imm = i.imm[lane];
            let alu = i.alu[lane];
            let compare = |op: u8| match op {
                SLT | BLT => (x as i32) < (y as i32),
                SLTU | BLTU => x < y,
                SLE | BLE => (x as i32) <= (y as i32),
                SLEU | BLEU => x <= y,
                _ => x == y,
            };
            let value = match alu.op {
                ADD => {
                    let (sum, out) = x.overflowing_add(y);
                    carry = u32::from(out);
                    sum
                }
                ADDC => x.wrapping_add(y).wrapping_add(carry),
                SUB => {
                    let (difference, borrow) = x.overflowing_sub(y);
                    carry = u32::from(borrow);
                    difference
                }
                SUBB => x.wrapping_sub(y).wrapping_sub(carry),
                MUL | MULU => {
                    let product = if alu.op == MUL {
                        (i64::from(x as i32) * i64::from(y as i32)) as u64
                    } else {
                        u64::from(x) * u64::from(y)
                    };
                    carry = (product >> 32) as u32;
                    product as u32
                }
                MULH => carry,
                // The load barrier: the words of `MME_DMA_READ_FIFOED` are already among the
                // parameters when the write is made, so there is nothing to wait for.
                EXTENDED if x == 0x1000 => 0,
                EXTENDED => {
                    return Err(Fault::Unknown(format!(
                        "extended operation {x:#x}, {y:#x} at {ip:#x}"
                    )))
                }
                CLZ => x.leading_zeros(),
                SLL => x << (y & 31),
                SRL => x >> (y & 31),
                SRA => ((x as i32) >> (y & 31)) as u32,
                AND => x & y,
                NAND => !(x & y),
                OR => x | y,
                XOR => x ^ y,
                MERGE => {
                    let to = u32::from(imm >> 10 & 0x3F);
                    let count = u32::from(imm >> 5 & 0x1F);
                    let from = u32::from(imm & 0x1F);
                    if to >= 32 {
                        return Err(Fault::Unknown(format!("a merge into bit {to} at {ip:#x}")));
                    }
                    let mask = ((1u64 << count) - 1) as u32;
                    (x & !(mask << to)) | ((y >> from) & mask) << to
                }
                SLT | SLTU | SLE | SLEU | SEQ => {
                    if compare(alu.op) {
                        u32::MAX
                    } else {
                        0
                    }
                }
                STATE => host.state(x.wrapping_add(y) & 0xFFFF),
                LOOP => {
                    if loop_count != 0 || imm < 2 {
                        return Err(Fault::Unknown(format!(
                            "a loop of {imm} instructions inside another at {ip:#x}"
                        )));
                    }
                    if x == 0 {
                        // NVN2 loops zero times around reads of parameters that are not there.
                        next = ip + usize::from(imm);
                    } else {
                        loop_count = x - 1;
                        loop_start = ip;
                        loop_end = ip + usize::from(imm) - 1;
                    }
                    0
                }
                JAL => {
                    next = ip + usize::from(imm & 0x7FFF);
                    0
                }
                BLT | BLTU | BLE | BLEU | BEQ => {
                    if compare(alu.op) == (imm & 0x8000 != 0) {
                        let offset = i32::from((imm << 3) as i16) >> 3;
                        if offset == STOP_OFFSET {
                            stopped = true;
                        } else {
                            next = (ip as i64 + i64::from(offset)) as usize;
                            keeps_going = imm & 0x4000 != 0;
                        }
                    }
                    0
                }
                DREAD => {
                    let traced = host.traced();
                    let ram = host.data_ram();
                    let value = *ram.get(x as usize).ok_or_else(|| {
                        Fault::Unknown(format!("a read of data RAM word {x:#x} at {ip:#x}"))
                    })?;
                    if traced {
                        eprintln!("[gpu]     {ip:#05x}: data RAM[{x:#x}] is {value:#x}");
                    }
                    value
                }
                DWRITE => {
                    if host.traced() {
                        eprintln!("[gpu]     {ip:#05x}: data RAM[{x:#x}] = {y:#x}");
                    }
                    let ram = host.data_ram();
                    *ram.get_mut(x as usize).ok_or_else(|| {
                        Fault::Unknown(format!("a write of data RAM word {x:#x} at {ip:#x}"))
                    })? = y;
                    0
                }
                _ => unreachable!("five bits"),
            };
            results[lane] = value;
            match alu.dst {
                0..=23 => regs[usize::from(alu.dst)] = value,
                R_ZERO => {}
                other => {
                    return Err(Fault::Unknown(format!(
                        "destination register {other} at {ip:#x}"
                    )))
                }
            }
        }
        for lane in 0..2 {
            if enabled & (4 << lane) == 0 {
                continue;
            }
            let operand = |what: u8| -> Result<u32, Fault> {
                Ok(match what {
                    O_ALU0 => results[0],
                    O_ALU1 => results[1],
                    O_LOAD0 => load[0],
                    O_LOAD1 => load[1],
                    O_IMM0 => u32::from(i.imm[0]),
                    O_IMM1 => u32::from(i.imm[1]),
                    O_IMM0_HIGH => u32::from(i.imm[0] >> 12),
                    O_IMM1_HIGH => u32::from(i.imm[1] >> 12),
                    O_IMM32 => i.imm32(),
                    other => {
                        return Err(Fault::Unknown(format!("output source {other} at {ip:#x}")))
                    }
                })
            };
            let out = i.out[lane];
            if out.method != O_NONE {
                let value = operand(out.method)?;
                method = value & 0xFFF;
                increment = value >> 12 & 0xF;
            }
            if out.emit != O_NONE {
                let data = operand(out.emit)?;
                host.emit(method, data, &mut params).map_err(Fault::Write)?;
                summary.emitted += 1;
                method = (method + increment) & 0xFFF;
            }
        }
        if ending || stopped {
            return Ok(summary);
        }
        ending = i.end && !keeps_going;
        if loop_count != 0 && ip == loop_end {
            loop_count -= 1;
            next = loop_start + 1;
        }
        ip = next;
    }
}

/// Building test programs, for this module's tests and the engines'.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// The inverse of `Instruction::decode`, for writing test programs.
    pub(super) fn encode(i: &Instruction) -> [u32; 3] {
        let mut bits = 0u128;
        let mut put = |lo: u32, len: u32, v: u32| bits |= (u128::from(v) & ((1 << len) - 1)) << lo;
        put(0, 1, u32::from(i.end));
        put(1, 4, u32::from(i.predicate_mode));
        put(5, 5, u32::from(i.predicate));
        for (n, at) in [10, 46].into_iter().enumerate() {
            put(at, 5, u32::from(i.alu[n].op));
            put(at + 5, 5, u32::from(i.alu[n].dst));
            put(at + 10, 5, u32::from(i.alu[n].a));
            put(at + 15, 5, u32::from(i.alu[n].b));
        }
        put(30, 16, u32::from(i.imm[0]));
        put(66, 16, u32::from(i.imm[1]));
        put(82, 3, u32::from(i.out[0].method));
        put(85, 4, u32::from(i.out[0].emit));
        put(89, 3, u32::from(i.out[1].method));
        put(92, 4, u32::from(i.out[1].emit));
        [(bits >> 64) as u32, (bits >> 32) as u32, bits as u32]
    }

    /// An instruction that does nothing.
    pub(super) fn nop() -> Instruction {
        let idle = Alu {
            op: ADD,
            dst: R_ZERO,
            a: R_ZERO,
            b: R_ZERO,
        };
        Instruction {
            predicate: R_ZERO,
            alu: [idle, idle],
            ..Instruction::default()
        }
    }

    pub(super) fn alu(op: u8, dst: u8, a: u8, b: u8) -> Alu {
        Alu { op, dst, a, b }
    }

    /// A macro that writes its first parameter to `method` and data RAM word `word` to the method
    /// after it, as the words to upload.
    pub(crate) fn store_macro(method: u16, word: u16) -> Vec<u32> {
        let mut first = nop();
        first.alu[0] = alu(ADD, 0, R_LOAD0, R_ZERO);
        first.alu[1] = alu(DREAD, 1, R_IMM, R_ZERO);
        first.imm = [0x1000 | method, word];
        first.out[0] = Out {
            method: O_IMM0,
            emit: O_ALU0,
        };
        first.out[1] = Out {
            method: O_NONE,
            emit: O_ALU1,
        };
        first.end = true;
        [first, nop()].iter().flat_map(encode).collect()
    }
}
#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    /// The program of one macro, entry 1, loaded at instruction 8 through the methods.
    fn program(code: &[Instruction]) -> Program {
        let mut p = Program::default();
        p.load(method::INSTRUCTION_RAM_POINTER, 8);
        for i in code {
            for w in encode(i) {
                p.load(method::INSTRUCTION_RAM, w);
            }
        }
        p.load(method::START_ADDRESS_RAM_POINTER, 1);
        p.load(method::START_ADDRESS_RAM, 8);
        p
    }

    #[derive(Default)]
    struct Recorder {
        writes: Vec<(u32, u32)>,
        ram: Vec<u32>,
        fifo: Vec<u32>,
    }

    impl Host for Recorder {
        fn state(&mut self, method: u32) -> u32 {
            // The last write of the method, else ten times its number.
            self.writes
                .iter()
                .rev()
                .find(|w| w.0 == method)
                .map_or(method * 10, |w| w.1)
        }
        fn emit(
            &mut self,
            method: u32,
            data: u32,
            params: &mut VecDeque<u32>,
        ) -> Result<(), String> {
            if method == method::DMA_READ_FIFOED {
                params.extend(self.fifo.drain(..));
            }
            self.writes.push((method, data));
            Ok(())
        }
        fn data_ram(&mut self) -> &mut [u32] {
            if self.ram.is_empty() {
                self.ram = vec![0; DATA_RAM_WORDS];
            }
            &mut self.ram
        }
    }

    fn call(p: &Program, params: &[u32]) -> (Result<Summary, Fault>, Recorder) {
        let mut host = Recorder::default();
        let r = run(p, p.start(1), params.to_vec(), &mut host);
        (r, host)
    }

    #[test]
    fn an_instruction_decodes_from_its_three_words() {
        // r1 = load0 >> 1 and r2 = load0 & 0 in one instruction (the first word is the highest).
        let i = Instruction::decode([0x0000_0003, 0x5C13_0000, 0x73C0_AB00]);
        assert_eq!(i.alu[0], alu(SRL, 1, R_LOAD0, R_IMM));
        assert_eq!(i.alu[1], alu(AND, 2, R_LOAD0, R_IMM_OTHER));
        assert_eq!(i.imm, [1, 0]);
        assert_eq!(i.loads, [true, false]);
        for i in [i, nop()] {
            let again = Instruction::decode(encode(&i));
            assert_eq!((again.alu, again.imm, again.out), (i.alu, i.imm, i.out));
        }
    }

    #[test]
    fn writes_advance_by_the_increment_and_the_macro_ends_after_its_delay_slot() {
        let mut first = nop();
        first.alu[0] = alu(ADD, 0, R_LOAD0, R_IMM);
        // Method 0x40 with an increment of 1, then two writes.
        first.imm = [5, 0x1040];
        first.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU0,
        };
        first.out[1] = Out {
            method: O_NONE,
            emit: O_LOAD0,
        };
        first.end = true;
        let mut slot = nop();
        slot.imm = [7, 0];
        slot.out[0] = Out {
            method: O_NONE,
            emit: O_IMM0,
        };
        let mut never = slot;
        never.imm = [9, 0];
        let (r, host) = call(&program(&[first, slot, never]), &[100]);
        assert_eq!(r.unwrap().steps, 2);
        assert_eq!(host.writes, [(0x40, 105), (0x41, 100), (0x42, 7)]);
    }

    #[test]
    fn a_loop_runs_its_body_that_many_times_and_zero_times_skips_it() {
        let mut head = nop();
        head.alu[0] = alu(LOOP, R_ZERO, R_LOAD0, R_ZERO);
        head.imm = [2, 0];
        let mut body = nop();
        body.alu[0] = alu(ADD, 3, 3, R_IMM);
        body.imm = [1, 0x20];
        body.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU0,
        };
        let mut tail = nop();
        tail.end = true;
        let p = program(&[head, body, tail, nop()]);
        let (_, host) = call(&p, &[3]);
        assert_eq!(host.writes, [(0x20, 1), (0x20, 2), (0x20, 3)]);
        let (_, host) = call(&p, &[0]);
        assert!(host.writes.is_empty(), "{:?}", host.writes);
    }

    #[test]
    fn both_alus_read_before_either_writes_and_state_reads_the_engine() {
        let mut first = nop();
        first.alu[0] = alu(ADD, 0, R_LOAD0, R_ZERO);
        let mut pair = nop();
        pair.alu[0] = alu(ADD, 0, 0, R_IMM);
        pair.alu[1] = alu(ADD, 1, 0, R_ZERO);
        pair.imm = [1, 0x80];
        pair.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU1,
        };
        let mut state = nop();
        state.alu[0] = alu(STATE, 2, R_IMM, R_ZERO);
        state.imm = [0x80, 0x81];
        state.alu[1] = alu(STATE, 3, R_IMM, R_ZERO);
        state.end = true;
        let mut emit = nop();
        emit.alu[0] = alu(ADD, 4, 2, 3);
        emit.imm = [0, 0x90];
        emit.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU0,
        };
        let (_, host) = call(&program(&[first, pair, state, emit]), &[0x10]);
        // ALU 1 saw r0 before ALU 0 changed it; the state read of 0x80 sees the write before it,
        // that of 0x81 the engine's value.
        assert_eq!(host.writes, [(0x80, 0x10), (0x90, 0x10 + 0x81 * 10)]);
    }

    #[test]
    fn branches_go_where_their_condition_says_and_can_stop_the_macro() {
        let mut branch = nop();
        branch.alu[0] = alu(BEQ, R_ZERO, R_LOAD0, R_ZERO);
        // Taken when equal (bit 15), two instructions on.
        branch.imm = [0x8000 | 2, 0];
        let mut skipped = nop();
        skipped.imm = [0, 0x11];
        skipped.out[0] = Out {
            method: O_IMM1,
            emit: O_IMM1,
        };
        let mut stop = nop();
        stop.alu[0] = alu(BEQ, R_ZERO, R_ZERO, R_ZERO);
        stop.imm = [0x8000 | 0x1000, 0x22];
        stop.out[0] = Out {
            method: O_IMM1,
            emit: O_IMM1,
        };
        let p = program(&[branch, skipped, stop, skipped]);
        assert_eq!(call(&p, &[0]).1.writes, [(0x22, 0x22)]);
        assert_eq!(call(&p, &[1]).1.writes, [(0x11, 0x11), (0x22, 0x22)]);
    }

    #[test]
    fn only_a_back_edge_with_bit_14_keeps_an_end_mark_from_ending() {
        let mut count = nop();
        count.alu[0] = alu(ADD, 5, R_LOAD0, R_ZERO);
        let mut body = nop();
        body.alu[0] = alu(SUB, 5, 5, R_IMM);
        body.imm = [1, 0x30];
        body.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU0,
        };
        // Back to the body while r5 is not zero (not equal: bit 15 clear), with the end mark.
        let mut back = nop();
        back.alu[0] = alu(BEQ, R_ZERO, 5, R_ZERO);
        back.imm = [0x1FFF | 0x4000, 0];
        back.end = true;
        let p = program(&[count, body, back, nop()]);
        assert_eq!(call(&p, &[3]).1.writes, [(0x30, 2), (0x30, 1), (0x30, 0)]);
        // Without bit 14 the end mark ends it after the delay slot, the body, once more.
        back.imm = [0x1FFF, 0];
        let p = program(&[count, body, back, nop()]);
        assert_eq!(call(&p, &[3]).1.writes, [(0x30, 2), (0x30, 1)]);
    }

    #[test]
    fn predicates_turn_parts_on_and_off() {
        let mut gated = nop();
        // TTUU: both ALUs only when the predicate (the parameter) is zero.
        gated.predicate_mode = 3;
        gated.predicate = R_LOAD0;
        gated.alu[0] = alu(ADD, 0, R_IMM, R_ZERO);
        gated.alu[1] = alu(ADD, 1, R_IMM, R_ZERO);
        gated.imm = [3, 4];
        gated.end = true;
        let mut emit = nop();
        emit.alu[0] = alu(ADD, 2, 0, 1);
        emit.imm = [0, 0x60];
        emit.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU0,
        };
        let p = program(&[gated, emit]);
        assert_eq!(call(&p, &[0]).1.writes, [(0x60, 7)]);
        assert_eq!(call(&p, &[1]).1.writes, [(0x60, 0)]);
    }

    #[test]
    fn nvn2s_fill_macro_writes_as_many_words_as_it_is_asked_for() {
        // Its code as the Switch 2 title uploads it (count, first word, value): pairs in a loop,
        // then the odd word under an FFFF predicate on the count's lowest bit.
        let mut code = [nop(); 7];
        code[0].alu = [
            alu(SRL, 1, R_LOAD0, R_IMM),
            alu(AND, 2, R_LOAD0, R_IMM_OTHER),
        ];
        code[0].imm = [1, 0];
        code[1].alu = [alu(ADD, 0, R_LOAD0, R_IMM), alu(ADD, 4, R_LOAD0, R_IMM)];
        code[1].imm = [0, 1];
        code[2].alu = [alu(LOOP, R_ZERO, 1, R_ZERO), alu(ADD, 3, R_LOAD0, R_IMM)];
        code[2].imm = [3, 0];
        code[3].alu = [alu(DWRITE, R_ZERO, 0, 3), alu(ADD, 0, 0, R_IMM)];
        code[3].imm = [0, 2];
        code[4].alu = [alu(DWRITE, R_ZERO, 4, 3), alu(ADD, 4, 4, R_IMM)];
        code[4].imm = [0, 2];
        code[5].alu[0] = alu(DWRITE, R_ZERO, 0, 3);
        code[5].predicate_mode = 2;
        code[5].predicate = 2;
        code[5].end = true;
        let p = program(&code);
        for (count, first) in [(8u32, 0x10usize), (5, 0x40)] {
            let mut host = Recorder::default();
            run(&p, p.start(1), vec![count, first as u32, 0xAA], &mut host).unwrap();
            let written: Vec<usize> = (0..DATA_RAM_WORDS)
                .filter(|&w| host.ram[w] == 0xAA)
                .collect();
            assert_eq!(
                written,
                (first..first + count as usize).collect::<Vec<_>>(),
                "{count}"
            );
        }
    }

    #[test]
    fn the_data_ram_is_read_and_written_and_a_dma_read_feeds_the_parameters() {
        let mut write = nop();
        write.alu[0] = alu(DWRITE, R_ZERO, R_IMM, R_LOAD0);
        write.imm = [0x27, 0];
        // MME_DMA_READ_FIFOED of two words; then read them as parameters.
        let mut dma = nop();
        dma.imm = [2, method::DMA_READ_FIFOED as u16];
        dma.out[0] = Out {
            method: O_IMM1,
            emit: O_IMM0,
        };
        let mut read = nop();
        read.alu[0] = alu(DREAD, 0, R_IMM, R_ZERO);
        read.imm = [0x27, 0];
        read.alu[1] = alu(ADD, 1, R_LOAD0, R_LOAD1);
        read.end = true;
        let mut emit = nop();
        emit.alu[0] = alu(ADD, 2, 0, 1);
        emit.imm = [0, 0x70];
        emit.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU0,
        };
        let p = program(&[write, dma, read, emit]);
        let mut host = Recorder {
            fifo: vec![0x100, 0x200],
            ..Recorder::default()
        };
        run(&p, p.start(1), vec![0xAB], &mut host).unwrap();
        assert_eq!(host.ram[0x27], 0xAB);
        assert_eq!(host.writes.last(), Some(&(0x70, 0xAB + 0x300)));
        // A parameter read past those of the call is 0, and counted.
        let mut short = Recorder::default();
        let summary = run(&p, p.start(1), vec![0xAB], &mut short).unwrap();
        assert_eq!(summary.short_reads, 2);
        assert_eq!(short.writes.last(), Some(&(0x70, 0xAB)));
    }

    #[test]
    fn merge_multiply_and_carry_follow_mesa() {
        let mut first = nop();
        // Bits 4..8 of load1 into bits 8..12 of load0.
        first.alu[0] = alu(MERGE, 0, R_LOAD0, R_LOAD1);
        first.imm = [(8 << 10) | (4 << 5) | 4, (-2i16) as u16];
        first.alu[1] = alu(MUL, 1, R_LOAD0, R_IMM);
        let mut high = nop();
        high.alu[0] = alu(MULH, 2, R_ZERO, R_ZERO);
        high.alu[1] = alu(ADD, 3, R_IMM32, R_IMM32);
        high.imm = [0x8000, 0];
        high.end = true;
        let mut emit = nop();
        emit.imm = [0, 0x1070];
        emit.alu[0] = alu(ADDC, 4, R_ZERO, R_ZERO);
        emit.out[0] = Out {
            method: O_IMM1,
            emit: O_ALU0,
        };
        emit.out[1] = Out {
            method: O_NONE,
            emit: O_ALU1,
        };
        emit.alu[1] = alu(OR, 5, 0, R_ZERO);
        let (_, host) = call(&program(&[first, high, emit]), &[0xFFFF_0000, 0xAB0]);
        // The carry of 0x8000_0000 + 0x8000_0000 is 1; the merge put 0xB at bits 8..12.
        assert_eq!(host.writes, [(0x70, 1), (0x71, 0xFFFF_0B00)]);
    }
}
