//! Differential harness: run the same program on Unicorn and on the project's interpreter from the
//! same initial state, then compare every architecturally visible result.

use super::unicorn::{self, reg, Api, Engine};
use nx_runtime::a64::bus::{Bus, Fault, FaultKind};
use nx_runtime::a64::{decode, step, CpuState, Trap};
use std::ffi::c_int;

pub const CODE_BASE: u64 = 0x1000_0000;
pub const CODE_SIZE: usize = 0x10_0000;
pub const DATA_BASE: u64 = 0x2000_0000;
pub const DATA_SIZE: usize = 0x4000;
const SLOT_BYTES: u64 = 32;
const SLOTS: u64 = (CODE_SIZE as u64 - 0x1000) / SLOT_BYTES;

/// Registers transferred to and from Unicorn, in the order of [`state_to_values`].
const REG_IDS: [c_int; 39] = {
    let mut ids = [0; 39];
    let mut i = 0;
    while i < 29 {
        ids[i] = reg::X0 + i as c_int;
        i += 1;
    }
    ids[29] = reg::X29;
    ids[30] = reg::X30;
    ids[31] = reg::SP;
    ids[32] = reg::PC;
    ids[33] = reg::NZCV;
    ids[34] = reg::FPCR;
    ids[35] = reg::FPSR;
    ids[36] = reg::TPIDR_EL0;
    ids[37] = reg::TPIDRRO_EL0;
    ids[38] = reg::CPACR_EL1;
    ids
};

#[derive(Clone)]
pub struct Case {
    pub words: Vec<u32>,
    /// How many instructions to execute (the words after the last one may be data).
    pub steps: usize,
    pub state: CpuState,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Class {
    Ok,
    Unmapped,
    Alignment,
    Exception,
    Unsupported,
    Other(String),
}

#[derive(Clone)]
pub struct Outcome {
    pub class: Class,
    pub state: CpuState,
    pub data: Vec<u8>,
}

/// Multi-region memory mirroring what is mapped in Unicorn.
pub struct TestBus {
    pub code: Vec<u8>,
    pub data: Vec<u8>,
}

impl TestBus {
    fn region(&mut self, addr: u64, size: u8) -> Option<&mut [u8]> {
        let end = addr.checked_add(u64::from(size))?;
        if addr >= CODE_BASE && end <= CODE_BASE + self.code.len() as u64 {
            let o = (addr - CODE_BASE) as usize;
            return Some(&mut self.code[o..o + size as usize]);
        }
        if addr >= DATA_BASE && end <= DATA_BASE + self.data.len() as u64 {
            let o = (addr - DATA_BASE) as usize;
            return Some(&mut self.data[o..o + size as usize]);
        }
        None
    }
}

impl Bus for TestBus {
    fn read(&mut self, addr: u64, size: u8) -> Result<u128, Fault> {
        match self.region(addr, size) {
            Some(b) => {
                let mut v = [0u8; 16];
                v[..b.len()].copy_from_slice(b);
                Ok(u128::from_le_bytes(v))
            }
            None => Err(Fault {
                addr,
                write: false,
                kind: FaultKind::Unmapped,
            }),
        }
    }

    fn write(&mut self, addr: u64, size: u8, value: u128) -> Result<(), Fault> {
        let in_code = addr >= CODE_BASE && addr < CODE_BASE + self.code.len() as u64;
        if in_code {
            return Err(Fault {
                addr,
                write: true,
                kind: FaultKind::Permission,
            });
        }
        match self.region(addr, size) {
            Some(b) => {
                b.copy_from_slice(&value.to_le_bytes()[..size as usize]);
                Ok(())
            }
            None => Err(Fault {
                addr,
                write: true,
                kind: FaultKind::Unmapped,
            }),
        }
    }

    fn counter(&mut self) -> u64 {
        0
    }
}

pub struct Harness<'a> {
    engine: Engine<'a>,
    bus: TestBus,
    next: u64,
}

fn values_of(s: &CpuState) -> Vec<u64> {
    let mut v: Vec<u64> = s.x[..29].to_vec();
    v.push(s.x[29]);
    v.push(s.x[30]);
    v.push(s.sp);
    v.push(s.pc);
    v.push(u64::from(s.nzcv));
    v.push(u64::from(s.fpcr));
    v.push(u64::from(s.fpsr));
    v.push(s.tpidr_el0);
    v.push(s.tpidrro_el0);
    // CPACR_EL1.FPEN = 0b11 so that floating point and SIMD do not trap (needed from M5 on).
    v.push(3 << 20);
    v
}

fn state_of(v: &[u64]) -> CpuState {
    let mut s = CpuState::default();
    s.x[..31].copy_from_slice(&v[..31]);
    s.sp = v[31];
    s.pc = v[32];
    s.nzcv = (v[33] as u32) & 0xF000_0000;
    s.fpcr = v[34] as u32;
    s.fpsr = v[35] as u32;
    s.tpidr_el0 = v[36];
    s.tpidrro_el0 = v[37];
    s
}

impl<'a> Harness<'a> {
    pub fn new(api: &'a Api) -> Harness<'a> {
        let mut engine = api.engine();
        // Guest stores must not rewrite the code under test, in either backend; the harness itself
        // writes instructions with host-side writes, which Unicorn allows on a read-only region.
        engine.map_with(
            CODE_BASE,
            CODE_SIZE,
            unicorn::PROT_READ | unicorn::PROT_EXEC,
        );
        engine.map(DATA_BASE, DATA_SIZE);
        // Unicorn fetches the branch target before it honours the instruction count, so the code
        // region holds NOPs rather than undefined zero words.
        let nop = 0xD503_201Fu32.to_le_bytes();
        let mut code = vec![0u8; CODE_SIZE];
        for word in code.as_chunks_mut::<4>().0 {
            *word = nop;
        }
        engine.write(CODE_BASE, &code);
        Harness {
            engine,
            bus: TestBus {
                code,
                data: vec![0; DATA_SIZE],
            },
            next: 0,
        }
    }

    /// Next code slot; both backends execute the case at this address.
    pub fn slot(&mut self) -> u64 {
        if self.next >= SLOTS {
            self.next = 0;
            self.engine.flush_translation_cache();
        }
        let pc = CODE_BASE + 0x1000 + self.next * SLOT_BYTES;
        self.next += 1;
        pc
    }

    pub fn run_unicorn(&mut self, case: &Case) -> Outcome {
        let mut bytes = Vec::new();
        for w in &case.words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        self.engine.write(case.state.pc, &bytes);
        self.engine
            .invalidate(case.state.pc, case.state.pc + SLOT_BYTES);
        self.engine.write(DATA_BASE, &case.data);
        let mut values = values_of(&case.state);
        self.engine.write_regs(&REG_IDS, &mut values);
        for n in 0..32u8 {
            self.engine.write_q(usize::from(n), case.state.vreg(n));
        }
        let err = self.engine.run(case.state.pc, case.steps);
        let class = match err {
            unicorn::ERR_OK => Class::Ok,
            unicorn::ERR_INSN_INVALID | unicorn::ERR_EXCEPTION => Class::Exception,
            unicorn::ERR_READ_UNMAPPED
            | unicorn::ERR_WRITE_UNMAPPED
            | unicorn::ERR_FETCH_UNMAPPED
            | unicorn::ERR_WRITE_PROT => Class::Unmapped,
            unicorn::ERR_READ_UNALIGNED | unicorn::ERR_WRITE_UNALIGNED => Class::Alignment,
            other => Class::Other(format!("unicorn error {other}")),
        };
        let (state, data) = if class == Class::Ok {
            let mut state = state_of(&self.engine.read_regs(&REG_IDS));
            for n in 0..32u8 {
                state.set_vreg(n, self.engine.read_q(usize::from(n)));
            }
            (state, self.engine.read(DATA_BASE, DATA_SIZE))
        } else {
            (case.state.clone(), case.data.clone())
        };
        Outcome { class, state, data }
    }

    pub fn run_mine(&mut self, case: &Case) -> Outcome {
        let mut cpu = case.state.clone();
        for (i, w) in case.words.iter().enumerate() {
            let o = (cpu.pc - CODE_BASE) as usize + 4 * i;
            self.bus.code[o..o + 4].copy_from_slice(&w.to_le_bytes());
        }
        self.bus.data.copy_from_slice(&case.data);
        let mut class = Class::Ok;
        for _ in 0..case.steps {
            match step(&mut cpu, &mut self.bus) {
                Ok(()) => {}
                Err(Trap::Undefined(_)) => class = Class::Exception,
                Err(Trap::Fault(f)) => {
                    class = if f.kind == FaultKind::Alignment {
                        Class::Alignment
                    } else {
                        Class::Unmapped
                    }
                }
                Err(Trap::Svc(_) | Trap::Brk(_)) => class = Class::Exception,
                // Unicorn raises UNDEFINED for system registers and operations EL0 may not use; the
                // interpreter stops with a report instead of inventing a value. Both are a stop.
                Err(Trap::SysReg { .. } | Trap::SysInst { .. }) => class = Class::Exception,
                Err(Trap::Unsupported { .. }) => class = Class::Unsupported,
            }
            if class != Class::Ok {
                break;
            }
        }
        Outcome {
            class,
            state: cpu,
            data: self.bus.data.clone(),
        }
    }
}

/// Describe how two outcomes differ, or `None` if they agree. `compare_fp` also compares FPCR and
/// FPSR; `compare_thread` compares the thread-pointer registers.
pub fn diff(uc: &Outcome, mine: &Outcome, compare_fp: bool) -> Option<String> {
    if uc.class != mine.class {
        return Some(format!(
            "outcome: unicorn {:?}, interpreter {:?}",
            uc.class, mine.class
        ));
    }
    if uc.class != Class::Ok {
        return None;
    }
    let (a, b) = (&uc.state, &mine.state);
    let mut out = Vec::new();
    for i in 0..31 {
        if a.x[i] != b.x[i] {
            out.push(format!(
                "x{i}: unicorn {:#x}, interpreter {:#x}",
                a.x[i], b.x[i]
            ));
        }
    }
    if a.sp != b.sp {
        out.push(format!("sp: unicorn {:#x}, interpreter {:#x}", a.sp, b.sp));
    }
    if a.pc != b.pc {
        out.push(format!("pc: unicorn {:#x}, interpreter {:#x}", a.pc, b.pc));
    }
    if a.nzcv != b.nzcv {
        out.push(format!(
            "nzcv: unicorn {:#x}, interpreter {:#x}",
            a.nzcv, b.nzcv
        ));
    }
    if a.tpidr_el0 != b.tpidr_el0 {
        out.push(format!(
            "tpidr_el0: unicorn {:#x}, interpreter {:#x}",
            a.tpidr_el0, b.tpidr_el0
        ));
    }
    if compare_fp {
        for n in 0..32u8 {
            if a.vreg(n) != b.vreg(n) {
                out.push(format!(
                    "v{n}: unicorn {:#034x}, interpreter {:#034x}",
                    a.vreg(n),
                    b.vreg(n)
                ));
            }
        }
        if a.fpcr != b.fpcr {
            out.push(format!(
                "fpcr: unicorn {:#x}, interpreter {:#x}",
                a.fpcr, b.fpcr
            ));
        }
        if a.fpsr != b.fpsr {
            out.push(format!(
                "fpsr: unicorn {:#x}, interpreter {:#x}",
                a.fpsr, b.fpsr
            ));
        }
    }
    if uc.data != mine.data {
        let at = uc
            .data
            .iter()
            .zip(&mine.data)
            .position(|(x, y)| x != y)
            .unwrap();
        out.push(format!(
            "memory at {:#x}: unicorn {:#04x}, interpreter {:#04x}",
            DATA_BASE + at as u64,
            uc.data[at],
            mine.data[at]
        ));
    }
    if out.is_empty() {
        None
    } else {
        Some(out.join("; "))
    }
}

pub fn describe(case: &Case) -> String {
    let mut s = String::new();
    for w in &case.words {
        s.push_str(&format!("{w:#010x} {:?}\n", decode(*w)));
    }
    let nonzero: Vec<String> = case
        .state
        .x
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .map(|(i, v)| format!("x{i}={v:#x}"))
        .collect();
    s.push_str(&format!(
        "  pc={:#x} sp={:#x} nzcv={:#x} {}\n",
        case.state.pc,
        case.state.sp,
        case.state.nzcv,
        nonzero.join(" ")
    ));
    s
}

/// Where the first byte of the case's data region is mapped, for generators.
pub fn data_addr(offset: u64) -> u64 {
    DATA_BASE + offset
}
