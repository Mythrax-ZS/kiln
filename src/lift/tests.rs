//! Differential tests: lifted IR against the interpreter.
//!
//! Both sides start from the same random state and memory and run the same code; everything
//! architecturally visible must match afterwards. A memory fault is compared by its access (the
//! IR, like the generated code, does not rebuild the registers of the faulting block).

use super::discover::{discover, CodeSource, Options};
use super::eval::{outcome_of_trap, Outcome, ProgramEval};
use super::lift_all;
use nx_runtime::a64::bits::decode_bit_masks;
use nx_runtime::a64::inst::{BranchReg, Inst, LogicOp};
use nx_runtime::a64::{decode, step, CpuState};
use nx_runtime::testing::{asm, FlatBus, Rng};

pub const CODE: u64 = 0x0800_0000;
pub const DATA: u64 = CODE + 0x1_0000;
pub const DATA_SIZE: usize = 0x4000;
pub const REGION: usize = 0x1_0000 + DATA_SIZE;
pub const EXIT: u64 = 0xFFFF_0000;

/// A code image for the analysis: words at `CODE`.
pub struct Image {
    pub words: Vec<u32>,
}

impl CodeSource for Image {
    fn is_code(&self, addr: u64) -> bool {
        addr >= CODE && addr < CODE + 4 * self.words.len() as u64
    }

    fn read(&self, addr: u64, size: usize) -> Option<u64> {
        let off = addr.checked_sub(CODE)? as usize;
        let mut bytes = [0u8; 8];
        for (i, b) in bytes.iter_mut().enumerate().take(size) {
            let at = off + i;
            let w = *self.words.get(at / 4)?;
            *b = w.to_le_bytes()[at % 4];
        }
        Some(u64::from_le_bytes(bytes))
    }
}

pub struct RunResult {
    pub outcome: Outcome,
    pub cpu: CpuState,
    pub mem: Vec<u8>,
}

fn bus_for(words: &[u32], data: &[u8]) -> FlatBus {
    let mut bus = FlatBus::new(CODE, REGION);
    bus.load_words(CODE, words);
    bus.mem[0x1_0000..0x1_0000 + data.len()].copy_from_slice(data);
    bus
}

/// Run on the interpreter with a step budget.
pub fn run_interpreter_bounded(
    words: &[u32],
    cpu0: &CpuState,
    data: &[u8],
    max_steps: usize,
) -> RunResult {
    let mut bus = bus_for(words, data);
    let mut cpu = cpu0.clone();
    cpu.pc = CODE;
    cpu.x[30] = EXIT;
    let mut outcome = Outcome::Halted;
    for _ in 0..max_steps {
        if cpu.pc == EXIT {
            outcome = Outcome::Exited;
            break;
        }
        if let Err(t) = step(&mut cpu, &mut bus) {
            outcome = outcome_of_trap(t, cpu.pc);
            break;
        }
    }
    RunResult {
        outcome,
        cpu,
        mem: bus.mem,
    }
}

/// Discover, lift and evaluate the program at `CODE`.
pub fn run_lifted(words: &[u32], cpu0: &CpuState, data: &[u8]) -> (RunResult, Vec<u64>) {
    let image = Image {
        words: words.to_vec(),
    };
    let d = discover(&image, &[CODE], &Options::default());
    let funcs = lift_all(&d);
    let mut bus = bus_for(words, data);
    let mut cpu = cpu0.clone();
    let mut ev = ProgramEval::new(&funcs, EXIT);
    let outcome = ev.run(CODE, &mut cpu, &mut bus);
    let unknown = ev.unknown.clone();
    (
        RunResult {
            outcome,
            cpu,
            mem: bus.mem,
        },
        unknown,
    )
}

/// `None` when the two runs agree.
pub fn compare(a: &RunResult, b: &RunResult) -> Option<String> {
    if a.outcome != b.outcome {
        return Some(format!(
            "outcome: interpreter {:?}, lifted {:?}",
            a.outcome, b.outcome
        ));
    }
    if a.mem != b.mem {
        let at = a.mem.iter().zip(&b.mem).position(|(x, y)| x != y).unwrap();
        return Some(format!(
            "memory differs at {:#x}: interpreter {:#04x}, lifted {:#04x}",
            CODE + at as u64,
            a.mem[at],
            b.mem[at]
        ));
    }
    if matches!(a.outcome, Outcome::Fault { .. }) {
        return None;
    }
    if a.cpu != b.cpu {
        let (x, y) = (&a.cpu, &b.cpu);
        for i in 0..31 {
            if x.x[i] != y.x[i] {
                return Some(format!(
                    "x{i}: interpreter {:#018x}, lifted {:#018x}",
                    x.x[i], y.x[i]
                ));
            }
        }
        return Some(format!(
            "state: interpreter sp={:#x} pc={:#x} nzcv={:#x} fpcr={:#x} fpsr={:#x} excl={:#x}/{}, \
             lifted sp={:#x} pc={:#x} nzcv={:#x} fpcr={:#x} fpsr={:#x} excl={:#x}/{}",
            x.sp,
            x.pc,
            x.nzcv,
            x.fpcr,
            x.fpsr,
            x.excl_addr,
            x.excl_size,
            y.sp,
            y.pc,
            y.nzcv,
            y.fpcr,
            y.fpsr,
            y.excl_addr,
            y.excl_size
        ));
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Random inputs
// ---------------------------------------------------------------------------------------------

pub struct Gen {
    pub rng: Rng,
}

impl Gen {
    pub fn new(seed: u64) -> Gen {
        Gen {
            rng: Rng::new(seed),
        }
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.rng.below(n)
    }

    /// Register values biased toward the boundaries where arithmetic goes wrong.
    pub fn value(&mut self) -> u64 {
        const SPECIAL: [u64; 16] = [
            0,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFF,
            0x1_0000_0000,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0x8000_0000_8000_0000,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x00FF_00FF_00FF_00FF,
            0xFFFF_0000_FFFF_0000,
        ];
        match self.rng.below(8) {
            0 | 1 => SPECIAL[self.rng.below(SPECIAL.len() as u64) as usize],
            2 => self.rng.below(256),
            3 => self.rng.next_u64() >> self.rng.below(64),
            4 => 1u64 << self.rng.below(64),
            5 => !(1u64 << self.rng.below(64)),
            _ => self.rng.next_u64(),
        }
    }

    /// An address inside the data region with room on both sides.
    pub fn pointer(&mut self) -> u64 {
        DATA + 0x1000 + self.below(DATA_SIZE as u64 - 0x2000)
    }

    pub fn state(&mut self) -> CpuState {
        let mut s = CpuState::default();
        for r in s.x.iter_mut() {
            *r = if self.below(3) == 0 {
                self.pointer()
            } else {
                self.value()
            };
        }
        s.sp = DATA + 0x2000 + 16 * self.below(64);
        s.nzcv = (self.below(16) as u32) << 28;
        s.fpcr = 0;
        s.tpidr_el0 = self.rng.next_u64();
        s.tpidrro_el0 = self.rng.next_u64();
        s
    }

    pub fn data(&mut self) -> Vec<u8> {
        self.rng.bytes(DATA_SIZE)
    }
}

/// Instructions the random-word test runs: everything straight-line the lifter translates.
fn straight_line(inst: &Inst) -> bool {
    !matches!(
        inst,
        Inst::B { .. }
            | Inst::BCond { .. }
            | Inst::Cbz { .. }
            | Inst::Tbz { .. }
            | Inst::BranchReg { .. }
    )
}

fn cases() -> usize {
    std::env::var("RECOMP_LIFT_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60_000)
}

fn check(words: &[u32], g: &mut Gen, what: &str) -> Result<(), String> {
    let cpu = g.state();
    let data = g.data();
    let a = run_interpreter_bounded(words, &cpu, &data, 200);
    let (b, _) = run_lifted(words, &cpu, &data);
    match compare(&a, &b) {
        None => Ok(()),
        Some(why) => Err(format!(
            "{what}: {} words {:08x?}\n  start x={:x?} sp={:#x} nzcv={:#x}\n  {why}",
            words.len(),
            words,
            &cpu.x[..8],
            cpu.sp,
            cpu.nzcv
        )),
    }
}

/// Encoding groups as `(name, mask, value)`: the fixed bits of the group, with every other bit
/// drawn at random. Purely random words rarely reach the narrow groups (`REV`, `ADC`, `CCMP`...).
const TEMPLATES: &[(&str, u32, u32)] = &[
    ("adr/adrp", 0x1F00_0000, 0x1000_0000),
    ("add/sub immediate", 0x1F80_0000, 0x1100_0000),
    ("logical immediate", 0x1F80_0000, 0x1200_0000),
    ("move wide", 0x1F80_0000, 0x1280_0000),
    ("bitfield", 0x1F80_0000, 0x1300_0000),
    ("extract", 0x1F80_0000, 0x1380_0000),
    ("logical shifted", 0x1F00_0000, 0x0A00_0000),
    ("add/sub shifted", 0x1F20_0000, 0x0B00_0000),
    ("add/sub extended", 0x1FE0_0000, 0x0B20_0000),
    ("add/sub carry", 0x1FE0_FC00, 0x1A00_0000),
    ("conditional compare", 0x3FE0_0410, 0x3A40_0000),
    ("conditional select", 0x3FE0_0800, 0x1A80_0000),
    ("data processing 1 source", 0x7FFF_E000, 0x5AC0_0000),
    ("divide", 0x7FE0_F800, 0x1AC0_0800),
    ("variable shift", 0x7FE0_F000, 0x1AC0_2000),
    ("crc32", 0x7FE0_E000, 0x1AC0_4000),
    ("data processing 3 source", 0x1F00_0000, 0x1B00_0000),
    ("load/store unsigned offset", 0x3F00_0000, 0x3900_0000),
    ("load/store immediate", 0x3F20_0000, 0x3800_0000),
    ("load/store register offset", 0x3F20_0C00, 0x3820_0800),
    ("load literal", 0x3F00_0000, 0x1800_0000),
    ("load/store pair", 0x3E00_0000, 0x2800_0000),
    ("exclusive and ordered", 0x3F00_0000, 0x0800_0000),
    ("system register", 0xFFD0_0000, 0xD510_0000),
    ("system instruction", 0xFFD8_0000, 0xD508_0000),
    ("hint", 0xFFFF_F01F, 0xD503_201F),
    ("barrier", 0xFFFF_F01F, 0xD503_301F),
];

fn per_template() -> usize {
    cases() / TEMPLATES.len()
}

/// Every straight-line instruction group, with the operand fields drawn at random.
#[test]
fn every_group_matches_the_interpreter() {
    let mut g = Gen::new(0x11F7);
    let mut failures = Vec::new();
    let mut report = Vec::new();
    let mut low = Vec::new();
    for &(name, mask, value) in TEMPLATES {
        let mut ran = 0usize;
        let mut tries = 0usize;
        while ran < per_template() && tries < per_template() * 40 {
            tries += 1;
            let word = (g.rng.next_u32() & !mask) | value;
            let inst = decode(word);
            if matches!(inst, Inst::Unallocated(_) | Inst::Unsupported { .. })
                || !straight_line(&inst)
                || matches!(inst, Inst::Svc { .. } | Inst::Brk { .. })
            {
                continue;
            }
            ran += 1;
            if let Err(e) = check(&[word, asm::svc(0)], &mut g, name) {
                failures.push(e);
                break;
            }
        }
        report.push(format!("{name}: {ran}"));
        if ran < per_template() / 2 {
            low.push(format!(
                "{name}: only {ran} valid encodings in {tries} draws"
            ));
        }
        if failures.len() >= 8 {
            break;
        }
    }
    eprintln!(
        "lifted vs interpreter, cases per group: {}",
        report.join(", ")
    );
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Plain random words as well, in case a group is missing from the table above.
#[test]
fn random_words_match_the_interpreter() {
    let mut g = Gen::new(0x5EED);
    let n = cases() / 2;
    let mut ran = 0usize;
    let mut failures = Vec::new();
    for _ in 0..n * 8 {
        if ran >= n {
            break;
        }
        let word = g.rng.next_u32();
        let inst = decode(word);
        if matches!(inst, Inst::Unallocated(_) | Inst::Unsupported { .. })
            || !straight_line(&inst)
            || matches!(inst, Inst::Svc { .. } | Inst::Brk { .. })
        {
            continue;
        }
        ran += 1;
        if let Err(e) = check(&[word, asm::svc(0)], &mut g, "random word") {
            failures.push(e);
            if failures.len() >= 8 {
                break;
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(ran >= n / 2, "only {ran} instructions drawn");
}

#[test]
fn encodings_that_trap_stop_the_same_way() {
    let mut g = Gen::new(0x7A);
    let words = [
        asm::svc(0x1234),
        asm::brk(0xF000),
        0x0000_0000,                           // udf #0
        asm::sysreg(true, 3, 3, 13, 0, 2, 3),  // mrs x3, tpidr_el0 (allowed)
        asm::sysreg(true, 3, 0, 0, 0, 0, 3),   // mrs x3, midr_el1 (EL1 only)
        asm::sysreg(false, 3, 3, 14, 0, 0, 3), // msr cntfrq_el0, x3 (read only)
        asm::sysreg(false, 3, 3, 13, 0, 3, 3), // msr tpidrro_el0, x3 (read only)
        asm::sys(false, 3, 7, 5, 0, 3),        // ic iallu is EL1
        asm::sys(true, 0, 1, 2, 3, 3),         // sysl
        0x1E20_1000,                           // an FP instruction: unsupported until M5
        0x4EA0_8400,                           // a NEON instruction
    ];
    for w in words {
        for _ in 0..20 {
            check(&[asm::nop(), w, asm::ret(30)], &mut g, "trapping word").unwrap();
        }
    }
}

#[test]
fn bitfield_and_logical_immediates_cover_the_valid_field_space() {
    let mut g = Gen::new(0xB17);
    for _ in 0..cases() / 4 {
        let sf = g.below(2) == 1;
        let w = if sf { 64 } else { 32 };
        let (n, immr, imms) = loop {
            let n = sf && g.below(2) == 1;
            let immr = g.below(64) as u8;
            let imms = g.below(64) as u8;
            if decode_bit_masks(n, imms, immr, true, w).is_some() {
                break (n, immr, imms);
            }
        };
        let rd = g.below(32) as u8;
        let rn = g.below(32) as u8;
        let logical = asm::logical_imm(sf, g.below(4) as u32, n, immr, imms, rn, rd);
        let bf = asm::bitfield(
            sf,
            [0u32, 1, 2][g.below(3) as usize],
            sf,
            (g.below(u64::from(w)) as u8) & 63,
            (g.below(u64::from(w)) as u8) & 63,
            rn,
            rd,
        );
        for word in [logical, bf] {
            if matches!(decode(word), Inst::Unallocated(_)) {
                continue;
            }
            check(&[word, asm::svc(0)], &mut g, "immediate").unwrap();
        }
    }
}

#[test]
fn only_branches_were_excluded_from_the_random_words() {
    // A sanity check on the filter itself: these are the instructions it skips.
    for inst in [
        decode(asm::jmp(8)),
        decode(asm::b_cond(0, 2)),
        decode(asm::cbz(true, false, 2, 1)),
        decode(asm::tbz(false, 3, 2, 1)),
        decode(asm::ret(30)),
    ] {
        assert!(!straight_line(&inst), "{inst:?}");
    }
    assert!(straight_line(&decode(asm::add_imm(true, 0, 1, 1))));
    let _ = (BranchReg::Br, LogicOp::And);
}
