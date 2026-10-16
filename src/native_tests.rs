//! Native differential tests: lifted code compiled by clang into a shared library, run, and
//! compared with the interpreter.
//!
//! A batch lifts many programs, compiles them together with the runtime into one library, and runs
//! each case natively. Needs clang; skips loudly without it (`KILN_REQUIRE_EXTERNAL=1` makes a
//! skip a failure). Sizes: `KILN_NATIVE_WORDS` instructions per encoding group (default 40),
//! `KILN_NATIVE_STATES` random states per instruction (default 4), `KILN_NATIVE_PROGRAMS`
//! random control-flow programs (default 150).

use crate::build::{runtime_lib, write_units, Toolchain, RUNTIME_EXPORTS};
use crate::lift::discover::{discover, Options};
use crate::lift::eval::Outcome;
use crate::lift::flow_tests::{catalogue, division, random_program};
use crate::lift::ir::{Func, TrapKind};
use crate::lift::lift_all;
use crate::lift::tests::{
    compare, run_interpreter_at, Gen, Image, RunResult, DATA_SIZE, EXIT, REGION, TEMPLATES,
};
use libloading::Library;
use nx_runtime::a64::bus::{Fault, FaultKind};
use nx_runtime::a64::inst::Inst;
use nx_runtime::a64::{decode, CpuState};
use nx_runtime::rt::abi::{perm, stop, trap, NativeCpu};
use nx_runtime::testing::{asm, Rng, TempDir};

/// Spacing between programs in the shared guest address space: wider than the reach of a literal
/// load (1 MiB), so a program's own accesses never land in another program's memory.
const SLOT: u64 = 0x20_0000;
const FIRST: u64 = 0x0800_0000;

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The toolchain, or `None` after saying why the test is skipped.
fn toolchain_or_skip() -> Option<Toolchain> {
    match Toolchain::discover() {
        Ok(t) => Some(t),
        Err(e) => {
            if std::env::var_os("KILN_REQUIRE_EXTERNAL").is_some() {
                panic!("clang is required: {e}");
            }
            eprintln!("SKIPPED (no clang): {e}");
            None
        }
    }
}

pub struct Program {
    pub words: Vec<u32>,
    pub hints: Vec<u64>,
}

impl Program {
    pub fn new(words: Vec<u32>) -> Program {
        Program {
            words,
            hints: Vec::new(),
        }
    }
}

type SvcHandler = extern "C" fn(*mut NativeCpu, u32) -> i32;

extern "C" fn native_test_svc(cpu: *mut NativeCpu, imm: u32) -> i32 {
    // SAFETY: the runtime passes the running thread's CPU.
    let c = unsafe { &mut *cpu };
    if imm == 1 {
        c.state.x[0] = c.state.x[0].wrapping_add(100);
        0
    } else {
        1
    }
}

/// Programs compiled into one library, each at its own address.
pub struct Batch {
    // Field order matters: the library must be unloaded before its directory goes.
    lib: Library,
    _dir: TempDir,
    bases: Vec<u64>,
    /// Each program's code region as it starts (the program, then zeros), restored before every run.
    code: Vec<Vec<u8>>,
    pub functions: usize,
    pub compile_seconds: f32,
}

impl Batch {
    pub fn base(&self, i: usize) -> u64 {
        self.bases[i]
    }

    pub fn build(tc: &Toolchain, programs: &[Program]) -> Result<Batch, String> {
        let start = std::time::Instant::now();
        let dir = TempDir::new("native");
        let mut funcs: Vec<Func> = Vec::new();
        let mut bases = Vec::new();
        for (i, p) in programs.iter().enumerate() {
            let base = FIRST + SLOT * i as u64;
            bases.push(base);
            // The programs assemble at the base they run at, so rebuild them there.
            let image = Image::at(base, p.words.clone());
            let mut roots = vec![base];
            roots.extend(p.hints.iter().map(|h| h - FIRST + base));
            let d = discover(&image, &roots, &Options::default());
            funcs.extend(lift_all(&d).into_values());
        }
        let refs: Vec<&Func> = funcs.iter().collect();
        let jobs = write_units(dir.path(), &refs, 150, "").map_err(|e| e.to_string())?;
        tc.compile_all(&jobs, 1).map_err(|e| e.to_string())?;
        let lib_name = if cfg!(windows) {
            "native.dll"
        } else {
            "libnative.so"
        };
        let out = dir.join(lib_name);
        let objs: Vec<_> = jobs.iter().map(|(_, o)| o.clone()).collect();
        let runtime = runtime_lib().map_err(|e| e.to_string())?;
        tc.link(&objs, &runtime, &out, true, RUNTIME_EXPORTS)
            .map_err(|e| e.to_string())?;
        // SAFETY: loading a library we just built from this crate's own output.
        let lib = unsafe { Library::new(&out) }.map_err(|e| e.to_string())?;
        let code = programs
            .iter()
            .map(|p| {
                let mut c: Vec<u8> = p.words.iter().flat_map(|w| w.to_le_bytes()).collect();
                assert!(c.len() <= 0x1_0000, "program is too large");
                c.resize(0x1_0000, 0);
                c
            })
            .collect();
        let batch = Batch {
            lib,
            _dir: dir,
            bases,
            code,
            functions: funcs.len(),
            compile_seconds: start.elapsed().as_secs_f32(),
        };
        batch.setup(programs)?;
        Ok(batch)
    }

    fn sym<T>(&self, name: &str) -> libloading::Symbol<'_, T> {
        // SAFETY: the type at each call site is the signature of the exported function.
        unsafe { self.lib.get(name.as_bytes()) }.unwrap_or_else(|e| panic!("missing {name}: {e}"))
    }

    fn setup(&self, programs: &[Program]) -> Result<(), String> {
        let init: libloading::Symbol<unsafe extern "C" fn()> = self.sym("nx_module_init");
        let map: libloading::Symbol<unsafe extern "C" fn(u64, u64, u32) -> i32> =
            self.sym("nx_rt_map");
        let write: libloading::Symbol<unsafe extern "C" fn(u64, *const u8, u64) -> i32> =
            self.sym("nx_rt_write");
        let set_svc: libloading::Symbol<unsafe extern "C" fn(Option<SvcHandler>)> =
            self.sym("nx_rt_set_svc_handler");
        // SAFETY: exported functions of the runtime we linked, called with valid arguments.
        unsafe {
            init();
            set_svc(Some(native_test_svc));
            for (i, p) in programs.iter().enumerate() {
                let base = self.bases[i];
                // The same memory the interpreter's flat bus has: code (writable, like the flat bus) and
                // a data region of `DATA_SIZE` bytes, nothing else.
                if map(base, 0x1_0000, perm::R | perm::W | perm::X) != 0
                    || map(base + 0x1_0000, DATA_SIZE as u64, perm::R | perm::W) != 0
                {
                    return Err(format!("cannot map program {i}"));
                }
                let bytes: Vec<u8> = p.words.iter().flat_map(|w| w.to_le_bytes()).collect();
                assert!(bytes.len() <= 0x1_0000, "program {i} is too large");
                if write(base, bytes.as_ptr(), bytes.len() as u64) != 0 {
                    return Err(format!("cannot load program {i}"));
                }
                // Only the program that is running can see its memory: the interpreter has no
                // other, and a stray access must fault the same way on both sides.
                self.set_access(base, 0, 0);
            }
        }
        Ok(())
    }

    /// Run program `i` from `cpu0` with `data` in its data region.
    pub fn run(&self, i: usize, cpu0: &CpuState, data: &[u8]) -> RunResult {
        let base = self.bases[i];
        let new: libloading::Symbol<unsafe extern "C" fn() -> *mut NativeCpu> =
            self.sym("nx_rt_cpu_new");
        let free: libloading::Symbol<unsafe extern "C" fn(*mut NativeCpu)> =
            self.sym("nx_rt_cpu_free");
        let run: libloading::Symbol<unsafe extern "C" fn(*mut NativeCpu, u64) -> u32> =
            self.sym("nx_rt_run");
        let write: libloading::Symbol<unsafe extern "C" fn(u64, *const u8, u64) -> i32> =
            self.sym("nx_rt_write");
        let read: libloading::Symbol<unsafe extern "C" fn(u64, *mut u8, u64) -> i32> =
            self.sym("nx_rt_read");
        // SAFETY: valid pointers and sizes throughout; the CPU is used by this thread only.
        unsafe {
            self.set_access(base, perm::R | perm::W | perm::X, perm::R | perm::W);
            assert_eq!(write(base, self.code[i].as_ptr(), 0x1_0000), 0);
            assert_eq!(write(base + 0x1_0000, data.as_ptr(), data.len() as u64), 0);
            let cpu = new();
            (*cpu).state = cpu0.clone();
            let reason = run(cpu, base);
            let own = base..base + REGION as u64;
            let c = &*cpu;
            let mut state = c.state.clone();
            let ctl = &c.ctl;
            let outcome = match reason {
                stop::NONE => Outcome::Exited,
                stop::SVC => {
                    state.pc = ctl.stop_pc + 4;
                    Outcome::Svc(ctl.trap_aux[0] as u16)
                }
                stop::FAULT => {
                    state.pc = ctl.stop_pc;
                    Outcome::Fault {
                        fault: Fault {
                            addr: ctl.fault_addr,
                            write: ctl.fault_write != 0,
                            kind: match ctl.fault_kind {
                                0 => FaultKind::Unmapped,
                                // Other programs' pages are present but inaccessible here; to the
                                // interpreter they are simply not there.
                                1 if !own.contains(&ctl.fault_addr) => FaultKind::Unmapped,
                                1 => FaultKind::Permission,
                                _ => FaultKind::Alignment,
                            },
                        },
                        pc: ctl.stop_pc,
                    }
                }
                stop::TRAP => {
                    state.pc = ctl.stop_pc;
                    let pc = ctl.stop_pc;
                    let aux = ctl.trap_aux;
                    match ctl.trap_kind {
                        trap::UNDEFINED => Outcome::Trap {
                            kind: TrapKind::Undefined(aux[0] as u32),
                            pc,
                        },
                        trap::UNSUPPORTED => Outcome::Unsupported {
                            word: aux[0] as u32,
                            pc,
                        },
                        trap::BRK => Outcome::Trap {
                            kind: TrapKind::Brk(aux[0] as u16),
                            pc,
                        },
                        trap::SYSREG => {
                            let p = aux[0];
                            Outcome::Trap {
                                kind: TrapKind::SysReg {
                                    write: aux[1] != 0,
                                    enc: [
                                        (p >> 14 & 3) as u8,
                                        (p >> 11 & 7) as u8,
                                        (p >> 7 & 15) as u8,
                                        (p >> 3 & 15) as u8,
                                        (p & 7) as u8,
                                    ],
                                },
                                pc,
                            }
                        }
                        trap::SYSINST => {
                            let p = aux[0];
                            Outcome::Trap {
                                kind: TrapKind::SysInst {
                                    load: aux[1] != 0,
                                    enc: [
                                        (p >> 11 & 7) as u8,
                                        (p >> 7 & 15) as u8,
                                        (p >> 3 & 15) as u8,
                                        (p & 7) as u8,
                                    ],
                                },
                                pc,
                            }
                        }
                        other => panic!("unexpected trap kind {other}"),
                    }
                }
                other => panic!("unexpected stop reason {other}: stop pc {:#x}", ctl.stop_pc),
            };
            let mut mem = vec![0u8; REGION];
            assert_eq!(read(base, mem.as_mut_ptr(), REGION as u64), 0);
            self.set_access(base, 0, 0);
            free(cpu);
            RunResult {
                outcome,
                cpu: state,
                mem,
            }
        }
    }

    fn set_access(&self, base: u64, code: u32, data: u32) {
        let protect: libloading::Symbol<unsafe extern "C" fn(u64, u64, u32) -> i32> =
            self.sym("nx_rt_protect");
        // SAFETY: both ranges were mapped by `setup`.
        unsafe {
            assert_eq!(protect(base, 0x1_0000, code), 0);
            assert_eq!(protect(base + 0x1_0000, DATA_SIZE as u64, data), 0);
        }
    }

    /// Addresses that went through the slow path in this library's runtime.
    pub fn unknown_targets(&self) -> Vec<u64> {
        let count: libloading::Symbol<unsafe extern "C" fn() -> u64> =
            self.sym("nx_rt_unknown_count");
        let at: libloading::Symbol<unsafe extern "C" fn(u64) -> u64> = self.sym("nx_rt_unknown_at");
        // SAFETY: plain queries.
        unsafe { (0..count()).map(|i| at(i)).collect() }
    }
}

fn check_case(
    batch: &Batch,
    i: usize,
    words: &[u32],
    g: &mut Gen,
    states: usize,
    what: &str,
) -> Result<(), String> {
    let base = batch.base(i);
    for _ in 0..states {
        let cpu = g.state_at(base);
        let data = g.data();
        let mut expect = run_interpreter_at(base, words, &cpu, &data, 50_000);
        if matches!(expect.outcome, Outcome::Halted) {
            return Err(format!("{what}: the interpreter did not finish"));
        }
        let got = batch.run(i, &cpu, &data);
        // The native runtime stops with the pc at a sentinel; the interpreter's final pc is
        // compared through the outcome for the stops that carry one.
        if matches!(expect.outcome, Outcome::Exited) {
            expect.cpu.pc = EXIT;
        }
        let mut got = got;
        if matches!(got.outcome, Outcome::Exited) {
            got.cpu.pc = EXIT;
        }
        if let Some(why) = compare(&expect, &got) {
            return Err(format!(
                "{what}: words {words:08x?}
  base {base:#x} start x={:x?} sp={:#x}
  {why}",
                &cpu.x[..],
                cpu.sp
            ));
        }
    }
    Ok(())
}

#[test]
fn a_lifted_function_runs_natively() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    let words = vec![
        asm::add_imm(true, 0, 0, 5),
        asm::add_sub_shifted(true, true, true, 0, 1, 0, 0, 2), // adds x2, x0, x1
        asm::svc(0),
    ];
    let batch = Batch::build(&tc, &[Program::new(words.clone())]).unwrap();
    let mut g = Gen::new(1);
    check_case(&batch, 0, &words, &mut g, 8, "smoke").unwrap();
    eprintln!(
        "native smoke: {} functions in {:.1}s",
        batch.functions, batch.compile_seconds
    );
}

#[test]
fn every_group_matches_the_interpreter_natively() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    let per_group = env_count("KILN_NATIVE_WORDS", 40);
    let states = env_count("KILN_NATIVE_STATES", 4);
    let mut g = Gen::new(0x4A71);
    let mut programs = Vec::new();
    let mut names = Vec::new();
    for &(name, mask, value) in TEMPLATES {
        let mut found = 0;
        let mut tries = 0;
        while found < per_group && tries < per_group * 60 {
            tries += 1;
            let word = (g.rng.next_u32() & !mask) | value;
            let inst = decode(word);
            if matches!(
                inst,
                Inst::Unallocated(_)
                    | Inst::Unsupported { .. }
                    | Inst::Svc { .. }
                    | Inst::Brk { .. }
                    | Inst::B { .. }
                    | Inst::BCond { .. }
                    | Inst::Cbz { .. }
                    | Inst::Tbz { .. }
                    | Inst::BranchReg { .. }
            ) {
                continue;
            }
            found += 1;
            programs.push(Program::new(vec![word, asm::svc(0)]));
            names.push(name);
        }
    }
    let batch = Batch::build(&tc, &programs).unwrap();
    eprintln!(
        "native groups: {} functions in {:.1}s, {states} states each",
        batch.functions, batch.compile_seconds
    );
    let mut failures = Vec::new();
    for (i, p) in programs.iter().enumerate() {
        if let Err(e) = check_case(&batch, i, &p.words, &mut g, states, names[i]) {
            failures.push(e);
            if failures.len() >= 6 {
                break;
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn control_flow_programs_match_the_interpreter_natively() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    let mut programs = Vec::new();
    let mut names = Vec::new();
    for e in catalogue(FIRST) {
        names.push(e.name.clone());
        programs.push(Program {
            words: e.words,
            hints: e.hints,
        });
    }
    // The catalogue is assembled at the first slot; the others are rebuilt at their own base.
    let count = programs.len();
    let mut rebuilt = Vec::new();
    for i in 0..count {
        let base = FIRST + SLOT * i as u64;
        let e = &catalogue(base)[i];
        rebuilt.push(Program {
            words: e.words.clone(),
            hints: e.hints.iter().map(|h| h - base + FIRST).collect(),
        });
    }
    let mut programs = rebuilt;
    let mut g = Gen::new(0xF10);
    let randoms = env_count("KILN_NATIVE_PROGRAMS", 150);
    for n in 0..randoms {
        let base = FIRST + SLOT * programs.len() as u64;
        programs.push(Program::new(random_program(&mut g, base)));
        names.push(format!("random program {n}"));
    }
    for (sf, opcode) in [(false, 2u32), (true, 2), (false, 3), (true, 3)] {
        programs.push(Program::new(division(sf, opcode)));
        names.push(format!("division sf={sf} opcode={opcode}"));
    }
    let batch = Batch::build(&tc, &programs).unwrap();
    eprintln!(
        "native programs: {} functions in {:.1}s",
        batch.functions, batch.compile_seconds
    );
    let mut failures = Vec::new();
    for (i, p) in programs.iter().enumerate() {
        let division_case = names[i].starts_with("division");
        if division_case {
            for (a, b) in [
                (7u64, 0u64),
                (0, 0),
                (u64::MAX, 0),
                (1 << 63, u64::MAX),
                (0x8000_0000, 0xFFFF_FFFF),
                (0x8000_0000, 1),
                (u64::MAX, u64::MAX),
                (100, 7),
            ] {
                let base = batch.base(i);
                let mut cpu = g.state_at(base);
                cpu.x[0] = a;
                cpu.x[1] = b;
                let data = vec![0u8; DATA_SIZE];
                let expect = run_interpreter_at(base, &p.words, &cpu, &data, 100);
                let got = batch.run(i, &cpu, &data);
                if let Some(why) = compare(&expect, &got) {
                    failures.push(format!("{}: {a:#x} / {b:#x}: {why}", names[i]));
                }
            }
            continue;
        }
        if let Err(e) = check_case(&batch, i, &p.words, &mut g, 4, &names[i]) {
            failures.push(e);
            if failures.len() >= 4 {
                break;
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    // The slow path ran for the programs that need it, and only for those.
    let unknown = batch.unknown_targets();
    assert!(!unknown.is_empty(), "no program used the slow path");
}

// ---- floating point and vectors lowered to IR (src/codegen/llvm/vecs.rs) ----

/// A single-precision operand: mostly ordinary numbers of a moderate size (random significands, so
/// that sums and products round, and short ones, so that some are exact), now and then a special
/// one: a zero, an infinity, a NaN, a denormal, a number next to the overflow or underflow limit.
fn single_operand(rng: &mut Rng) -> u32 {
    let sign = (rng.below(2) as u32) << 31;
    let fraction = rng.next_u32() & 0x007F_FFFF;
    match rng.below(24) {
        0 => sign,
        1 => sign | 0x7F80_0000,
        2 => sign | 0x7FC0_0000 | fraction,
        3 => sign | 0x7F80_0000 | (fraction | 1) & 0x003F_FFFF,
        4 => sign | fraction,
        5 => sign | ((0xFD + rng.below(2) as u32) << 23) | fraction,
        6 => sign | ((1 + rng.below(3) as u32) << 23) | fraction,
        7 => sign | (127 << 23),
        8..=12 => sign | ((100 + rng.below(54) as u32) << 23) | (fraction & 0x007F_E000),
        _ => sign | ((96 + rng.below(64) as u32) << 23) | fraction,
    }
}

/// The state for a case: ordinary registers, vector registers whose low words are operands (and
/// whose other bits are random, to show they are cleared), and mostly the default `FPCR`.
fn float_state(g: &mut Gen, base: u64) -> CpuState {
    let mut cpu = g.state_at(base);
    for n in 0..8u8 {
        let low = u128::from(single_operand(&mut g.rng));
        let rest = u128::from(g.rng.next_u64()) << 32 | u128::from(g.rng.next_u32()) << 96;
        cpu.set_vreg(n, low | rest);
    }
    match g.rng.below(10) {
        0..=5 => cpu.fpcr = 0,
        // Flush-to-zero, with default NaN and alternative half precision or without them: the fast
        // forms run under these modes when no operand is a denormal.
        6..=8 => cpu.fpcr = (1 << 24) | ((g.rng.below(4) as u32) << 25),
        // What the state was given: any of the writable bits.
        _ => {}
    }
    if g.rng.chance(2) {
        cpu.fpsr = 0;
    }
    cpu
}

/// Run `words` (one program each) natively and in the interpreter on `states` states made by
/// [`float_state`].
fn check_float_words(words: &[u32], states: usize, seed: u64) {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    for &w in words {
        assert!(
            matches!(decode(w), Inst::Vec { .. } | Inst::VecMem { .. }),
            "{w:08x} is not a floating-point or vector instruction: {:?}",
            decode(w)
        );
    }
    let programs: Vec<Program> = words
        .iter()
        .map(|&w| Program::new(vec![w, asm::svc(0)]))
        .collect();
    let batch = Batch::build(&tc, &programs).unwrap();
    let mut g = Gen::new(seed);
    let mut failures = Vec::new();
    for (i, p) in programs.iter().enumerate() {
        let base = batch.base(i);
        for _ in 0..states {
            let cpu = float_state(&mut g, base);
            let data = g.data();
            let mut expect = run_interpreter_at(base, &p.words, &cpu, &data, 50_000);
            let mut got = batch.run(i, &cpu, &data);
            if matches!(expect.outcome, Outcome::Exited) {
                expect.cpu.pc = EXIT;
            }
            if matches!(got.outcome, Outcome::Exited) {
                got.cpu.pc = EXIT;
            }
            if let Some(why) = compare(&expect, &got) {
                failures.push(format!(
                    "{:08x}: fpcr {:#x} fpsr {:#x} v0..v3 {:x?}\n  {why}",
                    p.words[0],
                    cpu.fpcr,
                    cpu.fpsr,
                    (0..4).map(|n| cpu.vreg(n)).collect::<Vec<_>>()
                ));
                break;
            }
        }
        if failures.len() >= 4 {
            break;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn single_precision_arithmetic_and_compares_match_the_interpreter_natively() {
    let mut words = Vec::new();
    // FMUL, FDIV, FADD, FSUB (single): Sd = Sn op Sm.
    for opcode in [0u32, 1, 2, 3] {
        for (rd, rn, rm) in [(0, 1, 2), (3, 3, 4), (5, 6, 6), (7, 7, 7), (1, 0, 0)] {
            words.push(0x1E20_0800 | opcode << 12 | rm << 16 | rn << 5 | rd);
        }
    }
    // FCMP, FCMPE against a register and against zero.
    for opcode2 in [0x00u32, 0x10, 0x08, 0x18] {
        for (rn, rm) in [(0, 1), (2, 2), (3, 0)] {
            // The forms against zero have no second register.
            let rm = if opcode2 & 8 != 0 { 0 } else { rm };
            words.push(0x1E20_2000 | rm << 16 | rn << 5 | opcode2);
        }
    }
    let states = env_count("KILN_FLOAT_STATES", 600);
    check_float_words(&words, states, 0xF10A7);
}

#[test]
fn register_moves_selects_and_bit_operations_match_the_interpreter_natively() {
    let mut words = Vec::new();
    // FMOV, FABS, FNEG (single and double), FMOV immediate, FMOV Sd,Wn / Wd,Sn / Dd,Xn / Xd,Dn /
    // Vd.D[1],Xn / Xd,Vn.D[1].
    for ftype in [0u32, 1] {
        for opcode in [0u32, 1, 2] {
            for (rd, rn) in [(0, 1), (2, 2)] {
                words.push(0x1E20_4000 | ftype << 22 | opcode << 15 | rn << 5 | rd);
            }
        }
        for imm8 in [0u32, 0x70, 0xFF, 0x3C] {
            words.push(0x1E20_1000 | ftype << 22 | imm8 << 13 | 3);
        }
        // FCSEL for every condition.
        for cond in 0u32..16 {
            words.push(0x1E20_0C00 | ftype << 22 | 2 << 16 | cond << 12 | 1 << 5 | 4);
        }
    }
    for word in [
        0x1E27_0020u32, // fmov s0, w1
        0x1E26_0040,    // fmov w0, s2
        0x9E67_0041,    // fmov d1, x2
        0x9E66_0041,    // fmov x1, d2
        0x9EAF_0061,    // fmov v1.d[1], x3
        0x9EAE_0062,    // fmov x2, v3.d[1]
    ] {
        words.push(word);
    }
    // AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF on 16 and on 8 bytes.
    for q in [1u32, 0] {
        for (size_opc, base) in [
            (0u32, 0x0E20_1C00u32), // and
            (1, 0x0E20_1C00),       // bic
            (2, 0x0E20_1C00),       // orr
            (3, 0x0E20_1C00),       // orn
        ] {
            words.push(base | q << 30 | size_opc << 22 | 2 << 16 | 1 << 5);
        }
        for (u, size) in [(1u32, 0u32), (1, 1), (1, 2), (1, 3)] {
            words.push(0x0E20_1C00 | q << 30 | u << 29 | size << 22 | 2 << 16 | 1 << 5 | 3);
        }
        words.push(0x0EA2_1C21 | q << 30); // orr v1, v1, v2 (mov-like)
        words.push(0x0EA1_1C20 | q << 30); // orr v0, v1, v1 (mov v0, v1)
    }
    let states = env_count("KILN_FLOAT_STATES", 600) / 4;
    check_float_words(&words, states.max(20), 0xB175);
}

#[test]
fn vector_loads_and_stores_of_one_or_two_registers_match_the_interpreter_natively() {
    let mut words = Vec::new();
    // LDR/STR of B, H, S, D, Q, unsigned offset, from x1 (which points into the test's memory).
    for (size, opc) in [(0u32, 0u32), (1, 0), (2, 0), (3, 0), (0, 2)] {
        for load in [0u32, 1] {
            words.push(0x3D00_0000 | size << 30 | (opc | load) << 22 | 8 << 10 | 1 << 5 | 2);
        }
    }
    // LDP/STP of S, D, Q: offset, pre-index and post-index.
    for opc in [0u32, 1, 2] {
        for load in [0u32, 1] {
            for mode in [0x2D00_0000u32, 0x2D80_0000, 0x2C80_0000] {
                words.push(mode | opc << 30 | load << 22 | 1 << 15 | 2 << 10 | 1 << 5 | 3);
            }
        }
    }
    let states = env_count("KILN_FLOAT_STATES", 600) / 4;
    check_float_words(&words, states.max(20), 0x7E5);
}

/// Like [`float_state`], but every lane of the first eight vector registers is an operand, and now
/// and then a small integer (for the conversions from integers and the by-element forms).
fn lane_state(g: &mut Gen, base: u64) -> CpuState {
    let mut cpu = float_state(g, base);
    for n in 0..8u8 {
        let mut v = 0u128;
        for lane in 0..4 {
            let word = if g.rng.chance(4) {
                (g.rng.below(1 << 14) as i32 - (1 << 13)) as u32
            } else {
                single_operand(&mut g.rng)
            };
            v |= u128::from(word) << (32 * lane);
        }
        cpu.set_vreg(n, v);
    }
    cpu
}

/// [`check_float_words`] with the states of [`lane_state`].
fn check_lane_words(words: &[u32], states: usize, seed: u64) {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    for &w in words {
        assert!(
            matches!(decode(w), Inst::Vec { .. } | Inst::VecMem { .. }),
            "{w:08x} is not a floating-point or vector instruction: {:?}",
            decode(w)
        );
    }
    let programs: Vec<Program> = words
        .iter()
        .map(|&w| Program::new(vec![w, asm::svc(0)]))
        .collect();
    let batch = Batch::build(&tc, &programs).unwrap();
    let mut g = Gen::new(seed);
    let mut failures = Vec::new();
    for (i, p) in programs.iter().enumerate() {
        let base = batch.base(i);
        for _ in 0..states {
            let cpu = lane_state(&mut g, base);
            let data = g.data();
            let mut expect = run_interpreter_at(base, &p.words, &cpu, &data, 50_000);
            let mut got = batch.run(i, &cpu, &data);
            if matches!(expect.outcome, Outcome::Exited) {
                expect.cpu.pc = EXIT;
            }
            if matches!(got.outcome, Outcome::Exited) {
                got.cpu.pc = EXIT;
            }
            if let Some(why) = compare(&expect, &got) {
                failures.push(format!(
                    "{:08x}: fpcr {:#x} fpsr {:#x} v0..v3 {:x?} x0..x3 {:x?}\n  {why}",
                    p.words[0],
                    cpu.fpcr,
                    cpu.fpsr,
                    (0..4).map(|n| cpu.vreg(n)).collect::<Vec<_>>(),
                    &cpu.x[..4]
                ));
                break;
            }
        }
        if failures.len() >= 4 {
            break;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn vector_single_precision_arithmetic_matches_the_interpreter_natively() {
    let mut words = Vec::new();
    for q in [1u32, 0] {
        // FADD, FSUB, FMUL, FMLA, FMLS (vector).
        for base in [
            0x0E20_D400u32,
            0x0EA0_D400,
            0x2E20_DC00,
            0x0E20_CC00,
            0x0EA0_CC00,
        ] {
            for (rd, rn, rm) in [(0, 1, 2), (3, 3, 4), (5, 6, 6), (7, 7, 7), (1, 0, 0)] {
                words.push(base | q << 30 | rm << 16 | rn << 5 | rd);
            }
        }
        // FMUL, FMLA, FMLS by element, every index.
        for opcode in [0x9u32, 0x1, 0x5] {
            for idx in 0u32..4 {
                let (h, l) = (idx >> 1, idx & 1);
                words.push(
                    0x0F80_0000 | q << 30 | l << 21 | 2 << 16 | opcode << 12 | h << 11 | 1 << 5 | 3,
                );
            }
        }
        // SCVTF, UCVTF (vector).
        for u in [0u32, 1] {
            words.push(0x0E21_D800 | q << 30 | u << 29 | 1 << 5 | 2);
        }
    }
    // SCVTF, UCVTF (scalar, vector register and general register), FCVTZS, FCVTZU (to W).
    for u in [0u32, 1] {
        words.push(0x5E21_D800 | u << 29 | 1 << 5 | 2);
        words.push(0x1E22_0000 | u << 16 | 1 << 5 | 2);
        words.push(0x1E38_0000 | u << 16 | 1 << 5 | 2);
    }
    let states = env_count("KILN_FLOAT_STATES", 600);
    check_lane_words(&words, states, 0x5EC7);
}

#[test]
fn lane_moves_and_permutes_match_the_interpreter_natively() {
    let mut words = Vec::new();
    for q in [1u32, 0] {
        // DUP (element) of 8, 16, 32 and 64 bit lanes, an index each.
        for (imm5, ok) in [
            (0b00111u32, q == 1 || true),
            (0b01010, true),
            (0b10100, true),
            (0b11000, q == 1),
        ] {
            if ok {
                words.push(0x0E00_0400 | q << 30 | imm5 << 16 | 1 << 5 | 2);
            }
        }
        for size in 0u32..4 {
            if size == 3 && q == 0 {
                continue;
            }
            // ZIP1, UZP1, TRN1, ZIP2, UZP2, TRN2.
            for opc in [3u32, 1, 2, 7, 5, 6] {
                words.push(0x0E00_0800 | q << 30 | size << 22 | 2 << 16 | opc << 12 | 1 << 5 | 3);
            }
        }
        // EXT.
        for imm4 in [0u32, 3, 7] {
            words.push(0x2E00_0000 | q << 30 | 2 << 16 | imm4 << 11 | 1 << 5 | 3);
        }
        if q == 1 {
            words.push(0x2E00_0000 | q << 30 | 2 << 16 | 15 << 11 | 1 << 5 | 3);
        }
    }
    // INS (element) of every lane size, to the same register and to another.
    for (imm5, imm4) in [
        (0b00001u32 | 3 << 1 << 1, 5u32),
        (0b00010 | 2 << 2, 6 << 1),
        (0b00100 | 1 << 3, 2 << 2),
        (0b01000 | 1 << 4, 0),
    ] {
        for (rd, rn) in [(1u32, 2u32), (3, 3)] {
            words.push(0x6E00_0400 | imm5 << 16 | imm4 << 11 | rn << 5 | rd);
        }
    }
    let states = env_count("KILN_FLOAT_STATES", 600) / 2;
    check_lane_words(&words, states.max(20), 0xA27E);
}
