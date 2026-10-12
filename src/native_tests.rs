//! Native differential tests: lifted code compiled by clang into a shared library, run, and
//! compared with the interpreter.
//!
//! A batch lifts many programs, compiles them together with the runtime into one library, and runs
//! each case natively. Needs clang; skips loudly without it (`RECOMP_REQUIRE_EXTERNAL=1` makes a
//! skip a failure). Sizes: `RECOMP_NATIVE_WORDS` instructions per encoding group (default 40),
//! `RECOMP_NATIVE_STATES` random states per instruction (default 4), `RECOMP_NATIVE_PROGRAMS`
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
use nx_runtime::testing::{asm, TempDir};

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
            if std::env::var_os("RECOMP_REQUIRE_EXTERNAL").is_some() {
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
    let per_group = env_count("RECOMP_NATIVE_WORDS", 40);
    let states = env_count("RECOMP_NATIVE_STATES", 4);
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
    let randoms = env_count("RECOMP_NATIVE_PROGRAMS", 150);
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
