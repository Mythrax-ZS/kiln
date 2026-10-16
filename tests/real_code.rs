//! Opt-in: lifted native code against the interpreter on real functions of an owner-supplied,
//! already extracted title (`KILN_TITLE_DIR`, a folder holding `exefs/`). Needs clang.
//!
//! The title's `main` is recompiled into a shared library. Sampled functions are then run twice
//! from the same random registers and the same memory, once in the interpreter and once as native
//! code, and everything architecturally visible is compared: how the run ended, the registers
//! and the contents of all writable memory. Nothing from the title is stored or printed except
//! aggregate numbers; the library is built in a temporary directory that is removed afterwards.
//!
//! `KILN_REAL_FUNCTIONS` (default 600) functions, `KILN_REAL_STATES` (default 3) states each,
//! `KILN_REAL_OPT` (default 1) clang optimisation level, `KILN_REAL_STEPS` (default 300000)
//! interpreter instruction budget per run.

use kiln::build::pipeline::{analysis_inputs, build, BuildOptions};
use kiln::build::Toolchain;
use kiln::lift::discover::discover;
use kiln::lift::ModuleCode;
use kiln::program::Program;
use libloading::{Library, Symbol};
use nx_runtime::rt::abi::{perm, stop, NativeCpu, EXIT_PC};
use nx_runtime::rt::host::ModuleDesc;
use nx_runtime::testing::{Rng, TempDir};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn skip_or_fail(why: &str) {
    if std::env::var_os("KILN_REQUIRE_EXTERNAL").is_some() {
        panic!("{why}");
    }
    eprintln!("SKIPPED: {why}");
}

// Above 4 GiB: a title may have gigabytes of `.bss` after its code.
const STACK_TOP: u64 = 0x1_4010_0000;
const STACK_SIZE: u64 = 0x10_0000;
const TLS: u64 = 0x1_5000_0000;
const TLS_SIZE: u64 = 0x4000;
const SCRATCH: u64 = 0x1_5100_0000;
const SCRATCH_SIZE: u64 = 0x1_0000;
/// How much of a writable segment the comparison reads back.
const MAX_COMPARED: u64 = 16 << 20;

struct Lib {
    lib: Library,
    module: *const ModuleDesc,
}

impl Lib {
    fn sym<T>(&self, name: &str) -> Symbol<'_, T> {
        // SAFETY: each call site names the exported function's real signature.
        unsafe { self.lib.get(name.as_bytes()) }.unwrap_or_else(|e| panic!("missing {name}: {e}"))
    }
}

/// A snapshot of every writable range.
fn snapshot(l: &Lib, ranges: &[(u64, u64)]) -> Vec<Vec<u8>> {
    let read: Symbol<unsafe extern "C" fn(u64, *mut u8, u64) -> i32> = l.sym("nx_rt_read");
    ranges
        .iter()
        .map(|&(addr, len)| {
            let mut v = vec![0u8; len as usize];
            // SAFETY: `v` is `len` bytes and the range is mapped.
            assert_eq!(unsafe { read(addr, v.as_mut_ptr(), len) }, 0);
            v
        })
        .collect()
}

fn restore(l: &Lib, ranges: &[(u64, u64)], saved: &[Vec<u8>]) {
    let write: Symbol<unsafe extern "C" fn(u64, *const u8, u64) -> i32> = l.sym("nx_rt_write");
    for (&(addr, _), bytes) in ranges.iter().zip(saved) {
        // SAFETY: as above.
        assert_eq!(
            unsafe { write(addr, bytes.as_ptr(), bytes.len() as u64) },
            0
        );
    }
}

struct Run {
    reason: u32,
    cpu: Box<NativeCpu>,
    memory: Vec<Vec<u8>>,
}

fn describe_difference(a: &Run, b: &Run) -> Option<String> {
    if a.reason != b.reason {
        return Some(format!(
            "ended differently: interpreter {} at {:#x} (kind {}), native {} at {:#x} (kind {})",
            a.reason,
            a.cpu.ctl.stop_pc,
            a.cpu.ctl.trap_kind,
            b.reason,
            b.cpu.ctl.stop_pc,
            b.cpu.ctl.trap_kind
        ));
    }
    let (x, y) = (&a.cpu.ctl, &b.cpu.ctl);
    if a.reason != 0
        && (x.stop_pc != y.stop_pc
            || x.trap_kind != y.trap_kind
            || x.trap_aux != y.trap_aux
            || x.fault_addr != y.fault_addr
            || x.fault_kind != y.fault_kind
            || x.fault_write != y.fault_write)
    {
        return Some(format!(
            "stopped differently: interpreter pc {:#x} kind {} aux {:x?} fault {:#x}/{}/{}, \
             native pc {:#x} kind {} aux {:x?} fault {:#x}/{}/{}",
            x.stop_pc,
            x.trap_kind,
            x.trap_aux,
            x.fault_addr,
            x.fault_kind,
            x.fault_write,
            y.stop_pc,
            y.trap_kind,
            y.trap_aux,
            y.fault_addr,
            y.fault_kind,
            y.fault_write
        ));
    }
    // Generated code does not rebuild the registers of a faulting block.
    let precise = !matches!(a.reason, stop::FAULT | stop::NO_CODE);
    if precise {
        let (s, t) = (&a.cpu.state, &b.cpu.state);
        for i in 0..31 {
            if s.x[i] != t.x[i] {
                return Some(format!(
                    "x{i}: interpreter {:#x}, native {:#x}",
                    s.x[i], t.x[i]
                ));
            }
        }
        if s.sp != t.sp || s.nzcv != t.nzcv || s.pc != t.pc {
            return Some(format!(
                "sp/nzcv/pc: interpreter {:#x}/{:#x}/{:#x}, native {:#x}/{:#x}/{:#x}",
                s.sp, s.nzcv, s.pc, t.sp, t.nzcv, t.pc
            ));
        }
        if s.excl_addr != t.excl_addr || s.excl_size != t.excl_size {
            return Some("exclusive monitor state differs".to_string());
        }
    }
    for (i, (m, n)) in a.memory.iter().zip(&b.memory).enumerate() {
        if m != n {
            let at = m.iter().zip(n).position(|(p, q)| p != q).unwrap();
            return Some(format!(
                "memory range {i} differs at +{at:#x}: interpreter {:#04x}, native {:#04x}",
                m[at], n[at]
            ));
        }
    }
    None
}

#[test]
fn real_functions_run_the_same_natively_and_interpreted() {
    let Some(dir) = std::env::var_os("KILN_TITLE_DIR").map(PathBuf::from) else {
        skip_or_fail("KILN_TITLE_DIR is not set; the real-code comparison did not run");
        return;
    };
    let tc = match Toolchain::discover() {
        Ok(t) => t,
        Err(e) => {
            skip_or_fail(&format!("no clang: {e}"));
            return;
        }
    };
    let program = Program::from_exefs_main(&dir).expect("the title's main loads");
    let work = TempDir::new("real-code");
    let mut opts = BuildOptions::new(work.path(), "title");
    opts.shared = true;
    // The functions are called directly, with no runtime linker to relocate the image.
    opts.pre_relocated = true;
    opts.opt_level = env_count("KILN_REAL_OPT", 1) as u8;
    let report = build(&program, &tc, &opts).expect("the title recompiles");
    eprintln!(
        "real code: {} functions, {} guest instructions, built in {:.1}s",
        report.functions,
        report.guest_instructions,
        report.timings.discover + report.timings.lift_and_compile + report.timings.link
    );

    // Sample function entries.
    let m = program.entry_module();
    let (roots, aopts) = analysis_inputs(&program.modules, &[]).unwrap();
    let d = discover(&ModuleCode(m), &roots, &aopts);
    let entries: Vec<u64> = d.funcs.keys().copied().collect();
    let mut rng = Rng::new(0x5EED_C0DE);

    // Load the library and its module; add the stack, thread-local block and a scratch area.
    // SAFETY: the library was built by this test from the crate's runtime.
    let lib = unsafe { Library::new(report.library.as_ref().unwrap()) }.unwrap();
    let module: Symbol<*const ModuleDesc> = unsafe { lib.get(b"nx_module") }.unwrap();
    let l = Lib {
        module: *module,
        lib,
    };
    {
        let init: Symbol<unsafe extern "C" fn()> = l.sym("nx_module_init");
        let load: Symbol<unsafe extern "C" fn(*const ModuleDesc) -> i32> = l.sym("nx_rt_load");
        let map: Symbol<unsafe extern "C" fn(u64, u64, u32) -> i32> = l.sym("nx_rt_map");
        // SAFETY: valid descriptor and ranges.
        unsafe {
            init();
            assert_eq!(load(l.module), 0);
            let rw = perm::R | perm::W;
            assert_eq!(map(STACK_TOP - STACK_SIZE, STACK_SIZE, rw), 0);
            assert_eq!(map(TLS, TLS_SIZE, rw), 0);
            assert_eq!(map(SCRATCH, SCRATCH_SIZE, rw), 0);
        }
    }
    let mut ranges: Vec<(u64, u64)> = vec![
        (STACK_TOP - STACK_SIZE, STACK_SIZE),
        (TLS, TLS_SIZE),
        (SCRATCH, SCRATCH_SIZE),
    ];
    for s in &m.segments {
        if s.perm.write {
            // A title may have gigabytes of `.bss`: the first megabytes are compared.
            ranges.push((m.base + s.offset, s.size.min(MAX_COMPARED)));
        }
    }
    // Random bytes in the scratch area and thread block, once.
    {
        let write: Symbol<unsafe extern "C" fn(u64, *const u8, u64) -> i32> = l.sym("nx_rt_write");
        // A graph of valid pointers, small numbers and noise, so that chains of dereferences
        // stay inside mapped memory and functions get past their first few loads.
        let graph = |base: u64, size: u64, rng: &mut Rng| -> Vec<u8> {
            let mut v = Vec::with_capacity(size as usize);
            for _ in 0..size / 8 {
                let w = match rng.below(10) {
                    0..=4 => base + 8 * rng.below(size / 8 - 8),
                    5..=7 => rng.below(64),
                    8 => SCRATCH + 8 * rng.below(SCRATCH_SIZE / 8 - 8),
                    _ => rng.next_u64(),
                };
                v.extend_from_slice(&w.to_le_bytes());
            }
            v
        };
        let junk = graph(SCRATCH, SCRATCH_SIZE, &mut rng);
        let tls = graph(TLS, TLS_SIZE, &mut rng);
        // SAFETY: both ranges are mapped.
        unsafe {
            assert_eq!(write(SCRATCH, junk.as_ptr(), junk.len() as u64), 0);
            assert_eq!(write(TLS, tls.as_ptr(), tls.len() as u64), 0);
        }
    }
    let initial = snapshot(&l, &ranges);

    let new: Symbol<unsafe extern "C" fn() -> *mut NativeCpu> = l.sym("nx_rt_cpu_new");
    let free: Symbol<unsafe extern "C" fn(*mut NativeCpu)> = l.sym("nx_rt_cpu_free");
    let run: Symbol<unsafe extern "C" fn(*mut NativeCpu, u64) -> u32> = l.sym("nx_rt_run");
    let interpret: Symbol<unsafe extern "C" fn(*mut NativeCpu, u64, u64) -> u32> =
        l.sym("nx_rt_interpret");
    let set_counter: Symbol<unsafe extern "C" fn(u64)> = l.sym("nx_rt_set_counter");
    let steps = env_count("KILN_REAL_STEPS", 300_000) as u64;

    // `KILN_REAL_ONLY=<hex address>,...`: just these functions, in this order.
    let only: Vec<u64> = std::env::var("KILN_REAL_ONLY")
        .map(|v| {
            v.split(',')
                .filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_default();
    let all = std::env::var_os("KILN_REAL_ALL").is_some() || !only.is_empty();
    let entries = if only.is_empty() { entries } else { only };
    let functions = if all {
        entries.len()
    } else {
        env_count("KILN_REAL_FUNCTIONS", 600)
    };
    let states = env_count("KILN_REAL_STATES", 3);
    let mut tally: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut told: Vec<u64> = Vec::new();
    let data_lo = m.base;
    let data_len = m.image.len() as u64;
    for index in 0..functions {
        let entry = if all {
            entries[index]
        } else {
            entries[rng.below(entries.len() as u64) as usize]
        };
        for _ in 0..states {
            // Arguments: small integers, boundary values, pointers into scratch, thread-local
            // and module memory.
            let mut cpu0 = NativeCpu::new(0).state;
            for reg in cpu0.x.iter_mut().take(29) {
                *reg = match rng.below(16) {
                    0 => rng.below(16),
                    1 => rng.next_u64(),
                    2..=5 => SCRATCH + rng.below(SCRATCH_SIZE - 256),
                    6 | 7 => TLS + rng.below(TLS_SIZE - 256),
                    8 | 9 => data_lo + rng.below(data_len),
                    10 => STACK_TOP - 0x400 - rng.below(0x800),
                    11 => u64::MAX - rng.below(4),
                    12 => 1u64 << rng.below(64),
                    13 => rng.below(256),
                    _ => rng.below(1 << 20),
                };
            }
            cpu0.x[29] = STACK_TOP - 0x100;
            cpu0.sp = STACK_TOP - 0x200 - 16 * rng.below(32);
            cpu0.tpidr_el0 = TLS + 0x100;
            cpu0.tpidrro_el0 = TLS + 0x100;
            cpu0.nzcv = (rng.below(16) as u32) << 28;

            let go = |native: bool| -> Run {
                restore(&l, &ranges, &initial);
                // SAFETY: a fresh CPU used by this thread only; the library functions have the
                // signatures declared above.
                unsafe {
                    let cpu = new();
                    (*cpu).state = cpu0.clone();
                    set_counter(1_000_000);
                    let reason = if native {
                        run(cpu, entry)
                    } else {
                        interpret(cpu, entry, steps)
                    };
                    set_counter(0);
                    let boxed = Box::new(std::ptr::read(cpu));
                    free(cpu);
                    Run {
                        reason,
                        cpu: boxed,
                        memory: snapshot(&l, &ranges),
                    }
                }
            };
            let reference = go(false);
            if reference.reason == stop::HALTED {
                *tally
                    .entry("interpreter budget exhausted (not compared)")
                    .or_default() += 1;
                continue;
            }
            let native = go(true);
            match describe_difference(&reference, &native) {
                None => {
                    let key = match reference.reason {
                        0 => "returned",
                        stop::FAULT => "memory fault",
                        stop::TRAP => "trap or unimplemented instruction",
                        stop::SVC => "supervisor call",
                        stop::NO_CODE => "reached non-code",
                        _ => "other stop",
                    };
                    *tally.entry(key).or_default() += 1;
                }
                Some(why) => {
                    // The first difference of each function, up to forty functions.
                    if failures.len() < 40 && !told.contains(&entry) {
                        told.push(entry);
                        failures.push(format!("function {entry:#x}: {why}"));
                    }
                    *tally.entry("MISMATCH").or_default() += 1;
                }
            }
        }
    }
    let _ = EXIT_PC;
    eprintln!("real code comparison, {functions} functions x {states} states:");
    for (k, v) in &tally {
        eprintln!("  {k}: {v}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        tally.get("returned").copied().unwrap_or(0) > 0,
        "no sampled function ran to completion"
    );
}
