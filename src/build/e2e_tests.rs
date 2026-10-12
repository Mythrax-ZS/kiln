//! End to end: a synthetic NSO is recompiled into a native executable and shared library, and both
//! are run. Needs clang; skips loudly without it.

use super::pipeline::{build, BuildOptions};
use super::Toolchain;
use crate::lift::flow_tests::{fib_program, jump_table_words};
use crate::loader::module::GUEST_BASE;
use crate::loader::nso;
use crate::loader::testing::{build_nso, build_parts, NsoFlags, Spec, CODE_OFF, SLOTS_OFF};
use crate::program::Program;
use libloading::Library;
use nx_runtime::rt::host::ModuleDesc;
use nx_runtime::testing::{asm, TempDir};
use std::path::Path;
use std::process::Command;

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

fn program_from(spec: &Spec) -> Program {
    let bytes = build_nso(&build_parts(spec), NsoFlags::default());
    let loaded = nso::load(&bytes, "main", GUEST_BASE, &nso::LoadOptions::default()).unwrap();
    Program {
        name: "main".into(),
        modules: vec![loaded.module],
        npdm: None,
    }
}

/// `main`: x0 = 10 then fib(10).
fn fib_main() -> Vec<u32> {
    let mut words = vec![asm::movz(true, 0, 10, 0)];
    words.extend(fib_program(CODE_OFF as u64 + 4));
    words
}

fn run(exe: &Path) -> std::process::Output {
    Command::new(exe)
        .output()
        .expect("run the produced program")
}

#[test]
fn a_synthetic_nso_becomes_a_working_executable() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    let program = program_from(&Spec {
        code: fib_main(),
        ..Spec::default()
    });
    let dir = TempDir::new("e2e");
    let opts = BuildOptions::new(dir.path(), "fibmain");
    let report = build(&program, &tc, &opts).unwrap();
    assert!(report.functions >= 2, "{report:?}");
    assert!(report.exe.is_file());
    let out = run(&report.exe);
    assert_eq!(
        out.status.code(),
        Some(55),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&report.report_file).unwrap();
    assert!(text.contains("functions"), "{text}");
    assert!(
        text.contains("interpreter fallbacks: 0 instructions"),
        "{text}"
    );
}

#[test]
fn exports_unwind_ranges_and_jump_tables_feed_the_analysis() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    // `nnMain` returns the table case for index 3 (103); the table is resolved by the analysis.
    let mut words = vec![asm::movz(true, 0, 3, 0)];
    words.extend(jump_table_words(CODE_OFF as u64 + 4, 8));
    let n = words.len() as u32;
    let program = program_from(&Spec {
        code: words,
        with_mod0: true,
        exports: vec![("nnMain", CODE_OFF as u64)],
        eh_funcs: vec![(CODE_OFF as u32, 4 * n)],
        ..Spec::default()
    });
    let dir = TempDir::new("e2e-table");
    let mut opts = BuildOptions::new(dir.path(), "tablemain");
    opts.shared = true;
    let report = build(&program, &tc, &opts).unwrap();
    assert_eq!(
        report.analysis.jump_tables.len(),
        1,
        "{:?}",
        report.analysis
    );
    assert!(report.analysis.unresolved_branches.is_empty());
    assert_eq!(run(&report.exe).status.code(), Some(103));

    // The library exposes the same code: load it, look the export up, call it.
    let lib_path = report.library.expect("a shared library was requested");
    // SAFETY: loading the library this test just built; the signatures below are the ones the
    // runtime exports.
    unsafe {
        let lib = Library::new(&lib_path).unwrap();
        let init: libloading::Symbol<unsafe extern "C" fn()> = lib.get(b"nx_module_init").unwrap();
        let module: libloading::Symbol<*const ModuleDesc> = lib.get(b"nx_module").unwrap();
        let load: libloading::Symbol<unsafe extern "C" fn(*const ModuleDesc) -> i32> =
            lib.get(b"nx_rt_load").unwrap();
        let find: libloading::Symbol<
            unsafe extern "C" fn(*const ModuleDesc, *const std::ffi::c_char) -> u64,
        > = lib.get(b"nx_rt_find_export").unwrap();
        let call: libloading::Symbol<unsafe extern "C" fn(u64, *const u64, u32) -> u64> =
            lib.get(b"nx_rt_call").unwrap();
        init();
        assert_eq!(load(*module), 0);
        let addr = find(*module, c"nnMain".as_ptr());
        assert_eq!(addr, GUEST_BASE + CODE_OFF as u64);
        assert_eq!(find(*module, c"missing".as_ptr()), 0);
        assert_eq!(call(addr, std::ptr::null(), 0), 103);
    }
}

#[test]
fn a_program_that_stops_says_why_and_exits_nonzero() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    // An unimplemented SVC, then a return that must never be reached.
    let program = program_from(&Spec {
        code: vec![asm::movz(true, 0, 1, 0), asm::svc(0x2a), asm::ret(30)],
        ..Spec::default()
    });
    let dir = TempDir::new("e2e-stop");
    let report = build(&program, &tc, &BuildOptions::new(dir.path(), "stopper")).unwrap();
    let out = run(&report.exe);
    assert_eq!(out.status.code(), Some(101));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("SVC 0x2a is not implemented (SVC at 0x8000184)"),
        "{err}"
    );

    // A wild memory access is reported with its address and the instruction.
    let program = program_from(&Spec {
        code: vec![
            asm::movz(true, 1, 0xDEAD, 0),
            asm::ldst_unsigned(3, 1, 0, 1, 0), // ldr x0, [x1]
            asm::ret(30),
        ],
        ..Spec::default()
    });
    let report = build(&program, &tc, &BuildOptions::new(dir.path(), "wild")).unwrap();
    let err = String::from_utf8_lossy(&run(&report.exe).stderr).into_owned();
    assert!(
        err.contains("read of 0xdead (unmapped) by the instruction at 0x8000184"),
        "{err}"
    );
}

#[test]
fn the_objects_of_a_build_can_be_linked_again() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    let program = program_from(&Spec {
        code: fib_main(),
        ..Spec::default()
    });
    let dir = TempDir::new("e2e-relink");
    let report = build(&program, &tc, &BuildOptions::new(dir.path(), "again")).unwrap();
    std::fs::remove_file(&report.exe).unwrap();
    let exe = super::pipeline::relink(&tc, dir.path(), "again").unwrap();
    assert_eq!(run(&exe).status.code(), Some(55));
    assert!(super::pipeline::relink(&tc, &dir.join("nothing"), "again").is_err());
}

/// Two NSOs mapped back to back, the way the kernel's loader maps `rtld` and `main`. The first
/// calls a function in the second (across the module boundary), then looks at a data slot that a
/// RELATIVE relocation covers and returns 42 if the slot is still zero (nobody relocated the
/// image) or 41 if it holds a pointer.
fn two_module_program() -> Program {
    let mov = |rd: u8, rm: u8| asm::logical_shifted(true, 1, 0, false, rm, 0, 31, rd);
    let a = Spec {
        code: vec![
            mov(19, 30),
            asm::b_imm(true, 0x5000 / 4 - 1), // main's function, one module further (the call is the second word)
            asm::movz(true, 2, SLOTS_OFF as u32, 0),
            asm::movk(true, 2, 0x0800, 1),
            asm::ldr_x(1, 2, 0),
            asm::cbz(true, false, 4, 1), // zero: skip to the "unrelocated" case
            asm::add_imm(true, 0, 0, 1),
            mov(30, 19),
            asm::ret(30),
            asm::add_imm(true, 0, 0, 2),
            mov(30, 19),
            asm::ret(30),
        ],
        with_mod0: true,
        relatives: vec![(SLOTS_OFF as u64, CODE_OFF as i64)],
        ..Spec::default()
    };
    let b = Spec {
        code: vec![asm::movz(true, 0, 40, 0), asm::ret(30)],
        ..Spec::default()
    };
    let load = |spec: &Spec, name: &str, base: u64| {
        let bytes = build_nso(&build_parts(spec), NsoFlags::default());
        nso::load(&bytes, name, base, &nso::LoadOptions::default())
            .unwrap()
            .module
    };
    let first = load(&a, "rtld", GUEST_BASE);
    assert_eq!(first.image.len(), 0x5000, "the test's call distance");
    let second = load(&b, "main", GUEST_BASE + first.image.len() as u64);
    Program {
        name: "main".into(),
        modules: vec![first, second],
        npdm: None,
    }
}

#[test]
fn modules_are_mapped_back_to_back_and_start_as_the_file_left_them() {
    let Some(tc) = toolchain_or_skip() else {
        return;
    };
    let program = two_module_program();
    let dir = TempDir::new("e2e-modules");
    let report = build(&program, &tc, &BuildOptions::new(dir.path(), "two")).unwrap();
    // Each module's start (`b CODE_OFF`) and its function.
    assert_eq!(report.functions, 4, "{report:?}");
    assert_eq!(
        std::fs::metadata(&report.image).unwrap().len(),
        0xA000,
        "both modules are in the image"
    );
    let out = run(&report.exe);
    assert_eq!(
        out.status.code(),
        Some(42),
        "an unrelocated image leaves the slot zero; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Asking for the relocations this tool applies puts a pointer in the slot instead.
    let mut opts = BuildOptions::new(dir.path(), "two_relocated");
    opts.pre_relocated = true;
    let report = build(&program, &tc, &opts).unwrap();
    assert_eq!(run(&report.exe).status.code(), Some(41));
}

// ---------------------------------------------------------------------------------------------
// Homebrew on the kernel layer
// ---------------------------------------------------------------------------------------------

mod homebrew {
    use super::*;
    use crate::loader::nro;
    use crate::loader::testing::build_nro;
    use nx_runtime::testing::prog::Prog;

    const X0: u8 = 0;
    const X1: u8 = 1;
    const X2: u8 = 2;
    const X3: u8 = 3;
    const X4: u8 = 4;
    const X5: u8 = 5;
    const SP: u8 = 31;

    fn mov_reg(rd: u8, rm: u8) -> u32 {
        asm::logical_shifted(true, 1, 0, false, rm, 0, 31, rd)
    }

    fn mov_imm(rd: u8, v: u32) -> u32 {
        asm::movz(true, rd, v, 0)
    }

    fn str_w(rt: u8, rn: u8, byte_off: u32) -> u32 {
        asm::ldst_unsigned(2, 0, byte_off / 4, rn, rt)
    }

    fn ldr_w(rt: u8, rn: u8, byte_off: u32) -> u32 {
        asm::ldst_unsigned(2, 1, byte_off / 4, rn, rt)
    }

    /// Emit a string as data words; the label names its start.
    fn data_string(p: &mut Prog, label: &str, text: &str) {
        p.label(label);
        let mut bytes = text.as_bytes().to_vec();
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        for c in bytes.chunks(4) {
            p.emit(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
    }

    /// `svcOutputDebugString(label, len)`.
    fn print(p: &mut Prog, label: &str, text: &str) {
        p.adr(X0, label);
        p.emit(mov_imm(X1, text.len() as u32));
        p.emit(asm::svc(0x27));
    }

    /// An IPC request to `sm:` in the thread-local buffer, sent on the session in `x25`.
    fn sm_request(p: &mut Prog, command: u32, pid: bool, name: &[u8; 8]) {
        // x26 = the thread-local region.
        p.emit(asm::sysreg(true, 3, 3, 13, 0, 3, 26));
        let words: Vec<(u32, u32)> = if pid {
            vec![
                (0, 4),
                (4, 9 | 1 << 31),
                (8, 1),
                (12, 0),
                (16, 0),
                (32, 0x4943_4653),
                (36, 0),
                (40, command),
                (44, 0),
                (48, 0),
                (52, 0),
            ]
        } else {
            let n = u64::from_le_bytes(*name);
            vec![
                (0, 4),
                (4, 8),
                (16, 0x4943_4653),
                (20, 0),
                (24, command),
                (28, 0),
                (32, n as u32),
                (36, (n >> 32) as u32),
            ]
        };
        for (off, v) in words {
            p.emit(asm::movz(false, 9, v & 0xFFFF, 0));
            if v >> 16 != 0 {
                p.emit(asm::movk(false, 9, v >> 16, 1));
            }
            p.emit(str_w(9, 26, off));
        }
        p.emit(mov_reg(X0, 25));
        p.emit(asm::svc(0x21));
    }

    /// Connect to `sm:` (session in x25).
    fn connect_sm(p: &mut Prog) {
        p.adr(X1, "sm_name");
        p.emit(asm::svc(0x1F));
        p.emit(mov_reg(25, X1));
    }

    /// The NRO's guest code and the address of its worker thread. It prints, grows the heap,
    /// starts a thread that increments a shared counter 1000 times with exclusive loads and stores
    /// while it does the same, waits on an event the thread signals, checks the counter, and talks
    /// to `sm:`.
    fn hello_nro() -> (Vec<u32>, u64) {
        let mut p = Prog::new(GUEST_BASE + CODE_OFF as u64);
        p.emit(mov_reg(19, X0)); // the loader configuration
        print(&mut p, "hello", "hello from a homebrew NRO\n");
        // Heap: 2 MiB.
        p.emit(asm::movz(true, X1, 0x20, 1));
        p.emit(asm::svc(0x01));
        p.cbnz(false, X0, "bad");
        p.emit(mov_reg(21, X1));
        // An event: x22 writable, x23 readable; the worker finds the writable handle in the heap.
        p.emit(asm::svc(0x45));
        p.emit(mov_reg(22, X1));
        p.emit(mov_reg(23, X2));
        p.emit(asm::str_x(22, 21, 8));
        // The worker thread: entry, argument (the heap), stack top, priority, default core.
        p.adr(X1, "worker");
        p.emit(mov_reg(X2, 21));
        p.emit(asm::movz(true, X3, 0x1F, 1));
        p.emit(asm::add_sub_shifted(true, false, false, 0, 21, 0, X3, X3));
        p.emit(mov_imm(X4, 0x2C));
        p.emit(asm::movz(false, X5, 0xFFFE, 0));
        p.emit(asm::movk(false, X5, 0xFFFF, 1));
        p.emit(mov_imm(X0, 0));
        p.emit(asm::svc(0x08));
        p.cbnz(false, X0, "bad");
        p.emit(mov_reg(24, X1));
        p.emit(mov_reg(X0, 24));
        p.emit(asm::svc(0x09));
        p.cbnz(false, X0, "bad");
        // 1000 atomic increments of [heap], racing the worker's.
        p.emit(mov_imm(9, 1000));
        p.label("main_loop");
        p.emit(asm::exclusive(3, false, true, false, 31, false, 31, 21, 10));
        p.emit(asm::add_imm(true, 10, 10, 1));
        p.emit(asm::exclusive(
            3, false, false, false, 11, false, 31, 21, 10,
        ));
        p.cbnz(false, 11, "main_loop");
        p.emit(asm::sub_imm(true, 9, 9, 1));
        p.cbnz(true, 9, "main_loop");
        // Wait for the worker's signal: WaitSynchronization(&[readable], 1, forever).
        p.emit(asm::sub_imm(true, SP, SP, 16));
        p.emit(asm::str_x(23, SP, 0));
        p.emit(asm::add_imm(true, X1, SP, 0));
        p.emit(mov_imm(X2, 1));
        p.emit(asm::movz(true, X3, 0xFFFF, 0));
        p.emit(asm::movk(true, X3, 0xFFFF, 1));
        p.emit(asm::movk(true, X3, 0xFFFF, 2));
        p.emit(asm::movk(true, X3, 0xFFFF, 3));
        p.emit(asm::svc(0x18));
        p.emit(asm::add_imm(true, SP, SP, 16));
        p.cbnz(false, X0, "bad");
        p.emit(asm::ldr_x(9, 21, 0));
        p.emit(asm::cmp_imm(true, 9, 2000));
        p.b_cond(1, "bad");
        print(&mut p, "counter_ok", "counter ok\n");
        // The service manager: connect, RegisterClient, check the reply's result.
        connect_sm(&mut p);
        sm_request(&mut p, 0, true, &[0; 8]);
        p.cbnz(false, X0, "bad");
        p.emit(ldr_w(9, 26, 24));
        p.cbnz(false, 9, "bad");
        print(&mut p, "sm_ok", "sm ok\n");
        print(&mut p, "done", "done\n");
        p.emit(mov_imm(X0, 0));
        p.emit(asm::ret(30));
        p.label("bad");
        print(&mut p, "failed", "FAILED\n");
        p.emit(mov_imm(X0, 1));
        p.emit(asm::ret(30));

        // The worker: x0 = the heap.
        p.label("worker");
        p.emit(mov_imm(9, 1000));
        p.label("worker_loop");
        p.emit(asm::exclusive(3, false, true, false, 31, false, 31, X0, 10));
        p.emit(asm::add_imm(true, 10, 10, 1));
        p.emit(asm::exclusive(
            3, false, false, false, 11, false, 31, X0, 10,
        ));
        p.cbnz(false, 11, "worker_loop");
        p.emit(asm::sub_imm(true, 9, 9, 1));
        p.cbnz(true, 9, "worker_loop");
        p.emit(asm::ldr_x(X0, X0, 8));
        p.emit(asm::svc(0x11));
        p.emit(asm::svc(0x0A));

        data_string(&mut p, "hello", "hello from a homebrew NRO\n");
        data_string(&mut p, "counter_ok", "counter ok\n");
        data_string(&mut p, "sm_ok", "sm ok\n");
        data_string(&mut p, "done", "done\n");
        data_string(&mut p, "failed", "FAILED\n");
        data_string(&mut p, "sm_name", "sm:");
        let worker = p.addr_of("worker");
        (p.finish(), worker)
    }

    fn nro_program(code: Vec<u32>) -> Program {
        let parts = build_parts(&Spec {
            code,
            ..Spec::default()
        });
        let bytes = build_nro(&parts, 0x1000, None);
        let loaded = nro::load(&bytes, "hello", GUEST_BASE).unwrap();
        Program {
            name: "hello".into(),
            modules: vec![loaded.module],
            npdm: None,
        }
    }

    /// Build and run `hello_nro`. Only `ADR` refers to the worker, so the analysis does not find
    /// it by itself: with `worker_is_native` its address is given as a root (the `--hints` of the
    /// command line) and the thread runs compiled code; without it the thread starts in the slow
    /// path and runs interpreted, racing the compiled main thread.
    fn run_hello(worker_is_native: bool, name: &str) {
        let Some(tc) = toolchain_or_skip() else {
            return;
        };
        let (code, worker) = hello_nro();
        let program = nro_program(code);
        let dir = TempDir::new(name);
        let mut options = BuildOptions::new(dir.path(), "hello");
        if worker_is_native {
            options.hints = vec![worker];
        }
        let report = build(&program, &tc, &options).unwrap();
        assert_eq!(report.functions, if worker_is_native { 2 } else { 1 });
        let out = run(&report.exe);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "stdout: {} stderr: {stderr}",
            String::from_utf8_lossy(&out.stdout)
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "hello from a homebrew NRO\ncounter ok\nsm ok\ndone\n"
        );
    }

    #[test]
    fn a_homebrew_nro_prints_runs_threads_and_talks_to_the_service_manager() {
        run_hello(true, "e2e-nro");
    }

    #[test]
    fn a_thread_that_starts_in_the_slow_path_races_compiled_code_correctly() {
        run_hello(false, "e2e-nro-slow");
    }

    #[test]
    fn asking_for_an_unimplemented_service_stops_with_its_name() {
        let Some(tc) = toolchain_or_skip() else {
            return;
        };
        let mut p = Prog::new(GUEST_BASE + CODE_OFF as u64);
        connect_sm(&mut p);
        sm_request(&mut p, 0, true, &[0; 8]);
        sm_request(&mut p, 1, false, b"ldr:ro\0\0");
        p.emit(mov_imm(X0, 0));
        p.emit(asm::ret(30));
        data_string(&mut p, "sm_name", "sm:");
        let program = nro_program(p.finish());
        let dir = TempDir::new("e2e-nro-stop");
        let report = build(&program, &tc, &BuildOptions::new(dir.path(), "askldr")).unwrap();
        let out = run(&report.exe);
        assert_eq!(out.status.code(), Some(101));
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("`ldr:ro`") && err.contains("not implement"),
            "{err}"
        );
        assert!(
            err.contains("SVC at 0x80"),
            "the report names the instruction: {err}"
        );
    }

    #[test]
    fn an_nso_is_entered_with_the_main_thread_handle() {
        let Some(tc) = toolchain_or_skip() else {
            return;
        };
        // x0 must be zero and x1 a handle; return 42 if so.
        let mut p = Prog::new(GUEST_BASE + CODE_OFF as u64);
        p.cbnz(true, X0, "bad");
        p.cbz(true, X1, "bad");
        p.emit(mov_imm(X0, 42));
        p.emit(asm::ret(30));
        p.label("bad");
        p.emit(mov_imm(X0, 1));
        p.emit(asm::ret(30));
        let program = program_from(&Spec {
            code: p.finish(),
            ..Spec::default()
        });
        let dir = TempDir::new("e2e-nso-entry");
        let report = build(&program, &tc, &BuildOptions::new(dir.path(), "entry")).unwrap();
        assert_eq!(run(&report.exe).status.code(), Some(42));
    }

    #[test]
    fn exit_process_ends_the_program_while_another_thread_still_runs() {
        let Some(tc) = toolchain_or_skip() else {
            return;
        };
        let mut p = Prog::new(GUEST_BASE + CODE_OFF as u64);
        // A heap for the second thread's stack.
        p.emit(asm::movz(true, X1, 0x20, 1));
        p.emit(asm::svc(0x01));
        p.emit(mov_reg(21, X1));
        // The thread sleeps forever in a loop.
        p.adr(X1, "sleeper");
        p.emit(mov_imm(X2, 0));
        p.emit(asm::movz(true, X3, 0x1F, 1));
        p.emit(asm::add_sub_shifted(true, false, false, 0, 21, 0, X3, X3));
        p.emit(mov_imm(X4, 0x2C));
        p.emit(asm::movz(false, X5, 0xFFFE, 0));
        p.emit(asm::movk(false, X5, 0xFFFF, 1));
        p.emit(mov_imm(X0, 0));
        p.emit(asm::svc(0x08));
        p.emit(mov_reg(X0, X1));
        p.emit(asm::svc(0x09));
        // Give it time to start, then end the process.
        p.emit(asm::movz(true, X0, 0x4240, 0));
        p.emit(asm::movk(true, X0, 0x000F, 1));
        p.emit(asm::svc(0x0B));
        print(&mut p, "bye", "bye\n");
        p.emit(asm::svc(0x07));
        p.emit(mov_imm(X0, 9));
        p.emit(asm::ret(30));
        p.label("sleeper");
        p.emit(asm::movz(true, X0, 0x4240, 0));
        p.emit(asm::movk(true, X0, 0x000F, 1));
        p.emit(asm::svc(0x0B));
        p.b("sleeper");
        data_string(&mut p, "bye", "bye\n");
        let program = nro_program(p.finish());
        let dir = TempDir::new("e2e-exit");
        let report = build(&program, &tc, &BuildOptions::new(dir.path(), "exiter")).unwrap();
        let start = std::time::Instant::now();
        let out = run(&report.exe);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&out.stdout), "bye\n");
        assert!(start.elapsed() < std::time::Duration::from_secs(20));
    }
}
