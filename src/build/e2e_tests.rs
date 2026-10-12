//! End to end: a synthetic NSO is recompiled into a native executable and shared library, and both
//! are run. Needs clang; skips loudly without it.

use super::pipeline::{build, BuildOptions};
use super::Toolchain;
use crate::lift::flow_tests::{fib_program, jump_table_words};
use crate::loader::module::GUEST_BASE;
use crate::loader::nso;
use crate::loader::testing::{build_nso, build_parts, NsoFlags, Spec, CODE_OFF};
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
        module: loaded.module,
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
        err.contains("unhandled supervisor call SVC #0x2a at 0x8000184"),
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
