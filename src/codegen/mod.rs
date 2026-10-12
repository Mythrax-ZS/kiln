//! Code generation: lifted functions to LLVM IR text, grouped into translation units.
//!
//! A *function unit* holds a batch of lifted functions (all with hidden visibility, so units can
//! reference each other); the *support unit* holds the dispatcher, the function table that
//! `nx_rt_register` takes, and the init function that registers it. Splitting the output lets
//! clang compile the units in parallel and keeps any one file a manageable size.

pub mod llvm;

use crate::lift::ir::Func;
use llvm::{declarations, dispatcher, emit_function, func_symbol, METADATA};
use std::collections::BTreeSet;
use std::fmt::Write;

/// The target triple the generated code is compiled for: the host.
pub fn host_triple() -> &'static str {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "x86_64-unknown-linux-gnu"
    } else {
        panic!("recomp produces x86_64 code for Windows and Linux hosts only")
    }
}

/// One translation unit with these functions.
pub fn function_unit(triple: &str, funcs: &[&Func]) -> String {
    let mut body = String::new();
    let mut intrinsics = BTreeSet::new();
    let mut callees = BTreeSet::new();
    for f in funcs {
        emit_function(f, &mut body, &mut intrinsics, &mut callees);
    }
    let defined: BTreeSet<u64> = funcs.iter().map(|f| f.addr).collect();
    let mut s = String::new();
    let _ = writeln!(s, "target triple = \"{triple}\"\n");
    s.push_str(&declarations(false));
    for addr in callees.difference(&defined) {
        let _ = writeln!(s, "declare hidden void @{}(ptr)", func_symbol(*addr));
    }
    for i in &intrinsics {
        let _ = writeln!(s, "{i}");
    }
    s.push('\n');
    s.push_str(&body);
    s.push_str(METADATA);
    s
}

/// The dispatcher, the function table for `entries` (every generated function's guest address)
/// and `nx_module_init`, which registers the table with the runtime.
pub fn support_unit(triple: &str, entries: &[u64]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "target triple = \"{triple}\"\n");
    s.push_str(&declarations(true));
    s.push_str("declare void @nx_rt_register(ptr, i64)\n");
    for a in entries {
        let _ = writeln!(s, "declare hidden void @{}(ptr)", func_symbol(*a));
    }
    s.push('\n');
    s.push_str(&dispatcher());
    let mut sorted = entries.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let _ = write!(
        s,
        "@nx_functab = internal constant [{} x {{ i64, ptr }}] [",
        sorted.len()
    );
    for (i, a) in sorted.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let _ = write!(
            s,
            "{sep}\n  {{ i64, ptr }} {{ i64 {}, ptr @{} }}",
            *a as i64,
            func_symbol(*a)
        );
    }
    s.push_str("\n]\n\n");
    let _ = writeln!(
        s,
        "define void @nx_module_init() nounwind {{\n  call void @nx_rt_register(ptr @nx_functab, i64 {})\n  ret void\n}}\n",
        sorted.len()
    );
    s.push_str(METADATA);
    s
}
