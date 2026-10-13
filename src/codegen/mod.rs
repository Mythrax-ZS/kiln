//! Code generation: lifted functions to LLVM IR text, grouped into translation units.
//!
//! A *function unit* holds a batch of lifted functions (all with hidden visibility, so units can
//! reference each other); the *support unit* holds the dispatcher, the function table that
//! `nx_rt_register` takes, and the init function that registers it. Splitting the output lets
//! clang compile the units in parallel and keeps any one file a manageable size.

mod descriptor;
pub mod llvm;

pub use descriptor::{descriptor_unit, Descriptor, SegmentInfo};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lift::discover::{discover, Options};
    use crate::lift::lift_all;
    use crate::lift::tests::{Image, CODE};
    use nx_runtime::testing::asm;

    fn unit_for(words: Vec<u32>) -> String {
        let d = discover(&Image::new(words), &[CODE], &Options::default());
        let funcs = lift_all(&d);
        let refs: Vec<&crate::lift::ir::Func> = funcs.values().collect();
        function_unit(host_triple(), &refs)
    }

    #[test]
    fn barriers_become_fences_that_stop_reordering() {
        let ll = unit_for(vec![
            asm::dmb(0b1001), // dmb ishld
            asm::dmb(0b1010), // dmb ishst
            asm::dmb(0b1011), // dmb ish
            asm::dsb(0b1111), // dsb sy
            asm::isb(0b1111),
            asm::ret(30),
        ]);
        assert_eq!(ll.matches("fence acquire").count(), 1, "{ll}");
        assert_eq!(ll.matches("fence release").count(), 1, "{ll}");
        assert_eq!(ll.matches("fence seq_cst").count(), 2, "{ll}");
        assert_eq!(
            ll.matches("fence syncscope(\"singlethread\") seq_cst")
                .count(),
            1,
            "{ll}"
        );
    }

    #[test]
    fn acquire_and_release_accesses_use_the_ordered_runtime_calls() {
        // ldar x0, [x1] ; stlr x0, [x1] ; ldr x0, [x1] ; str x0, [x1]
        let ll = unit_for(vec![
            asm::exclusive(3, true, true, false, 31, true, 31, 1, 0),
            asm::exclusive(3, true, false, false, 31, true, 31, 1, 0),
            asm::ldst_unsigned(3, 1, 0, 1, 0),
            asm::ldst_unsigned(3, 0, 0, 1, 0),
            asm::ret(30),
        ]);
        assert!(ll.contains("@nx_ldar64(ptr %cpu"), "{ll}");
        assert!(ll.contains("@nx_stlr64(ptr %cpu"), "{ll}");
        assert!(ll.contains("@nx_ld64(ptr %cpu"), "{ll}");
        assert!(ll.contains("@nx_st64(ptr %cpu"), "{ll}");
        // Every access is followed by a test of the fault flag.
        assert_eq!(ll.matches("icmp ne i32").count(), 4, "{ll}");
    }

    #[test]
    fn the_hot_floating_point_forms_are_lowered_and_the_others_keep_the_runtime_call() {
        // fadd s0, s1, s2 ; mov v3.16b, v4.16b ; ldr q5, [x0] ; fsqrt s6, s7
        let ll = unit_for(vec![
            0x1E22_2820,
            0x4EA4_1C83,
            0x3DC0_0005,
            0x1E21_C0E6,
            asm::ret(30),
        ]);
        // The add is IR, with the runtime call as its slow form; so is the single-precision
        // inexact test.
        assert!(ll.contains("fadd float"), "{ll}");
        // The move and the load have no call; the square root is only a call.
        assert_eq!(
            ll.matches("call i64 @nx_vec(").count(),
            2,
            "the slow form of the add and the square root: {ll}"
        );
        assert_eq!(ll.matches("call void @nx_vecmem(").count(), 0, "{ll}");
        // A 128-bit load is two 64-bit loads, each followed by the test of the fault flag.
        assert_eq!(ll.matches("@nx_ld64(ptr %cpu").count(), 2, "{ll}");
        assert_eq!(ll.matches("icmp ne i32").count(), 2, "{ll}");
    }

    #[test]
    fn divisions_are_guarded() {
        let ll = unit_for(vec![
            asm::dp2(true, 2, 1, 0, 2), // udiv x2, x0, x1
            asm::dp2(true, 3, 1, 0, 3), // sdiv x3, x0, x1
            asm::ret(30),
        ]);
        assert!(ll.contains("udiv i64"), "{ll}");
        assert!(ll.contains("sdiv i64"), "{ll}");
        // A zero divisor is replaced and the result forced to zero; MIN / -1 divides by one.
        assert!(
            ll.contains("icmp eq i64 %v") && ll.contains("select i1"),
            "{ll}"
        );
        assert!(
            ll.contains("-9223372036854775808") && ll.contains(", -1"),
            "{ll}"
        );
    }

    #[test]
    fn every_runtime_function_the_code_calls_is_declared_and_exported() {
        let declared: std::collections::BTreeSet<String> = llvm::declarations(false)
            .lines()
            .filter_map(|l| l.split('@').nth(1))
            .filter_map(|r| r.split('(').next())
            .map(str::to_string)
            .collect();
        let mut exported: std::collections::BTreeSet<String> =
            nx_runtime::rt::exports::GENERATED_CODE_SYMBOLS
                .iter()
                .map(|s| s.to_string())
                .collect();
        // The dispatcher is generated, not part of the runtime.
        exported.insert("nx_dispatch".into());
        assert_eq!(declared, exported);
    }

    #[test]
    fn the_support_unit_registers_a_sorted_table() {
        let ll = support_unit(host_triple(), &[0x3000, 0x1000, 0x2000, 0x1000]);
        let a = ll.find("i64 4096").expect("0x1000");
        let b = ll.find("i64 8192").expect("0x2000");
        let c = ll.find("i64 12288").expect("0x3000");
        assert!(a < b && b < c);
        assert!(ll.contains("[3 x { i64, ptr }]"), "{ll}");
        assert!(ll.contains("@nx_rt_register(ptr @nx_functab, i64 3)"));
        assert!(ll.contains("musttail call void %fn(ptr %cpu)"));
    }
}
