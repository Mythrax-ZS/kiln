//! Routines run natively (docs/ASSUMPTIONS.md A-187): a function a module exports under one of
//! the names in `nx_runtime::rt::hle::ROUTINES` is not translated. Its generated function calls
//! the runtime's routine and returns to the link register, as the guest function would have.

use crate::loader::module::Module;
use nx_runtime::a64::state::offsets as st;
use nx_runtime::rt::abi::offsets as cpu_offsets;
use std::collections::BTreeMap;
use std::fmt::Write;

/// The functions to run natively, by guest address, with the routine's number.
pub fn routines(modules: &[Module]) -> BTreeMap<u64, u32> {
    let mut out = BTreeMap::new();
    for m in modules {
        for e in &m.exports {
            if let Some(&(_, id)) = nx_runtime::rt::hle::ROUTINES
                .iter()
                .find(|(name, _)| *name == e.name)
            {
                out.insert(m.base + e.offset, id);
            }
        }
    }
    out
}

/// The generated function of a routine run natively: the call, the test of the fault flag that
/// every access has, and the return to the link register.
pub fn emit_stub(addr: u64, id: u32, out: &mut String) {
    let _ = writeln!(
        out,
        "define hidden void @{}(ptr %cpu) nounwind {{",
        super::llvm::func_symbol(addr)
    );
    let _ = writeln!(
        out,
        "  call void @nx_hle(ptr %cpu, i32 {id}, i64 {})",
        addr as i64
    );
    let _ = writeln!(
        out,
        "  %fp = getelementptr i8, ptr %cpu, i64 {}",
        cpu_offsets::FAULT_FLAG
    );
    let _ = writeln!(out, "  %f = load i32, ptr %fp");
    let _ = writeln!(out, "  %c = icmp ne i32 %f, 0");
    let _ = writeln!(out, "  br i1 %c, label %fault, label %back, !prof !0");
    let _ = writeln!(out, "back:");
    let _ = writeln!(
        out,
        "  %lrp = getelementptr i8, ptr %cpu, i64 {}",
        st::X + 8 * 30
    );
    let _ = writeln!(out, "  %lr = load i64, ptr %lrp");
    let _ = writeln!(out, "  %pcp = getelementptr i8, ptr %cpu, i64 {}", st::PC);
    let _ = writeln!(out, "  store i64 %lr, ptr %pcp");
    let _ = writeln!(out, "  ret void");
    let _ = writeln!(out, "fault:");
    let _ = writeln!(out, "  ret void");
    let _ = writeln!(out, "}}");
}
