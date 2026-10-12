//! IR to LLVM IR text.
//!
//! One LLVM function per guest function, `void f(ptr cpu)`, following the calling protocol in
//! `nx_runtime::rt::abi`. SSA values map one to one onto LLVM values; the guest state lives in
//! the CPU structure only at calls and exits, where the spilled variables are stored.
//!
//! What the lowering adds to the IR's plain meaning:
//! - `UDiv`/`SDiv` are guarded: a zero divisor is replaced by one and the result by zero, which is
//!   what A64 gives, and `MIN / -1` divides by one instead, so it yields `MIN` (the wrapped
//!   quotient) without the undefined behaviour LLVM assigns to it;
//! - every memory access is a call into the runtime, so LLVM can neither reorder nor remove
//!   accesses, and is followed by a test of the fault flag that leaves the function;
//! - barriers become LLVM fences (`DMB LD` acquire, `DMB ST` release, other barriers sequentially
//!   consistent, `ISB` a compiler-only barrier);
//! - a call is followed by a check that control came back to the return address.

use crate::lift::ir::*;
use nx_runtime::rt::abi::{offsets as cpu_offsets, trap};
use std::collections::BTreeSet;
use std::fmt::Write;

use nx_runtime::a64::state::offsets as st;

/// Symbol of the generated function for a guest address.
pub fn func_symbol(addr: u64) -> String {
    format!("f_{addr:x}")
}

/// Declarations every module needs: the runtime's entry points. The dispatcher is declared too
/// unless `defines_dispatcher` (the unit that defines it).
pub fn declarations(defines_dispatcher: bool) -> String {
    let mut s = String::new();
    for d in [
        "declare i8 @nx_ld8(ptr, i64, i64)",
        "declare i16 @nx_ld16(ptr, i64, i64)",
        "declare i32 @nx_ld32(ptr, i64, i64)",
        "declare i64 @nx_ld64(ptr, i64, i64)",
        "declare void @nx_st8(ptr, i64, i8, i64)",
        "declare void @nx_st16(ptr, i64, i16, i64)",
        "declare void @nx_st32(ptr, i64, i32, i64)",
        "declare void @nx_st64(ptr, i64, i64, i64)",
        "declare i8 @nx_ldar8(ptr, i64, i64)",
        "declare i16 @nx_ldar16(ptr, i64, i64)",
        "declare i32 @nx_ldar32(ptr, i64, i64)",
        "declare i64 @nx_ldar64(ptr, i64, i64)",
        "declare void @nx_stlr8(ptr, i64, i8, i64)",
        "declare void @nx_stlr16(ptr, i64, i16, i64)",
        "declare void @nx_stlr32(ptr, i64, i32, i64)",
        "declare void @nx_stlr64(ptr, i64, i64, i64)",
        "declare i64 @nx_ldx(ptr, i64, i64, i32, i32)",
        "declare i32 @nx_stx(ptr, i64, i64, i64, i64, i32, i32)",
        "declare void @nx_clrex(ptr)",
        "declare void @nx_dczva(ptr, i64, i64)",
        "declare i64 @nx_counter()",
        "declare i32 @nx_crc32(i32, i64, i32, i32) memory(none) nounwind",
        "declare void @nx_trap(ptr, i32, i64, i64, i64)",
        "declare void @nx_svc(ptr, i32)",
        "declare void @nx_interp1(ptr)",
        "declare ptr @nx_lookup(i64)",
        "declare void @nx_slow_path(ptr)",
    ] {
        s.push_str(d);
        s.push('\n');
    }
    if !defines_dispatcher {
        s.push_str("declare hidden void @nx_dispatch(ptr)\n");
    }
    s
}

/// Branch weights for "this access faulted": taken almost never.
pub const METADATA: &str = "!0 = !{!\"branch_weights\", i32 1, i32 4000}\n";

fn const_text(ty: Ty, v: u128) -> String {
    match ty {
        Ty::I1 => (if v != 0 { "true" } else { "false" }).to_string(),
        Ty::I8 => (v as u8 as i8).to_string(),
        Ty::I16 => (v as u16 as i16).to_string(),
        Ty::I32 => (v as u32 as i32).to_string(),
        Ty::I64 => (v as u64 as i64).to_string(),
        Ty::I128 => (v as i128).to_string(),
        Ty::Void => unreachable!("no void constants"),
    }
}

/// The trap code and the two words of detail the runtime records.
fn trap_args(kind: TrapKind) -> (u32, u64, u64) {
    match kind {
        TrapKind::Undefined(w) => (trap::UNDEFINED, u64::from(w), 0),
        TrapKind::Unsupported(w) => (trap::UNSUPPORTED, u64::from(w), 0),
        TrapKind::Brk(i) => (trap::BRK, u64::from(i), 0),
        TrapKind::SysReg { write, enc } => {
            let [a, b, c, d, e] = enc;
            let packed = (u64::from(a) << 14)
                | (u64::from(b) << 11)
                | (u64::from(c) << 7)
                | (u64::from(d) << 3)
                | u64::from(e);
            (trap::SYSREG, packed, u64::from(write))
        }
        TrapKind::SysInst { load, enc } => {
            let [op1, crn, crm, op2] = enc;
            let packed = (u64::from(op1) << 11)
                | (u64::from(crn) << 7)
                | (u64::from(crm) << 3)
                | u64::from(op2);
            (trap::SYSINST, packed, u64::from(load))
        }
        TrapKind::OutOfCode => (trap::OUT_OF_CODE, 0, 0),
    }
}

/// Edges of a terminator with multiplicity (a switch can reach one block through several cases).
fn edges(t: &Term) -> Vec<BlockId> {
    match t {
        Term::None | Term::Exit { .. } => Vec::new(),
        Term::Br(b) => vec![*b],
        Term::CondBr { then, els, .. } => vec![*then, *els],
        Term::Switch { cases, default, .. } => {
            let mut v: Vec<BlockId> = cases.iter().map(|&(_, b)| b).collect();
            v.push(*default);
            v
        }
        Term::Call(c) => vec![c.cont],
    }
}

fn can_fault(op: &Op) -> bool {
    matches!(
        op,
        Op::Load { .. }
            | Op::Store { .. }
            | Op::LoadExcl { .. }
            | Op::StoreExcl { .. }
            | Op::DcZva { .. }
    )
}

struct Emitter<'a> {
    f: &'a Func,
    out: String,
    /// Guest addresses of the functions this one calls or tail-calls directly.
    callees: BTreeSet<u64>,
    /// Offsets of the CPU fields used, for the prologue.
    ptrs: BTreeSet<usize>,
    /// Intrinsic declarations needed.
    intrinsics: BTreeSet<String>,
    /// Label of the last LLVM block of each IR block.
    last_label: Vec<String>,
    /// Predecessor edge counts: `edge_count[(pred, succ)]`.
    tmp: u32,
}

fn label(b: BlockId) -> String {
    format!("b{b}")
}

impl<'a> Emitter<'a> {
    fn val(&self, v: ValueId) -> String {
        match self.f.op(v) {
            Op::Const(c) => const_text(self.f.ty(v), *c),
            _ => format!("%v{v}"),
        }
    }

    fn tv(&self, v: ValueId) -> String {
        format!("{} {}", self.f.ty(v).llvm(), self.val(v))
    }

    fn ptr(&mut self, off: usize) -> String {
        self.ptrs.insert(off);
        format!("%p{off}")
    }

    fn temp(&mut self, hint: &str) -> String {
        self.tmp += 1;
        format!("%t{}.{hint}", self.tmp)
    }

    fn intrinsic(&mut self, decl: String) {
        self.intrinsics.insert(decl);
    }

    fn line(&mut self, s: &str) {
        self.out.push_str("  ");
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn spill(&mut self, spill: &[(Var, ValueId)]) {
        let mut flags: [Option<ValueId>; 4] = [None; 4];
        for &(var, v) in spill {
            match var {
                Var::X(n) => {
                    let p = self.ptr(st::X + 8 * usize::from(n));
                    let s = format!("store i64 {}, ptr {p}", self.val(v));
                    self.line(&s);
                }
                Var::Sp => {
                    let p = self.ptr(st::SP);
                    let s = format!("store i64 {}, ptr {p}", self.val(v));
                    self.line(&s);
                }
                Var::N => flags[0] = Some(v),
                Var::Z => flags[1] = Some(v),
                Var::C => flags[2] = Some(v),
                Var::V => flags[3] = Some(v),
            }
        }
        if flags.iter().any(Option::is_some) {
            let mut acc = "0".to_string();
            for (i, f) in flags.iter().enumerate() {
                let f = f.expect("all four flags are spilled together");
                let z = self.temp("fz");
                let s = format!("{z} = zext i1 {} to i32", self.val(f));
                self.line(&s);
                let sh = self.temp("fs");
                let s = format!("{sh} = shl i32 {z}, {}", 31 - i);
                self.line(&s);
                let or = self.temp("fo");
                let s = format!("{or} = or i32 {acc}, {sh}");
                self.line(&s);
                acc = or;
            }
            let p = self.ptr(st::NZCV);
            let s = format!("store i32 {acc}, ptr {p}");
            self.line(&s);
        }
    }

    /// Test the fault flag after an access and leave the function if it is set.
    fn fault_check(&mut self, b: BlockId, split: &mut u32) {
        let p = self.ptr(cpu_offsets::FAULT_FLAG);
        let f = self.temp("ff");
        let s = format!("{f} = load i32, ptr {p}");
        self.line(&s);
        let c = self.temp("fc");
        let s = format!("{c} = icmp ne i32 {f}, 0");
        self.line(&s);
        *split += 1;
        let next = format!("b{b}.m{split}");
        let s = format!("br i1 {c}, label %fault, label %{next}, !prof !0");
        self.line(&s);
        let _ = writeln!(self.out, "{next}:");
    }

    fn body_inst(&mut self, b: BlockId, v: ValueId, split: &mut u32) {
        let inst = &self.f.insts[v as usize];
        let ty = inst.ty;
        let t = ty.llvm();
        let r = format!("%v{v}");
        match inst.op.clone() {
            Op::Const(_) | Op::Phi(_) => unreachable!("not in a block body"),
            Op::LoadVar(var) => match var {
                Var::X(n) => {
                    let p = self.ptr(st::X + 8 * usize::from(n));
                    self.line(&format!("{r} = load i64, ptr {p}"));
                }
                Var::Sp => {
                    let p = self.ptr(st::SP);
                    self.line(&format!("{r} = load i64, ptr {p}"));
                }
                flag => {
                    let p = self.ptr(st::NZCV);
                    let w = self.temp("nz");
                    self.line(&format!("{w} = load i32, ptr {p}"));
                    let sh = self.temp("ns");
                    self.line(&format!("{sh} = lshr i32 {w}, {}", flag.nzcv_bit()));
                    self.line(&format!("{r} = trunc i32 {sh} to i1"));
                }
            },
            Op::Bin(op, a, c) => {
                let (av, cv) = (self.val(a), self.val(c));
                match op {
                    BinOp::UDiv => {
                        let z = self.temp("dz");
                        self.line(&format!("{z} = icmp eq {t} {cv}, 0"));
                        let safe = self.temp("ds");
                        self.line(&format!("{safe} = select i1 {z}, {t} 1, {t} {cv}"));
                        let q = self.temp("dq");
                        self.line(&format!("{q} = udiv {t} {av}, {safe}"));
                        self.line(&format!("{r} = select i1 {z}, {t} 0, {t} {q}"));
                    }
                    BinOp::SDiv => {
                        let min = const_text(ty, 1u128 << (ty.bits() - 1));
                        let z = self.temp("dz");
                        self.line(&format!("{z} = icmp eq {t} {cv}, 0"));
                        let m = self.temp("dm");
                        self.line(&format!("{m} = icmp eq {t} {av}, {min}"));
                        let n1 = self.temp("dn");
                        self.line(&format!("{n1} = icmp eq {t} {cv}, -1"));
                        let ovf = self.temp("do");
                        self.line(&format!("{ovf} = and i1 {m}, {n1}"));
                        let bad = self.temp("db");
                        self.line(&format!("{bad} = or i1 {z}, {ovf}"));
                        let safe = self.temp("ds");
                        self.line(&format!("{safe} = select i1 {bad}, {t} 1, {t} {cv}"));
                        let q = self.temp("dq");
                        self.line(&format!("{q} = sdiv {t} {av}, {safe}"));
                        self.line(&format!("{r} = select i1 {z}, {t} 0, {t} {q}"));
                    }
                    _ => self.line(&format!("{r} = {} {t} {av}, {cv}", op.name())),
                }
            }
            Op::Un(op, a) => {
                let av = self.val(a);
                match op {
                    UnOp::Clz => {
                        self.intrinsic(format!("declare {t} @llvm.ctlz.{t}({t}, i1)"));
                        self.line(&format!(
                            "{r} = call {t} @llvm.ctlz.{t}({t} {av}, i1 false)"
                        ));
                    }
                    UnOp::Bswap => {
                        self.intrinsic(format!("declare {t} @llvm.bswap.{t}({t})"));
                        self.line(&format!("{r} = call {t} @llvm.bswap.{t}({t} {av})"));
                    }
                    UnOp::Rbit => {
                        self.intrinsic(format!("declare {t} @llvm.bitreverse.{t}({t})"));
                        self.line(&format!("{r} = call {t} @llvm.bitreverse.{t}({t} {av})"));
                    }
                }
            }
            Op::Cast(op, a) => {
                let line = format!("{r} = {} {} to {t}", op.name(), self.tv(a));
                self.line(&line);
            }
            Op::Icmp(p, a, c) => {
                let at = self.f.ty(a).llvm();
                let line = format!(
                    "{r} = icmp {} {at} {}, {}",
                    p.name(),
                    self.val(a),
                    self.val(c)
                );
                self.line(&line);
            }
            Op::Select(c, a, d) => {
                let line = format!(
                    "{r} = select i1 {}, {}, {}",
                    self.val(c),
                    self.tv(a),
                    self.tv(d)
                );
                self.line(&line);
            }
            Op::Fshr(hi, lo, amt) => {
                self.intrinsic(format!("declare {t} @llvm.fshr.{t}({t}, {t}, {t})"));
                let line = format!(
                    "{r} = call {t} @llvm.fshr.{t}({}, {}, {})",
                    self.tv(hi),
                    self.tv(lo),
                    self.tv(amt)
                );
                self.line(&line);
            }
            Op::MulHi { signed, a, b: c } => {
                let ext = if signed { "sext" } else { "zext" };
                let ea = self.temp("ma");
                self.line(&format!("{ea} = {ext} i64 {} to i128", self.val(a)));
                let ec = self.temp("mb");
                self.line(&format!("{ec} = {ext} i64 {} to i128", self.val(c)));
                let m = self.temp("mm");
                self.line(&format!("{m} = mul i128 {ea}, {ec}"));
                let h = self.temp("mh");
                self.line(&format!("{h} = lshr i128 {m}, 64"));
                self.line(&format!("{r} = trunc i128 {h} to i64"));
            }
            Op::Crc32 {
                size,
                castagnoli,
                acc,
                data,
            } => {
                let line = format!(
                    "{r} = call i32 @nx_crc32({}, {}, i32 {size}, i32 {})",
                    self.tv(acc),
                    self.tv(data),
                    u8::from(castagnoli)
                );
                self.line(&line);
            }
            Op::Load {
                bytes,
                order,
                pc,
                addr,
            } => {
                let name = if order == Order::Plain { "ld" } else { "ldar" };
                let line = format!(
                    "{r} = call {t} @nx_{name}{}(ptr %cpu, i64 {}, i64 {})",
                    u32::from(bytes) * 8,
                    self.val(addr),
                    pc as i64
                );
                self.line(&line);
                self.fault_check(b, split);
            }
            Op::Store {
                bytes,
                order,
                pc,
                addr,
                value,
            } => {
                let name = if order == Order::Plain { "st" } else { "stlr" };
                let line = format!(
                    "call void @nx_{name}{}(ptr %cpu, i64 {}, {}, i64 {})",
                    u32::from(bytes) * 8,
                    self.val(addr),
                    self.tv(value),
                    pc as i64
                );
                self.line(&line);
                self.fault_check(b, split);
            }
            Op::LoadExcl {
                bytes,
                acquire,
                pc,
                addr,
            } => {
                let lo = if bytes == 16 {
                    self.temp("xl")
                } else {
                    r.clone()
                };
                let line = format!(
                    "{lo} = call i64 @nx_ldx(ptr %cpu, i64 {}, i64 {}, i32 {bytes}, i32 {})",
                    self.val(addr),
                    pc as i64,
                    u8::from(acquire)
                );
                self.line(&line);
                if bytes == 16 {
                    let p = self.ptr(cpu_offsets::AUX);
                    let hi = self.temp("xh");
                    self.line(&format!("{hi} = load i64, ptr {p}"));
                    let l128 = self.temp("xz");
                    self.line(&format!("{l128} = zext i64 {lo} to i128"));
                    let h128 = self.temp("xy");
                    self.line(&format!("{h128} = zext i64 {hi} to i128"));
                    let sh = self.temp("xs");
                    self.line(&format!("{sh} = shl i128 {h128}, 64"));
                    self.line(&format!("{r} = or i128 {l128}, {sh}"));
                }
                self.fault_check(b, split);
            }
            Op::StoreExcl {
                bytes,
                release,
                pc,
                addr,
                value,
            } => {
                let (lo, hi) = if bytes == 16 {
                    let lo = self.temp("sl");
                    self.line(&format!("{lo} = trunc i128 {} to i64", self.val(value)));
                    let sh = self.temp("ss");
                    self.line(&format!("{sh} = lshr i128 {}, 64", self.val(value)));
                    let hi = self.temp("sh");
                    self.line(&format!("{hi} = trunc i128 {sh} to i64"));
                    (lo, hi)
                } else {
                    (self.val(value), "0".to_string())
                };
                let line = format!(
                    "{r} = call i32 @nx_stx(ptr %cpu, i64 {}, i64 {lo}, i64 {hi}, i64 {}, i32 {bytes}, i32 {})",
                    self.val(addr),
                    pc as i64,
                    u8::from(release)
                );
                self.line(&line);
                self.fault_check(b, split);
            }
            Op::ClearExcl => self.line("call void @nx_clrex(ptr %cpu)"),
            Op::Fence(kind) => match kind {
                Fence::Acquire => self.line("fence acquire"),
                Fence::Release => self.line("fence release"),
                Fence::Full => self.line("fence seq_cst"),
                Fence::Instruction => self.line("fence syncscope(\"singlethread\") seq_cst"),
            },
            Op::DcZva { pc, addr } => {
                let line = format!(
                    "call void @nx_dczva(ptr %cpu, i64 {}, i64 {})",
                    self.val(addr),
                    pc as i64
                );
                self.line(&line);
                self.fault_check(b, split);
            }
            Op::Counter => self.line(&format!("{r} = call i64 @nx_counter()")),
            Op::LoadField(fld) => {
                let p = self.ptr(fld.offset());
                self.line(&format!("{r} = load {t}, ptr {p}"));
            }
            Op::StoreField(fld, val) => {
                let p = self.ptr(fld.offset());
                let line = format!("store {}, ptr {p}", self.tv(val));
                self.line(&line);
            }
        }
    }

    fn phi(&mut self, b: BlockId, v: ValueId) {
        let Op::Phi(ops) = self.f.op(v).clone() else {
            unreachable!("phi list holds phis")
        };
        let t = self.f.ty(v).llvm();
        let mut parts = Vec::new();
        for (p, val) in ops {
            let n = edges(&self.f.blocks[p as usize].term)
                .iter()
                .filter(|&&s| s == b)
                .count();
            for _ in 0..n {
                parts.push(format!(
                    "[ {}, %{} ]",
                    self.val(val),
                    self.last_label[p as usize]
                ));
            }
        }
        self.line(&format!("%v{v} = phi {t} {}", parts.join(", ")));
    }

    fn target_pc(&mut self, pc: &str) {
        let p = self.ptr(st::PC);
        self.line(&format!("store i64 {pc}, ptr {p}"));
    }

    fn terminator(&mut self, b: BlockId) {
        let term = self.f.blocks[b as usize].term.clone();
        match term {
            Term::None => unreachable!("verified"),
            Term::Br(t) => self.line(&format!("br label %{}", label(t))),
            Term::CondBr { cond, then, els } => {
                let line = format!(
                    "br i1 {}, label %{}, label %{}",
                    self.val(cond),
                    label(then),
                    label(els)
                );
                self.line(&line);
            }
            Term::Switch {
                value,
                cases,
                default,
            } => {
                let mut s = format!(
                    "switch i64 {}, label %{} [",
                    self.val(value),
                    label(default)
                );
                for (a, t) in cases {
                    let _ = write!(s, " i64 {}, label %{}", a as i64, label(t));
                }
                s.push_str(" ]");
                self.line(&s);
            }
            Term::Exit { kind, spill } => {
                self.spill(&spill);
                match kind {
                    ExitKind::Return(v) => {
                        let pc = self.val(v);
                        self.target_pc(&pc);
                        self.line("ret void");
                    }
                    ExitKind::Jump(addr) => {
                        self.callees.insert(addr);
                        self.target_pc(&(addr as i64).to_string());
                        self.line(&format!(
                            "musttail call void @{}(ptr %cpu)",
                            func_symbol(addr)
                        ));
                        self.line("ret void");
                    }
                    ExitKind::Dispatch(v) => {
                        let pc = self.val(v);
                        self.target_pc(&pc);
                        self.line("musttail call void @nx_dispatch(ptr %cpu)");
                        self.line("ret void");
                    }
                    ExitKind::Trap { kind, pc } => {
                        let (code, a0, a1) = trap_args(kind);
                        self.line(&format!(
                            "call void @nx_trap(ptr %cpu, i32 {code}, i64 {}, i64 {}, i64 {})",
                            pc as i64, a0 as i64, a1 as i64
                        ));
                        self.line("ret void");
                    }
                }
            }
            Term::Call(c) => {
                self.spill(&c.spill);
                match c.target {
                    CallTarget::Direct(a) => {
                        self.callees.insert(a);
                        self.target_pc(&(a as i64).to_string());
                        self.line(&format!("call void @{}(ptr %cpu)", func_symbol(a)));
                    }
                    CallTarget::Indirect(v) => {
                        let pc = self.val(v);
                        self.target_pc(&pc);
                        self.line("call void @nx_dispatch(ptr %cpu)");
                    }
                    CallTarget::Svc(imm) => {
                        self.target_pc(&(c.ret_pc as i64).to_string());
                        self.line(&format!("call void @nx_svc(ptr %cpu, i32 {imm})"));
                    }
                    CallTarget::Fallback => {
                        self.target_pc(&(c.pc as i64).to_string());
                        self.line("call void @nx_interp1(ptr %cpu)");
                    }
                }
                let p = self.ptr(st::PC);
                let pc = self.temp("rp");
                self.line(&format!("{pc} = load i64, ptr {p}"));
                let ok = self.temp("rk");
                self.line(&format!("{ok} = icmp eq i64 {pc}, {}", c.ret_pc as i64));
                self.line(&format!(
                    "br i1 {ok}, label %{}, label %unwind",
                    label(c.cont)
                ));
            }
        }
    }
}

/// Emit one function. Every `Jump`/`Direct` target must be defined somewhere in the program (the
/// lifter only emits them for addresses that have a function); the addresses are added to
/// `callees` so the unit can declare the ones defined elsewhere.
pub fn emit_function(
    f: &Func,
    out: &mut String,
    intrinsics: &mut BTreeSet<String>,
    callees: &mut BTreeSet<u64>,
) {
    let mut e = Emitter {
        f,
        out: String::new(),
        callees: BTreeSet::new(),
        ptrs: BTreeSet::new(),
        intrinsics: BTreeSet::new(),
        last_label: Vec::new(),
        tmp: 0,
    };
    // Where each IR block ends in LLVM terms, for the phis of its successors.
    for b in &f.blocks {
        let splits = b
            .body
            .iter()
            .filter(|&&v| can_fault(&f.insts[v as usize].op))
            .count();
        let id = e.last_label.len() as BlockId;
        e.last_label.push(if splits == 0 {
            label(id)
        } else {
            format!("b{id}.m{splits}")
        });
    }
    let mut needs_fault = false;
    let mut needs_unwind = false;
    for (bi, b) in f.blocks.iter().enumerate() {
        let id = bi as BlockId;
        let _ = writeln!(e.out, "{}:", label(id));
        for &p in &b.phis {
            e.phi(id, p);
        }
        let mut split = 0;
        for &v in &b.body {
            e.body_inst(id, v, &mut split);
        }
        if split > 0 {
            needs_fault = true;
        }
        if matches!(b.term, Term::Call(_)) {
            needs_unwind = true;
        }
        e.terminator(id);
    }
    if needs_fault {
        e.out.push_str("fault:\n  ret void\n");
    }
    if needs_unwind {
        e.out.push_str("unwind:\n  ret void\n");
    }
    let name = func_symbol(f.addr);
    let _ = writeln!(out, "define hidden void @{name}(ptr %cpu) nounwind {{");
    // The first block is the entry: it gets the pointers to the CPU fields before anything else.
    let (first, rest) = e.out.split_once('\n').expect("a block label");
    out.push_str(first);
    out.push('\n');
    for off in &e.ptrs {
        let _ = writeln!(
            out,
            "  %p{off} = getelementptr inbounds i8, ptr %cpu, i64 {off}"
        );
    }
    out.push_str(rest);
    out.push_str("}\n\n");
    intrinsics.extend(e.intrinsics);
    callees.extend(e.callees);
}

/// The dispatcher: look the guest pc up in the function table and continue there as a tail call,
/// or take the slow path. A proper tail call keeps `BR`-driven loops from growing the stack.
pub fn dispatcher() -> String {
    let pc = nx_runtime::a64::state::offsets::PC;
    format!(
        "define hidden void @nx_dispatch(ptr %cpu) nounwind {{\n\
         entry:\n\
         \x20 %pcp = getelementptr inbounds i8, ptr %cpu, i64 {pc}\n\
         \x20 %pc = load i64, ptr %pcp\n\
         \x20 %fn = call ptr @nx_lookup(i64 %pc)\n\
         \x20 %none = icmp eq ptr %fn, null\n\
         \x20 br i1 %none, label %slow, label %fast\n\
         fast:\n\
         \x20 musttail call void %fn(ptr %cpu)\n\
         \x20 ret void\n\
         slow:\n\
         \x20 musttail call void @nx_slow_path(ptr %cpu)\n\
         \x20 ret void\n\
         }}\n\n"
    )
}
