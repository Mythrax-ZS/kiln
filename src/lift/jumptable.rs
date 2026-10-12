//! Jump-table resolution for `BR Xn`.
//!
//! Compilers lower a dense `switch` to a bounds check, a table lookup and an indirect branch:
//!
//! ```text
//!     cmp   w8, #N            ; index
//!     b.hi  default
//!     adrp  x9, table
//!     add   x9, x9, :lo12:table
//!     ldrsw x8, [x9, x8, lsl #2]
//!     add   x8, x8, x9        ; or an `adr` base, byte or halfword entries, ...
//!     br    x8
//! ```
//!
//! Instead of recognising each spelling, this module *runs* the straight-line code in front of
//! the branch with the interpreter for every index the bounds check admits, reading the table
//! from the image, and collects the addresses the branch register ends up with. The run is
//! repeated with two different fillers for the registers the code does not set, and a result
//! that changes means the target depends on something unknown, so nothing is reported.
//!
//! The answer is only a *candidate set*: the lifter switches on the computed address and sends
//! anything outside the set through the function table. A wrong guess therefore costs speed, never
//! correctness.

use super::discover::CodeSource;
use nx_runtime::a64::bus::{Bus, Fault, FaultKind};
use nx_runtime::a64::inst::{BranchReg, Inst};
use nx_runtime::a64::interp::execute;
use nx_runtime::a64::state::CpuState;
use std::collections::BTreeMap;

/// Largest table accepted.
pub const MAX_ENTRIES: u64 = 4096;
const WINDOW: usize = 40;

/// Read-only memory over the image; stores are ignored so the setup code before the check can run.
struct ImageBus<'a> {
    code: &'a dyn CodeSource,
}

impl Bus for ImageBus<'_> {
    fn read(&mut self, addr: u64, size: u8) -> Result<u128, Fault> {
        match self.code.read(addr, usize::from(size)) {
            Some(v) => Ok(u128::from(v)),
            None => Err(Fault {
                addr,
                write: false,
                kind: FaultKind::Unmapped,
            }),
        }
    }

    fn write(&mut self, _addr: u64, _size: u8, _value: u128) -> Result<(), Fault> {
        Ok(())
    }
}

fn run(
    code: &dyn CodeSource,
    cpu: &mut CpuState,
    insts: &[(u64, Inst)],
    lenient: bool,
) -> Option<()> {
    let mut bus = ImageBus { code };
    for &(pc, inst) in insts {
        if inst.is_branch() {
            continue;
        }
        cpu.pc = pc;
        if execute(cpu, &mut bus, inst).is_err() && !lenient {
            return None;
        }
    }
    Some(())
}

/// Candidate targets of the `BR` at `br_pc` through register `rn`, or `None` when no table is
/// recognised. `inside` says whether an address may be a target (in the function, executable).
pub fn resolve(
    code: &dyn CodeSource,
    insts: &BTreeMap<u64, Inst>,
    br_pc: u64,
    rn: u8,
    inside: &dyn Fn(u64) -> bool,
) -> Option<Vec<u64>> {
    // The straight-line code in front of the branch.
    let mut window: Vec<(u64, Inst)> = Vec::new();
    let mut at = br_pc;
    while window.len() < WINDOW {
        let prev = at.checked_sub(4)?;
        let Some(&inst) = insts.get(&prev) else { break };
        match inst {
            Inst::B { .. }
            | Inst::BranchReg { .. }
            | Inst::Svc { .. }
            | Inst::Brk { .. }
            | Inst::Unallocated(_)
            | Inst::Unsupported { .. } => break,
            _ => {}
        }
        window.push((prev, inst));
        at = prev;
    }
    window.reverse();

    // `b.hi`/`b.hs` to the default case, preceded by `cmp idx, #imm`.
    let jb = window
        .iter()
        .rposition(|&(_, i)| matches!(i, Inst::BCond { cond, .. } if cond == 8 || cond == 2))?;
    let hs = matches!(window[jb].1, Inst::BCond { cond: 2, .. });
    let jc = window[..jb].iter().rposition(|&(_, i)| {
        matches!(
            i,
            Inst::AddSubImm {
                set_flags: true,
                ..
            } | Inst::AddSubShifted {
                set_flags: true,
                ..
            } | Inst::AddSubExtended {
                set_flags: true,
                ..
            } | Inst::AddSubCarry {
                set_flags: true,
                ..
            } | Inst::LogicalImm {
                op: nx_runtime::a64::inst::LogicOp::Ands,
                ..
            } | Inst::LogicalShifted {
                op: nx_runtime::a64::inst::LogicOp::Ands,
                ..
            } | Inst::CondCmp { .. }
        )
    })?;
    let Inst::AddSubImm {
        sub: true,
        set_flags: true,
        rd: 31,
        rn: idx,
        imm,
        ..
    } = window[jc].1
    else {
        return None;
    };
    let last = if hs {
        u64::from(imm).checked_sub(1)?
    } else {
        u64::from(imm)
    };
    if last >= MAX_ENTRIES {
        return None;
    }
    if idx == 31 {
        return None;
    }
    let prefix = &window[..jc];
    let post = &window[jb + 1..];

    let mut results: Vec<Vec<u64>> = Vec::new();
    for fill in [0u64, 0x5A5A_5A5A_5A5A_5A5A] {
        let mut cpu = CpuState {
            sp: fill,
            ..CpuState::default()
        };
        cpu.x = [fill; 31];
        run(code, &mut cpu, prefix, true)?;
        let mut targets = Vec::with_capacity(last as usize + 1);
        for i in 0..=last {
            let mut c = cpu.clone();
            c.x[usize::from(idx)] = i;
            run(code, &mut c, post, false)?;
            targets.push(c.x_zr(rn));
        }
        results.push(targets);
    }
    if results[0] != results[1] {
        return None;
    }
    let mut out = results.swap_remove(0);
    if !out
        .iter()
        .all(|&t| t % 4 == 0 && inside(t) && code.word(t).is_some())
    {
        return None;
    }
    out.sort_unstable();
    out.dedup();
    Some(out)
}

/// Is this instruction the `BR` this module resolves?
pub fn is_br(inst: &Inst) -> Option<u8> {
    match *inst {
        Inst::BranchReg {
            kind: BranchReg::Br,
            rn,
        } => Some(rn),
        _ => None,
    }
}
