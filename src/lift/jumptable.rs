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
use nx_runtime::a64::inst::{BranchReg, Inst, MemKind};
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
        sf,
        sub: true,
        set_flags: true,
        rd: 31,
        rn: idx,
        imm,
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
    let wmask = if sf { u64::MAX } else { 0xFFFF_FFFF };
    let fresh = |fill: u64| {
        let mut cpu = CpuState {
            sp: fill,
            ..CpuState::default()
        };
        cpu.x = [fill; 31];
        cpu
    };
    let observed = usize::from(idx);

    // Where does the compared register get its value? It may be a copy or adjustment of a
    // register (`mov w8, w0 ; cmp w0, #5`, `sub w9, w8, #3 ; cmp w9, #3`) or be loaded from memory.
    // Try injecting a value just after each instruction of the code in front of the compare, into
    // each register that instruction wrote (and, first of all, into every register before the
    // first instruction). Two probes show how the compared register moves with the injected
    // value; if it moves by one per step, the value that makes it equal each index can be
    // solved for, and the rest of the code is run with it.
    let attempt = |pos: usize, var: usize| -> Option<Vec<Vec<u64>>> {
        let mut results = Vec::new();
        for fill in [0u64, 0x5A5A_5A5A_5A5A_5A5A] {
            let start = |v0: u64| -> Option<CpuState> {
                let mut c = fresh(fill);
                run(code, &mut c, &prefix[..pos], true)?;
                c.x[var] = v0;
                run(code, &mut c, &prefix[pos..], true)?;
                Some(c)
            };
            let c0 = start(0)?.x[observed] & wmask;
            let c1 = start(1)?.x[observed] & wmask;
            let step = c1.wrapping_sub(c0) & wmask;
            let mut targets = Vec::with_capacity(last as usize + 1);
            for i in 0..=last {
                let v0 = if step == 1 {
                    i.wrapping_sub(c0) & wmask
                } else if step == wmask {
                    c0.wrapping_sub(i) & wmask
                } else {
                    return None;
                };
                let mut c = start(v0)?;
                if c.x[observed] & wmask != i {
                    return None;
                }
                run(code, &mut c, post, false)?;
                targets.push(c.x_zr(rn));
            }
            results.push(targets);
        }
        (results[0] == results[1]).then_some(results)
    };
    // The registers each instruction of the prefix changes, seen from a state where every
    // register holds a different value.
    let mut distinct = CpuState::default();
    for (r, x) in distinct.x.iter_mut().enumerate() {
        *x = 0x0101_0101_0101_0101u64.wrapping_mul(r as u64 + 1);
    }
    let mut changed: Vec<Vec<usize>> = vec![(0..31).collect()];
    for k in 0..prefix.len() {
        let before = distinct.clone();
        run(code, &mut distinct, &prefix[k..=k], true)?;
        let mut regs: Vec<usize> = (0..31).filter(|&r| distinct.x[r] != before.x[r]).collect();
        // A load from memory the image does not have changes nothing here, but it still defines
        // its destination: the value being switched on may be loaded.
        match prefix[k].1 {
            Inst::LdSt {
                kind: MemKind::Load | MemKind::LoadSigned32 | MemKind::LoadSigned64,
                rt,
                ..
            } if rt < 31 => regs.push(usize::from(rt)),
            Inst::LdStPair {
                load: true,
                rt,
                rt2,
                ..
            } => regs.extend(
                [usize::from(rt), usize::from(rt2)]
                    .into_iter()
                    .filter(|&r| r < 31),
            ),
            _ => {}
        }
        regs.sort_unstable();
        regs.dedup();
        changed.push(regs);
    }
    let mut results = (0..=prefix.len())
        .rev()
        .find_map(|pos| changed[pos].iter().find_map(|&var| attempt(pos, var)))?;
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
