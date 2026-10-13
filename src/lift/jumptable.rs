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
use nx_runtime::a64::inst::{BranchReg, Inst, MemKind, MovOp};
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
        let fault = || Fault {
            addr,
            write: false,
            kind: FaultKind::Unmapped,
        };
        let size = usize::from(size);
        if size <= 8 {
            return self.code.read(addr, size).map(u128::from).ok_or_else(fault);
        }
        // A vector register is read as two halves.
        let lo = self.code.read(addr, 8).ok_or_else(fault)?;
        let hi = self.code.read(addr + 8, size - 8).ok_or_else(fault)?;
        Ok(u128::from(lo) | u128::from(hi) << 64)
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
    known: &[(usize, u64)],
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
        for &(r, value) in known {
            cpu.x[r] = value;
        }
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

/// The callee-saved registers (x19 to x28) that the code decoded so far gives one constant value
/// and nothing else, as `(register, value)`: a compiler that switches inside a loop often puts the
/// address of the jump table in such a register once, before the loop, where the straight-line
/// window in front of the `BR` does not reach. Only the instructions that compute the constant
/// (`ADRP` and `ADD`, `ADR`, `MOVZ`) may write the register; loads from the stack, which restore
/// it at the end of the function, are not counted. A value guessed wrong costs speed only: the
/// candidate targets are checked against the function table when the branch runs.
pub fn constant_registers(code: &dyn CodeSource, insts: &BTreeMap<u64, Inst>) -> Vec<(usize, u64)> {
    let mut bus = ImageBus { code };
    let mut distinct = CpuState::default();
    for (r, x) in distinct.x.iter_mut().enumerate() {
        *x = 0x0101_0101_0101_0101u64.wrapping_mul(r as u64 + 1);
    }
    // Who writes each register, in address order.
    let mut writers: [Vec<(u64, Inst)>; 31] = std::array::from_fn(|_| Vec::new());
    for (&pc, &inst) in insts {
        if inst.is_branch() {
            continue;
        }
        let restores = match inst {
            Inst::LdSt { rn, .. } | Inst::LdStPair { rn, .. } => rn == 31 || rn == 29,
            _ => false,
        };
        let mut cpu = distinct.clone();
        cpu.pc = pc;
        let _ = execute(&mut cpu, &mut bus, inst);
        let mut written: Vec<usize> = (19..=28).filter(|&r| cpu.x[r] != distinct.x[r]).collect();
        match inst {
            Inst::LdSt {
                kind: MemKind::Load | MemKind::LoadSigned32 | MemKind::LoadSigned64,
                rt,
                ..
            } if (19..=28).contains(&rt) => written.push(usize::from(rt)),
            Inst::LdStPair {
                load: true,
                rt,
                rt2,
                ..
            } => written.extend(
                [usize::from(rt), usize::from(rt2)]
                    .into_iter()
                    .filter(|r| (19..=28).contains(r)),
            ),
            _ => {}
        }
        written.sort_unstable();
        written.dedup();
        if restores {
            continue;
        }
        for r in written {
            writers[r].push((pc, inst));
        }
    }
    let mut out = Vec::new();
    for (r, list) in writers.iter().enumerate().take(29).skip(19) {
        let value = match list[..] {
            [(
                pc,
                Inst::Adr {
                    page: false, imm, ..
                },
            )] => Some(pc.wrapping_add(imm as u64)),
            [(
                pc,
                Inst::Adr {
                    page: true, imm, ..
                },
            ), (
                _,
                Inst::AddSubImm {
                    sub: false,
                    set_flags: false,
                    rn,
                    imm: add,
                    sf: true,
                    ..
                },
            )] if usize::from(rn) == r => Some(
                (pc & !0xFFF)
                    .wrapping_add(imm as u64)
                    .wrapping_add(u64::from(add)),
            ),
            [(
                pc,
                Inst::Adr {
                    page: true, imm, ..
                },
            )] => Some((pc & !0xFFF).wrapping_add(imm as u64)),
            [(
                _,
                Inst::MovWide {
                    op: MovOp::Movz,
                    imm16,
                    shift,
                    sf: true,
                    ..
                },
            )] => Some(u64::from(imm16) << shift),
            _ => None,
        };
        if let Some(v) = value {
            out.push((r, v));
        }
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Bytes(Vec<u8>);

    impl CodeSource for Bytes {
        fn is_code(&self, _addr: u64) -> bool {
            true
        }

        fn read(&self, addr: u64, size: usize) -> Option<u64> {
            assert!((1..=8).contains(&size), "a source reads 1 to 8 bytes");
            let at = addr as usize;
            let mut v = [0u8; 8];
            v[..size].copy_from_slice(self.0.get(at..at + size)?);
            Some(u64::from_le_bytes(v))
        }
    }

    #[test]
    fn a_vector_sized_read_is_two_reads() {
        let source = Bytes((0u8..32).collect());
        let mut bus = ImageBus { code: &source };
        let expected = u128::from_le_bytes(std::array::from_fn(|i| 8 + i as u8));
        assert_eq!(bus.read(8, 16).unwrap(), expected);
        assert_eq!(bus.read(8, 4).unwrap(), 0x0B0A_0908);
        assert!(bus.read(24, 16).is_err(), "runs past the end");
    }
}
