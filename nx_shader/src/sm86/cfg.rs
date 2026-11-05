//! Control flow of an sm_86 program: the basic blocks reachable from its first instruction, in the
//! shape [`crate::cfg::Cfg`] gives the Maxwell ones, so that the same translation of blocks into a
//! loop (or a predicated sequence) serves both.
//!
//! What a thread does, which is all a translation that runs every invocation on its own needs:
//!
//! * `BRA target` under its guard and its own predicate (bits 87..90) jumps; `EXIT` and `KILL`
//!   end the thread (or the fragment) under the same two.
//! * The convergence barriers (`BSSY`, `BSYNC`, `BREAK`, `WARPSYNC`) only gather the threads of a
//!   warp again: for one thread they do nothing, and so are plain instructions here.
//! * A call is a jump: the program puts the return address in a register itself (a `MOV` of the
//!   offset after the `CALL`) and `RET Rn` goes to that register plus the relative base its offset
//!   field gives (the start of the code). A `RET` is therefore an indirect jump whose targets are
//!   the places after the program's calls. `CALL` to the instruction after next is how NVN2's
//!   compiler skips one instruction under a predicate (`@P CALL +2; BRA X`): a conditional jump
//!   (docs/ASSUMPTIONS.md A-241).
//! * `BRX Rn` jumps to `Rn` plus its base: the targets are found by reading the jump table the
//!   program loads `Rn` from, out of a constant buffer (the pattern NeXium recognises: an `LDC`
//!   indexed by a register scaled by four and clamped with an unsigned `IMNMX`).
//!
//! Offsets are bytes from the start of the code (the first instruction is at 0).

use super::{code_extent, op, Inst, BRANCH_PRED, INST_BYTES, RZ};
use crate::cfg::{Block, Cfg, Constants, JumpKind, Stacks, TableWord, Term, CC_ALWAYS};
use crate::inst::Guard;
use std::collections::{BTreeMap, BTreeSet};

/// The most entries a jump table may have.
const MAX_TABLE: u32 = 256;

/// How an instruction affects the control flow.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Flow {
    /// It does not.
    None,
    Jump {
        target: u32,
    },
    Exit,
    Kill,
    /// An indirect jump to `reg` plus `base`; `targets` are the (address, block) pairs it can go to.
    Indirect {
        reg: u8,
        base: u32,
        targets: Vec<(u32, u32)>,
    },
}

/// The condition under which a terminator takes effect: its guard and its own predicate operand.
fn condition(inst: Inst) -> (Guard, Guard) {
    (inst.guard(), inst.pred(87, 90, false))
}

/// The two conditions as one guard: either one alone, or [`BRANCH_PRED`] when both matter (the
/// translation of the block computes it, see [`needs_branch_predicate`]).
fn combined(inst: Inst) -> Guard {
    let (g, p) = condition(inst);
    match (g.is_always(), p.is_always()) {
        (_, true) => g,
        (true, false) => p,
        (false, false) => Guard {
            index: BRANCH_PRED,
            negated: false,
        },
    }
}

/// Does the terminator `inst` need [`BRANCH_PRED`] computed before it (both of its conditions
/// matter)?
pub fn needs_branch_predicate(inst: Inst) -> bool {
    let (g, p) = condition(inst);
    !g.is_always() && !p.is_always()
}

/// Is this a call that only skips the next instruction, a `BRA` (`CALL` to the instruction after
/// next)?
fn is_skip(code: &[u8], inst: Inst, pc: u32) -> bool {
    inst.opcode() == op::CALL
        && inst.branch_target(pc) == i64::from(pc + 2 * INST_BYTES)
        && Inst::at(code, pc + INST_BYTES).is_some_and(|next| next.opcode() == op::BRA)
}

/// The places after the program's calls: where its `RET`s go back to.
fn return_sites(code: &[u8], extent: u32) -> Vec<u32> {
    (0..extent)
        .step_by(INST_BYTES as usize)
        .filter(|&pc| {
            Inst::at(code, pc).is_some_and(|i| i.opcode() == op::CALL && !is_skip(code, i, pc))
        })
        .map(|pc| pc + INST_BYTES)
        .collect()
}

/// The control flow of the instruction at `pc`, or why it cannot be followed.
fn flow(
    code: &[u8],
    extent: u32,
    pc: u32,
    inst: Inst,
    returns: &[u32],
    constants: &(impl Constants + ?Sized),
    tables: &mut Vec<TableWord>,
) -> Result<Flow, String> {
    let target = |inst: Inst| -> Result<u32, String> {
        let t = inst.branch_target(pc);
        if t < 0 || t >= i64::from(extent) || t % i64::from(INST_BYTES) != 0 {
            return Err(format!(
                "{} at {pc:#x} to {t:#x}, outside the code",
                super::name(inst)
            ));
        }
        Ok(t as u32)
    };
    Ok(match inst.opcode() {
        op::BRA if inst.is_end(pc) => Flow::Exit,
        op::BRA => Flow::Jump {
            target: target(inst)?,
        },
        op::CALL => Flow::Jump {
            target: target(inst)?,
        },
        op::EXIT => Flow::Exit,
        op::KILL => Flow::Kill,
        // The address the jump computes (register plus base) is the place it goes to.
        op::RET => Flow::Indirect {
            reg: inst.reg(24),
            base: inst.branch_target(pc) as u32,
            targets: returns.iter().map(|&r| (r, r)).collect(),
        },
        op::BRX => {
            let base = inst.branch_target(pc);
            let reg = inst.reg(24);
            let found = jump_table(code, pc, reg, constants)?;
            let mut targets = Vec::new();
            for (word, value) in found {
                tables.push(word);
                let t = base.wrapping_add(i64::from(value));
                if t < 0 || t >= i64::from(extent) || t % i64::from(INST_BYTES) != 0 {
                    return Err(format!(
                        "the jump table of BRX at {pc:#x} names {t:#x}, outside the code"
                    ));
                }
                let t = t as u32;
                if !targets.iter().any(|&(a, _)| a == t) {
                    targets.push((t, t));
                }
            }
            Flow::Indirect {
                reg,
                base: base as u32,
                targets,
            }
        }
        _ => Flow::None,
    })
}

/// The instruction before `pc` (in the same straight run of code) that last wrote general register
/// `reg`, unguarded.
fn writer(code: &[u8], pc: u32, reg: u8) -> Option<(u32, Inst)> {
    let mut at = pc;
    while at >= INST_BYTES {
        at -= INST_BYTES;
        let i = Inst::at(code, at)?;
        if matches!(
            i.opcode(),
            op::BRA | op::CALL | op::RET | op::EXIT | op::BRX | op::KILL
        ) {
            return None;
        }
        if i.dst() == reg && !i.is_uniform() {
            return i.guard().is_always().then_some((at, i));
        }
    }
    None
}

/// The words of the jump table of the `BRX` at `pc` (which jumps through `reg`): each entry's
/// constant buffer word and its value.
fn jump_table(
    code: &[u8],
    pc: u32,
    reg: u8,
    constants: &(impl Constants + ?Sized),
) -> Result<Vec<(TableWord, u32)>, String> {
    let no = |why: &str| format!("the jump table of BRX at {pc:#x}: {why}");
    let (at, ldc) = writer(code, pc, reg).ok_or_else(|| no("no LDC loads its register"))?;
    if ldc.opcode() != op::LDC || ldc.field(73, 3) != 4 || ldc.field(78, 2) != 0 || ldc.bit(91) {
        return Err(no("its register does not come from a 32-bit LDC"));
    }
    let (slot, offset) = (ldc.field(54, 5) as u32, ldc.field(38, 16) as u32);
    let (at, scale) =
        writer(code, at, ldc.reg(24)).ok_or_else(|| no("nothing scales the LDC's index"))?;
    let scaled = match scale.opcode() {
        // IMAD Rd, Ra, 4, RZ
        0x824 => scale.field(32, 32) == 4 && scale.reg(64) == RZ,
        // SHF.L.U32 Rd, Ra, 2, RZ
        0x819 => {
            scale.field(32, 32) == 2 && scale.reg(64) == RZ && !scale.bit(76) && !scale.bit(80)
        }
        _ => false,
    };
    if !scaled {
        return Err(no("the LDC's index is not a register times four"));
    }
    let (_, clamp) =
        writer(code, at, scale.reg(24)).ok_or_else(|| no("nothing clamps the table index"))?;
    // IMNMX.U32 Rd, Ra, limit, PT: the minimum of the index and the last entry.
    let limit = clamp.field(32, 32) as u32;
    if clamp.opcode() != 0x817
        || !clamp.pred(87, 90, false).is_always()
        || clamp.bit(73)
        || limit >= MAX_TABLE
    {
        return Err(no("the table index is not clamped by an unsigned IMNMX"));
    }
    let mut words = Vec::new();
    for entry in 0..=limit {
        let off = offset + 4 * entry;
        let Some(value) = constants.word(slot, off) else {
            return Err(no(&format!(
                "constant buffer {slot} cannot be read at {off:#x}"
            )));
        };
        words.push((
            TableWord {
                slot,
                offset: off,
                value,
            },
            value,
        ));
    }
    Ok(words)
}

/// The graph of a program whose code is `code` (the bytes after the header), reading the jump
/// tables of indirect branches from `constants`.
pub fn discover(code: &[u8], constants: &(impl Constants + ?Sized)) -> Cfg {
    let mut cfg = Cfg {
        entry: 0,
        ..Cfg::default()
    };
    let Some(extent) = code_extent(code) else {
        cfg.issues
            .push("the code has no end (a branch to itself after the last EXIT)".to_string());
        return cfg;
    };
    cfg.extent = extent;
    let returns = return_sites(code, extent);
    // Phase 1: follow every path, noting where blocks start and how each terminator flows.
    let mut leaders: BTreeSet<u32> = BTreeSet::from([0]);
    let mut flows: BTreeMap<u32, Flow> = BTreeMap::new();
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut work = vec![0u32];
    let mut tables = Vec::new();
    while let Some(start) = work.pop() {
        let mut pc = start;
        loop {
            if !seen.insert(pc) {
                break;
            }
            let Some(inst) = Inst::at(code, pc).filter(|_| pc < extent) else {
                cfg.issues
                    .push(format!("the code ends at {extent:#x} without an exit"));
                break;
            };
            let next = pc + INST_BYTES;
            if inst.guard().is_never() {
                pc = next;
                continue;
            }
            let f = match flow(code, extent, pc, inst, &returns, constants, &mut tables) {
                Ok(f) => f,
                Err(why) => {
                    cfg.issues.push(why);
                    break;
                }
            };
            let always = combined(inst).is_always() && !needs_branch_predicate(inst);
            match &f {
                Flow::None => {
                    pc = next;
                    continue;
                }
                Flow::Jump { target } => {
                    leaders.insert(*target);
                    work.push(*target);
                }
                Flow::Indirect { targets, .. } => {
                    for &(_, t) in targets {
                        leaders.insert(t);
                        work.push(t);
                    }
                }
                Flow::Exit | Flow::Kill => {}
            }
            flows.insert(pc, f);
            if always {
                break;
            }
            leaders.insert(next);
            pc = next;
        }
    }
    cfg.tables = tables;
    // Phase 2: the blocks, each from a leader to its terminator or the next leader.
    let leader_list: Vec<u32> = leaders
        .iter()
        .copied()
        .filter(|l| seen.contains(l))
        .collect();
    for (k, &start) in leader_list.iter().enumerate() {
        let limit = leader_list.get(k + 1).copied().unwrap_or(extent);
        let mut pc = start;
        let mut block = Block {
            start,
            end: limit,
            term: Term::Fall(limit),
            stacks: Stacks::default(),
            targets: Vec::new(),
        };
        while pc < limit {
            let next = pc + INST_BYTES;
            if let Some(f) = flows.get(&pc) {
                let inst = Inst::at(code, pc).expect("followed before");
                let guard = combined(inst);
                let always = guard.is_always();
                let after = (!always).then_some(next);
                block.end = next;
                block.term = match f {
                    Flow::Jump { target } => Term::Jump {
                        kind: JumpKind::Bra,
                        guard,
                        cc: CC_ALWAYS,
                        target: *target,
                        next: after,
                    },
                    Flow::Exit => Term::Exit {
                        guard,
                        cc: CC_ALWAYS,
                        next: after,
                    },
                    Flow::Kill => Term::Kill {
                        guard,
                        cc: CC_ALWAYS,
                        next: after,
                    },
                    Flow::Indirect { reg, base, targets } => {
                        block.targets = targets.clone();
                        Term::Indirect {
                            guard,
                            cc: CC_ALWAYS,
                            reg: *reg,
                            base: *base,
                            next: after,
                        }
                    }
                    Flow::None => unreachable!("only terminators are kept"),
                };
                break;
            }
            pc = next;
        }
        cfg.blocks.insert(start, block);
    }
    cfg
}

/// The instructions of `block` in order, with their offsets.
pub fn instructions<'a>(code: &'a [u8], block: &Block) -> impl Iterator<Item = (u32, Inst)> + 'a {
    (block.start..block.end)
        .step_by(INST_BYTES as usize)
        .filter_map(move |pc| Inst::at(code, pc).map(|i| (pc, i)))
}

/// The terminator of `block`, when it ends with one.
pub fn terminator(code: &[u8], block: &Block) -> Option<(u32, Inst)> {
    if matches!(block.term, Term::Fall(_)) {
        return None;
    }
    let pc = block.end - INST_BYTES;
    Inst::at(code, pc).map(|i| (pc, i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm86::tests::word;

    fn code(insts: &[Inst]) -> Vec<u8> {
        insts.iter().flat_map(|i| i.0.to_le_bytes()).collect()
    }

    fn bra(pc: u32, to: u32, guard: u8, negated: bool) -> Inst {
        let off = (i64::from(to) - i64::from(pc) - 16) / 4;
        word(&[
            (0, 12, 0x947),
            (12, 3, u64::from(guard)),
            (15, 1, u64::from(negated)),
            (87, 3, 7),
            (34, 48, off as u64),
        ])
    }

    fn exit() -> Inst {
        word(&[(0, 12, 0x94d), (12, 3, 7), (87, 3, 7)])
    }

    fn nop() -> Inst {
        word(&[(0, 12, 0x918), (12, 3, 7)])
    }

    #[test]
    fn a_conditional_branch_splits_the_code_into_blocks() {
        // 0: @P0 BRA 0x30; 0x10: NOP; 0x20: EXIT; 0x30: EXIT; 0x40: BRA self.
        let c = code(&[
            bra(0, 0x30, 0, false),
            nop(),
            exit(),
            exit(),
            bra(0x40, 0x40, 7, false),
        ]);
        let g = discover(&c, &());
        assert!(g.issues.is_empty(), "{:?}", g.issues);
        assert_eq!(g.extent, 0x50);
        assert_eq!(
            g.blocks.keys().copied().collect::<Vec<_>>(),
            [0, 0x10, 0x30]
        );
        assert!(matches!(
            g.blocks[&0].term,
            Term::Jump {
                target: 0x30,
                next: Some(0x10),
                ..
            }
        ));
        assert!(matches!(
            g.blocks[&0x10].term,
            Term::Exit { next: None, .. }
        ));
    }

    #[test]
    fn a_call_and_its_return_are_a_jump_and_an_indirect_jump_back() {
        // 0: MOV R9, 0x20 (the return address); 0x10: CALL 0x30; 0x20: EXIT; 0x30: RET R9 (base:
        // the start of the code); 0x40: BRA self (the callees come before the end of the code).
        let mov = word(&[
            (0, 12, 0x802),
            (12, 3, 7),
            (16, 8, 9),
            (32, 32, 0x20),
            (72, 4, 0xf),
        ]);
        let call = word(&[
            (0, 12, 0x944),
            (12, 3, 7),
            (87, 3, 7),
            (34, 48, (0x30 - 0x20) / 4),
        ]);
        let ret_off = (0i64 - 0x30 - 16) / 4;
        let ret = word(&[
            (0, 12, 0x950),
            (12, 3, 7),
            (87, 3, 7),
            (24, 8, 9),
            (34, 48, ret_off as u64),
        ]);
        let c = code(&[mov, call, exit(), ret, bra(0x40, 0x40, 7, false)]);
        let g = discover(&c, &());
        assert!(g.issues.is_empty(), "{:?}", g.issues);
        assert!(matches!(
            g.blocks[&0].term,
            Term::Jump {
                target: 0x30,
                next: None,
                ..
            }
        ));
        let r = &g.blocks[&0x30];
        assert!(matches!(
            r.term,
            Term::Indirect {
                reg: 9,
                base: 0,
                ..
            }
        ));
        assert_eq!(r.targets, vec![(0x20, 0x20)]);
        assert!(g.blocks.contains_key(&0x20));
    }

    #[test]
    fn a_branch_with_two_conditions_is_taken_on_both() {
        // @P1 BRA !P2: the guard and the branch's own predicate.
        let mut b = bra(0, 0x20, 1, false);
        b = Inst(b.0 & !(0xFu128 << 87) | (2u128 << 87) | (1u128 << 90));
        let c = code(&[b, exit(), exit(), bra(0x30, 0x30, 7, false)]);
        let g = discover(&c, &());
        assert!(needs_branch_predicate(b));
        assert!(matches!(
            g.blocks[&0].term,
            Term::Jump {
                guard: Guard {
                    index: BRANCH_PRED,
                    negated: false
                },
                ..
            }
        ));
    }

    #[test]
    fn code_without_its_end_marker_is_refused() {
        let g = discover(&code(&[exit()]), &());
        assert!(!g.issues.is_empty());
    }
}
