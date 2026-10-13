//! Control flow of a Maxwell program: the basic blocks reachable from the first instruction.
//!
//! Maxwell has no structured control flow. A conditional branch is a `BRA` with a predicate guard;
//! where the paths meet again is declared in advance by `SSY target` and each path ends in `SYNC`,
//! which jumps to the innermost target that `SSY` pushed. `PBK`/`BRK` do the same for the exit of
//! a loop, `PCNT`/`CONT` for its continue point. A thread keeps these targets on small stacks of
//! its own, so the targets of `SYNC`, `BRK` and `CONT` are found here by following each path with
//! the stacks it has built up. A block reached on two paths with different stacks is reported.
//!
//! Offsets are bytes from the start of the code, which is where the first scheduling word sits;
//! the first instruction is the next slot. A branch target is the offset of the branch plus 8
//! plus its signed 24-bit offset field.

use crate::inst::{is_sched_slot, Guard, Inst, SLOT_BYTES};
use crate::ops::Op;
use std::collections::{BTreeMap, BTreeSet};

/// The code of a program (the bytes after the header).
#[derive(Clone, Copy)]
pub struct Code<'a>(pub &'a [u8]);

impl Code<'_> {
    pub fn len(&self) -> u32 {
        self.0.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The instruction at a byte offset, if the code reaches that far.
    pub fn get(&self, pc: u32) -> Option<Inst> {
        let at = pc as usize;
        let b = self.0.get(at..at + 8)?;
        Some(Inst(u64::from_le_bytes(b.try_into().ok()?)))
    }
}

/// The targets a path has pushed and not yet taken.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stacks {
    /// `SSY` targets, innermost last.
    pub ssy: Vec<u32>,
    /// `PBK` targets.
    pub pbk: Vec<u32>,
    /// `PCNT` targets.
    pub pcnt: Vec<u32>,
}

/// How a jump finds its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JumpKind {
    Bra,
    Sync,
    Brk,
    Cont,
}

/// How a block ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    /// The next block starts right after this one (it is a branch target).
    Fall(u32),
    /// Jump to `target` when `guard` holds; otherwise continue at `next` (absent when the guard is
    /// always true).
    Jump {
        kind: JumpKind,
        guard: Guard,
        target: u32,
        next: Option<u32>,
    },
    /// End the invocation when `guard` holds.
    Exit { guard: Guard, next: Option<u32> },
    /// Discard the fragment when `guard` holds.
    Kill { guard: Guard, next: Option<u32> },
    /// A kind of control flow not handled (the reason is in [`Cfg::issues`]).
    Unsupported,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub start: u32,
    /// The offset after the last instruction of the block (its terminator, when it has one).
    pub end: u32,
    pub term: Term,
    /// The stacks on entry.
    pub stacks: Stacks,
}

#[derive(Debug, Default)]
pub struct Cfg {
    pub entry: u32,
    pub blocks: BTreeMap<u32, Block>,
    /// Everything the discovery could not make sense of; the program is not translatable when any.
    pub issues: Vec<String>,
    /// Bytes of code that are reachable, counting from the start of the code.
    pub extent: u32,
}

/// The first instruction of a program: the slot after the first scheduling word.
pub const ENTRY: u32 = SLOT_BYTES;

fn next_pc(pc: u32) -> u32 {
    let mut n = pc + SLOT_BYTES;
    if is_sched_slot(n) {
        n += SLOT_BYTES;
    }
    n
}

/// Is the byte offset a place an instruction can be?
fn valid_target(code: Code<'_>, pc: u32) -> bool {
    pc.is_multiple_of(SLOT_BYTES) && !is_sched_slot(pc) && pc < code.len()
}

impl Cfg {
    pub fn discover(code: Code<'_>) -> Cfg {
        let mut cfg = Cfg {
            entry: ENTRY,
            ..Cfg::default()
        };
        // Phase 1: follow every path instruction by instruction with its stacks.
        let mut seen: BTreeMap<u32, Stacks> = BTreeMap::new();
        let mut leaders: BTreeSet<u32> = BTreeSet::new();
        leaders.insert(ENTRY);
        let mut work = vec![(ENTRY, Stacks::default())];
        let mut conflicts = BTreeSet::new();
        while let Some((start, stacks)) = work.pop() {
            let mut pc = start;
            let mut st = stacks;
            loop {
                if let Some(have) = seen.get(&pc) {
                    if *have != st {
                        conflicts.insert(pc);
                    }
                    break;
                }
                let Some(inst) = code.get(pc) else {
                    cfg.issues.push(format!(
                        "the code ends at {:#x} without an exit",
                        code.len()
                    ));
                    break;
                };
                seen.insert(pc, st.clone());
                let next = next_pc(pc);
                let guard = guard_of(inst);
                // An instruction that never executes does nothing.
                if guard.is_never() {
                    pc = next;
                    continue;
                }
                let op = inst.op();
                match op {
                    Some(Op::Ssy | Op::Pbk | Op::Pcnt) => {
                        let Some(t) = relative_target(code, pc, inst) else {
                            cfg.issues.push(format!(
                                "{} at {pc:#x} with a target out of the code",
                                opname(op)
                            ));
                            break;
                        };
                        match op {
                            Some(Op::Ssy) => st.ssy.push(t),
                            Some(Op::Pbk) => st.pbk.push(t),
                            _ => st.pcnt.push(t),
                        }
                        pc = next;
                    }
                    Some(Op::Bra) => {
                        if !condition_is_always(inst) {
                            cfg.issues.push(format!(
                                "BRA at {pc:#x} on a condition code, not a predicate"
                            ));
                            break;
                        }
                        let Some(t) = relative_target(code, pc, inst) else {
                            cfg.issues
                                .push(format!("BRA at {pc:#x} to a place outside the code"));
                            break;
                        };
                        leaders.insert(t);
                        work.push((t, st.clone()));
                        if guard.is_always() {
                            break;
                        }
                        leaders.insert(next);
                        pc = next;
                    }
                    Some(Op::Sync | Op::Brk | Op::Cont) => {
                        if !condition_is_always(inst) {
                            cfg.issues
                                .push(format!("{} at {pc:#x} on a condition code", opname(op)));
                            break;
                        }
                        let mut after = st.clone();
                        let target = match op {
                            Some(Op::Sync) => after.ssy.pop(),
                            Some(Op::Brk) => after.pbk.pop(),
                            _ => after.pcnt.pop(),
                        };
                        let Some(t) = target else {
                            cfg.issues.push(format!(
                                "{} at {pc:#x} with nothing pushed to go to",
                                opname(op)
                            ));
                            break;
                        };
                        if !valid_target(code, t) {
                            cfg.issues.push(format!(
                                "{} at {pc:#x} to {t:#x}, outside the code",
                                opname(op)
                            ));
                            break;
                        }
                        leaders.insert(t);
                        work.push((t, after));
                        if guard.is_always() {
                            break;
                        }
                        leaders.insert(next);
                        pc = next;
                    }
                    Some(Op::Exit | Op::Kil) => {
                        if !condition_is_always(inst) {
                            cfg.issues
                                .push(format!("{} at {pc:#x} on a condition code", opname(op)));
                            break;
                        }
                        if guard.is_always() {
                            break;
                        }
                        leaders.insert(next);
                        pc = next;
                    }
                    Some(op) if is_unsupported_flow(op) => {
                        cfg.issues
                            .push(format!("{} at {pc:#x} is not handled", opname(Some(op))));
                        break;
                    }
                    None => {
                        cfg.issues.push(format!(
                            "instruction {:#018x} at {pc:#x} is not known",
                            inst.0
                        ));
                        break;
                    }
                    Some(_) => pc = next,
                }
            }
        }
        for pc in conflicts {
            cfg.issues.push(format!(
                "the control-flow stacks at {pc:#x} differ between the paths that reach it"
            ));
        }
        // Phase 2: cut the visited instructions into blocks at the leaders.
        let starts: Vec<u32> = leaders
            .iter()
            .copied()
            .filter(|l| seen.contains_key(l))
            .collect();
        for start in starts {
            let stacks = seen[&start].clone();
            let mut pc = start;
            loop {
                let inst = code.get(pc).expect("visited instructions are in the code");
                let next = next_pc(pc);
                let guard = guard_of(inst);
                let op = inst.op();
                let ends = |term: Term| (Some(term), next);
                let (term, end) = if guard.is_never() {
                    (None, next)
                } else {
                    match op {
                        Some(Op::Bra) => {
                            let t = relative_target(code, pc, inst);
                            match t {
                                Some(target) if condition_is_always(inst) => ends(Term::Jump {
                                    kind: JumpKind::Bra,
                                    guard,
                                    target,
                                    next: (!guard.is_always()).then_some(next),
                                }),
                                _ => ends(Term::Unsupported),
                            }
                        }
                        Some(Op::Sync | Op::Brk | Op::Cont) => {
                            let mut after = seen[&pc].clone();
                            let (kind, target) = match op {
                                Some(Op::Sync) => (JumpKind::Sync, after.ssy.pop()),
                                Some(Op::Brk) => (JumpKind::Brk, after.pbk.pop()),
                                _ => (JumpKind::Cont, after.pcnt.pop()),
                            };
                            match target {
                                Some(target)
                                    if condition_is_always(inst) && valid_target(code, target) =>
                                {
                                    ends(Term::Jump {
                                        kind,
                                        guard,
                                        target,
                                        next: (!guard.is_always()).then_some(next),
                                    })
                                }
                                _ => ends(Term::Unsupported),
                            }
                        }
                        Some(Op::Exit) if condition_is_always(inst) => ends(Term::Exit {
                            guard,
                            next: (!guard.is_always()).then_some(next),
                        }),
                        Some(Op::Kil) if condition_is_always(inst) => ends(Term::Kill {
                            guard,
                            next: (!guard.is_always()).then_some(next),
                        }),
                        Some(Op::Exit | Op::Kil) => ends(Term::Unsupported),
                        Some(op) if is_unsupported_flow(op) => ends(Term::Unsupported),
                        None => ends(Term::Unsupported),
                        Some(_) => (None, next),
                    }
                };
                let falls_into_leader = term.is_none() && leaders.contains(&next);
                if let Some(term) = term {
                    cfg.finish(start, end, term, stacks);
                    break;
                }
                if falls_into_leader {
                    cfg.finish(start, next, Term::Fall(next), stacks);
                    break;
                }
                if !seen.contains_key(&next) {
                    // The path stopped here (the code ended, or the instruction was not known).
                    cfg.finish(start, next, Term::Unsupported, stacks);
                    break;
                }
                pc = next;
            }
        }
        cfg
    }

    fn finish(&mut self, start: u32, end: u32, term: Term, stacks: Stacks) {
        self.extent = self.extent.max(end);
        self.blocks.insert(
            start,
            Block {
                start,
                end,
                term,
                stacks,
            },
        );
    }

    /// The instructions of a block in order, with their offsets.
    pub fn instructions<'a>(
        &'a self,
        code: Code<'a>,
        block: &'a Block,
    ) -> impl Iterator<Item = (u32, Inst)> + 'a {
        let mut pc = block.start;
        std::iter::from_fn(move || {
            if pc >= block.end {
                return None;
            }
            let at = pc;
            pc = next_pc(pc);
            Some((at, code.get(at)?))
        })
    }

    /// Every distinct instruction of the reachable code, counted by what it is.
    pub fn histogram(&self, code: Code<'_>) -> BTreeMap<Option<Op>, u64> {
        let mut h = BTreeMap::new();
        for b in self.blocks.values() {
            for (_, inst) in self.instructions(code, b) {
                *h.entry(inst.op()).or_default() += 1;
            }
        }
        h
    }
}

/// The name of an instruction for messages.
fn opname(op: Option<Op>) -> String {
    op.map_or_else(|| "unknown".to_string(), |o| format!("{o:?}"))
}

/// Control flow this module does not follow: calls, returns and indirect jumps.
fn is_unsupported_flow(op: Op) -> bool {
    matches!(
        op,
        Op::Ret
            | Op::Cal
            | Op::CalCbuf
            | Op::Brx
            | Op::BrxCbuf
            | Op::JmpCbuf
            | Op::JmpImm32
            | Op::JmxCbuf
            | Op::JmxImm32
            | Op::JcalCbuf
            | Op::JcalImm32
            | Op::Pret
            | Op::PretCbuf
            | Op::Plongjmp
            | Op::PlongjmpCbuf
            | Op::Longjmp
            | Op::Pexit
    )
}

/// The condition code test of a control-flow instruction must be "always" (15).
fn condition_is_always(inst: Inst) -> bool {
    inst.bits(0, 5) == 0xF
}

/// The target of a relative branch at `pc`, if it lands in the code. A target at the place of a
/// scheduling word is the start of its bundle, and execution goes on with the first instruction
/// slot of it: assemblers give a label at the start of a bundle the address of the scheduling word
/// (docs/ASSUMPTIONS.md, seen in the programs of a real title).
fn relative_target(code: Code<'_>, pc: u32, inst: Inst) -> Option<u32> {
    let mut t = i64::from(pc) + 8 + i64::from(inst.branch_offset());
    if t >= 0 && is_sched_slot(t as u32) {
        t += i64::from(SLOT_BYTES);
    }
    let t = u32::try_from(t).ok()?;
    valid_target(code, t).then_some(t)
}

/// The predicate guard of an instruction. `SSY`, `PBK` and `PCNT` have none: the bits are unused.
fn guard_of(inst: Inst) -> Guard {
    match inst.op() {
        Some(Op::Ssy | Op::Pbk | Op::Pcnt) => Guard::ALWAYS,
        _ => inst.guard(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::*;
    use crate::inst::PT;

    const ALWAYS: Guard = Guard::ALWAYS;

    /// The code from a slot list where slot `n` (n >= 1, n % 4 != 0) is at byte offset 8 n.
    fn code_of(slots: &[(u32, u64)]) -> Vec<u8> {
        let max = slots.iter().map(|(s, _)| *s).max().unwrap_or(1);
        let mut words = vec![0u64; (max as usize + 4) & !3];
        for (slot, w) in slots {
            assert!(slot % 4 != 0, "slot {slot} is a scheduling word");
            words[*slot as usize] = *w;
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn mov() -> u64 {
        mov32i(1, 0)
    }

    #[test]
    fn straight_line_code_is_one_block_ending_in_exit() {
        let bytes = assemble(&[mov(), mov(), exit(ALWAYS), mov(), mov(), mov()]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(cfg.blocks.len(), 1);
        let b = &cfg.blocks[&ENTRY];
        // Three instructions fit the first bundle, the exit is in the second bundle's first slot.
        assert_eq!(b.start, 8);
        assert_eq!(
            b.term,
            Term::Exit {
                guard: ALWAYS,
                next: None
            }
        );
        let n = cfg.instructions(Code(&bytes), b).count();
        assert_eq!(n, 3);
        assert_eq!(cfg.extent, b.end);
    }

    #[test]
    fn a_predicated_branch_makes_a_diamond_closed_by_sync() {
        // slot1: SSY -> slot 9 (join)
        // slot2: @P0 BRA -> slot 6 (else)
        // slot3: MOV (then)      slot 5: SYNC (end of then)
        // slot 6: MOV (else)     slot 7: SYNC
        // slot 9: EXIT
        let at = |slot: u32| slot * 8;
        let slots = [
            (1, ssy(at(1), at(9))),
            (2, bra(at(2), at(6), P0)),
            (3, mov()),
            (5, sync(ALWAYS)),
            (6, mov()),
            (7, sync(ALWAYS)),
            (9, exit(ALWAYS)),
        ];
        let bytes = code_of(&slots);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        let starts: Vec<u32> = cfg.blocks.keys().copied().collect();
        assert_eq!(starts, vec![8, at(3), at(6), at(9)]);
        let head = &cfg.blocks[&8];
        assert_eq!(
            head.term,
            Term::Jump {
                kind: JumpKind::Bra,
                guard: P0,
                target: at(6),
                next: Some(at(3)),
            }
        );
        // The head's SSY was pushed before the branch, so both arms enter with it on the stack.
        assert_eq!(cfg.blocks[&at(3)].stacks.ssy, vec![at(9)]);
        assert_eq!(cfg.blocks[&at(6)].stacks.ssy, vec![at(9)]);
        assert!(cfg.blocks[&at(9)].stacks.ssy.is_empty());
        for arm in [at(3), at(6)] {
            assert_eq!(
                cfg.blocks[&arm].term,
                Term::Jump {
                    kind: JumpKind::Sync,
                    guard: ALWAYS,
                    target: at(9),
                    next: None,
                }
            );
        }
    }

    #[test]
    fn a_loop_is_found_through_pbk_and_brk() {
        // slot1: PBK -> slot 7 (after the loop)
        // slot2: MOV                 <- loop head (slot 2)
        // slot3: @P0 BRK
        // slot5: BRA -> slot 2
        // slot7: EXIT
        let at = |slot: u32| slot * 8;
        let slots = [
            (1, pbk(at(1), at(7))),
            (2, mov()),
            (3, brk(P0)),
            (5, bra(at(5), at(2), ALWAYS)),
            (7, exit(ALWAYS)),
        ];
        let bytes = code_of(&slots);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert!(
            cfg.blocks.contains_key(&at(2)),
            "the loop head starts a block"
        );
        let brk_block = cfg
            .blocks
            .values()
            .find(|b| {
                matches!(
                    b.term,
                    Term::Jump {
                        kind: JumpKind::Brk,
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(
            brk_block.term,
            Term::Jump {
                kind: JumpKind::Brk,
                guard: P0,
                target: at(7),
                next: Some(at(5)),
            }
        );
        // Both entries to the loop head carry the PBK on their stack.
        assert_eq!(cfg.blocks[&at(2)].stacks.pbk, vec![at(7)]);
    }

    #[test]
    fn a_conditional_exit_and_kill_fall_through() {
        let bytes = assemble(&[
            kil(P0),
            exit(Guard {
                index: 1,
                negated: true,
            }),
            exit(ALWAYS),
        ]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(cfg.blocks.len(), 3);
        assert!(matches!(
            cfg.blocks[&8].term,
            Term::Kill { next: Some(16), .. }
        ));
        assert!(matches!(
            cfg.blocks[&16].term,
            Term::Exit { next: Some(24), .. }
        ));
        assert!(matches!(
            cfg.blocks[&24].term,
            Term::Exit { next: None, .. }
        ));
    }

    #[test]
    fn an_instruction_that_never_executes_is_skipped_even_when_it_would_branch() {
        let never = Guard {
            index: PT,
            negated: true,
        };
        let bytes = assemble(&[bra(8, 0x1000, never), exit(ALWAYS)]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(cfg.blocks.len(), 1);
    }

    #[test]
    fn problems_are_reported_not_hidden() {
        // SYNC with nothing pushed.
        let bytes = assemble(&[sync(ALWAYS), exit(ALWAYS)]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(
            cfg.issues.iter().any(|i| i.contains("nothing pushed")),
            "{:?}",
            cfg.issues
        );
        // A branch out of the code.
        let bytes = assemble(&[bra(8, 0x4000, ALWAYS)]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(
            cfg.issues.iter().any(|i| i.contains("outside the code")),
            "{:?}",
            cfg.issues
        );
        // Running off the end.
        let bytes = assemble(&[mov(), mov(), mov()]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(
            cfg.issues.iter().any(|i| i.contains("without an exit")),
            "{:?}",
            cfg.issues
        );
        // A return needs call support.
        let bytes = assemble(&[ret()]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(
            cfg.issues.iter().any(|i| i.contains("Ret")),
            "{:?}",
            cfg.issues
        );
        // A branch to the middle of a slot.
        let bytes = assemble(&[bra(8, 12, ALWAYS), mov(), mov(), mov(), exit(ALWAYS)]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(!cfg.issues.is_empty());
    }

    #[test]
    fn a_target_at_a_scheduling_word_is_the_first_instruction_of_that_bundle() {
        // The branch at slot 1 goes to byte 32, the scheduling word of the second bundle, which is
        // how an assembler labels the start of a bundle; execution continues at byte 40.
        let bytes = assemble(&[bra(8, 32, ALWAYS), mov(), mov(), mov(), exit(ALWAYS)]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(cfg.blocks.keys().copied().collect::<Vec<_>>(), vec![8, 40]);
        assert!(matches!(
            cfg.blocks[&8].term,
            Term::Jump {
                target: 40,
                next: None,
                ..
            }
        ));
    }

    #[test]
    fn the_unused_bits_of_ssy_are_not_a_guard() {
        // A real SSY has zero in bits 16..19; read as a guard that would be @P0. It must still
        // push, and an @!PT in the same place must not make it disappear.
        let at = |slot: u32| slot * 8;
        let mut ssy_word = ssy(at(1), at(9));
        ssy_word &= !(0xF << 16);
        let slots = [
            (1, ssy_word),
            (2, bra(at(2), at(6), P0)),
            (3, sync(ALWAYS)),
            (6, sync(ALWAYS)),
            (9, exit(ALWAYS)),
        ];
        let bytes = code_of(&slots);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(cfg.blocks[&at(3)].stacks.ssy, vec![at(9)]);
        let mut never = ssy_word;
        never |= 0xF << 16;
        let slots = [
            (1, never),
            (2, bra(at(2), at(6), P0)),
            (3, sync(ALWAYS)),
            (6, sync(ALWAYS)),
            (9, exit(ALWAYS)),
        ];
        let bytes = code_of(&slots);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
    }

    #[test]
    fn paths_with_different_stacks_to_one_block_are_a_conflict() {
        // One path pushes an SSY target before reaching slot 6, the other does not.
        let at = |slot: u32| slot * 8;
        let slots = [
            (1, bra(at(1), at(6), P0)),
            (2, ssy(at(2), at(9))),
            (3, bra(at(3), at(6), ALWAYS)),
            (6, sync(ALWAYS)),
            (9, exit(ALWAYS)),
        ];
        let bytes = code_of(&slots);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(
            cfg.issues
                .iter()
                .any(|i| i.contains("differ between the paths")),
            "{:?}",
            cfg.issues
        );
    }

    #[test]
    fn the_histogram_counts_reachable_instructions_by_op() {
        let bytes = assemble(&[mov(), mov(), exit(ALWAYS)]);
        let cfg = Cfg::discover(Code(&bytes));
        let h = cfg.histogram(Code(&bytes));
        assert_eq!(h[&Some(Op::Mov32iImm32)], 2);
        assert_eq!(h[&Some(Op::Exit)], 1);
    }
}
