//! Control flow of a Maxwell program: the basic blocks reachable from the first instruction.
//!
//! Maxwell has no structured control flow. A conditional branch is a `BRA` with a predicate guard;
//! where the paths meet again is declared in advance by `SSY target` and each path ends in `SYNC`,
//! which jumps to the innermost target that `SSY` pushed. `PBK`/`BRK` do the same for the exit of
//! a loop, `PCNT`/`CONT` for its continue point. A thread keeps these targets on small stacks of
//! its own, so the targets of `SYNC`, `BRK` and `CONT` are found here by following each path with
//! the stacks it has built up. A block reached on two paths with different stacks is reported.
//!
//! `BRX` jumps to a computed address: the targets are found by reading the table of offsets the
//! program loads its register from out of a constant buffer (docs/ASSUMPTIONS.md A-155).
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

/// A word of a constant buffer that was read to find where an indirect branch goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableWord {
    pub slot: u32,
    pub offset: u32,
    pub value: u32,
}

/// Words of the constant buffers, for a program whose control flow depends on them (the jump
/// table of an indirect branch).
pub trait Constants {
    /// The word at byte `offset` of constant buffer `slot`, if it can be read.
    fn word(&self, slot: u32, offset: u32) -> Option<u32>;
}

/// No constant buffer can be read.
impl Constants for () {
    fn word(&self, _slot: u32, _offset: u32) -> Option<u32> {
        None
    }
}

/// Words that were read before (see [`Cfg::tables`]).
impl Constants for [TableWord] {
    fn word(&self, slot: u32, offset: u32) -> Option<u32> {
        self.iter()
            .find(|w| w.slot == slot && w.offset == offset)
            .map(|w| w.value)
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
    /// Jump to `target` when `guard` holds and the condition code passes `cc` (a test of the flags
    /// the `.CC` instructions set; [`CC_ALWAYS`] when there is none); otherwise continue at `next`
    /// (absent when the jump is unconditional).
    Jump {
        kind: JumpKind,
        guard: Guard,
        cc: u8,
        target: u32,
        next: Option<u32>,
    },
    /// Jump, when `guard` holds and the condition code passes `cc`, to the address that register
    /// `reg` holds plus `base` (the block's `targets` are the places it can be); otherwise continue
    /// at `next`.
    Indirect {
        guard: Guard,
        cc: u8,
        reg: u8,
        base: u32,
        next: Option<u32>,
    },
    /// End the invocation when `guard` holds and the condition code passes `cc`.
    Exit {
        guard: Guard,
        cc: u8,
        next: Option<u32>,
    },
    /// Discard the fragment when `guard` holds and the condition code passes `cc`.
    Kill {
        guard: Guard,
        cc: u8,
        next: Option<u32>,
    },
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
    /// The places an indirect jump that ends the block can go to: the address it computes for
    /// each entry of its table, and the block that starts there (an address that is the place of a
    /// scheduling word is the block after it).
    pub targets: Vec<(u32, u32)>,
}

#[derive(Debug, Default)]
pub struct Cfg {
    pub entry: u32,
    pub blocks: BTreeMap<u32, Block>,
    /// Everything the discovery could not make sense of; the program is not translatable when any.
    pub issues: Vec<String>,
    /// Bytes of code that are reachable, counting from the start of the code.
    pub extent: u32,
    /// The words of constant buffers the discovery read to find the targets of indirect jumps:
    /// the graph is right only while they hold these values.
    pub tables: Vec<TableWord>,
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
    /// The graph of a program that has no indirect jump.
    pub fn discover(code: Code<'_>) -> Cfg {
        Cfg::discover_with(code, &())
    }

    /// The graph of a program, finding the targets of its indirect jumps in `constants`.
    pub fn discover_with(code: Code<'_>, constants: &(impl Constants + ?Sized)) -> Cfg {
        let mut cfg = Cfg {
            entry: ENTRY,
            ..Cfg::default()
        };
        let mut indirect: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
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
                let op = inst.op();
                let cc = cc_of(inst);
                // An instruction that never executes does nothing.
                if guard.is_never() || cc == CC_NEVER {
                    pc = next;
                    continue;
                }
                // Nothing continues past an instruction that always takes effect.
                let always = guard.is_always() && cc == CC_ALWAYS;
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
                        let Some(t) = relative_target(code, pc, inst) else {
                            cfg.issues
                                .push(format!("BRA at {pc:#x} to a place outside the code"));
                            break;
                        };
                        leaders.insert(t);
                        work.push((t, st.clone()));
                        if always {
                            break;
                        }
                        leaders.insert(next);
                        pc = next;
                    }
                    Some(Op::Sync | Op::Brk | Op::Cont) => {
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
                        if always {
                            break;
                        }
                        leaders.insert(next);
                        pc = next;
                    }
                    Some(Op::Brx) => {
                        let (targets, words) = match indirect_targets(code, pc, inst, constants) {
                            Ok(found) => found,
                            Err(issue) => {
                                cfg.issues.push(issue);
                                break;
                            }
                        };
                        for word in words {
                            if !cfg.tables.contains(&word) {
                                cfg.tables.push(word);
                            }
                        }
                        for (_, t) in &targets {
                            leaders.insert(*t);
                            work.push((*t, st.clone()));
                        }
                        indirect.insert(pc, targets);
                        if always {
                            break;
                        }
                        leaders.insert(next);
                        pc = next;
                    }
                    Some(Op::Exit | Op::Kil) => {
                        if always {
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
                let cc = cc_of(inst);
                let ends = |term: Term| (Some(term), next);
                // The block goes on after an instruction that does not always take effect.
                let after_it = (!(guard.is_always() && cc == CC_ALWAYS)).then_some(next);
                let (term, end) = if guard.is_never() || cc == CC_NEVER {
                    (None, next)
                } else {
                    match op {
                        Some(Op::Bra) => {
                            let t = relative_target(code, pc, inst);
                            match t {
                                Some(target) => ends(Term::Jump {
                                    kind: JumpKind::Bra,
                                    guard,
                                    cc,
                                    target,
                                    next: after_it,
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
                                Some(target) if valid_target(code, target) => ends(Term::Jump {
                                    kind,
                                    guard,
                                    cc,
                                    target,
                                    next: after_it,
                                }),
                                _ => ends(Term::Unsupported),
                            }
                        }
                        Some(Op::Brx) if indirect.contains_key(&pc) => {
                            let base = (i64::from(pc) + 8 + inst.sbits(20, 24)) as u32;
                            ends(Term::Indirect {
                                guard,
                                cc,
                                reg: inst.ra(),
                                base,
                                next: after_it,
                            })
                        }
                        Some(Op::Brx) => ends(Term::Unsupported),
                        Some(Op::Exit) => ends(Term::Exit {
                            guard,
                            cc,
                            next: after_it,
                        }),
                        Some(Op::Kil) => ends(Term::Kill {
                            guard,
                            cc,
                            next: after_it,
                        }),
                        Some(op) if is_unsupported_flow(op) => ends(Term::Unsupported),
                        None => ends(Term::Unsupported),
                        Some(_) => (None, next),
                    }
                };
                let falls_into_leader = term.is_none() && leaders.contains(&next);
                if let Some(term) = term {
                    cfg.finish(start, end, term, stacks);
                    if let Some(targets) = indirect.get(&pc) {
                        if let Some(block) = cfg.blocks.get_mut(&start) {
                            block.targets = targets.clone();
                        }
                    }
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
                targets: Vec::new(),
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

/// The condition code test that always passes (`T`).
pub const CC_ALWAYS: u8 = 15;
/// The condition code test that never does (`F`).
pub const CC_NEVER: u8 = 0;

/// The condition code test of a control-flow instruction: the low five bits of `BRA`, `SYNC`,
/// `BRK`, `CONT`, `BRX`, `EXIT` and `KIL` (what the tests are is in `translate`, docs/ASSUMPTIONS.md
/// A-107). The other instructions do not have one.
fn cc_of(inst: Inst) -> u8 {
    match inst.op() {
        Some(Op::Bra | Op::Sync | Op::Brk | Op::Cont | Op::Brx | Op::Exit | Op::Kil) => {
            inst.bits(0, 5) as u8
        }
        _ => CC_ALWAYS,
    }
}

/// What reading a jump table gives: the address each entry computes with the block that starts
/// there, and the words that were read.
type JumpTable = (Vec<(u32, u32)>, Vec<TableWord>);

/// The places the `BRX` at `pc` can go to, and the words of the constant buffer that say so.
///
/// The compiler's pattern, looked for in the straight-line code before the jump: the register is
/// loaded from a table with `LDC`, at an address that is another register shifted left by two, and
/// that register was clamped to the table's size with an unsigned `IMNMX` against an immediate. The
/// address jumped to is the table word plus the jump's own offset field plus the address after the
/// jump instruction, which makes the words offsets from the start of the code.
fn indirect_targets(
    code: Code<'_>,
    pc: u32,
    inst: Inst,
    constants: &(impl Constants + ?Sized),
) -> Result<JumpTable, String> {
    let missing = |what: &str| format!("BRX at {pc:#x}: {what}");
    let base = (i64::from(pc) + 8 + inst.sbits(20, 24)) as u32;
    // (slot, offset of the table, the register of the table index)
    let mut table: Option<(u32, i64, u8)> = None;
    let mut scaled = None;
    let mut bound = None;
    let mut want = inst.ra();
    let mut at = pc;
    for _ in 0..32 {
        let Some(p) = previous_pc(at) else { break };
        at = p;
        let Some(i) = code.get(p) else { break };
        let op = i.op();
        if matches!(
            op,
            Some(
                Op::Bra
                    | Op::Sync
                    | Op::Brk
                    | Op::Cont
                    | Op::Exit
                    | Op::Kil
                    | Op::Brx
                    | Op::Ssy
                    | Op::Pbk
                    | Op::Pcnt
            )
        ) {
            break;
        }
        if i.guard().is_never() {
            continue;
        }
        let writes_it = i.rd() == want && want != crate::inst::RZ;
        match (table, scaled, op) {
            (None, _, Some(Op::Ldc)) if writes_it => {
                if !i.guard().is_always() || i.bits(44, 2) != 0 || i.bits(48, 3) != 4 {
                    return Err(missing("the table is loaded in a way not understood"));
                }
                table = Some((i.bits(36, 5) as u32, i.sbits(20, 16), i.ra()));
                want = i.ra();
            }
            (Some(_), None, Some(Op::ShlImm20)) if writes_it => {
                if !i.guard().is_always() || i.imm20() != 2 {
                    return Err(missing("the table index is scaled in a way not understood"));
                }
                scaled = Some(i.ra());
                want = i.ra();
            }
            (Some(_), Some(_), Some(Op::ImnmxImm20)) if writes_it => {
                // An unsigned minimum (the predicate that chooses it is PT).
                if !i.guard().is_always()
                    || i.bit(48)
                    || i.bits(39, 4) != u64::from(crate::inst::PT)
                {
                    return Err(missing(
                        "the table index is limited in a way not understood",
                    ));
                }
                bound = Some(i.imm20() as u32);
                break;
            }
            _ if writes_it && table.is_some() => {
                return Err(missing(
                    "the table index is computed in a way not understood",
                ));
            }
            _ => {}
        }
    }
    let (Some((slot, first, _)), Some(_), Some(last)) = (table, scaled, bound) else {
        return Err(missing("no jump table was found before it"));
    };
    if last > 4096 || first < 0 {
        return Err(missing(
            "the jump table is not of a size or place understood",
        ));
    }
    let (mut targets, mut words) = (Vec::new(), Vec::new());
    for index in 0..=last {
        let offset = first as u32 + 4 * index;
        let Some(value) = constants.word(slot, offset) else {
            return Err(missing(&format!(
                "the jump table word at c[{slot}][{offset:#x}] cannot be read"
            )));
        };
        words.push(TableWord {
            slot,
            offset,
            value,
        });
        let address = base.wrapping_add(value);
        let mut t = i64::from(address);
        if is_sched_slot(address) {
            t += i64::from(SLOT_BYTES);
        }
        let valid = u32::try_from(t).ok().filter(|t| valid_target(code, *t));
        let Some(t) = valid else {
            return Err(missing(&format!(
                "entry {index} of the jump table leads to {address:#x}, outside the code"
            )));
        };
        if !targets.iter().any(|(a, _)| *a == address) {
            targets.push((address, t));
        }
    }
    Ok((targets, words))
}

/// The instruction slot before `pc`, if there is one.
fn previous_pc(pc: u32) -> Option<u32> {
    let mut p = pc.checked_sub(SLOT_BYTES)?;
    if is_sched_slot(p) {
        p = p.checked_sub(SLOT_BYTES)?;
    }
    (p >= ENTRY).then_some(p)
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

    /// A jump through a table of three entries, as the compiler writes it.
    fn jump_table_program() -> (Vec<u8>, [TableWord; 3]) {
        use crate::ops::Op;
        // Instruction `i` of the assembled code is at this byte offset.
        let at = |i: u32| 8 * (1 + i + i / 3);
        let undo_the_place = ((-(i64::from(at(3)) + 8)) as u64) & 0xFF_FFFF;
        let bytes = assemble(&[
            Enc::op(Op::ImnmxImm20)
                .rd(1)
                .ra(1)
                .imm20(2)
                .pred(39, PT, false)
                .build(),
            Enc::op(Op::ShlImm20).rd(2).ra(1).imm20(2).build(),
            Enc::op(Op::Ldc)
                .rd(3)
                .ra(2)
                .field(36, 5, 1)
                .field(20, 16, 0x40)
                .field(48, 3, 4)
                .build(),
            Enc::op(Op::Brx)
                .ra(3)
                .field(0, 5, 0xF)
                .field(20, 24, undo_the_place)
                .build(),
            mov(),
            exit(ALWAYS),
            mov(),
            exit(ALWAYS),
            mov(),
            exit(ALWAYS),
        ]);
        let word = |i: u32, target: u32| TableWord {
            slot: 1,
            offset: 0x40 + 4 * i,
            value: target,
        };
        (bytes, [word(0, at(4)), word(1, at(6)), word(2, at(8))])
    }

    #[test]
    fn an_indirect_jump_goes_to_the_places_its_table_names() {
        let (bytes, words) = jump_table_program();
        let cfg = Cfg::discover_with(Code(&bytes), &words[..]);
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(cfg.tables, words);
        let entry = &cfg.blocks[&ENTRY];
        assert!(matches!(
            entry.term,
            Term::Indirect {
                reg: 3,
                base: 0,
                next: None,
                ..
            }
        ));
        assert_eq!(entry.targets, vec![(48, 48), (72, 72), (88, 88)]);
        for (_, target) in &entry.targets {
            assert!(matches!(cfg.blocks[target].term, Term::Exit { .. }));
        }
    }

    #[test]
    fn an_indirect_jump_without_its_table_is_reported() {
        let (bytes, words) = jump_table_program();
        let cfg = Cfg::discover(Code(&bytes));
        assert_eq!(cfg.issues.len(), 1);
        assert!(cfg.issues[0].contains("cannot be read"), "{:?}", cfg.issues);
        // A table that leads out of the code is reported too.
        let mut bad = words;
        bad[1].value = 0x10_0000;
        let cfg = Cfg::discover_with(Code(&bytes), &bad[..]);
        assert!(
            cfg.issues[0].contains("outside the code"),
            "{:?}",
            cfg.issues
        );
        // And so is a jump that is not preceded by the pattern.
        let lone = assemble(&[
            Enc::op(crate::ops::Op::Brx).ra(3).field(0, 5, 0xF).build(),
            exit(ALWAYS),
        ]);
        let cfg = Cfg::discover_with(Code(&lone), &words[..]);
        assert!(
            cfg.issues[0].contains("no jump table was found"),
            "{:?}",
            cfg.issues
        );
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
                cc: CC_ALWAYS,
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
                cc: CC_ALWAYS,
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
                    cc: CC_ALWAYS,
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
                cc: CC_ALWAYS,
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
    fn a_test_of_the_condition_code_makes_a_jump_conditional() {
        // `BRA.NEU` (13) with no predicate: taken or not by the flags, so the block goes on after it.
        let at = |slot: u32| slot * 8;
        let slots = [
            (1, with_cc(bra(at(1), at(5), ALWAYS), 13)),
            (2, mov()),
            (3, exit(ALWAYS)),
            (5, exit(ALWAYS)),
        ];
        let bytes = code_of(&slots);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(
            cfg.blocks[&8].term,
            Term::Jump {
                kind: JumpKind::Bra,
                guard: ALWAYS,
                cc: 13,
                target: at(5),
                next: Some(at(2)),
            }
        );
        assert!(
            cfg.blocks.contains_key(&at(2)),
            "the path that does not branch is followed"
        );
        // The same with a predicate, and on an exit and a kill.
        let bytes = assemble(&[
            with_cc(bra(8, 40, P0), 2),
            with_cc(kil(ALWAYS), 5),
            with_cc(exit(ALWAYS), 22),
            exit(ALWAYS),
            mov(),
            exit(ALWAYS),
        ]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert!(matches!(
            cfg.blocks[&8].term,
            Term::Jump {
                guard: P0,
                cc: 2,
                next: Some(16),
                ..
            }
        ));
        assert!(matches!(
            cfg.blocks[&16].term,
            Term::Kill {
                cc: 5,
                next: Some(24),
                ..
            }
        ));
        assert!(matches!(
            cfg.blocks[&24].term,
            Term::Exit {
                cc: 22,
                next: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn a_test_that_never_passes_skips_the_instruction() {
        // `BRA.F` goes nowhere: the target is not followed and the code goes on to the exit.
        let bytes = assemble(&[with_cc(bra(8, 0x4000, ALWAYS), 0), exit(ALWAYS)]);
        let cfg = Cfg::discover(Code(&bytes));
        assert!(cfg.issues.is_empty(), "{:?}", cfg.issues);
        assert_eq!(cfg.blocks.len(), 1);
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
