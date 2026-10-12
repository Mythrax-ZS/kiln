//! Lifting one function: guest blocks to IR blocks, with the terminators the pc protocol needs.
//!
//! The IR control-flow graph is planned first so that every block knows its predecessors before
//! any instruction is lifted; blocks are then lifted in reverse post-order, which seals most of
//! them immediately (see [`super::builder`]).
//!
//! Besides one IR block per guest block the plan has:
//! - the synthetic entry, an origin block that branches to the first guest block, so a loop back
//!   to the entry address does not confuse the loading of the initial state;
//! - a *continuation* block after every call-like terminator (`BL`, `BLR`, `SVC`, an instruction
//!   run by the interpreter), the origin from which the state is reloaded;
//! - the tail-call exit of a conditional branch whose taken side leaves the function;
//! - the default exit of a switch (a `BR` through a jump table), which dispatches through the
//!   function table when the register holds something the table analysis did not list.

use super::builder::Builder;
use super::cfg::{traps, End, FuncCfg, GuestBlock, Target};
use super::inst::Lifter;
use super::ir::*;
use nx_runtime::a64::inst::{BranchReg, Inst};
use std::collections::BTreeMap;

struct Plan {
    guest: BTreeMap<u64, BlockId>,
    cont: BTreeMap<u64, BlockId>,
    /// Exit block for the taken side of a conditional tail call, by guest block.
    tail: BTreeMap<u64, BlockId>,
    /// Default exit of an indirect branch with a table, by guest block.
    default: BTreeMap<u64, BlockId>,
    /// What each IR block is.
    kinds: Vec<BlockKind>,
    succs: Vec<Vec<BlockId>>,
}

#[derive(Clone, Copy)]
enum BlockKind {
    Entry,
    Guest(u64),
    Cont(u64),
    Tail(u64),
    Default(u64),
}

fn last_inst(b: &GuestBlock) -> Option<&(u64, Inst)> {
    b.insts.last()
}

/// The branch target as an IR block, or `None` for a tail call.
fn target_block(plan: &Plan, t: Target) -> Option<BlockId> {
    match t {
        Target::Block(a) => Some(plan.guest[&a]),
        Target::Tail(_) => None,
    }
}

fn make_plan(cfg: &FuncCfg, builder: &mut Builder) -> Plan {
    let mut plan = Plan {
        guest: BTreeMap::new(),
        cont: BTreeMap::new(),
        tail: BTreeMap::new(),
        default: BTreeMap::new(),
        kinds: Vec::new(),
        succs: Vec::new(),
    };
    let entry = builder.add_block(0, true);
    plan.kinds.push(BlockKind::Entry);
    for (&addr, block) in &cfg.blocks {
        let id = builder.add_block(addr, false);
        plan.guest.insert(addr, id);
        plan.kinds.push(BlockKind::Guest(addr));
        match &block.end {
            End::Call { .. } => {
                let pc = last_inst(block).map_or(addr, |&(pc, _)| pc);
                let k = builder.add_block(pc, true);
                plan.cont.insert(addr, k);
                plan.kinds.push(BlockKind::Cont(addr));
            }
            End::Cond {
                taken: Target::Tail(_),
                ..
            } => {
                let t = builder.add_block(addr, false);
                plan.tail.insert(addr, t);
                plan.kinds.push(BlockKind::Tail(addr));
            }
            End::Indirect { targets } if !targets.is_empty() => {
                let d = builder.add_block(addr, false);
                plan.default.insert(addr, d);
                plan.kinds.push(BlockKind::Default(addr));
            }
            _ => {}
        }
    }
    // Successors of every IR block.
    plan.succs = vec![Vec::new(); plan.kinds.len()];
    plan.succs[entry as usize] = vec![plan.guest[&cfg.entry]];
    for (&addr, block) in &cfg.blocks {
        let id = plan.guest[&addr] as usize;
        let mut s = Vec::new();
        match &block.end {
            End::Fall(a) => s.push(plan.guest[a]),
            End::Jump(t) => s.extend(target_block(&plan, *t)),
            End::Cond { taken, fall } => {
                match taken {
                    Target::Block(a) => s.push(plan.guest[a]),
                    Target::Tail(_) => s.push(plan.tail[&addr]),
                }
                s.push(plan.guest[fall]);
            }
            End::Call { ret } => {
                let k = plan.cont[&addr];
                s.push(k);
                plan.succs[k as usize] = vec![plan.guest[ret]];
            }
            End::Indirect { targets } if !targets.is_empty() => {
                s.extend(targets.iter().map(|a| plan.guest[a]));
                s.push(plan.default[&addr]);
            }
            End::Return | End::Trap | End::OutOfCode | End::Indirect { .. } => {}
        }
        s.sort_unstable();
        s.dedup();
        plan.succs[id] = s;
    }
    plan
}

/// Blocks in reverse post-order from the entry.
fn reverse_post_order(plan: &Plan) -> Vec<BlockId> {
    let n = plan.succs.len();
    let mut seen = vec![false; n];
    let mut out = Vec::with_capacity(n);
    let mut stack: Vec<(BlockId, usize)> = vec![(0, 0)];
    seen[0] = true;
    while let Some(&mut (b, ref mut i)) = stack.last_mut() {
        if *i < plan.succs[b as usize].len() {
            let s = plan.succs[b as usize][*i];
            *i += 1;
            if !seen[s as usize] {
                seen[s as usize] = true;
                stack.push((s, 0));
            }
        } else {
            out.push(b);
            stack.pop();
        }
    }
    out.reverse();
    out
}

/// Lift one function. `is_function(addr)` says whether a function is (or will be) generated at
/// `addr`; calls and tail calls to anything else go through the function table.
pub fn lift_function(cfg: &FuncCfg, is_function: &dyn Fn(u64) -> bool) -> Func {
    let mut b = Builder::new(cfg.entry);
    let plan = make_plan(cfg, &mut b);
    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); plan.kinds.len()];
    for (from, succs) in plan.succs.iter().enumerate() {
        for &s in succs {
            preds[s as usize].push(from as BlockId);
        }
    }
    for (id, p) in preds.into_iter().enumerate() {
        b.set_preds(id as BlockId, p);
    }
    b.begin();
    // The register a switch tests, by guest block, for the default exit.
    let mut switch_value: BTreeMap<u64, ValueId> = BTreeMap::new();
    for id in reverse_post_order(&plan) {
        match plan.kinds[id as usize] {
            BlockKind::Entry => {
                let first = plan.guest[&cfg.entry];
                b.set_term(id, Term::Br(first));
            }
            BlockKind::Guest(addr) => {
                lift_guest_block(
                    &mut b,
                    &plan,
                    &cfg.blocks[&addr],
                    id,
                    is_function,
                    &mut switch_value,
                );
            }
            BlockKind::Cont(addr) => {
                let End::Call { ret } = cfg.blocks[&addr].end else {
                    unreachable!("continuation of a call")
                };
                b.set_term(id, Term::Br(plan.guest[&ret]));
            }
            BlockKind::Tail(addr) => {
                let End::Cond {
                    taken: Target::Tail(t),
                    ..
                } = cfg.blocks[&addr].end
                else {
                    unreachable!("tail exit of a conditional branch")
                };
                let spill = b.spill_set(id);
                let kind = jump_kind(&mut b, t, is_function);
                b.set_term(id, Term::Exit { kind, spill });
            }
            BlockKind::Default(addr) => {
                let v = switch_value[&addr];
                let spill = b.spill_set(id);
                b.set_term(
                    id,
                    Term::Exit {
                        kind: ExitKind::Dispatch(v),
                        spill,
                    },
                );
            }
        }
        b.block_filled(id);
    }
    let func = b.finish();
    debug_assert_eq!(func.verify(), Ok(()), "{func}");
    func
}

fn jump_kind(b: &mut Builder, target: u64, is_function: &dyn Fn(u64) -> bool) -> ExitKind {
    if is_function(target) {
        ExitKind::Jump(target)
    } else {
        ExitKind::Dispatch(b.konst(Ty::I64, u128::from(target)))
    }
}

fn lift_guest_block(
    b: &mut Builder,
    plan: &Plan,
    block: &GuestBlock,
    id: BlockId,
    is_function: &dyn Fn(u64) -> bool,
    switch_value: &mut BTreeMap<u64, ValueId>,
) {
    // Every instruction but the terminating one is straight-line.
    let straight = match block.end {
        End::Fall(_) | End::OutOfCode => block.insts.len(),
        _ => block.insts.len().saturating_sub(1),
    };
    for (pc, inst) in &block.insts[..straight] {
        Lifter::new(b, id).straight(*pc, inst);
    }
    let last = last_inst(block);
    match &block.end {
        End::Fall(a) => b.set_term(id, Term::Br(plan.guest[a])),
        End::OutOfCode => {
            let spill = b.spill_set(id);
            b.set_term(
                id,
                Term::Exit {
                    kind: ExitKind::Trap {
                        kind: TrapKind::OutOfCode,
                        pc: block.addr + 4 * straight as u64,
                    },
                    spill,
                },
            );
        }
        End::Trap => {
            let &(pc, ref inst) = last.expect("a trapping block ends in the trapping instruction");
            let kind = traps(inst).expect("trap block ends in a trapping instruction");
            let spill = b.spill_set(id);
            b.set_term(
                id,
                Term::Exit {
                    kind: ExitKind::Trap { kind, pc },
                    spill,
                },
            );
        }
        End::Jump(t) => match t {
            Target::Block(a) => b.set_term(id, Term::Br(plan.guest[a])),
            Target::Tail(a) => {
                let spill = b.spill_set(id);
                let kind = jump_kind(b, *a, is_function);
                b.set_term(id, Term::Exit { kind, spill });
            }
        },
        End::Cond { taken, fall } => {
            let (_, inst) = last.expect("conditional branch");
            let cond = Lifter::new(b, id).branch_condition(inst);
            let then = match taken {
                Target::Block(a) => plan.guest[a],
                Target::Tail(_) => plan.tail[&block.addr],
            };
            let els = plan.guest[fall];
            if then == els {
                b.set_term(id, Term::Br(then));
            } else {
                b.set_term(id, Term::CondBr { cond, then, els });
            }
        }
        End::Return => {
            let &(_, Inst::BranchReg { rn, .. }) = last.expect("ret") else {
                unreachable!("a return block ends in RET")
            };
            let target = Lifter::new(b, id).x(rn, false);
            let spill = b.spill_set(id);
            b.set_term(
                id,
                Term::Exit {
                    kind: ExitKind::Return(target),
                    spill,
                },
            );
        }
        End::Indirect { targets } => {
            let &(_, Inst::BranchReg { rn, .. }) = last.expect("br") else {
                unreachable!("an indirect block ends in BR")
            };
            let target = Lifter::new(b, id).x(rn, false);
            if targets.is_empty() {
                let spill = b.spill_set(id);
                b.set_term(
                    id,
                    Term::Exit {
                        kind: ExitKind::Dispatch(target),
                        spill,
                    },
                );
            } else {
                switch_value.insert(block.addr, target);
                let mut cases: Vec<(u64, BlockId)> =
                    targets.iter().map(|&a| (a, plan.guest[&a])).collect();
                cases.sort_unstable();
                cases.dedup();
                b.set_term(
                    id,
                    Term::Switch {
                        value: target,
                        cases,
                        default: plan.default[&block.addr],
                    },
                );
            }
        }
        End::Call { .. } => {
            let &(pc, ref inst) = last.expect("call");
            let ret_pc = pc + 4;
            let k = plan.cont[&block.addr];
            let target = match *inst {
                Inst::B { imm, .. } => {
                    let t = pc.wrapping_add(imm as u64);
                    {
                        let mut l = Lifter::new(b, id);
                        let ra = l.k64(ret_pc);
                        l.set_x(30, ra, false);
                    }
                    if is_function(t) {
                        CallTarget::Direct(t)
                    } else {
                        CallTarget::Indirect(b.konst(Ty::I64, u128::from(t)))
                    }
                }
                Inst::BranchReg {
                    kind: BranchReg::Blr,
                    rn,
                } => {
                    // The target is read before the link register is written: `BLR X30`.
                    let mut l = Lifter::new(b, id);
                    let t = l.x(rn, false);
                    let ra = l.k64(ret_pc);
                    l.set_x(30, ra, false);
                    CallTarget::Indirect(t)
                }
                Inst::Svc { imm } => CallTarget::Svc(imm),
                _ => CallTarget::Fallback,
            };
            let spill = b.spill_set(id);
            b.set_term(
                id,
                Term::Call(CallTerm {
                    target,
                    pc,
                    ret_pc,
                    cont: k,
                    spill,
                }),
            );
        }
    }
}
