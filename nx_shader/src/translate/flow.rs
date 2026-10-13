//! Blocks and control flow: the straight-line case, and the dispatch loop for the rest.

use super::emit::{Tx, R};
use super::{alu, io};
use crate::cfg::{Block, Cfg, Code, Term};
use crate::inst::Inst;
use crate::ops::Op;
use crate::spirv::{op, storage, Id};

/// Translate the program: the prologue, every block, the epilogue.
pub(super) fn build(tx: &mut Tx<'_>, code: Code<'_>, graph: &Cfg) -> R<()> {
    io::begin(tx)?;
    let entry = &graph.blocks[&graph.entry];
    let straight = graph.blocks.len() == 1
        && matches!(entry.term, Term::Exit { guard, .. } if guard.is_always());
    if straight {
        body(tx, code, graph, entry)?;
        io::epilogue(tx)?;
        tx.f.ret();
        return Ok(());
    }
    tx.info.dispatch_loop = true;
    // The program counter, in bytes from the start of the code.
    let u = tx.t.u32;
    let pc_ptr = tx.m.ty_pointer(storage::FUNCTION, u);
    let pc_var = tx.m.id();
    let entry_pc = tx.c32(graph.entry);
    tx.f.local_with_init(pc_ptr, pc_var, entry_pc);

    let header = tx.m.id();
    let dispatch = tx.m.id();
    let cont = tx.m.id();
    let exit = tx.m.id();
    let switch_merge = tx.m.id();
    let labels: Vec<(u32, Id)> = graph.blocks.keys().map(|pc| (*pc, tx.m.id())).collect();
    tx.f.branch(header);
    tx.f.label(header);
    tx.f.loop_merge(exit, cont);
    tx.f.branch(dispatch);
    tx.f.label(dispatch);
    let pc = tx.f.load(&mut tx.m, u, pc_var);
    tx.f.selection_merge(switch_merge);
    tx.f.switch(pc, switch_merge, &labels);
    for (start, label) in &labels {
        let block = &graph.blocks[start];
        tx.f.label(*label);
        body(tx, code, graph, block)?;
        terminate(tx, block, &labels, pc_var, cont, exit)?;
    }
    // Every program counter value names a block, so the switch never gets here.
    tx.f.label(switch_merge);
    tx.f.unreachable();
    tx.f.label(cont);
    tx.f.branch(header);
    // The loop's merge block is where every exit goes: the outputs are written once, here.
    tx.f.label(exit);
    io::epilogue(tx)?;
    tx.f.ret();
    Ok(())
}

/// The instructions of a block, without its terminator.
fn body(tx: &mut Tx<'_>, code: Code<'_>, graph: &Cfg, block: &Block) -> R<()> {
    let mut insts: Vec<(u32, Inst)> = graph.instructions(code, block).collect();
    if !matches!(block.term, Term::Fall(_)) {
        insts.pop();
    }
    for (pc, inst) in insts {
        tx.pc = pc;
        if inst.guard().is_never() && !matches!(inst.op(), Some(Op::Ssy | Op::Pbk | Op::Pcnt)) {
            continue;
        }
        instruction(tx, inst)?;
        tx.info.instructions += 1;
    }
    Ok(())
}

fn label_of(labels: &[(u32, Id)], pc: u32) -> Id {
    labels
        .iter()
        .find(|(p, _)| *p == pc)
        .map(|(_, l)| *l)
        .expect("a block exists for every target the graph names")
}

/// End a block: set the program counter and go around the loop, or leave it.
fn terminate(
    tx: &mut Tx<'_>,
    block: &Block,
    labels: &[(u32, Id)],
    pc_var: Id,
    cont: Id,
    exit: Id,
) -> R<()> {
    let goto = |tx: &mut Tx<'_>, target: u32| {
        let c = tx.c32(target);
        tx.f.store(pc_var, c);
        tx.f.branch(cont);
    };
    match block.term {
        Term::Fall(next) => goto(tx, next),
        Term::Jump {
            guard,
            target,
            next,
            ..
        } => {
            let _ = label_of(labels, target);
            match (tx.guard_condition(guard), next) {
                (None, _) | (Some(_), None) => goto(tx, target),
                (Some(cond), Some(next)) => {
                    let t = tx.c32(target);
                    let n = tx.c32(next);
                    let pc = tx.select_u32(cond, t, n);
                    tx.f.store(pc_var, pc);
                    tx.f.branch(cont);
                }
            }
        }
        Term::Exit { guard, next } => match (tx.guard_condition(guard), next) {
            (None, _) | (Some(_), None) => tx.f.branch(exit),
            (Some(cond), Some(next)) => {
                let leave = tx.m.id();
                let rest = tx.m.id();
                tx.f.selection_merge(rest);
                tx.f.branch_conditional(cond, leave, rest);
                tx.f.label(leave);
                tx.f.branch(exit);
                tx.f.label(rest);
                goto(tx, next);
            }
        },
        Term::Kill { guard, next } => {
            tx.info.kills = true;
            match (tx.guard_condition(guard), next) {
                (None, _) | (Some(_), None) => tx.f.emit(op::KILL, &[]),
                (Some(cond), Some(next)) => {
                    let die = tx.m.id();
                    let rest = tx.m.id();
                    tx.f.selection_merge(rest);
                    tx.f.branch_conditional(cond, die, rest);
                    tx.f.label(die);
                    tx.f.emit(op::KILL, &[]);
                    tx.f.label(rest);
                    goto(tx, next);
                }
            }
        }
        Term::Unsupported => {
            return tx.fail("a block that ends in a kind of control flow that is not translated")
        }
    }
    Ok(())
}

/// One instruction that is not a terminator.
fn instruction(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let Some(row) = inst.row() else {
        return tx.fail(format!("instruction {:#018x} is not known", inst.0));
    };
    match row.op {
        // Nothing to do: scheduling hints, barriers of a model that runs in order, the
        // bookkeeping of control flow (the graph has used it already).
        Op::Nop | Op::Ssy | Op::Pbk | Op::Pcnt | Op::Depbar | Op::Membar => Ok(()),
        _ => {
            if alu::translate(tx, inst, row.op)? {
                return Ok(());
            }
            if io::translate(tx, inst, row.op)? {
                return Ok(());
            }
            tx.fail(format!("{} ({:?}) is not translated", row.name, row.op))
        }
    }
}
