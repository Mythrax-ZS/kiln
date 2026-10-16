//! Local and shared memory: `LDL` and `STL` reach an array of words that belongs to the invocation.
//! It is as big as the program's header says, and at least as big as the constant offsets the
//! program reaches (docs/ASSUMPTIONS.md A-160). `LDS` and `STS` reach an array of words that the
//! invocations of a workgroup have in common, as big as the launch says (A-170).

use super::emit::{Tx, R};
use super::io::extract_small_dynamic;
use super::Stage;
use crate::cfg::{Cfg, Code};
use crate::inst::{Inst, RZ};
use crate::ops::Op;
use crate::spirv::{glsl, op, storage, Id};

/// Which memory an access is to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Space {
    /// Belongs to the invocation (`LDL`, `STL`).
    Local,
    /// Belongs to the workgroup (`LDS`, `STS`).
    Shared,
}

impl Space {
    fn class(self) -> u32 {
        match self {
            Space::Local => storage::PRIVATE,
            Space::Shared => storage::WORKGROUP,
        }
    }

    fn accessed_by(self, op: Option<Op>) -> bool {
        match self {
            Space::Local => matches!(op, Some(Op::Ldl | Op::Stl)),
            Space::Shared => matches!(op, Some(Op::Lds | Op::Sts)),
        }
    }
}

/// Words of memory when there is no header or launch to say (the compute harness).
const HARNESS_WORDS: u32 = 1024;

/// How many words the constant-offset accesses of a program reach, for the size of the array.
pub(super) fn scan(code: Code<'_>, graph: &Cfg, space: Space) -> u32 {
    let mut words = 0;
    for block in graph.blocks.values() {
        for (_, inst) in graph.instructions(code, block) {
            if space.accessed_by(inst.op()) && inst.ra() == RZ {
                let end = inst.sbits(20, 24).max(0) as u32 + access_bytes(inst);
                words = words.max(end.div_ceil(4));
            }
        }
    }
    words
}

/// The bytes an access moves, counting a part of a word as the word it is in.
fn access_bytes(inst: Inst) -> u32 {
    match inst.bits(48, 3) {
        5 => 8,
        6 | 7 => 16,
        _ => 4,
    }
}

/// The variable that holds the memory and its length in words, made on first use.
fn ensure(tx: &mut Tx<'_>, space: Space) -> R<(Id, u32)> {
    let made = match space {
        Space::Local => tx.local,
        Space::Shared => tx.shared,
    };
    if let Some(l) = made {
        return Ok(l);
    }
    let (bytes, min_words) = match space {
        Space::Local => (
            tx.sph.map_or(HARNESS_WORDS * 4, |s| {
                s.local_memory_low() + s.local_memory_high()
            }),
            tx.local_min_words,
        ),
        Space::Shared => {
            if !matches!(tx.cfg.stage, Stage::Compute | Stage::Kernel) {
                return tx.fail("shared memory in a program that is not a compute program");
            }
            let launch = tx.cfg.shared_bytes;
            (
                if launch == 0 {
                    HARNESS_WORDS * 4
                } else {
                    launch
                },
                tx.shared_min_words,
            )
        }
    };
    let words = bytes.div_ceil(4).max(min_words).max(1);
    let u = tx.t.u32;
    let length = tx.c32(words);
    let array = tx.m.ty_array(u, length);
    let ptr = tx.m.ty_pointer(space.class(), array);
    let var = tx.m.global_variable(ptr, space.class());
    tx.interface.push(var);
    match space {
        Space::Local => tx.local = Some((var, words)),
        Space::Shared => tx.shared = Some((var, words)),
    }
    Ok((var, words))
}

/// A pointer to word `index`, kept inside the array (an access past its end reads and writes the
/// last word instead of whatever is there).
fn word_pointer(tx: &mut Tx<'_>, space: Space, var: Id, words: u32, index: Id) -> Id {
    let u = tx.t.u32;
    let last = tx.c32(words - 1);
    let inside = tx.ext(u, glsl::U_MIN, &[index, last]);
    let ptr_ty = tx.m.ty_pointer(space.class(), u);
    tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[var, inside])
}

/// The byte address `Ra + imm24`.
fn address(tx: &mut Tx<'_>, inst: Inst) -> Id {
    let base = tx.read(inst.ra());
    let imm = tx.c32(inst.sbits(20, 24) as u32);
    tx.iadd(base, imm)
}

/// The words a whole-word access moves (sizes 4 to 7 of bits 48..50: 32, 64 and 128 bits).
fn words_of(size: u64) -> Option<u32> {
    match size {
        4 => Some(1),
        5 => Some(2),
        6 | 7 => Some(4),
        _ => None,
    }
}

/// `LDL` and `LDS`.
pub(super) fn load(tx: &mut Tx<'_>, inst: Inst, space: Space) -> R<()> {
    let size = inst.bits(48, 3);
    let (var, words) = ensure(tx, space)?;
    let addr = address(tx, inst);
    let u = tx.t.u32;
    let two = tx.c32(2);
    let first = tx.shr(addr, two);
    match words_of(size) {
        Some(n) => {
            for w in 0..n {
                let k = tx.c32(w);
                let at = tx.iadd(first, k);
                let ptr = word_pointer(tx, space, var, words, at);
                let v = tx.f.load(&mut tx.m, u, ptr);
                tx.write(inst, inst.rd().wrapping_add(w as u8), v);
            }
        }
        None => {
            let ptr = word_pointer(tx, space, var, words, first);
            let word = tx.f.load(&mut tx.m, u, ptr);
            let three = tx.c32(3);
            let byte = tx.and(addr, three);
            let v = extract_small_dynamic(tx, word, byte, size);
            tx.write(inst, inst.rd(), v);
        }
    }
    Ok(())
}

/// `STL` and `STS`.
pub(super) fn store(tx: &mut Tx<'_>, inst: Inst, space: Space) -> R<()> {
    let size = inst.bits(48, 3);
    let (var, words) = ensure(tx, space)?;
    let addr = address(tx, inst);
    let u = tx.t.u32;
    let two = tx.c32(2);
    let first = tx.shr(addr, two);
    // The stores only happen when the guard holds.
    let cond = tx.guard_condition(inst.guard());
    let merge = tx.m.id();
    if let Some(c) = cond {
        let then = tx.m.id();
        tx.f.selection_merge(merge);
        tx.f.branch_conditional(c, then, merge);
        tx.f.label(then);
    }
    match words_of(size) {
        Some(n) => {
            for w in 0..n {
                let value = tx.read(inst.rd().wrapping_add(w as u8));
                let k = tx.c32(w);
                let at = tx.iadd(first, k);
                let ptr = word_pointer(tx, space, var, words, at);
                tx.f.store(ptr, value);
            }
        }
        None => {
            // A byte or a half word goes into the word that holds it.
            let value = tx.read(inst.rd());
            let ptr = word_pointer(tx, space, var, words, first);
            let three = tx.c32(3);
            let byte = tx.and(addr, three);
            let eight = tx.c32(8);
            let shift = tx.imul(byte, eight);
            let field = if size < 2 { 0xFF } else { 0xFFFF };
            let field = tx.c32(field);
            let window = tx.shl(field, shift);
            let moved = tx.shl(value, shift);
            let moved = tx.and(moved, window);
            let kept = tx.not(window);
            match space {
                Space::Local => {
                    let old = tx.f.load(&mut tx.m, u, ptr);
                    let kept = tx.and(old, kept);
                    let merged = tx.or(kept, moved);
                    tx.f.store(ptr, merged);
                }
                Space::Shared => {
                    // The other invocations may be writing the other bytes of the word.
                    let scope = tx.c32(2);
                    let relaxed = tx.c32(0);
                    tx.f.emit_result(&mut tx.m, op::ATOMIC_AND, u, &[ptr, scope, relaxed, kept]);
                    tx.f.emit_result(&mut tx.m, op::ATOMIC_OR, u, &[ptr, scope, relaxed, moved]);
                }
            }
        }
    }
    if cond.is_some() {
        tx.f.branch(merge);
        tx.f.label(merge);
    }
    Ok(())
}
