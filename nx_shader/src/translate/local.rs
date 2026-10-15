//! Local memory: `LDL` and `STL` reach an array of words that belongs to the invocation. It is as
//! big as the program's header says, and at least as big as the constant offsets the program
//! reaches (docs/ASSUMPTIONS.md A-160).

use super::emit::{Tx, R};
use super::io::extract_small_dynamic;
use crate::cfg::{Cfg, Code};
use crate::inst::{Inst, RZ};
use crate::ops::Op;
use crate::spirv::{glsl, op, storage, Id};

/// Words of local memory when there is no header to say (the compute harness).
const HARNESS_WORDS: u32 = 1024;

/// How many words the constant-offset accesses of a program reach, for the size of the array.
pub(super) fn scan(code: Code<'_>, graph: &Cfg) -> u32 {
    let mut words = 0;
    for block in graph.blocks.values() {
        for (_, inst) in graph.instructions(code, block) {
            if matches!(inst.op(), Some(Op::Ldl | Op::Stl)) && inst.ra() == RZ {
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

/// The variable that holds local memory and its length in words, made on first use.
fn ensure(tx: &mut Tx<'_>) -> (Id, u32) {
    if let Some(l) = tx.local {
        return l;
    }
    let header = tx.sph.map_or(HARNESS_WORDS * 4, |s| {
        s.local_memory_low() + s.local_memory_high()
    });
    let words = header.div_ceil(4).max(tx.local_min_words).max(1);
    let u = tx.t.u32;
    let length = tx.c32(words);
    let array = tx.m.ty_array(u, length);
    let ptr = tx.m.ty_pointer(storage::PRIVATE, array);
    let var = tx.m.global_variable(ptr, storage::PRIVATE);
    tx.interface.push(var);
    tx.local = Some((var, words));
    (var, words)
}

/// A pointer to word `index`, kept inside the array (an access past its end reads and writes the
/// last word instead of whatever is there).
fn word_pointer(tx: &mut Tx<'_>, var: Id, words: u32, index: Id) -> Id {
    let u = tx.t.u32;
    let last = tx.c32(words - 1);
    let inside = tx.ext(u, glsl::U_MIN, &[index, last]);
    let ptr_ty = tx.m.ty_pointer(storage::PRIVATE, u);
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

pub(super) fn ldl(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let size = inst.bits(48, 3);
    let (var, words) = ensure(tx);
    let addr = address(tx, inst);
    let u = tx.t.u32;
    let two = tx.c32(2);
    let first = tx.shr(addr, two);
    match words_of(size) {
        Some(n) => {
            for w in 0..n {
                let k = tx.c32(w);
                let at = tx.iadd(first, k);
                let ptr = word_pointer(tx, var, words, at);
                let v = tx.f.load(&mut tx.m, u, ptr);
                tx.write(inst, inst.rd().wrapping_add(w as u8), v);
            }
        }
        None => {
            let ptr = word_pointer(tx, var, words, first);
            let word = tx.f.load(&mut tx.m, u, ptr);
            let three = tx.c32(3);
            let byte = tx.and(addr, three);
            let v = extract_small_dynamic(tx, word, byte, size);
            tx.write(inst, inst.rd(), v);
        }
    }
    Ok(())
}

pub(super) fn stl(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let size = inst.bits(48, 3);
    let (var, words) = ensure(tx);
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
                let ptr = word_pointer(tx, var, words, at);
                tx.f.store(ptr, value);
            }
        }
        None => {
            // A byte or a half word goes into the word that holds it.
            let value = tx.read(inst.rd());
            let ptr = word_pointer(tx, var, words, first);
            let old = tx.f.load(&mut tx.m, u, ptr);
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
            let kept = tx.and(old, kept);
            let merged = tx.or(kept, moved);
            tx.f.store(ptr, merged);
        }
    }
    if cond.is_some() {
        tx.f.branch(merge);
        tx.f.label(merge);
    }
    Ok(())
}
