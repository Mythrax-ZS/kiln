//! Global memory: `LDG` and `STG` read and write memory by GPU virtual address.
//!
//! The shader cannot see GPU virtual addresses directly, so it gets a table of the ranges the host
//! has made visible (see [`super::Info::global_memory`]) and turns an address into a device
//! address with a small loop over it. The memory is then reached through a
//! `PhysicalStorageBuffer` pointer (`VK_KHR_buffer_device_address`). An address no entry covers
//! lands in a scratch page, so a stray access cannot fault.

use super::emit::{Tx, R};
use crate::inst::{Inst, RZ};
use crate::spirv::{cap, op, storage, Id};

/// Offsets (in 32-bit words) in the table.
const COUNT: u32 = 0;
const SCRATCH: u32 = 2;
const ENTRIES: u32 = 4;
const ENTRY_WORDS: u32 = 6;

/// The table variable and the address translation function, made on first use.
fn ensure(tx: &mut Tx<'_>) -> (Id, Id) {
    if let Some(g) = tx.gmap {
        return g;
    }
    tx.info.global_memory = true;
    tx.m.capability(cap::INT64);
    tx.m.capability(cap::PHYSICAL_STORAGE_BUFFER_ADDRESSES);
    tx.m.extension("SPV_KHR_physical_storage_buffer");
    let binding = 64 * tx.cfg.stage.index() + 20;
    let map = super::io::storage_buffer(tx, binding, false);
    let u64t = tx.m.ty_int(64, false);
    let fn_ty = tx.m.ty_function(u64t, &[u64t]);
    let mut g = tx.m.function(u64t, fn_ty);
    let va = g.parameter(&mut tx.m, u64t);
    let id = g.id;
    let outer = std::mem::replace(&mut tx.f, g);

    let u = tx.t.u32;
    let bool_t = tx.t.bool_;
    let zero = tx.c32(0);
    let one = tx.c32(1);
    let ptr_u = tx.m.ty_pointer(storage::FUNCTION, u);
    let i_var = tx.m.id();
    tx.f.local_with_init(ptr_u, i_var, zero);
    let word = |tx: &mut Tx<'_>, index: Id| -> Id {
        let p = super::io::word_ptr(tx, map, index);
        tx.f.load(&mut tx.m, u, p)
    };
    let wide = |tx: &mut Tx<'_>, lo: Id, hi: Id| -> Id {
        let lo64 = tx.op1(op::U_CONVERT, u64t, lo);
        let hi64 = tx.op1(op::U_CONVERT, u64t, hi);
        let c32 = tx.c32(32);
        let shifted = tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, hi64, c32);
        tx.op2(op::BITWISE_OR, u64t, lo64, shifted)
    };
    let header = tx.m.id();
    let check = tx.m.id();
    let body = tx.m.id();
    let hit = tx.m.id();
    let after = tx.m.id();
    let cont = tx.m.id();
    let exit = tx.m.id();
    tx.f.branch(header);
    tx.f.label(header);
    tx.f.loop_merge(exit, cont);
    tx.f.branch(check);
    tx.f.label(check);
    let i = tx.f.load(&mut tx.m, u, i_var);
    let count_ix = tx.c32(COUNT);
    let count = word(tx, count_ix);
    let more = tx.cmp(op::U_LESS_THAN, i, count);
    tx.f.branch_conditional(more, body, exit);
    tx.f.label(body);
    let stride = tx.c32(ENTRY_WORDS);
    let base0 = tx.imul(i, stride);
    let entries = tx.c32(ENTRIES);
    let base = tx.iadd(base0, entries);
    let mut w = [zero; 6];
    for (k, slot) in w.iter_mut().enumerate() {
        let off = tx.c32(k as u32);
        let ix = tx.iadd(base, off);
        *slot = word(tx, ix);
    }
    let start = wide(tx, w[0], w[1]);
    let end = wide(tx, w[2], w[3]);
    let dev = wide(tx, w[4], w[5]);
    let ge = tx.cmp(op::U_GREATER_THAN_EQUAL, va, start);
    let lt = tx.cmp(op::U_LESS_THAN, va, end);
    let inside = tx.land(ge, lt);
    tx.f.selection_merge(after);
    tx.f.branch_conditional(inside, hit, after);
    tx.f.label(hit);
    let delta = tx.op2(op::I_SUB, u64t, va, start);
    let result = tx.op2(op::I_ADD, u64t, dev, delta);
    tx.f.emit(op::RETURN_VALUE, &[result]);
    tx.f.label(after);
    tx.f.branch(cont);
    tx.f.label(cont);
    let next = tx.iadd(i, one);
    tx.f.store(i_var, next);
    tx.f.branch(header);
    tx.f.label(exit);
    let lo_ix = tx.c32(SCRATCH);
    let hi_ix = tx.c32(SCRATCH + 1);
    let lo = word(tx, lo_ix);
    let hi = word(tx, hi_ix);
    let scratch = wide(tx, lo, hi);
    tx.f.emit(op::RETURN_VALUE, &[scratch]);
    let _ = bool_t;

    let done = std::mem::replace(&mut tx.f, outer);
    tx.m.add_function(done);
    tx.gmap = Some((map, id));
    (map, id)
}

/// `(Ra+1 : Ra) + imm24` as a 64-bit GPU address.
fn address(tx: &mut Tx<'_>, inst: Inst) -> R<Id> {
    if !inst.bit(45) {
        return tx.fail("a global memory access with a 32-bit address");
    }
    let u64t = tx.m.ty_int(64, false);
    let (lo, hi) = if inst.ra() == RZ {
        (tx.c32(0), tx.c32(0))
    } else {
        (tx.read(inst.ra()), tx.read(inst.ra().wrapping_add(1)))
    };
    let lo64 = tx.op1(op::U_CONVERT, u64t, lo);
    let hi64 = tx.op1(op::U_CONVERT, u64t, hi);
    let c32 = tx.c32(32);
    let shifted = tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, hi64, c32);
    let base = tx.op2(op::BITWISE_OR, u64t, lo64, shifted);
    let imm = inst.sbits(20, 24);
    let imm_c = tx.m.const_u64(imm as u64);
    Ok(tx.op2(op::I_ADD, u64t, base, imm_c))
}

/// The device address behind a GPU address.
fn device(tx: &mut Tx<'_>, gpu: Id) -> Id {
    let (_, function) = ensure(tx);
    let u64t = tx.m.ty_int(64, false);
    tx.f.emit_result(&mut tx.m, op::FUNCTION_CALL, u64t, &[function, gpu])
}

/// A pointer to the 32-bit word at a device address.
fn word_pointer(tx: &mut Tx<'_>, device: Id) -> Id {
    let ptr_ty = tx.m.ty_pointer(storage::PHYSICAL_STORAGE_BUFFER, tx.t.u32);
    tx.f.emit_result(&mut tx.m, op::CONVERT_U_TO_PTR, ptr_ty, &[device])
}

/// The memory-access operands of an access aligned to 4 bytes.
const ALIGNED_4: [u32; 2] = [0x2, 4];

fn plus(tx: &mut Tx<'_>, device: Id, bytes: u32) -> Id {
    if bytes == 0 {
        return device;
    }
    let u64t = tx.m.ty_int(64, false);
    let c = tx.m.const_u64(u64::from(bytes));
    tx.op2(op::I_ADD, u64t, device, c)
}

/// The size field of `LDG`/`STG` (bits 48..50): 0 u8, 1 s8, 2 u16, 3 s16, 4 b32, 5 b64, 6 and 7
/// b128. Returns the words moved, or `None` for a size smaller than a word.
fn words_of(size: u64) -> Option<u32> {
    match size {
        4 => Some(1),
        5 => Some(2),
        6 | 7 => Some(4),
        _ => None,
    }
}

pub(super) fn ldg(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let size = inst.bits(48, 3);
    let gpu = address(tx, inst)?;
    let u = tx.t.u32;
    match words_of(size) {
        Some(words) => {
            let dev = device(tx, gpu);
            for w in 0..words {
                let at = plus(tx, dev, 4 * w);
                let ptr = word_pointer(tx, at);
                let v =
                    tx.f.emit_result(&mut tx.m, op::LOAD, u, &[ptr, ALIGNED_4[0], ALIGNED_4[1]]);
                tx.write(inst, inst.rd().wrapping_add(w as u8), v);
            }
        }
        None => {
            // A byte or a half word: load the word that holds it and take the bits.
            let u64t = tx.m.ty_int(64, false);
            let mask = tx.m.const_u64(!3);
            let aligned = tx.op2(op::BITWISE_AND, u64t, gpu, mask);
            let low = tx.op1(op::U_CONVERT, u, gpu);
            let three = tx.c32(3);
            let byte = tx.and(low, three);
            let dev = device(tx, aligned);
            let ptr = word_pointer(tx, dev);
            let word =
                tx.f.emit_result(&mut tx.m, op::LOAD, u, &[ptr, ALIGNED_4[0], ALIGNED_4[1]]);
            let eight = tx.c32(8);
            let shift = tx.imul(byte, eight);
            let shifted = tx.shr(word, shift);
            let zero = tx.c32(0);
            let (bits, signed) = match size {
                0 => (8, false),
                1 => (8, true),
                2 => (16, false),
                _ => (16, true),
            };
            let len = tx.c32(bits);
            let v = if signed {
                let s = tx.as_i32(shifted);
                let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, zero, len);
                tx.as_u32(e)
            } else {
                tx.op3(op::BIT_FIELD_U_EXTRACT, u, shifted, zero, len)
            };
            tx.write(inst, inst.rd(), v);
        }
    }
    Ok(())
}

pub(super) fn stg(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let size = inst.bits(48, 3);
    let Some(words) = words_of(size) else {
        return tx.fail("STG of less than a word");
    };
    let gpu = address(tx, inst)?;
    let dev = device(tx, gpu);
    // The stores only happen when the guard holds.
    let cond = tx.guard_condition(inst.guard());
    let merge = tx.m.id();
    if let Some(c) = cond {
        let then = tx.m.id();
        tx.f.selection_merge(merge);
        tx.f.branch_conditional(c, then, merge);
        tx.f.label(then);
    }
    for w in 0..words {
        let value = tx.read(inst.rd().wrapping_add(w as u8));
        let at = plus(tx, dev, 4 * w);
        let ptr = word_pointer(tx, at);
        tx.f.emit(op::STORE, &[ptr, value, ALIGNED_4[0], ALIGNED_4[1]]);
    }
    if cond.is_some() {
        tx.f.branch(merge);
        tx.f.label(merge);
    }
    Ok(())
}
