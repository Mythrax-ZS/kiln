//! What an sm_86 program reads and writes besides its registers: system registers (`S2R`, `CS2R`,
//! `S2UR`), constant buffers (`LDC`, `ULDC`), attributes (`ALD`, `AST`, `IPA`), global memory
//! (`LDG`, `STG`), local and shared memory (`LDL`, `STL`, `LDS`, `STS`), and the barriers.
//!
//! Field positions are NAK's: an access's size in bits 73..75 (0 u8, 1 s8, 2 u16, 3 s16, 4 b32,
//! 5 b64, 6 b128), its immediate offset in bits 40..63; a global address is a register (a pair with
//! bit 90 for the forms with a uniform register, bit 72 for the others) plus, for the forms with
//! one, a uniform register (a pair with bit 72) plus the offset.

use super::super::emit::{Tx, R};
use super::super::global::{device, plus, word_pointer as global_word, ALIGNED_4};
use super::super::io::{attribute_in, attribute_out, extract_small, extract_small_dynamic};
use super::super::local::{self, Space};
use super::super::Stage;
use super::{dst, read, reg_at, write, write_f, I};
use crate::sm86::{op, Inst, RZ, URZ};
use crate::spirv::{builtin, cap, deco, op as sop, storage, Id};

/// The value of a built-in input (made on first use) of type `ty`.
pub(in crate::translate) fn builtin_value(tx: &mut Tx<'_>, which: u32, ty: Id) -> Id {
    let var = match tx.builtins.get(&which) {
        Some(v) => *v,
        None => {
            let ptr = tx.m.ty_pointer(storage::INPUT, ty);
            let v = tx.m.global_variable(ptr, storage::INPUT);
            tx.m.decorate(v, deco::BUILT_IN, &[which]);
            if tx.cfg.stage == Stage::Pixel && ty == tx.t.u32 {
                tx.m.decorate(v, deco::FLAT, &[]);
            }
            tx.interface.push(v);
            tx.builtins.insert(which, v);
            v
        }
    };
    tx.f.load(&mut tx.m, ty, var)
}

/// A system register: the lane number and the lane masks of the warp (the host's subgroup, 32 wide
/// on the GPUs this runs on: docs/ASSUMPTIONS.md A-243), and in a compute program the invocation's
/// place in its workgroup and the workgroup's in the grid.
fn system_register(tx: &mut Tx<'_>, sr: u64) -> R<Id> {
    let u = tx.t.u32;
    match sr {
        // SR_LANEID
        0x00 => {
            tx.m.capability(cap::GROUP_NON_UNIFORM);
            Ok(builtin_value(tx, builtin::SUBGROUP_LOCAL_INVOCATION_ID, u))
        }
        // SR_EQMASK, SR_LTMASK, SR_LEMASK, SR_GTMASK, SR_GEMASK
        0x38..=0x3c => {
            tx.m.capability(cap::GROUP_NON_UNIFORM_BALLOT);
            let which = match sr {
                0x38 => builtin::SUBGROUP_EQ_MASK,
                0x39 => builtin::SUBGROUP_LT_MASK,
                0x3a => builtin::SUBGROUP_LE_MASK,
                0x3b => builtin::SUBGROUP_GT_MASK,
                _ => builtin::SUBGROUP_GE_MASK,
            };
            let v4 = tx.t.v4u32;
            let mask = builtin_value(tx, which, v4);
            Ok(tx
                .f
                .emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, u, &[mask, 0]))
        }
        // SR_TID.X..Z, SR_CTAID.X..Z
        0x21..=0x23 | 0x25..=0x27
            if matches!(tx.cfg.stage, Stage::Kernel | Stage::Task | Stage::Mesh) =>
        {
            let which = if sr < 0x25 {
                builtin::LOCAL_INVOCATION_ID
            } else {
                builtin::WORKGROUP_ID
            };
            let uvec3 = tx.m.ty_vector(u, 3);
            let all = builtin_value(tx, which, uvec3);
            let component = (if sr < 0x25 { sr - 0x21 } else { sr - 0x25 }) as u32;
            Ok(tx
                .f
                .emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, u, &[all, component]))
        }
        _ => tx.fail(format!("S2R of system register {sr:#x}")),
    }
}

/// `ISBERD Rd, [Ra + imm]` and `ISBEWR [Ra + imm], Rb`: the stage buffer of a task or mesh
/// program (A-270). Bit 75 says 32 bits (a byte otherwise), bits 76..77 the region (3 the
/// attributes, 0 the map), bit 79 the output buffer.
pub(super) fn stage_buffer(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let base = tx.read(inst.reg(24));
    let offset = tx.c32(inst.field(40, 24) as u32);
    let addr = tx.iadd(base, offset);
    let mode = super::super::mesh::Mode {
        wide: inst.bit(75),
        region: inst.field(76, 2) as u32,
        output: inst.bit(79),
    };
    if inst.opcode() == op::ISBERD {
        let v = super::super::mesh::read(tx, addr, mode)?;
        write(tx, i, inst.reg(16), v);
        Ok(())
    } else {
        let value = tx.read(inst.reg(32));
        super::super::mesh::write(tx, addr, value, mode)
    }
}

/// `S2R Rd, SR`.
pub(super) fn s2r(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let v = system_register(tx, i.inst.field(72, 8))?;
    write(tx, i, dst(i), v);
    Ok(())
}

/// `CS2R[.32] Rd, SR`: `SRZ` (255) zeroes one register, or two with bit 80.
pub(super) fn cs2r(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let sr = i.inst.field(72, 8);
    let count = 1 + u32::from(i.inst.bit(80));
    for k in 0..count {
        let v = if sr == 0xFF {
            tx.c32(0)
        } else {
            system_register(tx, sr + u64::from(k))?
        };
        write(tx, i, reg_at(i, dst(i), k), v);
    }
    Ok(())
}

/// `S2UR URd, SR`.
pub(super) fn s2ur(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let v = system_register(tx, i.inst.field(72, 8))?;
    tx.write_u_if(i.cond, i.inst.ureg(16), v);
    Ok(())
}

/// The words an access of `size` moves, `None` for less than a word.
fn words_of(size: u64) -> Option<u32> {
    match size {
        4 => Some(1),
        5 => Some(2),
        6 => Some(4),
        _ => None,
    }
}

/// The constant buffer slot an instruction names (the unbound one is the spare slot).
pub(super) fn slot_of(inst: Inst) -> u32 {
    match inst.field(54, 5) as u32 {
        super::UNBOUND_SLOT => super::SPARE_SLOT,
        s => s,
    }
}

/// A load of `size` from constant buffer `slot` at `index` (a register value, or none) plus the
/// byte offset `off`: the words for the destination registers.
fn load_cbuf(tx: &mut Tx<'_>, slot: u32, index: Option<Id>, off: u32, size: u64) -> R<Vec<Id>> {
    let words = match words_of(size) {
        Some(w) => w,
        None if size < 4 => 1,
        None => return tx.fail(format!("LDC of size code {size}")),
    };
    let mut out = Vec::new();
    match index {
        None => {
            for w in 0..words {
                let mut v = tx.cbuf_word(slot, (off & !3) + 4 * w)?;
                if size < 4 {
                    v = extract_small(tx, v, off & 3, size);
                }
                out.push(v);
            }
        }
        Some(base) => {
            let imm = tx.c32(off);
            let addr = tx.iadd(base, imm);
            for w in 0..words {
                let add = tx.c32(4 * w);
                let at = tx.iadd(addr, add);
                let mask = tx.c32(!3u32);
                let aligned = tx.and(at, mask);
                let mut v = tx.cbuf_word_dynamic(slot, aligned)?;
                if size < 4 {
                    let three = tx.c32(3);
                    let byte = tx.and(at, three);
                    v = extract_small_dynamic(tx, v, byte, size);
                }
                out.push(v);
            }
        }
    }
    Ok(out)
}

/// `LDC.size Rd, c[slot][Ra + offset]` (the offset in bits 38..53, unsigned; NeXium).
pub(super) fn ldc(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.bit(91) {
        return tx.fail("LDC from the constant buffer a uniform register names");
    }
    if inst.field(78, 2) != 0 {
        return tx.fail(format!("LDC in indexing mode {}", inst.field(78, 2)));
    }
    let size = inst.field(73, 3);
    let slot = slot_of(inst);
    let off = inst.field(38, 16) as u32;
    let r = inst.reg(24);
    let index = (r != RZ).then(|| tx.read(r));
    let words = load_cbuf(tx, slot, index, off, size)?;
    let d = dst(i);
    let count = words.len() as u32;
    for (k, v) in words.into_iter().enumerate() {
        write(tx, i, reg_at(i, d, k as u32), v);
    }
    // The word the register now holds, for a texture instruction that takes its handle from it.
    if size == 4 && r == RZ && off.is_multiple_of(4) && i.cond.is_none() && d != RZ {
        tx.reg_origin.insert(d, (slot, off / 4));
    }
    // The words the registers hold, for a global access whose base they are (A-258).
    if r == RZ && off.is_multiple_of(4) && size >= 4 {
        for k in 0..count {
            let reg = reg_at(i, d, k);
            if reg != RZ {
                tx.ptr_origin.insert(reg, (slot, off / 4 + k));
            }
        }
    }
    Ok(())
}

/// `ULDC.size URd, c[slot][URa + offset]` (`URa` only for the indexed form).
pub(super) fn uldc(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let indexed = inst.opcode() == op::ULDC_INDEXED;
    if inst.bit(91) != indexed {
        return tx.fail("ULDC from the constant buffer a uniform register names");
    }
    let size = inst.field(73, 3);
    let slot = slot_of(inst);
    let off = inst.field(38, 16) as u32;
    let index = if indexed && inst.ureg(24) != URZ {
        Some(tx.read_u(inst.ureg(24)))
    } else {
        None
    };
    let words = load_cbuf(tx, slot, index, off, size)?;
    let d = inst.ureg(16);
    let count = words.len() as u32;
    for (k, v) in words.into_iter().enumerate() {
        let r = if d == URZ { URZ } else { d + k as u8 };
        tx.write_u_if(i.cond, r, v);
    }
    // The words the registers hold, for a global access whose base they are (A-258).
    if index.is_none() && off.is_multiple_of(4) && size >= 4 && d != URZ {
        for k in 0..count {
            tx.uptr_origin.insert(d + k as u8, (slot, off / 4 + k));
        }
    }
    Ok(())
}

/// `ALD Rd, a[Ra + addr], Rvtx`: input attributes (bits 74..75: one to four words).
pub(super) fn ald(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.reg(24) != RZ {
        return tx.fail("ALD with a register in the address (indirect attribute access)");
    }
    if inst.bit(76) || inst.bit(79) {
        return tx.fail("ALD of a patch or output attribute");
    }
    let base = inst.field(40, 10) as u32;
    for k in 0..inst.field(74, 2) as u32 + 1 {
        let v = attribute_in(tx, base + 4 * k)?;
        write(tx, i, reg_at(i, dst(i), k), v);
    }
    Ok(())
}

/// The output attribute that picks a primitive's entry of the shading rate table (NAK's
/// `NAK_ATTR_VPRS_TABLE_INDEX`).
const VPRS_TABLE_INDEX: u32 = 0x5c;

/// `AST a[Ra + addr], Rdata`: output attributes.
pub(super) fn ast(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.reg(24) != RZ {
        return tx.fail("AST with a register in the address (indirect attribute access)");
    }
    if inst.bit(76) {
        return tx.fail("AST of a patch attribute");
    }
    let base = inst.field(40, 10) as u32;
    let data = inst.reg(32);
    for k in 0..inst.field(74, 2) as u32 + 1 {
        let addr = base + 4 * k;
        // The variable-rate shading table index: every pixel is shaded (A-245).
        if addr == VPRS_TABLE_INDEX {
            continue;
        }
        let v = read(tx, i, reg_at(i, data, k));
        attribute_out(tx, i.cond, addr, v)?;
    }
    Ok(())
}

/// `IPA Rd, a[addr]`: the interpolation unit's value of a pixel shader input (the address in words
/// in bits 64..71). As on Maxwell the program divides by w itself (NVN2 guide 9.1): a perspective
/// input comes multiplied by `1/w` ([`super::super::io::interpolated`]).
pub(super) fn ipa(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    // At the centroid (1) is at the centre for the single-sampled targets drawn here (A-245).
    if inst.field(76, 2) > 1 {
        return tx.fail("IPA at an offset");
    }
    let addr = inst.field(64, 8) as u32 * 4;
    let v = match tx.cfg.stage {
        Stage::Compute => {
            let bits = attribute_in(tx, addr)?;
            tx.as_f32(bits)
        }
        Stage::Pixel => super::super::io::interpolated(tx, addr)?,
        _ => return tx.fail("IPA outside a pixel shader"),
    };
    write_f(tx, i, dst(i), v);
    Ok(())
}

// ---- global memory -------------------------------------------------------------------------------

/// A 64-bit value from a 32-bit one.
fn wide(tx: &mut Tx<'_>, v: Id) -> Id {
    let u64t = tx.m.ty_int(64, false);
    tx.op1(sop::U_CONVERT, u64t, v)
}

/// `hi:lo`.
fn pair(tx: &mut Tx<'_>, lo: Id, hi: Id) -> Id {
    let u64t = tx.m.ty_int(64, false);
    let l = wide(tx, lo);
    let h = wide(tx, hi);
    let c32 = tx.c32(32);
    let up = tx.op2(sop::SHIFT_LEFT_LOGICAL, u64t, h, c32);
    tx.op2(sop::BITWISE_OR, u64t, up, l)
}

/// The GPU address of a global access: the register (a pair when `reg_pair`), plus the uniform
/// register at `ureg_at` (a pair with bit 72) for the forms that have one, plus the offset.
fn global_address(tx: &mut Tx<'_>, inst: Inst, reg_pair: bool, ureg_at: Option<u32>) -> Id {
    // The storage buffer the access is in: the constant buffer word its base is a pointer from
    // (the uniform register pair, else the register pair), for the host to make reachable (A-258).
    let ubase = ureg_at
        .map(|at| inst.ureg(at))
        .filter(|u| *u != URZ && inst.bit(72));
    // The one of the two that holds a pointer is the base (a uniform register the program does not
    // load holds an offset, as zero does).
    let from_uniform = ubase.and_then(|u| {
        tx.uptr_origin
            .get(&u)
            .or_else(|| tx.ureg_static_pointer.get(&u))
            .copied()
    });
    let from_register = (reg_pair && inst.reg(24) != RZ)
        .then(|| tx.ptr_origin.get(&inst.reg(24)).copied())
        .flatten();
    let base = from_uniform.or(from_register);
    match base {
        Some(b) => {
            if !tx.info.global_bases.contains(&b) {
                tx.info.global_bases.push(b);
                tx.info.global_bases.sort_unstable();
            }
        }
        None => tx.info.global_unknown = true,
    }
    tx.m.capability(cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let r = inst.reg(24);
    let mut addr = if r == RZ {
        tx.m.const_u64(0)
    } else if reg_pair {
        let lo = tx.read(r);
        let hi = tx.read(r.wrapping_add(1));
        pair(tx, lo, hi)
    } else {
        let lo = tx.read(r);
        wide(tx, lo)
    };
    if let Some(at) = ureg_at {
        let u = inst.ureg(at);
        if u != URZ {
            let base = if inst.bit(72) {
                let lo = tx.read_u(u);
                let hi = tx.read_u(u + 1);
                pair(tx, lo, hi)
            } else {
                let lo = tx.read_u(u);
                wide(tx, lo)
            };
            addr = tx.op2(sop::I_ADD, u64t, addr, base);
        }
    }
    let off = tx.m.const_u64(inst.sfield(40, 24) as u64);
    tx.op2(sop::I_ADD, u64t, addr, off)
}

/// `LDG.size Rd, [Ra + URb + offset]`.
pub(super) fn ldg(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let uniform = inst.opcode() == op::LDG_U;
    let gpu = if uniform {
        global_address(tx, inst, inst.bit(90), Some(32))
    } else {
        global_address(tx, inst, inst.bit(72), None)
    };
    let size = inst.field(73, 3);
    let u = tx.t.u32;
    let d = dst(i);
    match words_of(size) {
        Some(words) => {
            let dev = device(tx, gpu);
            for w in 0..words {
                let at = plus(tx, dev, 4 * w);
                let ptr = global_word(tx, at);
                let v =
                    tx.f.emit_result(&mut tx.m, sop::LOAD, u, &[ptr, ALIGNED_4[0], ALIGNED_4[1]]);
                write(tx, i, reg_at(i, d, w), v);
            }
        }
        None => {
            // A byte or a half word: the word that holds it, and its bits.
            let u64t = tx.m.ty_int(64, false);
            let mask = tx.m.const_u64(!3);
            let aligned = tx.op2(sop::BITWISE_AND, u64t, gpu, mask);
            let low = tx.op1(sop::U_CONVERT, u, gpu);
            let three = tx.c32(3);
            let byte = tx.and(low, three);
            let dev = device(tx, aligned);
            let ptr = global_word(tx, dev);
            let word =
                tx.f.emit_result(&mut tx.m, sop::LOAD, u, &[ptr, ALIGNED_4[0], ALIGNED_4[1]]);
            let v = extract_small_dynamic(tx, word, byte, size);
            write(tx, i, d, v);
        }
    }
    Ok(())
}

/// Run `body` only where the guard holds.
pub(super) fn guarded(
    tx: &mut Tx<'_>,
    cond: Option<Id>,
    body: impl FnOnce(&mut Tx<'_>) -> R<()>,
) -> R<()> {
    let Some(c) = cond else {
        return body(tx);
    };
    let then = tx.m.id();
    let merge = tx.m.id();
    tx.f.selection_merge(merge);
    tx.f.branch_conditional(c, then, merge);
    tx.f.label(then);
    body(tx)?;
    tx.f.branch(merge);
    tx.f.label(merge);
    Ok(())
}

/// `STG.size [Ra + URb + offset], Rdata`. A byte or a half word goes into the word that holds it
/// with two atomic operations on that word, since other invocations may be writing its other
/// bytes.
pub(super) fn stg(tx: &mut Tx<'_>, i: &I) -> R<()> {
    tx.info.global_writes = true;
    let inst = i.inst;
    let uniform = inst.opcode() == op::STG_U;
    let size = inst.field(73, 3);
    let gpu = if uniform {
        global_address(tx, inst, inst.bit(90), Some(64))
    } else {
        global_address(tx, inst, inst.bit(72), None)
    };
    let data = inst.reg(32);
    let Some(words) = words_of(size) else {
        if size > 3 {
            return tx.fail(format!("STG of size code {size}"));
        }
        let value = tx.read(data);
        let u64t = tx.m.ty_int(64, false);
        let u = tx.t.u32;
        let three = tx.m.const_u64(3);
        let low = tx.op2(sop::BITWISE_AND, u64t, gpu, three);
        let byte = tx.op1(sop::U_CONVERT, u, low);
        let not_three = tx.m.const_u64(!3);
        let word = tx.op2(sop::BITWISE_AND, u64t, gpu, not_three);
        let dev = device(tx, word);
        let eight = tx.c32(8);
        let shift = tx.imul(byte, eight);
        let field = tx.c32(if size < 2 { 0xFF } else { 0xFFFF });
        let window = tx.shl(field, shift);
        let moved = tx.shl(value, shift);
        let moved = tx.and(moved, window);
        let kept = tx.not(window);
        return guarded(tx, i.cond, |tx| {
            let ptr = global_word(tx, dev);
            let scope = tx.c32(1);
            let relaxed = tx.c32(0);
            tx.f.emit_result(&mut tx.m, sop::ATOMIC_AND, u, &[ptr, scope, relaxed, kept]);
            tx.f.emit_result(&mut tx.m, sop::ATOMIC_OR, u, &[ptr, scope, relaxed, moved]);
            Ok(())
        });
    };
    let values: Vec<Id> = (0..words).map(|w| tx.read(reg_at(i, data, w))).collect();
    let dev = device(tx, gpu);
    guarded(tx, i.cond, |tx| {
        for (w, v) in values.into_iter().enumerate() {
            let at = plus(tx, dev, 4 * w as u32);
            let ptr = global_word(tx, at);
            tx.f.emit(sop::STORE, &[ptr, v, ALIGNED_4[0], ALIGNED_4[1]]);
        }
        Ok(())
    })
}

/// `ATOMG.op.type Rd, [Ra + URb + offset], Rb` and `RED.op.type [Ra + URb + offset], Rb`: an
/// atomic operation on a 32-bit word of global memory, `ATOMG` with the word's old value in `Rd`.
/// NAK's fields: the operation in bits 87..90 (`RED`: 87..89; 0 add, 1 min, 2 max, 3 inc, 4 dec,
/// 5 and, 6 or, 7 xor, 8 exchange), the type in 73..75 (0 u32, 1 s32), the data in 32..39, the
/// address as for `STG` with a uniform register, the register a pair with bit 70 (`ATOMG`) or 90
/// (`RED`), or (opcode 0x3a8) a register alone, a pair with bit 72. The operation is relaxed, at the scope of the device (docs/ASSUMPTIONS.md A-253).
pub(super) fn atomg(tx: &mut Tx<'_>, i: &I) -> R<()> {
    tx.info.global_writes = true;
    let inst = i.inst;
    let red = inst.opcode() == op::RED;
    let ty = inst.field(73, 3);
    let operation = if red {
        inst.field(87, 3)
    } else {
        inst.field(87, 4)
    };
    let signed = match ty {
        0 => false,
        1 => true,
        t => return tx.fail(format!("an atomic operation on global memory of type {t}")),
    };
    let opcode = match (operation, signed) {
        (0, _) => sop::ATOMIC_I_ADD,
        (1, false) => sop::ATOMIC_U_MIN,
        (1, true) => sop::ATOMIC_S_MIN,
        (2, false) => sop::ATOMIC_U_MAX,
        (2, true) => sop::ATOMIC_S_MAX,
        (5, _) => sop::ATOMIC_AND,
        (6, _) => sop::ATOMIC_OR,
        (7, _) => sop::ATOMIC_XOR,
        (8, _) => sop::ATOMIC_EXCHANGE,
        (o, _) => return tx.fail(format!("the atomic operation {o} on global memory")),
    };
    let gpu = match inst.opcode() {
        // `RED` with a uniform base (bit 91) adds the uniform register pair in bits 64..69; without
        // one its address is the register pair (bit 72), as NeXium has it.
        op::RED if inst.bit(91) => global_address(tx, inst, false, Some(64)),
        op::RED => global_address(tx, inst, inst.bit(72), None),
        op::ATOMG_U => global_address(tx, inst, inst.bit(70), Some(64)),
        // The form without a uniform register: the register a pair with bit 72.
        _ => global_address(tx, inst, inst.bit(72), None),
    };
    let value = tx.read(inst.reg(32));
    let dev = device(tx, gpu);
    let d = if red { RZ } else { dst(i) };
    guarded(tx, i.cond, |tx| {
        let ptr = global_word(tx, dev);
        let u = tx.t.u32;
        let scope = tx.c32(1);
        let relaxed = tx.c32(0);
        let old =
            tx.f.emit_result(&mut tx.m, opcode, u, &[ptr, scope, relaxed, value]);
        tx.write_raw(d, old);
        Ok(())
    })
}

// ---- local and shared memory --------------------------------------------------------------

/// Where a local or shared access goes: its byte address, and whether it is a constant (with its
/// value).
fn local_address(tx: &mut Tx<'_>, inst: Inst, ureg_at: Option<u32>) -> Id {
    let r = inst.reg(24);
    let u = ureg_at.map(|at| inst.ureg(at)).filter(|&u| u != URZ);
    // The offset is signed when something is added to it (NeXium).
    let off = if r == RZ && u.is_none() {
        inst.field(40, 24) as u32
    } else {
        inst.sfield(40, 24) as u32
    };
    let mut addr = tx.c32(off);
    if r != RZ {
        let mut base = tx.read(r);
        // LDS and STS: the register scaled by 4, 8 or 16 (bits 78..79).
        if inst.opcode() & 0x1ff == 0x184 || inst.opcode() & 0x1ff == 0x188 {
            let stride = inst.field(78, 2) as u32;
            if stride != 0 {
                let by = tx.c32(stride + 1);
                base = tx.shl(base, by);
            }
        }
        addr = tx.iadd(addr, base);
    }
    if let Some(u) = u {
        let base = tx.read_u(u);
        addr = tx.iadd(addr, base);
    }
    addr
}

/// The uniform register field of a local or shared access, by its opcode.
fn ureg_field(inst: Inst) -> Option<u32> {
    match inst.opcode() {
        op::LDL | op::LDS => inst.bit(91).then_some(32),
        op::STL_U | op::STS_U => Some(64),
        _ => None,
    }
}

/// `LDL`, `LDS`.
pub(super) fn load_local(tx: &mut Tx<'_>, i: &I, space: Space) -> R<()> {
    let inst = i.inst;
    let size = inst.field(73, 3);
    let (var, words) = local::ensure(tx, space)?;
    let addr = local_address(tx, inst, ureg_field(inst));
    let two = tx.c32(2);
    let index = tx.shr(addr, two);
    let u = tx.t.u32;
    let d = dst(i);
    match words_of(size) {
        Some(n) => {
            for w in 0..n {
                let k = tx.c32(w);
                let at = tx.iadd(index, k);
                let ptr = local::word_pointer(tx, space, var, words, at);
                let v = tx.f.load(&mut tx.m, u, ptr);
                write(tx, i, reg_at(i, d, w), v);
            }
        }
        None if size < 4 => {
            let ptr = local::word_pointer(tx, space, var, words, index);
            let word = tx.f.load(&mut tx.m, u, ptr);
            let three = tx.c32(3);
            let byte = tx.and(addr, three);
            let v = extract_small_dynamic(tx, word, byte, size);
            write(tx, i, d, v);
        }
        None => return tx.fail(format!("a local load of size code {size}")),
    }
    Ok(())
}

/// `STL`, `STS`.
pub(super) fn store_local(tx: &mut Tx<'_>, i: &I, space: Space) -> R<()> {
    let inst = i.inst;
    let size = inst.field(73, 3);
    let (var, words) = local::ensure(tx, space)?;
    let addr = local_address(tx, inst, ureg_field(inst));
    let two = tx.c32(2);
    let index = tx.shr(addr, two);
    let data = inst.reg(32);
    let u = tx.t.u32;
    match words_of(size) {
        Some(n) => {
            let values: Vec<Id> = (0..n).map(|w| tx.read(reg_at(i, data, w))).collect();
            guarded(tx, i.cond, |tx| {
                for (w, v) in values.into_iter().enumerate() {
                    let k = tx.c32(w as u32);
                    let at = tx.iadd(index, k);
                    let ptr = local::word_pointer(tx, space, var, words, at);
                    tx.f.store(ptr, v);
                }
                Ok(())
            })
        }
        None if size < 4 => {
            // A byte or a half word, put into the word that holds it.
            let value = tx.read(data);
            guarded(tx, i.cond, |tx| {
                let ptr = local::word_pointer(tx, space, var, words, index);
                let old = tx.f.load(&mut tx.m, u, ptr);
                let three = tx.c32(3);
                let byte = tx.and(addr, three);
                let eight = tx.c32(8);
                let shift = tx.imul(byte, eight);
                let bits = tx.c32(if size < 2 { 8 } else { 16 });
                let new = tx.f.emit_result(
                    &mut tx.m,
                    sop::BIT_FIELD_INSERT,
                    u,
                    &[old, value, shift, bits],
                );
                tx.f.store(ptr, new);
                Ok(())
            })
        }
        None => tx.fail(format!("a local store of size code {size}")),
    }
}

/// The bytes past the end of the constant-offset accesses of `space` an instruction makes, for
/// the size of the memory.
pub(super) fn constant_extent(inst: Inst, space: Space) -> Option<u32> {
    let ours = match space {
        Space::Local => matches!(inst.opcode(), op::LDL | op::STL | op::STL_U),
        Space::Shared => matches!(inst.opcode(), op::LDS | op::STS | op::STS_U),
    };
    if !ours || inst.reg(24) != RZ || ureg_field(inst).is_some_and(|at| inst.ureg(at) != URZ) {
        return None;
    }
    let bytes = match inst.field(73, 3) {
        5 => 8,
        6 => 16,
        _ => 4,
    };
    Some(inst.field(40, 24) as u32 + bytes)
}

// ---- barriers ---------------------------------------------------------------------------------

/// `BAR.SYNC`: every invocation of the workgroup waits for the others (compute programs).
pub(super) fn bar(tx: &mut Tx<'_>, i: &I) -> R<()> {
    if !matches!(tx.cfg.stage, Stage::Kernel | Stage::Compute) {
        return tx.fail("BAR in a graphics program");
    }
    if i.cond.is_some() || i.inst.field(32, 64) != 0 {
        return tx.fail("BAR in a form other than BAR.SYNC 0 for every thread");
    }
    // Workgroup execution and memory scope, acquire-release of workgroup memory.
    let scope = tx.c32(2);
    let semantics = tx.c32(0x108);
    tx.f.emit(sop::CONTROL_BARRIER, &[scope, scope, semantics]);
    Ok(())
}

/// `MEMBAR.scope`: the accesses before it are seen before the ones after it. Within one invocation
/// of a graphics program that is already so.
pub(super) fn membar(tx: &mut Tx<'_>, i: &I) -> R<()> {
    if !matches!(tx.cfg.stage, Stage::Kernel | Stage::Compute) {
        return Ok(());
    }
    // CTA: the workgroup (2); otherwise the device (1). Acquire-release of uniform, workgroup and
    // image memory.
    let scope = tx.c32(if i.inst.field(76, 3) == 0 { 2 } else { 1 });
    let semantics = tx.c32(0x8 | 0x40 | 0x100 | 0x800);
    tx.f.emit(sop::MEMORY_BARRIER, &[scope, semantics]);
    Ok(())
}
