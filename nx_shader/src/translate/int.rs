//! Maxwell's multiply-add family and the bit manipulation instructions: `XMAD`, `IMAD`, `LEA`,
//! `LOP3`, `SHF` and `PRMT`.
//!
//! `XMAD` is how the shader compiler multiplies 32-bit integers on this architecture: three of
//! them (with the `.MRG`, `.PSL` and `.CBCC` flags) make one product. The semantics follow what
//! the compiler's sequences need; the carry flag forms and a few modes are refused
//! (docs/ASSUMPTIONS.md).

use super::alu::{add_carry, no_cc, set_add_flags, src_b};
use super::emit::{Tx, R};
use crate::inst::{Inst, PT};
use crate::ops::Op;
use crate::spirv::{cap, op, Id};

pub(super) fn translate(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<bool> {
    match opc {
        Op::XmadReg | Op::Xmad | Op::XmadCbuf_4e00 | Op::XmadCbuf_5100 => xmad(tx, inst, opc)?,
        Op::ImadReg | Op::ImadImm20 | Op::ImadCbuf_4a00 | Op::ImadCbuf_5200 => imad(tx, inst, opc)?,
        Op::LeaReg | Op::LeaCbuf | Op::LeaImm20 => lea(tx, inst)?,
        Op::Iadd3Reg | Op::Iadd3Cbuf | Op::Iadd3Imm20 => iadd3(tx, inst, opc)?,
        Op::LeaHiReg | Op::LeaHiCbuf => lea_hi(tx, inst, opc)?,
        Op::Lop3LutReg | Op::Lop3LutCbuf | Op::Lop3LutImm20 => lop3(tx, inst, opc)?,
        Op::ShfRReg | Op::ShfR | Op::ShfLReg | Op::ShfLImm20 => shf(tx, inst, opc)?,
        Op::PrmtReg | Op::PrmtCbuf_4bc0 | Op::PrmtCbuf_53c0 | Op::PrmtImm20 => prmt(tx, inst, opc)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// A 16-bit half of a register, sign- or zero-extended to 32 bits.
fn half(tx: &mut Tx<'_>, v: Id, high: bool, signed: bool) -> Id {
    let offset = tx.c32(if high { 16 } else { 0 });
    let width = tx.c32(16);
    if signed {
        let s = tx.as_i32(v);
        let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, offset, width);
        tx.as_u32(e)
    } else {
        tx.op3(op::BIT_FIELD_U_EXTRACT, tx.t.u32, v, offset, width)
    }
}

/// `XMAD.mode Rd, Ra.H?, B.H?, C`: the product of two 16-bit halves plus a 32-bit addend.
///
/// * `.PSL` shifts the product left by 16 first;
/// * the modes choose the addend: none takes `C`, `CLO` and `CHI` its low or high half and `CBCC`
///   adds `B << 16` to it (`CSFU` is refused);
/// * `.MRG` then replaces the high half of the result by the low half of `B`.
fn xmad(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    no_cc(tx, inst, 47)?;
    let (b, c, mode, half_b, psl, mrg, x) = match opc {
        Op::XmadReg => (
            tx.read(inst.rb()),
            tx.read(inst.rc()),
            inst.bits(50, 3),
            inst.bit(35),
            inst.bit(36),
            inst.bit(37),
            inst.bit(38),
        ),
        Op::Xmad => (
            tx.c32(inst.bits(20, 16) as u32),
            tx.read(inst.rc()),
            inst.bits(50, 3),
            false,
            inst.bit(36),
            inst.bit(37),
            inst.bit(38),
        ),
        Op::XmadCbuf_4e00 => (
            src_b(tx, inst)?,
            tx.read(inst.rc()),
            inst.bits(50, 2),
            inst.bit(52),
            inst.bit(55),
            inst.bit(56),
            inst.bit(54),
        ),
        _ => (
            tx.read(inst.rc()),
            src_b(tx, inst)?,
            inst.bits(50, 2),
            inst.bit(52),
            false,
            false,
            inst.bit(54),
        ),
    };
    if x {
        return tx.fail("XMAD with the extended (carry) flag");
    }
    let a = tx.read(inst.ra());
    let op_a = half(tx, a, inst.bit(53), inst.bit(48));
    let op_b = half(tx, b, half_b, inst.bit(49));
    let mut product = tx.imul(op_a, op_b);
    if psl {
        let sixteen = tx.c32(16);
        product = tx.shl(product, sixteen);
    }
    let sixteen = tx.c32(16);
    let op_c = match mode {
        0 => c,
        1 => half(tx, c, false, false),
        2 => half(tx, c, true, false),
        4 => {
            let shifted = tx.shl(b, sixteen);
            tx.iadd(shifted, c)
        }
        _ => return tx.fail("XMAD with the CSFU mode"),
    };
    let mut result = tx.iadd(product, op_c);
    if mrg {
        let low_mask = tx.c32(0xFFFF);
        let kept = tx.and(result, low_mask);
        let b_low = tx.and(b, low_mask);
        let b_high = tx.shl(b_low, sixteen);
        result = tx.or(kept, b_high);
    }
    tx.write(inst, inst.rd(), result);
    Ok(())
}

/// `IMAD Rd, Ra, B, C`: the low 32 bits of `Ra * B + C`, with either term negated by bits 51
/// and 52. The high, saturating and extended forms are refused.
fn imad(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bit(54) || inst.bit(50) || inst.bit(49) {
        return tx.fail("IMAD with the high, saturate or extended flag");
    }
    let (b, c) = match opc {
        Op::ImadReg => (tx.read(inst.rb()), tx.read(inst.rc())),
        Op::ImadImm20 | Op::ImadCbuf_4a00 => (src_b(tx, inst)?, tx.read(inst.rc())),
        _ => (tx.read(inst.rc()), src_b(tx, inst)?),
    };
    let a = tx.read(inst.ra());
    let mut product = tx.imul(a, b);
    if inst.bit(51) {
        product = tx.ineg(product);
    }
    let c = if inst.bit(52) { tx.ineg(c) } else { c };
    let r = tx.iadd(product, c);
    tx.write(inst, inst.rd(), r);
    Ok(())
}

/// `a + b`, with the carry in (for `.X`) and out (for `.CC`) of the flag the integer adds share.
fn add_flagged(tx: &mut Tx<'_>, inst: Inst, a: Id, b: Id, x: bool, cc: bool) -> Id {
    if !x && !cc {
        return tx.iadd(a, b);
    }
    let var = tx.carry_variable();
    let bool_t = tx.t.bool_;
    let previous = x.then(|| tx.f.load(&mut tx.m, bool_t, var));
    let (sum, carry) = add_carry(tx, a, b, previous);
    if cc {
        set_add_flags(tx, inst, a, b, sum, carry, x);
    }
    sum
}

/// `LEA Rd, [-]Ra, B, scale`: `B + (Ra << scale)`.
fn lea(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.bits(48, 3) != u64::from(PT) {
        return tx.fail("LEA with a predicate destination");
    }
    let mut a = tx.read(inst.ra());
    if inst.bit(45) {
        a = tx.ineg(a);
    }
    let scale = tx.c32(inst.bits(39, 5) as u32);
    let shifted = tx.shl(a, scale);
    let b = src_b(tx, inst)?;
    let r = add_flagged(tx, inst, b, shifted, inst.bit(46), inst.bit(47));
    tx.write(inst, inst.rd(), r);
    Ok(())
}

/// `LEA.HI Rd, [-]Ra, B, Rc, scale`: `B` plus the low word of the 64-bit value `Rc:Ra` shifted
/// right by `32 - scale` (the high part of `(Rc:Ra) << scale`).
fn lea_hi(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    if inst.bits(48, 3) != u64::from(PT) {
        return tx.fail("LEA.HI with a predicate destination");
    }
    let (scale, neg, x) = if opc == Op::LeaHiReg {
        (inst.bits(28, 5), inst.bit(37), inst.bit(38))
    } else {
        (inst.bits(51, 5), inst.bit(56), inst.bit(57))
    };
    tx.m.capability(cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let lo = tx.read(inst.ra());
    let hi = tx.read(inst.rc());
    let lo64 = tx.op1(op::U_CONVERT, u64t, lo);
    let hi64 = tx.op1(op::U_CONVERT, u64t, hi);
    let c32 = tx.c32(32);
    let hi_up = tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, hi64, c32);
    let mut packed = tx.op2(op::BITWISE_OR, u64t, lo64, hi_up);
    if neg {
        packed = tx.op1(op::S_NEGATE, u64t, packed);
    }
    let amount = tx.c32(32 - scale as u32);
    let moved = tx.op2(op::SHIFT_RIGHT_LOGICAL, u64t, packed, amount);
    let word = tx.op1(op::U_CONVERT, tx.t.u32, moved);
    let b = if opc == Op::LeaHiReg {
        tx.read(inst.rb())
    } else {
        src_b(tx, inst)?
    };
    let r = add_flagged(tx, inst, b, word, x, inst.bit(47));
    tx.write(inst, inst.rd(), r);
    Ok(())
}

/// `LOP3.LUT Rd, Ra, B, Rc, lut`: every result bit is bit `4a + 2b + c` of the 8-bit table.
fn lop3(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let (lut, x, mode) = match opc {
        Op::Lop3LutReg => (inst.bits(28, 8), inst.bit(38), inst.bits(36, 2)),
        Op::Lop3LutCbuf => (inst.bits(48, 8), inst.bit(56), 0),
        _ => (inst.bits(48, 8), false, 0),
    };
    if x || inst.bit(47) {
        return tx.fail("LOP3 with the extended or condition code flag");
    }
    let a = tx.read(inst.ra());
    let b = match opc {
        Op::Lop3LutReg => tx.read(inst.rb()),
        _ => src_b(tx, inst)?,
    };
    let c = tx.read(inst.rc());
    let na = tx.not(a);
    let nb = tx.not(b);
    let nc = tx.not(c);
    let mut result = tx.c32(0);
    for k in 0..8u64 {
        if lut >> k & 1 == 0 {
            continue;
        }
        let ta = if k & 4 != 0 { a } else { na };
        let tb = if k & 2 != 0 { b } else { nb };
        let tc = if k & 1 != 0 { c } else { nc };
        let ab = tx.and(ta, tb);
        let term = tx.and(ab, tc);
        result = tx.or(result, term);
    }
    tx.write(inst, inst.rd(), result);
    if opc == Op::Lop3LutReg {
        let p = inst.bits(48, 3) as u8;
        if p != PT {
            let zero = tx.c32(0);
            let value = match mode {
                0 => tx.cfalse(),
                1 => tx.ctrue(),
                2 => tx.cmp(op::I_EQUAL, result, zero),
                _ => tx.cmp(op::I_NOT_EQUAL, result, zero),
            };
            tx.write_pred(inst, p, value);
        }
    }
    Ok(())
}

/// `SHF.{L|R}[.W][.U64|.S64][.HI] Rd, Ra, shift, Rc`: a funnel shift of the 64-bit value `Rc:Ra`.
/// A right shift gives the low word (the high word with `.HI`), a left shift the high word.
fn shf(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bit(49) {
        return tx.fail("SHF with the extended flag");
    }
    let right = matches!(opc, Op::ShfRReg | Op::ShfR);
    let kind = inst.bits(37, 2);
    let wide = kind >= 2;
    let signed = kind == 3;
    let dst_high = inst.bit(48);
    let wrap = inst.bit(50);
    let shift = match opc {
        Op::ShfRReg | Op::ShfLReg => tx.read(inst.rb()),
        Op::ShfR => tx.c32(inst.bits(20, 6) as u32),
        _ => tx.c32(inst.imm20() as u32),
    };
    tx.m.capability(cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let bits = if wide { 64 } else { 32 };
    // The amount: wrapped to the width, or clamped to it.
    let amount = if wrap {
        let mask = tx.c32(bits - 1);
        tx.and(shift, mask)
    } else {
        let limit = tx.c32(bits);
        tx.ext(tx.t.u32, crate::spirv::glsl::U_MIN, &[shift, limit])
    };
    let low = tx.read(inst.ra());
    let high = tx.read(inst.rc());
    let lo64 = tx.op1(op::U_CONVERT, u64t, low);
    let hi64 = tx.op1(op::U_CONVERT, u64t, high);
    let c32 = tx.c32(32);
    let hi_up = tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, hi64, c32);
    let x = tx.op2(op::BITWISE_OR, u64t, lo64, hi_up);
    // A shift by 64 (the clamped maximum of the wide forms) is not defined for the shift
    // instructions, so it is made from a shift by 63 and a selection.
    let sixty_three = tx.c32(63);
    let limited = tx.ext(tx.t.u32, crate::spirv::glsl::U_MIN, &[amount, sixty_three]);
    let sixty_four = tx.c32(64);
    let full = tx.cmp(op::U_GREATER_THAN_EQUAL, amount, sixty_four);
    let zero64 = tx.m.const_u64(0);
    let shifted = if right && signed {
        tx.op2(op::SHIFT_RIGHT_ARITHMETIC, u64t, x, limited)
    } else {
        let raw = if right {
            tx.op2(op::SHIFT_RIGHT_LOGICAL, u64t, x, limited)
        } else {
            tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, x, limited)
        };
        tx.select(u64t, full, zero64, raw)
    };
    let word = if !right || dst_high {
        let moved = tx.op2(op::SHIFT_RIGHT_LOGICAL, u64t, shifted, c32);
        tx.op1(op::U_CONVERT, tx.t.u32, moved)
    } else {
        tx.op1(op::U_CONVERT, tx.t.u32, shifted)
    };
    tx.write(inst, inst.rd(), word);
    Ok(())
}

/// Byte `index` (0..7, the bytes of `a` then those of `b`) as a value 0..255.
fn permute_byte(tx: &mut Tx<'_>, a: Id, b: Id, index: Id) -> Id {
    let four = tx.c32(4);
    let three = tx.c32(3);
    let from_a = tx.cmp(op::U_LESS_THAN, index, four);
    let word = tx.select_u32(from_a, a, b);
    let lane = tx.and(index, three);
    let eight = tx.c32(8);
    let shift = tx.imul(lane, eight);
    let moved = tx.shr(word, shift);
    let mask = tx.c32(0xFF);
    tx.and(moved, mask)
}

/// `PRMT.mode Rd, Ra, SEL, Rc`: the bytes of the result are picked from the eight bytes of `Ra`
/// and `Rc` (`Ra` first). Default mode: each nibble of `SEL` names a byte (low 3 bits) and says
/// whether to replace it by its sign (bit 3). The extract and replicate modes derive the four
/// picks from the low two bits of `SEL`. The edge clamp modes and `RC16` are refused.
fn prmt(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let mode = inst.bits(48, 3);
    let a = tx.read(inst.ra());
    let (sel, b) = match opc {
        Op::PrmtReg => (tx.read(inst.rb()), tx.read(inst.rc())),
        Op::PrmtCbuf_4bc0 | Op::PrmtImm20 => (src_b(tx, inst)?, tx.read(inst.rc())),
        _ => (tx.read(inst.rc()), src_b(tx, inst)?),
    };
    if mode > 3 {
        return tx.fail(format!("PRMT in mode {mode}"));
    }
    let mut result = tx.c32(0);
    let seven = tx.c32(7);
    let one = tx.c32(1);
    let three = tx.c32(3);
    for i in 0..4u32 {
        let (index, replicate_sign) = match mode {
            0 => {
                let shift = tx.c32(4 * i);
                let nibble = tx.shr(sel, shift);
                let index = tx.and(nibble, seven);
                let sign_flag = tx.shr(nibble, three);
                let sign_flag = tx.and(sign_flag, one);
                (index, Some(sign_flag))
            }
            1 => {
                let n = tx.and(sel, three);
                let offset = tx.c32(i);
                let sum = tx.iadd(n, offset);
                (tx.and(sum, seven), None)
            }
            2 => {
                let n = tx.and(sel, three);
                let offset = tx.c32(i);
                let diff = tx.isub(n, offset);
                (tx.and(diff, seven), None)
            }
            _ => (tx.and(sel, three), None),
        };
        let mut byte = permute_byte(tx, a, b, index);
        if let Some(flag) = replicate_sign {
            // Bit 3 of the nibble: every bit of the byte becomes the sign bit of the selected one.
            let seven_shift = tx.c32(7);
            let top = tx.shr(byte, seven_shift);
            let top = tx.and(top, one);
            let zero = tx.c32(0);
            let negative = tx.cmp(op::I_NOT_EQUAL, top, zero);
            let ff = tx.c32(0xFF);
            let spread = tx.select_u32(negative, ff, zero);
            let want = tx.cmp(op::I_NOT_EQUAL, flag, zero);
            byte = tx.select_u32(want, spread, byte);
        }
        let place = tx.c32(8 * i);
        let placed = tx.shl(byte, place);
        result = tx.or(result, placed);
    }
    tx.write(inst, inst.rd(), result);
    Ok(())
}

/// A word cut to its low half (selector 1) or its high half (2), else whole.
fn half_of(tx: &mut Tx<'_>, v: Id, selector: u64) -> Id {
    match selector {
        1 => {
            let mask = tx.c32(0xFFFF);
            tx.and(v, mask)
        }
        2 => {
            let sixteen = tx.c32(16);
            tx.shr(v, sixteen)
        }
        _ => v,
    }
}

/// `IADD3 Rd, [-]Ra, [-]B, [-]Rc`: the sum of three words (negated by bits 51, 50 and 49). In the
/// register form each word can first be cut to its low (1) or high (2) half (bits 35..36, 33..34
/// and 31..32 for `Ra`, `B` and `Rc`), and bits 37..38 move the sum of the first two right (1) or
/// left (2) by 16 before the third is added. `.X` (bit 48) adds the carry flag and `.CC` (bit 47)
/// sets the carry (any add carried), zero (and, with `.X`, was zero before) and sign flags
/// (docs/ASSUMPTIONS.md A-107).
fn iadd3(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let register_form = opc == Op::Iadd3Reg;
    let mut a = tx.read(inst.ra());
    let mut b = src_b(tx, inst)?;
    let mut c = tx.read(inst.rc());
    if register_form {
        a = half_of(tx, a, inst.bits(35, 2));
        b = half_of(tx, b, inst.bits(33, 2));
        c = half_of(tx, c, inst.bits(31, 2));
    }
    if inst.bit(51) {
        a = tx.ineg(a);
    }
    if inst.bit(50) {
        b = tx.ineg(b);
    }
    if inst.bit(49) {
        c = tx.ineg(c);
    }
    let shift = if register_form { inst.bits(37, 2) } else { 0 };
    let (extended, cc) = (inst.bit(48), inst.bit(47));
    let sixteen = tx.c32(16);
    if !extended && !cc {
        let mut ab = tx.iadd(a, b);
        ab = match shift {
            1 => tx.shr(ab, sixteen),
            2 => tx.shl(ab, sixteen),
            _ => ab,
        };
        let r = tx.iadd(ab, c);
        tx.write(inst, inst.rd(), r);
        return Ok(());
    }
    let flags = tx.flags();
    let carry_before = extended.then(|| tx.read_flag(flags.c));
    let zero_before = extended.then(|| tx.read_flag(flags.z));
    // A carry out of the first add is lost when the sum is shifted.
    let (ab, carry0) = match shift {
        1 | 2 => {
            let (sum, _) = add_carry(tx, a, b, None);
            let moved = if shift == 1 {
                tx.shr(sum, sixteen)
            } else {
                tx.shl(sum, sixteen)
            };
            (moved, tx.cfalse())
        }
        _ => add_carry(tx, a, b, None),
    };
    let (abc, carry1) = add_carry(tx, ab, c, None);
    let zero = tx.c32(0);
    let (sum, carry2) = match carry_before {
        Some(before) => add_carry(tx, abc, zero, Some(before)),
        None => (abc, tx.cfalse()),
    };
    tx.write(inst, inst.rd(), sum);
    if cc {
        let carried = tx.lor(carry0, carry1);
        let carried = tx.lor(carried, carry2);
        tx.write_flag(inst, flags.c, carried);
        let mut is_zero = tx.cmp(op::I_EQUAL, sum, zero);
        if let Some(before) = zero_before {
            is_zero = tx.land(is_zero, before);
        }
        tx.write_flag(inst, flags.z, is_zero);
        let thirty_one = tx.c32(31);
        let top = tx.shr(sum, thirty_one);
        let negative = tx.cmp(op::I_NOT_EQUAL, top, zero);
        tx.write_flag(inst, flags.s, negative);
    }
    Ok(())
}
