//! Floating point: `FADD`, `FMUL`, `FFMA`, `FMNMX`, `FSEL`, `FSET`, `FSETP`, `MUFU`, `FRND` and the
//! conversions `F2I`, `I2F` (`I2FP`), `F2F` and `F2FP`.
//!
//! Field positions are NAK's (`sm70_encode.rs`): saturation in bit 77, the rounding mode in bits
//! 78..79 (0 to nearest even, 1 down, 2 up, 3 towards zero), flush to zero in bit 80, the
//! multiply-by-zero rule (`.FMZ`) in bit 76, `FMUL`'s power-of-two result scale in bits 84..86.

use super::super::alu::{bool_op, fcompare, fmz_product, fsat, round_float, set_value};
use super::super::emit::{Tx, R};
use super::{a, b, bc, dst, float, operand, pred, write, write_f, write_pred, I};
use crate::spirv::{glsl, op};

/// The rounding modes other than to nearest even are taken as it: a mode of one instruction would
/// need one for every float operation of the module, and the results are at most one unit in the
/// last place apart (docs/ASSUMPTIONS.md A-246).
fn round_to_nearest(_tx: &Tx<'_>, _i: &I, _what: &str) -> R<()> {
    Ok(())
}

fn saturate(tx: &mut Tx<'_>, i: &I, v: crate::spirv::Id) -> crate::spirv::Id {
    if i.inst.bit(77) {
        fsat(tx, v)
    } else {
        v
    }
}

/// `FADD Rd, ±|Ra|, ±|B|`.
pub(super) fn fadd(tx: &mut Tx<'_>, i: &I) -> R<()> {
    round_to_nearest(tx, i, "FADD")?;
    let x = float(tx, a(i))?;
    let bs = b(tx, i)?;
    let y = float(tx, bs)?;
    let r = tx.fadd(x, y);
    let r = saturate(tx, i, r);
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// `FMUL Rd, ±|Ra|, ±|B|`, scaled by 2^(n - 4) for the code `n` of bits 84..86.
pub(super) fn fmul(tx: &mut Tx<'_>, i: &I) -> R<()> {
    round_to_nearest(tx, i, "FMUL")?;
    let x = float(tx, a(i))?;
    let bs = b(tx, i)?;
    let y = float(tx, bs)?;
    let mut r = tx.fmul(x, y);
    if i.inst.bit(76) {
        r = fmz_product(tx, x, y, r);
    }
    match i.inst.field(84, 3) {
        4 => {}
        0 => return tx.fail("FMUL with result scale code 0"),
        n => {
            let factor = tx.cf32(2f32.powi(n as i32 - 4));
            r = tx.fmul(r, factor);
        }
    }
    let r = saturate(tx, i, r);
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// `FFMA Rd, ±|Ra|, ±|B|, ±|C|`: fused.
pub(super) fn ffma(tx: &mut Tx<'_>, i: &I) -> R<()> {
    round_to_nearest(tx, i, "FFMA")?;
    let x = float(tx, a(i))?;
    let (bs, cs) = bc(tx, i)?;
    let y = float(tx, bs)?;
    let z = float(tx, cs)?;
    let mut r = tx.fma(x, y, z);
    if i.inst.bit(76) {
        // With a zero factor the product is zero and only the addend is left.
        let zero = tx.cf32(0.0);
        let product = fmz_product(tx, x, y, zero);
        let zero_case = tx.fadd(product, z);
        let xz = tx.cmp(op::F_ORD_EQUAL, x, zero);
        let yz = tx.cmp(op::F_ORD_EQUAL, y, zero);
        let either = tx.lor(xz, yz);
        let f = tx.t.f32;
        r = tx.select(f, either, zero_case, r);
    }
    let r = saturate(tx, i, r);
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// `FMNMX Rd, ±|Ra|, ±|B|, Pc`: the minimum where the predicate holds, else the maximum (a NaN
/// operand gives the other).
pub(super) fn fmnmx(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let x = float(tx, a(i))?;
    let bs = b(tx, i)?;
    let y = float(tx, bs)?;
    let want_min = pred(tx, i.inst.pred(87, 90, i.uniform));
    let f = tx.t.f32;
    let lo = tx.ext(f, glsl::N_MIN, &[x, y]);
    let hi = tx.ext(f, glsl::N_MAX, &[x, y]);
    let r = tx.select(f, want_min, lo, hi);
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// `FSEL Rd, ±|Ra|, ±|B|, Pc`.
pub(super) fn fsel(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let x = float(tx, a(i))?;
    let bs = b(tx, i)?;
    let y = float(tx, bs)?;
    let p = pred(tx, i.inst.pred(87, 90, i.uniform));
    let f = tx.t.f32;
    let r = tx.select(f, p, x, y);
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// The comparison of `FSET`/`FSETP` (bits 76..79) of the two operands.
fn compare(tx: &mut Tx<'_>, i: &I) -> R<crate::spirv::Id> {
    let x = float(tx, a(i))?;
    let bs = b(tx, i)?;
    let y = float(tx, bs)?;
    fcompare(tx, i.inst.field(76, 4), x, y)
}

/// `FSET.cmp.bop Rd, Ra, B, Pc`: 1.0 where the comparison combined with the predicate holds, else
/// 0 (docs/ASSUMPTIONS.md A-242).
pub(super) fn fset(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let c = compare(tx, i)?;
    let p = pred(tx, i.inst.pred(87, 90, i.uniform));
    let r = bool_op(tx, i.inst.field(74, 2), c, p)?;
    let v = set_value(tx, r, true);
    write(tx, i, dst(i), v);
    Ok(())
}

/// `FSETP.cmp.bop Pu, Pv, Ra, B, Pc`: `Pu` the comparison combined with the predicate, `Pv` its
/// negation combined the same way.
pub(super) fn fsetp(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let c = compare(tx, i)?;
    let p = pred(tx, i.inst.pred(87, 90, i.uniform));
    let bop = i.inst.field(74, 2);
    let first = bool_op(tx, bop, c, p)?;
    let nc = tx.lnot(c);
    let second = bool_op(tx, bop, nc, p)?;
    let (u, v) = (
        i.inst.pred_dst(81, i.uniform),
        i.inst.pred_dst(84, i.uniform),
    );
    write_pred(tx, i, u, first);
    write_pred(tx, i, v, second);
    Ok(())
}

/// `MUFU.op Rd, ±|B|`: the special functions. `SIN` and `COS` take the angle in turns, not radians
/// (NVN2 guide 9.1; NeXium).
pub(super) fn mufu(tx: &mut Tx<'_>, i: &I) -> R<()> {
    if i.inst.bit(73) {
        return tx.fail("MUFU of a half-precision value");
    }
    let bs = b(tx, i)?;
    let x = float(tx, bs)?;
    let f = tx.t.f32;
    let r = match i.inst.field(74, 6) {
        code @ (0 | 1) => {
            let turn = tx.cf32(std::f32::consts::TAU);
            let radians = tx.fmul(x, turn);
            let function = if code == 0 { glsl::COS } else { glsl::SIN };
            tx.ext(f, function, &[radians])
        }
        2 => tx.ext(f, glsl::EXP2, &[x]),
        3 => tx.ext(f, glsl::LOG2, &[x]),
        4 => {
            let one = tx.cf32(1.0);
            tx.op2(op::F_DIV, f, one, x)
        }
        5 => tx.ext(f, glsl::INVERSE_SQRT, &[x]),
        8 => tx.ext(f, glsl::SQRT, &[x]),
        9 => tx.ext(f, glsl::TANH, &[x]),
        code => {
            return tx.fail(format!(
                "MUFU operation {code} (the halves of a double precision one)"
            ))
        }
    };
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// The sizes of a conversion: (destination, source), as the log2 of their bytes (bits 75..76 and
/// 84..85).
fn sizes(i: &I) -> (u64, u64) {
    (i.inst.field(75, 2), i.inst.field(84, 2))
}

/// `FRND.rnd Rd, ±|B|`: a 32-bit float rounded to an integral value.
pub(super) fn frnd(tx: &mut Tx<'_>, i: &I) -> R<()> {
    if sizes(i) != (2, 2) {
        return tx.fail("FRND of a float that is not 32 bits");
    }
    let bs = b(tx, i)?;
    let x = float(tx, bs)?;
    let r = round_float(tx, x, i.inst.field(78, 2));
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// `F2I.type.rnd Rd, ±|B|`: a 32-bit float rounded and converted to an integer of 8, 16 or 32 bits
/// (bit 72 signed), saturating; NaN gives zero.
pub(super) fn f2i(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let (dsize, ssize) = sizes(i);
    if ssize != 2 {
        return tx.fail("F2I from a float that is not 32 bits");
    }
    if dsize == 3 {
        return tx.fail("F2I to a 64-bit integer");
    }
    let signed = i.inst.bit(72);
    let bs = b(tx, i)?;
    let x = float(tx, bs)?;
    let rounded = round_float(tx, x, i.inst.field(78, 2));
    let r = float_to_int(tx, rounded, dsize, signed);
    write(tx, i, dst(i), r);
    Ok(())
}

/// A rounded float to an integer of 2^`size` bytes, saturating to its range; NaN gives zero. The
/// result is the integer in a 32-bit register (sign-extended for the signed types).
pub(super) fn float_to_int(
    tx: &mut Tx<'_>,
    rounded: crate::spirv::Id,
    size: u64,
    signed: bool,
) -> crate::spirv::Id {
    let f = tx.t.f32;
    let (lo, hi) = match (size, signed) {
        (0, false) => (0.0, 255.0),
        (0, true) => (-128.0, 127.0),
        (1, false) => (0.0, 65535.0),
        (1, true) => (-32768.0, 32767.0),
        (_, false) => (0.0, 4294967296.0),
        (_, true) => (-2147483648.0, 2147483648.0),
    };
    let nan = tx.op1(op::IS_NAN, tx.t.bool_, rounded);
    let zero = tx.c32(0);
    if size < 2 {
        let lo_c = tx.cf32(lo);
        let hi_c = tx.cf32(hi);
        let clamped = tx.ext(f, glsl::F_CLAMP, &[rounded, lo_c, hi_c]);
        let r = if signed {
            let it = tx.t.i32;
            let c = tx.op1(op::CONVERT_F_TO_S, it, clamped);
            tx.as_u32(c)
        } else {
            let u = tx.t.u32;
            tx.op1(op::CONVERT_F_TO_U, u, clamped)
        };
        return tx.select_u32(nan, zero, r);
    }
    // 32 bits: the range ends just below 2^31 (or 2^32); the conversion of a value in it is
    // defined, the ends saturate.
    let lo_c = tx.cf32(lo);
    let hi_c = tx.cf32(hi);
    let below = tx.cmp(op::F_ORD_LESS_THAN_EQUAL, rounded, lo_c);
    let above = tx.cmp(op::F_ORD_GREATER_THAN_EQUAL, rounded, hi_c);
    let top = super::super::alu::tx_prev_float(tx, hi);
    let safe = tx.ext(f, glsl::F_CLAMP, &[rounded, lo_c, top]);
    let (min, max, conv) = if signed {
        let it = tx.t.i32;
        let c = tx.op1(op::CONVERT_F_TO_S, it, safe);
        (tx.c32(0x8000_0000), tx.c32(0x7FFF_FFFF), tx.as_u32(c))
    } else {
        let u = tx.t.u32;
        let c = tx.op1(op::CONVERT_F_TO_U, u, safe);
        (tx.c32(0), tx.c32(0xFFFF_FFFF), c)
    };
    let r = tx.select_u32(above, max, conv);
    let r = tx.select_u32(below, min, r);
    tx.select_u32(nan, zero, r)
}

/// `I2F.F32.type Rd, B` and `I2FP.F32.type Rd, B`: an integer of 8, 16 or 32 bits (bit 74
/// signed) to a 32-bit float; a narrow source is the byte or half word bits 60..61 pick.
pub(super) fn i2f(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let (dsize, ssize) = sizes(i);
    if dsize != 2 {
        return tx.fail("I2F to a float that is not 32 bits");
    }
    if ssize == 3 {
        return tx.fail("I2F from a 64-bit integer");
    }
    let signed = i.inst.bit(74);
    let bs = b(tx, i)?;
    let raw = operand(tx, bs.operand)?;
    let mut v = raw;
    if ssize < 2 {
        let selector = if matches!(i.inst.form(), 2 | 4) {
            0
        } else {
            i.inst.field(60, 2) as u32
        };
        let width = if ssize == 0 { 8 } else { 16 };
        let offset = tx.c32(selector * width);
        let len = tx.c32(width);
        v = if signed {
            let s = tx.as_i32(raw);
            let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, offset, len);
            tx.as_u32(e)
        } else {
            let u = tx.t.u32;
            tx.op3(op::BIT_FIELD_U_EXTRACT, u, raw, offset, len)
        };
    }
    let f = tx.t.f32;
    let r = if signed {
        let s = tx.as_i32(v);
        tx.op1(op::CONVERT_S_TO_F, f, s)
    } else {
        tx.op1(op::CONVERT_U_TO_F, f, v)
    };
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// `F2F.dst.src Rd, ±|B|`: between 16- and 32-bit floats (a half is the low half of the register,
/// or the high one when bit 60 is set).
pub(super) fn f2f(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let bs = b(tx, i)?;
    let f = tx.t.f32;
    let r = match sizes(i) {
        (2, 2) => float(tx, bs)?,
        (1, 2) => {
            let x = float(tx, bs)?;
            let zero = tx.cf32(0.0);
            let v2 = tx.t.v2f32;
            let both =
                tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v2, &[x, zero]);
            let u = tx.t.u32;
            let packed = tx.ext(u, glsl::PACK_HALF_2X16, &[both]);
            write(tx, i, dst(i), packed);
            return Ok(());
        }
        (2, 1) => {
            let raw = operand(tx, bs.operand)?;
            let v2 = tx.t.v2f32;
            let both = tx.ext(v2, glsl::UNPACK_HALF_2X16, &[raw]);
            let half = u32::from(i.inst.bit(60));
            let x =
                tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[both, half]);
            super::super::alu::fmods(tx, x, bs.neg, bs.abs)
        }
        (d, s) => return tx.fail(format!("F2F from {} to {} bits", 8u32 << s, 8u32 << d)),
    };
    let r = saturate(tx, i, r);
    write_f(tx, i, dst(i), r);
    Ok(())
}

/// `F2FP.F16.F32.PACK_AB Rd, Ra, B`: two 32-bit floats as the halves of a register, `B` in the low
/// half; with `.MERGE_C` (bit 78) the high half is the low half of the register at 64.
pub(super) fn f2fp(tx: &mut Tx<'_>, i: &I) -> R<()> {
    if i.inst.field(79, 2) != 0 {
        return tx.fail("F2FP with a rounding mode");
    }
    let bs = b(tx, i)?;
    let lo = float(tx, bs)?;
    let v2 = tx.t.v2f32;
    let u = tx.t.u32;
    let r = if i.inst.bit(78) {
        let zero = tx.cf32(0.0);
        let both =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v2, &[lo, zero]);
        let low = tx.ext(u, glsl::PACK_HALF_2X16, &[both]);
        let c = super::read(tx, i, i.inst.reg(64));
        let sixteen = tx.c32(16);
        let high = tx.shl(c, sixteen);
        tx.or(high, low)
    } else {
        let hi = float(tx, a(i))?;
        let both =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v2, &[lo, hi]);
        tx.ext(u, glsl::PACK_HALF_2X16, &[both])
    };
    write(tx, i, dst(i), r);
    Ok(())
}

/// `HADD2`, `HMUL2`, `HFMA2`: not yet.
pub(super) fn half_arith(tx: &mut Tx<'_>, i: &I) -> R<()> {
    // `HADD2`, `HMUL2`, `HFMA2 Rd, Ra, B[, C]` on pairs of halves, as floats (NAK's fields): `Ra`
    // with its halves in bits 74..75, negated with 72 and absolute with 73; the operand of bits
    // 32..63 with its halves in 60..61 (an immediate is a pair as it is), negated with 63 and
    // absolute with 62; `HFMA2`'s register of bits 64..71 with its halves in 81..82, absolute with
    // 83 and negated with 84, the two exchanged in forms 2, 3 and 7. Saturation in bit 77; with
    // `.F32` (bit 78, `HADD2` and `HFMA2`) the result is the low lane as a 32-bit float, else the
    // pair packed.
    use crate::sm86::{op::alu, Operand};
    let inst = i.inst;
    let ra = tx.read(inst.reg(24));
    let x = halves(tx, ra, inst.field(74, 2), inst.bit(72), inst.bit(73));
    let Some(slot) = inst.slot32() else {
        return tx.fail(format!(
            "{} in operand form {}",
            crate::sm86::name(inst),
            inst.form()
        ));
    };
    let near = {
        let raw = operand(tx, slot.operand)?;
        let select = if matches!(slot.operand, Operand::Imm(_)) {
            0
        } else {
            inst.field(60, 2)
        };
        halves(tx, raw, select, slot.neg, slot.abs)
    };
    let base = inst.base() & 0x7F;
    let lanes: Vec<crate::spirv::Id> = if base == alu::HFMA2 & 0x7F {
        let rc = tx.read(inst.reg(64));
        let far = halves(tx, rc, inst.field(81, 2), inst.bit(84), inst.bit(83));
        let (y, z) = if matches!(inst.form(), 2 | 3 | 7) {
            (far, near)
        } else {
            (near, far)
        };
        let f = tx.t.f32;
        (0..2)
            .map(|k| tx.ext(f, glsl::FMA, &[x[k], y[k], z[k]]))
            .collect()
    } else if base == alu::HMUL2 & 0x7F {
        (0..2).map(|k| tx.fmul(x[k], near[k])).collect()
    } else {
        (0..2).map(|k| tx.fadd(x[k], near[k])).collect()
    };
    let lanes: Vec<_> = lanes.into_iter().map(|v| saturate(tx, i, v)).collect();
    if inst.bit(78) && base != alu::HMUL2 & 0x7F {
        write_f(tx, i, dst(i), lanes[0]);
        return Ok(());
    }
    let v2 = tx.t.v2f32;
    let pair = tx.f.emit_result(
        &mut tx.m,
        op::COMPOSITE_CONSTRUCT,
        v2,
        &[lanes[0], lanes[1]],
    );
    let u = tx.t.u32;
    let packed = tx.ext(u, glsl::PACK_HALF_2X16, &[pair]);
    write(tx, i, dst(i), packed);
    Ok(())
}

/// The two lanes of a pair of halves as floats: `select` 0 both as they are, 1 the register as a
/// 32-bit float in both, 2 the low half in both, 3 the high half in both; then the absolute value
/// and the negation.
fn halves(
    tx: &mut Tx<'_>,
    raw: crate::spirv::Id,
    select: u64,
    neg: bool,
    abs: bool,
) -> [crate::spirv::Id; 2] {
    let (lo, hi) = if select == 1 {
        let f = tx.as_f32(raw);
        (f, f)
    } else {
        let v2 = tx.t.v2f32;
        let pair = tx.ext(v2, glsl::UNPACK_HALF_2X16, &[raw]);
        let f = tx.t.f32;
        let lo =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[pair, 0]);
        let hi =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[pair, 1]);
        match select {
            2 => (lo, lo),
            3 => (hi, hi),
            _ => (lo, hi),
        }
    };
    [
        super::super::alu::fmods(tx, lo, neg, abs),
        super::super::alu::fmods(tx, hi, neg, abs),
    ]
}

/// `HSETP2`: not yet.
pub(super) fn hsetp2(tx: &mut Tx<'_>, i: &I) -> R<()> {
    tx.fail(format!("{} is not translated", crate::sm86::name(i.inst)))
}
