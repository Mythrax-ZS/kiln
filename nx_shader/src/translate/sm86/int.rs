//! Integers: `ISETP`, `IADD3`, `LEA`, `LOP3`, `IABS`, `PRMT`, `IMNMX`, `SHF`, `SGXT`, `BMSK`, `IMAD`
//! (and `.WIDE`), `FLO`, `BREV` and `POPC`, vector and uniform alike.

use super::super::alu::{bool_op, icompare};
use super::super::emit::{Tx, R};
use super::super::int::{lop3_value, prmt_value};
use super::{a, b, bc, dst, operand, pred, reg_at, write, write_pred, I};
use crate::sm86::{Operand, Source, PT};
use crate::spirv::{glsl, op, Id};

/// An integer operand's raw value; its negation flag is the caller's to apply.
fn raw(tx: &mut Tx<'_>, s: Source) -> R<Id> {
    operand(tx, s.operand)
}

fn u64_of(tx: &mut Tx<'_>, v: Id) -> Id {
    let u64t = tx.m.ty_int(64, false);
    tx.op1(op::U_CONVERT, u64t, v)
}

/// The 64-bit value `hi:lo`.
fn join(tx: &mut Tx<'_>, lo: Id, hi: Id) -> Id {
    tx.m.capability(crate::spirv::cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let lo64 = u64_of(tx, lo);
    let hi64 = u64_of(tx, hi);
    let c32 = tx.c32(32);
    let shifted = tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, hi64, c32);
    tx.op2(op::BITWISE_OR, u64t, shifted, lo64)
}

fn low(tx: &mut Tx<'_>, v: Id) -> Id {
    let u = tx.t.u32;
    tx.op1(op::U_CONVERT, u, v)
}

fn high(tx: &mut Tx<'_>, v: Id) -> Id {
    let u64t = tx.m.ty_int(64, false);
    let c32 = tx.c32(32);
    let shifted = tx.op2(op::SHIFT_RIGHT_LOGICAL, u64t, v, c32);
    low(tx, shifted)
}

/// `ISETP.cmp.type.bop Pu, Pv, Ra, B, Pc`; with `.EX` (bit 72) the comparison is the high words'
/// part of a 64-bit one, the low words' result coming in as the predicate in bits 68..71.
pub(super) fn isetp(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let x = raw(tx, a(i))?;
    let bs = b(tx, i)?;
    let y = raw(tx, bs)?;
    let signed = inst.bit(73);
    let code = inst.field(76, 3);
    let c = if inst.bit(72) {
        let lower = pred(tx, inst.pred(68, 71, i.uniform));
        let eq = tx.cmp(op::I_EQUAL, x, y);
        match code {
            0 => tx.cfalse(),
            7 => tx.ctrue(),
            2 => tx.land(eq, lower),
            5 => {
                let ne = tx.lnot(eq);
                tx.lor(ne, lower)
            }
            _ => {
                // LT and LE take the high words' strict less-than, GT and GE the greater-than; equal
                // high words leave it to the low words.
                let strict = if code <= 3 { 1 } else { 4 };
                let s = icompare(tx, strict, x, y, signed);
                let rest = tx.land(eq, lower);
                tx.lor(s, rest)
            }
        }
    } else {
        icompare(tx, code, x, y, signed)
    };
    let p = pred(tx, inst.pred(87, 90, i.uniform));
    let bop = inst.field(74, 2);
    let first = bool_op(tx, bop, c, p)?;
    let nc = tx.lnot(c);
    let second = bool_op(tx, bop, nc, p)?;
    write_pred(tx, i, inst.pred_dst(81, i.uniform), first);
    write_pred(tx, i, inst.pred_dst(84, i.uniform), second);
    Ok(())
}

/// `IADD3 Rd, Pu, Pv, ±Ra, ±B, ±C`: the sum of three words; `Pu` and `Pv` the two carries out
/// (the carry of the 34-bit sum is at least one and at least two). With `.X` (bit 74) the two
/// predicates in bits 87..90 and 77..80 add one each, and the negation flags are bitwise nots (the
/// high words of a multi-word sum); without it a negation is the two's complement.
pub(super) fn iadd3(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let sa = a(i);
    let (sb, sc) = bc(tx, i)?;
    if sa.abs || sb.abs {
        return tx.fail("IADD3 with an absolute value");
    }
    let extended = inst.bit(74);
    let terms = [sa, sb, sc];
    let pointer = super::pointer_of(tx, &terms);
    let carry_out = [inst.pred_dst(81, i.uniform), inst.pred_dst(84, i.uniform)];
    let wants_carry = carry_out.iter().any(|p| *p != PT);
    let mut ones = 0u32;
    let mut values = Vec::new();
    for s in terms {
        let v = raw(tx, s)?;
        if s.neg {
            values.push(tx.not(v));
            if !extended {
                ones += 1;
            }
        } else {
            values.push(v);
        }
    }
    let carries_in: Vec<Id> = if extended {
        [inst.pred(87, 90, i.uniform), inst.pred(77, 80, i.uniform)]
            .into_iter()
            .filter(|g| !(g.index == PT && g.negated))
            .map(|g| {
                let p = pred(tx, g);
                let one = tx.c32(1);
                let zero = tx.c32(0);
                tx.select_u32(p, one, zero)
            })
            .collect()
    } else {
        Vec::new()
    };
    if !wants_carry {
        let mut sum = tx.iadd(values[0], values[1]);
        sum = tx.iadd(sum, values[2]);
        if ones > 0 {
            let k = tx.c32(ones);
            sum = tx.iadd(sum, k);
        }
        for c in carries_in {
            sum = tx.iadd(sum, c);
        }
        write(tx, i, dst(i), sum);
        super::set_pointer(tx, i, dst(i), pointer);
        return Ok(());
    }
    // The carries need the sum's bits above 32: add as 64-bit numbers.
    tx.m.capability(crate::spirv::cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let mut total = tx.m.const_u64(u64::from(ones));
    for v in values.into_iter().chain(carries_in) {
        let w = u64_of(tx, v);
        total = tx.op2(op::I_ADD, u64t, total, w);
    }
    let sum = low(tx, total);
    let carry = high(tx, total);
    write(tx, i, dst(i), sum);
    super::set_pointer(tx, i, dst(i), pointer);
    for (k, p) in carry_out.into_iter().enumerate() {
        if p == PT {
            continue;
        }
        let at_least = tx.c32(k as u32 + 1);
        let set = tx.cmp(op::U_GREATER_THAN_EQUAL, carry, at_least);
        write_pred(tx, i, p, set);
    }
    Ok(())
}

/// `LEA Rd, Pu, ±Ra, ±B, shift`: `(Ra << shift) + B`, `Pu` its carry. `LEA.HI Rd, Ra, B, C,
/// shift`: the high word of `C:Ra` shifted left (with `.SX32`, bit 73, `C` is the sign of `Ra`)
/// plus `B`. `.X` (bit 74) adds the predicate in bits 87..90.
pub(super) fn lea(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let (sb, sc) = bc(tx, i)?;
    let pointer = super::pointer_of(tx, &[a(i), sb, sc]);
    let shift = inst.field(75, 5) as u32;
    let x = raw(tx, a(i))?;
    let y = raw(tx, sb)?;
    let carry_in = if inst.bit(74) {
        let p = pred(tx, inst.pred(87, 90, i.uniform));
        let one = tx.c32(1);
        let zero = tx.c32(0);
        Some(tx.select_u32(p, one, zero))
    } else {
        None
    };
    let (shifted, addend) = if inst.bit(80) {
        if inst.bit(72) || sb.neg {
            return tx.fail("LEA.HI with a negated operand");
        }
        let hi = if inst.bit(73) {
            let thirty_one = tx.c32(31);
            tx.sar(x, thirty_one)
        } else {
            raw(tx, sc)?
        };
        let shifted = if shift == 0 {
            hi
        } else {
            let s = tx.c32(shift);
            let back = tx.c32(32 - shift);
            let up = tx.shl(hi, s);
            let down = tx.shr(x, back);
            tx.or(up, down)
        };
        (shifted, y)
    } else {
        let s = tx.c32(shift);
        let mut shifted = tx.shl(x, s);
        if inst.bit(72) {
            shifted = if inst.bit(74) {
                tx.not(shifted)
            } else {
                tx.ineg(shifted)
            };
        }
        let addend = if sb.neg && matches!(inst.form(), 1 | 5 | 6) {
            if inst.bit(74) {
                tx.not(y)
            } else {
                tx.ineg(y)
            }
        } else {
            y
        };
        (shifted, addend)
    };
    let carry_out = inst.pred_dst(81, i.uniform);
    if carry_out == PT {
        let mut sum = tx.iadd(shifted, addend);
        if let Some(c) = carry_in {
            sum = tx.iadd(sum, c);
        }
        write(tx, i, dst(i), sum);
        super::set_pointer(tx, i, dst(i), pointer);
        return Ok(());
    }
    tx.m.capability(crate::spirv::cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let s64 = u64_of(tx, shifted);
    let a64 = u64_of(tx, addend);
    let mut total = tx.op2(op::I_ADD, u64t, s64, a64);
    if let Some(c) = carry_in {
        let c64 = u64_of(tx, c);
        total = tx.op2(op::I_ADD, u64t, total, c64);
    }
    let sum = low(tx, total);
    let carry = high(tx, total);
    let zero = tx.c32(0);
    let set = tx.cmp(op::I_NOT_EQUAL, carry, zero);
    write(tx, i, dst(i), sum);
    super::set_pointer(tx, i, dst(i), pointer);
    write_pred(tx, i, carry_out, set);
    Ok(())
}

/// `LOP3.LUT Pu, Rd, Ra, B, C, lut, Pc`: every result bit is bit `4a + 2b + c` of the table; `Pu`
/// whether the result is not zero, or-ed with `Pc` (and-ed with `.PAND`, bit 80).
pub(super) fn lop3(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let x = raw(tx, a(i))?;
    let (sb, sc) = bc(tx, i)?;
    let y = raw(tx, sb)?;
    let z = raw(tx, sc)?;
    let r = lop3_value(tx, inst.field(72, 8), x, y, z);
    write(tx, i, dst(i), r);
    let p = inst.pred_dst(81, i.uniform);
    if p != PT {
        let zero = tx.c32(0);
        let nonzero = tx.cmp(op::I_NOT_EQUAL, r, zero);
        let pin = pred(tx, inst.pred(87, 90, i.uniform));
        let v = if inst.bit(80) {
            tx.land(nonzero, pin)
        } else {
            tx.lor(nonzero, pin)
        };
        write_pred(tx, i, p, v);
    }
    Ok(())
}

/// `IABS Rd, B`.
pub(super) fn iabs(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let bs = b(tx, i)?;
    let v = raw(tx, bs)?;
    let s = tx.as_i32(v);
    let it = tx.t.i32;
    let r = tx.ext(it, glsl::S_ABS, &[s]);
    let r = tx.as_u32(r);
    write(tx, i, dst(i), r);
    Ok(())
}

/// `PRMT.mode Rd, Ra, SEL, C` (the mode in bits 72..74).
pub(super) fn prmt(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let x = raw(tx, a(i))?;
    let (ssel, sc) = bc(tx, i)?;
    let sel = raw(tx, ssel)?;
    let z = raw(tx, sc)?;
    let r = prmt_value(tx, i.inst.field(72, 3), x, sel, z)?;
    write(tx, i, dst(i), r);
    Ok(())
}

/// `IMNMX.type Rd, Ra, B, Pc`: the minimum where the predicate holds, else the maximum.
pub(super) fn imnmx(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let x = raw(tx, a(i))?;
    let bs = b(tx, i)?;
    let y = raw(tx, bs)?;
    let want_min = pred(tx, i.inst.pred(87, 90, i.uniform));
    let r = if i.inst.bit(73) {
        let it = tx.t.i32;
        let sx = tx.as_i32(x);
        let sy = tx.as_i32(y);
        let lo = tx.ext(it, glsl::S_MIN, &[sx, sy]);
        let hi = tx.ext(it, glsl::S_MAX, &[sx, sy]);
        let lo = tx.as_u32(lo);
        let hi = tx.as_u32(hi);
        tx.select_u32(want_min, lo, hi)
    } else {
        let u = tx.t.u32;
        let lo = tx.ext(u, glsl::U_MIN, &[x, y]);
        let hi = tx.ext(u, glsl::U_MAX, &[x, y]);
        tx.select_u32(want_min, lo, hi)
    };
    write(tx, i, dst(i), r);
    Ok(())
}

/// `SHF.{L|R}[.W].type[.HI] Rd, Ra, shift, C`: a funnel shift of the 64-bit value `C:Ra`. The
/// type (bits 73..74: 0 S64, 1 U64, 2 S32, 3 U32) gives the range the amount is clamped to (or
/// wrapped in, `.W`, bit 75: 64 or 32) and whether a right shift brings in the sign. The result is
/// the low word of the shifted value, or the high one with `.HI` (bit 80).
pub(super) fn shf(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let lo = raw(tx, a(i))?;
    let (sshift, shigh) = bc(tx, i)?;
    let amount = raw(tx, sshift)?;
    let hi = raw(tx, shigh)?;
    let kind = inst.field(73, 2);
    let width = if kind < 2 { 64 } else { 32 };
    let signed = kind == 0 || kind == 2;
    let right = inst.bit(76);
    let u = tx.t.u32;
    let s = if inst.bit(75) {
        let mask = tx.c32(width - 1);
        tx.and(amount, mask)
    } else {
        let w = tx.c32(width);
        tx.ext(u, glsl::U_MIN, &[amount, w])
    };
    // The 64-bit funnel `hi:lo` shifted by `s` (0..=64), word by word: every shift of SPIR-V here
    // is by less than 32, so each is defined, and no 64-bit integer is needed.
    let c1 = tx.c32(1);
    let c31 = tx.c32(31);
    let c32 = tx.c32(32);
    let c64 = tx.c32(64);
    let zero = tx.c32(0);
    let big = tx.cmp(op::U_GREATER_THAN_EQUAL, s, c32);
    let all = tx.cmp(op::I_EQUAL, s, c64);
    let m = tx.and(s, c31);
    let rest = tx.isub(c31, m);
    let (rlo, rhi) = if right {
        // The bits of `hi` that come into `lo`: `hi << (32 - m)`, none when `m` is 0.
        let hi1 = tx.shl(hi, c1);
        let cross = tx.shl(hi1, rest);
        let lo_part = tx.shr(lo, m);
        let lo_small = tx.or(lo_part, cross);
        let (hi_shifted, fill) = if signed {
            (tx.sar(hi, m), tx.sar(hi, c31))
        } else {
            (tx.shr(hi, m), zero)
        };
        // Below 32: both words move; from 32: `hi` comes into `lo`; 64: only the fill is left.
        let rlo = tx.select_u32(big, hi_shifted, lo_small);
        let rhi = tx.select_u32(big, fill, hi_shifted);
        (tx.select_u32(all, fill, rlo), tx.select_u32(all, fill, rhi))
    } else {
        // The bits of `lo` that come into `hi`: `lo >> (32 - m)`, none when `m` is 0.
        let lo1 = tx.shr(lo, c1);
        let cross = tx.shr(lo1, rest);
        let hi_part = tx.shl(hi, m);
        let hi_small = tx.or(hi_part, cross);
        let lo_shifted = tx.shl(lo, m);
        let rlo = tx.select_u32(big, zero, lo_shifted);
        let rhi = tx.select_u32(big, lo_shifted, hi_small);
        (rlo, tx.select_u32(all, zero, rhi))
    };
    let r = if inst.bit(80) { rhi } else { rlo };
    write(tx, i, dst(i), r);
    Ok(())
}

/// `SGXT[.U32] Rd, Ra, bits`: the low `bits` of `Ra`, sign-extended (zero-extended with `.U32`,
/// bit 73 clear).
pub(super) fn sgxt(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let x = raw(tx, a(i))?;
    let bs = b(tx, i)?;
    let n = raw(tx, bs)?;
    let u = tx.t.u32;
    let w = tx.c32(32);
    let count = tx.ext(u, glsl::U_MIN, &[n, w]);
    let zero = tx.c32(0);
    let r = if i.inst.bit(73) {
        let s = tx.as_i32(x);
        let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, zero, count);
        tx.as_u32(e)
    } else {
        tx.op3(op::BIT_FIELD_U_EXTRACT, u, x, zero, count)
    };
    write(tx, i, dst(i), r);
    Ok(())
}

/// `BMSK Rd, pos, width`: `width` ones from bit `pos` (`.W`, bit 75: both taken modulo 32, else
/// clamped to 32).
pub(super) fn bmsk(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let pos = raw(tx, a(i))?;
    let bs = b(tx, i)?;
    let width = raw(tx, bs)?;
    let u = tx.t.u32;
    let (pos, width) = if i.inst.bit(75) {
        let m = tx.c32(31);
        (tx.and(pos, m), tx.and(width, m))
    } else {
        let w = tx.c32(32);
        (
            tx.ext(u, glsl::U_MIN, &[pos, w]),
            tx.ext(u, glsl::U_MIN, &[width, w]),
        )
    };
    tx.m.capability(crate::spirv::cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let one = tx.m.const_u64(1);
    let w64 = u64_of(tx, width);
    let p64 = u64_of(tx, pos);
    let top = tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, one, w64);
    let ones = tx.op2(op::I_SUB, u64t, top, one);
    let placed = tx.op2(op::SHIFT_LEFT_LOGICAL, u64t, ones, p64);
    let r = low(tx, placed);
    write(tx, i, dst(i), r);
    Ok(())
}

/// `IMAD Rd, ±Ra, ±B, ±C`: the low word of `Ra * B + C`.
pub(super) fn imad(tx: &mut Tx<'_>, i: &I) -> R<()> {
    // `.X` (bit 74): the high word of a 64-bit sum, the carry of the low word's (the predicate in
    // bits 87..90) added in, and a negated addend complemented, as `IADD3.X` has them (A-240).
    let extended = i.inst.bit(74);
    if extended && i.inst.pred_dst(81, false) != PT {
        return tx.fail("IMAD.X with a carry out");
    }
    let sa = a(i);
    let (sb, sc) = bc(tx, i)?;
    let pointer = super::pointer_of(tx, &[sa, sb, sc]);
    let x = raw(tx, sa)?;
    let y = raw(tx, sb)?;
    let z = raw(tx, sc)?;
    let mut product = tx.imul(x, y);
    if sa.neg != sb.neg {
        product = tx.ineg(product);
    }
    let addend = match (sc.neg, extended) {
        (false, _) => z,
        (true, false) => tx.ineg(z),
        (true, true) => tx.not(z),
    };
    let mut r = tx.iadd(product, addend);
    if extended {
        let carry = pred(tx, i.inst.pred(87, 90, false));
        let u = tx.t.u32;
        let (one, zero) = (tx.c32(1), tx.c32(0));
        let c = tx.select(u, carry, one, zero);
        r = tx.iadd(r, c);
    }
    write(tx, i, dst(i), r);
    super::set_pointer(tx, i, dst(i), pointer);
    Ok(())
}

/// `IMAD.WIDE[.U32] Rd+1:Rd, Ra, B, C+1:C`: the 64-bit product (signed with bit 73) plus a 64-bit
/// addend from a register pair.
pub(super) fn imad_wide(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let sa = a(i);
    let (sb, sc) = bc(tx, i)?;
    if sa.neg || sb.neg || sc.neg {
        return tx.fail("IMAD.WIDE with a negated operand");
    }
    let pointer = super::pointer_of(tx, &[sa, sb, sc]);
    let x = raw(tx, sa)?;
    let y = raw(tx, sb)?;
    let (zl, zh) = match sc.operand {
        Operand::Reg(r) => (tx.read(r), tx.read(reg_at(i, r, 1))),
        Operand::UReg(r) => (tx.read_u(r), tx.read_u(reg_at(i, r, 1))),
        _ => return tx.fail("IMAD.WIDE with an addend that is not a register pair"),
    };
    tx.m.capability(crate::spirv::cap::INT64);
    let u64t = tx.m.ty_int(64, false);
    let (x64, y64) = if i.inst.bit(73) {
        let i64t = tx.m.ty_int(64, true);
        let sx = tx.as_i32(x);
        let sy = tx.as_i32(y);
        let ex = tx.op1(op::S_CONVERT, i64t, sx);
        let ey = tx.op1(op::S_CONVERT, i64t, sy);
        (tx.op1(op::BITCAST, u64t, ex), tx.op1(op::BITCAST, u64t, ey))
    } else {
        (u64_of(tx, x), u64_of(tx, y))
    };
    let product = tx.op2(op::I_MUL, u64t, x64, y64);
    let addend = join(tx, zl, zh);
    let r = tx.op2(op::I_ADD, u64t, product, addend);
    let lo = low(tx, r);
    let hi = high(tx, r);
    let d = dst(i);
    write(tx, i, d, lo);
    write(tx, i, reg_at(i, d, 1), hi);
    super::set_pointer(tx, i, d, pointer);
    Ok(())
}

/// `IMAD.HI`: not yet.
pub(super) fn imad_hi(tx: &mut Tx<'_>, _i: &I) -> R<()> {
    tx.fail("IMAD.HI is not translated")
}

/// `FLO[.U32|.S32][.SH] Rd, B`: the place of the highest one bit (for `.S32`, bit 73, the highest
/// bit unlike the sign), -1 for none; `.SH` (bit 74) counts it from the top. Bit 63 inverts `B`.
pub(super) fn flo(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let bs = b(tx, i)?;
    let mut v = raw(tx, bs)?;
    if i.inst.bit(63) {
        v = tx.not(v);
    }
    let u = tx.t.u32;
    let msb = if i.inst.bit(73) {
        let s = tx.as_i32(v);
        let it = tx.t.i32;
        let m = tx.ext(it, glsl::FIND_S_MSB, &[s]);
        tx.as_u32(m)
    } else {
        tx.ext(u, glsl::FIND_U_MSB, &[v])
    };
    let r = if i.inst.bit(74) {
        let none = tx.c32(0xFFFF_FFFF);
        let found = tx.cmp(op::I_NOT_EQUAL, msb, none);
        let thirty_one = tx.c32(31);
        let from_top = tx.isub(thirty_one, msb);
        tx.select_u32(found, from_top, none)
    } else {
        msb
    };
    write(tx, i, dst(i), r);
    Ok(())
}

/// `BREV Rd, B`: the bits in the other order.
pub(super) fn brev(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let bs = b(tx, i)?;
    let v = raw(tx, bs)?;
    let u = tx.t.u32;
    let r = tx.op1(op::BIT_REVERSE, u, v);
    write(tx, i, dst(i), r);
    Ok(())
}

/// `POPC Rd, B`: the number of one bits (of `~B` with bit 63).
pub(super) fn popc(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let bs = b(tx, i)?;
    let mut v = raw(tx, bs)?;
    if i.inst.bit(63) {
        v = tx.not(v);
    }
    let u = tx.t.u32;
    let r = tx.op1(op::BIT_COUNT, u, v);
    write(tx, i, dst(i), r);
    Ok(())
}
