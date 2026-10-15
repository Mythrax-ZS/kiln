//! Half-precision arithmetic: `HADD2`, `HMUL2`, `HFMA2`, `HSET2` and `HSETP2` work on two 16-bit
//! floats packed in a register (the first in the low half). The arithmetic is done in 32-bit
//! floats and the results are rounded to halves where they are stored; the result of one add,
//! multiply or fused multiply-add of halves is the same that way, up to the rare case of a double
//! rounding (docs/ASSUMPTIONS.md A-161).

use super::alu::{bool_op, fcompare, fmods, fsat, pred_at, src_b};
use super::emit::{Tx, R};
use crate::inst::Inst;
use crate::ops::Op;
use crate::spirv::{glsl, op, Id};

/// Which values a source operand gives: the two halves of a register, one of them twice, or one
/// 32-bit float used for both lanes (the `F32` swizzle: the other operand is promoted to match).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Swizzle {
    F32,
    H1H0,
    H0H0,
    H1H1,
}

fn swizzle(code: u64) -> Swizzle {
    match code {
        0 => Swizzle::F32,
        1 => Swizzle::H1H0,
        2 => Swizzle::H0H0,
        _ => Swizzle::H1H1,
    }
}

/// How a result goes into the register: both halves, one 32-bit float, or one half with the other
/// half of the register kept.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Merge {
    H1H0,
    F32,
    MrgH0,
    MrgH1,
}

fn merge(code: u64) -> Merge {
    match code {
        0 => Merge::H1H0,
        1 => Merge::F32,
        2 => Merge::MrgH0,
        _ => Merge::MrgH1,
    }
}

/// The two halves of a 32-bit word, as floats.
fn halves(tx: &mut Tx<'_>, word: Id) -> (Id, Id) {
    let v2 = tx.t.v2f32;
    let both = tx.ext(v2, glsl::UNPACK_HALF_2X16, &[word]);
    let f = tx.t.f32;
    let lo =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[both, 0]);
    let hi =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[both, 1]);
    (lo, hi)
}

/// The two values a source operand gives for the lanes.
fn lanes(tx: &mut Tx<'_>, word: Id, s: Swizzle) -> (Id, Id) {
    match s {
        Swizzle::F32 => {
            let f = tx.as_f32(word);
            (f, f)
        }
        Swizzle::H1H0 => halves(tx, word),
        Swizzle::H0H0 => {
            let (lo, _) = halves(tx, word);
            (lo, lo)
        }
        Swizzle::H1H1 => {
            let (_, hi) = halves(tx, word);
            (hi, hi)
        }
    }
}

/// Two floats rounded to halves and packed, the first in the low half.
fn pack(tx: &mut Tx<'_>, lo: Id, hi: Id) -> Id {
    let v2 = tx.t.v2f32;
    let both =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v2, &[lo, hi]);
    tx.ext(tx.t.u32, glsl::PACK_HALF_2X16, &[both])
}

/// A float rounded to the nearest half and back.
fn round_half(tx: &mut Tx<'_>, x: Id) -> Id {
    let packed = pack(tx, x, x);
    let (lo, _) = halves(tx, packed);
    lo
}

/// A result without the values a half cannot hold below its smallest normal number (`FTZ`).
fn flush(tx: &mut Tx<'_>, x: Id) -> Id {
    let magnitude = tx.fabs(x);
    let smallest = tx.cf32(6.103_515_6e-5);
    let tiny = tx.cmp(op::F_ORD_LESS_THAN, magnitude, smallest);
    let zero = tx.cf32(0.0);
    let f = tx.t.f32;
    tx.select(f, tiny, zero, x)
}

/// The register's new value from the two lane results.
fn merged(tx: &mut Tx<'_>, dest: u8, lo: Id, hi: Id, how: Merge) -> Id {
    match how {
        Merge::H1H0 => pack(tx, lo, hi),
        Merge::F32 => {
            let r = round_half(tx, lo);
            tx.as_u32(r)
        }
        Merge::MrgH0 | Merge::MrgH1 => {
            let old = tx.read(dest);
            let (value, window, shift) = if how == Merge::MrgH0 {
                (lo, 0xFFFF_u32, 0)
            } else {
                (hi, 0xFFFF_0000, 16)
            };
            let packed = pack(tx, value, value);
            let low_half = tx.c32(0xFFFF);
            let half = tx.and(packed, low_half);
            let moved = if shift == 0 {
                half
            } else {
                let by = tx.c32(shift);
                tx.shl(half, by)
            };
            let keep = tx.c32(!window);
            let kept = tx.and(old, keep);
            tx.or(kept, moved)
        }
    }
}

/// The two halves an immediate packs: nine bits of a half (its sign, exponent and the top of its
/// fraction) at bits 20 and 30, and their signs at bits 29 and 56.
fn immediate(inst: Inst) -> u32 {
    let low = inst.bits(20, 9) as u32;
    let high = inst.bits(30, 9) as u32;
    (low << 6) | (u32::from(inst.bit(29)) << 15) | (high << 22) | (u32::from(inst.bit(56)) << 31)
}

/// What an instruction form puts in the second operand: its lanes and the bits that modify them.
struct Operand {
    lanes: (Id, Id),
    neg: bool,
    abs: bool,
}

/// The first operand (`Ra`): its swizzle at bits 47..48, its modifiers at bits 43 and 44.
fn operand_a(tx: &mut Tx<'_>, inst: Inst) -> Operand {
    let word = tx.read(inst.ra());
    Operand {
        lanes: lanes(tx, word, swizzle(inst.bits(47, 2))),
        neg: inst.bit(43),
        abs: inst.bit(44),
    }
}

/// The second operand of the forms with a register, a constant buffer word (one float for both
/// lanes) and an immediate.
fn operand_b(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<Operand> {
    Ok(match opc {
        Op::Hadd2Reg | Op::Hmul2Reg | Op::Hset2Reg | Op::Hsetp2Reg => {
            let word = tx.read(inst.rb());
            Operand {
                lanes: lanes(tx, word, swizzle(inst.bits(28, 2))),
                neg: inst.bit(31),
                abs: inst.bit(30),
            }
        }
        Op::Hadd2Cbuf | Op::Hmul2Cbuf => {
            let word = src_b(tx, inst)?;
            Operand {
                lanes: lanes(tx, word, Swizzle::F32),
                neg: inst.bit(56),
                abs: inst.bit(54),
            }
        }
        Op::Hset2Cbuf | Op::Hsetp2Cbuf => {
            let word = src_b(tx, inst)?;
            Operand {
                lanes: lanes(tx, word, Swizzle::F32),
                neg: inst.bit(56),
                abs: inst.bit(54),
            }
        }
        Op::Hadd2 | Op::Hmul2 | Op::Hset2 | Op::Hsetp2 => {
            let word = tx.c32(immediate(inst));
            Operand {
                lanes: lanes(tx, word, Swizzle::H1H0),
                neg: false,
                abs: false,
            }
        }
        _ => return tx.fail("a half-precision operand of a form that is not known"),
    })
}

/// Apply a source's modifiers to both lanes.
fn modified(tx: &mut Tx<'_>, o: &Operand) -> (Id, Id) {
    (
        fmods(tx, o.lanes.0, o.neg, o.abs),
        fmods(tx, o.lanes.1, o.neg, o.abs),
    )
}

/// The `ftz` and `sat` bits of the arithmetic forms, and where the second operand's forms keep
/// them: `HADD2` and `HMUL2` have the flush at bit 39 and the clamp at bit 32 (register form) or
/// 52 (the others).
fn flags(inst: Inst, opc: Op) -> (bool, bool) {
    let sat_bit = match opc {
        Op::Hadd2Reg | Op::Hmul2Reg => 32,
        _ => 52,
    };
    (inst.bit(39), inst.bit(sat_bit))
}

pub(super) fn translate(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<bool> {
    match opc {
        Op::Hadd2Reg | Op::Hadd2Cbuf | Op::Hadd2 => hadd2(tx, inst, opc)?,
        Op::Hadd232i => hadd2_32i(tx, inst)?,
        Op::Hmul2Reg | Op::Hmul2Cbuf | Op::Hmul2 => hmul2(tx, inst, opc)?,
        Op::Hmul232i => hmul2_32i(tx, inst)?,
        Op::Hfma2Reg | Op::Hfma2Cbuf_6080 | Op::Hfma2Cbuf_7080 | Op::Hfma2 | Op::Hfma232i => {
            hfma2(tx, inst, opc)?
        }
        Op::Hset2Reg | Op::Hset2Cbuf | Op::Hset2 => hset2(tx, inst, opc)?,
        Op::Hsetp2Reg | Op::Hsetp2Cbuf | Op::Hsetp2 => hsetp2(tx, inst, opc)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Finish an arithmetic instruction: flush, clamp, and put the lanes in `Rd`.
fn finish(tx: &mut Tx<'_>, inst: Inst, how: Merge, ftz: bool, sat: bool, lo: Id, hi: Id) {
    let (mut lo, mut hi) = (lo, hi);
    if ftz {
        lo = flush(tx, lo);
        hi = flush(tx, hi);
    }
    if sat {
        lo = fsat(tx, lo);
        hi = fsat(tx, hi);
    }
    let value = merged(tx, inst.rd(), lo, hi, how);
    tx.write(inst, inst.rd(), value);
}

fn hadd2(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let a = operand_a(tx, inst);
    let b = operand_b(tx, inst, opc)?;
    let (a0, a1) = modified(tx, &a);
    let (b0, b1) = modified(tx, &b);
    let lo = tx.fadd(a0, b0);
    let hi = tx.fadd(a1, b1);
    let (ftz, sat) = flags(inst, opc);
    finish(tx, inst, merge(inst.bits(49, 2)), ftz, sat, lo, hi);
    Ok(())
}

/// `HADD2.32I`: the second operand is a 32-bit immediate of two halves.
fn hadd2_32i(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let word = tx.read(inst.ra());
    let (a0, a1) = lanes(tx, word, swizzle(inst.bits(53, 2)));
    let (a0, a1) = (
        fmods(tx, a0, inst.bit(56), false),
        fmods(tx, a1, inst.bit(56), false),
    );
    let imm = tx.c32(inst.imm32());
    let (b0, b1) = halves(tx, imm);
    let lo = tx.fadd(a0, b0);
    let hi = tx.fadd(a1, b1);
    finish(tx, inst, Merge::H1H0, inst.bit(55), inst.bit(52), lo, hi);
    Ok(())
}

fn hmul2(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let a = operand_a(tx, inst);
    let b = operand_b(tx, inst, opc)?;
    let (a0, a1) = modified(tx, &a);
    let (b0, b1) = modified(tx, &b);
    let lo = tx.fmul(a0, b0);
    let hi = tx.fmul(a1, b1);
    let (ftz, sat) = flags(inst, opc);
    finish(tx, inst, merge(inst.bits(49, 2)), ftz, sat, lo, hi);
    Ok(())
}

/// `HMUL2.32I`: the second operand is a 32-bit immediate of two halves.
fn hmul2_32i(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let word = tx.read(inst.ra());
    let (a0, a1) = lanes(tx, word, swizzle(inst.bits(53, 2)));
    let imm = tx.c32(inst.imm32());
    let (b0, b1) = halves(tx, imm);
    let lo = tx.fmul(a0, b0);
    let hi = tx.fmul(a1, b1);
    finish(tx, inst, Merge::H1H0, false, inst.bit(52), lo, hi);
    Ok(())
}

/// `HFMA2`: `Ra * B + C` in both lanes. The forms differ in where `B` and `C` come from: both
/// registers (`Rb` and the register at bit 39), a constant buffer word for `B` and the register for
/// `C`, the other way round, or an immediate for `B`.
fn hfma2(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let a = tx.read(inst.ra());
    let (swz_a, how) = (swizzle(inst.bits(47, 2)), merge(inst.bits(49, 2)));
    let (neg_b, neg_c, sat, swz_b, swz_c, ftz);
    let (b_word, c_word, b_f32, c_f32);
    match opc {
        Op::Hfma2Reg => {
            neg_b = inst.bit(31);
            neg_c = inst.bit(30);
            sat = inst.bit(32);
            swz_b = swizzle(inst.bits(28, 2));
            swz_c = swizzle(inst.bits(35, 2));
            ftz = inst.bits(37, 2) == 1;
            b_word = tx.read(inst.rb());
            c_word = tx.read(inst.rc());
            (b_f32, c_f32) = (false, false);
        }
        Op::Hfma2Cbuf_7080 | Op::Hfma2Cbuf_6080 | Op::Hfma2 => {
            neg_b = inst.bit(56);
            neg_c = inst.bit(51);
            sat = inst.bit(52);
            swz_c = swizzle(inst.bits(53, 2));
            swz_b = Swizzle::H1H0;
            ftz = inst.bits(57, 2) == 1;
            match opc {
                // The constant is `B`, as one float; `C` is the register at bit 39.
                Op::Hfma2Cbuf_7080 => {
                    b_word = src_b(tx, inst)?;
                    c_word = tx.read(inst.rc());
                    (b_f32, c_f32) = (true, false);
                }
                // `B` is the register at bit 39; the constant is `C`, as one float.
                Op::Hfma2Cbuf_6080 => {
                    b_word = tx.read(inst.rc());
                    c_word = src_b(tx, inst)?;
                    (b_f32, c_f32) = (false, true);
                }
                _ => {
                    b_word = tx.c32(immediate(inst));
                    c_word = tx.read(inst.rc());
                    (b_f32, c_f32) = (false, false);
                }
            }
        }
        _ => {
            // HFMA2.32I: `Ra * imm + Rd`, the third operand being the destination register.
            neg_b = false;
            neg_c = inst.bit(52);
            sat = inst.bit(55);
            swz_b = Swizzle::H1H0;
            swz_c = Swizzle::H1H0;
            ftz = false;
            b_word = tx.c32(inst.imm32());
            c_word = tx.read(inst.rd());
            (b_f32, c_f32) = (false, false);
        }
    }
    let (a0, a1) = lanes(
        tx,
        a,
        if opc == Op::Hfma232i {
            Swizzle::H1H0
        } else {
            swz_a
        },
    );
    let (b0, b1) = lanes(tx, b_word, if b_f32 { Swizzle::F32 } else { swz_b });
    let (c0, c1) = lanes(tx, c_word, if c_f32 { Swizzle::F32 } else { swz_c });
    let (b0, b1) = (fmods(tx, b0, neg_b, false), fmods(tx, b1, neg_b, false));
    let (c0, c1) = (fmods(tx, c0, neg_c, false), fmods(tx, c1, neg_c, false));
    let lo = tx.fma(a0, b0, c0);
    let hi = tx.fma(a1, b1, c1);
    finish(
        tx,
        inst,
        if opc == Op::Hfma232i {
            Merge::H1H0
        } else {
            how
        },
        ftz,
        sat,
        lo,
        hi,
    );
    Ok(())
}

/// The comparison of the lanes of `HSET2` and `HSETP2`: the operation's code, the way the forms
/// keep it, and whether one flag says both lanes.
struct Compare {
    code: u64,
    ftz: bool,
}

fn compare_fields(inst: Inst, opc: Op) -> Compare {
    match opc {
        Op::Hsetp2Reg => Compare {
            code: inst.bits(35, 4),
            ftz: inst.bit(6),
        },
        Op::Hset2Reg => Compare {
            code: inst.bits(35, 4),
            ftz: inst.bit(50),
        },
        Op::Hsetp2Cbuf | Op::Hsetp2 => Compare {
            code: inst.bits(49, 4),
            ftz: inst.bit(6),
        },
        _ => Compare {
            code: inst.bits(49, 4),
            ftz: inst.bit(54),
        },
    }
}

/// Both lanes of a comparison of `Ra` with the second operand.
fn lane_compare(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<(Id, Id)> {
    let a = operand_a(tx, inst);
    let b = operand_b(tx, inst, opc)?;
    let (mut a0, mut a1) = modified(tx, &a);
    let (mut b0, mut b1) = modified(tx, &b);
    if compare_fields(inst, opc).ftz {
        for v in [&mut a0, &mut a1, &mut b0, &mut b1] {
            *v = flush(tx, *v);
        }
    }
    let code = compare_fields(inst, opc).code;
    let lo = fcompare(tx, code, a0, b0)?;
    let hi = fcompare(tx, code, a1, b1)?;
    Ok((lo, hi))
}

fn hsetp2(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let (lo, hi) = lane_compare(tx, inst, opc)?;
    let pred = pred_at(tx, inst, 39);
    let bop = inst.bits(45, 2);
    let lo = bool_op(tx, bop, lo, pred)?;
    let hi = bool_op(tx, bop, hi, pred)?;
    let h_and = match opc {
        Op::Hsetp2Reg => inst.bit(49),
        _ => inst.bit(53),
    };
    let (first, second) = if h_and {
        let both = tx.land(lo, hi);
        let none = tx.lnot(both);
        (both, none)
    } else {
        (lo, hi)
    };
    tx.write_pred(inst, inst.bits(3, 3) as u8, first);
    tx.write_pred(inst, inst.bits(0, 3) as u8, second);
    Ok(())
}

fn hset2(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let (lo, hi) = lane_compare(tx, inst, opc)?;
    let pred = pred_at(tx, inst, 39);
    let bop = inst.bits(45, 2);
    let lo = bool_op(tx, bop, lo, pred)?;
    let hi = bool_op(tx, bop, hi, pred)?;
    // A true lane is 1.0 as a half with the `bf` flag, else all ones.
    let bf = match opc {
        Op::Hset2Reg => inst.bit(49),
        _ => inst.bit(53),
    };
    let one = if bf { 0x3C00 } else { 0xFFFF };
    let (low_true, high_true) = (tx.c32(one), tx.c32(one << 16));
    let zero = tx.c32(0);
    let low = tx.select_u32(lo, low_true, zero);
    let high = tx.select_u32(hi, high_true, zero);
    let value = tx.or(low, high);
    tx.write(inst, inst.rd(), value);
    Ok(())
}
