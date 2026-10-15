//! Arithmetic, logic, comparison and conversion instructions.
//!
//! Field positions are from envytools' GM107 tables and Mesa NAK's encoder; the rounding modes and
//! the flush-to-zero flag of floating point instructions are not modelled (the host's defaults
//! apply; docs/ASSUMPTIONS.md), and the condition-code flags other than the carry are not either:
//! an instruction that writes them stops the translation.

use super::emit::{Tx, R};
use crate::inst::{Inst, PT, RZ};
use crate::ops::{Form, Op};
use crate::spirv::{glsl, op, Id};

/// Translate an arithmetic instruction. `false` when the instruction is not one of these.
pub(super) fn translate(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<bool> {
    match opc {
        Op::FaddReg | Op::FaddCbuf | Op::FaddImm20 | Op::Fadd32iImm32 => fadd(tx, inst, opc)?,
        Op::FmulReg | Op::FmulCbuf | Op::FmulImm20 | Op::Fmul32iImm32 => fmul(tx, inst, opc)?,
        Op::FfmaReg | Op::FfmaCbuf_4980 | Op::FfmaCbuf_5180 | Op::FfmaImm20 | Op::Ffma32iImm32 => {
            ffma(tx, inst, opc)?
        }
        Op::FmnmxReg | Op::FmnmxCbuf | Op::FmnmxImm20 => fmnmx(tx, inst)?,
        Op::FsetpReg | Op::FsetpCbuf | Op::FsetpImm20 => fsetp(tx, inst)?,
        Op::FsetReg | Op::FsetCbuf | Op::FsetImm20 => fset(tx, inst)?,
        Op::Mufu => mufu(tx, inst)?,
        Op::RroReg | Op::RroCbuf | Op::RroImm20 => rro(tx, inst)?,
        Op::F2iReg | Op::F2iCbuf | Op::F2iImm20 => f2i(tx, inst)?,
        Op::I2fReg | Op::I2fCbuf | Op::I2fImm20 => i2f(tx, inst)?,
        Op::F2fReg | Op::F2fCbuf | Op::F2fImm20 => f2f(tx, inst)?,
        Op::I2iReg | Op::I2iCbuf | Op::I2iImm20 => i2i(tx, inst)?,
        Op::IaddReg | Op::IaddCbuf | Op::IaddImm20 | Op::Iadd32iImm32 => iadd(tx, inst, opc)?,
        Op::IscaddReg | Op::IscaddCbuf | Op::IscaddImm20 | Op::Iscadd32iImm32 => {
            iscadd(tx, inst, opc)?
        }
        Op::ShlReg | Op::ShlCbuf | Op::ShlImm20 => shl(tx, inst)?,
        Op::ShrReg | Op::ShrCbuf | Op::ShrImm20 => shr(tx, inst)?,
        Op::BfeReg | Op::BfeCbuf | Op::BfeImm20 => bfe(tx, inst)?,
        Op::LopReg | Op::LopCbuf | Op::LopImm20 | Op::Lop32iImm32 => lop(tx, inst, opc)?,
        Op::PopcReg | Op::PopcCbuf | Op::PopcImm20 => popc(tx, inst)?,
        Op::FloReg | Op::FloCbuf | Op::FloImm20 => flo(tx, inst)?,
        Op::ImnmxReg | Op::ImnmxCbuf | Op::ImnmxImm20 => imnmx(tx, inst)?,
        Op::IsetpReg | Op::IsetpCbuf | Op::IsetpImm20 => isetp(tx, inst)?,
        Op::IsetReg | Op::IsetCbuf | Op::IsetImm20 => iset(tx, inst)?,
        Op::SelReg | Op::SelCbuf | Op::SelImm20 => sel(tx, inst)?,
        Op::MovReg | Op::MovCbuf | Op::MovImm20 => mov(tx, inst)?,
        Op::Mov32iImm32 => mov32i(tx, inst)?,
        Op::Psetp => psetp(tx, inst)?,
        Op::R2pReg | Op::R2pCbuf | Op::R2pImm20 => r2p(tx, inst)?,
        Op::P2rReg | Op::P2rCbuf | Op::P2rImm20 => p2r(tx, inst)?,
        Op::CsetpCc => csetp(tx, inst)?,
        Op::Pset => pset(tx, inst)?,
        Op::ImulReg | Op::ImulCbuf | Op::ImulImm20 | Op::Imul32iImm32 => imul(tx, inst, opc)?,
        _ => return Ok(false),
    }
    Ok(true)
}

// ---- operands -----------------------------------------------------------------------------

/// The second source as 32 raw bits, from a register, a constant buffer or an immediate (the
/// 20-bit form is sign-extended).
pub(super) fn src_b(tx: &mut Tx<'_>, inst: Inst) -> R<Id> {
    match inst.form() {
        Form::Reg => Ok(tx.read(inst.rb())),
        Form::Cbuf => tx.cbuf_word(inst.cbuf_index(), inst.cbuf_offset()),
        Form::Imm20 => Ok(tx.c32(inst.imm20() as u32)),
        Form::Imm32 => Ok(tx.c32(inst.imm32())),
        Form::None => tx.fail("an instruction without a second source operand"),
    }
}

/// The second source of a floating point instruction: as `src_b`, but the 20-bit immediate is the
/// top of a float.
fn src_b_f(tx: &mut Tx<'_>, inst: Inst) -> R<Id> {
    match inst.form() {
        Form::Imm20 => Ok(tx.cf32(f32::from_bits(inst.imm20_f32_bits()))),
        Form::Imm32 => Ok(tx.cf32(f32::from_bits(inst.imm32()))),
        _ => {
            let u = src_b(tx, inst)?;
            Ok(tx.as_f32(u))
        }
    }
}

/// Apply the absolute value and then the negation of a float source.
fn fmods(tx: &mut Tx<'_>, v: Id, neg: bool, abs: bool) -> Id {
    let v = if abs { tx.fabs(v) } else { v };
    if neg {
        tx.fneg(v)
    } else {
        v
    }
}

/// Clamp to 0..1, NaN to 0.
fn fsat(tx: &mut Tx<'_>, v: Id) -> Id {
    let zero = tx.cf32(0.0);
    let one = tx.cf32(1.0);
    let t = tx.t.f32;
    tx.ext(t, glsl::N_CLAMP, &[v, zero, one])
}

/// Refuse an instruction that writes the condition code (other than integer adds' carry).
pub(super) fn no_cc(tx: &Tx<'_>, inst: Inst, bit: u32) -> R<()> {
    if inst.bit(bit) {
        return tx.fail(format!(
            "{:?} writes the condition code flags",
            inst.op()
                .map_or("an unknown instruction".to_string(), |o| format!("{o:?}"))
        ));
    }
    Ok(())
}

/// The predicate source at bits `lo..lo+2` with its negation in bit `lo+3`.
fn pred_at(tx: &mut Tx<'_>, inst: Inst, lo: u32) -> Id {
    let p = inst.bits(lo, 3) as u8;
    tx.read_pred(p, inst.bit(lo + 3))
}

/// `a op b` for the boolean operation code of `PSETP`/`ISETP`: 0 and, 1 or, 2 xor.
fn bool_op(tx: &mut Tx<'_>, code: u64, a: Id, b: Id) -> R<Id> {
    Ok(match code {
        0 => tx.land(a, b),
        1 => tx.lor(a, b),
        2 => tx.lxor(a, b),
        _ => return tx.fail(format!("boolean operation code {code}")),
    })
}

/// A floating point comparison by the 4-bit code of `FSETP`.
fn fcompare(tx: &mut Tx<'_>, code: u64, a: Id, b: Id) -> R<Id> {
    let opcode = match code {
        0 => return Ok(tx.cfalse()),
        1 => op::F_ORD_LESS_THAN,
        2 => op::F_ORD_EQUAL,
        3 => op::F_ORD_LESS_THAN_EQUAL,
        4 => op::F_ORD_GREATER_THAN,
        5 => op::F_ORD_NOT_EQUAL,
        6 => op::F_ORD_GREATER_THAN_EQUAL,
        7 | 8 => {
            let t = tx.t.bool_;
            let na = tx.op1(op::IS_NAN, t, a);
            let nb = tx.op1(op::IS_NAN, t, b);
            let any = tx.lor(na, nb);
            return Ok(if code == 8 { any } else { tx.lnot(any) });
        }
        9 => op::F_UNORD_LESS_THAN,
        10 => op::F_UNORD_EQUAL,
        11 => op::F_UNORD_LESS_THAN_EQUAL,
        12 => op::F_UNORD_GREATER_THAN,
        13 => op::F_UNORD_NOT_EQUAL,
        14 => op::F_UNORD_GREATER_THAN_EQUAL,
        _ => return Ok(tx.ctrue()),
    };
    Ok(tx.cmp(opcode, a, b))
}

/// An integer comparison by the 3-bit code of `ISETP`.
fn icompare(tx: &mut Tx<'_>, code: u64, a: Id, b: Id, signed: bool) -> Id {
    let opcode = match (code, signed) {
        (0, _) => return tx.cfalse(),
        (1, true) => op::S_LESS_THAN,
        (1, false) => op::U_LESS_THAN,
        (2, _) => op::I_EQUAL,
        (3, true) => op::S_LESS_THAN_EQUAL,
        (3, false) => op::U_LESS_THAN_EQUAL,
        (4, true) => op::S_GREATER_THAN,
        (4, false) => op::U_GREATER_THAN,
        (5, _) => op::I_NOT_EQUAL,
        (6, true) => op::S_GREATER_THAN_EQUAL,
        (6, false) => op::U_GREATER_THAN_EQUAL,
        _ => return tx.ctrue(),
    };
    tx.cmp(opcode, a, b)
}

/// The value a `SET` instruction stores for a true comparison: 1.0f with the `bf` flag, else all ones.
fn set_value(tx: &mut Tx<'_>, cond: Id, bf: bool) -> Id {
    let t = if bf {
        tx.c32(1.0f32.to_bits())
    } else {
        tx.c32(0xFFFF_FFFF)
    };
    let f = tx.c32(0);
    tx.select_u32(cond, t, f)
}

// ---- floating point -----------------------------------------------------------------------

fn fadd(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let is32 = opc == Op::Fadd32iImm32;
    no_cc(tx, inst, if is32 { 52 } else { 47 })?;
    let (a_abs, a_neg, b_neg, b_abs) = if is32 {
        (54, 56, 53, 57)
    } else {
        (46, 48, 45, 49)
    };
    let a = tx.read_f(inst.ra());
    let a = fmods(tx, a, inst.bit(a_neg), inst.bit(a_abs));
    let b = src_b_f(tx, inst)?;
    let b = fmods(tx, b, inst.bit(b_neg), inst.bit(b_abs));
    let mut r = tx.fadd(a, b);
    if !is32 && inst.bit(50) {
        r = fsat(tx, r);
    }
    tx.write_f(inst, inst.rd(), r);
    Ok(())
}

/// `product == 0` zeroing for the multiply-by-zero flag: if either factor is zero the product is
/// zero, whatever the other is (NaN and infinity included).
fn fmz_product(tx: &mut Tx<'_>, a: Id, b: Id, product: Id) -> Id {
    let zero = tx.cf32(0.0);
    let az = tx.cmp(op::F_ORD_EQUAL, a, zero);
    let bz = tx.cmp(op::F_ORD_EQUAL, b, zero);
    let either = tx.lor(az, bz);
    let t = tx.t.f32;
    tx.select(t, either, zero, product)
}

fn fmul(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let is32 = opc == Op::Fmul32iImm32;
    no_cc(tx, inst, if is32 { 52 } else { 47 })?;
    let a = tx.read_f(inst.ra());
    let mut b = src_b_f(tx, inst)?;
    if !is32 && inst.bit(48) {
        b = tx.fneg(b);
    }
    let (fmz, sat) = if is32 {
        (inst.bit(54), inst.bit(55))
    } else {
        (inst.bit(45), inst.bit(50))
    };
    let mut r = tx.fmul(a, b);
    if fmz {
        r = fmz_product(tx, a, b, r);
    }
    if !is32 {
        let scale = match inst.bits(41, 3) {
            0 => None,
            1 => Some(0.5),
            2 => Some(0.25),
            3 => Some(0.125),
            4 => Some(8.0),
            5 => Some(4.0),
            6 => Some(2.0),
            _ => return tx.fail("an FMUL scale code of 7"),
        };
        if let Some(s) = scale {
            let c = tx.cf32(s);
            r = tx.fmul(r, c);
        }
    }
    if sat {
        r = fsat(tx, r);
    }
    tx.write_f(inst, inst.rd(), r);
    Ok(())
}

fn ffma(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let is32 = opc == Op::Ffma32iImm32;
    no_cc(tx, inst, if is32 { 52 } else { 47 })?;
    let a = tx.read_f(inst.ra());
    let (b, c) = match opc {
        Op::FfmaReg => (tx.read_f(inst.rb()), tx.read_f(inst.rc())),
        Op::FfmaCbuf_4980 => (src_b_f(tx, inst)?, tx.read_f(inst.rc())),
        Op::FfmaCbuf_5180 => {
            let c = src_b_f(tx, inst)?;
            (tx.read_f(inst.rc()), c)
        }
        Op::FfmaImm20 => (src_b_f(tx, inst)?, tx.read_f(inst.rc())),
        _ => (src_b_f(tx, inst)?, tx.read_f(inst.rd())),
    };
    let (neg_mul, neg_add, sat, fmz) = if is32 {
        (inst.bit(56), inst.bit(57), inst.bit(55), inst.bit(54))
    } else {
        (inst.bit(48), inst.bit(49), inst.bit(50), inst.bit(54))
    };
    let b = if neg_mul { tx.fneg(b) } else { b };
    let c = if neg_add { tx.fneg(c) } else { c };
    let mut r = tx.fma(a, b, c);
    if fmz {
        // With a zero factor the product is zero and only the addend is left.
        let product = tx.cf32(0.0);
        let z = fmz_product(tx, a, b, product);
        let zero_case = tx.fadd(z, c);
        let zero = tx.cf32(0.0);
        let az = tx.cmp(op::F_ORD_EQUAL, a, zero);
        let bz = tx.cmp(op::F_ORD_EQUAL, b, zero);
        let either = tx.lor(az, bz);
        let t = tx.t.f32;
        r = tx.select(t, either, zero_case, r);
    }
    if sat {
        r = fsat(tx, r);
    }
    tx.write_f(inst, inst.rd(), r);
    Ok(())
}

fn fmnmx(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    let a = tx.read_f(inst.ra());
    let a = fmods(tx, a, inst.bit(48), inst.bit(46));
    let b = src_b_f(tx, inst)?;
    let b = fmods(tx, b, inst.bit(45), inst.bit(49));
    let min_wanted = pred_at(tx, inst, 39);
    let f = tx.t.f32;
    let lo = tx.ext(f, glsl::N_MIN, &[a, b]);
    let hi = tx.ext(f, glsl::N_MAX, &[a, b]);
    let r = tx.select(f, min_wanted, lo, hi);
    tx.write_f(inst, inst.rd(), r);
    Ok(())
}

fn fsetp(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let a = tx.read_f(inst.ra());
    let a = fmods(tx, a, inst.bit(43), inst.bit(7));
    let b = src_b_f(tx, inst)?;
    let b = fmods(tx, b, inst.bit(6), inst.bit(44));
    let c = fcompare(tx, inst.bits(48, 4), a, b)?;
    let pc = pred_at(tx, inst, 39);
    let bop = inst.bits(45, 2);
    let r1 = bool_op(tx, bop, c, pc)?;
    let nc = tx.lnot(c);
    let r2 = bool_op(tx, bop, nc, pc)?;
    tx.write_pred(inst, inst.bits(3, 3) as u8, r1);
    tx.write_pred(inst, inst.bits(0, 3) as u8, r2);
    Ok(())
}

fn fset(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    let a = tx.read_f(inst.ra());
    let a = fmods(tx, a, inst.bit(43), inst.bit(54));
    let b = src_b_f(tx, inst)?;
    let b = fmods(tx, b, inst.bit(53), inst.bit(44));
    let c = fcompare(tx, inst.bits(48, 4), a, b)?;
    let pc = pred_at(tx, inst, 39);
    let r = bool_op(tx, inst.bits(45, 2), c, pc)?;
    let v = set_value(tx, r, inst.bit(52));
    tx.write(inst, inst.rd(), v);
    Ok(())
}

/// `MUFU.op Rd, ±|Ra|`: the special functions.
fn mufu(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    // The operand is `Ra`; bits 20..23 hold the function, not a register.
    let a = tx.read_f(inst.ra());
    let a = fmods(tx, a, inst.bit(48), inst.bit(46));
    let f = tx.t.f32;
    let mut r = match inst.bits(20, 4) {
        0 => tx.ext(f, glsl::COS, &[a]),
        1 => tx.ext(f, glsl::SIN, &[a]),
        2 => tx.ext(f, glsl::EXP2, &[a]),
        3 => tx.ext(f, glsl::LOG2, &[a]),
        4 => {
            let one = tx.cf32(1.0);
            tx.op2(op::F_DIV, f, one, a)
        }
        5 => tx.ext(f, glsl::INVERSE_SQRT, &[a]),
        8 => tx.ext(f, glsl::SQRT, &[a]),
        code => {
            return tx.fail(format!(
                "MUFU operation {code} (the double precision halves)"
            ))
        }
    };
    if inst.bit(50) {
        r = fsat(tx, r);
    }
    tx.write_f(inst, inst.rd(), r);
    Ok(())
}

/// `RRO`: the range reduction before `MUFU.SIN/COS/EX2` is the identity here, as `MUFU` takes the
/// plain argument (docs/ASSUMPTIONS.md).
fn rro(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let a = src_b_f(tx, inst)?;
    let a = fmods(tx, a, inst.bit(45), inst.bit(49));
    tx.write_f(inst, inst.rd(), a);
    Ok(())
}

// ---- conversions --------------------------------------------------------------------------

fn round_float(tx: &mut Tx<'_>, v: Id, mode: u64) -> Id {
    let f = tx.t.f32;
    match mode {
        0 => tx.ext(f, glsl::ROUND_EVEN, &[v]),
        1 => tx.ext(f, glsl::FLOOR, &[v]),
        2 => tx.ext(f, glsl::CEIL, &[v]),
        _ => tx.ext(f, glsl::TRUNC, &[v]),
    }
}

fn f2i(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bits(10, 2) != 2 {
        return tx.fail("F2I from a source that is not a 32-bit float");
    }
    let size = inst.bits(8, 2);
    let signed = inst.bit(12);
    if size == 3 {
        return tx.fail("F2I to a 64-bit integer");
    }
    if size < 2 {
        // 8 and 16 bits: saturate to the range of the destination, NaN gives zero. The result is
        // the integer in a 32-bit register, sign-extended for the signed types.
        let (lo, hi) = match (size, signed) {
            (0, false) => (0.0, 255.0),
            (0, true) => (-128.0, 127.0),
            (_, false) => (0.0, 65535.0),
            (_, true) => (-32768.0, 32767.0),
        };
        let src = src_b_f(tx, inst)?;
        let src = fmods(tx, src, inst.bit(45), inst.bit(49));
        let rounded = round_float(tx, src, inst.bits(39, 2));
        let f = tx.t.f32;
        let lo_c = tx.cf32(lo);
        let hi_c = tx.cf32(hi);
        let above_lo = tx.ext(f, glsl::N_MAX, &[rounded, lo_c]);
        let clamped = tx.ext(f, glsl::N_MIN, &[above_lo, hi_c]);
        let r = if signed {
            let i = tx.t.i32;
            let conv = tx.op1(op::CONVERT_F_TO_S, i, clamped);
            tx.as_u32(conv)
        } else {
            let u = tx.t.u32;
            tx.op1(op::CONVERT_F_TO_U, u, clamped)
        };
        let nan = tx.op1(op::IS_NAN, tx.t.bool_, rounded);
        let zero = tx.c32(0);
        let r = tx.select_u32(nan, zero, r);
        tx.write(inst, inst.rd(), r);
        return Ok(());
    }
    let src = src_b_f(tx, inst)?;
    let src = fmods(tx, src, inst.bit(45), inst.bit(49));
    let rounded = round_float(tx, src, inst.bits(39, 2));
    let f = tx.t.f32;
    let nan = tx.op1(op::IS_NAN, tx.t.bool_, rounded);
    let r = if signed {
        let lo = tx.cf32(-2147483648.0);
        let hi = tx.cf32(2147483648.0);
        let below = tx.cmp(op::F_ORD_LESS_THAN_EQUAL, rounded, lo);
        let above = tx.cmp(op::F_ORD_GREATER_THAN_EQUAL, rounded, hi);
        // Keep the conversion in range so it is defined.
        let top = tx_prev_float(tx, 2147483648.0);
        let safe = tx.ext(f, glsl::F_CLAMP, &[rounded, lo, top]);
        let i = tx.t.i32;
        let conv = tx.op1(op::CONVERT_F_TO_S, i, safe);
        let conv = tx.as_u32(conv);
        let min = tx.c32(0x8000_0000);
        let max = tx.c32(0x7FFF_FFFF);
        let r = tx.select_u32(above, max, conv);
        let r = tx.select_u32(below, min, r);
        let zero = tx.c32(0);
        tx.select_u32(nan, zero, r)
    } else {
        let zero_f = tx.cf32(0.0);
        let hi = tx.cf32(4294967296.0);
        let below = tx.cmp(op::F_ORD_LESS_THAN_EQUAL, rounded, zero_f);
        let above = tx.cmp(op::F_ORD_GREATER_THAN_EQUAL, rounded, hi);
        let top = tx_prev_float(tx, 4294967296.0);
        let safe = tx.ext(f, glsl::F_CLAMP, &[rounded, zero_f, top]);
        let u = tx.t.u32;
        let conv = tx.op1(op::CONVERT_F_TO_U, u, safe);
        let max = tx.c32(0xFFFF_FFFF);
        let r = tx.select_u32(above, max, conv);
        let zero = tx.c32(0);
        let r = tx.select_u32(below, zero, r);
        tx.select_u32(nan, zero, r)
    };
    tx.write(inst, inst.rd(), r);
    Ok(())
}

/// The largest float below `v` (a power of two): the top of the range a conversion can take.
fn tx_prev_float(tx: &mut Tx<'_>, v: f32) -> Id {
    tx.cf32(f32::from_bits(v.to_bits() - 1))
}

fn i2f(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bits(8, 2) != 2 {
        return tx.fail("I2F to a float that is not 32 bits");
    }
    // Source type: bits 10..11 size (0 8-bit, 1 16, 2 32, 3 64) and bit 13 the signedness.
    let size = inst.bits(10, 2);
    let signed = inst.bit(13);
    if size == 3 {
        return tx.fail("I2F from a 64-bit integer");
    }
    let raw = src_b(tx, inst)?;
    let mut v = raw;
    if size < 2 {
        // A byte or half-word chosen by bits 41..42 (in units of 8 bits).
        let sel = inst.bits(41, 2) as u32;
        let width = if size == 0 { 8 } else { 16 };
        let offset = tx.c32(sel * 8);
        let len = tx.c32(width);
        let t = tx.t.u32;
        v = if signed {
            let s = tx.as_i32(raw);
            let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, offset, len);
            tx.as_u32(e)
        } else {
            tx.op3(op::BIT_FIELD_U_EXTRACT, t, raw, offset, len)
        };
    }
    if inst.bit(49) && signed {
        let i = tx.as_i32(v);
        let a = tx.ext(tx.t.i32, glsl::S_ABS, &[i]);
        v = tx.as_u32(a);
    }
    if inst.bit(45) {
        v = tx.ineg(v);
    }
    let f = tx.t.f32;
    let r = if signed {
        let i = tx.as_i32(v);
        tx.op1(op::CONVERT_S_TO_F, f, i)
    } else {
        tx.op1(op::CONVERT_U_TO_F, f, v)
    };
    tx.write_f(inst, inst.rd(), r);
    Ok(())
}

fn f2f(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bits(8, 2) != 2 || inst.bits(10, 2) != 2 {
        return tx.fail("F2F between sizes other than 32 bits");
    }
    let src = src_b_f(tx, inst)?;
    let src = fmods(tx, src, inst.bit(45), inst.bit(49));
    // Bits 39..40 and 42: the rounding applied to the value (pass keeps it).
    let mode = inst.bits(39, 2) | inst.bits(42, 1) << 2;
    let mut r = match mode {
        0b011 => src,
        0b100 => round_float(tx, src, 0),
        0b101 => round_float(tx, src, 1),
        0b110 => round_float(tx, src, 2),
        0b111 => round_float(tx, src, 3),
        _ => src,
    };
    if inst.bit(50) {
        r = fsat(tx, r);
    }
    tx.write_f(inst, inst.rd(), r);
    Ok(())
}

fn i2i(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    // Destination: bits 8..9 size, bit 12 signed; source: bits 10..11 size, bit 13 signed.
    let dsize = inst.bits(8, 2);
    let dsigned = inst.bit(12);
    let ssize = inst.bits(10, 2);
    let ssigned = inst.bit(13);
    if dsize == 3 || ssize == 3 {
        return tx.fail("I2I with 64-bit integers");
    }
    let raw = src_b(tx, inst)?;
    let mut v = raw;
    if ssize < 2 {
        let sel = inst.bits(41, 2) as u32;
        let width = if ssize == 0 { 8 } else { 16 };
        let offset = tx.c32(sel * 8);
        let len = tx.c32(width);
        let t = tx.t.u32;
        v = if ssigned {
            let s = tx.as_i32(raw);
            let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, offset, len);
            tx.as_u32(e)
        } else {
            tx.op3(op::BIT_FIELD_U_EXTRACT, t, raw, offset, len)
        };
    }
    if inst.bit(49) {
        let i = tx.as_i32(v);
        let a = tx.ext(tx.t.i32, glsl::S_ABS, &[i]);
        v = tx.as_u32(a);
    }
    if inst.bit(45) {
        v = tx.ineg(v);
    }
    if dsize < 2 {
        let width = if dsize == 0 { 8u32 } else { 16 };
        let zero = tx.c32(0);
        let len = tx.c32(width);
        let t = tx.t.u32;
        if inst.bit(50) {
            // Saturate to the destination's range.
            let (lo, hi) = match (dsigned, width) {
                (true, 8) => (-128i32, 127i32),
                (true, _) => (-32768, 32767),
                (false, 8) => (0, 255),
                (false, _) => (0, 65535),
            };
            let s = if ssigned || dsigned {
                tx.as_i32(v)
            } else {
                // An unsigned source above the signed range is clamped to the top.
                tx.as_i32(v)
            };
            let lo_c = tx.ci32(lo);
            let hi_c = tx.ci32(hi);
            let clamped = tx.ext(tx.t.i32, glsl::S_MIN, &[s, hi_c]);
            let clamped = tx.ext(tx.t.i32, glsl::S_MAX, &[clamped, lo_c]);
            v = tx.as_u32(clamped);
        }
        v = if dsigned {
            let s = tx.as_i32(v);
            let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, zero, len);
            tx.as_u32(e)
        } else {
            tx.op3(op::BIT_FIELD_U_EXTRACT, t, v, zero, len)
        };
    }
    tx.write(inst, inst.rd(), v);
    // `.CC`: the zero and sign flags of the result, the carry and overflow flags cleared (the
    // flags are set even when the result goes nowhere: `I2I.CC RZ, R` tests a register).
    if inst.bit(47) {
        tx.set_zero_and_sign(inst, v, true);
    }
    Ok(())
}

// ---- integer ------------------------------------------------------------------------------

/// `a + b + carry_in`, with the carry out of the 32-bit addition.
pub(super) fn add_carry(tx: &mut Tx<'_>, a: Id, b: Id, carry_in: Option<Id>) -> (Id, Id) {
    let u = tx.t.u32;
    let pair = tx.m.ty_struct(&[u, u]);
    let one = tx.c32(1);
    let zero = tx.c32(0);
    let r1 = tx.op2(op::I_ADD_CARRY, pair, a, b);
    let s1 =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[r1, 0]);
    let c1 =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[r1, 1]);
    let Some(cin) = carry_in else {
        let carry = tx.cmp(op::I_NOT_EQUAL, c1, zero);
        return (s1, carry);
    };
    let cin_u = tx.select_u32(cin, one, zero);
    let r2 = tx.op2(op::I_ADD_CARRY, pair, s1, cin_u);
    let s2 =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[r2, 0]);
    let c2 =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[r2, 1]);
    let both = tx.or(c1, c2);
    let carry = tx.cmp(op::I_NOT_EQUAL, both, zero);
    (s2, carry)
}

fn iadd(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let is32 = opc == Op::Iadd32iImm32;
    let (neg_a, neg_b, sat, x, cc) = if is32 {
        (
            inst.bit(56),
            false,
            inst.bit(54),
            inst.bit(53),
            inst.bit(52),
        )
    } else {
        (
            inst.bit(49),
            inst.bit(48),
            inst.bit(50),
            inst.bit(43),
            inst.bit(47),
        )
    };
    if sat {
        return tx.fail("IADD with saturation");
    }
    let mut a = tx.read(inst.ra());
    let mut b = src_b(tx, inst)?;
    if !x && !cc {
        if neg_a {
            a = tx.ineg(a);
        }
        if neg_b {
            b = tx.ineg(b);
        }
        let r = tx.iadd(a, b);
        tx.write(inst, inst.rd(), r);
        return Ok(());
    }
    if neg_a {
        return tx.fail("IADD with a negated first operand and the carry flag");
    }
    // With the carry flag, a subtraction is a + !b + 1 (the 1 is the carry in unless the
    // extended form passes the previous carry on).
    let carry_var = tx.carry_variable();
    let bool_t = tx.t.bool_;
    let previous = if x {
        Some(tx.f.load(&mut tx.m, bool_t, carry_var))
    } else {
        None
    };
    let cin = if neg_b {
        b = tx.not(b);
        Some(previous.unwrap_or_else(|| tx.ctrue()))
    } else {
        previous
    };
    let (sum, carry) = add_carry(tx, a, b, cin);
    tx.write(inst, inst.rd(), sum);
    if cc {
        set_add_flags(tx, inst, a, b, sum, carry, x);
    }
    Ok(())
}

/// The flags of an addition `a + b` that gave `sum` and the carry out `carry`: carry, signed
/// overflow (the operands have one sign and the sum has the other), zero (for the extended form
/// only if the words before it were zero too) and sign. (docs/ASSUMPTIONS.md A-107)
pub(super) fn set_add_flags(
    tx: &mut Tx<'_>,
    inst: Inst,
    a: Id,
    b: Id,
    sum: Id,
    carry: Id,
    extended: bool,
) {
    let flags = tx.flags();
    tx.write_flag(inst, flags.c, carry);
    let sum_a = tx.xor(sum, a);
    let a_b = tx.xor(a, b);
    let not_a_b = tx.not(a_b);
    let overflow_bits = tx.and(sum_a, not_a_b);
    let thirty_one = tx.c32(31);
    let zero = tx.c32(0);
    let top = tx.shr(overflow_bits, thirty_one);
    let overflow = tx.cmp(op::I_NOT_EQUAL, top, zero);
    tx.write_flag(inst, flags.o, overflow);
    let mut is_zero = tx.cmp(op::I_EQUAL, sum, zero);
    if extended {
        let before = tx.read_flag(flags.z);
        is_zero = tx.land(is_zero, before);
    }
    tx.write_flag(inst, flags.z, is_zero);
    let sign = tx.shr(sum, thirty_one);
    let negative = tx.cmp(op::I_NOT_EQUAL, sign, zero);
    tx.write_flag(inst, flags.s, negative);
}

fn iscadd(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let is32 = opc == Op::Iscadd32iImm32;
    no_cc(tx, inst, if is32 { 52 } else { 47 })?;
    let mut a = tx.read(inst.ra());
    let mut b = src_b(tx, inst)?;
    let shift = if is32 {
        inst.bits(53, 5)
    } else {
        inst.bits(39, 5)
    } as u32;
    if !is32 {
        if inst.bit(49) {
            a = tx.ineg(a);
        }
        if inst.bit(48) {
            b = tx.ineg(b);
        }
    }
    let s = tx.c32(shift);
    let shifted = tx.shl(a, s);
    let r = tx.iadd(shifted, b);
    tx.write(inst, inst.rd(), r);
    Ok(())
}

/// The shift count and whether it wraps: with `.W` only the low 5 bits count, otherwise a count
/// of 32 or more shifts everything out.
fn shift_amount(tx: &mut Tx<'_>, count: Id, wrap: bool) -> (Id, Option<Id>) {
    if wrap {
        let mask = tx.c32(31);
        (tx.and(count, mask), None)
    } else {
        let limit = tx.c32(32);
        let over = tx.cmp(op::U_GREATER_THAN_EQUAL, count, limit);
        let mask = tx.c32(31);
        (tx.and(count, mask), Some(over))
    }
}

fn shl(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bit(43) {
        return tx.fail("SHL with the extended flag");
    }
    let a = tx.read(inst.ra());
    let count = src_b(tx, inst)?;
    let (amount, over) = shift_amount(tx, count, inst.bit(39));
    let mut r = tx.shl(a, amount);
    if let Some(over) = over {
        let zero = tx.c32(0);
        r = tx.select_u32(over, zero, r);
    }
    tx.write(inst, inst.rd(), r);
    Ok(())
}

fn shr(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bit(44) {
        return tx.fail("SHR with the extended flag");
    }
    let signed = inst.bit(48);
    let mut a = tx.read(inst.ra());
    if inst.bit(40) {
        a = tx.op1(op::BIT_REVERSE, tx.t.u32, a);
    }
    let count = src_b(tx, inst)?;
    let (amount, over) = shift_amount(tx, count, inst.bit(39));
    let r = if signed {
        let amount = match over {
            Some(over) => {
                let top = tx.c32(31);
                tx.select_u32(over, top, amount)
            }
            None => amount,
        };
        tx.sar(a, amount)
    } else {
        let r = tx.shr(a, amount);
        match over {
            Some(over) => {
                let zero = tx.c32(0);
                tx.select_u32(over, zero, r)
            }
            None => r,
        }
    };
    tx.write(inst, inst.rd(), r);
    Ok(())
}

/// `BFE.[S32|U32] Rd, Ra, B`: B holds the start in its low byte and the length in the next.
fn bfe(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    let signed = inst.bit(48);
    let mut a = tx.read(inst.ra());
    if inst.bit(40) {
        a = tx.op1(op::BIT_REVERSE, tx.t.u32, a);
    }
    let b = src_b(tx, inst)?;
    let mask = tx.c32(0xFF);
    let start = tx.and(b, mask);
    let eight = tx.c32(8);
    let shifted = tx.shr(b, eight);
    let len = tx.and(shifted, mask);
    let u = tx.t.u32;
    let zero = tx.c32(0);
    let thirty_two = tx.c32(32);
    // BitField*Extract is undefined for offset + count above 32, so the field is cut at bit 32.
    let room = tx.isub(thirty_two, start);
    let start_ok = tx.cmp(op::U_LESS_THAN, start, thirty_two);
    let safe_start = tx.select_u32(start_ok, start, zero);
    let too_long = tx.cmp(op::U_GREATER_THAN, len, room);
    let cut = tx.select_u32(too_long, room, len);
    let safe_len = tx.select_u32(start_ok, cut, zero);
    let r = if signed {
        let s = tx.as_i32(a);
        let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, safe_start, safe_len);
        let e = tx.as_u32(e);
        // A length of zero, or a start past the end, gives zero.
        let empty = tx.cmp(op::I_EQUAL, safe_len, zero);
        tx.select_u32(empty, zero, e)
    } else {
        tx.op3(op::BIT_FIELD_U_EXTRACT, u, a, safe_start, safe_len)
    };
    tx.write(inst, inst.rd(), r);
    Ok(())
}

fn lop(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let is32 = opc == Op::Lop32iImm32;
    let (logic, inv_a, inv_b, x, cc) = if is32 {
        (
            inst.bits(53, 2),
            inst.bit(55),
            inst.bit(56),
            inst.bit(57),
            inst.bit(52),
        )
    } else {
        (
            inst.bits(41, 2),
            inst.bit(39),
            inst.bit(40),
            inst.bit(43),
            inst.bit(47),
        )
    };
    if x || cc {
        return tx.fail("LOP with the extended or condition code flag");
    }
    let mut a = tx.read(inst.ra());
    let mut b = src_b(tx, inst)?;
    if inv_a {
        a = tx.not(a);
    }
    if inv_b {
        b = tx.not(b);
    }
    let r = match logic {
        0 => tx.and(a, b),
        1 => tx.or(a, b),
        2 => tx.xor(a, b),
        _ => b,
    };
    tx.write(inst, inst.rd(), r);
    if !is32 {
        // Bits 44..45: the predicate (bits 48..50, `PT` for none) becomes 0 false, 1 true, 2 "the
        // result is zero" or 3 "the result is not zero".
        let mode = inst.bits(44, 2);
        let p = inst.bits(48, 3) as u8;
        if p != PT {
            let zero = tx.c32(0);
            let value = match mode {
                0 => tx.cfalse(),
                1 => tx.ctrue(),
                2 => tx.cmp(op::I_EQUAL, r, zero),
                _ => tx.cmp(op::I_NOT_EQUAL, r, zero),
            };
            tx.write_pred(inst, p, value);
        }
    }
    Ok(())
}

fn popc(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let mut b = src_b(tx, inst)?;
    if inst.bit(40) {
        b = tx.not(b);
    }
    let u = tx.t.u32;
    let r = tx.op1(op::BIT_COUNT, u, b);
    tx.write(inst, inst.rd(), r);
    Ok(())
}

/// `FLO.[U32|S32].SH Rd, B`: the position of the highest set bit (or highest bit unlike the sign).
fn flo(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    let mut b = src_b(tx, inst)?;
    if inst.bit(40) {
        b = tx.not(b);
    }
    let signed = inst.bit(48);
    let u = tx.t.u32;
    let pos = if signed {
        let i = tx.as_i32(b);
        let e = tx.ext(tx.t.i32, glsl::FIND_S_MSB, &[i]);
        tx.as_u32(e)
    } else {
        tx.ext(u, glsl::FIND_U_MSB, &[b])
    };
    let r = if inst.bit(41) {
        // The shift to bring the bit to the top: 31 - position, and all ones when there is none.
        let top = tx.c32(31);
        let none = tx.c32(0xFFFF_FFFF);
        let missing = tx.cmp(op::I_EQUAL, pos, none);
        let shift = tx.isub(top, pos);
        tx.select_u32(missing, none, shift)
    } else {
        pos
    };
    tx.write(inst, inst.rd(), r);
    Ok(())
}

fn imnmx(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bits(43, 2) != 0 {
        return tx.fail("IMNMX of 16-bit lanes");
    }
    let signed = inst.bit(48);
    let a = tx.read(inst.ra());
    let b = src_b(tx, inst)?;
    let min_wanted = pred_at(tx, inst, 39);
    let u = tx.t.u32;
    let (lo, hi) = if signed {
        let sa = tx.as_i32(a);
        let sb = tx.as_i32(b);
        let lo = tx.ext(tx.t.i32, glsl::S_MIN, &[sa, sb]);
        let hi = tx.ext(tx.t.i32, glsl::S_MAX, &[sa, sb]);
        (tx.as_u32(lo), tx.as_u32(hi))
    } else {
        (
            tx.ext(u, glsl::U_MIN, &[a, b]),
            tx.ext(u, glsl::U_MAX, &[a, b]),
        )
    };
    let r = tx.select_u32(min_wanted, lo, hi);
    tx.write(inst, inst.rd(), r);
    Ok(())
}

fn isetp(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.bit(43) {
        return tx.fail("ISETP with the extended flag");
    }
    let signed = inst.bit(48);
    let a = tx.read(inst.ra());
    let b = src_b(tx, inst)?;
    let c = icompare(tx, inst.bits(49, 3), a, b, signed);
    let pc = pred_at(tx, inst, 39);
    let bop = inst.bits(45, 2);
    let r1 = bool_op(tx, bop, c, pc)?;
    let nc = tx.lnot(c);
    let r2 = bool_op(tx, bop, nc, pc)?;
    tx.write_pred(inst, inst.bits(3, 3) as u8, r1);
    tx.write_pred(inst, inst.bits(0, 3) as u8, r2);
    Ok(())
}

fn iset(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    if inst.bit(43) {
        return tx.fail("ISET with the extended flag");
    }
    let signed = inst.bit(48);
    let a = tx.read(inst.ra());
    let b = src_b(tx, inst)?;
    let c = icompare(tx, inst.bits(49, 3), a, b, signed);
    let pc = pred_at(tx, inst, 39);
    let r = bool_op(tx, inst.bits(45, 2), c, pc)?;
    let v = set_value(tx, r, inst.bit(44));
    tx.write(inst, inst.rd(), v);
    Ok(())
}

fn sel(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let a = tx.read(inst.ra());
    let b = src_b(tx, inst)?;
    let p = pred_at(tx, inst, 39);
    let r = tx.select_u32(p, a, b);
    tx.write(inst, inst.rd(), r);
    Ok(())
}

fn mov(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.bits(39, 4) != 0xF {
        return tx.fail("MOV with a lane mask");
    }
    let v = src_b(tx, inst)?;
    tx.write(inst, inst.rd(), v);
    // The word the register now holds, for a bindless texture instruction that takes its handle
    // from it.
    if inst.form() == Form::Cbuf && inst.guard().is_always() && inst.rd() != RZ {
        tx.reg_origin
            .insert(inst.rd(), (inst.cbuf_index(), inst.cbuf_offset() / 4));
    }
    // An immediate (or zero), for an instruction that takes texel offsets from the register.
    let constant = match inst.form() {
        Form::Imm20 => Some(inst.imm20() as u32),
        Form::Imm32 => Some(inst.imm32()),
        Form::Reg if inst.rb() == RZ => Some(0),
        _ => None,
    };
    if let Some(value) = constant {
        if inst.guard().is_always() && inst.rd() != RZ {
            tx.reg_const.insert(inst.rd(), value);
        }
    }
    Ok(())
}

fn mov32i(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.bits(12, 4) != 0xF {
        return tx.fail("MOV32I with a lane mask");
    }
    let v = tx.c32(inst.imm32());
    tx.write(inst, inst.rd(), v);
    if inst.guard().is_always() && inst.rd() != RZ {
        tx.reg_const.insert(inst.rd(), inst.imm32());
    }
    Ok(())
}

fn psetp(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let a = pred_at(tx, inst, 12);
    let b = pred_at(tx, inst, 29);
    let c = pred_at(tx, inst, 39);
    let first = bool_op(tx, inst.bits(24, 2), a, b)?;
    let r1 = bool_op(tx, inst.bits(45, 2), first, c)?;
    let na = tx.lnot(a);
    let second = bool_op(tx, inst.bits(24, 2), na, b)?;
    let r2 = bool_op(tx, inst.bits(45, 2), second, c)?;
    tx.write_pred(inst, inst.bits(3, 3) as u8, r1);
    tx.write_pred(inst, inst.bits(0, 3) as u8, r2);
    Ok(())
}

/// `R2P PR, Ra, mask`: each of the predicates P0 to P6 that `mask` names takes the bit of `Ra` that
/// has its number within the byte the instruction selects (bits 41 and 42), the others keep their
/// value (docs/ASSUMPTIONS.md A-154).
fn r2p(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.bit(40) {
        return tx.fail("R2P to the condition code flags");
    }
    let shift = 8 * inst.bits(41, 2) as u32;
    let value = tx.read(inst.ra());
    let literal = (inst.form() == Form::Imm20).then(|| inst.imm20() as u32);
    let mask = src_b(tx, inst)?;
    let zero = tx.c32(0);
    let b = tx.t.bool_;
    for p in 0..7u8 {
        if literal.is_some_and(|m| m >> p & 1 == 0) {
            continue;
        }
        let bit = tx.c32(1 << (shift + u32::from(p)));
        let bits = tx.and(value, bit);
        let taken = tx.cmp(op::I_NOT_EQUAL, bits, zero);
        let new = if literal.is_some() {
            taken
        } else {
            let named = tx.c32(1 << p);
            let masked = tx.and(mask, named);
            let wanted = tx.cmp(op::I_NOT_EQUAL, masked, zero);
            let old = tx.read_pred(p, false);
            tx.select(b, wanted, taken, old)
        };
        tx.write_pred(inst, p, new);
    }
    Ok(())
}

/// `P2R Rd, PR, Ra, mask`: `Rd` is `Ra` with the bits of the byte the instruction selects that
/// `mask` names replaced by the predicates, P0 in the lowest bit and PT (always true) in bit 7
/// (docs/ASSUMPTIONS.md A-154).
fn p2r(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.bit(40) {
        return tx.fail("P2R of the condition code flags");
    }
    let shift = tx.c32(8 * inst.bits(41, 2) as u32);
    let value = tx.read(inst.ra());
    let mask = src_b(tx, inst)?;
    let byte = tx.c32(0xFF);
    let mask = tx.and(mask, byte);
    let mut predicates = tx.c32(0x80);
    let zero = tx.c32(0);
    for p in 0..7u8 {
        let on = tx.read_pred(p, false);
        let bit = tx.c32(1 << p);
        let set = tx.select_u32(on, bit, zero);
        predicates = tx.or(predicates, set);
    }
    let named = tx.and(predicates, mask);
    let moved = tx.shl(named, shift);
    let window = tx.shl(mask, shift);
    let kept = tx.not(window);
    let kept = tx.and(value, kept);
    let result = tx.or(kept, moved);
    tx.write(inst, inst.rd(), result);
    Ok(())
}

/// A test of the condition code flags by the 5-bit code of `CSETP` and the control-flow
/// instructions (docs/ASSUMPTIONS.md A-107): `S` is the sign (negative) flag, `Z` zero, `C` carry
/// and `O` overflow. The codes that name states of the hardware's own are refused.
fn condition(tx: &mut Tx<'_>, code: u64) -> R<Id> {
    if code == 15 {
        return Ok(tx.ctrue());
    }
    if code == 0 {
        return Ok(tx.cfalse());
    }
    let flags = tx.flags();
    let (z, s, c, o) = (
        tx.read_flag(flags.z),
        tx.read_flag(flags.s),
        tx.read_flag(flags.c),
        tx.read_flag(flags.o),
    );
    let (nz, ns, nc, no) = (tx.lnot(z), tx.lnot(s), tx.lnot(c), tx.lnot(o));
    Ok(match code {
        // LT: negative and not zero, as a signed comparison (overflow flips it).
        1 => {
            let lt = tx.land(s, nz);
            tx.lxor(lt, o)
        }
        2 => tx.land(ns, z),
        3 => {
            let zo = tx.lor(z, o);
            tx.lxor(s, zo)
        }
        4 => {
            let so = tx.lxor(s, o);
            let le = tx.lor(so, z);
            tx.lnot(le)
        }
        5 => nz,
        6 => {
            // GE: the sign and the overflow agree.
            let so = tx.lxor(s, o);
            tx.lnot(so)
        }
        7 => {
            let nan = tx.land(s, z);
            tx.lnot(nan)
        }
        8 => tx.land(s, z),
        9 => tx.lxor(s, o),
        10 => z,
        11 => {
            let so = tx.lxor(s, o);
            tx.lor(so, z)
        }
        12 => {
            let oz = tx.lor(o, z);
            tx.lxor(ns, oz)
        }
        13 => tx.lor(s, nz),
        14 => {
            let nsz = tx.lor(ns, z);
            tx.lxor(nsz, o)
        }
        16 => no,
        17 => nc,
        18 => ns,
        19 => tx.lor(z, nc),
        20 => tx.land(c, nz),
        21 => s,
        22 => c,
        23 => o,
        30 => tx.lor(s, z),
        31 => {
            let sz = tx.lor(s, z);
            tx.lnot(sz)
        }
        _ => return tx.fail(format!("condition code test {code}")),
    })
}

/// `CSETP.cc.op Pd, Pq, CC.test, Pc`: `Pd` is the test combined with `Pc` by the boolean
/// operation and `Pq` the same for the test's negation.
fn csetp(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let test = condition(tx, inst.bits(8, 5))?;
    let c = pred_at(tx, inst, 39);
    let first = bool_op(tx, inst.bits(45, 2), test, c)?;
    let negated = tx.lnot(test);
    let second = bool_op(tx, inst.bits(45, 2), negated, c)?;
    tx.write_pred(inst, inst.bits(3, 3) as u8, first);
    tx.write_pred(inst, inst.bits(0, 3) as u8, second);
    Ok(())
}

fn pset(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    no_cc(tx, inst, 47)?;
    let a = pred_at(tx, inst, 12);
    let b = pred_at(tx, inst, 29);
    let c = pred_at(tx, inst, 39);
    let first = bool_op(tx, inst.bits(24, 2), a, b)?;
    let r = bool_op(tx, inst.bits(45, 2), first, c)?;
    let v = set_value(tx, r, inst.bit(44));
    tx.write(inst, inst.rd(), v);
    Ok(())
}

/// `IMUL.[U32|S32].[U32|S32].HI Rd, Ra, B`: the low or the high word of the 64-bit product.
fn imul(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    let is32 = opc == Op::Imul32iImm32;
    no_cc(tx, inst, if is32 { 52 } else { 47 })?;
    let (hi, a_signed, b_signed) = if is32 {
        (inst.bit(53), inst.bit(55), inst.bit(54))
    } else {
        (inst.bit(39), inst.bit(40), inst.bit(41))
    };
    let a = tx.read(inst.ra());
    let b = src_b(tx, inst)?;
    let r = if !hi {
        tx.imul(a, b)
    } else if a_signed && b_signed {
        mul_high(tx, a, b, true)
    } else if !a_signed && !b_signed {
        mul_high(tx, a, b, false)
    } else {
        return tx.fail("IMUL.HI of a signed and an unsigned operand");
    };
    tx.write(inst, inst.rd(), r);
    Ok(())
}

fn mul_high(tx: &mut Tx<'_>, a: Id, b: Id, signed: bool) -> Id {
    let u = tx.t.u32;
    if signed {
        let i = tx.t.i32;
        let pair = tx.m.ty_struct(&[i, i]);
        let sa = tx.as_i32(a);
        let sb = tx.as_i32(b);
        let r = tx.op2(op::S_MUL_EXTENDED, pair, sa, sb);
        let hi =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, i, &[r, 1]);
        tx.as_u32(hi)
    } else {
        let pair = tx.m.ty_struct(&[u, u]);
        let r = tx.op2(op::U_MUL_EXTENDED, pair, a, b);
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[r, 1])
    }
}

/// `PT` as a predicate index, for callers that need to name it.
#[allow(dead_code)]
pub(super) const TRUE_PREDICATE: u8 = PT;
