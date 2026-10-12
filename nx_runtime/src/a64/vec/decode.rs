//! Decoding of the scalar floating-point and Advanced SIMD encoding space.
//!
//! The groups follow the Arm "Data Processing -- Scalar Floating-Point and Advanced SIMD" and
//! "Loads and Stores" encoding tables. As in the integer decoder, a reserved or unallocated pattern
//! is [`Reject::Unallocated`], and a valid encoding outside what is implemented is
//! [`Reject::Unsupported`]: nothing is silently ignored. Should-be-zero fields that are not zero
//! are rejected (see docs/UNICORN_DISAGREEMENTS.md for the encodings where Unicorn is laxer).
//!
//! The Cortex-A57 has no half-precision arithmetic (FEAT_FP16), so every half-precision form other
//! than the precision conversions is unallocated.

use super::{FpBinary, FpUnary, Reject, VAddr, VMem, VOp};
use crate::a64::fp::{Fmt, MinMax, Rounding};
use crate::a64::inst::AddrMode;

type Res<T> = Result<T, Reject>;

fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & (u32::MAX >> (31 - (hi - lo)))
}

fn bit(w: u32, n: u32) -> bool {
    (w >> n) & 1 != 0
}

fn simm(w: u32, hi: u32, lo: u32) -> i64 {
    let len = hi - lo + 1;
    let v = i64::from(bits(w, hi, lo));
    (v << (64 - len)) >> (64 - len)
}

const UNALLOC: Reject = Reject::Unallocated;

/// A floating-point type field for arithmetic: single or double (half is FEAT_FP16, absent).
fn arith_fmt(ftype: u32) -> Res<Fmt> {
    match ftype {
        0 => Ok(Fmt::S),
        1 => Ok(Fmt::D),
        _ => Err(UNALLOC),
    }
}

/// A floating-point type field for conversions between precisions: half is allowed.
fn conv_fmt(ftype: u32) -> Res<Fmt> {
    match ftype {
        0 => Ok(Fmt::S),
        1 => Ok(Fmt::D),
        3 => Ok(Fmt::H),
        _ => Err(UNALLOC),
    }
}

/// Decode a word of the data-processing floating-point and Advanced SIMD space
/// (`bits 27:25 == 111`).
pub fn decode(w: u32) -> Res<VOp> {
    let op0 = bits(w, 31, 28);
    // Scalar floating point: M 0 S 1 xxxx with M = S = 0 (or the conversions, where bit 31 is sf).
    if op0 & 0b0101 == 0b0001 {
        return decode_scalar_fp(w);
    }
    Err(Reject::Unsupported("Advanced SIMD"))
}

fn decode_scalar_fp(w: u32) -> Res<VOp> {
    if bit(w, 29) {
        return Err(UNALLOC); // S = 1
    }
    if bit(w, 24) {
        return fp_three_source(w);
    }
    if !bit(w, 21) {
        return fp_fixed_point(w);
    }
    // The remaining groups need bit 31 (M) clear, except the integer conversions.
    match bits(w, 11, 10) {
        0b01 => return fp_cond_compare(w),
        0b10 => return fp_two_source(w),
        0b11 => return fp_cond_select(w),
        _ => {}
    }
    if bit(w, 12) {
        return fp_immediate(w);
    }
    if bit(w, 13) {
        return fp_compare(w);
    }
    if bit(w, 14) {
        return fp_one_source(w);
    }
    if bit(w, 15) {
        return Err(UNALLOC);
    }
    fp_int_conversion(w)
}

fn fp_three_source(w: u32) -> Res<VOp> {
    if bit(w, 31) {
        return Err(UNALLOC);
    }
    let fmt = arith_fmt(bits(w, 23, 22))?;
    let o1 = bit(w, 21);
    let o0 = bit(w, 15);
    Ok(VOp::FpFused {
        neg_addend: o1,
        neg_product: o1 != o0,
        fmt,
        rd: bits(w, 4, 0) as u8,
        rn: bits(w, 9, 5) as u8,
        rm: bits(w, 20, 16) as u8,
        ra: bits(w, 14, 10) as u8,
    })
}

fn fp_two_source(w: u32) -> Res<VOp> {
    if bit(w, 31) {
        return Err(UNALLOC);
    }
    let fmt = arith_fmt(bits(w, 23, 22))?;
    let op = match bits(w, 15, 12) {
        0b0000 => FpBinary::Mul,
        0b0001 => FpBinary::Div,
        0b0010 => FpBinary::Add,
        0b0011 => FpBinary::Sub,
        0b0100 => FpBinary::MinMax(MinMax::Max),
        0b0101 => FpBinary::MinMax(MinMax::Min),
        0b0110 => FpBinary::MinMax(MinMax::MaxNum),
        0b0111 => FpBinary::MinMax(MinMax::MinNum),
        0b1000 => FpBinary::Nmul,
        _ => return Err(UNALLOC),
    };
    Ok(VOp::FpBinary {
        op,
        fmt,
        rd: bits(w, 4, 0) as u8,
        rn: bits(w, 9, 5) as u8,
        rm: bits(w, 20, 16) as u8,
    })
}

fn fp_one_source(w: u32) -> Res<VOp> {
    if bit(w, 31) {
        return Err(UNALLOC);
    }
    let ftype = bits(w, 23, 22);
    let rd = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    let opcode = bits(w, 20, 15);
    // FCVT: opcode 0001oo, oo the destination precision.
    if opcode & 0b111100 == 0b000100 {
        let from = conv_fmt(ftype)?;
        let to = match opcode & 3 {
            0 => Fmt::S,
            1 => Fmt::D,
            3 => Fmt::H,
            _ => return Err(UNALLOC),
        };
        if from == to {
            return Err(UNALLOC);
        }
        return Ok(VOp::FpConvert { from, to, rd, rn });
    }
    let fmt = arith_fmt(ftype)?;
    let round = |mode: Option<Rounding>, exact: bool| FpUnary::Round { mode, exact };
    let op = match opcode {
        0b000000 => return Ok(VOp::FpMov { fmt, rd, rn }),
        0b000001 => FpUnary::Abs,
        0b000010 => FpUnary::Neg,
        0b000011 => FpUnary::Sqrt,
        0b001000 => round(Some(Rounding::Nearest), false),
        0b001001 => round(Some(Rounding::PlusInf), false),
        0b001010 => round(Some(Rounding::MinusInf), false),
        0b001011 => round(Some(Rounding::Zero), false),
        0b001100 => round(Some(Rounding::NearestAway), false),
        0b001110 => round(None, true),
        0b001111 => round(None, false),
        _ => return Err(UNALLOC),
    };
    Ok(VOp::FpUnary { op, fmt, rd, rn })
}

fn fp_compare(w: u32) -> Res<VOp> {
    if bit(w, 31) || bits(w, 15, 14) != 0 || bits(w, 2, 0) != 0 {
        return Err(UNALLOC);
    }
    let fmt = arith_fmt(bits(w, 23, 22))?;
    let opcode2 = bits(w, 4, 3);
    let zero = opcode2 & 1 != 0;
    let rm = bits(w, 20, 16) as u8;
    if zero && rm != 0 {
        return Err(UNALLOC);
    }
    Ok(VOp::FpCompare {
        fmt,
        rn: bits(w, 9, 5) as u8,
        rm: if zero { None } else { Some(rm) },
        signal: opcode2 & 2 != 0,
    })
}

fn fp_immediate(w: u32) -> Res<VOp> {
    if bit(w, 31) || bits(w, 9, 5) != 0 {
        return Err(UNALLOC);
    }
    Ok(VOp::FpImmediate {
        fmt: arith_fmt(bits(w, 23, 22))?,
        rd: bits(w, 4, 0) as u8,
        imm8: bits(w, 20, 13) as u8,
    })
}

fn fp_cond_compare(w: u32) -> Res<VOp> {
    if bit(w, 31) {
        return Err(UNALLOC);
    }
    Ok(VOp::FpCondCompare {
        fmt: arith_fmt(bits(w, 23, 22))?,
        rn: bits(w, 9, 5) as u8,
        rm: bits(w, 20, 16) as u8,
        cond: bits(w, 15, 12) as u8,
        nzcv: bits(w, 3, 0) as u8,
        signal: bit(w, 4),
    })
}

fn fp_cond_select(w: u32) -> Res<VOp> {
    if bit(w, 31) {
        return Err(UNALLOC);
    }
    Ok(VOp::FpSelect {
        fmt: arith_fmt(bits(w, 23, 22))?,
        rd: bits(w, 4, 0) as u8,
        rn: bits(w, 9, 5) as u8,
        rm: bits(w, 20, 16) as u8,
        cond: bits(w, 15, 12) as u8,
    })
}

/// Conversions between floating point and fixed point.
fn fp_fixed_point(w: u32) -> Res<VOp> {
    let wide = bit(w, 31);
    let scale = bits(w, 15, 10);
    if !wide && scale < 32 {
        return Err(UNALLOC);
    }
    let fmt = arith_fmt(bits(w, 23, 22))?;
    let rmode = bits(w, 20, 19);
    let opcode = bits(w, 18, 16);
    let fbits = 64 - scale;
    let (rd, rn) = (bits(w, 4, 0) as u8, bits(w, 9, 5) as u8);
    match (rmode, opcode) {
        (0b00, 0b010 | 0b011) => Ok(VOp::IntToFp {
            fmt,
            wide,
            signed: opcode == 0b010,
            fbits,
            rd,
            rn,
        }),
        (0b11, 0b000 | 0b001) => Ok(VOp::FpToInt {
            fmt,
            wide,
            signed: opcode == 0b000,
            mode: Rounding::Zero,
            fbits,
            rn,
            rd,
        }),
        _ => Err(UNALLOC),
    }
}

/// Conversions between floating point and the general registers.
fn fp_int_conversion(w: u32) -> Res<VOp> {
    let wide = bit(w, 31);
    let ftype = bits(w, 23, 22);
    let rmode = bits(w, 20, 19);
    let opcode = bits(w, 18, 16);
    let (rd, rn) = (bits(w, 4, 0) as u8, bits(w, 9, 5) as u8);
    // FMOV between a general register and the top half of a 128-bit register.
    if ftype == 0b10 {
        if !wide || rmode != 0b01 {
            return Err(UNALLOC);
        }
        return match opcode {
            0b110 => Ok(VOp::FmovToGeneral {
                wide: true,
                high: true,
                rn,
                rd,
            }),
            0b111 => Ok(VOp::FmovFromGeneral {
                wide: true,
                high: true,
                rd,
                rn,
            }),
            _ => Err(UNALLOC),
        };
    }
    let fmt = arith_fmt(ftype)?;
    // A single converts only to or from a W register, a double only to or from an X register,
    // except the integer-to-float and float-to-integer conversions, which take either.
    let to_general = |mode: Rounding, signed: bool| VOp::FpToInt {
        fmt,
        wide,
        signed,
        mode,
        fbits: 0,
        rn,
        rd,
    };
    match (rmode, opcode) {
        (0b00, 0b000) => Ok(to_general(Rounding::Nearest, true)),
        (0b00, 0b001) => Ok(to_general(Rounding::Nearest, false)),
        (0b01, 0b000) => Ok(to_general(Rounding::PlusInf, true)),
        (0b01, 0b001) => Ok(to_general(Rounding::PlusInf, false)),
        (0b10, 0b000) => Ok(to_general(Rounding::MinusInf, true)),
        (0b10, 0b001) => Ok(to_general(Rounding::MinusInf, false)),
        (0b11, 0b000) => Ok(to_general(Rounding::Zero, true)),
        (0b11, 0b001) => Ok(to_general(Rounding::Zero, false)),
        (0b00, 0b100) => Ok(to_general(Rounding::NearestAway, true)),
        (0b00, 0b101) => Ok(to_general(Rounding::NearestAway, false)),
        (0b00, 0b010 | 0b011) => Ok(VOp::IntToFp {
            fmt,
            wide,
            signed: opcode == 0b010,
            fbits: 0,
            rd,
            rn,
        }),
        (0b00, 0b110) if wide == (fmt == Fmt::D) => Ok(VOp::FmovToGeneral {
            wide,
            high: false,
            rn,
            rd,
        }),
        (0b00, 0b111) if wide == (fmt == Fmt::D) => Ok(VOp::FmovFromGeneral {
            wide,
            high: false,
            rd,
            rn,
        }),
        (0b11, 0b110) if !wide && fmt == Fmt::D => Err(Reject::Unsupported("FJCVTZS (FEAT_JSCVT)")),
        _ => Err(UNALLOC),
    }
}

// ---------------------------------------------------------------------------------------------
// Loads and stores of vector registers
// ---------------------------------------------------------------------------------------------

/// Decode a load/store word with the `V` bit set.
pub fn decode_mem(w: u32) -> Res<(VMem, VAddr)> {
    match bits(w, 29, 27) {
        0b011 => literal(w),
        0b101 => pair(w),
        0b111 => single(w),
        0b001 => Err(Reject::Unsupported("Advanced SIMD load/store structures")),
        _ => Err(UNALLOC),
    }
}

fn literal(w: u32) -> Res<(VMem, VAddr)> {
    if bits(w, 25, 24) != 0 {
        return Err(UNALLOC);
    }
    let bytes = match bits(w, 31, 30) {
        0 => 4,
        1 => 8,
        2 => 16,
        _ => return Err(UNALLOC),
    };
    Ok((
        VMem::Single {
            load: true,
            bytes,
            rt: bits(w, 4, 0) as u8,
        },
        VAddr {
            rn: 0,
            mode: AddrMode::Literal(simm(w, 23, 5) << 2),
        },
    ))
}

fn pair(w: u32) -> Res<(VMem, VAddr)> {
    let scale = match bits(w, 31, 30) {
        0 => 2,
        1 => 3,
        2 => 4,
        _ => return Err(UNALLOC),
    };
    let imm = simm(w, 21, 15) << scale;
    let mode = match bits(w, 24, 23) {
        0b00 | 0b10 => AddrMode::Offset(imm),
        0b01 => AddrMode::PostIndex(imm),
        _ => AddrMode::PreIndex(imm),
    };
    Ok((
        VMem::Pair {
            load: bit(w, 22),
            bytes: 1 << scale,
            rt: bits(w, 4, 0) as u8,
            rt2: bits(w, 14, 10) as u8,
        },
        VAddr {
            rn: bits(w, 9, 5) as u8,
            mode,
        },
    ))
}

fn single(w: u32) -> Res<(VMem, VAddr)> {
    let size = bits(w, 31, 30);
    let opc = bits(w, 23, 22);
    let (load, scale) = match (size, opc) {
        (_, 0) => (false, size),
        (_, 1) => (true, size),
        (0, 2) => (false, 4),
        (0, 3) => (true, 4),
        _ => return Err(UNALLOC),
    };
    let mode = if bit(w, 24) {
        AddrMode::Offset(i64::from(bits(w, 21, 10)) << scale)
    } else if !bit(w, 21) {
        let imm9 = simm(w, 20, 12);
        match bits(w, 11, 10) {
            0b00 => AddrMode::Offset(imm9),
            0b01 => AddrMode::PostIndex(imm9),
            0b11 => AddrMode::PreIndex(imm9),
            _ => return Err(UNALLOC), // there is no unprivileged vector load or store
        }
    } else if bits(w, 11, 10) == 0b10 {
        let option = bits(w, 15, 13);
        if option & 0b010 == 0 {
            return Err(UNALLOC);
        }
        use crate::a64::inst::ExtendType::*;
        let extend = match option {
            0b010 => Uxtw,
            0b011 => Uxtx,
            0b110 => Sxtw,
            _ => Sxtx,
        };
        AddrMode::Register {
            rm: bits(w, 20, 16) as u8,
            extend,
            amount: if bit(w, 12) { scale as u8 } else { 0 },
        }
    } else {
        return Err(UNALLOC);
    };
    Ok((
        VMem::Single {
            load,
            bytes: 1 << scale,
            rt: bits(w, 4, 0) as u8,
        },
        VAddr {
            rn: bits(w, 9, 5) as u8,
            mode,
        },
    ))
}
