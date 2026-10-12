//! Decoding of the Advanced SIMD vector groups: the register-register groups here, the groups
//! with immediates and the scalar groups in `decode_simd_imm`.
//!
//! The field layouts follow the Arm "Data Processing -- Scalar Floating-Point and Advanced SIMD"
//! tables. Encodings the architecture leaves unallocated, and the ones only later architecture
//! versions allocate (half-precision arithmetic, the dot-product, complex-number and rounding
//! multiply-accumulate extensions), are [`Reject::Unallocated`] because the Cortex-A57 lacks them;
//! the cryptographic extension is [`Reject::Unsupported`].

use super::decode::{bit, bits, Res, UNALLOC};
use super::simd::*;
use super::Reject;
use crate::a64::fp::{Fmt, MinMax, Rounding};

pub(super) fn rd(w: u32) -> u8 {
    bits(w, 4, 0) as u8
}

pub(super) fn rn(w: u32) -> u8 {
    bits(w, 9, 5) as u8
}

pub(super) fn rm(w: u32) -> u8 {
    bits(w, 20, 16) as u8
}

const CRYPTO: Reject = Reject::Unsupported("cryptographic extension");

/// Decode a word of the Advanced SIMD space that is not a scalar floating-point instruction.
pub fn decode_simd(w: u32) -> Res<SimdOp> {
    if bit(w, 31) {
        return Err(UNALLOC);
    }
    if bit(w, 28) {
        return super::decode_simd_imm::scalar(w);
    }
    if bit(w, 24) {
        // x indexed element (bit 10 clear), or shift / modified immediate (bit 10 set).
        if !bit(w, 10) {
            return super::decode_simd_imm::vector_indexed(w);
        }
        if bit(w, 23) {
            return Err(UNALLOC);
        }
        return if bits(w, 22, 19) == 0 {
            modified_immediate(w)
        } else {
            super::decode_simd_imm::vector_shift(w)
        };
    }
    if bit(w, 21) {
        if bit(w, 10) {
            return three_same(w);
        }
        if !bit(w, 11) {
            return three_different(w);
        }
        return match bits(w, 20, 17) {
            0b0000 => two_reg_misc(w),
            0b1000 => across_lanes(w),
            0b0100 => crypto_aes(w),
            _ => Err(UNALLOC),
        };
    }
    // Bit 21 clear.
    if bit(w, 15) {
        return Err(UNALLOC); // three-register extension: dot product, RDM, complex numbers
    }
    if bit(w, 10) {
        return if bits(w, 23, 22) == 0 {
            copy(w)
        } else {
            Err(UNALLOC) // includes the half-precision three-same group
        };
    }
    if bit(w, 29) {
        return extract(w);
    }
    if bit(w, 11) {
        permute(w)
    } else {
        table_lookup(w)
    }
}

fn crypto_aes(w: u32) -> Res<SimdOp> {
    // 0100 1110 size 10100 opcode 10 Rn Rd, with opcode 00100..00111 and size 0.
    if bits(w, 31, 24) == 0x4E
        && bits(w, 23, 22) == 0
        && (0b00100..=0b00111).contains(&bits(w, 16, 12))
    {
        Err(CRYPTO)
    } else {
        Err(UNALLOC)
    }
}

// ---------------------------------------------------------------------------------------------
// Three registers, same element size
// ---------------------------------------------------------------------------------------------

fn three_same(w: u32) -> Res<SimdOp> {
    let (q, u, size, opcode) = (bit(w, 30), bit(w, 29), bits(w, 23, 22), bits(w, 15, 11));
    let (rd, rn, rm) = (rd(w), rn(w), rm(w));
    if opcode >= 0b11000 {
        return fp_three_same(w, q, u, size, opcode);
    }
    if opcode == 0b00011 {
        let op = if u {
            [Logic::Eor, Logic::Bsl, Logic::Bit, Logic::Bif][size as usize]
        } else {
            [Logic::And, Logic::Bic, Logic::Orr, Logic::Orn][size as usize]
        };
        return Ok(SimdOp::Logic { op, q, rd, rn, rm });
    }
    if size == 3 && !q {
        return Err(UNALLOC);
    }
    let sh = Shape::vector(size, q);
    let narrow_only = size != 3;
    use IntBin::*;
    let pick = |s: IntBin, un: IntBin| if u { un } else { s };
    let (op, pair) = match opcode {
        0b00000 if narrow_only => (pick(Shadd, Uhadd), false),
        0b00001 => (pick(Sqadd, Uqadd), false),
        0b00010 if narrow_only => (pick(Srhadd, Urhadd), false),
        0b00100 if narrow_only => (pick(Shsub, Uhsub), false),
        0b00101 => (pick(Sqsub, Uqsub), false),
        0b00110 => (pick(Cmgt, Cmhi), false),
        0b00111 => (pick(Cmge, Cmhs), false),
        0b01000 => (pick(Sshl, Ushl), false),
        0b01001 => (pick(Sqshl, Uqshl), false),
        0b01010 => (pick(Srshl, Urshl), false),
        0b01011 => (pick(Sqrshl, Uqrshl), false),
        0b01100 if narrow_only => (pick(Smax, Umax), false),
        0b01101 if narrow_only => (pick(Smin, Umin), false),
        0b01110 if narrow_only => (pick(Sabd, Uabd), false),
        0b01111 if narrow_only => (pick(Saba, Uaba), false),
        0b10000 => (pick(Add, Sub), false),
        0b10001 => (pick(Cmtst, Cmeq), false),
        0b10010 if narrow_only => (pick(Mla, Mls), false),
        0b10011 if narrow_only && !u => (Mul, false),
        0b10011 if u && size == 0 => (Pmul, false),
        0b10100 if narrow_only => (pick(Smax, Umax), true),
        0b10101 if narrow_only => (pick(Smin, Umin), true),
        0b10110 if size == 1 || size == 2 => (pick(Sqdmulh, Sqrdmulh), false),
        0b10111 if !u => (Add, true),
        _ => return Err(UNALLOC),
    };
    Ok(if pair {
        SimdOp::IntPair { op, sh, rd, rn, rm }
    } else {
        SimdOp::IntBin { op, sh, rd, rn, rm }
    })
}

/// The element format and lane count of a floating-point vector with the `sz` bit and `Q`.
pub(super) fn fp_vector_shape(sz: bool, q: bool) -> Res<(Fmt, u8)> {
    match (sz, q) {
        (false, false) => Ok((Fmt::S, 2)),
        (false, true) => Ok((Fmt::S, 4)),
        (true, true) => Ok((Fmt::D, 2)),
        (true, false) => Err(UNALLOC),
    }
}

fn fp_three_same(w: u32, q: bool, u: bool, size: u32, opcode: u32) -> Res<SimdOp> {
    let sel = size & 2 != 0;
    let (fmt, lanes) = fp_vector_shape(size & 1 != 0, q)?;
    let (op, pair) = fp_three_same_op(u, opcode, sel)?;
    let (rd, rn, rm) = (rd(w), rn(w), rm(w));
    Ok(if pair {
        SimdOp::FpPair {
            op,
            fmt,
            lanes,
            rd,
            rn,
            rm,
        }
    } else {
        SimdOp::FpBin {
            op,
            fmt,
            lanes,
            rd,
            rn,
            rm,
        }
    })
}

/// The operation of a floating-point three-same encoding (also used by the scalar group) and
/// whether it is pairwise.
fn fp_three_same_op(u: bool, opcode: u32, sel: bool) -> Res<(FpBin, bool)> {
    use FpBin::*;
    Ok(match (u, opcode, sel) {
        (false, 0b11000, false) => (MaxNum, false),
        (false, 0b11000, true) => (MinNum, false),
        (false, 0b11001, false) => (Mla, false),
        (false, 0b11001, true) => (Mls, false),
        (false, 0b11010, false) => (Add, false),
        (false, 0b11010, true) => (Sub, false),
        (false, 0b11011, false) => (Mulx, false),
        (false, 0b11100, false) => (Cmeq, false),
        (false, 0b11110, false) => (Max, false),
        (false, 0b11110, true) => (Min, false),
        (false, 0b11111, false) => (Recps, false),
        (false, 0b11111, true) => (Rsqrts, false),
        (true, 0b11000, false) => (MaxNum, true),
        (true, 0b11000, true) => (MinNum, true),
        (true, 0b11010, false) => (Add, true),
        (true, 0b11010, true) => (Abd, false),
        (true, 0b11011, false) => (Mul, false),
        (true, 0b11100, false) => (Cmge, false),
        (true, 0b11100, true) => (Cmgt, false),
        (true, 0b11101, false) => (Acge, false),
        (true, 0b11101, true) => (Acgt, false),
        (true, 0b11110, false) => (Max, true),
        (true, 0b11110, true) => (Min, true),
        (true, 0b11111, false) => (Div, false),
        _ => return Err(UNALLOC),
    })
}

/// The scalar floating-point three-same operations: the same table minus the arithmetic that has
/// scalar forms elsewhere.
pub(super) fn fp_scalar_three_same_op(u: bool, opcode: u32, sel: bool) -> Res<FpBin> {
    use FpBin::*;
    Ok(match (u, opcode, sel) {
        (false, 0b11011, false) => Mulx,
        (false, 0b11100, false) => Cmeq,
        (false, 0b11111, false) => Recps,
        (false, 0b11111, true) => Rsqrts,
        (true, 0b11010, true) => Abd,
        (true, 0b11100, false) => Cmge,
        (true, 0b11100, true) => Cmgt,
        (true, 0b11101, false) => Acge,
        (true, 0b11101, true) => Acgt,
        _ => return Err(UNALLOC),
    })
}

// ---------------------------------------------------------------------------------------------
// Three registers, different element sizes
// ---------------------------------------------------------------------------------------------

fn three_different(w: u32) -> Res<SimdOp> {
    let (q, u, size, opcode) = (bit(w, 30), bit(w, 29), bits(w, 23, 22), bits(w, 15, 12));
    let (rd, rn, rm) = (rd(w), rn(w), rm(w));
    if opcode == 0b1110 {
        // PMULL: 8-bit polynomials, or 64-bit with the cryptographic extension.
        if u || (size != 0 && size != 3) {
            return Err(UNALLOC);
        }
        let sh = if size == 0 {
            Shape::vector(0, false)
        } else {
            Shape {
                esize: 64,
                lanes: 1,
            }
        };
        return Ok(SimdOp::Long {
            op: Long::Pmull,
            sh,
            upper: q,
            rd,
            rn,
            rm,
        });
    }
    if size == 3 {
        return Err(UNALLOC);
    }
    let sh = Shape::vector(size, false);
    use Long::*;
    let op = match (opcode, u) {
        (0b0000, _) => Addl(!u),
        (0b0001, _) => Addw(!u),
        (0b0010, _) => Subl(!u),
        (0b0011, _) => Subw(!u),
        (0b0100, _) => Addhn(u),
        (0b0101, _) => Abal(!u),
        (0b0110, _) => Subhn(u),
        (0b0111, _) => Abdl(!u),
        (0b1000, _) => Mlal(!u),
        (0b1010, _) => Mlsl(!u),
        (0b1100, _) => Mull(!u),
        (0b1001, false) | (0b1011, false) | (0b1101, false) => {
            if size == 0 {
                return Err(UNALLOC);
            }
            match opcode {
                0b1001 => Sqdmlal,
                0b1011 => Sqdmlsl,
                _ => Sqdmull,
            }
        }
        _ => return Err(UNALLOC),
    };
    Ok(SimdOp::Long {
        op,
        sh,
        upper: q,
        rd,
        rn,
        rm,
    })
}

// ---------------------------------------------------------------------------------------------
// Two registers, miscellaneous
// ---------------------------------------------------------------------------------------------

fn two_reg_misc(w: u32) -> Res<SimdOp> {
    let (q, u, size, opcode) = (bit(w, 30), bit(w, 29), bits(w, 23, 22), bits(w, 16, 12));
    let (rd, rn) = (rd(w), rn(w));
    if opcode >= 0b11000
        || (0b01100..=0b01111).contains(&opcode)
        || (0b10110..=0b10111).contains(&opcode)
    {
        return fp_two_reg_misc(w, q, u, size, opcode);
    }
    use IntUn::*;
    let sh = Shape::vector(size, q);
    let narrow_only = size != 3;
    let un = |op: IntUn| -> Res<SimdOp> {
        if size == 3 && !q {
            return Err(UNALLOC);
        }
        Ok(SimdOp::IntUn { op, sh, rd, rn })
    };
    match (opcode, u) {
        (0b00000, false) if size < 3 => un(Rev64),
        (0b00000, true) if size < 2 => un(Rev32),
        (0b00001, false) if size == 0 => un(Rev16),
        (0b00010, _) | (0b00110, _) if narrow_only => {
            let op = match (opcode, u) {
                (0b00010, false) => PairLong::Saddlp,
                (0b00010, true) => PairLong::Uaddlp,
                (_, false) => PairLong::Sadalp,
                (_, true) => PairLong::Uadalp,
            };
            Ok(SimdOp::PairLong { op, sh, rd, rn })
        }
        (0b00011, false) => un(Suqadd),
        (0b00011, true) => un(Usqadd),
        (0b00100, false) if narrow_only => un(Cls),
        (0b00100, true) if narrow_only => un(Clz),
        (0b00101, false) if size == 0 => un(Cnt),
        (0b00101, true) if size == 0 => un(Not),
        // RBIT reverses the bits of each byte, whichever lane size the `size` field names.
        (0b00101, true) if size == 1 => Ok(SimdOp::IntUn {
            op: Rbit,
            sh: Shape::vector(0, q),
            rd,
            rn,
        }),
        (0b00111, false) => un(Sqabs),
        (0b00111, true) => un(Sqneg),
        (0b01000, false) => un(Cmgt0),
        (0b01000, true) => un(Cmge0),
        (0b01001, false) => un(Cmeq0),
        (0b01001, true) => un(Cmle0),
        (0b01010, false) => un(Cmlt0),
        (0b01011, false) => un(Abs),
        (0b01011, true) => un(Neg),
        (0b10010, _) | (0b10100, _) if narrow_only => {
            let op = match (opcode, u) {
                (0b10010, false) => Narrow::Xtn,
                (0b10010, true) => Narrow::Sqxtun,
                (_, false) => Narrow::Sqxtn,
                (_, true) => Narrow::Uqxtn,
            };
            Ok(SimdOp::Narrow {
                op,
                sh: Shape::vector(size, false),
                upper: q,
                rd,
                rn,
            })
        }
        (0b10011, true) if narrow_only => Ok(SimdOp::Shll {
            sh: Shape::vector(size, false),
            upper: q,
            rd,
            rn,
        }),
        _ => Err(UNALLOC),
    }
}

/// The floating-point members of the two-register miscellaneous group. `size<0>` is `sz`;
/// `size<1>` selects between paired operations.
fn fp_two_reg_misc(w: u32, q: bool, u: bool, size: u32, opcode: u32) -> Res<SimdOp> {
    let (rd, rn) = (rd(w), rn(w));
    let sel = size & 2 != 0;
    let sz = size & 1 != 0;
    use FpUn::*;
    // Operations that produce a value of the same format.
    let same = |op: FpUn| -> Res<SimdOp> {
        let (fmt, lanes) = fp_vector_shape(sz, q)?;
        Ok(SimdOp::FpUn {
            op,
            fmt,
            lanes,
            rd,
            rn,
        })
    };
    let to_int = |mode: Rounding, signed: bool| -> Res<SimdOp> {
        let (fmt, lanes) = fp_vector_shape(sz, q)?;
        Ok(SimdOp::FpToInt {
            mode,
            signed,
            fmt,
            lanes,
            rd,
            rn,
        })
    };
    let round = |mode: Option<Rounding>, exact: bool| same(Round { mode, exact });
    match (opcode, u, sel) {
        (0b01100, false, true) => same(Cmgt0),
        (0b01100, true, true) => same(Cmge0),
        (0b01101, false, true) => same(Cmeq0),
        (0b01101, true, true) => same(Cmle0),
        (0b01110, false, true) => same(Cmlt0),
        (0b01111, false, true) => same(Abs),
        (0b01111, true, true) => same(Neg),
        (0b10110, false, false) => {
            // FCVTN: single to half (sz = 0), double to single (sz = 1).
            Ok(SimdOp::FpNarrow {
                fmt: if sz { Fmt::S } else { Fmt::H },
                odd: false,
                upper: q,
                scalar: false,
                rd,
                rn,
            })
        }
        (0b10110, true, false) => {
            if !sz {
                return Err(UNALLOC);
            }
            Ok(SimdOp::FpNarrow {
                fmt: Fmt::S,
                odd: true,
                upper: q,
                scalar: false,
                rd,
                rn,
            })
        }
        (0b10111, false, false) => Ok(SimdOp::FpWiden {
            fmt: if sz { Fmt::D } else { Fmt::S },
            upper: q,
            rd,
            rn,
        }),
        (0b11000, false, false) => round(Some(Rounding::Nearest), false),
        (0b11000, false, true) => round(Some(Rounding::PlusInf), false),
        (0b11001, false, false) => round(Some(Rounding::MinusInf), false),
        (0b11001, false, true) => round(Some(Rounding::Zero), false),
        (0b11000, true, false) => round(Some(Rounding::NearestAway), false),
        (0b11001, true, false) => round(None, true),
        (0b11001, true, true) => round(None, false),
        (0b11010, _, false) => to_int(Rounding::Nearest, !u),
        (0b11010, _, true) => to_int(Rounding::PlusInf, !u),
        (0b11011, _, false) => to_int(Rounding::MinusInf, !u),
        (0b11011, _, true) => to_int(Rounding::Zero, !u),
        (0b11100, _, false) => to_int(Rounding::NearestAway, !u),
        (0b11101, _, false) => {
            let (fmt, lanes) = fp_vector_shape(sz, q)?;
            Ok(SimdOp::IntToFp {
                signed: !u,
                fmt,
                lanes,
                rd,
                rn,
            })
        }
        (0b11100, false, true) => {
            // URECPE: 32-bit elements only.
            if sz {
                return Err(UNALLOC);
            }
            Ok(SimdOp::IntUn {
                op: IntUn::Urecpe,
                sh: Shape::vector(2, q),
                rd,
                rn,
            })
        }
        (0b11100, true, true) => {
            if sz {
                return Err(UNALLOC);
            }
            Ok(SimdOp::IntUn {
                op: IntUn::Ursqrte,
                sh: Shape::vector(2, q),
                rd,
                rn,
            })
        }
        (0b11101, false, true) => same(Recpe),
        (0b11101, true, true) => same(Rsqrte),
        (0b11111, true, true) => same(Sqrt),
        _ => Err(UNALLOC),
    }
}

// ---------------------------------------------------------------------------------------------
// Across lanes
// ---------------------------------------------------------------------------------------------

fn across_lanes(w: u32) -> Res<SimdOp> {
    let (q, u, size, opcode) = (bit(w, 30), bit(w, 29), bits(w, 23, 22), bits(w, 16, 12));
    let (rd, rn) = (rd(w), rn(w));
    if matches!(opcode, 0b01100 | 0b01111) {
        // FMAXNMV, FMAXV, FMINNMV, FMINV: four singles only.
        if !u || size & 1 != 0 || !q {
            return Err(UNALLOC);
        }
        let min = size & 2 != 0;
        let op = match (opcode, min) {
            (0b01100, false) => FpBin::MaxNum,
            (0b01100, true) => FpBin::MinNum,
            (_, false) => FpBin::Max,
            (_, true) => FpBin::Min,
        };
        return Ok(SimdOp::FpAcross { op, rd, rn });
    }
    if size == 3 || (size == 2 && !q) {
        return Err(UNALLOC);
    }
    let op = match (opcode, u) {
        (0b00011, false) => Across::Saddlv,
        (0b00011, true) => Across::Uaddlv,
        (0b01010, false) => Across::Smaxv,
        (0b01010, true) => Across::Umaxv,
        (0b11010, false) => Across::Sminv,
        (0b11010, true) => Across::Uminv,
        (0b11011, false) => Across::Addv,
        _ => return Err(UNALLOC),
    };
    Ok(SimdOp::Across {
        op,
        sh: Shape::vector(size, q),
        rd,
        rn,
    })
}

// ---------------------------------------------------------------------------------------------
// Permute, extract, table lookup, copy
// ---------------------------------------------------------------------------------------------

fn permute(w: u32) -> Res<SimdOp> {
    let (q, size) = (bit(w, 30), bits(w, 23, 22));
    if size == 3 && !q {
        return Err(UNALLOC);
    }
    let op = match bits(w, 14, 12) {
        0b001 => Permute::Uzp1,
        0b010 => Permute::Trn1,
        0b011 => Permute::Zip1,
        0b101 => Permute::Uzp2,
        0b110 => Permute::Trn2,
        0b111 => Permute::Zip2,
        _ => return Err(UNALLOC),
    };
    Ok(SimdOp::Permute {
        op,
        sh: Shape::vector(size, q),
        rd: rd(w),
        rn: rn(w),
        rm: rm(w),
    })
}

fn extract(w: u32) -> Res<SimdOp> {
    let q = bit(w, 30);
    let imm4 = bits(w, 14, 11);
    if bits(w, 23, 22) != 0 || (!q && imm4 >= 8) {
        return Err(UNALLOC);
    }
    Ok(SimdOp::Ext {
        q,
        imm4: imm4 as u8,
        rd: rd(w),
        rn: rn(w),
        rm: rm(w),
    })
}

fn table_lookup(w: u32) -> Res<SimdOp> {
    if bits(w, 23, 22) != 0 {
        return Err(UNALLOC);
    }
    Ok(SimdOp::Table {
        q: bit(w, 30),
        regs: bits(w, 14, 13) as u8 + 1,
        extend: bit(w, 12),
        rd: rd(w),
        rn: rn(w),
        rm: rm(w),
    })
}

fn copy(w: u32) -> Res<SimdOp> {
    let (q, op) = (bit(w, 30), bit(w, 29));
    let imm5 = bits(w, 20, 16);
    let imm4 = bits(w, 14, 11);
    let (rd, rn) = (rd(w), rn(w));
    if imm5 & 0xF == 0 {
        // No element size: imm5 = 00000 and 10000 are reserved.
        return Err(UNALLOC);
    }
    let size = imm5.trailing_zeros();
    let esize = 8u8 << size;
    let index = (imm5 >> (size + 1)) as u8;
    if op {
        // INS (element): only with Q = 1; the source index comes from imm4.
        if !q {
            return Err(UNALLOC);
        }
        let src = (imm4 >> size) as u8;
        return Ok(SimdOp::InsElem {
            esize,
            dst: index,
            src,
            rd,
            rn,
        });
    }
    match imm4 {
        0b0000 => {
            // DUP (element)
            if size == 3 && !q {
                return Err(UNALLOC);
            }
            Ok(SimdOp::DupElem {
                sh: Shape::vector(size, q),
                idx: index,
                rd,
                rn,
            })
        }
        0b0001 => {
            if size == 3 && !q {
                return Err(UNALLOC);
            }
            Ok(SimdOp::DupGeneral {
                sh: Shape::vector(size, q),
                rd,
                rn,
            })
        }
        0b0011 => {
            // INS (general): Q must be 1.
            if !q {
                return Err(UNALLOC);
            }
            Ok(SimdOp::InsGeneral {
                esize,
                idx: index,
                rd,
                rn,
            })
        }
        0b0101 => {
            // SMOV: the source element must be narrower than the destination.
            if size > 2 || (size == 2 && !q) {
                return Err(UNALLOC);
            }
            Ok(SimdOp::MovToGeneral {
                signed: true,
                esize,
                idx: index,
                wide: q,
                rd,
                rn,
            })
        }
        0b0111 => {
            // UMOV: a 64-bit element needs Q, narrower ones must not have it.
            if (size < 3 && q) || (size == 3 && !q) {
                return Err(UNALLOC);
            }
            Ok(SimdOp::MovToGeneral {
                signed: false,
                esize,
                idx: index,
                wide: q,
                rd,
                rn,
            })
        }
        _ => Err(UNALLOC),
    }
}

// ---------------------------------------------------------------------------------------------
// Modified immediate
// ---------------------------------------------------------------------------------------------

fn modified_immediate(w: u32) -> Res<SimdOp> {
    let (q, op) = (bit(w, 30), bit(w, 29));
    let cmode = bits(w, 15, 12);
    let o2 = bit(w, 11);
    let imm8 = (bits(w, 18, 16) << 5 | bits(w, 9, 5)) as u64;
    let rd = rd(w);
    if o2 {
        // Half-precision FMOV, or reserved.
        return Err(UNALLOC);
    }
    let rep32 = |v: u64| v | (v << 32);
    let rep16 = |v: u64| {
        let v = v | (v << 16);
        v | (v << 32)
    };
    let (kind, imm) = match cmode >> 1 {
        0b000 => (ModImm::Set, rep32(imm8)),
        0b001 => (ModImm::Set, rep32(imm8 << 8)),
        0b010 => (ModImm::Set, rep32(imm8 << 16)),
        0b011 => (ModImm::Set, rep32(imm8 << 24)),
        0b100 => (ModImm::Set, rep16(imm8)),
        0b101 => (ModImm::Set, rep16(imm8 << 8)),
        0b110 => {
            if cmode & 1 == 0 {
                (ModImm::Set, rep32((imm8 << 8) | 0xFF))
            } else {
                (ModImm::Set, rep32((imm8 << 16) | 0xFFFF))
            }
        }
        _ => {
            // 0b111
            if cmode & 1 == 0 && !op {
                (ModImm::Set, imm8 * 0x0101_0101_0101_0101)
            } else if cmode & 1 == 0 {
                let mut v = 0u64;
                for i in 0..8 {
                    if imm8 >> i & 1 != 0 {
                        v |= 0xFFu64 << (8 * i);
                    }
                }
                (ModImm::Set, v)
            } else if !op {
                // FMOV (single)
                let sign = imm8 >> 7 & 1;
                let b6 = imm8 >> 6 & 1;
                let v = (sign << 31)
                    | ((1 - b6) << 30)
                    | ((if b6 == 1 { 0x1F } else { 0 }) << 25)
                    | ((imm8 & 0x3F) << 19);
                return Ok(SimdOp::ModImm {
                    op: ModImm::Set,
                    q,
                    imm: rep32(v),
                    rd,
                });
            } else {
                // FMOV (double), 2D only
                if !q {
                    return Err(UNALLOC);
                }
                let sign = imm8 >> 7 & 1;
                let b6 = imm8 >> 6 & 1;
                let v = (sign << 63)
                    | ((1 - b6) << 62)
                    | ((if b6 == 1 { 0xFF } else { 0 }) << 54)
                    | ((imm8 & 0x3F) << 48);
                return Ok(SimdOp::ModImm {
                    op: ModImm::Set,
                    q,
                    imm: v,
                    rd,
                });
            }
        }
    };
    // `op` selects the inverting forms for the shifted-immediate patterns, and BIC/ORR for the
    // patterns with cmode<0> set.
    match cmode >> 1 {
        0b000..=0b101 if cmode & 1 == 1 => Ok(SimdOp::ModImm {
            op: if op { ModImm::Bic } else { ModImm::Orr },
            q,
            imm,
            rd,
        }),
        0b000..=0b110 => Ok(SimdOp::ModImm {
            op: kind,
            q,
            imm: if op { !imm } else { imm },
            rd,
        }),
        _ => Ok(SimdOp::ModImm {
            op: kind,
            q,
            imm,
            rd,
        }),
    }
}

/// `MinMax` for the reductions.
pub(super) fn min_max_of(op: FpBin) -> MinMax {
    match op {
        FpBin::Max => MinMax::Max,
        FpBin::Min => MinMax::Min,
        FpBin::MaxNum => MinMax::MaxNum,
        _ => MinMax::MinNum,
    }
}
