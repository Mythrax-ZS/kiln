//! Advanced SIMD groups with immediates and element indexes, and the scalar groups.

use super::decode::{bit, bits, Res, UNALLOC};
use super::decode_simd::{fp_scalar_three_same_op, fp_vector_shape, rd, rm, rn};
use super::simd::*;
use super::Reject;
use crate::a64::fp::{Fmt, Rounding};

const CRYPTO: Reject = Reject::Unsupported("cryptographic extension");

// ---------------------------------------------------------------------------------------------
// Vector: shift by immediate
// ---------------------------------------------------------------------------------------------

/// The element size code (log2 of the size in bytes) of the highest set bit of `immh`.
fn immh_size(immh: u32) -> u32 {
    31 - immh.leading_zeros()
}

pub fn vector_shift(w: u32) -> Res<SimdOp> {
    let (q, u) = (bit(w, 30), bit(w, 29));
    let (immh, immb, opcode) = (bits(w, 22, 19), bits(w, 18, 16), bits(w, 15, 11));
    let (rd, rn) = (rd(w), rn(w));
    let immhb = (immh << 3) | immb;
    let size = immh_size(immh);
    let esize = 8u32 << size;
    let right = 2 * esize - immhb;
    let left = immhb - esize;
    // Operations that keep the element size cannot use a 64-bit element in a 64-bit register.
    let same_size = |op: Shift, shift: u32| -> Res<SimdOp> {
        if size == 3 && !q {
            return Err(UNALLOC);
        }
        Ok(SimdOp::Shift {
            op,
            sh: Shape::vector(size, q),
            shift: shift as u8,
            rd,
            rn,
        })
    };
    use Shift::*;
    match (opcode, u) {
        (0b00000, false) => same_size(Sshr, right),
        (0b00000, true) => same_size(Ushr, right),
        (0b00010, false) => same_size(Ssra, right),
        (0b00010, true) => same_size(Usra, right),
        (0b00100, false) => same_size(Srshr, right),
        (0b00100, true) => same_size(Urshr, right),
        (0b00110, false) => same_size(Srsra, right),
        (0b00110, true) => same_size(Ursra, right),
        (0b01000, true) => same_size(Sri, right),
        (0b01010, false) => same_size(Shl, left),
        (0b01010, true) => same_size(Sli, left),
        (0b01100, true) => same_size(Sqshlu, left),
        (0b01110, false) => same_size(Sqshl, left),
        (0b01110, true) => same_size(Uqshl, left),
        (0b10000..=0b10011, _) => {
            // Narrowing: the destination element is one of 8, 16 or 32 bits.
            if size > 2 {
                return Err(UNALLOC);
            }
            let op = match (opcode, u) {
                (0b10000, false) => ShiftNarrow::Shrn,
                (0b10001, false) => ShiftNarrow::Rshrn,
                (0b10010, false) => ShiftNarrow::Sqshrn,
                (0b10010, true) => ShiftNarrow::Uqshrn,
                (0b10011, false) => ShiftNarrow::Sqrshrn,
                (0b10011, true) => ShiftNarrow::Uqrshrn,
                (0b10000, true) => ShiftNarrow::Sqshrun,
                _ => ShiftNarrow::Sqrshrun,
            };
            Ok(SimdOp::ShiftNarrow {
                op,
                sh: Shape::vector(size, false),
                upper: q,
                shift: right as u8,
                rd,
                rn,
            })
        }
        (0b10100, _) => {
            if size > 2 {
                return Err(UNALLOC);
            }
            Ok(SimdOp::ShiftLong {
                signed: !u,
                sh: Shape::vector(size, false),
                upper: q,
                shift: left as u8,
                rd,
                rn,
            })
        }
        (0b11100, _) | (0b11111, _) => {
            // Fixed-point conversions: single (immh = 01xx) or double (1xxx) only.
            if size < 2 || (size == 3 && !q) {
                return Err(UNALLOC);
            }
            let fmt = if size == 3 { Fmt::D } else { Fmt::S };
            let lanes = (if q { 128 } else { 64 } / esize) as u8;
            Ok(SimdOp::FpFixed {
                to_int: opcode == 0b11111,
                signed: !u,
                fmt,
                lanes,
                fbits: right as u8,
                rd,
                rn,
            })
        }
        _ => Err(UNALLOC),
    }
}

// ---------------------------------------------------------------------------------------------
// Vector: x indexed element
// ---------------------------------------------------------------------------------------------

/// The element index and the register number for an indexed operand of the given element size.
fn element_index(w: u32, size: u32) -> (u8, u8) {
    let (h, l, m) = (bits(w, 11, 11), bits(w, 21, 21), bits(w, 20, 20));
    let reg = bits(w, 19, 16);
    if size == 1 {
        (((h << 2) | (l << 1) | m) as u8, reg as u8)
    } else {
        (((h << 1) | l) as u8, ((m << 4) | reg) as u8)
    }
}

pub fn vector_indexed(w: u32) -> Res<SimdOp> {
    let (q, u, size, opcode) = (bit(w, 30), bit(w, 29), bits(w, 23, 22), bits(w, 15, 12));
    let (rd, rn) = (rd(w), rn(w));
    if matches!(opcode, 0b0001 | 0b0101 | 0b1001) {
        return fp_indexed(w, q, u, size, opcode, false);
    }
    if size != 1 && size != 2 {
        return Err(UNALLOC);
    }
    let (idx, rm) = element_index(w, size);
    use IntBin::*;
    let plain = |op: IntBin| -> Res<SimdOp> {
        Ok(SimdOp::IntElem {
            op,
            sh: Shape::vector(size, q),
            rd,
            rn,
            rm,
            idx,
        })
    };
    let long = |op: Long| -> Res<SimdOp> {
        Ok(SimdOp::LongElem {
            op,
            sh: Shape::vector(size, false),
            upper: q,
            rd,
            rn,
            rm,
            idx,
        })
    };
    match (opcode, u) {
        (0b0000, true) => plain(Mla),
        (0b0100, true) => plain(Mls),
        (0b1000, false) => plain(Mul),
        (0b1100, false) => plain(Sqdmulh),
        (0b1101, false) => plain(Sqrdmulh),
        (0b0010, _) => long(Long::Mlal(!u)),
        (0b0110, _) => long(Long::Mlsl(!u)),
        (0b1010, _) => long(Long::Mull(!u)),
        (0b0011, false) => long(Long::Sqdmlal),
        (0b0111, false) => long(Long::Sqdmlsl),
        (0b1011, false) => long(Long::Sqdmull),
        _ => Err(UNALLOC),
    }
}

/// `FMLA`, `FMLS`, `FMUL` and `FMULX` by element; `scalar` for the scalar forms.
fn fp_indexed(w: u32, q: bool, u: bool, size: u32, opcode: u32, scalar: bool) -> Res<SimdOp> {
    // size = 10 selects single precision, 11 double; 0x would be half precision.
    if size < 2 {
        return Err(UNALLOC);
    }
    let sz = size & 1 != 0;
    let (h, l, m) = (bits(w, 11, 11), bits(w, 21, 21), bits(w, 20, 20));
    let (fmt, lanes, idx, rm) = if sz {
        if l != 0 {
            return Err(UNALLOC);
        }
        let lanes = if scalar {
            1
        } else {
            fp_vector_shape(true, q)?.1
        };
        (Fmt::D, lanes, h as u8, ((m << 4) | bits(w, 19, 16)) as u8)
    } else {
        let lanes = if scalar {
            1
        } else {
            fp_vector_shape(false, q)?.1
        };
        (
            Fmt::S,
            lanes,
            ((h << 1) | l) as u8,
            ((m << 4) | bits(w, 19, 16)) as u8,
        )
    };
    let op = match (opcode, u) {
        (0b0001, false) => FpBin::Mla,
        (0b0101, false) => FpBin::Mls,
        (0b1001, false) => FpBin::Mul,
        (0b1001, true) => FpBin::Mulx,
        _ => return Err(UNALLOC),
    };
    Ok(SimdOp::FpElem {
        op,
        fmt,
        lanes,
        rd: rd(w),
        rn: rn(w),
        rm,
        idx,
    })
}

// ---------------------------------------------------------------------------------------------
// Scalar groups
// ---------------------------------------------------------------------------------------------

pub fn scalar(w: u32) -> Res<SimdOp> {
    if !bit(w, 30) {
        return Err(UNALLOC);
    }
    if bit(w, 24) {
        if !bit(w, 10) {
            return scalar_indexed(w);
        }
        if bit(w, 23) {
            return Err(UNALLOC);
        }
        return scalar_shift(w);
    }
    if bit(w, 21) {
        if bit(w, 10) {
            return scalar_three_same(w);
        }
        if !bit(w, 11) {
            return scalar_three_different(w);
        }
        return match bits(w, 20, 17) {
            0b0000 => scalar_two_reg_misc(w),
            0b1000 => scalar_pairwise(w),
            0b0100 => crypto_sha_two_reg(w),
            _ => Err(UNALLOC),
        };
    }
    if bit(w, 15) {
        return Err(UNALLOC); // scalar three-register extension (RDM)
    }
    if bit(w, 10) {
        return scalar_copy(w);
    }
    if !bit(w, 11) {
        return crypto_sha_three_reg(w);
    }
    Err(UNALLOC)
}

fn crypto_sha_three_reg(w: u32) -> Res<SimdOp> {
    // 0101 1110 size 0 Rm 0 opcode 00 Rn Rd
    if bits(w, 31, 24) == 0x5E && bits(w, 23, 22) == 0 && bits(w, 14, 12) != 0b111 {
        Err(CRYPTO)
    } else {
        Err(UNALLOC)
    }
}

fn crypto_sha_two_reg(w: u32) -> Res<SimdOp> {
    if bits(w, 31, 24) == 0x5E && bits(w, 23, 22) == 0 && bits(w, 16, 12) <= 0b00010 {
        Err(CRYPTO)
    } else {
        Err(UNALLOC)
    }
}

fn scalar_copy(w: u32) -> Res<SimdOp> {
    // 01 op 11110000 imm5 0 imm4 1 Rn Rd: only DUP (element).
    if bit(w, 29) || bits(w, 23, 22) != 0 || bits(w, 14, 11) != 0 {
        return Err(UNALLOC);
    }
    let imm5 = bits(w, 20, 16);
    if imm5 & 0xF == 0 {
        return Err(UNALLOC);
    }
    let size = imm5.trailing_zeros();
    Ok(SimdOp::DupElem {
        sh: Shape::scalar(size),
        idx: (imm5 >> (size + 1)) as u8,
        rd: rd(w),
        rn: rn(w),
    })
}

fn scalar_three_same(w: u32) -> Res<SimdOp> {
    let (u, size, opcode) = (bit(w, 29), bits(w, 23, 22), bits(w, 15, 11));
    let (rd, rn, rm) = (rd(w), rn(w), rm(w));
    if opcode >= 0b11000 {
        let op = fp_scalar_three_same_op(u, opcode, size & 2 != 0)?;
        let fmt = if size & 1 != 0 { Fmt::D } else { Fmt::S };
        return Ok(SimdOp::FpBin {
            op,
            fmt,
            lanes: 1,
            rd,
            rn,
            rm,
        });
    }
    let d = size == 3;
    use IntBin::*;
    let op = match (opcode, u) {
        (0b00001, false) => Sqadd,
        (0b00001, true) => Uqadd,
        (0b00101, false) => Sqsub,
        (0b00101, true) => Uqsub,
        (0b00110, false) if d => Cmgt,
        (0b00110, true) if d => Cmhi,
        (0b00111, false) if d => Cmge,
        (0b00111, true) if d => Cmhs,
        (0b01000, false) if d => Sshl,
        (0b01000, true) if d => Ushl,
        (0b01001, false) => Sqshl,
        (0b01001, true) => Uqshl,
        (0b01010, false) if d => Srshl,
        (0b01010, true) if d => Urshl,
        (0b01011, false) => Sqrshl,
        (0b01011, true) => Uqrshl,
        (0b10000, false) if d => Add,
        (0b10000, true) if d => Sub,
        (0b10001, false) if d => Cmtst,
        (0b10001, true) if d => Cmeq,
        (0b10110, false) if size == 1 || size == 2 => Sqdmulh,
        (0b10110, true) if size == 1 || size == 2 => Sqrdmulh,
        _ => return Err(UNALLOC),
    };
    Ok(SimdOp::IntBin {
        op,
        sh: Shape::scalar(size),
        rd,
        rn,
        rm,
    })
}

fn scalar_three_different(w: u32) -> Res<SimdOp> {
    let (u, size, opcode) = (bit(w, 29), bits(w, 23, 22), bits(w, 15, 12));
    if u || (size != 1 && size != 2) {
        return Err(UNALLOC);
    }
    let op = match opcode {
        0b1001 => Long::Sqdmlal,
        0b1011 => Long::Sqdmlsl,
        0b1101 => Long::Sqdmull,
        _ => return Err(UNALLOC),
    };
    Ok(SimdOp::Long {
        op,
        sh: Shape::scalar(size),
        upper: false,
        rd: rd(w),
        rn: rn(w),
        rm: rm(w),
    })
}

fn scalar_two_reg_misc(w: u32) -> Res<SimdOp> {
    let (u, size, opcode) = (bit(w, 29), bits(w, 23, 22), bits(w, 16, 12));
    let (rd, rn) = (rd(w), rn(w));
    let sh = Shape::scalar(size);
    let d = size == 3;
    let un = |op: IntUn| -> Res<SimdOp> { Ok(SimdOp::IntUn { op, sh, rd, rn }) };
    use IntUn::*;
    match (opcode, u) {
        (0b00011, false) => return un(Suqadd),
        (0b00011, true) => return un(Usqadd),
        (0b00111, false) => return un(Sqabs),
        (0b00111, true) => return un(Sqneg),
        (0b01000, false) if d => return un(Cmgt0),
        (0b01000, true) if d => return un(Cmge0),
        (0b01001, false) if d => return un(Cmeq0),
        (0b01001, true) if d => return un(Cmle0),
        (0b01010, false) if d => return un(Cmlt0),
        (0b01011, false) if d => return un(Abs),
        (0b01011, true) if d => return un(Neg),
        (0b10010, true) | (0b10100, _) if !d => {
            let op = match (opcode, u) {
                (0b10010, _) => Narrow::Sqxtun,
                (_, false) => Narrow::Sqxtn,
                (_, true) => Narrow::Uqxtn,
            };
            return Ok(SimdOp::Narrow {
                op,
                sh,
                upper: false,
                rd,
                rn,
            });
        }
        _ => {}
    }
    // Floating point.
    let sel = size & 2 != 0;
    let fmt = if size & 1 != 0 { Fmt::D } else { Fmt::S };
    let same = |op: FpUn| -> Res<SimdOp> {
        Ok(SimdOp::FpUn {
            op,
            fmt,
            lanes: 1,
            rd,
            rn,
        })
    };
    let to_int = |mode: Rounding| -> Res<SimdOp> {
        Ok(SimdOp::FpToInt {
            mode,
            signed: !u,
            fmt,
            lanes: 1,
            rd,
            rn,
        })
    };
    match (opcode, u, sel) {
        (0b01100, false, true) => same(FpUn::Cmgt0),
        (0b01100, true, true) => same(FpUn::Cmge0),
        (0b01101, false, true) => same(FpUn::Cmeq0),
        (0b01101, true, true) => same(FpUn::Cmle0),
        (0b01110, false, true) => same(FpUn::Cmlt0),
        (0b10110, true, false) if fmt == Fmt::D => Ok(SimdOp::FpNarrow {
            fmt: Fmt::S,
            odd: true,
            upper: false,
            scalar: true,
            rd,
            rn,
        }),
        (0b11010, _, false) => to_int(Rounding::Nearest),
        (0b11010, _, true) => to_int(Rounding::PlusInf),
        (0b11011, _, false) => to_int(Rounding::MinusInf),
        (0b11011, _, true) => to_int(Rounding::Zero),
        (0b11100, _, false) => to_int(Rounding::NearestAway),
        (0b11101, _, false) => Ok(SimdOp::IntToFp {
            signed: !u,
            fmt,
            lanes: 1,
            rd,
            rn,
        }),
        (0b11101, false, true) => same(FpUn::Recpe),
        (0b11101, true, true) => same(FpUn::Rsqrte),
        (0b11111, false, true) => same(FpUn::Recpx),
        _ => Err(UNALLOC),
    }
}

fn scalar_pairwise(w: u32) -> Res<SimdOp> {
    let (u, size, opcode) = (bit(w, 29), bits(w, 23, 22), bits(w, 16, 12));
    let (rd, rn) = (rd(w), rn(w));
    if opcode == 0b11011 {
        return if !u && size == 3 {
            Ok(SimdOp::AddpScalar { rd, rn })
        } else {
            Err(UNALLOC)
        };
    }
    if !u {
        return Err(UNALLOC);
    }
    let sel = size & 2 != 0;
    let fmt = if size & 1 != 0 { Fmt::D } else { Fmt::S };
    let op = match (opcode, sel) {
        (0b01100, false) => FpBin::MaxNum,
        (0b01100, true) => FpBin::MinNum,
        (0b01101, false) => FpBin::Add,
        (0b01111, false) => FpBin::Max,
        (0b01111, true) => FpBin::Min,
        _ => return Err(UNALLOC),
    };
    Ok(SimdOp::FpPairScalar { op, fmt, rd, rn })
}

fn scalar_shift(w: u32) -> Res<SimdOp> {
    let u = bit(w, 29);
    let (immh, immb, opcode) = (bits(w, 22, 19), bits(w, 18, 16), bits(w, 15, 11));
    let (rd, rn) = (rd(w), rn(w));
    if immh == 0 {
        return Err(UNALLOC);
    }
    let immhb = (immh << 3) | immb;
    let size = immh_size(immh);
    let esize = 8u32 << size;
    let right = 2 * esize - immhb;
    let left = immhb - esize;
    let d = size == 3;
    let sh = Shape::scalar(size);
    let shift = |op: Shift, amount: u32| -> Res<SimdOp> {
        Ok(SimdOp::Shift {
            op,
            sh,
            shift: amount as u8,
            rd,
            rn,
        })
    };
    use Shift::*;
    match (opcode, u) {
        (0b00000, false) if d => shift(Sshr, right),
        (0b00000, true) if d => shift(Ushr, right),
        (0b00010, false) if d => shift(Ssra, right),
        (0b00010, true) if d => shift(Usra, right),
        (0b00100, false) if d => shift(Srshr, right),
        (0b00100, true) if d => shift(Urshr, right),
        (0b00110, false) if d => shift(Srsra, right),
        (0b00110, true) if d => shift(Ursra, right),
        (0b01000, true) if d => shift(Sri, right),
        (0b01010, false) if d => shift(Shl, left),
        (0b01010, true) if d => shift(Sli, left),
        (0b01100, true) => shift(Sqshlu, left),
        (0b01110, false) => shift(Sqshl, left),
        (0b01110, true) => shift(Uqshl, left),
        (0b10000..=0b10011, _) => {
            // Narrowing to B, H or S from H, S or D; only the saturating forms have scalar versions.
            if d {
                return Err(UNALLOC);
            }
            let op = match (opcode, u) {
                (0b10000, true) => ShiftNarrow::Sqshrun,
                (0b10001, true) => ShiftNarrow::Sqrshrun,
                (0b10010, false) => ShiftNarrow::Sqshrn,
                (0b10010, true) => ShiftNarrow::Uqshrn,
                (0b10011, false) => ShiftNarrow::Sqrshrn,
                (0b10011, true) => ShiftNarrow::Uqrshrn,
                _ => return Err(UNALLOC),
            };
            Ok(SimdOp::ShiftNarrow {
                op,
                sh,
                upper: false,
                shift: right as u8,
                rd,
                rn,
            })
        }
        (0b11100, _) | (0b11111, _) => {
            if size < 2 {
                return Err(UNALLOC);
            }
            Ok(SimdOp::FpFixed {
                to_int: opcode == 0b11111,
                signed: !u,
                fmt: if d { Fmt::D } else { Fmt::S },
                lanes: 1,
                fbits: right as u8,
                rd,
                rn,
            })
        }
        _ => Err(UNALLOC),
    }
}

fn scalar_indexed(w: u32) -> Res<SimdOp> {
    let (u, size, opcode) = (bit(w, 29), bits(w, 23, 22), bits(w, 15, 12));
    let (rd, rn) = (rd(w), rn(w));
    if matches!(opcode, 0b0001 | 0b0101 | 0b1001) {
        return fp_indexed(w, true, u, size, opcode, true);
    }
    if u || (size != 1 && size != 2) {
        return Err(UNALLOC);
    }
    let (idx, rm) = element_index(w, size);
    match opcode {
        0b0011 | 0b0111 | 0b1011 => Ok(SimdOp::LongElem {
            op: match opcode {
                0b0011 => Long::Sqdmlal,
                0b0111 => Long::Sqdmlsl,
                _ => Long::Sqdmull,
            },
            sh: Shape::scalar(size),
            upper: false,
            rd,
            rn,
            rm,
            idx,
        }),
        0b1100 | 0b1101 => Ok(SimdOp::IntElem {
            op: if opcode == 0b1100 {
                IntBin::Sqdmulh
            } else {
                IntBin::Sqrdmulh
            },
            sh: Shape::scalar(size),
            rd,
            rn,
            rm,
            idx,
        }),
        _ => Err(UNALLOC),
    }
}
