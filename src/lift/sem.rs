//! Meaning of the pure IR operations on constants.
//!
//! The builder folds constants with these functions and the evaluator computes every value with
//! them, so a fold can never disagree with execution. Values are carried in a `u128` and masked to
//! the operand type after every operation.

use super::ir::{BinOp, CastOp, Pred, Ty, UnOp};

fn sext(v: u128, bits: u32) -> i128 {
    if bits >= 128 {
        v as i128
    } else {
        let shift = 128 - bits;
        ((v << shift) as i128) >> shift
    }
}

/// `None` when the operation is ill-formed (a shift amount that is not below the width).
pub fn bin(op: BinOp, ty: Ty, a: u128, b: u128) -> Option<u128> {
    let bits = ty.bits();
    let m = ty.mask();
    let (a, b) = (a & m, b & m);
    let r = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => {
            if b >= u128::from(bits) {
                return None;
            }
            a << b
        }
        BinOp::LShr => {
            if b >= u128::from(bits) {
                return None;
            }
            a >> b
        }
        BinOp::AShr => {
            if b >= u128::from(bits) {
                return None;
            }
            (sext(a, bits) >> b) as u128
        }
        BinOp::UDiv => a.checked_div(b).unwrap_or(0),
        BinOp::SDiv => {
            let (sa, sb) = (sext(a, bits), sext(b, bits));
            if sb == 0 {
                0
            } else {
                sa.wrapping_div(sb) as u128
            }
        }
    };
    Some(r & m)
}

pub fn un(op: UnOp, ty: Ty, a: u128) -> u128 {
    let bits = ty.bits();
    let a = a & ty.mask();
    match op {
        UnOp::Clz => u128::from(a.leading_zeros() - (128 - bits)),
        UnOp::Bswap => a.swap_bytes() >> (128 - bits),
        UnOp::Rbit => a.reverse_bits() >> (128 - bits),
    }
}

pub fn cast(op: CastOp, from: Ty, to: Ty, a: u128) -> u128 {
    let a = a & from.mask();
    let r = match op {
        CastOp::Zext | CastOp::Trunc => a,
        CastOp::Sext => sext(a, from.bits()) as u128,
    };
    r & to.mask()
}

pub fn icmp(p: Pred, ty: Ty, a: u128, b: u128) -> bool {
    let bits = ty.bits();
    let m = ty.mask();
    let (a, b) = (a & m, b & m);
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    match p {
        Pred::Eq => a == b,
        Pred::Ne => a != b,
        Pred::Ult => a < b,
        Pred::Ule => a <= b,
        Pred::Ugt => a > b,
        Pred::Uge => a >= b,
        Pred::Slt => sa < sb,
        Pred::Sle => sa <= sb,
        Pred::Sgt => sa > sb,
        Pred::Sge => sa >= sb,
    }
}

/// `(hi:lo) >> (amount mod width)` truncated to the width.
pub fn fshr(ty: Ty, hi: u128, lo: u128, amount: u128) -> u128 {
    let bits = ty.bits();
    let m = ty.mask();
    let (hi, lo) = (hi & m, lo & m);
    let s = (amount % u128::from(bits)) as u32;
    if s == 0 {
        lo
    } else {
        ((lo >> s) | (hi << (bits - s))) & m
    }
}

pub fn mulhi(signed: bool, a: u64, b: u64) -> u64 {
    if signed {
        (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64
    } else {
        ((u128::from(a) * u128::from(b)) >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_follows_a64() {
        assert_eq!(bin(BinOp::UDiv, Ty::I32, 7, 0), Some(0));
        assert_eq!(bin(BinOp::SDiv, Ty::I32, 7, 0), Some(0));
        // MIN / -1 wraps to MIN.
        assert_eq!(
            bin(BinOp::SDiv, Ty::I32, 0x8000_0000, 0xFFFF_FFFF),
            Some(0x8000_0000)
        );
        assert_eq!(
            bin(BinOp::SDiv, Ty::I64, 1 << 63, u128::from(u64::MAX)),
            Some(1 << 63)
        );
        assert_eq!(bin(BinOp::SDiv, Ty::I32, 0xFFFF_FFF9, 2), Some(0xFFFF_FFFD));
    }

    #[test]
    fn shifts_require_in_range_amounts() {
        assert_eq!(bin(BinOp::Shl, Ty::I32, 1, 32), None);
        assert_eq!(bin(BinOp::Shl, Ty::I32, 1, 31), Some(0x8000_0000));
        assert_eq!(
            bin(BinOp::AShr, Ty::I32, 0x8000_0000, 31),
            Some(0xFFFF_FFFF)
        );
        assert_eq!(bin(BinOp::LShr, Ty::I8, 0x80, 7), Some(1));
        assert_eq!(bin(BinOp::Shl, Ty::I128, 1, 127), Some(1 << 127));
    }

    #[test]
    fn unary_and_funnel() {
        assert_eq!(un(UnOp::Clz, Ty::I32, 0), 32);
        assert_eq!(un(UnOp::Clz, Ty::I64, 1), 63);
        assert_eq!(un(UnOp::Bswap, Ty::I32, 0x1122_3344), 0x4433_2211);
        assert_eq!(un(UnOp::Rbit, Ty::I8, 0b0000_0110), 0b0110_0000);
        assert_eq!(fshr(Ty::I32, 0x1, 0x8000_0000, 31), 0x3);
        assert_eq!(fshr(Ty::I32, 0xAAAA_AAAA, 0x5555_5555, 0), 0x5555_5555);
        assert_eq!(fshr(Ty::I32, 0xAAAA_AAAA, 0x5555_5555, 32), 0x5555_5555);
    }

    #[test]
    fn casts_and_compares() {
        assert_eq!(cast(CastOp::Sext, Ty::I8, Ty::I32, 0x80), 0xFFFF_FF80);
        assert_eq!(cast(CastOp::Zext, Ty::I8, Ty::I32, 0x80), 0x80);
        assert_eq!(cast(CastOp::Trunc, Ty::I64, Ty::I8, 0x1FF), 0xFF);
        assert!(icmp(Pred::Slt, Ty::I8, 0x80, 0x7F));
        assert!(!icmp(Pred::Ult, Ty::I8, 0x80, 0x7F));
        assert_eq!(mulhi(true, u64::MAX, 2), u64::MAX);
        assert_eq!(mulhi(false, u64::MAX, 2), 1);
    }
}
