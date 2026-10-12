//! Helpers named after the Arm A-profile pseudocode they implement.

use super::inst::{ExtendType, ShiftType};

/// `Ones(width)` as an integer.
pub fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// `SignExtend(value<bits-1:0>, 64)`.
pub fn sign_extend(value: u64, bits: u32) -> u64 {
    if bits >= 64 {
        value
    } else {
        let shift = 64 - bits;
        (((value << shift) as i64) >> shift) as u64
    }
}

/// `ROR(value, amount)` within `width` bits.
pub fn ror(value: u64, amount: u32, width: u32) -> u64 {
    let amount = amount % width;
    let v = value & mask(width);
    if amount == 0 {
        v
    } else {
        ((v >> amount) | (v << (width - amount))) & mask(width)
    }
}

/// `Replicate(value<esize-1:0>)` to `total` bits.
pub fn replicate(value: u64, esize: u32, total: u32) -> u64 {
    let mut out = 0u64;
    let mut at = 0;
    while at < total {
        out |= (value & mask(esize)) << at;
        at += esize;
    }
    out & mask(total)
}

/// `DecodeBitMasks`: the `(wmask, tmask)` pair, or `None` where the pseudocode says UNDEFINED.
pub fn decode_bit_masks(
    immn: bool,
    imms: u8,
    immr: u8,
    immediate: bool,
    m: u32,
) -> Option<(u64, u64)> {
    let combined = (u32::from(immn) << 6) | (!u32::from(imms) & 0x3F);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 || m < (1 << len) {
        return None;
    }
    let levels = (1u32 << len) - 1;
    if immediate && (u32::from(imms) & levels) == levels {
        return None;
    }
    let s = u32::from(imms) & levels;
    let r = u32::from(immr) & levels;
    let d = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let welem = mask(s + 1);
    let telem = mask(d + 1);
    let wmask = replicate(ror(welem, r, esize), esize, m);
    let tmask = replicate(telem, esize, m);
    Some((wmask, tmask))
}

/// `AddWithCarry(x, y, carry_in)` over `width` bits. The flags come back as `NZCV` in bits 3..0.
pub fn add_with_carry(x: u64, y: u64, carry_in: bool, width: u32) -> (u64, u8) {
    let m = mask(width);
    let (x, y) = (x & m, y & m);
    let unsigned_sum = u128::from(x) + u128::from(y) + u128::from(carry_in);
    let sx = i128::from(sign_extend(x, width) as i64);
    let sy = i128::from(sign_extend(y, width) as i64);
    let signed_sum = sx + sy + i128::from(carry_in);
    let result = (unsigned_sum as u64) & m;
    let n = (result >> (width - 1)) & 1 == 1;
    let z = result == 0;
    let c = u128::from(result) != unsigned_sum;
    let v = i128::from(sign_extend(result, width) as i64) != signed_sum;
    (
        result,
        (u8::from(n) << 3) | (u8::from(z) << 2) | (u8::from(c) << 1) | u8::from(v),
    )
}

/// `ShiftReg` on an already-read register value.
pub fn shift_value(value: u64, ty: ShiftType, amount: u32, width: u32) -> u64 {
    let v = value & mask(width);
    match ty {
        ShiftType::Lsl => {
            if amount >= width {
                0
            } else {
                (v << amount) & mask(width)
            }
        }
        ShiftType::Lsr => {
            if amount >= width {
                0
            } else {
                v >> amount
            }
        }
        ShiftType::Asr => {
            let s = sign_extend(v, width) as i64;
            let amt = amount.min(63);
            ((s >> amt) as u64) & mask(width)
        }
        ShiftType::Ror => ror(v, amount, width),
    }
}

/// `ExtendReg` on an already-read register value: extend, then shift left by `shift` (0 to 4).
pub fn extend_value(value: u64, ty: ExtendType, shift: u32, width: u32) -> u64 {
    let extended = match ty {
        ExtendType::Uxtb => value & 0xFF,
        ExtendType::Uxth => value & 0xFFFF,
        ExtendType::Uxtw => value & 0xFFFF_FFFF,
        ExtendType::Uxtx => value,
        ExtendType::Sxtb => sign_extend(value, 8),
        ExtendType::Sxth => sign_extend(value, 16),
        ExtendType::Sxtw => sign_extend(value, 32),
        ExtendType::Sxtx => value,
    };
    (extended << shift) & mask(width)
}

/// `ConditionHolds(cond)` with `nzcv` in bits 3..0 as N, Z, C, V.
pub fn condition_holds(cond: u8, nzcv: u8) -> bool {
    let (n, z, c, v) = (nzcv & 8 != 0, nzcv & 4 != 0, nzcv & 2 != 0, nzcv & 1 != 0);
    let result = match cond >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if cond & 1 == 1 && cond != 0b1111 {
        !result
    } else {
        result
    }
}

/// `CountLeadingSignBits` of a `width`-bit value.
pub fn count_leading_sign_bits(value: u64, width: u32) -> u32 {
    let v = value & mask(width);
    let e = ((v >> 1) ^ v) & mask(width - 1);
    e.leading_zeros() - (64 - width) - 1
}

/// CRC32 update over the low `1 << size` bytes of `data`, little-endian, without the usual
/// initial and final inversions (the architecture leaves those to software).
pub fn crc32_update(acc: u32, data: u64, size: u8, castagnoli: bool) -> u32 {
    let poly: u32 = if castagnoli { 0x82F6_3B78 } else { 0xEDB8_8320 };
    let mut crc = acc;
    for i in 0..(1u32 << size) {
        crc ^= ((data >> (8 * i)) & 0xFF) as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (poly & 0u32.wrapping_sub(crc & 1));
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_and_sign_extend() {
        assert_eq!(mask(0), 0);
        assert_eq!(mask(8), 0xFF);
        assert_eq!(mask(64), u64::MAX);
        assert_eq!(sign_extend(0x80, 8), 0xFFFF_FFFF_FFFF_FF80);
        assert_eq!(sign_extend(0x7F, 8), 0x7F);
        assert_eq!(sign_extend(u64::MAX, 64), u64::MAX);
        assert_eq!(sign_extend(0x1_8000_0000, 32), 0xFFFF_FFFF_8000_0000);
    }

    #[test]
    fn rotate_and_replicate() {
        assert_eq!(ror(0x1, 1, 32), 0x8000_0000);
        assert_eq!(ror(0x1, 1, 64), 0x8000_0000_0000_0000);
        assert_eq!(ror(0xF0, 0, 8), 0xF0);
        assert_eq!(ror(0x12345678, 36, 32), ror(0x12345678, 4, 32));
        assert_eq!(replicate(0b01, 2, 8), 0b0101_0101);
        assert_eq!(replicate(0xAB, 8, 32), 0xABAB_ABAB);
        assert_eq!(replicate(0x1, 64, 64), 1);
    }

    /// Values from the Arm ARM description of logical immediates, checked by hand:
    /// `0x5555...` is N=0 imms=0b111100 immr=0; `0xFF` is N=1 imms=7 immr=0 (64-bit).
    #[test]
    fn bit_masks_for_known_immediates() {
        assert_eq!(
            decode_bit_masks(false, 0b111100, 0, true, 64).unwrap().0,
            0x5555_5555_5555_5555
        );
        assert_eq!(decode_bit_masks(true, 7, 0, true, 64).unwrap().0, 0xFF);
        assert_eq!(decode_bit_masks(true, 0, 0, true, 64).unwrap().0, 1);
        // 32-bit: ones(8) rotated right by 4 inside a 16-bit element
        assert_eq!(
            decode_bit_masks(false, 0b100111, 4, true, 32).unwrap().0,
            0xF00F_F00F
        );
        // 0x00FF00FF00FF00FF: element 16, ones(8)
        assert_eq!(
            decode_bit_masks(false, 0b100111, 0, true, 64).unwrap().0,
            0x00FF_00FF_00FF_00FF
        );
        // tmask for BFM-style use
        let (w, t) = decode_bit_masks(true, 15, 8, false, 64).unwrap();
        assert_eq!(w, ror(0xFFFF, 8, 64));
        assert_eq!(t, mask(8));
    }

    #[test]
    fn bit_masks_reject_reserved_patterns() {
        // all-ones S is reserved for logical immediates
        assert!(decode_bit_masks(true, 0b111111, 0, true, 64).is_none());
        assert!(decode_bit_masks(false, 0b111101, 0, true, 32).is_none());
        // the same S is fine when not an immediate (bitfield moves)
        assert!(decode_bit_masks(true, 0b111111, 0, false, 64).is_some());
        // immN = 0 with imms = 0b111110 has no element size
        assert!(decode_bit_masks(false, 0b111110, 0, true, 32).is_none());
        // N=1 needs a 64-bit datasize
        assert!(decode_bit_masks(true, 0, 0, true, 32).is_none());
    }

    #[test]
    fn add_with_carry_flags() {
        // 0xFFFF_FFFF + 1 = 0 with carry, zero, no overflow
        assert_eq!(add_with_carry(0xFFFF_FFFF, 1, false, 32), (0, 0b0110));
        // 0x7FFF_FFFF + 1 overflows into the sign
        assert_eq!(
            add_with_carry(0x7FFF_FFFF, 1, false, 32),
            (0x8000_0000, 0b1001)
        );
        // 64-bit
        assert_eq!(add_with_carry(u64::MAX, 1, false, 64), (0, 0b0110));
        assert_eq!(
            add_with_carry(i64::MAX as u64, 1, false, 64),
            (1 << 63, 0b1001)
        );
        // subtraction as x + !y + 1: 5 - 5 = 0, carry set (no borrow)
        assert_eq!(add_with_carry(5, !5u64, true, 64), (0, 0b0110));
        // 0 - 1 borrows: negative, carry clear
        assert_eq!(add_with_carry(0, !1u64, true, 64), (u64::MAX, 0b1000));
        // INT_MIN - 1 overflows
        let (r, f) = add_with_carry(1 << 63, !1u64, true, 64);
        assert_eq!(r, i64::MAX as u64);
        assert_eq!(f, 0b0011);
        // carry in participates
        assert_eq!(add_with_carry(1, 1, true, 32), (3, 0));
    }

    #[test]
    fn shifts_and_extends() {
        assert_eq!(shift_value(0x8000_0001, ShiftType::Lsl, 1, 32), 2);
        assert_eq!(shift_value(0x8000_0001, ShiftType::Lsr, 1, 32), 0x4000_0000);
        assert_eq!(
            shift_value(0x8000_0000, ShiftType::Asr, 31, 32),
            0xFFFF_FFFF
        );
        assert_eq!(
            shift_value(0x8000_0000_0000_0000, ShiftType::Asr, 63, 64),
            u64::MAX
        );
        assert_eq!(shift_value(0x1, ShiftType::Ror, 1, 32), 0x8000_0000);
        assert_eq!(
            shift_value(0xFFFF_FFFF_FFFF_FFFF, ShiftType::Lsl, 0, 32),
            0xFFFF_FFFF
        );
        assert_eq!(extend_value(0x1FF, ExtendType::Uxtb, 0, 64), 0xFF);
        assert_eq!(
            extend_value(0x80, ExtendType::Sxtb, 0, 64),
            0xFFFF_FFFF_FFFF_FF80
        );
        assert_eq!(extend_value(0x80, ExtendType::Sxtb, 2, 32), 0xFFFF_FE00);
        assert_eq!(
            extend_value(0x8000_0000, ExtendType::Sxtw, 0, 64),
            0xFFFF_FFFF_8000_0000
        );
        assert_eq!(
            extend_value(0x1234_5678_9ABC, ExtendType::Uxtx, 4, 64),
            0x1_2345_6789_ABC0
        );
    }

    #[test]
    fn conditions() {
        let (n, z, c, v) = (8u8, 4u8, 2u8, 1u8);
        assert!(condition_holds(0, z) && !condition_holds(1, z)); // EQ NE
        assert!(condition_holds(2, c) && !condition_holds(3, c)); // CS CC
        assert!(condition_holds(4, n) && !condition_holds(5, n)); // MI PL
        assert!(condition_holds(6, v) && !condition_holds(7, v)); // VS VC
        assert!(condition_holds(8, c) && !condition_holds(8, c | z)); // HI
        assert!(condition_holds(9, z) && condition_holds(9, 0)); // LS
        assert!(condition_holds(10, n | v) && condition_holds(10, 0)); // GE
        assert!(condition_holds(11, n) && condition_holds(11, v)); // LT
        assert!(condition_holds(12, 0) && !condition_holds(12, z)); // GT
        assert!(condition_holds(13, z) && condition_holds(13, n)); // LE
        assert!(condition_holds(14, 0) && condition_holds(15, n | z | c | v)); // AL NV
    }

    #[test]
    fn leading_sign_bits() {
        assert_eq!(count_leading_sign_bits(0, 32), 31);
        assert_eq!(count_leading_sign_bits(0xFFFF_FFFF, 32), 31);
        assert_eq!(count_leading_sign_bits(1, 32), 30);
        assert_eq!(count_leading_sign_bits(0x8000_0000, 32), 0);
        assert_eq!(count_leading_sign_bits(0x4000_0000, 32), 0);
        assert_eq!(count_leading_sign_bits(0, 64), 63);
        assert_eq!(count_leading_sign_bits(u64::MAX << 4, 64), 59);
    }

    #[test]
    fn crc32_matches_published_check_values() {
        // CRC-32 of "123456789" is 0xCBF43926 with the usual init/final inversion.
        let data = b"123456789";
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc = crc32_update(crc, u64::from(b), 0, false);
        }
        assert_eq!(!crc, 0xCBF4_3926);
        // CRC-32C check value 0xE3069283
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc = crc32_update(crc, u64::from(b), 0, true);
        }
        assert_eq!(!crc, 0xE306_9283);
        // wider updates equal repeated byte updates
        let word = 0x0807_0605_0403_0201u64;
        let mut by_bytes = 0x1234_5678u32;
        for b in 1..=8u64 {
            by_bytes = crc32_update(by_bytes, b, 0, false);
        }
        assert_eq!(crc32_update(0x1234_5678, word, 3, false), by_bytes);
    }
}
