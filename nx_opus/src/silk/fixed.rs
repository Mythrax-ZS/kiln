//! The fixed-point arithmetic of the SILK layer. SILK's decoder is integer throughout and its
//! output is defined by the exact rounding of these operations (the reference's `silk/macros.h`,
//! `silk/SigProc_FIX.h` and `silk/Inlines.h`), so each helper here does what its macro does,
//! including the wrap-around of 32-bit arithmetic where the reference relies on it.

/// `(a * (b as i16)) >> 16` (`silk_SMULWB`).
pub fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

/// `a + smulwb(b, c)` (`silk_SMLAWB`).
pub fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

/// The product of the low halves (`silk_SMULBB`).
pub fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

/// `a + smulbb(b, c)` (`silk_SMLABB`), wrapping.
pub fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulbb(b, c))
}

/// `(a * b) >> 16` with the rounding of `silk_SMULWW`.
pub fn smulww(a: i32, b: i32) -> i32 {
    smulwb(a, b).wrapping_add(a.wrapping_mul(rshift_round(b, 16)))
}

/// `a + smulww(b, c)` (`silk_SMLAWW`).
pub fn smlaww(a: i32, b: i32, c: i32) -> i32 {
    smlawb(a, b, c).wrapping_add(b.wrapping_mul(rshift_round(c, 16)))
}

/// The high 32 bits of the 64-bit product (`silk_SMMUL`).
pub fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

/// A right shift rounding to nearest (`silk_RSHIFT_ROUND`), `shift` > 0.
pub fn rshift_round(a: i32, shift: u32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// The 64-bit form (`silk_RSHIFT_ROUND64`).
pub fn rshift_round64(a: i64, shift: u32) -> i64 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// Clamp to 16 bits (`silk_SAT16`).
pub fn sat16(a: i32) -> i32 {
    a.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

/// `silk_LIMIT`: clamp between two bounds given in either order.
pub fn limit(a: i32, l1: i32, l2: i32) -> i32 {
    if l1 > l2 {
        a.clamp(l2, l1)
    } else {
        a.clamp(l1, l2)
    }
}

/// A left shift that saturates instead of overflowing (`silk_LSHIFT_SAT32`).
pub fn lshift_sat32(a: i32, shift: u32) -> i32 {
    limit(a, i32::MIN >> shift, i32::MAX >> shift) << shift
}

/// Leading zeros of a 32-bit value, 32 for 0 (`silk_CLZ32`).
pub fn clz32(a: i32) -> i32 {
    (a as u32).leading_zeros() as i32
}

/// The leading zeros and the 7 bits after the leading one (`silk_CLZ_FRAC`).
pub fn clz_frac(a: i32) -> (i32, i32) {
    let lz = clz32(a);
    let rot = 24 - lz;
    let x = a as u32;
    let r = if rot == 0 {
        x
    } else if rot < 0 {
        x.rotate_left((-rot) as u32)
    } else {
        x.rotate_right(rot as u32)
    };
    (lz, (r & 0x7F) as i32)
}

/// An approximate square root (`silk_SQRT_APPROX`).
pub fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32768 } else { 46214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(213, frac))
}

/// About `(a << q) / b` (`silk_DIV32_varQ`).
pub fn div32_varq(a: i32, b: i32, q: i32) -> i32 {
    let a_headrm = clz32(a.wrapping_abs()) - 1;
    let mut a_nrm = a << a_headrm;
    let b_headrm = clz32(b.wrapping_abs()) - 1;
    let b_nrm = b << b_headrm;
    // The inverse of b with 14 bits of precision.
    let b_inv = (i32::MAX >> 2) / (b_nrm >> 16);
    let mut result = smulwb(a_nrm, b_inv);
    // The residual, which may overflow on the way.
    a_nrm = a_nrm.wrapping_sub(((smmul(b_nrm, result) as u32) << 3) as i32);
    result = smlawb(result, a_nrm, b_inv);
    let lshift = 29 + a_headrm - b_headrm - q;
    if lshift < 0 {
        lshift_sat32(result, (-lshift) as u32)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// About `(1 << q) / b` (`silk_INVERSE32_varQ`).
pub fn inverse32_varq(b: i32, q: i32) -> i32 {
    let b_headrm = clz32(b.wrapping_abs()) - 1;
    let b_nrm = b << b_headrm;
    let b_inv = (i32::MAX >> 2) / (b_nrm >> 16);
    let mut result = b_inv << 16;
    let err_q32 = ((1 << 29) - smulwb(b_nrm, b_inv)) << 3;
    result = smlaww(result, err_q32, b_inv);
    let lshift = 61 - b_headrm - q;
    if lshift <= 0 {
        lshift_sat32(result, (-lshift) as u32)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// 2^(x / 128), approximately (`silk_log2lin`).
pub fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    let out = 1i32 << (in_log_q7 >> 7);
    let frac = in_log_q7 & 0x7F;
    let poly = smlawb(frac, smulbb(frac, 128 - frac), -174);
    if in_log_q7 < 2048 {
        out + ((out * poly) >> 7)
    } else {
        out + (out >> 7) * poly
    }
}

/// 128 log2(x), approximately (`silk_lin2log`): the decoder needs only its inverse, which the
/// tests check against it.
#[cfg(test)]
fn lin2log(in_lin: i32) -> i32 {
    let (lz, frac) = clz_frac(in_lin);
    ((31 - lz) << 7) + smlawb(frac, frac * (128 - frac), 179)
}

/// The generator of SILK's pseudo-random numbers (`silk_RAND`).
pub fn rand(seed: i32) -> i32 {
    907_633_515i32.wrapping_add(seed.wrapping_mul(196_314_165))
}

/// The energy of a signal with the right shift that keeps it in 32 bits (`silk_sum_sqr_shift`).
pub fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    let len = x.len() as i32 - 1;
    let mut nrg: i32 = 0;
    let mut shft = 0;
    let mut i = 0i32;
    while i < len {
        nrg = smlabb(nrg, i32::from(x[i as usize]), i32::from(x[i as usize]));
        nrg = smlabb(
            nrg,
            i32::from(x[i as usize + 1]),
            i32::from(x[i as usize + 1]),
        );
        if nrg < 0 {
            // The reference leaves `i` on this pair, which the second loop adds again.
            nrg = ((nrg as u32) >> 2) as i32;
            shft = 2;
            break;
        }
        i += 2;
    }
    while i < len {
        let mut t = smulbb(i32::from(x[i as usize]), i32::from(x[i as usize]));
        t = smlabb(
            t,
            i32::from(x[i as usize + 1]),
            i32::from(x[i as usize + 1]),
        );
        nrg = (nrg as u32).wrapping_add((t as u32) >> shft) as i32;
        if nrg < 0 {
            nrg = ((nrg as u32) >> 2) as i32;
            shft += 2;
        }
        i += 2;
    }
    if i == len {
        let t = smulbb(i32::from(x[i as usize]), i32::from(x[i as usize]));
        nrg = (nrg as u32).wrapping_add((t as u32) >> shft) as i32;
    }
    // At least two leading zeros.
    if (nrg as u32) & 0xC000_0000 != 0 {
        nrg = ((nrg as u32) >> 2) as i32;
        shft += 2;
    }
    (nrg, shft)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_multiplies_round_down() {
        assert_eq!(smulwb(1 << 16, 3), 3);
        assert_eq!(smulwb(-1, 1), -1);
        assert_eq!(
            smulwb(0x12345678, 0x0001_8000),
            ((0x12345678i64 * -32768) >> 16) as i32
        );
        assert_eq!(smulww(1 << 20, 3 << 16), 3 << 20);
        assert_eq!(smmul(1 << 30, 1 << 30), 1 << 28);
    }

    #[test]
    fn rounding_shifts_round_half_up() {
        assert_eq!(rshift_round(5, 1), 3);
        assert_eq!(rshift_round(-5, 1), -2);
        assert_eq!(rshift_round(6, 2), 2);
        assert_eq!(rshift_round(-6, 2), -1);
    }

    #[test]
    fn divisions_are_close_to_exact() {
        for &(a, b) in &[(1000, 7), (-5000, 123), (65536, 65536), (12345678, 4321)] {
            let q = div32_varq(a, b, 16);
            let exact = ((i64::from(a) << 16) / i64::from(b)) as i32;
            assert!(
                (q - exact).abs() <= (exact.abs() >> 12) + 2,
                "{a} / {b}: {q} vs {exact}"
            );
        }
        // 2^46 / 2^16 = 2^30.
        let inv = inverse32_varq(1 << 16, 46);
        assert!((inv - (1 << 30)).abs() < 1 << 16, "{inv}");
    }

    #[test]
    fn log_and_linear_are_near_inverses() {
        for v in [1i32, 100, 1000, 65536, 1 << 24] {
            let back = log2lin(lin2log(v));
            assert!((back - v).abs() <= v / 50 + 1, "{v} -> {back}");
        }
    }

    #[test]
    fn energy_with_shift_matches_the_sum() {
        let x: Vec<i16> = (0..40).map(|i| (i * 37 % 200 - 100) as i16).collect();
        let (e, s) = sum_sqr_shift(&x);
        let exact: i64 = x.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
        assert_eq!(s, 0);
        assert_eq!(i64::from(e), exact);
    }
}
