//! Register contents and control-register values for the floating-point and SIMD tests: the
//! values arithmetic gets wrong (zeros, denormals, infinities, NaNs of both kinds, ties, exact
//! powers of two, values next to integer limits), integer lanes at their boundaries, and
//! `FPCR`/`FPSR` values covering the rounding modes and the flush-to-zero and default-NaN switches.

use super::Rng;

pub const F32_SPECIAL: [u32; 40] = [
    0x0000_0000,
    0x8000_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x4000_0000,
    0x3F00_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x0080_0000,
    0x8080_0000,
    0x007F_FFFF,
    0x0000_0001,
    0x8000_0001,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFFC0_0000,
    0x7F80_0001,
    0xFF80_0001,
    0x7FC1_2345,
    0xFFC1_2345,
    0x7FA0_0000,
    0x4040_0000,
    0x3FC0_0000,
    0x4020_0000,
    0x4060_0000,
    0xC020_0000,
    0x4F00_0000,
    0x4F80_0000,
    0x5F00_0000,
    0x5F80_0000,
    0xCF00_0000,
    0xDF00_0000,
    0x4EFF_FFFF,
    0x3F7F_FFFF,
    0x3F80_0001,
    0x3400_0000,
    0x7E80_0000,
    0x0100_0000,
    0x4B00_0000,
];

pub const F64_SPECIAL: [u64; 40] = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000,
    0x4000_0000_0000_0000,
    0x3FE0_0000_0000_0000,
    0x7FEF_FFFF_FFFF_FFFF,
    0xFFEF_FFFF_FFFF_FFFF,
    0x0010_0000_0000_0000,
    0x8010_0000_0000_0000,
    0x000F_FFFF_FFFF_FFFF,
    0x0000_0000_0000_0001,
    0x8000_0000_0000_0001,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x7FF8_0000_0000_0000,
    0xFFF8_0000_0000_0000,
    0x7FF0_0000_0000_0001,
    0xFFF0_0000_0000_0001,
    0x7FF8_1234_5678_9ABC,
    0xFFF8_1234_5678_9ABC,
    0x7FF4_0000_0000_0000,
    0x4008_0000_0000_0000,
    0x3FF8_0000_0000_0000,
    0x4004_0000_0000_0000,
    0x400C_0000_0000_0000,
    0xC004_0000_0000_0000,
    0x41E0_0000_0000_0000,
    0x41F0_0000_0000_0000,
    0x43E0_0000_0000_0000,
    0x43F0_0000_0000_0000,
    0xC1E0_0000_0000_0000,
    0xC3E0_0000_0000_0000,
    0x41DF_FFFF_FFC0_0000,
    0x3FEF_FFFF_FFFF_FFFF,
    0x3FF0_0000_0000_0001,
    0x3CB0_0000_0000_0000,
    0x7FD0_0000_0000_0000,
    0x2000_0000_0000_0000,
    0x4330_0000_0000_0000,
];

/// One 128-bit register of floating-point edge values.
pub fn fp_register(rng: &mut Rng) -> u128 {
    let pick64 = |r: &mut Rng| F64_SPECIAL[r.below(F64_SPECIAL.len() as u64) as usize];
    let pick32 = |r: &mut Rng| u64::from(F32_SPECIAL[r.below(F32_SPECIAL.len() as u64) as usize]);
    match rng.below(6) {
        0 | 1 => {
            let (a, b) = (pick64(rng), pick64(rng));
            u128::from(a) | (u128::from(b) << 64)
        }
        2 | 3 => {
            let words: Vec<u64> = (0..4).map(|_| pick32(rng)).collect();
            u128::from(words[0] | (words[1] << 32))
                | (u128::from(words[2] | (words[3] << 32)) << 64)
        }
        4 => u128::from(pick32(rng)),
        _ => u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64),
    }
}

/// A register of integer lanes at the boundaries (zero, one, extremes, all ones, small values and
/// random bits) for a lane size picked at random.
pub fn lane_register(rng: &mut Rng) -> u128 {
    let esize = 8u32 << rng.below(4);
    let mask = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let mut r = 0u128;
    for i in 0..128 / esize {
        let v = match rng.below(10) {
            0 => 0,
            1 => 1,
            2 => mask,
            3 => 1u64 << (esize - 1),
            4 => (1u64 << (esize - 1)) - 1,
            5 => rng.below(8),
            6 => mask - rng.below(8),
            7 => (1u64 << (esize - 1)).wrapping_add(rng.below(4)) & mask,
            _ => rng.next_u64() & mask,
        };
        r |= u128::from(v & mask) << (i * esize);
    }
    r
}

/// A vector register for a test of either kind.
pub fn vector_register(rng: &mut Rng) -> u128 {
    if rng.below(4) == 0 {
        fp_register(rng)
    } else {
        lane_register(rng)
    }
}

/// An `FPCR` value: mostly the default, otherwise rounding modes, flush-to-zero, default NaN and
/// the other writable bits.
pub fn fpcr(rng: &mut Rng) -> u32 {
    match rng.below(10) {
        0..=3 => 0,
        4 => (rng.below(4) as u32) << 22,
        5 => 1 << 24,
        6 => 1 << 25,
        7 => ((rng.below(4) as u32) << 22) | (1 << 24) | (1 << 25),
        8 => ((rng.below(4) as u32) << 22) | ((rng.below(2) as u32) << 26),
        _ => rng.next_u32() & 0x07F7_0000,
    }
}

/// An `FPSR` value: often clear, otherwise random in the writable bits.
pub fn fpsr(rng: &mut Rng) -> u32 {
    if rng.below(5) < 3 {
        0
    } else {
        rng.next_u32() & 0xF800_009F
    }
}
