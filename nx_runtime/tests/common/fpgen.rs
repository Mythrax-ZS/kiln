//! Random floating-point and SIMD inputs for the differential tests: register contents built
//! from the values arithmetic gets wrong (zeros, denormals, infinities, NaNs of both kinds, ties,
//! exact powers of two, values next to integer limits), and control registers covering the
//! rounding modes and the flush-to-zero and default-NaN switches.

use super::gen::Gen;
use super::harness::{Case, DATA_BASE, DATA_SIZE};
use nx_runtime::a64::CpuState;

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

impl Gen {
    /// One 128-bit register worth of interesting bits.
    pub fn fp_register(&mut self) -> u128 {
        let pick64 = |g: &mut Gen| F64_SPECIAL[g.below(F64_SPECIAL.len() as u64) as usize];
        let pick32 =
            |g: &mut Gen| u64::from(F32_SPECIAL[g.below(F32_SPECIAL.len() as u64) as usize]);
        match self.below(6) {
            0 | 1 => {
                let (a, b) = (pick64(self), pick64(self));
                u128::from(a) | (u128::from(b) << 64)
            }
            2 | 3 => {
                let words: Vec<u64> = (0..4).map(|_| pick32(self)).collect();
                u128::from(words[0] | (words[1] << 32))
                    | (u128::from(words[2] | (words[3] << 32)) << 64)
            }
            4 => {
                let a = pick32(self);
                u128::from(a)
            }
            _ => u128::from(self.rng.next_u64()) | (u128::from(self.rng.next_u64()) << 64),
        }
    }

    /// A register of integer lanes at the boundaries (zero, one, extremes, all ones, small values
    /// and random bits) for a lane size picked at random.
    pub fn lane_register(&mut self) -> u128 {
        let esize = 8u32 << self.below(4);
        let mask = if esize == 64 {
            u64::MAX
        } else {
            (1u64 << esize) - 1
        };
        let mut r = 0u128;
        for i in 0..128 / esize {
            let v = match self.below(10) {
                0 => 0,
                1 => 1,
                2 => mask,
                3 => 1u64 << (esize - 1),
                4 => (1u64 << (esize - 1)) - 1,
                5 => self.below(8),
                6 => mask - self.below(8),
                7 => (1u64 << (esize - 1)).wrapping_add(self.below(4)) & mask,
                _ => self.rng.next_u64() & mask,
            };
            r |= u128::from(v & mask) << (i * esize);
        }
        r
    }

    /// A state for the integer SIMD tests: lane-boundary registers, a few floating-point ones.
    pub fn simd_state(&mut self) -> CpuState {
        let mut s = self.fp_state();
        for n in 0..32u8 {
            if self.below(4) != 0 {
                let v = self.lane_register();
                s.set_vreg(n, v);
            }
        }
        s
    }

    pub fn simd_case(&mut self, words: Vec<u32>) -> Case {
        Case {
            steps: words.len(),
            words,
            state: self.simd_state(),
            data: self.data(),
        }
    }

    pub fn fpcr(&mut self) -> u32 {
        match self.below(10) {
            0..=3 => 0,
            4 => (self.below(4) as u32) << 22,
            5 => 1 << 24,
            6 => 1 << 25,
            7 => ((self.below(4) as u32) << 22) | (1 << 24) | (1 << 25),
            8 => ((self.below(4) as u32) << 22) | ((self.below(2) as u32) << 26),
            _ => (self.rng.next_u32()) & 0x07F7_0000,
        }
    }

    pub fn fpsr(&mut self) -> u32 {
        if self.below(5) < 3 {
            0
        } else {
            self.rng.next_u32() & 0xF800_009F
        }
    }

    /// A state with floating-point registers, control registers, flags and general registers
    /// (some holding pointers into the data region).
    pub fn fp_state(&mut self) -> CpuState {
        let mut s = self.state();
        for n in 0..32u8 {
            let v = self.fp_register();
            s.set_vreg(n, v);
        }
        for r in 0..31 {
            if self.below(3) == 0 {
                s.x[r] = self.data_pointer(0x1000, 0x1000);
            }
        }
        s.fpcr = self.fpcr();
        s.fpsr = self.fpsr();
        s
    }

    pub fn fp_case(&mut self, words: Vec<u32>) -> Case {
        Case {
            steps: words.len(),
            words,
            state: self.fp_state(),
            data: self.data(),
        }
    }

    /// A word of an encoding group: the fixed bits `value` under `mask`, the rest random.
    pub fn template_word(&mut self, mask: u32, value: u32) -> u32 {
        (self.rng.next_u32() & !mask) | value
    }
}

/// Where the data region's middle is, for loads and stores through `x0..x30`.
pub fn data_middle() -> u64 {
    DATA_BASE + (DATA_SIZE as u64) / 2
}
