//! Random floating-point and SIMD inputs for the differential tests, built on the generators in
//! `nx_runtime::testing::fpvalues`.

use super::gen::Gen;
use super::harness::{Case, DATA_BASE, DATA_SIZE};
use nx_runtime::a64::CpuState;
use nx_runtime::testing::fpvalues;

impl Gen {
    /// One 128-bit register worth of interesting floating-point bits.
    pub fn fp_register(&mut self) -> u128 {
        fpvalues::fp_register(&mut self.rng)
    }

    pub fn lane_register(&mut self) -> u128 {
        fpvalues::lane_register(&mut self.rng)
    }

    pub fn fpcr(&mut self) -> u32 {
        fpvalues::fpcr(&mut self.rng)
    }

    pub fn fpsr(&mut self) -> u32 {
        fpvalues::fpsr(&mut self.rng)
    }

    /// A state with floating-point registers, control registers, flags and general registers
    /// (some holding pointers into the data region).
    pub fn fp_state(&mut self) -> CpuState {
        let mut s = self.state();
        for n in 0..32u8 {
            let v = self.fp_register();
            s.set_vreg(n, v);
        }
        for slot in s.x.iter_mut().take(31) {
            if self.below(3) == 0 {
                *slot = self.data_pointer(0x1000, 0x1000);
            }
        }
        s.fpcr = self.fpcr();
        s.fpsr = self.fpsr();
        s
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

    pub fn fp_case(&mut self, words: Vec<u32>) -> Case {
        Case {
            steps: words.len(),
            words,
            state: self.fp_state(),
            data: self.data(),
        }
    }

    pub fn simd_case(&mut self, words: Vec<u32>) -> Case {
        Case {
            steps: words.len(),
            words,
            state: self.simd_state(),
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
