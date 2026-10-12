//! Access to the vector register file.

use crate::a64::fp::Fmt;
use crate::a64::state::CpuState;

impl CpuState {
    /// `V[n]` as one 128-bit value.
    pub fn vreg(&self, n: u8) -> u128 {
        let [lo, hi] = self.v[usize::from(n)];
        u128::from(lo) | (u128::from(hi) << 64)
    }

    pub fn set_vreg(&mut self, n: u8, v: u128) {
        self.v[usize::from(n)] = [v as u64, (v >> 64) as u64];
    }

    /// The low bits of `V[n]` as a scalar of the format.
    pub fn scalar(&self, n: u8, fmt: Fmt) -> u64 {
        self.v[usize::from(n)][0] & fmt.mask()
    }

    /// Write a scalar: the rest of the 128-bit register becomes zero.
    pub fn set_scalar(&mut self, n: u8, bits: u64) {
        self.v[usize::from(n)] = [bits, 0];
    }
}
