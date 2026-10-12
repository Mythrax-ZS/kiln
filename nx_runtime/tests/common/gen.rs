//! Random inputs for the differential tests: register values biased toward the boundaries where
//! arithmetic goes wrong, and instruction fields drawn from the valid encoding space.

use super::harness::{Case, DATA_BASE, DATA_SIZE};
use nx_runtime::a64::bits::decode_bit_masks;
use nx_runtime::a64::CpuState;
use nx_runtime::testing::Rng;

pub struct Gen {
    pub rng: Rng,
}

impl Gen {
    pub fn new(seed: u64) -> Gen {
        Gen {
            rng: Rng::new(seed),
        }
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.rng.below(n)
    }

    pub fn flag(&mut self) -> bool {
        self.rng.below(2) == 1
    }

    pub fn reg(&mut self) -> u8 {
        self.rng.below(32) as u8
    }

    /// A register other than 31.
    pub fn gpr(&mut self) -> u8 {
        self.rng.below(31) as u8
    }

    pub fn value(&mut self) -> u64 {
        const SPECIAL: [u64; 16] = [
            0,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFF,
            0x1_0000_0000,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0x8000_0000_8000_0000,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x00FF_00FF_00FF_00FF,
            0xFFFF_0000_FFFF_0000,
        ];
        match self.rng.below(8) {
            0 | 1 => SPECIAL[self.rng.below(SPECIAL.len() as u64) as usize],
            2 => self.rng.below(256),
            3 => self.rng.next_u64() >> self.rng.below(64),
            4 => 1u64 << self.rng.below(64),
            5 => !(1u64 << self.rng.below(64)),
            _ => self.rng.next_u64(),
        }
    }

    pub fn nzcv(&mut self) -> u8 {
        self.rng.below(16) as u8
    }

    /// A random architectural state with a valid, 16-byte aligned stack pointer in the data region.
    pub fn state(&mut self) -> CpuState {
        let mut s = CpuState::default();
        for r in s.x.iter_mut() {
            *r = self.value();
        }
        s.sp = DATA_BASE + 0x2000 + 16 * self.below(64);
        s.set_nzcv4(self.nzcv());
        s.tpidr_el0 = self.rng.next_u64();
        s.tpidrro_el0 = self.rng.next_u64();
        s
    }

    pub fn data(&mut self) -> Vec<u8> {
        self.rng.bytes(DATA_SIZE)
    }

    pub fn case(&mut self, words: Vec<u32>) -> Case {
        Case {
            steps: words.len(),
            words,
            state: self.state(),
            data: self.data(),
        }
    }

    /// A valid (N, immr, imms) triple for a logical immediate of the given size.
    pub fn logical_imm_fields(&mut self, sf: bool) -> (bool, u8, u8) {
        loop {
            let n = sf && self.flag();
            let immr = self.below(64) as u8;
            let imms = self.below(64) as u8;
            if decode_bit_masks(n, imms, immr, true, if sf { 64 } else { 32 }).is_some() {
                return (n, immr, imms);
            }
        }
    }

    /// A valid (immr, imms) pair for a bitfield move.
    pub fn bitfield_fields(&mut self, sf: bool) -> (u8, u8) {
        let limit = if sf { 64 } else { 32 };
        (self.below(limit) as u8, self.below(limit) as u8)
    }

    /// An address inside the data region leaving `before` bytes below and `after` bytes above.
    pub fn data_pointer(&mut self, before: u64, after: u64) -> u64 {
        let span = DATA_SIZE as u64 - before - after;
        DATA_BASE + before + self.below(span)
    }
}
