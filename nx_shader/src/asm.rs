//! A small assembler for building Maxwell programs in tests: encoders for the instructions the tests
//! of this and other crates need, and the layout of instructions into bundles.

use crate::inst::Guard;

pub const P0: Guard = Guard {
    index: 0,
    negated: false,
};
pub const NOT_P0: Guard = Guard {
    index: 0,
    negated: true,
};

pub fn guard_bits(g: Guard) -> u64 {
    u64::from(g.index) << 16 | u64::from(g.negated) << 19
}

/// The scheduling word of a bundle (all zero: no scheduling information).
pub fn sched() -> u64 {
    0
}

fn branch_field(from: u32, to: u32) -> u64 {
    let off = i64::from(to) - i64::from(from) - 8;
    ((off as u64) & 0xFF_FFFF) << 20
}

/// `BRA to` placed at byte offset `at`, under `g`.
pub fn bra(at: u32, to: u32, g: Guard) -> u64 {
    0xe240_0000_0000_000f | guard_bits(g) | branch_field(at, to)
}

pub fn ssy(at: u32, to: u32) -> u64 {
    0xe290_0000_0000_0000 | 0x7 << 16 | branch_field(at, to)
}

pub fn pbk(at: u32, to: u32) -> u64 {
    0xe2a0_0000_0000_0000 | 0x7 << 16 | branch_field(at, to)
}

pub fn pcnt(at: u32, to: u32) -> u64 {
    0xe2b0_0000_0000_0000 | 0x7 << 16 | branch_field(at, to)
}

pub fn sync(g: Guard) -> u64 {
    0xf0f8_0000_0000_000f | guard_bits(g)
}

pub fn brk(g: Guard) -> u64 {
    0xe340_0000_0000_000f | guard_bits(g)
}

pub fn cont(g: Guard) -> u64 {
    0xe350_0000_0000_000f | guard_bits(g)
}

pub fn exit(g: Guard) -> u64 {
    0xe300_0000_0000_000f | guard_bits(g)
}

pub fn kil(g: Guard) -> u64 {
    0xe330_0000_0000_000f | guard_bits(g)
}

pub fn ret() -> u64 {
    0xe320_0000_0000_000f | 0x7 << 16
}

/// `MOV32I rd, imm` (all four lanes enabled).
pub fn mov32i(rd: u8, imm: u32) -> u64 {
    0x0100_0000_0000_0000 | 0x7 << 16 | u64::from(rd) | u64::from(imm) << 20 | 0xf << 12
}

/// Lay instructions out in bundles with scheduling words and return the bytes. The first
/// instruction goes at offset 8.
pub fn assemble(insts: &[u64]) -> Vec<u8> {
    let mut words = Vec::new();
    let mut it = insts.iter();
    loop {
        words.push(sched());
        for _ in 0..3 {
            words.push(it.next().copied().unwrap_or(0));
        }
        if it.len() == 0 {
            break;
        }
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
