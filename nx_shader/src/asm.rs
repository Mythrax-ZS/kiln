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

/// The control-flow instruction `inst` (made with the test 15, always) with the condition code test
/// `cc` in its low five bits.
pub fn with_cc(inst: u64, cc: u8) -> u64 {
    inst & !0x1f | u64::from(cc)
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

/// A builder for any instruction of the table: start from an op, set the fields.
#[derive(Clone, Copy, Debug)]
pub struct Enc(u64);

impl Enc {
    /// The op's fixed bits, guarded by `PT` (always).
    pub fn op(op: crate::ops::Op) -> Enc {
        let row = crate::ops::TABLE
            .iter()
            .find(|r| r.op == op)
            .expect("every op is in the table");
        Enc(row.value | u64::from(crate::inst::PT) << 16)
    }

    pub fn field(mut self, lo: u32, len: u32, v: u64) -> Enc {
        let mask = if len == 64 {
            u64::MAX
        } else {
            (1u64 << len) - 1
        };
        self.0 = self.0 & !(mask << lo) | (v & mask) << lo;
        self
    }

    pub fn bit(self, n: u32, v: bool) -> Enc {
        self.field(n, 1, u64::from(v))
    }

    pub fn rd(self, r: u8) -> Enc {
        self.field(0, 8, u64::from(r))
    }

    pub fn ra(self, r: u8) -> Enc {
        self.field(8, 8, u64::from(r))
    }

    pub fn rb(self, r: u8) -> Enc {
        self.field(20, 8, u64::from(r))
    }

    pub fn rc(self, r: u8) -> Enc {
        self.field(39, 8, u64::from(r))
    }

    pub fn guard(self, g: Guard) -> Enc {
        self.field(16, 4, u64::from(g.index) | u64::from(g.negated) << 3)
    }

    /// A predicate operand at bits `lo..lo+2` with its negation in bit `lo+3`.
    pub fn pred(self, lo: u32, p: u8, negated: bool) -> Enc {
        self.field(lo, 4, u64::from(p) | u64::from(negated) << 3)
    }

    /// A constant buffer operand: slot in bits 34..38, byte offset in bits 20..33 (in words).
    pub fn cbuf(self, slot: u32, offset: u32) -> Enc {
        self.field(34, 5, u64::from(slot))
            .field(20, 14, u64::from(offset / 4))
    }

    /// A 20-bit immediate: low 19 bits at 20, the sign in bit 56.
    pub fn imm20(self, v: i32) -> Enc {
        let raw = (v as u32) & 0xF_FFFF;
        self.field(20, 19, u64::from(raw & 0x7_FFFF))
            .field(56, 1, u64::from(raw >> 19))
    }

    /// A float immediate in 20 bits: the top of the `f32`'s bits.
    pub fn imm20_f32(self, v: f32) -> Enc {
        let raw = v.to_bits() >> 12;
        self.field(20, 19, u64::from(raw & 0x7_FFFF))
            .field(56, 1, u64::from(raw >> 19))
    }

    pub fn imm32(self, v: u32) -> Enc {
        self.field(20, 32, u64::from(v))
    }

    pub fn build(self) -> u64 {
        self.0
    }
}
