//! One 64-bit Maxwell instruction and the fields every instruction format shares.

use crate::ops::{Form, Op, Row, TABLE};

/// The register that reads as zero and ignores writes.
pub const RZ: u8 = 255;
/// The predicate that is always true.
pub const PT: u8 = 7;

/// A 64-bit instruction word.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Inst(pub u64);

impl std::fmt::Debug for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Inst({:#018x})", self.0)
    }
}

/// The predicate an instruction is guarded by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guard {
    /// 0..=6 for P0..P6, 7 for PT.
    pub index: u8,
    pub negated: bool,
}

impl Guard {
    pub const ALWAYS: Guard = Guard {
        index: PT,
        negated: false,
    };

    pub fn is_always(self) -> bool {
        self == Guard::ALWAYS
    }

    /// `@!PT`: the instruction never executes.
    pub fn is_never(self) -> bool {
        self.index == PT && self.negated
    }
}

impl Inst {
    /// `len` bits from bit `lo`.
    pub fn bits(self, lo: u32, len: u32) -> u64 {
        debug_assert!(len >= 1 && lo + len <= 64);
        let v = self.0 >> lo;
        if len == 64 {
            v
        } else {
            v & ((1u64 << len) - 1)
        }
    }

    pub fn bit(self, n: u32) -> bool {
        self.0 >> n & 1 != 0
    }

    /// `len` bits from bit `lo`, sign-extended.
    pub fn sbits(self, lo: u32, len: u32) -> i64 {
        let v = self.bits(lo, len) as i64;
        let shift = 64 - len;
        (v << shift) >> shift
    }

    /// The table row that describes this instruction, if one does.
    pub fn row(self) -> Option<&'static Row> {
        TABLE.iter().find(|r| self.0 & r.mask == r.value)
    }

    pub fn op(self) -> Option<Op> {
        self.row().map(|r| r.op)
    }

    /// The predicate guard (bits 16..18 and the negation in bit 19).
    pub fn guard(self) -> Guard {
        Guard {
            index: self.bits(16, 3) as u8,
            negated: self.bit(19),
        }
    }

    /// The destination register (bits 0..7).
    pub fn rd(self) -> u8 {
        self.bits(0, 8) as u8
    }

    /// The first source register (bits 8..15).
    pub fn ra(self) -> u8 {
        self.bits(8, 8) as u8
    }

    /// The second source register of the register form (bits 20..27).
    pub fn rb(self) -> u8 {
        self.bits(20, 8) as u8
    }

    /// The register at bits 39..46: the third source of the three-source forms.
    pub fn rc(self) -> u8 {
        self.bits(39, 8) as u8
    }

    /// The register at bits 28..35.
    pub fn r28(self) -> u8 {
        self.bits(28, 8) as u8
    }

    /// The constant buffer index of an operand (bits 34..38).
    pub fn cbuf_index(self) -> u32 {
        self.bits(34, 5) as u32
    }

    /// The byte offset into the constant buffer of an operand (14 bits at 20, in words).
    pub fn cbuf_offset(self) -> u32 {
        (self.bits(20, 14) as u32) << 2
    }

    /// The 20-bit signed immediate (bits 20..38 and the sign in bit 56).
    pub fn imm20(self) -> i32 {
        let raw = self.bits(20, 19) as u32 | (self.bit(56) as u32) << 19;
        ((raw << 12) as i32) >> 12
    }

    /// A 20-bit float immediate: the top bits of an `f32`.
    pub fn imm20_f32_bits(self) -> u32 {
        let raw = self.bits(20, 19) as u32 | (self.bit(56) as u32) << 19;
        raw << 12
    }

    /// The 32-bit immediate (bits 20..51).
    pub fn imm32(self) -> u32 {
        self.bits(20, 32) as u32
    }

    /// The signed 24-bit branch offset (bits 20..43) of a relative branch.
    pub fn branch_offset(self) -> i32 {
        self.sbits(20, 24) as i32
    }

    /// The form of the second source operand as the table gives it.
    pub fn form(self) -> Form {
        self.row().map_or(Form::None, |r| r.form)
    }
}

/// The 8-byte slots of a code stream: slot 0 of every group of four is a scheduling word.
pub const SLOTS_PER_BUNDLE: u32 = 4;
pub const SLOT_BYTES: u32 = 8;

/// Is the byte offset the place of a scheduling word?
pub fn is_sched_slot(offset: u32) -> bool {
    (offset / SLOT_BYTES).is_multiple_of(SLOTS_PER_BUNDLE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_come_from_the_documented_bits() {
        // FADD R3, R1, c[2][0x10] with the guard @!P4: 0x4c58 form.
        let w = 0x4c58_0000_0000_0000u64
            | 3
            | 1 << 8
            | (0x10u64 >> 2) << 20
            | 2 << 34
            | 4 << 16
            | 1 << 19;
        let i = Inst(w);
        assert_eq!(i.op(), Some(Op::FaddCbuf));
        assert_eq!(i.rd(), 3);
        assert_eq!(i.ra(), 1);
        assert_eq!(i.cbuf_index(), 2);
        assert_eq!(i.cbuf_offset(), 0x10);
        assert_eq!(
            i.guard(),
            Guard {
                index: 4,
                negated: true
            }
        );
        assert!(!i.guard().is_always());
        assert_eq!(i.form(), Form::Cbuf);
    }

    #[test]
    fn immediates_are_extended_from_their_sign_bits() {
        // IADD with the 20-bit immediate -3: bits 20..38 hold the low 19 bits, bit 56 the sign.
        let neg3 = (-3i32 as u32) & 0xF_FFFF;
        let i = Inst(
            0x3810_0000_0000_0000u64
                | u64::from(neg3 & 0x7_FFFF) << 20
                | u64::from(neg3 >> 19) << 56,
        );
        assert_eq!(i.imm20(), -3);
        assert_eq!(i.op(), Some(Op::IaddImm20));
        // A float immediate keeps the top 20 bits of the f32: 1.0 is 0x3f800000.
        let one = (0x3f80_0000u32 >> 12) as u64;
        let f = Inst(0x3858_0000_0000_0000u64 | (one & 0x7_FFFF) << 20 | (one >> 19) << 56);
        assert_eq!(f.imm20_f32_bits(), 0x3f80_0000);
        // A negative one has the sign in bit 56 (bit 19 of the field).
        let neg_one = (0xbf80_0000u32 >> 12) as u64;
        let g = Inst(0x3858_0000_0000_0000u64 | (neg_one & 0x7_FFFF) << 20 | (neg_one >> 19) << 56);
        assert_eq!(g.imm20_f32_bits(), 0xbf80_0000);
        let c = Inst(0x0100_0000_0000_0000u64 | 0xDEAD_BEEFu64 << 20);
        assert_eq!(c.imm32(), 0xDEAD_BEEF);
    }

    #[test]
    fn branch_offsets_are_signed_24_bit() {
        let back = (-16i32 as u32 & 0xFF_FFFF) as u64;
        let i = Inst(0xe240_0000_0000_000f | back << 20);
        assert_eq!(i.op(), Some(Op::Bra));
        assert_eq!(i.branch_offset(), -16);
    }

    #[test]
    fn every_fourth_slot_is_a_scheduling_word() {
        assert!(is_sched_slot(0));
        assert!(!is_sched_slot(8));
        assert!(!is_sched_slot(24));
        assert!(is_sched_slot(32));
        assert!(!is_sched_slot(40));
    }

    #[test]
    fn the_opcode_table_has_no_unreachable_rows() {
        // A row is shadowed when an earlier row accepts every instruction it accepts: that is
        // when the earlier mask is a subset of this one's bits and the values agree there.
        for (i, later) in TABLE.iter().enumerate() {
            for earlier in &TABLE[..i] {
                let shadows =
                    earlier.mask & !later.mask == 0 && later.value & earlier.mask == earlier.value;
                assert!(
                    !shadows,
                    "{} ({:#x}/{:#x}) can never be chosen: {} ({:#x}/{:#x}) comes first",
                    later.name, later.value, later.mask, earlier.name, earlier.value, earlier.mask
                );
            }
        }
    }

    #[test]
    fn mnemonics_of_known_encodings() {
        let cases: &[(u64, Op)] = &[
            (0xe300_0000_0000_000f, Op::Exit),
            (0xe290_0000_0000_0000, Op::Ssy),
            (0xf0f8_0000_0000_000f, Op::Sync),
            (0x5c68_0000_0000_0000, Op::FmulReg),
            (0x5980_0000_0000_0000, Op::FfmaReg),
            (0x5080_0000_0000_0000, Op::Mufu),
            (0xefd8_0000_0000_0000, Op::Ald),
            (0xeff0_0000_0000_0000, Op::Ast),
            (0xe000_0000_0000_ff00, Op::Ipa),
            (0x0100_0000_0000_0000, Op::Mov32iImm32),
        ];
        for (word, op) in cases {
            assert_eq!(Inst(*word).op(), Some(*op), "{word:#x}");
        }
    }
}
