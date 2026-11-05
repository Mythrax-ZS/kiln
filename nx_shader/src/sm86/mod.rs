//! The machine code of the Switch 2's GPU: NVIDIA Ampere (the T239), the instruction set NVN2
//! compiles to (sm_86, docs/ASSUMPTIONS.md A-240).
//!
//! An instruction is 128 bits, the scheduling control in its top bits (105..127: the stall count,
//! the yield flag, the write and read barriers, the wait mask and the operand reuse flags). A
//! graphics program has a 0x80-byte header (version 4, whose first 0x50 bytes keep version 3's
//! layout) and a compute program none; the code ends with an `EXIT` and a branch to itself.
//!
//! The encodings are those of Mesa's NAK encoder (`src/nouveau/compiler/nak/sm70_encode.rs`, MIT
//! licence) and of NeXium's sm_86 frontend (the owner's, MIT); what neither settles, and what
//! NVN2's code relies on, is in docs/ASSUMPTIONS.md.
//!
//! Registers: 255 general registers and `RZ` (255), 63 uniform registers and `URZ` (63), seven
//! predicates and `PT` (7), seven uniform predicates and `UPT`. A uniform instruction (the ones
//! whose 9-bit opcode is 0x080..0x0BF, but `VOTEU`) works on uniform registers and predicates: its
//! guard, its predicate operands and its register fields name them.

pub mod cfg;

use crate::inst::Guard;

/// The zero register.
pub const RZ: u8 = 255;
/// The zero uniform register.
pub const URZ: u8 = 63;
/// The true predicate, and the field value of the true uniform predicate.
pub const PT: u8 = 7;
/// Bytes per instruction.
pub const INST_BYTES: u32 = 16;
/// Where the translation keeps uniform predicate `n`: predicate number `UP_BASE + n` (`UPT` is
/// [`PT`]).
pub const UP_BASE: u8 = 8;
/// The predicate the translation computes a branch's two conditions into (its guard and its own
/// predicate operand) before the branch takes effect.
pub const BRANCH_PRED: u8 = 16;
/// Predicate slots the translation uses (general, uniform and its own).
pub const PRED_SLOTS: usize = 24;

/// Opcodes (bits 0..11; the ALU ones have their operand form in bits 9..11, see [`Inst::form`]).
pub mod op {
    pub const BRA: u16 = 0x947;
    pub const BRX: u16 = 0x949;
    pub const EXIT: u16 = 0x94d;
    pub const CALL: u16 = 0x944;
    pub const RET: u16 = 0x950;
    pub const BSSY: u16 = 0x945;
    pub const BSYNC: u16 = 0x941;
    pub const BREAK: u16 = 0x942;
    pub const KILL: u16 = 0x95b;
    pub const NOP: u16 = 0x918;
    pub const DEPBAR: u16 = 0x91a;
    pub const S2R: u16 = 0x919;
    pub const CS2R: u16 = 0x805;
    pub const S2UR: u16 = 0x9c3;
    pub const LDC: u16 = 0xb82;
    pub const ULDC: u16 = 0xab9;
    pub const ULDC_INDEXED: u16 = 0xabb;
    pub const ALD: u16 = 0x321;
    pub const AST: u16 = 0x322;
    pub const IPA: u16 = 0x326;
    pub const ISBERD: u16 = 0x923;
    pub const ISBEWR: u16 = 0x927;
    pub const OUT: u16 = 0x124;
    pub const LDG: u16 = 0x381;
    pub const LDG_U: u16 = 0x981;
    pub const STG: u16 = 0x386;
    pub const STG_U: u16 = 0x986;
    pub const LDL: u16 = 0x983;
    pub const STL: u16 = 0x387;
    pub const STL_U: u16 = 0x987;
    pub const LDS: u16 = 0x984;
    pub const STS: u16 = 0x388;
    pub const STS_U: u16 = 0x988;
    pub const ATOMG: u16 = 0x3a8;
    pub const ATOMG_U: u16 = 0x9a8;
    pub const ATOMS: u16 = 0x38c;
    pub const RED: u16 = 0x98e;
    pub const BAR: u16 = 0xb1d;
    pub const MEMBAR: u16 = 0x992;
    pub const ERRBAR: u16 = 0x9ab;
    pub const TEX: u16 = 0xb60;
    pub const TEX_B: u16 = 0x361;
    pub const TLD: u16 = 0xb66;
    pub const TLD_B: u16 = 0x367;
    pub const TLD4: u16 = 0xb63;
    pub const TLD4_B: u16 = 0x364;
    pub const TMML: u16 = 0xb69;
    pub const TMML_B: u16 = 0x36a;
    pub const TXD: u16 = 0xb6c;
    pub const TXQ: u16 = 0xb6f;
    pub const TXQ_B: u16 = 0x370;
    pub const SULD: u16 = 0x998;
    pub const SUST: u16 = 0x99c;
    /// The bound forms (the handle in a constant buffer, as the texture instructions' bound forms).
    pub const SULD_BOUND: u16 = 0xb97;
    pub const SUST_BOUND: u16 = 0xb9b;
    pub const SURED: u16 = 0x59f;
    pub const SUATOM: u16 = 0x593;
    pub const FSWZADD: u16 = 0x822;
    pub const MATCH: u16 = 0x3a1;
    pub const R2UR: u16 = 0x3c2;
    pub const REDUX: u16 = 0x3c4;
    pub const SHFL: u16 = 0x189;
    pub const PLOP3: u16 = 0x81c;
    pub const UPLOP3: u16 = 0x89c;

    /// The 9-bit opcodes of the ALU instructions.
    pub mod alu {
        pub const MOV: u16 = 0x002;
        pub const P2R: u16 = 0x003;
        pub const R2P: u16 = 0x004;
        pub const VOTE: u16 = 0x006;
        pub const SEL: u16 = 0x007;
        pub const FSEL: u16 = 0x008;
        pub const FMNMX: u16 = 0x009;
        pub const FSET: u16 = 0x00a;
        pub const FSETP: u16 = 0x00b;
        pub const ISETP: u16 = 0x00c;
        pub const IADD3: u16 = 0x010;
        pub const LEA: u16 = 0x011;
        pub const LOP3: u16 = 0x012;
        pub const IABS: u16 = 0x013;
        pub const PRMT: u16 = 0x016;
        pub const IMNMX: u16 = 0x017;
        pub const SHF: u16 = 0x019;
        pub const SGXT: u16 = 0x01a;
        pub const BMSK: u16 = 0x01b;
        pub const FMUL: u16 = 0x020;
        pub const FADD: u16 = 0x021;
        pub const FFMA: u16 = 0x023;
        pub const IMAD: u16 = 0x024;
        pub const IMAD_WIDE: u16 = 0x025;
        pub const IMAD_HI: u16 = 0x027;
        pub const HADD2: u16 = 0x030;
        pub const HFMA2: u16 = 0x031;
        pub const HMUL2: u16 = 0x032;
        pub const HSET2: u16 = 0x033;
        pub const HSETP2: u16 = 0x034;
        pub const F2FP: u16 = 0x03e;
        pub const HMNMX2: u16 = 0x040;
        pub const I2FP: u16 = 0x045;
        pub const FLO: u16 = 0x100;
        pub const BREV: u16 = 0x101;
        pub const F2F: u16 = 0x104;
        pub const F2I: u16 = 0x105;
        pub const I2F: u16 = 0x106;
        pub const FRND: u16 = 0x107;
        pub const MUFU: u16 = 0x108;
        pub const POPC: u16 = 0x109;
        pub const WARPSYNC: u16 = 0x148;
        pub const SHFL: u16 = 0x189;
        pub const R2UR: u16 = 0x1c2;
        pub const REDUX: u16 = 0x1c4;
        // Uniform: the vector opcode plus 0x80, with these exceptions.
        pub const UMOV: u16 = 0x082;
        pub const UP2UR: u16 = 0x083;
        pub const VOTEU: u16 = 0x086;
        pub const ULDC: u16 = 0x0b9;
        pub const UF2FP: u16 = 0x0ba;
        pub const ULDC_INDEXED: u16 = 0x0bb;
        pub const UFLO: u16 = 0x0bd;
        pub const UBREV: u16 = 0x0be;
        pub const UPOPC: u16 = 0x0bf;
    }
}

/// One instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inst(pub u128);

/// Where an ALU operand comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Reg(u8),
    UReg(u8),
    Imm(u32),
    /// A word of a constant buffer: its slot and byte offset.
    Cbuf {
        slot: u32,
        offset: u32,
    },
    /// A word of the constant buffer whose slot a uniform register holds (bindless).
    CbufUReg {
        ureg: u8,
        offset: u32,
    },
}

/// An ALU operand and its modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    pub operand: Operand,
    pub neg: bool,
    pub abs: bool,
}

impl Inst {
    /// The instruction at byte offset `pc` of `code`.
    pub fn at(code: &[u8], pc: u32) -> Option<Inst> {
        let at = pc as usize;
        let b = code.get(at..at + INST_BYTES as usize)?;
        Some(Inst(u128::from_le_bytes(b.try_into().ok()?)))
    }

    /// `len` bits (at most 64) from bit `lo`.
    pub fn field(self, lo: u32, len: u32) -> u64 {
        debug_assert!((1..=64).contains(&len) && lo + len <= 128);
        let v = self.0 >> lo;
        if len == 64 {
            v as u64
        } else {
            (v as u64) & ((1u64 << len) - 1)
        }
    }

    pub fn bit(self, n: u32) -> bool {
        self.0 >> n & 1 != 0
    }

    /// `len` bits from bit `lo`, sign-extended.
    pub fn sfield(self, lo: u32, len: u32) -> i64 {
        let v = self.field(lo, len);
        let shift = 64 - len;
        ((v << shift) as i64) >> shift
    }

    /// The 12-bit opcode.
    pub fn opcode(self) -> u16 {
        self.field(0, 12) as u16
    }

    /// The 9-bit opcode of an ALU instruction.
    pub fn base(self) -> u16 {
        self.field(0, 9) as u16
    }

    /// The operand form of an ALU instruction (bits 9..11): where its second and third operands
    /// come from (NAK's `encode_alu_base`).
    pub fn form(self) -> u8 {
        self.field(9, 3) as u8
    }

    /// Does the instruction run on the uniform datapath (its registers and predicates are the
    /// uniform ones)?
    pub fn is_uniform(self) -> bool {
        let b = self.base();
        (0x080..0x0c0).contains(&b) && b != op::alu::VOTEU
    }

    /// A predicate field: three bits at `at` and its negation bit; uniform when `uniform`.
    pub fn pred(self, at: u32, not_at: u32, uniform: bool) -> Guard {
        let index = self.field(at, 3) as u8;
        Guard {
            index: match index {
                PT => PT,
                i if uniform => UP_BASE + i,
                i => i,
            },
            negated: self.bit(not_at),
        }
    }

    /// The predicate a destination field names (`PT` for none).
    pub fn pred_dst(self, at: u32, uniform: bool) -> u8 {
        match self.field(at, 3) as u8 {
            PT => PT,
            i if uniform => UP_BASE + i,
            i => i,
        }
    }

    /// The guard (bits 12..15).
    pub fn guard(self) -> Guard {
        self.pred(12, 15, self.is_uniform())
    }

    /// A general register field (8 bits).
    pub fn reg(self, at: u32) -> u8 {
        self.field(at, 8) as u8
    }

    /// A uniform register field (6 bits).
    pub fn ureg(self, at: u32) -> u8 {
        self.field(at, 6) as u8
    }

    /// The destination register (bits 16..23; a uniform one, 6 bits, for a uniform instruction).
    pub fn dst(self) -> u8 {
        self.reg(16)
    }

    /// The first ALU operand (bits 24..31, negate 72, absolute 73).
    pub fn src_a(self) -> Source {
        Source {
            operand: if self.is_uniform() {
                Operand::UReg(self.ureg(24))
            } else {
                Operand::Reg(self.reg(24))
            },
            neg: self.bit(72),
            abs: self.bit(73),
        }
    }

    /// The operand in bits 32..63 (absolute 62, negate 63): a register, a uniform register, a
    /// 32-bit immediate or a constant buffer word, as the form says. `None` for a form that has
    /// none there.
    pub fn slot32(self) -> Option<Source> {
        let uniform = self.is_uniform();
        let operand = match self.form() {
            1 if uniform => Operand::UReg(self.ureg(32)),
            1 => Operand::Reg(self.reg(32)),
            2 | 4 => {
                return Some(Source {
                    operand: Operand::Imm(self.field(32, 32) as u32),
                    neg: false,
                    abs: false,
                })
            }
            3 | 5 if self.bit(91) => Operand::CbufUReg {
                ureg: self.ureg(32),
                offset: self.field(38, 16) as u32,
            },
            3 | 5 => Operand::Cbuf {
                slot: self.field(54, 5) as u32,
                offset: self.field(38, 16) as u32,
            },
            6 | 7 => Operand::UReg(self.ureg(32)),
            _ => return None,
        };
        Some(Source {
            operand,
            neg: self.bit(63),
            abs: self.bit(62),
        })
    }

    /// The operand in bits 64..71 (absolute 74, negate 75): a register, or a uniform one for a
    /// uniform instruction.
    pub fn slot64(self) -> Source {
        Source {
            operand: if self.is_uniform() {
                Operand::UReg(self.ureg(64))
            } else {
                Operand::Reg(self.reg(64))
            },
            neg: self.bit(75),
            abs: self.bit(74),
        }
    }

    /// The second and third ALU operands: forms 1, 4, 5 and 6 have the second in bits 32..63 and
    /// the third in 64..71; forms 2, 3 and 7 the other way round.
    pub fn src_bc(self) -> Option<(Source, Source)> {
        let wide = self.slot32()?;
        Some(match self.form() {
            2 | 3 | 7 => (self.slot64(), wide),
            _ => (wide, self.slot64()),
        })
    }

    /// The target of a relative branch at `pc`: the signed word offset in bits 34..81 counts from
    /// the next instruction (NAK's `set_rel_offset`).
    pub fn branch_target(self, pc: u32) -> i64 {
        i64::from(pc) + i64::from(INST_BYTES) + self.sfield(34, 48) * 4
    }

    /// The branch to itself that follows the last `EXIT` and marks the end of the code.
    pub fn is_end(self, pc: u32) -> bool {
        self.opcode() == op::BRA
            && self.guard().is_always()
            && self.pred(87, 90, false).is_always()
            && self.branch_target(pc) == i64::from(pc)
    }
}

/// The bytes of a program's code: everything from its first instruction to the end marker (the
/// branch to itself after the last `EXIT`) included. `None` when the bytes end first.
pub fn code_extent(code: &[u8]) -> Option<u32> {
    let mut pc = 0;
    while let Some(inst) = Inst::at(code, pc) {
        pc += INST_BYTES;
        if inst.is_end(pc - INST_BYTES) {
            return Some(pc);
        }
    }
    None
}

/// The name of an opcode, for messages.
pub fn name(inst: Inst) -> String {
    use op::alu;
    let full = match inst.opcode() {
        op::BRA => "BRA",
        op::BRX => "BRX",
        op::EXIT => "EXIT",
        op::CALL => "CALL",
        op::RET => "RET",
        op::BSSY => "BSSY",
        op::BSYNC => "BSYNC",
        op::BREAK => "BREAK",
        op::KILL => "KILL",
        op::NOP => "NOP",
        op::DEPBAR => "DEPBAR",
        op::S2R => "S2R",
        op::CS2R => "CS2R",
        op::S2UR => "S2UR",
        op::LDC => "LDC",
        op::ULDC => "ULDC",
        op::ULDC_INDEXED => "ULDC (indexed)",
        op::ALD => "ALD",
        op::AST => "AST",
        op::IPA => "IPA",
        op::ISBERD => "ISBERD",
        op::ISBEWR => "ISBEWR",
        op::OUT => "OUT",
        op::LDG => "LDG",
        op::LDG_U => "LDG (uniform base)",
        op::STG => "STG",
        op::STG_U => "STG (uniform base)",
        op::LDL => "LDL",
        op::STL | op::STL_U => "STL",
        op::LDS => "LDS",
        op::STS | op::STS_U => "STS",
        op::ATOMG | op::ATOMG_U => "ATOMG",
        op::ATOMS => "ATOMS",
        op::RED => "RED",
        op::BAR => "BAR",
        op::MEMBAR => "MEMBAR",
        op::ERRBAR => "ERRBAR",
        op::TEX | op::TEX_B => "TEX",
        op::TLD | op::TLD_B => "TLD",
        op::TLD4 | op::TLD4_B => "TLD4",
        op::TMML | op::TMML_B => "TMML",
        op::TXD => "TXD",
        op::TXQ | op::TXQ_B => "TXQ",
        op::SULD | op::SULD_BOUND => "SULD",
        op::SUST | op::SUST_BOUND => "SUST",
        op::SURED => "SURED",
        op::SUATOM => "SUATOM",
        op::FSWZADD => "FSWZADD",
        op::MATCH => "MATCH",
        op::REDUX => "REDUX",
        op::PLOP3 => "PLOP3",
        op::UPLOP3 => "UPLOP3",
        _ => "",
    };
    if !full.is_empty() {
        return full.to_string();
    }
    let base = match inst.base() {
        alu::MOV => "MOV",
        alu::P2R => "P2R",
        alu::R2P => "R2P",
        alu::VOTE => "VOTE",
        alu::SEL => "SEL",
        alu::FSEL => "FSEL",
        alu::FMNMX => "FMNMX",
        alu::FSET => "FSET",
        alu::FSETP => "FSETP",
        alu::ISETP => "ISETP",
        alu::IADD3 => "IADD3",
        alu::LEA => "LEA",
        alu::LOP3 => "LOP3",
        alu::IABS => "IABS",
        alu::PRMT => "PRMT",
        alu::IMNMX => "IMNMX",
        alu::SHF => "SHF",
        alu::SGXT => "SGXT",
        alu::BMSK => "BMSK",
        alu::FMUL => "FMUL",
        alu::FADD => "FADD",
        alu::FFMA => "FFMA",
        alu::IMAD => "IMAD",
        alu::IMAD_WIDE => "IMAD.WIDE",
        alu::IMAD_HI => "IMAD.HI",
        alu::HADD2 => "HADD2",
        alu::HFMA2 => "HFMA2",
        alu::HMUL2 => "HMUL2",
        alu::HSET2 => "HSET2",
        alu::HSETP2 => "HSETP2",
        alu::F2FP => "F2FP",
        alu::HMNMX2 => "HMNMX2",
        alu::I2FP => "I2FP",
        alu::FLO => "FLO",
        alu::BREV => "BREV",
        alu::F2F => "F2F",
        alu::F2I => "F2I",
        alu::I2F => "I2F",
        alu::FRND => "FRND",
        alu::MUFU => "MUFU",
        alu::POPC => "POPC",
        alu::WARPSYNC => "WARPSYNC",
        alu::R2UR => "R2UR",
        alu::REDUX => "REDUX",
        alu::UMOV => "UMOV",
        alu::UP2UR => "UP2UR",
        alu::VOTEU => "VOTEU",
        alu::ULDC => "ULDC",
        alu::UF2FP => "UF2FP",
        alu::UFLO => "UFLO",
        alu::UBREV => "UBREV",
        alu::UPOPC => "UPOPC",
        b if inst.is_uniform() => return format!("uniform opcode {b:#05x}"),
        _ => return format!("opcode {:#05x}", inst.opcode()),
    };
    if inst.is_uniform() && !base.starts_with('U') && base != "VOTEU" {
        format!("U{base}")
    } else {
        base.to_string()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A word with `fields` (lo, len, value) set.
    pub(crate) fn word(fields: &[(u32, u32, u64)]) -> Inst {
        let mut v = 0u128;
        for &(lo, len, value) in fields {
            let mask = ((1u128 << len) - 1) << lo;
            v = (v & !mask) | ((u128::from(value) << lo) & mask);
        }
        Inst(v)
    }

    #[test]
    fn fields_decode_as_nak_encodes_them() {
        // FFMA R1, R2, c[0x3][0x10], R4 under !P2 (form 5: the constant buffer is the second
        // operand).
        let i = word(&[
            (0, 9, 0x023),
            (9, 3, 5),
            (12, 3, 2),
            (15, 1, 1),
            (16, 8, 1),
            (24, 8, 2),
            (38, 16, 0x10),
            (54, 5, 3),
            (64, 8, 4),
        ]);
        assert_eq!(i.base(), op::alu::FFMA);
        assert_eq!(
            i.guard(),
            Guard {
                index: 2,
                negated: true
            }
        );
        let (b, c) = i.src_bc().unwrap();
        assert_eq!(
            b.operand,
            Operand::Cbuf {
                slot: 3,
                offset: 0x10
            }
        );
        assert_eq!(c.operand, Operand::Reg(4));
        // Form 2: the immediate is the third operand, the register at 64 the second.
        let j = word(&[(0, 9, 0x023), (9, 3, 2), (32, 32, 0x3f80_0000), (64, 8, 9)]);
        let (b, c) = j.src_bc().unwrap();
        assert_eq!(
            (b.operand, c.operand),
            (Operand::Reg(9), Operand::Imm(0x3f80_0000))
        );
    }

    #[test]
    fn uniform_instructions_name_uniform_registers_and_predicates() {
        // UIADD3 UR4, UR5, UR6, URZ under UP1.
        let i = word(&[
            (0, 9, 0x090),
            (9, 3, 1),
            (12, 3, 1),
            (16, 6, 4),
            (24, 6, 5),
            (32, 6, 6),
            (64, 6, 63),
            (91, 1, 1),
        ]);
        assert!(i.is_uniform());
        assert_eq!(i.guard().index, UP_BASE + 1);
        assert_eq!(i.src_a().operand, Operand::UReg(5));
        let (b, c) = i.src_bc().unwrap();
        assert_eq!(
            (b.operand, c.operand),
            (Operand::UReg(6), Operand::UReg(URZ))
        );
        // VOTEU's guard is a general predicate.
        assert!(!word(&[(0, 12, 0x886)]).is_uniform());
    }

    #[test]
    fn the_code_ends_at_the_branch_to_itself() {
        let exit = word(&[(0, 12, 0x94d), (12, 3, 7), (87, 3, 7)]);
        let end = word(&[
            (0, 12, 0x947),
            (12, 3, 7),
            (87, 3, 7),
            (34, 48, (-4i64) as u64),
        ]);
        let mut code = Vec::new();
        for i in [exit, end, exit] {
            code.extend_from_slice(&i.0.to_le_bytes());
        }
        assert_eq!(code_extent(&code), Some(32));
        assert!(end.is_end(16));
        assert_eq!(code_extent(&code[..16]), None);
    }
}
