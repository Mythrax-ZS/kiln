//! Decoded A64 instructions.
//!
//! One `Inst` variant per instruction *class*, with the operand fields the Arm pseudocode uses.
//! The interpreter and the lifter both work from this type, so they cannot disagree about what an
//! encoding means. Register fields hold the raw 5-bit number; whether 31 means SP or ZR is a
//! property of the operand and is stated on each variant.

use super::vec::{VAddr, VMem, VOp};

pub type Reg = u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Orr,
    Eor,
    Ands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MovOp {
    Movn,
    Movz,
    Movk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitfieldOp {
    Sbfm,
    Bfm,
    Ubfm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtendType {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

/// Data-processing (1 source).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp1Op {
    Rbit,
    Rev16,
    /// 32-bit byte reverse inside each word of a 64-bit register (`REV32`).
    Rev32,
    /// Full-width byte reverse (`REV`).
    Rev,
    Clz,
    Cls,
}

/// Data-processing (2 source).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp2Op {
    Udiv,
    Sdiv,
    Lslv,
    Lsrv,
    Asrv,
    Rorv,
    /// CRC32 of `1 << size` bytes; `c` selects the Castagnoli polynomial.
    Crc32 {
        size: u8,
        c: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp3Op {
    Madd,
    Msub,
    Smaddl,
    Smsubl,
    Smulh,
    Umaddl,
    Umsubl,
    Umulh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CondSelOp {
    Csel,
    Csinc,
    Csinv,
    Csneg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchReg {
    Br,
    Blr,
    Ret,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierKind {
    Dsb,
    Dmb,
    Isb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    Store,
    Load,
    /// Sign-extending load whose destination is a W register (upper half zeroed).
    LoadSigned32,
    /// Sign-extending load whose destination is an X register.
    LoadSigned64,
    /// `PRFM` and friends: no architectural effect we model.
    Prefetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddrMode {
    /// `[Xn, #imm]`; Xn may be SP.
    Offset(i64),
    /// `[Xn, #imm]!`
    PreIndex(i64),
    /// `[Xn], #imm`
    PostIndex(i64),
    /// `[Xn], Xm` (the Advanced SIMD structure loads and stores); Xm is never XZR here.
    PostIndexReg(Reg),
    /// `[Xn, Xm{, extend {#amount}}]`; Xm = 31 is XZR.
    Register {
        rm: Reg,
        extend: ExtendType,
        amount: u8,
    },
    /// PC-relative literal, byte offset from the instruction.
    Literal(i64),
}

/// System registers the EL0 model knows by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SysReg {
    Nzcv,
    Fpcr,
    Fpsr,
    TpidrEl0,
    TpidrroEl0,
    CntfrqEl0,
    CntpctEl0,
    CntvctEl0,
    CtrEl0,
    DczidEl0,
    /// Any other encoding, kept as `(op0, op1, CRn, CRm, op2)`.
    Other(u8, u8, u8, u8, u8),
}

impl SysReg {
    /// The `(op0, op1, CRn, CRm, op2)` encoding of the register.
    pub fn encoding(self) -> [u8; 5] {
        match self {
            SysReg::Nzcv => [3, 3, 4, 2, 0],
            SysReg::Fpcr => [3, 3, 4, 4, 0],
            SysReg::Fpsr => [3, 3, 4, 4, 1],
            SysReg::TpidrEl0 => [3, 3, 13, 0, 2],
            SysReg::TpidrroEl0 => [3, 3, 13, 0, 3],
            SysReg::CntfrqEl0 => [3, 3, 14, 0, 0],
            SysReg::CntpctEl0 => [3, 3, 14, 0, 1],
            SysReg::CntvctEl0 => [3, 3, 14, 0, 2],
            SysReg::CtrEl0 => [3, 3, 0, 0, 1],
            SysReg::DczidEl0 => [3, 3, 0, 0, 7],
            SysReg::Other(a, b, c, d, e) => [a, b, c, d, e],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inst {
    // ---- data processing, immediate ----
    /// ADR / ADRP. `imm` is the byte offset; for ADRP it is already scaled by 4096 and applies
    /// to the page of the PC.
    Adr {
        rd: Reg,
        imm: i64,
        page: bool,
    },
    /// ADD/SUB(S) immediate. `imm` already includes the optional shift by 12. Rn is SP-capable;
    /// Rd is SP-capable unless `set_flags`, then it is ZR.
    AddSubImm {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        imm: u32,
    },
    /// AND/ORR/EOR/ANDS immediate with the decoded bitmask. Rn is ZR; Rd is SP-capable unless ANDS.
    LogicalImm {
        sf: bool,
        op: LogicOp,
        rd: Reg,
        rn: Reg,
        imm: u64,
    },
    /// MOVN/MOVZ/MOVK. `shift` is 0, 16, 32 or 48.
    MovWide {
        sf: bool,
        op: MovOp,
        rd: Reg,
        imm16: u16,
        shift: u8,
    },
    Bitfield {
        sf: bool,
        op: BitfieldOp,
        rd: Reg,
        rn: Reg,
        immr: u8,
        imms: u8,
    },
    /// EXTR (and ROR immediate when `rn == rm`).
    Extr {
        sf: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        lsb: u8,
    },

    // ---- branches, exceptions, system ----
    /// B / BL with a byte offset.
    B {
        link: bool,
        imm: i64,
    },
    BCond {
        cond: u8,
        imm: i64,
    },
    Cbz {
        sf: bool,
        nonzero: bool,
        rt: Reg,
        imm: i64,
    },
    Tbz {
        nonzero: bool,
        rt: Reg,
        bit: u8,
        imm: i64,
    },
    /// BR / BLR / RET through a register (31 is XZR).
    BranchReg {
        kind: BranchReg,
        rn: Reg,
    },
    Svc {
        imm: u16,
    },
    Brk {
        imm: u16,
    },
    /// HINT #imm, `imm = CRm:op2`. All hints execute as NOPs on a core without the extension.
    Hint {
        imm: u8,
    },
    Barrier {
        kind: BarrierKind,
        crm: u8,
    },
    Clrex,
    Mrs {
        rt: Reg,
        reg: SysReg,
    },
    Msr {
        rt: Reg,
        reg: SysReg,
    },
    /// SYS / SYSL (cache maintenance and friends).
    Sys {
        load: bool,
        op1: u8,
        crn: u8,
        crm: u8,
        op2: u8,
        rt: Reg,
    },

    // ---- loads and stores ----
    /// Single register load/store. `size` is log2 of the access size in bytes (0 to 3).
    LdSt {
        kind: MemKind,
        size: u8,
        rt: Reg,
        rn: Reg,
        mode: AddrMode,
    },
    /// LDP/STP/LDNP/STNP/LDPSW. `size` is log2 of the size of each register (2 or 3).
    LdStPair {
        load: bool,
        signed_word: bool,
        size: u8,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
        imm: i64,
        writeback: bool,
        post_index: bool,
    },
    /// LDXR/LDAXR/STXR/STLXR and the pair forms LDXP/LDAXP/STXP/STLXP.
    Exclusive {
        load: bool,
        pair: bool,
        size: u8,
        acquire: bool,
        release: bool,
        rs: Reg,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
    },
    /// LDAR / STLR (load-acquire, store-release; alignment checked).
    AcqRel {
        load: bool,
        size: u8,
        rt: Reg,
        rn: Reg,
    },

    // ---- data processing, register ----
    /// AND/ORR/EOR/ANDS with an optionally inverted shifted register (BIC/ORN/EON/BICS).
    LogicalShifted {
        sf: bool,
        op: LogicOp,
        invert: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: ShiftType,
        amount: u8,
    },
    AddSubShifted {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: ShiftType,
        amount: u8,
    },
    /// Rn and Rd are SP-capable (Rd only when not setting flags); Rm is ZR.
    AddSubExtended {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        extend: ExtendType,
        amount: u8,
    },
    AddSubCarry {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// CCMN/CCMP. When `imm_operand` the second operand is the 5-bit immediate in `rm_or_imm`.
    CondCmp {
        sf: bool,
        negative: bool,
        imm_operand: bool,
        rn: Reg,
        rm_or_imm: u8,
        cond: u8,
        nzcv: u8,
    },
    CondSel {
        sf: bool,
        op: CondSelOp,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        cond: u8,
    },
    DataProc1 {
        sf: bool,
        op: Dp1Op,
        rd: Reg,
        rn: Reg,
    },
    DataProc2 {
        sf: bool,
        op: Dp2Op,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    DataProc3 {
        sf: bool,
        op: Dp3Op,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        ra: Reg,
    },

    // ---- floating point and Advanced SIMD ----
    /// A computation on the vector registers (see `a64::vec`). `word` is the encoding, which
    /// generated code passes to the runtime.
    Vec {
        word: u32,
        op: VOp,
    },
    /// A load or store of vector registers.
    VecMem {
        word: u32,
        mem: VMem,
        addr: VAddr,
    },

    // ---- not executable ----
    /// An encoding the architecture leaves unallocated or reserved: executing it is undefined.
    Unallocated(u32),
    /// A valid encoding outside what this project implements (feature extensions, FP/SIMD before
    /// milestone M5, privileged instructions). Executing it stops the run; it is never ignored.
    Unsupported {
        word: u32,
        what: &'static str,
    },
}

impl Inst {
    /// Does this instruction end straight-line execution (unconditionally or conditionally)?
    pub fn is_branch(&self) -> bool {
        matches!(
            self,
            Inst::B { .. }
                | Inst::BCond { .. }
                | Inst::Cbz { .. }
                | Inst::Tbz { .. }
                | Inst::BranchReg { .. }
        )
    }
}
