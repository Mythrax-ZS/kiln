//! Advanced SIMD operations: what the decoder produces and the executor consumes.
//!
//! Operands are described by a [`Shape`]: the element size in bits and the number of elements.
//! A scalar operand has one element. Instructions that write half of their destination (the `2`
//! forms of the narrowing and widening operations) carry an `upper` flag.

use crate::a64::fp::{Fmt, Rounding};

/// Element size in bits and number of elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub esize: u8,
    pub lanes: u8,
}

impl Shape {
    /// A vector arrangement from `size` (log2 of the element size in bytes) and `Q`.
    pub fn vector(size: u32, q: bool) -> Shape {
        let esize = 8u32 << size;
        Shape {
            esize: esize as u8,
            lanes: ((if q { 128 } else { 64 }) / esize) as u8,
        }
    }

    /// One element of `8 << size` bits.
    pub fn scalar(size: u32) -> Shape {
        Shape {
            esize: (8u32 << size) as u8,
            lanes: 1,
        }
    }

    pub fn bits(self) -> u32 {
        u32::from(self.esize) * u32::from(self.lanes)
    }
}

/// `AND`, `BIC`, `ORR`, `ORN`, `EOR` and the bit selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logic {
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    /// Bitwise select: `Vd = (Vd & Vn) | (!Vd & Vm)`.
    Bsl,
    /// Insert if true: `Vd = (Vd & !Vm) | (Vn & Vm)`.
    Bit,
    /// Insert if false: `Vd = (Vd & Vm) | (Vn & !Vm)`.
    Bif,
}

/// Integer operations on corresponding elements of two registers (and the pairwise forms of the
/// last few).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntBin {
    Shadd,
    Uhadd,
    Srhadd,
    Urhadd,
    Shsub,
    Uhsub,
    Sqadd,
    Uqadd,
    Sqsub,
    Uqsub,
    Cmgt,
    Cmhi,
    Cmge,
    Cmhs,
    Cmeq,
    Cmtst,
    Sshl,
    Ushl,
    Srshl,
    Urshl,
    Sqshl,
    Uqshl,
    Sqrshl,
    Uqrshl,
    Smax,
    Umax,
    Smin,
    Umin,
    Sabd,
    Uabd,
    Saba,
    Uaba,
    Add,
    Sub,
    Mul,
    Mla,
    Mls,
    Pmul,
    Sqdmulh,
    Sqrdmulh,
}

/// Operations of the "three different" group (and its by-element form) on 64-bit halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Long {
    /// `SADDL`/`UADDL`: narrow plus narrow, wide result.
    Addl(bool),
    /// `SADDW`/`UADDW`: wide plus narrow.
    Addw(bool),
    Subl(bool),
    Subw(bool),
    /// `ADDHN`/`RADDHN`: wide plus wide, high half, narrow result.
    Addhn(bool),
    Subhn(bool),
    /// `SABAL`/`UABAL`.
    Abal(bool),
    /// `SABDL`/`UABDL`.
    Abdl(bool),
    Mlal(bool),
    Mlsl(bool),
    Mull(bool),
    Sqdmlal,
    Sqdmlsl,
    Sqdmull,
    /// `PMULL`: polynomial multiply, 8 to 16 bits or 64 to 128.
    Pmull,
}

/// Single-source integer operations on every element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntUn {
    Rev64,
    Rev32,
    Rev16,
    Cls,
    Clz,
    Cnt,
    Not,
    Rbit,
    Abs,
    Neg,
    Sqabs,
    Sqneg,
    Cmgt0,
    Cmge0,
    Cmeq0,
    Cmle0,
    Cmlt0,
    Suqadd,
    Usqadd,
    Urecpe,
    Ursqrte,
}

/// Pairwise add long.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairLong {
    Saddlp,
    Uaddlp,
    Sadalp,
    Uadalp,
}

/// Narrowing moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Narrow {
    Xtn,
    Sqxtn,
    Uqxtn,
    Sqxtun,
}

/// Shifts by an immediate that keep the element size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    Sshr,
    Ushr,
    Ssra,
    Usra,
    Srshr,
    Urshr,
    Srsra,
    Ursra,
    Sri,
    Shl,
    Sli,
    Sqshlu,
    Sqshl,
    Uqshl,
}

/// Right shifts by an immediate that narrow the element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftNarrow {
    Shrn,
    Rshrn,
    Sqshrn,
    Uqshrn,
    Sqrshrn,
    Uqrshrn,
    Sqshrun,
    Sqrshrun,
}

/// Reductions across the lanes of one register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Across {
    Saddlv,
    Uaddlv,
    Addv,
    Smaxv,
    Umaxv,
    Sminv,
    Uminv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Permute {
    Uzp1,
    Uzp2,
    Trn1,
    Trn2,
    Zip1,
    Zip2,
}

/// The modified-immediate instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModImm {
    /// `MOVI`, `MVNI` and `FMOV`: the register becomes the (expanded) immediate.
    Set,
    Orr,
    Bic,
}

/// Floating-point operations on corresponding elements of two registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpBin {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MaxNum,
    MinNum,
    Mulx,
    /// `FABD`: the absolute difference.
    Abd,
    Recps,
    Rsqrts,
    /// `FMLA`/`FMLS`: fused multiply-accumulate into the destination.
    Mla,
    Mls,
    Cmeq,
    Cmge,
    Cmgt,
    Acge,
    Acgt,
}

/// Floating-point operations on one source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpUn {
    Abs,
    Neg,
    Sqrt,
    Recpe,
    Rsqrte,
    /// `FRECPX` (scalar only).
    Recpx,
    /// `FRINT*`; a mode of `None` is the one `FPCR` selects.
    Round {
        mode: Option<Rounding>,
        exact: bool,
    },
    Cmgt0,
    Cmge0,
    Cmeq0,
    Cmle0,
    Cmlt0,
}

/// A decoded Advanced SIMD instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdOp {
    Logic {
        op: Logic,
        q: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    IntBin {
        op: IntBin,
        sh: Shape,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `ADDP`, `SMAXP`, `UMAXP`, `SMINP`, `UMINP` over the concatenation `Vm:Vn`.
    IntPair {
        op: IntBin,
        sh: Shape,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `MUL`, `MLA`, `MLS`, `SQDMULH`, `SQRDMULH` by element `Vm.Ts[idx]`.
    IntElem {
        op: IntBin,
        sh: Shape,
        rd: u8,
        rn: u8,
        rm: u8,
        idx: u8,
    },
    /// Three-different operations; `esize` and `lanes` describe the narrow operands.
    Long {
        op: Long,
        sh: Shape,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    LongElem {
        op: Long,
        sh: Shape,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        idx: u8,
    },
    IntUn {
        op: IntUn,
        sh: Shape,
        rd: u8,
        rn: u8,
    },
    PairLong {
        op: PairLong,
        sh: Shape,
        rd: u8,
        rn: u8,
    },
    /// `XTN` and friends; `sh` describes the narrow result.
    Narrow {
        op: Narrow,
        sh: Shape,
        upper: bool,
        rd: u8,
        rn: u8,
    },
    /// `SHLL`: shift left by the element size, widening; `sh` describes the narrow source.
    Shll {
        sh: Shape,
        upper: bool,
        rd: u8,
        rn: u8,
    },
    Shift {
        op: Shift,
        sh: Shape,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    /// Narrowing right shifts; `sh` describes the narrow result.
    ShiftNarrow {
        op: ShiftNarrow,
        sh: Shape,
        upper: bool,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    /// `SSHLL`/`USHLL`; `sh` describes the narrow source.
    ShiftLong {
        signed: bool,
        sh: Shape,
        upper: bool,
        shift: u8,
        rd: u8,
        rn: u8,
    },
    Across {
        op: Across,
        sh: Shape,
        rd: u8,
        rn: u8,
    },
    Permute {
        op: Permute,
        sh: Shape,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Ext {
        q: bool,
        imm4: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `TBL`/`TBX` with `regs` table registers.
    Table {
        q: bool,
        regs: u8,
        extend: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `DUP Vd.T, Vn.Ts[idx]` and `DUP Vd, Vn.Ts[idx]` (scalar, one lane).
    DupElem {
        sh: Shape,
        idx: u8,
        rd: u8,
        rn: u8,
    },
    /// `DUP Vd.T, Wn|Xn`.
    DupGeneral {
        sh: Shape,
        rd: u8,
        rn: u8,
    },
    /// `INS Vd.Ts[idx], Wn|Xn`.
    InsGeneral {
        esize: u8,
        idx: u8,
        rd: u8,
        rn: u8,
    },
    /// `INS Vd.Ts[dst], Vn.Ts[src]`.
    InsElem {
        esize: u8,
        dst: u8,
        src: u8,
        rd: u8,
        rn: u8,
    },
    /// `SMOV`/`UMOV`: an element to a general register.
    MovToGeneral {
        signed: bool,
        esize: u8,
        idx: u8,
        wide: bool,
        rd: u8,
        rn: u8,
    },
    /// `MOVI`, `MVNI`, `ORR`, `BIC`, `FMOV` with the immediate expanded to 64 bits (replicated to
    /// 128 when `q`).
    ModImm {
        op: ModImm,
        q: bool,
        imm: u64,
        rd: u8,
    },

    // ---- floating point ----
    FpBin {
        op: FpBin,
        fmt: Fmt,
        lanes: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `FMLA`, `FMLS`, `FMUL`, `FMULX` by element.
    FpElem {
        op: FpBin,
        fmt: Fmt,
        lanes: u8,
        rd: u8,
        rn: u8,
        rm: u8,
        idx: u8,
    },
    /// Pairwise floating point over `Vm:Vn` (`FADDP`, `FMAXP`, ...).
    FpPair {
        op: FpBin,
        fmt: Fmt,
        lanes: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// Scalar pairwise: `FADDP Sd, Vn.2S` and friends (and `ADDP Dd, Vn.2D` as `IntPairScalar`).
    FpPairScalar {
        op: FpBin,
        fmt: Fmt,
        rd: u8,
        rn: u8,
    },
    AddpScalar {
        rd: u8,
        rn: u8,
    },
    FpUn {
        op: FpUn,
        fmt: Fmt,
        lanes: u8,
        rd: u8,
        rn: u8,
    },
    /// `FCVT{N,M,A,P,Z}{S,U}` to integer, per element.
    FpToInt {
        mode: Rounding,
        signed: bool,
        fmt: Fmt,
        lanes: u8,
        rd: u8,
        rn: u8,
    },
    /// `SCVTF`/`UCVTF` from integer, per element.
    IntToFp {
        signed: bool,
        fmt: Fmt,
        lanes: u8,
        rd: u8,
        rn: u8,
    },
    /// `FCVTZS`/`FCVTZU` (`to_int`) or `SCVTF`/`UCVTF` with fractional bits.
    FpFixed {
        to_int: bool,
        signed: bool,
        fmt: Fmt,
        lanes: u8,
        fbits: u8,
        rd: u8,
        rn: u8,
    },
    /// `FCVTN`/`FCVTN2` and `FCVTXN`/`FCVTXN2`: double to single, single to half. `fmt` is the
    /// narrow destination format.
    FpNarrow {
        fmt: Fmt,
        odd: bool,
        upper: bool,
        scalar: bool,
        rd: u8,
        rn: u8,
    },
    /// `FCVTL`/`FCVTL2`: single to double, half to single. `fmt` is the wide result format.
    FpWiden {
        fmt: Fmt,
        upper: bool,
        rd: u8,
        rn: u8,
    },
    /// `FMAXV` and friends across a 4S register.
    FpAcross {
        op: FpBin,
        rd: u8,
        rn: u8,
    },
}

impl SimdOp {
    /// The general register the instruction reads, if any.
    pub fn int_source(&self) -> Option<u8> {
        match *self {
            SimdOp::DupGeneral { rn, .. } | SimdOp::InsGeneral { rn, .. } => Some(rn),
            _ => None,
        }
    }

    /// The general register the instruction writes with its result, if any.
    pub fn int_dest(&self) -> Option<u8> {
        match *self {
            SimdOp::MovToGeneral { rd, .. } => Some(rd),
            _ => None,
        }
    }
}
