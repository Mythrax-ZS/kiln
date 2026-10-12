//! Scalar floating point and Advanced SIMD.
//!
//! These instructions are decoded into [`VOp`] (computation on the vector registers) and [`VMem`]
//! (transfers between them and memory). Both are executed by [`exec`], whose semantics are written
//! once and used by the interpreter and, through a runtime call, by generated code. The integer
//! register file is touched only through the three values `exec` takes and returns, so the lifter
//! can keep treating `X` registers as SSA values:
//!
//! - `src`: the value of the integer source register, if the instruction has one (`FMOV Dd, Xn`,
//!   `SCVTF`, `DUP Vd, Wn`, ...);
//! - the return value: the integer result (`FMOV Xd, Dn`, `FCVTZS`, `UMOV`, ...) or, for the
//!   compare instructions, the new `NZCV` nibble;
//! - `nzcv`: the current flags, for the instructions that test a condition (`FCSEL`, `FCCMP`).

pub mod decode;
pub mod exec;
mod lanes;

use super::fp::{Fmt, MinMax, Rounding};
use super::inst::{AddrMode, Reg};

/// Why an encoding is not executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    /// The architecture leaves the encoding unallocated.
    Unallocated,
    /// A valid encoding this project does not implement.
    Unsupported(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpUnary {
    Abs,
    Neg,
    Sqrt,
    /// `FRINT*`; `mode` of `None` is the current `FPCR` rounding mode.
    Round {
        mode: Option<Rounding>,
        exact: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpBinary {
    Mul,
    Div,
    Add,
    Sub,
    MinMax(MinMax),
    /// `FNMUL`: the negated product.
    Nmul,
}

/// A computation on the vector registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VOp {
    // ---- scalar floating point ----
    /// `FMOV Vd, Vn`: copy a scalar, clearing the rest of the destination.
    FpMov {
        fmt: Fmt,
        rd: u8,
        rn: u8,
    },
    FpUnary {
        op: FpUnary,
        fmt: Fmt,
        rd: u8,
        rn: u8,
    },
    /// `FCVT` between precisions.
    FpConvert {
        from: Fmt,
        to: Fmt,
        rd: u8,
        rn: u8,
    },
    FpBinary {
        op: FpBinary,
        fmt: Fmt,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `FMADD`, `FMSUB`, `FNMADD`, `FNMSUB`: `±ra ± rn * rm`.
    FpFused {
        neg_addend: bool,
        neg_product: bool,
        fmt: Fmt,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// `FCMP`/`FCMPE`; `rm` of `None` compares with zero. Result: the `NZCV` nibble.
    FpCompare {
        fmt: Fmt,
        rn: u8,
        rm: Option<u8>,
        signal: bool,
    },
    /// `FCCMP`/`FCCMPE`: tests `nzcv` against `cond`.
    FpCondCompare {
        fmt: Fmt,
        rn: u8,
        rm: u8,
        cond: u8,
        nzcv: u8,
        signal: bool,
    },
    FpSelect {
        fmt: Fmt,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: u8,
    },
    FpImmediate {
        fmt: Fmt,
        rd: u8,
        imm8: u8,
    },
    /// `FCVT{N,A,P,M,Z}{S,U}`, scalar to a general register (or fixed point with `fbits`).
    FpToInt {
        fmt: Fmt,
        /// 64-bit destination.
        wide: bool,
        signed: bool,
        mode: Rounding,
        fbits: u32,
        rn: u8,
        rd: Reg,
    },
    /// `SCVTF`/`UCVTF` from a general register.
    IntToFp {
        fmt: Fmt,
        wide: bool,
        signed: bool,
        fbits: u32,
        rd: u8,
        rn: Reg,
    },
    /// `FMOV Wd, Sn` / `FMOV Xd, Dn` / `FMOV Xd, Vn.D[1]`.
    FmovToGeneral {
        wide: bool,
        high: bool,
        rn: u8,
        rd: Reg,
    },
    /// `FMOV Sd, Wn` / `FMOV Dd, Xn` / `FMOV Vd.D[1], Xn`.
    FmovFromGeneral {
        wide: bool,
        high: bool,
        rd: u8,
        rn: Reg,
    },
}

impl VOp {
    /// The general register this operation reads, if any.
    pub fn int_source(&self) -> Option<Reg> {
        match *self {
            VOp::IntToFp { rn, .. } | VOp::FmovFromGeneral { rn, .. } => Some(rn),
            _ => None,
        }
    }

    /// The general register this operation writes with the value `exec` returns, if any.
    pub fn int_dest(&self) -> Option<Reg> {
        match *self {
            VOp::FpToInt { rd, .. } | VOp::FmovToGeneral { rd, .. } => Some(rd),
            _ => None,
        }
    }

    /// Does the operation test the condition flags?
    pub fn reads_nzcv(&self) -> bool {
        matches!(self, VOp::FpCondCompare { .. } | VOp::FpSelect { .. })
    }

    /// Does the value `exec` returns replace `NZCV`?
    pub fn writes_nzcv(&self) -> bool {
        matches!(self, VOp::FpCompare { .. } | VOp::FpCondCompare { .. })
    }
}

/// A transfer between the vector registers and memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VMem {
    /// One register of `bytes` (1, 2, 4, 8 or 16) bytes.
    Single { load: bool, bytes: u8, rt: u8 },
    /// Two registers of `bytes` (4, 8 or 16) bytes each, at consecutive addresses.
    Pair {
        load: bool,
        bytes: u8,
        rt: u8,
        rt2: u8,
    },
}

/// How a vector load or store forms its address, for the interpreter and the lifter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VAddr {
    pub rn: Reg,
    pub mode: AddrMode,
}

#[cfg(test)]
mod tests;
