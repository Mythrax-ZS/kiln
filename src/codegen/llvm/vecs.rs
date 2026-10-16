//! Floating-point and vector instructions lowered to LLVM IR.
//!
//! The runtime can execute every one of them (`nx_vec`, `nx_vecmem`: decode the word, run the
//! soft-float or the lane code of the interpreter), at tens of nanoseconds a call. The forms a
//! program runs most are done here instead, in a few instructions on the vector registers in the
//! CPU structure:
//!
//! - register moves, the bitwise operations (`mov v.16b`, `eor`, `bsl`, ...), `FMOV`, `FABS`,
//!   `FNEG`, `FCSEL`, the immediate and general-register forms of `FMOV`: exact, there is nothing
//!   to get wrong;
//! - the single-precision `FADD`, `FSUB`, `FMUL`, `FDIV` and `FCMP`: when `FPCR` is the default
//!   (or has only flush-to-zero, default NaN and alternative half precision set, and then no
//!   operand is a denormal), the operands are ordinary and the result is finite and not tiny, the
//!   host's result is the architecture's, and the only flag it can raise is "inexact", which is
//!   worked out exactly (a sum by error-free transformation, a product or quotient by the exact
//!   double-precision product), so `FPSR` ends up as the runtime would leave it. Anything else,
//!   NaNs, infinities, denormals, other rounding modes, overflow, takes the call, which is also
//!   the result's source of truth in the differential tests;
//! - the loads and stores of one or two registers of 1 to 16 bytes (`LDR`/`STR`/`LDP`/`STP` of
//!   `B`, `H`, `S`, `D`, `Q`), as 64-bit (or narrower) accesses through the same runtime
//!   functions the integer ones use.
//!
//! Everything else keeps the call.

use super::{Emitter, Op};
use crate::lift::ir::{BlockId, ValueId};
use nx_runtime::a64::fp::{Fmt, Rounding};
use nx_runtime::a64::state::offsets as st;
use nx_runtime::a64::vec::decode::{decode, decode_mem};
use nx_runtime::a64::vec::exec::expand_imm;
use nx_runtime::a64::vec::simd::{FpBin as SimdFp, Logic, Permute, SimdOp};
use nx_runtime::a64::vec::{FpBinary, FpUnary, VMem, VOp};
use std::fmt::Write;

/// `+inf`, `0.0` and 2^-125 as LLVM `float` constants (a double's hex form that a float holds).
const INF: &str = "0x7FF0000000000000";
const ZERO: &str = "0x0000000000000000";
const SMALL: &str = "0x3820000000000000";
/// 2^-126, the smallest normal `float`: anything below it (and not zero) is a denormal.
const MIN_NORMAL: &str = "0x3810000000000000";
/// The `FPCR` bits the fast forms accept besides a rounding to nearest: flush-to-zero (bit 24),
/// default NaN (25) and alternative half precision (26).
const FPCR_ALLOWED: u32 = 0x0700_0000;
/// Flush-to-zero: operands that are denormal are read as zero.
const FPCR_FZ: u32 = 0x0100_0000;

/// How a floating-point or vector instruction is lowered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lowering {
    /// By the runtime call.
    Call,
    /// Straight-line IR, no new blocks.
    Inline,
    /// A test, then IR or the call: one join block.
    Guarded,
}

fn lowering(op: &VOp) -> Lowering {
    match *op {
        VOp::FpMov { fmt, .. } if scalar(fmt) => Lowering::Inline,
        VOp::FpUnary {
            op: FpUnary::Abs | FpUnary::Neg,
            fmt,
            ..
        } if scalar(fmt) => Lowering::Inline,
        VOp::FpImmediate { fmt, .. } if scalar(fmt) => Lowering::Inline,
        VOp::FmovToGeneral { .. } | VOp::FmovFromGeneral { .. } => Lowering::Inline,
        VOp::FpSelect { fmt, .. } if scalar(fmt) => Lowering::Inline,
        VOp::FpBinary {
            op: FpBinary::Add | FpBinary::Sub | FpBinary::Mul | FpBinary::Div,
            fmt: Fmt::S,
            ..
        } => Lowering::Guarded,
        VOp::FpCompare { fmt: Fmt::S, .. } => Lowering::Guarded,
        VOp::Simd(SimdOp::Logic { .. }) => Lowering::Inline,
        VOp::Simd(
            SimdOp::DupElem { .. }
            | SimdOp::InsElem { .. }
            | SimdOp::Permute { .. }
            | SimdOp::Ext { .. },
        ) => Lowering::Inline,
        VOp::Simd(SimdOp::FpBin {
            op: SimdFp::Add | SimdFp::Sub | SimdFp::Mul | SimdFp::Mla | SimdFp::Mls,
            fmt: Fmt::S,
            lanes: 2 | 4,
            ..
        }) => Lowering::Guarded,
        VOp::Simd(SimdOp::FpElem {
            op: SimdFp::Mul | SimdFp::Mla | SimdFp::Mls,
            fmt: Fmt::S,
            lanes: 2 | 4,
            ..
        }) => Lowering::Guarded,
        VOp::Simd(SimdOp::IntToFp {
            fmt: Fmt::S,
            lanes: 1 | 2 | 4,
            ..
        }) => Lowering::Guarded,
        VOp::IntToFp {
            fmt: Fmt::S,
            wide: false,
            fbits: 0,
            ..
        } => Lowering::Guarded,
        VOp::FpToInt {
            fmt: Fmt::S,
            wide: false,
            fbits: 0,
            mode: Rounding::Zero,
            ..
        } => Lowering::Guarded,
        _ => Lowering::Call,
    }
}

/// Single or double precision.
fn scalar(fmt: Fmt) -> bool {
    matches!(fmt, Fmt::S | Fmt::D)
}

/// Which part of a vector register an access moves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    /// All of a register of 1 to 8 bytes: a load zeroes the rest of the register.
    Whole,
    /// The low 8 bytes of a 16-byte register.
    Low,
    /// The high 8 bytes of a 16-byte register.
    High,
}

/// One access of the memory transfer of a vector load or store.
#[derive(Clone, Copy)]
struct Access {
    reg: u8,
    /// Bytes from the address of the instruction.
    offset: u64,
    bytes: u8,
    part: Part,
}

/// The accesses (in order) of a vector load or store of one or two registers, and whether it is a
/// load; `None` for the other forms.
fn accesses(word: u32) -> Option<(bool, Vec<Access>)> {
    let (mem, _) = decode_mem(word).ok()?;
    let one = |reg: u8, base: u64, bytes: u8| -> Vec<Access> {
        if bytes == 16 {
            [(0, Part::Low), (8, Part::High)]
                .into_iter()
                .map(|(o, part)| Access {
                    reg,
                    offset: base + o,
                    bytes: 8,
                    part,
                })
                .collect()
        } else {
            vec![Access {
                reg,
                offset: base,
                bytes,
                part: Part::Whole,
            }]
        }
    };
    match mem {
        VMem::Single { load, bytes, rt } if matches!(bytes, 1 | 2 | 4 | 8 | 16) => {
            Some((load, one(rt, 0, bytes)))
        }
        VMem::Pair {
            load,
            bytes,
            rt,
            rt2,
        } if matches!(bytes, 4 | 8 | 16) => {
            let mut v = one(rt, 0, bytes);
            v.extend(one(rt2, u64::from(bytes), bytes));
            Some((load, v))
        }
        _ => None,
    }
}

/// How many join blocks the lowering of `op` makes: one for each access that can fault, one for a
/// guarded floating-point operation.
pub(super) fn splits_of(op: &Op) -> usize {
    match op {
        Op::Load { .. }
        | Op::Store { .. }
        | Op::LoadExcl { .. }
        | Op::StoreExcl { .. }
        | Op::DcZva { .. } => 1,
        Op::VecMem { word, .. } => accesses(*word).map_or(1, |(_, a)| a.len()),
        Op::Vec { word, .. } => match decode(*word) {
            Ok(op) => usize::from(lowering(&op) == Lowering::Guarded),
            Err(_) => 0,
        },
        _ => 0,
    }
}

impl Emitter<'_> {
    /// The address of the low (or high) half of vector register `n` in the CPU structure.
    fn vptr(&mut self, n: u8, high: bool) -> String {
        self.ptr(st::V + 16 * usize::from(n) + if high { 8 } else { 0 })
    }

    fn load_to(&mut self, ty: &str, ptr: &str, hint: &str) -> String {
        let t = self.temp(hint);
        self.line(&format!("{t} = load {ty}, ptr {ptr}"));
        t
    }

    /// Write `value` (an `i64`) to the low half of register `rd` and zero the high half.
    fn write_scalar(&mut self, rd: u8, value: &str) {
        let (lo, hi) = (self.vptr(rd, false), self.vptr(rd, true));
        self.line(&format!("store i64 {value}, ptr {lo}"));
        self.line(&format!("store i64 0, ptr {hi}"));
    }

    /// The low `fmt` of register `n` as an `i64` (zero extended).
    fn read_scalar_bits(&mut self, n: u8, fmt: Fmt) -> String {
        let p = self.vptr(n, false);
        if fmt == Fmt::D {
            return self.load_to("i64", &p, "fr");
        }
        let w = self.load_to("i32", &p, "fr");
        let z = self.temp("fz");
        self.line(&format!("{z} = zext i32 {w} to i64"));
        z
    }

    /// Lower the floating-point or vector instruction `word`, giving its result (the integer the
    /// call returns) the name of value `v`. Returns false when the instruction keeps the call.
    pub(super) fn native_vec(
        &mut self,
        b: BlockId,
        split: &mut u32,
        v: ValueId,
        word: u32,
        src: &str,
        nzcv: &str,
    ) -> bool {
        let Ok(op) = decode(word) else {
            return false;
        };
        let r = format!("%v{v}");
        match (lowering(&op), op) {
            (Lowering::Call, _) => return false,
            (_, VOp::FpMov { fmt, rd, rn }) => {
                let bits = self.read_scalar_bits(rn, fmt);
                self.write_scalar(rd, &bits);
            }
            (
                _,
                VOp::FpUnary {
                    op: unary,
                    fmt,
                    rd,
                    rn,
                },
            ) => {
                let bits = self.read_scalar_bits(rn, fmt);
                let sign: u64 = if fmt == Fmt::D { 1 << 63 } else { 1 << 31 };
                let out = self.temp("fu");
                if unary == FpUnary::Neg {
                    self.line(&format!("{out} = xor i64 {bits}, {}", sign as i64));
                } else {
                    self.line(&format!(
                        "{out} = and i64 {bits}, {}",
                        (!sign & mask(fmt)) as i64
                    ));
                }
                self.write_scalar(rd, &out);
            }
            (_, VOp::FpImmediate { fmt, rd, imm8 }) => {
                let value = expand_imm(fmt, imm8);
                self.write_scalar(rd, &(value as i64).to_string());
            }
            (_, VOp::FmovFromGeneral { wide, high, rd, .. }) => {
                if high {
                    let hi = self.vptr(rd, true);
                    self.line(&format!("store i64 {src}, ptr {hi}"));
                } else if wide {
                    self.write_scalar(rd, src);
                } else {
                    let lo = self.temp("fm");
                    self.line(&format!("{lo} = and i64 {src}, 4294967295"));
                    self.write_scalar(rd, &lo);
                }
            }
            (_, VOp::FmovToGeneral { wide, high, rn, .. }) => {
                let p = self.vptr(rn, high);
                if high || wide {
                    self.line(&format!("{r} = load i64, ptr {p}"));
                } else {
                    let w = self.load_to("i32", &p, "fg");
                    self.line(&format!("{r} = zext i32 {w} to i64"));
                }
            }
            (
                _,
                VOp::FpSelect {
                    fmt,
                    rd,
                    rn,
                    rm,
                    cond,
                },
            ) => {
                let holds = self.condition(cond, nzcv);
                let (pn, pm) = (self.vptr(rn, false), self.vptr(rm, false));
                let pick = self.temp("sp");
                self.line(&format!("{pick} = select i1 {holds}, ptr {pn}, ptr {pm}"));
                let ty = if fmt == Fmt::D { "i64" } else { "i32" };
                let x = self.load_to(ty, &pick, "sv");
                let wide = if fmt == Fmt::D {
                    x
                } else {
                    let z = self.temp("sz");
                    self.line(&format!("{z} = zext i32 {x} to i64"));
                    z
                };
                self.write_scalar(rd, &wide);
            }
            (_, VOp::Simd(SimdOp::Logic { op, q, rd, rn, rm })) => self.logic(op, q, rd, rn, rm),
            (
                _,
                VOp::FpBinary {
                    op: fop,
                    rd,
                    rn,
                    rm,
                    ..
                },
            ) => {
                self.single_arith(b, split, v, word, src, nzcv, fop, rd, rn, rm);
            }
            (_, VOp::FpCompare { rn, rm, .. }) => {
                self.single_compare(b, split, v, word, src, nzcv, rn, rm);
            }
            (_, VOp::Simd(SimdOp::DupElem { sh, idx, rd, rn })) => {
                self.dup_elem(sh.esize, sh.lanes, idx, rd, rn);
            }
            (
                _,
                VOp::Simd(SimdOp::InsElem {
                    esize,
                    dst,
                    src: from,
                    rd,
                    rn,
                }),
            ) => self.ins_elem(esize, dst, from, rd, rn),
            (_, VOp::Simd(SimdOp::Permute { op, sh, rd, rn, rm })) => {
                self.permute(op, sh.esize, sh.lanes, rd, rn, rm);
            }
            (
                _,
                VOp::Simd(SimdOp::Ext {
                    q,
                    imm4,
                    rd,
                    rn,
                    rm,
                }),
            ) => self.ext(q, imm4, rd, rn, rm),
            (
                _,
                VOp::Simd(SimdOp::FpBin {
                    op: fop,
                    lanes,
                    rd,
                    rn,
                    rm,
                    ..
                }),
            ) => {
                let regs = (rd, rn, rm);
                self.vector_single(
                    b,
                    split,
                    v,
                    word,
                    src,
                    nzcv,
                    fop,
                    u32::from(lanes),
                    regs,
                    None,
                );
            }
            (
                _,
                VOp::Simd(SimdOp::FpElem {
                    op: fop,
                    lanes,
                    rd,
                    rn,
                    rm,
                    idx,
                    ..
                }),
            ) => {
                let regs = (rd, rn, rm);
                self.vector_single(
                    b,
                    split,
                    v,
                    word,
                    src,
                    nzcv,
                    fop,
                    u32::from(lanes),
                    regs,
                    Some(idx),
                );
            }
            (
                _,
                VOp::Simd(SimdOp::IntToFp {
                    signed,
                    lanes,
                    rd,
                    rn,
                    ..
                }),
            ) => self.int_to_single(
                b,
                split,
                v,
                word,
                src,
                nzcv,
                signed,
                u32::from(lanes),
                rd,
                IntSource::Vector(rn),
            ),
            (_, VOp::IntToFp { signed, rd, .. }) => {
                self.int_to_single(
                    b,
                    split,
                    v,
                    word,
                    src,
                    nzcv,
                    signed,
                    1,
                    rd,
                    IntSource::General(src),
                );
            }
            (_, VOp::FpToInt { signed, rn, .. }) => {
                self.single_to_int(b, split, v, word, src, nzcv, signed, rn);
            }
            _ => unreachable!("lowering() and native_vec() list the same instructions"),
        }
        true
    }

    /// `N`, `Z`, `C` and `V` of the nibble `nzcv` (an `i64`) and the condition `cond` as an `i1`
    /// (`ConditionHolds`).
    fn condition(&mut self, cond: u8, nzcv: &str) -> String {
        let flag = |this: &mut Self, bit: u32, hint: &str| -> String {
            let s = this.temp(hint);
            this.line(&format!("{s} = lshr i64 {nzcv}, {bit}"));
            let t = this.temp(hint);
            this.line(&format!("{t} = trunc i64 {s} to i1"));
            t
        };
        let not = |this: &mut Self, x: &str| -> String {
            let t = this.temp("cn");
            this.line(&format!("{t} = xor i1 {x}, true"));
            t
        };
        let (n, z, c, vv) = (
            flag(self, 3, "cN"),
            flag(self, 2, "cZ"),
            flag(self, 1, "cC"),
            flag(self, 0, "cV"),
        );
        let result = match cond >> 1 {
            0 => z.clone(),
            1 => c.clone(),
            2 => n.clone(),
            3 => vv.clone(),
            4 => {
                let nz = not(self, &z);
                let t = self.temp("ch");
                self.line(&format!("{t} = and i1 {c}, {nz}"));
                t
            }
            5 => {
                let t = self.temp("ce");
                self.line(&format!("{t} = icmp eq i1 {n}, {vv}"));
                t
            }
            6 => {
                let e = self.temp("ce");
                self.line(&format!("{e} = icmp eq i1 {n}, {vv}"));
                let nz = not(self, &z);
                let t = self.temp("cg");
                self.line(&format!("{t} = and i1 {e}, {nz}"));
                t
            }
            _ => "true".to_string(),
        };
        if cond & 1 == 1 && cond != 0b1111 {
            not(self, &result)
        } else {
            result
        }
    }

    /// `AND`, `BIC`, `ORR`, `ORN`, `EOR`, `BSL`, `BIT`, `BIF` on the 64-bit halves of a register.
    fn logic(&mut self, op: Logic, q: bool, rd: u8, rn: u8, rm: u8) {
        for high in [false, true] {
            if high && !q {
                let hi = self.vptr(rd, true);
                self.line(&format!("store i64 0, ptr {hi}"));
                break;
            }
            let (pn, pm, pd) = (
                self.vptr(rn, high),
                self.vptr(rm, high),
                self.vptr(rd, high),
            );
            let a = self.load_to("i64", &pn, "la");
            let bb = self.load_to("i64", &pm, "lb");
            let out = self.temp("lr");
            let not = |this: &mut Self, x: &str| -> String {
                let t = this.temp("ln");
                this.line(&format!("{t} = xor i64 {x}, -1"));
                t
            };
            match op {
                Logic::And => self.line(&format!("{out} = and i64 {a}, {bb}")),
                Logic::Orr => self.line(&format!("{out} = or i64 {a}, {bb}")),
                Logic::Eor => self.line(&format!("{out} = xor i64 {a}, {bb}")),
                Logic::Bic => {
                    let nb = not(self, &bb);
                    self.line(&format!("{out} = and i64 {a}, {nb}"));
                }
                Logic::Orn => {
                    let nb = not(self, &bb);
                    self.line(&format!("{out} = or i64 {a}, {nb}"));
                }
                Logic::Bsl | Logic::Bit | Logic::Bif => {
                    let d = self.load_to("i64", &pd, "ld");
                    // BSL: d ? a : b. BIT: b ? a : d. BIF: b ? d : a.
                    let (x, y, sel) = match op {
                        Logic::Bsl => (a.clone(), bb.clone(), d.clone()),
                        Logic::Bit => (a.clone(), d.clone(), bb.clone()),
                        _ => (d.clone(), a.clone(), bb.clone()),
                    };
                    let diff = self.temp("lx");
                    self.line(&format!("{diff} = xor i64 {x}, {y}"));
                    let masked = self.temp("lm");
                    self.line(&format!("{masked} = and i64 {diff}, {sel}"));
                    self.line(&format!("{out} = xor i64 {masked}, {y}"));
                }
            }
            self.line(&format!("store i64 {out}, ptr {pd}"));
        }
    }

    /// Start a guarded operation: `go` is the test the fast form needs. Returns the labels
    /// `(write, call, join)`; the caller continues in `write` after this.
    fn guard(&mut self, b: BlockId, split: &mut u32, go: &str) -> (String, String, String) {
        *split += 1;
        let join = format!("b{b}.m{split}");
        let (write, call) = (format!("{join}.w"), format!("{join}.s"));
        self.line(&format!(
            "br i1 {go}, label %{write}, label %{call}, !prof !1"
        ));
        let _ = writeln!(self.out, "{write}:");
        (write, call, join)
    }

    /// Finish a guarded operation: the fast path jumps to the join block, the call path is
    /// emitted, and the result `fast` (an `i64`) or the call's is the value of `v`.
    #[allow(clippy::too_many_arguments)]
    fn unguard(
        &mut self,
        (write, call, join): &(String, String, String),
        v: ValueId,
        word: u32,
        src: &str,
        nzcv: &str,
        fast: &str,
    ) {
        self.line(&format!("br label %{join}"));
        let _ = writeln!(self.out, "{call}:");
        let slow = self.temp("vs");
        self.line(&format!(
            "{slow} = call i64 @nx_vec(ptr %cpu, i32 {}, i64 {src}, i64 {nzcv})",
            word as i32
        ));
        self.line(&format!("br label %{join}"));
        let _ = writeln!(self.out, "{join}:");
        self.line(&format!(
            "%v{v} = phi i64 [ {fast}, %{write} ], [ {slow}, %{call} ]"
        ));
    }

    /// Is `FPCR` clear but for flush-to-zero, default NaN and alternative half precision (the
    /// rounding is to nearest), and is flush-to-zero set? A result that is a NaN or a denormal
    /// takes the call whatever the modes are, so only the operands of flush-to-zero need a look.
    fn fpcr_modes(&mut self) -> (String, String) {
        let p = self.ptr(st::FPCR);
        let w = self.load_to("i32", &p, "fc");
        let rest = self.temp("f1");
        self.line(&format!("{rest} = and i32 {w}, {}", !FPCR_ALLOWED as i32));
        let plain = self.temp("f2");
        self.line(&format!("{plain} = icmp eq i32 {rest}, 0"));
        let fz = self.temp("f3");
        self.line(&format!("{fz} = and i32 {w}, {}", FPCR_FZ as i32));
        let flushing = self.temp("f4");
        self.line(&format!("{flushing} = icmp ne i32 {fz}, 0"));
        (plain, flushing)
    }

    /// May the fast form of a scalar single-precision operation on `operands` (LLVM values or
    /// constants) run? `FPCR` is the default, or has only the modes of `FPCR_ALLOWED`, and when
    /// flush-to-zero is set no operand is a denormal (the architecture reads it as zero).
    fn scalar_fpcr_ok(&mut self, operands: &[&str]) -> String {
        self.intrinsic("declare float @llvm.fabs.f32(float)".to_string());
        let (plain, flushing) = self.fpcr_modes();
        let mut normal = "true".to_string();
        for x in operands {
            let (ax, big, zero, nd, both) = (
                self.temp("n1"),
                self.temp("n2"),
                self.temp("n3"),
                self.temp("n4"),
                self.temp("n5"),
            );
            self.line(&format!("{ax} = call float @llvm.fabs.f32(float {x})"));
            self.line(&format!("{big} = fcmp oge float {ax}, {MIN_NORMAL}"));
            self.line(&format!("{zero} = fcmp oeq float {x}, {ZERO}"));
            self.line(&format!("{nd} = or i1 {big}, {zero}"));
            self.line(&format!("{both} = and i1 {normal}, {nd}"));
            normal = both;
        }
        self.fpcr_verdict(&plain, &flushing, &normal)
    }

    /// The same for the lanes of the vectors `operands`.
    fn vector_fpcr_ok(&mut self, lanes: u32, operands: &[&str]) -> String {
        let f = format!("<{lanes} x float>");
        self.intrinsic(format!(
            "declare <{lanes} x float> @llvm.fabs.v{lanes}f32(<{lanes} x float>)"
        ));
        let (plain, flushing) = self.fpcr_modes();
        let mut normal = "true".to_string();
        for x in operands {
            let (ax, big, zero, nd) = (
                self.temp("n1"),
                self.temp("n2"),
                self.temp("n3"),
                self.temp("n4"),
            );
            self.line(&format!("{ax} = call {f} @llvm.fabs.v{lanes}f32({f} {x})"));
            self.line(&format!(
                "{big} = fcmp oge {f} {ax}, {}",
                splat(lanes, "float", MIN_NORMAL)
            ));
            self.line(&format!(
                "{zero} = fcmp oeq {f} {x}, {}",
                splat(lanes, "float", ZERO)
            ));
            self.line(&format!("{nd} = or <{lanes} x i1> {big}, {zero}"));
            let all = self.reduce("and", lanes, &nd);
            let both = self.temp("n5");
            self.line(&format!("{both} = and i1 {normal}, {all}"));
            normal = both;
        }
        self.fpcr_verdict(&plain, &flushing, &normal)
    }

    /// `plain` and (`flushing` implies `normal`).
    fn fpcr_verdict(&mut self, plain: &str, flushing: &str, normal: &str) -> String {
        let (idle, either, ok) = (self.temp("v1"), self.temp("v2"), self.temp("v3"));
        self.line(&format!("{idle} = xor i1 {flushing}, true"));
        self.line(&format!("{either} = or i1 {idle}, {normal}"));
        self.line(&format!("{ok} = and i1 {plain}, {either}"));
        ok
    }

    /// Is `FPCR` zero: the rounding to nearest, no flush to zero, no other modes?
    fn default_fpcr(&mut self) -> String {
        let p = self.ptr(st::FPCR);
        let w = self.load_to("i32", &p, "fc");
        let t = self.temp("f0");
        self.line(&format!("{t} = icmp eq i32 {w}, 0"));
        t
    }

    #[allow(clippy::too_many_arguments)]
    fn single_arith(
        &mut self,
        b: BlockId,
        split: &mut u32,
        v: ValueId,
        word: u32,
        src: &str,
        nzcv: &str,
        op: FpBinary,
        rd: u8,
        rn: u8,
        rm: u8,
    ) {
        self.intrinsic("declare float @llvm.fabs.f32(float)".to_string());
        let (pa, pb) = (self.vptr(rn, false), self.vptr(rm, false));
        let a = self.load_to("float", &pa, "xa");
        let bb = self.load_to("float", &pb, "xb");
        let t = |this: &mut Self, h: &str| this.temp(h);
        let (r, inexact) = match op {
            FpBinary::Add | FpBinary::Sub => {
                let o = if op == FpBinary::Sub {
                    let nb = t(self, "xn");
                    self.line(&format!("{nb} = fneg float {bb}"));
                    nb
                } else {
                    bb.clone()
                };
                let r = t(self, "xr");
                self.line(&format!("{r} = fadd float {a}, {o}"));
                // Error-free transformation of the sum: the rounding error, exactly.
                let (v1, v2, e1, e2, e) = (
                    t(self, "xv"),
                    t(self, "xt"),
                    t(self, "xe"),
                    t(self, "xf"),
                    t(self, "xg"),
                );
                self.line(&format!("{v1} = fsub float {r}, {a}"));
                self.line(&format!("{v2} = fsub float {r}, {v1}"));
                self.line(&format!("{e1} = fsub float {a}, {v2}"));
                self.line(&format!("{e2} = fsub float {o}, {v1}"));
                self.line(&format!("{e} = fadd float {e1}, {e2}"));
                let inexact = t(self, "xi");
                self.line(&format!("{inexact} = fcmp une float {e}, {ZERO}"));
                (r, inexact)
            }
            FpBinary::Mul => {
                let (ad, bd, pd) = (t(self, "xd"), t(self, "xe"), t(self, "xp"));
                self.line(&format!("{ad} = fpext float {a} to double"));
                self.line(&format!("{bd} = fpext float {bb} to double"));
                // Two 24-bit significands: the product is exact in a double.
                self.line(&format!("{pd} = fmul double {ad}, {bd}"));
                let (r, rd2) = (t(self, "xr"), t(self, "xs"));
                self.line(&format!("{r} = fptrunc double {pd} to float"));
                self.line(&format!("{rd2} = fpext float {r} to double"));
                let inexact = t(self, "xi");
                self.line(&format!("{inexact} = fcmp une double {rd2}, {pd}"));
                (r, inexact)
            }
            _ => {
                let r = t(self, "xr");
                self.line(&format!("{r} = fdiv float {a}, {bb}"));
                let (ad, bd, rd2, pd) =
                    (t(self, "xd"), t(self, "xe"), t(self, "xs"), t(self, "xp"));
                self.line(&format!("{ad} = fpext float {a} to double"));
                self.line(&format!("{bd} = fpext float {bb} to double"));
                self.line(&format!("{rd2} = fpext float {r} to double"));
                // The quotient times the divisor is exact in a double: it is the dividend
                // when the quotient was exact.
                self.line(&format!("{pd} = fmul double {rd2}, {bd}"));
                let inexact = t(self, "xi");
                self.line(&format!("{inexact} = fcmp une double {pd}, {ad}"));
                (r, inexact)
            }
        };
        // The fast form's conditions: a default FPCR, a finite result that is zero (exactly) or
        // not tiny.
        let ar = t(self, "ya");
        self.line(&format!("{ar} = call float @llvm.fabs.f32(float {r})"));
        let (fin, big, zero, exact, zok, sz, ok) = (
            t(self, "yf"),
            t(self, "yb"),
            t(self, "yz"),
            t(self, "ye"),
            t(self, "yk"),
            t(self, "ys"),
            t(self, "yo"),
        );
        self.line(&format!("{fin} = fcmp olt float {ar}, {INF}"));
        self.line(&format!("{big} = fcmp oge float {ar}, {SMALL}"));
        self.line(&format!("{zero} = fcmp oeq float {r}, {ZERO}"));
        self.line(&format!("{exact} = xor i1 {inexact}, true"));
        self.line(&format!("{zok} = and i1 {zero}, {exact}"));
        self.line(&format!("{sz} = or i1 {big}, {zok}"));
        self.line(&format!("{ok} = and i1 {fin}, {sz}"));
        let fp0 = self.scalar_fpcr_ok(&[&a, &bb]);
        let go = t(self, "yg");
        self.line(&format!("{go} = and i1 {ok}, {fp0}"));
        let labels = self.guard(b, split, &go);
        // The result, and the inexact flag if the rounding lost something.
        let bits = t(self, "wb");
        self.line(&format!("{bits} = bitcast float {r} to i32"));
        let wide = t(self, "ww");
        self.line(&format!("{wide} = zext i32 {bits} to i64"));
        self.write_scalar(rd, &wide);
        let p = self.ptr(st::FPSR);
        let old = self.load_to("i32", &p, "fo");
        let flag = t(self, "fx");
        self.line(&format!("{flag} = select i1 {inexact}, i32 16, i32 0"));
        let new = t(self, "fn");
        self.line(&format!("{new} = or i32 {old}, {flag}"));
        self.line(&format!("store i32 {new}, ptr {p}"));
        self.unguard(&labels, v, word, src, nzcv, "0");
    }

    #[allow(clippy::too_many_arguments)]
    fn single_compare(
        &mut self,
        b: BlockId,
        split: &mut u32,
        v: ValueId,
        word: u32,
        src: &str,
        nzcv: &str,
        rn: u8,
        rm: Option<u8>,
    ) {
        let pa = self.vptr(rn, false);
        let a = self.load_to("float", &pa, "ca");
        let bb = match rm {
            Some(m) => {
                let pb = self.vptr(m, false);
                self.load_to("float", &pb, "cb")
            }
            None => ZERO.to_string(),
        };
        let ord = self.temp("co");
        self.line(&format!("{ord} = fcmp ord float {a}, {bb}"));
        let fp0 = self.scalar_fpcr_ok(&[&a, &bb]);
        let go = self.temp("cg");
        self.line(&format!("{go} = and i1 {ord}, {fp0}"));
        let labels = self.guard(b, split, &go);
        let (lt, eq, mid, res) = (
            self.temp("cl"),
            self.temp("ce"),
            self.temp("cm"),
            self.temp("cr"),
        );
        self.line(&format!("{lt} = fcmp olt float {a}, {bb}"));
        self.line(&format!("{eq} = fcmp oeq float {a}, {bb}"));
        self.line(&format!("{mid} = select i1 {eq}, i64 6, i64 2"));
        self.line(&format!("{res} = select i1 {lt}, i64 8, i64 {mid}"));
        self.unguard(&labels, v, word, src, nzcv, &res);
    }

    /// Lower a vector load or store of one or two registers. Returns false when the instruction
    /// keeps the call.
    pub(super) fn native_vecmem(
        &mut self,
        b: BlockId,
        split: &mut u32,
        word: u32,
        pc: u64,
        addr: &str,
    ) -> bool {
        let Some((load, list)) = accesses(word) else {
            return false;
        };
        // A load writes the registers only when every access went well.
        let mut loaded: Vec<String> = Vec::new();
        for a in &list {
            let at = if a.offset == 0 {
                addr.to_string()
            } else {
                let t = self.temp("va");
                self.line(&format!("{t} = add i64 {addr}, {}", a.offset));
                t
            };
            let bits = u32::from(a.bytes) * 8;
            if load {
                let x = self.temp("vl");
                self.line(&format!(
                    "{x} = call i{bits} @nx_ld{bits}(ptr %cpu, i64 {at}, i64 {})",
                    pc as i64
                ));
                self.fault_check(b, split);
                loaded.push(x);
            } else {
                let p = self.vptr(a.reg, a.part == Part::High);
                let x = self.load_to(&format!("i{bits}"), &p, "vs");
                self.line(&format!(
                    "call void @nx_st{bits}(ptr %cpu, i64 {at}, i{bits} {x}, i64 {})",
                    pc as i64
                ));
                self.fault_check(b, split);
            }
        }
        if load {
            for (a, x) in list.iter().zip(&loaded) {
                match a.part {
                    Part::Whole => {
                        let wide = if a.bytes == 8 {
                            x.clone()
                        } else {
                            let t = self.temp("vz");
                            self.line(&format!(
                                "{t} = zext i{} {x} to i64",
                                u32::from(a.bytes) * 8
                            ));
                            t
                        };
                        self.write_scalar(a.reg, &wide);
                    }
                    Part::Low | Part::High => {
                        let p = self.vptr(a.reg, a.part == Part::High);
                        self.line(&format!("store i64 {x}, ptr {p}"));
                    }
                }
            }
        }
        true
    }
}

/// A vector constant with every lane `value`: `<float 0x..., float 0x...>`.
fn splat(lanes: u32, ty: &str, value: &str) -> String {
    let one = format!("{ty} {value}");
    format!("<{}>", vec![one; lanes as usize].join(", "))
}

/// Where the integers of an integer-to-float conversion come from.
enum IntSource<'a> {
    /// The lanes of a vector register.
    Vector(u8),
    /// A general register, the value of which is given.
    General(&'a str),
}

impl Emitter<'_> {
    /// `<lanes x elem>` read from the start of register `n`.
    fn load_vec(&mut self, n: u8, lanes: u32, elem: &str) -> String {
        let p = self.vptr(n, false);
        let t = self.temp("lv");
        self.line(&format!("{t} = load <{lanes} x {elem}>, ptr {p}, align 8"));
        t
    }

    /// Make `value` (`<lanes x elem>`, `bits` wide in all: 8 to 128) the start of register `n`;
    /// what lies above it is cleared.
    fn store_vec(&mut self, n: u8, lanes: u32, elem: &str, bits: u32, value: &str) {
        let lo = self.vptr(n, false);
        if bits >= 64 {
            self.line(&format!(
                "store <{lanes} x {elem}> {value}, ptr {lo}, align 8"
            ));
            if bits == 64 {
                let hi = self.vptr(n, true);
                self.line(&format!("store i64 0, ptr {hi}"));
            }
            return;
        }
        // Narrower than a half: the value as an integer, widened, so that the bytes above it in
        // the low half are cleared too.
        let (whole, wide) = (self.temp("sv"), self.temp("sw"));
        self.line(&format!(
            "{whole} = bitcast <{lanes} x {elem}> {value} to i{bits}"
        ));
        self.line(&format!("{wide} = zext i{bits} {whole} to i64"));
        let hi = self.vptr(n, true);
        self.line(&format!("store i64 {wide}, ptr {lo}"));
        self.line(&format!("store i64 0, ptr {hi}"));
    }

    /// Raise the "inexact" flag of `FPSR` when `inexact` (an `i1`) is true.
    fn set_inexact(&mut self, inexact: &str) {
        let p = self.ptr(st::FPSR);
        let old = self.load_to("i32", &p, "fo");
        let flag = self.temp("fx");
        self.line(&format!("{flag} = select i1 {inexact}, i32 16, i32 0"));
        let new = self.temp("fn");
        self.line(&format!("{new} = or i32 {old}, {flag}"));
        self.line(&format!("store i32 {new}, ptr {p}"));
    }

    fn reduce(&mut self, kind: &str, lanes: u32, v: &str) -> String {
        self.intrinsic(format!(
            "declare i1 @llvm.vector.reduce.{kind}.v{lanes}i1(<{lanes} x i1>)"
        ));
        let t = self.temp("rd");
        self.line(&format!(
            "{t} = call i1 @llvm.vector.reduce.{kind}.v{lanes}i1(<{lanes} x i1> {v})"
        ));
        t
    }

    /// `DUP Vd.T, Vn.Ts[idx]`.
    fn dup_elem(&mut self, esize: u8, lanes: u8, idx: u8, rd: u8, rn: u8) {
        let (e, l) = (u32::from(esize), u32::from(lanes));
        let ity = format!("i{e}");
        let whole = 128 / e;
        let source = self.load_vec(rn, whole, &ity);
        let x = self.temp("dx");
        self.line(&format!(
            "{x} = extractelement <{whole} x {ity}> {source}, i32 {idx}"
        ));
        let one = self.temp("di");
        self.line(&format!(
            "{one} = insertelement <{l} x {ity}> poison, {ity} {x}, i32 0"
        ));
        let all = self.temp("ds");
        self.line(&format!(
            "{all} = shufflevector <{l} x {ity}> {one}, <{l} x {ity}> poison, <{l} x i32> zeroinitializer"
        ));
        self.store_vec(rd, l, &ity, l * e, &all);
    }

    /// `INS Vd.Ts[dst], Vn.Ts[src]`: one lane replaced, the others kept.
    fn ins_elem(&mut self, esize: u8, dst: u8, from: u8, rd: u8, rn: u8) {
        let e = u32::from(esize);
        let ity = format!("i{e}");
        let whole = 128 / e;
        let old = self.load_vec(rd, whole, &ity);
        let source = self.load_vec(rn, whole, &ity);
        let x = self.temp("ix");
        self.line(&format!(
            "{x} = extractelement <{whole} x {ity}> {source}, i32 {from}"
        ));
        let new = self.temp("in");
        self.line(&format!(
            "{new} = insertelement <{whole} x {ity}> {old}, {ity} {x}, i32 {dst}"
        ));
        self.store_vec(rd, whole, &ity, 128, &new);
    }

    /// `ZIP1`, `ZIP2`, `UZP1`, `UZP2`, `TRN1`, `TRN2`.
    fn permute(&mut self, op: Permute, esize: u8, lanes: u8, rd: u8, rn: u8, rm: u8) {
        let (e, l) = (u32::from(esize), u32::from(lanes));
        let ity = format!("i{e}");
        let n = self.load_vec(rn, l, &ity);
        let m = self.load_vec(rm, l, &ity);
        let half = l / 2;
        let part = u32::from(matches!(op, Permute::Uzp2 | Permute::Trn2 | Permute::Zip2));
        // Lanes `0..l` of the shuffle are those of `n`, `l..2l` those of `m`.
        let mut mask = vec![0u32; l as usize];
        for i in 0..half {
            let (i2, h) = ((2 * i) as usize, (half + i) as usize);
            match op {
                Permute::Uzp1 | Permute::Uzp2 => {
                    mask[i as usize] = 2 * i + part;
                    mask[h] = l + 2 * i + part;
                }
                Permute::Trn1 | Permute::Trn2 => {
                    mask[i2] = 2 * i + part;
                    mask[i2 + 1] = l + 2 * i + part;
                }
                Permute::Zip1 | Permute::Zip2 => {
                    mask[i2] = part * half + i;
                    mask[i2 + 1] = l + part * half + i;
                }
            }
        }
        let list: Vec<String> = mask.iter().map(|x| format!("i32 {x}")).collect();
        let r = self.temp("pr");
        self.line(&format!(
            "{r} = shufflevector <{l} x {ity}> {n}, <{l} x {ity}> {m}, <{l} x i32> <{}>",
            list.join(", ")
        ));
        self.store_vec(rd, l, &ity, l * e, &r);
    }

    /// `EXT Vd.T, Vn.T, Vm.T, #imm`: `size` bytes from the byte `imm4` of the bytes of `Vn`
    /// followed by those of `Vm`.
    fn ext(&mut self, q: bool, imm4: u8, rd: u8, rn: u8, rm: u8) {
        let size: u32 = if q { 16 } else { 8 };
        let n = self.load_vec(rn, size, "i8");
        let m = self.load_vec(rm, size, "i8");
        let list: Vec<String> = (0..size)
            .map(|i| format!("i32 {}", u32::from(imm4) + i))
            .collect();
        let r = self.temp("er");
        self.line(&format!(
            "{r} = shufflevector <{size} x i8> {n}, <{size} x i8> {m}, <{size} x i32> <{}>",
            list.join(", ")
        ));
        self.store_vec(rd, size, "i8", size * 8, &r);
    }

    /// Vector `FADD`, `FSUB`, `FMUL`, `FMLA` and `FMLS` on single-precision lanes, the second
    /// operand a register or, with `elem`, one lane of a register broadcast. As for the scalar
    /// forms (`single_arith`), the fast form needs a default `FPCR` and lanes that are all finite
    /// and not tiny; the fused forms are worked out in double precision, which holds a product of
    /// two singles exactly, with the one case where rounding twice could differ (the sum lands
    /// exactly between two singles after rounding to a double) left to the call.
    #[allow(clippy::too_many_arguments)]
    fn vector_single(
        &mut self,
        b: BlockId,
        split: &mut u32,
        v: ValueId,
        word: u32,
        src: &str,
        nzcv: &str,
        op: SimdFp,
        lanes: u32,
        regs: (u8, u8, u8),
        elem: Option<u8>,
    ) {
        let (rd, rn, rm) = regs;
        let (f, d) = (format!("<{lanes} x float>"), format!("<{lanes} x double>"));
        self.intrinsic(format!(
            "declare <{lanes} x float> @llvm.fabs.v{lanes}f32(<{lanes} x float>)"
        ));
        let t = |this: &mut Self, h: &str| this.temp(h);
        let mut a = self.load_vec(rn, lanes, "float");
        let bb = match elem {
            None => self.load_vec(rm, lanes, "float"),
            Some(idx) => {
                let whole = self.load_vec(rm, 4, "float");
                let x = t(self, "ex");
                self.line(&format!(
                    "{x} = extractelement <4 x float> {whole}, i32 {idx}"
                ));
                let one = t(self, "ei");
                self.line(&format!(
                    "{one} = insertelement {f} poison, float {x}, i32 0"
                ));
                let all = t(self, "es");
                self.line(&format!(
                    "{all} = shufflevector {f} {one}, {f} poison, <{lanes} x i32> zeroinitializer"
                ));
                all
            }
        };
        let zero_f = splat(lanes, "float", ZERO);
        let zero_d = splat(lanes, "double", "0.0");
        // The accumulator of a multiply-add is an operand too.
        let mut accumulator = None;
        let (r, inexact_v, hazard_v) = match op {
            SimdFp::Add | SimdFp::Sub => {
                let o = if op == SimdFp::Sub {
                    let nb = t(self, "xn");
                    self.line(&format!("{nb} = fneg {f} {bb}"));
                    nb
                } else {
                    bb.clone()
                };
                let r = t(self, "xr");
                self.line(&format!("{r} = fadd {f} {a}, {o}"));
                let (v1, v2, e1, e2, e) = (
                    t(self, "xv"),
                    t(self, "xt"),
                    t(self, "xe"),
                    t(self, "xf"),
                    t(self, "xg"),
                );
                self.line(&format!("{v1} = fsub {f} {r}, {a}"));
                self.line(&format!("{v2} = fsub {f} {r}, {v1}"));
                self.line(&format!("{e1} = fsub {f} {a}, {v2}"));
                self.line(&format!("{e2} = fsub {f} {o}, {v1}"));
                self.line(&format!("{e} = fadd {f} {e1}, {e2}"));
                let inexact = t(self, "xi");
                self.line(&format!("{inexact} = fcmp une {f} {e}, {zero_f}"));
                (r, inexact, None)
            }
            SimdFp::Mul | SimdFp::Mla | SimdFp::Mls => {
                if op == SimdFp::Mls {
                    let na = t(self, "xn");
                    self.line(&format!("{na} = fneg {f} {a}"));
                    a = na;
                }
                let (ad, bd, pd) = (t(self, "xd"), t(self, "xe"), t(self, "xp"));
                self.line(&format!("{ad} = fpext {f} {a} to {d}"));
                self.line(&format!("{bd} = fpext {f} {bb} to {d}"));
                // Two 24-bit significands: the product is exact in a double.
                self.line(&format!("{pd} = fmul {d} {ad}, {bd}"));
                if op == SimdFp::Mul {
                    let (r, rd2) = (t(self, "xr"), t(self, "xs"));
                    self.line(&format!("{r} = fptrunc {d} {pd} to {f}"));
                    self.line(&format!("{rd2} = fpext {f} {r} to {d}"));
                    let inexact = t(self, "xi");
                    self.line(&format!("{inexact} = fcmp une {d} {rd2}, {pd}"));
                    (r, inexact, None)
                } else {
                    // The accumulator is the destination register as it is now.
                    let acc = self.load_vec(rd, lanes, "float");
                    accumulator = Some(acc.clone());
                    let dd = t(self, "xc");
                    self.line(&format!("{dd} = fpext {f} {acc} to {d}"));
                    let s = t(self, "xs");
                    self.line(&format!("{s} = fadd {d} {pd}, {dd}"));
                    // The error of that sum, exactly (it is of doubles, which hold it).
                    let (v1, v2, e1, e2, e) = (
                        t(self, "xv"),
                        t(self, "xt"),
                        t(self, "xe"),
                        t(self, "xf"),
                        t(self, "xg"),
                    );
                    self.line(&format!("{v1} = fsub {d} {s}, {pd}"));
                    self.line(&format!("{v2} = fsub {d} {s}, {v1}"));
                    self.line(&format!("{e1} = fsub {d} {pd}, {v2}"));
                    self.line(&format!("{e2} = fsub {d} {dd}, {v1}"));
                    self.line(&format!("{e} = fadd {d} {e1}, {e2}"));
                    let sum_inexact = t(self, "xk");
                    self.line(&format!("{sum_inexact} = fcmp une {d} {e}, {zero_d}"));
                    let (r, back) = (t(self, "xr"), t(self, "xb"));
                    self.line(&format!("{r} = fptrunc {d} {s} to {f}"));
                    self.line(&format!("{back} = fpext {f} {r} to {d}"));
                    let narrow_inexact = t(self, "xj");
                    self.line(&format!("{narrow_inexact} = fcmp une {d} {back}, {s}"));
                    let inexact = t(self, "xi");
                    self.line(&format!(
                        "{inexact} = or <{lanes} x i1> {sum_inexact}, {narrow_inexact}"
                    ));
                    // A sum that was rounded to a double and lies exactly between two singles
                    // might round the wrong way when it is rounded again.
                    let (bits, low, tie, hazard) =
                        (t(self, "xz"), t(self, "xl"), t(self, "xm"), t(self, "xh"));
                    self.line(&format!("{bits} = bitcast {d} {s} to <{lanes} x i64>"));
                    self.line(&format!(
                        "{low} = and <{lanes} x i64> {bits}, {}",
                        splat(lanes, "i64", "536870911")
                    ));
                    self.line(&format!(
                        "{tie} = icmp eq <{lanes} x i64> {low}, {}",
                        splat(lanes, "i64", "268435456")
                    ));
                    self.line(&format!(
                        "{hazard} = and <{lanes} x i1> {sum_inexact}, {tie}"
                    ));
                    (r, inexact, Some(hazard))
                }
            }
            _ => unreachable!("lowering() and vector_single() list the same operations"),
        };
        // The fast form's conditions, in every lane.
        let ar = t(self, "ya");
        self.line(&format!("{ar} = call {f} @llvm.fabs.v{lanes}f32({f} {r})"));
        let (fin, big, zero, exact, zok, sz, okv) = (
            t(self, "yf"),
            t(self, "yb"),
            t(self, "yz"),
            t(self, "ye"),
            t(self, "yk"),
            t(self, "ys"),
            t(self, "yo"),
        );
        self.line(&format!(
            "{fin} = fcmp olt {f} {ar}, {}",
            splat(lanes, "float", INF)
        ));
        self.line(&format!(
            "{big} = fcmp oge {f} {ar}, {}",
            splat(lanes, "float", SMALL)
        ));
        self.line(&format!("{zero} = fcmp oeq {f} {r}, {zero_f}"));
        self.line(&format!(
            "{exact} = xor <{lanes} x i1> {inexact_v}, {}",
            splat(lanes, "i1", "true")
        ));
        self.line(&format!("{zok} = and <{lanes} x i1> {zero}, {exact}"));
        self.line(&format!("{sz} = or <{lanes} x i1> {big}, {zok}"));
        self.line(&format!("{okv} = and <{lanes} x i1> {fin}, {sz}"));
        let all_ok = self.reduce("and", lanes, &okv);
        let any_inexact = self.reduce("or", lanes, &inexact_v);
        let mut go = all_ok;
        if let Some(h) = hazard_v {
            let any = self.reduce("or", lanes, &h);
            let none = t(self, "yn");
            self.line(&format!("{none} = xor i1 {any}, true"));
            let both = t(self, "yc");
            self.line(&format!("{both} = and i1 {go}, {none}"));
            go = both;
        }
        let mut inputs = vec![a.clone(), bb.clone()];
        inputs.extend(accumulator);
        let inputs: Vec<&str> = inputs.iter().map(String::as_str).collect();
        let fp0 = self.vector_fpcr_ok(lanes, &inputs);
        let all = t(self, "yg");
        self.line(&format!("{all} = and i1 {go}, {fp0}"));
        let labels = self.guard(b, split, &all);
        self.store_vec(rd, lanes, "float", lanes * 32, &r);
        self.set_inexact(&any_inexact);
        self.unguard(&labels, v, word, src, nzcv, "0");
    }

    /// `SCVTF` and `UCVTF` of 32-bit integers (the lanes of a register, or a general register) to
    /// single precision: with a default `FPCR` the host rounds as the architecture does, and
    /// the conversion is inexact when the number is not the integer.
    #[allow(clippy::too_many_arguments)]
    fn int_to_single(
        &mut self,
        b: BlockId,
        split: &mut u32,
        v: ValueId,
        word: u32,
        src: &str,
        nzcv: &str,
        signed: bool,
        lanes: u32,
        rd: u8,
        from: IntSource<'_>,
    ) {
        let convert = if signed { "sitofp" } else { "uitofp" };
        let ints = match from {
            IntSource::Vector(rn) => self.load_vec(rn, lanes, "i32"),
            IntSource::General(value) => {
                let w = self.temp("gw");
                self.line(&format!("{w} = trunc i64 {value} to i32"));
                let one = self.temp("gv");
                self.line(&format!(
                    "{one} = insertelement <1 x i32> poison, i32 {w}, i32 0"
                ));
                one
            }
        };
        let (it, ft, dt) = (
            format!("<{lanes} x i32>"),
            format!("<{lanes} x float>"),
            format!("<{lanes} x double>"),
        );
        let (r, exact, back, inexact_v) = (
            self.temp("cr"),
            self.temp("cx"),
            self.temp("cb"),
            self.temp("ci"),
        );
        self.line(&format!("{r} = {convert} {it} {ints} to {ft}"));
        // Every 32-bit integer is exact in a double.
        self.line(&format!("{exact} = {convert} {it} {ints} to {dt}"));
        self.line(&format!("{back} = fpext {ft} {r} to {dt}"));
        self.line(&format!("{inexact_v} = fcmp une {dt} {back}, {exact}"));
        let any = self.reduce("or", lanes, &inexact_v);
        // An integer converts to a normal number or zero: flush-to-zero has nothing to flush.
        let (go, _) = self.fpcr_modes();
        let labels = self.guard(b, split, &go);
        self.store_vec(rd, lanes, "float", lanes * 32, &r);
        self.set_inexact(&any);
        self.unguard(&labels, v, word, src, nzcv, "0");
    }

    /// `FCVTZS` and `FCVTZU` of a single to a 32-bit general register, for the values that
    /// convert without saturating (the others, NaNs and the out-of-range, take the call).
    #[allow(clippy::too_many_arguments)]
    fn single_to_int(
        &mut self,
        b: BlockId,
        split: &mut u32,
        v: ValueId,
        word: u32,
        src: &str,
        nzcv: &str,
        signed: bool,
        rn: u8,
    ) {
        let p = self.vptr(rn, false);
        let x = self.load_to("float", &p, "tx");
        let (low, high, op, back_op) = if signed {
            (
                "0xC1E0000000000000",
                "0x41E0000000000000",
                "fptosi",
                "sitofp",
            )
        } else {
            (ZERO, "0x41F0000000000000", "fptoui", "uitofp")
        };
        let (lo_ok, hi_ok, ok, fp0, go) = (
            self.temp("tl"),
            self.temp("th"),
            self.temp("to"),
            self.default_fpcr(),
            self.temp("tg"),
        );
        self.line(&format!("{lo_ok} = fcmp oge float {x}, {low}"));
        self.line(&format!("{hi_ok} = fcmp olt float {x}, {high}"));
        self.line(&format!("{ok} = and i1 {lo_ok}, {hi_ok}"));
        self.line(&format!("{go} = and i1 {ok}, {fp0}"));
        let labels = self.guard(b, split, &go);
        let (t32, back, inexact, wide) = (
            self.temp("tt"),
            self.temp("tb"),
            self.temp("ti"),
            self.temp("tw"),
        );
        self.line(&format!("{t32} = {op} float {x} to i32"));
        self.line(&format!("{back} = {back_op} i32 {t32} to float"));
        self.line(&format!("{inexact} = fcmp une float {back}, {x}"));
        self.line(&format!("{wide} = zext i32 {t32} to i64"));
        self.set_inexact(&inexact);
        self.unguard(&labels, v, word, src, nzcv, &wide);
    }
}

fn mask(fmt: Fmt) -> u64 {
    if fmt == Fmt::D {
        u64::MAX
    } else {
        0xFFFF_FFFF
    }
}
