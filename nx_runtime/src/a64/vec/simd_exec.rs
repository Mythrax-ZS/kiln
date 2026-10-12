//! Execution of the Advanced SIMD operations.
//!
//! Each arm follows the Arm pseudocode for its instruction: sources are read first, the result is
//! assembled lane by lane in a 128-bit value, and the destination is written once, which makes a
//! destination that is also a source behave as the pseudocode says. Integer arithmetic is done in
//! `i128`, wide enough for every intermediate value (the widest is a 64-bit product shifted by a
//! 64-bit amount).

use super::crypto;
use super::decode_simd::min_max_of;
use super::simd::*;
use crate::a64::fp::{self, Env, Fmt, Rounding};
use crate::a64::state::CpuState;
use std::cmp::Ordering;

/// Low `bits` bits set.
fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn lane(v: u128, esize: u32, i: u32) -> u64 {
    ((v >> (i * esize)) as u64) & mask(esize)
}

fn set_lane(v: &mut u128, esize: u32, i: u32, x: u64) {
    let m = u128::from(mask(esize)) << (i * esize);
    *v = (*v & !m) | (u128::from(x & mask(esize)) << (i * esize));
}

/// Sign-extend the low `bits` bits.
fn sext(x: u64, bits: u32) -> i64 {
    ((x << (64 - bits)) as i64) >> (64 - bits)
}

/// Saturate to a signed integer of `bits` bits: the two's complement bits and whether it saturated.
fn sat_signed(v: i128, bits: u32) -> (u64, bool) {
    let max = (1i128 << (bits - 1)) - 1;
    let min = -(1i128 << (bits - 1));
    if v > max {
        (max as u64 & mask(bits), true)
    } else if v < min {
        (min as u64 & mask(bits), true)
    } else {
        (v as u64 & mask(bits), false)
    }
}

/// Saturate to an unsigned integer of `bits` bits.
fn sat_unsigned(v: i128, bits: u32) -> (u64, bool) {
    let max = (1i128 << bits) - 1;
    if v > max {
        (max as u64, true)
    } else if v < 0 {
        (0, true)
    } else {
        (v as u64, false)
    }
}

/// Carry-less multiplication of two values of at most 64 bits.
fn poly_mul(a: u64, b: u64) -> u128 {
    let mut r = 0u128;
    for i in 0..64 {
        if b >> i & 1 != 0 {
            r ^= u128::from(a) << i;
        }
    }
    r
}

/// The right-shift-with-rounding of a value: `(v + 2^(n-1)) >> n`, for `n` from 1 up.
fn round_shift_right(v: i128, n: u32) -> i128 {
    if n >= 100 {
        0
    } else {
        (v + (1i128 << (n - 1))) >> n
    }
}

struct Ctx<'a> {
    env: &'a mut Env,
    /// Cumulative saturation, becomes `FPSR.QC`.
    sat: bool,
}

/// The shift of the register-controlled shifts: `SInt(b<7:0>)`.
#[allow(clippy::too_many_arguments)]
fn shift_by_register(
    signed: bool,
    round: bool,
    saturate: bool,
    e: u32,
    a: u64,
    b: u64,
    sat: &mut bool,
) -> u64 {
    let shift = i32::from(b as u8 as i8);
    let val: i128 = if signed {
        i128::from(sext(a, e))
    } else {
        i128::from(a)
    };
    if shift >= 0 {
        let s = shift as u32;
        if saturate {
            let (r, sat_now) = if val == 0 {
                (0, false)
            } else if s >= e {
                // Any non-zero value shifted this far leaves the range.
                if signed {
                    sat_signed(if val < 0 { i128::MIN } else { i128::MAX }, e)
                } else {
                    sat_unsigned(i128::MAX, e)
                }
            } else if signed {
                sat_signed(val << s, e)
            } else {
                sat_unsigned(val << s, e)
            };
            *sat |= sat_now;
            r
        } else if s >= e {
            0
        } else {
            ((val << s) as u64) & mask(e)
        }
    } else {
        let n = (-shift) as u32;
        let r = if round {
            round_shift_right(val, n)
        } else {
            val >> n.min(127)
        };
        // A right shift cannot leave the range, so there is nothing to saturate.
        r as u64 & mask(e)
    }
}

/// One element of an integer operation. `a` and `b` are the source elements, `d` the old
/// destination element (for the accumulating operations).
fn int_bin(op: IntBin, e: u32, a: u64, b: u64, d: u64, sat: &mut bool) -> u64 {
    use IntBin::*;
    let m = mask(e);
    let (ua, ub) = (i128::from(a), i128::from(b));
    let (sa, sb) = (i128::from(sext(a, e)), i128::from(sext(b, e)));
    let truth = |c: bool| if c { m } else { 0 };
    let mut saturated = |(r, s): (u64, bool)| {
        *sat |= s;
        r
    };
    match op {
        Shadd => ((sa + sb) >> 1) as u64 & m,
        Uhadd => ((ua + ub) >> 1) as u64 & m,
        Srhadd => ((sa + sb + 1) >> 1) as u64 & m,
        Urhadd => ((ua + ub + 1) >> 1) as u64 & m,
        Shsub => ((sa - sb) >> 1) as u64 & m,
        Uhsub => ((ua - ub) >> 1) as u64 & m,
        Sqadd => saturated(sat_signed(sa + sb, e)),
        Uqadd => saturated(sat_unsigned(ua + ub, e)),
        Sqsub => saturated(sat_signed(sa - sb, e)),
        Uqsub => saturated(sat_unsigned(ua - ub, e)),
        Cmgt => truth(sa > sb),
        Cmhi => truth(ua > ub),
        Cmge => truth(sa >= sb),
        Cmhs => truth(ua >= ub),
        Cmeq => truth(a == b),
        Cmtst => truth(a & b != 0),
        Sshl => shift_by_register(true, false, false, e, a, b, sat),
        Ushl => shift_by_register(false, false, false, e, a, b, sat),
        Srshl => shift_by_register(true, true, false, e, a, b, sat),
        Urshl => shift_by_register(false, true, false, e, a, b, sat),
        Sqshl => shift_by_register(true, false, true, e, a, b, sat),
        Uqshl => shift_by_register(false, false, true, e, a, b, sat),
        Sqrshl => shift_by_register(true, true, true, e, a, b, sat),
        Uqrshl => shift_by_register(false, true, true, e, a, b, sat),
        Smax => sa.max(sb) as u64 & m,
        Umax => ua.max(ub) as u64,
        Smin => sa.min(sb) as u64 & m,
        Umin => ua.min(ub) as u64,
        Sabd => (sa - sb).unsigned_abs() as u64 & m,
        Uabd => (ua - ub).unsigned_abs() as u64,
        Saba => d.wrapping_add((sa - sb).unsigned_abs() as u64) & m,
        Uaba => d.wrapping_add((ua - ub).unsigned_abs() as u64) & m,
        Add => a.wrapping_add(b) & m,
        Sub => a.wrapping_sub(b) & m,
        Mul => a.wrapping_mul(b) & m,
        Mla => d.wrapping_add(a.wrapping_mul(b)) & m,
        Mls => d.wrapping_sub(a.wrapping_mul(b)) & m,
        Pmul => poly_mul(a, b) as u64 & m,
        Sqdmulh => saturated(sat_signed((2 * sa * sb) >> e, e)),
        Sqrdmulh => saturated(sat_signed(((2 * sa * sb) + (1i128 << (e - 1))) >> e, e)),
    }
}

/// One element of a single-source integer operation.
fn int_un(op: IntUn, e: u32, a: u64, d: u64, sat: &mut bool) -> u64 {
    use IntUn::*;
    let m = mask(e);
    let sa = i128::from(sext(a, e));
    let truth = |c: bool| if c { m } else { 0 };
    match op {
        Cls => {
            let v = if sa < 0 { !sa } else { sa };
            let bit_length = 128 - v.leading_zeros();
            u64::from(e - 1 - bit_length)
        }
        Clz => u64::from((a << (64 - e)).leading_zeros().min(e)),
        Cnt => u64::from(a.count_ones()),
        Not => !a & m,
        Rbit => u64::from((a as u8).reverse_bits()),
        Abs => sa.unsigned_abs() as u64 & m,
        Neg => (-sa) as u64 & m,
        Sqabs => {
            let (r, s) = sat_signed(sa.abs(), e);
            *sat |= s;
            r
        }
        Sqneg => {
            let (r, s) = sat_signed(-sa, e);
            *sat |= s;
            r
        }
        Cmgt0 => truth(sa > 0),
        Cmge0 => truth(sa >= 0),
        Cmeq0 => truth(sa == 0),
        Cmle0 => truth(sa <= 0),
        Cmlt0 => truth(sa < 0),
        Suqadd => {
            let (r, s) = sat_signed(i128::from(sext(d, e)) + i128::from(a), e);
            *sat |= s;
            r
        }
        Usqadd => {
            let (r, s) = sat_unsigned(i128::from(d) + sa, e);
            *sat |= s;
            r
        }
        Rev64 | Rev32 | Rev16 | Urecpe | Ursqrte => unreachable!("handled by the caller"),
    }
}

fn reverse_elements(v: u128, e: u32, lanes: u32, container_bits: u32) -> u128 {
    let per = container_bits / e;
    let mut out = 0u128;
    for i in 0..lanes {
        let (c, within) = (i / per, i % per);
        set_lane(&mut out, e, c * per + (per - 1 - within), lane(v, e, i));
    }
    out
}

fn write_half(cpu: &mut CpuState, rd: u8, upper: bool, half: u64) {
    if upper {
        let low = cpu.vreg(rd) & u128::from(u64::MAX);
        cpu.set_vreg(rd, low | (u128::from(half) << 64));
    } else {
        cpu.set_vreg(rd, u128::from(half));
    }
}

fn extend(x: u64, bits: u32, signed: bool) -> i128 {
    if signed {
        i128::from(sext(x, bits))
    } else {
        i128::from(x)
    }
}

/// The narrow source half of a long operation: `n` itself, or its upper 64 bits.
fn narrow_part(v: u128, upper: bool) -> u128 {
    if upper {
        v >> 64
    } else {
        v & u128::from(u64::MAX)
    }
}

/// Execute one Advanced SIMD instruction. `src` is the general source register, for the encodings
/// that have one. The result is the general destination's value for the encodings that write one.
pub fn exec(cpu: &mut CpuState, env: &mut Env, op: &SimdOp, src: u64) -> u64 {
    let mut cx = Ctx { env, sat: false };
    let r = run(cpu, &mut cx, op, src);
    if cx.sat {
        cx.env.fpsr |= fp::QC;
    }
    r
}

fn run(cpu: &mut CpuState, cx: &mut Ctx, op: &SimdOp, src: u64) -> u64 {
    match *op {
        SimdOp::Aes { op, rd, rn } => {
            let (d, n) = (cpu.vreg(rd), cpu.vreg(rn));
            let r = match op {
                Aes::Aese => crypto::aese(d, n),
                Aes::Aesd => crypto::aesd(d, n),
                Aes::Aesmc => crypto::aesmc(n),
                Aes::Aesimc => crypto::aesimc(n),
            };
            cpu.set_vreg(rd, r);
        }
        SimdOp::Sha { op, rd, rn, rm } => {
            let (d, n, m) = (cpu.vreg(rd), cpu.vreg(rn), cpu.vreg(rm));
            // `Sn` is the low word of `Vn`.
            let e = n as u32;
            let r = match op {
                Sha::Sha1c => crypto::sha1_rounds(crypto::Sha1Round::Choose, d, e, m),
                Sha::Sha1p => crypto::sha1_rounds(crypto::Sha1Round::Parity, d, e, m),
                Sha::Sha1m => crypto::sha1_rounds(crypto::Sha1Round::Majority, d, e, m),
                Sha::Sha1su0 => crypto::sha1su0(d, n, m),
                Sha::Sha256h => crypto::sha256h(d, n, m),
                Sha::Sha256h2 => crypto::sha256h2(d, n, m),
                Sha::Sha256su1 => crypto::sha256su1(d, n, m),
                Sha::Sha1h => u128::from(crypto::sha1h(e)),
                Sha::Sha1su1 => crypto::sha1su1(d, n),
                Sha::Sha256su0 => crypto::sha256su0(d, n),
            };
            cpu.set_vreg(rd, r);
        }
        SimdOp::Logic { op, q, rd, rn, rm } => {
            let (a, b, d) = (cpu.vreg(rn), cpu.vreg(rm), cpu.vreg(rd));
            let r = match op {
                Logic::And => a & b,
                Logic::Bic => a & !b,
                Logic::Orr => a | b,
                Logic::Orn => a | !b,
                Logic::Eor => a ^ b,
                Logic::Bsl => (d & a) | (!d & b),
                Logic::Bit => (d & !b) | (a & b),
                Logic::Bif => (d & b) | (a & !b),
            };
            cpu.set_vreg(rd, if q { r } else { r & u128::from(u64::MAX) });
        }
        SimdOp::IntBin { op, sh, rd, rn, rm } => {
            let (a, b, d) = (cpu.vreg(rn), cpu.vreg(rm), cpu.vreg(rd));
            let e = u32::from(sh.esize);
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) {
                let x = int_bin(
                    op,
                    e,
                    lane(a, e, i),
                    lane(b, e, i),
                    lane(d, e, i),
                    &mut cx.sat,
                );
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::IntPair { op, sh, rd, rn, rm } => {
            let (n, m) = (cpu.vreg(rn), cpu.vreg(rm));
            let e = u32::from(sh.esize);
            let lanes = u32::from(sh.lanes);
            let mut r = 0u128;
            for i in 0..lanes {
                let (v, j) = if i < lanes / 2 {
                    (n, i)
                } else {
                    (m, i - lanes / 2)
                };
                let x = int_bin(
                    op,
                    e,
                    lane(v, e, 2 * j),
                    lane(v, e, 2 * j + 1),
                    0,
                    &mut cx.sat,
                );
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::IntElem {
            op,
            sh,
            rd,
            rn,
            rm,
            idx,
        } => {
            let (a, d) = (cpu.vreg(rn), cpu.vreg(rd));
            let e = u32::from(sh.esize);
            let b = lane(cpu.vreg(rm), e, u32::from(idx));
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) {
                let x = int_bin(op, e, lane(a, e, i), b, lane(d, e, i), &mut cx.sat);
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::Long {
            op,
            sh,
            upper,
            rd,
            rn,
            rm,
        } => {
            let (n, m, d) = (cpu.vreg(rn), cpu.vreg(rm), cpu.vreg(rd));
            long_op(cpu, cx, op, sh, upper, rd, n, m, d, None);
        }
        SimdOp::LongElem {
            op,
            sh,
            upper,
            rd,
            rn,
            rm,
            idx,
        } => {
            let (n, d) = (cpu.vreg(rn), cpu.vreg(rd));
            let e = u32::from(sh.esize);
            let b = lane(cpu.vreg(rm), e, u32::from(idx));
            long_op(cpu, cx, op, sh, upper, rd, n, 0, d, Some(b));
        }
        SimdOp::IntUn { op, sh, rd, rn } => {
            let (a, d) = (cpu.vreg(rn), cpu.vreg(rd));
            let e = u32::from(sh.esize);
            let lanes = u32::from(sh.lanes);
            let r = match op {
                IntUn::Rev64 => reverse_elements(a, e, lanes, 64),
                IntUn::Rev32 => reverse_elements(a, e, lanes, 32),
                IntUn::Rev16 => reverse_elements(a, e, lanes, 16),
                IntUn::Urecpe | IntUn::Ursqrte => {
                    let mut r = 0u128;
                    for i in 0..lanes {
                        let x = lane(a, 32, i) as u32;
                        let y = if op == IntUn::Urecpe {
                            fp::unsigned_recip_estimate(x)
                        } else {
                            fp::unsigned_rsqrt_estimate(x)
                        };
                        set_lane(&mut r, 32, i, u64::from(y));
                    }
                    r
                }
                _ => {
                    let mut r = 0u128;
                    for i in 0..lanes {
                        let x = int_un(op, e, lane(a, e, i), lane(d, e, i), &mut cx.sat);
                        set_lane(&mut r, e, i, x);
                    }
                    r
                }
            };
            cpu.set_vreg(rd, r);
        }
        SimdOp::PairLong { op, sh, rd, rn } => {
            let (a, d) = (cpu.vreg(rn), cpu.vreg(rd));
            let e = u32::from(sh.esize);
            let signed = matches!(op, PairLong::Saddlp | PairLong::Sadalp);
            let accumulate = matches!(op, PairLong::Sadalp | PairLong::Uadalp);
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) / 2 {
                let s =
                    extend(lane(a, e, 2 * i), e, signed) + extend(lane(a, e, 2 * i + 1), e, signed);
                let base = if accumulate {
                    i128::from(lane(d, 2 * e, i))
                } else {
                    0
                };
                set_lane(&mut r, 2 * e, i, (s + base) as u64);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::Narrow {
            op,
            sh,
            upper,
            rd,
            rn,
        } => {
            let a = cpu.vreg(rn);
            let e = u32::from(sh.esize);
            let mut half = 0u128;
            for i in 0..u32::from(sh.lanes) {
                let x = lane(a, 2 * e, i);
                let (sx, ux) = (i128::from(sext(x, 2 * e)), i128::from(x));
                let (v, s) = match op {
                    Narrow::Xtn => (x & mask(e), false),
                    Narrow::Sqxtn => sat_signed(sx, e),
                    Narrow::Uqxtn => sat_unsigned(ux, e),
                    Narrow::Sqxtun => sat_unsigned(sx, e),
                };
                cx.sat |= s;
                set_lane(&mut half, e, i, v);
            }
            write_half(cpu, rd, upper, half as u64);
        }
        SimdOp::Shll { sh, upper, rd, rn } => {
            let a = narrow_part(cpu.vreg(rn), upper);
            let e = u32::from(sh.esize);
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) {
                set_lane(&mut r, 2 * e, i, lane(a, e, i) << e);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::Shift {
            op,
            sh,
            shift,
            rd,
            rn,
        } => {
            let (a, d) = (cpu.vreg(rn), cpu.vreg(rd));
            let e = u32::from(sh.esize);
            let s = u32::from(shift);
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) {
                let x = shift_imm(op, e, s, lane(a, e, i), lane(d, e, i), &mut cx.sat);
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::ShiftNarrow {
            op,
            sh,
            upper,
            shift,
            rd,
            rn,
        } => {
            let a = cpu.vreg(rn);
            let e = u32::from(sh.esize);
            let s = u32::from(shift);
            let mut half = 0u128;
            for i in 0..u32::from(sh.lanes) {
                let x = lane(a, 2 * e, i);
                let (sx, ux) = (i128::from(sext(x, 2 * e)), i128::from(x));
                let rc = 1i128 << (s - 1);
                let (v, sat_now) = match op {
                    ShiftNarrow::Shrn => ((ux >> s) as u64 & mask(e), false),
                    ShiftNarrow::Rshrn => (((ux + rc) >> s) as u64 & mask(e), false),
                    ShiftNarrow::Sqshrn => sat_signed(sx >> s, e),
                    ShiftNarrow::Uqshrn => sat_unsigned(ux >> s, e),
                    ShiftNarrow::Sqrshrn => sat_signed((sx + rc) >> s, e),
                    ShiftNarrow::Uqrshrn => sat_unsigned((ux + rc) >> s, e),
                    ShiftNarrow::Sqshrun => sat_unsigned(sx >> s, e),
                    ShiftNarrow::Sqrshrun => sat_unsigned((sx + rc) >> s, e),
                };
                cx.sat |= sat_now;
                set_lane(&mut half, e, i, v);
            }
            write_half(cpu, rd, upper, half as u64);
        }
        SimdOp::ShiftLong {
            signed,
            sh,
            upper,
            shift,
            rd,
            rn,
        } => {
            let a = narrow_part(cpu.vreg(rn), upper);
            let e = u32::from(sh.esize);
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) {
                let v = extend(lane(a, e, i), e, signed) << shift;
                set_lane(&mut r, 2 * e, i, v as u64);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::Across { op, sh, rd, rn } => {
            let a = cpu.vreg(rn);
            let e = u32::from(sh.esize);
            let signed = matches!(op, Across::Saddlv | Across::Smaxv | Across::Sminv);
            let mut acc: i128 = 0;
            for i in 0..u32::from(sh.lanes) {
                let x = extend(lane(a, e, i), e, signed);
                acc = match op {
                    Across::Saddlv | Across::Uaddlv | Across::Addv => acc + x,
                    Across::Smaxv | Across::Umaxv => {
                        if i == 0 {
                            x
                        } else {
                            acc.max(x)
                        }
                    }
                    Across::Sminv | Across::Uminv => {
                        if i == 0 {
                            x
                        } else {
                            acc.min(x)
                        }
                    }
                };
            }
            let width = if matches!(op, Across::Saddlv | Across::Uaddlv) {
                2 * e
            } else {
                e
            };
            cpu.set_vreg(rd, u128::from(acc as u64 & mask(width)));
        }
        SimdOp::Permute { op, sh, rd, rn, rm } => {
            let (n, m) = (cpu.vreg(rn), cpu.vreg(rm));
            let e = u32::from(sh.esize);
            let lanes = u32::from(sh.lanes);
            let half = lanes / 2;
            let part = u32::from(matches!(op, Permute::Uzp2 | Permute::Trn2 | Permute::Zip2));
            let mut r = 0u128;
            match op {
                Permute::Uzp1 | Permute::Uzp2 => {
                    for i in 0..half {
                        set_lane(&mut r, e, i, lane(n, e, 2 * i + part));
                        set_lane(&mut r, e, half + i, lane(m, e, 2 * i + part));
                    }
                }
                Permute::Trn1 | Permute::Trn2 => {
                    for i in 0..half {
                        set_lane(&mut r, e, 2 * i, lane(n, e, 2 * i + part));
                        set_lane(&mut r, e, 2 * i + 1, lane(m, e, 2 * i + part));
                    }
                }
                Permute::Zip1 | Permute::Zip2 => {
                    let base = part * half;
                    for i in 0..half {
                        set_lane(&mut r, e, 2 * i, lane(n, e, base + i));
                        set_lane(&mut r, e, 2 * i + 1, lane(m, e, base + i));
                    }
                }
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::Ext {
            q,
            imm4,
            rd,
            rn,
            rm,
        } => {
            let size = if q { 16 } else { 8 };
            let mut bytes = [0u8; 32];
            let (n, m) = (cpu.vreg(rn), cpu.vreg(rm));
            bytes[..size].copy_from_slice(&n.to_le_bytes()[..size]);
            bytes[size..2 * size].copy_from_slice(&m.to_le_bytes()[..size]);
            let mut out = [0u8; 16];
            let start = usize::from(imm4);
            out[..size].copy_from_slice(&bytes[start..start + size]);
            cpu.set_vreg(rd, u128::from_le_bytes(out));
        }
        SimdOp::Table {
            q,
            regs,
            extend,
            rd,
            rn,
            rm,
        } => {
            let mut table = [0u8; 64];
            for k in 0..usize::from(regs) {
                let reg = cpu.vreg(((usize::from(rn) + k) % 32) as u8);
                table[16 * k..16 * k + 16].copy_from_slice(&reg.to_le_bytes());
            }
            let (idx, old) = (cpu.vreg(rm).to_le_bytes(), cpu.vreg(rd).to_le_bytes());
            let mut out = [0u8; 16];
            for i in 0..(if q { 16 } else { 8 }) {
                let j = usize::from(idx[i]);
                out[i] = if j < 16 * usize::from(regs) {
                    table[j]
                } else if extend {
                    old[i]
                } else {
                    0
                };
            }
            cpu.set_vreg(rd, u128::from_le_bytes(out));
        }
        SimdOp::DupElem { sh, idx, rd, rn } => {
            let e = u32::from(sh.esize);
            let x = lane(cpu.vreg(rn), e, u32::from(idx));
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) {
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::DupGeneral { sh, rd, rn: _ } => {
            let e = u32::from(sh.esize);
            let mut r = 0u128;
            for i in 0..u32::from(sh.lanes) {
                set_lane(&mut r, e, i, src);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::InsGeneral { esize, idx, rd, .. } => {
            let mut r = cpu.vreg(rd);
            set_lane(&mut r, u32::from(esize), u32::from(idx), src);
            cpu.set_vreg(rd, r);
        }
        SimdOp::InsElem {
            esize,
            dst,
            src: from,
            rd,
            rn,
        } => {
            let e = u32::from(esize);
            let x = lane(cpu.vreg(rn), e, u32::from(from));
            let mut r = cpu.vreg(rd);
            set_lane(&mut r, e, u32::from(dst), x);
            cpu.set_vreg(rd, r);
        }
        SimdOp::MovToGeneral {
            signed,
            esize,
            idx,
            wide,
            rn,
            ..
        } => {
            let e = u32::from(esize);
            let x = lane(cpu.vreg(rn), e, u32::from(idx));
            return if signed {
                let v = sext(x, e) as u64;
                if wide {
                    v
                } else {
                    v & 0xFFFF_FFFF
                }
            } else {
                x
            };
        }
        SimdOp::ModImm { op, q, imm, rd } => {
            let imm128 = if q {
                u128::from(imm) | (u128::from(imm) << 64)
            } else {
                u128::from(imm)
            };
            let d = cpu.vreg(rd);
            let r = match op {
                ModImm::Set => imm128,
                ModImm::Orr => d | imm128,
                ModImm::Bic => d & !imm128,
            };
            // A 64-bit destination clears the upper half.
            cpu.set_vreg(rd, if q { r } else { r & u128::from(u64::MAX) });
        }

        // ---- floating point ----
        SimdOp::FpBin {
            op,
            fmt,
            lanes,
            rd,
            rn,
            rm,
        } => {
            let (a, b, d) = (cpu.vreg(rn), cpu.vreg(rm), cpu.vreg(rd));
            let e = fmt.bits();
            let mut r = 0u128;
            for i in 0..u32::from(lanes) {
                let x = fp_bin(cx.env, op, fmt, lane(a, e, i), lane(b, e, i), lane(d, e, i));
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::FpElem {
            op,
            fmt,
            lanes,
            rd,
            rn,
            rm,
            idx,
        } => {
            let (a, d) = (cpu.vreg(rn), cpu.vreg(rd));
            let e = fmt.bits();
            let b = lane(cpu.vreg(rm), e, u32::from(idx));
            let mut r = 0u128;
            for i in 0..u32::from(lanes) {
                let x = fp_bin(cx.env, op, fmt, lane(a, e, i), b, lane(d, e, i));
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::FpPair {
            op,
            fmt,
            lanes,
            rd,
            rn,
            rm,
        } => {
            let (n, m) = (cpu.vreg(rn), cpu.vreg(rm));
            let e = fmt.bits();
            let lanes = u32::from(lanes);
            let mut r = 0u128;
            for i in 0..lanes {
                let (v, j) = if i < lanes / 2 {
                    (n, i)
                } else {
                    (m, i - lanes / 2)
                };
                let x = fp_bin(cx.env, op, fmt, lane(v, e, 2 * j), lane(v, e, 2 * j + 1), 0);
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::FpPairScalar { op, fmt, rd, rn } => {
            let n = cpu.vreg(rn);
            let e = fmt.bits();
            let x = fp_bin(cx.env, op, fmt, lane(n, e, 0), lane(n, e, 1), 0);
            cpu.set_vreg(rd, u128::from(x));
        }
        SimdOp::AddpScalar { rd, rn } => {
            let n = cpu.vreg(rn);
            let x = lane(n, 64, 0).wrapping_add(lane(n, 64, 1));
            cpu.set_vreg(rd, u128::from(x));
        }
        SimdOp::FpUn {
            op,
            fmt,
            lanes,
            rd,
            rn,
        } => {
            let a = cpu.vreg(rn);
            let e = fmt.bits();
            let mut r = 0u128;
            for i in 0..u32::from(lanes) {
                let x = fp_un(cx.env, op, fmt, lane(a, e, i));
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::FpToInt {
            mode,
            signed,
            fmt,
            lanes,
            rd,
            rn,
        } => {
            let a = cpu.vreg(rn);
            let e = fmt.bits();
            let mut r = 0u128;
            for i in 0..u32::from(lanes) {
                let x = fp::to_int(cx.env, fmt, lane(a, e, i), e, signed, 0, mode);
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::IntToFp {
            signed,
            fmt,
            lanes,
            rd,
            rn,
        } => {
            let a = cpu.vreg(rn);
            let e = fmt.bits();
            let mut r = 0u128;
            for i in 0..u32::from(lanes) {
                let x = fp::from_int(cx.env, fmt, lane(a, e, i), e, signed, 0);
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::FpFixed {
            to_int,
            signed,
            fmt,
            lanes,
            fbits,
            rd,
            rn,
        } => {
            let a = cpu.vreg(rn);
            let e = fmt.bits();
            let mut r = 0u128;
            for i in 0..u32::from(lanes) {
                let x = if to_int {
                    fp::to_int(
                        cx.env,
                        fmt,
                        lane(a, e, i),
                        e,
                        signed,
                        u32::from(fbits),
                        Rounding::Zero,
                    )
                } else {
                    fp::from_int(cx.env, fmt, lane(a, e, i), e, signed, u32::from(fbits))
                };
                set_lane(&mut r, e, i, x);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::FpNarrow {
            fmt,
            odd,
            upper,
            scalar,
            rd,
            rn,
        } => {
            let a = cpu.vreg(rn);
            let from = if fmt == Fmt::H { Fmt::S } else { Fmt::D };
            let lanes = if scalar { 1 } else { 64 / fmt.bits() };
            let mut half = 0u128;
            for i in 0..lanes {
                let x = lane(a, from.bits(), i);
                let y = if odd {
                    fp::convert_with(cx.env, from, fmt, x, Rounding::Odd)
                } else {
                    fp::convert(cx.env, from, fmt, x)
                };
                set_lane(&mut half, fmt.bits(), i, y);
            }
            if scalar {
                cpu.set_vreg(rd, half);
            } else {
                write_half(cpu, rd, upper, half as u64);
            }
        }
        SimdOp::FpWiden { fmt, upper, rd, rn } => {
            let from = if fmt == Fmt::S { Fmt::H } else { Fmt::S };
            let a = narrow_part(cpu.vreg(rn), upper);
            let mut r = 0u128;
            for i in 0..64 / from.bits() {
                let y = fp::convert(cx.env, from, fmt, lane(a, from.bits(), i));
                set_lane(&mut r, fmt.bits(), i, y);
            }
            cpu.set_vreg(rd, r);
        }
        SimdOp::FpAcross { op, rd, rn } => {
            let a = cpu.vreg(rn);
            let mm = min_max_of(op);
            let lo = fp::min_max(cx.env, Fmt::S, mm, lane(a, 32, 0), lane(a, 32, 1));
            let hi = fp::min_max(cx.env, Fmt::S, mm, lane(a, 32, 2), lane(a, 32, 3));
            let r = fp::min_max(cx.env, Fmt::S, mm, lo, hi);
            cpu.set_vreg(rd, u128::from(r));
        }
    }
    0
}

/// A shift by an immediate on one element.
fn shift_imm(op: Shift, e: u32, s: u32, a: u64, d: u64, sat: &mut bool) -> u64 {
    let m = mask(e);
    let (ua, sa) = (i128::from(a), i128::from(sext(a, e)));
    match op {
        Shift::Sshr => (sa >> s) as u64 & m,
        Shift::Ushr => (ua >> s) as u64 & m,
        Shift::Ssra => d.wrapping_add((sa >> s) as u64) & m,
        Shift::Usra => d.wrapping_add((ua >> s) as u64) & m,
        Shift::Srshr => round_shift_right(sa, s) as u64 & m,
        Shift::Urshr => round_shift_right(ua, s) as u64 & m,
        Shift::Srsra => d.wrapping_add(round_shift_right(sa, s) as u64) & m,
        Shift::Ursra => d.wrapping_add(round_shift_right(ua, s) as u64) & m,
        Shift::Sri => {
            let keep = m.checked_shr(s).unwrap_or(0);
            (d & !keep) | ((ua >> s) as u64 & keep)
        }
        Shift::Shl => ((ua << s) as u64) & m,
        Shift::Sli => {
            let keep = (m << s) & m;
            (d & !keep) | (((ua << s) as u64) & keep)
        }
        Shift::Sqshlu => {
            let (r, s2) = sat_unsigned(sa << s, e);
            *sat |= s2;
            r
        }
        Shift::Sqshl => {
            let (r, s2) = sat_signed(sa << s, e);
            *sat |= s2;
            r
        }
        Shift::Uqshl => {
            let (r, s2) = sat_unsigned(ua << s, e);
            *sat |= s2;
            r
        }
    }
}

/// Operations of the three-different group on 64-bit halves. `n` and `m` are the full source
/// registers; `by_elem` replaces the `m` elements by one value.
#[allow(clippy::too_many_arguments)]
fn long_op(
    cpu: &mut CpuState,
    cx: &mut Ctx,
    op: Long,
    sh: Shape,
    upper: bool,
    rd: u8,
    n: u128,
    m: u128,
    d: u128,
    by_elem: Option<u64>,
) {
    let e = u32::from(sh.esize);
    let lanes = u32::from(sh.lanes);
    let w = 2 * e;
    let (nn, mm) = (narrow_part(n, upper), narrow_part(m, upper));
    let b_at = |i: u32| by_elem.unwrap_or_else(|| lane(mm, e, i));
    let mut r = 0u128;
    match op {
        Long::Pmull => {
            if e == 64 {
                let (a, b) = if upper {
                    ((n >> 64) as u64, (m >> 64) as u64)
                } else {
                    (n as u64, m as u64)
                };
                r = poly_mul(a, b);
            } else {
                for i in 0..lanes {
                    let p = poly_mul(lane(nn, e, i), lane(mm, e, i));
                    set_lane(&mut r, w, i, p as u64);
                }
            }
            cpu.set_vreg(rd, r);
        }
        Long::Addhn(round) | Long::Subhn(round) => {
            // Wide sources, narrow high-half result.
            let sub = matches!(op, Long::Subhn(_));
            let mut half = 0u128;
            for i in 0..lanes {
                let (a, b) = (lane(n, w, i), lane(m, w, i));
                let rc = if round { 1u128 << (e - 1) } else { 0 };
                let mut v = if sub {
                    u128::from(a).wrapping_sub(u128::from(b))
                } else {
                    u128::from(a) + u128::from(b)
                };
                v = v.wrapping_add(rc) & u128::from(mask(w));
                set_lane(&mut half, e, i, (v >> e) as u64);
            }
            write_half(cpu, rd, upper, half as u64);
        }
        _ => {
            for i in 0..lanes {
                let (signed, a, b, acc) = match op {
                    Long::Addl(s) | Long::Subl(s) | Long::Abal(s) | Long::Abdl(s) => {
                        (s, lane(nn, e, i), b_at(i), lane(d, w, i))
                    }
                    Long::Addw(s) | Long::Subw(s) => (s, lane(n, w, i), b_at(i), 0),
                    Long::Mlal(s) | Long::Mlsl(s) | Long::Mull(s) => {
                        (s, lane(nn, e, i), b_at(i), lane(d, w, i))
                    }
                    _ => (true, lane(nn, e, i), b_at(i), lane(d, w, i)),
                };
                let (ea, eb) = (extend(a, e, signed), extend(b, e, signed));
                let v: u64 = match op {
                    Long::Addl(_) => (ea + eb) as u64,
                    Long::Subl(_) => (ea - eb) as u64,
                    Long::Addw(_) => a.wrapping_add(eb as u64),
                    Long::Subw(_) => a.wrapping_sub(eb as u64),
                    Long::Abal(_) => acc.wrapping_add((ea - eb).unsigned_abs() as u64),
                    Long::Abdl(_) => (ea - eb).unsigned_abs() as u64,
                    Long::Mlal(_) => acc.wrapping_add((ea * eb) as u64),
                    Long::Mlsl(_) => acc.wrapping_sub((ea * eb) as u64),
                    Long::Mull(_) => (ea * eb) as u64,
                    Long::Sqdmull | Long::Sqdmlal | Long::Sqdmlsl => {
                        let (prod, s1) = sat_signed(2 * ea * eb, w);
                        cx.sat |= s1;
                        match op {
                            Long::Sqdmull => prod,
                            Long::Sqdmlal => {
                                let (v, s2) = sat_signed(
                                    i128::from(sext(acc, w)) + i128::from(sext(prod, w)),
                                    w,
                                );
                                cx.sat |= s2;
                                v
                            }
                            _ => {
                                let (v, s2) = sat_signed(
                                    i128::from(sext(acc, w)) - i128::from(sext(prod, w)),
                                    w,
                                );
                                cx.sat |= s2;
                                v
                            }
                        }
                    }
                    Long::Addhn(_) | Long::Subhn(_) | Long::Pmull => unreachable!(),
                };
                set_lane(&mut r, w, i, v);
            }
            cpu.set_vreg(rd, r);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Floating point lanes
// ---------------------------------------------------------------------------------------------

fn fp_bin(env: &mut Env, op: FpBin, fmt: Fmt, a: u64, b: u64, d: u64) -> u64 {
    let ones = fmt.mask();
    let truth = |c: bool| if c { ones } else { 0 };
    match op {
        FpBin::Add => fp::add(env, fmt, a, b),
        FpBin::Sub => fp::sub(env, fmt, a, b),
        FpBin::Mul => fp::mul(env, fmt, a, b),
        FpBin::Div => fp::div(env, fmt, a, b),
        FpBin::Max => fp::min_max(env, fmt, fp::MinMax::Max, a, b),
        FpBin::Min => fp::min_max(env, fmt, fp::MinMax::Min, a, b),
        FpBin::MaxNum => fp::min_max(env, fmt, fp::MinMax::MaxNum, a, b),
        FpBin::MinNum => fp::min_max(env, fmt, fp::MinMax::MinNum, a, b),
        FpBin::Mulx => fp::mul_extended(env, fmt, a, b),
        FpBin::Abd => {
            let diff = fp::sub(env, fmt, a, b);
            fp::abs(fmt, diff)
        }
        FpBin::Recps => fp::recip_step(env, fmt, a, b),
        FpBin::Rsqrts => fp::rsqrt_step(env, fmt, a, b),
        FpBin::Mla => fp::mul_add(env, fmt, d, a, b),
        FpBin::Mls => fp::mul_add(env, fmt, d, fp::neg(fmt, a), b),
        FpBin::Cmeq => truth(fp::compare_order(env, fmt, a, b, false) == Some(Ordering::Equal)),
        FpBin::Cmge => truth(matches!(
            fp::compare_order(env, fmt, a, b, true),
            Some(Ordering::Greater | Ordering::Equal)
        )),
        FpBin::Cmgt => truth(fp::compare_order(env, fmt, a, b, true) == Some(Ordering::Greater)),
        FpBin::Acge => truth(matches!(
            fp::compare_order(env, fmt, fp::abs(fmt, a), fp::abs(fmt, b), true),
            Some(Ordering::Greater | Ordering::Equal)
        )),
        FpBin::Acgt => truth(
            fp::compare_order(env, fmt, fp::abs(fmt, a), fp::abs(fmt, b), true)
                == Some(Ordering::Greater),
        ),
    }
}

fn fp_un(env: &mut Env, op: FpUn, fmt: Fmt, a: u64) -> u64 {
    let ones = fmt.mask();
    let truth = |c: bool| if c { ones } else { 0 };
    let zero = fmt.zero(false);
    match op {
        FpUn::Abs => fp::abs(fmt, a),
        FpUn::Neg => fp::neg(fmt, a),
        FpUn::Sqrt => fp::sqrt(env, fmt, a),
        FpUn::Recpe => fp::recip_estimate(env, fmt, a),
        FpUn::Rsqrte => fp::rsqrt_estimate(env, fmt, a),
        FpUn::Recpx => fp::recip_exponent(env, fmt, a),
        FpUn::Round { mode, exact } => {
            let mode = mode.unwrap_or_else(|| env.rounding());
            fp::round_int(env, fmt, a, mode, exact)
        }
        FpUn::Cmgt0 => truth(fp::compare_order(env, fmt, a, zero, true) == Some(Ordering::Greater)),
        FpUn::Cmge0 => truth(matches!(
            fp::compare_order(env, fmt, a, zero, true),
            Some(Ordering::Greater | Ordering::Equal)
        )),
        FpUn::Cmeq0 => truth(fp::compare_order(env, fmt, a, zero, false) == Some(Ordering::Equal)),
        FpUn::Cmle0 => truth(matches!(
            fp::compare_order(env, fmt, zero, a, true),
            Some(Ordering::Greater | Ordering::Equal)
        )),
        FpUn::Cmlt0 => truth(fp::compare_order(env, fmt, zero, a, true) == Some(Ordering::Greater)),
    }
}
