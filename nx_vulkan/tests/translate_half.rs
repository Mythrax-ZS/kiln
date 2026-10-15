//! The half-precision instructions against the GPU: two 16-bit floats in a register, added,
//! multiplied, fused, compared. The results are compared with what plain Rust arithmetic gives on
//! the same halves, rounded the same way (docs/ASSUMPTIONS.md A-161).

mod common;

use common::{random_states, run, vulkan, Rng, State};
use nx_shader::asm::{exit, Enc, P0};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;

fn half_to_f32(h: u16) -> f32 {
    let sign = if h >> 15 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1F);
    let frac = f32::from(h & 0x3FF);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => {
            if frac == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// The nearest half, ties to even.
fn f32_to_half(f: f32) -> u16 {
    let sign = if f.is_sign_negative() { 0x8000u16 } else { 0 };
    if f.is_nan() {
        return sign | 0x7E00;
    }
    let abs = f.abs();
    if abs >= 65520.0 {
        return sign | 0x7C00;
    }
    if abs < 2f32.powi(-14) {
        return sign | (abs * 2f32.powi(24)).round_ties_even() as u16;
    }
    let mut e = ((abs.to_bits() >> 23) & 0xFF) as i32 - 127;
    let m = abs / 2f32.powi(e);
    let mut frac = ((m - 1.0) * 1024.0).round_ties_even() as u32;
    if frac == 1024 {
        e += 1;
        frac = 0;
    }
    if e + 15 >= 31 {
        return sign | 0x7C00;
    }
    sign | (((e + 15) as u16) << 10) | frac as u16
}

fn halves(word: u32) -> (f32, f32) {
    (half_to_f32(word as u16), half_to_f32((word >> 16) as u16))
}

fn pack(lo: f32, hi: f32) -> u32 {
    u32::from(f32_to_half(lo)) | u32::from(f32_to_half(hi)) << 16
}

/// A register of two finite halves of moderate size, each of either sign.
fn random_half_pair(rng: &mut Rng) -> u32 {
    let one = |rng: &mut Rng| {
        let r = rng.next();
        let sign = (r & 1) << 15;
        let exp = 8 + (r >> 1) % 14;
        let frac = (r >> 8) & 0x3FF;
        sign | exp << 10 | frac
    };
    one(rng) | one(rng) << 16
}

/// States whose registers R1 to R3 and R8 hold half pairs.
fn half_states(n: usize, seed: u64) -> Vec<State> {
    let mut rng = Rng(seed | 1);
    random_states(n, seed)
        .into_iter()
        .map(|mut s| {
            for r in [1usize, 2, 3, 8] {
                s.regs[r] = random_half_pair(&mut rng);
            }
            s
        })
        .collect()
}

/// The lanes a source gives for a swizzle code (0 one float for both, 1 both halves, 2 the low
/// half twice, 3 the high half twice).
fn lanes(word: u32, swizzle: u64) -> (f32, f32) {
    let (lo, hi) = halves(word);
    match swizzle {
        0 => (f32::from_bits(word), f32::from_bits(word)),
        1 => (lo, hi),
        2 => (lo, lo),
        _ => (hi, hi),
    }
}

fn modified(v: (f32, f32), neg: bool, abs: bool) -> (f32, f32) {
    let m = |x: f32| {
        let x = if abs { x.abs() } else { x };
        if neg {
            -x
        } else {
            x
        }
    };
    (m(v.0), m(v.1))
}

fn saturate(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// How the result goes into the register: 0 both halves, 1 one float, 2 and 3 one half.
fn merged(old: u32, lo: f32, hi: f32, merge: u64) -> u32 {
    match merge {
        0 => pack(lo, hi),
        1 => half_to_f32(f32_to_half(lo)).to_bits(),
        2 => (old & 0xFFFF_0000) | u32::from(f32_to_half(lo)),
        _ => (old & 0x0000_FFFF) | u32::from(f32_to_half(hi)) << 16,
    }
}

#[test]
fn hadd2_adds_the_halves_in_each_arrangement() {
    let Some(vk) = vulkan() else { return };
    // Register form: R4 = R1 + R2; R5 = R1 - R2; R6 = |R1.lo| (twice) + R2.hi (twice) clamped to
    // 0..1; R7 = R1 + R2 into the low half only (R7 keeps its high half); R9 = R1 + R2 as a float.
    let hadd = |rd: u8| {
        Enc::op(Op::Hadd2Reg)
            .rd(rd)
            .ra(1)
            .rb(2)
            .field(47, 2, 1)
            .field(28, 2, 1)
    };
    let program = [
        hadd(4).build(),
        hadd(5).bit(31, true).build(),
        hadd(6)
            .field(47, 2, 2)
            .field(28, 2, 3)
            .bit(44, true)
            .bit(32, true)
            .build(),
        hadd(7).field(49, 2, 2).build(),
        hadd(9).field(49, 2, 1).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = half_states(256, 31);
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let a = lanes(inp.regs[1], 1);
        let b = lanes(inp.regs[2], 1);
        assert_eq!(got.regs[4], pack(a.0 + b.0, a.1 + b.1), "plain, input {i}");
        assert_eq!(
            got.regs[5],
            pack(a.0 - b.0, a.1 - b.1),
            "negated b, input {i}"
        );
        let a = modified(lanes(inp.regs[1], 2), false, true);
        let b = lanes(inp.regs[2], 3);
        assert_eq!(
            got.regs[6],
            pack(saturate(a.0 + b.0), saturate(a.1 + b.1)),
            "swizzled, |a|, saturated, input {i}"
        );
        let a = lanes(inp.regs[1], 1);
        let b = lanes(inp.regs[2], 1);
        assert_eq!(
            got.regs[7],
            merged(inp.regs[7], a.0 + b.0, a.1 + b.1, 2),
            "merged into the low half, input {i}"
        );
        assert_eq!(
            got.regs[9],
            merged(inp.regs[9], a.0 + b.0, a.1 + b.1, 1),
            "as one float, input {i}"
        );
    }
}

#[test]
fn hadd2_takes_an_immediate_and_a_constant() {
    let Some(vk) = vulkan() else { return };
    // The idiom of a compiler: under P0, R4's low half becomes 0 + 1.0 (the immediate holds 1.0 in
    // both lanes: nine bits of each half at bits 20 and 30), R4's high half is kept.
    // R5 = R1 + c[0][0] as one float promoted to both lanes.
    let program = [
        Enc::op(Op::Hadd2)
            .rd(4)
            .ra(RZ)
            .field(47, 2, 2)
            .field(49, 2, 2)
            .field(20, 9, 0xF0)
            .field(30, 9, 0xF0)
            .guard(P0)
            .build(),
        Enc::op(Op::Hadd2Cbuf)
            .rd(5)
            .ra(1)
            .field(47, 2, 1)
            .cbuf(0, 0)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = half_states(64, 41);
    let constant = 0.40625f32;
    let out = run(&vk, &program, &inputs, &[constant.to_bits()]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let p0 = inp.preds & 1 != 0;
        let expect4 = if p0 {
            (inp.regs[4] & 0xFFFF_0000) | 0x3C00
        } else {
            inp.regs[4]
        };
        assert_eq!(got.regs[4], expect4, "immediate under P0, input {i}");
        let a = lanes(inp.regs[1], 1);
        assert_eq!(
            got.regs[5],
            pack(a.0 + constant, a.1 + constant),
            "constant buffer word, input {i}"
        );
    }
}

#[test]
fn hmul2_and_hfma2_multiply_and_fuse() {
    let Some(vk) = vulkan() else { return };
    // R4 = R1 * R2; R5 = R1 * -R2 (modifier on b); R6 = R1 * R2 + R3; R7 = R1 * R2 - R3 clamped.
    let program = [
        Enc::op(Op::Hmul2Reg)
            .rd(4)
            .ra(1)
            .rb(2)
            .field(47, 2, 1)
            .field(28, 2, 1)
            .build(),
        Enc::op(Op::Hmul2Reg)
            .rd(5)
            .ra(1)
            .rb(2)
            .field(47, 2, 1)
            .field(28, 2, 1)
            .bit(31, true)
            .build(),
        Enc::op(Op::Hfma2Reg)
            .rd(6)
            .ra(1)
            .rb(2)
            .rc(3)
            .field(47, 2, 1)
            .field(28, 2, 1)
            .field(35, 2, 1)
            .build(),
        Enc::op(Op::Hfma2Reg)
            .rd(7)
            .ra(1)
            .rb(2)
            .rc(3)
            .field(47, 2, 1)
            .field(28, 2, 1)
            .field(35, 2, 1)
            .bit(30, true)
            .bit(32, true)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = half_states(256, 51);
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let (a, b, c) = (
            halves(inp.regs[1]),
            halves(inp.regs[2]),
            halves(inp.regs[3]),
        );
        assert_eq!(got.regs[4], pack(a.0 * b.0, a.1 * b.1), "hmul2, input {i}");
        assert_eq!(
            got.regs[5],
            pack(a.0 * -b.0, a.1 * -b.1),
            "hmul2 with -b, input {i}"
        );
        let fma = |x: f32, y: f32, z: f32| x.mul_add(y, z);
        assert_eq!(
            got.regs[6],
            pack(fma(a.0, b.0, c.0), fma(a.1, b.1, c.1)),
            "hfma2, input {i}"
        );
        assert_eq!(
            got.regs[7],
            pack(saturate(fma(a.0, b.0, -c.0)), saturate(fma(a.1, b.1, -c.1))),
            "hfma2 with -c, saturated, input {i}"
        );
    }
}

/// The result of the comparison with the 4-bit code of `FSETP` (no NaNs in these inputs).
fn compare(code: u64, a: f32, b: f32) -> bool {
    match code {
        0 => false,
        1 | 9 => a < b,
        2 | 10 => a == b,
        3 | 11 => a <= b,
        4 | 12 => a > b,
        5 | 13 => a != b,
        6 | 14 => a >= b,
        7 => true,
        8 => false,
        _ => true,
    }
}

fn combine(bop: u64, c: bool, p: bool) -> bool {
    match bop {
        0 => c && p,
        1 => c || p,
        _ => c != p,
    }
}

#[test]
fn hsetp2_and_hset2_compare_the_lanes() {
    let Some(vk) = vulkan() else { return };
    // Some inputs are equal in a lane so that the equality codes matter.
    let inputs = {
        let mut v = half_states(128, 61);
        for (i, s) in v.iter_mut().enumerate() {
            if i % 3 == 0 {
                s.regs[2] = (s.regs[2] & 0xFFFF_0000) | (s.regs[1] & 0xFFFF);
            }
            if i % 5 == 0 {
                s.regs[2] = (s.regs[2] & 0xFFFF) | (s.regs[1] & 0xFFFF_0000);
            }
        }
        v
    };
    for code in 0..16u64 {
        for bop in 0..3u64 {
            for h_and in [false, true] {
                // P0, P1 = the lanes (or both lanes anded, and its opposite) of R1 against R2,
                // combined with P2; R4 = the same comparison as the lanes' masks.
                let program = [
                    Enc::op(Op::Hsetp2Reg)
                        .ra(1)
                        .rb(2)
                        .field(3, 3, 0)
                        .field(0, 3, 1)
                        .pred(39, 2, false)
                        .field(45, 2, bop)
                        .field(35, 4, code)
                        .field(47, 2, 1)
                        .field(28, 2, 1)
                        .bit(49, h_and)
                        .build(),
                    Enc::op(Op::Hset2Reg)
                        .rd(4)
                        .ra(1)
                        .rb(2)
                        .pred(39, 2, false)
                        .field(45, 2, bop)
                        .field(35, 4, code)
                        .field(47, 2, 1)
                        .field(28, 2, 1)
                        .bit(49, false)
                        .build(),
                    Enc::op(Op::Hset2Reg)
                        .rd(5)
                        .ra(1)
                        .rb(2)
                        .pred(39, 2, false)
                        .field(45, 2, bop)
                        .field(35, 4, code)
                        .field(47, 2, 1)
                        .field(28, 2, 1)
                        .bit(49, true)
                        .build(),
                    exit(Guard::ALWAYS),
                ];
                let out = run(&vk, &program, &inputs, &[]);
                for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
                    let (a, b) = (halves(inp.regs[1]), halves(inp.regs[2]));
                    let p2 = inp.preds & 4 != 0;
                    let lo = combine(bop, compare(code, a.0, b.0), p2);
                    let hi = combine(bop, compare(code, a.1, b.1), p2);
                    let (first, second) = if h_and {
                        (lo && hi, !(lo && hi))
                    } else {
                        (lo, hi)
                    };
                    let msg = format!("code {code} bop {bop} and {h_and} input {i}");
                    assert_eq!(got.preds & 1 != 0, first, "P0, {msg}");
                    assert_eq!(got.preds & 2 != 0, second, "P1, {msg}");
                    let mask = |on: bool, true_value: u32| if on { true_value } else { 0 };
                    assert_eq!(
                        got.regs[4],
                        mask(lo, 0xFFFF) | mask(hi, 0xFFFF_0000),
                        "hset2 mask, {msg}"
                    );
                    assert_eq!(
                        got.regs[5],
                        mask(lo, 0x3C00) | mask(hi, 0x3C00_0000),
                        "hset2 as 1.0, {msg}"
                    );
                }
            }
        }
    }
}
