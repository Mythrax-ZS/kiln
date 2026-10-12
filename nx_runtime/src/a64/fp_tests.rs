//! Tests of the soft-float core.
//!
//! The host's SSE unit is a second implementation of IEEE 754 to compare against (inline assembly
//! sets the rounding mode and reads the exception flags). It agrees with the Arm pseudocode on
//! every finite result and flag except tininess (x86 detects it after rounding, Arm before), so
//! that one flag is compared with a stated exception. NaN payloads, the default NaN, flushing and
//! the saturating conversions are where the two differ on purpose and are tested with directed
//! vectors.

use super::fp::*;
use crate::testing::Rng;
use std::arch::asm;

// ---- the hardware reference -------------------------------------------------------------------

/// x86 rounding-control bits for an Arm rounding mode.
fn rc(r: Rounding) -> u32 {
    match r {
        Rounding::Nearest => 0,
        Rounding::MinusInf => 1,
        Rounding::PlusInf => 2,
        Rounding::Zero => 3,
        _ => unreachable!("no x86 equivalent"),
    }
}

/// Arm flag bits from x86 MXCSR exception bits.
fn arm_flags(csr: u32) -> u32 {
    let mut f = 0;
    if csr & 1 != 0 {
        f |= IOC;
    }
    if csr & 4 != 0 {
        f |= DZC;
    }
    if csr & 8 != 0 {
        f |= OFC;
    }
    if csr & 16 != 0 {
        f |= UFC;
    }
    if csr & 32 != 0 {
        f |= IXC;
    }
    f
}

#[derive(Clone, Copy, Debug)]
enum Op2 {
    Add,
    Sub,
    Mul,
    Div,
}

fn hw64(op: Op2, a: u64, b: u64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let mut x = f64::from_bits(a);
    let y = f64::from_bits(b);
    // SAFETY: plain SSE instructions on registers; MXCSR is restored to its default afterwards.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        match op {
            Op2::Add => {
                asm!("addsd {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
            Op2::Sub => {
                asm!("subsd {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
            Op2::Mul => {
                asm!("mulsd {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
            Op2::Div => {
                asm!("divsd {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
        }
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (x.to_bits(), arm_flags(csr_out))
}

fn hw32(op: Op2, a: u64, b: u64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let mut x = f32::from_bits(a as u32);
    let y = f32::from_bits(b as u32);
    // SAFETY: as above.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        match op {
            Op2::Add => {
                asm!("addss {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
            Op2::Sub => {
                asm!("subss {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
            Op2::Mul => {
                asm!("mulss {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
            Op2::Div => {
                asm!("divss {x}, {y}", x = inout(xmm_reg) x, y = in(xmm_reg) y, options(nostack, nomem))
            }
        }
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (u64::from(x.to_bits()), arm_flags(csr_out))
}

fn hw_sqrt64(a: u64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let mut x = f64::from_bits(a);
    // SAFETY: as above.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        asm!("sqrtsd {x}, {x}", x = inout(xmm_reg) x, options(nostack, nomem));
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (x.to_bits(), arm_flags(csr_out))
}

fn hw_sqrt32(a: u64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let mut x = f32::from_bits(a as u32);
    // SAFETY: as above.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        asm!("sqrtss {x}, {x}", x = inout(xmm_reg) x, options(nostack, nomem));
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (u64::from(x.to_bits()), arm_flags(csr_out))
}

/// `a + b * c` fused, on hardware with FMA.
fn hw_fma64(a: u64, b: u64, c: u64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let mut x = f64::from_bits(a);
    let (y, z) = (f64::from_bits(b), f64::from_bits(c));
    // SAFETY: the caller checked the FMA feature.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        // vfmadd231sd x, y, z: x = y * z + x
        asm!("vfmadd231sd {x}, {y}, {z}", x = inout(xmm_reg) x, y = in(xmm_reg) y, z = in(xmm_reg) z, options(nostack, nomem));
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (x.to_bits(), arm_flags(csr_out))
}

fn hw_fma32(a: u64, b: u64, c: u64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let mut x = f32::from_bits(a as u32);
    let (y, z) = (f32::from_bits(b as u32), f32::from_bits(c as u32));
    // SAFETY: as above.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        asm!("vfmadd231ss {x}, {y}, {z}", x = inout(xmm_reg) x, y = in(xmm_reg) y, z = in(xmm_reg) z, options(nostack, nomem));
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (u64::from(x.to_bits()), arm_flags(csr_out))
}

/// Convert double to single with the hardware (`cvtsd2ss`).
fn hw_d2s(a: u64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let x = f64::from_bits(a);
    let r: f32;
    // SAFETY: as above.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        asm!("cvtsd2ss {r}, {x}", r = out(xmm_reg) r, x = in(xmm_reg) x, options(nostack, nomem));
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (u64::from(r.to_bits()), arm_flags(csr_out))
}

/// Signed 64-bit integer to double (`cvtsi2sd`) and single (`cvtsi2ss`).
fn hw_i2d(v: i64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let r: f64;
    // SAFETY: as above.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        asm!("cvtsi2sd {r}, {v}", r = out(xmm_reg) r, v = in(reg) v, options(nostack, nomem));
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (r.to_bits(), arm_flags(csr_out))
}

fn hw_i2s(v: i64, mode: Rounding) -> (u64, u32) {
    let csr_in: u32 = 0x1F80 | (rc(mode) << 13);
    let mut csr_out: u32 = 0;
    let r: f32;
    // SAFETY: as above.
    unsafe {
        asm!("ldmxcsr [{0}]", in(reg) &csr_in, options(nostack));
        asm!("cvtsi2ss {r}, {v}", r = out(xmm_reg) r, v = in(reg) v, options(nostack, nomem));
        asm!("stmxcsr [{0}]", in(reg) &mut csr_out, options(nostack));
        let default: u32 = 0x1F80;
        asm!("ldmxcsr [{0}]", in(reg) &default, options(nostack));
    }
    (u64::from(r.to_bits()), arm_flags(csr_out))
}

/// `roundsd` with an explicit rounding mode; precision exceptions suppressed.
fn hw_roundsd(a: u64, mode: Rounding) -> u64 {
    let x = f64::from_bits(a);
    let r: f64;
    // SAFETY: SSE4.1, available on every machine these tests run on.
    unsafe {
        match mode {
            Rounding::Nearest => {
                asm!("roundsd {r}, {x}, 8", r = out(xmm_reg) r, x = in(xmm_reg) x, options(nostack, nomem))
            }
            Rounding::MinusInf => {
                asm!("roundsd {r}, {x}, 9", r = out(xmm_reg) r, x = in(xmm_reg) x, options(nostack, nomem))
            }
            Rounding::PlusInf => {
                asm!("roundsd {r}, {x}, 10", r = out(xmm_reg) r, x = in(xmm_reg) x, options(nostack, nomem))
            }
            Rounding::Zero => {
                asm!("roundsd {r}, {x}, 11", r = out(xmm_reg) r, x = in(xmm_reg) x, options(nostack, nomem))
            }
            _ => unreachable!(),
        }
    }
    r.to_bits()
}

// ---- random operands --------------------------------------------------------------------------

fn rand64(rng: &mut Rng) -> u64 {
    let sign = rng.below(2) << 63;
    match rng.below(14) {
        0 => rng.next_u64(),
        1 => sign | (rng.below(2047) << 52) | rng.below(1 << 52),
        2 => sign,                                                              // zero
        3 => sign | 0x7FF0_0000_0000_0000,                                      // infinity
        4 => sign | (rng.below(8) << 52) | rng.below(1 << 52),                  // tiny and denormal
        5 => sign | ((2046 - rng.below(8)) << 52) | rng.below(1 << 52),         // near overflow
        6 => sign | ((1023 + rng.below(40) - 20) << 52) | rng.below(1 << 52),   // around 1
        7 => sign | ((1023 + rng.below(40) - 20) << 52), // exact powers of two
        8 => sign | ((1023 + rng.below(40) - 20) << 52) | ((1 << 52) - 1), // all-ones fraction
        9 => sign | ((1023 + rng.below(4)) << 52) | (rng.below(1 << 12) << 40), // few-bit fraction
        10 => sign | 1,                                  // smallest denormal
        11 => sign | 0x000F_FFFF_FFFF_FFFF,              // largest denormal
        12 => sign | 0x0010_0000_0000_0000,              // smallest normal
        _ => sign | ((1023 + rng.below(100) - 50) << 52) | rng.below(1 << 52),
    }
}

fn rand32(rng: &mut Rng) -> u64 {
    let sign = rng.below(2) << 31;
    match rng.below(14) {
        0 => rng.next_u32() as u64,
        1 => sign | (rng.below(255) << 23) | rng.below(1 << 23),
        2 => sign,
        3 => sign | 0x7F80_0000,
        4 => sign | (rng.below(4) << 23) | rng.below(1 << 23),
        5 => sign | ((254 - rng.below(4)) << 23) | rng.below(1 << 23),
        6 => sign | ((127 + rng.below(20) - 10) << 23) | rng.below(1 << 23),
        7 => sign | ((127 + rng.below(20) - 10) << 23),
        8 => sign | ((127 + rng.below(20) - 10) << 23) | ((1 << 23) - 1),
        9 => sign | ((127 + rng.below(4)) << 23) | (rng.below(1 << 8) << 15),
        10 => sign | 1,
        11 => sign | 0x007F_FFFF,
        12 => sign | 0x0080_0000,
        _ => sign | ((127 + rng.below(60) - 30) << 23) | rng.below(1 << 23),
    }
}

const MODES: [Rounding; 4] = [
    Rounding::Nearest,
    Rounding::PlusInf,
    Rounding::MinusInf,
    Rounding::Zero,
];

fn fpcr_for(mode: Rounding) -> u32 {
    match mode {
        Rounding::Nearest => 0,
        Rounding::PlusInf => 1 << 22,
        Rounding::MinusInf => 2 << 22,
        Rounding::Zero => 3 << 22,
        _ => unreachable!(),
    }
}

/// Compare an Arm result with the hardware's. NaNs only need to be NaNs (the hardware's default
/// NaN has the sign bit set). Tininess is detected before rounding on Arm and after on x86, so a
/// disagreement on `UFC` is allowed when the result is exactly the smallest normal number.
fn check(fmt: Fmt, what: &str, arm: (u64, u32), hw: (u64, u32), inputs: &[u64]) {
    let (r, flags) = arm;
    let (hr, hflags) = hw;
    if fmt.is_nan(hr) {
        assert!(
            fmt.is_nan(r),
            "{what}: {inputs:x?}: hardware gave a NaN, got {r:#x}"
        );
    } else {
        assert_eq!(r, hr, "{what}: {inputs:x?}: result");
    }
    let mut a = flags;
    let mut h = hflags;
    let min_normal = 1u64 << fmt.frac();
    if (a ^ h) & UFC != 0 && (r & !fmt.sign_bit_for_tests()) == min_normal {
        a &= !UFC;
        h &= !UFC;
    }
    assert_eq!(a, h, "{what}: {inputs:x?}: flags (result {r:#x})");
}

impl Fmt {
    fn sign_bit_for_tests(self) -> u64 {
        1u64 << (self.bits() - 1)
    }
}

fn no_nan(fmt: Fmt, v: &[u64]) -> bool {
    v.iter().all(|&x| !fmt.is_nan(x))
}

#[test]
fn add_sub_mul_div_match_the_hardware_in_every_rounding_mode() {
    let mut rng = Rng::new(0xF10A7);
    for mode in MODES {
        let fpcr = fpcr_for(mode);
        for _ in 0..60_000 {
            let (a, b) = (rand64(&mut rng), rand64(&mut rng));
            if no_nan(Fmt::D, &[a, b]) {
                for (name, op, f) in [
                    ("add", Op2::Add, add as fn(&mut Env, Fmt, u64, u64) -> u64),
                    ("sub", Op2::Sub, sub),
                    ("mul", Op2::Mul, mul),
                    ("div", Op2::Div, div),
                ] {
                    let mut env = Env::new(fpcr, 0);
                    let r = f(&mut env, Fmt::D, a, b);
                    check(
                        Fmt::D,
                        &format!("{name} f64 {mode:?}"),
                        (r, env.fpsr),
                        hw64(op, a, b, mode),
                        &[a, b],
                    );
                }
            }
            let (a, b) = (rand32(&mut rng), rand32(&mut rng));
            if no_nan(Fmt::S, &[a, b]) {
                for (name, op, f) in [
                    ("add", Op2::Add, add as fn(&mut Env, Fmt, u64, u64) -> u64),
                    ("sub", Op2::Sub, sub),
                    ("mul", Op2::Mul, mul),
                    ("div", Op2::Div, div),
                ] {
                    let mut env = Env::new(fpcr, 0);
                    let r = f(&mut env, Fmt::S, a, b);
                    check(
                        Fmt::S,
                        &format!("{name} f32 {mode:?}"),
                        (r, env.fpsr),
                        hw32(op, a, b, mode),
                        &[a, b],
                    );
                }
            }
        }
    }
}

#[test]
fn square_root_matches_the_hardware() {
    let mut rng = Rng::new(0x5127);
    for mode in MODES {
        let fpcr = fpcr_for(mode);
        for _ in 0..80_000 {
            let a = rand64(&mut rng);
            if !Fmt::D.is_nan(a) {
                let mut env = Env::new(fpcr, 0);
                let r = sqrt(&mut env, Fmt::D, a);
                check(
                    Fmt::D,
                    &format!("sqrt f64 {mode:?}"),
                    (r, env.fpsr),
                    hw_sqrt64(a, mode),
                    &[a],
                );
            }
            let a = rand32(&mut rng);
            if !Fmt::S.is_nan(a) {
                let mut env = Env::new(fpcr, 0);
                let r = sqrt(&mut env, Fmt::S, a);
                check(
                    Fmt::S,
                    &format!("sqrt f32 {mode:?}"),
                    (r, env.fpsr),
                    hw_sqrt32(a, mode),
                    &[a],
                );
            }
        }
    }
}

#[test]
fn fused_multiply_add_matches_the_hardware() {
    if !is_x86_feature_detected!("fma") {
        eprintln!("SKIPPED: this CPU has no FMA");
        return;
    }
    let mut rng = Rng::new(0xF3A);
    for mode in MODES {
        let fpcr = fpcr_for(mode);
        for _ in 0..80_000 {
            let (a, b, c) = (rand64(&mut rng), rand64(&mut rng), rand64(&mut rng));
            if no_nan(Fmt::D, &[a, b, c]) {
                let mut env = Env::new(fpcr, 0);
                let r = mul_add(&mut env, Fmt::D, a, b, c);
                check(
                    Fmt::D,
                    &format!("fma f64 {mode:?}"),
                    (r, env.fpsr),
                    hw_fma64(a, b, c, mode),
                    &[a, b, c],
                );
            }
            let (a, b, c) = (rand32(&mut rng), rand32(&mut rng), rand32(&mut rng));
            if no_nan(Fmt::S, &[a, b, c]) {
                let mut env = Env::new(fpcr, 0);
                let r = mul_add(&mut env, Fmt::S, a, b, c);
                check(
                    Fmt::S,
                    &format!("fma f32 {mode:?}"),
                    (r, env.fpsr),
                    hw_fma32(a, b, c, mode),
                    &[a, b, c],
                );
            }
        }
    }
}

#[test]
fn conversions_between_precisions_and_from_integers_match_the_hardware() {
    let mut rng = Rng::new(0xC0DE);
    for mode in MODES {
        let fpcr = fpcr_for(mode);
        for _ in 0..80_000 {
            let a = rand64(&mut rng);
            if !Fmt::D.is_nan(a) {
                let mut env = Env::new(fpcr, 0);
                let r = convert(&mut env, Fmt::D, Fmt::S, a);
                check(
                    Fmt::S,
                    &format!("fcvt d->s {mode:?}"),
                    (r, env.fpsr),
                    hw_d2s(a, mode),
                    &[a],
                );
            }
            let v = match rng.below(4) {
                0 => rng.next_u64() as i64,
                1 => (rng.next_u64() >> rng.below(64)) as i64,
                2 => -((rng.next_u64() >> rng.below(64)) as i64),
                _ => (1i64 << rng.below(63)) + rng.below(3) as i64 - 1,
            };
            let mut env = Env::new(fpcr, 0);
            let r = from_int(&mut env, Fmt::D, v as u64, 64, true, 0);
            check(
                Fmt::D,
                &format!("scvtf x->d {mode:?}"),
                (r, env.fpsr),
                hw_i2d(v, mode),
                &[v as u64],
            );
            let mut env = Env::new(fpcr, 0);
            let r = from_int(&mut env, Fmt::S, v as u64, 64, true, 0);
            check(
                Fmt::S,
                &format!("scvtf x->s {mode:?}"),
                (r, env.fpsr),
                hw_i2s(v, mode),
                &[v as u64],
            );
        }
    }
    // Single to double is exact.
    for _ in 0..100_000 {
        let a = rand32(&mut rng);
        if Fmt::S.is_nan(a) {
            continue;
        }
        let mut env = Env::new(0, 0);
        let r = convert(&mut env, Fmt::S, Fmt::D, a);
        assert_eq!(r, f64::from(f32::from_bits(a as u32)).to_bits(), "{a:#x}");
        assert_eq!(env.fpsr, 0);
    }
}

#[test]
fn rounding_to_an_integral_value_matches_roundsd() {
    let mut rng = Rng::new(0x20D);
    for mode in MODES {
        for _ in 0..100_000 {
            let a = rand64(&mut rng);
            if Fmt::D.is_nan(a) {
                continue;
            }
            let mut env = Env::new(0, 0);
            let r = round_int(&mut env, Fmt::D, a, mode, false);
            assert_eq!(r, hw_roundsd(a, mode), "frint {mode:?} {a:#x}");
            assert_eq!(env.fpsr, 0, "{a:#x}");
        }
    }
}

#[test]
fn float_to_integer_saturates_and_rounds() {
    let mut rng = Rng::new(0x701);
    for _ in 0..200_000 {
        let a = rand64(&mut rng);
        let x = f64::from_bits(a);
        for mode in [
            Rounding::Zero,
            Rounding::Nearest,
            Rounding::PlusInf,
            Rounding::MinusInf,
        ] {
            let mut env = Env::new(0, 0);
            let r = to_int(&mut env, Fmt::D, a, 32, true, 0, mode) as u32 as i32;
            let expect: i32 = if x.is_nan() {
                0
            } else {
                let rounded = match mode {
                    Rounding::Zero => x.trunc(),
                    Rounding::Nearest => x.round_ties_even(),
                    Rounding::PlusInf => x.ceil(),
                    _ => x.floor(),
                };
                // `as` saturates in Rust, like the Arm conversion.
                rounded as i32
            };
            assert_eq!(r, expect, "fcvtzs-like {mode:?} {a:#x}");
            let invalid = x.is_nan() || x.is_infinite() || {
                let rounded = match mode {
                    Rounding::Zero => x.trunc(),
                    Rounding::Nearest => x.round_ties_even(),
                    Rounding::PlusInf => x.ceil(),
                    _ => x.floor(),
                };
                rounded >= 2147483648.0 || rounded < -2147483648.0
            };
            if invalid {
                assert_eq!(env.fpsr & IOC, IOC, "{a:#x}");
            } else {
                assert_eq!(env.fpsr & IOC, 0, "{a:#x}");
                let exact = x == x.trunc() && x == x.round_ties_even();
                assert_eq!(
                    env.fpsr & IXC != 0,
                    !exact && x.abs() >= 0.0 && x != 0.0 && x != x.trunc()
                        || (x != 0.0 && x != x.trunc()),
                    "{a:#x} {mode:?}"
                );
            }
        }
        // Unsigned and 64-bit.
        let mut env = Env::new(0, 0);
        let r = to_int(&mut env, Fmt::D, a, 64, false, 0, Rounding::Zero);
        let expect = if x.is_nan() { 0 } else { x as u64 };
        assert_eq!(r, expect, "fcvtzu {a:#x}");
    }
    // Fixed-point: 2^4 scaling.
    let mut env = Env::new(0, 0);
    assert_eq!(
        to_int(
            &mut env,
            Fmt::D,
            1.5f64.to_bits(),
            32,
            true,
            4,
            Rounding::Zero
        ),
        24
    );
    // Ties away from zero.
    let mut env = Env::new(0, 0);
    assert_eq!(
        to_int(
            &mut env,
            Fmt::D,
            2.5f64.to_bits(),
            32,
            true,
            0,
            Rounding::NearestAway
        ),
        3
    );
    assert_eq!(
        to_int(
            &mut env,
            Fmt::D,
            (-2.5f64).to_bits(),
            32,
            true,
            0,
            Rounding::NearestAway
        ) as u32 as i32,
        -3
    );
    assert_eq!(
        to_int(
            &mut env,
            Fmt::D,
            2.5f64.to_bits(),
            32,
            true,
            0,
            Rounding::Nearest
        ),
        2
    );
}

#[test]
fn half_precision_round_trips_and_rounds_correctly() {
    // Every half value converts to single exactly, and back.
    for h in 0..=0xFFFFu64 {
        let mut env = Env::new(0, 0);
        let s = convert(&mut env, Fmt::H, Fmt::S, h);
        if Fmt::H.is_nan(h) {
            assert!(Fmt::S.is_nan(s));
            continue;
        }
        // Reference: decode by formula.
        let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
        let e = ((h >> 10) & 0x1F) as i32;
        let f = (h & 0x3FF) as f64;
        let v = if e == 0x1F {
            f64::INFINITY * sign
        } else if e == 0 {
            sign * f * 2f64.powi(-24)
        } else {
            sign * (1.0 + f / 1024.0) * 2f64.powi(e - 15)
        };
        assert_eq!(f32::from_bits(s as u32) as f64, v, "{h:#x}");
        let mut env = Env::new(0, 0);
        assert_eq!(convert(&mut env, Fmt::S, Fmt::H, s), h, "{h:#x}");
        assert_eq!(env.fpsr, 0, "{h:#x}");
    }
    // Rounding to half, against the hardware's F16C conversion.
    if is_x86_feature_detected!("f16c") {
        let mut rng = Rng::new(0xF16);
        for _ in 0..200_000 {
            let a = rand32(&mut rng);
            if Fmt::S.is_nan(a) {
                continue;
            }
            let hw: u32;
            // SAFETY: F16C checked above; round-to-nearest-even.
            unsafe {
                let x = f32::from_bits(a as u32);
                asm!("vcvtps2ph {r}, {x}, 0", r = out(xmm_reg) _, x = in(xmm_reg) x, options(nostack, nomem));
                let r: u32;
                asm!("vcvtps2ph {x}, {x}, 0", "vmovd {r:e}, {x}", x = inout(xmm_reg) x => _, r = out(reg) r, options(nostack, nomem));
                hw = r & 0xFFFF;
            }
            let mut env = Env::new(0, 0);
            let r = convert(&mut env, Fmt::S, Fmt::H, a);
            assert_eq!(r as u32, hw, "single to half {a:#x}");
        }
    }
}

#[test]
fn nans_follow_the_arm_rules() {
    let qnan1 = 0x7FC0_1234u64;
    let qnan2 = 0x7FC0_5678u64;
    let snan1 = 0x7F80_0001u64;
    let snan2 = 0x7F80_0002u64;
    let one = 0x3F80_0000u64;
    let mut env = Env::new(0, 0);
    // One quiet NaN operand: that NaN comes back unchanged, no flags.
    assert_eq!(add(&mut env, Fmt::S, qnan1, one), qnan1);
    assert_eq!(add(&mut env, Fmt::S, one, qnan2), qnan2);
    assert_eq!(env.fpsr, 0);
    // Two quiet NaNs: the first.
    assert_eq!(mul(&mut env, Fmt::S, qnan1, qnan2), qnan1);
    // A signalling NaN is quieted and raises IOC; it wins over a quiet NaN.
    assert_eq!(add(&mut env, Fmt::S, snan1, one), snan1 | 0x0040_0000);
    assert_eq!(env.fpsr, IOC);
    let mut env = Env::new(0, 0);
    assert_eq!(add(&mut env, Fmt::S, qnan1, snan2), snan2 | 0x0040_0000);
    assert_eq!(env.fpsr, IOC);
    // Default-NaN mode returns the default NaN for any NaN.
    let mut env = Env::new(1 << 25, 0);
    assert_eq!(add(&mut env, Fmt::S, qnan1, one), 0x7FC0_0000);
    assert_eq!(env.fpsr, 0);
    let mut env = Env::new(1 << 25, 0);
    assert_eq!(
        sub(&mut env, Fmt::D, 0x7FF0_0000_0000_0001, one),
        0x7FF8_0000_0000_0000
    );
    assert_eq!(env.fpsr, IOC);
    // Invalid operations give the default NaN (positive) and IOC.
    let inf = 0x7F80_0000u64;
    let mut env = Env::new(0, 0);
    assert_eq!(sub(&mut env, Fmt::S, inf, inf), 0x7FC0_0000);
    assert_eq!(env.fpsr, IOC);
    let mut env = Env::new(0, 0);
    assert_eq!(mul(&mut env, Fmt::S, inf, 0), 0x7FC0_0000);
    assert_eq!(env.fpsr, IOC);
    let mut env = Env::new(0, 0);
    assert_eq!(div(&mut env, Fmt::S, 0, 0), 0x7FC0_0000);
    assert_eq!(env.fpsr, IOC);
    let mut env = Env::new(0, 0);
    assert_eq!(sqrt(&mut env, Fmt::S, 0xBF80_0000), 0x7FC0_0000);
    assert_eq!(env.fpsr, IOC);
    // Division by zero: infinity, DZC.
    let mut env = Env::new(0, 0);
    assert_eq!(div(&mut env, Fmt::S, one, 0), inf);
    assert_eq!(env.fpsr, DZC);
    // A quiet NaN addend with an invalid product: the default NaN.
    let mut env = Env::new(0, 0);
    assert_eq!(mul_add(&mut env, Fmt::S, qnan1, inf, 0), 0x7FC0_0000);
    assert_eq!(env.fpsr, IOC);
    // FABS and FNEG touch only the sign, even for NaNs.
    assert_eq!(neg(Fmt::S, snan1), snan1 | 0x8000_0000);
    assert_eq!(abs(Fmt::S, 0xFFC0_0001), 0x7FC0_0001);
}

#[test]
fn flush_to_zero_and_denormals() {
    let fz = 1 << 24;
    let denormal = 0x0000_0001u64; // smallest single denormal
    let one = 0x3F80_0000u64;
    // Without FZ a denormal operand is an ordinary number.
    let mut env = Env::new(0, 0);
    assert_eq!(add(&mut env, Fmt::S, denormal, one), one);
    assert_eq!(env.fpsr, IXC);
    // With FZ it reads as zero and IDC is set.
    let mut env = Env::new(fz, 0);
    assert_eq!(add(&mut env, Fmt::S, denormal, 0), 0);
    assert_eq!(env.fpsr, IDC);
    // A tiny result is flushed to zero with UFC (and no IXC).
    let mut env = Env::new(fz, 0);
    assert_eq!(mul(&mut env, Fmt::S, 0x0080_0000, 0x3F00_0000), 0); // min normal * 0.5
    assert_eq!(env.fpsr, UFC);
    // Without FZ the same product is an exact denormal: no flags.
    let mut env = Env::new(0, 0);
    assert_eq!(mul(&mut env, Fmt::S, 0x0080_0000, 0x3F00_0000), 0x0040_0000);
    assert_eq!(env.fpsr, 0);
    // Half precision never flushes.
    let mut env = Env::new(fz, 0);
    assert_eq!(convert(&mut env, Fmt::H, Fmt::S, 0x0001), 0x3380_0000);
    assert_eq!(env.fpsr, 0);
    // Tininess is detected before rounding: this product rounds up to the smallest normal but
    // is tiny and inexact beforehand.
    let mut env = Env::new(0, 0);
    let r = mul(&mut env, Fmt::S, 0x007F_FFFF, 0x3F80_0001); // just below min normal * (1 + 2^-23)
    let _ = r;
    assert_eq!(
        env.fpsr & UFC != 0,
        env.fpsr & IXC != 0 && r < 0x0080_0000 || r == 0x0080_0000
    );
}

#[test]
fn comparison_minimum_and_maximum() {
    let one = 1.0f32.to_bits() as u64;
    let two = 2.0f32.to_bits() as u64;
    let neg_zero = 0x8000_0000u64;
    let qnan = 0x7FC0_0000u64;
    let snan = 0x7F80_0001u64;
    let mut env = Env::new(0, 0);
    assert_eq!(compare(&mut env, Fmt::S, one, two, false), 0b1000); // less
    assert_eq!(compare(&mut env, Fmt::S, two, one, false), 0b0010); // greater
    assert_eq!(compare(&mut env, Fmt::S, one, one, false), 0b0110); // equal
    assert_eq!(compare(&mut env, Fmt::S, 0, neg_zero, false), 0b0110); // -0 == +0
    assert_eq!(env.fpsr, 0);
    assert_eq!(compare(&mut env, Fmt::S, one, qnan, false), 0b0011); // unordered
    assert_eq!(env.fpsr, 0);
    assert_eq!(compare(&mut env, Fmt::S, one, qnan, true), 0b0011); // FCMPE signals
    assert_eq!(env.fpsr, IOC);
    let mut env = Env::new(0, 0);
    assert_eq!(compare(&mut env, Fmt::S, snan, one, false), 0b0011);
    assert_eq!(env.fpsr, IOC);

    let mut env = Env::new(0, 0);
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::Max, one, two), two);
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::Min, one, two), one);
    // Zeros of opposite sign: max is +0, min is -0.
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::Max, neg_zero, 0), 0);
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::Max, 0, neg_zero), 0);
    assert_eq!(
        min_max(&mut env, Fmt::S, MinMax::Min, 0, neg_zero),
        neg_zero
    );
    // NaNs propagate for FMAX, are ignored for FMAXNM.
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::Max, one, qnan), qnan);
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::MaxNum, one, qnan), one);
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::MinNum, qnan, two), two);
    assert_eq!(min_max(&mut env, Fmt::S, MinMax::MaxNum, qnan, qnan), qnan);
    assert_eq!(env.fpsr, 0);
    let mut env = Env::new(0, 0);
    assert_eq!(
        min_max(&mut env, Fmt::S, MinMax::MaxNum, one, snan),
        snan | 0x0040_0000
    );
    assert_eq!(env.fpsr, IOC);
}

#[test]
fn estimates_and_newton_steps() {
    let mut env = Env::new(0, 0);
    // The reciprocal estimate of 1.0 is 0.998046875 (an 8-bit estimate), of 2.0 half that.
    assert_eq!(recip_estimate(&mut env, Fmt::S, 0x3F80_0000), 0x3F7F_8000);
    assert_eq!(recip_estimate(&mut env, Fmt::S, 0x4000_0000), 0x3EFF_8000);
    // Reciprocal square root estimate of 1.0 is 0.998046875, of 4.0 half that.
    assert_eq!(rsqrt_estimate(&mut env, Fmt::S, 0x3F80_0000), 0x3F7F_8000);
    assert_eq!(rsqrt_estimate(&mut env, Fmt::S, 0x4080_0000), 0x3EFF_8000);
    assert_eq!(env.fpsr, 0);
    // Special values.
    assert_eq!(recip_estimate(&mut env, Fmt::S, 0), 0x7F80_0000);
    assert_eq!(env.fpsr, DZC);
    let mut env = Env::new(0, 0);
    assert_eq!(recip_estimate(&mut env, Fmt::S, 0x7F80_0000), 0);
    assert_eq!(rsqrt_estimate(&mut env, Fmt::S, 0xBF80_0000), 0x7FC0_0000);
    assert_eq!(env.fpsr, IOC);
    // The estimates are within 1/256 of the true value over a sweep of mantissas.
    let mut env = Env::new(0, 0);
    for frac in (0..1 << 23).step_by(997) {
        for exp in [100u64, 127, 128, 150] {
            let a = (exp << 23) | frac;
            let x = f32::from_bits(a as u32) as f64;
            let r = f32::from_bits(recip_estimate(&mut env, Fmt::S, a) as u32) as f64;
            assert!((r * x - 1.0).abs() < 1.0 / 128.0, "recip {a:#x}: {r} {x}");
            let s = f32::from_bits(rsqrt_estimate(&mut env, Fmt::S, a) as u32) as f64;
            assert!(
                (s * s * x - 1.0).abs() < 1.0 / 64.0,
                "rsqrt {a:#x}: {s} {x}"
            );
        }
    }
    // FRECPS(a, b) = 2 - a*b, FRSQRTS(a, b) = (3 - a*b)/2.
    let mut env = Env::new(0, 0);
    let two = 2.0f32.to_bits() as u64;
    let three = 3.0f32.to_bits() as u64;
    let half = 0.5f32.to_bits() as u64;
    assert_eq!(
        recip_step(&mut env, Fmt::S, two, half),
        1.0f32.to_bits() as u64
    );
    assert_eq!(
        rsqrt_step(&mut env, Fmt::S, three, 1.0f32.to_bits() as u64),
        0
    );
    let inf = 0x7F80_0000u64;
    assert_eq!(recip_step(&mut env, Fmt::S, inf, 0), two);
    assert_eq!(
        rsqrt_step(&mut env, Fmt::S, inf, 0),
        1.5f32.to_bits() as u64
    );
    // FRECPX flips the exponent bits: 1.0 (exponent 127) becomes exponent 128, 2.0.
    assert_eq!(recip_exponent(&mut env, Fmt::S, 0x3F80_0000), 0x4000_0000);
    assert_eq!(recip_exponent(&mut env, Fmt::S, 0), 0x7F00_0000);
}
