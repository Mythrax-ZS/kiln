//! Hand-computed tests of the scalar floating-point instructions, with words assembled by LLVM.

use super::tests::{start, vop, Run, DZC, IOC, IXC, OFC, QNAN};
use crate::a64::state::CpuState;
use crate::testing::FlatBus;

fn fs(x: f32) -> u128 {
    u128::from(x.to_bits())
}

fn fd(x: f64) -> u128 {
    u128::from(x.to_bits())
}

const DN: u32 = 1 << 25;
const FZ: u32 = 1 << 24;
const IDC: u32 = 0x80;
const UFC: u32 = 8;
const RMODE_PLUS: u32 = 1 << 22;
const RMODE_MINUS: u32 = 2 << 22;
const RMODE_ZERO: u32 = 3 << 22;

fn with_fpcr(word: u32, regs: &[(u8, u128)], fpcr: u32) -> Run {
    start(word, regs, |c, _| c.fpcr = fpcr)
}

#[test]
fn scalar_arithmetic_clears_the_rest_of_the_register() {
    let wipe = (0, u128::MAX);
    // fadd s0, s1, s2; fsub d0, d1, d2; fmul s0, s1, s2; fdiv d0, d1, d2
    assert_eq!(
        vop(0x1E22_2820, &[wipe, (1, fs(1.5)), (2, fs(2.25))]).v(0),
        fs(3.75)
    );
    assert_eq!(
        vop(0x1E62_3820, &[wipe, (1, fd(5.5)), (2, fd(2.0))]).v(0),
        fd(3.5)
    );
    assert_eq!(
        vop(0x1E22_0820, &[wipe, (1, fs(3.0)), (2, fs(0.5))]).v(0),
        fs(1.5)
    );
    assert_eq!(
        vop(0x1E62_1820, &[wipe, (1, fd(1.0)), (2, fd(4.0))]).v(0),
        fd(0.25)
    );
    // fsqrt s0, s1; fabs s0, s1; fneg d0, d1; fnmul s0, s1, s2
    assert_eq!(vop(0x1E21_C020, &[wipe, (1, fs(2.25))]).v(0), fs(1.5));
    assert_eq!(vop(0x1E20_C020, &[wipe, (1, fs(-2.5))]).v(0), fs(2.5));
    assert_eq!(vop(0x1E61_4020, &[wipe, (1, fd(1.0))]).v(0), fd(-1.0));
    assert_eq!(
        vop(0x1E22_8820, &[wipe, (1, fs(2.0)), (2, fs(3.0))]).v(0),
        fs(-6.0)
    );
}

#[test]
fn fused_multiply_add_variants() {
    // d0 = op(d3, d1 * d2) with d1 = 2, d2 = 3, d3 = 1
    let regs = [(1, fd(2.0)), (2, fd(3.0)), (3, fd(1.0))];
    assert_eq!(vop(0x1F42_0C20, &regs).v(0), fd(7.0)); // fmadd:  1 + 6
    assert_eq!(vop(0x1F42_8C20, &regs).v(0), fd(-5.0)); // fmsub:  1 - 6
    assert_eq!(vop(0x1F62_0C20, &regs).v(0), fd(-7.0)); // fnmadd: -1 - 6
    assert_eq!(vop(0x1F62_8C20, &regs).v(0), fd(5.0)); // fnmsub: -1 + 6
                                                       // One rounding: a * a is not exact, so a fused multiply-subtract of the rounded product
                                                       // leaves the rounding error instead of zero.
    let a = f64::from_bits(0x3FF0_0000_2000_0001);
    let r = vop(
        0x1F42_8C20, // fmsub: d3 - d1 * d2
        &[(1, fd(a)), (2, fd(a)), (3, fd(a * a))],
    );
    let expect = (-a).mul_add(a, a * a);
    assert_ne!(expect, 0.0);
    assert_eq!(r.v(0), fd(expect));
}

#[test]
fn fmov_between_registers() {
    // fmov s0, #1.0; fmov d0, #-2.5
    assert_eq!(vop(0x1E2E_1000, &[(0, u128::MAX)]).v(0), fs(1.0));
    assert_eq!(vop(0x1E70_9000, &[(0, u128::MAX)]).v(0), fd(-2.5));
    // fmov d0, d1 copies 64 bits and clears the rest
    assert_eq!(
        vop(0x1E60_4020, &[(0, u128::MAX), (1, (7u128 << 64) | 0x1234)]).v(0),
        0x1234
    );
    // fmov x0, d1; fmov d0, x1; fmov w0, s1; fmov s0, w1
    let r = vop(0x9E66_0020, &[(1, (7u128 << 64) | 0xAABB_CCDD_1122_3344)]);
    assert_eq!(r.cpu.x[0], 0xAABB_CCDD_1122_3344);
    let r = start(0x9E67_0020, &[(0, u128::MAX)], |c, _| c.x[1] = 0x55);
    assert_eq!(r.v(0), 0x55);
    let r = vop(0x1E26_0020, &[(1, 0xAABB_CCDD_1122_3344)]);
    assert_eq!(r.cpu.x[0], 0x1122_3344);
    let r = start(0x1E27_0020, &[(0, u128::MAX)], |c, _| {
        c.x[1] = 0xFFFF_FFFF_8000_0001
    });
    assert_eq!(r.v(0), 0x8000_0001);
    // fmov x0, v1.d[1]; fmov v0.d[1], x1 keeps the low half
    let r = vop(0x9EAE_0020, &[(1, (0x99u128 << 64) | 1)]);
    assert_eq!(r.cpu.x[0], 0x99);
    let r = start(0x9EAF_0020, &[(0, 0x42)], |c, _| c.x[1] = 0x77);
    assert_eq!(r.v(0), (0x77u128 << 64) | 0x42);
}

#[test]
fn compares_set_the_flags() {
    let flags = |r: &Run| r.cpu.nzcv4();
    // fcmp s0, s1: less, equal, greater, unordered
    for (a, b, nzcv) in [
        (1.0f32, 2.0f32, 0b1000),
        (2.0, 2.0, 0b0110),
        (3.0, 2.0, 0b0010),
        (f32::NAN, 2.0, 0b0011),
    ] {
        let r = vop(0x1E21_2000, &[(0, fs(a)), (1, fs(b))]);
        assert_eq!(flags(&r), nzcv, "{a} vs {b}");
    }
    // A quiet NaN signals only in the E form: fcmpe d0, #0.0 against NaN
    let r = vop(0x1E60_2018, &[(0, fd(f64::NAN))]);
    assert_eq!((flags(&r), r.cpu.fpsr), (0b0011, IOC));
    let r = vop(0x1E60_2008, &[(0, fd(f64::NAN))]);
    assert_eq!((flags(&r), r.cpu.fpsr), (0b0011, 0));
    // fccmp s0, s1, #5, eq: the comparison when EQ holds, #5 otherwise
    let setup = |z: bool| {
        move |c: &mut CpuState, _: &mut FlatBus| {
            c.set_nzcv4(if z { 0b0100 } else { 0 });
        }
    };
    let r = start(0x1E21_0405, &[(0, fs(1.0)), (1, fs(2.0))], setup(true));
    assert_eq!(flags(&r), 0b1000);
    let r = start(0x1E21_0405, &[(0, fs(1.0)), (1, fs(2.0))], setup(false));
    assert_eq!(flags(&r), 0b0101);
    // fcsel d0, d1, d2, ne
    let r = start(0x1E62_1C20, &[(1, fd(1.0)), (2, fd(2.0))], setup(false));
    assert_eq!(r.v(0), fd(1.0));
    let r = start(0x1E62_1C20, &[(1, fd(1.0)), (2, fd(2.0))], setup(true));
    assert_eq!(r.v(0), fd(2.0));
}

#[test]
fn precision_conversions() {
    // fcvt d0, s1; fcvt s0, h1; fcvt h0, s1; fcvt s0, d1
    assert_eq!(vop(0x1E22_C020, &[(1, fs(1.5))]).v(0), fd(1.5));
    assert_eq!(vop(0x1EE2_4020, &[(1, 0x3C00)]).v(0), fs(1.0));
    assert_eq!(vop(0x1E23_C020, &[(1, fs(1.0))]).v(0), 0x3C00);
    let r = vop(0x1E62_4020, &[(1, fd(0.1))]);
    assert_eq!(r.v(0), fs(0.1));
    assert_eq!(r.cpu.fpsr, IXC);
    // 65520 rounds up to infinity in half precision
    let r = vop(0x1E63_C020, &[(1, fd(65520.0))]);
    assert_eq!(r.v(0), 0x7C00);
    assert_eq!(r.cpu.fpsr, OFC | IXC);
    // FPCR.AHP selects the alternative format: exponent 31 is an ordinary exponent, so 65520
    // (a tie between 65504 and 65536) rounds to 65536, and a value beyond 131008 saturates to the
    // largest number and signals instead of becoming an infinity.
    let ahp = 1 << 26;
    let r = with_fpcr(0x1E63_C020, &[(1, fd(65520.0))], ahp);
    assert_eq!((r.v(0), r.cpu.fpsr), (0x7C00, IXC));
    let r = with_fpcr(0x1E63_C020, &[(1, fd(1.0e6))], ahp);
    assert_eq!((r.v(0), r.cpu.fpsr), (0x7FFF, IOC));
    let r = vop(0x1E63_C020, &[(1, fd(1.0e6))]);
    assert_eq!((r.v(0), r.cpu.fpsr), (0x7C00, OFC | IXC));
}

#[test]
fn float_to_integer_saturates_and_nan_gives_zero() {
    // fcvtzs w0, s1
    for (x, expect, flags) in [
        (3.9f32, 3u64, IXC),
        (-3.9, 0xFFFF_FFFD, IXC),
        (f32::NAN, 0, IOC),
        (3.0e9, 0x7FFF_FFFF, IOC),
        (-3.0e9, 0x8000_0000, IOC),
    ] {
        let r = vop(0x1E38_0020, &[(1, fs(x))]);
        assert_eq!((r.cpu.x[0], r.cpu.fpsr), (expect, flags), "fcvtzs {x}");
    }
    // fcvtzu x0, d1
    for (x, expect, flags) in [(-1.0f64, 0u64, IOC), (1.0e30, u64::MAX, IOC), (2.5, 2, IXC)] {
        let r = vop(0x9E79_0020, &[(1, fd(x))]);
        assert_eq!((r.cpu.x[0], r.cpu.fpsr), (expect, flags), "fcvtzu {x}");
    }
    // The rounding modes: fcvtns, fcvtas, fcvtps, fcvtms, fcvtau
    assert_eq!(vop(0x1E20_0020, &[(1, fs(2.5))]).cpu.x[0], 2);
    assert_eq!(vop(0x1E20_0020, &[(1, fs(3.5))]).cpu.x[0], 4);
    assert_eq!(vop(0x9E64_0020, &[(1, fd(2.5))]).cpu.x[0], 3);
    assert_eq!(vop(0x9E64_0020, &[(1, fd(-2.5))]).cpu.x[0], (-3i64) as u64);
    assert_eq!(vop(0x1E28_0020, &[(1, fs(2.1))]).cpu.x[0], 3);
    assert_eq!(vop(0x1E28_0020, &[(1, fs(-2.1))]).cpu.x[0], 0xFFFF_FFFE);
    assert_eq!(vop(0x1E30_0020, &[(1, fs(2.1))]).cpu.x[0], 2);
    assert_eq!(vop(0x1E30_0020, &[(1, fs(-2.1))]).cpu.x[0], 0xFFFF_FFFD);
    assert_eq!(vop(0x1E25_0020, &[(1, fs(2.5))]).cpu.x[0], 3);
    // Fixed point: fcvtzs w0, d1, #4 and fcvtzu x0, s1, #16
    assert_eq!(vop(0x1E58_F020, &[(1, fd(1.5))]).cpu.x[0], 24);
    assert_eq!(vop(0x9E19_C020, &[(1, fs(1.0))]).cpu.x[0], 65536);
}

#[test]
fn integer_to_float() {
    // scvtf s0, w1; ucvtf d0, x1; scvtf d0, w1, #8
    let r = start(0x1E22_0020, &[], |c, _| c.x[1] = 0xFFFF_FFFF);
    assert_eq!(r.v(0), fs(-1.0));
    let r = start(0x9E63_0020, &[], |c, _| c.x[1] = u64::MAX);
    assert_eq!(r.v(0), fd(18446744073709551616.0));
    assert_eq!(r.cpu.fpsr, IXC);
    let r = start(0x1E42_E020, &[], |c, _| c.x[1] = 256);
    assert_eq!(r.v(0), fd(1.0));
}

#[test]
fn round_to_integral() {
    // frintn s, frintx s (inexact), frinti d (current mode), frintp, frintm, frinta, frintz d
    assert_eq!(vop(0x1E24_4020, &[(1, fs(2.5))]).v(0), fs(2.0));
    let r = vop(0x1E27_4020, &[(1, fs(2.5))]);
    assert_eq!((r.v(0), r.cpu.fpsr), (fs(2.0), IXC));
    let r = with_fpcr(0x1E67_C020, &[(1, fd(2.1))], RMODE_PLUS);
    assert_eq!((r.v(0), r.cpu.fpsr), (fd(3.0), 0));
    assert_eq!(vop(0x1E24_C020, &[(1, fs(2.1))]).v(0), fs(3.0));
    assert_eq!(vop(0x1E25_4020, &[(1, fs(-2.1))]).v(0), fs(-3.0));
    assert_eq!(vop(0x1E26_4020, &[(1, fs(2.5))]).v(0), fs(3.0));
    assert_eq!(vop(0x1E65_C020, &[(1, fd(-2.9))]).v(0), fd(-2.0));
}

#[test]
fn minimum_and_maximum_follow_the_nan_and_zero_rules() {
    let nan = u128::from(QNAN);
    let (pz, nz) = (fs(0.0), fs(-0.0));
    // fmax, fmin, fmaxnm, fminnm s0, s1, s2
    assert_eq!(
        vop(0x1E22_4820, &[(1, fs(1.0)), (2, fs(2.0))]).v(0),
        fs(2.0)
    );
    assert_eq!(vop(0x1E22_4820, &[(1, pz), (2, nz)]).v(0), pz);
    assert_eq!(vop(0x1E22_5820, &[(1, pz), (2, nz)]).v(0), nz);
    assert_eq!(vop(0x1E22_6820, &[(1, fs(1.0)), (2, nan)]).v(0), fs(1.0));
    assert_eq!(vop(0x1E22_7820, &[(1, nan), (2, fs(1.0))]).v(0), fs(1.0));
    assert_eq!(vop(0x1E22_4820, &[(1, fs(1.0)), (2, nan)]).v(0), nan);
}

#[test]
fn fpcr_modes_change_results() {
    let add = 0x1E22_2820; // fadd s0, s1, s2
                           // Default NaN: a NaN operand propagates unless FPCR.DN is set.
    let payload = u128::from(0x7FC1_2345u32);
    assert_eq!(vop(add, &[(1, payload), (2, fs(1.0))]).v(0), payload);
    assert_eq!(
        with_fpcr(add, &[(1, payload), (2, fs(1.0))], DN).v(0),
        u128::from(QNAN)
    );
    // A signalling NaN is quieted and signals.
    let r = vop(add, &[(1, u128::from(0x7F80_0001u32)), (2, fs(1.0))]);
    assert_eq!((r.v(0), r.cpu.fpsr), (u128::from(0x7FC0_0001u32), IOC));
    // Flush to zero: a denormal operand reads as zero (IDC) and a denormal result becomes zero
    // (UFC).
    let r = with_fpcr(add, &[(1, u128::from(1u32)), (2, fs(0.0))], FZ);
    assert_eq!((r.v(0), r.cpu.fpsr), (0, IDC));
    let r = vop(add, &[(1, u128::from(1u32)), (2, fs(0.0))]);
    assert_eq!((r.v(0), r.cpu.fpsr), (1, 0));
    let r = with_fpcr(
        0x1E22_0820, // fmul
        &[(1, u128::from(0x0080_0000u32)), (2, fs(0.5))],
        FZ,
    );
    assert_eq!((r.v(0), r.cpu.fpsr), (0, UFC));
    // Rounding modes on 1 + 2^-24 (a tie).
    let tiny = u128::from(0x3380_0000u32);
    let one_plus = u128::from(0x3F80_0001u32);
    for (fpcr, expect) in [
        (0, fs(1.0)),
        (RMODE_PLUS, one_plus),
        (RMODE_MINUS, fs(1.0)),
        (RMODE_ZERO, fs(1.0)),
    ] {
        let r = with_fpcr(add, &[(1, fs(1.0)), (2, tiny)], fpcr);
        assert_eq!((r.v(0), r.cpu.fpsr), (expect, IXC), "RMode {fpcr:#x}");
    }
    // The flags accumulate: a clean operation keeps what was already set.
    let r = start(add, &[(1, fs(1.0)), (2, fs(1.0))], |c, _| c.fpsr = DZC);
    assert_eq!(r.cpu.fpsr, DZC);
}
