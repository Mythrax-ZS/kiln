//! Hand-computed tests of the floating-point and Advanced SIMD instructions. The Unicorn
//! differential tests cover breadth; these need no external tool and pin each encoding group with
//! words assembled by LLVM and results worked out by hand from the Arm pseudocode.

use crate::a64::interp::step;
use crate::a64::state::CpuState;
use crate::testing::FlatBus;

const BASE: u64 = 0x10_0000;
const CODE: u64 = BASE;
const DATA: u64 = BASE + 0x1000;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn lanes<const N: usize>(bits: u32, values: [u64; N]) -> u128 {
    values.iter().enumerate().fold(0u128, |r, (i, &v)| {
        r | (u128::from(v) << (bits as usize * i))
    })
}

fn b8(v: [u8; 16]) -> u128 {
    u128::from_le_bytes(v)
}

fn h16(v: [u16; 8]) -> u128 {
    lanes(16, v.map(u64::from))
}

fn s32(v: [u32; 4]) -> u128 {
    lanes(32, v.map(u64::from))
}

fn d64(v: [u64; 2]) -> u128 {
    lanes(64, v)
}

fn f32s(v: [f32; 4]) -> u128 {
    s32(v.map(f32::to_bits))
}

fn f64s(v: [f64; 2]) -> u128 {
    d64(v.map(f64::to_bits))
}

fn splat8(b: u8) -> u128 {
    b8([b; 16])
}

/// The machine after one instruction, with the given vector registers set.
pub(super) struct Run {
    pub(super) cpu: CpuState,
    pub(super) bus: FlatBus,
}

impl Run {
    pub(super) fn v(&self, n: u8) -> u128 {
        self.cpu.vreg(n)
    }

    fn mem(&self, offset: usize, len: usize) -> Vec<u8> {
        let at = (DATA - BASE) as usize + offset;
        self.bus.mem[at..at + len].to_vec()
    }
}

pub(super) fn start(
    word: u32,
    regs: &[(u8, u128)],
    setup: impl FnOnce(&mut CpuState, &mut FlatBus),
) -> Run {
    let mut bus = FlatBus::new(BASE, 0x4000);
    bus.load_words(CODE, &[word]);
    let mut cpu = CpuState {
        pc: CODE,
        sp: DATA + 0x800,
        ..CpuState::default()
    };
    for &(n, v) in regs {
        cpu.set_vreg(n, v);
    }
    setup(&mut cpu, &mut bus);
    step(&mut cpu, &mut bus).expect("instruction executes");
    Run { cpu, bus }
}

pub(super) fn vop(word: u32, regs: &[(u8, u128)]) -> Run {
    start(word, regs, |_, _| {})
}

pub(super) const QC: u32 = 1 << 27;
pub(super) const IOC: u32 = 1;
pub(super) const DZC: u32 = 2;
pub(super) const OFC: u32 = 4;
pub(super) const IXC: u32 = 0x10;

// ---------------------------------------------------------------------------------------------
// Integer: three registers
// ---------------------------------------------------------------------------------------------

#[test]
fn add_sub_mul_and_multiply_accumulate() {
    // add v0.4s, v1.4s, v2.4s
    let r = vop(
        0x4EA28420,
        &[
            (1, s32([1, 0xFFFF_FFFF, 7, 0x8000_0000])),
            (2, s32([2, 1, 9, 0x8000_0000])),
        ],
    );
    assert_eq!(r.v(0), s32([3, 0, 16, 0]));
    // sub v0.8h, v1.8h, v2.8h
    let r = vop(
        0x6E628420,
        &[
            (1, h16([5, 0, 0x8000, 1, 2, 3, 4, 5])),
            (2, h16([3, 1, 0x7FFF, 1, 0, 0, 0, 6])),
        ],
    );
    assert_eq!(r.v(0), h16([2, 0xFFFF, 1, 0, 2, 3, 4, 0xFFFF]));
    // mul v0.4s, v1.4s, v2.4s
    let r = vop(
        0x4EA29C20,
        &[
            (1, s32([3, 0x10000, 0xFFFF_FFFF, 7])),
            (2, s32([5, 0x10000, 2, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([15, 0, 0xFFFF_FFFE, 0]));
    // mla v0.4s, v1.4s, v2.4s
    let r = vop(
        0x4EA29420,
        &[
            (0, s32([10, 20, 30, 40])),
            (1, s32([1, 2, 3, 4])),
            (2, s32([5, 6, 7, 8])),
        ],
    );
    assert_eq!(r.v(0), s32([15, 32, 51, 72]));
    // A 64-bit register form clears the upper half: add v0.2s, v1.2s, v2.2s
    let r = vop(
        0x0EA28420,
        &[
            (0, u128::MAX),
            (1, s32([1, 2, 3, 4])),
            (2, s32([10, 20, 30, 40])),
        ],
    );
    assert_eq!(r.v(0), s32([11, 22, 0, 0]));
}

#[test]
fn bitwise_operations_and_selects() {
    let ones = |b| splat8(b);
    let (c, a) = (ones(0xCC), ones(0xAA));
    // and, bic, orr, orn, eor (v0.16b, v1.16b, v2.16b)
    assert_eq!(vop(0x4E221C20, &[(1, c), (2, a)]).v(0), ones(0x88));
    assert_eq!(vop(0x4E621C20, &[(1, c), (2, a)]).v(0), ones(0x44));
    assert_eq!(vop(0x4EA21C20, &[(1, c), (2, a)]).v(0), ones(0xEE));
    assert_eq!(vop(0x4EE21C20, &[(1, c), (2, a)]).v(0), ones(0xDD));
    assert_eq!(vop(0x6E221C20, &[(1, c), (2, a)]).v(0), ones(0x66));
    // bsl: the destination is the mask. (0xF0 & 0xAA) | (0x0F & 0x55) = 0xA5
    let r = vop(
        0x6E621C20,
        &[(0, ones(0xF0)), (1, ones(0xAA)), (2, ones(0x55))],
    );
    assert_eq!(r.v(0), ones(0xA5));
    // bit: insert Vn where Vm is 1. (0x33 & 0x0F) | (0xAA & 0xF0) = 0xA3
    let r = vop(
        0x6EA21C20,
        &[(0, ones(0x33)), (1, ones(0xAA)), (2, ones(0xF0))],
    );
    assert_eq!(r.v(0), ones(0xA3));
    // bif: insert Vn where Vm is 0. (0x33 & 0xF0) | (0xAA & 0x0F) = 0x3A
    let r = vop(
        0x6EE21C20,
        &[(0, ones(0x33)), (1, ones(0xAA)), (2, ones(0xF0))],
    );
    assert_eq!(r.v(0), ones(0x3A));
}

#[test]
fn comparisons_produce_all_ones_or_zero() {
    let a = s32([1, 2, 3, 4]);
    let b = s32([1, 0, 3, 5]);
    // cmeq
    assert_eq!(vop(0x6EA28C20, &[(1, a), (2, b)]).v(0), s32([!0, 0, !0, 0]));
    let x = s32([1, 0xFFFF_FFFF, 5, 0x8000_0000]);
    let y = s32([0, 0, 5, 0x7FFF_FFFF]);
    // cmgt (signed): -1 > 0 and INT_MIN > INT_MAX are false
    assert_eq!(vop(0x4EA23420, &[(1, x), (2, y)]).v(0), s32([!0, 0, 0, 0]));
    // cmhi (unsigned)
    assert_eq!(
        vop(0x6EA23420, &[(1, x), (2, y)]).v(0),
        s32([!0, !0, 0, !0])
    );
    // cmtst
    let r = vop(
        0x4EA28C20,
        &[(1, s32([0xF0, 0x0F, 3, 0])), (2, s32([0x0F, 0x0F, 4, 5]))],
    );
    assert_eq!(r.v(0), s32([0, !0, 0, 0]));
}

#[test]
fn saturating_arithmetic_sets_qc() {
    // sqadd v0.16b: 127 + 1 and -128 + -1 saturate, 1 + 2 does not
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    (a[0], a[1], a[2]) = (0x7F, 0x80, 1);
    (b[0], b[1], b[2]) = (0x01, 0xFF, 2);
    let r = vop(0x4E220C20, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(&r.v(0).to_le_bytes()[..3], &[0x7F, 0x80, 3]);
    assert_eq!(r.cpu.fpsr & QC, QC);
    // The same operands without saturation leave QC clear.
    let r = vop(0x4E220C20, &[(1, b8([1; 16])), (2, b8([2; 16]))]);
    assert_eq!(r.v(0), splat8(3));
    assert_eq!(r.cpu.fpsr & QC, 0);
    // uqadd v0.16b: 255 + 1 saturates
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    (a[0], a[1]) = (0xFF, 1);
    (b[0], b[1]) = (1, 2);
    let r = vop(0x6E220C20, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[0xFF, 3]);
    assert_eq!(r.cpu.fpsr & QC, QC);
    // sqsub v0.8h: -32768 - 1 saturates; 5 - 7 = -2
    let r = vop(
        0x4E622C20,
        &[
            (1, h16([0x8000, 5, 0, 0, 0, 0, 0, 0])),
            (2, h16([1, 7, 0, 0, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), h16([0x8000, 0xFFFE, 0, 0, 0, 0, 0, 0]));
}

#[test]
fn halving_and_rounding_adds() {
    // shadd v0.4s: floor((a + b) / 2) on signed values
    let r = vop(
        0x4EA20420,
        &[
            (1, s32([3, 0xFFFF_FFFF, 0x7FFF_FFFF, 0x8000_0000])),
            (2, s32([4, 0xFFFF_FFFE, 0x7FFF_FFFF, 0x8000_0000])),
        ],
    );
    assert_eq!(r.v(0), s32([3, 0xFFFF_FFFE, 0x7FFF_FFFF, 0x8000_0000]));
    // urhadd v0.16b: (a + b + 1) >> 1 without overflow
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    (a[0], a[1]) = (1, 0xFF);
    (b[0], b[1]) = (2, 0xFF);
    let r = vop(0x6E221420, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(&r.v(0).to_le_bytes()[..3], &[2, 0xFF, 0]);
}

#[test]
fn minimum_maximum_and_absolute_difference() {
    let x = s32([1, 0xFFFF_FFFF, 5, 0x8000_0000]);
    let y = s32([2, 0, 0xFFFF_FFFB, 0x7FFF_FFFF]);
    // smax, umin
    assert_eq!(
        vop(0x4EA26420, &[(1, x), (2, y)]).v(0),
        s32([2, 0, 5, 0x7FFF_FFFF])
    );
    assert_eq!(
        vop(0x6EA26C20, &[(1, x), (2, y)]).v(0),
        s32([1, 0, 5, 0x7FFF_FFFF])
    );
    // sabd v0.16b: |127 - (-128)| = 255, |-128 - 127| = 255, |5 - 9| = 4
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    (a[0], a[1], a[2]) = (0x7F, 0x80, 5);
    (b[0], b[1], b[2]) = (0x80, 0x7F, 9);
    let r = vop(0x4E227420, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(&r.v(0).to_le_bytes()[..3], &[0xFF, 0xFF, 4]);
    // saba v0.16b accumulates: 10 + |5 - 9|
    let mut d = [0u8; 16];
    d[0] = 10;
    let r = vop(0x4E227C20, &[(0, b8(d)), (1, b8(a)), (2, b8(b))]);
    // (10 + 255) mod 256, 0 + 255, 0 + 4
    assert_eq!(&r.v(0).to_le_bytes()[..3], &[9, 0xFF, 4]);
}

#[test]
fn register_controlled_shifts() {
    let shifts = s32([4, 0xFC, 0xFF, 0x20]);
    let value = s32([1, 0xFFFF_FFF0, 0x8000_0000, 0xFF]);
    // sshl v0.4s: <<4, >>4 arithmetic (-16 >> 4 = -1), >>1 arithmetic, a shift of 32 gives 0
    assert_eq!(
        vop(0x4EA24420, &[(1, value), (2, shifts)]).v(0),
        s32([16, 0xFFFF_FFFF, 0xC000_0000, 0])
    );
    // ushl v0.4s: logical right shifts
    assert_eq!(
        vop(0x6EA24420, &[(1, value), (2, shifts)]).v(0),
        s32([16, 0x0FFF_FFFF, 0x4000_0000, 0])
    );
    // srshl v0.4s: rounding right shifts
    let r = vop(
        0x4EA25420,
        &[
            (1, s32([7, 0xFFFF_FFF9, 3, 0x10])),
            (2, s32([0xFF, 0xFF, 0xFE, 2])),
        ],
    );
    assert_eq!(r.v(0), s32([4, 0xFFFF_FFFD, 1, 0x40]));
}

#[test]
fn doubling_multiply_returning_the_high_half() {
    let a = h16([0x4000, 0x8000, 0x4000, 3, 0, 0, 0, 0]);
    let b = h16([0x4000, 0x8000, 1, 5, 0, 0, 0, 0]);
    // sqdmulh v0.8h: 2*0x4000*0x4000 >> 16 = 0x2000; -32768 * -32768 saturates; 2*0x4000*1 >> 16 = 0
    let r = vop(0x4E62B420, &[(1, a), (2, b)]);
    assert_eq!(r.v(0), h16([0x2000, 0x7FFF, 0, 0, 0, 0, 0, 0]));
    assert_eq!(r.cpu.fpsr & QC, QC);
    // sqrdmulh rounds: 2*0x4000*1 + 0x8000 >> 16 = 1
    let r = vop(0x6E62B420, &[(1, a), (2, b)]);
    assert_eq!(r.v(0), h16([0x2000, 0x7FFF, 1, 0, 0, 0, 0, 0]));
}

#[test]
fn pairwise_operations() {
    let a = s32([1, 2, 3, 4]);
    let b = s32([10, 20, 30, 40]);
    // addp: pairs of Vn then pairs of Vm
    assert_eq!(vop(0x4EA2BC20, &[(1, a), (2, b)]).v(0), s32([3, 7, 30, 70]));
    // smaxp
    let r = vop(
        0x4EA2A420,
        &[(1, s32([1, 5, 3, 2])), (2, s32([0, 0xFFFF_FFFF, 8, 7]))],
    );
    assert_eq!(r.v(0), s32([5, 3, 0, 8]));
}

#[test]
fn polynomial_multiply() {
    // pmul v0.16b: (x+1)(x^2+x+1) = x^3+1; 0x80 * 0x02 loses the carried-out bit
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    (a[0], a[1]) = (3, 0x80);
    (b[0], b[1]) = (7, 2);
    let r = vop(0x6E229C20, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[9, 0]);
    // pmull v0.8h, v1.8b, v2.8b keeps the whole 16-bit product
    let r = vop(0x0E22E020, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(r.v(0), h16([9, 0x100, 0, 0, 0, 0, 0, 0]));
}

// ---------------------------------------------------------------------------------------------
// Integer: widening and narrowing
// ---------------------------------------------------------------------------------------------

#[test]
fn long_and_wide_arithmetic() {
    // saddl v0.8h, v1.8b, v2.8b: -128 + -128 = -256, 127 + 127 = 254
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    (a[0], a[1]) = (0x80, 0x7F);
    (b[0], b[1]) = (0x80, 0x7F);
    let r = vop(0x0E220020, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(r.v(0), h16([0xFF00, 0x00FE, 0, 0, 0, 0, 0, 0]));
    // uaddl2 v0.8h, v1.16b, v2.16b uses the upper halves
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    (a[8], b[8]) = (0xFF, 0xFF);
    let r = vop(0x6E220020, &[(1, b8(a)), (2, b8(b))]);
    assert_eq!(r.v(0), h16([0x1FE, 0, 0, 0, 0, 0, 0, 0]));
    // saddw v0.4s, v1.4s, v2.4h
    let r = vop(
        0x0E621020,
        &[
            (1, s32([100, 0, 0xFFFF_FFFF, 5])),
            (2, h16([0xFFFF, 2, 3, 4, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([99, 2, 2, 9]));
    // umull v0.4s, v1.4h, v2.4h
    let r = vop(
        0x2E62C020,
        &[
            (1, h16([0xFFFF, 2, 0, 0, 0, 0, 0, 0])),
            (2, h16([0xFFFF, 3, 0, 0, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([0xFFFE_0001, 6, 0, 0]));
    // smlal v0.4s, v1.4h, v2.4h
    let r = vop(
        0x0E628020,
        &[
            (0, s32([100, 200, 300, 400])),
            (1, h16([0xFFFE, 3, 4, 5, 0, 0, 0, 0])),
            (2, h16([5, 6, 7, 8, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([90, 218, 328, 440]));
    // sqdmull v0.4s, v1.4h, v2.4h: 2 * -32768 * -32768 saturates
    let r = vop(
        0x0E62D020,
        &[
            (1, h16([0x8000, 3, 0, 0, 0, 0, 0, 0])),
            (2, h16([0x8000, 4, 0, 0, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([0x7FFF_FFFF, 24, 0, 0]));
    assert_eq!(r.cpu.fpsr & QC, QC);
}

#[test]
fn narrowing_high_half_adds() {
    // addhn v0.8b, v1.8h, v2.8h: the high byte of 0x1234 + 0x0111
    let r = vop(
        0x0E224020,
        &[
            (1, h16([0x1234, 0, 0, 0, 0, 0, 0, 0])),
            (2, h16([0x0111, 0, 0, 0, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0).to_le_bytes()[0], 0x13);
    // raddhn rounds: 0x12FF + 0x0081 + 0x80 = 0x1400
    let r = vop(
        0x2E224020,
        &[
            (1, h16([0x12FF, 0x12FF, 0, 0, 0, 0, 0, 0])),
            (2, h16([0x0001, 0x0081, 0, 0, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[0x13, 0x14]);
}

#[test]
fn narrowing_moves() {
    // xtn v0.4h, v1.4s truncates and clears the upper half
    let r = vop(
        0x0E612820,
        &[
            (0, u128::MAX),
            (1, s32([0x1_2345, 0xFFFF_FFFF, 0x8000_8000, 5])),
        ],
    );
    assert_eq!(r.v(0), h16([0x2345, 0xFFFF, 0x8000, 5, 0, 0, 0, 0]));
    // xtn2 keeps the lower half: xtn2 v0.8h, v1.4s is 0x4E612820
    let r = vop(
        0x4E612820,
        &[(0, h16([1, 2, 3, 4, 0, 0, 0, 0])), (1, s32([9, 8, 7, 6]))],
    );
    assert_eq!(r.v(0), h16([1, 2, 3, 4, 9, 8, 7, 6]));
    // sqxtn v0.4h, v1.4s
    let r = vop(
        0x0E614820,
        &[(1, s32([0x1_0000, 0xFFFF_0000, 5, 0xFFFF_FFFB]))],
    );
    assert_eq!(r.v(0), h16([0x7FFF, 0x8000, 5, 0xFFFB, 0, 0, 0, 0]));
    assert_eq!(r.cpu.fpsr & QC, QC);
    // uqxtn v0.8b, v1.8h
    let r = vop(
        0x2E214820,
        &[(1, h16([0x100, 0x7F, 0xFFFF, 0, 0, 0, 0, 0]))],
    );
    assert_eq!(&r.v(0).to_le_bytes()[..4], &[0xFF, 0x7F, 0xFF, 0]);
    // sqxtun v0.8b, v1.8h: negative gives 0
    let r = vop(
        0x2E212820,
        &[(1, h16([0xFFFF, 0x100, 0x7F, 0, 0, 0, 0, 0]))],
    );
    assert_eq!(&r.v(0).to_le_bytes()[..3], &[0, 0xFF, 0x7F]);
    assert_eq!(r.cpu.fpsr & QC, QC);
    // shll v0.4s, v1.4h, #16
    let r = vop(0x2E613820, &[(1, h16([1, 0xFFFF, 2, 3, 0, 0, 0, 0]))]);
    assert_eq!(r.v(0), s32([0x10000, 0xFFFF_0000, 0x20000, 0x30000]));
}

// ---------------------------------------------------------------------------------------------
// Integer: two registers
// ---------------------------------------------------------------------------------------------

#[test]
fn unary_integer_operations() {
    // abs, neg v0.4s
    let r = vop(0x4EA0B820, &[(1, s32([0xFFFF_FFFF, 0x8000_0000, 5, 0]))]);
    assert_eq!(r.v(0), s32([1, 0x8000_0000, 5, 0]));
    let r = vop(0x6EA0B820, &[(1, s32([1, 0x8000_0000, 0xFFFF_FFFB, 0]))]);
    assert_eq!(r.v(0), s32([0xFFFF_FFFF, 0x8000_0000, 5, 0]));
    // sqabs v0.16b: |-128| saturates
    let mut a = [0u8; 16];
    (a[0], a[1]) = (0x80, 0xFF);
    let r = vop(0x4E207820, &[(1, b8(a))]);
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[0x7F, 1]);
    assert_eq!(r.cpu.fpsr & QC, QC);
    // cls, clz v0.4s
    let r = vop(0x4EA04820, &[(1, s32([0, 0xFFFF_FFFF, 1, 0x4000_0000]))]);
    assert_eq!(r.v(0), s32([31, 31, 30, 0]));
    let r = vop(0x6EA04820, &[(1, s32([0, 1, 0x8000_0000, 0x00FF_0000]))]);
    assert_eq!(r.v(0), s32([32, 31, 0, 8]));
    // cnt, not, rbit v0.16b
    let mut a = [0u8; 16];
    (a[0], a[1], a[2], a[3]) = (0xFF, 0x0F, 0, 0x81);
    assert_eq!(
        &vop(0x4E205820, &[(1, b8(a))]).v(0).to_le_bytes()[..4],
        &[8, 4, 0, 2]
    );
    let mut a = [0u8; 16];
    (a[0], a[1], a[2]) = (0xF0, 0x01, 0xA5);
    assert_eq!(vop(0x6E205820, &[(1, b8(a))]).v(0).to_le_bytes()[0], 0x0F);
    assert_eq!(
        &vop(0x6E605820, &[(1, b8(a))]).v(0).to_le_bytes()[..3],
        &[0x0F, 0x80, 0xA5]
    );
    // cmeq, cmlt against zero
    let r = vop(0x4EA09820, &[(1, s32([0, 1, 0, 0xFFFF_FFFF]))]);
    assert_eq!(r.v(0), s32([!0, 0, !0, 0]));
    let r = vop(0x4EA0A820, &[(1, s32([0xFFFF_FFFF, 0, 5, 0x8000_0000]))]);
    assert_eq!(r.v(0), s32([!0, 0, 0, !0]));
    // suqadd v0.16b: 127 + 2 saturates; -128 + 255 = 127
    let mut d = [0u8; 16];
    (d[0], d[1]) = (0x7F, 0x80);
    let mut n = [0u8; 16];
    (n[0], n[1]) = (2, 0xFF);
    let r = vop(0x4E203820, &[(0, b8(d)), (1, b8(n))]);
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[0x7F, 0x7F]);
    assert_eq!(r.cpu.fpsr & QC, QC);
}

#[test]
fn reversals_and_pairwise_adds() {
    // rev64 v0.4s, rev32 v0.8h, rev16 v0.16b
    assert_eq!(
        vop(0x4EA00820, &[(1, s32([1, 2, 3, 4]))]).v(0),
        s32([2, 1, 4, 3])
    );
    assert_eq!(
        vop(0x6E600820, &[(1, h16([1, 2, 3, 4, 5, 6, 7, 8]))]).v(0),
        h16([2, 1, 4, 3, 6, 5, 8, 7])
    );
    let r = vop(
        0x4E201820,
        &[(
            1,
            b8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        )],
    );
    assert_eq!(
        r.v(0),
        b8([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14])
    );
    // saddlp v0.4s, v1.8h; uadalp accumulates
    let r = vop(0x4E602820, &[(1, h16([0xFFFF, 1, 2, 3, 0, 0, 0, 0]))]);
    assert_eq!(r.v(0), s32([0, 5, 0, 0]));
    let r = vop(
        0x6E606820,
        &[
            (0, s32([10, 10, 10, 10])),
            (1, h16([0xFFFF, 1, 2, 3, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([0x1_000A, 15, 10, 10]));
}

#[test]
fn reductions_across_lanes() {
    // addv s0, v1.4s
    let r = vop(0x4EB1B820, &[(0, u128::MAX), (1, s32([1, 2, 3, 4]))]);
    assert_eq!(r.v(0), 10);
    // saddlv s0, v1.8h: eight values of -1
    let r = vop(0x4E703820, &[(1, h16([0xFFFF; 8]))]);
    assert_eq!(r.v(0), 0xFFFF_FFF8);
    // umaxv b0, v1.16b
    let mut a = [0u8; 16];
    (a[0], a[1], a[2]) = (1, 200, 3);
    assert_eq!(vop(0x6E30A820, &[(1, b8(a))]).v(0), 200);
    // sminv h0, v1.8h
    let r = vop(0x4E71A820, &[(1, h16([5, 0xFFFE, 7, 9, 9, 9, 9, 9]))]);
    assert_eq!(r.v(0), 0xFFFE);
}

// ---------------------------------------------------------------------------------------------
// Permutes, table lookup, copies and immediates
// ---------------------------------------------------------------------------------------------

#[test]
fn permutes() {
    let (a, b) = (s32([1, 2, 3, 4]), s32([5, 6, 7, 8]));
    // zip1, zip2, uzp1, uzp2, trn1, trn2 (v0.4s, v1.4s, v2.4s)
    for (word, expect) in [
        (0x4E823820, [1, 5, 2, 6]),
        (0x4E827820, [3, 7, 4, 8]),
        (0x4E821820, [1, 3, 5, 7]),
        (0x4E825820, [2, 4, 6, 8]),
        (0x4E822820, [1, 5, 3, 7]),
        (0x4E826820, [2, 6, 4, 8]),
    ] {
        assert_eq!(vop(word, &[(1, a), (2, b)]).v(0), s32(expect), "{word:#x}");
    }
    // ext v0.16b, v1.16b, v2.16b, #5
    let lo = b8(core::array::from_fn(|i| i as u8));
    let hi = b8(core::array::from_fn(|i| 16 + i as u8));
    let r = vop(0x6E022820, &[(1, lo), (2, hi)]);
    assert_eq!(r.v(0), b8(core::array::from_fn(|i| 5 + i as u8)));
}

#[test]
fn table_lookups() {
    // tbl v0.16b, {v1.16b}, v2.16b: out of range indexes give zero
    let table = b8(core::array::from_fn(|i| 0x10 + i as u8));
    let mut idx = [0u8; 16];
    idx[..5].copy_from_slice(&[0, 15, 16, 255, 3]);
    let r = vop(0x4E020020, &[(0, u128::MAX), (1, table), (2, b8(idx))]);
    assert_eq!(&r.v(0).to_le_bytes()[..5], &[0x10, 0x1F, 0, 0, 0x13]);
    // tbx v0.16b, {v1.16b, v2.16b}, v3.16b: out of range keeps the destination
    let t1 = b8(core::array::from_fn(|i| i as u8));
    let t2 = b8(core::array::from_fn(|i| 0x20 + i as u8));
    let mut idx = [0u8; 16];
    idx[..5].copy_from_slice(&[0, 16, 31, 32, 200]);
    let r = vop(
        0x4E033020,
        &[(0, splat8(0xEE)), (1, t1), (2, t2), (3, b8(idx))],
    );
    assert_eq!(&r.v(0).to_le_bytes()[..5], &[0, 0x20, 0x2F, 0xEE, 0xEE]);
}

#[test]
fn element_moves() {
    // dup v0.4s, v1.s[2]
    let r = vop(0x4E140420, &[(1, s32([1, 2, 3, 4]))]);
    assert_eq!(r.v(0), s32([3, 3, 3, 3]));
    // dup v0.16b, w1
    let r = start(0x4E010C20, &[], |c, _| c.x[1] = 0x1234_56AB);
    assert_eq!(r.v(0), splat8(0xAB));
    // mov v0.s[1], w1 (ins general) keeps the other lanes
    let r = start(0x4E0C1C20, &[(0, s32([1, 2, 3, 4]))], |c, _| c.x[1] = 0x99);
    assert_eq!(r.v(0), s32([1, 0x99, 3, 4]));
    // mov v0.s[3], v1.s[0] (ins element)
    let r = vop(
        0x6E1C0420,
        &[(0, s32([1, 2, 3, 4])), (1, s32([0xAA, 0, 0, 0]))],
    );
    assert_eq!(r.v(0), s32([1, 2, 3, 0xAA]));
    // umov w0, v1.h[3]; smov x0, v1.b[1]
    let r = vop(0x0E0E3C20, &[(1, h16([0, 0, 0, 0xBEEF, 0, 0, 0, 0]))]);
    assert_eq!(r.cpu.x[0], 0xBEEF);
    let mut a = [0u8; 16];
    a[1] = 0x80;
    let r = vop(0x4E032C20, &[(1, b8(a))]);
    assert_eq!(r.cpu.x[0], 0xFFFF_FFFF_FFFF_FF80);
}

#[test]
fn modified_immediates() {
    // movi v0.4s, #0x12
    assert_eq!(vop(0x4F000640, &[]).v(0), s32([0x12; 4]));
    // movi v0.2d, #0xff00ff00ff00ff00
    assert_eq!(vop(0x6F05E540, &[]).v(0), d64([0xFF00_FF00_FF00_FF00; 2]));
    // mvni v0.4s, #0x12, lsl #8
    assert_eq!(vop(0x6F002640, &[]).v(0), s32([0xFFFF_EDFF; 4]));
    // orr v0.4s, #0x80, lsl #24
    assert_eq!(
        vop(0x4F047400, &[(0, s32([1, 2, 3, 4]))]).v(0),
        s32([0x8000_0001, 0x8000_0002, 0x8000_0003, 0x8000_0004])
    );
    // bic v0.8h, #0x1
    assert_eq!(
        vop(0x6F009420, &[(0, h16([0xFFFF; 8]))]).v(0),
        h16([0xFFFE; 8])
    );
    // fmov v0.4s, #1.0
    assert_eq!(vop(0x4F03F600, &[]).v(0), f32s([1.0; 4]));
    // movi v0.4s, #0x5, msl #8
    assert_eq!(vop(0x4F00C4A0, &[]).v(0), s32([0x5FF; 4]));
}

// ---------------------------------------------------------------------------------------------
// Shifts by an immediate
// ---------------------------------------------------------------------------------------------

#[test]
fn shifts_by_immediate() {
    let v = s32([0xFFFF_FFF0, 100, 7, 0x8000_0000]);
    // sshr #3
    assert_eq!(
        vop(0x4F3D0420, &[(1, v)]).v(0),
        s32([0xFFFF_FFFE, 12, 0, 0xF000_0000])
    );
    // ushr v0.2d, v1.2d, #1
    assert_eq!(
        vop(0x6F7F0420, &[(1, d64([0x8000_0000_0000_0001, 3]))]).v(0),
        d64([0x4000_0000_0000_0000, 1])
    );
    // ssra #3 accumulates
    assert_eq!(
        vop(
            0x4F3D1420,
            &[(0, s32([1; 4])), (1, s32([16, 0xFFFF_FFF0, 0, 7]))]
        )
        .v(0),
        s32([3, 0xFFFF_FFFF, 1, 1])
    );
    // srshr #3 rounds
    assert_eq!(
        vop(0x4F3D2420, &[(1, s32([4, 3, 12, 0xFFFF_FFFC]))]).v(0),
        s32([1, 0, 2, 0])
    );
    // sri #4 keeps the top four bits of the destination
    assert_eq!(
        vop(
            0x6F3C4420,
            &[(0, s32([0xFFFF_FFFF; 4])), (1, s32([0x1234_5678; 4]))]
        )
        .v(0),
        s32([0xF123_4567; 4])
    );
    // shl #5, sli #5 (keeps the low five bits of the destination)
    assert_eq!(
        vop(0x4F255420, &[(1, s32([1, 0x8000_0001, 0, 0]))]).v(0),
        s32([32, 32, 0, 0])
    );
    assert_eq!(
        vop(0x6F255420, &[(0, s32([0xFFFF_FFFF; 4])), (1, s32([1; 4]))]).v(0),
        s32([0x3F; 4])
    );
}

#[test]
fn saturating_shifts_by_immediate() {
    // sqshl #3: 0x10000000 << 3 overflows, 0xF0000000 << 3 = -2^31 does not
    let r = vop(
        0x4F237420,
        &[(1, s32([0x1000_0000, 5, 0xFFFF_FFFB, 0xF000_0000]))],
    );
    assert_eq!(r.v(0), s32([0x7FFF_FFFF, 40, 0xFFFF_FFD8, 0x8000_0000]));
    assert_eq!(r.cpu.fpsr & QC, QC);
    // uqshl v0.16b, #2
    let mut a = [0u8; 16];
    (a[0], a[1]) = (0x40, 0x10);
    let r = vop(0x6F0A7420, &[(1, b8(a))]);
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[0xFF, 0x40]);
    assert_eq!(r.cpu.fpsr & QC, QC);
    // sqshlu #4: negative gives zero, large overflows, small is exact
    let r = vop(0x6F246420, &[(1, s32([0xFFFF_FFFF, 0x1000_0000, 3, 0]))]);
    assert_eq!(r.v(0), s32([0, 0xFFFF_FFFF, 48, 0]));
}

#[test]
fn narrowing_and_widening_shifts() {
    // shrn v0.4h, v1.4s, #3
    let r = vop(0x0F1D8420, &[(1, s32([0x0001_0008, 0, 0, 0]))]);
    assert_eq!(r.v(0), h16([0x2001, 0, 0, 0, 0, 0, 0, 0]));
    // rshrn: (12 + 4) >> 3 = 2
    let r = vop(0x0F1D8C20, &[(1, s32([12, 0, 0, 0]))]);
    assert_eq!(r.v(0), h16([2, 0, 0, 0, 0, 0, 0, 0]));
    // sqshrn: 0x7FFFFFFF >> 3 does not fit in 16 bits
    let r = vop(0x0F1D9420, &[(1, s32([0x7FFF_FFFF, 0, 0, 0]))]);
    assert_eq!(r.v(0), h16([0x7FFF, 0, 0, 0, 0, 0, 0, 0]));
    assert_eq!(r.cpu.fpsr & QC, QC);
    // uqrshrn v0.8b, v1.8h, #3
    let r = vop(0x2F0D9C20, &[(1, h16([0xFFFF, 0x14, 0, 0, 0, 0, 0, 0]))]);
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[0xFF, 3]);
    // sqshrun v0.8b, v1.8h, #3: -8 >> 3 = -1 gives 0
    let r = vop(0x2F0D8420, &[(1, h16([0xFFF8, 0x100, 0, 0, 0, 0, 0, 0]))]);
    assert_eq!(&r.v(0).to_le_bytes()[..2], &[0, 0x20]);
    // sshll v0.2d, v1.2s, #3
    let r = vop(0x0F23A420, &[(1, s32([0xFFFF_FFFF, 5, 0, 0]))]);
    assert_eq!(r.v(0), d64([0xFFFF_FFFF_FFFF_FFF8, 40]));
    // ushll2 v0.8h, v1.16b, #0 widens the upper half
    let mut a = [0u8; 16];
    (a[8], a[9]) = (0xFF, 1);
    let r = vop(0x6F08A420, &[(1, b8(a))]);
    assert_eq!(r.v(0), h16([0xFF, 1, 0, 0, 0, 0, 0, 0]));
}

#[test]
fn fixed_point_conversions() {
    // scvtf v0.4s, v1.4s, #3
    let r = vop(0x4F3DE420, &[(1, s32([8, 12, 0xFFFF_FFF8, 0]))]);
    assert_eq!(r.v(0), f32s([1.0, 1.5, -1.0, 0.0]));
    // fcvtzs v0.2d, v1.2d, #4
    let r = vop(0x4F7CFC20, &[(1, f64s([1.5, -2.0]))]);
    assert_eq!(r.v(0), d64([24, (-32i64) as u64]));
}

// ---------------------------------------------------------------------------------------------
// By element
// ---------------------------------------------------------------------------------------------

#[test]
fn operations_by_element() {
    // mul v0.4s, v1.4s, v2.s[3]
    let r = vop(
        0x4FA28820,
        &[(1, s32([1, 2, 3, 4])), (2, s32([0, 0, 0, 10]))],
    );
    assert_eq!(r.v(0), s32([10, 20, 30, 40]));
    // mla v0.8h, v1.8h, v2.h[7]
    let r = vop(
        0x6F720820,
        &[
            (0, h16([1; 8])),
            (1, h16([1, 2, 3, 4, 5, 6, 7, 8])),
            (2, h16([0, 0, 0, 0, 0, 0, 0, 2])),
        ],
    );
    assert_eq!(r.v(0), h16([3, 5, 7, 9, 11, 13, 15, 17]));
    // smull v0.4s, v1.4h, v2.h[2]
    let r = vop(
        0x0F62A020,
        &[
            (1, h16([0xFFFF, 2, 3, 4, 0, 0, 0, 0])),
            (2, h16([0, 0, 3, 0, 0, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([0xFFFF_FFFD, 6, 9, 12]));
    // sqdmulh v0.4s, v1.4s, v2.s[1]
    let r = vop(
        0x4FA2C020,
        &[
            (1, s32([0x4000_0000, 0x1000_0000, 0, 0x8000_0000])),
            (2, s32([0, 0x4000_0000, 0, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([0x2000_0000, 0x0800_0000, 0, 0xC000_0000]));
    // fmla v0.4s, v1.4s, v2.s[3]
    let r = vop(
        0x4FA21820,
        &[
            (0, f32s([1.0; 4])),
            (1, f32s([1.0, 2.0, 3.0, 4.0])),
            (2, f32s([0.0, 0.0, 0.0, 0.5])),
        ],
    );
    assert_eq!(r.v(0), f32s([1.5, 2.0, 2.5, 3.0]));
    // fmul v0.2d, v1.2d, v2.d[1]
    let r = vop(0x4FC29820, &[(1, f64s([1.0, 2.0])), (2, f64s([0.0, 3.0]))]);
    assert_eq!(r.v(0), f64s([3.0, 6.0]));
    // fmulx s0, s1, v2.s[2]: zero times infinity is 2.0
    let r = vop(
        0x7F829820,
        &[(1, u128::from(0u32)), (2, s32([0, 0, 0x7F80_0000, 0]))],
    );
    assert_eq!(r.v(0), u128::from(2.0f32.to_bits()));
}

// ---------------------------------------------------------------------------------------------
// Floating point
// ---------------------------------------------------------------------------------------------

pub(super) const QNAN: u32 = 0x7FC0_0000;

#[test]
fn vector_arithmetic_and_flags() {
    // fadd v0.4s: FLT_MAX + FLT_MAX overflows to infinity (OFC and IXC), NaN propagates
    let r = vop(
        0x4E22D420,
        &[
            (1, s32([1.0f32.to_bits(), 0x7F7F_FFFF, QNAN, 0x8000_0000])),
            (
                2,
                s32([2.0f32.to_bits(), 0x7F7F_FFFF, 1.0f32.to_bits(), 0x8000_0000]),
            ),
        ],
    );
    assert_eq!(
        r.v(0),
        s32([3.0f32.to_bits(), 0x7F80_0000, QNAN, 0x8000_0000])
    );
    assert_eq!(r.cpu.fpsr, OFC | IXC);
    // fsub v0.2d
    let r = vop(0x4EE2D420, &[(1, f64s([5.0, 1.0])), (2, f64s([3.0, 4.0]))]);
    assert_eq!(r.v(0), f64s([2.0, -3.0]));
    // fmul v0.4s
    let r = vop(
        0x6E22DC20,
        &[
            (1, f32s([1.5, 2.0, 0.0, -1.0])),
            (2, f32s([2.0, 4.0, 5.0, -1.0])),
        ],
    );
    assert_eq!(r.v(0), f32s([3.0, 8.0, 0.0, 1.0]));
    // fdiv v0.4s: 1/0 = inf (DZC), 0/0 = default NaN (IOC), 1/3 inexact, 6/3 = 2
    let r = vop(
        0x6E22FC20,
        &[
            (1, f32s([1.0, 0.0, 1.0, 6.0])),
            (2, f32s([0.0, 0.0, 3.0, 3.0])),
        ],
    );
    assert_eq!(r.v(0), s32([0x7F80_0000, QNAN, 0x3EAA_AAAB, 0x4000_0000]));
    assert_eq!(r.cpu.fpsr, DZC | IOC | IXC);
    // fmax / fminnm with a quiet NaN: fmax propagates, fminnm returns the number
    let r = vop(
        0x4E22F420,
        &[
            (1, s32([1.0f32.to_bits(), 0, 0, 0])),
            (2, s32([QNAN, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0).to_le_bytes()[..4], QNAN.to_le_bytes());
    let r = vop(
        0x4EA2C420,
        &[
            (1, s32([1.0f32.to_bits(), 0, 0, 0])),
            (2, s32([QNAN, 0, 0, 0])),
        ],
    );
    assert_eq!(r.v(0).to_le_bytes()[..4], 1.0f32.to_bits().to_le_bytes());
    // fabd
    let r = vop(0x6EA2D420, &[(1, f32s([1.0; 4])), (2, f32s([3.0; 4]))]);
    assert_eq!(r.v(0), f32s([2.0; 4]));
    // fmla, fmls are fused: 1 + 2*3 and 1 - 2*3
    let regs = [
        (0, f32s([1.0; 4])),
        (1, f32s([2.0; 4])),
        (2, f32s([3.0; 4])),
    ];
    assert_eq!(vop(0x4E22CC20, &regs).v(0), f32s([7.0; 4]));
    assert_eq!(vop(0x4EA2CC20, &regs).v(0), f32s([-5.0; 4]));
}

#[test]
fn vector_floating_point_comparisons() {
    // fcmeq v0.4s: equal, NaN, +0 == -0, different
    let r = vop(
        0x4E22E420,
        &[
            (1, s32([1.0f32.to_bits(), QNAN, 0, 2.0f32.to_bits()])),
            (
                2,
                s32([1.0f32.to_bits(), QNAN, 0x8000_0000, 3.0f32.to_bits()]),
            ),
        ],
    );
    assert_eq!(r.v(0), s32([!0, 0, !0, 0]));
    assert_eq!(r.cpu.fpsr, 0, "a quiet NaN does not signal in fcmeq");
    // fcmge v0.4s: a quiet NaN signals
    let r = vop(
        0x6E22E420,
        &[
            (1, f32s([2.0, 1.0, 0.0, 0.0])),
            (2, s32([1.0f32.to_bits(), 2.0f32.to_bits(), QNAN, 0])),
        ],
    );
    assert_eq!(r.v(0), s32([!0, 0, 0, !0]));
    assert_eq!(r.cpu.fpsr, IOC);
    // facgt v0.4s: |-3| > |2|
    let r = vop(0x6EA2EC20, &[(1, f32s([-3.0; 4])), (2, f32s([2.0; 4]))]);
    assert_eq!(r.v(0), s32([!0; 4]));
    // fcmgt, fcmle against zero
    let r = vop(
        0x4EA0C820,
        &[(1, s32([1.0f32.to_bits(), (-1.0f32).to_bits(), 0, QNAN]))],
    );
    assert_eq!(r.v(0), s32([!0, 0, 0, 0]));
    assert_eq!(r.cpu.fpsr, IOC);
    let r = vop(
        0x6EA0D820,
        &[(1, s32([0, 0x8000_0000, 1.0f32.to_bits(), 0]))],
    );
    assert_eq!(r.v(0), s32([!0, !0, 0, !0]));
}

#[test]
fn pairwise_and_across_lane_floating_point() {
    // faddp v0.4s, v1.4s, v2.4s
    let r = vop(
        0x6E22D420,
        &[
            (1, f32s([1.0, 2.0, 3.0, 4.0])),
            (2, f32s([10.0, 20.0, 30.0, 40.0])),
        ],
    );
    assert_eq!(r.v(0), f32s([3.0, 7.0, 30.0, 70.0]));
    // fmaxp v0.4s
    let r = vop(
        0x6E22F420,
        &[
            (1, f32s([1.0, 5.0, 3.0, 2.0])),
            (2, f32s([0.0, -1.0, 8.0, 7.0])),
        ],
    );
    assert_eq!(r.v(0), f32s([5.0, 3.0, 0.0, 8.0]));
    // fmaxv s0, v1.4s; fminnmv ignores a quiet NaN
    let r = vop(0x6E30F820, &[(1, f32s([1.0, 5.0, 3.0, 2.0]))]);
    assert_eq!(r.v(0), u128::from(5.0f32.to_bits()));
    let r = vop(
        0x6EB0C820,
        &[(
            1,
            s32([QNAN, 1.0f32.to_bits(), 2.0f32.to_bits(), 0.5f32.to_bits()]),
        )],
    );
    assert_eq!(r.v(0), u128::from(0.5f32.to_bits()));
    // faddp s0, v1.2s and faddp d0, v1.2d, addp d0, v1.2d
    let r = vop(0x7E30D820, &[(1, f32s([1.5, 2.5, 0.0, 0.0]))]);
    assert_eq!(r.v(0), u128::from(4.0f32.to_bits()));
    let r = vop(0x7E70D820, &[(1, f64s([1.5, 2.5]))]);
    assert_eq!(r.v(0), u128::from(4.0f64.to_bits()));
    let r = vop(0x5EF1B820, &[(1, d64([u64::MAX, 2]))]);
    assert_eq!(r.v(0), 1);
}

#[test]
fn estimates_and_steps() {
    // frecps v0.4s: 2 - 1.0 * 0.5; frsqrts: (3 - 1.0 * 1.0) / 2
    let r = vop(0x4E22FC20, &[(1, f32s([1.0; 4])), (2, f32s([0.5; 4]))]);
    assert_eq!(r.v(0), f32s([1.5; 4]));
    let r = vop(0x4EA2FC20, &[(1, f32s([1.0; 4])), (2, f32s([1.0; 4]))]);
    assert_eq!(r.v(0), f32s([1.0; 4]));
    // frsqrts rounds once, after halving: (3 - FLT_MAX * 1.5) overflows a single but half of it
    // does not, so the result is -0.75 * FLT_MAX rounded, not an infinity.
    let r = vop(
        0x4EA2FC20,
        &[
            (1, s32([0x7F7F_FFFF, 0, 0, 0])),
            (2, f32s([1.5, 0.0, 0.0, 0.0])),
        ],
    );
    assert_eq!(r.v(0).to_le_bytes()[..4], 0xFF3F_FFFFu32.to_le_bytes());
    assert_eq!(r.cpu.fpsr & OFC, 0);
    // frecpe, frsqrte v0.4s
    let r = vop(0x4EA1D820, &[(1, f32s([1.0, 2.0, 0.0, 4.0]))]);
    assert_eq!(
        r.v(0),
        s32([0x3F7F_8000, 0x3EFF_8000, 0x7F80_0000, 0x3E7F_8000])
    );
    let r = vop(0x6EA1D820, &[(1, f32s([1.0, 4.0, 0.0, 16.0]))]);
    assert_eq!(
        r.v(0),
        s32([0x3F7F_8000, 0x3EFF_8000, 0x7F80_0000, 0x3E7F_8000])
    );
    // urecpe: 0x80000000 -> 0xFF800000; operands below it give all ones. ursqrte likewise.
    let r = vop(
        0x4EA1C820,
        &[(1, s32([0x8000_0000, 0x7FFF_FFFF, 0xFFFF_FFFF, 0]))],
    );
    assert_eq!(
        r.v(0),
        s32([0xFF80_0000, 0xFFFF_FFFF, 0x8000_0000, 0xFFFF_FFFF])
    );
    let r = vop(0x6EA1C820, &[(1, s32([0xC000_0000, 0x3FFF_FFFF, 0, 0]))]);
    assert_eq!(
        r.v(0),
        s32([0x9380_0000, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF])
    );
}

#[test]
fn unary_floating_point_and_rounding() {
    // fabs v0.4s, fneg v0.2d
    assert_eq!(
        vop(0x4EA0F820, &[(1, f32s([-1.5, 1.5, -0.0, 0.0]))]).v(0),
        f32s([1.5, 1.5, 0.0, 0.0])
    );
    assert_eq!(
        vop(0x6EE0F820, &[(1, f64s([1.0, -2.0]))]).v(0),
        f64s([-1.0, 2.0])
    );
    // fsqrt: sqrt(4) = 2; sqrt(-1) is the default NaN and signals
    let r = vop(0x6EA1F820, &[(1, f32s([4.0, -1.0, 0.0, 9.0]))]);
    assert_eq!(r.v(0), s32([2.0f32.to_bits(), QNAN, 0, 3.0f32.to_bits()]));
    assert_eq!(r.cpu.fpsr, IOC);
    // frintn: ties to even; frintz: toward zero; frinta: ties away; frintx raises inexact
    let r = vop(0x4E218820, &[(1, f32s([2.5, 3.5, -0.5, 1.4]))]);
    assert_eq!(r.v(0), f32s([2.0, 4.0, -0.0, 1.0]));
    let r = vop(0x4EE19820, &[(1, f64s([-1.9, 2.9]))]);
    assert_eq!(r.v(0), f64s([-1.0, 2.0]));
    let r = vop(0x6E218820, &[(1, f32s([2.5, -2.5, 0.5, 1.5]))]);
    assert_eq!(r.v(0), f32s([3.0, -3.0, 1.0, 2.0]));
    let r = vop(0x6E219820, &[(1, f32s([2.5, 3.0, 0.0, 0.0]))]);
    assert_eq!(r.v(0), f32s([2.0, 3.0, 0.0, 0.0]));
    assert_eq!(r.cpu.fpsr, IXC);
}

#[test]
fn vector_conversions() {
    // fcvtns v0.4s: ties to even, saturation signals
    let r = vop(0x4E21A820, &[(1, f32s([2.5, 3.5, -2.5, 1.0e10]))]);
    assert_eq!(r.v(0), s32([2, 4, 0xFFFF_FFFE, 0x7FFF_FFFF]));
    assert_eq!(r.cpu.fpsr & IOC, IOC);
    // fcvtzu v0.2d: negative gives zero
    let r = vop(0x6EE1B820, &[(1, f64s([-1.0, 1.9]))]);
    assert_eq!(r.v(0), d64([0, 1]));
    // fcvtas v0.4s: ties away; NaN gives zero
    let r = vop(
        0x4E21C820,
        &[(1, s32([2.5f32.to_bits(), (-2.5f32).to_bits(), QNAN, 0]))],
    );
    assert_eq!(r.v(0), s32([3, 0xFFFF_FFFD, 0, 0]));
    // scvtf v0.4s: INT_MAX rounds to 2^31 (inexact); ucvtf v0.2d: 2^64 - 1 rounds up
    let r = vop(0x4E21D820, &[(1, s32([0xFFFF_FFFF, 0x7FFF_FFFF, 0, 0]))]);
    assert_eq!(r.v(0), f32s([-1.0, 2147483648.0, 0.0, 0.0]));
    assert_eq!(r.cpu.fpsr, IXC);
    let r = vop(0x6E61D820, &[(1, d64([u64::MAX, 0]))]);
    assert_eq!(r.v(0), d64([0x43F0_0000_0000_0000, 0]));
    // fcvtn v0.4h, v1.4s; fcvtl v0.4s, v1.4h
    let r = vop(0x0E216820, &[(1, f32s([1.0, -2.0, 65504.0, 1.0e-8]))]);
    assert_eq!(r.v(0), h16([0x3C00, 0xC000, 0x7BFF, 0, 0, 0, 0, 0]));
    let r = vop(
        0x0E217820,
        &[(1, h16([0x3C00, 0xC000, 0x7C00, 0x0001, 0, 0, 0, 0]))],
    );
    assert_eq!(
        r.v(0),
        s32([0x3F80_0000, 0xC000_0000, 0x7F80_0000, 0x3380_0000])
    );
    // fcvtxn v0.2s, v1.2d rounds to odd: 1 + 2^-52 becomes 1 + 2^-23
    let r = vop(0x2E616820, &[(1, d64([0x3FF0_0000_0000_0001, 0]))]);
    assert_eq!(r.v(0).to_le_bytes()[..4], 0x3F80_0001u32.to_le_bytes());
}

#[test]
fn scalar_simd_forms() {
    // add d0, d1, d2 clears the rest of the register
    let r = vop(
        0x5EE28420,
        &[(0, u128::MAX), (1, d64([5, 9])), (2, d64([7, 9]))],
    );
    assert_eq!(r.v(0), 12);
    // cmgt d0, d1, d2
    let r = vop(0x5EE23420, &[(1, d64([5, 0])), (2, d64([3, 0]))]);
    assert_eq!(r.v(0), u128::from(u64::MAX));
    // sqadd b0, b1, b2 saturates
    let r = vop(0x5E220C20, &[(1, 0x7F), (2, 1)]);
    assert_eq!(r.v(0), 0x7F);
    assert_eq!(r.cpu.fpsr & QC, QC);
    // fmulx s0, s1, s2: 0 * inf = 2.0
    let r = vop(0x5E22DC20, &[(1, 0), (2, u128::from(0x7F80_0000u32))]);
    assert_eq!(r.v(0), u128::from(2.0f32.to_bits()));
    // frecpx s0, s1: the exponent is inverted
    let r = vop(0x5EA1F820, &[(1, u128::from(1.0f32.to_bits()))]);
    assert_eq!(r.v(0), u128::from(2.0f32.to_bits()));
    // fcvtzs s0, s1
    let r = vop(0x5EA1B820, &[(1, u128::from(3.9f32.to_bits()))]);
    assert_eq!(r.v(0), 3);
    // sqxtn b0, h1
    let r = vop(0x5E214820, &[(1, 0x100)]);
    assert_eq!(r.v(0), 0x7F);
    assert_eq!(r.cpu.fpsr & QC, QC);
    // sqshl d0, d1, #7
    let r = vop(0x5F477420, &[(1, 1)]);
    assert_eq!(r.v(0), 128);
}

// ---------------------------------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------------------------------

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| i as u8).collect()
}

fn with_data(word: u32, regs: &[(u8, u128)], len: usize) -> Run {
    start(word, regs, |c, bus| {
        c.x[0] = DATA;
        c.x[1] = 3;
        let at = (DATA - BASE) as usize;
        bus.mem[at..at + len].copy_from_slice(&pattern(len));
    })
}

#[test]
fn structure_loads() {
    let bytes = |lo: usize, n: usize| -> Vec<u8> { (lo..lo + n).map(|i| i as u8).collect() };
    // ld1 {v0.16b}, [x0]
    let r = with_data(0x4C407000, &[], 32);
    assert_eq!(r.v(0).to_le_bytes().to_vec(), bytes(0, 16));
    // ld1 {v0.4s, v1.4s}, [x0]
    let r = with_data(0x4C40A800, &[], 32);
    assert_eq!(r.v(1).to_le_bytes().to_vec(), bytes(16, 16));
    // ld2 {v0.4s, v1.4s}, [x0]: de-interleaved words
    let r = with_data(0x4C408800, &[], 32);
    let word = |i: usize| {
        u32::from_le_bytes([
            4 * i as u8,
            4 * i as u8 + 1,
            4 * i as u8 + 2,
            4 * i as u8 + 3,
        ])
    };
    assert_eq!(r.v(0), s32([word(0), word(2), word(4), word(6)]));
    assert_eq!(r.v(1), s32([word(1), word(3), word(5), word(7)]));
    // ld3 {v0.8b, v1.8b, v2.8b}, [x0]: every third byte
    let r = with_data(0x0C404000, &[], 32);
    assert_eq!(r.v(0).to_le_bytes()[..8], [0, 3, 6, 9, 12, 15, 18, 21]);
    assert_eq!(r.v(2).to_le_bytes()[..8], [2, 5, 8, 11, 14, 17, 20, 23]);
    assert_eq!(r.v(0) >> 64, 0, "a 64-bit load clears the upper half");
    // ld4 {v0.16b-v3.16b}, [x0], #64 writes back the 64 bytes transferred
    let r = with_data(0x4CDF0000, &[], 64);
    assert_eq!(r.v(1).to_le_bytes()[..4], [1, 5, 9, 13]);
    assert_eq!(r.cpu.x[0], DATA + 64);
    // ld1 {v0.s}[2], [x0] replaces one lane
    let r = with_data(0x4D408000, &[(0, s32([0xA, 0xB, 0xC, 0xD]))], 8);
    assert_eq!(r.v(0), s32([0xA, 0xB, word(0), 0xD]));
    // ld2 {v0.h, v1.h}[3], [x0], #4
    let r = with_data(0x0DFF5800, &[(0, h16([1; 8])), (1, h16([2; 8]))], 8);
    // Only the named lane changes, in the upper half of the register too.
    assert_eq!(r.v(0), h16([1, 1, 1, 0x0100, 1, 1, 1, 1]));
    assert_eq!(r.v(1), h16([2, 2, 2, 0x0302, 2, 2, 2, 2]));
    assert_eq!(r.cpu.x[0], DATA + 4);
    // ld1r {v0.4s}, [x0]
    let r = with_data(0x4D40C800, &[], 8);
    assert_eq!(r.v(0), s32([word(0); 4]));
    // ld3r {v0.8h, v1.8h, v2.8h}, [x0], x1 writes back x1
    let r = with_data(0x4DC1E400, &[], 8);
    assert_eq!(r.v(1), h16([0x0302; 8]));
    assert_eq!(r.cpu.x[0], DATA + 3);
}

#[test]
fn structure_stores() {
    let v = |n: u8| b8(core::array::from_fn(|i| 16 * n + i as u8));
    // st1 {v0.16b}, [x0], #16
    let r = with_data(0x4C9F7000, &[(0, v(0))], 0);
    assert_eq!(r.mem(0, 16), (0..16).collect::<Vec<u8>>());
    assert_eq!(r.cpu.x[0], DATA + 16);
    // st1 {v0.4s, v1.4s, v2.4s}, [x0]
    let r = with_data(0x4C006800, &[(0, v(0)), (1, v(1)), (2, v(2))], 0);
    assert_eq!(r.mem(0, 48), (0..48).collect::<Vec<u8>>());
    // st2 {v0.8b, v1.8b}, [x0] interleaves
    let r = with_data(0x0C008000, &[(0, v(0)), (1, v(1))], 0);
    assert_eq!(r.mem(0, 6), vec![0, 16, 1, 17, 2, 18]);
    // st4 {v0.4h-v3.4h}, [x0]
    let r = with_data(0x0C000400, &[(0, v(0)), (1, v(1)), (2, v(2)), (3, v(3))], 0);
    assert_eq!(r.mem(0, 10), vec![0, 1, 16, 17, 32, 33, 48, 49, 2, 3]);
    // st1 {v0.d}[1], [x0]
    let r = with_data(0x4D008400, &[(0, v(0))], 0);
    assert_eq!(r.mem(0, 8), (8..16).collect::<Vec<u8>>());
    // st3 {v0.b, v1.b, v2.b}[9], [x0], #3
    let r = with_data(0x4D9F2400, &[(0, v(0)), (1, v(1)), (2, v(2))], 0);
    assert_eq!(r.mem(0, 3), vec![9, 25, 41]);
    assert_eq!(r.cpu.x[0], DATA + 3);
}

#[test]
fn register_loads_and_stores() {
    // ldr q0, [x0, #16]
    let r = with_data(0x3DC00400, &[], 32);
    assert_eq!(r.v(0).to_le_bytes().to_vec(), (16..32).collect::<Vec<u8>>());
    // str d0, [x0], #8 stores 8 bytes and writes back
    let r = with_data(
        0xFC008400,
        &[(0, b8(core::array::from_fn(|i| 0x40 + i as u8)))],
        0,
    );
    assert_eq!(
        r.mem(0, 9),
        vec![0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0]
    );
    assert_eq!(r.cpu.x[0], DATA + 8);
    // ldp q0, q1, [x0, #32]! pre-indexes
    let r = with_data(0xADC10400, &[], 0);
    assert_eq!(r.cpu.x[0], DATA + 32);
    // stp s0, s1, [x0]
    let r = with_data(0x2D000400, &[(0, 0xAABB_CCDD), (1, 0x1122_3344)], 0);
    assert_eq!(
        r.mem(0, 8),
        vec![0xDD, 0xCC, 0xBB, 0xAA, 0x44, 0x33, 0x22, 0x11]
    );
}

#[test]
fn encodings_the_core_lacks_are_unallocated_and_the_crypto_extension_decodes() {
    use crate::a64::{decode, Inst};
    // fadd h0, h1, h2 (half-precision arithmetic needs FEAT_FP16)
    assert!(matches!(decode(0x1EE2_2820), Inst::Unallocated(_)));
    // sqrdmlah v0.4s, v1.4s, v2.4s (FEAT_RDM)
    assert!(matches!(decode(0x6E82_8420), Inst::Unallocated(_)));
    // sdot v0.4s, v1.16b, v2.16b (FEAT_DotProd)
    assert!(matches!(decode(0x4E82_9420), Inst::Unallocated(_)));
    // The cryptographic extension is part of the core: aese v0.16b, v1.16b and sha256h q0, q1,
    // v2.4s decode; the reserved neighbours (size != 0, the unused opcodes) do not.
    assert!(matches!(decode(0x4E28_4820), Inst::Vec { .. }));
    assert!(matches!(decode(0x5E02_4020), Inst::Vec { .. }));
    assert!(matches!(decode(0x4E68_4820), Inst::Unallocated(_)));
    assert!(matches!(
        decode(0x4E28_4820 | 0x1F000),
        Inst::Unallocated(_)
    ));
    assert!(matches!(decode(0x5E02_7020), Inst::Unallocated(_)));
    // fcmp d0, #0.0 with a non-zero Rm field
    assert!(matches!(decode(0x1E62_2018), Inst::Unallocated(_)));
}
