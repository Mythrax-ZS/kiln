//! Floating-point arithmetic with the Arm semantics.
//!
//! Every operation follows the Arm A-profile pseudocode (`FPUnpack`, `FPRound`, `FPAdd`, `FPMulAdd`,
//! `FPToFixed`, ...), not the behavior of the host's floating-point unit: NaN propagation and the
//! default NaN, flush-to-zero of inputs and results, tininess detected before rounding, the
//! cumulative exception flags in `FPSR`, and saturating conversions all differ from x86 and are
//! written out here. Values are bit patterns (binary16 in the low 16 bits, binary32 in the low 32).
//!
//! Arithmetic is exact: operands are unpacked into a normalized significand and exponent, the
//! operation is carried out in a 128-bit frame with a sticky bit for anything shifted out, and one
//! rounding step produces the result. Binary16 appears only in conversions (the Cortex-A57 has no
//! half-precision arithmetic).

pub const IOC: u32 = 1 << 0;
pub const DZC: u32 = 1 << 1;
pub const OFC: u32 = 1 << 2;
pub const UFC: u32 = 1 << 3;
pub const IXC: u32 = 1 << 4;
pub const IDC: u32 = 1 << 7;
/// Cumulative saturation flag (`FPSR.QC`), set by saturating Advanced SIMD operations.
pub const QC: u32 = 1 << 27;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    H,
    S,
    D,
}

impl Fmt {
    pub fn bits(self) -> u32 {
        match self {
            Fmt::H => 16,
            Fmt::S => 32,
            Fmt::D => 64,
        }
    }

    /// Fraction bits.
    pub fn frac(self) -> u32 {
        match self {
            Fmt::H => 10,
            Fmt::S => 23,
            Fmt::D => 52,
        }
    }

    pub fn exp_bits(self) -> u32 {
        self.bits() - self.frac() - 1
    }

    pub fn bias(self) -> i32 {
        (1 << (self.exp_bits() - 1)) - 1
    }

    /// Exponent of the smallest normal number.
    pub fn min_exp(self) -> i32 {
        1 - self.bias()
    }

    pub fn mask(self) -> u64 {
        if self.bits() == 64 {
            u64::MAX
        } else {
            (1u64 << self.bits()) - 1
        }
    }

    fn sign_bit(self) -> u64 {
        1u64 << (self.bits() - 1)
    }

    pub fn default_nan(self) -> u64 {
        (((1u64 << self.exp_bits()) - 1) << self.frac()) | (1u64 << (self.frac() - 1))
    }

    pub fn infinity(self, sign: bool) -> u64 {
        (u64::from(sign) * self.sign_bit()) | (((1u64 << self.exp_bits()) - 1) << self.frac())
    }

    pub fn zero(self, sign: bool) -> u64 {
        u64::from(sign) * self.sign_bit()
    }

    pub fn max_normal(self, sign: bool) -> u64 {
        (u64::from(sign) * self.sign_bit())
            | (((1u64 << self.exp_bits()) - 2) << self.frac())
            | ((1u64 << self.frac()) - 1)
    }

    /// Is `a` a NaN?
    pub fn is_nan(self, a: u64) -> bool {
        let f = self.frac();
        let e = (a >> f) & ((1 << self.exp_bits()) - 1);
        e == (1 << self.exp_bits()) - 1 && a & ((1u64 << f) - 1) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// To nearest, ties to even (`RN`).
    Nearest,
    /// Towards plus infinity (`RP`).
    PlusInf,
    /// Towards minus infinity (`RM`).
    MinusInf,
    /// Towards zero (`RZ`).
    Zero,
    /// To nearest, ties away from zero (`FCVTAS`, `FRINTA`).
    NearestAway,
    /// Round to odd (`FCVTXN`).
    Odd,
}

/// The floating-point control and status registers an operation reads and updates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Env {
    pub fpcr: u32,
    /// Cumulative flags accumulate here.
    pub fpsr: u32,
}

impl Env {
    pub fn new(fpcr: u32, fpsr: u32) -> Env {
        Env { fpcr, fpsr }
    }

    pub fn rounding(&self) -> Rounding {
        match (self.fpcr >> 22) & 3 {
            0 => Rounding::Nearest,
            1 => Rounding::PlusInf,
            2 => Rounding::MinusInf,
            _ => Rounding::Zero,
        }
    }

    pub fn dn(&self) -> bool {
        self.fpcr & (1 << 25) != 0
    }

    pub fn fz(&self) -> bool {
        self.fpcr & (1 << 24) != 0
    }

    pub fn ahp(&self) -> bool {
        self.fpcr & (1 << 26) != 0
    }

    pub fn raise(&mut self, flags: u32) {
        self.fpsr |= flags;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Zero,
    Inf,
    QNan,
    SNan,
    Num,
}

/// An unpacked operand. For `Num`, the value is `sig / 2^63 * 2^exp` with the top bit of `sig`
/// set; for NaNs `bits` keeps the original encoding.
#[derive(Clone, Copy, Debug)]
struct Unp {
    kind: Kind,
    sign: bool,
    exp: i32,
    sig: u64,
    bits: u64,
}

impl Unp {
    fn is_nan(&self) -> bool {
        matches!(self.kind, Kind::QNan | Kind::SNan)
    }
}

fn unpack(env: &mut Env, fmt: Fmt, bits: u64) -> Unp {
    unpack_ahp(env, fmt, bits, false)
}

/// `FPUnpack`; with `ahp`, a half-precision value uses the alternative format (no infinities or
/// NaNs, the top exponent is an ordinary one).
fn unpack_ahp(env: &mut Env, fmt: Fmt, bits: u64, ahp: bool) -> Unp {
    let f = fmt.frac();
    let sign = (bits >> (fmt.bits() - 1)) & 1 != 0;
    let exp_field = ((bits >> f) & ((1 << fmt.exp_bits()) - 1)) as i32;
    let frac = bits & ((1u64 << f) - 1);
    let all_ones = (1 << fmt.exp_bits()) - 1;
    let mk = |kind| Unp {
        kind,
        sign,
        exp: 0,
        sig: 0,
        bits,
    };
    if exp_field == 0 {
        if frac == 0 {
            return mk(Kind::Zero);
        }
        if env.fz() && fmt != Fmt::H {
            env.raise(IDC);
            return mk(Kind::Zero);
        }
        let len = 64 - frac.leading_zeros();
        return Unp {
            kind: Kind::Num,
            sign,
            exp: fmt.min_exp() - f as i32 + len as i32 - 1,
            sig: frac << (64 - len),
            bits,
        };
    }
    if exp_field == all_ones && !(ahp && fmt == Fmt::H) {
        if frac == 0 {
            return mk(Kind::Inf);
        }
        return mk(if (frac >> (f - 1)) & 1 != 0 {
            Kind::QNan
        } else {
            Kind::SNan
        });
    }
    Unp {
        kind: Kind::Num,
        sign,
        exp: exp_field - fmt.bias(),
        sig: ((1u64 << f) | frac) << (63 - f),
        bits,
    }
}

/// `FPProcessNaN`.
fn process_nan(env: &mut Env, fmt: Fmt, u: &Unp) -> u64 {
    if u.kind == Kind::SNan {
        env.raise(IOC);
    }
    if env.dn() {
        fmt.default_nan()
    } else {
        u.bits | (1u64 << (fmt.frac() - 1))
    }
}

/// `FPProcessNaNs`/`FPProcessNaNs3`: the first signalling NaN, else the first quiet NaN.
fn process_nans(env: &mut Env, fmt: Fmt, ops: &[&Unp]) -> Option<u64> {
    let pick = ops
        .iter()
        .find(|u| u.kind == Kind::SNan)
        .or_else(|| ops.iter().find(|u| u.kind == Kind::QNan))?;
    Some(process_nan(env, fmt, pick))
}

fn invalid(env: &mut Env, fmt: Fmt) -> u64 {
    env.raise(IOC);
    fmt.default_nan()
}

fn shr_sticky(x: u128, n: u32) -> (u128, bool) {
    if n == 0 {
        (x, false)
    } else if n >= 128 {
        (0, x != 0)
    } else {
        (x >> n, (x << (128 - n)) != 0)
    }
}

/// `FPRound`: round the exact value `m * 2^(e - 127)` (`m` normalized with bit 127 set, plus a
/// sticky bit for anything below) to `fmt`.
fn round_pack(
    env: &mut Env,
    fmt: Fmt,
    sign: bool,
    e: i32,
    m: u128,
    sticky: bool,
    rounding: Rounding,
) -> u64 {
    debug_assert!(m >> 127 == 1);
    let f = fmt.frac();
    let min_exp = fmt.min_exp();
    if env.fz() && fmt != Fmt::H && e < min_exp {
        // Flush-to-zero: the result would be denormal before rounding.
        env.raise(UFC);
        return fmt.zero(sign);
    }
    let mut biased = (e - min_exp + 1).max(0);
    let (mant, lost) = if biased == 0 {
        let (v, s) = shr_sticky(m, (min_exp - e) as u32);
        (v, s || sticky)
    } else {
        (m, sticky)
    };
    // Integer mantissa and the part below it.
    let shift = 127 - f;
    let mut int_mant = (mant >> shift) as u64;
    let rest = mant & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    let inexact = rest != 0 || lost;
    if biased == 0 && inexact {
        env.raise(UFC);
    }
    let above_half = rest > half || (rest == half && lost);
    let at_half = rest == half && !lost;
    let round_up = match rounding {
        Rounding::Nearest => above_half || (at_half && int_mant & 1 == 1),
        Rounding::NearestAway => above_half || at_half,
        Rounding::PlusInf => inexact && !sign,
        Rounding::MinusInf => inexact && sign,
        Rounding::Zero | Rounding::Odd => false,
    };
    let overflow_to_inf = match rounding {
        Rounding::Nearest | Rounding::NearestAway => true,
        Rounding::PlusInf => !sign,
        Rounding::MinusInf => sign,
        Rounding::Zero | Rounding::Odd => false,
    };
    if round_up {
        int_mant += 1;
        if int_mant == 1 << f {
            // A denormal rounded up to the smallest normal number.
            biased = 1;
        }
        if int_mant == 1 << (f + 1) {
            biased += 1;
            int_mant >>= 1;
        }
    } else if rounding == Rounding::Odd && inexact {
        int_mant |= 1;
    }
    let all_ones = (1i32 << fmt.exp_bits()) - 1;
    if biased >= all_ones {
        env.raise(OFC | IXC);
        return if overflow_to_inf {
            fmt.infinity(sign)
        } else {
            fmt.max_normal(sign)
        };
    }
    if inexact {
        env.raise(IXC);
    }
    (u64::from(sign) * fmt.sign_bit()) | ((biased as u64) << f) | (int_mant & ((1u64 << f) - 1))
}

/// An exact result: sign, exponent of the top bit, significand with bit 127 set, sticky.
type Exact = (bool, i32, u128, bool);

fn pack_exact(env: &mut Env, fmt: Fmt, x: Exact, rounding: Rounding) -> u64 {
    round_pack(env, fmt, x.0, x.1, x.2, x.3, rounding)
}

/// Exact sum of two frames. Each is `(sign, exponent, significand with bit 127 set)` whose low
/// bit is zero (significands of at most 127 bits), so that one bit of headroom for a carry costs
/// nothing. `None` when the sum is exactly zero.
fn add_frames(a: (bool, i32, u128), b: (bool, i32, u128)) -> Option<Exact> {
    let (big, small) = if (a.1, a.2) >= (b.1, b.2) {
        (a, b)
    } else {
        (b, a)
    };
    let d = (big.1 - small.1) as u32;
    let bm = big.2 >> 1;
    let (sm, st) = shr_sticky(small.2 >> 1, d);
    let sum = if big.0 == small.0 {
        bm + sm
    } else {
        let diff = bm - sm - u128::from(st);
        if diff == 0 && !st {
            return None;
        }
        diff
    };
    let lz = sum.leading_zeros();
    Some((big.0, big.1 + 1 - lz as i32, sum << lz, st))
}

fn frame(u: &Unp) -> (bool, i32, u128) {
    (u.sign, u.exp, u128::from(u.sig) << 64)
}

fn mul_exact(a: &Unp, b: &Unp) -> Exact {
    let p = u128::from(a.sig) * u128::from(b.sig);
    let sign = a.sign != b.sign;
    if p >> 127 == 1 {
        (sign, a.exp + b.exp + 1, p, false)
    } else {
        (sign, a.exp + b.exp, p << 1, false)
    }
}

fn div_exact(a: &Unp, b: &Unp) -> Exact {
    let n = u128::from(a.sig) << 64;
    let d = u128::from(b.sig);
    let q = n / d;
    let sticky = q * d != n;
    let sign = a.sign != b.sign;
    if q >> 64 != 0 {
        (sign, a.exp - b.exp, q << 63, sticky)
    } else {
        (sign, a.exp - b.exp - 1, q << 64, sticky)
    }
}

/// Integer square root (floor).
fn isqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let mut rem = n;
    let mut root = 0u128;
    let mut bit = 1u128 << ((127 - n.leading_zeros()) & !1);
    while bit != 0 {
        if rem >= root + bit {
            rem -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

fn sqrt_exact(a: &Unp) -> Exact {
    let parity = a.exp & 1;
    let k = (a.exp - parity) / 2;
    let n = u128::from(a.sig) << (63 + parity);
    let r = isqrt(n);
    (false, k, r << 64, r * r != n)
}

// ---------------------------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------------------------

/// A finite non-zero operand rounded into the format it came from.
fn pack_unp(env: &mut Env, fmt: Fmt, u: &Unp, rounding: Rounding) -> u64 {
    round_pack(
        env,
        fmt,
        u.sign,
        u.exp,
        u128::from(u.sig) << 64,
        false,
        rounding,
    )
}

fn add_unpacked(env: &mut Env, fmt: Fmt, ua: &Unp, ub: &Unp) -> u64 {
    if let Some(r) = process_nans(env, fmt, &[ua, ub]) {
        return r;
    }
    let rounding = env.rounding();
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Inf) if ua.sign != ub.sign => invalid(env, fmt),
        (Kind::Inf, _) => fmt.infinity(ua.sign),
        (_, Kind::Inf) => fmt.infinity(ub.sign),
        (Kind::Zero, Kind::Zero) => {
            if ua.sign == ub.sign {
                fmt.zero(ua.sign)
            } else {
                fmt.zero(rounding == Rounding::MinusInf)
            }
        }
        (Kind::Zero, _) => pack_unp(env, fmt, ub, rounding),
        (_, Kind::Zero) => pack_unp(env, fmt, ua, rounding),
        _ => match add_frames(frame(ua), frame(ub)) {
            Some(x) => pack_exact(env, fmt, x, rounding),
            None => fmt.zero(rounding == Rounding::MinusInf),
        },
    }
}

/// `FPAdd`.
pub fn add(env: &mut Env, fmt: Fmt, a: u64, b: u64) -> u64 {
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    add_unpacked(env, fmt, &ua, &ub)
}

/// `FPSub`.
pub fn sub(env: &mut Env, fmt: Fmt, a: u64, b: u64) -> u64 {
    let ua = unpack(env, fmt, a);
    let mut ub = unpack(env, fmt, b);
    ub.sign = !ub.sign;
    add_unpacked(env, fmt, &ua, &ub)
}

/// `FPMul`.
pub fn mul(env: &mut Env, fmt: Fmt, a: u64, b: u64) -> u64 {
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    if let Some(r) = process_nans(env, fmt, &[&ua, &ub]) {
        return r;
    }
    let rounding = env.rounding();
    let sign = ua.sign != ub.sign;
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Zero) | (Kind::Zero, Kind::Inf) => invalid(env, fmt),
        (Kind::Inf, _) | (_, Kind::Inf) => fmt.infinity(sign),
        (Kind::Zero, _) | (_, Kind::Zero) => fmt.zero(sign),
        _ => pack_exact(env, fmt, mul_exact(&ua, &ub), rounding),
    }
}

/// `FPMulX`: like `FPMul`, but infinity times zero is 2.0 with the sign of the product.
pub fn mul_extended(env: &mut Env, fmt: Fmt, a: u64, b: u64) -> u64 {
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    if let Some(r) = process_nans(env, fmt, &[&ua, &ub]) {
        return r;
    }
    if matches!(
        (ua.kind, ub.kind),
        (Kind::Inf, Kind::Zero) | (Kind::Zero, Kind::Inf)
    ) {
        return two(fmt, ua.sign != ub.sign);
    }
    mul(env, fmt, a, b)
}

/// `FPDiv`.
pub fn div(env: &mut Env, fmt: Fmt, a: u64, b: u64) -> u64 {
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    if let Some(r) = process_nans(env, fmt, &[&ua, &ub]) {
        return r;
    }
    let rounding = env.rounding();
    let sign = ua.sign != ub.sign;
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Inf) | (Kind::Zero, Kind::Zero) => invalid(env, fmt),
        (Kind::Inf, _) => fmt.infinity(sign),
        (_, Kind::Zero) => {
            env.raise(DZC);
            fmt.infinity(sign)
        }
        (Kind::Zero, _) | (_, Kind::Inf) => fmt.zero(sign),
        _ => pack_exact(env, fmt, div_exact(&ua, &ub), rounding),
    }
}

/// `FPSqrt`.
///
/// A positive normal number, with rounding to nearest, is done by the host: its square root is
/// correctly rounded like the architecture's, and the only flag it can raise is "inexact", which
/// is told by checking whether the root squared is the operand. The bit-by-bit root of the general
/// path took over two thirds of the time of the title's main thread in a scene.
pub fn sqrt(env: &mut Env, fmt: Fmt, a: u64) -> u64 {
    if env.rounding() == Rounding::Nearest {
        match fmt {
            Fmt::S => {
                let x = f32::from_bits(a as u32);
                if x.is_normal() && x > 0.0 {
                    let r = x.sqrt();
                    // A root of 24 bits squared is exact in a double.
                    let wide = f64::from(r);
                    if wide * wide != f64::from(x) {
                        env.raise(IXC);
                    }
                    return u64::from(r.to_bits());
                }
            }
            Fmt::D => {
                let x = f64::from_bits(a);
                if x.is_normal() && x > 0.0 {
                    let r = x.sqrt();
                    if !square_is_exact(r, x) {
                        env.raise(IXC);
                    }
                    return r.to_bits();
                }
            }
            Fmt::H => {}
        }
    }
    sqrt_general(env, fmt, a)
}

/// Is `r * r == x` exactly, for normal positive `r` and `x`?
fn square_is_exact(r: f64, x: f64) -> bool {
    // A normal number is `m * 2^e` with `m` of 53 bits.
    let parts = |v: f64| -> (u128, i32) {
        let bits = v.to_bits();
        (
            u128::from((bits & ((1 << 52) - 1)) | (1 << 52)),
            ((bits >> 52) & 0x7FF) as i32 - 1075,
        )
    };
    let (mr, er) = parts(r);
    let (mx, ex) = parts(x);
    // `mr * mr` has 105 or 106 bits and `mx` 53: they can only meet by shifting `mx` up.
    let shift = ex - 2 * er;
    (0..=60).contains(&shift) && mr * mr == mx << shift
}

/// `FPSqrt` by the exact path, for every operand.
fn sqrt_general(env: &mut Env, fmt: Fmt, a: u64) -> u64 {
    let ua = unpack(env, fmt, a);
    if ua.is_nan() {
        return process_nan(env, fmt, &ua);
    }
    let rounding = env.rounding();
    match ua.kind {
        Kind::Zero => fmt.zero(ua.sign),
        Kind::Inf if !ua.sign => fmt.infinity(false),
        _ if ua.sign => invalid(env, fmt),
        _ => pack_exact(env, fmt, sqrt_exact(&ua), rounding),
    }
}

/// `FPMulAdd(addend, op1, op2)`: `addend + op1 * op2` with one rounding.
pub fn mul_add(env: &mut Env, fmt: Fmt, addend: u64, op1: u64, op2: u64) -> u64 {
    mul_add_scaled(env, fmt, addend, op1, op2, false)
}

/// `mul_add` with the exact result halved before it is rounded when `halve` is set: the fused
/// step of `FRSQRTS`, `(3 - a * b) / 2`.
fn mul_add_scaled(env: &mut Env, fmt: Fmt, addend: u64, op1: u64, op2: u64, halve: bool) -> u64 {
    let scale = |x: Exact| -> Exact { (x.0, x.1 - i32::from(halve), x.2, x.3) };
    let ua = unpack(env, fmt, addend);
    let u1 = unpack(env, fmt, op1);
    let u2 = unpack(env, fmt, op2);
    let rounding = env.rounding();
    let invalid_product = matches!(
        (u1.kind, u2.kind),
        (Kind::Inf, Kind::Zero) | (Kind::Zero, Kind::Inf)
    );
    // A quiet NaN addend with an invalid product gives the default NaN, not the addend.
    if ua.kind == Kind::QNan && invalid_product {
        return invalid(env, fmt);
    }
    if let Some(r) = process_nans(env, fmt, &[&ua, &u1, &u2]) {
        return r;
    }
    if invalid_product {
        return invalid(env, fmt);
    }
    let psign = u1.sign != u2.sign;
    let p_inf = u1.kind == Kind::Inf || u2.kind == Kind::Inf;
    let p_zero = u1.kind == Kind::Zero || u2.kind == Kind::Zero;
    if ua.kind == Kind::Inf && p_inf && ua.sign != psign {
        return invalid(env, fmt);
    }
    if ua.kind == Kind::Inf {
        return fmt.infinity(ua.sign);
    }
    if p_inf {
        return fmt.infinity(psign);
    }
    if p_zero {
        if ua.kind == Kind::Zero {
            return if ua.sign == psign {
                fmt.zero(psign)
            } else {
                fmt.zero(rounding == Rounding::MinusInf)
            };
        }
        let (s, e, m) = frame(&ua);
        return pack_exact(env, fmt, scale((s, e, m, false)), rounding);
    }
    let (ps, pe, pm, _) = mul_exact(&u1, &u2);
    if ua.kind == Kind::Zero {
        return pack_exact(env, fmt, scale((ps, pe, pm, false)), rounding);
    }
    match add_frames((ps, pe, pm), frame(&ua)) {
        Some(x) => pack_exact(env, fmt, scale(x), rounding),
        None => fmt.zero(rounding == Rounding::MinusInf),
    }
}

// ---------------------------------------------------------------------------------------------
// Comparison, minimum and maximum
// ---------------------------------------------------------------------------------------------

/// Numeric order of two non-NaN operands (`-0 == +0`).
fn ordering(a: &Unp, b: &Unp) -> std::cmp::Ordering {
    use std::cmp::Ordering::Equal;
    let sgn = |u: &Unp| {
        if u.kind == Kind::Zero {
            0i8
        } else if u.sign {
            -1
        } else {
            1
        }
    };
    let key = |u: &Unp| -> (i8, i32, u64) {
        match u.kind {
            Kind::Num => (1, u.exp, u.sig),
            Kind::Inf => (2, 0, 0),
            _ => (0, 0, 0),
        }
    };
    let (sa, sb) = (sgn(a), sgn(b));
    if sa != sb {
        return sa.cmp(&sb);
    }
    if sa == 0 {
        return Equal;
    }
    let by_magnitude = key(a).cmp(&key(b));
    if sa < 0 {
        by_magnitude.reverse()
    } else {
        by_magnitude
    }
}

/// `FPCompare`: the `NZCV` nibble. `signal_all_nans` is `FCMPE`.
pub fn compare(env: &mut Env, fmt: Fmt, a: u64, b: u64, signal_all_nans: bool) -> u8 {
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    if ua.is_nan() || ub.is_nan() {
        if signal_all_nans || ua.kind == Kind::SNan || ub.kind == Kind::SNan {
            env.raise(IOC);
        }
        return 0b0011;
    }
    match ordering(&ua, &ub) {
        std::cmp::Ordering::Equal => 0b0110,
        std::cmp::Ordering::Less => 0b1000,
        std::cmp::Ordering::Greater => 0b0010,
    }
}

/// Quiet comparisons for the vector forms: `Some(ordering)`, or `None` when unordered (`IOC` for
/// signalling NaNs, or for any NaN with `signal_all_nans`).
pub fn compare_order(
    env: &mut Env,
    fmt: Fmt,
    a: u64,
    b: u64,
    signal_all_nans: bool,
) -> Option<std::cmp::Ordering> {
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    if ua.is_nan() || ub.is_nan() {
        if signal_all_nans || ua.kind == Kind::SNan || ub.kind == Kind::SNan {
            env.raise(IOC);
        }
        return None;
    }
    Some(ordering(&ua, &ub))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MinMax {
    Max,
    Min,
    MaxNum,
    MinNum,
}

/// `FPMax`, `FPMin`, `FPMaxNum`, `FPMinNum`.
pub fn min_max(env: &mut Env, fmt: Fmt, op: MinMax, a: u64, b: u64) -> u64 {
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    let num = matches!(op, MinMax::MaxNum | MinMax::MinNum);
    if num {
        // A quiet NaN against a number stands for "no value": the number wins.
        if ua.kind == Kind::QNan && !ub.is_nan() {
            return finish_num(fmt, &ub, b);
        }
        if ub.kind == Kind::QNan && !ua.is_nan() {
            return finish_num(fmt, &ua, a);
        }
    }
    if let Some(r) = process_nans(env, fmt, &[&ua, &ub]) {
        return r;
    }
    let is_max = matches!(op, MinMax::Max | MinMax::MaxNum);
    let pick_a = match ordering(&ua, &ub) {
        std::cmp::Ordering::Equal => {
            // Opposite-signed zeros: +0 is the larger.
            ua.sign == ub.sign || ua.sign != is_max
        }
        std::cmp::Ordering::Greater => is_max,
        std::cmp::Ordering::Less => !is_max,
    };
    if pick_a {
        finish_num(fmt, &ua, a)
    } else {
        finish_num(fmt, &ub, b)
    }
}

/// An operand returned as a result: a denormal flushed on input comes back as a zero.
fn finish_num(fmt: Fmt, u: &Unp, bits: u64) -> u64 {
    if u.kind == Kind::Zero {
        fmt.zero(u.sign)
    } else {
        bits & fmt.mask()
    }
}

// ---------------------------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------------------------

/// `FPConvert`: between formats, rounding as `FPCR` says. With `FPCR.AHP` the half-precision side
/// uses the alternative format.
pub fn convert(env: &mut Env, from: Fmt, to: Fmt, a: u64) -> u64 {
    let rounding = env.rounding();
    convert_with(env, from, to, a, rounding)
}

pub fn convert_with(env: &mut Env, from: Fmt, to: Fmt, a: u64, rounding: Rounding) -> u64 {
    let ahp = env.ahp();
    let u = unpack_ahp(env, from, a, ahp && from == Fmt::H);
    let alt_to = ahp && to == Fmt::H;
    match u.kind {
        Kind::SNan | Kind::QNan => {
            if alt_to {
                // The alternative format has no NaN: zero, flagged invalid.
                env.raise(IOC);
                return to.zero(u.sign);
            }
            if u.kind == Kind::SNan {
                env.raise(IOC);
            }
            if env.dn() {
                return to.default_nan();
            }
            // Keep the top bits of the payload, quieted.
            let (ff, ft) = (from.frac(), to.frac());
            let frac = u.bits & ((1u64 << ff) - 1);
            let frac = if ft >= ff {
                frac << (ft - ff)
            } else {
                frac >> (ff - ft)
            };
            (u64::from(u.sign) * to.sign_bit())
                | (((1u64 << to.exp_bits()) - 1) << ft)
                | frac
                | (1u64 << (ft - 1))
        }
        Kind::Inf => {
            if alt_to {
                env.raise(IOC);
                return (u64::from(u.sign) << 15) | 0x7FFF;
            }
            to.infinity(u.sign)
        }
        Kind::Zero => to.zero(u.sign),
        Kind::Num => {
            if alt_to {
                return round_pack_alt_half(env, &u, rounding);
            }
            pack_unp(env, to, &u, rounding)
        }
    }
}

/// A finite value to the alternative half-precision format: the top exponent holds ordinary
/// numbers and anything larger saturates to `0x7FFF` with `IOC`.
fn round_pack_alt_half(env: &mut Env, u: &Unp, rounding: Rounding) -> u64 {
    let mut scratch = Env {
        fpcr: env.fpcr & !(1 << 24),
        fpsr: 0,
    };
    // Round with exponent range extended by one binade: compute as binary32, which has room, then
    // pack by hand into the alternative layout.
    let f = Fmt::H.frac();
    let min_exp = Fmt::H.min_exp();
    let m = u128::from(u.sig) << 64;
    let mut biased = (u.exp - min_exp + 1).max(0);
    let (mant, lost) = if biased == 0 {
        shr_sticky(m, (min_exp - u.exp) as u32)
    } else {
        (m, false)
    };
    let shift = 127 - f;
    let mut int_mant = (mant >> shift) as u64;
    let rest = mant & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    let inexact = rest != 0 || lost;
    let up = match rounding {
        Rounding::Nearest => rest > half || (rest == half && (lost || int_mant & 1 == 1)),
        Rounding::NearestAway => rest >= half,
        Rounding::PlusInf => inexact && !u.sign,
        Rounding::MinusInf => inexact && u.sign,
        Rounding::Zero | Rounding::Odd => false,
    };
    if up {
        int_mant += 1;
        if int_mant == 1 << f {
            biased = 1;
        }
        if int_mant == 1 << (f + 1) {
            biased += 1;
            int_mant >>= 1;
        }
    }
    let _ = &mut scratch;
    if biased > 31 {
        env.raise(IOC);
        return (u64::from(u.sign) << 15) | 0x7FFF;
    }
    if biased == 0 && inexact {
        env.raise(UFC);
    }
    if inexact {
        env.raise(IXC);
    }
    (u64::from(u.sign) << 15) | ((biased as u64) << f) | (int_mant & ((1u64 << f) - 1))
}

/// `SCVTF`/`UCVTF`: an integer of `bits` bits divided by `2^fbits`, rounded as `FPCR` says.
pub fn from_int(env: &mut Env, fmt: Fmt, value: u64, bits: u32, signed: bool, fbits: u32) -> u64 {
    let rounding = env.rounding();
    let (sign, magnitude) = if signed {
        let v = if bits == 64 {
            i128::from(value as i64)
        } else {
            i128::from(((value << (64 - bits)) as i64) >> (64 - bits))
        };
        (v < 0, v.unsigned_abs())
    } else {
        let m = if bits == 64 {
            value
        } else {
            value & ((1u64 << bits) - 1)
        };
        (false, u128::from(m))
    };
    if magnitude == 0 {
        return fmt.zero(false);
    }
    let lz = magnitude.leading_zeros();
    let e = 127 - lz as i32 - fbits as i32;
    round_pack(env, fmt, sign, e, magnitude << lz, false, rounding)
}

/// Round `|value| * 2^fbits` of a finite non-zero operand to an integer; returns the signed
/// result and whether the rounding changed the value. Magnitudes of `2^127` and more come back as
/// the extreme `i128` values (no register is that wide).
fn round_to_integer(u: &Unp, fbits: u32, rounding: Rounding) -> (i128, bool) {
    let shift = u.exp + fbits as i32 - 63;
    let sig = u128::from(u.sig);
    let signed = |m: u128| -> i128 {
        let v = m.min(i128::MAX as u128) as i128;
        if u.sign {
            -v
        } else {
            v
        }
    };
    if shift >= 0 {
        if shift > 63 {
            return (signed(u128::MAX), false);
        }
        return (signed(sig << shift), false);
    }
    let s = (-shift) as u32;
    let (int_part, below, half) = if s > 127 {
        (0u128, 1u128, 2u128)
    } else {
        (sig >> s, sig & ((1u128 << s) - 1), 1u128 << (s - 1))
    };
    let nonzero = below != 0;
    let up = match rounding {
        Rounding::Nearest => below > half || (below == half && int_part & 1 == 1),
        Rounding::NearestAway => below >= half,
        Rounding::PlusInf => nonzero && !u.sign,
        Rounding::MinusInf => nonzero && u.sign,
        Rounding::Zero => false,
        Rounding::Odd => nonzero && int_part & 1 == 0,
    };
    (signed(int_part + u128::from(up)), nonzero)
}

/// `FPToFixed`: a floating-point value to a signed or unsigned integer of `bits` bits with `fbits`
/// fractional bits, saturating; `IOC` for NaN and overflow, `IXC` for inexact results.
pub fn to_int(
    env: &mut Env,
    fmt: Fmt,
    a: u64,
    bits: u32,
    signed: bool,
    fbits: u32,
    rounding: Rounding,
) -> u64 {
    let u = unpack(env, fmt, a);
    let (max, min): (i128, i128) = if signed {
        ((1i128 << (bits - 1)) - 1, -(1i128 << (bits - 1)))
    } else {
        ((1i128 << bits) - 1, 0)
    };
    let to_bits = |v: i128| -> u64 {
        if bits == 64 {
            v as u64
        } else {
            (v as u64) & ((1u64 << bits) - 1)
        }
    };
    match u.kind {
        Kind::QNan | Kind::SNan => {
            env.raise(IOC);
            0
        }
        Kind::Zero => 0,
        Kind::Inf => {
            env.raise(IOC);
            to_bits(if u.sign { min } else { max })
        }
        Kind::Num => {
            let (v, inexact) = round_to_integer(&u, fbits, rounding);
            if v > max || v < min {
                env.raise(IOC);
                to_bits(if v < 0 { min } else { max })
            } else {
                if inexact {
                    env.raise(IXC);
                }
                to_bits(v)
            }
        }
    }
}

/// `FPRoundInt`: round to an integral value in the same format. `exact` (`FRINTX`) raises `IXC`
/// when the result differs.
pub fn round_int(env: &mut Env, fmt: Fmt, a: u64, rounding: Rounding, exact: bool) -> u64 {
    let u = unpack(env, fmt, a);
    match u.kind {
        Kind::QNan | Kind::SNan => process_nan(env, fmt, &u),
        Kind::Inf => fmt.infinity(u.sign),
        Kind::Zero => fmt.zero(u.sign),
        Kind::Num => {
            if u.exp >= fmt.frac() as i32 {
                // Every value this large is already an integer.
                return a & fmt.mask();
            }
            let (v, inexact) = round_to_integer(&u, 0, rounding);
            if exact && inexact {
                env.raise(IXC);
            }
            if v == 0 {
                return fmt.zero(u.sign);
            }
            let mag = v.unsigned_abs();
            let lz = mag.leading_zeros();
            let mut scratch = Env {
                fpcr: env.fpcr & !(1 << 24),
                fpsr: 0,
            };
            round_pack(
                &mut scratch,
                fmt,
                v < 0,
                127 - lz as i32,
                mag << lz,
                false,
                Rounding::Zero,
            )
        }
    }
}

/// `FPAbs`: the sign cleared, NaNs untouched otherwise.
pub fn abs(fmt: Fmt, a: u64) -> u64 {
    a & fmt.mask() & !fmt.sign_bit()
}

/// `FPNeg`.
pub fn neg(fmt: Fmt, a: u64) -> u64 {
    (a & fmt.mask()) ^ fmt.sign_bit()
}

// ---------------------------------------------------------------------------------------------
// Estimates and Newton-Raphson steps
// ---------------------------------------------------------------------------------------------

fn two(fmt: Fmt, sign: bool) -> u64 {
    // 2.0 = 1.0 * 2^1
    ((fmt.bias() as u64 + 1) << fmt.frac()) | (u64::from(sign) * fmt.sign_bit())
}

fn three(fmt: Fmt) -> u64 {
    // 3.0 = 1.5 * 2^1
    ((fmt.bias() as u64 + 1) << fmt.frac()) | (1u64 << (fmt.frac() - 1))
}

fn one_point_five(fmt: Fmt) -> u64 {
    (fmt.bias() as u64) << fmt.frac() | (1u64 << (fmt.frac() - 1))
}

/// `RecipEstimate` on a value in `[256, 512)`.
fn recip_estimate_table(a: u32) -> u32 {
    let a = a * 2 + 1;
    let b = (1u32 << 19) / a;
    b.div_ceil(2)
}

/// `RecipSqrtEstimate` on a value in `[128, 512)`.
fn recip_sqrt_estimate_table(a: u32) -> u32 {
    let a = if a < 256 {
        a * 2 + 1
    } else {
        (((a >> 1) << 1) + 1) * 2
    };
    let mut b = 512u32;
    while u64::from(a) * u64::from(b + 1) * u64::from(b + 1) < (1u64 << 28) {
        b += 1;
    }
    b.div_ceil(2)
}

/// `UnsignedRecipEstimate` (`URECPE`): operands up to `0x7FFF_FFFF` give all ones.
pub fn unsigned_recip_estimate(operand: u32) -> u32 {
    if operand >> 31 == 0 {
        u32::MAX
    } else {
        recip_estimate_table(operand >> 23) << 23
    }
}

/// `UnsignedRSqrtEstimate` (`URSQRTE`): operands up to `0x3FFF_FFFF` give all ones.
pub fn unsigned_rsqrt_estimate(operand: u32) -> u32 {
    if operand >> 30 == 0 {
        u32::MAX
    } else {
        recip_sqrt_estimate_table(operand >> 23) << 23
    }
}

/// The fraction of `a` left-justified to 52 bits and its biased exponent field.
fn fraction52(fmt: Fmt, a: u64) -> (u64, i32) {
    let f = fmt.frac();
    let frac = (a & ((1u64 << f) - 1)) << (52 - f);
    let exp = ((a >> f) & ((1 << fmt.exp_bits()) - 1)) as i32;
    (frac, exp)
}

/// `FRECPE`.
pub fn recip_estimate(env: &mut Env, fmt: Fmt, a: u64) -> u64 {
    let u = unpack(env, fmt, a);
    let rounding = env.rounding();
    match u.kind {
        Kind::QNan | Kind::SNan => return process_nan(env, fmt, &u),
        Kind::Inf => return fmt.zero(u.sign),
        Kind::Zero => {
            env.raise(DZC);
            return fmt.infinity(u.sign);
        }
        Kind::Num => {}
    }
    let limit = match fmt {
        Fmt::H => -16,
        Fmt::S => -128,
        Fmt::D => -1024,
    };
    if u.exp < limit {
        let to_inf = match rounding {
            Rounding::Nearest | Rounding::NearestAway => true,
            Rounding::PlusInf => !u.sign,
            Rounding::MinusInf => u.sign,
            Rounding::Zero | Rounding::Odd => false,
        };
        env.raise(OFC | IXC);
        return if to_inf {
            fmt.infinity(u.sign)
        } else {
            fmt.max_normal(u.sign)
        };
    }
    if env.fz()
        && fmt != Fmt::H
        && ((fmt == Fmt::S && u.exp >= 126) || (fmt == Fmt::D && u.exp >= 1022))
    {
        env.raise(UFC);
        return fmt.zero(u.sign);
    }
    let (mut fraction, mut exp) = fraction52(fmt, a);
    if exp == 0 {
        if fraction >> 51 & 1 == 0 {
            exp = -1;
            fraction = (fraction & ((1 << 50) - 1)) << 2;
        } else {
            fraction = (fraction & ((1 << 51) - 1)) << 1;
        }
    }
    let scaled = (1 << 8) | (fraction >> 44) as u32;
    let mut result_exp = match fmt {
        Fmt::H => 29 - exp,
        Fmt::S => 253 - exp,
        Fmt::D => 2045 - exp,
    };
    let estimate = recip_estimate_table(scaled);
    let mut fraction = u64::from(estimate & 0xFF) << 44;
    if result_exp == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exp = 0;
    }
    let f = fmt.frac();
    (u64::from(u.sign) * fmt.sign_bit()) | ((result_exp as u64) << f) | (fraction >> (52 - f))
}

/// `FRSQRTE`.
pub fn rsqrt_estimate(env: &mut Env, fmt: Fmt, a: u64) -> u64 {
    let u = unpack(env, fmt, a);
    match u.kind {
        Kind::QNan | Kind::SNan => return process_nan(env, fmt, &u),
        Kind::Zero => {
            env.raise(DZC);
            return fmt.infinity(u.sign);
        }
        _ if u.sign => return invalid(env, fmt),
        Kind::Inf => return fmt.zero(false),
        Kind::Num => {}
    }
    let (mut fraction, mut exp) = fraction52(fmt, a);
    if exp == 0 {
        while fraction >> 51 & 1 == 0 {
            fraction = (fraction & ((1 << 51) - 1)) << 1;
            exp -= 1;
        }
        fraction = (fraction & ((1 << 51) - 1)) << 1;
    }
    let scaled = if exp & 1 == 0 {
        (1 << 8) | (fraction >> 44) as u32
    } else {
        (1 << 7) | (fraction >> 45) as u32
    };
    let result_exp = match fmt {
        Fmt::H => (44 - exp).div_euclid(2),
        Fmt::S => (380 - exp).div_euclid(2),
        Fmt::D => (3068 - exp).div_euclid(2),
    };
    let estimate = recip_sqrt_estimate_table(scaled);
    let f = fmt.frac();
    ((result_exp as u64) << f) | (u64::from(estimate & 0xFF) << (f - 8))
}

/// `FRECPX`.
pub fn recip_exponent(env: &mut Env, fmt: Fmt, a: u64) -> u64 {
    let u = unpack(env, fmt, a);
    if u.is_nan() {
        return process_nan(env, fmt, &u);
    }
    let f = fmt.frac();
    let exp = (a >> f) & ((1 << fmt.exp_bits()) - 1);
    let all = (1u64 << fmt.exp_bits()) - 1;
    let new = if exp == 0 { all - 1 } else { !exp & all };
    (a & fmt.sign_bit()) | (new << f)
}

/// `FRECPS`: `2 - a * b`, fused.
pub fn recip_step(env: &mut Env, fmt: Fmt, a: u64, b: u64) -> u64 {
    let a = neg(fmt, a);
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    if let Some(r) = process_nans(env, fmt, &[&ua, &ub]) {
        return r;
    }
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Zero) | (Kind::Zero, Kind::Inf) => two(fmt, false),
        (Kind::Inf, _) | (_, Kind::Inf) => fmt.infinity(ua.sign != ub.sign),
        _ => mul_add(env, fmt, two(fmt, false), a, b),
    }
}

/// `FRSQRTS`: `(3 - a * b) / 2`, fused.
pub fn rsqrt_step(env: &mut Env, fmt: Fmt, a: u64, b: u64) -> u64 {
    let a = neg(fmt, a);
    let (ua, ub) = (unpack(env, fmt, a), unpack(env, fmt, b));
    if let Some(r) = process_nans(env, fmt, &[&ua, &ub]) {
        return r;
    }
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Zero) | (Kind::Zero, Kind::Inf) => one_point_five(fmt),
        (Kind::Inf, _) | (_, Kind::Inf) => fmt.infinity(ua.sign != ub.sign),
        _ => mul_add_scaled(env, fmt, three(fmt), a, b, true),
    }
}
