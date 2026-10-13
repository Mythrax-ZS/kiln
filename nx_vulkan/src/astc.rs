//! ASTC (Adaptive Scalable Texture Compression), the low dynamic range profile: blocks of 128
//! bits decoded to 8-bit RGBA.
//!
//! The format is the one of the Khronos Data Format Specification (its ASTC chapter, which is also
//! the appendix of the `KHR_texture_compression_astc_hdr` extension). A block is a weight grid,
//! optionally at a lower resolution than the block, and one to four partitions, each with a pair
//! of endpoint colors; a texel is the endpoints of its partition mixed by its weight. Values are
//! stored as bounded integer sequences: a number of bits per value, plus a trit or a quint
//! (a base-3 or base-5 digit) shared by groups of five or three values.
//!
//! The decoder is the "LDR profile": endpoint modes for high dynamic range, and any encoding the
//! specification calls illegal, give the error color (magenta), as the specification asks for
//! decoders that cannot do better.

/// The texture header format codes of the ASTC footprints, as their block size in texels.
pub fn footprint(code: u32) -> Option<(u32, u32)> {
    Some(match code {
        0x40 => (4, 4),
        0x50 => (5, 4),
        0x41 => (5, 5),
        0x51 => (6, 5),
        0x42 => (6, 6),
        0x55 => (8, 5),
        0x52 => (8, 6),
        0x44 => (8, 8),
        0x56 => (10, 5),
        0x57 => (10, 6),
        0x53 => (10, 8),
        0x45 => (10, 10),
        0x54 => (12, 10),
        0x46 => (12, 12),
        _ => return None,
    })
}

/// What an illegal block decodes to.
pub const ERROR_COLOR: [u8; 4] = [255, 0, 255, 255];

/// How the values of a range are stored besides their plain bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Extra {
    None,
    Trit,
    Quint,
}

/// The ranges values can be stored in: the number of levels, the plain bits per value, and the
/// trit or quint that goes with them. Weights use the first twelve, endpoints the ones from 6 on.
const RANGES: [(u16, usize, Extra); 21] = [
    (2, 1, Extra::None),
    (3, 0, Extra::Trit),
    (4, 2, Extra::None),
    (5, 0, Extra::Quint),
    (6, 1, Extra::Trit),
    (8, 3, Extra::None),
    (10, 1, Extra::Quint),
    (12, 2, Extra::Trit),
    (16, 4, Extra::None),
    (20, 2, Extra::Quint),
    (24, 3, Extra::Trit),
    (32, 5, Extra::None),
    (40, 3, Extra::Quint),
    (48, 4, Extra::Trit),
    (64, 6, Extra::None),
    (80, 4, Extra::Quint),
    (96, 5, Extra::Trit),
    (128, 7, Extra::None),
    (160, 5, Extra::Quint),
    (192, 6, Extra::Trit),
    (256, 8, Extra::None),
];

/// The index of the first range endpoints may use (0 to 5), and of the last (0 to 255).
const FIRST_ENDPOINT_RANGE: usize = 4;
const LAST_RANGE: usize = RANGES.len() - 1;

/// `len` bits of `v` from bit `lo` (zero beyond the 128 bits).
fn get(v: u128, lo: usize, len: usize) -> u32 {
    if lo >= 128 || len == 0 {
        return 0;
    }
    ((v >> lo) & ((1u128 << len) - 1)) as u32
}

/// The bits a sequence of `count` values of range `range` takes.
fn sequence_bits(count: usize, range: usize) -> usize {
    let (_, bits, extra) = RANGES[range];
    count * bits
        + match extra {
            Extra::None => 0,
            Extra::Trit => (8 * count).div_ceil(5),
            Extra::Quint => (7 * count).div_ceil(3),
        }
}

/// The five trits of the eight bits `t` of a trit group.
fn trits_of(t: u32) -> [u32; 5] {
    let bit = |i: u32| (t >> i) & 1;
    let (c, t3, t4);
    if (t >> 2) & 7 == 7 {
        c = (((t >> 5) & 7) << 2) | (t & 3);
        t4 = 2;
        t3 = 2;
    } else {
        c = t & 0x1F;
        if (t >> 5) & 3 == 3 {
            t4 = 2;
            t3 = bit(7);
        } else {
            t4 = bit(7);
            t3 = (t >> 5) & 3;
        }
    }
    let (t0, t1, t2);
    if c & 3 == 3 {
        t2 = 2;
        t1 = (c >> 4) & 1;
        let c3 = (c >> 3) & 1;
        let c2 = (c >> 2) & 1;
        t0 = (c3 << 1) | (c2 & (c3 ^ 1));
    } else if (c >> 2) & 3 == 3 {
        t2 = 2;
        t1 = 2;
        t0 = c & 3;
    } else {
        t2 = (c >> 4) & 1;
        t1 = (c >> 2) & 3;
        let c1 = (c >> 1) & 1;
        let c0 = c & 1;
        t0 = (c1 << 1) | (c0 & (c1 ^ 1));
    }
    [t0, t1, t2, t3, t4]
}

/// The three quints of the seven bits `q` of a quint group.
fn quints_of(q: u32) -> [u32; 3] {
    let bit = |i: u32| (q >> i) & 1;
    if (q >> 1) & 3 == 3 && (q >> 5) & 3 == 0 {
        let q2 = (bit(0) << 2) | ((bit(4) & (bit(0) ^ 1)) << 1) | (bit(3) & (bit(0) ^ 1));
        return [4, 4, q2];
    }
    let (q2, c);
    if (q >> 1) & 3 == 3 {
        q2 = 4;
        c = (((q >> 3) & 3) << 3) | ((!(q >> 5) & 3) << 1) | bit(0);
    } else {
        q2 = (q >> 5) & 3;
        c = q & 0x1F;
    }
    if c & 7 == 5 {
        [(c >> 3) & 3, 4, q2]
    } else {
        [c & 7, (c >> 3) & 3, q2]
    }
}

/// Decode `count` values of `range` from the sequence that starts at bit `start` of `v`.
fn decode_sequence(v: u128, start: usize, count: usize, range: usize, out: &mut [u8]) {
    let (_, n, extra) = RANGES[range];
    // Only the bits of the sequence: what a last, partial group lacks reads as zero.
    let total = sequence_bits(count, range);
    let seq = if start >= 128 {
        0
    } else {
        let s = v >> start;
        if total >= 128 {
            s
        } else {
            s & ((1u128 << total) - 1)
        }
    };
    match extra {
        Extra::None => {
            for (i, o) in out.iter_mut().enumerate().take(count) {
                *o = get(seq, i * n, n) as u8;
            }
        }
        Extra::Trit => {
            for g in 0..count.div_ceil(5) {
                let base = g * (8 + 5 * n);
                let t = get(seq, base + n, 2)
                    | (get(seq, base + 2 * n + 2, 2) << 2)
                    | (get(seq, base + 3 * n + 4, 1) << 4)
                    | (get(seq, base + 4 * n + 5, 2) << 5)
                    | (get(seq, base + 5 * n + 7, 1) << 7);
                let trits = trits_of(t);
                for (i, trit) in trits.iter().enumerate() {
                    let index = g * 5 + i;
                    if index < count {
                        let m = get(seq, base + i * n + [0, 2, 4, 5, 7][i], n);
                        out[index] = ((trit << n) | m) as u8;
                    }
                }
            }
        }
        Extra::Quint => {
            for g in 0..count.div_ceil(3) {
                let base = g * (7 + 3 * n);
                let q = get(seq, base + n, 3)
                    | (get(seq, base + 2 * n + 3, 2) << 3)
                    | (get(seq, base + 3 * n + 5, 2) << 5);
                let quints = quints_of(q);
                for (i, quint) in quints.iter().enumerate() {
                    let index = g * 3 + i;
                    if index < count {
                        let m = get(seq, base + i * n + [0, 3, 5][i], n);
                        out[index] = ((quint << n) | m) as u8;
                    }
                }
            }
        }
    }
}

/// The bits `v` of `n` bits repeated to fill `to` bits (from the most significant end).
fn replicate(v: u32, n: usize, to: usize) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut out = 0u32;
    let mut filled = 0;
    while filled < to {
        out = (out << n) | v;
        filled += n;
    }
    out >> (filled - to)
}

/// An endpoint value of `range` (stored as `v`) as the 8-bit value it stands for.
fn unquantize_endpoint(range: usize, v: u32) -> u32 {
    let (_, n, extra) = RANGES[range];
    if extra == Extra::None {
        return replicate(v, n, 8);
    }
    let d = v >> n;
    let m = v & ((1 << n) - 1);
    let bit = |i: u32| (m >> i) & 1;
    let a: u32 = if bit(0) != 0 { 0x1FF } else { 0 };
    // The constants of the specification's table: C, and B from the extra bits (named a, b, c,
    // ... from the lowest).
    let (b, c): (u32, u32) = match (n, extra) {
        (1, Extra::Trit) => (0, 204),
        (1, Extra::Quint) => (0, 113),
        (2, Extra::Trit) => (
            (bit(1) << 8) | (bit(1) << 4) | (bit(1) << 2) | (bit(1) << 1),
            93,
        ),
        (2, Extra::Quint) => ((bit(1) << 8) | (bit(1) << 3) | (bit(1) << 2), 54),
        (3, Extra::Trit) => (
            (bit(2) << 8) | (bit(1) << 7) | (bit(2) << 3) | (bit(1) << 2) | (bit(2) << 1) | bit(1),
            44,
        ),
        (3, Extra::Quint) => (
            (bit(2) << 8) | (bit(1) << 7) | (bit(2) << 2) | (bit(1) << 1) | bit(2),
            26,
        ),
        (4, Extra::Trit) => (
            (bit(3) << 8) | (bit(2) << 7) | (bit(1) << 6) | (bit(3) << 2) | (bit(2) << 1) | bit(1),
            22,
        ),
        (4, Extra::Quint) => (
            (bit(3) << 8) | (bit(2) << 7) | (bit(1) << 6) | (bit(3) << 1) | bit(2),
            13,
        ),
        (5, Extra::Trit) => (
            (bit(4) << 8) | (bit(3) << 7) | (bit(2) << 6) | (bit(1) << 5) | (bit(4) << 1) | bit(3),
            11,
        ),
        (5, Extra::Quint) => (
            (bit(4) << 8) | (bit(3) << 7) | (bit(2) << 6) | (bit(1) << 5) | bit(4),
            6,
        ),
        (6, Extra::Trit) => (
            (bit(5) << 8) | (bit(4) << 7) | (bit(3) << 6) | (bit(2) << 5) | (bit(1) << 4) | bit(5),
            5,
        ),
        _ => unreachable!("no such endpoint range"),
    };
    let mut t = d * c + b;
    t ^= a;
    (a & 0x80) | (t >> 2)
}

/// A weight of `range` (stored as `v`) as a number from 0 to 64.
fn unquantize_weight(range: usize, v: u32) -> u32 {
    let (_, n, extra) = RANGES[range];
    let t = match extra {
        Extra::None => replicate(v, n, 6),
        Extra::Trit | Extra::Quint if n == 0 => match (extra, v) {
            (Extra::Trit, d) => [0, 32, 63][d as usize],
            (_, d) => [0, 16, 32, 47, 63][d as usize],
        },
        _ => {
            let d = v >> n;
            let m = v & ((1 << n) - 1);
            let bit = |i: u32| (m >> i) & 1;
            let a: u32 = if bit(0) != 0 { 0x7F } else { 0 };
            let (b, c): (u32, u32) = match (n, extra) {
                (1, Extra::Trit) => (0, 50),
                (1, _) => (0, 28),
                (2, Extra::Trit) => ((bit(1) << 6) | (bit(1) << 2) | bit(1), 23),
                (2, _) => ((bit(1) << 6) | (bit(1) << 1), 13),
                (3, _) => ((bit(2) << 6) | (bit(1) << 5) | (bit(2) << 1) | bit(1), 11),
                _ => unreachable!("no such weight range"),
            };
            let mut t = d * c + b;
            t ^= a;
            (a & 0x20) | (t >> 2)
        }
    };
    if t > 32 {
        t + 1
    } else {
        t
    }
}

/// The weight grid, whether there are two planes of weights, and the range of the weights, of a
/// block mode (the low eleven bits of a block), if it is a legal one.
fn block_mode(mode: u32) -> Option<(usize, usize, bool, usize)> {
    let bit = |i: u32| (mode >> i) & 1;
    let a = (mode >> 5) & 3;
    let mut h = bit(9);
    let mut d = bit(10);
    let r0 = bit(4);
    let (w, hgt, r);
    if mode & 3 != 0 {
        r = r0 | ((mode & 3) << 1);
        let b = (mode >> 7) & 3;
        match (mode >> 2) & 3 {
            0 => {
                w = b + 4;
                hgt = a + 2;
            }
            1 => {
                w = b + 8;
                hgt = a + 2;
            }
            2 => {
                w = a + 2;
                hgt = b + 8;
            }
            _ => {
                let b = b & 1;
                if bit(8) != 0 {
                    w = b + 2;
                    hgt = a + 2;
                } else {
                    w = a + 2;
                    hgt = b + 6;
                }
            }
        }
    } else {
        let rr = (mode >> 2) & 3;
        if rr == 0 {
            return None;
        }
        r = r0 | (rr << 1);
        let b = (mode >> 9) & 3;
        match (mode >> 7) & 3 {
            0 => {
                w = 12;
                hgt = a + 2;
            }
            1 => {
                w = a + 2;
                hgt = 12;
            }
            2 => {
                w = a + 6;
                hgt = b + 6;
                d = 0;
                h = 0;
            }
            _ => match a {
                0 => {
                    w = 6;
                    hgt = 10;
                }
                1 => {
                    w = 10;
                    hgt = 6;
                }
                _ => return None,
            },
        }
    }
    let range = (r as usize).checked_sub(2)? + 6 * h as usize;
    Some((w as usize, hgt as usize, d != 0, range))
}

/// The partition a texel of a block belongs to (the specification's pattern generator).
fn partition_of(seed: u32, x: u32, y: u32, count: usize, small: bool) -> usize {
    let (x, y) = if small { (x << 1, y << 1) } else { (x, y) };
    let seed = seed + (count as u32 - 1) * 1024;
    let mut p = seed;
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    let rnum = p;
    let nib = |shift: u32| (rnum >> shift) & 0xF;
    let mut s = [
        nib(0),
        nib(4),
        nib(8),
        nib(12),
        nib(16),
        nib(20),
        nib(24),
        nib(28),
        nib(18),
        nib(22),
        nib(26),
        rnum.rotate_left(2) & 0xF,
    ];
    for v in &mut s {
        *v *= *v;
    }
    let (sh1, sh2) = if seed & 1 != 0 {
        (
            if seed & 2 != 0 { 4 } else { 5 },
            if count == 3 { 6 } else { 5 },
        )
    } else {
        (
            if count == 3 { 6 } else { 5 },
            if seed & 2 != 0 { 4 } else { 5 },
        )
    };
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
    let shifts = [sh1, sh2, sh1, sh2, sh1, sh2, sh1, sh2, sh3, sh3, sh3, sh3];
    for (v, sh) in s.iter_mut().zip(shifts) {
        *v >>= sh;
    }
    // z is always zero for a 2D block.
    let a = (s[0] * x + s[1] * y + (rnum >> 14)) & 0x3F;
    let b = (s[2] * x + s[3] * y + (rnum >> 10)) & 0x3F;
    let mut c = (s[4] * x + s[5] * y + (rnum >> 6)) & 0x3F;
    let mut d = (s[6] * x + s[7] * y + (rnum >> 2)) & 0x3F;
    if count < 4 {
        d = 0;
    }
    if count < 3 {
        c = 0;
    }
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

/// A 16-bit decoded component as the 8-bit component of the texture: its top eight bits, which is
/// what the specification hands the sRGB conversion and what the reference decoder writes for
/// 8-bit output.
fn top8(c16: u32) -> u8 {
    (c16 >> 8) as u8
}

type Rgba = [i32; 4];

fn clamp8(c: Rgba) -> Rgba {
    c.map(|v| v.clamp(0, 255))
}

fn blue_contract(c: Rgba) -> Rgba {
    [(c[0] + c[2]) >> 1, (c[1] + c[2]) >> 1, c[2], c[3]]
}

/// Move the top bit of `a` to `b` and make `a` a signed six bit offset.
fn bit_transfer_signed(a: &mut i32, b: &mut i32) {
    *b = (*b >> 1) | (*a & 0x80);
    *a = (*a >> 1) & 0x3F;
    if *a & 0x20 != 0 {
        *a -= 0x40;
    }
}

/// The two endpoints of a partition from its color endpoint mode and unquantized values, or
/// `None` for the modes of high dynamic range.
fn endpoints(mode: u32, v: &[i32]) -> Option<(Rgba, Rgba)> {
    let mut v = [
        v.first().copied().unwrap_or(0),
        v.get(1).copied().unwrap_or(0),
        v.get(2).copied().unwrap_or(0),
        v.get(3).copied().unwrap_or(0),
        v.get(4).copied().unwrap_or(0),
        v.get(5).copied().unwrap_or(0),
        v.get(6).copied().unwrap_or(0),
        v.get(7).copied().unwrap_or(0),
    ];
    Some(match mode {
        0 => ([v[0], v[0], v[0], 255], [v[1], v[1], v[1], 255]),
        1 => {
            let l0 = (v[0] >> 2) | (v[1] & 0xC0);
            let l1 = (l0 + (v[1] & 0x3F)).min(255);
            ([l0, l0, l0, 255], [l1, l1, l1, 255])
        }
        4 => ([v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]),
        5 => {
            let (mut v0, mut v1, mut v2, mut v3) = (v[0], v[1], v[2], v[3]);
            bit_transfer_signed(&mut v1, &mut v0);
            bit_transfer_signed(&mut v3, &mut v2);
            (
                clamp8([v0, v0, v0, v2]),
                clamp8([v0 + v1, v0 + v1, v0 + v1, v2 + v3]),
            )
        }
        6 => (
            [
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                255,
            ],
            [v[0], v[1], v[2], 255],
        ),
        8 | 12 => {
            let alpha = if mode == 12 { (v[6], v[7]) } else { (255, 255) };
            let (s0, s1) = (v[0] + v[2] + v[4], v[1] + v[3] + v[5]);
            if s1 >= s0 {
                ([v[0], v[2], v[4], alpha.0], [v[1], v[3], v[5], alpha.1])
            } else {
                (
                    blue_contract([v[1], v[3], v[5], alpha.1]),
                    blue_contract([v[0], v[2], v[4], alpha.0]),
                )
            }
        }
        9 | 13 => {
            let pairs = if mode == 13 { 4 } else { 3 };
            for i in 0..pairs {
                let (mut a, mut b) = (v[2 * i + 1], v[2 * i]);
                bit_transfer_signed(&mut a, &mut b);
                v[2 * i + 1] = a;
                v[2 * i] = b;
            }
            let alpha = if mode == 13 {
                (v[6], v[6] + v[7])
            } else {
                (255, 255)
            };
            let base = [v[0], v[2], v[4], alpha.0];
            let sum = [v[0] + v[1], v[2] + v[3], v[4] + v[5], alpha.1];
            if v[1] + v[3] + v[5] >= 0 {
                (clamp8(base), clamp8(sum))
            } else {
                (clamp8(blue_contract(sum)), clamp8(blue_contract(base)))
            }
        }
        10 => (
            [
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                v[4],
            ],
            [v[0], v[1], v[2], v[5]],
        ),
        _ => return None,
    })
}

/// The unquantized weights of the grid (both planes, interleaved when there are two) and their
/// bilinear infill to the texels of a block.
struct Weights {
    grid: [u8; 64],
    width: usize,
    height: usize,
    planes: usize,
}

impl Weights {
    /// The weight of `plane` at texel (x, y) of a block of `bw` by `bh` texels, 0 to 64.
    fn at(&self, plane: usize, x: usize, y: usize, bw: usize, bh: usize) -> u32 {
        let ds = (1024 + bw / 2) / (bw - 1);
        let dt = (1024 + bh / 2) / (bh - 1);
        let (cs, ct) = (ds * x, dt * y);
        let gs = (cs * (self.width - 1) + 32) >> 6;
        let gt = (ct * (self.height - 1) + 32) >> 6;
        let (js, fs) = (gs >> 4, (gs & 15) as u32);
        let (jt, ft) = (gt >> 4, (gt & 15) as u32);
        let at = |sx: usize, sy: usize| -> u32 {
            let sx = sx.min(self.width - 1);
            let sy = sy.min(self.height - 1);
            u32::from(self.grid[(sy * self.width + sx) * self.planes + plane])
        };
        let (p00, p01, p10, p11) = (
            at(js, jt),
            at(js + 1, jt),
            at(js, jt + 1),
            at(js + 1, jt + 1),
        );
        let w11 = (fs * ft + 8) >> 4;
        let w10 = ft - w11;
        let w01 = fs - w11;
        let w00 = 16 + w11 - fs - ft;
        (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4
    }
}

/// Why a block is not decoded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Illegal {
    /// A reserved void-extent encoding, or one of high dynamic range.
    VoidExtent,
    /// A reserved block mode.
    BlockMode,
    /// A weight grid larger than the block.
    GridTooLarge,
    /// More than 64 weights, or fewer than 24 or more than 96 bits of them.
    WeightBits,
    /// Two planes of weights with four partitions.
    DualPlaneFourPartitions,
    /// More than 18 endpoint values, or too few bits left for them.
    EndpointBits,
}

/// What a legal block is made of (for looking at a block that decodes unexpectedly).
#[derive(Clone, Debug, Default)]
pub struct BlockInfo {
    pub void_extent: bool,
    pub grid: (usize, usize),
    pub dual_plane: bool,
    pub weight_levels: u16,
    pub partitions: usize,
    pub seed: u32,
    pub modes: [u32; 4],
    pub endpoint_levels: u16,
    pub plane2_channel: usize,
    /// How many partitions have endpoints of high dynamic range (and decode to the error color).
    pub hdr_partitions: usize,
}

/// What a block is made of, or why it is illegal.
pub fn inspect(block: &[u8; 16], bw: usize, bh: usize) -> Result<BlockInfo, Illegal> {
    let mut scratch = vec![[0u8; 4]; bw * bh];
    decode(u128::from_le_bytes(*block), bw, bh, false, &mut scratch)
}

/// Decode one block to `bw` x `bh` texels (row by row) in `out`. The color of an illegal block, or
/// of one that is for high dynamic range, is [`ERROR_COLOR`].
pub fn decode_block(block: &[u8; 16], bw: usize, bh: usize, srgb: bool, out: &mut [[u8; 4]]) {
    let out = &mut out[..bw * bh];
    if decode(u128::from_le_bytes(*block), bw, bh, srgb, out).is_err() {
        out.fill(ERROR_COLOR);
    }
}

fn decode(
    v: u128,
    bw: usize,
    bh: usize,
    srgb: bool,
    out: &mut [[u8; 4]],
) -> Result<BlockInfo, Illegal> {
    if get(v, 0, 9) == 0x1FC {
        // A void-extent block: one color for the whole block.
        if get(v, 9, 1) != 0 || get(v, 10, 2) != 3 {
            return Err(Illegal::VoidExtent);
        }
        let coords = [
            get(v, 12, 13),
            get(v, 25, 13),
            get(v, 38, 13),
            get(v, 51, 13),
        ];
        if coords != [0x1FFF; 4] && (coords[0] >= coords[1] || coords[2] >= coords[3]) {
            return Err(Illegal::VoidExtent);
        }
        let color = [
            top8(get(v, 64, 16)),
            top8(get(v, 80, 16)),
            top8(get(v, 96, 16)),
            top8(get(v, 112, 16)),
        ];
        out.fill(color);
        return Ok(BlockInfo {
            void_extent: true,
            ..BlockInfo::default()
        });
    }

    let (gw, gh, dual, wrange) = block_mode(get(v, 0, 11)).ok_or(Illegal::BlockMode)?;
    if gw > bw || gh > bh {
        return Err(Illegal::GridTooLarge);
    }
    let planes = 1 + usize::from(dual);
    let wcount = gw * gh * planes;
    let wbits = sequence_bits(wcount, wrange);
    if wcount > 64 || !(24..=96).contains(&wbits) {
        return Err(Illegal::WeightBits);
    }
    let parts = get(v, 11, 2) as usize + 1;
    if dual && parts == 4 {
        return Err(Illegal::DualPlaneFourPartitions);
    }

    // The color endpoint modes, and where the endpoint values start and how many bits they have.
    let mut modes = [0u32; 4];
    let extra_cem_bits;
    let start;
    if parts == 1 {
        modes[0] = get(v, 13, 4);
        extra_cem_bits = 0;
        start = 17;
    } else {
        start = 29;
        let high = 3 * parts - 4;
        let low = get(v, 13 + 10, 6);
        let class_field = low & 3;
        if class_field == 0 {
            for m in modes.iter_mut().take(parts) {
                *m = (low >> 2) & 0xF;
            }
            extra_cem_bits = 0;
        } else {
            let all = low | (get(v, 128 - wbits - high, high) << 6);
            let base = class_field - 1;
            for (i, m) in modes.iter_mut().enumerate().take(parts) {
                *m = (((all >> (2 + i)) & 1) + base) << 2;
            }
            for (i, m) in modes.iter_mut().enumerate().take(parts) {
                *m |= (all >> (2 + parts + 2 * i)) & 3;
            }
            extra_cem_bits = high;
        }
    }
    let ccs_bits = if dual { 2 } else { 0 };
    let used = start + wbits + extra_cem_bits + ccs_bits;
    let value_count: usize = modes
        .iter()
        .take(parts)
        .map(|m| 2 * ((*m as usize >> 2) + 1))
        .sum();
    if value_count > 18 || used > 128 {
        return Err(Illegal::EndpointBits);
    }
    let available = 128 - used;
    let erange = (FIRST_ENDPOINT_RANGE..=LAST_RANGE)
        .rev()
        .find(|r| sequence_bits(value_count, *r) <= available)
        .ok_or(Illegal::EndpointBits)?;
    let ccs = if dual {
        get(v, 128 - wbits - extra_cem_bits - 2, 2) as usize
    } else {
        0
    };

    // The endpoints of each partition.
    let mut stored = [0u8; 18];
    decode_sequence(v, start, value_count, erange, &mut stored);
    let mut ends = [([0i32; 4], [0i32; 4]); 4];
    // A partition of high dynamic range endpoints is the error color wherever it is used.
    let mut hdr = [false; 4];
    let mut at = 0;
    for (p, e) in ends.iter_mut().enumerate().take(parts) {
        let n = 2 * ((modes[p] as usize >> 2) + 1);
        let mut values = [0i32; 8];
        for (i, value) in values.iter_mut().enumerate().take(n) {
            *value = unquantize_endpoint(erange, u32::from(stored[at + i])) as i32;
        }
        at += n;
        match endpoints(modes[p], &values[..n]) {
            Some(pair) => *e = pair,
            None => hdr[p] = true,
        }
    }

    // The weights, stored from the end of the block backwards.
    let mut raw = [0u8; 64];
    decode_sequence(v.reverse_bits(), 0, wcount, wrange, &mut raw);
    let mut weights = Weights {
        grid: [0; 64],
        width: gw,
        height: gh,
        planes,
    };
    for (slot, stored) in weights.grid.iter_mut().zip(&raw).take(wcount) {
        *slot = unquantize_weight(wrange, u32::from(*stored)) as u8;
    }

    let seed = get(v, 13, 10);
    let small = bw * bh < 31;
    // The 16-bit value of an 8-bit endpoint component: a linear color repeats its byte, an sRGB
    // one has half a step below it.
    let widen = |c: i32| -> i32 {
        if srgb {
            (c << 8) | 0x80
        } else {
            (c << 8) | c
        }
    };
    for y in 0..bh {
        for x in 0..bw {
            let p = if parts == 1 {
                0
            } else {
                partition_of(seed, x as u32, y as u32, parts, small)
            };
            let w0 = weights.at(0, x, y, bw, bh) as i32;
            let w1 = if dual {
                weights.at(1, x, y, bw, bh) as i32
            } else {
                w0
            };
            if hdr[p] {
                out[y * bw + x] = ERROR_COLOR;
                continue;
            }
            let (e0, e1) = ends[p];
            let mut texel = [0u8; 4];
            for (c, t) in texel.iter_mut().enumerate() {
                let w = if dual && c == ccs { w1 } else { w0 };
                let c16 = (widen(e0[c]) * (64 - w) + widen(e1[c]) * w + 32) >> 6;
                *t = top8(c16 as u32);
            }
            out[y * bw + x] = texel;
        }
    }
    Ok(BlockInfo {
        void_extent: false,
        grid: (gw, gh),
        dual_plane: dual,
        weight_levels: RANGES[wrange].0,
        partitions: parts,
        seed: if parts > 1 { seed } else { 0 },
        modes,
        endpoint_levels: RANGES[erange].0,
        plane2_channel: ccs,
        hdr_partitions: hdr.iter().filter(|h| **h).count(),
    })
}

/// Decode a `width` x `height` image of ASTC blocks of `bw` x `bh` texels (rows of blocks one
/// after another, 16 bytes a block) to RGBA8 texels, rows tightly packed.
pub fn decode_image(
    blocks: &[u8],
    width: u32,
    height: u32,
    bw: u32,
    bh: u32,
    srgb: bool,
) -> Vec<u8> {
    let (width, height, bw, bh) = (width as usize, height as usize, bw as usize, bh as usize);
    let across = width.div_ceil(bw);
    let down = height.div_ceil(bh);
    let mut image = vec![0u8; width * height * 4];
    let rows_of_blocks = |first: usize, rows: &mut [u8]| {
        // `rows` holds the texel rows of the block rows from `first`.
        let mut texels = vec![[0u8; 4]; bw * bh];
        let block_rows = rows.len().div_ceil(width * 4 * bh).max(1);
        for br in 0..block_rows {
            let by = first + br;
            if by >= down {
                break;
            }
            for bx in 0..across {
                let at = (by * across + bx) * 16;
                let mut block = [0u8; 16];
                if at + 16 <= blocks.len() {
                    block.copy_from_slice(&blocks[at..at + 16]);
                }
                decode_block(&block, bw, bh, srgb, &mut texels);
                for ty in 0..bh {
                    let y = br * bh + ty;
                    if by * bh + ty >= height || (y + 1) * width * 4 > rows.len() {
                        break;
                    }
                    for tx in 0..bw {
                        let x = bx * bw + tx;
                        if x >= width {
                            break;
                        }
                        rows[(y * width + x) * 4..][..4].copy_from_slice(&texels[ty * bw + tx]);
                    }
                }
            }
        }
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(down)
        .min(16);
    if threads <= 1 || across * down < 2048 {
        rows_of_blocks(0, &mut image);
    } else {
        // Whole rows of blocks to each thread.
        let per = down.div_ceil(threads);
        std::thread::scope(|s| {
            for (i, chunk) in image.chunks_mut(per * bh * width * 4).enumerate() {
                let work = &rows_of_blocks;
                s.spawn(move || work(i * per, chunk));
            }
        });
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block being written field by field.
    #[derive(Default)]
    struct Block(u128);

    impl Block {
        fn put(&mut self, lo: usize, len: usize, value: u32) {
            let mask = ((1u128 << len) - 1) << lo;
            self.0 = (self.0 & !mask) | ((u128::from(value) << lo) & mask);
        }

        fn bytes(&self) -> [u8; 16] {
            self.0.to_le_bytes()
        }
    }

    fn decode_all(block: &Block, bw: usize, bh: usize, srgb: bool) -> Vec<[u8; 4]> {
        let mut out = vec![[0u8; 4]; bw * bh];
        decode_block(&block.bytes(), bw, bh, srgb, &mut out);
        out
    }

    fn all_error(texels: &[[u8; 4]]) -> bool {
        texels.iter().all(|t| *t == ERROR_COLOR)
    }

    #[test]
    fn every_trit_and_quint_combination_can_be_stored() {
        let mut seen = std::collections::HashSet::new();
        for t in 0..256 {
            let trits = trits_of(t);
            assert!(trits.iter().all(|x| *x < 3), "T = {t:#x}");
            seen.insert(trits);
        }
        assert_eq!(seen.len(), 243, "3^5 combinations");
        let mut seen = std::collections::HashSet::new();
        for q in 0..128 {
            let quints = quints_of(q);
            assert!(quints.iter().all(|x| *x < 5), "Q = {q:#x}");
            seen.insert(quints);
        }
        assert_eq!(seen.len(), 125, "5^3 combinations");
    }

    #[test]
    fn unquantized_endpoints_run_evenly_from_0_to_255() {
        for (range, &(levels, _, _)) in RANGES.iter().enumerate().skip(FIRST_ENDPOINT_RANGE) {
            let levels = u32::from(levels);
            let mut values: Vec<u32> = (0..levels).map(|v| unquantize_endpoint(range, v)).collect();
            values.sort_unstable();
            assert_eq!(values[0], 0, "range {levels}");
            assert_eq!(*values.last().unwrap(), 255, "range {levels}");
            for (i, v) in values.iter().enumerate() {
                let ideal = (i as f64 * 255.0 / f64::from(levels - 1)).round();
                assert!(
                    (f64::from(*v) - ideal).abs() <= 1.0,
                    "range {levels}, level {i}: {v} against {ideal}"
                );
            }
            values.dedup();
            assert_eq!(
                values.len() as u32,
                levels,
                "range {levels} has distinct values"
            );
        }
    }

    #[test]
    fn unquantized_weights_run_evenly_from_0_to_64() {
        for (range, &(levels, _, _)) in RANGES.iter().enumerate().take(12) {
            let levels = u32::from(levels);
            let mut values: Vec<u32> = (0..levels).map(|v| unquantize_weight(range, v)).collect();
            values.sort_unstable();
            assert_eq!(values[0], 0, "range {levels}");
            assert_eq!(*values.last().unwrap(), 64, "range {levels}");
            for (i, v) in values.iter().enumerate() {
                let ideal = (i as f64 * 64.0 / f64::from(levels - 1)).round();
                assert!(
                    (f64::from(*v) - ideal).abs() <= 1.0,
                    "range {levels}, level {i}: {v} against {ideal}"
                );
            }
            values.dedup();
            assert_eq!(
                values.len() as u32,
                levels,
                "range {levels} has distinct values"
            );
        }
    }

    #[test]
    fn a_void_extent_block_is_one_color() {
        let mut b = Block::default();
        b.put(0, 9, 0x1FC);
        b.put(10, 2, 3);
        // No extent: all ones.
        for lo in [12, 25, 38, 51] {
            b.put(lo, 13, 0x1FFF);
        }
        for (i, c) in [0x1234u32, 0x8000, 0xFFFF, 0x00FF].iter().enumerate() {
            b.put(64 + 16 * i, 16, *c);
        }
        let out = decode_all(&b, 6, 6, false);
        assert!(out.iter().all(|t| *t == [0x12, 0x80, 0xFF, 0x00]));
        // High dynamic range, or reserved bits that are not ones, is an error.
        b.put(9, 1, 1);
        assert!(all_error(&decode_all(&b, 4, 4, false)));
        b.put(9, 1, 0);
        b.put(10, 2, 1);
        assert!(all_error(&decode_all(&b, 4, 4, false)));
        // An extent that is empty (low not below high) is an error; one that is not is fine.
        b.put(10, 2, 3);
        b.put(12, 13, 5);
        b.put(25, 13, 5);
        assert!(all_error(&decode_all(&b, 4, 4, false)));
        b.put(25, 13, 6);
        b.put(38, 13, 0);
        b.put(51, 13, 9);
        let out = decode_all(&b, 4, 4, false);
        assert!(out.iter().all(|t| *t == [0x12, 0x80, 0xFF, 0x00]));
    }

    /// A 4x4 block with a 4x4 grid of two-bit weights (`weights`, each 0 to 3) and one partition
    /// of mode 12 (RGBA direct) with the endpoints `e0` and `e1` stored as 8 bits.
    fn weighted_block(weights: [u8; 16], e0: [u8; 4], e1: [u8; 4]) -> Block {
        let mut b = Block::default();
        // Block mode: 4x4 grid (A = 2, B = 0), range 0..3 (R2R1 = 10, R0 = 0, H = 0), one plane.
        b.put(0, 11, (2 << 5) | 2);
        b.put(11, 2, 0); // one partition
        b.put(13, 4, 12); // RGBA direct
        let values = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2], e0[3], e1[3]];
        for (i, v) in values.iter().enumerate() {
            b.put(17 + 8 * i, 8, u32::from(*v));
        }
        // The weights run backwards from the end of the block, low bit first.
        for (i, w) in weights.iter().enumerate() {
            b.put(127 - 2 * i, 1, u32::from(w & 1));
            b.put(126 - 2 * i, 1, u32::from(w >> 1 & 1));
        }
        b
    }

    /// As [`weighted_block`] with only the weights 0 (the first endpoint) and 3 (the second): bit
    /// i of `weights` picks the endpoint of texel i.
    fn simple_block(weights: u16, e0: [u8; 4], e1: [u8; 4]) -> Block {
        let w: [u8; 16] = std::array::from_fn(|i| if weights >> i & 1 != 0 { 3 } else { 0 });
        weighted_block(w, e0, e1)
    }

    #[test]
    fn a_direct_block_mixes_its_endpoints_by_the_weights() {
        let weights = 0b1010_0110_0000_1111u16;
        let b = simple_block(weights, [10, 20, 30, 40], [200, 210, 220, 230]);
        let out = decode_all(&b, 4, 4, false);
        for (i, t) in out.iter().enumerate() {
            let want = if weights >> i & 1 != 0 {
                [200, 210, 220, 230]
            } else {
                [10, 20, 30, 40]
            };
            assert_eq!(*t, want, "texel {i}");
        }
    }

    #[test]
    fn weights_between_the_ends_mix_in_sixteen_bits() {
        // Two-bit weights unquantize to 0, 21, 43 and 64 sixty-fourths.
        let weights = [0, 1, 2, 3, 3, 2, 1, 0, 1, 1, 2, 2, 0, 3, 0, 3];
        let (e0, e1) = ([10u8, 0, 255, 100], [250u8, 255, 0, 3]);
        let b = weighted_block(weights, e0, e1);
        let out = decode_all(&b, 4, 4, false);
        for (i, t) in out.iter().enumerate() {
            let w = [0, 21, 43, 64][weights[i] as usize];
            for c in 0..4 {
                let c16 =
                    (u32::from(e0[c]) * 257 * (64 - w) + u32::from(e1[c]) * 257 * w + 32) >> 6;
                assert_eq!(t[c], (c16 >> 8) as u8, "texel {i}, channel {c}");
            }
        }
    }

    #[test]
    fn endpoints_in_the_wrong_order_are_blue_contracted() {
        // The sum of the second endpoint is below the first: both are swapped and contracted.
        let b = simple_block(0, [100, 110, 120, 7], [10, 20, 30, 9]);
        let out = decode_all(&b, 4, 4, false);
        // e0 = blue_contract(e1) = ((10 + 30) / 2, (20 + 30) / 2, 30, 9).
        assert_eq!(out[0], [20, 25, 30, 9]);
        let b = simple_block(0xFFFF, [100, 110, 120, 7], [10, 20, 30, 9]);
        // e1 = blue_contract(e0) = ((100 + 120) / 2, (110 + 120) / 2, 120, 7).
        assert_eq!(decode_all(&b, 4, 4, false)[0], [110, 115, 120, 7]);
    }

    #[test]
    fn an_srgb_block_keeps_its_stored_values() {
        let b = simple_block(0, [255, 0, 1, 3], [255, 0, 1, 3]);
        assert_eq!(decode_all(&b, 4, 4, true)[0], [255, 0, 1, 3]);
        // Weight 1 (all ones) gives the second endpoint.
        let b = simple_block(0xFFFF, [0, 0, 0, 0], [255, 17, 129, 255]);
        assert_eq!(decode_all(&b, 4, 4, true)[0], [255, 17, 129, 255]);
    }

    #[test]
    fn reserved_and_impossible_encodings_give_the_error_color() {
        // Block mode with bits [3:0] clear is reserved.
        let mut b = simple_block(0, [0; 4], [0; 4]);
        b.put(0, 11, 0);
        assert!(all_error(&decode_all(&b, 4, 4, false)));
        // A grid wider than the block.
        let mut b = simple_block(0, [0; 4], [0; 4]);
        b.put(0, 11, (2 << 5) | 2 | (1 << 7)); // N = 5 in a block of 4
        assert!(all_error(&decode_all(&b, 4, 4, false)));
        // A high dynamic range endpoint mode (7).
        let mut b = simple_block(0, [0; 4], [0; 4]);
        b.put(13, 4, 7);
        assert!(all_error(&decode_all(&b, 4, 4, false)));
        // Dual plane with four partitions.
        let mut b = simple_block(0, [0; 4], [0; 4]);
        b.put(0, 11, (2 << 5) | 2 | (1 << 10));
        b.put(11, 2, 3);
        assert!(all_error(&decode_all(&b, 4, 4, false)));
    }

    #[test]
    fn the_footprints_follow_the_header_codes() {
        assert_eq!(footprint(0x40), Some((4, 4)));
        assert_eq!(footprint(0x46), Some((12, 12)));
        assert_eq!(footprint(0x54), Some((12, 10)));
        assert_eq!(footprint(0x24), None, "BC1 is not ASTC");
        let mut sizes: Vec<(u32, u32)> = (0x40..0x60).filter_map(footprint).collect();
        sizes.sort_unstable();
        sizes.dedup();
        assert_eq!(sizes.len(), 14, "fourteen footprints");
    }

    #[test]
    fn an_image_is_decoded_block_by_block_and_clipped_to_its_size() {
        // Two blocks across and one down of 4x4, an image of 6x3 texels.
        let left = simple_block(0, [1, 2, 3, 4], [5, 6, 7, 8]).bytes();
        let right = simple_block(0xFFFF, [1, 2, 3, 4], [50, 60, 70, 80]).bytes();
        let blocks: Vec<u8> = left.iter().chain(right.iter()).copied().collect();
        let image = decode_image(&blocks, 6, 3, 4, 4, false);
        assert_eq!(image.len(), 6 * 3 * 4);
        for y in 0..3 {
            for x in 0..6 {
                let want: [u8; 4] = if x < 4 {
                    [1, 2, 3, 4]
                } else {
                    [50, 60, 70, 80]
                };
                assert_eq!(&image[(y * 6 + x) * 4..][..4], want, "({x}, {y})");
            }
        }
    }

    #[test]
    fn decoding_in_threads_matches_decoding_in_one() {
        // Enough blocks to be split: a pattern of different blocks.
        let (across, down) = (70usize, 40usize);
        let mut blocks = Vec::new();
        for i in 0..across * down {
            let mut b = simple_block(
                (i * 2654435761) as u16,
                [i as u8, 2, 3, 4],
                [9, (i >> 3) as u8, 7, 8],
            );
            if i % 7 == 0 {
                b.put(0, 11, 0); // an error block
            }
            blocks.extend_from_slice(&b.bytes());
        }
        let (w, h) = ((across * 4) as u32, (down * 4) as u32);
        let threaded = decode_image(&blocks, w, h, 4, 4, false);
        // One block row at a time, in this thread.
        let mut single = Vec::new();
        for by in 0..down {
            let row = &blocks[by * across * 16..(by + 1) * across * 16];
            single.extend_from_slice(&decode_image(row, w, 4, 4, 4, false));
        }
        assert_eq!(threaded, single);
    }
}
