//! Prediction and reconstruction (VP9 specification sections 8.5, 8.6 and 8.7): intra
//! prediction, the block inter prediction filter, dequantization and the inverse transforms,
//! written as the specification states them (the butterflies of 8.7.1), for 8-bit samples.

use super::tables::{COS64_LOOKUP, SUBPEL_FILTERS};
use super::Plane;

const V_PRED: u8 = 1;
const H_PRED: u8 = 2;
const D45_PRED: u8 = 3;
const D135_PRED: u8 = 4;
const D117_PRED: u8 = 5;
const D153_PRED: u8 = 6;
const D207_PRED: u8 = 7;
const D63_PRED: u8 = 8;
const TM_PRED: u8 = 9;

#[inline]
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

#[inline]
fn round2_64(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

#[inline]
fn clip1(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

/// The intra prediction process (section 8.5.1) of one transform block at (`x`, `y`) of a plane;
/// `max_x` and `max_y` are the last sample of the plane's decoded area.
#[allow(clippy::too_many_arguments)]
pub fn predict_intra(
    p: &mut Plane,
    x: usize,
    y: usize,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    tx_size: u8,
    mode: u8,
    max_x: usize,
    max_y: usize,
) {
    let log2 = 2 + u32::from(tx_size);
    let size = 1usize << log2;
    let stride = p.stride;
    // above[0] is aboveRow[ -1 ], above[1 + i] is aboveRow[ i ].
    let mut above = [0i32; 65];
    let mut left = [0i32; 32];
    if have_above {
        let row = (y - 1) * stride;
        for i in 0..size {
            above[1 + i] = i32::from(p.data[row + max_x.min(x + i)]);
        }
        for i in size..2 * size {
            above[1 + i] = if not_on_right && tx_size == 0 {
                i32::from(p.data[row + max_x.min(x + i)])
            } else {
                above[size]
            };
        }
        above[0] = if have_left {
            i32::from(p.data[row + max_x.min(x - 1)])
        } else {
            129
        };
    } else {
        for v in above.iter_mut().take(2 * size + 1) {
            *v = 127;
        }
    }
    if have_left {
        for (i, v) in left.iter_mut().enumerate().take(size) {
            *v = i32::from(p.data[max_y.min(y + i) * stride + x - 1]);
        }
    } else {
        left[..size].iter_mut().for_each(|v| *v = 129);
    }
    let a = |i: isize| above[(i + 1) as usize];
    let mut pred = [[0u8; 32]; 32];
    match mode {
        V_PRED => {
            for row in pred.iter_mut().take(size) {
                for (j, v) in row.iter_mut().enumerate().take(size) {
                    *v = a(j as isize) as u8;
                }
            }
        }
        H_PRED => {
            for (i, row) in pred.iter_mut().enumerate().take(size) {
                row[..size].iter_mut().for_each(|v| *v = left[i] as u8);
            }
        }
        D207_PRED => {
            let mut t = [[0i32; 32]; 32];
            t[size - 1][..size].fill(left[size - 1]);
            for i in 0..size - 1 {
                t[i][0] = round2(left[i] + left[i + 1], 1);
            }
            for i in 0..size.saturating_sub(2) {
                t[i][1] = round2(left[i] + 2 * left[i + 1] + left[i + 2], 2);
            }
            t[size - 2][1] = round2(left[size - 2] + 3 * left[size - 1], 2);
            for j in 2..size {
                for i in (0..size - 1).rev() {
                    t[i][j] = t[i + 1][j - 2];
                }
            }
            for i in 0..size {
                for j in 0..size {
                    pred[i][j] = t[i][j] as u8;
                }
            }
        }
        D45_PRED => {
            for (i, row) in pred.iter_mut().enumerate().take(size) {
                for (j, v) in row.iter_mut().enumerate().take(size) {
                    let k = (i + j) as isize;
                    *v = if i + j + 2 < size * 2 {
                        round2(a(k) + a(k + 1) * 2 + a(k + 2), 2)
                    } else {
                        a(2 * size as isize - 1)
                    } as u8;
                }
            }
        }
        D63_PRED => {
            for (i, row) in pred.iter_mut().enumerate().take(size) {
                for (j, v) in row.iter_mut().enumerate().take(size) {
                    let k = (i / 2 + j) as isize;
                    *v = if i & 1 == 1 {
                        round2(a(k) + a(k + 1) * 2 + a(k + 2), 2)
                    } else {
                        round2(a(k) + a(k + 1), 1)
                    } as u8;
                }
            }
        }
        D117_PRED => {
            let mut t = [[0i32; 32]; 32];
            for (j, v) in t[0][..size].iter_mut().enumerate() {
                *v = round2(a(j as isize - 1) + a(j as isize), 1);
            }
            t[1][0] = round2(left[0] + 2 * a(-1) + a(0), 2);
            for (j, v) in t[1][..size].iter_mut().enumerate().skip(1) {
                *v = round2(a(j as isize - 2) + 2 * a(j as isize - 1) + a(j as isize), 2);
            }
            t[2][0] = round2(a(-1) + 2 * left[0] + left[1], 2);
            for i in 3..size {
                t[i][0] = round2(left[i - 3] + 2 * left[i - 2] + left[i - 1], 2);
            }
            for i in 2..size {
                for j in 1..size {
                    t[i][j] = t[i - 2][j - 1];
                }
            }
            for i in 0..size {
                for j in 0..size {
                    pred[i][j] = t[i][j] as u8;
                }
            }
        }
        D135_PRED => {
            let mut t = [[0i32; 32]; 32];
            t[0][0] = round2(left[0] + 2 * a(-1) + a(0), 2);
            for (j, v) in t[0][..size].iter_mut().enumerate().skip(1) {
                *v = round2(a(j as isize - 2) + 2 * a(j as isize - 1) + a(j as isize), 2);
            }
            t[1][0] = round2(a(-1) + 2 * left[0] + left[1], 2);
            for i in 2..size {
                t[i][0] = round2(left[i - 2] + 2 * left[i - 1] + left[i], 2);
            }
            for i in 1..size {
                for j in 1..size {
                    t[i][j] = t[i - 1][j - 1];
                }
            }
            for i in 0..size {
                for j in 0..size {
                    pred[i][j] = t[i][j] as u8;
                }
            }
        }
        D153_PRED => {
            let mut t = [[0i32; 32]; 32];
            t[0][0] = round2(left[0] + a(-1), 1);
            for i in 1..size {
                t[i][0] = round2(left[i - 1] + left[i], 1);
            }
            t[0][1] = round2(left[0] + 2 * a(-1) + a(0), 2);
            t[1][1] = round2(a(-1) + 2 * left[0] + left[1], 2);
            for i in 2..size {
                t[i][1] = round2(left[i - 2] + 2 * left[i - 1] + left[i], 2);
            }
            for (j, v) in t[0][..size].iter_mut().enumerate().skip(2) {
                *v = round2(
                    a(j as isize - 3) + 2 * a(j as isize - 2) + a(j as isize - 1),
                    2,
                );
            }
            for i in 1..size {
                for j in 2..size {
                    t[i][j] = t[i - 1][j - 2];
                }
            }
            for i in 0..size {
                for j in 0..size {
                    pred[i][j] = t[i][j] as u8;
                }
            }
        }
        TM_PRED => {
            for (i, row) in pred.iter_mut().enumerate().take(size) {
                for (j, v) in row.iter_mut().enumerate().take(size) {
                    *v = clip1(a(j as isize) + left[i] - a(-1));
                }
            }
        }
        _ => {
            // DC_PRED
            let avg = if have_left && have_above {
                let sum: i32 =
                    left[..size].iter().sum::<i32>() + above[1..=size].iter().sum::<i32>();
                (sum + size as i32) >> (log2 + 1)
            } else if have_left {
                let sum: i32 = left[..size].iter().sum();
                (sum + (1 << (log2 - 1))) >> log2
            } else if have_above {
                let sum: i32 = above[1..=size].iter().sum();
                (sum + (1 << (log2 - 1))) >> log2
            } else {
                128
            };
            for row in pred.iter_mut().take(size) {
                row[..size].iter_mut().for_each(|v| *v = avg as u8);
            }
        }
    }
    for (i, row) in pred.iter().enumerate().take(size) {
        let at = (y + i) * stride + x;
        p.data[at..at + size].copy_from_slice(&row[..size]);
    }
}

/// The block inter prediction process (section 8.5.2.4): `w` by `h` samples of `reference` from
/// (`x`, `y`) in sixteenths of a sample, `x_step` and `y_step` apart, filtered with `filter`.
#[allow(clippy::too_many_arguments)]
pub fn block_inter_prediction(
    reference: &Plane,
    last_x: i32,
    last_y: i32,
    x: i32,
    y: i32,
    x_step: i32,
    y_step: i32,
    w: usize,
    h: usize,
    filter: u8,
    out: &mut [u8],
    temp: &mut Vec<u8>,
) {
    let filters = &SUBPEL_FILTERS[usize::from(filter)];
    let ih = ((((h as i32 - 1) * y_step + 15) >> 4) + 8) as usize;
    temp.resize(ih * w, 0);
    let stride = reference.stride;
    let data = &reference.data;
    // The first sample each tap of each column reads.
    let x0 = (x >> 4) - 3;
    let inside_x = x_step == 16 && x0 >= 0 && x0 + w as i32 + 7 <= last_x;
    for r in 0..ih {
        let ry = ((y >> 4) + r as i32 - 3).clamp(0, last_y) as usize;
        let row = &data[ry * stride..];
        let dst = &mut temp[r * w..(r + 1) * w];
        if inside_x {
            let f = &filters[(x & 15) as usize];
            if x & 15 == 0 {
                dst.copy_from_slice(&row[(x >> 4) as usize..(x >> 4) as usize + w]);
            } else {
                for (c, d) in dst.iter_mut().enumerate() {
                    let s0 = x0 as usize + c;
                    let mut s = 0i32;
                    for t in 0..8 {
                        s += i32::from(f[t]) * i32::from(row[s0 + t]);
                    }
                    *d = clip1(round2(s, 7));
                }
            }
        } else {
            for (c, d) in dst.iter_mut().enumerate() {
                let p = x + x_step * c as i32;
                let f = &filters[(p & 15) as usize];
                let mut s = 0i32;
                for (t, &ft) in f.iter().enumerate() {
                    let sx = ((p >> 4) + t as i32 - 3).clamp(0, last_x) as usize;
                    s += i32::from(ft) * i32::from(row[sx]);
                }
                *d = clip1(round2(s, 7));
            }
        }
    }
    for r in 0..h {
        let p = (y & 15) + y_step * r as i32;
        let f = &filters[(p & 15) as usize];
        let base = (p >> 4) as usize;
        let dst = &mut out[r * w..(r + 1) * w];
        if p & 15 == 0 {
            dst.copy_from_slice(&temp[(base + 3) * w..(base + 3) * w + w]);
            continue;
        }
        for (c, d) in dst.iter_mut().enumerate() {
            let mut s = 0i32;
            for t in 0..8 {
                s += i32::from(f[t]) * i32::from(temp[(base + t) * w + c]);
            }
            *d = clip1(round2(s, 7));
        }
    }
}

// ---- Dequantization and the inverse transforms ----

const DCT_DCT: u8 = 0;
const ADST_DCT: u8 = 1;
const DCT_ADST: u8 = 2;

fn cos64(angle: i32) -> i64 {
    let a = angle & 127;
    i64::from(match a {
        0..=32 => COS64_LOOKUP[a as usize],
        33..=64 => -COS64_LOOKUP[(64 - a) as usize],
        65..=96 => -COS64_LOOKUP[(a - 64) as usize],
        _ => COS64_LOOKUP[(128 - a) as usize],
    })
}

fn sin64(angle: i32) -> i64 {
    cos64(angle - 32)
}

fn brev(bits: u32, x: usize) -> usize {
    let mut t = 0;
    for i in 0..bits {
        t |= ((x >> i) & 1) << (bits - 1 - i);
    }
    t
}

/// The array the 1D transforms work on (`T`) and its higher precision companion (`S`).
struct Transform {
    t: [i32; 32],
    s: [i64; 32],
}

impl Transform {
    /// `B( a, b, angle, flip )`.
    fn b(&mut self, a: usize, b: usize, angle: i32, flip: bool) {
        let (ta, tb) = (i64::from(self.t[a]), i64::from(self.t[b]));
        let x = ta * cos64(angle) - tb * sin64(angle);
        let y = ta * sin64(angle) + tb * cos64(angle);
        self.t[a] = round2_64(x, 14) as i32;
        self.t[b] = round2_64(y, 14) as i32;
        if flip {
            self.t.swap(a, b);
        }
    }

    /// `H( a, b, flip )`.
    fn h(&mut self, a: usize, b: usize, flip: bool) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        let (x, y) = (self.t[a], self.t[b]);
        self.t[a] = x.wrapping_add(y);
        self.t[b] = x.wrapping_sub(y);
    }

    /// `SB( a, b, angle, flip )`.
    fn sb(&mut self, a: usize, b: usize, angle: i32, flip: bool) {
        let (ta, tb) = (i64::from(self.t[a]), i64::from(self.t[b]));
        self.s[a] = ta * cos64(angle) - tb * sin64(angle);
        self.s[b] = ta * sin64(angle) + tb * cos64(angle);
        if flip {
            self.s.swap(a, b);
        }
    }

    /// `SH( a, b )`.
    fn sh(&mut self, a: usize, b: usize) {
        let (sa, sb) = (self.s[a], self.s[b]);
        self.t[a] = round2_64(sa + sb, 14) as i32;
        self.t[b] = round2_64(sa - sb, 14) as i32;
    }

    /// The inverse DCT array permutation (8.7.1.2).
    fn dct_permute(&mut self, n: u32) {
        let copy = self.t;
        for i in 0..1usize << n {
            self.t[i] = copy[brev(n, i)];
        }
    }

    /// The inverse DCT (8.7.1.3) of the permuted array.
    fn idct(&mut self, n: u32) {
        let n0 = 1usize << n;
        let n1 = 1usize << (n - 1);
        let n2 = 1usize << (n - 2);
        let ni = n as usize;
        if n == 2 {
            self.b(0, 1, 16, true);
        } else {
            self.idct(n - 1);
        }
        for i in 0..n2 {
            self.b(n1 + i, n0 - 1 - i, 32 - brev(5, n1 + i) as i32, false);
        }
        if n >= 3 {
            let n3 = 1usize << (n - 3);
            for i in 0..n3 {
                for j in 0..2 {
                    self.h(n1 + 4 * i + 2 * j, n1 + 1 + 4 * i + 2 * j, j == 1);
                }
            }
        }
        if n == 5 {
            for i in 0..2 {
                for j in 0..2 {
                    self.b(
                        n0 - ni + 3 - n2 * j - 4 * i,
                        n1 + ni - 4 + n2 * j + 4 * i,
                        28 - 16 * i as i32 + 56 * j as i32,
                        true,
                    );
                }
            }
            let n3 = 1usize << (n - 3);
            for i in 0..2 {
                for j in 0..4 {
                    self.h(n1 + n3 * j + i, n1 + n2 - 5 + n3 * j - i, j & 1 == 1);
                }
            }
        }
        if n >= 4 {
            for i in 0..=usize::from(n == 5) {
                for j in 0..2 {
                    self.b(
                        n0 - ni + 2 - i - n2 * j,
                        n1 + ni - 3 + i + n2 * j,
                        24 + 48 * j as i32,
                        true,
                    );
                }
            }
            for i in 0..2 * ni - 6 {
                for j in 0..2 {
                    self.h(n1 + n2 * j + i, n1 + n2 - 1 + n2 * j - i, j & 1 == 1);
                }
            }
        }
        if n >= 3 {
            let n3 = 1usize << (n - 3);
            for i in 0..n3 {
                self.b(n0 - n3 - 1 - i, n1 + n3 + i, 16, true);
            }
        }
        for i in 0..n1 {
            self.h(i, n0 - 1 - i, false);
        }
    }

    /// The ADST input array permutation (8.7.1.4).
    fn adst_in_permute(&mut self, n: u32) {
        let n0 = 1usize << n;
        let n1 = 1usize << (n - 1);
        let copy = self.t;
        for i in 0..n1 {
            self.t[2 * i] = copy[n0 - 1 - 2 * i];
            self.t[2 * i + 1] = copy[2 * i];
        }
    }

    /// The ADST output array permutation (8.7.1.5).
    fn adst_out_permute(&mut self, n: u32) {
        let copy = self.t;
        if n == 4 {
            for a in 0..2 {
                for b in 0..2 {
                    for c in 0..2 {
                        for d in 0..2 {
                            self.t[8 * a + 4 * b + 2 * c + d] =
                                copy[8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a];
                        }
                    }
                }
            }
        } else {
            for a in 0..2 {
                for b in 0..2 {
                    for c in 0..2 {
                        self.t[4 * a + 2 * b + c] = copy[4 * (c ^ b) + 2 * (b ^ a) + a];
                    }
                }
            }
        }
    }

    /// The inverse ADST4 (8.7.1.6).
    fn adst4(&mut self) {
        const SINPI_1_9: i64 = 5283;
        const SINPI_2_9: i64 = 9929;
        const SINPI_3_9: i64 = 13377;
        const SINPI_4_9: i64 = 15212;
        let t: [i64; 4] = std::array::from_fn(|i| i64::from(self.t[i]));
        let s0 = SINPI_1_9 * t[0];
        let s1 = SINPI_2_9 * t[0];
        let s2 = SINPI_3_9 * t[1];
        let s3 = SINPI_4_9 * t[2];
        let s4 = SINPI_1_9 * t[2];
        let s5 = SINPI_2_9 * t[3];
        let s6 = SINPI_4_9 * t[3];
        let v = t[0] - t[2] + t[3];
        let s7 = SINPI_3_9 * v;
        let x0 = s0 + s3 + s5;
        let x1 = s1 - s4 - s6;
        let x2 = s7;
        let x3 = s2;
        self.t[0] = round2_64(x0 + x3, 14) as i32;
        self.t[1] = round2_64(x1 + x3, 14) as i32;
        self.t[2] = round2_64(x2, 14) as i32;
        self.t[3] = round2_64(x0 + x1 - x3, 14) as i32;
    }

    /// The inverse ADST8 (8.7.1.7).
    fn adst8(&mut self) {
        self.adst_in_permute(3);
        for i in 0..4 {
            self.sb(2 * i, 1 + 2 * i, 30 - 8 * i as i32, true);
        }
        for i in 0..4 {
            self.sh(i, 4 + i);
        }
        for i in 0..2 {
            self.sb(4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
        }
        for i in 0..2 {
            self.sh(4 + i, 6 + i);
        }
        for i in 0..2 {
            self.h(i, 2 + i, false);
        }
        for i in 0..2 {
            self.b(2 + 4 * i, 3 + 4 * i, 16, true);
        }
        self.adst_out_permute(3);
        for i in 0..4 {
            self.t[1 + 2 * i] = -self.t[1 + 2 * i];
        }
    }

    /// The inverse ADST16 (8.7.1.8).
    fn adst16(&mut self) {
        self.adst_in_permute(4);
        for i in 0..8 {
            self.sb(2 * i, 1 + 2 * i, 31 - 4 * i as i32, true);
        }
        for i in 0..8 {
            self.sh(i, 8 + i);
        }
        for i in 0..4 {
            self.sb(8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
        }
        for i in 0..4 {
            self.sh(8 + i, 12 + i);
        }
        for i in 0..4 {
            self.h(i, 4 + i, false);
        }
        for i in 0..2 {
            for j in 0..2 {
                self.sb(4 + 8 * i + 3 * j, 5 + 8 * i + j, 24 - 16 * j as i32, true);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                self.sh(4 + 8 * j + i, 6 + 8 * j + i);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                self.h(8 * j + i, 2 + 8 * j + i, false);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                self.b(
                    2 + 4 * j + 8 * i,
                    3 + 4 * j + 8 * i,
                    48 + 64 * (i ^ j) as i32,
                    false,
                );
            }
        }
        self.adst_out_permute(4);
        for i in 0..2 {
            for j in 0..2 {
                self.t[1 + 12 * j + 2 * i] = -self.t[1 + 12 * j + 2 * i];
            }
        }
    }

    fn adst(&mut self, n: u32) {
        match n {
            2 => self.adst4(),
            3 => self.adst8(),
            _ => self.adst16(),
        }
    }

    /// The inverse Walsh-Hadamard transform (8.7.1.10).
    fn wht(&mut self, shift: u32) {
        let mut a = self.t[0] >> shift;
        let mut c = self.t[1] >> shift;
        let mut d = self.t[2] >> shift;
        let mut b = self.t[3] >> shift;
        a += c;
        d -= b;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= b;
        d += c;
        self.t[0] = a;
        self.t[1] = b;
        self.t[2] = c;
        self.t[3] = d;
    }
}

/// The reconstruction process (section 8.6.2): dequantize the coefficients of a transform block
/// (`coefs`, raster order, of which the first `eob` positions of the scan were read), transform
/// them back (section 8.7.2) and add them to the prediction in the plane.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct(
    p: &mut Plane,
    x: usize,
    y: usize,
    tx_size: u8,
    tx_type: u8,
    lossless: bool,
    coefs: &mut [i32],
    _eob: usize,
    dc_q: i32,
    ac_q: i32,
) {
    let n = 2 + u32::from(tx_size);
    let n0 = 1usize << n;
    let denom = if tx_size == 3 { 2 } else { 1 };
    let mut dq = [0i32; 1024];
    for i in 0..n0 * n0 {
        dq[i] = coefs[i] * ac_q / denom;
    }
    dq[0] = coefs[0] * dc_q / denom;
    let mut tr = Transform {
        t: [0; 32],
        s: [0; 32],
    };
    // Rows.
    for i in 0..n0 {
        let row = &mut dq[i * n0..(i + 1) * n0];
        if row.iter().all(|&v| v == 0) {
            continue;
        }
        tr.t[..n0].copy_from_slice(row);
        if lossless {
            tr.wht(2);
        } else if tx_type == DCT_DCT || tx_type == ADST_DCT {
            tr.dct_permute(n);
            tr.idct(n);
        } else {
            tr.adst(n);
        }
        row.copy_from_slice(&tr.t[..n0]);
    }
    // Columns, and the sum with the prediction.
    let shift = 6.min(n + 2);
    let stride = p.stride;
    for j in 0..n0 {
        for i in 0..n0 {
            tr.t[i] = dq[i * n0 + j];
        }
        if lossless {
            tr.wht(0);
        } else if tx_type == DCT_DCT || tx_type == DCT_ADST {
            tr.dct_permute(n);
            tr.idct(n);
        } else {
            tr.adst(n);
        }
        for (i, &tv) in tr.t.iter().enumerate().take(n0) {
            let v = if lossless { tv } else { round2(tv, shift) };
            let at = (y + i) * stride + x + j;
            p.data[at] = clip1(i32::from(p.data[at]) + v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DCT of known inputs: a DC coefficient alone gives a flat block, with the scale of the
    /// transform (cos(pi/4) per dimension, 1/8 for an 8x8 after the final shift).
    #[test]
    fn a_dc_coefficient_alone_gives_a_flat_block() {
        for tx in 0..4u8 {
            let n0 = 4usize << tx;
            let mut plane = Plane {
                data: vec![100; 64 * 64],
                stride: 64,
                width: 64,
                height: 64,
            };
            let mut coefs = vec![0i32; 1024];
            coefs[0] = 64;
            reconstruct(&mut plane, 0, 0, tx, DCT_DCT, false, &mut coefs, 1, 8, 8);
            let first = plane.data[0];
            for i in 0..n0 {
                for j in 0..n0 {
                    assert_eq!(plane.data[i * 64 + j], first, "tx {tx} at {i},{j}");
                }
            }
            assert!(first > 100, "a positive DC raises the block");
            // Outside the block nothing changed.
            assert_eq!(plane.data[n0], 100);
        }
    }

    #[test]
    fn the_lossless_transform_inverts_exactly() {
        // A WHT of a single coefficient: the 4x4 block it makes, by the specification's steps
        // done by hand for the first row: a = 4 >> 2 = 1, the rest 0; a += c (1), d -= b (0),
        // e = 0, b = 0, c = 0, a = 1, d = 0.
        let mut plane = Plane {
            data: vec![10; 16],
            stride: 4,
            width: 4,
            height: 4,
        };
        let mut coefs = vec![0i32; 1024];
        coefs[0] = 4;
        reconstruct(&mut plane, 0, 0, 0, DCT_DCT, true, &mut coefs, 1, 1, 1);
        let changed = plane.data.iter().filter(|&&v| v != 10).count();
        assert!(changed > 0);
    }

    #[test]
    fn a_whole_sample_motion_vector_copies() {
        let mut reference = Plane {
            data: (0..64 * 64).map(|i| (i % 251) as u8).collect(),
            stride: 64,
            width: 64,
            height: 64,
        };
        reference.data[0] = 7;
        let mut out = vec![0u8; 16];
        let mut temp = Vec::new();
        block_inter_prediction(
            &reference,
            63,
            63,
            16 * 10,
            16 * 20,
            16,
            16,
            4,
            4,
            0,
            &mut out,
            &mut temp,
        );
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(out[r * 4 + c], reference.data[(20 + r) * 64 + 10 + c]);
            }
        }
        // Off the frame the edge samples repeat.
        block_inter_prediction(
            &reference,
            63,
            63,
            -16 * 5,
            -16 * 5,
            16,
            16,
            4,
            4,
            0,
            &mut out,
            &mut temp,
        );
        assert!(out.iter().all(|&v| v == 7));
    }
}
