//! The bands' shapes (RFC 6716 section 4.3.4): each band is decoded as a PVQ vector, split in
//! halves (in time, frequency or stereo) when it has more bits than one codebook can use, and
//! filled by folding the lower spectrum or noise when it has none. The decisions are bit-exact
//! and follow the reference (`celt/bands.c`, `quant_band` and `quant_all_bands`), with the
//! folding fix of RFC 8251 section 9.

use super::mode::{mode, pulses_of, BITRES, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE};
use super::pvq::{decode_shape, renormalize};
use super::tables::{BANDS, EBANDS};
use crate::range::{ilog, RangeDecoder};

/// The generator of the folding noise and of the collapse filling (`celt_lcg_rand`).
pub fn lcg(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// Q15 multiply with rounding, on 16-bit values (`FRAC_MUL16`).
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + i32::from(a as i16) * i32::from(b as i16)) >> 15
}

/// The bit-exact cosine of the split angle (`bitexact_cos`): `x` is in 1/16384 of pi/2.
fn bitexact_cos(x: i32) -> i32 {
    let x2 = (4096 + x * x) >> 13;
    let x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    1 + x2
}

/// log2(sin / cos) of the split angle in Q11 (`bitexact_log2tan`).
fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ilog(icos as u32) as i32;
    let ls = ilog(isin as u32) as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// The resolution of the split angle (`compute_qn`).
fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let qb = (b - pulse_cap - (4 << BITRES))
        .min((b + n2 * offset) / n2)
        .min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

/// A Haar step between neighbours `stride` apart (`haar1`).
fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    let r = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..stride {
        for j in 0..n0 >> 1 {
            let a = r * x[stride * 2 * j + i];
            let b = r * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = a + b;
            x[stride * (2 * j + 1) + i] = a - b;
        }
    }
}

/// The order of the blocks after a Hadamard reordering (`ordery_table`), by block count.
fn hadamard_order(stride: usize) -> &'static [usize] {
    const ORDER: [usize; 30] = [
        1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2,
        10, 5,
    ];
    &ORDER[stride - 2..2 * stride - 2]
}

/// Gather the interleaved blocks into runs (`deinterleave_hadamard`).
fn deinterleave(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let mut tmp = vec![0f32; n0 * stride];
    for i in 0..stride {
        let row = if hadamard {
            hadamard_order(stride)[i]
        } else {
            i
        };
        for j in 0..n0 {
            tmp[row * n0 + j] = x[j * stride + i];
        }
    }
    x[..n0 * stride].copy_from_slice(&tmp);
}

/// Undo `deinterleave` (`interleave_hadamard`).
fn interleave(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let mut tmp = vec![0f32; n0 * stride];
    for i in 0..stride {
        let row = if hadamard {
            hadamard_order(stride)[i]
        } else {
            i
        };
        for j in 0..n0 {
            tmp[j * stride + i] = x[row * n0 + j];
        }
    }
    x[..n0 * stride].copy_from_slice(&tmp);
}

/// Turn a decoded mid and side back into left and right (`stereo_merge`).
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32) {
    let (mut xp, mut side) = (0f32, 0f32);
    for (a, b) in x.iter().zip(y.iter()) {
        xp += a * b;
        side += b * b;
    }
    xp *= mid;
    let el = mid * mid + side - 2.0 * xp;
    let er = mid * mid + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y.copy_from_slice(x);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for (a, b) in x.iter_mut().zip(y.iter_mut()) {
        let l = mid * *a;
        let r = *b;
        *a = lgain * (l - r);
        *b = rgain * (l + r);
    }
}

/// The integer square root, rounded down (`isqrt32`).
fn isqrt(v: u32) -> u32 {
    let mut r = (v as f64).sqrt() as u64;
    let v = u64::from(v);
    while r * r > v {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= v {
        r += 1;
    }
    r as u32
}

/// What every band of a frame shares.
struct Ctx<'a, 'b> {
    dec: &'a mut RangeDecoder<'b>,
    remaining_bits: i32,
    intensity: usize,
    spread: usize,
    seed: u32,
    /// Whether to apply the 180-degree phase of intensity stereo (RFC 8251 section 10).
    phase_inversion: bool,
}

/// The split angle of a band (section 4.3.4.4), in 1/16384 of pi/2, and for intensity stereo
/// whether the side is inverted.
fn decode_theta(ctx: &mut Ctx, n: usize, b: i32, qn: i32, b0: usize, stereo: bool) -> (i32, bool) {
    let mut itheta = 0;
    let mut inv = false;
    if qn != 1 {
        if stereo && n > 2 {
            // A step: probability 3 up to the middle, 1 after it.
            let p0 = 3;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            let fs = ctx.dec.decode(ft as u32) as i32;
            let x = if fs < (x0 + 1) * p0 {
                fs / p0
            } else {
                x0 + 1 + (fs - (x0 + 1) * p0)
            };
            let (fl, fh) = if x <= x0 {
                (p0 * x, p0 * (x + 1))
            } else {
                ((x - 1 - x0) + (x0 + 1) * p0, (x - x0) + (x0 + 1) * p0)
            };
            ctx.dec.update(fl as u32, fh as u32, ft as u32);
            itheta = x;
        } else if b0 > 1 || stereo {
            // Uniform.
            itheta = ctx.dec.uint(qn as u32 + 1) as i32;
        } else {
            // Triangular.
            let half = qn >> 1;
            let ft = (half + 1) * (half + 1);
            let fm = ctx.dec.decode(ft as u32) as i32;
            let (fl, fs);
            if fm < (half * (half + 1)) >> 1 {
                itheta = (isqrt(8 * fm as u32 + 1) as i32 - 1) >> 1;
                fs = itheta + 1;
                fl = (itheta * (itheta + 1)) >> 1;
            } else {
                itheta = (2 * (qn + 1) - isqrt(8 * (ft - fm - 1) as u32 + 1) as i32) >> 1;
                fs = qn + 1 - itheta;
                fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
            }
            ctx.dec.update(fl as u32, (fl + fs) as u32, ft as u32);
        }
        itheta = itheta * 16384 / qn;
    } else if stereo {
        if b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            inv = ctx.dec.bit_logp(2);
        }
        if !ctx.phase_inversion {
            inv = false;
        }
    }
    (itheta, inv)
}

/// Decode one band, or one part of a split band (`quant_band`); returns its collapse mask.
/// `lowband` is a private copy of the folding source, never aliasing the caller's buffers.
#[allow(clippy::too_many_arguments)]
fn quant_band(
    ctx: &mut Ctx,
    band: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    mut b: i32,
    mut blocks: usize,
    mut tf_change: i32,
    mut lowband: Option<&mut [f32]>,
    mut lm: i32,
    mut lowband_out: Option<&mut [f32]>,
    level: u32,
    gain: f32,
    mut fill: u32,
) -> u32 {
    let m = mode();
    let n0 = x.len();
    let mut n = n0;
    let mut n_b = n / blocks;
    let long_blocks = blocks == 1;
    let mut b0 = blocks;
    let stereo = y.is_some();
    let mut split = stereo;
    let mut time_divide = 0;
    let mut recombine = 0;
    let mut cm = 0u32;

    // One bin: just a sign per channel.
    if n == 1 {
        for ch in 0..1 + usize::from(stereo) {
            let mut sign = 0;
            if ctx.remaining_bits >= 1 << BITRES {
                sign = ctx.dec.bits(1);
                ctx.remaining_bits -= 1 << BITRES;
            }
            let v = if sign != 0 { -1.0 } else { 1.0 };
            if ch == 0 {
                x[0] = v;
            } else if let Some(y) = y.as_deref_mut() {
                y[0] = v;
            }
        }
        if let Some(out) = lowband_out {
            out[0] = x[0];
        }
        return 1;
    }

    if !stereo && level == 0 {
        if tf_change > 0 {
            recombine = tf_change;
        }
        // Recombine blocks to raise the frequency resolution.
        for k in 0..recombine {
            const BIT_INTERLEAVE: [u32; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
            if let Some(lb) = lowband.as_deref_mut() {
                haar1(lb, n >> k, 1 << k);
            }
            fill =
                BIT_INTERLEAVE[(fill & 0xF) as usize] | (BIT_INTERLEAVE[(fill >> 4) as usize] << 2);
        }
        blocks >>= recombine;
        n_b <<= recombine;
        // Divide in time to raise the time resolution.
        while n_b & 1 == 0 && tf_change < 0 {
            if let Some(lb) = lowband.as_deref_mut() {
                haar1(lb, n_b, blocks);
            }
            fill |= fill << blocks;
            blocks <<= 1;
            n_b >>= 1;
            time_divide += 1;
            tf_change += 1;
        }
        b0 = blocks;
        if b0 > 1 {
            if let Some(lb) = lowband.as_deref_mut() {
                deinterleave(lb, n_b >> recombine, b0 << recombine, long_blocks);
            }
        }
    }
    let n_b0 = n_b;

    // Split a band that has more bits than its largest codebook can use.
    if !stereo && lm != -1 && b > m.max_cost(band, lm) + 12 && n > 2 {
        n >>= 1;
        split = true;
        lm -= 1;
        if blocks == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        blocks = (blocks + 1) >> 1;
    }

    if split {
        let pulse_cap = m.log_n[band] + lm * (1 << BITRES);
        let offset = (pulse_cap >> 1)
            - if stereo && n == 2 {
                QTHETA_OFFSET_TWOPHASE
            } else {
                QTHETA_OFFSET
            };
        let mut qn = compute_qn(n as i32, b, offset, pulse_cap, stereo);
        if stereo && band >= ctx.intensity {
            qn = 1;
        }
        let tell = ctx.dec.tell_frac() as i32;
        let (itheta, inv) = decode_theta(ctx, n, b, qn, b0, stereo);
        let qalloc = ctx.dec.tell_frac() as i32 - tell;
        b -= qalloc;
        let orig_fill = fill;
        let (imid, iside, mut delta);
        if itheta == 0 {
            imid = 32767;
            iside = 0;
            fill &= (1 << blocks) - 1;
            delta = -16384;
        } else if itheta == 16384 {
            imid = 0;
            iside = 32767;
            fill &= ((1 << blocks) - 1) << blocks;
            delta = 16384;
        } else {
            imid = bitexact_cos(itheta);
            iside = bitexact_cos(16384 - itheta);
            // The split of the bits that minimizes the squared error.
            delta = frac_mul16(((n - 1) << 7) as i32, bitexact_log2tan(iside, imid));
        }
        let mid = imid as f32 / 32768.0;
        let side = iside as f32 / 32768.0;

        if n == 2 && stereo {
            // The side of a 2-bin stereo band is orthogonal to the mid: one sign bit.
            let y = y.as_deref_mut().expect("stereo");
            let sbits = if itheta != 0 && itheta != 16384 {
                1 << BITRES
            } else {
                0
            };
            let mbits = b - sbits;
            let swap = itheta > 8192;
            ctx.remaining_bits -= qalloc + sbits;
            let sign = if sbits != 0 {
                ctx.dec.bits(1) as i32
            } else {
                0
            };
            let sign = (1 - 2 * sign) as f32;
            {
                // The mid is coded in whichever channel dominates; the other follows it.
                let (x2, y2): (&mut [f32], &mut [f32]) = if swap {
                    (&mut *y, &mut *x)
                } else {
                    (&mut *x, &mut *y)
                };
                cm = quant_band(
                    ctx,
                    band,
                    x2,
                    None,
                    mbits,
                    blocks,
                    tf_change,
                    lowband,
                    lm,
                    lowband_out,
                    level,
                    gain,
                    orig_fill,
                );
                y2[0] = -sign * x2[1];
                y2[1] = sign * x2[0];
            }
            for j in 0..2 {
                let (l, r) = (mid * x[j], side * y[j]);
                x[j] = l - r;
                y[j] = l + r;
            }
            if inv {
                y.iter_mut().for_each(|v| *v = -*v);
            }
            return cm;
        }

        // The general split.
        if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                // Pre-echo masking.
                delta -= delta >> (4 - lm);
            } else {
                // Forward masking of 1.5 dB per 10 ms.
                delta = (delta + (((n as i32) << BITRES) >> (5 - lm))).min(0);
            }
        }
        let mut mbits = b.min((b - delta) / 2).max(0);
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;
        let next_level = if stereo { level } else { level + 1 };
        let mid_gain = if stereo { 1.0 } else { gain * mid };
        let side_gain = gain * side;
        let side_shift = if stereo { 0 } else { b0 >> 1 };
        {
            // Mono halves the band itself; stereo pairs it with the other channel.
            let (xs, ys): (&mut [f32], &mut [f32]) = if stereo {
                (&mut *x, y.as_deref_mut().expect("stereo"))
            } else {
                let (a, rest) = x.split_at_mut(n);
                (a, &mut rest[..n])
            };
            // Mono folds the second half from the second half of the source; stereo's side
            // does not fold.
            let (low_mid, low_side): (Option<&mut [f32]>, Option<&mut [f32]>) = match lowband {
                Some(lb) if !stereo => {
                    let (a, rest) = lb.split_at_mut(n);
                    (Some(a), Some(&mut rest[..n]))
                }
                other => (other, None),
            };
            // Only stereo hands the folding output on; mono writes it at the end.
            let out_mid = if stereo {
                lowband_out.as_deref_mut()
            } else {
                None
            };
            let rebalance_from = ctx.remaining_bits;
            if mbits >= sbits {
                cm = quant_band(
                    ctx, band, xs, None, mbits, blocks, tf_change, low_mid, lm, out_mid,
                    next_level, mid_gain, fill,
                );
                let rebalance = mbits - (rebalance_from - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 0 {
                    sbits += rebalance - (3 << BITRES);
                }
                let s = quant_band(
                    ctx,
                    band,
                    ys,
                    None,
                    sbits,
                    blocks,
                    tf_change,
                    low_side,
                    lm,
                    None,
                    next_level,
                    side_gain,
                    fill >> blocks,
                );
                cm |= s << side_shift;
            } else {
                let s = quant_band(
                    ctx,
                    band,
                    ys,
                    None,
                    sbits,
                    blocks,
                    tf_change,
                    low_side,
                    lm,
                    None,
                    next_level,
                    side_gain,
                    fill >> blocks,
                );
                cm = s << side_shift;
                let rebalance = sbits - (rebalance_from - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 16384 {
                    mbits += rebalance - (3 << BITRES);
                }
                cm |= quant_band(
                    ctx, band, xs, None, mbits, blocks, tf_change, low_mid, lm, out_mid,
                    next_level, mid_gain, fill,
                );
            }
        }
        if stereo {
            let y = y.expect("stereo");
            stereo_merge(x, y, mid);
            if inv {
                y.iter_mut().for_each(|v| *v = -*v);
            }
            return cm;
        }
    } else {
        // No split: one codebook.
        let mut q = m.bits_to_pulses(band, lm, b);
        let mut curr_bits = m.pulses_to_bits(band, lm, q);
        ctx.remaining_bits -= curr_bits;
        // Never spend more than is left.
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = m.pulses_to_bits(band, lm, q);
            ctx.remaining_bits -= curr_bits;
        }
        if q != 0 {
            let k = pulses_of(q) as usize;
            cm = decode_shape(&mut x[..n], k, ctx.spread, blocks, gain, ctx.dec);
        } else {
            // No pulses: fill the band anyway.
            let cm_mask = ((1u64 << blocks) - 1) as u32;
            fill &= cm_mask;
            if fill == 0 {
                x[..n].fill(0.0);
            } else {
                match lowband.as_deref() {
                    None => {
                        for v in x[..n].iter_mut() {
                            ctx.seed = lcg(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as f32;
                        }
                        cm = cm_mask;
                    }
                    Some(lb) => {
                        // The folded spectrum, with noise about 48 dB below it.
                        for (v, &l) in x[..n].iter_mut().zip(lb) {
                            ctx.seed = lcg(ctx.seed);
                            let tmp = if ctx.seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                            *v = l + tmp;
                        }
                        cm = fill;
                    }
                }
                renormalize(&mut x[..n], gain);
            }
        }
    }
    if level != 0 {
        return cm;
    }
    // Level 0 of a mono band: undo the reordering and the time-frequency changes, and keep
    // the result for folding.
    if b0 > 1 {
        interleave(x, n_b0 >> recombine, b0 << recombine, long_blocks);
    }
    let mut n_b = n_b0;
    let mut blocks = b0;
    for _ in 0..time_divide {
        blocks >>= 1;
        n_b <<= 1;
        cm |= cm >> blocks;
        haar1(x, n_b, blocks);
    }
    for k in 0..recombine {
        const BIT_DEINTERLEAVE: [u32; 16] = [
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
            0xFC, 0xFF,
        ];
        cm = BIT_DEINTERLEAVE[cm as usize];
        haar1(x, n0 >> k, 1 << k);
    }
    blocks <<= recombine;
    if let Some(out) = lowband_out {
        let scale = (n0 as f32).sqrt();
        for (o, &v) in out.iter_mut().zip(x.iter()) {
            *o = scale * v;
        }
    }
    cm & ((1u32 << blocks) - 1)
}

/// Everything `quant_all_bands` needs from the frame.
pub struct BandParams<'p> {
    pub start: usize,
    pub end: usize,
    pub lm: usize,
    pub channels: usize,
    pub short_blocks: bool,
    pub spread: usize,
    pub dual_stereo: bool,
    pub intensity: usize,
    pub tf_res: &'p [i32; BANDS],
    pub total_bits: i32,
    pub balance: i32,
    pub coded_bands: usize,
    pub pulses: &'p [i32; BANDS],
    pub phase_inversion: bool,
}

/// Decode the shapes of every band (`quant_all_bands`): `x` holds the channels' normalized
/// spectra, `n` bins each. Returns the collapse masks (`band * channels + channel`) and the
/// updated seed.
pub fn decode_bands(
    p: &BandParams,
    x: &mut [f32],
    n: usize,
    dec: &mut RangeDecoder,
    seed: u32,
) -> (Vec<u8>, u32) {
    let mm = 1usize << p.lm;
    let blocks = if p.short_blocks { mm } else { 1 };
    let c = p.channels;
    let norm_len = mm * EBANDS[BANDS];
    let mut norm = vec![0f32; c * norm_len];
    let mut masks = vec![0u8; c * BANDS];
    let mut ctx = Ctx {
        dec,
        remaining_bits: 0,
        intensity: p.intensity,
        spread: p.spread,
        seed,
        phase_inversion: p.phase_inversion,
    };
    let mut balance = p.balance;
    let mut dual_stereo = p.dual_stereo;
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let (xs, ys) = x.split_at_mut(n);
    let mut low_x: Vec<f32> = Vec::new();
    let mut low_y: Vec<f32> = Vec::new();
    for i in p.start..p.end {
        let lo = mm * EBANDS[i];
        let width = mm * EBANDS[i + 1] - lo;
        let tell = ctx.dec.tell_frac() as i32;
        if i != p.start {
            balance -= tell;
        }
        ctx.remaining_bits = p.total_bits - tell - 1;
        let b = if i < p.coded_bands {
            let curr_balance = balance / (p.coded_bands - i).min(3) as i32;
            (ctx.remaining_bits + 1)
                .min(p.pulses[i] + curr_balance)
                .clamp(0, 16383)
        } else {
            0
        };
        // RFC 8251 section 9: the second band may always fold from the first.
        if (lo as i32 - width as i32 >= (mm * EBANDS[p.start]) as i32 || i == p.start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if i == p.start + 1 {
            let n1 = mm * (EBANDS[p.start + 1] - EBANDS[p.start]);
            let n2 = mm * (EBANDS[p.start + 2] - EBANDS[p.start + 1]);
            let offset = mm * EBANDS[p.start];
            if n2 > n1 {
                for ch in 0..c {
                    let base = ch * norm_len + offset;
                    norm.copy_within(base + 2 * n1 - n2..base + n1, base + n1);
                }
            }
        }
        let tf_change = p.tf_res[i];
        // The collapse masks of the bands folded from.
        let mut effective_lowband: Option<usize> = None;
        let (x_cm, y_cm);
        if lowband_offset != 0 && (p.spread != 3 || blocks > 1 || tf_change < 0) {
            let eff =
                (mm * EBANDS[p.start]).max((mm * EBANDS[lowband_offset]).saturating_sub(width));
            effective_lowband = Some(eff);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if mm * EBANDS[fold_start] <= eff {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if !(fold_end < i && mm * EBANDS[fold_end] < eff + width) {
                    break;
                }
            }
            let (mut xm, mut ym) = (0u32, 0u32);
            for f in fold_start..fold_end {
                xm |= u32::from(masks[f * c]);
                ym |= u32::from(masks[f * c + c - 1]);
            }
            x_cm = xm;
            y_cm = ym;
        } else {
            x_cm = (1u32 << blocks) - 1;
            y_cm = x_cm;
        }
        if dual_stereo && i == p.intensity {
            // Intensity from here on: fold from the average of the channels.
            dual_stereo = false;
            for j in mm * EBANDS[p.start]..lo {
                norm[j] = 0.5 * (norm[j] + norm[norm_len + j]);
            }
        }
        // Private copies of the folding sources.
        if let Some(e) = effective_lowband {
            low_x.clear();
            low_x.extend_from_slice(&norm[e..e + width]);
            if dual_stereo {
                low_y.clear();
                low_y.extend_from_slice(&norm[norm_len + e..norm_len + e + width]);
            }
        }
        let has_low = effective_lowband.is_some();
        let (rx, ry);
        if dual_stereo {
            let (n_x, n_y) = norm.split_at_mut(norm_len);
            rx = quant_band(
                &mut ctx,
                i,
                &mut xs[lo..lo + width],
                None,
                b / 2,
                blocks,
                tf_change,
                has_low.then_some(low_x.as_mut_slice()),
                p.lm as i32,
                Some(&mut n_x[lo..lo + width]),
                0,
                1.0,
                x_cm,
            );
            ry = quant_band(
                &mut ctx,
                i,
                &mut ys[lo..lo + width],
                None,
                b / 2,
                blocks,
                tf_change,
                has_low.then_some(low_y.as_mut_slice()),
                p.lm as i32,
                Some(&mut n_y[lo..lo + width]),
                0,
                1.0,
                y_cm,
            );
        } else {
            let y = if c == 2 {
                Some(&mut ys[lo..lo + width])
            } else {
                None
            };
            rx = quant_band(
                &mut ctx,
                i,
                &mut xs[lo..lo + width],
                y,
                b,
                blocks,
                tf_change,
                has_low.then_some(low_x.as_mut_slice()),
                p.lm as i32,
                Some(&mut norm[lo..lo + width]),
                0,
                1.0,
                x_cm | y_cm,
            );
            ry = rx;
        }
        masks[i * c] = rx as u8;
        masks[i * c + c - 1] = ry as u8;
        balance += p.pulses[i] + tell;
        // Fold from here on only while there is at least one bit per bin.
        update_lowband = b > (width as i32) << BITRES;
    }
    (masks, ctx.seed)
}

/// Fill the blocks of a transient frame that received no pulse with noise (section 4.3.5).
#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    x: &mut [f32],
    n: usize,
    masks: &[u8],
    lm: usize,
    channels: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1: &[f32],
    prev2: &[f32],
    pulses: &[i32; BANDS],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = EBANDS[i + 1] - EBANDS[i];
        // The depth in eighths of a bit per bin.
        let depth = (1 + pulses[i]) / ((n0 << lm) as i32);
        let thresh = 0.5 * (-0.125 * depth as f32).exp2();
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();
        for c in 0..channels {
            let mut p1 = prev1[c * BANDS + i];
            let mut p2 = prev2[c * BANDS + i];
            if channels == 1 {
                p1 = p1.max(prev1[BANDS + i]);
                p2 = p2.max(prev2[BANDS + i]);
            }
            let ediff = (log_e[c * BANDS + i] - p1.min(p2)).max(0.0);
            let mut r = 2.0 * (-ediff).exp2();
            if lm == 3 {
                r *= std::f32::consts::SQRT_2;
            }
            r = thresh.min(r) * sqrt_1;
            let band = &mut x[c * n + (EBANDS[i] << lm)..][..n0 << lm];
            let mut renorm = false;
            for k in 0..1usize << lm {
                if masks[i * channels + c] & (1 << k) == 0 {
                    for j in 0..n0 {
                        seed = lcg(seed);
                        band[(j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renorm = true;
                }
            }
            if renorm {
                renormalize(band, 1.0);
            }
        }
    }
}

/// Scale the normalized bands by their amplitudes into the MDCT spectrum (`denormalise_bands`).
pub fn denormalize(
    x: &[f32],
    freq: &mut [f32],
    amp: &[f32],
    end: usize,
    n: usize,
    lm: usize,
    channels: usize,
) {
    for c in 0..channels {
        let f = &mut freq[c * n..(c + 1) * n];
        for i in 0..end {
            let g = amp[c * BANDS + i];
            for j in EBANDS[i] << lm..EBANDS[i + 1] << lm {
                f[j] = x[c * n + j] * g;
            }
        }
        f[EBANDS[end] << lm..].fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_cosine_spans_its_quadrant() {
        // Called strictly inside the quadrant: near 32768 at its start, near 0 at its end,
        // monotone between.
        assert!(bitexact_cos(1) >= 32766);
        assert!(bitexact_cos(16383) < 10);
        let mut prev = i32::MAX;
        for x in (1..16384).step_by(511) {
            let c = bitexact_cos(x);
            assert!(c <= prev);
            prev = c;
        }
        // log2(tan) is antisymmetric.
        let (a, b) = (bitexact_cos(4000), bitexact_cos(16384 - 4000));
        assert_eq!(bitexact_log2tan(a, b), -bitexact_log2tan(b, a));
    }

    #[test]
    fn interleaving_round_trips() {
        let orig: Vec<f32> = (0..24).map(|v| v as f32).collect();
        for &(n0, stride) in &[(3usize, 8usize), (6, 4), (12, 2)] {
            for hadamard in [false, true] {
                let mut x = orig.clone();
                deinterleave(&mut x, n0, stride, hadamard);
                interleave(&mut x, n0, stride, hadamard);
                assert_eq!(x, orig);
            }
        }
    }

    #[test]
    fn the_integer_square_root_rounds_down() {
        for v in [0u32, 1, 2, 3, 4, 8, 9, 15, 16, 17, 1 << 20, u32::MAX] {
            let r = u64::from(isqrt(v));
            assert!(
                r * r <= u64::from(v) && (r + 1) * (r + 1) > u64::from(v),
                "{v}"
            );
        }
    }
}
