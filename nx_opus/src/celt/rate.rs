//! The bit allocation (RFC 6716 section 4.3.3): how the bits of a frame are shared between the
//! bands' fine energy and their PVQ shapes. It is bit-exact and mirrors the encoder's choice, so
//! it follows the reference (`celt/rate.c`, `compute_allocation` and `interp_bits2pulses`)
//! step for step.

use super::mode::{mode, BITRES, FINE_OFFSET, MAX_FINE_BITS};
use super::tables::{BANDS, BAND_ALLOCATION, EBANDS, LOG2_FRAC};
use crate::range::RangeDecoder;

const ALLOC_STEPS: i32 = 6;

/// What the allocation decided.
#[derive(Clone, Debug, Default)]
pub struct Allocation {
    /// One past the last band that has PVQ bits; the rest only have fine energy.
    pub coded_bands: usize,
    /// The first band coded as intensity stereo.
    pub intensity: usize,
    pub dual_stereo: bool,
    /// Bits over the caps, handed to the bands' rebalancing, in eighths.
    pub balance: i32,
    /// The PVQ bits of each band, in eighths.
    pub pulses: [i32; BANDS],
    /// The fine energy bits of each band (per channel).
    pub fine_quant: [i32; BANDS],
    /// Whether each band gets one of the leftover bits in the first (0) or second (1) pass.
    pub fine_priority: [i32; BANDS],
}

/// Decode the allocation of a frame: `total` eighths of a bit are left for it, `offsets` are the
/// dynamic boosts, `caps` the caps of `init_caps`.
#[allow(clippy::too_many_arguments)]
pub fn compute_allocation(
    start: usize,
    end: usize,
    offsets: &[i32; BANDS],
    caps: &[i32; BANDS],
    trim: i32,
    total: i32,
    channels: usize,
    lm: usize,
    dec: &mut RangeDecoder,
) -> Allocation {
    let c = channels as i32;
    let lm_i = lm as i32;
    let mut total = total.max(0);
    let mut skip_start = start;
    // A bit to say where the skipped bands end.
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    let (mut intensity_rsv, mut dual_stereo_rsv) = (0, 0);
    if channels == 2 {
        intensity_rsv = LOG2_FRAC[end - start];
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }
    let width = |j: usize| (EBANDS[j + 1] - EBANDS[j]) as i32;
    let mut thresh = [0i32; BANDS];
    let mut trim_offset = [0i32; BANDS];
    for j in start..end {
        // Below this a band gets no PVQ bits at all.
        thresh[j] = (c << BITRES).max((3 * width(j)) << lm_i << BITRES >> 4);
        // The tilt of the allocation.
        trim_offset[j] = (c
            * width(j)
            * (trim - 5 - lm_i)
            * (end as i32 - j as i32 - 1)
            * (1 << (lm_i + BITRES)))
            >> 6;
        if width(j) << lm_i == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }
    let vector =
        |q: usize, j: usize| (c * width(j) * i32::from(BAND_ALLOCATION[q][j])) << lm_i >> 2;

    // The quality step: a bisection over the static allocation's rows.
    let (mut lo, mut hi) = (1i32, BAND_ALLOCATION.len() as i32 - 1);
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        let mut done = false;
        let mut psum = 0;
        for j in (start..end).rev() {
            let mut bits = vector(mid as usize, j);
            if bits > 0 {
                bits = (bits + trim_offset[j]).max(0);
            }
            bits += offsets[j];
            if bits >= thresh[j] || done {
                done = true;
                psum += bits.min(caps[j]);
            } else if bits >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    hi = lo;
    lo -= 1;
    let mut bits1 = [0i32; BANDS];
    let mut bits2 = [0i32; BANDS];
    for j in start..end {
        let mut b1 = vector(lo as usize, j);
        let mut b2 = if hi as usize >= BAND_ALLOCATION.len() {
            caps[j]
        } else {
            vector(hi as usize, j)
        };
        if b1 > 0 {
            b1 = (b1 + trim_offset[j]).max(0);
        }
        if b2 > 0 {
            b2 = (b2 + trim_offset[j]).max(0);
        }
        if lo > 0 {
            b1 += offsets[j];
        }
        b2 += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits1[j] = b1;
        bits2[j] = (b2 - b1).max(0);
    }
    interpolate(
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        caps,
        total,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
        channels,
        lm,
        dec,
    )
}

/// `interp_bits2pulses`: interpolate between the two rows, decode the skipped bands and the
/// stereo parameters, and split each band's bits between fine energy and PVQ.
#[allow(clippy::too_many_arguments)]
fn interpolate(
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32; BANDS],
    bits2: &[i32; BANDS],
    thresh: &[i32; BANDS],
    caps: &[i32; BANDS],
    mut total: i32,
    skip_rsv: i32,
    mut intensity_rsv: i32,
    mut dual_stereo_rsv: i32,
    channels: usize,
    lm: usize,
    dec: &mut RangeDecoder,
) -> Allocation {
    let m = mode();
    let c = channels as i32;
    let stereo = i32::from(channels > 1);
    let alloc_floor = c << BITRES;
    let log_m = (lm as i32) << BITRES;
    let width = |j: usize| (EBANDS[j + 1] - EBANDS[j]) as i32;
    let mut out = Allocation::default();

    let (mut lo, mut hi) = (0i32, 1i32 << ALLOC_STEPS);
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(caps[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let bits = &mut out.pulses;
    let mut psum = 0;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(caps[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Skip bands from the end while the decoder is told to.
    let mut coded = end;
    loop {
        let j = coded - 1;
        if j <= skip_start {
            // The reserved bit goes back to the pool.
            total += skip_rsv;
            break;
        }
        let span = (EBANDS[coded] - EBANDS[start]) as i32;
        let mut left = total - psum;
        let percoeff = left / span;
        left -= span * percoeff;
        let rem = (left - (EBANDS[j] - EBANDS[start]) as i32).max(0);
        let band_width = width(j);
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            if dec.bit_logp(1) {
                break;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC[j - start];
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded -= 1;
    }

    // The stereo parameters.
    out.intensity = if intensity_rsv > 0 {
        start + dec.uint((coded + 1 - start) as u32) as usize
    } else {
        0
    };
    if out.intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    out.dual_stereo = dual_stereo_rsv > 0 && dec.bit_logp(1);

    // Share what is left among the coded bands.
    let span = (EBANDS[coded] - EBANDS[start]) as i32;
    let mut left = total - psum;
    let percoeff = left / span;
    left -= span * percoeff;
    for (j, b) in bits.iter_mut().enumerate().take(coded).skip(start) {
        *b += percoeff * width(j);
    }
    for (j, b) in bits.iter_mut().enumerate().take(coded).skip(start) {
        let tmp = left.min(width(j));
        *b += tmp;
        left -= tmp;
    }

    let ebits = &mut out.fine_quant;
    let prio = &mut out.fine_priority;
    let mut balance = 0;
    let mut j = start;
    while j < coded {
        let n0 = width(j);
        let n = n0 << lm;
        bits[j] += balance;
        let mut excess;
        if n > 1 {
            excess = (bits[j] - caps[j]).max(0);
            bits[j] -= excess;
            // The extra degree of freedom of stereo.
            let den =
                c * n + i32::from(channels == 2 && n > 2 && !out.dual_stereo && j < out.intensity);
            let nclogn = den * (m.log_n[j] + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclogn >> 3;
            }
            ebits[j] = ((bits[j] + offset + (den << (BITRES - 1))) / (den << BITRES)).max(0);
            if c * ebits[j] > bits[j] >> BITRES {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            prio[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            // One bin: everything but a sign bit goes to fine energy.
            excess = (bits[j] - (c << BITRES)).max(0);
            bits[j] -= excess;
            ebits[j] = 0;
            prio[j] = 1;
        }
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            prio[j] = i32::from(extra_bits >= excess - balance);
            excess -= extra_bits;
        }
        balance = excess;
        j += 1;
    }
    out.balance = balance;
    // The skipped bands spend their bits on fine energy.
    while j < end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        prio[j] = i32::from(ebits[j] < 1);
        j += 1;
    }
    out.coded_bands = coded;
    out
}
