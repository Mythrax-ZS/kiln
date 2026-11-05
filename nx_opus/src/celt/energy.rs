//! The band energies (RFC 6716 section 4.3.2): a coarse value per band in 6 dB steps, predicted
//! across time and frequency and coded with a Laplace-like model, then fine refinements in raw
//! bits. Energies are kept in base-2 log units (1.0 is 6 dB) with the band's mean removed.

use super::mode::MAX_FINE_BITS;
use super::tables::{BANDS, BETA_COEF, BETA_INTRA, E_PROB_MODEL, PRED_COEF, SMALL_ENERGY_ICDF};
use crate::range::RangeDecoder;

/// The smallest probability of any value of the Laplace model, in 1/32768.
const LAPLACE_MINP: u32 = 1;
/// How many values on each side keep at least that probability.
const LAPLACE_NMIN: u32 = 16;

/// The probability of a magnitude of one, from that of zero (`fs0`) and the decay (Q14).
fn laplace_freq1(fs0: u32, decay: u32) -> u32 {
    let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    (ft * (16384 - decay)) >> 15
}

/// One value of the Laplace-like model (section 4.3.2.1, `celt/laplace.c`): `fs` is the
/// probability of zero in 1/32768, `decay` how fast the rest falls off, in Q14.
pub fn laplace(dec: &mut RangeDecoder, mut fs: u32, decay: u32) -> i32 {
    let fm = dec.decode_bin(15);
    let mut val = 0i32;
    let mut fl = 0u32;
    if fm >= fs {
        val += 1;
        fl = fs;
        fs = laplace_freq1(fs, decay) + LAPLACE_MINP;
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = ((fs - 2 * LAPLACE_MINP) * decay) >> 15;
            fs += LAPLACE_MINP;
            val += 1;
        }
        // Past the decaying part every value has the smallest probability.
        if fs <= LAPLACE_MINP {
            let di = (fm - fl) >> 1;
            val += di as i32;
            fl += 2 * di * LAPLACE_MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    dec.update(fl, (fl + fs).min(32768), 32768);
    val
}

/// The coarse energies (section 4.3.2.1). `old` holds each channel's energies of the last frame
/// (`BANDS` per channel) and receives this frame's.
#[allow(clippy::too_many_arguments)]
pub fn decode_coarse(
    old: &mut [f32],
    start: usize,
    end: usize,
    intra: bool,
    dec: &mut RangeDecoder,
    channels: usize,
    lm: usize,
) {
    let model = &E_PROB_MODEL[lm][usize::from(intra)];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let mut prev = [0f32; 2];
    let budget = dec.len() as i32 * 8;
    for i in start..end {
        for c in 0..channels {
            let tell = dec.tell() as i32;
            let qi = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                laplace(
                    dec,
                    u32::from(model[pi]) << 7,
                    u32::from(model[pi + 1]) << 6,
                )
            } else if budget - tell >= 2 {
                let q = dec.icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (q >> 1) ^ -(q & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.bit_logp(1))
            } else {
                -1
            };
            let q = qi as f32;
            let e = &mut old[i + c * BANDS];
            *e = e.max(-9.0);
            *e = coef * *e + prev[c] + q;
            prev[c] = prev[c] + q - beta * q;
        }
    }
}

/// The fine energies (section 4.3.2.2): `fine[i]` raw bits per channel refine each band.
pub fn decode_fine(
    old: &mut [f32],
    start: usize,
    end: usize,
    fine: &[i32; BANDS],
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for i in start..end {
        if fine[i] <= 0 {
            continue;
        }
        for c in 0..channels {
            let q2 = dec.bits(fine[i] as u32);
            let offset = (q2 as f32 + 0.5) * (1 << (14 - fine[i])) as f32 / 16384.0 - 0.5;
            old[i + c * BANDS] += offset;
        }
    }
}

/// The leftover bits, one per band and channel, in two passes by priority (section 4.3.2.2).
#[allow(clippy::too_many_arguments)]
pub fn decode_final(
    old: &mut [f32],
    start: usize,
    end: usize,
    fine: &[i32; BANDS],
    priority: &[i32; BANDS],
    mut bits_left: i32,
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine[i] < MAX_FINE_BITS && priority[i] == prio {
                for c in 0..channels {
                    let q2 = dec.bits(1);
                    let offset = (q2 as f32 - 0.5) * (1 << (14 - fine[i] - 1)) as f32 / 16384.0;
                    old[i + c * BANDS] += offset;
                    bits_left -= 1;
                }
            }
            i += 1;
        }
    }
}

/// The linear amplitude of each band from its log energy (`log2Amp`), with the mean added back
/// and the cap of RFC 8251 section 8, which keeps an extreme stream from overflowing.
pub fn amplitudes(old: &[f32], start: usize, end: usize, channels: usize) -> [f32; 2 * BANDS] {
    let mut out = [0f32; 2 * BANDS];
    for c in 0..channels {
        for i in start..end {
            let lg = (old[i + c * BANDS] + super::tables::e_mean(i)).min(32.0);
            out[i + c * BANDS] = lg.exp2();
        }
    }
    out
}
