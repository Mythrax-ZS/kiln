//! What a SILK frame codes (RFC 6716 sections 4.2.7.3 to 4.2.7.8): the side information
//! (`silk/decode_indices.c`), the excitation pulses with the shell coder (`silk/decode_pulses.c`,
//! `silk/shell_coder.c`, `silk/code_signs.c`), and the stereo predictors
//! (`silk/stereo_decode_pred.c`).

use super::channel::{Channel, Conditioning, MAX_FRAME_LENGTH, TYPE_VOICED};
use super::fixed::{smlabb, smulwb};
use super::tables::*;
use crate::range::RangeDecoder;

const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
const SHELL_FRAME: usize = 16;
const MAX_PULSES: i32 = 16;
const N_RATE_LEVELS: usize = 10;

/// The side information of a frame (`silk_decode_indices`).
pub fn decode_indices(
    ch: &mut Channel,
    dec: &mut RangeDecoder,
    frame: usize,
    lbrr: bool,
    cond: Conditioning,
) {
    let ix = if lbrr || ch.vad_flags[frame] {
        dec.icdf(&TYPE_OFFSET_VAD_ICDF, 8) as i32 + 2
    } else {
        dec.icdf(&TYPE_OFFSET_NO_VAD_ICDF, 8) as i32
    };
    let idx = &mut ch.indices;
    idx.signal_type = (ix >> 1) as i8;
    idx.quant_offset_type = (ix & 1) as i8;

    // Gains: the first absolute (or a delta), the rest deltas.
    if cond == Conditioning::Conditional {
        idx.gains[0] = dec.icdf(&DELTA_GAIN_ICDF, 8) as i8;
    } else {
        let msb = dec.icdf(&GAIN_ICDF[idx.signal_type as usize], 8) as i32;
        idx.gains[0] = ((msb << 3) + dec.icdf(&UNIFORM8_ICDF, 8) as i32) as i8;
    }
    for g in idx.gains.iter_mut().take(ch.nb_subfr).skip(1) {
        *g = dec.icdf(&DELTA_GAIN_ICDF, 8) as i8;
    }

    // NLSFs: the first stage's vector, then the residuals.
    let cb = ch.nlsf_cb;
    let stage1 = &cb.cb1_icdf[(idx.signal_type as usize >> 1) * cb.n_vectors..];
    idx.nlsf[0] = dec.icdf(stage1, 8) as i8;
    let (ec_ix, _) = cb.unpack(idx.nlsf[0] as usize);
    for (i, &table) in ec_ix.iter().enumerate().take(cb.order) {
        let mut v = dec.icdf(&cb.ec_icdf[table as usize..], 8) as i32;
        if v == 0 {
            v -= dec.icdf(&NLSF_EXT_ICDF, 8) as i32;
        } else if v == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            v += dec.icdf(&NLSF_EXT_ICDF, 8) as i32;
        }
        idx.nlsf[i + 1] = (v - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }
    idx.nlsf_interp_q2 = if ch.nb_subfr == 4 {
        dec.icdf(&NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8
    } else {
        4
    };

    if i32::from(idx.signal_type) == TYPE_VOICED {
        // The pitch lag, as a delta from the last frame's where it can be.
        let mut absolute = true;
        if cond == Conditioning::Conditional && ch.ec_prev_signal_type == TYPE_VOICED {
            let delta = dec.icdf(&PITCH_DELTA_ICDF, 8) as i32;
            if delta > 0 {
                idx.lag = (i32::from(ch.ec_prev_lag_index) + delta - 9) as i16;
                absolute = false;
            }
        }
        if absolute {
            let high = dec.icdf(&PITCH_LAG_ICDF, 8) as i32 * (ch.fs_khz >> 1);
            idx.lag = (high + dec.icdf(ch.pitch_lag_low_bits_icdf, 8) as i32) as i16;
        }
        ch.ec_prev_lag_index = idx.lag;
        idx.contour = dec.icdf(ch.pitch_contour_icdf, 8) as i8;
        // The LTP filters.
        idx.per_index = dec.icdf(&LTP_PER_INDEX_ICDF, 8) as i8;
        let table: &[u8] = match idx.per_index {
            0 => &LTP_GAIN_ICDF_0,
            1 => &LTP_GAIN_ICDF_1,
            _ => &LTP_GAIN_ICDF_2,
        };
        for l in idx.ltp.iter_mut().take(ch.nb_subfr) {
            *l = dec.icdf(table, 8) as i8;
        }
        idx.ltp_scale_index = if cond == Conditioning::Independent {
            dec.icdf(&LTP_SCALE_ICDF, 8) as i8
        } else {
            0
        };
    }
    ch.ec_prev_signal_type = i32::from(idx.signal_type);
    idx.seed = dec.icdf(&UNIFORM4_ICDF, 8) as i8;
}

/// Split `p` pulses between two halves (`decode_split`).
fn split(dec: &mut RangeDecoder, p: i32, table: &[u8]) -> (i32, i32) {
    if p > 0 {
        let a = dec.icdf(
            &table[usize::from(SHELL_CODE_TABLE_OFFSETS[p as usize])..],
            8,
        ) as i32;
        (a, p - a)
    } else {
        (0, 0)
    }
}

/// The pulses of one block of 16 (`silk_shell_decoder`): halves, quarters, eighths, pairs.
fn shell_decode(out: &mut [i32], dec: &mut RangeDecoder, pulses4: i32) {
    let (p3a, p3b) = split(dec, pulses4, &SHELL_CODE_TABLE3);
    let mut at = 0;
    for p3 in [p3a, p3b] {
        let (p2a, p2b) = split(dec, p3, &SHELL_CODE_TABLE2);
        for p2 in [p2a, p2b] {
            let (p1a, p1b) = split(dec, p2, &SHELL_CODE_TABLE1);
            for p1 in [p1a, p1b] {
                let (a, b) = split(dec, p1, &SHELL_CODE_TABLE0);
                out[at] = a;
                out[at + 1] = b;
                at += 2;
            }
        }
    }
}

/// The excitation pulses of a frame (`silk_decode_pulses`, section 4.2.7.8).
pub fn decode_pulses(
    dec: &mut RangeDecoder,
    pulses: &mut [i32; MAX_FRAME_LENGTH],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: usize,
) {
    let rate_level = dec.icdf(&RATE_LEVELS_ICDF[(signal_type >> 1) as usize], 8);
    // A 10 ms frame at 12 kHz has a partial last block.
    let blocks = frame_length.div_ceil(SHELL_FRAME);
    let mut sum = [0i32; MAX_FRAME_LENGTH / SHELL_FRAME + 1];
    let mut lshifts = [0i32; MAX_FRAME_LENGTH / SHELL_FRAME + 1];
    for i in 0..blocks {
        sum[i] = dec.icdf(&PULSES_PER_BLOCK_ICDF[rate_level], 8) as i32;
        // "17" says more least significant bits follow.
        while sum[i] == MAX_PULSES + 1 {
            lshifts[i] += 1;
            let table = &PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1][usize::from(lshifts[i] == 10)..];
            sum[i] = dec.icdf(table, 8) as i32;
        }
    }
    for i in 0..blocks {
        let block = &mut pulses[i * SHELL_FRAME..(i + 1) * SHELL_FRAME];
        if sum[i] > 0 {
            shell_decode(block, dec, sum[i]);
        } else {
            block.fill(0);
        }
    }
    for i in 0..blocks {
        if lshifts[i] > 0 {
            for p in pulses[i * SHELL_FRAME..(i + 1) * SHELL_FRAME].iter_mut() {
                let mut q = *p;
                for _ in 0..lshifts[i] {
                    q = (q << 1) + dec.icdf(&LSB_ICDF, 8) as i32;
                }
                *p = q;
            }
            // Mark the block as having pulses for the signs.
            sum[i] |= lshifts[i] << 5;
        }
    }
    // The signs (`silk_decode_signs`).
    let table = &SIGN_ICDF[7 * (quant_offset_type + (signal_type << 1)) as usize..];
    let sign_blocks = (frame_length + SHELL_FRAME / 2) >> 4;
    for i in 0..sign_blocks {
        let p = sum[i];
        if p > 0 {
            let icdf = [table[(p & 0x1F).min(6) as usize], 0];
            for q in pulses[i * SHELL_FRAME..(i + 1) * SHELL_FRAME].iter_mut() {
                if *q > 0 {
                    *q *= (dec.icdf(&icdf, 8) as i32) * 2 - 1;
                }
            }
        }
    }
}

/// The two stereo prediction weights of a frame, in Q13 (`silk_stereo_decode_pred`).
pub fn decode_stereo_pred(dec: &mut RangeDecoder) -> [i32; 2] {
    let n = dec.icdf(&STEREO_PRED_JOINT_ICDF, 8) as i32;
    let mut ix = [[0i32; 3]; 2];
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];
    for row in &mut ix {
        row[0] = dec.icdf(&UNIFORM3_ICDF, 8) as i32;
        row[1] = dec.icdf(&UNIFORM5_ICDF, 8) as i32;
    }
    let mut pred = [0i32; 2];
    for (p, row) in pred.iter_mut().zip(&mut ix) {
        row[0] += 3 * row[2];
        let low = i32::from(STEREO_PRED_QUANT_Q13[row[0] as usize]);
        // 0.5 / 5 in Q16.
        let step = smulwb(
            i32::from(STEREO_PRED_QUANT_Q13[row[0] as usize + 1]) - low,
            6554,
        );
        *p = smlabb(low, step, 2 * row[1] + 1);
    }
    pred[0] -= pred[1];
    pred
}

/// Whether only the mid channel is coded (`silk_stereo_decode_mid_only`).
pub fn decode_mid_only(dec: &mut RangeDecoder) -> bool {
    dec.icdf(&STEREO_ONLY_CODE_MID_ICDF, 8) == 1
}
