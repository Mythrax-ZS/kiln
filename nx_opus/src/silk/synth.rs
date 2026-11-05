//! SILK's synthesis (RFC 6716 section 4.2.7.9, the reference's `silk/decode_core.c`): the
//! excitation from the pulses, the long-term (pitch) prediction, the short-term (LPC)
//! prediction and the gains.

use super::channel::{
    Channel, Control, LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_SUB_FRAME_LENGTH, TYPE_VOICED,
};
use super::fixed::*;
use super::tables::QUANTIZATION_OFFSETS_Q10;

/// The prediction error of `input` under the filter `b` (`silk_LPC_analysis_filter`); the
/// first `b.len()` outputs are zero.
pub fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16]) {
    let d = b.len();
    let len = out.len();
    for ix in d..len {
        let mut acc = smulbb(i32::from(input[ix - 1]), i32::from(b[0]));
        for j in 1..d {
            // Wrapping, so that two wraps can cancel (only invalid streams reach them).
            acc = smlabb(acc, i32::from(input[ix - 1 - j]), i32::from(b[j]));
        }
        acc = (i32::from(input[ix]) << 12).wrapping_sub(acc);
        out[ix] = sat16(rshift_round(acc, 12)) as i16;
    }
    out[..d].fill(0);
}

/// The short-term prediction of one sample in Q10 from the last `a.len()` states in Q14
/// (`hist` ends just before the sample).
pub fn lpc_predict(hist: &[i32], a: &[i16]) -> i32 {
    // Starting at half the order avoids a bias, as silk_SMLAWB rounds down.
    let mut pred = (a.len() >> 1) as i32;
    let n = hist.len();
    for (j, &c) in a.iter().enumerate() {
        pred = smlawb(pred, hist[n - 1 - j], i32::from(c));
    }
    pred
}

/// The long-term prediction in Q13 (or Q12) from five taps around `lag`.
fn ltp_predict(s: &[i32], center: usize, b: &[i16]) -> i32 {
    let mut pred = 2;
    for (j, &c) in b.iter().enumerate().take(LTP_ORDER) {
        pred = smlawb(pred, s[center - j], i32::from(c));
    }
    pred
}

/// Synthesize a frame (`silk_decode_core`).
pub fn decode_core(ch: &mut Channel, ctrl: &mut Control, xq: &mut [i16], pulses: &[i32]) {
    let offset_q10 = i32::from(
        QUANTIZATION_OFFSETS_Q10[(ch.indices.signal_type >> 1) as usize]
            [ch.indices.quant_offset_type as usize],
    );
    let nlsf_interpolation = ch.indices.nlsf_interp_q2 < 4;
    let order = ch.lpc_order;

    // The excitation.
    let mut seed = i32::from(ch.indices.seed);
    for (out, &p) in ch.exc_q14.iter_mut().zip(pulses).take(ch.frame_length) {
        seed = rand(seed);
        let mut e = p << 14;
        if e > 0 {
            e -= 80 << 4;
        } else if e < 0 {
            e += 80 << 4;
        }
        e += offset_q10 << 4;
        if seed < 0 {
            e = -e;
        }
        *out = e;
        seed = seed.wrapping_add(p);
    }

    let mut slpc = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
    slpc[..MAX_LPC_ORDER].copy_from_slice(&ch.slpc_q14_buf);
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
    let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH];
    let mut res = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut buf_idx = ch.ltp_mem_length;
    let sub = ch.subfr_length;
    let mut lag = 0usize;
    for k in 0..ch.nb_subfr {
        let a_q12: [i16; MAX_LPC_ORDER] = ctrl.pred_coef_q12[k >> 1];
        let mut b_q14 = [0i16; LTP_ORDER];
        b_q14.copy_from_slice(&ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER]);
        let mut signal_type = i32::from(ch.indices.signal_type);
        let gain_q10 = ctrl.gains_q16[k] >> 6;
        let mut inv_gain_q31 = inverse32_varq(ctrl.gains_q16[k], 47);
        // A change of gain rescales the filter states.
        let gain_adj_q16 = if ctrl.gains_q16[k] != ch.prev_gain_q16 {
            let adj = div32_varq(ch.prev_gain_q16, ctrl.gains_q16[k], 16);
            for s in &mut slpc[..MAX_LPC_ORDER] {
                *s = smulww(adj, *s);
            }
            adj
        } else {
            1 << 16
        };
        ch.prev_gain_q16 = ctrl.gains_q16[k];
        // No abrupt change from voiced concealment to unvoiced decoding.
        if ch.loss_cnt != 0
            && ch.prev_signal_type == TYPE_VOICED
            && signal_type != TYPE_VOICED
            && k < 2
        {
            b_q14 = [0; LTP_ORDER];
            b_q14[LTP_ORDER / 2] = 4096; // 0.25 in Q14
            ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER].copy_from_slice(&b_q14);
            signal_type = TYPE_VOICED;
            ctrl.pitch_l[k] = ch.lag_prev;
        }
        if signal_type == TYPE_VOICED {
            lag = ctrl.pitch_l[k] as usize;
            // Rewhiten the history with the new filter.
            if k == 0 || (k == 2 && nlsf_interpolation) {
                let start = ch.ltp_mem_length - lag - order - LTP_ORDER / 2;
                if k == 2 {
                    ch.out_buf[ch.ltp_mem_length..ch.ltp_mem_length + 2 * sub]
                        .copy_from_slice(&xq[..2 * sub]);
                }
                let input =
                    &ch.out_buf[start + k * sub..start + k * sub + ch.ltp_mem_length - start];
                lpc_analysis_filter(&mut s_ltp[start..ch.ltp_mem_length], input, &a_q12[..order]);
                // After rewhitening the history is unscaled.
                if k == 0 {
                    inv_gain_q31 = smulwb(inv_gain_q31, ctrl.ltp_scale_q14) << 2;
                }
                for i in 0..lag + LTP_ORDER / 2 {
                    s_ltp_q15[buf_idx - i - 1] =
                        smulwb(inv_gain_q31, i32::from(s_ltp[ch.ltp_mem_length - i - 1]));
                }
            } else if gain_adj_q16 != 1 << 16 {
                for i in 0..lag + LTP_ORDER / 2 {
                    let p = buf_idx - i - 1;
                    s_ltp_q15[p] = smulww(gain_adj_q16, s_ltp_q15[p]);
                }
            }
        }
        let exc = &ch.exc_q14[k * sub..(k + 1) * sub];
        let residual: &[i32] = if signal_type == TYPE_VOICED {
            for i in 0..sub {
                let pred = ltp_predict(&s_ltp_q15, buf_idx - lag + LTP_ORDER / 2, &b_q14);
                res[i] = exc[i].wrapping_add(pred << 1);
                s_ltp_q15[buf_idx] = res[i] << 1;
                buf_idx += 1;
            }
            &res[..sub]
        } else {
            exc
        };
        for i in 0..sub {
            let pred = lpc_predict(&slpc[i..MAX_LPC_ORDER + i], &a_q12[..order]);
            slpc[MAX_LPC_ORDER + i] = residual[i].wrapping_add(pred << 4);
            xq[k * sub + i] =
                sat16(rshift_round(smulww(slpc[MAX_LPC_ORDER + i], gain_q10), 8)) as i16;
        }
        slpc.copy_within(sub..sub + MAX_LPC_ORDER, 0);
    }
    ch.slpc_q14_buf.copy_from_slice(&slpc[..MAX_LPC_ORDER]);
}
