//! SILK's packet loss concealment and comfort noise (RFC 6716 section 4.4 leaves concealment
//! to the decoder; this follows the reference's `silk/PLC.c` and `silk/CNG.c`, whose state
//! every frame updates).

use super::channel::{
    Channel, Control, LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR,
    TYPE_NO_VOICE_ACTIVITY, TYPE_VOICED,
};
use super::fixed::*;
use super::params::{bwexpander, lpc_inverse_pred_gain, nlsf2a};
use super::synth::{lpc_analysis_filter, lpc_predict};

const HARM_ATT_Q15: [i32; 2] = [32440, 31130];
const RAND_ATTENUATE_V_Q15: [i32; 2] = [31130, 26214];
const RAND_ATTENUATE_UV_Q15: [i32; 2] = [32440, 29491];
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;
const RAND_BUF_SIZE: usize = 128;

/// The concealment's memory (`silk_PLC_struct`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Plc {
    pub pitch_l_q8: i32,
    pub ltp_coef_q14: [i16; LTP_ORDER],
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    pub last_frame_lost: bool,
    pub rand_seed: i32,
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: i32,
    pub nb_subfr: usize,
    pub subfr_length: usize,
}

impl Plc {
    /// `silk_PLC_Reset`.
    pub fn reset(&mut self, frame_length: usize) {
        self.pitch_l_q8 = (frame_length as i32) << 7;
        self.prev_gain_q16 = [1 << 16, 1 << 16];
        self.subfr_length = 20;
        self.nb_subfr = 2;
    }
}

/// The comfort noise generator's memory (`silk_CNG_struct`).
#[derive(Clone, Copy, Debug)]
pub struct Cng {
    pub exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub synth_state: [i32; MAX_LPC_ORDER],
    pub smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: i32,
}

impl Default for Cng {
    fn default() -> Cng {
        Cng {
            exc_buf_q14: [0; MAX_FRAME_LENGTH],
            smth_nlsf_q15: [0; MAX_LPC_ORDER],
            synth_state: [0; MAX_LPC_ORDER],
            smth_gain_q16: 0,
            rand_seed: 0,
            fs_khz: 0,
        }
    }
}

impl Cng {
    /// `silk_CNG_Reset`: evenly spaced NLSFs, no gain.
    pub fn reset(&mut self, lpc_order: usize) {
        let step = i32::from(i16::MAX) / (lpc_order as i32 + 1);
        let mut acc = 0;
        for v in self.smth_nlsf_q15.iter_mut().take(lpc_order) {
            acc += step;
            *v = acc as i16;
        }
        self.smth_gain_q16 = 0;
        self.rand_seed = 3_176_576;
    }
}

/// Remember what a good frame was like (`silk_PLC_update`).
pub fn plc_update(ch: &mut Channel, ctrl: &Control) {
    ch.plc_check_rate();
    let nb = ch.nb_subfr;
    ch.prev_signal_type = i32::from(ch.indices.signal_type);
    let plc = &mut ch.plc;
    let mut ltp_gain_q14 = 0i32;
    if i32::from(ch.indices.signal_type) == TYPE_VOICED {
        // The last subframe with a whole pitch period in it.
        let mut j = 0;
        while (j * ch.subfr_length) < ctrl.pitch_l[nb - 1] as usize {
            if j == nb {
                break;
            }
            let row = &ctrl.ltp_coef_q14[(nb - 1 - j) * LTP_ORDER..(nb - j) * LTP_ORDER];
            let g: i32 = row.iter().map(|&c| i32::from(c)).sum();
            if g > ltp_gain_q14 {
                ltp_gain_q14 = g;
                plc.ltp_coef_q14.copy_from_slice(row);
                plc.pitch_l_q8 = ctrl.pitch_l[nb - 1 - j] << 8;
            }
            j += 1;
        }
        plc.ltp_coef_q14 = [0; LTP_ORDER];
        plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;
        // Limit the pitch gain.
        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let scale_q10 = (V_PITCH_GAIN_START_MIN_Q14 << 10) / ltp_gain_q14.max(1);
            for c in &mut plc.ltp_coef_q14 {
                *c = (smulbb(i32::from(*c), scale_q10) >> 10) as i16;
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let scale_q14 = (V_PITCH_GAIN_START_MAX_Q14 << 14) / ltp_gain_q14.max(1);
            for c in &mut plc.ltp_coef_q14 {
                *c = (smulbb(i32::from(*c), scale_q14) >> 14) as i16;
            }
        }
    } else {
        plc.pitch_l_q8 = smulbb(ch.fs_khz, 18) << 8;
        plc.ltp_coef_q14 = [0; LTP_ORDER];
    }
    plc.prev_lpc_q12[..ch.lpc_order].copy_from_slice(&ctrl.pred_coef_q12[1][..ch.lpc_order]);
    plc.prev_ltp_scale_q14 = ctrl.ltp_scale_q14 as i16;
    plc.prev_gain_q16
        .copy_from_slice(&ctrl.gains_q16[nb - 2..nb]);
    plc.subfr_length = ch.subfr_length;
    plc.nb_subfr = nb;
}

/// Make up a lost frame (`silk_PLC_conceal`).
pub fn plc_conceal(ch: &mut Channel, ctrl: &mut Control, frame: &mut [i16]) {
    ch.plc_check_rate();
    let order = ch.lpc_order;
    let prev_gain_q10 = [ch.plc.prev_gain_q16[0] >> 6, ch.plc.prev_gain_q16[1] >> 6];
    if ch.first_frame_after_reset {
        ch.plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
    }
    // The quieter of the last two subframes' excitation becomes the noise.
    let sl = ch.plc.subfr_length;
    let nbp = ch.plc.nb_subfr;
    let mut exc_buf = [0i16; 2 * 80];
    for k in 0..2 {
        for i in 0..sl {
            let e = ch.exc_q14[i + (k + nbp - 2) * sl];
            exc_buf[k * sl + i] = sat16(smulww(e, prev_gain_q10[k]) >> 8) as i16;
        }
    }
    let (e1, s1) = sum_sqr_shift(&exc_buf[..sl]);
    let (e2, s2) = sum_sqr_shift(&exc_buf[sl..2 * sl]);
    let rand_start = if (e1 >> s2) < (e2 >> s1) {
        ((nbp as i32 - 1) * sl as i32 - RAND_BUF_SIZE as i32).max(0) as usize
    } else {
        (nbp as i32 * sl as i32 - RAND_BUF_SIZE as i32).max(0) as usize
    };
    let loss = (ch.loss_cnt as usize).min(1);
    let harm_gain_q15 = HARM_ATT_Q15[loss];
    let mut rand_gain_q15 = if ch.prev_signal_type == TYPE_VOICED {
        RAND_ATTENUATE_V_Q15[loss]
    } else {
        RAND_ATTENUATE_UV_Q15[loss]
    };
    // A bandwidth-expanded copy of the last filter.
    bwexpander(&mut ch.plc.prev_lpc_q12[..order], 64881); // 0.99 in Q16
    let a_q12 = ch.plc.prev_lpc_q12;
    let mut rand_scale_q14 = i32::from(ch.plc.rand_scale_q14);
    if ch.loss_cnt == 0 {
        rand_scale_q14 = 1 << 14;
        if ch.prev_signal_type == TYPE_VOICED {
            // Less noise in a voiced frame.
            for &b in &ch.plc.ltp_coef_q14 {
                rand_scale_q14 -= i32::from(b);
            }
            rand_scale_q14 = rand_scale_q14.max(3277);
            rand_scale_q14 =
                (smulbb(rand_scale_q14, i32::from(ch.plc.prev_ltp_scale_q14)) >> 14) as i16 as i32;
        } else {
            // Less noise for an unvoiced frame with a high prediction gain.
            let inv_gain_q30 = lpc_inverse_pred_gain(&ch.plc.prev_lpc_q12[..order]);
            let mut down = ((1 << 30) >> 3).min(inv_gain_q30);
            down = ((1 << 30) >> 8).max(down);
            down <<= 3;
            rand_gain_q15 = smulwb(down, rand_gain_q15) >> 14;
        }
    }
    let mut rand_seed = ch.plc.rand_seed;
    let mut lag = rshift_round(ch.plc.pitch_l_q8, 8) as usize;
    let mem = ch.ltp_mem_length;
    let mut buf_idx = mem;
    // Rewhiten the history and scale it.
    let idx = mem - lag - order - LTP_ORDER / 2;
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
    lpc_analysis_filter(&mut s_ltp[idx..mem], &ch.out_buf[idx..mem], &a_q12[..order]);
    let inv_gain_q30 = inverse32_varq(ch.plc.prev_gain_q16[1], 46).min(i32::MAX >> 1);
    let mut s_ltp_q14 = [0i32; 2 * MAX_FRAME_LENGTH];
    for i in idx + order..mem {
        s_ltp_q14[i] = smulwb(inv_gain_q30, i32::from(s_ltp[i]));
    }
    // LTP synthesis.
    let rand_buf: [i32; MAX_FRAME_LENGTH] = ch.exc_q14;
    for _ in 0..ch.nb_subfr {
        for _ in 0..ch.subfr_length {
            let center = buf_idx - lag + LTP_ORDER / 2;
            let mut pred = 2;
            for (j, &b) in ch.plc.ltp_coef_q14.iter().enumerate() {
                pred = smlawb(pred, s_ltp_q14[center - j], i32::from(b));
            }
            rand_seed = rand(rand_seed);
            let r = ((rand_seed >> 25) as usize) & (RAND_BUF_SIZE - 1);
            s_ltp_q14[buf_idx] = smlawb(pred, rand_buf[rand_start + r], rand_scale_q14) << 2;
            buf_idx += 1;
        }
        // Fade the pitch and the noise, and let the lag drift up.
        for b in &mut ch.plc.ltp_coef_q14 {
            *b = (smulbb(harm_gain_q15, i32::from(*b)) >> 15) as i16;
        }
        rand_scale_q14 = smulbb(rand_scale_q14, rand_gain_q15) >> 15;
        ch.plc.pitch_l_q8 = smlawb(ch.plc.pitch_l_q8, ch.plc.pitch_l_q8, 655);
        ch.plc.pitch_l_q8 = ch.plc.pitch_l_q8.min(smulbb(18, ch.fs_khz) << 8);
        lag = rshift_round(ch.plc.pitch_l_q8, 8) as usize;
    }
    // LPC synthesis.
    let base = mem - MAX_LPC_ORDER;
    s_ltp_q14[base..mem].copy_from_slice(&ch.slpc_q14_buf);
    for i in 0..ch.frame_length {
        let pred = lpc_predict(&s_ltp_q14[base + i..mem + i], &a_q12[..order]);
        let v = s_ltp_q14[mem + i].wrapping_add(pred << 4);
        s_ltp_q14[mem + i] = v;
        frame[i] = sat16(rshift_round(smulww(v, prev_gain_q10[1]), 8)) as i16;
    }
    ch.slpc_q14_buf.copy_from_slice(
        &s_ltp_q14[base + ch.frame_length..base + ch.frame_length + MAX_LPC_ORDER],
    );
    ch.plc.rand_seed = rand_seed;
    ch.plc.rand_scale_q14 = rand_scale_q14 as i16;
    ctrl.pitch_l = [lag as i32; MAX_NB_SUBFR];
}

/// Fade from a concealed frame into the next good one (`silk_PLC_glue_frames`).
pub fn plc_glue_frames(ch: &mut Channel, frame: &mut [i16]) {
    let plc = &mut ch.plc;
    if ch.loss_cnt != 0 {
        let (e, s) = sum_sqr_shift(frame);
        plc.conc_energy = e;
        plc.conc_energy_shift = s;
        plc.last_frame_lost = true;
        return;
    }
    if plc.last_frame_lost {
        let (mut energy, shift) = sum_sqr_shift(frame);
        if shift > plc.conc_energy_shift {
            plc.conc_energy >>= shift - plc.conc_energy_shift;
        } else if shift < plc.conc_energy_shift {
            energy >>= plc.conc_energy_shift - shift;
        }
        if energy > plc.conc_energy {
            let lz = clz32(plc.conc_energy) - 1;
            plc.conc_energy <<= lz;
            energy >>= (24 - lz).max(0);
            let frac_q24 = plc.conc_energy / energy.max(1);
            let mut gain_q16 = sqrt_approx(frac_q24) << 4;
            // Four times steeper, not to miss an onset after DTX.
            let slope_q16 = (((1 << 16) - gain_q16) / frame.len() as i32) << 2;
            for s in frame.iter_mut() {
                *s = smulwb(gain_q16, i32::from(*s)) as i16;
                gain_q16 += slope_q16;
                if gain_q16 > 1 << 16 {
                    break;
                }
            }
        }
    }
    plc.last_frame_lost = false;
}

/// Learn the background noise in silent frames and fill concealed ones with it (`silk_CNG`).
pub fn cng(ch: &mut Channel, ctrl: &Control, frame: &mut [i16]) {
    let order = ch.lpc_order;
    if ch.fs_khz != ch.cng.fs_khz {
        ch.cng.reset(order);
        ch.cng.fs_khz = ch.fs_khz;
    }
    let c = &mut ch.cng;
    if ch.loss_cnt == 0 && ch.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        for i in 0..order {
            let d = i32::from(ch.prev_nlsf_q15[i]) - i32::from(c.smth_nlsf_q15[i]);
            c.smth_nlsf_q15[i] = (i32::from(c.smth_nlsf_q15[i]) + smulwb(d, 16348)) as i16;
        }
        // The excitation of the loudest subframe.
        let (mut max_gain, mut subfr) = (0, 0);
        for i in 0..ch.nb_subfr {
            if ctrl.gains_q16[i] > max_gain {
                max_gain = ctrl.gains_q16[i];
                subfr = i;
            }
        }
        let sl = ch.subfr_length;
        c.exc_buf_q14.copy_within(0..(ch.nb_subfr - 1) * sl, sl);
        c.exc_buf_q14[..sl].copy_from_slice(&ch.exc_q14[subfr * sl..(subfr + 1) * sl]);
        for i in 0..ch.nb_subfr {
            c.smth_gain_q16 += smulwb(ctrl.gains_q16[i] - c.smth_gain_q16, 4634);
        }
    }
    if ch.loss_cnt != 0 {
        let length = frame.len();
        let mut sig = [0i32; MAX_FRAME_LENGTH + MAX_LPC_ORDER];
        // The excitation (`silk_CNG_exc`).
        let mut mask = 255usize;
        while mask > length {
            mask >>= 1;
        }
        let mut seed = c.rand_seed;
        for v in &mut sig[MAX_LPC_ORDER..MAX_LPC_ORDER + length] {
            seed = rand(seed);
            let idx = ((seed >> 24) as usize) & mask;
            *v = sat16(smulww(c.exc_buf_q14[idx], c.smth_gain_q16 >> 4));
        }
        c.rand_seed = seed;
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut a_q12[..order], &c.smth_nlsf_q15[..order]);
        sig[..MAX_LPC_ORDER].copy_from_slice(&c.synth_state);
        for i in 0..length {
            let sum_q6 = lpc_predict(&sig[i..MAX_LPC_ORDER + i], &a_q12[..order]);
            sig[MAX_LPC_ORDER + i] = sig[MAX_LPC_ORDER + i].wrapping_add(sum_q6 << 4);
            frame[i] = sat16(i32::from(frame[i]) + rshift_round(sum_q6, 6)) as i16;
        }
        c.synth_state
            .copy_from_slice(&sig[length..length + MAX_LPC_ORDER]);
    } else {
        c.synth_state[..order].fill(0);
    }
}
