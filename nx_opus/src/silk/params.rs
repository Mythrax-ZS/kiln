//! A SILK frame's parameters from its indices (RFC 6716 sections 4.2.7.4 to 4.2.7.6): the
//! subframe gains (`silk/gain_quant.c`), the normalized line spectral frequencies and the LPC
//! filters made from them (`silk/NLSF_*.c`, `silk/NLSF2A.c`, `silk/LPC_inv_pred_gain.c`,
//! `silk/bwexpander*.c`), and the pitch lags and LTP filters (`silk/decode_pitch.c`,
//! `silk/decode_parameters.c`), with the corrections of RFC 8251 sections 6 and 7.

use super::channel::{Channel, Conditioning, Control, LTP_ORDER, MAX_LPC_ORDER, TYPE_VOICED};
use super::fixed::*;
use super::tables::*;

/// An NLSF codebook (`silk_NLSF_CB_struct`).
pub struct NlsfCodebook {
    pub n_vectors: usize,
    pub order: usize,
    pub quant_step_q16: i32,
    pub cb1_q8: &'static [u8],
    pub cb1_icdf: &'static [u8],
    pub pred_q8: &'static [u8],
    pub ec_sel: &'static [u8],
    pub ec_icdf: &'static [u8],
    pub delta_min_q15: &'static [i16],
}

/// The codebook of narrow and medium band (order 10, step 0.18).
pub static NLSF_CB_NB_MB: NlsfCodebook = NlsfCodebook {
    n_vectors: 32,
    order: 10,
    quant_step_q16: 11796,
    cb1_q8: &NLSF_CB1_NB_MB_Q8,
    cb1_icdf: &NLSF_CB1_ICDF_NB_MB,
    pred_q8: &NLSF_PRED_NB_MB_Q8,
    ec_sel: &NLSF_CB2_SELECT_NB_MB,
    ec_icdf: &NLSF_CB2_ICDF_NB_MB,
    delta_min_q15: &NLSF_DELTA_MIN_NB_MB_Q15,
};

/// The codebook of wide band (order 16, step 0.15).
pub static NLSF_CB_WB: NlsfCodebook = NlsfCodebook {
    n_vectors: 32,
    order: 16,
    quant_step_q16: 9830,
    cb1_q8: &NLSF_CB1_WB_Q8,
    cb1_icdf: &NLSF_CB1_ICDF_WB,
    pred_q8: &NLSF_PRED_WB_Q8,
    ec_sel: &NLSF_CB2_SELECT_WB,
    ec_icdf: &NLSF_CB2_ICDF_WB,
    delta_min_q15: &NLSF_DELTA_MIN_WB_Q15,
};

impl NlsfCodebook {
    /// The entropy tables and backward predictors of a first-stage vector (`silk_NLSF_unpack`).
    pub fn unpack(&self, cb1: usize) -> ([i16; MAX_LPC_ORDER], [u8; MAX_LPC_ORDER]) {
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred = [0u8; MAX_LPC_ORDER];
        let sel = &self.ec_sel[cb1 * self.order / 2..];
        for i in (0..self.order).step_by(2) {
            let entry = sel[i / 2];
            ec_ix[i] = (i32::from((entry >> 1) & 7) * 9) as i16;
            pred[i] = self.pred_q8[i + usize::from(entry & 1) * (self.order - 1)];
            ec_ix[i + 1] = (i32::from((entry >> 5) & 7) * 9) as i16;
            pred[i + 1] = self.pred_q8[i + usize::from((entry >> 4) & 1) * (self.order - 1) + 1];
        }
        (ec_ix, pred)
    }
}

/// The subframe gains in Q16 (`silk_gains_dequant`).
pub fn gains_dequant(gains: &mut [i32], ind: &[i8], prev_ind: &mut i8, conditional: bool) {
    const OFFSET: i32 = (2 * 128) / 6 + 16 * 128;
    const INV_SCALE_Q16: i32 = (65536 * (((88 - 2) * 128) / 6)) / (64 - 1);
    for (k, g) in gains.iter_mut().enumerate() {
        let mut prev = i32::from(*prev_ind);
        if k == 0 && !conditional {
            // At most 16 steps (about 21.8 dB) down.
            prev = i32::from(ind[k]).max(prev - 16);
        } else {
            let delta = i32::from(ind[k]) - 4;
            let threshold = 2 * 36 - 64 + prev;
            if delta > threshold {
                prev += (delta << 1) - threshold;
            } else {
                prev += delta;
            }
        }
        prev = prev.clamp(0, 63);
        *prev_ind = prev as i8;
        *g = log2lin((smulwb(INV_SCALE_Q16, prev) + OFFSET).min(3967));
    }
}

/// Keep the NLSFs ordered and apart (`silk_NLSF_stabilize`).
pub fn nlsf_stabilize(nlsf: &mut [i16], delta_min: &[i16]) {
    let l = nlsf.len();
    for _ in 0..20 {
        // The smallest distance.
        let mut min_diff = i32::from(nlsf[0]) - i32::from(delta_min[0]);
        let mut at = 0;
        for i in 1..l {
            let d = i32::from(nlsf[i]) - (i32::from(nlsf[i - 1]) + i32::from(delta_min[i]));
            if d < min_diff {
                min_diff = d;
                at = i;
            }
        }
        let d = (1 << 15) - (i32::from(nlsf[l - 1]) + i32::from(delta_min[l]));
        if d < min_diff {
            min_diff = d;
            at = l;
        }
        if min_diff >= 0 {
            return;
        }
        if at == 0 {
            nlsf[0] = delta_min[0];
        } else if at == l {
            nlsf[l - 1] = ((1 << 15) - i32::from(delta_min[l])) as i16;
        } else {
            let mut min_center: i32 = delta_min[..at].iter().map(|&d| i32::from(d)).sum();
            min_center += i32::from(delta_min[at]) >> 1;
            let mut max_center: i32 = 1 << 15;
            max_center -= delta_min[at + 1..=l]
                .iter()
                .map(|&d| i32::from(d))
                .sum::<i32>();
            max_center -= i32::from(delta_min[at]) >> 1;
            let center = limit(
                rshift_round(i32::from(nlsf[at - 1]) + i32::from(nlsf[at]), 1),
                min_center,
                max_center,
            ) as i16;
            nlsf[at - 1] = (i32::from(center) - (i32::from(delta_min[at]) >> 1)) as i16;
            nlsf[at] = (i32::from(nlsf[at - 1]) + i32::from(delta_min[at])) as i16;
        }
    }
    // The fallback after 20 tries: sort, then push apart.
    nlsf.sort_unstable();
    nlsf[0] = nlsf[0].max(delta_min[0]);
    for i in 1..l {
        // RFC 8251 section 7: saturate rather than wrap.
        let floor = sat16(i32::from(nlsf[i - 1]) + i32::from(delta_min[i]));
        nlsf[i] = (i32::from(nlsf[i]).max(floor)) as i16;
    }
    nlsf[l - 1] = (i32::from(nlsf[l - 1]).min((1 << 15) - i32::from(delta_min[l]))) as i16;
    for i in (0..l - 1).rev() {
        nlsf[i] =
            (i32::from(nlsf[i]).min(i32::from(nlsf[i + 1]) - i32::from(delta_min[i + 1]))) as i16;
    }
}

/// The Laroia weights of an NLSF vector (`silk_NLSF_VQ_weights_laroia`).
fn laroia_weights(nlsf: &[i16]) -> [i16; MAX_LPC_ORDER] {
    let d = nlsf.len();
    let inv = |x: i32| (1 << 17) / x.max(1);
    let mut w = [0i16; MAX_LPC_ORDER];
    let mut t1 = inv(i32::from(nlsf[0]));
    let mut t2 = inv(i32::from(nlsf[1]) - i32::from(nlsf[0]));
    w[0] = (t1 + t2).min(i32::from(i16::MAX)) as i16;
    let mut k = 1;
    while k < d - 1 {
        t1 = inv(i32::from(nlsf[k + 1]) - i32::from(nlsf[k]));
        w[k] = (t1 + t2).min(i32::from(i16::MAX)) as i16;
        t2 = inv(i32::from(nlsf[k + 2]) - i32::from(nlsf[k + 1]));
        w[k + 1] = (t1 + t2).min(i32::from(i16::MAX)) as i16;
        k += 2;
    }
    t1 = inv((1 << 15) - i32::from(nlsf[d - 1]));
    w[d - 1] = (t1 + t2).min(i32::from(i16::MAX)) as i16;
    w
}

/// Decode the NLSF vector of a frame (`silk_NLSF_decode`, section 4.2.7.5).
pub fn nlsf_decode(out: &mut [i16], indices: &[i8], cb: &NlsfCodebook) {
    let order = cb.order;
    let first = indices[0] as usize;
    for (o, &c) in out
        .iter_mut()
        .zip(&cb.cb1_q8[first * order..(first + 1) * order])
    {
        *o = (i32::from(c) << 7) as i16;
    }
    let (_, pred) = cb.unpack(first);
    // The residual, predicted backwards (`silk_NLSF_residual_dequant`).
    let mut res = [0i16; MAX_LPC_ORDER];
    let mut out_q10 = 0i32;
    for i in (0..order).rev() {
        let pred_q10 = smulbb(out_q10, i32::from(pred[i])) >> 8;
        out_q10 = i32::from(indices[i + 1]) << 10;
        if out_q10 > 0 {
            out_q10 -= 102;
        } else if out_q10 < 0 {
            out_q10 += 102;
        }
        out_q10 = smlawb(pred_q10, out_q10, cb.quant_step_q16);
        res[i] = out_q10 as i16;
    }
    let w = laroia_weights(&out[..order]);
    for i in 0..order {
        let w_q9 = sqrt_approx(i32::from(w[i]) << 16);
        let v = i32::from(out[i]) + (i32::from(res[i]) << 14) / w_q9;
        out[i] = v.clamp(0, 32767) as i16;
    }
    nlsf_stabilize(&mut out[..order], cb.delta_min_q15);
}

/// Bandwidth expansion of a 16-bit filter (`silk_bwexpander`).
pub fn bwexpander(ar: &mut [i16], mut chirp_q16: i32) {
    let minus_one = chirp_q16 - 65536;
    let d = ar.len();
    for a in ar.iter_mut().take(d - 1) {
        *a = rshift_round(chirp_q16 * i32::from(*a), 16) as i16;
        chirp_q16 += rshift_round(chirp_q16 * minus_one, 16);
    }
    ar[d - 1] = rshift_round(chirp_q16 * i32::from(ar[d - 1]), 16) as i16;
}

/// Bandwidth expansion of a 32-bit filter (`silk_bwexpander_32`).
fn bwexpander_32(ar: &mut [i32], mut chirp_q16: i32) {
    let minus_one = chirp_q16 - 65536;
    let d = ar.len();
    for a in ar.iter_mut().take(d - 1) {
        *a = smulww(chirp_q16, *a);
        chirp_q16 += rshift_round(chirp_q16 * minus_one, 16);
    }
    ar[d - 1] = smulww(chirp_q16, ar[d - 1]);
}

/// The inverse prediction gain of a filter in Q30, 0 if it is unstable
/// (`silk_LPC_inverse_pred_gain`, with RFC 8251 section 6).
pub fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
    const QA: u32 = 24;
    const A_LIMIT: i32 = 16_773_022; // 0.99975 in Q24
    let order = a_q12.len();
    let mut a = [[0i32; MAX_LPC_ORDER]; 2];
    let mut dc = 0i32;
    let cur = order & 1;
    for k in 0..order {
        dc += i32::from(a_q12[k]);
        a[cur][k] = i32::from(a_q12[k]) << (QA - 12);
    }
    if dc >= 4096 {
        return 0;
    }
    let mul_frac = |x: i32, y: i32, q: u32| rshift_round64(i64::from(x) * i64::from(y), q) as i32;
    let mut inv_gain: i32 = 1 << 30;
    let mut new = cur;
    for k in (1..order).rev() {
        if a[new][k] > A_LIMIT || a[new][k] < -A_LIMIT {
            return 0;
        }
        let rc_q31 = -(a[new][k] << (31 - QA));
        let rc_mult1_q30 = (1 << 30) - smmul(rc_q31, rc_q31);
        let mult2q = 32 - clz32(rc_mult1_q30.abs());
        let rc_mult2 = inverse32_varq(rc_mult1_q30, mult2q + 30);
        inv_gain = smmul(inv_gain, rc_mult1_q30) << 2;
        let old = new;
        new = k & 1;
        for n in 0..k {
            let tmp = a[old][n].saturating_sub(mul_frac(a[old][k - n - 1], rc_q31, 31));
            let t64 = rshift_round64(i64::from(tmp) * i64::from(rc_mult2), mult2q as u32);
            if t64 > i64::from(i32::MAX) || t64 < i64::from(i32::MIN) {
                return 0;
            }
            a[new][n] = t64 as i32;
        }
    }
    if a[new][0] > A_LIMIT || a[new][0] < -A_LIMIT {
        return 0;
    }
    let rc_q31 = -(a[new][0] << (31 - QA));
    let rc_mult1_q30 = (1 << 30) - smmul(rc_q31, rc_q31);
    smmul(inv_gain, rc_mult1_q30) << 2
}

/// The LPC filter of an NLSF vector in Q12 (`silk_NLSF2A`), made stable.
pub fn nlsf2a(a_q12: &mut [i16], nlsf: &[i16]) {
    const QA: u32 = 16;
    const ORDER16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDER10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
    let d = nlsf.len();
    let ordering: &[usize] = if d == 16 { &ORDER16 } else { &ORDER10 };
    let mut cos_lsf = [0i32; MAX_LPC_ORDER];
    for k in 0..d {
        let f_int = i32::from(nlsf[k]) >> 8;
        let f_frac = i32::from(nlsf[k]) - (f_int << 8);
        let cos_val = i32::from(LSF_COS_TAB_Q12[f_int as usize]);
        let delta = i32::from(LSF_COS_TAB_Q12[f_int as usize + 1]) - cos_val;
        cos_lsf[ordering[k]] = rshift_round((cos_val << 8) + delta * f_frac, 20 - QA);
    }
    let dd = d / 2;
    // The even and odd polynomials (`silk_NLSF2A_find_poly`).
    let find_poly = |start: usize| -> [i32; MAX_LPC_ORDER / 2 + 1] {
        let mut out = [0i32; MAX_LPC_ORDER / 2 + 1];
        out[0] = 1 << QA;
        out[1] = -cos_lsf[start];
        for k in 1..dd {
            let f = cos_lsf[start + 2 * k];
            out[k + 1] =
                (out[k - 1] << 1) - rshift_round64(i64::from(f) * i64::from(out[k]), QA) as i32;
            for n in (2..=k).rev() {
                out[n] +=
                    out[n - 2] - rshift_round64(i64::from(f) * i64::from(out[n - 1]), QA) as i32;
            }
            out[1] -= f;
        }
        out
    };
    let p = find_poly(0);
    let q = find_poly(1);
    let mut a32 = [0i32; MAX_LPC_ORDER];
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];
        a32[k] = -qtmp - ptmp;
        a32[d - k - 1] = qtmp - ptmp;
    }
    // Keep the coefficients within 16 bits.
    let mut i = 0;
    while i < 10 {
        let (mut maxabs, mut idx) = (0i32, 0usize);
        for (k, &v) in a32[..d].iter().enumerate() {
            if v.abs() > maxabs {
                maxabs = v.abs();
                idx = k;
            }
        }
        maxabs = rshift_round(maxabs, QA + 1 - 12);
        if maxabs > i32::from(i16::MAX) {
            maxabs = maxabs.min(163_838);
            let sc_q16 =
                65470 - ((maxabs - i32::from(i16::MAX)) << 14) / ((maxabs * (idx as i32 + 1)) >> 2);
            bwexpander_32(&mut a32[..d], sc_q16);
        } else {
            break;
        }
        i += 1;
    }
    if i == 10 {
        for k in 0..d {
            a_q12[k] = sat16(rshift_round(a32[k], QA + 1 - 12)) as i16;
            a32[k] = i32::from(a_q12[k]) << (QA + 1 - 12);
        }
    } else {
        for k in 0..d {
            a_q12[k] = rshift_round(a32[k], QA + 1 - 12) as i16;
        }
    }
    // Bandwidth expansion until the filter's prediction gain is below 10^4.
    for i in 0..16 {
        if lpc_inverse_pred_gain(&a_q12[..d]) < 107_374 {
            bwexpander_32(&mut a32[..d], 65536 - (2 << i));
            for k in 0..d {
                a_q12[k] = rshift_round(a32[k], QA + 1 - 12) as i16;
            }
        } else {
            break;
        }
    }
}

/// The pitch lags of the subframes (`silk_decode_pitch`).
fn decode_pitch(lag_index: i32, contour: usize, lags: &mut [i32], fs_khz: i32) {
    let four = lags.len() == 4;
    let min_lag = 2 * fs_khz;
    let max_lag = 18 * fs_khz;
    let lag = min_lag + lag_index;
    for (k, l) in lags.iter_mut().enumerate() {
        let delta = match (fs_khz == 8, four) {
            (true, true) => CB_LAGS_STAGE2[k][contour],
            (true, false) => CB_LAGS_STAGE2_10_MS[k][contour],
            (false, true) => CB_LAGS_STAGE3[k][contour],
            (false, false) => CB_LAGS_STAGE3_10_MS[k][contour],
        };
        *l = limit(lag + i32::from(delta), min_lag, max_lag);
    }
}

/// All of a frame's parameters (`silk_decode_parameters`).
pub fn decode_parameters(ch: &mut Channel, ctrl: &mut Control, cond: Conditioning) {
    let nb = ch.nb_subfr;
    gains_dequant(
        &mut ctrl.gains_q16[..nb],
        &ch.indices.gains,
        &mut ch.last_gain_index,
        cond == Conditioning::Conditional,
    );

    let order = ch.lpc_order;
    let mut nlsf = [0i16; MAX_LPC_ORDER];
    nlsf_decode(&mut nlsf, &ch.indices.nlsf, ch.nlsf_cb);
    nlsf2a(&mut ctrl.pred_coef_q12[1][..order], &nlsf[..order]);
    // No interpolation right after a reset.
    if ch.first_frame_after_reset {
        ch.indices.nlsf_interp_q2 = 4;
    }
    if ch.indices.nlsf_interp_q2 < 4 {
        let f = i32::from(ch.indices.nlsf_interp_q2);
        let mut nlsf0 = [0i16; MAX_LPC_ORDER];
        for i in 0..order {
            let prev = i32::from(ch.prev_nlsf_q15[i]);
            nlsf0[i] = (prev + ((f * (i32::from(nlsf[i]) - prev)) >> 2)) as i16;
        }
        nlsf2a(&mut ctrl.pred_coef_q12[0][..order], &nlsf0[..order]);
    } else {
        ctrl.pred_coef_q12[0] = ctrl.pred_coef_q12[1];
    }
    ch.prev_nlsf_q15[..order].copy_from_slice(&nlsf[..order]);
    // Bandwidth expansion after a loss.
    if ch.loss_cnt != 0 {
        bwexpander(&mut ctrl.pred_coef_q12[0][..order], 63570);
        bwexpander(&mut ctrl.pred_coef_q12[1][..order], 63570);
    }

    if i32::from(ch.indices.signal_type) == TYPE_VOICED {
        decode_pitch(
            i32::from(ch.indices.lag),
            ch.indices.contour as usize,
            &mut ctrl.pitch_l[..nb],
            ch.fs_khz,
        );
        let cbk: &[[i8; 5]] = match ch.indices.per_index {
            0 => &LTP_GAIN_VQ_0,
            1 => &LTP_GAIN_VQ_1,
            _ => &LTP_GAIN_VQ_2,
        };
        for k in 0..nb {
            let row = &cbk[ch.indices.ltp[k] as usize];
            for (c, &v) in ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER]
                .iter_mut()
                .zip(row)
            {
                *c = (i32::from(v) << 7) as i16;
            }
        }
        ctrl.ltp_scale_q14 = i32::from(LTP_SCALES_Q14[ch.indices.ltp_scale_index as usize]);
    } else {
        ctrl.pitch_l[..nb].fill(0);
        ctrl.ltp_coef_q14[..LTP_ORDER * nb].fill(0);
        ch.indices.per_index = 0;
        ctrl.ltp_scale_q14 = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_codebook_constants_are_the_reference_fixed_point_values() {
        // SILK_FIX_CONST(0.18, 16) and (0.15, 16); 0.999 in Q16; 1/1e4 in Q30.
        assert_eq!(
            NLSF_CB_NB_MB.quant_step_q16,
            (0.18f64 * 65536.0 + 0.5) as i32
        );
        assert_eq!(NLSF_CB_WB.quant_step_q16, (0.15f64 * 65536.0 + 0.5) as i32);
        assert_eq!(65470, (0.999f64 * 65536.0 + 0.5) as i32);
        assert_eq!(107_374, (1.0f64 / 1e4 * (1u64 << 30) as f64 + 0.5) as i32);
        assert_eq!(16_773_022, (0.99975f64 * (1u64 << 24) as f64 + 0.5) as i32);
        assert_eq!(102, (0.1f64 * 1024.0 + 0.5) as i32);
    }

    #[test]
    fn stabilized_nlsfs_keep_their_minimum_distances() {
        let mut nlsf = [
            100i16, 90, 5000, 5001, 9000, 20000, 20010, 30000, 32760, 32767,
        ];
        nlsf_stabilize(&mut nlsf, &NLSF_DELTA_MIN_NB_MB_Q15);
        let d = &NLSF_DELTA_MIN_NB_MB_Q15;
        assert!(nlsf[0] >= d[0]);
        for i in 1..10 {
            assert!(
                i32::from(nlsf[i]) - i32::from(nlsf[i - 1]) >= i32::from(d[i]),
                "{nlsf:?}"
            );
        }
        assert!(32768 - i32::from(nlsf[9]) >= i32::from(d[10]));
    }

    #[test]
    fn a_flat_spectrum_gives_a_stable_filter() {
        // Evenly spaced NLSFs are the flat spectrum: a filter near zero, and stable.
        let nlsf: Vec<i16> = (1..=16).map(|i| (i * 32768 / 17) as i16).collect();
        let mut a = [0i16; 16];
        nlsf2a(&mut a, &nlsf);
        assert!(a.iter().all(|&c| c.abs() < 200), "{a:?}");
        assert!(lpc_inverse_pred_gain(&a) > 1 << 29);
    }

    #[test]
    fn gains_step_and_saturate() {
        let mut g = [0i32; 4];
        let mut prev = 10i8;
        gains_dequant(&mut g, &[40, 4, 4, 0], &mut prev, false);
        // An independent first gain, two flat deltas (index 4 is a delta of 0), one step down.
        assert_eq!(g[1], g[0]);
        assert_eq!(g[2], g[0]);
        assert!(g[3] < g[2]);
        assert_eq!(prev, 36);
    }
}
