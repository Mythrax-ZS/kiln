//! What CELT's packet loss concealment needs (RFC 6716 section 4.4; concealment is not
//! normative, this follows the reference's `celt/celt_lpc.c` and `celt/pitch.c`): linear
//! prediction of the last output and a pitch search over it.

/// The autocorrelation of `x` at lags 0..=`lag`, with the ends windowed (`_celt_autocorr`).
pub fn autocorr(x: &[f32], ac: &mut [f32], window: Option<&[f32]>, lag: usize) {
    let n = x.len();
    let mut xx = x.to_vec();
    if let Some(w) = window {
        for (i, &wi) in w.iter().enumerate() {
            xx[i] = x[i] * wi;
            xx[n - i - 1] = x[n - i - 1] * wi;
        }
    }
    for l in (0..=lag).rev() {
        ac[l] = (l..n).map(|i| xx[i] * xx[i - l]).sum();
    }
    ac[0] += 10.0;
}

/// Levinson-Durbin: prediction coefficients from an autocorrelation (`_celt_lpc`).
pub fn lpc(out: &mut [f32], ac: &[f32]) {
    let p = out.len();
    out.fill(0.0);
    let mut error = ac[0];
    if ac[0] == 0.0 {
        return;
    }
    for i in 0..p {
        let mut rr = 0f32;
        for j in 0..i {
            rr += out[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        out[i] = r;
        for j in 0..(i + 1) >> 1 {
            let (t1, t2) = (out[j], out[i - 1 - j]);
            out[j] = t1 + r * t2;
            out[i - 1 - j] = t2 + r * t1;
        }
        error -= r * r * error;
        // Stop at 30 dB of prediction gain.
        if error < 0.001 * ac[0] {
            break;
        }
    }
}

/// The FIR filter `y = x + sum num[j] * x[-1-j]`, with its memory (`celt_fir`). `x` and `y`
/// may be the same buffer.
pub fn fir(x: &mut [f32], num: &[f32], mem: &mut [f32]) {
    for v in x.iter_mut() {
        let input = *v;
        let mut sum = input;
        for (n, m) in num.iter().zip(mem.iter()) {
            sum += n * m;
        }
        mem.copy_within(0..mem.len() - 1, 1);
        mem[0] = input;
        *v = sum;
    }
}

/// The IIR filter `y = x - sum den[j] * y[-1-j]`, with its memory (`celt_iir`).
pub fn iir(x: &mut [f32], den: &[f32], mem: &mut [f32]) {
    for v in x.iter_mut() {
        let mut sum = *v;
        for (d, m) in den.iter().zip(mem.iter()) {
            sum -= d * m;
        }
        mem.copy_within(0..mem.len() - 1, 1);
        mem[0] = sum;
        *v = sum;
    }
}

/// Halve the rate of the channels' sum and whiten it a little (`pitch_downsample`).
pub fn pitch_downsample(channels: &[&[f32]], len: usize) -> Vec<f32> {
    let half = len >> 1;
    let mut lp = vec![0f32; half];
    for x in channels {
        for i in 1..half {
            lp[i] += 0.5 * (0.5 * (x[2 * i - 1] + x[2 * i + 1]) + x[2 * i]);
        }
        lp[0] += 0.5 * (0.5 * x[1] + x[0]);
    }
    let mut ac = [0f32; 5];
    autocorr(&lp, &mut ac, None, 4);
    // A noise floor at -40 dB and lag windowing.
    ac[0] *= 1.0001;
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        *a -= *a * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut coefs = [0f32; 4];
    lpc(&mut coefs, &ac);
    let mut tmp = 1.0f32;
    for c in coefs.iter_mut() {
        tmp *= 0.9;
        *c *= tmp;
    }
    let mut mem = [0f32; 4];
    fir(&mut lp, &coefs, &mut mem);
    let mut mem1 = [0f32; 1];
    fir(&mut lp, &[0.8], &mut mem1);
    lp
}

/// The two best lags of a normalized correlation (`find_best_pitch`).
fn best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut syy = 1f32;
    for v in &y[..len] {
        syy += v * v;
    }
    let mut best_num = [-1f32; 2];
    let mut best_den = [0f32; 2];
    let mut best = [0usize, 1];
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let num = xcorr[i] * xcorr[i];
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best[1] = best[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
    best
}

/// The lag at which `y` best matches `x` (`pitch_search`): a search at a quarter of the rate,
/// refined at half the rate.
pub fn pitch_search(x: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    let x4: Vec<f32> = (0..len >> 2).map(|j| x[2 * j]).collect();
    let y4: Vec<f32> = (0..lag >> 2).map(|j| y[2 * j]).collect();
    let mut xcorr = vec![0f32; max_pitch >> 1];
    for i in 0..max_pitch >> 2 {
        let sum: f32 = x4.iter().zip(&y4[i..]).map(|(a, b)| a * b).sum();
        xcorr[i] = sum.max(-1.0);
    }
    let best = best_pitch(&xcorr, &y4, len >> 2, max_pitch >> 2);
    for i in 0..max_pitch >> 1 {
        xcorr[i] = 0.0;
        if (i as i32 - 2 * best[0] as i32).abs() > 2 && (i as i32 - 2 * best[1] as i32).abs() > 2 {
            continue;
        }
        let sum: f32 = x[..len >> 1].iter().zip(&y[i..]).map(|(a, b)| a * b).sum();
        xcorr[i] = sum.max(-1.0);
    }
    let best = best_pitch(&xcorr, y, len >> 1, max_pitch >> 1);
    // Refine by pseudo-interpolation.
    let mut offset = 0i32;
    if best[0] > 0 && best[0] < (max_pitch >> 1) - 1 {
        let (a, b, c) = (xcorr[best[0] - 1], xcorr[best[0]], xcorr[best[0] + 1]);
        if c - a > 0.7 * (b - a) {
            offset = 1;
        } else if a - c > 0.7 * (b - c) {
            offset = -1;
        }
    }
    (2 * best[0] as i32 - offset) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_of_a_sine_leaves_a_small_residual() {
        // At the scale of the decoder (16-bit units), where the noise floor of 10 is small.
        let x: Vec<f32> = (0..512).map(|i| 1000.0 * (i as f32 * 0.2).sin()).collect();
        let mut ac = [0f32; 3];
        autocorr(&x, &mut ac, None, 2);
        let mut a = [0f32; 2];
        lpc(&mut a, &ac);
        // Near x[n] = 2 cos(0.2) x[n-1] - x[n-2], the finite segment biasing it a little.
        assert!(
            (a[0] + 2.0 * 0.2f32.cos()).abs() < 0.2 && (a[1] - 1.0).abs() < 0.2,
            "{a:?}"
        );
        let mut residual = x.clone();
        fir(&mut residual, &a, &mut [0.0; 2]);
        let energy = |v: &[f32]| v.iter().map(|s| s * s).sum::<f32>();
        assert!(energy(&residual[2..]) < 0.05 * energy(&x), "{a:?}");
    }

    #[test]
    fn the_pitch_search_finds_a_period() {
        // As the concealment uses it: 2048 samples, the last 1328 compared with earlier ones.
        let period = 150.0f32;
        let tau = 2.0 * std::f32::consts::PI;
        let signal: Vec<f32> = (0..2048)
            .map(|i| (tau * i as f32 / period).sin() + 0.5 * (2.0 * tau * i as f32 / period).sin())
            .collect();
        let lp = pitch_downsample(&[&signal], 2048);
        let found = pitch_search(&lp[720 >> 1..], &lp, 2048 - 720, 720 - 100);
        let lag = 720.0 - found as f32;
        let off = (lag / period - (lag / period).round()).abs() * period;
        assert!(off <= 2.0, "a lag of {lag}");
    }
}
