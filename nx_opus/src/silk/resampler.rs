//! The resampler from SILK's internal rate (8, 12 or 16 kHz) to the output rate (RFC 6716
//! section 4.2.9; the method is not normative, but the reference's `silk/resampler*.c` is what
//! the test vectors were made with, so this follows it, with the buffer fix of RFC 8251
//! section 5).

use super::fixed::*;
use super::tables::*;

const ORDER_FIR_12: usize = 8;
const DOWN_ORDER_FIR0: usize = 18;
const DOWN_ORDER_FIR1: usize = 24;
const DOWN_ORDER_FIR2: usize = 36;
const MAX_BATCH_MS: i32 = 10;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Kind {
    #[default]
    Copy,
    Up2,
    IirFir,
    DownFir,
}

#[derive(Clone, Copy, Debug)]
pub struct Resampler {
    kind: Kind,
    s_iir: [i32; 6],
    /// The FIR history: 16-bit samples for the upsampler, 32-bit for the downsampler.
    s_fir16: [i16; ORDER_FIR_12],
    s_fir32: [i32; DOWN_ORDER_FIR2],
    delay_buf: [i16; 48],
    batch_size: usize,
    inv_ratio_q16: i32,
    fir_order: usize,
    fir_fracs: i32,
    fs_in_khz: usize,
    fs_out_khz: usize,
    input_delay: usize,
    coefs: &'static [i16],
}

impl Default for Resampler {
    fn default() -> Resampler {
        Resampler {
            kind: Kind::Copy,
            s_iir: [0; 6],
            s_fir16: [0; ORDER_FIR_12],
            s_fir32: [0; DOWN_ORDER_FIR2],
            delay_buf: [0; 48],
            batch_size: 0,
            inv_ratio_q16: 0,
            fir_order: 0,
            fir_fracs: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            input_delay: 0,
            coefs: &[],
        }
    }
}

/// The delay of the decoder's resamplers, by input (8, 12, 16 kHz) and output rate (8, 12, 16,
/// 24, 48 kHz).
const DELAY_MATRIX_DEC: [[usize; 5]; 3] = [[4, 0, 2, 0, 0], [0, 9, 4, 7, 4], [0, 3, 12, 7, 7]];

fn rate_id(r: i32) -> usize {
    match r {
        8000 => 0,
        12000 => 1,
        16000 => 2,
        24000 => 3,
        _ => 4,
    }
}

impl Resampler {
    /// `silk_resampler_init` for the decoder.
    pub fn new(fs_in: i32, fs_out: i32) -> Resampler {
        let mut s = Resampler {
            input_delay: DELAY_MATRIX_DEC[rate_id(fs_in)][rate_id(fs_out)],
            fs_in_khz: (fs_in / 1000) as usize,
            fs_out_khz: (fs_out / 1000) as usize,
            batch_size: (fs_in / 1000 * MAX_BATCH_MS) as usize,
            coefs: &[],
            ..Resampler::default()
        };
        let mut up2x = 0;
        if fs_out > fs_in {
            if fs_out == 2 * fs_in {
                s.kind = Kind::Up2;
            } else {
                s.kind = Kind::IirFir;
                up2x = 1;
            }
        } else if fs_out < fs_in {
            s.kind = Kind::DownFir;
            let (fracs, order, coefs): (i32, usize, &'static [i16]) = if fs_out * 4 == fs_in * 3 {
                (3, DOWN_ORDER_FIR0, &RESAMPLER_3_4_COEFS)
            } else if fs_out * 3 == fs_in * 2 {
                (2, DOWN_ORDER_FIR0, &RESAMPLER_2_3_COEFS)
            } else if fs_out * 2 == fs_in {
                (1, DOWN_ORDER_FIR1, &RESAMPLER_1_2_COEFS)
            } else if fs_out * 3 == fs_in {
                (1, DOWN_ORDER_FIR2, &RESAMPLER_1_3_COEFS)
            } else if fs_out * 4 == fs_in {
                (1, DOWN_ORDER_FIR2, &RESAMPLER_1_4_COEFS)
            } else {
                (1, DOWN_ORDER_FIR2, &RESAMPLER_1_6_COEFS)
            };
            s.fir_fracs = fracs;
            s.fir_order = order;
            s.coefs = coefs;
        }
        // The ratio of input to output samples, rounded up.
        s.inv_ratio_q16 = ((fs_in << (14 + up2x)) / fs_out) << 2;
        while smulww(s.inv_ratio_q16, fs_out) < fs_in << up2x {
            s.inv_ratio_q16 += 1;
        }
        s
    }

    /// Resample `input` (a whole number of milliseconds) into `out` (`silk_resampler`).
    pub fn process(&mut self, out: &mut [i16], input: &[i16]) {
        let n = self.fs_in_khz - self.input_delay;
        let d = self.input_delay;
        self.delay_buf[d..d + n].copy_from_slice(&input[..n]);
        let first: Vec<i16> = self.delay_buf[..self.fs_in_khz].to_vec();
        let rest = &input[n..];
        let k = self.fs_out_khz;
        match self.kind {
            Kind::Up2 => {
                up2_hq(&mut self.s_iir, &mut out[..2 * first.len()], &first);
                let m = input.len() - self.fs_in_khz;
                up2_hq(&mut self.s_iir, &mut out[k..k + 2 * m], &rest[..m]);
            }
            Kind::IirFir => {
                let used = self.iir_fir(out, &first);
                debug_assert_eq!(used, k);
                self.iir_fir(&mut out[k..], &rest[..input.len() - self.fs_in_khz]);
            }
            Kind::DownFir => {
                self.down_fir(out, &first);
                self.down_fir(&mut out[k..], &rest[..input.len() - self.fs_in_khz]);
            }
            Kind::Copy => {
                out[..self.fs_in_khz].copy_from_slice(&first);
                let m = input.len() - self.fs_in_khz;
                out[k..k + m].copy_from_slice(&rest[..m]);
            }
        }
        self.delay_buf[..d].copy_from_slice(&input[input.len() - d..]);
    }

    /// 2x upsampling then FIR interpolation (`silk_resampler_private_IIR_FIR`); returns the
    /// number of samples written.
    fn iir_fir(&mut self, out: &mut [i16], mut input: &[i16]) -> usize {
        let mut buf = vec![0i16; 2 * self.batch_size + ORDER_FIR_12];
        buf[..ORDER_FIR_12].copy_from_slice(&self.s_fir16);
        let inc = self.inv_ratio_q16;
        let mut written = 0;
        let mut n;
        loop {
            n = input.len().min(self.batch_size);
            up2_hq(
                &mut self.s_iir,
                &mut buf[ORDER_FIR_12..ORDER_FIR_12 + 2 * n],
                &input[..n],
            );
            let max_index_q16 = (n as i32) << 17;
            let mut index_q16 = 0;
            while index_q16 < max_index_q16 {
                let t = smulwb(index_q16 & 0xFFFF, 12) as usize;
                let b = &buf[(index_q16 >> 16) as usize..];
                let f = &RESAMPLER_FRAC_FIR_12;
                let mut r = smulbb(i32::from(b[0]), i32::from(f[t][0]));
                r = smlabb(r, i32::from(b[1]), i32::from(f[t][1]));
                r = smlabb(r, i32::from(b[2]), i32::from(f[t][2]));
                r = smlabb(r, i32::from(b[3]), i32::from(f[t][3]));
                r = smlabb(r, i32::from(b[4]), i32::from(f[11 - t][3]));
                r = smlabb(r, i32::from(b[5]), i32::from(f[11 - t][2]));
                r = smlabb(r, i32::from(b[6]), i32::from(f[11 - t][1]));
                r = smlabb(r, i32::from(b[7]), i32::from(f[11 - t][0]));
                out[written] = sat16(rshift_round(r, 15)) as i16;
                written += 1;
                index_q16 += inc;
            }
            input = &input[n..];
            if !input.is_empty() {
                buf.copy_within(2 * n..2 * n + ORDER_FIR_12, 0);
            } else {
                break;
            }
        }
        self.s_fir16
            .copy_from_slice(&buf[2 * n..2 * n + ORDER_FIR_12]);
        written
    }

    /// An AR2 filter then FIR decimation (`silk_resampler_private_down_FIR`).
    fn down_fir(&mut self, out: &mut [i16], mut input: &[i16]) {
        let order = self.fir_order;
        let mut buf = vec![0i32; self.batch_size + order];
        buf[..order].copy_from_slice(&self.s_fir32[..order]);
        let coefs = self.coefs;
        let fir = &coefs[2..];
        let inc = self.inv_ratio_q16;
        let mut written = 0;
        let mut n;
        loop {
            n = input.len().min(self.batch_size);
            // The second-order AR filter, output in Q8 (`silk_resampler_private_AR2`).
            for k in 0..n {
                let o = self.s_iir[0].wrapping_add(i32::from(input[k]) << 8);
                buf[order + k] = o;
                let o4 = o << 2;
                self.s_iir[0] = smlawb(self.s_iir[1], o4, i32::from(coefs[0]));
                self.s_iir[1] = smulwb(o4, i32::from(coefs[1]));
            }
            let max_index_q16 = (n as i32) << 16;
            let mut index_q16 = 0;
            while index_q16 < max_index_q16 {
                let b = &buf[(index_q16 >> 16) as usize..];
                let mut r;
                match order {
                    DOWN_ORDER_FIR0 => {
                        let ind = smulwb(index_q16 & 0xFFFF, self.fir_fracs) as usize;
                        let p = &fir[DOWN_ORDER_FIR0 / 2 * ind..];
                        r = smulwb(b[0], i32::from(p[0]));
                        for j in 1..9 {
                            r = smlawb(r, b[j], i32::from(p[j]));
                        }
                        let p = &fir[DOWN_ORDER_FIR0 / 2 * (self.fir_fracs as usize - 1 - ind)..];
                        for j in 0..9 {
                            r = smlawb(r, b[17 - j], i32::from(p[j]));
                        }
                    }
                    _ => {
                        // A symmetric filter of `order` taps.
                        r = smulwb(b[0].wrapping_add(b[order - 1]), i32::from(fir[0]));
                        for j in 1..order / 2 {
                            r = smlawb(r, b[j].wrapping_add(b[order - 1 - j]), i32::from(fir[j]));
                        }
                    }
                }
                out[written] = sat16(rshift_round(r, 6)) as i16;
                written += 1;
                index_q16 += inc;
            }
            input = &input[n..];
            if input.len() > 1 {
                buf.copy_within(n..n + order, 0);
            } else {
                break;
            }
        }
        self.s_fir32[..order].copy_from_slice(&buf[n..n + order]);
    }
}

/// 2x upsampling by two chains of all-pass filters (`silk_resampler_private_up2_HQ`).
fn up2_hq(s: &mut [i32; 6], out: &mut [i16], input: &[i16]) {
    let c0 = &RESAMPLER_UP2_HQ_0;
    let c1 = &RESAMPLER_UP2_HQ_1;
    for (k, &x) in input.iter().enumerate() {
        let in32 = i32::from(x) << 10;
        // Even output.
        let y = in32 - s[0];
        let xx = smulwb(y, i32::from(c0[0]));
        let o1 = s[0] + xx;
        s[0] = in32 + xx;
        let y = o1 - s[1];
        let xx = smulwb(y, i32::from(c0[1]));
        let o2 = s[1] + xx;
        s[1] = o1 + xx;
        let y = o2 - s[2];
        let xx = smlawb(y, y, i32::from(c0[2]));
        let o1 = s[2] + xx;
        s[2] = o2 + xx;
        out[2 * k] = sat16(rshift_round(o1, 10)) as i16;
        // Odd output.
        let y = in32 - s[3];
        let xx = smulwb(y, i32::from(c1[0]));
        let o1 = s[3] + xx;
        s[3] = in32 + xx;
        let y = o1 - s[4];
        let xx = smulwb(y, i32::from(c1[1]));
        let o2 = s[4] + xx;
        s[4] = o1 + xx;
        let y = o2 - s[5];
        let xx = smlawb(y, y, i32::from(c1[2]));
        let o1 = s[5] + xx;
        s[5] = o2 + xx;
        out[2 * k + 1] = sat16(rshift_round(o1, 10)) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_decoder_rate_pair_keeps_the_sample_count() {
        for fs_in in [8000, 12000, 16000] {
            for fs_out in [8000, 12000, 16000, 24000, 48000] {
                let mut r = Resampler::new(fs_in, fs_out);
                let n_in = (fs_in / 1000 * 20) as usize;
                let n_out = (fs_out / 1000 * 20) as usize;
                let input: Vec<i16> = (0..n_in)
                    .map(|i| ((i as f32 * 0.3).sin() * 8000.0) as i16)
                    .collect();
                let mut out = vec![0i16; n_out + 8];
                r.process(&mut out, &input);
                r.process(&mut out, &input);
                // A tone stays a tone of about the same level.
                let rms = |v: &[i16]| {
                    (v.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
                };
                let a = rms(&input);
                let b = rms(&out[n_out / 4..n_out]);
                assert!(
                    b > 0.5 * a && b < 1.5 * a,
                    "{fs_in} -> {fs_out}: {a} vs {b}"
                );
            }
        }
    }
}
