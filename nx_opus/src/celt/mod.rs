//! The CELT layer (RFC 6716 section 4.3): an MDCT codec of 2.5 to 20 ms frames. A frame codes
//! the energy of 21 bands, then each band's normalized shape by PVQ; the decoder scales the
//! shapes back, transforms them to time, and runs a pitch post-filter and de-emphasis.
//!
//! The RFC leaves many details to its normative reference implementation (appendix A, Simplified
//! BSD licence). This module is written from the prose and that code (`celt/celt.c`); the
//! decisions that touch the range coder are bit-exact, and the arithmetic of the synthesis is
//! single-precision floating point, as the reference's float build has it.

mod bands;
mod energy;
mod mdct;
pub(crate) mod mode;
mod plc;
mod pvq;
mod rate;
pub(crate) mod tables;

use crate::range::RangeDecoder;
use bands::{anti_collapse, decode_bands, denormalize, lcg, BandParams};
use energy::{amplitudes, decode_coarse, decode_final, decode_fine};
use mdct::Imdct;
use mode::{mode, BITRES, MAX_LM, OVERLAP, SHORT_MDCT};
use pvq::renormalize;
use rate::compute_allocation;
use tables::{
    BANDS, COMB_GAINS, EBANDS, PREEMPHASIS, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT, TRIM_ICDF,
};

/// The history each channel keeps: what the post-filter and the concealment look back on.
const DECODE_BUFFER: usize = 2048;
/// The longest pitch period the concealment repeats.
const MAX_PERIOD: usize = 1024;
const LPC_ORDER: usize = 24;
const COMBFILTER_MINPERIOD: i32 = 15;

pub struct CeltDecoder {
    /// Output channels.
    channels: usize,
    /// Channels coded in the stream.
    stream_channels: usize,
    /// 48 kHz divided by the output rate.
    downsample: usize,
    start: usize,
    end: usize,
    /// The range coder's final state of the last frame, which also seeds the folding noise.
    rng: u32,
    last_pitch_index: usize,
    loss_count: u32,
    postfilter_period: i32,
    postfilter_period_old: i32,
    postfilter_gain: f32,
    postfilter_gain_old: f32,
    postfilter_tapset: usize,
    postfilter_tapset_old: usize,
    preemph_mem: [f32; 2],
    /// Per channel: `DECODE_BUFFER` samples of output history, then the MDCT overlap.
    decode_mem: [Vec<f32>; 2],
    lpc: [[f32; LPC_ORDER]; 2],
    old_band_e: [f32; 2 * BANDS],
    old_log_e: [f32; 2 * BANDS],
    old_log_e2: [f32; 2 * BANDS],
    background_log_e: [f32; 2 * BANDS],
    /// Apply the 180-degree phase of intensity stereo (RFC 8251 section 10 allows leaving it).
    pub phase_inversion: bool,
}

/// One comb filter step on a buffer in place (`comb_filter`): the gains and taps cross-fade
/// from (t0, g0, tap0) to (t1, g1, tap1) over the first `window.len()` samples.
#[allow(clippy::too_many_arguments)]
fn comb_filter(
    buf: &mut [f32],
    at: usize,
    n: usize,
    t0: i32,
    t1: i32,
    g0: f32,
    g1: f32,
    tap0: usize,
    tap1: usize,
    window: &[f32],
) {
    if g0 == 0.0 && g1 == 0.0 {
        // The identity (and a period of 0 would reach past the buffer).
        return;
    }
    let (t0, t1) = (t0 as usize, t1 as usize);
    let (g00, g01, g02) = (
        g0 * COMB_GAINS[tap0][0],
        g0 * COMB_GAINS[tap0][1],
        g0 * COMB_GAINS[tap0][2],
    );
    let (g10, g11, g12) = (
        g1 * COMB_GAINS[tap1][0],
        g1 * COMB_GAINS[tap1][1],
        g1 * COMB_GAINS[tap1][2],
    );
    let overlap = window.len();
    for (i, w) in window.iter().enumerate() {
        let p = at + i;
        let f = w * w;
        let nf = 1.0 - f;
        buf[p] = buf[p]
            + nf * g00 * buf[p - t0]
            + nf * g01 * buf[p - t0 - 1]
            + nf * g01 * buf[p - t0 + 1]
            + nf * g02 * buf[p - t0 - 2]
            + nf * g02 * buf[p - t0 + 2]
            + f * g10 * buf[p - t1]
            + f * g11 * buf[p - t1 - 1]
            + f * g11 * buf[p - t1 + 1]
            + f * g12 * buf[p - t1 - 2]
            + f * g12 * buf[p - t1 + 2];
    }
    for i in overlap..n {
        let p = at + i;
        buf[p] = buf[p]
            + g10 * buf[p - t1]
            + g11 * buf[p - t1 - 1]
            + g11 * buf[p - t1 + 1]
            + g12 * buf[p - t1 - 2]
            + g12 * buf[p - t1 + 2];
    }
}

impl CeltDecoder {
    /// A decoder for `channels` output channels at `rate` (48, 24, 16, 12 or 8 kHz).
    pub fn new(rate: u32, channels: usize) -> CeltDecoder {
        let downsample = match rate {
            48000 => 1,
            24000 => 2,
            16000 => 3,
            12000 => 4,
            8000 => 6,
            _ => panic!("a CELT output rate of 48, 24, 16, 12 or 8 kHz"),
        };
        let mut d = CeltDecoder {
            channels,
            stream_channels: channels,
            downsample,
            start: 0,
            end: BANDS,
            rng: 0,
            last_pitch_index: 0,
            loss_count: 0,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            preemph_mem: [0.0; 2],
            decode_mem: [
                vec![0.0; DECODE_BUFFER + OVERLAP],
                vec![0.0; DECODE_BUFFER + OVERLAP],
            ],
            lpc: [[0.0; LPC_ORDER]; 2],
            old_band_e: [0.0; 2 * BANDS],
            old_log_e: [0.0; 2 * BANDS],
            old_log_e2: [0.0; 2 * BANDS],
            background_log_e: [0.0; 2 * BANDS],
            phase_inversion: true,
        };
        d.reset();
        d
    }

    /// Forget everything about the past frames (`OPUS_RESET_STATE`).
    pub fn reset(&mut self) {
        self.rng = 0;
        self.last_pitch_index = 0;
        self.loss_count = 0;
        self.postfilter_period = 0;
        self.postfilter_period_old = 0;
        self.postfilter_gain = 0.0;
        self.postfilter_gain_old = 0.0;
        self.postfilter_tapset = 0;
        self.postfilter_tapset_old = 0;
        self.preemph_mem = [0.0; 2];
        for m in &mut self.decode_mem {
            m.fill(0.0);
        }
        self.lpc = [[0.0; LPC_ORDER]; 2];
        self.old_band_e = [0.0; 2 * BANDS];
        self.old_log_e = [-28.0; 2 * BANDS];
        self.old_log_e2 = [-28.0; 2 * BANDS];
        self.background_log_e = [0.0; 2 * BANDS];
    }

    /// The first band coded (17 in hybrid frames, whose lower bands are SILK's).
    pub fn set_start_band(&mut self, band: usize) {
        self.start = band;
    }

    /// One past the last band coded, from the bandwidth.
    pub fn set_end_band(&mut self, band: usize) {
        self.end = band;
    }

    pub fn set_stream_channels(&mut self, channels: usize) {
        self.stream_channels = channels;
    }

    /// The range coder's state at the end of the last frame decoded.
    pub fn final_range(&self) -> u32 {
        self.rng
    }

    /// The pitch period of the post-filter, which the concealment of a hybrid stream reuses.
    pub fn pitch(&self) -> i32 {
        self.postfilter_period
    }

    /// Decode a frame of `frame_size` samples at the output rate into `pcm` (interleaved,
    /// scaled to [-1, 1]). `None` conceals a lost frame.
    pub fn decode(&mut self, dec: Option<&mut RangeDecoder>, pcm: &mut [f32], frame_size: usize) {
        let n = frame_size * self.downsample;
        let lm = (0..=MAX_LM)
            .find(|&lm| SHORT_MDCT << lm == n)
            .expect("a CELT frame of 2.5, 5, 10 or 20 ms");
        let dec = match dec {
            Some(d) if d.len() > 1 => d,
            _ => {
                self.decode_lost(pcm, n, lm);
                return;
            }
        };
        let m = 1usize << lm;
        let cc = self.channels;
        let c = self.stream_channels;
        let eff_end = self.end.min(BANDS);
        let (start, end) = (self.start, self.end);

        if c == 1 {
            for i in 0..BANDS {
                self.old_band_e[i] = self.old_band_e[i].max(self.old_band_e[BANDS + i]);
            }
        }
        let len = dec.len() as u32;
        let total_bits = len * 8;
        let mut tell = dec.tell();
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.bit_logp(15)
        } else {
            false
        };
        if silence {
            // As though every bit had been read.
            tell = total_bits;
            dec.skip_to_end();
        }

        // The pitch post-filter (section 4.3.7.1).
        let (mut pf_pitch, mut pf_gain, mut pf_tapset) = (0i32, 0f32, 0usize);
        if start == 0 && tell + 16 <= total_bits {
            if dec.bit_logp(1) {
                let octave = dec.uint(6);
                pf_pitch = ((16 << octave) + dec.bits(4 + octave) - 1) as i32;
                let qg = dec.bits(3);
                if dec.tell() + 2 <= total_bits {
                    pf_tapset = dec.icdf(&TAPSET_ICDF, 2);
                }
                pf_gain = 0.093_75 * (qg + 1) as f32;
            }
            tell = dec.tell();
        }
        let transient = if lm > 0 && tell + 3 <= total_bits {
            let t = dec.bit_logp(3);
            tell = dec.tell();
            t
        } else {
            false
        };
        let intra = tell + 3 <= total_bits && dec.bit_logp(3);

        decode_coarse(&mut self.old_band_e, start, end, intra, dec, c, lm);
        let mut tf_res = [0i32; BANDS];
        tf_decode(start, end, transient, &mut tf_res, lm, dec);
        let spread = if dec.tell() + 4 <= total_bits {
            dec.icdf(&SPREAD_ICDF, 5)
        } else {
            2
        };

        // Dynamic allocation boosts (section 4.3.3).
        let md = mode();
        let mut caps = [0i32; BANDS];
        for (i, cap) in caps.iter_mut().enumerate() {
            *cap = md.cap(i, lm, c);
        }
        let mut offsets = [0i32; BANDS];
        let mut dynalloc_logp = 6;
        let mut total_frac = (total_bits as i32) << BITRES;
        let mut tell_frac = dec.tell_frac() as i32;
        for i in start..end {
            let width = ((c * (EBANDS[i + 1] - EBANDS[i])) << lm) as i32;
            // Six bits, but at most one and at least an eighth of a bit per bin.
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut loop_logp = dynalloc_logp;
            let mut boost = 0;
            while tell_frac + (loop_logp << BITRES) < total_frac && boost < caps[i] {
                let flag = dec.bit_logp(loop_logp as u32);
                tell_frac = dec.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_frac -= quanta;
                loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = (dynalloc_logp - 1).max(2);
            }
        }
        let trim = if tell_frac + (6 << BITRES) <= total_frac {
            dec.icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };
        let mut bits = ((len as i32 * 8) << BITRES) - dec.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if transient && lm >= 2 && bits >= (lm as i32 + 2) << BITRES {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;
        let alloc = compute_allocation(start, end, &offsets, &caps, trim, bits, c, lm, dec);
        decode_fine(&mut self.old_band_e, start, end, &alloc.fine_quant, dec, c);

        // The shapes (section 4.3.4).
        let mut x = vec![0f32; c * n];
        let params = BandParams {
            start,
            end,
            lm,
            channels: c,
            short_blocks: transient,
            spread,
            dual_stereo: alloc.dual_stereo,
            intensity: alloc.intensity,
            tf_res: &tf_res,
            total_bits: len as i32 * (8 << BITRES) - anti_collapse_rsv,
            balance: alloc.balance,
            coded_bands: alloc.coded_bands,
            pulses: &alloc.pulses,
            phase_inversion: self.phase_inversion,
        };
        let (masks, seed) = decode_bands(&params, &mut x, n, dec, self.rng);
        self.rng = seed;
        let anti_collapse_on = anti_collapse_rsv > 0 && dec.bits(1) != 0;
        let bits_left = len as i32 * 8 - dec.tell() as i32;
        decode_final(
            &mut self.old_band_e,
            start,
            end,
            &alloc.fine_quant,
            &alloc.fine_priority,
            bits_left,
            dec,
            c,
        );
        if anti_collapse_on {
            anti_collapse(
                &mut x,
                n,
                &masks,
                lm,
                c,
                start,
                end,
                &self.old_band_e,
                &self.old_log_e,
                &self.old_log_e2,
                &alloc.pulses,
                self.rng,
            );
        }
        let mut amp = amplitudes(&self.old_band_e, start, end, c);
        if silence {
            amp = [0.0; 2 * BANDS];
            for e in &mut self.old_band_e[..c * BANDS] {
                *e = -28.0;
            }
        }

        // Synthesis (section 4.3.7).
        let mut freq = vec![0f32; cc.max(c) * n];
        denormalize(&x, &mut freq, &amp, eff_end, n, lm, c);
        for ch in 0..cc {
            self.decode_mem[ch].copy_within(n..DECODE_BUFFER, 0);
        }
        let mut bound = m * EBANDS[eff_end];
        if self.downsample != 1 {
            bound = bound.min(n / self.downsample);
        }
        for ch in 0..c {
            let f = &mut freq[ch * n..(ch + 1) * n];
            f[..m * EBANDS[start]].fill(0.0);
            f[bound..].fill(0.0);
        }
        if cc == 2 && c == 1 {
            freq.copy_within(0..n, n);
        }
        if cc == 1 && c == 2 {
            for i in 0..n {
                freq[i] = 0.5 * (freq[i] + freq[n + i]);
            }
        }
        self.inverse_mdcts(&freq, transient, n, lm);

        let syn = DECODE_BUFFER - n;
        self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
        self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
        let window = &md.window;
        for ch in 0..cc {
            let buf = &mut self.decode_mem[ch];
            comb_filter(
                buf,
                syn,
                SHORT_MDCT,
                self.postfilter_period_old,
                self.postfilter_period,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                window,
            );
            if lm != 0 {
                comb_filter(
                    buf,
                    syn + SHORT_MDCT,
                    n - SHORT_MDCT,
                    self.postfilter_period,
                    pf_pitch,
                    self.postfilter_gain,
                    pf_gain,
                    self.postfilter_tapset,
                    pf_tapset,
                    window,
                );
            }
        }
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = pf_pitch;
        self.postfilter_gain = pf_gain;
        self.postfilter_tapset = pf_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }

        if c == 1 {
            let (a, b) = self.old_band_e.split_at_mut(BANDS);
            b.copy_from_slice(a);
        }
        if !transient {
            self.old_log_e2 = self.old_log_e;
            self.old_log_e = self.old_band_e;
            for i in 0..2 * BANDS {
                self.background_log_e[i] =
                    (self.background_log_e[i] + m as f32 * 0.001).min(self.old_band_e[i]);
            }
        } else {
            for i in 0..2 * BANDS {
                self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
            }
        }
        for ch in 0..2 {
            for i in (0..start).chain(end..BANDS) {
                self.old_band_e[ch * BANDS + i] = 0.0;
                self.old_log_e[ch * BANDS + i] = -28.0;
                self.old_log_e2[ch * BANDS + i] = -28.0;
            }
        }
        self.rng = dec.range();
        self.deemphasis(pcm, n);
        self.loss_count = 0;
    }

    /// The inverse MDCTs of every channel, overlapped with the last frame (`compute_inv_mdcts`).
    /// The output lands in the last `n` samples of the history, the new overlap after them.
    fn inverse_mdcts(&mut self, freq: &[f32], short_blocks: bool, n: usize, lm: usize) {
        let window = &mode().window;
        let (blocks, bins_log) = if short_blocks {
            (1usize << lm, 0)
        } else {
            (1, lm)
        };
        let n2 = n / blocks;
        let imdct = Imdct::for_bins(bins_log);
        let mut x = vec![0f32; n + OVERLAP];
        for ch in 0..self.channels {
            x.fill(0.0);
            let spectrum = &freq[ch * n..(ch + 1) * n];
            for b in 0..blocks {
                imdct.backward(&spectrum[b..], blocks, &mut x[n2 * b..], window);
            }
            let mem = &mut self.decode_mem[ch];
            let syn = DECODE_BUFFER - n;
            for j in 0..OVERLAP {
                mem[syn + j] = x[j] + mem[DECODE_BUFFER + j];
            }
            mem[syn + OVERLAP..DECODE_BUFFER].copy_from_slice(&x[OVERLAP..n]);
            mem[DECODE_BUFFER..].copy_from_slice(&x[n..n + OVERLAP]);
        }
    }

    /// De-emphasis (section 4.3.7.2) of the last `n` samples of the history into `pcm`, taking
    /// every `downsample`-th sample.
    fn deemphasis(&mut self, pcm: &mut [f32], n: usize) {
        let cc = self.channels;
        for ch in 0..cc {
            let mut mem = self.preemph_mem[ch];
            let src = &self.decode_mem[ch][DECODE_BUFFER - n..DECODE_BUFFER];
            for (j, &v) in src.iter().enumerate() {
                let tmp = v + mem;
                mem = PREEMPHASIS * tmp;
                if j % self.downsample == 0 {
                    pcm[(j / self.downsample) * cc + ch] = tmp / 32768.0;
                }
            }
            self.preemph_mem[ch] = mem;
        }
    }

    /// Conceal a lost frame (`celt_decode_lost`; section 4.4 leaves the method open): repeat
    /// the last pitch period through the channel's prediction filter, or after five losses (or
    /// in a hybrid stream) shape noise with the band energies.
    fn decode_lost(&mut self, pcm: &mut [f32], n: usize, lm: usize) {
        let md = mode();
        let window = &md.window;
        let cc = self.channels;
        let out = DECODE_BUFFER - MAX_PERIOD; // where `out_mem` starts in the history
        if self.loss_count >= 5 || self.start != 0 {
            let eff_end = self.end.min(BANDS);
            let amp = if self.loss_count >= 5 {
                amplitudes(&self.background_log_e, self.start, self.end, cc)
            } else {
                let decay = if self.loss_count == 0 { 1.5 } else { 0.5 };
                for ch in 0..cc {
                    for i in self.start..self.end {
                        self.old_band_e[ch * BANDS + i] -= decay;
                    }
                }
                amplitudes(&self.old_band_e, self.start, self.end, cc)
            };
            let mut x = vec![0f32; cc * n];
            let mut seed = self.rng;
            for ch in 0..cc {
                for i in self.start..BANDS {
                    let lo = ch * n + (EBANDS[i] << lm);
                    let len = (EBANDS[i + 1] - EBANDS[i]) << lm;
                    for v in &mut x[lo..lo + len] {
                        seed = lcg(seed);
                        *v = ((seed as i32) >> 20) as f32;
                    }
                    renormalize(&mut x[lo..lo + len], 1.0);
                }
                x[ch * n + (EBANDS[self.end] << lm)..(ch + 1) * n].fill(0.0);
            }
            self.rng = seed;
            let mut freq = vec![0f32; cc * n];
            denormalize(&x, &mut freq, &amp, BANDS, n, lm, cc);
            let mut bound = EBANDS[eff_end] << lm;
            if self.downsample != 1 {
                bound = bound.min(n / self.downsample);
            }
            for ch in 0..cc {
                let f = &mut freq[ch * n..(ch + 1) * n];
                f[..EBANDS[self.start] << lm].fill(0.0);
                f[bound..].fill(0.0);
            }
            self.inverse_mdcts(&freq, false, n, lm);
        } else {
            let mut fade = 1.0f32;
            let pitch_index = if self.loss_count == 0 {
                let chans: Vec<&[f32]> = self.decode_mem[..cc]
                    .iter()
                    .map(|m| &m[..DECODE_BUFFER])
                    .collect();
                let lp = plc::pitch_downsample(&chans, DECODE_BUFFER);
                let poffset = 720;
                let found = plc::pitch_search(
                    &lp[poffset >> 1..],
                    &lp,
                    DECODE_BUFFER - poffset,
                    poffset - 100,
                );
                let p = poffset - found;
                self.last_pitch_index = p;
                p
            } else {
                fade = 0.8;
                self.last_pitch_index
            };
            let len = n + OVERLAP;
            for ch in 0..cc {
                let mem = &mut self.decode_mem[ch];
                let mut exc: Vec<f32> = mem[out..out + MAX_PERIOD].to_vec();
                if self.loss_count == 0 {
                    let mut ac = [0f32; LPC_ORDER + 1];
                    plc::autocorr(&exc, &mut ac, Some(window), LPC_ORDER);
                    ac[0] *= 1.0001;
                    for (i, a) in ac.iter_mut().enumerate().skip(1) {
                        *a -= *a * (0.008 * i as f32) * (0.008 * i as f32);
                    }
                    plc::lpc(&mut self.lpc[ch], &ac);
                }
                let lpc = self.lpc[ch];
                let mut fmem = [0f32; LPC_ORDER];
                for (i, f) in fmem.iter_mut().enumerate() {
                    *f = mem[out + MAX_PERIOD - 1 - i];
                }
                plc::fir(&mut exc, &lpc, &mut fmem);
                // Whether the waveform is decaying, and how fast.
                let decay = {
                    let period = pitch_index.min(MAX_PERIOD / 2);
                    let (mut e1, mut e2) = (1f32, 1f32);
                    for i in 0..period {
                        let a = exc[MAX_PERIOD - period + i];
                        let b = exc[MAX_PERIOD - 2 * period + i];
                        e1 += a * a;
                        e2 += b * b;
                    }
                    (e1.min(e2) / e2).sqrt()
                };
                let mut decay = decay;
                let mut offset = MAX_PERIOD - pitch_index;
                let mut e = vec![0f32; MAX_PERIOD + 2 * OVERLAP];
                let mut s1 = 0f32;
                for i in 0..len + OVERLAP {
                    if offset + i >= MAX_PERIOD {
                        offset -= pitch_index;
                        decay *= decay;
                    }
                    e[i] = decay * exc[offset + i];
                    let t = mem[out + offset + i];
                    s1 += t * t;
                }
                for (i, f) in fmem.iter_mut().enumerate() {
                    *f = mem[out + MAX_PERIOD - 1 - i];
                }
                for v in &mut e[..len + OVERLAP] {
                    *v *= fade;
                }
                plc::iir(&mut e[..len + OVERLAP], &lpc, &mut fmem);
                let s2: f32 = e[..len + OVERLAP].iter().map(|v| v * v).sum();
                // Catch an explosion of the synthesis (and NaN).
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(s1 > 0.2 * s2) {
                    e[..len + OVERLAP].fill(0.0);
                } else if s1 < s2 {
                    let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
                    for v in &mut e[..len + OVERLAP] {
                        *v *= ratio;
                    }
                }
                // The post-filter of the last frame's overlap, then make room.
                comb_filter(
                    mem,
                    out + MAX_PERIOD,
                    OVERLAP,
                    self.postfilter_period,
                    self.postfilter_period,
                    self.postfilter_gain,
                    self.postfilter_gain,
                    self.postfilter_tapset,
                    self.postfilter_tapset,
                    &[],
                );
                mem.copy_within(out + n..out + MAX_PERIOD + OVERLAP, out);
                // TDAC so that the concealed audio blends with the frames around it.
                for i in 0..OVERLAP / 2 {
                    let tmp =
                        window[i] * e[n + OVERLAP - 1 - i] + window[OVERLAP - i - 1] * e[n + i];
                    mem[out + MAX_PERIOD + i] = window[OVERLAP - i - 1] * tmp;
                    mem[out + MAX_PERIOD + OVERLAP - i - 1] = window[i] * tmp;
                }
                mem[out + MAX_PERIOD - n..out + MAX_PERIOD].copy_from_slice(&e[..n]);
                // The pre-filter of the overlap for the next frame, whose post-filter undoes it:
                // a plain FIR of what is there now.
                let src = if self.postfilter_gain == 0.0 {
                    Vec::new()
                } else {
                    mem.clone()
                };
                let (t, g) = (self.postfilter_period as usize, -self.postfilter_gain);
                let taps = COMB_GAINS[self.postfilter_tapset];
                for i in 0..if src.is_empty() { 0 } else { OVERLAP } {
                    let p = out + MAX_PERIOD + i;
                    mem[p] = src[p]
                        + g * taps[0] * src[p - t]
                        + g * taps[1] * src[p - t - 1]
                        + g * taps[1] * src[p - t + 1]
                        + g * taps[2] * src[p - t - 2]
                        + g * taps[2] * src[p - t + 2];
                }
            }
        }
        self.deemphasis(pcm, n);
        self.loss_count += 1;
    }
}

/// The time-frequency changes of each band (section 4.3.4.5, `tf_decode`).
fn tf_decode(
    start: usize,
    end: usize,
    transient: bool,
    tf_res: &mut [i32; BANDS],
    lm: usize,
    dec: &mut RangeDecoder,
) {
    let mut budget = dec.len() as u32 * 8;
    let mut tell = dec.tell();
    let mut logp = if transient { 2 } else { 4 };
    let select_rsv = lm > 0 && tell + logp < budget;
    budget -= u32::from(select_rsv);
    let (mut changed, mut curr) = (0usize, 0usize);
    for r in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            curr ^= usize::from(dec.bit_logp(logp));
            tell = dec.tell();
            changed |= curr;
        }
        *r = curr as i32;
        logp = if transient { 4 } else { 5 };
    }
    let t = 4 * usize::from(transient);
    let mut select = 0;
    if select_rsv && TF_SELECT[lm][t + changed] != TF_SELECT[lm][t + 2 + changed] {
        select = usize::from(dec.bit_logp(1));
    }
    for r in tf_res.iter_mut().take(end).skip(start) {
        *r = i32::from(TF_SELECT[lm][t + 2 * select + *r as usize]);
    }
}
