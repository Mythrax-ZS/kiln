//! The SILK layer (RFC 6716 section 4.2): a linear-prediction codec of 10 or 20 ms frames at
//! 8, 12 or 16 kHz. SILK's decoder is defined bit-exactly by integer arithmetic; this module is
//! written from the prose and from the normative reference implementation (appendix A,
//! `silk/dec_API.c` and the files the submodules name; Simplified BSD licence, copyright Skype
//! Limited and the Xiph.Org Foundation), with the corrections of RFC 8251 sections 3 and 5 to 7.

mod bitstream;
mod channel;
mod fixed;
mod params;
mod plc;
mod resampler;
mod synth;
mod tables;

use crate::range::RangeDecoder;
use bitstream::{decode_indices, decode_mid_only, decode_pulses, decode_stereo_pred};
use channel::{Channel, Conditioning, MAX_FRAME_LENGTH, TYPE_NO_VOICE_ACTIVITY};
use fixed::{rshift_round, sat16, smlawb, smulbb};

/// What a frame needs to know from the packet (`silk_DecControlStruct`).
#[derive(Clone, Copy, Debug)]
pub struct SilkControl {
    pub api_channels: usize,
    pub internal_channels: usize,
    pub api_rate: u32,
    pub internal_rate: u32,
    pub payload_ms: usize,
}

/// How a frame arrived (`lostFlag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    Normal,
    Lost,
    /// Decode the low bit-rate redundancy (forward error correction) of the next packet.
    Redundant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SilkError(pub &'static str);

/// The stereo state (`stereo_dec_state`).
#[derive(Clone, Copy, Debug, Default)]
struct Stereo {
    pred_prev_q13: [i16; 2],
    s_mid: [i16; 2],
    s_side: [i16; 2],
}

pub struct SilkDecoder {
    channels: [Channel; 2],
    stereo: Stereo,
    n_channels_api: usize,
    n_channels_internal: usize,
    prev_decode_only_middle: bool,
}

/// Convert mid and side into left and right (`silk_stereo_MS_to_LR`). `x1` and `x2` hold two
/// samples of history then the frame.
fn stereo_ms_to_lr(
    st: &mut Stereo,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: [i32; 2],
    fs_khz: i32,
    len: usize,
) {
    x1[..2].copy_from_slice(&st.s_mid);
    x2[..2].copy_from_slice(&st.s_side);
    st.s_mid.copy_from_slice(&x1[len..len + 2]);
    st.s_side.copy_from_slice(&x2[len..len + 2]);
    // Interpolate the predictors over 8 ms.
    let mut pred0 = i32::from(st.pred_prev_q13[0]);
    let mut pred1 = i32::from(st.pred_prev_q13[1]);
    let interp = (8 * fs_khz) as usize;
    let denom_q16 = (1 << 16) / (8 * fs_khz);
    let delta0 = rshift_round(smulbb(pred_q13[0] - pred0, denom_q16), 16);
    let delta1 = rshift_round(smulbb(pred_q13[1] - pred1, denom_q16), 16);
    for n in 0..len {
        if n < interp {
            pred0 += delta0;
            pred1 += delta1;
        } else if n == interp {
            pred0 = pred_q13[0];
            pred1 = pred_q13[1];
        }
        let mut sum =
            ((i32::from(x1[n]) + i32::from(x1[n + 2])) + (i32::from(x1[n + 1]) << 1)) << 9;
        sum = smlawb(i32::from(x2[n + 1]) << 8, sum, pred0);
        sum = smlawb(sum, i32::from(x1[n + 1]) << 11, pred1);
        x2[n + 1] = sat16(rshift_round(sum, 8)) as i16;
    }
    st.pred_prev_q13 = [pred_q13[0] as i16, pred_q13[1] as i16];
    for n in 0..len {
        let (m, s) = (i32::from(x1[n + 1]), i32::from(x2[n + 1]));
        x1[n + 1] = sat16(m + s) as i16;
        x2[n + 1] = sat16(m - s) as i16;
    }
}

impl SilkDecoder {
    pub fn new() -> SilkDecoder {
        SilkDecoder {
            channels: [Channel::new(), Channel::new()],
            stereo: Stereo::default(),
            n_channels_api: 0,
            n_channels_internal: 0,
            prev_decode_only_middle: false,
        }
    }

    /// `silk_InitDecoder`, with the stereo reset of RFC 8251 section 3.
    pub fn reset(&mut self) {
        self.channels = [Channel::new(), Channel::new()];
        self.stereo = Stereo::default();
        self.prev_decode_only_middle = false;
    }

    /// Decode (or conceal) one SILK frame of a packet into `out`, interleaved at the output
    /// rate (`silk_Decode`); `first_frame` starts a new packet. Returns the samples per channel.
    pub fn decode(
        &mut self,
        ctl: &SilkControl,
        arrival: Arrival,
        first_frame: bool,
        mut dec: Option<&mut RangeDecoder>,
        out: &mut [i16],
    ) -> Result<usize, SilkError> {
        let internal = ctl.internal_channels;
        if first_frame {
            for c in &mut self.channels[..internal] {
                c.n_frames_decoded = 0;
            }
        }
        // From mono to stereo: start the second channel afresh.
        if internal > self.n_channels_internal {
            self.channels[1] = Channel::new();
        }
        let stereo_to_mono = internal == 1
            && self.n_channels_internal == 2
            && ctl.internal_rate as i32 == 1000 * self.channels[0].fs_khz;
        if self.channels[0].n_frames_decoded == 0 {
            for c in &mut self.channels[..internal] {
                let (frames, subfr) = match ctl.payload_ms {
                    0 | 10 => (1, 2),
                    20 => (1, 4),
                    40 => (2, 4),
                    60 => (3, 4),
                    _ => return Err(SilkError("a SILK payload that is not 10, 20, 40 or 60 ms")),
                };
                c.n_frames_per_packet = frames;
                c.nb_subfr = subfr;
                let fs_khz = (ctl.internal_rate as i32 >> 10) + 1;
                if !matches!(fs_khz, 8 | 12 | 16) {
                    return Err(SilkError("a SILK internal rate other than 8, 12 or 16 kHz"));
                }
                c.set_fs(fs_khz, ctl.api_rate as i32);
            }
        }
        if ctl.api_channels == 2
            && internal == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.stereo.pred_prev_q13 = [0; 2];
            self.stereo.s_side = [0; 2];
            self.channels[1].resampler = self.channels[0].resampler;
        }
        self.n_channels_api = ctl.api_channels;
        self.n_channels_internal = internal;
        if !(8000..=48000).contains(&ctl.api_rate) {
            return Err(SilkError("an output rate outside 8 to 48 kHz"));
        }

        if arrival != Arrival::Lost && self.channels[0].n_frames_decoded == 0 {
            let d = dec
                .as_deref_mut()
                .ok_or(SilkError("a SILK frame without data"))?;
            // The voice activity and redundancy flags of every frame of the packet.
            for c in &mut self.channels[..internal] {
                for i in 0..c.n_frames_per_packet {
                    c.vad_flags[i] = d.bit_logp(1);
                }
                c.lbrr_flag = d.bit_logp(1);
            }
            for c in &mut self.channels[..internal] {
                c.lbrr_flags = [false; 3];
                if c.lbrr_flag {
                    if c.n_frames_per_packet == 1 {
                        c.lbrr_flags[0] = true;
                    } else {
                        let table: &[u8] = if c.n_frames_per_packet == 2 {
                            &tables::LBRR_FLAGS_2_ICDF
                        } else {
                            &tables::LBRR_FLAGS_3_ICDF
                        };
                        let symbol = d.icdf(table, 8) + 1;
                        for i in 0..c.n_frames_per_packet {
                            c.lbrr_flags[i] = (symbol >> i) & 1 != 0;
                        }
                    }
                }
            }
            if arrival == Arrival::Normal {
                // Skip the redundancy data.
                for i in 0..self.channels[0].n_frames_per_packet {
                    for n in 0..internal {
                        if !self.channels[n].lbrr_flags[i] {
                            continue;
                        }
                        if internal == 2 && n == 0 {
                            decode_stereo_pred(d);
                            if !self.channels[1].lbrr_flags[i] {
                                decode_mid_only(d);
                            }
                        }
                        let cond = if i > 0 && self.channels[n].lbrr_flags[i - 1] {
                            Conditioning::Conditional
                        } else {
                            Conditioning::Independent
                        };
                        let c = &mut self.channels[n];
                        decode_indices(c, d, i, true, cond);
                        let mut pulses = [0i32; MAX_FRAME_LENGTH];
                        decode_pulses(
                            d,
                            &mut pulses,
                            i32::from(c.indices.signal_type),
                            i32::from(c.indices.quant_offset_type),
                            c.frame_length,
                        );
                    }
                }
            }
        }

        // The stereo prediction of this frame.
        let mut ms_pred_q13 = [0i32; 2];
        let mut decode_only_middle = false;
        if internal == 2 {
            let fd = self.channels[0].n_frames_decoded;
            let coded = arrival == Arrival::Normal
                || (arrival == Arrival::Redundant && self.channels[0].lbrr_flags[fd]);
            if coded {
                let d = dec
                    .as_deref_mut()
                    .ok_or(SilkError("a SILK frame without data"))?;
                ms_pred_q13 = decode_stereo_pred(d);
                let side_absent = (arrival == Arrival::Normal && !self.channels[1].vad_flags[fd])
                    || (arrival == Arrival::Redundant && !self.channels[1].lbrr_flags[fd]);
                decode_only_middle = side_absent && decode_mid_only(d);
            } else {
                ms_pred_q13 = [
                    i32::from(self.stereo.pred_prev_q13[0]),
                    i32::from(self.stereo.pred_prev_q13[1]),
                ];
            }
        }
        // The side channel's first frame after mid-only frames starts from silence.
        if internal == 2 && !decode_only_middle && self.prev_decode_only_middle {
            let c = &mut self.channels[1];
            c.out_buf.fill(0);
            c.slpc_q14_buf.fill(0);
            c.lag_prev = 100;
            c.last_gain_index = 10;
            c.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            c.first_frame_after_reset = true;
        }
        let has_side = if arrival == Arrival::Normal {
            !decode_only_middle
        } else {
            !self.prev_decode_only_middle
                || (internal == 2
                    && arrival == Arrival::Redundant
                    && self.channels[1].lbrr_flags[self.channels[1].n_frames_decoded])
        };

        // Each channel's frame, after two samples of history.
        let mut tmp = [[0i16; MAX_FRAME_LENGTH + 2]; 2];
        let mut n_out_dec = 0;
        for (n, buf) in tmp.iter_mut().enumerate().take(internal) {
            if n == 0 || has_side {
                let frame_index = self.channels[0].n_frames_decoded as i32 - n as i32;
                let cond = if frame_index <= 0 {
                    Conditioning::Independent
                } else if arrival == Arrival::Redundant {
                    if self.channels[n].lbrr_flags[frame_index as usize - 1] {
                        Conditioning::Conditional
                    } else {
                        Conditioning::Independent
                    }
                } else if n > 0 && self.prev_decode_only_middle {
                    // The side skipped a frame: its LTP state is well defined anyway.
                    Conditioning::IndependentNoLtpScaling
                } else {
                    Conditioning::Conditional
                };
                n_out_dec =
                    self.channels[n].decode_frame(dec.as_deref_mut(), &mut buf[2..], arrival, cond);
            } else {
                buf[2..2 + n_out_dec].fill(0);
            }
            self.channels[n].n_frames_decoded += 1;
        }

        let fs_khz = self.channels[0].fs_khz;
        if ctl.api_channels == 2 && internal == 2 {
            let (a, b) = tmp.split_at_mut(1);
            stereo_ms_to_lr(
                &mut self.stereo,
                &mut a[0],
                &mut b[0],
                ms_pred_q13,
                fs_khz,
                n_out_dec,
            );
        } else {
            // The two samples of delay a stereo stream would have.
            tmp[0][..2].copy_from_slice(&self.stereo.s_mid);
            self.stereo
                .s_mid
                .copy_from_slice(&tmp[0][n_out_dec..n_out_dec + 2]);
        }

        let n_out = n_out_dec * ctl.api_rate as usize / (fs_khz as usize * 1000);
        let api = ctl.api_channels;
        let mut resampled = vec![0i16; n_out + 8];
        for n in 0..api.min(internal) {
            let r = &mut self.channels[n].resampler;
            r.process(&mut resampled, &tmp[n][1..1 + n_out_dec]);
            for i in 0..n_out {
                out[n + api * i] = resampled[i];
            }
        }
        if api == 2 && internal == 1 {
            if stereo_to_mono {
                // The right channel's own resampler, in case it was in use until now.
                self.channels[1]
                    .resampler
                    .process(&mut resampled, &tmp[0][1..1 + n_out_dec]);
                for i in 0..n_out {
                    out[1 + 2 * i] = resampled[i];
                }
            } else {
                for i in 0..n_out {
                    out[1 + 2 * i] = out[2 * i];
                }
            }
        }
        if arrival == Arrival::Lost {
            // No gain clamping after a loss, so that the energy does not bounce back.
            for c in &mut self.channels[..self.n_channels_internal] {
                c.last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }
        Ok(n_out)
    }
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}
