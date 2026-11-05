//! The Opus decoder (RFC 6716 section 4): a packet's frames go to SILK, to CELT or to both, with
//! the redundant CELT frames and the cross-fades of section 4.5 at the changes of mode. This
//! follows the reference's `src/opus_decoder.c` (`opus_decode_frame`, `opus_decode_native`).

use crate::celt::mode::mode as celt_mode;
use crate::celt::CeltDecoder;
use crate::packet::{self, Bandwidth, Malformed, Mode, Toc};
use crate::range::RangeDecoder;
use crate::silk::{Arrival, SilkControl, SilkDecoder};

/// Why a packet could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The packet breaks the rules of section 3.4, or SILK found it corrupt.
    InvalidPacket(&'static str),
    /// The output holds fewer samples than the packet decodes to.
    BufferTooSmall,
}

impl From<Malformed> for Error {
    fn from(m: Malformed) -> Error {
        Error::InvalidPacket(m.0)
    }
}

pub struct Decoder {
    rate: u32,
    channels: usize,
    celt: CeltDecoder,
    silk: SilkDecoder,
    stream_channels: usize,
    bandwidth: Bandwidth,
    mode: Mode,
    /// The mode of the last frame decoded, `None` before the first.
    prev_mode: Option<Mode>,
    /// The duration of the frames of the last packet, at the output rate.
    frame_size: usize,
    prev_redundancy: bool,
    range_final: u32,
}

/// The cross-fade of section 4.5.2 (`smooth_fade`): from `in1` to `in2` with the square of the
/// CELT window, over `overlap` samples of each channel.
fn smooth_fade(
    in1: &[f32],
    in2: &[f32],
    out: &mut [f32],
    overlap: usize,
    channels: usize,
    rate: u32,
) {
    let window = &celt_mode().window;
    let inc = (48000 / rate) as usize;
    for c in 0..channels {
        for i in 0..overlap {
            let w = window[i * inc] * window[i * inc];
            let k = i * channels + c;
            out[k] = w * in2[k] + (1.0 - w) * in1[k];
        }
    }
}

impl Decoder {
    /// A decoder for `channels` (1 or 2) channels at `rate` (48, 24, 16, 12 or 8 kHz).
    pub fn new(rate: u32, channels: usize) -> Option<Decoder> {
        if !matches!(rate, 48000 | 24000 | 16000 | 12000 | 8000) || !(1..=2).contains(&channels) {
            return None;
        }
        Some(Decoder {
            rate,
            channels,
            celt: CeltDecoder::new(rate, channels),
            silk: SilkDecoder::new(),
            stream_channels: channels,
            bandwidth: Bandwidth::Full,
            mode: Mode::Celt,
            prev_mode: None,
            frame_size: rate as usize / 400,
            prev_redundancy: false,
            range_final: 0,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Forget the past (`OPUS_RESET_STATE`).
    pub fn reset(&mut self) {
        self.celt.reset();
        self.silk.reset();
        self.stream_channels = self.channels;
        self.bandwidth = Bandwidth::Full;
        self.mode = Mode::Celt;
        self.prev_mode = None;
        self.frame_size = self.rate as usize / 400;
        self.prev_redundancy = false;
        self.range_final = 0;
    }

    /// Whether to apply the phase inversion of CELT's intensity stereo (RFC 8251 section 10
    /// lets a decoder leave it out, which helps a later mix to mono).
    pub fn set_phase_inversion(&mut self, on: bool) {
        self.celt.phase_inversion = on;
    }

    /// The range coder's final state for the last packet (XOR the redundant frame's), which
    /// an encoder reports so that a decoder can be checked.
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    /// Decode a packet into `pcm` (interleaved, [-1, 1]), which holds `pcm.len() / channels`
    /// samples per channel. `None` conceals a lost packet of that many samples. With `fec`, the
    /// packet's forward error correction data stands in for the packet before it. Returns the
    /// samples per channel decoded.
    pub fn decode(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [f32],
        fec: bool,
    ) -> Result<usize, Error> {
        let frame_size = pcm.len() / self.channels;
        let data = match packet {
            Some(d) if !d.is_empty() => d,
            _ => return self.decode_frame(None, pcm, frame_size, false),
        };
        let p = packet::parse(data)?;
        let toc = p.toc;
        self.mode = toc.mode;
        self.bandwidth = toc.bandwidth;
        self.frame_size = self.samples_at_rate(&toc);
        self.stream_channels = if toc.stereo { 2 } else { 1 };
        if p.frames.len() * self.frame_size > frame_size {
            return Err(Error::BufferTooSmall);
        }
        let mut done = 0;
        for f in &p.frames {
            let n = self.decode_frame(
                Some(f),
                &mut pcm[done * self.channels..],
                frame_size - done,
                fec,
            )?;
            done += n;
        }
        Ok(done)
    }

    /// `decode`, with the samples as 16-bit integers (rounded to nearest, saturated).
    pub fn decode_i16(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [i16],
        fec: bool,
    ) -> Result<usize, Error> {
        let mut out = vec![0f32; pcm.len()];
        let n = self.decode(packet, &mut out, fec)?;
        for (o, &v) in pcm.iter_mut().zip(&out[..n * self.channels]) {
            *o = (v * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16;
        }
        Ok(n)
    }

    fn samples_at_rate(&self, toc: &Toc) -> usize {
        toc.frame_samples * self.rate as usize / 48000
    }

    /// One frame (`opus_decode_frame`).
    fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> Result<usize, Error> {
        let ch = self.channels;
        let f20 = self.rate as usize / 50;
        let f10 = f20 >> 1;
        let f5 = f10 >> 1;
        let f2_5 = f5 >> 1;
        if frame_size < f2_5 {
            return Err(Error::BufferTooSmall);
        }
        let mut frame_size = frame_size;
        // A payload of 0 or 1 bytes is a lost frame (DTX or loss).
        let data = match data {
            Some(d) if d.len() > 1 => Some(d),
            _ => {
                frame_size = frame_size.min(self.frame_size);
                None
            }
        };
        let (audiosize, mode) = match data {
            Some(_) => (self.frame_size, self.mode),
            None => match self.prev_mode {
                None => {
                    // Nothing was ever decoded: silence.
                    pcm[..frame_size * ch].fill(0.0);
                    return Ok(frame_size);
                }
                Some(m) => (frame_size, m),
            },
        };
        // Conceal more than 20 ms of CELT or hybrid in steps of 20 ms.
        if data.is_none() && frame_size > f20 && mode != Mode::Silk {
            let mut done = 0;
            while done < frame_size {
                let n = self.decode_frame(None, &mut pcm[done * ch..], f20, false)?;
                done += n;
            }
            return Ok(frame_size);
        }
        let mut len = data.map_or(0, |d| d.len());
        let mut dec = data.map(RangeDecoder::new);

        let mut transition = false;
        let mut pcm_transition = vec![0f32; f5 * ch];
        if data.is_some() {
            if let Some(prev) = self.prev_mode {
                let to_celt = mode == Mode::Celt && prev != Mode::Celt && !self.prev_redundancy;
                let from_celt = mode != Mode::Celt && prev == Mode::Celt;
                if to_celt || from_celt {
                    transition = true;
                    if mode == Mode::Celt {
                        // The end of the last mode, concealed, to fade from.
                        let n = f5.min(audiosize);
                        self.decode_frame(None, &mut pcm_transition[..n * ch], n, false)?;
                    }
                }
            }
        }
        if audiosize > frame_size {
            return Err(Error::BufferTooSmall);
        }
        frame_size = audiosize;

        // SILK.
        let mut pcm_silk = vec![0i16; f10.max(frame_size) * ch];
        if mode != Mode::Celt {
            if self.prev_mode == Some(Mode::Celt) {
                self.silk.reset();
            }
            let internal_rate = match (mode, self.bandwidth) {
                (Mode::Silk, Bandwidth::Narrow) => 8000,
                (Mode::Silk, Bandwidth::Medium) => 12000,
                _ => 16000,
            };
            let ctl = SilkControl {
                api_channels: ch,
                internal_channels: self.stream_channels,
                api_rate: self.rate,
                internal_rate,
                // SILK's concealment makes no less than 10 ms.
                payload_ms: (1000 * audiosize / self.rate as usize).max(10),
            };
            let arrival = if data.is_none() {
                Arrival::Lost
            } else if fec {
                Arrival::Redundant
            } else {
                Arrival::Normal
            };
            let mut decoded = 0;
            while decoded < frame_size {
                let first = decoded == 0;
                match self.silk.decode(
                    &ctl,
                    arrival,
                    first,
                    dec.as_mut(),
                    &mut pcm_silk[decoded * ch..],
                ) {
                    Ok(n) => decoded += n,
                    Err(_) if arrival != Arrival::Normal => {
                        // Concealment that fails is silence, not an error.
                        pcm_silk[decoded * ch..frame_size * ch].fill(0);
                        decoded = frame_size;
                    }
                    Err(e) => return Err(Error::InvalidPacket(e.0)),
                }
            }
        }

        // A redundant CELT frame at the end of a SILK or hybrid frame (section 4.5.1).
        let (mut redundancy, mut celt_to_silk, mut redundancy_bytes) = (false, false, 0usize);
        if let Some(d) = dec.as_mut() {
            if !fec
                && mode != Mode::Celt
                && d.tell() + 17 + 20 * u32::from(self.mode == Mode::Hybrid) <= 8 * len as u32
            {
                redundancy = mode != Mode::Hybrid || d.bit_logp(12);
                if redundancy {
                    celt_to_silk = d.bit_logp(1);
                    redundancy_bytes = if mode == Mode::Hybrid {
                        d.uint(256) as usize + 2
                    } else {
                        len - ((d.tell() as usize + 7) >> 3)
                    };
                    // A sanity check that fails only for a malformed packet.
                    if redundancy_bytes > len || (len - redundancy_bytes) * 8 < d.tell() as usize {
                        len = 0;
                        redundancy_bytes = 0;
                        redundancy = false;
                    } else {
                        len -= redundancy_bytes;
                    }
                    d.shrink(redundancy_bytes);
                }
            }
        }
        let start_band = if mode != Mode::Celt { 17 } else { 0 };
        let end_band = match self.bandwidth {
            Bandwidth::Narrow => 13,
            Bandwidth::Medium | Bandwidth::Wide => 17,
            Bandwidth::SuperWide => 19,
            Bandwidth::Full => 21,
        };
        self.celt.set_end_band(end_band);
        self.celt.set_stream_channels(self.stream_channels);
        if redundancy {
            transition = false;
        }
        if transition && mode != Mode::Celt {
            let n = f5.min(audiosize);
            self.decode_frame(None, &mut pcm_transition[..n * ch], n, false)?;
        }
        let redundant = data.map(|d| &d[len..len + redundancy_bytes]);
        let mut redundant_audio = vec![0f32; f5 * ch];
        let mut redundant_rng = 0;
        if redundancy && celt_to_silk {
            self.celt.set_start_band(0);
            let mut rd = RangeDecoder::new(redundant.unwrap_or(&[]));
            self.celt.decode(Some(&mut rd), &mut redundant_audio, f5);
            redundant_rng = self.celt.final_range();
        }
        // After the concealment above.
        self.celt.set_start_band(start_band);
        if mode != Mode::Silk {
            let celt_frame = f20.min(frame_size);
            // Discard the CELT state of another mode.
            if Some(mode) != self.prev_mode && self.prev_mode.is_some() && !self.prev_redundancy {
                self.celt.reset();
            }
            let celt_dec = if fec || len <= 1 { None } else { dec.as_mut() };
            self.celt
                .decode(celt_dec, &mut pcm[..celt_frame * ch], celt_frame);
        } else {
            pcm[..frame_size * ch].fill(0.0);
            // From hybrid to SILK, let the CELT MDCT fade out by decoding a silent frame.
            if self.prev_mode == Some(Mode::Hybrid)
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                self.celt.set_start_band(0);
                let silence = [0xFFu8, 0xFF];
                let mut rd = RangeDecoder::new(&silence);
                self.celt.decode(Some(&mut rd), &mut pcm[..f2_5 * ch], f2_5);
            }
        }
        if mode != Mode::Celt {
            for (o, &s) in pcm[..frame_size * ch].iter_mut().zip(&pcm_silk) {
                *o += f32::from(s) / 32768.0;
            }
        }
        // A redundant frame from SILK to CELT ends this frame.
        if redundancy && !celt_to_silk {
            self.celt.reset();
            self.celt.set_start_band(0);
            let mut rd = RangeDecoder::new(redundant.unwrap_or(&[]));
            self.celt.decode(Some(&mut rd), &mut redundant_audio, f5);
            redundant_rng = self.celt.final_range();
            let at = ch * (frame_size - f2_5);
            let a = pcm[at..at + f2_5 * ch].to_vec();
            smooth_fade(
                &a,
                &redundant_audio[ch * f2_5..],
                &mut pcm[at..],
                f2_5,
                ch,
                self.rate,
            );
        }
        // One from CELT to SILK starts it.
        if redundancy && celt_to_silk {
            pcm[..f2_5 * ch].copy_from_slice(&redundant_audio[..f2_5 * ch]);
            let b = pcm[ch * f2_5..ch * 2 * f2_5].to_vec();
            smooth_fade(
                &redundant_audio[ch * f2_5..],
                &b,
                &mut pcm[ch * f2_5..],
                f2_5,
                ch,
                self.rate,
            );
        }
        if transition {
            if audiosize >= f5 {
                pcm[..ch * f2_5].copy_from_slice(&pcm_transition[..ch * f2_5]);
                let b = pcm[ch * f2_5..ch * 2 * f2_5].to_vec();
                smooth_fade(
                    &pcm_transition[ch * f2_5..],
                    &b,
                    &mut pcm[ch * f2_5..],
                    f2_5,
                    ch,
                    self.rate,
                );
            } else {
                // Not enough time for a clean transition; fade anyway.
                let b = pcm[..ch * f2_5].to_vec();
                smooth_fade(&pcm_transition, &b, pcm, f2_5, ch, self.rate);
            }
        }
        self.range_final = match dec {
            Some(d) if len > 1 => d.range() ^ redundant_rng,
            _ => 0,
        };
        self.prev_mode = Some(mode);
        self.prev_redundancy = redundancy && !celt_to_silk;
        Ok(audiosize)
    }
}
