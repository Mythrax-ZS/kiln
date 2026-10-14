//! The voices of the audio renderer: what an update says about one, the bookkeeping the program
//! reads back (wave buffers consumed, samples played) and the samples it makes in one frame
//! (docs/ASSUMPTIONS.md A-075, A-125).
//!
//! A voice plays its queue of up to four wave buffers, each a run of samples in the program's
//! memory, at its sample rate times its pitch, and the renderer takes one frame (`sample_count`
//! output samples) of it with each update. The samples between two source samples are linear
//! interpolations; filters (the two biquads of a voice) are not applied.

use super::adpcm::{self, Coefficients, History};
use super::{f32_at, u32_at, u64_at};

/// The wave buffers of a voice: a ring of four.
pub const WAVE_BUFFERS: usize = 4;
const WAVE_BUFFER_SIZE: usize = 0x38;
const WAVE_BUFFERS_AT: usize = 0x60;

/// Channels a voice can have, and mix buffers one can send to.
pub const MAX_CHANNELS: usize = 6;
pub const MAX_MIX_BUFFERS: usize = 24;

/// Voice play state: started (`AudioRendererVoicePlayState_Started`).
pub const VOICE_STARTED: u8 = 0;

/// Reads `buf.len()` bytes of the program's memory at an address; false if they are not there.
pub type ReadMemory<'a> = &'a dyn Fn(u64, &mut [u8]) -> bool;

/// How the samples of a wave buffer are stored (`PcmFormat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// A format that is not decoded here: the voice is silent.
    Unsupported,
    Pcm8,
    Pcm16,
    Pcm32,
    Float,
    Adpcm,
}

impl Format {
    fn from_code(code: u8) -> Format {
        match code {
            1 => Format::Pcm8,
            2 => Format::Pcm16,
            4 => Format::Pcm32,
            5 => Format::Float,
            6 => Format::Adpcm,
            _ => Format::Unsupported,
        }
    }

    /// Bytes of one sample of one channel, for the formats that are plain samples.
    fn bytes(self) -> usize {
        match self {
            Format::Pcm8 => 1,
            Format::Pcm16 => 2,
            Format::Pcm32 | Format::Float => 4,
            _ => 0,
        }
    }
}

/// One wave buffer as the program describes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Wave {
    pub address: u64,
    pub size: u64,
    /// The samples of the buffer that play: `start` up to (not including) `end`, counted per
    /// channel from the start of the data.
    pub start: i32,
    pub end: i32,
    pub looping: bool,
    /// The decoder state at `start` (three 16-bit values), for ADPCM.
    pub context: u64,
    pub context_size: u64,
}

impl Wave {
    fn parse(info: &[u8], at: usize) -> Wave {
        Wave {
            address: u64_at(info, at),
            size: u64_at(info, at + 8),
            start: u32_at(info, at + 0x10) as i32,
            end: u32_at(info, at + 0x14) as i32,
            looping: info[at + 0x18] != 0,
            context: u64_at(info, at + 0x20),
            context_size: u64_at(info, at + 0x28),
        }
    }

    /// Samples (per channel) that play.
    pub fn length(&self) -> u64 {
        (i64::from(self.end) - i64::from(self.start)).max(0) as u64
    }
}

/// What an update says about a voice.
#[derive(Clone, Debug)]
pub struct VoiceIn {
    pub is_new: bool,
    pub used: bool,
    pub started: bool,
    pub format: Format,
    pub sample_rate: u32,
    pub pitch: f32,
    pub volume: f32,
    pub channels: usize,
    /// How many wave buffers are queued and where the ring starts.
    pub queued: usize,
    pub head: usize,
    /// The mix the voice plays into (the final mix is 0) and its channel resources.
    pub dest_mix: u32,
    pub channel_ids: [u32; MAX_CHANNELS],
    /// The ADPCM coefficients: where they are and how many bytes.
    pub coefficients_at: u64,
    pub coefficients_size: u64,
    pub waves: [Wave; WAVE_BUFFERS],
}

impl VoiceIn {
    /// `info` is the entry of the input (0x170 bytes).
    pub fn parse(info: &[u8]) -> VoiceIn {
        let mut channel_ids = [0u32; MAX_CHANNELS];
        for (i, id) in channel_ids.iter_mut().enumerate() {
            *id = u32_at(info, 0x140 + i * 4);
        }
        let mut waves = [Wave::default(); WAVE_BUFFERS];
        for (i, wave) in waves.iter_mut().enumerate() {
            *wave = Wave::parse(info, WAVE_BUFFERS_AT + i * WAVE_BUFFER_SIZE);
        }
        VoiceIn {
            is_new: info[0x08] != 0,
            used: info[0x09] != 0,
            started: info[0x0A] == VOICE_STARTED,
            format: Format::from_code(info[0x0B]),
            sample_rate: u32_at(info, 0x0C),
            pitch: f32_at(info, 0x1C),
            volume: f32_at(info, 0x20),
            channels: (u32_at(info, 0x18) as usize).clamp(1, MAX_CHANNELS),
            queued: (u32_at(info, 0x3C) as usize).min(WAVE_BUFFERS),
            head: (u16::from_le_bytes([info[0x40], info[0x41]]) as usize) % WAVE_BUFFERS,
            dest_mix: u32_at(info, 0x58),
            channel_ids,
            coefficients_at: u64_at(info, 0x48),
            coefficients_size: u64_at(info, 0x50),
            waves,
        }
    }
}

/// How many source frames a window keeps, and how many it reads at a time.
const KEEP: usize = 1024;
const CHUNK: usize = 256;

/// The samples of the wave buffer being played, converted to floats, a stretch of them at a time.
#[derive(Debug, Default)]
struct Window {
    /// Which buffer the stretch belongs to.
    key: Option<(u64, i32, i32, usize)>,
    /// The index (in frames of the buffer's data) of the first frame held.
    first: u64,
    /// The frames held, `channels` floats each.
    data: Vec<f32>,
    channels: usize,
    /// ADPCM: the next frame of 14 samples to decode, and the samples before it.
    next_frame: u64,
    history: History,
}

impl Window {
    fn frames(&self) -> u64 {
        (self.data.len() / self.channels.max(1)) as u64
    }

    /// The decoder state a buffer starts with.
    fn context(wave: &Wave, mem: ReadMemory<'_>) -> History {
        let mut bytes = [0u8; 6];
        if wave.context != 0 && wave.context_size >= 6 && mem(wave.context, &mut bytes) {
            // A header (not used), then the two samples before the first, the older one last.
            History {
                y1: i16::from_le_bytes([bytes[2], bytes[3]]),
                y2: i16::from_le_bytes([bytes[4], bytes[5]]),
            }
        } else {
            History::default()
        }
    }

    /// Make frame `index` of `wave` available.
    fn reach(
        &mut self,
        wave: &Wave,
        format: Format,
        channels: usize,
        coefficients: &Coefficients,
        index: u64,
        mem: ReadMemory<'_>,
    ) {
        let key = (wave.address, wave.start, wave.end, channels);
        if self.key != Some(key) {
            *self = Window {
                key: Some(key),
                channels,
                ..Window::default()
            };
            self.seek(wave, format, index, mem);
        } else if index < self.first || index >= self.first + self.frames() + 4096 {
            self.seek(wave, format, index, mem);
        }
        while index >= self.first + self.frames() {
            self.extend(wave, format, coefficients, mem);
        }
        // Forget what is far behind.
        let behind = index - self.first;
        if behind > KEEP as u64 {
            let drop = (behind - 2) as usize;
            self.data.drain(..drop * self.channels);
            self.first += drop as u64;
        }
    }

    fn seek(&mut self, wave: &Wave, format: Format, index: u64, mem: ReadMemory<'_>) {
        self.data.clear();
        if format == Format::Adpcm {
            let (frame, _) = adpcm::locate(index);
            self.first = frame * adpcm::FRAME_SAMPLES as u64;
            self.next_frame = frame;
            self.history = Window::context(wave, mem);
        } else {
            self.first = index;
        }
    }

    /// Read or decode the next stretch after the frames held.
    fn extend(
        &mut self,
        wave: &Wave,
        format: Format,
        coefficients: &Coefficients,
        mem: ReadMemory<'_>,
    ) {
        let channels = self.channels;
        if format == Format::Adpcm {
            const FRAMES: usize = 16;
            let at = self.next_frame * adpcm::FRAME_BYTES as u64;
            let mut bytes = [0u8; FRAMES * adpcm::FRAME_BYTES];
            // Data past the end of the buffer is silence.
            let readable = wave.size.saturating_sub(at).min(bytes.len() as u64) as usize;
            let ok = readable >= adpcm::FRAME_BYTES
                && mem(wave.address + at, &mut bytes[..readable / 8 * 8]);
            let count = if ok { readable / adpcm::FRAME_BYTES } else { 0 };
            for f in 0..FRAMES {
                let mut out = [0i16; adpcm::FRAME_SAMPLES];
                if f < count {
                    adpcm::decode_frame(
                        &bytes[f * 8..f * 8 + 8],
                        coefficients,
                        &mut self.history,
                        &mut out,
                    );
                }
                self.data
                    .extend(out.iter().map(|s| f32::from(*s) / 32768.0));
            }
            self.next_frame += FRAMES as u64;
        } else {
            let width = format.bytes() * channels;
            let first = self.first + self.frames();
            let at = first * width as u64;
            let mut bytes = vec![0u8; CHUNK * width];
            let readable = wave.size.saturating_sub(at).min(bytes.len() as u64) as usize;
            let ok =
                readable >= width && mem(wave.address + at, &mut bytes[..readable / width * width]);
            let count = if ok { readable / width } else { 0 };
            for f in 0..CHUNK {
                for c in 0..channels {
                    let v = if f < count {
                        let o = f * width + c * format.bytes();
                        match format {
                            Format::Pcm8 => f32::from(bytes[o] as i8) / 128.0,
                            Format::Pcm16 => {
                                f32::from(i16::from_le_bytes([bytes[o], bytes[o + 1]])) / 32768.0
                            }
                            Format::Pcm32 => {
                                i32::from_le_bytes([
                                    bytes[o],
                                    bytes[o + 1],
                                    bytes[o + 2],
                                    bytes[o + 3],
                                ]) as f32
                                    / 2_147_483_648.0
                            }
                            Format::Float => f32::from_le_bytes([
                                bytes[o],
                                bytes[o + 1],
                                bytes[o + 2],
                                bytes[o + 3],
                            ]),
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    };
                    self.data.push(v);
                }
            }
        }
    }

    /// The channels of frame `index`, which `reach` has made available.
    fn frame(&self, index: u64) -> &[f32] {
        let at = (index - self.first) as usize * self.channels;
        &self.data[at..at + self.channels]
    }
}

/// What the renderer keeps for one voice between updates.
#[derive(Debug)]
pub struct Voice {
    /// Wave buffers finished since the voice began: the number the program reads back.
    pub consumed: u32,
    /// Samples played since the voice began.
    pub played: u64,
    /// Samples already played of the wave buffer at the head of the ring.
    position: u64,
    /// The part of a sample the pitch left over at the end of the last frame.
    fraction: f64,
    window: Window,
    /// The samples this frame made, a run for each channel; empty when the voice is silent.
    pub out: Vec<Vec<f32>>,
    /// The gain of each channel into each mix buffer at the end of the last frame (the next
    /// frame ramps from it), and whether they were set at all.
    pub gains: Vec<f32>,
    pub gains_set: bool,
}

impl Default for Voice {
    fn default() -> Voice {
        Voice {
            consumed: 0,
            played: 0,
            position: 0,
            fraction: 0.0,
            window: Window::default(),
            out: vec![Vec::new(); MAX_CHANNELS],
            gains: vec![0.0; MAX_CHANNELS * MAX_MIX_BUFFERS],
            gains_set: false,
        }
    }
}

impl Voice {
    /// Advance by one frame of `frame` output samples at `output_rate`. Afterwards `out` holds the
    /// samples the frame made (a run for each channel of the voice) if the voice made any.
    pub fn play_frame(
        &mut self,
        v: &VoiceIn,
        output_rate: u32,
        frame: usize,
        mem: ReadMemory<'_>,
    ) -> bool {
        if v.is_new || !v.used {
            *self = Voice::default();
        }
        for run in &mut self.out {
            run.clear();
        }
        if !v.used || !v.started {
            return false;
        }
        let output_rate = f64::from(output_rate.max(1));
        let pitch = f64::from(v.pitch).max(0.0);
        let rate = f64::from(v.sample_rate);
        let total = frame as f64 * rate / output_rate * pitch;

        let made = self.render(v, total / frame.max(1) as f64, frame, mem);

        // The bookkeeping: how far the wave buffers got.
        let mut budget = total + self.fraction;
        let mut finished = 0;
        while finished < v.queued && budget >= 1.0 {
            let wave = &v.waves[(v.head + finished) % WAVE_BUFFERS];
            let length = wave.length();
            let remaining = length.saturating_sub(self.position);
            let step = budget.floor() as u64;
            if step >= remaining {
                budget -= remaining as f64;
                self.played += remaining;
                self.position = 0;
                if wave.looping && length > 0 {
                    // Played again from the start; the voice never finishes this buffer.
                    continue;
                }
                self.consumed = self.consumed.wrapping_add(1);
                finished += 1;
            } else {
                budget -= step as f64;
                self.played += step;
                self.position += step;
            }
        }
        self.fraction = if finished < v.queued {
            budget.fract()
        } else {
            0.0
        };
        made
    }

    /// Make the samples of the frame. `step` is how many source samples one output sample
    /// advances.
    fn render(&mut self, v: &VoiceIn, step: f64, frame: usize, mem: ReadMemory<'_>) -> bool {
        if v.format == Format::Unsupported || v.queued == 0 {
            return false;
        }
        // The ADPCM coefficients: eight pairs.
        let mut coefficients: Coefficients = [0; 16];
        if v.format == Format::Adpcm {
            let mut bytes = [0u8; 32];
            if v.coefficients_size >= 32 && mem(v.coefficients_at, &mut bytes) {
                for (i, c) in coefficients.iter_mut().enumerate() {
                    *c = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                }
            }
        }
        let channels = if v.format == Format::Adpcm {
            1
        } else {
            v.channels
        };
        for run in self.out.iter_mut().take(channels) {
            run.resize(frame, 0.0);
        }
        let mut pos = self.position as f64 + self.fraction;
        let mut k = 0usize;
        let mut sounded = false;
        for n in 0..frame {
            // The buffer this output sample falls in.
            let found = loop {
                if k >= v.queued {
                    break None;
                }
                let wave = &v.waves[(v.head + k) % WAVE_BUFFERS];
                let length = wave.length() as f64;
                if length <= 0.0 {
                    k += 1;
                    continue;
                }
                if pos >= length {
                    if wave.looping {
                        pos = pos.rem_euclid(length);
                    } else {
                        pos -= length;
                        k += 1;
                        continue;
                    }
                }
                break Some(wave);
            };
            let Some(wave) = found else {
                break;
            };
            let whole = pos.floor();
            let frac = (pos - whole) as f32;
            let at = whole as u64;
            let index = wave.start as i64 as u64 + at;
            self.window
                .reach(wave, v.format, channels, &coefficients, index, mem);
            let a = self.window.frame(index);
            // The sample after, to interpolate with: the next of the buffer, or the first again
            // when it loops, or none.
            let next = if at + 1 < wave.length() {
                Some(index + 1)
            } else if wave.looping {
                Some(wave.start as i64 as u64)
            } else {
                None
            };
            let mut ys = [0.0f32; MAX_CHANNELS];
            ys[..channels].copy_from_slice(a);
            if let Some(next) = next {
                if frac > 0.0 {
                    // The window may have to move to the loop's start and back.
                    self.window
                        .reach(wave, v.format, channels, &coefficients, next, mem);
                    let b = self.window.frame(next);
                    for (y, next) in ys.iter_mut().zip(b) {
                        *y += (next - *y) * frac;
                    }
                }
            }
            for (run, y) in self.out.iter_mut().zip(ys).take(channels) {
                run[n] = y;
            }
            sounded = true;
            pos += step;
        }
        if !sounded {
            for run in &mut self.out {
                run.clear();
            }
        }
        sounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A voice entry (0x170 bytes) for tests.
    pub fn entry(
        format: u8,
        rate: u32,
        pitch: f32,
        waves: &[(u64, u64, i32, i32, bool)],
    ) -> Vec<u8> {
        let mut b = vec![0u8; 0x170];
        b[0x09] = 1;
        b[0x0A] = VOICE_STARTED;
        b[0x0B] = format;
        b[0x0C..0x10].copy_from_slice(&rate.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
        b[0x1C..0x20].copy_from_slice(&pitch.to_bits().to_le_bytes());
        b[0x20..0x24].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        b[0x3C..0x40].copy_from_slice(&(waves.len() as u32).to_le_bytes());
        for (i, (address, size, start, end, looping)) in waves.iter().enumerate() {
            let w = WAVE_BUFFERS_AT + i * WAVE_BUFFER_SIZE;
            b[w..w + 8].copy_from_slice(&address.to_le_bytes());
            b[w + 8..w + 16].copy_from_slice(&size.to_le_bytes());
            b[w + 0x10..w + 0x14].copy_from_slice(&start.to_le_bytes());
            b[w + 0x14..w + 0x18].copy_from_slice(&end.to_le_bytes());
            b[w + 0x18] = u8::from(*looping);
        }
        b
    }

    /// Memory for tests: bytes at an address.
    fn memory(at: u64, bytes: Vec<u8>) -> impl Fn(u64, &mut [u8]) -> bool {
        move |addr, out| {
            if addr < at || addr + out.len() as u64 > at + bytes.len() as u64 {
                return false;
            }
            let o = (addr - at) as usize;
            out.copy_from_slice(&bytes[o..o + out.len()]);
            true
        }
    }

    fn pcm16(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn a_pcm_voice_makes_its_samples_at_the_output_rate() {
        let data: Vec<i16> = (0..480).map(|i| (i * 10) as i16).collect();
        let mem = memory(0x1000, pcm16(&data));
        let info = entry(2, 48_000, 1.0, &[(0x1000, 960, 0, 480, false)]);
        let v = VoiceIn::parse(&info);
        let mut voice = Voice::default();
        assert!(voice.play_frame(&v, 48_000, 240, &mem));
        assert_eq!(voice.out[0].len(), 240);
        for n in [0usize, 1, 100, 239] {
            let want = (n * 10) as f32 / 32768.0;
            assert!((voice.out[0][n] - want).abs() < 1e-6, "{n}");
        }
        assert_eq!((voice.played, voice.consumed), (240, 0));
        // The second frame goes on where the first left off, and ends the buffer.
        let mut second = entry(2, 48_000, 1.0, &[(0x1000, 960, 0, 480, false)]);
        second[0x3C] = 1;
        let v = VoiceIn::parse(&second);
        assert!(voice.play_frame(&v, 48_000, 240, &mem));
        assert!((voice.out[0][0] - 2400.0 / 32768.0).abs() < 1e-6);
        assert_eq!((voice.played, voice.consumed), (480, 1));
    }

    #[test]
    fn half_the_rate_interpolates_between_samples() {
        let data = [0i16, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];
        let mem = memory(0x1000, pcm16(&data));
        let info = entry(2, 24_000, 1.0, &[(0x1000, 20, 0, 10, false)]);
        let v = VoiceIn::parse(&info);
        let mut voice = Voice::default();
        assert!(voice.play_frame(&v, 48_000, 24, &mem));
        // One source sample for two output samples: 0, 500, 1000, 1500 ...
        for n in 0..19 {
            let want = n as f32 * 500.0 / 32768.0;
            assert!(
                (voice.out[0][n] - want).abs() < 1e-5,
                "{n}: {}",
                voice.out[0][n]
            );
        }
        // After the last sample the value stays (nothing to interpolate with), then there is
        // silence, and the buffer was consumed.
        assert!((voice.out[0][19] - 9000.0 / 32768.0).abs() < 1e-5);
        assert_eq!(voice.consumed, 1);
        assert!(voice.out[0][20..].iter().all(|s| *s == 0.0));
    }

    #[test]
    fn a_looping_buffer_wraps_to_its_start() {
        let data = [100i16, 200, 300, 400];
        let mem = memory(0x1000, pcm16(&data));
        let info = entry(2, 48_000, 1.0, &[(0x1000, 8, 0, 4, true)]);
        let v = VoiceIn::parse(&info);
        let mut voice = Voice::default();
        assert!(voice.play_frame(&v, 48_000, 10, &mem));
        let got: Vec<i32> = voice.out[0]
            .iter()
            .map(|s| (s * 32768.0).round() as i32)
            .collect();
        assert_eq!(got, vec![100, 200, 300, 400, 100, 200, 300, 400, 100, 200]);
        assert_eq!(voice.consumed, 0);
    }

    #[test]
    fn two_buffers_play_one_after_the_other() {
        let mut bytes = pcm16(&[10, 20, 30]);
        bytes.extend(pcm16(&[-1, -2, -3, -4]));
        let mem = memory(0x1000, bytes);
        let info = entry(
            2,
            48_000,
            1.0,
            &[(0x1000, 6, 0, 3, false), (0x1006, 8, 1, 4, false)],
        );
        let v = VoiceIn::parse(&info);
        let mut voice = Voice::default();
        voice.play_frame(&v, 48_000, 8, &mem);
        let got: Vec<i32> = voice.out[0]
            .iter()
            .map(|s| (s * 32768.0).round() as i32)
            .collect();
        // 10 20 30, then the second buffer from its sample 1: -2 -3 -4, then nothing.
        assert_eq!(got, vec![10, 20, 30, -2, -3, -4, 0, 0]);
        assert_eq!((voice.played, voice.consumed), (6, 2));
    }

    #[test]
    fn an_adpcm_voice_decodes_its_frames() {
        // Two frames: the first counts up by two (a coefficient of 1.0 and a nibble of 2), the
        // second continues from its last sample with a nibble of -1.
        let mut coefficients = [0u8; 32];
        coefficients[0..2].copy_from_slice(&2048i16.to_le_bytes());
        let mut data = vec![0x00u8, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22];
        data.extend([0x00u8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        let mut memory_bytes = coefficients.to_vec();
        memory_bytes.extend(data);
        let mem = memory(0x2000, memory_bytes);
        let mut info = entry(6, 48_000, 1.0, &[(0x2020, 16, 0, 28, false)]);
        info[0x48..0x50].copy_from_slice(&0x2000u64.to_le_bytes());
        info[0x50..0x58].copy_from_slice(&32u64.to_le_bytes());
        let v = VoiceIn::parse(&info);
        let mut voice = Voice::default();
        voice.play_frame(&v, 48_000, 28, &mem);
        let got: Vec<i32> = voice.out[0]
            .iter()
            .map(|s| (s * 32768.0).round() as i32)
            .collect();
        let up: Vec<i32> = (1..=14).map(|i| i * 2).collect();
        let down: Vec<i32> = (1..=14).map(|i| 28 - i).collect();
        assert_eq!(&got[..14], &up[..]);
        assert_eq!(&got[14..], &down[..]);
    }

    #[test]
    fn a_format_nobody_decodes_is_silent_but_still_consumed() {
        let mem = memory(0x1000, vec![0; 100]);
        let info = entry(3, 48_000, 1.0, &[(0x1000, 30, 0, 10, false)]);
        let v = VoiceIn::parse(&info);
        let mut voice = Voice::default();
        assert!(!voice.play_frame(&v, 48_000, 20, &mem));
        assert_eq!((voice.played, voice.consumed), (10, 1));
    }

    #[test]
    fn memory_that_is_not_there_is_silence() {
        let mem = |_: u64, _: &mut [u8]| false;
        let info = entry(2, 48_000, 1.0, &[(0x1000, 200, 0, 100, false)]);
        let v = VoiceIn::parse(&info);
        let mut voice = Voice::default();
        // The frame is made, of zeros, and the buffer still plays out.
        voice.play_frame(&v, 48_000, 50, &mem);
        assert!(voice.out[0].iter().all(|s| *s == 0.0));
        assert_eq!(voice.played, 50);
    }
}
