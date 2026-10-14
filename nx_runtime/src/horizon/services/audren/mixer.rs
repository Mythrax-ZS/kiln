//! The mixes and sinks of the audio renderer: what an update says about them, and the mixing of
//! the voices of a frame through them to the samples of the output device (docs/ASSUMPTIONS.md
//! A-125).
//!
//! A frame starts with every mix buffer silent. Each voice adds its samples to the buffers of the
//! mix it plays into, scaled by its volume and by what the voice's channel resources say for each
//! buffer. Then the mixes pass what they hold on, deepest first: a mix adds each of its buffers to
//! the buffers of the mix it plays into, by the matrix of its own gains. The device sink picks
//! buffers of the final mix as the channels of the output. Effects (reverb, delay, auxiliary
//! sends, the limiter, biquad filters) are not applied.

use super::voice::MAX_MIX_BUFFERS;
use super::{f32_at, u32_at};

/// How loud the centre and the back channels are in each side of a stereo fold-down (minus 3 dB).
const FOLD: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// One entry of the mix section of an update.
pub const MIX_SIZE: usize = 0x930;
/// One entry of the sink section.
pub const SINK_SIZE: usize = 0x140;
/// One entry of the voice channel resources.
pub const CHANNEL_SIZE: usize = 0x70;

/// The mix that is the last, which feeds the sinks.
pub const FINAL_MIX: u32 = 0;
/// A destination that is no mix.
#[cfg(test)]
pub const NO_MIX: u32 = 0x7FFF_FFFF;

/// What an update says about a mix.
#[derive(Clone, Debug)]
pub struct Mix {
    pub used: bool,
    pub volume: f32,
    pub buffers: usize,
    /// The mix this one plays into.
    pub dest: u32,
    /// The gain from each of this mix's buffers (row) to each of the destination's (column).
    pub matrix: Vec<f32>,
    /// Where this mix's buffers start among all the mix buffers.
    pub offset: usize,
}

impl Mix {
    fn parse(entry: &[u8]) -> Mix {
        let mut matrix = vec![0.0; MAX_MIX_BUFFERS * MAX_MIX_BUFFERS];
        for (i, g) in matrix.iter_mut().enumerate() {
            *g = f32_at(entry, 0x24 + i * 4);
        }
        Mix {
            used: entry[0x0C] != 0,
            volume: f32_at(entry, 0),
            buffers: (u32_at(entry, 0x08) as usize).min(MAX_MIX_BUFFERS),
            dest: u32_at(entry, 0x924),
            matrix,
            offset: 0,
        }
    }
}

/// What an update says about a device sink: which mix buffers are its channels.
#[derive(Clone, Debug, Default)]
pub struct Sink {
    pub used: bool,
    /// A device (not a circular buffer in the program's memory).
    pub device: bool,
    pub inputs: Vec<usize>,
}

impl Sink {
    fn parse(entry: &[u8]) -> Sink {
        let kind = entry[0];
        let used = entry[1] != 0;
        // The device form: a name of 0x100 bytes, then the count and the inputs.
        let count = (u32_at(entry, 0x20 + 0x100) as usize).min(6);
        let inputs = (0..count)
            .map(|i| usize::from(entry[0x20 + 0x104 + i]))
            .collect();
        Sink {
            used,
            device: kind == 1,
            inputs,
        }
    }
}

/// The gains of the channels of a voice into the mix buffers: what its channel resources say.
#[derive(Clone, Debug)]
pub struct Channel {
    pub used: bool,
    pub mix: [f32; MAX_MIX_BUFFERS],
}

impl Channel {
    pub fn parse(entry: &[u8]) -> Channel {
        let mut mix = [0.0; MAX_MIX_BUFFERS];
        for (i, g) in mix.iter_mut().enumerate() {
            *g = f32_at(entry, 4 + i * 4);
        }
        Channel {
            used: entry[4 + MAX_MIX_BUFFERS * 4] != 0,
            mix,
        }
    }
}

/// The mixing of a frame.
#[derive(Debug, Default)]
pub struct Mixer {
    frame: usize,
    /// Every mix buffer of the renderer, `frame` samples each, one after the other.
    buffers: Vec<f32>,
    pub mixes: Vec<Mix>,
    pub sinks: Vec<Sink>,
}

impl Mixer {
    /// Read the mixes of an update (the `count` entries of `section`) and the sinks.
    pub fn read_graph(&mut self, mixes: &[u8], sinks: &[u8], buffer_count: usize, frame: usize) {
        self.frame = frame;
        self.mixes = mixes
            .as_chunks::<MIX_SIZE>()
            .0
            .iter()
            .map(|entry| Mix::parse(entry))
            .collect();
        // The buffers of the mixes are laid out one mix after another, in the order of the mixes.
        let mut offset = 0;
        for mix in &mut self.mixes {
            mix.offset = offset;
            if mix.used {
                offset += mix.buffers;
            }
        }
        self.sinks = sinks
            .as_chunks::<SINK_SIZE>()
            .0
            .iter()
            .map(|entry| Sink::parse(entry))
            .collect();
        let want = buffer_count.max(offset) * frame;
        self.buffers.clear();
        self.buffers.resize(want, 0.0);
    }

    /// Forget the graph: there is nothing to mix into.
    pub fn clear(&mut self) {
        self.mixes.clear();
        self.sinks.clear();
    }

    pub fn has_graph(&self) -> bool {
        self.mixes.iter().any(|m| m.used)
    }

    /// Add `samples` (one channel of a voice) into the buffers of the mix `dest` with gains
    /// that go from `from` to `to` over the frame.
    pub fn add_voice_channel(
        &mut self,
        dest: u32,
        samples: &[f32],
        from: &[f32; MAX_MIX_BUFFERS],
        to: &[f32; MAX_MIX_BUFFERS],
    ) {
        let Some(mix) = self.mixes.get(dest as usize).filter(|m| m.used) else {
            return;
        };
        let frame = self.frame;
        let n = samples.len().min(frame);
        for j in 0..mix.buffers {
            let (g0, g1) = (from[j], to[j]);
            if g0 == 0.0 && g1 == 0.0 {
                continue;
            }
            let at = (mix.offset + j) * frame;
            let Some(buffer) = self.buffers.get_mut(at..at + frame) else {
                continue;
            };
            let delta = (g1 - g0) / frame as f32;
            for (i, (out, s)) in buffer.iter_mut().zip(samples).take(n).enumerate() {
                *out += s * (g0 + delta * (i + 1) as f32);
            }
        }
    }

    /// How many mixes lie between `id` and the final mix (0 for the final mix itself).
    fn depth(&self, id: usize) -> usize {
        let mut depth = 0;
        let mut at = id;
        while at != FINAL_MIX as usize && depth < self.mixes.len() {
            let dest = self.mixes[at].dest as usize;
            if dest >= self.mixes.len() || !self.mixes[dest].used {
                break;
            }
            at = dest;
            depth += 1;
        }
        depth
    }

    /// Pass the mixes on, deepest first, and make the stereo samples of the device sink
    /// (interleaved, clamped to 16 bits). Returns false if there is no sink to play through.
    pub fn finish(&mut self, out: &mut Vec<i16>) -> bool {
        let frame = self.frame;
        let mut order: Vec<usize> = (0..self.mixes.len())
            .filter(|&i| self.mixes[i].used && i != FINAL_MIX as usize)
            .collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.depth(i)));
        for i in order {
            let (src, dest) = (&self.mixes[i], self.mixes[i].dest as usize);
            let Some(target) = self.mixes.get(dest).filter(|m| m.used) else {
                continue;
            };
            let (src_off, src_n, dst_off, dst_n) =
                (src.offset, src.buffers, target.offset, target.buffers);
            for a in 0..src_n {
                for b in 0..dst_n {
                    let gain = self.mixes[i].matrix[a * MAX_MIX_BUFFERS + b] * self.mixes[i].volume;
                    if gain == 0.0 {
                        continue;
                    }
                    let (from, to) = ((src_off + a) * frame, (dst_off + b) * frame);
                    if from + frame > self.buffers.len() || to + frame > self.buffers.len() {
                        continue;
                    }
                    for k in 0..frame {
                        let v = self.buffers[from + k] * gain;
                        self.buffers[to + k] += v;
                    }
                }
            }
        }

        let Some(sink) = self.sinks.iter().find(|s| s.used && s.device) else {
            return false;
        };
        let final_mix = self.mixes.get(FINAL_MIX as usize).filter(|m| m.used);
        let volume = final_mix.map_or(1.0, |m| m.volume);
        let base = final_mix.map_or(0, |m| m.offset);
        // The buffers the sink's inputs name.
        let channels: Vec<Option<&[f32]>> = sink
            .inputs
            .iter()
            .map(|&index| {
                let at = (base + index) * frame;
                self.buffers.get(at..at + frame)
            })
            .collect();
        out.clear();
        out.reserve(frame * 2);
        // Stereo: the first two inputs; one input plays on both sides; six (front left and
        // right, centre, low-frequency, back left and right) are folded down.
        for i in 0..frame {
            let get = |k: usize| channels.get(k).copied().flatten().map_or(0.0, |b| b[i]);
            let (l, r) = match channels.len() {
                0 => (0.0, 0.0),
                1 => (get(0), get(0)),
                2..=5 => (get(0), get(1)),
                _ => {
                    let centre = FOLD * get(2);
                    (
                        get(0) + centre + FOLD * get(4),
                        get(1) + centre + FOLD * get(5),
                    )
                }
            };
            for v in [l, r] {
                let s = (v * volume * 32768.0).round();
                out.push(s.clamp(-32768.0, 32767.0) as i16);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mix entry: the volume, the buffers, whether it is used, where it plays and a gain from
    /// buffer `a` to buffer `b` for each `(a, b, gain)`.
    pub fn mix_entry(
        volume: f32,
        buffers: u32,
        dest: u32,
        gains: &[(usize, usize, f32)],
    ) -> Vec<u8> {
        let mut e = vec![0u8; MIX_SIZE];
        e[0..4].copy_from_slice(&volume.to_bits().to_le_bytes());
        e[8..12].copy_from_slice(&buffers.to_le_bytes());
        e[0x0C] = 1;
        for (a, b, g) in gains {
            let at = 0x24 + (a * MAX_MIX_BUFFERS + b) * 4;
            e[at..at + 4].copy_from_slice(&g.to_bits().to_le_bytes());
        }
        e[0x924..0x928].copy_from_slice(&dest.to_le_bytes());
        e
    }

    /// A device sink entry whose inputs are the given buffers.
    pub fn sink_entry(inputs: &[u8]) -> Vec<u8> {
        let mut e = vec![0u8; SINK_SIZE];
        e[0] = 1;
        e[1] = 1;
        e[0x20 + 0x100..0x20 + 0x104].copy_from_slice(&(inputs.len() as u32).to_le_bytes());
        e[0x20 + 0x104..0x20 + 0x104 + inputs.len()].copy_from_slice(inputs);
        e
    }

    fn gains(pairs: &[(usize, f32)]) -> [f32; MAX_MIX_BUFFERS] {
        let mut g = [0.0; MAX_MIX_BUFFERS];
        for (j, v) in pairs {
            g[*j] = *v;
        }
        g
    }

    #[test]
    fn a_voice_reaches_the_sink_through_the_final_mix() {
        let mut mixer = Mixer::default();
        let mixes = mix_entry(1.0, 2, NO_MIX, &[]);
        mixer.read_graph(&mixes, &sink_entry(&[0, 1]), 2, 4);
        assert!(mixer.has_graph());
        // A voice with 0.5 into the left buffer and 0.25 into the right one.
        let from = gains(&[(0, 0.5), (1, 0.25)]);
        mixer.add_voice_channel(FINAL_MIX, &[1.0, -1.0, 0.5, 0.0], &from, &from);
        let mut out = Vec::new();
        assert!(mixer.finish(&mut out));
        assert_eq!(
            out,
            vec![16384, 8192, -16384, -8192, 8192, 4096, 0, 0],
            "left and right, interleaved"
        );
    }

    #[test]
    fn a_gain_ramps_over_the_frame() {
        let mut mixer = Mixer::default();
        mixer.read_graph(&mix_entry(1.0, 1, NO_MIX, &[]), &sink_entry(&[0]), 1, 4);
        mixer.add_voice_channel(
            FINAL_MIX,
            &[1.0; 4],
            &gains(&[(0, 0.0)]),
            &gains(&[(0, 1.0)]),
        );
        let mut out = Vec::new();
        mixer.finish(&mut out);
        // 0.25, 0.5, 0.75, 1.0 on both sides (one input plays on both).
        assert_eq!(
            out,
            vec![8192, 8192, 16384, 16384, 24576, 24576, 32767, 32767]
        );
    }

    #[test]
    fn a_submix_passes_its_buffers_on_by_its_matrix() {
        let mut mixer = Mixer::default();
        // The final mix has two buffers; mix 1 has one buffer, plays into the final mix, and
        // sends it to the second buffer at half.
        let mut mixes = mix_entry(1.0, 2, NO_MIX, &[]);
        mixes.extend(mix_entry(1.0, 1, FINAL_MIX, &[(0, 1, 0.5)]));
        mixer.read_graph(&mixes, &sink_entry(&[0, 1]), 3, 2);
        mixer.add_voice_channel(1, &[1.0, 0.5], &gains(&[(0, 1.0)]), &gains(&[(0, 1.0)]));
        let mut out = Vec::new();
        mixer.finish(&mut out);
        // Nothing on the left, half of the voice on the right.
        assert_eq!(out, vec![0, 16384, 0, 8192]);
    }

    #[test]
    fn deeper_mixes_pass_on_before_shallower_ones() {
        let mut mixer = Mixer::default();
        // Mix 2 plays into mix 1, which plays into the final mix: a chain listed out of order.
        let mut mixes = mix_entry(1.0, 1, NO_MIX, &[]);
        mixes.extend(mix_entry(1.0, 1, FINAL_MIX, &[(0, 0, 1.0)]));
        mixes.extend(mix_entry(1.0, 1, 1, &[(0, 0, 1.0)]));
        mixer.read_graph(&mixes, &sink_entry(&[0]), 3, 1);
        mixer.add_voice_channel(2, &[0.5], &gains(&[(0, 1.0)]), &gains(&[(0, 1.0)]));
        let mut out = Vec::new();
        mixer.finish(&mut out);
        assert_eq!(out, vec![16384, 16384]);
    }

    #[test]
    fn the_final_mix_volume_scales_and_the_output_is_clamped() {
        let mut mixer = Mixer::default();
        mixer.read_graph(&mix_entry(2.0, 1, NO_MIX, &[]), &sink_entry(&[0]), 1, 2);
        mixer.add_voice_channel(
            FINAL_MIX,
            &[0.25, 0.75],
            &gains(&[(0, 1.0)]),
            &gains(&[(0, 1.0)]),
        );
        let mut out = Vec::new();
        mixer.finish(&mut out);
        assert_eq!(out, vec![16384, 16384, 32767, 32767]);
    }

    #[test]
    fn no_device_sink_means_nothing_to_play() {
        let mut mixer = Mixer::default();
        mixer.read_graph(&mix_entry(1.0, 1, NO_MIX, &[]), &[], 1, 2);
        let mut out = Vec::new();
        assert!(!mixer.finish(&mut out));
    }

    #[test]
    fn a_voice_to_a_mix_that_is_not_used_makes_no_sound() {
        let mut mixer = Mixer::default();
        mixer.read_graph(&mix_entry(1.0, 1, NO_MIX, &[]), &sink_entry(&[0]), 1, 2);
        mixer.add_voice_channel(
            NO_MIX,
            &[1.0, 1.0],
            &gains(&[(0, 1.0)]),
            &gains(&[(0, 1.0)]),
        );
        let mut out = Vec::new();
        mixer.finish(&mut out);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn six_channels_fold_down_to_two() {
        let mut mixer = Mixer::default();
        mixer.read_graph(
            &mix_entry(1.0, 6, NO_MIX, &[]),
            &sink_entry(&[0, 1, 2, 3, 4, 5]),
            6,
            1,
        );
        let only = |j: usize| gains(&[(j, 1.0)]);
        mixer.add_voice_channel(FINAL_MIX, &[0.5], &only(2), &only(2));
        let mut out = Vec::new();
        mixer.finish(&mut out);
        // The centre goes to both sides, 3 dB down.
        let side = (0.5 * FOLD * 32768.0).round() as i16;
        assert_eq!(out, vec![side, side]);
    }
}
