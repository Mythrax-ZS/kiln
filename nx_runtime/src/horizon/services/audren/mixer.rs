//! The mixes and sinks of the audio renderer: what an update says about them, and the mixing of
//! the voices of a frame through them to the samples of the output device (docs/ASSUMPTIONS.md
//! A-125).
//!
//! A frame starts with every mix buffer silent. Each voice adds its samples to the buffers of the
//! mix it plays into, scaled by its volume and by what the voice's channel resources say for each
//! buffer. Then the mixes pass what they hold on, every mix before the mixes it feeds: a mix adds
//! each of its buffers to the buffers of the mix it plays into, by the matrix of its own gains.
//! The device sink picks buffers of the final mix as the channels of the output. A voice or a mix
//! may play into a splitter instead, which sends it on to several mixes with gains of their own
//! (docs/ASSUMPTIONS.md A-180). Effects (reverb, delay, auxiliary sends, the limiter, biquad
//! filters) are not applied.

use super::voice::MAX_MIX_BUFFERS;
use super::{f32_at, u32_at};

/// How loud the centre and the back channels are in each side of a stereo fold-down (minus 3 dB).
const FOLD: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// One entry of the mix section of an update.
pub const MIX_SIZE: usize = 0x930;
/// From revision 7 on the mix section starts with a header of this size, whose second word counts
/// the entries that follow: only the mixes that changed (docs/ASSUMPTIONS.md A-179).
pub const DIRTY_HEADER: usize = 0x20;
/// One entry of the sink section.
pub const SINK_SIZE: usize = 0x140;
/// One entry of the voice channel resources.
pub const CHANNEL_SIZE: usize = 0x70;

/// The splitter section (docs/ASSUMPTIONS.md A-180): a header (`SNDH`, the number of splitters
/// and of destinations that follow), the splitters (`SNDI`, id, sample rate, the number of its
/// destinations, their ids, three more words) and the destinations (`SNDD`, id, a gain for each
/// buffer of the mix, the mix, whether it is used), each section rounded up to 16 bytes. An
/// update holds only the splitters and destinations that changed.
const SPLITTER_HEADER: usize = 0x20;
const SPLITTER_INFO: usize = 0x1C;
const SPLITTER_DESTINATION: usize = 0x70;
/// More splitters or destinations than this in an id is taken for a layout that is not known.
const SPLITTER_LIMIT: usize = 0x1_0000;

/// The mix that is the last, which feeds the sinks.
pub const FINAL_MIX: u32 = 0;
/// A destination that is no mix.
#[cfg(test)]
pub const NO_MIX: u32 = 0x7FFF_FFFF;

/// What an update says about a mix.
#[derive(Clone, Debug, Default)]
pub struct Mix {
    pub used: bool,
    /// Which mix the entry is (0 the final mix): what a section of only the changed mixes goes by.
    pub id: u32,
    pub volume: f32,
    pub buffers: usize,
    /// The mix this one plays into.
    pub dest: u32,
    /// The splitter it plays into instead, if `dest` is no mix.
    pub splitter: u32,
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
            id: u32_at(entry, 0x10),
            volume: f32_at(entry, 0),
            buffers: (u32_at(entry, 0x08) as usize).min(MAX_MIX_BUFFERS),
            dest: u32_at(entry, 0x924),
            splitter: u32_at(entry, 0x928),
            matrix,
            offset: 0,
        }
    }
}

/// One send of a splitter: the mix it adds to, with a gain for each of that mix's buffers.
#[derive(Clone, Debug, Default)]
pub struct Destination {
    pub used: bool,
    pub mix: u32,
    pub gains: [f32; MAX_MIX_BUFFERS],
    /// The gains a voice played through it with in the frame before, where the next frame's ramp
    /// starts.
    last: Option<[f32; MAX_MIX_BUFFERS]>,
}

/// What an update says about a device sink: which mix buffers are its channels.
#[derive(Clone, Debug, Default)]
pub struct Sink {
    pub used: bool,
    /// The type byte: 1 a device, 2 a circular buffer in the program's memory.
    pub kind: u8,
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
            kind,
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
    /// The destinations of each splitter, by id, in their order.
    pub splitters: Vec<Vec<u32>>,
    pub destinations: Vec<Destination>,
}

impl Mixer {
    /// Read the mix section of an update. Up to revision 6 it lists every mix, in order, and
    /// replaces what was known. From revision 7 on (`dirty_only`) it is a header with a count and
    /// then only the mixes that changed, each naming itself, and the others keep what they had;
    /// `mix_count` is how many mixes the renderer has (the final mix and the submixes). A section
    /// of another layout forgets the graph and returns false.
    pub fn read_mixes(
        &mut self,
        section: &[u8],
        dirty_only: bool,
        mix_count: usize,
        buffer_count: usize,
        frame: usize,
    ) -> bool {
        self.frame = frame;
        if dirty_only {
            let count = u32_at(section, 4) as usize;
            if section.len() < DIRTY_HEADER || section.len() != DIRTY_HEADER + count * MIX_SIZE {
                self.clear();
                return false;
            }
            if self.mixes.len() != mix_count {
                self.mixes = vec![Mix::default(); mix_count];
            }
            for entry in section[DIRTY_HEADER..].as_chunks::<MIX_SIZE>().0 {
                let mix = Mix::parse(entry);
                let Some(slot) = self.mixes.get_mut(mix.id as usize) else {
                    self.clear();
                    return false;
                };
                *slot = mix;
            }
        } else {
            if section.is_empty() || !section.len().is_multiple_of(MIX_SIZE) {
                self.clear();
                return false;
            }
            self.mixes = section
                .as_chunks::<MIX_SIZE>()
                .0
                .iter()
                .map(|entry| Mix::parse(entry))
                .collect();
        }
        // The buffers of the mixes are laid out one mix after another, in the order of the mixes.
        let mut offset = 0;
        for mix in &mut self.mixes {
            mix.offset = offset;
            if mix.used {
                offset += mix.buffers;
            }
        }
        let want = buffer_count.max(offset) * frame;
        self.buffers.clear();
        self.buffers.resize(want, 0.0);
        true
    }

    /// Read the sink section of an update.
    pub fn read_sinks(&mut self, section: &[u8]) {
        self.sinks = section
            .as_chunks::<SINK_SIZE>()
            .0
            .iter()
            .map(|entry| Sink::parse(entry))
            .collect();
    }

    /// The mixes of a section that lists them all, and the sinks.
    #[cfg(test)]
    pub fn read_graph(&mut self, mixes: &[u8], sinks: &[u8], buffer_count: usize, frame: usize) {
        let count = mixes.len() / MIX_SIZE;
        self.read_mixes(mixes, false, count, buffer_count, frame);
        self.read_sinks(sinks);
    }

    /// Read the splitter section of an update: the splitters and destinations it holds replace
    /// what was known of them, and the others keep what they had. An empty section changes
    /// nothing; a section of another layout returns false.
    pub fn read_splitters(&mut self, section: &[u8]) -> bool {
        if section.is_empty() {
            return true;
        }
        if section.len() < SPLITTER_HEADER || section[..4] != *b"SNDH" {
            return false;
        }
        let (infos, sends) = (u32_at(section, 4) as usize, u32_at(section, 8) as usize);
        let mut at = SPLITTER_HEADER;
        for _ in 0..infos {
            if at + SPLITTER_INFO > section.len() || section[at..at + 4] != *b"SNDI" {
                return false;
            }
            let id = u32_at(section, at + 4) as usize;
            let count = u32_at(section, at + 0x0C) as usize;
            let size = SPLITTER_INFO + 4 * count;
            if id >= SPLITTER_LIMIT || count >= SPLITTER_LIMIT || at + size > section.len() {
                return false;
            }
            if self.splitters.len() <= id {
                self.splitters.resize(id + 1, Vec::new());
            }
            self.splitters[id] = (0..count)
                .map(|k| u32_at(section, at + 0x10 + 4 * k))
                .collect();
            at += size;
        }
        for _ in 0..sends {
            if at + SPLITTER_DESTINATION > section.len() || section[at..at + 4] != *b"SNDD" {
                return false;
            }
            let id = u32_at(section, at + 4) as usize;
            if id >= SPLITTER_LIMIT {
                return false;
            }
            if self.destinations.len() <= id {
                self.destinations.resize(id + 1, Destination::default());
            }
            let d = &mut self.destinations[id];
            for (j, g) in d.gains.iter_mut().enumerate() {
                *g = f32_at(section, at + 8 + 4 * j);
            }
            d.mix = u32_at(section, at + 0x68);
            d.used = section[at + 0x6C] != 0;
            at += SPLITTER_DESTINATION;
        }
        at.next_multiple_of(16) == section.len()
    }

    /// Forget the graph: there is nothing to mix into.
    pub fn clear(&mut self) {
        self.mixes.clear();
        self.sinks.clear();
        self.splitters.clear();
        self.destinations.clear();
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

    /// Add channel `channel` of a voice of `channels` channels into the mixes of splitter
    /// `splitter`: every `channels`-th destination of the splitter from the `channel`-th on adds
    /// the samples to its mix by its gains times `volume`, ramping from the gains of the frame
    /// before.
    pub fn add_voice_through_splitter(
        &mut self,
        splitter: u32,
        channel: usize,
        channels: usize,
        samples: &[f32],
        volume: f32,
    ) {
        let Some(list) = self.splitters.get(splitter as usize) else {
            return;
        };
        let destinations = &mut self.destinations;
        let sends: Vec<(u32, [f32; MAX_MIX_BUFFERS], [f32; MAX_MIX_BUFFERS])> = list
            .iter()
            .skip(channel)
            .step_by(channels.max(1))
            .filter_map(|&id| {
                let d = destinations.get_mut(id as usize).filter(|d| d.used)?;
                let to = d.gains.map(|g| g * volume);
                let from = d.last.unwrap_or(to);
                d.last = Some(to);
                Some((d.mix, from, to))
            })
            .collect();
        for (mix, from, to) in sends {
            self.add_voice_channel(mix, samples, &from, &to);
        }
    }

    /// The used destinations of splitter `id`, with their places in its list.
    fn sends(&self, id: u32) -> Vec<(usize, Destination)> {
        self.splitters
            .get(id as usize)
            .map_or_else(Vec::new, |list| {
                list.iter()
                    .enumerate()
                    .filter_map(|(k, &d)| {
                        let d = self.destinations.get(d as usize).filter(|d| d.used)?;
                        Some((k, d.clone()))
                    })
                    .collect()
            })
    }

    /// The mixes mix `i` plays into: its destination, or the mixes of its splitter's sends.
    fn feeds(&self, i: usize) -> Vec<usize> {
        let mix = &self.mixes[i];
        if (mix.dest as usize) < self.mixes.len() {
            return vec![mix.dest as usize];
        }
        self.sends(mix.splitter)
            .into_iter()
            .map(|(_, d)| d.mix as usize)
            .filter(|&m| m < self.mixes.len())
            .collect()
    }

    /// The used mixes but the final one, each before every mix it plays into (what a cycle, which
    /// the console refuses, leaves over comes last, in the order of the mixes).
    fn order(&self) -> Vec<usize> {
        let n = self.mixes.len();
        let used = |i: usize| self.mixes[i].used;
        let feeds: Vec<Vec<usize>> = (0..n)
            .map(|i| if used(i) { self.feeds(i) } else { Vec::new() })
            .collect();
        let mut waiting = vec![0usize; n];
        for (i, list) in feeds.iter().enumerate() {
            for &d in list {
                if used(d) && d != i {
                    waiting[d] += 1;
                }
            }
        }
        let mut ready: std::collections::VecDeque<usize> =
            (0..n).filter(|&i| used(i) && waiting[i] == 0).collect();
        let mut done = vec![false; n];
        let mut order = Vec::new();
        while let Some(i) = ready.pop_front() {
            done[i] = true;
            order.push(i);
            for &d in &feeds[i] {
                if used(d) && d != i && !done[d] {
                    waiting[d] -= 1;
                    if waiting[d] == 0 {
                        ready.push_back(d);
                    }
                }
            }
        }
        order.extend((0..n).filter(|&i| used(i) && !done[i]));
        order.retain(|&i| i != FINAL_MIX as usize);
        order
    }

    /// Add mix buffer `from` times `gain` into mix buffer `to`.
    fn add_buffer(&mut self, from: usize, to: usize, gain: f32) {
        let frame = self.frame;
        let (from, to) = (from * frame, to * frame);
        if gain == 0.0 || from + frame > self.buffers.len() || to + frame > self.buffers.len() {
            return;
        }
        for k in 0..frame {
            let v = self.buffers[from + k] * gain;
            self.buffers[to + k] += v;
        }
    }

    /// Pass the mixes on, each before the mixes it plays into, and make the stereo samples of the
    /// device sink (interleaved, clamped to 16 bits). Returns false if there is no sink to play
    /// through.
    pub fn finish(&mut self, out: &mut Vec<i16>) -> bool {
        let frame = self.frame;
        for i in self.order() {
            let (src_off, src_n, volume) = {
                let m = &self.mixes[i];
                (m.offset, m.buffers, m.volume)
            };
            let dest = self.mixes[i].dest as usize;
            if dest < self.mixes.len() {
                let Some(target) = self.mixes.get(dest).filter(|m| m.used) else {
                    continue;
                };
                let (dst_off, dst_n) = (target.offset, target.buffers);
                for a in 0..src_n {
                    for b in 0..dst_n {
                        let gain = self.mixes[i].matrix[a * MAX_MIX_BUFFERS + b] * volume;
                        self.add_buffer(src_off + a, dst_off + b, gain);
                    }
                }
            } else if src_n > 0 {
                // Through a splitter: its k-th send takes buffer k (counted round the mix's
                // buffers) to every buffer of its mix, by the send's gain for that buffer.
                for (k, send) in self.sends(self.mixes[i].splitter) {
                    let Some(target) = self.mixes.get(send.mix as usize).filter(|m| m.used) else {
                        continue;
                    };
                    let (dst_off, dst_n) = (target.offset, target.buffers);
                    for b in 0..dst_n {
                        self.add_buffer(src_off + k % src_n, dst_off + b, volume * send.gains[b]);
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
        // No splitter, as the titles say it.
        e[0x928..0x92C].copy_from_slice(&u32::MAX.to_le_bytes());
        e
    }

    /// `entry` naming itself mix `id`.
    fn with_id(mut entry: Vec<u8>, id: u32) -> Vec<u8> {
        entry[0x10..0x14].copy_from_slice(&id.to_le_bytes());
        entry
    }

    /// A mix section of revision 7 and later: the header and the entries of the changed mixes.
    fn dirty(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut section = vec![0u8; DIRTY_HEADER];
        section[4..8].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in entries {
            section.extend_from_slice(e);
        }
        section
    }

    /// A splitter section: the splitters (id, the ids of its destinations) and the destinations
    /// (id, mix, the gain of each buffer).
    type Send<'a> = (u32, u32, &'a [(usize, f32)]);
    fn splitter_section(infos: &[(u32, &[u32])], sends: &[Send<'_>]) -> Vec<u8> {
        let word = |e: &mut Vec<u8>, at: usize, v: u32| {
            e[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        let mut s = vec![0u8; SPLITTER_HEADER];
        s[0..4].copy_from_slice(b"SNDH");
        word(&mut s, 4, infos.len() as u32);
        word(&mut s, 8, sends.len() as u32);
        for (id, ids) in infos {
            let mut e = vec![0u8; SPLITTER_INFO + 4 * ids.len()];
            e[0..4].copy_from_slice(b"SNDI");
            word(&mut e, 4, *id);
            word(&mut e, 8, 48_000);
            word(&mut e, 0x0C, ids.len() as u32);
            for (k, d) in ids.iter().enumerate() {
                word(&mut e, 0x10 + 4 * k, *d);
            }
            s.extend(e);
        }
        for (id, mix, gains) in sends {
            let mut e = vec![0u8; SPLITTER_DESTINATION];
            e[0..4].copy_from_slice(b"SNDD");
            word(&mut e, 4, *id);
            for (j, g) in gains.iter() {
                word(&mut e, 8 + 4 * j, g.to_bits());
            }
            word(&mut e, 0x68, *mix);
            e[0x6C] = 1;
            s.extend(e);
        }
        s.resize(s.len().next_multiple_of(16), 0);
        s
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

    /// One frame of two samples of a voice into mix 1 of `mixer`, as the device sink plays it.
    fn play_into_mix_1(mixer: &mut Mixer) -> Vec<i16> {
        mixer.add_voice_channel(1, &[1.0, 0.5], &gains(&[(0, 1.0)]), &gains(&[(0, 1.0)]));
        let mut out = Vec::new();
        assert!(mixer.finish(&mut out));
        out
    }

    #[test]
    fn from_revision_7_an_update_holds_only_the_mixes_that_changed() {
        // The third title's renderer (revision 8) sent one changed mix an update, then only the
        // header: the mixes it sent before go on as they were.
        let mut mixer = Mixer::default();
        let sink = sink_entry(&[0, 1]);
        let final_mix = with_id(mix_entry(1.0, 2, NO_MIX, &[]), 0);
        let half_right = with_id(mix_entry(1.0, 1, FINAL_MIX, &[(0, 1, 0.5)]), 1);
        for section in [dirty(&[final_mix]), dirty(&[half_right])] {
            assert!(mixer.read_mixes(&section, true, 2, 3, 2));
            mixer.read_sinks(&sink);
        }
        assert_eq!(play_into_mix_1(&mut mixer), vec![0, 16384, 0, 8192]);
        // Nothing changed: the header alone, and the same sound.
        assert!(mixer.read_mixes(&dirty(&[]), true, 2, 3, 2));
        assert_eq!(play_into_mix_1(&mut mixer), vec![0, 16384, 0, 8192]);
        // The submix changes its gain; the final mix is not sent and stays.
        let quarter_left = with_id(mix_entry(1.0, 1, FINAL_MIX, &[(0, 0, 0.25)]), 1);
        assert!(mixer.read_mixes(&dirty(&[quarter_left]), true, 2, 3, 2));
        assert_eq!(play_into_mix_1(&mut mixer), vec![8192, 0, 4096, 0]);
    }

    #[test]
    fn a_changed_mix_the_renderer_does_not_have_or_a_short_section_is_not_read() {
        let mut mixer = Mixer::default();
        let final_mix = with_id(mix_entry(1.0, 2, NO_MIX, &[]), 0);
        assert!(mixer.read_mixes(&dirty(std::slice::from_ref(&final_mix)), true, 2, 2, 2));
        assert!(mixer.has_graph());
        let beyond = with_id(mix_entry(1.0, 1, FINAL_MIX, &[]), 2);
        assert!(!mixer.read_mixes(&dirty(&[beyond]), true, 2, 2, 2));
        assert!(!mixer.has_graph(), "the graph is forgotten");
        // A count that the section does not hold.
        let mut short = dirty(&[final_mix]);
        short[4] = 2;
        assert!(!mixer.read_mixes(&short, true, 2, 2, 2));
        assert!(!mixer.read_mixes(&[], true, 2, 2, 2));
        // Up to revision 6 the section holds whole entries and nothing else.
        assert!(!mixer.read_mixes(&dirty(&[]), false, 1, 2, 2));
    }

    #[test]
    fn a_voice_plays_through_the_destinations_of_its_splitter() {
        // The second title's voices play into splitters whose destinations name the mixes. Here
        // splitter 0 sends a stereo voice: its destinations alternate between the channels, the
        // left channel's to the left buffer of the final mix and at half to the right one of mix
        // 1, the right channel's to the right buffer of the final mix.
        let mut mixer = Mixer::default();
        let mut mixes = mix_entry(1.0, 2, NO_MIX, &[]);
        mixes.extend(mix_entry(1.0, 2, FINAL_MIX, &[(0, 0, 1.0), (1, 1, 1.0)]));
        mixer.read_graph(&mixes, &sink_entry(&[0, 1]), 4, 2);
        let section = splitter_section(
            &[(0, &[3, 1, 2])],
            &[
                (3, FINAL_MIX, &[(0, 1.0)]),
                (1, FINAL_MIX, &[(1, 1.0)]),
                (2, 1, &[(1, 0.5)]),
            ],
        );
        assert!(mixer.read_splitters(&section));
        mixer.add_voice_through_splitter(0, 0, 2, &[0.5, 0.25], 1.0);
        mixer.add_voice_through_splitter(0, 1, 2, &[-0.5, -0.25], 1.0);
        let mut out = Vec::new();
        assert!(mixer.finish(&mut out));
        // Left: the left channel. Right: the right channel, and half the left one through mix 1.
        assert_eq!(out, vec![16384, -8192, 8192, -4096]);
    }

    #[test]
    fn a_voice_ramps_from_the_gains_its_destination_had_and_by_its_volume() {
        let mut mixer = Mixer::default();
        mixer.read_graph(&mix_entry(1.0, 1, NO_MIX, &[]), &sink_entry(&[0]), 1, 4);
        let at = |g: f32| splitter_section(&[(0, &[0])], &[(0, FINAL_MIX, &[(0, g)])]);
        assert!(mixer.read_splitters(&at(1.0)));
        mixer.add_voice_through_splitter(0, 0, 1, &[1.0; 4], 0.5);
        let mut out = Vec::new();
        mixer.finish(&mut out);
        assert_eq!(out, vec![16384; 8], "a first frame starts at its gain");
        // The next frame the destination's gain is 0: the voice fades out over the frame.
        assert!(mixer.read_mixes(&mix_entry(1.0, 1, NO_MIX, &[]), false, 1, 1, 4));
        assert!(mixer.read_splitters(&at(0.0)));
        mixer.add_voice_through_splitter(0, 0, 1, &[1.0; 4], 0.5);
        mixer.finish(&mut out);
        assert_eq!(out, vec![12288, 12288, 8192, 8192, 4096, 4096, 0, 0]);
    }

    #[test]
    fn a_mix_plays_through_a_splitter_buffer_by_buffer_round_its_buffers() {
        // A stereo submix into a splitter of four destinations: its left buffer goes to the
        // first and the third, its right buffer to the second and the fourth (the order of the
        // sends counts round its two buffers).
        let mut mixer = Mixer::default();
        let mut mixes = mix_entry(1.0, 2, NO_MIX, &[]);
        let mut sub = mix_entry(0.5, 2, NO_MIX, &[]);
        sub[0x928..0x92C].copy_from_slice(&0u32.to_le_bytes());
        mixes.extend(sub);
        mixer.read_graph(&mixes, &sink_entry(&[0, 1]), 4, 1);
        let section = splitter_section(
            &[(0, &[0, 1, 2, 3])],
            &[
                (0, FINAL_MIX, &[(0, 1.0)]),
                (1, FINAL_MIX, &[(1, 1.0)]),
                (2, FINAL_MIX, &[(1, 0.5)]),
                (3, FINAL_MIX, &[(0, 0.5)]),
            ],
        );
        assert!(mixer.read_splitters(&section));
        let left = gains(&[(0, 1.0)]);
        mixer.add_voice_channel(1, &[0.5], &left, &left);
        let mut out = Vec::new();
        mixer.finish(&mut out);
        // The submix's volume (0.5) times the sends: 0.25 to the left, 0.125 to the right.
        assert_eq!(out, vec![8192, 4096]);
    }

    #[test]
    fn a_mix_that_feeds_another_through_a_splitter_passes_on_first() {
        // Mix 1 plays into the final mix; mix 2, listed after it, plays into mix 1 through a
        // splitter, so it must pass its buffer on before mix 1 does.
        let mut mixer = Mixer::default();
        let mut mixes = mix_entry(1.0, 1, NO_MIX, &[]);
        mixes.extend(mix_entry(1.0, 1, FINAL_MIX, &[(0, 0, 1.0)]));
        let mut feeder = mix_entry(1.0, 1, NO_MIX, &[]);
        feeder[0x928..0x92C].copy_from_slice(&0u32.to_le_bytes());
        mixes.extend(feeder);
        mixer.read_graph(&mixes, &sink_entry(&[0]), 3, 1);
        assert!(mixer.read_splitters(&splitter_section(&[(0, &[0])], &[(0, 1, &[(0, 1.0)])])));
        mixer.add_voice_channel(2, &[0.5], &gains(&[(0, 1.0)]), &gains(&[(0, 1.0)]));
        let mut out = Vec::new();
        mixer.finish(&mut out);
        assert_eq!(out, vec![16384, 16384]);
    }

    #[test]
    fn splitters_an_update_does_not_hold_keep_what_they_had() {
        let mut mixer = Mixer::default();
        mixer.read_graph(&mix_entry(1.0, 1, NO_MIX, &[]), &sink_entry(&[0]), 1, 1);
        assert!(mixer.read_splitters(&splitter_section(
            &[(1, &[4])],
            &[(4, FINAL_MIX, &[(0, 1.0)])]
        )));
        // The header alone, and an empty section: nothing changes.
        assert!(mixer.read_splitters(&splitter_section(&[], &[])));
        assert!(mixer.read_splitters(&[]));
        mixer.add_voice_through_splitter(1, 0, 1, &[0.5], 1.0);
        let mut out = Vec::new();
        mixer.finish(&mut out);
        assert_eq!(out, vec![16384, 16384]);
    }

    #[test]
    fn a_splitter_section_of_another_layout_is_not_read() {
        let mut mixer = Mixer::default();
        let good = splitter_section(&[(0, &[0])], &[(0, FINAL_MIX, &[(0, 1.0)])]);
        let mut magic = good.clone();
        magic[0x20] = b'X';
        assert!(
            !mixer.read_splitters(&magic),
            "a splitter without its magic"
        );
        let mut count = good.clone();
        count[8] = 2;
        assert!(
            !mixer.read_splitters(&count),
            "a destination the section does not hold"
        );
        let mut longer = good.clone();
        longer.extend([0u8; 16]);
        assert!(!mixer.read_splitters(&longer), "bytes after the last entry");
        assert!(!mixer.read_splitters(&good[..0x10]), "a short header");
        assert!(mixer.read_splitters(&good));
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
