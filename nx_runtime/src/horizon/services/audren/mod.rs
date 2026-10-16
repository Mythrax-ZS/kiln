//! `audren:u`, the audio renderer (Switchbrew, "Audio services"; the layouts of the update buffers
//! are those `libnx` documents for the revisions 1 to 6 of the renderer, used as they are for 7 to
//! 11: docs/ASSUMPTIONS.md A-140).
//!
//! The renderer is what a program can see of the console's audio DSP (docs/ASSUMPTIONS.md A-075,
//! A-125). Each update is answered with the state of the memory pools, voices and sinks, and plays
//! one audio frame (`sample_count` output samples, 5 ms) of the voices' wave buffers: the samples
//! are decoded, mixed through the mixes to the device sink and handed to the sound device (see
//! [`super::super::audio_out`]). The system event is set once per audio frame so the program's
//! audio thread goes on. Effects and the filters of voices are not applied. A layout this code
//! does not know stops the run.

mod adpcm;
mod mixer;
mod voice;

use super::super::audio_out::Outputs;
use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use mixer::{Channel, Mixer, CHANNEL_SIZE, MIX_SIZE, SINK_SIZE};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;
use voice::{Format, ReadMemory, Voice, VoiceIn};

/// The audio frame: 5 ms (200 per second), whatever the sample rate (Switchbrew; `libnx`).
const FRAME_PERIOD: Duration = Duration::from_millis(5);

/// The `REV` tag every revision number carries in its low three bytes.
const REVISION_TAG: u32 = 0x0056_4552;

/// The newest revision a program may ask for: 6 is the last whose layouts `libnx` documents, and
/// 7 to 11 are taken to have the same section sizes (docs/ASSUMPTIONS.md A-140).
const LAST_KNOWN_REVISION: u32 = 11;

/// The size of the parameter block `OpenAudioRenderer` and `GetWorkBufferSize` take.
const PARAMETER_SIZE: usize = 0x34;

/// The update header, in both directions.
const HEADER_SIZE: usize = 0x40;

// Entry sizes of the input sections and the output sections (`libnx`, `audren.h`).
const MEMPOOL_IN: usize = 0x20;
const MEMPOOL_OUT: usize = 0x10;
const VOICE_IN: usize = 0x170;
const VOICE_OUT: usize = 0x10;
const CHANNEL_IN: usize = CHANNEL_SIZE;
const EFFECT_IN: usize = 0xC0;
const EFFECT_OUT: usize = 0x10;
/// From revision 9 on an effect's answer is 0x90 bytes: the state, then the statistics of a
/// limiter (docs/ASSUMPTIONS.md A-127).
const EFFECT_OUT_V2: usize = 0x90;
const EFFECT_V2_REVISION: u32 = 9;
/// What the first byte of an effect's answer says: still in use, or removed.
const EFFECT_IN_USE: u8 = 3;
const EFFECT_REMOVED: u8 = 4;
const SINK_OUT: usize = 0x20;
const PERFORMANCE_OUT: usize = 0x10;
const BEHAVIOR_OUT: usize = 0xB0;
const BEHAVIOR_IN: usize = 0x10;

// Memory pool states (`libnx`: `AudioRendererMemPoolState`).
const POOL_REQUEST_DETACH: u32 = 2;
const POOL_DETACHED: u32 = 3;
const POOL_REQUEST_ATTACH: u32 = 4;
const POOL_ATTACHED: u32 = 5;

/// The parameters a renderer was opened with (`libnx`: `AudioRendererParameter`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Parameters {
    sample_rate: u32,
    sample_count: u32,
    mix_buffer_count: u32,
    submix_count: u32,
    voice_count: u32,
    sink_count: u32,
    effect_count: u32,
    splitter_count: u32,
    revision: u32,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4)
        .map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from(u32_at(b, o)) | (u64::from(u32_at(b, o + 4)) << 32)
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(b, o))
}

impl Parameters {
    fn parse(data: &[u8]) -> Option<Parameters> {
        if data.len() < PARAMETER_SIZE {
            return None;
        }
        Some(Parameters {
            sample_rate: u32_at(data, 0x00),
            sample_count: u32_at(data, 0x04),
            mix_buffer_count: u32_at(data, 0x08),
            submix_count: u32_at(data, 0x0C),
            voice_count: u32_at(data, 0x10),
            sink_count: u32_at(data, 0x14),
            effect_count: u32_at(data, 0x18),
            splitter_count: u32_at(data, 0x24),
            revision: u32_at(data, 0x30),
        })
    }

    /// How much memory the work buffer must have. The console's figure depends on the layout of
    /// its renderer; this one is room for the counts it was opened with (docs/ASSUMPTIONS.md
    /// A-075), and the program only allocates what it is told.
    fn work_buffer_size(&self) -> u64 {
        let counts = u64::from(self.voice_count) * 0x800
            + u64::from(self.effect_count) * 0x800
            + u64::from(self.submix_count) * 0x800
            + u64::from(self.sink_count) * 0x200
            + u64::from(self.splitter_count) * 0x400
            + u64::from(self.mix_buffer_count) * u64::from(self.sample_count) * 8;
        (0x4_0000 + counts).next_multiple_of(0x1000)
    }
}

/// Where an effect is in its life as the renderer sees it (`libnx` documents no effect, the names
/// and the way the states follow are those of the console's renderer as other emulators describe
/// it, docs/ASSUMPTIONS.md A-127).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EffectUse {
    /// The slot holds no effect.
    Invalid,
    /// An effect the program has just made: reported as in use.
    New,
    Enabled,
    /// An effect the program switched off: reported as removed, which is what it waits for.
    Disabled,
}

/// The state of the renderer and the processing of one update.
#[derive(Debug)]
struct Core {
    params: Parameters,
    voices: Vec<Voice>,
    /// What the renderer has done with each effect so far (an `EffectUse`): effects are not run,
    /// but a program that removes one waits for the renderer to say it is gone.
    effects: Vec<EffectUse>,
    /// `NX_TRACE_AUDIO`: what each effect entry said (type, new, enabled) and what was answered
    /// (removed) the last time it was reported, so that only changes are said.
    effect_said: Vec<Option<(u8, bool, bool, bool)>>,
    mixer: Mixer,
    /// The interleaved stereo samples of the last frame, and whether it made any (it needs mixes
    /// and a device sink to play through).
    audio: Vec<i16>,
    sounding: bool,
    /// What has been said already, so that it is said once.
    said_header: bool,
    said_formats: u32,
    /// `NX_TRACE_AUDIO`: say every second (200 updates) what the voices did.
    trace: bool,
    seen: Seen,
}

/// What the voices did since the last line of `NX_TRACE_AUDIO`.
#[derive(Debug, Default)]
struct Seen {
    updates: u32,
    /// Voice-updates in which a voice was used, started, and made samples.
    used: u32,
    started: u32,
    made: u32,
    /// Started voices by sample format: ADPCM, 16-bit PCM, others.
    adpcm: u32,
    pcm16: u32,
    other: u32,
    /// Updates that were mixed to the sink, and the largest sample of those.
    sounding: u32,
    peak: i32,
}

impl Core {
    fn new(params: Parameters) -> Core {
        Core {
            params,
            voices: (0..params.voice_count).map(|_| Voice::default()).collect(),
            effects: vec![EffectUse::Invalid; params.effect_count as usize],
            effect_said: vec![None; params.effect_count as usize],
            mixer: Mixer::default(),
            audio: Vec::new(),
            sounding: false,
            said_header: false,
            said_formats: 0,
            trace: std::env::var_os("NX_TRACE_AUDIO").is_some(),
            seen: Seen::default(),
        }
    }

    /// Process an update buffer and return the output buffer; `mem` reads the program's memory
    /// (the wave buffers), `active` says whether the renderer has been started. An `Err` names a
    /// layout this code does not know.
    fn update(
        &mut self,
        input: &[u8],
        mem: ReadMemory<'_>,
        active: bool,
    ) -> Result<Vec<u8>, String> {
        if input.len() < HEADER_SIZE {
            return Err(format!(
                "an audio renderer update of {} bytes, which has no header",
                input.len()
            ));
        }
        let revision = u32_at(input, 0);
        if revision & 0x00FF_FFFF != REVISION_TAG {
            return Err(format!(
                "an audio renderer update with revision {revision:#x}"
            ));
        }
        let behavior = u32_at(input, 0x04) as usize;
        let pools = u32_at(input, 0x08) as usize;
        let voices = u32_at(input, 0x0C) as usize;
        let channels = u32_at(input, 0x10) as usize;
        let effects = u32_at(input, 0x14) as usize;
        let mixes = u32_at(input, 0x18) as usize;
        let sinks = u32_at(input, 0x1C) as usize;
        let total = u32_at(input, 0x3C) as usize;
        if !self.said_header && std::env::var_os("NX_TRACE_SVC").is_some() {
            self.said_header = true;
            let words: Vec<String> = (0..16)
                .map(|i| format!("{:#x}", u32_at(input, i * 4)))
                .collect();
            eprintln!(
                "[audren] the first update has {:#x} bytes, header {}",
                input.len(),
                words.join(" ")
            );
        }
        if behavior != BEHAVIOR_IN
            || !pools.is_multiple_of(MEMPOOL_IN)
            || !voices.is_multiple_of(VOICE_IN)
            || !channels.is_multiple_of(CHANNEL_IN)
            || voices / VOICE_IN != channels / CHANNEL_IN
            || total > input.len()
            || HEADER_SIZE + behavior + pools + channels + voices > total
        {
            return Err(format!(
                "an audio renderer update (revision {revision:#x}) whose sections do not have \
                 the sizes this code knows: behavior {behavior:#x}, memory pools {pools:#x}, \
                 voice resources {channels:#x}, voices {voices:#x}, total {total:#x} of {:#x}",
                input.len()
            ));
        }
        let pool_count = pools / MEMPOOL_IN;
        let voice_count = voices / VOICE_IN;
        if voice_count != self.voices.len() {
            return Err(format!(
                "an audio renderer update with {voice_count} voices for a renderer opened with {}",
                self.voices.len()
            ));
        }
        let pools_at = HEADER_SIZE + behavior;
        let channels_at = pools_at + pools;
        let voices_at = channels_at + channels;

        let effect_entry = if revision_number(revision) >= EFFECT_V2_REVISION {
            EFFECT_OUT_V2
        } else {
            EFFECT_OUT
        };
        let effect_out = self.params.effect_count as usize * effect_entry;
        let sinks_out = self.params.sink_count as usize * SINK_OUT;
        let out_pools = pool_count * MEMPOOL_OUT;
        let out_voices = voice_count * VOICE_OUT;
        let size = HEADER_SIZE
            + out_pools
            + out_voices
            + effect_out
            + sinks_out
            + PERFORMANCE_OUT
            + BEHAVIOR_OUT;
        let mut out = vec![0u8; size];
        let put =
            |out: &mut [u8], o: usize, v: u32| out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut out, 0, revision);
        put(&mut out, 0x04, BEHAVIOR_OUT as u32);
        put(&mut out, 0x08, out_pools as u32);
        put(&mut out, 0x0C, out_voices as u32);
        put(&mut out, 0x14, effect_out as u32);
        put(&mut out, 0x1C, sinks_out as u32);
        put(&mut out, 0x20, PERFORMANCE_OUT as u32);
        put(&mut out, 0x3C, size as u32);

        // Memory pools: a request to attach or detach is granted; any other state is left as it
        // is (the program reads "invalid" as "no change").
        let mut at = HEADER_SIZE;
        for i in 0..pool_count {
            let state = u32_at(input, pools_at + i * MEMPOOL_IN + 0x10);
            let new_state = match state {
                POOL_REQUEST_ATTACH => POOL_ATTACHED,
                POOL_REQUEST_DETACH => POOL_DETACHED,
                _ => 0,
            };
            put(&mut out, at, new_state);
            at += MEMPOOL_OUT;
        }

        // The mixes and the sinks follow the voices and the effects in the input (and whatever
        // the splitters take); a layout that does not add up plays nothing.
        let splitters = u32_at(input, 0x24) as usize;
        let mixes_at = voices_at + voices + effects + splitters;
        let sinks_at = mixes_at + mixes;
        let graph = mixes.is_multiple_of(MIX_SIZE)
            && sinks.is_multiple_of(SINK_SIZE)
            && mixes > 0
            && sinks > 0
            && sinks_at + sinks <= total;
        let frame = self.params.sample_count as usize;
        if graph {
            self.mixer.read_graph(
                &input[mixes_at..sinks_at],
                &input[sinks_at..sinks_at + sinks],
                self.params.mix_buffer_count as usize,
                frame,
            );
        } else {
            self.mixer.clear();
        }
        let resources: Vec<Channel> = input[channels_at..voices_at]
            .as_chunks::<CHANNEL_SIZE>()
            .0
            .iter()
            .map(|entry| Channel::parse(entry))
            .collect();

        // Voices: one audio frame of the wave buffers of every voice that is playing, mixed.
        for i in 0..voice_count {
            let base = voices_at + i * VOICE_IN;
            let v = VoiceIn::parse(&input[base..base + VOICE_IN]);
            if v.used && v.format == Format::Unsupported && self.said_formats & (1 << (i % 32)) == 0
            {
                self.said_formats |= 1 << (i % 32);
                eprintln!(
                    "[audren] voice {i} uses sample format {}, which is not decoded: it is silent",
                    input[base + 0x0B]
                );
            }
            let voice = &mut self.voices[i];
            let made = voice.play_frame(&v, self.params.sample_rate, frame, mem);
            self.seen.used += u32::from(v.used);
            self.seen.started += u32::from(v.used && v.started);
            self.seen.made += u32::from(made);
            if v.used && v.started {
                match v.format {
                    Format::Adpcm => self.seen.adpcm += 1,
                    Format::Pcm16 => self.seen.pcm16 += 1,
                    _ => self.seen.other += 1,
                }
            }
            if made && self.mixer.has_graph() {
                for c in 0..v.channels {
                    if voice.out[c].is_empty() {
                        continue;
                    }
                    let mut target = [0.0f32; voice::MAX_MIX_BUFFERS];
                    if let Some(res) = resources.get(v.channel_ids[c] as usize).filter(|r| r.used) {
                        for (t, g) in target.iter_mut().zip(res.mix) {
                            *t = g * v.volume;
                        }
                    }
                    let mut from = target;
                    if voice.gains_set {
                        from.copy_from_slice(
                            &voice.gains
                                [c * voice::MAX_MIX_BUFFERS..(c + 1) * voice::MAX_MIX_BUFFERS],
                        );
                    }
                    self.mixer
                        .add_voice_channel(v.dest_mix, &voice.out[c], &from, &target);
                    voice.gains[c * voice::MAX_MIX_BUFFERS..(c + 1) * voice::MAX_MIX_BUFFERS]
                        .copy_from_slice(&target);
                }
                voice.gains_set = true;
            }
            out[at..at + 8].copy_from_slice(&voice.played.to_le_bytes());
            put(&mut out, at + 8, voice.consumed);
            at += VOICE_OUT;
        }
        // Effects: not run, but kept in the books. The answer for an effect is what the renderer
        // made of it in the frame before (a program that switches an effect off sees it used for
        // one more update, then removed), and the frame this update starts makes it enabled, or
        // disabled if the program does not want it.
        let effects_at = voices_at + voices;
        let effect_entries = if effects_at + effects <= total {
            (effects / EFFECT_IN).min(self.effects.len())
        } else {
            0
        };
        for i in 0..effect_entries {
            let entry = &input[effects_at + i * EFFECT_IN..effects_at + (i + 1) * EFFECT_IN];
            let (kind, is_new, enabled) = (entry[0], entry[1] != 0, entry[2] != 0);
            let usage = &mut self.effects[i];
            if kind == 0 {
                *usage = EffectUse::Invalid;
            } else if is_new || *usage == EffectUse::Invalid {
                *usage = EffectUse::New;
            }
            let removed = if active {
                *usage == EffectUse::Disabled
            } else {
                *usage != EffectUse::New
            };
            out[at + i * effect_entry] = if removed {
                EFFECT_REMOVED
            } else {
                EFFECT_IN_USE
            };
            if self.trace && self.effect_said[i] != Some((kind, is_new, enabled, removed)) {
                self.effect_said[i] = Some((kind, is_new, enabled, removed));
                eprintln!(
                    "[audio] effect {i}: type {kind}, new {is_new}, enabled {enabled}: answered {}",
                    if removed { "removed" } else { "in use" }
                );
            }
            if active && *usage != EffectUse::Invalid {
                *usage = if enabled {
                    EffectUse::Enabled
                } else {
                    EffectUse::Disabled
                };
            }
        }
        self.sounding = self.mixer.has_graph() && self.mixer.finish(&mut self.audio);
        self.seen.updates += 1;
        if self.sounding {
            self.seen.sounding += 1;
            let peak = self.audio.iter().map(|s| i32::from(*s).abs()).max();
            self.seen.peak = self.seen.peak.max(peak.unwrap_or(0));
        }
        if self.trace && self.seen.updates >= 200 {
            let n = &self.seen;
            eprintln!(
                "[audio] last {} updates: voices used {:.1}, started {:.1} (ADPCM {:.1}, PCM16 {:.1}, others {:.1}), made samples {:.1} on average; {} frames mixed to the sink, loudest sample {}",
                n.updates,
                f64::from(n.used) / f64::from(n.updates),
                f64::from(n.started) / f64::from(n.updates),
                f64::from(n.adpcm) / f64::from(n.updates),
                f64::from(n.pcm16) / f64::from(n.updates),
                f64::from(n.other) / f64::from(n.updates),
                f64::from(n.made) / f64::from(n.updates),
                n.sounding,
                n.peak
            );
            self.seen = Seen::default();
        }
        Ok(out)
    }
}

/// `nn::audio::detail::IAudioRendererManager` (`audren:u`).
#[derive(Debug)]
pub struct Manager {
    kernel: Weak<Kernel>,
}

impl Manager {
    pub fn new(kernel: Weak<Kernel>) -> Manager {
        Manager { kernel }
    }
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// What the renderer's result is when the parameters ask for more than it has. (Audio module 153,
/// description 2: "invalid parameter" in Switchbrew's list of results; not seen from a program.)
const INVALID_PARAMETER: ResultCode = ResultCode::new(153, 2);

impl Service for Manager {
    fn name(&self) -> &str {
        "nn::audio::detail::IAudioRendererManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // OpenAudioRenderer(AudioRendererParameter, u64 work buffer size, u64 aruid, pid,
            // transfer memory, process) -> IAudioRenderer
            0 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let Some(params) = Parameters::parse(&req.data) else {
                    return Err(Stop("OpenAudioRenderer without its parameters".into()));
                };
                if params.sample_count == 0 || params.sample_rate == 0 {
                    return Ok(Response::error(INVALID_PARAMETER));
                }
                check_revision(&params)?;
                if std::env::var_os("NX_TRACE_SVC").is_some() {
                    eprintln!("[audren] OpenAudioRenderer {params:?}");
                }
                Ok(Response::ok().with_object(Box::new(Renderer::new(&kernel, params))))
            }
            // GetWorkBufferSize(AudioRendererParameter) -> u64
            1 => {
                let Some(params) = Parameters::parse(&req.data) else {
                    return Err(Stop("GetWorkBufferSize without its parameters".into()));
                };
                check_revision(&params)?;
                Ok(Response::ok().with_u64(params.work_buffer_size()))
            }
            // GetAudioDeviceService(u64 aruid) -> IAudioDevice, and the 4.0.0 form with the
            // revision of the program's SDK in front of the aruid
            2 | 4 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                Ok(Response::ok().with_object(Box::new(Device::new(&kernel))))
            }
            other => Err(Stop::unknown_command("audren:u", other)),
        }
    }
}

/// The names of the output devices (docs/ASSUMPTIONS.md A-076), each in a field of `NAME_SIZE`.
const DEVICE_NAMES: [&str; 4] = [
    "AudioStereoJackOutput",
    "AudioBuiltInSpeakerOutput",
    "AudioTvOutput",
    "AudioUsbDeviceOutput",
];

/// The device a handheld console plays through, and how many channels it has.
const ACTIVE_DEVICE: &str = "AudioBuiltInSpeakerOutput";
const ACTIVE_CHANNELS: u32 = 2;

/// The size of a `DeviceName`.
const NAME_SIZE: usize = 0x100;

/// `nn::audio::detail::IAudioDevice`.
#[derive(Debug)]
struct Device {
    kernel: Weak<Kernel>,
    /// The events of the device and of its inputs and outputs: nothing changes, so none is set.
    events: [(KObject, KObject); 3],
    volume: f32,
}

impl Device {
    fn new(kernel: &Arc<Kernel>) -> Device {
        Device {
            kernel: Arc::downgrade(kernel),
            events: [kernel.new_event(), kernel.new_event(), kernel.new_event()],
            volume: 1.0,
        }
    }

    /// Write `names` into the output buffer of the request (a receive buffer, or when the caller
    /// chose its pointer buffer, the first receive-list entry) and say how many bytes.
    fn write_names(&self, req: &Request, names: &[&str]) -> Result<(u64, bool), Stop> {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let (buffer, pointer) = match (req.output_buffer(0), req.pointer_output(0)) {
            (Some(b), _) => (b, false),
            (None, Some(p)) => (p, true),
            (None, None) => return Err(Stop("an audio device name list without a buffer".into())),
        };
        let fit = (buffer.size as usize / NAME_SIZE).min(names.len());
        let mut bytes = vec![0u8; fit * NAME_SIZE];
        for (i, name) in names.iter().take(fit).enumerate() {
            bytes[i * NAME_SIZE..i * NAME_SIZE + name.len()].copy_from_slice(name.as_bytes());
        }
        if !kernel.write_memory(buffer.addr, &bytes) {
            return Err(Stop("the audio device name buffer is not writable".into()));
        }
        Ok((fit as u64, pointer))
    }
}

impl Service for Device {
    fn name(&self) -> &str {
        "nn::audio::detail::IAudioDevice"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // ListAudioDeviceName() -> (buffer of names, s32 count), and its automatic form
            0 | 6 => {
                let (count, pointer) = self.write_names(req, &DEVICE_NAMES)?;
                let mut r = Response::ok().with_u32(count as u32);
                if pointer {
                    r = r.with_pointer_reply(0, count * NAME_SIZE as u64);
                }
                Ok(r)
            }
            // SetAudioDeviceOutputVolume(f32 volume, buffer name), and its automatic form
            1 | 7 => {
                self.volume = f32::from_bits(req.u32_at(0));
                Ok(Response::ok())
            }
            // GetAudioDeviceOutputVolume(buffer name) -> f32, and its automatic form
            2 | 8 => Ok(Response::ok().with_u32(self.volume.to_bits())),
            // ListAudioOutputDeviceName(s32 first) -> (buffer of names, s32 count) [13.0.0+]: the
            // names from the `first`th on (the title that sent it sent 0; docs/ASSUMPTIONS.md A-141)
            14 => {
                let first = (req.u32_at(0) as usize).min(DEVICE_NAMES.len());
                let (count, pointer) = self.write_names(req, &DEVICE_NAMES[first..])?;
                let mut r = Response::ok().with_u32(count as u32);
                if pointer {
                    r = r.with_pointer_reply(0, count * NAME_SIZE as u64);
                }
                Ok(r)
            }
            // GetActiveAudioDeviceName() -> buffer with one name, and its automatic form;
            // GetActiveAudioOutputDeviceName() -> buffer with one name [13.0.0+]
            3 | 10 | 13 => {
                let (_, pointer) = self.write_names(req, &[ACTIVE_DEVICE])?;
                let mut r = Response::ok();
                if pointer {
                    r = r.with_pointer_reply(0, NAME_SIZE as u64);
                }
                Ok(r)
            }
            // QueryAudioDeviceSystemEvent(), QueryAudioDeviceInputEvent(),
            // QueryAudioDeviceOutputEvent() -> event
            4 => Ok(Response::ok().with_copy_handle(self.events[0].1.clone())),
            11 => Ok(Response::ok().with_copy_handle(self.events[1].1.clone())),
            12 => Ok(Response::ok().with_copy_handle(self.events[2].1.clone())),
            // GetActiveChannelCount() -> u32
            5 => Ok(Response::ok().with_u32(ACTIVE_CHANNELS)),
            other => Err(Stop::unknown_command("IAudioDevice", other)),
        }
    }
}

/// The number of a revision: its top byte is a character, '1' to '9', then ':' for 10, ';' for 11.
fn revision_number(revision: u32) -> u32 {
    (revision >> 24).wrapping_sub(u32::from(b'0'))
}

/// A program that asks for a revision newer than the layouts known here stops the run: its update
/// buffers would be read wrongly.
fn check_revision(params: &Parameters) -> Result<(), Stop> {
    let tag_ok = params.revision & 0x00FF_FFFF == REVISION_TAG;
    let number = revision_number(params.revision);
    match (tag_ok, number) {
        (true, n) if (1..=LAST_KNOWN_REVISION).contains(&n) => Ok(()),
        _ => Err(Stop(format!(
            "the audio renderer revision {:#x}, of which only REV1 to REV{LAST_KNOWN_REVISION} \
             are known",
            params.revision
        ))),
    }
}

/// `nn::audio::detail::IAudioRenderer`.
#[derive(Debug)]
struct Renderer {
    core: Core,
    kernel: Weak<Kernel>,
    /// The frame event: (set by the timer, read by the program).
    event: (KObject, KObject),
    started: bool,
    time_limit: u32,
    /// Ends the frame timer when the renderer is closed.
    stop: Arc<AtomicBool>,
    /// Where the samples go.
    outputs: Outputs,
}

impl Renderer {
    fn new(kernel: &Arc<Kernel>, params: Parameters) -> Renderer {
        let event = kernel.new_event();
        let stop = Arc::new(AtomicBool::new(false));
        let (weak, writable, flag) = (Arc::downgrade(kernel), event.0.clone(), stop.clone());
        std::thread::spawn(move || loop {
            std::thread::sleep(FRAME_PERIOD);
            let Some(kernel) = weak.upgrade() else {
                break;
            };
            if flag.load(Ordering::Acquire) || kernel.exit_state().is_some() {
                break;
            }
            kernel.set_event(&writable, true);
        });
        Renderer {
            core: Core::new(params),
            kernel: Arc::downgrade(kernel),
            event,
            started: false,
            time_limit: 80,
            stop,
            outputs: Outputs::open(params.sample_rate),
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

impl Service for Renderer {
    fn name(&self) -> &str {
        "nn::audio::detail::IAudioRenderer"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetSampleRate(), GetSampleCount(), GetMixBufferCount() -> u32
            0 => Ok(Response::ok().with_u32(self.core.params.sample_rate)),
            1 => Ok(Response::ok().with_u32(self.core.params.sample_count)),
            2 => Ok(Response::ok().with_u32(self.core.params.mix_buffer_count)),
            // GetState() -> u32: 0 started, 1 stopped
            3 => Ok(Response::ok().with_u32(u32::from(!self.started))),
            // RequestUpdateAudioRenderer(input buffer) -> (output buffer, performance buffer),
            // and its automatic-buffer form (3.0.0 and later)
            4 | 10 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let (Some(input), Some(output)) = (req.input_buffer(0), req.output_buffer(0))
                else {
                    return Err(Stop(
                        "RequestUpdateAudioRenderer without its buffers".into(),
                    ));
                };
                let Some(data) = kernel.read_buffer(input, 0x40_0000) else {
                    return Err(Stop("the audio update buffer is not readable".into()));
                };
                let mem = |addr: u64, out: &mut [u8]| kernel.read_memory(addr, out);
                let out = self.core.update(&data, &mem, self.started).map_err(Stop)?;
                if self.started && self.core.sounding {
                    self.outputs.write(&self.core.audio);
                }
                if out.len() as u64 > output.size {
                    return Err(Stop(format!(
                        "the audio update answer ({:#x} bytes) does not fit the output buffer \
                         ({:#x})",
                        out.len(),
                        output.size
                    )));
                }
                if !kernel.write_memory(output.addr, &out) {
                    return Err(Stop(
                        "the audio update output buffer is not writable".into(),
                    ));
                }
                Ok(Response::ok())
            }
            // StartAudioRenderer(), StopAudioRenderer()
            5 => {
                self.started = true;
                Ok(Response::ok())
            }
            6 => {
                self.started = false;
                Ok(Response::ok())
            }
            // QuerySystemEvent() -> event
            7 => Ok(Response::ok().with_copy_handle(self.event.1.clone())),
            // SetRenderingTimeLimit(u32), GetRenderingTimeLimit() -> u32
            8 => {
                self.time_limit = req.u32_at(0);
                Ok(Response::ok())
            }
            9 => Ok(Response::ok().with_u32(self.time_limit)),
            other => Err(Stop::unknown_command("IAudioRenderer", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::voice::{VOICE_STARTED, WAVE_BUFFERS};
    use super::*;

    impl Core {
        /// An update that can read no memory.
        fn run(&mut self, input: &[u8]) -> Result<Vec<u8>, String> {
            self.update(input, &|_, _| false, true)
        }
    }

    const WAVE_BUFFER_SIZE: usize = 0x38;
    const WAVE_BUFFERS_AT: usize = 0x60;

    fn params() -> Parameters {
        Parameters {
            sample_rate: 48_000,
            sample_count: 240,
            mix_buffer_count: 6,
            submix_count: 0,
            voice_count: 2,
            sink_count: 1,
            effect_count: 3,
            splitter_count: 0,
            revision: 0x3556_4552, // REV5
        }
    }

    /// One wave buffer of a voice: (start sample, end sample, looping).
    type Wave = (i32, i32, bool);

    struct VoiceSpec {
        is_new: bool,
        used: bool,
        state: u8,
        sample_rate: u32,
        pitch: f32,
        head: u16,
        waves: Vec<Wave>,
    }

    fn voice(waves: Vec<Wave>) -> VoiceSpec {
        VoiceSpec {
            is_new: false,
            used: true,
            state: VOICE_STARTED,
            sample_rate: 48_000,
            pitch: 1.0,
            head: 0,
            waves,
        }
    }

    /// An update buffer for `pools` memory pools (each given its state) and the voices.
    fn update(revision: u32, pools: &[u32], voices: &[VoiceSpec]) -> Vec<u8> {
        let pool_size = pools.len() * MEMPOOL_IN;
        let channel_size = voices.len() * CHANNEL_IN;
        let voice_size = voices.len() * VOICE_IN;
        let size = HEADER_SIZE + BEHAVIOR_IN + pool_size + channel_size + voice_size + 0x100;
        let mut b = vec![0u8; size];
        let put = |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut b, 0, revision);
        put(&mut b, 0x04, BEHAVIOR_IN as u32);
        put(&mut b, 0x08, pool_size as u32);
        put(&mut b, 0x0C, voice_size as u32);
        put(&mut b, 0x10, channel_size as u32);
        put(&mut b, 0x3C, size as u32);
        let pools_at = HEADER_SIZE + BEHAVIOR_IN;
        for (i, state) in pools.iter().enumerate() {
            put(&mut b, pools_at + i * MEMPOOL_IN + 0x10, *state);
        }
        let voices_at = pools_at + pool_size + channel_size;
        for (i, v) in voices.iter().enumerate() {
            let at = voices_at + i * VOICE_IN;
            b[at + 0x08] = u8::from(v.is_new);
            b[at + 0x09] = u8::from(v.used);
            b[at + 0x0A] = v.state;
            put(&mut b, at + 0x0C, v.sample_rate);
            put(&mut b, at + 0x1C, v.pitch.to_bits());
            put(&mut b, at + 0x3C, v.waves.len() as u32);
            b[at + 0x40..at + 0x42].copy_from_slice(&v.head.to_le_bytes());
            for (k, (start, end, looping)) in v.waves.iter().enumerate() {
                let w = at
                    + WAVE_BUFFERS_AT
                    + ((v.head as usize + k) % WAVE_BUFFERS) * WAVE_BUFFER_SIZE;
                put(&mut b, w + 0x10, *start as u32);
                put(&mut b, w + 0x14, *end as u32);
                b[w + 0x18] = u8::from(*looping);
            }
        }
        b
    }

    /// The (played, consumed) pair the answer holds for voice `i` of a renderer with 2 pools.
    fn voice_out(out: &[u8], pools: usize, i: usize) -> (u64, u32) {
        let at = HEADER_SIZE + pools * MEMPOOL_OUT + i * VOICE_OUT;
        (u64_at(out, at), u32_at(out, at + 8))
    }

    #[test]
    fn the_answer_has_the_sections_the_program_checks() {
        let mut core = Core::new(params());
        let input = update(0x3556_4552, &[0, 0, 0], &[voice(vec![]), voice(vec![])]);
        let out = core.run(&input).unwrap();
        assert_eq!(u32_at(&out, 0), 0x3556_4552, "the revision is echoed");
        assert_eq!(u32_at(&out, 0x04), BEHAVIOR_OUT as u32);
        assert_eq!(u32_at(&out, 0x08), 3 * MEMPOOL_OUT as u32);
        assert_eq!(u32_at(&out, 0x0C), 2 * VOICE_OUT as u32);
        assert_eq!(
            u32_at(&out, 0x14),
            3 * EFFECT_OUT as u32,
            "an entry per effect"
        );
        assert_eq!(u32_at(&out, 0x1C), SINK_OUT as u32);
        assert_eq!(u32_at(&out, 0x20), PERFORMANCE_OUT as u32);
        assert_eq!(u32_at(&out, 0x3C) as usize, out.len());
        assert_eq!(
            out.len(),
            HEADER_SIZE + 3 * 0x10 + 2 * 0x10 + 3 * 0x10 + 0x20 + 0x10 + 0xB0
        );
    }

    /// `base` with the effect entries `(type, new, enabled)` after its voices.
    fn with_effects(mut base: Vec<u8>, entries: &[(u8, bool, bool)]) -> Vec<u8> {
        let at = HEADER_SIZE
            + u32_at(&base, 0x04) as usize
            + u32_at(&base, 0x08) as usize
            + u32_at(&base, 0x10) as usize
            + u32_at(&base, 0x0C) as usize;
        let mut bytes = Vec::new();
        for (kind, new, enabled) in entries {
            let mut e = vec![0u8; EFFECT_IN];
            e[0] = *kind;
            e[1] = u8::from(*new);
            e[2] = u8::from(*enabled);
            bytes.extend(e);
        }
        base.splice(at..at, bytes.iter().copied());
        let total = base.len() as u32;
        base[0x14..0x18].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        base[0x3C..0x40].copy_from_slice(&total.to_le_bytes());
        base
    }

    /// The state the answer gives for effect `i` (renderers with 3 pools and 2 voices) when the
    /// answer has `entry` bytes for each effect.
    fn effect_state_of(out: &[u8], i: usize, entry: usize) -> u8 {
        out[HEADER_SIZE + 3 * MEMPOOL_OUT + 2 * VOICE_OUT + i * entry]
    }

    fn effect_state(out: &[u8], i: usize) -> u8 {
        effect_state_of(out, i, EFFECT_OUT)
    }

    #[test]
    fn an_effect_the_program_switches_off_is_reported_removed_an_update_later() {
        let mut core = Core::new(params());
        let base = update(0x3556_4552, &[0; 3], &[voice(vec![]), voice(vec![])]);
        let on = with_effects(
            base.clone(),
            &[(4, true, true), (0, false, false), (0, false, false)],
        );
        let off = with_effects(
            base,
            &[(4, false, false), (0, false, false), (0, false, false)],
        );
        // In use, and an empty slot is not reported removed.
        let out = core.run(&on).unwrap();
        assert_eq!(
            [effect_state(&out, 0), effect_state(&out, 1)],
            [EFFECT_IN_USE, EFFECT_IN_USE]
        );
        assert_eq!(effect_state(&core.run(&on).unwrap(), 0), EFFECT_IN_USE);
        // Switched off: the frame before still had it, the next says it is gone.
        assert_eq!(effect_state(&core.run(&off).unwrap(), 0), EFFECT_IN_USE);
        assert_eq!(effect_state(&core.run(&off).unwrap(), 0), EFFECT_REMOVED);
        assert_eq!(
            effect_state(&core.run(&off).unwrap(), 0),
            EFFECT_REMOVED,
            "and it stays gone"
        );
    }

    #[test]
    fn from_revision_9_an_effect_has_0x90_bytes_in_the_answer() {
        // The title that stopped for ever in a scene it left (A-127) asked for revision 11: the
        // library reads the effects of the answer 0x90 bytes apart, and found the states of the
        // third effect and the ones after it in the sections that follow.
        let revision = 0x3b56_4552;
        let mut core = Core::new(Parameters {
            revision,
            effect_count: 3,
            ..params()
        });
        let base = update(revision, &[0; 3], &[voice(vec![]), voice(vec![])]);
        let on = with_effects(
            base.clone(),
            &[(6, true, true), (6, true, true), (6, true, true)],
        );
        let off = with_effects(
            base,
            &[(6, false, false), (6, false, true), (6, false, false)],
        );
        let out = core.run(&on).unwrap();
        assert_eq!(u32_at(&out, 0x14), 3 * EFFECT_OUT_V2 as u32);
        assert_eq!(u32_at(&out, 0x3C) as usize, out.len());
        assert_eq!(
            out.len(),
            HEADER_SIZE + 3 * 0x10 + 2 * 0x10 + 3 * 0x90 + 0x20 + 0x10 + 0xB0
        );
        let states = |out: &[u8]| [0, 1, 2].map(|i| effect_state_of(out, i, EFFECT_OUT_V2));
        assert_eq!(states(&out), [EFFECT_IN_USE; 3]);
        // Two of them are switched off: they are gone an update later, the third is not.
        assert_eq!(states(&core.run(&off).unwrap()), [EFFECT_IN_USE; 3]);
        assert_eq!(
            states(&core.run(&off).unwrap()),
            [EFFECT_REMOVED, EFFECT_IN_USE, EFFECT_REMOVED]
        );
    }

    #[test]
    fn a_renderer_that_was_not_started_runs_nothing_so_a_new_effect_stays_in_use() {
        let mut core = Core::new(params());
        let base = update(0x3556_4552, &[0; 3], &[voice(vec![]), voice(vec![])]);
        let fresh = with_effects(
            base.clone(),
            &[(4, true, true), (0, false, false), (0, false, false)],
        );
        let out = core.update(&fresh, &|_, _| false, false).unwrap();
        assert_eq!(
            effect_state(&out, 0),
            EFFECT_IN_USE,
            "a new effect is in use"
        );
        // The program switches it off, but without the DSP running nothing makes it removed.
        let off = with_effects(
            base,
            &[(4, false, false), (0, false, false), (0, false, false)],
        );
        let out = core.update(&off, &|_, _| false, false).unwrap();
        assert_eq!(effect_state(&out, 0), EFFECT_IN_USE);
        // Started, the update after the next one says it is gone.
        assert_eq!(effect_state(&core.run(&off).unwrap(), 0), EFFECT_IN_USE);
        assert_eq!(effect_state(&core.run(&off).unwrap(), 0), EFFECT_REMOVED);
    }

    #[test]
    fn memory_pools_are_attached_and_detached_when_asked() {
        let mut core = Core::new(params());
        let input = update(
            0x3556_4552,
            &[
                POOL_REQUEST_ATTACH,
                POOL_REQUEST_DETACH,
                POOL_ATTACHED,
                6,
                0,
            ],
            &[voice(vec![]), voice(vec![])],
        );
        let out = core.run(&input).unwrap();
        let state = |i: usize| u32_at(&out, HEADER_SIZE + i * MEMPOOL_OUT);
        assert_eq!(state(0), POOL_ATTACHED);
        assert_eq!(state(1), POOL_DETACHED);
        assert_eq!(state(2), 0, "an attached pool stays as it is");
        assert_eq!(state(3), 0, "a released one too");
        assert_eq!(state(4), 0);
    }

    #[test]
    fn a_wave_buffer_is_consumed_in_the_time_its_samples_take() {
        let mut core = Core::new(params());
        // 600 samples at one sample per sample: 240 per frame, so the third frame ends it.
        let playing = || {
            update(
                0x3556_4552,
                &[0; 3],
                &[voice(vec![(0, 600, false)]), voice(vec![])],
            )
        };
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(voice_out(&core.run(&playing()).unwrap(), 3, 0));
        }
        assert_eq!(seen[0], (240, 0));
        assert_eq!(seen[1], (480, 0));
        assert_eq!(
            seen[2],
            (600, 1),
            "the buffer ends mid-frame and the voice has nothing more"
        );
        // The client has taken the buffer off its ring by now: nothing queued, nothing played.
        let idle = update(0x3556_4552, &[0; 3], &[voice(vec![]), voice(vec![])]);
        assert_eq!(voice_out(&core.run(&idle).unwrap(), 3, 0), (600, 1));
        assert_eq!(
            voice_out(&core.run(&idle).unwrap(), 3, 1),
            (0, 0),
            "the other voice is idle"
        );
    }

    #[test]
    fn a_frame_that_ends_a_buffer_goes_on_into_the_next_one() {
        let mut core = Core::new(params());
        let input = update(
            0x3556_4552,
            &[0; 3],
            &[
                voice(vec![(0, 100, false), (10, 70, false), (0, 1000, false)]),
                voice(vec![]),
            ],
        );
        // 100 + 60 samples finish two buffers, and the remaining 80 start the third.
        assert_eq!(voice_out(&core.run(&input).unwrap(), 3, 0), (240, 2));
    }

    #[test]
    fn a_looping_buffer_never_finishes() {
        let mut core = Core::new(params());
        let input = update(
            0x3556_4552,
            &[0; 3],
            &[voice(vec![(0, 100, true)]), voice(vec![])],
        );
        for frame in 1..=5u64 {
            assert_eq!(
                voice_out(&core.run(&input).unwrap(), 3, 0),
                (240 * frame, 0)
            );
        }
    }

    #[test]
    fn pitch_and_sample_rate_scale_the_time() {
        let mut core = Core::new(params());
        let mut v = voice(vec![(0, 100_000, false)]);
        v.sample_rate = 24_000; // half the output rate
        v.pitch = 2.0; // twice as fast again
        let input = update(0x3556_4552, &[0; 3], &[v, voice(vec![])]);
        assert_eq!(voice_out(&core.run(&input).unwrap(), 3, 0), (240, 0));
        // A voice at 32 kHz plays two thirds of 240 samples a frame, carrying the fraction.
        let mut core = Core::new(params());
        let mut v = voice(vec![(0, 100_000, false)]);
        v.sample_rate = 32_000;
        let input = update(0x3556_4552, &[0; 3], &[v, voice(vec![])]);
        let played: Vec<u64> = (0..3)
            .map(|_| voice_out(&core.run(&input).unwrap(), 3, 0).0)
            .collect();
        assert_eq!(played, vec![160, 320, 480]);
    }

    #[test]
    fn a_stopped_voice_does_not_move_and_a_new_one_starts_over() {
        let mut core = Core::new(params());
        let play = update(
            0x3556_4552,
            &[0; 3],
            &[voice(vec![(0, 100_000, false)]), voice(vec![])],
        );
        core.run(&play).unwrap();
        let mut stopped = voice(vec![(0, 100_000, false)]);
        stopped.state = 1;
        let still = update(0x3556_4552, &[0; 3], &[stopped, voice(vec![])]);
        assert_eq!(voice_out(&core.run(&still).unwrap(), 3, 0), (240, 0));
        let mut fresh = voice(vec![(0, 100_000, false)]);
        fresh.is_new = true;
        let again = update(0x3556_4552, &[0; 3], &[fresh, voice(vec![])]);
        assert_eq!(
            voice_out(&core.run(&again).unwrap(), 3, 0),
            (240, 0),
            "counted from zero"
        );
    }

    #[test]
    fn a_layout_nobody_documented_stops_the_run() {
        let mut core = Core::new(params());
        let mut input = update(0x3556_4552, &[0; 3], &[voice(vec![]), voice(vec![])]);
        input[0x0C..0x10].copy_from_slice(&0x171u32.to_le_bytes());
        let e = core.run(&input).unwrap_err();
        assert!(e.contains("do not have the sizes"), "{e}");
        let e = core.run(&[0; 8]).unwrap_err();
        assert!(e.contains("no header"), "{e}");
        let mut bad = update(0x3556_4552, &[0; 3], &[voice(vec![]), voice(vec![])]);
        bad[0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        assert!(core.run(&bad).unwrap_err().contains("revision"));
    }

    #[test]
    fn revisions_are_checked_against_what_is_known() {
        let with = |revision: u32| Parameters {
            revision,
            ..params()
        };
        assert!(check_revision(&with(0x3156_4552)).is_ok(), "REV1");
        assert!(check_revision(&with(0x3656_4552)).is_ok(), "REV6");
        assert!(check_revision(&with(0x3956_4552)).is_ok(), "REV9");
        assert!(check_revision(&with(0x3A56_4552)).is_ok(), "REV10");
        assert!(check_revision(&with(0x3B56_4552)).is_ok(), "REV11");
        assert!(
            check_revision(&with(0x3C56_4552)).is_err(),
            "REV12 is not known"
        );
        assert!(check_revision(&with(0x3056_4552)).is_err(), "nor REV0");
        assert!(check_revision(&with(0x3156_4553)).is_err(), "not a REV tag");
    }

    #[test]
    fn the_work_buffer_is_page_sized_and_grows_with_the_counts() {
        let small = params().work_buffer_size();
        let big = Parameters {
            voice_count: 200,
            ..params()
        }
        .work_buffer_size();
        assert_eq!(small % 0x1000, 0);
        assert!(big > small);
    }
}
