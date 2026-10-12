//! `audren:u`, the audio renderer (Switchbrew, "Audio services"; the layouts of the update buffers
//! are those `libnx` documents for the revisions 1 to 6 of the renderer).
//!
//! Nothing is mixed and nothing is played (docs/ASSUMPTIONS.md A-075). The renderer is the
//! bookkeeping a program can see: it answers each update with the state of its memory pools, voices
//! and sinks, lets the voices consume their wave buffers in the time those samples would take, and
//! sets the system event once per audio frame (5 ms) so the program's audio thread goes on. A
//! layout this code does not know stops the run.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

/// The audio frame: 5 ms (200 per second), whatever the sample rate (Switchbrew; `libnx`).
const FRAME_PERIOD: Duration = Duration::from_millis(5);

/// The `REV` tag every revision number carries in its low three bytes.
const REVISION_TAG: u32 = 0x0056_4552;

/// The newest revision whose layouts are known here (`libnx`: `AUDREN_REVISION_6`).
const LAST_KNOWN_REVISION: u32 = 6;

/// The size of the parameter block `OpenAudioRenderer` and `GetWorkBufferSize` take.
const PARAMETER_SIZE: usize = 0x34;

/// The update header, in both directions.
const HEADER_SIZE: usize = 0x40;

// Entry sizes of the input sections and the output sections (`libnx`, `audren.h`).
const MEMPOOL_IN: usize = 0x20;
const MEMPOOL_OUT: usize = 0x10;
const VOICE_IN: usize = 0x170;
const VOICE_OUT: usize = 0x10;
const CHANNEL_IN: usize = 0x70;
const EFFECT_OUT: usize = 0x10;
const SINK_OUT: usize = 0x20;
const PERFORMANCE_OUT: usize = 0x10;
const BEHAVIOR_OUT: usize = 0xB0;
const BEHAVIOR_IN: usize = 0x10;

// Memory pool states (`libnx`: `AudioRendererMemPoolState`).
const POOL_REQUEST_DETACH: u32 = 2;
const POOL_DETACHED: u32 = 3;
const POOL_REQUEST_ATTACH: u32 = 4;
const POOL_ATTACHED: u32 = 5;

/// Voice play state: started (`AudioRendererVoicePlayState_Started`).
const VOICE_STARTED: u8 = 0;

/// The wave buffers of a voice: a ring of four.
const WAVE_BUFFERS: usize = 4;
const WAVE_BUFFER_SIZE: usize = 0x38;
const WAVE_BUFFERS_AT: usize = 0x60;

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

#[cfg(test)]
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

/// What the renderer keeps for one voice between updates.
#[derive(Clone, Copy, Debug, Default)]
struct Voice {
    /// Wave buffers finished since the voice began: the number the program reads back.
    consumed: u32,
    /// Samples played since the voice began.
    played: u64,
    /// Samples already played of the wave buffer at the head of the ring.
    position: u64,
    /// The part of a sample the pitch left over at the end of the last frame.
    fraction: f64,
}

/// The state of the renderer and the processing of one update.
#[derive(Debug)]
struct Core {
    params: Parameters,
    voices: Vec<Voice>,
}

impl Core {
    fn new(params: Parameters) -> Core {
        Core {
            params,
            voices: vec![Voice::default(); params.voice_count as usize],
        }
    }

    /// Process an update buffer and return the output buffer. An `Err` names a layout this code
    /// does not know.
    fn update(&mut self, input: &[u8]) -> Result<Vec<u8>, String> {
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
        let total = u32_at(input, 0x3C) as usize;
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
        let voices_at = pools_at + pools + channels;

        let effects = self.params.effect_count as usize * EFFECT_OUT;
        let sinks = self.params.sink_count as usize * SINK_OUT;
        let out_pools = pool_count * MEMPOOL_OUT;
        let out_voices = voice_count * VOICE_OUT;
        let size =
            HEADER_SIZE + out_pools + out_voices + effects + sinks + PERFORMANCE_OUT + BEHAVIOR_OUT;
        let mut out = vec![0u8; size];
        let put =
            |out: &mut [u8], o: usize, v: u32| out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut out, 0, revision);
        put(&mut out, 0x04, BEHAVIOR_OUT as u32);
        put(&mut out, 0x08, out_pools as u32);
        put(&mut out, 0x0C, out_voices as u32);
        put(&mut out, 0x14, effects as u32);
        put(&mut out, 0x1C, sinks as u32);
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

        // Voices: one audio frame of the wave buffers of every voice that is playing.
        for i in 0..voice_count {
            let base = voices_at + i * VOICE_IN;
            self.play_frame(i, &input[base..base + VOICE_IN]);
            let v = &self.voices[i];
            out[at..at + 8].copy_from_slice(&v.played.to_le_bytes());
            put(&mut out, at + 8, v.consumed);
            at += VOICE_OUT;
        }
        Ok(out)
    }

    /// Advance voice `index` by one frame. `info` is its entry of the input.
    fn play_frame(&mut self, index: usize, info: &[u8]) {
        let is_new = info[0x08] != 0;
        let is_used = info[0x09] != 0;
        let state = info[0x0A];
        let voice = &mut self.voices[index];
        if is_new || !is_used {
            *voice = Voice::default();
        }
        if !is_used || state != VOICE_STARTED {
            return;
        }
        let output_rate = f64::from(self.params.sample_rate.max(1));
        let pitch = f64::from(f32_at(info, 0x1C)).max(0.0);
        let rate = f64::from(u32_at(info, 0x0C));
        let frame = f64::from(self.params.sample_count);
        let mut budget = frame * rate / output_rate * pitch + voice.fraction;
        let queued = (u32_at(info, 0x3C) as usize).min(WAVE_BUFFERS);
        let head = (u16::from_le_bytes([info[0x40], info[0x41]]) as usize) % WAVE_BUFFERS;
        let mut finished = 0;
        while finished < queued && budget >= 1.0 {
            let wave = WAVE_BUFFERS_AT + ((head + finished) % WAVE_BUFFERS) * WAVE_BUFFER_SIZE;
            let start = i64::from(u32_at(info, wave + 0x10) as i32);
            let end = i64::from(u32_at(info, wave + 0x14) as i32);
            let looping = info[wave + 0x18] != 0;
            let length = (end - start).max(0) as u64;
            let remaining = length.saturating_sub(voice.position);
            let step = budget.floor() as u64;
            if step >= remaining {
                budget -= remaining as f64;
                voice.played += remaining;
                voice.position = 0;
                if looping && length > 0 {
                    // Played again from the start; the voice never finishes this buffer.
                    continue;
                }
                voice.consumed = voice.consumed.wrapping_add(1);
                finished += 1;
            } else {
                budget -= step as f64;
                voice.played += step;
                voice.position += step;
            }
        }
        voice.fraction = if finished < queued {
            budget.fract()
        } else {
            0.0
        };
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
            // GetActiveAudioDeviceName() -> buffer with one name, and its automatic form
            3 | 10 => {
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

/// A program that asks for a revision newer than the layouts known here stops the run: its update
/// buffers would be read wrongly.
fn check_revision(params: &Parameters) -> Result<(), Stop> {
    let tag_ok = params.revision & 0x00FF_FFFF == REVISION_TAG;
    let number = (params.revision >> 24) as u8 as char;
    match (tag_ok, number.to_digit(10)) {
        (true, Some(n)) if (1..=LAST_KNOWN_REVISION).contains(&n) => Ok(()),
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
                let out = self.core.update(&data).map_err(Stop)?;
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
    use super::*;

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
        let out = core.update(&input).unwrap();
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
        let out = core.update(&input).unwrap();
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
            seen.push(voice_out(&core.update(&playing()).unwrap(), 3, 0));
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
        assert_eq!(voice_out(&core.update(&idle).unwrap(), 3, 0), (600, 1));
        assert_eq!(
            voice_out(&core.update(&idle).unwrap(), 3, 1),
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
        assert_eq!(voice_out(&core.update(&input).unwrap(), 3, 0), (240, 2));
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
                voice_out(&core.update(&input).unwrap(), 3, 0),
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
        assert_eq!(voice_out(&core.update(&input).unwrap(), 3, 0), (240, 0));
        // A voice at 32 kHz plays two thirds of 240 samples a frame, carrying the fraction.
        let mut core = Core::new(params());
        let mut v = voice(vec![(0, 100_000, false)]);
        v.sample_rate = 32_000;
        let input = update(0x3556_4552, &[0; 3], &[v, voice(vec![])]);
        let played: Vec<u64> = (0..3)
            .map(|_| voice_out(&core.update(&input).unwrap(), 3, 0).0)
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
        core.update(&play).unwrap();
        let mut stopped = voice(vec![(0, 100_000, false)]);
        stopped.state = 1;
        let still = update(0x3556_4552, &[0; 3], &[stopped, voice(vec![])]);
        assert_eq!(voice_out(&core.update(&still).unwrap(), 3, 0), (240, 0));
        let mut fresh = voice(vec![(0, 100_000, false)]);
        fresh.is_new = true;
        let again = update(0x3556_4552, &[0; 3], &[fresh, voice(vec![])]);
        assert_eq!(
            voice_out(&core.update(&again).unwrap(), 3, 0),
            (240, 0),
            "counted from zero"
        );
    }

    #[test]
    fn a_layout_nobody_documented_stops_the_run() {
        let mut core = Core::new(params());
        let mut input = update(0x3556_4552, &[0; 3], &[voice(vec![]), voice(vec![])]);
        input[0x0C..0x10].copy_from_slice(&0x171u32.to_le_bytes());
        let e = core.update(&input).unwrap_err();
        assert!(e.contains("do not have the sizes"), "{e}");
        let e = core.update(&[0; 8]).unwrap_err();
        assert!(e.contains("no header"), "{e}");
        let mut bad = update(0x3556_4552, &[0; 3], &[voice(vec![]), voice(vec![])]);
        bad[0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        assert!(core.update(&bad).unwrap_err().contains("revision"));
    }

    #[test]
    fn revisions_are_checked_against_what_is_known() {
        let with = |revision: u32| Parameters {
            revision,
            ..params()
        };
        assert!(check_revision(&with(0x3156_4552)).is_ok(), "REV1");
        assert!(check_revision(&with(0x3656_4552)).is_ok(), "REV6");
        assert!(
            check_revision(&with(0x3756_4552)).is_err(),
            "REV7 is not known"
        );
        assert!(check_revision(&with(0x4256_4552)).is_err(), "nor REVB");
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
