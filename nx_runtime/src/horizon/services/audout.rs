//! `audout:u`, audio output (Switchbrew, "Audio services"): a program hands over buffers of
//! interleaved 16-bit samples and gets each one back when the time it takes has been played
//! (docs/ASSUMPTIONS.md A-175).
//!
//! Every output has a thread that plays the appended buffers one after the other in real time,
//! while the output is started: the samples go to the sound device (see
//! [`super::super::audio_out`]), the buffer's tag moves to the released ones when its duration is
//! over, and the buffer event is set so that the program's audio thread goes on.

use super::super::audio_out::Outputs;
use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Buffer, Request, Response, Service, ServiceResult, Stop};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

/// The size of an output device's name, and the one device there is.
const NAME_SIZE: usize = 0x20;
const DEVICE_NAME: &str = "DeviceOut";
/// What the device plays: 48 kHz, two channels of 16-bit samples.
const DEVICE_RATE: u32 = 48_000;
const DEVICE_CHANNELS: u32 = 2;
const PCM_INT16: u32 = 2;

/// `nn::audio::AudioOutState`.
const STARTED: u32 = 0;
const STOPPED: u32 = 1;

/// An `AudioOutBuffer` as `libnx` lays it out: the next one (unused), the sample buffer, its
/// size, the size of the data in it and the offset of the data.
const BUFFER_DESCRIPTOR: usize = 0x28;

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

fn u64_in(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8).map_or(0, |s| {
        u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
    })
}

/// `nn::audio::detail::IAudioOutManager` (`audout:u`).
#[derive(Debug)]
pub struct Manager {
    kernel: Weak<Kernel>,
}

impl Manager {
    pub fn new(kernel: Weak<Kernel>) -> Manager {
        Manager { kernel }
    }

    /// Write the one device name into the output buffer of the request (a receive buffer, or the
    /// caller's pointer buffer) and say whether it went through the pointer buffer.
    fn write_name(&self, req: &Request) -> Result<bool, Stop> {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let (buffer, pointer) = match (req.output_buffer(0), req.pointer_output(0)) {
            (Some(b), _) => (b, false),
            (None, Some(p)) => (p, true),
            (None, None) => return Err(Stop("an audio output name without a buffer".into())),
        };
        if (buffer.size as usize) < NAME_SIZE {
            return Err(Stop("the audio output name buffer is too small".into()));
        }
        let mut bytes = vec![0u8; NAME_SIZE];
        bytes[..DEVICE_NAME.len()].copy_from_slice(DEVICE_NAME.as_bytes());
        if !kernel.write_memory(buffer.addr, &bytes) {
            return Err(Stop("the audio output name buffer is not writable".into()));
        }
        Ok(pointer)
    }
}

impl Service for Manager {
    fn name(&self) -> &str {
        "nn::audio::detail::IAudioOutManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // ListAudioOuts() -> (buffer of names, u32 count), and its automatic form
            0 | 2 => {
                let pointer = self.write_name(req)?;
                let mut r = Response::ok().with_u32(1);
                if pointer {
                    r = r.with_pointer_reply(0, NAME_SIZE as u64);
                }
                Ok(r)
            }
            // OpenAudioOut(u32 sample rate, u16 channels, u16 padding, u64 aruid, pid, process,
            // buffer name) -> (u32 rate, u32 channels, u32 format, u32 state, IAudioOut,
            // buffer name), and its automatic form
            1 | 3 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let wanted_rate = req.u32_at(0);
                let wanted_channels = u32::from(req.u32_at(4) as u16);
                let rate = if wanted_rate == 0 {
                    DEVICE_RATE
                } else {
                    wanted_rate
                };
                let channels = if wanted_channels == 0 {
                    DEVICE_CHANNELS
                } else {
                    wanted_channels
                };
                if std::env::var_os("NX_TRACE_SVC").is_some() {
                    eprintln!("[audout] OpenAudioOut {wanted_rate} Hz, {wanted_channels} channels");
                }
                let pointer = self.write_name(req)?;
                let out = AudioOut::new(
                    &kernel,
                    rate,
                    channels,
                    Outputs::open_labeled(rate, "audout"),
                );
                let mut r = Response::ok()
                    .with_u32(rate)
                    .with_u32(channels)
                    .with_u32(PCM_INT16)
                    .with_u32(STOPPED)
                    .with_object(Box::new(out));
                if pointer {
                    r = r.with_pointer_reply(0, NAME_SIZE as u64);
                }
                Ok(r)
            }
            other => Err(Stop::unknown_command("audout:u", other)),
        }
    }
}

/// A buffer the program appended, as the samples it holds.
#[derive(Debug)]
struct Appended {
    tag: u64,
    /// Interleaved stereo samples, at the rate of the output.
    samples: Vec<i16>,
}

#[derive(Debug, Default)]
struct State {
    started: bool,
    closed: bool,
    /// Appended and not yet played.
    queue: VecDeque<Appended>,
    /// The buffer being played.
    playing: Option<u64>,
    /// Played and not yet taken by the program.
    released: VecDeque<u64>,
    /// Frames played so far.
    played: u64,
    volume: f32,
}

/// What the output and its thread share.
#[derive(Debug)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// `nn::audio::detail::IAudioOut`.
#[derive(Debug)]
struct AudioOut {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
    channels: u32,
    /// The event the thread sets when a buffer has been played: (writable, readable).
    event: (KObject, KObject),
}

impl AudioOut {
    fn new(kernel: &Arc<Kernel>, rate: u32, channels: u32, outputs: Outputs) -> AudioOut {
        let event = kernel.new_event();
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                volume: 1.0,
                ..State::default()
            }),
            wake: Condvar::new(),
        });
        let (weak, writable, share) = (Arc::downgrade(kernel), event.0.clone(), shared.clone());
        std::thread::spawn(move || play(&weak, &share, &writable, rate, outputs));
        AudioOut {
            kernel: Arc::downgrade(kernel),
            shared,
            channels,
            event,
        }
    }

    /// The samples of the buffer a descriptor names, as interleaved stereo.
    fn read_buffer(&self, descriptor: Buffer) -> Result<Vec<i16>, Stop> {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let mut raw = vec![0u8; (descriptor.size as usize).min(BUFFER_DESCRIPTOR)];
        if raw.len() < BUFFER_DESCRIPTOR || !kernel.read_memory(descriptor.addr, &mut raw) {
            return Err(Stop(
                "an audio output buffer descriptor is not readable".into(),
            ));
        }
        let (at, size, offset) = (u64_in(&raw, 8), u64_in(&raw, 0x18), u64_in(&raw, 0x20));
        let mut bytes = vec![0u8; size as usize];
        if !kernel.read_memory(at.wrapping_add(offset), &mut bytes) {
            return Err(Stop(format!(
                "the samples of an audio output buffer ({size:#x} bytes at {:#x}) are not readable",
                at.wrapping_add(offset)
            )));
        }
        let all: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b))
            .collect();
        Ok(stereo(&all, self.channels as usize))
    }

    fn append(&mut self, req: &Request) -> ServiceResult {
        let tag = req.u64_at(0);
        let Some(descriptor) = req.input_buffer(0) else {
            return Err(Stop("AppendAudioOutBuffer without a buffer".into()));
        };
        let samples = self.read_buffer(descriptor)?;
        let mut s = self.shared.lock();
        s.queue.push_back(Appended { tag, samples });
        drop(s);
        self.shared.wake.notify_all();
        Ok(Response::ok())
    }

    /// Hand the released tags to the program, as many as its buffer takes.
    fn released(&mut self, req: &Request) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let Some(buffer) = req.output_buffer(0) else {
            return Err(Stop("GetReleasedAudioOutBuffers without a buffer".into()));
        };
        let mut s = self.shared.lock();
        let fit = (buffer.size as usize / 8).min(s.released.len());
        let tags: Vec<u8> = s
            .released
            .drain(..fit)
            .flat_map(|t| t.to_le_bytes())
            .collect();
        if s.released.is_empty() {
            kernel.set_event(&self.event.0, false);
        }
        drop(s);
        if !tags.is_empty() && !kernel.write_memory(buffer.addr, &tags) {
            return Err(Stop("the released audio buffers are not writable".into()));
        }
        Ok(Response::ok().with_u32(fit as u32))
    }
}

impl Service for AudioOut {
    fn name(&self) -> &str {
        "nn::audio::detail::IAudioOut"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetAudioOutState() -> u32
            0 => {
                let started = self.shared.lock().started;
                Ok(Response::ok().with_u32(if started { STARTED } else { STOPPED }))
            }
            // StartAudioOut(), StopAudioOut()
            1 | 2 => {
                self.shared.lock().started = req.command == 1;
                self.shared.wake.notify_all();
                if std::env::var_os("NX_TRACE_AUDIO").is_some() {
                    let what = if req.command == 1 {
                        "started"
                    } else {
                        "stopped"
                    };
                    eprintln!("[audout] {what}");
                }
                Ok(Response::ok())
            }
            // AppendAudioOutBuffer(u64 tag, buffer descriptor), and its automatic form
            3 | 7 => self.append(req),
            // RegisterBufferEvent() -> event
            4 => Ok(Response::ok().with_copy_handle(self.event.1.clone())),
            // GetReleasedAudioOutBuffers() -> (buffer of tags, u32 count), and its automatic form
            5 | 8 => self.released(req),
            // ContainsAudioOutBuffer(u64 tag) -> bool
            6 => {
                let tag = req.u64_at(0);
                let s = self.shared.lock();
                let has = s.playing == Some(tag)
                    || s.queue.iter().any(|b| b.tag == tag)
                    || s.released.contains(&tag);
                Ok(Response::ok().with_u32(u32::from(has)))
            }
            // GetAudioOutBufferCount() -> u32: the buffers the program has not got back
            9 => {
                let s = self.shared.lock();
                let count = s.queue.len() + s.released.len() + usize::from(s.playing.is_some());
                Ok(Response::ok().with_u32(count as u32))
            }
            // GetAudioOutPlayedSampleCount() -> u64
            10 => Ok(Response::ok().with_u64(self.shared.lock().played)),
            // FlushAudioOutBuffers() -> bool: every buffer not played yet is given back
            11 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let mut s = self.shared.lock();
                let any = !s.queue.is_empty();
                while let Some(b) = s.queue.pop_front() {
                    s.released.push_back(b.tag);
                }
                drop(s);
                if any {
                    kernel.set_event(&self.event.0, true);
                }
                Ok(Response::ok().with_u32(u32::from(any)))
            }
            // SetAudioOutVolume(f32), GetAudioOutVolume() -> f32
            12 => {
                self.shared.lock().volume = f32::from_bits(req.u32_at(0));
                Ok(Response::ok())
            }
            13 => Ok(Response::ok().with_u32(self.shared.lock().volume.to_bits())),
            other => Err(Stop::unknown_command("IAudioOut", other)),
        }
    }
}

impl Drop for AudioOut {
    fn drop(&mut self) {
        self.shared.lock().closed = true;
        self.shared.wake.notify_all();
    }
}

/// Interleaved samples of `channels` channels as interleaved stereo: a mono signal on both sides,
/// the first two channels of anything else but six, which are mixed down (front left, front
/// right, centre, low frequency, rear left, rear right).
fn stereo(samples: &[i16], channels: usize) -> Vec<i16> {
    let clamp = |v: f32| v.round().clamp(-32768.0, 32767.0) as i16;
    match channels {
        0 | 1 => samples.iter().flat_map(|&s| [s, s]).collect(),
        2 => samples.to_vec(),
        6 => samples
            .as_chunks::<6>()
            .0
            .iter()
            .flat_map(|f| {
                let (c, l, r) = (0.707 * f32::from(f[2]), f32::from(f[0]), f32::from(f[1]));
                let (rl, rr) = (0.707 * f32::from(f[4]), 0.707 * f32::from(f[5]));
                [clamp(l + c + rl), clamp(r + c + rr)]
            })
            .collect(),
        n => samples.chunks_exact(n).flat_map(|f| [f[0], f[1]]).collect(),
    }
}

/// The thread of an output: play what is appended while the output is started, in real time.
fn play(
    kernel: &Weak<Kernel>,
    shared: &Shared,
    writable: &KObject,
    rate: u32,
    mut outputs: Outputs,
) {
    // When the buffer that is playing ends: buffers follow each other without a gap, and after a
    // pause the clock starts again.
    let mut ends = Instant::now();
    // `NX_TRACE_AUDIO`: buffers and frames played since the last line, and the loudest sample.
    let trace = std::env::var_os("NX_TRACE_AUDIO").is_some();
    let (mut said_buffers, mut said_frames, mut said_peak) = (0u32, 0u64, 0i32);
    loop {
        let item = {
            let mut s = shared.lock();
            loop {
                if s.closed {
                    return;
                }
                if s.started && !s.queue.is_empty() {
                    break;
                }
                s = shared
                    .wake
                    .wait_timeout(s, Duration::from_millis(100))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
                if kernel.upgrade().is_none_or(|k| k.exit_state().is_some()) {
                    return;
                }
            }
            let item = s.queue.pop_front().expect("the queue is not empty");
            s.playing = Some(item.tag);
            (item, s.volume)
        };
        let (item, volume) = item;
        if !outputs.is_empty() {
            if volume < 1.0 {
                let scaled: Vec<i16> = item
                    .samples
                    .iter()
                    .map(|&v| (f32::from(v) * volume) as i16)
                    .collect();
                outputs.write(&scaled);
            } else {
                outputs.write(&item.samples);
            }
        }
        let frames = (item.samples.len() / 2) as u64;
        if trace {
            said_buffers += 1;
            said_frames += frames;
            let peak = item.samples.iter().map(|s| i32::from(*s).abs()).max();
            said_peak = said_peak.max(peak.unwrap_or(0));
            if said_frames >= u64::from(rate) {
                eprintln!(
                    "[audout] played {said_buffers} buffers, {said_frames} frames, loudest \
                     sample {said_peak} (volume {volume})"
                );
                (said_buffers, said_frames, said_peak) = (0, 0, 0);
            }
        }
        let now = Instant::now();
        let start = ends.max(now);
        ends = start + Duration::from_secs_f64(frames as f64 / f64::from(rate.max(1)));
        // Wait for the end of the buffer in slices, so that a closed output does not hold on.
        loop {
            let left = ends.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            std::thread::sleep(left.min(Duration::from_millis(20)));
            if shared.lock().closed {
                return;
            }
        }
        {
            let mut s = shared.lock();
            s.playing = None;
            s.released.push_back(item.tag);
            s.played += frames;
        }
        let Some(kernel) = kernel.upgrade() else {
            return;
        };
        kernel.set_event(writable, true);
    }
}
