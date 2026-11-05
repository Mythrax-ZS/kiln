//! The Switch 2's video decoder (`nn::codec::video`, docs/ASSUMPTIONS.md A-261 to A-264): the
//! session the graphics supervisor opens for it (`nx2:gsv` 906, 902 for older SDKs) and the
//! decoder services that session creates.
//!
//! A decoder service takes compressed frames (109, 400) and decodes them with
//! [`nx_gpu::media::vp9`] on a thread of its own, writing each into the next of the output
//! images the program registered (311) or gave back (314), in that order. It counts the images it
//! has written on a syncpoint of its own: each registration and return of an image is answered
//! with the value the syncpoint reaches once that image holds a frame (a `NativeSyncHandle`),
//! which the program waits for, and 213 then says what the image holds.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use super::gsv::NativeObjects;
use super::nvdrv;
use nx_gpu::media::vp9;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

/// What a decoder output image holds (the output code of its preference).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// The luma plane, then the chroma plane at half the height (U and V interleaved).
    Nv12,
    Bgra,
    Rgba,
}

/// The rows each plane of a decoder output image is padded to.
const PLANE_ROWS: u64 = 128;
/// The alignment 991 asks for a pool's memory.
pub const DECODER_IMAGE_ALIGNMENT: u64 = 0x1000;
/// The size of what 961 answers.
pub const CPU_DESCRIPTOR_SIZE: usize = 128;

/// The layout of a decoder output image, from the 16 words of its preference
/// (`nn::codec::MakeDecodeImagePreference`, A-262): the output at word 1 (1 and 4 NV12, 2 and 5
/// BGRA, 3 and 6 RGBA, one of each pair for each of the codec's modes), the width and height less
/// one at word 2. The server chooses the layout and describes it with 961: pitch-linear rows of a
/// byte (NV12) or four a pixel, 64-byte aligned, each plane padded to 128 rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderLayout {
    pub output: Output,
    pub width: u32,
    pub height: u32,
    pub pitch: u64,
    /// Where NV12's chroma plane begins: the size of the first plane.
    pub chroma: u64,
    pub size: u64,
}

impl DecoderLayout {
    pub fn from_preference(words: &[u32]) -> Result<DecoderLayout, String> {
        let word = |n: usize| words.get(n).copied().unwrap_or(0);
        let output = match word(1) {
            1 | 4 => Output::Nv12,
            2 | 5 => Output::Bgra,
            3 | 6 => Output::Rgba,
            other => {
                return Err(format!(
                    "a decoder output image of output code {other} (preference {:x?})",
                    &words[..words.len().min(8)]
                ))
            }
        };
        let (width, height) = ((word(2) & 0xFFFF) + 1, (word(2) >> 16) + 1);
        let bytes = if output == Output::Nv12 { 1 } else { 4 };
        let pitch = (u64::from(width) * bytes).next_multiple_of(64);
        let chroma = pitch * u64::from(height).next_multiple_of(PLANE_ROWS);
        let size = chroma
            + match output {
                Output::Nv12 => pitch * u64::from(height.div_ceil(2)).next_multiple_of(PLANE_ROWS),
                _ => 0,
            };
        Ok(DecoderLayout {
            output,
            width,
            height,
            pitch,
            chroma,
            size,
        })
    }

    /// The `nn::gsv::CpuUserImageDescriptor` of an image `offset` bytes into its pool's memory
    /// (961, A-263): the kind at 0 (8 RGBA, 9 BGRA, 10 NV12), the first plane's offset in the
    /// memory at 8 and its row pitch at 0x10, NV12's chroma plane's at 0x18 and 0x20.
    pub fn cpu_descriptor(&self, offset: u64) -> [u8; CPU_DESCRIPTOR_SIZE] {
        let mut d = [0u8; CPU_DESCRIPTOR_SIZE];
        d[0] = match self.output {
            Output::Rgba => 8,
            Output::Bgra => 9,
            Output::Nv12 => 10,
        };
        let mut put = |at: usize, v: u64| d[at..at + 8].copy_from_slice(&v.to_le_bytes());
        put(0x08, offset);
        put(0x10, self.pitch);
        if self.output == Output::Nv12 {
            put(0x18, offset + self.chroma);
            put(0x20, self.pitch);
        }
        d
    }

    /// The bytes of the image holding `frame` (cropped to the image), the rest zero; the size the
    /// frame was written at.
    pub fn render(&self, frame: &vp9::Frame, color: vp9::Color, out: &mut Vec<u8>) -> (u32, u32) {
        out.clear();
        out.resize(self.size as usize, 0);
        let w = frame.width.min(self.width as usize);
        let h = frame.height.min(self.height as usize);
        let pitch = self.pitch as usize;
        let [y, u, v] = &frame.planes;
        match self.output {
            Output::Nv12 => {
                for row in 0..h {
                    out[row * pitch..row * pitch + w]
                        .copy_from_slice(&y.data[row * y.stride..row * y.stride + w]);
                }
                let chroma = self.chroma as usize;
                for row in 0..h.div_ceil(2) {
                    let line = &mut out[chroma + row * pitch..];
                    for x in 0..w.div_ceil(2) {
                        line[2 * x] = u.at(x, row);
                        line[2 * x + 1] = v.at(x, row);
                    }
                }
            }
            Output::Rgba | Output::Bgra => {
                let m = Matrix::of(color);
                let order = if self.output == Output::Rgba {
                    [0, 1, 2]
                } else {
                    [2, 1, 0]
                };
                for row in 0..h {
                    let line = &mut out[row * pitch..row * pitch + w * 4];
                    for (x, px) in line.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        let rgb = m.rgb(y.at(x, row), u.at(x / 2, row / 2), v.at(x / 2, row / 2));
                        px[0] = rgb[order[0]];
                        px[1] = rgb[order[1]];
                        px[2] = rgb[order[2]];
                        px[3] = 255;
                    }
                }
            }
        }
        (w as u32, h as u32)
    }
}

/// A conversion from Y'CbCr to R'G'B' in 16.16 fixed point (A-264): the luma and chroma
/// coefficients of the stream's color space, with the studio swing scaled out unless the stream
/// says its samples use the full range.
struct Matrix {
    y: i32,
    y_offset: i32,
    r_v: i32,
    g_u: i32,
    g_v: i32,
    b_u: i32,
}

impl Matrix {
    fn of(color: vp9::Color) -> Matrix {
        let (kr, kb) = match color.space {
            vp9::CS_BT_709 => (0.2126, 0.0722),
            vp9::CS_SMPTE_240 => (0.212, 0.087),
            vp9::CS_BT_2020 => (0.2627, 0.0593),
            // Unknown, BT.601 and SMPTE 170.
            _ => (0.299, 0.114),
        };
        let kg = 1.0 - kr - kb;
        let (y_scale, c_scale, y_offset) = if color.full_range {
            (1.0, 1.0, 0)
        } else {
            (255.0 / 219.0, 255.0 / 224.0, 16)
        };
        let fixed = |v: f64| (v * 65536.0).round() as i32;
        Matrix {
            y: fixed(y_scale),
            y_offset,
            r_v: fixed(2.0 * (1.0 - kr) * c_scale),
            g_u: fixed(2.0 * kb * (1.0 - kb) / kg * c_scale),
            g_v: fixed(2.0 * kr * (1.0 - kr) / kg * c_scale),
            b_u: fixed(2.0 * (1.0 - kb) * c_scale),
        }
    }

    fn rgb(&self, y: u8, u: u8, v: u8) -> [u8; 3] {
        let y = (i32::from(y) - self.y_offset) * self.y;
        let (u, v) = (i32::from(u) - 128, i32::from(v) - 128);
        let clamp = |s: i32| ((s + 0x8000) >> 16).clamp(0, 255) as u8;
        [
            clamp(y + self.r_v * v),
            clamp(y - self.g_u * u - self.g_v * v),
            clamp(y + self.b_u * u),
        ]
    }
}

/// `NX_TRACE_VDEC`: every request of the video decoder's sessions and every frame written.
fn tracing() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("NX_TRACE_VDEC").is_some())
}

pub(super) fn trace_request(service: &str, req: &Request) {
    if !tracing() {
        return;
    }
    let words: Vec<String> = (0..req.data.len().min(64) / 4)
        .map(|i| format!("{:08x}", req.u32_at(4 * i)))
        .collect();
    eprintln!(
        "[vdec] {service} {}: {} (send buffers {:x?}, {} copied handles)",
        req.command,
        words.join(" "),
        req.send_buffers,
        req.copy_handles.len()
    );
}

/// Which codec a session opened with 906 or 1002 is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Decoder,
    Encoder,
}

/// The session `nx2:gsv` 906 (902) or 1002 hands out (`nn::codec::video::detail::
/// InitializeServerContext`): its command 1 creates a decoder service (an encoder service for
/// 1002's, which is not implemented).
#[derive(Debug)]
pub struct VideoServer {
    kernel: Weak<Kernel>,
    gpu: Arc<nvdrv::Shared>,
    native: Arc<Mutex<NativeObjects>>,
    codec: Codec,
}

impl VideoServer {
    pub fn new(
        kernel: Weak<Kernel>,
        gpu: Arc<nvdrv::Shared>,
        native: Arc<Mutex<NativeObjects>>,
        codec: Codec,
    ) -> VideoServer {
        VideoServer {
            kernel,
            gpu,
            native,
            codec,
        }
    }
}

impl Service for VideoServer {
    fn name(&self) -> &str {
        match self.codec {
            Codec::Decoder => "nx2:vdec",
            Codec::Encoder => "nx2:venc",
        }
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop(format!(
            "{} was sent a CMIF message (command {}); it speaks TIPC",
            self.name(),
            req.command
        )))
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        trace_request(self.name(), req);
        match (self.codec, req.command) {
            // CreateVideoDecoderService() -> session, from
            // nn::codec::video::detail::CreateVideoDecoderService.
            (Codec::Decoder, 1) => Ok(Response::ok().with_copy_handle(KObject::session(Box::new(
                DecoderService::new(self.kernel.clone(), self.gpu.clone(), self.native.clone()),
            )))),
            (_, other) => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

/// The frames the program may have queued before the submission-ready event stops being set.
const QUEUED_INPUTS: usize = 8;
/// The decoded frames kept while no output image is free.
const DECODED_AHEAD: usize = 2;
/// How long a frame waits for the program's reads of the image it goes into to end (A-261).
const RELEASE_WAIT: Duration = Duration::from_secs(1);

/// A compressed frame the program submitted.
struct Input {
    data: Vec<u8>,
    /// Microseconds.
    pts: u64,
    end: bool,
}

/// A decoded frame waiting for an output image; no picture for the mark of the end of a stream
/// whose last input showed nothing.
struct Decoded {
    picture: Option<Arc<vp9::Frame>>,
    color: vp9::Color,
    pts: u64,
    end: bool,
}

/// What 213 says of an image written.
struct Written {
    image: u32,
    width: u32,
    height: u32,
    pts: u64,
    end: bool,
}

/// The program's reads of an image it gave back end when this syncpoint reaches this value.
#[derive(Clone, Copy, Debug)]
struct Fence {
    syncpoint: u32,
    value: u32,
}

#[derive(Default)]
struct State {
    inputs: VecDeque<Input>,
    decoded: VecDeque<Decoded>,
    /// The images handed out, in the order frames go into them, each with the fence of the
    /// program's reads when it gave it back.
    free: VecDeque<(u32, Option<Fence>)>,
    written: VecDeque<Written>,
    /// The syncpoint the images written are counted on, from the first registration.
    syncpoint: Option<u32>,
    /// Its value then: the n-th image handed out holds a frame once it reaches `base + n`.
    base: u32,
    handed: u32,
    submitted: u64,
    /// The decoder must start again (8, 207): what it decodes now is not kept.
    flush: bool,
    quit: bool,
    /// The writable end of the submission-ready event (3).
    event: Option<KObject>,
    /// Why decoding stopped, said at the next request.
    failed: Option<String>,
}

/// What a decoder service and its thread share.
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    kernel: Weak<Kernel>,
    gpu: Arc<nvdrv::Shared>,
    native: Arc<Mutex<NativeObjects>>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Set the submission-ready event (it is attached clearing itself: set once for every wait
    /// that may pass).
    fn set_event(&self, event: Option<KObject>) {
        if let (Some(event), Some(kernel)) = (event, self.kernel.upgrade()) {
            kernel.set_event(&event, true);
        }
    }
}

/// A decoder service (`nn::codec::video::detail::ipc`'s decoder, A-261).
pub struct DecoderService {
    shared: Arc<Shared>,
    /// The readable end of the submission-ready event.
    readable: Option<KObject>,
}

impl std::fmt::Debug for DecoderService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DecoderService")
    }
}

impl DecoderService {
    pub(super) fn new(
        kernel: Weak<Kernel>,
        gpu: Arc<nvdrv::Shared>,
        native: Arc<Mutex<NativeObjects>>,
    ) -> DecoderService {
        let shared = Arc::new(Shared {
            state: Mutex::default(),
            wake: Condvar::new(),
            kernel,
            gpu,
            native,
        });
        let worker = shared.clone();
        std::thread::Builder::new()
            .name("nx2:vdec".into())
            .spawn(move || decode_thread(worker))
            .expect("the decoder's thread starts");
        DecoderService {
            shared,
            readable: None,
        }
    }

    /// Hand out `image` for a frame to go into (311, 314): the `NativeSyncHandle` that passes
    /// once it holds one.
    fn hand_out(&mut self, image: u32, fence: Option<Fence>) -> ServiceResult {
        let mut st = self.shared.lock();
        let syncpoint = match st.syncpoint {
            Some(s) => s,
            None => {
                let s = self.shared.gpu.host_syncpoint().ok_or_else(|| {
                    Stop("nx2:vdecsvc: no syncpoint is free to count decoded frames on".into())
                })?;
                st.base = self.shared.gpu.syncpoint_value(s).unwrap_or(0);
                st.syncpoint = Some(s);
                s
            }
        };
        st.handed += 1;
        let value = st.base.wrapping_add(st.handed);
        st.free.push_back((image, fence));
        drop(st);
        self.shared.wake.notify_all();
        Ok(Response::ok()
            .with_u32(!syncpoint)
            .with_u32(0)
            .with_u64(u64::from(value)))
    }

    /// Drop the frames queued and not yet written (8, 207, 115); with `complete`, the images
    /// handed out are given back too, the syncpoint moved past each.
    fn drop_queued(&mut self, complete: bool) {
        let mut st = self.shared.lock();
        st.inputs.clear();
        st.decoded.clear();
        st.flush = true;
        let mut passed = 0;
        if complete {
            passed = st.free.len() as u32;
            st.free.clear();
        }
        let (syncpoint, event) = (st.syncpoint, st.event.clone());
        drop(st);
        self.shared.wake.notify_all();
        if let (Some(id), Some(kernel)) = (syncpoint, self.shared.kernel.upgrade()) {
            for _ in 0..passed {
                self.shared.gpu.host_increment(&kernel, id);
            }
        }
        self.shared.set_event(event);
    }

    fn submit(&mut self, req: &Request) -> ServiceResult {
        let kernel = self
            .shared
            .kernel
            .upgrade()
            .ok_or_else(|| Stop("the process is gone".into()))?;
        let buffer = req.input_buffer(0).ok_or_else(|| {
            Stop(format!(
                "nx2:vdecsvc {} was not given the frame's buffer",
                req.command
            ))
        })?;
        let data = kernel.read_buffer(buffer, 64 << 20).ok_or_else(|| {
            Stop(format!(
                "nx2:vdecsvc {}: the frame's buffer at {:#x} ({:#x} bytes) is not readable",
                req.command, buffer.addr, buffer.size
            ))
        })?;
        let input = Input {
            data,
            pts: req.u64_at(0),
            end: req.u32_at(8) & 0xFF != 0,
        };
        let mut st = self.shared.lock();
        st.inputs.push_back(input);
        st.submitted += 1;
        let id = st.submitted;
        let event = (st.inputs.len() < QUEUED_INPUTS)
            .then(|| st.event.clone())
            .flatten();
        drop(st);
        self.shared.wake.notify_all();
        self.shared.set_event(event);
        Ok(Response::ok().with_u64(id))
    }
}

impl Drop for DecoderService {
    fn drop(&mut self) {
        self.shared.lock().quit = true;
        self.shared.wake.notify_all();
    }
}

impl Service for DecoderService {
    fn name(&self) -> &str {
        "nx2:vdecsvc"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop(format!(
            "nx2:vdecsvc was sent a CMIF message (command {}); it speaks TIPC",
            req.command
        )))
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        trace_request(self.name(), req);
        if let Some(why) = self.shared.lock().failed.take() {
            return Err(Stop(format!("nx2:vdecsvc: {why}")));
        }
        match req.command {
            // Initialize(work buffer transfer memory, u64 size, ...): the buffer is the
            // decoder's to work in; this one needs none of it (A-261).
            200..=202 => {
                let kernel = self
                    .shared
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".into()))?;
                let lent = req
                    .copy_handles
                    .first()
                    .and_then(|o| kernel.transfer_memory(o));
                if lent.is_none() {
                    return Err(Stop(format!(
                        "nx2:vdecsvc {} was not given its work buffer's transfer memory",
                        req.command
                    )));
                }
                Ok(Response::ok())
            }
            // Configure(A: codec name, u32 width, u32 height, u32 rate, ...).
            101 => {
                let kernel = self
                    .shared
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".into()))?;
                let name = req
                    .input_buffer(0)
                    .and_then(|b| kernel.read_buffer(b, 64))
                    .unwrap_or_default();
                let name: String = name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as char)
                    .collect();
                if name != "video/vp9" {
                    return Err(Stop(format!(
                        "nx2:vdecsvc 101 configures the codec `{name}`; only `video/vp9` is \
                         decoded"
                    )));
                }
                Ok(Response::ok())
            }
            // GetSubmissionReadyEvent() -> event: set while frames may be submitted.
            3 => {
                if self.readable.is_none() {
                    let kernel = self
                        .shared
                        .kernel
                        .upgrade()
                        .ok_or_else(|| Stop("the process is gone".into()))?;
                    let (writable, readable) = kernel.new_event();
                    self.shared.lock().event = Some(writable.clone());
                    self.readable = Some(readable);
                    kernel.set_event(&writable, true);
                }
                Ok(Response::ok().with_copy_handle(self.readable.clone().expect("made above")))
            }
            // Start.
            6 => Ok(Response::ok()),
            // Flush: the frames queued go, the images handed out keep their places.
            8 => {
                self.drop_queued(false);
                Ok(Response::ok())
            }
            // Stop and Finalize: the frames queued go and the images handed out come back (A-261).
            207 | 115 => {
                self.drop_queued(true);
                Ok(Response::ok())
            }
            // SubmitInputBuffer(A: frame, u64 pts in microseconds, u8 end of stream) -> u64; 400
            // adds a second buffer, of information about the frame, which is not needed.
            109 | 400 => self.submit(req),
            // RegisterOutputBuffer(u32 image, ...) -> NativeSyncHandle.
            311 => self.hand_out(req.u32_at(0), None),
            // ReleaseOutputBuffer(u32 image, u32, NativeSyncHandle of the program's reads) ->
            // NativeSyncHandle.
            314 => {
                let mode = req.u32_at(12);
                if mode != 0 {
                    return Err(Stop(format!(
                        "nx2:vdecsvc 314 gives back the image {} with a sync handle of mode {mode}",
                        req.u32_at(0)
                    )));
                }
                let fence = Fence {
                    syncpoint: !req.u32_at(8),
                    value: req.u64_at(16) as u32,
                };
                self.hand_out(req.u32_at(0), Some(fence))
            }
            // UnregisterOutputBuffer(u32 image).
            212 => {
                let image = req.u32_at(0);
                self.shared.lock().free.retain(|(i, _)| *i != image);
                Ok(Response::ok())
            }
            // GetOutputBufferProperties() -> (u32 image, u32 width, u32 height, u32 width, u32
            // height, u32, u64 pts in nanoseconds, u8 end of stream): of the next image written.
            213 => {
                let next = self.shared.lock().written.pop_front();
                match next {
                    Some(w) => Ok(Response::ok()
                        .with_u32(w.image)
                        .with_u32(w.width)
                        .with_u32(w.height)
                        .with_u32(w.width)
                        .with_u32(w.height)
                        .with_u32(0)
                        .with_u64(w.pts.saturating_mul(1000))
                        .with_u32(u32::from(w.end))),
                    None => Err(Stop(
                        "nx2:vdecsvc 213 asks what an image holds before any was written".into(),
                    )),
                }
            }
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

/// Wait, for at most [`RELEASE_WAIT`], for the program's reads of an image to end.
fn wait_for_release(gpu: &nvdrv::Shared, fence: Fence) {
    if fence.syncpoint == 0 || fence.syncpoint as usize >= nx_gpu::syncpt::COUNT {
        return;
    }
    let deadline = Instant::now() + RELEASE_WAIT;
    while !gpu
        .syncpoint_value(fence.syncpoint)
        .is_none_or(|v| nx_gpu::syncpt::reached(v, fence.value))
    {
        if Instant::now() >= deadline {
            if tracing() {
                eprintln!(
                    "[vdec] the reads of an image did not end in {RELEASE_WAIT:?} (syncpoint {} \
                     to {})",
                    fence.syncpoint, fence.value
                );
            }
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The thread of a decoder service: compressed frames into decoded ones, decoded frames into the
/// output images, in order.
fn decode_thread(shared: Arc<Shared>) {
    let mut decoder = vp9::Decoder::new();
    let mut bytes = Vec::new();
    let mut frames = 0u64;
    let mut st = shared.lock();
    loop {
        st = shared
            .wake
            .wait_while(st, |s| {
                !s.quit
                    && !s.flush
                    && !(!s.decoded.is_empty() && !s.free.is_empty())
                    && !(!s.inputs.is_empty() && s.decoded.len() < DECODED_AHEAD)
            })
            .unwrap_or_else(|e| e.into_inner());
        if st.quit {
            break;
        }
        if st.flush {
            st.flush = false;
            st.decoded.clear();
            decoder = vp9::Decoder::new();
            continue;
        }
        if !st.decoded.is_empty() && !st.free.is_empty() {
            let frame = st.decoded.pop_front().expect("not empty");
            let (image, fence) = st.free.pop_front().expect("not empty");
            drop(st);
            if let Some(fence) = fence {
                wait_for_release(&shared.gpu, fence);
            }
            let size = write_image(&shared, image, &frame, &mut bytes);
            st = shared.lock();
            let (width, height) = match size {
                Ok(size) => size,
                Err(why) => {
                    eprintln!("nx2:vdecsvc: {why}");
                    st.failed = Some(why);
                    (0, 0)
                }
            };
            frames += 1;
            dump_frame(&frame, frames);
            if tracing() {
                eprintln!(
                    "[vdec] frame {frames} ({width}x{height}, pts {} us{}) into image {image}",
                    frame.pts,
                    if frame.end { ", the end" } else { "" }
                );
            }
            st.written.push_back(Written {
                image,
                width,
                height,
                pts: frame.pts,
                end: frame.end,
            });
            let (syncpoint, event) = (st.syncpoint, st.event.clone());
            drop(st);
            if let (Some(id), Some(kernel)) = (syncpoint, shared.kernel.upgrade()) {
                shared.gpu.host_increment(&kernel, id);
            }
            shared.set_event(event);
            st = shared.lock();
            continue;
        }
        let input = st.inputs.pop_front().expect("not empty");
        let event = (st.inputs.len() < QUEUED_INPUTS)
            .then(|| st.event.clone())
            .flatten();
        drop(st);
        shared.set_event(event);
        let started = Instant::now();
        let result = if input.data.is_empty() {
            Ok(Vec::new())
        } else {
            decoder.decode(&input.data)
        };
        let color = decoder.color();
        st = shared.lock();
        if tracing() {
            eprintln!(
                "[vdec] decoded {} bytes (pts {} us) in {:.1} ms",
                input.data.len(),
                input.pts,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        if st.flush {
            continue;
        }
        match result {
            Ok(shown) => st.decoded.extend(shown.into_iter().map(|picture| Decoded {
                picture: Some(picture),
                color,
                pts: input.pts,
                end: false,
            })),
            Err(why) => {
                let why = format!("decoding the frame at {} us: {why:?}", input.pts);
                eprintln!("nx2:vdecsvc: {why}");
                st.failed = Some(why);
            }
        }
        if input.end {
            match st.decoded.back_mut() {
                Some(last) => last.end = true,
                None => st.decoded.push_back(Decoded {
                    picture: None,
                    color,
                    pts: input.pts,
                    end: true,
                }),
            }
        }
    }
    if let Some(id) = st.syncpoint {
        drop(st);
        shared.gpu.host_release(id);
    }
}

/// `NX_VDEC_DUMP=<dir>`: every 60th frame decoded, as the decoder made it (converted to RGB as
/// an RGBA image would hold it), apart from what the program does with the images.
#[cfg(feature = "vulkan")]
fn dump_frame(frame: &Decoded, n: u64) {
    let Some(dir) = std::env::var_os("NX_VDEC_DUMP") else {
        return;
    };
    let Some(picture) = &frame.picture else {
        return;
    };
    if n % 60 != 1 {
        return;
    }
    let mut preference = [0u32; 16];
    preference[1] = 3;
    preference[2] = (picture.width as u32 - 1) | (picture.height as u32 - 1) << 16;
    let Ok(layout) = DecoderLayout::from_preference(&preference) else {
        return;
    };
    let mut bytes = Vec::new();
    let (w, h) = layout.render(picture, frame.color, &mut bytes);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        let at = row * layout.pitch as usize;
        rgba.extend_from_slice(&bytes[at..at + w as usize * 4]);
    }
    let _ = std::fs::create_dir_all(&dir);
    let path = std::path::Path::new(&dir).join(format!("vdec_{n:05}.png"));
    if let Err(e) = nx_vulkan::png::write_opaque_png(&path, w, h, &rgba) {
        eprintln!("[vdec] writing {}: {e}", path.display());
    }
}

#[cfg(not(feature = "vulkan"))]
fn dump_frame(_frame: &Decoded, _n: u64) {}

/// Write a decoded frame into an output image; the size it was written at.
fn write_image(
    shared: &Shared,
    image: u32,
    frame: &Decoded,
    bytes: &mut Vec<u8>,
) -> Result<(u32, u32), String> {
    let found = shared
        .native
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .decoder_image_address(image);
    let Some((address, layout)) = found else {
        return Err(format!(
            "the output image {image} is not a decoder image whose memory is bound"
        ));
    };
    let Some(picture) = &frame.picture else {
        return Ok((0, 0));
    };
    let size = layout.render(picture, frame.color, bytes);
    let kernel = shared
        .kernel
        .upgrade()
        .ok_or_else(|| "the process is gone".to_string())?;
    if !kernel.write_lent_memory(address, bytes) {
        return Err(format!(
            "the output image {image} at {address:#x} ({:#x} bytes) is not mapped",
            bytes.len()
        ));
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preference(code: u32, width: u32, height: u32) -> Vec<u32> {
        let mut w = vec![0u32; 16];
        w[1] = code;
        w[2] = (width - 1) | (height - 1) << 16;
        w
    }

    #[test]
    fn a_decoder_image_is_laid_out_in_rows_with_nv12_chroma_after_the_luma() {
        let l = DecoderLayout::from_preference(&preference(1, 1920, 1080)).unwrap();
        assert_eq!((l.output, l.width, l.height), (Output::Nv12, 1920, 1080));
        assert_eq!(l.pitch, 1920);
        assert_eq!(
            l.chroma,
            1920 * 1152,
            "the luma plane is padded to 128 rows"
        );
        assert_eq!(l.size, 1920 * 1152 + 1920 * 640);
        let l = DecoderLayout::from_preference(&preference(6, 100, 50)).unwrap();
        assert_eq!((l.output, l.pitch, l.size), (Output::Rgba, 448, 448 * 128));
        assert!(DecoderLayout::from_preference(&preference(10, 64, 64)).is_err());
        let d = l.cpu_descriptor(0x2000);
        assert_eq!(d[0], 8);
        assert_eq!(u64::from_le_bytes(d[8..16].try_into().unwrap()), 0x2000);
        assert_eq!(u64::from_le_bytes(d[16..24].try_into().unwrap()), 448);
    }

    #[test]
    fn a_frame_is_written_as_nv12_and_as_rgba() {
        let mut f = vp9::Frame::new(4, 2);
        for (i, p) in f.planes[0].data.iter_mut().enumerate() {
            *p = i as u8;
        }
        f.planes[1].data.fill(100);
        f.planes[2].data.fill(200);
        let nv12 = DecoderLayout::from_preference(&preference(1, 4, 2)).unwrap();
        let mut out = Vec::new();
        assert_eq!(nv12.render(&f, vp9::Color::default(), &mut out), (4, 2));
        let stride = f.planes[0].stride as u8;
        assert_eq!(&out[..4], &[0, 1, 2, 3]);
        assert_eq!(&out[64..68], &[stride, stride + 1, stride + 2, stride + 3]);
        let c = nv12.chroma as usize;
        assert_eq!(&out[c..c + 4], &[100, 200, 100, 200]);
        // Studio-swing white, black and a BT.601 red.
        let mut g = vp9::Frame::new(2, 2);
        g.planes[0].data[0] = 235;
        g.planes[0].data[1] = 16;
        g.planes[1].data.fill(128);
        g.planes[2].data.fill(128);
        let rgba = DecoderLayout::from_preference(&preference(3, 2, 2)).unwrap();
        rgba.render(&g, vp9::Color::default(), &mut out);
        assert_eq!(&out[..8], &[255, 255, 255, 255, 0, 0, 0, 255]);
        let bgra = DecoderLayout::from_preference(&preference(2, 2, 2)).unwrap();
        g.planes[0].data[0] = 81;
        g.planes[1].data.fill(90);
        g.planes[2].data.fill(240);
        bgra.render(&g, vp9::Color::default(), &mut out);
        let (b, gr, r) = (out[0], out[1], out[2]);
        assert!(r >= 253 && gr <= 2 && b <= 2, "red: {r} {gr} {b}");
    }
}
