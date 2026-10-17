//! The display services: `vi:u` and the interfaces behind it (Switchbrew, "Display services" and
//! "Nvnflinger services").
//!
//! `vi:u` hands out the application display service; the display service hands out displays and
//! layers and the binder driver, whose transactions are the Android buffer queue of a layer. The
//! display itself is `nx_gpu::display`; this is the IPC around it, and the kernel events it uses.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use nx_gpu::display::Compositor;
use nx_gpu::event::EventToken;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// `vi` module 114, description 5: the command is not available on this session (Switchbrew).
const NOT_AVAILABLE: ResultCode = ResultCode::new(114, 5);

/// The console's display refreshes at 60 Hz: the vsync event is set that often (A-059), and a frame
/// queued with swap interval `n` is on screen for `n` such periods (A-181).
const VSYNC_PERIOD: Duration = Duration::from_micros(16_667);

/// The display state of the process and the events behind its tokens.
pub struct Shared {
    comp: Mutex<Compositor>,
    events: Mutex<HashMap<EventToken, (KObject, KObject)>>,
    vsync_running: AtomicBool,
    /// Stop after this many presented frames (`NX_MAX_FRAMES`), for measuring.
    frame_limit: Option<u64>,
    /// The driver, which tells the renderer which buffer is on screen.
    gpu: Mutex<Option<Arc<super::nvdrv::Shared>>>,
    /// When the call that queued the last frame returned, if frames are paced as the display
    /// shows them (A-181; `NX_UNPACED=1` lets them go as fast as the program makes them).
    pacing: Option<Mutex<Option<Instant>>>,
}

impl Shared {
    pub fn new() -> Shared {
        Shared {
            comp: Mutex::new(Compositor::new()),
            events: Mutex::new(HashMap::new()),
            vsync_running: AtomicBool::new(false),
            frame_limit: std::env::var("NX_MAX_FRAMES")
                .ok()
                .and_then(|v| v.parse().ok()),
            gpu: Mutex::new(None),
            pacing: std::env::var_os("NX_UNPACED")
                .is_none_or(|v| v == "0")
                .then(|| Mutex::new(None)),
        }
    }

    /// Hold the call that queued a frame of swap interval `interval` until the frame before has
    /// had its time on screen: the call returns no sooner than `interval` vsync periods after the
    /// one before it, so a program shows at most 60 / `interval` frames a second, as on the
    /// console, whatever the refresh rate of the machine's screen. A program that is late is not
    /// held, and does not catch up. Interval 0 is not held.
    fn pace(&self, interval: u32) {
        let Some(last) = &self.pacing else {
            return;
        };
        if interval == 0 {
            return;
        }
        let mut last = last.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let due = last.map_or(now, |t| t + VSYNC_PERIOD * interval);
        if due > now {
            std::thread::sleep(due - now);
            *last = Some(due);
        } else {
            *last = Some(now);
        }
    }

    /// Connect the display to the driver, so displayed buffers reach the renderer.
    pub fn set_gpu(&self, gpu: Arc<super::nvdrv::Shared>) {
        *self.gpu.lock().unwrap_or_else(|e| e.into_inner()) = Some(gpu);
    }

    fn gpu(&self) -> Option<Arc<super::nvdrv::Shared>> {
        self.gpu.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn comp(&self) -> std::sync::MutexGuard<'_, Compositor> {
        self.comp.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The (writable, readable) ends of the event behind `token`, made on first use.
    fn event(&self, kernel: &Kernel, token: EventToken) -> (KObject, KObject) {
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events
            .entry(token)
            .or_insert_with(|| kernel.new_event())
            .clone()
    }

    fn deliver_signals(&self, kernel: &Kernel) {
        let signals = self.comp().take_signals();
        for s in signals {
            let (writable, _) = self.event(kernel, s.token);
            kernel.set_event(&writable, s.signaled);
        }
    }

    /// Start the thread that sets the vsync events, once.
    fn start_vsync(self: &Arc<Shared>, kernel: &Arc<Kernel>) {
        if self.vsync_running.swap(true, Ordering::AcqRel) {
            return;
        }
        let (shared, kernel) = (Arc::clone(self), Arc::downgrade(kernel));
        std::thread::spawn(move || loop {
            std::thread::sleep(VSYNC_PERIOD);
            let Some(kernel) = kernel.upgrade() else {
                break;
            };
            if kernel.exit_state().is_some() {
                break;
            }
            shared.comp().tick_vsync();
            shared.deliver_signals(&kernel);
        });
    }
}

impl Default for Shared {
    fn default() -> Shared {
        Shared::new()
    }
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shared")
    }
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// A `DisplayName`: 0x40 bytes, null terminated.
fn display_name(bytes: &[u8]) -> String {
    let raw = bytes.get(..0x40).unwrap_or(bytes);
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

/// `nn::visrv::sf::IApplicationRootService` (`vi:u`).
#[derive(Debug)]
pub struct RootService {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl RootService {
    pub fn new(kernel: Weak<Kernel>, shared: Arc<Shared>) -> RootService {
        RootService { kernel, shared }
    }
}

impl Service for RootService {
    fn name(&self) -> &str {
        "nn::visrv::sf::IApplicationRootService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetDisplayService(u32 privilege) -> IApplicationDisplayService. 0 is an application.
            // GetDisplayServiceWithProxyNameExchange(u32 privilege) -> IApplicationDisplayService
            // [9.0.0+] is the same call here (docs/ASSUMPTIONS.md A-135).
            0 | 1 => {
                if req.u32_at(0) != 0 {
                    return Ok(Response::error(NOT_AVAILABLE));
                }
                Ok(Response::ok().with_object(Box::new(DisplayService {
                    kernel: self.kernel.clone(),
                    shared: self.shared.clone(),
                })))
            }
            other => Err(Stop::unknown_command("vi:u", other)),
        }
    }
}

/// `nn::visrv::sf::IApplicationDisplayService`.
#[derive(Debug)]
struct DisplayService {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl DisplayService {
    /// Write a native window parcel to the output buffer; returns its size.
    fn write_window(&self, req: &Request, window: &[u8]) -> Result<u64, Stop> {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let Some(out) = req.output_buffer(0) else {
            return Err(Stop("a layer was opened without an output buffer".into()));
        };
        if (window.len() as u64) > out.size || !kernel.write_memory(out.addr, window) {
            return Err(Stop(format!(
                "the native window ({} bytes) does not fit the {}-byte output buffer",
                window.len(),
                out.size
            )));
        }
        Ok(window.len() as u64)
    }
}

impl Service for DisplayService {
    fn name(&self) -> &str {
        "nn::visrv::sf::IApplicationDisplayService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetRelayService() -> IHOSBinderDriver
            100 => Ok(Response::ok().with_object(Box::new(BinderDriver {
                kernel: self.kernel.clone(),
                shared: self.shared.clone(),
            }))),
            // The system and manager displays and the indirect binder are not available to an
            // application session (Switchbrew: such commands answer 0xA72).
            101..=103 => Ok(Response::error(NOT_AVAILABLE)),
            // ListDisplays() -> (u64 count, buffer of DisplayInfo)
            1000 => {
                let displays = self.shared.comp().list_displays();
                let mut bytes = Vec::new();
                for d in &displays {
                    let mut info = [0u8; 0x60];
                    info[..d.name.len()].copy_from_slice(d.name.as_bytes());
                    info[0x40] = 1; // has a layer limit
                    info[0x48..0x50].copy_from_slice(&d.layer_count_max.to_le_bytes());
                    info[0x50..0x58].copy_from_slice(&d.width_max.to_le_bytes());
                    info[0x58..0x60].copy_from_slice(&d.height_max.to_le_bytes());
                    bytes.extend_from_slice(&info);
                }
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                if let Some(out) = req.output_buffer(0) {
                    let n = bytes.len().min(out.size as usize);
                    kernel.write_memory(out.addr, &bytes[..n]);
                }
                Ok(Response::ok().with_u64(displays.len() as u64))
            }
            // OpenDisplay(DisplayName) -> u64 display id
            1010 => {
                let name = display_name(&req.data);
                match self.shared.comp().open_display(&name) {
                    Some(id) => Ok(Response::ok().with_u64(id)),
                    None => Err(Stop(format!(
                        "the program opened the display `{name}`, which this runtime does not have"
                    ))),
                }
            }
            // CloseDisplay(u64 display id)
            1020 => {
                self.shared.comp().close_display(req.u64_at(0));
                Ok(Response::ok())
            }
            // GetDisplayResolution(u64 display id) -> (s64 width, s64 height)
            1102 => match self.shared.comp().display_resolution(req.u64_at(0)) {
                Some((w, h)) => Ok(Response::ok().with_u64(u64::from(w)).with_u64(u64::from(h))),
                None => Err(Stop(format!(
                    "the resolution of display {}, which is not open",
                    req.u64_at(0)
                ))),
            },
            // OpenLayer(DisplayName, u64 layer id, u64 aruid, pid) -> (u64 size, buffer window)
            2020 => {
                let layer = req.u64_at(0x40);
                let pid = req.pid.unwrap_or(0) as u32;
                let window = self.shared.comp().open_layer(layer, pid);
                let Some(window) = window else {
                    return Err(Stop(format!(
                        "the program opened layer {layer}, which was never created"
                    )));
                };
                let size = self.write_window(req, &window)?;
                Ok(Response::ok().with_u64(size))
            }
            // CloseLayer(u64 layer id)
            2021 => {
                self.shared.comp().close_layer(req.u64_at(0));
                Ok(Response::ok())
            }
            // CreateStrayLayer(u32 flags, u32 pad, u64 display id) -> (u64 layer id, u64 size)
            2030 => {
                let display = req.u64_at(8);
                let (layer, window) = {
                    let mut comp = self.shared.comp();
                    if comp.display(display).is_none() {
                        return Err(Stop(format!(
                            "a stray layer on display {display}, not open"
                        )));
                    }
                    let layer = comp.create_layer(display);
                    let window = comp.open_layer(layer, req.pid.unwrap_or(0) as u32);
                    (layer, window.unwrap_or_default())
                };
                let size = self.write_window(req, &window)?;
                Ok(Response::ok().with_u64(layer).with_u64(size))
            }
            // DestroyStrayLayer(u64 layer id)
            2031 => {
                self.shared.comp().close_layer(req.u64_at(0));
                Ok(Response::ok())
            }
            // SetLayerScalingMode(u64 mode, u64 layer id): the layer is shown as it is
            2101 => Ok(Response::ok()),
            // GetDisplayVsyncEvent(u64 display id) -> event
            5202 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let token = self.shared.comp().vsync_event(req.u64_at(0));
                let Some(token) = token else {
                    return Err(Stop(format!(
                        "the vsync event of display {}",
                        req.u64_at(0)
                    )));
                };
                self.shared.start_vsync(&kernel);
                let (_, readable) = self.shared.event(&kernel, token);
                Ok(Response::ok().with_copy_handle(readable))
            }
            other => Err(Stop::unknown_command(
                "nn::visrv::sf::IApplicationDisplayService",
                other,
            )),
        }
    }
}

/// `nns::hosbinder::IHOSBinderDriver`: the transport of the layers' buffer queues.
#[derive(Debug)]
struct BinderDriver {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl BinderDriver {
    fn transact(&mut self, req: &Request) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let (binder, code) = (req.u32_at(0) as i32, req.u32_at(4));
        let Some(input) = req.input_buffer(0) else {
            return Err(Stop("a transaction without a parcel".into()));
        };
        let Some(out) = req.output_buffer(0) else {
            return Err(Stop("a transaction without a reply buffer".into()));
        };
        let data = kernel
            .read_buffer(input, 0x1000)
            .ok_or_else(|| Stop("a transaction parcel the process cannot read".into()))?;
        let reply = self
            .shared
            .comp()
            .transact(binder, code, &data)
            .map_err(|why| Stop(format!("binder {binder}, code {code}: {why}")))?;
        self.shared.deliver_signals(&kernel);
        if reply.len() as u64 > out.size || !kernel.write_memory(out.addr, &reply) {
            return Err(Stop(format!(
                "the reply ({} bytes) does not fit the {}-byte buffer",
                reply.len(),
                out.size
            )));
        }
        if code == nx_gpu::display::code::QUEUE_BUFFER {
            let frame = self.shared.comp().last_frame_of(binder);
            if let (Some(frame), Some(gpu)) = (&frame, self.shared.gpu()) {
                gpu.present(frame).map_err(|why| {
                    Stop(format!(
                        "the renderer could not show frame {}: {why}",
                        frame.number
                    ))
                })?;
            }
            if let Some(frame) = &frame {
                self.shared.pace(frame.input.swap_interval);
            }
        }
        if code == nx_gpu::display::code::QUEUE_BUFFER {
            // The clock of `NX_INPUT_SCRIPT`.
            let frames = self.shared.comp().frames_presented();
            super::super::input::frames_presented(frames);
        }
        if let Some(limit) = self.shared.frame_limit {
            let frames = self.shared.comp().frames_presented();
            if code == nx_gpu::display::code::QUEUE_BUFFER && frames >= limit {
                eprintln!("kiln runtime: NX_MAX_FRAMES reached after {frames} presented frames");
                kernel.exit_process(0);
            }
        }
        Ok(Response::ok())
    }
}

impl Service for BinderDriver {
    fn name(&self) -> &str {
        "nns::hosbinder::IHOSBinderDriver"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // TransactParcel(s32 id, u32 code, u32 flags, buffer parcel) -> buffer reply
            // TransactParcelAuto: the same with auto-selected buffers
            0 | 3 => self.transact(req),
            // AdjustRefcount(s32 id, s32 addval, s32 type): the layer lives as long as it is open
            1 => Ok(Response::ok()),
            // GetNativeHandle(s32 id, u32 kind) -> event, set while a buffer is free
            2 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let binder = req.u32_at(0) as i32;
                let token = self.shared.comp().buffer_event(binder);
                let Some(token) = token else {
                    return Err(Stop(format!("the buffer event of binder {binder}")));
                };
                self.shared.deliver_signals(&kernel);
                let (_, readable) = self.shared.event(&kernel, token);
                Ok(Response::ok().with_copy_handle(readable))
            }
            other => Err(Stop::unknown_command(
                "nns::hosbinder::IHOSBinderDriver",
                other,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paced() -> Shared {
        let mut s = Shared::new();
        s.pacing = Some(Mutex::new(None));
        s
    }

    fn last(s: &Shared) -> Instant {
        let held = s.pacing.as_ref().unwrap().lock().unwrap();
        held.expect("a frame was paced")
    }

    #[test]
    fn queued_frames_are_held_to_the_rate_of_the_display() {
        let s = paced();
        let start = Instant::now();
        // The first frame goes at once, each of the next three a period after the one before.
        for _ in 0..4 {
            s.pace(1);
        }
        assert!(start.elapsed() >= VSYNC_PERIOD * 3, "{:?}", start.elapsed());
        // A swap interval of 2 holds the next frame for two periods.
        let before = last(&s);
        s.pace(2);
        assert_eq!(last(&s), before + VSYNC_PERIOD * 2);
        assert!(Instant::now() >= before + VSYNC_PERIOD * 2);
    }

    #[test]
    fn a_late_frame_is_not_held_and_does_not_catch_up() {
        let s = paced();
        s.pace(1);
        std::thread::sleep(VSYNC_PERIOD * 3);
        let before = Instant::now();
        s.pace(1);
        // Its time is counted from when it came, not from where the one before left off.
        assert!(last(&s) >= before);
        // Interval 0 is never held.
        let before = last(&s);
        s.pace(0);
        assert_eq!(last(&s), before);
    }

    #[test]
    fn unpaced_frames_are_never_held() {
        let mut s = Shared::new();
        s.pacing = None;
        let start = Instant::now();
        for _ in 0..10 {
            s.pace(1);
        }
        assert!(start.elapsed() < VSYNC_PERIOD * 9, "{:?}", start.elapsed());
    }
}
