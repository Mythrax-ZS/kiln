//! `appletOE`, the applet manager an application talks to (Switchbrew, "Applet Manager services").
//!
//! The program is the only applet: it is always in focus, the console is handheld, nothing asks it
//! to suspend or exit, and the home menu does not exist. Everything an interface returns is either
//! state the program set itself or one of those constants (docs/ASSUMPTIONS.md A-043). A command
//! this module does not implement stops the run.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

/// The message queue's "nothing waiting" answer: module 128 (`am`), description 3.
const NO_MESSAGE: ResultCode = ResultCode::new(128, 3);

/// `AppletMessage` values this module produces (Switchbrew, "AppletMessage").
#[allow(dead_code)]
pub mod message {
    pub const FOCUS_STATE_CHANGED: u32 = 15;
    pub const RESUME: u32 = 16;
    pub const OPERATION_MODE_CHANGED: u32 = 30;
    pub const PERFORMANCE_MODE_CHANGED: u32 = 31;
}

/// `FocusState`: 1 in focus, 2 out of focus, 3 background.
const IN_FOCUS: u8 = 1;
/// `OperationMode`: 0 handheld, 1 console (docked).
const HANDHELD: u8 = 0;
/// The applet resource user id handed to `hid` and friends. The real one is chosen by the
/// applet manager; programs only pass it on.
pub const APPLET_RESOURCE_USER_ID: u64 = 0x1;

/// What every interface of the one applet shares.
#[derive(Debug, Default)]
pub struct AppletState {
    /// Messages waiting for `ReceiveMessage`, oldest first.
    messages: VecDeque<u32>,
    /// The event `GetEventHandle` returns: signalled while a message waits.
    message_event: Option<(KObject, KObject)>,
    /// Events handed out by getters that nothing ever signals.
    idle_events: Vec<(KObject, KObject)>,
    audio_volume: [f32; 2],
    /// The first `OpenApplicationProxy` queues the message an application starts with.
    started: bool,
}

pub type Shared = Arc<Mutex<AppletState>>;

fn lock(shared: &Shared) -> std::sync::MutexGuard<'_, AppletState> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// Queue a message for `ReceiveMessage` and signal the message event if it exists.
fn push_message(kernel: &Kernel, shared: &Shared, m: u32) {
    let mut s = lock(shared);
    s.messages.push_back(m);
    if let Some((writable, _)) = &s.message_event {
        kernel.set_event(writable, true);
    }
}

/// A readable event that nothing signals, shared by every getter of the same kind.
fn idle_event(kernel: &Kernel, shared: &Shared, slot: usize) -> KObject {
    let mut s = lock(shared);
    while s.idle_events.len() <= slot {
        s.idle_events.push(kernel.new_event());
    }
    s.idle_events[slot].1.clone()
}

/// `nn::am::service::IApplicationProxyService` (`appletOE`).
#[derive(Debug)]
pub struct AppletOe {
    kernel: Weak<Kernel>,
    shared: Shared,
}

impl AppletOe {
    pub fn new(kernel: Weak<Kernel>, shared: Shared) -> AppletOe {
        AppletOe { kernel, shared }
    }
}

impl Service for AppletOe {
    fn name(&self) -> &str {
        "nn::am::service::IApplicationProxyService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // OpenApplicationProxy(u64 reserved, pid, process handle) -> IApplicationProxy
            0 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                // An application starts in the foreground, which the applet manager announces
                // with a focus message (docs/ASSUMPTIONS.md A-043).
                let first = !std::mem::replace(&mut lock(&self.shared).started, true);
                if first {
                    push_message(&kernel, &self.shared, message::FOCUS_STATE_CHANGED);
                }
                Ok(Response::ok().with_object(Box::new(ApplicationProxy {
                    kernel: self.kernel.clone(),
                    shared: self.shared.clone(),
                })))
            }
            other => Err(Stop::unknown_command("appletOE", other)),
        }
    }
}

/// `nn::am::service::IApplicationProxy`: hands out the other interfaces.
#[derive(Debug)]
struct ApplicationProxy {
    kernel: Weak<Kernel>,
    shared: Shared,
}

impl Service for ApplicationProxy {
    fn name(&self) -> &str {
        "nn::am::service::IApplicationProxy"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        let (kernel, shared) = (self.kernel.clone(), self.shared.clone());
        let object: Box<dyn Service> = match req.command {
            0 => Box::new(CommonStateGetter { kernel, shared }),
            1 => Box::new(SelfController { kernel, shared }),
            2 => Box::new(WindowController),
            3 => Box::new(AudioController { shared }),
            // The display controller, the process winding controller and the debug functions
            // exist but none of their commands is implemented: the first one used stops the run.
            4 => Box::new(Bare("nn::am::service::IDisplayController")),
            10 => Box::new(Bare("nn::am::service::IProcessWindingController")),
            1000 => Box::new(Bare("nn::am::service::IDebugFunctions")),
            11 => Box::new(LibraryAppletCreator { kernel }),
            20 => Box::new(ApplicationFunctions { kernel, shared }),
            other => return Err(Stop::unknown_command("IApplicationProxy", other)),
        };
        Ok(Response::ok().with_object(object))
    }
}

/// An interface object with no command implemented: handing it out is right, using it stops the
/// run with the command named.
#[derive(Debug)]
struct Bare(&'static str);

impl Service for Bare {
    fn name(&self) -> &str {
        self.0
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop::unknown_command(self.0, req.command))
    }
}

/// `nn::am::service::ICommonStateGetter`.
#[derive(Debug)]
struct CommonStateGetter {
    kernel: Weak<Kernel>,
    shared: Shared,
}

impl Service for CommonStateGetter {
    fn name(&self) -> &str {
        "nn::am::service::ICommonStateGetter"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        match req.command {
            // GetEventHandle() -> event
            0 => {
                let mut s = lock(&self.shared);
                if s.message_event.is_none() {
                    let ends = kernel.new_event();
                    kernel.set_event(&ends.0, !s.messages.is_empty());
                    s.message_event = Some(ends);
                }
                let readable = s.message_event.as_ref().map(|e| e.1.clone());
                Ok(Response::ok().with_copy_handle(readable.expect("created above")))
            }
            // ReceiveMessage() -> AppletMessage
            1 => {
                let mut s = lock(&self.shared);
                let Some(m) = s.messages.pop_front() else {
                    return Ok(Response::error(NO_MESSAGE));
                };
                if s.messages.is_empty() {
                    if let Some((w, _)) = &s.message_event {
                        kernel.set_event(w, false);
                    }
                }
                Ok(Response::ok().with_u32(m))
            }
            // GetOperationMode() -> u8
            5 => Ok(Response::ok().with_bytes(&[HANDHELD])),
            // GetPerformanceMode() -> u32
            6 => Ok(Response::ok().with_u32(0)),
            // GetCurrentFocusState() -> u8
            9 => Ok(Response::ok().with_bytes(&[IN_FOCUS])),
            // GetDefaultDisplayResolution() -> (s32 width, s32 height) [3.0.0+]
            60 => Ok(Response::ok().with_u32(1280).with_u32(720)),
            // GetDefaultDisplayResolutionChangeEvent() -> event [3.0.0+]
            61 => Ok(Response::ok().with_copy_handle(idle_event(&kernel, &self.shared, 0))),
            other => Err(Stop::unknown_command("ICommonStateGetter", other)),
        }
    }
}

/// `nn::am::service::ISelfController`.
#[derive(Debug)]
struct SelfController {
    kernel: Weak<Kernel>,
    shared: Shared,
}

impl Service for SelfController {
    fn name(&self) -> &str {
        "nn::am::service::ISelfController"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // LockExit(), UnlockExit(): nothing asks the program to exit.
            1 | 2 => Ok(Response::ok()),
            // GetLibraryAppletLaunchableEvent() -> event
            9 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                Ok(Response::ok().with_copy_handle(idle_event(&kernel, &self.shared, 1)))
            }
            // SetScreenShotPermission(s32), SetOperationModeChangedNotification(bool),
            // SetPerformanceModeChangedNotification(bool), SetFocusHandlingMode(3 x bool),
            // SetRestartMessageEnabled(bool), SetOutOfFocusSuspendingEnabled(bool)
            10..=14 | 16 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("ISelfController", other)),
        }
    }
}

/// `nn::am::service::IWindowController`.
#[derive(Debug)]
struct WindowController;

impl Service for WindowController {
    fn name(&self) -> &str {
        "nn::am::service::IWindowController"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetAppletResourceUserId() -> u64
            1 => Ok(Response::ok().with_u64(APPLET_RESOURCE_USER_ID)),
            // AcquireForegroundRights(): the program is already in the foreground.
            10 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("IWindowController", other)),
        }
    }
}

/// `nn::am::service::IAudioController`.
#[derive(Debug)]
struct AudioController {
    shared: Shared,
}

impl Service for AudioController {
    fn name(&self) -> &str {
        "nn::am::service::IAudioController"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // SetExpectedMasterVolume(f32 main applet, f32 library applet)
            0 => {
                let f = |o: usize| f32::from_bits(req.u32_at(o));
                lock(&self.shared).audio_volume = [f(0), f(4)];
                Ok(Response::ok())
            }
            // GetMainAppletExpectedMasterVolume() -> f32
            1 => Ok(Response::ok().with_u32(lock(&self.shared).audio_volume[0].to_bits())),
            // GetLibraryAppletExpectedMasterVolume() -> f32
            2 => Ok(Response::ok().with_u32(lock(&self.shared).audio_volume[1].to_bits())),
            other => Err(Stop::unknown_command("IAudioController", other)),
        }
    }
}

/// `nn::am::service::IApplicationFunctions`.
#[derive(Debug)]
struct ApplicationFunctions {
    kernel: Weak<Kernel>,
    shared: Shared,
}

/// `LanguageCode` of the system language, "en-US" (docs/ASSUMPTIONS.md A-044).
const LANGUAGE_CODE: u64 = u64::from_le_bytes(*b"en-US\0\0\0");

impl Service for ApplicationFunctions {
    fn name(&self) -> &str {
        "nn::am::service::IApplicationFunctions"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // PopLaunchParameter(u32 LaunchParameterKind) -> IStorage: this program was not
            // started with any (docs/ASSUMPTIONS.md A-043), which the service reports as "no
            // data in the channel".
            1 => Ok(Response::error(NO_DATA_IN_CHANNEL)),
            // GetDesiredLanguage() -> LanguageCode
            21 => Ok(Response::ok().with_u64(LANGUAGE_CODE)),
            // BeginBlockingHomeButtonShortAndLongPressed(s64), EndBlocking...(),
            // BeginBlockingHomeButton(s64), EndBlockingHomeButton()
            30..=33 => Ok(Response::ok()),
            // NotifyRunning() -> bool
            40 => Ok(Response::ok().with_bytes(&[1])),
            // GetPseudoDeviceId() -> Uuid [2.0.0+]
            50 => Ok(Response::ok().with_bytes(&PSEUDO_DEVICE_ID)),
            // SetMediaPlaybackStateForApplication(bool) [2.0.0+]
            60 => Ok(Response::ok()),
            // IsGamePlayRecordingSupported() -> bool [3.0.0+]
            65 => Ok(Response::ok().with_bytes(&[0])),
            // InitializeGamePlayRecording(TransferMemory handle, u64 size) [3.0.0+]: the size is
            // always 0x6000000. Nothing is ever recorded; the memory is only kept lent.
            66 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let lent = req
                    .copy_handles
                    .first()
                    .and_then(|h| kernel.transfer_memory(h));
                match lent {
                    Some((_, size)) if size == req.u64_at(0) && size == 0x600_0000 => {
                        Ok(Response::ok())
                    }
                    _ => Err(Stop(
                        "InitializeGamePlayRecording without the 0x6000000-byte transfer memory \
                         it is documented to take"
                            .into(),
                    )),
                }
            }
            // SetGamePlayRecordingState(u32) [3.0.0+]
            67 => Ok(Response::ok()),
            // EnableApplicationCrashReport(bool) [4.0.0+]
            90 => Ok(Response::ok()),
            // EnableApplicationAllThreadDumpOnCrash(bool) [6.0.0+]
            124 => Ok(Response::ok()),
            // GetGpuErrorDetectedSystemEvent() -> event [8.0.0+]
            130 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                Ok(Response::ok().with_copy_handle(idle_event(&kernel, &self.shared, 2)))
            }
            // GetHealthWarningDisappearedSystemEvent() -> event [9.0.0+]
            160 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                Ok(Response::ok().with_copy_handle(idle_event(&kernel, &self.shared, 3)))
            }
            other => Err(Stop::unknown_command("IApplicationFunctions", other)),
        }
    }
}

/// The 16-byte id `GetPseudoDeviceId` returns: fixed, because the real one is derived from the
/// console (docs/ASSUMPTIONS.md A-043).
const PSEUDO_DEVICE_ID: [u8; 16] = *b"recomp-pseudo-id";

/// `PopLaunchParameter` with nothing to pop: module 128, description 2.
const NO_DATA_IN_CHANNEL: ResultCode = ResultCode::new(128, 2);

/// `nn::am::service::ILibraryAppletCreator`. No library applet can be created: there is no home
/// menu and no applet to launch; storages, which programs also use for their own purposes, work.
#[derive(Debug)]
struct LibraryAppletCreator {
    kernel: Weak<Kernel>,
}

impl Service for LibraryAppletCreator {
    fn name(&self) -> &str {
        "nn::am::service::ILibraryAppletCreator"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateStorage(s64 size) -> IStorage
            10 => {
                let size = req.u64_at(0) as i64;
                if !(0..=1 << 28).contains(&size) {
                    return Ok(Response::error(super::super::result::kernel::INVALID_SIZE));
                }
                Ok(Response::ok().with_object(Box::new(Storage {
                    data: Arc::new(Mutex::new(vec![0u8; size as usize])),
                    kernel: self.kernel.clone(),
                })))
            }
            other => Err(Stop::unknown_command("ILibraryAppletCreator", other)),
        }
    }
}

/// `nn::am::service::IStorage`: a byte buffer shared between the program and the applet manager.
#[derive(Debug)]
struct Storage {
    data: Arc<Mutex<Vec<u8>>>,
    kernel: Weak<Kernel>,
}

impl Service for Storage {
    fn name(&self) -> &str {
        "nn::am::service::IStorage"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Open() -> IStorageAccessor
            0 => Ok(Response::ok().with_object(Box::new(StorageAccessor {
                data: self.data.clone(),
                kernel: self.kernel.clone(),
            }))),
            other => Err(Stop::unknown_command("IStorage", other)),
        }
    }
}

/// `nn::am::service::IStorageAccessor`.
#[derive(Debug)]
struct StorageAccessor {
    data: Arc<Mutex<Vec<u8>>>,
    kernel: Weak<Kernel>,
}

impl Service for StorageAccessor {
    fn name(&self) -> &str {
        "nn::am::service::IStorageAccessor"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        let mut data = self.data.lock().unwrap_or_else(|e| e.into_inner());
        match req.command {
            // GetSize() -> s64
            0 => Ok(Response::ok().with_u64(data.len() as u64)),
            // Write(s64 offset, buffer<0x21> in)
            10 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let offset = req.u64_at(0) as i64;
                let Some(buf) = req.input_buffer(0) else {
                    return Err(Stop(
                        "IStorageAccessor::Write without an input buffer".into(),
                    ));
                };
                let end = (offset >= 0).then(|| offset as u64 + buf.size);
                let Some(end) = end.filter(|&e| e <= data.len() as u64) else {
                    return Err(out_of_range(req.command, offset, buf.size, data.len()));
                };
                let Some(bytes) = kernel.read_buffer(buf, buf.size) else {
                    return Ok(Response::error(
                        super::super::result::kernel::INVALID_POINTER,
                    ));
                };
                data[offset as usize..end as usize].copy_from_slice(&bytes);
                Ok(Response::ok())
            }
            // Read(s64 offset, buffer<0x22> out)
            11 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let offset = req.u64_at(0) as i64;
                let Some(buf) = req.output_buffer(0) else {
                    return Err(Stop(
                        "IStorageAccessor::Read without an output buffer".into(),
                    ));
                };
                let end = (offset >= 0).then(|| offset as u64 + buf.size);
                let Some(end) = end.filter(|&e| e <= data.len() as u64) else {
                    return Err(out_of_range(req.command, offset, buf.size, data.len()));
                };
                if !kernel.write_memory(buf.addr, &data[offset as usize..end as usize]) {
                    return Ok(Response::error(
                        super::super::result::kernel::INVALID_POINTER,
                    ));
                }
                Ok(Response::ok())
            }
            other => Err(Stop::unknown_command("IStorageAccessor", other)),
        }
    }
}

/// An access outside a storage: the result the service gives is not documented, so the run stops.
fn out_of_range(command: u32, offset: i64, size: u64, len: usize) -> Stop {
    Stop(format!(
        "IStorageAccessor command {command} at offset {offset} for {size} bytes lies outside the 
         {len}-byte storage; the result the service answers with is not documented"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(command: u32, data: &[u8]) -> Request {
        Request {
            command,
            data: data.to_vec(),
            ..Request::default()
        }
    }

    #[test]
    fn the_language_code_spells_the_locale() {
        assert_eq!(&LANGUAGE_CODE.to_le_bytes(), b"en-US\0\0\0");
    }

    #[test]
    fn the_window_controller_answers_with_the_resource_user_id() {
        let mut w = WindowController;
        let r = w.request(&req(1, &[])).unwrap();
        assert_eq!(r.data, APPLET_RESOURCE_USER_ID.to_le_bytes());
        assert!(w.request(&req(10, &[])).unwrap().result.is_success());
        assert!(w.request(&req(77, &[])).is_err());
    }

    #[test]
    fn audio_volumes_are_remembered() {
        let shared: Shared = Arc::default();
        let mut a = AudioController {
            shared: shared.clone(),
        };
        let mut data = 0.25f32.to_le_bytes().to_vec();
        data.extend_from_slice(&0.5f32.to_le_bytes());
        a.request(&req(0, &data)).unwrap();
        assert_eq!(a.request(&req(1, &[])).unwrap().data, 0.25f32.to_le_bytes());
        assert_eq!(a.request(&req(2, &[])).unwrap().data, 0.5f32.to_le_bytes());
    }

    #[test]
    fn the_proxy_hands_out_interfaces_and_refuses_the_rest() {
        let mut p = ApplicationProxy {
            kernel: Weak::new(),
            shared: Arc::default(),
        };
        for c in [0, 1, 2, 3, 4, 10, 11, 20, 1000] {
            assert_eq!(p.request(&req(c, &[])).unwrap().objects.len(), 1, "{c}");
        }
        assert!(p.request(&req(5, &[])).is_err());
        // An interface with nothing implemented stops the run on its first command.
        let mut bare = Bare("nn::am::service::IDisplayController");
        let stop = bare.request(&req(3, &[])).unwrap_err();
        assert!(
            stop.0.contains("IDisplayController") && stop.0.contains('3'),
            "{}",
            stop.0
        );
    }
}
