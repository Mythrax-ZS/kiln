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
use super::Bare;
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
    /// The display the program's layers are made on (`CreateManagedDisplayLayer`).
    pub display: Option<Arc<super::vi::Shared>>,
    /// Has the pre-selected user been popped? A launch parameter can be popped once.
    preselected_popped: bool,
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
    event_ends(kernel, shared, slot).1
}

/// The (writable, readable) ends of the event in `slot`, made on first use.
fn event_ends(kernel: &Kernel, shared: &Shared, slot: usize) -> (KObject, KObject) {
    let mut s = lock(shared);
    while s.idle_events.len() <= slot {
        s.idle_events.push(kernel.new_event());
    }
    s.idle_events[slot].clone()
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
        let kernel = || self.kernel.upgrade().ok_or_else(gone);
        match req.command {
            // GetEventHandle() -> event
            0 => {
                let kernel = kernel()?;
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
                let kernel = kernel()?;
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
            61 => Ok(Response::ok().with_copy_handle(idle_event(&*kernel()?, &self.shared, 0))),
            // SetCpuBoostMode(u32: 0 normal, 1 fast load) [6.0.0+], CancelCpuBoostMode() [10.0.0+]: a
            // request for the console's CPU to run faster while the program loads; the host's
            // clock is not the program's to change (docs/ASSUMPTIONS.md A-134)
            66 | 67 => Ok(Response::ok()),
            // SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled() [11.0.0+]: it only
            // changes what a later ExecuteProgram does, and that stops the run, so it is accepted
            // and nothing more (docs/ASSUMPTIONS.md A-131).
            900 => Ok(Response::ok()),
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
            // GetLibraryAppletLaunchableEvent() -> event: library applets can be launched, so the
            // event is set whenever the program asks for it (docs/ASSUMPTIONS.md A-043)
            9 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let (writable, readable) = event_ends(&kernel, &self.shared, 1);
                kernel.set_event(&writable, true);
                Ok(Response::ok().with_copy_handle(readable))
            }
            // SetScreenShotPermission(s32), SetOperationModeChangedNotification(bool),
            // SetPerformanceModeChangedNotification(bool), SetFocusHandlingMode(3 x bool),
            // SetRestartMessageEnabled(bool), SetOutOfFocusSuspendingEnabled(bool)
            10..=14 | 16 => Ok(Response::ok()),
            // SetAlbumImageOrientation(u32): which way up the pictures of the capture button are
            // taken; nothing takes them (docs/ASSUMPTIONS.md A-116)
            19 => Ok(Response::ok()),
            // GetAccumulatedSuspendedTickValue() -> u64 [6.0.0+]: nothing suspends the program, so
            // it has been suspended for no ticks (docs/ASSUMPTIONS.md A-132)
            90 => Ok(Response::ok().with_u64(0)),
            // GetAccumulatedSuspendedTickChangedEvent() -> event [6.0.0+]: the value never changes,
            // so the event is never signalled (docs/ASSUMPTIONS.md A-132)
            91 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                Ok(Response::ok().with_copy_handle(idle_event(&kernel, &self.shared, 2)))
            }
            // CreateManagedDisplayLayer() -> u64 layer id, for `vi`'s OpenLayer.
            // CreateManagedDisplaySeparableLayer() -> (u64 layer id, u64 recording layer id)
            // [10.0.0+]: the first is used as the one of command 40, the second only by
            // `nn::vi::CreateRecordingLayer`, which opens it the same way; both are made here the
            // same way (docs/ASSUMPTIONS.md A-136).
            40 | 44 => {
                let display = lock(&self.shared).display.clone();
                let Some(display) = display else {
                    return Err(Stop("a layer was asked for but there is no display".into()));
                };
                let mut comp = display.comp();
                let Some(d) = comp.open_display("Default") else {
                    return Err(Stop("the default display cannot be opened".into()));
                };
                let mut response = Response::ok().with_u64(comp.create_layer(d));
                if req.command == 44 {
                    response = response.with_u64(comp.create_layer(d));
                }
                Ok(response)
            }
            // SetManagedDisplayLayerSeparationMode(u32, 0 or 1) [10.0.0+], which the SDK exposes as
            // `SetRecordingLayerEnabled`: nothing composes a recording layer, so it is accepted
            // (docs/ASSUMPTIONS.md A-136).
            45 => match req.u32_at(0) {
                0 | 1 => Ok(Response::ok()),
                other => Err(Stop(format!(
                    "SetManagedDisplayLayerSeparationMode was given {other}; it takes 0 or 1"
                ))),
            },
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
            // PopLaunchParameter(u32 LaunchParameterKind) -> IStorage. The program was started
            // with the user it selected (kind 2, once) and with no other parameter
            // (docs/ASSUMPTIONS.md A-043, A-066).
            1 => {
                let popped = req.u32_at(0) == 2
                    && !std::mem::replace(&mut lock(&self.shared).preselected_popped, true);
                if !popped {
                    return Ok(Response::error(STORAGE_NOT_AVAILABLE));
                }
                Ok(Response::ok().with_object(Box::new(Storage {
                    data: Arc::new(Mutex::new(super::acc::preselected_user_selection())),
                    kernel: self.kernel.clone(),
                })))
            }
            // EnsureSaveData(Uid) -> u64: the save data of the user exists afterwards (it is a
            // directory on the host); the answer is the space it still needs, none
            // (docs/ASSUMPTIONS.md A-068)
            20 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let mut attr = [0u8; 0x40];
                attr[8..24].copy_from_slice(&req.data[..16.min(req.data.len())]);
                attr[32] = 1; // account save data
                let dir = super::savefs::save_dir(kernel.process.program_id, &attr);
                match dir.map(|d| (std::fs::create_dir_all(&d), d)) {
                    Some((Ok(()), _)) => Ok(Response::ok().with_u64(0)),
                    Some((Err(e), d)) => Err(Stop(format!(
                        "the save data directory {} cannot be made: {e}",
                        d.display()
                    ))),
                    None => Err(Stop("EnsureSaveData has no directory to use".into())),
                }
            }
            // GetDesiredLanguage() -> LanguageCode
            21 => Ok(Response::ok().with_u64(LANGUAGE_CODE)),
            // GetDisplayVersion() -> DisplayVersion (16 bytes, a NUL-padded string). The version
            // is a field of the program's control data, which is not among the inputs
            // (docs/ASSUMPTIONS.md A-117).
            23 => Ok(Response::ok().with_bytes(&DISPLAY_VERSION)),
            // SetTerminateResult(Result): nothing reads the result the program ends with
            22 => {
                if let Some(kernel) = self.kernel.upgrade() {
                    if kernel.tracing() {
                        let r = req.u32_at(0);
                        eprintln!(
                            "[am] SetTerminateResult({r:#x}): module {}, description {}",
                            (r & 0x1FF) + 2000,
                            r >> 9
                        );
                    }
                }
                Ok(Response::ok())
            }
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

/// The version string `GetDisplayVersion` returns, "1.0.0" in 16 bytes (docs/ASSUMPTIONS.md A-117).
const DISPLAY_VERSION: [u8; 16] = *b"1.0.0\0\0\0\0\0\0\0\0\0\0\0";

/// The 16-byte id `GetPseudoDeviceId` returns: fixed, because the real one is derived from the
/// console (docs/ASSUMPTIONS.md A-043). It spells the program's first name and stays as it is: a
/// game may have stored it.
const PSEUDO_DEVICE_ID: [u8; 16] = *b"recomp-pseudo-id";

/// `PopLaunchParameter` with nothing to pop: 2128-0002, "IStorage not available" in Switchbrew's
/// error code list.
const STORAGE_NOT_AVAILABLE: ResultCode = ResultCode::new(128, 2);

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
            // CreateLibraryApplet(u32 applet id, u32 mode) -> ILibraryAppletAccessor
            0 => Err(Stop(format!(
                "the program asked to launch the library applet {:#x} (mode {}); library applets \
                 are part of the console's firmware, which this runtime does not have",
                req.u32_at(0),
                req.u32_at(4)
            ))),
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
                    return Ok(Response::error(STORAGE_OUT_OF_BOUNDS));
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
                    return Ok(Response::error(STORAGE_OUT_OF_BOUNDS));
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

/// A read or write outside a storage: 2128-0503, "IStorage read/write out of bounds" in
/// Switchbrew's error code list.
const STORAGE_OUT_OF_BOUNDS: ResultCode = ResultCode::new(128, 503);

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
    fn the_display_version_is_a_nul_padded_string_of_sixteen_bytes() {
        let mut f = ApplicationFunctions {
            kernel: Weak::new(),
            shared: Arc::default(),
        };
        let r = f.request(&req(23, &[])).unwrap();
        assert!(r.result.is_success());
        assert_eq!(r.data.len(), 16);
        assert_eq!(&r.data[..5], b"1.0.0");
        assert!(r.data[5..].iter().all(|&b| b == 0));
        // The neighbouring commands that are not implemented still stop the run.
        let stop = f.request(&req(24, &[])).unwrap_err();
        assert!(stop.0.contains("IApplicationFunctions") && stop.0.contains("24"));
    }

    #[test]
    fn the_common_state_getter_accepts_the_exit_to_library_applet_request_and_stops_on_others() {
        let mut g = CommonStateGetter {
            kernel: Weak::new(),
            shared: Arc::default(),
        };
        let r = g.request(&req(900, &[])).unwrap();
        assert!(r.result.is_success() && r.data.is_empty());
        // 910 (GetLaunchRequiredTick) is not implemented and says so.
        let stop = g.request(&req(910, &[])).unwrap_err();
        assert!(stop.0.contains("ICommonStateGetter") && stop.0.contains("910"));
    }

    #[test]
    fn the_self_controller_says_the_program_was_never_suspended() {
        let mut c = SelfController {
            kernel: Weak::new(),
            shared: Arc::default(),
        };
        let r = c.request(&req(90, &[])).unwrap();
        assert_eq!(r.data, 0u64.to_le_bytes());
        // The event needs a kernel; without one the request says the process is gone.
        assert!(c.request(&req(91, &[])).unwrap_err().0.contains("gone"));
        assert!(c.request(&req(92, &[])).unwrap_err().0.contains("92"));
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
    fn the_self_controller_accepts_its_settings_and_stops_on_the_rest() {
        let mut s = SelfController {
            kernel: Weak::new(),
            shared: Arc::default(),
        };
        // Settings the program makes and nothing reads back, among them the orientation of the
        // pictures of the capture button (command 19, A-116).
        for command in [1, 2, 10, 11, 12, 13, 14, 16, 19] {
            assert!(s
                .request(&req(command, &[0; 4]))
                .unwrap()
                .result
                .is_success());
        }
        // A command nobody implemented stops the run and says which it was.
        let stop = s.request(&req(77, &[])).unwrap_err();
        assert!(stop.0.contains("ISelfController") && stop.0.contains("77"));
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
