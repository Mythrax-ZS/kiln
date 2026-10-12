//! `hid`, the input service (Switchbrew, "HID services").
//!
//! A program asks `hid` for an applet resource, maps the shared memory that comes with it, and
//! reads input state the service publishes there. Nothing is connected to this runtime: the shared
//! memory stays zero (no controller, no touch, no key) and the activation commands are accepted
//! and change nothing (docs/ASSUMPTIONS.md A-070). A command not listed stops the run.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::{Arc, Mutex, Weak};

/// The size of the shared memory (Switchbrew: `0x40000`).
const SHARED_MEMORY_SIZE: u64 = 0x4_0000;

/// What every session of the process shares: the shared memory object, made on first use.
#[derive(Default)]
pub struct Shared {
    memory: Mutex<Option<KObject>>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shared")
    }
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// `nn::hid::IHidServer`.
#[derive(Debug)]
pub struct HidServer {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
    style_set: u32,
    joy_hold_type: u64,
    handheld_mode: u64,
}

impl HidServer {
    pub fn new(kernel: Weak<Kernel>, shared: Arc<Shared>) -> HidServer {
        HidServer {
            kernel,
            shared,
            style_set: 0,
            joy_hold_type: 0,
            handheld_mode: 0,
        }
    }
}

impl Service for HidServer {
    fn name(&self) -> &str {
        "nn::hid::IHidServer"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateAppletResource(u64 aruid, pid) -> IAppletResource
            0 => Ok(Response::ok().with_object(Box::new(AppletResource {
                kernel: self.kernel.clone(),
                shared: self.shared.clone(),
            }))),
            // ActivateDebugPad, ActivateTouchScreen, ActivateMouse, ActivateKeyboard (aruid, pid)
            1 | 11 | 21 | 31 => Ok(Response::ok()),
            // SetSupportedNpadStyleSet(u32 style set, u64 aruid, pid)
            100 => {
                self.style_set = req.u32_at(0);
                Ok(Response::ok())
            }
            // GetSupportedNpadStyleSet(u64 aruid, pid) -> u32
            101 => Ok(Response::ok().with_u32(self.style_set)),
            // SetSupportedNpadIdType(u64 aruid, buffer of NpadIdType, pid): the ids are not kept,
            // as no controller ever connects
            102 => Ok(Response::ok()),
            // ActivateNpad(u64 aruid, pid); ActivateNpadWithRevision(s32 revision, u64 aruid, pid)
            103 | 109 => Ok(Response::ok()),
            // SetNpadJoyHoldType(s64 type, u64 aruid, pid); GetNpadJoyHoldType(aruid, pid) -> s64
            120 => {
                self.joy_hold_type = req.u64_at(8);
                Ok(Response::ok())
            }
            121 => Ok(Response::ok().with_u64(self.joy_hold_type)),
            // SetNpadJoyAssignmentModeSingleByDefault / Single / Dual, MergeSingleJoyAsDualJoy:
            // no controller is attached, so there is nothing to assign
            122..=125 => Ok(Response::ok()),
            // SetNpadHandheldActivationMode(aruid, s64 mode, pid); Get(aruid, pid) -> s64
            128 => {
                self.handheld_mode = req.u64_at(8);
                Ok(Response::ok())
            }
            129 => Ok(Response::ok().with_u64(self.handheld_mode)),
            other => Err(Stop::unknown_command("hid", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        // `SetSupportedNpadIdType` and `SendVibrationValues` pass arrays through pointer buffers
        // (docs/ASSUMPTIONS.md A-067).
        0x1000
    }
}

/// `nn::hid::IAppletResource`.
#[derive(Debug)]
struct AppletResource {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl Service for AppletResource {
    fn name(&self) -> &str {
        "nn::hid::IAppletResource"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetSharedMemoryHandle() -> the shared memory with the input state
            0 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let mut slot = self.shared.memory.lock().unwrap_or_else(|e| e.into_inner());
                let object = match slot.as_ref() {
                    Some(o) => o.clone(),
                    None => {
                        let (o, _) =
                            kernel
                                .create_shared_memory(SHARED_MEMORY_SIZE)
                                .map_err(|e| {
                                    Stop(format!(
                                        "the input shared memory cannot be made ({:#x})",
                                        e.0
                                    ))
                                })?;
                        *slot = Some(o.clone());
                        o
                    }
                };
                Ok(Response::ok().with_copy_handle(object))
            }
            other => Err(Stop::unknown_command("IAppletResource", other)),
        }
    }
}
