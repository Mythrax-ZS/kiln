//! `nfp:user`, the amiibo service (Switchbrew, "NFC services").
//!
//! No NFC device is attached (docs/ASSUMPTIONS.md A-072): the list of devices is empty, the events
//! never fire, and the commands that take a device handle stop the run, because a program only
//! has one after listing a device.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::{Arc, Weak};

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// `nn::nfp::detail::IUserManager`.
#[derive(Debug)]
pub struct UserManager {
    kernel: Weak<Kernel>,
}

impl UserManager {
    pub fn new(kernel: Weak<Kernel>) -> UserManager {
        UserManager { kernel }
    }
}

impl Service for UserManager {
    fn name(&self) -> &str {
        "nn::nfp::detail::IUserManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateUserInterface() -> IUser
            0 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                Ok(Response::ok().with_object(Box::new(User::new(&kernel))))
            }
            other => Err(Stop::unknown_command("nfp:user", other)),
        }
    }
}

/// `nn::nfp::detail::IUser`.
#[derive(Debug)]
struct User {
    initialized: bool,
    /// The activate, deactivate and availability events; nothing sets them.
    events: [(KObject, KObject); 3],
}

impl User {
    fn new(kernel: &Arc<Kernel>) -> User {
        User {
            initialized: false,
            events: [kernel.new_event(), kernel.new_event(), kernel.new_event()],
        }
    }
}

impl Service for User {
    fn name(&self) -> &str {
        "nn::nfp::detail::IUser"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Initialize(u64 aruid, u64 reserved, pid, buffer) / Finalize()
            0 => {
                self.initialized = true;
                Ok(Response::ok())
            }
            1 => {
                self.initialized = false;
                Ok(Response::ok())
            }
            // ListDevices() -> (pointer buffer of DeviceHandle, s32 count): none
            2 => Ok(Response::ok().with_u32(0).with_pointer_reply(0, 0)),
            // GetActivateEventHandle(DeviceHandle), GetDeactivateEventHandle(DeviceHandle)
            17 => Ok(Response::ok().with_copy_handle(self.events[0].1.clone())),
            18 => Ok(Response::ok().with_copy_handle(self.events[1].1.clone())),
            // GetState() -> u32: 0 not initialized, 1 initialized
            19 => Ok(Response::ok().with_u32(u32::from(self.initialized))),
            // GetAvailabilityChangeEventHandle() -> event
            23 => Ok(Response::ok().with_copy_handle(self.events[2].1.clone())),
            other => Err(Stop::unknown_command("nn::nfp::detail::IUser", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        // `ListDevices` returns the handles through a pointer buffer (docs/ASSUMPTIONS.md A-067).
        0x1000
    }
}
