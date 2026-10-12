//! `nifm:u`, the network interface manager (Switchbrew, "Network Interface services").
//!
//! The console is offline (docs/ASSUMPTIONS.md A-072): there is no network interface, a request for
//! a connection is never accepted, and the internet connection status is "not connected". A
//! command not listed stops the run.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::{Arc, Mutex, Weak};

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// `nn::nifm::detail::IStaticService`.
#[derive(Debug)]
pub struct StaticService {
    kernel: Weak<Kernel>,
}

impl StaticService {
    pub fn new(kernel: Weak<Kernel>) -> StaticService {
        StaticService { kernel }
    }
}

impl Service for StaticService {
    fn name(&self) -> &str {
        "nn::nifm::detail::IStaticService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateGeneralServiceOld() / CreateGeneralService(u64 reserved, pid) -> IGeneralService
            4 | 5 => Ok(Response::ok().with_object(Box::new(GeneralService {
                kernel: self.kernel.clone(),
            }))),
            other => Err(Stop::unknown_command("nifm:u", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        // `GetClientId` returns the id through a pointer buffer (docs/ASSUMPTIONS.md A-067).
        0x1000
    }
}

/// `nn::nifm::detail::IGeneralService`.
#[derive(Debug)]
struct GeneralService {
    kernel: Weak<Kernel>,
}

impl Service for GeneralService {
    fn name(&self) -> &str {
        "nn::nifm::detail::IGeneralService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetClientId() -> pointer buffer with the ClientId (4 bytes)
            1 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let Some(out) = req.pointer_output(0) else {
                    return Err(Stop("GetClientId without a buffer".into()));
                };
                if out.size < 4 || !kernel.write_memory(out.addr, &1u32.to_le_bytes()) {
                    return Err(Stop("the buffer for the client id is not writable".into()));
                }
                Ok(Response::ok().with_pointer_reply(0, 4))
            }
            // CreateRequest(s32 requirement preset) -> IRequest
            4 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                Ok(Response::ok().with_object(Box::new(NetRequest::new(&kernel))))
            }
            // IsWirelessCommunicationEnabled() -> bool: the radio is on, nothing is connected
            17 => Ok(Response::ok().with_bytes(&[1])),
            // GetInternetConnectionStatus() -> (u8 interface type, u8 wifi strength, u8 status):
            // no interface, no signal, disconnected
            18 => Ok(Response::ok().with_bytes(&[0, 0, 0])),
            // IsAnyInternetRequestAccepted(ClientId) -> bool
            21 => Ok(Response::ok().with_bytes(&[0])),
            other => Err(Stop::unknown_command("IGeneralService", other)),
        }
    }
}

/// `nn::nifm::detail::IRequest`: a request for a connection that is never granted.
#[derive(Debug)]
struct NetRequest {
    kernel: Weak<Kernel>,
    /// The two events `GetSystemEventReadableHandles` returns; the first is set when the state
    /// changes, which a submitted request does once (it ends up not accepted).
    events: Arc<Mutex<[(KObject, KObject); 2]>>,
    submitted: bool,
}

impl NetRequest {
    fn new(kernel: &Arc<Kernel>) -> NetRequest {
        NetRequest {
            kernel: Arc::downgrade(kernel),
            events: Arc::new(Mutex::new([kernel.new_event(), kernel.new_event()])),
            submitted: false,
        }
    }
}

impl Service for NetRequest {
    fn name(&self) -> &str {
        "nn::nifm::detail::IRequest"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetRequestState() -> RequestState: 1 is Free, a request that was not accepted
            0 => Ok(Response::ok().with_u32(1)),
            // GetSystemEventReadableHandles() -> (event, event)
            2 => {
                let events = self.events.lock().unwrap_or_else(|e| e.into_inner());
                Ok(Response::ok()
                    .with_copy_handle(events[0].1.clone())
                    .with_copy_handle(events[1].1.clone()))
            }
            // Cancel()
            3 => Ok(Response::ok()),
            // Submit(): the state changes (and stays Free), so the first event is set
            4 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                self.submitted = true;
                let events = self.events.lock().unwrap_or_else(|e| e.into_inner());
                kernel.set_event(&events[0].0, true);
                Ok(Response::ok())
            }
            // SetRequirementPreset(s32), SetPriority(u32), SetRejectable(bool),
            // SetConnectionConfirmationOption(s32), SetPersistent(bool), SetInstant(bool),
            // SetSustainable(bool), SetGreedy(bool), SetSharable(bool): a request that is never
            // granted ignores what it was asked to be
            5 | 6 | 8 | 10 | 11 | 12 | 13 | 14 | 15 | 16 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("IRequest", other)),
        }
    }
}
