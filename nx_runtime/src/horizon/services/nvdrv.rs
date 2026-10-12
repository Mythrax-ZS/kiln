//! `nvdrv`, the graphics driver service (Switchbrew, "NV services").
//!
//! This is only the IPC half: the commands carry a descriptor and an ioctl number with buffers, and
//! everything they mean is `nx_gpu::nvdrv`. Kernel events are made here, on demand, for the tokens
//! the GPU model hands out.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use crate::rt::runtime::rt;
use nx_gpu::nvdrv::args::Fail;
use nx_gpu::nvdrv::{EventToken, Nvdrv};
use nx_gpu::{GuestMemory, NvError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// The guest process's memory as the GPU sees it: no CPU permissions in the way.
struct ProcessMemory;

impl GuestMemory for ProcessMemory {
    fn read(&self, addr: u64, out: &mut [u8]) -> bool {
        rt().mem.read_bytes(addr, out, false).is_ok()
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        rt().mem.write_bytes(addr, data, false).is_ok()
    }
}

/// The driver and the events behind its tokens, shared by every connection of the process.
pub struct Shared {
    core: Mutex<Nvdrv>,
    events: Mutex<HashMap<EventToken, (KObject, KObject)>>,
}

impl Shared {
    pub fn new() -> Shared {
        Shared {
            core: Mutex::new(Nvdrv::new(Arc::new(ProcessMemory))),
            events: Mutex::new(HashMap::new()),
        }
    }

    /// The (writable, readable) ends of the event behind `token`, made on first use.
    fn event(&self, kernel: &Kernel, token: EventToken) -> (KObject, KObject) {
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events
            .entry(token)
            .or_insert_with(|| kernel.new_event())
            .clone()
    }

    /// Apply what the driver asked for since the last call to the kernel's events.
    fn deliver_signals(&self, kernel: &Kernel) {
        let signals = self
            .core
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take_signals();
        for s in signals {
            let (writable, _) = self.event(kernel, s.token);
            kernel.set_event(&writable, s.signaled);
        }
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

/// `nns::nvdrv::INvDrvServices`.
#[derive(Debug)]
pub struct NvDrv {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl NvDrv {
    pub fn new(kernel: Weak<Kernel>, shared: Arc<Shared>) -> NvDrv {
        NvDrv { kernel, shared }
    }
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

fn stop(what: String) -> Stop {
    Stop(what)
}

impl NvDrv {
    fn core(&self) -> std::sync::MutexGuard<'_, Nvdrv> {
        self.shared.core.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// An ioctl with up to two input and two output buffers (`Ioctl`, `Ioctl2`, `Ioctl3`).
    fn ioctl(&mut self, req: &Request, variant: u32) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let (fd, code) = (req.u32_at(0), req.u32_at(4));
        let ioctl = nx_gpu::nvdrv::args::Ioctl(code);
        // The argument struct goes in the first input buffer when the number says it is sent, and
        // comes back in the first output buffer when it says it is read. Ioctl2 adds a second
        // input buffer; Ioctl3 a second output buffer.
        let input = if ioctl.has_input() {
            match req.input_buffer(0) {
                Some(b) if b.size > 0 => {
                    kernel.read_buffer(b, ioctl.size() as u64).ok_or_else(|| {
                        stop(format!("ioctl {code:#x}: the input buffer is unreadable"))
                    })?
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let extra_in = if variant == 11 {
            match req.input_buffer(1) {
                Some(b) if b.size > 0 => kernel.read_buffer(b, b.size).ok_or_else(|| {
                    stop(format!("ioctl {code:#x}: the second input is unreadable"))
                })?,
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let reply = self
            .core()
            .ioctl(fd, code, &input, &extra_in)
            .map_err(|why| stop(format!("nvdrv: {why}")))?;
        self.shared.deliver_signals(&kernel);
        if ioctl.has_output() {
            if let Some(b) = req.output_buffer(0).filter(|b| b.size > 0) {
                let n = (b.size as usize).min(ioctl.size()).min(reply.args.len());
                if !kernel.write_memory(b.addr, &reply.args.data[..n]) {
                    return Err(stop(format!("ioctl {code:#x}: the output is unwritable")));
                }
            }
        }
        if variant == 12 {
            if let Some(b) = req.output_buffer(1).filter(|b| b.size > 0) {
                let n = (b.size as usize).min(reply.extra_out.len());
                if !kernel.write_memory(b.addr, &reply.extra_out[..n]) {
                    return Err(stop(format!(
                        "ioctl {code:#x}: the second output is unwritable"
                    )));
                }
            }
        }
        Ok(Response::ok().with_u32(reply.error.0))
    }
}

impl Service for NvDrv {
    fn name(&self) -> &str {
        "nns::nvdrv::INvDrvServices"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Open(buffer<path>) -> (u32 fd, u32 error)
            0 => {
                let Some(b) = req.input_buffer(0).filter(|b| b.size > 0) else {
                    return Ok(Response::ok()
                        .with_u32(0)
                        .with_u32(NvError::BAD_PARAMETER.0));
                };
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let bytes = kernel
                    .read_buffer(b, 0x100)
                    .ok_or_else(|| stop("nvdrv Open: the path is unreadable".to_string()))?;
                let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
                let path = String::from_utf8_lossy(&bytes[..end]).into_owned();
                let opened = self.core().open(&path);
                match opened {
                    Ok(fd) => Ok(Response::ok().with_u32(fd).with_u32(0)),
                    Err(Fail::Nv(e)) => Ok(Response::ok().with_u32(0).with_u32(e.0)),
                    Err(Fail::Unimplemented(why)) => Err(stop(format!("nvdrv Open: {why}"))),
                }
            }
            // Ioctl(u32 fd, u32 code, buffer in, buffer out) -> u32 error
            1 => self.ioctl(req, 1),
            // Close(u32 fd) -> u32 error
            2 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let e = self.core().close(req.u32_at(0));
                self.shared.deliver_signals(&kernel);
                Ok(Response::ok().with_u32(e.0))
            }
            // Initialize(u32 transfer memory size, process handle, transfer memory handle)
            // -> u32 error. The driver keeps its own books; the memory lent is not needed.
            3 => Ok(Response::ok().with_u32(0)),
            // QueryEvent(u32 fd, u32 event id) -> (u32 error, event handle)
            4 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let queried = self.core().query_event(req.u32_at(0), req.u32_at(4));
                match queried {
                    Ok(token) => {
                        let (_, readable) = self.shared.event(&kernel, token);
                        Ok(Response::ok().with_u32(0).with_copy_handle(readable))
                    }
                    Err(Fail::Nv(e)) => Ok(Response::ok().with_u32(e.0)),
                    Err(Fail::Unimplemented(why)) => Err(stop(format!("nvdrv QueryEvent: {why}"))),
                }
            }
            // SetAruid(u64 applet resource user id, pid) -> u32 error
            8 => Ok(Response::ok().with_u32(0)),
            // Ioctl2(u32 fd, u32 code, two input buffers, output buffer) -> u32 error
            11 => self.ioctl(req, 11),
            // Ioctl3(u32 fd, u32 code, input buffer, two output buffers) -> u32 error
            12 => self.ioctl(req, 12),
            // SetGraphicsFirmwareMemoryMarginEnabled(u64 enabled)
            13 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("nvdrv", other)),
        }
    }
}
