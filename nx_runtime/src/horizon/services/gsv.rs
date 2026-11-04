//! The Switch 2's graphics supervisor (`nx2:gsv` in the NVN2 guide): the session a Switch 2
//! program's SDK asks the applet for (`ISelfController` command 2000, from
//! `nn::gsv::BootWithRemoteMemory`) and through which it reaches its GPU. It speaks TIPC
//! (docs/ASSUMPTIONS.md A-194).
//!
//! Only the commands a title was seen to send are answered; any other stops the run with its
//! number, as every service here does.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use super::nvdrv;
use std::sync::{Arc, Weak};

/// `nn::gsv`'s supervisor session.
#[derive(Debug)]
pub struct Supervisor {
    kernel: Weak<Kernel>,
    gpu: Arc<nvdrv::Shared>,
    /// Command 11 was sent.
    client_registered: bool,
}

impl Supervisor {
    pub fn new(kernel: Weak<Kernel>, gpu: Arc<nvdrv::Shared>) -> Supervisor {
        Supervisor {
            kernel,
            gpu,
            client_registered: false,
        }
    }
}

/// The layout of the syncpoint pages (A-196): in the read-only page, a header, the values from
/// 0x40 (eight bytes each), a copy of the summary at 0x1040 and the acknowledged bits at 0x1100;
/// in the read-write page, the summary at 0 and the armed bits at 0x100.
mod page {
    pub const VERSION: u32 = 1;
    pub const VALUES: u64 = 0x40;
    pub const CAPACITY: u32 = (0x1000 - 0x40) / 8;
    pub const RW_SUMMARY: u64 = 0;
    pub const RO_SUMMARY: u64 = 0x1040;
    pub const RW_ARMED: u64 = 0x100;
    pub const RO_ACKNOWLEDGED: u64 = 0x1100;
    pub const BITS: usize = 0x200;
    pub const RO_SIZE: u64 = 0x2000;
    /// How often the values are published and the armed bits looked at.
    pub const PERIOD: std::time::Duration = std::time::Duration::from_millis(1);
}

impl Supervisor {
    fn attach_page(&mut self, req: &Request) -> ServiceResult {
        let kernel = self
            .kernel
            .upgrade()
            .ok_or_else(|| Stop("the process is gone".into()))?;
        let lent = |i: usize| {
            req.copy_handles
                .get(i)
                .and_then(|o| kernel.transfer_memory(o))
        };
        let (Some((rw, _)), Some((ro, ro_size))) = (lent(0), lent(1)) else {
            return Err(Stop(
                "nx2:gsv 204 was not given its two transfer memories".into(),
            ));
        };
        if ro_size < page::RO_SIZE {
            return Err(Stop(format!(
                "nx2:gsv 204: the read-only page is {ro_size:#x} bytes, less than {:#x}",
                page::RO_SIZE
            )));
        }
        let count = req.u32_at(4).min(page::CAPACITY);
        let header = [
            page::VERSION,
            0,
            count,
            page::VALUES as u32,
            page::RW_SUMMARY as u32,
            page::RO_SUMMARY as u32,
            page::RW_ARMED as u32,
            page::RO_ACKNOWLEDGED as u32,
        ];
        let bytes: Vec<u8> = header.iter().flat_map(|w| w.to_le_bytes()).collect();
        if !kernel.write_lent_memory(ro, &bytes) {
            return Err(Stop("nx2:gsv 204: the read-only page is not mapped".into()));
        }
        let (request, _) = kernel.new_event();
        let (reply, reply_readable) = kernel.new_event();
        publish(
            Arc::downgrade(&kernel),
            self.gpu.clone(),
            rw,
            ro,
            count,
            reply,
        );
        Ok(Response::ok()
            .with_copy_handle(request)
            .with_copy_handle(reply_readable))
    }
}

/// The thread that keeps the read-only page current: every period it writes the syncpoints'
/// values and, when the program has armed bits it has not had acknowledged, acknowledges them,
/// copies the summary and sets the reply event (A-196).
fn publish(
    kernel: Weak<Kernel>,
    gpu: Arc<nvdrv::Shared>,
    rw: u64,
    ro: u64,
    count: u32,
    reply: KObject,
) {
    let mut next = std::time::Instant::now();
    std::thread::spawn(move || {
        let mut values = vec![0u32; count.min(nx_gpu::syncpt::COUNT as u32) as usize];
        let (mut armed, mut acknowledged) = (vec![0u8; page::BITS], vec![0u8; page::BITS]);
        loop {
            next += page::PERIOD;
            super::super::kernel::sleep_until(next);
            let Some(kernel) = kernel.upgrade() else {
                break;
            };
            if kernel.exit_state().is_some() {
                break;
            }
            gpu.syncpoint_values(&mut values);
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| u64::from(*v).to_le_bytes())
                .collect();
            if !kernel.write_lent_memory(ro + page::VALUES, &bytes)
                || !kernel.read_memory(rw + page::RW_ARMED, &mut armed)
                || !kernel.read_memory(ro + page::RO_ACKNOWLEDGED, &mut acknowledged)
            {
                // The program took its pages back.
                break;
            }
            if armed == acknowledged {
                continue;
            }
            let mut summary = [0u8; 8];
            let _ = kernel.read_memory(rw + page::RW_SUMMARY, &mut summary);
            let _ = kernel.write_lent_memory(ro + page::RO_ACKNOWLEDGED, &armed)
                && kernel.write_lent_memory(ro + page::RO_SUMMARY, &summary);
            kernel.set_event(&reply, true);
        }
    });
}

/// `nx2:syncpt`: waits on the GPU's syncpoints through numbered slots, each with an event
/// (A-195). A syncpoint is named by its bitwise NOT.
#[derive(Debug)]
pub struct Syncpt {
    kernel: Weak<Kernel>,
    gpu: Arc<nvdrv::Shared>,
}

/// The counters the service says it has (command 1).
const COUNTERS: u32 = 32;

impl Syncpt {
    fn handle(&mut self, req: &Request) -> ServiceResult {
        let kernel = self
            .kernel
            .upgrade()
            .ok_or_else(|| Stop("the process is gone".into()))?;
        let syncpoint = |at: usize| !req.u32_at(at);
        match req.command {
            // A handshake: the u64 the program sends comes back.
            0 => Ok(Response::ok().with_u64(req.u64_at(0))),
            1 => Ok(Response::ok().with_u32(COUNTERS)),
            21 => Ok(Response::ok()),
            // GetSlotEvent(u32 slot) -> event
            22 => Ok(
                Response::ok().with_copy_handle(self.gpu.nx2_slot_event(&kernel, req.u32_at(0)))
            ),
            // Read(u32 !syncpoint) -> (u32 value, u32, u32)
            103 => {
                let id = syncpoint(0);
                let value = self.gpu.syncpoint_value(id).ok_or_else(|| {
                    Stop(format!(
                        "nx2:syncpt read {:#x}, which names no syncpoint",
                        !id
                    ))
                })?;
                Ok(Response::ok().with_u32(value).with_u32(0).with_u32(0))
            }
            // Wait(u32 slot, u32 !syncpoint, u32 threshold): the slot's event is set when the
            // syncpoint reaches the threshold.
            201 => {
                let (slot, id, threshold) = (req.u32_at(0), syncpoint(4), req.u32_at(8));
                if !self.gpu.nx2_slot_wait(&kernel, slot, id, threshold) {
                    return Err(Stop(format!(
                        "nx2:syncpt wait on {:#x}, which names no syncpoint (a counter is not modelled)",
                        !id
                    )));
                }
                Ok(Response::ok())
            }
            // CancelWait(u32 slot): the wait ends and the slot's event is set.
            202 => {
                self.gpu.nx2_slot_cancel(&kernel, req.u32_at(0));
                Ok(Response::ok())
            }
            other => Err(Stop::unknown_command("nx2:syncpt", other)),
        }
    }
}

impl Service for Syncpt {
    fn name(&self) -> &str {
        "nx2:syncpt"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        self.handle(req)
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        self.handle(req)
    }
}

/// `nx2:nvsrv`: the GPU driver's session, which, unlike the supervisor, speaks CMIF: `nvdrv`'s
/// commands (Open, Ioctl, Close, Initialize, QueryEvent and the rest) with the same arguments
/// (A-194).
#[derive(Debug)]
pub struct NvSrv(nvdrv::NvDrv);

impl Service for NvSrv {
    fn name(&self) -> &str {
        "nx2:nvsrv"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        self.0.request(req)
    }
}

impl Service for Supervisor {
    fn name(&self) -> &str {
        "nx2:gsv"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop(format!(
            "the graphics supervisor was sent a CMIF message (command {}); it speaks TIPC",
            req.command
        )))
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // nn::gsv::BootWithRemoteMemory: the program's remote memory (the region of
            // `GetInfo` 29 and 30) and a handle, for a graphics server in another process to map.
            // The GPU here runs in the program's own process and reads its memory where it is:
            // there is nothing to map (A-194). 101 follows it with no arguments.
            4 | 101 => Ok(Response::ok()),
            // RegisterClientHandle(copied process handle): once per session.
            11 => {
                if std::mem::replace(&mut self.client_registered, true) {
                    return Err(Stop(
                        "nx2:gsv 11 (RegisterClientHandle) was sent a second time".into(),
                    ));
                }
                Ok(Response::ok())
            }
            // Permission checks: each tests a bit of what the session was opened with and fails
            // when it is clear; a program asks only for what it was granted, so all pass.
            201 | 301 | 401..=403 | 494 | 801 | 1101 | 1102 | 1501..=1503 | 1601 => {
                Ok(Response::ok())
            }
            // QueryGrantedFeatures(u64 requested) -> u64 granted: everything asked for (a zero
            // answer would deny every feature), from nn::gs::sdk::BootNativeMemoryManagement.
            115 => Ok(Response::ok().with_u64(req.u64_at(0))),
            // AttachSyncpointPage(u32 version, u32 count, u64 rw size, u64 ro size, rw transfer
            // memory, ro transfer memory) -> (request event, reply event) (A-196).
            204 => self.attach_page(req),
            // The syncpoint session (A-195).
            202 => Ok(
                Response::ok().with_copy_handle(KObject::session(Box::new(Syncpt {
                    kernel: self.kernel.clone(),
                    gpu: self.gpu.clone(),
                }))),
            ),
            // nn::nvsrv::BootWithRemoteMemory: a session of the GPU driver (A-194).
            1402 => Ok(
                Response::ok().with_copy_handle(KObject::session(Box::new(NvSrv(
                    nvdrv::NvDrv::new(self.kernel.clone(), self.gpu.clone()),
                )))),
            ),
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(command: u32) -> Request {
        Request {
            command,
            data: vec![0; 48],
            ..Request::default()
        }
    }

    #[test]
    fn the_supervisor_answers_what_it_knows_over_tipc_and_stops_on_the_rest() {
        let mut s = Supervisor::new(Weak::new(), nvdrv::Shared::new());
        assert!(s.request_tipc(&req(4)).is_ok(), "BootWithRemoteMemory");
        assert!(s.request_tipc(&req(11)).is_ok() && s.request_tipc(&req(101)).is_ok());
        assert!(
            s.request_tipc(&req(11)).is_err(),
            "the client is registered once"
        );
        assert!(s.request_tipc(&req(1101)).is_ok(), "a permission check");
        let mut asked = req(115);
        asked.data = 6u64.to_le_bytes().to_vec();
        assert_eq!(
            s.request_tipc(&asked).unwrap().data,
            6u64.to_le_bytes(),
            "all granted"
        );
        let stop = s.request_tipc(&req(12)).unwrap_err();
        assert!(stop.0.contains("command 12"), "{}", stop.0);
        assert!(
            s.request(&req(4)).is_err(),
            "a CMIF message is not the supervisor's"
        );
    }
}
