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
use super::{nvdrv, vi};
use std::sync::{Arc, Weak};

/// `nn::gsv`'s supervisor session.
#[derive(Debug)]
pub struct Supervisor {
    kernel: Weak<Kernel>,
    gpu: Arc<nvdrv::Shared>,
    display: Arc<vi::Shared>,
    /// Command 11 was sent.
    client_registered: bool,
    /// What the `nx2:gpu` sessions handed out (497) know: the address spaces 434 attaches.
    gpu_state: Arc<std::sync::Mutex<GpuState>>,
    /// The GPU address spaces attached with 434, by id.
    attached: std::collections::BTreeSet<u32>,
}

impl Supervisor {
    pub fn new(
        kernel: Weak<Kernel>,
        gpu: Arc<nvdrv::Shared>,
        display: Arc<vi::Shared>,
    ) -> Supervisor {
        Supervisor {
            kernel,
            gpu,
            display,
            client_registered: false,
            gpu_state: Arc::default(),
            attached: Default::default(),
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

/// The events `nx2:gpu` jobs complete on, by id modulo this.
const GPU_EVENT_IDS: u32 = 32;
/// The GPU addresses NVN2 uses: its address space's user partition (A-197).
const GPU_VA_START: u64 = 0x800_0000;
const GPU_VA_END: u64 = 0xC0_0000_0000;

/// What `nx2:gpu` and its sub-sessions share (A-197).
#[derive(Debug, Default)]
pub struct GpuState {
    /// The (writable, readable) ends of the event of each id, made on first use.
    events: std::collections::HashMap<u32, (KObject, KObject)>,
    /// The graphics devices made with 41, by id.
    devices: std::collections::BTreeSet<u8>,
    /// The GPU address spaces made with 33, by id.
    spaces: std::collections::BTreeSet<u32>,
    /// The GPU memory made with 101, by handle: its CPU address and size.
    memories: std::collections::BTreeMap<u32, (u64, u64)>,
    /// The handle the next memory gets.
    next_memory: u32,
}

/// The graphics devices there may be at once: ids 1 to 63 (A-197).
const GPU_DEVICE_IDS: u8 = 64;

/// `nx2:gpu`: GPU memory, mapped into the GPU's address space, and the events its jobs complete
/// on (A-197).
#[derive(Debug)]
pub struct GpuSession {
    kernel: Weak<Kernel>,
    #[allow(dead_code)]
    gpu: Arc<nvdrv::Shared>,
    state: Arc<std::sync::Mutex<GpuState>>,
}

impl GpuSession {
    /// Set the event of a job's slot: nothing runs elsewhere, so a job is done once it is asked
    /// for (an unmap does not wait for the work queued before it; A-201).
    fn complete(&self, kernel: &Kernel, slot: u32) {
        kernel.set_event(&self.event(kernel, slot).0, true);
    }

    /// The memory 101 made under the handle: its CPU address and size.
    fn memory(&self, req: &Request, handle: u32) -> Result<(u64, u64), Stop> {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.memories.get(&handle).copied().ok_or_else(|| {
            Stop(format!(
                "nx2:gpu {} of the GPU memory {handle}, which 101 did not make",
                req.command
            ))
        })
    }

    /// The bytes of `[first, last]` from a request, or a stop when the range is empty.
    fn span(req: &Request, first: u64, last: u64) -> Result<u64, Stop> {
        match last.checked_sub(first) {
            Some(d) if d < u64::MAX => Ok(d + 1),
            _ => Err(Stop(format!(
                "nx2:gpu {} of the range [{first:#x}, {last:#x}]",
                req.command
            ))),
        }
    }

    /// Stop unless the request's u32 at 0 names an address space made with 33.
    fn space(&self, req: &Request) -> Result<u32, Stop> {
        let id = req.u32_at(0);
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !st.spaces.contains(&id) {
            return Err(Stop(format!(
                "nx2:gpu {} of the address space {id}, which 33 did not make",
                req.command
            )));
        }
        Ok(id)
    }

    fn event(&self, kernel: &Kernel, id: u32) -> (KObject, KObject) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.events
            .entry(id % GPU_EVENT_IDS)
            .or_insert_with(|| kernel.new_event())
            .clone()
    }
}

impl Service for GpuSession {
    fn name(&self) -> &str {
        "nx2:gpu"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop(format!(
            "nx2:gpu was sent a CMIF message (command {}); it speaks TIPC",
            req.command
        )))
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        let kernel = self
            .kernel
            .upgrade()
            .ok_or_else(|| Stop("the process is gone".into()))?;
        match req.command {
            // GetIdRange -> (u32 first, u32 last): the ids whose events jobs complete on.
            11 => Ok(Response::ok().with_u32(0).with_u32(GPU_EVENT_IDS - 1)),
            // GetIdEvent(u32 id) -> event
            12 => Ok(Response::ok().with_copy_handle(self.event(&kernel, req.u32_at(0)).1)),
            // Start(u32 id): nothing runs elsewhere, so the id's event is set at once.
            21 => {
                kernel.set_event(&self.event(&kernel, req.u32_at(0)).0, true);
                Ok(Response::ok())
            }
            // OpenSubSession: another session on the same state, whose jobs set the same events.
            22 => Ok(
                Response::ok().with_copy_handle(KObject::session(Box::new(GpuSession {
                    kernel: self.kernel.clone(),
                    gpu: self.gpu.clone(),
                    state: self.state.clone(),
                }))),
            ),
            // CreateGpuAddressSpace(u32 flags) -> u32 id, from nn::gpu::sdk::CreateGpuAddressSpace
            // (the SDK keeps eight): an address space of the GPU, which 202, 203 and the
            // supervisor's 434 name by its id, the lowest free one from 1. Only flags 0 is known.
            33 => {
                if req.u32_at(0) != 0 {
                    return Err(Stop(format!(
                        "nx2:gpu 33 makes an address space with the flags {:#x}; only 0 is known",
                        req.u32_at(0)
                    )));
                }
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                let id = (1..).find(|id| !st.spaces.contains(id)).unwrap_or(u32::MAX);
                st.spaces.insert(id);
                Ok(Response::ok().with_u32(id))
            }
            // CreateGraphicsDevice(u64 name, u64 flags, u64 debug label) -> u8 id, from
            // nn::gpu::sdk::CreateGraphicsDevice (NvRm's `NvRmGpu`): the lowest free id from 1,
            // which the SDK keeps as a byte; 0 names no device.
            41 => {
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                let Some(id) = (1..GPU_DEVICE_IDS).find(|id| !st.devices.contains(id)) else {
                    return Err(Stop(format!(
                        "nx2:gpu 41: all {} graphics devices are made",
                        GPU_DEVICE_IDS - 1
                    )));
                };
                st.devices.insert(id);
                Ok(Response::ok().with_u32(u32::from(id)))
            }
            // CreateGpuAccessibleMemory(u64 start, u64 last, u32, u32) -> u32 handle, after the
            // SVC 0x80 that lets the GPU at the memory: the CPU memory from start to last becomes
            // GPU memory, mapped at its own address, which 102 and 211 name by the handle (from
            // 1). The two u32s are not known and not used (A-201).
            101 => {
                let (start, last) = (req.u64_at(0), req.u64_at(8));
                let len = Self::span(req, start, last)?;
                self.gpu
                    .switch_2_map(start, len, start, 0)
                    .map_err(|e| Stop(format!("nx2:gpu 101 of [{start:#x}, {last:#x}]: {e}")))?;
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                st.next_memory += 1;
                let handle = st.next_memory;
                st.memories.insert(handle, (start, len));
                Ok(Response::ok().with_u32(handle))
            }
            // DestroyGpuAccessibleMemory(u32 handle, u32 slot): the memory is GPU memory no more
            // (the program reuses it at once) and its mapping at its own address goes.
            102 => {
                let (handle, slot) = (req.u32_at(0), req.u32_at(4));
                let (start, len) = self.memory(req, handle)?;
                self.gpu
                    .switch_2_unmap(start, len)
                    .map_err(|e| Stop(format!("nx2:gpu 102 of the memory {handle}: {e}")))?;
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                st.memories.remove(&handle);
                drop(st);
                self.complete(&kernel, slot);
                Ok(Response::ok())
            }
            // NormalMapGpuMemory4K(u32 address space, u64 first, u64 last, u32 handle, u64 offset,
            // ..., u32 slot at 60): from first to last, the address space becomes a window onto
            // the handle's memory from the offset on.
            211 => {
                self.space(req)?;
                let (first, last) = (req.u64_at(4), req.u64_at(12));
                let (handle, offset, slot) = (req.u32_at(20), req.u64_at(24), req.u32_at(60));
                let len = Self::span(req, first, last)?;
                let (start, size) = self.memory(req, handle)?;
                if offset.checked_add(len).is_none_or(|end| end > size) {
                    return Err(Stop(format!(
                        "nx2:gpu 211 maps {len:#x} bytes from {offset:#x} into the memory \
                         {handle} of {size:#x} bytes"
                    )));
                }
                self.gpu
                    .switch_2_map(first, len, start + offset, 0)
                    .map_err(|e| Stop(format!("nx2:gpu 211 of [{first:#x}, {last:#x}]: {e}")))?;
                self.complete(&kernel, slot);
                Ok(Response::ok())
            }
            // NormalModifyMemoryKind4K(u32 address space, u64 first, u64 last, u32 kind, u32
            // fallback kind, u32 sync, u32 slot) -> u32 kind applied: a mapping's kind changes
            // nothing in the GPU front here (A-177), so the kind asked for is applied.
            212 => {
                self.space(req)?;
                Self::span(req, req.u64_at(4), req.u64_at(12))?;
                self.complete(&kernel, req.u32_at(32));
                Ok(Response::ok().with_u32(req.u32_at(20) & 0xFF))
            }
            // NormalUnmapGpuMemory4K(u32 address space, u64 first, u64 last, ..., u32 slot at 24)
            213 => {
                self.space(req)?;
                let (first, last) = (req.u64_at(4), req.u64_at(12));
                let len = Self::span(req, first, last)?;
                self.gpu
                    .switch_2_unmap(first, len)
                    .map_err(|e| Stop(format!("nx2:gpu 213 of [{first:#x}, {last:#x}]: {e}")))?;
                self.complete(&kernel, req.u32_at(24));
                Ok(Response::ok())
            }
            // DestroyGraphicsDevice(u8 id)
            42 => {
                let id = req.u32_at(0) as u8;
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if !st.devices.remove(&id) {
                    return Err(Stop(format!(
                        "nx2:gpu 42 of the graphics device {id}, which is not made"
                    )));
                }
                Ok(Response::ok())
            }
            // GetUserPartition -> (u64 start, u64 size)
            201 => Ok(Response::ok()
                .with_u64(GPU_VA_START)
                .with_u64(GPU_VA_END - GPU_VA_START)),
            // GetGsvPartition(u32 address space) -> empty, minimum above maximum.
            202 => {
                self.space(req)?;
                Ok(Response::ok().with_u64(u64::MAX).with_u64(0))
            }
            // GetVaRange(u32 address space) -> (u64 kind, u64 start, u64 size): the SDK takes a
            // kind of 0 for no range and stops at 5 or more.
            203 => {
                self.space(req)?;
                Ok(Response::ok()
                    .with_u64(1)
                    .with_u64(GPU_VA_START)
                    .with_u64(GPU_VA_END - GPU_VA_START))
            }
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

/// `nx2:vi2`: the display, as a Switch 2 program sees it (A-198).
#[derive(Debug)]
pub struct Vi2Session {
    kernel: Weak<Kernel>,
    display: Arc<vi::Shared>,
    /// The layers made with 201, by the client's token.
    layers: std::collections::BTreeSet<u64>,
    /// This session's keys in the display's list of vsync listeners: this, with the slot in the
    /// low byte.
    key: u64,
    /// The (writable, readable) ends of each native event slot's event, once acquired (31).
    native: [Option<(KObject, KObject)>; VI2_NATIVE_EVENTS],
}

/// The buffer slots of a layer, as the bits `vi2` hands back.
const VI2_ALL_SLOTS: u64 = 0x7FFF;
/// The displays `vi2` has and the timestamp recorders it says it has.
const VI2_DISPLAYS: u32 = 1;
const VI2_TIMESTAMP_RECORDERS: u32 = 7;
/// The native event slots of a session (31, 32 and the bindings of 261 to 272).
const VI2_NATIVE_EVENTS: usize = 8;

/// The eight-byte name (of a display or a layer stack) at `at` of a request, without its zeros.
fn vi2_name(req: &Request, at: usize) -> Vec<u8> {
    let raw = req.data.get(at..at + 8).unwrap_or_default();
    raw[..raw.iter().position(|&b| b == 0).unwrap_or(raw.len())].to_vec()
}

/// Stop unless the request names the one display, `default`, at `at`.
fn vi2_default_display(req: &Request, at: usize) -> Result<(), Stop> {
    let name = vi2_name(req, at);
    if name != b"default" {
        return Err(Stop(format!(
            "nx2:vi2 {} of the display `{}`; only `default` exists",
            req.command,
            String::from_utf8_lossy(&name)
        )));
    }
    Ok(())
}

impl Vi2Session {
    pub fn new(kernel: Weak<Kernel>, display: Arc<vi::Shared>) -> Vi2Session {
        static SESSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Vi2Session {
            kernel,
            display,
            layers: Default::default(),
            key: SESSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) << 8,
            native: Default::default(),
        }
    }

    /// The native event slot named by the request's u32 at `at`.
    fn slot(req: &Request, at: usize) -> Result<usize, Stop> {
        let slot = req.u32_at(at) as usize;
        if slot >= VI2_NATIVE_EVENTS {
            return Err(Stop(format!(
                "nx2:vi2 {} of the native event {slot}; a session has {VI2_NATIVE_EVENTS}",
                req.command
            )));
        }
        Ok(slot)
    }

    /// The (writable, readable) ends of an acquired slot's event.
    fn acquired(&self, req: &Request, slot: usize) -> Result<(KObject, KObject), Stop> {
        self.native[slot].clone().ok_or_else(|| {
            Stop(format!(
                "nx2:vi2 {} of the native event {slot}, which was not acquired",
                req.command
            ))
        })
    }

    fn listener(&self, slot: usize) -> u64 {
        self.key | slot as u64
    }
}

impl Drop for Vi2Session {
    fn drop(&mut self) {
        for slot in 0..VI2_NATIVE_EVENTS {
            self.display.unlisten_vsync(self.listener(slot));
        }
    }
}

impl Service for Vi2Session {
    fn name(&self) -> &str {
        "nx2:vi2"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop(format!(
            "nx2:vi2 was sent a CMIF message (command {}); it speaks TIPC",
            req.command
        )))
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        let kernel = self
            .kernel
            .upgrade()
            .ok_or_else(|| Stop("the process is gone".into()))?;
        match req.command {
            // Handshake(u64) -> the same u64
            0 => Ok(Response::ok().with_u64(req.u64_at(0))),
            1 | 3 => Ok(Response::ok().with_u32(VI2_DISPLAYS)),
            // GetTimestampRecorderCount -> u32 (the SDK's stub stops the program at 8 or more).
            5 => Ok(Response::ok().with_u32(VI2_TIMESTAMP_RECORDERS)),
            // An event the program waits on whose cause is not known: nothing sets it.
            11 => Ok(Response::ok().with_copy_handle(kernel.new_event().1)),
            // (u32 count) from nn::vi2::CreateTimestampRecorder, before its first recorder: how many
            // of the recorders of 5 the program uses, the count of 5 or 7 if that is less; only
            // the result is read.
            51 => {
                if req.u32_at(0) > VI2_TIMESTAMP_RECORDERS {
                    return Err(Stop(format!(
                        "nx2:vi2 51 asks for {} timestamp recorders of {VI2_TIMESTAMP_RECORDERS}",
                        req.u32_at(0)
                    )));
                }
                Ok(Response::ok())
            }
            // AcquireNativeEvent(u32 slot) -> event: the slot's event, made the first time. Nothing
            // sets it until the slot is bound to a source of the display (261 to 271).
            31 => {
                let slot = Self::slot(req, 0)?;
                let (_, readable) = self.native[slot]
                    .get_or_insert_with(|| kernel.new_event())
                    .clone();
                Ok(Response::ok().with_copy_handle(readable))
            }
            // DuplicateNativeEvent(handle, u32 slot) -> event: the client hands back the handle of
            // an acquired slot and gets the slot's event again.
            32 => {
                let slot = Self::slot(req, 0)?;
                Ok(Response::ok().with_copy_handle(self.acquired(req, slot)?.1))
            }
            // OpenDisplay(char name[8], u32), from nn::vi::detail::DefaultDisplay::OnOpen: the one
            // display is "default"; only the result is read.
            101 => {
                vi2_default_display(req, 0)?;
                Ok(Response::ok())
            }
            // QueryDisplayRefreshRate(display name) -> u64: 1 is 60 Hz (2 is 120 Hz and bit 63
            // none, the SDK's conversion to nn::vi::RefreshRate).
            103 => {
                vi2_default_display(req, 0)?;
                Ok(Response::ok().with_u64(1))
            }
            // QueryDisplayHdrStatus(display name) -> u64 format, u16: format 1 is SDR output (2 is
            // HDR); the SDK passes the u16 on, taken for the panel's peak luminance in nits.
            104 => {
                vi2_default_display(req, 0)?;
                Ok(Response::ok()
                    .with_u64(1)
                    .with_bytes(&VI2_PEAK_NITS.to_le_bytes()))
            }
            // QueryDisplayHdrPaperWhite(display name) -> u16 nits
            105 => {
                vi2_default_display(req, 0)?;
                Ok(Response::ok().with_bytes(&VI2_PAPER_WHITE_NITS.to_le_bytes()))
            }
            // QueryDisplayBufferMaxResolution(display name) -> u64: 2 is 1920x1080, the handheld
            // screen's (1 is 1280x720, 4 2560x1440 and 8 3840x2160 in the SDK's table).
            106 => {
                vi2_default_display(req, 0)?;
                Ok(Response::ok().with_u64(2))
            }
            // (char display[8], char stack[8], u64 layer): the layer joins a layer stack of the
            // display (Layer::OnOpen); one layer is shown here, so there is nothing to order. Only
            // the result is read.
            111 => {
                vi2_default_display(req, 0)?;
                if !self.layers.contains(&req.u64_at(16)) {
                    return Err(Stop(format!(
                        "nx2:vi2 111 of the unknown layer {:#x}",
                        req.u64_at(16)
                    )));
                }
                Ok(Response::ok())
            }
            // CreateLayer(u64 client token, u32, u32), from nn::vi::CreateLayer through
            // nn::vi::Layer::OnOpen: the token names the layer from then on; only the result is
            // read.
            201 => {
                if !self.layers.insert(req.u64_at(0)) {
                    return Err(Stop(format!(
                        "nx2:vi2 201 made the layer {:#x} a second time",
                        req.u64_at(0)
                    )));
                }
                Ok(Response::ok())
            }
            // GetLayerBufferBits(u64 layer) -> u64: the slots the client may draw into, all of
            // them while the display holds none.
            212 => {
                if !self.layers.contains(&req.u64_at(0)) {
                    return Err(Stop(format!(
                        "nx2:vi2 212 of the unknown layer {:#x}",
                        req.u64_at(0)
                    )));
                }
                Ok(Response::ok().with_u64(VI2_ALL_SLOTS))
            }
            // (u64 layer, char display[8]): the layer goes on the display (from Layer::OnOpen
            // too); only the result is read.
            251 => {
                vi2_default_display(req, 8)?;
                if !self.layers.contains(&req.u64_at(0)) {
                    return Err(Stop(format!(
                        "nx2:vi2 251 of the unknown layer {:#x}",
                        req.u64_at(0)
                    )));
                }
                Ok(Response::ok())
            }
            // Bind(display name, u32 slot): 261 to the vsync, which sets the slot at every vblank
            // from then on; 263 the refresh rate, 265 the HDR status, 267 the HDR paper white, 269
            // the largest buffer resolution and 271 the mode capability. None of those changes
            // here, so their slot is set once, when it is bound: the SDK asks for a value only
            // after its event was set, and keeps the one it has otherwise.
            261 | 263 | 265 | 267 | 269 | 271 => {
                vi2_default_display(req, 0)?;
                let slot = Self::slot(req, 8)?;
                let (writable, _) = self.acquired(req, slot)?;
                self.display.unlisten_vsync(self.listener(slot));
                if req.command == 261 {
                    self.display
                        .listen_vsync(&kernel, self.listener(slot), writable);
                } else {
                    kernel.set_event(&writable, true);
                }
                Ok(Response::ok())
            }
            // Unbind(display name, u32 slot), the even numbers after each: the slot is no longer
            // set by its source.
            262 | 264 | 266 | 268 | 270 | 272 => {
                vi2_default_display(req, 0)?;
                let slot = Self::slot(req, 8)?;
                self.acquired(req, slot)?;
                self.display.unlisten_vsync(self.listener(slot));
                Ok(Response::ok())
            }
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

/// What `vi2` says of the panel's luminance: its peak and its reference white, in nits (A-198).
const VI2_PEAK_NITS: u16 = 400;
const VI2_PAPER_WHITE_NITS: u16 = 203;

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
            // AttachGpuAddressSpace(u32 id), from nn::gpu::sdk::CreateGpuAddressSpace: the
            // graphics system takes the address space nx2:gpu 33 made; only the result is read.
            434 => {
                let id = req.u32_at(0);
                let made = {
                    let st = self.gpu_state.lock().unwrap_or_else(|e| e.into_inner());
                    st.spaces.contains(&id)
                };
                if !made || !self.attached.insert(id) {
                    return Err(Stop(format!(
                        "nx2:gsv 434 of the GPU address space {id}, {}",
                        if made {
                            "attached before"
                        } else {
                            "which nx2:gpu did not make"
                        }
                    )));
                }
                Ok(Response::ok())
            }
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
            // OpenGpuSession(u64): the GPU memory session (A-197).
            497 => Ok(
                Response::ok().with_copy_handle(KObject::session(Box::new(GpuSession {
                    kernel: self.kernel.clone(),
                    gpu: self.gpu.clone(),
                    state: self.gpu_state.clone(),
                }))),
            ),
            // OpenViSession(u64): presentation (A-198).
            305 => Ok(Response::ok().with_copy_handle(KObject::session(Box::new(
                Vi2Session::new(self.kernel.clone(), self.display.clone()),
            )))),
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
        let mut s = Supervisor::new(
            Weak::new(),
            nvdrv::Shared::new(),
            Arc::new(vi::Shared::new()),
        );
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

    #[test]
    fn graphics_devices_take_the_lowest_free_id_from_1() {
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        let kernel = Kernel::new(ProcessInfo::new("gpu", ProcessKind::Nso, None));
        let mut g = GpuSession {
            kernel: Arc::downgrade(&kernel),
            gpu: nvdrv::Shared::new(),
            state: Arc::default(),
        };
        let make = |g: &mut GpuSession| g.request_tipc(&req(41)).unwrap().data;
        assert_eq!(make(&mut g), 1u32.to_le_bytes());
        assert_eq!(make(&mut g), 2u32.to_le_bytes());
        let mut destroy = req(42);
        destroy.data = 1u32.to_le_bytes().to_vec();
        assert!(g.request_tipc(&destroy).is_ok());
        assert!(
            g.request_tipc(&destroy).is_err(),
            "a device is destroyed once"
        );
        assert_eq!(make(&mut g), 1u32.to_le_bytes(), "the lowest free id");
    }

    #[test]
    fn a_gpu_address_space_is_made_then_attached_to_the_graphics_system() {
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        let kernel = Kernel::new(ProcessInfo::new("gpu", ProcessKind::Nso, None));
        let mut s = Supervisor::new(
            Arc::downgrade(&kernel),
            nvdrv::Shared::new(),
            Arc::new(vi::Shared::new()),
        );
        let mut g = GpuSession {
            kernel: Arc::downgrade(&kernel),
            gpu: s.gpu.clone(),
            state: s.gpu_state.clone(),
        };
        let with = |command, v: u32| Request {
            command,
            data: v.to_le_bytes().to_vec(),
            ..Request::default()
        };
        assert!(s.request_tipc(&with(434, 1)).is_err(), "not made yet");
        assert_eq!(
            g.request_tipc(&with(33, 0)).unwrap().data,
            1u32.to_le_bytes()
        );
        assert!(g.request_tipc(&with(203, 1)).is_ok() && g.request_tipc(&with(202, 1)).is_ok());
        assert!(g.request_tipc(&with(203, 2)).is_err(), "no address space 2");
        assert!(
            g.request_tipc(&with(33, 4)).is_err(),
            "flags that are not known"
        );
        assert!(s.request_tipc(&with(434, 1)).is_ok());
        assert!(s.request_tipc(&with(434, 1)).is_err(), "attached once");
    }

    #[test]
    fn gpu_memory_is_mapped_at_its_own_address_and_where_211_puts_it() {
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        let kernel = Kernel::new(ProcessInfo::new("gpu", ProcessKind::Nso, None));
        let gpu = nvdrv::Shared::new();
        gpu.set_switch_2(true);
        let mut g = GpuSession {
            kernel: Arc::downgrade(&kernel),
            gpu: gpu.clone(),
            state: Arc::default(),
        };
        let data = |command: u32, fields: &[(usize, u64, usize)]| {
            let mut data = vec![0u8; 64];
            for &(at, value, width) in fields {
                data[at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
            }
            Request {
                command,
                data,
                ..Request::default()
            }
        };
        let space = g.request_tipc(&data(33, &[])).unwrap().data;
        assert_eq!(space, 1u32.to_le_bytes());
        // 101: two pages of CPU memory at 0x30_0003_5000, which become GPU memory there too.
        let made = data(
            101,
            &[(0, 0x30_0003_5000, 8), (8, 0x30_0003_6FFF, 8), (20, 1, 4)],
        );
        let handle = g.request_tipc(&made).unwrap().data;
        assert_eq!(handle, 1u32.to_le_bytes());
        assert_eq!(gpu.switch_2_translate(0x30_0003_6004), Some(0x30_0003_6004));
        // 211: the second page at GPU address 0x1000_0000, its event set.
        let map = data(
            211,
            &[
                (0, 1, 4),
                (4, 0x1000_0000, 8),
                (12, 0x1000_0FFF, 8),
                (20, 1, 4),
                (24, 0x1000, 8),
                (60, 3, 4),
            ],
        );
        g.request_tipc(&map).unwrap();
        assert_eq!(gpu.switch_2_translate(0x1000_0010), Some(0x30_0003_6010));
        assert!(kernel.event_signaled(&g.event(&kernel, 3).0));
        // Past the memory's end, of an address space or a memory that was not made: stops.
        let past = data(
            211,
            &[
                (0, 1, 4),
                (4, 0x1000_0000, 8),
                (12, 0x1000_1FFF, 8),
                (20, 1, 4),
                (24, 0x1000, 8),
            ],
        );
        assert!(g.request_tipc(&past).is_err());
        let other = data(
            211,
            &[
                (0, 2, 4),
                (4, 0x1000_0000, 8),
                (12, 0x1000_0FFF, 8),
                (20, 1, 4),
            ],
        );
        assert!(g.request_tipc(&other).is_err());
        // 213 unmaps; 102 destroys the memory and its mapping at its own address.
        g.request_tipc(&data(
            213,
            &[
                (0, 1, 4),
                (4, 0x1000_0000, 8),
                (12, 0x1000_0FFF, 8),
                (24, 4, 4),
            ],
        ))
        .unwrap();
        assert_eq!(gpu.switch_2_translate(0x1000_0010), None);
        g.request_tipc(&data(102, &[(0, 1, 4), (4, 5, 4)])).unwrap();
        assert_eq!(gpu.switch_2_translate(0x30_0003_5000), None);
        assert!(
            g.request_tipc(&data(102, &[(0, 1, 4)])).is_err(),
            "destroyed once"
        );
    }

    /// A vi2 request naming a display and a slot (the shape of 261 to 272).
    fn named(command: u32, display: &[u8; 8], slot: u32) -> Request {
        Request {
            command,
            data: [display.as_slice(), &slot.to_le_bytes()].concat(),
            ..Request::default()
        }
    }

    #[test]
    fn a_vi2_native_event_is_set_by_the_source_it_is_bound_to() {
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        let kernel = Kernel::new(ProcessInfo::new("vi2", ProcessKind::Nso, None));
        let mut v = Vi2Session::new(Arc::downgrade(&kernel), Arc::new(vi::Shared::new()));
        let display = b"default\0";
        assert!(
            v.request_tipc(&named(261, display, 0)).is_err(),
            "a slot is acquired before it is bound"
        );
        let mut acquire = req(31);
        acquire.data = 0u32.to_le_bytes().to_vec();
        let event = v.request_tipc(&acquire).unwrap().copy_handles.remove(0);
        assert!(
            !kernel.event_signaled(&event),
            "nothing sets an unbound slot"
        );
        // A change source (here the refresh rate) sets the slot once, when it is bound.
        v.request_tipc(&named(263, display, 0)).unwrap();
        assert!(kernel.event_signaled(&event));
        kernel.set_event(&event, false);
        // The vsync sets it at every vblank.
        v.request_tipc(&named(261, display, 0)).unwrap();
        let start = std::time::Instant::now();
        while !kernel.event_signaled(&event) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(2),
                "no vsync"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        // Unbound, it is set no more.
        v.request_tipc(&named(262, display, 0)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        kernel.set_event(&event, false);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!kernel.event_signaled(&event), "an unbound slot is quiet");
        // The duplicate of an acquired slot is the same event.
        let mut duplicate = req(32);
        duplicate.data = 0u32.to_le_bytes().to_vec();
        let again = v.request_tipc(&duplicate).unwrap();
        assert!(matches!(
            (&again.copy_handles[0], &event),
            (KObject::ReadableEvent(a), KObject::ReadableEvent(b)) if a == b
        ));
        // The queries answer in vi2's own values: 60 Hz, SDR, 1920x1080.
        let query = |v: &mut Vi2Session, command| v.request_tipc(&named(command, display, 0));
        assert_eq!(query(&mut v, 103).unwrap().data, 1u64.to_le_bytes());
        assert_eq!(query(&mut v, 104).unwrap().data[..8], 1u64.to_le_bytes());
        assert_eq!(query(&mut v, 106).unwrap().data, 2u64.to_le_bytes());
        // Another display, or a slot past the eight, stops the run.
        assert!(v.request_tipc(&named(103, b"other\0\0\0", 0)).is_err());
        acquire.data = 8u32.to_le_bytes().to_vec();
        assert!(v.request_tipc(&acquire).is_err());
        // 51 takes up to the seven recorders 5 says there are.
        let mut count = req(51);
        count.data = 7u32.to_le_bytes().to_vec();
        assert!(v.request_tipc(&count).is_ok());
        count.data = 8u32.to_le_bytes().to_vec();
        assert!(v.request_tipc(&count).is_err());
    }
}
