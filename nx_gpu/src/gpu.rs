//! The device as a whole: what the nvdrv nodes share, and the submission of commands.

use crate::engine::{Cx, Stats};
use crate::error::NvError;
use crate::event::EventToken;
use crate::mem::GuestMemory;
use crate::nvmap::Nvmap;
use crate::pushbuf::GpEntry;
use crate::render::Backend;
use crate::syncpt::{reached, Syncpoints};
use crate::vm::AddressSpace;
use std::collections::BTreeMap;
use std::sync::Arc;

pub use crate::channel::Channel;

/// The engine classes `ALLOC_OBJ_CTX` accepts (the ones the characteristics ioctl reports).
pub mod class {
    pub const TWOD: u32 = 0x902D;
    pub const THREED: u32 = 0xB197;
    pub const COMPUTE: u32 = 0xB1C0;
    pub const INLINE_TO_MEMORY: u32 = 0xA140;
    pub const DMA_COPY: u32 = 0xB0B5;
    pub const GPFIFO: u32 = 0xB06F;
}

/// `SUBMIT_GPFIFO` flags (Switchbrew, "NV services", section 9; bit 8 is what `libnx` sets).
pub mod submit_flags {
    /// Wait for the fence given before running the entries.
    pub const FENCE_WAIT: u32 = 1 << 0;
    /// Increment the channel's syncpoint once the entries are done and return the fence.
    pub const FENCE_GET: u32 = 1 << 1;
    /// The entries are in the GPU's own format (they always have the same bits here).
    pub const HW_FORMAT: u32 = 1 << 2;
    /// The fence is a sync file descriptor.
    pub const SYNC_FENCE: u32 = 1 << 3;
    /// `fence.value` is the number of syncpoint increments the entries perform; return the fence
    /// of the last.
    pub const INCREMENT: u32 = 1 << 8;
}

/// Why a submission did not complete.
#[derive(Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// The driver answers with this error.
    Nv(NvError),
    /// The commands ask for something the model does not implement; the run stops.
    Unimplemented(String),
}

/// What a completed submission leaves.
#[derive(Debug, PartialEq, Eq)]
pub struct Submitted {
    /// The fence to give back: (syncpoint id, value).
    pub fence: (u32, u32),
    /// The events the syncpoint increments have set.
    pub woken: Vec<EventToken>,
}

/// The state every node of `nvdrv` shares.
pub struct Gpu {
    pub mem: Arc<dyn GuestMemory>,
    pub nvmap: Nvmap,
    pub spaces: BTreeMap<u32, AddressSpace>,
    pub channels: BTreeMap<u32, Channel>,
    pub syncpoints: Syncpoints,
    pub stats: Stats,
    /// What draws, clears and displayed buffers are given to; without one nothing is drawn.
    pub backend: Option<Box<dyn Backend>>,
    /// A draw the backend fails is counted and the run goes on (`NX_GPU_LENIENT=1`), instead of
    /// stopping at the first.
    pub lenient: bool,
    /// The trace level from `NX_TRACE_GPU`: 0 none, 1 submissions, 2 every method.
    pub level: u8,
    /// Where submissions go instead of being run by the call that makes them, if anywhere
    /// (A-124).
    pub queue: Option<Arc<dyn Queue>>,
    next_space: u32,
    next_channel: u32,
}

/// The trace level `NX_TRACE_GPU` asks for: a number, or any other value for 1.
pub fn trace_level() -> u8 {
    match std::env::var("NX_TRACE_GPU") {
        Ok(v) => v.parse().unwrap_or(1),
        Err(_) => 0,
    }
}

impl Gpu {
    pub fn new(mem: Arc<dyn GuestMemory>) -> Gpu {
        Gpu {
            mem,
            nvmap: Nvmap::new(),
            spaces: BTreeMap::new(),
            channels: BTreeMap::new(),
            syncpoints: Syncpoints::new(),
            stats: Stats::default(),
            backend: None,
            lenient: std::env::var("NX_GPU_LENIENT").is_ok_and(|v| v != "0"),
            level: trace_level(),
            queue: None,
            next_space: 1,
            next_channel: 1,
        }
    }

    /// The title queued a buffer for display: tell the renderer where it is. The `nvmap` handle
    /// is resolved to the address of its memory in the guest process, which is how a render
    /// target is matched with the buffer that shows it.
    pub fn present(&mut self, frame: &crate::display::Frame) -> Result<(), String> {
        let Some(request) = self.present_request(frame)? else {
            return Ok(());
        };
        match self.backend.as_deref_mut() {
            Some(backend) => backend.present(&request),
            None => Ok(()),
        }
    }

    /// What showing a queued buffer asks of the renderer; nothing without one.
    pub fn present_request(
        &self,
        frame: &crate::display::Frame,
    ) -> Result<Option<crate::render::Present>, String> {
        if self.backend.is_none() {
            return Ok(None);
        }
        let buffer = &frame.buffer;
        let Some(base) = self.nvmap.get(buffer.nvmap_id).and_then(|h| h.addr) else {
            return Err(format!(
                "the displayed buffer is in nvmap handle {}, which has no memory",
                buffer.nvmap_id
            ));
        };
        if self.level >= 1 && frame.number <= 2 {
            eprintln!(
                "[gpu] frame {} shows nvmap handle {} (memory at {base:#x}), offset {:#x}, \
                 layout {}, pitch {}, block height {}; native handle words {:x?}",
                frame.number,
                buffer.nvmap_id,
                buffer.offset(),
                buffer.layout(),
                buffer.pitch(),
                buffer.block_height_log2(),
                buffer.ints
            );
        }
        Ok(Some(crate::render::Present {
            address: base + u64::from(buffer.offset()),
            width: buffer.width,
            height: buffer.height,
            pitch: buffer.pitch(),
            format: buffer.format,
            layout: buffer.layout(),
            block_height_log2: buffer.block_height_log2(),
            crop: frame.input.crop,
            frame: frame.number,
        }))
    }

    pub fn new_space(&mut self) -> u32 {
        let id = self.next_space;
        self.next_space += 1;
        self.spaces.insert(id, AddressSpace::new());
        id
    }

    pub fn new_channel(&mut self) -> u32 {
        let id = self.next_channel;
        self.next_channel += 1;
        self.channels.insert(id, Channel::default());
        id
    }

    /// Run GPFIFO entries on a channel to completion, then do what the flags say about fences.
    pub fn submit(
        &mut self,
        channel: u32,
        entries: &[u64],
        flags: u32,
        fence: (u32, u32),
    ) -> Result<Submitted, SubmitError> {
        let mut job = self.prepare(channel, entries, flags, fence)?;
        loop {
            if let Some(done) = self.step(&mut job)? {
                return Ok(done);
            }
        }
    }

    /// Accept a submission without running it: check what can be checked now, read the commands
    /// out of guest memory and promise the syncpoint increments the flags speak of. The job is
    /// run by [`Gpu::step`], now (what [`Gpu::submit`] does) or later and elsewhere.
    pub fn prepare(
        &mut self,
        channel: u32,
        entries: &[u64],
        flags: u32,
        fence: (u32, u32),
    ) -> Result<Job, SubmitError> {
        use submit_flags::*;
        let ch = self
            .channels
            .get(&channel)
            .ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        let space_id = ch.space.ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        let syncpt = ch.syncpt.ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        if flags & SYNC_FENCE != 0 {
            return Err(SubmitError::Unimplemented(
                "a submission whose fence is a sync file descriptor".into(),
            ));
        }
        let space = self
            .spaces
            .get(&space_id)
            .ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        // The commands are read now, as the console's driver takes the entries now: what the
        // program does with that memory after the call does not change what runs.
        let mut parts = Vec::with_capacity(entries.len());
        for (i, raw) in entries.iter().enumerate() {
            let entry = GpEntry::decode(*raw);
            let mut bytes = vec![0u8; entry.words as usize * 4];
            if entry.words != 0 && !space.read(&*self.mem, entry.va, &mut bytes) {
                return Err(SubmitError::Unimplemented(format!(
                    "pushbuffer entry {i} ({} words at GPU address {:#x}) is not mapped",
                    entry.words, entry.va
                )));
            }
            parts.push((entry, bytes));
        }
        // The increments the program says the entries perform are promised now, so that the fence
        // of the last can be handed back before they have happened.
        let target = if flags & INCREMENT != 0 {
            self.syncpoints.promise(syncpt, fence.1).unwrap_or(0)
        } else {
            0
        };
        Ok(Job {
            channel,
            space: space_id,
            syncpt,
            flags,
            fence,
            target,
            entries: parts,
            next: 0,
            started: false,
            woken: Vec::new(),
        })
    }

    /// Run the next entry of a job, or finish it when none is left: the renderer is told the
    /// submission ended, then the fences are completed. Returns what the submission leaves once it
    /// is over.
    pub fn step(&mut self, job: &mut Job) -> Result<Option<Submitted>, SubmitError> {
        use submit_flags::*;
        let unimplemented = |why: String| SubmitError::Unimplemented(why);
        if !job.started {
            job.started = true;
            if job.flags & FENCE_WAIT != 0 && !self.syncpoints.is_reached(job.fence.0, job.fence.1)
            {
                return Err(unimplemented(format!(
                    "a submission that waits for syncpoint {} to reach {}, which no work will \
                     bring it to",
                    job.fence.0, job.fence.1
                )));
            }
            self.stats.submits += 1;
            if self.level >= 1 {
                eprintln!(
                    "[gpu] channel {}: {} entries, flags {:#x}, fence ({}, {})",
                    job.channel,
                    job.entries.len(),
                    job.flags,
                    job.fence.0,
                    job.fence.1
                );
            }
        }
        if job.next < job.entries.len() {
            let i = job.next;
            job.next += 1;
            self.stats.entries += 1;
            let (entry, bytes) = &job.entries[i];
            if entry.words == 0 {
                return Ok(None);
            }
            if self.level >= 1 {
                eprintln!(
                    "[gpu]   entry {i}: {} words at {:#x}{}",
                    entry.words,
                    entry.va,
                    if entry.sync_wait { ", sync" } else { "" }
                );
            }
            // The channel's state is out of the table while its commands run, which nobody can
            // see: the caller holds the whole device.
            let mut ch = self
                .channels
                .remove(&job.channel)
                .ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
            let result = (|| {
                let space = self
                    .spaces
                    .get(&job.space)
                    .ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
                let mut cx = Cx {
                    mem: &*self.mem,
                    space,
                    syncpoints: &mut self.syncpoints,
                    woken: &mut job.woken,
                    stats: &mut self.stats,
                    backend: self.backend.as_deref_mut(),
                    lenient: self.lenient,
                    channel: job.channel,
                    syncpt: job.syncpt,
                    level: self.level,
                };
                let _phase = crate::progress::enter(crate::progress::METHODS_RUNNING);
                for chunk in bytes.as_chunks::<4>().0 {
                    let word = u32::from_le_bytes(*chunk);
                    cx.stats.words += 1;
                    match ch.pusher.feed(word) {
                        Ok(Some(w)) => ch.execute(&mut cx, w).map_err(unimplemented)?,
                        Ok(None) => {}
                        Err(e) => return Err(unimplemented(e.0)),
                    }
                }
                Ok(())
            })();
            self.channels.insert(job.channel, ch);
            result?;
            return Ok(None);
        }
        // What was recorded for the renderer is finished before any fence says it is.
        if let Some(backend) = self.backend.as_deref_mut() {
            if let Err(why) = backend.finish() {
                if self.lenient {
                    eprintln!("[gpu] the renderer could not finish a submission: {why}");
                } else {
                    return Err(unimplemented(format!(
                        "the renderer could not finish a submission: {why}"
                    )));
                }
            }
        }
        // Fences. The increments the commands performed are in the counter already; the ones
        // the program promised and the one it asked for are completed here.
        let (syncpt, fence) = (job.syncpt, job.fence);
        let mut woken = std::mem::take(&mut job.woken);
        let mut out = fence;
        if job.flags & INCREMENT != 0 {
            let value = self.syncpoints.value(syncpt).unwrap_or(0);
            if !reached(value, job.target) {
                woken.extend(
                    self.syncpoints
                        .increment(syncpt, job.target.wrapping_sub(value)),
                );
            }
            out = (syncpt, job.target);
        }
        if job.flags & FENCE_GET != 0 {
            woken.extend(self.syncpoints.increment(syncpt, 1));
            out = (syncpt, self.syncpoints.value(syncpt).unwrap_or(0));
        }
        if self.level >= 1 && out != fence {
            eprintln!("[gpu]   fence out ({}, {})", out.0, out.1);
        }
        Ok(Some(Submitted { fence: out, woken }))
    }

    /// The fence a deferred submission gives back at once.
    fn promised_fence(job: &Job) -> (u32, u32) {
        if job.flags & submit_flags::INCREMENT != 0 {
            (job.syncpt, job.target)
        } else {
            job.fence
        }
    }

    /// Can a submission with these flags be handed to the queue? Its fence has to be known before
    /// it has run: a fence that is made by running it (`FENCE_GET`) cannot be.
    pub fn defers(&self, flags: u32) -> bool {
        self.queue.is_some() && flags & submit_flags::FENCE_GET == 0
    }

    /// Hand a submission to the queue, to be run later in the order of arrival, and give its
    /// fence back at once.
    pub fn defer(
        &mut self,
        channel: u32,
        entries: &[u64],
        flags: u32,
        fence: (u32, u32),
    ) -> Result<Submitted, SubmitError> {
        let job = self.prepare(channel, entries, flags, fence)?;
        let out = Gpu::promised_fence(&job);
        match &self.queue {
            Some(queue) => queue.push(Task::Submit(job)),
            None => {
                return Err(SubmitError::Unimplemented(
                    "a submission was deferred with no queue to run it".into(),
                ))
            }
        }
        Ok(Submitted {
            fence: out,
            woken: Vec::new(),
        })
    }

    /// Run a step of a task the queue was given: `None` while it has more to do.
    pub fn run_task(&mut self, task: &mut Task) -> Result<Option<Submitted>, SubmitError> {
        match task {
            Task::Submit(job) => self.step(job),
            Task::Present(request) => {
                if let Some(backend) = self.backend.as_deref_mut() {
                    backend
                        .present(request)
                        .map_err(SubmitError::Unimplemented)?;
                }
                Ok(Some(Submitted {
                    fence: (0, 0),
                    woken: Vec::new(),
                }))
            }
        }
    }
}

/// A submission accepted and not yet run: [`Gpu::prepare`] makes it and [`Gpu::step`] runs it.
pub struct Job {
    channel: u32,
    space: u32,
    syncpt: u32,
    flags: u32,
    /// The fence the program gave: waited for first (`FENCE_WAIT`) and otherwise handed back.
    fence: (u32, u32),
    /// The value of the syncpoint that `INCREMENT`'s fence stands for.
    target: u32,
    /// The entries and the commands each points at, read when the submission was accepted.
    entries: Vec<(GpEntry, Vec<u8>)>,
    next: usize,
    started: bool,
    /// The events the syncpoint increments of the commands have set so far.
    woken: Vec<EventToken>,
}

impl Job {
    /// The channel the commands run on.
    pub fn channel(&self) -> u32 {
        self.channel
    }
}

/// Work for [`Queue`].
pub enum Task {
    /// A submission of commands.
    Submit(Job),
    /// A buffer to show, after everything before it.
    Present(crate::render::Present),
}

/// Somewhere to put work that another thread runs later, one task after the other in the order
/// they were put (docs/ASSUMPTIONS.md A-124). `push` never waits.
pub trait Queue: Send + Sync {
    fn push(&self, task: Task);
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu")
            .field("handles", &self.nvmap.len())
            .field("spaces", &self.spaces.len())
            .field("channels", &self.channels.len())
            .finish()
    }
}
