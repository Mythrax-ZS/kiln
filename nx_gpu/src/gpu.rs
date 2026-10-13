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
            next_space: 1,
            next_channel: 1,
        }
    }

    /// The title queued a buffer for display: tell the renderer where it is. The `nvmap` handle
    /// is resolved to the address of its memory in the guest process, which is how a render
    /// target is matched with the buffer that shows it.
    pub fn present(&mut self, frame: &crate::display::Frame) -> Result<(), String> {
        let Some(backend) = self.backend.as_deref_mut() else {
            return Ok(());
        };
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
        backend.present(&crate::render::Present {
            address: base + u64::from(buffer.offset()),
            width: buffer.width,
            height: buffer.height,
            pitch: buffer.pitch(),
            format: buffer.format,
            layout: buffer.layout(),
            block_height_log2: buffer.block_height_log2(),
            crop: frame.input.crop,
            frame: frame.number,
        })
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
        let mut ch = self
            .channels
            .remove(&channel)
            .ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        let result = self.run(&mut ch, channel, entries, flags, fence);
        self.channels.insert(channel, ch);
        result
    }

    fn run(
        &mut self,
        ch: &mut Channel,
        channel: u32,
        entries: &[u64],
        flags: u32,
        fence: (u32, u32),
    ) -> Result<Submitted, SubmitError> {
        use submit_flags::*;
        let space_id = ch.space.ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        let syncpt = ch.syncpt.ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        let unimplemented = |why: String| SubmitError::Unimplemented(why);
        if flags & SYNC_FENCE != 0 {
            return Err(unimplemented(
                "a submission whose fence is a sync file descriptor".into(),
            ));
        }
        if flags & FENCE_WAIT != 0 && !self.syncpoints.is_reached(fence.0, fence.1) {
            return Err(unimplemented(format!(
                "a submission that waits for syncpoint {} to reach {}, which no work will bring \
                 it to",
                fence.0, fence.1
            )));
        }
        let promised = self.syncpoints.max(syncpt).unwrap_or(0);
        let space = self
            .spaces
            .get(&space_id)
            .ok_or(SubmitError::Nv(NvError::INVALID_STATE))?;
        let mut woken = Vec::new();
        self.stats.submits += 1;
        if self.level >= 1 {
            eprintln!(
                "[gpu] channel {channel}: {} entries, flags {flags:#x}, fence ({}, {})",
                entries.len(),
                fence.0,
                fence.1
            );
        }
        for (i, raw) in entries.iter().enumerate() {
            let entry = GpEntry::decode(*raw);
            self.stats.entries += 1;
            if entry.words == 0 {
                continue;
            }
            let mut bytes = vec![0u8; entry.words as usize * 4];
            if !space.read(&*self.mem, entry.va, &mut bytes) {
                return Err(unimplemented(format!(
                    "pushbuffer entry {i} ({} words at GPU address {:#x}) is not mapped",
                    entry.words, entry.va
                )));
            }
            if self.level >= 1 {
                eprintln!(
                    "[gpu]   entry {i}: {} words at {:#x}{}",
                    entry.words,
                    entry.va,
                    if entry.sync_wait { ", sync" } else { "" }
                );
            }
            let mut cx = Cx {
                mem: &*self.mem,
                space,
                syncpoints: &mut self.syncpoints,
                woken: &mut woken,
                stats: &mut self.stats,
                backend: self.backend.as_deref_mut(),
                lenient: self.lenient,
                channel,
                syncpt,
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
        let mut out = fence;
        if flags & INCREMENT != 0 {
            let target = promised.wrapping_add(fence.1);
            let value = self.syncpoints.value(syncpt).unwrap_or(0);
            if !reached(value, target) {
                woken.extend(
                    self.syncpoints
                        .increment(syncpt, target.wrapping_sub(value)),
                );
            }
            out = (syncpt, target);
        }
        if flags & FENCE_GET != 0 {
            woken.extend(self.syncpoints.increment(syncpt, 1));
            out = (syncpt, self.syncpoints.value(syncpt).unwrap_or(0));
        }
        if self.level >= 1 && out != fence {
            eprintln!("[gpu]   fence out ({}, {})", out.0, out.1);
        }
        Ok(Submitted { fence: out, woken })
    }
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
