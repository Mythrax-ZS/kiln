//! The device as a whole: what the nvdrv nodes share.

use crate::mem::GuestMemory;
use crate::nvmap::Nvmap;
use crate::syncpt::Syncpoints;
use crate::vm::AddressSpace;
use std::collections::BTreeMap;
use std::sync::Arc;

/// The engine classes `ALLOC_OBJ_CTX` accepts (the ones the characteristics ioctl reports).
pub mod class {
    pub const TWOD: u32 = 0x902D;
    pub const THREED: u32 = 0xB197;
    pub const COMPUTE: u32 = 0xB1C0;
    pub const INLINE_TO_MEMORY: u32 = 0xA140;
    pub const DMA_COPY: u32 = 0xB0B5;
    pub const GPFIFO: u32 = 0xB06F;
}

/// A GPU channel: the command queue behind one `/dev/nvhost-gpu` descriptor.
#[derive(Debug, Default)]
pub struct Channel {
    /// The address space commands run in, once `BIND_CHANNEL` has been called.
    pub space: Option<u32>,
    /// The syncpoint the channel's fences count on, reserved by `ALLOC_GPFIFO_EX2`.
    pub syncpt: Option<u32>,
    pub gpfifo_entries: u32,
    pub nvmap_fd: Option<u32>,
    pub priority: u32,
    pub timeslice: u32,
    pub timeout: u32,
    pub user_data: u64,
    pub error_notifier: bool,
    /// The classes bound with `ALLOC_OBJ_CTX`, with the object id answered for each.
    pub objects: Vec<(u32, u64)>,
}

/// The state every node of `nvdrv` shares.
pub struct Gpu {
    pub mem: Arc<dyn GuestMemory>,
    pub nvmap: Nvmap,
    pub spaces: BTreeMap<u32, AddressSpace>,
    pub channels: BTreeMap<u32, Channel>,
    pub syncpoints: Syncpoints,
    next_space: u32,
    next_channel: u32,
}

impl Gpu {
    pub fn new(mem: Arc<dyn GuestMemory>) -> Gpu {
        Gpu {
            mem,
            nvmap: Nvmap::new(),
            spaces: BTreeMap::new(),
            channels: BTreeMap::new(),
            syncpoints: Syncpoints::new(),
            next_space: 1,
            next_channel: 1,
        }
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
