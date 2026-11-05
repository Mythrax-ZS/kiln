//! The Switch 2's graphics supervisor (`nx2:gsv` in the NVN2 guide): the session a Switch 2
//! program's SDK asks the applet for (`ISelfController` command 2000, from
//! `nn::gsv::BootWithRemoteMemory`) and through which it reaches its GPU. It speaks TIPC
//! (docs/ASSUMPTIONS.md A-194).
//!
//! Only the commands a title was seen to send are answered; any other stops the run with its
//! number, as every service here does.

use super::super::kernel::Kernel;
use super::super::layout;
use super::super::object::KObject;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use super::vdec::{self, DecoderLayout};
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
    /// The native memory and images of the graphics system, which `vi2` presents (A-212).
    native: Arc<std::sync::Mutex<NativeObjects>>,
}

/// The layout of a native image, from the 16 words of its preference (A-212): the kind at word 1,
/// the width and height less one at word 2, the block height (log2 of its rows) at word 4, the row
/// pitch in 64-byte units at word 5, the format at word 6. The image is block-linear: its size is
/// the pitch times its height rounded up to a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageLayout {
    pub kind: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub pitch: u64,
    pub block_rows: u64,
    pub size: u64,
    pub alignment: u64,
}

impl ImageLayout {
    pub fn from_preference(words: &[u32]) -> Result<ImageLayout, String> {
        let word = |n: usize| words.get(n).copied().unwrap_or(0);
        let (width, height) = ((word(2) & 0xFFFF) + 1, (word(2) >> 16) + 1);
        let (block_log2, units) = (word(4), u64::from(word(5)));
        // The pitch the SDK gives holds the width at four bytes a pixel or more.
        let least = (u64::from(width) * 4).div_ceil(64);
        if block_log2 > 8 || !(least..=4096).contains(&units) {
            return Err(format!(
                "a native image of {width}x{height} with a block of 2^{block_log2} rows and a \
                 pitch of {units} units of 64 bytes (preference {:x?})",
                &words[..words.len().min(8)]
            ));
        }
        let (pitch, block_rows) = (units * 64, 1u64 << block_log2);
        Ok(ImageLayout {
            kind: word(1),
            width,
            height,
            format: word(6),
            pitch,
            block_rows,
            size: pitch * u64::from(height).next_multiple_of(block_rows),
            alignment: (64 * block_rows).max(0x1000),
        })
    }
}

impl ImageLayout {
    /// A GameChat image (1291, A-220): `nn::chat::gsv::MakeImagePreference` gives only the size
    /// and leaves the block height, the pitch and the format to the graphics server (zero); taken
    /// as four bytes a pixel in blocks of 128 rows. A preference that gives them is read as any.
    pub fn from_chat_preference(words: &[u32]) -> Result<ImageLayout, String> {
        let word = |n: usize| words.get(n).copied().unwrap_or(0);
        if word(1) == 1 && word(4) == 0 && word(5) == 0 && word(6) == 0 {
            let mut w = words.to_vec();
            w.resize(16, 0);
            w[4] = 7;
            w[5] = ((word(2) & 0xFFFF) + 1).saturating_mul(4).div_ceil(64);
            return Self::from_preference(&w);
        }
        Self::from_preference(words)
    }
}

/// Native memory: its size and, once bound (114, 423), its address.
#[derive(Clone, Copy, Debug, Default)]
struct NativeMemory {
    size: u64,
    address: Option<u64>,
}

/// A pool of native images made with 429: `count` images of one layout, one after the other in
/// a native memory.
#[derive(Clone, Copy, Debug)]
struct NativePool {
    memory: u32,
    count: u32,
    layout: ImageLayout,
}

/// A pool of decoder output images made with 911 (A-262): `count` images of one layout, one
/// after the other in a native memory.
#[derive(Clone, Copy, Debug)]
struct DecoderPool {
    memory: u32,
    count: u32,
    layout: DecoderLayout,
}

/// The native memory and image pools of the graphics system (A-212), and the pools of the video
/// decoder's output images (A-262), whose handles are taken from the same numbers.
#[derive(Debug, Default)]
pub struct NativeObjects {
    memories: std::collections::BTreeMap<u32, NativeMemory>,
    pools: std::collections::BTreeMap<u32, NativePool>,
    decoder_pools: std::collections::BTreeMap<u32, DecoderPool>,
    next_memory: u32,
    next_image: u32,
}

impl NativeObjects {
    /// The offset of the decoder output image `handle` in its pool's memory, that memory's
    /// address once bound, and the image's layout; None once the memory is gone.
    fn decoder_image(&self, handle: u32) -> Option<(u64, Option<u64>, DecoderLayout)> {
        let (&first, pool) = self.decoder_pools.range(..=handle).next_back()?;
        if handle - first >= pool.count {
            return None;
        }
        let memory = self.memories.get(&pool.memory)?;
        Some((
            u64::from(handle - first) * pool.layout.size,
            memory.address,
            pool.layout,
        ))
    }

    /// The address in the process of the decoder output image `handle` and its layout; None
    /// unless its memory is bound.
    pub fn decoder_image_address(&self, handle: u32) -> Option<(u64, DecoderLayout)> {
        let (offset, address, layout) = self.decoder_image(handle)?;
        Some((address? + offset, layout))
    }

    /// Where the image `handle` (the first image of its pool, or one after it) lies in its pool's
    /// native memory, bound or not, and its layout; None once that memory is gone (112).
    pub fn image_in_memory(&self, handle: u32) -> Option<(u64, ImageLayout)> {
        let (&first, pool) = self.pools.range(..=handle).next_back()?;
        if handle - first >= pool.count || !self.memories.contains_key(&pool.memory) {
            return None;
        }
        Some((u64::from(handle - first) * pool.layout.size, pool.layout))
    }

    /// The address in the process of the image `handle` and its layout; None unless its native
    /// memory is bound.
    pub fn image_address(&self, handle: u32) -> Option<(u64, ImageLayout)> {
        let (offset, layout) = self.image_in_memory(handle)?;
        let (_, pool) = self.pools.range(..=handle).next_back()?;
        let base = self.memories.get(&pool.memory)?.address?;
        Some((base + offset, layout))
    }
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
            native: Arc::default(),
        }
    }

    /// The native memory and image commands (A-212); `None` for another command.
    fn native_command(&mut self, req: &Request) -> Option<ServiceResult> {
        let mut st = self.native.lock().unwrap_or_else(|e| e.into_inner());
        let preference = || ImageLayout::from_preference(&words_of(req, 16));
        let count = req.u32_at(64).max(1);
        Some(match req.command {
            // CalculateRequirements(preference, u32) -> (u64 size, u64 alignment), from
            // nn::gsv::shim::CalculateRequirements.
            1510 => preference()
                .map(|l| Response::ok().with_u64(l.size).with_u64(l.alignment))
                .map_err(|e| Stop(format!("nx2:gsv 1510: {e}"))),
            // nn::chat::gsv::CalculateRequiredBufferMemory(preference, u32 count) -> (u64 size,
            // u64 alignment): the memory of a pool of `count` GameChat images (A-220).
            1291 => ImageLayout::from_chat_preference(&words_of(req, 16))
                .map(|l| {
                    Response::ok()
                        .with_u64(l.size * u64::from(count))
                        .with_u64(l.alignment)
                })
                .map_err(|e| Stop(format!("nx2:gsv 1291: {e}"))),
            // nn::gpu::sdk::QueryNativeImage(u32 image) -> (8 bytes the SDK does not read, u64
            // offset of the image in its native memory, a GpuOunceUserImageDescriptor of 32
            // words, u32 flags), which NVN reads to use an image another part of the system made
            // (GameChat's camera images, 1211; A-221).
            428 => match st.image_in_memory(req.u32_at(0)) {
                Some((offset, layout)) => Ok(native_image_reply(0, offset, &layout)),
                None => Err(Stop(format!(
                    "nx2:gsv 428 of the native image {}, which 429 or 1211 did not make",
                    req.u32_at(0)
                ))),
            },
            // nn::gpu::sdk::MapNativeImage(u32 image, u32 1, u32 address space, u8 permission)
            // -> 428's reply with the image's GPU address in its first two words: the image is
            // mapped at its own CPU address, as GPU-accessible memory is (A-201, A-221).
            426 => {
                let (handle, space) = (req.u32_at(0), req.u32_at(8));
                let known = self
                    .gpu_state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .spaces
                    .contains(&space);
                if !known || req.u32_at(4) != 1 {
                    return Some(Err(Stop(format!(
                        "nx2:gsv 426 of the image {handle} into the address space {space} (second \
                         word {})",
                        req.u32_at(4)
                    ))));
                }
                let found = st.image_in_memory(handle).and_then(|(offset, layout)| {
                    let (&first, pool) = st.pools.range(..=handle).next_back()?;
                    debug_assert!(handle >= first);
                    let base = st.memories.get(&pool.memory)?.address?;
                    Some((base + offset, offset, layout))
                });
                let Some((address, offset, layout)) = found else {
                    return Some(Err(Stop(format!(
                        "nx2:gsv 426 of the native image {handle}, which is not made or whose \
                         memory is not bound"
                    ))));
                };
                if let Err(e) = self.gpu.switch_2_map(address, layout.size, address, 0) {
                    return Some(Err(Stop(format!("nx2:gsv 426 of the image {handle}: {e}"))));
                }
                Ok(native_image_reply(address, offset, &layout))
            }
            // CalculateNativeImageSize(preference, u32 count) -> u64 size
            435 => preference()
                .map(|l| Response::ok().with_u64(l.size * u64::from(count)))
                .map_err(|e| Stop(format!("nx2:gsv 435: {e}"))),
            // CreateNativeMemory(u8 mode, u64 size) -> u32 id (from 1)
            111 => {
                st.next_memory += 1;
                let id = st.next_memory;
                st.memories.insert(
                    id,
                    NativeMemory {
                        size: req.u64_at(4),
                        address: None,
                    },
                );
                Ok(Response::ok().with_u32(id))
            }
            // DestroyNativeMemory(u32 id)
            112 => match st.memories.remove(&req.u32_at(0)) {
                Some(_) => Ok(Response::ok()),
                None => Err(Stop(format!(
                    "nx2:gsv 112 of the native memory {}, which 111 did not make",
                    req.u32_at(0)
                ))),
            },
            // BindNativeMemory(u32 id, u64 address, u64 size, u64 flags, u64 offset) and
            // BindNativeGpuMemory(u32 id, u64 address, u64 size, u32 GPU memory, u64 offset): the
            // memory is at the address plus the offset.
            114 | 423 => {
                let id = req.u32_at(0);
                let offset = if req.command == 114 {
                    req.u64_at(28)
                } else {
                    req.u64_at(24)
                };
                let Some(memory) = st.memories.get_mut(&id) else {
                    return Some(Err(Stop(format!(
                        "nx2:gsv {} of the native memory {id}, which 111 did not make",
                        req.command
                    ))));
                };
                memory.address = Some(req.u64_at(4) + offset);
                if memory.size == 0 {
                    memory.size = req.u64_at(12);
                }
                Ok(Response::ok())
            }
            // CreateNativeImage(preference, u32 count, u32 memory) -> u32 first image: the pool
            // takes as many handles as it has images. 1211 is nn::chat::gsv::CreateImagePool, the
            // same for GameChat's images (A-220).
            429 | 1211 => {
                let memory = req.u32_at(68);
                if !st.memories.contains_key(&memory) {
                    return Some(Err(Stop(format!(
                        "nx2:gsv {} in the native memory {memory}, which 111 did not make",
                        req.command
                    ))));
                }
                let layout = if req.command == 1211 {
                    ImageLayout::from_chat_preference(&words_of(req, 16))
                } else {
                    preference()
                };
                match layout {
                    Ok(layout) => {
                        let first = st.next_image + 1;
                        st.next_image += count;
                        st.pools.insert(
                            first,
                            NativePool {
                                memory,
                                count,
                                layout,
                            },
                        );
                        Ok(Response::ok().with_u32(first))
                    }
                    Err(e) => Err(Stop(format!("nx2:gsv {}: {e}", req.command))),
                }
            }
            // CalculateDecoderImagePoolBuffer(preference, u32 count) -> (u64 size, u64
            // alignment), from nn::codec::CalculateDecoderImagePoolBuffer (A-262).
            991 => DecoderLayout::from_preference(&words_of(req, 16))
                .map(|l| {
                    Response::ok()
                        .with_u64(l.size * u64::from(count))
                        .with_u64(vdec::DECODER_IMAGE_ALIGNMENT)
                })
                .map_err(|e| Stop(format!("nx2:gsv 991: {e}"))),
            // CreateDecoderImage(preference, u32 count, u32 memory) -> u32 first image, from
            // nn::codec::InitializeDecoderImagePool: the images lie one after the other in the
            // memory and take as many handles (A-262).
            911 => {
                let memory = req.u32_at(68);
                let Some(size) = st.memories.get(&memory).map(|m| m.size) else {
                    return Some(Err(Stop(format!(
                        "nx2:gsv 911 in the native memory {memory}, which 111 did not make"
                    ))));
                };
                match DecoderLayout::from_preference(&words_of(req, 16)) {
                    Ok(layout) if size != 0 && layout.size * u64::from(count) > size => {
                        Err(Stop(format!(
                            "nx2:gsv 911: {count} decoder images of {:#x} bytes do not fit in \
                             the native memory {memory} of {size:#x}",
                            layout.size
                        )))
                    }
                    Ok(layout) => {
                        let first = st.next_image + 1;
                        st.next_image += count;
                        st.decoder_pools.insert(
                            first,
                            DecoderPool {
                                memory,
                                count,
                                layout,
                            },
                        );
                        Ok(Response::ok().with_u32(first))
                    }
                    Err(e) => Err(Stop(format!("nx2:gsv 911: {e}"))),
                }
            }
            // DestroyDecoderImage(u32 first image), from nn::codec::FinalizeDecoderImagePool.
            912 => match st.decoder_pools.remove(&req.u32_at(0)) {
                Some(_) => Ok(Response::ok()),
                None => Err(Stop(format!(
                    "nx2:gsv 912 of the decoder image {}, which 911 did not make",
                    req.u32_at(0)
                ))),
            },
            // GetCpuUserImageDescriptor(u32 image) -> 128 bytes, which
            // nn::codec::ReadDecoderOutputImage* copy a decoder output image out with (A-263).
            961 => match st.decoder_image(req.u32_at(0)) {
                Some((offset, _, layout)) => {
                    Ok(Response::ok().with_bytes(&layout.cpu_descriptor(offset)))
                }
                None => Err(Stop(format!(
                    "nx2:gsv 961 of the image {}, which is not a decoder image 911 made",
                    req.u32_at(0)
                ))),
            },
            // DestroyNativeImage(u32 first image)
            430 => match st.pools.remove(&req.u32_at(0)) {
                Some(_) => Ok(Response::ok()),
                None => Err(Stop(format!(
                    "nx2:gsv 430 of the native image {}, which 429 did not make",
                    req.u32_at(0)
                ))),
            },
            // (u32 image, u32 count) / (u32 image): a pool becomes or stops being a window's.
            314 | 315 => {
                if !st.pools.contains_key(&req.u32_at(0)) {
                    return Some(Err(Stop(format!(
                        "nx2:gsv {} of the native image {}, which 429 did not make",
                        req.command,
                        req.u32_at(0)
                    ))));
                }
                Ok(Response::ok())
            }
            _ => return None,
        })
    }
}

/// GameChat's graphics session (`nn::chat::gsv`, supervisor command 1202, A-219): event ids and
/// their events, which nothing sets, since GameChat is never in a call here.
#[derive(Debug)]
struct ChatGsv {
    kernel: Weak<Kernel>,
    /// The (writable, readable) ends of each event, by the id 1 handed out.
    events: Vec<(KObject, KObject)>,
    /// What 103 and 121 opened, by the id the program gave it (one space of ids, which the
    /// program allocates itself): a description.
    opened: std::collections::BTreeMap<u32, String>,
}

/// The highest event id the SDK accepts from command 1.
const CHAT_EVENT_IDS: usize = 0x400;

impl ChatGsv {
    /// Record what the program opened under id, unless something is open under it already.
    fn open(&mut self, req: &Request, id: u32, what: String) -> ServiceResult {
        if let Some(open) = self.opened.get(&id) {
            return Err(Stop(format!(
                "nx2:chat:gsv {} opens {what} as {id}, which {open} is already",
                req.command
            )));
        }
        self.opened.insert(id, what);
        Ok(Response::ok())
    }
}

impl Service for ChatGsv {
    fn name(&self) -> &str {
        "nx2:chat:gsv"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop(format!(
            "nx2:chat:gsv was sent a CMIF message (command {}); it speaks TIPC",
            req.command
        )))
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // () -> u32 id: a new event, from 0 (the SDK stops above 0x3FF).
            1 => {
                if self.events.len() >= CHAT_EVENT_IDS {
                    return Err(Stop(
                        "nx2:chat:gsv 1: all 0x400 event ids are handed out".into(),
                    ));
                }
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".into()))?;
                self.events.push(kernel.new_event());
                Ok(Response::ok().with_u32(self.events.len() as u32 - 1))
            }
            // (u32 id) -> event: the event of an id 1 handed out, which the SDK attaches as a
            // system event that clears itself.
            11 => {
                let id = req.u32_at(0) as usize;
                match self.events.get(id) {
                    Some(event) => Ok(Response::ok().with_copy_handle(event.1.clone())),
                    None => Err(Stop(format!(
                        "nx2:chat:gsv 11 of the event {id}, which 1 did not hand out"
                    ))),
                }
            }
            // (8 bytes of name, u32 id), from nn::chat::gsv::OpenVirtualDisplay: a named virtual
            // display ("chat2", ...) under an id the program chose. Nothing is ever shown on it.
            103 => {
                let raw = &req.data[..req.data.len().min(8)];
                let name: String = raw
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as char)
                    .collect();
                let id = req.u32_at(8);
                if name.is_empty() || !name.bytes().all(|b| b.is_ascii_graphic()) {
                    return Err(Stop(format!("nx2:chat:gsv 103 with the name {raw:x?}")));
                }
                self.open(req, id, format!("the virtual display `{name}`"))
            }
            // (u64 address, u64 count, u32 id), from nn::chat::gsv::CreateMetadataPool: a pool of
            // that many metadata entries in the program's memory. Nothing is ever written to it.
            121 => {
                let (address, count) = (req.u64_at(0), req.u64_at(8));
                if address == 0 || count == 0 {
                    return Err(Stop(format!(
                        "nx2:chat:gsv 121 of {count} entries at {address:#x}"
                    )));
                }
                let id = req.u32_at(16);
                self.open(
                    req,
                    id,
                    format!("the metadata pool of {count} at {address:#x}"),
                )
            }
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

/// What the camera's session answers when the camera asked for is not attached: 2452-0006, the
/// result the Switch 2 title takes to mean "no camera" (A-222).
const NO_CAMERA: ResultCode = ResultCode::new(452, 6);

/// The camera's session (supervisor command 602, A-222), which speaks CMIF. No camera is attached.
#[derive(Debug)]
struct Camera {
    kernel: Weak<Kernel>,
    /// The event 21 hands out, made on first use; nothing sets it, since no camera comes.
    event: Option<(KObject, KObject)>,
}

impl Service for Camera {
    fn name(&self) -> &str {
        "nx2:camera"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // (16 bytes of camera name: "Default"): the camera of that name is not attached.
            3 => {
                let name = &req.data[..req.data.len().min(16)];
                if !name.first().is_some_and(|b| b.is_ascii_graphic()) {
                    return Err(Stop(format!("nx2:camera 3 with the name {name:x?}")));
                }
                Ok(Response::error(NO_CAMERA))
            }
            // (16 bytes) -> event, which the title attaches as a system event: a camera being
            // attached, which never happens here.
            21 => {
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".into()))?;
                let event = self.event.get_or_insert_with(|| kernel.new_event());
                Ok(Response::ok().with_copy_handle(event.1.clone()))
            }
            // Polled every frame, with a u32 (0x180): answered with success and nothing, as
            // Eglantine answers it (A-234).
            29 => Ok(Response::ok()),
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }

    fn request_tipc(&mut self, req: &Request) -> ServiceResult {
        Err(Stop(format!(
            "nx2:camera was sent a TIPC message (command {}); it speaks CMIF",
            req.command
        )))
    }
}

/// The reply of 428 and 426 (A-221): the image's GPU address (426) or 0, its offset in its native
/// memory, a `GpuOunceUserImageDescriptor` of 32 words and the flags.
fn native_image_reply(gpu_address: u64, offset: u64, layout: &ImageLayout) -> Response {
    let mut descriptor = [0u32; 32];
    descriptor[0] = 1; // checked by the SDK
    descriptor[1] = layout.kind;
    descriptor[2] = (layout.width - 1) | (layout.height - 1) << 16;
    descriptor[4] = layout.block_rows.trailing_zeros();
    descriptor[5] = (layout.pitch / 64) as u32;
    descriptor[8] = 1; // checked by the SDK
    descriptor[9] = 1;
    descriptor[10] = 0; // the image starts at the address and offset before
    descriptor[11] = layout.size as u32; // checked: not zero
    let mut r = Response::ok().with_u64(gpu_address).with_u64(offset);
    for w in descriptor {
        r = r.with_u32(w);
    }
    r.with_u32(0)
}

/// The first `n` u32s of a request's data.
fn words_of(req: &Request, n: usize) -> Vec<u32> {
    (0..n).map(|i| req.u32_at(4 * i)).collect()
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
    /// The GPU-only regions made with 214, by first GPU address: the CPU address of their
    /// backing and their size (A-215).
    regions: std::collections::BTreeMap<u64, (u64, u64)>,
    /// Backing that 215 gave back, kept mapped for a later region of the same size: the guest's
    /// memory does not give unmapped pages back to the host.
    spare: Vec<(u64, u64)>,
    /// The end of the backing handed out so far, from [`layout::GPU_SERVER_BASE`].
    server_end: u64,
}

impl GpuState {
    /// Backing for a GPU-only region of `size` bytes, zeroed: spare backing of that size, or fresh
    /// pages past what was handed out.
    fn back(&mut self, kernel: &Kernel, size: u64) -> Result<u64, String> {
        if let Some(i) = self.spare.iter().position(|&(_, s)| s == size) {
            let (address, _) = self.spare.swap_remove(i);
            let zeros = vec![0u8; 1 << 20];
            let mut at = 0;
            while at < size {
                let n = (size - at).min(zeros.len() as u64);
                crate::rt::runtime::rt()
                    .mem
                    .write_bytes(address + at, &zeros[..n as usize], false)
                    .map_err(|e| format!("zeroing the backing at {:#x}: {e:?}", address + at))?;
                at += n;
            }
            return Ok(address);
        }
        let address = layout::GPU_SERVER_BASE + self.server_end;
        if self.server_end + size > layout::GPU_SERVER_SIZE {
            return Err(format!(
                "{size:#x} bytes of GPU-only memory do not fit: {:#x} bytes are handed out",
                self.server_end
            ));
        }
        kernel
            .map_server_memory(address, size)
            .map_err(|e| format!("mapping {size:#x} bytes at {address:#x}: {e:?}"))?;
        self.server_end += size;
        Ok(address)
    }
}

/// The flag of 214 that makes a region sparse: the program maps its own memory into parts of it
/// (216), and the rest reads zero.
const SPARSE_REGION: u32 = 0x200;

/// The size of a request of 216.
const SPARSE_MAP_REQUEST: usize = 40;

/// A request of 216 (`nn::gpu::sdk::SparseMapRequest`, A-216): a GPU range, the kind wanted, and
/// the GPU memory (101's handle, 0 for none) and the offset into it that the range shows.
#[derive(Clone, Copy, Debug)]
struct SparseMap {
    gpu_address: u64,
    size: u64,
    kind: u8,
    memory: u32,
    offset: u64,
}

impl SparseMap {
    fn parse(b: &[u8; SPARSE_MAP_REQUEST]) -> SparseMap {
        let u64_at = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        SparseMap {
            gpu_address: u64_at(0),
            size: u64_at(8),
            kind: b[19],
            memory: u32::from_le_bytes(b[24..28].try_into().unwrap()),
            offset: u64_at(32),
        }
    }
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
        self.space_at(req, 0)
    }

    /// Stop unless the request's u32 at `at` names an address space made with 33.
    fn space_at(&self, req: &Request, at: usize) -> Result<u32, Stop> {
        let id = req.u32_at(at);
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !st.spaces.contains(&id) {
            return Err(Stop(format!(
                "nx2:gpu {} of the address space {id}, which 33 did not make",
                req.command
            )));
        }
        Ok(id)
    }

    /// Carry out one request of 216: the range shows the memory and offset named, or with no
    /// memory what it showed before any was mapped there (a GPU-only region's own backing, else
    /// nothing).
    fn sparse_map(&self, r: &SparseMap) -> Result<(), String> {
        if r.size == 0 {
            return Ok(());
        }
        if r.gpu_address.checked_add(r.size).is_none() {
            return Err("the range wraps around".into());
        }
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if r.memory == 0 {
            let own = st
                .regions
                .range(..=r.gpu_address)
                .next_back()
                .filter(|&(&s, &(_, n))| r.gpu_address + r.size <= s + n)
                .map(|(&s, &(backing, _))| backing + (r.gpu_address - s));
            drop(st);
            return match own {
                Some(backing) => self.gpu.switch_2_map(r.gpu_address, r.size, backing, 0),
                None => self.gpu.switch_2_unmap(r.gpu_address, r.size),
            };
        }
        let &(start, len) = st
            .memories
            .get(&r.memory)
            .ok_or_else(|| format!("the GPU memory {}, which 101 did not make", r.memory))?;
        if r.offset.checked_add(r.size).is_none_or(|end| end > len) {
            return Err(format!(
                "past the end of the memory {}, of {len:#x} bytes",
                r.memory
            ));
        }
        drop(st);
        self.gpu
            .switch_2_map(r.gpu_address, r.size, start + r.offset, u32::from(r.kind))
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
            // CreateSparseRegion(u32 address space, u64 first, u64 last, u32 flags, ..., u32 slot
            // at 36), nn::gpu::sdk::CreateSparseRegion: GPU memory the program does not map from
            // its own. Without the flag 0x200 the graphics server backs it, the program never
            // reaching it (render targets): here zeroed pages outside the program's regions
            // (A-215). A sparse region stops the run.
            214 => {
                self.space(req)?;
                let (first, last, flags) = (req.u64_at(4), req.u64_at(12), req.u32_at(20));
                let len = Self::span(req, first, last)?;
                if flags & SPARSE_REGION != 0 {
                    return Err(Stop(format!(
                        "nx2:gpu 214 makes a sparse region [{first:#x}, {last:#x}] (flags \
                         {flags:#x}), which is not modelled"
                    )));
                }
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if st
                    .regions
                    .range(..=last)
                    .next_back()
                    .is_some_and(|(&s, &(_, n))| s + n > first)
                {
                    return Err(Stop(format!(
                        "nx2:gpu 214 makes [{first:#x}, {last:#x}] over another region"
                    )));
                }
                let backing = st
                    .back(&kernel, len)
                    .map_err(|e| Stop(format!("nx2:gpu 214 of [{first:#x}, {last:#x}]: {e}")))?;
                st.regions.insert(first, (backing, len));
                drop(st);
                self.gpu
                    .switch_2_map(first, len, backing, 0)
                    .map_err(|e| Stop(format!("nx2:gpu 214 of [{first:#x}, {last:#x}]: {e}")))?;
                self.complete(&kernel, req.u32_at(36));
                Ok(Response::ok())
            }
            // DestroySparseRegion(u32 address space, u64 first, u64 last, ..., u32 slot at 24):
            // the region 214 made goes; its backing is kept for a region of the same size.
            215 => {
                self.space(req)?;
                let (first, last) = (req.u64_at(4), req.u64_at(12));
                let len = Self::span(req, first, last)?;
                let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                let backing = match st.regions.get(&first) {
                    Some(&(backing, n)) if n == len => backing,
                    _ => {
                        return Err(Stop(format!(
                            "nx2:gpu 215 of [{first:#x}, {last:#x}], which 214 did not make"
                        )))
                    }
                };
                st.regions.remove(&first);
                st.spare.push((backing, len));
                drop(st);
                self.gpu
                    .switch_2_unmap(first, len)
                    .map_err(|e| Stop(format!("nx2:gpu 215 of [{first:#x}, {last:#x}]: {e}")))?;
                self.complete(&kernel, req.u32_at(24));
                Ok(Response::ok())
            }
            // ExecuteSparseMapRequests(u64 count, u32 address space, u64 request size, u32 sync,
            // u32 slot), nn::gpu::sdk::ExecuteSparseMapRequests: the requests are in the A buffer,
            // the B buffer gets the kind applied to each, the reply is how many were done (A-216).
            216 => {
                self.space_at(req, 8)?;
                let (count, size) = (req.u64_at(0), req.u64_at(12));
                if size != SPARSE_MAP_REQUEST as u64 {
                    return Err(Stop(format!(
                        "nx2:gpu 216 with requests of {size} bytes; they are of \
                         {SPARSE_MAP_REQUEST}"
                    )));
                }
                let bytes = count
                    .checked_mul(size)
                    .filter(|&n| req.input_buffer(0).is_some_and(|b| b.size >= n))
                    .ok_or_else(|| {
                        Stop(format!(
                            "nx2:gpu 216 of {count} requests, which its buffer does not hold"
                        ))
                    })?;
                let raw = kernel
                    .read_buffer(req.input_buffer(0).unwrap(), bytes)
                    .ok_or_else(|| Stop("nx2:gpu 216: its requests are not readable".into()))?;
                let mut kinds = Vec::with_capacity(raw.len() / SPARSE_MAP_REQUEST);
                for r in raw.as_chunks::<SPARSE_MAP_REQUEST>().0 {
                    let r = SparseMap::parse(r);
                    self.sparse_map(&r)
                        .map_err(|e| Stop(format!("nx2:gpu 216 {r:x?}: {e}")))?;
                    kinds.push(r.kind);
                }
                if let Some(out) = req.output_buffer(0) {
                    let n = (out.size as usize).min(kinds.len());
                    if !kernel.write_memory(out.addr, &kinds[..n]) {
                        return Err(Stop("nx2:gpu 216: its kinds buffer is not writable".into()));
                    }
                }
                self.complete(&kernel, req.u32_at(24));
                Ok(Response::ok().with_u64(count))
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
    /// The graphics system's native images, which 229 presents (A-212).
    images: Arc<std::sync::Mutex<NativeObjects>>,
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
    pub fn new(
        kernel: Weak<Kernel>,
        display: Arc<vi::Shared>,
        images: Arc<std::sync::Mutex<NativeObjects>>,
    ) -> Vi2Session {
        static SESSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Vi2Session {
            kernel,
            display,
            layers: Default::default(),
            key: SESSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) << 8,
            images,
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

    /// PresentClientBuffer (229), from nn::wsi::ounce::native::NativeWindow::Present: (u64 layer,
    /// u32 slot, u32 native image, ..., u32 shown size at byte 20 as ((height - 1) << 16) |
    /// (width - 1), u32 timing at byte 40 as (swap interval << 16) | refresh rate, the fence at
    /// bytes 80 and 88) -> (u64 slots the client may draw into, 16 bytes). The image is shown
    /// from its memory, after the commands submitted before the call (the renderer's order, so
    /// the fence is not waited on); the display holds no slot afterwards (A-236).
    fn present_client_buffer(&self, kernel: &Kernel, req: &Request) -> ServiceResult {
        let (layer, image) = (req.u64_at(0), req.u32_at(12));
        if !self.layers.contains(&layer) {
            return Err(Stop(format!("nx2:vi2 229 of the unknown layer {layer:#x}")));
        }
        let found = self
            .images
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .image_address(image);
        let Some((address, layout)) = found else {
            return Err(Stop(format!(
                "nx2:vi2 229 presents the native image {image}, which is not in bound memory"
            )));
        };
        let shown = req.u32_at(20);
        let (width, height) = ((shown & 0xFFFF) + 1, (shown >> 16) + 1);
        if width > layout.width || height > layout.height {
            return Err(Stop(format!(
                "nx2:vi2 229 shows {width}x{height} of the {}x{} image {image}",
                layout.width, layout.height
            )));
        }
        let request = nx_gpu::render::Present {
            address,
            width: layout.width,
            height: layout.height,
            pitch: layout.pitch as u32,
            // What the renderer shows is the render target at the address, in its own format.
            format: nx_gpu::display::FORMAT_RGBA_8888,
            layout: 3,
            block_height_log2: layout.block_rows.trailing_zeros().saturating_sub(3),
            crop: [0, 0, width as i32, height as i32],
            frame: 0,
            due: None,
        };
        self.display
            .present_native(kernel, request, req.u32_at(40) >> 16)?;
        Ok(Response::ok()
            .with_u64(VI2_ALL_SLOTS)
            .with_u64(0)
            .with_u64(0))
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
            // (u64, u64), from nn::wsi::ounce::native::NativeWindow::Present before the frame is
            // handed over; only the result is read, and nothing here depends on it (A-235).
            351 => Ok(Response::ok()),
            // (u64 layer, f32, f32, f32), from the same Present when three values of the window
            // that it keeps have changed; only the result is read, and the display here does
            // nothing with them (A-235).
            352 => Ok(Response::ok()),
            229 => self.present_client_buffer(&kernel, req),
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
        if let Some(answer) = self.native_command(req) {
            return answer;
        }
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
            201 | 301 | 401..=403 | 494 | 601 | 801 | 1101 | 1102 | 1201 | 1501..=1503 | 1601 => {
                Ok(Response::ok())
            }
            // The video decoder's and the video encoder's, from
            // nn::codec::video::detail::InitializeServerContext (A-261).
            901 | 1001 => Ok(Response::ok()),
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
                Vi2Session::new(
                    self.kernel.clone(),
                    self.display.clone(),
                    self.native.clone(),
                ),
            )))),
            // CreateCameraContext(u64 1), from nn::camera::sdk::CreateCameraContext after the check
            // 601: the camera's session (A-222).
            602 => {
                if req.u64_at(0) != 1 {
                    return Err(Stop(format!(
                        "nx2:gsv 602 with {:#x}; only 1 is known",
                        req.u64_at(0)
                    )));
                }
                Ok(
                    Response::ok().with_copy_handle(KObject::session(Box::new(Camera {
                        kernel: self.kernel.clone(),
                        event: None,
                    }))),
                )
            }
            // nn::chat::gsv::Initialize, after the check 1201: GameChat's graphics session (A-219).
            1202 => Ok(
                Response::ok().with_copy_handle(KObject::session(Box::new(ChatGsv {
                    kernel: self.kernel.clone(),
                    events: Vec::new(),
                    opened: Default::default(),
                }))),
            ),
            // The video decoder's and the video encoder's sessions, from
            // nn::codec::video::detail::InitializeServerContext after the checks 901 and 1001
            // (906 takes a u8 and the decoder library's version; 902 is older SDKs', A-261).
            902 | 906 | 1002 => Ok(Response::ok().with_copy_handle(KObject::session(Box::new(
                vdec::VideoServer::new(
                    self.kernel.clone(),
                    self.gpu.clone(),
                    self.native.clone(),
                    if req.command == 1002 {
                        vdec::Codec::Encoder
                    } else {
                        vdec::Codec::Decoder
                    },
                ),
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
        assert!(s.request_tipc(&req(1201)).is_ok(), "GameChat's");
        assert!(s.request_tipc(&req(601)).is_ok(), "the camera's");
        let mut camera = req(602);
        camera.data[..8].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(s.request_tipc(&camera).unwrap().copy_handles.len(), 1);
        // The camera session: "Default" is not attached, the result the title reads as no camera.
        let mut open = req(3);
        open.data[..8].copy_from_slice(b"Default\0");
        let mut camera = Camera {
            kernel: Weak::new(),
            event: None,
        };
        assert_eq!(camera.request(&open).unwrap().result, NO_CAMERA);
        assert!(
            camera.request(&req(21)).unwrap_err().0.contains("gone"),
            "needs a kernel"
        );
        assert!(camera.request(&req(22)).is_err() && camera.request_tipc(&req(3)).is_err());
        assert_eq!(s.request_tipc(&req(1202)).unwrap().copy_handles.len(), 1);
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
    fn image_layouts_from_preferences() {
        // A 3840x2160 window image as the title asks for one: kind 2, blocks of 2^7 rows, a pitch
        // of 240 units of 64 bytes, format 1.
        let mut window = [0u32; 16];
        window[1] = 2;
        window[2] = 3839 | 2159 << 16;
        window[4] = 7;
        window[5] = 240;
        window[6] = 1;
        let l = ImageLayout::from_preference(&window).unwrap();
        assert_eq!(
            (l.width, l.height, l.pitch, l.block_rows),
            (3840, 2160, 15360, 128)
        );
        assert_eq!((l.size, l.format), (15360 * 2176, 1));
        // A GameChat camera image gives only its size: four bytes a pixel, blocks of 128 rows.
        let mut chat = [0u32; 16];
        chat[1] = 1;
        chat[2] = 383 | 215 << 16;
        assert!(ImageLayout::from_preference(&chat).is_err(), "no pitch");
        let c = ImageLayout::from_chat_preference(&chat).unwrap();
        assert_eq!(
            (c.pitch, c.block_rows, c.size, c.alignment),
            (1536, 128, 1536 * 256, 0x2000)
        );
        let mut s = Supervisor::new(
            Weak::new(),
            nvdrv::Shared::new(),
            Arc::new(vi::Shared::new()),
        );
        let mut size = req(1291);
        size.data = chat
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .chain(3u32.to_le_bytes())
            .collect();
        let r = s.request_tipc(&size).unwrap();
        assert_eq!(r.data[..8], (3u64 * 1536 * 256).to_le_bytes());
        assert_eq!(r.data[8..16], 0x2000u64.to_le_bytes());
        // 111 and 1211: three GameChat images in native memory; 428 describes the second.
        let mut made = req(111);
        made.data[4..12].copy_from_slice(&0x12_0000u64.to_le_bytes());
        let memory = s.request_tipc(&made).unwrap().data;
        let mut pool = size;
        pool.command = 1211;
        pool.data.extend_from_slice(&memory);
        let first =
            u32::from_le_bytes(s.request_tipc(&pool).unwrap().data[..4].try_into().unwrap());
        let mut query = req(428);
        query.data[..4].copy_from_slice(&(first + 1).to_le_bytes());
        let d = s.request_tipc(&query).unwrap().data;
        assert_eq!(d.len(), 37 * 4);
        let word = |n: usize| u32::from_le_bytes(d[4 * n..4 * n + 4].try_into().unwrap());
        assert_eq!(
            d[8..16],
            (1536u64 * 256).to_le_bytes(),
            "the second image's offset"
        );
        // The fields the SDK checks: byte 0 and byte 0x20 of the descriptor are 1, its size not 0.
        assert_eq!((word(4), word(4 + 8), word(4 + 11)), (1, 1, 1536 * 256));
        assert_eq!(word(4 + 2), 383 | 215 << 16);
        query.data[..4].copy_from_slice(&(first + 3).to_le_bytes());
        assert!(s.request_tipc(&query).is_err(), "the pool has three");
        // 426 maps an image at its own address once its memory is bound (114).
        let mut map = req(426);
        map.data[..16].copy_from_slice(&[first + 1, 1, 1, 3].map(u32::to_le_bytes).concat());
        assert!(s.request_tipc(&map).is_err(), "no address space 1 yet");
        s.gpu.set_switch_2(true);
        s.gpu_state.lock().unwrap().spaces.insert(1);
        assert!(s.request_tipc(&map).is_err(), "the memory is not bound");
        let mut bind = req(114);
        bind.data[..4].copy_from_slice(&memory);
        bind.data[4..12].copy_from_slice(&0x30_0040_0000u64.to_le_bytes());
        bind.data[12..20].copy_from_slice(&0x12_0000u64.to_le_bytes());
        s.request_tipc(&bind).unwrap();
        let d = s.request_tipc(&map).unwrap().data;
        let image: u64 = 0x30_0040_0000 + 1536 * 256;
        assert_eq!(d[..8], image.to_le_bytes(), "the image's GPU address");
        assert_eq!(s.gpu.switch_2_translate(image + 0x10), Some(image + 0x10));
    }

    #[test]
    fn gamechat_events_are_handed_out_by_id_and_never_set() {
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        let kernel = Kernel::new(ProcessInfo::new("chat", ProcessKind::Nso, None));
        let mut c = ChatGsv {
            kernel: Arc::downgrade(&kernel),
            events: Vec::new(),
            opened: Default::default(),
        };
        assert_eq!(c.request_tipc(&req(1)).unwrap().data, 0u32.to_le_bytes());
        assert_eq!(c.request_tipc(&req(1)).unwrap().data, 1u32.to_le_bytes());
        let mut get = req(11);
        get.data[..4].copy_from_slice(&1u32.to_le_bytes());
        let event = c.request_tipc(&get).unwrap();
        assert_eq!(event.copy_handles.len(), 1);
        assert!(!kernel.event_signaled(&event.copy_handles[0]));
        get.data[..4].copy_from_slice(&5u32.to_le_bytes());
        assert!(c.request_tipc(&get).is_err(), "an id 1 did not hand out");
        assert!(c.request_tipc(&req(2)).is_err());
        // 103: a virtual display by name and id, once per id.
        let mut open = req(103);
        open.data[..8].copy_from_slice(b"chat2\0\0\0");
        open.data[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert!(c.request_tipc(&open).is_ok());
        assert!(c.request_tipc(&open).is_err(), "the id is taken");
        open.data[..8].copy_from_slice(&[0; 8]);
        open.data[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(c.request_tipc(&open).is_err(), "no name");
        // 121: a metadata pool, in the same space of ids.
        let mut pool = req(121);
        pool.data[..8].copy_from_slice(&0x30_0000_1000u64.to_le_bytes());
        pool.data[8..16].copy_from_slice(&3u64.to_le_bytes());
        pool.data[16..20].copy_from_slice(&6u32.to_le_bytes());
        assert!(c.request_tipc(&pool).is_ok());
        pool.data[16..20].copy_from_slice(&1u32.to_le_bytes());
        assert!(c.request_tipc(&pool).is_err(), "1 is the display's");
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

    #[test]
    fn a_gpu_only_region_is_backed_by_the_server_and_its_backing_reused() {
        let _serial = SERVER_MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        use crate::rt::runtime::rt;
        let kernel = Kernel::new(ProcessInfo::new("gpu", ProcessKind::Nso, None));
        let gpu = nvdrv::Shared::new();
        gpu.set_switch_2(true);
        let mut g = GpuSession {
            kernel: Arc::downgrade(&kernel),
            gpu: gpu.clone(),
            state: Arc::default(),
        };
        let region = |command: u32, first: u64, last: u64, flags: u32, slot: (usize, u32)| {
            let mut data = vec![0u8; 64];
            data[0..4].copy_from_slice(&1u32.to_le_bytes());
            data[4..12].copy_from_slice(&first.to_le_bytes());
            data[12..20].copy_from_slice(&last.to_le_bytes());
            data[20..24].copy_from_slice(&flags.to_le_bytes());
            data[slot.0..slot.0 + 4].copy_from_slice(&slot.1.to_le_bytes());
            Request {
                command,
                data,
                ..Request::default()
            }
        };
        g.request_tipc(&with_u32(33, 0)).unwrap();
        // Two pages at GPU address 0x81_0000_0000, the flags a render target's region has.
        g.request_tipc(&region(214, 0x81_0000_0000, 0x81_0000_1FFF, 0xC, (36, 5)))
            .unwrap();
        let backing = gpu.switch_2_translate(0x81_0000_0000).unwrap();
        assert_eq!(backing, layout::GPU_SERVER_BASE);
        assert_eq!(
            gpu.switch_2_translate(0x81_0000_1008),
            Some(backing + 0x1008)
        );
        assert!(kernel.event_signaled(&g.event(&kernel, 5).0));
        let block = kernel.query_memory(backing);
        assert_eq!(
            (block.base, block.size, block.info.mem_type),
            (backing, 0x2000, layout::mem_type::INACCESSIBLE),
            "the backing is a block of its own the program is told it cannot use"
        );
        // Over the region, or a sparse one: stops.
        assert!(g
            .request_tipc(&region(214, 0x81_0000_1000, 0x81_0000_2FFF, 0xC, (36, 5)))
            .is_err());
        assert!(g
            .request_tipc(&region(214, 0x82_0000_0000, 0x82_0000_FFFF, 0x60C, (36, 5)))
            .is_err());
        // 215 unmaps it; a region of the same size gets the backing again, zeroed.
        rt().mem.write_bytes(backing + 8, &[7], false).unwrap();
        g.request_tipc(&region(215, 0x81_0000_0000, 0x81_0000_1FFF, 0, (24, 6)))
            .unwrap();
        assert_eq!(gpu.switch_2_translate(0x81_0000_0000), None);
        assert!(kernel.event_signaled(&g.event(&kernel, 6).0));
        assert!(
            g.request_tipc(&region(215, 0x81_0000_0000, 0x81_0000_1FFF, 0, (24, 6)))
                .is_err(),
            "destroyed once"
        );
        g.request_tipc(&region(214, 0x90_0000_0000, 0x90_0000_1FFF, 0xC, (36, 7)))
            .unwrap();
        assert_eq!(gpu.switch_2_translate(0x90_0000_0000), Some(backing));
        let mut byte = [1u8];
        rt().mem.read_bytes(backing + 8, &mut byte, false).unwrap();
        assert_eq!(byte, [0]);
    }

    #[test]
    fn sparse_map_requests_point_ranges_at_gpu_memory_and_back() {
        let _serial = SERVER_MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        use super::super::super::service::Buffer;
        use crate::rt::{abi::perm, runtime::rt};
        // The requests and the kinds, in a page of guest memory of this test's own.
        const PAGE: u64 = 0x3E_0000_0000;
        let _ = rt().mem.unmap(PAGE, 0x1000);
        rt().mem.map(PAGE, 0x1000, perm::R | perm::W).unwrap();
        let kernel = Kernel::new(ProcessInfo::new("gpu", ProcessKind::Nso, None));
        let gpu = nvdrv::Shared::new();
        gpu.set_switch_2(true);
        let mut g = GpuSession {
            kernel: Arc::downgrade(&kernel),
            gpu: gpu.clone(),
            state: Arc::default(),
        };
        let fields = |command: u32, fields: &[(usize, u64, usize)]| {
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
        g.request_tipc(&with_u32(33, 0)).unwrap();
        // Four pages of GPU memory at 0x30_0010_0000, and a GPU-only region of four pages.
        g.request_tipc(&fields(
            101,
            &[(0, 0x30_0010_0000, 8), (8, 0x30_0010_3FFF, 8)],
        ))
        .unwrap();
        let region = [
            (0, 1, 4),
            (4, 0x81_0000_0000, 8),
            (12, 0x81_0000_3FFF, 8),
            (20, 0xC, 4),
        ];
        g.request_tipc(&fields(214, &region)).unwrap();
        let backing = gpu.switch_2_translate(0x81_0000_0000).unwrap();
        // The region's second page shows the memory's third, kind 0xFE; a range outside any
        // region shows its first.
        let request = |gpu_address: u64, size: u64, kind: u8, memory: u32, offset: u64| {
            let mut r = [0u8; 40];
            r[0..8].copy_from_slice(&gpu_address.to_le_bytes());
            r[8..16].copy_from_slice(&size.to_le_bytes());
            r[19] = kind;
            r[24..28].copy_from_slice(&memory.to_le_bytes());
            r[32..40].copy_from_slice(&offset.to_le_bytes());
            r
        };
        let send = |g: &mut GpuSession, requests: &[[u8; 40]], slot: u64| {
            let raw = requests.concat();
            rt().mem.write_bytes(PAGE, &raw, false).unwrap();
            let mut r = fields(
                216,
                &[
                    (0, requests.len() as u64, 8),
                    (8, 1, 4),
                    (12, 40, 8),
                    (24, slot, 4),
                ],
            );
            r.send_buffers = vec![Buffer {
                addr: PAGE,
                size: raw.len() as u64,
            }];
            r.receive_buffers = vec![Buffer {
                addr: PAGE + 0x800,
                size: requests.len() as u64,
            }];
            g.request_tipc(&r)
        };
        let done = send(
            &mut g,
            &[
                request(0x81_0000_1000, 0x1000, 0xFE, 1, 0x2000),
                request(0x50_0000_0000, 0x1000, 0, 1, 0),
            ],
            9,
        )
        .unwrap();
        assert_eq!(done.data, 2u64.to_le_bytes());
        assert_eq!(gpu.switch_2_translate(0x81_0000_1010), Some(0x30_0010_2010));
        assert_eq!(
            gpu.switch_2_translate(0x81_0000_2000),
            Some(backing + 0x2000)
        );
        assert_eq!(gpu.switch_2_translate(0x50_0000_0004), Some(0x30_0010_0004));
        let mut kinds = [0u8; 2];
        rt().mem
            .read_bytes(PAGE + 0x800, &mut kinds, false)
            .unwrap();
        assert_eq!(kinds, [0xFE, 0]);
        assert!(kernel.event_signaled(&g.event(&kernel, 9).0));
        // Without memory: the region's page shows its own backing again, the other nothing.
        send(
            &mut g,
            &[
                request(0x81_0000_1000, 0x1000, 0, 0, 0),
                request(0x50_0000_0000, 0x1000, 0, 0, 0),
            ],
            9,
        )
        .unwrap();
        assert_eq!(
            gpu.switch_2_translate(0x81_0000_1010),
            Some(backing + 0x1010)
        );
        assert_eq!(gpu.switch_2_translate(0x50_0000_0004), None);
        // Memory 101 did not make, past its end, or more requests than the buffer holds: stops.
        assert!(send(&mut g, &[request(0x50_0000_0000, 0x1000, 0, 2, 0)], 9).is_err());
        assert!(send(&mut g, &[request(0x50_0000_0000, 0x2000, 0, 1, 0x3000)], 9).is_err());
        let mut short = fields(216, &[(0, 3, 8), (8, 1, 4), (12, 40, 8)]);
        short.send_buffers = vec![Buffer {
            addr: PAGE,
            size: 80,
        }];
        assert!(g.request_tipc(&short).is_err());
        rt().mem.unmap(PAGE, 0x1000).unwrap();
    }

    /// The tests whose kernels map GPU-only backing, at the same addresses, one at a time.
    static SERVER_MEMORY: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_u32(command: u32, v: u32) -> Request {
        Request {
            command,
            data: v.to_le_bytes().to_vec(),
            ..Request::default()
        }
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
        let mut v = Vi2Session::new(
            Arc::downgrade(&kernel),
            Arc::new(vi::Shared::new()),
            Arc::default(),
        );
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

    /// A request whose data is `words`, little-endian.
    fn words(command: u32, words: &[u32]) -> Request {
        Request {
            command,
            data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            ..Request::default()
        }
    }

    #[test]
    fn decoded_frames_go_into_the_images_in_the_order_they_were_handed_out() {
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        use super::super::super::service::Buffer;
        let _one_heap = super::super::super::tests::LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let kernel = Kernel::new(ProcessInfo::new("vdec", ProcessKind::Nso, None));
        let heap = kernel.set_heap_size(0x40_0000).unwrap();
        let mut s = Supervisor::new(
            Arc::downgrade(&kernel),
            nvdrv::Shared::new(),
            Arc::new(vi::Shared::new()),
        );
        assert!(s.request_tipc(&req(901)).is_ok(), "the decoder's check");
        let session = s.request_tipc(&req(906)).unwrap();
        assert!(matches!(session.copy_handles[..], [KObject::Session(_)]));
        // A pool of two NV12 images of 8x8 in the memory bound at heap + 1 MiB.
        let mut preference = vec![0u32; 16];
        preference[1] = 1;
        preference[2] = 7 | 7 << 16;
        let layout = DecoderLayout::from_preference(&preference).unwrap();
        let mut ask = preference.clone();
        ask.push(2);
        let r = s.request_tipc(&words(991, &ask)).unwrap();
        assert_eq!(r.data[..8], (2 * layout.size).to_le_bytes());
        let mut create = req(111);
        create.data[4..12].copy_from_slice(&(2 * layout.size).to_le_bytes());
        assert_eq!(s.request_tipc(&create).unwrap().data, 1u32.to_le_bytes());
        let memory = heap + 0x10_0000;
        let mut bind = req(114);
        bind.data[..4].copy_from_slice(&1u32.to_le_bytes());
        bind.data[4..12].copy_from_slice(&memory.to_le_bytes());
        bind.data[12..20].copy_from_slice(&(2 * layout.size).to_le_bytes());
        assert!(s.request_tipc(&bind).is_ok());
        let mut pool = ask.clone();
        pool.push(1);
        let first = s.request_tipc(&words(911, &pool)).unwrap().data;
        let first = u32::from_le_bytes(first[..4].try_into().unwrap());
        let d = s.request_tipc(&words(961, &[first + 1])).unwrap().data;
        assert_eq!(d.len(), vdec::CPU_DESCRIPTOR_SIZE);
        assert_eq!(d[0], 10, "NV12");
        assert_eq!(d[8..16], layout.size.to_le_bytes(), "the second image");
        assert_eq!(d[0x18..0x20], (layout.size + layout.chroma).to_le_bytes());

        let mut v =
            vdec::DecoderService::new(Arc::downgrade(&kernel), s.gpu.clone(), s.native.clone());
        kernel.create_transfer_memory(heap, 0x1000, 3).unwrap();
        let mut init = req(201);
        init.copy_handles.push(KObject::Memory(0));
        assert!(v.request_tipc(&init).is_ok());
        assert!(v.request_tipc(&req(201)).is_err(), "no work buffer");
        // The codec's name and the frames come in send buffers in guest memory.
        let at = heap + 0x2000;
        assert!(kernel.write_memory(at, b"video/vp9\0"));
        let mut configure = req(101);
        configure.send_buffers.push(Buffer { addr: at, size: 10 });
        assert!(v.request_tipc(&configure).is_ok());
        let handle = |r: Response| {
            let w = |i: usize| u32::from_le_bytes(r.data[4 * i..4 * i + 4].try_into().unwrap());
            (
                !w(0),
                w(1),
                u64::from_le_bytes(r.data[8..16].try_into().unwrap()),
            )
        };
        let (syncpoint, mode, one) = handle(v.request_tipc(&words(311, &[first])).unwrap());
        let (again, _, two) = handle(v.request_tipc(&words(311, &[first + 1])).unwrap());
        assert_eq!((again, mode, two), (syncpoint, 0, one + 1));
        let frame = nx_gpu::media::vp9::gray_key_frame(nx_gpu::media::vp9::Color::default());
        let at = heap + 0x3000;
        assert!(kernel.write_memory(at, &frame));
        let submit = |v: &mut vdec::DecoderService, size: u64, pts: u64, end: u32| {
            let mut r = words(109, &[pts as u32, (pts >> 32) as u32, end]);
            r.send_buffers.push(Buffer { addr: at, size });
            v.request_tipc(&r).unwrap().data
        };
        assert_eq!(
            submit(&mut v, frame.len() as u64, 1000, 0),
            1u64.to_le_bytes()
        );
        let reached = |value: u64| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !nx_gpu::syncpt::reached(s.gpu.syncpoint_value(syncpoint).unwrap(), value as u32)
            {
                assert!(
                    std::time::Instant::now() < deadline,
                    "the syncpoint reaches {value}"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        reached(one);
        let props = v.request_tipc(&req(213)).unwrap().data;
        let mut expect: Vec<u8> = [first, 8, 8, 8, 8, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        expect.extend_from_slice(&1_000_000u64.to_le_bytes());
        expect.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            props, expect,
            "the image, its size, and the time in nanoseconds"
        );
        let mut luma = vec![0u8; 8];
        let mut chroma = vec![0u8; 8];
        for row in 0..8 {
            assert!(kernel.read_memory(memory + row * layout.pitch, &mut luma));
            assert!(luma.iter().all(|&b| b == 128), "luma row {row}");
        }
        for row in 0..4 {
            assert!(kernel.read_memory(memory + layout.chroma + row * layout.pitch, &mut chroma));
            assert!(chroma.iter().all(|&b| b == 128), "chroma row {row}");
        }
        // The end of the stream, with no frame in it, still takes the next image.
        assert_eq!(submit(&mut v, 0, 2000, 1), 2u64.to_le_bytes());
        reached(two);
        let props = v.request_tipc(&req(213)).unwrap().data;
        assert_eq!(props[..4], (first + 1).to_le_bytes());
        assert_eq!(props[32..36], 1u32.to_le_bytes(), "the end");
        assert!(
            v.request_tipc(&req(213)).is_err(),
            "nothing else was written"
        );
        // An image given back with the fence of the program's reads is handed out again; stopping
        // gives it back to the program at once.
        let mut release = words(314, &[first, 0, !0, 0, 0, 0]);
        release.data.truncate(24);
        let (_, _, three) = handle(v.request_tipc(&release).unwrap());
        assert_eq!(three, two + 1);
        assert!(v.request_tipc(&req(207)).is_ok());
        reached(three);
        assert!(
            v.request_tipc(&req(999)).is_err(),
            "an unknown command stops"
        );
        let mut other = req(101);
        assert!(kernel.write_memory(heap + 0x2000, b"video/avc\0"));
        other.send_buffers.push(Buffer {
            addr: heap + 0x2000,
            size: 10,
        });
        assert!(v.request_tipc(&other).is_err(), "only VP9 is decoded");
    }
}
