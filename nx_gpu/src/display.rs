//! The display: displays, layers, and the buffer queue behind each layer.
//!
//! A title draws to a layer through an Android `IGraphicBufferProducer`: it hands the queue its
//! render targets (`setPreallocatedBuffer`), takes one (`dequeueBuffer`), draws, and gives it back
//! (`queueBuffer`). The consumer on the other side is the system's compositor, which shows the
//! newest queued buffer and releases the one it showed before. Here the consumer takes a queued
//! buffer at once, so the queue always has a free buffer for a producer that returns what it takes.
//!
//! The codes, parcel layouts and field orders are those `libnx` uses for the Switch's variant of
//! the interface (an older Android one). Nothing is drawn yet: a presented frame is counted and
//! its buffer is recorded.

use crate::event::{EventToken, Events, Signal};
use crate::parcel::{ParcelReader, ParcelWriter};
use std::collections::{BTreeMap, VecDeque};

pub const DEFAULT_WIDTH: u32 = 1280;
pub const DEFAULT_HEIGHT: u32 = 720;
pub const SLOT_COUNT: usize = 64;
pub const INTERFACE: &str = "android.gui.IGraphicBufferProducer";

/// `IGraphicBufferProducer` transaction codes (the Switch's variant).
pub mod code {
    pub const REQUEST_BUFFER: u32 = 1;
    pub const SET_BUFFER_COUNT: u32 = 2;
    pub const DEQUEUE_BUFFER: u32 = 3;
    pub const DETACH_BUFFER: u32 = 4;
    pub const DETACH_NEXT_BUFFER: u32 = 5;
    pub const ATTACH_BUFFER: u32 = 6;
    pub const QUEUE_BUFFER: u32 = 7;
    pub const CANCEL_BUFFER: u32 = 8;
    pub const QUERY: u32 = 9;
    pub const CONNECT: u32 = 10;
    pub const DISCONNECT: u32 = 11;
    pub const SET_SIDEBAND_STREAM: u32 = 12;
    pub const ALLOCATE_BUFFERS: u32 = 13;
    pub const SET_PREALLOCATED_BUFFER: u32 = 14;
}

/// Android's `status_t` values, negated errno.
pub mod status {
    pub const OK: i32 = 0;
    pub const WOULD_BLOCK: i32 = -11;
    pub const NO_INIT: i32 = -19;
    pub const BAD_VALUE: i32 = -22;
    pub const INVALID_OPERATION: i32 = -38;
}

/// `NATIVE_WINDOW_*` query selectors.
mod query {
    pub const WIDTH: i32 = 0;
    pub const HEIGHT: i32 = 1;
    pub const FORMAT: i32 = 2;
    pub const MIN_UNDEQUEUED_BUFFERS: i32 = 3;
    pub const DEFAULT_WIDTH: i32 = 6;
    pub const DEFAULT_HEIGHT: i32 = 7;
    pub const TRANSFORM_HINT: i32 = 8;
    pub const CONSUMER_RUNNING_BEHIND: i32 = 9;
}

/// `PIXEL_FORMAT_RGBA_8888`.
pub const FORMAT_RGBA_8888: u32 = 1;

/// A render target registered with the queue (Android's `GraphicBuffer`, flattened).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphicBuffer {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32,
    pub usage: u32,
    /// The `nvmap` handle its memory is behind (the second word of the native handle's data).
    pub nvmap_id: u32,
    /// The words of the native handle after its header.
    pub ints: Vec<u32>,
    /// The flattened bytes as the program sent them, answered back by `requestBuffer`.
    raw: Vec<u8>,
}

const GRAPHIC_BUFFER_MAGIC: u32 = 0x4742_4652; // "GBFR"

impl GraphicBuffer {
    /// Parse the flattened form: the magic, the size and format fields, the native handle.
    pub fn parse(flat: &[u8]) -> Option<GraphicBuffer> {
        let word = |i: usize| -> Option<u32> {
            let b = flat.get(i * 4..i * 4 + 4)?;
            Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        if word(0)? != GRAPHIC_BUFFER_MAGIC || word(8)? != 0 {
            return None;
        }
        let num_ints = word(9)? as usize;
        let ints = (0..num_ints)
            .map(|i| word(10 + i))
            .collect::<Option<Vec<u32>>>()?;
        Some(GraphicBuffer {
            width: word(1)?,
            height: word(2)?,
            stride: word(3)?,
            format: word(4)?,
            usage: word(5)?,
            nvmap_id: *ints.get(1)?,
            ints,
            raw: flat.to_vec(),
        })
    }

    /// A word of the native handle after its header, for the fields of the surface description
    /// `NvGraphicBuffer` carries (`libnx`'s `graphic_buffer.h`): the first plane starts at word
    /// 13 with width, height, color format, layout, pitch, an unused word, offset, kind and
    /// the block height.
    fn plane_word(&self, index: usize) -> u32 {
        self.ints.get(13 + index).copied().unwrap_or(0)
    }

    /// The layout of the first plane (`libnx`'s `NvLayout`: 1 pitch linear, 2 Tegra tiled, 3 block
    /// linear).
    pub fn layout(&self) -> u32 {
        self.plane_word(3)
    }

    /// The distance between rows in bytes (of a pitch linear plane).
    pub fn pitch(&self) -> u32 {
        self.plane_word(4)
    }

    /// Where the plane starts in its `nvmap` memory.
    pub fn offset(&self) -> u32 {
        self.plane_word(6)
    }

    /// The height of a block of a block linear plane, as a power of two in GOBs.
    pub fn block_height_log2(&self) -> u32 {
        self.plane_word(8)
    }
}

/// What `queueBuffer` says about the buffer (the Switch's `QueueBufferInput`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueInput {
    pub timestamp: i64,
    pub crop: [i32; 4],
    pub scaling_mode: i32,
    pub transform: u32,
    pub sticky_transform: u32,
    pub swap_interval: u32,
    /// The fences to wait for before reading the buffer: (syncpoint id, value).
    pub fences: Vec<(u32, u32)>,
}

impl QueueInput {
    /// 84 bytes: timestamp, auto-timestamp flag, crop, scaling mode, transform, sticky transform,
    /// one word, swap interval, and a fence list of a count and four (id, value) pairs.
    pub fn parse(flat: &[u8]) -> Option<QueueInput> {
        if flat.len() < 84 {
            return None;
        }
        let word = |off: usize| u32::from_le_bytes(flat[off..off + 4].try_into().unwrap());
        let count = word(48) as usize;
        if count > 4 {
            return None;
        }
        Some(QueueInput {
            timestamp: i64::from_le_bytes(flat[0..8].try_into().unwrap()),
            crop: [
                word(12) as i32,
                word(16) as i32,
                word(20) as i32,
                word(24) as i32,
            ],
            scaling_mode: word(28) as i32,
            transform: word(32),
            sticky_transform: word(36),
            swap_interval: word(44),
            fences: (0..count)
                .map(|i| (word(52 + 8 * i), word(56 + 8 * i)))
                .collect(),
        })
    }
}

/// A frame the compositor showed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub number: u64,
    pub slot: usize,
    pub buffer: GraphicBuffer,
    pub input: QueueInput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Dequeued,
    /// Shown, and still on screen.
    Acquired,
}

#[derive(Debug)]
struct Slot {
    state: SlotState,
    buffer: Option<GraphicBuffer>,
    /// The number of the last frame this slot carried; the oldest free slot is dequeued first.
    last_frame: u64,
    requested: bool,
}

/// The producer's queue of one layer.
#[derive(Debug)]
pub struct BufferQueue {
    slots: Vec<Slot>,
    connected: bool,
    width: u32,
    height: u32,
    buffer_count: i32,
    front: Option<usize>,
    frames: u64,
    last_frame: Option<Frame>,
    recent: VecDeque<(usize, u32)>,
}

impl BufferQueue {
    fn new() -> BufferQueue {
        BufferQueue {
            slots: (0..SLOT_COUNT)
                .map(|_| Slot {
                    state: SlotState::Free,
                    buffer: None,
                    last_frame: 0,
                    requested: false,
                })
                .collect(),
            connected: false,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            buffer_count: 0,
            front: None,
            frames: 0,
            last_frame: None,
            recent: VecDeque::new(),
        }
    }

    /// Is a buffer free for `dequeueBuffer`?
    pub fn has_free_buffer(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.state == SlotState::Free && s.buffer.is_some())
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn last_frame(&self) -> Option<&Frame> {
        self.last_frame.as_ref()
    }

    fn output(&self, w: &mut ParcelWriter) {
        // QueueBufferOutput: width, height, transform hint, number of pending buffers.
        w.write_u32(self.width);
        w.write_u32(self.height);
        w.write_u32(0);
        w.write_u32(0);
    }

    fn slot(&mut self, i: i32) -> Option<&mut Slot> {
        usize::try_from(i).ok().and_then(|i| self.slots.get_mut(i))
    }

    /// One transaction. The reply is the parcel's data (the caller wraps it); `Err` is a
    /// transaction this implementation does not handle.
    fn transact(&mut self, code: u32, r: &mut ParcelReader<'_>) -> Result<ParcelWriter, String> {
        let mut w = ParcelWriter::new();
        match code {
            // s32 slot -> [s32 nonnull, flattened buffer], s32 status
            code::REQUEST_BUFFER => {
                let slot = r.read_i32().ok_or("requestBuffer without a slot")?;
                match self.slot(slot) {
                    Some(s) if s.state == SlotState::Dequeued && s.buffer.is_some() => {
                        s.requested = true;
                        w.write_i32(1);
                        w.write_flattened(&s.buffer.as_ref().map(|b| b.raw.clone()).unwrap());
                        w.write_i32(status::OK);
                    }
                    _ => {
                        w.write_i32(0);
                        w.write_i32(status::BAD_VALUE);
                    }
                }
            }
            // s32 count -> s32 status
            code::SET_BUFFER_COUNT => {
                let n = r.read_i32().ok_or("setBufferCount without a count")?;
                if !(0..=SLOT_COUNT as i32).contains(&n) {
                    w.write_i32(status::BAD_VALUE);
                } else {
                    self.buffer_count = n;
                    w.write_i32(status::OK);
                }
            }
            // s32 async, u32 width, u32 height, s32 format, u32 usage
            //   -> s32 slot, s32 has fence, flattened fence, s32 status
            code::DEQUEUE_BUFFER => {
                let _async = r.read_i32().ok_or("dequeueBuffer: short parcel")?;
                let free = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.state == SlotState::Free && s.buffer.is_some())
                    .min_by_key(|(i, s)| (s.last_frame, *i))
                    .map(|(i, _)| i);
                match free {
                    Some(i) => {
                        self.slots[i].state = SlotState::Dequeued;
                        w.write_i32(i as i32);
                        w.write_i32(1);
                        // An empty fence list: the buffer can be written at once.
                        w.write_flattened(&[0u8; 36]);
                        w.write_i32(status::OK);
                    }
                    None => {
                        // Nothing frees a buffer but the program itself, so a blocking dequeue
                        // would never return; the answer is the one a non-blocking call gets.
                        w.write_i32(-1);
                        w.write_i32(0);
                        w.write_i32(status::WOULD_BLOCK);
                    }
                }
            }
            // s32 slot, flattened queue input -> queue output, s32 status
            code::QUEUE_BUFFER => {
                let slot = r.read_i32().ok_or("queueBuffer without a slot")?;
                let flat = r.read_flattened().ok_or("queueBuffer without its input")?;
                let input =
                    QueueInput::parse(flat).ok_or("queueBuffer: input of the wrong size")?;
                if !matches!(self.slot(slot), Some(s) if s.state == SlotState::Dequeued) {
                    self.output(&mut w);
                    w.write_i32(status::BAD_VALUE);
                } else {
                    self.present(slot as usize, input);
                    self.output(&mut w);
                    w.write_i32(status::OK);
                }
            }
            // s32 slot, flattened fence -> (nothing)
            code::CANCEL_BUFFER => {
                let slot = r.read_i32().ok_or("cancelBuffer without a slot")?;
                if let Some(s) = self.slot(slot) {
                    if s.state == SlotState::Dequeued {
                        s.state = SlotState::Free;
                    }
                }
            }
            // s32 what -> s32 value, s32 status
            code::QUERY => {
                let what = r.read_i32().ok_or("query without a selector")?;
                let value = match what {
                    query::WIDTH | query::DEFAULT_WIDTH => Some(self.width as i32),
                    query::HEIGHT | query::DEFAULT_HEIGHT => Some(self.height as i32),
                    query::FORMAT => Some(FORMAT_RGBA_8888 as i32),
                    query::MIN_UNDEQUEUED_BUFFERS => Some(1),
                    query::TRANSFORM_HINT | query::CONSUMER_RUNNING_BEHIND => Some(0),
                    _ => None,
                };
                match value {
                    Some(v) => {
                        w.write_i32(v);
                        w.write_i32(status::OK);
                    }
                    None => {
                        w.write_i32(0);
                        w.write_i32(status::BAD_VALUE);
                    }
                }
            }
            // s32 listener present, s32 api, s32 producer controlled by app -> output, s32 status
            code::CONNECT => {
                let listener = r.read_i32().ok_or("connect: short parcel")?;
                let _api = r.read_i32().ok_or("connect: short parcel")?;
                let _controlled = r.read_i32().ok_or("connect: short parcel")?;
                if listener != 0 {
                    return Err("connect with a listener binder object".into());
                }
                self.output(&mut w);
                if self.connected {
                    w.write_i32(status::BAD_VALUE);
                } else {
                    self.connected = true;
                    w.write_i32(status::OK);
                }
            }
            // s32 api -> s32 status
            code::DISCONNECT => {
                let _api = r.read_i32().ok_or("disconnect without an api")?;
                if self.connected {
                    self.connected = false;
                    for s in &mut self.slots {
                        if s.state == SlotState::Dequeued {
                            s.state = SlotState::Free;
                        }
                    }
                    w.write_i32(status::OK);
                } else {
                    w.write_i32(status::BAD_VALUE);
                }
            }
            // s32 slot, s32 has buffer, flattened graphic buffer -> (nothing)
            code::SET_PREALLOCATED_BUFFER => {
                let slot = r.read_i32().ok_or("setPreallocatedBuffer without a slot")?;
                let has = r.read_i32().ok_or("setPreallocatedBuffer: short parcel")?;
                let buffer = if has != 0 {
                    let flat = r
                        .read_flattened()
                        .ok_or("setPreallocatedBuffer: no buffer")?;
                    Some(
                        GraphicBuffer::parse(flat)
                            .ok_or("setPreallocatedBuffer: not a graphic buffer")?,
                    )
                } else {
                    None
                };
                if let Some(b) = &buffer {
                    self.width = b.width;
                    self.height = b.height;
                }
                if let Some(s) = self.slot(slot) {
                    s.buffer = buffer;
                    s.state = SlotState::Free;
                    s.requested = false;
                } else {
                    return Err(format!("setPreallocatedBuffer into slot {slot}"));
                }
            }
            other => {
                return Err(format!(
                    "buffer queue transaction {other} (the interface has {} known codes)",
                    14
                ))
            }
        }
        Ok(w)
    }

    /// The consumer takes the buffer: it is shown, and the one shown before is released.
    fn present(&mut self, slot: usize, input: QueueInput) {
        if let Some(prev) = self.front.replace(slot) {
            self.slots[prev].state = SlotState::Free;
        }
        self.frames += 1;
        let number = self.frames;
        let s = &mut self.slots[slot];
        s.state = SlotState::Acquired;
        s.last_frame = number;
        let buffer = s.buffer.clone().expect("queued slots have buffers");
        self.recent.push_back((slot, buffer.nvmap_id));
        if self.recent.len() > 8 {
            self.recent.pop_front();
        }
        self.last_frame = Some(Frame {
            number,
            slot,
            buffer,
            input,
        });
    }
}

/// A display of the console.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Display {
    pub name: String,
    pub width: u32,
    pub height: u32,
}

/// A window a program draws into.
#[derive(Debug)]
pub struct Layer {
    pub display: u64,
    /// The id the program passes to the binder interface.
    pub binder: i32,
    pub queue: BufferQueue,
    /// Set while a buffer is free: the event `GetNativeHandle` hands out.
    pub buffer_event: EventToken,
    buffer_event_state: bool,
    opened: bool,
}

/// `DisplayInfo` as `ListDisplays` returns it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayInfo {
    pub name: String,
    pub layer_count_max: u64,
    pub width_max: u64,
    pub height_max: u64,
}

#[derive(Debug)]
pub struct Compositor {
    displays: BTreeMap<u64, Display>,
    layers: BTreeMap<u64, Layer>,
    next_display: u64,
    next_layer: u64,
    next_binder: i32,
    vsync: BTreeMap<u64, EventToken>,
    events: Events,
    trace: bool,
}

impl Default for Compositor {
    fn default() -> Compositor {
        Compositor::new()
    }
}

impl Compositor {
    pub fn new() -> Compositor {
        Compositor {
            displays: BTreeMap::new(),
            layers: BTreeMap::new(),
            next_display: 1,
            next_layer: 1,
            next_binder: 1,
            vsync: BTreeMap::new(),
            events: Events::new(),
            trace: std::env::var_os("NX_TRACE_GPU").is_some(),
        }
    }

    /// `ListDisplays`: the one display there is.
    pub fn list_displays(&self) -> Vec<DisplayInfo> {
        vec![DisplayInfo {
            name: "Default".into(),
            layer_count_max: 1,
            width_max: 1920,
            height_max: 1080,
        }]
    }

    /// `OpenDisplay`: only "Default" exists.
    pub fn open_display(&mut self, name: &str) -> Option<u64> {
        if name != "Default" {
            return None;
        }
        if let Some((&id, _)) = self.displays.iter().find(|(_, d)| d.name == name) {
            return Some(id);
        }
        let id = self.next_display;
        self.next_display += 1;
        self.displays.insert(
            id,
            Display {
                name: name.into(),
                width: DEFAULT_WIDTH,
                height: DEFAULT_HEIGHT,
            },
        );
        Some(id)
    }

    pub fn close_display(&mut self, id: u64) -> bool {
        self.displays.remove(&id).is_some()
    }

    pub fn display_resolution(&self, id: u64) -> Option<(u32, u32)> {
        self.displays.get(&id).map(|d| (d.width, d.height))
    }

    pub fn display(&self, id: u64) -> Option<&Display> {
        self.displays.get(&id)
    }

    /// A layer that exists but nobody has opened yet (`CreateManagedDisplayLayer`).
    pub fn create_layer(&mut self, display: u64) -> u64 {
        let id = self.next_layer;
        self.next_layer += 1;
        let binder = self.next_binder;
        self.next_binder += 1;
        let buffer_event = self.events.new_token();
        self.layers.insert(
            id,
            Layer {
                display,
                binder,
                queue: BufferQueue::new(),
                buffer_event,
                buffer_event_state: false,
                opened: false,
            },
        );
        id
    }

    /// `OpenLayer`: the native window parcel naming the layer's producer.
    pub fn open_layer(&mut self, layer: u64, pid: u32) -> Option<Vec<u8>> {
        let l = self.layers.get_mut(&layer)?;
        l.opened = true;
        Some(native_window(l.binder, pid))
    }

    pub fn close_layer(&mut self, layer: u64) -> bool {
        self.layers.remove(&layer).is_some()
    }

    pub fn layer(&self, layer: u64) -> Option<&Layer> {
        self.layers.get(&layer)
    }

    fn layer_by_binder(&mut self, binder: i32) -> Option<&mut Layer> {
        self.layers.values_mut().find(|l| l.binder == binder)
    }

    /// `GetNativeHandle`: the event that is set while the layer has a free buffer.
    pub fn buffer_event(&mut self, binder: i32) -> Option<EventToken> {
        self.layer_by_binder(binder).map(|l| l.buffer_event)
    }

    /// `GetDisplayVsyncEvent`.
    pub fn vsync_event(&mut self, display: u64) -> Option<EventToken> {
        self.displays.get(&display)?;
        if let Some(t) = self.vsync.get(&display) {
            return Some(*t);
        }
        let t = self.events.new_token();
        self.vsync.insert(display, t);
        Some(t)
    }

    /// The tokens of the vsync events handed out, to tick.
    pub fn vsync_tokens(&self) -> Vec<EventToken> {
        self.vsync.values().copied().collect()
    }

    /// `IHOSBinderDriver::TransactParcel`: `data` is the parcel, the answer is the reply parcel.
    pub fn transact(&mut self, binder: i32, code: u32, data: &[u8]) -> Result<Vec<u8>, String> {
        let trace = self.trace;
        let layer = self
            .layer_by_binder(binder)
            .ok_or_else(|| format!("a transaction on binder {binder}, which names no layer"))?;
        let mut r = ParcelReader::parse(data).ok_or("a transaction parcel with a bad header")?;
        match r.read_interface_token() {
            Some(t) if t == INTERFACE => {}
            other => return Err(format!("a transaction for interface {other:?}")),
        }
        let frames_before = layer.queue.frames();
        let reply = layer.queue.transact(code, &mut r)?;
        if trace {
            eprintln!(
                "[display] binder {binder} transaction {code} -> {} reply bytes{}",
                reply.len(),
                if layer.queue.frames() != frames_before {
                    format!(", frame {} presented", layer.queue.frames())
                } else {
                    String::new()
                }
            );
        }
        // The buffer event follows whether a buffer is free.
        let free = layer.queue.has_free_buffer();
        if free != layer.buffer_event_state {
            layer.buffer_event_state = free;
            let token = layer.buffer_event;
            self.events.set(token, free);
        }
        Ok(reply.finish())
    }

    /// The last frame shown on the layer behind `binder`.
    pub fn last_frame_of(&self, binder: i32) -> Option<Frame> {
        self.layers
            .values()
            .find(|l| l.binder == binder)
            .and_then(|l| l.queue.last_frame().cloned())
    }

    /// The frames presented on all layers.
    pub fn frames_presented(&self) -> u64 {
        self.layers.values().map(|l| l.queue.frames()).sum()
    }

    pub fn take_signals(&mut self) -> Vec<Signal> {
        self.events.take()
    }

    /// Set the vsync event of every display that handed one out.
    pub fn tick_vsync(&mut self) {
        let tokens = self.vsync_tokens();
        for t in tokens {
            self.events.set(t, true);
        }
    }
}

/// The native window of a layer: a flattened binder object naming the producer, and the name of
/// the driver ("dispdrv") the program reaches it through (Switchbrew, "Display services").
pub fn native_window(binder: i32, pid: u32) -> Vec<u8> {
    let mut w = ParcelWriter::new();
    // A 24-byte flat_binder_object: type, flags (the process), handle (the binder id), cookie.
    w.write_u32(2);
    w.write_u32(pid);
    w.write_i32(binder);
    w.write_bytes(&[0u8; 12]);
    w.write_bytes(b"dispdrv\0");
    w.write_bytes(&[0u8; 8]);
    w.add_object(0);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parcel::HEADER;

    /// A flattened `GraphicBuffer` the way `libnx` writes one: the Switch's native handle holds
    /// the nvmap id as its second word.
    fn graphic_buffer(nvmap_id: u32, w: u32, h: u32) -> Vec<u8> {
        let mut ints = vec![0u32; 51];
        ints[0] = u32::MAX;
        ints[1] = nvmap_id;
        ints[3] = 0xDAFF_CAFF;
        let mut b = Vec::new();
        for v in [
            GRAPHIC_BUFFER_MAGIC,
            w,
            h,
            w,
            1,
            0xB00,
            42,
            0,
            0,
            ints.len() as u32,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in ints {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn token_parcel(f: impl FnOnce(&mut ParcelWriter)) -> Vec<u8> {
        let mut w = ParcelWriter::new();
        w.write_interface_token(INTERFACE);
        f(&mut w);
        w.finish()
    }

    fn input_blob() -> Vec<u8> {
        let mut b = vec![0u8; 84];
        b[44..48].copy_from_slice(&1u32.to_le_bytes()); // swap interval
        b
    }

    struct Window {
        c: Compositor,
        binder: i32,
    }

    impl Window {
        /// A layer with `n` preallocated buffers, connected.
        fn new(n: usize) -> Window {
            let mut c = Compositor::new();
            let display = c.open_display("Default").unwrap();
            let layer = c.create_layer(display);
            let binder = c.layer(layer).unwrap().binder;
            let mut win = Window { c, binder };
            win.call(
                code::CONNECT,
                token_parcel(|w| {
                    w.write_i32(0);
                    w.write_i32(2);
                    w.write_i32(0);
                }),
            );
            for i in 0..n {
                win.call(
                    code::SET_PREALLOCATED_BUFFER,
                    token_parcel(|w| {
                        w.write_i32(i as i32);
                        w.write_i32(1);
                        w.write_flattened(&graphic_buffer(10 + i as u32, 1280, 720));
                    }),
                );
            }
            win
        }

        fn call(&mut self, code: u32, parcel: Vec<u8>) -> Vec<u8> {
            self.c.transact(self.binder, code, &parcel).unwrap()
        }

        fn dequeue(&mut self) -> (i32, i32) {
            let r = self.call(
                code::DEQUEUE_BUFFER,
                token_parcel(|w| {
                    w.write_i32(1);
                    w.write_u32(1280);
                    w.write_u32(720);
                    w.write_i32(1);
                    w.write_u32(0xB00);
                }),
            );
            let mut r = ParcelReader::parse(&r).unwrap();
            let slot = r.read_i32().unwrap();
            let has_fence = r.read_i32().unwrap();
            if has_fence != 0 {
                assert_eq!(r.read_flattened().unwrap().len(), 36);
            }
            (slot, r.read_i32().unwrap())
        }

        fn queue(&mut self, slot: i32) -> i32 {
            let r = self.call(
                code::QUEUE_BUFFER,
                token_parcel(|w| {
                    w.write_i32(slot);
                    w.write_flattened(&input_blob());
                }),
            );
            let mut r = ParcelReader::parse(&r).unwrap();
            assert_eq!(r.read_u32(), Some(DEFAULT_WIDTH));
            assert_eq!(r.read_u32(), Some(DEFAULT_HEIGHT));
            r.read_u32();
            r.read_u32();
            r.read_i32().unwrap()
        }
    }

    #[test]
    fn the_native_window_names_the_binder_the_way_switchbrew_documents() {
        let p = native_window(7, 0x33);
        let word = |i: usize| u32::from_le_bytes(p[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (word(0), word(1), word(2), word(3)),
            (0x28, 16, 4, 16 + 0x28)
        );
        let data = &p[HEADER..HEADER + 0x28];
        assert_eq!(u32::from_le_bytes(data[0..4].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 0x33);
        assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()), 7);
        assert!(data[12..24].iter().all(|&b| b == 0));
        assert_eq!(&data[24..32], b"dispdrv\0");
        assert_eq!(p.len(), 16 + 0x28 + 4);
        // libnx reads the binder id as the third word of the data.
        let r = ParcelReader::parse(&p).unwrap();
        let mut r = r;
        r.read_u32();
        r.read_u32();
        assert_eq!(r.read_i32(), Some(7));
    }

    #[test]
    fn graphic_buffers_parse_and_report_their_nvmap_id() {
        let g = GraphicBuffer::parse(&graphic_buffer(77, 1280, 720)).unwrap();
        assert_eq!(
            (g.width, g.height, g.stride, g.format),
            (1280, 720, 1280, 1)
        );
        assert_eq!(g.nvmap_id, 77);
        assert_eq!(g.ints.len(), 51);
        assert!(GraphicBuffer::parse(&[0; 40]).is_none());
        let mut bad = graphic_buffer(1, 4, 4);
        bad[32] = 1; // a descriptor
        assert!(GraphicBuffer::parse(&bad).is_none());
        assert!(GraphicBuffer::parse(&graphic_buffer(1, 4, 4)[..60]).is_none());
    }

    #[test]
    fn queue_input_parses_the_switch_layout() {
        let mut b = vec![0u8; 84];
        b[0..8].copy_from_slice(&(-5i64).to_le_bytes());
        b[12..16].copy_from_slice(&1i32.to_le_bytes());
        b[20..24].copy_from_slice(&1280i32.to_le_bytes());
        b[24..28].copy_from_slice(&720i32.to_le_bytes());
        b[32..36].copy_from_slice(&4u32.to_le_bytes());
        b[44..48].copy_from_slice(&2u32.to_le_bytes());
        b[48..52].copy_from_slice(&2u32.to_le_bytes());
        b[52..60].copy_from_slice(&[9, 0, 0, 0, 5, 0, 0, 0]);
        b[60..68].copy_from_slice(&[3, 0, 0, 0, 1, 0, 0, 0]);
        let q = QueueInput::parse(&b).unwrap();
        assert_eq!(q.timestamp, -5);
        assert_eq!(q.crop, [1, 0, 1280, 720]);
        assert_eq!((q.transform, q.swap_interval), (4, 2));
        assert_eq!(q.fences, vec![(9, 5), (3, 1)]);
        assert!(QueueInput::parse(&b[..80]).is_none());
        b[48] = 5;
        assert!(QueueInput::parse(&b).is_none(), "at most four fences");
    }

    #[test]
    fn a_frame_goes_dequeue_queue_and_the_old_front_is_released() {
        let mut win = Window::new(2);
        assert!(win.c.layer(1).unwrap().queue.has_free_buffer());
        let (a, st) = win.dequeue();
        assert_eq!((a, st), (0, 0));
        assert_eq!(win.queue(a), 0);
        assert_eq!(win.c.frames_presented(), 1);
        // The front buffer stays on screen, so the other one is dequeued next.
        let (b, _) = win.dequeue();
        assert_eq!(b, 1);
        assert_eq!(win.queue(b), 0);
        // Showing the second released the first.
        let (c, _) = win.dequeue();
        assert_eq!(c, 0, "the oldest free buffer comes first");
        let frame = win.c.layer(1).unwrap().queue.last_frame().unwrap().clone();
        assert_eq!(
            (frame.number, frame.slot, frame.buffer.nvmap_id),
            (2, 1, 11)
        );
        assert_eq!(frame.input.swap_interval, 1);
    }

    #[test]
    fn dequeuing_everything_without_queueing_would_block() {
        let mut win = Window::new(2);
        assert_eq!(win.dequeue(), (0, 0));
        assert_eq!(win.dequeue(), (1, 0));
        let (_, status) = win.dequeue();
        assert_eq!(status, status::WOULD_BLOCK);
        // Cancelling gives one back.
        win.call(
            code::CANCEL_BUFFER,
            token_parcel(|w| {
                w.write_i32(1);
                w.write_flattened(&[0u8; 36]);
            }),
        );
        assert_eq!(win.dequeue(), (1, 0));
    }

    #[test]
    fn the_buffer_event_follows_whether_a_buffer_is_free() {
        let mut win = Window::new(2);
        let token = win.c.buffer_event(win.binder).unwrap();
        let sigs = win.c.take_signals();
        assert_eq!(
            sigs.last(),
            Some(&Signal {
                token,
                signaled: true
            })
        );
        let (a, _) = win.dequeue();
        let (b, _) = win.dequeue();
        let sigs = win.c.take_signals();
        assert_eq!(
            sigs.last(),
            Some(&Signal {
                token,
                signaled: false
            }),
            "none left"
        );
        win.queue(a);
        // Showing the first buffer frees nothing yet: it is the front; the second is still out.
        assert!(win.c.take_signals().is_empty());
        win.queue(b);
        assert_eq!(
            win.c.take_signals(),
            vec![Signal {
                token,
                signaled: true
            }],
            "the first was released"
        );
    }

    #[test]
    fn connect_query_and_disconnect() {
        let mut c = Compositor::new();
        let display = c.open_display("Default").unwrap();
        let layer = c.create_layer(display);
        let binder = c.layer(layer).unwrap().binder;
        let connect = |c: &mut Compositor| {
            let p = token_parcel(|w| {
                w.write_i32(0);
                w.write_i32(2);
                w.write_i32(0);
            });
            let r = c.transact(binder, code::CONNECT, &p).unwrap();
            let mut r = ParcelReader::parse(&r).unwrap();
            let size = (r.read_u32().unwrap(), r.read_u32().unwrap());
            r.read_u32();
            r.read_u32();
            (size, r.read_i32().unwrap())
        };
        assert_eq!(connect(&mut c), ((1280, 720), status::OK));
        assert_eq!(connect(&mut c).1, status::BAD_VALUE, "already connected");
        let q = |c: &mut Compositor, what: i32| {
            let p = token_parcel(|w| w.write_i32(what));
            let r = c.transact(binder, code::QUERY, &p).unwrap();
            let mut r = ParcelReader::parse(&r).unwrap();
            (r.read_i32().unwrap(), r.read_i32().unwrap())
        };
        assert_eq!(q(&mut c, 0), (1280, 0));
        assert_eq!(q(&mut c, 1), (720, 0));
        assert_eq!(q(&mut c, 2), (1, 0));
        assert_eq!(q(&mut c, 9), (0, 0));
        assert_eq!(q(&mut c, 99).1, status::BAD_VALUE);
        let p = token_parcel(|w| w.write_i32(2));
        let r = c.transact(binder, code::DISCONNECT, &p).unwrap();
        assert_eq!(ParcelReader::parse(&r).unwrap().read_i32(), Some(0));
        assert_eq!(connect(&mut c).1, status::OK, "reconnect after disconnect");
    }

    #[test]
    fn request_buffer_returns_what_was_preallocated() {
        let mut win = Window::new(1);
        let (slot, _) = win.dequeue();
        let r = win.call(code::REQUEST_BUFFER, token_parcel(|w| w.write_i32(slot)));
        let mut r = ParcelReader::parse(&r).unwrap();
        assert_eq!(r.read_i32(), Some(1));
        assert_eq!(
            r.read_flattened().unwrap(),
            &graphic_buffer(10, 1280, 720)[..]
        );
        assert_eq!(r.read_i32(), Some(0));
        // A slot that was not dequeued is refused.
        let r = win.call(code::REQUEST_BUFFER, token_parcel(|w| w.write_i32(5)));
        let mut r = ParcelReader::parse(&r).unwrap();
        assert_eq!(r.read_i32(), Some(0));
        assert_eq!(r.read_i32(), Some(status::BAD_VALUE));
    }

    #[test]
    fn a_transaction_nobody_implemented_is_reported() {
        let mut win = Window::new(1);
        let p = token_parcel(|_| {});
        let err = win
            .c
            .transact(win.binder, code::DETACH_BUFFER, &p)
            .unwrap_err();
        assert!(err.contains("transaction 4"), "{err}");
        // So is one on a binder that names nothing, or for another interface.
        assert!(win.c.transact(99, code::QUERY, &p).is_err());
        let mut w = ParcelWriter::new();
        w.write_interface_token("android.gui.IGraphicBufferConsumer");
        assert!(win
            .c
            .transact(win.binder, code::QUERY, &w.finish())
            .is_err());
    }

    #[test]
    fn displays_open_by_name_and_hand_out_one_vsync_event() {
        let mut c = Compositor::new();
        assert_eq!(c.open_display("External"), None);
        let d = c.open_display("Default").unwrap();
        assert_eq!(c.open_display("Default"), Some(d), "the same display again");
        assert_eq!(c.display_resolution(d), Some((1280, 720)));
        let v = c.vsync_event(d).unwrap();
        assert_eq!(c.vsync_event(d), Some(v));
        c.tick_vsync();
        assert_eq!(
            c.take_signals(),
            vec![Signal {
                token: v,
                signaled: true
            }]
        );
        assert!(c.close_display(d));
        assert_eq!(c.vsync_event(d), None);
        assert_eq!(c.list_displays()[0].name, "Default");
    }
}
