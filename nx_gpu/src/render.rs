//! The interface between the command processor and a renderer.
//!
//! The processor keeps the register file of the 3D engine and executes the commands that have an
//! effect on memory; a [`Backend`] is told about the commands that would put something on a
//! screen: draws, clears, the end of a submission, and the display of a buffer. It reads the
//! state it needs from the [`Scene`] it is given with each call, so the processor does not
//! interpret the registers a draw uses (a renderer needs a different view of them than a
//! tracer does), and the registers are read at the moment of the draw.

use crate::compute::Dispatch;
use crate::mem::GuestMemory;
use crate::state::{CbufBindings, Regs};
pub use crate::twod::Blit;
use crate::vm::AddressSpace;

/// What a renderer reads the world with: the registers of the 3D engine, the constant buffers the
/// engine has bound, and the GPU address space with the memory behind it.
pub struct Scene<'a> {
    pub regs: Regs<'a>,
    /// The class of the engine the registers are of.
    pub class: u32,
    pub cbufs: &'a CbufBindings,
    pub mem: &'a dyn GuestMemory,
    pub space: &'a AddressSpace,
    /// What changed of the registers and the bindings since a renderer on another thread was
    /// last given them (A-288).
    pub changes: &'a Changes,
}

/// Which registers of an engine, and whether its constant buffer bindings, changed since they
/// were last taken: a renderer on a thread of its own keeps a copy of each engine's state, and is
/// sent only what changed (A-288). Every engine has one; a new one says everything changed.
#[derive(Debug)]
pub struct Changes {
    id: u64,
    words: [std::cell::Cell<u64>; crate::engine::REGISTERS / 64],
    cbufs: std::cell::Cell<bool>,
}

impl Default for Changes {
    fn default() -> Changes {
        Changes::new()
    }
}

impl Changes {
    pub fn new() -> Changes {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Changes {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            words: std::array::from_fn(|_| std::cell::Cell::new(!0)),
            cbufs: std::cell::Cell::new(true),
        }
    }

    /// Which engine this is.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Register `index` (a word index) was written.
    #[inline]
    pub fn register(&self, index: usize) {
        if let Some(w) = self.words.get(index / 64) {
            w.set(w.get() | 1 << (index % 64));
        }
    }

    /// The constant buffer bindings changed.
    pub fn bindings(&self) {
        self.cbufs.set(true);
    }

    /// The registers written since the last call, with their values in `regs`, and whether the
    /// bindings changed; both are forgotten.
    pub fn take(&self, regs: &[u32]) -> (Vec<(u16, u32)>, bool) {
        let mut out = Vec::new();
        for (i, w) in self.words.iter().enumerate() {
            let mut bits = w.replace(0);
            while bits != 0 {
                let index = i * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if let Some(&v) = regs.get(index) {
                    out.push((index as u16, v));
                }
            }
        }
        (out, self.cbufs.replace(false))
    }
}

impl Scene<'_> {
    /// The programs of the draw, as the engine's class gives them.
    pub fn stages(&self) -> Vec<crate::state::StageSlot> {
        self.regs.stages_of(self.class)
    }

    /// Read `out.len()` bytes of GPU memory at `va`. `false` when any of it is not mapped.
    pub fn read(&self, va: u64, out: &mut [u8]) -> bool {
        self.space.read(self.mem, va, out)
    }

    /// Read GPU memory at `va` into the start of `out` as far as it can be read, and say how many
    /// bytes that is (see [`AddressSpace::read_prefix`]).
    pub fn read_prefix(&self, va: u64, out: &mut [u8]) -> usize {
        self.space.read_prefix(self.mem, va, out)
    }

    /// A hash of up to `len` bytes of GPU memory at `va` as they are now, and how many of them
    /// are mapped (see [`AddressSpace::hash`]).
    pub fn hash(&self, va: u64, len: u64) -> (u64, u64) {
        self.space.hash(self.mem, va, len)
    }

    /// Start a new epoch of write tracking, and say its number (0: writes are not tracked); see
    /// [`GuestMemory::begin_write_epoch`].
    pub fn begin_write_epoch(&self) -> u32 {
        self.mem.begin_write_epoch()
    }

    /// Whether none of the `len` bytes of GPU memory at `va` was written since epoch `epoch`
    /// began (see [`AddressSpace::untouched_since`]).
    pub fn untouched_since(&self, va: u64, len: u64, epoch: u32) -> Option<bool> {
        self.space.untouched_since(self.mem, va, len, epoch)
    }

    /// `len` bytes of GPU memory at `va`.
    pub fn read_vec(&self, va: u64, len: usize) -> Option<Vec<u8>> {
        let mut v = vec![0u8; len];
        self.read(va, &mut v).then_some(v)
    }

    /// The address in the guest process that GPU address `va` is backed by, and how many bytes
    /// of the mapping are left from there.
    pub fn host_address(&self, va: u64) -> Option<(u64, u64)> {
        self.space.translate(va)
    }
}

/// Where the vertices of a draw come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Consecutive vertices of the vertex streams, from `first`.
    Arrays { first: u32, count: u32 },
    /// Indices from the index buffer, `count` of them from index number `first`. The size of an
    /// index is the buffer's own unless the draw method fixes it (`size` in bytes).
    Indexed {
        first: u32,
        count: u32,
        size: Option<u32>,
    },
    /// Vertices or indices that were sent in the command stream itself.
    Inline { count: u32 },
    /// A kind of draw the processor does not describe further (`DRAW_AUTO`, `DRAW_ZERO_INDEX`).
    Unsupported { count: u32 },
}

/// A draw call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Draw {
    pub source: Source,
    /// The primitive type (`BEGIN`'s operation, 0 points, 1 lines, 4 triangles, ...).
    pub topology: u32,
    /// The number of the instance this draw is: 0 for the first of a series, counting up with
    /// each `SUBSEQUENT` begin.
    pub instance: u32,
    /// How many instances it is, from `instance` on (at least 1): the next instances of a series
    /// drawn the same way with nothing changed between them are one draw (A-281).
    pub instances: u32,
}

/// The buffers a `CLEAR_SURFACE` clears (its parameter word).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clear {
    pub depth: bool,
    pub stencil: bool,
    pub red: bool,
    pub green: bool,
    pub blue: bool,
    pub alpha: bool,
    /// The color target (0..8) it applies to.
    pub target: u32,
    /// The layer of the target.
    pub layer: u32,
}

impl Clear {
    /// Decode the parameter of `CLEAR_SURFACE`: Z in bit 0, stencil in 1, red 2, green 3, blue 4,
    /// alpha 5, the render target in bits 6..9 and its layer in bits 10..25.
    pub fn from_word(w: u32) -> Clear {
        Clear {
            depth: w & 1 != 0,
            stencil: w & 2 != 0,
            red: w & 4 != 0,
            green: w & 8 != 0,
            blue: w & 16 != 0,
            alpha: w & 32 != 0,
            target: w >> 6 & 0xF,
            layer: w >> 10 & 0xFFFF,
        }
    }
}

/// A buffer the title queued for display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Present {
    /// The address in the guest process of the buffer's first byte.
    pub address: u64,
    pub width: u32,
    pub height: u32,
    /// The distance between rows in bytes for a pitch-linear buffer.
    pub pitch: u32,
    /// The Android pixel format (`PIXEL_FORMAT_*`).
    pub format: u32,
    /// How the buffer is laid out in memory (`NvLayout`: 1 pitch linear, 2 Tegra tiled, 3 block
    /// linear) and, if block linear, the height of a block in GOBs as a power of two.
    pub layout: u32,
    pub block_height_log2: u32,
    /// The part of the buffer to show (left, top, right, bottom): the producer's crop. A
    /// rectangle with no area stands for the whole buffer.
    pub crop: [i32; 4],
    /// The number of this frame (counting from 1).
    pub frame: u64,
    /// When the frame is to reach the screen: one display period after the tick at which the
    /// program queued it (A-181), so that the frames reach the screen as evenly as they were
    /// queued, whatever each took to draw. `None` when the program is not paced.
    pub due: Option<std::time::Instant>,
}

/// Something that turns the commands of the processor into pictures. An error stops the run with
/// its message, unless the processor was told to carry on past failed draws.
pub trait Backend: Send {
    fn draw(&mut self, scene: &Scene<'_>, draw: &Draw) -> Result<(), String>;

    fn clear(&mut self, scene: &Scene<'_>, clear: &Clear) -> Result<(), String>;

    /// The commands of a submission are all processed: finish what was recorded, so that what
    /// the fence that follows promises is true.
    fn finish(&mut self) -> Result<(), String>;

    /// The title queued a buffer for display.
    fn present(&mut self, present: &Present) -> Result<(), String>;

    /// A compute launch. The scene holds the register file of the compute engine (its texture
    /// pools) and the memory; what the program writes is in memory when this returns.
    fn dispatch(&mut self, _scene: &Scene<'_>, _dispatch: &Dispatch) -> Result<(), String> {
        Err("this renderer does not run compute programs".to_string())
    }

    /// A copy made with the 2D engine. The scene holds the register file of that engine; what is
    /// copied is in memory, or in a render target of the renderer, when this returns.
    fn blit(&mut self, _scene: &Scene<'_>, _blit: &Blit) -> Result<(), String> {
        Err("this renderer does not copy surfaces".to_string())
    }

    /// A copy made with the copy engine (A-268): the bytes are in memory when this returns. A
    /// renderer whose render targets stand for memory puts their pixels there first.
    fn dma_copy(&mut self, scene: &Scene<'_>, copy: &crate::dma::Copy) -> Result<(), String> {
        copy.run(scene.mem, scene.space)
    }

    /// The GPU front is about to read or write guest memory itself: what the renderer recorded
    /// that changes memory in place (compute launches it has not waited for) has run and is in
    /// memory first, with the releases that follow it (docs/ASSUMPTIONS.md A-274). `range` (a GPU
    /// address and a length) asks that only of the work that reaches those bytes or releases into
    /// them, for an access that says nothing of what is done (an inline upload, a semaphore
    /// acquire); `None` asks it of everything (a semaphore release, the end of a submission).
    fn settle(
        &mut self,
        _space: &AddressSpace,
        _mem: &dyn GuestMemory,
        _range: Option<(u64, u64)>,
    ) -> Result<(), String> {
        Ok(())
    }

    /// A semaphore release of the GPU front: `data` written at `va` once the work recorded before
    /// it has run (at once, unless the renderer still has launches to wait for, A-274).
    fn write_after(
        &mut self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        va: u64,
        data: &[u8],
    ) -> Result<(), String> {
        if space.write(mem, va, data) {
            Ok(())
        } else {
            Err(format!(
                "a release of {} bytes to GPU address {va:#x}, which is not mapped",
                data.len()
            ))
        }
    }

    /// The word at `va` as it will be once the releases still to be made are made, when only
    /// releases are still to change it (no launch not waited for reaches it): what a semaphore
    /// acquire compares, since the commands after the acquire run after those launches anyway
    /// (A-274). `None` when nothing is still to be released there, or a launch may write it.
    fn released_value(
        &mut self,
        _space: &AddressSpace,
        _mem: &dyn GuestMemory,
        _va: u64,
    ) -> Option<u32> {
        None
    }

    /// A QMD's release (A-210), made once the work recorded before it has run: at once, unless
    /// the renderer still has launches to wait for (A-274).
    fn release(
        &mut self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        release: crate::qmd3::Release,
    ) -> Result<(), String> {
        release.apply(&mut crate::qmd3::GpuMemory { mem, space })
    }

    /// The GPU front writes `data` at `va` itself (a constant buffer word, an inline upload, an
    /// inline compute descriptor): after the commands given to the renderer before it read
    /// memory (A-288).
    fn write(
        &mut self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        va: u64,
        data: &[u8],
    ) -> Result<(), String> {
        if space.write(mem, va, data) {
            Ok(())
        } else {
            Err(format!(
                "a write of {} bytes to GPU address {va:#x}, which is not mapped",
                data.len()
            ))
        }
    }

    /// The GPU front is about to read memory that what it gave the renderer may change (a
    /// semaphore, a macro's read, a compute descriptor): all of that is done first. A renderer
    /// that does everything in the call has nothing to do (A-288).
    fn sync(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Whether the GPU front may read `len` bytes at `va` as memory holds them now, without
    /// waiting: nothing it gave a renderer on another thread writes them (A-288). A renderer that
    /// does everything in the call says no, and the front asks it as it always does.
    fn readable_now(&mut self, _space: &AddressSpace, _va: u64, _len: u64) -> bool {
        false
    }

    /// The word at `va` as the work the GPU front gave a renderer on another thread will leave
    /// it, when the front can tell that without waiting (A-288); `None` otherwise, and always
    /// for a renderer that does everything in the call.
    fn word_now(&mut self, _space: &AddressSpace, _mem: &dyn GuestMemory, _va: u64) -> Option<u32> {
        None
    }
}

#[cfg(test)]
pub mod testing {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// What a [`Recorder`] was told, in order.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Event {
        Draw(Draw),
        Clear(Clear),
        Finish,
        Present(Present),
        Dispatch(Dispatch),
    }

    /// A backend that remembers its calls (and fails on demand).
    #[derive(Clone, Default)]
    pub struct Recorder {
        pub events: Arc<Mutex<Vec<Event>>>,
        pub fail_draws: bool,
    }

    impl Backend for Recorder {
        fn draw(&mut self, _scene: &Scene<'_>, draw: &Draw) -> Result<(), String> {
            self.events.lock().unwrap().push(Event::Draw(*draw));
            if self.fail_draws {
                Err("the recorder refuses draws".to_string())
            } else {
                Ok(())
            }
        }

        fn clear(&mut self, _scene: &Scene<'_>, clear: &Clear) -> Result<(), String> {
            self.events.lock().unwrap().push(Event::Clear(*clear));
            Ok(())
        }

        fn finish(&mut self) -> Result<(), String> {
            self.events.lock().unwrap().push(Event::Finish);
            Ok(())
        }

        fn present(&mut self, present: &Present) -> Result<(), String> {
            self.events.lock().unwrap().push(Event::Present(*present));
            Ok(())
        }

        fn dispatch(&mut self, _scene: &Scene<'_>, dispatch: &Dispatch) -> Result<(), String> {
            self.events
                .lock()
                .unwrap()
                .push(Event::Dispatch(dispatch.clone()));
            Ok(())
        }
    }
}
