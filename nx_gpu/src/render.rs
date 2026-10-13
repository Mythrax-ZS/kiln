//! The interface between the command processor and a renderer.
//!
//! The processor keeps the register file of the 3D engine and executes the commands that have an
//! effect on memory; a [`Backend`] is told about the commands that would put something on a
//! screen: draws, clears, the end of a submission, and the display of a buffer. It reads the
//! state it needs from the [`Scene`] it is given with each call, so the processor does not
//! interpret the registers a draw uses (a renderer needs a different view of them than a
//! tracer does), and the registers are read at the moment of the draw.

use crate::mem::GuestMemory;
use crate::state::{CbufBindings, Regs};
use crate::vm::AddressSpace;

/// What a renderer reads the world with: the registers of the 3D engine, the constant buffers the
/// engine has bound, and the GPU address space with the memory behind it.
pub struct Scene<'a> {
    pub regs: Regs<'a>,
    pub cbufs: &'a CbufBindings,
    pub mem: &'a dyn GuestMemory,
    pub space: &'a AddressSpace,
}

impl Scene<'_> {
    /// Read `out.len()` bytes of GPU memory at `va`. `false` when any of it is not mapped.
    pub fn read(&self, va: u64, out: &mut [u8]) -> bool {
        self.space.read(self.mem, va, out)
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
    }
}
