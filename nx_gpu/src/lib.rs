//! nx_gpu: the GPU side of the Horizon emulation.
//!
//! The Switch's graphics driver (NVN) is part of the title; it talks to the console's `nvdrv`
//! service with ioctls on a handful of device nodes, and to the display through Android-style
//! binder transactions. This crate models what answers those: the memory handles (`nvmap`), the
//! GPU address spaces, syncpoints and fences, the channels that take command lists, the engines
//! those commands drive, and the window the display consumes.
//!
//! Nothing here knows about Horizon IPC. The Horizon layer in `nx_runtime` turns IPC requests into
//! calls on [`nvdrv::Nvdrv`] and the display types, and gives the GPU access to guest memory through
//! [`mem::GuestMemory`]. Every layout is from Switchbrew, the NVIDIA class headers or `libnx`, and
//! every choice the public documentation leaves open is recorded in docs/ASSUMPTIONS.md.

pub mod channel;
pub mod compute;
pub mod display;
pub mod draw_state;
pub mod engine;
pub mod error;
pub mod event;
pub mod gpu;
pub mod i2m;
pub mod mem;
pub mod methods;
pub mod mme;
pub mod names;
pub mod nvdrv;
pub mod nvmap;
pub mod parcel;
pub mod pushbuf;
pub mod render;
pub mod shaders;
pub mod state;
pub mod syncpt;
pub mod vm;

pub use error::{NvError, NvResult};
pub use mem::GuestMemory;

/// Counters of what the command processor has done, and where it is, for a watchdog that looks at
/// a run from another thread (`NX_WATCHDOG` in `nx_runtime`).
pub mod progress {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    /// Methods written to an engine, from the command stream or a macro.
    pub static METHODS: AtomicU64 = AtomicU64::new(0);
    /// Draws and clears given to the backend.
    pub static COMMANDS: AtomicU64 = AtomicU64::new(0);

    /// A method was written.
    pub(crate) fn method() {
        METHODS.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw or a clear is being given to the backend.
    pub(crate) fn command() {
        COMMANDS.fetch_add(1, Ordering::Relaxed);
    }

    /// What the GPU side is doing (the last thing it said it was doing).
    static PHASE: AtomicU32 = AtomicU32::new(0);

    const PHASES: [&str; 12] = [
        "nothing in particular",
        "running methods and macros",
        "in the backend, drawing",
        "in the backend, clearing",
        "in the backend, running a compute launch",
        "in the backend, showing a frame",
        "submitting commands to the device",
        "waiting for the device",
        "reading a picture back from the device",
        "translating a shader",
        "making a texture (reading, decoding, uploading)",
        "writing render targets back to memory",
    ];

    pub const IDLE: u32 = 0;
    pub const METHODS_RUNNING: u32 = 1;
    pub const DRAWING: u32 = 2;
    pub const CLEARING: u32 = 3;
    pub const DISPATCHING: u32 = 4;
    pub const PRESENTING: u32 = 5;
    pub const SUBMITTING: u32 = 6;
    pub const WAITING_FOR_DEVICE: u32 = 7;
    pub const READING_BACK: u32 = 8;
    pub const TRANSLATING: u32 = 9;
    pub const MAKING_TEXTURE: u32 = 10;
    pub const WRITING_BACK: u32 = 11;

    /// The GPU side is doing `phase` until the guard is dropped, which restores what it was doing.
    pub struct Phase(u32);

    pub fn enter(phase: u32) -> Phase {
        Phase(PHASE.swap(phase, Ordering::Relaxed))
    }

    impl Drop for Phase {
        fn drop(&mut self) {
            PHASE.store(self.0, Ordering::Relaxed);
        }
    }

    /// What the GPU side says it is doing.
    pub fn phase() -> &'static str {
        PHASES
            .get(PHASE.load(Ordering::Relaxed) as usize)
            .copied()
            .unwrap_or("something unnamed")
    }
}
