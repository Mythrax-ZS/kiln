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
pub mod display;
pub mod engine;
pub mod error;
pub mod event;
pub mod gpu;
pub mod mem;
pub mod methods;
pub mod names;
pub mod nvdrv;
pub mod nvmap;
pub mod parcel;
pub mod pushbuf;
pub mod syncpt;
pub mod vm;

pub use error::{NvError, NvResult};
pub use mem::GuestMemory;
