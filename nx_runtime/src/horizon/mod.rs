//! The Horizon kernel layer: what a guest program sees of the operating system.
//!
//! Supervisor calls ([`svc`]), handles and kernel objects ([`handle`], [`object`]), the process's
//! memory map ([`memory`], [`layout`]), threads and the blocking primitives ([`kernel`]), IPC
//! ([`ipc`]) and the service manager ([`sm`]). Services are Rust objects implementing
//! [`service::Service`]; the kernel exposes only the ones a program's NPDM allows.
//!
//! One [`kernel::Kernel`] is one process. It runs the guest's threads on host threads, each
//! executing generated code through the runtime in [`crate::rt`].

pub mod gamepad;
pub mod handle;
pub mod input;
pub mod ipc;
pub mod kernel;
pub mod layout;
pub mod memobj;
pub mod memory;
pub mod object;
pub mod process;
pub mod result;
pub mod service;
pub mod services;
pub mod shmem;
pub mod sm;
pub mod svc;
#[cfg(test)]
mod tests;
pub mod watchdog;

pub use kernel::{Exit, Kernel, ProcessInfo, ProcessKind};
pub use process::{Segment, StartConfig};
pub use result::ResultCode;
