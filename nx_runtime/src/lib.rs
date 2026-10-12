//! nx_runtime: guest CPU model and Horizon OS emulation layer.
//!
//! Recompiled programs and the `recomp` tool both link against this crate.
//! It is built as an `rlib` for Rust consumers and as a `staticlib` for the
//! native executables and shared libraries that clang produces.

pub mod error;
pub mod romfs;
pub mod storage;
pub mod util;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use error::{Error, Result};

/// Crate version, used in diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
