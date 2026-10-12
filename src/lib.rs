//! recomp: static recompiler from Horizon OS AArch64 programs to native x86_64.
//!
//! The library holds the input layer, loader, analysis, lifter and code generator. The guest CPU
//! model and Horizon emulation layer live in the `nx_runtime` crate, which every produced binary
//! links against. Inputs are an already extracted ExeFS and RomFS; no keys are involved.

pub mod cli;
pub mod container;
pub mod loader;

pub use nx_runtime::{error, romfs, storage, util};
pub use nx_runtime::{Error, Result};

/// Version string reported by the CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
