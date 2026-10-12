//! recomp: static recompiler from Horizon OS AArch64 programs to native x86_64.
//!
//! The library holds the container layer, loader, analysis, lifter and code
//! generator. The guest CPU model and Horizon emulation layer live in the
//! `nx_runtime` crate, which every produced binary links against.

/// Version string reported by the CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
