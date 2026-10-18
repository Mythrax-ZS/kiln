//! The native runtime: what recompiled programs link against.
//!
//! - [`abi`] defines the contract with generated code;
//! - [`mem`] is guest memory behind one checked interface and [`monitor`] the exclusive monitor;
//! - [`runtime`] holds the process-wide state, the slow path and the run loop;
//! - [`census`] counts what runs outside the fast paths (`NX_CENSUS`);
//! - [`exports`] is the C ABI the generated code and the host program call.

pub mod abi;
pub mod census;
pub mod exports;
pub mod host;
pub mod mem;
pub mod runtime;
pub mod trap;

#[cfg(test)]
mod tests;

pub use abi::{NativeCpu, EXIT_PC, STOP_PC};
pub use runtime::rt;
