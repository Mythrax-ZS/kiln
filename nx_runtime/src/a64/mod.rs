//! The AArch64 guest CPU model: decoder, interpreter and shared CPU state.

pub mod bits;
pub mod bus;
pub mod decode;
#[cfg(test)]
mod decode_tests;
pub mod fp;
#[cfg(all(test, target_arch = "x86_64"))]
mod fp_tests;
pub mod inst;
pub mod interp;
#[cfg(test)]
mod interp_tests;
pub mod state;

pub use bus::{Bus, Fault, FaultKind};
pub use decode::decode;
pub use inst::Inst;
pub use interp::{execute, step, Trap};
pub use state::CpuState;
