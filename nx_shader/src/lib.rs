//! nx_shader: the shaders of the Switch's GPU (NVIDIA Maxwell, SM50 "GM20B") and their translation
//! to SPIR-V.
//!
//! A shader program in GPU memory is a 20-word header (the SPH) followed by instructions of 64 bits,
//! three to a 32-byte bundle that starts with a scheduling word. This crate parses the header,
//! decodes the instructions, finds the control flow, and (in later modules) translates the program.
//! It knows nothing about the GPU model that feeds it: the caller gives it bytes and gets back
//! structures. The encodings are from the NVIDIA Maxwell instruction set as documented by envytools
//! (MIT licence) and Mesa's NAK encoder (MIT licence); everything they leave open is recorded in
//! docs/ASSUMPTIONS.md.

pub mod asm;
pub mod cfg;
pub mod inst;
pub mod ops;
pub mod sph;
pub mod spirv;
pub mod translate;
pub mod validate;
