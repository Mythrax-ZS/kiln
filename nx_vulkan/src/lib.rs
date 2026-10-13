//! nx_vulkan: the Vulkan backend.
//!
//! Runs on a host GPU what the GPU model decoded: the compute harness the shader translator is
//! tested with, and (in later modules) the draws themselves.

pub mod compute;
pub mod device;

pub use device::{Capabilities, Options, Vk};
