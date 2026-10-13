//! nx_vulkan: the Vulkan backend.
//!
//! Runs on a host GPU what the GPU model decoded: the compute harness the shader translator is
//! tested with, and (in later modules) the draws themselves.

pub mod astc;
pub mod blocklinear;
pub mod compute;
pub mod device;
pub mod fast;
pub mod formats;
pub mod image;
pub mod input;
pub mod png;
pub mod render;
#[cfg(feature = "window")]
pub mod window;

pub use compute::{AddressPatch, SampledImage, SampledKind};
pub use device::{Capabilities, Options, Vk};
pub use input::{InputEvent, InputSink};
pub use render::{Config as RenderConfig, Renderer};
