//! Translation of a Maxwell shader program to SPIR-V.
//!
//! Every thread of the console's GPU runs the program on its own; so does every invocation of a
//! SPIR-V shader. The translation keeps that model: the 255 general registers and 7 predicates a
//! program touches become function-local variables, each instruction becomes a few SPIR-V
//! operations on them, and the blocks of the control-flow graph become the cases of a loop over a
//! program counter (a "dispatch loop") unless the program is straight-line code.
//!
//! Resources are bound as follows, so the host can build the pipeline layout from [`Info`]:
//!
//! * descriptor set 0 only;
//! * binding `64 * stage + k` for constant buffer slot `k` (0..18): a read-only storage buffer of
//!   32-bit words holding the buffer's bytes;
//! * binding `64 * stage + 20`: the global memory map (see [`Info::global_memory`]);
//! * binding `64 * stage + 24 + t` for the texture the program names with index `t` (0..40): a
//!   combined image sampler;
//! * compute tests (the harness mode) use bindings 100 and up, listed in [`test_bindings`].
//!
//! `stage` is [`Stage::index`]: vertex 0, tessellation init 1, tessellation 2, geometry 3, pixel 4,
//! compute 5.

pub(crate) mod alu;
mod emit;
mod flow;
mod global;
mod io;
mod tex;

pub use emit::translate;

use crate::sph::Sph;

/// The stage a program is translated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    Pixel,
    /// The harness mode: one invocation per element, registers in and out through buffers.
    Compute,
}

impl Stage {
    pub fn index(self) -> u32 {
        match self {
            Stage::Vertex => 0,
            Stage::Pixel => 4,
            Stage::Compute => 5,
        }
    }
}

/// The numeric class of a vertex attribute or a render target as the shader sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Class {
    #[default]
    Float,
    Sint,
    Uint,
}

/// Everything besides the program that the translation depends on.
#[derive(Clone, Debug)]
pub struct Config {
    pub stage: Stage,
    /// Vertex stage: the class of each vertex attribute location (inputs are `vec4` of that class).
    pub attribute_classes: [Class; 32],
    /// Pixel stage: the class of each color target.
    pub target_classes: [Class; 8],
    /// Vertex stage: clip space depth runs from -w to w (OpenGL) and is moved to 0 to w.
    pub depth_minus_one_to_one: bool,
    /// Pixel stage: the position's y counts from the bottom of the target (OpenGL window origin);
    /// the value read is `target_height - 1 - y`... see [`Config::flip_y_height`].
    pub flip_y_height: Option<f32>,
    /// Write a human-readable name for every register variable and block (for `spirv-dis`).
    pub debug_names: bool,
}

impl Config {
    pub fn new(stage: Stage) -> Config {
        Config {
            stage,
            attribute_classes: [Class::Float; 32],
            target_classes: [Class::Float; 8],
            depth_minus_one_to_one: true,
            flip_y_height: None,
            debug_names: false,
        }
    }
}

/// A texture a program samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureUse {
    /// The index in the instruction (the slot of the constant buffer table of handles).
    pub index: u32,
    pub dim: Dim,
    pub arrayed: bool,
    /// Sampled with a depth reference (a shadow sampler).
    pub shadow: bool,
}

/// The dimensionality of a sampled image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    D1,
    D2,
    D3,
    Cube,
}

/// What a translated program needs from the host.
#[derive(Clone, Debug, Default)]
pub struct Info {
    /// Per constant buffer slot: one past the highest byte offset the program reads with a fixed
    /// address (0 when unused), and whether it indexes the buffer with a register.
    pub cbuf_extent: [u32; 18],
    pub cbuf_dynamic: [bool; 18],
    pub textures: Vec<TextureUse>,
    /// The program reads or writes global memory. Binding `64 * stage + 20` is then a storage
    /// buffer of 32-bit words: word 0 the number of entries, word 1 reserved (zero), words 2 and
    /// 3 the device address (low, high) of a zeroed scratch page used for addresses no entry
    /// covers, then from word 4 six words per entry: the first GPU address (low, high), one past
    /// the last (low, high) and the device address of the first byte (low, high).
    pub global_memory: bool,
    /// Vertex stage: which components (bit `c` of entry `i`) of attribute location `i` are read.
    pub attributes_read: [u8; 32],
    /// Output attribute locations written (bit `c` of entry `i` for component `c`).
    pub outputs_written: [u8; 32],
    /// Pixel stage: the render targets written (bit `t`).
    pub targets_written: u8,
    pub writes_depth: bool,
    pub writes_point_size: bool,
    pub writes_layer: bool,
    pub writes_viewport_index: bool,
    pub clip_distances: u8,
    /// The program can discard fragments.
    pub kills: bool,
    /// The program has more than one block (it uses the dispatch loop).
    pub dispatch_loop: bool,
    /// Instructions translated.
    pub instructions: u32,
}

/// The descriptor bindings the compute harness mode uses.
pub mod test_bindings {
    /// `uint[]`: 16 registers and one word of predicate bits per invocation, on entry.
    pub const REGS_IN: u32 = 100;
    /// The same layout, written when the program ends.
    pub const REGS_OUT: u32 = 101;
    /// `uint[]`: attribute words read with `ALD`/`IPA`, 64 per invocation (offset / 4).
    pub const ATTR_IN: u32 = 102;
    /// `uint[]`: attribute words written with `AST`, 64 per invocation.
    pub const ATTR_OUT: u32 = 103;
    /// Registers per invocation in the in/out buffers (the 16th word holds the predicates).
    pub const REGS: u32 = 16;
    /// Attribute words per invocation.
    pub const ATTRS: u32 = 64;
}

/// A translation that could not be done: what, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub message: String,
    /// The byte offset of the instruction in the code, when it is about one.
    pub at: Option<u32>,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.at {
            Some(pc) => write!(f, "{} (at code offset {pc:#x})", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for Error {}

/// The translated module and what it needs.
#[derive(Debug)]
pub struct Translation {
    pub spirv: Vec<u32>,
    pub info: Info,
}

/// The header of a program is needed for graphics stages; the compute harness has none.
pub type Header<'a> = Option<&'a Sph>;
