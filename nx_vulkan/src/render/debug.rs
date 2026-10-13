//! Switches for looking at what the renderer does, read once from the environment. With none of
//! them set the renderer behaves as it does without this module.

use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub struct Debug {
    /// `NX_TRACE_GPU`: say which device is used and how much was imported, cached and written.
    pub trace_gpu: bool,
    /// `NX_TRACE_TEXTURES`: the texture handles, headers and samplers of every draw.
    pub trace_textures: bool,
    /// `NX_TRACE_CBUFS`: the first words of every constant buffer a draw reads.
    pub trace_cbufs: bool,
    /// `NX_TRACE_STATE`: with `NX_TRACE_DRAWS`, the state, indices and vertices of each draw.
    pub trace_state: bool,
    /// `NX_TRACE_DRAWS=<frame>`: list every draw and clear of that frame (counting the buffers
    /// shown so far) with its target, programs and outcome.
    pub trace_frame: Option<u64>,
    /// `NX_DUMP_DRAWS=<dir>`: also write the first color target after each of those commands (the
    /// work is run to the end each time, so this is slow); `NX_DUMP_FROM=<n>` skips the first n.
    pub dump_draws: Option<PathBuf>,
    pub dump_from: u64,
    /// `NX_DUMP_TARGETS`: with `NX_DUMP_FRAMES`, also write every color target at each dump.
    pub dump_targets: bool,
    /// `NX_SPIRV_CHECK`: validate the translated programs with `spirv-val` when it is installed.
    pub spirv_check: bool,
    /// `NX_DEBUG_NO_CULL`: ignore face culling.
    pub no_cull: bool,
    /// `NX_DEBUG_FULL_VIEWPORT`: ignore the registers' viewport and scissor and use the target.
    pub full_viewport: bool,
    /// `NX_DEBUG_SOLID_PIXEL`: every pixel program writes magenta (is a primitive covered at all?).
    pub solid_pixel: bool,
}

impl Debug {
    pub fn from_env() -> Debug {
        let set = |name: &str| std::env::var_os(name).is_some();
        let number = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
        Debug {
            trace_gpu: set("NX_TRACE_GPU"),
            trace_textures: set("NX_TRACE_TEXTURES"),
            trace_cbufs: set("NX_TRACE_CBUFS"),
            trace_state: set("NX_TRACE_STATE"),
            trace_frame: number("NX_TRACE_DRAWS"),
            dump_draws: std::env::var_os("NX_DUMP_DRAWS").map(PathBuf::from),
            dump_from: number("NX_DUMP_FROM").unwrap_or(0),
            dump_targets: set("NX_DUMP_TARGETS"),
            spirv_check: set("NX_SPIRV_CHECK"),
            no_cull: set("NX_DEBUG_NO_CULL"),
            full_viewport: set("NX_DEBUG_FULL_VIEWPORT"),
            solid_pixel: set("NX_DEBUG_SOLID_PIXEL"),
        }
    }
}
