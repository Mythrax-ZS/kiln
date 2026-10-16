//! Switches for looking at what the renderer does, read once from the environment. With none of
//! them set the renderer behaves as it does without this module.

use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub struct Debug {
    /// `NX_TRACE_GPU` (or `NX_TRACE_PERF`, which leaves out what the command processor prints): say
    /// which device is used and how much was imported, cached and written, and every 100 frames
    /// where the time went.
    pub trace_gpu: bool,
    /// `NX_TRACE_TEXTURES`: the texture handles, headers and samplers of every draw.
    pub trace_textures: bool,
    /// `NX_TRACE_COMPUTE`: how long the device took over each compute launch (by timestamps).
    pub trace_compute: bool,
    /// `NX_TRACE_CBUFS`: the first words of every constant buffer a draw reads.
    pub trace_cbufs: bool,
    /// `NX_TRACE_STATE`: with `NX_TRACE_DRAWS`, the state, indices and vertices of each draw.
    pub trace_state: bool,
    /// `NX_TRACE_DRAWS=<frame>`: list every draw and clear of that frame (counting the buffers
    /// shown so far) with its target, programs and outcome.
    pub trace_frame: Option<u64>,
    /// `NX_TRACE_AFTER_BUSY`: with `NX_DUMP_BUSY`, trace the frame after the busy one as
    /// `NX_TRACE_DRAWS` does.
    pub trace_after_busy: bool,
    /// `NX_DUMP_DRAWS=<dir>`: also write the first color target after each of those commands (the
    /// work is run to the end each time, so this is slow); `NX_DUMP_FROM=<n>` skips the first n.
    pub dump_draws: Option<PathBuf>,
    pub dump_from: u64,
    /// `NX_DUMP_PASSES=<dir>`: with the tracing of a frame, write a picture of each color target
    /// when the draws move on to another one (the picture of one pass of the frame).
    pub dump_passes: Option<PathBuf>,
    /// `NX_DUMP_UNTIL=<n>`: stop at command n (see `NX_DUMP_FROM`).
    pub dump_until: u64,
    /// `NX_PROBE_PIXELS=<x>:<y>[,...]`: with `NX_DUMP_DRAWS`, also print what those pixels of the
    /// first color target hold after each dumped command, as numbers (floats are not clamped).
    pub probe_pixels: Vec<(u32, u32)>,
    /// `NX_PROBE_EXTREME=<limit>`: with `NX_DUMP_DRAWS`, also say how many pixels of the first color
    /// target hold a component that is not a number or is above the limit, and the first few.
    pub probe_extreme: Option<f32>,
    /// `NX_TRACE_DRAWS_TO=<hex GPU address>[,...]`: every draw with a color target at one of
    /// these addresses is described (state included), in whatever frame it comes.
    pub trace_draws_to: Vec<u64>,
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
    /// `NX_TRACE_TARGETS`: say each color target configuration the first time it is drawn to.
    pub trace_targets: bool,
    /// `NX_DEBUG_EQUAL_AS_LEQUAL`: a depth test for equality passes when the depth is less or equal
    /// (does a pass after a depth prepass draw nothing only because the depths differ in a bit?).
    pub equal_as_lequal: bool,
    /// `NX_DEBUG_NO_DEPTH_TEST`: no draw is depth tested (what do they cover when nothing is
    /// rejected for its depth?).
    pub no_depth_test: bool,
    /// `NX_DEBUG_FLIP_FRONT`: the winding that makes a face a front face is the other one.
    pub flip_front: bool,
    /// `NX_DEBUG_SYNC_DRAWS=<n>`: from the first frame with `n` draws on (at once for 1 or no
    /// number), run the commands and wait for them after every draw, so that a draw that loses the
    /// device (a program that never ends, a bad address) is named when it does.
    pub sync_draws: Option<u64>,
    /// `NX_DEBUG_SYNC_FROM=<frame>`: the same from that presented frame on (a loss of the device
    /// that comes at a known frame, after a run that is too long to wait for at every draw).
    pub sync_from: Option<u64>,
    /// `NX_DEBUG_WAIT_FINISH`: wait for the device at the end of every submission of the guest, as
    /// the renderer did before it overlapped the device's work with recording (A-123).
    pub wait_at_finish: bool,
    /// `NX_DEBUG_NO_SAVE_TARGETS`: a render target that is let go has not its pixels put into
    /// memory first (the behavior before docs/ASSUMPTIONS.md A-171).
    pub no_save_targets: bool,
    /// `NX_DEBUG_SKIP_PROGRAMS=<hex address>[,...]`: a draw whose vertex or pixel program is at one
    /// of these addresses is not drawn (does the loss of a device come from those draws?).
    pub skip_programs: Vec<u64>,
    /// `NX_DEBUG_DISCARD_PROGRAMS=<hex address>[,...]`: a draw whose vertex program is at one of
    /// these addresses runs its vertex stage and rasterizes nothing (does the device get lost in
    /// the vertex stage or after it?).
    pub discard_programs: Vec<u64>,
}

impl Debug {
    pub fn from_env() -> Debug {
        let set = |name: &str| std::env::var_os(name).is_some();
        let number = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
        Debug {
            trace_gpu: set("NX_TRACE_GPU") || set("NX_TRACE_PERF"),
            trace_textures: set("NX_TRACE_TEXTURES"),
            trace_cbufs: set("NX_TRACE_CBUFS"),
            trace_compute: set("NX_TRACE_COMPUTE"),
            trace_state: set("NX_TRACE_STATE"),
            trace_frame: number("NX_TRACE_DRAWS"),
            dump_draws: std::env::var_os("NX_DUMP_DRAWS").map(PathBuf::from),
            dump_from: number("NX_DUMP_FROM").unwrap_or(0),
            dump_until: number("NX_DUMP_UNTIL").unwrap_or(u64::MAX),
            trace_draws_to: addresses("NX_TRACE_DRAWS_TO"),
            probe_extreme: std::env::var("NX_PROBE_EXTREME")
                .ok()
                .and_then(|v| v.parse().ok()),
            probe_pixels: std::env::var("NX_PROBE_PIXELS")
                .map(|v| {
                    v.split(',')
                        .filter_map(|p| {
                            let (x, y) = p.split_once(':')?;
                            Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            dump_passes: std::env::var_os("NX_DUMP_PASSES").map(PathBuf::from),
            trace_after_busy: set("NX_TRACE_AFTER_BUSY"),
            dump_targets: set("NX_DUMP_TARGETS"),
            spirv_check: set("NX_SPIRV_CHECK"),
            no_cull: set("NX_DEBUG_NO_CULL"),
            full_viewport: set("NX_DEBUG_FULL_VIEWPORT"),
            solid_pixel: set("NX_DEBUG_SOLID_PIXEL"),
            trace_targets: set("NX_TRACE_TARGETS"),
            equal_as_lequal: set("NX_DEBUG_EQUAL_AS_LEQUAL"),
            no_depth_test: set("NX_DEBUG_NO_DEPTH_TEST"),
            flip_front: set("NX_DEBUG_FLIP_FRONT"),
            sync_draws: set("NX_DEBUG_SYNC_DRAWS")
                .then(|| number("NX_DEBUG_SYNC_DRAWS").unwrap_or(1)),
            sync_from: number("NX_DEBUG_SYNC_FROM"),
            wait_at_finish: set("NX_DEBUG_WAIT_FINISH"),
            no_save_targets: set("NX_DEBUG_NO_SAVE_TARGETS"),
            skip_programs: addresses("NX_DEBUG_SKIP_PROGRAMS"),
            discard_programs: addresses("NX_DEBUG_DISCARD_PROGRAMS"),
        }
    }
}

/// The hexadecimal addresses a switch lists, separated by commas.
pub fn addresses(name: &str) -> Vec<u64> {
    std::env::var(name)
        .map(|v| {
            v.split(',')
                .filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_default()
}
