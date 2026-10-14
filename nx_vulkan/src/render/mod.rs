//! The renderer: what `nx_gpu`'s command processor tells a backend, done on a Vulkan device.
//!
//! The device and everything on it are made when the first draw, clear or displayed buffer
//! arrives, so a program that never draws does not pay for them. The commands of a submission are
//! recorded and run when the submission ends ([`Backend::finish`]), after which the guest may
//! rely on them having happened.

#[cfg(feature = "window")]
mod blit;
mod compute;
mod convert;
mod debug;
#[cfg(feature = "window")]
mod display;
mod draw;
mod frame;
mod global;
mod shaders;
mod targets;
mod textures;

use crate::device::Vk;
use crate::formats::{color_format, depth_format};
use ash::vk;
use debug::Debug;
use frame::Frame;
use nx_gpu::compute::Dispatch;
use nx_gpu::draw_state::Layout;
use nx_gpu::render::{Backend, Clear, Draw, Present, Scene};
use std::path::PathBuf;
use targets::{Key, Targets, Tiling};

/// What the renderer is asked to do besides rendering.
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// Write each displayed frame as a PNG into this directory (`NX_DUMP_FRAMES`).
    pub dump_dir: Option<PathBuf>,
    /// Dump every n-th frame only (`NX_DUMP_EVERY`, default 1).
    pub dump_every: u64,
    /// Stop dumping after this many pictures (`NX_DUMP_COUNT`).
    pub dump_limit: Option<u64>,
    /// Dump the first frame that has this many draws or more, with every render target
    /// (`NX_DUMP_BUSY`): finds a busy scene whatever frame number it comes at.
    pub dump_busy: Option<u64>,
    /// Show the displayed buffers in a window of this size in pixels (`NX_WINDOW`, with
    /// `NX_WINDOW_SIZE=<width>x<height>`, 1280x720 by default). Needs the `window` feature.
    pub window: Option<(u32, u32)>,
    /// The title of the window (`NX_WINDOW_TITLE`).
    pub window_title: String,
    /// Told what the person does in the window: keys and the keyboard focus.
    pub input: Option<crate::InputSink>,
}

impl Config {
    pub fn from_env() -> Config {
        let number = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
        let window = std::env::var("NX_WINDOW").is_ok_and(|v| v != "0").then(|| {
            let size = std::env::var("NX_WINDOW_SIZE").ok().and_then(|v| {
                let (w, h) = v.split_once('x')?;
                Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))
            });
            size.filter(|(w, h)| *w > 0 && *h > 0)
                .unwrap_or((1280, 720))
        });
        Config {
            dump_dir: std::env::var_os("NX_DUMP_FRAMES").map(PathBuf::from),
            dump_every: number("NX_DUMP_EVERY").unwrap_or(1).max(1),
            dump_limit: number("NX_DUMP_COUNT"),
            dump_busy: number("NX_DUMP_BUSY"),
            window,
            window_title: std::env::var("NX_WINDOW_TITLE").unwrap_or_else(|_| "recomp".to_string()),
            input: None,
        }
    }

    /// Open the window this configuration asks for (if it asks for one) now, rather than when the
    /// renderer first has something to draw, which a game gets to seconds after it starts. The
    /// renderer takes the window that is open when it comes to make its device.
    pub fn open_window_early(&self) {
        #[cfg(feature = "window")]
        if let Some((width, height)) = self.window {
            if let Err(e) =
                crate::window::open_early(&self.window_title, width, height, self.input.clone())
            {
                eprintln!("[vulkan] no window: {e}");
            }
        }
    }
}

/// The attachments of the rendering in progress.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pass {
    colors: Vec<Key>,
    depth: Option<Key>,
    extent: (u32, u32),
}

struct Core {
    vk: Vk,
    frame: Frame,
    targets: Targets,
    shaders: shaders::Shaders,
    textures: textures::Textures,
    global: global::GlobalMemory,
    layouts: shaders::Layouts,
    pipelines: draw::Pipelines,
    compute_pipelines: compute::ComputePipelines,
    /// The attachments of the rendering that is begun in the command buffer, if any.
    pass: Option<Pass>,
    dumped: u64,
    debug: Debug,
    /// Commands of the traced frame so far (`NX_TRACE_DRAWS`).
    traced_commands: u64,
    /// The render target configurations drawn to so far (`NX_TRACE_TARGETS`).
    seen_targets: std::collections::HashSet<String>,
    /// The color target (host address, width, height) the traced frame was drawing to.
    pass_target: Option<(u64, u32, u32)>,
    /// Where the draws spent their time since the last report.
    phases: Phases,
    /// Vectors a draw fills and empties, kept so that their memory is not asked for again every
    /// draw.
    scratch: Scratch,
    /// Draws wait for the device (`NX_DEBUG_SYNC_DRAWS`).
    sync_on: bool,
    /// A program that reads or writes guest memory in place was recorded since the device was
    /// last waited for: the guest may take a submission as over when its fence is reached, so
    /// the end of the submission waits (A-123).
    global_pending: bool,
    /// Draws recorded since the last batch was sent to the device.
    batch_draws: u32,
    /// What the textures of the draw being recorded are, kept while draws wait for the device so
    /// that the one that loses it can say.
    last_textures: Vec<String>,
    /// The textures that made targets be written back since the last report: the address of the
    /// texture, how often, and how many targets the last time.
    write_back_causes: std::collections::HashMap<u64, (u64, usize, String)>,
    /// The window frames are shown in, when there is one.
    #[cfg(feature = "window")]
    display: Option<display::Display>,
}

/// How many draws a batch has at least when the end of a pass sends it to the device.
const BATCH_DRAWS: u32 = 24;

/// What a draw keeps between draws: the memory of vectors that are empty when it begins.
#[derive(Default)]
struct Scratch {
    image_infos: Vec<(u32, vk::DescriptorImageInfo)>,
    handles: Vec<(u32, draw::HandleFrom)>,
    buffer_infos: Vec<(u32, vk::DescriptorBufferInfo)>,
    attributes: Vec<(u32, u32, vk::Format, u32)>,
    streams: Vec<u32>,
    bound: Vec<(u32, vk::Buffer, u64)>,
    binding_state: Vec<(u32, u32, bool)>,
}

/// The time the draws spent in each of their steps (reported with `NX_TRACE_GPU`).
#[derive(Clone, Copy, Debug, Default)]
struct Phases {
    /// Finding the translated programs (comparing them with memory).
    programs: std::time::Duration,
    /// The texture handles, headers, images and samplers.
    textures: std::time::Duration,
    /// Beginning the pass (the attachments the registers name).
    pass: std::time::Duration,
    /// The vertex attributes, indices and vertex buffers.
    vertices: std::time::Duration,
    /// The constant buffers and the descriptor set.
    descriptors: std::time::Duration,
    /// The pipeline.
    pipeline: std::time::Duration,
    /// Inside the textures: the handles and headers, finding out whether the texture is a
    /// render target, writing targets back, and making or finding the image and sampler.
    texture_headers: std::time::Duration,
    texture_target: std::time::Duration,
    texture_write_back: std::time::Duration,
    texture_bind: std::time::Duration,
    /// The dynamic state and the draw command.
    record: std::time::Duration,
    /// The steps of compute launches: the program, the constant buffers, the textures, the
    /// pipeline, the recording and sending the work and waiting for it.
    compute: [std::time::Duration; 6],
}

/// Calls of one kind and the time they took.
#[derive(Clone, Copy, Debug, Default)]
struct Lap {
    calls: u64,
    time: std::time::Duration,
}

impl Lap {
    fn add(&mut self, since: Option<std::time::Instant>) {
        if let Some(since) = since {
            self.calls += 1;
            self.time += since.elapsed();
        }
    }
}

/// Whether the time of the steps is measured (`NX_TRACE_GPU`, `NX_TRACE_PERF`). Reading the clock
/// costs tens of nanoseconds and a draw reads it twenty times, so it is read only when a report
/// will say what it found.
static TIMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The time now, if the steps are being measured.
fn stopwatch() -> Option<std::time::Instant> {
    TIMING
        .load(std::sync::atomic::Ordering::Relaxed)
        .then(std::time::Instant::now)
}

/// Where the renderer spent its time since the last report (`NX_TRACE_GPU` reports every 100
/// displayed frames), to tell a slow renderer from a slow program.
#[derive(Clone, Copy, Debug, Default)]
struct Spent {
    draw: Lap,
    /// Draws that failed (and were skipped when the run is lenient).
    failed: u64,
    clear: Lap,
    finish: Lap,
    present: Lap,
    dispatch: Lap,
}

pub struct Renderer {
    config: Config,
    core: Option<Result<Core, String>>,
    spent: Spent,
    /// When the last report was made.
    reported: Option<std::time::Instant>,
    /// Draws asked for since the last displayed buffer, and whether the busy frame was dumped.
    frame_draws: u64,
    busy_dumped: bool,
}

impl Renderer {
    pub fn new(config: Config) -> Renderer {
        Renderer {
            config,
            core: None,
            spent: Spent::default(),
            reported: None,
            frame_draws: 0,
            busy_dumped: false,
        }
    }

    fn core(&mut self) -> Result<&mut Core, String> {
        if self.core.is_none() {
            let debug = Debug::from_env();
            TIMING.store(debug.trace_gpu, std::sync::atomic::Ordering::Relaxed);
            // A window comes first: the device is made with what presenting to it needs.
            #[cfg(feature = "window")]
            let window = self.config.window.and_then(|(w, h)| {
                let (title, input) = (&self.config.window_title, self.config.input.clone());
                match crate::window::open(title, w, h, input) {
                    Ok(window) => Some(window),
                    Err(e) => {
                        eprintln!("[vulkan] no window: {e}");
                        None
                    }
                }
            });
            #[cfg(feature = "window")]
            let options = crate::Options {
                presentation: window.is_some(),
            };
            #[cfg(not(feature = "window"))]
            let options = crate::Options::default();
            let made = Vk::create(options).and_then(|vk| {
                let frame = Frame::new(&vk)?;
                #[cfg(feature = "window")]
                let display = window.and_then(|w| match display::Display::new(&vk, w) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        eprintln!("[vulkan] frames will not be shown in the window: {e}");
                        None
                    }
                });
                if debug.trace_gpu {
                    eprintln!(
                        "[vulkan] {} (Vulkan {}.{})",
                        vk.caps.name,
                        vk::api_version_major(vk.caps.api_version),
                        vk::api_version_minor(vk.caps.api_version)
                    );
                }
                Ok(Core {
                    vk,
                    frame,
                    targets: Targets::default(),
                    shaders: shaders::Shaders::new(&debug),
                    textures: textures::Textures::new(debug.trace_textures),
                    global: global::GlobalMemory::new(debug.trace_gpu),
                    layouts: shaders::Layouts::default(),
                    pipelines: draw::Pipelines::default(),
                    compute_pipelines: compute::ComputePipelines::default(),
                    pass: None,
                    dumped: 0,
                    debug,
                    traced_commands: 0,
                    seen_targets: std::collections::HashSet::new(),
                    pass_target: None,
                    phases: Phases::default(),
                    scratch: Scratch::default(),
                    sync_on: false,
                    global_pending: false,
                    batch_draws: 0,
                    last_textures: Vec::new(),
                    write_back_causes: std::collections::HashMap::new(),
                    #[cfg(feature = "window")]
                    display,
                })
            });
            self.core = Some(made);
        }
        match self.core.as_mut().expect("just made") {
            Ok(core) if core.frame.lost() => Err("the Vulkan device was lost".to_string()),
            Ok(core) => Ok(core),
            Err(e) => Err(format!("no Vulkan device to render with: {e}")),
        }
    }

    /// The color target at a guest address as a picture (RGBA bytes, width, height), as of the
    /// commands recorded so far. For tests and tools.
    pub fn picture_at(&mut self, address: u64) -> Result<Option<(Vec<u8>, u32, u32)>, String> {
        let core = self.core()?;
        core.end_pass();
        let Core {
            vk, frame, targets, ..
        } = core;
        let Some(target) = targets.at_address(address) else {
            return Ok(None);
        };
        let (w, h) = (target.key.width, target.key.height);
        let rgba = targets::read_rgba(vk, frame, target)?;
        Ok(Some((rgba, w, h)))
    }
}

impl Core {
    /// `NX_TRACE_DRAWS=<frame>` lists every draw and clear of that frame (counting the buffers
    /// shown so far); this is that frame while it is the one being recorded.
    fn traced(&self) -> Option<u64> {
        let frame = self.debug.trace_frame?;
        (frame == self.textures.frame).then_some(frame)
    }

    /// Say what a command of the traced frame was; with `NX_DUMP_DRAWS` the first color target is
    /// also written as it is after the command.
    fn trace_command(&mut self, scene: &Scene<'_>, frame: u64, said: &str) {
        let n = self.traced_commands;
        self.traced_commands += 1;
        eprintln!("[vulkan] #{n} {said}");
        if let Some(dir) = self.debug.dump_draws.clone() {
            if n >= self.debug.dump_from && n <= self.debug.dump_until {
                if let Err(e) = self.dump_first_target(scene, &dir, frame, n) {
                    eprintln!("[vulkan] dumping the target after command {n}: {e}");
                }
            }
        }
        if let Some(dir) = self.debug.dump_passes.clone() {
            // The target the draws were on is finished when a command goes to another one.
            let now = scene
                .regs
                .color_targets_in_use()
                .into_iter()
                .next()
                .and_then(|(_, t)| {
                    scene
                        .host_address(t.address)
                        .map(|(a, _)| (a, t.width, t.height))
                });
            if now.map(|t| t.0) != self.pass_target.map(|t| t.0) {
                if let Some((address, width, height)) = self.pass_target.take() {
                    if let Err(e) = self.dump_pass(&dir, frame, n, address, (width, height)) {
                        eprintln!("[vulkan] dumping the pass before command {n}: {e}");
                    }
                }
                self.pass_target = now;
            }
        }
    }

    /// Write the color target at `address` as a picture: the pass of the traced frame that ended
    /// before command `n`.
    fn dump_pass(
        &mut self,
        dir: &std::path::Path,
        frame: u64,
        n: u64,
        address: u64,
        size: (u32, u32),
    ) -> Result<(), String> {
        self.end_pass();
        let Some(target) = self.targets.at_address(address) else {
            return Ok(());
        };
        let (w, h) = (target.key.width, target.key.height);
        let format = target.key.format;
        let rgba = targets::read_rgba(&self.vk, &mut self.frame, target)?;
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(format!(
            "pass_{frame}_before_{n:04}_{address:x}_{}x{}.png",
            size.0, size.1
        ));
        eprintln!(
            "[vulkan]   the pass before command {n}: target {address:#x} {w}x{h} {format:?}: {}",
            crate::png::describe(w, h, &rgba)
        );
        crate::png::write_opaque_png(&path, w, h, &rgba)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Write the first color target the registers name as a picture, as of the commands so far.
    fn dump_first_target(
        &mut self,
        scene: &Scene<'_>,
        dir: &std::path::Path,
        frame: u64,
        n: u64,
    ) -> Result<(), String> {
        let Some((_, t)) = scene.regs.color_targets_in_use().into_iter().next() else {
            return Ok(());
        };
        let Some((address, _)) = scene.host_address(t.address) else {
            return Err(format!("the target at {:#x} is not mapped", t.address));
        };
        self.end_pass();
        let Some(target) = self.targets.at_address(address) else {
            return Ok(());
        };
        let (w, h) = (target.key.width, target.key.height);
        let rgba = targets::read_rgba(&self.vk, &mut self.frame, target)?;
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(format!("draw_{frame}_{n:04}_{address:x}.png"));
        eprintln!(
            "[vulkan]   after command {n}: {}",
            crate::png::describe(w, h, &rgba)
        );
        crate::png::write_opaque_png(&path, w, h, &rgba)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // The depth the draws test against: what it holds, in numbers.
        if let Some(z) = scene.regs.depth_target() {
            if let Some((cpu, _)) = scene.host_address(z.address) {
                match self.targets.depth_at_address(cpu) {
                    Some(target) => match targets::read_depth(&self.vk, &mut self.frame, target) {
                        Ok(depths) => {
                            let finite = depths.iter().copied().filter(|d| d.is_finite());
                            let (lo, hi) = finite
                                .fold((f32::MAX, f32::MIN), |(lo, hi), d| (lo.min(d), hi.max(d)));
                            let below = depths.iter().filter(|d| **d < 1.0).count();
                            let zero = depths.iter().filter(|d| **d == 0.0).count();
                            eprintln!(
                                "[vulkan]   the depth target at {cpu:#x}: from {lo} to {hi}, {below} of {} below 1, {zero} zero",
                                depths.len()
                            );
                        }
                        Err(e) => eprintln!("[vulkan]   the depth target at {cpu:#x}: {e}"),
                    },
                    None => eprintln!("[vulkan]   no depth target was drawn to at {cpu:#x}"),
                }
            }
        }
        Ok(())
    }

    /// Leave the rendering that is begun, if there is one.
    fn end_pass(&mut self) {
        if self.pass.take().is_some() {
            if let Ok(cmd) = self.frame.cmd(&self.vk) {
                // SAFETY: a rendering was begun in this command buffer.
                unsafe { self.vk.device.cmd_end_rendering(cmd) };
            }
        }
    }

    /// The attachments the registers name, begun as the rendering of the command buffer.
    fn begin_pass(&mut self, scene: &Scene<'_>) -> Result<(), String> {
        let (colors, depth) = self.attachments(scene)?;
        let extent = self.extent_of(&colors, depth.as_ref())?;
        let wanted = Pass {
            colors,
            depth,
            extent,
        };
        if self.pass.as_ref() == Some(&wanted) {
            return Ok(());
        }
        self.end_pass();
        // A pass that ends is a good place to give the device what was recorded: it works on that
        // while the next pass is recorded (A-123).
        if self.batch_draws >= BATCH_DRAWS && !self.sync_on {
            self.send_batch()?;
        }
        let cmd = self.frame.cmd(&self.vk)?;
        let mut color_infos = Vec::new();
        for key in &wanted.colors {
            let t = self.targets.get(&self.vk, &mut self.frame, *key)?;
            t.transition(&self.vk, cmd, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            color_infos.push(
                vk::RenderingAttachmentInfo::default()
                    .image_view(t.image.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE),
            );
        }
        let mut depth_info = None;
        if let Some(key) = &wanted.depth {
            let t = self.targets.get(&self.vk, &mut self.frame, *key)?;
            t.transition(
                &self.vk,
                cmd,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            );
            depth_info = Some(
                vk::RenderingAttachmentInfo::default()
                    .image_view(t.image.view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE),
            );
        }
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: wanted.extent.0,
                    height: wanted.extent.1,
                },
            })
            .layer_count(1)
            .color_attachments(&color_infos);
        if let Some(d) = depth_info.as_ref() {
            info = info.depth_attachment(d);
            if wanted.depth.is_some_and(|k| {
                crate::image::aspect_of(k.format).contains(vk::ImageAspectFlags::STENCIL)
            }) {
                info = info.stencil_attachment(d);
            }
        }
        // SAFETY: the command buffer is recording, the attachments are in their layouts.
        unsafe { self.vk.device.cmd_begin_rendering(cmd, &info) };
        self.pass = Some(wanted);
        Ok(())
    }

    /// The keys of the color and depth targets the registers name.
    fn attachments(&mut self, scene: &Scene<'_>) -> Result<(Vec<Key>, Option<Key>), String> {
        let mut colors = Vec::new();
        for (output, t) in scene.regs.color_targets_in_use() {
            let Some(format) = color_format(t.format) else {
                return Err(format!(
                    "color target {output} has format {:#x}, which is not a color format",
                    t.format
                ));
            };
            let Some((address, _)) = scene.host_address(t.address) else {
                return Err(format!(
                    "color target {output} is at GPU address {:#x}, which is not mapped",
                    t.address
                ));
            };
            if self.debug.trace_targets {
                let said = format!(
                    "[vulkan] a color target {output}: {:#x} {}x{} format {:#x} {:?}, {} {} layer {} of pitch {:#x}",
                    t.address,
                    t.width,
                    t.height,
                    t.format,
                    t.layout,
                    t.depth,
                    if t.depth_is_size { "deep" } else { "layers" },
                    t.layer,
                    t.array_pitch
                );
                if self.seen_targets.insert(said.clone()) {
                    eprintln!("{said}");
                }
            }
            let width = match t.layout {
                Layout::Pitch => t.width / format.bytes,
                Layout::BlockLinear { .. } => t.width,
            };
            colors.push(Key {
                address,
                format: format.vk,
                width,
                height: t.height,
                layers: 1,
                tiling: match t.layout {
                    Layout::Pitch => Tiling::Pitch(t.width),
                    Layout::BlockLinear { block } => Tiling::Block(block[1]),
                },
            });
        }
        let depth = match scene.regs.depth_target() {
            None => None,
            Some(z) => {
                let Some(format) = depth_format(z.format) else {
                    return Err(format!("the depth target has format {:#x}", z.format));
                };
                let Some((address, _)) = scene.host_address(z.address) else {
                    return Err(format!(
                        "the depth target is at GPU address {:#x}, which is not mapped",
                        z.address
                    ));
                };
                Some(Key {
                    address,
                    format: format.vk,
                    width: z.width,
                    height: z.height,
                    layers: 1,
                    tiling: match z.layout {
                        Layout::Pitch => Tiling::Pitch(z.width),
                        Layout::BlockLinear { block } => Tiling::Block(block[1]),
                    },
                })
            }
        };
        Ok((colors, depth))
    }

    fn extent_of(&self, colors: &[Key], depth: Option<&Key>) -> Result<(u32, u32), String> {
        let mut sizes = colors.iter().chain(depth).map(|k| (k.width, k.height));
        let Some(first) = sizes.next() else {
            return Err("a draw or clear with no render target".to_string());
        };
        Ok(sizes.fold(first, |(w, h), (w2, h2)| (w.min(w2), h.min(h2))))
    }

    fn clear(&mut self, scene: &Scene<'_>, clear: &Clear) -> Result<(), String> {
        let wants_color = clear.red || clear.green || clear.blue || clear.alpha;
        if wants_color && !(clear.red && clear.green && clear.blue && clear.alpha) {
            return Err(format!(
                "a clear of only some color channels (r {} g {} b {} a {})",
                clear.red, clear.green, clear.blue, clear.alpha
            ));
        }
        if !wants_color && !clear.depth && !clear.stencil {
            return Ok(());
        }
        self.begin_pass(scene)?;
        let pass = self.pass.clone().expect("a pass was just begun");
        let values = scene.regs.clear_values();
        // The area: the clear rectangle, else scissor 0 when asked, else the whole surface.
        let (mut x0, mut x1, mut y0, mut y1) = (0, pass.extent.0, 0, pass.extent.1);
        if values.use_clear_rect {
            (x0, x1, y0, y1) = (
                values.rect[0],
                values.rect[1],
                values.rect[2],
                values.rect[3],
            );
        } else if values.use_scissor0 {
            let s = scene.regs.scissor(0);
            if s.enabled {
                (x0, x1, y0, y1) = (s.x_min, s.x_max, s.y_min, s.y_max);
            }
        }
        let (x1, y1) = (x1.min(pass.extent.0), y1.min(pass.extent.1));
        if x0 >= x1 || y0 >= y1 {
            return Ok(());
        }
        let rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D {
                    x: x0 as i32,
                    y: y0 as i32,
                },
                extent: vk::Extent2D {
                    width: x1 - x0,
                    height: y1 - y0,
                },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        let mut attachments = Vec::new();
        if wants_color {
            let index = clear.target as usize;
            let Some(key) = pass.colors.get(index) else {
                return Err(format!(
                    "a clear of render target {index}, which is not one of the {} bound",
                    pass.colors.len()
                ));
            };
            let class = scene
                .regs
                .color_targets_in_use()
                .get(index)
                .and_then(|(_, t)| color_format(t.format))
                .map(|f| f.class)
                .unwrap_or_default();
            let c = values.color;
            let value = match class {
                nx_shader::translate::Class::Float => vk::ClearColorValue {
                    float32: c.map(f32::from_bits),
                },
                nx_shader::translate::Class::Sint => vk::ClearColorValue {
                    int32: c.map(|w| w as i32),
                },
                nx_shader::translate::Class::Uint => vk::ClearColorValue { uint32: c },
            };
            let _ = key;
            attachments.push(
                vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(index as u32)
                    .clear_value(vk::ClearValue { color: value }),
            );
        }
        if clear.depth || clear.stencil {
            if let Some(key) = &pass.depth {
                let have = crate::image::aspect_of(key.format);
                let mut aspect = vk::ImageAspectFlags::empty();
                if clear.depth && have.contains(vk::ImageAspectFlags::DEPTH) {
                    aspect |= vk::ImageAspectFlags::DEPTH;
                }
                if clear.stencil && have.contains(vk::ImageAspectFlags::STENCIL) {
                    aspect |= vk::ImageAspectFlags::STENCIL;
                }
                if !aspect.is_empty() {
                    attachments.push(
                        vk::ClearAttachment::default()
                            .aspect_mask(aspect)
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: values.depth,
                                    stencil: values.stencil,
                                },
                            }),
                    );
                }
            }
        }
        if attachments.is_empty() {
            return Ok(());
        }
        let cmd = self.frame.cmd(&self.vk)?;
        // SAFETY: a rendering is begun in the recording command buffer.
        unsafe {
            self.vk
                .device
                .cmd_clear_attachments(cmd, &attachments, &[rect]);
        }
        self.frame.mark(&self.vk, cmd, || {
            format!("clear of {} attachments", attachments.len())
        });
        Ok(())
    }

    /// Run what was recorded and wait for it, and for what the device was given before.
    fn flush(&mut self) -> Result<(), String> {
        self.end_pass();
        self.global_pending = false;
        self.batch_draws = 0;
        self.frame.submit(&self.vk)?;
        self.submission_over();
        Ok(())
    }

    /// Count a submission and let go of the targets nothing draws to any more.
    fn submission_over(&mut self) {
        self.targets.submissions += 1;
        let retired = self.targets.retire(&mut self.frame);
        if self.debug.trace_textures {
            for key in retired {
                eprintln!("[vulkan] the render target {key:?} was let go (nothing drew to it for a long time)");
            }
        }
    }

    /// Send what was recorded to the device and go on recording while it runs (A-123).
    fn send_batch(&mut self) -> Result<(), String> {
        self.end_pass();
        self.batch_draws = 0;
        self.frame.kick(&self.vk)
    }

    /// The commands of a submission of the guest are all recorded. They are sent to the device
    /// and not waited for, unless one of them touches guest memory in place (the guest may use
    /// that memory again as soon as its fence is reached) or `NX_DEBUG_WAIT_FINISH` says to.
    fn end_submission(&mut self) -> Result<(), String> {
        if self.global_pending || self.debug.wait_at_finish || self.sync_on {
            return self.flush();
        }
        self.send_batch()?;
        self.submission_over();
        Ok(())
    }
}

impl Renderer {
    fn draw_command(&mut self, scene: &Scene<'_>, draw: &Draw) -> Result<(), String> {
        let core = self.core()?;
        let mut result = core.draw(scene, draw);
        core.batch_draws += 1;
        if core.sync_on && result.is_ok() {
            if let Err(e) = core.flush() {
                // What was submitted up to this draw failed, and this was the last to join it.
                eprintln!("[vulkan] the commands up to this draw failed: {e}");
                let stages: Vec<String> = scene
                    .regs
                    .stages()
                    .iter()
                    .map(|s| format!("{} {:#x}", s.kind.name(), s.address))
                    .collect();
                eprintln!(
                    "[vulkan]     draw {:?} topology {} instance {}; programs {}",
                    draw.source,
                    draw.topology,
                    draw.instance,
                    stages.join(", ")
                );
                for line in describe_draw(scene, draw) {
                    eprintln!("[vulkan]     {line}");
                }
                for line in &core.last_textures {
                    eprintln!("[vulkan]     {line}");
                }
                result = Err(e);
            }
        }
        if let Some(frame) = core.traced() {
            let regs = scene.regs;
            let stages: Vec<String> = regs
                .stages()
                .iter()
                .map(|s| format!("{} {:#x}", s.kind.name(), s.address))
                .collect();
            let said = format!(
                "draw {:?} topology {} instance {} -> {}; {}; {}",
                draw.source,
                draw.topology,
                draw.instance,
                first_target(scene),
                stages.join(", "),
                match &result {
                    Ok(()) => "ok".to_string(),
                    Err(e) => format!("FAILED: {}", e.chars().take(120).collect::<String>()),
                }
            );
            if core.debug.trace_state {
                for line in describe_draw(scene, draw) {
                    eprintln!("[vulkan]     {line}");
                }
            }
            core.trace_command(scene, frame, &said);
        }
        result
    }

    fn clear_command(&mut self, scene: &Scene<'_>, clear: &Clear) -> Result<(), String> {
        let core = self.core()?;
        let result = core.clear(scene, clear);
        if let Some(frame) = core.traced() {
            let v = scene.regs.clear_values();
            let said = format!(
                "clear (color {} depth {} stencil {}) {:?} rect {} {:?} scissor0 {} -> {}; {}",
                clear.red || clear.green || clear.blue || clear.alpha,
                clear.depth,
                clear.stencil,
                v.color.map(f32::from_bits),
                v.use_clear_rect,
                v.rect,
                v.use_scissor0,
                first_target(scene),
                match &result {
                    Ok(()) => "ok".to_string(),
                    Err(e) => format!("FAILED: {}", e.chars().take(120).collect::<String>()),
                }
            );
            core.trace_command(scene, frame, &said);
        }
        result
    }

    fn finish_command(&mut self) -> Result<(), String> {
        match self.core.as_mut() {
            Some(Ok(core)) => core.end_submission(),
            _ => Ok(()),
        }
    }

    /// Show a displayed buffer, and every 100th say where the time of the last 100 went.
    fn present_command(&mut self, p: &Present) -> Result<(), String> {
        let _phase = nx_gpu::progress::enter(nx_gpu::progress::PRESENTING);
        let entered = stopwatch();
        let result = self.present_frame(p);
        self.frame_draws = 0;
        self.spent.present.add(entered);
        if let Some(Ok(core)) = self.core.as_mut() {
            if core.debug.trace_gpu && p.frame.is_multiple_of(100) {
                let now = std::time::Instant::now();
                let wall = self.reported.map(|t| now.duration_since(t));
                self.reported = Some(now);
                let ph = std::mem::take(&mut core.phases);
                eprintln!(
                    "[vulkan] frame {}: the draws spent {:?} on programs, {:?} on textures, {:?} \
                     on the pass, {:?} on vertices, {:?} on descriptors, {:?} on the pipeline, \
                     {:?} on recording; of the textures, {:?} on headers, {:?} on whether a texture \
                     is a target, {:?} on writing targets back, {:?} on images and samplers",
                    p.frame,
                    ph.programs,
                    ph.textures,
                    ph.pass,
                    ph.vertices,
                    ph.descriptors,
                    ph.pipeline,
                    ph.record,
                    ph.texture_headers,
                    ph.texture_target,
                    ph.texture_write_back,
                    ph.texture_bind
                );
                #[cfg(feature = "alloc_stats")]
                {
                    let laps: Vec<String> = crate::alloc_stats::take_laps()
                        .iter()
                        .map(|(name, n)| format!("{name} {n}"))
                        .collect();
                    eprintln!(
                        "[vulkan] frame {}: allocations by the draws' steps over these 100 frames: {}",
                        p.frame,
                        laps.join(", ")
                    );
                }
                let (hashes, hashed) = nx_gpu::vm::hash_counts();
                eprintln!(
                    "[vulkan] frame {}: memory was hashed {hashes} times over {:.1} MB (textures and programs are compared with memory once a frame)",
                    p.frame,
                    hashed as f64 / 1e6
                );
                let wb = std::mem::take(&mut core.targets.write_backs);
                let mut causes: Vec<(u64, (u64, usize, String))> =
                    std::mem::take(&mut core.write_back_causes)
                        .into_iter()
                        .collect();
                causes.sort_by_key(|(_, (n, _, _))| std::cmp::Reverse(*n));
                if wb.runs > 0 {
                    eprintln!(
                        "[vulkan] frame {}: targets were written back to memory {} times ({} \
                         targets, {} bytes): {:?} running the commands and reading the targets \
                         out, {:?} laying the pixels out and writing them",
                        p.frame, wb.runs, wb.targets, wb.bytes, wb.device, wb.memory
                    );
                    for (address, (times, targets, what)) in causes.iter().take(6) {
                        eprintln!(
                            "[vulkan]   the texture at GPU address {address:#x}: {times} times, \n                             {targets} targets at the last; {what}"
                        );
                    }
                }
                if ph.compute.iter().any(|d| !d.is_zero()) {
                    let ms: Vec<String> = ph
                        .compute
                        .iter()
                        .map(|d| format!("{:.0}", d.as_secs_f64() * 1000.0))
                        .collect();
                    eprintln!(
                        "[vulkan] frame {}: the compute launches spent {} ms (program, constant buffers, textures, pipeline, recording, sending and waiting)",
                        p.frame,
                        ms.join(", ")
                    );
                }
                let s = std::mem::take(&mut self.spent);
                let renderer =
                    s.draw.time + s.clear.time + s.finish.time + s.present.time + s.dispatch.time;
                eprintln!(
                    "[vulkan] frame {}: the last 100 took {wall:?}, {renderer:?} of it in the \
                     renderer: draw {} calls ({} failed) {:?}, clear {} calls {:?}, finish {} \
                     calls {:?}, present {:?}, compute {} launches {:?}",
                    p.frame,
                    s.draw.calls,
                    s.failed,
                    s.draw.time,
                    s.clear.calls,
                    s.clear.time,
                    s.finish.calls,
                    s.finish.time,
                    s.present.time,
                    s.dispatch.calls,
                    s.dispatch.time
                );
            }
        }
        result
    }

    /// Show a displayed buffer (in the window and as a picture, as configured).
    fn present_frame(&mut self, p: &Present) -> Result<(), String> {
        let config = self.config.clone();
        // The first frame with enough draws (`NX_DUMP_BUSY`) is dumped whatever its number.
        let busy = !self.busy_dumped && config.dump_busy.is_some_and(|n| self.frame_draws >= n);
        if busy {
            self.busy_dumped = true;
        }
        let draws = self.frame_draws;
        let core = self.core()?;
        core.end_submission()?;
        if core.debug.sync_draws.is_some_and(|n| draws >= n.max(1))
            || core.debug.sync_from.is_some_and(|f| p.frame >= f)
        {
            core.sync_on = true;
        }
        // A displayed buffer ends a frame: textures are compared with memory again in the next.
        core.textures.frame += 1;
        if busy && core.debug.trace_after_busy {
            core.debug.trace_frame = Some(core.textures.frame);
            core.traced_commands = 0;
        }
        #[cfg(feature = "window")]
        if let Some(display) = core.display.as_mut() {
            if let Some(target) = core.targets.at_address(p.address) {
                if let Err(e) = display.show(&core.vk, target, p.crop) {
                    // A window that cannot be shown to does not stop the game.
                    eprintln!("[vulkan] the window is given up: {e}");
                    if let Some(mut gone) = core.display.take() {
                        gone.destroy(&core.vk);
                    }
                }
            }
        }
        let Some(dir) = config.dump_dir.as_ref() else {
            return Ok(());
        };
        if !busy && !p.frame.is_multiple_of(config.dump_every) && p.frame != 1 {
            return Ok(());
        }
        if config.dump_limit.is_some_and(|n| core.dumped >= n) {
            return Ok(());
        }
        let Core {
            vk, frame, targets, ..
        } = core;
        let Some(target) = targets.at_address(p.address) else {
            eprintln!(
                "[vulkan] frame {}: no render target was drawn at the displayed buffer ({:#x}, {}x{})",
                p.frame, p.address, p.width, p.height
            );
            eprintln!("[vulkan]   targets drawn so far: {}", targets.describe());
            return Ok(());
        };
        let (w, h) = (target.key.width, target.key.height);
        let rgba = targets::read_rgba(vk, frame, target)?;
        // What the display shows is the producer's crop of the buffer.
        let (rgba, w, h) = targets::crop_rgba(&rgba, w, h, p.crop);
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(format!("frame_{:05}.png", p.frame));
        if core.debug.trace_gpu {
            eprintln!(
                "[vulkan] frame {}: {}",
                p.frame,
                crate::png::describe(w, h, &rgba)
            );
        }
        crate::png::write_opaque_png(&path, w, h, &rgba)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        core.dumped += 1;
        if core.debug.dump_targets || busy {
            let trace = core.debug.trace_gpu;
            let Core {
                vk, frame, targets, ..
            } = core;
            for key in targets.color_keys().into_iter().take(16) {
                let Some(t) = targets.existing(&key) else {
                    continue;
                };
                match targets::read_rgba(vk, frame, t) {
                    Ok(rgba) => {
                        if trace {
                            eprintln!(
                                "[vulkan] target {:#x} {}x{} {:?}: {}",
                                key.address,
                                key.width,
                                key.height,
                                key.format,
                                crate::png::describe(key.width, key.height, &rgba)
                            );
                        }
                        let name = format!(
                            "target_{:x}_{}x{}_{:05}.png",
                            key.address, key.width, key.height, p.frame
                        );
                        let _ = crate::png::write_opaque_png(
                            &dir.join(name),
                            key.width,
                            key.height,
                            &rgba,
                        );
                    }
                    Err(e) => eprintln!("[vulkan] target {:#x}: {e}", key.address),
                }
            }
        }
        Ok(())
    }
}

impl Backend for Renderer {
    fn draw(&mut self, scene: &Scene<'_>, draw: &Draw) -> Result<(), String> {
        let started = stopwatch();
        let result = self.draw_command(scene, draw);
        self.spent.draw.add(started);
        self.frame_draws += 1;
        if result.is_err() {
            self.spent.failed += 1;
        }
        result
    }

    fn clear(&mut self, scene: &Scene<'_>, clear: &Clear) -> Result<(), String> {
        let started = stopwatch();
        let result = self.clear_command(scene, clear);
        self.spent.clear.add(started);
        result
    }

    fn finish(&mut self) -> Result<(), String> {
        let started = stopwatch();
        let result = self.finish_command();
        self.spent.finish.add(started);
        result
    }

    fn present(&mut self, p: &Present) -> Result<(), String> {
        self.present_command(p)
    }

    fn dispatch(&mut self, scene: &Scene<'_>, dispatch: &Dispatch) -> Result<(), String> {
        let started = stopwatch();
        let result = self.core().and_then(|core| core.dispatch(scene, dispatch));
        self.spent.dispatch.add(started);
        if result.is_err() {
            self.spent.failed += 1;
        }
        result
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if let Some(Ok(mut core)) = self.core.take() {
            let _ = core.flush();
            if core.debug.trace_gpu {
                eprintln!(
                    "[vulkan] {} bytes of textures in the cache, {} bytes of guest memory imported, {} pictures written",
                    core.textures.bytes(),
                    core.global.bytes(),
                    core.dumped
                );
            }
            #[cfg(feature = "window")]
            if let Some(mut display) = core.display.take() {
                display.destroy(&core.vk);
            }
            core.pipelines.destroy(&core.vk);
            core.compute_pipelines.destroy(&core.vk);
            core.layouts.destroy(&core.vk);
            core.shaders.destroy(&core.vk);
            core.textures.destroy(&core.vk);
            core.global.destroy(&core.vk);
            core.targets.destroy(&core.vk);
            core.frame.destroy(&core.vk);
        }
    }
}

/// The first color target the registers name, for messages.
fn first_target(scene: &Scene<'_>) -> String {
    scene
        .regs
        .color_targets_in_use()
        .first()
        .map(|(_, t)| {
            format!(
                "{:#x} {}x{} fmt {:#x}",
                t.address, t.width, t.height, t.format
            )
        })
        .unwrap_or_else(|| "no target".to_string())
}

/// What a draw was given, for `NX_TRACE_STATE`: the state that places and culls its primitives
/// and the first indices and vertices it reads.
fn describe_draw(scene: &Scene<'_>, draw: &Draw) -> Vec<String> {
    let regs = scene.regs;
    let mut lines = vec![
        format!(
            "viewport 0 {:?}; scissor 0 {:?}",
            regs.viewport(0),
            regs.scissor(0)
        ),
        format!(
            "window {:?}; SET_VIEWPORT_SCALE_OFFSET {:#x}, SET_VIEWPORT_CLIP_CONTROL {:#x}, SET_WINDOW_ORIGIN {:#x}",
            regs.window(),
            regs.word(0x192c),
            regs.word(0x193c),
            regs.word(0x13ac)
        ),
        format!("raster {:?}", regs.raster()),
        format!(
            "depth and stencil {:?}; target {:?}; clear values {:?}",
            regs.depth_stencil(),
            regs.depth_target(),
            regs.clear_values()
        ),
        {
            let b = regs.blend();
            format!(
                "blend 0: enabled {} mask {:#x} {:?} independent {} logic {} {:#x}; color targets {:?}",
                b.enabled[0],
                b.write_mask[0],
                b.equation[0],
                b.independent,
                b.logic_op,
                b.logic_op_func,
                regs.color_targets_in_use()
            )
        },
    ];
    let floats = |va: u64, n: usize| -> String {
        match scene.read_vec(va, 4 * n) {
            Some(bytes) => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| format!("{}", f32::from_le_bytes(*c)))
                .collect::<Vec<_>>()
                .join(" "),
            None => "unreadable".to_string(),
        }
    };
    if let nx_gpu::render::Source::Indexed { count, .. } = draw.source {
        let ib = regs.index_buffer();
        let n = (count as usize).min(12);
        let idx = scene
            .read_vec(ib.address, n * ib.size as usize)
            .map(|b| {
                b.chunks_exact(ib.size as usize)
                    .map(|c| {
                        c.iter()
                            .rev()
                            .fold(0u32, |acc, byte| acc << 8 | u32::from(*byte))
                            .to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| "unreadable".to_string());
        lines.push(format!("index buffer {ib:?}: {idx}"));
        // The range of all the indices, and the registers that offset them.
        if let Some(all) = scene.read_vec(ib.address, count as usize * ib.size as usize) {
            let value = |c: &[u8]| {
                c.iter()
                    .rev()
                    .fold(0u32, |acc, byte| acc << 8 | u32::from(*byte))
            };
            let indices = all.chunks_exact(ib.size as usize).map(value);
            let (lo, hi) = indices.fold((u32::MAX, 0), |(lo, hi), i| (lo.min(i), hi.max(i)));
            lines.push(format!(
                "indices run from {lo} to {hi}; bases (vertex, instance, id) {:?}",
                regs.bases()
            ));
        }
    }
    for s in 0..16 {
        let st = regs.vertex_stream(s);
        if st.enabled {
            let words = (st.stride as usize / 4).max(1) * 4;
            lines.push(format!(
                "stream {s} {st:?}: {}",
                floats(st.address, words.min(32))
            ));
        }
    }
    for i in 0..32 {
        let a = regs.vertex_attribute(i);
        if a.active {
            lines.push(format!("attribute {i} {a:?}"));
        }
    }
    lines
}
