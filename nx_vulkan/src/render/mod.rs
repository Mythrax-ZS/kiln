//! The renderer: what `nx_gpu`'s command processor tells a backend, done on a Vulkan device.
//!
//! The device and everything on it are made when the first draw, clear or displayed buffer
//! arrives, so a program that never draws does not pay for them. The commands of a submission are
//! recorded and run when the submission ends ([`Backend::finish`]), after which the guest may
//! rely on them having happened.

mod convert;
mod debug;
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
use nx_gpu::draw_state::Layout;
use nx_gpu::render::{Backend, Clear, Draw, Present, Scene};
use std::path::PathBuf;
use targets::{Key, Targets};

/// What the renderer is asked to do besides rendering.
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// Write each displayed frame as a PNG into this directory (`NX_DUMP_FRAMES`).
    pub dump_dir: Option<PathBuf>,
    /// Dump every n-th frame only (`NX_DUMP_EVERY`, default 1).
    pub dump_every: u64,
    /// Stop dumping after this many pictures (`NX_DUMP_COUNT`).
    pub dump_limit: Option<u64>,
}

impl Config {
    pub fn from_env() -> Config {
        let number = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
        Config {
            dump_dir: std::env::var_os("NX_DUMP_FRAMES").map(PathBuf::from),
            dump_every: number("NX_DUMP_EVERY").unwrap_or(1).max(1),
            dump_limit: number("NX_DUMP_COUNT"),
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
    /// The attachments of the rendering that is begun in the command buffer, if any.
    pass: Option<Pass>,
    dumped: u64,
    debug: Debug,
    /// Commands of the traced frame so far (`NX_TRACE_DRAWS`).
    traced_commands: u64,
}

pub struct Renderer {
    config: Config,
    core: Option<Result<Core, String>>,
}

impl Renderer {
    pub fn new(config: Config) -> Renderer {
        Renderer { config, core: None }
    }

    fn core(&mut self) -> Result<&mut Core, String> {
        if self.core.is_none() {
            let debug = Debug::from_env();
            let made = Vk::new().and_then(|vk| {
                let frame = Frame::new(&vk)?;
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
                    textures: textures::Textures::default(),
                    global: global::GlobalMemory::new(debug.trace_gpu),
                    layouts: shaders::Layouts::default(),
                    pipelines: draw::Pipelines::default(),
                    pass: None,
                    dumped: 0,
                    debug,
                    traced_commands: 0,
                })
            });
            self.core = Some(made);
        }
        match self.core.as_mut().expect("just made") {
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
            if n >= self.debug.dump_from {
                if let Err(e) = self.dump_first_target(scene, &dir, frame, n) {
                    eprintln!("[vulkan] dumping the target after command {n}: {e}");
                }
            }
        }
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
            .map_err(|e| format!("{}: {e}", path.display()))
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
        Ok(())
    }

    /// Run what was recorded and wait for it.
    fn flush(&mut self) -> Result<(), String> {
        self.end_pass();
        self.frame.submit(&self.vk)?;
        self.targets.submissions += 1;
        Ok(())
    }
}

impl Backend for Renderer {
    fn draw(&mut self, scene: &Scene<'_>, draw: &Draw) -> Result<(), String> {
        let core = self.core()?;
        let result = core.draw(scene, draw);
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

    fn clear(&mut self, scene: &Scene<'_>, clear: &Clear) -> Result<(), String> {
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

    fn finish(&mut self) -> Result<(), String> {
        match self.core.as_mut() {
            Some(Ok(core)) => core.flush(),
            _ => Ok(()),
        }
    }

    fn present(&mut self, p: &Present) -> Result<(), String> {
        let config = self.config.clone();
        let core = self.core()?;
        core.flush()?;
        // A displayed buffer ends a frame: textures are compared with memory again in the next.
        core.textures.frame += 1;
        let Some(dir) = config.dump_dir.as_ref() else {
            return Ok(());
        };
        if !p.frame.is_multiple_of(config.dump_every) && p.frame != 1 {
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
        if core.debug.dump_targets {
            let Core {
                vk, frame, targets, ..
            } = core;
            for key in targets.color_keys().into_iter().take(16) {
                let Some(t) = targets.existing(&key) else {
                    continue;
                };
                match targets::read_rgba(vk, frame, t) {
                    Ok(rgba) => {
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
            core.pipelines.destroy(&core.vk);
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
        format!("window {:?}", regs.window()),
        format!("raster {:?}", regs.raster()),
        format!(
            "depth {:?}; target {:?}",
            regs.depth_stencil().depth_test,
            regs.depth_target()
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
