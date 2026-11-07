//! Explicit native display images. Their identity and lifetime do not depend on guest-memory
//! hashes or the command processor's render-target cache (A-316).

use super::{targets, Renderer};
use crate::image::ImageInfo;
use crate::native::{Completion, Context, Queue};
use crate::Vk;
use ash::vk;
use nx_gpu::render::Present;
use std::sync::{Arc, Mutex};

struct State {
    target: Option<targets::Target>,
    queue: Queue,
    last: Arc<Completion>,
    stats: SurfaceStats,
}

/// Transfers counted at the explicit image boundary. These exclude GPU-to-GPU display blits.
#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceStats {
    pub clears: u64,
    pub draws: u64,
    pub presents: u64,
    pub readback_bytes: u64,
    pub upload_bytes: u64,
    pub uploads: u64,
    pub texel_reads: u64,
    pub texel_readback_bytes: u64,
}

fn readback_size(target: &targets::Target) -> u64 {
    u64::from(target.key.width)
        * u64::from(target.key.height)
        * if target.key.format == vk::Format::R16G16B16A16_SFLOAT {
            8
        } else {
            4
        }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceFormat {
    Rgba8,
    Rgba8Srgb,
    Rgb10A2,
    Rgba16Float,
}

impl SurfaceFormat {
    pub(crate) fn vulkan(self) -> vk::Format {
        match self {
            Self::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            Self::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            Self::Rgb10A2 => vk::Format::A2B10G10R10_UNORM_PACK32,
            Self::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        }
    }
}

/// A retained image on one renderer's Vulkan device. Only that renderer may present/read it.
pub struct Surface {
    vk: Arc<Vk>,
    state: Mutex<State>,
}

/// Already validated typed image operations, recorded in one native GPU submission.
pub enum SurfaceCommand {
    Clear {
        surface: Arc<Surface>,
        color: [f32; 4],
        /// A checked top-left rectangle in pixels. None clears the complete image.
        rect: Option<[u32; 4]>,
    },
    Draw {
        surface: Arc<Surface>,
        draw: Arc<crate::native::NativeDraw>,
    },
    Signal(Arc<crate::native::Signal>),
    Barrier,
}

fn memory_barrier(vk: &Vk, cmd: vk::CommandBuffer) {
    // SAFETY: recording outside rendering; this dependency orders all preceding image writes.
    unsafe {
        vk.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)],
            &[],
            &[],
        );
    }
}

impl Queue {
    /// Retain/lock each distinct image through recording and queue submission. Consistent
    /// address order avoids a lock cycle when independent logical queues share images.
    pub fn submit_surfaces(
        &mut self,
        commands: &[SurfaceCommand],
        done: Arc<Completion>,
    ) -> Result<(), String> {
        let mut surfaces: Vec<_> = commands
            .iter()
            .filter_map(|c| match c {
                SurfaceCommand::Clear { surface, .. } | SurfaceCommand::Draw { surface, .. } => {
                    Some(Arc::clone(surface))
                }
                SurfaceCommand::Signal(_) | SurfaceCommand::Barrier => None,
            })
            .collect();
        surfaces.sort_unstable_by_key(|s| Arc::as_ptr(s) as usize);
        surfaces.dedup_by(|a, b| Arc::ptr_eq(a, b));
        if surfaces.iter().any(|s| !self.context().uses_device(&s.vk)) {
            return Err("native command image belongs to another Vulkan device".to_string());
        }
        let mut guards: Vec<_> = surfaces
            .iter()
            .map(|s| s.state.lock().unwrap_or_else(|e| e.into_inner()))
            .collect();
        for operation in commands {
            if let SurfaceCommand::Signal(signal) = operation {
                signal.validate(self.context())?;
            }
            if let SurfaceCommand::Draw { surface, draw } = operation {
                let i = surfaces
                    .binary_search_by_key(&(Arc::as_ptr(surface) as usize), |s| {
                        Arc::as_ptr(s) as usize
                    })
                    .expect("retained image");
                let target = guards[i].target.as_ref().expect("live surface");
                draw.validate(
                    self.context(),
                    target.key.format,
                    [target.key.width, target.key.height],
                )?;
            }
            if let SurfaceCommand::Clear {
                surface,
                rect: Some([x, y, width, height]),
                ..
            } = operation
            {
                let i = surfaces
                    .binary_search_by_key(&(Arc::as_ptr(surface) as usize), |s| {
                        Arc::as_ptr(s) as usize
                    })
                    .expect("retained image");
                let target = guards[i].target.as_ref().expect("live surface");
                if *width == 0
                    || *height == 0
                    || !x.checked_add(*width).is_some_and(|v| v <= target.key.width)
                    || !y
                        .checked_add(*height)
                        .is_some_and(|v| v <= target.key.height)
                {
                    return Err("native clear rectangle is empty or outside the image".into());
                }
            }
        }
        let mut clears = vec![0; guards.len()];
        let mut draws = vec![0; guards.len()];
        self.submit_commands(Arc::clone(&done), false, |vk, cmd| {
            for operation in commands {
                match operation {
                    SurfaceCommand::Draw { surface, draw } => {
                        let i = surfaces
                            .binary_search_by_key(&(Arc::as_ptr(surface) as usize), |s| {
                                Arc::as_ptr(s) as usize
                            })
                            .expect("retained image");
                        let target = guards[i].target.as_mut().expect("live surface");
                        memory_barrier(vk, cmd);
                        target.transition(vk, cmd, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                        let attachments = [vk::RenderingAttachmentInfo::default()
                            .image_view(target.image.view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)];
                        let rendering = vk::RenderingInfo::default()
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D::default(),
                                extent: vk::Extent2D {
                                    width: target.key.width,
                                    height: target.key.height,
                                },
                            })
                            .layer_count(1)
                            .color_attachments(&attachments);
                        // SAFETY: resources and target were validated before recording.
                        unsafe {
                            vk.device.cmd_begin_rendering(cmd, &rendering);
                        }
                        draw.record(vk, cmd);
                        // SAFETY: this rendering instance was begun immediately above.
                        unsafe {
                            vk.device.cmd_end_rendering(cmd);
                        }
                        draws[i] += 1;
                    }
                    SurfaceCommand::Clear {
                        surface,
                        color,
                        rect,
                    } => {
                        let i = surfaces
                            .binary_search_by_key(&(Arc::as_ptr(surface) as usize), |s| {
                                Arc::as_ptr(s) as usize
                            })
                            .expect("retained image");
                        let target = guards[i].target.as_mut().expect("live surface");
                        // Same-layout consecutive transfer writes need an explicit dependency.
                        memory_barrier(vk, cmd);
                        if let Some([x, y, width, height]) = *rect {
                            target.transition(vk, cmd, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                            let attachments = [vk::RenderingAttachmentInfo::default()
                                .image_view(target.image.view)
                                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::STORE)];
                            let rendering = vk::RenderingInfo::default()
                                .render_area(vk::Rect2D {
                                    offset: vk::Offset2D::default(),
                                    extent: vk::Extent2D {
                                        width: target.key.width,
                                        height: target.key.height,
                                    },
                                })
                                .layer_count(1)
                                .color_attachments(&attachments);
                            // SAFETY: one retained single-sample color view; the rectangle is
                            // inside its extent. This clear ignores viewport/blend/channel state.
                            unsafe {
                                vk.device.cmd_begin_rendering(cmd, &rendering);
                                vk.device.cmd_clear_attachments(
                                    cmd,
                                    &[vk::ClearAttachment::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .color_attachment(0)
                                        .clear_value(vk::ClearValue {
                                            color: vk::ClearColorValue { float32: *color },
                                        })],
                                    &[vk::ClearRect::default()
                                        .rect(vk::Rect2D {
                                            offset: vk::Offset2D {
                                                x: x as i32,
                                                y: y as i32,
                                            },
                                            extent: vk::Extent2D { width, height },
                                        })
                                        .base_array_layer(0)
                                        .layer_count(1)],
                                );
                                vk.device.cmd_end_rendering(cmd);
                            }
                            clears[i] += 1;
                            continue;
                        }
                        target.transition(vk, cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                        // SAFETY: a retained single-layer color image with transfer usage, in
                        // the correct layout. Every use of its state remains locked until submit.
                        unsafe {
                            vk.device.cmd_clear_color_image(
                                cmd,
                                target.image.image,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                &vk::ClearColorValue { float32: *color },
                                &[vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1)],
                            );
                        }
                        clears[i] += 1;
                    }
                    SurfaceCommand::Signal(signal) => signal.record(vk, cmd),
                    SurfaceCommand::Barrier => memory_barrier(vk, cmd),
                }
            }
        })?;
        let retained: Vec<Arc<dyn Send + Sync>> = commands
            .iter()
            .filter_map(|operation| match operation {
                SurfaceCommand::Draw { draw, .. } => Some(Arc::clone(draw) as Arc<dyn Send + Sync>),
                SurfaceCommand::Signal(signal) => Some(Arc::clone(signal) as Arc<dyn Send + Sync>),
                _ => None,
            })
            .collect();
        self.keep_alive(&done, retained);
        for (i, state) in guards.iter_mut().enumerate() {
            state.last = Arc::clone(&done);
            state.stats.clears += clears[i];
            state.stats.draws += draws[i];
        }
        Ok(())
    }
}

impl Surface {
    /// Raw storage bytes, without the diagnostic display conversion (including float formats).
    pub fn read_texels(
        &self,
        region: crate::native::TextureRegion,
    ) -> Result<(u64, crate::native::Readback), String> {
        use crate::native::texture::{record_copy, Readback, Staging, TexelFormat};
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let target = state.target.as_ref().expect("live surface");
        let (copy, size) =
            region.checked(target.image.info, TexelFormat::display(target.key.format))?;
        let staging = Staging::new(&self.vk, size, None)?;
        let done = state.queue.context().completion()?;
        let version = state.stats.clears + state.stats.draws;
        let State { target, queue, .. } = &mut *state;
        let target = target.as_mut().expect("live surface");
        queue.submit_commands(Arc::clone(&done), true, |vk, cmd| {
            memory_barrier(vk, cmd);
            target.transition(vk, cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            record_copy(
                vk,
                cmd,
                &target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging.handle(),
                copy,
                false,
            );
        })?;
        queue.keep_alive(&done, vec![Arc::clone(&staging) as Arc<dyn Send + Sync>]);
        state.last = Arc::clone(&done);
        state.stats.readback_bytes += size as u64;
        state.stats.texel_reads += 1;
        state.stats.texel_readback_bytes += size as u64;
        Ok((version, Readback { done, staging }))
    }
    pub fn write_texels(
        &self,
        region: crate::native::TextureRegion,
        bytes: &[u8],
    ) -> Result<Arc<Completion>, String> {
        use crate::native::texture::{record_copy, Staging, TexelFormat};
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let target = state.target.as_ref().expect("live surface");
        let (copy, size) =
            region.checked(target.image.info, TexelFormat::display(target.key.format))?;
        if size != bytes.len() {
            return Err("native display upload byte count differs from region".into());
        }
        let staging = Staging::new(&self.vk, size, Some(bytes))?;
        let done = state.queue.context().completion()?;
        let State { target, queue, .. } = &mut *state;
        let target = target.as_mut().expect("live surface");
        queue.submit_commands(Arc::clone(&done), false, |vk, cmd| {
            memory_barrier(vk, cmd);
            target.transition(vk, cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            record_copy(
                vk,
                cmd,
                &target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                staging.handle(),
                copy,
                true,
            );
        })?;
        queue.keep_alive(&done, vec![staging]);
        state.last = Arc::clone(&done);
        state.stats.upload_bytes += size as u64;
        state.stats.uploads += 1;
        Ok(done)
    }
    pub fn stats(&self) -> SurfaceStats {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).stats
    }
    pub fn completion(&self) -> Arc<Completion> {
        Arc::clone(&self.state.lock().unwrap_or_else(|e| e.into_inner()).last)
    }

    /// A typed GPU clear; no upload, guest-method encoding or CPU pixel buffer is involved.
    pub fn clear_color(&self, color: [f32; 4]) -> Result<Arc<Completion>, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let next = state.queue.context().completion()?;
        let State { target, queue, .. } = &mut *state;
        let target = target.as_mut().expect("live surface");
        queue.submit_commands(Arc::clone(&next), false, |vk, cmd| {
            target.transition(vk, cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            // SAFETY: a single-layer color image with transfer-destination usage, in that
            // layout, recorded outside rendering. State serializes all uses of this image.
            unsafe {
                vk.device.cmd_clear_color_image(
                    cmd,
                    target.image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: color },
                    &[vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1)],
                );
            }
        })?;
        state.last = Arc::clone(&next);
        state.stats.clears += 1;
        Ok(next)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        let _ = state.last.wait(u64::MAX);
        // Presentation may have failed between its blit and retirement marker. Destruction is
        // rare; wait for the device here, never on the normal per-frame presentation path.
        let _ = self.vk.wait_idle();
        // SAFETY: all image uses are complete and this final owner retains the actual device.
        self.vk
            .destroy_image(state.target.take().expect("live surface").image);
    }
}

impl Renderer {
    /// Allocate an explicit display image, initially transparent black by native policy.
    pub fn create_native_surface(
        &mut self,
        width: u32,
        height: u32,
        format: SurfaceFormat,
    ) -> Result<Arc<Surface>, String> {
        if width == 0 || height == 0 {
            return Err("unsupported native display dimensions or format".to_string());
        }
        let format = format.vulkan();
        let vk = Arc::clone(&self.core()?.vk);
        let context = Context::new(Arc::clone(&vk));
        let queue = context.queue()?;
        let last = context.completion()?;
        // The first clear replaces this never-submitted placeholder.
        last.cancel("native image has not been initialized");
        let image = vk.create_image(ImageInfo {
            format,
            width,
            height,
            depth: 1,
            layers: 1,
            levels: 1,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            view_type: vk::ImageViewType::TYPE_2D,
            components: vk::ComponentMapping::default(),
        })?;
        let surface = Arc::new(Surface {
            vk,
            state: Mutex::new(State {
                target: Some(targets::Target {
                    key: targets::Key {
                        address: 0,
                        format,
                        width,
                        height,
                        layers: 1,
                        tiling: targets::Tiling::Pitch(width.saturating_mul(
                            if format == vk::Format::R16G16B16A16_SFLOAT {
                                8
                            } else {
                                4
                            },
                        )),
                        block_depth_log2: 0,
                    },
                    image,
                    layout: vk::ImageLayout::UNDEFINED,
                    touched: 0,
                    dirty: false,
                    serial: 0,
                    baseline: None,
                    fresh: false,
                    unchanged: std::cell::Cell::new(None),
                }),
                queue,
                last,
                stats: SurfaceStats::default(),
            }),
        });
        surface.clear_color([0.0; 4])?;
        Ok(surface)
    }

    /// Read back only when explicitly requested by a diagnostic or a configured frame dump.
    pub fn read_native_surface(&mut self, surface: &Surface) -> Result<Vec<u8>, String> {
        let core = self.core()?;
        if !Arc::ptr_eq(&core.vk, &surface.vk) {
            return Err("native image belongs to another Vulkan device".to_string());
        }
        let mut state = surface.state.lock().unwrap_or_else(|e| e.into_inner());
        let target = state.target.as_mut().expect("live surface");
        let bytes = readback_size(target);
        let result = targets::read_rgba(&core.vk, &mut core.frame, target);
        if result.is_ok() {
            state.stats.readback_bytes += bytes;
        }
        result
    }

    /// Present the retained GPU image directly. Headless presentation performs no pixel copy
    /// unless a dump was requested. The returned fence also covers the window's GPU blit.
    pub fn present_native_surface(
        &mut self,
        surface: &Surface,
        request: &Present,
    ) -> Result<Arc<Completion>, String> {
        let config = self.config.clone();
        let core = self.core()?;
        if !Arc::ptr_eq(&core.vk, &surface.vk) {
            return Err("native image belongs to another Vulkan device".to_string());
        }
        let mut state = surface.state.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "window")]
        if let Some(display) = core.display.as_mut() {
            if let Some(due) = request.due {
                nx_gpu::time::sleep_until(due);
            }
            display.show(
                &core.vk,
                state.target.as_mut().expect("live surface"),
                request.crop,
            )?;
        }
        if let Some(dir) = config.dump_dir.as_ref().filter(|_| {
            request.frame >= config.dump_start
                && (request.frame == 1 || request.frame.is_multiple_of(config.dump_every.max(1)))
                && config.dump_limit.is_none_or(|n| core.dumped < n)
        }) {
            let target = state.target.as_mut().expect("live surface");
            let rgba = targets::read_rgba(&core.vk, &mut core.frame, target)?;
            let bytes = readback_size(target);
            let (rgba, w, h) =
                targets::crop_rgba(&rgba, target.key.width, target.key.height, request.crop);
            if config.dump_hash {
                let hash = rgba.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
                    (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
                });
                eprintln!("[nvn2] frame {}: picture {hash:016x}", request.frame);
            } else {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let path = dir.join(format!("frame_{:05}.png", request.frame));
                crate::png::write_opaque_png(&path, w, h, &rgba)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
            core.dumped += 1;
            state.stats.readback_bytes += bytes;
        }
        let done = state.queue.context().completion()?;
        state.queue.submit_marker(Arc::clone(&done), false)?;
        state.last = Arc::clone(&done);
        state.stats.presents += 1;
        Ok(done)
    }
}
