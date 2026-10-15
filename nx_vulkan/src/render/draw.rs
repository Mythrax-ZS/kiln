//! One draw call: shaders, pipeline, the data it reads, and the commands.

use super::convert;
use super::convert::Topology;
use super::shaders::{bindings_of, Program, Variation};
use super::textures::{texture_format, Copy, Overlay, Tic, Tsc};
use super::Core;
use crate::fast::FastMap;
use crate::image::aspect_of;
use ash::vk;
use nx_gpu::draw_state::{Blend, DepthStencil, Raster, StencilFace};
use nx_gpu::render::{Draw, Scene, Source};
use nx_gpu::state::{StageKind, StageSlot};
use nx_shader::translate::{Class, Dim, Stage, TextureUse};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Add the time since `mark` to `into` and start the next step.
fn lap(mark: &mut Option<Instant>, into: &mut Duration, _step: &'static str) {
    if let Some(mark) = mark {
        let now = Instant::now();
        *into += now - *mark;
        *mark = now;
    }
    #[cfg(feature = "alloc_stats")]
    crate::alloc_stats::lap(_step);
}

/// The fixed state of a pipeline (the rest is set when drawing).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    vertex: u64,
    pixel: Option<u64>,
    /// `(location, binding, format, offset)`.
    attributes: Vec<(u32, u32, i32, u32)>,
    /// `(binding, stride, per instance)`.
    bindings: Vec<(u32, u32, bool)>,
    topology: i32,
    polygon_mode: i32,
    depth_clamp: bool,
    /// Nothing is rasterized (`NX_DEBUG_DISCARD_PROGRAMS`).
    discard: bool,
    color_formats: Vec<i32>,
    depth_format: i32,
    blend: Vec<BlendKey>,
    logic_op: Option<i32>,
    layout: Vec<super::shaders::Binding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BlendKey {
    enable: bool,
    color: (i32, i32, i32),
    alpha: (i32, i32, i32),
    write_mask: u32,
}

#[derive(Default)]
pub struct Pipelines {
    map: FastMap<PipelineKey, vk::Pipeline>,
}

impl Pipelines {
    pub fn destroy(&mut self, vk: &crate::device::Vk) {
        for (_, p) in self.map.drain() {
            // SAFETY: nothing uses the pipeline any more.
            unsafe { vk.device.destroy_pipeline(p, None) };
        }
    }
}

/// Where the handle of a texture was read from, for messages.
#[derive(Clone, Copy, Debug)]
pub(super) enum HandleFrom {
    /// A word of constant buffer `slot` of the binding group of the stage.
    Group { group: u32, slot: u32 },
    /// A word of constant buffer `slot` of a compute launch, the buffer at this address.
    Launch { slot: u32, address: u64 },
}

impl HandleFrom {
    fn describe(self, scene: &Scene<'_>) -> String {
        match self {
            HandleFrom::Group { group, slot } => format!(
                "slot {slot} of group {group} ({:?})",
                scene.cbufs.get(group, slot)
            ),
            HandleFrom::Launch { slot, address } => {
                format!("constant buffer {slot} of the launch at {address:#x}")
            }
        }
    }
}

/// Whose constant buffer it is, for messages.
#[derive(Clone, Copy, Debug)]
pub(super) enum Owner {
    /// The binding group of a stage of a draw.
    Group(u32),
    /// A compute launch.
    Launch,
}

impl std::fmt::Display for Owner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Owner::Group(g) => write!(f, "group {g}"),
            Owner::Launch => write!(f, "the launch"),
        }
    }
}

/// The vertices and indices of a draw as Vulkan will fetch them.
struct Geometry {
    /// Index buffer, offset, type and count; `None` for a plain vertex draw.
    indices: Option<(vk::Buffer, u64, vk::IndexType, u32)>,
    /// First vertex and count of a plain vertex draw.
    vertices: (u32, u32),
    vertex_offset: i32,
    topology: vk::PrimitiveTopology,
    restart: bool,
}

fn logic_op(func: u32) -> vk::LogicOp {
    match func {
        0x1500 => vk::LogicOp::CLEAR,
        0x1501 => vk::LogicOp::AND,
        0x1502 => vk::LogicOp::AND_REVERSE,
        0x1504 => vk::LogicOp::AND_INVERTED,
        0x1505 => vk::LogicOp::NO_OP,
        0x1506 => vk::LogicOp::XOR,
        0x1507 => vk::LogicOp::OR,
        0x1508 => vk::LogicOp::NOR,
        0x1509 => vk::LogicOp::EQUIVALENT,
        0x150A => vk::LogicOp::INVERT,
        0x150B => vk::LogicOp::OR_REVERSE,
        0x150C => vk::LogicOp::COPY_INVERTED,
        0x150D => vk::LogicOp::OR_INVERTED,
        0x150E => vk::LogicOp::NAND,
        0x150F => vk::LogicOp::SET,
        _ => vk::LogicOp::COPY,
    }
}

impl Core {
    pub(super) fn draw(&mut self, scene: &Scene<'_>, draw: &Draw) -> Result<(), String> {
        #[cfg(feature = "alloc_stats")]
        crate::alloc_stats::lap("outside the draws");
        self.last_textures.clear();
        self.depth_read_only = false;
        let regs = scene.regs;
        let raster = regs.raster();
        if !raster.enabled {
            return Ok(());
        }
        let (first_source, count) = match draw.source {
            Source::Arrays { first, count } => (Some(first), count),
            Source::Indexed { count, .. } => (None, count),
            Source::Inline { .. } => {
                return Err("vertices or indices sent in the command stream".to_string())
            }
            Source::Unsupported { .. } => return Err("an automatic or zero-index draw".to_string()),
        };
        let _ = first_source;
        if count == 0 {
            return Ok(());
        }
        // The programs of the draw.
        let stages = regs.stages();
        for s in &stages {
            match s.kind {
                StageKind::Vertex | StageKind::Pixel => {}
                other => return Err(format!("a draw with a {} program", other.name())),
            }
        }
        let vs_slot: &StageSlot = stages
            .iter()
            .find(|s| s.kind == StageKind::Vertex)
            .ok_or("a draw without a vertex program")?;
        let ps_slot = stages.iter().find(|s| s.kind == StageKind::Pixel);
        if stages
            .iter()
            .any(|s| self.debug.skip_programs.contains(&s.address))
        {
            return Ok(());
        }

        // What the shaders see of the state.
        let window = regs.window();
        let mut attribute_classes = [Class::Float; 32];
        let mut formats: [Option<vk::Format>; 32] = [None; 32];
        for (i, class_slot) in attribute_classes.iter_mut().enumerate() {
            let a = regs.vertex_attribute(i);
            if a.active {
                if let Some((format, class)) = convert::vertex_format(a.widths, a.numeric, a.bgra) {
                    formats[i] = Some(format);
                    *class_slot = class;
                }
            }
        }
        let mut target_classes = [Class::Float; 8];
        for (i, t) in regs.color_targets_in_use() {
            if let Some(f) = crate::formats::color_format(t.format) {
                target_classes[i] = f.class;
            }
        }
        let variation = |stage: Stage| Variation {
            stage_index: stage.index(),
            attribute_classes: if stage == Stage::Vertex {
                attribute_classes
            } else {
                [Class::Float; 32]
            },
            target_classes: if stage == Stage::Pixel {
                target_classes
            } else {
                [Class::Float; 8]
            },
            depth_minus_one_to_one: !window.z_zero_to_one,
        };
        let mut mark = super::stopwatch();
        let vs = self.shaders.program(
            &self.vk,
            scene,
            vs_slot,
            variation(Stage::Vertex),
            self.textures.frame,
        )?;
        let ps = match ps_slot {
            Some(slot) => Some(self.shaders.program(
                &self.vk,
                scene,
                slot,
                variation(Stage::Pixel),
                self.textures.frame,
            )?),
            None => None,
        };
        lap(&mut mark, &mut self.phases.programs, "programs");

        if self.debug.trace_textures && self.traced().is_some() {
            for group in 0..5u32 {
                let bound: Vec<String> = (0..18u32)
                    .filter_map(|s| {
                        scene
                            .cbufs
                            .get(group, s)
                            .map(|b| format!("c{s}={:#x}+{:#x}", b.address, b.size))
                    })
                    .collect();
                eprintln!("[vulkan] group {group}: {}", bound.join(" "));
            }
            for s in &stages {
                eprintln!(
                    "[vulkan] stage {} at {:#x} uses group {}",
                    s.kind.name(),
                    s.address,
                    s.group
                );
            }
        }
        // The textures the programs sample: handle, headers, image and sampler.
        let pools = regs.texture_pools();
        let mut image_infos = std::mem::take(&mut self.scratch.image_infos);
        image_infos.clear();
        let mut handles = std::mem::take(&mut self.scratch.handles);
        for (p, slot) in std::iter::once((&vs, vs_slot)).chain(ps.iter().zip(ps_slot)) {
            // The handle is in the program's table, or (a bindless access) in a word of another
            // constant buffer the program names.
            handles.clear();
            for used in &p.info.textures {
                let from_slot = used.source.unwrap_or(pools.bindless_slot);
                let handle = self.texture_handle(scene, slot.group, from_slot, used.index)?;
                handles.push((
                    handle,
                    HandleFrom::Group {
                        group: slot.group,
                        slot: from_slot,
                    },
                ));
            }
            self.bind_textures(
                scene,
                p,
                slot.kind.name(),
                &pools,
                &handles,
                &mut image_infos,
            )?;
        }
        self.scratch.handles = handles;
        lap(&mut mark, &mut self.phases.textures, "textures");
        self.begin_pass(scene)?;
        let pass = self.pass.clone().expect("the pass was begun");
        lap(&mut mark, &mut self.phases.pass, "pass");

        // Vertex input.
        let mut attributes = std::mem::take(&mut self.scratch.attributes);
        attributes.clear();
        let mut streams = std::mem::take(&mut self.scratch.streams);
        streams.clear();
        for (loc, &format) in formats.iter().enumerate() {
            if vs.info.attributes_read[loc] == 0 {
                continue;
            }
            let a = regs.vertex_attribute(loc);
            let Some(format) = format else {
                if a.active {
                    return Err(format!(
                        "vertex attribute {loc} has the format {:#x} with numerical type {}",
                        a.widths, a.numeric
                    ));
                }
                return Err(format!(
                    "vertex attribute {loc} is read by the vertex program and switched off"
                ));
            };
            if !streams.contains(&a.stream) {
                streams.push(a.stream);
            }
            attributes.push((loc as u32, a.stream, format, a.offset));
        }
        streams.sort_unstable();

        // The geometry: indices (or plain vertices) and the vertex memory to cover.
        let (geometry, vertex_range) = self.geometry(scene, draw, &vs)?;

        // Vertex buffers.
        let mut bound = std::mem::take(&mut self.scratch.bound);
        bound.clear();
        let mut binding_state = std::mem::take(&mut self.scratch.binding_state);
        binding_state.clear();
        for &s in &streams {
            let stream = regs.vertex_stream(s as usize);
            if !stream.enabled {
                return Err(format!("vertex stream {s} is read and not enabled"));
            }
            let stride = u64::from(stream.stride);
            let (buffer, offset, state_stride) = if stream.instanced {
                // One element for this instance: bound with a zero stride so that every vertex
                // reads it.
                let divisor = u64::from(stream.frequency.max(1));
                let (_, base_instance, _) = regs.bases();
                let element = u64::from(base_instance) + u64::from(draw.instance) / divisor;
                let n = stride.max(4);
                let (b, o, ptr) = self.frame.reserve(&self.vk, n, 16)?;
                // SAFETY: `reserve` gave `n` writable bytes at `ptr`, which nothing else uses.
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr, n as usize) };
                self.read_exact(scene, stream.address + element * stride, dst)?;
                (b, o, 0)
            } else {
                let (lo, hi) = vertex_range;
                let bytes = (hi + 1) * stride;
                let (b, o, ptr) = self.frame.reserve(&self.vk, bytes, 16)?;
                let from = lo * stride;
                // SAFETY: `reserve` gave `bytes` writable bytes at `ptr`, which nothing else uses;
                // the vertices fit after `from`.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(ptr.add(from as usize), (bytes - from) as usize)
                };
                self.read_exact(scene, stream.address + from, dst)?;
                (b, o, stream.stride)
            };
            bound.push((s, buffer, offset));
            binding_state.push((s, state_stride, false));
        }

        lap(&mut mark, &mut self.phases.vertices, "vertices");

        // Descriptors: the constant buffers the programs read.
        let layout_bindings = match ps.as_deref() {
            Some(p) => bindings_of(&[&*vs, p]),
            None => bindings_of(&[&*vs]),
        };
        let (set_layout, pipeline_layout) = self.layouts.get(&self.vk, &layout_bindings)?;
        let set = self.frame.descriptor_set(&self.vk, set_layout)?;
        let mut buffer_infos = std::mem::take(&mut self.scratch.buffer_infos);
        buffer_infos.clear();
        for (p, slot) in std::iter::once((&vs, vs_slot)).chain(ps.iter().zip(ps_slot)) {
            let base = 64 * p.stage.index();
            for cb in 0..18u32 {
                if p.info.cbuf_extent[cb as usize] == 0 && !p.info.cbuf_dynamic[cb as usize] {
                    continue;
                }
                let want = if p.info.cbuf_dynamic[cb as usize] {
                    u32::MAX
                } else {
                    p.info.cbuf_extent[cb as usize]
                };
                let (buffer, offset, range) = self.constant_buffer(scene, slot.group, cb, want)?;
                buffer_infos.push((
                    base + cb,
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(range),
                ));
            }
            if p.info.global_memory {
                self.global_pending = true;
                let (buffer, size) = self.global.table(&self.vk, &mut self.frame, scene)?;
                buffer_infos.push((
                    base + 20,
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(0)
                        .range(size),
                ));
            }
        }
        // The writes are made in an array on the stack: at most 20 buffers and 40 textures a
        // stage.
        const MAX_WRITES: usize = 128;
        if buffer_infos.len() + image_infos.len() > MAX_WRITES {
            return Err(format!(
                "{} constant buffers and {} textures in one draw",
                buffer_infos.len(),
                image_infos.len()
            ));
        }
        let mut writes =
            [const { std::mem::MaybeUninit::<vk::WriteDescriptorSet<'_>>::uninit() }; MAX_WRITES];
        let mut n = 0;
        for (binding, info) in &buffer_infos {
            writes[n].write(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
            n += 1;
        }
        for (binding, info) in &image_infos {
            writes[n].write(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(info)),
            );
            n += 1;
        }
        // SAFETY: the first `n` writes were made above; the set, the buffers and the images are
        // valid, and the infos the writes point to live until the call is over.
        unsafe {
            let made =
                std::slice::from_raw_parts(writes.as_ptr().cast::<vk::WriteDescriptorSet<'_>>(), n);
            self.vk.device.update_descriptor_sets(made, &[]);
        }
        self.scratch.buffer_infos = buffer_infos;
        self.scratch.image_infos = image_infos;
        lap(&mut mark, &mut self.phases.descriptors, "descriptors");

        // The pipeline.
        let blend = regs.blend();
        let key = self.pipeline_key(
            &pass,
            &vs,
            ps.as_deref(),
            &attributes,
            &binding_state,
            geometry.topology,
            &raster,
            &blend,
            layout_bindings.clone(),
            self.debug.discard_programs.contains(&vs_slot.address),
        );
        let pipeline = self.pipeline(&key, &vs, ps.as_deref(), pipeline_layout, &blend)?;
        self.scratch.attributes = attributes;
        self.scratch.streams = streams;
        self.scratch.binding_state = binding_state;
        lap(&mut mark, &mut self.phases.pipeline, "pipeline");

        // Commands.
        let cmd = self.frame.cmd(&self.vk)?;
        let d = &self.vk.device;
        // SAFETY: the command buffer is recording inside a rendering with the attachments the
        // pipeline was made for; every handle used is valid for the submission.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[set],
                &[],
            );
            // The instance counter of the draw: the hardware runs the instances of a draw one by
            // one, and the program sees which it is through this.
            d.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &draw.instance.to_ne_bytes(),
            );
            for &(binding, buffer, offset) in &bound {
                d.cmd_bind_vertex_buffers(cmd, binding, &[buffer], &[offset]);
            }
        }
        if !self.set_dynamic_state(scene, cmd, &pass, &raster, &regs.depth_stencil(), &blend) {
            return Ok(());
        }
        self.frame.mark(&self.vk, cmd, || {
            format!(
                "draw {:?} topology {} instance {} with the programs at {:#x} and {:?}",
                draw.source,
                draw.topology,
                draw.instance,
                vs_slot.address,
                ps_slot.map(|s| s.address)
            )
        });
        let d = &self.vk.device;
        // SAFETY: as above.
        unsafe {
            match geometry.indices {
                Some((buffer, offset, ty, n)) => {
                    d.cmd_bind_index_buffer(cmd, buffer, offset, ty);
                    d.cmd_draw_indexed(cmd, n, 1, 0, geometry.vertex_offset, 0);
                }
                None => d.cmd_draw(cmd, geometry.vertices.1, 1, geometry.vertices.0, 0),
            }
        }
        let _ = geometry.restart;
        self.scratch.bound = bound;
        lap(&mut mark, &mut self.phases.record, "record");
        Ok(())
    }

    /// A view of the render target that stands for the memory of the texture `tic`, in the layout
    /// for sampling, when there is one of the same size and format: what the title drew into that
    /// memory is in the target and never reaches the memory itself. A target the draw writes to is
    /// not given (it cannot be read while it is written; the memory is read instead).
    fn render_target_view(
        &mut self,
        scene: &Scene<'_>,
        tic: &Tic,
        used: &TextureUse,
    ) -> Result<Option<(vk::ImageView, vk::ImageLayout)>, String> {
        // Only a plain 2D texture is a target's picture (the others have more to them); that is
        // a 2D header, and the first layer of an array or a cube header (as `shape_of` has it).
        if used.dim != Dim::D2 || used.arrayed || !matches!(tic.texture_type(), 1 | 3 | 5 | 7) {
            if matches!(tic.components(), 0x29..=0x2B) && self.debug.trace_textures {
                eprintln!(
                    "[vulkan] a depth texture sampled as {:?}{} from a header of type {}",
                    used.dim,
                    if used.arrayed { " array" } else { "" },
                    tic.texture_type()
                );
            }
            return Ok(None);
        }
        let Some((cpu, _)) = scene.host_address(tic.address()) else {
            if matches!(tic.components(), 0x29..=0x2B) && self.debug.trace_textures {
                eprintln!(
                    "[vulkan] a depth texture at GPU address {:#x} is not mapped",
                    tic.address()
                );
            }
            return Ok(None);
        };
        let Some(format) = texture_format(tic.components(), tic.types()[0], tic.srgb()) else {
            return Ok(None);
        };
        // Compressed memory is not what a render target holds, whatever its size.
        if format.astc {
            return Ok(None);
        }
        // A color target, or else a depth target (a shadow map sampled by the next pass).
        let found = match self.targets.at_address(cpu) {
            Some(t) => Some(t.key),
            None => self.targets.depth_at_address(cpu).map(|t| t.key),
        };
        let wants_depth = aspect_of(format.vk) != vk::ImageAspectFlags::COLOR;
        let trace = wants_depth && self.debug.trace_textures;
        let Some(key) = found else {
            if trace {
                self.say_once(format!(
                    "[vulkan] a depth texture at {cpu:#x} ({}x{}, format {:#x}) has no render target",
                    tic.width(),
                    tic.height(),
                    tic.components()
                ));
            }
            return Ok(None);
        };
        // A depth texture may be a texel narrower or shorter than the target it shows (a title
        // that scales its resolution describes the part of the target it drew; docs/ASSUMPTIONS.md
        // A-153).
        let slack = u32::from(wants_depth);
        if key.format != format.vk
            || key.layers != 1
            || key.width.abs_diff(tic.width()) > slack
            || key.height.abs_diff(tic.height()) > slack
        {
            if trace {
                self.say_once(format!(
                    "[vulkan] a depth texture at {cpu:#x} ({}x{}, {:?}) differs from the target {key:?}",
                    tic.width(),
                    tic.height(),
                    format.vk
                ));
            }
            return Ok(None);
        }
        let (colors, depth) = self.attachments(scene)?;
        // The depth target the draw tests against is sampled while it is attached, as long as
        // the draw changes neither its depth nor its stencil; any other target the draw uses
        // cannot be read while it is written, and its memory is read instead.
        let mut layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        if colors.contains(&key) || (depth == Some(key) && !self.leaves_depth_alone(scene)) {
            if trace {
                self.say_once(format!(
                    "[vulkan] a depth texture at {cpu:#x} is the target the draw writes to (color {}, depth write {}, stencil test {})",
                    colors.contains(&key),
                    scene.regs.depth_stencil().depth_write,
                    scene.regs.depth_stencil().stencil_test
                ));
            }
            return Ok(None);
        }
        if depth == Some(key) {
            self.depth_read_only = true;
            layout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        }
        // Leaving the rendering is what lets the image change layout.
        self.end_pass();
        let cmd = self.frame.cmd(&self.vk)?;
        let Some(target) = self.targets.existing(&key) else {
            return Ok(None);
        };
        target.transition(&self.vk, cmd, layout);
        let view = self
            .vk
            .swizzled_view(&mut target.image, tic.view_mapping())?;
        if self.debug.trace_textures && self.traced().is_some() {
            eprintln!(
                "[vulkan] a texture is the render target at {cpu:#x} ({}x{})",
                key.width, key.height
            );
        }
        Ok(Some((view, layout)))
    }

    /// Print a line the first time it is said (the trace of a state that comes again and again).
    fn say_once(&mut self, line: String) {
        if self.seen_targets.insert(line.clone()) {
            eprintln!("{line}");
        }
    }

    /// The draw neither writes depth nor changes the stencil.
    fn leaves_depth_alone(&self, scene: &Scene<'_>) -> bool {
        let ds = scene.regs.depth_stencil();
        !ds.depth_write
            && (!ds.stencil_test
                || (ds.front.write_mask == 0 && (!ds.two_sided || ds.back.write_mask == 0)))
    }

    /// Read GPU memory into `out`, failing with a message when it is not all mapped.
    pub(super) fn read_exact(
        &self,
        scene: &Scene<'_>,
        va: u64,
        out: &mut [u8],
    ) -> Result<(), String> {
        if scene.read(va, out) {
            Ok(())
        } else {
            Err(format!(
                "{} bytes of GPU memory at {va:#x}, which is not all mapped",
                out.len()
            ))
        }
    }

    /// Read GPU memory, failing with a message when it is not all mapped.
    pub(super) fn read_gpu(
        &self,
        scene: &Scene<'_>,
        va: u64,
        len: usize,
    ) -> Result<Vec<u8>, String> {
        scene
            .read_vec(va, len)
            .ok_or_else(|| format!("{len} bytes of GPU memory at {va:#x}, which is not all mapped"))
    }

    /// The descriptors (binding and image info) of the textures program `p` samples: for each,
    /// from its handle (with a description of where the handle came from, for messages), the
    /// headers of the pools, the image and the sampler.
    pub(super) fn bind_textures(
        &mut self,
        scene: &Scene<'_>,
        p: &Program,
        kind_name: &str,
        pools: &nx_gpu::draw_state::TexturePools,
        handles: &[(u32, HandleFrom)],
        out: &mut Vec<(u32, vk::DescriptorImageInfo)>,
    ) -> Result<(), String> {
        let base = 64 * p.stage.index();
        for (used, (handle, from)) in p.info.textures.iter().zip(handles) {
            let handle = *handle;
            let (tic_id, tsc_id) = (handle & 0xF_FFFF, handle >> 20);
            if tic_id >= pools.header_count || tsc_id >= pools.sampler_count {
                return Err(format!(
                    "the texture handle {handle:#x} (header {tic_id} of {}, sampler {tsc_id} of {})",
                    pools.header_count, pools.sampler_count
                ));
            }
            let mut mark = super::stopwatch();
            let mut tic_bytes = [0u8; 32];
            self.read_exact(
                scene,
                pools.header_address + 32 * u64::from(tic_id),
                &mut tic_bytes,
            )?;
            let mut tsc_bytes = [0u8; 32];
            self.read_exact(
                scene,
                pools.sampler_address + 32 * u64::from(tsc_id),
                &mut tsc_bytes,
            )?;
            let words = |b: &[u8]| -> [u32; 8] {
                std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
            };
            let (tic, tsc) = (Tic(words(&tic_bytes)), Tsc(words(&tsc_bytes)));
            if self.sync_on {
                self.last_textures.push(format!(
                    "texture {} of the {kind_name} program: handle {handle:#x} from {}, header {tic_id} {:08x?} sampler {tsc_id} {:08x?}",
                    used.index,
                    from.describe(scene),
                    tic.0,
                    tsc.0
                ));
            }
            if self.debug.trace_textures && self.traced().is_some() {
                eprintln!(
                    "[vulkan] texture {} of the {kind_name} program: header {tic_id} {:08x?} sampler {tsc_id} {:08x?}",
                    used.index, tic.0, tsc.0
                );
                // What its memory holds now: the image is made once, so this tells a texture that
                // was empty when it was made from one the cache has not seen change.
                let at = tic.address();
                let held = self
                    .textures
                    .span(&tic, used)
                    .and_then(|s| scene.read_vec(at, s.min(4 << 20) as usize));
                eprintln!(
                    "[vulkan]   at GPU address {at:#x}, guest address {:x?}, {}x{}, format {:#x}: {}",
                    scene.host_address(at).map(|(cpu, _)| cpu),
                    tic.width(),
                    tic.height(),
                    tic.components(),
                    match held {
                        Some(b) => format!(
                            "{} of {} bytes are not zero",
                            b.iter().filter(|x| **x != 0).count(),
                            b.len()
                        ),
                        None => "not readable".to_string(),
                    }
                );
            }
            lap(
                &mut mark,
                &mut self.phases.texture_headers,
                "texture_headers",
            );
            let target_view = self.render_target_view(scene, &tic, used)?;
            lap(&mut mark, &mut self.phases.texture_target, "texture_target");
            let decorate = |e: String| {
                format!(
                    "{e} [texture {}: handle {handle:#x} from {}, header {tic_id} at {:#x}, sampler {tsc_id} at {:#x}]",
                    used.index,
                    from.describe(scene),
                    pools.header_address + 32 * u64::from(tic_id),
                    pools.sampler_address + 32 * u64::from(tsc_id)
                )
            };
            let bound = match target_view {
                Some((view, layout)) => self
                    .textures
                    .bind_view(&self.vk, &tic, tsc, used, view, layout)
                    .map_err(decorate)?,
                None => {
                    // The render targets that stand for parts of this texture: the ones that are
                    // exactly a layer and a level of it are copied into its image on the device,
                    // the others that were drawn to are written into the memory it is read from.
                    let overlays = self.sync_targets(scene, &tic, used)?;
                    lap(
                        &mut mark,
                        &mut self.phases.texture_write_back,
                        "texture_write_back",
                    );
                    let (bound, copies) = self
                        .textures
                        .bind(&self.vk, &mut self.frame, scene, tic, tsc, used, &overlays)
                        .map_err(decorate)?;
                    self.record_copies(&copies)?;
                    lap(&mut mark, &mut self.phases.texture_bind, "texture_bind");
                    bound
                }
            };
            out.push((
                base + 24 + used.slot,
                vk::DescriptorImageInfo::default()
                    .sampler(bound.sampler)
                    .image_view(bound.view)
                    .image_layout(bound.layout),
            ));
        }
        Ok(())
    }

    /// The render targets that stand for parts of the memory the texture `tic` reads. Those that
    /// are exactly a layer and a level of it are returned, for the texture's image to be brought
    /// up to date with by copying them on the device (docs/ASSUMPTIONS.md A-113); the others
    /// that were drawn to are written to that memory first (A-111): the title draws into the
    /// faces of a cube map and the levels of a mip chain and then reads them as one texture.
    fn sync_targets(
        &mut self,
        scene: &Scene<'_>,
        tic: &Tic,
        used: &TextureUse,
    ) -> Result<Vec<Overlay>, String> {
        let Some(bytes) = self.textures.span(tic, used) else {
            return Ok(Vec::new());
        };
        let Some((cpu, _)) = scene.host_address(tic.address()) else {
            return Ok(Vec::new());
        };
        let covering = self.targets.covering(cpu, cpu + bytes);
        if covering.is_empty() {
            return Ok(Vec::new());
        }
        let mut overlays = Vec::new();
        let mut dirty = Vec::new();
        for c in covering {
            match super::textures::overlay_of(tic, used, cpu, &c.key, c.serial) {
                Some(overlay) => overlays.push(overlay),
                None if c.dirty => dirty.push(c),
                None => {}
            }
        }
        // A target drawn to before the overlays that hold the whole texture lies under them: what
        // it drew is not in the texture, and putting it in memory would cost a wait for the
        // device and the layout of every pixel for nothing (A-121).
        let keys: Vec<_> = dirty
            .into_iter()
            .filter(|c| {
                let newer: Vec<_> = overlays
                    .iter()
                    .filter(|o| o.serial > c.serial)
                    .copied()
                    .collect();
                !super::textures::overlays_cover(tic, used, &newer)
            })
            .map(|c| c.key)
            .collect();
        if keys.is_empty() {
            return Ok(overlays);
        }
        let _phase = nx_gpu::progress::enter(nx_gpu::progress::WRITING_BACK);
        let cause = self.write_back_causes.entry(tic.address()).or_default();
        cause.0 += 1;
        cause.1 = keys.len();
        if cause.2.is_empty() {
            cause.2 = format!(
                "the header says {}x{}, type {}, {} levels, components {:#x}, array {}, block heights (log2 of GOBs, width, height, depth) {:?}; the first target is {}x{} with format {:?}, {} layers, tiling {:?}, {:#x} bytes after the texture starts",
                tic.width(),
                tic.height(),
                tic.texture_type(),
                tic.levels(),
                tic.components(),
                used.arrayed,
                tic.block_log2(),
                keys[0].width,
                keys[0].height,
                keys[0].format,
                keys[0].layers,
                keys[0].tiling,
                keys[0].address.wrapping_sub(cpu)
            );
        }
        if self.debug.trace_textures {
            for key in &keys {
                eprintln!(
                    "[vulkan] write-back of {key:?}, drawn to {} submissions ago",
                    self.targets.age(key).unwrap_or(0)
                );
            }
        }
        // The rendering in progress ends, and what was recorded runs, so that the pixels are
        // the ones drawn so far.
        self.end_pass();
        self.targets
            .write_back(&self.vk, &mut self.frame, scene.mem, &keys)?;
        for key in &keys {
            let start = key.address;
            self.textures
                .memory_written(start, start + key.memory_bytes().unwrap_or(u64::MAX / 2));
        }
        if self.debug.trace_textures {
            eprintln!(
                "[vulkan] {} render targets written to the memory of the texture at {:#x} ({bytes} bytes)",
                keys.len(),
                tic.address()
            );
        }
        Ok(overlays)
    }

    /// Record the copies of render targets into layers and levels of texture images.
    fn record_copies(&mut self, copies: &[Copy]) -> Result<(), String> {
        if copies.is_empty() {
            return Ok(());
        }
        // The rendering in progress ends: neither a copy nor a layout change can be in one.
        self.end_pass();
        let cmd = self.frame.cmd(&self.vk)?;
        let range = |c: &Copy| vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: c.levels,
            base_array_layer: 0,
            layer_count: c.layers,
        };
        let barrier = |c: &Copy, old, new, from, to| {
            vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(from)
                .dst_access_mask(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(c.image)
                .subresource_range(range(c))
        };
        for c in copies {
            let Some(target) = self.targets.existing(&c.overlay.key) else {
                continue;
            };
            target.transition(&self.vk, cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            let source = target.image.image;
            let o = c.overlay;
            self.frame.mark(&self.vk, cmd, || {
                format!(
                    "copy of the target at {:#x} into layer {} level {} of a texture",
                    o.key.address, o.layer, o.level
                )
            });
            let region = [vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: o.level,
                    base_array_layer: o.layer,
                    layer_count: 1,
                })
                .dst_offset(vk::Offset3D {
                    x: 0,
                    y: o.y as i32,
                    z: 0,
                })
                .extent(vk::Extent3D {
                    width: o.width,
                    height: o.height,
                    depth: 1,
                })];
            let to_dst = [barrier(
                c,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::TRANSFER_WRITE,
            )];
            let to_read = [barrier(
                c,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            )];
            // SAFETY: the command buffer is recording outside a rendering; the images are valid
            // (the target stays in the map and the texture in its cache until the submission
            // ends) and in the layouts the barriers and the copy say.
            unsafe {
                let d = &self.vk.device;
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_dst,
                );
                d.cmd_copy_image(
                    cmd,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    c.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                );
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_read,
                );
            }
        }
        Ok(())
    }

    /// The texture handle a program names with `index`: word `index` of the constant buffer that
    /// holds the handles (`slot` of the stage's binding group).
    fn texture_handle(
        &self,
        scene: &Scene<'_>,
        group: u32,
        slot: u32,
        index: u32,
    ) -> Result<u32, String> {
        let Some(b) = scene.cbufs.get(group, slot) else {
            return Err(format!(
                "texture {index} is named by a handle in constant buffer slot {slot} of group {group}, which is not bound"
            ));
        };
        let offset = 4 * u64::from(index);
        if offset + 4 > u64::from(b.size) {
            return Err(format!(
                "texture {index} is named at byte {offset} of a constant buffer of {} bytes",
                b.size
            ));
        }
        if self.debug.trace_textures && self.traced().is_some() {
            let n = (b.size as usize).min(128);
            if let Some(table) = scene.read_vec(b.address, n) {
                let words: Vec<String> = table
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| format!("{:x}", u32::from_le_bytes(*c)))
                    .collect();
                eprintln!(
                    "[vulkan] handles in slot {slot} of group {group} ({:#x}): {}",
                    b.address,
                    words.join(" ")
                );
            }
        }
        let mut word = [0u8; 4];
        self.read_exact(scene, b.address + offset, &mut word)?;
        Ok(u32::from_le_bytes(word))
    }

    /// The bytes of the constant buffer bound to `slot` of binding group `group` (at most `want`
    /// of them) in upload memory.
    fn constant_buffer(
        &mut self,
        scene: &Scene<'_>,
        group: u32,
        slot: u32,
        want: u32,
    ) -> Result<(vk::Buffer, u64, u64), String> {
        let bound = scene.cbufs.get(group, slot).map(|b| (b.address, b.size));
        self.upload_constant_buffer(scene, Owner::Group(group), slot, bound, want)
    }

    /// The bytes of a constant buffer (its GPU address and size, or none when nothing is bound,
    /// which reads as zeros), at most `want` of them, in upload memory. `owner` names the
    /// buffer's group for the trace.
    pub(super) fn upload_constant_buffer(
        &mut self,
        scene: &Scene<'_>,
        owner: Owner,
        slot: u32,
        bound: Option<(u64, u32)>,
        want: u32,
    ) -> Result<(vk::Buffer, u64, u64), String> {
        let align = self.vk.caps.min_storage_buffer_offset_alignment.max(16);
        let Some((address, bound_size)) = bound else {
            // An unbound slot reads as zeros.
            if self.debug.trace_cbufs && self.traced().is_some() {
                eprintln!(
                    "[vulkan] constant buffer {slot} of {owner} is not bound, so the program reads zeros (it wants {want} bytes)"
                );
            }
            let (buffer, offset) = self.frame.upload(&self.vk, &[0u8; 256], align)?;
            return Ok((buffer, offset, 256));
        };
        let size = bound_size.min(want).max(16);
        let padded = u64::from(size.next_multiple_of(16));
        // Read the bytes straight into upload memory, with zeros after them.
        let (buffer, offset, ptr) = self.frame.reserve(&self.vk, padded, align)?;
        // SAFETY: `reserve` gave `padded` writable bytes at `ptr`, which nothing else uses.
        let data = unsafe { std::slice::from_raw_parts_mut(ptr, padded as usize) };
        let readable = scene.read(address, &mut data[..size as usize]);
        if !readable {
            return Err(format!(
                "the constant buffer of slot {slot} at GPU address {:#x} ({size} bytes) is not mapped",
                address
            ));
        }
        data[size as usize..].fill(0);
        if self.debug.trace_cbufs && self.traced().is_some() {
            // The first words of the buffer, then the others that are not zero, as `word:value`.
            let words: Vec<String> = data
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
                .filter(|(i, c)| *i < 16 || **c != [0u8; 4])
                .take(64)
                .map(|(i, c)| {
                    let w = u32::from_le_bytes(*c);
                    let f = f32::from_bits(w);
                    if f.is_finite() && f != 0.0 && f.abs() > 1e-6 && f.abs() < 1e6 {
                        format!("{i}:{f}")
                    } else {
                        format!("{i}:{w:#x}")
                    }
                })
                .collect();
            let nonzero = data
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|c| **c != [0u8; 4])
                .count();
            eprintln!(
                "[vulkan] constant buffer {slot} of {owner} at {:#x}, {size} bytes, {nonzero} words not zero: {}",
                address,
                words.join(" ")
            );
        }
        Ok((buffer, offset, padded))
    }

    /// Where the vertices come from: indices read from the index buffer (rewritten for the
    /// primitive types Vulkan lacks) or a plain range, and the range of vertices `[lo, hi]` the
    /// vertex streams must cover.
    fn geometry(
        &mut self,
        scene: &Scene<'_>,
        draw: &Draw,
        _vs: &Arc<Program>,
    ) -> Result<(Geometry, (u64, u64)), String> {
        let regs = scene.regs;
        let topology = convert::topology(draw.topology)?;
        let (base_vertex, _base_instance, _id_base) = regs.bases();
        match draw.source {
            Source::Arrays { first, count } => {
                match topology {
                    Topology::Native(t) => {
                        let lo = u64::from(first);
                        let hi = lo + u64::from(count) - 1;
                        Ok((
                            Geometry {
                                indices: None,
                                vertices: (first, count),
                                vertex_offset: 0,
                                topology: t,
                                restart: false,
                            },
                            (lo, hi),
                        ))
                    }
                    other => {
                        // Rewrite the vertex order as indices.
                        let seq: Vec<u32> = (first..first + count).collect();
                        let (t, idx) = rewrite(other, &seq)?;
                        let lo = u64::from(first);
                        let hi = lo + u64::from(count) - 1;
                        let (buffer, offset) = self.upload_indices(&idx)?;
                        Ok((
                            Geometry {
                                indices: Some((
                                    buffer,
                                    offset,
                                    vk::IndexType::UINT32,
                                    idx.len() as u32,
                                )),
                                vertices: (0, 0),
                                vertex_offset: 0,
                                topology: t,
                                restart: false,
                            },
                            (lo, hi),
                        ))
                    }
                }
            }
            Source::Indexed { first, count, size } => {
                let ib = regs.index_buffer();
                let size = size.unwrap_or(ib.size);
                if !matches!(size, 1 | 2 | 4) {
                    return Err(format!("an index size of {size} bytes"));
                }
                let at = ib.address + u64::from(first) * u64::from(size);
                // The common draw: indices of 16 or 32 bits, no restart, a topology Vulkan has.
                // They are read straight into upload memory and used as they are.
                if let (Topology::Native(t), false, 2 | 4) = (topology, ib.restart, size) {
                    let bytes = u64::from(count) * u64::from(size);
                    let (buffer, offset, ptr) = self.frame.reserve(&self.vk, bytes, 4)?;
                    // SAFETY: `reserve` gave `bytes` writable bytes at `ptr`, which nothing else
                    // uses.
                    let data = unsafe { std::slice::from_raw_parts_mut(ptr, bytes as usize) };
                    self.read_exact(scene, at, data)?;
                    let (mut lo, mut hi) = (u32::MAX, 0u32);
                    if size == 2 {
                        for c in data.as_chunks::<2>().0 {
                            let i = u32::from(u16::from_le_bytes(*c));
                            lo = lo.min(i);
                            hi = hi.max(i);
                        }
                    } else {
                        for c in data.as_chunks::<4>().0 {
                            let i = u32::from_le_bytes(*c);
                            lo = lo.min(i);
                            hi = hi.max(i);
                        }
                    }
                    let lo = i64::from(lo) + i64::from(base_vertex as i32);
                    let hi = i64::from(hi) + i64::from(base_vertex as i32);
                    if lo < 0 {
                        return Err(format!("indices that reach the vertex {lo}"));
                    }
                    let ty = if size == 4 {
                        vk::IndexType::UINT32
                    } else {
                        vk::IndexType::UINT16
                    };
                    return Ok((
                        Geometry {
                            indices: Some((buffer, offset, ty, count)),
                            vertices: (0, 0),
                            vertex_offset: base_vertex as i32,
                            topology: t,
                            restart: false,
                        },
                        (lo as u64, hi as u64),
                    ));
                }
                let raw = self.read_gpu(scene, at, count as usize * size as usize)?;
                let mut idx: Vec<u32> = raw
                    .chunks_exact(size as usize)
                    .map(|c| match size {
                        1 => u32::from(c[0]),
                        2 => u32::from(u16::from_le_bytes([c[0], c[1]])),
                        _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    })
                    .collect();
                let restart_value = if ib.restart {
                    Some(ib.restart_index)
                } else {
                    None
                };
                let real = |i: u32| restart_value != Some(i);
                let (mut lo, mut hi) = (u32::MAX, 0u32);
                for &i in idx.iter().filter(|i| real(**i)) {
                    lo = lo.min(i);
                    hi = hi.max(i);
                }
                if lo == u32::MAX {
                    // Nothing but restarts: nothing is drawn.
                    lo = 0;
                    hi = 0;
                }
                let t = match topology {
                    Topology::Native(t) => t,
                    other => {
                        let (t, rewritten) = rewrite(other, &idx)?;
                        idx = rewritten;
                        t
                    }
                };
                // Vulkan restarts on the maximum value of the index type.
                let ty = if idx.iter().any(|i| *i > 0xFFFF && real(*i)) || size == 4 {
                    vk::IndexType::UINT32
                } else {
                    vk::IndexType::UINT16
                };
                let max_value = if ty == vk::IndexType::UINT32 {
                    u32::MAX
                } else {
                    0xFFFF
                };
                if let Some(r) = restart_value {
                    for i in idx.iter_mut() {
                        if *i == r {
                            *i = max_value;
                        }
                    }
                }
                let lo = i64::from(lo) + i64::from(base_vertex as i32);
                let hi = i64::from(hi) + i64::from(base_vertex as i32);
                if lo < 0 {
                    return Err(format!("indices that reach the vertex {lo}"));
                }
                let (buffer, offset) = if ty == vk::IndexType::UINT16 {
                    let bytes: Vec<u8> =
                        idx.iter().flat_map(|i| (*i as u16).to_le_bytes()).collect();
                    self.frame.upload(&self.vk, &bytes, 4)?
                } else {
                    self.upload_indices(&idx)?
                };
                Ok((
                    Geometry {
                        indices: Some((buffer, offset, ty, idx.len() as u32)),
                        vertices: (0, 0),
                        vertex_offset: base_vertex as i32,
                        topology: t,
                        restart: restart_value.is_some(),
                    },
                    (lo as u64, hi as u64),
                ))
            }
            _ => unreachable!("handled by the caller"),
        }
    }

    fn upload_indices(&mut self, idx: &[u32]) -> Result<(vk::Buffer, u64), String> {
        let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
        self.frame.upload(&self.vk, &bytes, 4)
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline_key(
        &self,
        pass: &super::Pass,
        vs: &Program,
        ps: Option<&Program>,
        attributes: &[(u32, u32, vk::Format, u32)],
        bindings: &[(u32, u32, bool)],
        topology: vk::PrimitiveTopology,
        raster: &Raster,
        blend: &Blend,
        layout: Vec<super::shaders::Binding>,
        discard: bool,
    ) -> PipelineKey {
        let color_formats: Vec<i32> = pass.colors.iter().map(|k| k.format.as_raw()).collect();
        let blend_keys = (0..pass.colors.len())
            .map(|i| {
                let e = blend.equation[i];
                let (alpha_op, alpha_src, alpha_dst) = if e.separate_alpha {
                    (e.alpha_op, e.alpha_src, e.alpha_dst)
                } else {
                    (e.color_op, e.color_src, e.color_dst)
                };
                BlendKey {
                    enable: blend.enabled[i],
                    color: (
                        convert::blend_op(e.color_op).as_raw(),
                        convert::blend_factor(e.color_src).as_raw(),
                        convert::blend_factor(e.color_dst).as_raw(),
                    ),
                    alpha: (
                        convert::blend_op(alpha_op).as_raw(),
                        convert::blend_factor(alpha_src).as_raw(),
                        convert::blend_factor(alpha_dst).as_raw(),
                    ),
                    write_mask: u32::from(blend.write_mask[i]),
                }
            })
            .collect();
        PipelineKey {
            vertex: vs.id,
            pixel: ps.map(|p| p.id),
            attributes: attributes
                .iter()
                .map(|(l, b, f, o)| (*l, *b, f.as_raw(), *o))
                .collect(),
            bindings: bindings.to_vec(),
            topology: topology.as_raw(),
            polygon_mode: convert::polygon_mode(raster.front_mode).as_raw(),
            depth_clamp: raster.depth_clamp && self.vk.caps.depth_clamp,
            discard,
            color_formats,
            depth_format: pass.depth.map_or(0, |k| k.format.as_raw()),
            blend: blend_keys,
            logic_op: blend
                .logic_op
                .then(|| logic_op(blend.logic_op_func).as_raw()),
            layout,
        }
    }

    fn pipeline(
        &mut self,
        key: &PipelineKey,
        vs: &Program,
        ps: Option<&Program>,
        layout: vk::PipelineLayout,
        _blend: &Blend,
    ) -> Result<vk::Pipeline, String> {
        if let Some(p) = self.pipelines.map.get(key) {
            return Ok(*p);
        }
        let vertex_bindings: Vec<vk::VertexInputBindingDescription> = key
            .bindings
            .iter()
            .map(|(b, stride, instanced)| {
                vk::VertexInputBindingDescription::default()
                    .binding(*b)
                    .stride(*stride)
                    .input_rate(if *instanced {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    })
            })
            .collect();
        let vertex_attributes: Vec<vk::VertexInputAttributeDescription> = key
            .attributes
            .iter()
            .map(|(location, binding, format, offset)| {
                vk::VertexInputAttributeDescription::default()
                    .location(*location)
                    .binding(*binding)
                    .format(vk::Format::from_raw(*format))
                    .offset(*offset)
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_bindings)
            .vertex_attribute_descriptions(&vertex_attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::from_raw(key.topology))
            .primitive_restart_enable(false);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::from_raw(key.polygon_mode))
            .depth_clamp_enable(key.depth_clamp)
            .rasterizer_discard_enable(key.discard)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let attachments: Vec<vk::PipelineColorBlendAttachmentState> = key
            .blend
            .iter()
            .map(|b| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(b.enable)
                    .color_blend_op(vk::BlendOp::from_raw(b.color.0))
                    .src_color_blend_factor(vk::BlendFactor::from_raw(b.color.1))
                    .dst_color_blend_factor(vk::BlendFactor::from_raw(b.color.2))
                    .alpha_blend_op(vk::BlendOp::from_raw(b.alpha.0))
                    .src_alpha_blend_factor(vk::BlendFactor::from_raw(b.alpha.1))
                    .dst_alpha_blend_factor(vk::BlendFactor::from_raw(b.alpha.2))
                    .color_write_mask(vk::ColorComponentFlags::from_raw(b.write_mask))
            })
            .collect();
        let mut color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        if let Some(op) = key.logic_op {
            color_blend = color_blend
                .logic_op_enable(true)
                .logic_op(vk::LogicOp::from_raw(op));
        }
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::DEPTH_BOUNDS,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut stages = vec![vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs.module)
            .name(c"main")];
        if let Some(p) = ps {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(p.module)
                    .name(c"main"),
            );
        }
        let color_formats: Vec<vk::Format> = key
            .color_formats
            .iter()
            .map(|f| vk::Format::from_raw(*f))
            .collect();
        let depth_format = vk::Format::from_raw(key.depth_format);
        let has_stencil = key.depth_format != 0
            && crate::image::aspect_of(depth_format).contains(vk::ImageAspectFlags::STENCIL);
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if key.depth_format != 0 {
            rendering = rendering.depth_attachment_format(depth_format);
            if has_stencil {
                rendering = rendering.stencil_attachment_format(depth_format);
            }
        }
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering)];
        // SAFETY: the create info and everything it points to live through the call.
        let created = unsafe {
            self.vk
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &info, None)
        }
        .map_err(|(_, e)| format!("creating a graphics pipeline: {e}"))?;
        self.pipelines.map.insert(key.clone(), created[0]);
        Ok(created[0])
    }

    /// Everything that is set per draw: viewport, scissor, culling, depth and stencil, blend
    /// constants. False when the viewport has no area (every vertex lands on a point, so nothing
    /// is drawn and Vulkan has no such viewport).
    fn set_dynamic_state(
        &self,
        scene: &Scene<'_>,
        cmd: vk::CommandBuffer,
        pass: &super::Pass,
        raster: &Raster,
        ds: &DepthStencil,
        blend: &Blend,
    ) -> bool {
        let regs = scene.regs;
        let window = regs.window();
        let v = regs.viewport(0);
        let (w, h) = (pass.extent.0 as f32, pass.extent.1 as f32);
        // The viewport of the registers always applies, whatever `SET_VIEWPORT_SCALE_OFFSET`
        // says (docs/ASSUMPTIONS.md A-112).
        let (vx, vy) = (v.offset[0] - v.scale[0], v.offset[1] - v.scale[1]);
        let (vw, vh) = (2.0 * v.scale[0], 2.0 * v.scale[1]);
        let (near, far) = if window.z_zero_to_one {
            (v.offset[2], v.offset[2] + v.scale[2])
        } else {
            (v.offset[2] - v.scale[2], v.offset[2] + v.scale[2])
        };
        let viewport = vk::Viewport {
            x: vx,
            y: vy,
            width: vw,
            height: vh,
            min_depth: near.clamp(0.0, 1.0),
            max_depth: far.clamp(0.0, 1.0),
        };
        let s = regs.scissor(0);
        let scissor = if s.enabled {
            vk::Rect2D {
                offset: vk::Offset2D {
                    x: s.x_min as i32,
                    y: s.y_min as i32,
                },
                extent: vk::Extent2D {
                    width: s.x_max.saturating_sub(s.x_min),
                    height: s.y_max.saturating_sub(s.y_min),
                },
            }
        } else {
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: pass.extent.0,
                    height: pass.extent.1,
                },
            }
        };
        // Bring-up experiment: ignore the registers' viewport and scissor.
        let (viewport, scissor) = if self.debug.full_viewport {
            (
                vk::Viewport {
                    x: 0.0,
                    y: h,
                    width: w,
                    height: -h,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: pass.extent.0,
                        height: pass.extent.1,
                    },
                },
            )
        } else {
            (viewport, scissor)
        };
        if viewport.width.is_nan() || viewport.width <= 0.0 || viewport.height == 0.0 {
            return false;
        }
        let d = &self.vk.device;
        let cull = if raster.cull && !self.debug.no_cull {
            match raster.cull_face {
                0x404 => vk::CullModeFlags::FRONT,
                0x408 => vk::CullModeFlags::FRONT_AND_BACK,
                _ => vk::CullModeFlags::BACK,
            }
        } else {
            vk::CullModeFlags::NONE
        };
        let front = if (raster.front_face == 0x900) != self.debug.flip_front {
            vk::FrontFace::CLOCKWISE
        } else {
            vk::FrontFace::COUNTER_CLOCKWISE
        };
        let stencil = |face: &StencilFace| {
            (
                convert::stencil_op(face.fail),
                convert::stencil_op(face.z_pass),
                convert::stencil_op(face.z_fail),
                convert::compare_op(face.func),
            )
        };
        // SAFETY: the command buffer is recording; the state set matches the pipeline's
        // dynamic state list.
        unsafe {
            d.cmd_set_viewport(cmd, 0, &[viewport]);
            d.cmd_set_scissor(cmd, 0, &[scissor]);
            d.cmd_set_line_width(cmd, raster.line_width.max(1.0));
            d.cmd_set_cull_mode(cmd, cull);
            d.cmd_set_front_face(cmd, front);
            d.cmd_set_depth_test_enable(cmd, ds.depth_test && !self.debug.no_depth_test);
            d.cmd_set_depth_write_enable(cmd, ds.depth_write);
            let depth_op = match convert::compare_op(ds.depth_func) {
                vk::CompareOp::EQUAL if self.debug.equal_as_lequal => vk::CompareOp::LESS_OR_EQUAL,
                op => op,
            };
            d.cmd_set_depth_compare_op(cmd, depth_op);
            d.cmd_set_depth_bounds_test_enable(cmd, false);
            d.cmd_set_depth_bounds(cmd, 0.0, 1.0);
            d.cmd_set_stencil_test_enable(cmd, ds.stencil_test);
            let front_ops = stencil(&ds.front);
            d.cmd_set_stencil_op(
                cmd,
                vk::StencilFaceFlags::FRONT,
                front_ops.0,
                front_ops.1,
                front_ops.2,
                front_ops.3,
            );
            let back = if ds.two_sided { &ds.back } else { &ds.front };
            let back_ops = stencil(back);
            d.cmd_set_stencil_op(
                cmd,
                vk::StencilFaceFlags::BACK,
                back_ops.0,
                back_ops.1,
                back_ops.2,
                back_ops.3,
            );
            d.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::FRONT, ds.front.reference);
            d.cmd_set_stencil_compare_mask(cmd, vk::StencilFaceFlags::FRONT, ds.front.func_mask);
            d.cmd_set_stencil_write_mask(cmd, vk::StencilFaceFlags::FRONT, ds.front.write_mask);
            d.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::BACK, back.reference);
            d.cmd_set_stencil_compare_mask(cmd, vk::StencilFaceFlags::BACK, back.func_mask);
            d.cmd_set_stencil_write_mask(cmd, vk::StencilFaceFlags::BACK, back.write_mask);
            let biased = raster.bias_fill;
            d.cmd_set_depth_bias_enable(cmd, biased);
            d.cmd_set_depth_bias(cmd, raster.depth_bias, 0.0, raster.slope_scale_depth_bias);
            d.cmd_set_blend_constants(cmd, &blend.constant);
            d.cmd_set_primitive_restart_enable(cmd, false);
        }
        true
    }
}

/// Rewrite the vertices (`seq`, in drawing order) of a primitive type Vulkan does not have as
/// the indices of one it does.
fn rewrite(topology: Topology, seq: &[u32]) -> Result<(vk::PrimitiveTopology, Vec<u32>), String> {
    use vk::PrimitiveTopology as T;
    Ok(match topology {
        Topology::Native(t) => (t, seq.to_vec()),
        Topology::Quads => {
            let mut out = Vec::with_capacity(seq.len() / 4 * 6);
            for q in seq.as_chunks::<4>().0 {
                out.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
            }
            (T::TRIANGLE_LIST, out)
        }
        Topology::QuadStrip => {
            // Pairs of vertices make a strip of quads: (0 1 2 3), (2 3 4 5), ... with the
            // hardware's vertex order 0 1 3 2 around each quad.
            let mut out = Vec::new();
            let mut i = 0;
            while i + 3 < seq.len() {
                out.extend_from_slice(&[
                    seq[i],
                    seq[i + 1],
                    seq[i + 3],
                    seq[i],
                    seq[i + 3],
                    seq[i + 2],
                ]);
                i += 2;
            }
            (T::TRIANGLE_LIST, out)
        }
        Topology::LineLoop => {
            let mut out = seq.to_vec();
            if let Some(first) = seq.first() {
                out.push(*first);
            }
            (T::LINE_STRIP, out)
        }
        Topology::Polygon => (T::TRIANGLE_FAN, seq.to_vec()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quads_become_two_triangles_each() {
        let (t, idx) = rewrite(Topology::Quads, &[10, 11, 12, 13, 20, 21, 22, 23]).unwrap();
        assert_eq!(t, vk::PrimitiveTopology::TRIANGLE_LIST);
        assert_eq!(idx, vec![10, 11, 12, 10, 12, 13, 20, 21, 22, 20, 22, 23]);
    }

    #[test]
    fn a_quad_strip_and_a_line_loop_are_rewritten() {
        let (_, idx) = rewrite(Topology::QuadStrip, &[0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(idx, vec![0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]);
        let (t, idx) = rewrite(Topology::LineLoop, &[5, 6, 7]).unwrap();
        assert_eq!(t, vk::PrimitiveTopology::LINE_STRIP);
        assert_eq!(idx, vec![5, 6, 7, 5]);
    }

    #[test]
    fn logic_ops_follow_the_gl_numbers() {
        assert_eq!(logic_op(0x1503), vk::LogicOp::COPY);
        assert_eq!(logic_op(0x1506), vk::LogicOp::XOR);
        assert_eq!(logic_op(0x150F), vk::LogicOp::SET);
    }
}
