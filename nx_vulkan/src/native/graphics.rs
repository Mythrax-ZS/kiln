//! Native graphics pipelines and immutable input buffers, independent of GPU method streams.

use super::{Context, ShaderModule};
use crate::compute::{DeviceBuffer, HostBuffer};
use crate::fast::FastMap;
use crate::Vk;
use ash::vk;
use nx_shader::translate::{Class, Stage};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexKind {
    Float,
    Uint,
    Sint,
    Unorm,
    Snorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexFormat {
    pub channels: u8,
    pub bits: u8,
    pub kind: VertexKind,
}
impl VertexFormat {
    pub fn bytes(self) -> u32 {
        u32::from(self.channels) * u32::from(self.bits) / 8
    }
    pub fn class(self) -> Class {
        match self.kind {
            VertexKind::Uint => Class::Uint,
            VertexKind::Sint => Class::Sint,
            _ => Class::Float,
        }
    }
    fn vk(self) -> Result<vk::Format, String> {
        use vk::Format as F;
        use VertexKind::*;
        let formats = match (self.bits, self.kind) {
            (8, Unorm) => [
                F::R8_UNORM,
                F::R8G8_UNORM,
                F::R8G8B8_UNORM,
                F::R8G8B8A8_UNORM,
            ],
            (8, Snorm) => [
                F::R8_SNORM,
                F::R8G8_SNORM,
                F::R8G8B8_SNORM,
                F::R8G8B8A8_SNORM,
            ],
            (8, Uint) => [F::R8_UINT, F::R8G8_UINT, F::R8G8B8_UINT, F::R8G8B8A8_UINT],
            (8, Sint) => [F::R8_SINT, F::R8G8_SINT, F::R8G8B8_SINT, F::R8G8B8A8_SINT],
            (16, Float) => [
                F::R16_SFLOAT,
                F::R16G16_SFLOAT,
                F::R16G16B16_SFLOAT,
                F::R16G16B16A16_SFLOAT,
            ],
            (16, Unorm) => [
                F::R16_UNORM,
                F::R16G16_UNORM,
                F::R16G16B16_UNORM,
                F::R16G16B16A16_UNORM,
            ],
            (16, Snorm) => [
                F::R16_SNORM,
                F::R16G16_SNORM,
                F::R16G16B16_SNORM,
                F::R16G16B16A16_SNORM,
            ],
            (16, Uint) => [
                F::R16_UINT,
                F::R16G16_UINT,
                F::R16G16B16_UINT,
                F::R16G16B16A16_UINT,
            ],
            (16, Sint) => [
                F::R16_SINT,
                F::R16G16_SINT,
                F::R16G16B16_SINT,
                F::R16G16B16A16_SINT,
            ],
            (32, Float) => [
                F::R32_SFLOAT,
                F::R32G32_SFLOAT,
                F::R32G32B32_SFLOAT,
                F::R32G32B32A32_SFLOAT,
            ],
            (32, Uint) => [
                F::R32_UINT,
                F::R32G32_UINT,
                F::R32G32B32_UINT,
                F::R32G32B32A32_UINT,
            ],
            (32, Sint) => [
                F::R32_SINT,
                F::R32G32_SINT,
                F::R32G32B32_SINT,
                F::R32G32B32A32_SINT,
            ],
            _ => return Err("native vertex format is not implemented".into()),
        };
        formats
            .get(self.channels.wrapping_sub(1) as usize)
            .copied()
            .ok_or_else(|| "native vertex channel count is invalid".into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Attribute {
    pub location: u32,
    pub binding: u32,
    pub format: VertexFormat,
    pub offset: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stream {
    pub binding: u32,
    pub stride: u32,
    pub instanced: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primitive {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}
impl Primitive {
    fn vk(self) -> vk::PrimitiveTopology {
        match self {
            Self::Points => vk::PrimitiveTopology::POINT_LIST,
            Self::Lines => vk::PrimitiveTopology::LINE_LIST,
            Self::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            Self::Triangles => vk::PrimitiveTopology::TRIANGLE_LIST,
            Self::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            Self::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}
impl BlendOp {
    fn vk(self) -> vk::BlendOp {
        vk::BlendOp::from_raw(self as i32)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
}
impl BlendFactor {
    fn vk(self) -> vk::BlendFactor {
        vk::BlendFactor::from_raw(self as i32)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorBlend {
    pub color: (BlendOp, BlendFactor, BlendFactor),
    pub alpha: (BlendOp, BlendFactor, BlendFactor),
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphicsState {
    pub attributes: Vec<Attribute>,
    pub streams: Vec<Stream>,
    pub primitive: Primitive,
    /// Front/back bits, as VkCullModeFlags: zero, front, back or both.
    pub cull: u8,
    pub clockwise: bool,
    pub blend: Option<ColorBlend>,
    pub write_mask: u8,
    /// VkLogicOp's 0..15 values; None disables the operation.
    pub logic: Option<u8>,
}

pub struct InputBuffer {
    vk: Arc<Vk>,
    data: Option<DeviceBuffer>,
    shared: Option<(Arc<super::SharedMemory>, u64)>,
    len: u64,
    upload: Mutex<Option<Upload>>,
    upload_pending: AtomicBool,
}
struct Upload {
    queue: super::Queue,
    done: Arc<super::Completion>,
    staging: Option<HostBuffer>,
}
impl InputBuffer {
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(crate) fn handle(&self) -> vk::Buffer {
        self.shared.as_ref().map_or_else(
            || self.data.as_ref().expect("live input").buffer,
            |(memory, _)| memory.buffer(),
        )
    }
    pub(crate) fn offset(&self) -> u64 {
        self.shared.as_ref().map_or(0, |(_, offset)| *offset)
    }
    pub fn upload_bytes(&self) -> u64 {
        self.data.as_ref().map_or(0, |data| data.size)
    }
    /// Release completed staging/pool resources without waiting in the draw path.
    pub fn reap_upload(&self) -> Result<(), String> {
        if !self.upload_pending.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut upload = self.upload.lock().unwrap_or_else(|e| e.into_inner());
        let completed = if let Some(u) = upload.as_ref() {
            u.done.signaled()?
        } else {
            true
        };
        if completed {
            if let Some(mut u) = upload.take() {
                u.queue
                    .context()
                    .vk
                    .destroy_host_buffer(u.staging.take().expect("live staging"));
            }
            self.upload_pending.store(false, Ordering::Release);
        }
        Ok(())
    }
}
impl Drop for InputBuffer {
    fn drop(&mut self) {
        if let Some(mut upload) = self
            .upload
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = upload.done.wait(u64::MAX);
            self.vk
                .destroy_host_buffer(upload.staging.take().expect("live staging"));
        }
        if let Some(data) = self.data.take() {
            self.vk.destroy_device_buffer(data);
        }
    }
}

pub struct GraphicsPipeline {
    vk: Arc<Vk>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) format: vk::Format,
    pub(crate) streams: Vec<Stream>,
    attributes: Vec<Attribute>,
    viewport_limits: ([u32; 2], [f32; 2]),
    _modules: [Arc<ShaderModule>; 2],
}
impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        // SAFETY: submissions retain their pipeline; all its uses have completed.
        unsafe {
            self.vk.device.destroy_pipeline(self.pipeline, None);
            self.vk.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    vertex: u64,
    pixel: u64,
    format: i32,
    state: GraphicsState,
}
#[derive(Default)]
pub(super) struct Cache {
    pipelines: FastMap<Key, Arc<GraphicsPipeline>>,
}

impl Context {
    pub fn shared_input(
        &self,
        memory: Arc<super::SharedMemory>,
        offset: u64,
        len: u64,
    ) -> Result<Arc<InputBuffer>, String> {
        if !Arc::ptr_eq(&self.vk, &memory.vk)
            || len == 0
            || !offset
                .checked_add(len)
                .is_some_and(|end| end <= memory.len())
        {
            return Err("shared vertex input is outside its device's import".into());
        }
        Ok(Arc::new(InputBuffer {
            vk: Arc::clone(&self.vk),
            data: None,
            shared: Some((memory, offset)),
            len,
            upload: Mutex::new(None),
            upload_pending: AtomicBool::new(false),
        }))
    }
    pub fn input_buffer(&self, bytes: &[u8]) -> Result<Arc<InputBuffer>, String> {
        if bytes.is_empty() {
            return Err("native input buffer is empty".into());
        }
        let size = (bytes.len() as u64).next_multiple_of(4);
        let mut staging = self
            .vk
            .create_host_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        staging.as_mut_slice().fill(0);
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        let data = match self
            .vk
            .create_device_buffer(size, vk::BufferUsageFlags::VERTEX_BUFFER)
        {
            Ok(data) => data,
            Err(e) => {
                self.vk.destroy_host_buffer(staging);
                return Err(e);
            }
        };
        let result = (|| {
            let mut queue = self.queue()?;
            let done = self.completion()?;
            queue.submit_commands(Arc::clone(&done), false, |vk, command| {
                // SAFETY: live source/destination buffers, aligned sizes, retained until done.
                unsafe {
                    vk.device.cmd_copy_buffer(
                        command,
                        staging.buffer,
                        data.buffer,
                        &[vk::BufferCopy::default().size(size)],
                    );
                }
            })?;
            Ok((queue, done))
        })();
        let (queue, done) = match result {
            Ok(v) => v,
            Err(e) => {
                self.vk.destroy_host_buffer(staging);
                self.vk.destroy_device_buffer(data);
                return Err(e);
            }
        };
        Ok(Arc::new(InputBuffer {
            vk: Arc::clone(&self.vk),
            data: Some(data),
            shared: None,
            len: bytes.len() as u64,
            upload: Mutex::new(Some(Upload {
                queue,
                done,
                staging: Some(staging),
            })),
            upload_pending: AtomicBool::new(true),
        }))
    }
    pub fn graphics_pipeline(
        &self,
        vertex: Arc<ShaderModule>,
        pixel: Arc<ShaderModule>,
        format: super::super::render::native::SurfaceFormat,
        state: GraphicsState,
    ) -> Result<Arc<GraphicsPipeline>, String> {
        if !Arc::ptr_eq(&vertex.vk, &self.vk)
            || !Arc::ptr_eq(&pixel.vk, &self.vk)
            || vertex.stage != Stage::Vertex
            || pixel.stage != Stage::Pixel
        {
            return Err("native graphics modules belong to a different device or stage".into());
        }
        for module in [&vertex, &pixel] {
            let i = &module.info;
            if i.cbuf_extent.iter().any(|n| *n != 0)
                || i.cbuf_dynamic.contains(&true)
                || !i.textures.is_empty()
                || !i.images.is_empty()
                || i.global_memory
            {
                return Err(
                    "native shader descriptor/global-memory binding is not implemented".into(),
                );
            }
        }
        if pixel.info.targets_written & !1 != 0
            || pixel.info.writes_depth
            || state.cull > 3
            || state.write_mask > 15
            || state.logic.is_some_and(|v| v > 15)
        {
            return Err("native pipeline needs unsupported targets or fixed state".into());
        }
        if vertex.info.writes_layer || vertex.info.writes_viewport_index {
            return Err("native layered or multiple-viewport draw is not implemented".into());
        }
        let format = format.vulkan();
        let key = Key {
            vertex: vertex.id,
            pixel: pixel.id,
            format: format.as_raw(),
            state,
        };
        let mut cache = self.graphics.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.pipelines.get(&key) {
            return Ok(Arc::clone(p));
        }
        let pipeline = self.create_graphics(&key, vertex, pixel)?;
        if cache.pipelines.len() >= 4096 {
            cache.pipelines.clear();
        }
        cache.pipelines.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    fn create_graphics(
        &self,
        key: &Key,
        vertex: Arc<ShaderModule>,
        pixel: Arc<ShaderModule>,
    ) -> Result<Arc<GraphicsPipeline>, String> {
        // SAFETY: a property query on the retained physical device.
        let limits = unsafe {
            self.vk
                .instance
                .get_physical_device_properties(self.vk.physical)
        }
        .limits;
        let mut locations = std::collections::BTreeSet::new();
        let mut stream_indices = std::collections::BTreeSet::new();
        for s in &key.state.streams {
            if s.binding >= limits.max_vertex_input_bindings
                || s.stride > limits.max_vertex_input_binding_stride
                || !stream_indices.insert(s.binding)
            {
                return Err(
                    "native vertex binding exceeds device limits or repeats an index".into(),
                );
            }
        }
        for a in &key.state.attributes {
            if a.location >= limits.max_vertex_input_attributes
                || a.offset > limits.max_vertex_input_attribute_offset
                || !stream_indices.contains(&a.binding)
                || !locations.insert(a.location)
            {
                return Err("native vertex attribute exceeds device limits or has no unique binding/location".into());
            }
        }
        let mut attributes = Vec::new();
        for a in &key.state.attributes {
            let format = a.format.vk()?;
            // SAFETY: a property query against a retained physical device.
            let props = unsafe {
                self.vk
                    .instance
                    .get_physical_device_format_properties(self.vk.physical, format)
            };
            if !props
                .buffer_features
                .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
            {
                return Err("native vertex format is not supported by the device".into());
            }
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .location(a.location)
                    .binding(a.binding)
                    .offset(a.offset)
                    .format(format),
            );
        }
        let bindings: Vec<_> = key
            .state
            .streams
            .iter()
            .map(|s| {
                vk::VertexInputBindingDescription::default()
                    .binding(s.binding)
                    .stride(s.stride)
                    .input_rate(if s.instanced {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    })
            })
            .collect();
        let input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&attributes)
            .vertex_binding_descriptions(&bindings);
        let assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(key.state.primitive.vk());
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::from_raw(u32::from(key.state.cull)))
            .front_face(if key.state.clockwise {
                vk::FrontFace::CLOCKWISE
            } else {
                vk::FrontFace::COUNTER_CLOCKWISE
            });
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let mut blend = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
            vk::ColorComponentFlags::from_raw(u32::from(key.state.write_mask)),
        );
        if let Some(b) = key.state.blend {
            blend = blend
                .blend_enable(true)
                .color_blend_op(b.color.0.vk())
                .src_color_blend_factor(b.color.1.vk())
                .dst_color_blend_factor(b.color.2.vk())
                .alpha_blend_op(b.alpha.0.vk())
                .src_alpha_blend_factor(b.alpha.1.vk())
                .dst_alpha_blend_factor(b.alpha.2.vk());
        }
        let attachments = [blend];
        let mut color = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        if let Some(op) = key.state.logic {
            color = color
                .logic_op_enable(true)
                .logic_op(vk::LogicOp::from_raw(i32::from(op)));
        }
        let dynamic_values = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_values);
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(4)];
        // SAFETY: no descriptors are used in this subset; only the optional instance counter.
        let layout = unsafe {
            self.vk.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push),
                None,
            )
        }
        .map_err(|e| format!("creating a native graphics layout: {e}"))?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex.module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(pixel.module)
                .name(c"main"),
        ];
        let formats = [vk::Format::from_raw(key.format)];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&color)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        // SAFETY: the modules/layout and all description arrays stay alive through creation.
        let result = unsafe {
            self.vk
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        };
        let pipeline = match result {
            Ok(v) => v[0],
            Err((made, e)) => {
                // SAFETY: creation failed; none of these objects was submitted.
                unsafe {
                    for p in made {
                        self.vk.device.destroy_pipeline(p, None);
                    }
                    self.vk.device.destroy_pipeline_layout(layout, None);
                }
                return Err(format!("creating a native graphics pipeline: {e}"));
            }
        };
        Ok(Arc::new(GraphicsPipeline {
            vk: Arc::clone(&self.vk),
            pipeline,
            layout,
            format: formats[0],
            streams: key.state.streams.clone(),
            attributes: key.state.attributes.clone(),
            viewport_limits: (limits.max_viewport_dimensions, limits.viewport_bounds_range),
            _modules: [vertex, pixel],
        }))
    }
}

pub struct NativeDraw {
    pub pipeline: Arc<GraphicsPipeline>,
    pub inputs: Vec<(u32, Arc<InputBuffer>)>,
    pub viewport: [f32; 6],
    pub scissor: [u32; 4],
    pub blend_color: [f32; 4],
    pub first: u32,
    pub count: u32,
    pub first_instance: u32,
    pub instances: u32,
}
impl NativeDraw {
    pub(crate) fn validate(
        &self,
        context: &Context,
        format: vk::Format,
        extent: [u32; 2],
    ) -> Result<(), String> {
        if !context.uses_device(&self.pipeline.vk)
            || self.pipeline.format != format
            || self.inputs.iter().any(|(_, b)| !context.uses_device(&b.vk))
        {
            return Err(
                "native draw resources belong to a different device or image format".into(),
            );
        }
        let [x, y, w, h, n, f] = self.viewport;
        let [sx, sy, sw, sh] = self.scissor;
        if [x, y, w, h, n, f].iter().any(|v| !v.is_finite())
            || w <= 0.0
            || h == 0.0
            || !(0.0..=1.0).contains(&n)
            || !(0.0..=1.0).contains(&f)
            || sw == 0
            || sh == 0
            || !sx.checked_add(sw).is_some_and(|v| v <= extent[0])
            || !sy.checked_add(sh).is_some_and(|v| v <= extent[1])
        {
            return Err("native draw viewport or scissor is invalid".into());
        }
        let (dimensions, bounds) = self.pipeline.viewport_limits;
        if w > dimensions[0] as f32
            || h.abs() > dimensions[1] as f32
            || x < bounds[0]
            || x + w > bounds[1]
            || y.min(y + h) < bounds[0]
            || y.max(y + h) > bounds[1]
        {
            return Err("native viewport exceeds device limits".into());
        }
        for stream in &self.pipeline.streams {
            if !self.inputs.iter().any(|(i, _)| *i == stream.binding) {
                return Err("native draw has an unbound vertex stream".into());
            }
        }
        if self.count != 0 && self.instances != 0 {
            for a in &self.pipeline.attributes {
                let stream = self
                    .pipeline
                    .streams
                    .iter()
                    .find(|s| s.binding == a.binding)
                    .expect("validated pipeline binding");
                let input = self
                    .inputs
                    .iter()
                    .find(|(i, _)| *i == a.binding)
                    .expect("validated input");
                let last = if stream.instanced {
                    self.first_instance.checked_add(self.instances - 1)
                } else {
                    self.first.checked_add(self.count - 1)
                }
                .ok_or("native draw range overflows")?;
                let end = u64::from(last) * u64::from(stream.stride)
                    + u64::from(a.offset)
                    + u64::from(a.format.bytes());
                if end > input.1.len() {
                    return Err("native draw would read outside a vertex input".into());
                }
            }
        }
        Ok(())
    }
    pub(crate) fn record(&self, vk: &Vk, command: vk::CommandBuffer) {
        // SAFETY: resources and rectangles were checked; caller retains this draw through its fence.
        unsafe {
            vk.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            for (index, buffer) in &self.inputs {
                vk.device.cmd_bind_vertex_buffers(
                    command,
                    *index,
                    &[buffer.handle()],
                    &[buffer.offset()],
                );
            }
            let [x, y, width, height, min_depth, max_depth] = self.viewport;
            vk.device.cmd_set_viewport(
                command,
                0,
                &[vk::Viewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                }],
            );
            let [x, y, width, height] = self.scissor;
            vk.device.cmd_set_scissor(
                command,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: x as i32,
                        y: y as i32,
                    },
                    extent: vk::Extent2D { width, height },
                }],
            );
            vk.device
                .cmd_set_blend_constants(command, &self.blend_color);
            vk.device.cmd_push_constants(
                command,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &self.first_instance.to_le_bytes(),
            );
            vk.device.cmd_draw(
                command,
                self.count,
                self.instances,
                self.first,
                self.first_instance,
            );
        }
    }
}
