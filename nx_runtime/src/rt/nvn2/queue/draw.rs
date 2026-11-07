//! Typed graphics draws and watched immutable input versions (A-326, A-337..339).

use super::*;
use crate::rt::nvn2::command::{Draw, Lease, VertexBinding};
use crate::rt::nvn2::event;
use crate::rt::nvn2::state::Bindings;
use crate::rt::nvn2::texture::DisplayImage;
use nx_vulkan::native::{
    Attribute, BlendFactor, BlendOp, ColorBlend, Context, GraphicsState, IndexBuffer, IndexType,
    InputBuffer, NativeDraw, Primitive, ShaderOptions, Stream, VertexFormat, VertexKind,
};
use nx_vulkan::render::{native::SurfaceFormat, Variation};
use std::sync::Mutex;

type InputKey = (u64, u64, u64, u64);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    Vertex,
    Index {
        kind: IndexType,
        restart: Option<u32>,
        native_restart: bool,
    },
}
#[derive(Clone)]
enum Data {
    Vertex(Arc<InputBuffer>),
    Index(Arc<IndexBuffer>),
}
impl Data {
    fn len(&self) -> u64 {
        match self {
            Self::Vertex(b) => b.len(),
            Self::Index(b) => b.len(),
        }
    }
    fn upload_bytes(&self) -> u64 {
        match self {
            Self::Vertex(b) => b.upload_bytes(),
            Self::Index(b) => b.upload_bytes(),
        }
    }
    fn reap(&self) -> Result<(), String> {
        match self {
            Self::Vertex(b) => b.reap_upload(),
            Self::Index(b) => b.reap_upload(),
        }
    }
}
struct Source {
    key: InputKey,
    spans: Vec<(u64, usize)>,
    hosts: Vec<(usize, u64)>,
    size: u64,
    capacity: u64,
}
fn host_spans(spans: &[(u64, usize)]) -> Result<Vec<(usize, u64)>, Error> {
    let mut hosts = Vec::new();
    for &(cpu, n) in spans {
        let mut at = 0;
        while at < n as u64 {
            let (ptr, len) = rt()
                .mem
                .host_run(cpu + at, n as u64 - at)
                .ok_or_else(|| invalid("vertex backing is no longer mapped"))?;
            hosts.push((ptr as usize, len));
            at += len;
        }
    }
    Ok(hosts)
}
impl Source {
    fn capture(
        binding: VertexBinding,
        size: u64,
        memory: &Memory,
        devices: &Devices,
    ) -> Result<Self, Error> {
        binding.lease.validate(memory, &|d| devices.identity(d))?;
        if size == 0 || size > binding.size {
            return Err(invalid(
                "vertex input exceeds its bound range or native upload budget",
            ));
        }
        let key = (
            binding.lease.pool,
            binding.lease.info.cookie,
            binding.offset,
            size,
        );
        let spans = memory.backing_spans(binding.lease.info, binding.offset, size)?;
        let hosts = host_spans(&spans)?;
        Ok(Self {
            key,
            spans,
            hosts,
            size,
            capacity: binding.size,
        })
    }
    fn prefix(mut self, size: u64) -> Result<Self, Error> {
        if size == 0 || size > self.size {
            return Err(invalid("indexed draw exceeds its bound vertex range"));
        }
        self.size = size;
        self.key.3 = size;
        let mut left = size;
        self.spans.retain_mut(|(_, n)| {
            let take = left.min(*n as u64);
            *n = take as usize;
            left -= take;
            take != 0
        });
        let mut left = size;
        self.hosts.retain_mut(|(_, n)| {
            let take = left.min(*n);
            *n = take;
            left -= take;
            take != 0
        });
        Ok(self)
    }
}
struct Entry {
    buffer: Data,
    spans: Vec<(u64, usize)>,
    epoch: u32,
    used: u64,
}
#[derive(Default)]
pub(super) struct Inputs {
    cache: BTreeMap<(InputKey, Role), Entry>,
    constant: Option<Arc<InputBuffer>>,
    bytes: u64,
    clock: u64,
    pub copied: u64,
    pub uploaded: u64,
    pub uploads: u64,
    pub hits: u64,
    pub shared_reads: u64,
    pub index_copied: u64,
    pub index_uploaded: u64,
    pub index_hits: u64,
    pub index_shared: u64,
}
impl Inputs {
    fn get(
        &mut self,
        source: Source,
        context: &Context,
        visibility: &event::Visibility,
    ) -> Result<Arc<InputBuffer>, Error> {
        match self.resolve(source, Role::Vertex, context, visibility)? {
            Data::Vertex(b) => Ok(b),
            Data::Index(_) => unreachable!(),
        }
    }
    fn indices(
        &mut self,
        source: Source,
        role: Role,
        context: &Context,
        visibility: &event::Visibility,
    ) -> Result<Arc<IndexBuffer>, Error> {
        match self.resolve(source, role, context, visibility)? {
            Data::Index(b) => Ok(b),
            Data::Vertex(_) => unreachable!(),
        }
    }
    fn resolve(
        &mut self,
        source: Source,
        role: Role,
        context: &Context,
        visibility: &event::Visibility,
    ) -> Result<Data, Error> {
        let Source {
            key,
            spans,
            hosts,
            size,
            capacity,
        } = source;
        if host_spans(&spans)? != hosts {
            return Err(invalid(
                "vertex backing was remapped during deferred submission",
            ));
        }
        for &(host, len) in &hosts {
            if let Some(input) = visibility.input(host, len, context)? {
                if hosts.len() != 1 {
                    return Err(invalid(
                        "fragmented vertex/event aliases are not implemented",
                    ));
                }
                return match role {
                    Role::Vertex => {
                        self.shared_reads += size;
                        Ok(Data::Vertex(input))
                    }
                    Role::Index {
                        kind,
                        restart,
                        native_restart,
                    } => {
                        let b = context
                            .shared_indices(input, kind, restart, native_restart)
                            .map_err(invalid_gpu)?;
                        self.index_shared += size;
                        Ok(Data::Index(b))
                    }
                };
            }
        }
        self.clock += 1;
        let key = (key, role);
        let valid = |e: &Entry| {
            prefix_matches(&e.spans, &spans, size)
                && spans.iter().all(|(cpu, n)| {
                    rt().mem.untouched_since(*cpu, *n as u64, e.epoch) == Some(true)
                })
        };
        let reusable = if self.cache.get(&key).is_some_and(valid) {
            Some(key)
        } else if role == Role::Vertex {
            let ((pool, cookie, offset, _), _) = key;
            self.cache
                .range(
                    ((pool, cookie, offset, size), Role::Vertex)
                        ..=((pool, cookie, offset, capacity), Role::Vertex),
                )
                .find(|((_, role), e)| *role == Role::Vertex && valid(e))
                .map(|(&key, _)| key)
        } else {
            None
        };
        if let Some(e) = reusable.and_then(|key| self.cache.get_mut(&key)) {
            e.used = self.clock;
            match role {
                Role::Vertex => self.hits += 1,
                Role::Index { .. } => self.index_hits += 1,
            }
            e.buffer.reap().map_err(invalid_gpu)?;
            return Ok(e.buffer.clone());
        }
        if size > 16 * 1024 * 1024 {
            return Err(invalid("native input upload exceeds the 16 MiB budget"));
        }
        // Arm observation before the snapshot and start its epoch before reading. A write
        // after the snapshot is then newer than the saved epoch, including direct JIT stores.
        for (cpu, n) in &spans {
            rt().mem.untouched_since(*cpu, *n as u64, 1);
        }
        let epoch = rt().mem.begin_epoch();
        let mut bytes = vec![0; size as usize];
        let mut offset = 0;
        for (cpu, n) in &spans {
            rt().mem
                .read_bytes(*cpu, &mut bytes[offset..offset + n], false)
                .map_err(|kind| Error::Memory {
                    address: *cpu,
                    write: false,
                    kind,
                })?;
            offset += n;
        }
        let buffer = match role {
            Role::Vertex => Data::Vertex(context.input_buffer(&bytes).map_err(invalid_gpu)?),
            Role::Index {
                kind,
                restart,
                native_restart,
            } => Data::Index(
                context
                    .index_buffer(&bytes, kind, restart, native_restart)
                    .map_err(invalid_gpu)?,
            ),
        };
        if let Some(old) = self.cache.remove(&key) {
            self.bytes -= old.buffer.len();
        }
        while self.bytes + buffer.len() > 64 * 1024 * 1024 || self.cache.len() >= 4096 {
            let Some((&old, _)) = self.cache.iter().min_by_key(|(_, e)| e.used) else {
                break;
            };
            self.bytes -= self.cache.remove(&old).unwrap().buffer.len();
        }
        match role {
            Role::Vertex => {
                self.copied += size;
                self.uploaded += buffer.upload_bytes();
            }
            Role::Index { .. } => {
                self.index_copied += size;
                self.index_uploaded += buffer.upload_bytes();
            }
        }
        self.uploads += 1;
        self.bytes += buffer.len();
        self.cache.insert(
            key,
            Entry {
                buffer: buffer.clone(),
                spans,
                epoch,
                used: self.clock,
            },
        );
        Ok(buffer)
    }
    fn constant(&mut self, context: &Context) -> Result<Arc<InputBuffer>, Error> {
        if let Some(b) = &self.constant {
            return Ok(Arc::clone(b));
        }
        let bytes = [0f32, 0.0, 0.0, 1.0].map(f32::to_le_bytes).concat();
        let buffer = context.input_buffer(&bytes).map_err(invalid_gpu)?;
        self.copied += 16;
        self.uploaded += buffer.upload_bytes();
        self.uploads += 1;
        self.constant = Some(Arc::clone(&buffer));
        Ok(buffer)
    }
}
fn prefix_matches(full: &[(u64, usize)], prefix: &[(u64, usize)], size: u64) -> bool {
    let mut left = size;
    let mut used = 0;
    for &(at, n) in full {
        if left == 0 {
            break;
        }
        let take = left.min(n as u64);
        if prefix.get(used) != Some(&(at, take as usize)) {
            return false;
        }
        left -= take;
        used += 1;
    }
    left == 0 && used == prefix.len()
}

pub(super) struct Scope<'a> {
    pub validation: Option<(&'a Memory, &'a Devices)>,
    pub context: &'a Context,
    pub inputs: Arc<Mutex<Inputs>>,
    pub visibility: Arc<Mutex<event::Visibility>>,
}

pub(super) fn clip(rect: [i32; 4], extent: [u32; 3], lower_left: bool) -> [u32; 4] {
    let [x, y, w, h] = rect.map(i64::from);
    let [width, height, _] = extent;
    let y = if lower_left {
        i64::from(height) - y - h
    } else {
        y
    };
    let (x0, y0) = (x.clamp(0, i64::from(width)), y.clamp(0, i64::from(height)));
    let (x1, y1) = (
        (x + w).clamp(0, i64::from(width)),
        (y + h).clamp(0, i64::from(height)),
    );
    [x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32]
}
fn vertex_format(code: u32) -> Result<VertexFormat, Error> {
    if !(1..=48).contains(&code) {
        return Err(invalid("packed/scaled vertex format is not implemented"));
    }
    let index = (code - 1) % 12;
    let (bits, kind) = match index {
        0..=3 => (
            8,
            [
                VertexKind::Unorm,
                VertexKind::Snorm,
                VertexKind::Uint,
                VertexKind::Sint,
            ][index as usize],
        ),
        4..=8 => (
            16,
            [
                VertexKind::Float,
                VertexKind::Unorm,
                VertexKind::Snorm,
                VertexKind::Uint,
                VertexKind::Sint,
            ][(index - 4) as usize],
        ),
        _ => (
            32,
            [VertexKind::Float, VertexKind::Uint, VertexKind::Sint][(index - 9) as usize],
        ),
    };
    Ok(VertexFormat {
        channels: ((code - 1) / 12 + 1) as u8,
        bits,
        kind,
    })
}
fn factor(v: u32) -> Result<BlendFactor, Error> {
    Ok(match v {
        1 => BlendFactor::Zero,
        2 => BlendFactor::One,
        3 => BlendFactor::SrcColor,
        4 => BlendFactor::OneMinusSrcColor,
        5 => BlendFactor::SrcAlpha,
        6 => BlendFactor::OneMinusSrcAlpha,
        7 => BlendFactor::DstAlpha,
        8 => BlendFactor::OneMinusDstAlpha,
        9 => BlendFactor::DstColor,
        10 => BlendFactor::OneMinusDstColor,
        11 => BlendFactor::SrcAlphaSaturate,
        0x61 => BlendFactor::ConstantColor,
        0x62 => BlendFactor::OneMinusConstantColor,
        0x63 => BlendFactor::ConstantAlpha,
        0x64 => BlendFactor::OneMinusConstantAlpha,
        _ => {
            return Err(invalid(
                "dual-source or unknown blend factor is not implemented",
            ))
        }
    })
}
fn equation(v: u32) -> Result<BlendOp, Error> {
    Ok(match v {
        1 => BlendOp::Add,
        2 => BlendOp::Subtract,
        3 => BlendOp::ReverseSubtract,
        4 => BlendOp::Min,
        5 => BlendOp::Max,
        _ => return Err(invalid("blend equation is invalid")),
    })
}

impl Scope<'_> {
    pub(super) fn draw(
        &mut self,
        draw: &Draw,
        image: &DisplayImage,
        state: &Bindings,
        vertices: &[Option<VertexBinding>; 32],
        programs: &[Option<crate::rt::nvn2::program::Binding>; 6],
    ) -> Result<Option<nx_vulkan::native::Operation>, Error> {
        if draw.count == 0 || draw.instances == 0 {
            return Ok(None);
        }
        let Some((memory, devices)) = self.validation else {
            return Err(invalid("draw needs live owned memory"));
        };
        Lease {
            pool: image.pool,
            info: image.parent,
        }
        .validate(memory, &|d| devices.identity(d))?;
        let vs = programs[0]
            .as_ref()
            .ok_or_else(|| invalid("draw has no vertex program"))?;
        let ps = programs[1]
            .as_ref()
            .ok_or_else(|| invalid("draw has no fragment program"))?;
        vs.link.validate(memory, devices)?;
        ps.link.validate(memory, devices)?;
        if vs.link.device != image.device || ps.link.device != image.device {
            return Err(invalid("draw resources belong to another device"));
        }
        let vs = vs
            .link
            .shaders
            .iter()
            .find(|s| s.stage == 0)
            .ok_or_else(|| invalid("draw has no vertex binary"))?;
        let ps = ps
            .link
            .shaders
            .iter()
            .find(|s| s.stage == 1)
            .ok_or_else(|| invalid("draw has no fragment binary"))?;
        let mut formats = [None; 32];
        let mut classes = [VertexFormat {
            channels: 4,
            bits: 32,
            kind: VertexKind::Float,
        }
        .class(); 32];
        for (i, raw) in state
            .attributes
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            if raw & 64 != 0 {
                let f = vertex_format((raw >> 21) & 1023)?;
                formats[i] = Some(f);
                classes[i] = f.class();
            }
        }
        let vs = vs
            .binary
            .prepare(
                ShaderOptions::Graphics(Variation {
                    stage_index: 0,
                    attribute_classes: classes,
                    target_classes: Default::default(),
                    depth_minus_one_to_one: draw.depth_minus_one_to_one,
                    mesh: None,
                    per_primitive: 0,
                }),
                &(),
            )
            .map_err(invalid_gpu)?;
        let fragment = Variation {
            stage_index: 4,
            attribute_classes: Default::default(),
            target_classes: Default::default(),
            depth_minus_one_to_one: false,
            mesh: None,
            per_primitive: 0,
        };
        let fragment = if draw.lower_left {
            ShaderOptions::LowerLeftFragment {
                variation: fragment,
                target_height: image.dimensions[1],
            }
        } else {
            ShaderOptions::Graphics(fragment)
        };
        let ps = ps.binary.prepare(fragment, &()).map_err(invalid_gpu)?;
        let raw_attributes = state.attributes.as_deref().unwrap_or(&[]);
        let raw_streams = state.streams.as_deref().unwrap_or(&[]);
        let mut attributes = Vec::new();
        let mut streams = Vec::new();
        let mut needed = BTreeMap::<usize, u64>::new();
        let mut indexed_fetches = BTreeMap::<usize, Vec<(u32, u32, u32)>>::new();
        let mut disabled = Vec::new();
        for (i, format) in formats.iter().enumerate() {
            if vs.info.attributes_read[i] == 0 {
                continue;
            }
            if let Some(format) = format {
                let raw = raw_attributes[i];
                let binding = (raw & 31) as usize;
                let offset = (raw >> 7) & 16383;
                let stream = raw_streams
                    .get(binding)
                    .ok_or_else(|| invalid("vertex stream state was not bound"))?;
                if stream[1] > 1 {
                    return Err(invalid(
                        "vertex instance divisors greater than one are not implemented",
                    ));
                }
                let last = if stream[1] == 1 {
                    draw.first_instance + draw.instances - 1
                } else {
                    draw.first + draw.count - 1
                };
                let size = if draw.indices.is_some() && stream[1] == 0 {
                    indexed_fetches.entry(binding).or_default().push((
                        offset,
                        format.bytes(),
                        stream[0],
                    ));
                    vertices[binding]
                        .ok_or_else(|| invalid("indexed vertex stream has no bound buffer"))?
                        .size
                } else {
                    u64::from(last) * u64::from(stream[0])
                        + u64::from(offset)
                        + u64::from(format.bytes())
                };
                needed
                    .entry(binding)
                    .and_modify(|v| *v = (*v).max(size))
                    .or_insert(size);
                attributes.push(Attribute {
                    location: i as u32,
                    binding: binding as u32,
                    format: *format,
                    offset,
                });
            } else {
                disabled.push(i as u32);
            }
        }
        for &binding in needed.keys() {
            streams.push(Stream {
                binding: binding as u32,
                stride: raw_streams[binding][0],
                instanced: raw_streams[binding][1] != 0,
            });
        }
        let constant_binding = if !disabled.is_empty() {
            let binding = (0..32)
                .rev()
                .find(|i| !streams.iter().any(|s| s.binding == *i))
                .ok_or_else(|| invalid("no vertex binding is available for default attributes"))?;
            streams.push(Stream {
                binding,
                stride: 0,
                instanced: false,
            });
            for location in disabled {
                attributes.push(Attribute {
                    location,
                    binding,
                    format: VertexFormat {
                        channels: 4,
                        bits: 32,
                        kind: VertexKind::Float,
                    },
                    offset: 0,
                });
            }
            Some(binding)
        } else {
            None
        };
        let color = state.color.unwrap_or(0x0803_0000);
        let polygon = state.polygon.unwrap_or(20);
        let depth = state.depth_stencil.unwrap_or([0x22, 0x1118_1118]);
        let ms =
            state
                .multisample
                .unwrap_or([5, 0x8888_8888, 0x8888_8888, 0x8888_8888, 0x8888_8888, 0]);
        if color >> 24 & 15 != 8
            || depth[0] & 5 != 0
            || polygon >> 3 & 3 != 2
            || polygon >> 15 & 15 != 0
            || !matches!(ms[0] >> 8 & 31, 0 | 1)
            || ms[0] & 0x00FF_E03A != 0
            || ms[0] >> 15 & 31 != 0
        {
            return Err(invalid(
                "draw alpha/depth/stencil/polygon/multisample mode is not implemented",
            ));
        }
        let blend = if color & (1 << 8) != 0 {
            let [equations, factors] = state.blend[0].unwrap_or([0x110C_0000, 0x0102_0102]);
            if equations & 0x3F00 != 0 {
                return Err(invalid("advanced blending is not implemented"));
            }
            Some(ColorBlend {
                color: (
                    equation(equations >> 24 & 7)?,
                    factor(factors & 127)?,
                    factor(factors >> 8 & 127)?,
                ),
                alpha: (
                    equation(equations >> 28 & 7)?,
                    factor(factors >> 16 & 127)?,
                    factor(factors >> 24 & 127)?,
                ),
            })
        } else {
            None
        };
        let primitive = match draw.primitive {
            0 => Primitive::Points,
            1 => Primitive::Lines,
            3 => Primitive::LineStrip,
            4 => Primitive::Triangles,
            5 => Primitive::TriangleStrip,
            6 => Primitive::TriangleFan,
            _ => return Err(invalid("draw primitive is not implemented")),
        };
        let logic = (color >> 16 & 255) as u8;
        if logic > 15 {
            return Err(invalid("logic operation is invalid"));
        }
        let format = match image.format {
            37 => SurfaceFormat::Rgba8,
            56 => SurfaceFormat::Rgba8Srgb,
            61 => SurfaceFormat::Rgb10A2,
            41 => SurfaceFormat::Rgba16Float,
            _ => return Err(invalid("draw target format is not implemented")),
        };
        let graphics = GraphicsState {
            restart: false,
            attributes,
            streams,
            primitive,
            cull: (polygon & 3) as u8,
            clockwise: (polygon & 4 != 0) ^ draw.lower_left,
            blend,
            write_mask: (state.channels.unwrap_or(u32::MAX) & 15) as u8,
            logic: (logic != 3).then_some(logic),
        };
        let pipeline = self
            .context
            .graphics_pipeline(Arc::clone(&vs), Arc::clone(&ps), format, graphics.clone())
            .map_err(invalid_gpu)?;
        let [width, height, _] = image.dimensions;
        let [x, y, w, h] = state.viewports[0].unwrap_or([0.0, 0.0, width as f32, height as f32]);
        let viewport = if draw.lower_left {
            [x, height as f32 - y, w, -h]
        } else {
            [x, y, w, h]
        };
        let [n, f] = state.depth_ranges[0].unwrap_or([0.0, 1.0]);
        let scissor = state.scissors[0].map_or([0, 0, width, height], |r| {
            clip(r, image.dimensions, draw.lower_left)
        });
        if w == 0.0 || h == 0.0 || scissor[2] == 0 || scissor[3] == 0 {
            return Ok(None);
        }
        // Refuse unsupported pipelines and empty draws before copying any vertex data.
        let mut sources = Vec::new();
        for (binding, size) in needed {
            let source =
                vertices[binding].ok_or_else(|| invalid("vertex stream has no bound buffer"))?;
            sources.push((
                binding as u32,
                Source::capture(source, size, memory, devices)?,
            ));
        }
        let index_source = draw
            .indices
            .map(|index| {
                let restart = state
                    .restart
                    .0
                    .then_some(state.restart.1)
                    .filter(|&n| n <= index.kind.maximum());
                let role = Role::Index {
                    kind: index.kind,
                    restart,
                    native_restart: restart == Some(index.kind.maximum())
                        && matches!(draw.primitive, 3 | 5 | 6),
                };
                Source::capture(index.source, index.source.size, memory, devices)
                    .map(|source| (source, role, index.base_vertex))
            })
            .transpose()?;
        let mut draw = NativeDraw {
            indices: None,
            base_vertex: 0,
            pipeline,
            inputs: Vec::new(),
            viewport: [viewport[0], viewport[1], viewport[2], viewport[3], n, f],
            scissor,
            blend_color: state.blend_color,
            first: draw.first,
            count: draw.count,
            first_instance: draw.first_instance,
            instances: draw.instances,
        };
        let inputs = Arc::clone(&self.inputs);
        let visibility = Arc::clone(&self.visibility);
        let context = self.context.clone();
        let surface = Arc::clone(&image.surface);
        Ok(Some(nx_vulkan::native::Operation::Prepare(Box::new(
            move || {
                let visibility = visibility.lock().unwrap_or_else(|e| e.into_inner());
                let mut inputs = inputs.lock().unwrap_or_else(|e| e.into_inner());
                if let Some((source, role, base)) = index_source {
                    let index = inputs
                        .indices(source, role, &context, &visibility)
                        .map_err(|e| format!("native index input: {e:?}"))?;
                    if index.is_empty() {
                        return Ok(nx_vulkan::render::native::SurfaceCommand::Barrier);
                    }
                    index.vertex_range(base)?;
                    if index.native_restart() {
                        let mut graphics = graphics;
                        graphics.restart = true;
                        draw.pipeline = context.graphics_pipeline(vs, ps, format, graphics)?;
                    }
                    draw.base_vertex = base;
                    draw.indices = Some(index);
                }
                for (binding, source) in sources {
                    let source = if let Some(fetches) = indexed_fetches.get(&(binding as usize)) {
                        let indices = draw.indices.as_ref().expect("indexed fetches");
                        let max = indices.vertex_range(draw.base_vertex)?.map(|(_, max)| max);
                        let mut needed = 0;
                        for &(offset, bytes, stride) in fetches {
                            let size = match max {
                                Some(max) => {
                                    u64::from(max) * u64::from(stride)
                                        + u64::from(offset)
                                        + u64::from(bytes)
                                }
                                None if stride == 0 => u64::from(offset) + u64::from(bytes),
                                None => source.size,
                            };
                            needed = needed.max(size);
                        }
                        source
                            .prefix(needed)
                            .map_err(|e| format!("native indexed vertex range: {e:?}"))?
                    } else {
                        source
                    };
                    draw.inputs.push((
                        binding,
                        inputs
                            .get(source, &context, &visibility)
                            .map_err(|e| format!("native vertex input: {e:?}"))?,
                    ));
                }
                if let Some(binding) = constant_binding {
                    draw.inputs.push((
                        binding,
                        inputs
                            .constant(&context)
                            .map_err(|e| format!("native default vertex input: {e:?}"))?,
                    ));
                }
                Ok(nx_vulkan::render::native::SurfaceCommand::Draw {
                    surface,
                    draw: Arc::new(draw),
                })
            },
        ))))
    }
}
