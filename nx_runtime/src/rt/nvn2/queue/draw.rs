//! Typed array draws and watched immutable vertex-input versions (A-326).

use super::*;
use crate::rt::nvn2::command::{Draw, Lease, VertexBinding};
use crate::rt::nvn2::state::Bindings;
use crate::rt::nvn2::texture::DisplayImage;
use nx_vulkan::native::{
    Attribute, BlendFactor, BlendOp, ColorBlend, Context, GraphicsState, InputBuffer, NativeDraw,
    Primitive, ShaderOptions, Stream, VertexFormat, VertexKind,
};
use nx_vulkan::render::{native::SurfaceFormat, Variation};

type InputKey = (u64, u64, u64, u64);
struct Entry {
    buffer: Arc<InputBuffer>,
    spans: Vec<(u64, usize)>,
    epoch: u32,
    used: u64,
}
#[derive(Default)]
pub(super) struct Inputs {
    cache: BTreeMap<InputKey, Entry>,
    constant: Option<Arc<InputBuffer>>,
    bytes: u64,
    clock: u64,
    pub copied: u64,
    pub uploaded: u64,
    pub uploads: u64,
    pub hits: u64,
}
impl Inputs {
    fn get(
        &mut self,
        binding: VertexBinding,
        size: u64,
        memory: &Memory,
        devices: &Devices,
        context: &Context,
    ) -> Result<Arc<InputBuffer>, Error> {
        binding.lease.validate(memory, &|d| devices.identity(d))?;
        if size == 0 || size > binding.size || size > 16 * 1024 * 1024 {
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
        self.clock += 1;
        if let Some(e) = self.cache.get_mut(&key) {
            if e.spans == spans
                && spans.iter().all(|(cpu, n)| {
                    rt().mem.untouched_since(*cpu, *n as u64, e.epoch) == Some(true)
                })
            {
                e.used = self.clock;
                self.hits += 1;
                e.buffer.reap_upload().map_err(invalid_gpu)?;
                return Ok(Arc::clone(&e.buffer));
            }
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
        let buffer = context.input_buffer(&bytes).map_err(invalid_gpu)?;
        if let Some(old) = self.cache.remove(&key) {
            self.bytes -= old.buffer.len();
        }
        while self.bytes + size > 64 * 1024 * 1024 || self.cache.len() >= 4096 {
            let Some((&old, _)) = self.cache.iter().min_by_key(|(_, e)| e.used) else {
                break;
            };
            self.bytes -= self.cache.remove(&old).unwrap().buffer.len();
        }
        self.copied += size;
        self.uploaded += buffer.upload_bytes();
        self.uploads += 1;
        self.bytes += size;
        self.cache.insert(
            key,
            Entry {
                buffer: Arc::clone(&buffer),
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

pub(super) struct Scope<'a> {
    pub validation: Option<(&'a Memory, &'a Devices)>,
    pub context: &'a Context,
    pub inputs: &'a mut Inputs,
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
    ) -> Result<Option<Arc<NativeDraw>>, Error> {
        if draw.count == 0 || draw.instances == 0 {
            return Ok(None);
        }
        let Some((memory, devices)) = self.validation else {
            return Err(invalid("array draw needs live owned memory"));
        };
        Lease {
            pool: image.pool,
            info: image.parent,
        }
        .validate(memory, &|d| devices.identity(d))?;
        let vs = programs[0]
            .as_ref()
            .ok_or_else(|| invalid("array draw has no vertex program"))?;
        let ps = programs[1]
            .as_ref()
            .ok_or_else(|| invalid("array draw has no fragment program"))?;
        vs.link.validate(memory, devices)?;
        ps.link.validate(memory, devices)?;
        if vs.link.device != image.device || ps.link.device != image.device {
            return Err(invalid("array draw resources belong to another device"));
        }
        let vs = vs
            .link
            .shaders
            .iter()
            .find(|s| s.stage == 0)
            .ok_or_else(|| invalid("array draw has no vertex binary"))?;
        let ps = ps
            .link
            .shaders
            .iter()
            .find(|s| s.stage == 1)
            .ok_or_else(|| invalid("array draw has no fragment binary"))?;
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
                let size = u64::from(last) * u64::from(stream[0])
                    + u64::from(offset)
                    + u64::from(format.bytes());
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
                "array draw alpha/depth/stencil/polygon/multisample mode is not implemented",
            ));
        }
        if state.restart.0 {
            return Err(invalid("array draw primitive restart is not implemented"));
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
            _ => return Err(invalid("array draw primitive is not implemented")),
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
            _ => return Err(invalid("array draw target format is not implemented")),
        };
        let pipeline = self
            .context
            .graphics_pipeline(
                vs,
                ps,
                format,
                GraphicsState {
                    attributes,
                    streams,
                    primitive,
                    cull: (polygon & 3) as u8,
                    clockwise: (polygon & 4 != 0) ^ draw.lower_left,
                    blend,
                    write_mask: (state.channels.unwrap_or(u32::MAX) & 15) as u8,
                    logic: (logic != 3).then_some(logic),
                },
            )
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
        let mut inputs = Vec::new();
        for (binding, size) in needed {
            let source =
                vertices[binding].ok_or_else(|| invalid("vertex stream has no bound buffer"))?;
            inputs.push((
                binding as u32,
                self.inputs
                    .get(source, size, memory, devices, self.context)?,
            ));
        }
        if let Some(binding) = constant_binding {
            inputs.push((binding, self.inputs.constant(self.context)?));
        }
        Ok(Some(Arc::new(NativeDraw {
            pipeline,
            inputs,
            viewport: [viewport[0], viewport[1], viewport[2], viewport[3], n, f],
            scissor,
            blend_color: state.blend_color,
            first: draw.first,
            count: draw.count,
            first_instance: draw.first_instance,
            instances: draw.instances,
        })))
    }
}
