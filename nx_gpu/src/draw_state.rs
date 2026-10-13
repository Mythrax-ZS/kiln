//! The state a draw reads from the 3D engine's registers, as plain values.
//!
//! Offsets and fields are those of NVIDIA's class headers `cl9097.h`, `cla097.h`, `clb097.h` and
//! `clb197.h` (open-gpu-doc, MIT licence). Values that are enumerations of the hardware (blend
//! factors, comparison functions, formats) stay as the register holds them: turning them into
//! the vocabulary of a graphics API is the renderer's business, and several of them have two
//! encodings (OpenGL's and Direct3D's) that it must tell apart.

use crate::state::Regs;

/// Registers by byte offset.
mod r {
    pub const COLOR_TARGET: u32 = 0x0800;
    pub const COLOR_TARGET_STRIDE: u32 = 64;
    pub const CT_SELECT: u32 = 0x121c;
    pub const SURFACE_CLIP_HORIZONTAL: u32 = 0x0ff4;
    pub const SURFACE_CLIP_VERTICAL: u32 = 0x0ff8;
    pub const ZT_A: u32 = 0x0fe0;
    pub const ZT_FORMAT: u32 = 0x0fe8;
    pub const ZT_BLOCK_SIZE: u32 = 0x0fec;
    pub const ZT_ARRAY_PITCH: u32 = 0x0ff0;
    pub const ZT_SIZE_A: u32 = 0x1228;
    pub const ZT_SIZE_B: u32 = 0x122c;
    pub const ZT_SIZE_C: u32 = 0x1230;
    pub const ZT_SELECT: u32 = 0x1538;
    pub const ZT_LAYER: u32 = 0x179c;
    pub const VIEWPORT_SCALE: u32 = 0x0a00;
    pub const VIEWPORT_CLIP: u32 = 0x0c00;
    pub const SCISSOR: u32 = 0x0e00;
    pub const WINDOW_ORIGIN: u32 = 0x13ac;
    pub const WINDOW_OFFSET_X: u32 = 0x0df8;
    pub const VIEWPORT_SCALE_OFFSET: u32 = 0x192c;
    pub const VIEWPORT_CLIP_CONTROL: u32 = 0x193c;
    pub const RASTER_ENABLE: u32 = 0x037c;
    pub const CULL: u32 = 0x1918;
    pub const FRONT_FACE: u32 = 0x191c;
    pub const CULL_FACE: u32 = 0x1920;
    pub const FRONT_POLYGON_MODE: u32 = 0x0dac;
    pub const BACK_POLYGON_MODE: u32 = 0x0db0;
    pub const LINE_WIDTH: u32 = 0x13b0;
    pub const ALIASED_LINE_WIDTH: u32 = 0x13b4;
    pub const ALIASED_LINE_WIDTH_ENABLE: u32 = 0x020c;
    pub const POINT_SIZE: u32 = 0x1518;
    pub const DEPTH_BIAS: u32 = 0x15bc;
    pub const SLOPE_SCALE_DEPTH_BIAS: u32 = 0x156c;
    pub const DEPTH_BIAS_CLAMP: u32 = 0x187c;
    pub const POLY_OFFSET_POINT: u32 = 0x0dc0;
    pub const POLY_OFFSET_LINE: u32 = 0x0dc4;
    pub const POLY_OFFSET_FILL: u32 = 0x0dc8;
    pub const PROVOKING_VERTEX: u32 = 0x1684;
    pub const DEPTH_TEST: u32 = 0x12cc;
    pub const DEPTH_WRITE: u32 = 0x12e8;
    pub const DEPTH_FUNC: u32 = 0x130c;
    pub const DEPTH_BOUNDS_TEST: u32 = 0x19bc;
    pub const DEPTH_BOUNDS_MIN: u32 = 0x0f9c;
    pub const DEPTH_BOUNDS_MAX: u32 = 0x0fa0;
    pub const STENCIL_TEST: u32 = 0x1380;
    pub const STENCIL_FRONT: u32 = 0x1384;
    pub const STENCIL_FRONT_REF: u32 = 0x1394;
    pub const STENCIL_FRONT_FUNC_MASK: u32 = 0x1398;
    pub const STENCIL_FRONT_MASK: u32 = 0x139c;
    pub const STENCIL_TWO_SIDED: u32 = 0x1594;
    pub const STENCIL_BACK: u32 = 0x1598;
    pub const STENCIL_BACK_REF: u32 = 0x0f54;
    pub const STENCIL_BACK_MASK: u32 = 0x0f58;
    pub const STENCIL_BACK_FUNC_MASK: u32 = 0x0f5c;
    pub const BLEND: u32 = 0x1360;
    pub const BLEND_SEPARATE_FOR_ALPHA: u32 = 0x133c;
    pub const BLEND_COLOR_OP: u32 = 0x1340;
    pub const BLEND_STATE_PER_TARGET: u32 = 0x12e4;
    pub const BLEND_PER_TARGET: u32 = 0x1e00;
    pub const BLEND_CONSTANT: u32 = 0x131c;
    pub const CT_WRITE: u32 = 0x1a00;
    pub const LOGIC_OP: u32 = 0x19c4;
    pub const LOGIC_OP_FUNC: u32 = 0x19c8;
    pub const VERTEX_ATTRIBUTE: u32 = 0x1160;
    pub const VERTEX_STREAM_A: u32 = 0x1c00;
    pub const VERTEX_STREAM_B: u32 = 0x1d00;
    pub const VERTEX_STREAM_INSTANCE_A: u32 = 0x1880;
    pub const VERTEX_STREAM_INSTANCE_B: u32 = 0x18c0;
    pub const VERTEX_STREAM_LIMIT_A: u32 = 0x1f00;
    pub const VERTEX_STREAM_LIMIT_B: u32 = 0x1f80;
    pub const INDEX_BUFFER_A: u32 = 0x17c8;
    pub const PRIMITIVE_RESTART: u32 = 0x1644;
    pub const PRIMITIVE_RESTART_INDEX: u32 = 0x1648;
    pub const GLOBAL_BASE_VERTEX_INDEX: u32 = 0x1434;
    pub const GLOBAL_BASE_INSTANCE_INDEX: u32 = 0x1438;
    pub const VERTEX_ID_BASE: u32 = 0x1118;
    pub const TEX_SAMPLER_POOL: u32 = 0x155c;
    pub const TEX_HEADER_POOL: u32 = 0x1574;
    pub const BINDLESS_TEXTURE: u32 = 0x2608;
    pub const COLOR_CLEAR_VALUE: u32 = 0x0d80;
    pub const Z_CLEAR_VALUE: u32 = 0x0d90;
    pub const STENCIL_CLEAR_VALUE: u32 = 0x0da0;
    pub const CLEAR_RECT_HORIZONTAL: u32 = 0x0d6c;
    pub const CLEAR_RECT_VERTICAL: u32 = 0x0d70;
    pub const CLEAR_SURFACE_CONTROL: u32 = 0x10f8;
}

/// The number of color targets, viewports, vertex attributes and vertex streams.
pub const TARGETS: usize = 8;
pub const VIEWPORTS: usize = 16;
pub const ATTRIBUTES: usize = 32;
pub const STREAMS: usize = 32;

/// How a surface is laid out in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Rows one after another; `width` of a target is then the pitch in bytes.
    Pitch,
    /// Blocks of GOBs: `block` is the size of a block in GOBs as powers of two (width, height,
    /// depth).
    BlockLinear { block: [u32; 3] },
}

impl Layout {
    fn from_memory_word(w: u32) -> Layout {
        if w >> 12 & 1 != 0 {
            Layout::Pitch
        } else {
            Layout::BlockLinear {
                block: [w & 0xF, w >> 4 & 0xF, w >> 8 & 0xF],
            }
        }
    }
}

/// A color render target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorTarget {
    pub address: u64,
    /// Pixels, or bytes of a row for a pitch-linear surface.
    pub width: u32,
    pub height: u32,
    /// `SET_COLOR_TARGET_FORMAT` (0: the target is disabled).
    pub format: u32,
    pub layout: Layout,
    /// The third dimension: layers (or depth when `depth_is_size`).
    pub depth: u32,
    pub depth_is_size: bool,
    pub array_pitch: u32,
    pub layer: u32,
}

/// The depth/stencil target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthTarget {
    pub address: u64,
    pub width: u32,
    pub height: u32,
    /// `SET_ZT_FORMAT` (a `ZT_FORMAT` code).
    pub format: u32,
    pub layout: Layout,
    pub layers: u32,
    pub array_pitch: u32,
    pub layer: u32,
}

/// One of the 16 viewports: the transform from clip space to the window and the clip rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub scale: [f32; 3],
    pub offset: [f32; 3],
    pub clip_x: u32,
    pub clip_width: u32,
    pub clip_y: u32,
    pub clip_height: u32,
    pub min_z: f32,
    pub max_z: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scissor {
    pub enabled: bool,
    pub x_min: u32,
    pub x_max: u32,
    pub y_min: u32,
    pub y_max: u32,
}

/// Rasterization state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Raster {
    /// `false`: primitives are discarded before rasterization.
    pub enabled: bool,
    pub cull: bool,
    /// `OGL_SET_CULL_FACE`: 0x404 front, 0x405 back, 0x408 both.
    pub cull_face: u32,
    /// `OGL_SET_FRONT_FACE`: 0x900 clockwise, 0x901 counter-clockwise.
    pub front_face: u32,
    /// `SET_FRONT_POLYGON_MODE` / `SET_BACK_POLYGON_MODE`: 0x1B00 point, 0x1B01 line, 0x1B02 fill.
    pub front_mode: u32,
    pub back_mode: u32,
    pub line_width: f32,
    pub point_size: f32,
    pub depth_bias: f32,
    pub slope_scale_depth_bias: f32,
    pub depth_bias_clamp: f32,
    pub bias_fill: bool,
    pub bias_line: bool,
    pub bias_point: bool,
    pub provoking_last: bool,
    /// Depth outside the viewport's range is clamped to it, not clipped (both the "pixel min z"
    /// and "pixel max z" fields of `SET_VIEWPORT_CLIP_CONTROL` say clamp).
    pub depth_clamp: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    pub fail: u32,
    pub z_fail: u32,
    pub z_pass: u32,
    pub func: u32,
    pub reference: u32,
    pub func_mask: u32,
    pub write_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthStencil {
    pub depth_test: bool,
    pub depth_write: bool,
    /// `SET_DEPTH_FUNC`: 0x200..0x207 (OpenGL) or 1..8 (Direct3D).
    pub depth_func: u32,
    pub bounds_test: bool,
    pub bounds: [f32; 2],
    pub stencil_test: bool,
    pub two_sided: bool,
    pub front: StencilFace,
    pub back: StencilFace,
}

/// The blend equation of a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendEquation {
    pub separate_alpha: bool,
    pub color_op: u32,
    pub color_src: u32,
    pub color_dst: u32,
    pub alpha_op: u32,
    pub alpha_src: u32,
    pub alpha_dst: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blend {
    /// Per target: blending enabled.
    pub enabled: [bool; TARGETS],
    /// One equation for all targets, unless `independent` is set.
    pub independent: bool,
    pub equation: [BlendEquation; TARGETS],
    pub constant: [f32; 4],
    /// Per target: the write mask, bit 0 red to bit 3 alpha.
    pub write_mask: [u8; TARGETS],
    pub logic_op: bool,
    pub logic_op_func: u32,
}

/// A vertex attribute (a shader input location) and where its data is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    /// The stream the data comes from.
    pub stream: u32,
    /// `false` when the attribute is switched off (the shader reads a constant).
    pub active: bool,
    /// Byte offset in the vertex.
    pub offset: u32,
    /// `COMPONENT_BIT_WIDTHS` (the layout of the components, e.g. 1: four 32-bit).
    pub widths: u32,
    /// `NUMERICAL_TYPE`: 1 snorm, 2 unorm, 3 sint, 4 uint, 5 uscaled, 6 sscaled, 7 float.
    pub numeric: u32,
    /// The red and blue components are swapped.
    pub bgra: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexStream {
    pub enabled: bool,
    pub stride: u32,
    pub address: u64,
    /// One past the last byte the stream may read.
    pub limit: u64,
    /// The instance divisor: 0 when per vertex.
    pub frequency: u32,
    pub instanced: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexBuffer {
    pub address: u64,
    pub limit: u64,
    /// Bytes per index: 1, 2 or 4.
    pub size: u32,
    pub first: u32,
    pub restart: bool,
    pub restart_index: u32,
}

/// The pools of texture headers and samplers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexturePools {
    pub header_address: u64,
    pub header_count: u32,
    pub sampler_address: u64,
    pub sampler_count: u32,
    /// The constant buffer slot that holds the texture handles of "bound" textures.
    pub bindless_slot: u32,
}

/// How the window and the transform behave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// The origin is the lower left corner.
    pub lower_left: bool,
    /// Flip y (bit 4 of `SET_WINDOW_ORIGIN`).
    pub flip_y: bool,
    pub offset: [u32; 2],
    /// The viewport transform of the registers applies (else clip space is the window).
    pub transform_enabled: bool,
    /// Clip space z runs from 0 to 1 (else from -1 to 1): bit 0 of `SET_VIEWPORT_CLIP_CONTROL`.
    pub z_zero_to_one: bool,
    pub pixel_center_integer: bool,
}

/// What a clear writes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearValues {
    /// The raw words of `SET_COLOR_CLEAR_VALUE`.
    pub color: [u32; 4],
    pub depth: f32,
    pub stencil: u32,
    pub use_scissor0: bool,
    pub use_clear_rect: bool,
    /// `(x_min, x_max, y_min, y_max)` of the clear rectangle.
    pub rect: [u32; 4],
}

impl Regs<'_> {
    pub fn color_target(&self, j: usize) -> ColorTarget {
        let base = r::COLOR_TARGET + j as u32 * r::COLOR_TARGET_STRIDE;
        let memory = self.word(base + 0x14);
        ColorTarget {
            address: self.address(base),
            width: self.field(base + 8, 0, 28),
            height: self.field(base + 0xC, 0, 17),
            format: self.field(base + 0x10, 0, 8),
            layout: Layout::from_memory_word(memory),
            depth: self.field(base + 0x18, 0, 28),
            depth_is_size: memory >> 16 & 1 != 0,
            array_pitch: self.word(base + 0x1C),
            layer: self.field(base + 0x20, 0, 16),
        }
    }

    /// The color targets the pixel shader writes to, in the order of its outputs: the mapping of
    /// `SET_CT_SELECT` applied to the targets that are enabled.
    pub fn color_targets_in_use(&self) -> Vec<(usize, ColorTarget)> {
        let select = self.word(r::CT_SELECT);
        let count = (select & 0xF) as usize;
        (0..count.min(TARGETS))
            .map(|i| {
                let j = (select >> (4 + 3 * i) & 7) as usize;
                (i, self.color_target(j))
            })
            .collect()
    }

    /// `(x, width, y, height)` of the surface clip rectangle.
    pub fn surface_clip(&self) -> [u32; 4] {
        [
            self.field(r::SURFACE_CLIP_HORIZONTAL, 0, 16),
            self.field(r::SURFACE_CLIP_HORIZONTAL, 16, 16),
            self.field(r::SURFACE_CLIP_VERTICAL, 0, 16),
            self.field(r::SURFACE_CLIP_VERTICAL, 16, 16),
        ]
    }

    /// The depth target, when `SET_ZT_SELECT` says there is one.
    pub fn depth_target(&self) -> Option<DepthTarget> {
        if self.field(r::ZT_SELECT, 0, 1) == 0 {
            return None;
        }
        let block = self.word(r::ZT_BLOCK_SIZE);
        let size_c = self.word(r::ZT_SIZE_C);
        Some(DepthTarget {
            address: self.address(r::ZT_A),
            width: self.field(r::ZT_SIZE_A, 0, 28),
            height: self.field(r::ZT_SIZE_B, 0, 17),
            format: self.field(r::ZT_FORMAT, 0, 5),
            layout: Layout::BlockLinear {
                block: [block & 0xF, block >> 4 & 0xF, block >> 8 & 0xF],
            },
            layers: if size_c >> 16 & 1 != 0 {
                1
            } else {
                size_c & 0xFFFF
            },
            array_pitch: self.word(r::ZT_ARRAY_PITCH),
            layer: self.field(r::ZT_LAYER, 0, 16),
        })
    }

    pub fn viewport(&self, j: usize) -> Viewport {
        let s = r::VIEWPORT_SCALE + j as u32 * 32;
        let c = r::VIEWPORT_CLIP + j as u32 * 16;
        Viewport {
            scale: [self.float(s), self.float(s + 4), self.float(s + 8)],
            offset: [
                self.float(s + 0xC),
                self.float(s + 0x10),
                self.float(s + 0x14),
            ],
            clip_x: self.field(c, 0, 16),
            clip_width: self.field(c, 16, 16),
            clip_y: self.field(c + 4, 0, 16),
            clip_height: self.field(c + 4, 16, 16),
            min_z: self.float(c + 8),
            max_z: self.float(c + 0xC),
        }
    }

    pub fn scissor(&self, j: usize) -> Scissor {
        let b = r::SCISSOR + j as u32 * 16;
        Scissor {
            enabled: self.flag(b),
            x_min: self.field(b + 4, 0, 16),
            x_max: self.field(b + 4, 16, 16),
            y_min: self.field(b + 8, 0, 16),
            y_max: self.field(b + 8, 16, 16),
        }
    }

    pub fn window(&self) -> Window {
        let origin = self.word(r::WINDOW_ORIGIN);
        let clip = self.word(r::VIEWPORT_CLIP_CONTROL);
        Window {
            lower_left: origin & 1 != 0,
            flip_y: origin >> 4 & 1 != 0,
            offset: [
                self.word(r::WINDOW_OFFSET_X),
                self.word(r::WINDOW_OFFSET_X + 4),
            ],
            transform_enabled: self.flag(r::VIEWPORT_SCALE_OFFSET),
            z_zero_to_one: clip & 1 != 0,
            pixel_center_integer: self.flag(0x1924),
        }
    }

    pub fn raster(&self) -> Raster {
        let aliased = self.flag(r::ALIASED_LINE_WIDTH_ENABLE);
        Raster {
            enabled: self.flag(r::RASTER_ENABLE),
            cull: self.flag(r::CULL),
            cull_face: self.word(r::CULL_FACE),
            front_face: self.word(r::FRONT_FACE),
            front_mode: self.word(r::FRONT_POLYGON_MODE),
            back_mode: self.word(r::BACK_POLYGON_MODE),
            line_width: self.float(if aliased {
                r::ALIASED_LINE_WIDTH
            } else {
                r::LINE_WIDTH
            }),
            point_size: self.float(r::POINT_SIZE),
            depth_bias: self.float(r::DEPTH_BIAS),
            slope_scale_depth_bias: self.float(r::SLOPE_SCALE_DEPTH_BIAS),
            depth_bias_clamp: self.float(r::DEPTH_BIAS_CLAMP),
            bias_fill: self.flag(r::POLY_OFFSET_FILL),
            bias_line: self.flag(r::POLY_OFFSET_LINE),
            bias_point: self.flag(r::POLY_OFFSET_POINT),
            provoking_last: self.flag(r::PROVOKING_VERTEX),
            depth_clamp: self.word(r::VIEWPORT_CLIP_CONTROL) >> 3 & 3 == 3,
        }
    }

    pub fn depth_stencil(&self) -> DepthStencil {
        let face = |ops: u32, reference: u32, func_mask: u32, mask: u32| StencilFace {
            fail: self.word(ops),
            z_fail: self.word(ops + 4),
            z_pass: self.word(ops + 8),
            func: self.word(ops + 12),
            reference: self.field(reference, 0, 8),
            func_mask: self.field(func_mask, 0, 8),
            write_mask: self.field(mask, 0, 8),
        };
        DepthStencil {
            depth_test: self.flag(r::DEPTH_TEST),
            depth_write: self.flag(r::DEPTH_WRITE),
            depth_func: self.word(r::DEPTH_FUNC),
            bounds_test: self.flag(r::DEPTH_BOUNDS_TEST),
            bounds: [
                self.float(r::DEPTH_BOUNDS_MIN),
                self.float(r::DEPTH_BOUNDS_MAX),
            ],
            stencil_test: self.flag(r::STENCIL_TEST),
            two_sided: self.flag(r::STENCIL_TWO_SIDED),
            front: face(
                r::STENCIL_FRONT,
                r::STENCIL_FRONT_REF,
                r::STENCIL_FRONT_FUNC_MASK,
                r::STENCIL_FRONT_MASK,
            ),
            back: face(
                r::STENCIL_BACK,
                r::STENCIL_BACK_REF,
                r::STENCIL_BACK_FUNC_MASK,
                r::STENCIL_BACK_MASK,
            ),
        }
    }

    pub fn blend(&self) -> Blend {
        let independent = self.flag(r::BLEND_STATE_PER_TARGET);
        let shared = BlendEquation {
            separate_alpha: self.flag(r::BLEND_SEPARATE_FOR_ALPHA),
            color_op: self.word(r::BLEND_COLOR_OP),
            color_src: self.word(r::BLEND_COLOR_OP + 4),
            color_dst: self.word(r::BLEND_COLOR_OP + 8),
            alpha_op: self.word(r::BLEND_COLOR_OP + 12),
            alpha_src: self.word(r::BLEND_COLOR_OP + 16),
            alpha_dst: self.word(r::BLEND_COLOR_OP + 24),
        };
        let mut equation = [shared; TARGETS];
        let mut enabled = [false; TARGETS];
        let mut write_mask = [0u8; TARGETS];
        for j in 0..TARGETS {
            enabled[j] = self.flag(r::BLEND + 4 * j as u32);
            let w = self.word(r::CT_WRITE + 4 * j as u32);
            write_mask[j] =
                (w & 1 | (w >> 4 & 1) << 1 | (w >> 8 & 1) << 2 | (w >> 12 & 1) << 3) as u8;
            if independent {
                let b = r::BLEND_PER_TARGET + 32 * j as u32;
                equation[j] = BlendEquation {
                    separate_alpha: self.flag(b),
                    color_op: self.word(b + 4),
                    color_src: self.word(b + 8),
                    color_dst: self.word(b + 12),
                    alpha_op: self.word(b + 16),
                    alpha_src: self.word(b + 20),
                    alpha_dst: self.word(b + 24),
                };
            }
        }
        Blend {
            enabled,
            independent,
            equation,
            constant: [
                self.float(r::BLEND_CONSTANT),
                self.float(r::BLEND_CONSTANT + 4),
                self.float(r::BLEND_CONSTANT + 8),
                self.float(r::BLEND_CONSTANT + 12),
            ],
            write_mask,
            logic_op: self.flag(r::LOGIC_OP),
            logic_op_func: self.word(r::LOGIC_OP_FUNC),
        }
    }

    pub fn vertex_attribute(&self, i: usize) -> VertexAttribute {
        let w = self.word(r::VERTEX_ATTRIBUTE + 4 * i as u32);
        VertexAttribute {
            stream: w & 0x1F,
            active: w >> 6 & 1 == 0,
            offset: w >> 7 & 0x3FFF,
            widths: w >> 21 & 0x3F,
            numeric: w >> 27 & 7,
            bgra: w >> 31 & 1 != 0,
        }
    }

    pub fn vertex_stream(&self, s: usize) -> VertexStream {
        let (base, instance, limit, k) = if s < 16 {
            (
                r::VERTEX_STREAM_A,
                r::VERTEX_STREAM_INSTANCE_A,
                r::VERTEX_STREAM_LIMIT_A,
                s as u32,
            )
        } else {
            (
                r::VERTEX_STREAM_B,
                r::VERTEX_STREAM_INSTANCE_B,
                r::VERTEX_STREAM_LIMIT_B,
                s as u32 - 16,
            )
        };
        let b = base + 16 * k;
        VertexStream {
            enabled: self.word(b) >> 12 & 1 != 0,
            stride: self.field(b, 0, 12),
            address: self.address(b + 4),
            limit: self.address(limit + 8 * k),
            frequency: self.word(b + 12),
            instanced: self.flag(instance + 4 * k),
        }
    }

    pub fn index_buffer(&self) -> IndexBuffer {
        IndexBuffer {
            address: self.address(r::INDEX_BUFFER_A),
            limit: self.address(r::INDEX_BUFFER_A + 8),
            size: 1 << self.field(r::INDEX_BUFFER_A + 16, 0, 2),
            first: self.word(r::INDEX_BUFFER_A + 20),
            restart: self.flag(r::PRIMITIVE_RESTART),
            restart_index: self.word(r::PRIMITIVE_RESTART_INDEX),
        }
    }

    /// `(base vertex, base instance, vertex id base)`.
    pub fn bases(&self) -> (u32, u32, u32) {
        (
            self.word(r::GLOBAL_BASE_VERTEX_INDEX),
            self.word(r::GLOBAL_BASE_INSTANCE_INDEX),
            self.word(r::VERTEX_ID_BASE),
        )
    }

    pub fn texture_pools(&self) -> TexturePools {
        TexturePools {
            header_address: self.address(r::TEX_HEADER_POOL),
            header_count: self.field(r::TEX_HEADER_POOL + 8, 0, 22) + 1,
            sampler_address: self.address(r::TEX_SAMPLER_POOL),
            sampler_count: self.field(r::TEX_SAMPLER_POOL + 8, 0, 20) + 1,
            bindless_slot: self.field(r::BINDLESS_TEXTURE, 0, 5),
        }
    }

    pub fn clear_values(&self) -> ClearValues {
        let control = self.word(r::CLEAR_SURFACE_CONTROL);
        ClearValues {
            color: [
                self.word(r::COLOR_CLEAR_VALUE),
                self.word(r::COLOR_CLEAR_VALUE + 4),
                self.word(r::COLOR_CLEAR_VALUE + 8),
                self.word(r::COLOR_CLEAR_VALUE + 12),
            ],
            depth: self.float(r::Z_CLEAR_VALUE),
            stencil: self.field(r::STENCIL_CLEAR_VALUE, 0, 8),
            use_scissor0: control >> 8 & 1 != 0,
            use_clear_rect: control >> 4 & 1 != 0,
            rect: [
                self.field(r::CLEAR_RECT_HORIZONTAL, 0, 16),
                self.field(r::CLEAR_RECT_HORIZONTAL, 16, 16),
                self.field(r::CLEAR_RECT_VERTICAL, 0, 16),
                self.field(r::CLEAR_RECT_VERTICAL, 16, 16),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> Vec<u32> {
        vec![0; 0x1000]
    }

    #[test]
    fn a_color_target_reads_its_address_size_format_and_layout() {
        let mut f = file();
        // Target 2: 0x800 + 2 * 64 = 0x880.
        f[0x880 / 4] = 0x2;
        f[0x884 / 4] = 0x8100_0000;
        f[0x888 / 4] = 1600;
        f[0x88C / 4] = 900;
        f[0x890 / 4] = 0xCF;
        // Block linear, 1 GOB wide, 16 GOBs (4) high, 1 GOB deep.
        f[0x894 / 4] = 4 << 4;
        f[0x898 / 4] = 1;
        f[0x89C / 4] = 0x1_0000;
        f[0x8A0 / 4] = 3;
        let t = Regs(&f).color_target(2);
        assert_eq!(t.address, 0x2_8100_0000);
        assert_eq!((t.width, t.height, t.format), (1600, 900, 0xCF));
        assert_eq!(t.layout, Layout::BlockLinear { block: [0, 4, 0] });
        assert_eq!((t.depth, t.depth_is_size), (1, false));
        assert_eq!((t.array_pitch, t.layer), (0x1_0000, 3));
        // Pitch linear: bit 12 of the memory word.
        f[0x894 / 4] = 1 << 12;
        assert_eq!(Regs(&f).color_target(2).layout, Layout::Pitch);
    }

    #[test]
    fn the_color_target_select_maps_outputs_to_targets() {
        let mut f = file();
        // Two targets: output 0 goes to target 3, output 1 to target 0.
        f[0x121c / 4] = 2 | 3 << 4;
        f[(0x800 + 3 * 64 + 0x10) / 4] = 0xD5;
        f[(0x800 + 0x10) / 4] = 0xCF;
        let in_use = Regs(&f).color_targets_in_use();
        assert_eq!(in_use.len(), 2);
        assert_eq!((in_use[0].0, in_use[0].1.format), (0, 0xD5));
        assert_eq!((in_use[1].0, in_use[1].1.format), (1, 0xCF));
    }

    #[test]
    fn the_depth_target_exists_only_when_selected() {
        let mut f = file();
        f[0xfe0 / 4] = 0x1;
        f[0xfe4 / 4] = 0x9000_0000;
        f[0xfe8 / 4] = 0x14;
        f[0xfec / 4] = 3 << 4;
        f[0x1228 / 4] = 1280;
        f[0x122c / 4] = 720;
        f[0x1230 / 4] = 1 << 16;
        assert!(Regs(&f).depth_target().is_none());
        f[0x1538 / 4] = 1;
        let z = Regs(&f).depth_target().unwrap();
        assert_eq!(z.address, 0x1_9000_0000);
        assert_eq!(
            (z.width, z.height, z.format, z.layers),
            (1280, 720, 0x14, 1)
        );
        assert_eq!(z.layout, Layout::BlockLinear { block: [0, 3, 0] });
    }

    #[test]
    fn viewports_scissors_and_the_window_are_read_per_index() {
        let mut f = file();
        // Viewport 1: scale at 0xa00 + 32, clip at 0xc00 + 16.
        f[(0xa00 + 32) / 4] = 800.0f32.to_bits();
        f[(0xa00 + 36) / 4] = (-450.0f32).to_bits();
        f[(0xa00 + 40) / 4] = 0.5f32.to_bits();
        f[(0xa00 + 44) / 4] = 800.0f32.to_bits();
        f[(0xa00 + 48) / 4] = 450.0f32.to_bits();
        f[(0xa00 + 52) / 4] = 0.5f32.to_bits();
        f[(0xc00 + 16) / 4] = 10 | 1600 << 16;
        f[(0xc00 + 20) / 4] = 20 | 900 << 16;
        f[(0xc00 + 24) / 4] = 0.0f32.to_bits();
        f[(0xc00 + 28) / 4] = 1.0f32.to_bits();
        let v = Regs(&f).viewport(1);
        assert_eq!(v.scale, [800.0, -450.0, 0.5]);
        assert_eq!(v.offset, [800.0, 450.0, 0.5]);
        assert_eq!(
            (v.clip_x, v.clip_width, v.clip_y, v.clip_height),
            (10, 1600, 20, 900)
        );
        assert_eq!((v.min_z, v.max_z), (0.0, 1.0));
        // Scissor 3.
        f[(0xe00 + 48) / 4] = 1;
        f[(0xe04 + 48) / 4] = 5 | 100 << 16;
        f[(0xe08 + 48) / 4] = 6 | 200 << 16;
        let s = Regs(&f).scissor(3);
        assert_eq!(
            (s.enabled, s.x_min, s.x_max, s.y_min, s.y_max),
            (true, 5, 100, 6, 200)
        );
        assert!(!Regs(&f).scissor(2).enabled);
        // Window origin: lower left with y flipped.
        f[0x13ac / 4] = 1 | 1 << 4;
        f[0x192c / 4] = 1;
        let w = Regs(&f).window();
        assert!(w.lower_left && w.flip_y && w.transform_enabled && !w.z_zero_to_one);
        assert!(!Regs(&f).raster().depth_clamp);
    }

    #[test]
    fn the_clip_control_register_is_read_field_by_field() {
        let mut f = file();
        // Other fields set (a guardband and a clip mode) do not make the z range 0 to 1.
        f[0x193c / 4] = 1 << 7 | 3 << 11;
        let w = Regs(&f).window();
        assert!(!w.z_zero_to_one && !Regs(&f).raster().depth_clamp);
        f[0x193c / 4] = 1;
        assert!(Regs(&f).window().z_zero_to_one);
        // Depth is clamped when both ends say so, not for one.
        f[0x193c / 4] = 1 << 3;
        assert!(!Regs(&f).raster().depth_clamp);
        f[0x193c / 4] = 1 << 3 | 1 << 4;
        let w = Regs(&f).window();
        assert!(Regs(&f).raster().depth_clamp && !w.z_zero_to_one);
    }

    #[test]
    fn blend_state_is_shared_or_per_target() {
        let mut f = file();
        f[0x1360 / 4] = 1;
        f[(0x1360 + 8) / 4] = 1;
        f[0x133c / 4] = 1;
        f[0x1340 / 4] = 0x8006;
        f[0x1344 / 4] = 0x4302;
        f[0x1348 / 4] = 0x4303;
        f[0x134c / 4] = 0x8006;
        f[0x1350 / 4] = 0x4001;
        f[0x1358 / 4] = 0x4000;
        // Target 0 may write red and alpha only; target 1 everything.
        f[0x1a00 / 4] = 1 | 1 << 12;
        f[0x1a04 / 4] = 1 | 1 << 4 | 1 << 8 | 1 << 12;
        let b = Regs(&f).blend();
        assert_eq!(
            b.enabled,
            [true, false, true, false, false, false, false, false]
        );
        assert!(!b.independent);
        let e = b.equation[5];
        assert!(e.separate_alpha);
        assert_eq!(
            (
                e.color_op,
                e.color_src,
                e.color_dst,
                e.alpha_op,
                e.alpha_src,
                e.alpha_dst
            ),
            (0x8006, 0x4302, 0x4303, 0x8006, 0x4001, 0x4000)
        );
        assert_eq!(b.write_mask[0], 0b1001);
        assert_eq!(b.write_mask[1], 0b1111);
        // Independent blending reads the per-target registers instead.
        f[0x12e4 / 4] = 1;
        f[(0x1e00 + 32 + 4) / 4] = 0x800A;
        let b = Regs(&f).blend();
        assert!(b.independent);
        assert_eq!(b.equation[1].color_op, 0x800A);
        assert_eq!(
            b.equation[0].color_op, 0,
            "target 0's registers were not written"
        );
    }

    #[test]
    fn vertex_attributes_and_streams_decode() {
        let mut f = file();
        // Attribute 3: stream 2, active, offset 12, four 32-bit components (1), float (7).
        f[(0x1160 + 12) / 4] = 2 | 12 << 7 | 1 << 21 | 7 << 27;
        let a = Regs(&f).vertex_attribute(3);
        assert_eq!(
            a,
            VertexAttribute {
                stream: 2,
                active: true,
                offset: 12,
                widths: 1,
                numeric: 7,
                bgra: false
            }
        );
        f[(0x1160 + 16) / 4] = 1 << 6 | 1 << 31;
        let a = Regs(&f).vertex_attribute(4);
        assert!(!a.active && a.bgra);
        // Stream 2: stride 32, enabled; address; limit; divisor.
        f[(0x1c00 + 32) / 4] = 32 | 1 << 12;
        f[(0x1c00 + 36) / 4] = 0x3;
        f[(0x1c00 + 40) / 4] = 0x1000;
        f[(0x1c00 + 44) / 4] = 4;
        f[(0x1f00 + 16) / 4] = 0x3;
        f[(0x1f00 + 20) / 4] = 0x2000;
        f[(0x1880 + 8) / 4] = 1;
        let s = Regs(&f).vertex_stream(2);
        assert_eq!(
            s,
            VertexStream {
                enabled: true,
                stride: 32,
                address: 0x3_0000_1000,
                limit: 0x3_0000_2000,
                frequency: 4,
                instanced: true
            }
        );
        // Streams from 16 are in the second array.
        f[(0x1d00 + 16) / 4] = 8 | 1 << 12;
        assert_eq!(Regs(&f).vertex_stream(17).stride, 8);
    }

    #[test]
    fn the_index_buffer_and_the_texture_pools_decode() {
        let mut f = file();
        f[0x17c8 / 4] = 0x2;
        f[0x17cc / 4] = 0x4000;
        f[0x17d0 / 4] = 0x2;
        f[0x17d4 / 4] = 0x5000;
        f[0x17d8 / 4] = 1;
        f[0x17dc / 4] = 9;
        f[0x1644 / 4] = 1;
        f[0x1648 / 4] = 0xFFFF;
        let i = Regs(&f).index_buffer();
        assert_eq!(
            (i.address, i.limit, i.size, i.first),
            (0x2_0000_4000, 0x2_0000_5000, 2, 9)
        );
        assert!(i.restart && i.restart_index == 0xFFFF);
        f[0x1574 / 4] = 0x1;
        f[0x1578 / 4] = 0x8000;
        f[0x157c / 4] = 0xFF;
        f[0x155c / 4] = 0x1;
        f[0x1560 / 4] = 0x9000;
        f[0x1564 / 4] = 0x3F;
        f[0x2608 / 4] = 3;
        let p = Regs(&f).texture_pools();
        assert_eq!(
            p,
            TexturePools {
                header_address: 0x1_0000_8000,
                header_count: 256,
                sampler_address: 0x1_0000_9000,
                sampler_count: 64,
                bindless_slot: 3
            }
        );
    }

    #[test]
    fn rasterizer_depth_and_stencil_state_decode() {
        let mut f = file();
        f[0x1918 / 4] = 1;
        f[0x1920 / 4] = 0x405;
        f[0x191c / 4] = 0x901;
        f[0xdac / 4] = 0x1B02;
        f[0x13b0 / 4] = 2.5f32.to_bits();
        f[0x12cc / 4] = 1;
        f[0x12e8 / 4] = 1;
        f[0x130c / 4] = 0x203;
        f[0x1380 / 4] = 1;
        f[0x1384 / 4] = 0x1E00;
        f[0x1394 / 4] = 0x80;
        f[0x1398 / 4] = 0xF0;
        f[0x139c / 4] = 0xFF;
        let r = Regs(&f).raster();
        assert!(r.cull);
        assert_eq!(
            (r.cull_face, r.front_face, r.front_mode),
            (0x405, 0x901, 0x1B02)
        );
        assert_eq!(r.line_width, 2.5);
        let d = Regs(&f).depth_stencil();
        assert!(d.depth_test && d.depth_write && d.stencil_test && !d.two_sided);
        assert_eq!(d.depth_func, 0x203);
        assert_eq!(
            (
                d.front.fail,
                d.front.reference,
                d.front.func_mask,
                d.front.write_mask
            ),
            (0x1E00, 0x80, 0xF0, 0xFF)
        );
    }

    #[test]
    fn clear_values_carry_the_rect_and_the_scissor_choice() {
        let mut f = file();
        f[0xd80 / 4] = 1.0f32.to_bits();
        f[0xd84 / 4] = 0.5f32.to_bits();
        f[0xd90 / 4] = 0.25f32.to_bits();
        f[0xda0 / 4] = 0x7F;
        f[0x10f8 / 4] = 1 << 4 | 1 << 8;
        f[0xd6c / 4] = 1 | 640 << 16;
        f[0xd70 / 4] = 2 | 480 << 16;
        let c = Regs(&f).clear_values();
        assert_eq!(f32::from_bits(c.color[0]), 1.0);
        assert_eq!(f32::from_bits(c.color[1]), 0.5);
        assert_eq!((c.depth, c.stencil), (0.25, 0x7F));
        assert!(c.use_clear_rect && c.use_scissor0);
        assert_eq!(c.rect, [1, 640, 2, 480]);
    }
}
