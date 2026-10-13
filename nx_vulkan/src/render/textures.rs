//! Textures: the headers the shaders' handles name, the memory behind them, and the Vulkan
//! images and samplers made from both.
//!
//! A shader names a texture by a handle in a constant buffer; the low 20 bits index the pool of
//! texture headers (32 bytes each) and the bits above them the pool of samplers. Layouts are
//! those of NVIDIA's `clb097tex.h` (open-gpu-doc, MIT licence), where bit positions count over
//! the whole 256-bit header.

use super::frame::{Frame, Garbage};
use crate::blocklinear::{deswizzle, Geometry};
use crate::device::Vk;
use crate::image::{aspect_of, Image, ImageInfo};
use ash::vk;
use nx_gpu::render::Scene;
use nx_gpu::shaders::hash_bytes;
use nx_shader::translate::{Dim, TextureUse};
use std::collections::HashMap;

/// A texture header: eight words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tic(pub [u32; 8]);

impl Tic {
    /// `len` bits (at most 32) from bit `lo` of the header.
    pub fn bits(&self, lo: usize, len: usize) -> u32 {
        debug_assert!(len <= 32 && lo + len <= 256);
        let first = lo / 32;
        let mut v = u128::from(self.0[first]);
        for (i, w) in self.0.iter().enumerate().skip(first + 1).take(2) {
            v |= u128::from(*w) << (32 * (i - first));
        }
        ((v >> (lo % 32)) & ((1u128 << len) - 1)) as u32
    }

    pub fn components(&self) -> u32 {
        self.bits(0, 7)
    }

    /// The data type of the red, green, blue and alpha components.
    pub fn types(&self) -> [u32; 4] {
        [
            self.bits(7, 3),
            self.bits(10, 3),
            self.bits(13, 3),
            self.bits(16, 3),
        ]
    }

    /// What the shader's x, y, z and w components are made from (0 zero, 2..5 r, g, b, a, 6 and 7
    /// one).
    pub fn sources(&self) -> [u32; 4] {
        [
            self.bits(19, 3),
            self.bits(22, 3),
            self.bits(25, 3),
            self.bits(28, 3),
        ]
    }

    /// 0 one-dimensional buffer, 1 pitch with a color key, 2 pitch, 3 block linear, 4 block linear
    /// with a color key.
    pub fn version(&self) -> u32 {
        self.bits(85, 3)
    }

    pub fn is_block_linear(&self) -> bool {
        matches!(self.version(), 3 | 4)
    }

    pub fn is_pitch(&self) -> bool {
        matches!(self.version(), 1 | 2)
    }

    /// The GPU address of the first byte.
    pub fn address(&self) -> u64 {
        let high = u64::from(self.bits(64, 16)) << 32;
        match self.version() {
            3 | 4 => u64::from(self.bits(41, 23)) << 9 | high,
            1 | 2 => u64::from(self.bits(37, 27)) << 5 | high,
            _ => u64::from(self.bits(32, 32)) | high,
        }
    }

    /// The size of a block of a block-linear texture, in GOBs, as powers of two (width, height,
    /// depth).
    pub fn block_log2(&self) -> [u32; 3] {
        [self.bits(96, 3), self.bits(99, 3), self.bits(102, 3)]
    }

    /// Bytes from one row to the next of a pitch-linear texture.
    pub fn pitch(&self) -> u32 {
        self.bits(96, 16) << 5
    }

    pub fn srgb(&self) -> bool {
        self.bits(150, 1) != 0
    }

    /// 0 1D, 1 2D, 2 3D, 3 cube, 4 1D array, 5 2D array, 6 1D buffer, 7 2D without mips, 8 cube
    /// array.
    pub fn texture_type(&self) -> u32 {
        self.bits(151, 4)
    }

    pub fn width(&self) -> u32 {
        if self.version() == 0 {
            (self.bits(96, 16) << 16 | self.bits(128, 16)) + 1
        } else {
            self.bits(128, 16) + 1
        }
    }

    pub fn height(&self) -> u32 {
        self.bits(160, 16) + 1
    }

    /// The depth of a 3D texture or the layers of an array.
    pub fn depth(&self) -> u32 {
        self.bits(176, 14) + 1
    }

    /// The number of mip levels.
    pub fn levels(&self) -> u32 {
        self.bits(124, 4) + 1
    }

    /// What the view of the texture makes of the components.
    pub fn component_mapping(&self) -> vk::ComponentMapping {
        let s = self.sources();
        vk::ComponentMapping {
            r: swizzle(s[0]),
            g: swizzle(s[1]),
            b: swizzle(s[2]),
            a: swizzle(s[3]),
        }
    }
}

/// A sampler header: eight words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tsc(pub [u32; 8]);

fn address_mode(v: u32) -> vk::SamplerAddressMode {
    match v {
        0 => vk::SamplerAddressMode::REPEAT,
        1 => vk::SamplerAddressMode::MIRRORED_REPEAT,
        3 | 6 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        5 | 7 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        _ => vk::SamplerAddressMode::CLAMP_TO_EDGE,
    }
}

impl Tsc {
    fn sampler_info(&self, vk: &Vk) -> (vk::SamplerCreateInfo<'static>, [f32; 4]) {
        let w = &self.0;
        let mag = w[1] & 7;
        let min = w[1] >> 4 & 3;
        let mip = w[1] >> 6 & 3;
        let aniso: f32 = match w[0] >> 20 & 7 {
            0 => 1.0,
            1 => 2.0,
            2 => 4.0,
            3 => 6.0,
            4 => 8.0,
            5 => 10.0,
            6 => 12.0,
            _ => 16.0,
        };
        let anisotropic = min == 3 && vk.caps.sampler_anisotropy;
        // Fixed point with 8 fractional bits: a signed 13-bit bias, unsigned 12-bit limits.
        let bias_raw = (w[1] >> 12 & 0x1FFF) as i32;
        let bias = ((bias_raw << 19) >> 19) as f32 / 256.0;
        let min_lod = (w[2] & 0xFFF) as f32 / 256.0;
        let max_lod = (w[2] >> 12 & 0xFFF) as f32 / 256.0;
        let border = [
            f32::from_bits(w[4]),
            f32::from_bits(w[5]),
            f32::from_bits(w[6]),
            f32::from_bits(w[7]),
        ];
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(if mag == 1 {
                vk::Filter::NEAREST
            } else {
                vk::Filter::LINEAR
            })
            .min_filter(if min == 1 {
                vk::Filter::NEAREST
            } else {
                vk::Filter::LINEAR
            })
            .mipmap_mode(if mip == 3 {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            })
            .address_mode_u(address_mode(w[0] & 7))
            .address_mode_v(address_mode(w[0] >> 3 & 7))
            .address_mode_w(address_mode(w[0] >> 6 & 7))
            .mip_lod_bias(bias)
            .anisotropy_enable(anisotropic)
            .max_anisotropy(if anisotropic {
                aniso.min(vk.caps.max_sampler_anisotropy)
            } else {
                1.0
            })
            .min_lod(min_lod)
            // No mip chain: the level of detail stays on the base level (and with `MIP_NONE`
            // the hardware does not look at the others).
            .max_lod(if mip == 1 {
                min_lod
            } else {
                max_lod.max(min_lod)
            });
        if w[0] >> 9 & 1 != 0 {
            let func = match w[0] >> 10 & 7 {
                0 => vk::CompareOp::NEVER,
                1 => vk::CompareOp::LESS,
                2 => vk::CompareOp::EQUAL,
                3 => vk::CompareOp::LESS_OR_EQUAL,
                4 => vk::CompareOp::GREATER,
                5 => vk::CompareOp::NOT_EQUAL,
                6 => vk::CompareOp::GREATER_OR_EQUAL,
                _ => vk::CompareOp::ALWAYS,
            };
            info = info.compare_enable(true).compare_op(func);
        }
        let colour = if border == [0.0, 0.0, 0.0, 0.0] {
            vk::BorderColor::FLOAT_TRANSPARENT_BLACK
        } else if border == [0.0, 0.0, 0.0, 1.0] {
            vk::BorderColor::FLOAT_OPAQUE_BLACK
        } else {
            vk::BorderColor::FLOAT_OPAQUE_WHITE
        };
        (info.border_color(colour), border)
    }
}

/// A texture format: the Vulkan format of the image made of it and how texels are grouped in
/// memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexFormat {
    pub vk: vk::Format,
    /// Texels per element across and down (4x4 for the block compressed formats).
    pub block: (u32, u32),
    /// Bytes per element in memory.
    pub bytes: u32,
    /// The memory holds ASTC blocks, which are decoded on the host: `vk` is the (uncompressed)
    /// format of the decoded texels.
    pub astc: bool,
}

const fn tf(vk: vk::Format, block: (u32, u32), bytes: u32) -> Option<TexFormat> {
    Some(TexFormat {
        vk,
        block,
        bytes,
        astc: false,
    })
}

/// The format of a texture: `components` is the header's format code and `data_type` the type of
/// its first component (1 snorm, 2 unorm, 3 sint, 4 uint, 7 float).
pub fn texture_format(components: u32, data_type: u32, srgb: bool) -> Option<TexFormat> {
    use vk::Format as F;
    // ASTC is decoded here into plain texels, since most devices cannot sample it.
    if let Some(block) = crate::astc::footprint(components) {
        return Some(TexFormat {
            vk: if srgb {
                F::R8G8B8A8_SRGB
            } else {
                F::R8G8B8A8_UNORM
            },
            block,
            bytes: 16,
            astc: true,
        });
    }
    let one = (1, 1);
    let bc = (4, 4);
    let pick = |unorm: Option<F>,
                snorm: Option<F>,
                uint: Option<F>,
                sint: Option<F>,
                float: Option<F>|
     -> Option<F> {
        match data_type {
            1 | 5 => snorm,
            2 | 6 => unorm,
            3 => sint,
            4 => uint,
            7 => float,
            _ => None,
        }
    };
    let n = None;
    let (format, block, bytes) = match components {
        0x01 => (
            pick(
                n,
                n,
                Some(F::R32G32B32A32_UINT),
                Some(F::R32G32B32A32_SINT),
                Some(F::R32G32B32A32_SFLOAT),
            )?,
            one,
            16,
        ),
        0x03 => (
            pick(
                Some(F::R16G16B16A16_UNORM),
                Some(F::R16G16B16A16_SNORM),
                Some(F::R16G16B16A16_UINT),
                Some(F::R16G16B16A16_SINT),
                Some(F::R16G16B16A16_SFLOAT),
            )?,
            one,
            8,
        ),
        0x04 => (
            pick(
                n,
                n,
                Some(F::R32G32_UINT),
                Some(F::R32G32_SINT),
                Some(F::R32G32_SFLOAT),
            )?,
            one,
            8,
        ),
        0x07 | 0x08 => (
            pick(
                Some(if srgb {
                    F::R8G8B8A8_SRGB
                } else {
                    F::R8G8B8A8_UNORM
                }),
                Some(F::R8G8B8A8_SNORM),
                Some(F::R8G8B8A8_UINT),
                Some(F::R8G8B8A8_SINT),
                n,
            )?,
            one,
            4,
        ),
        0x09 => (
            pick(
                Some(F::A2B10G10R10_UNORM_PACK32),
                n,
                Some(F::A2B10G10R10_UINT_PACK32),
                n,
                n,
            )?,
            one,
            4,
        ),
        0x0C => (
            pick(
                Some(F::R16G16_UNORM),
                Some(F::R16G16_SNORM),
                Some(F::R16G16_UINT),
                Some(F::R16G16_SINT),
                Some(F::R16G16_SFLOAT),
            )?,
            one,
            4,
        ),
        0x0F => (
            pick(
                n,
                n,
                Some(F::R32_UINT),
                Some(F::R32_SINT),
                Some(F::R32_SFLOAT),
            )?,
            one,
            4,
        ),
        0x15 => (F::B5G6R5_UNORM_PACK16, one, 2),
        0x18 => (
            pick(
                Some(F::R8G8_UNORM),
                Some(F::R8G8_SNORM),
                Some(F::R8G8_UINT),
                Some(F::R8G8_SINT),
                n,
            )?,
            one,
            2,
        ),
        0x1B => (
            pick(
                Some(F::R16_UNORM),
                Some(F::R16_SNORM),
                Some(F::R16_UINT),
                Some(F::R16_SINT),
                Some(F::R16_SFLOAT),
            )?,
            one,
            2,
        ),
        0x1D => (
            pick(
                Some(F::R8_UNORM),
                Some(F::R8_SNORM),
                Some(F::R8_UINT),
                Some(F::R8_SINT),
                n,
            )?,
            one,
            1,
        ),
        0x20 => (F::E5B9G9R9_UFLOAT_PACK32, one, 4),
        0x21 => (F::B10G11R11_UFLOAT_PACK32, one, 4),
        0x24 => (
            if srgb {
                F::BC1_RGBA_SRGB_BLOCK
            } else {
                F::BC1_RGBA_UNORM_BLOCK
            },
            bc,
            8,
        ),
        0x25 => (
            if srgb {
                F::BC2_SRGB_BLOCK
            } else {
                F::BC2_UNORM_BLOCK
            },
            bc,
            16,
        ),
        0x26 => (
            if srgb {
                F::BC3_SRGB_BLOCK
            } else {
                F::BC3_UNORM_BLOCK
            },
            bc,
            16,
        ),
        0x27 => (
            if data_type == 1 {
                F::BC4_SNORM_BLOCK
            } else {
                F::BC4_UNORM_BLOCK
            },
            bc,
            8,
        ),
        0x28 => (
            if data_type == 1 {
                F::BC5_SNORM_BLOCK
            } else {
                F::BC5_UNORM_BLOCK
            },
            bc,
            16,
        ),
        0x10 => (F::BC6H_SFLOAT_BLOCK, bc, 16),
        0x11 => (F::BC6H_UFLOAT_BLOCK, bc, 16),
        0x17 => (
            if srgb {
                F::BC7_SRGB_BLOCK
            } else {
                F::BC7_UNORM_BLOCK
            },
            bc,
            16,
        ),
        // ZF32: 32-bit floating point depth, sampled as a depth image.
        0x2F => (F::D32_SFLOAT, one, 4),
        _ => return None,
    };
    tf(format, block, bytes)
}

fn swizzle(source: u32) -> vk::ComponentSwizzle {
    match source {
        0 => vk::ComponentSwizzle::ZERO,
        2 => vk::ComponentSwizzle::R,
        3 => vk::ComponentSwizzle::G,
        4 => vk::ComponentSwizzle::B,
        5 => vk::ComponentSwizzle::A,
        _ => vk::ComponentSwizzle::ONE,
    }
}

/// The elements (texels, or blocks of `block` texels) across mip level `level` of a texture that is
/// `texels` wide at level 0: half as many texels each level, and at least one element.
pub fn level_elements(texels: u32, block: u32, level: u32) -> u32 {
    (texels >> level).max(1).div_ceil(block)
}

/// The tiling of mip level `level` of a texture whose level 0 is `base`, of `texels` by `texels`
/// texels in elements of `block` texels: the size of the level, and a block that shrinks while the
/// level is small enough for a smaller one.
pub fn mip_geometry(
    base: &Geometry,
    level: u32,
    texels: (u32, u32),
    block: (u32, u32),
) -> Geometry {
    let mut g = *base;
    g.width = level_elements(texels.0, block.0, level);
    g.height = level_elements(texels.1, block.1, level);
    g.depth = (base.depth >> level).max(1);
    while g.block_height_log2 > 0 && g.height <= (8u32 << g.block_height_log2) / 2 {
        g.block_height_log2 -= 1;
    }
    while g.block_depth_log2 > 0 && g.depth <= (1u32 << g.block_depth_log2) / 2 {
        g.block_depth_log2 -= 1;
    }
    g
}

/// A texture on the device.
struct Texture {
    image: Image,
    /// The frame it was last compared with memory in, and the hash of the bytes then.
    checked: u64,
    hash: u64,
    bytes: u64,
    /// The value of [`Textures::generation`] when it was compared with memory.
    generation: u64,
}

/// How a shader sees a texture: the type of image view the descriptor needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    D2,
    D2Array,
    Cube,
    D3,
}

/// The image made of a texture header for one way of sampling it: the header says what the memory
/// holds, the shader what it wants to see (a 2D array header sampled by a 2D instruction is its
/// first layer, a 2D header sampled as an array is an array of one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub kind: Kind,
    /// Array layers of the image (6 for a cube, 1 for a 3D image or a plain 2D one).
    pub layers: u32,
    /// The depth of a 3D image, else 1.
    pub depth: u32,
}

/// The shape of the image a shader that samples the texture `tic` as `used` needs.
pub fn shape_of(tic: &Tic, used: &TextureUse) -> Result<Shape, String> {
    let ty = tic.texture_type();
    let wrong = || {
        format!(
            "a texture header of type {ty} sampled as a {:?}{} texture (header words {:08x?})",
            used.dim,
            if used.arrayed { " array" } else { "" },
            tic.0
        )
    };
    let shape = |kind, layers, depth| {
        Ok(Shape {
            kind,
            layers,
            depth,
        })
    };
    match (used.dim, used.arrayed) {
        // Plain 2D: a 2D texture, or the first layer of an array or of a cube.
        (Dim::D2, false) if matches!(ty, 1 | 3 | 5 | 7) => shape(Kind::D2, 1, 1),
        (Dim::D2, true) if matches!(ty, 1 | 7) => shape(Kind::D2Array, 1, 1),
        (Dim::D2, true) if ty == 5 => shape(Kind::D2Array, tic.depth(), 1),
        (Dim::Cube, false) if ty == 3 => shape(Kind::Cube, 6, 1),
        (Dim::D3, false) if ty == 2 => shape(Kind::D3, 1, tic.depth()),
        _ => Err(wrong()),
    }
}

/// Where the bytes of a texture are in memory and what image they make.
struct Plan {
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
    levels: u32,
    aspect: vk::ImageAspectFlags,
    /// The geometry of each mip level of one layer.
    level_geometry: Vec<Geometry>,
    /// The distance between layers in bytes.
    layer_stride: u64,
    /// The bytes of the whole texture in memory.
    total: u64,
}

/// The image `tic` makes for a shader that samples it with `shape`, in `format`, and where its
/// bytes are.
fn plan(tic: &Tic, shape: Shape, format: TexFormat) -> Result<Plan, String> {
    let (width, height) = (tic.width(), tic.height());
    let volume = shape.kind == Kind::D3;
    let layers = shape.layers;
    let depth = if volume { shape.depth } else { 1 };
    let largest = if volume {
        width.max(height).max(depth)
    } else {
        width.max(height)
    };
    let levels = tic.levels().min(32 - largest.leading_zeros());
    let aspect = aspect_of(format.vk);
    let elements = (
        width.div_ceil(format.block.0),
        height.div_ceil(format.block.1),
    );
    let base = if tic.is_block_linear() {
        let b = tic.block_log2();
        Geometry {
            width: elements.0,
            height: elements.1,
            depth,
            bytes_per_element: format.bytes,
            block_height_log2: b[1],
            block_depth_log2: if volume { b[2] } else { 0 },
        }
    } else if tic.is_pitch() {
        Geometry {
            width: elements.0,
            height: elements.1,
            depth: 1,
            bytes_per_element: format.bytes,
            block_height_log2: 0,
            block_depth_log2: 0,
        }
    } else {
        return Err(format!("a texture header of version {}", tic.version()));
    };
    // Where the bytes are: each mip level after the one before; layers one after another.
    let level_geometry: Vec<Geometry> = (0..levels)
        .map(|l| mip_geometry(&base, l, (width, height), format.block))
        .collect();
    let layer_size: u64 = if tic.is_block_linear() {
        level_geometry.iter().map(Geometry::tiled_size).sum()
    } else {
        u64::from(tic.pitch()) * u64::from(elements.1)
    };
    // The layers of an array or a cube are a whole block apart (the block of level 0).
    let layer_stride = if tic.is_block_linear() {
        let b = tic.block_log2();
        layer_size.next_multiple_of(512 << (b[1] + b[2]))
    } else {
        layer_size
    };
    Ok(Plan {
        width,
        height,
        depth,
        layers,
        levels,
        aspect,
        level_geometry,
        layer_stride,
        total: layer_stride * u64::from(layers),
    })
}

/// How many bytes of memory the texture `tic` takes when a shader samples it as `used`.
pub fn memory_bytes(tic: &Tic, used: &TextureUse) -> Option<u64> {
    let shape = shape_of(tic, used).ok()?;
    let format = texture_format(tic.components(), tic.types()[0], tic.srgb())?;
    plan(tic, shape, format).ok().map(|p| p.total)
}

#[derive(Default)]
pub struct Textures {
    map: HashMap<(Tic, Shape), Texture>,
    /// Samplers by header and by whether the shader compares against a depth.
    samplers: HashMap<(Tsc, bool), vk::Sampler>,
    /// The number of buffers shown so far: a texture is compared with memory once per frame.
    pub frame: u64,
    /// Say what each new image is made of (`NX_TRACE_TEXTURES`).
    pub trace: bool,
    /// Counts the times memory was written by the renderer itself (a render target put into it): a
    /// texture compared with memory before that is compared again.
    pub generation: u64,
}

/// Everything a draw needs of a texture.
pub struct Bound {
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl Textures {
    pub fn new(trace: bool) -> Textures {
        Textures {
            trace,
            ..Textures::default()
        }
    }

    /// The view and sampler for the headers `tic` and `tsc` that a shader samples as `used`.
    pub fn bind(
        &mut self,
        vk: &Vk,
        frame: &mut Frame,
        scene: &Scene<'_>,
        tic: Tic,
        tsc: Tsc,
        used: &TextureUse,
    ) -> Result<Bound, String> {
        let shape = shape_of(&tic, used)?;
        let view = self.texture(vk, frame, scene, tic, shape)?;
        Ok(Bound {
            view,
            sampler: self.sampler(vk, tsc, used.shadow)?,
        })
    }

    /// As [`Textures::bind`] for a texture whose memory a render target stands for: `view` is the
    /// target's own, already in the layout for sampling, and nothing is read from memory.
    pub fn bind_view(
        &mut self,
        vk: &Vk,
        tic: &Tic,
        tsc: Tsc,
        used: &TextureUse,
        view: vk::ImageView,
    ) -> Result<Bound, String> {
        if shape_of(tic, used)?.kind != Kind::D2 {
            return Err(format!(
                "a render target sampled as a {:?}{} texture",
                used.dim,
                if used.arrayed { " array" } else { "" }
            ));
        }
        Ok(Bound {
            view,
            sampler: self.sampler(vk, tsc, used.shadow)?,
        })
    }

    /// The sampler of a header. A shader that compares against a depth needs a sampler that does
    /// (the header's own function when it has one, else less-or-equal), and one that does not must
    /// have a sampler that does not.
    fn sampler(&mut self, vk: &Vk, tsc: Tsc, shadow: bool) -> Result<vk::Sampler, String> {
        if let Some(s) = self.samplers.get(&(tsc, shadow)) {
            return Ok(*s);
        }
        let (info, border) = tsc.sampler_info(vk);
        let mut info = info;
        if shadow && info.compare_enable == vk::FALSE {
            info = info
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL);
        } else if !shadow {
            info = info.compare_enable(false);
        }
        let mut custom;
        if vk.caps.custom_border_color
            && info.border_color == vk::BorderColor::FLOAT_OPAQUE_WHITE
            && border != [1.0, 1.0, 1.0, 1.0]
        {
            custom = vk::SamplerCustomBorderColorCreateInfoEXT::default()
                .custom_border_color(vk::ClearColorValue { float32: border });
            info = info
                .border_color(vk::BorderColor::FLOAT_CUSTOM_EXT)
                .push_next(&mut custom);
        }
        // SAFETY: valid create info; the sampler lives as long as the cache.
        let s = unsafe { vk.device.create_sampler(&info, None) }
            .map_err(|e| format!("creating a sampler: {e}"))?;
        self.samplers.insert((tsc, shadow), s);
        Ok(s)
    }

    fn texture(
        &mut self,
        vk: &Vk,
        frame: &mut Frame,
        scene: &Scene<'_>,
        tic: Tic,
        shape: Shape,
    ) -> Result<vk::ImageView, String> {
        let types = tic.types();
        let Some(format) = texture_format(tic.components(), types[0], tic.srgb()) else {
            return Err(format!(
                "a texture of format {:#x} with data type {}",
                tic.components(),
                types[0]
            ));
        };
        if !vk.format_supports(format.vk, vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            return Err(format!(
                "the device cannot sample {:?} (texture format {:#x})",
                format.vk,
                tic.components()
            ));
        }
        let Plan {
            width,
            height,
            depth,
            layers,
            levels,
            aspect,
            level_geometry,
            layer_stride,
            total,
            ..
        } = plan(&tic, shape, format)?;
        // A texture compared with memory in this frame already is not read again.
        let current = self.frame;
        let key = (tic, shape);
        if let Some(t) = self.map.get(&key) {
            if t.checked == current && t.generation == self.generation {
                return Ok(t.image.view);
            }
        }
        // The memory as it is now.
        let mut raw = vec![0u8; total as usize];
        let readable = read_partial(scene, tic.address(), &mut raw);
        if readable == 0 {
            return Err(format!(
                "a texture at GPU address {:#x} ({total} bytes), which is not mapped",
                tic.address()
            ));
        }
        let hash = hash_bytes(&raw);
        let known = self.map.get(&key).is_some_and(|t| t.hash == hash);
        if self.trace && !known {
            let filled = raw[..readable.min(raw.len())]
                .iter()
                .filter(|b| **b != 0)
                .count();
            eprintln!(
                "[vulkan] a texture at GPU address {:#x}: {:?} {width}x{height} (depth {depth}, {layers} layers, {levels} levels) as {:?}; {filled} of {} bytes are not zero ({readable} could be read)",
                tic.address(),
                format.vk,
                shape.kind,
                raw.len()
            );
        }
        if let Some(t) = self.map.get_mut(&key) {
            if (t.checked == current && t.generation == self.generation) || t.hash == hash {
                t.checked = current;
                t.generation = self.generation;
                return Ok(t.image.view);
            }
            // The contents changed: the old image goes once the submission is over.
            if let Some(old) = self.map.remove(&key) {
                frame.garbage.push(Garbage::Image(old.image));
            }
        }
        let components = tic.component_mapping();
        let image = vk.create_image(ImageInfo {
            format: format.vk,
            width,
            height,
            depth,
            layers,
            levels,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            view_type: match shape.kind {
                Kind::D2 => vk::ImageViewType::TYPE_2D,
                Kind::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
                Kind::Cube => vk::ImageViewType::CUBE,
                Kind::D3 => vk::ImageViewType::TYPE_3D,
            },
            components,
        })?;
        // Linear texels of every level and layer in upload order.
        let mut packed: Vec<u8> = Vec::new();
        let mut regions = Vec::new();
        for layer in 0..layers {
            let mut at = u64::from(layer) * layer_stride;
            for (level, g) in level_geometry.iter().enumerate() {
                let linear_size = g.linear_size() as usize;
                let mut linear = vec![0u8; linear_size];
                if tic.is_block_linear() {
                    let end = (at + g.tiled_size()).min(raw.len() as u64) as usize;
                    deswizzle(g, &raw[(at as usize).min(end)..end], &mut linear);
                    at += g.tiled_size();
                } else {
                    // Rows are `pitch` apart in memory and packed in the upload.
                    let row = (g.width * g.bytes_per_element) as usize;
                    for y in 0..g.height as usize {
                        let from = at as usize + y * tic.pitch() as usize;
                        if from + row <= raw.len() {
                            linear[y * row..(y + 1) * row].copy_from_slice(&raw[from..from + row]);
                        }
                    }
                }
                if format.astc {
                    linear = crate::astc::decode_image(
                        &linear,
                        (width >> level).max(1),
                        (height >> level).max(1),
                        format.block.0,
                        format.block.1,
                        tic.srgb(),
                    );
                }
                let offset = (packed.len() as u64).next_multiple_of(16);
                packed.resize(offset as usize, 0);
                packed.extend_from_slice(&linear);
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: aspect,
                            mip_level: level as u32,
                            base_array_layer: layer,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width: (width >> level).max(1),
                            height: (height >> level).max(1),
                            depth: (depth >> level).max(1),
                        }),
                );
            }
        }
        let (buffer, base_offset) = frame.upload(vk, &packed, 16)?;
        for r in &mut regions {
            r.buffer_offset += base_offset;
        }
        let cmd = frame.upload_cmd(vk)?;
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: levels,
            base_array_layer: 0,
            layer_count: layers,
        };
        let to_dst = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .subresource_range(range)];
        let to_read = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .subresource_range(range)];
        // SAFETY: the upload command buffer is recording; the handles are valid.
        unsafe {
            vk.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_dst,
            );
            vk.device.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            vk.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_read,
            );
        }
        let view = image.view;
        self.map.insert(
            key,
            Texture {
                image,
                checked: current,
                hash,
                bytes: total,
                generation: self.generation,
            },
        );
        Ok(view)
    }

    pub fn destroy(&mut self, vk: &Vk) {
        for (_, t) in self.map.drain() {
            vk.destroy_image(t.image);
        }
        for (_, s) in self.samplers.drain() {
            // SAFETY: nothing uses the sampler any more.
            unsafe { vk.device.destroy_sampler(s, None) };
        }
    }

    /// How many bytes of texture memory the cache holds.
    pub fn bytes(&self) -> u64 {
        self.map.values().map(|t| t.bytes).sum()
    }
}

/// Read as much of `out.len()` bytes at GPU address `va` as is mapped, in order; the number of
/// bytes read (the rest of `out` is left as it was).
fn read_partial(scene: &Scene<'_>, va: u64, out: &mut [u8]) -> usize {
    let mut done = 0;
    while done < out.len() {
        let Some((cpu, avail)) = scene.host_address(va + done as u64) else {
            break;
        };
        let n = (out.len() - done).min(avail as usize);
        if !scene.mem.read(cpu, &mut out[done..done + n]) {
            break;
        }
        done += n;
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tic_with(set: &[(usize, usize, u32)]) -> Tic {
        let mut w = [0u32; 8];
        for &(lo, len, v) in set {
            for i in 0..len {
                if v >> i & 1 != 0 {
                    let bit = lo + i;
                    w[bit / 32] |= 1 << (bit % 32);
                }
            }
        }
        Tic(w)
    }

    #[test]
    fn a_block_linear_header_reads_back_its_fields() {
        let t = tic_with(&[
            (0, 7, 0x08),
            (7, 3, 2),
            (10, 3, 2),
            (13, 3, 2),
            (16, 3, 2),
            (19, 3, 2),
            (22, 3, 3),
            (25, 3, 4),
            (28, 3, 5),
            (41, 23, 0x40_0000),
            (64, 16, 0x12),
            (85, 3, 3),
            (99, 3, 4),
            (124, 4, 5),
            (128, 16, 1023),
            (151, 4, 1),
            (150, 1, 1),
            (160, 16, 511),
        ]);
        assert_eq!(t.components(), 8);
        assert_eq!(t.types(), [2, 2, 2, 2]);
        assert_eq!(t.sources(), [2, 3, 4, 5]);
        assert!(t.is_block_linear() && !t.is_pitch());
        assert_eq!(t.address(), 0x12_8000_0000);
        assert_eq!(t.block_log2(), [0, 4, 0]);
        assert_eq!((t.width(), t.height()), (1024, 512));
        assert_eq!((t.texture_type(), t.srgb(), t.levels()), (1, true, 6));
    }

    #[test]
    fn a_pitch_header_has_its_own_address_and_pitch() {
        let t = tic_with(&[
            (85, 3, 2),
            (37, 27, 0x10_0000),
            (64, 16, 1),
            (96, 16, 0x40),
            (128, 16, 99),
            (160, 16, 49),
            (151, 4, 1),
        ]);
        assert!(t.is_pitch());
        assert_eq!(t.address(), 0x1_0000_0000 + (0x10_0000u64 << 5));
        assert_eq!(t.pitch(), 0x40 << 5);
        assert_eq!((t.width(), t.height()), (100, 50));
    }

    #[test]
    fn bits_may_cross_a_word_boundary() {
        let t = tic_with(&[(28, 8, 0xA5)]);
        assert_eq!(t.bits(28, 8), 0xA5);
        assert_eq!(t.bits(24, 4), 0);
    }

    #[test]
    fn formats_follow_the_component_code_and_data_type() {
        let f = texture_format(0x08, 2, false).unwrap();
        assert_eq!(
            (f.vk, f.bytes, f.block),
            (vk::Format::R8G8B8A8_UNORM, 4, (1, 1))
        );
        assert_eq!(
            texture_format(0x08, 2, true).unwrap().vk,
            vk::Format::R8G8B8A8_SRGB
        );
        assert_eq!(
            texture_format(0x24, 2, false).unwrap().vk,
            vk::Format::BC1_RGBA_UNORM_BLOCK
        );
        assert_eq!(
            texture_format(0x24, 2, true).unwrap().vk,
            vk::Format::BC1_RGBA_SRGB_BLOCK
        );
        assert_eq!(texture_format(0x24, 2, false).unwrap().bytes, 8);
        assert_eq!(
            texture_format(0x1D, 2, false).unwrap().vk,
            vk::Format::R8_UNORM
        );
        assert_eq!(
            texture_format(0x03, 7, false).unwrap().vk,
            vk::Format::R16G16B16A16_SFLOAT
        );
        let astc = texture_format(0x40, 2, false).unwrap();
        assert_eq!(
            (astc.vk, astc.block, astc.bytes, astc.astc),
            (vk::Format::R8G8B8A8_UNORM, (4, 4), 16, true),
            "ASTC is decoded to plain texels"
        );
        let astc = texture_format(0x46, 2, true).unwrap();
        assert_eq!((astc.vk, astc.block), (vk::Format::R8G8B8A8_SRGB, (12, 12)));
        assert!(
            texture_format(0x08, 7, false).is_none(),
            "8-bit floats do not exist"
        );
    }

    #[test]
    fn smaller_mip_levels_use_smaller_blocks() {
        let base = Geometry {
            width: 256,
            height: 256,
            depth: 1,
            bytes_per_element: 4,
            block_height_log2: 4,
            block_depth_log2: 0,
        };
        let size = (256, 256);
        let one = (1, 1);
        assert_eq!(mip_geometry(&base, 0, size, one), base);
        let l1 = mip_geometry(&base, 1, size, one);
        assert_eq!((l1.width, l1.height, l1.block_height_log2), (128, 128, 4));
        let l4 = mip_geometry(&base, 4, size, one);
        assert_eq!((l4.width, l4.height), (16, 16));
        assert_eq!(l4.block_height_log2, 1, "16 rows need two GOBs");
        let l8 = mip_geometry(&base, 8, size, one);
        assert_eq!((l8.width, l8.height, l8.block_height_log2), (1, 1, 0));
    }

    #[test]
    fn the_levels_of_a_compressed_texture_count_blocks_of_the_smaller_texture() {
        // 100 texels are 25 blocks of 4, but half of 100 is 50 texels, which is 13 blocks, not 12.
        assert_eq!(level_elements(100, 4, 0), 25);
        assert_eq!(level_elements(100, 4, 1), 13);
        assert_eq!(level_elements(100, 4, 2), 7);
        assert_eq!(level_elements(100, 4, 6), 1);
        assert_eq!(level_elements(100, 4, 9), 1, "never fewer than one block");
        assert_eq!(level_elements(100, 1, 3), 12);
        let base = Geometry {
            width: 25,
            height: 25,
            depth: 1,
            bytes_per_element: 16,
            block_height_log2: 0,
            block_depth_log2: 0,
        };
        let l1 = mip_geometry(&base, 1, (100, 100), (4, 4));
        assert_eq!((l1.width, l1.height), (13, 13));
    }

    #[test]
    fn a_sampler_header_gives_filters_and_modes() {
        let mut w = [0u32; 8];
        // Wrap in u, mirror in v, clamp to edge in w; linear everywhere; linear mips.
        w[0] = 1 << 3 | 2 << 6;
        w[1] = 2 | 2 << 4 | 3 << 6;
        w[2] = 0xFFF << 12;
        let vk = crate::Vk::new();
        let Ok(vk) = vk else { return };
        let (info, _) = Tsc(w).sampler_info(&vk);
        assert_eq!(info.address_mode_u, vk::SamplerAddressMode::REPEAT);
        assert_eq!(info.address_mode_v, vk::SamplerAddressMode::MIRRORED_REPEAT);
        assert_eq!(info.address_mode_w, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(info.mag_filter, vk::Filter::LINEAR);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert!(info.max_lod > 15.0);
    }
}
