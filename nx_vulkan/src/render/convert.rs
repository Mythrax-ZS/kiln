//! The values of the console GPU's state registers in Vulkan's vocabulary.
//!
//! Several registers accept two encodings of the same thing, one in OpenGL's numbering (the
//! one the console's graphics library uses) and one in Direct3D's; both are understood.

use ash::vk;
use nx_shader::translate::Class;

pub fn compare_op(v: u32) -> vk::CompareOp {
    match v {
        0x200 | 1 => vk::CompareOp::NEVER,
        0x201 | 2 => vk::CompareOp::LESS,
        0x202 | 3 => vk::CompareOp::EQUAL,
        0x203 | 4 => vk::CompareOp::LESS_OR_EQUAL,
        0x204 | 5 => vk::CompareOp::GREATER,
        0x205 | 6 => vk::CompareOp::NOT_EQUAL,
        0x206 | 7 => vk::CompareOp::GREATER_OR_EQUAL,
        _ => vk::CompareOp::ALWAYS,
    }
}

pub fn stencil_op(v: u32) -> vk::StencilOp {
    match v {
        0x1E00 | 1 => vk::StencilOp::KEEP,
        0 | 2 => vk::StencilOp::ZERO,
        0x1E01 | 3 => vk::StencilOp::REPLACE,
        0x1E02 | 4 => vk::StencilOp::INCREMENT_AND_CLAMP,
        0x1E03 | 5 => vk::StencilOp::DECREMENT_AND_CLAMP,
        0x150A | 6 => vk::StencilOp::INVERT,
        0x8507 | 7 => vk::StencilOp::INCREMENT_AND_WRAP,
        0x8508 | 8 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}

pub fn blend_op(v: u32) -> vk::BlendOp {
    match v {
        0x800A | 2 => vk::BlendOp::SUBTRACT,
        0x800B | 3 => vk::BlendOp::REVERSE_SUBTRACT,
        0x8007 | 4 => vk::BlendOp::MIN,
        0x8008 | 5 => vk::BlendOp::MAX,
        _ => vk::BlendOp::ADD,
    }
}

pub fn blend_factor(v: u32) -> vk::BlendFactor {
    match v {
        0x4000 | 1 => vk::BlendFactor::ZERO,
        0x4001 | 2 => vk::BlendFactor::ONE,
        0x4300 | 3 => vk::BlendFactor::SRC_COLOR,
        0x4301 | 4 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        0x4302 | 5 => vk::BlendFactor::SRC_ALPHA,
        0x4303 | 6 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        0x4304 | 7 => vk::BlendFactor::DST_ALPHA,
        0x4305 | 8 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        0x4306 | 9 => vk::BlendFactor::DST_COLOR,
        0x4307 | 10 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        0x4308 | 11 => vk::BlendFactor::SRC_ALPHA_SATURATE,
        0xC001 | 13 => vk::BlendFactor::CONSTANT_COLOR,
        0xC002 | 14 => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        0xC003 => vk::BlendFactor::CONSTANT_ALPHA,
        0xC004 => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
        0xC900 | 16 => vk::BlendFactor::SRC1_COLOR,
        0xC901 | 17 => vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        0xC902 | 18 => vk::BlendFactor::SRC1_ALPHA,
        0xC903 | 19 => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
        _ => vk::BlendFactor::ONE,
    }
}

pub fn polygon_mode(v: u32) -> vk::PolygonMode {
    match v {
        0x1B00 => vk::PolygonMode::POINT,
        0x1B01 => vk::PolygonMode::LINE,
        _ => vk::PolygonMode::FILL,
    }
}

/// A primitive type of `BEGIN` as Vulkan's topology, with the way the vertices must be rewritten
/// when the two differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    /// Vulkan draws it as is.
    Native(vk::PrimitiveTopology),
    /// Each group of four vertices is a quad: two triangles.
    Quads,
    /// A strip of quads: each pair of new vertices adds one.
    QuadStrip,
    /// The last vertex joins back to the first.
    LineLoop,
    /// A fan of triangles in a polygon.
    Polygon,
}

pub fn topology(op: u32) -> Result<Topology, String> {
    use vk::PrimitiveTopology as T;
    Ok(match op {
        0 => Topology::Native(T::POINT_LIST),
        1 => Topology::Native(T::LINE_LIST),
        2 => Topology::LineLoop,
        3 => Topology::Native(T::LINE_STRIP),
        4 => Topology::Native(T::TRIANGLE_LIST),
        5 => Topology::Native(T::TRIANGLE_STRIP),
        6 => Topology::Native(T::TRIANGLE_FAN),
        7 => Topology::Quads,
        8 => Topology::QuadStrip,
        9 => Topology::Polygon,
        0xA => Topology::Native(T::LINE_LIST_WITH_ADJACENCY),
        0xB => Topology::Native(T::LINE_STRIP_WITH_ADJACENCY),
        0xC => Topology::Native(T::TRIANGLE_LIST_WITH_ADJACENCY),
        0xD => Topology::Native(T::TRIANGLE_STRIP_WITH_ADJACENCY),
        _ => return Err(format!("primitive type {op}")),
    })
}

/// The vertex format of an attribute: `widths` is `COMPONENT_BIT_WIDTHS`, `numeric` the numerical
/// type, `bgra` the red and blue swap. Also what class of value the shader gets.
pub fn vertex_format(widths: u32, numeric: u32, bgra: bool) -> Option<(vk::Format, Class)> {
    use vk::Format as F;
    // unorm, snorm, uint, sint, uscaled, sscaled, float for each layout (None: not available).
    type Row = [Option<F>; 7];
    let row = |r: Row| -> Option<F> {
        match numeric {
            2 => r[0],
            1 => r[1],
            4 => r[2],
            3 => r[3],
            5 => r[4],
            6 => r[5],
            7 => r[6],
            _ => None,
        }
    };
    let n = None;
    let format = match widths {
        0x01 => row([
            n,
            n,
            Some(F::R32G32B32A32_UINT),
            Some(F::R32G32B32A32_SINT),
            n,
            n,
            Some(F::R32G32B32A32_SFLOAT),
        ]),
        0x02 => row([
            n,
            n,
            Some(F::R32G32B32_UINT),
            Some(F::R32G32B32_SINT),
            n,
            n,
            Some(F::R32G32B32_SFLOAT),
        ]),
        0x03 => row([
            Some(F::R16G16B16A16_UNORM),
            Some(F::R16G16B16A16_SNORM),
            Some(F::R16G16B16A16_UINT),
            Some(F::R16G16B16A16_SINT),
            Some(F::R16G16B16A16_USCALED),
            Some(F::R16G16B16A16_SSCALED),
            Some(F::R16G16B16A16_SFLOAT),
        ]),
        0x04 => row([
            n,
            n,
            Some(F::R32G32_UINT),
            Some(F::R32G32_SINT),
            n,
            n,
            Some(F::R32G32_SFLOAT),
        ]),
        0x05 => row([
            Some(F::R16G16B16_UNORM),
            Some(F::R16G16B16_SNORM),
            Some(F::R16G16B16_UINT),
            Some(F::R16G16B16_SINT),
            Some(F::R16G16B16_USCALED),
            Some(F::R16G16B16_SSCALED),
            Some(F::R16G16B16_SFLOAT),
        ]),
        0x0A | 0x2F | 0x33 => {
            if bgra {
                row([
                    Some(F::B8G8R8A8_UNORM),
                    Some(F::B8G8R8A8_SNORM),
                    Some(F::B8G8R8A8_UINT),
                    Some(F::B8G8R8A8_SINT),
                    Some(F::B8G8R8A8_USCALED),
                    Some(F::B8G8R8A8_SSCALED),
                    n,
                ])
            } else {
                row([
                    Some(F::R8G8B8A8_UNORM),
                    Some(F::R8G8B8A8_SNORM),
                    Some(F::R8G8B8A8_UINT),
                    Some(F::R8G8B8A8_SINT),
                    Some(F::R8G8B8A8_USCALED),
                    Some(F::R8G8B8A8_SSCALED),
                    n,
                ])
            }
        }
        0x0F => row([
            Some(F::R16G16_UNORM),
            Some(F::R16G16_SNORM),
            Some(F::R16G16_UINT),
            Some(F::R16G16_SINT),
            Some(F::R16G16_USCALED),
            Some(F::R16G16_SSCALED),
            Some(F::R16G16_SFLOAT),
        ]),
        0x12 => row([
            n,
            n,
            Some(F::R32_UINT),
            Some(F::R32_SINT),
            n,
            n,
            Some(F::R32_SFLOAT),
        ]),
        0x13 => row([
            Some(F::R8G8B8_UNORM),
            Some(F::R8G8B8_SNORM),
            Some(F::R8G8B8_UINT),
            Some(F::R8G8B8_SINT),
            Some(F::R8G8B8_USCALED),
            Some(F::R8G8B8_SSCALED),
            n,
        ]),
        0x18 | 0x32 => row([
            Some(F::R8G8_UNORM),
            Some(F::R8G8_SNORM),
            Some(F::R8G8_UINT),
            Some(F::R8G8_SINT),
            Some(F::R8G8_USCALED),
            Some(F::R8G8_SSCALED),
            n,
        ]),
        0x1B => row([
            Some(F::R16_UNORM),
            Some(F::R16_SNORM),
            Some(F::R16_UINT),
            Some(F::R16_SINT),
            Some(F::R16_USCALED),
            Some(F::R16_SSCALED),
            Some(F::R16_SFLOAT),
        ]),
        0x1D | 0x34 => row([
            Some(F::R8_UNORM),
            Some(F::R8_SNORM),
            Some(F::R8_UINT),
            Some(F::R8_SINT),
            Some(F::R8_USCALED),
            Some(F::R8_SSCALED),
            n,
        ]),
        0x30 => row([
            Some(F::A2B10G10R10_UNORM_PACK32),
            Some(F::A2B10G10R10_SNORM_PACK32),
            Some(F::A2B10G10R10_UINT_PACK32),
            Some(F::A2B10G10R10_SINT_PACK32),
            Some(F::A2B10G10R10_USCALED_PACK32),
            Some(F::A2B10G10R10_SSCALED_PACK32),
            n,
        ]),
        0x31 => row([n, n, n, n, n, n, Some(F::B10G11R11_UFLOAT_PACK32)]),
        _ => None,
    }?;
    let class = match numeric {
        3 => Class::Sint,
        4 => Class::Uint,
        _ => Class::Float,
    };
    Some((format, class))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_numberings_of_a_comparison_agree() {
        for (ogl, d3d) in [(0x200, 1), (0x201, 2), (0x203, 4), (0x207, 8)] {
            assert_eq!(compare_op(ogl), compare_op(d3d));
        }
        assert_eq!(compare_op(0x203), vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(compare_op(0x204), vk::CompareOp::GREATER);
    }

    #[test]
    fn blending_is_translated_in_both_numberings() {
        assert_eq!(blend_op(0x8006), vk::BlendOp::ADD);
        assert_eq!(blend_op(0x800A), vk::BlendOp::SUBTRACT);
        assert_eq!(blend_op(3), vk::BlendOp::REVERSE_SUBTRACT);
        assert_eq!(blend_factor(0x4302), vk::BlendFactor::SRC_ALPHA);
        assert_eq!(blend_factor(5), vk::BlendFactor::SRC_ALPHA);
        assert_eq!(blend_factor(0x4303), vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
        assert_eq!(blend_factor(0x4000), vk::BlendFactor::ZERO);
        assert_eq!(blend_factor(0xC001), vk::BlendFactor::CONSTANT_COLOR);
    }

    #[test]
    fn stencil_operations_and_polygon_modes() {
        assert_eq!(stencil_op(0x1E00), vk::StencilOp::KEEP);
        assert_eq!(stencil_op(0), vk::StencilOp::ZERO);
        assert_eq!(stencil_op(0x8507), vk::StencilOp::INCREMENT_AND_WRAP);
        assert_eq!(polygon_mode(0x1B01), vk::PolygonMode::LINE);
        assert_eq!(polygon_mode(0x1B02), vk::PolygonMode::FILL);
    }

    #[test]
    fn primitive_types() {
        assert_eq!(
            topology(4).unwrap(),
            Topology::Native(vk::PrimitiveTopology::TRIANGLE_LIST)
        );
        assert_eq!(topology(7).unwrap(), Topology::Quads);
        assert!(topology(14).is_err(), "patches need tessellation");
    }

    #[test]
    fn vertex_formats() {
        assert_eq!(
            vertex_format(1, 7, false),
            Some((vk::Format::R32G32B32A32_SFLOAT, Class::Float))
        );
        assert_eq!(
            vertex_format(0x0A, 2, false),
            Some((vk::Format::R8G8B8A8_UNORM, Class::Float))
        );
        assert_eq!(
            vertex_format(0x0A, 2, true),
            Some((vk::Format::B8G8R8A8_UNORM, Class::Float))
        );
        assert_eq!(
            vertex_format(0x04, 4, false),
            Some((vk::Format::R32G32_UINT, Class::Uint))
        );
        assert_eq!(
            vertex_format(0x12, 3, false),
            Some((vk::Format::R32_SINT, Class::Sint))
        );
        assert!(
            vertex_format(1, 2, false).is_none(),
            "32-bit unorm does not exist"
        );
        assert!(vertex_format(0x7F, 7, false).is_none());
    }
}
