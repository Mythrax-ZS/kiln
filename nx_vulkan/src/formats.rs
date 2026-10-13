//! Surface formats of the console GPU and their Vulkan counterparts.
//!
//! The names of the hardware formats list the components from the most significant end of the
//! word, and a suffix says what a component is: none for unsigned normalized, `N` signed
//! normalized, `S` signed integer, `U` unsigned integer, `F` float, `L` sRGB (`A8R8G8B8` is a
//! 32-bit word with alpha on top, so its bytes in memory are B, G, R, A).

use ash::vk;
use nx_shader::translate::Class;

/// A render target format: what Vulkan calls it, what a shader writing it handles, and the size
/// of a pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorFormat {
    pub vk: vk::Format,
    pub class: Class,
    pub bytes: u32,
}

const fn cf(vk: vk::Format, class: Class, bytes: u32) -> Option<ColorFormat> {
    Some(ColorFormat { vk, class, bytes })
}

/// `SET_COLOR_TARGET_FORMAT` to a Vulkan format. `None` for a code that is not a color format
/// (or not one a Vulkan device has: `X` and `Z`/`O` formats map to their alpha twins).
pub fn color_format(code: u32) -> Option<ColorFormat> {
    use Class::{Float as F, Sint as S, Uint as U};
    match code {
        0xC0 | 0xC3 => cf(vk::Format::R32G32B32A32_SFLOAT, F, 16),
        0xC1 | 0xC4 => cf(vk::Format::R32G32B32A32_SINT, S, 16),
        0xC2 | 0xC5 => cf(vk::Format::R32G32B32A32_UINT, U, 16),
        0xC6 => cf(vk::Format::R16G16B16A16_UNORM, F, 8),
        0xC7 => cf(vk::Format::R16G16B16A16_SNORM, F, 8),
        0xC8 => cf(vk::Format::R16G16B16A16_SINT, S, 8),
        0xC9 => cf(vk::Format::R16G16B16A16_UINT, U, 8),
        0xCA | 0xCE => cf(vk::Format::R16G16B16A16_SFLOAT, F, 8),
        0xCB => cf(vk::Format::R32G32_SFLOAT, F, 8),
        0xCC => cf(vk::Format::R32G32_SINT, S, 8),
        0xCD => cf(vk::Format::R32G32_UINT, U, 8),
        0xCF | 0xE6 | 0xFD | 0xFE => cf(vk::Format::B8G8R8A8_UNORM, F, 4),
        0xD0 | 0xE7 => cf(vk::Format::B8G8R8A8_SRGB, F, 4),
        0xD1 => cf(vk::Format::A2B10G10R10_UNORM_PACK32, F, 4),
        0xD2 => cf(vk::Format::A2B10G10R10_UINT_PACK32, U, 4),
        0xD5 | 0xF9 => cf(vk::Format::R8G8B8A8_UNORM, F, 4),
        0xD6 | 0xFA => cf(vk::Format::R8G8B8A8_SRGB, F, 4),
        0xD7 => cf(vk::Format::R8G8B8A8_SNORM, F, 4),
        0xD8 => cf(vk::Format::R8G8B8A8_SINT, S, 4),
        0xD9 => cf(vk::Format::R8G8B8A8_UINT, U, 4),
        0xDA => cf(vk::Format::R16G16_UNORM, F, 4),
        0xDB => cf(vk::Format::R16G16_SNORM, F, 4),
        0xDC => cf(vk::Format::R16G16_SINT, S, 4),
        0xDD => cf(vk::Format::R16G16_UINT, U, 4),
        0xDE => cf(vk::Format::R16G16_SFLOAT, F, 4),
        0xDF => cf(vk::Format::A2R10G10B10_UNORM_PACK32, F, 4),
        0xE0 => cf(vk::Format::B10G11R11_UFLOAT_PACK32, F, 4),
        0xE3 => cf(vk::Format::R32_SINT, S, 4),
        0xE4 => cf(vk::Format::R32_UINT, U, 4),
        0xE5 | 0xFF => cf(vk::Format::R32_SFLOAT, F, 4),
        0xE8 => cf(vk::Format::R5G6B5_UNORM_PACK16, F, 2),
        0xE9 | 0xF8 | 0xFB | 0xFC => cf(vk::Format::A1R5G5B5_UNORM_PACK16, F, 2),
        0xEA => cf(vk::Format::R8G8_UNORM, F, 2),
        0xEB => cf(vk::Format::R8G8_SNORM, F, 2),
        0xEC => cf(vk::Format::R8G8_SINT, S, 2),
        0xED => cf(vk::Format::R8G8_UINT, U, 2),
        0xEE => cf(vk::Format::R16_UNORM, F, 2),
        0xEF => cf(vk::Format::R16_SNORM, F, 2),
        0xF0 => cf(vk::Format::R16_SINT, S, 2),
        0xF1 => cf(vk::Format::R16_UINT, U, 2),
        0xF2 => cf(vk::Format::R16_SFLOAT, F, 2),
        0xF3 | 0xF7 => cf(vk::Format::R8_UNORM, F, 1),
        0xF4 => cf(vk::Format::R8_SNORM, F, 1),
        0xF5 => cf(vk::Format::R8_SINT, S, 1),
        0xF6 => cf(vk::Format::R8_UINT, U, 1),
        _ => None,
    }
}

/// A depth/stencil format: Vulkan's, and whether it has a stencil part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthFormat {
    pub vk: vk::Format,
    pub stencil: bool,
    pub bytes: u32,
}

/// `SET_ZT_FORMAT` to a Vulkan format.
pub fn depth_format(code: u32) -> Option<DepthFormat> {
    let (vk, stencil, bytes) = match code {
        0x13 => (vk::Format::D16_UNORM, false, 2),
        0x14 | 0x16 | 0x18 => (vk::Format::D24_UNORM_S8_UINT, true, 4),
        0x15 => (vk::Format::X8_D24_UNORM_PACK32, false, 4),
        0x0A => (vk::Format::D32_SFLOAT, false, 4),
        0x17 => (vk::Format::S8_UINT, true, 1),
        0x19 | 0x1E | 0x1F => (vk::Format::D32_SFLOAT_S8_UINT, true, 8),
        0x1D => (vk::Format::D24_UNORM_S8_UINT, true, 8),
        _ => return None,
    };
    Some(DepthFormat { vk, stencil, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_formats_a_title_renders_to_are_known() {
        let f = color_format(0xCF).unwrap();
        assert_eq!((f.vk, f.bytes), (vk::Format::B8G8R8A8_UNORM, 4));
        assert_eq!(color_format(0xD5).unwrap().vk, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(color_format(0xD6).unwrap().vk, vk::Format::R8G8B8A8_SRGB);
        assert_eq!(color_format(0xCA).unwrap().bytes, 8);
        assert_eq!(color_format(0xC1).unwrap().class, Class::Sint);
        assert_eq!(color_format(0xC2).unwrap().class, Class::Uint);
        assert!(color_format(0).is_none(), "disabled is not a format");
        assert_eq!(
            depth_format(0x14).unwrap().vk,
            vk::Format::D24_UNORM_S8_UINT
        );
        assert!(depth_format(0x0A).is_some_and(|d| !d.stencil));
    }

    #[test]
    fn every_listed_code_has_a_nonzero_pixel_size() {
        for code in 0..=0xFF {
            if let Some(f) = color_format(code) {
                assert!(f.bytes > 0, "{code:#x}");
            }
        }
    }
}
