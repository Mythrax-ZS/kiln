//! The 2D engine (class 0x902D): surface to surface copies, with scaling and a change of
//! format, as a title makes them to copy a texture or to resolve a target into one. The engine
//! keeps a destination and a source surface and the rectangle to copy; a write to
//! `PIXELS_FROM_MEMORY_SRC_Y0_INT` starts the copy.

use crate::draw_state::Layout;
use crate::state::Regs;

/// Registers by byte offset.
pub mod reg {
    pub const SET_DST_FORMAT: u32 = 0x200;
    pub const SET_DST_MEMORY_LAYOUT: u32 = 0x204;
    pub const SET_DST_BLOCK_SIZE: u32 = 0x208;
    pub const SET_DST_DEPTH: u32 = 0x20c;
    pub const SET_DST_LAYER: u32 = 0x210;
    pub const SET_DST_PITCH: u32 = 0x214;
    pub const SET_DST_WIDTH: u32 = 0x218;
    pub const SET_DST_HEIGHT: u32 = 0x21c;
    pub const SET_DST_OFFSET_UPPER: u32 = 0x220;
    pub const SET_SRC_FORMAT: u32 = 0x230;
    pub const SET_SRC_MEMORY_LAYOUT: u32 = 0x234;
    pub const SET_SRC_BLOCK_SIZE: u32 = 0x238;
    pub const SET_SRC_DEPTH: u32 = 0x23c;
    pub const SET_SRC_PITCH: u32 = 0x244;
    pub const SET_SRC_WIDTH: u32 = 0x248;
    pub const SET_SRC_HEIGHT: u32 = 0x24c;
    pub const SET_SRC_OFFSET_UPPER: u32 = 0x250;
    pub const SET_CLIP_X0: u32 = 0x280;
    pub const SET_CLIP_ENABLE: u32 = 0x290;
    pub const SET_COLOR_KEY_ENABLE: u32 = 0x29c;
    pub const SET_OPERATION: u32 = 0x2ac;
    pub const SET_PIXELS_FROM_MEMORY_SAMPLE_MODE: u32 = 0x88c;
    pub const SET_PIXELS_FROM_MEMORY_DST_X0: u32 = 0x8b0;
    pub const SET_PIXELS_FROM_MEMORY_DU_DX_FRAC: u32 = 0x8c0;
    pub const SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC: u32 = 0x8d0;
    /// The write that starts the copy.
    pub const PIXELS_FROM_MEMORY_SRC_Y0_INT: u32 = 0x8dc;
}

/// One of the two surfaces of a copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Surface {
    /// The GPU address of the surface's first byte (of the layer `layer` for a layered one).
    pub address: u64,
    /// The format code (the same numbers as a color target's).
    pub format: u32,
    pub layout: Layout,
    /// Bytes from one row to the next of a pitch-linear surface.
    pub pitch: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layer: u32,
}

/// A copy the engine was asked to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blit {
    pub dst: Surface,
    pub src: Surface,
    /// `SET_OPERATION`: 0 source copy with a mask, 3 source copy, 5 premultiplied, 6 blend...
    pub operation: u32,
    /// The rectangle of the destination written: its corner and size in pixels.
    pub dst_x: i32,
    pub dst_y: i32,
    pub dst_width: u32,
    pub dst_height: u32,
    /// Where the corner of that rectangle is in the source and how far a step of one destination
    /// pixel goes: all in 32.32 fixed point, in source pixels.
    pub src_x: i64,
    pub src_y: i64,
    pub du_dx: i64,
    pub dv_dy: i64,
    /// Linear filtering (else the nearest pixel).
    pub bilinear: bool,
    /// The origin of source coordinates is the corner of a pixel (else its center).
    pub corner_origin: bool,
    /// The clip rectangle `(x, y, width, height)` when it is enabled.
    pub clip: Option<[u32; 4]>,
}

impl Regs<'_> {
    fn twod_surface(&self, dst: bool) -> Surface {
        let (format, layout, block, depth, pitch, width, height, upper) = if dst {
            (
                reg::SET_DST_FORMAT,
                reg::SET_DST_MEMORY_LAYOUT,
                reg::SET_DST_BLOCK_SIZE,
                reg::SET_DST_DEPTH,
                reg::SET_DST_PITCH,
                reg::SET_DST_WIDTH,
                reg::SET_DST_HEIGHT,
                reg::SET_DST_OFFSET_UPPER,
            )
        } else {
            (
                reg::SET_SRC_FORMAT,
                reg::SET_SRC_MEMORY_LAYOUT,
                reg::SET_SRC_BLOCK_SIZE,
                reg::SET_SRC_DEPTH,
                reg::SET_SRC_PITCH,
                reg::SET_SRC_WIDTH,
                reg::SET_SRC_HEIGHT,
                reg::SET_SRC_OFFSET_UPPER,
            )
        };
        let block_word = self.word(block);
        Surface {
            address: u64::from(self.word(upper)) << 32 | u64::from(self.word(upper + 4)),
            format: self.word(format),
            // 0 is block linear, 1 pitch linear.
            layout: if self.word(layout) & 1 != 0 {
                Layout::Pitch
            } else {
                Layout::BlockLinear {
                    block: [
                        block_word & 0xF,
                        block_word >> 4 & 0xF,
                        block_word >> 8 & 0xF,
                    ],
                }
            },
            pitch: self.word(pitch),
            width: self.word(width),
            height: self.word(height),
            depth: self.word(depth),
            layer: if dst {
                self.word(reg::SET_DST_LAYER)
            } else {
                0
            },
        }
    }

    /// The copy the registers of the 2D engine describe.
    pub fn blit(&self) -> Blit {
        let fixed = |int: u32, frac: u32| {
            i64::from(self.word(int) as i32) << 32 | i64::from(self.word(frac))
        };
        let sample = self.word(reg::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE);
        let x0 = reg::SET_PIXELS_FROM_MEMORY_DST_X0;
        let du = reg::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC;
        let src = reg::SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC;
        Blit {
            dst: self.twod_surface(true),
            src: self.twod_surface(false),
            operation: self.word(reg::SET_OPERATION),
            dst_x: self.word(x0) as i32,
            dst_y: self.word(x0 + 4) as i32,
            dst_width: self.word(x0 + 8),
            dst_height: self.word(x0 + 12),
            // The fraction registers come first, the integer part after: du_dx (0x8c0, 0x8c4),
            // dv_dy (0x8c8, 0x8cc), src_x0 (0x8d0, 0x8d4), src_y0 (0x8d8, 0x8dc).
            du_dx: fixed(du + 4, du),
            dv_dy: fixed(du + 12, du + 8),
            src_x: fixed(src + 4, src),
            src_y: fixed(src + 12, src + 8),
            // Bit 0 of the sample mode: the origin (0 center, 1 corner); bits 4..5: the filter
            // (0 point, 1 bilinear).
            bilinear: sample >> 4 & 3 == 1,
            corner_origin: sample & 1 != 0,
            clip: (self.word(reg::SET_CLIP_ENABLE) & 1 != 0).then(|| {
                let c = reg::SET_CLIP_X0;
                [
                    self.word(c),
                    self.word(c + 4),
                    self.word(c + 8),
                    self.word(c + 12),
                ]
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(f: &mut [u32], offset: u32, value: u32) {
        f[offset as usize / 4] = value;
    }

    #[test]
    fn the_registers_describe_a_copy() {
        let mut f = vec![0u32; 0x1000];
        // The destination: block linear with blocks of 4 GOBs, 256 x 128 pixels of format 0xCA.
        set(&mut f, reg::SET_DST_FORMAT, 0xCA);
        set(&mut f, reg::SET_DST_MEMORY_LAYOUT, 0);
        set(&mut f, reg::SET_DST_BLOCK_SIZE, 2 << 4);
        set(&mut f, reg::SET_DST_DEPTH, 1);
        set(&mut f, reg::SET_DST_WIDTH, 256);
        set(&mut f, reg::SET_DST_HEIGHT, 128);
        set(&mut f, reg::SET_DST_OFFSET_UPPER, 0x6);
        set(&mut f, reg::SET_DST_OFFSET_UPPER + 4, 0x1234_5000);
        // The source: pitch linear, 512 x 256 pixels of format 0xD5 in rows of 2048 bytes.
        set(&mut f, reg::SET_SRC_FORMAT, 0xD5);
        set(&mut f, reg::SET_SRC_MEMORY_LAYOUT, 1);
        set(&mut f, reg::SET_SRC_PITCH, 2048);
        set(&mut f, reg::SET_SRC_WIDTH, 512);
        set(&mut f, reg::SET_SRC_HEIGHT, 256);
        set(&mut f, reg::SET_SRC_DEPTH, 1);
        set(&mut f, reg::SET_SRC_OFFSET_UPPER, 0x5);
        set(&mut f, reg::SET_SRC_OFFSET_UPPER + 4, 0x40_0000);
        // Half the size, a step of two source pixels, from the center of the pixel at (4, 8).
        set(&mut f, reg::SET_OPERATION, 3);
        set(&mut f, reg::SET_PIXELS_FROM_MEMORY_SAMPLE_MODE, 1 | 1 << 4);
        let x0 = reg::SET_PIXELS_FROM_MEMORY_DST_X0;
        set(&mut f, x0, 8);
        set(&mut f, x0 + 4, 16);
        set(&mut f, x0 + 8, 128);
        set(&mut f, x0 + 12, 64);
        let du = reg::SET_PIXELS_FROM_MEMORY_DU_DX_FRAC;
        set(&mut f, du, 0);
        set(&mut f, du + 4, 2);
        set(&mut f, du + 8, 0);
        set(&mut f, du + 12, 2);
        let src = reg::SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC;
        set(&mut f, src, 0x8000_0000);
        set(&mut f, src + 4, 4);
        set(&mut f, src + 8, 0x8000_0000);
        set(&mut f, src + 12, 8);
        let b = Regs(&f).blit();
        assert_eq!(b.dst.address, 0x6_1234_5000);
        assert_eq!(
            (b.dst.format, b.dst.width, b.dst.height, b.dst.depth),
            (0xCA, 256, 128, 1)
        );
        assert_eq!(b.dst.layout, Layout::BlockLinear { block: [0, 2, 0] });
        assert_eq!(b.src.address, 0x5_0040_0000);
        assert_eq!(
            (b.src.format, b.src.pitch, b.src.width, b.src.height),
            (0xD5, 2048, 512, 256)
        );
        assert_eq!(b.src.layout, Layout::Pitch);
        assert_eq!(b.operation, 3);
        assert_eq!(
            (b.dst_x, b.dst_y, b.dst_width, b.dst_height),
            (8, 16, 128, 64)
        );
        assert_eq!((b.du_dx, b.dv_dy), (2 << 32, 2 << 32));
        assert_eq!(
            (b.src_x, b.src_y),
            (4 << 32 | 0x8000_0000, 8 << 32 | 0x8000_0000)
        );
        assert!(b.bilinear && b.corner_origin);
        assert_eq!(b.clip, None);
        // A negative integer part is a negative coordinate.
        set(&mut f, src + 4, (-3i32) as u32);
        assert_eq!(Regs(&f).blit().src_x, -3i64 << 32 | 0x8000_0000);
        // A clip rectangle only when it is enabled.
        set(&mut f, reg::SET_CLIP_X0, 1);
        set(&mut f, reg::SET_CLIP_X0 + 4, 2);
        set(&mut f, reg::SET_CLIP_X0 + 8, 30);
        set(&mut f, reg::SET_CLIP_X0 + 12, 40);
        assert_eq!(Regs(&f).blit().clip, None);
        set(&mut f, reg::SET_CLIP_ENABLE, 1);
        assert_eq!(Regs(&f).blit().clip, Some([1, 2, 30, 40]));
    }
}
