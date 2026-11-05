//! The 2D engine (class 0x902D): surface to surface copies, with scaling and a change of
//! format, as a title makes them to copy a texture or to resolve a target into one. The engine
//! keeps a destination and a source surface and the rectangle to copy; a write to
//! `PIXELS_FROM_MEMORY_SRC_Y0_INT` starts the copy.

use crate::draw_state::Layout;
use crate::mem::GuestMemory;
use crate::state::Regs;
use crate::vm::AddressSpace;

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

/// The bytes of a pixel of a surface of format `code` (the numbers of a color target's), or
/// `None` for a code that is not a color format.
pub fn pixel_bytes(code: u32) -> Option<u32> {
    Some(match code {
        0xC0..=0xC5 => 16,
        0xC6..=0xCE => 8,
        0xCF..=0xD2 | 0xD5..=0xE0 | 0xE3..=0xE7 | 0xF9 | 0xFA | 0xFD..=0xFF => 4,
        0xE8..=0xF2 | 0xF8 | 0xFB | 0xFC => 2,
        0xF3..=0xF7 => 1,
        _ => return None,
    })
}

impl Surface {
    /// Bytes from one row of blocks (block linear) or of pixels (pitch linear) to the next, and
    /// how many rows of pixels such a row holds.
    fn band(&self, bytes: u32) -> (u64, u32) {
        match self.layout {
            Layout::Pitch => (u64::from(self.pitch), 1),
            Layout::BlockLinear { block } => {
                let gobs_across = (u64::from(self.width) * u64::from(bytes)).div_ceil(64);
                (gobs_across * (512 << block[1]), 8 << block[1])
            }
        }
    }

    /// Where byte `x` of row `y` is, from the surface's address, for pixels of `bytes` bytes.
    fn offset(&self, x: u64, y: u32, bytes: u32) -> u64 {
        let (band, rows) = self.band(bytes);
        match self.layout {
            Layout::Pitch => u64::from(y) * band + x,
            Layout::BlockLinear { .. } => {
                u64::from(y / rows) * band
                    + (x / 64) * 512 * u64::from(rows / 8)
                    + u64::from(y % rows / 8) * 512
                    + u64::from(crate::gob::offset((x % 64) as u32, y % 8))
            }
        }
    }

    /// The bytes of memory the surface takes, for pixels of `bytes` bytes (its last row of blocks
    /// whole).
    pub fn memory_bytes(&self, bytes: u32) -> u64 {
        let (band, rows) = self.band(bytes);
        band * u64::from(self.height.div_ceil(rows))
    }

    /// A layout the copy in memory can address: pitch, or blocks one GOB wide and deep.
    fn addressable(&self) -> bool {
        match self.layout {
            Layout::Pitch => self.pitch > 0,
            Layout::BlockLinear { block } => block[0] == 0 && block[2] == 0 && block[1] <= 5,
        }
    }
}

/// Make the copy in guest memory, for surfaces nothing on the device stands for: every pixel of
/// the destination's rectangle (inside the clip rectangle, when there is one) takes the source
/// pixel its center falls on, by the rule of A-165, the coordinates clamped to the source. Both
/// surfaces must have the same format, and a linear filter only without scaling (where it gives
/// the same pixels). Says which bytes of the destination's memory, as offsets from its address, it
/// may have written.
pub fn copy_in_memory(
    space: &AddressSpace,
    mem: &dyn GuestMemory,
    b: &Blit,
) -> Result<(u64, u64), String> {
    let Some(bytes) = pixel_bytes(b.src.format) else {
        return Err(format!(
            "a copy in memory from a surface of format {:#x}",
            b.src.format
        ));
    };
    if b.dst.format != b.src.format {
        return Err(format!(
            "a copy in memory from format {:#x} into format {:#x}",
            b.src.format, b.dst.format
        ));
    }
    if !matches!(b.operation, 0 | 3 | 4) {
        return Err(format!(
            "a copy in memory with the operation {}",
            b.operation
        ));
    }
    const ONE: i64 = 1 << 32;
    if b.bilinear && (b.du_dx != ONE || b.dv_dy != ONE) {
        return Err("a copy in memory with a linear filter and scaling".to_string());
    }
    if !b.src.addressable() || !b.dst.addressable() || b.src.depth > 1 || b.dst.depth > 1 {
        return Err(
            "a copy in memory of a layered surface or of blocks wider or deeper than a GOB"
                .to_string(),
        );
    }
    if b.dst.layer != 0 {
        return Err(format!("a copy in memory into layer {}", b.dst.layer));
    }
    if b.src.width == 0 || b.src.height == 0 || b.dst_width == 0 || b.dst_height == 0 {
        return Ok((0, 0));
    }
    // The destination's pixels written: the rectangle, inside the clip and the surface.
    let (mut x0, mut y0) = (i64::from(b.dst_x), i64::from(b.dst_y));
    let (mut x1, mut y1) = (x0 + i64::from(b.dst_width), y0 + i64::from(b.dst_height));
    if let Some([cx, cy, cw, ch]) = b.clip {
        x0 = x0.max(i64::from(cx));
        y0 = y0.max(i64::from(cy));
        x1 = x1.min(i64::from(cx) + i64::from(cw));
        y1 = y1.min(i64::from(cy) + i64::from(ch));
    }
    x0 = x0.max(0);
    y0 = y0.max(0);
    x1 = x1.min(i64::from(b.dst.width));
    y1 = y1.min(i64::from(b.dst.height));
    if x0 >= x1 || y0 >= y1 {
        return Ok((0, 0));
    }
    // Destination pixel `i` takes the source at `src + i * step` (A-165): from the corner of the
    // source's pixels, or from the center of its first, which is half a pixel further on.
    let shift = if b.corner_origin { 0 } else { ONE / 2 };
    let source = |start: i64, step: i64, i: i64, size: u32| -> u32 {
        let at = start + shift + step * i;
        (at >> 32).clamp(0, i64::from(size) - 1) as u32
    };
    let src_rows: Vec<u32> = (y0..y1)
        .map(|y| source(b.src_y, b.dv_dy, y - i64::from(b.dst_y), b.src.height))
        .collect();
    let src_columns: Vec<u32> = (x0..x1)
        .map(|x| source(b.src_x, b.du_dx, x - i64::from(b.dst_x), b.src.width))
        .collect();
    // The bands of rows each surface is read in, whole.
    let span = |s: &Surface, first: u32, last: u32| {
        let (band, rows) = s.band(bytes);
        (
            u64::from(first / rows) * band,
            u64::from(last / rows + 1) * band,
        )
    };
    let (src_lo, src_hi) = {
        let first = *src_rows.iter().min().expect("rows");
        let last = *src_rows.iter().max().expect("rows");
        span(&b.src, first, last)
    };
    let (dst_lo, dst_hi) = span(&b.dst, y0 as u32, (y1 - 1) as u32);
    let read = |address: u64, lo: u64, hi: u64| -> Result<(Vec<u8>, usize), String> {
        let mut v = vec![0u8; (hi - lo) as usize];
        let n = space.read_prefix(mem, address + lo, &mut v);
        if n == 0 {
            return Err(format!(
                "a copy in memory with GPU address {:#x}, which is not mapped",
                address + lo
            ));
        }
        Ok((v, n))
    };
    let (src, _) = read(b.src.address, src_lo, src_hi)?;
    let (mut dst, mapped) = read(b.dst.address, dst_lo, dst_hi)?;
    let n = bytes as usize;
    for (y, &sy) in (y0 as u32..y1 as u32).zip(&src_rows) {
        for (x, &sx) in (x0 as u32..x1 as u32).zip(&src_columns) {
            let from =
                (b.src.offset(u64::from(sx) * u64::from(bytes), sy, bytes) - src_lo) as usize;
            let to = (b.dst.offset(u64::from(x) * u64::from(bytes), y, bytes) - dst_lo) as usize;
            // A pixel is inside one 16-byte sector of its GOB (the sizes divide 16), so its bytes
            // are together; a pixel past what is mapped is left alone.
            if let (Some(p), Some(q)) = (src.get(from..from + n), dst.get_mut(to..to + n)) {
                q.copy_from_slice(p);
            }
        }
    }
    if !space.write(mem, b.dst.address + dst_lo, &dst[..mapped]) {
        return Err(format!(
            "a copy in memory to GPU address {:#x}, which is not mapped",
            b.dst.address + dst_lo
        ));
    }
    Ok((dst_lo, dst_lo + mapped as u64))
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

    fn surface(address: u64, layout: Layout, pitch: u32, width: u32, height: u32) -> Surface {
        Surface {
            address,
            format: 0xD5,
            layout,
            pitch,
            width,
            height,
            depth: 1,
            layer: 0,
        }
    }

    fn copy(dst: Surface, src: Surface) -> Blit {
        Blit {
            dst,
            src,
            operation: 3,
            dst_x: 0,
            dst_y: 0,
            dst_width: dst.width,
            dst_height: dst.height,
            src_x: 1 << 31,
            src_y: 1 << 31,
            du_dx: 1 << 32,
            dv_dy: 1 << 32,
            bilinear: false,
            corner_origin: true,
            clip: None,
        }
    }

    fn memory() -> (AddressSpace, crate::mem::testing::TestMemory) {
        let mut space = AddressSpace::new();
        space
            .initialize(0, 0, 0x1_0000, 0x100_0000, 0x80_0000)
            .unwrap();
        space
            .map_range(0x10_0000, 0x10_0000, 0x5000_0000, 0)
            .unwrap();
        let mem = crate::mem::testing::TestMemory::new();
        mem.map(0x5000_0000, 0x10_0000);
        (space, mem)
    }

    #[test]
    fn a_copy_in_memory_moves_pixels_between_layouts() {
        let (space, mem) = memory();
        // A pitch-linear source of 100 x 20 pixels whose pixel (x, y) holds x + 1000 y.
        let src = surface(0x10_0000, Layout::Pitch, 512, 100, 20);
        for y in 0..20u32 {
            for x in 0..100u32 {
                let v = x + 1000 * y;
                let at = src.address + u64::from(y * 512 + 4 * x);
                assert!(space.write(&mem, at, &v.to_le_bytes()));
            }
        }
        // Into a block-linear surface with blocks of 2 GOBs, and back out to another pitch one.
        let block = Layout::BlockLinear { block: [0, 1, 0] };
        let tiled = surface(0x14_0000, block, 0, 100, 20);
        let (lo, hi) = copy_in_memory(&space, &mem, &copy(tiled, src)).unwrap();
        assert_eq!((lo, hi), (0, tiled.memory_bytes(4)));
        let back = surface(0x18_0000, Layout::Pitch, 400, 100, 20);
        copy_in_memory(&space, &mem, &copy(back, tiled)).unwrap();
        let at = |s: &Surface, x: u32, y: u32| {
            let mut w = [0u8; 4];
            let offset = s.offset(u64::from(4 * x), y, 4);
            assert!(space.read(&mem, s.address + offset, &mut w));
            u32::from_le_bytes(w)
        };
        for y in 0..20 {
            for x in 0..100 {
                assert_eq!(at(&tiled, x, y), x + 1000 * y, "({x}, {y})");
                assert_eq!(at(&back, x, y), x + 1000 * y, "({x}, {y})");
            }
        }
        // Pixel (17, 9) of the block-linear surface: in the second block across (the first is
        // 2 GOBs of 512 bytes), the second GOB of it, row 1 of that.
        let expected = 1024 + 512 + u64::from(crate::gob::offset(4, 1));
        assert_eq!(tiled.offset(4 * 17, 9, 4), expected);
    }

    #[test]
    fn a_copy_in_memory_keeps_to_its_rectangle_and_clamps_the_source() {
        let (space, mem) = memory();
        let src = surface(0x10_0000, Layout::Pitch, 64, 4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v = 1 + x + 4 * y;
                let at = src.address + u64::from(y * 64 + 4 * x);
                assert!(space.write(&mem, at, &v.to_le_bytes()));
            }
        }
        let dst = surface(0x14_0000, Layout::Pitch, 64, 8, 8);
        assert!(space.write(&mem, dst.address, &[0xEE; 512]));
        // An 8 x 2 rectangle at (0, 3) from source column 2 on: the columns past the source's
        // right edge take its last one.
        let mut b = copy(dst, src);
        b.dst_y = 3;
        b.dst_height = 2;
        b.src_x = 2 << 32 | 1 << 31;
        copy_in_memory(&space, &mem, &b).unwrap();
        let word = |x: u32, y: u32| {
            let mut w = [0u8; 4];
            assert!(space.read(&mem, dst.address + u64::from(y * 64 + 4 * x), &mut w));
            u32::from_le_bytes(w)
        };
        assert_eq!(word(0, 2), 0xEEEE_EEEE);
        assert_eq!((word(0, 3), word(1, 3), word(5, 3)), (3, 4, 4));
        assert_eq!((word(0, 4), word(7, 4)), (7, 8));
        assert_eq!(word(0, 5), 0xEEEE_EEEE);
        // Formats that differ are not copied.
        b.dst.format = 0xCA;
        assert!(copy_in_memory(&space, &mem, &b).is_err());
    }
}
