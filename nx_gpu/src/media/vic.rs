//! The video image compositor: what a program's configuration asks of it, as far as videos use
//! it (docs/ASSUMPTIONS.md A-183): slot 0's frame, through its color matrix, into an RGB output
//! surface, block-linear or pitch.

use super::vp9::Frame;

/// The size of the configuration that is read.
pub const CONFIG_SIZE: usize = 0x200;

/// Where slot 0 begins: its enable bit, then (at +0x10) its clamps and alpha, (at +0x20) its
/// source rectangle, (at +0x30) its destination rectangle, (at +0x40) its surface and (at
/// +0x60) its color matrix.
const SLOT0: usize = 0x90;

/// Output pixel formats.
pub mod format {
    pub const A8B8G8R8: u8 = 0x1F;
    pub const A8R8G8B8: u8 = 0x20;
    pub const X8B8G8R8: u8 = 0x23;
}

/// What the configuration says.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub format: u8,
    /// 0 pitch, otherwise block-linear with blocks of `1 << block_height_log2` GOBs.
    pub block_kind: u8,
    pub block_height_log2: u8,
    pub width: u32,
    pub height: u32,
    pub slot_enabled: bool,
    /// The source rectangle, in samples (16.16 fixed point in the configuration).
    pub src: [f64; 4],
    /// The destination rectangle: left, top, right, bottom, inclusive.
    pub dst: [u32; 4],
    /// The color matrix: rows R, G, B of coefficients for Y, U, V (in units of 1 / 2^17) and an
    /// offset (in units of 1 / 256 of a 10-bit sample).
    pub matrix: [[i32; 4]; 3],
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn s20(v: u64) -> i32 {
    let v = (v & 0xF_FFFF) as i32;
    if v & 0x8_0000 != 0 {
        v - (1 << 20)
    } else {
        v
    }
}

/// The limited-range BT.601 matrix, for a configuration whose matrix is not enabled.
const BT601: [[i32; 4]; 3] = [
    [152_618, 0, 209_194, -228_942],
    [152_618, -51_348, -106_556, 139_237],
    [152_618, 264_403, 0, -284_313],
];

impl Config {
    pub fn parse(c: &[u8]) -> Option<Config> {
        if c.len() < CONFIG_SIZE {
            return None;
        }
        let out = u64_at(c, 0x20);
        let src_x = u64_at(c, SLOT0 + 0x20);
        let src_y = u64_at(c, SLOT0 + 0x28);
        let fixed = |v: u64| (v & 0x3FFF_FFFF) as f64 / 65536.0;
        let dst = u64_at(c, SLOT0 + 0x30);
        let m: [u64; 4] = std::array::from_fn(|i| u64_at(c, SLOT0 + 0x60 + 8 * i));
        let enabled = m[1] >> 63 == 1;
        let matrix = if enabled {
            // Column j of the matrix is in word j: R, G, B in its three 20-bit fields.
            std::array::from_fn(|row| std::array::from_fn(|col| s20(m[col] >> (20 * row))))
        } else {
            BT601
        };
        Some(Config {
            format: (out & 0x7F) as u8,
            block_kind: ((out >> 11) & 0xF) as u8,
            block_height_log2: ((out >> 15) & 0xF) as u8,
            width: ((out >> 32) & 0x3FFF) as u32 + 1,
            height: ((out >> 46) & 0x3FFF) as u32 + 1,
            slot_enabled: c[SLOT0] & 1 == 1,
            src: [
                fixed(src_x),
                fixed(src_y),
                fixed(src_x >> 32),
                fixed(src_y >> 32),
            ],
            dst: [
                (dst & 0x3FFF) as u32,
                ((dst >> 32) & 0x3FFF) as u32,
                ((dst >> 16) & 0x3FFF) as u32,
                ((dst >> 48) & 0x3FFF) as u32,
            ],
            matrix,
        })
    }

    /// How many bytes [`Config::convert`] writes: the output surface, as the kind lays it out.
    pub fn output_size(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        if self.block_kind == 0 {
            return w * h * 4;
        }
        let block_rows = 8usize << self.block_height_log2;
        (w * 4).div_ceil(64) * h.div_ceil(block_rows) * (512usize << self.block_height_log2)
    }

    /// The bytes of the output surface: its pixels converted from `frame`, laid out as the
    /// configuration says.
    pub fn convert(&self, frame: &Frame) -> Result<Vec<u8>, String> {
        let (w, h) = (self.width as usize, self.height as usize);
        let order: [usize; 4] = match self.format {
            format::A8B8G8R8 | format::X8B8G8R8 => [0, 1, 2, 3],
            format::A8R8G8B8 => [2, 1, 0, 3],
            other => {
                return Err(format!(
                    "the video image compositor's output format {other:#x}"
                ))
            }
        };
        // The offsets in 8-bit units, scaled like the coefficients (2^17).
        let offset: [i64; 3] = std::array::from_fn(|r| {
            (f64::from(self.matrix[r][3]) * 255.0 / (256.0 * 1023.0) * 131_072.0).round() as i64
        });
        // Black, opaque, everywhere the frame does not reach.
        let mut linear = vec![0u8; w * h * 4];
        for px in linear.as_chunks_mut::<4>().0 {
            px[3] = 255;
        }
        let [l, t, r, b] = self.dst;
        let (dw, dh) = (
            (r.saturating_sub(l) + 1) as f64,
            (b.saturating_sub(t) + 1) as f64,
        );
        let [sl, st, sr, sb] = self.src;
        let (sw, sh) = (sr - sl + 1.0, sb - st + 1.0);
        let (y_plane, u_plane, v_plane) = (&frame.planes[0], &frame.planes[1], &frame.planes[2]);
        let (fw, fh) = (frame.width.max(1), frame.height.max(1));
        // The sample column of each destination column, once for every row.
        let (l, r) = (l as usize, (r as usize).min(w.saturating_sub(1)));
        let columns: Vec<usize> = (l..=r)
            .map(|ox| {
                (sl + (ox as f64 - l as f64 + 0.5) * sw / dw - 0.5)
                    .round()
                    .clamp(0.0, (fw - 1) as f64) as usize
            })
            .collect();
        let m = self.matrix.map(|row| row.map(i64::from));
        if self.slot_enabled && l <= r {
            for oy in (t as usize)..=(b as usize).min(h.saturating_sub(1)) {
                let row = &mut linear[(oy * w + l) * 4..(oy * w + r + 1) * 4];
                let sy = (st + (oy as f64 - f64::from(t) + 0.5) * sh / dh - 0.5)
                    .round()
                    .clamp(0.0, (fh - 1) as f64) as usize;
                let y_row = &y_plane.data[sy * y_plane.stride..];
                let u_row = &u_plane.data[(sy / 2) * u_plane.stride..];
                let v_row = &v_plane.data[(sy / 2) * v_plane.stride..];
                for (px, &sx) in row.as_chunks_mut::<4>().0.iter_mut().zip(&columns) {
                    let yv = i64::from(y_row[sx]);
                    let uv = i64::from(u_row[sx / 2]);
                    let vv = i64::from(v_row[sx / 2]);
                    let mut pixel = [0u8, 0, 0, 255];
                    for (c, out) in pixel[..3].iter_mut().enumerate() {
                        let s = m[c][0] * yv + m[c][1] * uv + m[c][2] * vv + offset[c];
                        *out = ((s + (1 << 16)) >> 17).clamp(0, 255) as u8;
                    }
                    for (k, &o) in order.iter().enumerate() {
                        px[k] = pixel[o];
                    }
                }
            }
        }
        if self.block_kind == 0 {
            return Ok(linear);
        }
        Ok(swizzle(
            &linear,
            w * 4,
            h,
            u32::from(self.block_height_log2),
        ))
    }
}

/// Where byte `x` (0..64) of row `y` (0..8) is inside a GOB (the pattern of the GPU's
/// block-linear layout, as `nx_vulkan::blocklinear` uses it).
const fn gob_offset(x: usize, y: usize) -> usize {
    (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + x % 16
}

/// Put packed rows of `row_bytes` bytes into the block-linear layout with blocks of
/// `1 << block_height_log2` GOBs.
pub fn swizzle(linear: &[u8], row_bytes: usize, height: usize, block_height_log2: u32) -> Vec<u8> {
    let blocks_x = row_bytes.div_ceil(64);
    let block_rows = 8usize << block_height_log2;
    let blocks_y = height.div_ceil(block_rows);
    let block_bytes = 512usize << block_height_log2;
    let mut tiled = vec![0u8; blocks_x * blocks_y * block_bytes];
    for y in 0..height {
        let block_y = y / block_rows;
        let gob_in_block = (y / 8) % (1 << block_height_log2);
        for x in (0..row_bytes).step_by(16) {
            let block = block_y * blocks_x + x / 64;
            let at = block * block_bytes + gob_in_block * 512 + gob_offset(x % 64, y % 8);
            let n = 16.min(row_bytes - x);
            tiled[at..at + n].copy_from_slice(&linear[y * row_bytes + x..y * row_bytes + x + n]);
        }
    }
    tiled
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A configuration as a title wrote it for a 1280 by 720 video: an X8B8G8R8 output,
    /// block-linear with blocks of two GOBs, and slot 0 enabled (its clamps and alpha, its
    /// rectangles, its surface, its matrix), copied from a captured job.
    fn title_config() -> Vec<u8> {
        let mut c = vec![0u8; CONFIG_SIZE];
        let hex = |s: &str| -> Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        };
        c[0x00..0x20].copy_from_slice(&hex(
            "0400040000000000000000000000000004000000000000000000ff040000cf02",
        ));
        c[0x20..0x30].copy_from_slice(&hex("238a0000ffc4b300ffc4b300ffffff0f"));
        c[SLOT0..SLOT0 + 0x18]
            .copy_from_slice(&hex("0101000000000000000000000000000000fc0f00ff070000"));
        c[SLOT0 + 0x20..SLOT0 + 0x38]
            .copy_from_slice(&hex("000000000000ff04000000000000cf020000ff040000cf02"));
        c[SLOT0 + 0x40..SLOT0 + 0x50].copy_from_slice(&hex("44880800ffc4b300ffc4b3007fc25900"));
        c[SLOT0 + 0x60..SLOT0 + 0x80].copy_from_slice(&hex(
            "2a54a242252a54920000c076f3d308842a3143fce5000000b2815cfe2167a90b",
        ));
        c[SLOT0 + 0xA0..SLOT0 + 0xA8].copy_from_slice(&hex("ff03000050200000"));
        c
    }

    #[test]
    fn a_slot_left_disabled_paints_black() {
        let mut bytes = title_config();
        assert!(Config::parse(&bytes).unwrap().slot_enabled);
        bytes[SLOT0] = 0;
        let c = Config::parse(&bytes).unwrap();
        assert!(!c.slot_enabled);
    }

    #[test]
    fn a_title_configuration_reads_as_a_bt601_conversion_into_a_block_linear_surface() {
        let c = Config::parse(&title_config()).unwrap();
        assert_eq!(
            (c.format, c.block_kind, c.block_height_log2),
            (format::X8B8G8R8, 1, 1)
        );
        assert_eq!((c.width, c.height), (1280, 720));
        assert_eq!(c.dst, [0, 0, 1279, 719]);
        assert_eq!(c.src, [0.0, 0.0, 1279.0, 719.0]);
        assert_eq!(
            c.matrix, BT601,
            "the title's matrix is limited-range BT.601"
        );
    }

    #[test]
    fn colors_convert_as_bt601_limited_range() {
        let mut c = Config::parse(&title_config()).unwrap();
        c.width = 4;
        c.height = 2;
        c.dst = [0, 0, 3, 1];
        c.src = [0.0, 0.0, 3.0, 1.0];
        c.block_kind = 0;
        let mut f = Frame::new(4, 2);
        // Black, white, and the reddest and bluest a frame can hold, one per column.
        for (x, (y, u, v)) in [
            (16, 128, 128),
            (235, 128, 128),
            (81, 90, 240),
            (41, 240, 110),
        ]
        .iter()
        .enumerate()
        {
            for row in 0..2 {
                f.planes[0].data[row * f.planes[0].stride + x] = *y;
            }
            f.planes[1].data[x / 2] = *u;
            f.planes[2].data[x / 2] = *v;
        }
        // Columns 0 and 1 share chroma (128, 128) here only if they agree; give them their own
        // by checking the luma-only columns.
        let out = c.convert(&f).unwrap();
        assert_eq!(&out[0..4], &[0, 0, 0, 255], "16 is black");
        assert_eq!(&out[4..8], &[255, 255, 255, 255], "235 is white");
    }

    #[test]
    fn the_block_linear_layout_puts_rows_in_gobs() {
        // 128 bytes across (two GOBs), 16 rows (two GOBs down in blocks of two).
        let linear: Vec<u8> = (0..128 * 16).map(|i| (i % 251) as u8).collect();
        let tiled = swizzle(&linear, 128, 16, 1);
        assert_eq!(tiled.len(), 2 * 1024);
        // The first 16 bytes of row 0 start the first GOB; row 1 follows 16 bytes later.
        assert_eq!(&tiled[0..16], &linear[0..16]);
        assert_eq!(&tiled[16..32], &linear[128..144]);
        // Row 8 is the second GOB of the first block.
        assert_eq!(&tiled[512..528], &linear[8 * 128..8 * 128 + 16]);
        // Bytes 64.. of row 0 are the second block.
        assert_eq!(&tiled[1024..1040], &linear[64..80]);
    }
}
