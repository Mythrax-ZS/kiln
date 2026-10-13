//! The block-linear layout of surfaces in the console's GPU memory.
//!
//! A surface is cut into blocks of GOBs ("groups of bytes"): a GOB is 64 bytes wide and 8 rows
//! high (512 bytes), and a block stacks `1 << block_height_log2` GOBs vertically and
//! `1 << block_depth_log2` in depth; blocks are 1 GOB wide and follow one another across and then
//! down and then through the depth. Inside a GOB the bytes are interleaved in the pattern of
//! [`gob_offset`]. Compressed formats use the same layout with a 4x4 block of texels as one
//! element. (docs/ASSUMPTIONS.md; the pattern is the one NVIDIA's Fermi to Maxwell memory layout
//! uses, as nouveau and other emulators implement it.)

/// The size and tiling of one surface (a single mip level of one layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Elements across and down (texels, or 4x4 blocks of texels for a compressed format).
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub bytes_per_element: u32,
    pub block_height_log2: u32,
    pub block_depth_log2: u32,
}

impl Geometry {
    fn row_bytes(&self) -> u64 {
        u64::from(self.width) * u64::from(self.bytes_per_element)
    }

    fn blocks_x(&self) -> u64 {
        self.row_bytes().div_ceil(64)
    }

    fn blocks_y(&self) -> u64 {
        u64::from(self.height)
            .div_ceil(8)
            .div_ceil(1 << self.block_height_log2)
    }

    fn blocks_z(&self) -> u64 {
        u64::from(self.depth).div_ceil(1 << self.block_depth_log2)
    }

    fn block_bytes(&self) -> u64 {
        512 << (self.block_height_log2 + self.block_depth_log2)
    }

    /// The bytes the tiled surface takes, every block whole.
    pub fn tiled_size(&self) -> u64 {
        self.blocks_x() * self.blocks_y() * self.blocks_z() * self.block_bytes()
    }

    /// The bytes of the same surface with rows packed one after another.
    pub fn linear_size(&self) -> u64 {
        self.row_bytes() * u64::from(self.height) * u64::from(self.depth)
    }

    /// The offset in the tiled surface of byte `x` of row `y` of slice `z`.
    pub fn offset(&self, x: u64, y: u32, z: u32) -> u64 {
        let gob_x = x / 64;
        let gob_y = u64::from(y) / 8;
        let block_h = 1u64 << self.block_height_log2;
        let block_d = 1u64 << self.block_depth_log2;
        let block_x = gob_x;
        let block_y = gob_y / block_h;
        let block_z = u64::from(z) / block_d;
        let gob_in_block = (u64::from(z) % block_d) * block_h + gob_y % block_h;
        let block = (block_z * self.blocks_y() + block_y) * self.blocks_x() + block_x;
        block * self.block_bytes()
            + gob_in_block * 512
            + u64::from(gob_offset((x % 64) as u32, y % 8))
    }
}

/// Where byte `x` (0..64) of row `y` (0..8) is inside a GOB.
pub const fn gob_offset(x: u32, y: u32) -> u32 {
    (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + x % 16
}

/// Copy a block-linear surface into packed rows. `tiled` must hold [`Geometry::tiled_size`]
/// bytes (a shorter one makes the missing bytes zero) and `linear` [`Geometry::linear_size`].
pub fn deswizzle(g: &Geometry, tiled: &[u8], linear: &mut [u8]) {
    walk(g, |tiled_at, linear_at, n| {
        let end = tiled_at + n;
        if end <= tiled.len() {
            linear[linear_at..linear_at + n].copy_from_slice(&tiled[tiled_at..end]);
        } else if tiled_at < tiled.len() {
            let have = tiled.len() - tiled_at;
            linear[linear_at..linear_at + have].copy_from_slice(&tiled[tiled_at..]);
            linear[linear_at + have..linear_at + n].fill(0);
        } else {
            linear[linear_at..linear_at + n].fill(0);
        }
    });
}

/// The reverse: put packed rows into a block-linear surface.
pub fn swizzle(g: &Geometry, linear: &[u8], tiled: &mut [u8]) {
    walk(g, |tiled_at, linear_at, n| {
        let end = tiled_at + n;
        if end <= tiled.len() {
            tiled[tiled_at..end].copy_from_slice(&linear[linear_at..linear_at + n]);
        }
    });
}

/// Where each 16-byte chunk of a row of a GOB lies inside it: the chunk across (0..4) in each of
/// the 8 rows.
const CHUNK: [[u16; 4]; 8] = {
    let mut table = [[0u16; 4]; 8];
    let mut y = 0;
    while y < 8 {
        let mut c = 0;
        while c < 4 {
            table[y][c] = gob_offset(16 * c as u32, y as u32) as u16;
            c += 1;
        }
        y += 1;
    }
    table
};

/// Visit the surface in runs of up to 16 bytes that are contiguous in both layouts, calling
/// `f(tiled offset, linear offset, length)`: GOB by GOB, so that the position of a GOB is worked
/// out once for its 32 chunks.
fn walk(g: &Geometry, mut f: impl FnMut(usize, usize, usize)) {
    let row = g.row_bytes() as usize;
    let (blocks_x, blocks_y) = (g.blocks_x() as usize, g.blocks_y() as usize);
    let block_height = 1usize << g.block_height_log2;
    let block_depth = 1usize << g.block_depth_log2;
    let block_bytes = g.block_bytes() as usize;
    let height = g.height as usize;
    for z in 0..g.depth as usize {
        let (block_z, z_in_block) = (z / block_depth, z % block_depth);
        for gob_row in 0..height.div_ceil(8) {
            let (block_y, gob_y) = (gob_row / block_height, gob_row % block_height);
            let gob_in_block = z_in_block * block_height + gob_y;
            let first_block = (block_z * blocks_y + block_y) * blocks_x;
            for gob_x in 0..blocks_x {
                let gob = (first_block + gob_x) * block_bytes + gob_in_block * 512;
                let x0 = gob_x * 64;
                let width = (row - x0).min(64);
                for (y_in_gob, chunks) in CHUNK.iter().enumerate() {
                    let y = gob_row * 8 + y_in_gob;
                    if y >= height {
                        break;
                    }
                    let linear = (z * height + y) * row + x0;
                    for (c, chunk) in chunks.iter().enumerate() {
                        let skipped = 16 * c;
                        if skipped >= width {
                            break;
                        }
                        f(
                            gob + usize::from(*chunk),
                            linear + skipped,
                            (width - skipped).min(16),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    /// The surface walked the way the formula of [`Geometry::offset`] says, a run at a time.
    fn walk_by_the_formula(g: &Geometry) -> Vec<(usize, usize, usize)> {
        let row = g.row_bytes();
        let mut runs = Vec::new();
        for z in 0..g.depth {
            for y in 0..g.height {
                let linear_row = (u64::from(z) * u64::from(g.height) + u64::from(y)) * row;
                let mut x = 0u64;
                while x < row {
                    let n = (row - x).min(16 - x % 16);
                    runs.push((
                        g.offset(x, y, z) as usize,
                        (linear_row + x) as usize,
                        n as usize,
                    ));
                    x += n;
                }
            }
        }
        runs
    }

    #[test]
    fn the_walk_by_gobs_makes_the_runs_the_formula_gives() {
        for (w, h, d, bpe, bh, bd) in [
            (64, 8, 1, 1, 0, 0),
            (16, 16, 1, 4, 1, 0),
            (100, 70, 1, 4, 2, 0),
            (320, 360, 1, 4, 4, 0),
            (33, 17, 1, 2, 3, 0),
            (7, 5, 1, 16, 0, 0),
            (40, 46, 1, 4, 3, 0),
            (13, 9, 1, 1, 5, 0),
            (40, 40, 3, 4, 1, 1),
            (8, 8, 9, 8, 0, 3),
            (1, 1, 1, 4, 0, 0),
        ] {
            let g = Geometry {
                width: w,
                height: h,
                depth: d,
                bytes_per_element: bpe,
                block_height_log2: bh,
                block_depth_log2: bd,
            };
            let mut seen = Vec::new();
            walk(&g, |t, l, n| seen.push((t, l, n)));
            // The same runs, in whatever order: GOB by GOB is not row by row.
            let mut want = walk_by_the_formula(&g);
            seen.sort_unstable();
            want.sort_unstable();
            assert_eq!(seen, want, "{g:?}");
        }
    }

    #[test]
    fn the_gob_pattern_is_a_permutation_of_512_bytes_with_known_corners() {
        let mut seen = [false; 512];
        for y in 0..8 {
            for x in 0..64 {
                let o = gob_offset(x, y) as usize;
                assert!(!seen[o], "({x},{y}) lands on a used byte");
                seen[o] = true;
            }
        }
        assert!(seen.iter().all(|s| *s));
        assert_eq!(gob_offset(0, 0), 0);
        assert_eq!(gob_offset(15, 0), 15);
        assert_eq!(gob_offset(16, 0), 32);
        assert_eq!(gob_offset(32, 0), 256);
        assert_eq!(gob_offset(0, 1), 16);
        assert_eq!(gob_offset(0, 2), 64);
        assert_eq!(gob_offset(63, 7), 511);
    }

    #[test]
    fn the_sizes_are_whole_blocks() {
        let g = Geometry {
            width: 100,
            height: 70,
            depth: 1,
            bytes_per_element: 4,
            block_height_log2: 2,
            block_depth_log2: 0,
        };
        // 400 bytes a row is 7 GOBs across; 70 rows is 9 GOBs high, in blocks of 4: 3.
        assert_eq!(g.tiled_size(), 7 * 3 * 4 * 512);
        assert_eq!(g.linear_size(), 100 * 70 * 4);
    }

    #[test]
    fn a_surface_survives_the_round_trip_for_many_shapes() {
        for (w, h, d, bpe, bh, bd) in [
            (64, 8, 1, 1, 0, 0),
            (16, 16, 1, 4, 1, 0),
            (100, 70, 1, 4, 2, 0),
            (1600, 900, 1, 4, 4, 0),
            (33, 17, 1, 2, 3, 0),
            (7, 5, 1, 16, 0, 0),
            (40, 40, 3, 4, 1, 1),
            (8, 8, 9, 8, 0, 3),
        ] {
            let g = Geometry {
                width: w,
                height: h,
                depth: d,
                bytes_per_element: bpe,
                block_height_log2: bh,
                block_depth_log2: bd,
            };
            let linear = pattern(g.linear_size() as usize);
            let mut tiled = vec![0xAAu8; g.tiled_size() as usize];
            swizzle(&g, &linear, &mut tiled);
            let mut back = vec![0u8; g.linear_size() as usize];
            deswizzle(&g, &tiled, &mut back);
            assert_eq!(back, linear, "{g:?}");
        }
    }

    #[test]
    fn a_pixel_lands_where_the_formula_says() {
        // A 128x16 surface of 4-byte pixels with blocks of two GOBs: pixel (20, 9) is in the
        // second GOB row (y 8..16), which is the second GOB of the first block; byte x = 80 is in
        // the second GOB across, a second block.
        let g = Geometry {
            width: 32,
            height: 16,
            depth: 1,
            bytes_per_element: 4,
            block_height_log2: 1,
            block_depth_log2: 0,
        };
        let x: u64 = 20 * 4;
        let expected = 2 * 512            // the second block, as blocks are 1 GOB wide
            + 512                         // the second GOB of the block
            + u64::from(gob_offset((x % 64) as u32, 9 % 8));
        assert_eq!(g.offset(x, 9, 0), expected);
    }

    #[test]
    fn a_short_source_reads_as_zeros() {
        let g = Geometry {
            width: 16,
            height: 8,
            depth: 1,
            bytes_per_element: 4,
            block_height_log2: 0,
            block_depth_log2: 0,
        };
        let tiled = vec![0xFFu8; 100];
        let mut out = vec![1u8; g.linear_size() as usize];
        deswizzle(&g, &tiled, &mut out);
        let ones = out.iter().filter(|b| **b == 0xFF).count();
        assert_eq!(ones, 100);
        assert!(out.iter().all(|b| *b == 0xFF || *b == 0));
    }
}
