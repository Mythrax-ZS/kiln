//! The ASTC decoder of the device (`shaders/astc_decode.comp`) against the one of the host
//! (`nx_vulkan::astc`, itself checked against ARM's reference decoder): the same bytes for random
//! blocks of every footprint, legal and illegal, sRGB or not, with images that end inside a block.

mod common;

use common::vulkan;
use nx_vulkan::astc::{decode_image, inspect};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn block(&mut self) -> [u8; 16] {
        let mut b = [0u8; 16];
        for chunk in b.chunks_mut(4) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        b
    }
}

const FOOTPRINTS: [(u32, u32); 14] = [
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
];

/// Blocks across and down of each test image.
const ACROSS: u32 = 64;
const DOWN: u32 = 64;

#[test]
fn the_device_decodes_astc_blocks_as_the_host_does() {
    let Some(vk) = vulkan() else { return };
    let spirv = nx_vulkan::astc_gpu::spirv();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for (bw, bh) in FOOTPRINTS {
        for srgb in [false, true] {
            // Half the blocks legal ones, half anything (mostly illegal: the error color).
            let mut blocks = Vec::new();
            for i in 0..ACROSS * DOWN {
                let block = loop {
                    let b = rng.block();
                    if i % 2 == 1 || inspect(&b, bw as usize, bh as usize).is_ok() {
                        break b;
                    }
                };
                blocks.extend_from_slice(&block);
            }
            // The image ends a texel inside the last column and row of blocks.
            let (width, height) = (ACROSS * bw - 1, DOWN * bh - 1);
            let want = decode_image(&blocks, width, height, bw, bh, srgb);
            let params: Vec<u8> = [width, height, bw, bh, ACROSS, DOWN, u32::from(srgb), 0, 0]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let mut params = params;
            let mut texels = vec![0u8; (width * height * 4) as usize];
            let mut list: Vec<(u32, &mut [u8])> = vec![
                (0, &mut params[..]),
                (1, &mut blocks[..]),
                (2, &mut texels[..]),
            ];
            vk.run_compute_with(
                &spirv,
                &mut list,
                &[],
                &[],
                [(ACROSS * DOWN).div_ceil(64), 1, 1],
            )
            .unwrap();
            drop(list);
            let differ = texels
                .as_chunks::<4>()
                .0
                .iter()
                .zip(want.as_chunks::<4>().0)
                .position(|(a, b)| a != b);
            if let Some(i) = differ {
                let (x, y) = (i as u32 % width, i as u32 / width);
                let block = ((y / bh) * ACROSS + x / bw) as usize;
                panic!(
                    "{bw}x{bh} (sRGB {srgb}): texel ({x}, {y}) of block {:02x?} is {:?} on the device and {:?} on the host",
                    &blocks[block * 16..block * 16 + 16],
                    &texels[i * 4..i * 4 + 4],
                    &want[i * 4..i * 4 + 4]
                );
            }
        }
    }
}

/// The BC7 encoder of the device (A-291), decoded on the host: a block of one color comes back
/// within a step of it, smooth gradients within a few, and anything at all within reason, in an
/// image that ends inside its last column and row of blocks.
#[test]
fn the_device_encodes_texels_as_bc7_close_to_them() {
    let Some(vk) = vulkan() else { return };
    let spirv = nx_vulkan::astc_gpu::bc7_spirv();
    let (width, height) = (63u32, 37u32);
    let (across, down) = (width.div_ceil(4), height.div_ceil(4));
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    // Four kinds of picture: flat colors a block each, gradients along one direction (the colors
    // of a block on a line), gradients along two (on a plane, which one line of mode 6 can only
    // come near), and noise.
    for kind in 0..4 {
        let mut texels = vec![0u8; (width * height * 4) as usize];
        let flat: Vec<[u8; 4]> = (0..across * down)
            .map(|_| rng.next().to_le_bytes())
            .collect();
        for y in 0..height {
            for x in 0..width {
                let t = match kind {
                    0 => flat[((y / 4) * across + x / 4) as usize],
                    1 => [
                        (x * 4) as u8,
                        (x * 3 + 20) as u8,
                        (200 - x * 2) as u8,
                        (x * 4) as u8,
                    ],
                    2 => [
                        (x * 4) as u8,
                        (y * 6) as u8,
                        ((x + y) * 2) as u8,
                        (255 - x * 2) as u8,
                    ],
                    _ => rng.next().to_le_bytes(),
                };
                let at = ((y * width + x) * 4) as usize;
                texels[at..at + 4].copy_from_slice(&t);
            }
        }
        let mut params: Vec<u8> = [width, height, across, down, 0, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let mut source = texels.clone();
        let mut blocks = vec![0u8; (across * down * 16) as usize];
        let mut list: Vec<(u32, &mut [u8])> = vec![
            (0, &mut params[..]),
            (1, &mut source[..]),
            (2, &mut blocks[..]),
        ];
        vk.run_compute_with(
            &spirv,
            &mut list,
            &[],
            &[],
            [(across * down).div_ceil(64), 1, 1],
        )
        .unwrap();
        drop(list);
        let (mut worst, mut squares, mut count) = (0u32, 0f64, 0f64);
        for by in 0..down {
            for bx in 0..across {
                let at = ((by * across + bx) * 16) as usize;
                let block: [u8; 16] = blocks[at..at + 16].try_into().unwrap();
                let decoded = nx_vulkan::bc7::decode_mode6(&block).expect("a block of mode 6");
                for (i, d) in decoded.iter().enumerate() {
                    let (x, y) = (bx * 4 + i as u32 % 4, by * 4 + i as u32 / 4);
                    if x >= width || y >= height {
                        continue;
                    }
                    let t = &texels[((y * width + x) * 4) as usize..][..4];
                    for c in 0..4 {
                        let e = (i32::from(d[c]) - i32::from(t[c])).unsigned_abs();
                        worst = worst.max(e);
                        squares += f64::from(e * e);
                        count += 1.0;
                    }
                }
            }
        }
        let rms = (squares / count).sqrt();
        match kind {
            0 => assert!(worst <= 1, "flat blocks: worst error {worst}"),
            1 => assert!(
                rms <= 1.0,
                "gradients on a line: RMS error {rms:.2}, worst {worst}"
            ),
            2 => assert!(
                rms <= 4.0,
                "gradients on a plane: RMS error {rms:.2}, worst {worst}"
            ),
            _ => assert!(rms <= 64.0, "noise: RMS error {rms:.2}"),
        }
    }
}
