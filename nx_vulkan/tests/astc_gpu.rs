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
