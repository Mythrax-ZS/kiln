//! The device's layout of surfaces into block-linear memory (`shaders/block_linear.comp`, A-292)
//! against the host's (`nx_vulkan::blocklinear::swizzle`): the same bytes for surfaces of many
//! shapes, the device writing by address into memory it reaches as a pointer.

mod common;

use common::vulkan;
use nx_vulkan::blocklinear::{device_params, device_spirv, swizzle, Geometry};
use nx_vulkan::compute::AddressPatch;

#[test]
fn the_device_lays_surfaces_out_as_the_host_does() {
    let Some(vk) = vulkan() else { return };
    let spirv = device_spirv();
    // (width, height, depth, bytes per element, block height, block depth), each whole blocks.
    let shapes = [
        (16, 8, 1, 4, 0, 0),
        (64, 64, 1, 4, 3, 0),
        (32, 48, 1, 8, 1, 0),
        (128, 32, 1, 2, 2, 0),
        (64, 16, 1, 1, 1, 0),
        (8, 16, 4, 16, 1, 2),
        (48, 16, 2, 4, 0, 1),
    ];
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for (width, height, depth, bytes_per_element, block_height_log2, block_depth_log2) in shapes {
        let g = Geometry {
            width,
            height,
            depth,
            bytes_per_element,
            block_height_log2,
            block_depth_log2,
        };
        assert!(g.fills_its_blocks(), "{g:?}");
        let mut linear: Vec<u8> = (0..g.linear_size())
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 24) as u8
            })
            .collect();
        let mut want = vec![0u8; g.tiled_size() as usize];
        swizzle(&g, &linear, &mut want);
        let units = g.linear_size() / 16;
        // Few workgroups too, so that each invocation goes on through the units.
        for groups in [units.div_ceil(64) as u32, 3] {
            let mut params = device_params(&g, 0, 0);
            let mut got = vec![0u8; g.tiled_size() as usize];
            let mut list: Vec<(u32, &mut [u8])> = vec![
                (0, &mut params[..]),
                (1, &mut linear[..]),
                (2, &mut got[..]),
            ];
            let patch = [AddressPatch {
                into: 0,
                offset: 0,
                of: 2,
                addend: 0,
            }];
            vk.run_compute_with(&spirv, &mut list, &[], &patch, [groups, 1, 1])
                .unwrap();
            assert!(got == want, "{g:?} in {groups} workgroups");
        }
    }
}
