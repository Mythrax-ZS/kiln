//! Copies of the 2D engine: a render target copied into the memory of another surface is the
//! picture a target there would hold.

mod render_common;

use nx_gpu::draw_state::Layout;
use nx_gpu::render::{Backend, Blit, Draw, Source};
use nx_gpu::twod::Surface;
use nx_vulkan::{RenderConfig, Renderer};
use render_common::*;

const FORMAT: u32 = 0xD5;

fn surface(address: u64, size: u32) -> Surface {
    Surface {
        address,
        format: FORMAT,
        layout: Layout::BlockLinear { block: [0, 4, 0] },
        pitch: 0,
        width: size,
        height: size,
        depth: 1,
        layer: 0,
    }
}

/// A copy of all of `src` (of `src_size` square) into `dst` (of `dst_size` square) without a
/// change of format.
fn copy(src: u64, src_size: u32, dst: u64, dst_size: u32) -> Blit {
    let step = i64::from(src_size) * (1 << 32) / i64::from(dst_size);
    Blit {
        dst: surface(dst, dst_size),
        src: surface(src, src_size),
        operation: 3,
        dst_x: 0,
        dst_y: 0,
        dst_width: dst_size,
        dst_height: dst_size,
        // The center of the first source pixel, the origin being the corner of a pixel.
        src_x: step / 2,
        src_y: step / 2,
        du_dx: step,
        dv_dy: step,
        bilinear: false,
        corner_origin: true,
        clip: None,
    }
}

/// A renderer that has drawn a green triangle over a `size` square target at `target`.
fn green_target(w: &mut World, target: u64, size: u32) -> Renderer {
    *w = scene(&pass_through_vertex(), &flat_color_pixel(), size, target);
    let green = [0.0, 1.0, 0.0, 1.0];
    vertices(
        w,
        GPU_BASE + 0x30_0000,
        &[
            ([-1.0, -1.0, 0.5, 1.0], green),
            ([3.0, -1.0, 0.5, 1.0], green),
            ([-1.0, 3.0, 0.5, 1.0], green),
        ],
    );
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(
        &w.scene(),
        &Draw {
            source: Source::Arrays { first: 0, count: 3 },
            topology: 4,
            instance: 0,
            instances: 1,
        },
    )
    .unwrap();
    r
}

#[test]
fn a_render_target_copied_into_another_surface_is_its_picture() {
    if vulkan().is_none() {
        return;
    }
    let (from, to) = (GPU_BASE + 0x20_0000, GPU_BASE + 0x50_0000);
    let mut w = World::new();
    let mut r = green_target(&mut w, from, 16);
    r.blit(&w.scene(), &copy(from, 16, to, 16)).unwrap();
    let (rgba, width, height) = r.picture_at(w.cpu(to)).unwrap().expect("a target");
    assert_eq!((width, height), (16, 16));
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[0, 255, 0, 255]);
    }
}

#[test]
fn a_copy_that_scales_down_makes_a_smaller_picture() {
    if vulkan().is_none() {
        return;
    }
    let (from, to) = (GPU_BASE + 0x20_0000, GPU_BASE + 0x50_0000);
    let mut w = World::new();
    let mut r = green_target(&mut w, from, 16);
    r.blit(&w.scene(), &copy(from, 16, to, 8)).unwrap();
    let (rgba, width, height) = r.picture_at(w.cpu(to)).unwrap().expect("a target");
    assert_eq!((width, height), (8, 8));
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[0, 255, 0, 255]);
    }
}

/// How many pixels of the surface's memory hold `rgba`.
fn pixels_holding(w: &World, s: &Surface, rgba: [u8; 4]) -> usize {
    use nx_gpu::GuestMemory;
    let mut bytes = vec![0u8; s.memory_bytes(4) as usize];
    assert!(w.mem.read(w.cpu(s.address), &mut bytes));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| **p == rgba)
        .count()
}

#[test]
fn what_the_device_cannot_copy_is_copied_in_memory() {
    if vulkan().is_none() {
        return;
    }
    let (from, to) = (GPU_BASE + 0x20_0000, GPU_BASE + 0x50_0000);
    let mut w = World::new();
    let mut r = green_target(&mut w, from, 16);
    // From memory no render target stands for (A-249): its pixels, wherever they are in the
    // layout, are the destination's.
    let plain = GPU_BASE + 0x70_0000;
    let src = surface(plain, 16);
    let pattern: Vec<u8> = (0..src.memory_bytes(4) / 4)
        .flat_map(|_| [1u8, 2, 3, 4])
        .collect();
    w.put(plain, &pattern);
    r.blit(&w.scene(), &copy(plain, 16, to, 16)).unwrap();
    assert_eq!(pixels_holding(&w, &surface(to, 16), [1, 2, 3, 4]), 16 * 16);
    // Into a part of a surface: the render target's pixels are put into its memory first.
    let elsewhere = GPU_BASE + 0x60_0000;
    let mut part = copy(from, 16, elsewhere, 16);
    part.dst_width = 8;
    r.blit(&w.scene(), &part).unwrap();
    assert_eq!(
        pixels_holding(&w, &surface(elsewhere, 16), [0, 255, 0, 255]),
        8 * 16
    );
}

#[test]
fn what_no_copy_can_make_stops_with_its_parameters() {
    if vulkan().is_none() {
        return;
    }
    let (from, to) = (GPU_BASE + 0x20_0000, GPU_BASE + 0x50_0000);
    let mut w = World::new();
    let mut r = green_target(&mut w, from, 16);
    // An operation that is not a copy.
    let mut blend = copy(from, 16, to, 16);
    blend.operation = 2;
    let e = r.blit(&w.scene(), &blend).unwrap_err();
    assert!(e.contains("operation 2"), "{e}");
    // From memory into another format.
    let mut convert = copy(GPU_BASE + 0x70_0000, 16, to, 16);
    convert.dst.format = 0xE9;
    let e = r.blit(&w.scene(), &convert).unwrap_err();
    assert!(e.contains("into format 0xe9"), "{e}");
}
