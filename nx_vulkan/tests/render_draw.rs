//! Draws through the whole path: programs in GPU memory, state in registers, vertices in memory,
//! a picture out.

mod render_common;

use nx_gpu::render::{Backend, Draw, Source};
use nx_shader::asm::{exit, Enc};
use nx_shader::inst::Guard;
use nx_shader::ops::Op;
use nx_vulkan::{RenderConfig, Renderer};
use render_common::*;

#[test]
fn a_triangle_covering_the_target_is_painted_with_its_flat_color() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 64, target);
    let green = [0.0, 1.0, 0.0, 1.0];
    vertices(
        &mut w,
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
        },
    )
    .unwrap();
    r.finish().unwrap();
    let (rgba, width, height) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    assert_eq!((width, height), (64, 64));
    for (i, px) in rgba.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(px, &[0, 255, 0, 255], "pixel {}", i);
    }
}

#[test]
fn a_smaller_triangle_covers_part_of_the_target() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 64, target);
    let red = [1.0, 0.0, 0.0, 1.0];
    // The lower half of the target in clip space: y from -1 up to 0 at the left edge.
    vertices(
        &mut w,
        GPU_BASE + 0x30_0000,
        &[
            ([-1.0, -1.0, 0.5, 1.0], red),
            ([1.0, -1.0, 0.5, 1.0], red),
            ([-1.0, 1.0, 0.5, 1.0], red),
        ],
    );
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(
        &w.scene(),
        &Draw {
            source: Source::Arrays { first: 0, count: 3 },
            topology: 4,
            instance: 0,
        },
    )
    .unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    let at = |x: usize, y: usize| rgba[(y * width as usize + x) * 4..][..4].to_vec();
    // With the viewport as set (scale and offset positive, y not flipped), clip y of -1 is
    // window y 0: the triangle has its right angle at the top left of the picture.
    assert_eq!(at(2, 2), [255, 0, 0, 255], "near the right angle");
    assert_eq!(at(60, 60), [0, 0, 0, 0], "the corner away from it is empty");
    assert_eq!(at(20, 10), [255, 0, 0, 255]);
    assert_eq!(at(10, 50), [255, 0, 0, 255]);
}

#[test]
fn indexed_quads_are_drawn_as_two_triangles() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 32, target);
    let blue = [0.0, 0.0, 1.0, 1.0];
    // Four corners of the right half (x from 0 to 1), the vertices not in drawing order.
    vertices(
        &mut w,
        GPU_BASE + 0x30_0000,
        &[
            ([0.0, 1.0, 0.5, 1.0], blue),
            ([1.0, 1.0, 0.5, 1.0], blue),
            ([1.0, -1.0, 0.5, 1.0], blue),
            ([0.0, -1.0, 0.5, 1.0], blue),
        ],
    );
    // The index buffer: 16-bit indices 0 1 2 3, a quad.
    let ib = GPU_BASE + 0x31_0000;
    let idx: Vec<u8> = [0u16, 1, 2, 3]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    w.put(ib, &idx);
    w.set_address(0x17c8, ib);
    w.set_address(0x17d0, ib + 7);
    w.set(0x17d8, 1);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(
        &w.scene(),
        &Draw {
            source: Source::Indexed {
                first: 0,
                count: 4,
                size: None,
            },
            topology: 7,
            instance: 0,
        },
    )
    .unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    let at = |x: usize, y: usize| rgba[(y * width as usize + x) * 4..][..4].to_vec();
    assert_eq!(at(24, 8), [0, 0, 255, 255]);
    assert_eq!(at(20, 28), [0, 0, 255, 255]);
    assert_eq!(at(31, 31), [0, 0, 255, 255]);
    assert_eq!(at(8, 8), [0, 0, 0, 0], "the left half is empty");
}

#[test]
fn blending_mixes_with_what_is_there() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 16, target);
    let quad = |color: [f32; 4]| {
        [
            ([-1.0, -1.0, 0.5, 1.0], color),
            ([3.0, -1.0, 0.5, 1.0], color),
            ([-1.0, 3.0, 0.5, 1.0], color),
        ]
    };
    let draw = Draw {
        source: Source::Arrays { first: 0, count: 3 },
        topology: 4,
        instance: 0,
    };
    let mut r = Renderer::new(RenderConfig::default());
    vertices(&mut w, GPU_BASE + 0x30_0000, &quad([1.0, 0.0, 0.0, 1.0]));
    r.draw(&w.scene(), &draw).unwrap();
    // Then 40% blue over it with source alpha: the usual "over" blend.
    vertices(&mut w, GPU_BASE + 0x31_0000, &quad([0.0, 0.0, 1.0, 0.5]));
    w.set(0x1360, 1);
    w.set(0x133c, 0);
    w.set(0x1340, 0x8006);
    w.set(0x1344, 0x4302);
    w.set(0x1348, 0x4303);
    r.draw(&w.scene(), &draw).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    let px = &rgba[..4];
    // Half red, half blue, with the alpha blended the same way (0.5 * 0.5 + 1 * 0.5 = 0.75).
    assert!((i32::from(px[0]) - 128).abs() <= 2, "{px:?}");
    assert_eq!(px[1], 0);
    assert!((i32::from(px[2]) - 128).abs() <= 2, "{px:?}");
    assert!((i32::from(px[3]) - 191).abs() <= 2, "{px:?}");
}

#[test]
fn a_program_that_cannot_be_translated_stops_the_draw_with_its_address() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    // A vertex program with an instruction the translator has no translation for (ATOM).
    let atom = Enc::op(Op::AtomReg).build();
    let bad = vertex_program(&[atom, exit(Guard::ALWAYS)]);
    let mut w = scene(&bad, &flat_color_pixel(), 16, target);
    vertices(
        &mut w,
        GPU_BASE + 0x30_0000,
        &[
            ([0.0; 4], [0.0; 4]),
            ([0.0; 4], [0.0; 4]),
            ([0.0; 4], [0.0; 4]),
        ],
    );
    let mut r = Renderer::new(RenderConfig::default());
    let err = r
        .draw(
            &w.scene(),
            &Draw {
                source: Source::Arrays { first: 0, count: 3 },
                topology: 4,
                instance: 0,
            },
        )
        .unwrap_err();
    assert!(err.contains("VERTEX program at 0x100100000"), "{err}");
    assert!(err.contains("is not translated"), "{err}");
}
