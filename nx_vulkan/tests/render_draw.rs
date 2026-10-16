//! Draws through the whole path: programs in GPU memory, state in registers, vertices in memory,
//! a picture out.

mod render_common;

use nx_gpu::render::{Backend, Draw, Source};
use nx_shader::asm::{exit, mov32i, Enc};
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
fn an_attribute_that_is_switched_off_reads_as_zero_zero_zero_one() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    // The program takes its color from location 2, which the state leaves switched off.
    let vertex = vertex_program(&[
        ald(0, 0x80, 4),
        ald(4, 0xA0, 4),
        ast(0, 0x70, 4),
        ast(4, 0x80, 4),
        exit(Guard::ALWAYS),
    ]);
    let mut w = scene(&vertex, &flat_color_pixel(), 16, target);
    // Location 2 is marked as a constant (bit 6 of its format register).
    w.set(0x1168, 1 << 6);
    let unused = [0.5, 0.5, 0.5, 0.5];
    vertices(
        &mut w,
        GPU_BASE + 0x30_0000,
        &[
            ([-1.0, -1.0, 0.5, 1.0], unused),
            ([3.0, -1.0, 0.5, 1.0], unused),
            ([-1.0, 3.0, 0.5, 1.0], unused),
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
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[0, 0, 0, 255]);
    }
}

/// Draw the red triangle with its right angle at clip (-1, -1), counter-clockwise in clip space
/// (y up), in a world whose viewport and culling the caller has set; the picture and its width.
fn red_triangle(w: &mut World, target: u64) -> (Vec<u8>, usize) {
    red_triangle_at(w, target, 0.5)
}

/// The same triangle with clip z `z` at all three vertices (w is 1).
fn red_triangle_at(w: &mut World, target: u64, z: f32) -> (Vec<u8>, usize) {
    let red = [1.0, 0.0, 0.0, 1.0];
    vertices(
        w,
        GPU_BASE + 0x30_0000,
        &[
            ([-1.0, -1.0, z, 1.0], red),
            ([1.0, -1.0, z, 1.0], red),
            ([-1.0, 1.0, z, 1.0], red),
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
    (rgba, width as usize)
}

fn covered(rgba: &[u8]) -> usize {
    rgba.as_chunks::<4>().0.iter().filter(|p| p[3] != 0).count()
}

#[test]
fn a_negative_viewport_scale_puts_clip_y_up_in_the_picture() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 64, target);
    // The viewport of a title that has the window origin at the upper left: scale y is negative.
    w.set(0xa04, (-32.0f32).to_bits());
    let (rgba, width) = red_triangle(&mut w, target);
    let at = |x: usize, y: usize| rgba[(y * width + x) * 4..][..4].to_vec();
    // Clip y of -1 is the bottom of the picture: the right angle is at the bottom left.
    assert_eq!(at(2, 61), [255, 0, 0, 255], "near the right angle");
    assert_eq!(at(10, 50), [255, 0, 0, 255]);
    assert_eq!(at(60, 2), [0, 0, 0, 0], "the corner away from it is empty");
    assert_eq!(at(50, 10), [0, 0, 0, 0]);
}

#[test]
fn the_front_face_is_judged_on_the_picture_whichever_way_the_viewport_turns_it() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    // The triangle is counter-clockwise in clip space. A negative scale y turns the picture
    // over (as the picture is read, with y down, it stays counter-clockwise: the title's
    // quads face front); a positive one makes it clockwise.
    for (scale_y, front_face, drawn) in [
        (-32.0f32, 0x901, true),
        (-32.0, 0x900, false),
        (32.0, 0x901, false),
        (32.0, 0x900, true),
    ] {
        let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 64, target);
        w.set(0xa04, scale_y.to_bits());
        // Cull the back faces.
        w.set(0x1918, 1);
        w.set(0x1920, 0x405);
        w.set(0x191c, front_face);
        let (rgba, _) = red_triangle(&mut w, target);
        let pixels = covered(&rgba);
        assert_eq!(
            pixels > 0,
            drawn,
            "scale y {scale_y}, front face {front_face:#x}: {pixels} pixels"
        );
    }
}

/// How many pixels the triangle at clip z `z` covers, in a world where the caller has set
/// the registers that decide the range of clip z.
fn covered_at(w: &mut World, target: u64, z: f32) -> usize {
    covered(&red_triangle_at(w, target, z).0)
}

#[test]
fn clip_z_runs_from_minus_w_to_w_unless_the_z_clip_register_says_otherwise() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    // A title sets the "minimum z zero, maximum z one" bit of the clip control while its vertex
    // programs put clip z between -w and w (docs/ASSUMPTIONS.md A-096): what lies in front of the
    // middle of that range is drawn.
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 16, target);
    w.set(0x193c, 1);
    assert!(covered_at(&mut w, target, -0.5) > 0, "z -0.5");
    assert!(covered_at(&mut w, target, 0.5) > 0, "z 0.5");
    assert_eq!(
        covered_at(&mut w, target, -1.5),
        0,
        "z -1.5 is in front of the volume"
    );
    assert_eq!(
        covered_at(&mut w, target, 1.5),
        0,
        "z 1.5 is behind the volume"
    );
    // With the fields that clamp depth set as well (all the title's value 0x181d says of it),
    // what is outside the volume is clamped to the range of depths, not clipped.
    w.set(0x193c, 0x181d);
    assert!(
        covered_at(&mut w, target, -0.5) > 0,
        "z -0.5 with the clamp"
    );
    assert!(covered_at(&mut w, target, -1.5) > 0, "z -1.5 clamped");
    assert!(covered_at(&mut w, target, 1.5) > 0, "z 1.5 clamped");
}

#[test]
fn clip_z_runs_from_zero_to_w_when_the_z_clip_register_says_so() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 16, target);
    w.set(0xd7c, 1);
    assert_eq!(
        covered_at(&mut w, target, -0.5),
        0,
        "z -0.5 is in front of the volume"
    );
    assert!(covered_at(&mut w, target, 0.5) > 0, "z 0.5");
    assert_eq!(
        covered_at(&mut w, target, 1.5),
        0,
        "z 1.5 is behind the volume"
    );
}

/// A pixel program that is red as far as the first word of the constant buffer in slot 1 (`c[0]`) says,
/// read through a register.
fn red_from_a_constant_buffer() -> Vec<u8> {
    pixel_program(
        &[],
        &[
            mov32i(8, 0),
            Enc::op(Op::Ldc)
                .rd(0)
                .ra(8)
                .field(20, 16, 0)
                .field(36, 5, 0)
                .field(44, 2, 0)
                .field(48, 3, 4)
                .build(),
            mov32i(1, 0),
            mov32i(2, 0),
            mov32i(3, 0x3F80_0000),
            exit(Guard::ALWAYS),
        ],
    )
}

#[test]
fn a_constant_buffer_bound_past_the_end_of_a_mapping_reads_what_is_mapped() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(
        &pass_through_vertex(),
        &red_from_a_constant_buffer(),
        16,
        target,
    );
    // The buffer is 32 bytes before the end of the memory the GPU address space maps, and bound with
    // the largest size, as a title binds a small buffer it sub-allocated: only 32 bytes are there.
    let cbuf = GPU_BASE + SIZE - 32;
    w.put(cbuf, &1.0f32.to_le_bytes());
    w.cbufs.bind(4, 1, (cbuf, 65536));
    let (rgba, _) = red_triangle(&mut w, target);
    for px in rgba.as_chunks::<4>().0.iter().filter(|p| p[3] != 0) {
        assert_eq!(px, &[255, 0, 0, 255]);
    }
    assert!(covered(&rgba) > 0);
    // Nothing mapped at all is still refused.
    let mut w = scene(
        &pass_through_vertex(),
        &red_from_a_constant_buffer(),
        16,
        target,
    );
    w.cbufs.bind(4, 1, (GPU_BASE + SIZE + 0x1000, 65536));
    let red = [1.0, 0.0, 0.0, 1.0];
    vertices(
        &mut w,
        GPU_BASE + 0x30_0000,
        &[
            ([-1.0, -1.0, 0.5, 1.0], red),
            ([3.0, -1.0, 0.5, 1.0], red),
            ([-1.0, 3.0, 0.5, 1.0], red),
        ],
    );
    let mut r = Renderer::new(RenderConfig::default());
    let e = r
        .draw(
            &w.scene(),
            &Draw {
                source: Source::Arrays { first: 0, count: 3 },
                topology: 4,
                instance: 0,
            },
        )
        .unwrap_err();
    assert!(e.contains("is not mapped"), "{e}");
}

#[test]
fn programs_in_a_mapping_that_runs_past_its_memory_are_read() {
    use nx_gpu::mem::GuestMemory;
    use nx_gpu::vm::{map_flags, MapRequest};
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 16, target);
    // The programs are in the last two pages of the memory, mapped as a 64 KiB window the way a
    // title maps a buffer of a size that is not a whole big page: the rest of the window is not
    // backed by anything, and a program is read a megabyte at a time (docs/ASSUMPTIONS.md A-069).
    let window = GPU_BASE + SIZE + 0x10_0000;
    w.space
        .map_buffer(&MapRequest {
            flags: map_flags::FIXED_OFFSET,
            kind: 0,
            handle: 2,
            handle_cpu: CPU_BASE + SIZE - 0x2000,
            handle_size: 0x2000,
            page_size: 0,
            buffer_offset: 0,
            mapping_size: 0x1_0000,
            offset: window,
        })
        .unwrap();
    let base = CPU_BASE + SIZE - 0x2000;
    assert!(w.mem.write(base + VERTEX_PROGRAM, &pass_through_vertex()));
    assert!(w.mem.write(base + PIXEL_PROGRAM, &flat_color_pixel()));
    w.set_address(0x1608, window);
    let (rgba, _) = red_triangle(&mut w, target);
    assert!(covered(&rgba) > 0, "nothing was drawn");
    for px in rgba.as_chunks::<4>().0.iter().filter(|p| p[3] != 0) {
        assert_eq!(px, &[255, 0, 0, 255]);
    }
}

#[test]
fn a_viewport_without_width_draws_nothing() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pass_through_vertex(), &flat_color_pixel(), 64, target);
    w.set(0xa00, 0.0f32.to_bits());
    let (rgba, _) = red_triangle(&mut w, target);
    assert_eq!(covered(&rgba), 0);
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
