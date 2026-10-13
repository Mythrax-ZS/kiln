//! Draws through the whole path: programs in GPU memory, state in registers, vertices in memory,
//! a picture out.

mod render_common;

use nx_gpu::render::{Backend, Draw, Source};
use nx_shader::asm::{assemble, exit, Enc};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_shader::sph::SPH_WORDS;
use nx_vulkan::{RenderConfig, Renderer};
use render_common::{vulkan, World, GPU_BASE};

/// A vertex program header followed by `code`.
fn vertex_program(code: &[u64]) -> Vec<u8> {
    let mut words = [0u32; SPH_WORDS];
    words[0] = 1 | 3 << 5 | 1 << 10;
    program(&words, code)
}

/// A pixel program header (the attributes in `flat` are read flat, target 0 is written) and code.
fn pixel_program(flat: &[(usize, usize)], code: &[u64]) -> Vec<u8> {
    let mut words = [0u32; SPH_WORDS];
    words[0] = 2 | 3 << 5 | 5 << 10;
    let set = |words: &mut [u32; SPH_WORDS], bit: usize, value: u32| {
        words[bit / 32] |= value << (bit % 32);
    };
    for &(attr, comp) in flat {
        set(&mut words, 192 + (attr * 4 + comp) * 2, 1);
    }
    for comp in 0..4 {
        set(&mut words, 576 + comp, 1);
    }
    program(&words, code)
}

fn program(words: &[u32; SPH_WORDS], code: &[u64]) -> Vec<u8> {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.extend_from_slice(&assemble(code));
    bytes
}

fn ald(rd: u8, addr: u32, words: u64) -> u64 {
    Enc::op(Op::Ald)
        .rd(rd)
        .ra(RZ)
        .field(20, 10, u64::from(addr))
        .field(47, 2, words - 1)
        .build()
}

fn ast(rd: u8, addr: u32, words: u64) -> u64 {
    Enc::op(Op::Ast)
        .rd(rd)
        .ra(RZ)
        .field(20, 10, u64::from(addr))
        .field(47, 2, words - 1)
        .build()
}

fn ipa(rd: u8, addr: u32) -> u64 {
    Enc::op(Op::Ipa)
        .rd(rd)
        .ra(RZ)
        .rb(RZ)
        .field(28, 10, u64::from(addr))
        .build()
}

/// A vertex program that passes the position (location 0) and the color (location 1) through.
fn pass_through_vertex() -> Vec<u8> {
    vertex_program(&[
        ald(0, 0x80, 4),
        ald(4, 0x90, 4),
        ast(0, 0x70, 4),
        ast(4, 0x80, 4),
        exit(Guard::ALWAYS),
    ])
}

/// A pixel program that writes the flat color of location 0.
fn flat_color_pixel() -> Vec<u8> {
    let flat: Vec<(usize, usize)> = (0..4).map(|c| (0, c)).collect();
    pixel_program(
        &flat,
        &[
            ipa(0, 0x80),
            ipa(1, 0x84),
            ipa(2, 0x88),
            ipa(3, 0x8C),
            exit(Guard::ALWAYS),
        ],
    )
}

const REGION: u64 = GPU_BASE + 0x10_0000;
const VERTEX_PROGRAM: u64 = 0;
const PIXEL_PROGRAM: u64 = 0x1000;

/// A world with the programs placed, a `size` x `size` target at `target` and the pipeline state
/// a title would set: viewport over the target, writes enabled, vertices from stream 0.
fn scene(vertex: &[u8], pixel: &[u8], size: u32, target: u64) -> World {
    let mut w = World::new();
    w.put(REGION + VERTEX_PROGRAM, vertex);
    w.put(REGION + PIXEL_PROGRAM, pixel);
    w.set_address(0x1608, REGION);
    // Slot 1 is the vertex program, slot 5 the pixel program.
    w.set(0x2044, VERTEX_PROGRAM as u32);
    w.set(0x2140, 1 | 5 << 4);
    w.set(0x2144, PIXEL_PROGRAM as u32);
    w.set(0x2150, 4);
    w.color_target(0, target, size, size, 0xD5);
    w.set(0x037c, 1);
    w.set(0x1a00, 0x1111);
    // The viewport transform over the whole target; depth 0..1.
    w.set(0x192c, 1);
    let half = size as f32 / 2.0;
    for (i, v) in [half, half, 0.5, half, half, 0.5].iter().enumerate() {
        w.set(0xa00 + 4 * i as u32, v.to_bits());
    }
    w.set(0xc08, 0.0f32.to_bits());
    w.set(0xc0c, 1.0f32.to_bits());
    // Attributes: 0 is the position and 1 the color, both four floats in stream 0.
    w.set(0x1160, 7 << 27 | 1 << 21);
    w.set(0x1164, 16 << 7 | 7 << 27 | 1 << 21);
    w
}

/// Four floats of position and four of color per vertex, as stream 0 at `va`.
fn vertices(w: &mut World, va: u64, list: &[([f32; 4], [f32; 4])]) {
    let bytes: Vec<u8> = list
        .iter()
        .flat_map(|(p, c)| p.iter().chain(c.iter()).flat_map(|f| f.to_le_bytes()))
        .collect();
    w.put(va, &bytes);
    w.set(0x1c00, 32 | 1 << 12);
    w.set_address(0x1c04, va);
    w.set_address(0x1f00, va + bytes.len() as u64 - 1);
}

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
