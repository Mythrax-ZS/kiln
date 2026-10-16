//! Textures through the whole path: a handle in a constant buffer, a header and a sampler in their
//! pools, texels in block-linear memory, and a picture of what a pixel program sampled.

mod render_common;

use nx_gpu::render::{Backend, Clear, Draw, Present, Source};
use nx_gpu::GuestMemory;
use nx_shader::asm::{exit, Enc};
use nx_shader::inst::Guard;
use nx_shader::ops::Op;
use nx_vulkan::blocklinear::{swizzle, Geometry};
use nx_vulkan::{RenderConfig, Renderer};
use render_common::*;

const HEADERS: u64 = GPU_BASE + 0x40_0000;
const SAMPLERS: u64 = GPU_BASE + 0x41_0000;
const TEXELS: u64 = GPU_BASE + 0x42_0000;
const HANDLES: u64 = GPU_BASE + 0x43_0000;

fn texs(rd: u8, ra: u8, rb: u8, r28: u8, index: u64, mask: u64, kind: u64) -> u64 {
    Enc::op(Op::Texs)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .field(28, 8, u64::from(r28))
        .field(36, 13, index)
        .field(50, 3, mask)
        .field(53, 4, kind)
        .bit(59, true)
        .build()
}

fn set_bits(words: &mut [u32; 8], lo: usize, len: usize, value: u32) {
    for i in 0..len {
        let bit = lo + i;
        words[bit / 32] &= !(1 << (bit % 32));
        words[bit / 32] |= (value >> i & 1) << (bit % 32);
    }
}

/// A block-linear 2D texture header: the format code (and the same data type for all
/// components), the identity swizzle, the size and the address.
fn header(components: u32, data_type: u32, width: u32, height: u32, address: u64) -> [u32; 8] {
    let mut w = [0u32; 8];
    set_bits(&mut w, 0, 7, components);
    for (i, lo) in [7, 10, 13, 16].into_iter().enumerate() {
        let _ = i;
        set_bits(&mut w, lo, 3, data_type);
    }
    for (i, lo) in [19, 22, 25, 28].into_iter().enumerate() {
        set_bits(&mut w, lo, 3, 2 + i as u32);
    }
    set_bits(&mut w, 41, 23, (address >> 9) as u32);
    set_bits(&mut w, 64, 16, (address >> 32) as u32);
    set_bits(&mut w, 85, 3, 3);
    set_bits(&mut w, 128, 16, width - 1);
    set_bits(&mut w, 151, 4, 1);
    set_bits(&mut w, 160, 16, height - 1);
    w
}

/// A sampler header with nearest filtering and clamping.
fn nearest_sampler() -> [u32; 8] {
    let mut w = [0u32; 8];
    w[0] = 2 | 2 << 3 | 2 << 6;
    w[1] = 1 | 1 << 4 | 1 << 6;
    w[2] = 0xFFF << 12;
    w
}

fn put_words(w: &World, va: u64, words: &[u32; 8]) {
    let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.put(va, &bytes);
}

/// Position and texture coordinates per vertex as stream 0 (location 0 and 1).
fn uv_vertices(w: &mut World, va: u64, list: &[([f32; 4], [f32; 2])]) {
    let bytes: Vec<u8> = list
        .iter()
        .flat_map(|(p, t)| p.iter().chain(t.iter()).flat_map(|f| f.to_le_bytes()))
        .collect();
    w.put(va, &bytes);
    w.set(0x1c00, 24 | 1 << 12);
    w.set_address(0x1c04, va);
    w.set_address(0x1f00, va + bytes.len() as u64 - 1);
    // Location 1 is two floats at offset 16.
    w.set(0x1164, 16 << 7 | 7 << 27 | 4 << 21);
}

fn uv_vertex() -> Vec<u8> {
    vertex_program(&[
        ald(0, 0x80, 4),
        ald(4, 0x90, 2),
        ast(0, 0x70, 4),
        ast(4, 0x80, 2),
        exit(Guard::ALWAYS),
    ])
}

/// Samples texture 0 at the (screen linear) coordinates of the varying and writes the texel.
fn texture_pixel() -> Vec<u8> {
    pixel_program_with(
        &[(0, 0), (0, 1)],
        3,
        &[
            ipa(4, 0x80),
            ipa(5, 0x84),
            texs(0, 4, 5, 2, 0, 4, 2),
            exit(Guard::ALWAYS),
        ],
    )
}

/// The world for a textured full-target triangle: texture 0 is the entry 1 of the header pool
/// and the sampler the entry 2 of the sampler pool.
fn textured(size: u32, target: u64, tic: [u32; 8], tsc: [u32; 8]) -> World {
    let mut w = scene(&uv_vertex(), &texture_pixel(), size, target);
    put_words(&w, HEADERS + 32, &tic);
    put_words(&w, SAMPLERS + 64, &tsc);
    w.put(HANDLES, &(1u32 | 2 << 20).to_le_bytes());
    w.set_address(0x1574, HEADERS);
    w.set(0x157c, 7);
    w.set_address(0x155c, SAMPLERS);
    w.set(0x1564, 7);
    w.set(0x2608, 3);
    w.cbufs.bind(4, 1 | 3 << 4, (HANDLES, 64));
    uv_vertices(
        &mut w,
        GPU_BASE + 0x44_0000,
        &[
            ([-1.0, -1.0, 0.5, 1.0], [0.0, 0.0]),
            ([3.0, -1.0, 0.5, 1.0], [2.0, 0.0]),
            ([-1.0, 3.0, 0.5, 1.0], [0.0, 2.0]),
        ],
    );
    w
}

const DRAW: Draw = Draw {
    source: Source::Arrays { first: 0, count: 3 },
    topology: 4,
    instance: 0,
};

/// A 4x4 texture of A8B8G8R8 texels (bytes R, G, B, A) in block-linear memory at `TEXELS`.
fn put_rgba_texture(w: &World, texel: impl Fn(usize, usize) -> [u8; 4]) {
    let g = Geometry {
        width: 4,
        height: 4,
        depth: 1,
        bytes_per_element: 4,
        block_height_log2: 0,
        block_depth_log2: 0,
    };
    let mut linear = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            linear.extend_from_slice(&texel(x, y));
        }
    }
    let mut tiled = vec![0u8; g.tiled_size() as usize];
    swizzle(&g, &linear, &mut tiled);
    w.put(TEXELS, &tiled);
}

#[test]
fn a_block_linear_texture_is_sampled_texel_by_texel() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let w = textured(32, target, header(0x08, 2, 4, 4, TEXELS), nearest_sampler());
    put_rgba_texture(&w, |x, y| [(x * 60) as u8, (y * 60) as u8, 128, 255]);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for py in 0..32usize {
        for px in 0..32usize {
            let got = &rgba[(py * width as usize + px) * 4..][..4];
            let want = [(px / 8 * 60) as u8, (py / 8 * 60) as u8, 128, 255];
            assert_eq!(got, want, "pixel ({px}, {py})");
        }
    }
}

#[test]
fn the_component_swizzle_of_the_header_reorders_the_channels() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut tic = header(0x08, 2, 4, 4, TEXELS);
    // x comes from blue, z from red, y from one, w from zero.
    set_bits(&mut tic, 19, 3, 4);
    set_bits(&mut tic, 22, 3, 7);
    set_bits(&mut tic, 25, 3, 2);
    set_bits(&mut tic, 28, 3, 0);
    let w = textured(16, target, tic, nearest_sampler());
    put_rgba_texture(&w, |_, _| [10, 20, 30, 255]);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    assert_eq!(&rgba[..4], [30, 255, 10, 0]);
}

#[test]
fn a_compressed_texture_is_uploaded_whole_blocks_at_a_time() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let w = textured(16, target, header(0x24, 2, 4, 4, TEXELS), nearest_sampler());
    // One BC1 block: both end colors red (565 0xF800) and every index zero, so solid red.
    let block = [0x00, 0xF8, 0x00, 0xF8, 0, 0, 0, 0];
    let g = Geometry {
        width: 1,
        height: 1,
        depth: 1,
        bytes_per_element: 8,
        block_height_log2: 0,
        block_depth_log2: 0,
    };
    let mut tiled = vec![0u8; g.tiled_size() as usize];
    swizzle(&g, &block, &mut tiled);
    w.put(TEXELS, &tiled);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    assert_eq!(&rgba[..4], [255, 0, 0, 255]);
    assert_eq!(&rgba[rgba.len() - 4..], [255, 0, 0, 255]);
}

#[test]
fn a_texture_is_compared_with_memory_once_a_frame() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let w = textured(8, target, header(0x08, 2, 4, 4, TEXELS), nearest_sampler());
    put_rgba_texture(&w, |_, _| [200, 0, 0, 255]);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(&w.scene(), &DRAW).unwrap();
    // The contents change inside the frame: the cached copy is still what a draw sees.
    put_rgba_texture(&w, |_, _| [0, 200, 0, 255]);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    assert_eq!(&rgba[..4], [200, 0, 0, 255]);
    // A frame later, the new contents are used.
    r.present(&Present {
        address: w.cpu(target),
        width: 8,
        height: 8,
        pitch: 0,
        format: 1,
        layout: 3,
        block_height_log2: 4,
        crop: [0; 4],
        frame: 1,
    })
    .unwrap();
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    assert_eq!(&rgba[..4], [0, 200, 0, 255]);
}

#[test]
fn a_handle_that_names_no_header_stops_the_draw_with_what_it_named() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let w = textured(8, target, header(0x08, 2, 4, 4, TEXELS), nearest_sampler());
    put_rgba_texture(&w, |_, _| [1, 2, 3, 4]);
    w.put(HANDLES, &(100u32 | 2 << 20).to_le_bytes());
    let mut r = Renderer::new(RenderConfig::default());
    let err = r.draw(&w.scene(), &DRAW).unwrap_err();
    assert!(err.contains("header 100 of 8"), "{err}");
}

fn clear_to(r: &mut Renderer, w: &mut World, rgba: [f32; 4], rect: Option<(u32, u32, u32, u32)>) {
    for (i, v) in rgba.iter().enumerate() {
        w.set(0xd80 + 4 * i as u32, v.to_bits());
    }
    match rect {
        Some((x0, x1, y0, y1)) => {
            w.set(0x10f8, 1 << 4);
            w.set(0xd6c, x0 | x1 << 16);
            w.set(0xd70, y0 | y1 << 16);
        }
        None => w.set(0x10f8, 0),
    }
    let all = Clear {
        depth: false,
        stencil: false,
        red: true,
        green: true,
        blue: true,
        alpha: true,
        target: 0,
        layer: 0,
    };
    r.clear(&w.scene(), &all).unwrap();
}

#[test]
fn a_render_target_is_sampled_from_the_target_and_not_from_memory() {
    if vulkan().is_none() {
        return;
    }
    let source = GPU_BASE + 0x50_0000;
    let target = GPU_BASE + 0x20_0000;
    // The memory of `source` stays zero: only the renderer's target holds what was drawn there.
    let mut w = textured(
        16,
        target,
        header(0x08, 2, 16, 16, source),
        nearest_sampler(),
    );
    let mut r = Renderer::new(RenderConfig::default());
    w.color_target(0, source, 16, 16, 0xD5);
    clear_to(&mut r, &mut w, [0.0, 1.0, 0.0, 1.0], None);
    clear_to(&mut r, &mut w, [1.0, 0.0, 0.0, 1.0], Some((0, 8, 0, 16)));
    w.color_target(0, target, 16, 16, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for y in 0..16usize {
        for x in 0..16usize {
            let want = if x < 8 {
                [255, 0, 0, 255]
            } else {
                [0, 255, 0, 255]
            };
            assert_eq!(
                &rgba[(y * width as usize + x) * 4..][..4],
                want,
                "({x}, {y})"
            );
        }
    }
    // The source is drawn to again after having been read, and read again.
    w.color_target(0, source, 16, 16, 0xD5);
    clear_to(&mut r, &mut w, [0.0, 0.0, 1.0, 1.0], None);
    w.color_target(0, target, 16, 16, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[0, 0, 255, 255]);
    }
}

/// More submissions than a target that nothing draws to is kept for.
const LONG: usize = 350;

#[test]
fn a_target_drawn_once_and_read_long_after_keeps_its_pixels() {
    if vulkan().is_none() {
        return;
    }
    let source = GPU_BASE + 0x50_0000;
    let other = GPU_BASE + 0x60_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut w = textured(
        16,
        target,
        header(0x08, 2, 16, 16, source),
        nearest_sampler(),
    );
    let mut r = Renderer::new(RenderConfig::default());
    // Drawn to once, as a title draws the text of its interface, and read much later.
    w.color_target(0, source, 16, 16, 0xD5);
    clear_to(&mut r, &mut w, [1.0, 0.0, 0.0, 1.0], None);
    // Other work goes on for longer than a target that nothing draws to is kept (A-118).
    w.color_target(0, other, 16, 16, 0xD5);
    for _ in 0..LONG {
        clear_to(&mut r, &mut w, [0.0, 0.0, 1.0, 1.0], None);
        r.finish().unwrap();
    }
    w.color_target(0, target, 16, 16, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(
            px,
            &[255, 0, 0, 255],
            "what was drawn is what the memory holds"
        );
    }
}

#[test]
fn a_target_whose_memory_was_used_for_something_else_does_not_overwrite_it() {
    if vulkan().is_none() {
        return;
    }
    let source = GPU_BASE + 0x50_0000;
    let other = GPU_BASE + 0x60_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut w = textured(
        16,
        target,
        header(0x08, 2, 16, 16, source),
        nearest_sampler(),
    );
    let mut r = Renderer::new(RenderConfig::default());
    w.color_target(0, source, 16, 16, 0xD5);
    clear_to(&mut r, &mut w, [1.0, 0.0, 0.0, 1.0], None);
    // The title frees the memory and puts something else in it: a page of a pattern.
    let pattern: Vec<u8> = (0..0x1000u32).map(|i| (i % 251) as u8 | 1).collect();
    w.put(source, &pattern);
    w.color_target(0, other, 16, 16, 0xD5);
    for _ in 0..LONG {
        clear_to(&mut r, &mut w, [0.0, 0.0, 1.0, 1.0], None);
        r.finish().unwrap();
    }
    w.color_target(0, target, 16, 16, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let mut now = vec![0u8; pattern.len()];
    assert!(w.mem.read(w.cpu(source), &mut now));
    assert_eq!(now, pattern, "the target let go was not written over it");
}

#[test]
fn the_header_swizzle_applies_to_a_render_target_read_as_a_texture() {
    if vulkan().is_none() {
        return;
    }
    let source = GPU_BASE + 0x50_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut tic = header(0x08, 2, 8, 8, source);
    // x from blue, y from red, z from one, w from zero.
    set_bits(&mut tic, 19, 3, 4);
    set_bits(&mut tic, 22, 3, 2);
    set_bits(&mut tic, 25, 3, 7);
    set_bits(&mut tic, 28, 3, 0);
    let mut w = textured(8, target, tic, nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    w.color_target(0, source, 8, 8, 0xD5);
    clear_to(&mut r, &mut w, [1.0, 0.0, 0.0, 1.0], None);
    clear_to(&mut r, &mut w, [0.0, 0.0, 1.0, 1.0], Some((4, 8, 0, 8)));
    w.color_target(0, target, 8, 8, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    assert_eq!(&rgba[..4], [0, 255, 255, 0], "red source: x = b, y = r");
    let right = &rgba[(7 * width as usize + 7) * 4..][..4];
    assert_eq!(right, [255, 0, 255, 0], "blue source");
}

#[test]
fn a_target_is_not_read_while_the_same_draw_writes_it() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let w = textured(8, target, header(0x08, 2, 8, 8, target), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    // The first draw makes the target; the second would find it, and reads the memory (zeros).
    r.draw(&w.scene(), &DRAW).unwrap();
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[0, 0, 0, 0]);
    }
}

/// Bind a depth target of `size` x `size` at `va` with the zeta format `format`.
fn depth_target(w: &mut World, va: u64, size: u32, format: u32) {
    w.set_address(0xfe0, va);
    w.set(0xfe8, format);
    w.set(0xfec, 4 << 4);
    w.set(0x1228, size);
    w.set(0x122c, size);
    w.set(0x1230, 1 << 16);
    w.set(0x1538, 1);
}

/// Depth test and write as given, passing every fragment.
fn depth_state(w: &mut World, test: bool, write: bool) {
    w.set(0x12cc, u32::from(test));
    w.set(0x12e8, u32::from(write));
    w.set(0x130c, 0x207);
}

/// Clear the depth target to `value`.
fn clear_depth(r: &mut Renderer, w: &mut World, value: f32) {
    w.set(0xd90, value.to_bits());
    w.set(0x10f8, 0);
    let depth_only = Clear {
        depth: true,
        stencil: false,
        red: false,
        green: false,
        blue: false,
        alpha: false,
        target: 0,
        layer: 0,
    };
    r.clear(&w.scene(), &depth_only).unwrap();
}

/// The red channel of the picture at `target` where the draw sampled a depth: every pixel the
/// same, within a step of the 8 bits.
fn sampled_depth(r: &mut Renderer, w: &World, target: u64) -> u8 {
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    let first = rgba[0];
    for px in rgba.as_chunks::<4>().0 {
        assert!(px[0].abs_diff(first) <= 1, "{px:?} against {first}");
        assert_eq!(&px[1..], &[0, 0, 255], "{px:?}");
    }
    first
}

#[test]
fn a_depth_target_is_sampled_as_a_texture_by_a_later_pass() {
    if vulkan().is_none() {
        return;
    }
    let depth = GPU_BASE + 0x60_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut w = textured(8, target, header(0x2F, 7, 8, 8, depth), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    depth_target(&mut w, depth, 8, 0x0A);
    clear_depth(&mut r, &mut w, 0.25);
    // The next pass has no depth target of its own and reads the one that was cleared.
    w.set(0x1538, 0);
    r.draw(&w.scene(), &DRAW).unwrap();
    assert!(sampled_depth(&mut r, &w, target).abs_diff(64) <= 1);
}

#[test]
fn the_depth_of_a_combined_format_is_the_second_component_of_its_header() {
    if vulkan().is_none() {
        return;
    }
    let depth = GPU_BASE + 0x60_0000;
    let target = GPU_BASE + 0x20_0000;
    // S8Z24: the first component is an unsigned integer (the stencil), the others normalized; the
    // shader reads the second component into x, y and z.
    let mut tic = header(0x29, 2, 8, 8, depth);
    set_bits(&mut tic, 7, 3, 4);
    for lo in [19, 22, 25] {
        set_bits(&mut tic, lo, 3, 3);
    }
    set_bits(&mut tic, 28, 3, 7);
    let mut w = textured(8, target, tic, nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    depth_target(&mut w, depth, 8, 0x14);
    clear_depth(&mut r, &mut w, 0.25);
    w.set(0x1538, 0);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for px in rgba.as_chunks::<4>().0 {
        assert!(px[0].abs_diff(64) <= 1, "{px:?}");
        assert_eq!(px[0], px[1], "{px:?}");
        assert_eq!(px[0], px[2], "{px:?}");
        assert_eq!(px[3], 255, "{px:?}");
    }
}

#[test]
fn a_draw_that_writes_the_depth_it_samples_samples_a_copy_of_it() {
    if vulkan().is_none() {
        return;
    }
    let depth = GPU_BASE + 0x60_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut w = textured(8, target, header(0x2F, 7, 8, 8, depth), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    depth_target(&mut w, depth, 8, 0x0A);
    depth_state(&mut w, true, true);
    clear_depth(&mut r, &mut w, 0.25);
    // The draw writes the depth of its triangle and reads the target it writes: it sees what was
    // there before it.
    r.draw(&w.scene(), &DRAW).unwrap();
    assert!(
        sampled_depth(&mut r, &w, target).abs_diff(64) <= 1,
        "the copy"
    );
    // What it wrote is in the target afterwards: the next pass reads that.
    w.set(0x1538, 0);
    r.draw(&w.scene(), &DRAW).unwrap();
    let written = sampled_depth(&mut r, &w, target);
    assert!(
        written.abs_diff(64) > 8,
        "the depth the triangle wrote: {written}"
    );
}

#[test]
fn a_draw_that_only_tests_the_depth_it_samples_reads_it_where_it_is() {
    if vulkan().is_none() {
        return;
    }
    let depth = GPU_BASE + 0x60_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut w = textured(8, target, header(0x2F, 7, 8, 8, depth), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    depth_target(&mut w, depth, 8, 0x0A);
    depth_state(&mut w, true, false);
    clear_depth(&mut r, &mut w, 0.25);
    r.draw(&w.scene(), &DRAW).unwrap();
    assert!(sampled_depth(&mut r, &w, target).abs_diff(64) <= 1);
    // Nothing was written to the depth, and a draw that writes it afterwards still works.
    depth_state(&mut w, true, true);
    r.draw(&w.scene(), &DRAW).unwrap();
    w.set(0x1538, 0);
    r.draw(&w.scene(), &DRAW).unwrap();
    assert!(sampled_depth(&mut r, &w, target).abs_diff(64) > 8);
}

/// A void-extent ASTC block of one color (16-bit components) with no extent.
fn astc_solid(rgba: [u16; 4]) -> [u8; 16] {
    let mut v: u128 = 0x1FC | 3 << 10;
    for lo in [12, 25, 38, 51] {
        v |= 0x1FFF << lo;
    }
    for (i, c) in rgba.iter().enumerate() {
        v |= u128::from(*c) << (64 + 16 * i);
    }
    v.to_le_bytes()
}

#[test]
fn an_astc_texture_is_decoded_on_the_host_and_sampled() {
    if vulkan().is_none() {
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    // 8x8 texels in 4x4 blocks (format code 0x40): two blocks across and down.
    let w = textured(16, target, header(0x40, 2, 8, 8, TEXELS), nearest_sampler());
    let colors = [
        [0xFFFFu16, 0, 0, 0xFFFF],
        [0, 0xFFFF, 0, 0xFFFF],
        [0, 0, 0xFFFF, 0xFFFF],
        [0x8000, 0x8000, 0, 0x4000],
    ];
    let linear: Vec<u8> = colors.iter().flat_map(|c| astc_solid(*c)).collect();
    let g = Geometry {
        width: 2,
        height: 2,
        depth: 1,
        bytes_per_element: 16,
        block_height_log2: 0,
        block_depth_log2: 0,
    };
    let mut tiled = vec![0u8; g.tiled_size() as usize];
    swizzle(&g, &linear, &mut tiled);
    w.put(TEXELS, &tiled);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for py in 0..16usize {
        for px in 0..16usize {
            let got = &rgba[(py * width as usize + px) * 4..][..4];
            let want = match (px / 8, py / 8) {
                (0, 0) => [255, 0, 0, 255],
                (1, 0) => [0, 255, 0, 255],
                (0, 1) => [0, 0, 255, 255],
                _ => [128, 128, 0, 64],
            };
            assert_eq!(got, want, "pixel ({px}, {py})");
        }
    }
}

#[test]
fn a_render_target_that_a_texture_reads_through_memory_is_written_to_it_first() {
    if vulkan().is_none() {
        return;
    }
    let source = GPU_BASE + 0x50_0000;
    let target = GPU_BASE + 0x20_0000;
    // The texture is 8 by 8 texels over a 16 by 16 target: not the target's picture (the sizes
    // differ), so it is made from memory, which must hold what was drawn.
    let mut w = textured(8, target, header(0x08, 2, 8, 8, source), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    w.color_target(0, source, 16, 16, 0xD5);
    clear_to(&mut r, &mut w, [0.0, 1.0, 0.0, 1.0], None);
    clear_to(&mut r, &mut w, [1.0, 0.0, 0.0, 1.0], Some((0, 8, 0, 16)));
    w.color_target(0, target, 8, 8, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for y in 0..8usize {
        for x in 0..8usize {
            assert_eq!(
                &rgba[(y * width as usize + x) * 4..][..4],
                [255, 0, 0, 255],
                "({x}, {y})"
            );
        }
    }
    // Drawn to again, it is written again before the next read.
    w.color_target(0, source, 16, 16, 0xD5);
    clear_to(&mut r, &mut w, [0.0, 0.0, 1.0, 1.0], None);
    w.color_target(0, target, 8, 8, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[0, 0, 255, 255]);
    }
}

#[test]
fn a_render_target_nothing_draws_to_for_long_is_not_written_over_reused_memory() {
    if vulkan().is_none() {
        return;
    }
    let source = GPU_BASE + 0x50_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut w = textured(8, target, header(0x08, 2, 8, 8, source), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    // A 16 by 16 target at the texture's address is drawn to, and nothing reads it...
    w.color_target(0, source, 16, 16, 0xD5);
    clear_to(&mut r, &mut w, [1.0, 0.0, 0.0, 1.0], None);
    assert!(r.picture_at(w.cpu(source)).unwrap().is_some());
    // ... for a long time, while the title frees that memory and puts something else there.
    for _ in 0..400 {
        r.finish().unwrap();
    }
    let reused: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    w.put(source, &reused);
    // A texture made of that memory is sampled. The old target is no longer one of the targets,
    // so it is not written over the memory (its pixels would overwrite what lives there now).
    w.color_target(0, target, 8, 8, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let mut memory = vec![0u8; reused.len()];
    assert!(w.mem.read(w.cpu(source), &mut memory));
    assert!(
        memory == reused,
        "the memory was overwritten by a stale target"
    );
    assert!(r.picture_at(w.cpu(source)).unwrap().is_none());
}

#[test]
fn a_render_target_that_is_a_part_of_a_texture_is_copied_into_it_on_the_device() {
    if vulkan().is_none() {
        return;
    }
    let source = GPU_BASE + 0x50_0000;
    let target = GPU_BASE + 0x20_0000;
    // The texture is 8 by 8 texels. The target at its address is 16 by 8: not the texture's
    // picture (the sizes differ), but its rows lie where the texture's do (one 64-byte column of
    // a GOB each, one GOB high), so its left half is the texture, copied on the device.
    let mut w = textured(8, target, header(0x08, 2, 8, 8, source), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    w.color_target_blocks(0, source, 16, 8, 0xD5, 0);
    clear_to(&mut r, &mut w, [0.0, 1.0, 0.0, 1.0], None);
    clear_to(&mut r, &mut w, [1.0, 0.0, 0.0, 1.0], Some((0, 8, 0, 8)));
    w.color_target(0, target, 8, 8, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    let all = |rgba: &[u8], px: [u8; 4], what: &str| {
        for y in 0..8usize {
            for x in 0..8usize {
                assert_eq!(
                    &rgba[(y * width as usize + x) * 4..][..4],
                    px,
                    "{what} ({x}, {y})"
                );
            }
        }
    };
    all(&rgba, [255, 0, 0, 255], "the left half of the target");
    // Nothing was written to the memory: the pixels stayed on the device.
    let mut memory = vec![0u8; 512];
    assert!(w.mem.read(w.cpu(source), &mut memory));
    assert!(memory.iter().all(|b| *b == 0), "no write-back");
    // Drawn to again, it is copied again; and a draw with the target unchanged does not copy.
    w.color_target_blocks(0, source, 16, 8, 0xD5, 0);
    clear_to(&mut r, &mut w, [0.0, 0.0, 1.0, 1.0], None);
    w.color_target(0, target, 8, 8, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    all(
        &rgba,
        [0, 0, 255, 255],
        "after the target was drawn to again",
    );
    assert!(w.mem.read(w.cpu(source), &mut memory));
    assert!(memory.iter().all(|b| *b == 0), "still no write-back");
}

/// A texture of 8 by 8 texels read from memory that two color targets cover: a green one of 64 by
/// 64 whose memory runs from before the texture to after it, and a red one at the texture's
/// address, `red_height` rows of 16 pixels (the texture's rows lie in its left half, so it is the
/// texture's rows from the top down). `red_last` says which of them is drawn to last. Returns what
/// the memory of the texture holds afterwards and the 8 by 8 picture of a draw that samples it.
fn two_targets_over_a_texture(red_last: bool, red_height: u32) -> (Vec<u8>, Vec<[u8; 4]>) {
    let source = GPU_BASE + 0x50_0000;
    let target = GPU_BASE + 0x20_0000;
    let under = source - 0x2000;
    let mut w = textured(8, target, header(0x08, 2, 8, 8, source), nearest_sampler());
    let mut r = Renderer::new(RenderConfig::default());
    let green = |r: &mut Renderer, w: &mut World| {
        w.color_target_blocks(0, under, 64, 64, 0xD5, 0);
        clear_to(r, w, [0.0, 1.0, 0.0, 1.0], None);
    };
    let red = |r: &mut Renderer, w: &mut World| {
        w.color_target_blocks(0, source, 16, red_height, 0xD5, 0);
        clear_to(r, w, [1.0, 0.0, 0.0, 1.0], None);
    };
    if red_last {
        green(&mut r, &mut w);
        red(&mut r, &mut w);
    } else {
        red(&mut r, &mut w);
        green(&mut r, &mut w);
    }
    w.color_target(0, target, 8, 8, 0xD5);
    r.draw(&w.scene(), &DRAW).unwrap();
    let mut memory = vec![0u8; 512];
    assert!(w.mem.read(w.cpu(source), &mut memory));
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    let mut pixels = Vec::new();
    for y in 0..8usize {
        for x in 0..8usize {
            let at = (y * width as usize + x) * 4;
            pixels.push([rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]]);
        }
    }
    (memory, pixels)
}

#[test]
fn a_target_drawn_before_the_targets_that_make_a_texture_is_not_written_to_its_memory() {
    if vulkan().is_none() {
        return;
    }
    let (memory, pixels) = two_targets_over_a_texture(true, 8);
    assert!(pixels.iter().all(|p| *p == [255, 0, 0, 255]), "{pixels:?}");
    // The green target lies under the red one that is the whole texture: it cost no write-back.
    assert!(
        memory.iter().all(|b| *b == 0),
        "the older target was written"
    );
}

#[test]
fn a_target_that_a_texture_is_not_wholly_made_of_is_still_written_under_it() {
    if vulkan().is_none() {
        return;
    }
    // The red target is the texture's top half only: the green one shows below it.
    let (memory, pixels) = two_targets_over_a_texture(true, 4);
    assert!(memory.iter().any(|b| *b != 0), "no write-back");
    for (i, p) in pixels.iter().enumerate() {
        let want = if i / 8 < 4 {
            [255, 0, 0, 255]
        } else {
            [0, 255, 0, 255]
        };
        assert_eq!(*p, want, "pixel {i}");
    }
    // A target drawn to after the one that is the whole texture is not under it.
    let (memory, _) = two_targets_over_a_texture(false, 8);
    assert!(
        memory.iter().any(|b| *b != 0),
        "the later target was not written"
    );
}

#[test]
fn work_sent_to_the_device_in_batches_sees_what_the_batches_before_it_made() {
    if vulkan().is_none() {
        return;
    }
    // A target is cleared in one batch and read as a texture in the next, many times over, with
    // the host never waiting for the device in between: every picture must be the colour the
    // batch before it made.
    let source = GPU_BASE + 0x50_0000;
    let target = GPU_BASE + 0x20_0000;
    let mut w = textured(
        16,
        target,
        header(0x08, 2, 16, 16, source),
        nearest_sampler(),
    );
    let mut r = Renderer::new(RenderConfig::default());
    let colors = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    for round in 0..40usize {
        let color = colors[round % colors.len()];
        w.color_target(0, source, 16, 16, 0xD5);
        clear_to(&mut r, &mut w, color, None);
        r.finish().unwrap();
        w.color_target(0, target, 16, 16, 0xD5);
        r.draw(&w.scene(), &DRAW).unwrap();
        r.finish().unwrap();
        let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
        let want = color.map(|c| (c * 255.0) as u8);
        for px in rgba.as_chunks::<4>().0 {
            assert_eq!(px, &want, "round {round}");
        }
    }
}
