//! Textures through the whole path: a handle in a constant buffer, a header and a sampler in their
//! pools, texels in block-linear memory, and a picture of what a pixel program sampled.

mod render_common;

use nx_gpu::render::{Backend, Draw, Present, Source};
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
