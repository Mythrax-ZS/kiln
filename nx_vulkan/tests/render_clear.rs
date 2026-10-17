//! The renderer's first steps: a color target is made on a clear, found again by its address, and
//! read back; a displayed buffer is written as a picture.

mod render_common;

use nx_gpu::render::{Backend, Clear, Present};
use nx_vulkan::{RenderConfig, Renderer};
use render_common::{vulkan, World, GPU_BASE};

fn clear_all() -> Clear {
    Clear {
        depth: false,
        stencil: false,
        red: true,
        green: true,
        blue: true,
        alpha: true,
        target: 0,
        layer: 0,
    }
}

#[test]
fn a_clear_fills_the_target_with_the_clear_color() {
    if vulkan().is_none() {
        return;
    }
    let mut w = World::new();
    // 64x32 A8B8G8R8 (bytes R, G, B, A).
    let target = GPU_BASE + 0x10000;
    w.color_target(0, target, 64, 32, 0xD5);
    for (i, v) in [1.0f32, 0.2, 0.0, 1.0].iter().enumerate() {
        w.set(0xd80 + 4 * i as u32, v.to_bits());
    }
    let mut r = Renderer::new(RenderConfig::default());
    r.clear(&w.scene(), &clear_all()).unwrap();
    r.finish().unwrap();
    let (rgba, width, height) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    assert_eq!((width, height), (64, 32));
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[255, 51, 0, 255]);
    }
}

#[test]
fn a_clear_rectangle_leaves_the_rest_alone_and_bgra_targets_swap_channels() {
    if vulkan().is_none() {
        return;
    }
    let mut w = World::new();
    let target = GPU_BASE + 0x20000;
    // 32x32 A8R8G8B8 (bytes B, G, R, A in memory; the picture is still RGBA).
    w.color_target(0, target, 32, 32, 0xCF);
    for (i, v) in [0.0f32, 0.0, 0.0, 1.0].iter().enumerate() {
        w.set(0xd80 + 4 * i as u32, v.to_bits());
    }
    let mut r = Renderer::new(RenderConfig::default());
    r.clear(&w.scene(), &clear_all()).unwrap();
    // Then red in the rectangle x 8..24, y 4..12.
    for (i, v) in [1.0f32, 0.0, 0.0, 1.0].iter().enumerate() {
        w.set(0xd80 + 4 * i as u32, v.to_bits());
    }
    w.set(0x10f8, 1 << 4);
    w.set(0xd6c, 8 | 24 << 16);
    w.set(0xd70, 4 | 12 << 16);
    r.clear(&w.scene(), &clear_all()).unwrap();
    let (rgba, width, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for y in 0..32usize {
        for x in 0..32usize {
            let px = &rgba[(y * width as usize + x) * 4..][..4];
            let inside = (8..24).contains(&x) && (4..12).contains(&y);
            let expected: [u8; 4] = if inside {
                [255, 0, 0, 255]
            } else {
                [0, 0, 0, 255]
            };
            assert_eq!(px, expected, "pixel ({x}, {y})");
        }
    }
}

#[test]
fn a_displayed_buffer_is_dumped_as_a_picture_when_asked() {
    if vulkan().is_none() {
        return;
    }
    let mut w = World::new();
    let target = GPU_BASE + 0x30000;
    w.color_target(0, target, 16, 16, 0xD5);
    for (i, v) in [0.0f32, 1.0, 0.0, 1.0].iter().enumerate() {
        w.set(0xd80 + 4 * i as u32, v.to_bits());
    }
    let dir = std::env::temp_dir().join(format!("nx_vulkan_dump_{}", std::process::id()));
    let mut r = Renderer::new(RenderConfig {
        dump_dir: Some(dir.clone()),
        dump_every: 1,
        dump_limit: None,
        ..RenderConfig::default()
    });
    r.clear(&w.scene(), &clear_all()).unwrap();
    r.finish().unwrap();
    r.present(&Present {
        address: w.cpu(target),
        width: 16,
        height: 16,
        pitch: 0,
        format: 1,
        layout: 3,
        block_height_log2: 4,
        crop: [0; 4],
        frame: 7,
        due: None,
    })
    .unwrap();
    let png = std::fs::read(dir.join("frame_00007.png")).expect("the picture was written");
    assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_clear_with_no_target_or_a_partial_channel_mask_is_refused_not_ignored() {
    if vulkan().is_none() {
        return;
    }
    let w = World::new();
    let mut r = Renderer::new(RenderConfig::default());
    let err = r.clear(&w.scene(), &clear_all()).unwrap_err();
    assert!(err.contains("no render target"), "{err}");
    let mut w = World::new();
    w.color_target(0, GPU_BASE, 8, 8, 0xD5);
    let partial = Clear {
        blue: false,
        ..clear_all()
    };
    let err = r.clear(&w.scene(), &partial).unwrap_err();
    assert!(err.contains("only some color channels"), "{err}");
}
