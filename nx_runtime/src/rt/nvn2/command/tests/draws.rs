use super::*;

fn word(fields: &[(u32, u32, u64)]) -> u128 {
    let mut value = (7u128 << 12) | (7 << 87) | (7 << 110) | (7 << 113);
    for &(at, len, v) in fields {
        let mask = ((1u128 << len) - 1) << at;
        value = (value & !mask) | ((u128::from(v) << at) & mask);
    }
    value
}
fn binary(f: &Fixture, stage: u8, instructions: &[u128], data: u64, control: u64) {
    let mut bytes = vec![0; 128];
    let header = if stage == 0 {
        1 | (4u32 << 5) | (1 << 10)
    } else {
        2 | (4u32 << 5) | (5 << 10)
    };
    bytes[..4].copy_from_slice(&header.to_le_bytes());
    if stage == 0 {
        bytes[53] = 0xF0;
    } else {
        bytes[72] = 15;
    }
    for i in instructions.iter().copied().chain([
        word(&[(0, 12, 0x94D)]),
        word(&[(0, 12, 0x947), (34, 48, (-4i64) as u64)]),
    ]) {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    write(f.backing + data, &bytes).unwrap();
    let mut meta = [0; 216];
    meta[..4].copy_from_slice(&0x9876_1234u32.to_le_bytes());
    meta[4..8].copy_from_slice(&201u32.to_le_bytes());
    meta[8..12].copy_from_slice(&7u32.to_le_bytes());
    meta[48..56].copy_from_slice(&216u64.to_le_bytes());
    meta[56..64].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    meta[86] = stage;
    write(f.backing + control, &meta).unwrap();
    write(
        f.b() + 0xD00 + u64::from(stage) * 16,
        &(f.backing + data).to_le_bytes(),
    )
    .unwrap();
    write(
        f.b() + 0xD08 + u64::from(stage) * 16,
        &(f.backing + control).to_le_bytes(),
    )
    .unwrap();
}
fn shaders(f: &Fixture) {
    binary(
        f,
        0,
        &[
            word(&[
                (0, 12, 0x321),
                (16, 8, 0),
                (24, 8, 255),
                (32, 8, 255),
                (40, 10, 0x80),
                (74, 2, 3),
            ]),
            word(&[
                (0, 12, 0x322),
                (24, 8, 255),
                (32, 8, 0),
                (40, 10, 0x70),
                (74, 2, 3),
            ]),
        ],
        0x1000,
        0x4000,
    );
    let mov = |r, value| word(&[(0, 12, 0x802), (16, 8, r), (32, 32, value), (72, 4, 15)]);
    binary(
        f,
        1,
        &[
            mov(0, 0),
            mov(1, 0x3F80_0000),
            mov(2, 0),
            mov(3, 0x3F80_0000),
        ],
        0x2000,
        0x5000,
    );
    f.call("nvnProgramInitialize", &[f.b() + 0x600, f.b()]);
    assert_eq!(
        f.call("nvnProgramSetShaders", &[f.b() + 0x600, 2, f.b() + 0xD00]),
        1
    );
}
fn geometry(f: &Fixture, x: f32) {
    let positions = [
        [x - 0.75, -0.75, 0.0, 1.0],
        [x + 0.75, -0.75, 0.0, 1.0],
        [x, 0.75, 0.0, 1.0],
    ];
    write(
        f.backing + 0x3000,
        &positions
            .into_iter()
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
}
fn read_image(f: &Fixture) -> Vec<u8> {
    let (renderer, surface) = {
        let mut d = f.session.driver.lock().unwrap();
        let crate::rt::nvn2::Driver {
            devices,
            textures,
            memory,
            ..
        } = &mut *d;
        (
            devices.renderer(f.b()).unwrap(),
            textures
                .display_image(f.b() + 0x880, memory, devices)
                .unwrap()
                .surface,
        )
    };
    let bytes = renderer
        .lock()
        .unwrap()
        .read_native_surface(&surface)
        .unwrap();
    bytes
}
fn record_with(f: &Fixture, buffer: u64, draw: impl FnOnce(&Fixture, u64)) -> u64 {
    let b = f.b();
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call(
        "nvnCommandBufferSetRenderTargets",
        &[buffer, 1, b + 0xC80, 0, 0, 0],
    );
    f.call("nvnCommandBufferBindProgram", &[buffer, b + 0x600, 0x1F]);
    f.call(
        "nvnCommandBufferBindVertexAttribState",
        &[buffer, 1, b + 0xD40],
    );
    f.call(
        "nvnCommandBufferBindVertexStreamState",
        &[buffer, 1, b + 0xD48],
    );
    f.call(
        "nvnCommandBufferBindVertexBuffer",
        &[buffer, 0, f.backing + 0x3000, 48],
    );
    f.call("nvnCommandBufferSetViewport", &[buffer, 0, 0, 32, 24]);
    f.call("nvnCommandBufferSetScissor", &[buffer, 0, 0, 32, 24]);
    f.call("nvnCommandBufferClearColor", &[buffer, 0, b + 0xD60, 15]);
    draw(f, buffer);
    f.call("nvnCommandBufferEndRecording", &[buffer])
}
fn record(f: &Fixture, buffer: u64) -> u64 {
    record_with(f, buffer, |f, buffer| {
        f.call("nvnCommandBufferDrawArrays", &[buffer, 4, 0, 3]);
    })
}
fn setup_draw(f: &Fixture, buffer: u64) {
    let b = f.b();
    f.window();
    shaders(f);
    geometry(f, 0.0);
    f.call("nvnVertexAttribStateSetDefaults", &[b + 0xD40]);
    f.call("nvnVertexAttribStateSetFormat", &[b + 0xD40, 46, 0]);
    f.call("nvnVertexStreamStateSetDefaults", &[b + 0xD48]);
    f.call("nvnVertexStreamStateSetStride", &[b + 0xD48, 16]);
    write(
        b + 0xD60,
        &[0f32, 0.0, 1.0, 1.0].map(f32::to_le_bytes).concat(),
    )
    .unwrap();
    f.initialize(buffer);
}

fn center(pixels: &[u8]) -> [u8; 4] {
    pixels[(12 * 32 + 16) * 4..(12 * 32 + 16) * 4 + 4]
        .try_into()
        .unwrap()
}

#[test]
fn array_draw_fixed_state_changes_the_actual_pixels() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    for (origin, cull, expected) in [
        (0, 2, [0, 255, 0, 255]),
        (1, 2, [0, 255, 0, 255]),
        (1, 1, [0, 0, 255, 255]),
    ] {
        f.call("nvnDeviceSetWindowOriginMode", &[b, origin]);
        f.call("nvnPolygonStateSetDefaults", &[b + 0xDA0]);
        f.call("nvnPolygonStateSetCullFace", &[b + 0xDA0, cull]);
        f.memory(buffer, 0x8000, 0xD000);
        let handle = record_with(&f, buffer, |f, buffer| {
            f.call("nvnCommandBufferBindPolygonState", &[buffer, b + 0xDA0]);
            f.call("nvnCommandBufferDrawArrays", &[buffer, 4, 0, 3]);
        });
        f.submit(handle);
        f.call("nvnQueueFlush", &[b + 0x200]);
        assert_eq!(
            center(&read_image(&f)),
            expected,
            "origin {origin}, cull {cull}"
        );
    }
    f.call("nvnPolygonStateSetCullFace", &[b + 0xDA0, 0]);
    f.call("nvnColorStateSetDefaults", &[b + 0xDB0]);
    f.call("nvnColorStateSetBlendEnable", &[b + 0xDB0, 0, 1]);
    f.call("nvnBlendStateSetDefaults", &[b + 0xDB8]);
    f.call("nvnBlendStateSetBlendFunc", &[b + 0xDB8, 0x61, 0x62, 2, 1]);
    write(b + 0xDC0, &[0.5f32; 4].map(f32::to_le_bytes).concat()).unwrap();
    for (mask, logic, expected) in [
        (15, 3, [0, 128, 128, 255]),
        (2, 3, [0, 128, 255, 255]),
        (15, 6, [0, 255, 255, 0]),
    ] {
        f.call("nvnColorStateSetLogicOp", &[b + 0xDB0, logic]);
        f.call("nvnChannelMaskStateSetDefaults", &[b + 0xDD0]);
        f.call(
            "nvnChannelMaskStateSetChannelMask",
            &[
                b + 0xDD0,
                0,
                mask & 1,
                mask >> 1 & 1,
                mask >> 2 & 1,
                mask >> 3 & 1,
            ],
        );
        f.memory(buffer, 0x8000, 0xD000);
        let handle = record_with(&f, buffer, |f, buffer| {
            f.call("nvnCommandBufferBindPolygonState", &[buffer, b + 0xDA0]);
            f.call("nvnCommandBufferBindColorState", &[buffer, b + 0xDB0]);
            f.call("nvnCommandBufferBindBlendState", &[buffer, b + 0xDB8]);
            f.call("nvnCommandBufferBindChannelMaskState", &[buffer, b + 0xDD0]);
            f.call("nvnCommandBufferSetBlendColor", &[buffer, b + 0xDC0]);
            f.call("nvnCommandBufferDrawArrays", &[buffer, 4, 0, 3]);
        });
        // The recording owns state, including the blend constants and output mask.
        write(b + 0xDC0, &[0; 16]).unwrap();
        f.submit(handle);
        f.call("nvnQueueFlush", &[b + 0x200]);
        let got = center(&read_image(&f));
        assert!(
            got.iter().zip(expected).all(|(&a, b)| a.abs_diff(b) <= 1),
            "mask {mask}, logic {logic}: {got:?}"
        );
        write(b + 0xDC0, &[0.5f32; 4].map(f32::to_le_bytes).concat()).unwrap();
    }
}

#[test]
fn array_instances_use_the_nonzero_base_and_increment_the_shader_id() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    binary(
        &f,
        0,
        &[
            word(&[
                (0, 12, 0x321),
                (16, 8, 0),
                (24, 8, 255),
                (32, 8, 255),
                (40, 10, 0x80),
                (74, 2, 3),
            ]),
            word(&[
                (0, 12, 0x321),
                (16, 8, 4),
                (24, 8, 255),
                (32, 8, 255),
                (40, 10, 0x2F8),
            ]),
            word(&[
                (0, 12, 0x306),
                (16, 8, 4),
                (32, 8, 4),
                (75, 2, 2),
                (84, 2, 2),
            ]),
            word(&[
                (0, 12, 0x820),
                (16, 8, 4),
                (24, 8, 4),
                (32, 32, 0.75f32.to_bits().into()),
                (84, 3, 4),
            ]),
            word(&[(0, 12, 0x221), (16, 8, 0), (24, 8, 0), (32, 8, 4)]),
            word(&[
                (0, 12, 0x322),
                (24, 8, 255),
                (32, 8, 0),
                (40, 10, 0x70),
                (74, 2, 3),
            ]),
        ],
        0x1000,
        0x4000,
    );
    assert_eq!(
        f.call("nvnProgramSetShaders", &[b + 0x600, 2, b + 0xD00]),
        1
    );
    let positions = [
        [-1.325f32, -0.5, 0.0, 1.0],
        [-0.925, -0.5, 0.0, 1.0],
        [-1.125, 0.5, 0.0, 1.0],
    ];
    write(
        f.backing + 0x3000,
        &positions
            .into_iter()
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let handle = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawArraysInstanced",
            &[buffer, 4, 0, 3, 1, 2],
        );
    });
    f.submit(handle);
    f.call("nvnQueueFlush", &[b + 0x200]);
    let pixels = read_image(&f);
    for (x, expected) in [
        (10, [0, 255, 0, 255]),
        (22, [0, 255, 0, 255]),
        (16, [0, 0, 255, 255]),
        (2, [0, 0, 255, 255]),
    ] {
        assert_eq!(
            &pixels[(12 * 32 + x) * 4..(12 * 32 + x) * 4 + 4],
            &expected,
            "x {x}"
        );
    }
}

#[test]
fn fragment_position_y_uses_pixel_centers_in_the_recorded_origin() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    let mov = |r, value| word(&[(0, 12, 0x802), (16, 8, r), (32, 32, value), (72, 4, 15)]);
    binary(
        &f,
        1,
        &[
            mov(0, 0),
            mov(2, 0),
            mov(3, 1f32.to_bits().into()),
            word(&[(0, 12, 0x326), (16, 8, 1), (64, 8, 0x74 / 4)]),
            word(&[
                (0, 12, 0x820),
                (16, 8, 1),
                (24, 8, 1),
                (32, 32, (1f32 / 32.0).to_bits().into()),
                (84, 3, 4),
            ]),
        ],
        0x2000,
        0x5000,
    );
    assert_eq!(
        f.call("nvnProgramSetShaders", &[b + 0x600, 2, b + 0xD00]),
        1
    );
    for origin in [0, 1] {
        f.call("nvnDeviceSetWindowOriginMode", &[b, origin]);
        f.memory(buffer, 0x8000, 0xD000);
        let handle = record(&f, buffer);
        f.submit(handle);
        f.call("nvnQueueFlush", &[b + 0x200]);
        let pixels = read_image(&f);
        for y in [9, 12, 15] {
            let p = &pixels[(y * 32 + 16) * 4..(y * 32 + 16) * 4 + 4];
            let physical_y = y as f32 + 0.5;
            let logical_y = if origin == 0 {
                24.0 - physical_y
            } else {
                physical_y
            };
            let green = (logical_y / 32.0 * 255.0).round() as u8;
            assert_eq!([p[0], p[2], p[3]], [0, 0, 255]);
            assert!(
                p[1].abs_diff(green) <= 1,
                "origin {origin}, y {y}: {p:?}, expected green {green}"
            );
        }
    }
}

#[test]
fn an_out_of_range_vertex_read_refuses_the_batch_before_drawing() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    f.memory(buffer, 0x8000, 0xD000);
    let handle = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferBindVertexBuffer",
            &[buffer, 0, f.backing + 0x3000, 32],
        );
        f.call("nvnCommandBufferDrawArrays", &[buffer, 4, 0, 3]);
    });
    f.submit(handle);
    assert_ne!(
        native_call(&f.session, "nvnQueueFlush", &[b + 0x200])
            .ctl
            .stop,
        stop::NONE
    );
    let d = f.session.driver.lock().unwrap();
    assert_eq!(d.queues.input_stats(), (0, 0, 0));
    assert_eq!(d.queues.command_stats().0, 0);
    assert_eq!(d.windows.stats().draws, 0);
}

#[test]
fn typed_array_draws_run_sm86_and_reuse_only_unchanged_vertex_inputs() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    f.window();
    shaders(&f);
    geometry(&f, 0.0);
    f.call("nvnVertexAttribStateSetDefaults", &[b + 0xD40]);
    f.call("nvnVertexAttribStateSetFormat", &[b + 0xD40, 46, 0]);
    f.call("nvnVertexStreamStateSetDefaults", &[b + 0xD48]);
    f.call("nvnVertexStreamStateSetStride", &[b + 0xD48, 16]);
    write(
        b + 0xD60,
        &[0f32, 0.0, 1.0, 1.0].map(f32::to_le_bytes).concat(),
    )
    .unwrap();
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    let handle = record(&f, buffer);
    // Bindings are copied; vertex bytes remain deferred inputs until submission.
    write(b + 0xD40, &[0; 16]).unwrap();
    f.call("nvnProgramFinalize", &[b + 0x600]);
    f.call("nvnCommandBufferFinalize", &[buffer]);
    f.submit(handle);
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert_eq!(
        f.session.driver.lock().unwrap().queues.input_stats(),
        (48, 0, 48)
    );
    let pixels = read_image(&f);
    assert_eq!(
        &pixels[(12 * 32 + 16) * 4..(12 * 32 + 16) * 4 + 4],
        &[0, 255, 0, 255]
    );
    assert_eq!(&pixels[..4], &[0, 0, 255, 255]);
    let green = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| **p == [0, 255, 0, 255])
        .count();
    assert!((150..350).contains(&green), "triangle pixels {green}");
    f.submit(handle);
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert_eq!(
        f.session.driver.lock().unwrap().queues.input_stats(),
        (48, 1, 48)
    );
    assert_eq!(read_image(&f), pixels);
    // Rewriting a watched input creates a fresh GPU buffer; an older submission keeps its version.
    geometry(&f, 0.6);
    f.submit(handle);
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert_eq!(
        f.session.driver.lock().unwrap().queues.input_stats(),
        (96, 1, 96)
    );
    let changed = read_image(&f);
    assert_ne!(changed, pixels);
    let d = f.session.driver.lock().unwrap();
    assert_eq!(d.windows.stats().draws, 3);
    assert_eq!(d.programs.counts().0, 0);
    let readback = d.windows.stats().readback_bytes;
    drop(d);
    f.call("nvnQueuePresentTexture", &[b + 0x200, b + 0xA00, 0]);
    assert_eq!(
        f.session
            .driver
            .lock()
            .unwrap()
            .windows
            .stats()
            .readback_bytes,
        readback
    );
    if let Some(dir) = std::env::var_os("KILN_NVN2_COMMAND_DUMP") {
        let path = std::path::PathBuf::from(dir).join("native_array_triangle.png");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        nx_vulkan::png::write_png(&path, 32, 24, &pixels).unwrap();
    }
}

#[test]
fn scissored_gpu_clears_copy_state_and_respect_recorded_window_origin() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let buffer = b + 0x500;
    f.window();
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    write(
        b + 0xD60,
        &[1f32, 0.0, 0.0, 1.0].map(f32::to_le_bytes).concat(),
    )
    .unwrap();
    write(
        b + 0xD70,
        &[0f32, 1.0, 0.0, 1.0].map(f32::to_le_bytes).concat(),
    )
    .unwrap();
    write(b + 0xD80, &[-3i32, 2, 11, 7].map(i32::to_le_bytes).concat()).unwrap();
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call(
        "nvnCommandBufferSetRenderTargets",
        &[buffer, 1, b + 0xC80, 0, 0, 0],
    );
    f.call("nvnCommandBufferClearColor", &[buffer, 0, b + 0xD60, 15]);
    f.call("nvnCommandBufferSetViewport", &[buffer, 0, 0, 1, 1]);
    f.call("nvnCommandBufferSetScissors", &[buffer, 0, 1, b + 0xD80]);
    f.call("nvnCommandBufferClearColor", &[buffer, 0, b + 0xD70, 15]);
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.call("nvnDeviceSetWindowOriginMode", &[b, 1]);
    write(b + 0xD80, &[0; 16]).unwrap();
    f.submit(handle);
    f.call("nvnQueueFlush", &[b + 0x200]);
    let pixels = read_image(&f);
    for y in 0..24 {
        for x in 0..32 {
            let expected = if x < 8 && (15..22).contains(&y) {
                [0, 255, 0, 255]
            } else {
                [255, 0, 0, 255]
            };
            assert_eq!(
                &pixels[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4],
                &expected,
                "pixel {x},{y}"
            );
        }
    }
    if let Some(dir) = std::env::var_os("KILN_NVN2_COMMAND_DUMP") {
        let path = std::path::PathBuf::from(dir).join("native_scissor_clear.png");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        nx_vulkan::png::write_png(&path, 32, 24, &pixels).unwrap();
    }
}
