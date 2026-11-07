use super::*;
use crate::rt::nvn2::tests::Memory;

fn builder(name: &str, args: &[u64]) -> NativeCpu {
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..args.len()].copy_from_slice(args);
    builder_call(name, &mut cpu, &|_| Ok(1)).unwrap().unwrap();
    cpu
}

#[test]
fn window_builder_preserves_device_and_handles_interval_encoding() {
    let m = Memory::new();
    let b = m.0;
    write(b, &[0xAA; 104]).unwrap();
    builder("nvnWindowBuilderSetDevice", &[b, 0x8000]);
    builder("nvnWindowBuilderSetDefaults", &[b]);
    assert_eq!(read::<8>(b).unwrap(), [0xAA; 8]);
    assert_eq!(
        builder("nvnWindowBuilderGetDevice", &[b]).state.x[0],
        0x8000
    );
    assert_eq!(read::<8>(b + 96).unwrap(), [0xAA; 8]);
    assert_eq!(
        builder("nvnWindowBuilderGetPresentInterval", &[b]).state.x[0],
        1
    );
    builder("nvnWindowBuilderSetPresentInterval", &[b, 0]);
    assert_eq!(read::<4>(b + 32).unwrap(), [0xFF; 4]);
    assert_eq!(
        builder("nvnWindowBuilderGetPresentInterval", &[b]).state.x[0],
        0
    );
    write(b + 0x200, &0x1234u64.to_le_bytes()).unwrap();
    builder("nvnWindowBuilderSetTextures", &[b, 1, b + 0x200]);
    assert_eq!(
        builder("nvnWindowBuilderGetTexture", &[b, 0]).state.x[0],
        0x1234
    );
    assert_eq!(builder("nvnWindowBuilderGetTexture", &[b, 1]).state.x[0], 0);
    assert_eq!(
        builder("nvnWindowBuilderGetTexture", &[b, u64::MAX])
            .state
            .x[0],
        0
    );
}

#[test]
fn hdr_metadata_uses_fp_registers_stack_arguments_and_checked_outputs() {
    let m = Memory::new();
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[0] = m.0;
    cpu.state.sp = m.0 + 0x100;
    for i in 0..8 {
        cpu.state.v[i][0] = f32::to_bits(i as f32 / 10.0) as u64;
    }
    write(cpu.state.sp, &8f32.to_le_bytes()).unwrap();
    write(cpu.state.sp + 8, &9f32.to_le_bytes()).unwrap();
    builder_call("nvnWindowBuilderSetHdrDisplayMetadata", &mut cpu, &|_| {
        Ok(1)
    })
    .unwrap()
    .unwrap();
    assert_eq!(read::<4>(m.0 + 80).unwrap(), 8f32.to_le_bytes());
    assert_eq!(read::<4>(m.0 + 84).unwrap(), 9f32.to_le_bytes());
    for i in 0..7 {
        cpu.state.x[i + 1] = m.0 + 0x200 + i as u64 * 4;
    }
    for i in 7..10 {
        write(
            cpu.state.sp + (i - 7) * 8,
            &(m.0 + 0x200 + i * 4).to_le_bytes(),
        )
        .unwrap();
    }
    builder_call("nvnWindowBuilderGetHdrDisplayMetadata", &mut cpu, &|_| {
        Ok(1)
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        read::<40>(m.0 + 48).unwrap(),
        read::<40>(m.0 + 0x200).unwrap()
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn native_windows_present_gpu_images_and_enforce_acquisition_and_parent_lifetimes() {
    use crate::rt::abi::{perm, stop};
    use crate::rt::nvn2::{rt, tests::native_call, Mode, Session};
    let m = Memory::new();
    let session = Session::new(Mode::Hle);
    let (device, db, pool, queue, qb, tb, wb, array, window, sync, output, pb) = (
        m.0,
        m.0 + 0x100,
        m.0 + 0x180,
        m.0 + 0x300,
        m.0 + 0x400,
        m.0 + 0x480,
        m.0 + 0x680,
        m.0 + 0x700,
        m.0 + 0x800,
        m.0 + 0xA20,
        m.0 + 0xA60,
        m.0 + 0xB00,
    );
    let textures = [m.0 + 0x500, m.0 + 0x580, m.0 + 0x600];
    native_call(&session, "nvnDeviceBuilderSetDefaults", &[db]);
    if native_call(&session, "nvnDeviceInitialize", &[device, db])
        .state
        .x[0]
        == 0
    {
        assert!(
            std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
            "Vulkan is required"
        );
        eprintln!("SKIPPED (no Vulkan device)");
        return;
    }
    let call = |name: &str, args: &[u64]| {
        let c = native_call(&session, name, args);
        assert_eq!(c.ctl.stop, stop::NONE, "{name}");
        assert_eq!(c.ctl.fault_flag, 0, "{name}");
        c.state.x[0]
    };
    let backing = 0x6B_0000_0000;
    rt().mem.map(backing, 0x10000, perm::R | perm::W).unwrap();
    call("nvnMemoryPoolBuilderSetDefaults", &[pb]);
    call("nvnMemoryPoolBuilderSetDevice", &[pb, device]);
    call("nvnMemoryPoolBuilderSetFlags", &[pb, 0x22]);
    call("nvnMemoryPoolBuilderSetStorage", &[pb, backing, 0x10000]);
    assert_eq!(call("nvnMemoryPoolInitialize", &[pool, pb]), 1);
    call("nvnQueueBuilderSetDefaults", &[qb]);
    call("nvnQueueBuilderSetDevice", &[qb, device]);
    assert_eq!(call("nvnQueueInitialize", &[queue, qb]), 1);
    call("nvnTextureBuilderSetDefaults", &[tb]);
    call("nvnTextureBuilderSetDevice", &[tb, device]);
    call("nvnTextureBuilderSetFlags", &[tb, 1]);
    call("nvnTextureBuilderSetSize2D", &[tb, 32, 24]);
    call("nvnTextureBuilderSetFormat", &[tb, 37]);
    for (i, t) in textures.into_iter().enumerate() {
        call(
            "nvnTextureBuilderSetStorage",
            &[tb, pool, i as u64 * 0x2000],
        );
        assert_eq!(call("nvnTextureInitialize", &[t, tb]), 1);
        write(array + i as u64 * 8, &t.to_le_bytes()).unwrap();
    }
    call("nvnWindowBuilderSetDefaults", &[wb]);
    call("nvnWindowBuilderSetDevice", &[wb, device]);
    let native_handle = m.0 + 0xB80;
    call("nvnWindowBuilderSetNativeWindow", &[wb, native_handle]);
    call("nvnWindowBuilderSetTextures", &[wb, 3, array]);
    call("nvnWindowBuilderSetPresentInterval", &[wb, 0]);
    assert_eq!(call("nvnWindowInitialize", &[window, wb]), 1);
    assert!(session.report().contains("# native windows owned: 1"));
    assert_eq!(call("nvnWindowGetNumActiveTextures", &[window]), 3);
    assert_eq!(call("nvnWindowGetNativeWindow", &[window]), native_handle);
    assert_eq!(call("nvnSyncInitialize", &[sync, device]), 1);
    assert_eq!(call("nvnSyncWait", &[sync, 0]), 3);
    let failed = native_call(
        &session,
        "nvnWindowAcquireTexture",
        &[window, sync, m.0 + 0xFFE],
    );
    assert_ne!(failed.ctl.fault_flag, 0);
    assert_eq!(
        call("nvnSyncWait", &[sync, 0]),
        3,
        "bad output must not replace the sync"
    );
    assert_ne!(
        native_call(&session, "nvnTextureFinalize", &[textures[0]])
            .ctl
            .stop,
        stop::NONE
    );
    let renderer = session
        .driver
        .lock()
        .unwrap()
        .devices
        .renderer(device)
        .unwrap();
    let images: Vec<_> = {
        let mut driver = session.driver.lock().unwrap();
        let super::super::Driver {
            textures: owned,
            memory,
            devices,
            ..
        } = &mut *driver;
        textures
            .iter()
            .map(|&t| owned.display_image(t, memory, devices).unwrap().surface)
            .collect()
    };
    for frame in 0..6 {
        let index = frame % 3;
        assert_eq!(call("nvnWindowAcquireTexture", &[window, sync, output]), 0);
        assert_eq!(u32::from_le_bytes(read(output).unwrap()) as usize, index);
        assert_eq!(
            call("nvnSyncWait", &[sync, 0]),
            0,
            "availability is a completed GPU fence"
        );
        assert_ne!(
            native_call(&session, "nvnWindowAcquireTexture", &[window, sync, output])
                .ctl
                .stop,
            stop::NONE
        );
        assert_ne!(
            native_call(&session, "nvnWindowSetNumActiveTextures", &[window, 2])
                .ctl
                .stop,
            stop::NONE
        );
        let mut color = [0.0; 4];
        color[index] = 1.0;
        color[3] = 1.0;
        images[index].clear_color(color).unwrap();
        let readbacks = images[index].stats().readback_bytes;
        call("nvnQueuePresentTexture", &[queue, window, index as u64]);
        assert_eq!(
            images[index].stats().readback_bytes,
            readbacks,
            "headless presentation does not copy pixels to the CPU"
        );
        assert_ne!(
            native_call(
                &session,
                "nvnQueuePresentTexture",
                &[queue, window, index as u64]
            )
            .ctl
            .stop,
            stop::NONE
        );
        let rgba = renderer
            .lock()
            .unwrap()
            .read_native_surface(&images[index])
            .unwrap();
        let pixel = color.map(|v| (v * 255.0) as u8);
        assert!(rgba.as_chunks::<4>().0.iter().all(|p| *p == pixel));
        if let Some(dir) = std::env::var_os("KILN_NVN2_WINDOW_DUMP") {
            let path = std::path::PathBuf::from(dir).join(format!("native_window_{frame}.png"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            nx_vulkan::png::write_png(&path, 32, 24, &rgba).unwrap();
        }
    }
    call("nvnWindowSetNumActiveTextures", &[window, 2]);
    call("nvnSyncFinalize", &[sync]);
    assert_eq!(
        call("nvnWindowAcquireTexture", &[window, sync, output]),
        0,
        "acquisition constructs an availability sync in uninitialized storage"
    );
    assert_eq!(call("nvnSyncWait", &[sync, 0]), 0);
    call("nvnQueuePresentTexture", &[queue, window, 0]);
    let mut foreign = nx_vulkan::Renderer::new(nx_vulkan::RenderConfig::default());
    assert!(foreign
        .read_native_surface(&images[0])
        .unwrap_err()
        .contains("another Vulkan device"));
    drop(foreign);
    call("nvnWindowSetCrop", &[window, 3, 4, 12, 9]);
    call("nvnWindowGetCrop", &[window, output]);
    assert_eq!(
        read::<16>(output).unwrap(),
        [3i32, 4, 12, 9].map(i32::to_le_bytes).concat().as_slice()
    );
    call("nvnWindowSetCrop", &[window, 99, 99, 0, 1]);
    call("nvnWindowGetCrop", &[window, output]);
    assert_eq!(
        read::<16>(output).unwrap(),
        [0i32, 0, 32, 24].map(i32::to_le_bytes).concat().as_slice()
    );
    call("nvnWindowFinalize", &[window]);
    assert!(session.report().contains("# native windows owned: 0"));
    for t in textures {
        call("nvnTextureFinalize", &[t]);
    }
    call("nvnMemoryPoolFinalize", &[pool]);
    call("nvnQueueFinalize", &[queue]);
    call("nvnSyncFinalize", &[sync]);
    call("nvnDeviceFinalize", &[device]);
    assert!(
        images[2].completion().wait(0).unwrap(),
        "retained image/fence keeps the device alive"
    );
    assert_eq!(
        renderer
            .lock()
            .unwrap()
            .read_native_surface(&images[2])
            .unwrap()
            .len(),
        32 * 24 * 4
    );
    drop(images);
    drop(renderer);
    rt().mem.unmap(backing, 0x10000).unwrap();
}
