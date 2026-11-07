use super::*;
use crate::rt::nvn2::tests::{native_call, Memory as GuestPage};

fn index_bytes(kind: u32, values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&n| n.to_le_bytes()[..1usize << kind].to_vec())
        .collect()
}
#[test]
fn indexed_draws_use_all_widths_signed_base_vertices_and_warm_cached_inputs() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    for kind in 0..3 {
        write(f.backing + 0x6000, &index_bytes(kind, &[3, 4, 5, 3, 4, 5])).unwrap();
        f.memory(buffer, 0x8000, 0xD000);
        let set = record_with(&f, buffer, |f, buffer| {
            f.call(
                "nvnCommandBufferDrawElementsBaseVertex",
                &[
                    buffer,
                    4,
                    u64::from(kind),
                    6,
                    f.backing + 0x6000,
                    (-3i64) as u64,
                ],
            );
        });
        for _ in 0..2 {
            f.submit(set);
            f.flush();
            assert_eq!(center(&read_image(&f)), [0, 255, 0, 255]);
        }
    }
    let driver = f.session.driver.lock().unwrap();
    assert_eq!(driver.windows.stats().draws, 6);
    assert_eq!(
        driver.queues.input_stats().0,
        48,
        "vertex upload follows the index range, not index count"
    );
    let (copied, hits, uploaded, shared) = driver.queues.index_stats();
    assert_eq!((copied, hits, shared), (42, 3, 0));
    assert!(matches!(uploaded, 44 | 48));
}

#[test]
fn indexed_restart_splits_custom_indices_and_uses_fixed_strip_restart() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    for (mode, restart, values, green) in [
        (4, 7, vec![0, 1, 2, 7, 0, 1, 2], true),
        (4, 7, vec![0, 1, 7, 2], false),
        (5, 255, vec![0, 1, 2, 255, 0, 1, 2], true),
        (5, 255, vec![255, 255, 255], false),
        (4, 256, vec![0, 1, 2], true),
    ] {
        write(f.backing + 0x6000, &index_bytes(0, &values)).unwrap();
        f.memory(buffer, 0x8000, 0xD000);
        let set = record_with(&f, buffer, |f, buffer| {
            f.call("nvnCommandBufferSetPrimitiveRestart", &[buffer, 1, restart]);
            f.call(
                "nvnCommandBufferDrawElements",
                &[buffer, mode, 0, values.len() as u64, f.backing + 0x6000],
            );
        });
        f.submit(set);
        f.flush();
        assert_eq!(
            center(&read_image(&f)),
            if green {
                [0, 255, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        );
    }
}

#[test]
fn a_gpu_signal_before_an_indexed_draw_changes_the_actual_index_fetch() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    write(f.backing + 0x6000, &index_bytes(2, &[1, 1, 2])).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let first = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawElements",
            &[buffer, 4, 2, 3, f.backing + 0x6000],
        );
    });
    f.submit(first);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 0, 255, 255]);
    super::super::events::event(&f, b + 0xE00, b + 0x380, 0x6000);
    f.memory(buffer, 0x8000, 0xD000);
    let second = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferSignalEvent",
            &[buffer, b + 0xE00, 0, 2, 0, 0],
        );
        f.call(
            "nvnCommandBufferDrawElements",
            &[buffer, 4, 2, 3, f.backing + 0x6000],
        );
    });
    f.submit(second);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 255, 0, 255]);
    let driver = f.session.driver.lock().unwrap();
    assert_eq!(driver.queues.input_stats().0, 48);
    assert_eq!(
        driver.queues.index_stats(),
        (12, 0, 12, 12),
        "GPU-written indices bypass the old immutable cache without copyback/upload"
    );
}

fn pool(f: &Fixture, object: u64, builder: u64, address: u64, size: u64, flags: u64) {
    f.call("nvnMemoryPoolBuilderSetDefaults", &[builder]);
    f.call("nvnMemoryPoolBuilderSetDevice", &[builder, f.b()]);
    f.call("nvnMemoryPoolBuilderSetFlags", &[builder, flags]);
    f.call("nvnMemoryPoolBuilderSetStorage", &[builder, address, size]);
    assert_eq!(f.call("nvnMemoryPoolInitialize", &[object, builder]), 1);
}
#[test]
fn indexed_recordings_pin_the_index_pool_and_retire_with_its_identity() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let index_memory = GuestPage::new();
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    let index_pool = b + 0xE80;
    pool(&f, index_pool, b + 0xE00, index_memory.0, 4096, 0x22);
    write(index_memory.0, &[0, 1, 2]).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let set = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawElements",
            &[buffer, 4, 0, 3, index_memory.0],
        );
        assert_eq!(
            native_call(&f.session, "nvnMemoryPoolFinalize", &[index_pool])
                .ctl
                .stop,
            stop::UNIMPLEMENTED
        );
    });
    f.submit(set);
    assert_eq!(
        native_call(&f.session, "nvnMemoryPoolFinalize", &[index_pool])
            .ctl
            .stop,
        stop::UNIMPLEMENTED
    );
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 255, 0, 255]);
    f.call("nvnMemoryPoolFinalize", &[index_pool]);
    write(b + 0xD00, &set.to_le_bytes()).unwrap();
    assert_eq!(
        native_call(
            &f.session,
            "nvnQueueSubmitCommands",
            &[b + 0x200, 1, b + 0xD00]
        )
        .ctl
        .stop,
        stop::UNIMPLEMENTED
    );
}

#[test]
fn malformed_index_addresses_and_out_of_range_vertices_fail_before_vertex_upload() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    f.memory(buffer, 0x8000, 0xD000);
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    for args in [
        [buffer, 4, 3, 3, f.backing + 0x6000],
        [buffer, 4, 1, 3, f.backing + 0x6001],
        [buffer, 4, 2, u64::MAX, f.backing + 0x6000],
        [buffer, 4, 2, 3, f.backing + 0xFFFC],
    ] {
        assert_eq!(
            native_call(&f.session, "nvnCommandBufferDrawElements", &args)
                .ctl
                .stop,
            stop::UNIMPLEMENTED
        );
    }
    f.call("nvnCommandBufferEndRecording", &[buffer]);
    write(f.backing + 0x6000, &index_bytes(2, &[0, 1, 1000])).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let set = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawElements",
            &[buffer, 4, 2, 3, f.backing + 0x6000],
        );
    });
    f.submit(set);
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert_eq!(
        native_call(&f.session, "nvnQueueFinish", &[b + 0x200])
            .ctl
            .stop,
        stop::UNIMPLEMENTED
    );
    let driver = f.session.driver.lock().unwrap();
    assert_eq!(driver.queues.input_stats().0, 0);
    assert_eq!(driver.windows.stats().draws, 0);
    assert!(driver
        .queues
        .background_error()
        .unwrap()
        .contains("indexed"));
}

#[test]
fn indexed_instances_keep_base_vertex_separate_from_base_instance() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    instance_geometry(&f);
    write(f.backing + 0x6000, &[1, 2, 3]).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let set = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawElementsInstanced",
            &[buffer, 4, 0, 3, f.backing + 0x6000, (-1i64) as u64, 1, 2],
        );
    });
    f.submit(set);
    f.flush();
    let pixels = read_image(&f);
    for (x, green) in [(10, true), (22, true), (16, false), (2, false)] {
        assert_eq!(
            &pixels[(12 * 32 + x) * 4..(12 * 32 + x) * 4 + 4],
            if green {
                &[0, 255, 0, 255]
            } else {
                &[0, 0, 255, 255]
            },
            "x{x}"
        );
    }
}

struct Storage(u64);
impl Storage {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0x70_0000_0000);
        let at = NEXT.fetch_add(0x40000, Ordering::Relaxed);
        rt().mem.map(at, 0x20000, perm::R | perm::W).unwrap();
        Self(at)
    }
}
impl Drop for Storage {
    fn drop(&mut self) {
        rt().mem.unmap(self.0, 0x20000).unwrap();
    }
}
#[test]
fn virtual_index_fetches_cross_physical_pages_and_observe_changed_cpu_bytes() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let meta = GuestPage::new();
    let storage = Storage::new();
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    let (physical, virtual_pool, builder, request) =
        (meta.0, meta.0 + 0x100, meta.0 + 0x200, meta.0 + 0x300);
    pool(&f, physical, builder, storage.0, 0x20000, 0x10A);
    pool(&f, virtual_pool, builder, 0, 0x20000, 0x221);
    let mut map = [0u8; 40];
    map[..8].copy_from_slice(&physical.to_le_bytes());
    map[24..32].copy_from_slice(&0x20000u64.to_le_bytes());
    write(request, &map).unwrap();
    assert_eq!(
        f.call("nvnMemoryPoolMapVirtual", &[virtual_pool, 1, request]),
        1
    );
    let gpu = f.call("nvnMemoryPoolGetBufferAddress", &[virtual_pool]);
    assert!(gpu >= 1 << 39);
    write(storage.0 + 65534, &index_bytes(1, &[0, 1, 2])).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let set = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawElements",
            &[buffer, 4, 1, 3, gpu + 65534],
        );
    });
    f.submit(set);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 255, 0, 255]);
    write(storage.0 + 65536, &index_bytes(1, &[2, 2])).unwrap();
    f.submit(set);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 0, 255, 255]);
    assert_eq!(
        f.session.driver.lock().unwrap().queues.index_stats(),
        (12, 0, 16, 0)
    );
    f.call("nvnMemoryPoolFinalize", &[virtual_pool]);
    f.call("nvnMemoryPoolFinalize", &[physical]);
}

#[test]
fn empty_indexed_draws_need_no_index_data_or_render_bindings() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let buffer = b + 0x500;
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call("nvnCommandBufferDrawElements", &[buffer, 4, 2, 0, 1]);
    f.call(
        "nvnCommandBufferDrawElementsInstanced",
        &[buffer, 4, 2, 3, 1, 0, 0, 0],
    );
    let set = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.submit(set);
    f.flush();
    assert_eq!(
        f.session.driver.lock().unwrap().queues.index_stats(),
        (0, 0, 0, 0)
    );
}

#[test]
fn a_smaller_index_range_reuses_a_larger_unchanged_vertex_prefix() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    write(f.backing + 0x6000, &[0, 1, 2]).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let set = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawElements",
            &[buffer, 4, 0, 3, f.backing + 0x6000],
        );
    });
    f.submit(set);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 255, 0, 255]);
    write(f.backing + 0x6000, &[0, 1, 1]).unwrap();
    f.submit(set);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 0, 255, 255]);
    let driver = f.session.driver.lock().unwrap();
    assert_eq!(driver.queues.input_stats(), (48, 1, 48));
    assert_eq!(driver.queues.index_stats().0, 6);
}

#[test]
fn a_fresh_larger_vertex_prefix_replaces_a_stale_exact_cache_entry() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let b = f.b();
    let buffer = b + 0x500;
    setup_draw(&f, buffer);
    write(f.backing + 0x6000, &[0, 1, 1]).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let set = record_with(&f, buffer, |f, buffer| {
        f.call(
            "nvnCommandBufferDrawElements",
            &[buffer, 4, 0, 3, f.backing + 0x6000],
        );
    });
    f.submit(set);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 0, 255, 255]);
    write(f.backing + 0x3000, &(-0.4f32).to_le_bytes()).unwrap();
    write(f.backing + 0x6000, &[0, 1, 2]).unwrap();
    f.submit(set);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 255, 0, 255]);
    write(f.backing + 0x6000, &[0, 1, 1]).unwrap();
    f.submit(set);
    f.flush();
    assert_eq!(center(&read_image(&f)), [0, 0, 255, 255]);
    assert_eq!(
        f.session.driver.lock().unwrap().queues.input_stats(),
        (80, 1, 80)
    );
}

#[test]
fn custom_restart_keeps_instance_major_blend_order_and_shader_ids() {
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
                (32, 32, 0.5f32.to_bits().into()),
                (84, 3, 4),
            ]),
            word(&[
                (0, 12, 0x322),
                (24, 8, 255),
                (32, 8, 0),
                (40, 10, 0x70),
                (74, 2, 3),
            ]),
            word(&[(0, 12, 0x322), (24, 8, 255), (32, 8, 4), (40, 10, 0x80)]),
        ],
        0x1000,
        0x4000,
    );
    let mov = |r, value| word(&[(0, 12, 0x802), (16, 8, r), (32, 32, value), (72, 4, 15)]);
    binary(
        &f,
        1,
        &[
            mov(0, 0),
            word(&[(0, 12, 0x326), (16, 8, 1), (64, 8, 0x80 / 4)]),
            mov(2, 0),
            mov(3, 0.5f32.to_bits().into()),
        ],
        0x2000,
        0x5000,
    );
    assert_eq!(
        f.call("nvnProgramSetShaders", &[b + 0x600, 2, b + 0xD00]),
        1
    );
    f.call("nvnColorStateSetDefaults", &[b + 0xDB0]);
    f.call("nvnColorStateSetBlendEnable", &[b + 0xDB0, 0, 1]);
    f.call("nvnBlendStateSetDefaults", &[b + 0xDB8]);
    f.call("nvnBlendStateSetBlendFunc", &[b + 0xDB8, 5, 6, 2, 1]);
    write(f.backing + 0x6000, &[0, 1, 2, 7, 0, 1, 2]).unwrap();
    f.memory(buffer, 0x8000, 0xD000);
    let set = record_with(&f, buffer, |f, buffer| {
        f.call("nvnCommandBufferBindColorState", &[buffer, b + 0xDB0]);
        f.call("nvnCommandBufferBindBlendState", &[buffer, b + 0xDB8]);
        f.call("nvnCommandBufferSetPrimitiveRestart", &[buffer, 1, 7]);
        f.call(
            "nvnCommandBufferDrawElementsInstanced",
            &[buffer, 4, 0, 7, f.backing + 0x6000, 0, 1, 2],
        );
    });
    f.submit(set);
    f.flush();
    let pixel = center(&read_image(&f));
    assert!(
        pixel
            .iter()
            .zip([0u8, 215, 16, 128])
            .all(|(&got, want)| got.abs_diff(want) <= 2),
        "{pixel:?}"
    );
}
