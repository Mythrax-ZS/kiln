use super::*;

pub(super) fn event(f: &Fixture, object: u64, pool: u64, offset: u64) {
    let builder = f.b() + 0xE80;
    f.call("nvnEventBuilderSetDefaults", &[builder]);
    f.call("nvnEventBuilderSetStorage", &[builder, pool, offset]);
    assert_eq!(f.call("nvnEventInitialize", &[object, builder]), 1);
}
#[test]
fn event_abi_preserves_storage_padding_output_order_and_cpu_visibility() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let builder = b + 0xE80;
    let object = b + 0xE00;
    write(builder, &[0xAB; 24]).unwrap();
    f.call("nvnEventBuilderSetDefaults", &[builder]);
    assert_eq!(read::<8>(builder + 16).unwrap(), [0xAB; 8]);
    f.call("nvnEventBuilderSetStorage", &[builder, b + 0x380, 0x6000]);
    // GetStorage stores offset before rereading the pool, including aliased output storage.
    assert_eq!(
        f.call("nvnEventBuilderGetStorage", &[builder, builder]),
        0x6000
    );
    write(f.backing + 0x6000, &(u32::MAX - 3).to_le_bytes()).unwrap();
    event(&f, object, b + 0x380, 0x6000);
    assert_eq!(
        f.call("nvnEventGetValue", &[object]),
        u64::from(u32::MAX - 3)
    );
    assert_eq!(
        u64::from_le_bytes(read(object + 8).unwrap()),
        f.backing + 0x6000
    );
    assert_eq!(f.call("nvnEventGetMemoryPool", &[object]), b + 0x380);
    assert_eq!(f.call("nvnEventGetMemoryOffset", &[object]), 0x6000);
    f.call("nvnEventSignal", &[object, 1, 8]);
    assert_eq!(f.call("nvnEventGetValue", &[object]), 4);
    f.call("nvnEventSignal", &[object, 0, 0xABCD_0000_1020_3040]);
    assert_eq!(
        u32::from_le_bytes(read(f.backing + 0x6000).unwrap()),
        0x1020_3040
    );
    assert_ne!(
        native_call(&f.session, "nvnMemoryPoolFinalize", &[b + 0x380])
            .ctl
            .stop,
        stop::NONE
    );
    f.call("nvnEventFinalize", &[object]);
    assert_ne!(
        native_call(&f.session, "nvnEventGetValue", &[object])
            .ctl
            .stop,
        stop::NONE
    );
    f.call("nvnMemoryPoolFinalize", &[b + 0x380]);
}

#[test]
fn deferred_events_keep_their_pool_until_gpu_completion_and_observe_inline_stores() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x6F_0000_0000);
    let backing = NEXT.fetch_add(0x20000, Ordering::Relaxed);
    rt().mem.map(backing, 0x10000, perm::R | perm::W).unwrap();
    let pool = b + 0x700;
    let builder = b + 0x780;
    f.call("nvnMemoryPoolBuilderSetDefaults", &[builder]);
    f.call("nvnMemoryPoolBuilderSetDevice", &[builder, b]);
    f.call("nvnMemoryPoolBuilderSetFlags", &[builder, 0x22]);
    f.call(
        "nvnMemoryPoolBuilderSetStorage",
        &[builder, backing, 0x10000],
    );
    assert_eq!(f.call("nvnMemoryPoolInitialize", &[pool, builder]), 1);
    event(&f, b + 0xE00, pool, 0x100);
    let buffer = b + 0x500;
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call("nvnCommandBufferWaitEvent", &[buffer, b + 0xE00, 0, 7]);
    f.call(
        "nvnCommandBufferSignalEvent",
        &[buffer, b + 0xE00, 1, 2, 0, 5],
    );
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.call("nvnCommandBufferFinalize", &[buffer]);
    f.call("nvnEventFinalize", &[b + 0xE00]);
    f.submit(handle);
    f.call("nvnSyncInitialize", &[b + 0xC40, b]);
    f.call("nvnQueueFenceSync", &[b + 0x200, b + 0xC40, 0, 1]);
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert_eq!(f.call("nvnSyncWait", &[b + 0xC40, 1_000_000]), 2);
    assert_ne!(
        native_call(&f.session, "nvnMemoryPoolFinalize", &[pool])
            .ctl
            .stop,
        stop::NONE
    );
    assert!(
        f.session.driver.try_lock().is_ok(),
        "event wait must not hold the driver mutex"
    );
    // A caller's inline store has no native notification. The worker observes actual storage.
    rt().mem
        .write(backing + 0x100, 4, 7, Ordering::Release)
        .unwrap();
    assert!(f.call("nvnSyncWait", &[b + 0xC40, 5_000_000_000]) <= 1);
    assert_eq!(u32::from_le_bytes(read(backing + 0x100).unwrap()), 12);
    f.call("nvnMemoryPoolFinalize", &[pool]);
    assert_ne!(
        native_call(
            &f.session,
            "nvnQueueSubmitCommands",
            &[b + 0x200, 1, b + 0xD00]
        )
        .ctl
        .stop,
        stop::NONE
    );
    rt().mem.unmap(backing, 0x10000).unwrap();
}

#[test]
fn queue_finalization_waits_outside_the_driver_so_another_thread_can_signal() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    event(&f, b + 0xE00, b + 0x380, 0x6000);
    let buffer = b + 0x500;
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call("nvnCommandBufferWaitEvent", &[buffer, b + 0xE00, 0, 7]);
    f.call(
        "nvnCommandBufferSignalEvent",
        &[buffer, b + 0xE00, 1, 2, 0, 1],
    );
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.submit(handle);
    let session = Arc::clone(&f.session);
    let (sent, received) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let cpu = native_call(&session, "nvnQueueFinalize", &[b + 0x200]);
        sent.send(cpu.ctl.stop).unwrap();
    });
    assert!(received
        .recv_timeout(std::time::Duration::from_millis(10))
        .is_err());
    f.call("nvnEventSignal", &[b + 0xE00, 0, 7]);
    assert_eq!(
        received
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        stop::NONE
    );
    worker.join().unwrap();
    assert_eq!(f.call("nvnEventGetValue", &[b + 0xE00]), 8);
    f.call("nvnEventFinalize", &[b + 0xE00]);
}

#[test]
fn failed_event_initialization_leaves_no_object_or_storage_write() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let object = b + 0xE00;
    write(object, &[0xAB; 40]).unwrap();
    write(b + 0xFF8, &(b + 0x380).to_le_bytes()).unwrap();
    let cpu = native_call(&f.session, "nvnEventInitialize", &[object, b + 0xFF8]);
    assert_eq!(cpu.ctl.fault_flag, 1);
    assert_eq!(read::<40>(object).unwrap(), [0xAB; 40]);
    assert_eq!(f.session.driver.lock().unwrap().events.count(), 0);
    let builder = b + 0xE80;
    f.call("nvnEventBuilderSetDefaults", &[builder]);
    f.call("nvnEventBuilderSetStorage", &[builder, b + 0x380, 3]);
    assert_ne!(
        native_call(&f.session, "nvnEventInitialize", &[object, builder])
            .ctl
            .stop,
        stop::NONE
    );
    assert_eq!(read::<40>(object).unwrap(), [0xAB; 40]);
}

#[test]
fn a_cross_queue_sync_wait_tracks_a_flushed_prefix_still_waiting_on_an_event() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    event(&f, b + 0xE00, b + 0x380, 0x6000);
    let buffer = b + 0x500;
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call("nvnCommandBufferWaitEvent", &[buffer, b + 0xE00, 0, 9]);
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.submit(handle);
    f.call("nvnSyncInitialize", &[b + 0xC40, b]);
    f.call("nvnQueueFenceSync", &[b + 0x200, b + 0xC40, 0, 1]);
    f.call("nvnQueueBuilderSetDefaults", &[b + 0x800]);
    f.call("nvnQueueBuilderSetDevice", &[b + 0x800, b]);
    assert_eq!(f.call("nvnQueueInitialize", &[b + 0xF00, b + 0x800]), 1);
    assert_ne!(
        native_call(&f.session, "nvnQueueWaitSync", &[b + 0xF00, b + 0xC40])
            .ctl
            .stop,
        stop::NONE
    );
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert_eq!(f.call("nvnQueueWaitSync", &[b + 0xF00, b + 0xC40]), 1);
    f.call("nvnSyncInitialize", &[b + 0xC60, b]);
    f.call("nvnQueueFenceSync", &[b + 0xF00, b + 0xC60, 0, 1]);
    f.call("nvnQueueFlush", &[b + 0xF00]);
    assert_eq!(
        f.call("nvnSyncWait", &[b + 0xC60, 1_000_000]),
        2,
        "the second queue cannot outrun the first deferred prefix"
    );
    f.call("nvnEventSignal", &[b + 0xE00, 0, 9]);
    assert!(f.call("nvnSyncWait", &[b + 0xC60, 5_000_000_000]) <= 1);
    f.call("nvnQueueFinalize", &[b + 0xF00]);
    f.call("nvnEventFinalize", &[b + 0xE00]);
}
