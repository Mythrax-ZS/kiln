use super::*;
use crate::rt::abi::stop;
use crate::rt::nvn2::tests::Memory;

#[test]
fn queue_builders_preserve_padding_and_use_boolean_size_getters_with_checked_outputs() {
    let memory = Memory::new();
    let (builder, output) = (memory.0, memory.0 + 0x100);
    write(builder, &[0xAA; 80]).unwrap();
    let mut cpu = NativeCpu::new(0);
    let owner = |_| Ok(1);
    let mut call = |name: &str, args: &[u64]| {
        cpu.state.x[..args.len()].copy_from_slice(args);
        builder_call(name, &mut cpu, &owner).unwrap().unwrap();
        cpu.state.x[0]
    };
    call("nvnQueueBuilderSetDevice", &[builder, 0x1234_0000]);
    call("nvnQueueBuilderSetDefaults", &[builder]);
    let bytes = read::<80>(builder).unwrap();
    assert_eq!(&bytes[..8], &[0xAA; 8]);
    assert_eq!(word(&bytes, 8), 0x1234_0000);
    assert_eq!(&bytes[16..41], &[0; 25]);
    assert_eq!(&bytes[41..44], &[0xAA; 3]);
    assert_eq!(&bytes[44..68], &[0; 24]);
    assert_eq!(integer(&bytes, 68), 100);
    assert_eq!(&bytes[72..], &[0xAA; 8]);
    write(output, &[0xFF; 8]).unwrap();
    assert_eq!(
        call("nvnQueueBuilderGetCommandMemorySize", &[builder, output]),
        0
    );
    assert_eq!(
        read::<8>(output).unwrap(),
        [0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF]
    );
    call(
        "nvnQueueBuilderSetCommandMemorySize",
        &[builder, 0xABCD_FFFF_0002_0000],
    );
    assert_eq!(
        call("nvnQueueBuilderGetCommandMemorySize", &[builder, output]),
        1
    );
    assert_eq!(u32::from_le_bytes(read(output).unwrap()), 131072);
    assert_eq!(
        call("nvnQueueBuilderGetQueueMemorySize", &[builder]),
        147456
    );
    for (set, get, value) in [
        (
            "nvnQueueBuilderSetComputeMemorySize",
            "nvnQueueBuilderGetComputeMemorySize",
            0,
        ),
        (
            "nvnQueueBuilderSetControlMemorySize",
            "nvnQueueBuilderGetControlMemorySize",
            8192,
        ),
        (
            "nvnQueueBuilderSetRayTracingMemorySize",
            "nvnQueueBuilderGetRayTracingMemorySize",
            4096,
        ),
        (
            "nvnQueueBuilderSetCommandFlushThreshold",
            "nvnQueueBuilderGetCommandFlushThreshold",
            4096,
        ),
    ] {
        call(set, &[builder, value]);
        assert_eq!(call(get, &[builder, output]), 1);
        assert_eq!(u32::from_le_bytes(read(output).unwrap()), value as u32);
    }
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..2].copy_from_slice(&[builder, memory.0 + 0xFFE]);
    let error = builder_call("nvnQueueBuilderGetControlMemorySize", &mut cpu, &owner)
        .unwrap()
        .unwrap_err();
    error.report(&mut cpu, 0x800_1000);
    assert_eq!(cpu.ctl.stop, stop::FAULT);
    assert_eq!(cpu.ctl.fault_addr, memory.0 + 0x1000);
}

#[cfg(feature = "vulkan")]
#[test]
fn native_queue_fences_wait_for_flush_and_keep_parent_and_device_lifetimes() {
    use crate::rt::nvn2::{tests::native_call, Mode, Session};
    use std::sync::{Arc, Barrier};
    let mut probe = nx_vulkan::Renderer::new(nx_vulkan::RenderConfig::default());
    if let Err(e) = probe.initialize() {
        assert!(
            std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
            "Vulkan required: {e}"
        );
        eprintln!("SKIPPED (no Vulkan device): {e}");
        return;
    }
    drop(probe);
    let memory = Memory::new();
    let (device, device_builder, queue, secondary, builder, sync) = (
        memory.0,
        memory.0 + 0x100,
        memory.0 + 0x200,
        memory.0 + 0x300,
        memory.0 + 0x400,
        memory.0 + 0x500,
    );
    let session = Arc::new(Session::new(Mode::Hle));
    native_call(&session, "nvnDeviceBuilderSetDefaults", &[device_builder]);
    assert_eq!(
        native_call(&session, "nvnDeviceInitialize", &[device, device_builder])
            .state
            .x[0],
        1
    );
    native_call(&session, "nvnQueueBuilderSetDevice", &[builder, device]);
    native_call(&session, "nvnQueueBuilderSetDefaults", &[builder]);
    native_call(&session, "nvnQueueBuilderSetFlags", &[builder, 9]);
    assert_eq!(
        native_call(&session, "nvnQueueInitialize", &[queue, builder])
            .state
            .x[0],
        1
    );
    native_call(&session, "nvnQueueBuilderSetFlags", &[builder, 5]);
    native_call(
        &session,
        "nvnQueueBuilderSetPrimaryQueue",
        &[builder, queue],
    );
    assert_eq!(
        native_call(&session, "nvnQueueInitialize", &[secondary, builder])
            .state
            .x[0],
        1
    );
    assert_eq!(
        native_call(&session, "nvnQueueFinalize", &[queue]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    assert_eq!(
        native_call(&session, "nvnDeviceFinalize", &[device])
            .ctl
            .stop,
        stop::UNIMPLEMENTED
    );
    assert_eq!(
        native_call(&session, "nvnSyncInitialize", &[sync, device])
            .state
            .x[0],
        1
    );
    assert_eq!(
        native_call(&session, "nvnSyncWait", &[sync, 0]).state.x[0],
        3
    );
    assert_eq!(
        native_call(&session, "nvnQueueFenceSync", &[queue, sync, 0, 1])
            .ctl
            .stop,
        stop::NONE
    );
    assert_eq!(
        native_call(&session, "nvnSyncWait", &[sync, 0]).state.x[0],
        2,
        "an unflushed fence is pending"
    );
    assert_eq!(
        native_call(&session, "nvnQueueWaitSync", &[secondary, sync])
            .ctl
            .stop,
        stop::UNIMPLEMENTED,
        "cross-queue waits require a flush"
    );
    let rendezvous = Arc::new(Barrier::new(2));
    let (worker_session, worker_barrier) = (Arc::clone(&session), Arc::clone(&rendezvous));
    let join = std::thread::spawn(move || {
        worker_barrier.wait();
        let cpu = native_call(&worker_session, "nvnSyncWait", &[sync, 5_000_000_000]);
        (cpu.ctl.stop, cpu.state.x[0])
    });
    rendezvous.wait();
    assert_eq!(
        native_call(&session, "nvnQueueFlush", &[queue]).ctl.stop,
        stop::NONE
    );
    let (stopped, result) = join.join().unwrap();
    assert_eq!(stopped, stop::NONE);
    assert!(result <= 1, "CPU wait observes a real GPU completion");
    assert_eq!(
        native_call(&session, "nvnSyncWait", &[sync, 0]).state.x[0],
        0
    );
    assert_eq!(
        native_call(&session, "nvnQueueWaitSync", &[secondary, sync])
            .state
            .x[0],
        1
    );
    assert_eq!(
        native_call(&session, "nvnQueueFinish", &[secondary])
            .ctl
            .stop,
        stop::NONE
    );
    // Re-fencing must create a new completion instead of resetting a fence another waiter owns.
    native_call(&session, "nvnQueueFenceSync", &[queue, sync, 1, 0]);
    assert_eq!(
        native_call(&session, "nvnSyncWait", &[sync, 0]).state.x[0],
        2
    );
    native_call(&session, "nvnQueueFinish", &[queue]);
    assert_eq!(
        native_call(&session, "nvnSyncWait", &[sync, 0]).state.x[0],
        0
    );
    assert_eq!(
        native_call(&session, "nvnQueueGetError", &[queue, 0])
            .state
            .x[0],
        0
    );
    for object in [secondary, queue] {
        assert_eq!(
            native_call(&session, "nvnQueueFinalize", &[object])
                .ctl
                .stop,
            stop::NONE
        );
    }
    assert_eq!(
        native_call(&session, "nvnDeviceFinalize", &[device])
            .ctl
            .stop,
        stop::UNIMPLEMENTED,
        "live sync holds its device"
    );
    native_call(&session, "nvnSyncFinalize", &[sync]);
    assert_eq!(
        native_call(&session, "nvnSyncWait", &[sync, 0]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    assert_eq!(
        native_call(&session, "nvnDeviceFinalize", &[device])
            .ctl
            .stop,
        stop::NONE
    );
}
