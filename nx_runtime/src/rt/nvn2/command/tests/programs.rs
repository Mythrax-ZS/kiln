use super::*;
use nx_vulkan::native::ShaderOptions;
use nx_vulkan::render::Variation;

fn shader(f: &Fixture, stage: u8, data_offset: u64, control_offset: u64) {
    let mut bytes = vec![0; 128];
    let header = match stage {
        0 => 1 | (4u32 << 5) | (1 << 10),
        1 => 2 | (4u32 << 5) | (5 << 10),
        5 => {
            bytes.clear();
            0
        }
        _ => panic!("synthetic stage"),
    };
    if !bytes.is_empty() {
        bytes[..4].copy_from_slice(&header.to_le_bytes());
    }
    let exit = 0x94du128 | (7 << 12) | (7 << 87);
    let end = 0x947u128 | (7 << 12) | (7 << 87) | (((1u128 << 48) - 4) << 34);
    bytes.extend_from_slice(&exit.to_le_bytes());
    bytes.extend_from_slice(&end.to_le_bytes());
    write(f.backing + data_offset, &bytes).unwrap();
    let mut control = [0; 216];
    control[..4].copy_from_slice(&0x9876_1234u32.to_le_bytes());
    control[4..8].copy_from_slice(&201u32.to_le_bytes());
    control[8..12].copy_from_slice(&7u32.to_le_bytes());
    control[48..56].copy_from_slice(&216u64.to_le_bytes());
    control[56..64].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    control[85] = 0xAB; // The adjacent field is not the stage.
    control[86] = stage;
    write(f.backing + control_offset, &control).unwrap();
}

fn entry(f: &Fixture, index: u64, data_offset: u64, control_offset: u64, extended: bool) {
    let at = f.b() + 0xC00 + index * if extended { 24 } else { 16 };
    write(at, &(f.backing + data_offset).to_le_bytes()).unwrap();
    write(at + 8, &(f.backing + control_offset).to_le_bytes()).unwrap();
    if extended {
        write(at + 16, &(f.backing + 0x7000 + index * 24).to_le_bytes()).unwrap();
    }
}

fn variation(stage_index: u32) -> ShaderOptions {
    ShaderOptions::Graphics(Variation {
        stage_index,
        attribute_classes: Default::default(),
        target_classes: Default::default(),
        depth_minus_one_to_one: false,
        mesh: None,
        per_primitive: 0,
    })
}

#[test]
fn program_builder_captures_binaries_and_relinks_atomically() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let (b, program, builder) = (f.b(), f.b() + 0x800, f.b() + 0x900);
    write(builder, &[0xAB; 104]).unwrap();
    f.call("nvnProgramBuilderSetDevice", &[builder, b]);
    f.call("nvnProgramBuilderSetDefaults", &[builder]);
    let defaults = read::<104>(builder).unwrap();
    for at in [0, 16, 44, 60, 76, 100] {
        assert_eq!(defaults[at], 0xAB);
    }
    assert_eq!(u64::from_le_bytes(defaults[8..16].try_into().unwrap()), b);
    assert_eq!(
        f.call("nvnProgramBuilderGetMaxTraceRecursionDepth", &[builder]),
        2
    );
    shader(&f, 0, 0x1000, 0x4000);
    shader(&f, 1, 0x2000, 0x5000);
    entry(&f, 0, 0x1000, 0x4000, false);
    entry(&f, 1, 0x2000, 0x5000, false);
    write(f.backing + 0x7000, &[0x54; 48]).unwrap();
    f.call("nvnProgramBuilderSetShaders", &[builder, 2, b + 0xC00]);
    f.call(
        "nvnProgramBuilderSetDebugDataHashes",
        &[builder, 1, f.backing + 0x7000],
    );
    f.call("nvnProgramBuilderSetSampleShading", &[builder, 2]);
    assert_eq!(
        f.call("nvnProgramInitializeFromBuilder", &[program, builder]),
        1
    );
    assert_eq!(f.call("nvnProgramInitialize", &[program, b]), 0);
    let snapshot = {
        let d = f.session.driver.lock().unwrap();
        assert_eq!(d.programs.counts(), (1, 2, 0));
        assert_eq!(d.programs.captured_bytes, 320);
        d.programs.snapshot(program, &d.devices, &d.memory).unwrap()
    };
    assert_eq!(snapshot.sample_shading, 2);
    assert_eq!(snapshot.link.mask, 3);
    assert_ne!(
        native_call(&f.session, "nvnMemoryPoolFinalize", &[b + 0x380])
            .ctl
            .stop,
        stop::NONE
    );
    assert_ne!(
        native_call(&f.session, "nvnDeviceFinalize", &[b]).ctl.stop,
        stop::NONE
    );
    // A fault on the last control pointer cannot replace the first/previous linked stages.
    write(b + 0xC18, &(b + 0xFF8).to_le_bytes()).unwrap();
    let fault = native_call(&f.session, "nvnProgramSetShaders", &[program, 2, b + 0xC00]);
    assert_ne!(fault.ctl.fault_flag, 0);
    {
        let d = f.session.driver.lock().unwrap();
        let after = d.programs.snapshot(program, &d.devices, &d.memory).unwrap();
        assert!(Arc::ptr_eq(&after.link, &snapshot.link));
        assert_eq!(d.programs.captured_bytes, 320);
    }
    // Caller metadata is copied. Code changes without shader-cache invalidation do not
    // alter already captured binaries; their executable pool must nevertheless stay alive.
    write(f.backing + 0x1000, &[0xFF; 160]).unwrap();
    write(f.backing + 0x4000, &[0xFF; 216]).unwrap();
    write(b + 0xC00, &[0xFF; 32]).unwrap();
    write(f.backing + 0x7000, &[0xFF; 48]).unwrap();
    let first = snapshot.link.shaders[0]
        .binary
        .prepare(variation(0), &())
        .unwrap();
    let again = snapshot.link.shaders[0]
        .binary
        .prepare(variation(0), &())
        .unwrap();
    assert!(Arc::ptr_eq(&first, &again));
    snapshot.link.shaders[1]
        .binary
        .prepare(variation(4), &())
        .unwrap();
    assert_eq!(f.call("nvnProgramSetShaders", &[program, 0, 0]), 1);
    assert_eq!(snapshot.link.mask, 3);
    f.call("nvnProgramFinalize", &[program]);
    assert_ne!(
        native_call(&f.session, "nvnProgramFinalize", &[program])
            .ctl
            .stop,
        stop::NONE
    );
    f.call("nvnQueueFinalize", &[b + 0x200]);
    f.call("nvnMemoryPoolFinalize", &[b + 0x380]);
    f.call("nvnDeviceFinalize", &[b]);
    // Modules still retain their own Vulkan device after API object retirement.
    assert_eq!(first.id, again.id);
}

#[test]
fn recorded_shader_bindings_survive_relink_and_object_finalization() {
    let Some(f) = Fixture::with_pool_flags(0x62) else {
        return;
    };
    let (b, program, buffer) = (f.b(), f.b() + 0x800, f.b() + 0x500);
    shader(&f, 0, 0x1000, 0x4000);
    shader(&f, 1, 0x2000, 0x5000);
    entry(&f, 0, 0x1000, 0x4000, true);
    entry(&f, 1, 0x2000, 0x5000, true);
    assert_eq!(f.call("nvnProgramInitialize", &[program, b]), 1);
    assert_eq!(
        f.call("nvnProgramSetShadersExt", &[program, 2, b + 0xC00]),
        1
    );
    f.call("nvnProgramSetSampleShading", &[program, 1]);
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call(
        "nvnCommandBufferBindProgram",
        &[buffer, program, 0xFFFF_FFFF_0000_001F],
    );
    let used = f.call("nvnCommandBufferGetCommandMemoryUsed", &[buffer]);
    assert_ne!(
        native_call(&f.session, "nvnCommandBufferBarrier", &[buffer, 0x20])
            .ctl
            .stop,
        stop::NONE
    );
    assert_eq!(
        f.call("nvnCommandBufferGetCommandMemoryUsed", &[buffer]),
        used
    );
    shader(&f, 5, 0x3000, 0x6000);
    entry(&f, 0, 0x3000, 0x6000, false);
    assert_eq!(f.call("nvnProgramSetShaders", &[program, 1, b + 0xC00]), 1);
    f.call("nvnCommandBufferBindProgram", &[buffer, program, 0x20]);
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.call("nvnProgramFinalize", &[program]);
    f.submit(handle);
    f.call("nvnQueueFlush", &[b + 0x200]);
    {
        let d = f.session.driver.lock().unwrap();
        assert_eq!(d.queues.binding_stats(), (3, 1));
        assert_eq!(
            d.queues.command_stats().0,
            0,
            "binding alone submits no GPU work"
        );
        let r = d.commands.resolve(handle, &d.memory, &d.devices).unwrap();
        let Op::Program {
            binding: Some(old), ..
        } = &r.ops[0]
        else {
            panic!("graphics snapshot")
        };
        assert_eq!(old.link.mask, 3);
        assert!(r.has_pool(b + 0x380));
    }
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    f.call("nvnCommandBufferBindProgram", &[buffer, 0, 2]);
    let unbind = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.submit(unbind);
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert_eq!(
        f.session.driver.lock().unwrap().queues.binding_stats(),
        (2, 0)
    );
    f.call("nvnCommandBufferFinalize", &[buffer]);
    f.call("nvnQueueFinalize", &[b + 0x200]);
    f.call("nvnMemoryPoolFinalize", &[b + 0x380]);
    let d = f.session.driver.lock().unwrap();
    assert!(d.commands.resolve(handle, &d.memory, &d.devices).is_err());
}

#[test]
fn program_rejects_unsupported_controls_stages_and_nonexecutable_storage() {
    let Some(f) = Fixture::with_pool_flags(0x22) else {
        return;
    };
    let (b, program) = (f.b(), f.b() + 0x800);
    shader(&f, 0, 0x1000, 0x4000);
    entry(&f, 0, 0x1000, 0x4000, false);
    f.call("nvnProgramInitialize", &[program, b]);
    assert_ne!(
        native_call(&f.session, "nvnProgramSetShaders", &[program, 1, b + 0xC00])
            .ctl
            .stop,
        stop::NONE
    );
    write(f.backing + 0x4008, &99u32.to_le_bytes()).unwrap();
    assert_ne!(
        native_call(&f.session, "nvnProgramSetShaders", &[program, 1, b + 0xC00])
            .ctl
            .stop,
        stop::NONE
    );
    write(f.backing + 0x4008, &7u32.to_le_bytes()).unwrap();
    write(f.backing + 0x4056, &[8]).unwrap();
    assert_ne!(
        native_call(&f.session, "nvnProgramSetShaders", &[program, 1, b + 0xC00])
            .ctl
            .stop,
        stop::NONE
    );
    let d = f.session.driver.lock().unwrap();
    assert_eq!(d.programs.counts(), (1, 0, 0));
    assert_eq!(d.programs.captured_bytes, 0);
}

#[test]
fn shader_pool_leases_guard_recording_and_queued_sets_on_another_command_pool() {
    let Some(f) = Fixture::new() else { return };
    let storage = GuestPage::new();
    let (b, pool, builder, program, buffer) = (
        f.b(),
        f.b() + 0xE00,
        f.b() + 0xF00,
        f.b() + 0x800,
        f.b() + 0x500,
    );
    f.call("nvnMemoryPoolBuilderSetDefaults", &[builder]);
    f.call("nvnMemoryPoolBuilderSetDevice", &[builder, b]);
    f.call("nvnMemoryPoolBuilderSetFlags", &[builder, 0x62]);
    f.call(
        "nvnMemoryPoolBuilderSetStorage",
        &[builder, storage.0, 4096],
    );
    assert_eq!(f.call("nvnMemoryPoolInitialize", &[pool, builder]), 1);
    shader(&f, 0, 0x1000, 0x4000);
    write(storage.0, &read::<160>(f.backing + 0x1000).unwrap()).unwrap();
    write(storage.0 + 0x400, &read::<216>(f.backing + 0x4000).unwrap()).unwrap();
    write(b + 0xC00, &storage.0.to_le_bytes()).unwrap();
    write(b + 0xC08, &(storage.0 + 0x400).to_le_bytes()).unwrap();
    f.call("nvnProgramInitialize", &[program, b]);
    assert_eq!(f.call("nvnProgramSetShaders", &[program, 1, b + 0xC00]), 1);
    // A different device cannot source this pool's code or bind into this device's recorder.
    f.call("nvnDeviceBuilderSetDefaults", &[b + 0xB00]);
    assert_eq!(f.call("nvnDeviceInitialize", &[b + 0xA00, b + 0xB00]), 1);
    f.call("nvnProgramInitialize", &[b + 0x900, b + 0xA00]);
    assert_ne!(
        native_call(
            &f.session,
            "nvnProgramSetShaders",
            &[b + 0x900, 1, b + 0xC00]
        )
        .ctl
        .stop,
        stop::NONE
    );
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    assert_ne!(
        native_call(
            &f.session,
            "nvnCommandBufferBindProgram",
            &[buffer, b + 0x900, 1]
        )
        .ctl
        .stop,
        stop::NONE
    );
    f.call("nvnProgramFinalize", &[b + 0x900]);
    f.call("nvnDeviceFinalize", &[b + 0xA00]);
    f.call("nvnCommandBufferBindProgram", &[buffer, program, 1]);
    f.call("nvnProgramFinalize", &[program]);
    assert_ne!(
        native_call(&f.session, "nvnMemoryPoolFinalize", &[pool])
            .ctl
            .stop,
        stop::NONE
    );
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    f.call("nvnCommandBufferFinalize", &[buffer]);
    {
        let d = f.session.driver.lock().unwrap();
        assert!(d
            .commands
            .resolve(handle, &d.memory, &d.devices)
            .unwrap()
            .has_pool(pool));
    }
    f.submit(handle);
    assert_ne!(
        native_call(&f.session, "nvnMemoryPoolFinalize", &[pool])
            .ctl
            .stop,
        stop::NONE
    );
    f.call("nvnQueueFlush", &[b + 0x200]);
    f.call("nvnMemoryPoolFinalize", &[pool]);
    {
        let d = f.session.driver.lock().unwrap();
        assert!(d.commands.resolve(handle, &d.memory, &d.devices).is_err());
    }
    f.call("nvnQueueFinalize", &[b + 0x200]);
}
