use super::*;
use crate::rt::abi::{perm, stop, FuncEntry, STOP_PC};
use crate::rt::nvn2::{
    tests::{native_call, Memory as GuestPage},
    Mode, Session,
};
use std::cell::RefCell;

thread_local! {
    static CALLBACK_SESSION: RefCell<Option<Arc<Session>>> = const { RefCell::new(None) };
}
const ALLOCATE: u64 = 0x7D00_0B00;
const NO_MEMORY: u64 = 0x7D00_0B04;
const UNWIND: u64 = 0x7D00_0B08;
const FAULT: u64 = 0x7D00_0B0C;

struct Fixture {
    page: GuestPage,
    session: Arc<Session>,
    backing: u64,
}
impl Fixture {
    fn new() -> Option<Self> {
        let page = GuestPage::new();
        let session = Arc::new(Session::new(Mode::Hle));
        let b = page.0;
        native_call(&session, "nvnDeviceBuilderSetDefaults", &[b + 0x100]);
        if native_call(&session, "nvnDeviceInitialize", &[b, b + 0x100])
            .state
            .x[0]
            == 0
        {
            assert!(
                std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
                "Vulkan required"
            );
            eprintln!("SKIPPED (no Vulkan device)");
            return None;
        }
        static NEXT: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0x6C_0000_0000);
        let backing = NEXT.fetch_add(0x20000, Ordering::Relaxed);
        rt().mem.map(backing, 0x10000, perm::R | perm::W).unwrap();
        let f = Self {
            page,
            session,
            backing,
        };
        f.call("nvnMemoryPoolBuilderSetDefaults", &[b + 0x400]);
        f.call("nvnMemoryPoolBuilderSetDevice", &[b + 0x400, b]);
        f.call("nvnMemoryPoolBuilderSetFlags", &[b + 0x400, 0x22]);
        f.call(
            "nvnMemoryPoolBuilderSetStorage",
            &[b + 0x400, backing, 0x10000],
        );
        assert_eq!(
            f.call("nvnMemoryPoolInitialize", &[b + 0x380, b + 0x400]),
            1
        );
        f.call("nvnQueueBuilderSetDefaults", &[b + 0x300]);
        f.call("nvnQueueBuilderSetDevice", &[b + 0x300, b]);
        assert_eq!(f.call("nvnQueueInitialize", &[b + 0x200, b + 0x300]), 1);
        Some(f)
    }
    fn b(&self) -> u64 {
        self.page.0
    }
    fn call(&self, name: &str, args: &[u64]) -> u64 {
        let cpu = native_call(&self.session, name, args);
        assert_eq!(cpu.ctl.stop, stop::NONE, "{name}");
        assert_eq!(cpu.ctl.fault_flag, 0, "{name}");
        cpu.state.x[0]
    }
    fn initialize(&self, buffer: u64) {
        assert_eq!(
            self.call("nvnCommandBufferInitialize", &[buffer, self.b()]),
            1
        );
    }
    fn memory(&self, buffer: u64, command_offset: u64, control_offset: u64) {
        self.call(
            "nvnCommandBufferAddCommandMemory",
            &[buffer, self.b() + 0x380, command_offset, 1024],
        );
        self.call(
            "nvnCommandBufferAddControlMemory",
            &[buffer, self.backing + control_offset, 512],
        );
    }
    fn record_barrier(&self, buffer: u64) -> u64 {
        self.call("nvnCommandBufferBeginRecording", &[buffer]);
        self.call("nvnCommandBufferBarrier", &[buffer, 0x102]);
        self.call("nvnCommandBufferEndRecording", &[buffer])
    }
    fn submit(&self, handle: u64) {
        write(self.b() + 0xD00, &handle.to_le_bytes()).unwrap();
        self.call(
            "nvnQueueSubmitCommands",
            &[self.b() + 0x200, 1, self.b() + 0xD00],
        );
    }
    fn window(&self) {
        let b = self.b();
        self.call("nvnTextureBuilderSetDefaults", &[b + 0x800]);
        self.call("nvnTextureBuilderSetDevice", &[b + 0x800, b]);
        self.call("nvnTextureBuilderSetFlags", &[b + 0x800, 1]);
        self.call("nvnTextureBuilderSetSize2D", &[b + 0x800, 32, 24]);
        self.call("nvnTextureBuilderSetFormat", &[b + 0x800, 37]);
        for (i, texture) in [b + 0x880, b + 0x900].into_iter().enumerate() {
            self.call(
                "nvnTextureBuilderSetStorage",
                &[b + 0x800, b + 0x380, i as u64 * 8192],
            );
            assert_eq!(self.call("nvnTextureInitialize", &[texture, b + 0x800]), 1);
            write(b + 0xC80 + i as u64 * 8, &texture.to_le_bytes()).unwrap();
        }
        self.call("nvnWindowBuilderSetDefaults", &[b + 0x980]);
        self.call("nvnWindowBuilderSetDevice", &[b + 0x980, b]);
        self.call("nvnWindowBuilderSetNativeWindow", &[b + 0x980, 1]);
        self.call("nvnWindowBuilderSetTextures", &[b + 0x980, 2, b + 0xC80]);
        self.call("nvnWindowBuilderSetPresentInterval", &[b + 0x980, 0]);
        assert_eq!(self.call("nvnWindowInitialize", &[b + 0xA00, b + 0x980]), 1);
        self.call(
            "nvnWindowAcquireTexture",
            &[b + 0xA00, b + 0xC40, b + 0xCA0],
        );
        assert_eq!(read::<4>(b + 0xCA0).unwrap(), [0; 4]);
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        CALLBACK_SESSION.with(|s| *s.borrow_mut() = None);
        rt().mem.unmap(self.backing, 0x10000).unwrap();
    }
}

extern "C" fn allocate(cpu: *mut NativeCpu) {
    // SAFETY: the dispatcher lends the one live caller CPU for this guest callback.
    let cpu = unsafe { &mut *cpu };
    let (object, event, data, lr) = (
        cpu.state.x[0],
        cpu.state.x[1],
        cpu.state.x[3],
        cpu.state.x[30],
    );
    let session = CALLBACK_SESSION.with(|s| Arc::clone(s.borrow().as_ref().unwrap()));
    if session.driver.try_lock().is_err() {
        cpu.ctl.stop = stop::UNIMPLEMENTED;
        cpu.state.pc = STOP_PC;
        return;
    }
    let pool = u64::from_le_bytes(read(data).unwrap());
    let at = data + 8 + event * 8;
    let next = u64::from_le_bytes(read(at).unwrap());
    write(at, &(next + 512).to_le_bytes()).unwrap();
    let counter = data + 32 + event * 8;
    let count = u64::from_le_bytes(read(counter).unwrap());
    write(counter, &(count + 1).to_le_bytes()).unwrap();
    let (name, args) = if event == 0 {
        (
            "nvnCommandBufferAddCommandMemory",
            vec![object, pool, next, 512],
        )
    } else {
        ("nvnCommandBufferAddControlMemory", vec![object, next, 512])
    };
    cpu.state.x[..args.len()].copy_from_slice(&args);
    cpu.state.pc = session.bind(name, 0).unwrap();
    session.call(cpu);
    if cpu.ctl.stop == stop::NONE && cpu.ctl.fault_flag == 0 {
        cpu.state.fpsr |= 0x10;
        cpu.state.tpidr_el0 = data + 0x55;
        cpu.state.pc = lr;
    }
}
extern "C" fn no_memory(cpu: *mut NativeCpu) {
    // SAFETY: as in allocate.
    let cpu = unsafe { &mut *cpu };
    cpu.state.pc = cpu.state.x[30];
}
extern "C" fn unwind(cpu: *mut NativeCpu) {
    // SAFETY: as in allocate.
    let cpu = unsafe { &mut *cpu };
    cpu.state.pc = 0x800_2B00;
    cpu.state.x[0] = 77;
}
extern "C" fn fault(cpu: *mut NativeCpu) {
    // SAFETY: as in allocate.
    let cpu = unsafe { &mut *cpu };
    crate::rt::runtime::record_fault(
        cpu,
        0xBAD0_0000,
        cpu.state.pc,
        true,
        crate::a64::bus::FaultKind::Unmapped,
    );
}
fn register_callbacks() {
    for (addr, func) in [
        (ALLOCATE, allocate as *const std::ffi::c_void),
        (NO_MEMORY, no_memory as *const std::ffi::c_void),
        (UNWIND, unwind as *const std::ffi::c_void),
        (FAULT, fault as *const std::ffi::c_void),
    ] {
        rt().register(&[FuncEntry { addr, func }]);
    }
}

#[test]
fn native_callbacks_record_and_submit_a_real_gpu_clear_batch_without_pixel_uploads() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    f.window();
    let buffer = b + 0x500;
    let data = b + 0xE00;
    register_callbacks();
    CALLBACK_SESSION.with(|s| *s.borrow_mut() = Some(Arc::clone(&f.session)));
    write(data, &(b + 0x380).to_le_bytes()).unwrap();
    write(data + 8, &0x8000u64.to_le_bytes()).unwrap();
    write(data + 16, &(f.backing + 0xD000).to_le_bytes()).unwrap();
    f.initialize(buffer);
    f.call("nvnCommandBufferSetMemoryCallback", &[buffer, ALLOCATE]);
    f.call("nvnCommandBufferSetMemoryCallbackData", &[buffer, data]);
    let mut cpu = NativeCpu::new(0);
    cpu.state.pc = f.session.bind("nvnCommandBufferBeginRecording", 0).unwrap();
    cpu.state.x[0] = buffer;
    cpu.state.x[19] = 0x1234_5678;
    cpu.state.x[30] = 0x800_1004;
    cpu.state.sp = b + 0xF00;
    cpu.state.v[3] = [15, 16];
    f.session.call(&mut cpu);
    assert_eq!(cpu.ctl.stop, stop::NONE);
    assert_eq!(
        (cpu.state.x[19], cpu.state.sp, cpu.state.v[3]),
        (0x1234_5678, b + 0xF00, [15, 16])
    );
    assert_eq!((cpu.state.fpsr, cpu.state.tpidr_el0), (0x10, data + 0x55));
    f.call(
        "nvnCommandBufferSetRenderTargets",
        &[buffer, 1, b + 0xC80, 0, 0, 0],
    );
    for i in 0..30 {
        let color = if i == 29 {
            [0f32, 1.0, 0.0, 1.0]
        } else {
            [1f32, 0.0, 0.0, 1.0]
        };
        write(b + 0xD80, &color.map(f32::to_le_bytes).concat()).unwrap();
        f.call(
            "nvnCommandBufferClearColor",
            &[buffer, 0xABCD_0000_0000, b + 0xD80, 0xFFFF_FFFF_0000_000F],
        );
    }
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    // Non-deferred arguments are copied at recording time. Reusing the caller's arrays before
    // submission must not redirect the target or turn the recorded green clear blue.
    write(
        b + 0xD80,
        &[0f32, 0.0, 1.0, 1.0].map(f32::to_le_bytes).concat(),
    )
    .unwrap();
    write(b + 0xC80, &(b + 0x900).to_le_bytes()).unwrap();
    assert_eq!(u64::from_le_bytes(read(data + 32).unwrap()), 2);
    assert_eq!(u64::from_le_bytes(read(data + 40).unwrap()), 1);
    f.call("nvnCommandBufferFinalize", &[buffer]);
    f.submit(handle);
    f.call("nvnQueueFenceSync", &[b + 0x200, b + 0xC40, 0, 1]);
    assert_eq!(f.call("nvnSyncWait", &[b + 0xC40, 0]), 2);
    f.call("nvnQueueFlush", &[b + 0x200]);
    assert!(f.call("nvnSyncWait", &[b + 0xC40, u64::MAX]) <= 1);
    let (renderer, surface, submissions) = {
        let mut driver = f.session.driver.lock().unwrap();
        let super::super::Driver {
            devices,
            textures,
            memory,
            queues,
            ..
        } = &mut *driver;
        (
            devices.renderer(b).unwrap(),
            textures
                .display_image(b + 0x880, memory, devices)
                .unwrap()
                .surface,
            queues.command_stats().0,
        )
    };
    assert_eq!(
        submissions, 2,
        "thirty clears share one submission, followed by one fence marker"
    );
    assert_eq!(surface.stats().readback_bytes, 0);
    assert_eq!(
        surface.stats().clears,
        31,
        "initial clear and thirty recorded GPU clears"
    );
    f.call("nvnQueuePresentTexture", &[b + 0x200, b + 0xA00, 0]);
    assert_eq!(
        surface.stats().readback_bytes,
        0,
        "headless presentation does not read pixels"
    );
    let rgba = renderer
        .lock()
        .unwrap()
        .read_native_surface(&surface)
        .unwrap();
    assert!(rgba
        .as_chunks::<4>()
        .0
        .iter()
        .all(|p| *p == [0, 255, 0, 255]));
    if let Some(dir) = std::env::var_os("KILN_NVN2_COMMAND_DUMP") {
        let path = std::path::PathBuf::from(dir).join("native_command_clear.png");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        nx_vulkan::png::write_png(&path, 32, 24, &rgba).unwrap();
    }
}

#[test]
fn command_sets_outlive_recorders_copy_independently_and_detect_memory_reuse() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let buffers = [b + 0x500, b + 0x600, b + 0x700];
    for (i, buffer) in buffers.into_iter().enumerate() {
        f.initialize(buffer);
        f.memory(
            buffer,
            0x8000 + i as u64 * 0x1000,
            0xD000 + i as u64 * 0x400,
        );
    }
    let source = f.record_barrier(buffers[0]);
    write(b + 0xD00, &source.to_le_bytes()).unwrap();
    let mut handles = [0; 2];
    for (i, name) in [
        "nvnCommandBufferCallCommands",
        "nvnCommandBufferCopyCommands",
    ]
    .into_iter()
    .enumerate()
    {
        let buffer = buffers[i + 1];
        f.call("nvnCommandBufferBeginRecording", &[buffer]);
        f.call(name, &[buffer, 1, b + 0xD00]);
        handles[i] = f.call("nvnCommandBufferEndRecording", &[buffer]);
    }
    f.call(
        "nvnCommandBufferAddCommandMemory",
        &[buffers[0], b + 0x380, 0x8000, 1024],
    );
    for handle in [source, handles[0]] {
        write(b + 0xD00, &handle.to_le_bytes()).unwrap();
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
    }
    for buffer in buffers {
        f.call("nvnCommandBufferFinalize", &[buffer]);
    }
    f.submit(handles[1]);
    assert_ne!(
        native_call(&f.session, "nvnMemoryPoolFinalize", &[b + 0x380])
            .ctl
            .stop,
        stop::NONE
    );
    f.call("nvnQueueFinish", &[b + 0x200]);
    f.call("nvnMemoryPoolFinalize", &[b + 0x380]);
    f.call("nvnQueueFinalize", &[b + 0x200]);
    f.call("nvnDeviceFinalize", &[b]);
}

#[test]
fn command_callbacks_propagate_unwinds_faults_and_insufficient_or_disabled_allocations() {
    let Some(f) = Fixture::new() else { return };
    let buffer = f.b() + 0x500;
    register_callbacks();
    f.initialize(buffer);
    f.call("nvnCommandBufferSetMemoryCallback", &[buffer, NO_MEMORY]);
    f.call(
        "nvnCommandBufferSetControlMemoryCallbackEnabled",
        &[buffer, 0],
    );
    assert_ne!(
        native_call(&f.session, "nvnCommandBufferBeginRecording", &[buffer])
            .ctl
            .stop,
        stop::NONE
    );
    f.call(
        "nvnCommandBufferSetControlMemoryCallbackEnabled",
        &[buffer, 1],
    );
    assert_ne!(
        native_call(&f.session, "nvnCommandBufferBeginRecording", &[buffer])
            .ctl
            .stop,
        stop::NONE
    );
    f.call("nvnCommandBufferSetMemoryCallback", &[buffer, UNWIND]);
    let cpu = native_call(&f.session, "nvnCommandBufferBeginRecording", &[buffer]);
    assert_eq!(
        (cpu.state.pc, cpu.state.x[0], cpu.ctl.stop),
        (0x800_2B00, 77, stop::NONE)
    );
    f.call("nvnCommandBufferSetMemoryCallback", &[buffer, FAULT]);
    let cpu = native_call(&f.session, "nvnCommandBufferBeginRecording", &[buffer]);
    assert_eq!((cpu.ctl.fault_flag, cpu.ctl.fault_addr), (1, 0xBAD0_0000));
    f.call("nvnCommandBufferFinalize", &[buffer]);
}

#[test]
fn inline_recording_headers_work_and_bad_submission_arrays_do_not_queue_a_prefix() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let buffer = b + 0x500;
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    // Synthetic inline BeginRecording bookkeeping; no guest-driver method words are replayed.
    write(buffer + 17, &[1, 0]).unwrap();
    write(buffer + 80, &(f.backing + 0xD000).to_le_bytes()).unwrap();
    write(buffer + 40, &(f.backing + 0xD010).to_le_bytes()).unwrap();
    assert_eq!(
        f.call("nvnCommandBufferGetControlMemoryUsed", &[buffer]),
        16,
        "memory queries observe an inlined begin before the first typed operation"
    );
    f.call("nvnCommandBufferBarrier", &[buffer, 1]);
    let handle = f.call("nvnCommandBufferEndRecording", &[buffer]);
    write(b + 0xFF8, &handle.to_le_bytes()).unwrap();
    let cpu = native_call(
        &f.session,
        "nvnQueueSubmitCommands",
        &[b + 0x200, 2, b + 0xFF8],
    );
    assert_eq!(cpu.ctl.fault_flag, 1);
    f.call("nvnQueueFinish", &[b + 0x200]);
    assert_eq!(
        f.session.driver.lock().unwrap().queues.command_stats().0,
        1,
        "a bad last handle did not leak the first command set into the queue"
    );
    f.call("nvnCommandBufferBeginRecording", &[buffer]);
    let cursor = u64::from_le_bytes(read(buffer).unwrap());
    write(buffer, &(cursor + 4).to_le_bytes()).unwrap();
    assert_ne!(
        native_call(&f.session, "nvnCommandBufferBarrier", &[buffer, 1])
            .ctl
            .stop,
        stop::NONE
    );
}

#[test]
fn overwritten_control_storage_cancels_queued_fences_before_any_gpu_submission() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let buffer = b + 0x500;
    f.initialize(buffer);
    f.memory(buffer, 0x8000, 0xD000);
    let handle = f.record_barrier(buffer);
    f.submit(handle);
    assert_eq!(f.call("nvnSyncInitialize", &[b + 0xC40, b]), 1);
    f.call("nvnQueueFenceSync", &[b + 0x200, b + 0xC40, 0, 1]);
    write(f.backing + 0xD000, &[0; 16]).unwrap();
    assert_ne!(
        native_call(&f.session, "nvnQueueFlush", &[b + 0x200])
            .ctl
            .stop,
        stop::NONE
    );
    assert_ne!(
        native_call(&f.session, "nvnSyncWait", &[b + 0xC40, 0])
            .ctl
            .stop,
        stop::NONE,
        "the invalid prefix cancels its fence instead of leaving a pending wait"
    );
    assert_eq!(f.session.driver.lock().unwrap().queues.command_stats().0, 0);
    f.call("nvnCommandBufferFinalize", &[buffer]);
    f.call("nvnSyncFinalize", &[b + 0xC40]);
    f.call("nvnMemoryPoolFinalize", &[b + 0x380]);
    f.call("nvnQueueFinalize", &[b + 0x200]);
}

#[test]
fn command_sets_cannot_be_submitted_to_another_native_device() {
    let Some(f) = Fixture::new() else { return };
    let b = f.b();
    let (second, builder, pool, buffer) = (b + 0x100, b + 0xE80, b + 0x800, b + 0x600);
    f.call("nvnDeviceBuilderSetDefaults", &[builder]);
    assert_eq!(f.call("nvnDeviceInitialize", &[second, builder]), 1);
    f.call("nvnMemoryPoolBuilderSetDefaults", &[builder]);
    f.call("nvnMemoryPoolBuilderSetDevice", &[builder, second]);
    f.call("nvnMemoryPoolBuilderSetFlags", &[builder, 0x22]);
    let second_backing = f.backing + 0x10000;
    rt().mem
        .map(second_backing, 0x10000, perm::R | perm::W)
        .unwrap();
    f.call(
        "nvnMemoryPoolBuilderSetStorage",
        &[builder, second_backing, 0x10000],
    );
    assert_eq!(f.call("nvnMemoryPoolInitialize", &[pool, builder]), 1);
    assert_eq!(f.call("nvnCommandBufferInitialize", &[buffer, second]), 1);
    f.call(
        "nvnCommandBufferAddCommandMemory",
        &[buffer, pool, 0x8000, 1024],
    );
    f.call(
        "nvnCommandBufferAddControlMemory",
        &[buffer, f.backing + 0xD000, 512],
    );
    let handle = f.record_barrier(buffer);
    write(b + 0xD00, &handle.to_le_bytes()).unwrap();
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
    assert_eq!(f.session.driver.lock().unwrap().queues.command_stats().0, 0);
    f.call("nvnCommandBufferFinalize", &[buffer]);
    f.call("nvnMemoryPoolFinalize", &[pool]);
    f.call("nvnDeviceFinalize", &[second]);
    rt().mem.unmap(second_backing, 0x10000).unwrap();
}
