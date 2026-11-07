//! Optional NVN2 interception (A-298), selected once before the guest starts.
//!
//! `trace` forwards every intercepted call to the guest driver. `hle` owns the whole boundary:
//! it never calls that driver, and a missing implementation stops the run with the API name.
//! The loader keeps its generated body so the default path is still the guest implementation.

use super::abi::{stop, NativeCpu};
use super::host::{c_slice, ModuleDesc};
use super::runtime::{record_fault, record_stop, rt, slow_path};
use crate::a64::bus::FaultKind;
use crate::horizon::kernel::{Kernel, ThreadCtx};
use crate::npdm::Platform;
use std::collections::BTreeMap;
use std::ffi::{c_void, CStr};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

#[cfg(feature = "vulkan")]
mod command;
mod descriptors;
mod device;
mod event;
mod memory;
#[cfg(feature = "vulkan")]
mod program;
mod queue;
mod sampler;
mod state;
mod texture;
mod window;

#[derive(Default)]
struct Driver {
    devices: device::Devices,
    memory: memory::Objects,
    descriptors: descriptors::Objects,
    textures: texture::Objects,
    events: event::Objects,
    #[cfg(feature = "vulkan")]
    programs: program::Objects,
    #[cfg(feature = "vulkan")]
    queues: queue::Objects,
    #[cfg(feature = "vulkan")]
    windows: window::Objects,
    #[cfg(feature = "vulkan")]
    commands: command::Objects,
}

impl Driver {
    fn call(&mut self, name: &str, cpu: &mut NativeCpu) -> Option<Result<(), device::Error>> {
        #[cfg(feature = "vulkan")]
        if let Some(error) = self.queues.background_error() {
            return Some(Err(device::Error::Invalid(error)));
        }
        if let Some(result) = event::builder_call(name, cpu) {
            return Some(result);
        }
        if let Some(result) = state::call(name, cpu) {
            return Some(result);
        }
        if name == "nvnDeviceFinalize"
            && (self.memory.has_children(cpu.state.x[0])
                || self.descriptors.has_device_children(cpu.state.x[0])
                || self.textures.has_device_children(cpu.state.x[0])
                || self.events.has_device(cpu.state.x[0]))
        {
            return Some(Err(device::Error::Invalid(
                "NVN2 device still owns native resources".to_string(),
            )));
        }
        #[cfg(feature = "vulkan")]
        if name == "nvnDeviceFinalize"
            && (self.queues.has_children(cpu.state.x[0])
                || self.windows.has_device_children(cpu.state.x[0])
                || self.commands.has_device_children(cpu.state.x[0])
                || self.programs.has_device_children(cpu.state.x[0]))
        {
            return Some(Err(device::Error::Invalid(
                "NVN2 device still owns queues, syncs, windows or programs".to_string(),
            )));
        }
        if let Some(result) =
            queue::builder_call(name, cpu, &|address| self.devices.identity(address))
        {
            return Some(result);
        }
        if let Some(result) =
            window::builder_call(name, cpu, &|address| self.devices.identity(address))
        {
            return Some(result);
        }
        #[cfg(feature = "vulkan")]
        if name == "nvnTextureFinalize" && self.windows.has_texture(cpu.state.x[0]) {
            return Some(Err(device::Error::Invalid(
                "NVN2 texture is owned by a window".to_string(),
            )));
        }
        #[cfg(feature = "vulkan")]
        if let Some(result) =
            self.windows
                .call(name, cpu, &self.devices, &mut self.textures, &self.memory)
        {
            return Some(result);
        }
        if let Some(result) =
            texture::builder_call(name, cpu, &|address| self.devices.identity(address))
        {
            return Some(result);
        }
        #[cfg(feature = "vulkan")]
        if let Some(result) = self.queues.call(
            name,
            cpu,
            &self.devices,
            &self.memory,
            &self.commands,
            Arc::clone(&self.events.visibility),
        ) {
            return Some(result);
        }
        if name == "nvnMemoryPoolFinalize"
            && (self.descriptors.has_pool_children(cpu.state.x[0])
                || self.textures.has_pool_children(cpu.state.x[0])
                || self.events.has_pool(cpu.state.x[0]))
        {
            return Some(Err(device::Error::Invalid(
                "NVN2 memory pool still owns descriptor pools or textures".to_string(),
            )));
        }
        #[cfg(feature = "vulkan")]
        if name == "nvnMemoryPoolFinalize"
            && (self.commands.has_pool_children(cpu.state.x[0])
                || self.queues.has_pool_children(cpu.state.x[0])
                || self.programs.has_pool_children(cpu.state.x[0]))
        {
            return Some(Err(device::Error::Invalid(
                "NVN2 memory pool still supplies commands or program binaries".to_string(),
            )));
        }
        #[cfg(feature = "vulkan")]
        if let Some(result) = self.programs.call(name, cpu, &self.devices, &self.memory) {
            return Some(result);
        }
        if let Some(result) = self.events.call(name, cpu, &self.memory, &self.devices) {
            return Some(result);
        }
        if let Some(result) = self.devices.call(name, cpu) {
            if name == "nvnDeviceFinalize" && result.is_ok() {
                self.memory.forget_device(cpu.state.x[0]);
                #[cfg(feature = "vulkan")]
                self.commands.retire_device(cpu.state.x[0]);
            }
            return Some(result);
        }
        if name == "nvnTexturePoolRegisterTexture" {
            return Some(self.descriptors.register_texture(
                cpu,
                &self.textures,
                &self.memory,
                &|address| self.devices.identity(address),
            ));
        }
        if let Some(result) = self.descriptors.call(name, cpu, &self.memory, &|address| {
            self.devices.identity(address)
        }) {
            return Some(result);
        }
        if let Some(result) = self.textures.call(name, cpu, &self.memory, &|address| {
            self.devices.identity(address)
        }) {
            return Some(result);
        }
        let address = cpu.state.x[0];
        let result = self
            .memory
            .call(name, cpu, &|address| self.devices.identity(address));
        if name == "nvnMemoryPoolFinalize" && matches!(result, Some(Ok(()))) {
            self.events.retire_pool(address);
            #[cfg(feature = "vulkan")]
            self.commands.retire_pool(address);
        }
        result
    }
}

/// The generated function ABI. A forward uses the same CPU, including FP arguments and stack.
pub type GuestFunction = extern "C" fn(*mut NativeCpu);

// Like EXIT_PC and STOP_PC these are dispatcher tokens, outside guest memory, never host pointers.
const PROC_BASE: u64 = 0xFFFF_FFFE_0000_0000;
const MAX_PROCS: usize = 65536;
const PROC_END: u64 = PROC_BASE + 4 * MAX_PROCS as u64;
const MAX_NAME: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Trace,
    Hle,
}

impl Mode {
    fn parse(value: &str) -> Result<Option<Mode>, String> {
        match value {
            "" | "0" | "off" => Ok(None),
            "trace" => Ok(Some(Mode::Trace)),
            "hle" => Ok(Some(Mode::Hle)),
            _ => Err("NX_NVN2 must be off, trace or hle".to_string()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Mode::Trace => "trace",
            Mode::Hle => "hle",
        }
    }
}

struct Proc {
    name: String,
    guest: u64,
    calls: AtomicU64,
}

#[derive(Default)]
struct Registry {
    procs: Vec<Arc<Proc>>,
    // Different devices or layers may resolve the same name to different guest entry points.
    by_name: BTreeMap<(String, u64), usize>,
}

pub(crate) struct Session {
    mode: Mode,
    registry: RwLock<Registry>,
    bootstraps: AtomicU64,
    /// The values the guest driver wrote for each DeviceGetInteger property in trace mode.
    integers: Mutex<BTreeMap<(u32, i32), u64>>,
    /// Matched, different, unavailable sampler descriptors observed after forwarded registration.
    sampler_checks: Mutex<[u64; 3]>,
    texture_checks: Mutex<BTreeMap<(u8, u32, u32), [u64; 3]>>,
    texture_details: Mutex<BTreeMap<(u8, u32, u32), String>>,
    driver: Mutex<Driver>,
}

impl Session {
    fn new(mode: Mode) -> Session {
        Session {
            mode,
            registry: RwLock::new(Registry::default()),
            bootstraps: AtomicU64::new(0),
            integers: Mutex::new(BTreeMap::new()),
            sampler_checks: Mutex::new([0; 3]),
            texture_checks: Mutex::new(BTreeMap::new()),
            texture_details: Mutex::new(BTreeMap::new()),
            driver: Mutex::new(Driver::default()),
        }
    }

    fn bind(&self, name: &str, guest: u64) -> Result<u64, String> {
        if self.mode == Mode::Trace && guest == 0 {
            return Ok(0);
        }
        let mut r = self.registry.write().unwrap_or_else(|e| e.into_inner());
        let key = (name.to_string(), guest);
        let id = if let Some(&id) = r.by_name.get(&key) {
            id
        } else {
            let id = r.procs.len();
            if id == MAX_PROCS {
                return Err("NVN2 procedure table is full".to_string());
            }
            r.procs.push(Arc::new(Proc {
                name: name.to_string(),
                guest,
                calls: AtomicU64::new(0),
            }));
            r.by_name.insert(key, id);
            id
        };
        Ok(PROC_BASE + 4 * id as u64)
    }

    fn proc(&self, pc: u64) -> Option<Arc<Proc>> {
        let id = proc_id(pc)?;
        self.registry
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .procs
            .get(id)
            .cloned()
    }

    fn bootstrap(&self, c: &mut NativeCpu, pc: u64, guest: GuestFunction) {
        self.bootstraps.fetch_add(1, Ordering::Relaxed);
        let name = read_name(c.state.x[0]);
        if self.mode == Mode::Trace {
            let lr = c.state.x[30];
            guest(c);
            if returned(c, lr) {
                match name {
                    Ok(name) => self.wrap_result(c, &name),
                    Err(_) if c.state.x[0] != 0 => {
                        fail(c, "cannot trace an unreadable NVN2 loader name")
                    }
                    Err(_) => {}
                }
            }
        } else {
            self.resolve(c, name, pc);
        }
    }

    fn wrap_result(&self, c: &mut NativeCpu, name: &str) {
        match self.bind(name, c.state.x[0]) {
            Ok(pc) => c.state.x[0] = pc,
            Err(e) => fail(c, &e),
        }
    }

    fn resolve(&self, c: &mut NativeCpu, name: Result<String, NameError>, pc: u64) {
        match name {
            Ok(name) if name.starts_with("nvn") && name.len() > 3 => {
                // Missing APIs get a named trap entry. They cannot return a made-up result or
                // fall back to a driver that owns different state.
                match self.bind(&name, 0) {
                    Ok(proc) => c.state.x[0] = proc,
                    Err(e) => {
                        fail(c, &e);
                        return;
                    }
                }
            }
            Ok(_) => c.state.x[0] = 0,
            Err(NameError::Fault(addr, kind)) => {
                record_fault(c, addr, pc, false, kind);
                return;
            }
            Err(NameError::Invalid) => {
                fail(c, "NVN2 procedure name is invalid or unterminated");
                return;
            }
        }
        c.state.pc = c.state.x[30];
    }

    fn call(&self, c: &mut NativeCpu) {
        let Some(proc) = self.proc(c.state.pc) else {
            fail(c, "unknown NVN2 procedure token");
            return;
        };
        proc.calls.fetch_add(1, Ordering::Relaxed);
        let caller = c.state.x[30].wrapping_sub(4);
        if self.mode == Mode::Trace {
            let name = (proc.name == "nvnDeviceGetProcAddress").then(|| read_name(c.state.x[1]));
            let integer =
                (proc.name == "nvnDeviceGetInteger").then_some((c.state.x[1] as u32, c.state.x[2]));
            let sampler = sampler::observe(&proc.name, c);
            let texture = texture::observe(&proc.name, c);
            let lr = c.state.x[30];
            c.state.pc = proc.guest;
            // SAFETY: this is the exclusively used CPU of the calling thread. The pointer came
            // from the guest loader; a missing translation uses the normal checked slow path.
            unsafe {
                if let Some(f) = rt().lookup(proc.guest) {
                    let f: GuestFunction = std::mem::transmute(f);
                    f(c);
                } else {
                    slow_path(c);
                }
            }
            if returned(c, lr) {
                if let Some(observation) = texture {
                    let key = observation.key;
                    let (index, detail) = match observation.compare(c) {
                        Ok(None) => (0, None),
                        Ok(Some(detail)) => (1, Some(detail)),
                        Err(e) => (2, Some(format!("unavailable: {e:?}"))),
                    };
                    if let Some(detail) = detail {
                        self.texture_details
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .entry(key)
                            .or_insert(detail);
                    }
                    self.texture_checks
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .entry(key)
                        .or_default()[index] += 1;
                }
                if let Some(observation) = sampler {
                    let index = match observation.and_then(|o| o.matches()) {
                        Ok(true) => 0,
                        Ok(false) => 1,
                        Err(_) => 2,
                    };
                    self.sampler_checks
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())[index] += 1;
                }
                if let Some((property, output)) = integer {
                    let mut bytes = [0; 4];
                    if rt().mem.read_bytes(output, &mut bytes, true).is_ok() {
                        *self
                            .integers
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .entry((property, i32::from_le_bytes(bytes)))
                            .or_default() += 1;
                    }
                }
                match name {
                    Some(Ok(name)) => self.wrap_result(c, &name),
                    Some(Err(_)) if c.state.x[0] != 0 => {
                        fail(c, "cannot trace an unreadable NVN2 procedure name")
                    }
                    Some(Err(_)) => {}
                    None => {}
                }
            }
            return;
        }
        match proc.name.as_str() {
            "nvnDeviceGetProcAddress" => {
                if c.state.x[0] != 0 {
                    let result = self
                        .driver
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .devices
                        .validate(c.state.x[0]);
                    if let Err(e) = result {
                        e.report(c, caller);
                        return;
                    }
                }
                self.resolve(c, read_name(c.state.x[1]), caller);
            }
            // These builder layouts are pinned by observation and tests (A-299).
            "nvnDeviceBuilderSetDefaults" => {
                if let Err(k) = rt().mem.write_bytes(c.state.x[0], &[0; 24], true) {
                    record_fault(c, c.state.x[0], caller, true, k);
                    return;
                }
                c.state.pc = c.state.x[30];
            }
            "nvnDeviceBuilderSetFlags" => {
                let addr = c.state.x[0].wrapping_add(4);
                if let Err(k) =
                    rt().mem
                        .write_bytes(addr, &(c.state.x[1] as u32).to_le_bytes(), true)
                {
                    record_fault(c, addr, caller, true, k);
                    return;
                }
                c.state.pc = c.state.x[30];
            }
            _ => {
                #[cfg(feature = "vulkan")]
                if texture::is_texel_call(&proc.name) {
                    let work = {
                        let mut driver = self.driver.lock().unwrap_or_else(|e| e.into_inner());
                        let Driver {
                            textures,
                            memory,
                            devices,
                            ..
                        } = &mut *driver;
                        textures.prepare_texels(&proc.name, c, memory, devices)
                    };
                    let result = work.and_then(|work| {
                        let ctx = c.ctl.user.cast::<ThreadCtx>();
                        // SAFETY: Horizon retains ThreadCtx for the call; tests use null.
                        if let Some(ctx) = unsafe { ctx.as_ref() } {
                            ctx.kernel
                                .wait_on_host(ctx.thread, &ctx.core, || work.run())
                        } else {
                            work.run()
                        }
                    });
                    match result {
                        Ok(()) => c.state.pc = c.state.x[30],
                        Err(e) => e.report(c, caller),
                    }
                    return;
                }
                #[cfg(feature = "vulkan")]
                if proc.name.starts_with("nvnCommandBuffer") {
                    for _ in 0..8 {
                        let step = {
                            let mut driver = self.driver.lock().unwrap_or_else(|e| e.into_inner());
                            let Driver {
                                commands,
                                devices,
                                memory,
                                textures,
                                programs,
                                events,
                                ..
                            } = &mut *driver;
                            commands
                                .call(
                                    &proc.name,
                                    c,
                                    &mut command::Resources {
                                        devices,
                                        memory,
                                        textures,
                                        programs,
                                        events,
                                    },
                                )
                                .expect("command entry")
                        };
                        match step {
                            Ok(command::Step::Done) => {
                                c.state.pc = c.state.x[30];
                                return;
                            }
                            Ok(command::Step::Callback(callback)) => {
                                let normal = callback.run(c);
                                let result = {
                                    let mut driver =
                                        self.driver.lock().unwrap_or_else(|e| e.into_inner());
                                    let Driver {
                                        commands, devices, ..
                                    } = &mut *driver;
                                    commands.finish_callback(&callback, devices, normal)
                                };
                                if !normal {
                                    return;
                                }
                                if let Err(error) = result {
                                    error.report(c, caller);
                                    return;
                                }
                            }
                            Err(error) => {
                                error.report(c, caller);
                                return;
                            }
                        }
                    }
                    fail(c, "NVN2 command allocation callbacks did not converge");
                    return;
                }
                #[cfg(feature = "vulkan")]
                if matches!(
                    proc.name.as_str(),
                    "nvnWindowAcquireTexture" | "nvnQueuePresentTexture"
                ) {
                    let work = {
                        let mut driver = self.driver.lock().unwrap_or_else(|e| e.into_inner());
                        let Driver {
                            windows,
                            queues,
                            devices,
                            memory,
                            ..
                        } = &mut *driver;
                        windows.prepare(&proc.name, c, devices, queues, memory)
                    };
                    let result = work.and_then(|work| {
                        let ctx = c.ctl.user.cast::<ThreadCtx>();
                        // SAFETY: Horizon calls retain ThreadCtx; synthetic calls use null.
                        let done = if let Some(ctx) = unsafe { ctx.as_ref() } {
                            ctx.kernel
                                .wait_on_host(ctx.thread, &ctx.core, || work.run(c))
                        } else {
                            work.run(c)
                        }?;
                        let mut driver = self.driver.lock().unwrap_or_else(|e| e.into_inner());
                        let Driver {
                            windows,
                            queues,
                            devices,
                            ..
                        } = &mut *driver;
                        windows.finish(work, done, c, devices, queues)
                    });
                    match result {
                        Ok(()) => c.state.pc = c.state.x[30],
                        Err(e) => e.report(c, caller),
                    }
                    return;
                }
                #[cfg(feature = "vulkan")]
                if matches!(
                    proc.name.as_str(),
                    "nvnSyncWait" | "nvnQueueFinish" | "nvnQueueFinalize"
                ) {
                    let wait = {
                        let mut driver = self.driver.lock().unwrap_or_else(|e| e.into_inner());
                        let Driver {
                            queues,
                            devices,
                            memory,
                            ..
                        } = &mut *driver;
                        queues.prepare_wait(&proc.name, c, devices, memory)
                    };
                    let result = wait
                        .and_then(|w| {
                            let ctx = c.ctl.user.cast::<ThreadCtx>();
                            // SAFETY: a Horizon thread keeps ThreadCtx alive for its whole run;
                            // synthetic calls leave a null user pointer.
                            if let Some(ctx) = unsafe { ctx.as_ref() } {
                                ctx.kernel.wait_on_host(ctx.thread, &ctx.core, || w.run(c))
                            } else {
                                w.run(c)
                            }
                        })
                        .and_then(|()| {
                            if proc.name == "nvnQueueFinalize" {
                                self.driver
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .call(&proc.name, c)
                                    .expect("queue finalization entry")
                            } else {
                                Ok(())
                            }
                        });
                    match result {
                        Ok(()) => c.state.pc = c.state.x[30],
                        Err(e) => e.report(c, caller),
                    }
                    return;
                }
                let result = self
                    .driver
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .call(&proc.name, c);
                match result {
                    Some(Ok(())) => c.state.pc = c.state.x[30],
                    Some(Err(e)) => e.report(c, caller),
                    None => fail(
                        c,
                        &format!(
                            "NVN2 HLE {} is not implemented (called at {caller:#x})",
                            proc.name
                        ),
                    ),
                }
            }
        }
    }

    fn report(&self) -> String {
        // A thread can still be leaving its final call while finish hooks run. Take each count
        // once so the sort has a stable order, and release the registry before formatting.
        let mut rows: Vec<_> = self
            .registry
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .procs
            .iter()
            .map(|p| (p.calls.load(Ordering::Relaxed), p.guest, p.name.clone()))
            .collect();
        rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
        let mut out = format!(
            "# NVN2 {}: {} loader calls, {} resolved procedures\n# calls guest name\n",
            self.mode.name(),
            self.bootstraps.load(Ordering::Relaxed),
            rows.len()
        );
        if self.mode == Mode::Hle {
            let driver = self.driver.lock().unwrap_or_else(|e| e.into_inner());
            #[cfg(feature = "vulkan")]
            let _ = writeln!(
                out,
                "# native windows owned: {}",
                driver.windows.live_count()
            );
            #[cfg(feature = "vulkan")]
            {
                let stats = driver.windows.stats();
                let _ = writeln!(out, "# native display GPU draws: {}", stats.draws);
                let _ = writeln!(out, "# native display GPU clears: {}\n# native display presentations: {}\n# native display readback bytes: {}",
                    stats.clears, stats.presents, stats.readback_bytes);
            }
            #[cfg(feature = "vulkan")]
            let _ = writeln!(
                out,
                "# native command sets recorded: {}",
                driver.commands.recordings
            );
            #[cfg(feature = "vulkan")]
            let _ = writeln!(
                out,
                "# native command buffers owned: {}",
                driver.commands.live_count()
            );
            #[cfg(feature = "vulkan")]
            {
                let (submissions, scratch) = driver.queues.command_stats();
                let _ = writeln!(out, "# native queue GPU submissions: {submissions}\n# native queue scratch binding bytes: {scratch}");
                let (bindings, sample) = driver.queues.binding_stats();
                let (copied, hits, uploaded) = driver.queues.input_stats();
                let (indices, index_hits, index_uploads, index_shared) =
                    driver.queues.index_stats();
                let _=writeln!(out,"# native index input bytes copied: {indices}\n# native index input cache hits: {index_hits}\n# native index GPU upload bytes: {index_uploads}\n# native shared index binding bytes: {index_shared}");
                let _ = writeln!(
                    out,
                    "# native shared vertex binding bytes: {}",
                    driver.queues.shared_input_bytes()
                );
                let _=writeln!(out,"# native vertex input bytes copied: {copied}\n# native vertex input cache hits: {hits}\n# native vertex GPU upload bytes: {uploaded}");
                let _ = writeln!(
                    out,
                    "# native queue state updates: {}",
                    driver.queues.state_updates()
                );
                let _ = writeln!(out, "# native queue shader stage bindings: {bindings}\n# native queue forced sample shading bindings: {sample}");
                let (programs, binaries, modules) = driver.programs.counts();
                let (resident, texels) = driver.textures.texel_stats();
                let _=writeln!(out,"# native resident textures: {resident}\n# native texel GPU uploads: {}\n# native texel GPU upload bytes: {}\n# native texel GPU reads: {}\n# native texel GPU readback bytes: {}",texels.uploads,texels.upload_bytes,texels.reads,texels.readback_bytes);
                let _ = writeln!(out, "# native programs owned: {programs}\n# native shader binaries owned: {binaries}\n# native shader modules translated: {modules}\n# native shader GPU input bytes captured: {}", driver.programs.captured_bytes);
            }
            let _ = writeln!(
                out,
                "# native textures owned: {}",
                driver.textures.live_count()
            );
            let _ = writeln!(out, "# native events owned: {}", driver.events.count());
        }
        for (count, guest, name) in rows {
            let _ = writeln!(out, "{count} {guest:#x} {name}");
        }
        for (&(property, value), &calls) in self
            .integers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            let _ = writeln!(out, "# integer {property:#x} {value} {calls}");
        }
        let checks = self
            .sampler_checks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(
            out,
            "# sampler descriptors: {} matched, {} different, {} unavailable",
            checks[0], checks[1], checks[2]
        );
        for (&(kind, flags, format), counts) in self
            .texture_checks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            let kind = [
                "size",
                "alignment",
                "descriptor-fields",
                "initialization-layout",
            ][usize::from(kind)];
            let _ = writeln!(out, "# texture {kind} flags={flags:#x} format={format}: {} matched, {} different, {} unavailable",
                counts[0], counts[1], counts[2]);
        }
        for (&(kind, flags, format), detail) in self
            .texture_details
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            let _ = writeln!(
                out,
                "# texture detail {kind} flags={flags:#x} format={format}: {detail}"
            );
        }
        out
    }
}

fn returned(c: &NativeCpu, lr: u64) -> bool {
    c.ctl.stop == stop::NONE && c.ctl.fault_flag == 0 && c.state.pc == lr
}

enum NameError {
    Fault(u64, FaultKind),
    Invalid,
}

fn read_name(addr: u64) -> Result<String, NameError> {
    let mut bytes = Vec::new();
    for offset in 0..MAX_NAME as u64 {
        let at = addr.checked_add(offset).ok_or(NameError::Invalid)?;
        let mut byte = [0];
        rt().mem
            .read_bytes(at, &mut byte, true)
            .map_err(|k| NameError::Fault(at, k))?;
        if byte[0] == 0 {
            return String::from_utf8(bytes).map_err(|_| NameError::Invalid);
        }
        if !byte[0].is_ascii_alphanumeric() && byte[0] != b'_' {
            return Err(NameError::Invalid);
        }
        bytes.push(byte[0]);
    }
    Err(NameError::Invalid)
}

fn proc_id(pc: u64) -> Option<usize> {
    ((PROC_BASE..PROC_END).contains(&pc) && pc.is_multiple_of(4))
        .then(|| ((pc - PROC_BASE) / 4) as usize)
}

/// Called only when no generated function has this address, so normal dispatch adds no table
/// probe, lock or range check. The session that owns the calling CPU resolves the token.
pub(super) fn lookup(pc: u64) -> Option<*const c_void> {
    proc_id(pc).map(|_| entry as *const c_void)
}

fn kernel(c: &NativeCpu) -> Option<Arc<Kernel>> {
    let ctx = c.ctl.user.cast::<ThreadCtx>();
    // SAFETY: Horizon threads keep a ThreadCtx here for their whole run; plain calls leave null.
    unsafe { ctx.as_ref() }.map(|ctx| Arc::clone(&ctx.kernel))
}

fn fail(c: &mut NativeCpu, message: &str) {
    let pc = c.state.x[30].wrapping_sub(4);
    record_stop(c, stop::UNIMPLEMENTED, pc, 0);
    if let Some(kernel) = kernel(c) {
        kernel.fatal(message.to_string());
    } else {
        eprintln!("kiln runtime: {message}");
    }
}

extern "C" fn entry(p: *mut NativeCpu) {
    // SAFETY: called through the native dispatcher with its exclusively used thread CPU.
    let c = unsafe { &mut *p };
    let session = kernel(c).and_then(|k| k.nvn2.get().cloned());
    if let Some(session) = session {
        session.call(c);
    } else {
        fail(c, "NVN2 procedure token without an enabled NVN2 session");
    }
}

pub(crate) fn bootstrap(c: &mut NativeCpu, pc: u64, guest: GuestFunction) {
    let session = kernel(c).and_then(|k| k.nvn2.get().cloned());
    if let Some(session) = session {
        session.bootstrap(c, pc, guest);
    } else {
        guest(c);
    }
}

/// Install one mode for the whole process before any thread runs.
///
/// # Safety
/// `m` must be a valid generated module descriptor with valid NUL-terminated export names.
pub(crate) unsafe fn start(kernel: &Arc<Kernel>, m: &ModuleDesc) -> Result<(), String> {
    let value = std::env::var("NX_NVN2").unwrap_or_default();
    let Some(mode) = Mode::parse(&value)? else {
        return Ok(());
    };
    if kernel.process.platform() != Platform::Nx2 {
        return Err("NX_NVN2 requires a Switch 2 program (Npdm::platform)".to_string());
    }
    if m.pad & super::host::MODULE_HAS_NVN2_BOOTSTRAP == 0 {
        return Err("NX_NVN2 needs a rebuilt loader stub; build once with --reuse-objects (relink alone is insufficient)".to_string());
    }
    if !c_slice(m.exports, m.nexports as usize)
        .iter()
        .any(|e| CStr::from_ptr(e.name) == c"nvnBootstrapLoader")
    {
        return Err("NX_NVN2 requires the nvnBootstrapLoader export".to_string());
    }
    let report_path = std::env::var_os("NX_NVN2_REPORT").map(PathBuf::from);
    if let Some(path) = &report_path {
        std::fs::write(path, format!("# NVN2 {}: running\n", mode.name()))
            .map_err(|e| format!("cannot open NVN2 report {}: {e}", path.display()))?;
    }
    let mut session = Session::new(mode);
    session
        .driver
        .get_mut()
        .unwrap_or_else(|e| e.into_inner())
        .devices
        .configure_from_env();
    let session = Arc::new(session);
    kernel
        .nvn2
        .set(Arc::clone(&session))
        .map_err(|_| "NVN2 was already started".to_string())?;
    kernel.on_finish(Box::new(move || {
        let report = session.report();
        eprintln!(
            "[nvn2] {}",
            report
                .lines()
                .next()
                .unwrap_or_default()
                .trim_start_matches("# ")
        );
        if let Some(path) = &report_path {
            if let Err(e) = std::fs::write(path, report) {
                eprintln!("[nvn2] cannot write {}: {e}", path.display());
            }
        }
    }));
    eprintln!("[nvn2] {} enabled", mode.name());
    Ok(())
}

#[cfg(test)]
mod tests;
