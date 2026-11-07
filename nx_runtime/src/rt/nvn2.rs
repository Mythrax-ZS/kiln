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

mod device;
mod memory;

#[derive(Default)]
struct Driver {
    devices: device::Devices,
    memory: memory::Objects,
}

impl Driver {
    fn call(&mut self, name: &str, cpu: &mut NativeCpu) -> Option<Result<(), device::Error>> {
        if name == "nvnDeviceFinalize" && self.memory.has_children(cpu.state.x[0]) {
            return Some(Err(device::Error::Invalid(
                "NVN2 device still owns memory pools".to_string(),
            )));
        }
        if let Some(result) = self.devices.call(name, cpu) {
            if name == "nvnDeviceFinalize" && result.is_ok() {
                self.memory.forget_device(cpu.state.x[0]);
            }
            return Some(result);
        }
        self.memory
            .call(name, cpu, &|address| self.devices.identity(address))
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
    driver: Mutex<Driver>,
}

impl Session {
    fn new(mode: Mode) -> Session {
        Session {
            mode,
            registry: RwLock::new(Registry::default()),
            bootstraps: AtomicU64::new(0),
            integers: Mutex::new(BTreeMap::new()),
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
