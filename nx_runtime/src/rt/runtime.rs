//! Process-wide runtime state and the operations generated code and the slow path share.

use super::abi::{self, stop, trap, FuncEntry, NativeCpu, EXIT_PC, STOP_PC};
use super::mem::GuestMemory;
use super::monitor::{Monitors, Reservation};
use crate::a64::bus::{Bus, Fault, FaultKind, COUNTER_FREQUENCY};
use crate::a64::inst::SysReg;
use crate::a64::interp::{step, Trap};
use crate::a64::state::CpuState;
use crate::slow::{self, SlowHost};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

/// Services `SVC #imm` for a thread. On entry `cpu.state.pc` is the instruction after the `SVC`.
/// Returns zero to continue, anything else to stop the run (the runtime records an unhandled
/// supervisor call unless the handler recorded a stop of its own).
pub type SvcHandler = extern "C" fn(*mut NativeCpu, u32) -> i32;

pub struct Runtime {
    pub mem: GuestMemory,
    pub monitors: Monitors,
    funcs: RwLock<Vec<FuncEntry>>,
    svc: Mutex<Option<SvcHandler>>,
    /// Threads whose every `n`th store-exclusive fails even with an intact reservation:
    /// `(n, attempts so far)`. Empty in normal runs.
    spurious: Mutex<BTreeMap<u64, (u64, u64)>>,
    spurious_any: AtomicBool,
    /// When non-zero, the system counter returns this value and adds one per read, so a test can
    /// replay a run and get the same readings.
    counter_fixed: AtomicU64,
    unknown: Mutex<BTreeSet<u64>>,
    next_thread: AtomicU64,
    start: Instant,
}

static RT: OnceLock<Runtime> = OnceLock::new();

pub fn rt() -> &'static Runtime {
    RT.get_or_init(Runtime::new)
}

impl Runtime {
    fn new() -> Runtime {
        Runtime {
            mem: GuestMemory::new(),
            monitors: Monitors::new(),
            funcs: RwLock::new(Vec::new()),
            svc: Mutex::new(None),
            spurious: Mutex::new(BTreeMap::new()),
            spurious_any: AtomicBool::new(false),
            counter_fixed: AtomicU64::new(0),
            unknown: Mutex::new(BTreeSet::new()),
            next_thread: AtomicU64::new(1),
            start: Instant::now(),
        }
    }

    pub fn new_thread_id(&self) -> u64 {
        self.next_thread.fetch_add(1, Ordering::Relaxed)
    }

    /// Add functions to the dispatch table; entries for an address already present are ignored.
    pub fn register(&self, entries: &[FuncEntry]) {
        let mut t = self.funcs.write().unwrap_or_else(|e| e.into_inner());
        t.extend_from_slice(entries);
        t.sort_by_key(|e| e.addr);
        t.dedup_by_key(|e| e.addr);
    }

    pub fn lookup(&self, pc: u64) -> Option<*const c_void> {
        let t = self.funcs.read().unwrap_or_else(|e| e.into_inner());
        t.binary_search_by_key(&pc, |e| e.addr)
            .ok()
            .map(|i| t[i].func)
    }

    pub fn set_svc_handler(&self, h: Option<SvcHandler>) {
        *self.svc.lock().unwrap() = h;
    }

    pub fn svc_handler(&self) -> Option<SvcHandler> {
        *self.svc.lock().unwrap()
    }

    /// Make every `n`th store-exclusive of `thread` fail (0 = never).
    pub fn set_spurious_period(&self, thread: u64, n: u64) {
        let mut m = self.spurious.lock().unwrap();
        if n == 0 {
            m.remove(&thread);
        } else {
            m.insert(thread, (n, 0));
        }
        self.spurious_any.store(!m.is_empty(), Ordering::Relaxed);
    }

    fn take_spurious(&self, thread: u64) -> bool {
        if !self.spurious_any.load(Ordering::Relaxed) {
            return false;
        }
        let mut m = self.spurious.lock().unwrap();
        match m.get_mut(&thread) {
            Some((n, count)) => {
                *count += 1;
                *count % *n == 0
            }
            None => false,
        }
    }

    /// Record that control went to an address no function covers. The first sight of an address
    /// is appended to the file named by `NX_UNKNOWN_LOG`, one `0x...` per line, which can be fed
    /// back to the recompiler as a hints file.
    pub fn note_unknown(&self, pc: u64) {
        let fresh = self.unknown.lock().unwrap().insert(pc);
        if fresh {
            if let Ok(path) = std::env::var("NX_UNKNOWN_LOG") {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(f, "{pc:#x}");
                }
            }
        }
    }

    pub fn unknown_targets(&self) -> Vec<u64> {
        self.unknown.lock().unwrap().iter().copied().collect()
    }

    /// Make the counter deterministic, starting at `start` (0 restores the real counter).
    pub fn set_counter_fixed(&self, start: u64) {
        self.counter_fixed.store(start, Ordering::Relaxed);
    }

    /// The free-running system counter at `COUNTER_FREQUENCY`.
    pub fn counter(&self) -> u64 {
        if self.counter_fixed.load(Ordering::Relaxed) != 0 {
            return self.counter_fixed.fetch_add(1, Ordering::Relaxed);
        }
        let ns = self.start.elapsed().as_nanos();
        (ns * u128::from(COUNTER_FREQUENCY) / 1_000_000_000) as u64
    }

    // ---- memory with the exclusive monitor ----

    /// A plain store. When some thread holds a reservation the store runs under the monitor lock,
    /// so it cannot slip between a store-exclusive's check and its write.
    pub fn store(
        &self,
        thread: u64,
        addr: u64,
        size: u8,
        value: u128,
        order: Ordering,
    ) -> Result<(), FaultKind> {
        if self.monitors.any_armed() {
            let mut g = self.monitors.lock();
            self.mem.write(addr, size, value, order)?;
            g.note_store(thread, addr, size, value);
            return Ok(());
        }
        self.mem.write(addr, size, value, order)
    }

    /// A compare-and-exchange by `thread` (the kernel's atomic updates of guest words): like a
    /// plain store, a successful exchange ends other threads' reservations on the location.
    pub fn compare_exchange(
        &self,
        thread: u64,
        addr: u64,
        size: u8,
        expected: u64,
        new: u64,
        order: Ordering,
    ) -> Result<bool, FaultKind> {
        if self.monitors.any_armed() {
            let mut g = self.monitors.lock();
            let done = self
                .mem
                .compare_exchange(addr, size, expected, new, order)?;
            if done {
                g.note_store(thread, addr, size, u128::from(new));
            }
            return Ok(done);
        }
        self.mem.compare_exchange(addr, size, expected, new, order)
    }

    /// `LDXR`: read, then record the reservation.
    pub fn load_exclusive(
        &self,
        thread: u64,
        st: &mut CpuState,
        addr: u64,
        bytes: u8,
        acquire: bool,
    ) -> Result<u128, FaultKind> {
        let order = if acquire {
            Ordering::Acquire
        } else {
            Ordering::Relaxed
        };
        let mut g = self.monitors.lock();
        let v = self.mem.read(addr, bytes, order)?;
        g.arm(
            thread,
            Reservation {
                addr,
                bytes,
                observed: [v as u64, (v >> 64) as u64],
            },
        );
        st.excl_addr = addr;
        st.excl_size = u64::from(bytes);
        Ok(v)
    }

    /// `STXR`: store if the reservation for exactly this range is intact. The reservation ends
    /// either way. A failed check does not touch memory, so it cannot fault.
    pub fn store_exclusive(
        &self,
        thread: u64,
        st: &mut CpuState,
        addr: u64,
        bytes: u8,
        value: u128,
        release: bool,
    ) -> Result<bool, FaultKind> {
        let order = if release {
            Ordering::AcqRel
        } else {
            Ordering::Relaxed
        };
        let mut g = self.monitors.lock();
        let mine = g.get(thread);
        let armed = st.excl_size == u64::from(bytes) && st.excl_addr == addr;
        g.disarm(thread);
        st.excl_size = 0;
        let Some(res) = mine.filter(|r| armed && r.addr == addr && r.bytes == bytes) else {
            return Ok(false);
        };
        if self.take_spurious(thread) {
            return Ok(false);
        }
        let done = if bytes <= 8 {
            self.mem
                .compare_exchange(addr, bytes, res.observed[0], value as u64, order)?
        } else {
            let seen = u128::from(res.observed[0]) | (u128::from(res.observed[1]) << 64);
            if self.mem.read(addr, bytes, Ordering::Relaxed)? != seen {
                false
            } else {
                self.mem.write(addr, bytes, value, order)?;
                true
            }
        };
        if done {
            g.note_store(thread, addr, bytes, value);
        }
        Ok(done)
    }

    pub fn clear_exclusive(&self, thread: u64, st: &mut CpuState) {
        st.excl_size = 0;
        if self.monitors.any_armed() {
            self.monitors.lock().disarm(thread);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stops
// ---------------------------------------------------------------------------------------------

pub fn record_fault(cpu: &mut NativeCpu, addr: u64, pc: u64, write: bool, kind: FaultKind) {
    let c = &mut cpu.ctl;
    c.fault_flag = 1;
    c.stop = stop::FAULT;
    c.fault_kind = match kind {
        FaultKind::Unmapped => 0,
        FaultKind::Permission => 1,
        FaultKind::Alignment => 2,
    };
    c.fault_write = u32::from(write);
    c.fault_addr = addr;
    c.stop_pc = pc;
    cpu.state.pc = STOP_PC;
}

pub fn record_trap(cpu: &mut NativeCpu, kind: u32, pc: u64, aux0: u64, aux1: u64) {
    let c = &mut cpu.ctl;
    c.stop = stop::TRAP;
    c.trap_kind = kind;
    c.trap_aux = [aux0, aux1];
    c.stop_pc = pc;
    cpu.state.pc = STOP_PC;
}

pub fn record_stop(cpu: &mut NativeCpu, reason: u32, pc: u64, aux0: u64) {
    let c = &mut cpu.ctl;
    c.stop = reason;
    c.stop_pc = pc;
    c.trap_aux = [aux0, 0];
    cpu.state.pc = STOP_PC;
}

/// `op0:op1:CRn:CRm:op2` packed the way [`trap::SYSREG`] documents.
pub fn pack_sysreg(reg: SysReg) -> u64 {
    let [a, b, c, d, e] = reg.encoding();
    (u64::from(a) << 14)
        | (u64::from(b) << 11)
        | (u64::from(c) << 7)
        | (u64::from(d) << 3)
        | u64::from(e)
}

/// Turn an interpreter trap into a recorded stop. `SVC` is serviced; the result says whether
/// execution can go on.
pub fn handle_trap(cpu: &mut NativeCpu, t: Trap) -> bool {
    let pc = cpu.state.pc;
    match t {
        Trap::Svc(imm) => return service_svc(cpu, u32::from(imm)),
        Trap::Brk(imm) => record_trap(cpu, trap::BRK, pc, u64::from(imm), 0),
        Trap::Undefined(w) => record_trap(cpu, trap::UNDEFINED, pc, u64::from(w), 0),
        Trap::Unsupported { word, .. } => {
            record_trap(cpu, trap::UNSUPPORTED, pc, u64::from(word), 0)
        }
        Trap::Fault(f) => record_fault(cpu, f.addr, pc, f.write, f.kind),
        Trap::SysReg { reg, write } => {
            record_trap(cpu, trap::SYSREG, pc, pack_sysreg(reg), u64::from(write))
        }
        Trap::SysInst {
            load,
            op1,
            crn,
            crm,
            op2,
        } => {
            let packed = (u64::from(op1) << 11)
                | (u64::from(crn) << 7)
                | (u64::from(crm) << 3)
                | u64::from(op2);
            record_trap(cpu, trap::SYSINST, pc, packed, u64::from(load));
        }
    }
    false
}

/// Run the handler for `SVC #imm`; `cpu.state.pc` is the following instruction.
pub fn service_svc(cpu: &mut NativeCpu, imm: u32) -> bool {
    let after = cpu.state.pc;
    let stop_run = match rt().svc_handler() {
        Some(h) => h(cpu, imm) != 0,
        None => true,
    };
    if stop_run {
        if cpu.ctl.stop == stop::NONE {
            record_stop(cpu, stop::SVC, after.wrapping_sub(4), u64::from(imm));
        }
        return false;
    }
    cpu.ctl.stop == stop::NONE
}

// ---------------------------------------------------------------------------------------------
// The interpreter's view of the runtime
// ---------------------------------------------------------------------------------------------

/// Memory and the monitor for one guest thread, as the interpreter needs them.
pub struct NativeBus {
    pub thread: u64,
}

fn to_fault(addr: u64, write: bool, kind: FaultKind) -> Fault {
    Fault { addr, write, kind }
}

impl Bus for NativeBus {
    fn read(&mut self, addr: u64, size: u8) -> Result<u128, Fault> {
        rt().mem
            .read(addr, size, Ordering::Relaxed)
            .map_err(|k| to_fault(addr, false, k))
    }

    fn write(&mut self, addr: u64, size: u8, value: u128) -> Result<(), Fault> {
        rt().store(self.thread, addr, size, value, Ordering::Relaxed)
            .map_err(|k| to_fault(addr, true, k))
    }

    fn load_exclusive(&mut self, cpu: &mut CpuState, addr: u64, size: u8) -> Result<u128, Fault> {
        rt().load_exclusive(self.thread, cpu, addr, size, false)
            .map_err(|k| to_fault(addr, false, k))
    }

    fn store_exclusive(
        &mut self,
        cpu: &mut CpuState,
        addr: u64,
        size: u8,
        value: u128,
    ) -> Result<bool, Fault> {
        rt().store_exclusive(self.thread, cpu, addr, size, value, false)
            .map_err(|k| to_fault(addr, true, k))
    }

    fn clear_exclusive(&mut self, cpu: &mut CpuState) {
        rt().clear_exclusive(self.thread, cpu);
    }

    fn counter(&mut self) -> u64 {
        rt().counter()
    }
}

/// Call the generated function registered for `cpu.state.pc`.
///
/// # Safety
/// `cpu` must be valid and a function must be registered at its `pc`.
unsafe fn call_function(cpu: *mut NativeCpu, f: *const c_void) {
    let f: extern "C" fn(*mut NativeCpu) = std::mem::transmute(f);
    f(cpu);
}

struct Host {
    cpu: *mut NativeCpu,
}

impl SlowHost for Host {
    fn has_native(&mut self, addr: u64) -> bool {
        rt().lookup(addr).is_some()
    }

    fn call_native(&mut self, _state: &mut CpuState, _bus: &mut dyn Bus) -> bool {
        // SAFETY: `cpu` outlives the slow path; the interpreter's `&mut CpuState` is its first
        // field and is not used while the native function runs.
        unsafe {
            let pc = (*self.cpu).state.pc;
            let f = rt().lookup(pc).expect("has_native said so");
            call_function(self.cpu, f);
            (*self.cpu).ctl.stop == stop::NONE
        }
    }

    fn svc(&mut self, _state: &mut CpuState, _bus: &mut dyn Bus, imm: u16) -> bool {
        // SAFETY: as above.
        service_svc(unsafe { &mut *self.cpu }, u32::from(imm))
    }
}

/// Run guest code that has no native translation, from `cpu.state.pc`, until it returns to its
/// native caller or something stops the run.
///
/// # Safety
/// `cpu` must be valid for the duration of the call.
pub unsafe fn slow_path(cpu: *mut NativeCpu) {
    let thread = (*cpu).ctl.thread_id;
    rt().note_unknown((*cpu).state.pc);
    let mut bus = NativeBus { thread };
    let mut host = Host { cpu };
    let state: *mut CpuState = &mut (*cpu).state;
    match slow::run(&mut *state, &mut bus, &mut host, u64::MAX) {
        slow::Exit::Returned => {}
        slow::Exit::Halted => {
            if (*cpu).ctl.stop == stop::NONE {
                record_stop(&mut *cpu, stop::HALTED, (*cpu).state.pc, 0);
            }
            (*cpu).state.pc = STOP_PC;
        }
        slow::Exit::Trap(t) => {
            // An instruction fetch that fails means there is no code to run here.
            if let Trap::Fault(f) = t {
                if f.kind == FaultKind::Unmapped && f.addr == (*cpu).state.pc {
                    record_stop(&mut *cpu, stop::NO_CODE, f.addr, 0);
                    return;
                }
            }
            handle_trap(&mut *cpu, t);
        }
    }
}

/// Execute one instruction in the interpreter at `cpu.state.pc` (the lifter's fallback for
/// instructions it has no translation for).
///
/// # Safety
/// `cpu` must be valid.
pub unsafe fn interpret_one(cpu: *mut NativeCpu) {
    let thread = (*cpu).ctl.thread_id;
    let mut bus = NativeBus { thread };
    let state: *mut CpuState = &mut (*cpu).state;
    if let Err(t) = step(&mut *state, &mut bus) {
        handle_trap(&mut *cpu, t);
    }
}

/// Run a thread from `entry` until the entry function returns (returns 0) or something stops the
/// run (returns the [`stop`] reason).
///
/// # Safety
/// `cpu` must be valid and exclusively used by the caller.
pub unsafe fn run(cpu: *mut NativeCpu, entry: u64) -> u32 {
    (*cpu).state.pc = entry;
    (*cpu).state.x[30] = EXIT_PC;
    (*cpu).ctl.stop = stop::NONE;
    (*cpu).ctl.fault_flag = 0;
    loop {
        if (*cpu).ctl.stop != stop::NONE {
            return (*cpu).ctl.stop;
        }
        let pc = (*cpu).state.pc;
        if pc == EXIT_PC {
            return 0;
        }
        match rt().lookup(pc) {
            Some(f) => call_function(cpu, f),
            None => slow_path(cpu),
        }
    }
}

/// Used by tests and the CLI to see what [`abi`] promises.
pub fn exit_pc() -> u64 {
    abi::EXIT_PC
}

/// Run a thread in the interpreter only, from `entry` until the entry function returns (returns
/// 0), something stops the run (returns the [`stop`] reason), or `max_steps` instructions have
/// executed (returns [`stop::HALTED`]). The reference the native code is compared against.
///
/// # Safety
/// `cpu` must be valid and exclusively used by the caller.
pub unsafe fn interpret(cpu: *mut NativeCpu, entry: u64, max_steps: u64) -> u32 {
    (*cpu).state.pc = entry;
    (*cpu).state.x[30] = EXIT_PC;
    (*cpu).ctl.stop = stop::NONE;
    (*cpu).ctl.fault_flag = 0;
    let thread = (*cpu).ctl.thread_id;
    let mut bus = NativeBus { thread };
    let state: *mut CpuState = &mut (*cpu).state;
    for _ in 0..max_steps {
        if (*state).pc == EXIT_PC {
            return 0;
        }
        if let Err(t) = step(&mut *state, &mut bus) {
            // A failed instruction fetch means there is no code there, as in the slow path.
            if let Trap::Fault(f) = t {
                if f.kind == FaultKind::Unmapped && !f.write && f.addr == (*state).pc {
                    record_stop(&mut *cpu, stop::NO_CODE, f.addr, 0);
                    return stop::NO_CODE;
                }
            }
            if !handle_trap(&mut *cpu, t) {
                return (*cpu).ctl.stop;
            }
        }
    }
    if (*state).pc == EXIT_PC {
        return 0;
    }
    record_stop(&mut *cpu, stop::HALTED, (*state).pc, 0);
    stop::HALTED
}
