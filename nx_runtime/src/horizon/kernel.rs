//! The kernel: process state, threads, events, and the blocking primitives.
//!
//! All shared state lives behind one lock with one condition variable. A blocked thread re-checks
//! its own condition whenever anything changes, which is simple and correct; it costs a wake-up
//! per change per waiter, which does not matter at this scale. Guest memory is touched through the
//! runtime's checked interface, never directly.

use super::cores::Cores;
use super::handle::{Handle, HandleTable, CURRENT_PROCESS, CURRENT_THREAD};
use super::layout::{attr, mem_type, PAGE, TLS_BASE, TLS_SLOT_SIZE};
use super::memory::{Info, MemoryMap};
use super::object::KObject;
use super::result::{kernel as err, KResult, ResultCode};
use super::service::ServiceFactory;
use super::watchdog::ThreadMonitor;
use crate::npdm::Npdm;
use crate::rt::abi::{perm, stop, NativeCpu, EXIT_PC};
use crate::rt::host::describe_stop;
use crate::rt::runtime::{rt, run};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The id the runtime's exclusive monitors see for the kernel's own stores to guest memory.
pub const KERNEL_THREAD_ID: u64 = u64::MAX;

/// Bit 30 of a guest mutex word: "the owner must call the kernel to release".
pub const MUTEX_WAIT_MASK: u32 = 0x4000_0000;

/// Native stack size of each guest thread's host thread. Generated code runs on it and uses it
/// the way the guest uses its own stack, plus the spilled state of each call; it is address space,
/// committed as it is touched.
const GUEST_STACK_BYTES: usize = 512 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessKind {
    /// An NSO started by the loader: `X0` is zero and `X1` the main thread handle.
    Nso,
    /// A homebrew NRO: `X0` points at the loader configuration, `X1` is `u64::MAX`.
    Nro,
}

/// What the kernel knows about the process it runs.
pub struct ProcessInfo {
    pub name: String,
    pub kind: ProcessKind,
    pub process_id: u64,
    pub program_id: u64,
    pub npdm: Option<Npdm>,
    /// The RomFS the program reads: a directory or an image (`NX_ROMFS` or the build's `--romfs`).
    pub romfs: Option<std::path::PathBuf>,
}

impl ProcessInfo {
    pub fn new(name: &str, kind: ProcessKind, npdm: Option<Npdm>) -> ProcessInfo {
        ProcessInfo {
            name: name.to_string(),
            kind,
            process_id: 0x80,
            program_id: npdm.as_ref().map_or(0, |n| n.program_id),
            npdm,
            romfs: None,
        }
    }
}

/// How the process ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exit {
    /// `ExitProcess`, or the main thread returned: the status.
    Code(i32),
    /// The run stopped: the report.
    Fatal(String),
}

pub(super) struct Thread {
    pub priority: i32,
    pub ideal_core: i32,
    pub core_mask: u64,
    pub tls: u64,
    pub entry: u64,
    pub arg: u64,
    pub stack_top: u64,
    pub started: bool,
    pub finished: bool,
    pub is_main: bool,
    /// `CancelSynchronization` was called and the thread has not seen it yet.
    pub cancel_sync: bool,
    /// Set by whoever ends the thread's wait on a mutex, condition variable or address.
    pub wake: Option<ResultCode>,
}

impl Thread {
    /// The core the thread is on, as `GetCurrentProcessorNumber` tells it: its ideal core when the
    /// core mask allows that core, otherwise the lowest core the mask names (docs/ASSUMPTIONS.md
    /// A-149).
    pub fn running_core(&self) -> i32 {
        let mask = self.core_mask & 0xF;
        if (0..4).contains(&self.ideal_core) && mask & (1 << self.ideal_core) != 0 {
            self.ideal_core
        } else if mask != 0 {
            mask.trailing_zeros() as i32
        } else {
            0
        }
    }
}

pub(super) struct Event {
    pub signaled: bool,
}

#[derive(Clone, Copy)]
pub(super) struct Waiter {
    pub thread: u64,
    pub tag: u32,
    pub priority: i32,
    pub seq: u64,
}

#[derive(Clone, Copy)]
pub(super) struct CvWaiter {
    pub waiter: Waiter,
    /// The mutex the waiter released and takes back.
    pub mutex: u64,
}

pub(super) struct State {
    pub handles: HandleTable<KObject>,
    pub memory: MemoryMap,
    pub heap_size: u64,
    pub threads: BTreeMap<u64, Thread>,
    pub events: Vec<Event>,
    pub tls_free: Vec<u64>,
    pub tls_pages: u64,
    pub mutex_waiters: HashMap<u64, Vec<Waiter>>,
    pub cv_waiters: HashMap<u64, Vec<CvWaiter>>,
    pub arbiter_waiters: HashMap<u64, Vec<Waiter>>,
    pub exit: Option<Exit>,
    /// The report of a stop is being printed: the process does not end before it is over.
    pub reporting: bool,
    pub seq: u64,
    pub main_handle: Handle,
    /// Regions this kernel mapped into guest memory, unmapped again when it is dropped.
    pub mapped: Vec<(u64, u64)>,
    /// Ranges lent to services with `CreateTransferMemory`, by object index.
    pub transfer_memories: Vec<super::memobj::TransferMemory>,
    pub(super) shared: Vec<super::shmem::SharedBlock>,
    /// The next free address of the kernel's own pages (`KERNEL_SCRATCH_BASE` and up).
    pub(super) kernel_cursor: u64,
}

/// What each guest thread's `ctl.user` points at, so an SVC finds its kernel.
pub struct ThreadCtx {
    pub kernel: Arc<Kernel>,
    pub thread: u64,
    /// What the thread is doing, as the watchdog sees it.
    pub monitor: Arc<ThreadMonitor>,
    /// The core the thread holds while it runs guest code (`NX_CORES`).
    pub core: AtomicUsize,
}

pub struct Kernel {
    pub process: ProcessInfo,
    pub(super) state: Mutex<State>,
    wake: Condvar,
    registry: Mutex<BTreeMap<String, ServiceFactory>>,
    pub(super) output: Mutex<Box<dyn Write + Send>>,
    pub(super) random: Mutex<u64>,
    /// Log every supervisor call (`NX_TRACE_SVC`).
    pub(super) trace: bool,
    /// Log the supervisor calls that block and took at least this long, with the thread and the
    /// time (`NX_TRACE_WAITS=<microseconds>`: where the threads of a program wait for each other).
    pub(super) trace_waits: Option<std::time::Duration>,
    /// When the kernel was made, for the times `NX_TRACE_WAITS` prints.
    pub(super) started: std::time::Instant,
    /// Print what the program logs through `lm` (`NX_TRACE_LOG`).
    trace_log: bool,
    /// Things to do once the process has ended (services print their summaries).
    finish_hooks: Mutex<Vec<Box<dyn Fn() + Send + Sync>>>,
    /// What each running guest thread is doing, for the watchdog (`NX_WATCHDOG`).
    monitors: Mutex<BTreeMap<u64, Arc<ThreadMonitor>>>,
    /// A watchdog is running (`NX_WATCHDOG`): calls say what they are about.
    watched: bool,
    /// Threads hold a core to run guest code (`NX_CORES`).
    cores: Option<Cores>,
    /// `NX_JITTER=<percent>[,<microseconds>]`: that share of the supervisor calls first sleeps for
    /// up to that long (500 by default), to stir the threads' timing when looking for a race.
    jitter: Option<(u64, u64)>,
    jitter_state: std::sync::atomic::AtomicU64,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        let st = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        for &(addr, size) in &st.mapped {
            let _ = rt().mem.unmap(addr, size);
        }
    }
}

/// The kernel no longer owns the pages `lo..hi` (it unmapped them): cut them out of the ranges it
/// unmaps when it is dropped, keeping what is left of a range that only partly overlaps.
fn forget_mapped(st: &mut State, lo: u64, hi: u64) {
    let mut kept = Vec::new();
    for &(addr, size) in &st.mapped {
        let end = addr + size;
        if end <= lo || addr >= hi {
            kept.push((addr, size));
            continue;
        }
        if addr < lo {
            kept.push((addr, lo - addr));
        }
        if end > hi {
            kept.push((hi, end - hi));
        }
    }
    st.mapped = kept;
}

/// How long a wait lasts: `None` is forever.
pub fn deadline_of(timeout_ns: i64) -> Option<Instant> {
    if timeout_ns < 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_nanos(timeout_ns as u64))
    }
}

pub(super) enum Waited<R> {
    Ready(R),
    TimedOut,
    Terminating,
}

/// For reading a crash: 256 bytes at each distinct register value of a stopped thread that is a
/// mapped guest address outside its stack, as words eight to a line, and 64 bytes at what the words of
/// the first few of them point at in the heap (the blocks of a heap's lists, say).
fn dump_pointees(cpu: &NativeCpu) {
    // The words of `len` bytes at `addr`, eight to a line, or None when they are not readable.
    fn words(addr: u64, len: usize) -> Option<Vec<u64>> {
        let mut bytes = vec![0u8; len];
        rt().mem.read_bytes(addr, &mut bytes, false).ok()?;
        Some(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|w| u64::from_le_bytes(*w))
                .collect(),
        )
    }
    fn print(label: &str, ws: &[u64]) {
        eprintln!("[stop]   {label}:");
        for (n, line) in ws.chunks(8).enumerate() {
            let text: Vec<String> = line.iter().map(|w| format!("{w:016x}")).collect();
            eprintln!("[stop]     +{:#05x}: {}", n * 64, text.join(" "));
        }
    }
    let sp = cpu.state.sp;
    let mut seen: Vec<u64> = Vec::new();
    let mut objects: Vec<(u64, Vec<u64>)> = Vec::new();
    for (i, &a) in cpu.state.x.iter().enumerate().take(29) {
        if a < 0x1_0000 || seen.contains(&a) || a.abs_diff(sp) < 0x4_0000 || seen.len() >= 10 {
            continue;
        }
        let Some(ws) = words(a, 256) else { continue };
        seen.push(a);
        print(&format!("x{i} = {a:#x} points at"), &ws);
        objects.push((a, ws));
    }
    // What the first few of those objects point at in the heap: the blocks of a heap's lists, say.
    let heap = 0x30_0000_0000u64..0x40_0000_0000;
    let mut followed = 0;
    for (a, ws) in objects.iter().take(4) {
        for (n, &w) in ws.iter().enumerate() {
            if followed >= 16
                || !heap.contains(&w)
                || seen.contains(&w)
                || w.abs_diff(sp) < 0x4_0000
            {
                continue;
            }
            if let Some(inner) = words(w, 64) {
                seen.push(w);
                followed += 1;
                print(
                    &format!("{w:#x}, the word at {a:#x}+{:#x}, points at", 8 * n),
                    &inner,
                );
            }
        }
    }
}

/// `NX_JITTER`: the share of supervisor calls to delay and the longest delay in microseconds.
fn jitter_from_env() -> Option<(u64, u64)> {
    let v = std::env::var("NX_JITTER").ok()?;
    let mut parts = v.split(',');
    let percent: u64 = parts.next()?.trim().parse().ok()?;
    let micros: u64 = parts
        .next()
        .and_then(|m| m.trim().parse().ok())
        .unwrap_or(500);
    (percent > 0 && micros > 0).then_some((percent.min(100), micros))
}

/// The seed for the kernel's pseudo-random values: `NX_RANDOM_SEED` when set, otherwise the clock.
fn random_seed() -> u64 {
    if let Some(v) = std::env::var("NX_RANDOM_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return v | 1;
    }
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64);
    t | 1
}

impl State {
    pub fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    pub fn thread_of(&self, h: Handle, current: u64) -> Option<u64> {
        if h == CURRENT_THREAD {
            return Some(current);
        }
        match self.handles.get(h)? {
            KObject::Thread(id) => Some(*id),
            _ => None,
        }
    }

    /// Map a TLS page when none is free and hand out a slot (zeroed).
    fn alloc_tls(&mut self) -> KResult<u64> {
        if self.tls_free.is_empty() {
            let page = TLS_BASE + self.tls_pages * PAGE;
            rt().mem
                .map(page, PAGE, perm::R | perm::W)
                .map_err(|_| err::OUT_OF_MEMORY)?;
            self.mapped.push((page, PAGE));
            self.memory.insert(
                page,
                PAGE,
                Info {
                    mem_type: mem_type::THREAD_LOCAL,
                    perm: perm::R | perm::W,
                    attr: 0,
                },
            );
            self.tls_pages += 1;
            // Slot zero of a page is left for the real kernel's exception handling.
            for slot in (1..PAGE / TLS_SLOT_SIZE).rev() {
                self.tls_free.push(page + slot * TLS_SLOT_SIZE);
            }
        }
        let slot = self.tls_free.pop().expect("just filled");
        let _ = rt()
            .mem
            .write_bytes(slot, &[0u8; TLS_SLOT_SIZE as usize], false);
        Ok(slot)
    }
}

/// The waiter that should run next: the highest priority (lowest number), then the earliest.
fn take_best<T: Copy>(list: &mut Vec<T>, key: impl Fn(&T) -> (i32, u64)) -> Option<T> {
    let best = (0..list.len()).min_by_key(|&i| key(&list[i]))?;
    Some(list.remove(best))
}

impl Kernel {
    pub fn new(process: ProcessInfo) -> Arc<Kernel> {
        let state = State {
            handles: HandleTable::new(),
            memory: MemoryMap::new(),
            heap_size: 0,
            threads: BTreeMap::new(),
            events: Vec::new(),
            tls_free: Vec::new(),
            tls_pages: 0,
            mutex_waiters: HashMap::new(),
            cv_waiters: HashMap::new(),
            arbiter_waiters: HashMap::new(),
            exit: None,
            reporting: false,
            seq: 0,
            main_handle: 0,
            mapped: Vec::new(),
            transfer_memories: Vec::new(),
            shared: Vec::new(),
            kernel_cursor: 0,
        };
        Arc::new(Kernel {
            process,
            state: Mutex::new(state),
            wake: Condvar::new(),
            registry: Mutex::new(BTreeMap::new()),
            output: Mutex::new(Box::new(std::io::stdout())),
            random: Mutex::new(random_seed()),
            trace: std::env::var_os("NX_TRACE_SVC").is_some(),
            trace_waits: std::env::var("NX_TRACE_WAITS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(std::time::Duration::from_micros),
            started: std::time::Instant::now(),
            trace_log: std::env::var_os("NX_TRACE_LOG").is_some(),
            finish_hooks: Mutex::new(Vec::new()),
            monitors: Mutex::new(BTreeMap::new()),
            watched: super::watchdog::requested().is_some(),
            cores: Cores::from_env(),
            jitter: jitter_from_env(),
            jitter_state: std::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15),
        })
    }

    /// The monitors of the running guest threads, by thread.
    pub(super) fn monitors(&self) -> MutexGuard<'_, BTreeMap<u64, Arc<ThreadMonitor>>> {
        self.monitors.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Tell the watchdog what the supervisor call `thread` is in is about (built only when a
    /// watchdog is running).
    pub(super) fn note(&self, thread: u64, text: impl FnOnce() -> String) {
        if self.watched {
            if let Some(m) = self.monitors().get(&thread) {
                m.set_detail(text());
            }
        }
    }

    /// The entry point, argument, priority and ideal core a thread was created with (they may
    /// have been changed since).
    pub(super) fn thread_entry(&self, thread: u64) -> Option<(u64, u64, i32, i32)> {
        self.lock()
            .threads
            .get(&thread)
            .map(|t| (t.entry, t.arg, t.priority, t.running_core()))
    }

    pub fn trace_log(&self) -> bool {
        self.trace_log
    }

    /// Is `NX_TRACE_SVC` on?
    pub fn tracing(&self) -> bool {
        self.trace
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wake every blocked thread so it re-checks its condition.
    pub(super) fn notify(&self) {
        self.wake.notify_all();
    }

    /// Block until `ready` returns a value, the deadline passes, or the process is ending.
    pub(super) fn wait_until<'a, R>(
        &'a self,
        mut st: MutexGuard<'a, State>,
        deadline: Option<Instant>,
        mut ready: impl FnMut(&mut State) -> Option<R>,
    ) -> (MutexGuard<'a, State>, Waited<R>) {
        loop {
            if let Some(r) = ready(&mut st) {
                return (st, Waited::Ready(r));
            }
            if st.exit.is_some() {
                return (st, Waited::Terminating);
            }
            match deadline {
                None => st = self.wake.wait(st).unwrap_or_else(|e| e.into_inner()),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return (st, Waited::TimedOut);
                    }
                    st = self
                        .wake
                        .wait_timeout(st, d - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            }
        }
    }

    // ---- services ----

    /// Make `name` connectable through the service manager. A later registration replaces an
    /// earlier one.
    pub fn register_service(&self, name: &str, factory: ServiceFactory) {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string(), factory);
    }

    pub(super) fn service_object(&self, name: &str) -> Option<Box<dyn super::service::Service>> {
        let reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.get(name).map(|f| f())
    }

    pub fn implemented_services(&self) -> Vec<String> {
        let reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.keys().cloned().collect()
    }

    // ---- guest memory ----

    pub fn read_u32(&self, addr: u64) -> Option<u32> {
        rt().mem
            .read(addr, 4, Ordering::Acquire)
            .ok()
            .map(|v| v as u32)
    }

    pub fn write_u32(&self, addr: u64, v: u32) -> bool {
        rt().store(KERNEL_THREAD_ID, addr, 4, u128::from(v), Ordering::Release)
            .is_ok()
    }

    /// Replace `old` by `new` in the word at `addr` unless it changed. `None` if the access fails.
    pub fn cas_u32(&self, addr: u64, old: u32, new: u32) -> Option<bool> {
        rt().compare_exchange(
            KERNEL_THREAD_ID,
            addr,
            4,
            u64::from(old),
            u64::from(new),
            Ordering::AcqRel,
        )
        .ok()
    }

    /// Copy bytes out of guest memory (read permission checked).
    pub fn read_memory(&self, addr: u64, out: &mut [u8]) -> bool {
        rt().mem.read_bytes(addr, out, true).is_ok()
    }

    /// Copy bytes into guest memory (write permission checked).
    pub fn write_memory(&self, addr: u64, data: &[u8]) -> bool {
        rt().mem.write_bytes(addr, data, true).is_ok()
    }

    /// The contents of an IPC buffer, at most `limit` bytes of it; `None` when it is not readable.
    pub fn read_buffer(&self, buf: super::service::Buffer, limit: u64) -> Option<Vec<u8>> {
        let mut out = vec![0u8; buf.size.min(limit) as usize];
        self.read_memory(buf.addr, &mut out).then_some(out)
    }

    /// Map fresh zeroed pages the kernel owns (unmapped when the kernel is dropped).
    pub(super) fn map_owned(
        &self,
        st: &mut State,
        addr: u64,
        size: u64,
        info: Info,
    ) -> Result<(), ResultCode> {
        rt().mem
            .map(addr, size, info.perm)
            .map_err(|_| err::OUT_OF_MEMORY)?;
        st.mapped.push((addr, size));
        if !st.memory.insert(addr, size, info) {
            let _ = rt().mem.unmap(addr, size);
            return Err(err::INVALID_CURRENT_MEMORY);
        }
        Ok(())
    }

    // ---- ending the process ----

    /// The process exits with a status (`ExitProcess`, or the main thread returned).
    pub fn exit_process(&self, code: i32) {
        let mut st = self.lock();
        st.exit.get_or_insert(Exit::Code(code));
        drop(st);
        self.notify();
    }

    /// The run cannot go on; the first report wins.
    pub fn fatal(&self, report: String) {
        let mut st = self.lock();
        let first = st.exit.is_none();
        st.exit.get_or_insert(Exit::Fatal(report));
        st.reporting |= first;
        drop(st);
        self.notify();
        if first {
            // What the other threads were doing when the run stopped is half of the story, and
            // the process must not end while it is told.
            eprintln!("[stop] the other threads at the stop:");
            super::watchdog::report_threads(self, "stop");
            self.lock().reporting = false;
            self.notify();
        }
    }

    pub fn exit_state(&self) -> Option<Exit> {
        self.lock().exit.clone()
    }

    /// Block until the process has ended.
    pub fn wait_for_exit(&self) -> Exit {
        let mut st = self.lock();
        loop {
            // The process has ended once it has an exit and nobody is telling the story of a
            // stop any more (`wait_until` cannot be used: it gives up as soon as there is an exit).
            if let Some(e) = &st.exit {
                if !st.reporting {
                    return e.clone();
                }
            }
            st = self.wake.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    // ---- handles ----

    pub fn insert_object(&self, o: KObject) -> KResult<Handle> {
        self.lock().handles.insert(o)
    }

    /// The object a handle names, with the pseudo handles resolved.
    pub fn object(&self, h: Handle, current: u64) -> Option<KObject> {
        if h == CURRENT_PROCESS {
            return Some(KObject::Process);
        }
        if h == CURRENT_THREAD {
            return Some(KObject::Thread(current));
        }
        self.lock().handles.get(h).cloned()
    }

    pub fn close_handle(&self, h: Handle) -> ResultCode {
        let removed = self.lock().handles.close(h);
        match removed {
            Ok(object) => {
                self.release_memory_object(&object);
                ResultCode::SUCCESS
            }
            Err(e) => e,
        }
    }

    // ---- threads ----

    /// Create a thread object (not yet running). `main` is the process's first thread.
    #[allow(clippy::too_many_arguments)]
    pub fn create_thread(
        &self,
        entry: u64,
        arg: u64,
        stack_top: u64,
        priority: i32,
        core: i32,
        is_main: bool,
    ) -> KResult<(Handle, u64)> {
        if !is_main {
            let (lo, hi) = self
                .process
                .npdm
                .as_ref()
                .and_then(|n| n.kernel.thread_info)
                .map_or((0, 0x3F), |t| {
                    (i32::from(t.highest_priority), i32::from(t.lowest_priority))
                });
            if !(0..=0x3F).contains(&priority) || priority < lo.min(hi) || priority > lo.max(hi) {
                return Err(err::INVALID_PRIORITY);
            }
            if !(core == -2 || (0..=3).contains(&core)) {
                return Err(err::INVALID_CORE_ID);
            }
        }
        let mut st = self.lock();
        let tls = st.alloc_tls()?;
        let id = rt().new_thread_id();
        let handle = match st.handles.insert(KObject::Thread(id)) {
            Ok(h) => h,
            Err(e) => {
                st.tls_free.push(tls);
                return Err(e);
            }
        };
        st.threads.insert(
            id,
            Thread {
                priority,
                ideal_core: if core == -2 { 0 } else { core },
                core_mask: if core < 0 { 0xF } else { 1 << core },
                tls,
                entry,
                arg,
                stack_top,
                started: false,
                finished: false,
                is_main,
                cancel_sync: false,
                wake: None,
            },
        );
        Ok((handle, id))
    }

    /// Start a created thread on its own host thread.
    pub fn start_thread(self: &Arc<Kernel>, thread: u64) -> ResultCode {
        let (entry, is_main) = {
            let mut st = self.lock();
            if st.exit.is_some() {
                return err::THREAD_TERMINATING;
            }
            let Some(t) = st.threads.get_mut(&thread) else {
                return err::INVALID_HANDLE;
            };
            if t.started {
                return err::INVALID_STATE;
            }
            t.started = true;
            (t.entry, t.is_main)
        };
        let k = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name(format!("guest-{thread}"))
            .stack_size(GUEST_STACK_BYTES)
            .spawn(move || k.run_thread(thread, entry, is_main));
        match spawned {
            Ok(_) => ResultCode::SUCCESS,
            Err(_) => err::OUT_OF_RESOURCE,
        }
    }

    /// The registers a new thread starts with, as the real kernel sets them: `X0` the argument,
    /// the stack pointer, and `TPIDRRO_EL0` the thread-local region. The return address is the
    /// runtime's exit sentinel, so a thread whose entry function returns ends normally.
    pub fn new_cpu(self: &Arc<Kernel>, thread: u64) -> Box<NativeCpu> {
        let (arg, sp, tls) = {
            let st = self.lock();
            let t = &st.threads[&thread];
            (t.arg, t.stack_top, t.tls)
        };
        let mut cpu = Box::new(NativeCpu::new(thread));
        cpu.state.x[0] = arg;
        cpu.state.sp = sp;
        cpu.state.x[30] = EXIT_PC;
        cpu.state.tpidrro_el0 = tls;
        cpu
    }

    fn run_thread(self: Arc<Kernel>, thread: u64, entry: u64, is_main: bool) {
        let mut cpu = self.new_cpu(thread);
        if is_main {
            self.prepare_main_thread(&mut cpu);
        }
        self.run_on(&mut cpu, thread, entry);
    }

    /// Run `cpu` from `entry` until its thread ends, then account for how it ended.
    pub fn run_on(self: &Arc<Kernel>, cpu: &mut NativeCpu, thread: u64, entry: u64) {
        let monitor = Arc::new(ThreadMonitor::new(cpu));
        self.monitors().insert(thread, Arc::clone(&monitor));
        let ctx = Box::into_raw(Box::new(ThreadCtx {
            kernel: Arc::clone(self),
            thread,
            monitor: Arc::clone(&monitor),
            core: AtomicUsize::new(usize::MAX),
        }));
        cpu.ctl.user = ctx.cast::<c_void>();
        // SAFETY: just created, and only this thread touches it.
        let core = unsafe { &(*ctx).core };
        self.take_core(thread, core);
        // SAFETY: the CPU belongs to this thread alone and generated code is registered.
        let reason = unsafe { run(cpu, entry) };
        self.give_core(core);
        cpu.ctl.user = std::ptr::null_mut();
        self.monitors().remove(&thread);
        monitor.retire();
        // SAFETY: created above and no longer referenced.
        drop(unsafe { Box::from_raw(ctx) });
        self.thread_ended(thread, reason, cpu);
    }

    /// Sleep a little at random before a supervisor call, when `NX_JITTER` asks.
    pub(super) fn stir(&self) {
        let Some((percent, micros)) = self.jitter else {
            return;
        };
        // xorshift on a state all threads share: the sequence a thread sees is not repeatable, which
        // is the point.
        let mut x = self.jitter_state.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.jitter_state.store(x, Ordering::Relaxed);
        if x % 100 < percent {
            std::thread::sleep(Duration::from_micros((x >> 8) % micros));
        }
    }

    /// Wait for the core `thread` runs on, if cores are modelled (`NX_CORES`); `held` records it.
    pub(super) fn take_core(&self, thread: u64, held: &AtomicUsize) {
        if let Some(cores) = &self.cores {
            let ideal = self
                .lock()
                .threads
                .get(&thread)
                .map_or(0, Thread::running_core);
            let core = cores.core_of(ideal);
            cores.acquire(core);
            held.store(core, Ordering::Relaxed);
        }
    }

    /// Let go of the core the thread holds.
    pub(super) fn give_core(&self, held: &AtomicUsize) {
        if let Some(cores) = &self.cores {
            cores.release(held.load(Ordering::Relaxed));
        }
    }

    /// Does the supervisor call `imm` leave the core for others (when cores are modelled)?
    pub(super) fn call_gives_way(&self, imm: u32) -> bool {
        self.cores.is_some() && super::cores::gives_way(imm)
    }

    fn thread_ended(&self, thread: u64, reason: u32, cpu: &NativeCpu) {
        let is_main = self.lock().threads.get(&thread).is_some_and(|t| t.is_main);
        match reason {
            0 => {
                if is_main {
                    // The entry function returned: the process ends with its result.
                    self.exit_process(cpu.state.x[0] as i32);
                }
            }
            stop::EXIT_THREAD | stop::EXIT_PROCESS => {}
            _ => {
                // Generated code keeps the registers in variables and stores them in the CPU
                // structure at calls, so after a fault the structure holds them as of the last
                // call: the chain of callers is right, the faulting function itself is `pc`.
                let stack = self.guest_call_stack(cpu);
                let mut report = describe_stop(cpu);
                if !stack.is_empty() {
                    let frames: Vec<String> = stack.iter().map(|a| format!("{a:#x}")).collect();
                    report.push_str(&format!(
                        " (thread {thread}); return addresses as of its last call: {}",
                        frames.join(" ")
                    ));
                }
                let registers: Vec<String> = cpu
                    .state
                    .x
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("x{i}={v:#x}"))
                    .collect();
                let (entry, arg, priority, core) = self.thread_entry(thread).unwrap_or_default();
                eprintln!(
                    "[stop] registers of thread {thread} (entry {entry:#x}, argument {arg:#x}, \
                     priority {priority}, core {core}) as of its last call: {} sp={:#x}",
                    registers.join(" "),
                    cpu.state.sp
                );
                dump_pointees(cpu);
                self.fatal(report);
            }
        }
        let mut st = self.lock();
        if let Some(t) = st.threads.get_mut(&thread) {
            t.finished = true;
            let tls = t.tls;
            st.tls_free.push(tls);
        }
        // When the last started thread has ended the process has nothing left to run.
        if st.exit.is_none() && st.threads.values().all(|t| !t.started || t.finished) {
            st.exit = Some(Exit::Code(0));
        }
        drop(st);
        self.notify();
    }

    // ---- events ----

    pub fn create_event(&self) -> KResult<(Handle, Handle)> {
        let mut st = self.lock();
        let index = st.events.len();
        st.events.push(Event { signaled: false });
        let writable = st.handles.insert(KObject::WritableEvent(index))?;
        match st.handles.insert(KObject::ReadableEvent(index)) {
            Ok(readable) => Ok((writable, readable)),
            Err(e) => {
                let _ = st.handles.close(writable);
                Err(e)
            }
        }
    }

    /// A new event as two kernel objects without handles: the writable end for whoever signals it
    /// (a service) and the readable end to hand the program in a reply.
    pub fn new_event(&self) -> (KObject, KObject) {
        let mut st = self.lock();
        let index = st.events.len();
        st.events.push(Event { signaled: false });
        if self.trace {
            eprintln!("[evt] event {index} made for a service");
        }
        (KObject::WritableEvent(index), KObject::ReadableEvent(index))
    }

    /// Signal (`true`) or clear (`false`) the event one of whose ends is `end`.
    pub fn set_event(&self, end: &KObject, signaled: bool) {
        let (KObject::WritableEvent(i) | KObject::ReadableEvent(i)) = end else {
            return;
        };
        let mut st = self.lock();
        if let Some(e) = st.events.get_mut(*i) {
            e.signaled = signaled;
        }
        drop(st);
        if signaled {
            self.notify();
        }
    }

    pub fn signal_event(&self, h: Handle) -> ResultCode {
        let mut st = self.lock();
        match st.handles.get(h) {
            Some(KObject::WritableEvent(i)) => {
                let i = *i;
                st.events[i].signaled = true;
                drop(st);
                self.notify();
                ResultCode::SUCCESS
            }
            _ => err::INVALID_HANDLE,
        }
    }

    pub fn clear_event(&self, h: Handle) -> ResultCode {
        let mut st = self.lock();
        match st.handles.get(h) {
            Some(KObject::WritableEvent(i) | KObject::ReadableEvent(i)) => {
                let i = *i;
                st.events[i].signaled = false;
                ResultCode::SUCCESS
            }
            _ => err::INVALID_HANDLE,
        }
    }

    pub fn reset_signal(&self, h: Handle) -> ResultCode {
        let mut st = self.lock();
        match st.handles.get(h) {
            Some(KObject::ReadableEvent(i)) => {
                let i = *i;
                if st.events[i].signaled {
                    st.events[i].signaled = false;
                    ResultCode::SUCCESS
                } else {
                    err::INVALID_STATE
                }
            }
            Some(KObject::Process) => err::INVALID_STATE,
            _ => err::INVALID_HANDLE,
        }
    }

    /// Wait until one of the handles is signalled. Returns the result and the index.
    pub fn wait_synchronization(
        &self,
        current: u64,
        handles: &[Handle],
        timeout_ns: i64,
    ) -> (ResultCode, i32) {
        enum Target {
            Event(usize),
            Thread(u64),
            Process,
            Never,
        }
        let deadline = deadline_of(timeout_ns);
        let st = self.lock();
        let mut targets = Vec::with_capacity(handles.len());
        for &h in handles {
            let obj = if h == CURRENT_PROCESS {
                Some(KObject::Process)
            } else if h == CURRENT_THREAD {
                Some(KObject::Thread(current))
            } else {
                st.handles.get(h).cloned()
            };
            match obj {
                Some(KObject::ReadableEvent(i)) => targets.push(Target::Event(i)),
                Some(KObject::Thread(t)) => targets.push(Target::Thread(t)),
                Some(KObject::Process) => targets.push(Target::Process),
                Some(KObject::Session(_)) => targets.push(Target::Never),
                _ => return (err::INVALID_HANDLE, -1),
            }
        }
        let (_st, out) = self.wait_until(st, deadline, |st| {
            if let Some(t) = st.threads.get_mut(&current) {
                if t.cancel_sync {
                    t.cancel_sync = false;
                    return Some(Err(err::CANCELLED));
                }
            }
            for (i, t) in targets.iter().enumerate() {
                let signaled = match t {
                    Target::Event(e) => st.events[*e].signaled,
                    Target::Thread(id) => st.threads.get(id).is_some_and(|t| t.finished),
                    Target::Process => st.exit.is_some(),
                    Target::Never => false,
                };
                if signaled {
                    return Some(Ok(i as i32));
                }
            }
            None
        });
        match out {
            Waited::Ready(Ok(i)) => (ResultCode::SUCCESS, i),
            Waited::Ready(Err(e)) => (e, -1),
            Waited::TimedOut => (err::TIMED_OUT, -1),
            Waited::Terminating => (err::THREAD_TERMINATING, -1),
        }
    }

    pub fn cancel_synchronization(&self, current: u64, h: Handle) -> ResultCode {
        let mut st = self.lock();
        let Some(id) = st.thread_of(h, current) else {
            return err::INVALID_HANDLE;
        };
        if let Some(t) = st.threads.get_mut(&id) {
            t.cancel_sync = true;
        }
        drop(st);
        self.notify();
        ResultCode::SUCCESS
    }

    /// Sleep, or yield for the special values 0, -1 and -2.
    pub fn sleep(&self, ns: i64) {
        if ns <= 0 {
            std::thread::yield_now();
            return;
        }
        let st = self.lock();
        let (_st, _out) = self.wait_until(st, deadline_of(ns), |_| None::<()>);
    }

    // ---- mutexes, condition variables, address arbitration ----

    fn user_word(addr: u64) -> KResult<()> {
        if !addr.is_multiple_of(4) {
            return Err(err::INVALID_ADDRESS);
        }
        Ok(())
    }

    /// Give a mutex word to the best waiter (or clear it); the caller holds the state lock.
    fn unlock_mutex_locked(&self, st: &mut State, addr: u64) -> ResultCode {
        let next = st
            .mutex_waiters
            .get_mut(&addr)
            .and_then(|l| take_best(l, |w| (w.priority, w.seq)));
        let more = st.mutex_waiters.get(&addr).is_some_and(|l| !l.is_empty());
        let value = match next {
            Some(w) => w.tag | if more { MUTEX_WAIT_MASK } else { 0 },
            None => 0,
        };
        if !self.write_u32(addr, value) {
            return err::INVALID_CURRENT_MEMORY;
        }
        if let Some(w) = next {
            if let Some(t) = st.threads.get_mut(&w.thread) {
                t.wake = Some(ResultCode::SUCCESS);
            }
            self.wake.notify_all();
        }
        ResultCode::SUCCESS
    }

    /// `ArbitrateLock`: wait for the mutex at `addr`, whose owner holds `owner`.
    pub fn arbitrate_lock(&self, current: u64, owner: Handle, addr: u64, tag: u32) -> ResultCode {
        if let Err(e) = Self::user_word(addr) {
            return e;
        }
        let mut st = self.lock();
        let Some(word) = self.read_u32(addr) else {
            return err::INVALID_CURRENT_MEMORY;
        };
        // If the word no longer says "owned by `owner`, with waiters", the lock moved on while
        // the caller was getting here: nothing to wait for.
        if word != (owner | MUTEX_WAIT_MASK) {
            return ResultCode::SUCCESS;
        }
        // The owner has to be a thread of this process.
        if st.thread_of(owner, current).is_none() {
            return err::INVALID_HANDLE;
        }
        let seq = st.next_seq();
        let priority = st.threads.get(&current).map_or(0x2C, |t| t.priority);
        st.mutex_waiters.entry(addr).or_default().push(Waiter {
            thread: current,
            tag,
            priority,
            seq,
        });
        if let Some(t) = st.threads.get_mut(&current) {
            t.wake = None;
        }
        let (mut st, out) = self.wait_until(st, None, |st| {
            st.threads.get_mut(&current).and_then(|t| t.wake.take())
        });
        match out {
            Waited::Ready(r) => r,
            _ => {
                if let Some(l) = st.mutex_waiters.get_mut(&addr) {
                    l.retain(|w| w.thread != current);
                }
                err::THREAD_TERMINATING
            }
        }
    }

    /// `ArbitrateUnlock`.
    pub fn arbitrate_unlock(&self, addr: u64) -> ResultCode {
        if let Err(e) = Self::user_word(addr) {
            return e;
        }
        let mut st = self.lock();
        self.unlock_mutex_locked(&mut st, addr)
    }

    /// `WaitProcessWideKeyAtomic`: release the mutex at `mutex`, wait on `key`, and take the mutex
    /// back when signalled. A timeout returns without the mutex.
    pub fn wait_process_wide_key(
        &self,
        current: u64,
        mutex: u64,
        key: u64,
        tag: u32,
        timeout_ns: i64,
    ) -> ResultCode {
        if let Err(e) = Self::user_word(mutex) {
            return e;
        }
        let deadline = deadline_of(timeout_ns);
        let mut st = self.lock();
        if st.exit.is_some() {
            return err::THREAD_TERMINATING;
        }
        let released = self.unlock_mutex_locked(&mut st, mutex);
        if !released.is_success() {
            return released;
        }
        let seq = st.next_seq();
        let priority = st.threads.get(&current).map_or(0x2C, |t| t.priority);
        st.cv_waiters.entry(key).or_default().push(CvWaiter {
            waiter: Waiter {
                thread: current,
                tag,
                priority,
                seq,
            },
            mutex,
        });
        // The kernel tells user mode that somebody waits by setting the key word to 1; a failed
        // write is not an error (Atmosphère's `KConditionVariable::Wait`).
        let _ = self.write_u32(key, 1);
        if let Some(t) = st.threads.get_mut(&current) {
            t.wake = None;
        }
        let take = |st: &mut State| st.threads.get_mut(&current).and_then(|t| t.wake.take());
        let (mut st, out) = self.wait_until(st, deadline, take);
        match out {
            Waited::Ready(r) => r,
            Waited::TimedOut => {
                let in_cv = st.cv_waiters.get_mut(&key).is_some_and(|l| {
                    let before = l.len();
                    l.retain(|w| w.waiter.thread != current);
                    l.len() != before
                });
                if in_cv {
                    return err::TIMED_OUT;
                }
                // A signal already queued the thread on the mutex: the timeout no longer applies,
                // only the lock is awaited.
                let (_st, out) = self.wait_until(st, None, take);
                match out {
                    Waited::Ready(r) => r,
                    _ => err::THREAD_TERMINATING,
                }
            }
            Waited::Terminating => {
                if let Some(l) = st.cv_waiters.get_mut(&key) {
                    l.retain(|w| w.waiter.thread != current);
                }
                if let Some(l) = st.mutex_waiters.get_mut(&mutex) {
                    l.retain(|w| w.thread != current);
                }
                err::THREAD_TERMINATING
            }
        }
    }

    /// `SignalProcessWideKey`: wake up to `count` waiters (all when `count` is not positive).
    pub fn signal_process_wide_key(&self, key: u64, count: i32) {
        let mut st = self.lock();
        let mut woken = 0;
        while count <= 0 || woken < count {
            let Some(cw) = st
                .cv_waiters
                .get_mut(&key)
                .and_then(|l| take_best(l, |w| (w.waiter.priority, w.waiter.seq)))
            else {
                break;
            };
            woken += 1;
            // The woken thread must own the mutex before it runs: take it if free, otherwise
            // queue the thread behind the owner and tell the owner it has waiters.
            while let Some(word) = self.read_u32(cw.mutex) {
                if word == 0 {
                    if self.cas_u32(cw.mutex, 0, cw.waiter.tag) == Some(true) {
                        if let Some(t) = st.threads.get_mut(&cw.waiter.thread) {
                            t.wake = Some(ResultCode::SUCCESS);
                        }
                        break;
                    }
                    continue;
                }
                if word & MUTEX_WAIT_MASK == 0
                    && self.cas_u32(cw.mutex, word, word | MUTEX_WAIT_MASK) != Some(true)
                {
                    continue;
                }
                st.mutex_waiters
                    .entry(cw.mutex)
                    .or_default()
                    .push(cw.waiter);
                break;
            }
        }
        // When nobody waits on the key any more the key word goes back to 0.
        if st.cv_waiters.get(&key).is_none_or(|l| l.is_empty()) {
            let _ = self.write_u32(key, 0);
        }
        drop(st);
        self.notify();
    }

    /// `WaitForAddress` with the three arbitration types.
    pub fn wait_for_address(
        &self,
        current: u64,
        addr: u64,
        kind: u32,
        value: i32,
        timeout_ns: i64,
    ) -> ResultCode {
        if let Err(e) = Self::user_word(addr) {
            return e;
        }
        if kind > 2 {
            return err::INVALID_ENUM_VALUE;
        }
        let deadline = deadline_of(timeout_ns);
        let mut st = self.lock();
        if st.exit.is_some() {
            return err::THREAD_TERMINATING;
        }
        let Some(cur) = self.read_u32(addr).map(|v| v as i32) else {
            return err::INVALID_CURRENT_MEMORY;
        };
        let proceed = match kind {
            0 => cur < value,
            1 => {
                // Decrement only if the comparison holds, atomically with the check.
                cur < value
                    && self.cas_u32(addr, cur as u32, cur.wrapping_sub(1) as u32) == Some(true)
            }
            _ => cur == value,
        };
        if !proceed {
            return err::INVALID_STATE;
        }
        if timeout_ns == 0 {
            return err::TIMED_OUT;
        }
        let seq = st.next_seq();
        let priority = st.threads.get(&current).map_or(0x2C, |t| t.priority);
        st.arbiter_waiters.entry(addr).or_default().push(Waiter {
            thread: current,
            tag: 0,
            priority,
            seq,
        });
        if let Some(t) = st.threads.get_mut(&current) {
            t.wake = None;
        }
        let (mut st, out) = self.wait_until(st, deadline, |st| {
            st.threads.get_mut(&current).and_then(|t| t.wake.take())
        });
        match out {
            Waited::Ready(r) => r,
            other => {
                if let Some(l) = st.arbiter_waiters.get_mut(&addr) {
                    l.retain(|w| w.thread != current);
                }
                match other {
                    Waited::TimedOut => err::TIMED_OUT,
                    _ => err::THREAD_TERMINATING,
                }
            }
        }
    }

    /// `SignalToAddress`: type 0 signals, type 1 adds one to the word first when it equals `value`,
    /// and type 2 sets it, when it equals `value`, from the number of threads waiting on it
    /// (docs/ASSUMPTIONS.md A-148).
    pub fn signal_to_address(&self, addr: u64, kind: u32, value: i32, count: i32) -> ResultCode {
        if let Err(e) = Self::user_word(addr) {
            return e;
        }
        let mut st = self.lock();
        if kind == 1 || kind == 2 {
            let new_value = if kind == 1 {
                value.wrapping_add(1)
            } else {
                let waiting = st.arbiter_waiters.get(&addr).map_or(0, Vec::len) as i32;
                match (count <= 0, waiting) {
                    // Nobody waits: one more than the value.
                    (_, 0) => value.wrapping_add(1),
                    // Everyone is woken: two less.
                    (true, _) => value.wrapping_sub(2),
                    // Some are woken: one less when that is all of them, else the value stays.
                    (false, n) if n <= count => value.wrapping_sub(1),
                    (false, _) => value,
                }
            };
            let Some(cur) = self.read_u32(addr).map(|v| v as i32) else {
                return err::INVALID_CURRENT_MEMORY;
            };
            if cur != value
                || (new_value != cur
                    && self.cas_u32(addr, cur as u32, new_value as u32) != Some(true))
            {
                return err::INVALID_STATE;
            }
        } else if kind != 0 {
            return err::INVALID_ENUM_VALUE;
        }
        let mut woken = 0;
        while count <= 0 || woken < count {
            let Some(w) = st
                .arbiter_waiters
                .get_mut(&addr)
                .and_then(|l| take_best(l, |w| (w.priority, w.seq)))
            else {
                break;
            };
            woken += 1;
            if let Some(t) = st.threads.get_mut(&w.thread) {
                t.wake = Some(ResultCode::SUCCESS);
            }
        }
        drop(st);
        self.notify();
        ResultCode::SUCCESS
    }

    // ---- process memory ----

    /// `SetHeapSize`: the heap is `size` bytes (a multiple of 2 MiB) at the start of the heap
    /// region. Returns its address.
    pub fn set_heap_size(&self, size: u64) -> KResult<u64> {
        use super::layout::{HEAP_BASE, HEAP_REGION_SIZE};
        if !size.is_multiple_of(0x20_0000) || size > 1 << 32 {
            return Err(err::INVALID_SIZE);
        }
        if size > HEAP_REGION_SIZE {
            return Err(err::OUT_OF_MEMORY);
        }
        let mut st = self.lock();
        let cur = st.heap_size;
        let info = Info {
            mem_type: mem_type::NORMAL,
            perm: perm::R | perm::W,
            attr: 0,
        };
        if size > cur {
            self.map_owned(&mut st, HEAP_BASE + cur, size - cur, info)?;
        } else if size < cur {
            if !st.memory.remove(HEAP_BASE + size, cur - size) {
                return Err(err::INVALID_CURRENT_MEMORY);
            }
            let _ = rt().mem.unmap(HEAP_BASE + size, cur - size);
            // Those pages are no longer this kernel's to unmap again.
            st.mapped
                .retain(|&(a, _)| !(HEAP_BASE + size..HEAP_BASE + cur).contains(&a));
        }
        st.heap_size = size;
        Ok(HEAP_BASE)
    }

    /// The checks `MapPhysicalMemory` and `UnmapPhysicalMemory` share: the sizes, the alias region
    /// and, for a program that has an NPDM, a system resource (docs/ASSUMPTIONS.md A-130).
    fn check_physical_range(&self, addr: u64, size: u64) -> Result<(), ResultCode> {
        use super::layout::{ALIAS_BASE, ALIAS_SIZE};
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(err::INVALID_SIZE);
        }
        if !addr.is_multiple_of(PAGE) {
            return Err(err::INVALID_ADDRESS);
        }
        let end = addr.checked_add(size).ok_or(err::INVALID_MEMORY_REGION)?;
        if addr < ALIAS_BASE || end > ALIAS_BASE + ALIAS_SIZE {
            return Err(err::INVALID_MEMORY_REGION);
        }
        if let Some(npdm) = &self.process.npdm {
            if npdm.meta.system_resource_size == 0 {
                return Err(err::INVALID_STATE);
            }
        }
        Ok(())
    }

    /// The pages `MapPhysicalMemory` makes: zeroed, read-write, of the type the heap has.
    fn physical_info() -> Info {
        Info {
            mem_type: mem_type::NORMAL,
            perm: perm::R | perm::W,
            attr: 0,
        }
    }

    /// `MapPhysicalMemory`: zeroed read-write pages at `addr` in the alias region. The parts of the
    /// range that are already such pages stay as they are; any other page in it is an error.
    pub fn map_physical_memory(&self, addr: u64, size: u64) -> ResultCode {
        if let Err(e) = self.check_physical_range(addr, size) {
            return e;
        }
        let info = Self::physical_info();
        let mut st = self.lock();
        // The free parts of the range.
        let mut gaps = Vec::new();
        let mut at = addr;
        for (start, len, found) in st.memory.blocks_in(addr, size) {
            if found != info {
                return err::INVALID_CURRENT_MEMORY;
            }
            if start > at {
                gaps.push((at, start - at));
            }
            at = start + len;
        }
        if at < addr + size {
            gaps.push((at, addr + size - at));
        }
        let new: u64 = gaps.iter().map(|g| g.1).sum();
        let used = st.memory.total(|t| {
            matches!(
                t,
                mem_type::CODE
                    | mem_type::CODE_DATA
                    | mem_type::NORMAL
                    | mem_type::STACK
                    | mem_type::THREAD_LOCAL
            )
        });
        if used + new > super::svc::TOTAL_MEMORY_SIZE {
            return err::LIMIT_REACHED;
        }
        let mut done = Vec::new();
        for (gap_addr, gap_size) in gaps {
            if let Err(e) = self.map_owned(&mut st, gap_addr, gap_size, info) {
                // Leave nothing of this call behind.
                for (a, s) in done {
                    st.memory.remove(a, s);
                    let _ = rt().mem.unmap(a, s);
                    forget_mapped(&mut st, a, a + s);
                }
                return e;
            }
            done.push((gap_addr, gap_size));
        }
        ResultCode::SUCCESS
    }

    /// `UnmapPhysicalMemory`: give back the pages of the range that `MapPhysicalMemory` made; the
    /// free parts of it are left alone and any other page is an error.
    pub fn unmap_physical_memory(&self, addr: u64, size: u64) -> ResultCode {
        if let Err(e) = self.check_physical_range(addr, size) {
            return e;
        }
        let info = Self::physical_info();
        let mut st = self.lock();
        let blocks = st.memory.blocks_in(addr, size);
        if blocks.iter().any(|b| b.2 != info) {
            return err::INVALID_CURRENT_MEMORY;
        }
        for (a, s, _) in blocks {
            st.memory.remove(a, s);
            let _ = rt().mem.unmap(a, s);
            forget_mapped(&mut st, a, a + s);
        }
        ResultCode::SUCCESS
    }

    pub fn query_memory(&self, addr: u64) -> super::memory::MemoryInfo {
        self.lock().memory.query(addr)
    }

    /// `SetMemoryPermission`: ---, r-- or rw- on pages whose state allows changing it.
    pub fn set_memory_permission(&self, addr: u64, size: u64, permission: u32) -> ResultCode {
        if !addr.is_multiple_of(PAGE) {
            return err::INVALID_ADDRESS;
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return err::INVALID_SIZE;
        }
        if ![0, perm::R, perm::R | perm::W].contains(&permission) {
            return err::INVALID_NEW_MEMORY_PERMISSION;
        }
        let mut st = self.lock();
        let blocks = st.memory.blocks_in(addr, size);
        let covered: u64 = blocks.iter().map(|b| b.1).sum();
        let reprotectable = |t: u8| {
            matches!(
                t,
                mem_type::CODE_DATA
                    | mem_type::NORMAL
                    | mem_type::ALIAS
                    | mem_type::ALIAS_CODE_DATA
            )
        };
        if covered != size
            || blocks
                .iter()
                .any(|(_, _, i)| !reprotectable(i.mem_type) || i.attr & attr::LOCKED != 0)
        {
            return err::INVALID_CURRENT_MEMORY;
        }
        if rt().mem.protect(addr, size, permission).is_err() {
            return err::INVALID_CURRENT_MEMORY;
        }
        st.memory.update(addr, size, |i| Info {
            perm: permission,
            ..i
        });
        ResultCode::SUCCESS
    }

    /// `SetMemoryAttribute`: only the uncached bit can change.
    pub fn set_memory_attribute(&self, addr: u64, size: u64, mask: u32, value: u32) -> ResultCode {
        if !addr.is_multiple_of(PAGE) {
            return err::INVALID_ADDRESS;
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return err::INVALID_SIZE;
        }
        if mask != attr::UNCACHED || value & !attr::UNCACHED != 0 {
            return err::INVALID_COMBINATION;
        }
        let mut st = self.lock();
        if !st.memory.update(addr, size, |i| Info {
            attr: (i.attr & !mask) | (value & mask),
            ..i
        }) {
            return err::INVALID_CURRENT_MEMORY;
        }
        ResultCode::SUCCESS
    }

    /// `MapMemory`: `dst` (inside the stack region) becomes a second view of `src`, which turns
    /// inaccessible until `UnmapMemory`.
    pub fn map_memory(&self, dst: u64, src: u64, size: u64) -> ResultCode {
        use super::layout::{STACK_BASE, STACK_REGION_SIZE};
        if !dst.is_multiple_of(PAGE) || !src.is_multiple_of(PAGE) {
            return err::INVALID_ADDRESS;
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return err::INVALID_SIZE;
        }
        if dst < STACK_BASE || dst.saturating_add(size) > STACK_BASE + STACK_REGION_SIZE {
            return err::INVALID_MEMORY_REGION;
        }
        let mut st = self.lock();
        let blocks = st.memory.blocks_in(src, size);
        let covered: u64 = blocks.iter().map(|b| b.1).sum();
        let aliasable = |t: u8| matches!(t, mem_type::NORMAL | mem_type::CODE_DATA);
        if covered != size
            || blocks.iter().any(|(_, _, i)| {
                !aliasable(i.mem_type) || i.attr != 0 || i.perm != perm::R | perm::W
            })
        {
            return err::INVALID_CURRENT_MEMORY;
        }
        if !st.memory.blocks_in(dst, size).is_empty() {
            return err::INVALID_CURRENT_MEMORY;
        }
        if rt().mem.alias(dst, src, size, perm::R | perm::W).is_err() {
            return err::INVALID_CURRENT_MEMORY;
        }
        let _ = rt().mem.protect(src, size, 0);
        st.memory.update(src, size, |i| Info {
            perm: 0,
            attr: i.attr | attr::LOCKED,
            ..i
        });
        st.memory.insert(
            dst,
            size,
            Info {
                mem_type: mem_type::STACK,
                perm: perm::R | perm::W,
                attr: 0,
            },
        );
        ResultCode::SUCCESS
    }

    /// `UnmapMemory`: undo `MapMemory`.
    pub fn unmap_memory(&self, dst: u64, src: u64, size: u64) -> ResultCode {
        if !dst.is_multiple_of(PAGE) || !src.is_multiple_of(PAGE) {
            return err::INVALID_ADDRESS;
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return err::INVALID_SIZE;
        }
        let mut st = self.lock();
        let dst_ok = st
            .memory
            .blocks_in(dst, size)
            .iter()
            .map(|b| b.1)
            .sum::<u64>()
            == size
            && st
                .memory
                .blocks_in(dst, size)
                .iter()
                .all(|(_, _, i)| i.mem_type == mem_type::STACK);
        let src_ok = st
            .memory
            .blocks_in(src, size)
            .iter()
            .map(|b| b.1)
            .sum::<u64>()
            == size
            && st
                .memory
                .blocks_in(src, size)
                .iter()
                .all(|(_, _, i)| i.attr & attr::LOCKED != 0);
        if !dst_ok || !src_ok {
            return err::INVALID_CURRENT_MEMORY;
        }
        let _ = rt().mem.unmap(dst, size);
        let _ = rt().mem.protect(src, size, perm::R | perm::W);
        st.memory.remove(dst, size);
        st.memory.update(src, size, |i| Info {
            perm: perm::R | perm::W,
            attr: i.attr & !attr::LOCKED,
            ..i
        });
        ResultCode::SUCCESS
    }
}

impl Kernel {
    /// Make this kernel the one that answers supervisor calls, prepare the process and start its
    /// main thread.
    pub fn start(self: &Arc<Kernel>, cfg: &super::process::StartConfig) -> KResult<()> {
        rt().set_svc_handler(Some(super::svc::svc_entry));
        if let Some(limit) = super::watchdog::requested() {
            super::watchdog::spawn(Arc::clone(self), limit);
        }
        let main = self.prepare(cfg)?;
        let r = self.start_thread(main);
        if r.is_success() {
            Ok(())
        } else {
            Err(r)
        }
    }

    /// Run the process to its end.
    pub fn run_process(self: &Arc<Kernel>, cfg: &super::process::StartConfig) -> Exit {
        let exit = match self.start(cfg) {
            Ok(()) => self.wait_for_exit(),
            Err(e) => Exit::Fatal(format!(
                "the process could not be started (result {:#x})",
                e.0
            )),
        };
        self.run_finish_hooks();
        exit
    }

    /// Call `f` when the process has ended, after its last instruction.
    pub fn on_finish(&self, f: Box<dyn Fn() + Send + Sync>) {
        self.finish_hooks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(f);
    }

    fn run_finish_hooks(&self) {
        let hooks =
            std::mem::take(&mut *self.finish_hooks.lock().unwrap_or_else(|e| e.into_inner()));
        for h in hooks {
            h();
        }
    }
}
