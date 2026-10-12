//! The kernel: process state, threads, events, and the blocking primitives.
//!
//! All shared state lives behind one lock with one condition variable. A blocked thread re-checks
//! its own condition whenever anything changes, which is simple and correct; it costs a wake-up
//! per change per waiter, which does not matter at this scale. Guest memory is touched through the
//! runtime's checked interface, never directly.

use super::handle::{Handle, HandleTable, CURRENT_PROCESS, CURRENT_THREAD};
use super::layout::{attr, mem_type, PAGE, TLS_BASE, TLS_SLOT_SIZE};
use super::memory::{Info, MemoryMap};
use super::object::KObject;
use super::result::{kernel as err, KResult, ResultCode};
use super::service::ServiceFactory;
use crate::npdm::Npdm;
use crate::rt::abi::{perm, stop, NativeCpu, EXIT_PC};
use crate::rt::host::describe_stop;
use crate::rt::runtime::{rt, run};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::Ordering;
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
    pub seq: u64,
    pub main_handle: Handle,
    /// Regions this kernel mapped into guest memory, unmapped again when it is dropped.
    pub mapped: Vec<(u64, u64)>,
    /// Ranges lent to services with `CreateTransferMemory`, by object index.
    pub transfer_memories: Vec<super::memobj::TransferMemory>,
}

/// What each guest thread's `ctl.user` points at, so an SVC finds its kernel.
pub struct ThreadCtx {
    pub kernel: Arc<Kernel>,
    pub thread: u64,
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
    /// Print what the program logs through `lm` (`NX_TRACE_LOG`).
    trace_log: bool,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        let st = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        for &(addr, size) in &st.mapped {
            let _ = rt().mem.unmap(addr, size);
        }
    }
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
            seq: 0,
            main_handle: 0,
            mapped: Vec::new(),
            transfer_memories: Vec::new(),
        };
        Arc::new(Kernel {
            process,
            state: Mutex::new(state),
            wake: Condvar::new(),
            registry: Mutex::new(BTreeMap::new()),
            output: Mutex::new(Box::new(std::io::stdout())),
            random: Mutex::new(random_seed()),
            trace: std::env::var_os("NX_TRACE_SVC").is_some(),
            trace_log: std::env::var_os("NX_TRACE_LOG").is_some(),
        })
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
        st.exit.get_or_insert(Exit::Fatal(report));
        drop(st);
        self.notify();
    }

    pub fn exit_state(&self) -> Option<Exit> {
        self.lock().exit.clone()
    }

    /// Block until the process has ended.
    pub fn wait_for_exit(&self) -> Exit {
        let st = self.lock();
        let (_st, out) = self.wait_until(st, None, |st| st.exit.clone());
        match out {
            Waited::Ready(e) => e,
            _ => unreachable!("waiting for the exit cannot end another way"),
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
        let ctx = Box::into_raw(Box::new(ThreadCtx {
            kernel: Arc::clone(self),
            thread,
        }));
        cpu.ctl.user = ctx.cast::<c_void>();
        // SAFETY: the CPU belongs to this thread alone and generated code is registered.
        let reason = unsafe { run(cpu, entry) };
        cpu.ctl.user = std::ptr::null_mut();
        // SAFETY: created above and no longer referenced.
        drop(unsafe { Box::from_raw(ctx) });
        self.thread_ended(thread, reason, cpu);
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
            _ => self.fatal(describe_stop(cpu)),
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

    /// `SignalToAddress` for the signal types 0 and 1. Type 2 is reported by the caller.
    pub fn signal_to_address(&self, addr: u64, kind: u32, value: i32, count: i32) -> ResultCode {
        if let Err(e) = Self::user_word(addr) {
            return e;
        }
        let mut st = self.lock();
        if kind == 1 {
            let Some(cur) = self.read_u32(addr).map(|v| v as i32) else {
                return err::INVALID_CURRENT_MEMORY;
            };
            if cur != value
                || self.cas_u32(addr, cur as u32, cur.wrapping_add(1) as u32) != Some(true)
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
        match self.start(cfg) {
            Ok(()) => self.wait_for_exit(),
            Err(e) => Exit::Fatal(format!(
                "the process could not be started (result {:#x})",
                e.0
            )),
        }
    }
}
