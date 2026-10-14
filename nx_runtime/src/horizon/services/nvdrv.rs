//! `nvdrv`, the graphics driver service (Switchbrew, "NV services").
//!
//! This is only the IPC half: the commands carry a descriptor and an ioctl number with buffers, and
//! everything they mean is `nx_gpu::nvdrv`. Kernel events are made here, on demand, for the tokens
//! the GPU model hands out.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use crate::rt::runtime::rt;
use nx_gpu::gpu::{Queue, SubmitError, Task};
use nx_gpu::nvdrv::args::Fail;
use nx_gpu::nvdrv::{EventToken, Nvdrv};
use nx_gpu::{GuestMemory, NvError};
use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Duration;

/// The guest process's memory as the GPU sees it: no CPU permissions in the way.
struct ProcessMemory;

impl GuestMemory for ProcessMemory {
    fn read(&self, addr: u64, out: &mut [u8]) -> bool {
        rt().mem.read_bytes(addr, out, false).is_ok()
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        rt().mem.write_bytes(addr, data, false).is_ok()
    }

    fn host_run(&self, addr: u64, max: u64) -> Option<(*mut u8, u64)> {
        rt().mem.host_run(addr, max)
    }

    fn begin_write_epoch(&self) -> u32 {
        rt().mem.begin_epoch()
    }

    fn untouched_since(&self, addr: u64, len: u64, epoch: u32) -> Option<bool> {
        let (mut at, mut left) = (addr, len);
        while left > 0 {
            let (host, run) = rt().mem.host_run(at, left)?;
            if !rt().mem.untouched_since(host, run as usize, epoch)? {
                return Some(false);
            }
            at += run;
            left -= run;
        }
        Some(true)
    }
}

/// The driver and the events behind its tokens, shared by every connection of the process.
pub struct Shared {
    core: Mutex<Nvdrv>,
    events: Mutex<HashMap<EventToken, (KObject, KObject)>>,
    /// The kernel whose events the thread that runs the commands sets.
    kernel: Mutex<Weak<Kernel>>,
    /// The thread that runs the commands of submissions and shows the buffers, when there is a
    /// renderer to draw with (docs/ASSUMPTIONS.md A-124); without one they are run in the call.
    worker: Option<Arc<Worker>>,
    /// Calls that want the driver now: the worker lets them in between two of its steps.
    callers: AtomicUsize,
    /// Held while a call that is over tells the worker, so that the worker does not miss it.
    way: Mutex<()>,
    way_clear: Condvar,
    /// How many calls have locked the driver (for the report of `NX_TRACE_PERF`).
    calls: std::sync::atomic::AtomicU64,
}

/// The driver, locked for a call. While a caller waits for it the worker makes way.
struct Core<'a> {
    guard: MutexGuard<'a, Nvdrv>,
    shared: &'a Shared,
}

impl Deref for Core<'_> {
    type Target = Nvdrv;

    fn deref(&self) -> &Nvdrv {
        &self.guard
    }
}

impl DerefMut for Core<'_> {
    fn deref_mut(&mut self) -> &mut Nvdrv {
        &mut self.guard
    }
}

impl Drop for Core<'_> {
    fn drop(&mut self) {
        // The worker may be waiting for the call to be over.
        let _way = self.shared.way.lock().unwrap_or_else(|e| e.into_inner());
        if self.shared.callers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.shared.way_clear.notify_all();
        }
    }
}

/// How many tasks the worker may be behind by before the calls that give it more wait.
const MAX_TASKS: usize = 64;
/// How many buffers to show it may be behind by (a frame the title has queued and the window has
/// not shown is one): the title is held back to the pace of the renderer, as the display's
/// refresh holds it back on the console.
const MAX_PRESENTS: usize = 2;

/// What the worker has been given and not finished.
#[derive(Default)]
struct Backlog {
    tasks: VecDeque<Task>,
    /// The tasks that show a buffer, among those queued and the one being run.
    presents: usize,
    /// A task is being run.
    running: bool,
}

/// The queue the commands of submissions wait in for the thread that runs them.
struct Worker {
    backlog: Mutex<Backlog>,
    /// Signaled when a task arrives and when one has been run.
    changed: Condvar,
}

impl Worker {
    fn lock(&self) -> MutexGuard<'_, Backlog> {
        self.backlog.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wait until the backlog has room for one more task (a buffer to show, when `present`).
    fn wait_for_room(&self, present: bool) {
        let mut b = self.lock();
        while b.tasks.len() + usize::from(b.running) >= MAX_TASKS
            || (present && b.presents >= MAX_PRESENTS)
        {
            b = self.changed.wait(b).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Wait (for `limit` at most) until everything given so far has been run.
    fn drain(&self, limit: Duration) {
        let b = self.lock();
        let _ = self
            .changed
            .wait_timeout_while(b, limit, |b| !b.tasks.is_empty() || b.running)
            .unwrap_or_else(|e| e.into_inner());
    }
}

impl Queue for Worker {
    fn push(&self, task: Task) {
        let mut b = self.lock();
        if matches!(task, Task::Present(_)) {
            b.presents += 1;
        }
        b.tasks.push_back(task);
        drop(b);
        self.changed.notify_all();
    }
}

/// The renderer `NX_RENDERER` asks for: `vulkan`, or nothing (the default: the commands are
/// decoded and counted, not drawn).
fn renderer() -> Option<Box<dyn nx_gpu::render::Backend>> {
    match std::env::var("NX_RENDERER").as_deref() {
        Ok("vulkan") => vulkan_renderer(),
        Ok("none") | Ok("") | Err(_) => None,
        Ok(other) => {
            eprintln!("kiln runtime: NX_RENDERER={other} is not a renderer (vulkan, none)");
            None
        }
    }
}

#[cfg(feature = "vulkan")]
fn vulkan_renderer() -> Option<Box<dyn nx_gpu::render::Backend>> {
    use super::super::input;
    use nx_vulkan::{InputEvent, InputSink};
    let mut config = nx_vulkan::RenderConfig::from_env();
    // What the person does in the window is the input of the program (the `hid` service).
    config.input = Some(InputSink::new(|event| match event {
        InputEvent::Key { key, down } => input::key_event(key, down),
        InputEvent::Focus(true) => input::focus_gained(),
        InputEvent::Focus(false) => input::focus_lost(),
    }));
    // A game takes seconds to draw its first frame: the window it will draw in opens now.
    config.open_window_early();
    Some(Box::new(nx_vulkan::Renderer::new(config)))
}

#[cfg(not(feature = "vulkan"))]
fn vulkan_renderer() -> Option<Box<dyn nx_gpu::render::Backend>> {
    eprintln!("kiln runtime: NX_RENDERER=vulkan, and this runtime was built without it");
    None
}

impl Shared {
    pub fn new() -> Arc<Shared> {
        let mut core = Nvdrv::new(Arc::new(ProcessMemory));
        core.gpu.backend = renderer();
        // With a renderer the commands of a submission run on a thread of their own, and the
        // call that made them is over at once, as the console's driver's is (`NX_GPU_SYNC=1`:
        // in the call, as every run did before).
        let queued =
            core.gpu.backend.is_some() && !std::env::var("NX_GPU_SYNC").is_ok_and(|v| v != "0");
        let worker = queued.then(|| {
            Arc::new(Worker {
                backlog: Mutex::new(Backlog::default()),
                changed: Condvar::new(),
            })
        });
        if let Some(w) = &worker {
            core.gpu.queue = Some(w.clone());
        }
        Arc::new_cyclic(|weak: &Weak<Shared>| {
            if let Some(w) = &worker {
                let (weak, w) = (weak.clone(), w.clone());
                std::thread::Builder::new()
                    .name("gpu".to_string())
                    .spawn(move || run_worker(weak, w))
                    .expect("the thread that runs the renderer's commands");
            }
            Shared {
                core: Mutex::new(core),
                events: Mutex::new(HashMap::new()),
                kernel: Mutex::new(Weak::new()),
                worker,
                callers: AtomicUsize::new(0),
                way: Mutex::new(()),
                way_clear: Condvar::new(),
                calls: std::sync::atomic::AtomicU64::new(0),
            }
        })
    }

    /// The driver, locked for a call.
    fn lock(&self) -> Core<'_> {
        self.callers.fetch_add(1, Ordering::AcqRel);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Core {
            guard: self.core.lock().unwrap_or_else(|e| e.into_inner()),
            shared: self,
        }
    }

    /// Wait until no call of the program is waiting for the driver: a short spin first, since
    /// most calls are over in a few microseconds, then sleeping until the last one says so.
    fn wait_for_calls(&self) {
        for _ in 0..400 {
            if self.callers.load(Ordering::Acquire) == 0 {
                return;
            }
            std::hint::spin_loop();
        }
        let mut way = self.way.lock().unwrap_or_else(|e| e.into_inner());
        while self.callers.load(Ordering::Acquire) > 0 {
            way = self.way_clear.wait(way).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Let the worker catch up when it is too far behind.
    fn wait_for_room(&self) {
        if let Some(w) = &self.worker {
            w.wait_for_room(false);
        }
    }

    /// Stop the run for a failure of the commands the worker ran.
    fn fail(&self, what: String) {
        let kernel = self
            .kernel
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .upgrade();
        match kernel {
            Some(kernel) => kernel.fatal(what),
            None => {
                eprintln!("kiln runtime: {what}");
                std::process::exit(1);
            }
        }
    }

    /// The (writable, readable) ends of the event behind `token`, made on first use.
    fn event(&self, kernel: &Kernel, token: EventToken) -> (KObject, KObject) {
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events
            .entry(token)
            .or_insert_with(|| kernel.new_event())
            .clone()
    }

    /// The title queued a buffer for display: tell the renderer (if there is one).
    pub fn present(&self, frame: &nx_gpu::display::Frame) -> Result<(), String> {
        let Some(worker) = &self.worker else {
            let mut core = self.lock();
            core.gpu.note_frame(frame.number);
            return core.gpu.present(frame);
        };
        // The buffer is shown after the commands queued before it have run.
        worker.wait_for_room(true);
        let request = {
            let mut core = self.lock();
            core.gpu.note_frame(frame.number);
            core.gpu.present_request(frame)?
        };
        if let Some(request) = request {
            worker.push(Task::Present(request));
        }
        Ok(())
    }

    /// Print what the command processor counted (`NX_TRACE_GPU`), once the process has ended.
    pub fn report(&self) {
        if let Some(w) = &self.worker {
            w.drain(Duration::from_secs(10));
        }
        let core = self.core.lock().unwrap_or_else(|e| e.into_inner());
        if core.gpu.level >= 1 && core.gpu.stats.submits > 0 {
            // `NX_GPU_TOP` is how many of the busiest methods the report lists (25 if unset, all
            // of them for 0).
            let top = match std::env::var("NX_GPU_TOP")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                Some(0) => usize::MAX,
                Some(n) => n,
                None => 25,
            };
            eprint!("{}", core.gpu.stats.report(top));
        }
    }

    /// Apply what the driver asked for since the last call to the kernel's events.
    fn deliver_signals(&self, kernel: &Kernel) {
        let signals = self
            .core
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take_signals();
        for s in signals {
            let (writable, _) = self.event(kernel, s.token);
            kernel.set_event(&writable, s.signaled);
        }
    }
}

/// The thread that runs what the queue was given, in order: a step of a submission at a time,
/// with the driver let go of between two steps so that the calls of the program get in.
fn run_worker(shared: Weak<Shared>, worker: Arc<Worker>) {
    // Where the thread's time goes, reported every few seconds with `NX_TRACE_PERF`.
    let trace = std::env::var_os("NX_TRACE_PERF").is_some();
    let mut spent = [Duration::ZERO; 4];
    let (mut steps, mut yields, mut tasks_run) = (0u64, 0u64, 0u64);
    let mut calls_before = 0u64;
    #[cfg(feature = "alloc_stats")]
    let mut allocs_before = (0u64, 0u64);
    let mut reported = std::time::Instant::now();
    loop {
        let waiting = std::time::Instant::now();
        let mut task = {
            let mut b = worker.lock();
            loop {
                if let Some(task) = b.tasks.pop_front() {
                    b.running = true;
                    break task;
                }
                b.running = false;
                worker.changed.notify_all();
                b = worker.changed.wait(b).unwrap_or_else(|e| e.into_inner());
            }
        };
        if trace {
            spent[0] += waiting.elapsed();
        }
        let Some(shared) = shared.upgrade() else {
            return;
        };
        let is_present = matches!(task, Task::Present(_));
        let outcome = loop {
            // A call of the program that is waiting for the driver goes first.
            let made_way = std::time::Instant::now();
            let mut yielded = false;
            if shared.callers.load(Ordering::Acquire) > 0 {
                yielded = true;
                shared.wait_for_calls();
            }
            let locking = std::time::Instant::now();
            let mut core = shared.core.lock().unwrap_or_else(|e| e.into_inner());
            let running = std::time::Instant::now();
            let stepped = core.gpu.run_task(&mut task);
            if trace {
                spent[1] += locking.duration_since(made_way);
                spent[2] += running.duration_since(locking);
                spent[3] += running.elapsed();
                steps += 1;
                yields += u64::from(yielded);
            }
            match stepped {
                Ok(None) => {}
                Ok(Some(done)) => {
                    core.wake(done.woken);
                    break Ok(());
                }
                Err(e) => break Err(e),
            }
        };
        tasks_run += 1;
        if trace && reported.elapsed() >= Duration::from_secs(2) {
            eprintln!(
                "[gpu thread] in the last {:.1} s: {} tasks, {} steps; no task to run {:.0} ms, made way {:.0} ms ({yields} times), waited for the driver {:.0} ms, ran {:.0} ms; the program called the driver {} times",
                reported.elapsed().as_secs_f64(),
                tasks_run,
                steps,
                spent[0].as_secs_f64() * 1000.0,
                spent[1].as_secs_f64() * 1000.0,
                spent[2].as_secs_f64() * 1000.0,
                spent[3].as_secs_f64() * 1000.0,
                shared.calls.load(Ordering::Relaxed) - calls_before
            );
            #[cfg(feature = "alloc_stats")]
            {
                {
                    let counts: Vec<String> = super::super::svc::SVC_COUNTS
                        .iter()
                        .enumerate()
                        .filter_map(|(i, c)| {
                            let n = c.swap(0, std::sync::atomic::Ordering::Relaxed);
                            (n > 0).then(|| {
                                format!(
                                    "{} {n}",
                                    super::super::svc::svc_name(i as u32).unwrap_or("?")
                                )
                            })
                        })
                        .collect();
                    eprintln!(
                        "[gpu thread] supervisor calls since the last report: {}",
                        counts.join(", ")
                    );
                }
                {
                    let reads = crate::rt::exports::COUNTER_READS
                        .swap(0, std::sync::atomic::Ordering::Relaxed);
                    let cycles = crate::rt::exports::COUNTER_CYCLES
                        .swap(0, std::sync::atomic::Ordering::Relaxed);
                    eprintln!(
                        "[gpu thread] the guest read the system counter {reads} times, {} cycles a read",
                        cycles / reads.max(1)
                    );
                }
                let (n, bytes) = nx_vulkan::alloc_stats::this_thread();
                eprintln!(
                    "[gpu thread] allocations since the last report: {} ({:.1} MB), {} a task",
                    n - allocs_before.0,
                    (bytes - allocs_before.1) as f64 / 1e6,
                    (n - allocs_before.0) / tasks_run.max(1)
                );
                allocs_before = (n, bytes);
            }
            spent = [Duration::ZERO; 4];
            (steps, yields, tasks_run) = (0, 0, 0);
            calls_before = shared.calls.load(Ordering::Relaxed);
            reported = std::time::Instant::now();
        }
        match outcome {
            Ok(()) => {
                let kernel = shared
                    .kernel
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .upgrade();
                if let Some(kernel) = kernel {
                    shared.deliver_signals(&kernel);
                }
            }
            Err(e) => shared.fail(match e {
                SubmitError::Nv(e) => format!("a submission failed with {e:?}"),
                SubmitError::Unimplemented(why) => why,
            }),
        }
        let mut b = worker.lock();
        if is_present {
            b.presents -= 1;
        }
        b.running = false;
        drop(b);
        worker.changed.notify_all();
    }
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shared")
    }
}

/// `nns::nvdrv::INvDrvServices`.
#[derive(Debug)]
pub struct NvDrv {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl NvDrv {
    pub fn new(kernel: Weak<Kernel>, shared: Arc<Shared>) -> NvDrv {
        *shared.kernel.lock().unwrap_or_else(|e| e.into_inner()) = kernel.clone();
        NvDrv { kernel, shared }
    }
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

fn stop(what: String) -> Stop {
    Stop(what)
}

impl NvDrv {
    fn core(&self) -> Core<'_> {
        self.shared.lock()
    }

    /// An ioctl with up to two input and two output buffers (`Ioctl`, `Ioctl2`, `Ioctl3`).
    fn ioctl(&mut self, req: &Request, variant: u32) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let (fd, code) = (req.u32_at(0), req.u32_at(4));
        let ioctl = nx_gpu::nvdrv::args::Ioctl(code);
        // The argument struct goes in the first input buffer when the number says it is sent, and
        // comes back in the first output buffer when it says it is read. Ioctl2 adds a second
        // input buffer; Ioctl3 a second output buffer.
        let input = if ioctl.has_input() {
            match req.input_buffer(0) {
                Some(b) if b.size > 0 => {
                    kernel.read_buffer(b, ioctl.size() as u64).ok_or_else(|| {
                        stop(format!("ioctl {code:#x}: the input buffer is unreadable"))
                    })?
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let extra_in = if variant == 11 {
            match req.input_buffer(1) {
                Some(b) if b.size > 0 => kernel.read_buffer(b, b.size).ok_or_else(|| {
                    stop(format!("ioctl {code:#x}: the second input is unreadable"))
                })?,
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        self.shared.wait_for_room();
        let reply = self
            .core()
            .ioctl(fd, code, &input, &extra_in)
            .map_err(|why| stop(format!("nvdrv: {why}")))?;
        self.shared.deliver_signals(&kernel);
        if ioctl.has_output() {
            if let Some(b) = req.output_buffer(0).filter(|b| b.size > 0) {
                let n = (b.size as usize).min(ioctl.size()).min(reply.args.len());
                if !kernel.write_memory(b.addr, &reply.args.data[..n]) {
                    return Err(stop(format!("ioctl {code:#x}: the output is unwritable")));
                }
            }
        }
        if variant == 12 {
            if let Some(b) = req.output_buffer(1).filter(|b| b.size > 0) {
                let n = (b.size as usize).min(reply.extra_out.len());
                if !kernel.write_memory(b.addr, &reply.extra_out[..n]) {
                    return Err(stop(format!(
                        "ioctl {code:#x}: the second output is unwritable"
                    )));
                }
            }
        }
        Ok(Response::ok().with_u32(reply.error.0))
    }
}

impl Service for NvDrv {
    fn name(&self) -> &str {
        "nns::nvdrv::INvDrvServices"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Open(buffer<path>) -> (u32 fd, u32 error)
            0 => {
                let Some(b) = req.input_buffer(0).filter(|b| b.size > 0) else {
                    return Ok(Response::ok()
                        .with_u32(0)
                        .with_u32(NvError::BAD_PARAMETER.0));
                };
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let bytes = kernel
                    .read_buffer(b, 0x100)
                    .ok_or_else(|| stop("nvdrv Open: the path is unreadable".to_string()))?;
                let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
                let path = String::from_utf8_lossy(&bytes[..end]).into_owned();
                let opened = self.core().open(&path);
                match opened {
                    Ok(fd) => Ok(Response::ok().with_u32(fd).with_u32(0)),
                    Err(Fail::Nv(e)) => Ok(Response::ok().with_u32(0).with_u32(e.0)),
                    Err(Fail::Unimplemented(why)) => Err(stop(format!("nvdrv Open: {why}"))),
                }
            }
            // Ioctl(u32 fd, u32 code, buffer in, buffer out) -> u32 error
            1 => self.ioctl(req, 1),
            // Close(u32 fd) -> u32 error
            2 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let e = self.core().close(req.u32_at(0));
                self.shared.deliver_signals(&kernel);
                Ok(Response::ok().with_u32(e.0))
            }
            // Initialize(u32 transfer memory size, process handle, transfer memory handle)
            // -> u32 error. The driver keeps its own books; the memory lent is not needed.
            3 => Ok(Response::ok().with_u32(0)),
            // QueryEvent(u32 fd, u32 event id) -> (u32 error, event handle)
            4 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let queried = self.core().query_event(req.u32_at(0), req.u32_at(4));
                match queried {
                    Ok(token) => {
                        let (_, readable) = self.shared.event(&kernel, token);
                        Ok(Response::ok().with_u32(0).with_copy_handle(readable))
                    }
                    Err(Fail::Nv(e)) => Ok(Response::ok().with_u32(e.0)),
                    Err(Fail::Unimplemented(why)) => Err(stop(format!("nvdrv QueryEvent: {why}"))),
                }
            }
            // SetAruid(u64 applet resource user id, pid) -> u32 error
            8 => Ok(Response::ok().with_u32(0)),
            // Ioctl2(u32 fd, u32 code, two input buffers, output buffer) -> u32 error
            11 => self.ioctl(req, 11),
            // Ioctl3(u32 fd, u32 code, input buffer, two output buffers) -> u32 error
            12 => self.ioctl(req, 12),
            // SetGraphicsFirmwareMemoryMarginEnabled(u64 enabled)
            13 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("nvdrv", other)),
        }
    }
}
