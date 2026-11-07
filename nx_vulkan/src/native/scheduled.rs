//! A logical queue can pause at a shared-memory event without blocking the API caller or
//! the device's physical queue. Each worker submits only its ready prefix.

use super::{Completion, Context, Queue, SharedMemory, SharedWord};
use crate::render::native::SurfaceCommand;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

pub enum Operation {
    WaitCompletion(Arc<Completion>),
    Gpu(SurfaceCommand),
    Prepare(Box<dyn FnOnce() -> Result<SurfaceCommand, String> + Send>),
    Wait {
        word: Arc<SharedWord>,
        mode: u32,
        value: u32,
    },
}
struct Job {
    serial: u64,
    operations: Vec<Operation>,
    done: Arc<Completion>,
    for_cpu: bool,
}
#[derive(Default)]
struct Observations {
    pages: Vec<Arc<SharedMemory>>,
    writers: BTreeMap<usize, Arc<SharedWord>>,
}
impl Observations {
    fn stamp(
        &mut self,
        queue: &mut Queue,
        operations: &mut [SurfaceCommand],
    ) -> Result<(), String> {
        let count = operations
            .iter()
            .filter(|op| matches!(op, SurfaceCommand::Signal(_)))
            .count();
        if count == 0 {
            return Ok(());
        }
        queue.reap()?;
        self.writers.retain(|_, done| done.load() != 1);
        let need = (count as u64 * 4).next_multiple_of(4096);
        let page = if let Some(page) = self
            .pages
            .iter()
            .find(|page| Arc::strong_count(page) == 1 && page.len() >= need)
        {
            Arc::clone(page)
        } else {
            if self.pages.len() >= 4096
                || self.pages.iter().map(|page| page.len()).sum::<u64>() + need > 16 * 1024 * 1024
            {
                return Err("native event observation storage budget exceeded".into());
            }
            let page = queue.context().owned_shared_memory(need)?;
            self.pages.push(Arc::clone(&page));
            page
        };
        let mut offset = 0;
        for operation in operations {
            if let SurfaceCommand::Signal(signal) = operation {
                let word = page.word(offset)?;
                offset += 4;
                word.reset();
                let address = signal.word.memory.host() + signal.word.offset as usize;
                self.writers.insert(address, Arc::clone(&word));
                *signal = signal.observed(word);
            }
        }
        Ok(())
    }
    fn wait(&mut self, word: &SharedWord, stop: &AtomicBool) -> Result<(), String> {
        let address = word.memory.host() + word.offset as usize;
        if let Some(done) = self.writers.remove(&address) {
            done.wait(0, 1, stop)?;
        }
        Ok(())
    }
}
#[derive(Default)]
struct State {
    jobs: VecDeque<Job>,
    submitted: u64,
    recorded: u64,
    operations: usize,
    failed: Option<String>,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    stop: AtomicBool,
    failed: AtomicBool,
    gpu_submissions: Arc<AtomicU64>,
}
#[derive(Clone)]
pub struct Recorded {
    shared: Arc<Shared>,
    serial: u64,
}
impl Recorded {
    /// Wait for the prefix to reach Vulkan submission, rather than unnecessarily waiting for
    /// its GPU fence. A window uses this before reading the surface's last-use completion.
    pub fn wait(&self, cancelled: impl Fn() -> bool) -> Result<(), String> {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.recorded < self.serial {
            if let Some(e) = &state.failed {
                return Err(e.clone());
            }
            if self.shared.stop.load(Ordering::Acquire) || cancelled() {
                return Err("native queue recording wait was cancelled".into());
            }
            state = self
                .shared
                .changed
                .wait_timeout(state, Duration::from_millis(25))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok(())
    }
}
pub struct ScheduledQueue {
    context: Context,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}
impl Context {
    pub fn scheduled_queue(&self) -> Result<ScheduledQueue, String> {
        let queue = self.queue()?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            stop: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            gpu_submissions: Arc::clone(&queue.submissions),
        });
        let work = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("nvn2-submit".into())
            .spawn(move || run(queue, work))
            .map_err(|e| format!("starting a native queue worker: {e}"))?;
        Ok(ScheduledQueue {
            context: self.clone(),
            shared,
            worker: Some(worker),
        })
    }
}
impl ScheduledQueue {
    pub fn context(&self) -> &Context {
        &self.context
    }
    pub fn submissions(&self) -> u64 {
        self.shared.gpu_submissions.load(Ordering::Acquire)
    }
    pub fn error(&self) -> Option<String> {
        if !self.shared.failed.load(Ordering::Acquire) {
            return None;
        }
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .failed
            .clone()
    }
    pub fn recorded(&self) -> Recorded {
        let state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        Recorded {
            shared: Arc::clone(&self.shared),
            serial: state.submitted,
        }
    }
    pub fn submit(
        &mut self,
        operations: Vec<Operation>,
        done: Arc<Completion>,
        for_cpu: bool,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&done.vk, &self.context.vk) {
            return Err("scheduled completion belongs to another device".into());
        }
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = &state.failed {
            return Err(e.clone());
        }
        if state.jobs.len() >= 4096 || state.operations.saturating_add(operations.len()) > 1 << 20 {
            return Err("native queue deferred-work budget exceeded".into());
        }
        let serial = state
            .submitted
            .checked_add(1)
            .ok_or("native submission identity exhausted")?;
        done.mark_enqueued()?;
        state.submitted = serial;
        state.operations += operations.len();
        let serial = state.submitted;
        state.jobs.push_back(Job {
            serial,
            operations,
            done,
            for_cpu,
        });
        self.shared.changed.notify_one();
        Ok(())
    }
    pub fn submit_marker(&mut self, done: Arc<Completion>, for_cpu: bool) -> Result<(), String> {
        self.submit(Vec::new(), done, for_cpu)
    }
}
impl Drop for ScheduledQueue {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.changed.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn send(
    queue: &mut Queue,
    observations: &mut Observations,
    operations: &mut Vec<SurfaceCommand>,
    done: Arc<Completion>,
    for_cpu: bool,
) -> Result<(), String> {
    if operations.is_empty() {
        queue.submit_marker(done, for_cpu)?;
    } else {
        observations.stamp(queue, operations)?;
        let surface_done = if for_cpu {
            queue.context().completion()?
        } else {
            Arc::clone(&done)
        };
        queue.submit_surfaces(operations, surface_done)?;
        if for_cpu {
            queue.submit_marker(done, true)?;
        }
        operations.clear();
    }
    Ok(())
}
fn execute(
    queue: &mut Queue,
    shared: &Shared,
    observations: &mut Observations,
    job: Job,
) -> Result<u64, String> {
    let mut operations = Vec::new();
    let result = (|| {
        for operation in job.operations {
            if shared.stop.load(Ordering::Acquire) {
                return Err("native queue retired before submission".into());
            }
            match operation {
                Operation::WaitCompletion(done) => {
                    if !operations.is_empty() {
                        let prefix = queue.context().completion()?;
                        send(queue, observations, &mut operations, prefix, false)?;
                    }
                    while !done.wait(1_000_000)? {
                        if shared.stop.load(Ordering::Acquire) {
                            return Err("native queue retired during a completion wait".into());
                        }
                    }
                }
                Operation::Gpu(operation) => operations.push(operation),
                Operation::Prepare(prepare) => operations.push(prepare()?),
                Operation::Wait { word, mode, value } => {
                    if !operations.is_empty() {
                        let done = queue.context().completion()?;
                        send(queue, observations, &mut operations, done, false)?;
                    }
                    observations.wait(&word, &shared.stop)?;
                    word.wait(mode, value, &shared.stop)?;
                }
            }
        }
        send(
            queue,
            observations,
            &mut operations,
            Arc::clone(&job.done),
            job.for_cpu,
        )
    })();
    if let Err(e) = &result {
        job.done.cancel(e);
    }
    result.map(|()| job.serial)
}
fn run(mut queue: Queue, shared: Arc<Shared>) {
    let mut observations = Observations::default();
    loop {
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.jobs.is_empty() && !shared.stop.load(Ordering::Acquire) {
            state = shared
                .changed
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        if shared.stop.load(Ordering::Acquire) {
            break;
        }
        let job = state.jobs.pop_front().expect("ready job");
        state.operations -= job.operations.len();
        drop(state);
        let completion = Arc::clone(&job.done);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute(&mut queue, &shared, &mut observations, job)
        }))
        .unwrap_or_else(|_| Err("native queue worker panicked during preparation".into()));
        if let Err(error) = &result {
            completion.cancel(error);
        }
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        match result {
            Ok(serial) => state.recorded = serial,
            Err(e) => {
                state.failed = Some(e);
                shared.failed.store(true, Ordering::Release);
                shared.stop.store(true, Ordering::Release);
            }
        }
        shared.changed.notify_all();
    }
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    for job in state.jobs.drain(..) {
        job.done
            .cancel("native queue retired before deferred submission");
    }
    shared.changed.notify_all();
}
