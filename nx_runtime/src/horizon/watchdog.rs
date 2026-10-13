//! A watchdog for runs that stop presenting frames (`NX_WATCHDOG=<seconds>`).
//!
//! When no frame has been presented for that long it says, on standard error, what every guest
//! thread is doing (inside which supervisor call and for how long, or running guest code, with the
//! return addresses as of its last call: generated code keeps its registers in variables and
//! stores them in the CPU structure at calls) and how far the command processor has got, then
//! again every `seconds` while the run stays stuck. `recomp symbols` names the addresses.

use super::kernel::Kernel;
use super::svc::svc_name;
use crate::rt::abi::NativeCpu;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Milliseconds since the first call: the clock the monitors stamp supervisor calls with.
fn now_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// What one guest thread is doing, readable from the watchdog's thread.
pub struct ThreadMonitor {
    /// The supervisor call the thread is in, plus one; 0 while it runs guest code.
    svc: AtomicU32,
    /// The first arguments of that call.
    args: [AtomicU64; 3],
    /// When the call began (`now_ms`).
    since_ms: AtomicU64,
    /// What the call is about, when the kernel knows (the service and command of a request).
    detail: Mutex<String>,
    detailed: AtomicBool,
    /// The address of the thread's CPU structure while the thread runs (0 otherwise); the lock
    /// keeps the structure from being freed while the watchdog reads it.
    cpu: Mutex<usize>,
}

impl ThreadMonitor {
    pub fn new(cpu: &mut NativeCpu) -> ThreadMonitor {
        ThreadMonitor {
            svc: AtomicU32::new(0),
            args: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            since_ms: AtomicU64::new(0),
            detail: Mutex::new(String::new()),
            detailed: AtomicBool::new(false),
            cpu: Mutex::new(cpu as *mut NativeCpu as usize),
        }
    }

    /// The thread enters supervisor call `imm`.
    pub fn enter(&self, imm: u32, x: &[u64; 31]) {
        for (slot, value) in self.args.iter().zip(x) {
            slot.store(*value, Ordering::Relaxed);
        }
        self.since_ms.store(now_ms(), Ordering::Relaxed);
        self.svc.store(imm + 1, Ordering::Release);
    }

    /// The thread is back in guest code.
    pub fn leave(&self) {
        self.svc.store(0, Ordering::Release);
        if self.detailed.swap(false, Ordering::Relaxed) {
            self.detail
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
        }
    }

    /// Say what the call the thread is in is about.
    pub fn set_detail(&self, text: String) {
        *self.detail.lock().unwrap_or_else(|e| e.into_inner()) = text;
        self.detailed.store(true, Ordering::Relaxed);
    }

    /// The thread has ended: its CPU structure is about to go.
    pub fn retire(&self) {
        *self.cpu.lock().unwrap_or_else(|e| e.into_inner()) = 0;
    }
}

/// `NX_WATCHDOG`: how long without a frame is too long, if the watchdog is wanted.
pub fn requested() -> Option<Duration> {
    let seconds: u64 = std::env::var("NX_WATCHDOG").ok()?.trim().parse().ok()?;
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

/// Start the watchdog of `kernel`.
pub fn spawn(kernel: Arc<Kernel>, limit: Duration) {
    let _ = std::thread::Builder::new()
        .name("watchdog".to_string())
        .spawn(move || watch(&kernel, limit));
}

fn watch(kernel: &Kernel, limit: Duration) {
    let mut frames = super::input::frames();
    let mut moved = Instant::now();
    let mut last_report: Option<Instant> = None;
    let (mut methods, mut commands) = (0u64, 0u64);
    loop {
        std::thread::sleep(Duration::from_millis(500));
        if kernel.exit_state().is_some() {
            return;
        }
        let now_frames = super::input::frames();
        if now_frames != frames {
            frames = now_frames;
            moved = Instant::now();
            last_report = None;
            continue;
        }
        let quiet = moved.elapsed();
        if quiet < limit || last_report.is_some_and(|t| t.elapsed() < limit) {
            continue;
        }
        last_report = Some(Instant::now());
        let (m, c) = (
            nx_gpu::progress::METHODS.load(Ordering::Relaxed),
            nx_gpu::progress::COMMANDS.load(Ordering::Relaxed),
        );
        eprintln!(
            "[watchdog] no frame was presented for {:.0} s (frame {frames}); the command processor \
             has handled {m} methods (+{} since the last look) and {c} draws and clears (+{}); the GPU \n             side is {}",
            quiet.as_secs_f64(),
            m - methods,
            c - commands,
            nx_gpu::progress::phase()
        );
        (methods, commands) = (m, c);
        report_threads(kernel, "watchdog");
    }
}

/// Say what every running guest thread is doing, one line each, starting with `[label]`.
pub(super) fn report_threads(kernel: &Kernel, label: &str) {
    let monitors: Vec<(u64, Arc<ThreadMonitor>)> = kernel
        .monitors()
        .iter()
        .map(|(id, m)| (*id, Arc::clone(m)))
        .collect();
    let now = now_ms();
    for (id, monitor) in monitors {
        let (entry, arg, priority, core) = kernel.thread_entry(id).unwrap_or((0, 0, 0, 0));
        let svc = monitor.svc.load(Ordering::Acquire);
        let what = if svc > 0 {
            let imm = svc - 1;
            let detail = monitor
                .detail
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            format!(
                "in {} ({imm:#x}) for {:.1} s, x0..x2 = {:#x} {:#x} {:#x}{}",
                svc_name(imm).unwrap_or("an unknown call"),
                now.saturating_sub(monitor.since_ms.load(Ordering::Relaxed)) as f64 / 1000.0,
                monitor.args[0].load(Ordering::Relaxed),
                monitor.args[1].load(Ordering::Relaxed),
                monitor.args[2].load(Ordering::Relaxed),
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(" [{detail}]")
                }
            )
        } else {
            "running guest code".to_string()
        };
        let guard = monitor.cpu.lock().unwrap_or_else(|e| e.into_inner());
        let stack = if *guard == 0 {
            "its CPU is gone".to_string()
        } else {
            // SAFETY: the address is of the thread's CPU structure, which stays alive while the
            // lock is held (`retire` takes it before the thread lets go of the structure); the
            // thread may be writing it, and a torn read only makes the report wrong.
            let cpu = unsafe { &*(*guard as *const NativeCpu) };
            let frames: Vec<String> = kernel
                .guest_call_stack(cpu)
                .iter()
                .map(|a| format!("{a:#x}"))
                .collect();
            format!(
                "pc {:#x} at its last call, return addresses {}",
                cpu.state.pc,
                frames.join(" ")
            )
        };
        drop(guard);
        eprintln!(
            "[{label}]   thread {id} (entry {entry:#x}, argument {arg:#x}, priority {priority}, \
             core {core}): {what}; {stack}"
        );
    }
}
