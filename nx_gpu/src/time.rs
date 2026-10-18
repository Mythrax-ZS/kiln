//! Sleeping to a moment, as exactly as the machine allows (docs/ASSUMPTIONS.md A-186).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How late the system's sleep has been returning, in microseconds (a running average).
static LATE_US: AtomicU64 = AtomicU64::new(1000);

/// Sleep until `deadline`: the system's sleep, which returns a millisecond or so late, until a
/// margin before the moment, then yielding until it. The margin follows how late the system's
/// sleeps have been returning.
pub fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let remaining = deadline - now;
        let margin = Duration::from_micros(LATE_US.load(Ordering::Relaxed) + 100);
        if remaining > margin {
            let asked = remaining - margin;
            let began = Instant::now();
            std::thread::sleep(asked);
            let late = began.elapsed().saturating_sub(asked).as_micros() as u64;
            let old = LATE_US.load(Ordering::Relaxed);
            let new = ((old * 7 + late.min(10_000)) / 8).clamp(100, 5_000);
            LATE_US.store(new, Ordering::Relaxed);
        } else {
            std::thread::yield_now();
        }
    }
}

/// Give the calling thread a priority above the normal one, once, so that the load of other
/// programs does not make it late: for the threads that keep time (the display's vsync, the
/// audio renderer's frame), draw and present, and decode video. On Windows this is
/// `THREAD_PRIORITY_ABOVE_NORMAL`; elsewhere nothing is done.
pub fn raise_priority() {
    thread_local! {
        static RAISED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    let first = !RAISED.with(|r| r.replace(true));
    #[cfg(windows)]
    if first {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(thread: isize, priority: i32) -> i32;
        }
        const ABOVE_NORMAL: i32 = 1;
        // SAFETY: plain calls on the pseudo-handle of the calling thread.
        unsafe {
            SetThreadPriority(GetCurrentThread(), ABOVE_NORMAL);
        }
    }
    #[cfg(not(windows))]
    let _ = first;
}

/// How busy the whole machine is: the share of all processors' time that was not idle since the
/// last question, from the system's own counters (Windows; `None` elsewhere). For the reports of
/// `NX_TRACE_PERF`, to tell a run slowed by other programs from one slowed by this one.
#[derive(Default)]
pub struct MachineLoad {
    last: Option<(u64, u64)>,
}

impl MachineLoad {
    pub fn new() -> MachineLoad {
        let mut load = MachineLoad::default();
        let _ = load.busy_percent();
        load
    }

    /// The percentage of the machine's processor time that was busy since the last call.
    pub fn busy_percent(&mut self) -> Option<f64> {
        let (idle, total) = system_times()?;
        let out = self
            .last
            .map(|(i0, t0)| 100.0 * (1.0 - (idle - i0) as f64 / (total - t0).max(1) as f64));
        self.last = Some((idle, total));
        out
    }
}

/// (idle, idle + kernel + user) of all processors, in 100 ns units.
#[cfg(windows)]
fn system_times() -> Option<(u64, u64)> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetSystemTimes(idle: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
    }
    let (mut idle, mut kernel, mut user) = (0u64, 0u64, 0u64);
    // SAFETY: three out-parameters of the sizes the call writes (FILETIME is 64 bits).
    let ok = unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) };
    // The kernel time counts the idle time too.
    (ok != 0).then_some((idle, kernel + user))
}

#[cfg(not(windows))]
fn system_times() -> Option<(u64, u64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sleep_ends_at_its_moment_not_a_timer_tick_later() {
        // Twenty sleeps of 3 ms, each from the end of the one before: never early, and together
        // well under the 20 ticks of a coarse timer they would take if each were a millisecond
        // late.
        let began = Instant::now();
        for _ in 0..20 {
            let at = Instant::now();
            sleep_until(at + Duration::from_millis(3));
            assert!(
                at.elapsed() >= Duration::from_millis(3),
                "{:?}",
                at.elapsed()
            );
        }
        let took = began.elapsed();
        assert!(took < Duration::from_millis(75), "{took:?}");
    }
}
