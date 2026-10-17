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
