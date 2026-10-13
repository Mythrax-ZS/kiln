//! A model of the console's cores for the guest's threads (`NX_CORES`).
//!
//! By default the guest's threads run at once on host threads, as many as the host has cores. On
//! the console a core runs one thread at a time and keeps running it until it blocks, yields or
//! gives way to a thread of a higher priority; threads of the same priority on one core never
//! overlap, and a program written for that may rely on it. With `NX_CORES=per-core` a thread has
//! to hold the core it is assigned to (its ideal core, as `SetThreadCoreMask` last set it) to run
//! guest code, and gives it up for the supervisor calls that can block or reschedule; the threads
//! waiting for a core are served in the order they asked. `NX_CORES=one` puts every thread on one
//! core. Priorities are not modelled (docs/ASSUMPTIONS.md).

use std::sync::{Condvar, Mutex};

/// One core: a queue of tickets. The thread whose ticket is being served owns the core.
#[derive(Default)]
struct Core {
    /// `(next ticket, ticket being served)`.
    tickets: Mutex<(u64, u64)>,
    turn: Condvar,
}

pub struct Cores {
    cores: [Core; 4],
    one: bool,
}

/// Does the supervisor call `imm` leave the core free for others to run? The ones that only read
/// something never block, and a thread that makes them keeps its core.
pub fn gives_way(imm: u32) -> bool {
    !matches!(
        imm,
        0x06 | 0x0C | 0x0E | 0x10 | 0x16 | 0x1E | 0x24 | 0x25 | 0x27 | 0x29
    )
}

impl Cores {
    /// The model `NX_CORES` asks for, if any.
    pub fn from_env() -> Option<Cores> {
        match std::env::var("NX_CORES").ok()?.as_str() {
            "one" => Some(Cores::new(true)),
            "per-core" => Some(Cores::new(false)),
            _ => None,
        }
    }

    fn new(one: bool) -> Cores {
        Cores {
            cores: Default::default(),
            one,
        }
    }

    /// The core a thread whose ideal core is `ideal` runs on.
    pub fn core_of(&self, ideal: i32) -> usize {
        if self.one {
            0
        } else {
            ideal.clamp(0, 3) as usize
        }
    }

    /// Wait for the core, behind the threads that asked before.
    pub fn acquire(&self, core: usize) {
        let c = &self.cores[core];
        let mut t = c.tickets.lock().unwrap_or_else(|e| e.into_inner());
        let mine = t.0;
        t.0 += 1;
        while t.1 != mine {
            t = c.turn.wait(t).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Give the core up to the next thread in line.
    pub fn release(&self, core: usize) {
        let c = &self.cores[core];
        c.tickets.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
        c.turn.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn threads_hold_a_core_one_at_a_time_in_the_order_they_asked() {
        let cores = Arc::new(Cores::new(true));
        cores.acquire(0);
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut joins = Vec::new();
        for n in 0..3 {
            let (held, log) = (Arc::clone(&cores), Arc::clone(&log));
            joins.push(std::thread::spawn(move || {
                held.acquire(0);
                log.lock().unwrap().push(n);
                held.release(0);
            }));
            // The threads ask in the order 0, 1, 2.
            while cores.cores[0].tickets.lock().unwrap().0 < n + 2 {
                std::thread::yield_now();
            }
        }
        assert!(log.lock().unwrap().is_empty(), "the holder has not let go");
        cores.release(0);
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(*log.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn a_thread_per_core_model_keeps_cores_apart_and_one_core_joins_them() {
        let per_core = Cores::new(false);
        assert_eq!(per_core.core_of(2), 2);
        assert_eq!(per_core.core_of(-1), 0, "no preference: the first core");
        assert_eq!(per_core.core_of(9), 3);
        per_core.acquire(0);
        per_core.acquire(1);
        per_core.release(0);
        per_core.release(1);
        assert_eq!(Cores::new(true).core_of(2), 0);
    }

    #[test]
    fn reading_calls_keep_the_core_and_blocking_ones_give_it_up() {
        assert!(!gives_way(0x1E), "GetSystemTick");
        assert!(gives_way(0x0B), "SleepThread");
        assert!(gives_way(0x1A), "ArbitrateLock");
        assert!(gives_way(0x21), "SendSyncRequest");
    }
}
