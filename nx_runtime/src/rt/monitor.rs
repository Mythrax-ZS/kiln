//! The global exclusive monitor behind `LDXR`/`STXR`.
//!
//! Each thread that has executed a load-exclusive and not yet resolved it has one *reservation*:
//! the address range and the bytes it saw. A store by another thread to an overlapping range
//! clears the reservation, as the architecture's global monitor does, and a store-exclusive
//! succeeds only while its thread's reservation for exactly that range is intact. As a second
//! line of defence the store is a compare-and-exchange against the bytes the reservation saw, so
//! a store by another thread that raced past the cheap "is anything armed" check (below) still
//! cannot be overwritten; the reservation is the model, the exchange only closes that race.
//!
//! A thread's own plain stores do not clear its reservation (the architecture does not require
//! it, and the interpreter does not do it); they update what the reservation saw, so the
//! exchange does not mistake them for another thread's write.
//!
//! The cost on the common path is one relaxed load per plain store: when no reservation exists
//! anywhere, stores do not take the lock. The residual race is a plain store that overlaps a
//! reservation being armed in the same instant; it is covered by the exchange unless the store
//! wrote the value already there (docs/ASSUMPTIONS.md A-021).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub addr: u64,
    pub bytes: u8,
    /// The bytes the reservation saw, little-endian (the second word for a 16-byte pair).
    pub observed: [u64; 2],
}

impl Reservation {
    fn overlaps(&self, addr: u64, bytes: u64) -> bool {
        self.addr < addr.saturating_add(bytes)
            && addr < self.addr.saturating_add(u64::from(self.bytes))
    }

    /// Account for a store by the reservation's own thread.
    fn apply_own_store(&mut self, addr: u64, bytes: u8, value: u128) {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&self.observed[0].to_le_bytes());
        buf[8..].copy_from_slice(&self.observed[1].to_le_bytes());
        let v = value.to_le_bytes();
        for i in 0..u64::from(bytes) {
            let at = addr.wrapping_add(i);
            if at >= self.addr && at < self.addr.saturating_add(u64::from(self.bytes)) {
                buf[(at - self.addr) as usize] = v[i as usize];
            }
        }
        self.observed[0] = u64::from_le_bytes(buf[..8].try_into().unwrap());
        self.observed[1] = u64::from_le_bytes(buf[8..].try_into().unwrap());
    }
}

pub struct Monitors {
    armed: AtomicUsize,
    state: Mutex<BTreeMap<u64, Reservation>>,
}

pub struct Locked<'a> {
    owner: &'a Monitors,
    map: MutexGuard<'a, BTreeMap<u64, Reservation>>,
}

impl Default for Monitors {
    fn default() -> Self {
        Self::new()
    }
}

impl Monitors {
    pub const fn new() -> Monitors {
        Monitors {
            armed: AtomicUsize::new(0),
            state: Mutex::new(BTreeMap::new()),
        }
    }

    /// Is any reservation outstanding? Plain stores take the lock only when this is true.
    pub fn any_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed) != 0
    }

    pub fn lock(&self) -> Locked<'_> {
        Locked {
            owner: self,
            map: self.state.lock().unwrap_or_else(|e| e.into_inner()),
        }
    }
}

impl Locked<'_> {
    pub fn get(&self, thread: u64) -> Option<Reservation> {
        self.map.get(&thread).copied()
    }

    pub fn arm(&mut self, thread: u64, r: Reservation) {
        self.map.insert(thread, r);
        self.sync();
    }

    pub fn disarm(&mut self, thread: u64) {
        self.map.remove(&thread);
        self.sync();
    }

    /// A store of `bytes` bytes at `addr` by `thread`: other threads' overlapping reservations
    /// end, the thread's own is updated with what it wrote.
    pub fn note_store(&mut self, thread: u64, addr: u64, bytes: u8, value: u128) {
        if let Some(own) = self.map.get_mut(&thread) {
            if own.overlaps(addr, u64::from(bytes)) {
                own.apply_own_store(addr, bytes, value);
            }
        }
        self.map
            .retain(|&t, r| t == thread || !r.overlaps(addr, u64::from(bytes)));
        self.sync();
    }

    fn sync(&self) {
        self.owner.armed.store(self.map.len(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(addr: u64, bytes: u8) -> Reservation {
        Reservation {
            addr,
            bytes,
            observed: [0; 2],
        }
    }

    #[test]
    fn a_store_by_another_thread_ends_overlapping_reservations_only() {
        let m = Monitors::new();
        assert!(!m.any_armed());
        let mut g = m.lock();
        g.arm(1, r(0x1000, 8));
        g.arm(2, r(0x1008, 4));
        g.arm(3, r(0x2000, 8));
        drop(g);
        assert!(m.any_armed());
        let mut g = m.lock();
        // Thread 9 writes one byte inside thread 1's range and next to thread 2's.
        g.note_store(9, 0x1007, 1, 0xFF);
        assert!(g.get(1).is_none());
        assert!(g.get(2).is_some() && g.get(3).is_some());
        g.note_store(9, 0x1008, 4, 0);
        assert!(g.get(2).is_none());
        drop(g);
        let mut g = m.lock();
        g.disarm(3);
        drop(g);
        assert!(!m.any_armed());
    }

    #[test]
    fn a_threads_own_stores_keep_its_reservation_and_update_what_it_saw() {
        let m = Monitors::new();
        let mut g = m.lock();
        g.arm(1, r(0x1000, 8));
        g.note_store(1, 0x1002, 2, 0xBBAA);
        let mine = g.get(1).expect("own store keeps the reservation");
        assert_eq!(mine.observed[0], 0xBBAA_0000);
        // A store outside the range changes nothing.
        g.note_store(1, 0x1010, 8, u128::MAX);
        assert_eq!(g.get(1).unwrap().observed[0], 0xBBAA_0000);
    }

    #[test]
    fn pair_reservations_track_both_words() {
        let mut res = r(0x1000, 16);
        res.apply_own_store(0x1008, 8, 0x1122_3344_5566_7788);
        assert_eq!(res.observed, [0, 0x1122_3344_5566_7788]);
    }
}
