//! Syncpoints: the counters that fences are made of (Switchbrew, "NV services", `nvhost-ctrl`).
//!
//! A syncpoint is a 32-bit counter that only goes up. A fence is a syncpoint id and a value; it has
//! passed when the counter has reached the value, compared with wrap-around. The GPU increments a
//! channel's syncpoint as its commands finish; programs read it, wait for it, or ask for an event
//! when it reaches a threshold.
//!
//! The commands of a channel run to completion inside the call that submits them, so a syncpoint
//! the GPU would increment later is already at its final value when the call returns. What a
//! wait can still see is a threshold nothing will ever reach.

/// The number of syncpoints (docs/ASSUMPTIONS.md: 192, the Tegra X1 count).
pub const COUNT: usize = 192;

/// Has `value` reached `threshold`, counting modulo 2^32?
pub fn reached(value: u32, threshold: u32) -> bool {
    (value.wrapping_sub(threshold) as i32) >= 0
}

/// Somebody waiting for a syncpoint, identified by the caller's own token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Waiter {
    pub token: u32,
    pub id: u32,
    pub threshold: u32,
}

#[derive(Debug)]
pub struct Syncpoints {
    value: [u32; COUNT],
    /// The value the syncpoint will have once every submitted increment has happened.
    max: [u32; COUNT],
    reserved: [bool; COUNT],
    waiters: Vec<Waiter>,
}

impl Default for Syncpoints {
    fn default() -> Syncpoints {
        Syncpoints::new()
    }
}

impl Syncpoints {
    pub fn new() -> Syncpoints {
        // Syncpoint 0 is never handed out: a fence with id 0 and no value is "no fence".
        let mut reserved = [false; COUNT];
        reserved[0] = true;
        Syncpoints {
            value: [0; COUNT],
            max: [0; COUNT],
            reserved,
            waiters: Vec::new(),
        }
    }

    /// Reserve a syncpoint for a channel or a client.
    pub fn allocate(&mut self) -> Option<u32> {
        let id = self.reserved.iter().position(|r| !r)?;
        self.reserved[id] = true;
        self.value[id] = 0;
        self.max[id] = 0;
        Some(id as u32)
    }

    pub fn release(&mut self, id: u32) {
        if let Some(r) = self.reserved.get_mut(id as usize) {
            if id != 0 {
                *r = false;
            }
        }
    }

    pub fn is_valid(&self, id: u32) -> bool {
        (id as usize) < COUNT
    }

    pub fn value(&self, id: u32) -> Option<u32> {
        self.value.get(id as usize).copied()
    }

    pub fn max(&self, id: u32) -> Option<u32> {
        self.max.get(id as usize).copied()
    }

    /// Has the fence passed? An id past the table never has.
    pub fn is_reached(&self, id: u32, threshold: u32) -> bool {
        self.value(id).is_some_and(|v| reached(v, threshold))
    }

    /// Promise `by` more increments and return the value the syncpoint will have after them: the
    /// threshold of the fence that covers them.
    pub fn promise(&mut self, id: u32, by: u32) -> Option<u32> {
        let m = self.max.get_mut(id as usize)?;
        *m = m.wrapping_add(by);
        Some(*m)
    }

    /// Increment the counter by `by`. Returns the tokens of the waiters this satisfied, which are
    /// then no longer waiting. The promised maximum never lags behind the value.
    pub fn increment(&mut self, id: u32, by: u32) -> Vec<u32> {
        let Some(v) = self.value.get_mut(id as usize) else {
            return Vec::new();
        };
        *v = v.wrapping_add(by);
        let now = *v;
        if !reached(self.max[id as usize], now) {
            self.max[id as usize] = now;
        }
        self.take_ready(id)
    }

    /// Start waiting. Returns `false` (and records nothing) when the fence has already passed.
    pub fn wait(&mut self, w: Waiter) -> bool {
        if self.is_reached(w.id, w.threshold) {
            return false;
        }
        self.waiters.retain(|o| o.token != w.token);
        self.waiters.push(w);
        true
    }

    /// Stop waiting for `token`. Returns whether it was waiting.
    pub fn cancel(&mut self, token: u32) -> bool {
        let before = self.waiters.len();
        self.waiters.retain(|w| w.token != token);
        self.waiters.len() != before
    }

    fn take_ready(&mut self, id: u32) -> Vec<u32> {
        let value = self.value[id as usize];
        let mut ready = Vec::new();
        self.waiters.retain(|w| {
            if w.id == id && reached(value, w.threshold) {
                ready.push(w.token);
                false
            } else {
                true
            }
        });
        ready
    }

    pub fn waiting(&self) -> &[Waiter] {
        &self.waiters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reached_wraps_around() {
        assert!(reached(5, 5));
        assert!(reached(6, 5));
        assert!(!reached(4, 5));
        assert!(reached(2, u32::MAX - 1), "past the wrap");
        assert!(!reached(u32::MAX - 1, 2), "before the wrap");
    }

    #[test]
    fn allocation_skips_zero_and_reuses_released_ids() {
        let mut s = Syncpoints::new();
        let a = s.allocate().unwrap();
        let b = s.allocate().unwrap();
        assert_eq!((a, b), (1, 2));
        s.release(a);
        assert_eq!(s.allocate(), Some(1));
        s.release(0);
        assert_eq!(s.allocate(), Some(3), "zero stays reserved");
        for _ in 0..COUNT {
            s.allocate();
        }
        assert_eq!(s.allocate(), None);
    }

    #[test]
    fn promises_and_increments_track_value_and_max() {
        let mut s = Syncpoints::new();
        let id = s.allocate().unwrap();
        assert_eq!(s.promise(id, 2), Some(2));
        assert_eq!(s.promise(id, 1), Some(3));
        assert_eq!(s.value(id), Some(0));
        assert!(!s.is_reached(id, 3));
        s.increment(id, 3);
        assert!(s.is_reached(id, 3));
        // A CPU increment past the promise pushes the promise along.
        s.increment(id, 1);
        assert_eq!((s.value(id), s.max(id)), (Some(4), Some(4)));
        assert_eq!(s.promise(999, 1), None);
        assert!(!s.is_reached(999, 0));
    }

    #[test]
    fn waiters_wake_when_their_threshold_is_reached() {
        let mut s = Syncpoints::new();
        let id = s.allocate().unwrap();
        let other = s.allocate().unwrap();
        assert!(s.wait(Waiter {
            token: 10,
            id,
            threshold: 2
        }));
        assert!(s.wait(Waiter {
            token: 11,
            id,
            threshold: 1
        }));
        assert!(s.wait(Waiter {
            token: 12,
            id: other,
            threshold: 1
        }));
        assert_eq!(s.increment(id, 1), vec![11]);
        assert_eq!(s.waiting().len(), 2);
        assert_eq!(s.increment(id, 1), vec![10]);
        assert_eq!(s.waiting().len(), 1, "the other syncpoint's waiter stays");
        assert!(s.cancel(12));
        assert!(!s.cancel(12));
    }

    #[test]
    fn waiting_for_a_passed_fence_records_nothing() {
        let mut s = Syncpoints::new();
        let id = s.allocate().unwrap();
        s.increment(id, 4);
        assert!(!s.wait(Waiter {
            token: 1,
            id,
            threshold: 4
        }));
        assert!(s.waiting().is_empty());
    }

    #[test]
    fn a_token_waits_for_one_thing_at_a_time() {
        let mut s = Syncpoints::new();
        let id = s.allocate().unwrap();
        assert!(s.wait(Waiter {
            token: 1,
            id,
            threshold: 5
        }));
        assert!(s.wait(Waiter {
            token: 1,
            id,
            threshold: 7
        }));
        assert_eq!(s.waiting().len(), 1);
        assert_eq!(s.increment(id, 5), Vec::<u32>::new());
        assert_eq!(s.increment(id, 2), vec![1]);
    }
}
