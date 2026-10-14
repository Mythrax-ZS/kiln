//! A counting allocator for measuring (`--features alloc_stats`): how many allocations each
//! thread makes, so that the work a hot path does with the heap can be counted per frame. It
//! changes nothing else.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

pub struct Counting;

// SAFETY: every call is the system allocator's; the counters are thread local cells that are
// never touched through anything that allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + layout.size() as u64));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + layout.size() as u64));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + new_size as u64));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// The allocations and the bytes asked for by this thread so far.
pub fn this_thread() -> (u64, u64) {
    (
        ALLOCATIONS.try_with(Cell::get).unwrap_or(0),
        BYTES.try_with(Cell::get).unwrap_or(0),
    )
}

#[global_allocator]
static COUNTING: Counting = Counting;

thread_local! {
    static LAST: Cell<u64> = const { Cell::new(0) };
    static LAPS: Cell<[(&'static str, u64); 24]> = const { Cell::new([("", 0); 24]) };
}

/// The allocations this thread made since the last `lap`, added to the step called `name`.
pub fn lap(name: &'static str) {
    let now = this_thread().0;
    let delta = LAST.try_with(|l| now - l.replace(now)).unwrap_or(0);
    let _ = LAPS.try_with(|laps| {
        let mut all = laps.get();
        for entry in all.iter_mut() {
            if entry.0 == name || entry.0.is_empty() {
                entry.0 = name;
                entry.1 += delta;
                break;
            }
        }
        laps.set(all);
    });
}

/// The steps and their allocations since the last call, most first.
pub fn take_laps() -> Vec<(&'static str, u64)> {
    let all = LAPS
        .try_with(|laps| laps.replace([("", 0); 24]))
        .unwrap_or([("", 0); 24]);
    let mut v: Vec<(&'static str, u64)> = all.into_iter().filter(|e| !e.0.is_empty()).collect();
    v.sort_by_key(|e| std::cmp::Reverse(e.1));
    v
}
