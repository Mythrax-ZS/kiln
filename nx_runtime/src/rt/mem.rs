//! Guest memory behind one checked interface.
//!
//! A 39-bit guest address space (the Horizon process address space) is translated through a
//! two-level table of 4 KiB pages: the top level has one slot per 2 MiB, each slot points to 512
//! page entries, and an entry is the address of the page's host bytes with the permissions in the
//! low bits. Every guest access goes through [`GuestMemory::read`] or [`GuestMemory::write`]
//! (or the byte-copy helpers that bypass permissions for the loader and tests), so an unmapped or
//! protected access is reported as a fault instead of touching host memory.
//!
//! Aligned accesses of 1, 2, 4 and 8 bytes use atomic loads and stores, which gives them the
//! single-copy atomicity the architecture promises; unaligned and 16-byte accesses are copies.
//! Entries are atomics so a thread can be remapping memory while others run.
//!
//! Limits: the table is not shrunk when pages are unmapped, and the host bytes of an unmapped page
//! are not freed until the process ends (aliasing pages through `svcMapMemory` shares host bytes,
//! so freeing needs the kernel's bookkeeping, which belongs to a later milestone).

use super::abi::perm;
use crate::a64::bus::FaultKind;
use crate::a64::inst::AtomicOp;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::ptr;
use std::sync::atomic::{
    AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicPtr, AtomicU16, AtomicU32, AtomicU64,
    AtomicU8, Ordering,
};
use std::sync::Mutex;

pub const PAGE_SIZE: u64 = 4096;
pub const ADDRESS_BITS: u32 = 39;
/// The first level is indexed by the address bits from this one up, the second by the page bits
/// below it: `L2_ENTRIES` pages of `PAGE_SIZE`.
const L1_SHIFT: u32 = 21;
const L1_ENTRIES: usize = 1 << (ADDRESS_BITS - L1_SHIFT);
const L2_ENTRIES: usize = 512;

const PRESENT: u64 = 1 << 3;
const READ: u64 = 1;
const WRITE: u64 = 2;
const EXEC: u64 = 4;
/// The page is one `NX_TRAP_WRITES` asks about: it is writable but its entry does not say so, so
/// that every write takes the checked path, which reports it ([`super::trap`]).
const TRAP: u64 = 1 << 4;
/// The page's writes are tracked (A-126): its entry's write bit is taken away while it is
/// watched, so that the first write to it goes the checked way, which stamps the page and gives
/// the bit back.
const WATCH: u64 = 1 << 5;
/// A watched page's permissions allow writes (its entry's write bit is away while it is watched).
const WAS_W: u64 = 1 << 6;
const FLAGS: u64 = 0xFFF;

/// The entry of a watched page as it is when the page is no longer watched.
fn unwatched(e: u64) -> u64 {
    let mut n = e & !(WATCH | WAS_W);
    if e & WAS_W != 0 && e & TRAP == 0 {
        n |= WRITE;
    }
    n
}

/// The entry of a page that is to be watched, or `None` when it is not a page or is watched.
fn watched(e: u64) -> Option<u64> {
    if e & PRESENT == 0 || e & WATCH != 0 {
        return None;
    }
    Some(if e & WRITE != 0 {
        (e & !WRITE) | WATCH | WAS_W
    } else if e & TRAP != 0 {
        e | WATCH | WAS_W
    } else {
        e | WATCH
    })
}

type L2 = [AtomicU64; L2_ENTRIES];

/// The first level of the process's guest memory (the one in [`rt`](super::rt)), null until the
/// runtime has made it. The access functions generated code calls read it from here, without
/// going through `rt()`, to reach a page in a few instructions ([`fast_ptr`]).
#[no_mangle]
#[allow(non_upper_case_globals)]
pub static nx_mem_l1: AtomicPtr<AtomicPtr<L2>> = AtomicPtr::new(ptr::null_mut());

/// The entry flag a fast read needs (a readable page is a present one).
pub const FAST_READ: u64 = READ;
/// The entry flag a fast write needs.
pub const FAST_WRITE: u64 = WRITE;

/// The host address of the `size` bytes (1, 2, 4 or 8) at `addr` in the process's guest memory
/// when one aligned access reaches them and the page allows it (`need`: [`FAST_READ`] or
/// [`FAST_WRITE`]). `None` for anything else, which the slow path then handles: an address
/// beyond the 39 bits, an unaligned one (it may straddle pages), an unmapped page or a missing
/// permission.
#[inline(always)]
pub fn fast_ptr(addr: u64, size: u64, need: u64) -> Option<*mut u8> {
    if addr & (size - 1) != 0 || addr >> ADDRESS_BITS != 0 {
        return None;
    }
    let l1 = nx_mem_l1.load(Ordering::Acquire);
    if l1.is_null() {
        return None;
    }
    // SAFETY: the address is below 2^39, so the index is inside the first level, which is never
    // freed once published.
    let l2 = unsafe { (*l1.add((addr >> 21) as usize)).load(Ordering::Acquire) };
    if l2.is_null() {
        return None;
    }
    // SAFETY: a published second level is a leaked box of 512 entries.
    let entry = unsafe {
        (*l2.cast::<AtomicU64>().add(((addr >> 12) & 511) as usize)).load(Ordering::Relaxed)
    };
    if entry & need == 0 {
        return None;
    }
    Some(((entry & !FLAGS) | (addr & (PAGE_SIZE - 1))) as *mut u8)
}

/// Like [`fast_ptr`] for an access that may be unaligned but lies within one page: the host
/// address of its bytes and whether it is aligned (an aligned access is made atomically, an
/// unaligned one as the host does it, which may tear, as the architecture allows an unaligned
/// access to). `None` for what the slow path handles: one that straddles two pages, an
/// address beyond the space, an unmapped page or a missing permission. The copies a program
/// makes are often unaligned by a few bytes; sent through the slow path (a second walk of the
/// table and a byte-wise assembly), they held the third title's opening cutscene at 31 ms a
/// frame.
pub fn fast_ptr_any(addr: u64, size: u64, need: u64) -> Option<(*mut u8, bool)> {
    let offset = addr & (PAGE_SIZE - 1);
    if offset + size > PAGE_SIZE || addr >> ADDRESS_BITS != 0 {
        return None;
    }
    let l1 = nx_mem_l1.load(Ordering::Acquire);
    if l1.is_null() {
        return None;
    }
    // SAFETY: as in `fast_ptr`.
    let l2 = unsafe { (*l1.add((addr >> L1_SHIFT) as usize)).load(Ordering::Acquire) };
    if l2.is_null() {
        return None;
    }
    // SAFETY: as in `fast_ptr`.
    let entry = unsafe {
        (*l2.cast::<AtomicU64>()
            .add(((addr >> 12) & (L2_ENTRIES as u64 - 1)) as usize))
        .load(Ordering::Relaxed)
    };
    if entry & need == 0 {
        return None;
    }
    Some((
        ((entry & !FLAGS) | offset) as *mut u8,
        addr & (size - 1) == 0,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    /// Address or size not a multiple of the page size, or outside the address space.
    Range,
    /// Part of the range is already mapped.
    Overlap,
    /// The host could not provide the memory.
    OutOfMemory,
}

struct Block {
    ptr: *mut u8,
    layout: Layout,
    /// Allocated from the system with write tracking (see [`Writes`]).
    watched: bool,
}

pub struct GuestMemory {
    l1: Box<[AtomicPtr<L2>]>,
    blocks: Mutex<Vec<Block>>,
    writes: Mutex<Writes>,
    traps: super::trap::Traps,
}

/// The smallest block of memory allocated from the system directly (a large one is page-aligned
/// and zeroed by the system).
const WATCH_MIN: usize = 1 << 20;

/// Which pages of memory were written (docs/ASSUMPTIONS.md A-126), as the page table itself sees
/// it: a watched page's entry has its write bit taken away ([`WATCH`]), so that the first write to
/// it goes the checked way, which stamps the page and gives the bit back, and the pages written
/// in an epoch are watched again when the next one begins. Time is cut into epochs: each call of
/// [`GuestMemory::begin_epoch`] ends one and starts the next, and a page written during an epoch
/// is stamped with the number of the epoch that follows it. A reader that looked at the bytes of
/// a range after epoch `e` began knows they are as it saw them if no page of the range has a
/// stamp above `e`.
#[derive(Default)]
struct Writes {
    epoch: u32,
    /// The chunks whose pages are watched, by number: the epoch each was watched from, and the
    /// epoch it was last asked about in. Only the chunks somebody asked about lately are watched.
    chunks: HashMap<u64, (u32, u32)>,
    /// The stamps of the pages of each chunk, by chunk.
    stamps: HashMap<u64, Box<[u32; 512]>>,
    /// The pages written since the epoch began, to be watched again when the next one begins.
    written: Vec<u64>,
    /// For the trace (`NX_TRACE_PERF`): the pages found written, and the chunks watched from a
    /// question, since the last line.
    pages: usize,
    begun: usize,
}

/// A chunk: the 512 pages of a second-level table, watched or not as a whole.
const CHUNK_SHIFT: u32 = L1_SHIFT;

/// The epochs a chunk nobody asks about stays watched. A chunk let go is no longer tracked: the
/// next question about it watches it again and is answered "not known" (A-277).
const IDLE_EPOCHS: u32 = 60;

// SAFETY: the raw pointers in `blocks` are host allocations owned by this value; guest data races
// are the guest program's business, like on hardware.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

impl Default for GuestMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GuestMemory {
    fn drop(&mut self) {
        for slot in self.l1.iter() {
            let p = slot.load(Ordering::Relaxed);
            if !p.is_null() {
                // SAFETY: allocated by `Box::leak` in `l2_for_map`.
                drop(unsafe { Box::from_raw(p) });
            }
        }
        for b in self.blocks.get_mut().unwrap().drain(..) {
            if b.watched {
                // SAFETY: allocated by `tracking::alloc` in `map`.
                unsafe { tracking::free(b.ptr) };
            } else {
                // SAFETY: allocated with this layout in `map`.
                unsafe { dealloc(b.ptr, b.layout) };
            }
        }
    }
}

fn flags_of(p: u32) -> u64 {
    let mut f = PRESENT;
    if p & perm::R != 0 {
        f |= READ;
    }
    if p & perm::W != 0 {
        f |= WRITE;
    }
    if p & perm::X != 0 {
        f |= EXEC;
    }
    f
}

impl GuestMemory {
    pub fn new() -> GuestMemory {
        Self::with_traps(super::trap::Traps::from_env())
    }

    /// Memory that reports the writes to the ranges of `traps`.
    pub(crate) fn with_traps(traps: super::trap::Traps) -> GuestMemory {
        let l1: Vec<AtomicPtr<L2>> = std::iter::repeat_with(|| AtomicPtr::new(ptr::null_mut()))
            .take(L1_ENTRIES)
            .collect();
        GuestMemory {
            l1: l1.into_boxed_slice(),
            blocks: Mutex::new(Vec::new()),
            writes: Mutex::new(Writes {
                epoch: 1,
                ..Writes::default()
            }),
            traps,
        }
    }

    /// The flags of the entry of the page at `addr` that is to have `flags`: a writable page that a
    /// trap covers is made a trapped one.
    fn entry_flags(&self, addr: u64, flags: u64) -> u64 {
        if flags & WRITE != 0 && !self.traps.is_empty() && self.traps.covers(addr) {
            (flags & !WRITE) | TRAP
        } else {
            flags
        }
    }

    /// Make this the memory [`fast_ptr`] looks in. Only the process's own memory does this.
    pub(crate) fn publish(&self) {
        nx_mem_l1.store(self.l1.as_ptr().cast_mut(), Ordering::Release);
    }

    fn l2(&self, addr: u64) -> Option<&L2> {
        let p = self.l1.get((addr >> 21) as usize)?.load(Ordering::Acquire);
        // SAFETY: non-null pointers come from `Box::leak` and live until `drop`.
        unsafe { p.as_ref() }
    }

    fn l2_for_map(&self, addr: u64) -> &L2 {
        let slot = &self.l1[(addr >> 21) as usize];
        let p = slot.load(Ordering::Acquire);
        if !p.is_null() {
            // SAFETY: as above.
            return unsafe { &*p };
        }
        let fresh: Box<L2> = Box::new(std::array::from_fn(|_| AtomicU64::new(0)));
        let raw = Box::into_raw(fresh);
        match slot.compare_exchange(ptr::null_mut(), raw, Ordering::AcqRel, Ordering::Acquire) {
            // SAFETY: just published, never freed before `drop`.
            Ok(_) => unsafe { &*raw },
            Err(existing) => {
                // SAFETY: lost the race; `raw` was never shared.
                drop(unsafe { Box::from_raw(raw) });
                // SAFETY: published by the winner.
                unsafe { &*existing }
            }
        }
    }

    fn entry(&self, addr: u64) -> u64 {
        if addr >> ADDRESS_BITS != 0 {
            return 0;
        }
        match self.l2(addr) {
            Some(t) => t[((addr >> 12) & 511) as usize].load(Ordering::Relaxed),
            None => 0,
        }
    }

    fn check_range(addr: u64, size: u64) -> Result<(), MapError> {
        let end = addr.checked_add(size).ok_or(MapError::Range)?;
        if !addr.is_multiple_of(PAGE_SIZE)
            || !size.is_multiple_of(PAGE_SIZE)
            || size == 0
            || end > 1 << ADDRESS_BITS
        {
            return Err(MapError::Range);
        }
        Ok(())
    }

    /// The bytes of host memory behind guest pages, in all.
    fn mapped_bytes(&self) -> usize {
        let blocks = self.blocks.lock().unwrap_or_else(|e| e.into_inner());
        blocks.iter().map(|b| b.layout.size()).sum()
    }

    /// Map fresh zeroed pages at `addr`.
    pub fn map(&self, addr: u64, size: u64, permissions: u32) -> Result<(), MapError> {
        Self::check_range(addr, size)?;
        for page in 0..size / PAGE_SIZE {
            if self.entry(addr + page * PAGE_SIZE) & PRESENT != 0 {
                return Err(MapError::Overlap);
            }
        }
        let layout = Layout::from_size_align(size as usize, PAGE_SIZE as usize)
            .map_err(|_| MapError::Range)?;
        let watched = tracking::AVAILABLE && size as usize >= WATCH_MIN;
        let host = if watched {
            tracking::alloc(size as usize)
        } else {
            // SAFETY: the layout has a non-zero size.
            unsafe { alloc_zeroed(layout) }
        };
        if host.is_null() {
            // Said here, as the kernel's result for it names no cause.
            eprintln!(
                "kiln runtime: the host gave no memory for {size:#x} bytes of guest memory at \
                 {addr:#x} ({}; {:#x} bytes mapped before)",
                std::io::Error::last_os_error(),
                self.mapped_bytes()
            );
            return Err(MapError::OutOfMemory);
        }
        self.blocks.lock().unwrap().push(Block {
            ptr: host,
            layout,
            watched,
        });
        let flags = flags_of(permissions);
        for page in 0..size / PAGE_SIZE {
            let a = addr + page * PAGE_SIZE;
            // SAFETY: inside the block just allocated.
            let p = unsafe { host.add((page * PAGE_SIZE) as usize) } as u64;
            self.l2_for_map(a)[((a >> 12) & 511) as usize]
                .store(p | self.entry_flags(a, flags), Ordering::Release);
        }
        self.rewatch(addr, size);
        Ok(())
    }

    /// Make `dst` refer to the same host bytes as `src` (`svcMapMemory`). Every page of `src` must
    /// be mapped and every page of `dst` unmapped; the source keeps its permissions.
    pub fn alias(
        &self,
        dst: u64,
        src: u64,
        size: u64,
        dst_permissions: u32,
    ) -> Result<(), MapError> {
        Self::check_range(dst, size)?;
        Self::check_range(src, size)?;
        for page in 0..size / PAGE_SIZE {
            if self.entry(src + page * PAGE_SIZE) & PRESENT == 0 {
                return Err(MapError::Range);
            }
            if self.entry(dst + page * PAGE_SIZE) & PRESENT != 0 {
                return Err(MapError::Overlap);
            }
        }
        let flags = flags_of(dst_permissions);
        for page in 0..size / PAGE_SIZE {
            let host = self.entry(src + page * PAGE_SIZE) & !FLAGS;
            let a = dst + page * PAGE_SIZE;
            self.l2_for_map(a)[((a >> 12) & 511) as usize]
                .store(host | self.entry_flags(a, flags), Ordering::Release);
        }
        self.rewatch(dst, size);
        Ok(())
    }

    /// Make the pages unreachable. Their host bytes stay allocated (see the module comment).
    pub fn unmap(&self, addr: u64, size: u64) -> Result<(), MapError> {
        Self::check_range(addr, size)?;
        for page in 0..size / PAGE_SIZE {
            let a = addr + page * PAGE_SIZE;
            if let Some(t) = self.l2(a) {
                t[((a >> 12) & 511) as usize].store(0, Ordering::Release);
            }
        }
        self.rewatch(addr, size);
        Ok(())
    }

    /// Change the permissions of mapped pages; unmapped pages in the range are an error.
    pub fn protect(&self, addr: u64, size: u64, permissions: u32) -> Result<(), MapError> {
        Self::check_range(addr, size)?;
        for page in 0..size / PAGE_SIZE {
            if self.entry(addr + page * PAGE_SIZE) & PRESENT == 0 {
                return Err(MapError::Range);
            }
        }
        let flags = flags_of(permissions);
        for page in 0..size / PAGE_SIZE {
            let a = addr + page * PAGE_SIZE;
            let t = self.l2(a).expect("checked");
            let slot = &t[((a >> 12) & 511) as usize];
            let old = slot.load(Ordering::Relaxed);
            slot.store(
                (old & !FLAGS) | self.entry_flags(a, flags),
                Ordering::Release,
            );
        }
        self.rewatch(addr, size);
        Ok(())
    }

    /// The entry of the page at `addr`, where there is a second-level table for it.
    fn slot(&self, addr: u64) -> Option<&AtomicU64> {
        if addr >> ADDRESS_BITS != 0 {
            return None;
        }
        self.l2(addr).map(|t| &t[((addr >> 12) & 511) as usize])
    }

    /// Watch the page at `addr` (with the tracking locked).
    fn watch_page(&self, addr: u64) {
        if let Some(slot) = self.slot(addr) {
            let _ = slot.try_update(Ordering::AcqRel, Ordering::Relaxed, watched);
        }
    }

    /// A write to the watched page at `addr`: the page is stamped and written freely until the
    /// next epoch begins.
    #[cold]
    fn note_write(&self, addr: u64) {
        let mut w = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = self.slot(addr) else {
            return;
        };
        let freed = slot.try_update(Ordering::AcqRel, Ordering::Relaxed, |e| {
            (e & WATCH != 0).then(|| unwatched(e))
        });
        // Another write was first.
        if freed.is_err() {
            return;
        }
        let page = addr >> 12;
        let stamp = w.epoch.wrapping_add(1);
        w.stamps
            .entry(addr >> CHUNK_SHIFT)
            .or_insert_with(|| Box::new([0; 512]))[(page & 511) as usize] = stamp;
        w.written.push(addr & !(PAGE_SIZE - 1));
        w.pages += 1;
    }

    /// The pages of `[addr, addr + size)` changed under the tracking (mapped, unmapped, given
    /// other permissions): those of watched chunks count as written now and are watched again.
    fn rewatch(&self, addr: u64, size: u64) {
        let mut w = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        if w.chunks.is_empty() || size == 0 {
            return;
        }
        let stamp = w.epoch.wrapping_add(1);
        let end = addr + size;
        for c in addr >> CHUNK_SHIFT..=(end - 1) >> CHUNK_SHIFT {
            if !w.chunks.contains_key(&c) {
                continue;
            }
            let from = addr.max(c << CHUNK_SHIFT);
            let to = end.min((c + 1) << CHUNK_SHIFT);
            let stamps = w.stamps.entry(c).or_insert_with(|| Box::new([0; 512]));
            for a in (from..to).step_by(PAGE_SIZE as usize) {
                stamps[((a >> 12) & 511) as usize] = stamp;
                self.watch_page(a);
            }
            w.written.extend((from..to).step_by(PAGE_SIZE as usize));
        }
    }

    /// End the epoch of write tracking that is running and start the next: the number of the new
    /// one, to be kept by whoever looks at memory from now on and asks [`Self::untouched_since`]
    /// later.
    pub fn begin_epoch(&self) -> u32 {
        self.begin_epoch_into(&mut Vec::new())
    }

    /// As [`Self::begin_epoch`], and put in `changed` the guest ranges (address and length) that
    /// may have changed since the epoch before began: the pages written in it (or remapped, or
    /// given other permissions), and the chunks no longer watched from now on.
    pub fn begin_epoch_into(&self, changed: &mut Vec<(u64, u64)>) -> u32 {
        changed.clear();
        let mut w = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        let began = std::time::Instant::now();
        // The pages written in the epoch that ends are watched again before it ends: a write
        // after this is stamped with the number of the next.
        let mut written = std::mem::take(&mut w.written);
        for &a in &written {
            if w.chunks.contains_key(&(a >> CHUNK_SHIFT)) {
                self.watch_page(a);
            }
        }
        written.sort_unstable();
        for &a in &written {
            match changed.last_mut() {
                Some((at, len)) if *at + *len == a => *len += PAGE_SIZE,
                Some((at, len)) if *at + *len > a => {}
                _ => changed.push((a, PAGE_SIZE)),
            }
        }
        written.clear();
        w.written = written;
        // The chunks nobody asked about lately are let go.
        let epoch = w.epoch;
        let idle: Vec<u64> = w
            .chunks
            .iter()
            .filter(|(_, (_, asked))| epoch.wrapping_sub(*asked) > IDLE_EPOCHS)
            .map(|(c, _)| *c)
            .collect();
        for c in idle {
            w.chunks.remove(&c);
            w.stamps.remove(&c);
            changed.push((c << CHUNK_SHIFT, 1 << CHUNK_SHIFT));
            for p in 0..512 {
                if let Some(slot) = self.slot((c << CHUNK_SHIFT) + p * PAGE_SIZE) {
                    let _ = slot.try_update(Ordering::AcqRel, Ordering::Relaxed, |e| {
                        (e & WATCH != 0).then(|| unwatched(e))
                    });
                }
            }
        }
        w.epoch = w.epoch.wrapping_add(1).max(1);
        if w.epoch.is_multiple_of(100) && std::env::var_os("NX_TRACE_PERF").is_some() {
            eprintln!(
                "[memory] epoch {}: {} chunks of {} MB watched, {} pages written since the last line, {} chunks watched from a question; the epoch began in {:?}",
                w.epoch,
                w.chunks.len(),
                (1u64 << CHUNK_SHIFT) >> 20,
                w.pages,
                w.begun,
                began.elapsed()
            );
            w.pages = 0;
            w.begun = 0;
        }
        w.epoch
    }

    /// A reader still keeps what it found of the `len` bytes of guest memory at `addr`: the
    /// chunks of them that are watched count as asked about now, and are not let go.
    pub fn keep_watching(&self, addr: u64, len: u64) {
        let Some(end) = addr.checked_add(len).filter(|_| len > 0) else {
            return;
        };
        let mut w = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        let now = w.epoch;
        for c in addr >> CHUNK_SHIFT..=(end - 1) >> CHUNK_SHIFT {
            if let Some((_, asked)) = w.chunks.get_mut(&c) {
                *asked = now;
            }
        }
    }

    /// Whether none of the `len` bytes of guest memory at `addr` was written since epoch `epoch`
    /// began (`Some(true)`), or some was (`Some(false)`). `None` if that cannot be told: a page of
    /// the range is not mapped, or a part of it was not watched before the epoch began. Asking
    /// starts the watching of that part, so a later question about it can be answered.
    pub fn untouched_since(&self, addr: u64, len: u64, epoch: u32) -> Option<bool> {
        if len == 0 || epoch == 0 {
            return None;
        }
        let end = addr.checked_add(len)?;
        if end > 1 << ADDRESS_BITS || !self.is_mapped(addr, len) {
            return None;
        }
        let mut w = self.writes.lock().unwrap_or_else(|e| e.into_inner());
        let now = w.epoch;
        let mut known = true;
        for c in addr >> CHUNK_SHIFT..=(end - 1) >> CHUNK_SHIFT {
            match w.chunks.get_mut(&c) {
                Some((since, asked)) => {
                    *asked = now;
                    // Watched from within the epoch or later: a write between the reader's look
                    // and then was not seen.
                    known &= *since < epoch;
                }
                None => {
                    w.chunks.insert(c, (now, now));
                    w.begun += 1;
                    for p in 0..512 {
                        self.watch_page((c << CHUNK_SHIFT) + p * PAGE_SIZE);
                    }
                    known = false;
                }
            }
        }
        if !known {
            return None;
        }
        let (mut page, last) = (addr >> 12, (end - 1) >> 12);
        while page <= last {
            let c = page >> 9;
            let upto = last.min((c << 9) + 511);
            if let Some(stamps) = w.stamps.get(&c) {
                if stamps[(page & 511) as usize..=(upto & 511) as usize]
                    .iter()
                    .any(|s| *s > epoch)
                {
                    return Some(false);
                }
            }
            page = upto + 1;
        }
        Some(true)
    }

    /// The host address of the guest byte at `addr` and how many bytes from there, at most `max`,
    /// are contiguous in host memory (neighbouring pages of one allocation). `None` when `addr`
    /// is not mapped. Host pages stay where they are until the process ends (see the module
    /// comment), so the address can be handed to a device.
    pub fn host_run(&self, addr: u64, max: u64) -> Option<(*mut u8, u64)> {
        let e = self.entry(addr);
        if e & PRESENT == 0 {
            return None;
        }
        let mut host = e & !FLAGS;
        let start = host.wrapping_add(addr & (PAGE_SIZE - 1)) as *mut u8;
        let mut len = PAGE_SIZE - (addr & (PAGE_SIZE - 1));
        let mut at = (addr | (PAGE_SIZE - 1)) + 1;
        while len < max {
            let next = self.entry(at);
            if next & PRESENT == 0 || next & !FLAGS != host + PAGE_SIZE {
                break;
            }
            host += PAGE_SIZE;
            len += PAGE_SIZE;
            at += PAGE_SIZE;
        }
        Some((start, len.min(max)))
    }

    /// Is every page of the range mapped?
    pub fn is_mapped(&self, addr: u64, size: u64) -> bool {
        let first = addr & !(PAGE_SIZE - 1);
        let last = addr.saturating_add(size.max(1) - 1) & !(PAGE_SIZE - 1);
        (first..=last)
            .step_by(PAGE_SIZE as usize)
            .all(|a| self.entry(a) & PRESENT != 0)
    }

    /// Host pointer for `bytes` bytes at `addr` inside one page, if the access is allowed. A write
    /// to a trapped page is allowed and reported (with `value`, when it is a store of a value).
    fn page_ptr(
        &self,
        addr: u64,
        bytes: usize,
        write: bool,
        value: Option<u128>,
    ) -> Result<*mut u8, FaultKind> {
        let e = self.entry(addr);
        if e & PRESENT == 0 {
            return Err(FaultKind::Unmapped);
        }
        if e & if write { WRITE } else { READ } == 0 {
            // A trapped page and a watched writable one are written, and say so.
            if write && e & (TRAP | WAS_W) != 0 {
                if e & TRAP != 0 {
                    self.traps.hit(addr, bytes, value);
                }
                if e & WATCH != 0 {
                    self.note_write(addr);
                }
            } else {
                return Err(FaultKind::Permission);
            }
        }
        let off = (addr & (PAGE_SIZE - 1)) as usize;
        debug_assert!(off + bytes <= PAGE_SIZE as usize);
        // SAFETY: the entry holds the address of a live 4 KiB host page.
        Ok(unsafe { ((e & !FLAGS) as *mut u8).add(off) })
    }

    /// Read `size` (1, 2, 4, 8 or 16) bytes, little-endian.
    pub fn read(&self, addr: u64, size: u8, order: Ordering) -> Result<u128, FaultKind> {
        let n = usize::from(size);
        if (addr & (PAGE_SIZE - 1)) as usize + n <= PAGE_SIZE as usize {
            let p = self.page_ptr(addr, n, false, None)?;
            // SAFETY: `p` points at `n` readable bytes inside one page.
            return Ok(unsafe { load(p, size, order) });
        }
        // Straddles a page boundary: both pages must be readable.
        let mut buf = [0u8; 16];
        self.read_bytes(addr, &mut buf[..n], true)?;
        Ok(u128::from_le_bytes(buf))
    }

    /// Write the low `size` bytes of `value`.
    pub fn write(
        &self,
        addr: u64,
        size: u8,
        value: u128,
        order: Ordering,
    ) -> Result<(), FaultKind> {
        let n = usize::from(size);
        if (addr & (PAGE_SIZE - 1)) as usize + n <= PAGE_SIZE as usize {
            let p = self.page_ptr(addr, n, true, Some(value))?;
            // SAFETY: `p` points at `n` writable bytes inside one page.
            unsafe { store(p, size, value, order) };
            return Ok(());
        }
        // Check both pages before writing anything. Addresses beyond the 39-bit space (including a
        // range that wraps past 2^64) fail at the first page that is not there.
        let last = addr.checked_add(n as u64 - 1).ok_or(FaultKind::Unmapped)?;
        let mut at = addr;
        loop {
            self.page_ptr(at, 0, true, None)?;
            let next = (at | (PAGE_SIZE - 1)) + 1;
            if next > last {
                break;
            }
            at = next;
        }
        self.write_bytes(addr, &value.to_le_bytes()[..n], false)
    }

    /// Copy bytes out of guest memory. `check` applies the read permission; the loader and tests
    /// pass false.
    pub fn read_bytes(&self, addr: u64, out: &mut [u8], check: bool) -> Result<(), FaultKind> {
        let mut at = addr;
        let mut done = 0;
        while done < out.len() {
            let chunk = (PAGE_SIZE - (at & (PAGE_SIZE - 1))) as usize;
            let n = chunk.min(out.len() - done);
            let e = self.entry(at);
            if e & PRESENT == 0 {
                return Err(FaultKind::Unmapped);
            }
            if check && e & READ == 0 {
                return Err(FaultKind::Permission);
            }
            let src = ((e & !FLAGS) as *const u8).wrapping_add((at & (PAGE_SIZE - 1)) as usize);
            // SAFETY: `src..src + n` is inside a live page.
            unsafe { ptr::copy_nonoverlapping(src, out[done..].as_mut_ptr(), n) };
            at += n as u64;
            done += n;
        }
        Ok(())
    }

    pub fn write_bytes(&self, addr: u64, data: &[u8], check: bool) -> Result<(), FaultKind> {
        let mut at = addr;
        let mut done = 0;
        while done < data.len() {
            let chunk = (PAGE_SIZE - (at & (PAGE_SIZE - 1))) as usize;
            let n = chunk.min(data.len() - done);
            let e = self.entry(at);
            if e & PRESENT == 0 {
                return Err(FaultKind::Unmapped);
            }
            if check && e & (WRITE | TRAP | WAS_W) == 0 {
                return Err(FaultKind::Permission);
            }
            if e & TRAP != 0 {
                self.traps.hit(at, n, None);
            }
            if e & WATCH != 0 {
                self.note_write(at);
            }
            let dst = ((e & !FLAGS) as *mut u8).wrapping_add((at & (PAGE_SIZE - 1)) as usize);
            // SAFETY: `dst..dst + n` is inside a live page.
            unsafe { ptr::copy_nonoverlapping(data[done..].as_ptr(), dst, n) };
            at += n as u64;
            done += n;
        }
        Ok(())
    }

    /// Atomically replace `size` bytes at `addr` if they still hold `expected`. Returns whether
    /// the exchange happened. `size` is 1, 2, 4 or 8 and the address must be aligned.
    pub fn compare_exchange(
        &self,
        addr: u64,
        size: u8,
        expected: u64,
        new: u64,
        order: Ordering,
    ) -> Result<bool, FaultKind> {
        let p = self.page_ptr(addr, usize::from(size), true, Some(u128::from(new)))?;
        // SAFETY: aligned, inside one writable page.
        let ok = unsafe {
            match size {
                1 => AtomicU8::from_ptr(p)
                    .compare_exchange(expected as u8, new as u8, order, Ordering::Relaxed)
                    .is_ok(),
                2 => AtomicU16::from_ptr(p.cast())
                    .compare_exchange(expected as u16, new as u16, order, Ordering::Relaxed)
                    .is_ok(),
                4 => AtomicU32::from_ptr(p.cast())
                    .compare_exchange(expected as u32, new as u32, order, Ordering::Relaxed)
                    .is_ok(),
                _ => AtomicU64::from_ptr(p.cast())
                    .compare_exchange(expected, new, order, Ordering::Relaxed)
                    .is_ok(),
            }
        };
        Ok(ok)
    }

    /// Atomically replace the 16 bytes at `addr`, which must be 16-byte aligned, if they still
    /// hold `expected`: one exchange, the host's 16-byte compare-and-exchange. Where the host
    /// has none, a read, a compare and a write under a lock stand in, which a plain store by
    /// another thread can slip between.
    pub fn compare_exchange_128(
        &self,
        addr: u64,
        expected: u128,
        new: u128,
        order: Ordering,
    ) -> Result<bool, FaultKind> {
        if !addr.is_multiple_of(16) {
            return Err(FaultKind::Alignment);
        }
        let p = self.page_ptr(addr, 16, true, Some(new))?;
        // SAFETY: 16 writable bytes inside one page, aligned to 16.
        Ok(unsafe { cas128(p.cast(), expected, new, order) } == expected)
    }

    /// An LSE atomic (`LD<op>`, `SWP`) on `size` bytes (1, 2, 4 or 8) at `addr`, which must be
    /// aligned: one host atomic operation. Returns what memory held.
    pub fn atomic(
        &self,
        addr: u64,
        size: u8,
        op: AtomicOp,
        operand: u64,
        order: Ordering,
    ) -> Result<u64, FaultKind> {
        if !addr.is_multiple_of(u64::from(size)) {
            return Err(FaultKind::Alignment);
        }
        let p = self.page_ptr(addr, usize::from(size), true, Some(u128::from(operand)))?;
        macro_rules! rmw {
            ($au:ident, $ai:ident, $u:ty, $i:ty) => {{
                // SAFETY: aligned, inside one writable page.
                let (a, s) = unsafe { ($au::from_ptr(p.cast()), $ai::from_ptr(p.cast())) };
                let v = operand as $u;
                u64::from(match op {
                    AtomicOp::Add => a.fetch_add(v, order),
                    AtomicOp::Clr => a.fetch_and(!v, order),
                    AtomicOp::Eor => a.fetch_xor(v, order),
                    AtomicOp::Set => a.fetch_or(v, order),
                    AtomicOp::Smax => s.fetch_max(v as $i, order) as $u,
                    AtomicOp::Smin => s.fetch_min(v as $i, order) as $u,
                    AtomicOp::Umax => a.fetch_max(v, order),
                    AtomicOp::Umin => a.fetch_min(v, order),
                    AtomicOp::Swp => a.swap(v, order),
                })
            }};
        }
        Ok(match size {
            1 => rmw!(AtomicU8, AtomicI8, u8, i8),
            2 => rmw!(AtomicU16, AtomicI16, u16, i16),
            4 => rmw!(AtomicU32, AtomicI32, u32, i32),
            _ => rmw!(AtomicU64, AtomicI64, u64, i64),
        })
    }

    /// `CAS` of `size` bytes (1, 2, 4 or 8) at an aligned `addr`: `new` replaces `expected` if
    /// memory holds it, in one host compare-and-exchange. Returns what memory held. A compare
    /// that fails still needs a writable page, as the Arm pseudocode's atomic access does.
    pub fn compare_swap(
        &self,
        addr: u64,
        size: u8,
        expected: u64,
        new: u64,
        order: Ordering,
    ) -> Result<u64, FaultKind> {
        if !addr.is_multiple_of(u64::from(size)) {
            return Err(FaultKind::Alignment);
        }
        let p = self.page_ptr(addr, usize::from(size), true, Some(u128::from(new)))?;
        let failure = failure_order(order);
        macro_rules! cas {
            ($au:ident, $u:ty) => {{
                // SAFETY: aligned, inside one writable page.
                let a = unsafe { $au::from_ptr(p.cast()) };
                let r = a.compare_exchange(expected as $u, new as $u, order, failure);
                u64::from(r.unwrap_or_else(|v| v))
            }};
        }
        Ok(match size {
            1 => cas!(AtomicU8, u8),
            2 => cas!(AtomicU16, u16),
            4 => cas!(AtomicU32, u32),
            _ => cas!(AtomicU64, u64),
        })
    }

    /// `CASP` of X registers: [`GuestMemory::compare_swap`] of the 16 bytes at `addr`, which must be
    /// 16-byte aligned. Returns what memory held.
    pub fn compare_swap_128(
        &self,
        addr: u64,
        expected: u128,
        new: u128,
        order: Ordering,
    ) -> Result<u128, FaultKind> {
        if !addr.is_multiple_of(16) {
            return Err(FaultKind::Alignment);
        }
        let p = self.page_ptr(addr, 16, true, Some(new))?;
        // SAFETY: 16 writable bytes inside one page, aligned to 16.
        Ok(unsafe { cas128(p.cast(), expected, new, order) })
    }

    /// Does the page at `addr` allow execution?
    pub fn is_executable(&self, addr: u64) -> bool {
        self.entry(addr) & (PRESENT | EXEC) == PRESENT | EXEC
    }
}

/// The ordering of a compare-and-exchange that fails: never a release, never stronger.
fn failure_order(order: Ordering) -> Ordering {
    match order {
        Ordering::Relaxed | Ordering::Release => Ordering::Relaxed,
        Ordering::SeqCst => Ordering::SeqCst,
        _ => Ordering::Acquire,
    }
}

/// A 16-byte compare-and-exchange; returns what the bytes held.
///
/// # Safety
/// `p` must point at 16 writable bytes, aligned to 16.
#[cfg(target_arch = "x86_64")]
unsafe fn cas128(p: *mut u128, expected: u128, new: u128, order: Ordering) -> u128 {
    #[target_feature(enable = "cmpxchg16b")]
    unsafe fn exchange(p: *mut u128, expected: u128, new: u128, order: Ordering) -> u128 {
        // The failure ordering may be neither a release nor stronger than the success one.
        let failure = match order {
            Ordering::Relaxed | Ordering::Release => Ordering::Relaxed,
            _ => Ordering::Acquire,
        };
        core::arch::x86_64::cmpxchg16b(p, expected, new, order, failure)
    }
    if std::is_x86_feature_detected!("cmpxchg16b") {
        return exchange(p, expected, new, order);
    }
    cas128_locked(p, expected, new)
}

/// # Safety
/// `p` must point at 16 writable bytes, aligned to 16.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn cas128(p: *mut u128, expected: u128, new: u128, _order: Ordering) -> u128 {
    cas128_locked(p, expected, new)
}

/// The 16-byte exchange where the host cannot do it in one instruction: under a lock, so it is
/// atomic against the other exchanges, not against plain stores.
///
/// # Safety
/// `p` must point at 16 writable bytes, aligned to 8.
unsafe fn cas128_locked(p: *mut u128, expected: u128, new: u128) -> u128 {
    static LOCK: Mutex<()> = Mutex::new(());
    let _held = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let words = p.cast::<u64>();
    let lo = AtomicU64::from_ptr(words).load(Ordering::Relaxed);
    let hi = AtomicU64::from_ptr(words.add(1)).load(Ordering::Relaxed);
    let old = u128::from(lo) | (u128::from(hi) << 64);
    if old == expected {
        AtomicU64::from_ptr(words).store(new as u64, Ordering::Relaxed);
        AtomicU64::from_ptr(words.add(1)).store((new >> 64) as u64, Ordering::Release);
    }
    old
}

/// # Safety
/// `p` must point at `size` readable bytes.
unsafe fn load(p: *const u8, size: u8, order: Ordering) -> u128 {
    let aligned = |n: usize| (p as usize).is_multiple_of(n);
    match size {
        1 => u128::from(AtomicU8::from_ptr(p as *mut u8).load(order)),
        2 if aligned(2) => u128::from(AtomicU16::from_ptr(p as *mut u16).load(order)),
        4 if aligned(4) => u128::from(AtomicU32::from_ptr(p as *mut u32).load(order)),
        8 if aligned(8) => u128::from(AtomicU64::from_ptr(p as *mut u64).load(order)),
        16 if aligned(8) => {
            let lo = AtomicU64::from_ptr(p as *mut u64).load(order);
            let hi = AtomicU64::from_ptr((p as *mut u64).add(1)).load(order);
            u128::from(lo) | (u128::from(hi) << 64)
        }
        n => {
            let mut buf = [0u8; 16];
            ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), usize::from(n));
            u128::from_le_bytes(buf)
        }
    }
}

/// # Safety
/// `p` must point at `size` writable bytes.
unsafe fn store(p: *mut u8, size: u8, value: u128, order: Ordering) {
    let aligned = |n: usize| (p as usize).is_multiple_of(n);
    match size {
        1 => AtomicU8::from_ptr(p).store(value as u8, order),
        2 if aligned(2) => AtomicU16::from_ptr(p as *mut u16).store(value as u16, order),
        4 if aligned(4) => AtomicU32::from_ptr(p as *mut u32).store(value as u32, order),
        8 if aligned(8) => AtomicU64::from_ptr(p as *mut u64).store(value as u64, order),
        16 if aligned(8) => {
            AtomicU64::from_ptr(p as *mut u64).store(value as u64, Ordering::Relaxed);
            AtomicU64::from_ptr((p as *mut u64).add(1)).store((value >> 64) as u64, order);
        }
        n => {
            let bytes = value.to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), p, usize::from(n));
        }
    }
}

/// Large blocks of memory straight from the system.
#[cfg(windows)]
mod tracking {
    use std::ffi::c_void;

    extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    }

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 4;

    pub const AVAILABLE: bool = true;

    /// `size` bytes of zeroed memory.
    pub fn alloc(size: usize) -> *mut u8 {
        // SAFETY: a plain allocation; null on failure.
        unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .cast()
        }
    }

    /// # Safety
    /// `ptr` came from [`alloc`] and is not used again.
    pub unsafe fn free(ptr: *mut u8) {
        VirtualFree(ptr.cast(), 0, MEM_RELEASE);
    }
}

#[cfg(not(windows))]
mod tracking {
    pub const AVAILABLE: bool = false;

    pub fn alloc(_size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    /// # Safety
    /// Never called: nothing is allocated.
    pub unsafe fn free(_ptr: *mut u8) {}
}

#[cfg(test)]
mod tests {

    use super::*;

    const RO: u32 = perm::R;
    const RW: u32 = perm::R | perm::W;

    #[test]
    fn mapped_memory_reads_back_what_was_written() {
        let m = GuestMemory::new();
        m.map(0x0800_0000, 0x3000, RW).unwrap();
        for (size, value) in [
            (1u8, 0xABu128),
            (2, 0xBEEF),
            (4, 0xDEAD_BEEF),
            (8, u64::MAX as u128 - 7),
        ] {
            m.write(0x0800_0010, size, value, Ordering::Relaxed)
                .unwrap();
            assert_eq!(m.read(0x0800_0010, size, Ordering::Relaxed).unwrap(), value);
        }
        let wide = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFFu128;
        m.write(0x0800_0020, 16, wide, Ordering::Relaxed).unwrap();
        assert_eq!(m.read(0x0800_0020, 16, Ordering::Relaxed).unwrap(), wide);
    }

    #[test]
    fn fresh_pages_are_zero_and_accesses_may_straddle_pages() {
        let m = GuestMemory::new();
        m.map(0x1000_0000, 0x2000, RW).unwrap();
        assert_eq!(m.read(0x1000_0ff8, 8, Ordering::Relaxed).unwrap(), 0);
        let v = 0x0102_0304_0506_0708u128;
        m.write(0x1000_0ffc, 8, v, Ordering::Relaxed).unwrap();
        assert_eq!(m.read(0x1000_0ffc, 8, Ordering::Relaxed).unwrap(), v);
        // The low half landed at the end of the first page, the high half at the start of the next.
        assert_eq!(
            m.read(0x1000_0ffc, 4, Ordering::Relaxed).unwrap(),
            0x0506_0708
        );
        assert_eq!(
            m.read(0x1000_1000, 4, Ordering::Relaxed).unwrap(),
            0x0102_0304
        );
    }

    #[test]
    fn faults_are_classified() {
        let m = GuestMemory::new();
        m.map(0x2000_0000, 0x1000, RO).unwrap();
        assert_eq!(m.read(0x2000_0000, 4, Ordering::Relaxed), Ok(0));
        assert_eq!(
            m.write(0x2000_0000, 4, 1, Ordering::Relaxed),
            Err(FaultKind::Permission)
        );
        assert_eq!(
            m.read(0x2000_1000, 4, Ordering::Relaxed),
            Err(FaultKind::Unmapped)
        );
        assert_eq!(m.read(0, 1, Ordering::Relaxed), Err(FaultKind::Unmapped));
        // Outside the 39-bit space.
        assert_eq!(
            m.read(1 << 40, 1, Ordering::Relaxed),
            Err(FaultKind::Unmapped)
        );
        // A write straddling into an unmapped page changes nothing.
        let m2 = GuestMemory::new();
        m2.map(0x3000_0000, 0x1000, RW).unwrap();
        assert_eq!(
            m2.write(0x3000_0ffc, 8, u128::MAX, Ordering::Relaxed),
            Err(FaultKind::Unmapped)
        );
        assert_eq!(m2.read(0x3000_0ffc, 4, Ordering::Relaxed), Ok(0));
    }

    #[test]
    fn map_unmap_and_protect() {
        let m = GuestMemory::new();
        assert_eq!(m.map(0x123, 0x1000, RW), Err(MapError::Range));
        assert_eq!(m.map(0x1000, 0x123, RW), Err(MapError::Range));
        assert_eq!(m.map(1 << 38, (1 << 38) + 0x1000, RW), Err(MapError::Range));
        m.map(0x4000_0000, 0x2000, RW).unwrap();
        assert_eq!(m.map(0x4000_1000, 0x1000, RW), Err(MapError::Overlap));
        m.write(0x4000_0000, 4, 7, Ordering::Relaxed).unwrap();
        m.protect(0x4000_0000, 0x1000, RO).unwrap();
        assert_eq!(
            m.write(0x4000_0000, 4, 8, Ordering::Relaxed),
            Err(FaultKind::Permission)
        );
        assert_eq!(m.read(0x4000_0000, 4, Ordering::Relaxed), Ok(7));
        assert!(m.is_mapped(0x4000_0000, 0x2000));
        m.unmap(0x4000_1000, 0x1000).unwrap();
        assert!(!m.is_mapped(0x4000_0000, 0x2000));
        assert_eq!(
            m.read(0x4000_1000, 1, Ordering::Relaxed),
            Err(FaultKind::Unmapped)
        );
        assert_eq!(m.protect(0x4000_1000, 0x1000, RW), Err(MapError::Range));
        // The address space can be reused.
        m.map(0x4000_1000, 0x1000, RW).unwrap();
        assert_eq!(m.read(0x4000_1000, 8, Ordering::Relaxed), Ok(0));
    }

    #[test]
    fn accesses_that_straddle_pages_respect_both_pages_permissions() {
        let m = GuestMemory::new();
        m.map(0x9000_0000, 0x1000, RW).unwrap();
        m.map(0x9000_1000, 0x1000, 0).unwrap();
        m.map(0x9000_2000, 0x1000, RW).unwrap();
        m.map(0x9000_3000, 0x1000, RO).unwrap();
        for size in [2u8, 4, 8, 16] {
            let at = 0x9000_1000 - u64::from(size) / 2;
            assert_eq!(
                m.read(at, size, Ordering::Relaxed),
                Err(FaultKind::Permission),
                "read of {size} bytes into a page with no access"
            );
            assert_eq!(
                m.write(at, size, 1, Ordering::Relaxed),
                Err(FaultKind::Permission),
                "write of {size} bytes into a page with no access"
            );
            // The page that does allow the access is untouched by the failed write.
            assert_eq!(m.read(0x9000_0ff0, 16, Ordering::Relaxed), Ok(0));
            // Leaving a page with no access is just as much a fault.
            let at = 0x9000_2000 - u64::from(size) / 2;
            assert_eq!(
                m.read(at, size, Ordering::Relaxed),
                Err(FaultKind::Permission)
            );
            // A read crossing from one readable page into a read-only page is allowed, a write is not.
            let at = 0x9000_3000 - u64::from(size) / 2;
            assert!(m.read(at, size, Ordering::Relaxed).is_ok());
            assert_eq!(
                m.write(at, size, 1, Ordering::Relaxed),
                Err(FaultKind::Permission)
            );
        }
    }

    #[test]
    fn byte_copies_bypass_permissions_when_asked() {
        let m = GuestMemory::new();
        m.map(0x5000_0000, 0x2000, RO).unwrap();
        let data: Vec<u8> = (0..=255).collect();
        assert_eq!(
            m.write_bytes(0x5000_0f80, &data, true),
            Err(FaultKind::Permission)
        );
        m.write_bytes(0x5000_0f80, &data, false).unwrap();
        let mut out = vec![0u8; 256];
        m.read_bytes(0x5000_0f80, &mut out, true).unwrap();
        assert_eq!(out, data);
        let mut past = vec![0u8; 16];
        assert_eq!(
            m.read_bytes(0x5000_1ff8, &mut past, false),
            Err(FaultKind::Unmapped)
        );
    }

    #[test]
    fn compare_exchange_is_atomic_and_checks_permissions() {
        let m = GuestMemory::new();
        m.map(0x6000_0000, 0x1000, RW).unwrap();
        m.write(0x6000_0008, 8, 5, Ordering::Relaxed).unwrap();
        assert_eq!(
            m.compare_exchange(0x6000_0008, 8, 5, 9, Ordering::AcqRel),
            Ok(true)
        );
        assert_eq!(
            m.compare_exchange(0x6000_0008, 8, 5, 11, Ordering::AcqRel),
            Ok(false)
        );
        assert_eq!(m.read(0x6000_0008, 8, Ordering::Relaxed), Ok(9));
        assert_eq!(
            m.compare_exchange(0x7000_0000, 4, 0, 1, Ordering::AcqRel),
            Err(FaultKind::Unmapped)
        );
    }

    #[test]
    fn accesses_at_the_top_of_the_address_space_fault_instead_of_overflowing() {
        let m = GuestMemory::new();
        m.map(0x7F_FFFF_F000, 0x1000, RW).unwrap();
        for addr in [u64::MAX - 3, u64::MAX, u64::MAX - 7, 1 << 39, (1 << 39) - 4] {
            for size in [1u8, 2, 4, 8, 16] {
                let r = m.read(addr, size, Ordering::Relaxed);
                let w = m.write(addr, size, 1, Ordering::Relaxed);
                let inside = addr
                    .checked_add(u64::from(size))
                    .is_some_and(|end| end <= 1 << 39);
                assert_eq!(r.is_ok(), inside, "{addr:#x} {size}");
                assert_eq!(w.is_ok(), inside, "{addr:#x} {size}");
            }
        }
    }

    #[test]
    fn concurrent_threads_share_the_pages() {
        use std::sync::Arc;
        let m = Arc::new(GuestMemory::new());
        m.map(0x8000_0000, 0x1000, RW).unwrap();
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let m = Arc::clone(&m);
                std::thread::spawn(move || {
                    for i in 0..1000u64 {
                        let a = 0x8000_0000 + 8 * (t * 8 + (i % 8));
                        m.write(a, 8, u128::from(i), Ordering::Release).unwrap();
                        assert!(m.read(a, 8, Ordering::Acquire).unwrap() <= 999);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Writes are noticed by the epochs, and only the pages written.
    #[test]
    fn written_pages_are_found_by_the_epoch_they_were_written_in() {
        let mem = GuestMemory::new();
        let base = 0x4000_0000u64;
        let size = 4u64 << 20;
        mem.map(base, size, perm::R | perm::W).unwrap();
        // The first question about a chunk cannot be answered; it starts the watching.
        let e1 = mem.begin_epoch();
        assert!(e1 > 0);
        assert_eq!(mem.untouched_since(base, 8192, e1), None);
        // The chunk far away is watched on its own (A-277): it is asked about too.
        assert_eq!(mem.untouched_since(base + (3 << 20), 100, e1), None);
        let e2 = mem.begin_epoch();
        assert!(e2 > e1);
        assert_eq!(mem.untouched_since(base, 8192, e2), Some(true));
        assert_eq!(mem.untouched_since(base + (3 << 20), 100, e2), Some(true));
        // A write to the second page of the range, and one far away, each made a way the
        // runtime writes.
        mem.write_bytes(base + 4096 + 8, &[1, 2, 3], true).unwrap();
        mem.write(base + (3 << 20), 1, 9, Ordering::Relaxed)
            .unwrap();
        // Seen at once, and the pages stay writable.
        assert_eq!(mem.untouched_since(base, 8192, e2), Some(false));
        mem.write(base + 4096 + 16, 8, 5, Ordering::Relaxed)
            .unwrap();
        // The next epoch says which pages may have changed: those two, each once.
        let mut changed = Vec::new();
        let e3 = mem.begin_epoch_into(&mut changed);
        assert_eq!(
            changed,
            [(base + 4096, PAGE_SIZE), (base + (3 << 20), PAGE_SIZE)]
        );
        assert_eq!(
            mem.untouched_since(base, 8192, e2),
            Some(false),
            "the second page was written after epoch e2 began"
        );
        assert_eq!(
            mem.untouched_since(base, 4096, e2),
            Some(true),
            "the first page was not"
        );
        assert_eq!(mem.untouched_since(base + (3 << 20), 100, e2), Some(false));
        // Looked at again after the write: untouched from then on, until the next write, which
        // the page being watched again sees.
        assert_eq!(mem.untouched_since(base, 8192, e3), Some(true));
        mem.compare_exchange(base + 4096 + 16, 8, 5, 6, Ordering::Relaxed)
            .unwrap();
        assert_eq!(mem.untouched_since(base, 8192, e3), Some(false));
        let mut back = [0u8; 8];
        mem.read_bytes(base + 4096 + 16, &mut back, true).unwrap();
        assert_eq!(u64::from_le_bytes(back), 6);
        // A range past the memory cannot be answered.
        assert_eq!(mem.untouched_since(base + size - 4096, 8192, e3), None);
    }

    /// A chunk nobody asks about is let go after a while, and the next question about it is not
    /// answered (A-277): a write made meanwhile is not lost on the reader, who looks again.
    #[test]
    fn a_chunk_nobody_asks_about_is_let_go_and_cannot_say_afterwards() {
        let mem = GuestMemory::new();
        let base = 0x4000_0000u64;
        mem.map(base, 4 << 20, perm::R | perm::W).unwrap();
        let e = mem.begin_epoch();
        assert_eq!(mem.untouched_since(base, 4096, e), None);
        let looked = mem.begin_epoch();
        assert_eq!(mem.untouched_since(base, 4096, looked), Some(true));
        let mut said = Vec::new();
        for _ in 0..IDLE_EPOCHS + 2 {
            let mut changed = Vec::new();
            mem.begin_epoch_into(&mut changed);
            said.extend(changed);
        }
        // Let go: the page is as its permissions say, and a write to it is not seen; the epoch
        // that let it go said so, once.
        assert_eq!(mem.entry(base) & (WRITE | WATCH), WRITE);
        assert_eq!(said, [(base, 1 << CHUNK_SHIFT)]);
        mem.write_bytes(base + 16, &[7], true).unwrap();
        mem.begin_epoch();
        assert_eq!(
            mem.untouched_since(base, 4096, looked),
            None,
            "a chunk let go cannot say"
        );
        assert_eq!(
            mem.untouched_since(base, 4096, looked),
            None,
            "nor when asked again: it was watched after the reader looked"
        );
        let after = mem.begin_epoch();
        assert_eq!(mem.untouched_since(base, 4096, after), Some(true));
    }

    /// Watching a page changes nothing a program can see: a read-only page stays read-only, a
    /// writable one writable, and the permissions given to a watched page later are kept, the
    /// page still watched.
    #[test]
    fn watching_a_page_keeps_its_permissions() {
        let mem = GuestMemory::new();
        mem.map(0x5000_0000, 64 << 10, perm::R | perm::W).unwrap();
        mem.map(0x5001_0000, PAGE_SIZE, perm::R).unwrap();
        let e = mem.begin_epoch();
        assert_eq!(mem.untouched_since(0x5000_0000, 0x11000, e), None);
        assert_ne!(mem.entry(0x5000_0000) & WATCH, 0);
        assert_eq!(
            mem.write(0x5001_0000, 4, 1, Ordering::Relaxed),
            Err(FaultKind::Permission)
        );
        assert_eq!(
            mem.write_bytes(0x5001_0000, &[1], true),
            Err(FaultKind::Permission)
        );
        let e = mem.begin_epoch();
        assert_eq!(mem.untouched_since(0x5000_0000, 0x11000, e), Some(true));
        // What the runtime writes past the permissions (the loader, the GPU) is seen too.
        mem.write_bytes(0x5001_0000, &[1], false).unwrap();
        assert_eq!(mem.untouched_since(0x5001_0000, 16, e), Some(false));
        // Made read-only, then writable again: still watched.
        let e = mem.begin_epoch();
        mem.protect(0x5000_0000, PAGE_SIZE, perm::R).unwrap();
        assert_eq!(
            mem.write(0x5000_0000, 4, 1, Ordering::Relaxed),
            Err(FaultKind::Permission)
        );
        mem.protect(0x5000_0000, PAGE_SIZE, perm::R | perm::W)
            .unwrap();
        let e2 = mem.begin_epoch();
        assert_eq!(
            mem.untouched_since(0x5000_0000, 16, e),
            Some(false),
            "a change of permissions counts as a write"
        );
        assert_eq!(mem.untouched_since(0x5000_0000, 16, e2), Some(true));
        mem.write(0x5000_0000, 4, 1, Ordering::Relaxed).unwrap();
        assert_eq!(mem.untouched_since(0x5000_0000, 16, e2), Some(false));
        // Unmapped and mapped again under a watched chunk: fresh pages, which count as written.
        let e3 = mem.begin_epoch();
        mem.unmap(0x5000_4000, PAGE_SIZE).unwrap();
        assert_eq!(mem.untouched_since(0x5000_4000, 16, e3), None);
        mem.map(0x5000_4000, PAGE_SIZE, perm::R | perm::W).unwrap();
        assert_eq!(mem.untouched_since(0x5000_4000, 16, e3), Some(false));
        let e4 = mem.begin_epoch();
        mem.write(0x5000_4000, 4, 2, Ordering::Relaxed).unwrap();
        assert_eq!(mem.untouched_since(0x5000_4000, 16, e4), Some(false));
    }

    #[test]
    fn writes_to_a_trapped_page_are_reported_and_still_done() {
        use crate::rt::trap::Traps;
        let m = GuestMemory::with_traps(Traps::parse("12000:1000"));
        m.map(0x10000, 4 * PAGE_SIZE, perm::R | perm::W).unwrap();
        // The page is writable as far as the guest can tell, but its entry says otherwise.
        assert_eq!(m.entry(0x12000) & (WRITE | TRAP), TRAP);
        assert_eq!(m.entry(0x11000) & (WRITE | TRAP), WRITE);
        assert_eq!(m.entry(0x13000) & (WRITE | TRAP), WRITE);
        // A write elsewhere is not reported.
        m.write(0x11008, 8, 7, Ordering::Relaxed).unwrap();
        assert_eq!(m.traps.hits(), 0);
        // A write to the page is, whatever way it is made, and it is done.
        m.write(0x12008, 4, 0xDEAD_BEEF, Ordering::Relaxed).unwrap();
        assert_eq!(m.traps.hits(), 1);
        m.write_bytes(0x12010, &[1, 2, 3], true).unwrap();
        assert_eq!(m.traps.hits(), 2);
        m.compare_exchange(0x12008, 4, 0xDEAD_BEEF, 5, Ordering::Relaxed)
            .unwrap();
        assert_eq!(m.traps.hits(), 3);
        assert_eq!(m.read(0x12008, 4, Ordering::Relaxed).unwrap(), 5);
        let mut back = [0u8; 3];
        m.read_bytes(0x12010, &mut back, true).unwrap();
        assert_eq!(back, [1, 2, 3]);
        // Reads are not reported.
        assert_eq!(m.traps.hits(), 3);
        // Taking the page's write permission away and giving it back keeps it trapped.
        m.protect(0x12000, PAGE_SIZE, perm::R).unwrap();
        assert!(m.write(0x12008, 4, 1, Ordering::Relaxed).is_err());
        m.protect(0x12000, PAGE_SIZE, perm::R | perm::W).unwrap();
        assert_eq!(m.entry(0x12000) & (WRITE | TRAP), TRAP);
    }

    #[test]
    fn a_trap_does_not_make_a_read_only_page_writable() {
        use crate::rt::trap::Traps;
        let m = GuestMemory::with_traps(Traps::parse("20000:2000"));
        m.map(0x20000, PAGE_SIZE, perm::R).unwrap();
        assert_eq!(m.entry(0x20000) & TRAP, 0);
        assert_eq!(
            m.write(0x20000, 4, 1, Ordering::Relaxed),
            Err(FaultKind::Permission)
        );
        assert_eq!(m.traps.hits(), 0);
    }
}
