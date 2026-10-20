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
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering};
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
const FLAGS: u64 = 0xFFF;

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

/// The smallest block of memory whose writes are tracked: the memory of textures is in large
/// blocks, and a small block is not worth the system's bookkeeping.
const WATCH_MIN: usize = 1 << 20;

/// Which pages of memory were written, as far as the system's write tracking says (Windows
/// `MEM_WRITE_WATCH`; docs/ASSUMPTIONS.md A-126). Time is cut into epochs: each call of
/// [`GuestMemory::begin_epoch`] ends one and starts the next, and a page that was written during
/// an epoch is stamped with the number of the epoch that follows it. A reader that looked at the
/// bytes of a range after epoch `e` began knows they are as it saw them if no page of the range
/// has a stamp above `e`.
#[derive(Default)]
struct Writes {
    epoch: u32,
    /// The tracked blocks (address and size), by address, and whether they are harvested.
    blocks: Vec<Watched>,
    /// The stamps, in runs of 512 pages, by the number of the first page.
    stamps: HashMap<usize, Box<[u32; 512]>>,
    /// Room for the pages one harvest finds.
    scratch: Vec<usize>,
    /// How many pages the harvests found (for the trace).
    pages: usize,
}

struct Watched {
    base: usize,
    len: usize,
    /// For each [`CHUNK`] of the block, whether it is harvested at each epoch: only the parts
    /// somebody asked about are, as asking costs less than reading the system's bits of memory
    /// nobody looks at (a block of the title's is 3 GB).
    active: Vec<bool>,
}

/// The part of a block that is harvested, or not, as a whole.
const CHUNK: usize = 16 << 20;

/// The most the system is asked about in one call.
const HARVEST_STEP: usize = 64 << 20;

impl Writes {
    /// Stamp the pages of `[base, base + len)` that were written since they were last harvested.
    fn harvest_range(&mut self, base: usize, len: usize) {
        let mut scratch = std::mem::take(&mut self.scratch);
        let stamp = self.epoch;
        let mut at = base;
        while at < base + len {
            let n = HARVEST_STEP.min(base + len - at);
            tracking::harvest(at, n, &mut scratch);
            let mut run: Option<(usize, *mut [u32; 512])> = None;
            for page in scratch.iter().map(|a| a >> 12) {
                let first = page & !511;
                let stamps = match run {
                    Some((f, p)) if f == first => p,
                    _ => {
                        let p: *mut [u32; 512] = &mut **self
                            .stamps
                            .entry(first)
                            .or_insert_with(|| Box::new([0; 512]));
                        run = Some((first, p));
                        p
                    }
                };
                // SAFETY: the box lives in `stamps`, which is not touched while `run` is used.
                unsafe { (*stamps)[page & 511] = stamp };
            }
            self.pages += scratch.len();
            at += n;
        }
        self.scratch = scratch;
    }

    /// Harvest every chunk of block `i` that is active.
    fn harvest_active(&mut self, i: usize) {
        let (base, len) = (self.blocks[i].base, self.blocks[i].len);
        let chunks = self.blocks[i].active.len();
        let mut c = 0;
        while c < chunks {
            if !self.blocks[i].active[c] {
                c += 1;
                continue;
            }
            let first = c;
            while c < chunks && self.blocks[i].active[c] {
                c += 1;
            }
            let from = first * CHUNK;
            let to = (c * CHUNK).min(len);
            self.harvest_range(base + from, to - from);
        }
    }
}

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
            return Err(MapError::OutOfMemory);
        }
        self.blocks.lock().unwrap().push(Block {
            ptr: host,
            layout,
            watched,
        });
        if watched {
            let mut w = self.writes.lock().unwrap();
            let at = w.blocks.partition_point(|b| b.base < host as usize);
            w.blocks.insert(
                at,
                Watched {
                    base: host as usize,
                    len: size as usize,
                    active: vec![false; (size as usize).div_ceil(CHUNK)],
                },
            );
        }
        let flags = flags_of(permissions);
        for page in 0..size / PAGE_SIZE {
            let a = addr + page * PAGE_SIZE;
            // SAFETY: inside the block just allocated.
            let p = unsafe { host.add((page * PAGE_SIZE) as usize) } as u64;
            self.l2_for_map(a)[((a >> 12) & 511) as usize]
                .store(p | self.entry_flags(a, flags), Ordering::Release);
        }
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
        Ok(())
    }

    /// End the epoch of write tracking that is running and start the next: the number of the new
    /// one, to be kept by whoever looks at memory from now on and asks [`Self::untouched_since`]
    /// later. 0 when memory is not tracked.
    pub fn begin_epoch(&self) -> u32 {
        if !tracking::AVAILABLE {
            return 0;
        }
        let mut w = self.writes.lock().unwrap();
        w.epoch = w.epoch.wrapping_add(1).max(1);
        let began = std::time::Instant::now();
        w.pages = 0;
        let mut chunks = 0;
        for i in 0..w.blocks.len() {
            chunks += w.blocks[i].active.iter().filter(|a| **a).count();
            w.harvest_active(i);
        }
        if w.epoch.is_multiple_of(100) && std::env::var_os("NX_TRACE_PERF").is_some() {
            eprintln!(
                "[memory] epoch {}: harvested {chunks} chunks of {} MB in {:?}, {} pages written",
                w.epoch,
                CHUNK >> 20,
                began.elapsed(),
                w.pages
            );
        }
        w.epoch
    }

    /// Whether none of the `len` bytes at host address `host` was written since epoch `epoch`
    /// began (`Some(true)`), or some was (`Some(false)`). `None` if that cannot be told: the
    /// bytes are not all in a tracked block, or the part of the block has not been looked at
    /// before. Asking starts the harvesting of that part, so a later question about it can be
    /// answered.
    pub fn untouched_since(&self, host: *const u8, len: usize, epoch: u32) -> Option<bool> {
        if !tracking::AVAILABLE || len == 0 || epoch == 0 {
            return None;
        }
        let (start, end) = (host as usize, (host as usize).checked_add(len)?);
        let mut w = self.writes.lock().unwrap();
        let i = w
            .blocks
            .partition_point(|b| b.base <= start)
            .checked_sub(1)?;
        let b = &w.blocks[i];
        if end > b.base + b.len {
            return None;
        }
        let (first, last) = ((start - b.base) / CHUNK, (end - 1 - b.base) / CHUNK);
        if !b.active[first..=last].iter().all(|a| *a) {
            // Bring the bits of the chunks up to date: the next epoch stamps what is written from
            // now.
            for c in first..=last {
                if !w.blocks[i].active[c] {
                    w.blocks[i].active[c] = true;
                    let base = w.blocks[i].base + c * CHUNK;
                    let len = CHUNK.min(w.blocks[i].len - c * CHUNK);
                    w.harvest_range(base, len);
                }
            }
            return None;
        }
        let mut page = start >> 12;
        let last = (end - 1) >> 12;
        while page <= last {
            let first = page & !511;
            let upto = last.min(first + 511);
            if let Some(run) = w.stamps.get(&first) {
                if run[page & 511..=upto & 511].iter().any(|s| *s > epoch) {
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
            if write && e & TRAP != 0 {
                self.traps.hit(addr, bytes, value);
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
            if check && e & (WRITE | TRAP) == 0 {
                return Err(FaultKind::Permission);
            }
            if e & TRAP != 0 {
                self.traps.hit(at, n, None);
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
        Ok(unsafe { cas128(p.cast(), expected, new, order) })
    }

    /// Does the page at `addr` allow execution?
    pub fn is_executable(&self, addr: u64) -> bool {
        self.entry(addr) & (PRESENT | EXEC) == PRESENT | EXEC
    }
}

/// # Safety
/// `p` must point at 16 writable bytes, aligned to 16.
#[cfg(target_arch = "x86_64")]
unsafe fn cas128(p: *mut u128, expected: u128, new: u128, order: Ordering) -> bool {
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
        return exchange(p, expected, new, order) == expected;
    }
    cas128_locked(p, expected, new)
}

/// # Safety
/// `p` must point at 16 writable bytes, aligned to 16.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn cas128(p: *mut u128, expected: u128, new: u128, _order: Ordering) -> bool {
    cas128_locked(p, expected, new)
}

/// The 16-byte exchange where the host cannot do it in one instruction: under a lock, so it is
/// atomic against the other exchanges, not against plain stores.
///
/// # Safety
/// `p` must point at 16 writable bytes, aligned to 8.
unsafe fn cas128_locked(p: *mut u128, expected: u128, new: u128) -> bool {
    static LOCK: Mutex<()> = Mutex::new(());
    let _held = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let words = p.cast::<u64>();
    let lo = AtomicU64::from_ptr(words).load(Ordering::Relaxed);
    let hi = AtomicU64::from_ptr(words.add(1)).load(Ordering::Relaxed);
    if u128::from(lo) | (u128::from(hi) << 64) != expected {
        return false;
    }
    AtomicU64::from_ptr(words).store(new as u64, Ordering::Relaxed);
    AtomicU64::from_ptr(words.add(1)).store((new >> 64) as u64, Ordering::Release);
    true
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

/// The system's write tracking.
#[cfg(windows)]
mod tracking {
    use std::ffi::c_void;

    extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
        fn GetWriteWatch(
            flags: u32,
            base: *mut c_void,
            size: usize,
            addresses: *mut *mut c_void,
            count: *mut usize,
            granularity: *mut u32,
        ) -> u32;
    }

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    const PAGE_READWRITE: u32 = 4;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;

    pub const AVAILABLE: bool = true;

    /// `size` bytes of zeroed memory whose writes the system tracks.
    pub fn alloc(size: usize) -> *mut u8 {
        // SAFETY: a plain allocation; null on failure.
        unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
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

    /// Put in `pages` the addresses of the pages of the block that were written since the last
    /// harvest, and forget that they were.
    pub fn harvest(base: usize, len: usize, pages: &mut Vec<usize>) {
        // Room for every page of the block, so that one call finds them all.
        pages.clear();
        pages.resize(len / 4096, 0);
        let mut count = pages.len();
        let mut granularity = 0u32;
        // SAFETY: the range is one block from `alloc`; `pages` has room for `count` addresses (a
        // `usize` is a pointer's size).
        let status = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                base as *mut c_void,
                len,
                pages.as_mut_ptr().cast::<*mut c_void>(),
                &mut count,
                &mut granularity,
            )
        };
        pages.truncate(if status == 0 { count } else { 0 });
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

    pub fn harvest(_base: usize, _len: usize, _pages: &mut Vec<usize>) {}
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

    /// Writes to a large block are noticed by the epochs, and only the pages written.
    #[cfg(windows)]
    #[test]
    fn written_pages_are_found_by_the_epoch_they_were_written_in() {
        let mem = GuestMemory::new();
        let base = 0x4000_0000u64;
        let size = 4u64 << 20;
        mem.map(base, size, perm::R | perm::W).unwrap();
        let host = |a: u64| mem.host_run(a, 4096).unwrap().0 as *const u8;
        // The first question about a block cannot be answered; it starts the tracking.
        let e1 = mem.begin_epoch();
        assert!(e1 > 0);
        assert_eq!(mem.untouched_since(host(base), 8192, e1), None);
        let e2 = mem.begin_epoch();
        assert!(e2 > e1);
        assert_eq!(mem.untouched_since(host(base), 8192, e2), Some(true));
        // A write to the second page of the range, and one far away.
        mem.write_bytes(base + 4096 + 8, &[1, 2, 3], true).unwrap();
        mem.write_bytes(base + (3 << 20), &[9], true).unwrap();
        let e3 = mem.begin_epoch();
        assert_eq!(
            mem.untouched_since(host(base), 8192, e2),
            Some(false),
            "the second page was written after epoch e2 began"
        );
        assert_eq!(
            mem.untouched_since(host(base), 4096, e2),
            Some(true),
            "the first page was not"
        );
        assert_eq!(
            mem.untouched_since(host(base + (3 << 20)), 100, e2),
            Some(false)
        );
        // Looked at again after the write: untouched from then on.
        assert_eq!(mem.untouched_since(host(base), 8192, e3), Some(true));
        // A range past the block cannot be answered.
        assert_eq!(
            mem.untouched_since(host(base + size - 4096), 8192, e3),
            None
        );
    }

    /// A block too small to be tracked is not answered for.
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

    #[test]
    fn small_blocks_are_not_tracked() {
        let mem = GuestMemory::new();
        mem.map(0x5000_0000, 64 << 10, perm::R | perm::W).unwrap();
        let p = mem.host_run(0x5000_0000, 4096).unwrap().0 as *const u8;
        let e = mem.begin_epoch();
        assert_eq!(mem.untouched_since(p, 4096, e), None);
    }
}
