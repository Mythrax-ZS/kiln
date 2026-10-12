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
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

pub const PAGE_SIZE: u64 = 4096;
pub const ADDRESS_BITS: u32 = 39;
const L1_ENTRIES: usize = 1 << (ADDRESS_BITS - 21);
const L2_ENTRIES: usize = 512;

const PRESENT: u64 = 1 << 3;
const READ: u64 = 1;
const WRITE: u64 = 2;
const EXEC: u64 = 4;
const FLAGS: u64 = 0xFFF;

type L2 = [AtomicU64; L2_ENTRIES];

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
}

pub struct GuestMemory {
    l1: Box<[AtomicPtr<L2>]>,
    blocks: Mutex<Vec<Block>>,
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
            // SAFETY: allocated with this layout in `map`.
            unsafe { dealloc(b.ptr, b.layout) };
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
        let l1: Vec<AtomicPtr<L2>> = std::iter::repeat_with(|| AtomicPtr::new(ptr::null_mut()))
            .take(L1_ENTRIES)
            .collect();
        GuestMemory {
            l1: l1.into_boxed_slice(),
            blocks: Mutex::new(Vec::new()),
        }
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
            || end >> ADDRESS_BITS != 0
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
        // SAFETY: the layout has a non-zero size.
        let host = unsafe { alloc_zeroed(layout) };
        if host.is_null() {
            return Err(MapError::OutOfMemory);
        }
        self.blocks
            .lock()
            .unwrap()
            .push(Block { ptr: host, layout });
        let flags = flags_of(permissions);
        for page in 0..size / PAGE_SIZE {
            let a = addr + page * PAGE_SIZE;
            // SAFETY: inside the block just allocated.
            let p = unsafe { host.add((page * PAGE_SIZE) as usize) } as u64;
            self.l2_for_map(a)[((a >> 12) & 511) as usize].store(p | flags, Ordering::Release);
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
            slot.store((old & !FLAGS) | flags, Ordering::Release);
        }
        Ok(())
    }

    /// Is every page of the range mapped?
    pub fn is_mapped(&self, addr: u64, size: u64) -> bool {
        let first = addr & !(PAGE_SIZE - 1);
        let last = addr.saturating_add(size.max(1) - 1) & !(PAGE_SIZE - 1);
        (first..=last)
            .step_by(PAGE_SIZE as usize)
            .all(|a| self.entry(a) & PRESENT != 0)
    }

    /// Host pointer for `bytes` bytes at `addr` inside one page, if the access is allowed.
    fn page_ptr(&self, addr: u64, bytes: usize, write: bool) -> Result<*mut u8, FaultKind> {
        let e = self.entry(addr);
        if e & PRESENT == 0 {
            return Err(FaultKind::Unmapped);
        }
        if e & if write { WRITE } else { READ } == 0 {
            return Err(FaultKind::Permission);
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
            let p = self.page_ptr(addr, n, false)?;
            // SAFETY: `p` points at `n` readable bytes inside one page.
            return Ok(unsafe { load(p, size, order) });
        }
        let mut buf = [0u8; 16];
        self.read_bytes(addr, &mut buf[..n], false)?;
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
            let p = self.page_ptr(addr, n, true)?;
            // SAFETY: `p` points at `n` writable bytes inside one page.
            unsafe { store(p, size, value, order) };
            return Ok(());
        }
        // Check both pages before writing anything.
        let mut at = addr;
        let end = addr + n as u64;
        while at < end {
            self.page_ptr(at, 0, true)?;
            at = (at | (PAGE_SIZE - 1)) + 1;
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
            if check && e & WRITE == 0 {
                return Err(FaultKind::Permission);
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
        let p = self.page_ptr(addr, usize::from(size), true)?;
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

    /// Does the page at `addr` allow execution?
    pub fn is_executable(&self, addr: u64) -> bool {
        self.entry(addr) & (PRESENT | EXEC) == PRESENT | EXEC
    }
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
        assert_eq!(m.map(1 << 38, 1 << 38, RW), Err(MapError::Range));
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
}
