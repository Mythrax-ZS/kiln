//! GPU address spaces: `/dev/nvhost-as-gpu` (Switchbrew, "NV services", section 8.10).
//!
//! An address space is a range of GPU virtual addresses in which a program reserves regions
//! (`ALLOC_SPACE`) and maps `nvmap` handles (`MAP_BUFFER_EX`). The GPU's own page tables are not
//! modelled: a mapping is a linear window onto the CPU memory behind a handle, and a GPU address
//! resolves to the CPU address of the byte it names.

use crate::error::{NvError, NvResult};
use crate::mem::GuestMemory;
use std::collections::BTreeMap;

pub const SMALL_PAGE: u32 = 0x1000;
pub const BIG_PAGE: u32 = 0x10000;

/// `ALLOC_SPACE` flags.
pub mod alloc_flags {
    pub const FIXED_OFFSET: u32 = 1;
    pub const SPARSE: u32 = 2;
}

/// `MAP_BUFFER_EX` flags.
pub mod map_flags {
    pub const FIXED_OFFSET: u32 = 1;
    pub const CACHEABLE: u32 = 4;
}

/// The range an address space covers when the program never sets one (docs/ASSUMPTIONS.md).
pub const DEFAULT_START: u64 = 0x0400_0000;
pub const DEFAULT_END: u64 = 1 << 40;

/// A region reserved with `ALLOC_SPACE`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub start: u64,
    pub len: u64,
    pub page_size: u32,
    pub sparse: bool,
}

/// A window onto the memory behind an `nvmap` handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapping {
    pub start: u64,
    pub len: u64,
    pub handle: u32,
    /// The CPU address that `start` corresponds to.
    pub cpu: u64,
    pub kind: u32,
    pub page_size: u32,
    pub cacheable: bool,
}

/// What `MAP_BUFFER_EX` asks for, with the handle's backing already looked up.
#[derive(Clone, Copy, Debug)]
pub struct MapRequest {
    pub flags: u32,
    pub kind: u32,
    pub handle: u32,
    /// The CPU address and size of the handle's memory.
    pub handle_cpu: u64,
    pub handle_size: u64,
    /// The page size asked for; 0 lets the address space choose.
    pub page_size: u32,
    pub buffer_offset: u64,
    /// 0 maps the rest of the handle.
    pub mapping_size: u64,
    /// The GPU address, when the mapping is fixed.
    pub offset: u64,
}

/// Where a `REMAP` points a range of an address space.
#[derive(Clone, Copy, Debug)]
pub struct Remap {
    pub handle: u32,
    /// The CPU address the start of the range corresponds to.
    pub cpu: u64,
    pub kind: u32,
    pub cacheable: bool,
}

#[derive(Debug)]
pub struct AddressSpace {
    start: u64,
    end: u64,
    /// Small pages below, big pages from here up.
    split: u64,
    pub big_page_size: u32,
    initialized: bool,
    /// The program chooses every address itself (`ALLOC_AS_EX` flag 0).
    pub userspace_managed: bool,
    regions: BTreeMap<u64, Region>,
    maps: BTreeMap<u64, Mapping>,
}

impl Default for AddressSpace {
    fn default() -> AddressSpace {
        AddressSpace::new()
    }
}

fn valid_page_size(p: u32) -> bool {
    p.is_power_of_two() && (SMALL_PAGE..=2 * BIG_PAGE).contains(&p)
}

/// How many times and over how many bytes memory was hashed since the last [`hash_counts`]: what
/// comparing textures and programs with memory costs (`NX_TRACE_PERF`).
static HASHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HASHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The calls of [`AddressSpace::hash`] and the bytes they covered since the last call.
pub fn hash_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (HASHES.swap(0, Relaxed), HASHED.swap(0, Relaxed))
}

impl AddressSpace {
    pub fn new() -> AddressSpace {
        AddressSpace {
            start: DEFAULT_START,
            end: DEFAULT_END,
            split: DEFAULT_START + (DEFAULT_END - DEFAULT_START) / 2,
            big_page_size: BIG_PAGE,
            initialized: false,
            userspace_managed: false,
            regions: BTreeMap::new(),
            maps: BTreeMap::new(),
        }
    }

    /// `ALLOC_AS_EX`: set the range. Allowed once, before anything is reserved. A range of zeros
    /// keeps the default one. Bit 0 of `flags` says the program manages the addresses itself.
    pub fn initialize(
        &mut self,
        flags: u32,
        big_page_size: u32,
        start: u64,
        end: u64,
        split: u64,
    ) -> NvResult<()> {
        if self.initialized || !self.regions.is_empty() || !self.maps.is_empty() {
            return Err(NvError::INVALID_STATE);
        }
        let (start, end, split) = if (start, end, split) == (0, 0, 0) {
            (self.start, self.end, self.split)
        } else {
            (start, end, split)
        };
        if start >= end || split < start || split > end || end > DEFAULT_END {
            return Err(NvError::BAD_VALUE);
        }
        self.userspace_managed = flags & 1 != 0;
        if big_page_size != 0 {
            if !valid_page_size(big_page_size) || big_page_size < BIG_PAGE {
                return Err(NvError::BAD_VALUE);
            }
            self.big_page_size = big_page_size;
        }
        self.start = start;
        self.end = end;
        self.split = split;
        self.initialized = true;
        Ok(())
    }

    /// `GET_VA_REGIONS`: the small-page range, then the big-page range.
    pub fn va_regions(&self) -> [(u64, u32, u64); 2] {
        let pages = |lo: u64, hi: u64, size: u32| (hi - lo) / u64::from(size);
        [
            (
                self.start,
                SMALL_PAGE,
                pages(self.start, self.split, SMALL_PAGE),
            ),
            (
                self.split,
                self.big_page_size,
                pages(self.split, self.end, self.big_page_size),
            ),
        ]
    }

    pub fn range(&self) -> (u64, u64) {
        (self.start, self.end)
    }

    /// The intervals in use, sorted.
    fn occupied(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self
            .regions
            .values()
            .map(|r| (r.start, r.start + r.len))
            .chain(self.maps.values().map(|m| (m.start, m.start + m.len)))
            .collect();
        v.sort_unstable();
        v
    }

    /// The first free range of `len` bytes aligned to `align` whose page size class is `page_size`.
    fn find_free(&self, len: u64, align: u64, page_size: u32) -> Option<u64> {
        let (lo, hi) = if page_size >= BIG_PAGE {
            (self.split, self.end)
        } else {
            (self.start, self.split)
        };
        let mut candidate = lo.next_multiple_of(align);
        for (s, e) in self.occupied() {
            if e <= candidate {
                continue;
            }
            if s >= candidate.checked_add(len)? {
                break;
            }
            candidate = e.next_multiple_of(align);
        }
        (candidate.checked_add(len)? <= hi).then_some(candidate)
    }

    fn overlaps_region(&self, start: u64, len: u64) -> bool {
        self.regions
            .values()
            .any(|r| r.start < start + len && start < r.start + r.len)
    }

    fn overlaps_map(&self, start: u64, len: u64) -> bool {
        self.maps
            .values()
            .any(|m| m.start < start + len && start < m.start + m.len)
    }

    /// `ALLOC_SPACE`: reserve `pages` pages. `align_or_offset` is the alignment, or the address
    /// when the reservation is fixed. Returns the address.
    pub fn alloc_space(
        &mut self,
        pages: u32,
        page_size: u32,
        flags: u32,
        align_or_offset: u64,
    ) -> NvResult<u64> {
        if pages == 0 || !valid_page_size(page_size) {
            return Err(NvError::BAD_VALUE);
        }
        let len = u64::from(pages) * u64::from(page_size);
        let sparse = flags & alloc_flags::SPARSE != 0;
        let start = if flags & alloc_flags::FIXED_OFFSET != 0 {
            let at = align_or_offset;
            if !at.is_multiple_of(u64::from(page_size)) || at < self.start || at + len > self.end {
                return Err(NvError::BAD_VALUE);
            }
            if self.overlaps_region(at, len) || self.overlaps_map(at, len) {
                return Err(NvError::BUSY);
            }
            at
        } else {
            let align = if align_or_offset == 0 {
                u64::from(page_size)
            } else if align_or_offset.is_power_of_two() {
                align_or_offset.max(u64::from(page_size))
            } else {
                return Err(NvError::BAD_VALUE);
            };
            self.find_free(len, align, page_size)
                .ok_or(NvError::INSUFFICIENT_MEMORY)?
        };
        self.regions.insert(
            start,
            Region {
                start,
                len,
                page_size,
                sparse,
            },
        );
        Ok(start)
    }

    /// `FREE_SPACE`: release a reservation made at exactly `offset`. Mappings inside it go too.
    pub fn free_space(&mut self, offset: u64, pages: u32, page_size: u32) -> NvResult<()> {
        let r = self.regions.get(&offset).ok_or(NvError::BAD_VALUE)?;
        if r.page_size != page_size || r.len != u64::from(pages) * u64::from(page_size) {
            return Err(NvError::BAD_VALUE);
        }
        let end = r.start + r.len;
        self.regions.remove(&offset);
        self.maps.retain(|&s, _| s < offset || s >= end);
        Ok(())
    }

    /// `MAP_BUFFER_EX`: map a handle. Returns the GPU address.
    pub fn map_buffer(&mut self, req: &MapRequest) -> NvResult<u64> {
        // A title maps whole big pages of a handle that is smaller: the mapping may run past the
        // handle up to the next big page (the console accepted a 64 KiB mapping of an 8 KiB
        // handle; docs/ASSUMPTIONS.md A-069).
        let reach = req
            .handle_size
            .next_multiple_of(u64::from(self.big_page_size));
        if req.buffer_offset > req.handle_size {
            return Err(NvError::BAD_VALUE);
        }
        let len = if req.mapping_size == 0 {
            req.handle_size - req.buffer_offset
        } else {
            req.mapping_size
        };
        if len == 0 || req.buffer_offset + len > reach {
            return Err(NvError::BAD_VALUE);
        }
        let len = len.next_multiple_of(u64::from(SMALL_PAGE));
        let page_size = match req.page_size {
            0 if len >= u64::from(BIG_PAGE) => self.big_page_size,
            0 => SMALL_PAGE,
            p if valid_page_size(p) => p,
            _ => return Err(NvError::BAD_VALUE),
        };
        let start = if req.flags & map_flags::FIXED_OFFSET != 0 {
            let at = req.offset;
            if !at.is_multiple_of(u64::from(SMALL_PAGE)) || at < self.start || at + len > self.end {
                return Err(NvError::BAD_VALUE);
            }
            if self.overlaps_map(at, len) {
                return Err(NvError::BUSY);
            }
            at
        } else {
            self.find_free(len, u64::from(page_size), page_size)
                .ok_or(NvError::INSUFFICIENT_MEMORY)?
        };
        self.maps.insert(
            start,
            Mapping {
                start,
                len,
                handle: req.handle,
                cpu: req.handle_cpu + req.buffer_offset,
                kind: req.kind,
                page_size,
                cacheable: req.flags & map_flags::CACHEABLE != 0,
            },
        );
        Ok(start)
    }

    /// `UNMAP_BUFFER`: remove the mapping that starts at `offset`.
    pub fn unmap_buffer(&mut self, offset: u64) -> NvResult<Mapping> {
        self.maps.remove(&offset).ok_or(NvError::BAD_VALUE)
    }

    /// Cut `[start, start + len)` out of the mappings: those inside go, those that stick out on
    /// either side are shortened (a mapping that spans the range becomes two).
    fn carve(&mut self, start: u64, len: u64) {
        let end = start + len;
        let hit: Vec<u64> = self
            .maps
            .range(..end)
            .filter(|(_, m)| m.start + m.len > start)
            .map(|(k, _)| *k)
            .collect();
        for k in hit {
            let Some(m) = self.maps.remove(&k) else {
                continue;
            };
            if m.start < start {
                let left = Mapping {
                    len: start - m.start,
                    ..m.clone()
                };
                self.maps.insert(left.start, left);
            }
            if m.start + m.len > end {
                let right = Mapping {
                    start: end,
                    len: m.start + m.len - end,
                    cpu: m.cpu + (end - m.start),
                    ..m
                };
                self.maps.insert(right.start, right);
            }
        }
    }

    /// `REMAP`: make `[start, start + len)` window onto the CPU memory at `target`'s address
    /// (or onto nothing, when `target` is `None`), replacing whatever was mapped there.
    pub fn remap(&mut self, start: u64, len: u64, target: Option<Remap>) -> NvResult<()> {
        let page = u64::from(self.big_page_size);
        if len == 0
            || !start.is_multiple_of(page)
            || !len.is_multiple_of(page)
            || start < self.start
            || start.checked_add(len).is_none_or(|e| e > self.end)
        {
            return Err(NvError::BAD_VALUE);
        }
        self.carve(start, len);
        if let Some(t) = target {
            self.maps.insert(
                start,
                Mapping {
                    start,
                    len,
                    handle: t.handle,
                    cpu: t.cpu,
                    kind: t.kind,
                    page_size: self.big_page_size,
                    cacheable: t.cacheable,
                },
            );
        }
        Ok(())
    }

    pub fn mappings(&self) -> impl Iterator<Item = &Mapping> {
        self.maps.values()
    }

    pub fn regions(&self) -> impl Iterator<Item = &Region> {
        self.regions.values()
    }

    /// The CPU address of GPU address `va` and how many bytes of the same mapping follow it.
    pub fn translate(&self, va: u64) -> Option<(u64, u64)> {
        let (_, m) = self.maps.range(..=va).next_back()?;
        let off = va - m.start;
        (off < m.len).then(|| (m.cpu + off, m.len - off))
    }

    /// A hash of up to `len` bytes of GPU memory at `va` as they are now, and how many of them
    /// are mapped (the hash covers those). The bytes are not copied when memory can be reached
    /// by address: a texture or a program is compared with what it was made of every frame, which
    /// would otherwise copy megabytes. The same memory hashes the same as long as the mappings
    /// under it stay as they are, which is all a comparison with an earlier call needs.
    pub fn hash(&self, mem: &dyn GuestMemory, mut va: u64, len: u64) -> (u64, u64) {
        HASHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        HASHED.fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        const K: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut h = K;
        let mut done = 0u64;
        while done < len {
            let Some((cpu, avail)) = self.translate(va) else {
                break;
            };
            let wanted = (len - done).min(avail);
            let (part, n) = match mem.host_run(cpu, wanted) {
                // SAFETY: `host_run` gives a pointer valid for the run it names, for as long as
                // the process runs.
                Some((ptr, run)) if run > 0 => unsafe {
                    let run = run.min(wanted);
                    (crate::shaders::hash_raw(ptr, run as usize), run)
                },
                _ => {
                    let mut copy = vec![0u8; wanted as usize];
                    if !mem.read(cpu, &mut copy) {
                        break;
                    }
                    (crate::shaders::hash_bytes(&copy), wanted)
                }
            };
            h = (h ^ part).wrapping_mul(K);
            h ^= h >> 32;
            done += n;
            va += n;
        }
        (h ^ done, done)
    }

    /// Whether none of the `len` bytes of GPU memory at `va` was written since epoch `epoch`
    /// began (see [`GuestMemory::untouched_since`]); `None` when that cannot be told or part of
    /// the range is not mapped.
    pub fn untouched_since(
        &self,
        mem: &dyn GuestMemory,
        mut va: u64,
        len: u64,
        epoch: u32,
    ) -> Option<bool> {
        let mut done = 0u64;
        while done < len {
            let (cpu, avail) = self.translate(va)?;
            let n = (len - done).min(avail);
            if !mem.untouched_since(cpu, n, epoch)? {
                return Some(false);
            }
            done += n;
            va += n;
        }
        Some(true)
    }

    /// Read GPU memory, across mappings that touch. `false` when any byte is unmapped.
    pub fn read(&self, mem: &dyn GuestMemory, mut va: u64, out: &mut [u8]) -> bool {
        let mut done = 0;
        while done < out.len() {
            let Some((cpu, avail)) = self.translate(va) else {
                return false;
            };
            let n = (out.len() - done).min(avail as usize);
            if !mem.read(cpu, &mut out[done..done + n]) {
                return false;
            }
            done += n;
            va += n as u64;
        }
        true
    }

    /// Write GPU memory, across mappings that touch. `false` when any byte is unmapped.
    pub fn write(&self, mem: &dyn GuestMemory, mut va: u64, data: &[u8]) -> bool {
        let mut done = 0;
        while done < data.len() {
            let Some((cpu, avail)) = self.translate(va) else {
                return false;
            };
            let n = (data.len() - done).min(avail as usize);
            if !mem.write(cpu, &data[done..done + n]) {
                return false;
            }
            done += n;
            va += n as u64;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::testing::TestMemory;

    fn req(handle: u32, cpu: u64, size: u64) -> MapRequest {
        MapRequest {
            flags: 0,
            kind: 0,
            handle,
            handle_cpu: cpu,
            handle_size: size,
            page_size: 0,
            buffer_offset: 0,
            mapping_size: 0,
            offset: 0,
        }
    }

    #[test]
    fn the_range_and_regions_follow_alloc_as_ex() {
        let mut a = AddressSpace::new();
        a.initialize(0, 0x20000, 0x0400_0000, 0x10_0000_0000, 0x1_0000_0000)
            .unwrap();
        assert_eq!(a.big_page_size, 0x20000);
        let r = a.va_regions();
        assert_eq!(
            r[0],
            (0x0400_0000, 0x1000, (0x1_0000_0000 - 0x0400_0000) / 0x1000)
        );
        assert_eq!(r[1], (0x1_0000_0000, 0x20000, 0xF_0000_0000 / 0x20000));
        assert_eq!(
            a.initialize(0, 0, 0, 1 << 20, 0),
            Err(NvError::INVALID_STATE),
            "only once"
        );
        let mut b = AddressSpace::new();
        assert_eq!(
            b.initialize(0, 0, 0x2000, 0x1000, 0x2000),
            Err(NvError::BAD_VALUE)
        );
        assert_eq!(
            b.initialize(0, 0x3000, 0, 1 << 30, 1 << 20),
            Err(NvError::BAD_VALUE)
        );
    }

    #[test]
    fn a_zero_range_keeps_the_default_and_the_flag_is_remembered() {
        let mut a = AddressSpace::new();
        assert!(!a.userspace_managed);
        a.initialize(1, 0x10000, 0, 0, 0).unwrap();
        assert!(a.userspace_managed);
        assert_eq!(a.range(), (DEFAULT_START, DEFAULT_END));
        assert_eq!(a.big_page_size, 0x10000);
        let r = a.va_regions();
        assert_eq!(r[0].0, DEFAULT_START);
        assert_eq!(r[1].0, DEFAULT_START + (DEFAULT_END - DEFAULT_START) / 2);
        // A range past what the GPU can address (40 bits) is refused.
        let mut b = AddressSpace::new();
        assert_eq!(
            b.initialize(0, 0, 0x1000, DEFAULT_END + 1, 0x2000),
            Err(NvError::BAD_VALUE)
        );
    }
    #[test]
    fn reservations_are_aligned_distinct_and_in_the_right_half() {
        let mut a = AddressSpace::new();
        a.initialize(0, 0, 0x1_0000, 0x100_0000, 0x80_0000).unwrap();
        let s1 = a.alloc_space(4, SMALL_PAGE, 0, 0).unwrap();
        let s2 = a.alloc_space(2, SMALL_PAGE, 0, 0x4000).unwrap();
        assert_eq!(s1, 0x1_0000);
        assert_eq!(s2, 0x1_4000, "after the first, aligned to 16 KiB");
        let b1 = a.alloc_space(3, BIG_PAGE, 0, 0).unwrap();
        assert_eq!(b1, 0x80_0000, "big pages live in the upper half");
        assert_eq!(a.alloc_space(0, SMALL_PAGE, 0, 0), Err(NvError::BAD_VALUE));
        assert_eq!(a.alloc_space(1, 0x3000, 0, 0), Err(NvError::BAD_VALUE));
        assert_eq!(a.alloc_space(1, SMALL_PAGE, 0, 3), Err(NvError::BAD_VALUE));
    }

    #[test]
    fn fixed_reservations_must_not_overlap() {
        let mut a = AddressSpace::new();
        a.initialize(0, 0, 0x1_0000, 0x100_0000, 0x80_0000).unwrap();
        let f = alloc_flags::FIXED_OFFSET;
        assert_eq!(a.alloc_space(16, BIG_PAGE, f, 0x20_0000), Ok(0x20_0000));
        assert_eq!(
            a.alloc_space(1, BIG_PAGE, f, 0x20_0000 + 0x10_0000 - 0x10000),
            Err(NvError::BUSY)
        );
        assert_eq!(
            a.alloc_space(1, BIG_PAGE, f, 0x20_0001),
            Err(NvError::BAD_VALUE)
        );
        assert_eq!(
            a.alloc_space(1, BIG_PAGE, f, 0x1000),
            Err(NvError::BAD_VALUE)
        );
        // A free search skips it.
        let next = a.alloc_space(1, BIG_PAGE, 0, 0).unwrap();
        assert!(next >= 0x80_0000 || next + 0x10000 <= 0x20_0000 || next >= 0x30_0000);
    }

    #[test]
    fn freeing_a_reservation_takes_its_mappings_with_it() {
        let mut a = AddressSpace::new();
        a.initialize(0, 0, 0x1_0000, 0x100_0000, 0x80_0000).unwrap();
        let s = a.alloc_space(8, SMALL_PAGE, 0, 0).unwrap();
        let mut r = req(1, 0x1000_0000, 0x2000);
        r.flags = map_flags::FIXED_OFFSET;
        r.offset = s;
        a.map_buffer(&r).unwrap();
        assert_eq!(a.free_space(s, 8, SMALL_PAGE + 1), Err(NvError::BAD_VALUE));
        a.free_space(s, 8, SMALL_PAGE).unwrap();
        assert_eq!(a.translate(s), None);
        assert_eq!(a.free_space(s, 8, SMALL_PAGE), Err(NvError::BAD_VALUE));
    }

    #[test]
    fn mapping_translates_to_the_cpu_address_of_the_handle() {
        let mut a = AddressSpace::new();
        let mut r = req(7, 0x2000_0000, 0x4000);
        r.buffer_offset = 0x1000;
        r.mapping_size = 0x2000;
        let va = a.map_buffer(&r).unwrap();
        assert_eq!(a.translate(va), Some((0x2000_1000, 0x2000)));
        assert_eq!(a.translate(va + 0x1800), Some((0x2000_2800, 0x800)));
        assert_eq!(a.translate(va + 0x2000), None);
        assert_eq!(a.translate(va - 1), None);
        assert_eq!(a.unmap_buffer(va).unwrap().handle, 7);
        assert_eq!(a.translate(va), None);
        assert_eq!(a.unmap_buffer(va), Err(NvError::BAD_VALUE));
    }

    #[test]
    fn map_requests_are_checked_against_the_handle() {
        let mut a = AddressSpace::new();
        let mut r = req(1, 0x1000, 0x2000);
        r.buffer_offset = 0x3000;
        assert_eq!(a.map_buffer(&r), Err(NvError::BAD_VALUE));
        r.buffer_offset = 0x1000;
        // Up to the next big page past the handle is accepted (the title maps 64 KiB of an
        // 8 KiB handle); beyond that is not.
        r.mapping_size = 0xF000;
        assert!(a.map_buffer(&r).is_ok());
        r.mapping_size = 0x10000;
        assert_eq!(a.map_buffer(&r), Err(NvError::BAD_VALUE));
        r.mapping_size = 0;
        r.page_size = 0x3000;
        assert_eq!(a.map_buffer(&r), Err(NvError::BAD_VALUE));
        r.page_size = 0;
        assert!(a.map_buffer(&r).is_ok(), "the rest of the handle");
    }

    #[test]
    fn fixed_mappings_do_not_overlap_each_other() {
        let mut a = AddressSpace::new();
        let mut r = req(1, 0x1000, 0x4000);
        r.flags = map_flags::FIXED_OFFSET | map_flags::CACHEABLE;
        r.offset = 0x500_0000;
        a.map_buffer(&r).unwrap();
        r.offset = 0x500_3000;
        assert_eq!(a.map_buffer(&r), Err(NvError::BUSY));
        r.offset = 0x500_4000;
        assert_eq!(a.map_buffer(&r), Ok(0x500_4000));
        assert!(a.mappings().next().unwrap().cacheable);
    }

    #[test]
    fn big_handles_get_big_page_addresses() {
        let mut a = AddressSpace::new();
        let small = a.map_buffer(&req(1, 0x1000, 0x2000)).unwrap();
        let big = a.map_buffer(&req(2, 0x10_0000, 0x20000)).unwrap();
        assert!(small < a.va_regions()[1].0);
        assert!(big >= a.va_regions()[1].0);
        assert!(big.is_multiple_of(0x10000));
        // Exactly one big page is the smallest that gets one; a page less does not.
        let exact = a.map_buffer(&req(3, 0x1_0000, 0x1_0000)).unwrap();
        let under = a.map_buffer(&req(4, 0xF000, 0xF000)).unwrap();
        assert!(exact >= a.va_regions()[1].0 && exact.is_multiple_of(0x10000));
        assert!(under < a.va_regions()[1].0);
    }

    #[test]
    fn free_placement_rounds_up_to_the_page_size_after_an_odd_start_or_end() {
        // A split that is not a multiple of the (128 KiB) big page: the first big mapping starts at
        // the next multiple.
        let mut a = AddressSpace::new();
        a.initialize(0, 0x20000, 0x0400_0000, 0x10_0000_0000, 0x1_0001_0000)
            .unwrap();
        let first = a.map_buffer(&req(1, 0x1000_0000, 0x20000)).unwrap();
        assert_eq!(first, 0x1_0002_0000);
        // A big-page mapping that ends between pages (lengths round to 4 KiB): the next one starts
        // at the following multiple of its page size, not at the odd end.
        let mut odd = req(2, 0x2000_0000, 0x21000);
        odd.page_size = 0x20000;
        let odd_at = a.map_buffer(&odd).unwrap();
        assert!(odd_at.is_multiple_of(0x20000));
        let mut next = req(3, 0x3000_0000, 0x20000);
        next.page_size = 0x20000;
        let next_at = a.map_buffer(&next).unwrap();
        assert!(next_at.is_multiple_of(0x20000), "{next_at:#x}");
        assert!(next_at >= odd_at + 0x21000);
    }

    fn remap_to(handle: u32, cpu: u64) -> Option<Remap> {
        Some(Remap {
            handle,
            cpu,
            kind: 0,
            cacheable: true,
        })
    }

    #[test]
    fn remap_replaces_what_was_there_and_shortens_what_sticks_out() {
        let mut a = AddressSpace::new();
        let page = 0x10000u64;
        let base = 0x100_0000_0000 / 2; // inside the default range, 64 KiB aligned
                                        // One handle mapped over four pages.
        a.remap(base, 4 * page, remap_to(1, 0x1_0000_0000)).unwrap();
        assert_eq!(a.translate(base + 3 * page), Some((0x1_0003_0000, page)));
        // A different handle over the middle two: the first is cut into two pieces around it.
        a.remap(base + page, 2 * page, remap_to(2, 0x2_0000_0000))
            .unwrap();
        assert_eq!(a.translate(base), Some((0x1_0000_0000, page)));
        assert_eq!(a.translate(base + page), Some((0x2_0000_0000, 2 * page)));
        assert_eq!(a.translate(base + 3 * page), Some((0x1_0003_0000, page)));
        assert_eq!(a.mappings().count(), 3);
        // Nothing (a sparse hole) over the first page and half of the second.
        a.remap(base, page, None).unwrap();
        assert_eq!(a.translate(base), None);
        assert_eq!(a.translate(base + page), Some((0x2_0000_0000, 2 * page)));
        // A range that overlaps the end of one mapping and the start of the next.
        a.remap(base + 2 * page, 2 * page, remap_to(3, 0x3_0000_0000))
            .unwrap();
        assert_eq!(a.translate(base + page), Some((0x2_0000_0000, page)));
        assert_eq!(
            a.translate(base + 2 * page),
            Some((0x3_0000_0000, 2 * page))
        );
        assert_eq!(a.translate(base + 4 * page), None);
    }

    #[test]
    fn remap_ranges_are_checked() {
        let mut a = AddressSpace::new();
        let page = 0x10000u64;
        assert_eq!(a.remap(0x500_0001, page, None), Err(NvError::BAD_VALUE));
        assert_eq!(a.remap(0x500_0000, page - 1, None), Err(NvError::BAD_VALUE));
        assert_eq!(a.remap(0x500_0000, 0, None), Err(NvError::BAD_VALUE));
        assert_eq!(
            a.remap(0x1_0000, page, None),
            Err(NvError::BAD_VALUE),
            "below the range"
        );
        assert_eq!(
            a.remap(DEFAULT_END, page, None),
            Err(NvError::BAD_VALUE),
            "above it"
        );
        assert_eq!(
            a.remap(u64::MAX - page + 1, page, None),
            Err(NvError::BAD_VALUE)
        );
    }

    #[test]
    fn gpu_reads_and_writes_cross_adjacent_mappings() {
        let mem = TestMemory::new();
        mem.map(0x1000_0000, 0x1000);
        mem.map(0x2000_0000, 0x1000);
        let mut a = AddressSpace::new();
        let mut r1 = req(1, 0x1000_0000, 0x1000);
        r1.flags = map_flags::FIXED_OFFSET;
        r1.offset = 0x500_0000;
        let mut r2 = req(2, 0x2000_0000, 0x1000);
        r2.flags = map_flags::FIXED_OFFSET;
        r2.offset = 0x500_1000;
        a.map_buffer(&r1).unwrap();
        a.map_buffer(&r2).unwrap();
        let data: Vec<u8> = (0..16).collect();
        assert!(a.write(&mem, 0x500_0FF8, &data));
        let mut back = [0u8; 16];
        assert!(a.read(&mem, 0x500_0FF8, &mut back));
        assert_eq!(&back[..], &data[..]);
        let mut tail = [0u8; 8];
        assert!(mem.read(0x2000_0000, &mut tail));
        assert_eq!(&tail[..], &data[8..]);
        let mut past = [0u8; 8];
        assert!(
            !a.read(&mem, 0x500_1FFC, &mut past),
            "runs off the second mapping"
        );
    }

    /// Memory that can be reached by address, as the guest's is: one block at `base`.
    struct Direct {
        base: u64,
        data: std::sync::Mutex<Box<[u8]>>,
        ptr: *mut u8,
    }

    // SAFETY: the pointer is into the boxed bytes, which live as long as the value; the tests
    // touch the bytes from one thread at a time.
    unsafe impl Send for Direct {}
    unsafe impl Sync for Direct {}

    impl Direct {
        fn new(base: u64, size: usize) -> Direct {
            let mut data = vec![0u8; size].into_boxed_slice();
            let ptr = data.as_mut_ptr();
            Direct {
                base,
                data: std::sync::Mutex::new(data),
                ptr,
            }
        }
    }

    impl GuestMemory for Direct {
        fn read(&self, addr: u64, out: &mut [u8]) -> bool {
            let data = self.data.lock().unwrap();
            let Some(off) = addr.checked_sub(self.base).map(|o| o as usize) else {
                return false;
            };
            match data.get(off..off + out.len()) {
                Some(src) => {
                    out.copy_from_slice(src);
                    true
                }
                None => false,
            }
        }

        fn write(&self, addr: u64, src: &[u8]) -> bool {
            let mut data = self.data.lock().unwrap();
            let Some(off) = addr.checked_sub(self.base).map(|o| o as usize) else {
                return false;
            };
            match data.get_mut(off..off + src.len()) {
                Some(dst) => {
                    dst.copy_from_slice(src);
                    true
                }
                None => false,
            }
        }

        fn host_run(&self, addr: u64, max: u64) -> Option<(*mut u8, u64)> {
            let size = self.data.lock().unwrap().len() as u64;
            let off = addr.checked_sub(self.base)?;
            (off < size).then(|| {
                // SAFETY: `off` is inside the block.
                (unsafe { self.ptr.add(off as usize) }, max.min(size - off))
            })
        }
    }

    fn space_over(cpu: u64, size: u64, va: u64) -> AddressSpace {
        let mut a = AddressSpace::new();
        let mut r = req(1, cpu, size);
        r.flags = map_flags::FIXED_OFFSET;
        r.offset = va;
        a.map_buffer(&r).unwrap();
        a
    }

    #[test]
    fn a_hash_of_memory_notices_a_change_anywhere_by_either_route() {
        let by_address = Direct::new(0x1000_0000, 0x3000);
        let by_copy = TestMemory::new();
        by_copy.map(0x1000_0000, 0x3000);
        let a = space_over(0x1000_0000, 0x3000, 0x500_0000);
        let pattern: Vec<u8> = (0..0x3000u32).map(|i| (i * 7 + i / 251) as u8).collect();
        assert!(by_address.write(0x1000_0000, &pattern));
        assert!(by_copy.write(0x1000_0000, &pattern));
        let (h, n) = a.hash(&by_address, 0x500_0000, 0x3000);
        assert_eq!(n, 0x3000);
        assert_eq!(
            a.hash(&by_copy, 0x500_0000, 0x3000),
            (h, n),
            "the route is not part of the hash"
        );
        assert_eq!(
            a.hash(&by_address, 0x500_0000, 0x3000),
            (h, n),
            "and it is stable"
        );
        for at in [0u64, 1, 31, 32, 33, 0x1234, 0x2FDF, 0x2FFF] {
            let old = pattern[at as usize];
            assert!(by_address.write(0x1000_0000 + at, &[old ^ 0x10]));
            assert_ne!(
                a.hash(&by_address, 0x500_0000, 0x3000).0,
                h,
                "byte {at:#x} changed"
            );
            assert!(by_address.write(0x1000_0000 + at, &[old]));
            assert_eq!(a.hash(&by_address, 0x500_0000, 0x3000).0, h, "and restored");
        }
    }

    #[test]
    fn a_hash_covers_what_is_mapped_and_says_how_much_that_is() {
        let mem = Direct::new(0x1000_0000, 0x3000);
        let a = space_over(0x1000_0000, 0x3000, 0x500_0000);
        let (whole, n) = a.hash(&mem, 0x500_0000, 0x3000);
        assert_eq!(n, 0x3000);
        let (past, n) = a.hash(&mem, 0x500_0000, 0x4000);
        assert_eq!(n, 0x3000, "the rest is not mapped");
        assert_eq!(
            past, whole,
            "the hash is of what is mapped, not of the length asked for"
        );
        assert_eq!(
            a.hash(&mem, 0x600_0000, 0x1000).1,
            0,
            "nothing is mapped there"
        );
        assert_eq!(a.hash(&mem, 0x500_0000, 0).1, 0);
    }

    #[test]
    fn the_hash_of_bytes_depends_on_every_byte_and_on_the_length() {
        let zeros = [0u8; 80];
        let mut seen = std::collections::HashSet::new();
        for len in 0..=80 {
            assert!(
                seen.insert(crate::shaders::hash_bytes(&zeros[..len])),
                "{len} zeros hash like fewer"
            );
        }
        let mut bytes = [0u8; 70];
        let base = crate::shaders::hash_bytes(&bytes);
        for at in 0..70 {
            bytes[at] = 1;
            assert_ne!(crate::shaders::hash_bytes(&bytes), base, "byte {at}");
            bytes[at] = 0;
        }
    }
}
