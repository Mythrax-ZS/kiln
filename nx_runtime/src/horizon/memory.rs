//! The kernel's account of what is mapped where: a map of blocks with a memory type, permissions
//! and attributes, which is what `QueryMemory` answers from and what `SetMemoryPermission`,
//! `MapMemory` and the heap calls check before touching the guest pages themselves.
//!
//! Neighbouring blocks with the same type, permissions and attributes are kept merged, so a query
//! returns the maximal range the real kernel would report.

use super::layout::{mem_type, ADDRESS_SPACE_END, ASLR_BASE};
use std::collections::BTreeMap;

/// What a block of pages is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info {
    pub mem_type: u8,
    /// `rt::abi::perm` bits: read 1, write 2, execute 4 (the values `MemoryPermission` has).
    pub perm: u32,
    pub attr: u32,
}

/// One answer of `QueryMemory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    pub base: u64,
    pub size: u64,
    pub info: Info,
}

#[derive(Default)]
pub struct MemoryMap {
    /// Start address to (size, info); blocks never overlap and are never adjacent with equal info.
    blocks: BTreeMap<u64, (u64, Info)>,
}

impl MemoryMap {
    pub fn new() -> MemoryMap {
        MemoryMap::default()
    }

    /// Ensure a block boundary at `at` (a no-op when there is one or no block covers it).
    fn split(&mut self, at: u64) {
        let Some((&start, &(size, info))) = self.blocks.range(..at).next_back() else {
            return;
        };
        if start + size > at {
            self.blocks.insert(start, (at - start, info));
            self.blocks.insert(at, (start + size - at, info));
        }
    }

    /// Merge the block starting at `addr` with its neighbours where they are equal.
    fn merge_around(&mut self, addr: u64) {
        let mut at = addr;
        if let Some((&prev, &(psize, pinfo))) = self.blocks.range(..addr).next_back() {
            if prev + psize == addr && self.blocks.get(&addr).is_some_and(|b| b.1 == pinfo) {
                let (size, _) = self.blocks.remove(&addr).expect("present");
                self.blocks.insert(prev, (psize + size, pinfo));
                at = prev;
            }
        }
        if let Some(&(size, info)) = self.blocks.get(&at) {
            let end = at + size;
            if let Some(&(nsize, ninfo)) = self.blocks.get(&end) {
                if ninfo == info {
                    self.blocks.remove(&end);
                    self.blocks.insert(at, (size + nsize, info));
                }
            }
        }
    }

    /// Add a block over a range nothing else covers. False if any part is already mapped.
    pub fn insert(&mut self, addr: u64, size: u64, info: Info) -> bool {
        let Some(end) = addr.checked_add(size) else {
            return false;
        };
        if size == 0 || end > ADDRESS_SPACE_END {
            return false;
        }
        if let Some((&start, &(s, _))) = self.blocks.range(..end).next_back() {
            if start + s > addr {
                return false;
            }
        }
        self.blocks.insert(addr, (size, info));
        self.merge_around(addr);
        true
    }

    /// Is every page of the range covered by blocks?
    pub fn is_mapped(&self, addr: u64, size: u64) -> bool {
        let Some(end) = addr.checked_add(size) else {
            return false;
        };
        let mut at = addr;
        while at < end {
            match self.find(at) {
                Some((start, bsize, _)) => at = start + bsize,
                None => return false,
            }
        }
        true
    }

    /// The block covering `addr`: `(start, size, info)`.
    pub fn find(&self, addr: u64) -> Option<(u64, u64, Info)> {
        let (&start, &(size, info)) = self.blocks.range(..=addr).next_back()?;
        (addr < start + size).then_some((start, size, info))
    }

    /// The blocks overlapping the range, clipped to it, in address order.
    pub fn blocks_in(&self, addr: u64, size: u64) -> Vec<(u64, u64, Info)> {
        let end = addr.saturating_add(size);
        let mut out = Vec::new();
        let first = self.find(addr).map_or(addr, |b| b.0);
        for (&start, &(bsize, info)) in self.blocks.range(first..end) {
            let lo = start.max(addr);
            let hi = (start + bsize).min(end);
            if lo < hi {
                out.push((lo, hi - lo, info));
            }
        }
        out
    }

    /// Remove the mapped range; it must be wholly mapped. Returns false (and changes nothing)
    /// otherwise.
    pub fn remove(&mut self, addr: u64, size: u64) -> bool {
        if !self.is_mapped(addr, size) {
            return false;
        }
        let end = addr + size;
        self.split(addr);
        self.split(end);
        let inside: Vec<u64> = self.blocks.range(addr..end).map(|(&s, _)| s).collect();
        for s in inside {
            self.blocks.remove(&s);
        }
        true
    }

    /// Change the info of a wholly mapped range. Returns false (and changes nothing) otherwise.
    pub fn update(&mut self, addr: u64, size: u64, f: impl Fn(Info) -> Info) -> bool {
        if !self.is_mapped(addr, size) {
            return false;
        }
        let end = addr + size;
        self.split(addr);
        self.split(end);
        let inside: Vec<u64> = self.blocks.range(addr..end).map(|(&s, _)| s).collect();
        for s in &inside {
            let entry = self.blocks.get_mut(s).expect("present");
            entry.1 = f(entry.1);
        }
        for s in inside {
            if self.blocks.contains_key(&s) {
                self.merge_around(s);
            }
        }
        self.merge_around(addr);
        true
    }

    /// The answer `QueryMemory` gives for `addr`.
    pub fn query(&self, addr: u64) -> MemoryInfo {
        let inaccessible = Info {
            mem_type: mem_type::INACCESSIBLE,
            perm: 0,
            attr: 0,
        };
        if addr >= ADDRESS_SPACE_END {
            return MemoryInfo {
                base: ADDRESS_SPACE_END,
                size: ADDRESS_SPACE_END.wrapping_neg(),
                info: inaccessible,
            };
        }
        if addr < ASLR_BASE {
            return MemoryInfo {
                base: 0,
                size: ASLR_BASE,
                info: inaccessible,
            };
        }
        if let Some((base, size, info)) = self.find(addr) {
            return MemoryInfo { base, size, info };
        }
        // A gap: from the end of the previous block (or the start of the usable space) to the
        // start of the next one.
        let base = self
            .blocks
            .range(..addr)
            .next_back()
            .map_or(ASLR_BASE, |(&s, &(size, _))| (s + size).max(ASLR_BASE));
        let end = self
            .blocks
            .range(addr..)
            .next()
            .map_or(ADDRESS_SPACE_END, |(&s, _)| s);
        MemoryInfo {
            base,
            size: end - base,
            info: Info {
                mem_type: mem_type::FREE,
                perm: 0,
                attr: 0,
            },
        }
    }

    /// Total size of the blocks whose type satisfies `pred`.
    pub fn total(&self, pred: impl Fn(u8) -> bool) -> u64 {
        self.blocks
            .values()
            .filter(|(_, i)| pred(i.mem_type))
            .map(|(s, _)| s)
            .sum()
    }

    /// The lowest address of a free range of `size` bytes inside `[lo, hi)` aligned to `align`.
    pub fn find_free(&self, lo: u64, hi: u64, size: u64, align: u64) -> Option<u64> {
        let mut at = lo.div_ceil(align) * align;
        while at.checked_add(size)? <= hi {
            match self.blocks_in(at, size).first() {
                None => return Some(at),
                Some(&(start, bsize, _)) => {
                    let next = self.find(start).map_or(start + bsize, |b| b.0 + b.1);
                    at = next.div_ceil(align) * align;
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RW: u32 = 3;
    const RX: u32 = 5;
    const R: u32 = 1;

    fn info(t: u8, perm: u32) -> Info {
        Info {
            mem_type: t,
            perm,
            attr: 0,
        }
    }

    #[test]
    fn neighbours_with_equal_info_merge_and_differing_ones_do_not() {
        let mut m = MemoryMap::new();
        assert!(m.insert(0x0800_0000, 0x2000, info(mem_type::CODE, RX)));
        assert!(m.insert(0x0800_2000, 0x1000, info(mem_type::CODE, R)));
        assert!(m.insert(0x0800_3000, 0x1000, info(mem_type::CODE_DATA, RW)));
        assert!(m.insert(0x0800_4000, 0x1000, info(mem_type::CODE_DATA, RW)));
        let q = m.query(0x0800_1000);
        assert_eq!((q.base, q.size, q.info.perm), (0x0800_0000, 0x2000, RX));
        let q = m.query(0x0800_3800);
        assert_eq!((q.base, q.size, q.info.mem_type), (0x0800_3000, 0x2000, 4));
        // Overlaps are refused.
        assert!(!m.insert(0x0800_1000, 0x1000, info(mem_type::NORMAL, RW)));
        assert!(!m.insert(0x07FF_F000, 0x2000, info(mem_type::NORMAL, RW)));
    }

    #[test]
    fn gaps_are_free_and_the_ends_are_inaccessible() {
        let mut m = MemoryMap::new();
        m.insert(0x0800_0000, 0x1000, info(mem_type::CODE, RX));
        m.insert(0x0801_0000, 0x1000, info(mem_type::CODE, R));
        let q = m.query(0x0800_5000);
        assert_eq!((q.base, q.size, q.info.mem_type), (0x0800_1000, 0xF000, 0));
        let q = m.query(0x0801_1000);
        assert_eq!(q.base, 0x0801_1000);
        assert_eq!(q.base + q.size, ADDRESS_SPACE_END);
        let q = m.query(0x1000);
        assert_eq!((q.base, q.size, q.info.mem_type), (0, 0x0800_0000, 0x10));
        let q = m.query(ADDRESS_SPACE_END + 5);
        assert_eq!((q.base, q.info.mem_type), (ADDRESS_SPACE_END, 0x10));
        assert_eq!(q.base.wrapping_add(q.size), 0);
    }

    #[test]
    fn updating_part_of_a_block_splits_it_and_merging_restores_it() {
        let mut m = MemoryMap::new();
        m.insert(0x3000_0000, 0x4000, info(mem_type::NORMAL, RW));
        assert!(m.update(0x3000_1000, 0x1000, |i| Info { perm: R, ..i }));
        assert_eq!(m.query(0x3000_0000).size, 0x1000);
        assert_eq!(m.query(0x3000_1000).info.perm, R);
        assert_eq!(m.query(0x3000_2000).size, 0x2000);
        assert!(m.update(0x3000_1000, 0x1000, |i| Info { perm: RW, ..i }));
        assert_eq!(m.query(0x3000_0000).size, 0x4000);
        // A range that is not wholly mapped changes nothing.
        assert!(!m.update(0x3000_3000, 0x2000, |i| Info { perm: 0, ..i }));
        assert_eq!(m.query(0x3000_3000).info.perm, RW);
    }

    #[test]
    fn removing_and_reinserting() {
        let mut m = MemoryMap::new();
        m.insert(0x3000_0000, 0x4000, info(mem_type::NORMAL, RW));
        assert!(m.remove(0x3000_2000, 0x2000));
        assert_eq!(m.query(0x3000_0000).size, 0x2000);
        assert_eq!(m.query(0x3000_2000).info.mem_type, 0);
        assert!(!m.remove(0x3000_0000, 0x3000));
        assert!(m.insert(0x3000_2000, 0x2000, info(mem_type::NORMAL, RW)));
        assert_eq!(m.query(0x3000_0000).size, 0x4000);
        assert_eq!(m.total(|t| t == mem_type::NORMAL), 0x4000);
    }

    #[test]
    fn finding_free_space() {
        let mut m = MemoryMap::new();
        m.insert(0x1000_0000, 0x10_0000, info(mem_type::STACK, RW));
        m.insert(0x1020_0000, 0x1000, info(mem_type::STACK, RW));
        assert_eq!(
            m.find_free(0x1000_0000, 0x2000_0000, 0x10_0000, 0x1000),
            Some(0x1010_0000)
        );
        assert_eq!(
            m.find_free(0x1000_0000, 0x2000_0000, 0x20_0000, 0x1000),
            Some(0x1020_1000)
        );
        assert_eq!(
            m.find_free(0x1000_0000, 0x1010_0000, 0x10_0000, 0x1000),
            None
        );
    }
}
