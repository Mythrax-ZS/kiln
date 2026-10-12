//! `/dev/nvmap`: handles to memory the GPU can use (Switchbrew, "NV services", section 8.2).
//!
//! A program creates a handle with a size, then backs it with memory of its own by giving
//! `NVMAP_IOC_ALLOC` the CPU address of a buffer. The GPU address spaces map handles, and the
//! GPU reads and writes the memory at that CPU address. Nothing is copied.

use crate::error::{NvError, NvResult};
use std::collections::BTreeMap;

/// Handles are sized in whole CPU pages.
pub const PAGE: u64 = 0x1000;

/// `NVMAP_IOC_PARAM` selectors.
pub mod param {
    pub const SIZE: u32 = 1;
    pub const ALIGNMENT: u32 = 2;
    pub const BASE: u32 = 3;
    pub const HEAP: u32 = 4;
    pub const KIND: u32 = 5;
    pub const COMPR: u32 = 6;
}

/// The heap `NVMAP_IOC_PARAM` reports for every handle (Switchbrew: "always 0x40000000").
pub const HEAP_ID: u32 = 0x4000_0000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Handle {
    pub id: u32,
    /// Bytes, a multiple of [`PAGE`].
    pub size: u32,
    pub align: u32,
    pub kind: u8,
    pub heap_mask: u32,
    pub flags: u32,
    /// The CPU address of the backing memory, once allocated.
    pub addr: Option<u64>,
    refs: u32,
}

/// What `NVMAP_IOC_FREE` reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Freed {
    /// The backing address when this call released the last reference, else 0.
    pub addr: u64,
    pub size: u32,
    /// 0 when the handle is gone, 1 while references remain.
    pub flags: u32,
}

#[derive(Debug, Default)]
pub struct Nvmap {
    handles: BTreeMap<u32, Handle>,
    next_id: u32,
}

impl Nvmap {
    pub fn new() -> Nvmap {
        Nvmap {
            handles: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// `NVMAP_IOC_CREATE`: a handle of `size` bytes rounded up to whole pages.
    pub fn create(&mut self, size: u32) -> NvResult<u32> {
        if size == 0 {
            return Err(NvError::BAD_PARAMETER);
        }
        let rounded = u64::from(size).next_multiple_of(PAGE);
        let size = u32::try_from(rounded).map_err(|_| NvError::BAD_PARAMETER)?;
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(NvError::RESOURCE_ERROR)?;
        self.handles.insert(
            id,
            Handle {
                id,
                size,
                align: 0,
                kind: 0,
                heap_mask: 0,
                flags: 0,
                addr: None,
                refs: 1,
            },
        );
        Ok(id)
    }

    /// `NVMAP_IOC_ALLOC`: back the handle with memory at `addr`.
    pub fn alloc(
        &mut self,
        handle: u32,
        heap_mask: u32,
        flags: u32,
        align: u32,
        kind: u8,
        addr: u64,
    ) -> NvResult<()> {
        let h = self.handles.get_mut(&handle).ok_or(NvError::BAD_VALUE)?;
        if h.addr.is_some() {
            return Err(NvError::ALREADY_ALLOCATED);
        }
        // The alignment is a power of two (zero means a page) and the address obeys it.
        let align = if align == 0 { PAGE as u32 } else { align };
        if !align.is_power_of_two() || !addr.is_multiple_of(u64::from(align.max(PAGE as u32))) {
            return Err(NvError::BAD_VALUE);
        }
        h.heap_mask = heap_mask;
        h.flags = flags;
        h.align = align;
        h.kind = kind;
        h.addr = Some(addr);
        Ok(())
    }

    /// `NVMAP_IOC_PARAM`.
    pub fn param(&self, handle: u32, which: u32) -> NvResult<u32> {
        let h = self.handles.get(&handle).ok_or(NvError::BAD_VALUE)?;
        match which {
            param::SIZE => Ok(h.size),
            param::ALIGNMENT => Ok(h.align),
            param::HEAP => Ok(HEAP_ID),
            param::KIND => Ok(u32::from(h.kind)),
            // "BASE (returns error)", "COMPR (unused)" in Switchbrew.
            param::BASE | param::COMPR => Err(NvError::NOT_SUPPORTED),
            _ => Err(NvError::BAD_VALUE),
        }
    }

    /// `NVMAP_IOC_GET_ID`: the id other processes use to find the handle. Here it is the handle.
    pub fn get_id(&self, handle: u32) -> NvResult<u32> {
        self.handles
            .get(&handle)
            .map(|h| h.id)
            .ok_or(NvError::BAD_VALUE)
    }

    /// `NVMAP_IOC_FROM_ID`: a new reference to the handle with that id.
    pub fn from_id(&mut self, id: u32) -> NvResult<u32> {
        let h = self.handles.get_mut(&id).ok_or(NvError::BAD_VALUE)?;
        h.refs += 1;
        Ok(h.id)
    }

    /// `NVMAP_IOC_FREE`: drop one reference.
    pub fn free(&mut self, handle: u32) -> NvResult<Freed> {
        let h = self.handles.get_mut(&handle).ok_or(NvError::BAD_VALUE)?;
        h.refs -= 1;
        if h.refs > 0 {
            return Ok(Freed {
                addr: 0,
                size: h.size,
                flags: 1,
            });
        }
        let h = self.handles.remove(&handle).ok_or(NvError::BAD_VALUE)?;
        Ok(Freed {
            addr: h.addr.unwrap_or(0),
            size: h.size,
            flags: 0,
        })
    }

    pub fn get(&self, handle: u32) -> Option<&Handle> {
        self.handles.get(&handle)
    }

    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_created_handle_has_a_page_rounded_size_and_no_memory() {
        let mut m = Nvmap::new();
        let h = m.create(0x1234).unwrap();
        let handle = m.get(h).unwrap();
        assert_eq!(handle.size, 0x2000);
        assert_eq!(handle.addr, None);
        assert_eq!(m.param(h, param::SIZE), Ok(0x2000));
        assert_eq!(m.create(0), Err(NvError::BAD_PARAMETER));
        assert_eq!(m.create(u32::MAX), Err(NvError::BAD_PARAMETER));
    }

    #[test]
    fn alloc_binds_memory_once_and_checks_alignment() {
        let mut m = Nvmap::new();
        let h = m.create(0x10000).unwrap();
        assert_eq!(
            m.alloc(h, 0, 1, 0x10000, 0, 0x1_0001_0000 + 0x1000),
            Err(NvError::BAD_VALUE),
            "address not aligned to the requested 64 KiB"
        );
        assert_eq!(m.alloc(h, 0, 1, 0x3000, 0, 0x1000), Err(NvError::BAD_VALUE));
        assert_eq!(m.alloc(h, 0x1, 1, 0x1000, 6, 0x1_0000_0000), Ok(()));
        assert_eq!(m.get(h).unwrap().addr, Some(0x1_0000_0000));
        assert_eq!(
            m.alloc(h, 0, 1, 0x1000, 0, 0x2000),
            Err(NvError::ALREADY_ALLOCATED)
        );
        assert_eq!(
            m.alloc(99, 0, 1, 0x1000, 0, 0x2000),
            Err(NvError::BAD_VALUE)
        );
        assert_eq!(m.param(h, param::KIND), Ok(6));
        assert_eq!(m.param(h, param::ALIGNMENT), Ok(0x1000));
    }

    #[test]
    fn param_answers_the_documented_selectors() {
        let mut m = Nvmap::new();
        let h = m.create(0x1000).unwrap();
        assert_eq!(m.param(h, param::HEAP), Ok(0x4000_0000));
        assert_eq!(m.param(h, param::BASE), Err(NvError::NOT_SUPPORTED));
        assert_eq!(m.param(h, 77), Err(NvError::BAD_VALUE));
        assert_eq!(m.param(1234, param::SIZE), Err(NvError::BAD_VALUE));
    }

    #[test]
    fn ids_and_references_follow_the_handle() {
        let mut m = Nvmap::new();
        let h = m.create(0x2000).unwrap();
        m.alloc(h, 0, 1, 0x1000, 0, 0x4000_0000).unwrap();
        let id = m.get_id(h).unwrap();
        assert_eq!(m.from_id(id), Ok(h));
        // Two references: the first free leaves the handle alive and reports nothing freed.
        assert_eq!(
            m.free(h),
            Ok(Freed {
                addr: 0,
                size: 0x2000,
                flags: 1
            })
        );
        assert!(m.get(h).is_some());
        assert_eq!(
            m.free(h),
            Ok(Freed {
                addr: 0x4000_0000,
                size: 0x2000,
                flags: 0
            })
        );
        assert!(m.is_empty());
        assert_eq!(m.free(h), Err(NvError::BAD_VALUE));
        assert_eq!(m.from_id(id), Err(NvError::BAD_VALUE));
    }

    #[test]
    fn handle_ids_are_not_reused() {
        let mut m = Nvmap::new();
        let a = m.create(0x1000).unwrap();
        m.free(a).unwrap();
        let b = m.create(0x1000).unwrap();
        assert_ne!(a, b);
    }
}
