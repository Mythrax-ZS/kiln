//! The console's media engines: the video decoder (NVDEC) and the video image compositor (VIC),
//! which programs drive through host1x channels (docs/ASSUMPTIONS.md A-119, A-182, A-183), and
//! the VP9 decoder behind the first.
//!
//! A program maps its buffers into the engines' address space (`MAP_CMD_BUFFER`), writes the
//! addresses (shifted right by 8) into engine methods with command buffers, and submits them.
//! Here the addresses are given out by [`Iova`]; a decoder job decodes a VP9 frame into the store
//! of [`vp9::nvdec::Engine`], under the guest address of its output surface (a program maps and
//! unmaps its buffers around each job, so their engine addresses come and go; the memory stays
//! until the handle is freed, and the frames with it), and a compositor job converts a frame of
//! that store into the program's output surface in its memory.

pub mod host1x;
pub mod vic;
pub mod vp9;

use crate::mem::GuestMemory;
use crate::nvmap::Nvmap;
use std::collections::{BTreeMap, HashMap};

/// The engines' address space: each handle mapped gets a range of its own, kept while it is
/// mapped (a handle mapped again gets the same one).
#[derive(Debug, Default)]
pub struct Iova {
    /// handle -> (address, size, times mapped)
    handles: HashMap<u32, (u64, u64, u32)>,
    /// address -> (handle, size)
    ranges: BTreeMap<u64, (u32, u64)>,
}

/// Where the address space starts and ends: the answer of a mapping is a 32-bit address.
const IOVA_START: u64 = 0x10_0000;
const IOVA_END: u64 = 0xFFF0_0000;

impl Iova {
    /// Map `handle` of `size` bytes; returns its address.
    pub fn map(&mut self, handle: u32, size: u64) -> Option<u64> {
        if let Some(e) = self.handles.get_mut(&handle) {
            e.2 += 1;
            return Some(e.0);
        }
        let size = size.max(1).next_multiple_of(0x1000);
        let mut at = IOVA_START;
        for (&start, &(_, len)) in &self.ranges {
            if start >= at + size {
                break;
            }
            at = at.max((start + len).next_multiple_of(0x1000));
        }
        if at + size > IOVA_END {
            return None;
        }
        self.handles.insert(handle, (at, size, 1));
        self.ranges.insert(at, (handle, size));
        Some(at)
    }

    /// Unmap `handle` once; returns its range if that was the last mapping.
    pub fn unmap(&mut self, handle: u32) -> Option<std::ops::Range<u64>> {
        let e = self.handles.get_mut(&handle)?;
        e.2 -= 1;
        if e.2 > 0 {
            return None;
        }
        let (at, size, _) = self.handles.remove(&handle)?;
        self.ranges.remove(&at);
        Some(at..at + size)
    }

    /// The address, size and mapping count of a mapped handle.
    pub fn handle_range(&self, handle: u32) -> Option<(u64, u64, u32)> {
        self.handles.get(&handle).copied()
    }

    /// The handle and the offset in it of address `iova`.
    pub fn translate(&self, iova: u64) -> Option<(u32, u64)> {
        let (&start, &(handle, size)) = self.ranges.range(..=iova).next_back()?;
        (iova < start + size).then_some((handle, iova - start))
    }
}

/// What the media engines keep: the address space and the decoder's frames.
#[derive(Default)]
pub struct Media {
    pub iova: Iova,
    pub vp9: vp9::nvdec::Engine,
    /// `NX_TRACE_MEDIA`: every decode and conversion.
    pub trace: bool,
    /// `NX_TRACE_MEDIA=2`: every method write and every mapping too.
    pub trace_writes: bool,
    /// Codecs that were asked for and not decoded, said once each.
    said: Vec<u32>,
}

/// An engine's method that starts a job.
pub const EXECUTE: u32 = 0x300;
pub const CLASS_NVDEC: u32 = 0xF0;
pub const CLASS_VIC: u32 = 0x5D;
const CODEC_VP9: u32 = 9;

impl Media {
    pub fn new() -> Media {
        let level = std::env::var_os("NX_TRACE_MEDIA");
        Media {
            trace: level.is_some(),
            trace_writes: level.is_some_and(|v| v == "2"),
            ..Media::default()
        }
    }

    /// Read `len` bytes at engine address `iova` (a method's value, shifted left by 8).
    fn read(
        &self,
        nvmap: &Nvmap,
        mem: &dyn GuestMemory,
        iova: u64,
        len: usize,
    ) -> Result<Vec<u8>, String> {
        let at = self.guest_address(nvmap, iova, len as u64)?;
        let mut out = vec![0u8; len];
        if !mem.read(at, &mut out) {
            return Err(format!(
                "the media engine address {iova:#x} is not readable"
            ));
        }
        Ok(out)
    }

    /// The guest address of `len` bytes at engine address `iova`, all in one mapped handle.
    fn guest_address(&self, nvmap: &Nvmap, iova: u64, len: u64) -> Result<u64, String> {
        let (handle, offset) = self
            .iova
            .translate(iova)
            .ok_or_else(|| format!("the media engine address {iova:#x} is not mapped"))?;
        let h = nvmap
            .get(handle)
            .ok_or_else(|| format!("the media engine address {iova:#x} names a freed handle"))?;
        let base = h.addr.ok_or_else(|| {
            format!("the media engine address {iova:#x} names a handle with no memory")
        })?;
        if offset + len > u64::from(h.size) {
            return Err(format!(
                "{len:#x} bytes at media engine address {iova:#x} run past handle {handle}"
            ));
        }
        Ok(base + offset)
    }

    /// Run the job an engine's `EXECUTE` starts, with the engine's methods `regs` (by byte
    /// offset). A job the engines cannot do is said and left undone (A-119); a broken one stops
    /// the run.
    pub fn execute(
        &mut self,
        class: u32,
        regs: &HashMap<u32, u32>,
        nvmap: &Nvmap,
        mem: &dyn GuestMemory,
    ) -> Result<(), String> {
        let addr = |m: u32| -> Result<u64, String> {
            regs.get(&m)
                .map(|&v| u64::from(v) << 8)
                .ok_or_else(|| format!("a job of class {class:#x} without method {m:#x}"))
        };
        match class {
            CLASS_NVDEC => {
                let codec = regs.get(&0x200).copied().unwrap_or(0);
                if codec != CODEC_VP9 {
                    if !self.said.contains(&codec) {
                        self.said.push(codec);
                        eprintln!("[media] the video decoder was asked for codec {codec}, which is not decoded: its frames stay empty");
                    }
                    return Ok(());
                }
                let pic = self.read(nvmap, mem, addr(0x404)?, vp9::nvdec::PICTURE_SIZE)?;
                let size =
                    u32::from_le_bytes([pic[0x30], pic[0x31], pic[0x32], pic[0x33]]) as usize;
                let table = self.read(nvmap, mem, addr(0x5C0)?, vp9::probs::NVDEC_TABLE_SIZE)?;
                let bits = self.read(nvmap, mem, addr(0x408)?, size)?;
                // The surfaces by their guest addresses (a reference may be a surface that was
                // never decoded into, for a key frame: its address is not needed then).
                let surface =
                    |m: u32| -> Result<u64, String> { self.guest_address(nvmap, addr(m)?, 1) };
                let refs = [
                    surface(0x430).unwrap_or(0),
                    surface(0x434).unwrap_or(0),
                    surface(0x438).unwrap_or(0),
                ];
                let out = surface(0x43C)?;
                match self.vp9.decode(&pic, &table, &bits, refs, out) {
                    Ok(f) => {
                        if self.trace {
                            eprintln!(
                                "[media] decoded a {}x{} VP9 frame into {out:#x}",
                                f.width, f.height
                            );
                        }
                    }
                    Err(e) => eprintln!("[media] a VP9 frame for {out:#x} was not decoded: {e}"),
                }
                Ok(())
            }
            CLASS_VIC => {
                let config = self.read(nvmap, mem, addr(0x708)?, vic::CONFIG_SIZE)?;
                let config =
                    vic::Config::parse(&config).ok_or("a compositor configuration cut short")?;
                let input = self.guest_address(nvmap, addr(0x400)?, 1)?;
                let Some(frame) = self.vp9.frame(input) else {
                    if self.trace {
                        eprintln!(
                            "[media] the compositor's input {input:#x} holds no decoded frame"
                        );
                    }
                    return Ok(());
                };
                let pixels = config.convert(&frame)?;
                let out = addr(0x720)?;
                let at = self.guest_address(nvmap, out, pixels.len() as u64)?;
                if !mem.write(at, &pixels) {
                    return Err(format!(
                        "the compositor's output at {out:#x} is not writable"
                    ));
                }
                if self.trace {
                    let lit = pixels
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .filter(|p| p[0] | p[1] | p[2] != 0)
                        .count();
                    eprintln!(
                        "[media] converted {input:#x} into a {}x{} surface at {out:#x}, guest {at:#x} ({:#x} bytes, format {:#x}, kind {}): {lit} pixels not black",
                        config.width, config.height, pixels.len(), config.format, config.block_kind
                    );
                }
                Ok(())
            }
            other => {
                if !self.said.contains(&(other | 0x1000)) {
                    self.said.push(other | 0x1000);
                    eprintln!("[media] a job of class {other:#x}, which no engine here runs");
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_get_ranges_of_their_own_and_keep_them_while_mapped() {
        let mut v = Iova::default();
        let a = v.map(5, 0x3000).unwrap();
        let b = v.map(6, 0x1000).unwrap();
        assert_ne!(a, b);
        assert!(b >= a + 0x3000, "no overlap");
        assert_eq!(v.map(5, 0x3000), Some(a), "mapped again: the same range");
        assert_eq!(v.translate(a + 0x2FFF), Some((5, 0x2FFF)));
        assert_eq!(v.translate(b + 0x10), Some((6, 0x10)));
        assert_eq!(v.translate(a - 1), None);
        assert_eq!(v.unmap(5), None, "still mapped once");
        assert_eq!(v.unmap(5), Some(a..a + 0x3000));
        assert_eq!(v.translate(a), None);
        // The range is free again for the next.
        assert_eq!(v.map(7, 0x2000), Some(a));
        // A gigabyte fits, and runs out at the top of the 32-bit space.
        assert!(v.map(8, 0x4000_0000).is_some());
        assert!(v.map(9, 0xF000_0000).is_none());
    }
}
