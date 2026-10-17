//! The console's media engines: the video decoder (NVDEC) and the video image compositor (VIC),
//! which programs drive through host1x channels (docs/ASSUMPTIONS.md A-119, A-182, A-183), and
//! the VP9 decoder behind the first.
//!
//! A program maps its buffers into the engines' address space (`MAP_CMD_BUFFER`), writes the
//! addresses (shifted right by 8) into engine methods with command buffers, and submits them.
//! Here the addresses are given out by [`Iova`]. A submission's job is prepared in the call that
//! submits it ([`Media::prepare`]: what it reads is copied out of the program's memory) and run
//! by [`Engines`], on a thread of their own when there is one (A-184) or else in the call. A
//! decoder job decodes a VP9 frame into the store of [`vp9::nvdec::Engine`], under the guest
//! address of its output surface (a program maps and unmaps its buffers around each job, so
//! their engine addresses come and go; the memory stays until the handle is freed, and the
//! frames with it), and a compositor job converts a frame of that store into the program's
//! output surface in its memory.

pub mod host1x;
pub mod vic;
pub mod vp9;

use crate::mem::GuestMemory;
use crate::nvmap::Nvmap;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

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
    pub fn unmap(&mut self, handle: u32) -> Option<Range<u64>> {
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

/// A job of the engines, with everything it reads copied out of the program's memory when it
/// was submitted: it can run later, on another thread, while the program goes on (A-184).
#[derive(Clone, Debug)]
pub enum Job {
    /// Decode a VP9 frame (A-182) from the frames in the surfaces at `refs` (guest addresses of
    /// the last, golden and alternate references) into the surface at `out`.
    Decode {
        pic: Vec<u8>,
        table: Vec<u8>,
        tiles: Vec<u8>,
        refs: [u64; 3],
        out: u64,
    },
    /// Convert the frame decoded into the surface at `input` into the surface at `out` (guest
    /// addresses; A-183).
    Convert {
        config: vic::Config,
        input: u64,
        out: u64,
    },
    /// Memory was freed: the frames decoded into it go with it.
    Forget(Range<u64>),
}

/// The jobs of one submission, and the syncpoint increments (id, by) that follow them.
#[derive(Debug, Default)]
pub struct Batch {
    pub jobs: Vec<Job>,
    pub increments: Vec<(u32, u32)>,
}

/// Somewhere to put batches for another thread to run, one after the other in the order they
/// were put (A-184). `push` never waits.
pub trait MediaQueue: Send + Sync {
    fn push(&self, batch: Batch);
}

/// What running jobs changes: the decoder's frames. Owned by whoever runs the jobs.
#[derive(Default)]
pub struct Engines {
    vp9: vp9::nvdec::Engine,
    /// `NX_TRACE_MEDIA`: every decode and conversion.
    trace: bool,
    /// When the last decode began, for the trace.
    last_decode: Option<Instant>,
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

impl Engines {
    pub fn new() -> Engines {
        Engines {
            vp9: vp9::nvdec::Engine::new(),
            trace: std::env::var_os("NX_TRACE_MEDIA").is_some(),
            last_decode: None,
        }
    }

    /// Run one job. A frame that cannot be decoded, or converted from a surface that holds no
    /// frame, is said and left undone; a job that cannot be run at all stops the run.
    pub fn run(&mut self, job: &Job, mem: &dyn GuestMemory) -> Result<(), String> {
        match job {
            Job::Decode {
                pic,
                table,
                tiles,
                refs,
                out,
            } => {
                let began = Instant::now();
                let since_last = self.last_decode.replace(began).map_or(0.0, ms);
                match self.vp9.decode(pic, table, tiles, *refs, *out) {
                    Ok(f) => {
                        if self.trace {
                            eprintln!(
                                "[media] decoded a {}x{} VP9 frame into {out:#x} in {:.1} ms, {since_last:.1} ms after the one before",
                                f.width,
                                f.height,
                                ms(began)
                            );
                        }
                    }
                    Err(e) => eprintln!("[media] a VP9 frame for {out:#x} was not decoded: {e}"),
                }
            }
            Job::Convert { config, input, out } => {
                let Some(frame) = self.vp9.frame(*input) else {
                    if self.trace {
                        eprintln!(
                            "[media] the compositor's input {input:#x} holds no decoded frame"
                        );
                    }
                    return Ok(());
                };
                let began = Instant::now();
                let pixels = config.convert(&frame)?;
                if !mem.write(*out, &pixels) {
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
                        "[media] converted {input:#x} into a {}x{} surface at {out:#x} ({:#x} bytes, format {:#x}, kind {}) in {:.1} ms: {lit} pixels not black",
                        config.width,
                        config.height,
                        pixels.len(),
                        config.format,
                        config.block_kind,
                        ms(began)
                    );
                }
            }
            Job::Forget(range) => self.vp9.forget(range.clone()),
        }
        Ok(())
    }
}

/// What the driver keeps of the media engines: the address space, where jobs go, and the
/// engines themselves for the jobs that run in the call.
#[derive(Default)]
pub struct Media {
    pub iova: Iova,
    /// `NX_TRACE_MEDIA`: every decode and conversion.
    pub trace: bool,
    /// `NX_TRACE_MEDIA=2`: every method write and every mapping too.
    pub trace_writes: bool,
    /// Codecs and classes that were asked for and not run, said once each.
    said: Vec<u32>,
    /// Where batches go for another thread to run them (A-184); without one they run in the call
    /// that submits them.
    pub queue: Option<Arc<dyn MediaQueue>>,
    /// The engines, for the jobs run in the call.
    pub engines: Engines,
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
            engines: Engines::new(),
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

    /// Prepare the job an engine's `EXECUTE` starts, with the engine's methods `regs` (by byte
    /// offset): what it reads is copied out of the program's memory now. `None` for a job the
    /// engines cannot do, which is said once and left undone (A-119); a broken one stops the run.
    pub fn prepare(
        &mut self,
        class: u32,
        regs: &HashMap<u32, u32>,
        nvmap: &Nvmap,
        mem: &dyn GuestMemory,
    ) -> Result<Option<Job>, String> {
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
                    return Ok(None);
                }
                let pic = self.read(nvmap, mem, addr(0x404)?, vp9::nvdec::PICTURE_SIZE)?;
                let size =
                    u32::from_le_bytes([pic[0x30], pic[0x31], pic[0x32], pic[0x33]]) as usize;
                let table = self.read(nvmap, mem, addr(0x5C0)?, vp9::probs::NVDEC_TABLE_SIZE)?;
                let tiles = self.read(nvmap, mem, addr(0x408)?, size)?;
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
                Ok(Some(Job::Decode {
                    pic,
                    table,
                    tiles,
                    refs,
                    out,
                }))
            }
            CLASS_VIC => {
                let config = self.read(nvmap, mem, addr(0x708)?, vic::CONFIG_SIZE)?;
                let config =
                    vic::Config::parse(&config).ok_or("a compositor configuration cut short")?;
                let input = self.guest_address(nvmap, addr(0x400)?, 1)?;
                let out = self.guest_address(nvmap, addr(0x720)?, config.output_size() as u64)?;
                Ok(Some(Job::Convert { config, input, out }))
            }
            other => {
                if !self.said.contains(&(other | 0x1000)) {
                    self.said.push(other | 0x1000);
                    eprintln!("[media] a job of class {other:#x}, which no engine here runs");
                }
                Ok(None)
            }
        }
    }

    /// Memory was freed: the frames decoded into it are forgotten, after the jobs before.
    pub fn forget(&mut self, range: Range<u64>) {
        match &self.queue {
            Some(q) => q.push(Batch {
                jobs: vec![Job::Forget(range)],
                increments: Vec::new(),
            }),
            None => self.engines.vp9.forget(range),
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

    /// A queue that keeps what it is given.
    #[derive(Default)]
    struct Kept(std::sync::Mutex<Vec<Batch>>);

    impl MediaQueue for Kept {
        fn push(&self, batch: Batch) {
            self.0.lock().unwrap().push(batch);
        }
    }

    #[test]
    fn freed_memory_is_forgotten_through_the_queue_when_there_is_one() {
        let mut m = Media::new();
        m.forget(0x1000..0x2000);
        let kept = Arc::new(Kept::default());
        m.queue = Some(kept.clone());
        m.forget(0x3000..0x4000);
        let batches = kept.0.lock().unwrap();
        assert_eq!(batches.len(), 1);
        assert!(batches[0].increments.is_empty());
        assert!(matches!(&batches[0].jobs[..], [Job::Forget(r)] if *r == (0x3000..0x4000)));
    }
}
