//! Global memory: the table of GPU address ranges a program that reads or writes memory by GPU
//! address looks its addresses up in, and the host memory the ranges stand for.
//!
//! The memory behind a GPU mapping is the guest's own; the device reads and writes it where it is
//! (`VK_EXT_external_memory_host`), so what the program does is visible to the guest at once and
//! nothing is copied. A mapping is imported in runs of host-contiguous pages.

use super::frame::{Frame, Garbage};
use crate::compute::{DeviceBuffer, HostBuffer, ImportedMemory};
use crate::device::Vk;
use ash::vk;
use nx_gpu::render::Scene;
use std::collections::{BTreeMap, HashMap};

/// The import alignment every device reports at most.
const PAGE: u64 = 4096;
/// The longest run imported at once.
const MAX_IMPORT: u64 = 1 << 30;

/// How long an import nothing has used stays (in frames).
const KEPT_FRAMES: u64 = 600;
/// What is made reachable of a storage buffer whose size word is not one (A-258).
const UNSIZED_WINDOW: u64 = 64 << 20;
/// How far past the start of a storage buffer an import of its memory reaches.
const IMPORT_WINDOW: u64 = 16 << 20;
/// The largest size word taken as a storage buffer's size.
const LARGEST_BUFFER: u64 = 1 << 30;

#[derive(Default)]
pub struct GlobalMemory {
    /// Imported host memory by (aligned host address, size), with the frame it was last used in.
    imports: BTreeMap<(usize, u64), (ImportedMemory, u64)>,
    /// The frame the imports were last looked over in, for the ones nothing uses any more.
    swept: u64,
    scratch: Option<HostBuffer>,
    /// The table built for the mappings with this signature.
    table: Option<(DeviceBuffer, u64)>,
    /// Say why an import failed (`NX_TRACE_GPU`).
    trace: bool,
    /// Imports that failed, by reason, so each is said once.
    failures: HashMap<String, u64>,
}

fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x0000_0100_0000_01b3)
}

impl GlobalMemory {
    pub fn new(trace: bool) -> GlobalMemory {
        GlobalMemory {
            trace,
            ..GlobalMemory::default()
        }
    }

    /// The table for the address space of `scene` (its buffer and size in bytes), built again
    /// when the mappings changed since it was made. Layout: word 0 the number of entries, words 2
    /// and 3 the device address of a zeroed scratch page, then six words per entry (see
    /// `nx_shader::translate::Info::global_memory`).
    pub fn table(
        &mut self,
        vk: &Vk,
        frame: &mut Frame,
        scene: &Scene<'_>,
        now: u64,
    ) -> Result<(vk::Buffer, u64), String> {
        self.sweep(frame, now);
        let maps: Vec<(u64, u64, u64)> = scene
            .space
            .mappings()
            .map(|m| (m.start, m.len, m.cpu))
            .collect();
        let signature = maps.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, (s, l, c)| {
            mix(mix(mix(h, *s), *l), *c)
        });
        if let Some((buffer, sig)) = &self.table {
            if *sig == signature {
                return Ok((buffer.buffer, buffer.size));
            }
        }
        if self.scratch.is_none() {
            self.scratch = Some(vk.create_host_buffer(
                PAGE,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )?);
        }
        let scratch = vk.buffer_address(self.scratch.as_ref().expect("made above"));
        // Every host-contiguous run of every mapping.
        let mut entries: Vec<(u64, u64, u64)> = Vec::new();
        for (start, len, cpu) in maps {
            let mut off = 0u64;
            while off < len {
                let want = (len - off).min(MAX_IMPORT);
                let Some((ptr, run)) = scene.mem.host_run(cpu + off, want) else {
                    // An unmapped page: nothing is there to reach.
                    off += PAGE - ((cpu + off) % PAGE);
                    continue;
                };
                match self.import(vk, ptr, run, now) {
                    Ok(device) => entries.push((start + off, start + off + run, device)),
                    Err(why) => *self.failures.entry(why).or_default() += 1,
                }
                off += run;
            }
        }
        for (why, n) in &self.failures {
            if *n == 1 && self.trace {
                eprintln!("[vulkan] global memory: {why}");
            }
        }
        let words = table_words(scratch, &entries);
        let bytes = words.len() as u64;
        // The table is read at every access of every program that reaches memory, so it lives
        // in memory the device reads fast (not host memory across the bus): it is copied there
        // before the commands recorded from now on.
        let table = vk.create_device_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let (staging, offset) = frame.upload(vk, &words, 4)?;
        let cmd = frame.upload_cmd(vk)?;
        let copy = [vk::BufferCopy::default()
            .src_offset(offset)
            .dst_offset(0)
            .size(table.size.min(bytes.max(4)))];
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
        // SAFETY: the upload command buffer is recording; the buffers are valid.
        unsafe {
            vk.device.cmd_copy_buffer(cmd, staging, table.buffer, &copy);
            vk.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
        if let Some((old, _)) = self.table.take() {
            frame.garbage.push(Garbage::Device(old));
        }
        let (buffer, size) = (table.buffer, table.size);
        self.table = Some((table, signature));
        Ok((buffer, size))
    }

    /// The device address of the host bytes `[ptr, ptr + len)`, importing the pages they are in
    /// unless an import covers them already.
    fn import(&mut self, vk: &Vk, ptr: *mut u8, len: u64, now: u64) -> Result<u64, String> {
        if let Some(device) = self.covering(ptr, len, now) {
            return Ok(device);
        }
        let page_offset = ptr as u64 % PAGE;
        let base = ptr as u64 - page_offset;
        let size = (len + page_offset).next_multiple_of(PAGE);
        let m = vk.import_host_memory(base as *mut u8, size)?;
        let address = m.address;
        self.imports.insert((base as usize, size), (m, now));
        Ok(address + page_offset)
    }

    /// The device address of the host bytes `[ptr, ptr + len)` in an import that covers them.
    fn covering(&mut self, ptr: *mut u8, len: u64, now: u64) -> Option<u64> {
        let start = ptr as usize;
        let end = start + len as usize;
        // The imports that start at or below the bytes, nearest first: one of the first few that
        // reaches past them serves.
        for ((base, size), (m, used)) in self.imports.range_mut(..=(start, u64::MAX)).rev().take(8)
        {
            if *base + *size as usize >= end {
                *used = now;
                return Some(m.address + (start - *base) as u64);
            }
        }
        None
    }

    /// Let go of the imports nothing has used for `KEPT_FRAMES` frames, once no batch can use
    /// them.
    fn sweep(&mut self, frame: &mut Frame, now: u64) {
        if now < self.swept + KEPT_FRAMES / 4 {
            return;
        }
        self.swept = now;
        let stale: Vec<(usize, u64)> = self
            .imports
            .iter()
            .filter(|(_, (_, used))| used + KEPT_FRAMES < now)
            .map(|(k, _)| *k)
            .collect();
        for k in stale {
            if let Some((m, _)) = self.imports.remove(&k) {
                frame.garbage.push(Garbage::Imported(m));
            }
        }
    }

    /// The table of a program whose accesses are known to be in `buffers` (GPU address and
    /// length each, A-258): only they are reachable (each up to the end of its mapping), the rest
    /// of the address space the scratch page. In upload memory: the buffer, offset and size of
    /// its words.
    pub fn table_for(
        &mut self,
        vk: &Vk,
        frame: &mut Frame,
        scene: &Scene<'_>,
        buffers: &[(u64, u64)],
        now: u64,
    ) -> Result<(vk::Buffer, u64, u64), String> {
        self.sweep(frame, now);
        if self.scratch.is_none() {
            self.scratch = Some(vk.create_host_buffer(
                PAGE,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )?);
        }
        let scratch = vk.buffer_address(self.scratch.as_ref().expect("made above"));
        // The ranges in order, each inside one mapping, overlapping ones joined.
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        for &(gpu, len) in buffers {
            let Some((_, left)) = scene.space.translate(gpu) else {
                continue;
            };
            ranges.push((gpu, gpu + len.min(left)));
        }
        ranges.sort_unstable();
        let mut joined: Vec<(u64, u64)> = Vec::new();
        for (s, e) in ranges {
            match joined.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => joined.push((s, e)),
            }
        }
        let mut entries: Vec<(u64, u64, u64)> = Vec::new();
        for (start, end) in joined {
            let mut at = start;
            while at < end {
                let Some((cpu, left)) = scene.space.translate(at) else {
                    break;
                };
                let need = (end - at).min(left).min(MAX_IMPORT);
                let Some((ptr, run)) = scene.mem.host_run(cpu, need) else {
                    at += PAGE - (cpu % PAGE);
                    continue;
                };
                // An import that covers the bytes serves; else what is imported reaches
                // `IMPORT_WINDOW` past them where the mapping goes on: the buffers of a ring move
                // along the same memory from one launch to the next, and find it imported (each
                // import costs milliseconds).
                let device = match self.covering(ptr, run, now) {
                    Some(device) => Ok(device),
                    None => {
                        let want = need.max(IMPORT_WINDOW).min(left).min(MAX_IMPORT);
                        let (window, size) = scene.mem.host_run(cpu, want).unwrap_or((ptr, run));
                        self.import(vk, window, size.max(run), now)
                    }
                };
                match device {
                    Ok(device) => entries.push((at, at + run, device)),
                    Err(why) => *self.failures.entry(why).or_default() += 1,
                }
                at += run;
            }
        }
        let words = table_words(scratch, &entries);
        let align = vk.caps.min_storage_buffer_offset_alignment.max(16);
        let (buffer, offset) = frame.upload(vk, &words, align)?;
        Ok((buffer, offset, words.len() as u64))
    }

    /// How many bytes of host memory are imported.
    pub fn bytes(&self) -> u64 {
        self.imports.values().map(|(m, _)| m.size).sum()
    }

    /// How many runs of host memory are imported.
    pub fn runs(&self) -> usize {
        self.imports.len()
    }

    /// The bytes of host memory the imports cover (each byte once, however many imports
    /// cover it).
    pub fn union_bytes(&self) -> u64 {
        let mut ranges: Vec<(usize, u64)> = self.imports.keys().copied().collect();
        ranges.sort_unstable();
        let (mut total, mut end) = (0u64, 0u64);
        for (base, size) in ranges {
            let (start, stop) = (base as u64, base as u64 + size);
            if stop > end {
                total += stop - start.max(end);
                end = stop;
            }
        }
        total
    }

    pub fn destroy(&mut self, vk: &Vk) {
        if let Some((t, _)) = self.table.take() {
            vk.destroy_device_buffer(t);
        }
        if let Some(s) = self.scratch.take() {
            vk.destroy_host_buffer(s);
        }
        while let Some((_, (m, _))) = self.imports.pop_first() {
            vk.destroy_imported(m);
        }
    }
}

/// The words of a table: the number of entries, the scratch page's device address, then each
/// entry's first GPU address, one past its last and its device address (two words each).
fn table_words(scratch: u64, entries: &[(u64, u64, u64)]) -> Vec<u8> {
    let bytes = (4 + 6 * entries.len()) * 4;
    let mut words = vec![0u8; bytes];
    let mut put = |i: usize, v: u32| words[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    put(0, entries.len() as u32);
    put(2, scratch as u32);
    put(3, (scratch >> 32) as u32);
    for (i, (s, e, d)) in entries.iter().enumerate() {
        let at = 4 + 6 * i;
        for (k, v) in [*s, *e, *d].into_iter().enumerate() {
            put(at + 2 * k, v as u32);
            put(at + 2 * k + 1, (v >> 32) as u32);
        }
    }
    words
}

/// The storage buffers a program reaches (A-258): for each constant buffer word its accesses'
/// bases are pointers from, the GPU address there and the size in the word after the next
/// (`read` gives the 12 bytes at a slot and byte offset); with an access whose base is not known,
/// every entry of the launch's table in constant buffer 0 too (16 bytes each from 0x20). `None`
/// for a program whose accesses are not attributed (Maxwell code), which reaches every mapping.
pub fn storage_buffers(
    info: &nx_shader::translate::Info,
    read: impl Fn(u32, u32) -> Option<[u8; 12]>,
) -> Option<Vec<(u64, u64)>> {
    if info.global_bases.is_empty() && !info.global_unknown {
        return None;
    }
    let mut words: Vec<(u32, u32)> = info.global_bases.clone();
    if info.global_unknown {
        words.extend((0x20..0x200u32).step_by(16).map(|b| (0, b / 4)));
    }
    let mut out = Vec::new();
    for (slot, word) in words {
        let Some(b) = read(slot, word * 4) else {
            continue;
        };
        let address = u64::from_le_bytes(b[..8].try_into().expect("eight bytes"));
        let size = u64::from(u32::from_le_bytes(b[8..].try_into().expect("four bytes")));
        if address == 0 {
            continue;
        }
        let len = if size > 0 && size <= LARGEST_BUFFER {
            size
        } else {
            UNSIZED_WINDOW
        };
        out.push((address, len));
    }
    Some(out)
}
