//! Global memory: the table of GPU address ranges a program that reads or writes memory by GPU
//! address looks its addresses up in, and the host memory the ranges stand for.
//!
//! The memory behind a GPU mapping is the guest's own; the device reads and writes it where it is
//! (`VK_EXT_external_memory_host`), so what the program does is visible to the guest at once and
//! nothing is copied. A mapping is imported in runs of host-contiguous pages.

use super::frame::{Frame, Garbage};
use crate::compute::{HostBuffer, ImportedMemory};
use crate::device::Vk;
use ash::vk;
use nx_gpu::render::Scene;
use std::collections::HashMap;

/// The import alignment every device reports at most.
const PAGE: u64 = 4096;
/// The longest run imported at once.
const MAX_IMPORT: u64 = 1 << 30;

#[derive(Default)]
pub struct GlobalMemory {
    /// Imported host memory by (aligned host address, size).
    imports: HashMap<(usize, u64), ImportedMemory>,
    scratch: Option<HostBuffer>,
    /// The table built for the mappings with this signature.
    table: Option<(HostBuffer, u64)>,
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
    ) -> Result<(vk::Buffer, u64), String> {
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
                match self.import(vk, ptr, run) {
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
        let bytes = (4 + 6 * entries.len() as u64) * 4;
        let mut table = vk.create_host_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        {
            let words = table.as_mut_slice();
            let mut put =
                |i: usize, v: u32| words[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
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
        }
        if let Some((old, _)) = self.table.take() {
            frame.garbage.push(Garbage::Buffer(old));
        }
        let (buffer, size) = (table.buffer, table.size);
        self.table = Some((table, signature));
        Ok((buffer, size))
    }

    /// The device address of the host bytes `[ptr, ptr + len)`, importing the pages they are in.
    fn import(&mut self, vk: &Vk, ptr: *mut u8, len: u64) -> Result<u64, String> {
        let page_offset = ptr as u64 % PAGE;
        let base = ptr as u64 - page_offset;
        let size = (len + page_offset).next_multiple_of(PAGE);
        if let Some(m) = self.imports.get(&(base as usize, size)) {
            return Ok(m.address + page_offset);
        }
        let m = vk.import_host_memory(base as *mut u8, size)?;
        let address = m.address;
        self.imports.insert((base as usize, size), m);
        Ok(address + page_offset)
    }

    /// How many bytes of host memory are imported.
    pub fn bytes(&self) -> u64 {
        self.imports.values().map(|m| m.size).sum()
    }

    pub fn destroy(&mut self, vk: &Vk) {
        if let Some((t, _)) = self.table.take() {
            vk.destroy_host_buffer(t);
        }
        if let Some(s) = self.scratch.take() {
            vk.destroy_host_buffer(s);
        }
        for (_, m) in self.imports.drain() {
            vk.destroy_imported(m);
        }
    }
}
