//! Compute launches: the program translated for its workgroup, the constant buffers the descriptor
//! binds, and a dispatch recorded after the commands before it.
//!
//! What later commands read of memory (constant buffers, vertices, textures) is copied when they
//! are recorded, so what a launch writes has to be in memory by then (docs/ASSUMPTIONS.md A-110).
//! A run of launches is not waited for launch by launch: it is waited for, and the surfaces it
//! wrote put into memory, when something needs that: a draw, a clear or a copy, a launch that
//! reads what one before it writes, or the GPU front reading or writing memory itself (A-274).

use super::shaders::{bindings_of, Binding, Program};
use super::Core;
use crate::fast::FastMap;
use ash::vk;
use nx_gpu::compute::Dispatch;
use nx_gpu::render::Scene;
use nx_gpu::vm::AddressSpace;
use nx_gpu::GuestMemory;
use std::time::Instant;

/// The constant buffer slot that holds the handles of the textures a compute program names (the
/// title puts them there for a launch as it does for a draw).
const TEXTURE_TABLE: u32 = 2;

/// Why launches were waited for (A-274), for the report.
pub(super) mod settle {
    /// The GPU front reads or writes memory itself, or a submission ends.
    pub const FRONT: usize = 0;
    /// A draw, a clear or a copy comes.
    pub const COMMAND: usize = 1;
    /// A launch reads on the host, or reaches in place, what one before it writes.
    pub const HAZARD: usize = 2;
    /// A launch waited for at once (`NX_DEBUG_COMPUTE_WAIT`, `NX_TRACE_COMPUTE`,
    /// `NX_DEBUG_SYNC_DRAWS`).
    pub const SYNC: usize = 3;
    /// The GPU front reads or writes bytes that a launch reaches, a surface or a release covers.
    pub const FRONT_RANGE: usize = 4;
}

/// What is made in memory once the launches before it have run (A-274): a QMD's release, or a
/// semaphore release of the GPU front (its bytes).
pub enum Completion {
    Qmd(nx_gpu::qmd3::Release),
    Bytes(u64, Vec<u8>),
}

impl Completion {
    pub fn apply(&self, space: &AddressSpace, mem: &dyn GuestMemory) -> Result<(), String> {
        match self {
            Completion::Qmd(r) => r.apply(&mut nx_gpu::qmd3::GpuMemory { mem, space }),
            Completion::Bytes(va, data) => {
                if space.write(mem, *va, data) {
                    Ok(())
                } else {
                    Err(format!(
                        "a release of {} bytes to GPU address {va:#x}, which is not mapped",
                        data.len()
                    ))
                }
            }
        }
    }

    /// The word at `va` after this release, from what it was (`word`).
    fn applied_to(&self, va: u64, word: u32) -> u32 {
        match self {
            Completion::Qmd(r) if r.va == va => match r.op {
                nx_gpu::qmd3::ReleaseOp::Set => r.payload,
                nx_gpu::qmd3::ReleaseOp::Add => word.wrapping_add(r.payload),
                nx_gpu::qmd3::ReleaseOp::Max => word.max(r.payload),
            },
            Completion::Qmd(_) => word,
            Completion::Bytes(at, data) => {
                let mut bytes = word.to_le_bytes();
                for (k, b) in bytes.iter_mut().enumerate() {
                    let address = va + k as u64;
                    if address >= *at && address < at + data.len() as u64 {
                        *b = data[(address - at) as usize];
                    }
                }
                u32::from_le_bytes(bytes)
            }
        }
    }

    /// Whether it writes into `[from, to)`.
    fn overlaps(&self, from: u64, to: u64) -> bool {
        let (at, len) = match self {
            Completion::Qmd(r) => (r.va, 4),
            Completion::Bytes(va, data) => (*va, data.len() as u64),
        };
        at < to && from < at + len
    }
}

/// Whether `[from, to)` overlaps one of `ranges`.
fn overlaps(ranges: &[(u64, u64)], from: u64, to: u64) -> bool {
    ranges.iter().any(|&(a, b)| a < to && from < b)
}

/// Compute pipelines by the program and the layout they were made for.
#[derive(Default)]
pub struct ComputePipelines {
    map: FastMap<(u64, Vec<Binding>), vk::Pipeline>,
}

impl ComputePipelines {
    pub fn destroy(&mut self, vk: &crate::device::Vk) {
        for (_, p) in self.map.drain() {
            // SAFETY: nothing uses the pipeline any more.
            unsafe { vk.device.destroy_pipeline(p, None) };
        }
    }
}

impl Core {
    pub(super) fn dispatch(&mut self, scene: &Scene<'_>, d: &Dispatch) -> Result<(), String> {
        if self.commands_at_launch != self.commands_seen {
            self.phases.launch_kinds[3] += 1;
            self.commands_at_launch = self.commands_seen;
        }
        let mut mark = super::stopwatch();
        let mut lap = |into: &mut std::time::Duration| {
            if let Some(mark) = &mut mark {
                let now = Instant::now();
                *into += now - *mark;
                *mark = now;
            }
        };
        let program = self
            .shaders
            .kernel(&self.vk, scene, d, self.textures.frame, (0, 0))?;
        // A program whose surfaces hold integers reads and writes their bits (A-272): it is made
        // again knowing which they are.
        let classes = self.integer_surfaces(scene, d, &program)?;
        let program = if classes == (0, 0) {
            program
        } else {
            self.shaders
                .kernel(&self.vk, scene, d, self.textures.frame, classes)?
        };
        // What the launch reads may be what a launch before it writes: those are waited for
        // first, before anything of the batch is taken for this one (A-274).
        self.settle_hazards(scene, d, &program)?;
        let bindings = bindings_of(&[&program]);
        let (set_layout, pipeline_layout) = self.layouts.get(&self.vk, &bindings)?;
        let set = self.frame.descriptor_set(&self.vk, set_layout)?;
        let base = 64 * program.stage.index();
        lap(&mut self.phases.compute[0]);

        // The constant buffers the program reads, from the descriptor.
        let mut buffer_infos: Vec<(u32, vk::DescriptorBufferInfo)> = Vec::new();
        for cb in 0..18u32 {
            if program.info.cbuf_extent[cb as usize] == 0 && !program.info.cbuf_dynamic[cb as usize]
            {
                continue;
            }
            let want = if program.info.cbuf_dynamic[cb as usize] {
                u32::MAX
            } else {
                program.info.cbuf_extent[cb as usize]
            };
            let bound = d
                .cbufs
                .get(cb as usize)
                .copied()
                .flatten()
                .map(|c| (c.address, c.size));
            let (buffer, offset, range) =
                self.upload_constant_buffer(scene, super::draw::Owner::Launch, cb, bound, want)?;
            buffer_infos.push((
                base + cb,
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(offset)
                    .range(range),
            ));
        }
        if self.debug.trace_cbufs {
            // What the launch's own parameters are: the words of constant buffer 0 that are not
            // zero, by byte offset.
            if let Some(Some(c)) = d.cbufs.first() {
                if let Some(bytes) = scene.read_vec(c.address, (c.size as usize).min(4096)) {
                    let words: Vec<String> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(_, w)| **w != [0u8; 4])
                        .map(|(i, w)| format!("{:#x}={:#x}", 4 * i, u32::from_le_bytes(*w)))
                        .collect();
                    eprintln!(
                        "[vulkan] compute launch {:#x}: constant buffer 0 at {:#x} ({} bytes): {}",
                        d.qmd,
                        c.address,
                        c.size,
                        words.join(" ")
                    );
                }
            }
        }
        let mut reached = None;
        if program.info.global_memory {
            // The storage buffers the program's accesses are in, read from the launch's constant
            // buffers (A-258); all of memory for a program whose accesses are not attributed.
            let now = self.textures.frame;
            let read = |slot: u32, at: u32| -> Option<[u8; 12]> {
                let c = d.cbufs.get(slot as usize).copied().flatten()?;
                if u64::from(at) + 12 > u64::from(c.size) {
                    return None;
                }
                scene
                    .read_vec(c.address + u64::from(at), 12)?
                    .try_into()
                    .ok()
            };
            reached = super::global::storage_buffers(&program.info, read);
            let (buffer, offset, size) = match &reached {
                Some(buffers) => {
                    self.global
                        .table_for(&self.vk, &mut self.frame, scene, buffers, now)?
                }
                None => {
                    let (b, s) = self.global.table(&self.vk, &mut self.frame, scene, now)?;
                    (b, 0, s)
                }
            };
            buffer_infos.push((
                base + 20,
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(offset)
                    .range(size),
            ));
        }

        lap(&mut self.phases.compute[1]);
        // The textures: their handles are words of constant buffers of the descriptor.
        let pools = scene.regs.texture_pools();
        let mut handles = Vec::new();
        for used in &program.info.textures {
            let slot = used.source.unwrap_or(TEXTURE_TABLE);
            let Some(Some(buffer)) = d.cbufs.get(slot as usize) else {
                return Err(format!(
                    "texture {} is named by a handle in constant buffer {slot}, which the launch does not bind",
                    used.index
                ));
            };
            let at = 4 * u64::from(used.index);
            if at + 4 > u64::from(buffer.size) {
                return Err(format!(
                    "texture {} is named at byte {at} of a constant buffer of {} bytes",
                    used.index, buffer.size
                ));
            }
            let mut word = [0u8; 4];
            self.read_exact(scene, buffer.address + at, &mut word)?;
            handles.push((
                u32::from_le_bytes(word),
                super::draw::HandleFrom::Launch {
                    slot,
                    address: buffer.address,
                },
            ));
        }
        let mut images = Vec::new();
        self.bind_textures(
            scene,
            &program,
            d.program,
            "compute",
            &pools,
            &handles,
            &mut images,
        )?;
        // The surfaces: their handles are words of constant buffers too, and their headers are in
        // the texture pool (A-252).
        let mut surfaces = Vec::new();
        for used in &program.info.images {
            let slot = used.source.unwrap_or(TEXTURE_TABLE);
            let Some(Some(buffer)) = d.cbufs.get(slot as usize) else {
                return Err(format!(
                    "surface {} is named by a handle in constant buffer {slot}, which the launch does not bind",
                    used.index
                ));
            };
            let mut word = [0u8; 4];
            self.read_exact(scene, buffer.address + 4 * u64::from(used.index), &mut word)?;
            let tic_id = u32::from_le_bytes(word) & 0xF_FFFF;
            if tic_id >= pools.header_count {
                return Err(format!(
                    "surface {}: header {tic_id} of {}",
                    used.index, pools.header_count
                ));
            }
            let mut tic_bytes = [0u8; 32];
            self.read_exact(
                scene,
                pools.header_address + 32 * u64::from(tic_id),
                &mut tic_bytes,
            )?;
            let tic = super::textures::Tic(std::array::from_fn(|i| {
                u32::from_le_bytes(tic_bytes[4 * i..4 * i + 4].try_into().unwrap())
            }));
            let surface = self.surface(scene, tic, used)?;
            surfaces.push((
                nx_shader::translate::ImageUse::binding(program.stage, used.slot),
                surface,
            ));
        }
        lap(&mut self.phases.compute[2]);

        let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
            buffer_infos.iter().map(|(_, i)| [*i]).collect();
        let image_arrays: Vec<[vk::DescriptorImageInfo; 1]> =
            images.iter().map(|(_, i)| [*i]).collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .zip(&infos)
            .map(|((binding, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        writes.extend(
            images
                .iter()
                .zip(&image_arrays)
                .map(|((binding, _), info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(info)
                }),
        );
        let surface_arrays: Vec<[vk::DescriptorImageInfo; 1]> = surfaces
            .iter()
            .map(|(_, s)| {
                [vk::DescriptorImageInfo::default()
                    .image_view(s.view())
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        writes.extend(
            surfaces
                .iter()
                .zip(&surface_arrays)
                .map(|((binding, _), info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(info)
                }),
        );
        // SAFETY: the set and the buffers are valid.
        unsafe { self.vk.device.update_descriptor_sets(&writes, &[]) };

        let pipeline = self.compute_pipeline(&program, pipeline_layout, &bindings)?;
        lap(&mut self.phases.compute[3]);

        // Leave the rendering that is begun; what was recorded runs first, then the launch, and
        // the wait that follows is what puts its writes in memory.
        self.end_pass();
        let cmd = self.frame.cmd(&self.vk)?;
        let device = &self.vk.device;
        // With `NX_TRACE_COMPUTE`, the device times the launch.
        let timestamps = if self.debug.trace_compute {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            // SAFETY: valid create info; destroyed below, after the launch has run.
            unsafe { device.create_query_pool(&info, None) }.ok()
        } else {
            None
        };
        let everything = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        // SAFETY: the command buffer is recording outside a rendering; every handle used is valid
        // for the submission.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &everything,
                &[],
                &[],
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[set],
                &[],
            );
            if let Some(pool) = timestamps {
                device.cmd_reset_query_pool(cmd, pool, 0, 2);
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, pool, 0);
            }
            device.cmd_dispatch(cmd, d.grid[0], d.grid[1], d.grid[2]);
            if let Some(pool) = timestamps {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, 1);
            }
            self.frame.mark(&self.vk, cmd, || {
                format!(
                    "compute launch of the program at {:#x}, grid {:?}",
                    d.program, d.grid
                )
            });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &everything,
                &[],
                &[],
            );
        }
        lap(&mut self.phases.compute[4]);
        // What the launch writes reaches memory when something needs it there (A-274): its
        // surfaces are copied out for the host after it, and what it reaches of global memory is
        // noted for the commands after it to wait for when they read it.
        let staged = self.stage_surfaces(scene, surfaces.into_iter().map(|(_, s)| s).collect())?;
        if program.info.global_memory {
            self.phases.launch_kinds[2] += 1;
            self.global_pending = true;
            self.note_reach(reached.as_deref(), program.info.global_writes);
        }
        if !staged.is_empty() {
            self.phases.launch_kinds[0] += 1;
            self.phases.launch_kinds[1] += staged.len() as u64;
        }
        self.deferred.extend(staged);
        // A launch that is timed, or that a switch asks to wait for, is waited for at once.
        // Else it joins the batch as a draw does: a batch of one launch would have the next one
        // wait for the device's slots almost every time.
        self.batch_draws += 1;
        let sent = if self.debug.compute_wait || self.debug.trace_compute || self.sync_on {
            self.settle(scene.space, scene.mem, settle::SYNC)
        } else if self.batch_draws >= super::BATCH_DRAWS {
            self.send_batch()
        } else {
            Ok(())
        };
        if let Some(pool) = timestamps {
            let mut ticks = [0u64; 2];
            // SAFETY: the pool has two queries written by the launch that has run.
            let read = unsafe {
                self.vk.device.get_query_pool_results(
                    pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            };
            if read.is_ok() {
                let us = ticks[1].saturating_sub(ticks[0]) as f64
                    * f64::from(self.vk.caps.timestamp_period)
                    / 1000.0;
                eprintln!(
                    "[vulkan] compute launch of the program at {:#x}, grid {:?}, block {:?}: {us:.0} us on the device",
                    d.program, d.grid, d.block
                );
            }
            // SAFETY: nothing uses the pool any more.
            unsafe { self.vk.device.destroy_query_pool(pool, None) };
        }
        lap(&mut self.phases.compute[5]);
        sent
    }

    /// Whether a launch not waited for writes `[from, to)` (GPU addresses) in place, or a surface
    /// still to be put there covers it: what the host reads there now is not what the commands
    /// after the launch must see (A-284).
    pub(super) fn host_read_hazard(&self, from: u64, to: u64) -> bool {
        self.writes_everywhere
            || overlaps(&self.pending_writes, from, to)
            || self.deferred.iter().any(|s| s.overlaps(from, to))
    }

    /// What a launch, or a draw that writes global memory, reaches of global memory in place (the
    /// storage buffers it accesses, all of memory when they are not known): the host and the GPU
    /// front wait for it before they read what it writes or change what it reaches (A-274,
    /// A-287).
    pub(super) fn note_reach(&mut self, reached: Option<&[(u64, u64)]>, writes: bool) {
        match reached {
            Some(buffers) => {
                let ranges: Vec<(u64, u64)> = buffers
                    .iter()
                    .map(|&(at, len)| (at, at.saturating_add(len)))
                    .collect();
                if writes {
                    self.pending_writes.extend_from_slice(&ranges);
                }
                self.pending_reach.extend(ranges);
            }
            None => {
                self.reach_everywhere = true;
                self.writes_everywhere |= writes;
            }
        }
    }

    /// Compute launches were recorded whose effects are not all in memory yet.
    pub(super) fn launches_pending(&self) -> bool {
        self.reach_everywhere
            || !self.pending_reach.is_empty()
            || !self.deferred.is_empty()
            || !self.releases.is_empty()
    }

    /// Wait for what was recorded, put what the launches' surfaces hold into memory and make the
    /// releases that follow them (A-274).
    pub(super) fn settle(
        &mut self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        cause: usize,
    ) -> Result<(), String> {
        if !self.launches_pending() {
            return Ok(());
        }
        self.memory_writes += 1;
        let mut mark = super::stopwatch();
        self.phases.settles[cause] += 1;
        let waited = self.flush();
        // What the launches wrote in place is in memory now: the surfaces kept from before that
        // it may have reached are looked at again (A-285).
        self.recheck_residents();
        let deferred = std::mem::take(&mut self.deferred);
        let put = std::time::Instant::now();
        let written = waited.and_then(|()| self.write_staged(space, mem, deferred));
        self.phases.surfaces_put.0 += put.elapsed();
        self.pending_writes.clear();
        self.pending_reach.clear();
        self.writes_everywhere = false;
        self.reach_everywhere = false;
        let mut released = Ok(());
        for r in std::mem::take(&mut self.releases) {
            released = released.and(r.apply(space, mem));
        }
        if let Some(mark) = mark.take() {
            self.phases.compute[6] += mark.elapsed();
        }
        written.and(released)
    }

    /// As [`Core::settle`], for the GPU front: `range` (a GPU address and a length) asks it only
    /// when a launch reaches those bytes, writes a surface over them or releases into them.
    pub(super) fn settle_for_front(
        &mut self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        range: Option<(u64, u64)>,
    ) -> Result<(), String> {
        if let Some((at, len)) = range {
            let to = at.saturating_add(len);
            // The front reads or writes those bytes itself: a surface kept over them may no
            // longer be what memory holds (A-285).
            self.forget_residents(at, to);
            if !self.reach_everywhere
                && !overlaps(&self.pending_reach, at, to)
                && !self.deferred.iter().any(|s| s.overlaps(at, to))
                && !self.releases.iter().any(|r| r.overlaps(at, to))
            {
                return Ok(());
            }
            // `NX_TRACE_PERF`: the first of these waits, with what they overlapped.
            if self.debug.trace_gpu && self.phases.settles[settle::FRONT_RANGE] < 3 {
                let hit: Vec<String> = self
                    .pending_reach
                    .iter()
                    .filter(|&&(a, b)| a < to && at < b)
                    .map(|(a, b)| format!("{a:#x}..{b:#x}"))
                    .collect();
                eprintln!(
                    "[vulkan] the GPU front's access of {len:#x} bytes at {at:#x} waits for launches: everywhere {}, reached {hit:?}, surfaces {}, releases {}",
                    self.reach_everywhere,
                    self.deferred.iter().any(|s| s.overlaps(at, to)),
                    self.releases.iter().any(|r| r.overlaps(at, to))
                );
            }
            return self.settle(space, mem, settle::FRONT_RANGE);
        }
        self.settle(space, mem, settle::FRONT)
    }

    /// Whether the launches not waited for, the surfaces still to be put and the releases still to
    /// be made may write any of `[from, to)` (A-274).
    pub(super) fn pending_may_write(&self, from: u64, to: u64) -> bool {
        self.launches_pending()
            && (self.writes_everywhere
                || overlaps(&self.pending_writes, from, to)
                || self.deferred.iter().any(|s| s.overlaps(from, to))
                || self.releases.iter().any(|r| r.overlaps(from, to)))
    }

    /// The word at `va` as the releases still to be made will leave it, when they alone are still
    /// to change it: no launch reaches it and no surface is to be put over it (A-274).
    pub(super) fn released_value(
        &self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        va: u64,
    ) -> Option<u32> {
        let to = va + 4;
        if !self.releases.iter().any(|r| r.overlaps(va, to))
            || self.reach_everywhere
            || overlaps(&self.pending_reach, va, to)
            || self.deferred.iter().any(|s| s.overlaps(va, to))
        {
            return None;
        }
        let mut bytes = [0u8; 4];
        if !space.read(mem, va, &mut bytes) {
            return None;
        }
        let word = u32::from_le_bytes(bytes);
        Some(self.releases.iter().fold(word, |w, r| r.applied_to(va, w)))
    }

    /// Wait for the launches before this one when it reads what they write: its constant buffers,
    /// textures and surfaces are copied on the host from memory, and what it reaches of global
    /// memory in place must not be where a surface is still to be put (A-274).
    fn settle_hazards(
        &mut self,
        scene: &Scene<'_>,
        d: &Dispatch,
        program: &Program,
    ) -> Result<(), String> {
        if !self.launches_pending() {
            return Ok(());
        }
        // What a launch before it writes there: 0 global memory, 1 a surface still to be put.
        let writer = |core: &Core, from: u64, to: u64| -> Option<usize> {
            if core.writes_everywhere || overlaps(&core.pending_writes, from, to) {
                Some(0)
            } else if core.deferred.iter().any(|s| s.overlaps(from, to)) {
                Some(1)
            } else {
                None
            }
        };
        // The kinds counted for the report: a constant buffer (0, 1), a texture or surface (2,
        // 3), after global writes and after surfaces; global memory over a surface (4).
        let hazard = |core: &mut Core, kind: usize| {
            core.phases.hazard_kinds[kind] += 1;
            core.settle(scene.space, scene.mem, settle::HAZARD)
        };
        for (cb, bound) in d.cbufs.iter().enumerate().take(18) {
            let used = program.info.cbuf_extent[cb] != 0 || program.info.cbuf_dynamic[cb];
            if let (true, Some(c)) = (used, bound) {
                if let Some(w) = writer(self, c.address, c.address + u64::from(c.size)) {
                    return hazard(self, w);
                }
            }
        }
        let pools = scene.regs.texture_pools();
        let handle = |slot: u32, index: u32| -> Option<u32> {
            let buffer = d.cbufs.get(slot as usize).copied().flatten()?;
            let mut word = [0u8; 4];
            scene
                .read(buffer.address + 4 * u64::from(index), &mut word)
                .then(|| u32::from_le_bytes(word) & 0xF_FFFF)
        };
        let header = |tic_id: u32| -> Option<super::textures::Tic> {
            if tic_id >= pools.header_count {
                return None;
            }
            let mut bytes = [0u8; 32];
            scene
                .read(pools.header_address + 32 * u64::from(tic_id), &mut bytes)
                .then(|| {
                    super::textures::Tic(std::array::from_fn(|i| {
                        u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
                    }))
                })
        };
        // (address, bytes, the kind of a hazard there after global writes).
        let mut spans = Vec::new();
        for used in &program.info.textures {
            let slot = used.source.unwrap_or(TEXTURE_TABLE);
            if let Some(tic) = handle(slot, used.index).and_then(header) {
                // A texture that is the image a launch before stored, not in memory yet, is
                // sampled as it is (A-283): no wait. What global writes put there is under what
                // the surface puts there afterwards anyway (A-274).
                if self
                    .surface_texture_view(scene, &tic, used, false)
                    .is_some()
                {
                    continue;
                }
                let span = self.textures.span(&tic, used).unwrap_or(1);
                spans.push((tic.address(), span, 2));
            }
        }
        for used in &program.info.images {
            let slot = used.source.unwrap_or(TEXTURE_TABLE);
            if let Some(tic) = handle(slot, used.index).and_then(header) {
                // A surface a launch before wrote, with the same image, is used as it is: no wait
                // (A-280), unless global writes reach it too.
                let forwarded = self
                    .surface_plan(&tic, used)
                    .ok()
                    .and_then(|(info, place)| self.forwardable(&info, &place).map(|_| place));
                if let Some(place) = forwarded {
                    if writer(self, place.address, place.end()) == Some(1) {
                        continue;
                    }
                }
                spans.push((
                    tic.address(),
                    super::surfaces::span(&tic, used).unwrap_or(1),
                    5,
                ));
            }
        }
        if let Some(kind) = spans
            .iter()
            .find_map(|&(at, len, kind)| writer(self, at, at.saturating_add(len)).map(|w| kind + w))
        {
            return hazard(self, kind);
        }
        if program.info.global_memory && !self.deferred.is_empty() {
            let read = |slot: u32, at: u32| -> Option<[u8; 12]> {
                let c = d.cbufs.get(slot as usize).copied().flatten()?;
                if u64::from(at) + 12 > u64::from(c.size) {
                    return None;
                }
                scene
                    .read_vec(c.address + u64::from(at), 12)?
                    .try_into()
                    .ok()
            };
            let clear = match super::global::storage_buffers(&program.info, read) {
                Some(buffers) => !buffers.iter().any(|&(at, len)| {
                    self.deferred
                        .iter()
                        .any(|s| s.overlaps(at, at.saturating_add(len)))
                }),
                None => false,
            };
            if !clear {
                return hazard(self, 4);
            }
        }
        Ok(())
    }

    /// The compute pipeline of `program` in `layout`, made on first use.
    fn compute_pipeline(
        &mut self,
        program: &Program,
        layout: vk::PipelineLayout,
        bindings: &[Binding],
    ) -> Result<vk::Pipeline, String> {
        let key = (program.id, bindings.to_vec());
        if let Some(p) = self.compute_pipelines.map.get(&key) {
            return Ok(*p);
        }
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(program.module)
            .name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        // SAFETY: valid create info; the pipeline is destroyed with the cache.
        let made = unsafe {
            self.vk
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
        }
        .map_err(|(_, e)| format!("creating the compute pipeline: {e}"))?;
        self.compute_pipelines.map.insert(key, made[0]);
        Ok(made[0])
    }
}

impl Core {
    /// Which surfaces of a launch's program (by slot) hold unsigned and signed integers, as
    /// their headers say (the types 4 and 3; A-272). The atomic ones are words of bits already.
    fn integer_surfaces(
        &mut self,
        scene: &Scene<'_>,
        d: &Dispatch,
        program: &super::shaders::Program,
    ) -> Result<(u16, u16), String> {
        let pools = scene.regs.texture_pools();
        let (mut uint, mut sint) = (0u16, 0u16);
        for used in &program.info.images {
            if used.atomic || used.slot >= 16 {
                continue;
            }
            let slot = used.source.unwrap_or(TEXTURE_TABLE);
            let Some(Some(buffer)) = d.cbufs.get(slot as usize) else {
                continue;
            };
            let mut word = [0u8; 4];
            self.read_exact(scene, buffer.address + 4 * u64::from(used.index), &mut word)?;
            let tic_id = u32::from_le_bytes(word) & 0xF_FFFF;
            if tic_id >= pools.header_count {
                continue;
            }
            let mut tic_bytes = [0u8; 32];
            self.read_exact(
                scene,
                pools.header_address + 32 * u64::from(tic_id),
                &mut tic_bytes,
            )?;
            let tic = super::textures::Tic(std::array::from_fn(|i| {
                u32::from_le_bytes(tic_bytes[4 * i..4 * i + 4].try_into().unwrap())
            }));
            match tic.types()[0] {
                4 => uint |= 1 << used.slot,
                3 => sint |= 1 << used.slot,
                _ => {}
            }
        }
        Ok((uint, sint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nx_gpu::qmd3::{Release, ReleaseOp};

    #[test]
    fn releases_still_to_be_made_leave_a_word_as_they_say() {
        let qmd = |op, payload| {
            Completion::Qmd(Release {
                va: 0x100,
                op,
                payload,
            })
        };
        assert_eq!(qmd(ReleaseOp::Set, 7).applied_to(0x100, 3), 7);
        assert_eq!(qmd(ReleaseOp::Add, 7).applied_to(0x100, 3), 10);
        assert_eq!(qmd(ReleaseOp::Max, 2).applied_to(0x100, 3), 3);
        assert_eq!(
            qmd(ReleaseOp::Set, 7).applied_to(0x104, 3),
            3,
            "another word"
        );
        // A semaphore release of 16 bytes (payload, then a timestamp) puts its first word there.
        let mut bytes = 9u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0xAA; 12]);
        assert_eq!(Completion::Bytes(0x100, bytes).applied_to(0x100, 3), 9);
        // One that covers part of the word changes those bytes only.
        let part = Completion::Bytes(0x102, vec![0x11, 0x22]);
        assert_eq!(part.applied_to(0x100, 0x4433_0201), 0x2211_0201);
    }
}
