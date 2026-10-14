//! Compute launches: the program translated for its workgroup, the constant buffers the descriptor
//! binds, and a dispatch that has finished (and so has written its memory) when it returns.
//!
//! The commands recorded so far are run first, and the launch after them, because what later
//! draws read of memory (their constant buffers, vertices and textures) is copied when they are
//! recorded, so it has to be in memory by then (docs/ASSUMPTIONS.md A-110).

use super::shaders::{bindings_of, Binding, Program};
use super::Core;
use crate::fast::FastMap;
use ash::vk;
use nx_gpu::compute::Dispatch;
use nx_gpu::render::Scene;
use std::time::Instant;

/// The constant buffer slot that holds the handles of the textures a compute program names (the
/// title puts them there for a launch as it does for a draw).
const TEXTURE_TABLE: u32 = 2;

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
        let mut mark = super::stopwatch();
        let mut lap = |into: &mut std::time::Duration| {
            if let Some(mark) = &mut mark {
                let now = Instant::now();
                *into += now - *mark;
                *mark = now;
            }
        };
        let program =
            self.shaders
                .kernel(&self.vk, scene, d.program, d.block, self.textures.frame)?;
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
        if program.info.global_memory {
            let (buffer, size) = self.global.table(&self.vk, &mut self.frame, scene)?;
            buffer_infos.push((
                base + 20,
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(0)
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
        self.bind_textures(scene, &program, "compute", &pools, &handles, &mut images)?;
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
        let flushed = self.flush();
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
        flushed
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
