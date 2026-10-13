//! Host-visible buffers and a one-shot compute dispatch: the harness the translator's tests run
//! shaders in, and a plain way to see what a SPIR-V module computes.

use crate::device::Vk;
use ash::vk;

/// A buffer in host-visible, coherent memory, mapped for as long as it lives.
pub struct HostBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    ptr: *mut u8,
}

// SAFETY: the mapping is only touched through `&self`/`&mut self` methods of the owner.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` maps `size` bytes for the life of the memory.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }
}

impl Vk {
    pub fn create_host_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<HostBuffer, String> {
        let size = size.max(4);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info; the handles are destroyed in `destroy_host_buffer`.
        unsafe {
            let buffer = self
                .device
                .create_buffer(&info, None)
                .map_err(|e| format!("creating a buffer of {size} bytes: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let wanted =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let Some(ty) = self.find_memory_type(req.memory_type_bits, wanted) else {
                self.device.destroy_buffer(buffer, None);
                return Err("no host-visible coherent memory type".to_string());
            };
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ty);
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                alloc = alloc.push_next(&mut flags);
            }
            let memory = match self.device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(format!("allocating {} bytes of memory: {e}", req.size));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buffer, memory, 0) {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
                return Err(format!("binding buffer memory: {e}"));
            }
            let ptr =
                match self
                    .device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                {
                    Ok(p) => p.cast::<u8>(),
                    Err(e) => {
                        self.device.free_memory(memory, None);
                        self.device.destroy_buffer(buffer, None);
                        return Err(format!("mapping buffer memory: {e}"));
                    }
                };
            std::ptr::write_bytes(ptr, 0, size as usize);
            Ok(HostBuffer {
                buffer,
                memory,
                size,
                ptr,
            })
        }
    }

    pub fn destroy_host_buffer(&self, b: HostBuffer) {
        // SAFETY: the buffer is no longer in use by the caller.
        unsafe {
            self.device.unmap_memory(b.memory);
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    /// Run a compute shader once: one storage buffer per `(binding, data)` of descriptor set 0,
    /// filled from `data` before the dispatch and copied back after it. `groups` is the number of
    /// workgroups. The entry point is `main`.
    pub fn run_compute(
        &self,
        spirv: &[u32],
        buffers: &mut [(u32, &mut [u8])],
        groups: [u32; 3],
    ) -> Result<(), String> {
        let d = &self.device;
        let mut host: Vec<HostBuffer> = Vec::new();
        for (_, data) in buffers.iter() {
            let mut b = self.create_host_buffer(
                data.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )?;
            b.as_mut_slice()[..data.len()].copy_from_slice(data);
            host.push(b);
        }
        // SAFETY: every object created below is destroyed before returning, and the command
        // buffer completes (fence wait) before the buffers are read or destroyed.
        let result = unsafe {
            (|| -> Result<(), String> {
                let module = d
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
                    .map_err(|e| format!("creating the shader module: {e}"))?;
                let bindings: Vec<_> = buffers
                    .iter()
                    .map(|(binding, _)| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(*binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    })
                    .collect();
                let set_layout = d
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )
                    .map_err(|e| format!("creating the descriptor set layout: {e}"))?;
                let set_layouts = [set_layout];
                let layout = d
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                        None,
                    )
                    .map_err(|e| format!("creating the pipeline layout: {e}"))?;
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                let pipeline_info = [vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)];
                let pipelines = d
                    .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                    .map_err(|(_, e)| format!("creating the compute pipeline: {e}"))?;
                let pipeline = pipelines[0];
                let pool_sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(buffers.len().max(1) as u32)];
                let pool = d
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(1)
                            .pool_sizes(&pool_sizes),
                        None,
                    )
                    .map_err(|e| format!("creating the descriptor pool: {e}"))?;
                let set = d
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(pool)
                            .set_layouts(&set_layouts),
                    )
                    .map_err(|e| format!("allocating the descriptor set: {e}"))?[0];
                let infos: Vec<[vk::DescriptorBufferInfo; 1]> = host
                    .iter()
                    .map(|b| {
                        [vk::DescriptorBufferInfo::default()
                            .buffer(b.buffer)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]
                    })
                    .collect();
                let writes: Vec<_> = buffers
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
                d.update_descriptor_sets(&writes, &[]);
                let cmd_pool = d
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family),
                        None,
                    )
                    .map_err(|e| format!("creating the command pool: {e}"))?;
                let cmd = d
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .map_err(|e| format!("allocating a command buffer: {e}"))?[0];
                d.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("beginning the command buffer: {e}"))?;
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                d.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
                d.end_command_buffer(cmd)
                    .map_err(|e| format!("ending the command buffer: {e}"))?;
                let fence = d
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| format!("creating a fence: {e}"))?;
                let cmds = [cmd];
                let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
                let outcome = d
                    .queue_submit(self.queue, &submit, fence)
                    .and_then(|()| d.wait_for_fences(&[fence], true, 60_000_000_000))
                    .map_err(|e| format!("running the dispatch: {e}"));
                d.destroy_fence(fence, None);
                d.destroy_command_pool(cmd_pool, None);
                d.destroy_descriptor_pool(pool, None);
                d.destroy_pipeline(pipeline, None);
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                d.destroy_shader_module(module, None);
                outcome
            })()
        };
        if result.is_ok() {
            for ((_, data), b) in buffers.iter_mut().zip(&host) {
                let n = data.len();
                data.copy_from_slice(&b.as_slice()[..n]);
            }
        }
        for b in host {
            self.destroy_host_buffer(b);
        }
        result
    }
}

impl Vk {
    /// The device address of a buffer made with `SHADER_DEVICE_ADDRESS` usage.
    pub fn buffer_address(&self, buffer: &HostBuffer) -> u64 {
        // SAFETY: the buffer is valid and was created with the device address usage.
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer),
            )
        }
    }
}
