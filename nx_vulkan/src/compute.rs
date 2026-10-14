//! Host-visible buffers and a one-shot compute dispatch: the harness the translator's tests run
//! shaders in, and a plain way to see what a SPIR-V module computes.

use crate::device::Vk;
use crate::image::{Image, ImageInfo};
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
    /// The address of byte `offset` of the mapping.
    pub fn ptr_at(&self, offset: u64) -> *mut u8 {
        debug_assert!(offset <= self.size);
        // SAFETY: the offset stays inside the mapping (checked in debug builds).
        unsafe { self.ptr.add(offset as usize) }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` maps `size` bytes for the life of the memory.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }
}

/// A texture bound as a combined image sampler (nearest filtering, clamped to the edge) in a
/// compute test.
pub struct SampledImage<'a> {
    pub binding: u32,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// The depth of a 3D image (`SampledKind::D3`), else 1.
    pub depth: u32,
    /// Array layers (6 for a cube, 1 for a plain 2D or 3D image).
    pub layers: u32,
    /// How the image is seen.
    pub kind: SampledKind,
    /// Read through a sampler that compares against a depth (less than or equal) instead of the
    /// plain one.
    pub compare: bool,
    /// The texels of every layer one after another, rows tightly packed (a 3D image's slices
    /// likewise).
    pub texels: &'a [u8],
}

/// The kind of view a test image is bound through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampledKind {
    D1,
    D2,
    D2Array,
    Cube,
    D3,
}

/// Before the dispatch, write the device address of buffer `of` (an index into the buffers of the
/// call) plus `addend` as a little-endian `u64` at byte `offset` of buffer `into`.
#[derive(Clone, Copy, Debug)]
pub struct AddressPatch {
    pub into: usize,
    pub offset: usize,
    pub of: usize,
    pub addend: u64,
}

/// A buffer in device local memory (see [`Vk::create_device_buffer`]).
pub struct DeviceBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
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
            // Cached memory where there is some. The host reads uncached (write-combined) memory
            // about thirty times slower than cached, which is what a buffer the device writes and
            // the host reads (a render target put into memory) needs; and what the host writes to
            // it a piece at a time (constants and vertices of every draw) goes out over the bus in
            // partial lines and costs the thread about a tenth of a frame.
            let Some(ty) = self
                .find_memory_type(
                    req.memory_type_bits,
                    wanted | vk::MemoryPropertyFlags::HOST_CACHED,
                )
                .or_else(|| self.find_memory_type(req.memory_type_bits, wanted))
            else {
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

    /// A buffer in memory only the device reaches (device local), for what shaders read many
    /// times and the host writes once: filled by a copy the caller records.
    pub fn create_device_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<DeviceBuffer, String> {
        let size = size.max(4);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info; the handles are destroyed in `destroy_device_buffer`.
        unsafe {
            let buffer = self
                .device
                .create_buffer(&info, None)
                .map_err(|e| format!("creating a buffer of {size} bytes: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let Some(ty) = self
                .find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .or_else(|| {
                    self.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty())
                })
            else {
                self.device.destroy_buffer(buffer, None);
                return Err("no memory type for a device buffer".to_string());
            };
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ty);
            let memory = match self.device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(format!(
                        "allocating {} bytes of device memory: {e}",
                        req.size
                    ));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buffer, memory, 0) {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
                return Err(format!("binding buffer memory: {e}"));
            }
            Ok(DeviceBuffer {
                buffer,
                memory,
                size,
            })
        }
    }

    pub fn destroy_device_buffer(&self, b: DeviceBuffer) {
        // SAFETY: the buffer is no longer in use by the caller or the device.
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    /// The device address of a buffer made with `SHADER_DEVICE_ADDRESS` usage.
    pub fn buffer_address(&self, buffer: &HostBuffer) -> u64 {
        // SAFETY: the buffer is valid and was created with the device address usage.
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer),
            )
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
        self.run_compute_with(spirv, buffers, &[], &[], groups)
    }

    /// As [`Vk::run_compute`], with textures bound too and device addresses written into the
    /// buffers (for shaders that follow pointers).
    pub fn run_compute_with(
        &self,
        spirv: &[u32],
        buffers: &mut [(u32, &mut [u8])],
        images: &[SampledImage<'_>],
        patches: &[AddressPatch],
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
        for p in patches {
            let address = self.buffer_address(&host[p.of]) + p.addend;
            host[p.into].as_mut_slice()[p.offset..p.offset + 8]
                .copy_from_slice(&address.to_le_bytes());
        }
        let mut gpu_images: Vec<Image> = Vec::new();
        let mut sampler = None;
        let mut compare_sampler = None;
        let prepared = (|| -> Result<(), String> {
            if images.iter().any(|i| !i.compare) {
                sampler = Some(self.create_nearest_sampler()?);
            }
            if images.iter().any(|i| i.compare) {
                compare_sampler = Some(self.create_compare_sampler()?);
            }
            for i in images {
                let image = self.create_image(ImageInfo {
                    format: i.format,
                    width: i.width,
                    height: i.height,
                    depth: i.depth,
                    layers: i.layers,
                    usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    view_type: match i.kind {
                        SampledKind::D1 => vk::ImageViewType::TYPE_1D,
                        SampledKind::D2 => vk::ImageViewType::TYPE_2D,
                        SampledKind::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
                        SampledKind::Cube => vk::ImageViewType::CUBE,
                        SampledKind::D3 => vk::ImageViewType::TYPE_3D,
                    },
                    components: vk::ComponentMapping::default(),
                    levels: 1,
                })?;
                let uploaded = self.upload_image(&image, i.texels);
                gpu_images.push(image);
                uploaded?;
            }
            Ok(())
        })();
        // SAFETY: every object created below is destroyed before returning, and the command
        // buffer completes (fence wait) before the buffers are read or destroyed.
        let result = prepared.and_then(|()| unsafe {
            (|| -> Result<(), String> {
                let module = d
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
                    .map_err(|e| format!("creating the shader module: {e}"))?;
                let mut bindings: Vec<_> = buffers
                    .iter()
                    .map(|(binding, _)| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(*binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    })
                    .collect();
                bindings.extend(images.iter().map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i.binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                }));
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
                let mut pool_sizes = vec![vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(buffers.len().max(1) as u32)];
                if !images.is_empty() {
                    pool_sizes.push(
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(images.len() as u32),
                    );
                }
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
                let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = gpu_images
                    .iter()
                    .zip(images)
                    .map(|(i, wanted)| {
                        let chosen = if wanted.compare {
                            compare_sampler
                        } else {
                            sampler
                        };
                        [vk::DescriptorImageInfo::default()
                            .sampler(chosen.expect("a sampler exists for each image"))
                            .image_view(i.view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
                    })
                    .collect();
                let mut writes: Vec<_> = buffers
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
                writes.extend(images.iter().zip(&image_infos).map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i.binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(info)
                }));
                d.update_descriptor_sets(&writes, &[]);
                let outcome = self.one_shot(|d, cmd| {
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
                });
                d.destroy_descriptor_pool(pool, None);
                d.destroy_pipeline(pipeline, None);
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                d.destroy_shader_module(module, None);
                outcome
            })()
        });
        if result.is_ok() {
            for ((_, data), b) in buffers.iter_mut().zip(&host) {
                let n = data.len();
                data.copy_from_slice(&b.as_slice()[..n]);
            }
        }
        for i in gpu_images {
            self.destroy_image(i);
        }
        for s in [sampler, compare_sampler].into_iter().flatten() {
            // SAFETY: the dispatch has finished, so nothing uses the sampler.
            unsafe { d.destroy_sampler(s, None) };
        }
        for b in host {
            self.destroy_host_buffer(b);
        }
        result
    }
}

/// Host memory the device reads and writes in place (`VK_EXT_external_memory_host`).
pub struct ImportedMemory {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// The device address of the first byte.
    pub address: u64,
    pub size: u64,
}

impl Vk {
    /// Make `size` bytes of this process's memory at `ptr` (both a multiple of the import
    /// alignment) visible to the device, as a storage buffer with a device address. The device
    /// reads and writes the bytes themselves.
    pub fn import_host_memory(&self, ptr: *mut u8, size: u64) -> Result<ImportedMemory, String> {
        let Some(ext) = self.external_memory_host.as_ref() else {
            return Err(
                "the device cannot import host memory (VK_EXT_external_memory_host)".to_string(),
            );
        };
        let align = self.caps.min_imported_host_pointer_alignment.max(1);
        if !(ptr as u64).is_multiple_of(align) || !size.is_multiple_of(align) || size == 0 {
            return Err(format!(
                "host memory at {:p} of {size} bytes does not meet the import alignment of {align}",
                ptr
            ));
        }
        let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        // SAFETY: the pointer is to memory this process owns for as long as the import lives;
        // every handle made here is destroyed on failure or by `destroy_imported`.
        unsafe {
            let mut props = vk::MemoryHostPointerPropertiesEXT::default();
            (ext.fp().get_memory_host_pointer_properties_ext)(
                ext.device(),
                handle_type,
                ptr.cast(),
                &mut props,
            )
            .result()
            .map_err(|e| format!("asking what host memory at {ptr:p} can be: {e}"))?;
            let mut external =
                vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external);
            let buffer = self
                .device
                .create_buffer(&info, None)
                .map_err(|e| format!("creating a buffer over {size} bytes of host memory: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let Some(ty) = self.find_memory_type(
                req.memory_type_bits & props.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            ) else {
                self.device.destroy_buffer(buffer, None);
                return Err("no memory type can hold imported host memory".to_string());
            };
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(handle_type)
                .host_pointer(ptr.cast());
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(ty)
                .push_next(&mut import)
                .push_next(&mut flags);
            let memory = match self.device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(format!("importing {size} bytes of host memory: {e}"));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buffer, memory, 0) {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
                return Err(format!("binding imported host memory: {e}"));
            }
            let address = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok(ImportedMemory {
                buffer,
                memory,
                address,
                size,
            })
        }
    }

    pub fn destroy_imported(&self, m: ImportedMemory) {
        // SAFETY: the buffer is no longer in use by the caller.
        unsafe {
            self.device.destroy_buffer(m.buffer, None);
            self.device.free_memory(m.memory, None);
        }
    }
}
