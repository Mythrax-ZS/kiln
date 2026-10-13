//! Images: creation in device-local memory, upload through a staging buffer, and the one-shot
//! command submission both need.

use crate::device::Vk;
use ash::vk;

/// What to create.
#[derive(Clone, Copy, Debug)]
pub struct ImageInfo {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub usage: vk::ImageUsageFlags,
    /// The view type: `TYPE_2D` or `TYPE_2D_ARRAY` for now.
    pub view_type: vk::ImageViewType,
}

/// An image with its memory and a view of all of it.
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub info: ImageInfo,
}

/// The aspect an image of `format` is read through.
pub fn aspect_of(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

impl Vk {
    /// Record commands with `record`, submit them and wait for them to finish.
    pub fn one_shot(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), String> {
        let d = &self.device;
        // SAFETY: every object made here is destroyed before returning, and the work has finished
        // (fence wait) by then.
        unsafe {
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family),
                    None,
                )
                .map_err(|e| format!("creating a command pool: {e}"))?;
            let result = (|| -> Result<(), String> {
                let cmd = d
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .map_err(|e| format!("allocating a command buffer: {e}"))?[0];
                d.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("beginning a command buffer: {e}"))?;
                record(d, cmd);
                d.end_command_buffer(cmd)
                    .map_err(|e| format!("ending a command buffer: {e}"))?;
                let fence = d
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| format!("creating a fence: {e}"))?;
                let cmds = [cmd];
                let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
                let outcome = d
                    .queue_submit(self.queue, &submit, fence)
                    .and_then(|()| d.wait_for_fences(&[fence], true, 60_000_000_000))
                    .map_err(|e| format!("running a command buffer: {e}"));
                d.destroy_fence(fence, None);
                outcome
            })();
            d.destroy_command_pool(pool, None);
            result
        }
    }

    /// An image in device-local memory (the layout is undefined until it is first used).
    pub fn create_image(&self, info: ImageInfo) -> Result<Image, String> {
        let d = &self.device;
        let ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(info.format)
            .extent(vk::Extent3D {
                width: info.width,
                height: info.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(info.layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid create info; the handles are destroyed in `destroy_image`.
        unsafe {
            let image = d
                .create_image(&ci, None)
                .map_err(|e| format!("creating a {}x{} image: {e}", info.width, info.height))?;
            let req = d.get_image_memory_requirements(image);
            let Some(ty) =
                self.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            else {
                d.destroy_image(image, None);
                return Err("no device-local memory type for an image".to_string());
            };
            let memory = match d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            ) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_image(image, None);
                    return Err(format!("allocating {} bytes for an image: {e}", req.size));
                }
            };
            if let Err(e) = d.bind_image_memory(image, memory, 0) {
                d.free_memory(memory, None);
                d.destroy_image(image, None);
                return Err(format!("binding image memory: {e}"));
            }
            let view = d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(info.view_type)
                    .format(info.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect_of(info.format),
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: info.layers,
                    }),
                None,
            );
            match view {
                Ok(view) => Ok(Image {
                    image,
                    view,
                    memory,
                    info,
                }),
                Err(e) => {
                    d.free_memory(memory, None);
                    d.destroy_image(image, None);
                    Err(format!("creating an image view: {e}"))
                }
            }
        }
    }

    pub fn destroy_image(&self, image: Image) {
        // SAFETY: the image is no longer in use by the caller.
        unsafe {
            self.device.destroy_image_view(image.view, None);
            self.device.destroy_image(image.image, None);
            self.device.free_memory(image.memory, None);
        }
    }

    /// Fill all layers of an image from `texels` (the layers one after another, rows tightly
    /// packed) and leave it in the layout for sampling in shaders. The image needs
    /// `TRANSFER_DST` usage.
    pub fn upload_image(&self, image: &Image, texels: &[u8]) -> Result<(), String> {
        let mut staging =
            self.create_host_buffer(texels.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
        staging.as_mut_slice()[..texels.len()].copy_from_slice(texels);
        let aspect = aspect_of(image.info.format);
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: image.info.layers,
        };
        let result = self.one_shot(|d, cmd| {
            let to_dst = [vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(range)];
            let region = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: image.info.layers,
                })
                .image_extent(vk::Extent3D {
                    width: image.info.width,
                    height: image.info.height,
                    depth: 1,
                })];
            let to_read = [vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(range)];
            // SAFETY: the command buffer is recording and the handles are valid.
            unsafe {
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_dst,
                );
                d.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                );
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_read,
                );
            }
        });
        self.destroy_host_buffer(staging);
        result
    }

    /// A sampler with nearest filtering and clamping to the edge, the plainest one there is.
    pub fn create_nearest_sampler(&self) -> Result<vk::Sampler, String> {
        let ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        // SAFETY: valid create info; the caller destroys the sampler.
        unsafe { self.device.create_sampler(&ci, None) }
            .map_err(|e| format!("creating a sampler: {e}"))
    }
}
