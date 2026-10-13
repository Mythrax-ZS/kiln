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
    /// The depth of a 3D image (a view type of `TYPE_3D`); 1 for every other.
    pub depth: u32,
    /// Array layers (6 for a cube, 1 for a 3D image).
    pub layers: u32,
    /// Mip levels (the base level alone is 1).
    pub levels: u32,
    pub usage: vk::ImageUsageFlags,
    /// The view type: `TYPE_2D`, `TYPE_2D_ARRAY`, `CUBE` or `TYPE_3D`.
    pub view_type: vk::ImageViewType,
    /// What the view makes of the components (identity unless a texture says otherwise).
    pub components: vk::ComponentMapping,
}

/// An image with its memory and a view of all of it.
pub struct Image {
    pub image: vk::Image,
    /// A view of all of it (null for an image that is only ever transferred to and from).
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub info: ImageInfo,
    /// Other views of the image, by the swizzle they apply (see [`Vk::swizzled_view`]).
    pub extra_views: Vec<(vk::ComponentMapping, vk::ImageView)>,
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
        let volume = info.view_type == vk::ImageViewType::TYPE_3D;
        let cube = matches!(
            info.view_type,
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
        );
        let mut ci = vk::ImageCreateInfo::default()
            .image_type(if volume {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            })
            .format(info.format)
            .extent(vk::Extent3D {
                width: info.width,
                height: info.height,
                depth: if volume { info.depth } else { 1 },
            })
            .mip_levels(info.levels.max(1))
            .array_layers(if volume { 1 } else { info.layers })
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if cube {
            ci = ci.flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);
        }
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
            // An image that only transfers (a copy's scratch space) cannot have a view.
            let viewable = info.usage.intersects(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            );
            let view = if viewable {
                d.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(info.view_type)
                        .components(info.components)
                        .format(info.format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: aspect_of(info.format),
                            base_mip_level: 0,
                            level_count: info.levels.max(1),
                            base_array_layer: 0,
                            layer_count: if volume { 1 } else { info.layers },
                        }),
                    None,
                )
            } else {
                Ok(vk::ImageView::null())
            };
            match view {
                Ok(view) => Ok(Image {
                    image,
                    view,
                    memory,
                    info,
                    extra_views: Vec::new(),
                }),
                Err(e) => {
                    d.free_memory(memory, None);
                    d.destroy_image(image, None);
                    Err(format!("creating an image view: {e}"))
                }
            }
        }
    }

    /// A view of all of `image` that applies `components` (made once per swizzle).
    pub fn swizzled_view(
        &self,
        image: &mut Image,
        components: vk::ComponentMapping,
    ) -> Result<vk::ImageView, String> {
        if let Some((_, v)) = image.extra_views.iter().find(|(c, _)| {
            (c.r, c.g, c.b, c.a) == (components.r, components.g, components.b, components.a)
        }) {
            return Ok(*v);
        }
        // A view that is sampled reads one aspect: the depth of a depth and stencil image.
        let aspect = aspect_of(image.info.format);
        let aspect = if aspect.contains(vk::ImageAspectFlags::DEPTH) {
            vk::ImageAspectFlags::DEPTH
        } else {
            aspect
        };
        // SAFETY: valid create info over an image that lives as long as the view.
        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.image)
                    .view_type(image.info.view_type)
                    .components(components)
                    .format(image.info.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: image.info.levels.max(1),
                        base_array_layer: 0,
                        layer_count: if image.info.view_type == vk::ImageViewType::TYPE_3D {
                            1
                        } else {
                            image.info.layers
                        },
                    }),
                None,
            )
        }
        .map_err(|e| format!("creating an image view: {e}"))?;
        image.extra_views.push((components, view));
        Ok(view)
    }

    pub fn destroy_image(&self, image: Image) {
        // SAFETY: the image is no longer in use by the caller.
        unsafe {
            for (_, v) in &image.extra_views {
                self.device.destroy_image_view(*v, None);
            }
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
        let volume = image.info.view_type == vk::ImageViewType::TYPE_3D;
        let layers = if volume { 1 } else { image.info.layers };
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
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
                    layer_count: layers,
                })
                .image_extent(vk::Extent3D {
                    width: image.info.width,
                    height: image.info.height,
                    depth: if volume { image.info.depth } else { 1 },
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

    /// As [`Vk::create_nearest_sampler`], comparing the reference of a shadow sample against the
    /// depth read with "less than or equal".
    pub fn create_compare_sampler(&self) -> Result<vk::Sampler, String> {
        let ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .compare_enable(true)
            .compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .max_lod(0.0);
        // SAFETY: valid create info; the caller destroys the sampler.
        unsafe { self.device.create_sampler(&ci, None) }
            .map_err(|e| format!("creating a comparison sampler: {e}"))
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
