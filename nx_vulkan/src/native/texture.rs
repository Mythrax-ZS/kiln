//! Retained native texture storage and raw, bounded subresource transfers (A-334).

use super::{Completion, Context, Queue};
use crate::compute::HostBuffer;
use crate::image::{Image, ImageInfo};
use crate::Vk;
use ash::vk;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug)]
pub struct TexelFormat {
    pub(crate) vk: vk::Format,
    block: [u32; 2],
    bytes: u32,
}
impl TexelFormat {
    /// Component/storage encoding, independent of guest command processor state.
    pub fn new(component: u32, data_type: u32, srgb: bool) -> Result<Self, String> {
        let f = crate::render::texture_format(component, data_type, srgb)
            .ok_or("native texel format is unsupported")?;
        if f.astc || crate::image::aspect_of(f.vk) != vk::ImageAspectFlags::COLOR {
            return Err("native ASTC/depth/stencil transfers are not implemented".into());
        }
        Ok(Self {
            vk: f.vk,
            block: [f.block.0, f.block.1],
            bytes: f.bytes,
        })
    }
    pub(crate) fn display(format: vk::Format) -> Self {
        Self {
            vk: format,
            block: [1, 1],
            bytes: if format == vk::Format::R16G16B16A16_SFLOAT {
                8
            } else {
                4
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ImageKind {
    Line,
    Plane,
    Volume,
    LineArray,
    PlaneArray,
    Cube,
    CubeArray,
}
impl ImageKind {
    fn view(self) -> vk::ImageViewType {
        match self {
            Self::Line => vk::ImageViewType::TYPE_1D,
            Self::Plane => vk::ImageViewType::TYPE_2D,
            Self::Volume => vk::ImageViewType::TYPE_3D,
            Self::LineArray => vk::ImageViewType::TYPE_1D_ARRAY,
            Self::PlaneArray => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::Cube => vk::ImageViewType::CUBE,
            Self::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct TextureInfo {
    pub kind: ImageKind,
    pub dimensions: [u32; 3],
    pub layers: u32,
    pub levels: u32,
    pub format: TexelFormat,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRegion {
    pub level: u32,
    pub layer: u32,
    pub layers: u32,
    pub origin: [u32; 3],
    pub extent: [u32; 3],
}
impl TextureRegion {
    pub(crate) fn checked(
        self,
        info: ImageInfo,
        f: TexelFormat,
    ) -> Result<(vk::BufferImageCopy, usize), String> {
        let dimensions =
            [info.width, info.height, info.depth].map(|n| (n >> self.level.min(31)).max(1));
        if self.level >= info.levels
            || self.layers == 0
            || self.extent.contains(&0)
            || !self
                .layer
                .checked_add(self.layers)
                .is_some_and(|n| n <= info.layers)
            || (0..3).any(|i| {
                !self.origin[i]
                    .checked_add(self.extent[i])
                    .is_some_and(|n| n <= dimensions[i])
            })
            || info.view_type == vk::ImageViewType::TYPE_3D && (self.layer != 0 || self.layers != 1)
        {
            return Err("native texel region exceeds image subresources".into());
        }
        for (i, &unit) in f.block.iter().enumerate() {
            if !self.origin[i].is_multiple_of(unit)
                || !self.extent[i].is_multiple_of(unit)
                    && self.origin[i] + self.extent[i] != dimensions[i]
            {
                return Err("native compressed texel region is not block aligned".into());
            }
        }
        let size = u64::from(self.extent[0].div_ceil(f.block[0]))
            .checked_mul(u64::from(self.extent[1].div_ceil(f.block[1])))
            .and_then(|n| n.checked_mul(u64::from(self.extent[2])))
            .and_then(|n| n.checked_mul(u64::from(self.layers)))
            .and_then(|n| n.checked_mul(u64::from(f.bytes)))
            .filter(|&n| n <= 256 * 1024 * 1024)
            .ok_or("native texel transfer exceeds the 256 MiB budget")? as usize;
        Ok((
            vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(self.level)
                        .base_array_layer(self.layer)
                        .layer_count(self.layers),
                )
                .image_offset(vk::Offset3D {
                    x: self.origin[0] as i32,
                    y: self.origin[1] as i32,
                    z: self.origin[2] as i32,
                })
                .image_extent(vk::Extent3D {
                    width: self.extent[0],
                    height: self.extent[1],
                    depth: self.extent[2],
                }),
            size,
        ))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TransferStats {
    pub uploads: u64,
    pub upload_bytes: u64,
    pub reads: u64,
    pub readback_bytes: u64,
}

pub(crate) struct Staging {
    vk: Arc<Vk>,
    buffer: Mutex<Option<HostBuffer>>,
}
impl Staging {
    pub(crate) fn new(
        vk: &Arc<Vk>,
        size: usize,
        upload: Option<&[u8]>,
    ) -> Result<Arc<Self>, String> {
        let mut buffer = vk.create_host_buffer(
            size as u64,
            if upload.is_some() {
                vk::BufferUsageFlags::TRANSFER_SRC
            } else {
                vk::BufferUsageFlags::TRANSFER_DST
            },
        )?;
        if let Some(bytes) = upload {
            buffer.as_mut_slice().copy_from_slice(bytes);
        }
        Ok(Arc::new(Self {
            vk: Arc::clone(vk),
            buffer: Mutex::new(Some(buffer)),
        }))
    }
    pub(crate) fn handle(&self) -> vk::Buffer {
        self.buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .expect("live staging")
            .buffer
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        self.vk.destroy_host_buffer(
            self.buffer
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("live staging"),
        );
    }
}
pub struct Readback {
    pub(crate) done: Arc<Completion>,
    pub(crate) staging: Arc<Staging>,
}
impl Readback {
    /// GPU completion is waited outside the driver/guest-core locks.
    pub fn wait(self) -> Result<Vec<u8>, String> {
        self.done.wait(u64::MAX)?;
        Ok(self
            .staging
            .buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .expect("live staging")
            .as_slice()
            .to_vec())
    }
}

pub(crate) fn transfer_barrier(
    vk: &Vk,
    cmd: vk::CommandBuffer,
    image: &Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    // SAFETY: the caller serializes image state and retains it through GPU completion.
    unsafe {
        vk.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .src_access_mask(if old == vk::ImageLayout::UNDEFINED {
                    vk::AccessFlags::empty()
                } else {
                    vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ
                })
                .dst_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(image.info.levels)
                        .layer_count(image.info.layers),
                )],
        );
    }
}
pub(crate) fn record_copy(
    vk: &Vk,
    cmd: vk::CommandBuffer,
    image: &Image,
    layout: vk::ImageLayout,
    buffer: vk::Buffer,
    region: vk::BufferImageCopy,
    upload: bool,
) {
    // SAFETY: checked tightly packed buffer/image region; state and staging are retained.
    unsafe {
        if upload {
            vk.device
                .cmd_copy_buffer_to_image(cmd, buffer, image.image, layout, &[region]);
        } else {
            vk.device
                .cmd_copy_image_to_buffer(cmd, image.image, layout, buffer, &[region]);
            vk.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
        }
    }
}

struct State {
    image: Option<Image>,
    queue: Queue,
    layout: vk::ImageLayout,
    stats: TransferStats,
    last: Option<Arc<Completion>>,
}
pub struct Texture {
    vk: Arc<Vk>,
    format: TexelFormat,
    state: Mutex<State>,
}
impl Context {
    pub fn texture(&self, info: TextureInfo) -> Result<Arc<Texture>, String> {
        let view_type = info.kind.view();
        if info.dimensions.contains(&0)
            || info.layers == 0
            || info.levels == 0
            || info.levels > 16
            || info.levels > 32 - info.dimensions.iter().max().unwrap().leading_zeros()
            || view_type == vk::ImageViewType::TYPE_3D && info.layers != 1
            || view_type != vk::ImageViewType::TYPE_3D && info.dimensions[2] != 1
            || matches!(
                view_type,
                vk::ImageViewType::TYPE_1D
                    | vk::ImageViewType::TYPE_2D
                    | vk::ImageViewType::TYPE_3D
            ) && info.layers != 1
            || matches!(
                view_type,
                vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
            ) && (info.dimensions[0] != info.dimensions[1] || !info.layers.is_multiple_of(6))
            || view_type == vk::ImageViewType::CUBE && info.layers != 6
            || matches!(
                view_type,
                vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY
            ) && (info.dimensions[1] != 1 || info.format.block != [1, 1])
        {
            return Err("native texture shape is invalid or unsupported".into());
        }
        let mut bytes = 0u64;
        for level in 0..info.levels {
            let d = info.dimensions.map(|n| (n >> level).max(1));
            let size = u64::from(d[0].div_ceil(info.format.block[0]))
                .checked_mul(u64::from(d[1].div_ceil(info.format.block[1])))
                .and_then(|n| n.checked_mul(u64::from(d[2])))
                .and_then(|n| n.checked_mul(u64::from(info.layers)))
                .and_then(|n| n.checked_mul(u64::from(info.format.bytes)))
                .ok_or("native texture storage size overflows")?;
            bytes = bytes
                .checked_add(size)
                .filter(|&n| n <= 256 * 1024 * 1024)
                .ok_or("native texture exceeds the 256 MiB storage budget")?;
        }
        let image_info = ImageInfo {
            format: info.format.vk,
            width: info.dimensions[0],
            height: info.dimensions[1],
            depth: info.dimensions[2],
            layers: info.layers,
            levels: info.levels,
            usage: vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            view_type,
            components: vk::ComponentMapping::default(),
        };
        // SAFETY: capability query of this live physical device. Invalid extents/usages fail
        // before allocating or changing CPU texture storage.
        let supported = unsafe {
            self.vk
                .instance
                .get_physical_device_image_format_properties(
                    self.vk.physical,
                    image_info.format,
                    if view_type == vk::ImageViewType::TYPE_3D {
                        vk::ImageType::TYPE_3D
                    } else if matches!(
                        view_type,
                        vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY
                    ) {
                        vk::ImageType::TYPE_1D
                    } else {
                        vk::ImageType::TYPE_2D
                    },
                    vk::ImageTiling::OPTIMAL,
                    image_info.usage,
                    if matches!(
                        view_type,
                        vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
                    ) {
                        vk::ImageCreateFlags::CUBE_COMPATIBLE
                    } else {
                        vk::ImageCreateFlags::empty()
                    },
                )
        }
        .map_err(|e| format!("native texture format/shape is unsupported: {e}"))?;
        if info.dimensions[0] > supported.max_extent.width
            || info.dimensions[1] > supported.max_extent.height
            || info.dimensions[2] > supported.max_extent.depth
            || info.layers > supported.max_array_layers
            || info.levels > supported.max_mip_levels
        {
            return Err("native texture exceeds device image limits".into());
        }
        let queue = self.queue()?;
        let image = self.vk.create_image(image_info)?;
        Ok(Arc::new(Texture {
            vk: Arc::clone(&self.vk),
            format: info.format,
            state: Mutex::new(State {
                image: Some(image),
                queue,
                layout: vk::ImageLayout::UNDEFINED,
                stats: TransferStats::default(),
                last: None,
            }),
        }))
    }
}
impl Texture {
    pub fn stats(&self) -> TransferStats {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).stats
    }
    pub fn upload(&self, region: TextureRegion, bytes: &[u8]) -> Result<Arc<Completion>, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (copy, size) =
            region.checked(state.image.as_ref().expect("live image").info, self.format)?;
        if bytes.len() != size {
            return Err("native texture upload byte count differs from region".into());
        }
        let staging = Staging::new(&self.vk, size, Some(bytes))?;
        let done = state.queue.context().completion()?;
        let State {
            image,
            queue,
            layout,
            ..
        } = &mut *state;
        let image = image.as_ref().expect("live image");
        queue.submit_commands(Arc::clone(&done), false, |vk, cmd| {
            transfer_barrier(vk, cmd, image, *layout, vk::ImageLayout::GENERAL);
            record_copy(
                vk,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                staging.handle(),
                copy,
                true,
            );
        })?;
        queue.keep_alive(&done, vec![staging]);
        state.layout = vk::ImageLayout::GENERAL;
        state.stats.uploads += 1;
        state.stats.upload_bytes += size as u64;
        state.last = Some(Arc::clone(&done));
        Ok(done)
    }
    pub fn read(&self, region: TextureRegion) -> Result<Readback, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (copy, size) =
            region.checked(state.image.as_ref().expect("live image").info, self.format)?;
        if state.layout == vk::ImageLayout::UNDEFINED {
            return Err("native texture has no defined GPU contents".into());
        }
        let staging = Staging::new(&self.vk, size, None)?;
        let done = state.queue.context().completion()?;
        let State {
            image,
            queue,
            layout,
            ..
        } = &mut *state;
        let image = image.as_ref().expect("live image");
        queue.submit_commands(Arc::clone(&done), true, |vk, cmd| {
            transfer_barrier(vk, cmd, image, *layout, vk::ImageLayout::GENERAL);
            record_copy(
                vk,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                staging.handle(),
                copy,
                false,
            );
        })?;
        queue.keep_alive(&done, vec![Arc::clone(&staging) as Arc<dyn Send + Sync>]);
        state.layout = vk::ImageLayout::GENERAL;
        state.stats.reads += 1;
        state.stats.readback_bytes += size as u64;
        state.last = Some(Arc::clone(&done));
        Ok(Readback { done, staging })
    }
}
impl Drop for Texture {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(done) = &state.last {
            let _ = done.wait(u64::MAX);
        }
        self.vk
            .destroy_image(state.image.take().expect("live image"));
    }
}

#[cfg(test)]
mod tests;
