//! Surfaces: the images compute programs load and store texels of (`SULD`, `SUST`).
//!
//! A surface is made from its memory for the launch that uses it: a storage image of level 0 of
//! the texture its header describes (the handle's low 20 bits are the header's number in the
//! texture pool, as for a texture: NVN registers image descriptors in the texture pool). What
//! the launch stores is put back into that memory after it (the launch waits for the device in
//! any case, A-110), so that whatever reads the memory later finds it there. The images are not
//! kept from one launch to the next (docs/ASSUMPTIONS.md A-252).

use super::textures::{plan, texture_format, Kind, Shape, Tic};
use super::Core;
use crate::blocklinear::{deswizzle, swizzle, Geometry};
use crate::image::{aspect_of, Image, ImageInfo};
use ash::vk;
use nx_gpu::render::Scene;
use nx_shader::translate::{Dim, ImageUse};

/// A surface made for a launch.
pub struct Surface {
    image: Image,
    /// The GPU address of the texture's memory and the bytes from one layer to the next.
    address: u64,
    layer_stride: u64,
    layers: u32,
    /// Level 0 of a layer as it lies in memory (block linear), or its rows (pitch linear).
    geometry: Geometry,
    pitch: Option<u32>,
    written: bool,
}

impl Surface {
    pub fn view(&self) -> vk::ImageView {
        self.image.view
    }

    /// The bytes of a layer's level 0 in memory.
    fn layer_bytes(&self) -> u64 {
        match self.pitch {
            Some(pitch) => u64::from(pitch) * u64::from(self.geometry.height),
            None => self.geometry.tiled_size(),
        }
    }

    fn region(&self, layer: u32, offset: u64) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: layer,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: self.image.info.width,
                height: self.image.info.height,
                depth: self.image.info.depth,
            })
    }

    fn range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: self.layers,
        }
    }

    /// Lay the packed rows of a layer out as memory has them, over what memory held (`tiled`):
    /// the bytes that are not the texels' stay as they were.
    fn lay_out(&self, linear: &[u8], tiled: &mut [u8]) {
        match self.pitch {
            Some(pitch) => {
                let row = (self.geometry.width * self.geometry.bytes_per_element) as usize;
                for y in 0..self.geometry.height as usize {
                    let at = y * pitch as usize;
                    if at + row <= tiled.len() {
                        tiled[at..at + row].copy_from_slice(&linear[y * row..(y + 1) * row]);
                    }
                }
            }
            None => swizzle(&self.geometry, linear, tiled),
        }
    }
}

fn barrier(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    (from, to): (vk::ImageLayout, vk::ImageLayout),
    (src, dst): (vk::AccessFlags, vk::AccessFlags),
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
}

impl Core {
    /// The surface of header `tic` a program uses as `used`, made from its memory and in the
    /// layout a launch accesses it in.
    pub(super) fn surface(
        &mut self,
        scene: &Scene<'_>,
        tic: Tic,
        used: &ImageUse,
    ) -> Result<Surface, String> {
        let describe = || format!("a surface of header {:08x?}", tic.0);
        let format = texture_format(tic.components(), tic.types()[0], false)
            .ok_or_else(|| format!("{}: its format {:#x}", describe(), tic.components()))?;
        if format.astc
            || format.block != (1, 1)
            || aspect_of(format.vk) != vk::ImageAspectFlags::COLOR
        {
            return Err(format!("{} in the format {:?}", describe(), format.vk));
        }
        // The program reads and writes floats (A-252): an integer format is not given to it. An
        // atomic operation (`SURED`) is on 32-bit words, which an image of unsigned integers has
        // the bits of whatever the header calls them.
        let format = if used.atomic {
            if format.bytes != 4 || format.block != (1, 1) {
                return Err(format!(
                    "{} for atomic operations in the format {:?}",
                    describe(),
                    format.vk
                ));
            }
            super::textures::TexFormat {
                vk: vk::Format::R32_UINT,
                ..format
            }
        } else if matches!(tic.types()[0], 3 | 4) {
            return Err(format!("{} of an integer format", describe()));
        } else {
            format
        };
        let features = if used.atomic {
            vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC
        } else {
            vk::FormatFeatureFlags::STORAGE_IMAGE
        };
        if !self.vk.format_supports(format.vk, features) {
            return Err(format!(
                "{}: the device has no storage images of {:?}",
                describe(),
                format.vk
            ));
        }
        let (kind, layers, depth, view_type) = match (used.dim, used.arrayed) {
            (Dim::D1, false) => (Kind::D1, 1, 1, vk::ImageViewType::TYPE_1D),
            (Dim::D2, false) => (Kind::D2, 1, 1, vk::ImageViewType::TYPE_2D),
            (Dim::D2, true) => (
                Kind::D2Array,
                tic.depth(),
                1,
                vk::ImageViewType::TYPE_2D_ARRAY,
            ),
            (Dim::D3, false) => (Kind::D3, 1, tic.depth(), vk::ImageViewType::TYPE_3D),
            (dim, arrayed) => {
                return Err(format!(
                    "{} used as a {dim:?}{} image",
                    describe(),
                    if arrayed { " array" } else { "" }
                ))
            }
        };
        let p = plan(
            &tic,
            Shape {
                kind,
                layers,
                depth,
            },
            format,
        )?;
        let image = self.vk.create_image(ImageInfo {
            format: format.vk,
            width: tic.width(),
            height: tic.height(),
            depth,
            layers,
            levels: 1,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            view_type,
            components: vk::ComponentMapping::default(),
        })?;
        let surface = Surface {
            image,
            address: tic.address(),
            layer_stride: p.layer_stride,
            layers,
            geometry: p.level_geometry[0],
            pitch: tic.is_pitch().then(|| tic.pitch()),
            written: used.written,
        };
        // What memory holds, layer by layer.
        let mut packed = Vec::new();
        let mut regions = Vec::new();
        for layer in 0..layers {
            let at = surface.address + u64::from(layer) * surface.layer_stride;
            let mut held = vec![0u8; surface.layer_bytes() as usize];
            let n = super::textures::read_partial(scene, at, &mut held);
            held.truncate(n);
            let g = surface.geometry;
            let mut linear = vec![0u8; g.linear_size() as usize];
            match surface.pitch {
                Some(pitch) => {
                    let row = (g.width * g.bytes_per_element) as usize;
                    for y in 0..g.height as usize {
                        let from = y * pitch as usize;
                        if from + row <= held.len() {
                            linear[y * row..(y + 1) * row].copy_from_slice(&held[from..from + row]);
                        }
                    }
                }
                None => deswizzle(&g, &held, &mut linear),
            }
            let offset = (packed.len() as u64).next_multiple_of(16);
            packed.resize(offset as usize, 0);
            packed.extend_from_slice(&linear);
            regions.push(surface.region(layer, offset));
        }
        let (buffer, base) = self.frame.upload(&self.vk, &packed, 16)?;
        for r in &mut regions {
            r.buffer_offset += base;
        }
        let cmd = self.frame.upload_cmd(&self.vk)?;
        let range = surface.range();
        let img = surface.image.image;
        // SAFETY: the upload command buffer is recording; the image was just made.
        unsafe {
            let d = &self.vk.device;
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    img,
                    range,
                    (
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                    (vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE),
                )],
            );
            d.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    img,
                    range,
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                    ),
                    (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    ),
                )],
            );
        }
        Ok(surface)
    }

    /// After a launch has run: put what it stored in its surfaces into their memory, let go the
    /// textures and render targets that memory was taken into, and let the images go.
    pub(super) fn put_back(
        &mut self,
        scene: &Scene<'_>,
        surfaces: Vec<Surface>,
    ) -> Result<(), String> {
        let written: Vec<&Surface> = surfaces.iter().filter(|s| s.written).collect();
        let mut staged = Vec::new();
        if !written.is_empty() {
            let cmd = self.frame.cmd(&self.vk)?;
            for s in &written {
                let size = s.geometry.linear_size() * u64::from(s.layers);
                let staging = self
                    .vk
                    .create_host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
                let per_layer = s.geometry.linear_size();
                let regions: Vec<vk::BufferImageCopy> = (0..s.layers)
                    .map(|layer| s.region(layer, u64::from(layer) * per_layer))
                    .collect();
                // SAFETY: the command buffer is recording outside a rendering; the image is in
                // the general layout, written by the launch before.
                unsafe {
                    let d = &self.vk.device;
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier(
                            s.image.image,
                            s.range(),
                            (
                                vk::ImageLayout::GENERAL,
                                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            ),
                            (
                                vk::AccessFlags::SHADER_WRITE,
                                vk::AccessFlags::TRANSFER_READ,
                            ),
                        )],
                    );
                    d.cmd_copy_image_to_buffer(
                        cmd,
                        s.image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        staging.buffer,
                        &regions,
                    );
                }
                staged.push(staging);
            }
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            // SAFETY: the command buffer is recording.
            unsafe {
                self.vk.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &to_host,
                    &[],
                    &[],
                );
            }
        }
        let submitted = if staged.is_empty() {
            Ok(())
        } else {
            self.frame.submit(&self.vk)
        };
        for (s, staging) in written.iter().zip(&staged) {
            if submitted.is_err() {
                break;
            }
            let per_layer = s.geometry.linear_size() as usize;
            for layer in 0..s.layers as usize {
                let linear = &staging.as_slice()[layer * per_layer..(layer + 1) * per_layer];
                let at = s.address + layer as u64 * s.layer_stride;
                let mut tiled = vec![0u8; s.layer_bytes() as usize];
                let n = super::textures::read_partial(scene, at, &mut tiled);
                tiled.truncate(n);
                s.lay_out(linear, &mut tiled);
                if !scene.space.write(scene.mem, at, &tiled) {
                    return Err(format!(
                        "a surface's memory at GPU address {at:#x} is not mapped"
                    ));
                }
                if let Some((cpu, _)) = scene.host_address(at) {
                    let end = cpu + tiled.len() as u64;
                    self.textures.memory_written(cpu, end);
                    let stale: Vec<_> = self
                        .targets
                        .covering(cpu, end)
                        .into_iter()
                        .map(|c| c.key)
                        .collect();
                    if !stale.is_empty() {
                        self.targets.release(&stale, &mut self.frame);
                    }
                }
            }
        }
        for staging in staged {
            self.vk.destroy_host_buffer(staging);
        }
        // The launch has run: nothing uses the images any more once the batch is over.
        for s in surfaces {
            self.frame
                .garbage
                .push(super::frame::Garbage::Image(s.image));
        }
        submitted
    }
}
