//! Render targets: the images a draw writes, found again by the address of the memory they stand
//! for in the guest, and read back as pictures.

use super::frame::{Frame, Garbage};
use crate::device::Vk;
use crate::image::{aspect_of, Image, ImageInfo};
use ash::vk;
use std::collections::HashMap;

/// What identifies a render target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    /// The address in the guest process of the memory the target stands for.
    pub address: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

pub struct Target {
    pub key: Key,
    pub image: Image,
    pub layout: vk::ImageLayout,
    /// The submission counter when the target was last drawn to.
    pub touched: u64,
}

impl Target {
    pub fn aspect(&self) -> vk::ImageAspectFlags {
        aspect_of(self.key.format)
    }

    fn range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: self.key.layers,
        }
    }

    /// Fill a new target with what freshly mapped memory holds (zeros; 1.0 for depth), so that what
    /// a draw leaves uncovered is the same on every device. The target is left in the layout of
    /// the transfer.
    fn zero(&mut self, vk: &Vk, cmd: vk::CommandBuffer) {
        self.transition(vk, cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let range = self.range();
        // SAFETY: the command buffer is recording outside a rendering; the image is in the
        // transfer layout and has transfer usage.
        unsafe {
            if self.aspect() == vk::ImageAspectFlags::COLOR {
                vk.device.cmd_clear_color_image(
                    cmd,
                    self.image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { uint32: [0; 4] },
                    &[range],
                );
            } else {
                vk.device.cmd_clear_depth_stencil_image(
                    cmd,
                    self.image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                    &[range],
                );
            }
        }
    }

    /// Record the barrier that moves the image to `layout` (all earlier work finishes first and
    /// the image's contents are kept).
    pub fn transition(&mut self, vk: &Vk, cmd: vk::CommandBuffer, layout: vk::ImageLayout) {
        if self.layout == layout {
            return;
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(self.layout)
            .new_layout(layout)
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image.image)
            .subresource_range(self.range())];
        // SAFETY: the command buffer is recording and the image is valid.
        unsafe {
            vk.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        self.layout = layout;
    }
}

/// All the targets made so far.
#[derive(Default)]
pub struct Targets {
    map: HashMap<Key, Target>,
    /// How many submissions have run, to age targets by.
    pub submissions: u64,
}

impl Targets {
    /// The target for `key`, made if it is new (usage depends on the format's aspect).
    pub fn get(&mut self, vk: &Vk, frame: &mut Frame, key: Key) -> Result<&mut Target, String> {
        if !self.map.contains_key(&key) {
            // Another target at the same address with other dimensions stands for the same memory:
            // the old one is retired (what is on it is lost, which the title meant by changing
            // the shape of its buffer).
            let stale: Vec<Key> = self
                .map
                .keys()
                .filter(|k| k.address == key.address)
                .copied()
                .collect();
            for k in stale {
                if let Some(t) = self.map.remove(&k) {
                    frame.garbage.push(Garbage::Image(t.image));
                }
            }
            let depth = aspect_of(key.format) != vk::ImageAspectFlags::COLOR;
            let usage = if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            } | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST;
            let image = vk.create_image(ImageInfo {
                format: key.format,
                width: key.width,
                height: key.height,
                layers: key.layers,
                usage,
                view_type: if key.layers > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                },
                components: vk::ComponentMapping::default(),
                levels: 1,
            })?;
            let mut target = Target {
                key,
                image,
                layout: vk::ImageLayout::UNDEFINED,
                touched: 0,
            };
            match frame.cmd(vk) {
                Ok(cmd) => target.zero(vk, cmd),
                Err(e) => {
                    vk.destroy_image(target.image);
                    return Err(e);
                }
            }
            self.map.insert(key, target);
        }
        let submissions = self.submissions;
        let target = self.map.get_mut(&key).expect("just inserted");
        target.touched = submissions;
        Ok(target)
    }

    /// The color target standing for the guest memory at `address`, if one was drawn to.
    pub fn at_address(&mut self, address: u64) -> Option<&mut Target> {
        self.map
            .values_mut()
            .filter(|t| {
                t.key.address == address && aspect_of(t.key.format) == vk::ImageAspectFlags::COLOR
            })
            .max_by_key(|t| t.touched)
    }

    /// The keys of the color targets made so far.
    pub fn color_keys(&self) -> Vec<Key> {
        let mut keys: Vec<Key> = self
            .map
            .keys()
            .filter(|k| aspect_of(k.format) == vk::ImageAspectFlags::COLOR)
            .copied()
            .collect();
        keys.sort_by_key(|k| (k.address, k.width, k.height));
        keys
    }

    /// A target already made.
    pub fn existing(&mut self, key: &Key) -> Option<&mut Target> {
        self.map.get_mut(key)
    }

    /// The targets made so far, for messages.
    pub fn describe(&self) -> String {
        let mut list: Vec<String> = self
            .map
            .values()
            .map(|t| {
                format!(
                    "{:#x} {}x{} {:?}",
                    t.key.address, t.key.width, t.key.height, t.key.format
                )
            })
            .collect();
        list.sort();
        if list.is_empty() {
            "none".to_string()
        } else {
            list.join(", ")
        }
    }

    pub fn destroy(&mut self, vk: &Vk) {
        for (_, t) in self.map.drain() {
            vk.destroy_image(t.image);
        }
    }
}

/// Bytes per pixel of the formats a picture can be made of.
fn pixel_bytes(format: vk::Format) -> Option<u32> {
    Some(match format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::A2R10G10B10_UNORM_PACK32 => 4,
        vk::Format::R16G16B16A16_SFLOAT | vk::Format::R16G16B16A16_UNORM => 8,
        vk::Format::R5G6B5_UNORM_PACK16 | vk::Format::R8G8_UNORM => 2,
        vk::Format::R8_UNORM => 1,
        _ => return None,
    })
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from(h >> 10 & 0x1F);
    let frac = f32::from(h & 0x3FF);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => {
            if frac == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn to_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Convert the pixels of a target to RGBA bytes (floats are clamped, the 10-bit formats
/// rescaled). `None` for a format a picture cannot be made of.
pub fn to_rgba8(format: vk::Format, data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let bpp = pixel_bytes(format)? as usize;
    let n = width as usize * height as usize;
    if data.len() < n * bpp {
        return None;
    }
    let mut out = Vec::with_capacity(n * 4);
    for px in data[..n * bpp].chunks_exact(bpp) {
        let rgba: [u8; 4] = match format {
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => [px[0], px[1], px[2], px[3]],
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => [px[2], px[1], px[0], px[3]],
            vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32 => {
                let w = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
                let (lo, mid, hi) = (w & 0x3FF, w >> 10 & 0x3FF, w >> 20 & 0x3FF);
                let (r, b) = if format == vk::Format::A2B10G10R10_UNORM_PACK32 {
                    (lo, hi)
                } else {
                    (hi, lo)
                };
                let s = |v: u32| (v * 255 / 1023) as u8;
                [s(r), s(mid), s(b), (w >> 30) as u8 * 85]
            }
            vk::Format::R16G16B16A16_SFLOAT => {
                let c = |i: usize| half_to_f32(u16::from_le_bytes([px[2 * i], px[2 * i + 1]]));
                [to_byte(c(0)), to_byte(c(1)), to_byte(c(2)), to_byte(c(3))]
            }
            vk::Format::R16G16B16A16_UNORM => {
                let c = |i: usize| (u16::from_le_bytes([px[2 * i], px[2 * i + 1]]) >> 8) as u8;
                [c(0), c(1), c(2), c(3)]
            }
            vk::Format::R5G6B5_UNORM_PACK16 => {
                let w = u16::from_le_bytes([px[0], px[1]]);
                let (r, g, b) = (w >> 11 & 31, w >> 5 & 63, w & 31);
                [
                    (r * 255 / 31) as u8,
                    (g * 255 / 63) as u8,
                    (b * 255 / 31) as u8,
                    255,
                ]
            }
            vk::Format::R8G8_UNORM => [px[0], px[1], 0, 255],
            vk::Format::R8_UNORM => [px[0], px[0], px[0], 255],
            _ => return None,
        };
        out.extend_from_slice(&rgba);
    }
    Some(out)
}

/// The part of an RGBA picture inside `crop` (left, top, right, bottom), clamped to the picture;
/// the whole picture when the rectangle has no area.
pub fn crop_rgba(rgba: &[u8], width: u32, height: u32, crop: [i32; 4]) -> (Vec<u8>, u32, u32) {
    let clamp = |v: i32, max: u32| v.clamp(0, max as i32) as u32;
    let (left, right) = (clamp(crop[0], width), clamp(crop[2], width));
    let (top, bottom) = (clamp(crop[1], height), clamp(crop[3], height));
    if left >= right || top >= bottom || rgba.len() < width as usize * height as usize * 4 {
        return (rgba.to_vec(), width, height);
    }
    let mut out = Vec::with_capacity((right - left) as usize * (bottom - top) as usize * 4);
    for y in top..bottom {
        let row = y as usize * width as usize * 4;
        out.extend_from_slice(&rgba[row + left as usize * 4..row + right as usize * 4]);
    }
    (out, right - left, bottom - top)
}

/// Read a color target back as RGBA bytes: its contents as of the commands recorded so far (they
/// are run first).
pub fn read_rgba(vk: &Vk, frame: &mut Frame, target: &mut Target) -> Result<Vec<u8>, String> {
    let key = target.key;
    let Some(bpp) = pixel_bytes(key.format) else {
        return Err(format!("a picture cannot be made of {:?}", key.format));
    };
    let size = u64::from(key.width) * u64::from(key.height) * u64::from(bpp);
    let staging = vk.create_host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
    let cmd = frame.cmd(vk)?;
    target.transition(vk, cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    let region = [vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width: key.width,
            height: key.height,
            depth: 1,
        })];
    // SAFETY: the command buffer is recording; the image is in the transfer layout.
    unsafe {
        vk.device.cmd_copy_image_to_buffer(
            cmd,
            target.image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging.buffer,
            &region,
        );
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        vk.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }
    frame.submit(vk)?;
    let rgba = to_rgba8(key.format, staging.as_slice(), key.width, key.height);
    vk.destroy_host_buffer(staging);
    rgba.ok_or_else(|| format!("a picture cannot be made of {:?}", key.format))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_convert() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert!(half_to_f32(0x7E00).is_nan());
        assert_eq!(half_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn a_crop_keeps_the_rectangle_and_an_empty_one_keeps_everything() {
        // 4x3 pixels whose red channel is x + 10 y.
        let rgba: Vec<u8> = (0..3u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [x + 10 * y, 0, 0, 255]))
            .collect();
        let (out, w, h) = crop_rgba(&rgba, 4, 3, [1, 1, 3, 3]);
        assert_eq!((w, h), (2, 2));
        let reds: Vec<u8> = out.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        assert_eq!(reds, vec![11, 12, 21, 22]);
        // Clamped to the picture.
        let (_, w, h) = crop_rgba(&rgba, 4, 3, [-5, 0, 100, 2]);
        assert_eq!((w, h), (4, 2));
        // No area, or all zero: the whole picture.
        for crop in [[0; 4], [2, 2, 2, 2], [3, 0, 1, 3]] {
            let (out, w, h) = crop_rgba(&rgba, 4, 3, crop);
            assert_eq!((w, h, out.len()), (4, 3, rgba.len()));
        }
    }

    #[test]
    fn pixels_are_made_into_rgba() {
        let bgra = to_rgba8(vk::Format::B8G8R8A8_UNORM, &[1, 2, 3, 4], 1, 1).unwrap();
        assert_eq!(bgra, vec![3, 2, 1, 4]);
        let rgba = to_rgba8(vk::Format::R8G8B8A8_UNORM, &[1, 2, 3, 4], 1, 1).unwrap();
        assert_eq!(rgba, vec![1, 2, 3, 4]);
        let half = [0x00, 0x3C, 0x00, 0x38, 0x00, 0x00, 0x00, 0x3C];
        let p = to_rgba8(vk::Format::R16G16B16A16_SFLOAT, &half, 1, 1).unwrap();
        assert_eq!(p, vec![255, 128, 0, 255]);
        let w = 0x3FFu32 << 20 | 3 << 30; // blue and alpha full in A2B10G10R10
        let p = to_rgba8(vk::Format::A2B10G10R10_UNORM_PACK32, &w.to_le_bytes(), 1, 1).unwrap();
        assert_eq!(p, vec![0, 0, 255, 255]);
        assert!(to_rgba8(vk::Format::D32_SFLOAT, &[0; 4], 1, 1).is_none());
        assert!(to_rgba8(vk::Format::R8G8B8A8_UNORM, &[0; 3], 1, 1).is_none());
    }
}
