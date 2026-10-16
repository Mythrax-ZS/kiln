//! Copies made with the 2D engine: a title copies a render target it drew (a lookup table, a face
//! of a cube map, a resolved picture) into the memory of a texture, with scaling and a change of
//! format where it says so (docs/ASSUMPTIONS.md A-165).
//!
//! The copy is made on the device, from the image of the source target into an image of the
//! destination: that image is a render target at the destination's address, the picture of that
//! memory (what a draw into it would leave), and a texture that is sampled from it finds the
//! target there and reads it (`render_target_view`). The memory itself is written when something
//! needs it there, as for any other target.

use super::targets::{Key, Tiling};
use super::Core;
use crate::formats::color_format;
use ash::vk;
use nx_gpu::draw_state::Layout;
use nx_gpu::render::{Blit, Scene};

/// 32.32 fixed point to a number.
fn real(fixed: i64) -> f64 {
    fixed as f64 / 4_294_967_296.0
}

impl Core {
    /// Make a copy of the 2D engine.
    pub(super) fn copy_surface(&mut self, scene: &Scene<'_>, b: &Blit) -> Result<(), String> {
        let describe = || format!("{b:?}");
        let Some(src_format) = color_format(b.src.format) else {
            return Err(format!(
                "a 2D copy from a surface of format {:#x}: {}",
                b.src.format,
                describe()
            ));
        };
        let Some(dst_format) = color_format(b.dst.format) else {
            return Err(format!(
                "a 2D copy into a surface of format {:#x}: {}",
                b.dst.format,
                describe()
            ));
        };
        if !matches!(b.operation, 3 | 4) && b.operation != 0 {
            return Err(format!(
                "a 2D copy with the operation {} (only copies are made): {}",
                b.operation,
                describe()
            ));
        }
        if b.clip.is_some() || b.src.depth > 1 || b.dst.depth > 1 || b.dst.layer != 0 {
            return Err(format!(
                "a 2D copy with a clip rectangle or a layered surface: {}",
                describe()
            ));
        }
        // The rectangle of the destination must be all of it: a part of a surface would leave the
        // rest to what its memory holds, which the target made for it does not know.
        if b.dst_x != 0
            || b.dst_y != 0
            || b.dst_width != b.dst.width
            || b.dst_height != b.dst.height
            || b.dst_width == 0
            || b.dst_height == 0
        {
            return Err(format!(
                "a 2D copy into a part of a surface is not implemented: {}",
                describe()
            ));
        }
        let Some((src_cpu, _)) = scene.host_address(b.src.address) else {
            return Err(format!(
                "a 2D copy from GPU address {:#x}, which is not mapped",
                b.src.address
            ));
        };
        let Some((dst_cpu, _)) = scene.host_address(b.dst.address) else {
            return Err(format!(
                "a 2D copy to GPU address {:#x}, which is not mapped",
                b.dst.address
            ));
        };
        let Some(src_key) = self.targets.at_address(src_cpu).map(|t| t.key) else {
            return Err(format!(
                "a 2D copy from memory that no render target stands for is not implemented: {}",
                describe()
            ));
        };
        if src_key.format != src_format.vk || src_key.layers != 1 {
            return Err(format!(
                "a 2D copy from a surface of format {:?} where the render target is {src_key:?}: {}",
                src_format.vk,
                describe()
            ));
        }
        // The part of the source the destination's pixels cover: pixel i of the destination takes
        // the source at `src + i * step`, where the origin of coordinates is the corner of the
        // first pixel (or its center) and the pixel centers are half a pixel in.
        let shift = if b.corner_origin { 0.0 } else { 0.5 };
        let (step_x, step_y) = (real(b.du_dx), real(b.dv_dy));
        let left = real(b.src_x) + shift - step_x / 2.0;
        let top = real(b.src_y) + shift - step_y / 2.0;
        let right = left + step_x * f64::from(b.dst_width);
        let bottom = top + step_y * f64::from(b.dst_height);
        let (x0, x1) = (left.round() as i64, right.round() as i64);
        let (y0, y1) = (top.round() as i64, bottom.round() as i64);
        let (width, height) = (i64::from(src_key.width), i64::from(src_key.height));
        if x0.min(x1) < 0 || y0.min(y1) < 0 || x0.max(x1) > width || y0.max(y1) > height {
            return Err(format!(
                "a 2D copy of [{left}, {right}] x [{top}, {bottom}] of a {width}x{height} target: {}",
                describe()
            ));
        }
        if x0 == x1 || y0 == y1 {
            return Ok(());
        }
        let dst_key = Key {
            address: dst_cpu,
            format: dst_format.vk,
            width: b.dst.width,
            height: b.dst.height,
            layers: 1,
            tiling: match b.dst.layout {
                Layout::Pitch => Tiling::Pitch(b.dst.pitch),
                Layout::BlockLinear { block } => Tiling::Block(block[1]),
            },
        };
        if dst_key == src_key {
            return Ok(());
        }
        // What was recorded so far must come first, and a rendering in progress ends.
        self.end_pass();
        let cmd = self.frame.cmd(&self.vk)?;
        let src_image = {
            let src = self
                .targets
                .existing(&src_key)
                .ok_or_else(|| format!("the render target of a 2D copy is gone: {src_key:?}"))?;
            src.transition(&self.vk, cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            src.image.image
        };
        let dst = self.targets.get(&self.vk, &mut self.frame, dst_key)?;
        dst.transition(&self.vk, cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = [vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [
                vk::Offset3D {
                    x: x0 as i32,
                    y: y0 as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: x1 as i32,
                    y: y1 as i32,
                    z: 1,
                },
            ],
            dst_subresource: layers,
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: b.dst.width as i32,
                    y: b.dst.height as i32,
                    z: 1,
                },
            ],
        }];
        let filter = if b.bilinear {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        // SAFETY: the command buffer is recording outside a rendering; both images are in the
        // transfer layouts and have transfer usage.
        unsafe {
            self.vk.device.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
                filter,
            );
        }
        // What textures made from that memory held is not what the memory is now.
        if let Some(bytes) = dst_key.memory_bytes() {
            self.textures.memory_written(dst_cpu, dst_cpu + bytes);
        }
        Ok(())
    }
}
