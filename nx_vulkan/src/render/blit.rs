//! Showing a render target: the part of it that is displayed, scaled into a picture of another
//! size with black bars to keep its shape.

use super::targets::Target;
use crate::device::Vk;
use crate::image::{Image, ImageInfo};
use ash::vk;

/// The displayed part of a target of `size` as `[left, top, right, bottom]`: the producer's `crop`
/// clamped to the target, or all of the target when the crop has no area.
pub fn visible(crop: [i32; 4], size: (u32, u32)) -> [i32; 4] {
    let clamp = |v: i32, max: u32| v.clamp(0, max as i32);
    let (left, right) = (clamp(crop[0], size.0), clamp(crop[2], size.0));
    let (top, bottom) = (clamp(crop[1], size.1), clamp(crop[3], size.1));
    if left >= right || top >= bottom {
        [0, 0, size.0 as i32, size.1 as i32]
    } else {
        [left, top, right, bottom]
    }
}

/// The largest rectangle with the shape of `src` (width, height) that fits in `dst`, centered, as
/// `[left, top, right, bottom]`.
pub fn fit(src: (u32, u32), dst: (u32, u32)) -> [i32; 4] {
    let (sw, sh) = (u64::from(src.0.max(1)), u64::from(src.1.max(1)));
    let (dw, dh) = (u64::from(dst.0), u64::from(dst.1));
    // Compare sw / sh with dw / dh without dividing.
    let (w, h) = if sw * dh >= sh * dw {
        (dw, (dw * sh / sw).max(1).min(dh))
    } else {
        ((dh * sw / sh).max(1).min(dw), dh)
    };
    let (x, y) = ((dw - w) / 2, (dh - h) / 2);
    [x as i32, y as i32, (x + w) as i32, (y + h) as i32]
}

fn layout_barrier(
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: (vk::PipelineStageFlags, vk::AccessFlags),
    dst: (vk::PipelineStageFlags, vk::AccessFlags),
    vk: &Vk,
    cmd: vk::CommandBuffer,
) {
    let barrier = [vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src.1)
        .dst_access_mask(dst.1)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })];
    // SAFETY: the command buffer is recording and the image is valid.
    unsafe {
        vk.device.cmd_pipeline_barrier(
            cmd,
            src.0,
            dst.0,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }
}

/// Record into `cmd` the showing of `target` in `dst`, an image of `dst_size` whose contents are
/// discarded: `dst` is cleared to black and the displayed part of the target (see [`visible`]) is
/// blitted into the largest centered rectangle of it that has its shape. When that part is not the
/// whole target it is first copied into `scratch` (made, or made again, to fit), so that the
/// filtering of the blit does not reach what the crop leaves out. Afterwards `dst` is in the layout
/// for transfer writes (the caller moves it on) and the target in the one for transfer reads (the
/// next rendering to it moves it back). `scratch` and `dst` must not be in use by earlier commands.
pub fn record(
    vk: &Vk,
    cmd: vk::CommandBuffer,
    target: &mut Target,
    crop: [i32; 4],
    scratch: &mut Option<Image>,
    dst: vk::Image,
    dst_size: (u32, u32),
) {
    target.transition(vk, cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    let transfer_write = (
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_WRITE,
    );
    let transfer_read = (
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_READ,
    );
    layout_barrier(
        dst,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::empty()),
        transfer_write,
        vk,
        cmd,
    );
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let layers = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let offset = |x: i32, y: i32| vk::Offset3D { x, y, z: 0 };
    // The corners of a blit region: the second one is one slice deep for a 2D image.
    let corner = |x: i32, y: i32| vk::Offset3D { x, y, z: 1 };
    let whole = [0, 0, target.key.width as i32, target.key.height as i32];
    let shown = visible(crop, (target.key.width, target.key.height));
    let src_size = ((shown[2] - shown[0]) as u32, (shown[3] - shown[1]) as u32);
    // What is blitted from: the target itself, or a copy of the part of it that is shown.
    let (source, source_rect) = if shown == whole {
        (target.image.image, shown)
    } else {
        let fits = scratch.as_ref().is_some_and(|s| {
            s.info.format == target.key.format && (s.info.width, s.info.height) == src_size
        });
        if !fits {
            if let Some(old) = scratch.take() {
                vk.destroy_image(old);
            }
            *scratch = vk
                .create_image(ImageInfo {
                    format: target.key.format,
                    width: src_size.0,
                    height: src_size.1,
                    layers: 1,
                    levels: 1,
                    usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                    view_type: vk::ImageViewType::TYPE_2D,
                    components: vk::ComponentMapping::default(),
                })
                .ok();
        }
        match scratch.as_ref() {
            Some(s) => {
                layout_barrier(
                    s.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::empty()),
                    transfer_write,
                    vk,
                    cmd,
                );
                let copy = [vk::ImageCopy {
                    src_subresource: layers,
                    src_offset: offset(shown[0], shown[1]),
                    dst_subresource: layers,
                    dst_offset: offset(0, 0),
                    extent: vk::Extent3D {
                        width: src_size.0,
                        height: src_size.1,
                        depth: 1,
                    },
                }];
                // SAFETY: recording; the images are in the layouts named.
                unsafe {
                    vk.device.cmd_copy_image(
                        cmd,
                        target.image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        s.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &copy,
                    );
                }
                layout_barrier(
                    s.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    transfer_write,
                    transfer_read,
                    vk,
                    cmd,
                );
                (s.image, [0, 0, src_size.0 as i32, src_size.1 as i32])
            }
            // No memory for a copy: the blit goes from the target and may blur the edge.
            None => (target.image.image, shown),
        }
    };
    let into = fit(src_size, dst_size);
    let linear = vk.format_supports(
        target.key.format,
        vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR | vk::FormatFeatureFlags::BLIT_SRC,
    );
    let region = [vk::ImageBlit {
        src_subresource: layers,
        src_offsets: [
            offset(source_rect[0], source_rect[1]),
            corner(source_rect[2], source_rect[3]),
        ],
        dst_subresource: layers,
        dst_offsets: [offset(into[0], into[1]), corner(into[2], into[3])],
    }];
    // SAFETY: the command buffer is recording; both images are in the layouts named.
    unsafe {
        vk.device.cmd_clear_color_image(
            cmd,
            dst,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            &[range],
        );
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
        vk.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
        vk.device.cmd_blit_image(
            cmd,
            source,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
            if linear {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            },
        );
    }
}

/// Move `image` from the layout `record` leaves it in to the one a swapchain presents from.
pub fn to_present(vk: &Vk, cmd: vk::CommandBuffer, image: vk::Image) {
    layout_barrier(
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
        (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        (
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::empty(),
        ),
        vk,
        cmd,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::frame::Frame;
    use crate::render::targets::{Key, Targets};

    #[test]
    fn the_visible_part_is_the_crop_or_everything() {
        assert_eq!(visible([0, 0, 1280, 720], (1600, 900)), [0, 0, 1280, 720]);
        assert_eq!(visible([0; 4], (1600, 900)), [0, 0, 1600, 900]);
        assert_eq!(visible([-5, -5, 5000, 5000], (64, 32)), [0, 0, 64, 32]);
        assert_eq!(visible([10, 20, 10, 40], (64, 64)), [0, 0, 64, 64]);
        assert_eq!(visible([8, 4, 40, 20], (64, 64)), [8, 4, 40, 20]);
    }

    #[test]
    fn a_picture_is_fitted_with_bars_where_the_shapes_differ() {
        // The same shape fills the window.
        assert_eq!(fit((1280, 720), (640, 360)), [0, 0, 640, 360]);
        // A wider window: bars left and right.
        assert_eq!(fit((1280, 720), (1000, 360)), [180, 0, 820, 360]);
        // A taller window: bars above and below.
        assert_eq!(fit((1280, 720), (640, 480)), [0, 60, 640, 420]);
        // A square picture in a wide window.
        assert_eq!(fit((100, 100), (300, 100)), [100, 0, 200, 100]);
        // Nothing to show or nowhere to show it stays valid.
        let r = fit((0, 0), (10, 10));
        assert!(r[2] > r[0] && r[3] > r[1]);
    }

    /// Read a `size` image in the transfer destination layout back as RGBA bytes.
    fn read_back(
        vk: &Vk,
        image: vk::Image,
        size: (u32, u32),
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Vec<u8> {
        let buffer = vk
            .create_host_buffer(
                u64::from(size.0) * u64::from(size.1) * 4,
                vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
        vk.one_shot(|d, cmd| {
            record(cmd);
            layout_barrier(
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_WRITE,
                ),
                (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                vk,
                cmd,
            );
            let region = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: size.0,
                    height: size.1,
                    depth: 1,
                })];
            // SAFETY: recording; the image is in the transfer source layout.
            unsafe {
                d.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer.buffer,
                    &region,
                );
            }
        })
        .unwrap();
        let bytes = buffer.as_slice().to_vec();
        vk.destroy_host_buffer(buffer);
        bytes
    }

    #[test]
    fn a_target_is_shown_cropped_and_fitted_in_another_image() {
        let Ok(vk) = Vk::new() else {
            eprintln!("SKIPPED: no Vulkan device");
            return;
        };
        let mut frame = Frame::new(&vk).unwrap();
        let mut targets = Targets::default();
        // A 16x8 target: the left half red, the right half green, and a 2-pixel blue column on
        // the far right that the crop leaves out.
        let key = Key {
            address: 0x1000,
            format: vk::Format::R8G8B8A8_UNORM,
            width: 16,
            height: 8,
            layers: 1,
        };
        targets.get(&vk, &mut frame, key).unwrap();
        // The zero fill of the new target runs before its texels are put there.
        frame.submit(&vk).unwrap();
        let target = targets.existing(&key).unwrap();
        let mut texels = Vec::new();
        for _y in 0..8 {
            for x in 0..16 {
                let c: [u8; 4] = if x >= 14 {
                    [0, 0, 255, 255]
                } else if x < 7 {
                    [255, 0, 0, 255]
                } else {
                    [0, 255, 0, 255]
                };
                texels.extend_from_slice(&c);
            }
        }
        vk.upload_image(&target.image, &texels).unwrap();
        target.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        // Shown in a 28 x 16 image: the crop (14 x 8) is twice as large, with no bars; then in
        // 28 x 24, where bars of 4 rows go above and below.
        let mut scratch: Option<Image> = None;
        for (size, bars) in [((28u32, 16u32), 0u32), ((28, 24), 4)] {
            let dst: Image = vk
                .create_image(ImageInfo {
                    format: vk::Format::R8G8B8A8_UNORM,
                    width: size.0,
                    height: size.1,
                    layers: 1,
                    levels: 1,
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
                    view_type: vk::ImageViewType::TYPE_2D,
                    components: vk::ComponentMapping::default(),
                })
                .unwrap();
            let target = targets.existing(&key).unwrap();
            let rgba = read_back(&vk, dst.image, size, |cmd| {
                record(
                    &vk,
                    cmd,
                    target,
                    [0, 0, 14, 8],
                    &mut scratch,
                    dst.image,
                    size,
                );
            });
            let at = |x: u32, y: u32| -> [u8; 4] {
                let i = ((y * size.0 + x) * 4) as usize;
                [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
            };
            assert_eq!(at(2, bars + 2), [255, 0, 0, 255], "red on the left");
            assert_eq!(at(25, bars + 3), [0, 255, 0, 255], "green on the right");
            assert_eq!(
                at(27, size.1 - bars - 1),
                [0, 255, 0, 255],
                "the crop cuts the blue off"
            );
            if bars > 0 {
                assert_eq!(at(10, 1), [0, 0, 0, 255], "a bar above");
                assert_eq!(at(10, size.1 - 2), [0, 0, 0, 255], "a bar below");
            }
            vk.destroy_image(dst);
        }
        if let Some(s) = scratch.take() {
            vk.destroy_image(s);
        }
        targets.destroy(&vk);
        frame.destroy(&vk);
    }
}
