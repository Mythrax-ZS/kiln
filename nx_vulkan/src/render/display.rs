//! Showing displayed buffers in a window: a swapchain, and the blit of the buffer's crop into the
//! image it hands out (`blit.rs`).
//!
//! The present mode is mailbox (the newest frame at each refresh of the screen) unless the surface
//! has none or `NX_PRESENT_MODE` says `fifo` or `immediate`; the guest is paced to 60 frames a
//! second by the display service (A-181), not by the window.

use super::blit;
use super::targets::Target;
use crate::device::Vk;
use crate::image::Image;
use crate::window::Window;
use ash::{khr, vk};

struct Swapchain {
    handle: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    /// One per image: the present waits for it, and may do so after the next frame has begun.
    release: Vec<vk::Semaphore>,
}

pub struct Display {
    window: Window,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: Option<Swapchain>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    acquire: vk::Semaphore,
    mode: vk::PresentModeKHR,
    /// The copy of the shown part of a buffer that is larger than it (see `blit::record`).
    scratch: Option<Image>,
    /// Whether the blit of the last frame is still running (the fence is not yet waited for).
    busy: bool,
}

fn is_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

impl Display {
    /// Make a surface of `window` and everything frames are shown with.
    pub fn new(vk: &Vk, window: Window) -> Result<Display, String> {
        let (display, handle) = window.raw_handles();
        // SAFETY: the handles belong to a window that lives as long as the process; the instance
        // was made with the surface extensions of this platform.
        let surface =
            unsafe { ash_window::create_surface(&vk.entry, &vk.instance, display, handle, None) }
                .map_err(|e| format!("making a surface for the window: {e}"))?;
        let surface_loader = khr::surface::Instance::new(&vk.entry, &vk.instance);
        // SAFETY: valid handles from this device.
        let can_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                vk.physical,
                vk.queue_family,
                surface,
            )
        }
        .map_err(|e| format!("asking whether the queue can present: {e}"))?;
        if !can_present || !vk.caps.swapchain {
            // SAFETY: nothing else uses the surface.
            unsafe { surface_loader.destroy_surface(surface, None) };
            return Err(format!(
                "{} cannot present to the window (swapchain extension: {}, queue family can present: {})",
                vk.caps.name, vk.caps.swapchain, can_present
            ));
        }
        let swapchain_loader = khr::swapchain::Device::new(&vk.instance, &vk.device);
        // Mailbox by default: each refresh of the screen shows the newest frame, and frames that
        // come close together are never played back one per refresh as a FIFO queue plays them
        // (which, on a screen that does not refresh 60 times a second, shows as stutter). FIFO
        // when the surface has no mailbox, or when asked for.
        let mode = match std::env::var("NX_PRESENT_MODE").as_deref() {
            Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
            Ok("fifo") => vk::PresentModeKHR::FIFO,
            _ => vk::PresentModeKHR::MAILBOX,
        };
        // SAFETY: plain object creation; destroyed in `destroy`.
        unsafe {
            let pool = vk
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(vk.queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("creating the display's command pool: {e}"))?;
            let cmd = vk
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| format!("allocating the display's command buffer: {e}"))?[0];
            let fence = vk
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("creating the display's fence: {e}"))?;
            let acquire = vk
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|e| format!("creating a semaphore: {e}"))?;
            Ok(Display {
                window,
                surface_loader,
                swapchain_loader,
                surface,
                swapchain: None,
                pool,
                cmd,
                fence,
                acquire,
                mode,
                scratch: None,
                busy: false,
            })
        }
    }

    /// Make the swapchain again (or for the first time) for a window of `size`, in an sRGB format
    /// when the buffers shown are, so that the blit does not change what the values mean.
    fn make_swapchain(&mut self, vk: &Vk, size: (u32, u32), srgb: bool) -> Result<(), String> {
        // SAFETY: valid handles; the old swapchain is retired through `old_swapchain` and
        // destroyed after the device is idle.
        unsafe {
            let caps = self
                .surface_loader
                .get_physical_device_surface_capabilities(vk.physical, self.surface)
                .map_err(|e| format!("asking for the surface capabilities: {e}"))?;
            let formats = self
                .surface_loader
                .get_physical_device_surface_formats(vk.physical, self.surface)
                .map_err(|e| format!("asking for the surface formats: {e}"))?;
            let wanted = if srgb {
                [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB]
            } else {
                [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM]
            };
            let chosen = wanted
                .iter()
                .find_map(|w| formats.iter().find(|f| f.format == *w))
                .or_else(|| formats.iter().find(|f| is_srgb(f.format) == srgb))
                .or(formats.first())
                .ok_or("the surface has no formats")?;
            let extent = if caps.current_extent.width == u32::MAX {
                vk::Extent2D {
                    width: size
                        .0
                        .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: size
                        .1
                        .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            } else {
                caps.current_extent
            };
            if extent.width == 0 || extent.height == 0 {
                return Ok(());
            }
            if !caps
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::TRANSFER_DST)
            {
                return Err("the surface's images cannot be written by a transfer".to_string());
            }
            let modes = self
                .surface_loader
                .get_physical_device_surface_present_modes(vk.physical, self.surface)
                .unwrap_or_default();
            let mode = if modes.contains(&self.mode) {
                self.mode
            } else {
                vk::PresentModeKHR::FIFO
            };
            let mut count = caps.min_image_count + 1;
            if caps.max_image_count != 0 {
                count = count.min(caps.max_image_count);
            }
            let alpha = if caps
                .supported_composite_alpha
                .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
            {
                vk::CompositeAlphaFlagsKHR::OPAQUE
            } else {
                vk::CompositeAlphaFlagsKHR::INHERIT
            };
            let old = self
                .swapchain
                .as_ref()
                .map_or(vk::SwapchainKHR::null(), |s| s.handle);
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(chosen.format)
                .image_color_space(chosen.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(alpha)
                .present_mode(mode)
                .clipped(true)
                .old_swapchain(old);
            let handle = self
                .swapchain_loader
                .create_swapchain(&info, None)
                .map_err(|e| format!("making the swapchain: {e}"))?;
            let images = self
                .swapchain_loader
                .get_swapchain_images(handle)
                .map_err(|e| format!("getting the swapchain images: {e}"))?;
            let mut release = Vec::with_capacity(images.len());
            for _ in &images {
                release.push(
                    vk.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(|e| format!("creating a semaphore: {e}"))?,
                );
            }
            self.retire(vk);
            if std::env::var_os("NX_TRACE_GPU").is_some() {
                eprintln!(
                    "[vulkan] swapchain of {}x{} in {:?} (window {}x{}, buffers {})",
                    extent.width,
                    extent.height,
                    chosen.format,
                    size.0,
                    size.1,
                    if srgb { "sRGB" } else { "linear" }
                );
            }
            self.swapchain = Some(Swapchain {
                handle,
                format: chosen.format,
                extent,
                images,
                release,
            });
        }
        Ok(())
    }

    /// Destroy the swapchain in use (the device is made idle first).
    fn retire(&mut self, vk: &Vk) {
        if let Some(old) = self.swapchain.take() {
            // SAFETY: nothing of ours uses the swapchain once the device is idle.
            unsafe {
                let _ = vk.device.device_wait_idle();
                for s in old.release {
                    vk.device.destroy_semaphore(s, None);
                }
                self.swapchain_loader.destroy_swapchain(old.handle, None);
            }
        }
    }

    /// Show the displayed part of `target` (see [`blit::visible`]) in the window.
    pub fn show(&mut self, vk: &Vk, target: &mut Target, crop: [i32; 4]) -> Result<(), String> {
        let size = self.window.size();
        if size.0 == 0 || size.1 == 0 {
            // Minimized: nothing to show it in. Frames are still paced (the guest is not let run
            // away) by a short wait.
            std::thread::sleep(std::time::Duration::from_millis(16));
            return Ok(());
        }
        let srgb = is_srgb(target.key.format);
        let wrong = match &self.swapchain {
            None => true,
            Some(s) => is_srgb(s.format) != srgb || (s.extent.width, s.extent.height) != size,
        };
        if wrong {
            self.make_swapchain(vk, size, srgb)?;
        }
        // The last frame's blit must be over before its command buffer is recorded again.
        if self.busy {
            // SAFETY: the fence was submitted with the last frame.
            unsafe {
                vk.device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(|e| format!("waiting for the last frame: {e}"))?;
            }
            self.busy = false;
        }
        for _ in 0..2 {
            let Some(sc) = self.swapchain.as_ref() else {
                return Ok(());
            };
            // SAFETY: valid swapchain and semaphore; the semaphore is unsignaled (the last
            // acquire was waited for by the last submission, which has finished).
            let acquired = unsafe {
                self.swapchain_loader.acquire_next_image(
                    sc.handle,
                    u64::MAX,
                    self.acquire,
                    vk::Fence::null(),
                )
            };
            let index = match acquired {
                Ok((i, _suboptimal)) => i as usize,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.make_swapchain(vk, self.window.size(), srgb)?;
                    continue;
                }
                Err(e) => return Err(format!("acquiring a swapchain image: {e}")),
            };
            let (image, release, handle, extent) = {
                let s = self.swapchain.as_ref().expect("checked above");
                (s.images[index], s.release[index], s.handle, s.extent)
            };
            // SAFETY: recording into a command buffer that is not in use (the fence was waited
            // for) and submitting it with the semaphores of this frame.
            unsafe {
                vk.device
                    .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| format!("resetting the display's command buffer: {e}"))?;
                vk.device
                    .begin_command_buffer(
                        self.cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("beginning the display's command buffer: {e}"))?;
                blit::record(
                    vk,
                    self.cmd,
                    target,
                    crop,
                    &mut self.scratch,
                    image,
                    (extent.width, extent.height),
                );
                blit::to_present(vk, self.cmd, image);
                vk.device
                    .end_command_buffer(self.cmd)
                    .map_err(|e| format!("ending the display's command buffer: {e}"))?;
                vk.device
                    .reset_fences(&[self.fence])
                    .map_err(|e| format!("resetting the display's fence: {e}"))?;
                let waits = [self.acquire];
                let stages = [vk::PipelineStageFlags::TRANSFER];
                let cmds = [self.cmd];
                let signals = [release];
                let submit = [vk::SubmitInfo::default()
                    .wait_semaphores(&waits)
                    .wait_dst_stage_mask(&stages)
                    .command_buffers(&cmds)
                    .signal_semaphores(&signals)];
                vk.device
                    .queue_submit(vk.queue, &submit, self.fence)
                    .map_err(|e| format!("submitting the frame to show: {e}"))?;
                self.busy = true;
                let swapchains = [handle];
                let indices = [index as u32];
                let present = vk::PresentInfoKHR::default()
                    .wait_semaphores(&signals)
                    .swapchains(&swapchains)
                    .image_indices(&indices);
                match self.swapchain_loader.queue_present(vk.queue, &present) {
                    Ok(_) => crate::window::note_presented(),
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                        // The window changed under the frame: the next one starts afresh.
                        self.swapchain_stale();
                    }
                    Err(e) => return Err(format!("presenting a frame: {e}")),
                }
            }
            return Ok(());
        }
        Ok(())
    }

    /// Make the next `show` build the swapchain again.
    fn swapchain_stale(&mut self) {
        if let Some(s) = self.swapchain.as_mut() {
            s.extent = vk::Extent2D {
                width: 0,
                height: 0,
            };
        }
    }

    pub fn destroy(&mut self, vk: &Vk) {
        // SAFETY: the device is idle before anything is destroyed.
        unsafe {
            let _ = vk.device.device_wait_idle();
        }
        self.retire(vk);
        if let Some(s) = self.scratch.take() {
            vk.destroy_image(s);
        }
        // SAFETY: nothing uses these any more.
        unsafe {
            vk.device.destroy_semaphore(self.acquire, None);
            vk.device.destroy_fence(self.fence, None);
            vk.device.destroy_command_pool(self.pool, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
