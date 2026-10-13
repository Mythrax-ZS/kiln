//! The commands of one submission and the memory they use.
//!
//! The processor delivers the commands of a submission one at a time; they are recorded into a
//! command buffer, together with the data they read (constant buffers, vertices, indices, texture
//! uploads), which is copied into host-visible chunks at the moment of the command because the
//! guest may change its memory afterwards. [`Frame::submit`] runs everything and waits, then the
//! chunks and descriptor pools are reused by the next submission.

use crate::compute::HostBuffer;
use crate::device::Vk;
use ash::vk;

/// Size of the chunks the upload memory is carved from.
const CHUNK: u64 = 16 << 20;

struct Chunk {
    buffer: HostBuffer,
    used: u64,
}

pub struct Frame {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    /// Commands that must run before the main ones: texture uploads, which cannot be recorded
    /// inside a rendering.
    staging: vk::CommandBuffer,
    staging_recording: bool,
    fence: vk::Fence,
    recording: bool,
    chunks: Vec<Chunk>,
    /// How many chunks the submission in progress has used (the rest are free).
    in_use: usize,
    descriptor_pools: Vec<vk::DescriptorPool>,
    pools_in_use: usize,
    /// Resources to destroy once the submission has finished.
    pub garbage: Vec<Garbage>,
    /// What each checkpoint of the submission in progress stands for (`NX_DEBUG_CHECKPOINTS`), by
    /// its number less one.
    marks: Vec<String>,
    /// The device was lost: nothing more is asked of it.
    lost: bool,
}

/// Something a submission needed that can be destroyed after it ran.
pub enum Garbage {
    Image(crate::image::Image),
    Buffer(HostBuffer),
}

impl Frame {
    pub fn new(vk: &Vk) -> Result<Frame, String> {
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
                .map_err(|e| format!("creating the command pool: {e}"))?;
            let buffers = vk
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(2),
                )
                .map_err(|e| format!("allocating the command buffers: {e}"))?;
            let fence = vk
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("creating the fence: {e}"))?;
            Ok(Frame {
                pool,
                cmd: buffers[0],
                staging: buffers[1],
                staging_recording: false,
                fence,
                recording: false,
                chunks: Vec::new(),
                in_use: 0,
                descriptor_pools: Vec::new(),
                pools_in_use: 0,
                garbage: Vec::new(),
                marks: Vec::new(),
                lost: false,
            })
        }
    }

    /// The command buffer, started if it was not.
    pub fn cmd(&mut self, vk: &Vk) -> Result<vk::CommandBuffer, String> {
        if !self.recording {
            // SAFETY: the buffer is not in use (the last submission was waited for).
            unsafe {
                vk.device
                    .begin_command_buffer(
                        self.cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("beginning the command buffer: {e}"))?;
            }
            self.recording = true;
        }
        Ok(self.cmd)
    }

    /// The command buffer for uploads, run before the main one; started if it was not.
    pub fn upload_cmd(&mut self, vk: &Vk) -> Result<vk::CommandBuffer, String> {
        if !self.staging_recording {
            // SAFETY: the buffer is not in use (the last submission was waited for).
            unsafe {
                vk.device
                    .begin_command_buffer(
                        self.staging,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("beginning the upload command buffer: {e}"))?;
            }
            self.staging_recording = true;
        }
        Ok(self.staging)
    }

    /// Copy `data` into upload memory aligned to `align` and give the buffer and offset.
    pub fn upload(
        &mut self,
        vk: &Vk,
        data: &[u8],
        align: u64,
    ) -> Result<(vk::Buffer, u64), String> {
        let (buffer, offset, ptr) = self.reserve(vk, data.len() as u64, align)?;
        // SAFETY: `reserve` gives `data.len()` writable bytes at `ptr`.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        Ok((buffer, offset))
    }

    /// Upload memory of `size` bytes: the buffer, the offset in it and where to write.
    pub fn reserve(
        &mut self,
        vk: &Vk,
        size: u64,
        align: u64,
    ) -> Result<(vk::Buffer, u64, *mut u8), String> {
        let align = align.max(1);
        let size = size.max(4);
        // Room in the chunk taken last?
        if self.in_use > 0 {
            let chunk = &mut self.chunks[self.in_use - 1];
            let at = chunk.used.next_multiple_of(align);
            if at + size <= chunk.buffer.size {
                chunk.used = at + size;
                return Ok((chunk.buffer.buffer, at, chunk.buffer.ptr_at(at)));
            }
        }
        // A free chunk of an earlier submission that is big enough, or a new one.
        let reusable = self
            .chunks
            .get(self.in_use)
            .is_some_and(|c| c.buffer.size >= size);
        if !reusable {
            let buffer = vk.create_host_buffer(
                size.max(CHUNK),
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            self.chunks.insert(self.in_use, Chunk { buffer, used: 0 });
        }
        let chunk = &mut self.chunks[self.in_use];
        chunk.used = size;
        self.in_use += 1;
        Ok((chunk.buffer.buffer, 0, chunk.buffer.ptr_at(0)))
    }

    /// A descriptor set with the layout, from the pools of this submission.
    pub fn descriptor_set(
        &mut self,
        vk: &Vk,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        for _ in 0..64 {
            if self.pools_in_use == self.descriptor_pools.len() {
                let sizes = [
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(8192),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(8192),
                ];
                // SAFETY: valid create info; destroyed in `destroy`.
                let pool = unsafe {
                    vk.device.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(512)
                            .pool_sizes(&sizes),
                        None,
                    )
                }
                .map_err(|e| format!("creating a descriptor pool: {e}"))?;
                self.descriptor_pools.push(pool);
            }
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pools[self.pools_in_use])
                .set_layouts(&layouts);
            // SAFETY: the pool and layout are valid.
            match unsafe { vk.device.allocate_descriptor_sets(&info) } {
                Ok(sets) => return Ok(sets[0]),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                    self.pools_in_use += 1;
                }
                Err(e) => return Err(format!("allocating a descriptor set: {e}")),
            }
        }
        Err("no descriptor set could be allocated".to_string())
    }

    /// The device was lost in an earlier submission.
    pub fn lost(&self) -> bool {
        self.lost
    }

    /// Say in `cmd` where the commands recorded so far end, as `what` (`NX_DEBUG_CHECKPOINTS`; a
    /// device that is lost says which of these it reached).
    pub fn mark(&mut self, vk: &Vk, cmd: vk::CommandBuffer, what: impl FnOnce() -> String) {
        if vk.caps.checkpoints {
            self.marks.push(what());
            vk.checkpoint(cmd, self.marks.len() as u64);
        }
    }

    /// Print the checkpoints a lost device reached.
    fn say_where_it_stopped(&self, vk: &Vk, after: std::time::Duration) {
        eprintln!(
            "[vulkan] the device was lost {:.2} s after the commands were submitted",
            after.as_secs_f64()
        );
        if !vk.caps.checkpoints {
            eprintln!(
                "[vulkan]   (NX_DEBUG_CHECKPOINTS=1 makes it say which command it was running)"
            );
            return;
        }
        let reached = vk.checkpoints_reached();
        eprintln!(
            "[vulkan]   {} commands were marked in the submission; the device reached {}:",
            self.marks.len(),
            reached.len()
        );
        let describe = |marker: u64| {
            marker
                .checked_sub(1)
                .and_then(|i| self.marks.get(i as usize))
                .map_or("unknown", String::as_str)
        };
        for &(stage, marker) in &reached {
            eprintln!(
                "[vulkan]   {stage:?} had finished the commands up to {marker}: {}",
                describe(marker)
            );
        }
        // The commands between the last one that is all done and the last one that was begun are
        // the ones the device can have stopped in.
        let first = reached.iter().map(|r| r.1).min().unwrap_or(0);
        let last = reached.iter().map(|r| r.1).max().unwrap_or(0);
        for marker in first + 1..=last.min(first + 40) {
            eprintln!("[vulkan]   command {marker}: {}", describe(marker));
        }
    }

    /// End the recording, run it and wait for it. Afterwards the upload memory and the
    /// descriptor pools are free again and the garbage is gone.
    pub fn submit(&mut self, vk: &Vk) -> Result<(), String> {
        if self.lost {
            return Err("the Vulkan device was lost earlier".to_string());
        }
        if !self.recording && !self.staging_recording {
            return Ok(());
        }
        let began = std::time::Instant::now();
        // SAFETY: the command buffers are recording; the fence is unsignaled; the wait makes
        // everything the commands used safe to release.
        let result = unsafe {
            (|| -> Result<(), (String, bool)> {
                let lost = |e: vk::Result| e == vk::Result::ERROR_DEVICE_LOST;
                // Uploads first, then the commands that use what they upload.
                let mut cmds = Vec::new();
                if self.staging_recording {
                    vk.device
                        .end_command_buffer(self.staging)
                        .map_err(|e| (format!("ending the upload command buffer: {e}"), lost(e)))?;
                    cmds.push(self.staging);
                }
                if self.recording {
                    vk.device
                        .end_command_buffer(self.cmd)
                        .map_err(|e| (format!("ending the command buffer: {e}"), lost(e)))?;
                    cmds.push(self.cmd);
                }
                let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
                let _phase = nx_gpu::progress::enter(nx_gpu::progress::SUBMITTING);
                vk.device
                    .queue_submit(vk.queue, &submit, self.fence)
                    .map_err(|e| (format!("submitting the commands: {e}"), lost(e)))?;
                let _phase = nx_gpu::progress::enter(nx_gpu::progress::WAITING_FOR_DEVICE);
                vk.device
                    .wait_for_fences(&[self.fence], true, 120_000_000_000)
                    .map_err(|e| (format!("waiting for the commands: {e}"), lost(e)))?;
                vk.device
                    .reset_fences(&[self.fence])
                    .map_err(|e| (format!("resetting the fence: {e}"), lost(e)))?;
                Ok(())
            })()
        };
        self.recording = false;
        self.staging_recording = false;
        let result = match result {
            Ok(()) => Ok(()),
            Err((message, true)) => {
                // Nothing more can be asked of a lost device (its calls may never return), but it can
                // be asked where it was when it went.
                self.lost = true;
                self.say_where_it_stopped(vk, began.elapsed());
                return Err(message);
            }
            Err((message, false)) => Err(message),
        };
        // SAFETY: nothing is using the pools now.
        unsafe {
            for cmd in [self.cmd, self.staging] {
                vk.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| format!("resetting a command buffer: {e}"))?;
            }
            for pool in &self.descriptor_pools {
                let _ = vk
                    .device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty());
            }
        }
        self.marks.clear();
        self.pools_in_use = 0;
        for chunk in &mut self.chunks {
            chunk.used = 0;
        }
        self.in_use = 0;
        for g in self.garbage.drain(..) {
            match g {
                Garbage::Image(i) => vk.destroy_image(i),
                Garbage::Buffer(b) => vk.destroy_host_buffer(b),
            }
        }
        result
    }

    pub fn destroy(mut self, vk: &Vk) {
        let _ = self.submit(vk);
        for chunk in self.chunks.drain(..) {
            vk.destroy_host_buffer(chunk.buffer);
        }
        // SAFETY: the device is idle after the last submit.
        unsafe {
            for pool in self.descriptor_pools.drain(..) {
                vk.device.destroy_descriptor_pool(pool, None);
            }
            vk.device.destroy_fence(self.fence, None);
            vk.device.destroy_command_pool(self.pool, None);
        }
    }
}
