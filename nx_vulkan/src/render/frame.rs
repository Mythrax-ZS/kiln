//! The commands of a submission and the memory they use.
//!
//! The processor delivers the commands of a submission one at a time; they are recorded into a
//! command buffer, together with the data they read (constant buffers, vertices, indices, texture
//! uploads), which is copied into host-visible chunks at the moment of the command because the
//! guest may change its memory afterwards.
//!
//! The commands are kept in a few *slots*. [`Frame::kick`] sends the slot being recorded to the
//! device without waiting and goes on recording into the next one, so the device runs one batch
//! while the host records the next (docs/ASSUMPTIONS.md A-123); [`Frame::submit`] sends what is
//! recorded and waits for everything that is in flight. A slot's chunks and descriptor pools are
//! reused, and the garbage it was given is destroyed, once the device has finished its batch.

use crate::compute::HostBuffer;
use crate::device::Vk;
use ash::vk;
use std::collections::VecDeque;

/// Size of the chunks the upload memory is carved from.
const CHUNK: u64 = 16 << 20;

/// How many batches there are: one is recorded while the others run.
const SLOTS: usize = 3;

struct Chunk {
    buffer: HostBuffer,
    used: u64,
}

/// What one batch of commands owns.
struct Slot {
    cmd: vk::CommandBuffer,
    /// Commands that must run before the main ones: texture uploads, which cannot be recorded
    /// inside a rendering.
    staging: vk::CommandBuffer,
    staging_recording: bool,
    fence: vk::Fence,
    recording: bool,
    chunks: Vec<Chunk>,
    /// How many chunks the batch in progress has used (the rest are free).
    in_use: usize,
    descriptor_pools: Vec<vk::DescriptorPool>,
    pools_in_use: usize,
    /// Resources to destroy once the batch has finished.
    garbage: Vec<Garbage>,
    /// What each checkpoint of the batch stands for (`NX_DEBUG_CHECKPOINTS`), by its number less
    /// one.
    marks: Vec<String>,
    /// When the batch was submitted, while the device has it.
    submitted: Option<std::time::Instant>,
}

pub struct Frame {
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    /// The slot commands are being recorded into.
    current: usize,
    /// The slots the device has, the oldest first.
    flying: VecDeque<usize>,
    /// Resources to destroy once the commands recorded so far (and those before them) have run.
    pub garbage: Vec<Garbage>,
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
                        .command_buffer_count(2 * SLOTS as u32),
                )
                .map_err(|e| format!("allocating the command buffers: {e}"))?;
            let mut slots = Vec::new();
            for pair in buffers.chunks(2) {
                let fence = vk
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| format!("creating a fence: {e}"))?;
                slots.push(Slot {
                    cmd: pair[0],
                    staging: pair[1],
                    staging_recording: false,
                    fence,
                    recording: false,
                    chunks: Vec::new(),
                    in_use: 0,
                    descriptor_pools: Vec::new(),
                    pools_in_use: 0,
                    garbage: Vec::new(),
                    marks: Vec::new(),
                    submitted: None,
                });
            }
            Ok(Frame {
                pool,
                slots,
                current: 0,
                flying: VecDeque::new(),
                garbage: Vec::new(),
                lost: false,
            })
        }
    }

    /// The command buffer, started if it was not.
    pub fn cmd(&mut self, vk: &Vk) -> Result<vk::CommandBuffer, String> {
        let slot = &mut self.slots[self.current];
        if !slot.recording {
            // SAFETY: the buffer is not in use (the slot was waited for before it was taken).
            unsafe {
                vk.device
                    .begin_command_buffer(
                        slot.cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("beginning the command buffer: {e}"))?;
            }
            slot.recording = true;
        }
        Ok(slot.cmd)
    }

    /// The command buffer for uploads, run before the main one; started if it was not.
    pub fn upload_cmd(&mut self, vk: &Vk) -> Result<vk::CommandBuffer, String> {
        let slot = &mut self.slots[self.current];
        if !slot.staging_recording {
            // SAFETY: the buffer is not in use (the slot was waited for before it was taken).
            unsafe {
                vk.device
                    .begin_command_buffer(
                        slot.staging,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("beginning the upload command buffer: {e}"))?;
            }
            slot.staging_recording = true;
        }
        Ok(slot.staging)
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
        let slot = &mut self.slots[self.current];
        let align = align.max(1);
        let size = size.max(4);
        // Room in the chunk taken last?
        if slot.in_use > 0 {
            let chunk = &mut slot.chunks[slot.in_use - 1];
            let at = chunk.used.next_multiple_of(align);
            if at + size <= chunk.buffer.size {
                chunk.used = at + size;
                return Ok((chunk.buffer.buffer, at, chunk.buffer.ptr_at(at)));
            }
        }
        // A free chunk of an earlier batch that is big enough, or a new one.
        let reusable = slot
            .chunks
            .get(slot.in_use)
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
            slot.chunks.insert(slot.in_use, Chunk { buffer, used: 0 });
        }
        let chunk = &mut slot.chunks[slot.in_use];
        chunk.used = size;
        slot.in_use += 1;
        Ok((chunk.buffer.buffer, 0, chunk.buffer.ptr_at(0)))
    }

    /// A descriptor set with the layout, from the pools of this batch.
    pub fn descriptor_set(
        &mut self,
        vk: &Vk,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, String> {
        let slot = &mut self.slots[self.current];
        let layouts = [layout];
        for _ in 0..64 {
            if slot.pools_in_use == slot.descriptor_pools.len() {
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
                slot.descriptor_pools.push(pool);
            }
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(slot.descriptor_pools[slot.pools_in_use])
                .set_layouts(&layouts);
            // SAFETY: the pool and layout are valid.
            match unsafe { vk.device.allocate_descriptor_sets(&info) } {
                Ok(sets) => return Ok(sets[0]),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                    slot.pools_in_use += 1;
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

    /// Is anything recorded that has not been sent to the device?
    pub fn pending(&self) -> bool {
        let slot = &self.slots[self.current];
        slot.recording || slot.staging_recording
    }

    /// Say in `cmd` where the commands recorded so far end, as `what` (`NX_DEBUG_CHECKPOINTS`; a
    /// device that is lost says which of these it reached).
    pub fn mark(&mut self, vk: &Vk, cmd: vk::CommandBuffer, what: impl FnOnce() -> String) {
        if vk.caps.checkpoints {
            let marks = &mut self.slots[self.current].marks;
            marks.push(what());
            vk.checkpoint(cmd, marks.len() as u64);
        }
    }

    /// Print the checkpoints a lost device reached.
    fn say_where_it_stopped(&self, vk: &Vk, slot: usize) {
        let slot = &self.slots[slot];
        let after = slot.submitted.map(|t| t.elapsed()).unwrap_or_default();
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
            slot.marks.len(),
            reached.len()
        );
        let describe = |marker: u64| {
            marker
                .checked_sub(1)
                .and_then(|i| slot.marks.get(i as usize))
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

    /// End the recording of the slot, send it to the device and go on with the next slot, waiting
    /// for the device first if that is still busy. Nothing is waited for otherwise: what the
    /// batch does is seen by the host only after a [`Frame::submit`].
    pub fn kick(&mut self, vk: &Vk) -> Result<(), String> {
        if self.lost {
            return Err("the Vulkan device was lost earlier".to_string());
        }
        if !self.pending() {
            return Ok(());
        }
        let current = self.current;
        let result = {
            let slot = &mut self.slots[current];
            // SAFETY: the command buffers are recording; the fence is unsignaled and belongs to
            // this slot alone.
            unsafe {
                (|| -> Result<(), (String, bool)> {
                    let lost = |e: vk::Result| e == vk::Result::ERROR_DEVICE_LOST;
                    // Uploads first, then the commands that use what they upload.
                    let mut cmds = Vec::new();
                    if slot.staging_recording {
                        vk.device.end_command_buffer(slot.staging).map_err(|e| {
                            (format!("ending the upload command buffer: {e}"), lost(e))
                        })?;
                        cmds.push(slot.staging);
                    }
                    if slot.recording {
                        vk.device
                            .end_command_buffer(slot.cmd)
                            .map_err(|e| (format!("ending the command buffer: {e}"), lost(e)))?;
                        cmds.push(slot.cmd);
                    }
                    let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
                    let _phase = nx_gpu::progress::enter(nx_gpu::progress::SUBMITTING);
                    vk.device
                        .queue_submit(vk.queue, &submit, slot.fence)
                        .map_err(|e| (format!("submitting the commands: {e}"), lost(e)))?;
                    Ok(())
                })()
            }
        };
        let slot = &mut self.slots[current];
        slot.recording = false;
        slot.staging_recording = false;
        if let Err((message, lost)) = result {
            if lost {
                // Nothing more can be asked of a lost device (its calls may never return), but
                // it can be asked where it was when it went.
                self.lost = true;
                slot.submitted = Some(std::time::Instant::now());
                self.say_where_it_stopped(vk, current);
            }
            return Err(message);
        }
        slot.submitted = Some(std::time::Instant::now());
        slot.garbage.append(&mut self.garbage);
        self.flying.push_back(current);
        // The next slot takes the next batch: the device may still have it.
        self.current = (current + 1) % SLOTS;
        while self.slots[self.current].submitted.is_some() {
            self.wait_oldest(vk)?;
        }
        Ok(())
    }

    /// Wait for the oldest batch the device has and take its slot back.
    fn wait_oldest(&mut self, vk: &Vk) -> Result<(), String> {
        let Some(index) = self.flying.pop_front() else {
            return Ok(());
        };
        let fence = self.slots[index].fence;
        let _phase = nx_gpu::progress::enter(nx_gpu::progress::WAITING_FOR_DEVICE);
        // SAFETY: the fence was submitted with the slot's batch and is waited for once.
        let waited = unsafe {
            vk.device
                .wait_for_fences(&[fence], true, 120_000_000_000)
                .and_then(|()| vk.device.reset_fences(&[fence]))
        };
        if let Err(e) = waited {
            if e == vk::Result::ERROR_DEVICE_LOST {
                self.lost = true;
                self.say_where_it_stopped(vk, index);
            }
            return Err(format!("waiting for the commands: {e}"));
        }
        let slot = &mut self.slots[index];
        slot.submitted = None;
        // SAFETY: nothing is using the buffers and pools now.
        unsafe {
            for cmd in [slot.cmd, slot.staging] {
                vk.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| format!("resetting a command buffer: {e}"))?;
            }
            for pool in &slot.descriptor_pools {
                let _ = vk
                    .device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty());
            }
        }
        slot.marks.clear();
        slot.pools_in_use = 0;
        for chunk in &mut slot.chunks {
            chunk.used = 0;
        }
        slot.in_use = 0;
        for g in slot.garbage.drain(..) {
            match g {
                Garbage::Image(i) => vk.destroy_image(i),
                Garbage::Buffer(b) => vk.destroy_host_buffer(b),
            }
        }
        Ok(())
    }

    /// Send what is recorded to the device and wait for everything it has. Afterwards the upload
    /// memory and the descriptor pools are free again and the garbage is gone.
    pub fn submit(&mut self, vk: &Vk) -> Result<(), String> {
        if self.lost {
            return Err("the Vulkan device was lost earlier".to_string());
        }
        let sent = self.kick(vk);
        // Every batch is waited for, whether or not the last could be sent.
        let mut waited = Ok(());
        while !self.flying.is_empty() {
            if let Err(e) = self.wait_oldest(vk) {
                waited = waited.and(Err(e));
            }
        }
        sent.and(waited)
    }

    pub fn destroy(mut self, vk: &Vk) {
        let _ = self.submit(vk);
        // SAFETY: the device is idle after the last submit.
        unsafe {
            for slot in &mut self.slots {
                for chunk in slot.chunks.drain(..) {
                    vk.destroy_host_buffer(chunk.buffer);
                }
                for pool in slot.descriptor_pools.drain(..) {
                    vk.device.destroy_descriptor_pool(pool, None);
                }
                vk.device.destroy_fence(slot.fence, None);
            }
            vk.device.destroy_command_pool(self.pool, None);
        }
        // What was left to destroy after the last batch.
        for g in self.garbage.drain(..) {
            match g {
                Garbage::Image(i) => vk.destroy_image(i),
                Garbage::Buffer(b) => vk.destroy_host_buffer(b),
            }
        }
    }
}
