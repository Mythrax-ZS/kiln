//! Native API queue submissions and completion, sharing the renderer's actual Vulkan device.
//!
//! Each logical queue owns a command pool. Completion is a real Vulkan fence, with a separate
//! unsubmitted state so a CPU wait can sleep while another thread flushes its owning queue.

use crate::Vk;
use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

mod shader;
pub use shader::{ShaderBinary, ShaderModule, ShaderOptions, ShaderStage};
mod graphics;
mod shared;
pub use shared::{SharedMemory, SharedWord};
mod signal;
pub use signal::Signal;
mod scheduled;
pub(crate) mod texture;
pub use graphics::{
    Attribute, BlendFactor, BlendOp, ColorBlend, GraphicsPipeline, GraphicsState, InputBuffer,
    NativeDraw, Primitive, Stream, VertexFormat, VertexKind,
};
pub use scheduled::{Operation, Recorded, ScheduledQueue};
pub use texture::{
    ImageKind, Readback, TexelFormat, Texture, TextureInfo, TextureRegion, TransferStats,
};

#[derive(Clone)]
pub struct Context {
    vk: Arc<Vk>,
    graphics: Arc<Mutex<graphics::Cache>>,
    signals: Arc<Mutex<[Option<Arc<signal::Kernel>>; 2]>>,
}

impl Context {
    pub(crate) fn uses_device(&self, vk: &Arc<Vk>) -> bool {
        Arc::ptr_eq(&self.vk, vk)
    }
    pub(crate) fn new(vk: Arc<Vk>) -> Self {
        Self {
            vk,
            graphics: Arc::new(Mutex::new(graphics::Cache::default())),
            signals: Arc::new(Mutex::new([None, None])),
        }
    }

    pub fn queue(&self) -> Result<Queue, String> {
        // SAFETY: the device is alive through the context. This pool belongs to one queue.
        let pool = unsafe {
            self.vk.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.vk.queue_family),
                None,
            )
        }
        .map_err(|e| format!("creating a native command pool: {e}"))?;
        Ok(Queue {
            context: self.clone(),
            pool,
            flying: Vec::new(),
            submissions: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn completion(&self) -> Result<Arc<Completion>, String> {
        // SAFETY: the fence is owned by Completion; its device is retained until destruction.
        let fence = unsafe {
            self.vk
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| format!("creating a native fence: {e}"))?;
        Ok(Arc::new(Completion {
            vk: Arc::clone(&self.vk),
            fence,
            state: Mutex::new(State::Pending),
            changed: Condvar::new(),
            scheduled: AtomicBool::new(false),
        }))
    }
}

enum State {
    Pending,
    Submitting,
    Submitted,
    Failed(String),
}

pub struct Completion {
    vk: Arc<Vk>,
    fence: vk::Fence,
    state: Mutex<State>,
    changed: Condvar,
    scheduled: AtomicBool,
}

impl Completion {
    pub fn enqueued(&self) -> bool {
        self.scheduled.load(Ordering::Acquire) || self.submitted()
    }
    fn mark_enqueued(&self) -> Result<(), String> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !matches!(*state, State::Pending) || self.scheduled.swap(true, Ordering::AcqRel) {
            return Err("native completion was already queued or submitted".into());
        }
        Ok(())
    }
    pub fn submitted(&self) -> bool {
        matches!(
            *self.state.lock().unwrap_or_else(|e| e.into_inner()),
            State::Submitted
        )
    }

    pub fn signaled(&self) -> Result<bool, String> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            State::Pending | State::Submitting => Ok(false),
            State::Failed(e) => Err(e.clone()),
            State::Submitted => {
                // SAFETY: the live fence is never reset or reused, and this is a status query.
                unsafe { self.vk.device.get_fence_status(self.fence) }
                    .map_err(|e| format!("querying a native fence: {e}"))
            }
        }
    }

    /// Wait for submission as well as GPU completion, with one timeout budget for both.
    pub fn wait(&self, timeout_ns: u64) -> Result<bool, String> {
        let began = Instant::now();
        let budget = Duration::from_nanos(timeout_ns);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while matches!(*state, State::Pending | State::Submitting) {
            if timeout_ns == u64::MAX {
                state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
            } else {
                let remaining = budget.saturating_sub(began.elapsed());
                if remaining.is_zero() {
                    return Ok(false);
                }
                state = self
                    .changed
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        }
        if let State::Failed(e) = &*state {
            return Err(e.clone());
        }
        drop(state);
        let remaining = if timeout_ns == u64::MAX {
            u64::MAX
        } else {
            budget
                .saturating_sub(began.elapsed())
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64
        };
        // SAFETY: the fence and device stay alive for the wait and are never reset.
        match unsafe {
            self.vk
                .device
                .wait_for_fences(&[self.fence], true, remaining)
        } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(format!("waiting for a native fence: {e}")),
        }
    }

    fn finish_submission(&self, result: &Result<(), String>) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = match result {
            Ok(()) => State::Submitted,
            Err(e) => State::Failed(e.clone()),
        };
        self.changed.notify_all();
    }

    /// Reject work that was never submitted, waking CPU waits on its canceled completion.
    pub fn cancel(&self, message: &str) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*state, State::Pending) {
            *state = State::Failed(message.to_string());
            self.changed.notify_all();
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        // A queue retains submitted completions until their command buffers are done. Pending
        // fences have never been submitted. No caller can wait on this last Arc any more.
        if self.submitted() {
            let _ = self.wait(u64::MAX);
        }
        // SAFETY: no operation still uses this fence, and its retained device is alive.
        unsafe { self.vk.device.destroy_fence(self.fence, None) };
    }
}

struct Flight {
    command: vk::CommandBuffer,
    done: Arc<Completion>,
    objects: Vec<Arc<dyn Send + Sync>>,
}
pub struct Queue {
    context: Context,
    pool: vk::CommandPool,
    flying: Vec<Flight>,
    submissions: Arc<AtomicU64>,
}

impl Queue {
    pub(crate) fn keep_alive(
        &mut self,
        done: &Arc<Completion>,
        objects: Vec<Arc<dyn Send + Sync>>,
    ) {
        let flight = self.flying.last_mut().expect("submitted commands");
        assert!(Arc::ptr_eq(&flight.done, done));
        flight.objects.extend(objects);
    }
    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn submissions(&self) -> u64 {
        self.submissions.load(Ordering::Acquire)
    }

    pub fn submit_marker(
        &mut self,
        completion: Arc<Completion>,
        for_cpu: bool,
    ) -> Result<(), String> {
        self.submit_commands(completion, for_cpu, |_, _| {})
    }

    pub(crate) fn submit_commands(
        &mut self,
        completion: Arc<Completion>,
        for_cpu: bool,
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.vk, &completion.vk) {
            return Err("a native fence belongs to another Vulkan device".to_string());
        }
        {
            let mut state = completion.state.lock().unwrap_or_else(|e| e.into_inner());
            if !matches!(*state, State::Pending) {
                return Err("a native fence was already submitted".to_string());
            }
            *state = State::Submitting;
        }
        let result = self.submit(&completion, for_cpu, record);
        completion.finish_submission(&result);
        result
    }

    fn submit(
        &mut self,
        completion: &Arc<Completion>,
        for_cpu: bool,
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) -> Result<(), String> {
        self.reap()?;
        let vk = &self.context.vk;
        // SAFETY: this pool is exclusively used by this queue.
        let cmd = unsafe {
            vk.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| format!("allocating a native command buffer: {e}"))?[0];
        // SAFETY: cmd is new; the memory dependency covers preceding work on the same physical
        // queue. Submission is serialized across all logical queues from this context.
        let result = unsafe {
            (|| {
                vk.device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("beginning a native command buffer: {e}"))?;
                let access = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(if for_cpu {
                        access | vk::AccessFlags::HOST_READ
                    } else {
                        access
                    })];
                vk.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    if for_cpu {
                        vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST
                    } else {
                        vk::PipelineStageFlags::ALL_COMMANDS
                    },
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
                record(vk, cmd);
                vk.device
                    .end_command_buffer(cmd)
                    .map_err(|e| format!("ending a native command buffer: {e}"))?;
                let buffers = [cmd];
                let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
                let _lock = vk.queue_lock.lock().unwrap_or_else(|e| e.into_inner());
                vk.device
                    .queue_submit(vk.queue, &submit, completion.fence)
                    .map_err(|e| format!("submitting native commands: {e}"))
            })()
        };
        match result {
            Ok(()) => {
                self.submissions.fetch_add(1, Ordering::Release);
                self.flying.push(Flight {
                    command: cmd,
                    done: Arc::clone(completion),
                    objects: Vec::new(),
                });
                Ok(())
            }
            Err(e) => {
                // SAFETY: a failed submission did not leave cmd pending.
                unsafe { vk.device.free_command_buffers(self.pool, &[cmd]) };
                Err(e)
            }
        }
    }

    fn reap(&mut self) -> Result<(), String> {
        let mut index = 0;
        while index < self.flying.len() {
            if self.flying[index].done.signaled()? {
                let flight = self.flying.swap_remove(index);
                // SAFETY: this command buffer's fence completed, and the pool is exclusive.
                unsafe {
                    self.context
                        .vk
                        .device
                        .free_command_buffers(self.pool, &[flight.command])
                };
            } else {
                index += 1;
            }
        }
        Ok(())
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        for flight in &self.flying {
            let _ = flight.done.wait(u64::MAX);
        }
        // SAFETY: submitted work has completed or the device was lost; no buffer remains in use.
        unsafe { self.context.vk.device.destroy_command_pool(self.pool, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_native_completion_is_pending_until_submission_and_retains_its_device() {
        let mut renderer = crate::Renderer::new(crate::RenderConfig::default());
        let context = match renderer.native_context() {
            Ok(context) => context,
            Err(e) => {
                assert!(
                    std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
                    "Vulkan required: {e}"
                );
                eprintln!("SKIPPED (no Vulkan device): {e}");
                return;
            }
        };
        let completion = context.completion().unwrap();
        assert!(!completion.signaled().unwrap());
        assert!(!completion.wait(0).unwrap());
        let mut queue = context.queue().unwrap();
        let waited = Arc::clone(&completion);
        let join = std::thread::spawn(move || waited.wait(5_000_000_000).unwrap());
        queue.submit_marker(Arc::clone(&completion), true).unwrap();
        assert!(join.join().unwrap());
        assert_eq!(queue.submissions(), 1);
        assert!(
            queue.submit_marker(Arc::clone(&completion), false).is_err(),
            "fences cannot be reused"
        );
        let canceled = context.completion().unwrap();
        canceled.cancel("test cancellation");
        assert!(canceled.wait(0).unwrap_err().contains("test cancellation"));
        drop(queue);
        drop(context);
        drop(renderer);
        assert!(
            completion.signaled().unwrap(),
            "a live fence retains its actual device"
        );
    }
}
