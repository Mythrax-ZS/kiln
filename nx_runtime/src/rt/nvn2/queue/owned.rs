//! Native queue/sync lifetimes and GPU fence ordering (A-308).

use super::{invalid, read, write, Builder, Error, NativeCpu};
use crate::rt::nvn2::{device::Devices, memory::label, rt};
use nx_vulkan::native::{Completion, Queue as GpuQueue};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Queue {
    cookie: u64,
    device: u64,
    generation: u64,
    primary: u64,
    flags: u32,
    gpu: GpuQueue,
    pending: Vec<(Arc<Completion>, bool)>,
    _storage: Vec<u8>,
    _guest_storage: (u64, u64),
    _compute_size: u64,
    label: Vec<u8>,
}

impl Queue {
    fn flush(&mut self) -> Result<(), Error> {
        let pending = std::mem::take(&mut self.pending);
        for (index, (completion, for_cpu)) in pending.iter().enumerate() {
            if let Err(e) = self.gpu.submit_marker(Arc::clone(completion), *for_cpu) {
                for (completion, _) in &pending[index..] {
                    completion.cancel(&e);
                }
                return Err(invalid_gpu(e));
            }
        }
        Ok(())
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // A retired queue must not leave a pending sync permanently waiting for a flush.
        let _ = self.flush();
    }
}

struct Sync {
    cookie: u64,
    device: u64,
    generation: u64,
    origin: Option<(u64, u64)>,
    completion: Option<Arc<Completion>>,
    label: Vec<u8>,
}

#[derive(Default)]
pub(in crate::rt::nvn2) struct Objects {
    queues: BTreeMap<u64, Queue>,
    syncs: BTreeMap<u64, Sync>,
    next_id: u32,
}

fn invalid_gpu(message: String) -> Error {
    invalid(&message)
}

/// Captured under the driver lock; the potentially blocking wait runs after that lock is gone.
pub(in crate::rt::nvn2) struct Waiter {
    completion: Option<Arc<Completion>>,
    timeout: u64,
    returns_status: bool,
}

impl Waiter {
    fn wait(&self, cpu: &NativeCpu, completion: &Completion) -> Result<bool, Error> {
        let ctx = cpu.ctl.user.cast::<crate::horizon::kernel::ThreadCtx>();
        // SAFETY: Horizon threads keep ThreadCtx alive for the whole run; plain calls leave null.
        let kernel = unsafe { ctx.as_ref() }.map(|ctx| Arc::clone(&ctx.kernel));
        let began = Instant::now();
        let budget = Duration::from_nanos(self.timeout);
        loop {
            if kernel.as_ref().is_some_and(|k| k.exit_state().is_some()) {
                return Err(invalid("wait canceled because the process is ending"));
            }
            let remaining = if self.timeout == u64::MAX {
                50_000_000
            } else {
                budget
                    .saturating_sub(began.elapsed())
                    .as_nanos()
                    .min(50_000_000) as u64
            };
            if completion.wait(remaining).map_err(invalid_gpu)? {
                return Ok(true);
            }
            if self.timeout != u64::MAX && began.elapsed() >= budget {
                return Ok(false);
            }
        }
    }

    pub(in crate::rt::nvn2) fn run(self, cpu: &mut NativeCpu) -> Result<(), Error> {
        if !self.returns_status {
            if !self.wait(
                cpu,
                self.completion.as_ref().expect("finish has a completion"),
            )? {
                return Err(invalid("finish timed out"));
            }
            return Ok(());
        }
        cpu.state.x[0] = if let Some(completion) = self.completion.as_ref() {
            if completion.signaled().map_err(invalid_gpu)? {
                0
            } else if self.wait(cpu, completion)? {
                1
            } else {
                2
            }
        } else {
            3
        };
        Ok(())
    }
}

impl Objects {
    pub(in crate::rt::nvn2) fn has_children(&self, device: u64) -> bool {
        self.queues.values().any(|q| q.device == device)
            || self.syncs.values().any(|s| s.device == device)
    }

    fn queue(&self, address: u64, devices: &Devices) -> Result<&Queue, Error> {
        let cookie = u64::from_le_bytes(read(address.wrapping_add(0xF8))?);
        let queue = self
            .queues
            .get(&address)
            .filter(|q| q.cookie == cookie)
            .ok_or_else(|| invalid("is not initialized or its storage was replaced"))?;
        if devices.identity(queue.device)? != queue.generation {
            return Err(invalid("belongs to a retired device"));
        }
        Ok(queue)
    }

    fn sync(&self, address: u64, devices: &Devices) -> Result<&Sync, Error> {
        let cookie = u64::from_le_bytes(read(address.wrapping_add(24))?);
        let sync = self
            .syncs
            .get(&address)
            .filter(|s| s.cookie == cookie)
            .ok_or_else(|| invalid("sync is not initialized or its storage was replaced"))?;
        if devices.identity(sync.device)? != sync.generation {
            return Err(invalid("sync belongs to a retired device"));
        }
        Ok(sync)
    }

    fn initialize(&mut self, cpu: &mut NativeCpu, devices: &Devices) -> Result<(), Error> {
        let address = cpu.state.x[0];
        let previous = read::<256>(address)?;
        write(address, &previous)?;
        let builder = Builder::read(cpu.state.x[1])?;
        let generation = devices.identity(builder.device)?;
        let size = match builder.size() {
            Ok(size) => size,
            Err(_) => {
                cpu.state.x[0] = 0;
                return Ok(());
            }
        };
        if self.next_id == u32::MAX
            || self.queues.contains_key(&address)
            || self.syncs.contains_key(&address)
            || !address.is_multiple_of(8)
            || address == builder.device
            || builder.flags & !0xF != 0
            || builder.flags & 0xC == 0xC
            || builder.ray != 0
            || builder.priority != 100
            || size > 64 << 20
            || builder.memory != 0
                && (!builder.memory.is_multiple_of(4096)
                    || !builder.memory_size.is_multiple_of(4096)
                    || builder.memory_size < size
                    || !rt().mem.is_mapped(builder.memory, builder.memory_size))
            || builder.memory == 0 && builder.memory_size != 0
        {
            eprintln!(
                "[nvn2] QueueInitialize refused flags {:#x}, priority {}, memory {size:#x}",
                builder.flags, builder.priority
            );
            cpu.state.x[0] = 0;
            return Ok(());
        }
        if builder.flags & 4 != 0 {
            if builder.primary == 0 {
                cpu.state.x[0] = 0;
                return Ok(());
            }
            let primary = self.queue(builder.primary, devices)?;
            if primary.device != builder.device || primary.flags & 8 == 0 {
                cpu.state.x[0] = 0;
                return Ok(());
            }
        } else if builder.primary != 0 {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let gpu = match devices.context(builder.device)?.queue() {
            Ok(queue) => queue,
            Err(e) => {
                eprintln!("[nvn2] QueueInitialize failed: {e}");
                cpu.state.x[0] = 0;
                return Ok(());
            }
        };
        let mut storage = Vec::new();
        if builder.memory == 0 {
            if storage.try_reserve_exact(size as usize).is_err() {
                cpu.state.x[0] = 0;
                return Ok(());
            }
            storage.resize(size as usize, 0);
        }
        let cookie = 0x4B51_5545_0000_0000 | u64::from(self.next_id + 1);
        let mut header = [0; 256];
        header[16..24].copy_from_slice(&builder.device.to_le_bytes());
        header[0xF8..].copy_from_slice(&cookie.to_le_bytes());
        write(address, &header)?;
        self.next_id += 1;
        self.queues.insert(
            address,
            Queue {
                cookie,
                device: builder.device,
                generation,
                primary: builder.primary,
                flags: builder.flags,
                gpu,
                pending: Vec::new(),
                _storage: storage,
                _guest_storage: (builder.memory, builder.memory_size),
                _compute_size: builder.compute,
                label: Vec::new(),
            },
        );
        eprintln!(
            "[nvn2] native queue initialized (flags {:#x}, command {}, control {}, compute {})",
            builder.flags, builder.command, builder.control, builder.compute
        );
        cpu.state.x[0] = 1;
        Ok(())
    }

    pub(in crate::rt::nvn2) fn prepare_wait(
        &mut self,
        name: &str,
        cpu: &NativeCpu,
        devices: &Devices,
    ) -> Result<Waiter, Error> {
        let address = cpu.state.x[0];
        if name == "nvnSyncWait" {
            return Ok(Waiter {
                completion: self.sync(address, devices)?.completion.clone(),
                timeout: cpu.state.x[1],
                returns_status: true,
            });
        }
        let queue = self.queue(address, devices)?;
        let completion = queue.gpu.context().completion().map_err(invalid_gpu)?;
        let queue = self.queues.get_mut(&address).expect("validated queue");
        queue.pending.push((Arc::clone(&completion), true));
        queue.flush()?;
        Ok(Waiter {
            completion: Some(completion),
            timeout: u64::MAX,
            returns_status: false,
        })
    }

    pub(in crate::rt::nvn2) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        devices: &Devices,
    ) -> Option<Result<(), Error>> {
        let (object, a, b, c) = (
            cpu.state.x[0],
            cpu.state.x[1],
            cpu.state.x[2],
            cpu.state.x[3],
        );
        let result = match name {
            "nvnQueueInitialize" => self.initialize(cpu, devices),
            "nvnQueueFinalize" => (|| {
                self.queue(object, devices)?;
                if self.queues.values().any(|q| q.primary == object) {
                    return Err(invalid("still owns secondary queues"));
                }
                self.queues
                    .get_mut(&object)
                    .expect("validated queue")
                    .flush()?;
                write(object.wrapping_add(0xF8), &0u64.to_le_bytes())?;
                self.queues.remove(&object);
                Ok(())
            })(),
            "nvnQueueFlush" => (|| {
                self.queue(object, devices)?;
                self.queues
                    .get_mut(&object)
                    .expect("validated queue")
                    .flush()
            })(),
            "nvnQueueSetDebugLabel" => (|| {
                self.queue(object, devices)?;
                let value = label(a)?;
                self.queues.get_mut(&object).expect("validated queue").label = value;
                Ok(())
            })(),
            "nvnQueueGetError" => (|| {
                self.queue(object, devices)?;
                // Submission errors are fatal diagnostics; a live queue has no recorded error.
                cpu.state.x[0] = 0;
                Ok(())
            })(),
            "nvnSyncInitialize" => (|| {
                let previous = read::<32>(object)?;
                write(object, &previous)?;
                let generation = devices.identity(a)?;
                if self.next_id == u32::MAX
                    || self.syncs.contains_key(&object)
                    || self.queues.contains_key(&object)
                    || object == a
                    || !object.is_multiple_of(8)
                {
                    cpu.state.x[0] = 0;
                    return Ok(());
                }
                let cookie = 0x4B53_594E_0000_0000 | u64::from(self.next_id + 1);
                let mut header = [0; 32];
                header[..8].copy_from_slice(&a.to_le_bytes());
                header[24..].copy_from_slice(&cookie.to_le_bytes());
                write(object, &header)?;
                self.next_id += 1;
                self.syncs.insert(
                    object,
                    Sync {
                        cookie,
                        device: a,
                        generation,
                        origin: None,
                        completion: None,
                        label: Vec::new(),
                    },
                );
                cpu.state.x[0] = 1;
                Ok(())
            })(),
            "nvnSyncFinalize" => (|| {
                self.sync(object, devices)?;
                write(object.wrapping_add(24), &0u64.to_le_bytes())?;
                self.syncs.remove(&object);
                Ok(())
            })(),
            "nvnSyncSetDebugLabel" => (|| {
                self.sync(object, devices)?;
                let value = label(a)?;
                self.syncs.get_mut(&object).expect("validated sync").label = value;
                Ok(())
            })(),
            "nvnQueueFenceSync" => (|| {
                let queue = self.queue(object, devices)?;
                let sync = self.sync(a, devices)?;
                if sync.device != queue.device || b as u32 > 1 || c as u32 & !1 != 0 {
                    return Err(invalid(
                        "fence has a different device or unsupported condition/flags",
                    ));
                }
                let origin = (object, queue.cookie);
                let completion = queue.gpu.context().completion().map_err(invalid_gpu)?;
                self.queues
                    .get_mut(&object)
                    .expect("validated queue")
                    .pending
                    .push((Arc::clone(&completion), c as u32 & 1 != 0));
                let sync = self.syncs.get_mut(&a).expect("validated sync");
                sync.origin = Some(origin);
                sync.completion = Some(completion);
                Ok(())
            })(),
            "nvnQueueWaitSync" => (|| {
                let queue = self.queue(object, devices)?;
                let sync = self.sync(a, devices)?;
                let Some(completion) = sync.completion.as_ref() else {
                    cpu.state.x[0] = 0;
                    return Ok(());
                };
                if sync.device != queue.device {
                    return Err(invalid("wait uses a sync from another device"));
                }
                if sync.origin != Some((object, queue.cookie)) && !completion.submitted() {
                    return Err(invalid(
                        "cross-queue sync must be flushed before it is waited on",
                    ));
                }
                // Logical queues share one serialized physical queue. This marker's ALL_COMMANDS
                // memory dependency orders subsequent submissions after that flushed prefix.
                let barrier = queue.gpu.context().completion().map_err(invalid_gpu)?;
                self.queues
                    .get_mut(&object)
                    .expect("validated queue")
                    .pending
                    .push((barrier, false));
                cpu.state.x[0] = 1;
                Ok(())
            })(),
            _ => return None,
        };
        Some(result)
    }
}
