//! Coherent, explicitly owned imports of stable host pages. No shadow event values.

use super::Context;
use crate::compute::ImportedMemory;
use crate::Vk;
use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[cfg(test)]
mod tests;

pub struct SharedMemory {
    pub(super) vk: Arc<Vk>,
    memory: Option<ImportedMemory>,
    host: usize,
    len: u64,
    changed: Mutex<()>,
    wake: Condvar,
    _owner: Option<Allocation>,
}
struct Allocation {
    host: usize,
    layout: std::alloc::Layout,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: the import is destroyed before its owned allocation, after GPU owners release it.
        unsafe {
            std::alloc::dealloc(self.host as *mut u8, self.layout);
        }
    }
}
impl SharedMemory {
    pub fn host(&self) -> usize {
        self.host
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.memory.as_ref().expect("live shared memory").buffer
    }
    pub(super) fn address(&self) -> u64 {
        self.memory.as_ref().expect("live shared memory").address
    }
    pub fn word(self: &Arc<Self>, offset: u64) -> Result<Arc<SharedWord>, String> {
        if !offset.is_multiple_of(4) || !offset.checked_add(4).is_some_and(|end| end <= self.len) {
            return Err("shared event word is unaligned or outside its import".into());
        }
        Ok(Arc::new(SharedWord {
            memory: Arc::clone(self),
            offset,
        }))
    }
}
impl Drop for SharedMemory {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            self.vk.destroy_imported(memory);
        }
    }
}
pub struct SharedWord {
    pub memory: Arc<SharedMemory>,
    pub offset: u64,
}
impl SharedWord {
    pub(super) fn reset(&self) {
        // SAFETY: used only for a private completion cell with no outstanding GPU/CPU owners.
        unsafe {
            (&*((self.memory.host + self.offset as usize) as *const AtomicU32))
                .store(0, Ordering::Release);
        }
    }
    pub fn load(&self) -> u32 {
        // SAFETY: the constructor checks alignment/range; the import's owner keeps these pages
        // alive. Event storage is a naturally aligned, single-copy atomic four-byte value.
        unsafe {
            (&*((self.memory.host + self.offset as usize) as *const AtomicU32))
                .load(Ordering::Acquire)
        }
    }
    pub fn satisfied(&self, mode: u32, value: u32) -> Result<bool, String> {
        let current = self.load();
        match mode {
            0 => Ok(current == value),
            1 => Ok(current.wrapping_sub(value) < 0x8000_0000),
            _ => Err("unknown event comparison mode".into()),
        }
    }
    pub fn notify(&self) {
        let _guard = self
            .memory
            .changed
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.memory.wake.notify_all();
    }
    pub(super) fn wait(&self, mode: u32, value: u32, stop: &AtomicBool) -> Result<(), String> {
        if self.satisfied(mode, value)? {
            return Ok(());
        }
        let mut guard = self
            .memory
            .changed
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while !self.satisfied(mode, value)? {
            if stop.load(Ordering::Acquire) {
                return Err("native queue retired during an event wait".into());
            }
            // CPU API signals notify immediately. GPU and inlined CPU stores need a bounded
            // observation interval; the value remains in coherent memory, without readback.
            guard = self
                .memory
                .wake
                .wait_timeout(guard, Duration::from_millis(1))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok(())
    }
    pub(super) fn device_address(&self) -> u64 {
        self.memory.address() + self.offset
    }
}

impl Context {
    /// # Safety
    /// The complete page-aligned range must remain allocated at this host address until every
    /// owner of the returned import, including pending GPU submissions, has released it.
    pub unsafe fn shared_memory(&self, host: usize, len: u64) -> Result<Arc<SharedMemory>, String> {
        let memory = self.vk.import_host_memory_flags(
            host as *mut u8,
            len,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        Ok(Arc::new(SharedMemory {
            vk: Arc::clone(&self.vk),
            memory: Some(memory),
            host,
            len,
            changed: Mutex::new(()),
            wake: Condvar::new(),
            _owner: None,
        }))
    }
    pub fn host_import_alignment(&self) -> u64 {
        self.vk.caps.min_imported_host_pointer_alignment.max(1)
    }
    pub(super) fn owned_shared_memory(&self, len: u64) -> Result<Arc<SharedMemory>, String> {
        let align = self.host_import_alignment().max(4096);
        let size = len
            .checked_add(align - 1)
            .ok_or("private shared allocation overflows")?
            / align
            * align;
        let layout = std::alloc::Layout::from_size_align(size as usize, align as usize)
            .map_err(|e| format!("private shared allocation: {e}"))?;
        // SAFETY: private aligned storage, attached to the import's last owner below.
        let host = unsafe { std::alloc::alloc_zeroed(layout) } as usize;
        if host == 0 {
            return Err("allocating private event completion storage failed".into());
        }
        let allocation = Allocation { host, layout };
        // SAFETY: Allocation is retained by the import and released after all GPU uses.
        let mut memory = unsafe { self.shared_memory(host, size) }?;
        Arc::get_mut(&mut memory)
            .expect("new private import")
            ._owner = Some(allocation);
        Ok(memory)
    }
}
