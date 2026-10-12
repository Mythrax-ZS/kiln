//! Shared memory: pages a service owns and the process maps (`MapSharedMemory`, Switchbrew SVC
//! page).
//!
//! A service creates the object, fills its pages through the address
//! [`Kernel::create_shared_memory`] gives it (inside a region the process cannot see), and hands the
//! process a handle. The process maps the object at an address of its choosing and sees the same
//! bytes the service writes. This is how `hid` publishes input to a program
//! (docs/ASSUMPTIONS.md A-074).

use super::handle::Handle;
use super::kernel::Kernel;
use super::layout::{mem_type, ASLR_BASE, KERNEL_SCRATCH_BASE, KERNEL_SCRATCH_SIZE, PAGE};
use super::memory::Info;
use super::object::KObject;
use super::result::{kernel as err, KResult, ResultCode};
use crate::rt::abi::perm;
use crate::rt::runtime::rt;

/// One shared memory object.
#[derive(Debug)]
pub(super) struct SharedBlock {
    /// Where the service reaches the pages.
    pub addr: u64,
    pub size: u64,
    /// The ranges of the process it is mapped at.
    pub mapped: Vec<u64>,
}

impl Kernel {
    /// A shared memory object of `size` bytes, zeroed, and the address the service writes it at.
    pub fn create_shared_memory(&self, size: u64) -> KResult<(KObject, u64)> {
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(err::INVALID_SIZE);
        }
        let mut st = self.lock();
        let addr = st.kernel_cursor.max(KERNEL_SCRATCH_BASE);
        if addr
            .checked_add(size)
            .is_none_or(|e| e > KERNEL_SCRATCH_BASE + KERNEL_SCRATCH_SIZE)
        {
            return Err(err::OUT_OF_MEMORY);
        }
        rt().mem
            .map(addr, size, perm::R | perm::W)
            .map_err(|_| err::OUT_OF_MEMORY)?;
        st.mapped.push((addr, size));
        st.kernel_cursor = addr + size;
        let index = st.shared.len();
        st.shared.push(SharedBlock {
            addr,
            size,
            mapped: Vec::new(),
        });
        Ok((KObject::Shared(index), addr))
    }

    /// The address and size a shared memory object is reachable at, for the service.
    pub fn shared_memory_range(&self, object: &KObject) -> Option<(u64, u64)> {
        let KObject::Shared(i) = object else {
            return None;
        };
        self.lock().shared.get(*i).map(|b| (b.addr, b.size))
    }

    /// `MapSharedMemory(handle, addr, size, permission)`.
    pub fn map_shared_memory(
        &self,
        current: u64,
        handle: Handle,
        addr: u64,
        size: u64,
        permission: u32,
    ) -> ResultCode {
        let Some(KObject::Shared(index)) = self.object(handle, current) else {
            return err::INVALID_HANDLE;
        };
        if !addr.is_multiple_of(PAGE) {
            return err::INVALID_ADDRESS;
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return err::INVALID_SIZE;
        }
        // Read, or read and write (Switchbrew).
        if ![perm::R, perm::R | perm::W].contains(&permission) {
            return err::INVALID_NEW_MEMORY_PERMISSION;
        }
        let mut st = self.lock();
        let Some(block) = st.shared.get(index) else {
            return err::INVALID_HANDLE;
        };
        if size != block.size {
            return err::INVALID_SIZE;
        }
        let source = block.addr;
        if addr < ASLR_BASE || addr.saturating_add(size) > KERNEL_SCRATCH_BASE {
            return err::INVALID_MEMORY_REGION;
        }
        if !st.memory.blocks_in(addr, size).is_empty() {
            return err::INVALID_CURRENT_MEMORY;
        }
        if rt().mem.alias(addr, source, size, permission).is_err() {
            return err::INVALID_CURRENT_MEMORY;
        }
        st.memory.insert(
            addr,
            size,
            Info {
                mem_type: mem_type::SHARED,
                perm: permission,
                attr: 0,
            },
        );
        st.shared[index].mapped.push(addr);
        ResultCode::SUCCESS
    }

    /// `UnmapSharedMemory(handle, addr, size)`.
    pub fn unmap_shared_memory(
        &self,
        current: u64,
        handle: Handle,
        addr: u64,
        size: u64,
    ) -> ResultCode {
        let Some(KObject::Shared(index)) = self.object(handle, current) else {
            return err::INVALID_HANDLE;
        };
        if !addr.is_multiple_of(PAGE) {
            return err::INVALID_ADDRESS;
        }
        let mut st = self.lock();
        let Some(block) = st.shared.get(index) else {
            return err::INVALID_HANDLE;
        };
        if size != block.size {
            return err::INVALID_SIZE;
        }
        let Some(at) = block.mapped.iter().position(|&a| a == addr) else {
            return err::INVALID_CURRENT_MEMORY;
        };
        let _ = rt().mem.unmap(addr, size);
        st.memory.remove(addr, size);
        st.shared[index].mapped.swap_remove(at);
        ResultCode::SUCCESS
    }
}
