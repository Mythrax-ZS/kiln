//! Memory objects: the pieces of a process's memory it hands to services.
//!
//! A *transfer memory* is a range of the process's own memory that it lends to a service
//! (`CreateTransferMemory`, Switchbrew SVC page): the range is reprotected with the permission the
//! process asked for and marked locked until the last handle is closed, and then it goes back to
//! normal. The services here run in the process, so they read and write the range through the same
//! guest addresses; what the lending changes is what the program itself may do meanwhile.

use super::kernel::Kernel;
use super::layout::{attr, mem_type, PAGE};
use super::memory::Info;
use super::object::KObject;
use super::result::{kernel as err, KResult};
use crate::rt::abi::perm;
use crate::rt::runtime::rt;

#[derive(Clone, Copy, Debug)]
pub(super) struct TransferMemory {
    pub addr: u64,
    pub size: u64,
    /// The program closed its handle and the range is its own again.
    pub released: bool,
}

impl Kernel {
    /// `CreateTransferMemory(addr, size, permission)`.
    pub fn create_transfer_memory(&self, addr: u64, size: u64, permission: u32) -> KResult<u32> {
        if !addr.is_multiple_of(PAGE) {
            return Err(err::INVALID_ADDRESS);
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Err(err::INVALID_SIZE);
        }
        // None, read or read and write; execute is not allowed.
        if ![0, perm::R, perm::R | perm::W].contains(&permission) {
            return Err(err::INVALID_NEW_MEMORY_PERMISSION);
        }
        let mut st = self.lock();
        let blocks = st.memory.blocks_in(addr, size);
        let covered: u64 = blocks.iter().map(|b| b.1).sum();
        if covered != size
            || blocks.iter().any(|(_, _, i)| {
                // The heap and the data and `.bss` of a module (A-174).
                !matches!(
                    i.mem_type,
                    mem_type::NORMAL | mem_type::CODE_DATA | mem_type::ALIAS_CODE_DATA
                ) || i.attr != 0
                    || i.perm != perm::R | perm::W
            })
        {
            return Err(err::INVALID_CURRENT_MEMORY);
        }
        let index = st.transfer_memories.len();
        let handle = st.handles.insert(KObject::Memory(index))?;
        if rt().mem.protect(addr, size, permission).is_err() {
            let _ = st.handles.close(handle);
            return Err(err::INVALID_CURRENT_MEMORY);
        }
        st.memory.update(addr, size, |i| Info {
            perm: permission,
            attr: i.attr | attr::LOCKED,
            ..i
        });
        st.transfer_memories.push(TransferMemory {
            addr,
            size,
            released: false,
        });
        Ok(handle)
    }

    /// The range a transfer memory object lends, for the service that received it.
    pub fn transfer_memory(&self, object: &KObject) -> Option<(u64, u64)> {
        let KObject::Memory(i) = object else {
            return None;
        };
        let st = self.lock();
        st.transfer_memories
            .get(*i)
            .filter(|t| !t.released)
            .map(|t| (t.addr, t.size))
    }

    /// The program closed a handle to a memory object: the lent range is the program's again.
    pub(super) fn release_memory_object(&self, object: &KObject) {
        let KObject::Memory(i) = object else { return };
        let mut st = self.lock();
        let Some(t) = st.transfer_memories.get(*i).copied() else {
            return;
        };
        if t.released {
            return;
        }
        st.transfer_memories[*i].released = true;
        let _ = rt().mem.protect(t.addr, t.size, perm::R | perm::W);
        st.memory.update(t.addr, t.size, |i| Info {
            perm: perm::R | perm::W,
            attr: i.attr & !attr::LOCKED,
            ..i
        });
    }
}
