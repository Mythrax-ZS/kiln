//! The process handle table.
//!
//! A handle is a 15-bit table index plus a 15-bit generation counter above it, so a closed handle
//! stops working when its slot is reused, and the value stays below `1 << 30`: guest mutexes keep
//! thread handles in a word whose bit 30 means "has waiters". Two pseudo handles name the calling
//! thread and the process without a table slot.

use super::result::{kernel, KResult};

pub type Handle = u32;

/// Names the calling thread in thread calls.
pub const CURRENT_THREAD: Handle = 0xFFFF_8000;
/// Names the process in process calls.
pub const CURRENT_PROCESS: Handle = 0xFFFF_8001;

const INDEX_BITS: u32 = 15;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const GENERATION_MASK: u32 = (1 << INDEX_BITS) - 1;

/// Slots in the table (the real limit comes from the process's kernel capabilities).
pub const CAPACITY: usize = 1024;

struct Slot<T> {
    generation: u32,
    object: Option<T>,
}

pub struct HandleTable<T> {
    slots: Vec<Slot<T>>,
    free: Vec<usize>,
}

impl<T> Default for HandleTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HandleTable<T> {
    pub fn new() -> HandleTable<T> {
        HandleTable {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    fn handle_of(index: usize, generation: u32) -> Handle {
        (generation << INDEX_BITS) | index as u32
    }

    pub fn insert(&mut self, object: T) -> KResult<Handle> {
        let index = match self.free.pop() {
            Some(i) => i,
            None if self.slots.len() < CAPACITY => {
                self.slots.push(Slot {
                    generation: 0,
                    object: None,
                });
                self.slots.len() - 1
            }
            None => return Err(kernel::OUT_OF_HANDLES),
        };
        let slot = &mut self.slots[index];
        // Generation zero is never used, so handle zero is never valid.
        slot.generation = slot.generation % GENERATION_MASK + 1;
        slot.object = Some(object);
        Ok(Self::handle_of(index, slot.generation))
    }

    fn slot_of(&self, h: Handle) -> Option<usize> {
        let index = (h & INDEX_MASK) as usize;
        let slot = self.slots.get(index)?;
        (slot.object.is_some() && slot.generation == h >> INDEX_BITS).then_some(index)
    }

    pub fn get(&self, h: Handle) -> Option<&T> {
        self.slot_of(h).and_then(|i| self.slots[i].object.as_ref())
    }

    pub fn get_mut(&mut self, h: Handle) -> Option<&mut T> {
        let i = self.slot_of(h)?;
        self.slots[i].object.as_mut()
    }

    /// Close a handle and return the object it named.
    pub fn close(&mut self, h: Handle) -> KResult<T> {
        let i = self.slot_of(h).ok_or(kernel::INVALID_HANDLE)?;
        self.free.push(i);
        Ok(self.slots[i].object.take().expect("slot_of checked"))
    }

    /// How many handles are open.
    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_are_nonzero_and_stay_below_bit_30() {
        let mut t = HandleTable::new();
        let hs: Vec<Handle> = (0..100).map(|i| t.insert(i).unwrap()).collect();
        assert!(hs.iter().all(|&h| h != 0 && h < 1 << 30));
        assert_eq!(*t.get(hs[7]).unwrap(), 7);
        assert_eq!(t.len(), 100);
    }

    #[test]
    fn a_closed_handle_does_not_name_the_next_object_in_its_slot() {
        let mut t = HandleTable::new();
        let a = t.insert("a").unwrap();
        assert_eq!(t.close(a), Ok("a"));
        let b = t.insert("b").unwrap();
        assert_ne!(a, b);
        assert!(t.get(a).is_none());
        assert_eq!(t.close(a), Err(kernel::INVALID_HANDLE));
        assert_eq!(t.get(b), Some(&"b"));
        // The pseudo handles and zero are never in the table.
        assert!(t.get(0).is_none() && t.get(CURRENT_THREAD).is_none());
    }

    #[test]
    fn the_table_is_bounded() {
        let mut t = HandleTable::new();
        for i in 0..CAPACITY {
            t.insert(i).unwrap();
        }
        assert_eq!(t.insert(0), Err(kernel::OUT_OF_HANDLES));
    }
}
