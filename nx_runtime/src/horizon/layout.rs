//! Where the parts of the guest process live, and the memory types `QueryMemory` reports.
//!
//! The regions are inside the 39-bit address space and 2 MiB aligned; their placement is ours
//! (the real kernel randomizes it) and is recorded in docs/ASSUMPTIONS.md A-034.

pub const PAGE: u64 = 0x1000;
pub const ADDRESS_SPACE_END: u64 = 1 << 39;

/// The first module is mapped here (docs/ASSUMPTIONS.md A-006).
pub const CODE_BASE: u64 = 0x0800_0000;

/// The region modules and other mappings may use (`GetInfo` 12 and 13).
pub const ASLR_BASE: u64 = 0x0800_0000;
pub const ASLR_SIZE: u64 = ADDRESS_SPACE_END - ASLR_BASE;

/// The alias region (`GetInfo` 2 and 3): 64 GiB.
pub const ALIAS_BASE: u64 = 0x20_0000_0000;
pub const ALIAS_SIZE: u64 = 0x10_0000_0000;

/// The heap region (`GetInfo` 4 and 5): 6 GiB. `SetHeapSize` always places the heap at its start.
pub const HEAP_BASE: u64 = 0x30_0000_0000;
pub const HEAP_REGION_SIZE: u64 = 0x1_8000_0000;

/// The stack region (`GetInfo` 14 and 15): 2 GiB. `MapMemory` may map only here.
pub const STACK_BASE: u64 = 0x38_0000_0000;
pub const STACK_REGION_SIZE: u64 = 0x8000_0000;

/// The main thread's stack ends here, one megabyte below the end of the stack region.
pub const MAIN_STACK_TOP: u64 = STACK_BASE + STACK_REGION_SIZE - 0x10_0000;

/// Thread-local regions: 0x200 bytes each, eight to a page, the first slot of a page unused (the
/// slot the real kernel keeps for user exception handling).
pub const TLS_BASE: u64 = 0x3C_0000_0000;
pub const TLS_REGION_SIZE: u64 = 0x4000_0000;
pub const TLS_SLOT_SIZE: u64 = 0x200;

/// Memory types (the low byte of the memory state), as `QueryMemory` reports them.
pub mod mem_type {
    pub const FREE: u8 = 0x00;
    pub const IO: u8 = 0x01;
    pub const STATIC: u8 = 0x02;
    pub const CODE: u8 = 0x03;
    pub const CODE_DATA: u8 = 0x04;
    pub const NORMAL: u8 = 0x05;
    pub const SHARED: u8 = 0x06;
    pub const ALIAS: u8 = 0x07;
    pub const ALIAS_CODE: u8 = 0x08;
    pub const ALIAS_CODE_DATA: u8 = 0x09;
    pub const IPC: u8 = 0x0A;
    pub const STACK: u8 = 0x0B;
    pub const THREAD_LOCAL: u8 = 0x0C;
    pub const TRANSFERED: u8 = 0x0D;
    pub const SHARED_TRANSFERED: u8 = 0x0E;
    pub const SHARED_CODE: u8 = 0x0F;
    pub const INACCESSIBLE: u8 = 0x10;
}

/// Memory attribute bits.
pub mod attr {
    /// Set on the source of a `MapMemory` while it is mapped elsewhere.
    pub const LOCKED: u32 = 1;
    pub const IPC_LOCKED: u32 = 2;
    pub const DEVICE_SHARED: u32 = 4;
    pub const UNCACHED: u32 = 8;
}

/// Pages the kernel layer keeps for objects the program cannot see (the backing of shared memory
/// a service fills): the last 4 GiB below the end of the address space.
pub const KERNEL_SCRATCH_BASE: u64 = ADDRESS_SPACE_END - KERNEL_SCRATCH_SIZE;
pub const KERNEL_SCRATCH_SIZE: u64 = 0x1_0000_0000;
