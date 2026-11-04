//! What a Switch 2 process sees differently from a Switch one, decided once from its NPDM
//! (docs/ASSUMPTIONS.md A-188): the counter's rate, the heap region and the memory a program may
//! use, and the information types only the Switch 2's kernel answers (A-190).

use super::layout;
use super::result::{kernel as err, ResultCode};
use crate::a64::bus::{COUNTER_FREQUENCY, COUNTER_FREQUENCY_NX2};
use crate::npdm::Platform;

/// The counter's rate, `CNTFRQ_EL0`.
pub fn counter_frequency(p: Platform) -> u64 {
    match p {
        Platform::Nx => COUNTER_FREQUENCY,
        Platform::Nx2 => COUNTER_FREQUENCY_NX2,
    }
}

/// The cores a program's threads may run on, `0..cores`: four on the Switch, six on the Switch 2
/// (whose NPDMs name cores 0 to 5).
pub fn cores(p: Platform) -> i32 {
    match p {
        Platform::Nx => 4,
        Platform::Nx2 => 6,
    }
}

/// The heap region at [`layout::HEAP_BASE`] (`GetInfo` 4 and 5): 6 GiB on the Switch, 10 GiB on
/// the Switch 2, more than all the memory such a program may use, so that every heap it can have
/// fits (it still ends below the stack region).
pub fn heap_region_size(p: Platform) -> u64 {
    match p {
        Platform::Nx => layout::HEAP_REGION_SIZE,
        Platform::Nx2 => 0x2_8000_0000,
    }
}

/// The largest heap `SetHeapSize` takes.
pub fn max_heap_size(p: Platform) -> u64 {
    match p {
        Platform::Nx => 1 << 32,
        Platform::Nx2 => heap_region_size(p),
    }
}

/// The memory a program may use (`GetInfo` 6 and 21): the application pool.
pub fn total_memory_size(p: Platform) -> u64 {
    match p {
        Platform::Nx => super::svc::TOTAL_MEMORY_SIZE,
        Platform::Nx2 => 9809 << 20,
    }
}

/// `GetInfo` 29 to 33, which only the Switch 2's kernel answers: the remote region (the upper half
/// of the alias region) and how much of it is used, and the page size.
pub fn info(p: Platform, id: u32) -> Option<u64> {
    if p != Platform::Nx2 {
        return None;
    }
    Some(match id {
        29 => layout::ALIAS_BASE + layout::ALIAS_SIZE / 2,
        30 => layout::ALIAS_SIZE / 2,
        31 | 32 => 0,
        33 => layout::PAGE,
        _ => return None,
    })
}

/// `SetGpuMemoryAttribute` (SVC 0x80, the Switch 2's): marks `[addr, addr + size)` for the GPU.
/// The GPU here reads guest memory where it is, so there is nothing to change; the arguments
/// are checked as NeXium checks them (only the attribute 0x20 exists).
pub fn set_gpu_memory_attribute(addr: u64, size: u64, mask: u32, attribute: u32) -> ResultCode {
    const GPU_SHARABLE: u32 = 0x20;
    if !addr.is_multiple_of(layout::PAGE) {
        err::INVALID_ADDRESS
    } else if size == 0 || !size.is_multiple_of(layout::PAGE) || addr.checked_add(size).is_none() {
        err::INVALID_SIZE
    } else if attribute & !mask != 0 || (mask | attribute) & !GPU_SHARABLE != 0 {
        err::INVALID_COMBINATION
    } else {
        ResultCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_2_regions_fit_the_layout() {
        let heap_end = layout::HEAP_BASE + heap_region_size(Platform::Nx2);
        const { assert!(layout::ALIAS_BASE + layout::ALIAS_SIZE <= layout::HEAP_BASE) };
        assert!(heap_end <= layout::STACK_BASE);
        assert!(total_memory_size(Platform::Nx2) <= heap_region_size(Platform::Nx2));
        assert_eq!(info(Platform::Nx, 29), None);
        assert_eq!(info(Platform::Nx2, 33), Some(0x1000));
        assert_eq!(info(Platform::Nx2, 34), None);
    }

    #[test]
    fn gpu_memory_attribute_arguments() {
        let ok = ResultCode::SUCCESS;
        assert_eq!(set_gpu_memory_attribute(0x1000, 0x2000, 0x20, 0x20), ok);
        assert_eq!(set_gpu_memory_attribute(0x1000, 0x2000, 0x20, 0), ok);
        assert_eq!(
            set_gpu_memory_attribute(0x1001, 0x2000, 0x20, 0x20),
            err::INVALID_ADDRESS
        );
        assert_eq!(
            set_gpu_memory_attribute(0x1000, 0, 0x20, 0x20),
            err::INVALID_SIZE
        );
        let combination = err::INVALID_COMBINATION;
        assert_eq!(
            set_gpu_memory_attribute(0x1000, 0x2000, 0, 0x20),
            combination
        );
        assert_eq!(
            set_gpu_memory_attribute(0x1000, 0x2000, 0x8, 0x8),
            combination
        );
    }
}
