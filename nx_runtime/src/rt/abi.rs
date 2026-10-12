//! The contract between generated code and the runtime.
//!
//! Generated code sees one pointer per guest thread, to a [`NativeCpu`]: the architectural state
//! ([`CpuState`], the registers it keeps in SSA values between calls) followed by [`Ctl`], the
//! bookkeeping the runtime keeps per thread. The code generator addresses fields by the byte
//! offsets in [`offsets`], which the tests pin.
//!
//! # Calling protocol
//!
//! Every generated function has the signature `void f(NativeCpu *cpu)`. On entry `cpu` holds the
//! guest state at the first instruction. On return `cpu` holds the full state and `cpu->pc` is the
//! guest address where execution continues. A call site continues only if `cpu->pc` is the return
//! address it expects; otherwise it returns at once, so a function that returns somewhere else
//! (a `longjmp`, a coroutine switch) unwinds the native stack to the frame that can take it, or to
//! the run loop, which dispatches through the function table.
//!
//! A stop (fault, trap, unhandled supervisor call) records its details in [`Ctl`] and sets
//! `cpu->pc` to [`STOP_PC`], which no call site expects, so the whole native stack unwinds.

use crate::a64::state::CpuState;
use std::ffi::c_void;
use std::mem::offset_of;

/// The address a run returns to when the entry function returns (the initial `X30`).
pub const EXIT_PC: u64 = 0xFFFF_FFFF_FFFF_E000;
/// `cpu->pc` after a stop. Matches no return address.
pub const STOP_PC: u64 = 0xFFFF_FFFF_FFFF_F000;

/// Why a run stopped.
pub mod stop {
    pub const NONE: u32 = 0;
    /// A guest memory access failed.
    pub const FAULT: u32 = 1;
    /// An instruction that stops the run: see [`super::trap`].
    pub const TRAP: u32 = 2;
    /// A supervisor call nothing handled; `stop_pc` is the `SVC` instruction, `trap_aux[0]` its
    /// immediate.
    pub const SVC: u32 = 3;
    /// Control reached an address with no code the interpreter can run.
    pub const NO_CODE: u32 = 4;
    /// The slow path gave up (step limit).
    pub const HALTED: u32 = 5;
}

/// Kinds of [`stop::TRAP`], with what `trap_aux` holds.
pub mod trap {
    /// An unallocated encoding: `aux[0]` is the word.
    pub const UNDEFINED: u32 = 1;
    /// A valid encoding the project does not implement: `aux[0]` is the word.
    pub const UNSUPPORTED: u32 = 2;
    /// `BRK`: `aux[0]` is the immediate.
    pub const BRK: u32 = 3;
    /// A system register access: `aux[0]` packs `op0:op1:CRn:CRm:op2` as `op0 << 14 | op1 << 11 |
    /// CRn << 7 | CRm << 3 | op2`, `aux[1]` is 1 for a write.
    pub const SYSREG: u32 = 4;
    /// A `SYS`/`SYSL` operation: `aux[0]` packs `op1 << 11 | CRn << 7 | CRm << 3 | op2`, `aux[1]`
    /// is 1 for `SYSL`.
    pub const SYSINST: u32 = 5;
    /// Execution reached a non-executable address (`stop_pc` is the address).
    pub const OUT_OF_CODE: u32 = 6;
}

/// Per-thread runtime bookkeeping that generated code reads or writes.
#[repr(C)]
pub struct Ctl {
    /// Non-zero after a failed memory access, until the run unwinds. Generated code tests it after
    /// every access.
    pub fault_flag: u32,
    /// A [`stop`] reason; zero while running.
    pub stop: u32,
    /// `FaultKind` as 0 unmapped, 1 permission, 2 alignment.
    pub fault_kind: u32,
    pub fault_write: u32,
    pub fault_addr: u64,
    /// Guest address of the instruction that stopped the run.
    pub stop_pc: u64,
    pub trap_kind: u32,
    pub trap_pad: u32,
    pub trap_aux: [u64; 2],
    /// Second half of a 128-bit exclusive load.
    pub aux: u64,
    /// Identifies the thread to the exclusive monitor.
    pub thread_id: u64,
    /// Owned by the runtime.
    pub user: *mut c_void,
}

/// What generated functions receive: the architectural state, then the bookkeeping.
#[repr(C)]
pub struct NativeCpu {
    pub state: CpuState,
    pub ctl: Ctl,
}

impl NativeCpu {
    pub fn new(thread_id: u64) -> NativeCpu {
        NativeCpu {
            state: CpuState::default(),
            ctl: Ctl {
                fault_flag: 0,
                stop: stop::NONE,
                fault_kind: 0,
                fault_write: 0,
                fault_addr: 0,
                stop_pc: 0,
                trap_kind: 0,
                trap_pad: 0,
                trap_aux: [0; 2],
                aux: 0,
                thread_id,
                user: std::ptr::null_mut(),
            },
        }
    }
}

/// Byte offsets from the `NativeCpu` pointer.
pub mod offsets {
    use super::*;
    pub const FAULT_FLAG: usize = offset_of!(NativeCpu, ctl.fault_flag);
    pub const STOP: usize = offset_of!(NativeCpu, ctl.stop);
    pub const AUX: usize = offset_of!(NativeCpu, ctl.aux);
    pub const SIZE: usize = std::mem::size_of::<NativeCpu>();
}

/// One entry of the function table a module registers: sorted by guest address.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FuncEntry {
    pub addr: u64,
    pub func: *const c_void,
}

// SAFETY: the table only holds addresses of immutable generated code.
unsafe impl Send for FuncEntry {}
unsafe impl Sync for FuncEntry {}

/// Page permissions for [`super::mem::GuestMemory`].
pub mod perm {
    pub const R: u32 = 1;
    pub const W: u32 = 2;
    pub const X: u32 = 4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_pinned_for_generated_code() {
        let state = std::mem::size_of::<CpuState>();
        assert_eq!(offset_of!(NativeCpu, state), 0);
        assert_eq!(offset_of!(NativeCpu, ctl), state);
        assert_eq!(offsets::FAULT_FLAG, state);
        assert_eq!(offsets::STOP, state + 4);
        assert_eq!(offsets::AUX, state + offset_of!(Ctl, aux));
        assert_eq!(std::mem::align_of::<NativeCpu>(), 8);
    }

    #[test]
    fn sentinels_are_not_guest_code_addresses() {
        // The guest address space is 39 bits wide.
        const { assert!(EXIT_PC >> 39 != 0 && STOP_PC >> 39 != 0) };
        assert_ne!(EXIT_PC, STOP_PC);
    }
}
