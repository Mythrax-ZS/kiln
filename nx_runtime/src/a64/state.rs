//! Guest CPU state shared by the interpreter and the generated code.
//!
//! The layout is `repr(C)` and part of the ABI with generated code: the code generator addresses
//! fields by byte offset, and `tests` below pin those offsets.

#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuState {
    /// X0 to X30.
    pub x: [u64; 31],
    pub sp: u64,
    /// Address of the instruction about to execute (the interpreter) or the address control
    /// returns to (generated code).
    pub pc: u64,
    /// PSTATE.NZCV in bits 31 to 28.
    pub nzcv: u32,
    pub fpcr: u32,
    pub fpsr: u32,
    pub reserved: u32,
    /// V0 to V31 as (low, high) 64-bit halves.
    pub v: [[u64; 2]; 32],
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    /// Local exclusive monitor: the monitored address and its size in bytes (0 when open).
    pub excl_addr: u64,
    pub excl_size: u64,
}

/// Byte offsets of the fields generated code addresses.
pub mod offsets {
    use super::CpuState;
    use std::mem::offset_of;

    pub const X: usize = offset_of!(CpuState, x);
    pub const SP: usize = offset_of!(CpuState, sp);
    pub const PC: usize = offset_of!(CpuState, pc);
    pub const NZCV: usize = offset_of!(CpuState, nzcv);
    pub const FPCR: usize = offset_of!(CpuState, fpcr);
    pub const FPSR: usize = offset_of!(CpuState, fpsr);
    pub const V: usize = offset_of!(CpuState, v);
    pub const TPIDR_EL0: usize = offset_of!(CpuState, tpidr_el0);
    pub const TPIDRRO_EL0: usize = offset_of!(CpuState, tpidrro_el0);
    pub const EXCL_ADDR: usize = offset_of!(CpuState, excl_addr);
    pub const EXCL_SIZE: usize = offset_of!(CpuState, excl_size);
    pub const SIZE: usize = std::mem::size_of::<CpuState>();
}

impl CpuState {
    /// Register `n` where 31 reads as zero (XZR).
    pub fn x_zr(&self, n: u8) -> u64 {
        if n == 31 {
            0
        } else {
            self.x[usize::from(n)]
        }
    }

    /// Write register `n` where 31 discards the write (XZR).
    pub fn set_x_zr(&mut self, n: u8, v: u64) {
        if n != 31 {
            self.x[usize::from(n)] = v;
        }
    }

    /// Register `n` where 31 is the stack pointer.
    pub fn x_sp(&self, n: u8) -> u64 {
        if n == 31 {
            self.sp
        } else {
            self.x[usize::from(n)]
        }
    }

    /// Write register `n` where 31 is the stack pointer.
    pub fn set_x_sp(&mut self, n: u8, v: u64) {
        if n == 31 {
            self.sp = v;
        } else {
            self.x[usize::from(n)] = v;
        }
    }

    /// The flags as N, Z, C, V in bits 3 to 0.
    pub fn nzcv4(&self) -> u8 {
        (self.nzcv >> 28) as u8
    }

    pub fn set_nzcv4(&mut self, flags: u8) {
        self.nzcv = u32::from(flags & 0xF) << 28;
    }

    pub fn carry(&self) -> bool {
        self.nzcv & (1 << 29) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_31_semantics() {
        let mut c = CpuState {
            sp: 0x1000,
            ..CpuState::default()
        };
        c.x[3] = 7;
        assert_eq!(c.x_zr(31), 0);
        assert_eq!(c.x_sp(31), 0x1000);
        assert_eq!(c.x_zr(3), 7);
        c.set_x_zr(31, 5);
        assert_eq!(c.sp, 0x1000);
        c.set_x_sp(31, 0x2000);
        assert_eq!(c.sp, 0x2000);
        c.set_x_zr(30, 9);
        assert_eq!(c.x[30], 9);
    }

    #[test]
    fn flags_round_trip() {
        let mut c = CpuState::default();
        c.set_nzcv4(0b1001);
        assert_eq!(c.nzcv, 0x9000_0000);
        assert_eq!(c.nzcv4(), 0b1001);
        assert!(!c.carry());
        c.set_nzcv4(0b0010);
        assert!(c.carry());
        c.set_nzcv4(0xFF);
        assert_eq!(c.nzcv4(), 0xF);
    }

    #[test]
    fn layout_is_pinned_for_generated_code() {
        assert_eq!(offsets::X, 0);
        assert_eq!(offsets::SP, 31 * 8);
        assert_eq!(offsets::PC, 32 * 8);
        assert_eq!(offsets::NZCV, 33 * 8);
        assert_eq!(offsets::FPCR, 33 * 8 + 4);
        assert_eq!(offsets::FPSR, 34 * 8);
        assert_eq!(offsets::V, 35 * 8);
        assert_eq!(offsets::TPIDR_EL0, 35 * 8 + 32 * 16);
        assert_eq!(offsets::TPIDRRO_EL0, offsets::TPIDR_EL0 + 8);
        assert_eq!(offsets::EXCL_ADDR, offsets::TPIDR_EL0 + 16);
        assert_eq!(offsets::EXCL_SIZE, offsets::TPIDR_EL0 + 24);
        assert_eq!(offsets::SIZE, offsets::EXCL_SIZE + 8);
        assert_eq!(std::mem::align_of::<CpuState>(), 8);
    }
}
