//! What the interpreter needs from its surroundings: memory, the exclusive monitor, and the few
//! system registers that live outside the CPU state.

use super::inst::{AtomicOp, SysReg};
use super::state::CpuState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind {
    Unmapped,
    Permission,
    Alignment,
}

/// A failed guest memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    pub addr: u64,
    pub write: bool,
    pub kind: FaultKind,
}

/// Counter frequency reported by `CNTFRQ_EL0` (docs/ASSUMPTIONS.md A-001).
pub const COUNTER_FREQUENCY: u64 = 19_200_000;
/// Cortex-A57 value of `CTR_EL0`.
pub const CTR_EL0: u64 = 0x8444_C004;
/// `DCZID_EL0`: `DC ZVA` block size is 4 words squared, 64 bytes, and the instruction is allowed.
pub const DCZID_EL0: u64 = 4;
/// Bytes zeroed by `DC ZVA`.
pub const ZVA_BLOCK: u64 = 64;

/// Memory and system services for the interpreter. Sizes are in bytes (1, 2, 4, 8 or 16) and
/// values are little-endian.
pub trait Bus {
    fn read(&mut self, addr: u64, size: u8) -> Result<u128, Fault>;
    fn write(&mut self, addr: u64, size: u8, value: u128) -> Result<(), Fault>;

    fn fetch(&mut self, addr: u64) -> Result<u32, Fault> {
        self.read(addr, 4).map(|v| v as u32)
    }

    /// `LDXR` family: read and arm the exclusive monitor for `[addr, addr + size)`.
    fn load_exclusive(&mut self, cpu: &mut CpuState, addr: u64, size: u8) -> Result<u128, Fault> {
        let v = self.read(addr, size)?;
        cpu.excl_addr = addr;
        cpu.excl_size = u64::from(size);
        Ok(v)
    }

    /// `STXR` family: store if the monitor is still armed for this exact range. Returns whether
    /// the store happened. The monitor is open afterwards either way.
    fn store_exclusive(
        &mut self,
        cpu: &mut CpuState,
        addr: u64,
        size: u8,
        value: u128,
    ) -> Result<bool, Fault> {
        let armed = cpu.excl_size == u64::from(size) && cpu.excl_addr == addr;
        cpu.excl_size = 0;
        if armed {
            self.write(addr, size, value)?;
        }
        Ok(armed)
    }

    /// `CLREX`.
    fn clear_exclusive(&mut self, cpu: &mut CpuState) {
        cpu.excl_size = 0;
    }

    /// An LSE read-modify-write of `size` bytes (1 to 8) at an aligned `addr`, which memory
    /// observes as one access: returns what memory held. The access counts as a write for its
    /// faults. This default is for buses that one thread uses; a shared one overrides it.
    fn atomic(&mut self, addr: u64, size: u8, op: AtomicOp, operand: u64) -> Result<u64, Fault> {
        let as_write = |f: Fault| Fault { write: true, ..f };
        let old = self.read(addr, size).map_err(as_write)? as u64;
        self.write(
            addr,
            size,
            u128::from(op.apply(old, operand, 8 * u32::from(size))),
        )?;
        Ok(old)
    }

    /// `CAS`/`CASP` of `size` bytes (1 to 16) at an aligned `addr`: `new` replaces `expected` if
    /// memory holds it; returns what memory held. A compare that fails still counts as a write
    /// for faults (the Arm pseudocode's atomic access); this default writes the old bytes back.
    fn compare_swap(
        &mut self,
        addr: u64,
        size: u8,
        expected: u128,
        new: u128,
    ) -> Result<u128, Fault> {
        let as_write = |f: Fault| Fault { write: true, ..f };
        let old = self.read(addr, size).map_err(as_write)?;
        self.write(addr, size, if old == expected { new } else { old })?;
        Ok(old)
    }

    /// The system counter behind `CNTPCT_EL0` and `CNTVCT_EL0`.
    fn counter(&mut self) -> u64 {
        0
    }

    /// Read a system register that is not part of [`CpuState`]. `None` means the register is not
    /// implemented, which stops the run instead of returning an invented value.
    fn system_register(&mut self, reg: SysReg) -> Option<u64> {
        match reg {
            SysReg::CntfrqEl0 => Some(COUNTER_FREQUENCY),
            SysReg::CntpctEl0 | SysReg::CntvctEl0 => Some(self.counter()),
            SysReg::CtrEl0 => Some(CTR_EL0),
            SysReg::DczidEl0 => Some(DCZID_EL0),
            _ => None,
        }
    }
}
