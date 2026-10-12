//! What the interpreter needs from its surroundings: memory, the exclusive monitor, and the few
//! system registers that live outside the CPU state.

use super::inst::SysReg;
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
