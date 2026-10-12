//! The guest-level control-flow graph of one function, as discovery finds it and the lifter
//! consumes it.

use nx_runtime::a64::inst::{Inst, SysReg};
use std::collections::BTreeMap;

/// Where a direct branch goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// A block of this function.
    Block(u64),
    /// The entry of another function: a tail call.
    Tail(u64),
}

/// How a block ends. The terminating instruction, when there is one, is the last of `insts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum End {
    /// Runs into the block at this address (a leader splits straight-line code).
    Fall(u64),
    /// `B`.
    Jump(Target),
    /// `B.cond`, `CBZ`, `CBNZ`, `TBZ`, `TBNZ`; `fall` is the next instruction.
    Cond { taken: Target, fall: u64 },
    /// `BL`, `BLR`, `SVC`, or an instruction the interpreter executes: control comes back to the
    /// next instruction, which starts the block at `ret`.
    Call { ret: u64 },
    /// `RET`.
    Return,
    /// `BR`. `targets` lists the destinations a jump-table pattern found; anything else the
    /// register may hold goes through the function table.
    Indirect { targets: Vec<u64> },
    /// The last instruction always stops the run (`BRK`, an undefined encoding, a system access
    /// EL0 cannot make).
    Trap,
    /// The straight-line code reaches an address that is not executable.
    OutOfCode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestBlock {
    pub addr: u64,
    pub insts: Vec<(u64, Inst)>,
    pub end: End,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncCfg {
    pub entry: u64,
    pub blocks: BTreeMap<u64, GuestBlock>,
}

impl FuncCfg {
    pub fn instruction_count(&self) -> usize {
        self.blocks.values().map(|b| b.insts.len()).sum()
    }
}

/// The system registers `MSR` may write at EL0 in this model.
pub fn msr_allowed(reg: SysReg) -> bool {
    matches!(
        reg,
        SysReg::Nzcv | SysReg::Fpcr | SysReg::Fpsr | SysReg::TpidrEl0
    )
}

/// The system registers `MRS` can read at EL0 in this model.
pub fn mrs_allowed(reg: SysReg) -> bool {
    !matches!(reg, SysReg::Other(..))
}

/// `SYS` operations the model implements: `DC ZVA`, `DC CVAC/CVAU/CIVAC` and `IC IVAU`.
pub fn sys_allowed(load: bool, op1: u8, crn: u8, crm: u8, op2: u8) -> bool {
    matches!(
        (load, op1, crn, crm, op2),
        (false, 3, 7, 4, 1)
            | (false, 3, 7, 10, 1)
            | (false, 3, 7, 11, 1)
            | (false, 3, 7, 14, 1)
            | (false, 3, 7, 5, 1)
    )
}

/// Does this instruction always stop the run? Shared by discovery (it ends a block) and the
/// lifter (it becomes a trap exit).
pub fn traps(inst: &Inst) -> Option<super::ir::TrapKind> {
    use super::ir::TrapKind;
    match *inst {
        Inst::Unallocated(w) => Some(TrapKind::Undefined(w)),
        Inst::Brk { imm } => Some(TrapKind::Brk(imm)),
        Inst::Mrs { reg, .. } if !mrs_allowed(reg) => Some(TrapKind::sysreg(reg, false)),
        Inst::Msr { reg, .. } if !msr_allowed(reg) => Some(TrapKind::sysreg(reg, true)),
        Inst::Sys {
            load,
            op1,
            crn,
            crm,
            op2,
            ..
        } if !sys_allowed(load, op1, crn, crm, op2) => Some(TrapKind::SysInst {
            load,
            enc: [op1, crn, crm, op2],
        }),
        _ => None,
    }
}
