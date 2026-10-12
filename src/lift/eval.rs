//! A reference evaluator for the IR.
//!
//! The generated code is checked against the interpreter in two ways: this evaluator runs the same
//! IR in-process (fast enough for millions of cases and independent of any toolchain), and the
//! native tests compile the IR with LLVM. The evaluator gives the IR its meaning: pure operations
//! go through [`super::sem`], memory goes through the interpreter's [`Bus`].
//!
//! A memory fault ends the run with [`Stop::Fault`]. Like the native code, the evaluator does not
//! reconstruct the registers of the faulting block, only the faulting access.

use super::ir::*;
use super::sem;
use nx_runtime::a64::bits::crc32_update;
use nx_runtime::a64::bus::{Bus, Fault, FaultKind, ZVA_BLOCK};
use nx_runtime::a64::state::CpuState;

/// How a run of one function ended. For everything but `Fault` the architectural state is in the
/// CPU structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    /// Returned to this address.
    Return(u64),
    /// Tail call to a known function entry.
    Jump(u64),
    /// Tail call through the function table.
    Dispatch(u64),
    Trap {
        kind: TrapKind,
        pc: u64,
    },
    /// A call-like terminator: the caller performs it and may continue in `cont`.
    Call {
        target: CallTarget,
        /// For an indirect call, the target address.
        target_pc: u64,
        pc: u64,
        ret_pc: u64,
        cont: BlockId,
    },
    Fault {
        fault: Fault,
        pc: u64,
    },
    /// The block budget ran out (an endless loop in the program under test).
    OutOfFuel,
}

pub struct Evaluator<'f> {
    func: &'f Func,
    vals: Vec<u128>,
    /// Blocks left before a run gives up.
    pub fuel: u64,
}

fn align_fault(addr: u64, bytes: u64, write: bool) -> Option<Fault> {
    if !addr.is_multiple_of(bytes) {
        Some(Fault {
            addr,
            write,
            kind: FaultKind::Alignment,
        })
    } else {
        None
    }
}

impl<'f> Evaluator<'f> {
    pub fn new(func: &'f Func) -> Evaluator<'f> {
        Evaluator {
            func,
            vals: vec![0; func.insts.len()],
            fuel: 5_000_000,
        }
    }

    fn get(&self, v: ValueId) -> u128 {
        match self.func.insts[v as usize].op {
            Op::Const(c) => c,
            _ => self.vals[v as usize],
        }
    }

    /// Run from the entry block.
    pub fn run(&mut self, cpu: &mut CpuState, bus: &mut dyn Bus) -> Stop {
        self.run_from(0, cpu, bus)
    }

    /// Run from `start`, which must be an origin block (the entry or a call continuation).
    pub fn run_from(&mut self, start: BlockId, cpu: &mut CpuState, bus: &mut dyn Bus) -> Stop {
        let func = self.func;
        let mut prev: Option<BlockId> = None;
        let mut cur = start;
        loop {
            if self.fuel == 0 {
                return Stop::OutOfFuel;
            }
            self.fuel -= 1;
            let block = &func.blocks[cur as usize];
            // Phis read the values as of the end of the previous block, all at once.
            if let Some(p) = prev {
                let mut incoming = Vec::with_capacity(block.phis.len());
                for &phi in &block.phis {
                    let Op::Phi(ops) = &func.insts[phi as usize].op else {
                        unreachable!("phi list holds phis")
                    };
                    let &(_, v) = ops
                        .iter()
                        .find(|&&(b, _)| b == p)
                        .unwrap_or_else(|| panic!("phi %{phi} has no operand for b{p}"));
                    incoming.push(self.get(v));
                }
                for (&phi, val) in block.phis.iter().zip(incoming) {
                    self.vals[phi as usize] = val;
                }
            }
            for &v in &block.body {
                if let Err((fault, pc)) = self.exec(v, cpu, bus) {
                    return Stop::Fault { fault, pc };
                }
            }
            match &block.term {
                Term::None => panic!("block b{cur} has no terminator"),
                Term::Br(t) => {
                    prev = Some(cur);
                    cur = *t;
                }
                Term::CondBr { cond, then, els } => {
                    prev = Some(cur);
                    cur = if self.get(*cond) != 0 { *then } else { *els };
                }
                Term::Switch {
                    value,
                    cases,
                    default,
                } => {
                    let x = self.get(*value) as u64;
                    prev = Some(cur);
                    cur = cases
                        .iter()
                        .find(|&&(a, _)| a == x)
                        .map_or(*default, |&(_, b)| b);
                }
                Term::Exit { kind, spill } => {
                    self.apply_spill(spill, cpu);
                    return match kind {
                        ExitKind::Return(v) => Stop::Return(self.get(*v) as u64),
                        ExitKind::Jump(a) => Stop::Jump(*a),
                        ExitKind::Dispatch(v) => Stop::Dispatch(self.get(*v) as u64),
                        ExitKind::Trap { kind, pc } => Stop::Trap {
                            kind: *kind,
                            pc: *pc,
                        },
                    };
                }
                Term::Call(c) => {
                    self.apply_spill(&c.spill, cpu);
                    let target_pc = match c.target {
                        CallTarget::Indirect(v) => self.get(v) as u64,
                        CallTarget::Direct(a) => a,
                        _ => 0,
                    };
                    return Stop::Call {
                        target: c.target.clone(),
                        target_pc,
                        pc: c.pc,
                        ret_pc: c.ret_pc,
                        cont: c.cont,
                    };
                }
            }
        }
    }

    fn apply_spill(&self, spill: &[(Var, ValueId)], cpu: &mut CpuState) {
        let mut nzcv = None;
        for &(var, v) in spill {
            let x = self.get(v);
            match var {
                Var::X(n) => cpu.x[usize::from(n)] = x as u64,
                Var::Sp => cpu.sp = x as u64,
                flag => {
                    let word = nzcv.get_or_insert(0u32);
                    *word |= ((x as u32) & 1) << flag.nzcv_bit();
                }
            }
        }
        if let Some(word) = nzcv {
            cpu.nzcv = word;
        }
    }

    fn exec(
        &mut self,
        v: ValueId,
        cpu: &mut CpuState,
        bus: &mut dyn Bus,
    ) -> Result<(), (Fault, u64)> {
        let func = self.func;
        let inst = &func.insts[v as usize];
        let ty = inst.ty;
        let m = ty.mask();
        let r: u128 = match &inst.op {
            Op::Const(c) => *c,
            Op::LoadVar(var) => match var {
                Var::X(n) => u128::from(cpu.x[usize::from(*n)]),
                Var::Sp => u128::from(cpu.sp),
                flag => u128::from((cpu.nzcv >> flag.nzcv_bit()) & 1),
            },
            Op::Phi(_) => unreachable!("phis are not in the body"),
            Op::Bin(op, a, b) => {
                let (a, b) = (self.get(*a), self.get(*b));
                sem::bin(*op, ty, a, b)
                    .unwrap_or_else(|| panic!("ill-formed {op:?} {ty:?} {a:#x} {b:#x}"))
            }
            Op::Un(op, a) => sem::un(*op, ty, self.get(*a)),
            Op::Cast(op, a) => sem::cast(*op, func.ty(*a), ty, self.get(*a)),
            Op::Icmp(p, a, b) => u128::from(sem::icmp(*p, func.ty(*a), self.get(*a), self.get(*b))),
            Op::Select(c, a, b) => {
                if self.get(*c) != 0 {
                    self.get(*a)
                } else {
                    self.get(*b)
                }
            }
            Op::Fshr(hi, lo, amt) => sem::fshr(ty, self.get(*hi), self.get(*lo), self.get(*amt)),
            Op::MulHi { signed, a, b } => u128::from(sem::mulhi(
                *signed,
                self.get(*a) as u64,
                self.get(*b) as u64,
            )),
            Op::Crc32 {
                size,
                castagnoli,
                acc,
                data,
            } => u128::from(crc32_update(
                self.get(*acc) as u32,
                self.get(*data) as u64,
                *size,
                *castagnoli,
            )),
            Op::Load {
                bytes,
                order,
                pc,
                addr,
            } => {
                let a = self.get(*addr) as u64;
                if *order != Order::Plain {
                    if let Some(f) = align_fault(a, u64::from(*bytes), false) {
                        return Err((f, *pc));
                    }
                }
                bus.read(a, *bytes).map_err(|f| (f, *pc))?
            }
            Op::Store {
                bytes,
                order,
                pc,
                addr,
                value,
            } => {
                let a = self.get(*addr) as u64;
                if *order != Order::Plain {
                    if let Some(f) = align_fault(a, u64::from(*bytes), true) {
                        return Err((f, *pc));
                    }
                }
                bus.write(a, *bytes, self.get(*value))
                    .map_err(|f| (f, *pc))?;
                0
            }
            Op::LoadExcl {
                bytes, pc, addr, ..
            } => {
                let a = self.get(*addr) as u64;
                if let Some(f) = align_fault(a, u64::from(*bytes), false) {
                    return Err((f, *pc));
                }
                bus.load_exclusive(cpu, a, *bytes).map_err(|f| (f, *pc))?
            }
            Op::StoreExcl {
                bytes,
                pc,
                addr,
                value,
                ..
            } => {
                let a = self.get(*addr) as u64;
                if let Some(f) = align_fault(a, u64::from(*bytes), true) {
                    return Err((f, *pc));
                }
                let ok = bus
                    .store_exclusive(cpu, a, *bytes, self.get(*value))
                    .map_err(|f| (f, *pc))?;
                u128::from(!ok)
            }
            Op::ClearExcl => {
                bus.clear_exclusive(cpu);
                0
            }
            Op::Fence(_) => 0,
            Op::DcZva { pc, addr } => {
                let base = (self.get(*addr) as u64) & !(ZVA_BLOCK - 1);
                for i in 0..ZVA_BLOCK / 8 {
                    bus.write(base + i * 8, 8, 0).map_err(|f| (f, *pc))?;
                }
                0
            }
            Op::Counter => u128::from(bus.counter()),
            Op::LoadField(f) => match f {
                Field::TpidrEl0 => u128::from(cpu.tpidr_el0),
                Field::TpidrroEl0 => u128::from(cpu.tpidrro_el0),
                Field::Fpcr => u128::from(cpu.fpcr),
                Field::Fpsr => u128::from(cpu.fpsr),
            },
            Op::StoreField(f, val) => {
                let x = self.get(*val);
                match f {
                    Field::TpidrEl0 => cpu.tpidr_el0 = x as u64,
                    Field::TpidrroEl0 => cpu.tpidrro_el0 = x as u64,
                    Field::Fpcr => cpu.fpcr = x as u32,
                    Field::Fpsr => cpu.fpsr = x as u32,
                }
                0
            }
        };
        self.vals[v as usize] = r & m;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Whole programs
// ---------------------------------------------------------------------------------------------

use nx_runtime::a64::interp::{step, Trap};
use nx_runtime::slow::{self, SlowHost};
use std::collections::BTreeMap;

/// How a program run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Control reached the exit address.
    Exited,
    /// An `SVC` the host declined to service (`pc` is the following instruction).
    Svc(u16),
    Trap {
        kind: TrapKind,
        pc: u64,
    },
    Fault {
        fault: Fault,
        pc: u64,
    },
    /// A valid encoding the interpreter does not implement.
    Unsupported {
        word: u32,
        pc: u64,
    },
    /// The slow path gave up (step limit).
    Halted,
}

/// The interpreter's view of a stop, in the vocabulary of [`Outcome`].
pub fn outcome_of_trap(trap: Trap, pc: u64) -> Outcome {
    match trap {
        Trap::Svc(imm) => Outcome::Svc(imm),
        Trap::Brk(imm) => Outcome::Trap {
            kind: TrapKind::Brk(imm),
            pc,
        },
        Trap::Undefined(w) => Outcome::Trap {
            kind: TrapKind::Undefined(w),
            pc,
        },
        Trap::Unsupported { word, .. } => Outcome::Unsupported { word, pc },
        Trap::Fault(fault) => Outcome::Fault { fault, pc },
        Trap::SysReg { reg, write } => Outcome::Trap {
            kind: TrapKind::sysreg(reg, write),
            pc,
        },
        Trap::SysInst {
            load,
            op1,
            crn,
            crm,
            op2,
        } => Outcome::Trap {
            kind: TrapKind::SysInst {
                load,
                enc: [op1, crn, crm, op2],
            },
            pc,
        },
    }
}

/// Services `SVC`; returning false stops the run.
pub type SvcHandler<'a> = Box<dyn FnMut(&mut CpuState, u16) -> bool + 'a>;

/// Runs lifted functions the way the generated code does: the same call and return protocol, the
/// same slow path for unknown targets.
pub struct ProgramEval<'a> {
    funcs: &'a BTreeMap<u64, Func>,
    exit_pc: u64,
    /// Addresses control went to that no function covers, in order.
    pub unknown: Vec<u64>,
    pub svc: SvcHandler<'a>,
    pub slow_steps: u64,
    /// Function calls made so far, and the limit that ends a runaway program.
    pub calls: u64,
    pub max_calls: u64,
}

impl<'a> ProgramEval<'a> {
    pub fn new(funcs: &'a BTreeMap<u64, Func>, exit_pc: u64) -> ProgramEval<'a> {
        ProgramEval {
            funcs,
            exit_pc,
            unknown: Vec::new(),
            svc: Box::new(|_, _| false),
            slow_steps: 100_000,
            calls: 0,
            max_calls: 200_000,
        }
    }

    pub fn run(&mut self, entry: u64, cpu: &mut CpuState, bus: &mut dyn Bus) -> Outcome {
        cpu.pc = entry;
        cpu.x[30] = self.exit_pc;
        loop {
            if cpu.pc == self.exit_pc {
                return Outcome::Exited;
            }
            if let Some(o) = self.call_function(cpu, bus) {
                return o;
            }
        }
    }

    /// Run the function at `cpu.pc`. `None` means it returned (and `cpu.pc` is where control goes
    /// next); `Some` is a stop.
    fn call_function(&mut self, cpu: &mut CpuState, bus: &mut dyn Bus) -> Option<Outcome> {
        self.calls += 1;
        if self.calls > self.max_calls {
            return Some(Outcome::Halted);
        }
        let funcs = self.funcs;
        let Some(func) = funcs.get(&cpu.pc) else {
            return self.slow_path(cpu, bus);
        };
        let mut ev = Evaluator::new(func);
        let mut stop = ev.run(cpu, bus);
        loop {
            match stop {
                Stop::Return(pc) => {
                    cpu.pc = pc;
                    return None;
                }
                Stop::Jump(a) => {
                    cpu.pc = a;
                    return self.call_function(cpu, bus);
                }
                Stop::Dispatch(pc) => {
                    cpu.pc = pc;
                    return self.call_function(cpu, bus);
                }
                Stop::Trap { kind, pc } => {
                    cpu.pc = pc;
                    return Some(Outcome::Trap { kind, pc });
                }
                Stop::Fault { fault, pc } => return Some(Outcome::Fault { fault, pc }),
                Stop::OutOfFuel => return Some(Outcome::Halted),
                Stop::Call {
                    target,
                    target_pc,
                    pc,
                    ret_pc,
                    cont,
                } => {
                    let r = match target {
                        CallTarget::Direct(t) => {
                            cpu.pc = t;
                            self.call_function(cpu, bus)
                        }
                        CallTarget::Indirect(_) => {
                            cpu.pc = target_pc;
                            self.call_function(cpu, bus)
                        }
                        CallTarget::Svc(imm) => {
                            cpu.pc = ret_pc;
                            if (self.svc)(cpu, imm) {
                                None
                            } else {
                                Some(Outcome::Svc(imm))
                            }
                        }
                        CallTarget::Fallback => {
                            cpu.pc = pc;
                            match step(cpu, bus) {
                                Ok(()) => None,
                                Err(t) => Some(outcome_of_trap(t, cpu.pc)),
                            }
                        }
                    };
                    if r.is_some() {
                        return r;
                    }
                    if cpu.pc != ret_pc {
                        return None;
                    }
                    stop = ev.run_from(cont, cpu, bus);
                }
            }
        }
    }

    fn slow_path(&mut self, cpu: &mut CpuState, bus: &mut dyn Bus) -> Option<Outcome> {
        self.unknown.push(cpu.pc);
        let max = self.slow_steps;
        let mut host = Host {
            eval: self,
            stop: None,
        };
        match slow::run(cpu, bus, &mut host, max) {
            slow::Exit::Returned => None,
            slow::Exit::Halted => host.stop.take().or(Some(Outcome::Halted)),
            slow::Exit::Trap(t) => Some(outcome_of_trap(t, cpu.pc)),
        }
    }
}

struct Host<'e, 'a> {
    eval: &'e mut ProgramEval<'a>,
    stop: Option<Outcome>,
}

impl SlowHost for Host<'_, '_> {
    fn has_native(&mut self, addr: u64) -> bool {
        self.eval.funcs.contains_key(&addr)
    }

    fn call_native(&mut self, cpu: &mut CpuState, bus: &mut dyn Bus) -> bool {
        match self.eval.call_function(cpu, bus) {
            None => true,
            Some(o) => {
                self.stop = Some(o);
                false
            }
        }
    }

    fn svc(&mut self, cpu: &mut CpuState, _bus: &mut dyn Bus, imm: u16) -> bool {
        if (self.eval.svc)(cpu, imm) {
            true
        } else {
            self.stop = Some(Outcome::Svc(imm));
            false
        }
    }
}
