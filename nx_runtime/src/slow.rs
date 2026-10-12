//! The slow path: running guest code that has no native translation in the interpreter.
//!
//! Generated code reaches it when control goes to an address that is not in the function table: a
//! `BR`/`BLR` through a register nothing explained, a return into a function that is no longer on
//! the native stack, a jump into a gap the analysis could not decode. The interpreter then runs
//! the code, and hands control back to native code the moment it reaches a known entry.
//!
//! The protocol between native frames is the same as for generated code: a function returns with
//! `cpu.pc` set to where execution continues, and its caller carries on only if that is the return
//! address it expects. The interpreter keeps a call depth so that it knows which `RET` is the one
//! that returns to its native caller: `BL`/`BLR` count up, `RET` counts down, and a `RET` at depth
//! zero leaves.

use crate::a64::bus::Bus;
use crate::a64::decode::decode;
use crate::a64::inst::{BranchReg, Inst};
use crate::a64::interp::{execute, Trap};
use crate::a64::state::CpuState;

/// What the slow path needs from the native side.
pub trait SlowHost {
    /// Is there native code that starts at `addr`?
    fn has_native(&mut self, addr: u64) -> bool;
    /// Run the native function at `cpu.pc` until it returns; `cpu.pc` is then the address
    /// execution continues at. Returns false when the run was stopped (a fault or an unhandled
    /// call inside native code): the slow path then ends too.
    fn call_native(&mut self, cpu: &mut CpuState, bus: &mut dyn Bus) -> bool;
    /// Perform `SVC #imm`; `cpu.pc` already points after it. Returns false to stop the run.
    fn svc(&mut self, cpu: &mut CpuState, bus: &mut dyn Bus, imm: u16) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// The interpreted code returned (or tail-called native code that returned) to its caller;
    /// `cpu.pc` is the address to continue at.
    Returned,
    /// A native call or `SVC` asked to stop the run.
    Halted,
    /// The interpreter could not continue; `cpu.pc` is the instruction.
    Trap(Trap),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Seq,
    Call,
    Jump,
    Return,
}

fn flow_of(inst: &Inst) -> Flow {
    match *inst {
        Inst::B { link: true, .. }
        | Inst::BranchReg {
            kind: BranchReg::Blr,
            ..
        } => Flow::Call,
        Inst::BranchReg {
            kind: BranchReg::Ret,
            ..
        } => Flow::Return,
        Inst::B { .. }
        | Inst::BCond { .. }
        | Inst::Cbz { .. }
        | Inst::Tbz { .. }
        | Inst::BranchReg { .. } => Flow::Jump,
        _ => Flow::Seq,
    }
}

/// Interpret from `cpu.pc` until control returns to the native caller. `max_steps` bounds the
/// run (a safety net for tests; the runtime passes `u64::MAX`).
pub fn run(cpu: &mut CpuState, bus: &mut dyn Bus, host: &mut dyn SlowHost, max_steps: u64) -> Exit {
    let mut depth = 0usize;
    for _ in 0..max_steps {
        let word = match bus.fetch(cpu.pc) {
            Ok(w) => w,
            Err(f) => return Exit::Trap(Trap::Fault(f)),
        };
        let inst = decode(word);
        let mut flow = flow_of(&inst);
        match execute(cpu, bus, inst) {
            Ok(()) => {}
            Err(Trap::Svc(imm)) => {
                if !host.svc(cpu, bus, imm) {
                    return Exit::Halted;
                }
                flow = Flow::Seq;
            }
            Err(t) => return Exit::Trap(t),
        }
        match flow {
            Flow::Call => depth += 1,
            Flow::Return => {
                if depth == 0 {
                    return Exit::Returned;
                }
                depth -= 1;
            }
            Flow::Seq | Flow::Jump => {}
        }
        // Native code takes over at its entries.
        let mut entered = flow;
        while host.has_native(cpu.pc) {
            if !host.call_native(cpu, bus) {
                return Exit::Halted;
            }
            // The native function returned to `cpu.pc`: that finishes a call, or the interpreted
            // frame that tail-called it.
            match entered {
                Flow::Call => depth -= 1,
                Flow::Seq | Flow::Jump | Flow::Return => {
                    if depth == 0 {
                        return Exit::Returned;
                    }
                    depth -= 1;
                }
            }
            entered = Flow::Return;
        }
    }
    Exit::Trap(Trap::Unsupported {
        word: 0,
        what: "step limit reached in the slow path",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::asm;
    use crate::testing::flatbus::FlatBus;
    use std::collections::BTreeSet;

    const BASE: u64 = 0x1000;

    struct Host {
        native: BTreeSet<u64>,
        calls: Vec<u64>,
        /// What a native function does: set x0 and return to x30.
        ret_value: u64,
    }

    impl SlowHost for Host {
        fn has_native(&mut self, addr: u64) -> bool {
            self.native.contains(&addr)
        }
        fn call_native(&mut self, cpu: &mut CpuState, _bus: &mut dyn Bus) -> bool {
            self.calls.push(cpu.pc);
            cpu.x[0] = self.ret_value;
            cpu.pc = cpu.x[30];
            true
        }
        fn svc(&mut self, _cpu: &mut CpuState, _bus: &mut dyn Bus, _imm: u16) -> bool {
            true
        }
    }

    fn bus_with(words: &[u32]) -> FlatBus {
        let mut bus = FlatBus::new(BASE, 0x1000);
        bus.load_words(BASE, words);
        bus
    }

    #[test]
    fn a_ret_at_depth_zero_returns_to_the_native_caller() {
        let mut bus = bus_with(&[asm::movz(true, 0, 5, 0), asm::ret(30)]);
        let mut cpu = CpuState {
            pc: BASE,
            ..CpuState::default()
        };
        cpu.x[30] = 0x9000;
        let mut host = Host {
            native: BTreeSet::new(),
            calls: vec![],
            ret_value: 0,
        };
        assert_eq!(run(&mut cpu, &mut bus, &mut host, 100), Exit::Returned);
        assert_eq!((cpu.x[0], cpu.pc), (5, 0x9000));
    }

    /// `mov x19, x30` and `mov x30, x19`: a function that calls must keep its return address.
    fn save_lr() -> u32 {
        asm::logical_shifted(true, 1, 0, false, 30, 0, 31, 19)
    }

    fn restore_lr() -> u32 {
        asm::logical_shifted(true, 1, 0, false, 19, 0, 31, 30)
    }

    #[test]
    fn nested_interpreted_calls_do_not_leave_early() {
        // f: mov x19, x30 ; bl g ; mov x30, x19 ; ret        g: movz x1, 7 ; ret
        let mut bus = bus_with(&[
            save_lr(),
            asm::call(12),
            restore_lr(),
            asm::ret(30),
            asm::movz(true, 1, 7, 0),
            asm::ret(30),
        ]);
        let mut cpu = CpuState {
            pc: BASE,
            ..CpuState::default()
        };
        cpu.x[30] = 0x9000;
        let mut host = Host {
            native: BTreeSet::new(),
            calls: vec![],
            ret_value: 0,
        };
        assert_eq!(run(&mut cpu, &mut bus, &mut host, 100), Exit::Returned);
        assert_eq!((cpu.x[1], cpu.pc), (7, 0x9000));
    }

    #[test]
    fn a_call_into_a_native_entry_runs_native_code_and_continues() {
        // f: mov x19, x30 ; bl native ; add x2, x0, #1 ; mov x30, x19 ; ret
        let mut bus = bus_with(&[
            save_lr(),
            asm::call(0x100),
            asm::add_imm(true, 2, 0, 1),
            restore_lr(),
            asm::ret(30),
        ]);
        let mut cpu = CpuState {
            pc: BASE,
            ..CpuState::default()
        };
        cpu.x[30] = 0x9000;
        let mut host = Host {
            native: BTreeSet::from([BASE + 0x104]),
            calls: vec![],
            ret_value: 41,
        };
        assert_eq!(run(&mut cpu, &mut bus, &mut host, 100), Exit::Returned);
        assert_eq!(host.calls, vec![BASE + 0x104]);
        assert_eq!((cpu.x[2], cpu.pc), (42, 0x9000));
    }

    #[test]
    fn a_tail_jump_into_native_code_returns_to_the_caller() {
        // 0: b 0x100 (native) -- the native function returns to x30.
        let mut bus = bus_with(&[asm::jmp(0x100)]);
        let mut cpu = CpuState {
            pc: BASE,
            ..CpuState::default()
        };
        cpu.x[30] = 0x9000;
        let mut host = Host {
            native: BTreeSet::from([BASE + 0x100]),
            calls: vec![],
            ret_value: 3,
        };
        assert_eq!(run(&mut cpu, &mut bus, &mut host, 10), Exit::Returned);
        assert_eq!((cpu.x[0], cpu.pc), (3, 0x9000));
    }

    #[test]
    fn faults_and_undefined_code_stop_the_run() {
        let mut bus = bus_with(&[asm::ldr_x(0, 1, 0)]);
        let mut cpu = CpuState {
            pc: BASE,
            ..CpuState::default()
        };
        cpu.x[1] = 0xDEAD_0000;
        let mut host = Host {
            native: BTreeSet::new(),
            calls: vec![],
            ret_value: 0,
        };
        assert!(matches!(
            run(&mut cpu, &mut bus, &mut host, 10),
            Exit::Trap(Trap::Fault(_))
        ));
    }
}
