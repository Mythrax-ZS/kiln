//! The supervisor-call interface: arguments in `X0`..`X7`, results in `X0`..`X7`, as the
//! Switchbrew SVC page documents each call.
//!
//! An SVC this runtime does not implement stops the run and says which one and where; no call is
//! answered with an invented success. Calls the program's NPDM does not enable are refused the
//! same way.

use super::handle::Handle;
use super::kernel::{Kernel, ThreadCtx};
use super::layout::{
    mem_type, ALIAS_BASE, ALIAS_SIZE, ASLR_BASE, ASLR_SIZE, HEAP_BASE, HEAP_REGION_SIZE,
    STACK_BASE, STACK_REGION_SIZE,
};
use super::object::KObject;
use super::result::{kernel as err, ResultCode};
use super::sm::Sm;
use crate::rt::abi::{stop, NativeCpu};
use crate::rt::runtime::{record_stop, rt};
use std::sync::Arc;

/// Whether the thread goes on after the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// What `GetInfo` reports as the memory the process may use in total: the size of the application
/// pool of a 4 GiB console as I remember it from Atmosphere (docs/ASSUMPTIONS.md A-036). A title
/// sets a heap of about 3.1 GiB, so this must be at least that.
pub const TOTAL_MEMORY_SIZE: u64 = 0xCD50_0000;

/// The SVCs this kernel implements.
pub const IMPLEMENTED: &[u32] = &[
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x21,
    0x24, 0x25, 0x26, 0x27, 0x29, 0x34, 0x35, 0x36, 0x45,
];

/// The names of the SVCs, for reports.
pub fn svc_name(id: u32) -> Option<&'static str> {
    Some(match id {
        0x01 => "SetHeapSize",
        0x02 => "SetMemoryPermission",
        0x03 => "SetMemoryAttribute",
        0x04 => "MapMemory",
        0x05 => "UnmapMemory",
        0x06 => "QueryMemory",
        0x07 => "ExitProcess",
        0x08 => "CreateThread",
        0x09 => "StartThread",
        0x0A => "ExitThread",
        0x0B => "SleepThread",
        0x0C => "GetThreadPriority",
        0x0D => "SetThreadPriority",
        0x0E => "GetThreadCoreMask",
        0x0F => "SetThreadCoreMask",
        0x10 => "GetCurrentProcessorNumber",
        0x11 => "SignalEvent",
        0x12 => "ClearEvent",
        0x13 => "MapSharedMemory",
        0x14 => "UnmapSharedMemory",
        0x15 => "CreateTransferMemory",
        0x16 => "CloseHandle",
        0x17 => "ResetSignal",
        0x18 => "WaitSynchronization",
        0x19 => "CancelSynchronization",
        0x1A => "ArbitrateLock",
        0x1B => "ArbitrateUnlock",
        0x1C => "WaitProcessWideKeyAtomic",
        0x1D => "SignalProcessWideKey",
        0x1E => "GetSystemTick",
        0x1F => "ConnectToNamedPort",
        0x20 => "SendSyncRequestLight",
        0x21 => "SendSyncRequest",
        0x22 => "SendSyncRequestWithUserBuffer",
        0x23 => "SendAsyncRequestWithUserBuffer",
        0x24 => "GetProcessId",
        0x25 => "GetThreadId",
        0x26 => "Break",
        0x27 => "OutputDebugString",
        0x28 => "ReturnFromException",
        0x29 => "GetInfo",
        0x34 => "WaitForAddress",
        0x35 => "SignalToAddress",
        0x36 => "SynchronizePreemptionState",
        0x40 => "CreateSession",
        0x41 => "AcceptSession",
        0x42 => "ReplyAndReceiveLight",
        0x43 => "ReplyAndReceive",
        0x44 => "ReplyAndReceiveWithUserBuffer",
        0x45 => "CreateEvent",
        _ => return None,
    })
}

/// The mask `SyscallAvailableHint` reports for SVCs `base..base + 64`.
pub fn availability_mask(base: u32) -> u64 {
    IMPLEMENTED
        .iter()
        .filter(|&&id| id >= base && id < base + 64)
        .fold(0u64, |m, &id| m | 1 << (id - base))
}

/// The entry point generated code reaches through `nx_svc`.
fn cpu_of<'a>(p: *mut NativeCpu) -> &'a mut NativeCpu {
    // SAFETY: the runtime passes the CPU of the thread that executed the SVC.
    unsafe { &mut *p }
}

pub extern "C" fn svc_entry(cpu: *mut NativeCpu, imm: u32) -> i32 {
    let cpu = cpu_of(cpu);
    let ctx = cpu.ctl.user.cast::<ThreadCtx>();
    if ctx.is_null() {
        // A thread that was not started by a kernel: the runtime records the unhandled call.
        return 1;
    }
    // SAFETY: set by `Kernel::run_on` and alive for the whole run.
    let (kernel, thread, monitor) = unsafe {
        (
            Arc::clone(&(*ctx).kernel),
            (*ctx).thread,
            Arc::clone(&(*ctx).monitor),
        )
    };
    // SAFETY: as above; only this thread uses the field.
    let core = unsafe { &(*ctx).core };
    kernel.stir();
    monitor.enter(imm, &cpu.state.x);
    // A call that can block or reschedule leaves the core for the threads waiting for it, and
    // the thread asks for a core again when the call is over (`NX_CORES`).
    let gives_way = kernel.call_gives_way(imm);
    if gives_way {
        kernel.give_core(core);
    }
    let flow = kernel.svc(cpu, thread, imm);
    if gives_way {
        kernel.take_core(thread, core);
    }
    monitor.leave();
    i32::from(flow == Flow::Stop)
}

fn w(x: u64) -> u32 {
    x as u32
}

fn result(cpu: &mut NativeCpu, r: ResultCode) {
    cpu.state.x[0] = u64::from(r.0);
}

impl Kernel {
    /// Stop the run for a reason that is the program's or this runtime's, with the report.
    fn stop_run(&self, cpu: &mut NativeCpu, message: String) -> Flow {
        let pc = cpu.state.pc.wrapping_sub(4);
        let stack = self.guest_call_stack(cpu);
        let stack = if stack.is_empty() {
            String::new()
        } else {
            let frames: Vec<String> = stack.iter().map(|a| format!("{a:#x}")).collect();
            format!("; return addresses up the stack: {}", frames.join(" "))
        };
        self.fatal(format!("{message} (SVC at {pc:#x}){stack}"));
        record_stop(cpu, stop::UNIMPLEMENTED, pc, 0);
        Flow::Stop
    }

    /// The return addresses of the thread's frames, innermost first, found by following the frame
    /// pointer chain (`x29` points at a saved `x29` and `x30`, as the compilers of Horizon
    /// programs lay out their frames). The walk stops at the first pointer that does not look like
    /// a frame.
    pub(super) fn guest_call_stack(&self, cpu: &NativeCpu) -> Vec<u64> {
        let mut out = Vec::new();
        // A value that is not an address of the 39-bit space is a marker (the return address of
        // a thread entry point, say), not a place in the program.
        let plausible = |a: u64| a != 0 && a < 1 << 39;
        if plausible(cpu.state.x[30]) {
            out.push(cpu.state.x[30]);
        }
        let mut fp = cpu.state.x[29];
        for _ in 0..24 {
            if fp == 0 || !fp.is_multiple_of(8) {
                break;
            }
            let mut frame = [0u8; 16];
            if !self.read_memory(fp, &mut frame) {
                break;
            }
            let next = u64::from_le_bytes(frame[..8].try_into().unwrap_or([0; 8]));
            let lr = u64::from_le_bytes(frame[8..].try_into().unwrap_or([0; 8]));
            if !plausible(lr) {
                break;
            }
            out.push(lr);
            if next <= fp {
                break;
            }
            fp = next;
        }
        out
    }

    /// Execute `SVC #imm` for `thread`. On entry `cpu.state.pc` is the instruction after it.
    /// `NX_TRACE_SVC` in the environment logs every call and its result on standard error.
    pub fn svc(self: &Arc<Kernel>, cpu: &mut NativeCpu, thread: u64, imm: u32) -> Flow {
        if !self.trace {
            return self.svc_dispatch(cpu, thread, imm);
        }
        let args = [
            cpu.state.x[0],
            cpu.state.x[1],
            cpu.state.x[2],
            cpu.state.x[3],
        ];
        let pc = cpu.state.pc.wrapping_sub(4);
        let name = svc_name(imm).unwrap_or("?");
        // A call that can block is logged when it starts as well, so a hang shows where it is.
        if matches!(imm, 0x0B | 0x18 | 0x1A | 0x1C | 0x34) {
            eprintln!(
                "[svc] thread {thread} at {pc:#x}: {name} ({imm:#x}) x0..x3 = {:#x} {:#x} {:#x} {:#x} (may block)",
                args[0], args[1], args[2], args[3]
            );
            if imm == 0x18 {
                self.trace_wait_list(thread, args[1], args[2]);
            }
        }
        let flow = self.svc_dispatch(cpu, thread, imm);
        eprintln!(
            "[svc] thread {thread} at {pc:#x}: {name} ({imm:#x}) x0..x3 = {:#x} {:#x} {:#x} {:#x} -> x0 = {:#x} x1 = {:#x}{}",
            args[0],
            args[1],
            args[2],
            args[3],
            cpu.state.x[0],
            cpu.state.x[1],
            if flow == Flow::Stop { " (stop)" } else { "" }
        );
        flow
    }

    /// Trace what a `WaitSynchronization` waits on: each handle and the object it names.
    fn trace_wait_list(&self, thread: u64, list: u64, count: u64) {
        let count = (count as u32 as i32).clamp(0, 0x40) as usize;
        let mut raw = vec![0u8; 4 * count];
        if count == 0 || !self.read_memory(list, &mut raw) {
            return;
        }
        let waits: Vec<String> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| {
                let h = u32::from_le_bytes(*c);
                match self.object(h, thread) {
                    Some(o) => format!("{h:#x} ({o:?})"),
                    None => format!("{h:#x} (no object)"),
                }
            })
            .collect();
        eprintln!("[svc]   thread {thread} waits on {}", waits.join(", "));
    }

    fn svc_dispatch(self: &Arc<Kernel>, cpu: &mut NativeCpu, thread: u64, imm: u32) -> Flow {
        if let Some(npdm) = &self.process.npdm {
            if !npdm.allows_svc(imm) {
                return self.stop_run(
                    cpu,
                    format!(
                        "SVC {imm:#x}{} is not enabled by the program's kernel capabilities",
                        svc_name(imm).map_or(String::new(), |n| format!(" ({n})"))
                    ),
                );
            }
        }
        let x = cpu.state.x;
        match imm {
            0x01 => match self.set_heap_size(x[1]) {
                Ok(addr) => {
                    result(cpu, ResultCode::SUCCESS);
                    cpu.state.x[1] = addr;
                }
                Err(e) => {
                    result(cpu, e);
                    cpu.state.x[1] = 0;
                }
            },
            0x02 => {
                let r = self.set_memory_permission(x[0], x[1], w(x[2]));
                result(cpu, r);
            }
            0x03 => {
                let r = self.set_memory_attribute(x[0], x[1], w(x[2]), w(x[3]));
                result(cpu, r);
            }
            0x04 => {
                let r = self.map_memory(x[0], x[1], x[2]);
                result(cpu, r);
            }
            0x05 => {
                let r = self.unmap_memory(x[0], x[1], x[2]);
                result(cpu, r);
            }
            0x06 => {
                let q = self.query_memory(x[2]);
                let mut info = [0u8; 0x28];
                info[0..8].copy_from_slice(&q.base.to_le_bytes());
                info[8..16].copy_from_slice(&q.size.to_le_bytes());
                info[16..20].copy_from_slice(&u32::from(q.info.mem_type).to_le_bytes());
                info[20..24].copy_from_slice(&q.info.attr.to_le_bytes());
                info[24..28].copy_from_slice(&q.info.perm.to_le_bytes());
                if self.write_memory(x[0], &info) {
                    result(cpu, ResultCode::SUCCESS);
                } else {
                    result(cpu, err::INVALID_POINTER);
                }
                cpu.state.x[1] = 0;
            }
            0x07 => {
                self.exit_process(0);
                let pc = cpu.state.pc.wrapping_sub(4);
                record_stop(cpu, stop::EXIT_PROCESS, pc, 0);
                return Flow::Stop;
            }
            0x08 => match self.create_thread(x[1], x[2], x[3], x[4] as i32, x[5] as i32, false) {
                Ok((h, _)) => {
                    result(cpu, ResultCode::SUCCESS);
                    cpu.state.x[1] = u64::from(h);
                }
                Err(e) => {
                    result(cpu, e);
                    cpu.state.x[1] = 0;
                }
            },
            0x09 => {
                let target = self.lock().thread_of(w(x[0]), thread);
                let r = match target {
                    Some(id) => self.start_thread(id),
                    None => err::INVALID_HANDLE,
                };
                result(cpu, r);
            }
            0x0A => {
                let pc = cpu.state.pc.wrapping_sub(4);
                record_stop(cpu, stop::EXIT_THREAD, pc, 0);
                return Flow::Stop;
            }
            0x0B => {
                self.sleep(x[0] as i64);
            }
            0x0C => {
                let st = self.lock();
                match st
                    .thread_of(w(x[1]), thread)
                    .and_then(|id| st.threads.get(&id))
                {
                    Some(t) => {
                        let p = t.priority;
                        drop(st);
                        result(cpu, ResultCode::SUCCESS);
                        cpu.state.x[1] = u64::from(p as u32);
                    }
                    None => {
                        drop(st);
                        result(cpu, err::INVALID_HANDLE);
                    }
                }
            }
            0x0D => {
                let priority = w(x[1]) as i32;
                let mut st = self.lock();
                let r = if !(0..=0x3F).contains(&priority) {
                    err::INVALID_PRIORITY
                } else {
                    match st.thread_of(w(x[0]), thread) {
                        Some(id) => {
                            if let Some(t) = st.threads.get_mut(&id) {
                                t.priority = priority;
                            }
                            ResultCode::SUCCESS
                        }
                        None => err::INVALID_HANDLE,
                    }
                };
                drop(st);
                result(cpu, r);
            }
            0x0E => {
                let st = self.lock();
                match st
                    .thread_of(w(x[2]), thread)
                    .and_then(|id| st.threads.get(&id))
                {
                    Some(t) => {
                        let (ideal, mask) = (t.ideal_core, t.core_mask);
                        drop(st);
                        result(cpu, ResultCode::SUCCESS);
                        cpu.state.x[1] = u64::from(ideal as u32);
                        cpu.state.x[2] = mask;
                    }
                    None => {
                        drop(st);
                        result(cpu, err::INVALID_HANDLE);
                    }
                }
            }
            0x0F => {
                let (ideal, mask) = (w(x[1]) as i32, x[2]);
                let mut st = self.lock();
                let r = match st.thread_of(w(x[0]), thread) {
                    None => err::INVALID_HANDLE,
                    Some(_) if mask == 0 => err::INVALID_COMBINATION,
                    Some(id) => {
                        if let Some(t) = st.threads.get_mut(&id) {
                            t.ideal_core = if ideal < 0 { t.ideal_core } else { ideal };
                            t.core_mask = mask;
                        }
                        ResultCode::SUCCESS
                    }
                };
                drop(st);
                result(cpu, r);
            }
            0x10 => {
                let core = self.lock().threads.get(&thread).map_or(0, |t| t.ideal_core);
                cpu.state.x[0] = core.max(0) as u64;
            }
            0x11 => {
                let r = self.signal_event(w(x[0]));
                result(cpu, r);
            }
            0x12 => {
                let r = self.clear_event(w(x[0]));
                result(cpu, r);
            }
            // MapSharedMemory(handle, addr, size, permission), UnmapSharedMemory(handle, addr, size)
            0x13 => {
                let r = self.map_shared_memory(thread, w(x[0]), x[1], x[2], w(x[3]));
                result(cpu, r);
            }
            0x14 => {
                let r = self.unmap_shared_memory(thread, w(x[0]), x[1], x[2]);
                result(cpu, r);
            }
            0x15 => match self.create_transfer_memory(x[1], x[2], w(x[3])) {
                Ok(h) => {
                    result(cpu, ResultCode::SUCCESS);
                    cpu.state.x[1] = u64::from(h);
                }
                Err(e) => {
                    result(cpu, e);
                    cpu.state.x[1] = 0;
                }
            },
            0x16 => {
                let r = self.close_handle(w(x[0]));
                result(cpu, r);
            }
            0x17 => {
                let r = self.reset_signal(w(x[0]));
                result(cpu, r);
            }
            0x18 => {
                let count = w(x[2]) as i32;
                if !(0..=0x40).contains(&count) {
                    result(cpu, err::OUT_OF_RANGE);
                    cpu.state.x[1] = u64::from(u32::MAX);
                } else {
                    let mut raw = vec![0u8; 4 * count as usize];
                    if !raw.is_empty() && !self.read_memory(x[1], &mut raw) {
                        result(cpu, err::INVALID_POINTER);
                        cpu.state.x[1] = u64::from(u32::MAX);
                    } else {
                        let handles: Vec<Handle> = raw
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|c| u32::from_le_bytes(*c))
                            .collect();
                        let (r, index) = self.wait_synchronization(thread, &handles, x[3] as i64);
                        result(cpu, r);
                        cpu.state.x[1] = u64::from(index as u32);
                    }
                }
            }
            0x19 => {
                let r = self.cancel_synchronization(thread, w(x[0]));
                result(cpu, r);
            }
            0x1A => {
                let r = self.arbitrate_lock(thread, w(x[0]), x[1], w(x[2]));
                result(cpu, r);
            }
            0x1B => {
                let r = self.arbitrate_unlock(x[0]);
                result(cpu, r);
            }
            0x1C => {
                let r = self.wait_process_wide_key(thread, x[0], x[1], w(x[2]), x[3] as i64);
                result(cpu, r);
            }
            0x1D => {
                self.signal_process_wide_key(x[0], w(x[1]) as i32);
                result(cpu, ResultCode::SUCCESS);
            }
            0x1E => {
                cpu.state.x[0] = rt().counter();
            }
            0x1F => {
                let mut raw = [0u8; 12];
                if !self.read_memory(x[1], &mut raw) {
                    result(cpu, err::INVALID_POINTER);
                    cpu.state.x[1] = 0;
                } else {
                    let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                    if len == raw.len() {
                        result(cpu, err::OUT_OF_RANGE);
                        cpu.state.x[1] = 0;
                    } else {
                        let name = String::from_utf8_lossy(&raw[..len]).into_owned();
                        if name != "sm:" {
                            return self.stop_run(
                                cpu,
                                format!("the program connected to the named port `{name}`, which is not implemented"),
                            );
                        }
                        let sm = Sm::new(Arc::downgrade(self));
                        match self.insert_object(KObject::session(Box::new(sm))) {
                            Ok(h) => {
                                result(cpu, ResultCode::SUCCESS);
                                cpu.state.x[1] = u64::from(h);
                            }
                            Err(e) => {
                                result(cpu, e);
                                cpu.state.x[1] = 0;
                            }
                        }
                    }
                }
            }
            0x21 => {
                let tls = cpu.state.tpidrro_el0;
                match self.send_sync_request(thread, tls, w(x[0])) {
                    Ok(r) => result(cpu, r),
                    Err(s) => return self.stop_run(cpu, s.0),
                }
            }
            0x24 => match self.object(w(x[1]), thread) {
                Some(KObject::Process) => {
                    result(cpu, ResultCode::SUCCESS);
                    cpu.state.x[1] = self.process.process_id;
                }
                _ => {
                    result(cpu, err::INVALID_HANDLE);
                    cpu.state.x[1] = 0;
                }
            },
            0x25 => match self.object(w(x[1]), thread) {
                Some(KObject::Thread(id)) => {
                    result(cpu, ResultCode::SUCCESS);
                    cpu.state.x[1] = id;
                }
                _ => {
                    result(cpu, err::INVALID_HANDLE);
                    cpu.state.x[1] = 0;
                }
            },
            0x26 => {
                let reason = x[0];
                if reason & 0x8000_0000 != 0 {
                    // A notification only: the program goes on.
                    result(cpu, ResultCode::SUCCESS);
                } else {
                    let kind = match reason & 0xFF {
                        0 => "panic",
                        1 => "assert",
                        2 => "user",
                        7 => "C++ exception",
                        _ => "other",
                    };
                    return self.stop_run(
                        cpu,
                        format!(
                            "the program stopped itself with Break (reason {reason:#x}: {kind}, \
                             argument {:#x}, size {:#x})",
                            x[1], x[2]
                        ),
                    );
                }
            }
            0x27 => {
                // A string longer than a megabyte is a bug, not a message.
                if x[1] > 1 << 20 {
                    result(cpu, err::INVALID_SIZE);
                } else {
                    let mut text = vec![0u8; x[1] as usize];
                    if !text.is_empty() && !self.read_memory(x[0], &mut text) {
                        result(cpu, err::INVALID_POINTER);
                    } else {
                        self.write_output(&text);
                        result(cpu, ResultCode::SUCCESS);
                    }
                }
            }
            0x29 => match self.get_info(w(x[1]), w(x[2]), x[3], thread) {
                Ok(Ok(v)) => {
                    result(cpu, ResultCode::SUCCESS);
                    cpu.state.x[1] = v;
                }
                Ok(Err(e)) => {
                    result(cpu, e);
                    cpu.state.x[1] = 0;
                }
                Err(msg) => return self.stop_run(cpu, msg),
            },
            0x34 => {
                let r = self.wait_for_address(thread, x[0], w(x[1]), w(x[2]) as i32, x[3] as i64);
                result(cpu, r);
            }
            0x35 => {
                if w(x[1]) == 2 {
                    return self.stop_run(
                        cpu,
                        "SignalToAddress with type SignalAndModifyBasedOnWaitingThreadCountIfEqual is not implemented"
                            .to_string(),
                    );
                }
                let r = self.signal_to_address(x[0], w(x[1]), w(x[2]) as i32, w(x[3]) as i32);
                result(cpu, r);
            }
            0x36 => {}
            0x45 => match self.create_event() {
                Ok((writable, readable)) => {
                    result(cpu, ResultCode::SUCCESS);
                    cpu.state.x[1] = u64::from(writable);
                    cpu.state.x[2] = u64::from(readable);
                }
                Err(e) => {
                    result(cpu, e);
                    cpu.state.x[1] = 0;
                    cpu.state.x[2] = 0;
                }
            },
            other => {
                let name = svc_name(other).map_or(String::new(), |n| format!(" ({n})"));
                return self.stop_run(cpu, format!("SVC {other:#x}{name} is not implemented"));
            }
        }
        // A call can end the process or a wait can find it ending.
        if self.exit_state().is_some() && cpu.state.x[0] == u64::from(err::THREAD_TERMINATING.0) {
            let pc = cpu.state.pc.wrapping_sub(4);
            record_stop(cpu, stop::EXIT_PROCESS, pc, 0);
            return Flow::Stop;
        }
        Flow::Continue
    }

    /// `GetInfo`. The outer error stops the run (an information type this runtime does not
    /// implement); the inner one is the kernel result.
    fn get_info(
        &self,
        id: u32,
        handle: Handle,
        sub: u64,
        thread: u64,
    ) -> Result<Result<u64, ResultCode>, String> {
        let process_handle = matches!(self.object(handle, thread), Some(KObject::Process));
        let needs_process = matches!(id, 0..=7 | 12..=18 | 20..=24 | 26..=28);
        let needs_zero = matches!(id, 8..=11 | 19);
        if id > 28 {
            return Ok(Err(err::INVALID_ENUM_VALUE));
        }
        if needs_process && !process_handle {
            return Ok(Err(err::INVALID_HANDLE));
        }
        if needs_zero && handle != 0 {
            return Ok(Err(err::INVALID_HANDLE));
        }
        if id == 25 {
            return Err("GetInfo with ThreadTickCount is not implemented".to_string());
        }
        // Types with no sub-type take zero; the others check theirs below.
        if !matches!(id, 10 | 11) && sub != 0 {
            return Ok(Err(err::INVALID_COMBINATION));
        }
        let thread_info = self
            .process
            .npdm
            .as_ref()
            .and_then(|n| n.kernel.thread_info);
        let value = match id {
            0 => thread_info.map_or(0x7, |t| {
                (u64::from(t.min_core)..=u64::from(t.max_core)).fold(0u64, |m, c| m | 1 << c)
            }),
            1 => thread_info.map_or(u64::MAX, |t| {
                let (lo, hi) = (
                    u64::from(t.highest_priority.min(t.lowest_priority)),
                    u64::from(t.highest_priority.max(t.lowest_priority)),
                );
                (lo..=hi.min(63)).fold(0u64, |m, p| m | 1 << p)
            }),
            2 => ALIAS_BASE,
            3 => ALIAS_SIZE,
            4 => HEAP_BASE,
            5 => HEAP_REGION_SIZE,
            6 => TOTAL_MEMORY_SIZE,
            7 => self.used_memory(),
            8 => 0,
            10 => {
                // Time the cores spent idle is not measured.
                if sub != u64::MAX && sub > 3 {
                    return Ok(Err(err::INVALID_COMBINATION));
                }
                0
            }
            11 => {
                if sub > 3 {
                    return Ok(Err(err::INVALID_COMBINATION));
                }
                self.random_u64()
            }
            12 => ASLR_BASE,
            13 => ASLR_SIZE,
            14 => STACK_BASE,
            15 => STACK_REGION_SIZE,
            16 => self
                .process
                .npdm
                .as_ref()
                .map_or(0, |n| u64::from(n.meta.system_resource_size)),
            17 => 0,
            18 => self.process.program_id,
            19 => return Ok(Err(err::INVALID_ENUM_VALUE)),
            21 => TOTAL_MEMORY_SIZE,
            22 => self.used_memory(),
            23 => u64::from(self.process.kind == super::kernel::ProcessKind::Nso),
            9 | 20 | 24 | 26..=28 => {
                return Err(format!(
                    "GetInfo with information type {id} is not implemented"
                ))
            }
            _ => return Ok(Err(err::INVALID_ENUM_VALUE)),
        };
        Ok(Ok(value))
    }

    fn used_memory(&self) -> u64 {
        self.lock().memory.total(|t| {
            matches!(
                t,
                mem_type::CODE
                    | mem_type::CODE_DATA
                    | mem_type::NORMAL
                    | mem_type::STACK
                    | mem_type::THREAD_LOCAL
            )
        })
    }
}
