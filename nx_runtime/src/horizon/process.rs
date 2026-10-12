//! Starting the process: the module's memory as the kernel records it, the main thread with its
//! stack and thread-local region, and what the entry point is handed.
//!
//! An NSO is entered the way the loader enters one: `X0` is zero and `X1` holds the main thread's
//! handle. A homebrew NRO is entered the way the homebrew loader enters one (Switchbrew,
//! "Homebrew ABI"): `X0` points at a list of loader configuration entries and `X1` is all ones.

use super::kernel::{Kernel, ProcessKind};
use super::layout::{mem_type, MAIN_STACK_TOP, PAGE};
use super::memory::Info;
use super::object::KObject;
use super::result::{kernel as err, KResult};
use super::svc::availability_mask;
use crate::rt::abi::{perm, NativeCpu};
use crate::rt::runtime::rt;
use std::io::Write;
use std::sync::Arc;

/// Where the homebrew loader configuration is placed.
pub const ENV_BASE: u64 = 0x2F_0000_0000;

/// The firmware version the loader configuration reports (`12.1.0`; docs/ASSUMPTIONS.md A-036).
pub const HOS_VERSION: u64 = (12 << 16) | (1 << 8);

/// A mapped segment of the module: address, size and `perm` bits.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub addr: u64,
    pub size: u64,
    pub perm: u32,
}

/// What the kernel needs to start the process.
pub struct StartConfig {
    pub entry: u64,
    pub segments: Vec<Segment>,
}

/// Loader configuration keys (Switchbrew, Homebrew ABI).
mod key {
    pub const END_OF_LIST: u32 = 0;
    pub const MAIN_THREAD_HANDLE: u32 = 1;
    pub const SYSCALL_AVAILABLE_HINT: u32 = 6;
    pub const APPLET_TYPE: u32 = 7;
    pub const PROCESS_HANDLE: u32 = 10;
    pub const RANDOM_SEED: u32 = 14;
    pub const HOS_VERSION: u32 = 16;
    pub const SYSCALL_AVAILABLE_HINT2: u32 = 17;
}

fn entry(key: u32, mandatory: bool, v0: u64, v1: u64) -> [u8; 24] {
    let mut e = [0u8; 24];
    e[0..4].copy_from_slice(&key.to_le_bytes());
    e[4..8].copy_from_slice(&u32::from(mandatory).to_le_bytes());
    e[8..16].copy_from_slice(&v0.to_le_bytes());
    e[16..24].copy_from_slice(&v1.to_le_bytes());
    e
}

impl Kernel {
    /// Record the module's segments, map the main stack, create the main thread and, for a
    /// homebrew NRO, the loader configuration. The thread is not started.
    pub fn prepare(self: &Arc<Kernel>, cfg: &StartConfig) -> KResult<u64> {
        for s in &cfg.segments {
            let t = if s.perm & perm::W != 0 {
                mem_type::CODE_DATA
            } else {
                mem_type::CODE
            };
            let mut st = self.lock();
            st.memory.insert(
                s.addr,
                s.size,
                Info {
                    mem_type: t,
                    perm: s.perm,
                    attr: 0,
                },
            );
        }
        let npdm_stack = self
            .process
            .npdm
            .as_ref()
            .map_or(0, |n| u64::from(n.meta.main_thread_stack_size));
        let stack_size = match (self.process.kind, npdm_stack) {
            (ProcessKind::Nro, _) => 8 << 20,
            (ProcessKind::Nso, 0) => 1 << 20,
            (ProcessKind::Nso, s) => s.next_multiple_of(PAGE),
        };
        {
            let mut st = self.lock();
            self.map_owned(
                &mut st,
                MAIN_STACK_TOP - stack_size,
                stack_size,
                Info {
                    mem_type: mem_type::STACK,
                    perm: perm::R | perm::W,
                    attr: 0,
                },
            )?;
        }
        let (priority, core) = self.process.npdm.as_ref().map_or((0x2C, 0), |n| {
            (
                i32::from(n.meta.main_thread_priority),
                i32::from(n.meta.main_thread_core),
            )
        });
        let (handle, id) =
            self.create_thread(cfg.entry, 0, MAIN_STACK_TOP, priority, core, true)?;
        self.lock().main_handle = handle;
        if self.process.kind == ProcessKind::Nro {
            self.write_loader_config(handle)?;
        }
        Ok(id)
    }

    /// The loader configuration entries of the homebrew ABI.
    fn write_loader_config(&self, main_thread: u32) -> KResult<()> {
        let process_handle = self.insert_object(KObject::Process)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&entry(
            key::MAIN_THREAD_HANDLE,
            true,
            u64::from(main_thread),
            0,
        ));
        // AppletType_Application.
        bytes.extend_from_slice(&entry(key::APPLET_TYPE, true, 0, 0));
        bytes.extend_from_slice(&entry(
            key::SYSCALL_AVAILABLE_HINT,
            false,
            availability_mask(0),
            availability_mask(0x40),
        ));
        bytes.extend_from_slice(&entry(
            key::SYSCALL_AVAILABLE_HINT2,
            false,
            availability_mask(0x80),
            0,
        ));
        bytes.extend_from_slice(&entry(
            key::PROCESS_HANDLE,
            false,
            u64::from(process_handle),
            0,
        ));
        bytes.extend_from_slice(&entry(
            key::RANDOM_SEED,
            false,
            self.random_u64(),
            self.random_u64(),
        ));
        bytes.extend_from_slice(&entry(key::HOS_VERSION, false, HOS_VERSION, 0));
        bytes.extend_from_slice(&entry(key::END_OF_LIST, true, 0, 0));
        let size = (bytes.len() as u64).next_multiple_of(PAGE);
        let mut st = self.lock();
        self.map_owned(
            &mut st,
            ENV_BASE,
            size,
            Info {
                mem_type: mem_type::NORMAL,
                perm: perm::R | perm::W,
                attr: 0,
            },
        )?;
        drop(st);
        rt().mem
            .write_bytes(ENV_BASE, &bytes, false)
            .map_err(|_| err::INVALID_POINTER)
    }

    /// Set the registers the entry point expects (see the module comment).
    pub(super) fn prepare_main_thread(&self, cpu: &mut NativeCpu) {
        let handle = self.lock().main_handle;
        match self.process.kind {
            ProcessKind::Nso => {
                cpu.state.x[0] = 0;
                cpu.state.x[1] = u64::from(handle);
            }
            ProcessKind::Nro => {
                cpu.state.x[0] = ENV_BASE;
                cpu.state.x[1] = u64::MAX;
            }
        }
    }

    /// Print what the guest sent to `OutputDebugString`.
    pub fn write_output(&self, bytes: &[u8]) {
        let mut sink = self.output.lock().unwrap_or_else(|e| e.into_inner());
        let _ = sink.write_all(bytes);
        let _ = sink.flush();
    }

    /// Send the guest's debug output somewhere else than standard output (tests capture it).
    pub fn set_output(&self, sink: Box<dyn Write + Send>) {
        *self.output.lock().unwrap_or_else(|e| e.into_inner()) = sink;
    }

    /// A pseudo-random 64-bit value for `GetInfo` and the loader configuration. `NX_RANDOM_SEED`
    /// makes the sequence reproducible.
    pub fn random_u64(&self) -> u64 {
        let mut s = self.random.lock().unwrap_or_else(|e| e.into_inner());
        // xorshift64*
        let mut x = *s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Run a loaded module as a Horizon process and return its exit status: the produced
/// executable's `main` for an NSO or an NRO.
///
/// # Safety
/// `m` must describe memory that is valid for the lengths it states (see `load_module`), and the
/// module's functions must be registered.
pub unsafe fn run_module(m: &crate::rt::host::ModuleDesc) -> i32 {
    use super::kernel::{Exit, ProcessInfo};
    use crate::npdm::Npdm;
    use crate::rt::host::{c_slice, kind, STOPPED_EXIT_CODE};

    let npdm_bytes = c_slice(m.npdm, m.npdm_len as usize);
    let npdm = if npdm_bytes.is_empty() {
        None
    } else {
        match Npdm::parse(npdm_bytes) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("recomp runtime: the embedded NPDM does not parse: {e}");
                return STOPPED_EXIT_CODE;
            }
        }
    };
    let process_kind = if m.kind == kind::NRO {
        ProcessKind::Nro
    } else {
        ProcessKind::Nso
    };
    let name = npdm
        .as_ref()
        .map_or_else(|| "main".to_string(), |n| n.meta.name.clone());
    let kernel = Kernel::new(ProcessInfo::new(&name, process_kind, npdm));
    super::services::register_defaults(&kernel);
    let segments = c_slice(m.segs, m.nsegs as usize)
        .iter()
        .map(|s| Segment {
            addr: s.addr,
            size: s.size,
            perm: s.perm,
        })
        .collect();
    let cfg = StartConfig {
        entry: m.entry,
        segments,
    };
    match kernel.run_process(&cfg) {
        Exit::Code(code) => code,
        Exit::Fatal(report) => {
            eprintln!("recomp runtime: {report}");
            STOPPED_EXIT_CODE
        }
    }
}
