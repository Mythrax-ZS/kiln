//! Tests of the kernel layer. Supervisor calls are made the way generated code makes them (a CPU
//! with the argument registers set, `Kernel::svc`), and "guest code" that has to run on its own
//! threads is a native function registered at an address, which calls the SVC entry point the
//! way generated code calls `nx_svc`.
//!
//! The runtime's guest memory is process-wide, so the tests take turns.

use super::handle::{CURRENT_PROCESS, CURRENT_THREAD};
use super::kernel::{Exit, Kernel, ProcessInfo, ProcessKind};
use super::layout::*;
use super::process::{Segment, StartConfig, ENV_BASE};
use super::result::{kernel as err, ResultCode};
use super::service::{Request, Response, Service, ServiceResult, Stop};
use super::svc::{svc_entry, Flow};
use crate::npdm::{builder, Npdm};
use crate::rt::abi::{perm, stop, FuncEntry, NativeCpu, STOP_PC};
use crate::rt::runtime::rt;
use std::ffi::c_void;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

static LOCK: Mutex<()> = Mutex::new(());

/// A scratch region of guest memory the tests use for buffers.
const SCRATCH: u64 = 0x4000_0000;
const SCRATCH_SIZE: u64 = 0x10_000;

struct Env {
    kernel: Arc<Kernel>,
    cpu: Box<NativeCpu>,
    tid: u64,
    out: Arc<Mutex<Vec<u8>>>,
    _guard: MutexGuard<'static, ()>,
}

struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Env {
    fn new() -> Env {
        Env::with(ProcessKind::Nso, None, Vec::new())
    }

    fn with(kind: ProcessKind, npdm: Option<Npdm>, segments: Vec<Segment>) -> Env {
        Env::build(kind, npdm, segments, None)
    }

    /// A process whose RomFS is the directory `romfs`, with the default services registered.
    fn with_romfs(romfs: &std::path::Path) -> Env {
        let e = Env::build(
            ProcessKind::Nso,
            None,
            Vec::new(),
            Some(romfs.to_path_buf()),
        );
        super::services::register_defaults(&e.kernel);
        e
    }

    fn build(
        kind: ProcessKind,
        npdm: Option<Npdm>,
        segments: Vec<Segment>,
        romfs: Option<std::path::PathBuf>,
    ) -> Env {
        let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = rt().mem.unmap(SCRATCH, SCRATCH_SIZE);
        rt().mem
            .map(SCRATCH, SCRATCH_SIZE, perm::R | perm::W)
            .unwrap();
        let mut info = ProcessInfo::new("test", kind, npdm);
        info.romfs = romfs;
        let kernel = Kernel::new(info);
        let out = Arc::new(Mutex::new(Vec::new()));
        kernel.set_output(Box::new(Capture(Arc::clone(&out))));
        let tid = kernel
            .prepare(&StartConfig {
                entry: 0x0800_0000,
                segments,
            })
            .unwrap();
        // The test's own thread stands for the running main thread.
        kernel.lock().threads.get_mut(&tid).unwrap().started = true;
        let cpu = kernel.new_cpu(tid);
        Env {
            kernel,
            cpu,
            tid,
            out,
            _guard: guard,
        }
    }

    fn svc(&mut self, imm: u32, args: &[u64]) -> Flow {
        for (i, a) in args.iter().enumerate() {
            self.cpu.state.x[i] = *a;
        }
        self.cpu.state.pc = 0x1004;
        self.kernel.svc(&mut self.cpu, self.tid, imm)
    }

    fn x(&self, i: usize) -> u64 {
        self.cpu.state.x[i]
    }

    /// The kernel result of the last call.
    fn r(&self) -> u32 {
        self.cpu.state.x[0] as u32
    }

    fn write(&self, addr: u64, data: &[u8]) {
        rt().mem.write_bytes(addr, data, false).unwrap();
    }

    fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        rt().mem.read_bytes(addr, &mut v, false).unwrap();
        v
    }

    fn printed(&self) -> String {
        String::from_utf8_lossy(&self.out.lock().unwrap()).into_owned()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = rt().mem.unmap(SCRATCH, SCRATCH_SIZE);
    }
}

// ---------------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------------

#[test]
fn the_heap_grows_shrinks_and_is_queryable() {
    let mut e = Env::new();
    // Not a multiple of 2 MiB, and too large.
    e.svc(0x01, &[0, 0x1000]);
    assert_eq!(e.r(), err::INVALID_SIZE.0);
    e.svc(0x01, &[0, (1 << 32) + 0x20_0000]);
    assert_eq!(e.r(), err::INVALID_SIZE.0);

    e.svc(0x01, &[0, 0x40_0000]);
    assert_eq!((e.r(), e.x(1)), (0, HEAP_BASE));
    rt().mem
        .write(HEAP_BASE + 0x3F_FFF8, 8, 0xAB, Relaxed)
        .unwrap();
    e.svc(0x06, &[SCRATCH, 0, HEAP_BASE + 0x1000]);
    assert_eq!(e.r(), 0);
    let info = e.read(SCRATCH, 0x28);
    let u64_at = |o: usize| u64::from_le_bytes(info[o..o + 8].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(info[o..o + 4].try_into().unwrap());
    assert_eq!((u64_at(0), u64_at(8)), (HEAP_BASE, 0x40_0000));
    assert_eq!((u32_at(16), u32_at(20), u32_at(24)), (5, 0, 3));
    assert_eq!(u32_at(36), 0, "the padding is zero");

    // Shrinking unmaps the tail; the rest keeps its contents.
    rt().mem.write(HEAP_BASE + 0x10, 8, 0xCD, Relaxed).unwrap();
    e.svc(0x01, &[0, 0x20_0000]);
    assert_eq!(e.r(), 0);
    assert!(rt().mem.read(HEAP_BASE + 0x3F_FFF8, 8, Relaxed).is_err());
    assert_eq!(rt().mem.read(HEAP_BASE + 0x10, 8, Relaxed), Ok(0xCD));
    e.svc(0x06, &[SCRATCH, 0, HEAP_BASE + 0x20_0000]);
    let info = e.read(SCRATCH, 0x28);
    assert_eq!(u32::from_le_bytes(info[16..20].try_into().unwrap()), 0);
    // Growing again gives fresh zeroed pages.
    e.svc(0x01, &[0, 0x40_0000]);
    assert_eq!(rt().mem.read(HEAP_BASE + 0x3F_FFF8, 8, Relaxed), Ok(0));
    e.svc(0x01, &[0, 0]);
    assert!(rt().mem.read(HEAP_BASE, 8, Relaxed).is_err());
}

#[test]
fn memory_permissions_follow_the_state_rules() {
    let mut e = Env::with(
        ProcessKind::Nso,
        None,
        vec![
            Segment {
                addr: CODE_BASE,
                size: 0x1000,
                perm: perm::R | perm::X,
            },
            Segment {
                addr: CODE_BASE + 0x1000,
                size: 0x1000,
                perm: perm::R | perm::W,
            },
        ],
    );
    e.svc(0x01, &[0, 0x20_0000]);
    // Read-only heap pages refuse writes; read-write restores them.
    e.svc(0x02, &[HEAP_BASE + 0x1000, 0x2000, 1]);
    assert_eq!(e.r(), 0);
    assert!(rt().mem.write(HEAP_BASE + 0x1000, 4, 1, Relaxed).is_err());
    assert_eq!(rt().mem.read(HEAP_BASE + 0x1000, 4, Relaxed), Ok(0));
    e.svc(0x06, &[SCRATCH, 0, HEAP_BASE + 0x1800]);
    assert_eq!(e.read(SCRATCH, 0x28)[24], 1);
    e.svc(0x02, &[HEAP_BASE + 0x1000, 0x2000, 3]);
    assert_eq!(e.r(), 0);
    assert!(rt().mem.write(HEAP_BASE + 0x1000, 4, 1, Relaxed).is_ok());
    // Execute is refused, write-only is refused, unaligned and zero sizes are refused.
    e.svc(0x02, &[HEAP_BASE, 0x1000, 5]);
    assert_eq!(e.r(), err::INVALID_NEW_MEMORY_PERMISSION.0);
    e.svc(0x02, &[HEAP_BASE, 0x1000, 2]);
    assert_eq!(e.r(), err::INVALID_NEW_MEMORY_PERMISSION.0);
    e.svc(0x02, &[HEAP_BASE + 1, 0x1000, 1]);
    assert_eq!(e.r(), err::INVALID_ADDRESS.0);
    e.svc(0x02, &[HEAP_BASE, 0, 1]);
    assert_eq!(e.r(), err::INVALID_SIZE.0);
    // Code that was mapped read-only/executable cannot be reprotected; data can.
    e.svc(0x02, &[CODE_BASE, 0x1000, 3]);
    assert_eq!(e.r(), err::INVALID_CURRENT_MEMORY.0);
    // A range that is not all mapped is refused.
    e.svc(0x02, &[HEAP_BASE, 0x40_0000, 1]);
    assert_eq!(e.r(), err::INVALID_CURRENT_MEMORY.0);
    // The uncached attribute can be set; nothing else can.
    e.svc(0x03, &[HEAP_BASE, 0x1000, 8, 8]);
    assert_eq!(e.r(), 0);
    e.svc(0x06, &[SCRATCH, 0, HEAP_BASE]);
    assert_eq!(e.read(SCRATCH, 0x28)[20], 8);
    e.svc(0x03, &[HEAP_BASE, 0x1000, 1, 1]);
    assert_eq!(e.r(), err::INVALID_COMBINATION.0);
}

#[test]
fn map_memory_makes_a_second_view_and_unmap_gives_the_pages_back() {
    let mut e = Env::new();
    e.svc(0x01, &[0, 0x20_0000]);
    rt().mem.write(HEAP_BASE, 8, 0x1122_3344, Relaxed).unwrap();
    let dst = STACK_BASE + 0x10_0000;
    e.svc(0x04, &[dst, HEAP_BASE, 0x2000]);
    assert_eq!(e.r(), 0);
    // The new view shows the data and shares the bytes; the source is inaccessible.
    assert_eq!(rt().mem.read(dst, 8, Relaxed), Ok(0x1122_3344));
    assert!(rt().mem.read(HEAP_BASE, 8, Relaxed).is_err());
    rt().mem.write(dst + 8, 8, 0x77, Relaxed).unwrap();
    e.svc(0x06, &[SCRATCH, 0, dst]);
    assert_eq!(e.read(SCRATCH, 0x28)[16], mem_type::STACK);
    e.svc(0x06, &[SCRATCH, 0, HEAP_BASE]);
    let src = e.read(SCRATCH, 0x28);
    assert_eq!((src[16], src[20], src[24]), (mem_type::NORMAL, 1, 0));
    // Mapping outside the stack region, or from unmapped memory, is refused.
    e.svc(0x04, &[ALIAS_BASE, HEAP_BASE + 0x10_0000, 0x1000]);
    assert_eq!(e.r(), err::INVALID_MEMORY_REGION.0);
    e.svc(0x04, &[dst + 0x10_0000, HEAP_BASE + 0x80_0000, 0x1000]);
    assert_eq!(e.r(), err::INVALID_CURRENT_MEMORY.0);
    // Unmapping restores the source with what was written through the view.
    e.svc(0x05, &[dst, HEAP_BASE, 0x2000]);
    assert_eq!(e.r(), 0);
    assert_eq!(rt().mem.read(HEAP_BASE + 8, 8, Relaxed), Ok(0x77));
    assert!(rt().mem.read(dst, 8, Relaxed).is_err());
    e.svc(0x06, &[SCRATCH, 0, HEAP_BASE]);
    let src = e.read(SCRATCH, 0x28);
    assert_eq!((src[16], src[20], src[24]), (mem_type::NORMAL, 0, 3));
}

#[test]
fn query_memory_describes_modules_gaps_and_the_ends() {
    let mut e = Env::with(
        ProcessKind::Nso,
        None,
        vec![
            Segment {
                addr: CODE_BASE,
                size: 0x2000,
                perm: perm::R | perm::X,
            },
            Segment {
                addr: CODE_BASE + 0x2000,
                size: 0x1000,
                perm: perm::R,
            },
            Segment {
                addr: CODE_BASE + 0x3000,
                size: 0x3000,
                perm: perm::R | perm::W,
            },
        ],
    );
    let query = |e: &mut Env, addr: u64| {
        e.svc(0x06, &[SCRATCH, 0, addr]);
        assert_eq!(e.r(), 0);
        let b = e.read(SCRATCH, 0x28);
        (
            u64::from_le_bytes(b[0..8].try_into().unwrap()),
            u64::from_le_bytes(b[8..16].try_into().unwrap()),
            u32::from_le_bytes(b[16..20].try_into().unwrap()),
            u32::from_le_bytes(b[24..28].try_into().unwrap()),
        )
    };
    assert_eq!(query(&mut e, CODE_BASE + 0x1234), (CODE_BASE, 0x2000, 3, 5));
    assert_eq!(
        query(&mut e, CODE_BASE + 0x2000),
        (CODE_BASE + 0x2000, 0x1000, 3, 1)
    );
    assert_eq!(
        query(&mut e, CODE_BASE + 0x5000),
        (CODE_BASE + 0x3000, 0x3000, 4, 3)
    );
    // The main stack and a thread-local region exist too.
    let (base, _, t, p) = query(&mut e, MAIN_STACK_TOP - 8);
    assert!(base < MAIN_STACK_TOP && t == 0x0B && p == 3);
    let tls = e.kernel.lock().threads[&e.tid].tls;
    assert_eq!(query(&mut e, tls).2, 0x0C);
    // Free space after the module, and the inaccessible ends.
    let (base, size, t, p) = query(&mut e, CODE_BASE + 0x6000);
    assert_eq!((base, t, p), (CODE_BASE + 0x6000, 0, 0));
    assert!(size > 0x1_0000_0000);
    assert_eq!(query(&mut e, 0x1000).2, 0x10);
    assert_eq!(query(&mut e, ADDRESS_SPACE_END + 5).2, 0x10);
    // An unwritable result buffer is an error.
    e.svc(0x06, &[0xDEAD_0000, 0, CODE_BASE]);
    assert_eq!(e.r(), err::INVALID_POINTER.0);
}

#[test]
fn get_info_reports_the_regions_and_refuses_what_it_does_not_know() {
    let mut e = Env::new();
    let info = |e: &mut Env, id: u64, handle: u64, sub: u64| {
        e.svc(0x29, &[0, id, handle, sub]);
        (e.r(), e.x(1))
    };
    let proc = u64::from(CURRENT_PROCESS);
    assert_eq!(info(&mut e, 2, proc, 0), (0, ALIAS_BASE));
    assert_eq!(info(&mut e, 3, proc, 0), (0, ALIAS_SIZE));
    assert_eq!(info(&mut e, 4, proc, 0), (0, HEAP_BASE));
    assert_eq!(info(&mut e, 5, proc, 0), (0, HEAP_REGION_SIZE));
    assert_eq!(info(&mut e, 12, proc, 0), (0, ASLR_BASE));
    assert_eq!(info(&mut e, 13, proc, 0), (0, ASLR_SIZE));
    assert_eq!(info(&mut e, 14, proc, 0), (0, STACK_BASE));
    assert_eq!(info(&mut e, 15, proc, 0), (0, STACK_REGION_SIZE));
    let (r, total) = info(&mut e, 6, proc, 0);
    let (_, used) = info(&mut e, 7, proc, 0);
    assert!(r == 0 && total > used && used > 0);
    assert_eq!(info(&mut e, 8, 0, 0), (0, 0));
    // A process-typed query with the wrong kind of handle, a zero-handle query with a handle.
    assert_eq!(info(&mut e, 2, 0, 0).0, err::INVALID_HANDLE.0);
    assert_eq!(info(&mut e, 8, proc, 0).0, err::INVALID_HANDLE.0);
    assert_eq!(
        info(&mut e, 2, u64::from(CURRENT_THREAD), 0).0,
        err::INVALID_HANDLE.0
    );
    // Entropy takes sub-types 0 to 3 and differs between calls.
    let a = info(&mut e, 11, 0, 0);
    let b = info(&mut e, 11, 0, 1);
    assert!(a.0 == 0 && b.0 == 0 && a.1 != b.1);
    assert_eq!(info(&mut e, 11, 0, 4).0, err::INVALID_COMBINATION.0);
    // Out-of-range types are an enum error; known-but-unimplemented ones stop the run.
    assert_eq!(info(&mut e, 99, proc, 0).0, err::INVALID_ENUM_VALUE.0);
    assert_eq!(e.svc(0x29, &[0, 9, 0, 0]), Flow::Stop);
    assert!(
        matches!(e.kernel.exit_state(), Some(Exit::Fatal(m)) if m.contains("information type 9"))
    );
}

#[test]
fn debug_output_and_the_system_tick() {
    let mut e = Env::new();
    e.write(SCRATCH, b"hello, horizon\n");
    e.svc(0x27, &[SCRATCH, 15]);
    assert_eq!(e.r(), 0);
    assert_eq!(e.printed(), "hello, horizon\n");
    e.svc(0x27, &[0xDEAD_0000, 4]);
    assert_eq!(e.r(), err::INVALID_POINTER.0);
    e.svc(0x27, &[SCRATCH, 1 << 21]);
    assert_eq!(e.r(), err::INVALID_SIZE.0);
    rt().set_counter_fixed(1000);
    e.svc(0x1E, &[]);
    assert_eq!(e.x(0), 1000);
    rt().set_counter_fixed(0);
}

#[test]
fn an_svc_nothing_implements_stops_the_run_and_says_which() {
    {
        let mut e = Env::new();
        assert_eq!(e.svc(0x2A, &[]), Flow::Stop);
        assert_eq!(e.cpu.ctl.stop, stop::UNIMPLEMENTED);
        assert_eq!(e.cpu.state.pc, STOP_PC);
        match e.kernel.exit_state() {
            Some(Exit::Fatal(m)) => assert!(m.contains("SVC 0x2a") && m.contains("0x1000"), "{m}"),
            other => panic!("{other:?}"),
        }
    }
    // A named one reports its name.
    let mut e = Env::new();
    assert_eq!(e.svc(0x13, &[]), Flow::Stop);
    match e.kernel.exit_state() {
        Some(Exit::Fatal(m)) => assert!(m.contains("MapSharedMemory"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_npdm_decides_which_svcs_exist() {
    let spec = builder::NpdmSpec {
        kc: vec![builder::enable_svcs(0, &[6])],
        ..builder::NpdmSpec::default()
    };
    let npdm = Npdm::parse(&builder::build(&spec)).unwrap();
    let mut e = Env::with(ProcessKind::Nso, Some(npdm), Vec::new());
    // QueryMemory (6) is enabled; OutputDebugString (0x27) is in group 1 and was not enabled.
    assert_eq!(e.svc(0x06, &[SCRATCH, 0, CODE_BASE]), Flow::Continue);
    assert_eq!(e.svc(0x27, &[SCRATCH, 0]), Flow::Stop);
    match e.kernel.exit_state() {
        Some(Exit::Fatal(m)) => assert!(
            m.contains("not enabled") && m.contains("OutputDebugString"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }
}

#[test]
fn break_with_the_notification_bit_returns_and_any_other_stops() {
    let mut e = Env::new();
    assert_eq!(e.svc(0x26, &[0x8000_0001, 0, 0]), Flow::Continue);
    assert_eq!(e.r(), 0);
    assert_eq!(e.svc(0x26, &[1, 0x1234, 8]), Flow::Stop);
    match e.kernel.exit_state() {
        Some(Exit::Fatal(m)) => assert!(m.contains("assert") && m.contains("0x1234"), "{m}"),
        other => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Events and waiting
// ---------------------------------------------------------------------------------------------

#[test]
fn events_signal_clear_and_reset() {
    let mut e = Env::new();
    e.svc(0x45, &[]);
    assert_eq!(e.r(), 0);
    let (writable, readable) = (e.x(1), e.x(2));
    e.write(SCRATCH, &(readable as u32).to_le_bytes());
    let wait = |e: &mut Env, timeout: i64| {
        e.svc(0x18, &[0, SCRATCH, 1, timeout as u64]);
        (e.r(), e.x(1) as u32)
    };
    // Unsignalled: a poll times out and leaves the index at -1.
    assert_eq!(wait(&mut e, 0), (err::TIMED_OUT.0, u32::MAX));
    // Only a writable end signals; a readable end cannot.
    e.svc(0x11, &[readable]);
    assert_eq!(e.r(), err::INVALID_HANDLE.0);
    e.svc(0x11, &[writable]);
    assert_eq!(e.r(), 0);
    assert_eq!(wait(&mut e, 0), (0, 0));
    // The signal stays until reset.
    assert_eq!(wait(&mut e, 0), (0, 0));
    e.svc(0x17, &[readable]);
    assert_eq!(e.r(), 0);
    e.svc(0x17, &[readable]);
    assert_eq!(e.r(), err::INVALID_STATE.0);
    assert_eq!(wait(&mut e, 0).0, err::TIMED_OUT.0);
    // ClearEvent works on either end.
    e.svc(0x11, &[writable]);
    e.svc(0x12, &[readable]);
    assert_eq!(wait(&mut e, 0).0, err::TIMED_OUT.0);
    e.svc(0x11, &[writable]);
    e.svc(0x12, &[writable]);
    assert_eq!(wait(&mut e, 0).0, err::TIMED_OUT.0);
    // Closing handles: a second close fails.
    e.svc(0x16, &[writable]);
    assert_eq!(e.r(), 0);
    e.svc(0x16, &[writable]);
    assert_eq!(e.r(), err::INVALID_HANDLE.0);
}

#[test]
fn wait_synchronization_checks_its_arguments_and_reports_the_first_signalled_index() {
    let mut e = Env::new();
    e.svc(0x45, &[]);
    let (w1, r1) = (e.x(1), e.x(2));
    e.svc(0x45, &[]);
    let (w2, r2) = (e.x(1), e.x(2));
    let handles = [r1 as u32, r2 as u32, 0x7FFF_FFFF];
    let mut raw = Vec::new();
    for h in handles {
        raw.extend_from_slice(&h.to_le_bytes());
    }
    e.write(SCRATCH, &raw);
    // An invalid handle in the list is an error, whichever others are signalled.
    e.svc(0x11, &[w1]);
    e.svc(0x18, &[0, SCRATCH, 3, 0]);
    assert_eq!(e.r(), err::INVALID_HANDLE.0);
    // Two signalled: the lower index wins.
    e.svc(0x11, &[w2]);
    e.svc(0x18, &[0, SCRATCH, 2, 0]);
    assert_eq!((e.r(), e.x(1)), (0, 0));
    e.svc(0x17, &[r1]);
    e.svc(0x18, &[0, SCRATCH, 2, 0]);
    assert_eq!((e.r(), e.x(1)), (0, 1));
    // Too many handles, and a bad pointer.
    e.svc(0x18, &[0, SCRATCH, 0x41, 0]);
    assert_eq!(e.r(), err::OUT_OF_RANGE.0);
    e.svc(0x18, &[0, 0xDEAD_0000, 2, 0]);
    assert_eq!(e.r(), err::INVALID_POINTER.0);
    // Zero handles with a timeout just times out.
    e.svc(0x18, &[0, SCRATCH, 0, 1_000_000]);
    assert_eq!(e.r(), err::TIMED_OUT.0);
}

/// Another thread's view of the kernel: its own CPU and thread id.
fn second_thread(k: &Arc<Kernel>) -> (u64, Box<NativeCpu>) {
    let (_, id) = k.create_thread(0, 0, 0, 0x2C, 0, false).unwrap();
    (id, k.new_cpu(id))
}

#[test]
fn a_blocked_wait_wakes_when_the_event_is_signalled_and_a_timeout_expires() {
    let mut e = Env::new();
    e.svc(0x45, &[]);
    let (writable, readable) = (e.x(1) as u32, e.x(2) as u32);
    let k = Arc::clone(&e.kernel);
    let (other, _cpu) = second_thread(&k);
    let waiter = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.wait_synchronization(other, &[readable], -1))
    };
    std::thread::sleep(Duration::from_millis(50));
    assert!(!waiter.is_finished());
    assert_eq!(k.signal_event(writable), ResultCode::SUCCESS);
    assert_eq!(waiter.join().unwrap(), (ResultCode::SUCCESS, 0));
    // A timeout takes about as long as asked.
    k.reset_signal(readable);
    let start = Instant::now();
    let (r, i) = k.wait_synchronization(other, &[readable], 30_000_000);
    assert_eq!((r, i), (err::TIMED_OUT, -1));
    assert!(start.elapsed() >= Duration::from_millis(25));
}

#[test]
fn cancel_synchronization_interrupts_a_wait_or_the_next_one() {
    let mut e = Env::new();
    e.svc(0x45, &[]);
    let readable = e.x(2) as u32;
    let k = Arc::clone(&e.kernel);
    let (other, _cpu) = second_thread(&k);
    let other_handle = k
        .insert_object(super::object::KObject::Thread(other))
        .unwrap();
    let waiter = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.wait_synchronization(other, &[readable], -1))
    };
    std::thread::sleep(Duration::from_millis(30));
    e.svc(0x19, &[u64::from(other_handle)]);
    assert_eq!(e.r(), 0);
    assert_eq!(waiter.join().unwrap(), (err::CANCELLED, -1));
    // Cancelling a thread that is not waiting flags its next wait, once.
    e.svc(0x19, &[u64::from(other_handle)]);
    assert_eq!(
        k.wait_synchronization(other, &[readable], 0),
        (err::CANCELLED, -1)
    );
    assert_eq!(
        k.wait_synchronization(other, &[readable], 0),
        (err::TIMED_OUT, -1)
    );
}

#[test]
fn ending_the_process_wakes_every_waiter_with_the_terminating_result() {
    let mut e = Env::new();
    e.svc(0x45, &[]);
    let readable = e.x(2) as u32;
    let k = Arc::clone(&e.kernel);
    let (other, _cpu) = second_thread(&k);
    let waiter = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.wait_synchronization(other, &[readable], -1))
    };
    std::thread::sleep(Duration::from_millis(30));
    k.exit_process(3);
    assert_eq!(waiter.join().unwrap(), (err::THREAD_TERMINATING, -1));
    assert_eq!(k.wait_for_exit(), Exit::Code(3));
    // The thread that asked to end the process stops too.
    assert_eq!(e.svc(0x07, &[]), Flow::Stop);
    assert_eq!(e.cpu.ctl.stop, stop::EXIT_PROCESS);
}

// ---------------------------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------------------------

/// Guest code for the thread tests: store the argument's low byte plus one at `[x0]`, then exit
/// the thread by calling the SVC entry point as generated code does.
extern "C" fn thread_worker(p: *mut NativeCpu) {
    // SAFETY: the runtime passes the running thread's CPU.
    let c = unsafe { &mut *p };
    let target = c.state.x[0];
    let tls = c.state.tpidrro_el0;
    // The thread-local region the kernel gave this thread is its own and zeroed.
    let first = rt().mem.read(tls, 8, Relaxed).unwrap();
    rt().mem.write(target, 8, 0x1000 + first, Relaxed).unwrap();
    rt().mem.write(tls, 8, 0x5555, Relaxed).unwrap();
    c.state.pc = 0x0900_0004;
    svc_entry(p, 0x0A);
}

fn register(addr: u64, f: extern "C" fn(*mut NativeCpu)) {
    rt().register(&[FuncEntry {
        addr,
        func: f as *const c_void,
    }]);
}

#[test]
fn a_created_thread_runs_guest_code_and_can_be_waited_for() {
    let mut e = Env::new();
    register(0x0900_0000, thread_worker);
    let stack = SCRATCH + 0x8000;
    e.svc(0x08, &[0, 0x0900_0000, SCRATCH, stack, 0x2C, 1]);
    assert_eq!(e.r(), 0);
    let handle = e.x(1);
    // The thread has not run yet; its id is known and it is not signalled.
    e.svc(0x25, &[0, handle]);
    assert_eq!(e.r(), 0);
    assert!(e.x(1) > e.tid);
    e.write(SCRATCH + 0x100, &(handle as u32).to_le_bytes());
    e.svc(0x18, &[0, SCRATCH + 0x100, 1, 0]);
    assert_eq!(e.r(), err::TIMED_OUT.0);
    assert_eq!(rt().mem.read(SCRATCH, 8, Relaxed), Ok(0));
    // Starting it twice is an error; once is enough.
    e.svc(0x09, &[handle]);
    assert_eq!(e.r(), 0);
    e.svc(0x18, &[0, SCRATCH + 0x100, 1, 2_000_000_000]);
    assert_eq!((e.r(), e.x(1)), (0, 0));
    assert_eq!(rt().mem.read(SCRATCH, 8, Relaxed), Ok(0x1000));
    e.svc(0x09, &[handle]);
    assert_eq!(e.r(), err::INVALID_STATE.0);
}

#[test]
fn thread_arguments_are_validated() {
    let mut e = Env::new();
    // Priority outside 0..=0x3F, and a core that does not exist.
    e.svc(0x08, &[0, 0x0900_0000, 0, SCRATCH, 0x40, 0]);
    assert_eq!(e.r(), err::INVALID_PRIORITY.0);
    e.svc(0x08, &[0, 0x0900_0000, 0, SCRATCH, 0x2C, 4]);
    assert_eq!(e.r(), err::INVALID_CORE_ID.0);
    // -2 means the process default.
    e.svc(
        0x08,
        &[0, 0x0900_0000, 0, SCRATCH, 0x2C, (-2i32) as u32 as u64],
    );
    assert_eq!(e.r(), 0);
    let h = e.x(1);
    // Priority and core mask round-trip; the current-thread pseudo handle works.
    e.svc(0x0D, &[h, 0x30]);
    assert_eq!(e.r(), 0);
    e.svc(0x0C, &[0, h]);
    assert_eq!((e.r(), e.x(1)), (0, 0x30));
    e.svc(0x0D, &[h, 0x70]);
    assert_eq!(e.r(), err::INVALID_PRIORITY.0);
    e.svc(0x0F, &[h, 2, 0b100]);
    assert_eq!(e.r(), 0);
    e.svc(0x0E, &[0, 0, h]);
    assert_eq!((e.r(), e.x(1), e.x(2)), (0, 2, 0b100));
    e.svc(0x0F, &[h, 0, 0]);
    assert_eq!(e.r(), err::INVALID_COMBINATION.0);
    e.svc(0x0C, &[0, u64::from(CURRENT_THREAD)]);
    assert_eq!(e.r(), 0);
    e.svc(0x25, &[0, u64::from(CURRENT_THREAD)]);
    assert_eq!((e.r(), e.x(1)), (0, e.tid));
    e.svc(0x24, &[0, u64::from(CURRENT_PROCESS)]);
    assert_eq!((e.r(), e.x(1)), (0, e.kernel.process.process_id));
    e.svc(0x24, &[0, h]);
    assert_eq!(e.r(), err::INVALID_HANDLE.0);
}

// ---------------------------------------------------------------------------------------------
// Mutexes, condition variables and address arbitration
// ---------------------------------------------------------------------------------------------

const WAIT: u32 = 0x4000_0000;

fn word(addr: u64) -> u32 {
    rt().mem.read(addr, 4, Relaxed).unwrap() as u32
}

fn set_word(addr: u64, v: u32) {
    rt().mem.write(addr, 4, u128::from(v), Relaxed).unwrap();
}

#[test]
fn a_waiter_gets_the_mutex_when_the_owner_unlocks() {
    let e = Env::new();
    let k = Arc::clone(&e.kernel);
    let m = SCRATCH + 0x40;
    let (b, _cpu) = second_thread(&k);
    // The owner is a thread of the process; it holds the lock and knows there is a waiter.
    let (owner, _) = k.create_thread(0, 0, 0, 0x2C, 0, false).unwrap();
    set_word(m, owner | WAIT);
    let waiter = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.arbitrate_lock(b, owner, m, 0x222))
    };
    std::thread::sleep(Duration::from_millis(40));
    assert!(!waiter.is_finished());
    assert_eq!(k.arbitrate_unlock(m), ResultCode::SUCCESS);
    assert_eq!(waiter.join().unwrap(), ResultCode::SUCCESS);
    assert_eq!(
        word(m),
        0x222,
        "the waiter is the owner now, without waiters"
    );
    // A lock whose word no longer names the owner returns at once.
    assert_eq!(k.arbitrate_lock(b, 0x999, m, 0x222), ResultCode::SUCCESS);
    // An owner that is not a thread handle of the process is an invalid handle.
    set_word(m, 0x555 | WAIT);
    assert_eq!(k.arbitrate_lock(b, 0x555, m, 0x222), err::INVALID_HANDLE);
    // Unlocking a mutex nobody waits for clears it.
    assert_eq!(k.arbitrate_unlock(m), ResultCode::SUCCESS);
    assert_eq!(word(m), 0);
    // Unaligned addresses are refused.
    assert_eq!(k.arbitrate_unlock(m + 1), err::INVALID_ADDRESS);
}

#[test]
fn the_highest_priority_waiter_gets_the_mutex_first_and_the_next_still_waits() {
    let e = Env::new();
    let k = Arc::clone(&e.kernel);
    let m = SCRATCH + 0x80;
    let (lo_id, lo_h) = {
        let (h, id) = k.create_thread(0, 0, 0, 0x30, 0, false).unwrap();
        (id, h)
    };
    let (hi_id, hi_h) = {
        let (h, id) = k.create_thread(0, 0, 0, 0x10, 0, false).unwrap();
        (id, h)
    };
    let (owner, _) = k.create_thread(0, 0, 0, 0x2C, 0, false).unwrap();
    set_word(m, owner | WAIT);
    let low = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.arbitrate_lock(lo_id, owner, m, lo_h))
    };
    std::thread::sleep(Duration::from_millis(30));
    let high = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.arbitrate_lock(hi_id, owner, m, hi_h))
    };
    std::thread::sleep(Duration::from_millis(30));
    k.arbitrate_unlock(m);
    assert_eq!(high.join().unwrap(), ResultCode::SUCCESS);
    assert_eq!(
        word(m),
        hi_h | WAIT,
        "the new owner is told it has a waiter"
    );
    assert!(!low.is_finished());
    k.arbitrate_unlock(m);
    assert_eq!(low.join().unwrap(), ResultCode::SUCCESS);
    assert_eq!(word(m), lo_h);
}

#[test]
fn a_condition_variable_wait_releases_the_mutex_and_takes_it_back_when_signalled() {
    let e = Env::new();
    let k = Arc::clone(&e.kernel);
    let (m, key) = (SCRATCH + 0xC0, SCRATCH + 0xD0);
    let (w, _cpu) = second_thread(&k);
    set_word(m, 0x333); // the waiter holds the mutex
    set_word(key, 0);
    let waiter = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.wait_process_wide_key(w, m, key, 0x333, -1))
    };
    std::thread::sleep(Duration::from_millis(40));
    assert!(!waiter.is_finished());
    assert_eq!(word(m), 0, "waiting released the mutex");
    assert_eq!(word(key), 1, "the key word says somebody waits");
    k.signal_process_wide_key(key, 1);
    assert_eq!(waiter.join().unwrap(), ResultCode::SUCCESS);
    assert_eq!(word(m), 0x333, "and the waiter holds it again");
    assert_eq!(
        word(key),
        0,
        "nobody waits any more, so the key word is clear"
    );
    // Signalling with nobody waiting leaves the key word clear.
    set_word(key, 1);
    k.signal_process_wide_key(key, -1);
    assert_eq!(word(key), 0);
}

#[test]
fn a_signalled_waiter_queues_behind_the_current_owner_of_the_mutex() {
    let e = Env::new();
    let k = Arc::clone(&e.kernel);
    let (m, key) = (SCRATCH + 0x100, SCRATCH + 0x110);
    let (w, _cpu) = second_thread(&k);
    set_word(m, 0x333);
    let waiter = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.wait_process_wide_key(w, m, key, 0x333, -1))
    };
    std::thread::sleep(Duration::from_millis(40));
    // Somebody else takes the mutex before the signal.
    set_word(m, 0x444);
    k.signal_process_wide_key(key, 1);
    std::thread::sleep(Duration::from_millis(40));
    assert!(
        !waiter.is_finished(),
        "the mutex is still owned by someone else"
    );
    assert_eq!(word(m), 0x444 | WAIT);
    k.arbitrate_unlock(m);
    assert_eq!(waiter.join().unwrap(), ResultCode::SUCCESS);
    assert_eq!(word(m), 0x333);
}

#[test]
fn a_condition_variable_timeout_returns_without_the_mutex() {
    let e = Env::new();
    let k = Arc::clone(&e.kernel);
    let (m, key) = (SCRATCH + 0x140, SCRATCH + 0x150);
    let (w, _cpu) = second_thread(&k);
    set_word(m, 0x333);
    let r = k.wait_process_wide_key(w, m, key, 0x333, 20_000_000);
    assert_eq!(r, err::TIMED_OUT);
    assert_eq!(word(m), 0, "the mutex was released and not taken back");
    // The waiter is gone: a later signal finds nobody.
    k.signal_process_wide_key(key, 1);
}

#[test]
fn wait_for_address_and_signal_to_address() {
    let e = Env::new();
    let k = Arc::clone(&e.kernel);
    let a = SCRATCH + 0x180;
    let (w, _cpu) = second_thread(&k);
    set_word(a, 5);
    // Conditions that do not hold are an invalid state.
    assert_eq!(k.wait_for_address(w, a, 0, 5, -1), err::INVALID_STATE); // 5 < 5
    assert_eq!(k.wait_for_address(w, a, 2, 6, -1), err::INVALID_STATE); // 5 == 6
    assert_eq!(k.wait_for_address(w, a, 7, 6, -1), err::INVALID_ENUM_VALUE);
    // A condition that holds with a zero timeout times out without waiting.
    assert_eq!(k.wait_for_address(w, a, 0, 6, 0), err::TIMED_OUT);
    // A wait with a deadline expires.
    assert_eq!(k.wait_for_address(w, a, 2, 5, 20_000_000), err::TIMED_OUT);
    // A signal wakes a waiter blocked on equality.
    let waiter = {
        let k = Arc::clone(&k);
        std::thread::spawn(move || k.wait_for_address(w, a, 2, 5, -1))
    };
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(k.signal_to_address(a, 0, 0, 1), ResultCode::SUCCESS);
    assert_eq!(waiter.join().unwrap(), ResultCode::SUCCESS);
    // Decrement-and-wait decrements only when the comparison holds.
    set_word(a, 3);
    assert_eq!(k.wait_for_address(w, a, 1, 4, 20_000_000), err::TIMED_OUT);
    assert_eq!(word(a), 2);
    assert_eq!(
        k.wait_for_address(w, a, 1, 2, 20_000_000),
        err::INVALID_STATE
    );
    assert_eq!(word(a), 2);
    // Signal-and-increment needs the expected value.
    assert_eq!(k.signal_to_address(a, 1, 9, 1), err::INVALID_STATE);
    assert_eq!(k.signal_to_address(a, 1, 2, 1), ResultCode::SUCCESS);
    assert_eq!(word(a), 3);
}

#[test]
fn sleeping_yields_or_waits_and_the_end_of_the_process_interrupts_it() {
    let mut e = Env::new();
    let start = Instant::now();
    e.svc(0x0B, &[30_000_000]);
    assert!(start.elapsed() >= Duration::from_millis(25));
    for yield_kind in [0u64, u64::MAX, (-2i64) as u64] {
        e.svc(0x0B, &[yield_kind]);
    }
    let k = Arc::clone(&e.kernel);
    let sleeper = std::thread::spawn({
        let k = Arc::clone(&k);
        move || {
            let t = Instant::now();
            k.sleep(60_000_000_000);
            t.elapsed()
        }
    });
    std::thread::sleep(Duration::from_millis(30));
    k.exit_process(0);
    assert!(sleeper.join().unwrap() < Duration::from_secs(5));
}

// ---------------------------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------------------------

/// A service that adds two numbers (command 1), hands out another object (command 2), and counts
/// requests; anything else is an unknown command.
#[derive(Debug)]
struct Calc {
    calls: u32,
}

#[derive(Debug)]
struct Child(u32);

impl Service for Child {
    fn name(&self) -> &str {
        "tst:chd"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            0 => Ok(Response::ok().with_u32(self.0)),
            c => Err(Stop::unknown_command("tst:chd", c)),
        }
    }
}

impl Service for Calc {
    fn name(&self) -> &str {
        "tst:calc"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        self.calls += 1;
        match req.command {
            1 => Ok(Response::ok().with_u64(req.u64_at(0) + req.u64_at(8))),
            2 => Ok(Response::ok().with_object(Box::new(Child(7 + self.calls)))),
            3 => Ok(Response::ok().with_u32(req.pid.map_or(0, |p| p as u32))),
            4 => Ok(Response::error(ResultCode::new(2, 9))),
            c => Err(Stop::unknown_command("tst:calc", c)),
        }
    }
}

const SFCI: u32 = 0x4943_4653;

/// A libnx-style request in a buffer: no handles, the CMIF header 16-byte aligned.
fn request_bytes(command: u32, payload: &[u8], with_pid: bool, kind: u32) -> Vec<u8> {
    let mut m = vec![0u8; 8];
    let mut off: usize = 8;
    if with_pid {
        m.extend_from_slice(&1u32.to_le_bytes());
        m.extend_from_slice(&0u64.to_le_bytes());
        off += 12;
    }
    m.resize(off.next_multiple_of(16), 0);
    m.extend_from_slice(&SFCI.to_le_bytes());
    m.extend_from_slice(&0u32.to_le_bytes());
    m.extend_from_slice(&command.to_le_bytes());
    m.extend_from_slice(&0u32.to_le_bytes());
    m.extend_from_slice(payload);
    let words = (m.len() - off).div_ceil(4);
    m[0..4].copy_from_slice(&kind.to_le_bytes());
    m[4..8].copy_from_slice(&(words as u32 | (u32::from(with_pid) << 31)).to_le_bytes());
    m
}

fn domain_request(object: u32, command: u32, payload: &[u8]) -> Vec<u8> {
    let inner = {
        let mut b = Vec::new();
        b.extend_from_slice(&SFCI.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&command.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(payload);
        b
    };
    let mut raw = vec![1u8, 0];
    raw.extend_from_slice(&(inner.len() as u16).to_le_bytes());
    raw.extend_from_slice(&object.to_le_bytes());
    raw.extend_from_slice(&[0u8; 8]);
    raw.extend_from_slice(&inner);
    let mut m = vec![0u8; 8];
    m.resize(16, 0);
    m.extend_from_slice(&raw);
    let words = (m.len() - 8).div_ceil(4);
    m[0..4].copy_from_slice(&4u32.to_le_bytes());
    m[4..8].copy_from_slice(&(words as u32).to_le_bytes());
    m
}

/// Everything after the CMIF output header of a reply, and its result.
struct Reply {
    result: u32,
    data: Vec<u8>,
    moved: Vec<u32>,
    copied: Vec<u32>,
    object_ids: Vec<u32>,
}

impl Env {
    fn tls(&self) -> u64 {
        self.cpu.state.tpidrro_el0
    }

    /// Send a message on a session handle and parse the reply. `domain` says the session is one.
    fn send(&mut self, session: u64, message: &[u8], domain: bool) -> Option<Reply> {
        let mut buf = message.to_vec();
        buf.resize(0x100, 0);
        self.write(self.tls(), &buf);
        let flow = self.svc(0x21, &[session]);
        if flow == Flow::Stop {
            return None;
        }
        assert_eq!(self.r(), 0, "SendSyncRequest failed");
        let out = self.read(self.tls(), 0x100);
        let h1 = u32::from_le_bytes(out[4..8].try_into().unwrap());
        let mut off = 8;
        let mut moved = Vec::new();
        let mut copied = Vec::new();
        if h1 >> 31 != 0 {
            let tag = u32::from_le_bytes(out[8..12].try_into().unwrap());
            off = 12;
            for _ in 0..(tag >> 1) & 0xF {
                copied.push(u32::from_le_bytes(out[off..off + 4].try_into().unwrap()));
                off += 4;
            }
            for _ in 0..(tag >> 5) & 0xF {
                moved.push(u32::from_le_bytes(out[off..off + 4].try_into().unwrap()));
                off += 4;
            }
        }
        let raw_start = off.next_multiple_of(16);
        let raw_end = off + 4 * (h1 & 0x3FF) as usize;
        let mut p = raw_start;
        let mut nobjects = 0;
        if domain {
            nobjects = u32::from_le_bytes(out[p..p + 4].try_into().unwrap()) as usize;
            p += 16;
        }
        assert_eq!(&out[p..p + 4], b"SFCO");
        let result = u32::from_le_bytes(out[p + 8..p + 12].try_into().unwrap());
        let body_end = raw_end - 4 * nobjects;
        let data = out[p + 16..body_end].to_vec();
        let object_ids = (0..nobjects)
            .map(|i| {
                u32::from_le_bytes(
                    out[body_end + 4 * i..body_end + 4 * i + 4]
                        .try_into()
                        .unwrap(),
                )
            })
            .collect();
        Some(Reply {
            result,
            data,
            moved,
            copied,
            object_ids,
        })
    }

    fn connect_sm(&mut self) -> u64 {
        self.write(SCRATCH + 0x200, b"sm:\0");
        self.svc(0x1F, &[0, SCRATCH + 0x200]);
        assert_eq!(self.r(), 0);
        self.x(1)
    }

    /// `sm:` initialization and a `GetServiceHandle`.
    fn get_service(&mut self, sm: u64, name: &str) -> Option<Reply> {
        let mut n = [0u8; 8];
        n[..name.len()].copy_from_slice(name.as_bytes());
        let m = request_bytes(1, &n, false, 4);
        self.send(sm, &m, false)
    }
}

fn calc_env(npdm: Option<Npdm>) -> Env {
    let e = Env::with(ProcessKind::Nso, npdm, Vec::new());
    e.kernel
        .register_service("tst:calc", Box::new(|| Box::new(Calc { calls: 0 })));
    e
}

fn svcs_for_ipc() -> Vec<u32> {
    // SVC ids are 24 per descriptor: 6, 11 and 22 in the first; 0x18, 0x1F and 0x21 in the second.
    vec![
        builder::enable_svcs(0, &[6, 11, 22]),
        builder::enable_svcs(1, &[0, 7, 9]),
    ]
}

#[test]
fn a_program_asks_sm_for_a_service_and_calls_it() {
    let mut e = calc_env(None);
    let sm = e.connect_sm();
    // Without RegisterClient first, GetServiceHandle is refused.
    let r = e.get_service(sm, "tst:calc").unwrap();
    assert_eq!(r.result, 0x415);
    // RegisterClient carries the caller's process id.
    let m = request_bytes(0, &[0; 8], true, 4);
    let r = e.send(sm, &m, false).unwrap();
    assert_eq!(r.result, 0);
    let r = e.get_service(sm, "tst:calc").unwrap();
    assert_eq!(r.result, 0);
    assert_eq!(r.moved.len(), 1);
    let calc = u64::from(r.moved[0]);
    // calc.Add(40, 2)
    let mut payload = Vec::new();
    payload.extend_from_slice(&40u64.to_le_bytes());
    payload.extend_from_slice(&2u64.to_le_bytes());
    let r = e
        .send(calc, &request_bytes(1, &payload, false, 4), false)
        .unwrap();
    assert_eq!(r.result, 0);
    assert_eq!(u64::from_le_bytes(r.data[..8].try_into().unwrap()), 42);
    // A service error comes back as the IPC result, not as a kernel error.
    let r = e
        .send(calc, &request_bytes(4, &[], false, 4), false)
        .unwrap();
    assert_eq!(r.result, ResultCode::new(2, 9).0);
    // The pid in the header is the process's.
    let r = e
        .send(calc, &request_bytes(3, &[], true, 4), false)
        .unwrap();
    assert_eq!(
        u32::from_le_bytes(r.data[..4].try_into().unwrap()),
        e.kernel.process.process_id as u32
    );
    // A returned object becomes a new session of its own.
    let r = e
        .send(calc, &request_bytes(2, &[], false, 4), false)
        .unwrap();
    let child = u64::from(r.moved[0]);
    let r = e
        .send(child, &request_bytes(0, &[], false, 4), false)
        .unwrap();
    assert_eq!(u32::from_le_bytes(r.data[..4].try_into().unwrap()), 7 + 4);
    // Control commands: QueryPointerBufferSize and CloneCurrentObject.
    let r = e
        .send(calc, &request_bytes(3, &[], false, 5), false)
        .unwrap();
    assert_eq!(u32::from_le_bytes(r.data[..4].try_into().unwrap()), 0);
    let r = e
        .send(calc, &request_bytes(2, &[], false, 5), false)
        .unwrap();
    let clone = u64::from(r.moved[0]);
    let r = e
        .send(clone, &request_bytes(1, &payload, false, 4), false)
        .unwrap();
    assert_eq!(u64::from_le_bytes(r.data[..8].try_into().unwrap()), 42);
    // Closing a handle ends the session; sending on it fails.
    e.svc(0x16, &[calc]);
    e.svc(0x21, &[calc]);
    assert_eq!(e.r(), err::INVALID_HANDLE.0);
}

#[test]
fn a_domain_holds_many_objects_behind_one_handle() {
    let mut e = calc_env(None);
    let sm = e.connect_sm();
    e.send(sm, &request_bytes(0, &[0; 8], true, 4), false)
        .unwrap();
    let calc = u64::from(e.get_service(sm, "tst:calc").unwrap().moved[0]);
    // ConvertCurrentObjectToDomain answers with the root object's id.
    let r = e
        .send(calc, &request_bytes(0, &[], false, 5), false)
        .unwrap();
    assert_eq!(u32::from_le_bytes(r.data[..4].try_into().unwrap()), 1);
    // A request names its target object; objects the service returns arrive as ids.
    let mut payload = Vec::new();
    payload.extend_from_slice(&5u64.to_le_bytes());
    payload.extend_from_slice(&6u64.to_le_bytes());
    let r = e.send(calc, &domain_request(1, 1, &payload), true).unwrap();
    assert_eq!(r.result, 0);
    assert_eq!(u64::from_le_bytes(r.data[..8].try_into().unwrap()), 11);
    let r = e.send(calc, &domain_request(1, 2, &[]), true).unwrap();
    assert_eq!(r.object_ids.len(), 1);
    assert!(r.moved.is_empty());
    let child = r.object_ids[0];
    assert!(child >= 2);
    let r = e.send(calc, &domain_request(child, 0, &[]), true).unwrap();
    assert_eq!(u32::from_le_bytes(r.data[..4].try_into().unwrap()), 7 + 2);
    // An id that does not exist is an invalid handle, answered in the IPC result.
    let r = e.send(calc, &domain_request(99, 0, &[]), true).unwrap();
    assert_eq!(r.result, err::INVALID_HANDLE.0);
}

#[test]
fn services_the_program_may_not_use_are_refused_and_unimplemented_ones_stop_the_run() {
    let spec = builder::NpdmSpec {
        services: vec![("sm:", false), ("tst:calc", false), ("hid", false)],
        kc: svcs_for_ipc(),
        ..builder::NpdmSpec::default()
    };
    let npdm = Npdm::parse(&builder::build(&spec)).unwrap();
    {
        let mut e = calc_env(Some(npdm));
        let sm = e.connect_sm();
        e.send(sm, &request_bytes(0, &[0; 8], true, 4), false)
            .unwrap();
        // `fsp-srv` is not in the list: refused with the service manager's error.
        let r = e.get_service(sm, "fsp-srv").unwrap();
        assert_eq!(r.result, 0x1015);
        // `hid` is allowed and nothing implements it: the run stops and names it.
        assert!(e.get_service(sm, "hid").is_none());
        match e.kernel.exit_state() {
            Some(Exit::Fatal(m)) => {
                assert!(m.contains("`hid`") && m.contains("not implement"), "{m}")
            }
            other => panic!("{other:?}"),
        }
    }
    // An invalid name.
    let mut e = calc_env(None);
    let sm = e.connect_sm();
    e.send(sm, &request_bytes(0, &[0; 8], true, 4), false)
        .unwrap();
    assert_eq!(e.get_service(sm, "").unwrap().result, 0xC15);
}

#[test]
fn unknown_commands_and_ports_stop_the_run_with_their_numbers() {
    {
        let mut e = calc_env(None);
        let sm = e.connect_sm();
        e.send(sm, &request_bytes(0, &[0; 8], true, 4), false)
            .unwrap();
        let calc = u64::from(e.get_service(sm, "tst:calc").unwrap().moved[0]);
        assert!(e
            .send(calc, &request_bytes(77, &[], false, 4), false)
            .is_none());
        match e.kernel.exit_state() {
            Some(Exit::Fatal(m)) => {
                assert!(
                    m.contains("tst:calc") && m.contains("command 77") && m.contains("SVC at"),
                    "{m}"
                )
            }
            other => panic!("{other:?}"),
        }
    }
    {
        let mut e = calc_env(None);
        let sm = e.connect_sm();
        assert!(e
            .send(sm, &request_bytes(9, &[], false, 4), false)
            .is_none());
        match e.kernel.exit_state() {
            Some(Exit::Fatal(m)) => assert!(m.contains("sm:") && m.contains("command 9"), "{m}"),
            other => panic!("{other:?}"),
        }
    }
    let mut e = calc_env(None);
    e.write(SCRATCH + 0x200, b"hid:sys\0");
    assert_eq!(e.svc(0x1F, &[0, SCRATCH + 0x200]), Flow::Stop);
    match e.kernel.exit_state() {
        Some(Exit::Fatal(m)) => assert!(m.contains("hid:sys"), "{m}"),
        other => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Whole processes
// ---------------------------------------------------------------------------------------------

/// A native stand-in for a homebrew `main`: print a line, then return 7 from the entry function.
extern "C" fn nro_main(p: *mut NativeCpu) {
    // SAFETY: the runtime passes the running thread's CPU.
    let c = unsafe { &mut *p };
    // The loader configuration: X0 points at it, X1 is all ones.
    assert_eq!(c.state.x[1], u64::MAX);
    let mut at = c.state.x[0];
    let mut main_handle = 0;
    let mut hos_version = 0;
    let mut hint = 0;
    loop {
        let key = rt().mem.read(at, 4, Relaxed).unwrap() as u32;
        let v0 = rt().mem.read(at + 8, 8, Relaxed).unwrap() as u64;
        match key {
            0 => break,
            1 => main_handle = v0,
            6 => hint = v0,
            16 => hos_version = v0,
            _ => {}
        }
        at += 24;
    }
    // Report what was found as the output string.
    let text = format!("handle {main_handle} hos {hos_version:#x} hint {hint:#x}\n");
    rt().mem
        .write_bytes(SCRATCH, text.as_bytes(), false)
        .unwrap();
    c.state.x[0] = SCRATCH;
    c.state.x[1] = text.len() as u64;
    c.state.pc = 0x0800_0104;
    svc_entry(p, 0x27);
    c.state.x[0] = 7;
    c.state.pc = c.state.x[30];
}

#[test]
fn an_nro_is_entered_with_the_loader_configuration_and_its_return_value_is_the_exit_status() {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = rt().mem.unmap(SCRATCH, SCRATCH_SIZE);
    rt().mem
        .map(SCRATCH, SCRATCH_SIZE, perm::R | perm::W)
        .unwrap();
    register(0x0800_0100, nro_main);
    let kernel = Kernel::new(ProcessInfo::new("hello", ProcessKind::Nro, None));
    let out = Arc::new(Mutex::new(Vec::new()));
    kernel.set_output(Box::new(Capture(Arc::clone(&out))));
    let exit = kernel.run_process(&StartConfig {
        entry: 0x0800_0100,
        segments: Vec::new(),
    });
    assert_eq!(exit, Exit::Code(7));
    let printed = String::from_utf8_lossy(&out.lock().unwrap()).into_owned();
    assert!(printed.starts_with("handle "), "{printed}");
    assert!(printed.contains("hos 0xc0100"), "{printed}");
    // The hint names the SVCs this kernel implements (QueryMemory is 6, OutputDebugString 0x27).
    let hint = u64::from_str_radix(printed.split("hint 0x").nth(1).unwrap().trim(), 16).unwrap();
    assert!(hint & 1 << 6 != 0 && hint & 1 << 0x27 != 0 && hint & 1 << 0x2A == 0);
    let _ = rt().mem.unmap(SCRATCH, SCRATCH_SIZE);
    let _ = rt().mem.read(ENV_BASE, 1, Relaxed);
    drop(kernel);
    assert!(
        rt().mem.read(ENV_BASE, 1, Relaxed).is_err(),
        "the kernel unmapped its memory"
    );
    drop(guard);
}

/// An NSO `main` that makes the process fail: it reads memory nobody mapped.
extern "C" fn faulting_main(p: *mut NativeCpu) {
    // SAFETY: the runtime passes the running thread's CPU.
    let c = unsafe { &mut *p };
    c.state.pc = c.state.x[30];
    crate::rt::runtime::record_fault(
        c,
        0xDEAD_0000,
        0x0800_0200,
        false,
        crate::a64::FaultKind::Unmapped,
    );
}

#[test]
fn a_fault_in_any_thread_ends_the_process_with_a_report() {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register(0x0800_0200, faulting_main);
    let kernel = Kernel::new(ProcessInfo::new("faulty", ProcessKind::Nso, None));
    let exit = kernel.run_process(&StartConfig {
        entry: 0x0800_0200,
        segments: Vec::new(),
    });
    match exit {
        Exit::Fatal(m) => assert!(
            m.contains("read of 0xdead0000") && m.contains("0x8000200"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }
    drop(kernel);
    drop(guard);
}

/// An NSO `main` that spawns a worker, waits for it and returns the worker's result.
extern "C" fn spawning_main(p: *mut NativeCpu) {
    // SAFETY: the runtime passes the running thread's CPU.
    let c = unsafe { &mut *p };
    let call = |c: &mut NativeCpu, imm: u32, args: &[u64]| {
        for (i, a) in args.iter().enumerate() {
            c.state.x[i] = *a;
        }
        c.state.pc = 0x0800_0304;
        assert_eq!(svc_entry(c, imm), 0);
    };
    call(c, 0x01, &[0, 0x20_0000]);
    let heap = c.state.x[1];
    call(
        c,
        0x08,
        &[0, 0x0800_0400, heap, heap + 0x20_0000 - 0x1000, 0x2C, 0],
    );
    let thread = c.state.x[1];
    call(c, 0x09, &[thread]);
    rt().mem
        .write(heap + 0x100, 4, u128::from(thread as u32), Relaxed)
        .unwrap();
    call(c, 0x18, &[0, heap + 0x100, 1, u64::MAX]);
    assert_eq!(c.state.x[0], 0);
    c.state.x[0] = rt().mem.read(heap, 8, Relaxed).unwrap() as u64;
    c.state.pc = c.state.x[30];
}

#[test]
fn the_main_thread_can_start_another_and_wait_for_it() {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register(0x0800_0300, spawning_main);
    register(0x0800_0400, thread_worker);
    let kernel = Kernel::new(ProcessInfo::new("threads", ProcessKind::Nso, None));
    let exit = kernel.run_process(&StartConfig {
        entry: 0x0800_0300,
        segments: Vec::new(),
    });
    // The worker stored 0x1000 + the first word of its (zeroed) thread-local region.
    assert_eq!(exit, Exit::Code(0x1000));
    drop(kernel);
    drop(guard);
}

// ---------------------------------------------------------------------------------------------
// The services themselves, through the kernel
// ---------------------------------------------------------------------------------------------

/// A request with handles and buffer descriptors, laid out the way the Nintendo SDK does.
struct Message {
    pid: bool,
    copy_handles: Vec<u32>,
    /// Send (A) descriptors: what the service may read.
    send: Vec<(u64, u64)>,
    /// Receive (B) descriptors: where the service may write.
    receive: Vec<(u64, u64)>,
    command: u32,
    payload: Vec<u8>,
}

impl Message {
    fn new(command: u32) -> Message {
        Message {
            pid: false,
            copy_handles: Vec::new(),
            send: Vec::new(),
            receive: Vec::new(),
            command,
            payload: Vec::new(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut m = vec![0u8; 8];
        let special = self.pid || !self.copy_handles.is_empty();
        if special {
            let tag = u32::from(self.pid) | ((self.copy_handles.len() as u32) << 1);
            m.extend_from_slice(&tag.to_le_bytes());
            if self.pid {
                m.extend_from_slice(&0u64.to_le_bytes());
            }
            for h in &self.copy_handles {
                m.extend_from_slice(&h.to_le_bytes());
            }
        }
        for &(addr, size) in self.send.iter().chain(&self.receive) {
            // size[31:0], address[31:0], then address[38:36] in bits 2-4, size[35:32] in 24-27
            // and address[35:32] in 28-31 of the last word.
            m.extend_from_slice(&(size as u32).to_le_bytes());
            m.extend_from_slice(&(addr as u32).to_le_bytes());
            let d2 = (((addr >> 32) & 0xF) << 28) as u32
                | (((size >> 32) & 0xF) << 24) as u32
                | (((addr >> 36) & 0x7) << 2) as u32;
            m.extend_from_slice(&d2.to_le_bytes());
        }
        let after_descriptors = m.len();
        m.resize(after_descriptors.next_multiple_of(16), 0);
        m.extend_from_slice(&SFCI.to_le_bytes());
        m.extend_from_slice(&0u32.to_le_bytes());
        m.extend_from_slice(&self.command.to_le_bytes());
        m.extend_from_slice(&0u32.to_le_bytes());
        m.extend_from_slice(&self.payload);
        let words = (m.len() - after_descriptors).div_ceil(4);
        let h0 = 4u32 | ((self.send.len() as u32) << 20) | ((self.receive.len() as u32) << 24);
        m[0..4].copy_from_slice(&h0.to_le_bytes());
        m[4..8].copy_from_slice(&(words as u32 | (u32::from(special) << 31)).to_le_bytes());
        m
    }
}

impl Env {
    /// Connect to `sm:`, register, and get the named service's session handle.
    fn service(&mut self, name: &str) -> u64 {
        let sm = self.connect_sm();
        let r = self
            .send(sm, &request_bytes(0, &[0; 8], true, 4), false)
            .unwrap();
        assert_eq!(r.result, 0);
        let r = self.get_service(sm, name).unwrap();
        assert_eq!(r.result, 0, "{name}");
        u64::from(r.moved[0])
    }

    fn call(&mut self, session: u64, m: &Message) -> Reply {
        self.send(session, &m.bytes(), false).unwrap()
    }
}

#[test]
fn a_transfer_memory_locks_its_range_until_the_handle_is_closed() {
    let mut e = Env::new();
    e.svc(0x01, &[0, 0x20_0000]);
    let range = HEAP_BASE + 0x10_0000;
    // Arguments are checked: alignment, size, permission, and the memory has to be plain heap.
    e.svc(0x15, &[0, range + 1, 0x2000, 0]);
    assert_eq!(e.r(), err::INVALID_ADDRESS.0);
    e.svc(0x15, &[0, range, 0, 0]);
    assert_eq!(e.r(), err::INVALID_SIZE.0);
    e.svc(0x15, &[0, range, 0x2000, 5]);
    assert_eq!(e.r(), err::INVALID_NEW_MEMORY_PERMISSION.0);
    e.svc(0x15, &[0, HEAP_BASE + 0x40_0000, 0x1000, 0]);
    assert_eq!(e.r(), err::INVALID_CURRENT_MEMORY.0);

    rt().mem.write(range, 8, 0xFEED, Relaxed).unwrap();
    e.svc(0x15, &[0, range, 0x2000, 0]);
    assert_eq!(e.r(), 0);
    let handle = e.x(1);
    // The program has no access meanwhile, and the block reads as locked with no permission.
    assert!(rt().mem.read(range, 8, Relaxed).is_err());
    e.svc(0x06, &[SCRATCH, 0, range]);
    let info = e.read(SCRATCH, 0x28);
    assert_eq!((info[16], info[20], info[24]), (mem_type::NORMAL, 1, 0));
    // A range that is lent cannot be lent again.
    e.svc(0x15, &[0, range, 0x1000, 0]);
    assert_eq!(e.r(), err::INVALID_CURRENT_MEMORY.0);
    // The service side sees the range through the object.
    let object = e.kernel.object(handle as u32, e.tid).unwrap();
    assert_eq!(e.kernel.transfer_memory(&object), Some((range, 0x2000)));
    // Closing the handle gives the memory back with its contents.
    e.svc(0x16, &[handle]);
    assert_eq!(e.r(), 0);
    assert_eq!(rt().mem.read(range, 8, Relaxed), Ok(0xFEED));
    e.svc(0x06, &[SCRATCH, 0, range]);
    let info = e.read(SCRATCH, 0x28);
    assert_eq!((info[16], info[20], info[24]), (mem_type::NORMAL, 0, 3));
    assert_eq!(e.kernel.transfer_memory(&object), None);
}

#[test]
fn the_applet_manager_announces_focus_once_and_then_has_nothing_to_say() {
    let mut e = Env::new();
    super::services::register_defaults(&e.kernel);
    let applet = e.service("appletOE");
    // OpenApplicationProxy(u64 reserved, pid, the process handle) -> IApplicationProxy
    let mut open = Message::new(0);
    open.pid = true;
    open.copy_handles = vec![CURRENT_PROCESS];
    open.payload = vec![0; 8];
    let proxy = u64::from(e.call(applet, &open).moved[0]);
    // GetCommonStateGetter, then its event.
    let state = u64::from(e.call(proxy, &Message::new(0)).moved[0]);
    let event = u64::from(e.call(state, &Message::new(0)).copied[0]);
    // The focus message is waiting, so the event is signalled.
    e.write(SCRATCH + 0x400, &(event as u32).to_le_bytes());
    e.svc(0x18, &[0, SCRATCH + 0x400, 1, 0]);
    assert_eq!((e.r(), e.x(1)), (0, 0));
    let r = e.call(state, &Message::new(1));
    assert_eq!(r.result, 0);
    assert_eq!(u32::from_le_bytes(r.data[..4].try_into().unwrap()), 15);
    // Nothing else is queued: the event clears and the next ReceiveMessage reports no message.
    e.svc(0x18, &[0, SCRATCH + 0x400, 1, 0]);
    assert_eq!(e.r(), err::TIMED_OUT.0);
    assert_eq!(e.call(state, &Message::new(1)).result, 0x680);
    // The state it reports: in focus, handheld.
    let r = e.call(state, &Message::new(9));
    assert_eq!(r.data[0], 1);
    let r = e.call(state, &Message::new(5));
    assert_eq!(r.data[0], 0);
}

#[test]
fn the_romfs_is_read_through_the_file_system_proxy() {
    use crate::testing::TempDir;
    let dir = TempDir::new("romfs-ipc");
    let pattern: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
    dir.write("data/pattern.bin", &pattern);
    dir.write("readme.txt", b"hello");
    let image = crate::romfs::open_path(dir.path()).unwrap();
    let mut e = Env::with_romfs(dir.path());
    let fs = e.service("fsp-srv");
    // SetCurrentProcess(u64 reserved, pid)
    let mut set = Message::new(1);
    set.pid = true;
    set.payload = vec![0; 8];
    assert_eq!(e.call(fs, &set).result, 0);
    assert_eq!(e.call(fs, &Message::new(1005)).data, [0, 0, 0, 0]);
    // OpenDataStorageByCurrentProcess() -> IStorage
    let storage = u64::from(e.call(fs, &Message::new(200)).moved[0]);
    let size = u64::from_le_bytes(
        e.call(storage, &Message::new(4)).data[..8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(size, image.size());
    // Read(s64 offset, u64 size) into a buffer the caller gave; compare with the image itself.
    for (offset, len) in [(0u64, 0x50u64), (0x10, 0x200), (size - 100, 100), (0, size)] {
        let mut read = Message::new(0);
        read.payload.extend_from_slice(&offset.to_le_bytes());
        read.payload.extend_from_slice(&len.to_le_bytes());
        read.receive = vec![(SCRATCH + 0x1000, len)];
        let r = e.call(storage, &read);
        assert_eq!(r.result, 0, "{offset:#x} {len:#x}");
        let mut want = vec![0u8; len as usize];
        image.read_at(offset, &mut want).unwrap();
        assert_eq!(e.read(SCRATCH + 0x1000, len as usize), want, "{offset:#x}");
    }
    // The file data is in the image where the tables say.
    let rom = crate::romfs::RomFs::open(image.clone()).unwrap();
    assert_eq!(rom.read_file("/data/pattern.bin").unwrap(), pattern);
    // A read past the end is `ResultOutOfRange`, 2002-3005; so is one whose end overflows.
    for (offset, len) in [(size - 4, 16u64), (u64::MAX - 3, 8)] {
        let mut read = Message::new(0);
        read.payload.extend_from_slice(&offset.to_le_bytes());
        read.payload.extend_from_slice(&len.to_le_bytes());
        read.receive = vec![(SCRATCH + 0x1000, len)];
        let r = e.call(storage, &read);
        assert_eq!(r.result, ResultCode::new(2, 3005).0);
        assert_eq!(r.result, 0x0017_7A02);
    }
    assert!(e.kernel.exit_state().is_none());
}

#[test]
fn asking_for_a_romfs_the_process_does_not_have_stops_the_run() {
    let mut e = Env::new();
    super::services::register_defaults(&e.kernel);
    let fs = e.service("fsp-srv");
    assert!(e.send(fs, &Message::new(200).bytes(), false).is_none());
    match e.kernel.exit_state() {
        Some(Exit::Fatal(m)) => assert!(m.contains("--romfs") && m.contains("NX_ROMFS"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_storage_the_program_creates_holds_what_it_writes() {
    let mut e = Env::new();
    super::services::register_defaults(&e.kernel);
    let applet = e.service("appletOE");
    let mut open = Message::new(0);
    open.pid = true;
    open.copy_handles = vec![CURRENT_PROCESS];
    open.payload = vec![0; 8];
    let proxy = u64::from(e.call(applet, &open).moved[0]);
    // GetLibraryAppletCreator, CreateStorage(0x40), IStorage::Open.
    let creator = u64::from(e.call(proxy, &Message::new(11)).moved[0]);
    let mut create = Message::new(10);
    create.payload = 0x40u64.to_le_bytes().to_vec();
    let storage = u64::from(e.call(creator, &create).moved[0]);
    let accessor = u64::from(e.call(storage, &Message::new(0)).moved[0]);
    let size = |e: &mut Env| {
        u64::from_le_bytes(
            e.call(accessor, &Message::new(0)).data[..8]
                .try_into()
                .unwrap(),
        )
    };
    assert_eq!(size(&mut e), 0x40);
    // Write(s64 offset, buffer in), then Read(s64 offset, buffer out).
    e.write(SCRATCH + 0x800, b"storage!");
    let mut write = Message::new(10);
    write.payload = 4i64.to_le_bytes().to_vec();
    write.send = vec![(SCRATCH + 0x800, 8)];
    assert_eq!(e.call(accessor, &write).result, 0);
    let mut read = Message::new(11);
    read.payload = 0i64.to_le_bytes().to_vec();
    read.receive = vec![(SCRATCH + 0x900, 16)];
    assert_eq!(e.call(accessor, &read).result, 0);
    assert_eq!(e.read(SCRATCH + 0x900, 16), b"\0\0\0\0storage!\0\0\0\0");
    // A write or read that leaves the storage is 2128-0503, and a negative offset too.
    let out_of_bounds = ResultCode::new(128, 503).0;
    write.payload = 0x3Ci64.to_le_bytes().to_vec();
    assert_eq!(e.call(accessor, &write).result, out_of_bounds);
    read.payload = (-1i64).to_le_bytes().to_vec();
    assert_eq!(e.call(accessor, &read).result, out_of_bounds);
    // Nothing was changed by the refused write.
    read.payload = 0x38i64.to_le_bytes().to_vec();
    read.receive = vec![(SCRATCH + 0x900, 8)];
    assert_eq!(e.call(accessor, &read).result, 0);
    assert_eq!(e.read(SCRATCH + 0x900, 8), [0u8; 8]);
    // PopLaunchParameter: this program was started with none (2128-0002).
    let functions = u64::from(e.call(proxy, &Message::new(20)).moved[0]);
    let mut pop = Message::new(1);
    pop.payload = 1u32.to_le_bytes().to_vec();
    assert_eq!(e.call(functions, &pop).result, ResultCode::new(128, 2).0);
}
