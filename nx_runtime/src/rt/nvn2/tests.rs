use super::*;
use crate::rt::abi::{perm, FuncEntry, STOP_PC};

const RESOLVER: u64 = 0x7D00_0100;
const LEAF: u64 = 0x7D00_0200;
const RETURN: u64 = 0x0800_1004;

struct Memory(u64);

impl Memory {
    fn new() -> Memory {
        static NEXT: AtomicU64 = AtomicU64::new(0x51_0000_0000);
        let addr = NEXT.fetch_add(0x2000, Ordering::Relaxed);
        rt().mem.map(addr, 0x1000, perm::R | perm::W).unwrap();
        Memory(addr)
    }

    fn name(&self, offset: u64, text: &str) -> u64 {
        let addr = self.0 + offset;
        rt().mem.write_bytes(addr, text.as_bytes(), true).unwrap();
        rt().mem
            .write_bytes(addr + text.len() as u64, &[0], true)
            .unwrap();
        addr
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        rt().mem.unmap(self.0, 0x1000).unwrap();
    }
}

extern "C" fn guest_bootstrap(p: *mut NativeCpu) {
    // SAFETY: the test passes its live, exclusively used CPU.
    let c = unsafe { &mut *p };
    c.state.x[0] = RESOLVER;
    c.state.pc = c.state.x[30];
}

extern "C" fn guest_resolver(p: *mut NativeCpu) {
    // SAFETY: as in guest_bootstrap.
    let c = unsafe { &mut *p };
    c.state.x[0] = if matches!(read_name(c.state.x[1]), Ok(n) if n == "nvnSynthetic") {
        LEAF
    } else {
        0
    };
    c.state.pc = c.state.x[30];
}

extern "C" fn guest_leaf(p: *mut NativeCpu) {
    // SAFETY: as in guest_bootstrap. Exercise FP/vector arguments, stack, TLS, flags and integer
    // arguments beyond x0; a trace must produce exactly the state the guest call would have.
    let c = unsafe { &mut *p };
    c.state.x[0] = c.state.x[1].wrapping_add(c.state.x[7]);
    c.state.x[9] = c.state.sp ^ c.state.tpidr_el0 ^ c.state.tpidrro_el0;
    c.state.v[0] = c.state.v[3];
    c.state.nzcv ^= 0xF000_0000;
    c.state.fpsr ^= 0x10;
    c.state.pc = c.state.x[30];
}

extern "C" fn guest_unwinds(p: *mut NativeCpu) {
    // SAFETY: as in guest_bootstrap.
    let c = unsafe { &mut *p };
    c.state.x[0] = LEAF;
    c.state.pc = 0x0800_2000;
}

extern "C" fn forbidden_guest(p: *mut NativeCpu) {
    // SAFETY: as in guest_bootstrap. If HLE calls the guest driver the test sees the trap.
    let c = unsafe { &mut *p };
    record_stop(c, stop::TRAP, 0xBAD, 0);
}

fn cpu() -> NativeCpu {
    let mut c = NativeCpu::new(0);
    c.state.x[30] = RETURN;
    c
}

#[test]
fn a_disabled_loader_runs_its_preserved_guest_body() {
    let mut c = cpu();
    c.state.x[0] = 123;
    bootstrap(&mut c, 0x4000, guest_bootstrap);
    assert_eq!(c.state.x[0], RESOLVER);
    assert_eq!(c.state.pc, RETURN);
}

#[test]
fn trace_follows_the_bootstrap_and_device_resolver_and_preserves_the_call_abi() {
    rt().register(&[
        FuncEntry {
            addr: RESOLVER,
            func: guest_resolver as *const c_void,
        },
        FuncEntry {
            addr: LEAF,
            func: guest_leaf as *const c_void,
        },
    ]);
    let mem = Memory::new();
    let session = Session::new(Mode::Trace);
    let mut c = cpu();
    c.state.x[0] = mem.name(0, "nvnDeviceGetProcAddress");
    session.bootstrap(&mut c, 0x4000, guest_bootstrap);
    let resolver = c.state.x[0];
    assert_eq!(session.proc(resolver).unwrap().guest, RESOLVER);
    c.state.pc = resolver;
    c.state.x[0] = 42; // device, passed to the guest resolver unchanged
    c.state.x[1] = mem.name(0x100, "nvnSynthetic");
    session.call(&mut c);
    let leaf = c.state.x[0];
    assert_eq!(session.proc(leaf).unwrap().guest, LEAF);
    assert_eq!(session.bind("nvnSynthetic", LEAF).unwrap(), leaf);
    c.state.x[1] = 17;
    c.state.x[7] = 31;
    c.state.sp = mem.0 + 0x800;
    c.state.tpidr_el0 = 0x1234;
    c.state.tpidrro_el0 = 0x5678;
    c.state.v[3] = [0x4080_0000, 0x1122_3344_5566_7788];
    c.state.nzcv = 0x6000_0000;
    c.state.fpsr = 0x80;
    let mut expected = cpu();
    expected.state = c.state.clone();
    guest_leaf(&mut expected);
    c.state.pc = leaf;
    session.call(&mut c);
    assert_eq!(c.state, expected.state);
    assert_eq!(c.ctl.stop, stop::NONE);
    let report = session.report();
    assert!(
        report.contains("1 0x7d000100 nvnDeviceGetProcAddress"),
        "{report}"
    );
    assert!(report.contains("1 0x7d000200 nvnSynthetic"), "{report}");
}

#[test]
fn trace_keeps_null_results_unwinds_and_distinct_device_entries() {
    let session = Session::new(Mode::Trace);
    assert_eq!(session.bind("nvnAbsent", 0).unwrap(), 0);
    let a = session.bind("nvnSynthetic", LEAF).unwrap();
    let b = session.bind("nvnSynthetic", LEAF + 4).unwrap();
    assert_ne!(a, b);
    assert_eq!(session.bind("nvnSynthetic", LEAF).unwrap(), a);
    let mem = Memory::new();
    let mut c = cpu();
    c.state.x[0] = mem.name(0, "nvnDeviceGetProcAddress");
    session.bootstrap(&mut c, 0x4000, guest_unwinds);
    assert_eq!(c.state.x[0], LEAF);
    assert_eq!(c.state.pc, 0x0800_2000);
    session.bootstrap(&mut c, 0x4000, forbidden_guest);
    assert_eq!(c.ctl.stop, stop::TRAP);
    assert_eq!(c.state.pc, STOP_PC);
}

#[test]
fn hle_owns_resolution_and_device_builders_without_calling_the_guest_driver() {
    let mem = Memory::new();
    let session = Session::new(Mode::Hle);
    let mut c = cpu();
    c.state.x[0] = mem.name(0, "nvnDeviceGetProcAddress");
    session.bootstrap(&mut c, 0x4000, forbidden_guest);
    let resolver = c.state.x[0];
    assert_eq!(c.ctl.stop, stop::NONE);
    assert_eq!(c.state.pc, RETURN);
    let builder = mem.0 + 0x200;
    rt().mem
        .write_bytes(builder - 1, &[0xAA; 26], true)
        .unwrap();
    for (name, flags) in [
        ("nvnDeviceBuilderSetDefaults", 0),
        ("nvnDeviceBuilderSetFlags", 0x1_1234_5678),
    ] {
        c.state.pc = resolver;
        c.state.x[0] = 0; // device-independent GetProcAddress
        c.state.x[1] = mem.name(0x100, name);
        session.call(&mut c);
        c.state.pc = c.state.x[0];
        c.state.x[0] = builder;
        c.state.x[1] = flags;
        session.call(&mut c);
        assert_eq!(c.ctl.stop, stop::NONE);
        assert_eq!(c.state.pc, RETURN);
        assert_eq!(c.state.x[0], builder); // both setters return void
    }
    let mut bytes = [0; 26];
    rt().mem.read_bytes(builder - 1, &mut bytes, true).unwrap();
    assert_eq!(bytes[0], 0xAA);
    assert_eq!(&bytes[1..5], &[0; 4]);
    assert_eq!(&bytes[5..9], &0x1234_5678u32.to_le_bytes());
    assert_eq!(&bytes[9..25], &[0; 16]);
    assert_eq!(bytes[25], 0xAA);
}

#[test]
fn missing_hle_and_invalid_tokens_stop_at_the_guest_caller() {
    let session = Session::new(Mode::Hle);
    let mut c = cpu();
    c.state.pc = session.bind("nvnDeviceInitialize", 0).unwrap();
    c.state.x[0] = 0x1234;
    session.call(&mut c);
    assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
    assert_eq!(c.ctl.stop_pc, RETURN - 4);
    assert_eq!(c.state.pc, STOP_PC);
    assert_eq!(c.state.x[0], 0x1234); // no fabricated result
    c = cpu();
    c.state.pc = PROC_END - 4;
    session.call(&mut c);
    assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
    for pc in [
        PROC_BASE - 4,
        PROC_BASE + 1,
        PROC_END,
        super::super::abi::EXIT_PC,
    ] {
        assert!(lookup(pc).is_none(), "{pc:#x}");
    }
    assert!(lookup(PROC_BASE).is_some());
}

#[test]
fn hle_reports_bad_name_and_builder_memory_as_guest_faults() {
    let mem = Memory::new();
    let session = Session::new(Mode::Hle);
    let mut c = cpu();
    c.state.x[0] = mem.0 + 0x1000;
    session.bootstrap(&mut c, 0x4000, forbidden_guest);
    assert_eq!(c.ctl.stop, stop::FAULT);
    assert_eq!(c.ctl.stop_pc, 0x4000);
    assert_eq!(c.ctl.fault_addr, mem.0 + 0x1000);
    c = cpu();
    c.state.pc = session.bind("nvnDeviceBuilderSetFlags", 0).unwrap();
    c.state.x[0] = mem.0 + 0xFFC; // flags land outside the mapped page
    session.call(&mut c);
    assert_eq!(c.ctl.stop, stop::FAULT);
    assert_eq!(c.ctl.stop_pc, RETURN - 4);
    assert_eq!(c.ctl.fault_addr, mem.0 + 0x1000);
    c = cpu();
    rt().mem
        .write_bytes(mem.0, &[b'A'; MAX_NAME], true)
        .unwrap();
    c.state.x[0] = mem.0;
    session.bootstrap(&mut c, 0x4000, forbidden_guest);
    assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
}
