//! The C-ABI functions generated code calls (`nx_*`) and the entry points a host program uses
//! (`nx_rt_*`). Names and signatures are the contract with the code generator
//! (`kiln::codegen`); the declarations it emits are checked against this list by a test.
//!
//! Memory functions return the loaded value (zero after a failure) or nothing, and report a
//! failure by recording it in the thread's [`Ctl`](super::abi::Ctl) and setting `fault_flag`,
//! which generated code tests after every access.

use super::abi::{perm, stop, trap, FuncEntry, NativeCpu, STOP_PC};
use super::census::{self, Kind};
use super::mem::{fast_ptr, MapError, FAST_READ, FAST_WRITE};
use super::monitor::nx_mem_armed;
use super::runtime::{
    handle_trap, interpret, interpret_one, record_fault, record_stop, record_trap, rt, run,
    service_svc, slow_path, NativeBus, SvcHandler,
};
use crate::a64::bits::crc32_update;
use crate::a64::bus::{FaultKind, ZVA_BLOCK};
use crate::a64::vec;
use std::ffi::c_void;
use std::sync::atomic::Ordering::{self, Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8};

fn cpu<'a>(p: *mut NativeCpu) -> &'a mut NativeCpu {
    // SAFETY: generated code and hosts pass the pointer they were given for this thread.
    unsafe { &mut *p }
}

// The general case of an access: any size and alignment, the exclusive monitor, faults. The plain
// accesses below try the page table directly first (`fast_ptr`) and come here for what it declines.
#[inline(never)]
fn do_load(p: *mut NativeCpu, addr: u64, pc: u64, size: u8, order: Ordering) -> u64 {
    let c = cpu(p);
    match rt().mem.read(addr, size, order) {
        Ok(v) => v as u64,
        Err(k) => {
            record_fault(c, addr, pc, false, k);
            0
        }
    }
}

#[inline(never)]
fn do_store(p: *mut NativeCpu, addr: u64, pc: u64, size: u8, value: u64, order: Ordering) {
    let c = cpu(p);
    if let Err(k) = rt().store(c.ctl.thread_id, addr, size, u128::from(value), order) {
        record_fault(c, addr, pc, true, k);
    }
}

/// Load-acquire and store-release must be aligned.
fn misaligned(p: *mut NativeCpu, addr: u64, pc: u64, size: u8, write: bool) -> bool {
    if !addr.is_multiple_of(u64::from(size)) {
        record_fault(cpu(p), addr, pc, write, FaultKind::Alignment);
        true
    } else {
        false
    }
}

macro_rules! plain_access {
    ($ld:ident, $st:ident, $ldar:ident, $stlr:ident, $ty:ty, $atomic:ty, $size:expr) => {
        #[no_mangle]
        pub extern "C" fn $ld(p: *mut NativeCpu, addr: u64, pc: u64) -> $ty {
            if let Some(host) = fast_ptr(addr, $size, FAST_READ) {
                // SAFETY: `fast_ptr` found `$size` readable bytes, aligned for the access.
                return unsafe { <$atomic>::from_ptr(host.cast()).load(Relaxed) };
            }
            do_load(p, addr, pc, $size, Relaxed) as $ty
        }

        #[no_mangle]
        pub extern "C" fn $st(p: *mut NativeCpu, addr: u64, value: $ty, pc: u64) {
            // With a reservation outstanding a store must tell the monitor (`Runtime::store`).
            if nx_mem_armed.load(Relaxed) == 0 {
                if let Some(host) = fast_ptr(addr, $size, FAST_WRITE) {
                    // SAFETY: `fast_ptr` found `$size` writable bytes, aligned for the access.
                    unsafe { <$atomic>::from_ptr(host.cast()).store(value, Relaxed) };
                    return;
                }
            }
            do_store(p, addr, pc, $size, u64::from(value), Relaxed)
        }

        #[no_mangle]
        pub extern "C" fn $ldar(p: *mut NativeCpu, addr: u64, pc: u64) -> $ty {
            if let Some(host) = fast_ptr(addr, $size, FAST_READ) {
                // SAFETY: as above; `fast_ptr` also found the address aligned.
                return unsafe { <$atomic>::from_ptr(host.cast()).load(Acquire) };
            }
            if misaligned(p, addr, pc, $size, false) {
                return 0;
            }
            do_load(p, addr, pc, $size, Acquire) as $ty
        }

        #[no_mangle]
        pub extern "C" fn $stlr(p: *mut NativeCpu, addr: u64, value: $ty, pc: u64) {
            if nx_mem_armed.load(Relaxed) == 0 {
                if let Some(host) = fast_ptr(addr, $size, FAST_WRITE) {
                    // SAFETY: as above.
                    unsafe { <$atomic>::from_ptr(host.cast()).store(value, Release) };
                    return;
                }
            }
            if misaligned(p, addr, pc, $size, true) {
                return;
            }
            do_store(p, addr, pc, $size, u64::from(value), Release)
        }
    };
}

plain_access!(nx_ld8, nx_st8, nx_ldar8, nx_stlr8, u8, AtomicU8, 1);
plain_access!(nx_ld16, nx_st16, nx_ldar16, nx_stlr16, u16, AtomicU16, 2);
plain_access!(nx_ld32, nx_st32, nx_ldar32, nx_stlr32, u32, AtomicU32, 4);
plain_access!(nx_ld64, nx_st64, nx_ldar64, nx_stlr64, u64, AtomicU64, 8);

/// `LDXR` family. `bytes` is 1, 2, 4, 8 or 16; the low 64 bits are returned and, for 16, the high
/// 64 bits are left in `ctl.aux`.
#[no_mangle]
pub extern "C" fn nx_ldx(p: *mut NativeCpu, addr: u64, pc: u64, bytes: u32, acquire: u32) -> u64 {
    let bytes = bytes as u8;
    if misaligned(p, addr, pc, bytes, false) {
        return 0;
    }
    let c = cpu(p);
    match rt().load_exclusive(c.ctl.thread_id, &mut c.state, addr, bytes, acquire != 0) {
        Ok(v) => {
            c.ctl.aux = (v >> 64) as u64;
            v as u64
        }
        Err(k) => {
            record_fault(c, addr, pc, false, k);
            0
        }
    }
}

/// `STXR` family: returns 0 when the store happened, 1 when it did not. `hi` is the second word of
/// a 16-byte store.
#[no_mangle]
pub extern "C" fn nx_stx(
    p: *mut NativeCpu,
    addr: u64,
    lo: u64,
    hi: u64,
    pc: u64,
    bytes: u32,
    release: u32,
) -> u32 {
    let bytes = bytes as u8;
    if misaligned(p, addr, pc, bytes, true) {
        return 1;
    }
    let c = cpu(p);
    let value = u128::from(lo) | (u128::from(hi) << 64);
    match rt().store_exclusive(
        c.ctl.thread_id,
        &mut c.state,
        addr,
        bytes,
        value,
        release != 0,
    ) {
        Ok(done) => u32::from(!done),
        Err(k) => {
            record_fault(c, addr, pc, true, k);
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn nx_clrex(p: *mut NativeCpu) {
    let c = cpu(p);
    rt().clear_exclusive(c.ctl.thread_id, &mut c.state);
}

/// `DC ZVA`: zero the aligned 64-byte block containing `addr`.
#[no_mangle]
pub extern "C" fn nx_dczva(p: *mut NativeCpu, addr: u64, pc: u64) {
    let base = addr & !(ZVA_BLOCK - 1);
    for i in 0..ZVA_BLOCK / 8 {
        do_store(p, base + i * 8, pc, 8, 0, Relaxed);
        if cpu(p).ctl.fault_flag != 0 {
            return;
        }
    }
}

#[cfg(not(feature = "alloc_stats"))]
#[no_mangle]
pub extern "C" fn nx_counter() -> u64 {
    rt().counter()
}

/// The diagnostic build counts the reads and the cycles they take.
#[cfg(feature = "alloc_stats")]
#[no_mangle]
pub extern "C" fn nx_counter() -> u64 {
    COUNTER_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: reading the time stamp counter has no preconditions.
    let began = unsafe { core::arch::x86_64::_rdtsc() };
    let value = rt().counter();
    // SAFETY: as above.
    let took = unsafe { core::arch::x86_64::_rdtsc() } - began;
    COUNTER_CYCLES.fetch_add(took, std::sync::atomic::Ordering::Relaxed);
    value
}

/// The cycles those reads took.
#[cfg(feature = "alloc_stats")]
pub static COUNTER_CYCLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many times the guest read the system counter (for the report of the diagnostic build).
#[cfg(feature = "alloc_stats")]
pub static COUNTER_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[no_mangle]
pub extern "C" fn nx_crc32(acc: u32, data: u64, size: u32, castagnoli: u32) -> u32 {
    crc32_update(acc, data, size as u8, castagnoli != 0)
}

/// A floating-point or SIMD computation (see [`crate::a64::vec`]): `word` is the instruction,
/// `src` the integer source register's value and `nzcv` the flags in bits 3 to 0, for the
/// encodings that take them. Returns the integer result, or the new flags. The vector registers,
/// `FPCR` and `FPSR` are read and written in the CPU structure.
#[no_mangle]
pub extern "C" fn nx_vec(p: *mut NativeCpu, word: u32, src: u64, nzcv: u64) -> u64 {
    if census::enabled() {
        census::note(Kind::Vec, u64::from(word));
    }
    let c = cpu(p);
    match vec::decode::decode(word) {
        Ok(op) => vec::exec::exec(&mut c.state, &op, src, nzcv as u8),
        Err(_) => {
            // The code generator emits only encodings the decoder accepts, so this means the
            // generated code and the runtime come from different versions.
            let pc = c.state.pc;
            record_trap(c, trap::UNSUPPORTED, pc, u64::from(word), 0);
            0
        }
    }
}

/// A vector register load or store at `addr`; a failed access is recorded like any other.
#[no_mangle]
pub extern "C" fn nx_vecmem(p: *mut NativeCpu, word: u32, addr: u64, pc: u64) {
    if census::enabled() {
        census::note(Kind::VecMem, u64::from(word));
    }
    let c = cpu(p);
    let Ok((mem, _)) = vec::decode::decode_mem(word) else {
        record_trap(c, trap::UNSUPPORTED, pc, u64::from(word), 0);
        return;
    };
    let mut bus = NativeBus {
        thread: c.ctl.thread_id,
    };
    if let Err(f) = vec::exec::exec_mem(&mut c.state, &mut bus, &mem, addr) {
        record_fault(c, f.addr, pc, f.write, f.kind);
    }
}

/// Stop the run at an instruction that always stops it. `kind` is a [`trap`] constant.
#[no_mangle]
pub extern "C" fn nx_trap(p: *mut NativeCpu, kind: u32, pc: u64, aux0: u64, aux1: u64) {
    let c = cpu(p);
    if kind == trap::OUT_OF_CODE {
        record_stop(c, stop::NO_CODE, pc, 0);
    } else {
        record_trap(c, kind, pc, aux0, aux1);
    }
}

/// `SVC #imm`; `cpu.pc` is the instruction after it.
#[no_mangle]
pub extern "C" fn nx_svc(p: *mut NativeCpu, imm: u32) {
    service_svc(cpu(p), imm);
}

/// Run the one instruction at `cpu.pc` in the interpreter.
///
/// # Safety
/// `p` must be the thread's CPU.
#[no_mangle]
pub unsafe extern "C" fn nx_interp1(p: *mut NativeCpu) {
    interpret_one(p);
}

/// The generated function for guest address `pc`, or null.
#[no_mangle]
pub extern "C" fn nx_lookup(pc: u64) -> *const c_void {
    rt().lookup(pc).unwrap_or(std::ptr::null())
}

/// Control went to an address with no generated function: interpret from `cpu.pc`.
///
/// # Safety
/// `p` must be the thread's CPU.
#[no_mangle]
pub unsafe extern "C" fn nx_slow_path(p: *mut NativeCpu) {
    slow_path(p);
}

// ---------------------------------------------------------------------------------------------
// Entry points for the host program
// ---------------------------------------------------------------------------------------------

/// A new guest thread context. Free it with [`nx_rt_cpu_free`].
#[no_mangle]
pub extern "C" fn nx_rt_cpu_new() -> *mut NativeCpu {
    Box::into_raw(Box::new(NativeCpu::new(rt().new_thread_id())))
}

/// # Safety
/// `p` must come from [`nx_rt_cpu_new`] and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_cpu_free(p: *mut NativeCpu) {
    if !p.is_null() {
        let b = Box::from_raw(p);
        let mut st = b.state.clone();
        rt().clear_exclusive(b.ctl.thread_id, &mut st);
    }
}

fn map_status(r: Result<(), MapError>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(MapError::Range) => 1,
        Err(MapError::Overlap) => 2,
        Err(MapError::OutOfMemory) => 3,
    }
}

/// Map zeroed pages. `permissions` is a mask of [`perm`] bits. Returns 0, or 1 for a bad range,
/// 2 for an overlap, 3 when the host is out of memory.
#[no_mangle]
pub extern "C" fn nx_rt_map(addr: u64, size: u64, permissions: u32) -> i32 {
    map_status(
        rt().mem
            .map(addr, size, permissions & (perm::R | perm::W | perm::X)),
    )
}

#[no_mangle]
pub extern "C" fn nx_rt_unmap(addr: u64, size: u64) -> i32 {
    map_status(rt().mem.unmap(addr, size))
}

#[no_mangle]
pub extern "C" fn nx_rt_protect(addr: u64, size: u64, permissions: u32) -> i32 {
    map_status(
        rt().mem
            .protect(addr, size, permissions & (perm::R | perm::W | perm::X)),
    )
}

/// Copy bytes into guest memory, ignoring page permissions (loader and tests). Returns 0 or 1.
///
/// # Safety
/// `data` must point at `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_write(addr: u64, data: *const u8, len: u64) -> i32 {
    let bytes = if len == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(data, len as usize)
    };
    i32::from(rt().mem.write_bytes(addr, bytes, false).is_err())
}

/// Copy bytes out of guest memory, ignoring page permissions. Returns 0 or 1.
///
/// # Safety
/// `out` must point at `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_read(addr: u64, out: *mut u8, len: u64) -> i32 {
    let bytes = if len == 0 {
        &mut [][..]
    } else {
        std::slice::from_raw_parts_mut(out, len as usize)
    };
    i32::from(rt().mem.read_bytes(addr, bytes, false).is_err())
}

/// Register generated functions: `count` entries of [`FuncEntry`].
///
/// # Safety
/// `table` must point at `count` valid entries whose functions outlive the process.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_register(table: *const FuncEntry, count: u64) {
    if count != 0 {
        rt().register(std::slice::from_raw_parts(table, count as usize));
    }
}

#[no_mangle]
pub extern "C" fn nx_rt_set_svc_handler(h: Option<SvcHandler>) {
    rt().set_svc_handler(h);
}

/// Make every `period`th store-exclusive of this thread fail even when its reservation is intact
/// (0 = never).
#[no_mangle]
pub extern "C" fn nx_rt_set_spurious_failure_period(p: *mut NativeCpu, period: u64) {
    rt().set_spurious_period(cpu(p).ctl.thread_id, period);
}

/// Run a thread from `entry` until the entry function returns (0) or a stop (a [`stop`] reason).
///
/// # Safety
/// `p` must be a CPU from [`nx_rt_cpu_new`] used by one thread at a time.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_run(p: *mut NativeCpu, entry: u64) -> u32 {
    run(p, entry)
}

/// Number of addresses that went through the slow path.
#[no_mangle]
pub extern "C" fn nx_rt_unknown_count() -> u64 {
    rt().unknown_targets().len() as u64
}

/// The `i`th such address in increasing order, or 0.
#[no_mangle]
pub extern "C" fn nx_rt_unknown_at(i: u64) -> u64 {
    rt().unknown_targets().get(i as usize).copied().unwrap_or(0)
}

/// Keep the linker from dropping the interpreter fallback entry points that only generated code
/// references, and give tests one place to see the full list.
pub const GENERATED_CODE_SYMBOLS: &[&str] = &[
    "nx_ld8",
    "nx_ld16",
    "nx_ld32",
    "nx_ld64",
    "nx_st8",
    "nx_st16",
    "nx_st32",
    "nx_st64",
    "nx_ldar8",
    "nx_ldar16",
    "nx_ldar32",
    "nx_ldar64",
    "nx_stlr8",
    "nx_stlr16",
    "nx_stlr32",
    "nx_stlr64",
    "nx_ldx",
    "nx_stx",
    "nx_clrex",
    "nx_dczva",
    "nx_counter",
    "nx_crc32",
    "nx_vec",
    "nx_vecmem",
    "nx_trap",
    "nx_svc",
    "nx_interp1",
    "nx_lookup",
    "nx_slow_path",
];

#[doc(hidden)]
pub fn __trap_handler_linked() -> bool {
    // Referenced so the helpers used only through the C ABI stay in the rlib's public surface.
    let _ = handle_trap as fn(&mut NativeCpu, crate::a64::interp::Trap) -> bool;
    STOP_PC != 0
}

/// Run a thread in the interpreter only (see [`interpret`]); for tests that compare it with the
/// generated code.
///
/// # Safety
/// `p` must be a CPU from [`nx_rt_cpu_new`] used by one thread at a time.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_interpret(p: *mut NativeCpu, entry: u64, max_steps: u64) -> u32 {
    interpret(p, entry, max_steps)
}

/// Make the system counter deterministic, starting at `start` and counting one per read; 0
/// restores the real counter.
#[no_mangle]
pub extern "C" fn nx_rt_set_counter(start: u64) {
    rt().set_counter_fixed(start);
}
