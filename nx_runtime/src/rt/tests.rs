//! Tests of the runtime through its C ABI, with Rust functions standing in for generated code.
//!
//! The runtime is a process-wide singleton, so every test works in its own address range.

use super::abi::{perm, stop, trap, FuncEntry, NativeCpu, EXIT_PC, STOP_PC};
use super::exports::*;
use crate::testing::asm;
use std::ffi::c_void;

const RW: u32 = perm::R | perm::W;

struct Cpu(*mut NativeCpu);

impl Cpu {
    fn new() -> Cpu {
        Cpu(nx_rt_cpu_new())
    }
    #[allow(clippy::mut_from_ref)]
    fn c(&self) -> &mut NativeCpu {
        // SAFETY: allocated by `nx_rt_cpu_new`, freed on drop, used by one test thread.
        unsafe { &mut *self.0 }
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        // SAFETY: allocated by `nx_rt_cpu_new`, used by this wrapper only.
        unsafe { nx_rt_cpu_free(self.0) };
    }
}

fn map(addr: u64, size: u64) {
    assert_eq!(nx_rt_map(addr, size, RW), 0);
}

fn put_words(addr: u64, words: &[u32]) {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    // SAFETY: `bytes` is a live slice.
    assert_eq!(
        unsafe { nx_rt_write(addr, bytes.as_ptr(), bytes.len() as u64) },
        0
    );
}

#[test]
fn loads_and_stores_through_the_c_abi() {
    map(0x1000_0000, 0x2000);
    let cpu = Cpu::new();
    nx_st64(cpu.0, 0x1000_0010, 0x1122_3344_5566_7788, 0x400);
    nx_st8(cpu.0, 0x1000_0020, 0xAB, 0x404);
    nx_st16(cpu.0, 0x1000_0022, 0xBEEF, 0x408);
    nx_st32(cpu.0, 0x1000_0024, 0xCAFE_F00D, 0x40C);
    assert_eq!(cpu.c().ctl.fault_flag, 0);
    assert_eq!(nx_ld64(cpu.0, 0x1000_0010, 0x410), 0x1122_3344_5566_7788);
    assert_eq!(nx_ld8(cpu.0, 0x1000_0020, 0x414), 0xAB);
    assert_eq!(nx_ld16(cpu.0, 0x1000_0022, 0x418), 0xBEEF);
    assert_eq!(nx_ld32(cpu.0, 0x1000_0024, 0x41C), 0xCAFE_F00D);
    // Unaligned accesses are fine, also across a page boundary.
    nx_st64(cpu.0, 0x1000_0FFC, 0x0102_0304_0506_0708, 0x420);
    assert_eq!(nx_ld64(cpu.0, 0x1000_0FFC, 0x424), 0x0102_0304_0506_0708);
    assert_eq!(cpu.c().ctl.fault_flag, 0);
}

#[test]
fn a_failed_access_records_the_fault_and_unwinds() {
    map(0x1100_0000, 0x1000);
    let cpu = Cpu::new();
    assert_eq!(nx_ld32(cpu.0, 0x1100_1000, 0x500), 0);
    let c = cpu.c();
    assert_eq!(c.ctl.fault_flag, 1);
    assert_eq!(c.ctl.stop, stop::FAULT);
    assert_eq!(
        (c.ctl.fault_addr, c.ctl.stop_pc, c.ctl.fault_write),
        (0x1100_1000, 0x500, 0)
    );
    assert_eq!(c.ctl.fault_kind, 0);
    assert_eq!(c.state.pc, STOP_PC);

    let cpu = Cpu::new();
    assert_eq!(nx_rt_protect(0x1100_0000, 0x1000, perm::R), 0);
    nx_st32(cpu.0, 0x1100_0000, 1, 0x504);
    assert_eq!((cpu.c().ctl.fault_kind, cpu.c().ctl.fault_write), (1, 1));
}

#[test]
fn ordered_accesses_must_be_aligned() {
    map(0x1200_0000, 0x1000);
    let cpu = Cpu::new();
    nx_stlr32(cpu.0, 0x1200_0004, 7, 0x600);
    assert_eq!(nx_ldar32(cpu.0, 0x1200_0004, 0x604), 7);
    assert_eq!(cpu.c().ctl.fault_flag, 0);
    assert_eq!(nx_ldar64(cpu.0, 0x1200_0004, 0x608), 0);
    assert_eq!((cpu.c().ctl.stop, cpu.c().ctl.fault_kind), (stop::FAULT, 2));
    let cpu = Cpu::new();
    nx_stlr16(cpu.0, 0x1200_0001, 1, 0x60C);
    assert_eq!(
        (cpu.c().ctl.stop, cpu.c().ctl.fault_write),
        (stop::FAULT, 1)
    );
}

/// Four pages: read-write, read-only, mapped with no access, read-write; then a hole.
const MIX: u64 = 0x1700_0000;
const MIX_PERMISSIONS: [u32; 4] = [RW, perm::R, 0, RW];

fn mix_pattern() -> Vec<u8> {
    (0..0x4000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect()
}

/// The addresses worth trying around the pages of [`MIX`] and the ends of the address space.
fn mix_addresses() -> Vec<u64> {
    let mut v = Vec::new();
    for page in 0..=5u64 {
        let base = MIX + page * 0x1000;
        for d in -9i64..=9 {
            v.push(base.wrapping_add_signed(d));
        }
        v.push(base + 0x7c0);
    }
    v.extend([
        0,
        7,
        MIX - 8,
        (1 << 39) - 8,
        (1 << 39) - 3,
        1 << 39,
        u64::MAX - 3,
        u64::MAX,
    ]);
    v
}

#[test]
fn the_fast_path_does_what_the_general_one_does() {
    for (page, p) in MIX_PERMISSIONS.iter().enumerate() {
        assert_eq!(nx_rt_map(MIX + page as u64 * 0x1000, 0x1000, *p), 0);
    }
    let pattern = mix_pattern();
    // SAFETY: `pattern` is a live slice.
    assert_eq!(unsafe { nx_rt_write(MIX, pattern.as_ptr(), 0x4000) }, 0);
    // An instance of its own with the same pages stands for "the general path": the C ABI of the
    // process must leave memory as it does, and fail where it fails.
    let oracle = super::mem::GuestMemory::new();
    for (page, p) in MIX_PERMISSIONS.iter().enumerate() {
        oracle
            .map(MIX + page as u64 * 0x1000, 0x1000, *p)
            .expect("map");
    }
    oracle.write_bytes(MIX, &pattern, false).expect("fill");

    type Load = fn(*mut NativeCpu, u64, u64) -> u64;
    type Store = fn(*mut NativeCpu, u64, u64, u64);
    let loads: [(u8, Load); 4] = [
        (1, |c, a, pc| u64::from(nx_ld8(c, a, pc))),
        (2, |c, a, pc| u64::from(nx_ld16(c, a, pc))),
        (4, |c, a, pc| u64::from(nx_ld32(c, a, pc))),
        (8, |c, a, pc| nx_ld64(c, a, pc)),
    ];
    let stores: [(u8, Store); 4] = [
        (1, |c, a, v, pc| nx_st8(c, a, v as u8, pc)),
        (2, |c, a, v, pc| nx_st16(c, a, v as u16, pc)),
        (4, |c, a, v, pc| nx_st32(c, a, v as u32, pc)),
        (8, |c, a, v, pc| nx_st64(c, a, v, pc)),
    ];
    for addr in mix_addresses() {
        for (size, load) in loads {
            let cpu = Cpu::new();
            let got = load(cpu.0, addr, 0x700);
            let c = cpu.c();
            match oracle.read(addr, size, std::sync::atomic::Ordering::Relaxed) {
                Ok(v) => assert_eq!(
                    (got, c.ctl.fault_flag),
                    (v as u64, 0),
                    "load {size} at {addr:#x}"
                ),
                Err(kind) => {
                    assert_eq!(
                        (got, c.ctl.fault_flag, c.ctl.fault_addr),
                        (0, 1, addr),
                        "load {size} at {addr:#x}"
                    );
                    assert_eq!(c.ctl.fault_write, 0);
                    assert_eq!(c.ctl.fault_kind, kind as u32, "load {size} at {addr:#x}");
                }
            }
        }
        for (size, store) in stores {
            let value = 0x8877_6655_4433_2211u64.rotate_left(addr as u32 & 63) | 0x0101;
            let cpu = Cpu::new();
            store(cpu.0, addr, value, 0x704);
            let c = cpu.c();
            let expect = oracle.write(
                addr,
                size,
                u128::from(value),
                std::sync::atomic::Ordering::Relaxed,
            );
            assert_eq!(
                c.ctl.fault_flag,
                u32::from(expect.is_err()),
                "store {size} at {addr:#x}"
            );
            if let Err(kind) = expect {
                assert_eq!((c.ctl.fault_addr, c.ctl.fault_write), (addr, 1));
                assert_eq!(c.ctl.fault_kind, kind as u32, "store {size} at {addr:#x}");
            }
            let (mut ours, mut theirs) = (vec![0u8; 0x4000], vec![0u8; 0x4000]);
            // SAFETY: the buffer has the length passed.
            assert_eq!(unsafe { nx_rt_read(MIX, ours.as_mut_ptr(), 0x4000) }, 0);
            oracle.read_bytes(MIX, &mut theirs, false).expect("read");
            assert!(
                ours == theirs,
                "memory differs after store {size} at {addr:#x}"
            );
        }
    }
}

#[test]
fn the_fast_lookup_takes_only_what_it_can_do_in_one_access() {
    use super::mem::{fast_ptr, FAST_READ, FAST_WRITE};
    assert_eq!(nx_rt_map(0x1900_0000, 0x1000, RW), 0);
    assert_eq!(nx_rt_map(0x1900_1000, 0x1000, perm::R), 0);
    let host = |addr| super::rt().mem.host_run(addr, 1).expect("mapped").0;
    for size in [1u64, 2, 4, 8] {
        let at = 0x1900_0000 + 3 * size;
        assert_eq!(fast_ptr(at, size, FAST_READ), Some(host(at)));
        assert_eq!(fast_ptr(at, size, FAST_WRITE), Some(host(at)));
        // The last aligned access of the page is still inside it.
        let last = 0x1900_1000 - size;
        assert_eq!(fast_ptr(last, size, FAST_READ), Some(host(last)));
        // Read-only: reads only. Beyond the mapping and beyond the address space: nothing.
        assert!(fast_ptr(0x1900_1000, size, FAST_READ).is_some());
        assert!(fast_ptr(0x1900_1000, size, FAST_WRITE).is_none());
        assert!(fast_ptr(0x1900_2000, size, FAST_READ).is_none());
        assert!(fast_ptr(0x7_0000_0000, size, FAST_READ).is_none());
        assert!(fast_ptr(1 << 39, size, FAST_READ).is_none());
        assert!(fast_ptr(u64::MAX - size + 1, size, FAST_READ).is_none());
        if size > 1 {
            assert!(fast_ptr(0x1900_0000 + 1, size, FAST_READ).is_none());
            assert!(fast_ptr(0x1900_0000 + size / 2, size, FAST_READ).is_none());
        }
    }
}

#[test]
fn another_threads_store_of_every_size_fails_a_store_exclusive() {
    map(0x1800_0000, 0x1000);
    let a = Cpu::new();
    let b = Cpu::new();
    let addr = 0x1800_0040;
    type Store = fn(*mut NativeCpu, u64);
    let stores: [(&str, Store); 8] = [
        ("st8", |c, a| nx_st8(c, a, 1, 0)),
        ("st16", |c, a| nx_st16(c, a, 1, 0)),
        ("st32", |c, a| nx_st32(c, a, 1, 0)),
        ("st64", |c, a| nx_st64(c, a, 1, 0)),
        ("stlr8", |c, a| nx_stlr8(c, a, 1, 0)),
        ("stlr16", |c, a| nx_stlr16(c, a, 1, 0)),
        ("stlr32", |c, a| nx_stlr32(c, a, 1, 0)),
        ("stlr64", |c, a| nx_stlr64(c, a, 1, 0)),
    ];
    for (name, store) in stores {
        nx_st64(a.0, addr, 0x55, 0);
        nx_ldx(a.0, addr, 0x10, 8, 0);
        store(b.0, addr);
        assert_eq!(
            nx_stx(a.0, addr, 5, 0, 0x14, 8, 0),
            1,
            "{name} did not end the reservation"
        );
    }
    // Without a reservation anywhere the same stores work as plain stores.
    nx_stlr64(b.0, addr, 77, 0);
    assert_eq!(nx_ld64(a.0, addr, 0), 77);
    // A store by another thread that leaves the bytes as the load saw them is not noticed: the
    // store-exclusive is a compare-and-exchange against those bytes (A-021).
    nx_ldx(a.0, addr, 0x10, 8, 0);
    nx_st64(b.0, addr, 77, 0);
    assert_eq!(nx_stx(a.0, addr, 78, 0, 0x14, 8, 0), 0);
    assert_eq!(nx_ld64(a.0, addr, 0), 78);
}

#[test]
fn store_exclusive_needs_its_own_reservation() {
    map(0x1300_0000, 0x1000);
    let a = Cpu::new();
    let b = Cpu::new();
    let addr = 0x1300_0040;
    nx_st64(a.0, addr, 10, 0);
    // The plain case: load, store succeeds, a second store fails.
    assert_eq!(nx_ldx(a.0, addr, 0x10, 8, 0), 10);
    assert_eq!(nx_stx(a.0, addr, 11, 0, 0x14, 8, 0), 0);
    assert_eq!(nx_ld64(a.0, addr, 0), 11);
    assert_eq!(nx_stx(a.0, addr, 12, 0, 0x18, 8, 0), 1);
    assert_eq!(nx_ld64(a.0, addr, 0), 11);
    // A thread's own plain store into the range ends it too: the bytes are no longer those the
    // load saw (the architecture leaves this one implementation defined, A-021).
    nx_ldx(a.0, addr, 0x10, 8, 1);
    nx_st64(a.0, addr, 20, 0x14);
    assert_eq!(nx_stx(a.0, addr, 21, 0, 0x18, 8, 1), 1);
    assert_eq!(nx_ld64(a.0, addr, 0), 20);
    // Another thread's store does: the bytes are no longer those the load saw.
    nx_ldx(a.0, addr, 0x10, 8, 0);
    nx_st64(b.0, addr, 22, 0x20);
    assert_eq!(nx_stx(a.0, addr, 99, 0, 0x14, 8, 0), 1);
    assert_eq!(nx_ld64(a.0, addr, 0), 22);
    // CLREX, a different address and a different size all fail the store.
    nx_ldx(a.0, addr, 0x10, 8, 0);
    nx_clrex(a.0);
    assert_eq!(nx_stx(a.0, addr, 5, 0, 0x14, 8, 0), 1);
    nx_ldx(a.0, addr, 0x10, 8, 0);
    assert_eq!(nx_stx(a.0, addr + 8, 5, 0, 0x14, 8, 0), 1);
    nx_ldx(a.0, addr, 0x10, 8, 0);
    assert_eq!(nx_stx(a.0, addr, 5, 0, 0x14, 4, 0), 1);
    assert_eq!(nx_ld64(a.0, addr, 0), 22);
    // A failed store-exclusive to unmapped memory does not fault: it never touches memory.
    assert_eq!(nx_stx(a.0, 0x7000_0000, 5, 0, 0x14, 8, 0), 1);
    assert_eq!(a.c().ctl.stop, stop::NONE);
}

#[test]
fn the_store_exclusive_can_fail_spuriously() {
    map(0x1400_0000, 0x1000);
    let cpu = Cpu::new();
    let addr = 0x1400_0000;
    nx_rt_set_spurious_failure_period(cpu.0, 3);
    let mut results = Vec::new();
    for i in 0..6u64 {
        nx_ldx(cpu.0, addr, 0x10, 8, 0);
        results.push(nx_stx(cpu.0, addr, i, 0, 0x14, 8, 0));
    }
    nx_rt_set_spurious_failure_period(cpu.0, 0);
    // Every third attempt fails with the reservation intact; the others succeed.
    assert_eq!(results, vec![0, 0, 1, 0, 0, 1]);
    nx_ldx(cpu.0, addr, 0x10, 8, 0);
    assert_eq!(nx_stx(cpu.0, addr, 77, 0, 0x14, 8, 0), 0);
}

#[test]
fn exclusive_pairs_move_both_halves() {
    map(0x1500_0000, 0x1000);
    let cpu = Cpu::new();
    let addr = 0x1500_0010;
    nx_st64(cpu.0, addr, 0x1111, 0);
    nx_st64(cpu.0, addr + 8, 0x2222, 0);
    assert_eq!(nx_ldx(cpu.0, addr, 0x10, 16, 0), 0x1111);
    assert_eq!(cpu.c().ctl.aux, 0x2222);
    assert_eq!(nx_stx(cpu.0, addr, 0xAAAA, 0xBBBB, 0x14, 16, 0), 0);
    assert_eq!(nx_ld64(cpu.0, addr, 0), 0xAAAA);
    assert_eq!(nx_ld64(cpu.0, addr + 8, 0), 0xBBBB);
    // A store by another thread into either half fails the pair's store: the exchange is of
    // all 16 bytes.
    let other = Cpu::new();
    nx_ldx(cpu.0, addr, 0x10, 16, 0);
    nx_st64(other.0, addr + 8, 0xCCCC, 0);
    assert_eq!(nx_stx(cpu.0, addr, 1, 2, 0x14, 16, 0), 1);
    nx_ldx(cpu.0, addr, 0x10, 16, 0);
    nx_st64(other.0, addr, 0xDDDD, 0);
    assert_eq!(nx_stx(cpu.0, addr, 1, 2, 0x14, 16, 0), 1);
    assert_eq!(
        (nx_ld64(cpu.0, addr, 0), nx_ld64(cpu.0, addr + 8, 0)),
        (0xDDDD, 0xCCCC)
    );
    nx_ldx(cpu.0, addr, 0x10, 16, 0);
    assert_eq!(nx_stx(cpu.0, addr, 1, 2, 0x14, 16, 0), 0);
    assert_eq!(
        (nx_ld64(cpu.0, addr, 0), nx_ld64(cpu.0, addr + 8, 0)),
        (1, 2)
    );
    // A misaligned pair faults.
    nx_ldx(cpu.0, addr + 8, 0x18, 16, 0);
    assert_eq!(cpu.c().ctl.stop, stop::FAULT);
}

#[test]
fn zva_zeroes_an_aligned_block() {
    map(0x1600_0000, 0x1000);
    let cpu = Cpu::new();
    for i in 0..16u64 {
        nx_st64(cpu.0, 0x1600_0000 + 8 * i, u64::MAX, 0);
    }
    nx_dczva(cpu.0, 0x1600_0050, 0x20);
    assert_eq!(nx_ld64(cpu.0, 0x1600_003F - 7, 0), u64::MAX); // below the block
    assert_eq!(nx_ld64(cpu.0, 0x1600_0040, 0), 0);
    assert_eq!(nx_ld64(cpu.0, 0x1600_0078, 0), 0);
    assert_eq!(nx_ld64(cpu.0, 0x1600_0080 - 8, 0), 0);
}

#[test]
fn crc32_matches_the_reference_values() {
    // CRC-32 of "123456789" with the usual init/final inversion is 0xCBF43926 (zlib polynomial)
    // and 0xE3069283 for the Castagnoli polynomial.
    for (c, expect) in [(0u32, 0xCBF4_3926u32), (1, 0xE306_9283)] {
        let mut crc = !0u32;
        for b in b"123456789" {
            crc = nx_crc32(crc, u64::from(*b), 0, c);
        }
        assert_eq!(!crc, expect, "castagnoli={c}");
    }
}

// ---- dispatch, the slow path and the run loop, with Rust standing in for generated code ----

extern "C" fn returns_seven(p: *mut NativeCpu) {
    // SAFETY: the runtime passes its thread's CPU.
    let c = unsafe { &mut *p };
    c.state.x[0] = 7;
    c.state.pc = c.state.x[30];
}

extern "C" fn adds_one_then_returns(p: *mut NativeCpu) {
    // SAFETY: as above.
    let c = unsafe { &mut *p };
    c.state.x[0] += 1;
    c.state.pc = c.state.x[30];
}

fn register(addr: u64, f: extern "C" fn(*mut NativeCpu)) {
    let entry = FuncEntry {
        addr,
        func: f as *const c_void,
    };
    // SAFETY: one valid entry for a function that lives forever.
    unsafe { nx_rt_register(&entry, 1) };
}

#[test]
fn the_run_loop_calls_registered_functions_and_exits_on_return() {
    register(0x2000_0000, returns_seven);
    assert_eq!(nx_lookup(0x2000_0000), returns_seven as *const c_void);
    assert!(nx_lookup(0x2000_0004).is_null());
    let cpu = Cpu::new();
    // SAFETY: the CPU is ours alone.
    assert_eq!(unsafe { nx_rt_run(cpu.0, 0x2000_0000) }, 0);
    assert_eq!((cpu.c().state.x[0], cpu.c().state.pc), (7, EXIT_PC));
}

#[test]
fn unknown_targets_run_in_the_interpreter_and_are_logged() {
    // Guest code with no generated function: x0 += 5 ; ret
    map(0x2100_0000, 0x1000);
    put_words(0x2100_0000, &[asm::add_imm(true, 0, 0, 5), asm::ret(30)]);
    let cpu = Cpu::new();
    cpu.c().state.x[0] = 100;
    // SAFETY: the CPU is ours alone.
    assert_eq!(unsafe { nx_rt_run(cpu.0, 0x2100_0000) }, 0);
    assert_eq!(cpu.c().state.x[0], 105);
    let n = nx_rt_unknown_count();
    assert!((0..n).any(|i| nx_rt_unknown_at(i) == 0x2100_0000));
}

#[test]
fn interpreted_code_calls_back_into_native_code() {
    // Interpreted: mov x19, x30 ; bl native ; add x0, x0, #10 ; mov x30, x19 ; ret
    map(0x2200_0000, 0x1000);
    register(0x2200_0100, adds_one_then_returns);
    put_words(
        0x2200_0000,
        &[
            asm::logical_shifted(true, 1, 0, false, 30, 0, 31, 19),
            asm::call(0x100 - 4),
            asm::add_imm(true, 0, 0, 10),
            asm::logical_shifted(true, 1, 0, false, 19, 0, 31, 30),
            asm::ret(30),
        ],
    );
    let cpu = Cpu::new();
    // SAFETY: the CPU is ours alone.
    assert_eq!(unsafe { nx_rt_run(cpu.0, 0x2200_0000) }, 0);
    assert_eq!(cpu.c().state.x[0], 11);
}

#[test]
fn stops_are_recorded_with_their_details() {
    map(0x2300_0000, 0x1000);
    // add x0, x0, #1 ; svc #9 ; ret
    put_words(
        0x2300_0000,
        &[asm::add_imm(true, 0, 0, 1), asm::svc(9), asm::ret(30)],
    );
    let cpu = Cpu::new();
    // SAFETY: the CPU is ours alone.
    assert_eq!(unsafe { nx_rt_run(cpu.0, 0x2300_0000) }, stop::SVC);
    let c = cpu.c();
    assert_eq!(
        (c.ctl.stop_pc, c.ctl.trap_aux[0], c.state.x[0]),
        (0x2300_0004, 9, 1)
    );
    assert_eq!(c.state.pc, STOP_PC);

    // brk, an undefined word, an unsupported word (an Armv8.1 atomic), an EL1-only register.
    for (word, kind, aux) in [
        (asm::brk(0x123), trap::BRK, 0x123u64),
        (0x0000_0000, trap::UNDEFINED, 0),
        (0xC8A0_7C41, trap::UNSUPPORTED, 0xC8A0_7C41),
        (asm::sysreg(true, 3, 0, 0, 0, 0, 3), trap::SYSREG, 3 << 14),
    ] {
        put_words(0x2300_0100, &[word]);
        let cpu = Cpu::new();
        // SAFETY: the CPU is ours alone.
        assert_eq!(
            unsafe { nx_rt_run(cpu.0, 0x2300_0100) },
            stop::TRAP,
            "{word:#x}"
        );
        let c = cpu.c();
        assert_eq!(
            (c.ctl.trap_kind, c.ctl.stop_pc),
            (kind, 0x2300_0100),
            "{word:#x}"
        );
        assert_eq!(c.ctl.trap_aux[0], aux, "{word:#x}");
    }

    // Running off the mapped code stops with "no code", not a made-up success.
    let cpu = Cpu::new();
    // SAFETY: the CPU is ours alone.
    assert_eq!(unsafe { nx_rt_run(cpu.0, 0x2F00_0000) }, stop::NO_CODE);
    assert_eq!(cpu.c().ctl.stop_pc, 0x2F00_0000);
}

extern "C" fn adds_a_hundred(p: *mut NativeCpu, imm: u32) -> i32 {
    // SAFETY: as above.
    let c = unsafe { &mut *p };
    if imm == 1 {
        c.state.x[0] += 100;
        0
    } else {
        1
    }
}

#[test]
fn a_registered_handler_services_supervisor_calls() {
    map(0x2400_0000, 0x1000);
    put_words(
        0x2400_0000,
        &[
            asm::svc(1),
            asm::add_imm(true, 0, 0, 1),
            asm::svc(2),
            asm::ret(30),
        ],
    );
    nx_rt_set_svc_handler(Some(adds_a_hundred));
    let cpu = Cpu::new();
    // SAFETY: the CPU is ours alone.
    let r = unsafe { nx_rt_run(cpu.0, 0x2400_0000) };
    nx_rt_set_svc_handler(None);
    assert_eq!(r, stop::SVC);
    assert_eq!(cpu.c().state.x[0], 101);
    assert_eq!(cpu.c().ctl.stop_pc, 0x2400_0008);
}
