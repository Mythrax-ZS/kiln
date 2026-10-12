//! Randomized differential tests: the interpreter against Unicorn's AArch64 emulation (QEMU TCG,
//! Cortex-A57 model). Where they disagree the Arm pseudocode decides; see
//! docs/UNICORN_DISAGREEMENTS.md for every accepted difference.

mod common;

use common::gen::Gen;
use common::harness::{Case, Class, Harness, Outcome, CODE_BASE, CODE_SIZE, DATA_BASE, DATA_SIZE};
use common::{case_count, run_batch, unicorn_or_skip, Accepted};
use nx_runtime::a64::{decode, Inst};
use nx_runtime::testing::asm;

fn never(_: &Case, _: &Outcome, _: &Outcome) -> Option<Accepted> {
    None
}

fn reg_value(s: &nx_runtime::a64::CpuState, n: u8, sp: bool) -> u64 {
    match (n, sp) {
        (31, true) => s.sp,
        (31, false) => 0,
        _ => s.x[usize::from(n)],
    }
}

/// Differences between Unicorn and the architecture that are understood and accepted for
/// exclusive and ordered accesses. The pseudocode wins in every case; each id is explained in
/// docs/UNICORN_DISAGREEMENTS.md.
fn known_exclusive_differences(case: &Case, uc: &Outcome, mine: &Outcome) -> Option<Accepted> {
    let words: Vec<Inst> = case.words[..case.steps]
        .iter()
        .map(|w| decode(*w))
        .collect();
    let last = *words.last()?;
    if mine.class == Class::Alignment {
        return match (&uc.class, last) {
            // H-2: a misaligned exclusive access is reported as a generic exception.
            (Class::Exception, _) => Some(Accepted { id: "H-2" }),
            // U-3: a store-exclusive whose monitor check fails never reaches the alignment check
            // in QEMU; the pseudocode checks alignment first (the architecture leaves the order
            // IMPLEMENTATION DEFINED).
            (
                Class::Ok,
                Inst::Exclusive {
                    load: false, rs, ..
                },
            ) if rs == 31 || uc.state.x[usize::from(rs)] == 1 => Some(Accepted { id: "U-3" }),
            // U-4: QEMU 5 (Unicorn 2.1) does not fault on a misaligned LDAR/STLR; the pseudocode
            // requires alignment for ordered accesses.
            (Class::Ok, Inst::AcqRel { .. }) => Some(Accepted { id: "U-4" }),
            _ => None,
        };
    }
    // U-7: QEMU keeps -1 in its exclusive-address register to mean "monitor open", so a
    // store-exclusive to address 0xFFFF_FFFF_FFFF_FFFF looks armed and faults on the unmapped
    // address. In the pseudocode an unarmed monitor fails the store without any memory access.
    if uc.class == Class::Unmapped && mine.class == Class::Ok && words.len() == 1 {
        if let Inst::Exclusive {
            load: false, rn, ..
        } = last
        {
            if reg_value(&case.state, rn, true) == u64::MAX {
                return Some(Accepted { id: "U-7" });
            }
        }
    }
    // U-6: QEMU models LDXR/STXR as a compare-and-swap of the value LDXR observed, so an
    // intervening plain store that changes the monitored location makes the store-exclusive fail.
    // In the pseudocode the PE's own stores do not touch its monitor, so the store succeeds.
    if uc.class == Class::Ok && mine.class == Class::Ok {
        let (
            Inst::Exclusive {
                load: true,
                rn: lrn,
                size: lsize,
                pair: lpair,
                ..
            },
            Inst::Exclusive {
                load: false, rs, ..
            },
        ) = (*words.first()?, last)
        else {
            return None;
        };
        let plain = words.iter().find_map(|i| match i {
            Inst::LdSt {
                kind: nx_runtime::a64::inst::MemKind::Store,
                size,
                rn,
                ..
            } => Some((*size, *rn)),
            _ => None,
        })?;
        let excl_addr = reg_value(&case.state, lrn, true);
        let excl_len = (1u64 << lsize) << u64::from(lpair);
        let plain_addr = reg_value(&case.state, plain.1, true);
        let plain_len = 1u64 << plain.0;
        let overlap = plain_addr < excl_addr + excl_len && excl_addr < plain_addr + plain_len;
        let status = |o: &Outcome| reg_value(&o.state, rs, false);
        if overlap && status(uc) == 1 && status(mine) == 0 {
            return Some(Accepted { id: "U-6" });
        }
    }
    None
}

#[test]
fn harness_smoke() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    // adds x0, x1, x2 sets the flags
    let word = asm::add_sub_shifted(true, false, true, 0, 2, 0, 1, 0);
    let mut g = Gen::new(1);
    let mut case = g.case(vec![word]);
    case.state.pc = h.slot();
    case.state.x[1] = u64::MAX;
    case.state.x[2] = 1;
    let uc = h.run_unicorn(&case);
    let mine = h.run_mine(&case);
    assert_eq!(uc.state.x[0], 0);
    assert_eq!(uc.state.x[0], mine.state.x[0]);
    assert_eq!(uc.state.nzcv, mine.state.nzcv);
    assert_eq!(mine.state.nzcv4(), 0b0110);
    // an undefined word and an SVC are exceptions in both
    for w in [0x0000_0000, asm::svc(1), asm::brk(1)] {
        let mut c = g.case(vec![w]);
        c.state.pc = h.slot();
        assert_eq!(h.run_unicorn(&c).class, Class::Exception);
        assert_eq!(h.run_mine(&c).class, Class::Exception);
    }
}

// ---------------------------------------------------------------------------------------------
// Data processing
// ---------------------------------------------------------------------------------------------

fn gen_dp_imm(g: &mut Gen) -> Case {
    let sf = g.flag();
    let (rd, rn) = (g.reg(), g.reg());
    let word = match g.below(9) {
        0 => asm::pc_rel(g.flag(), g.below(1 << 21) as i32 - (1 << 20), rd),
        1..=3 => {
            let imm12 = match g.below(4) {
                0 => 0,
                1 => 0xFFF,
                2 => 1,
                _ => g.below(0x1000) as u32,
            };
            asm::add_sub_imm(sf, g.flag(), g.flag(), g.flag(), imm12, rn, rd)
        }
        4 | 5 => {
            let (n, immr, imms) = g.logical_imm_fields(sf);
            asm::logical_imm(sf, g.below(4) as u32, n, immr, imms, rn, rd)
        }
        6 => {
            let opc = [0u32, 2, 3][g.below(3) as usize];
            let hw = if sf {
                g.below(4) as u32
            } else {
                g.below(2) as u32
            };
            asm::mov_wide(sf, opc, hw, g.below(0x10000) as u32, rd)
        }
        7 => {
            let (immr, imms) = g.bitfield_fields(sf);
            asm::bitfield(sf, g.below(3) as u32, sf, immr, imms, rn, rd)
        }
        _ => {
            let limit = if sf { 64 } else { 32 };
            asm::extr(sf, sf, g.reg(), g.below(limit) as u8, rn, rd)
        }
    };
    g.case(vec![word])
}

#[test]
fn data_processing_immediate() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0xD1);
    run_batch(
        "dp-imm",
        &mut h,
        case_count(40_000),
        false,
        |_, _| gen_dp_imm(&mut g),
        never,
    );
}

fn gen_dp_reg(g: &mut Gen) -> Case {
    let sf = g.flag();
    let (rd, rn, rm) = (g.reg(), g.reg(), g.reg());
    let word = match g.below(12) {
        0 | 1 => {
            let amount = g.below(if sf { 64 } else { 32 }) as u8;
            asm::logical_shifted(
                sf,
                g.below(4) as u32,
                g.below(4) as u32,
                g.flag(),
                rm,
                amount,
                rn,
                rd,
            )
        }
        2 | 3 => {
            let amount = g.below(if sf { 64 } else { 32 }) as u8;
            asm::add_sub_shifted(
                sf,
                g.flag(),
                g.flag(),
                g.below(3) as u32,
                rm,
                amount,
                rn,
                rd,
            )
        }
        4 | 5 => asm::add_sub_ext(
            sf,
            g.flag(),
            g.flag(),
            rm,
            g.below(8) as u32,
            g.below(5) as u32,
            rn,
            rd,
        ),
        6 => asm::add_sub_carry(sf, g.flag(), g.flag(), rm, rn, rd),
        7 => asm::cond_cmp(
            sf,
            g.flag(),
            g.flag(),
            g.reg(),
            g.below(16) as u8,
            rn,
            g.below(16) as u8,
        ),
        8 => asm::cond_sel(sf, g.below(4) as u32, rm, g.below(16) as u8, rn, rd),
        9 => {
            // one source: RBIT, REV16, REV (w) / REV32 (x), REV (x), CLZ, CLS
            let opcode = if sf {
                [0u32, 1, 2, 3, 4, 5][g.below(6) as usize]
            } else {
                [0u32, 1, 2, 4, 5][g.below(5) as usize]
            };
            asm::dp1(sf, opcode, rn, rd)
        }
        10 => {
            // two source: divides, variable shifts, CRC32 family
            let opcode = match g.below(8) {
                0 => 0b000010,
                1 => 0b000011,
                2 => 0b001000,
                3 => 0b001001,
                4 => 0b001010,
                5 => 0b001011,
                _ => {
                    let size = if sf { 3 } else { g.below(3) as u32 };
                    0b010000 | (g.below(2) as u32) << 2 | size
                }
            };
            asm::dp2(sf, opcode, rm, rn, rd)
        }
        _ => {
            let (op31, o0) = if sf {
                [
                    (0u32, false),
                    (0, true),
                    (1, false),
                    (1, true),
                    (2, false),
                    (5, false),
                    (5, true),
                    (6, false),
                ][g.below(8) as usize]
            } else {
                (0, g.flag())
            };
            let ra = if matches!(op31, 2 | 6) { 31 } else { g.reg() };
            asm::dp3(sf, op31, o0, rm, ra, rn, rd)
        }
    };
    g.case(vec![word])
}

#[test]
fn data_processing_register() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0xD2);
    run_batch(
        "dp-reg",
        &mut h,
        case_count(80_000),
        false,
        |_, _| gen_dp_reg(&mut g),
        never,
    );
}

// ---------------------------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------------------------

/// A word offset from `pc` that fits `bits` signed bits and keeps the target in the code region.
fn code_words(g: &mut Gen, pc: u64, bits: u32) -> i32 {
    let lim = 1i64 << (bits - 1);
    let lo = (-lim).max((CODE_BASE as i64 - pc as i64) / 4);
    let hi = (lim - 1).min(((CODE_BASE + CODE_SIZE as u64) as i64 - 4 - pc as i64) / 4);
    (lo + g.below((hi - lo + 1) as u64) as i64) as i32
}

fn gen_branch(g: &mut Gen, pc: u64) -> Case {
    let (rn, rt) = (g.reg(), g.reg());
    let wild = g.below(12) == 0;
    let off = |g: &mut Gen, bits: u32| -> i32 {
        if wild {
            (g.below(1 << bits) as i64 - (1 << (bits - 1))) as i32
        } else {
            code_words(g, pc, bits)
        }
    };
    let (word, reg_target) = match g.below(7) {
        0 => (asm::b_imm(g.flag(), off(g, 26)), None),
        1 => (asm::b_cond(g.below(16) as u8, off(g, 19)), None),
        2 => (asm::cbz(g.flag(), g.flag(), off(g, 19), rt), None),
        3 => (asm::tbz(g.flag(), g.below(64) as u8, off(g, 14), rt), None),
        4 => (asm::branch_reg(0, rn), Some(rn)),
        5 => (asm::branch_reg(1, rn), Some(rn)),
        _ => (asm::branch_reg(2, rn), Some(rn)),
    };
    let mut case = g.case(vec![word]);
    if let Some(r) = reg_target {
        if r != 31 && !wild {
            case.state.x[usize::from(r)] = CODE_BASE + 4 * g.below(CODE_SIZE as u64 / 4);
        }
    }
    case
}

/// Unicorn reports a fetch fault for a taken branch to an unmapped target even when only one
/// instruction was requested, while the interpreter stops after the branch (harness artifact H-1).
fn wild_branch_target(case: &Case, uc: &Outcome, mine: &Outcome) -> Option<Accepted> {
    let target_unmapped = !(CODE_BASE..CODE_BASE + CODE_SIZE as u64).contains(&mine.state.pc)
        && !(DATA_BASE..DATA_BASE + DATA_SIZE as u64).contains(&mine.state.pc);
    if uc.class == Class::Unmapped
        && mine.class == Class::Ok
        && decode(case.words[0]).is_branch()
        && target_unmapped
    {
        Some(Accepted { id: "H-1" })
    } else {
        None
    }
}

#[test]
fn branches() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0xB1);
    run_batch(
        "branches",
        &mut h,
        case_count(40_000),
        false,
        |pc, _| gen_branch(&mut g, pc),
        wild_branch_target,
    );
}

// ---------------------------------------------------------------------------------------------
// System instructions
// ---------------------------------------------------------------------------------------------

fn gen_system(g: &mut Gen) -> Case {
    let rt = g.reg();
    let word = match g.below(8) {
        0 => asm::hint(g.below(128) as u8),
        1 => asm::dsb(g.below(16) as u8),
        2 => asm::dmb(g.below(16) as u8),
        3 => asm::isb(g.below(16) as u8),
        4 => asm::clrex(g.below(16) as u8),
        5 => {
            // cache maintenance by address: DC CVAC / CVAU / CIVAC, IC IVAU
            let (crm, op2) = [(10u8, 1u8), (11, 1), (14, 1), (5, 1)][g.below(4) as usize];
            asm::sys(false, 3, 7, crm, op2, rt)
        }
        6 => asm::sysreg(false, 3, 3, 4, 2, 0, rt), // msr nzcv, xt
        _ => asm::sysreg(true, 3, 3, 4, 2, 0, rt),  // mrs xt, nzcv
    };
    g.case(vec![word])
}

#[test]
fn hints_barriers_and_nzcv() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x51);
    run_batch(
        "system",
        &mut h,
        case_count(20_000),
        false,
        |_, _| gen_system(&mut g),
        never,
    );
}

#[test]
fn dc_zva() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x5A);
    run_batch(
        "dc zva",
        &mut h,
        case_count(5_000),
        false,
        |_, _| {
            let rt = g.gpr();
            let mut case = g.case(vec![asm::sys(false, 3, 7, 4, 1, rt)]);
            case.state.x[usize::from(rt)] = g.data_pointer(0x200, 0x200);
            case
        },
        never,
    );
}

#[test]
fn floating_point_control_and_thread_registers() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x5F);
    run_batch(
        "fpcr/fpsr/tpidr",
        &mut h,
        case_count(10_000),
        true,
        |_, _| {
            let rt = g.reg();
            // (read, op1, crn, crm, op2) for FPCR, FPSR, TPIDR_EL0, TPIDRRO_EL0 (read only)
            let (op0, op1, crn, crm, op2, can_write) = [
                (3u8, 3u8, 4u8, 4u8, 0u8, true),
                (3, 3, 4, 4, 1, true),
                (3, 3, 13, 0, 2, true),
                (3, 3, 13, 0, 3, false),
            ][g.below(4) as usize];
            let read = !can_write || g.flag();
            g.case(vec![asm::sysreg(read, op0, op1, crn, crm, op2, rt)])
        },
        never,
    );
}

// ---------------------------------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------------------------------

/// Point the base register of a memory instruction into the middle of the data region.
fn set_base(case: &mut Case, rn: u8, addr: u64) {
    if rn == 31 {
        case.state.sp = addr;
    } else {
        case.state.x[usize::from(rn)] = addr;
    }
}

fn gen_ldst(g: &mut Gen) -> Case {
    let size = g.below(4) as u32;
    let rn = g.reg();
    let mut rt = g.reg();
    let mut rm = g.reg();
    if rm == rn {
        rm = (rm + 1) % 31;
    }
    // opc: 0 store, 1 load, 2 signed load (64-bit dest; size 3 is prefetch), 3 signed load (32-bit dest)
    let mut opc = g.below(4) as u32;
    if size == 3 && opc == 3 {
        opc = 1;
    }
    if size == 2 && opc == 3 {
        opc = 1;
    }
    let word;
    let mut index = None; // (register, value)
    let writeback;
    match g.below(5) {
        0 => {
            let imm12 = if g.below(8) == 0 {
                g.below(4096)
            } else {
                g.below(128)
            } as u32;
            word = asm::ldst_unsigned(size, opc, imm12, rn, rt);
            writeback = false;
        }
        1 => {
            // unscaled, post-index, unprivileged, pre-index
            let mode = g.below(4) as u32;
            word = asm::ldst_imm9(size, opc, g.below(512) as i32 - 256, mode, rn, rt);
            writeback = mode == 1 || mode == 3;
        }
        _ => {
            // register offset with every legal extend
            let option = [2u32, 3, 6, 7][g.below(4) as usize];
            let s = g.flag();
            word = asm::ldst_reg(size, opc, rm, option, s, rn, rt);
            let n = g.below(1025) as i64 - 512;
            let upper = if g.flag() { g.rng.next_u64() << 32 } else { 0 };
            let v = match option {
                // 32-bit extends: the low word carries the signed index, the upper word is noise
                2 | 6 => u64::from(n as i32 as u32) | upper,
                _ => n as u64,
            };
            index = Some((rm, v));
            writeback = false;
        }
    }
    // Writeback with Rt == Rn is UNPREDICTABLE; keep the destination distinct.
    if writeback && rt == rn && rn != 31 {
        rt = (rt + 1) % 31;
        let _ = rt;
    }
    let mut case = g.case(vec![word]);
    set_base(&mut case, rn, g.data_pointer(0x1000, 0x1000));
    if let Some((r, v)) = index {
        if r != 31 {
            case.state.x[usize::from(r)] = v;
        }
    }
    case
}

#[test]
fn single_register_loads_and_stores() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x1D);
    run_batch(
        "ldst",
        &mut h,
        case_count(80_000),
        false,
        |_, _| gen_ldst(&mut g),
        never,
    );
}

fn gen_pair(g: &mut Gen) -> Case {
    let opc = [0u32, 2, 1][g.below(3) as usize];
    let load = if opc == 1 { true } else { g.flag() };
    let kind = if opc == 1 {
        1 + g.below(3) as u32
    } else {
        g.below(4) as u32
    };
    let rn = g.reg();
    let mut rt = g.reg();
    let mut rt2 = g.reg();
    let writeback = kind == 1 || kind == 3;
    if writeback && rn != 31 {
        while rt == rn {
            rt = g.reg();
        }
        while rt2 == rn {
            rt2 = g.reg();
        }
    }
    if load {
        while rt2 == rt && rt != 31 {
            rt2 = g.reg();
        }
    }
    let imm7 = g.below(128) as i32 - 64;
    let mut case = g.case(vec![asm::ldst_pair(opc, kind, load, imm7, rt2, rn, rt)]);
    set_base(&mut case, rn, g.data_pointer(0x1000, 0x1000));
    case
}

#[test]
fn load_store_pairs() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x1E);
    run_batch(
        "ldp/stp",
        &mut h,
        case_count(40_000),
        false,
        |_, _| gen_pair(&mut g),
        never,
    );
}

#[test]
fn literal_loads() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x1F);
    run_batch(
        "ldr literal",
        &mut h,
        case_count(10_000),
        false,
        |_, _| {
            let words_ahead = 1 + g.below(5) as i32;
            let opc = g.below(3) as u32; // ldr w, ldr x, ldrsw
            let mut words = vec![asm::ldr_literal(opc, words_ahead, g.reg())];
            for _ in 0..6 {
                words.push(g.rng.next_u32());
            }
            let mut case = g.case(words);
            case.steps = 1; // the literal words after the load are data
            case
        },
        never,
    );
}

// ---------------------------------------------------------------------------------------------
// Exclusive and ordered accesses
// ---------------------------------------------------------------------------------------------

fn gen_exclusive(g: &mut Gen) -> Case {
    let size = g.below(4) as u32;
    let pair = size >= 2 && g.below(3) == 0;
    let rn = g.reg();
    let mut rt = g.gpr();
    let mut rt2 = g.gpr();
    let mut rs = g.gpr();
    // keep the status register distinct from the data registers and the base (UNPREDICTABLE otherwise)
    while rs == rt || rs == rt2 || rs == usize::from(rn) as u8 {
        rs = g.gpr();
    }
    if rt == rt2 {
        rt2 = (rt2 + 1) % 31;
    }
    if rt2 == rs {
        rt2 = (rs + 1) % 31;
    }
    let _ = &mut rt;
    let acq = g.flag();
    let rel = g.flag();
    let o1 = pair;
    let load = asm::exclusive(
        size,
        false,
        true,
        o1,
        31,
        acq,
        if pair { rt2 } else { 31 },
        rn,
        rt,
    );
    let store = asm::exclusive(
        size,
        false,
        false,
        o1,
        rs,
        rel,
        if pair { rt2 } else { 31 },
        rn,
        rt,
    );
    let mut words = vec![load];
    match g.below(6) {
        0 => words.push(asm::clrex(15)),
        1 => words.push(asm::nop()),
        2 => {
            // an unrelated plain store in between (to another location)
            words.push(asm::ldst_unsigned(3, 0, 0, 29, 28));
        }
        _ => {}
    }
    words.push(store);
    let count_stores = words.len();
    let _ = count_stores;
    let mut case = g.case(words);
    let bytes = 1u64 << size << u64::from(pair);
    let mut addr = g.data_pointer(0x1000, 0x1000);
    if g.below(4) != 0 {
        addr &= !(bytes - 1); // mostly aligned
    }
    set_base(&mut case, rn, addr);
    case.state.x[29] = g.data_pointer(0x1000, 0x1000) & !7;
    case
}

#[test]
fn exclusive_sequences() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x1A);
    run_batch(
        "exclusives",
        &mut h,
        case_count(40_000),
        false,
        |_, _| gen_exclusive(&mut g),
        known_exclusive_differences,
    );
}

fn gen_acquire_release(g: &mut Gen) -> Case {
    let size = g.below(4) as u32;
    let rn = g.reg();
    let rt = g.reg();
    let load = g.flag();
    let word = asm::exclusive(size, true, load, false, 31, true, 31, rn, rt);
    let mut case = g.case(vec![word]);
    let bytes = 1u64 << size;
    let mut addr = g.data_pointer(0x1000, 0x1000);
    if g.below(4) != 0 {
        addr &= !(bytes - 1);
    }
    set_base(&mut case, rn, addr);
    case
}

#[test]
fn load_acquire_store_release() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x1B);
    run_batch(
        "ldar/stlr",
        &mut h,
        case_count(20_000),
        false,
        |_, _| gen_acquire_release(&mut g),
        known_exclusive_differences,
    );
}

// ---------------------------------------------------------------------------------------------
// Random words: decoder coverage and execution parity
// ---------------------------------------------------------------------------------------------

#[test]
fn random_words_decode_and_execute_like_unicorn() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0xF0);
    let n = case_count(300_000);
    let mut supported = 0usize;
    let mut unallocated_both = 0usize;
    let mut skipped_unsupported = 0usize;
    let mut skipped_unpredictable = 0usize;
    // Encodings the decoder rejects but Unicorn executes, by family.
    let mut gaps: std::collections::BTreeMap<&'static str, Vec<u32>> = Default::default();
    let mut failures: Vec<String> = Vec::new();
    for i in 0..n {
        let word = g.rng.next_u32();
        let inst = decode(word);
        if matches!(inst, Inst::Unsupported { .. }) {
            skipped_unsupported += 1;
            continue;
        }
        if constrained_unpredictable(&inst) {
            skipped_unpredictable += 1;
            continue;
        }
        // Registers: a mix of data pointers and boundary values so that memory forms hit mapped memory.
        let mut case = g.case(vec![word]);
        for r in 0..31 {
            if g.below(3) == 0 {
                case.state.x[r] = g.data_pointer(0x1000, 0x1000);
            }
        }
        case.state.pc = h.slot();
        let uc = h.run_unicorn(&case);
        let mine = h.run_mine(&case);
        match inst {
            Inst::Unallocated(_) => {
                if uc.class == Class::Exception {
                    unallocated_both += 1;
                } else {
                    gaps.entry(gap_family(word)).or_default().push(word);
                }
            }
            _ => {
                supported += 1;
                if wild_branch_target(&case, &uc, &mine).is_some()
                    || known_exclusive_differences(&case, &uc, &mine).is_some()
                {
                    continue;
                }
                if let Some(d) = common::harness::diff(&uc, &mine, false) {
                    if failures.len() < 10 {
                        failures.push(format!(
                            "word {i}:\n{}  {d}",
                            common::harness::describe(&case)
                        ));
                    }
                }
            }
        }
    }
    let total_gaps: usize = gaps.values().map(Vec::len).sum();
    eprintln!(
        "random words: {n} drawn; {supported} executed on both, {unallocated_both} unallocated on both, \
         {skipped_unsupported} in unsupported groups, {skipped_unpredictable} CONSTRAINED UNPREDICTABLE (not compared), \
         {total_gaps} strict-decoder rejections Unicorn executes"
    );
    let mut unexplained = 0usize;
    for (family, words) in &gaps {
        eprintln!("  {family}: {} words, e.g. {:#010x}", words.len(), words[0]);
        if *family == "other" {
            unexplained += words.len();
            for w in words.iter().take(30) {
                eprintln!("    unexplained: {w:#010x}");
            }
        }
    }
    assert!(
        failures.is_empty(),
        "interpreter and Unicorn disagree on random words:\n{}",
        failures.join("\n")
    );
    assert_eq!(
        unexplained, 0,
        "encodings Unicorn executes that the decoder rejects for no documented reason"
    );
}

/// Encodings the architecture marks CONSTRAINED UNPREDICTABLE: any result is allowed, so the two
/// implementations are not compared on them.
fn constrained_unpredictable(inst: &Inst) -> bool {
    match *inst {
        Inst::LdStPair {
            load,
            rt,
            rt2,
            rn,
            writeback,
            ..
        } => (load && rt == rt2 && rt != 31) || (writeback && rn != 31 && (rt == rn || rt2 == rn)),
        Inst::LdSt {
            kind, rt, rn, mode, ..
        } => {
            use nx_runtime::a64::inst::{AddrMode, MemKind};
            matches!(mode, AddrMode::PreIndex(_) | AddrMode::PostIndex(_))
                && rn != 31
                && rt == rn
                && kind != MemKind::Prefetch
        }
        Inst::Exclusive {
            load,
            pair,
            rs,
            rt,
            rt2,
            rn,
            ..
        } => {
            (!load && (rs == rt || (rs == rn && rn != 31) || (pair && rs == rt2)))
                || (load && pair && rt == rt2)
        }
        _ => false,
    }
}

/// Why a strict decoder may reject an encoding that QEMU executes.
fn gap_family(word: u32) -> &'static str {
    let bits = |hi: u32, lo: u32| (word >> lo) & (u32::MAX >> (31 - (hi - lo)));
    // LDAR/STLR (and the ordered forms) with should-be-one Rs/Rt2 fields clear.
    if bits(29, 24) == 0b001000 && bits(23, 23) == 1 && bits(21, 21) == 0 && bits(15, 15) == 1 {
        return "LDAR/STLR with should-be-one Rs/Rt2 fields not all ones";
    }
    // SMULH/UMULH with Ra not 31.
    if bits(28, 24) == 0b11011 && matches!(bits(23, 21), 0b010 | 0b110) && bits(14, 10) != 0b11111 {
        return "SMULH/UMULH with should-be-one Ra field not all ones";
    }
    "other"
}

#[test]
fn cache_identification_registers_match_the_a57() {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(0x1C);
    // CTR_EL0 and DCZID_EL0 are properties of the core, so the interpreter's constants must equal
    // the Cortex-A57 values Unicorn reports.
    for (name, op2) in [("CTR_EL0", 1u8), ("DCZID_EL0", 7)] {
        let mut case = g.case(vec![asm::sysreg(true, 3, 3, 0, 0, op2, 0)]);
        case.state.pc = h.slot();
        let (uc, mine) = (h.run_unicorn(&case), h.run_mine(&case));
        assert_eq!(
            (uc.class.clone(), mine.class.clone()),
            (Class::Ok, Class::Ok),
            "{name}"
        );
        assert_eq!(uc.state.x[0], mine.state.x[0], "{name}");
    }
    // CNTFRQ_EL0 is a platform property. Unicorn reports QEMU's 62.5 MHz default; the interpreter
    // reports the Switch's 19.2 MHz (docs/ASSUMPTIONS.md A-001, docs/UNICORN_DISAGREEMENTS.md U-8).
    let mut case = g.case(vec![asm::sysreg(true, 3, 3, 14, 0, 0, 0)]);
    case.state.pc = h.slot();
    let (uc, mine) = (h.run_unicorn(&case), h.run_mine(&case));
    assert_eq!(mine.state.x[0], 19_200_000);
    assert_eq!(
        uc.state.x[0], 62_500_000,
        "Unicorn's timer frequency changed; revisit U-8"
    );
}
