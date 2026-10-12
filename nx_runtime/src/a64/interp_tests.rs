//! Hand-computed interpreter tests. The Unicorn differential tests cover breadth; these pin the
//! semantics that matter most (flags, SP versus ZR, writeback, monitors, traps) with expected
//! values worked out by hand from the Arm pseudocode.

use super::bits::crc32_update;
use super::bus::{Bus, FaultKind};
use super::inst::SysReg;
use super::interp::{step, Trap, FPCR_MASK, FPSR_MASK};
use super::state::CpuState;
use crate::testing::{asm, FlatBus};

const BASE: u64 = 0x10_0000;
const CODE: u64 = BASE;
const DATA: u64 = BASE + 0x1000;

struct Machine {
    cpu: CpuState,
    bus: FlatBus,
}

impl Machine {
    fn new(code: &[u32]) -> Machine {
        let mut bus = FlatBus::new(BASE, 0x4000);
        bus.load_words(CODE, code);
        let cpu = CpuState {
            pc: CODE,
            sp: DATA + 0x800,
            ..CpuState::default()
        };
        Machine { cpu, bus }
    }

    fn run(&mut self, n: usize) -> Result<(), Trap> {
        for _ in 0..n {
            step(&mut self.cpu, &mut self.bus)?;
        }
        Ok(())
    }

    fn x(&self, n: usize) -> u64 {
        self.cpu.x[n]
    }

    fn flags(&self) -> u8 {
        self.cpu.nzcv4()
    }
}

/// Flag nibble as N Z C V.
const N: u8 = 8;
const Z: u8 = 4;
const C: u8 = 2;
const V: u8 = 1;

fn one(word: u32, setup: impl FnOnce(&mut CpuState)) -> Machine {
    let mut m = Machine::new(&[word]);
    setup(&mut m.cpu);
    m.run(1).expect("instruction executes");
    m
}

// ---------------------------------------------------------------------------------------------
// Arithmetic and flags
// ---------------------------------------------------------------------------------------------

#[test]
fn adds_sets_carry_and_zero_on_wraparound() {
    let w = asm::add_sub_shifted(true, false, true, 0, 2, 0, 1, 0); // adds x0, x1, x2
    let m = one(w, |c| {
        c.x[1] = u64::MAX;
        c.x[2] = 1;
    });
    assert_eq!(m.x(0), 0);
    assert_eq!(m.flags(), Z | C);
}

#[test]
fn subs_borrow_and_overflow_cases_on_32_bits() {
    let w = asm::add_sub_imm(false, true, true, false, 5, 4, 3); // subs w3, w4, #5
    let m = one(w, |c| c.x[4] = 3);
    assert_eq!(m.x(3), 0xFFFF_FFFE, "32-bit result is zero extended");
    assert_eq!(m.flags(), N);
    let m = one(w, |c| c.x[4] = 5);
    assert_eq!((m.x(3), m.flags()), (0, Z | C));
    let w = asm::add_sub_imm(false, true, true, false, 1, 4, 3); // subs w3, w4, #1
    let m = one(w, |c| c.x[4] = 0x8000_0000);
    assert_eq!((m.x(3), m.flags()), (0x7FFF_FFFF, C | V));
    // upper bits of the source register are ignored by 32-bit operations
    let m = one(w, |c| c.x[4] = 0xFFFF_FFFF_0000_0001);
    assert_eq!((m.x(3), m.flags()), (0, Z | C));
}

#[test]
fn sp_versus_zr_in_arithmetic() {
    // add x0, sp, #16 reads SP
    let m = one(
        asm::add_sub_imm(true, false, false, false, 16, 31, 0),
        |_| {},
    );
    assert_eq!(m.x(0), DATA + 0x800 + 16);
    // sub sp, sp, #32 writes SP
    let m = one(
        asm::add_sub_imm(true, true, false, false, 32, 31, 31),
        |_| {},
    );
    assert_eq!(m.cpu.sp, DATA + 0x800 - 32);
    // cmp x0, #0 (subs xzr) must not touch SP
    let m = one(asm::add_sub_imm(true, true, true, false, 0, 0, 31), |c| {
        c.x[0] = 0
    });
    assert_eq!(m.cpu.sp, DATA + 0x800);
    assert_eq!(m.flags(), Z | C);
    // adds x2, sp, #16 writes x2 (flags forms use ZR for Rd)
    let m = one(
        asm::add_sub_imm(true, false, true, false, 16, 31, 2),
        |_| {},
    );
    assert_eq!(m.x(2), DATA + 0x800 + 16);
    // shifted-register forms read register 31 as zero, not SP: add x0, xzr, x1
    let m = one(
        asm::add_sub_shifted(true, false, false, 0, 1, 0, 31, 0),
        |c| c.x[1] = 7,
    );
    assert_eq!(m.x(0), 7);
    // extended-register forms read Rn = 31 as SP: add x0, sp, x1, uxtx #2
    let m = one(asm::add_sub_ext(true, false, false, 1, 3, 2, 31, 0), |c| {
        c.x[1] = 3
    });
    assert_eq!(m.x(0), DATA + 0x800 + 12);
    // ...and write SP when not setting flags: sub sp, sp, x9
    let m = one(asm::add_sub_ext(true, true, false, 9, 3, 0, 31, 31), |c| {
        c.x[9] = 0x100
    });
    assert_eq!(m.cpu.sp, DATA + 0x800 - 0x100);
}

#[test]
fn shifted_and_extended_operands() {
    let m = one(
        asm::add_sub_shifted(true, false, false, 1, 2, 4, 1, 0),
        |c| {
            c.x[1] = 1;
            c.x[2] = 0x100;
        },
    ); // add x0, x1, x2, lsr #4
    assert_eq!(m.x(0), 0x11);
    let m = one(
        asm::add_sub_shifted(false, true, false, 2, 2, 4, 1, 0),
        |c| {
            c.x[1] = 0;
            c.x[2] = 0x8000_0000;
        },
    ); // sub w0, w1, w2, asr #4
    assert_eq!(m.x(0), (0u32.wrapping_sub(0xF800_0000)) as u64);
    let m = one(asm::add_sub_ext(true, false, false, 2, 4, 2, 1, 0), |c| {
        c.x[1] = 0x1000;
        c.x[2] = 0xFFFF_FFFF;
    }); // add x0, x1, w2, sxtw #2  => 0x1000 + (-1 << 2)
    assert_eq!(m.x(0), 0x1000 - 4);
    let m = one(asm::add_sub_ext(true, false, true, 2, 1, 1, 1, 0), |c| {
        c.x[1] = 0;
        c.x[2] = 0x1_8000;
    }); // adds x0, x1, w2, uxth #1
    assert_eq!(m.x(0), 0x1_0000);
}

#[test]
fn add_and_subtract_with_carry() {
    let adc = asm::add_sub_carry(true, false, false, 2, 1, 0);
    let m = one(adc, |c| {
        c.x[1] = 5;
        c.x[2] = 6;
        c.set_nzcv4(C);
    });
    assert_eq!(m.x(0), 12);
    let sbc = asm::add_sub_carry(true, true, false, 2, 1, 0); // sbc x0, x1, x2 = x1 - x2 - !C
    let m = one(sbc, |c| {
        c.x[1] = 5;
        c.x[2] = 6;
        c.set_nzcv4(0);
    });
    assert_eq!(m.x(0), 5u64.wrapping_sub(6).wrapping_sub(1));
    let sbcs = asm::add_sub_carry(false, true, true, 2, 1, 0); // sbcs w0, w1, w2
    let m = one(sbcs, |c| {
        c.x[1] = 1;
        c.x[2] = 1;
        c.set_nzcv4(C);
    });
    assert_eq!((m.x(0), m.flags()), (0, Z | C));
    let ngc = asm::add_sub_carry(true, true, false, 13, 31, 12); // ngc x12, x13
    let m = one(ngc, |c| {
        c.x[13] = 0;
        c.set_nzcv4(0);
    });
    assert_eq!(m.x(12), u64::MAX, "0 - 0 - 1");
}

// ---------------------------------------------------------------------------------------------
// Moves, logical, bitfield
// ---------------------------------------------------------------------------------------------

#[test]
fn movz_movk_movn_compose_constants() {
    let mut m = Machine::new(&[
        asm::mov_wide(true, 2, 0, 0x7788, 0),
        asm::mov_wide(true, 3, 1, 0x5566, 0),
        asm::mov_wide(true, 3, 2, 0x3344, 0),
        asm::mov_wide(true, 3, 3, 0x1122, 0),
        asm::mov_wide(false, 2, 0, 0xFFFF, 1),
        asm::mov_wide(false, 3, 1, 0xAAAA, 1),
        asm::mov_wide(true, 0, 0, 0x1234, 2),
        asm::mov_wide(false, 0, 1, 0x1234, 3),
        asm::mov_wide(true, 2, 0, 5, 31), // movz xzr is discarded
    ]);
    m.run(4).unwrap();
    assert_eq!(m.x(0), 0x1122_3344_5566_7788);
    m.run(2).unwrap();
    assert_eq!(
        m.x(1),
        0xAAAA_FFFF,
        "32-bit movk keeps the upper half clear"
    );
    m.run(1).unwrap();
    assert_eq!(m.x(2), !0x1234u64);
    m.run(1).unwrap();
    assert_eq!(m.x(3), 0xEDCB_FFFF, "32-bit movn is zero extended");
    m.run(1).unwrap();
}

#[test]
fn logical_immediates_and_ands_flags() {
    let and = asm::logical_imm(true, 0, true, 0, 7, 1, 0); // and x0, x1, #0xff
    assert_eq!(one(and, |c| c.x[1] = 0x1234).x(0), 0x34);
    let orr = asm::logical_imm(false, 1, false, 1, 0b111100, 3, 2); // orr w2, w3, #0xaaaaaaaa
    assert_eq!(one(orr, |c| c.x[3] = 0x5555_5555).x(2), 0xFFFF_FFFF);
    let ands = asm::logical_imm(true, 3, true, 0, 0, 7, 31); // tst x7, #1 (ands xzr)
    let m = one(ands, |c| c.x[7] = 2);
    assert_eq!(m.flags(), Z);
    let m = one(ands, |c| c.x[7] = 3);
    assert_eq!(m.flags(), 0);
    let ands_n = asm::logical_imm(true, 3, true, 1, 0, 1, 0); // ands x0, x1, #0x8000_0000_0000_0000
    assert_eq!(one(ands_n, |c| c.x[1] = u64::MAX).flags(), N);
    // AND/ORR/EOR immediate write SP when Rd = 31; ANDS writes ZR
    let and_sp = asm::logical_imm(true, 0, true, 0, 7, 1, 31); // and sp, x1, #0xff
    assert_eq!(one(and_sp, |c| c.x[1] = 0x1234).cpu.sp, 0x34);
}

#[test]
fn logical_shifted_registers() {
    let bic = asm::logical_shifted(true, 0, 1, true, 8, 4, 7, 6); // bic x6, x7, x8, lsr #4
    let m = one(bic, |c| {
        c.x[7] = u64::MAX;
        c.x[8] = 0xF0;
    });
    assert_eq!(m.x(6), !0xFu64);
    let orn = asm::logical_shifted(false, 1, 3, true, 14, 6, 13, 12); // orn w12, w13, w14, ror #6
    let m = one(orn, |c| {
        c.x[13] = 0;
        c.x[14] = 0x40;
    });
    assert_eq!(m.x(12), !(0x40u32.rotate_right(6)) as u64);
    let eon = asm::logical_shifted(false, 2, 0, true, 20, 0, 19, 18); // eon w18, w19, w20
    let m = one(eon, |c| {
        c.x[19] = 0xF0F0_F0F0;
        c.x[20] = 0xFF00_FF00;
    });
    assert_eq!(m.x(18), !(0xF0F0_F0F0u32 ^ 0xFF00_FF00) as u64);
    let mvn = asm::logical_shifted(true, 1, 0, true, 3, 0, 31, 2); // mvn x2, x3
    assert_eq!(one(mvn, |c| c.x[3] = 5).x(2), !5u64);
    let bics = asm::logical_shifted(false, 3, 0, true, 26, 0, 25, 24); // bics w24, w25, w26
    let m = one(bics, |c| {
        c.x[25] = 0x0F;
        c.x[26] = 0x0F;
    });
    assert_eq!((m.x(24), m.flags()), (0, Z));
}

#[test]
fn bitfield_moves_and_aliases() {
    let ubfx = asm::bitfield(false, 2, false, 8, 15, 3, 2); // ubfx w2, w3, #8, #8
    assert_eq!(one(ubfx, |c| c.x[3] = 0xAABB_CCDD).x(2), 0xCC);
    let sbfx = asm::bitfield(true, 0, true, 4, 7, 1, 0); // sbfx x0, x1, #4, #4
    assert_eq!(one(sbfx, |c| c.x[1] = 0xF0).x(0), u64::MAX);
    assert_eq!(one(sbfx, |c| c.x[1] = 0x70).x(0), 7);
    let bfi = asm::bitfield(true, 1, true, 56, 15, 5, 4); // bfi x4, x5, #8, #16
    let m = one(bfi, |c| {
        c.x[4] = u64::MAX;
        c.x[5] = 0x1234;
    });
    assert_eq!(m.x(4), 0xFFFF_FFFF_FF12_34FF);
    let bfxil = asm::bitfield(false, 1, false, 4, 7, 7, 6); // bfxil w6, w7, #4, #4
    let m = one(bfxil, |c| {
        c.x[6] = 0xFFFF_FFFF;
        c.x[7] = 0xA0;
    });
    assert_eq!(m.x(6), 0xFFFF_FFFA);
    let sxtb = asm::bitfield(true, 0, true, 0, 7, 1, 0); // sxtb x0, w1
    assert_eq!(one(sxtb, |c| c.x[1] = 0x80).x(0), 0xFFFF_FFFF_FFFF_FF80);
    let uxth = asm::bitfield(false, 2, false, 0, 15, 3, 2); // uxth w2, w3
    assert_eq!(one(uxth, |c| c.x[3] = 0xFFFF_ABCD).x(2), 0xABCD);
    let lsl = asm::bitfield(true, 2, true, 60, 59, 1, 0); // lsl x0, x1, #4
    assert_eq!(
        one(lsl, |c| c.x[1] = 0x0F00_0000_0000_0001).x(0),
        0xF000_0000_0000_0010
    );
    let lsr = asm::bitfield(false, 2, false, 3, 31, 3, 2); // lsr w2, w3, #3
    assert_eq!(one(lsr, |c| c.x[3] = 0x8000_0000).x(2), 0x1000_0000);
    let asr = asm::bitfield(true, 0, true, 63, 63, 5, 4); // asr x4, x5, #63
    assert_eq!(one(asr, |c| c.x[5] = 1 << 63).x(4), u64::MAX);
}

#[test]
fn extract_and_rotate() {
    let extr = asm::extr(true, true, 2, 12, 1, 0); // extr x0, x1, x2, #12
    let m = one(extr, |c| {
        c.x[1] = 1;
        c.x[2] = 0xFFFF_0000_0000_0000;
    });
    assert_eq!(m.x(0), (0xFFFF_0000_0000_0000u64 >> 12) | (1u64 << 52));
    let ror = asm::extr(false, false, 4, 7, 4, 3); // ror w3, w4, #7
    assert_eq!(one(ror, |c| c.x[4] = 0x8000_0001).x(3), 0x0300_0000);
    let lsb0 = asm::extr(true, true, 2, 0, 1, 0);
    assert_eq!(
        one(lsb0, |c| {
            c.x[1] = 9;
            c.x[2] = 7
        })
        .x(0),
        7,
        "lsb 0 selects Rm"
    );
}

// ---------------------------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------------------------

fn run_to(m: &mut Machine, pc: u64, limit: usize) {
    for _ in 0..limit {
        if m.cpu.pc == pc {
            return;
        }
        m.run(1).unwrap();
    }
    panic!("did not reach {pc:#x}, stopped at {:#x}", m.cpu.pc);
}

#[test]
fn counting_loop_with_conditional_branch() {
    // x0 = sum of 10..=1
    let mut m = Machine::new(&[
        asm::mov_wide(true, 2, 0, 0, 0),  // movz x0, #0
        asm::mov_wide(true, 2, 0, 10, 1), // movz x1, #10
        asm::add_sub_shifted(true, false, false, 0, 1, 0, 0, 0), // loop: add x0, x0, x1
        asm::add_sub_imm(true, true, true, false, 1, 1, 1), // subs x1, x1, #1
        asm::b_cond(1, -2),               // b.ne loop
        asm::nop(),
    ]);
    run_to(&mut m, CODE + 0x14, 200);
    assert_eq!(m.x(0), 55);
    assert_eq!(m.x(1), 0);
}

#[test]
fn call_and_return_through_the_link_register() {
    let mut m = Machine::new(&[
        asm::b_imm(true, 3),              // 0: bl +12 (to 0xc)
        asm::mov_wide(true, 2, 0, 99, 5), // 4: movz x5, #99  (after return)
        asm::b_imm(false, 3),             // 8: b +12 (to 0x14)
        asm::mov_wide(true, 2, 0, 7, 4),  // c: movz x4, #7   (callee)
        asm::branch_reg(2, 30),           // 10: ret
        asm::nop(),                       // 14
    ]);
    m.run(1).unwrap();
    assert_eq!(m.x(30), CODE + 4);
    assert_eq!(m.cpu.pc, CODE + 12);
    run_to(&mut m, CODE + 0x14, 20);
    assert_eq!((m.x(4), m.x(5)), (7, 99));
}

#[test]
fn compare_and_test_branches() {
    // cbz x0 taken, cbnz w1 not taken (upper bits ignored), tbz/tbnz on specific bits
    let cases: Vec<(u32, [u64; 2], bool)> = vec![
        (asm::cbz(true, false, 4, 0), [0, 0], true),
        (asm::cbz(true, false, 4, 0), [1, 0], false),
        (asm::cbz(false, false, 4, 0), [0x1_0000_0000, 0], true), // w0 is zero
        (asm::cbz(false, true, 4, 0), [0x1_0000_0000, 0], false),
        (asm::cbz(true, true, 4, 0), [0x1_0000_0000, 0], true),
        (asm::tbz(false, 5, 4, 0), [1 << 5, 0], false),
        (asm::tbz(false, 5, 4, 0), [!(1 << 5), 0], true),
        (asm::tbz(true, 63, 4, 0), [1 << 63, 0], true), // tbnz x0, #63
        (asm::tbz(true, 31, 4, 0), [1 << 32, 0], false),
        (asm::tbz(false, 33, 4, 0), [1 << 33, 0], false),
    ];
    for (word, regs, taken) in cases {
        let m = one(word, |c| c.x[0] = regs[0]);
        let expect = if taken { CODE + 16 } else { CODE + 4 };
        assert_eq!(m.cpu.pc, expect, "{word:#010x} with x0={:#x}", regs[0]);
    }
}

#[test]
fn indirect_branches_and_link() {
    let m = one(asm::branch_reg(0, 16), |c| c.x[16] = 0x4000);
    assert_eq!(m.cpu.pc, 0x4000);
    assert_eq!(m.x(30), 0, "br does not link");
    let m = one(asm::branch_reg(1, 8), |c| c.x[8] = 0x5000);
    assert_eq!((m.cpu.pc, m.x(30)), (0x5000, CODE + 4));
    // blr x30 branches to the old link value, then links
    let m = one(asm::branch_reg(1, 30), |c| c.x[30] = 0x6000);
    assert_eq!((m.cpu.pc, m.x(30)), (0x6000, CODE + 4));
    // register 31 in BR is XZR
    let m = one(asm::branch_reg(0, 31), |_| {});
    assert_eq!(m.cpu.pc, 0);
}

#[test]
fn adr_and_adrp() {
    let mut m = Machine::new(&[asm::pc_rel(false, 0x20, 1), asm::pc_rel(true, 3, 2)]);
    m.cpu.pc = CODE + 0x44;
    m.bus.load_words(
        CODE + 0x44,
        &[asm::pc_rel(false, -4, 1), asm::pc_rel(true, 3, 2)],
    );
    m.run(2).unwrap();
    assert_eq!(m.x(1), CODE + 0x44 - 4);
    assert_eq!(m.x(2), ((CODE + 0x48) & !0xFFF) + 3 * 4096);
}

// ---------------------------------------------------------------------------------------------
// Conditional instructions
// ---------------------------------------------------------------------------------------------

#[test]
fn conditional_select_family() {
    for (op, cond_true, expect_true, expect_false) in [
        (0u32, 0u8, 11u64, 22u64),        // csel
        (1, 0, 11, 23),                   // csinc
        (2, 0, 11, !22u64),               // csinv
        (3, 0, 11, 22u64.wrapping_neg()), // csneg
    ] {
        let w = asm::cond_sel(true, op, 2, cond_true, 1, 0); // op x0, x1, x2, eq
        let m = one(w, |c| {
            c.x[1] = 11;
            c.x[2] = 22;
            c.set_nzcv4(Z);
        });
        assert_eq!(m.x(0), expect_true, "op {op} true");
        let m = one(w, |c| {
            c.x[1] = 11;
            c.x[2] = 22;
            c.set_nzcv4(0);
        });
        assert_eq!(m.x(0), expect_false, "op {op} false");
    }
    // 32-bit CSINC wraps in 32 bits
    let w = asm::cond_sel(false, 1, 2, 0, 1, 0);
    let m = one(w, |c| {
        c.x[1] = 5;
        c.x[2] = 0xFFFF_FFFF;
    });
    assert_eq!(m.x(0), 0);
    // cset x12, eq == csinc x12, xzr, xzr, ne
    let cset = asm::cond_sel(true, 1, 31, 1, 31, 12);
    assert_eq!(one(cset, |c| c.set_nzcv4(Z)).x(12), 1);
    assert_eq!(one(cset, |c| c.set_nzcv4(0)).x(12), 0);
}

#[test]
fn conditional_compare() {
    // ccmp x0, x1, #4, eq
    let w = asm::cond_cmp(true, true, false, 1, 0, 0, 4);
    let m = one(w, |c| {
        c.x[0] = 5;
        c.x[1] = 5;
        c.set_nzcv4(Z);
    });
    assert_eq!(m.flags(), Z | C, "condition true: flags from cmp x0, x1");
    let m = one(w, |c| {
        c.x[0] = 5;
        c.x[1] = 5;
        c.set_nzcv4(0);
    });
    assert_eq!(m.flags(), 4, "condition false: the immediate nzcv");
    // ccmn w2, #31, #8, lt
    let w = asm::cond_cmp(false, false, true, 31, 11, 2, 8);
    let m = one(w, |c| {
        c.x[2] = 0xFFFF_FFFF;
        c.set_nzcv4(N);
    });
    assert_eq!(m.flags(), C, "w2 + 31 wraps to 30: N=0 Z=0 C=1 V=0");
}

// ---------------------------------------------------------------------------------------------
// Divide, multiply, bit manipulation, CRC
// ---------------------------------------------------------------------------------------------

#[test]
fn division_edge_cases_follow_the_architecture() {
    let udiv = asm::dp2(true, 0b000010, 2, 1, 0);
    let sdiv = asm::dp2(true, 0b000011, 2, 1, 0);
    let udiv_w = asm::dp2(false, 0b000010, 2, 1, 0);
    let sdiv_w = asm::dp2(false, 0b000011, 2, 1, 0);
    let go = |w: u32, a: u64, b: u64| {
        one(w, |c| {
            c.x[1] = a;
            c.x[2] = b;
        })
        .x(0)
    };
    assert_eq!(go(udiv, 100, 7), 14);
    assert_eq!(go(udiv, 100, 0), 0, "divide by zero gives zero");
    assert_eq!(go(udiv, u64::MAX, 2), u64::MAX / 2);
    assert_eq!(
        go(sdiv, (-100i64) as u64, 7),
        (-14i64) as u64,
        "rounds toward zero"
    );
    assert_eq!(go(sdiv, 100, (-7i64) as u64), (-14i64) as u64);
    assert_eq!(go(sdiv, 100, 0), 0);
    assert_eq!(
        go(sdiv, i64::MIN as u64, u64::MAX),
        i64::MIN as u64,
        "MIN / -1 wraps"
    );
    assert_eq!(
        go(sdiv_w, 0x8000_0000, 0xFFFF_FFFF),
        0x8000_0000,
        "32-bit MIN / -1 wraps, zero extended"
    );
    assert_eq!(
        go(sdiv_w, 0xFFFF_FFFF_0000_0010, 4),
        4,
        "upper halves are ignored"
    );
    assert_eq!(go(udiv_w, 0xFFFF_FFFF, 3), 0x5555_5555);
    assert_eq!(go(udiv_w, 1, 0), 0);
    assert_eq!(go(sdiv_w, 0xFFFF_FFF6, 3), 0xFFFF_FFFD, "-10 / 3 = -3");
}

#[test]
fn multiply_and_multiply_long() {
    let go = |w: u32, a: u64, b: u64, acc: u64| {
        one(w, |c| {
            c.x[1] = a;
            c.x[2] = b;
            c.x[3] = acc;
        })
        .x(0)
    };
    assert_eq!(go(asm::dp3(true, 0, false, 2, 3, 1, 0), 6, 7, 100), 142); // madd
    assert_eq!(go(asm::dp3(true, 0, true, 2, 3, 1, 0), 6, 7, 100), 58); // msub
    assert_eq!(
        go(asm::dp3(true, 0, false, 2, 31, 1, 0), u64::MAX, 2, 0),
        u64::MAX - 1
    ); // mul wraps
    assert_eq!(
        go(asm::dp3(false, 0, false, 2, 3, 1, 0), 0xFFFF_FFFF, 2, 5),
        3
    ); // 32-bit madd wraps
    assert_eq!(
        go(
            asm::dp3(true, 1, false, 2, 3, 1, 0),
            0xFFFF_FFFF,
            0xFFFF_FFFF,
            10
        ),
        11
    ); // smaddl (-1*-1)
    assert_eq!(
        go(asm::dp3(true, 1, true, 2, 3, 1, 0), 0xFFFF_FFFF, 5, 10),
        15
    ); // smsubl 10 - (-1*5)
    assert_eq!(
        go(
            asm::dp3(true, 5, false, 2, 31, 1, 0),
            0xFFFF_FFFF,
            0xFFFF_FFFF,
            0
        ),
        0xFFFF_FFFE_0000_0001
    ); // umull
    assert_eq!(go(asm::dp3(true, 5, true, 2, 3, 1, 0), 3, 4, 20), 8); // umsubl
    assert_eq!(
        go(asm::dp3(true, 2, false, 2, 31, 1, 0), u64::MAX, u64::MAX, 0),
        0
    ); // smulh(-1, -1) = 0
    assert_eq!(
        go(asm::dp3(true, 2, false, 2, 31, 1, 0), i64::MIN as u64, 2, 0),
        u64::MAX
    ); // smulh(MIN, 2) = -1
    assert_eq!(
        go(asm::dp3(true, 6, false, 2, 31, 1, 0), u64::MAX, u64::MAX, 0),
        u64::MAX - 1
    ); // umulh
    assert_eq!(go(asm::dp3(true, 6, false, 2, 31, 1, 0), 1 << 63, 4, 0), 2); // umulh
}

#[test]
fn shifts_by_register_use_the_amount_modulo_the_width() {
    let lslv = asm::dp2(true, 0b001000, 2, 1, 0);
    let lsrv_w = asm::dp2(false, 0b001001, 2, 1, 0);
    let asrv = asm::dp2(true, 0b001010, 2, 1, 0);
    let rorv_w = asm::dp2(false, 0b001011, 2, 1, 0);
    let go = |w: u32, a: u64, b: u64| {
        one(w, |c| {
            c.x[1] = a;
            c.x[2] = b;
        })
        .x(0)
    };
    assert_eq!(go(lslv, 1, 65), 2);
    assert_eq!(go(lslv, 1, 63), 1 << 63);
    assert_eq!(go(lsrv_w, 0x8000_0000, 33), 0x4000_0000);
    assert_eq!(go(asrv, 1 << 63, 63), u64::MAX);
    assert_eq!(go(rorv_w, 0x0000_0001, 1), 0x8000_0000);
}

#[test]
fn bit_and_byte_manipulation() {
    let go = |sf: bool, opcode: u32, v: u64| one(asm::dp1(sf, opcode, 1, 0), |c| c.x[1] = v).x(0);
    assert_eq!(go(true, 0, 1), 1 << 63, "rbit x");
    assert_eq!(go(false, 0, 1), 1 << 31, "rbit w");
    assert_eq!(go(false, 1, 0x1122_3344), 0x2211_4433, "rev16 w");
    assert_eq!(
        go(true, 1, 0x1122_3344_5566_7788),
        0x2211_4433_6655_8877,
        "rev16 x"
    );
    assert_eq!(go(false, 2, 0x1122_3344), 0x4433_2211, "rev w");
    assert_eq!(
        go(true, 2, 0x1122_3344_5566_7788),
        0x4433_2211_8877_6655,
        "rev32 x"
    );
    assert_eq!(
        go(true, 3, 0x1122_3344_5566_7788),
        0x8877_6655_4433_2211,
        "rev x"
    );
    assert_eq!(go(true, 4, 0), 64);
    assert_eq!(go(false, 4, 0), 32);
    assert_eq!(go(true, 4, 1), 63);
    assert_eq!(
        go(false, 4, 0xFFFF_FFFF_0000_0001),
        31,
        "clz w ignores the upper half"
    );
    assert_eq!(go(true, 5, 0), 63);
    assert_eq!(go(false, 5, 0xFFFF_FFFF), 31);
    assert_eq!(go(true, 5, 1), 62);
    assert_eq!(go(false, 5, 0x8000_0000), 0);
}

#[test]
fn crc32_instructions_match_the_software_definition() {
    // crc32b w0, w0, w1 over "123456789" with the usual init and final inversion
    let crcb = asm::dp2(false, 0b010000, 1, 0, 0);
    let mut m = Machine::new(&[]);
    m.cpu.x[0] = 0xFFFF_FFFF;
    for (i, &b) in b"123456789".iter().enumerate() {
        m.bus.load_words(CODE + 4 * i as u64, &[crcb]);
        m.cpu.x[1] = u64::from(b);
        m.cpu.pc = CODE + 4 * i as u64;
        m.run(1).unwrap();
    }
    assert_eq!(!(m.x(0) as u32), 0xCBF4_3926);
    // crc32cx and crc32w agree with the helper
    for (opcode, size, castagnoli, sf) in [
        (0b010011u32, 3u8, false, true),
        (0b010111, 3, true, true),
        (0b010010, 2, false, false),
        (0b010110, 2, true, false),
        (0b010001, 1, false, false),
        (0b010101, 1, true, false),
    ] {
        let m = one(asm::dp2(sf, opcode, 2, 1, 0), |c| {
            c.x[1] = 0xDEAD_BEEF_1234_5678;
            c.x[2] = 0x0102_0304_0506_0708;
        });
        let data = if size == 3 {
            0x0102_0304_0506_0708
        } else {
            0x0102_0304_0506_0708u64 & ((1u64 << (8 << size)) - 1)
        };
        let expect = crc32_update(0x1234_5678, data, size, castagnoli);
        assert_eq!(m.x(0), u64::from(expect), "opcode {opcode:#b}");
    }
}

// ---------------------------------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------------------------------

fn data_machine(word: u32, setup: impl FnOnce(&mut Machine)) -> Machine {
    let mut m = Machine::new(&[word]);
    for i in 0..0x100u64 {
        m.bus.poke(
            DATA + i,
            1,
            u128::from((i as u8).wrapping_mul(3).wrapping_add(1)),
        );
    }
    m.cpu.x[1] = DATA + 0x10;
    setup(&mut m);
    m.run(1).expect("instruction executes");
    m
}

fn byte_at(i: u64) -> u8 {
    (i as u8).wrapping_mul(3).wrapping_add(1)
}

fn le(offset: u64, n: usize) -> u64 {
    (0..n as u64).fold(0u64, |acc, k| {
        acc | (u64::from(byte_at(offset + k)) << (8 * k))
    })
}

#[test]
fn loads_extend_correctly() {
    let off = 0x10;
    // ldr x0, [x1]; ldr w0, [x1]; ldrh; ldrb
    assert_eq!(
        data_machine(asm::ldst_unsigned(3, 1, 0, 1, 0), |_| {}).x(0),
        le(off, 8)
    );
    assert_eq!(
        data_machine(asm::ldst_unsigned(2, 1, 0, 1, 0), |_| {}).x(0),
        le(off, 4)
    );
    assert_eq!(
        data_machine(asm::ldst_unsigned(1, 1, 0, 1, 0), |_| {}).x(0),
        le(off, 2)
    );
    assert_eq!(
        data_machine(asm::ldst_unsigned(0, 1, 0, 1, 0), |_| {}).x(0),
        le(off, 1)
    );
    // byte 0x10*3+1 = 0x31, 0x13*3+1 = 0x3a etc. are positive; find signed examples at bigger offsets
    let neg = (0..0x100u64)
        .find(|&i| byte_at(i) & 0x80 != 0 && byte_at(i + 1) & 0x80 != 0)
        .unwrap();
    let setup = move |m: &mut Machine| m.cpu.x[1] = DATA + neg;
    let sb = byte_at(neg);
    assert_eq!(
        data_machine(asm::ldst_unsigned(0, 2, 0, 1, 0), setup).x(0),
        i64::from(sb as i8) as u64,
        "ldrsb x"
    );
    assert_eq!(
        data_machine(asm::ldst_unsigned(0, 3, 0, 1, 0), setup).x(0),
        u64::from(sb as i8 as i32 as u32),
        "ldrsb w"
    );
    let sh = le(neg, 2) as u16;
    assert_eq!(
        data_machine(asm::ldst_unsigned(1, 2, 0, 1, 0), setup).x(0),
        i64::from(sh as i16) as u64,
        "ldrsh x"
    );
    assert_eq!(
        data_machine(asm::ldst_unsigned(1, 3, 0, 1, 0), setup).x(0),
        u64::from(sh as i16 as i32 as u32),
        "ldrsh w"
    );
    let sw = le(neg, 4) as u32;
    assert_eq!(
        data_machine(asm::ldst_unsigned(2, 2, 0, 1, 0), setup).x(0),
        i64::from(sw as i32) as u64,
        "ldrsw"
    );
}

#[test]
fn stores_write_only_their_width() {
    let m = data_machine(asm::ldst_unsigned(2, 0, 0, 1, 5), |m| {
        m.cpu.x[5] = 0x1122_3344_5566_7788
    });
    assert_eq!(m.bus.peek(DATA + 0x10, 4), 0x5566_7788);
    assert_eq!(
        m.bus.peek(DATA + 0x14, 1),
        u128::from(byte_at(0x14)),
        "neighbouring byte untouched"
    );
    let m = data_machine(asm::ldst_unsigned(0, 0, 0, 1, 5), |m| m.cpu.x[5] = 0xABCD);
    assert_eq!(
        m.bus.peek(DATA + 0x10, 2),
        (0xCD | (u128::from(byte_at(0x11)) << 8))
    );
    let m = data_machine(asm::ldst_unsigned(3, 0, 0, 1, 31), |_| {}); // str xzr, [x1]
    assert_eq!(m.bus.peek(DATA + 0x10, 8), 0);
}

#[test]
fn addressing_modes_and_writeback() {
    // pre-index: ldr x0, [x1, #8]!
    let m = data_machine(asm::ldst_imm9(3, 1, 8, 3, 1, 0), |_| {});
    assert_eq!((m.x(0), m.x(1)), (le(0x18, 8), DATA + 0x18));
    // post-index: ldr w2, [x1], #4
    let m = data_machine(asm::ldst_imm9(2, 1, 4, 1, 1, 2), |_| {});
    assert_eq!((m.x(2), m.x(1)), (le(0x10, 4), DATA + 0x14));
    // negative unscaled: ldur x0, [x1, #-8]
    let m = data_machine(asm::ldst_imm9(3, 1, -8, 0, 1, 0), |_| {});
    assert_eq!((m.x(0), m.x(1)), (le(0x08, 8), DATA + 0x10));
    // unprivileged variants behave as plain accesses: ldtr x0, [x1, #8]
    let m = data_machine(asm::ldst_imm9(3, 1, 8, 2, 1, 0), |_| {});
    assert_eq!((m.x(0), m.x(1)), (le(0x18, 8), DATA + 0x10));
    // register offset: ldr x0, [x1, x2, lsl #3]
    let m = data_machine(asm::ldst_reg(3, 1, 2, 3, true, 1, 0), |m| m.cpu.x[2] = 2);
    assert_eq!(m.x(0), le(0x10 + 16, 8));
    // register offset with sign-extended word: ldr w0, [x1, w2, sxtw #2] with w2 = -2
    let m = data_machine(asm::ldst_reg(2, 1, 2, 6, true, 1, 0), |m| {
        m.cpu.x[2] = 0xFFFF_FFFE
    });
    assert_eq!(m.x(0), le(0x10 - 8, 4));
    // register offset with unsigned word extend ignores the upper half
    let m = data_machine(asm::ldst_reg(2, 1, 2, 2, false, 1, 0), |m| {
        m.cpu.x[2] = 0xFFFF_FFFF_0000_0004
    });
    assert_eq!(m.x(0), le(0x14, 4));
    // sp-based access: str x5, [sp, #8]
    let m = data_machine(asm::ldst_unsigned(3, 0, 1, 31, 5), |m| m.cpu.x[5] = 0xAAAA);
    assert_eq!(m.bus.peek(DATA + 0x800 + 8, 8), 0xAAAA);
    // writeback to sp: str x5, [sp, #-16]!
    let m = data_machine(asm::ldst_imm9(3, 0, -16, 3, 31, 5), |m| m.cpu.x[5] = 0xBBBB);
    assert_eq!(m.cpu.sp, DATA + 0x800 - 16);
    assert_eq!(m.bus.peek(DATA + 0x800 - 16, 8), 0xBBBB);
}

#[test]
fn literal_loads_and_prefetch() {
    let mut m = Machine::new(&[
        asm::ldr_literal(1, 8, 0),
        asm::ldr_literal(0, 9, 3),
        asm::ldr_literal(2, 10, 4),
    ]);
    m.bus.poke(CODE + 32, 8, 0x1122_3344_5566_7788);
    m.bus.poke(CODE + 40, 4, 0xFFFF_FFF0);
    m.bus.poke(CODE + 48, 4, 0x8000_0001);
    m.run(3).unwrap();
    assert_eq!(m.x(0), 0x1122_3344_5566_7788);
    assert_eq!(m.x(3), 0xFFFF_FFF0);
    assert_eq!(m.x(4), 0xFFFF_FFFF_8000_0001, "ldrsw literal");
    // prfm has no architectural effect and does not fault on unmapped addresses
    let m = one(asm::ldst_unsigned(3, 2, 0, 0, 0), |c| {
        c.x[0] = 0xDEAD_0000_0000
    });
    assert_eq!(m.cpu.pc, CODE + 4);
}

#[test]
fn load_and_store_pairs() {
    // ldp x0, x1, [x2, #16]
    let m = data_machine(asm::ldst_pair(2, 2, true, 2, 1, 2, 0), |m| {
        m.cpu.x[2] = DATA
    });
    assert_eq!((m.x(0), m.x(1)), (le(16, 8), le(24, 8)));
    // stp x29, x30, [sp, #-16]! then ldp x29, x30, [sp], #16
    let mut m = Machine::new(&[
        asm::ldst_pair(2, 3, false, -2, 30, 31, 29),
        asm::ldst_pair(2, 1, true, 2, 30, 31, 29),
    ]);
    m.cpu.x[29] = 0x1111;
    m.cpu.x[30] = 0x2222;
    let sp0 = m.cpu.sp;
    m.run(1).unwrap();
    assert_eq!(m.cpu.sp, sp0 - 16);
    assert_eq!(
        (m.bus.peek(sp0 - 16, 8), m.bus.peek(sp0 - 8, 8)),
        (0x1111, 0x2222)
    );
    m.cpu.x[29] = 0;
    m.cpu.x[30] = 0;
    m.run(1).unwrap();
    assert_eq!((m.x(29), m.x(30), m.cpu.sp), (0x1111, 0x2222, sp0));
    // ldpsw x6, x7, [x8, #8] sign extends both words
    let mut m = Machine::new(&[asm::ldst_pair(1, 2, true, 2, 7, 8, 6)]);
    m.cpu.x[8] = DATA;
    m.bus.poke(DATA + 8, 4, 0x8000_0000);
    m.bus.poke(DATA + 12, 4, 0x0000_0001);
    m.run(1).unwrap();
    assert_eq!((m.x(6), m.x(7)), (0xFFFF_FFFF_8000_0000, 1));
    // w registers: stp w3, w4, [x5]; ldnp / stnp behave like plain accesses
    let mut m = Machine::new(&[
        asm::ldst_pair(0, 2, false, 0, 4, 5, 3),
        asm::ldst_pair(0, 0, true, 0, 7, 5, 6),
    ]);
    m.cpu.x[3] = 0xAAAA_BBBB_CCCC_DDDD;
    m.cpu.x[4] = 0x1111_2222_3333_4444;
    m.cpu.x[5] = DATA;
    m.run(2).unwrap();
    assert_eq!((m.x(6), m.x(7)), (0xCCCC_DDDD, 0x3333_4444));
}

// ---------------------------------------------------------------------------------------------
// Exclusives and ordered accesses
// ---------------------------------------------------------------------------------------------

fn ldxr(size: u32, rn: u8, rt: u8) -> u32 {
    asm::exclusive(size, false, true, false, 31, false, 31, rn, rt)
}

fn stxr(size: u32, rs: u8, rn: u8, rt: u8) -> u32 {
    asm::exclusive(size, false, false, false, rs, false, 31, rn, rt)
}

#[test]
fn exclusive_pairs_of_instructions() {
    let mut m = Machine::new(&[ldxr(3, 1, 0), stxr(3, 2, 1, 3)]);
    m.cpu.x[1] = DATA;
    m.cpu.x[3] = 0xFEED;
    m.bus.poke(DATA, 8, 0x77);
    m.run(2).unwrap();
    assert_eq!((m.x(0), m.x(2)), (0x77, 0), "load value and success status");
    assert_eq!(m.bus.peek(DATA, 8), 0xFEED);

    // CLREX between them makes the store fail without storing
    let mut m = Machine::new(&[ldxr(3, 1, 0), asm::clrex(15), stxr(3, 2, 1, 3)]);
    m.cpu.x[1] = DATA;
    m.cpu.x[3] = 0xFEED;
    m.run(3).unwrap();
    assert_eq!(m.x(2), 1);
    assert_eq!(m.bus.peek(DATA, 8), 0);

    // a store without a preceding load fails; a second store after success fails
    let mut m = Machine::new(&[
        stxr(2, 2, 1, 3),
        ldxr(2, 1, 0),
        stxr(2, 4, 1, 3),
        stxr(2, 5, 1, 3),
    ]);
    m.cpu.x[1] = DATA;
    m.cpu.x[3] = 9;
    m.run(4).unwrap();
    assert_eq!((m.x(2), m.x(4), m.x(5)), (1, 0, 1));

    // a store to a different address than the load fails
    let mut m = Machine::new(&[ldxr(3, 1, 0), stxr(3, 2, 6, 3)]);
    m.cpu.x[1] = DATA;
    m.cpu.x[6] = DATA + 8;
    m.run(2).unwrap();
    assert_eq!(m.x(2), 1);
    assert_eq!(m.bus.peek(DATA + 8, 8), 0);

    // sizes: ldxrb / stxrh keep their widths
    let mut m = Machine::new(&[ldxr(0, 1, 0), stxr(1, 2, 1, 3)]);
    m.cpu.x[1] = DATA;
    m.bus.poke(DATA, 4, 0x1234_5678);
    m.run(2).unwrap();
    assert_eq!(
        (m.x(0), m.x(2)),
        (0x78, 1),
        "different size than the load: store fails"
    );
}

#[test]
fn exclusive_register_pairs() {
    let ldxp = asm::exclusive(3, false, true, true, 31, false, 4, 1, 3); // ldxp x3, x4, [x1]
    let stxp = asm::exclusive(3, false, false, true, 6, false, 8, 1, 7); // stxp w6, x7, x8, [x1]
    let mut m = Machine::new(&[ldxp, stxp]);
    m.cpu.x[1] = DATA;
    m.cpu.x[7] = 0xAA;
    m.cpu.x[8] = 0xBB;
    m.bus.poke(DATA, 8, 0x11);
    m.bus.poke(DATA + 8, 8, 0x22);
    m.run(2).unwrap();
    assert_eq!((m.x(3), m.x(4), m.x(6)), (0x11, 0x22, 0));
    assert_eq!((m.bus.peek(DATA, 8), m.bus.peek(DATA + 8, 8)), (0xAA, 0xBB));
    // 32-bit pair
    let ldxp = asm::exclusive(2, false, true, true, 31, true, 4, 1, 3); // ldaxp w3, w4, [x1]
    let mut m = Machine::new(&[ldxp]);
    m.cpu.x[1] = DATA;
    m.bus.poke(DATA, 4, 0xDEAD_BEEF);
    m.bus.poke(DATA + 4, 4, 0xCAFE_F00D);
    m.run(1).unwrap();
    assert_eq!((m.x(3), m.x(4)), (0xDEAD_BEEF, 0xCAFE_F00D));
}

#[test]
fn acquire_release_accesses_and_alignment() {
    let ldar = asm::exclusive(3, true, true, false, 31, true, 31, 1, 0);
    let stlr = asm::exclusive(2, true, false, false, 31, true, 31, 1, 5);
    let mut m = Machine::new(&[stlr, ldar]);
    m.cpu.x[1] = DATA;
    m.cpu.x[5] = 0x1_2345_6789;
    m.bus.poke(DATA + 4, 4, 0);
    m.run(2).unwrap();
    assert_eq!(m.bus.peek(DATA, 4), 0x2345_6789, "stlr w stores 32 bits");
    assert_eq!(m.x(0) & 0xFFFF_FFFF, 0x2345_6789);

    // unaligned exclusive and acquire/release accesses fault; plain unaligned accesses do not
    for word in [ldxr(3, 1, 0), stxr(2, 2, 1, 0), ldar, stlr] {
        let mut m = Machine::new(&[word]);
        m.cpu.x[1] = DATA + 1;
        match m.run(1) {
            Err(Trap::Fault(f)) => assert_eq!(f.kind, FaultKind::Alignment, "{word:#010x}"),
            other => panic!("{word:#010x}: {other:?}"),
        }
        assert_eq!(m.cpu.pc, CODE, "faulting instruction does not advance");
    }
    let plain = asm::ldst_unsigned(3, 1, 0, 1, 0);
    let mut m = Machine::new(&[plain]);
    m.cpu.x[1] = DATA + 3;
    m.run(1).unwrap();
}

// ---------------------------------------------------------------------------------------------
// System registers, hints, cache maintenance
// ---------------------------------------------------------------------------------------------

#[test]
fn nzcv_fpcr_fpsr_and_thread_registers() {
    let mrs_nzcv = asm::sysreg(true, 3, 3, 4, 2, 0, 0);
    let msr_nzcv = asm::sysreg(false, 3, 3, 4, 2, 0, 1);
    let m = one(mrs_nzcv, |c| c.set_nzcv4(N | C));
    assert_eq!(m.x(0), 0xA000_0000);
    let m = one(msr_nzcv, |c| c.x[1] = 0xFFFF_FFFF_5FFF_FFFF);
    assert_eq!(m.flags(), Z | V, "only bits 31:28 are written");

    let msr_fpcr = asm::sysreg(false, 3, 3, 4, 4, 0, 1);
    let mrs_fpcr = asm::sysreg(true, 3, 3, 4, 4, 0, 0);
    let mut m = Machine::new(&[msr_fpcr, mrs_fpcr]);
    m.cpu.x[1] = u64::MAX;
    m.run(2).unwrap();
    assert_eq!(m.x(0), u64::from(FPCR_MASK));
    let msr_fpsr = asm::sysreg(false, 3, 3, 4, 4, 1, 1);
    let mrs_fpsr = asm::sysreg(true, 3, 3, 4, 4, 1, 0);
    let mut m = Machine::new(&[msr_fpsr, mrs_fpsr]);
    m.cpu.x[1] = u64::MAX;
    m.run(2).unwrap();
    assert_eq!(m.x(0), u64::from(FPSR_MASK));

    let msr_tpidr = asm::sysreg(false, 3, 3, 13, 0, 2, 1);
    let mrs_tpidr = asm::sysreg(true, 3, 3, 13, 0, 2, 0);
    let mrs_tpidrro = asm::sysreg(true, 3, 3, 13, 0, 3, 2);
    let mut m = Machine::new(&[msr_tpidr, mrs_tpidr, mrs_tpidrro]);
    m.cpu.x[1] = 0x1234_5678;
    m.cpu.tpidrro_el0 = 0xBEEF_0000;
    m.run(3).unwrap();
    assert_eq!(
        (m.x(0), m.x(2), m.cpu.tpidr_el0),
        (0x1234_5678, 0xBEEF_0000, 0x1234_5678)
    );
}

#[test]
fn counter_and_cache_identification_registers() {
    let mrs = |op2: u8, crm: u8, crn: u8, rt: u8| asm::sysreg(true, 3, 3, crn, crm, op2, rt);
    let mut m = Machine::new(&[
        mrs(0, 0, 14, 0),
        mrs(1, 0, 14, 1),
        mrs(1, 0, 14, 2),
        mrs(2, 0, 14, 3),
        mrs(1, 0, 0, 4),
        mrs(7, 0, 0, 5),
    ]);
    m.run(6).unwrap();
    assert_eq!(m.x(0), 19_200_000, "CNTFRQ_EL0 (docs/ASSUMPTIONS.md A-001)");
    assert_eq!(m.x(1), 1, "first counter read");
    assert_eq!(m.x(2), 2, "the counter advances between reads");
    assert_eq!(m.x(3), 3);
    assert_eq!(m.x(4), 0x8444_C004);
    assert_eq!(m.x(5), 4);
}

#[test]
fn unavailable_system_registers_stop_the_run() {
    // MIDR_EL1 is not readable from EL0 and is not invented
    let midr = asm::sysreg(true, 3, 0, 0, 0, 0, 0);
    let mut m = Machine::new(&[midr]);
    assert_eq!(
        m.run(1),
        Err(Trap::SysReg {
            reg: SysReg::Other(3, 0, 0, 0, 0),
            write: false
        })
    );
    assert_eq!(m.cpu.pc, CODE);
    // TPIDRRO_EL0 is read-only at EL0, CNTFRQ_EL0 too
    let msr_ro = asm::sysreg(false, 3, 3, 13, 0, 3, 1);
    let mut m = Machine::new(&[msr_ro]);
    assert_eq!(
        m.run(1),
        Err(Trap::SysReg {
            reg: SysReg::TpidrroEl0,
            write: true
        })
    );
    let msr_freq = asm::sysreg(false, 3, 3, 14, 0, 0, 1);
    let mut m = Machine::new(&[msr_freq]);
    assert_eq!(
        m.run(1),
        Err(Trap::SysReg {
            reg: SysReg::CntfrqEl0,
            write: true
        })
    );
    // an unimplemented SYS operation
    let sys = asm::sys(false, 0, 8, 7, 0, 0); // tlbi vmalle1is
    let mut m = Machine::new(&[sys]);
    assert!(matches!(m.run(1), Err(Trap::SysInst { .. })));
}

#[test]
fn hints_barriers_and_cache_maintenance_are_no_ops() {
    for w in [
        asm::nop(),
        asm::hint(1),
        asm::hint(2),
        asm::hint(3),
        asm::hint(4),
        asm::hint(5),
        asm::hint(25), // paciasp
        asm::hint(127),
        asm::dsb(15),
        asm::dmb(11),
        asm::isb(15),
        asm::clrex(0),
        asm::sys(false, 3, 7, 10, 1, 2), // dc cvac, x2
        asm::sys(false, 3, 7, 11, 1, 2), // dc cvau
        asm::sys(false, 3, 7, 14, 1, 2), // dc civac
        asm::sys(false, 3, 7, 5, 1, 2),  // ic ivau
    ] {
        let m = one(w, |c| {
            c.x[2] = 0xDEAD_0000_0000; // never dereferenced
            c.x[30] = 77;
        });
        assert_eq!(m.cpu.pc, CODE + 4, "{w:#010x}");
        assert_eq!(m.x(30), 77);
    }
}

#[test]
fn dc_zva_clears_one_aligned_block() {
    let w = asm::sys(false, 3, 7, 4, 1, 0); // dc zva, x0
    let mut m = Machine::new(&[w]);
    for i in 0..0x100u64 {
        m.bus.poke(DATA + i, 1, 0xFF);
    }
    m.cpu.x[0] = DATA + 0x47;
    m.run(1).unwrap();
    for i in 0..0x100u64 {
        let expect = if (0x40..0x80).contains(&i) { 0 } else { 0xFF };
        assert_eq!(m.bus.peek(DATA + i, 1), expect, "offset {i:#x}");
    }
}

// ---------------------------------------------------------------------------------------------
// Traps
// ---------------------------------------------------------------------------------------------

#[test]
fn svc_and_brk_traps() {
    let mut m = Machine::new(&[asm::svc(0x27), asm::brk(0x1234)]);
    assert_eq!(m.run(1), Err(Trap::Svc(0x27)));
    assert_eq!(m.cpu.pc, CODE + 4, "svc resumes after itself");
    assert_eq!(m.run(1), Err(Trap::Brk(0x1234)));
    assert_eq!(m.cpu.pc, CODE + 4, "brk stays on itself");
}

#[test]
fn undefined_and_unsupported_instructions_stop_the_run() {
    let mut m = Machine::new(&[0x0000_0000]);
    assert_eq!(m.run(1), Err(Trap::Undefined(0)));
    assert_eq!(m.cpu.pc, CODE);
    // cas x0, x1, [x2]: an Armv8.1 atomic, valid but not on the modelled core and not implemented.
    let cas = 0xC8A0_7C41;
    let mut m = Machine::new(&[cas]);
    assert!(matches!(m.run(1), Err(Trap::Unsupported { word, .. }) if word == cas));
    assert_eq!(m.cpu.pc, CODE);
}

#[test]
fn memory_faults_are_precise() {
    // ldp x0, x1, [x2] where the second half is unmapped: nothing is written
    let end = BASE + 0x4000;
    let mut m = Machine::new(&[asm::ldst_pair(2, 2, true, 0, 1, 2, 0)]);
    m.cpu.x[0] = 0x55;
    m.cpu.x[1] = 0x66;
    m.cpu.x[2] = end - 8;
    match m.run(1) {
        Err(Trap::Fault(f)) => {
            assert_eq!((f.addr, f.write, f.kind), (end, false, FaultKind::Unmapped))
        }
        other => panic!("{other:?}"),
    }
    assert_eq!((m.x(0), m.x(1), m.cpu.pc), (0x55, 0x66, CODE));
    // pre-index writeback does not happen when the access faults
    let mut m = Machine::new(&[asm::ldst_imm9(3, 1, 8, 3, 3, 4)]);
    m.cpu.x[3] = 0x10;
    assert!(matches!(m.run(1), Err(Trap::Fault(_))));
    assert_eq!(m.x(3), 0x10);
    // instruction fetch from unmapped memory
    let mut m = Machine::new(&[]);
    m.cpu.pc = 0x1000;
    assert!(matches!(m.run(1), Err(Trap::Fault(f)) if f.addr == 0x1000));
    // a store past the end reports a write fault
    let mut m = Machine::new(&[asm::ldst_unsigned(3, 0, 0, 3, 4)]);
    m.cpu.x[3] = end;
    match m.run(1) {
        Err(Trap::Fault(f)) => assert!(f.write),
        other => panic!("{other:?}"),
    }
}

#[test]
fn zero_register_is_never_written() {
    // every destination form with Rd = 31 leaves all of x0..x30 alone
    let words = [
        asm::mov_wide(true, 2, 0, 5, 31),
        asm::add_sub_shifted(true, false, true, 0, 1, 0, 2, 31),
        asm::logical_shifted(true, 0, 0, false, 1, 0, 2, 31),
        asm::bitfield(true, 2, true, 1, 2, 1, 31),
        asm::cond_sel(true, 0, 1, 0, 2, 31),
        asm::dp1(true, 0, 1, 31),
        asm::dp2(true, 0b000010, 1, 2, 31),
        asm::dp3(true, 0, false, 1, 2, 3, 31),
        asm::ldst_unsigned(3, 1, 0, 6, 31), // ldr xzr, [x6]
    ];
    for w in words {
        let mut m = Machine::new(&[w]);
        for (i, r) in m.cpu.x.iter_mut().enumerate() {
            *r = 0x1000 + i as u64;
        }
        m.cpu.x[6] = DATA;
        let before = m.cpu.clone();
        m.run(1).unwrap();
        let mut after = m.cpu.clone();
        after.pc = before.pc;
        after.nzcv = before.nzcv;
        assert_eq!(after, before, "{w:#010x}");
    }
}

#[test]
fn bus_trait_default_monitor_semantics() {
    let mut bus = FlatBus::new(BASE, 0x100);
    let mut cpu = CpuState::default();
    bus.write(BASE, 8, 5).unwrap();
    assert_eq!(bus.load_exclusive(&mut cpu, BASE, 8).unwrap(), 5);
    assert!(bus.store_exclusive(&mut cpu, BASE, 8, 6).unwrap());
    assert!(
        !bus.store_exclusive(&mut cpu, BASE, 8, 7).unwrap(),
        "monitor closes after a store"
    );
    assert_eq!(bus.read(BASE, 8).unwrap(), 6);
    bus.load_exclusive(&mut cpu, BASE, 8).unwrap();
    bus.clear_exclusive(&mut cpu);
    assert!(!bus.store_exclusive(&mut cpu, BASE, 8, 9).unwrap());
    assert_eq!(bus.system_register(SysReg::Other(3, 0, 0, 0, 0)), None);
}
