//! Whole-program differential tests: control flow, calls, tables and the slow path.

use super::discover::{discover, Options};
use super::lift_all;
use super::tests::{
    compare, run_interpreter_bounded, run_lifted, Gen, Image, CODE, DATA, DATA_SIZE, EXIT,
};
use nx_runtime::a64::inst::Inst;
use nx_runtime::a64::{decode, CpuState};
use nx_runtime::testing::asm;
use nx_runtime::testing::prog::Prog;

/// Run a program from several random states and demand identical results.
fn same_as_interpreter(words: &[u32], seed: u64, states: usize) -> Vec<u64> {
    let mut g = Gen::new(seed);
    let mut unknown = Vec::new();
    for i in 0..states {
        let mut cpu = g.state();
        if i == 0 {
            cpu.x[0] = 0;
        }
        if i == 1 {
            cpu.x[0] = 1;
        }
        let data = g.data();
        let a = run_interpreter_bounded(words, &cpu, &data, 20_000);
        let (b, u) = run_lifted(words, &cpu, &data);
        if let Some(why) = compare(&a, &b) {
            panic!(
                "state {i}: {why}\n  start x0={:#x} x1={:#x} sp={:#x}\n  program {words:08x?}",
                cpu.x[0], cpu.x[1], cpu.sp
            );
        }
        assert!(
            !matches!(a.outcome, super::eval::Outcome::Halted),
            "state {i}: the program did not finish"
        );
        unknown.extend(u);
    }
    unknown
}

fn lift(words: &[u32]) -> super::discover::Discovery {
    discover(
        &Image {
            words: words.to_vec(),
        },
        &[CODE],
        &Options::default(),
    )
}

fn mov(rd: u8, rm: u8) -> u32 {
    asm::logical_shifted(true, 1, 0, false, rm, 0, 31, rd)
}

fn stp_pre(rt: u8, rt2: u8) -> u32 {
    asm::ldst_pair(2, 3, false, -2, rt2, 31, rt) // stp rt, rt2, [sp, #-16]!
}

fn ldp_post(rt: u8, rt2: u8) -> u32 {
    asm::ldst_pair(2, 1, true, 2, rt2, 31, rt) // ldp rt, rt2, [sp], #16
}

#[test]
fn a_loop_sums_the_first_n_numbers() {
    let mut p = Prog::new(CODE);
    p.emit(asm::logical_imm(true, 0, true, 0, 5, 0, 0)); // and x0, x0, #63
    p.emit(asm::movz(true, 1, 0, 0));
    p.label("loop").cbz(true, 0, "done");
    p.emit(asm::add_sub_shifted(true, false, false, 0, 0, 0, 1, 1)); // add x1, x1, x0
    p.emit(asm::sub_imm(true, 0, 0, 1));
    p.b("loop");
    p.label("done").emit(mov(0, 1));
    p.emit(asm::ret(30));
    let words = p.finish();
    let d = lift(&words);
    assert_eq!(d.funcs.len(), 1);
    same_as_interpreter(&words, 1, 40);
}

fn fib_program() -> Vec<u32> {
    let mut p = Prog::new(CODE);
    p.emit(asm::logical_imm(true, 0, true, 0, 3, 0, 0)); // and x0, x0, #15
    p.emit(stp_pre(29, 30));
    p.bl("fib");
    p.emit(ldp_post(29, 30));
    p.emit(asm::ret(30));
    p.label("fib").emit(asm::cmp_imm(true, 0, 2));
    p.b_cond(2, "rec"); // b.hs
    p.emit(asm::ret(30));
    p.label("rec")
        .emit(asm::ldst_pair(2, 3, false, -4, 30, 31, 29)); // stp x29, x30, [sp, #-32]!
    p.emit(asm::ldst_pair(2, 2, false, 2, 20, 31, 19)); // stp x19, x20, [sp, #16]
    p.emit(mov(19, 0));
    p.emit(asm::sub_imm(true, 0, 19, 1));
    p.bl("fib");
    p.emit(mov(20, 0));
    p.emit(asm::sub_imm(true, 0, 19, 2));
    p.bl("fib");
    p.emit(asm::add_sub_shifted(true, false, false, 0, 20, 0, 0, 0)); // add x0, x0, x20
    p.emit(asm::ldst_pair(2, 2, true, 2, 20, 31, 19)); // ldp x19, x20, [sp, #16]
    p.emit(asm::ldst_pair(2, 1, true, 4, 30, 31, 29)); // ldp x29, x30, [sp], #32
    p.emit(asm::ret(30));
    p.finish()
}

#[test]
fn recursion_uses_native_calls_and_the_stack() {
    let words = fib_program();
    let d = lift(&words);
    assert_eq!(d.funcs.len(), 2, "entry and fib");
    let unknown = same_as_interpreter(&words, 2, 30);
    assert!(unknown.is_empty(), "no slow path needed: {unknown:x?}");
}

#[test]
fn tail_calls_and_loops_to_the_entry() {
    let mut p = Prog::new(CODE);
    p.emit(asm::logical_imm(true, 0, true, 0, 2, 0, 0)); // and x0, x0, #7
    p.emit(mov(19, 30));
    p.bl("wrapper");
    p.emit(asm::add_imm(true, 0, 0, 1));
    p.bl("target"); // makes `target` a function of its own
    p.bl("counter");
    p.emit(mov(30, 19));
    p.emit(asm::ret(30));
    p.label("wrapper").emit(asm::add_imm(true, 0, 0, 10));
    p.b("target"); // a tail call
    p.label("target")
        .emit(asm::add_sub_shifted(true, false, false, 0, 0, 0, 0, 0)); // add x0, x0, x0
    p.emit(asm::ret(30));
    // A function that loops back to its own entry.
    p.label("counter").emit(asm::sub_imm(true, 0, 0, 1));
    p.cbnz(true, 0, "counter");
    p.emit(asm::ret(30));
    let words = p.finish();
    let d = lift(&words);
    assert!(
        d.funcs.len() >= 4,
        "{:x?}",
        d.funcs.keys().collect::<Vec<_>>()
    );
    let unknown = same_as_interpreter(&words, 3, 30);
    assert!(unknown.is_empty(), "{unknown:x?}");
}

#[test]
fn conditional_tail_call_and_shared_flags() {
    let mut p = Prog::new(CODE);
    p.emit(mov(19, 30));
    p.bl("f");
    p.emit(mov(1, 0));
    p.emit(asm::movz(true, 0, 0, 0));
    p.bl("f");
    p.bl("g");
    p.emit(mov(30, 19));
    p.emit(asm::ret(30));
    p.label("f").cbz(true, 0, "g"); // a conditional tail call
    p.emit(asm::add_imm(true, 0, 0, 1));
    p.emit(asm::ret(30));
    p.label("g").emit(asm::movz(true, 0, 50, 0));
    p.emit(asm::ret(30));
    let words = p.finish();
    same_as_interpreter(&words, 4, 20);

    // Flags set in one block and used after a join: a phi per flag.
    let mut p = Prog::new(CODE);
    p.emit(asm::add_sub_shifted(true, true, true, 0, 1, 0, 0, 31)); // cmp x0, x1
    p.cbz(true, 2, "join");
    p.emit(asm::add_imm(true, 3, 3, 1)); // does not touch the flags
    p.label("join").emit(asm::cond_sel(true, 0, 1, 11, 0, 4)); // csel x4, x0, x1, lt
    p.emit(asm::cond_sel(true, 1, 3, 8, 3, 5)); // csinc x5, x3, x3, hi
    p.emit(asm::ret(30));
    same_as_interpreter(&p.finish(), 5, 60);
}

#[test]
fn indirect_calls_go_through_the_function_table() {
    let mut p = Prog::new(CODE);
    p.emit(mov(19, 30));
    p.adr(9, "a");
    p.emit(asm::blr(9));
    p.adr(9, "b");
    p.emit(asm::blr(9));
    p.emit(mov(30, 19));
    p.emit(asm::ret(30));
    p.label("a").emit(asm::add_imm(true, 0, 0, 3));
    p.emit(asm::ret(30));
    p.label("b")
        .emit(asm::add_sub_shifted(true, false, false, 0, 0, 0, 0, 0));
    p.emit(asm::ret(30));
    let words = p.finish();
    // `a` and `b` are only reached through `adr`, so the analysis does not know them: the slow
    // path runs them, and the result is the same.
    let unknown = same_as_interpreter(&words, 6, 10);
    assert!(!unknown.is_empty());
    // Handing the analysis the entries (as a hints file would) removes the slow path.
    let mut g = Gen::new(6);
    let cpu = g.state();
    let data = g.data();
    let image = Image {
        words: words.clone(),
    };
    let a_addr = CODE + 4 * 7;
    let b_addr = CODE + 4 * 9;
    let d = discover(&image, &[CODE, a_addr, b_addr], &Options::default());
    let funcs = lift_all(&d);
    let mut cpu2 = cpu.clone();
    let mut bus = nx_runtime::testing::FlatBus::new(CODE, super::tests::REGION);
    bus.load_words(CODE, &words);
    bus.mem[0x1_0000..0x1_0000 + data.len()].copy_from_slice(&data);
    let mut ev = super::eval::ProgramEval::new(&funcs, EXIT);
    let outcome = ev.run(CODE, &mut cpu2, &mut bus);
    assert_eq!(outcome, super::eval::Outcome::Exited);
    assert!(ev.unknown.is_empty(), "{:x?}", ev.unknown);
    let reference = run_interpreter_bounded(&words, &cpu, &data, 1000);
    assert_eq!(reference.cpu.x[0], cpu2.x[0]);
}

fn jump_table_program(entries: u32) -> Vec<u32> {
    let mut p = Prog::new(CODE);
    p.emit(asm::logical_imm(false, 0, false, 0, 3, 0, 0)); // and w0, w0, #15
    p.emit(asm::cmp_imm(false, 0, entries - 1));
    p.b_cond(8, "default"); // b.hi
    p.adr(9, "table");
    p.emit(asm::ldst_reg(2, 2, 0, 3, true, 9, 10)); // ldrsw x10, [x9, x0, lsl #2]
    p.emit(asm::add_sub_shifted(true, false, false, 0, 10, 0, 9, 9)); // add x9, x9, x10
    p.emit(asm::br(9));
    for i in 0..entries {
        p.label(&format!("case{i}"))
            .emit(asm::movz(true, 0, 100 + i, 0));
        p.emit(asm::ret(30));
    }
    p.label("default").emit(asm::movz(true, 0, 999, 0));
    p.emit(asm::ret(30));
    p.label("table");
    for i in 0..entries {
        p.rel32(&format!("case{i}"), "table");
    }
    p.finish()
}

#[test]
fn a_jump_table_is_resolved_and_run_natively() {
    let words = jump_table_program(8);
    let d = lift(&words);
    assert_eq!(d.report.jump_tables.len(), 1, "{:?}", d.report);
    assert_eq!(d.report.jump_tables[0].targets.len(), 8);
    assert!(d.report.unresolved_branches.is_empty());
    let unknown = same_as_interpreter(&words, 7, 60);
    assert!(
        unknown.is_empty(),
        "every case is a block of the function: {unknown:x?}"
    );
}

#[test]
fn a_jump_table_with_byte_entries_scaled_from_a_base() {
    // The GCC spelling: byte offsets from a label, scaled by 4.
    let mut p = Prog::new(CODE);
    p.emit(asm::logical_imm(false, 0, false, 0, 2, 0, 0)); // and w0, w0, #7
    p.emit(asm::cmp_imm(false, 0, 4));
    p.b_cond(8, "default");
    p.adr(9, "table");
    p.emit(asm::ldst_reg(0, 1, 0, 2, false, 9, 10)); // ldrb w10, [x9, w0, uxtw]
    p.adr(11, "base");
    p.emit(asm::add_sub_ext(true, false, false, 10, 3, 2, 11, 9)); // add x9, x11, x10, uxtx #2
    p.emit(asm::br(9));
    p.label("base");
    for i in 0..5u32 {
        p.label(&format!("c{i}"))
            .emit(asm::movz(true, 0, 10 + i, 0));
        p.emit(asm::ret(30));
    }
    p.label("default").emit(asm::movz(true, 0, 77, 0));
    p.emit(asm::ret(30));
    // Four-byte slots after the code, one byte per entry.
    p.label("table");
    let mut words = p.finish();
    // Entries are (case - base) / 4: c0 is at base + 0, c1 at +2 words, ...
    let entries: [u8; 5] = [0, 2, 4, 6, 8];
    let mut word = 0u32;
    for (i, e) in entries.iter().enumerate() {
        word |= u32::from(*e) << (8 * (i % 4));
        if i % 4 == 3 {
            words.push(word);
            word = 0;
        }
    }
    words.push(word);
    let d = lift(&words);
    assert_eq!(d.report.jump_tables.len(), 1, "{:?}", d.report);
    let unknown = same_as_interpreter(&words, 8, 60);
    assert!(unknown.is_empty(), "{unknown:x?}");
}

#[test]
fn a_branch_nothing_explains_runs_in_the_interpreter() {
    let mut p = Prog::new(CODE);
    p.adr(16, "mid");
    p.emit(asm::br(16));
    p.emit(asm::movz(true, 0, 1234, 0)); // skipped
    p.label("mid").emit(asm::add_imm(true, 0, 0, 5));
    p.emit(asm::ret(30));
    let words = p.finish();
    let d = lift(&words);
    assert_eq!(d.report.unresolved_branches.len(), 1);
    let unknown = same_as_interpreter(&words, 9, 10);
    assert!(unknown.contains(&(CODE + 12)), "{unknown:x?}");
}

#[test]
fn a_return_to_a_new_address_leaves_the_caller() {
    let mut p = Prog::new(CODE);
    p.emit(mov(19, 30));
    p.bl("f");
    p.emit(asm::movz(true, 0, 1, 0));
    p.emit(mov(30, 19));
    p.emit(asm::ret(30));
    p.label("landing").emit(mov(30, 19));
    p.emit(asm::movz(true, 0, 2, 0));
    p.emit(asm::ret(30));
    p.label("f").adr(30, "landing");
    p.emit(asm::ret(30));
    let words = p.finish();
    let unknown = same_as_interpreter(&words, 10, 5);
    assert!(!unknown.is_empty());
}

#[test]
fn supervisor_calls_and_unsupported_instructions() {
    let mut p = Prog::new(CODE);
    p.emit(asm::movz(true, 0, 5, 0));
    p.emit(asm::svc(1)); // the test handler adds 100 to x0
    p.emit(asm::add_imm(true, 0, 0, 1));
    p.emit(asm::svc(2)); // the test handler refuses
    p.emit(asm::ret(30));
    same_as_interpreter(&p.finish(), 11, 5);

    let mut p = Prog::new(CODE);
    p.emit(asm::add_imm(true, 0, 0, 1));
    p.emit(0x1E62_2820); // fadd d0, d1, d2: not implemented until M5
    p.emit(asm::add_imm(true, 0, 0, 2));
    p.emit(asm::ret(30));
    let words = p.finish();
    let d = lift(&words);
    assert_eq!(d.report.fallbacks.len(), 1);
    same_as_interpreter(&words, 12, 5);
}

#[test]
fn division_edge_cases() {
    // sdiv/udiv in both widths over the awkward operand pairs.
    let pairs: [(u64, u64); 8] = [
        (7, 0),
        (0, 0),
        (u64::MAX, 0),
        (1 << 63, u64::MAX),
        (0x8000_0000, 0xFFFF_FFFF),
        (0x8000_0000, 1),
        (u64::MAX, u64::MAX),
        (100, 7),
    ];
    for sf in [false, true] {
        for opcode in [2u32, 3] {
            let words = vec![asm::dp2(sf, opcode, 1, 0, 2), asm::svc(0)];
            for (a, b) in pairs {
                let mut cpu = CpuState::default();
                cpu.x[0] = a;
                cpu.x[1] = b;
                cpu.sp = DATA + 0x2000;
                let data = vec![0u8; DATA_SIZE];
                let x = run_interpreter_bounded(&words, &cpu, &data, 100);
                let (y, _) = run_lifted(&words, &cpu, &data);
                assert_eq!(compare(&x, &y), None, "sf={sf} op={opcode} {a:#x} {b:#x}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Random structured programs
// ---------------------------------------------------------------------------------------------

/// A random straight-line instruction over x0 to x7, memory through x8 (a fixed data pointer),
/// never touching x9 (the loop counter) or the stack pointer.
fn random_op(g: &mut Gen) -> u32 {
    let r = |g: &mut Gen| g.below(8) as u8;
    let sf = g.below(2) == 1;
    match g.below(16) {
        0 => asm::add_sub_imm(
            sf,
            g.below(2) == 1,
            g.below(2) == 1,
            false,
            g.below(4096) as u32,
            r(g),
            r(g),
        ),
        1 => asm::logical_shifted(
            sf,
            g.below(4) as u32,
            g.below(4) as u32,
            g.below(2) == 1,
            r(g),
            g.below(if sf { 64 } else { 32 }) as u8,
            r(g),
            r(g),
        ),
        2 => asm::add_sub_shifted(
            sf,
            g.below(2) == 1,
            g.below(2) == 1,
            g.below(3) as u32,
            r(g),
            g.below(if sf { 64 } else { 32 }) as u8,
            r(g),
            r(g),
        ),
        3 => asm::add_sub_carry(sf, g.below(2) == 1, g.below(2) == 1, r(g), r(g), r(g)),
        4 => asm::cond_sel(sf, g.below(4) as u32, r(g), g.below(16) as u8, r(g), r(g)),
        5 => asm::cond_cmp(
            sf,
            g.below(2) == 1,
            g.below(2) == 1,
            g.below(32) as u8,
            g.below(16) as u8,
            r(g),
            g.below(16) as u8,
        ),
        6 => asm::dp3(
            sf,
            [0u32, 0, 1, 5, 6][g.below(5) as usize],
            g.below(2) == 1,
            r(g),
            r(g),
            r(g),
            r(g),
        ),
        7 => asm::dp2(
            sf,
            [2u32, 3, 8, 9, 10, 11][g.below(6) as usize],
            r(g),
            r(g),
            r(g),
        ),
        8 => {
            let (immr, imms) = (
                g.below(if sf { 64 } else { 32 }) as u8,
                g.below(if sf { 64 } else { 32 }) as u8,
            );
            asm::bitfield(
                sf,
                [0u32, 1, 2][g.below(3) as usize],
                sf,
                immr,
                imms,
                r(g),
                r(g),
            )
        }
        9 => asm::mov_wide(
            sf,
            [0u32, 2, 3][g.below(3) as usize],
            g.below(if sf { 4 } else { 2 }) as u32,
            g.below(65536) as u32,
            r(g),
        ),
        10 => asm::dp1(sf, g.below(6) as u32, r(g), r(g)),
        11 => asm::extr(
            sf,
            sf,
            r(g),
            g.below(if sf { 64 } else { 32 }) as u8,
            r(g),
            r(g),
        ),
        12 => {
            // A store or load at a small offset from the data pointer.
            let size = g.below(4) as u32;
            let off = (g.below(64) as u32) << size;
            let opc = [0u32, 1][g.below(2) as usize];
            asm::ldst_unsigned(size, opc, off >> size, 8, r(g))
        }
        13 => asm::logical_imm(
            sf,
            g.below(4) as u32,
            false,
            0,
            g.below(sf as u64 + 1 + 31) as u8 & 31,
            r(g),
            r(g),
        ),
        14 => asm::add_sub_ext(
            sf,
            g.below(2) == 1,
            g.below(2) == 1,
            r(g),
            g.below(8) as u32,
            g.below(5) as u32,
            r(g),
            r(g),
        ),
        _ => asm::cond_sel(sf, 0, r(g), 0, r(g), r(g)),
    }
}

/// A function of forward branches, one counted loop and a call, as a program that returns.
fn random_program(g: &mut Gen) -> Vec<u32> {
    let blocks = 3 + g.below(7) as usize;
    let mut p = Prog::new(CODE);
    // x19 keeps the return address across the call; x8 points at the middle of the data region and
    // x9 counts the loop.
    p.emit(mov(19, 30));
    p.emit(asm::mov_wide(
        true,
        2,
        0,
        (DATA + 0x2000) as u32 & 0xFFFF,
        8,
    ));
    p.emit(asm::mov_wide(
        true,
        3,
        1,
        ((DATA + 0x2000) >> 16) as u32 & 0xFFFF,
        8,
    ));
    p.emit(asm::movz(true, 9, 1 + g.below(4) as u32, 0));
    let loop_block = g.below(blocks as u64) as usize;
    for b in 0..blocks {
        p.label(&format!("b{b}"));
        for _ in 0..1 + g.below(5) {
            let w = random_op(g);
            if matches!(decode(w), Inst::Unallocated(_) | Inst::Unsupported { .. }) {
                continue;
            }
            p.emit(w);
        }
        if g.below(6) == 0 {
            p.bl("leaf");
        }
        // A forward branch, taken or not, unless this is the last block.
        let ahead = |g: &mut Gen| {
            format!(
                "b{}",
                b + 1 + g.below((blocks - b - 1).max(1) as u64) as usize
            )
        };
        if b + 1 < blocks {
            match g.below(5) {
                0 => {
                    p.b_cond(g.below(15) as u8, &ahead(g));
                }
                1 => {
                    p.cbz(g.below(2) == 1, g.below(8) as u8, &ahead(g));
                }
                2 => {
                    let bit = g.below(64) as u8;
                    p.tbnz(g.below(8) as u8, bit, &ahead(g));
                }
                3 => {
                    p.b(&ahead(g));
                }
                _ => {}
            }
        }
        if b == blocks - 1 {
            // The loop: count x9 down and go back to the header.
            p.emit(asm::sub_imm(true, 9, 9, 1));
            p.cbnz(true, 9, &format!("b{loop_block}"));
        }
    }
    p.emit(mov(30, 19));
    p.emit(asm::ret(30));
    p.label("leaf").emit(asm::add_imm(true, 0, 0, 1));
    p.emit(asm::add_sub_shifted(true, true, true, 0, 1, 0, 0, 2)); // adds x2, x0, x1
    p.emit(asm::ret(30));
    p.finish()
}

#[test]
fn random_control_flow_matches_the_interpreter() {
    let mut g = Gen::new(0xC0FFEE);
    let programs: usize = std::env::var("RECOMP_LIFT_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    for n in 0..programs {
        let words = random_program(&mut g);
        for _ in 0..4 {
            let state = g.state();
            let data = g.data();
            let a = run_interpreter_bounded(&words, &state, &data, 50_000);
            let (b, _) = run_lifted(&words, &state, &data);
            if let Some(why) = compare(&a, &b) {
                panic!("program {n}: {why}\n  {words:08x?}");
            }
            assert_ne!(
                a.outcome,
                super::eval::Outcome::Halted,
                "program {n} did not finish"
            );
        }
    }
}
