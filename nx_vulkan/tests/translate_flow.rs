//! Control flow against the GPU: loops, divergence and reconvergence, and early exits.

mod common;

use common::{random_states, run, run_with, vulkan, Extras, State};
use nx_shader::asm::{bra, brk, exit, mov32i, pbk, ssy, sync, Enc, NOT_P0, P0};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;

/// The byte address of instruction `i` of a program laid out by the assembler (three
/// instructions per bundle after a scheduling word).
fn at(i: u32) -> u32 {
    8 * (1 + i + i / 3)
}

const P1: Guard = Guard {
    index: 1,
    negated: false,
};

#[test]
fn a_backward_branch_makes_a_loop() {
    let Some(vk) = vulkan() else { return };
    // R3 = 1 + 2 + ... + R1, counting R4 down from R1.
    let program = [
        mov32i(3, 0),
        Enc::op(Op::MovReg).rd(4).rb(1).field(39, 4, 0xF).build(),
        // 2: loop head: P0 = R4 > 0 (signed).
        Enc::op(Op::IsetpReg)
            .ra(4)
            .rb(RZ)
            .field(0, 3, 7)
            .field(3, 3, 0)
            .field(49, 3, 4)
            .bit(48, true)
            .pred(39, 7, false)
            .build(),
        // 3: leave when not.
        bra(at(3), at(9), NOT_P0),
        Enc::op(Op::IaddReg).rd(3).ra(3).rb(4).build(),
        Enc::op(Op::Iadd32iImm32)
            .rd(4)
            .ra(4)
            .imm32(u32::MAX)
            .build(),
        bra(at(6), at(2), Guard::ALWAYS),
        // 7, 8: skipped.
        mov32i(5, 0xDEAD),
        mov32i(5, 0xBEEF),
        // 9: after the loop.
        mov32i(6, 0xBEEF),
        exit(Guard::ALWAYS),
    ];
    let inputs: Vec<State> = (0..40u32)
        .map(|i| {
            let mut s = State::default();
            s.regs[1] = i;
            s.regs[5] = 0x1111;
            s
        })
        .collect();
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let n = inp.regs[1];
        assert_eq!(got.regs[3], n * (n + 1) / 2, "sum 1..={n}");
        assert_eq!(got.regs[4], 0, "the counter, n = {n}");
        assert_eq!(got.regs[5], 0x1111, "skipped code, n = {n}");
        assert_eq!(got.regs[6], 0xBEEF, "the code after the loop, n = {n}");
    }
}

#[test]
fn ssy_and_sync_reconverge_after_a_divergent_branch() {
    let Some(vk) = vulkan() else { return };
    let program = [
        ssy(at(0), at(6)),
        bra(at(1), at(4), P0),
        // The path without P0.
        mov32i(3, 1),
        sync(Guard::ALWAYS),
        // The path with P0.
        mov32i(3, 2),
        sync(Guard::ALWAYS),
        // The join.
        mov32i(4, 0x77),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(64, 5);
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        assert_eq!(got.regs[3], if inp.preds & 1 != 0 { 2 } else { 1 });
        assert_eq!(got.regs[4], 0x77, "everyone reaches the join");
    }
}

#[test]
fn pbk_and_brk_leave_a_loop_from_inside() {
    let Some(vk) = vulkan() else { return };
    // R3 grows by three until it passes R2; BRK leaves through the PBK target.
    let program = [
        pbk(at(0), at(7)),
        mov32i(3, 0),
        // 2: loop head.
        Enc::op(Op::Iadd32iImm32).rd(3).ra(3).imm32(3).build(),
        Enc::op(Op::IsetpReg)
            .ra(3)
            .rb(2)
            .field(0, 3, 7)
            .field(3, 3, 1)
            .field(49, 3, 4)
            .bit(48, false)
            .pred(39, 7, false)
            .build(),
        brk(P1),
        bra(at(5), at(2), Guard::ALWAYS),
        mov32i(5, 0xDEAD),
        // 7: where BRK lands.
        mov32i(6, 0x99),
        exit(Guard::ALWAYS),
    ];
    let inputs: Vec<State> = (0..60u32)
        .map(|i| {
            let mut s = State::default();
            s.regs[2] = i;
            s
        })
        .collect();
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let limit = inp.regs[2];
        assert_eq!(got.regs[3], (limit / 3 + 1) * 3, "limit {limit}");
        assert_eq!(got.regs[5], 0, "skipped code");
        assert_eq!(got.regs[6], 0x99, "after the break");
    }
}

#[test]
fn a_guarded_exit_ends_only_the_invocations_it_holds_for() {
    let Some(vk) = vulkan() else { return };
    let program = [mov32i(3, 1), exit(P0), mov32i(3, 2), exit(Guard::ALWAYS)];
    let inputs = random_states(64, 6);
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        assert_eq!(got.regs[3], if inp.preds & 1 != 0 { 1 } else { 2 });
    }
}

#[test]
fn the_loop_free_form_agrees_with_the_loop_on_every_state() {
    let Some(vk) = vulkan() else { return };
    // Jumps that only go forward, one on a predicate from the input and one on a comparison, a
    // join they both reach, and an exit that holds for some invocations only.
    let program = [
        // 0: P1 = R1 > R2 (signed).
        Enc::op(Op::IsetpReg)
            .ra(1)
            .rb(2)
            .field(0, 3, 7)
            .field(3, 3, 1)
            .field(49, 3, 4)
            .bit(48, true)
            .pred(39, 7, false)
            .build(),
        // 1: over the next two when P0.
        bra(at(1), at(4), P0),
        Enc::op(Op::IaddReg).rd(3).ra(1).rb(2).build(),
        // 3: over the next one when P1.
        bra(at(3), at(5), P1),
        mov32i(4, 0xAAAA),
        // 5: the join.
        Enc::op(Op::IaddReg).rd(5).ra(3).rb(4).build(),
        exit(NOT_P0),
        mov32i(6, 7),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(256, 11);
    let loop_free = run_with(&vk, &program, &inputs, &[], Extras::default()).states;
    let looped = run_with(
        &vk,
        &program,
        &inputs,
        &[],
        Extras {
            force_loop: true,
            ..Extras::default()
        },
    )
    .states;
    for (i, (a, b)) in loop_free.iter().zip(&looped).enumerate() {
        assert_eq!(a.regs, b.regs, "registers of invocation {i}");
        assert_eq!(a.preds, b.preds, "predicates of invocation {i}");
    }
    // And the program did something: the exit held for the invocations without P0.
    let sevens = loop_free.iter().filter(|s| s.regs[6] == 7).count();
    assert!(sevens > 0 && sevens < inputs.len());
}
