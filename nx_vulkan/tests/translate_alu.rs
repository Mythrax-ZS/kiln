//! The translator against the GPU: programs of the harness mode run for many random inputs and the
//! results are compared with what plain Rust arithmetic gives. Skipped (loudly) without Vulkan or
//! without `spirv-val`.

mod common;

use common::{random_states, run, vulkan, Rng, State};
use nx_shader::asm::{bra, exit, Enc, P0};
use nx_shader::inst::Guard;
use nx_shader::ops::Op;

#[test]
fn integer_arithmetic_matches_rust() {
    let Some(vk) = vulkan() else { return };
    // R3 = R1 + R2; R4 = R1 - R2 (IADD with the second operand negated); R5 = (R1 << 3) + R2
    // (ISCADD); R6 = R1 & R2; R7 = R1 | 0x00FF00FF (LOP32I); R8 = ~R2 (LOP pass-b inverted).
    let program = [
        Enc::op(Op::IaddReg).rd(3).ra(1).rb(2).build(),
        Enc::op(Op::IaddReg).rd(4).ra(1).rb(2).bit(48, true).build(),
        Enc::op(Op::IscaddReg)
            .rd(5)
            .ra(1)
            .rb(2)
            .field(39, 5, 3)
            .build(),
        Enc::op(Op::LopReg)
            .rd(6)
            .ra(1)
            .rb(2)
            .field(41, 2, 0)
            .build(),
        Enc::op(Op::Lop32iImm32)
            .rd(7)
            .ra(1)
            .imm32(0x00FF_00FF)
            .field(53, 2, 1)
            .build(),
        Enc::op(Op::LopReg)
            .rd(8)
            .ra(1)
            .rb(2)
            .field(41, 2, 3)
            .bit(40, true)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(256, 7);
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let (a, b) = (inp.regs[1], inp.regs[2]);
        assert_eq!(got.regs[3], a.wrapping_add(b), "iadd, input {i}");
        assert_eq!(
            got.regs[4],
            a.wrapping_sub(b),
            "iadd with negation, input {i}"
        );
        assert_eq!(got.regs[5], (a << 3).wrapping_add(b), "iscadd, input {i}");
        assert_eq!(got.regs[6], a & b, "lop and, input {i}");
        assert_eq!(got.regs[7], a | 0x00FF_00FF, "lop32i or, input {i}");
        assert_eq!(got.regs[8], !b, "lop pass-b inverted, input {i}");
        assert_eq!(got.regs[1], a, "sources are untouched");
    }
}

#[test]
fn float_arithmetic_matches_rust() {
    let Some(vk) = vulkan() else { return };
    // R3 = R1 + R2; R4 = R1 * R2; R5 = fma(R1, R2, R3); R6 = |R1| + -R2 (modifiers).
    let program = [
        Enc::op(Op::FaddReg).rd(3).ra(1).rb(2).build(),
        Enc::op(Op::FmulReg).rd(4).ra(1).rb(2).build(),
        Enc::op(Op::FfmaReg).rd(5).ra(1).rb(2).rc(3).build(),
        Enc::op(Op::FaddReg)
            .rd(6)
            .ra(1)
            .rb(2)
            .bit(46, true)
            .bit(45, true)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let mut inputs = random_states(256, 11);
    // Keep the values ordinary: finite, moderate magnitudes.
    for s in &mut inputs {
        for reg in s.regs.iter_mut().take(4) {
            let m = (*reg as f32 / u32::MAX as f32 - 0.5) * 200.0;
            *reg = m.to_bits();
        }
    }
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let (a, b) = (f32::from_bits(inp.regs[1]), f32::from_bits(inp.regs[2]));
        let sum = a + b;
        assert_eq!(got.regs[3], sum.to_bits(), "fadd, input {i}");
        assert_eq!(got.regs[4], (a * b).to_bits(), "fmul, input {i}");
        // A fused multiply-add rounds once. A software device (a CPU implementation of Vulkan)
        // may not fuse and then rounds twice, so it is allowed that result as well.
        let fused = a.mul_add(b, sum).to_bits();
        let unfused = (a * b + sum).to_bits();
        if vk.caps.discrete {
            assert_eq!(got.regs[5], fused, "ffma, input {i}");
        } else {
            assert!(
                got.regs[5] == fused || got.regs[5] == unfused,
                "ffma, input {i}: {:#x} is neither {fused:#x} nor {unfused:#x}",
                got.regs[5]
            );
        }
        assert_eq!(
            got.regs[6],
            (a.abs() + -b).to_bits(),
            "fadd with modifiers, input {i}"
        );
    }
}

#[test]
fn predicates_move_to_and_from_registers() {
    let Some(vk) = vulkan() else { return };
    // R4 = R1 with the byte B1 holding the predicates named by 0x5B (P2R), R5 the same of byte B3
    // for all of them from R2, then the predicates are set again from R1's byte 2 under a mask
    // (R2P with an immediate) and from byte 0 under the mask in R3 (R2P with a register).
    let program = [
        Enc::op(Op::P2rImm20)
            .rd(4)
            .ra(1)
            .imm20(0x5B)
            .field(41, 2, 1)
            .build(),
        Enc::op(Op::P2rImm20)
            .rd(5)
            .ra(2)
            .imm20(0xFF)
            .field(41, 2, 3)
            .build(),
        Enc::op(Op::R2pImm20)
            .ra(1)
            .imm20(0x0F)
            .field(41, 2, 2)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(256, 11);
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let pr = (inp.preds & 0x7F) | 0x80;
        let mask = 0x5Bu32;
        let expect4 = (inp.regs[1] & !(mask << 8)) | ((pr & mask) << 8);
        assert_eq!(got.regs[4], expect4, "p2r with a mask, input {i}");
        let expect5 = (inp.regs[2] & !(0xFFu32 << 24)) | (pr << 24);
        assert_eq!(got.regs[5], expect5, "p2r of all, input {i}");
        let from = (inp.regs[1] >> 16) & 0x0F;
        let expect_preds = (inp.preds & !0x0F) | from;
        assert_eq!(got.preds & 0x7F, expect_preds & 0x7F, "r2p, input {i}");
    }
    // The register form takes its mask from a register, and a guard keeps the predicates.
    let program = [
        Enc::op(Op::R2pReg).ra(1).rb(3).field(41, 2, 0).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let mask = inp.regs[3] & 0x7F;
        let expect = (inp.preds & !mask) | (inp.regs[1] & mask);
        assert_eq!(
            got.preds & 0x7F,
            expect & 0x7F,
            "r2p by register, input {i}"
        );
    }
}

#[test]
fn a_guarded_instruction_and_a_branch_follow_the_predicates() {
    let Some(vk) = vulkan() else { return };
    // Slots 1..3 are the first bundle, slot 4 is its scheduling word... no: slot 0 is the
    // scheduling word, slots 1..3 the instructions, slot 4 the next scheduling word.
    //   slot 1: @P0 R3 = R1 + R2
    //   slot 2: @P1 BRA slot 5
    //   slot 3: R4 = 0x1234
    //   slot 5: R5 = 0x5678       (the first instruction of the second bundle)
    //   slot 6: EXIT
    let at = |slot: u32| slot * 8;
    let program = vec![
        Enc::op(Op::IaddReg).rd(3).ra(1).rb(2).guard(P0).build(),
        bra(
            at(2),
            at(5),
            Guard {
                index: 1,
                negated: false,
            },
        ),
        Enc::op(Op::Mov32iImm32)
            .rd(4)
            .imm32(0x1234)
            .field(12, 4, 0xF)
            .build(),
        Enc::op(Op::Mov32iImm32)
            .rd(5)
            .imm32(0x5678)
            .field(12, 4, 0xF)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(64, 3);
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let p0 = inp.preds & 1 != 0;
        let p1 = inp.preds & 2 != 0;
        let expect3 = if p0 {
            inp.regs[1].wrapping_add(inp.regs[2])
        } else {
            inp.regs[3]
        };
        assert_eq!(got.regs[3], expect3, "guarded add, input {i}");
        let expect4 = if p1 { inp.regs[4] } else { 0x1234 };
        assert_eq!(
            got.regs[4], expect4,
            "the instruction the branch skips, input {i}"
        );
        assert_eq!(got.regs[5], 0x5678, "after the join, input {i}");
        assert_eq!(got.preds, inp.preds, "predicates are untouched");
    }
}

/// `F2I` to 8, 16 and 32 bits saturates, rounds to nearest even and turns NaN into zero.
#[test]
fn float_to_integer_conversions_saturate() {
    let Some(vk) = vulkan() else { return };
    // The destination type: bits 8..9 the size (0 8-bit, 1 16, 2 32), bit 12 signed; the source
    // is a 32-bit float (bits 10..11 = 2). The source register is in bits 20..27.
    let f2i = |rd: u8, size: u64, signed: bool| {
        Enc::op(Op::F2iReg)
            .rd(rd)
            .rb(1)
            .field(8, 2, size)
            .field(10, 2, 2)
            .bit(12, signed)
            .build()
    };
    let program = [
        f2i(3, 0, false),
        f2i(4, 0, true),
        f2i(5, 1, false),
        f2i(6, 1, true),
        f2i(7, 2, false),
        f2i(8, 2, true),
        exit(Guard::ALWAYS),
    ];
    let mut values: Vec<f32> = vec![
        0.0,
        -0.0,
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        127.5,
        128.0,
        255.4,
        255.6,
        300.0,
        32767.5,
        32768.0,
        65535.0,
        65535.5,
        70000.0,
        -129.0,
        -32768.0,
        -40000.0,
        1.0e9,
        -1.0e9,
        3.0e9,
        5.0e9,
        -5.0e9,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        2147483520.0,
        4294967040.0,
    ];
    let mut rng = Rng(99);
    for _ in 0..200 {
        values.push((rng.next() as f32 / u32::MAX as f32 - 0.5) * 140_000.0);
    }
    let inputs: Vec<State> = values
        .iter()
        .map(|v| {
            let mut s = State::default();
            s.regs[1] = v.to_bits();
            s
        })
        .collect();
    let out = run(&vk, &program, &inputs, &[]);
    for (v, got) in values.iter().zip(&out) {
        let r = v.round_ties_even();
        let clamp = |lo: f32, hi: f32| if v.is_nan() { 0.0 } else { r.clamp(lo, hi) };
        assert_eq!(got.regs[3], clamp(0.0, 255.0) as u32, "u8 of {v}");
        assert_eq!(got.regs[4], clamp(-128.0, 127.0) as i32 as u32, "s8 of {v}");
        assert_eq!(got.regs[5], clamp(0.0, 65535.0) as u32, "u16 of {v}");
        assert_eq!(
            got.regs[6],
            clamp(-32768.0, 32767.0) as i32 as u32,
            "s16 of {v}"
        );
        // Rust's float to integer casts saturate and map NaN to zero, which is what the
        // hardware does too.
        assert_eq!(got.regs[7], r as u32, "u32 of {v}");
        assert_eq!(got.regs[8], r as i32 as u32, "s32 of {v}");
    }
}

// ---- comparisons, selection and minimum / maximum --------------------------------------------

/// Values with the awkward cases in: zeros, infinities, NaN, equal and near values.
fn float_pool() -> Vec<f32> {
    vec![
        0.0,
        1.0,
        -1.0,
        0.5,
        -2.5,
        3.0,
        100.25,
        -100.25,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.000_000_1,
    ]
}

fn pair_states(values: &[f32]) -> Vec<State> {
    let mut out = Vec::new();
    for a in values {
        for b in values {
            let mut s = State::default();
            s.regs[1] = a.to_bits();
            s.regs[2] = b.to_bits();
            out.push(s);
        }
    }
    out
}

/// The 4-bit comparison of `FSETP`/`FSET`.
fn fcmp(code: u32, a: f32, b: f32) -> bool {
    let unordered = a.is_nan() || b.is_nan();
    match code {
        0 => false,
        1 => a < b,
        2 => a == b,
        3 => a <= b,
        4 => a > b,
        5 => !unordered && a != b,
        6 => a >= b,
        7 => !unordered,
        8 => unordered,
        9 => unordered || a < b,
        10 => unordered || a == b,
        11 => unordered || a <= b,
        12 => unordered || a > b,
        13 => unordered || a != b,
        14 => unordered || a >= b,
        _ => true,
    }
}

fn bool_op(code: u32, a: bool, b: bool) -> bool {
    match code {
        0 => a && b,
        1 => a || b,
        _ => a != b,
    }
}

#[test]
fn float_comparisons_set_predicates_and_registers() {
    let Some(vk) = vulkan() else { return };
    let inputs = pair_states(&float_pool());
    for code in 0..16u32 {
        for bop in 0..3u32 {
            // FSETP.cmp.bop P0, P1, R1, R2, P2 and FSET.cmp.bop R3, R1, R2, P2 (bf: 1.0 for true).
            let program = [
                Enc::op(Op::FsetpReg)
                    .ra(1)
                    .rb(2)
                    .field(0, 3, 1)
                    .field(3, 3, 0)
                    .field(48, 4, u64::from(code))
                    .field(45, 2, u64::from(bop))
                    .pred(39, 2, false)
                    .build(),
                Enc::op(Op::FsetReg)
                    .rd(3)
                    .ra(1)
                    .rb(2)
                    .field(48, 4, u64::from(code))
                    .field(45, 2, u64::from(bop))
                    .bit(52, true)
                    .pred(39, 2, false)
                    .build(),
                Enc::op(Op::FsetReg)
                    .rd(4)
                    .ra(1)
                    .rb(2)
                    .field(48, 4, u64::from(code))
                    .field(45, 2, u64::from(bop))
                    .pred(39, 2, false)
                    .build(),
                exit(Guard::ALWAYS),
            ];
            for pc in [false, true] {
                let mut states = inputs.clone();
                for s in &mut states {
                    s.preds = if pc { 4 } else { 0 };
                }
                let out = run(&vk, &program, &states, &[]);
                for (inp, got) in states.iter().zip(&out) {
                    let (a, b) = (f32::from_bits(inp.regs[1]), f32::from_bits(inp.regs[2]));
                    let c = fcmp(code, a, b);
                    let msg = format!("cmp {code} bop {bop} pc {pc} on {a} and {b}");
                    assert_eq!(got.preds & 1 != 0, bool_op(bop, c, pc), "P0, {msg}");
                    assert_eq!(got.preds & 2 != 0, bool_op(bop, !c, pc), "P1, {msg}");
                    let t = bool_op(bop, c, pc);
                    assert_eq!(
                        got.regs[3],
                        if t { 1.0f32.to_bits() } else { 0 },
                        "FSET.BF, {msg}"
                    );
                    assert_eq!(got.regs[4], if t { u32::MAX } else { 0 }, "FSET, {msg}");
                }
            }
        }
    }
}

#[test]
fn float_modifiers_apply_to_comparisons_and_extremes() {
    let Some(vk) = vulkan() else { return };
    let inputs = pair_states(&float_pool());
    // FSETP.LT with |Ra| and -Rb; FMNMX min and max with -Ra and |Rb|.
    let program = [
        Enc::op(Op::FsetpReg)
            .ra(1)
            .rb(2)
            .field(0, 3, 1)
            .field(3, 3, 0)
            .field(48, 4, 1)
            .pred(39, 7, false)
            .bit(7, true)
            .bit(6, true)
            .build(),
        Enc::op(Op::FmnmxReg)
            .rd(3)
            .ra(1)
            .rb(2)
            .pred(39, 7, false)
            .bit(48, true)
            .bit(49, true)
            .build(),
        Enc::op(Op::FmnmxReg)
            .rd(4)
            .ra(1)
            .rb(2)
            .pred(39, 7, true)
            .bit(48, true)
            .bit(49, true)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b) = (f32::from_bits(inp.regs[1]), f32::from_bits(inp.regs[2]));
        let msg = format!("on {a} and {b}");
        assert_eq!(got.preds & 1 != 0, a.abs() < -b, "FSETP.LT |a|, -b, {msg}");
        // A NaN operand yields the other one; two NaNs give NaN.
        let (x, y) = (-a, b.abs());
        let lo = x.min(y);
        let hi = x.max(y);
        let (glo, ghi) = (f32::from_bits(got.regs[3]), f32::from_bits(got.regs[4]));
        if lo.is_nan() {
            assert!(glo.is_nan() && ghi.is_nan(), "FMNMX of two NaNs, {msg}");
        } else {
            assert_eq!(glo, lo, "FMNMX min, {msg}");
            assert_eq!(ghi, hi, "FMNMX max, {msg}");
        }
    }
}

#[test]
fn integer_comparisons_selection_and_extremes() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(300, 21);
    // Equal and adjacent values, and the signed extremes.
    for (a, b) in [
        (5u32, 5u32),
        (0, 0),
        (u32::MAX, 0),
        (0, u32::MAX),
        (0x8000_0000, 0x7FFF_FFFF),
        (1, 2),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        inputs.push(s);
    }
    for signed in [false, true] {
        for code in 0..8u32 {
            for bop in 0..3u32 {
                let program = [
                    Enc::op(Op::IsetpReg)
                        .ra(1)
                        .rb(2)
                        .field(0, 3, 1)
                        .field(3, 3, 0)
                        .field(49, 3, u64::from(code))
                        .field(45, 2, u64::from(bop))
                        .bit(48, signed)
                        .pred(39, 2, false)
                        .build(),
                    Enc::op(Op::IsetReg)
                        .rd(3)
                        .ra(1)
                        .rb(2)
                        .field(49, 3, u64::from(code))
                        .field(45, 2, u64::from(bop))
                        .bit(48, signed)
                        .bit(44, true)
                        .pred(39, 2, false)
                        .build(),
                    exit(Guard::ALWAYS),
                ];
                let out = run(&vk, &program, &inputs, &[]);
                for (inp, got) in inputs.iter().zip(&out) {
                    let (a, b) = (inp.regs[1], inp.regs[2]);
                    let ord = if signed {
                        (a as i32).cmp(&(b as i32))
                    } else {
                        a.cmp(&b)
                    };
                    let c = match code {
                        0 => false,
                        1 => ord.is_lt(),
                        2 => ord.is_eq(),
                        3 => ord.is_le(),
                        4 => ord.is_gt(),
                        5 => ord.is_ne(),
                        6 => ord.is_ge(),
                        _ => true,
                    };
                    let pc = inp.preds & 4 != 0;
                    let msg = format!("signed {signed} cmp {code} bop {bop} on {a:#x} and {b:#x}");
                    assert_eq!(got.preds & 1 != 0, bool_op(bop, c, pc), "P0, {msg}");
                    assert_eq!(got.preds & 2 != 0, bool_op(bop, !c, pc), "P1, {msg}");
                    let t = bool_op(bop, c, pc);
                    assert_eq!(
                        got.regs[3],
                        if t { 1.0f32.to_bits() } else { 0 },
                        "ISET.BF, {msg}"
                    );
                }
            }
        }
    }
    // SEL and IMNMX.
    for signed in [false, true] {
        let program = [
            Enc::op(Op::SelReg)
                .rd(3)
                .ra(1)
                .rb(2)
                .pred(39, 2, false)
                .build(),
            Enc::op(Op::SelReg)
                .rd(4)
                .ra(1)
                .rb(2)
                .pred(39, 2, true)
                .build(),
            Enc::op(Op::ImnmxReg)
                .rd(5)
                .ra(1)
                .rb(2)
                .pred(39, 7, false)
                .bit(48, signed)
                .build(),
            Enc::op(Op::ImnmxReg)
                .rd(6)
                .ra(1)
                .rb(2)
                .pred(39, 7, true)
                .bit(48, signed)
                .build(),
            exit(Guard::ALWAYS),
        ];
        let out = run(&vk, &program, &inputs, &[]);
        for (inp, got) in inputs.iter().zip(&out) {
            let (a, b) = (inp.regs[1], inp.regs[2]);
            let p2 = inp.preds & 4 != 0;
            assert_eq!(got.regs[3], if p2 { a } else { b }, "SEL");
            assert_eq!(
                got.regs[4],
                if p2 { b } else { a },
                "SEL with the predicate negated"
            );
            let (lo, hi) = if signed {
                (
                    (a as i32).min(b as i32) as u32,
                    (a as i32).max(b as i32) as u32,
                )
            } else {
                (a.min(b), a.max(b))
            };
            assert_eq!(got.regs[5], lo, "IMNMX min, signed {signed}");
            assert_eq!(got.regs[6], hi, "IMNMX max, signed {signed}");
        }
    }
}

#[test]
fn predicate_logic_matches_rust() {
    let Some(vk) = vulkan() else { return };
    let inputs: Vec<State> = (0..128u32)
        .map(|bits| State {
            preds: bits,
            ..State::default()
        })
        .collect();
    for op1 in 0..3u32 {
        for op2 in 0..3u32 {
            // PSETP.op1.op2 P3, P4, P0, !P1, P2 and PSET.op1.op2.BF R3, P0, P1, !P2.
            let program = [
                Enc::op(Op::Psetp)
                    .field(0, 3, 4)
                    .field(3, 3, 3)
                    .pred(12, 0, false)
                    .pred(29, 1, true)
                    .pred(39, 2, false)
                    .field(24, 2, u64::from(op1))
                    .field(45, 2, u64::from(op2))
                    .build(),
                Enc::op(Op::Pset)
                    .rd(3)
                    .pred(12, 0, false)
                    .pred(29, 1, false)
                    .pred(39, 2, true)
                    .field(24, 2, u64::from(op1))
                    .field(45, 2, u64::from(op2))
                    .bit(44, true)
                    .build(),
                exit(Guard::ALWAYS),
            ];
            let out = run(&vk, &program, &inputs, &[]);
            for (inp, got) in inputs.iter().zip(&out) {
                let p = |i: u32| inp.preds >> i & 1 != 0;
                let r3 = bool_op(op2, bool_op(op1, p(0), !p(1)), p(2));
                let r4 = bool_op(op2, bool_op(op1, !p(0), !p(1)), p(2));
                let msg = format!("ops {op1} {op2}, predicates {:#b}", inp.preds);
                assert_eq!(got.preds >> 3 & 1 != 0, r3, "P3, {msg}");
                assert_eq!(got.preds >> 4 & 1 != 0, r4, "P4, {msg}");
                let s = bool_op(op2, bool_op(op1, p(0), p(1)), !p(2));
                assert_eq!(
                    got.regs[3],
                    if s { 1.0f32.to_bits() } else { 0 },
                    "PSET, {msg}"
                );
            }
        }
    }
}

// ---- shifts, bit fields and multiplication ---------------------------------------------------

/// Random operands with the counts and bit-field descriptors that matter in R2.
fn bit_states(seed: u64) -> Vec<State> {
    let mut inputs = random_states(400, seed);
    let mut rng = Rng(seed ^ 0x55);
    for s in &mut inputs {
        // Shift counts around the interesting range; field starts and lengths.
        s.regs[3] = rng.next() % 40;
        s.regs[4] = (rng.next() % 40) | (rng.next() % 40) << 8 | (rng.next() & 0xFFFF) << 16;
    }
    for v in [0u32, 1, u32::MAX, 0x8000_0000, 0x7FFF_FFFF, 0xFFFF_0000] {
        for count in [0u32, 1, 31, 32, 33] {
            let mut s = State::default();
            s.regs[1] = v;
            s.regs[3] = count;
            s.regs[4] = count | 8 << 8;
            inputs.push(s);
        }
    }
    inputs
}

#[test]
fn shifts_wrap_or_saturate_as_flagged() {
    let Some(vk) = vulkan() else { return };
    let inputs = bit_states(31);
    let program = [
        // SHL R5 = R1 << R3 (no wrap: counts of 32 and more give 0); SHL.W wraps the count.
        Enc::op(Op::ShlReg).rd(5).ra(1).rb(3).build(),
        Enc::op(Op::ShlReg).rd(6).ra(1).rb(3).bit(39, true).build(),
        // SHR.U32 and SHR.S32 and their wrapping forms (bit 48 set means signed).
        Enc::op(Op::ShrReg).rd(7).ra(1).rb(3).build(),
        Enc::op(Op::ShrReg).rd(8).ra(1).rb(3).bit(48, true).build(),
        Enc::op(Op::ShrReg).rd(9).ra(1).rb(3).bit(39, true).build(),
        Enc::op(Op::ShrReg)
            .rd(10)
            .ra(1)
            .rb(3)
            .bit(39, true)
            .bit(48, true)
            .build(),
        // SHR with the bits of the source reversed first.
        Enc::op(Op::ShrReg).rd(11).ra(1).rb(3).bit(40, true).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, n) = (inp.regs[1], inp.regs[3]);
        let msg = format!("{a:#x} by {n}");
        let wrap = n & 31;
        assert_eq!(got.regs[5], if n >= 32 { 0 } else { a << n }, "SHL, {msg}");
        assert_eq!(got.regs[6], a << wrap, "SHL.W, {msg}");
        assert_eq!(
            got.regs[7],
            if n >= 32 { 0 } else { a >> n },
            "SHR.U32, {msg}"
        );
        assert_eq!(
            got.regs[8],
            ((a as i32) >> n.min(31)) as u32,
            "SHR.S32, {msg}"
        );
        assert_eq!(got.regs[9], a >> wrap, "SHR.U32.W, {msg}");
        assert_eq!(
            got.regs[10],
            ((a as i32) >> wrap) as u32,
            "SHR.S32.W, {msg}"
        );
        let rev = a.reverse_bits();
        assert_eq!(
            got.regs[11],
            if n >= 32 { 0 } else { rev >> n },
            "SHR.BREV, {msg}"
        );
    }
}

/// `BFE` as the PTX `bfe` instruction defines it, for a start below 32.
fn bfe_model(a: u32, start: u32, len: u32, signed: bool) -> u32 {
    // Bits past the end of the word, and past the length, take the sign bit of the field.
    let sbit = signed && len > 0 && a >> (start + len - 1).min(31) & 1 != 0;
    (0..32).fold(0u32, |d, i| {
        let bit = if i < len && start + i <= 31 {
            a >> (start + i) & 1 != 0
        } else {
            sbit
        };
        d | u32::from(bit) << i
    })
}

#[test]
fn bit_fields_population_and_leading_bits() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = bit_states(32);
    // Descriptors with a start below 32 (the model's domain).
    for s in &mut inputs {
        s.regs[4] = (s.regs[4] & 0x1F) | (s.regs[4] >> 8 & 0x3F) << 8;
    }
    let program = [
        Enc::op(Op::BfeReg).rd(5).ra(1).rb(4).build(),
        Enc::op(Op::BfeReg).rd(6).ra(1).rb(4).bit(48, true).build(),
        Enc::op(Op::PopcReg).rd(7).rb(1).build(),
        Enc::op(Op::PopcReg).rd(8).rb(1).bit(40, true).build(),
        Enc::op(Op::FloReg).rd(9).rb(1).build(),
        Enc::op(Op::FloReg).rd(10).rb(1).bit(48, true).build(),
        Enc::op(Op::FloReg).rd(11).rb(1).bit(41, true).build(),
        Enc::op(Op::FloReg)
            .rd(12)
            .rb(1)
            .bit(48, true)
            .bit(41, true)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let a = inp.regs[1];
        let (start, len) = (inp.regs[4] & 0xFF, inp.regs[4] >> 8 & 0xFF);
        let msg = format!("{a:#x}, field {start}+{len}");
        // Bit 48 set is the signed form: R5 is BFE.U32 and R6 BFE.S32.
        assert_eq!(
            got.regs[5],
            bfe_model(a, start, len, false),
            "BFE.U32, {msg}"
        );
        assert_eq!(
            got.regs[6],
            bfe_model(a, start, len, true),
            "BFE.S32, {msg}"
        );
        assert_eq!(got.regs[7], a.count_ones(), "POPC, {msg}");
        assert_eq!(got.regs[8], (!a).count_ones(), "POPC.INV, {msg}");
        // FLO: the position of the highest set bit; for the signed form, of the highest bit that
        // differs from the sign; all ones when there is none.
        let unsigned_pos = if a == 0 {
            u32::MAX
        } else {
            31 - a.leading_zeros()
        };
        let signed_pos = if a == 0 || a == u32::MAX {
            u32::MAX
        } else {
            let x = if (a as i32) < 0 { !a } else { a };
            31 - x.leading_zeros()
        };
        assert_eq!(got.regs[9], unsigned_pos, "FLO.U32, {msg}");
        assert_eq!(got.regs[10], signed_pos, "FLO.S32, {msg}");
        let shift = |p: u32| if p == u32::MAX { u32::MAX } else { 31 - p };
        assert_eq!(got.regs[11], shift(unsigned_pos), "FLO.U32.SH, {msg}");
        assert_eq!(got.regs[12], shift(signed_pos), "FLO.S32.SH, {msg}");
    }
}

#[test]
fn integer_multiplication_low_and_high() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(300, 41);
    for (a, b) in [
        (u32::MAX, u32::MAX),
        (0x8000_0000, 0x8000_0000),
        (0x8000_0000, u32::MAX),
        (0, u32::MAX),
        (0xFFFF, 0xFFFF),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        inputs.push(s);
    }
    let program = [
        Enc::op(Op::ImulReg).rd(3).ra(1).rb(2).build(),
        Enc::op(Op::ImulReg).rd(4).ra(1).rb(2).bit(39, true).build(),
        Enc::op(Op::ImulReg)
            .rd(5)
            .ra(1)
            .rb(2)
            .bit(39, true)
            .bit(40, true)
            .bit(41, true)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b) = (inp.regs[1], inp.regs[2]);
        let msg = format!("{a:#x} * {b:#x}");
        assert_eq!(got.regs[3], a.wrapping_mul(b), "IMUL, {msg}");
        assert_eq!(
            got.regs[4],
            ((u64::from(a) * u64::from(b)) >> 32) as u32,
            "IMUL.U32.U32.HI, {msg}"
        );
        assert_eq!(
            got.regs[5],
            ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32,
            "IMUL.S32.S32.HI, {msg}"
        );
    }
}

// ---- conversions and special functions -------------------------------------------------------

#[test]
fn integer_to_float_and_back_between_sizes() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(300, 51);
    for v in [
        0u32,
        1,
        0xFF,
        0x80,
        0x7F,
        0xFFFF,
        0x8000,
        0x7FFF,
        u32::MAX,
        0x8000_0000,
    ] {
        let mut s = State::default();
        s.regs[1] = v;
        inputs.push(s);
    }
    // I2F with the source taken as: u32, s32, u8 (byte 0), s8 (byte 1), u16 (half 1), s16, and
    // s32 with the absolute value and with the negation.
    let i2f = |rd: u8, size: u64, signed: bool, sel: u64| {
        Enc::op(Op::I2fReg)
            .rd(rd)
            .rb(1)
            .field(8, 2, 2)
            .field(10, 2, size)
            .bit(13, signed)
            .field(41, 2, sel)
    };
    // I2I: the destination size and signedness, saturating or not, from a 32-bit source.
    let i2i = |rd: u8, dsize: u64, dsigned: bool, sat: bool| {
        Enc::op(Op::I2iReg)
            .rd(rd)
            .rb(1)
            .field(8, 2, dsize)
            .bit(12, dsigned)
            .field(10, 2, 2)
            .bit(13, true)
            .bit(50, sat)
    };
    let program = [
        i2f(3, 2, false, 0).build(),
        i2f(4, 2, true, 0).build(),
        i2f(5, 0, false, 0).build(),
        i2f(6, 0, true, 1).build(),
        i2f(7, 1, false, 2).build(),
        i2f(8, 1, true, 0).build(),
        i2f(9, 2, true, 0).bit(49, true).build(),
        i2f(10, 2, true, 0).bit(45, true).build(),
        i2i(11, 0, false, true).build(),
        i2i(12, 0, true, true).build(),
        i2i(13, 1, false, true).build(),
        i2i(14, 1, true, true).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let a = inp.regs[1];
        let msg = format!("{a:#x}");
        let f = |r: usize| f32::from_bits(got.regs[r]);
        assert_eq!(f(3), a as f32, "I2F.U32, {msg}");
        assert_eq!(f(4), a as i32 as f32, "I2F.S32, {msg}");
        assert_eq!(f(5), (a & 0xFF) as f32, "I2F.U8, {msg}");
        assert_eq!(
            f(6),
            ((a >> 8) as u8 as i8) as f32,
            "I2F.S8 of byte 1, {msg}"
        );
        assert_eq!(f(7), (a >> 16) as f32, "I2F.U16 of the high half, {msg}");
        assert_eq!(f(8), (a as u16 as i16) as f32, "I2F.S16, {msg}");
        assert_eq!(
            f(9),
            (a as i32).wrapping_abs() as u32 as i32 as f32,
            "I2F.S32.ABS, {msg}"
        );
        assert_eq!(
            f(10),
            (a as i32).wrapping_neg() as f32,
            "I2F.S32.NEG, {msg}"
        );
        // The saturating narrowing conversions read the source as signed 32-bit.
        let s = a as i32;
        assert_eq!(got.regs[11], s.clamp(0, 255) as u32, "I2I.U8.SAT, {msg}");
        assert_eq!(got.regs[12], s.clamp(-128, 127) as u32, "I2I.S8.SAT, {msg}");
        assert_eq!(got.regs[13], s.clamp(0, 65535) as u32, "I2I.U16.SAT, {msg}");
        assert_eq!(
            got.regs[14],
            s.clamp(-32768, 32767) as u32,
            "I2I.S16.SAT, {msg}"
        );
    }
}

#[test]
fn float_to_float_rounding_modes() {
    let Some(vk) = vulkan() else { return };
    let mut values: Vec<f32> = vec![
        0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4, 2.6, -2.4, -2.6, 7.0, -7.0,
    ];
    let mut rng = Rng(61);
    for _ in 0..100 {
        values.push((rng.next() as f32 / u32::MAX as f32 - 0.5) * 2000.0);
    }
    let inputs: Vec<State> = values
        .iter()
        .map(|v| {
            let mut s = State::default();
            s.regs[1] = v.to_bits();
            s
        })
        .collect();
    // F2F with each rounding (bits 39..40 and 42: round 0b100, floor 0b101, ceil 0b110, trunc
    // 0b111, pass 0b011), with the negation and the absolute value, and saturating.
    let f2f = |rd: u8, mode: u64| {
        Enc::op(Op::F2fReg)
            .rd(rd)
            .rb(1)
            .field(8, 2, 2)
            .field(10, 2, 2)
            .field(39, 2, mode & 3)
            .bit(42, mode & 4 != 0)
    };
    let program = [
        f2f(3, 0b011).build(),
        f2f(4, 0b100).build(),
        f2f(5, 0b101).build(),
        f2f(6, 0b110).build(),
        f2f(7, 0b111).build(),
        f2f(8, 0b100).bit(45, true).build(),
        f2f(9, 0b111).bit(49, true).build(),
        f2f(10, 0b011).bit(50, true).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (v, got) in values.iter().zip(&out) {
        let f = |r: usize| f32::from_bits(got.regs[r]);
        assert_eq!(f(3), *v, "pass, {v}");
        assert_eq!(f(4), v.round_ties_even(), "round, {v}");
        assert_eq!(f(5), v.floor(), "floor, {v}");
        assert_eq!(f(6), v.ceil(), "ceil, {v}");
        assert_eq!(f(7), v.trunc(), "trunc, {v}");
        assert_eq!(f(8), (-v).round_ties_even(), "round of the negation, {v}");
        assert_eq!(f(9), v.abs().trunc(), "trunc of the absolute value, {v}");
        assert_eq!(f(10), v.clamp(0.0, 1.0), "saturate, {v}");
    }
}

#[test]
fn special_functions_are_close_to_the_exact_ones() {
    let Some(vk) = vulkan() else { return };
    let mut rng = Rng(71);
    let mut values: Vec<f32> = vec![0.25, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1.5e-3, 12345.0];
    for _ in 0..200 {
        values.push(rng.next() as f32 / u32::MAX as f32 * 50.0 + 1e-3);
    }
    let inputs: Vec<State> = values
        .iter()
        .map(|v| {
            let mut s = State::default();
            s.regs[1] = v.to_bits();
            s
        })
        .collect();
    let mufu = |rd: u8, function: u64| Enc::op(Op::Mufu).rd(rd).ra(1).field(20, 4, function);
    let program = [
        mufu(3, 4).build(),                             // rcp
        mufu(4, 5).build(),                             // rsq
        mufu(5, 8).build(),                             // sqrt
        mufu(6, 3).build(),                             // lg2
        mufu(7, 4).bit(46, true).bit(48, true).build(), // rcp of -|x|
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    let close = |got: f32, want: f32, what: &str, x: f32| {
        let tolerance = want.abs() * 1e-5 + 1e-6;
        assert!(
            (got - want).abs() <= tolerance,
            "{what}({x}) = {got}, expected {want}"
        );
    };
    for (x, got) in values.iter().zip(&out) {
        let f = |r: usize| f32::from_bits(got.regs[r]);
        close(f(3), 1.0 / x, "rcp", *x);
        close(f(4), 1.0 / x.sqrt(), "rsq", *x);
        close(f(5), x.sqrt(), "sqrt", *x);
        close(f(6), x.log2(), "lg2", *x);
        close(f(7), -1.0 / x, "rcp(-|x|)", *x);
    }
    // Sine, cosine and the base-2 exponential over their usual ranges.
    let angles: Vec<f32> = (0..64).map(|i| -3.1 + 6.2 * i as f32 / 63.0).collect();
    let inputs: Vec<State> = angles
        .iter()
        .map(|v| {
            let mut s = State::default();
            s.regs[1] = v.to_bits();
            s
        })
        .collect();
    let program = [
        mufu(3, 1).build(), // sin
        mufu(4, 0).build(), // cos
        mufu(5, 2).build(), // ex2
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (x, got) in angles.iter().zip(&out) {
        let f = |r: usize| f32::from_bits(got.regs[r]);
        assert!((f(3) - x.sin()).abs() < 1e-3, "sin({x}) = {}", f(3));
        assert!((f(4) - x.cos()).abs() < 1e-3, "cos({x}) = {}", f(4));
        close(f(5), x.exp2(), "ex2", *x);
    }
}
