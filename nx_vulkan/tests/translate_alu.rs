//! The translator against the GPU: programs of the harness mode run for many random inputs and the
//! results are compared with what plain Rust arithmetic gives. Skipped (loudly) without Vulkan or
//! without `spirv-val`.

use nx_shader::asm::{assemble, bra, exit, Enc, P0};
use nx_shader::cfg::Code;
use nx_shader::inst::Guard;
use nx_shader::ops::Op;
use nx_shader::translate::{test_bindings as tb, translate, Config, Stage};
use nx_vulkan::Vk;

fn vulkan() -> Option<Vk> {
    match Vk::new() {
        Ok(v) => Some(v),
        Err(e) => {
            if std::env::var("RECOMP_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
                panic!("no Vulkan device and RECOMP_REQUIRE_EXTERNAL=1: {e}");
            }
            eprintln!("SKIPPED: no Vulkan device: {e}");
            None
        }
    }
}

/// A small deterministic generator for test inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// The registers (and predicates) of one invocation.
#[derive(Clone, Copy, Default)]
struct State {
    regs: [u32; 15],
    preds: u32,
}

/// Run `program` once per element of `inputs` and return the final states. `cbuf0` is bound as
/// constant buffer 0.
fn run(vk: &Vk, program: &[u64], inputs: &[State], cbuf0: &[u32]) -> Vec<State> {
    let bytes = assemble(program);
    let t = translate(Code(&bytes), None, &Config::new(Stage::Compute))
        .unwrap_or_else(|e| panic!("translation failed: {e}"));
    nx_shader::validate::assert_valid(&t.spirv);
    let n = inputs.len();
    let mut regs_in = Vec::with_capacity(n * 16 * 4);
    for s in inputs {
        for r in s.regs {
            regs_in.extend_from_slice(&r.to_le_bytes());
        }
        regs_in.extend_from_slice(&s.preds.to_le_bytes());
    }
    let mut regs_out = vec![0u8; n * 16 * 4];
    let mut attr_in = vec![0u8; n * tb::ATTRS as usize * 4];
    let mut attr_out = vec![0u8; n * tb::ATTRS as usize * 4];
    let mut c0: Vec<u8> = cbuf0.iter().flat_map(|w| w.to_le_bytes()).collect();
    c0.resize(c0.len().max(4096), 0);
    let cbuf_binding = 64 * Stage::Compute.index();
    vk.run_compute(
        &t.spirv,
        &mut [
            (tb::REGS_IN, &mut regs_in),
            (tb::REGS_OUT, &mut regs_out),
            (tb::ATTR_IN, &mut attr_in),
            (tb::ATTR_OUT, &mut attr_out),
            (cbuf_binding, &mut c0),
        ],
        [n as u32, 1, 1],
    )
    .unwrap();
    (0..n)
        .map(|i| {
            let word = |j: usize| {
                u32::from_le_bytes(
                    regs_out[(i * 16 + j) * 4..(i * 16 + j) * 4 + 4]
                        .try_into()
                        .unwrap(),
                )
            };
            let mut s = State::default();
            for (j, r) in s.regs.iter_mut().enumerate() {
                *r = word(j);
            }
            s.preds = word(15);
            s
        })
        .collect()
}

fn random_states(n: usize, seed: u64) -> Vec<State> {
    let mut rng = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let mut s = State::default();
            for r in s.regs.iter_mut() {
                *r = rng.next();
            }
            s.preds = rng.next() & 0x7F;
            s
        })
        .collect()
}

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
        for r in 0..4 {
            let m = (s.regs[r] as f32 / u32::MAX as f32 - 0.5) * 200.0;
            s.regs[r] = m.to_bits();
        }
    }
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let (a, b) = (f32::from_bits(inp.regs[1]), f32::from_bits(inp.regs[2]));
        let sum = a + b;
        assert_eq!(got.regs[3], sum.to_bits(), "fadd, input {i}");
        assert_eq!(got.regs[4], (a * b).to_bits(), "fmul, input {i}");
        assert_eq!(got.regs[5], a.mul_add(b, sum).to_bits(), "ffma, input {i}");
        assert_eq!(
            got.regs[6],
            (a.abs() + -b).to_bits(),
            "fadd with modifiers, input {i}"
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
