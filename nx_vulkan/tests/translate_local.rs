//! Local memory against the GPU: `STL` and `LDL` of every size, at constant and computed
//! addresses, under a guard (docs/ASSUMPTIONS.md A-160).

mod common;

use common::{random_states, run, vulkan};
use nx_shader::asm::{exit, mov32i, Enc, P0};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;

/// `STL.size [Ra + imm], Rd`.
fn stl(rd: u8, ra: u8, imm: i32, size: u64) -> Enc {
    Enc::op(Op::Stl)
        .rd(rd)
        .ra(ra)
        .field(20, 24, (imm as u64) & 0xFF_FFFF)
        .field(48, 3, size)
}

/// `LDL.size Rd, [Ra + imm]`.
fn ldl(rd: u8, ra: u8, imm: i32, size: u64) -> Enc {
    Enc::op(Op::Ldl)
        .rd(rd)
        .ra(ra)
        .field(20, 24, (imm as u64) & 0xFF_FFFF)
        .field(48, 3, size)
}

const U8: u64 = 0;
const S8: u64 = 1;
const U16: u64 = 2;
const S16: u64 = 3;
const B32: u64 = 4;
const B64: u64 = 5;
const B128: u64 = 6;

#[test]
fn words_pairs_quads_and_parts_of_words_round_trip() {
    let Some(vk) = vulkan() else { return };
    let program = [
        stl(1, RZ, 0, B32).build(),
        stl(2, RZ, 4, B32).build(),
        stl(1, RZ, 8, B64).build(),
        // R4..R7 = words 0..3 = R1, R2, R1, R2.
        ldl(4, RZ, 0, B128).build(),
        mov32i(8, 16),
        // Words 4, 5 and 6, then a byte into word 5 and a half word into word 6, by a computed
        // address.
        stl(3, 8, 0, B32).build(),
        stl(1, 8, 4, B32).build(),
        stl(3, 8, 8, B32).build(),
        stl(2, 8, 5, U8).build(),
        stl(2, 8, 10, U16).build(),
        ldl(9, RZ, 20, B32).build(),
        ldl(10, RZ, 24, B32).build(),
        ldl(11, 8, 5, U8).build(),
        ldl(12, 8, 5, S8).build(),
        ldl(13, 8, 10, U16).build(),
        ldl(14, 8, 10, S16).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(64, 71);
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let (r1, r2, r3) = (inp.regs[1], inp.regs[2], inp.regs[3]);
        assert_eq!(
            &got.regs[4..8],
            &[r1, r2, r1, r2],
            "a quad of words, input {i}"
        );
        let word5 = (r1 & !0xFF00) | (r2 & 0xFF) << 8;
        let word6 = (r3 & 0xFFFF) | (r2 & 0xFFFF) << 16;
        assert_eq!(got.regs[9], word5, "a byte stored into a word, input {i}");
        assert_eq!(
            got.regs[10], word6,
            "a half word stored into a word, input {i}"
        );
        let byte = (word5 >> 8) & 0xFF;
        assert_eq!(got.regs[11], byte, "u8, input {i}");
        assert_eq!(
            got.regs[12], byte as u8 as i8 as i32 as u32,
            "s8, input {i}"
        );
        let half = (word6 >> 16) & 0xFFFF;
        assert_eq!(got.regs[13], half, "u16, input {i}");
        assert_eq!(
            got.regs[14], half as u16 as i16 as i32 as u32,
            "s16, input {i}"
        );
    }
}

#[test]
fn a_guarded_store_only_happens_when_the_guard_holds() {
    let Some(vk) = vulkan() else { return };
    let program = [
        stl(2, RZ, 0, B32).build(),
        stl(1, RZ, 0, B32).guard(P0).build(),
        ldl(4, RZ, 0, B32).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(64, 72);
    let out = run(&vk, &program, &inputs, &[]);
    for (i, (inp, got)) in inputs.iter().zip(&out).enumerate() {
        let expect = if inp.preds & 1 != 0 {
            inp.regs[1]
        } else {
            inp.regs[2]
        };
        assert_eq!(got.regs[4], expect, "guarded store, input {i}");
    }
}
