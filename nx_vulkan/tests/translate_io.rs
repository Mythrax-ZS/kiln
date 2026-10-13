//! Attributes, constant buffers and special registers in the harness mode, against the GPU.

mod common;

use common::{random_states, run_with, vulkan, Extras, Rng, State};
use nx_shader::asm::{exit, Enc, P0};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;

/// `ALD`: `words` consecutive attribute words at byte address `addr` into `rd...`.
fn ald(rd: u8, addr: u32, words: u64) -> Enc {
    Enc::op(Op::Ald)
        .rd(rd)
        .ra(RZ)
        .field(20, 10, u64::from(addr))
        .field(47, 2, words - 1)
}

/// `AST`: the registers `rd...` to the attribute words at `addr`.
fn ast(rd: u8, addr: u32, words: u64) -> Enc {
    Enc::op(Op::Ast)
        .rd(rd)
        .ra(RZ)
        .field(20, 10, u64::from(addr))
        .field(47, 2, words - 1)
}

#[test]
fn attributes_move_between_the_buffers_and_registers() {
    let Some(vk) = vulkan() else { return };
    let program = [
        ald(3, 0x80, 1).build(),
        ald(4, 0x88, 2).build(),
        ald(6, 0x90, 4).build(),
        ast(3, 0xC0, 1).build(),
        ast(4, 0xC8, 2).build(),
        ast(6, 0xD0, 4).build(),
        ast(3, 0xE0, 1).guard(P0).build(),
        exit(Guard::ALWAYS),
    ];
    let mut rng = Rng(5);
    let inputs = random_states(48, 3);
    let attrs_in: Vec<[u32; 64]> = inputs
        .iter()
        .map(|_| {
            let mut w = [0u32; 64];
            for x in &mut w {
                *x = rng.next();
            }
            w
        })
        .collect();
    let extras = Extras {
        attrs_in: attrs_in.clone(),
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (inp, got)) in inputs.iter().zip(&out.states).enumerate() {
        let a = &attrs_in[i];
        assert_eq!(got.regs[3], a[0x80 / 4], "one word, invocation {i}");
        assert_eq!(
            [got.regs[4], got.regs[5]],
            [a[0x88 / 4], a[0x88 / 4 + 1]],
            "two words"
        );
        assert_eq!(got.regs[6..10], a[0x90 / 4..0x90 / 4 + 4], "four words");
        let o = &out.attrs_out[i];
        assert_eq!(o[0xC0 / 4], a[0x80 / 4], "stored word, invocation {i}");
        assert_eq!(
            o[0xC8 / 4..0xC8 / 4 + 2],
            a[0x88 / 4..0x88 / 4 + 2],
            "stored pair"
        );
        assert_eq!(
            o[0xD0 / 4..0xD0 / 4 + 4],
            a[0x90 / 4..0x90 / 4 + 4],
            "stored quad"
        );
        let guarded = if inp.preds & 1 != 0 { a[0x80 / 4] } else { 0 };
        assert_eq!(o[0xE0 / 4], guarded, "guarded store, invocation {i}");
    }
}

/// `LDC.size`: `rd` from `c[0][Ra + imm]`; the size is 0 u8, 1 s8, 2 u16, 3 s16, 4 b32, 5 b64.
fn ldc(rd: u8, ra: u8, imm: i32, size: u64) -> Enc {
    Enc::op(Op::Ldc)
        .rd(rd)
        .ra(ra)
        .field(20, 16, imm as u32 as u64)
        .field(36, 5, 0)
        .field(44, 2, 0)
        .field(48, 3, size)
}

#[test]
fn constant_buffer_loads_by_size_and_by_register() {
    let Some(vk) = vulkan() else { return };
    let mut rng = Rng(8);
    let cbuf: Vec<u32> = (0..256).map(|_| rng.next()).collect();
    let bytes: Vec<u8> = cbuf.iter().flat_map(|w| w.to_le_bytes()).collect();
    let byte = |at: usize| bytes[at];
    let half = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let program = [
        ldc(3, RZ, 0x13, 0).build(),
        ldc(4, RZ, 0x21, 1).build(),
        ldc(5, RZ, 0x32, 2).build(),
        ldc(6, RZ, 0x44, 3).build(),
        ldc(7, RZ, 0x50, 4).build(),
        ldc(8, RZ, 0x58, 5).build(),
        // By register: R1 is a byte offset (a multiple of four for the word loads).
        ldc(10, 1, 0x10, 4).build(),
        ldc(11, 2, 0x10, 0).build(),
        ldc(12, 2, -4, 3).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs: Vec<State> = (0..64usize)
        .map(|i| {
            let mut s = State::default();
            s.regs[1] = ((i as u32 * 12) % 600) & !3;
            s.regs[2] = 8 + ((i as u32 * 7) % 350) * 2;
            s
        })
        .collect();
    let out = run_with(&vk, &program, &inputs, &cbuf, Extras::default());
    for (i, (inp, got)) in inputs.iter().zip(&out.states).enumerate() {
        assert_eq!(got.regs[3], u32::from(byte(0x13)), "u8, invocation {i}");
        assert_eq!(got.regs[4], i32::from(byte(0x21) as i8) as u32, "s8");
        assert_eq!(got.regs[5], u32::from(half(0x32)), "u16");
        assert_eq!(got.regs[6], i32::from(half(0x44) as i16) as u32, "s16");
        assert_eq!(got.regs[7], word(0x50), "b32");
        assert_eq!([got.regs[8], got.regs[9]], [word(0x58), word(0x5C)], "b64");
        let r1 = inp.regs[1] as usize;
        assert_eq!(
            got.regs[10],
            word(r1 + 0x10),
            "b32 by register, invocation {i}"
        );
        let r2 = inp.regs[2] as usize;
        assert_eq!(got.regs[11], u32::from(byte(r2 + 0x10)), "u8 by register");
        let at = r2 - 4;
        assert_eq!(
            got.regs[12],
            i32::from(half(at) as i16) as u32,
            "s16 by register"
        );
    }
}

#[test]
fn the_thread_id_register_is_the_invocation_index() {
    let Some(vk) = vulkan() else { return };
    let program = [
        Enc::op(Op::S2r).rd(3).field(20, 8, 0x21).build(),
        Enc::op(Op::S2r).rd(4).field(20, 8, 0x22).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = random_states(100, 9);
    let out = run_with(&vk, &program, &inputs, &[], Extras::default());
    for (i, got) in out.states.iter().enumerate() {
        assert_eq!(got.regs[3], i as u32, "SR_TID.X");
        assert_eq!(got.regs[4], 0, "SR_TID.Y");
    }
}
