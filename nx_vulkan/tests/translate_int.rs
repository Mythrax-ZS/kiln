//! The multiply-add family and the bit instructions of Maxwell, against the GPU.

mod common;

use common::{random_states, run, vulkan, Rng, State};
use nx_shader::asm::{exit, Enc};
use nx_shader::inst::{Guard, PT, RZ};
use nx_shader::ops::Op;

/// `XMAD` with registers. `flags` are `(bit, value)` pairs on top of the operands.
fn xmad_reg(rd: u8, ra: u8, rb: u8, rc: u8) -> Enc {
    Enc::op(Op::XmadReg).rd(rd).ra(ra).rb(rb).rc(rc)
}

#[test]
fn three_xmads_make_a_32_bit_product() {
    let Some(vk) = vulkan() else { return };
    // The compiler's sequence for R1 * R2:
    //   XMAD R3, R1, R2, RZ
    //   XMAD.MRG R4, R1, R2.H1, RZ
    //   XMAD.PSL.CBCC R5, R1.H1, R4.H1, R3
    let program = [
        xmad_reg(3, 1, 2, RZ).build(),
        xmad_reg(4, 1, 2, RZ).bit(35, true).bit(37, true).build(),
        xmad_reg(5, 1, 4, 3)
            .bit(53, true)
            .bit(35, true)
            .bit(36, true)
            .field(50, 3, 4)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let mut inputs = random_states(500, 91);
    for (a, b) in [
        (u32::MAX, u32::MAX),
        (0, u32::MAX),
        (0x10000, 0x10000),
        (0xFFFF, 0xFFFF),
        (0x8000_0000, 2),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        inputs.push(s);
    }
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b) = (inp.regs[1], inp.regs[2]);
        assert_eq!(got.regs[5], a.wrapping_mul(b), "{a:#x} * {b:#x}");
    }
}

#[test]
fn xmad_halves_types_modes_and_forms() {
    let Some(vk) = vulkan() else { return };
    let mut rng = Rng(5);
    let cbuf: Vec<u32> = (0..16).map(|_| rng.next()).collect();
    let mut inputs = random_states(200, 92);
    for s in &mut inputs {
        s.regs[7] = rng.next();
    }
    let program = [
        // Unsigned halves, low of both, plus R7.
        xmad_reg(3, 1, 2, 7).build(),
        // Signed times signed (bits 48 and 49), high half of A, low of B.
        xmad_reg(4, 1, 2, 7)
            .bit(48, true)
            .bit(49, true)
            .bit(53, true)
            .build(),
        // Signed A, unsigned B, high of B.
        xmad_reg(5, 1, 2, 7).bit(48, true).bit(35, true).build(),
        // CLO and CHI: the addend's low or high half.
        xmad_reg(6, 1, 2, 7).field(50, 3, 1).build(),
        xmad_reg(8, 1, 2, 7).field(50, 3, 2).build(),
        // PSL: the product shifted by 16.
        xmad_reg(9, 1, 2, 7).bit(36, true).build(),
        // MRG: the low half of B in the high half of the result.
        xmad_reg(10, 1, 2, 7).bit(37, true).build(),
        // The 16-bit immediate as B.
        Enc::op(Op::Xmad)
            .rd(11)
            .ra(1)
            .rc(7)
            .field(20, 16, 0xBEEF)
            .build(),
        // B from constant buffer word 3, high half of it, addend R7.
        Enc::op(Op::XmadCbuf_4e00)
            .rd(12)
            .ra(1)
            .rc(7)
            .cbuf(0, 12)
            .bit(52, true)
            .build(),
        // B in a register and the addend from the constant buffer (word 5).
        Enc::op(Op::XmadCbuf_5100)
            .rd(13)
            .ra(1)
            .rc(2)
            .cbuf(0, 20)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let out = run_cbuf(&vk, &program, &inputs, &cbuf);
    let lo = |v: u32| v & 0xFFFF;
    let hi = |v: u32| v >> 16;
    let s16 = |v: u32| i32::from(v as u16 as i16);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b, c) = (inp.regs[1], inp.regs[2], inp.regs[7]);
        let msg = format!("a {a:#x} b {b:#x} c {c:#x}");
        assert_eq!(got.regs[3], (lo(a) * lo(b)).wrapping_add(c), "plain, {msg}");
        let expect = (s16(a >> 16).wrapping_mul(s16(b))) as u32;
        assert_eq!(got.regs[4], expect.wrapping_add(c), "signed, {msg}");
        let expect = (s16(a)).wrapping_mul(hi(b) as i32) as u32;
        assert_eq!(
            got.regs[5],
            expect.wrapping_add(c),
            "signed by unsigned, {msg}"
        );
        assert_eq!(
            got.regs[6],
            (lo(a) * lo(b)).wrapping_add(lo(c)),
            "CLO, {msg}"
        );
        assert_eq!(
            got.regs[8],
            (lo(a) * lo(b)).wrapping_add(hi(c)),
            "CHI, {msg}"
        );
        assert_eq!(
            got.regs[9],
            ((lo(a) * lo(b)) << 16).wrapping_add(c),
            "PSL, {msg}"
        );
        let merged = ((lo(a) * lo(b)).wrapping_add(c) & 0xFFFF) | (lo(b) << 16);
        assert_eq!(got.regs[10], merged, "MRG, {msg}");
        assert_eq!(
            got.regs[11],
            (lo(a) * 0xBEEF).wrapping_add(c),
            "immediate, {msg}"
        );
        assert_eq!(
            got.regs[12],
            (lo(a) * hi(cbuf[3])).wrapping_add(c),
            "constant buffer B, {msg}"
        );
        assert_eq!(
            got.regs[13],
            (lo(a) * lo(b)).wrapping_add(cbuf[5]),
            "constant buffer addend, {msg}"
        );
    }
}

fn run_cbuf(vk: &nx_vulkan::Vk, program: &[u64], inputs: &[State], cbuf: &[u32]) -> Vec<State> {
    common::run_with(vk, program, inputs, cbuf, common::Extras::default()).states
}

#[test]
fn imad_multiplies_and_adds_with_negations() {
    let Some(vk) = vulkan() else { return };
    let inputs = random_states(300, 93);
    let program = [
        Enc::op(Op::ImadReg).rd(3).ra(1).rb(2).rc(4).build(),
        Enc::op(Op::ImadReg)
            .rd(5)
            .ra(1)
            .rb(2)
            .rc(4)
            .bit(51, true)
            .build(),
        Enc::op(Op::ImadReg)
            .rd(6)
            .ra(1)
            .rb(2)
            .rc(4)
            .bit(52, true)
            .build(),
        Enc::op(Op::ImadImm20).rd(7).ra(1).rc(4).imm20(-123).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b, c) = (inp.regs[1], inp.regs[2], inp.regs[4]);
        let p = a.wrapping_mul(b);
        assert_eq!(got.regs[3], p.wrapping_add(c), "IMAD");
        assert_eq!(
            got.regs[5],
            p.wrapping_neg().wrapping_add(c),
            "IMAD with -(a*b)"
        );
        assert_eq!(got.regs[6], p.wrapping_sub(c), "IMAD with -c");
        assert_eq!(
            got.regs[7],
            a.wrapping_mul(-123i32 as u32).wrapping_add(c),
            "IMAD immediate"
        );
    }
}

fn lea(rd: u8, ra: u8, rb: u8, scale: u64) -> Enc {
    Enc::op(Op::LeaReg)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .field(39, 5, scale)
        .field(48, 3, u64::from(PT))
}

fn lea_hi(rd: u8, ra: u8, rb: u8, rc: u8, scale: u64) -> Enc {
    Enc::op(Op::LeaHiReg)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .rc(rc)
        .field(28, 5, scale)
        .field(48, 3, u64::from(PT))
}

#[test]
fn lea_adds_a_scaled_index_and_makes_64_bit_sums() {
    let Some(vk) = vulkan() else { return };
    let inputs = random_states(400, 94);
    let program = [
        lea(3, 1, 2, 3).build(),
        lea(4, 1, 2, 31).bit(45, true).build(),
        // The 64-bit sum (R7:R2) + (R1 << 4): the low word sets the carry, the high word takes it.
        lea(5, 1, 2, 4).bit(47, true).build(),
        lea_hi(6, 1, 7, RZ, 4).bit(38, true).build(),
        // LEA.HI without the carry: R7 + ((R8:R1) >> 28).
        lea_hi(9, 1, 7, 8, 4).build(),
        lea_hi(10, 1, 7, 8, 0).bit(37, true).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b, hi_b, rc) = (inp.regs[1], inp.regs[2], inp.regs[7], inp.regs[8]);
        let msg = format!("a {a:#x} b {b:#x}");
        assert_eq!(got.regs[3], b.wrapping_add(a << 3), "LEA, {msg}");
        assert_eq!(
            got.regs[4],
            b.wrapping_add(a.wrapping_neg() << 31),
            "LEA with -a, {msg}"
        );
        let wide = ((u64::from(hi_b) << 32) | u64::from(b)).wrapping_add(u64::from(a) << 4);
        assert_eq!(got.regs[5], wide as u32, "64-bit sum, low word, {msg}");
        assert_eq!(
            got.regs[6],
            (wide >> 32) as u32,
            "64-bit sum, high word, {msg}"
        );
        let packed = (u64::from(rc) << 32) | u64::from(a);
        assert_eq!(
            got.regs[9],
            hi_b.wrapping_add((packed >> 28) as u32),
            "LEA.HI, {msg}"
        );
        assert_eq!(
            got.regs[10],
            hi_b.wrapping_add((packed.wrapping_neg() >> 32) as u32),
            "LEA.HI with a negated pair and no scale, {msg}"
        );
    }
}

#[test]
fn lop3_follows_the_truth_table() {
    let Some(vk) = vulkan() else { return };
    let inputs = random_states(40, 95);
    for lut in 0..=255u64 {
        let program = [
            Enc::op(Op::Lop3LutReg)
                .rd(3)
                .ra(1)
                .rb(2)
                .rc(4)
                .field(28, 8, lut)
                .field(48, 3, u64::from(PT))
                .build(),
            Enc::op(Op::Lop3LutImm20)
                .rd(5)
                .ra(1)
                .rc(4)
                .imm20(-77)
                .field(48, 8, lut)
                .build(),
            exit(Guard::ALWAYS),
        ];
        let out = run(&vk, &program, &inputs, &[]);
        for (inp, got) in inputs.iter().zip(&out) {
            let (a, b, c) = (inp.regs[1], inp.regs[2], inp.regs[4]);
            let table = |a: u32, b: u32, c: u32| {
                (0..32).fold(0u32, |r, i| {
                    let index = (a >> i & 1) << 2 | (b >> i & 1) << 1 | (c >> i & 1);
                    r | ((lut >> index & 1) as u32) << i
                })
            };
            assert_eq!(got.regs[3], table(a, b, c), "lut {lut:#04x}");
            assert_eq!(
                got.regs[5],
                table(a, -77i32 as u32, c),
                "lut {lut:#04x} immediate"
            );
        }
    }
}

#[test]
fn lop3_sets_a_predicate_from_the_result() {
    let Some(vk) = vulkan() else { return };
    let inputs = random_states(100, 96);
    // AND of the three: mostly zero. Predicates: P0 ← zero, P1 ← not zero, P2 ← true, P3 ← false.
    let lop3 = |pred: u64, mode: u64| {
        Enc::op(Op::Lop3LutReg)
            .rd(3)
            .ra(1)
            .rb(2)
            .rc(4)
            .field(28, 8, 0x80)
            .field(36, 2, mode)
            .field(48, 3, pred)
    };
    let program = [
        lop3(0, 2).build(),
        lop3(1, 3).build(),
        lop3(2, 1).build(),
        lop3(3, 0).build(),
        exit(Guard::ALWAYS),
    ];
    let mut inputs = inputs;
    for s in inputs.iter_mut().take(30) {
        // Make the AND nonzero in these.
        s.regs[1] |= 0x0F0F_0F0F;
        s.regs[2] |= 0x0F0F_0F0F;
        s.regs[4] |= 0x0F0F_0F0F;
        s.preds = 0b1000;
    }
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let r = inp.regs[1] & inp.regs[2] & inp.regs[4];
        assert_eq!(got.regs[3], r);
        assert_eq!(got.preds & 1 != 0, r == 0, "zero");
        assert_eq!(got.preds & 2 != 0, r != 0, "not zero");
        assert!(got.preds & 4 != 0, "true");
        assert!(got.preds & 8 == 0, "false");
    }
}

/// The funnel shift as a plain 64-bit computation.
fn shf_model(
    low: u32,
    high: u32,
    shift: u32,
    right: bool,
    wrap: bool,
    kind: u64,
    dst_high: bool,
) -> u32 {
    let bits: u32 = if kind >= 2 { 64 } else { 32 };
    let shift = if wrap {
        shift & (bits - 1)
    } else {
        shift.min(bits)
    };
    let x = u64::from(low) | u64::from(high) << 32;
    let shifted = if right {
        if kind == 3 {
            ((x as i64) >> shift.min(63)) as u64
        } else {
            x.checked_shr(shift).unwrap_or(0)
        }
    } else {
        x.checked_shl(shift).unwrap_or(0)
    };
    if !right || dst_high {
        (shifted >> 32) as u32
    } else {
        shifted as u32
    }
}

#[test]
fn funnel_shifts_match_the_64_bit_model() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(300, 97);
    let mut rng = Rng(98);
    for s in &mut inputs {
        s.regs[2] = rng.next() % 70;
    }
    for right in [false, true] {
        for wrap in [false, true] {
            for kind in [0u64, 2, 3] {
                for dst_high in [false, true] {
                    if !right && dst_high {
                        // The hardware rejects it.
                        continue;
                    }
                    let base = if right { Op::ShfRReg } else { Op::ShfLReg };
                    let program = [
                        Enc::op(base)
                            .rd(3)
                            .ra(1)
                            .rb(2)
                            .rc(4)
                            .field(37, 2, kind)
                            .bit(48, dst_high)
                            .bit(50, wrap)
                            .build(),
                        exit(Guard::ALWAYS),
                    ];
                    let out = run(&vk, &program, &inputs, &[]);
                    for (inp, got) in inputs.iter().zip(&out) {
                        let (low, high, n) = (inp.regs[1], inp.regs[4], inp.regs[2]);
                        // A 32-bit type shifts by at most 32, so only those counts are comparable
                        // when the shift is not wrapped; the wide types go up to 64.
                        assert_eq!(
                            got.regs[3],
                            shf_model(low, high, n, right, wrap, kind, dst_high),
                            "right {right} wrap {wrap} kind {kind} high {dst_high}: ({high:#x}:{low:#x}) by {n}"
                        );
                    }
                }
            }
        }
    }
    // The immediate forms.
    let program = [
        Enc::op(Op::ShfR)
            .rd(3)
            .ra(1)
            .rc(4)
            .field(20, 6, 13)
            .field(37, 2, 2)
            .bit(50, true)
            .build(),
        Enc::op(Op::ShfLImm20)
            .rd(5)
            .ra(1)
            .rc(4)
            .imm20(7)
            .bit(50, true)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (low, high) = (inp.regs[1], inp.regs[4]);
        assert_eq!(
            got.regs[3],
            shf_model(low, high, 13, true, true, 2, false),
            "SHF.R.U64 imm"
        );
        assert_eq!(
            got.regs[5],
            shf_model(low, high, 7, false, true, 0, false),
            "SHF.L imm"
        );
    }
}

/// `PRMT` as the PTX `prmt` instruction defines the modes it supports.
fn prmt_model(a: u32, b: u32, sel: u32, mode: u64) -> u32 {
    let bytes: [u8; 8] = {
        let mut x = [0u8; 8];
        x[..4].copy_from_slice(&a.to_le_bytes());
        x[4..].copy_from_slice(&b.to_le_bytes());
        x
    };
    let mut out = [0u8; 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = match mode {
            0 => {
                let nibble = (sel >> (4 * i)) & 0xF;
                let byte = bytes[(nibble & 7) as usize];
                if nibble & 8 != 0 {
                    if byte & 0x80 != 0 {
                        0xFF
                    } else {
                        0
                    }
                } else {
                    byte
                }
            }
            1 => bytes[((sel & 3) as usize + i) & 7],
            2 => bytes[((sel & 3) as usize).wrapping_sub(i) & 7],
            _ => bytes[(sel & 3) as usize],
        };
    }
    u32::from_le_bytes(out)
}

#[test]
fn prmt_permutes_bytes_in_each_mode() {
    let Some(vk) = vulkan() else { return };
    let inputs = random_states(300, 99);
    let cbuf: Vec<u32> = vec![0x7654, 0x3210];
    for mode in 0..4u64 {
        let program = [
            Enc::op(Op::PrmtReg)
                .rd(3)
                .ra(1)
                .rb(2)
                .rc(4)
                .field(48, 3, mode)
                .build(),
            Enc::op(Op::PrmtImm20)
                .rd(5)
                .ra(1)
                .rc(4)
                .imm20(0x5432)
                .field(48, 3, mode)
                .build(),
            Enc::op(Op::PrmtCbuf_4bc0)
                .rd(6)
                .ra(1)
                .rc(4)
                .cbuf(0, 4)
                .field(48, 3, mode)
                .build(),
            exit(Guard::ALWAYS),
        ];
        let out = run_cbuf(&vk, &program, &inputs, &cbuf);
        for (inp, got) in inputs.iter().zip(&out) {
            let (a, sel, b) = (inp.regs[1], inp.regs[2], inp.regs[4]);
            assert_eq!(
                got.regs[3],
                prmt_model(a, b, sel, mode),
                "mode {mode}, sel {sel:#x}"
            );
            assert_eq!(
                got.regs[5],
                prmt_model(a, b, 0x5432, mode),
                "mode {mode}, immediate"
            );
            assert_eq!(
                got.regs[6],
                prmt_model(a, b, cbuf[1], mode),
                "mode {mode}, constant buffer"
            );
        }
    }
}
