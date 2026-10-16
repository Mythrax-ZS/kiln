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

// ---- IADD3, the condition code flags and CSETP ----------------------------------------------

/// A word cut to a half as the selector of `IADD3` says (1 low, 2 high, else whole).
fn cut(v: u32, selector: u32) -> u32 {
    match selector {
        1 => v & 0xFFFF,
        2 => v >> 16,
        _ => v,
    }
}

fn iadd3_reg(rd: u8, ra: u8, rb: u8, rc: u8) -> Enc {
    Enc::op(Op::Iadd3Reg).rd(rd).ra(ra).rb(rb).rc(rc)
}

#[test]
fn iadd3_adds_three_words_with_halves_negations_and_shifts() {
    let Some(vk) = vulkan() else { return };
    let mut rng = Rng(77);
    let cbuf: Vec<u32> = (0..16).map(|_| rng.next()).collect();
    let inputs = random_states(300, 311);
    let program = [
        // The plain sum.
        iadd3_reg(4, 1, 2, 3).build(),
        // -Ra + B - Rc (bits 51 and 49).
        iadd3_reg(5, 1, 2, 3).bit(51, true).bit(49, true).build(),
        // The low half of Ra (35..36 = 1), the high half of B (33..34 = 2), the low half of Rc.
        iadd3_reg(6, 1, 2, 3)
            .field(35, 2, 1)
            .field(33, 2, 2)
            .field(31, 2, 1)
            .build(),
        // The sum of the first two moved right by 16 (37..38 = 1), then Rc added.
        iadd3_reg(7, 1, 2, 3).field(37, 2, 1).build(),
        // And left by 16 (37..38 = 2).
        iadd3_reg(8, 1, 2, 3).field(37, 2, 2).build(),
        // B from constant buffer 0, and B an immediate.
        Enc::op(Op::Iadd3Cbuf).rd(9).ra(1).rc(3).cbuf(0, 24).build(),
        Enc::op(Op::Iadd3Imm20)
            .rd(10)
            .ra(1)
            .rc(3)
            .imm20(-1000)
            .build(),
        // The high half of Rc with the shift: halves come before negation and the shift.
        iadd3_reg(11, 1, 2, 3)
            .field(31, 2, 2)
            .bit(50, true)
            .field(37, 2, 2)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let out = run_cbuf(&vk, &program, &inputs, &cbuf);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b, c) = (inp.regs[1], inp.regs[2], inp.regs[3]);
        assert_eq!(got.regs[4], a.wrapping_add(b).wrapping_add(c), "plain");
        assert_eq!(
            got.regs[5],
            a.wrapping_neg().wrapping_add(b).wrapping_sub(c),
            "negated"
        );
        assert_eq!(
            got.regs[6],
            cut(a, 1).wrapping_add(cut(b, 2)).wrapping_add(cut(c, 1)),
            "halves"
        );
        assert_eq!(
            got.regs[7],
            (a.wrapping_add(b) >> 16).wrapping_add(c),
            "shifted right"
        );
        assert_eq!(
            got.regs[8],
            (a.wrapping_add(b) << 16).wrapping_add(c),
            "shifted left"
        );
        assert_eq!(
            got.regs[9],
            a.wrapping_add(cbuf[6]).wrapping_add(c),
            "constant buffer"
        );
        assert_eq!(
            got.regs[10],
            a.wrapping_add(-1000i32 as u32).wrapping_add(c),
            "immediate"
        );
        let nb = b.wrapping_neg();
        assert_eq!(
            got.regs[11],
            (a.wrapping_add(nb) << 16).wrapping_add(cut(c, 2)),
            "halves, negation and shift together"
        );
    }
}

/// `CSETP.test.op Pd, Pq, CC.cc, Pc` with `Pc` = PT: `pd` gets the test, `Pq` (PT) is dropped.
fn csetp(pd: u8, cc: u64) -> Enc {
    Enc::op(Op::CsetpCc)
        .field(0, 3, u64::from(PT))
        .field(3, 3, u64::from(pd))
        .field(8, 5, cc)
        .field(39, 3, u64::from(PT))
}

/// The four flags `(zero, sign, carry, overflow)` of `a + b + carry_in`.
fn add_flags(a: u32, b: u32, carry_in: bool) -> (bool, bool, bool, bool) {
    let wide = u64::from(a) + u64::from(b) + u64::from(carry_in);
    let sum = wide as u32;
    let signed = i64::from(a as i32) + i64::from(b as i32) + i64::from(carry_in);
    (
        sum == 0,
        sum >> 31 != 0,
        wide >> 32 != 0,
        signed != i64::from(sum as i32),
    )
}

/// The condition codes of `CSETP`, evaluated from the flags as the table says.
fn condition(code: u64, (z, s, c, o): (bool, bool, bool, bool)) -> Option<bool> {
    Some(match code {
        0 => false,
        1 => (s && !z) != o,
        2 => !s && z,
        3 => s != (z || o),
        4 => !((s != o) || z),
        5 => !z,
        6 => s == o,
        7 => !(s && z),
        8 => s && z,
        9 => s != o,
        10 => z,
        11 => (s != o) || z,
        12 => (!s) != (o || z),
        13 => s || !z,
        14 => ((!s) || z) != o,
        15 => true,
        16 => !o,
        17 => !c,
        18 => !s,
        19 => z || !c,
        20 => c && !z,
        21 => s,
        22 => c,
        23 => o,
        30 => s || z,
        31 => !(s || z),
        _ => return None,
    })
}

#[test]
fn iadd_with_cc_sets_the_flags_and_csetp_tests_them() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(200, 313);
    // The edges of an addition: carries, overflows and zeros.
    for (a, b) in [
        (0u32, 0u32),
        (1, u32::MAX),
        (u32::MAX, u32::MAX),
        (0x7FFF_FFFF, 1),
        (0x8000_0000, 0x8000_0000),
        (0x8000_0000, 0x7FFF_FFFF),
        (5, 5u32.wrapping_neg()),
        (0x4000_0000, 0x4000_0000),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        inputs.push(s);
    }
    for code in (0..32u64).filter(|c| condition(*c, (false, false, false, false)).is_some()) {
        let program = [
            Enc::op(Op::IaddReg).rd(3).ra(1).rb(2).bit(47, true).build(),
            csetp(0, code).build(),
            exit(Guard::ALWAYS),
        ];
        let out = run(&vk, &program, &inputs, &[]);
        for (inp, got) in inputs.iter().zip(&out) {
            let (a, b) = (inp.regs[1], inp.regs[2]);
            assert_eq!(got.regs[3], a.wrapping_add(b));
            let want = condition(code, add_flags(a, b, false)).unwrap();
            assert_eq!(
                got.preds & 1 != 0,
                want,
                "condition {code} after {a:#x} + {b:#x}"
            );
        }
    }
}

#[test]
fn iscadd_with_cc_sets_the_flags_of_the_addition_of_the_shifted_operand() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(200, 331);
    // Shifted operands that carry and overflow, and sums that are zero.
    for (a, b) in [
        (0x2000_0000u32, 0x8000_0000u32),
        (0x1000_0000, 0x7FFF_FFFF),
        (0x0000_0003, 0u32.wrapping_sub(24)),
        (0, 0),
        (u32::MAX, 1),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        inputs.push(s);
    }
    for (shift, negate_b) in [(3u64, false), (0, false), (31, false), (2, true)] {
        for code in [2u64, 10, 17, 21, 22, 23, 1, 4, 6] {
            let mut add = Enc::op(Op::IscaddReg)
                .rd(3)
                .ra(1)
                .rb(2)
                .field(39, 5, shift)
                .bit(47, true);
            if negate_b {
                add = add.bit(48, true);
            }
            let program = [add.build(), csetp(0, code).build(), exit(Guard::ALWAYS)];
            let out = run(&vk, &program, &inputs, &[]);
            for (inp, got) in inputs.iter().zip(&out) {
                let a = inp.regs[1] << shift;
                let b = inp.regs[2];
                let (want, flags) = if negate_b {
                    (a.wrapping_sub(b), add_flags(a, !b, true))
                } else {
                    (a.wrapping_add(b), add_flags(a, b, false))
                };
                assert_eq!(got.regs[3], want, "{a:#x} {b:#x} shift {shift}");
                assert_eq!(
                    got.preds & 1 != 0,
                    condition(code, flags).unwrap(),
                    "condition {code} after ({:#x} << {shift}) {} {b:#x}",
                    inp.regs[1],
                    if negate_b { "-" } else { "+" }
                );
            }
        }
    }
}

#[test]
fn a_subtraction_with_cc_sets_the_flags_of_a_minus_b() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(200, 317);
    for (a, b) in [
        (5u32, 5u32),
        (0, 1),
        (0x8000_0000, 1),
        (1, 0x8000_0000),
        (7, 0),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        inputs.push(s);
    }
    // IADD.CC with B negated (bit 48): the flags are those of `a + !b + 1`.
    for code in [2u64, 10, 17, 21, 22, 23, 1, 4, 6] {
        let program = [
            Enc::op(Op::IaddReg)
                .rd(3)
                .ra(1)
                .rb(2)
                .bit(48, true)
                .bit(47, true)
                .build(),
            csetp(0, code).build(),
            exit(Guard::ALWAYS),
        ];
        let out = run(&vk, &program, &inputs, &[]);
        for (inp, got) in inputs.iter().zip(&out) {
            let (a, b) = (inp.regs[1], inp.regs[2]);
            assert_eq!(got.regs[3], a.wrapping_sub(b));
            let want = condition(code, add_flags(a, !b, true)).unwrap();
            assert_eq!(
                got.preds & 1 != 0,
                want,
                "condition {code} after {a:#x} - {b:#x}"
            );
        }
    }
}

#[test]
fn iadd_x_carries_the_flag_and_keeps_the_zero_flag_across_words() {
    let Some(vk) = vulkan() else { return };
    // A 64-bit add: IADD.CC R4, R1, R2 then IADD.X.CC R5, R3, R6; the zero flag after the
    // second only holds if both words of the sum are zero.
    let mut inputs = random_states(300, 319);
    for (a, b, c, d) in [
        (0u32, 0u32, 0u32, 0u32),
        (1, u32::MAX, 0, 0),
        (1, u32::MAX, u32::MAX, 0),
        (0, 0, 7, 7u32.wrapping_neg()),
        (3, 4, 5, 6),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        s.regs[3] = c;
        s.regs[6] = d;
        inputs.push(s);
    }
    let program = [
        Enc::op(Op::IaddReg).rd(4).ra(1).rb(2).bit(47, true).build(),
        Enc::op(Op::IaddReg)
            .rd(5)
            .ra(3)
            .rb(6)
            .bit(47, true)
            .bit(43, true)
            .build(),
        csetp(0, 10).build(),
        csetp(1, 22).build(),
        csetp(2, 21).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b, c, d) = (inp.regs[1], inp.regs[2], inp.regs[3], inp.regs[6]);
        let low = add_flags(a, b, false);
        let high = add_flags(c, d, low.2);
        let sum_lo = a.wrapping_add(b);
        let sum_hi = c.wrapping_add(d).wrapping_add(u32::from(low.2));
        assert_eq!(got.regs[4], sum_lo);
        assert_eq!(got.regs[5], sum_hi);
        // Zero only when both words are; carry and sign are the high word's.
        assert_eq!(got.preds & 1 != 0, sum_lo == 0 && sum_hi == 0, "zero");
        assert_eq!(got.preds >> 1 & 1 != 0, high.2, "carry");
        assert_eq!(got.preds >> 2 & 1 != 0, sum_hi >> 31 != 0, "sign");
    }
}

#[test]
fn iadd3_with_cc_and_x_makes_a_three_word_sum_with_flags() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(300, 323);
    for (a, b, c) in [
        (0u32, 0u32, 0u32),
        (u32::MAX, 1, 0),
        (u32::MAX, 0, 1),
        (u32::MAX, u32::MAX, u32::MAX),
        (1, u32::MAX, 0),
    ] {
        let mut s = State::default();
        s.regs[1] = a;
        s.regs[2] = b;
        s.regs[3] = c;
        inputs.push(s);
    }
    let program = [
        // IADD3.CC R4, R1, R2, R3; then the extended form on R5 = R1 + R2 + R3 + carry.
        iadd3_reg(4, 1, 2, 3).bit(47, true).build(),
        csetp(0, 22).build(),
        csetp(1, 10).build(),
        csetp(2, 21).build(),
        iadd3_reg(5, 1, 2, 3).bit(48, true).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let (a, b, c) = (inp.regs[1], inp.regs[2], inp.regs[3]);
        let ab = u64::from(a) + u64::from(b);
        let abc = u64::from(ab as u32) + u64::from(c);
        let sum = a.wrapping_add(b).wrapping_add(c);
        let carried = ab >> 32 != 0 || abc >> 32 != 0;
        assert_eq!(got.regs[4], sum);
        assert_eq!(got.preds & 1 != 0, carried, "carry of {a:#x} {b:#x} {c:#x}");
        assert_eq!(got.preds >> 1 & 1 != 0, sum == 0, "zero");
        assert_eq!(got.preds >> 2 & 1 != 0, sum >> 31 != 0, "sign");
        assert_eq!(
            got.regs[5],
            sum.wrapping_add(u32::from(carried)),
            ".X adds the carry the last instruction left"
        );
    }
}

#[test]
fn i2i_with_cc_sets_zero_and_sign_and_clears_carry_and_overflow() {
    let Some(vk) = vulkan() else { return };
    let mut inputs = random_states(300, 329);
    for v in [0u32, 1, 0x8000_0000, u32::MAX] {
        let mut s = State::default();
        s.regs[2] = v;
        inputs.push(s);
    }
    let program = [
        // Set the carry and the overflow flags first: 0x80000000 + 0x80000000 does both.
        nx_shader::asm::mov32i(8, 0x8000_0000),
        Enc::op(Op::IaddReg).rd(3).ra(8).rb(8).bit(47, true).build(),
        // I2I.CC RZ, R2 (32-bit to 32-bit): the flags come from R2's value.
        Enc::op(Op::I2iReg)
            .rd(RZ)
            .rb(2)
            .field(8, 2, 2)
            .field(10, 2, 2)
            .bit(47, true)
            .build(),
        csetp(0, 10).build(),
        csetp(1, 21).build(),
        csetp(2, 22).build(),
        csetp(3, 23).build(),
        exit(Guard::ALWAYS),
    ];
    let out = run(&vk, &program, &inputs, &[]);
    for (inp, got) in inputs.iter().zip(&out) {
        let v = inp.regs[2];
        assert_eq!(got.preds & 1 != 0, v == 0, "zero of {v:#x}");
        assert_eq!(got.preds >> 1 & 1 != 0, v >> 31 != 0, "sign of {v:#x}");
        assert_eq!(got.preds >> 2 & 1, 0, "carry cleared");
        assert_eq!(got.preds >> 3 & 1, 0, "overflow cleared");
    }
}
