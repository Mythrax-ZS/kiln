//! sm_86 programs run on the device in the harness mode: what the Switch 2's scoreboard means for
//! the values a program reads (docs/ASSUMPTIONS.md A-256).

mod common;

use common::{vulkan, State};
use nx_shader::translate::{test_bindings as tb, translate_sm86, Config, Stage};

/// An instruction with `fields` (lo, len, value) set.
fn word(fields: &[(u32, u32, u64)]) -> u128 {
    let mut v = 0u128;
    for &(lo, len, value) in fields {
        let mask = ((1u128 << len) - 1) << lo;
        v = (v & !mask) | ((u128::from(value) << lo) & mask);
    }
    v
}

const RZ: u64 = 255;

/// The scheduling fields: the write barrier set (7 none) and the barriers waited for.
fn sched(barrier: u64, wait: u64) -> [(u32, u32, u64); 3] {
    [(110, 3, barrier), (113, 3, 7), (116, 6, wait)]
}

fn with(mut fields: Vec<(u32, u32, u64)>, s: [(u32, u32, u64); 3]) -> u128 {
    fields.extend_from_slice(&s);
    word(&fields)
}

/// `MUFU.RCP Rd, Rs`.
fn rcp(d: u64, s: u64, barrier: u64, wait: u64) -> u128 {
    with(
        vec![
            (0, 12, 0x308),
            (12, 4, 7),
            (16, 8, d),
            (32, 8, s),
            (74, 4, 4),
        ],
        sched(barrier, wait),
    )
}

/// `FADD Rd, Ra, RZ`: a copy of `Ra` as a float.
fn copy(d: u64, a: u64, wait: u64) -> u128 {
    with(
        vec![
            (0, 12, 0x221),
            (12, 4, 7),
            (16, 8, d),
            (24, 8, a),
            (32, 8, RZ),
        ],
        sched(7, wait),
    )
}

/// `DEPBAR.LE SBn, count`.
fn depbar(barrier: u64, count: u64) -> u128 {
    with(
        vec![
            (0, 12, 0x91a),
            (12, 4, 7),
            (38, 6, count),
            (44, 3, barrier),
            (47, 1, 1),
        ],
        sched(7, 0),
    )
}

fn program(insts: &[u128]) -> Vec<u8> {
    let exit = word(&[(0, 12, 0x94d), (12, 4, 7), (87, 3, 7)]);
    let end = word(&[
        (0, 12, 0x947),
        (12, 4, 7),
        (87, 3, 7),
        (34, 48, (-4i64 as u64) & ((1 << 48) - 1)),
    ]);
    insts
        .iter()
        .chain([&exit, &end])
        .flat_map(|i| i.to_le_bytes())
        .collect()
}

/// Run an sm_86 program once on `input` and return the final registers.
fn run(vk: &nx_vulkan::Vk, code: &[u8], input: State) -> State {
    let config = Config::new(Stage::Compute);
    let t = translate_sm86(code, None, &config).unwrap_or_else(|e| panic!("translation: {e}"));
    nx_shader::validate::assert_valid(&t.spirv);
    let mut regs_in = Vec::new();
    for r in input.regs {
        regs_in.extend_from_slice(&r.to_le_bytes());
    }
    regs_in.extend_from_slice(&input.preds.to_le_bytes());
    let mut regs_out = [0u8; 16 * 4];
    let mut attr_in = vec![0u8; tb::ATTRS as usize * 4];
    let mut attr_out = vec![0u8; tb::ATTRS as usize * 4];
    let mut c0 = vec![0u8; 4096];
    let mut list: Vec<(u32, &mut [u8])> = vec![
        (tb::REGS_IN, &mut regs_in[..]),
        (tb::REGS_OUT, &mut regs_out[..]),
        (tb::ATTR_IN, &mut attr_in[..]),
        (tb::ATTR_OUT, &mut attr_out[..]),
        (64 * Stage::Compute.index(), &mut c0[..]),
    ];
    vk.run_compute_with(&t.spirv, &mut list, &[], &[], [1, 1, 1])
        .unwrap();
    let mut s = State::default();
    for (j, r) in s.regs.iter_mut().enumerate() {
        *r = u32::from_le_bytes(regs_out[j * 4..j * 4 + 4].try_into().unwrap());
    }
    s
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn a_register_holds_its_old_value_until_its_barrier_is_waited_for() {
    let Some(vk) = vulkan() else { return };
    // R1 = 1/R0 on barrier 0; R2 copies R1 before the wait, R3 after it.
    let code = program(&[rcp(1, 0, 0, 0), copy(2, 1, 0), copy(3, 1, 1)]);
    let mut input = State::default();
    input.regs[0] = f(4.0);
    input.regs[1] = f(7.0);
    let out = run(&vk, &code, input);
    assert_eq!(
        out.regs[2],
        f(7.0),
        "the copy before the wait reads the old value"
    );
    assert_eq!(
        out.regs[3],
        f(0.25),
        "the copy after the wait reads the new one"
    );
    assert_eq!(out.regs[1], f(0.25));
}

#[test]
fn a_counted_wait_lands_the_oldest_writes_first() {
    let Some(vk) = vulkan() else { return };
    // Two writes on barrier 1; `DEPBAR.LE SB1, 1` lets the older one land.
    let code = program(&[
        rcp(1, 0, 1, 0),
        rcp(4, 5, 1, 0),
        depbar(1, 1),
        copy(2, 1, 0),
        copy(6, 4, 0),
    ]);
    let mut input = State::default();
    input.regs[0] = f(2.0);
    input.regs[1] = f(9.0);
    input.regs[4] = f(3.0);
    input.regs[5] = f(8.0);
    let out = run(&vk, &code, input);
    assert_eq!(out.regs[2], f(0.5), "the older write has landed");
    assert_eq!(out.regs[6], f(3.0), "the newer one has not");
    // Everything lands at the end of the program.
    assert_eq!(out.regs[4], f(0.125));
}

/// `F2I.type.F32.TRUNC Rd, Rs` (bits 75..76: 0 8-bit, 1 16-bit, 2 32-bit; bit 72 signed).
fn f2i(d: u64, s: u64, size: u64, signed: bool) -> u128 {
    with(
        vec![
            (0, 12, 0x305),
            (12, 4, 7),
            (16, 8, d),
            (32, 8, s),
            (72, 1, u64::from(signed)),
            (75, 2, size),
            (78, 2, 3),
            (84, 2, 2),
        ],
        sched(7, 0),
    )
}

/// What the hardware's conversion gives: toward zero, saturating to the type's range (sign
/// extended to 32 bits), NaN 0. Rust's `as` saturates the same way.
fn f2i_reference(x: f32, size: u64, signed: bool) -> u32 {
    let t = x.trunc();
    match (size, signed) {
        (0, false) => u32::from(t as u8),
        (0, true) => t as i8 as i32 as u32,
        (1, false) => u32::from(t as u16),
        (1, true) => t as i16 as i32 as u32,
        (_, false) => t as u32,
        (_, true) => t as i32 as u32,
    }
}

#[test]
fn a_float_converts_to_an_integer_saturating_and_nan_gives_zero() {
    let Some(vk) = vulkan() else { return };
    let sets: [[f32; 7]; 3] = [
        [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            3e9,
            -3e9,
            2147483520.0,
            2147483648.0,
        ],
        [-2147483648.0, 1.7, -1.7, 255.9, 300.0, -129.5, 65535.5],
        [
            -0.0,
            4294967040.0,
            4294967296.0,
            70000.0,
            -40000.0,
            -32768.9,
            127.5,
        ],
    ];
    for size in 0..3 {
        for signed in [false, true] {
            let code = program(
                &(0..7)
                    .map(|k| f2i(7 + k, k, size, signed))
                    .collect::<Vec<_>>(),
            );
            for values in &sets {
                let mut input = State::default();
                for (k, v) in values.iter().enumerate() {
                    input.regs[k] = v.to_bits();
                }
                let out = run(&vk, &code, input);
                for (k, v) in values.iter().enumerate() {
                    assert_eq!(
                        out.regs[7 + k],
                        f2i_reference(*v, size, signed),
                        "{v} to size {size}, signed {signed}"
                    );
                }
            }
        }
    }
}

/// `SHF.{L|R}[.W].type[.HI] Rd, Ra, Rb, Rc` (the register form): the 64-bit funnel `Rc:Ra`
/// shifted by `Rb`.
fn shf(d: u64, kind: u64, wrap: bool, right: bool, high: bool) -> u128 {
    with(
        vec![
            (0, 12, 0x219),
            (12, 4, 7),
            (16, 8, d),
            (24, 8, 0),
            (32, 8, 2),
            (64, 8, 1),
            (73, 2, kind),
            (75, 1, u64::from(wrap)),
            (76, 1, u64::from(right)),
            (80, 1, u64::from(high)),
        ],
        sched(7, 0),
    )
}

/// The funnel shift as a 64-bit number: the amount clamped to (or wrapped in) 64 bits for the
/// 64-bit types and 32 for the others, a right shift of a signed type bringing in the sign.
fn shf_reference(
    lo: u32,
    hi: u32,
    amount: u32,
    kind: u64,
    wrap: bool,
    right: bool,
    high: bool,
) -> u32 {
    let width = if kind < 2 { 64 } else { 32 };
    let signed = kind == 0 || kind == 2;
    let s = if wrap {
        amount & (width - 1)
    } else {
        amount.min(width)
    };
    let v = u64::from(hi) << 32 | u64::from(lo);
    let r = match (right, signed) {
        (true, true) => ((v as i64) >> s.min(63)) as u64,
        (true, false) => v.checked_shr(s).unwrap_or(0),
        (false, _) => v.checked_shl(s).unwrap_or(0),
    };
    if high {
        (r >> 32) as u32
    } else {
        r as u32
    }
}

/// `IMAD.HI[.U32] Rd, R0, R1, R2`: the high word of the product, plus R2.
fn imad_hi(d: u64, signed: bool) -> u128 {
    with(
        vec![
            (0, 12, 0x227),
            (12, 4, 7),
            (16, 8, d),
            (24, 8, 0),
            (32, 8, 1),
            (64, 8, 2),
            (73, 1, u64::from(signed)),
        ],
        sched(7, 0),
    )
}

#[test]
fn the_high_word_of_a_product_is_added_to() {
    let Some(vk) = vulkan() else { return };
    let code = program(&[imad_hi(3, false), imad_hi(4, true)]);
    let values = [
        (0u32, 0u32, 0u32),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 1),
        (0x8000_0000, 2, 5),
        (0x1234_5678, 0x9ABC_DEF0, 0xFFFF_FFFF),
        (0x7FFF_FFFF, 0x8000_0001, 0x10),
    ];
    for (x, y, c) in values {
        let mut input = State::default();
        input.regs[0] = x;
        input.regs[1] = y;
        input.regs[2] = c;
        let out = run(&vk, &code, input);
        let unsigned = ((u64::from(x) * u64::from(y)) >> 32) as u32;
        let signed = ((i64::from(x as i32) * i64::from(y as i32)) >> 32) as u32;
        assert_eq!(
            out.regs[3],
            unsigned.wrapping_add(c),
            "{x:#x} * {y:#x} + {c:#x}"
        );
        assert_eq!(
            out.regs[4],
            signed.wrapping_add(c),
            "signed {x:#x} * {y:#x} + {c:#x}"
        );
    }
}

#[test]
fn a_funnel_shift_moves_the_bits_of_both_words() {
    let Some(vk) = vulkan() else { return };
    // Every form, twelve to a program (R3 to R14).
    let forms: Vec<(u64, bool, bool, bool)> = (0..4u64)
        .flat_map(|kind| {
            [false, true].into_iter().flat_map(move |wrap| {
                [false, true]
                    .into_iter()
                    .flat_map(move |right| [false, true].map(move |high| (kind, wrap, right, high)))
            })
        })
        .collect();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 16) as u32
    };
    let amounts = [0, 1, 7, 31, 32, 33, 47, 63, 64, 65, 100, 0xFFFF_FFFF];
    for chunk in forms.chunks(12) {
        let code = program(
            &chunk
                .iter()
                .enumerate()
                .map(|(k, &(kind, wrap, right, high))| shf(3 + k as u64, kind, wrap, right, high))
                .collect::<Vec<_>>(),
        );
        for &amount in &amounts {
            let (lo, hi) = (next() | 0x8000_0001, next() | 0x8000_0000);
            let mut input = State::default();
            input.regs[0] = lo;
            input.regs[1] = hi;
            input.regs[2] = amount;
            let out = run(&vk, &code, input);
            for (k, &(kind, wrap, right, high)) in chunk.iter().enumerate() {
                assert_eq!(
                    out.regs[3 + k],
                    shf_reference(lo, hi, amount, kind, wrap, right, high),
                    "{hi:#x}:{lo:#x} by {amount}, type {kind}, wrap {wrap}, right {right}, high {high}"
                );
            }
        }
    }
}

/// `HSETP2.cmp.bop P0, P1, R0, R1, PT` (the register form, both pairs as they are), `.H_AND`
/// with `h_and`.
fn hsetp2(cmp: u64, h_and: bool) -> u128 {
    with(
        vec![
            (0, 12, 0x234),
            (12, 4, 7),
            (24, 8, 0),
            (32, 8, 1),
            (69, 2, 0),
            (71, 1, u64::from(h_and)),
            (76, 4, cmp),
            (81, 3, 0),
            (84, 3, 1),
            (87, 3, 7),
        ],
        sched(7, 0),
    )
}

/// `@Pp MOV Rd, imm32`.
fn mov_if(p: u64, d: u64, imm: u32) -> u128 {
    with(
        vec![
            (0, 12, 0x802),
            (12, 4, p),
            (16, 8, d),
            (32, 32, u64::from(imm)),
            (72, 4, 0xF),
        ],
        sched(7, 0),
    )
}

#[test]
fn hsetp2_compares_the_lanes_alone_or_both_together() {
    let Some(vk) = vulkan() else { return };
    // Halves: 1.0 0x3C00, 1.5 0x3E00, 2.0 0x4000, 3.0 0x4200; the low lane first.
    let pair = |lo: u32, hi: u32| lo | hi << 16;
    let cases = [
        // LT (1) lane by lane: 1 < 2, not 3 < 2.
        (1, false, pair(0x3C00, 0x4200), (1, 0)),
        // LT with .H_AND: not both, so the first is false and the second true.
        (1, true, pair(0x3C00, 0x4200), (0, 1)),
        // GT (4) lane by lane: not 1 > 2, 3 > 2.
        (4, false, pair(0x3C00, 0x4200), (0, 1)),
        // LT with .H_AND where both lanes are below: 1 and 1.5 < 2.
        (1, true, pair(0x3C00, 0x3E00), (1, 0)),
    ];
    for (cmp, h_and, a, (p0, p1)) in cases {
        let code = program(&[hsetp2(cmp, h_and), mov_if(0, 10, 1), mov_if(1, 11, 1)]);
        let mut input = State::default();
        input.regs[0] = a;
        input.regs[1] = pair(0x4000, 0x4000);
        let out = run(&vk, &code, input);
        assert_eq!(
            (out.regs[10], out.regs[11]),
            (p0, p1),
            "compare {cmp}, .H_AND {h_and}, Ra {a:#x}"
        );
    }
}
