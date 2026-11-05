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
