//! Randomized differential tests of floating point and Advanced SIMD: the interpreter against
//! Unicorn's AArch64 emulation (QEMU TCG, Cortex-A57 model). Where they disagree the Arm
//! pseudocode decides; see docs/UNICORN_DISAGREEMENTS.md.

mod common;

use common::gen::Gen;
use common::harness::{Case, Class, Harness, Outcome};
use common::{case_count, run_batch, unicorn_or_skip, Accepted};
use nx_runtime::a64::{decode, Inst};

fn never(_: &Case, _: &Outcome, _: &Outcome) -> Option<Accepted> {
    None
}

/// Draw words from an encoding group until one decodes as a vector instruction.
fn vector_word(g: &mut Gen, mask: u32, value: u32) -> u32 {
    loop {
        let w = g.template_word(mask, value);
        if matches!(decode(w), Inst::Vec { .. } | Inst::VecMem { .. }) {
            return w;
        }
    }
}

fn group(name: &str, seed: u64, mask: u32, value: u32, cases: usize) {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(seed);
    run_batch(
        name,
        &mut h,
        case_count(cases),
        true,
        |_, _| {
            let w = vector_word(&mut g, mask, value);
            g.fp_case(vec![w])
        },
        never,
    );
}

#[test]
fn scalar_one_source() {
    group("fp one source", 0xA1, 0xFF20_7C00, 0x1E20_4000, 30_000);
}

#[test]
fn scalar_two_source() {
    group("fp two source", 0xA2, 0xFF20_0C00, 0x1E20_0800, 40_000);
}

#[test]
fn scalar_three_source() {
    group("fp three source", 0xA3, 0xFF00_0000, 0x1F00_0000, 30_000);
}

#[test]
fn scalar_compare_and_select() {
    group("fp compare", 0xA4, 0xFF20_FC00, 0x1E20_2000, 15_000);
    group(
        "fp conditional compare",
        0xA5,
        0xFF20_0C00,
        0x1E20_0400,
        15_000,
    );
    group(
        "fp conditional select",
        0xA6,
        0xFF20_0C00,
        0x1E20_0C00,
        15_000,
    );
    group("fp immediate", 0xA7, 0xFF20_1FE0, 0x1E20_1000, 5_000);
}

#[test]
fn scalar_conversions() {
    group(
        "fp integer conversion",
        0xA8,
        0x7F20_FC00,
        0x1E20_0000,
        40_000,
    );
    group(
        "fp fixed-point conversion",
        0xA9,
        0x7F20_0000,
        0x1E00_0000,
        20_000,
    );
}

#[test]
fn vector_register_loads_and_stores() {
    group(
        "simd load/store unsigned offset",
        0xB1,
        0x3F00_0000,
        0x3D00_0000,
        15_000,
    );
    group(
        "simd load/store immediate",
        0xB2,
        0x3F20_0000,
        0x3C00_0000,
        15_000,
    );
    group(
        "simd load/store register",
        0xB3,
        0x3F20_0C00,
        0x3C20_0800,
        15_000,
    );
    group("simd load literal", 0xB4, 0x3F00_0000, 0x1C00_0000, 5_000);
    group(
        "simd load/store pair",
        0xB5,
        0x3E00_0000,
        0x2C00_0000,
        20_000,
    );
}

#[test]
fn unused_classes_are_referenced() {
    // Keep the shared helpers in use when a group above is filtered out of a run.
    let _ = Class::Ok;
}
