//! Randomized differential tests of Advanced SIMD: the interpreter against Unicorn's AArch64
//! emulation (QEMU TCG, Cortex-A57 model). Words are drawn from each encoding group whether or
//! not the interpreter implements them, so an encoding one side executes and the other rejects is
//! a mismatch too. Where they disagree the Arm pseudocode decides; see
//! docs/UNICORN_DISAGREEMENTS.md.
//!
//! Mismatches are reported per operation (the decoded instruction with its register numbers and
//! immediates removed) with one example each, so a run shows which instructions are wrong rather
//! than the first few failing cases.

mod common;

use common::gen::Gen;
use common::harness::{describe, diff, Case, Harness};
use common::{case_count, unicorn_or_skip};
use nx_runtime::a64::decode;
use std::collections::BTreeMap;

/// The decoded instruction without the fields that differ between instances of one operation.
fn signature(w: u32) -> String {
    let text = format!("{:?}", decode(w));
    let mut out = String::new();
    let mut rest = text.as_str();
    while !rest.is_empty() {
        let dropped = [
            "word", "rd", "rn", "rm", "ra", "idx", "imm", "src", "dst", "shift", "fbits", "imm4",
            "nzcv",
        ]
        .iter()
        .find_map(|name| {
            let key = format!("{name}: ");
            rest.strip_prefix(key.as_str()).map(|r| (key.len(), r))
        });
        match dropped {
            Some((_, after)) => {
                let end = after
                    .find(|c: char| !(c.is_ascii_digit() || c == '-'))
                    .unwrap_or(after.len());
                rest = &after[end..];
                if let Some(r) = rest.strip_prefix(", ") {
                    rest = r;
                }
            }
            None => {
                let c = rest.chars().next().unwrap();
                out.push(c);
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    out
}

fn group(name: &str, seed: u64, mask: u32, value: u32, cases: usize) {
    group_with(name, seed, mask, value, cases, |g, w| g.simd_case(vec![w]));
}

fn group_with(
    name: &str,
    seed: u64,
    mask: u32,
    value: u32,
    cases: usize,
    make: impl Fn(&mut Gen, u32) -> Case,
) {
    let Some(api) = unicorn_or_skip() else { return };
    let mut h = Harness::new(&api);
    let mut g = Gen::new(seed);
    let n = case_count(cases);
    let mut wrong: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut classes: BTreeMap<String, usize> = BTreeMap::new();
    for _ in 0..n {
        let pc = h.slot();
        let w = g.template_word(mask, value);
        let mut case = make(&mut g, w);
        case.state.pc = pc;
        let uc = h.run_unicorn(&case);
        let mine = h.run_mine(&case);
        *classes.entry(format!("{:?}", mine.class)).or_default() += 1;
        if let Some(d) = diff(&uc, &mine, true) {
            let entry = wrong
                .entry(signature(w))
                .or_insert_with(|| (0, format!("{}  {d}", describe(&case))));
            entry.0 += 1;
        }
    }
    eprintln!(
        "{name}: {n} cases, outcomes {classes:?}, {} operations wrong",
        wrong.len()
    );
    if !wrong.is_empty() {
        let mut report = String::new();
        for (sig, (count, example)) in &wrong {
            let first_line = example.lines().next().unwrap_or("");
            report.push_str(&format!("{count:6} x {sig}\n        e.g. {first_line}\n"));
            for line in example
                .lines()
                .skip(1)
                .filter(|l| !l.trim_start().starts_with("pc="))
            {
                report.push_str(&format!("        {line}\n"));
            }
        }
        panic!(
            "{name}: interpreter and Unicorn disagree on {} operations:\n{report}",
            wrong.len()
        );
    }
}

// Vector groups: bit 31 is zero, Q (30) and U (29) are free.

#[test]
fn vector_three_same() {
    group("three same", 0xC1, 0x9F20_0400, 0x0E20_0400, 150_000);
}

#[test]
fn vector_three_different() {
    group("three different", 0xC2, 0x9F20_0C00, 0x0E20_0000, 100_000);
}

#[test]
fn vector_two_register_miscellaneous() {
    group("two-register misc", 0xC3, 0x9F3E_0C00, 0x0E20_0800, 150_000);
}

#[test]
fn vector_across_lanes() {
    group("across lanes", 0xC4, 0x9F3E_0C00, 0x0E30_0800, 40_000);
}

#[test]
fn vector_permute_extract_table_copy() {
    group("permute", 0xC5, 0xBF20_8C00, 0x0E00_0800, 40_000);
    group("extract", 0xC6, 0xBFE0_8400, 0x2E00_0000, 20_000);
    group("table lookup", 0xC7, 0xBFE0_8C00, 0x0E00_0000, 40_000);
    group("copy", 0xC8, 0x9FE0_8400, 0x0E00_0400, 60_000);
}

#[test]
fn vector_immediates() {
    group("modified immediate", 0xC9, 0x9FF8_0400, 0x0F00_0400, 40_000);
    group(
        "shift by immediate",
        0xCA,
        0x9F80_0400,
        0x0F00_0400,
        120_000,
    );
    group("x indexed element", 0xCB, 0x9F00_0400, 0x0F00_0000, 120_000);
}

#[test]
fn scalar_groups() {
    group("scalar three same", 0xD1, 0xDF20_0400, 0x5E20_0400, 100_000);
    group(
        "scalar three different",
        0xD2,
        0xDF20_0C00,
        0x5E20_0000,
        20_000,
    );
    group(
        "scalar two-register misc",
        0xD3,
        0xDF3E_0C00,
        0x5E20_0800,
        80_000,
    );
    group("scalar pairwise", 0xD4, 0xDF3E_0C00, 0x5E30_0800, 20_000);
    group("scalar shift", 0xD5, 0xDF80_0400, 0x5F00_0400, 60_000);
    group(
        "scalar x indexed element",
        0xD6,
        0xDF00_0400,
        0x5F00_0000,
        60_000,
    );
    group("scalar copy", 0xD7, 0xDFE0_8400, 0x5E00_0400, 10_000);
}

/// The cryptographic extension: AES, SHA-1 and SHA-256 (64-bit `PMULL` is in the "three
/// different" group). The groups include their reserved neighbours, which both sides must reject.
#[test]
fn cryptographic_extension() {
    group("aes", 0xE1, 0xFF3E_0C00, 0x4E28_0800, 40_000);
    group("sha three-register", 0xE2, 0xFF20_8C00, 0x5E00_0000, 60_000);
    group("sha two-register", 0xE3, 0xFF3E_0C00, 0x5E28_0800, 40_000);
    // The same with the size and the high opcode bits fixed, so nearly every word is a valid
    // instruction.
    group("aes (valid forms)", 0xE4, 0xFFFF_CC00, 0x4E28_4800, 60_000);
    group(
        "sha three-register (size 0)",
        0xE5,
        0xFFE0_8C00,
        0x5E00_0000,
        120_000,
    );
    group(
        "sha two-register (low opcodes)",
        0xE6,
        0xFFFF_CC00,
        0x5E28_0800,
        60_000,
    );
}

/// A case whose base register points into the data region and whose index register, if the word
/// post-indexes by one, holds a small offset.
fn memory_case(g: &mut Gen, w: u32) -> Case {
    let mut case = g.simd_case(vec![w]);
    let rn = ((w >> 5) & 31) as usize;
    let rm = ((w >> 16) & 31) as usize;
    let base = g.data_pointer(0x800, 0x800);
    if rn == 31 {
        case.state.sp = base & !0xF;
    } else {
        case.state.x[rn] = base;
    }
    if rm < 31 && rm != rn {
        case.state.x[rm] = g.below(300);
    }
    case
}

#[test]
fn structure_loads_and_stores() {
    group_with(
        "multiple structures",
        0xE1,
        0xBF00_0000,
        0x0C00_0000,
        120_000,
        memory_case,
    );
    group_with(
        "single structure",
        0xE2,
        0xBF00_0000,
        0x0D00_0000,
        120_000,
        memory_case,
    );
}
