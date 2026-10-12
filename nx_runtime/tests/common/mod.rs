//! Shared support for the integration tests.

#![allow(dead_code)]

pub mod gen;
pub mod harness;
pub mod unicorn;

use harness::{describe, diff, Case, Harness, Outcome};

/// How many random cases each differential test runs. `RECOMP_DIFF_CASES` scales them.
pub fn case_count(default: usize) -> usize {
    std::env::var("RECOMP_DIFF_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Load Unicorn or announce that the differential tests were skipped.
pub fn unicorn_or_skip() -> Option<unicorn::Api> {
    match unicorn::Api::load() {
        Some(api) => {
            let (major, minor) = api.version();
            eprintln!("using Unicorn {major}.{minor} (Cortex-A57 model)");
            Some(api)
        }
        None => {
            if std::env::var_os("RECOMP_REQUIRE_EXTERNAL").is_some() {
                panic!("Unicorn was not found and RECOMP_REQUIRE_EXTERNAL is set");
            }
            eprintln!("SKIPPED: Unicorn not found (set UNICORN_DLL or `pip install unicorn` into .venv); differential tests did not run");
            None
        }
    }
}

/// A recorded difference that is understood and accepted. The pseudocode wins; see
/// docs/UNICORN_DISAGREEMENTS.md.
pub struct Accepted {
    pub id: &'static str,
}

/// Run `n` generated cases through both backends and fail with a readable report on any difference.
pub fn run_batch(
    name: &str,
    harness: &mut Harness,
    n: usize,
    compare_fp: bool,
    mut make: impl FnMut(u64, u64) -> Case,
    accept: impl Fn(&Case, &Outcome, &Outcome) -> Option<Accepted>,
) {
    let mut failures = Vec::new();
    let mut accepted: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let mut classes: std::collections::BTreeMap<String, usize> = Default::default();
    for i in 0..n {
        let pc = harness.slot();
        let mut case = make(pc, i as u64);
        case.state.pc = pc;
        let uc = harness.run_unicorn(&case);
        let mine = harness.run_mine(&case);
        *classes.entry(format!("{:?}", mine.class)).or_default() += 1;
        if let Some(d) = diff(&uc, &mine, compare_fp) {
            match accept(&case, &uc, &mine) {
                Some(a) => *accepted.entry(a.id).or_default() += 1,
                None => {
                    if failures.len() < 12 {
                        failures.push(format!("case {i}:\n{}  {d}", describe(&case)));
                    } else if failures.len() == 12 {
                        failures.push("... further mismatches omitted".to_string());
                    }
                }
            }
        }
    }
    eprintln!("{name}: {n} cases, outcomes {classes:?}, accepted differences {accepted:?}");
    assert!(
        failures.is_empty(),
        "{name}: interpreter and Unicorn disagree:\n{}",
        failures.join("\n")
    );
}
