//! A census of what runs outside the fast paths (`NX_CENSUS=<file>`).
//!
//! Counts which floating-point and SIMD instruction words reach the runtime (`nx_vec`,
//! `nx_vecmem`), which words the lifter left to the interpreter (`nx_interp1`) and at which
//! addresses the slow path interprets guest code (and the targets of indirect branches), and
//! writes the counts to the file every ten seconds, most frequent first, with the decoded
//! instruction. It says what is worth lowering to
//! native code: nothing else tells how often a program really runs an instruction, and the
//! numbers are the only thing it keeps of the program.

use crate::a64::vec::decode as vdecode;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

const BITS: u32 = 18;
const SLOTS: usize = 1 << BITS;
/// How far a lookup walks before giving up on a full table.
const PROBES: usize = 64;
/// How many lines a report keeps: all of the table, so that what runs once is counted too.
const REPORT_LINES: usize = SLOTS;

/// What a count is about.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An instruction word given to `nx_vec`.
    Vec = 1,
    /// An instruction word given to `nx_vecmem`.
    VecMem = 2,
    /// An instruction word the lifter left to the interpreter.
    Fallback = 3,
    /// The address of an instruction the slow path interpreted.
    Interpreted = 4,
    /// The address of a load or store that took the runtime's slow path (an access straddling
    /// two pages, an unmapped page, a missing permission), with the instruction.
    Slow = 5,
    /// The target of an indirect branch or call (`nx_lookup`): which functions are reached
    /// through pointers, as a graphics driver's entry points are.
    Indirect = 6,
    /// The target of an indirect branch whose return address (`x30`) is in the range
    /// `NX_CENSUS_FROM` names: a program's own calls into a library, told from the library's.
    IndirectFrom = 7,
}

struct Table {
    keys: Box<[AtomicU64]>,
    counts: Box<[AtomicU64]>,
}

impl Table {
    fn new() -> Table {
        let zeros = || (0..SLOTS).map(|_| AtomicU64::new(0)).collect();
        Table {
            keys: zeros(),
            counts: zeros(),
        }
    }

    fn note(&self, kind: Kind, value: u64) {
        // The kind is never zero, so a key is never the empty marker.
        let key = ((kind as u64) << 56) | (value & ((1 << 56) - 1));
        let mut i = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - BITS)) as usize;
        for _ in 0..PROBES {
            let seen = self.keys[i].load(Ordering::Relaxed);
            let ours = seen == key
                || (seen == 0
                    && self.keys[i]
                        .compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed)
                        .map_or_else(|now| now == key, |_| true));
            if ours {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
            i = (i + 1) & (SLOTS - 1);
        }
    }

    /// Every counted `(kind, value, count)`, most frequent first.
    fn entries(&self) -> Vec<(u8, u64, u64)> {
        let mut v: Vec<(u8, u64, u64)> = (0..SLOTS)
            .filter_map(|i| {
                let key = self.keys[i].load(Ordering::Relaxed);
                let count = self.counts[i].load(Ordering::Relaxed);
                (key != 0 && count != 0).then_some((
                    (key >> 56) as u8,
                    key & ((1 << 56) - 1),
                    count,
                ))
            })
            .collect();
        v.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        v
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static TABLE: OnceLock<Table> = OnceLock::new();
/// The return addresses `[lo, hi)` whose indirect branches are counted as [`Kind::IndirectFrom`].
static FROM: OnceLock<(u64, u64)> = OnceLock::new();

thread_local! {
    /// The CPU of the guest thread this host thread runs, for the return address of a branch.
    static CPU: std::cell::Cell<*const crate::rt::abi::NativeCpu> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// This host thread runs the guest thread of `cpu` (looked at only while the census is on).
pub fn enter(cpu: *const crate::rt::abi::NativeCpu) {
    if enabled() {
        CPU.with(|c| c.set(cpu));
    }
}

/// Count the target `pc` of an indirect branch, and again as [`Kind::IndirectFrom`] when the
/// running thread's return address is in the range of `NX_CENSUS_FROM`.
pub fn note_indirect(pc: u64) {
    note(Kind::Indirect, pc);
    if let Some(&(lo, hi)) = FROM.get() {
        let cpu = CPU.with(|c| c.get());
        // SAFETY: set by `enter` for the guest thread this host thread is running.
        let lr = (!cpu.is_null()).then(|| unsafe { (*cpu).state.x[30] });
        if lr.is_some_and(|lr| (lo..hi).contains(&lr)) {
            note(Kind::IndirectFrom, pc);
        }
    }
}

/// `NX_CENSUS_FROM=<hex lo>:<hex hi>`.
fn parse_from(text: &str) -> Option<(u64, u64)> {
    let (lo, hi) = text.split_once(':')?;
    let hex = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    Some((hex(lo)?, hex(hi)?))
}

/// Is the census on? A relaxed load: the callers check it on every floating-point instruction.
#[inline(always)]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Count one occurrence. Callers check [`enabled`] first.
pub fn note(kind: Kind, value: u64) {
    if let Some(t) = TABLE.get() {
        t.note(kind, value);
    }
}

/// Start the census if `NX_CENSUS` names a file for its report.
pub fn start_from_env() {
    let Some(path) = std::env::var_os("NX_CENSUS") else {
        return;
    };
    if TABLE.set(Table::new()).is_err() {
        return;
    }
    if let Some(range) = std::env::var("NX_CENSUS_FROM")
        .ok()
        .and_then(|t| parse_from(&t))
    {
        let _ = FROM.set(range);
    }
    ENABLED.store(true, Ordering::Release);
    let _ = std::thread::Builder::new()
        .name("census".to_string())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_secs(10));
            if let Some(t) = TABLE.get() {
                let _ = std::fs::write(&path, report(&t.entries()));
            }
        });
}

/// The decoded instruction of a counted word, for the report.
fn describe(kind: u8, value: u64) -> String {
    let word = value as u32;
    match kind {
        k if k == Kind::Vec as u8 => match vdecode::decode(word) {
            Ok(op) => format!("{op:?}"),
            Err(_) => "(undecodable)".to_string(),
        },
        k if k == Kind::VecMem as u8 => match vdecode::decode_mem(word) {
            Ok((op, _)) => format!("{op:?}"),
            Err(_) => "(undecodable)".to_string(),
        },
        k if k == Kind::Fallback as u8 => format!("{:?}", crate::a64::decode(word)),
        _ => String::new(),
    }
}

fn kind_name(kind: u8) -> &'static str {
    match kind {
        k if k == Kind::Vec as u8 => "vec",
        k if k == Kind::VecMem as u8 => "vecmem",
        k if k == Kind::Fallback as u8 => "fallback",
        k if k == Kind::Slow as u8 => "slow",
        k if k == Kind::Indirect as u8 => "indirect",
        k if k == Kind::IndirectFrom as u8 => "indirect-from",
        _ => "interpreted",
    }
}

/// The text of a report: totals per kind, then one line per word or address, most frequent first.
fn report(entries: &[(u8, u64, u64)]) -> String {
    let mut s = String::new();
    for kind in [
        Kind::Vec,
        Kind::VecMem,
        Kind::Fallback,
        Kind::Interpreted,
        Kind::Slow,
        Kind::Indirect,
        Kind::IndirectFrom,
    ] {
        let (distinct, total) = entries
            .iter()
            .filter(|e| e.0 == kind as u8)
            .fold((0u64, 0u64), |(d, t), e| (d + 1, t + e.2));
        let _ = writeln!(
            s,
            "# {}: {total} executions of {distinct} distinct",
            kind_name(kind as u8)
        );
    }
    for &(kind, value, count) in entries.iter().take(REPORT_LINES) {
        let _ = writeln!(
            s,
            "{count} {} {value:#x} {}",
            kind_name(kind),
            describe(kind, value)
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_kept_per_kind_and_value_and_listed_by_frequency() {
        let t = Table::new();
        for _ in 0..5 {
            t.note(Kind::Vec, 0x1E22_2820);
        }
        for _ in 0..2 {
            t.note(Kind::Interpreted, 0x8000_0000);
        }
        t.note(Kind::Vec, 0x1E22_2821);
        // The same number under another kind is another entry.
        t.note(Kind::VecMem, 0x1E22_2820);
        let e = t.entries();
        assert_eq!(e[0], (Kind::Vec as u8, 0x1E22_2820, 5));
        assert_eq!(e[1], (Kind::Interpreted as u8, 0x8000_0000, 2));
        assert_eq!(e.len(), 4);
        let text = report(&e);
        assert!(text.contains("# vec: 6 executions of 2 distinct"), "{text}");
        assert!(text.contains("\n5 vec 0x1e222820 "), "{text}");
    }

    #[test]
    fn the_range_of_return_addresses_is_read_in_hexadecimal() {
        assert_eq!(
            parse_from("8020000:c1e0000"),
            Some((0x802_0000, 0xc1e_0000))
        );
        assert_eq!(parse_from("0x10:0x20"), Some((0x10, 0x20)));
        assert_eq!(parse_from("10"), None);
    }

    #[test]
    fn threads_counting_together_lose_nothing() {
        let t = std::sync::Arc::new(Table::new());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let t = std::sync::Arc::clone(&t);
                std::thread::spawn(move || {
                    for i in 0..10_000u64 {
                        t.note(Kind::Fallback, i % 100);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let e = t.entries();
        assert_eq!(e.len(), 100);
        assert!(e.iter().all(|x| x.2 == 400));
    }
}
