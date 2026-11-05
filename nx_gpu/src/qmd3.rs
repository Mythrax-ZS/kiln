//! Compute launches of the Switch 2's Ampere compute class: queue meta data (QMD) of version 3, and
//! the way NVN2 schedules them through each other (docs/ASSUMPTIONS.md A-210).
//!
//! A QMD is 64 words in memory that describe one launch: the program, the grid and block sizes, the
//! constant buffers, and up to three semaphore releases made when the launch is done. NVN2 names a
//! QMD with `SEND_PCAS_A` and then acts on it with `SEND_SIGNALING_PCAS2_B`: it schedules it, or
//! changes how many other launches it still waits for (its dependence count, word 4). A QMD that
//! is scheduled and waits for nothing more is launched; when it is done its releases are made and it
//! may act on another QMD in turn (words 15 and 16), which is how NVN2 chains its queues. The
//! releases are handed back with the launch, to be made once it has run (A-274).

use crate::mem::GuestMemory;
use crate::vm::AddressSpace;
use std::collections::HashMap;

/// Words of a QMD.
pub const WORDS: usize = 64;

/// The methods (word indices) of the compute class that launch.
pub mod method {
    /// `INLINE_QMD_LAUNCH_CONTROL`: bit 24 schedules an inline QMD when its last word is written.
    pub const INLINE_CONTROL: u32 = 0xA7;
    /// `SEND_PCAS_A`: the address of the QMD the next action is for, in units of 256 bytes.
    pub const PCAS_A: u32 = 0xAD;
    pub const PCAS_B: u32 = 0xAE;
    /// `SEND_SIGNALING_PCAS_B`: bit 0 invalidates, bit 1 schedules.
    pub const SIGNALING_PCAS_B: u32 = 0xAF;
    /// `SEND_SIGNALING_PCAS2_B`: an action and a count.
    pub const SIGNALING_PCAS2_B: u32 = 0xB0;
    /// `SET_INLINE_QMD_ADDRESS_A`/`_B`, then the 64 words of an inline QMD.
    pub const INLINE_ADDRESS_A: u32 = 0xC6;
    pub const INLINE_ADDRESS_B: u32 = 0xC7;
    pub const INLINE_FIRST: u32 = 0xC8;
    pub const INLINE_LAST: u32 = INLINE_FIRST + super::WORDS as u32 - 1;
}

/// What NVN2 asks of a QMD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Launch it once nothing is left to wait for.
    Schedule,
    /// Read it again from memory, then schedule it (`INVALIDATE_COPY_SCHEDULE`).
    Reload,
    /// It waits for that many fewer launches.
    Decrement(u32),
    /// It waits for that many more.
    Increment(u32),
}

impl Action {
    /// The action of a `SEND_SIGNALING_PCAS2_B` value: bits 3..0 name it, bits 23..14 are a count
    /// less one. `None` for the actions that leave the QMD as it is (`NOP`, `INVALIDATE`).
    pub fn from_pcas2(value: u32) -> Result<Option<Action>, String> {
        let count = (value >> 14 & 0x3FF) + 1;
        Ok(match value & 0xF {
            0 | 1 => None,
            2 => Some(Action::Schedule),
            3 => Some(Action::Reload),
            7 => Some(Action::Decrement(count)),
            12 => Some(Action::Increment(count)),
            other => {
                return Err(format!(
                    "the QMD action {other} (SEND_SIGNALING_PCAS2_B {value:#x})"
                ))
            }
        })
    }

    /// The action of a `SEND_SIGNALING_PCAS_B` value.
    pub fn from_pcas(value: u32) -> Option<Action> {
        match value & 3 {
            2 => Some(Action::Schedule),
            3 => Some(Action::Reload),
            _ => None,
        }
    }

    /// The action a done QMD takes on its dependent one (word 16, bits 3..1).
    fn dependent(kind: u32) -> Result<Action, String> {
        Ok(match kind {
            1 => Action::Schedule,
            3 => Action::Reload,
            4 => Action::Decrement(1),
            other => return Err(format!("the dependent QMD action {other}")),
        })
    }
}

/// A launch's QMD, read from memory when it was launched, and the releases it makes when it is
/// done, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub address: u64,
    pub qmd: [u32; WORDS],
    pub releases: Vec<Release>,
}

/// A semaphore release of one word that a QMD makes when its launch is done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    pub va: u64,
    pub op: ReleaseOp,
    pub payload: u32,
}

/// What a release puts in its word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseOp {
    /// The payload.
    Set,
    /// The word plus the payload.
    Add,
    /// The larger of the word and the payload (NVN2 counts completions so, in a word other queues
    /// advance too).
    Max,
}

impl Release {
    /// Make the release in memory now.
    pub fn apply(&self, mem: &mut dyn Memory) -> Result<(), String> {
        let value = match self.op {
            ReleaseOp::Set => self.payload,
            ReleaseOp::Add | ReleaseOp::Max => {
                let mut old = [0u32];
                mem.read(self.va, &mut old)?;
                if self.op == ReleaseOp::Add {
                    old[0].wrapping_add(self.payload)
                } else {
                    old[0].max(self.payload)
                }
            }
        };
        mem.write(self.va, &[value])
    }
}

impl Launch {
    /// The GPU address of the program.
    pub fn program(&self) -> u64 {
        u64::from(self.qmd[49] & 0x1FFFF) << 32 | u64::from(self.qmd[48])
    }

    pub fn grid(&self) -> [u32; 3] {
        [self.qmd[12], self.qmd[13] & 0xFFFF, self.qmd[14] & 0xFFFF]
    }

    pub fn block(&self) -> [u32; 3] {
        [
            self.qmd[18] >> 16,
            self.qmd[19] & 0xFFFF,
            self.qmd[19] >> 16,
        ]
    }

    /// The QMD's version, 0x30 for version 3.
    pub fn version(&self) -> u32 {
        self.qmd[18] & 0xFF
    }

    /// Whether it launches anything: a grid with no blocks only makes its releases.
    pub fn has_work(&self) -> bool {
        self.grid().iter().all(|&n| n != 0)
    }
}

/// The memory the scheduler reads QMDs from and makes releases to (GPU addresses).
pub trait Memory {
    fn read(&mut self, va: u64, out: &mut [u32]) -> Result<(), String>;
    fn write(&mut self, va: u64, words: &[u32]) -> Result<(), String>;
}

/// The memory of an address space.
pub struct GpuMemory<'a> {
    pub mem: &'a dyn GuestMemory,
    pub space: &'a AddressSpace,
}

impl Memory for GpuMemory<'_> {
    fn read(&mut self, va: u64, out: &mut [u32]) -> Result<(), String> {
        let mut bytes = vec![0u8; out.len() * 4];
        if !self.space.read(self.mem, va, &mut bytes) {
            return Err(format!(
                "a QMD read of GPU address {va:#x}, which is not mapped"
            ));
        }
        for (w, b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *w = u32::from_le_bytes(*b);
        }
        Ok(())
    }

    fn write(&mut self, va: u64, words: &[u32]) -> Result<(), String> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        if !self.space.write(self.mem, va, &bytes) {
            return Err(format!(
                "a QMD release to GPU address {va:#x}, which is not mapped"
            ));
        }
        Ok(())
    }
}

/// A QMD the scheduler has heard of and that has not been launched.
#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    /// Its dependence count was read since it was last launched or reloaded.
    loaded: bool,
    scheduled: bool,
    /// The launches it still waits for; decrements made before it was read make this negative.
    waiting: i64,
}

/// The QMDs of a process that are scheduled or waited for, by address: one table for all channels
/// (A-210).
#[derive(Debug, Default)]
pub struct Scheduler {
    entries: HashMap<u64, Entry>,
}

/// Release flags (word 1 of a release slot).
const RELEASE_ENABLE: u32 = 1 << 23;
const RELEASE_MAX: u32 = 1 << 19;
const RELEASE_ADD: u32 = 1 << 26;
const RELEASE_UNKNOWN: u32 = 1 << 24 | 1 << 25 | 1 << 27 | 1 << 28 | 1 << 29;
const RELEASE_KIND: u32 = 3 << 30;
const RELEASE_ONE_WORD: u32 = 1 << 30;
/// Where the three release slots are.
const RELEASES: [usize; 3] = [24, 28, 52];

impl Scheduler {
    /// Act on the QMD at `address`. Returns the QMDs launched by it, in order, each with the
    /// releases to make once it is done; the actions they take on their dependents are applied.
    pub fn signal(
        &mut self,
        address: u64,
        action: Action,
        mem: &mut dyn Memory,
    ) -> Result<Vec<Launch>, String> {
        let mut launched = Vec::new();
        let mut todo = vec![(address, action)];
        while let Some((address, action)) = todo.pop() {
            let entry = self.entries.entry(address).or_default();
            match action {
                Action::Schedule => {
                    if !entry.loaded {
                        load(entry, address, mem)?;
                    }
                    entry.scheduled = true;
                }
                Action::Reload => {
                    // Decrements already made still count; what was read before does not.
                    entry.waiting = entry.waiting.min(0);
                    entry.loaded = false;
                    load(entry, address, mem)?;
                    entry.scheduled = true;
                }
                Action::Decrement(n) => entry.waiting -= i64::from(n),
                Action::Increment(n) => entry.waiting += i64::from(n),
            }
            if entry.loaded && entry.scheduled && entry.waiting <= 0 {
                self.entries.remove(&address);
                let mut qmd = [0u32; WORDS];
                mem.read(address, &mut qmd)?;
                let mut releases = Vec::new();
                for slot in RELEASES {
                    releases.extend(release(&qmd[slot..slot + 3])?);
                }
                let launch = Launch {
                    address,
                    qmd,
                    releases,
                };
                let dependent = launch.qmd[16];
                if dependent & 1 != 0 {
                    let target = u64::from(launch.qmd[15]) << 8;
                    todo.push((target, Action::dependent(dependent >> 1 & 7)?));
                }
                launched.push(launch);
            }
        }
        Ok(launched)
    }
}

/// Read a QMD's dependence count (word 4, bits 29..14) into its entry.
fn load(entry: &mut Entry, address: u64, mem: &mut dyn Memory) -> Result<(), String> {
    let mut word = [0u32];
    mem.read(address + 4 * 4, &mut word)?;
    let waiting_before = if entry.loaded { 0 } else { entry.waiting };
    entry.waiting = i64::from(word[0] >> 14 & 0xFFFF) + waiting_before;
    entry.loaded = true;
    Ok(())
}

/// The release of a slot (low address, flags with the high address, payload), if it is on.
fn release(slot: &[u32]) -> Result<Option<Release>, String> {
    let (low, flags, payload) = (slot[0], slot[1], slot[2]);
    if flags & RELEASE_ENABLE == 0 {
        return Ok(None);
    }
    if flags & RELEASE_UNKNOWN != 0 || flags & RELEASE_KIND != RELEASE_ONE_WORD {
        return Err(format!("a QMD release with the flags {flags:#x}"));
    }
    let op = if flags & RELEASE_ADD != 0 {
        ReleaseOp::Add
    } else if flags & RELEASE_MAX != 0 {
        ReleaseOp::Max
    } else {
        ReleaseOp::Set
    };
    Ok(Some(Release {
        va: u64::from(flags & 0xFF) << 32 | u64::from(low),
        op,
        payload,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Memory of 64 KiB at GPU address 0x10_0000.
    struct Ram(Vec<u32>);

    const BASE: u64 = 0x10_0000;

    impl Memory for Ram {
        fn read(&mut self, va: u64, out: &mut [u32]) -> Result<(), String> {
            let at = (va - BASE) as usize / 4;
            out.copy_from_slice(&self.0[at..at + out.len()]);
            Ok(())
        }
        fn write(&mut self, va: u64, words: &[u32]) -> Result<(), String> {
            let at = (va - BASE) as usize / 4;
            self.0[at..at + words.len()].copy_from_slice(words);
            Ok(())
        }
    }

    /// Act on a QMD and make the releases of what it launched, as the GPU front does once the
    /// launches are done.
    fn run(
        s: &mut Scheduler,
        at: u64,
        action: Action,
        ram: &mut Ram,
    ) -> Result<Vec<Launch>, String> {
        let launched = s.signal(at, action, ram)?;
        for l in &launched {
            for r in &l.releases {
                r.apply(ram)?;
            }
        }
        Ok(launched)
    }

    impl Ram {
        fn new() -> Ram {
            Ram(vec![0; 0x4000])
        }
        fn word(&self, va: u64) -> u32 {
            self.0[(va - BASE) as usize / 4]
        }
        fn set(&mut self, va: u64, value: u32) {
            self.0[(va - BASE) as usize / 4] = value;
        }
        /// A QMD at `at` that waits for `waiting` launches and releases `payload` to `target`
        /// with the extra flags `flags`.
        fn qmd(&mut self, at: u64, waiting: u32, release: Option<(u64, u32, u32)>) {
            self.set(at + 18 * 4, 0x30 | 32 << 16);
            self.set(at + 12 * 4, 4);
            self.set(at + 13 * 4, 1);
            self.set(at + 14 * 4, 1);
            self.set(at + 4 * 4, waiting << 14);
            if let Some((target, flags, payload)) = release {
                self.set(at + 24 * 4, target as u32);
                self.set(
                    at + 25 * 4,
                    RELEASE_ONE_WORD | RELEASE_ENABLE | flags | (target >> 32) as u32,
                );
                self.set(at + 26 * 4, payload);
            }
        }
    }

    #[test]
    fn a_scheduled_qmd_that_waits_for_nothing_launches_and_releases() {
        let mut ram = Ram::new();
        let (at, target) = (BASE + 0x100, BASE + 0x8000);
        ram.qmd(at, 0, Some((target, 0, 7)));
        let mut s = Scheduler::default();
        let launched = s.signal(at, Action::Schedule, &mut ram).unwrap();
        assert_eq!(
            ram.word(target),
            0,
            "the release waits for the launch to be done"
        );
        for r in &launched[0].releases {
            r.apply(&mut ram).unwrap();
        }
        assert_eq!(launched.len(), 1);
        assert_eq!(
            (
                launched[0].grid(),
                launched[0].block(),
                launched[0].version()
            ),
            ([4, 1, 1], [32, 0, 0], 0x30)
        );
        assert_eq!(ram.word(target), 7);
    }

    #[test]
    fn a_qmd_waits_for_its_count_of_decrements_even_those_made_before_it_was_scheduled() {
        let mut ram = Ram::new();
        let (at, target) = (BASE + 0x200, BASE + 0x8000);
        ram.qmd(at, 2, Some((target, 0, 9)));
        let mut s = Scheduler::default();
        assert!(run(&mut s, at, Action::Decrement(1), &mut ram)
            .unwrap()
            .is_empty());
        assert!(run(&mut s, at, Action::Schedule, &mut ram)
            .unwrap()
            .is_empty());
        assert_eq!(ram.word(target), 0, "one launch still waited for");
        assert_eq!(
            run(&mut s, at, Action::Decrement(1), &mut ram)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(ram.word(target), 9);
        // The action values: decrement by a count less one in bits 23..14, increment, schedule.
        assert_eq!(
            Action::from_pcas2(7 | 2 << 14),
            Ok(Some(Action::Decrement(3)))
        );
        assert_eq!(Action::from_pcas2(12), Ok(Some(Action::Increment(1))));
        assert_eq!(Action::from_pcas2(3), Ok(Some(Action::Reload)));
        assert!(Action::from_pcas2(9).is_err(), "a prefetch is not known");
    }

    #[test]
    fn add_and_max_releases_read_what_is_there() {
        let mut ram = Ram::new();
        let target = BASE + 0x8000;
        ram.set(target, 10);
        let mut s = Scheduler::default();
        ram.qmd(BASE + 0x300, 0, Some((target, RELEASE_ADD, 5)));
        run(&mut s, BASE + 0x300, Action::Reload, &mut ram).unwrap();
        assert_eq!(ram.word(target), 15);
        ram.qmd(BASE + 0x400, 0, Some((target, RELEASE_MAX, 12)));
        run(&mut s, BASE + 0x400, Action::Schedule, &mut ram).unwrap();
        assert_eq!(ram.word(target), 15, "a maximum does not go back");
        // Flags whose meaning is not known stop the launch.
        ram.qmd(BASE + 0x500, 0, Some((target, 1 << 24, 1)));
        assert!(run(&mut s, BASE + 0x500, Action::Schedule, &mut ram).is_err());
    }

    #[test]
    fn a_done_qmd_acts_on_its_dependent() {
        let mut ram = Ram::new();
        let target = BASE + 0x8000;
        let (first, second) = (BASE + 0x600, BASE + 0x700);
        ram.qmd(first, 0, None);
        // Done, the first decrements the second, which waits for one launch.
        ram.set(first + 15 * 4, (second >> 8) as u32);
        ram.set(first + 16 * 4, 1 | 4 << 1);
        ram.qmd(second, 1, Some((target, 0, 3)));
        let mut s = Scheduler::default();
        assert!(run(&mut s, second, Action::Schedule, &mut ram)
            .unwrap()
            .is_empty());
        let launched = run(&mut s, first, Action::Schedule, &mut ram).unwrap();
        let addresses: Vec<u64> = launched.iter().map(|l| l.address).collect();
        assert_eq!(addresses, [first, second]);
        assert_eq!(ram.word(target), 3);
    }
}
