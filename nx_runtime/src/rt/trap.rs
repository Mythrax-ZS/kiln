//! `NX_TRAP_WRITES=<address>:<length>[,<address>:<length>...]` (guest addresses, hexadecimal): say
//! who writes into those pages of guest memory. It is a way of asking a running title "which
//! code fills this texture, and when does it?" without a debugger.
//!
//! The pages of the ranges lose their fast write permission in the page table, so every write of
//! the guest to them goes through the checked path of [`super::mem::GuestMemory`], which reports it
//! here before it is done. A report names the thread, the address, the value (for a store of up
//! to eight bytes) and the call chain as offsets in the program's image (`+0x1234`): look them up
//! in the link map (`KILN_LINK_MAP`, which is what `kiln relink` writes next to the program) to
//! get the names of the functions. A call chain is reported the first time it is seen and again
//! when it has been seen 10, 100, 1000... times, so a loop that fills a page is one line.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

/// How many addresses of a call chain are kept.
const CHAIN: usize = 14;
/// How many different call chains are told about.
const MAX_CHAINS: usize = 400;

thread_local! {
    /// The guest address of the instruction whose store the thread is making, as the store
    /// helpers of generated code say it (0 when they did not).
    static PC: Cell<u64> = const { Cell::new(0) };
}

/// Say which guest instruction the store the thread is about to make belongs to.
pub fn note_pc(pc: u64) {
    PC.with(|c| c.set(pc));
}

#[derive(Default)]
pub struct Traps {
    /// `[start, end)` guest ranges.
    ranges: Vec<(u64, u64)>,
    /// The call chains seen, by hash, with how many times.
    seen: Mutex<HashMap<u64, u32>>,
    /// How many writes there were in all.
    hits: AtomicU32,
}

impl Traps {
    /// The ranges `NX_TRAP_WRITES` names; none when it is not set.
    pub fn from_env() -> Traps {
        match std::env::var_os("NX_TRAP_WRITES") {
            Some(text) => Traps::parse(&text.to_string_lossy()),
            None => Traps::default(),
        }
    }

    /// The ranges of a list `<address>:<length>,...`; what does not parse is told about and left
    /// out.
    pub fn parse(text: &str) -> Traps {
        let mut traps = Traps::default();
        for part in text.split(',') {
            let Some((a, n)) = part.split_once(':') else {
                eprintln!("[trap] NX_TRAP_WRITES: {part:?} is not <address>:<length>");
                continue;
            };
            let parse = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16);
            match (parse(a), parse(n)) {
                (Ok(start), Ok(len)) if len > 0 => {
                    eprintln!(
                        "[trap] writes to {start:#x}..{:#x} will be reported",
                        start + len
                    );
                    traps.ranges.push((start, start + len));
                }
                _ => eprintln!("[trap] NX_TRAP_WRITES: {part:?} is not <address>:<length>"),
            }
        }
        traps
    }

    /// Does a trap cover the page at `page`?
    pub fn covers(&self, page: u64) -> bool {
        self.ranges
            .iter()
            .any(|&(start, end)| page < end && page + super::mem::PAGE_SIZE > start)
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// A guest write of `len` bytes at `addr` (and `value`, for a store of up to eight bytes) is
    /// about to happen.
    pub fn hit(&self, addr: u64, len: usize, value: Option<u128>) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        let chain = host::call_chain(CHAIN);
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for rva in &chain {
            hash = (hash ^ *rva as u64).wrapping_mul(0x0100_0000_01b3);
        }
        let times = {
            let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
            if seen.len() >= MAX_CHAINS && !seen.contains_key(&hash) {
                return;
            }
            let n = seen.entry(hash).or_insert(0);
            *n += 1;
            *n
        };
        // The first time, and then at the powers of ten.
        if 10u32.checked_pow(times.ilog10()) != Some(times) {
            return;
        }
        let value = value.map_or(String::new(), |v| format!(" (value {v:#x})"));
        let chain: Vec<String> = chain.iter().map(|r| format!("+{r:#x}")).collect();
        let pc = PC.with(Cell::get);
        let at = if pc == 0 {
            String::new()
        } else {
            format!(" (instruction {pc:#x})")
        };
        eprintln!(
            "[trap] write of {len} bytes at {addr:#x}{value}{at} by thread {}, seen {times} times: {}",
            host::thread_id(),
            chain.join(" ")
        );
    }

    /// How many writes there have been.
    pub fn hits(&self) -> u32 {
        self.hits.load(Ordering::Relaxed)
    }
}

#[cfg(windows)]
mod host {
    use std::ffi::c_void;

    /// How many words of the stack are looked at for return addresses.
    const STACK_WORDS: usize = 4096;

    extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
        fn GetCurrentThreadId() -> u32;
    }

    pub fn thread_id() -> u32 {
        // SAFETY: a plain query.
        unsafe { GetCurrentThreadId() }
    }

    /// The words of the stack above this call that point into the program's own image, the
    /// first `max` of them, as offsets from its base: the return addresses of the calls that led
    /// here, nearest first, with a few stale ones among them.
    pub fn call_chain(max: usize) -> Vec<usize> {
        let marker = 0u8;
        let here = std::ptr::addr_of!(marker) as usize & !7;
        let (mut low, mut high) = (0usize, 0usize);
        // SAFETY: a plain query of the thread's own stack.
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        // SAFETY: the handle of the program's own image.
        let base = unsafe { GetModuleHandleW(std::ptr::null()) } as usize;
        if base == 0 || here < low {
            return Vec::new();
        }
        // SAFETY: a loaded image starts with its DOS header, which says where the NT header is
        // (`e_lfanew` at 0x3C); the image size is at 0x50 of the optional header.
        let size = unsafe {
            let nt = base + *((base + 0x3C) as *const u32) as usize;
            *((nt + 0x50) as *const u32) as usize
        };
        let mut chain = Vec::new();
        let end = high.min(here + 8 * STACK_WORDS);
        let mut at = here;
        while at + 8 <= end && chain.len() < max {
            // SAFETY: inside the thread's own stack, between its current position and its base.
            let word = unsafe { *(at as *const usize) };
            if word >= base && word < base + size {
                chain.push(word - base);
            }
            at += 8;
        }
        chain
    }
}

#[cfg(not(windows))]
mod host {
    pub fn thread_id() -> u32 {
        0
    }

    pub fn call_chain(_max: usize) -> Vec<usize> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_of_ranges_is_read_as_hexadecimal_and_what_is_wrong_is_left_out() {
        let t = Traps::parse("30adb6c000:1000, 0x10:20,nonsense,5:0,zz:1");
        assert_eq!(
            t.ranges,
            [(0x0030_adb6_c000, 0x0030_adb6_d000), (0x10, 0x30)]
        );
        assert!(!t.is_empty());
        assert!(Traps::parse("").is_empty());
    }

    #[test]
    fn a_page_is_covered_when_any_of_it_is_in_a_range() {
        let t = Traps::parse("1800:1000");
        assert!(
            t.covers(0x1000),
            "the first page holds the start of the range"
        );
        assert!(t.covers(0x2000), "the second holds the end");
        assert!(!t.covers(0x3000));
        assert!(!t.covers(0x0));
    }

    #[test]
    fn a_call_chain_is_found_on_the_stack_of_a_running_program() {
        let chain = host::call_chain(CHAIN);
        if cfg!(windows) {
            // The test is in the image it asks about, so there are return addresses in it.
            assert!(!chain.is_empty());
            assert!(chain.len() <= CHAIN);
        }
    }
}
