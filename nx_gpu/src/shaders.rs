//! Which shader programs the draws use: found, parsed and counted.
//!
//! Nothing here translates a program. It reads each program a draw names out of GPU memory, parses
//! its header, follows its control flow and counts the instructions, so the report says what the
//! translator has to handle. Only counts and names of instructions are kept, never the code.

use crate::engine::Cx;
use crate::state::{StageKind, StageSlot};
use nx_shader::cfg::{Cfg, Code};
use nx_shader::sph::{Sph, SPH_BYTES};
use std::collections::BTreeMap;

/// How much of GPU memory after a program's start is read to find its end.
const WINDOW: usize = 1 << 20;

/// A program found at a GPU address.
#[derive(Debug)]
pub struct Program {
    pub kind: StageKind,
    pub address: u64,
    /// Bytes from the start of the header to the end of the last reachable instruction.
    pub size: usize,
    pub hash: u64,
    pub header: Result<Sph, String>,
    pub blocks: usize,
    pub instructions: u64,
    pub issues: Vec<String>,
    pub histogram: BTreeMap<String, u64>,
    /// Draws that used the program.
    pub uses: u64,
    /// How many times its code changed under the same address.
    pub rewrites: u64,
}

/// A fast hash of bytes: enough to notice that code or texels under an address changed, and
/// nothing more. Four lanes over 8-byte words, so that memory is the limit and not the chain of
/// multiplications.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    // SAFETY: the pointer and the length are those of a slice.
    unsafe { hash_raw(bytes.as_ptr(), bytes.len()) }
}

/// [`hash_bytes`] of `len` bytes at `ptr`.
///
/// # Safety
///
/// `ptr` must be valid for reads of `len` bytes. Another thread writing the bytes meanwhile makes
/// the hash wrong (it is a hash of what was there at some time), and nothing worse.
pub unsafe fn hash_raw(ptr: *const u8, len: usize) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut lane = [
        0x243F_6A88_85A3_08D3u64,
        0x1319_8A2E_0370_7344,
        0xA409_3822_299F_31D0,
        0x082E_FA98_EC4E_6C89,
    ];
    let mut at = 0;
    while at + 32 <= len {
        for (k, l) in lane.iter_mut().enumerate() {
            // SAFETY: `at + 32 <= len`, so the word is inside the bytes.
            let word = unsafe { (ptr.add(at + 8 * k) as *const u64).read_unaligned() };
            *l = (*l ^ word).wrapping_mul(K);
            *l ^= *l >> 29;
        }
        at += 32;
    }
    let mut h = len as u64;
    for l in lane {
        h = (h ^ l).wrapping_mul(K);
        h ^= h >> 32;
    }
    while at < len {
        // SAFETY: `at < len`.
        let byte = unsafe { *ptr.add(at) };
        h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        at += 1;
    }
    h
}

/// Read up to `max` bytes at GPU address `va`, as many as are mapped.
pub fn read_window(cx: &Cx<'_>, va: u64, max: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut at = va;
    while out.len() < max {
        let Some((cpu, avail)) = cx.space.translate(at) else {
            break;
        };
        let n = (max - out.len()).min(avail as usize);
        let start = out.len();
        out.resize(start + n, 0);
        if !cx.mem.read(cpu, &mut out[start..]) {
            out.truncate(start);
            break;
        }
        at += n as u64;
    }
    out
}

fn analyze(kind: StageKind, address: u64, bytes: &[u8]) -> Program {
    let header = Sph::parse(bytes);
    let mut program = Program {
        kind,
        address,
        size: 0,
        hash: 0,
        header,
        blocks: 0,
        instructions: 0,
        issues: Vec::new(),
        histogram: BTreeMap::new(),
        uses: 0,
        rewrites: 0,
    };
    if let Err(e) = &program.header {
        program.issues.push(e.clone());
        program.size = bytes.len().min(SPH_BYTES);
        program.hash = hash_bytes(&bytes[..program.size]);
        return program;
    }
    let code = Code(&bytes[SPH_BYTES..]);
    let cfg = Cfg::discover(code);
    program.size = SPH_BYTES + cfg.extent as usize;
    program.hash = hash_bytes(&bytes[..program.size.min(bytes.len())]);
    program.blocks = cfg.blocks.len();
    program.issues = cfg.issues.clone();
    for (op, n) in cfg.histogram(code) {
        let name = match op {
            Some(op) => format!("{op:?}"),
            None => "unknown".to_string(),
        };
        *program.histogram.entry(name).or_default() += n;
        program.instructions += n;
    }
    program
}

/// What the draws used.
#[derive(Debug, Default)]
pub struct Inventory {
    pub draws: u64,
    /// Draws by the set of stages they enabled.
    pub combinations: BTreeMap<String, u64>,
    pub programs: BTreeMap<u64, Program>,
    /// Programs that could not be read from GPU memory at all, by reason.
    pub unreadable: BTreeMap<String, u64>,
}

impl Inventory {
    /// A draw uses these stages.
    pub fn observe(&mut self, cx: &Cx<'_>, stages: &[StageSlot]) {
        self.draws += 1;
        let key = stages
            .iter()
            .map(|s| s.kind.name())
            .collect::<Vec<_>>()
            .join("+");
        *self.combinations.entry(key).or_default() += 1;
        for s in stages {
            self.observe_program(cx, s);
        }
    }

    fn observe_program(&mut self, cx: &Cx<'_>, s: &StageSlot) {
        if let Some(p) = self.programs.get_mut(&s.address) {
            // The same address: unchanged when the bytes it was made of still hash the same.
            let (hash, mapped) = cx.space.hash(cx.mem, s.address, p.size as u64);
            let same = mapped == p.size as u64 && hash == p.hash;
            if same && p.kind == s.kind {
                p.uses += 1;
                return;
            }
            let uses = p.uses + 1;
            let rewrites = p.rewrites + 1;
            if let Some(mut fresh) = self.read_program(cx, s) {
                fresh.uses = uses;
                fresh.rewrites = rewrites;
                self.programs.insert(s.address, fresh);
            }
            return;
        }
        if let Some(mut fresh) = self.read_program(cx, s) {
            fresh.uses = 1;
            self.programs.insert(s.address, fresh);
        }
    }

    fn read_program(&mut self, cx: &Cx<'_>, s: &StageSlot) -> Option<Program> {
        let bytes = read_window(cx, s.address, WINDOW);
        if bytes.len() < SPH_BYTES {
            *self
                .unreadable
                .entry(format!(
                    "{} program at {:#x}: {} bytes mapped",
                    s.kind.name(),
                    s.address,
                    bytes.len()
                ))
                .or_default() += 1;
            return None;
        }
        let mut program = analyze(s.kind, s.address, &bytes);
        // Hashed the way the draws compare it with memory.
        program.hash = cx.space.hash(cx.mem, s.address, program.size as u64).0;
        if let Some(dir) = std::env::var_os("NX_SHADER_DUMP") {
            // For looking at programs by hand: the program and, when it has problems, the window
            // that was read. The directory is the user's; nothing is stored in the repository.
            let keep = if program.issues.is_empty() {
                program.size
            } else {
                bytes.len().min(0x10000)
            };
            let name = format!(
                "{}_{:x}_{:016x}.bin",
                program.kind.name(),
                s.address,
                program.hash
            );
            let path = std::path::Path::new(&dir).join(name);
            if let Err(e) = std::fs::create_dir_all(&dir)
                .and_then(|()| std::fs::write(&path, &bytes[..keep.min(bytes.len())]))
            {
                eprintln!(
                    "[gpu] could not dump a shader program to {}: {e}",
                    path.display()
                );
            }
        }
        Some(program)
    }

    /// A summary of the programs: how many, what they are made of, what they need.
    pub fn report(&self, top: usize) -> String {
        let mut s = format!(
            "  shaders: {} draws used {} different programs\n",
            self.draws,
            self.programs.len()
        );
        for (combo, n) in &self.combinations {
            s.push_str(&format!("  {n:>8} draws with stages {combo}\n"));
        }
        for (why, n) in &self.unreadable {
            s.push_str(&format!("  {n:>8} unreadable: {why}\n"));
        }
        let mut by_kind: BTreeMap<&str, (u64, u64, u64)> = BTreeMap::new();
        let mut issues: BTreeMap<String, u64> = BTreeMap::new();
        let mut ops: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
        let mut rewritten = 0;
        for p in self.programs.values() {
            let e = by_kind.entry(p.kind.name()).or_default();
            e.0 += 1;
            e.1 += p.instructions;
            e.2 = e.2.max(p.instructions);
            rewritten += p.rewrites;
            for i in &p.issues {
                *issues.entry(i.clone()).or_default() += 1;
            }
            for (name, n) in &p.histogram {
                let e = ops.entry(name).or_default();
                e.0 += n;
                e.1 += n * p.uses;
            }
        }
        for (kind, (n, total, biggest)) in by_kind {
            s.push_str(&format!(
                "  {n:>8} {kind} programs, {total} instructions in all, the largest {biggest}\n"
            ));
        }
        if rewritten > 0 {
            s.push_str(&format!(
                "  {rewritten} times the code under a program address changed\n"
            ));
        }
        for (i, n) in &issues {
            s.push_str(&format!("  {n:>8} programs with the problem: {i}\n"));
        }
        let mut busiest: Vec<&Program> = self.programs.values().collect();
        busiest.sort_by_key(|p| (std::cmp::Reverse(p.uses), p.address));
        for p in busiest.into_iter().take(top) {
            let header = match &p.header {
                Ok(h) => format!(
                    "local memory {} B, {}call stack {} B",
                    h.local_memory_low(),
                    if h.kills_pixels() { "can kill, " } else { "" },
                    h.crs_size()
                ),
                Err(_) => "no valid header".to_string(),
            };
            s.push_str(&format!(
                "  {:>8} draws: {} at {:#x}: {} instructions in {} blocks, {} bytes, {}\n",
                p.uses,
                p.kind.name(),
                p.address,
                p.instructions,
                p.blocks,
                p.size,
                header
            ));
        }
        let mut sorted: Vec<_> = ops.into_iter().collect();
        sorted.sort_by_key(|(name, (n, w))| (std::cmp::Reverse(*w), std::cmp::Reverse(*n), *name));
        s.push_str(&format!(
            "  {} different instructions; static count, count weighted by draws:\n",
            sorted.len()
        ));
        for (name, (n, w)) in sorted.iter().take(top) {
            s.push_str(&format!("  {n:>8} {w:>10} {name}\n"));
        }
        s
    }
}
