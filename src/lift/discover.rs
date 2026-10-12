//! Finding the code: functions and their control-flow graphs, by recursive descent.
//!
//! Starting from the roots (entry points, exports, init/fini arrays, `.eh_frame` function starts,
//! code pointers in relocations, and addresses a previous run reported), every reachable
//! instruction is decoded. Nothing is decoded by linear sweep, so literal pools and tables after
//! an unconditional branch are never mistaken for code. What cannot be followed is listed in the
//! [`Report`], never guessed:
//! - a `BR` whose register no jump table explains goes through the function table at run time;
//! - a call or branch to an address that is not executable keeps its target and stops the run
//!   when reached;
//! - an instruction the lifter has no translation for becomes an interpreter fallback and is
//!   listed with its address.
//!
//! A function is whatever is reachable from its entry through branches inside it. A `BL` target
//! is a new function; a `B` that leaves the function is a tail call to a new function (leaving
//! means leaving the function's `.eh_frame` range when one is known, otherwise reaching another
//! root). Code shared by two functions is lifted once in each.

use super::cfg::{traps, End, FuncCfg, GuestBlock, Target};
use super::ir::TrapKind;
use super::jumptable;
use nx_runtime::a64::decode::decode;
use nx_runtime::a64::inst::{BranchReg, Inst};
use std::collections::{BTreeMap, BTreeSet};

/// The program image the analysis reads.
pub trait CodeSource {
    /// Is `addr` in executable memory?
    fn is_code(&self, addr: u64) -> bool;
    /// A little-endian read of `size` (1 to 8) bytes from any mapped part of the image.
    fn read(&self, addr: u64, size: usize) -> Option<u64>;

    /// The instruction word at `addr`, if it is executable.
    fn word(&self, addr: u64) -> Option<u32> {
        if addr.is_multiple_of(4) && self.is_code(addr) {
            self.read(addr, 4).map(|v| v as u32)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Function extents `(start, end)`, for example from `.eh_frame_hdr`.
    pub ranges: Vec<(u64, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JumpTable {
    pub br: u64,
    pub targets: Vec<u64>,
}

/// What the analysis could not settle, for the report.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub jump_tables: Vec<JumpTable>,
    /// `BR` instructions no table explains.
    pub unresolved_branches: Vec<u64>,
    /// Calls and branches to addresses that are not executable: `(instruction, target)`.
    pub bad_targets: Vec<(u64, u64)>,
    /// Straight-line code that runs into non-executable memory: the first bad address.
    pub out_of_code: Vec<u64>,
    /// Instructions executed by the interpreter, with the reason.
    pub fallbacks: BTreeMap<u64, &'static str>,
    /// Instructions that stop the run when executed.
    pub traps: Vec<(u64, TrapKind)>,
}

#[derive(Clone, Debug, Default)]
pub struct Discovery {
    pub funcs: BTreeMap<u64, FuncCfg>,
    pub report: Report,
}

struct Ranges<'a>(&'a [(u64, u64)]);

impl Ranges<'_> {
    /// The extent that contains `addr`.
    fn of(&self, addr: u64) -> Option<(u64, u64)> {
        let i = self.0.partition_point(|&(start, _)| start <= addr);
        let &(lo, hi) = self.0.get(i.checked_sub(1)?)?;
        (addr < hi).then_some((lo, hi))
    }
}

struct Build<'a> {
    code: &'a dyn CodeSource,
    entry: u64,
    bounds: Option<(u64, u64)>,
    is_root: &'a dyn Fn(u64) -> bool,
    insts: BTreeMap<u64, Inst>,
    leaders: BTreeSet<u64>,
    /// Addresses that are not executable but were reached.
    bad: BTreeSet<u64>,
    /// Addresses straight-line code fell off the end of the function's extent at.
    fell_off: BTreeSet<u64>,
    work: Vec<u64>,
    calls: Vec<(u64, u64)>,
    new_roots: BTreeSet<u64>,
    brs: Vec<(u64, u8, usize)>,
    tables: BTreeMap<u64, Vec<u64>>,
}

impl Build<'_> {
    fn in_bounds(&self, a: u64) -> bool {
        match self.bounds {
            Some((lo, hi)) => lo <= a && a < hi,
            None => true,
        }
    }

    /// Does a branch to `t` leave the function?
    fn is_tail(&self, t: u64) -> bool {
        match self.bounds {
            Some((lo, hi)) => !(lo <= t && t < hi),
            None => t != self.entry && (self.is_root)(t),
        }
    }

    fn add_block(&mut self, a: u64) {
        if self.leaders.insert(a) {
            self.work.push(a);
        }
    }

    fn explore(&mut self) {
        while let Some(start) = self.work.pop() {
            let mut a = start;
            loop {
                if a != start && self.insts.contains_key(&a) {
                    self.leaders.insert(a);
                    break;
                }
                if self.insts.contains_key(&a) || self.bad.contains(&a) {
                    break;
                }
                if !self.in_bounds(a) {
                    // Straight-line code left the function: continue as a tail call.
                    self.fell_off.insert(a);
                    self.new_roots.insert(a);
                    break;
                }
                let Some(word) = self.code.word(a) else {
                    self.bad.insert(a);
                    break;
                };
                let inst = decode(word);
                self.insts.insert(a, inst);
                match inst {
                    Inst::B { link, imm } => {
                        let t = a.wrapping_add(imm as u64);
                        if link {
                            self.calls.push((a, t));
                            self.add_block(a + 4);
                        } else if self.is_tail(t) {
                            self.new_roots.insert(t);
                        } else {
                            self.add_block(t);
                        }
                        break;
                    }
                    Inst::BCond { imm, .. } | Inst::Cbz { imm, .. } | Inst::Tbz { imm, .. } => {
                        let t = a.wrapping_add(imm as u64);
                        if self.is_tail(t) {
                            self.new_roots.insert(t);
                        } else {
                            self.add_block(t);
                        }
                        self.add_block(a + 4);
                        break;
                    }
                    Inst::BranchReg { kind, rn } => {
                        match kind {
                            BranchReg::Blr => self.add_block(a + 4),
                            BranchReg::Br => self.brs.push((a, rn, usize::MAX)),
                            BranchReg::Ret => {}
                        }
                        break;
                    }
                    Inst::Svc { .. } | Inst::Unsupported { .. } => {
                        self.add_block(a + 4);
                        break;
                    }
                    _ => {
                        if traps(&inst).is_some() {
                            break;
                        }
                    }
                }
                a += 4;
            }
        }
    }

    /// Find the jump tables behind the `BR` instructions decoded so far. Returns whether any new
    /// code became reachable.
    fn resolve_tables(&mut self) -> bool {
        let mut progress = false;
        for i in 0..self.brs.len() {
            let (pc, rn, tried_at) = self.brs[i];
            if self.tables.contains_key(&pc) || tried_at == self.insts.len() {
                continue;
            }
            self.brs[i].2 = self.insts.len();
            let bounds = self.bounds;
            let code = self.code;
            let inside =
                move |t: u64| code.is_code(t) && bounds.is_none_or(|(lo, hi)| lo <= t && t < hi);
            if let Some(targets) = jumptable::resolve(self.code, &self.insts, pc, rn, &inside) {
                for &t in &targets {
                    self.add_block(t);
                }
                self.tables.insert(pc, targets);
                progress = true;
            }
        }
        progress
    }

    fn run(&mut self) {
        self.add_block(self.entry);
        loop {
            self.explore();
            if !self.resolve_tables() && self.work.is_empty() {
                break;
            }
        }
    }

    fn terminator(&self, a: u64, inst: &Inst) -> Option<End> {
        let target = |t: u64| {
            if self.is_tail(t) {
                Target::Tail(t)
            } else {
                Target::Block(t)
            }
        };
        match *inst {
            Inst::B { link: true, .. } => Some(End::Call { ret: a + 4 }),
            Inst::B { imm, .. } => Some(End::Jump(target(a.wrapping_add(imm as u64)))),
            Inst::BCond { imm, .. } | Inst::Cbz { imm, .. } | Inst::Tbz { imm, .. } => {
                Some(End::Cond {
                    taken: target(a.wrapping_add(imm as u64)),
                    fall: a + 4,
                })
            }
            Inst::BranchReg { kind, .. } => Some(match kind {
                BranchReg::Blr => End::Call { ret: a + 4 },
                BranchReg::Ret => End::Return,
                BranchReg::Br => End::Indirect {
                    targets: self.tables.get(&a).cloned().unwrap_or_default(),
                },
            }),
            Inst::Svc { .. } | Inst::Unsupported { .. } => Some(End::Call { ret: a + 4 }),
            _ => traps(inst).map(|_| End::Trap),
        }
    }

    fn into_cfg(self) -> (FuncCfg, BuildOut) {
        let mut blocks = BTreeMap::new();
        for &l in &self.leaders {
            if self.bad.contains(&l) {
                blocks.insert(
                    l,
                    GuestBlock {
                        addr: l,
                        insts: Vec::new(),
                        end: End::OutOfCode,
                    },
                );
                continue;
            }
            if self.fell_off.contains(&l) {
                blocks.insert(
                    l,
                    GuestBlock {
                        addr: l,
                        insts: Vec::new(),
                        end: End::Jump(Target::Tail(l)),
                    },
                );
                continue;
            }
            let mut insts = Vec::new();
            let mut a = l;
            let end = loop {
                let inst = self.insts[&a];
                insts.push((a, inst));
                if let Some(e) = self.terminator(a, &inst) {
                    break e;
                }
                a += 4;
                if self.leaders.contains(&a) {
                    break End::Fall(a);
                }
            };
            blocks.insert(
                l,
                GuestBlock {
                    addr: l,
                    insts,
                    end,
                },
            );
        }
        let out = BuildOut {
            calls: self.calls,
            new_roots: self.new_roots,
            tables: self.tables,
            brs: self.brs.iter().map(|&(pc, _, _)| pc).collect(),
            bad: self.bad,
        };
        (
            FuncCfg {
                entry: self.entry,
                blocks,
            },
            out,
        )
    }
}

struct BuildOut {
    calls: Vec<(u64, u64)>,
    new_roots: BTreeSet<u64>,
    tables: BTreeMap<u64, Vec<u64>>,
    brs: Vec<u64>,
    bad: BTreeSet<u64>,
}

fn build(
    code: &dyn CodeSource,
    entry: u64,
    ranges: &Ranges,
    is_root: &dyn Fn(u64) -> bool,
) -> (FuncCfg, BuildOut) {
    let mut b = Build {
        code,
        entry,
        bounds: ranges.of(entry),
        is_root,
        insts: BTreeMap::new(),
        leaders: BTreeSet::new(),
        bad: BTreeSet::new(),
        fell_off: BTreeSet::new(),
        work: Vec::new(),
        calls: Vec::new(),
        new_roots: BTreeSet::new(),
        brs: Vec::new(),
        tables: BTreeMap::new(),
    };
    b.run();
    b.into_cfg()
}

/// Discover every function reachable from `roots`.
pub fn discover(code: &dyn CodeSource, roots: &[u64], opts: &Options) -> Discovery {
    let mut ranges_sorted = opts.ranges.clone();
    ranges_sorted.sort_unstable();
    let ranges = Ranges(&ranges_sorted);
    let mut report = Report::default();

    // Pass one: find every function. Branches that stay inside an unbounded function are followed,
    // so the over-approximated bodies still reveal their callees.
    let mut known: BTreeSet<u64> = BTreeSet::new();
    let mut queue: Vec<u64> = Vec::new();
    for &r in roots {
        if code.word(r).is_some() && known.insert(r) {
            queue.push(r);
        }
    }
    while let Some(entry) = queue.pop() {
        let (_, out) = build(code, entry, &ranges, &|t| known.contains(&t));
        let callees = out
            .calls
            .iter()
            .map(|&(_, t)| t)
            .chain(out.new_roots.iter().copied());
        for t in callees {
            if code.word(t).is_some() && known.insert(t) {
                queue.push(t);
            }
        }
    }

    // Pass two: with the final set of entries known, build the graphs for good.
    let mut funcs = BTreeMap::new();
    for &entry in &known {
        let (cfg, out) = build(code, entry, &ranges, &|t| known.contains(&t));
        for &(pc, t) in &out.calls {
            if code.word(t).is_none() {
                report.bad_targets.push((pc, t));
            }
        }
        for (&pc, t) in &out.tables {
            report.jump_tables.push(JumpTable {
                br: pc,
                targets: t.clone(),
            });
        }
        for pc in &out.brs {
            if !out.tables.contains_key(pc) {
                report.unresolved_branches.push(*pc);
            }
        }
        report.out_of_code.extend(out.bad.iter().copied());
        for block in cfg.blocks.values() {
            for &(pc, ref inst) in &block.insts {
                if let Inst::Unsupported { what, .. } = inst {
                    report.fallbacks.insert(pc, what);
                }
                if let Some(t) = traps(inst) {
                    report.traps.push((pc, t));
                }
                if let Inst::B { imm, .. } = inst {
                    let t = pc.wrapping_add(*imm as u64);
                    if code.word(t).is_none() {
                        report.bad_targets.push((pc, t));
                    }
                }
            }
        }
        funcs.insert(entry, cfg);
    }
    report.jump_tables.sort_by_key(|t| t.br);
    report.jump_tables.dedup();
    report.unresolved_branches.sort_unstable();
    report.unresolved_branches.dedup();
    report.bad_targets.sort_unstable();
    report.bad_targets.dedup();
    report.out_of_code.sort_unstable();
    report.out_of_code.dedup();
    report.traps.sort_by_key(|t| t.0);
    report.traps.dedup();
    Discovery { funcs, report }
}
