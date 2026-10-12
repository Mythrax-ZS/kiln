//! The engines a channel drives (3D, compute, 2D, copy, inline-to-memory) and the context they
//! run in.
//!
//! Each engine is a register file addressed by method. What a write does is implemented where the
//! effect is visible to the program: the memory a semaphore release writes, the code a macro upload
//! stores. Every other write is recorded and counted, which is what the tracing processor is for.

use crate::event::EventToken;
use crate::gpu::class;
use crate::mem::GuestMemory;
use crate::mme;
use crate::names;
use crate::syncpt::Syncpoints;
use crate::vm::AddressSpace;
use std::collections::BTreeMap;

/// How many registers (words) an engine has: methods are 12 bits wide.
pub const REGISTERS: usize = 0x1000;

/// The draws and clears the 3D engine was asked for. Nothing is drawn: this is what a renderer
/// would be given.
#[derive(Debug, Default)]
pub struct DrawStats {
    /// Draws by the method that started them.
    pub by_kind: BTreeMap<&'static str, u64>,
    /// The vertices or indices the draws named, in all.
    pub elements: u64,
    /// `CLEAR_SURFACE` writes.
    pub clears: u64,
    /// The first draws, each described (kept for the report).
    pub first: Vec<String>,
}

impl DrawStats {
    pub fn draws(&self) -> u64 {
        self.by_kind.values().sum()
    }
}

/// What the processor counted.
#[derive(Debug, Default)]
pub struct Stats {
    pub submits: u64,
    pub entries: u64,
    pub words: u64,
    pub methods: u64,
    /// Writes per (class, byte offset of the method).
    pub per_method: BTreeMap<(u32, u32), u64>,
    /// Macro calls per macro number: (calls, parameters).
    pub macro_calls: BTreeMap<u32, (u64, u64)>,
    /// Instructions the macros ran.
    pub macro_steps: u64,
    /// Method writes the macros made, per (class, byte offset of the method).
    pub macro_writes: BTreeMap<(u32, u32), u64>,
    pub draws: DrawStats,
    pub semaphore_releases: u64,
}

impl Stats {
    /// A report of the counts: totals, then the most written methods.
    pub fn report(&self, top: usize) -> String {
        let mut s = format!(
            "GPU commands: {} submissions, {} pushbuffer entries, {} words, {} method writes, \
             {} semaphore releases\n",
            self.submits, self.entries, self.words, self.methods, self.semaphore_releases
        );
        let mut by_count: Vec<_> = self.per_method.iter().collect();
        by_count.sort_by_key(|(k, c)| (std::cmp::Reverse(**c), **k));
        for ((class_num, offset), count) in by_count.into_iter().take(top) {
            s.push_str(&format!(
                "  {count:>8} x {} {}\n",
                names::class_name(*class_num),
                names::method_name(*class_num, *offset)
            ));
        }
        if !self.macro_calls.is_empty() {
            let total: u64 = self.macro_calls.values().map(|(c, _)| c).sum();
            s.push_str(&format!(
                "  macro calls: {total} calls of {} different macros\n",
                self.macro_calls.len()
            ));
            let mut macros: Vec<_> = self.macro_calls.iter().collect();
            macros.sort_by_key(|(k, (c, _))| (std::cmp::Reverse(*c), **k));
            for (id, (calls, params)) in macros.into_iter().take(top) {
                s.push_str(&format!(
                    "  {calls:>8} x macro {id} ({params} parameters in all)\n"
                ));
            }
            let written: u64 = self.macro_writes.values().sum();
            s.push_str(&format!(
                "  macros ran {} instructions and wrote {written} methods, most often:\n",
                self.macro_steps
            ));
            let mut by_count: Vec<_> = self.macro_writes.iter().collect();
            by_count.sort_by_key(|(k, c)| (std::cmp::Reverse(**c), **k));
            for ((class_num, offset), count) in by_count.into_iter().take(top) {
                s.push_str(&format!(
                    "  {count:>8} x {} {}\n",
                    names::class_name(*class_num),
                    names::method_name(*class_num, *offset)
                ));
            }
        }
        let d = &self.draws;
        s.push_str(&format!(
            "  draws: {} ({} elements named), clears: {}\n",
            d.draws(),
            d.elements,
            d.clears
        ));
        for (kind, count) in &d.by_kind {
            s.push_str(&format!("  {count:>8} x {kind}\n"));
        }
        for first in &d.first {
            s.push_str(&format!("  first draws: {first}\n"));
        }
        s
    }
}

/// What a method write may touch.
pub struct Cx<'a> {
    pub mem: &'a dyn GuestMemory,
    pub space: &'a AddressSpace,
    pub syncpoints: &'a mut Syncpoints,
    /// Tokens of the events the syncpoint increments of the commands have set.
    pub woken: &'a mut Vec<EventToken>,
    pub stats: &'a mut Stats,
    pub channel: u32,
    /// The channel's syncpoint.
    pub syncpt: u32,
    /// 0 none, 1 submissions, 2 every method.
    pub level: u8,
}

impl Cx<'_> {
    pub fn read_u32(&self, va: u64) -> Result<u32, String> {
        let mut b = [0u8; 4];
        if self.space.read(self.mem, va, &mut b) {
            Ok(u32::from_le_bytes(b))
        } else {
            Err(format!(
                "a read of GPU address {va:#x}, which is not mapped"
            ))
        }
    }

    pub fn write(&self, va: u64, data: &[u8]) -> Result<(), String> {
        if self.space.write(self.mem, va, data) {
            Ok(())
        } else {
            Err(format!(
                "a write of {} bytes to GPU address {va:#x}, which is not mapped",
                data.len()
            ))
        }
    }

    /// The GPU's timestamp counter, in nanoseconds since the first use (docs/ASSUMPTIONS.md).
    pub fn timestamp(&self) -> u64 {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// The macro memory of an engine: code, and where each macro starts.
#[derive(Debug, Default)]
pub struct Mme {
    pub code: Vec<u32>,
    code_ptr: usize,
    pub starts: Vec<u32>,
    start_ptr: usize,
}

/// Methods of the 3D engine handled specially, as word indices.
mod m3d {
    pub const LOAD_MME_INSTRUCTION_RAM_POINTER: u32 = 0x114 / 4;
    pub const LOAD_MME_INSTRUCTION_RAM: u32 = 0x118 / 4;
    pub const LOAD_MME_START_ADDRESS_RAM_POINTER: u32 = 0x11c / 4;
    pub const LOAD_MME_START_ADDRESS_RAM: u32 = 0x120 / 4;
    pub const REPORT_SEMAPHORE_A: u32 = 0x1b00 / 4;
    pub const REPORT_SEMAPHORE_D: u32 = 0x1b0c / 4;
    /// `CALL_MME_MACRO(j)` is `0x3800 + 8 j`, `CALL_MME_DATA(j)` is `0x3804 + 8 j`.
    pub const CALL_MME_FIRST: u32 = 0x3800 / 4;
    pub const CALL_MME_LAST: u32 = 0x3ffc / 4;
    // Draws.
    pub const DRAW_ZERO_INDEX: u32 = 0x304 / 4;
    pub const SET_VERTEX_ARRAY_START: u32 = 0xd74 / 4;
    pub const DRAW_VERTEX_ARRAY: u32 = 0xd78 / 4;
    pub const DRAW_VERTEX_ARRAY_BEGIN_END_FIRST: u32 = 0x1214 / 4;
    pub const DRAW_VERTEX_ARRAY_BEGIN_END_SUBSEQUENT: u32 = 0x1218 / 4;
    pub const DRAW_AUTO: u32 = 0x123c / 4;
    pub const DRAW_INLINE_INDEX4X8: u32 = 0x1304 / 4;
    pub const DRAW_INLINE_INDEX: u32 = 0x15e8 / 4;
    pub const DRAW_INLINE_INDEX2X16: u32 = 0x15f0 / 4;
    pub const END: u32 = 0x1614 / 4;
    pub const BEGIN: u32 = 0x1618 / 4;
    pub const DRAW_INLINE_VERTEX: u32 = 0x1640 / 4;
    pub const DRAW_INDEX_BUFFER: u32 = 0x17e0 / 4;
    /// `DRAW_INDEX_BUFFER32/16/8_BEGIN_END_INSTANCE_FIRST` then the same three `_SUBSEQUENT`.
    pub const DRAW_INDEX_BEGIN_END_FIRST: u32 = 0x17e4 / 4;
    pub const DRAW_INDEX_BEGIN_END_LAST: u32 = 0x17f8 / 4;
    pub const CLEAR_SURFACE: u32 = 0x19d0 / 4;
    pub const SET_COLOR_TARGET_A: u32 = 0x800 / 4;
}

#[derive(Debug)]
pub struct Engine {
    pub class: u32,
    regs: Vec<u32>,
    pub mme: Mme,
    /// A macro call being collected: its number and the parameters so far.
    pending: Option<(u32, Vec<u32>)>,
    /// The macro that is running, while one is.
    running: Option<u32>,
    /// The primitive type of the last `BEGIN`.
    topology: u32,
    /// Vertices or indices sent inline since the last `BEGIN`.
    inline_elements: u32,
}

/// The name of a primitive type (`BEGIN`'s operation; the topology field of the combined draws).
fn topology_name(op: u32) -> String {
    const NAMES: [&str; 15] = [
        "points",
        "lines",
        "line loop",
        "line strip",
        "triangles",
        "triangle strip",
        "triangle fan",
        "quads",
        "quad strip",
        "polygon",
        "line list with adjacency",
        "line strip with adjacency",
        "triangle list with adjacency",
        "triangle strip with adjacency",
        "patches",
    ];
    NAMES
        .get(op as usize)
        .map_or_else(|| format!("primitive {op}"), |n| (*n).to_string())
}

/// What a macro sees of the engine it runs in.
struct MacroHost<'a, 'b> {
    engine: &'a mut Engine,
    cx: &'a mut Cx<'b>,
    id: u32,
}

impl mme::Host for MacroHost<'_, '_> {
    fn state(&mut self, method: u32) -> u32 {
        self.engine.reg(method / 4)
    }

    fn emit(&mut self, method: u32, data: u32) -> Result<(), String> {
        let word = method / 4;
        if (m3d::CALL_MME_FIRST..=m3d::CALL_MME_LAST).contains(&word) {
            return Err(format!(
                "macro {} called a macro (a write to {})",
                self.id,
                names::method_name(class::THREED, method)
            ));
        }
        *self
            .cx
            .stats
            .macro_writes
            .entry((class::THREED, method))
            .or_default() += 1;
        if self.cx.level >= 2 {
            eprintln!(
                "[gpu]   macro {} writes {} = {data:#x}",
                self.id,
                names::method_name(class::THREED, method)
            );
        }
        self.engine.write(self.cx, word, data, false)
    }
}

impl Engine {
    pub fn new(class_num: u32) -> Engine {
        Engine {
            class: class_num,
            regs: vec![0; REGISTERS],
            mme: Mme::default(),
            pending: None,
            running: None,
            topology: 0,
            inline_elements: 0,
        }
    }

    pub fn reg(&self, method: u32) -> u32 {
        self.regs.get(method as usize).copied().unwrap_or(0)
    }

    /// A write to `method` (a word index). `last` marks the last word of its command.
    pub fn write(
        &mut self,
        cx: &mut Cx<'_>,
        method: u32,
        data: u32,
        last: bool,
    ) -> Result<(), String> {
        let slot = self
            .regs
            .get_mut(method as usize)
            .ok_or_else(|| format!("method {method:#x} is outside the engine"))?;
        *slot = data;
        if self.class == class::THREED {
            self.write_3d(cx, method, data, last)?;
        }
        Ok(())
    }

    fn write_3d(
        &mut self,
        cx: &mut Cx<'_>,
        method: u32,
        data: u32,
        last: bool,
    ) -> Result<(), String> {
        match method {
            m3d::LOAD_MME_INSTRUCTION_RAM_POINTER => self.mme.code_ptr = data as usize,
            m3d::LOAD_MME_INSTRUCTION_RAM => {
                let at = self.mme.code_ptr;
                if self.mme.code.len() <= at {
                    self.mme.code.resize(at + 1, 0);
                }
                self.mme.code[at] = data;
                self.mme.code_ptr += 1;
            }
            m3d::LOAD_MME_START_ADDRESS_RAM_POINTER => self.mme.start_ptr = data as usize,
            m3d::LOAD_MME_START_ADDRESS_RAM => {
                let at = self.mme.start_ptr;
                if self.mme.starts.len() <= at {
                    self.mme.starts.resize(at + 1, 0);
                }
                self.mme.starts[at] = data;
                self.mme.start_ptr += 1;
            }
            // The semaphore is released when the last of its four registers is written.
            m3d::REPORT_SEMAPHORE_D => self.report_semaphore(cx, data)?,
            m3d::CALL_MME_FIRST..=m3d::CALL_MME_LAST => {
                let j = (method - m3d::CALL_MME_FIRST) / 2;
                if (method - m3d::CALL_MME_FIRST).is_multiple_of(2) {
                    // A call begins: a call still open is finished first.
                    if self.pending.is_some() {
                        self.finish_macro(cx)?;
                    }
                    self.pending = Some((j, vec![data]));
                } else {
                    match &mut self.pending {
                        Some((id, params)) if *id == j => params.push(data),
                        _ => {
                            return Err(format!(
                                "a parameter for macro {j} with no call of that macro open"
                            ))
                        }
                    }
                }
                if last {
                    self.finish_macro(cx)?;
                }
            }
            m3d::BEGIN => {
                self.topology = data & 0xFFFF;
                self.inline_elements = 0;
            }
            // Vertices and indices sent in the method stream are drawn when the primitive ends.
            m3d::DRAW_INLINE_VERTEX | m3d::DRAW_INLINE_INDEX => self.inline_elements += 1,
            m3d::DRAW_INLINE_INDEX2X16 => self.inline_elements += 2,
            m3d::DRAW_INLINE_INDEX4X8 => self.inline_elements += 4,
            m3d::END => {
                if self.inline_elements > 0 {
                    let count = std::mem::take(&mut self.inline_elements);
                    self.note_draw(cx, "inline vertices or indices", 0, count, self.topology);
                }
            }
            m3d::DRAW_VERTEX_ARRAY => {
                let first = self.reg(m3d::SET_VERTEX_ARRAY_START);
                self.note_draw(cx, "DRAW_VERTEX_ARRAY", first, data, self.topology);
            }
            m3d::DRAW_VERTEX_ARRAY_BEGIN_END_FIRST
            | m3d::DRAW_VERTEX_ARRAY_BEGIN_END_SUBSEQUENT => {
                let kind = if method == m3d::DRAW_VERTEX_ARRAY_BEGIN_END_FIRST {
                    "DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST"
                } else {
                    "DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT"
                };
                self.note_draw(cx, kind, data & 0xFFFF, (data >> 16) & 0xFFF, data >> 28);
            }
            m3d::DRAW_INDEX_BUFFER => {
                self.note_draw(cx, "DRAW_INDEX_BUFFER", 0, data, self.topology);
            }
            m3d::DRAW_INDEX_BEGIN_END_FIRST..=m3d::DRAW_INDEX_BEGIN_END_LAST => {
                const KINDS: [&str; 6] = [
                    "DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST",
                    "DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST",
                    "DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST",
                    "DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT",
                    "DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT",
                    "DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT",
                ];
                let kind = KINDS[(method - m3d::DRAW_INDEX_BEGIN_END_FIRST) as usize];
                self.note_draw(cx, kind, data & 0xFFFF, (data >> 16) & 0xFFF, data >> 28);
            }
            m3d::DRAW_AUTO => self.note_draw(cx, "DRAW_AUTO", 0, data, self.topology),
            m3d::DRAW_ZERO_INDEX => self.note_draw(cx, "DRAW_ZERO_INDEX", 0, data, self.topology),
            m3d::CLEAR_SURFACE => cx.stats.draws.clears += 1,
            _ => {}
        }
        Ok(())
    }

    /// A draw was asked for: count it, and describe the first few.
    fn note_draw(
        &mut self,
        cx: &mut Cx<'_>,
        kind: &'static str,
        first: u32,
        count: u32,
        topology: u32,
    ) {
        let d = &mut cx.stats.draws;
        *d.by_kind.entry(kind).or_default() += 1;
        d.elements += u64::from(count);
        let number = d.draws();
        if number <= 3 {
            let target = (u64::from(self.reg(m3d::SET_COLOR_TARGET_A) & 0xFF) << 32)
                | u64::from(self.reg(m3d::SET_COLOR_TARGET_A + 1));
            let via = match self.running {
                Some(id) => format!("macro {id}"),
                None => "the method stream".to_string(),
            };
            let text = format!(
                "#{number} {kind}, {} first {first}, count {count}, color target 0 at {target:#x} \
                 {}x{}, from {via}",
                topology_name(topology),
                self.reg(m3d::SET_COLOR_TARGET_A + 2),
                self.reg(m3d::SET_COLOR_TARGET_A + 3),
            );
            if cx.level >= 1 {
                eprintln!("[gpu] draw {text}");
            }
            d.first.push(text);
        }
    }

    /// A macro call is complete: it is counted and run.
    fn finish_macro(&mut self, cx: &mut Cx<'_>) -> Result<(), String> {
        let Some((id, params)) = self.pending.take() else {
            return Ok(());
        };
        let e = cx.stats.macro_calls.entry(id).or_default();
        e.0 += 1;
        e.1 += params.len() as u64;
        let Some(start) = self.mme.starts.get(id as usize).copied() else {
            return Err(format!("macro {id} was called and has no start address"));
        };
        if cx.level >= 2 {
            eprintln!(
                "[gpu]   macro {id} (code at {start:#x}) called with {} parameters: {:x?}",
                params.len(),
                &params[..params.len().min(8)]
            );
        }
        let code = self.mme.code.clone();
        self.running = Some(id);
        let outcome = {
            let mut host = MacroHost {
                engine: &mut *self,
                cx: &mut *cx,
                id,
            };
            mme::run(&code, start as usize, &params, &mut host)
        };
        self.running = None;
        match outcome {
            Ok(summary) => {
                cx.stats.macro_steps += summary.steps;
                Ok(())
            }
            Err(fault) => {
                if cx.level >= 1 {
                    eprintln!(
                        "[gpu] macro {id} stopped ({fault}); its code from instruction {start}:\n{}",
                        mme::listing(&code, start as usize, 40)
                    );
                }
                Err(format!(
                    "macro {id} (code at instruction {start}, {} parameters): {fault}",
                    params.len()
                ))
            }
        }
    }

    /// `SET_REPORT_SEMAPHORE_A..D`: release the payload (and a timestamp) to memory.
    fn report_semaphore(&mut self, cx: &mut Cx<'_>, d: u32) -> Result<(), String> {
        let a = self.reg(m3d::REPORT_SEMAPHORE_A);
        let b = self.reg(m3d::REPORT_SEMAPHORE_A + 1);
        let payload = self.reg(m3d::REPORT_SEMAPHORE_A + 2);
        let va = (u64::from(a & 0xFF) << 32) | u64::from(b);
        let operation = d & 3;
        let one_word = d >> 28 & 1 != 0;
        let report = d >> 23 & 0x1F;
        match operation {
            // RELEASE, or REPORT_ONLY of "no report": the payload. Reports of counters (pixels
            // drawn, invocations) are zero, as nothing has been drawn.
            0 | 2 => {
                let value = if report == 0 { payload } else { 0 };
                if one_word {
                    cx.write(va, &value.to_le_bytes())?;
                } else {
                    let mut out = [0u8; 16];
                    out[..4].copy_from_slice(&value.to_le_bytes());
                    out[8..].copy_from_slice(&cx.timestamp().to_le_bytes());
                    cx.write(va, &out)?;
                }
                cx.stats.semaphore_releases += 1;
                Ok(())
            }
            // ACQUIRE: continue when the memory has the value (>= with bit 16 set).
            1 => {
                let have = cx.read_u32(va)?;
                let ge = d >> 16 & 1 != 0;
                if (ge && have >= payload) || (!ge && have == payload) {
                    Ok(())
                } else {
                    Err(format!(
                        "a semaphore acquire at GPU address {va:#x} for {payload:#x}, which holds \
                         {have:#x} and nothing is left to change it"
                    ))
                }
            }
            _ => Err(format!("report semaphore operation {operation} (TRAP)")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::testing::TestMemory;
    use crate::vm::{map_flags, MapRequest};

    /// An address space with 0x1000 bytes of memory at GPU address 0x500_0000.
    fn world() -> (TestMemory, AddressSpace, Syncpoints) {
        let mem = TestMemory::new();
        mem.map(0x1000_0000, 0x1000);
        let mut space = AddressSpace::new();
        space
            .map_buffer(&MapRequest {
                flags: map_flags::FIXED_OFFSET,
                kind: 0,
                handle: 1,
                handle_cpu: 0x1000_0000,
                handle_size: 0x1000,
                page_size: 0,
                buffer_offset: 0,
                mapping_size: 0,
                offset: 0x500_0000,
            })
            .unwrap();
        (mem, space, Syncpoints::new())
    }

    fn with_cx<R>(f: impl FnOnce(&mut Cx<'_>) -> R) -> (R, TestMemory, Stats) {
        let (mem, space, mut sp) = world();
        let mut woken = Vec::new();
        let mut stats = Stats::default();
        let r = {
            let mut cx = Cx {
                mem: &mem,
                space: &space,
                syncpoints: &mut sp,
                woken: &mut woken,
                stats: &mut stats,
                channel: 1,
                syncpt: 1,
                level: 0,
            };
            f(&mut cx)
        };
        (r, mem, stats)
    }

    #[test]
    fn writes_are_kept_in_the_register_file() {
        let (r, _, _) = with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            e.write(cx, 0x1234 / 4, 77, true).unwrap();
            e.reg(0x1234 / 4)
        });
        assert_eq!(r, 77);
    }

    #[test]
    fn macro_memory_is_loaded_through_the_pointers() {
        with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            e.write(cx, 0x114 / 4, 2, false).unwrap();
            for w in [0xAA, 0xBB, 0xCC] {
                e.write(cx, 0x118 / 4, w, false).unwrap();
            }
            assert_eq!(e.mme.code, vec![0, 0, 0xAA, 0xBB, 0xCC]);
            e.write(cx, 0x11c / 4, 1, false).unwrap();
            e.write(cx, 0x120 / 4, 0x40, false).unwrap();
            e.write(cx, 0x120 / 4, 0x50, false).unwrap();
            assert_eq!(e.mme.starts, vec![0, 0x40, 0x50]);
        });
    }

    /// Load macro code and set the start address of macro `id`.
    fn load_macro(cx: &mut Cx<'_>, e: &mut Engine, id: u32, at: u32, code: &[u32]) {
        e.write(cx, 0x114 / 4, at, false).unwrap();
        for w in code {
            e.write(cx, 0x118 / 4, *w, false).unwrap();
        }
        e.write(cx, 0x11c / 4, id, false).unwrap();
        e.write(cx, 0x120 / 4, at, false).unwrap();
    }

    /// A macro that reads `n` parameters and does nothing else.
    fn reader(n: usize) -> Vec<u32> {
        use crate::mme::asm::*;
        let mut code: Vec<u32> = (1..n).map(|_| alu(ADD, LOAD, 2, 0, 0)).collect();
        match code.last_mut() {
            Some(last) => *last |= END_NEXT,
            None => code.push(alu(ADD, MOVE, 2, 0, 0) | END_NEXT),
        }
        code.push(alu(ADD, MOVE, 2, 0, 0));
        code
    }

    #[test]
    fn a_macro_call_collects_its_parameters_until_the_command_ends() {
        let ((), _, stats) = with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            load_macro(cx, &mut e, 3, 0, &reader(3));
            load_macro(cx, &mut e, 0, 8, &reader(1));
            // CALL_MME_MACRO(3) is word 0xE06, CALL_MME_DATA(3) the next.
            e.write(cx, 0xE06, 9, false).unwrap();
            e.write(cx, 0xE07, 1, false).unwrap();
            e.write(cx, 0xE07, 2, true).unwrap();
            // A single-word call.
            e.write(cx, 0xE00, 5, true).unwrap();
            // A parameter for a macro that is not being called is refused.
            assert!(e.write(cx, 0xE09, 1, true).is_err());
        });
        assert_eq!(stats.macro_calls.get(&3), Some(&(1, 3)));
        assert_eq!(stats.macro_calls.get(&0), Some(&(1, 1)));
        assert!(stats.macro_steps > 0);
    }

    #[test]
    fn calling_a_macro_that_was_never_loaded_is_an_error() {
        with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            let err = e.write(cx, 0xE00, 5, true).unwrap_err();
            assert!(err.contains("no start address"), "{err}");
        });
    }

    #[test]
    fn a_macro_that_misbehaves_names_itself_and_what_went_wrong() {
        with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            // Reads two parameters and is called with one.
            load_macro(cx, &mut e, 2, 0, &reader(2));
            let err = e.write(cx, 0xE04, 1, true).unwrap_err();
            assert!(
                err.contains("macro 2") && err.contains("more parameters"),
                "{err}"
            );
        });
    }

    #[test]
    fn a_macro_that_draws_is_counted_with_its_primitive_and_where_it_came_from() {
        use crate::mme::asm::*;
        let ((), _, stats) = with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            // BEGIN triangles, DRAW_VERTEX_ARRAY with the first parameter as the count, END.
            let code = [
                add_imm(MOVE_SET_ADDRESS, 2, 0, address(0x1618, 0) as i32),
                add_imm(MOVE_EMIT, 3, 0, 4),
                add_imm(MOVE_SET_ADDRESS, 2, 0, address(0xd78, 0) as i32),
                alu(OR, MOVE_EMIT, 3, 1, 0),
                add_imm(MOVE_SET_ADDRESS, 2, 0, address(0x1614, 0) as i32),
                add_imm(MOVE_EMIT, 3, 0, 0) | END_NEXT,
                alu(OR, MOVE, 0, 0, 0),
            ];
            load_macro(cx, &mut e, 5, 0, &code);
            // SET_VERTEX_ARRAY_START 30, then the call: 6 vertices.
            e.write(cx, 0xd74 / 4, 30, false).unwrap();
            e.write(cx, 0xE0A, 6, true).unwrap();
            e.write(cx, 0xE0A, 12, true).unwrap();
        });
        let d = &stats.draws;
        assert_eq!(d.draws(), 2);
        assert_eq!(d.by_kind["DRAW_VERTEX_ARRAY"], 2);
        assert_eq!(d.elements, 18);
        assert!(
            d.first[0].contains("triangles first 30, count 6")
                && d.first[0].contains("from macro 5"),
            "{:?}",
            d.first
        );
        // Everything the macro wrote is counted as the macro's, with the methods by name.
        assert_eq!(stats.macro_writes[&(class::THREED, 0xd78)], 2);
        assert_eq!(stats.macro_writes[&(class::THREED, 0x1618)], 2);
        let report = stats.report(8);
        assert!(report.contains("draws: 2 (18 elements named)"), "{report}");
        assert!(report.contains("macros ran"), "{report}");
    }

    #[test]
    fn a_single_inline_vertex_is_a_draw_when_the_primitive_ends() {
        let ((), _, stats) = with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            e.write(cx, 0x1618 / 4, 1, true).unwrap();
            e.write(cx, 0x1640 / 4, 0x1234, true).unwrap();
            e.write(cx, 0x1614 / 4, 0, true).unwrap();
            // A primitive with nothing inline is not a draw.
            e.write(cx, 0x1618 / 4, 1, true).unwrap();
            e.write(cx, 0x1614 / 4, 0, true).unwrap();
        });
        assert_eq!(stats.draws.draws(), 1);
        assert_eq!(stats.draws.elements, 1);
    }
    #[test]
    fn draws_in_the_method_stream_are_decoded_too() {
        let ((), _, stats) = with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            // Combined begin-draw-end: first 3, count 9, triangle strip (5).
            e.write(cx, 0x1214 / 4, 3 | 9 << 16 | 5 << 28, true)
                .unwrap();
            // A 16-bit indexed draw, first 0, count 600, triangles (4).
            e.write(cx, 0x17e8 / 4, 600 << 16 | 4 << 28, true).unwrap();
            // A plain indexed draw inside BEGIN/END, and a draw with inline indices.
            e.write(cx, 0x1618 / 4, 7, true).unwrap();
            e.write(cx, 0x17e0 / 4, 12, true).unwrap();
            e.write(cx, 0x1614 / 4, 0, true).unwrap();
            e.write(cx, 0x1618 / 4, 4, true).unwrap();
            e.write(cx, 0x15f0 / 4, 1 | 2 << 16, true).unwrap();
            e.write(cx, 0x1614 / 4, 0, true).unwrap();
            e.write(cx, 0x19d0 / 4, 0xF0, true).unwrap();
        });
        let d = &stats.draws;
        assert_eq!(d.draws(), 4);
        assert_eq!(d.elements, 9 + 600 + 12 + 2);
        assert_eq!(d.clears, 1);
        assert!(
            d.first[0].contains("triangle strip first 3, count 9"),
            "{:?}",
            d.first
        );
        assert!(
            d.first[0].contains("from the method stream"),
            "{:?}",
            d.first
        );
        assert_eq!(d.by_kind["DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST"], 1);
        assert_eq!(d.by_kind["inline vertices or indices"], 1);
    }

    #[test]
    fn a_report_semaphore_release_writes_the_payload_and_a_timestamp() {
        let ((), mem, stats) = with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            e.write(cx, 0x1b00 / 4, 0, false).unwrap();
            e.write(cx, 0x1b04 / 4, 0x500_0100, false).unwrap();
            e.write(cx, 0x1b08 / 4, 0xCAFE, false).unwrap();
            // RELEASE, four words, no report.
            e.write(cx, 0x1b0c / 4, 0, true).unwrap();
            // A one-word release to another address.
            e.write(cx, 0x1b04 / 4, 0x500_0200, false).unwrap();
            e.write(cx, 0x1b08 / 4, 0xBEEF, false).unwrap();
            e.write(cx, 0x1b0c / 4, 1 << 28, true).unwrap();
        });
        assert_eq!(mem.read_u32(0x1000_0100), Some(0xCAFE));
        assert_eq!(mem.read_u32(0x1000_0104), Some(0));
        assert!(mem.read_u64(0x1000_0108).is_some());
        assert_eq!(mem.read_u32(0x1000_0200), Some(0xBEEF));
        assert_eq!(mem.read_u32(0x1000_0204), Some(0), "one word only");
        assert_eq!(stats.semaphore_releases, 2);
    }

    #[test]
    fn a_semaphore_acquire_that_cannot_succeed_stops() {
        with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            e.write(cx, 0x1b04 / 4, 0x500_0000, false).unwrap();
            e.write(cx, 0x1b08 / 4, 5, false).unwrap();
            // ACQUIRE with equality on a word that holds 0.
            let err = e.write(cx, 0x1b0c / 4, 1, true).unwrap_err();
            assert!(err.contains("holds 0x0"), "{err}");
            // The same with "greater or equal" and a payload of 0 is satisfied.
            e.write(cx, 0x1b08 / 4, 0, false).unwrap();
            e.write(cx, 0x1b0c / 4, 1 | 1 << 16, true).unwrap();
        });
    }

    #[test]
    fn a_write_to_unmapped_gpu_memory_is_an_error() {
        with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
            e.write(cx, 0x1b04 / 4, 0x600_0000, false).unwrap();
            let err = e.write(cx, 0x1b0c / 4, 0, true).unwrap_err();
            assert!(err.contains("not mapped"), "{err}");
        });
    }

    #[test]
    fn the_report_lists_totals_and_the_busiest_methods() {
        let mut s = Stats {
            submits: 2,
            ..Stats::default()
        };
        s.per_method.insert((class::THREED, 0x100), 9);
        s.per_method.insert((class::THREED, 0x3800), 3);
        s.macro_calls.insert(4, (7, 20));
        let r = s.report(5);
        assert!(r.contains("2 submissions"), "{r}");
        assert!(r.contains("9 x 3D NO_OPERATION"), "{r}");
        assert!(r.contains("3 x 3D CALL_MME_MACRO[0]"), "{r}");
        assert!(r.contains("7 x macro 4 (20 parameters in all)"), "{r}");
    }
}
