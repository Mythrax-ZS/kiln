//! The engines a channel drives (3D, compute, 2D, copy, inline-to-memory) and the context they
//! run in.
//!
//! Each engine is a register file addressed by method. What a write does is implemented where the
//! effect is visible to the program: the memory a semaphore release writes, the code a macro upload
//! stores. Every other write is recorded and counted, which is what the tracing processor is for.

use crate::event::EventToken;
use crate::gpu::class;
use crate::mem::GuestMemory;
use crate::names;
use crate::syncpt::Syncpoints;
use crate::vm::AddressSpace;
use std::collections::BTreeMap;

/// How many registers (words) an engine has: methods are 12 bits wide.
pub const REGISTERS: usize = 0x1000;

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
}

#[derive(Debug)]
pub struct Engine {
    pub class: u32,
    regs: Vec<u32>,
    pub mme: Mme,
    /// A macro call being collected: its number and the parameters so far.
    pending: Option<(u32, Vec<u32>)>,
}

impl Engine {
    pub fn new(class_num: u32) -> Engine {
        Engine {
            class: class_num,
            regs: vec![0; REGISTERS],
            mme: Mme::default(),
            pending: None,
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
                        self.finish_macro(cx);
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
                    self.finish_macro(cx);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// A macro call is complete: it is counted (macros are not run yet).
    fn finish_macro(&mut self, cx: &mut Cx<'_>) {
        let Some((id, params)) = self.pending.take() else {
            return;
        };
        let e = cx.stats.macro_calls.entry(id).or_default();
        e.0 += 1;
        e.1 += params.len() as u64;
        if cx.level >= 2 {
            let start = self.mme.starts.get(id as usize).copied().unwrap_or(0);
            eprintln!(
                "[gpu]   macro {id} (code at {start:#x}) called with {} parameters: {:x?}",
                params.len(),
                &params[..params.len().min(8)]
            );
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

    #[test]
    fn a_macro_call_collects_its_parameters_until_the_command_ends() {
        let ((), _, stats) = with_cx(|cx| {
            let mut e = Engine::new(class::THREED);
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
