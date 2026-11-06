//! A GPU channel: the command queue behind one `/dev/nvhost-gpu` descriptor, and what runs its
//! commands.
//!
//! Commands arrive as GPFIFO entries (see [`crate::pushbuf`]). Methods below `0x100` bytes belong
//! to the channel itself (binding an engine to a subchannel, semaphores, fences); the others go to
//! the engine on the subchannel.

use crate::engine::{Cx, Engine};
use crate::gpu::class;
use crate::names;
use crate::pushbuf::{MethodWrite, Pusher};
use std::collections::BTreeMap;

/// The first method (as a word index) that belongs to an engine.
pub const ENGINE_METHODS: u32 = 0x100 / 4;

/// Host methods, as word indices.
mod host {
    // Word indices; the byte offsets are four times as much.
    pub const SET_OBJECT: u32 = 0; // 0x00
    pub const ILLEGAL: u32 = 1; // 0x04
    pub const NOP: u32 = 2; // 0x08
    pub const SEMAPHORE_A: u32 = 4; // 0x10
    pub const SEMAPHORE_D: u32 = 7; // 0x1c
    pub const NON_STALL_INTERRUPT: u32 = 8; // 0x20
    pub const FB_FLUSH: u32 = 9; // 0x24
    pub const MEM_OP_A: u32 = 0xA; // 0x28 (Kepler; gone from the Maxwell header)
    pub const MEM_OP_B: u32 = 0xB; // 0x2c
    pub const MEM_OP_C: u32 = 0xC; // 0x30
    pub const MEM_OP_D: u32 = 0xD; // 0x34
    pub const SET_REFERENCE: u32 = 0x14; // 0x50
    pub const WFI: u32 = 0x1E; // 0x78
    pub const CRC_CHECK: u32 = 0x1F; // 0x7c
    pub const YIELD: u32 = 0x20; // 0x80
}

#[derive(Debug, Default)]
pub struct Channel {
    /// The address space commands run in, once `BIND_CHANNEL` has been called.
    pub space: Option<u32>,
    /// The syncpoint the channel's fences count on, reserved by `ALLOC_GPFIFO_EX2`.
    pub syncpt: Option<u32>,
    pub gpfifo_entries: u32,
    pub nvmap_fd: Option<u32>,
    pub priority: u32,
    pub timeslice: u32,
    pub timeout: u32,
    pub user_data: u64,
    pub error_notifier: bool,
    /// The classes bound with `ALLOC_OBJ_CTX`, with the object id answered for each.
    pub objects: Vec<(u32, u64)>,
    pub pusher: Pusher,
    /// The class bound to each subchannel with `SET_OBJECT` (0 for none).
    pub subchannels: [u32; 8],
    pub engines: BTreeMap<u32, Engine>,
    semaphore: [u32; 4],
    pub reference: u32,
}

impl Channel {
    /// Run one method write.
    pub fn execute(&mut self, cx: &mut Cx<'_>, w: MethodWrite) -> Result<(), String> {
        cx.stats.methods += 1;
        let sub = usize::from(w.subchannel);
        // A draw held by a 3D engine goes to the renderer before anything another engine or the
        // channel does (A-281).
        if w.method < ENGINE_METHODS || class::kind(self.subchannels[sub]) != class::Kind::ThreeD {
            self.flush_draws(cx)?;
        }
        if w.method < ENGINE_METHODS {
            if cx.level >= 2 {
                eprintln!(
                    "[gpu]   ch{} sc{} HOST {} = {:#x}",
                    cx.channel,
                    sub,
                    names::method_name(class::GPFIFO, w.method * 4),
                    w.data
                );
            }
            // Counted for the report of `NX_TRACE_GPU`, which alone shows it.
            if cx.level >= 1 {
                *cx.stats
                    .per_method
                    .entry((class::GPFIFO, w.method * 4))
                    .or_default() += 1;
            }
            return self.host_method(cx, w);
        }
        let class_num = self.subchannels[sub];
        if class_num == 0 {
            return Err(format!(
                "method {:#x} on subchannel {sub}, to which no engine is bound",
                w.method * 4
            ));
        }
        if cx.level >= 1 {
            *cx.stats
                .per_method
                .entry((class_num, w.method * 4))
                .or_default() += 1;
        }
        if cx.level >= 2 {
            eprintln!(
                "[gpu]   ch{} sc{} {} {} = {:#x}",
                cx.channel,
                sub,
                names::class_name(class_num),
                names::method_name(class_num, w.method * 4),
                w.data
            );
        }
        let engine = self
            .engines
            .entry(class_num)
            .or_insert_with(|| Engine::new(class_num));
        engine.write(cx, w.method, w.data, w.last)
    }

    /// Give the renderer the draws the channel's engines hold (A-281).
    pub fn flush_draws(&mut self, cx: &mut Cx<'_>) -> Result<(), String> {
        for engine in self.engines.values_mut() {
            engine.flush_draw(cx)?;
        }
        Ok(())
    }

    fn host_method(&mut self, cx: &mut Cx<'_>, w: MethodWrite) -> Result<(), String> {
        match w.method {
            // class in bits 15:0, engine in 20:16
            host::SET_OBJECT => {
                let class_num = w.data & 0xFFFF;
                // The graphics engine runs the 3D, compute, 2D and inline-to-memory classes once
                // the channel has an object context; the copy engine needs none. A title binds
                // the others without asking for a context for each.
                let known = [
                    class::THREED,
                    class::COMPUTE,
                    class::TWOD,
                    class::INLINE_TO_MEMORY,
                    class::DMA_COPY,
                    // The Switch 2's (A-208).
                    class::AMPERE_THREED,
                    class::AMPERE_COMPUTE,
                    class::AMPERE_DMA_COPY,
                ];
                if !known.contains(&class_num) {
                    return Err(format!(
                        "SET_OBJECT of class {class_num:#x}, which is not an engine class"
                    ));
                }
                if class::kind(class_num) != class::Kind::Copy && self.objects.is_empty() {
                    return Err(format!(
                        "SET_OBJECT of class {class_num:#x} on a channel with no object context"
                    ));
                }
                self.subchannels[usize::from(w.subchannel)] = class_num;
                if cx.level >= 1 {
                    eprintln!(
                        "[gpu]   ch{} subchannel {} is bound to {} ({class_num:#x})",
                        cx.channel,
                        w.subchannel,
                        names::class_name(class_num)
                    );
                }
                Ok(())
            }
            host::ILLEGAL | host::NOP => Ok(()),
            host::SEMAPHORE_A..=host::SEMAPHORE_D => {
                let i = (w.method - host::SEMAPHORE_A) as usize;
                self.semaphore[i] = w.data;
                if w.method == host::SEMAPHORE_D {
                    self.semaphore_op(cx)?;
                }
                Ok(())
            }
            host::NON_STALL_INTERRUPT
            | host::FB_FLUSH
            | host::MEM_OP_A
            | host::MEM_OP_B
            | host::MEM_OP_C
            | host::MEM_OP_D => Ok(()),
            host::SET_REFERENCE => {
                self.reference = w.data;
                Ok(())
            }
            // The commands before this one have completed, as everything runs in order.
            host::WFI | host::CRC_CHECK | host::YIELD => Ok(()),
            other => Err(format!(
                "host method {:#x} (data {:#x}) is not implemented",
                other * 4,
                w.data
            )),
        }
    }

    /// `SEMAPHORED`: the operation in bits 4:0 (acquire 1, release 2, acquire-geq 4, acquire-and
    /// 8, reduction 0x10), the size of a release in bit 24 (1 is four bytes, 0 sixteen).
    fn semaphore_op(&mut self, cx: &mut Cx<'_>) -> Result<(), String> {
        let [a, b, payload, d] = self.semaphore;
        let va = (u64::from(a & 0xFF) << 32) | u64::from(b & !3);
        // A release is made once the launches before it are done; an acquire compares the word as
        // the releases still to be made leave it (A-274).
        match d & 0x1F {
            2 => {
                if d >> 24 & 1 != 0 {
                    cx.write_after(va, &payload.to_le_bytes())?;
                } else {
                    let mut out = [0u8; 16];
                    out[..4].copy_from_slice(&payload.to_le_bytes());
                    out[8..].copy_from_slice(&cx.timestamp().to_le_bytes());
                    cx.write_after(va, &out)?;
                }
                cx.stats.semaphore_releases += 1;
                Ok(())
            }
            op @ (1 | 4 | 8) => {
                let met = |have: u32| match op {
                    1 => have == payload,
                    4 if d >> 31 != 0 => have >= payload,
                    4 => (have as i32) >= (payload as i32),
                    _ => have & payload != 0,
                };
                cx.stats.front_reads[0] += 1;
                // With a renderer on another thread, memory as the front knows it answers
                // without waiting when it satisfies the acquire, and when the acquire is one that
                // stopped its channel, tried again (A-288).
                let mut have = match cx.word_now(va) {
                    Some(v) if met(v) || cx.retrying => {
                        cx.stats.front_reads[5] += 1;
                        v
                    }
                    _ => cx.acquire_value(va)?,
                };
                if !met(have) && !cx.retrying {
                    // Before the channel stops, what is still to be released is made: it may be
                    // what it waits for (A-274). Made again later, the acquire has that already.
                    cx.stats.front_reads[4] += 1;
                    cx.settle()?;
                    have = cx.read_u32(va)?;
                }
                let ok = met(have);
                // Not yet: the channel stops here until another channel's commands (or the
                // program) put the value there, and this write is made again then (A-238).
                if !ok {
                    cx.blocked = Some(format!(
                        "a semaphore acquire at GPU address {va:#x} for {payload:#x} (operation \
                         {d:#x}), which holds {have:#x}"
                    ));
                }
                Ok(())
            }
            op => Err(format!("semaphore operation {op:#x} is not implemented")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Stats;
    use crate::mem::testing::TestMemory;
    use crate::mem::GuestMemory;
    use crate::syncpt::Syncpoints;
    use crate::vm::{map_flags, AddressSpace, MapRequest};

    fn write(subchannel: u8, method_bytes: u32, data: u32, last: bool) -> MethodWrite {
        MethodWrite {
            subchannel,
            method: method_bytes / 4,
            data,
            last,
        }
    }

    fn env() -> (TestMemory, AddressSpace) {
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
        (mem, space)
    }

    fn run(ch: &mut Channel, ws: &[MethodWrite]) -> (Vec<Result<(), String>>, TestMemory, Stats) {
        let (mem, space) = env();
        let mut sp = Syncpoints::new();
        let mut woken = Vec::new();
        let mut stats = Stats::default();
        let mut macros = crate::mme64::Store::default();
        let mut qmds = crate::qmd3::Scheduler::default();
        let results = {
            let mut cx = Cx {
                mem: &mem,
                space: &space,
                syncpoints: &mut sp,
                woken: &mut woken,
                stats: &mut stats,
                backend: None,
                lenient: false,
                channel: 1,
                syncpt: 1,
                level: 1,
                macros: &mut macros,
                qmds: &mut qmds,
                blocked: None,
                retrying: false,
            };
            // A write that stops the channel at an acquire is reported with what it waits for.
            ws.iter()
                .map(|w| {
                    ch.execute(&mut cx, *w)
                        .and_then(|()| cx.blocked.take().map_or(Ok(()), Err))
                })
                .collect()
        };
        (results, mem, stats)
    }

    fn channel_with(classes: &[u32]) -> Channel {
        let mut ch = Channel::default();
        for (i, c) in classes.iter().enumerate() {
            ch.objects.push((*c, i as u64 + 1));
        }
        ch
    }

    #[test]
    fn a_semaphore_acquire_reads_a_syncpoint_through_the_switch_2s_shim() {
        let (mem, mut space) = env();
        let shim = 0xC0_FFFF_F000;
        space.set_syncpoint_shim(shim);
        let mut sp = Syncpoints::new();
        sp.increment(5, 3);
        let (mut woken, mut stats) = (Vec::new(), Stats::default());
        let mut macros = crate::mme64::Store::default();
        let mut qmds = crate::qmd3::Scheduler::default();
        let mut cx = Cx {
            mem: &mem,
            space: &space,
            syncpoints: &mut sp,
            woken: &mut woken,
            stats: &mut stats,
            backend: None,
            lenient: false,
            channel: 1,
            syncpt: 1,
            level: 1,
            macros: &mut macros,
            qmds: &mut qmds,
            blocked: None,
            retrying: false,
        };
        let mut ch = Channel::default();
        // SEMAPHOREA..D at syncpoint 5's word of the shim, payload `want`, operation `op`.
        let mut semaphore = |cx: &mut Cx<'_>, want: u32, op: u32| {
            let va = shim + 5 * 4;
            ch.execute(cx, write(0, 0x10, (va >> 32) as u32, false))?;
            ch.execute(cx, write(0, 0x14, va as u32, false))?;
            ch.execute(cx, write(0, 0x18, want, false))?;
            ch.execute(cx, write(0, 0x1C, op, true))
        };
        assert_eq!(semaphore(&mut cx, 3, 1), Ok(()), "acquire: the value is 3");
        assert_eq!(semaphore(&mut cx, 2, 4 | 1 << 31), Ok(()), "acquire-geq 2");
        assert_eq!(cx.blocked, None);
        assert_eq!(semaphore(&mut cx, 4, 1), Ok(()));
        let blocked = cx
            .blocked
            .take()
            .expect("4 is not reached: the channel waits");
        assert!(blocked.contains("holds 0x3"), "{blocked}");
        let release = semaphore(&mut cx, 9, 2 | 1 << 24).unwrap_err();
        assert!(release.contains("read-only"), "{release}");
    }

    #[test]
    fn set_object_binds_engine_classes_once_the_channel_has_a_context() {
        // One context (for the 3D class) is enough for the other graphics classes.
        let mut ch = channel_with(&[class::THREED]);
        let (r, _, _) = run(
            &mut ch,
            &[
                write(0, 0, class::THREED, false),
                write(1, 0, class::COMPUTE, false),
                write(2, 0, class::INLINE_TO_MEMORY, false),
                write(3, 0, class::TWOD, false),
                write(4, 0, class::DMA_COPY, false),
                write(5, 0, 0xC397, false),
            ],
        );
        assert!(r[..5].iter().all(|x| x.is_ok()), "{r:?}");
        assert!(r[5].as_ref().unwrap_err().contains("not an engine class"));
        assert_eq!(
            ch.subchannels[..5],
            [0xB197, 0xB1C0, 0xA140, 0x902D, 0xB0B5]
        );
        assert_eq!(ch.subchannels[5], 0);
        // Without any context only the copy engine can be bound.
        let mut bare = channel_with(&[]);
        let (r, _, _) = run(
            &mut bare,
            &[
                write(0, 0, class::THREED, false),
                write(1, 0, class::DMA_COPY, false),
            ],
        );
        assert!(r[0].as_ref().unwrap_err().contains("no object context"));
        assert_eq!(r[1], Ok(()));
    }

    #[test]
    fn engine_methods_need_a_bound_subchannel_and_go_to_its_engine() {
        let mut ch = channel_with(&[class::THREED]);
        let (r, _, stats) = run(
            &mut ch,
            &[
                write(1, 0x100, 0, false),
                write(0, 0, class::THREED, false),
                write(0, 0x1234, 55, false),
            ],
        );
        assert!(r[0].as_ref().unwrap_err().contains("no engine is bound"));
        assert_eq!(r[2], Ok(()));
        assert_eq!(ch.engines[&class::THREED].reg(0x1234 / 4), 55);
        assert_eq!(stats.per_method[&(class::THREED, 0x1234)], 1);
        assert_eq!(stats.methods, 3);
    }

    #[test]
    fn a_host_semaphore_release_writes_to_memory() {
        let mut ch = channel_with(&[]);
        let (r, mem, stats) = run(
            &mut ch,
            &[
                write(0, 0x10, 0, false),
                write(0, 0x14, 0x500_0040, false),
                write(0, 0x18, 0x1234, false),
                // RELEASE, 16 bytes
                write(0, 0x1c, 2, false),
                write(0, 0x14, 0x500_0080, false),
                write(0, 0x1c, 2 | 1 << 24, false),
            ],
        );
        assert!(r.iter().all(|x| x.is_ok()), "{r:?}");
        assert_eq!(mem.read_u32(0x1000_0040), Some(0x1234));
        assert_eq!(mem.read_u32(0x1000_0080), Some(0x1234));
        assert_eq!(
            mem.read_u32(0x1000_0088),
            Some(0),
            "four bytes: no timestamp"
        );
        assert_eq!(stats.semaphore_releases, 2);
    }

    #[test]
    fn a_host_semaphore_acquire_is_checked_against_memory() {
        let mut ch = channel_with(&[]);
        let (r, _, _) = run(
            &mut ch,
            &[
                write(0, 0x14, 0x500_0000, false),
                write(0, 0x18, 7, false),
                write(0, 0x1c, 1, false),
            ],
        );
        assert!(r[2].as_ref().unwrap_err().contains("holds 0x0"));
        // Greater-or-equal to zero holds for a word of zeros; so does a signed comparison.
        let (r, _, _) = run(
            &mut ch,
            &[write(0, 0x18, 0, false), write(0, 0x1c, 4 | 1 << 31, false)],
        );
        assert!(r.iter().all(|x| x.is_ok()), "{r:?}");
        let (r, _, _) = run(&mut ch, &[write(0, 0x1c, 4, false)]);
        assert!(r[0].is_ok());
    }

    /// A backend that holds semaphore releases until it is asked to settle everything, as the
    /// renderer does while it has compute launches not waited for (A-274).
    #[derive(Default)]
    struct Holding {
        held: Vec<(u64, Vec<u8>)>,
        settles: usize,
    }

    impl crate::render::Backend for Holding {
        fn draw(
            &mut self,
            _: &crate::render::Scene<'_>,
            _: &crate::render::Draw,
        ) -> Result<(), String> {
            Ok(())
        }
        fn clear(
            &mut self,
            _: &crate::render::Scene<'_>,
            _: &crate::render::Clear,
        ) -> Result<(), String> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn present(&mut self, _: &crate::render::Present) -> Result<(), String> {
            Ok(())
        }
        fn write_after(
            &mut self,
            _: &AddressSpace,
            _: &dyn GuestMemory,
            va: u64,
            data: &[u8],
        ) -> Result<(), String> {
            self.held.push((va, data.to_vec()));
            Ok(())
        }
        fn settle(
            &mut self,
            space: &AddressSpace,
            mem: &dyn GuestMemory,
            range: Option<(u64, u64)>,
        ) -> Result<(), String> {
            if range.is_none() {
                self.settles += 1;
                for (va, data) in self.held.drain(..) {
                    assert!(space.write(mem, va, &data));
                }
            }
            Ok(())
        }
    }

    #[test]
    fn an_acquire_of_a_word_still_to_be_released_waits_for_the_release() {
        let mut ch = channel_with(&[]);
        let (mem, space) = env();
        let mut sp = Syncpoints::new();
        let (mut woken, mut stats) = (Vec::new(), Stats::default());
        let mut macros = crate::mme64::Store::default();
        let mut qmds = crate::qmd3::Scheduler::default();
        let mut holding = Holding::default();
        let word = |mem: &TestMemory| {
            let mut b = [0u8; 4];
            assert!(space.read(mem, 0x500_0000, &mut b));
            u32::from_le_bytes(b)
        };
        {
            let mut cx = Cx {
                mem: &mem,
                space: &space,
                syncpoints: &mut sp,
                woken: &mut woken,
                stats: &mut stats,
                backend: Some(&mut holding),
                lenient: false,
                channel: 1,
                syncpt: 1,
                level: 0,
                macros: &mut macros,
                qmds: &mut qmds,
                blocked: None,
                retrying: false,
            };
            // Release 7 (one word): the backend holds it.
            for w in [
                write(0, 0x14, 0x500_0000, false),
                write(0, 0x18, 7, false),
                write(0, 0x1c, 2 | 1 << 24, false),
            ] {
                ch.execute(&mut cx, w).unwrap();
            }
            assert_eq!(word(&mem), 0, "the release is held");
            // An acquire of 7 finds it: the release is made before the channel would stop.
            ch.execute(&mut cx, write(0, 0x1c, 1, false)).unwrap();
            assert_eq!(cx.blocked, None);
        }
        assert_eq!(word(&mem), 7);
        assert_eq!(holding.settles, 1);
    }

    /// [`Holding`], which also says what the word will be once the held releases are made.
    #[derive(Default)]
    struct Foreseeing(Holding);

    impl crate::render::Backend for Foreseeing {
        fn draw(
            &mut self,
            s: &crate::render::Scene<'_>,
            d: &crate::render::Draw,
        ) -> Result<(), String> {
            self.0.draw(s, d)
        }
        fn clear(
            &mut self,
            s: &crate::render::Scene<'_>,
            c: &crate::render::Clear,
        ) -> Result<(), String> {
            self.0.clear(s, c)
        }
        fn finish(&mut self) -> Result<(), String> {
            self.0.finish()
        }
        fn present(&mut self, p: &crate::render::Present) -> Result<(), String> {
            self.0.present(p)
        }
        fn write_after(
            &mut self,
            space: &AddressSpace,
            mem: &dyn GuestMemory,
            va: u64,
            data: &[u8],
        ) -> Result<(), String> {
            self.0.write_after(space, mem, va, data)
        }
        fn settle(
            &mut self,
            space: &AddressSpace,
            mem: &dyn GuestMemory,
            range: Option<(u64, u64)>,
        ) -> Result<(), String> {
            self.0.settle(space, mem, range)
        }
        fn released_value(
            &mut self,
            _: &AddressSpace,
            _: &dyn GuestMemory,
            va: u64,
        ) -> Option<u32> {
            self.0
                .held
                .iter()
                .rev()
                .find(|(at, data)| *at == va && data.len() >= 4)
                .map(|(_, data)| u32::from_le_bytes(data[..4].try_into().unwrap()))
        }
    }

    #[test]
    fn an_acquire_compares_the_word_as_the_held_releases_will_leave_it() {
        let mut ch = channel_with(&[]);
        let (mem, space) = env();
        let mut sp = Syncpoints::new();
        let (mut woken, mut stats) = (Vec::new(), Stats::default());
        let mut macros = crate::mme64::Store::default();
        let mut qmds = crate::qmd3::Scheduler::default();
        let mut backend = Foreseeing::default();
        {
            let mut cx = Cx {
                mem: &mem,
                space: &space,
                syncpoints: &mut sp,
                woken: &mut woken,
                stats: &mut stats,
                backend: Some(&mut backend),
                lenient: false,
                channel: 1,
                syncpt: 1,
                level: 0,
                macros: &mut macros,
                qmds: &mut qmds,
                blocked: None,
                retrying: false,
            };
            for w in [
                write(0, 0x14, 0x500_0000, false),
                write(0, 0x18, 7, false),
                write(0, 0x1c, 2 | 1 << 24, false),
                write(0, 0x1c, 1, false),
            ] {
                ch.execute(&mut cx, w).unwrap();
            }
            assert_eq!(cx.blocked, None);
        }
        // Nothing waited: the release is still held.
        assert_eq!(backend.0.settles, 0);
        assert_eq!(backend.0.held.len(), 1);
    }

    #[test]
    fn harmless_host_methods_succeed_and_unknown_ones_stop() {
        let mut ch = channel_with(&[]);
        let (r, _, _) = run(
            &mut ch,
            &[
                write(0, 0x08, 0, false),
                write(0, 0x24, 0, false),
                write(0, 0x50, 0xABCD, false),
                write(0, 0x78, 0, false),
                write(0, 0x70, 5, false),
            ],
        );
        assert!(r[..4].iter().all(|x| x.is_ok()));
        assert_eq!(ch.reference, 0xABCD);
        assert!(r[4].as_ref().unwrap_err().contains("host method 0x70"));
    }
}
