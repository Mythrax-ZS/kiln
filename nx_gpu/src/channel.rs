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
            *cx.stats
                .per_method
                .entry((class::GPFIFO, w.method * 4))
                .or_default() += 1;
            return self.host_method(cx, w);
        }
        let class_num = self.subchannels[sub];
        if class_num == 0 {
            return Err(format!(
                "method {:#x} on subchannel {sub}, to which no engine is bound",
                w.method * 4
            ));
        }
        *cx.stats
            .per_method
            .entry((class_num, w.method * 4))
            .or_default() += 1;
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
                ];
                if !known.contains(&class_num) {
                    return Err(format!(
                        "SET_OBJECT of class {class_num:#x}, which is not an engine class"
                    ));
                }
                if class_num != class::DMA_COPY && self.objects.is_empty() {
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
        match d & 0x1F {
            2 => {
                if d >> 24 & 1 != 0 {
                    cx.write(va, &payload.to_le_bytes())?;
                } else {
                    let mut out = [0u8; 16];
                    out[..4].copy_from_slice(&payload.to_le_bytes());
                    out[8..].copy_from_slice(&cx.timestamp().to_le_bytes());
                    cx.write(va, &out)?;
                }
                cx.stats.semaphore_releases += 1;
                Ok(())
            }
            op @ (1 | 4 | 8) => {
                let have = cx.read_u32(va)?;
                let ok = match op {
                    1 => have == payload,
                    4 if d >> 31 != 0 => have >= payload,
                    4 => (have as i32) >= (payload as i32),
                    _ => have & payload != 0,
                };
                if ok {
                    Ok(())
                } else {
                    Err(format!(
                        "a semaphore acquire at GPU address {va:#x} for {payload:#x}, which holds \
                         {have:#x} and nothing is left to change it"
                    ))
                }
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
        let results = {
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
            ws.iter().map(|w| ch.execute(&mut cx, *w)).collect()
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
