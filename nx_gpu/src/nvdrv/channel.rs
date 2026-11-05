//! `/dev/nvhost-gpu` ioctls (Switchbrew, "NV services", section 9): the command queue of the GPU.

use super::args::{unimplemented, Args, Fail, IoResult, Ioctl};
use super::{Node, Nvdrv};
use crate::error::NvError;
use crate::gpu::{class, SubmitError};

/// The type byte of the GPU channel ioctls (`'H'`). `SET_USER_DATA` and `GET_USER_DATA` use `'G'`.
pub const TYPE: u32 = 0x48;
const USER_DATA_TYPE: u32 = 0x47;

pub const SET_NVMAP_FD: u32 = 1;
pub const SET_TIMEOUT: u32 = 3;
pub const SUBMIT_GPFIFO: u32 = 8;
pub const ALLOC_OBJ_CTX: u32 = 9;
pub const ZCULL_BIND: u32 = 0x0B;
pub const SET_ERROR_NOTIFIER: u32 = 0x0C;
pub const SET_PRIORITY: u32 = 0x0D;
pub const ENABLE: u32 = 0x0E;
pub const DISABLE: u32 = 0x0F;
pub const PREEMPT: u32 = 0x10;
pub const FORCE_RESET: u32 = 0x11;
pub const EVENT_ID_CONTROL: u32 = 0x12;
pub const GET_ERROR_INFO: u32 = 0x16;
pub const GET_ERROR_NOTIFICATION: u32 = 0x17;
pub const ALLOC_GPFIFO_EX: u32 = 0x18;
pub const ALLOC_GPFIFO_EX2: u32 = 0x1A;
pub const SUBMIT_GPFIFO2: u32 = 0x1B;
pub const SET_TIMESLICE: u32 = 0x1D;
/// The Switch 2's: the runlist the channel runs on (A-203).
pub const SET_RUNLIST: u32 = 0x20;
/// The Switch 2's: how the channel's graphics and compute work may be preempted (A-206).
pub const SET_PREEMPTION_MODE: u32 = 0x1F;

const SET_USER_DATA: u32 = 0x14;
const GET_USER_DATA: u32 = 0x15;

pub fn name(code: Ioctl) -> Option<&'static str> {
    if code.ty() == USER_DATA_TYPE {
        return match code.nr() {
            SET_USER_DATA => Some("NVGPU_IOCTL_CHANNEL_SET_USER_DATA"),
            GET_USER_DATA => Some("NVGPU_IOCTL_CHANNEL_GET_USER_DATA"),
            _ => None,
        };
    }
    if code.ty() != TYPE {
        return None;
    }
    Some(match code.nr() {
        SET_NVMAP_FD => "NVGPU_IOCTL_CHANNEL_SET_NVMAP_FD",
        SET_TIMEOUT => "NVGPU_IOCTL_CHANNEL_SET_TIMEOUT",
        SUBMIT_GPFIFO => "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO",
        ALLOC_OBJ_CTX => "NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX",
        ZCULL_BIND => "NVGPU_IOCTL_CHANNEL_ZCULL_BIND",
        SET_ERROR_NOTIFIER => "NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER",
        SET_PRIORITY => "NVGPU_IOCTL_CHANNEL_SET_PRIORITY",
        ENABLE => "NVGPU_IOCTL_CHANNEL_ENABLE",
        DISABLE => "NVGPU_IOCTL_CHANNEL_DISABLE",
        PREEMPT => "NVGPU_IOCTL_CHANNEL_PREEMPT",
        FORCE_RESET => "NVGPU_IOCTL_CHANNEL_FORCE_RESET",
        EVENT_ID_CONTROL => "NVGPU_IOCTL_CHANNEL_EVENT_ID_CONTROL",
        GET_ERROR_INFO => "NVGPU_IOCTL_CHANNEL_GET_ERROR_INFO",
        GET_ERROR_NOTIFICATION => "NVGPU_IOCTL_CHANNEL_GET_ERROR_NOTIFICATION",
        ALLOC_GPFIFO_EX => "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX",
        ALLOC_GPFIFO_EX2 => "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX2",
        SUBMIT_GPFIFO2 => "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2",
        SET_TIMESLICE => "NVGPU_IOCTL_CHANNEL_SET_TIMESLICE",
        SET_RUNLIST => "the channel's runlist (Switch 2)",
        SET_PREEMPTION_MODE => "NVGPU_IOCTL_CHANNEL_SET_PREEMPTION_MODE",
        _ => return None,
    })
}

impl Nvdrv {
    pub(super) fn channel_ioctl(
        &mut self,
        id: u32,
        code: Ioctl,
        args: &mut Args,
        extra_in: &[u8],
    ) -> IoResult {
        let (ty, nr, size) = (code.ty(), code.nr(), code.size());
        match (ty, nr, size) {
            // u64 addr: opaque data the program keeps with the channel
            (USER_DATA_TYPE, SET_USER_DATA, 8) => {
                self.channel(id)?.user_data = args.u64(0);
            }
            (USER_DATA_TYPE, GET_USER_DATA, 8) => {
                let v = self.channel(id)?.user_data;
                args.set_u64(0, v);
            }
            // u32 nvmap fd
            (TYPE, SET_NVMAP_FD, 4) => {
                let fd = args.u32(0);
                if !matches!(self.nodes.get(&fd), Some(Node::Nvmap)) {
                    return Err(NvError::BAD_VALUE.into());
                }
                self.channel(id)?.nvmap_fd = Some(fd);
            }
            (TYPE, SET_TIMEOUT, 4) => {
                let t = args.u32(0);
                self.channel(id)?.timeout = t;
            }
            // The Switch 2's (u32 runlist): one of the runlists GET_ENGINE_INFO lists, from
            // NvRmGpuChannelCreate. The channels here run in the order of their submissions
            // whatever their runlist, so it is checked and not kept (A-203).
            (TYPE, SET_RUNLIST, 4) if self.is_switch_2() => {
                let runlist = args.u32(0);
                if !super::ctrl_gpu::SWITCH_2_ENGINES
                    .iter()
                    .any(|e| e[2] == runlist)
                {
                    return unimplemented(format!(
                        "a channel on the runlist {runlist}, which GET_ENGINE_INFO does not list"
                    ));
                }
                self.channel(id)?;
            }
            // The Switch 2's SET_PREEMPTION_MODE(u32 graphics, u32 compute): 0 leaves a mode as it
            // is, graphics 1 WFI and 2 GFXP, compute 1 WFI, 2 CTA and 4 CILP (NvRm sends compute
            // CTA). Nothing preempts here, so the modes are checked and not kept (A-206).
            (TYPE, SET_PREEMPTION_MODE, 8) if self.is_switch_2() => {
                let (graphics, compute) = (args.u32(0), args.u32(4));
                if graphics > 2 || ![0, 1, 2, 4].contains(&compute) {
                    return unimplemented(format!(
                        "preemption modes {graphics:#x} (graphics) and {compute:#x} (compute)"
                    ));
                }
                self.channel(id)?;
            }
            (TYPE, SET_PRIORITY, 4) => {
                let p = args.u32(0);
                self.channel(id)?.priority = p;
            }
            (TYPE, SET_TIMESLICE, 4) => {
                let t = args.u32(0);
                self.channel(id)?.timeslice = t;
            }
            // u32 num_entries, u32 num_jobs, u32 flags, fence out {id, value}, u32 reserved[3]
            (TYPE, ALLOC_GPFIFO_EX | ALLOC_GPFIFO_EX2, 32) => {
                let entries = args.u32(0);
                if entries == 0 {
                    return Err(NvError::BAD_VALUE.into());
                }
                if self.channel(id)?.syncpt.is_some() {
                    return Err(NvError::ALREADY_ALLOCATED.into());
                }
                let syncpt = self
                    .gpu
                    .syncpoints
                    .allocate()
                    .ok_or(NvError::INSUFFICIENT_MEMORY)?;
                let max = self.gpu.syncpoints.max(syncpt).unwrap_or(0);
                let ch = self.channel(id)?;
                ch.syncpt = Some(syncpt);
                ch.gpfifo_entries = entries;
                if nr == ALLOC_GPFIFO_EX2 {
                    args.set_u32(12, syncpt);
                    args.set_u32(16, max);
                }
            }
            // u32 class, u32 flags, u64 object id (out)
            (TYPE, ALLOC_OBJ_CTX, 16) => {
                let class_num = args.u32(0);
                let switch_2 = self.is_switch_2();
                // The Switch 2's driver has the Ampere classes instead (A-199, A-204).
                let known: &[u32] = if switch_2 {
                    &[
                        class::AMPERE_THREED,
                        class::AMPERE_COMPUTE,
                        class::AMPERE_DMA_COPY,
                        class::TWOD,
                        class::INLINE_TO_MEMORY,
                    ]
                } else {
                    &[
                        class::THREED,
                        class::COMPUTE,
                        class::TWOD,
                        class::INLINE_TO_MEMORY,
                        class::DMA_COPY,
                    ]
                };
                if !known.contains(&class_num) {
                    return unimplemented(format!("an object context of class {class_num:#x}"));
                }
                let ch = self.channel(id)?;
                // On the Switch 2 a channel of a timeslice group asks for its object contexts
                // before it has an address space and a queue (A-204).
                if !switch_2 && (ch.space.is_none() || ch.syncpt.is_none()) {
                    // The address space and the queue must exist first (Switchbrew).
                    return Err(NvError::INVALID_STATE.into());
                }
                let obj = ch.objects.len() as u64 + 1;
                ch.objects.push((class_num, obj));
                args.set_u64(8, obj);
            }
            // u64 gpu_va, u32 mode, u32 padding
            (TYPE, ZCULL_BIND, 16) => {}
            // u64 offset, u64 size, u32 mem, u32 reserved
            (TYPE, SET_ERROR_NOTIFIER, 24) => {
                let on = args.u32(16) != 0;
                self.channel(id)?.error_notifier = on;
            }
            (TYPE, ENABLE | DISABLE | PREEMPT | FORCE_RESET, 0) => {}
            // u32 cmd, u32 id: the events are never raised, so enabling them changes nothing
            (TYPE, EVENT_ID_CONTROL, 8) => {}
            // The channel never faults: no error to report.
            (TYPE, GET_ERROR_INFO, 128) | (TYPE, GET_ERROR_NOTIFICATION, 16) => {}
            // u64 gpfifo (ignored), u32 num_entries, u32 flags, fence {id, value} (in and out),
            // then the entries; SUBMIT_GPFIFO2 sends the entries in the second input buffer.
            (TYPE, SUBMIT_GPFIFO | SUBMIT_GPFIFO2, 24..) => {
                let n = args.u32(8) as usize;
                let (flags, fence) = (args.u32(12), (args.u32(16), args.u32(20)));
                let raw = if nr == SUBMIT_GPFIFO {
                    args.data.get(24..)
                } else {
                    Some(extra_in)
                };
                let Some(raw) = raw.filter(|r| r.len() >= n * 8) else {
                    return Err(NvError::INVALID_SIZE.into());
                };
                let entries: Vec<u64> = raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .take(n)
                    .map(|c| u64::from_le_bytes(*c))
                    .collect();
                let result = if self.gpu.defers(flags) {
                    self.gpu.defer(id, &entries, flags, fence)
                } else if self.gpu.queue.is_some() {
                    return unimplemented(
                        "a submission whose fence is made by running it (FENCE_GET) while the \
                         renderer runs submissions on a thread of its own (NX_GPU_SYNC=1 runs \
                         them in the call)",
                    );
                } else {
                    self.gpu.submit(id, &entries, flags, fence)
                };
                match result {
                    Ok(done) => {
                        args.set_u32(16, done.fence.0);
                        args.set_u32(20, done.fence.1);
                        self.signal(done.woken);
                    }
                    Err(SubmitError::Nv(e)) => return Err(e.into()),
                    Err(SubmitError::Unimplemented(why)) => return unimplemented(why),
                }
            }
            _ => {
                return unimplemented(format!(
                    "GPU channel command {nr:#x} (type {ty:#x}, size {size})"
                ))
            }
        }
        Ok(())
    }

    fn channel(&mut self, id: u32) -> Result<&mut crate::gpu::Channel, Fail> {
        self.gpu
            .channels
            .get_mut(&id)
            .ok_or(Fail::Nv(NvError::INVALID_STATE))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::mem::GuestMemory;

    const SET_NVMAP_FD_CODE: u32 = 0x4004_4801;
    const ALLOC_GPFIFO_EX2_CODE: u32 = 0xC020_481A;
    const ALLOC_OBJ_CTX_CODE: u32 = 0xC010_4809;
    const BIND_CHANNEL_CODE: u32 = 0x4004_4101;
    const SET_USER_DATA_CODE: u32 = 0x4008_4714;
    const GET_USER_DATA_CODE: u32 = 0x8008_4715;

    fn channel() -> (Nvdrv, u32, u32, u32) {
        let (mut nv, _) = nvdrv();
        let nvmap = nv.open("/dev/nvmap").unwrap();
        let as_fd = nv.open("/dev/nvhost-as-gpu").unwrap();
        let ch = nv.open("/dev/nvhost-gpu").unwrap();
        (nv, nvmap, as_fd, ch)
    }

    #[test]
    fn a_channel_is_set_up_like_the_documented_sequence() {
        let (mut nv, nvmap, as_fd, ch) = channel();
        let (e, _) = call(&mut nv, ch, SET_NVMAP_FD_CODE, &nvmap.to_le_bytes());
        assert_eq!(e, NvError::SUCCESS);
        let (e, _) = call(&mut nv, ch, SET_NVMAP_FD_CODE, &as_fd.to_le_bytes());
        assert_eq!(e, NvError::BAD_VALUE, "not an nvmap descriptor");

        // An object context needs the address space and the queue first.
        let (e, _) = call(&mut nv, ch, ALLOC_OBJ_CTX_CODE, &u32s(&[0xB197, 0]));
        assert_eq!(e, NvError::INVALID_STATE);

        call(&mut nv, as_fd, BIND_CHANNEL_CODE, &ch.to_le_bytes());
        let mut ex2 = vec![0u8; 32];
        ex2[..4].copy_from_slice(&0x800u32.to_le_bytes());
        ex2[4..8].copy_from_slice(&1u32.to_le_bytes());
        let (e, out) = call(&mut nv, ch, ALLOC_GPFIFO_EX2_CODE, &ex2);
        assert_eq!(e, NvError::SUCCESS);
        let syncpt = out.u32(12);
        assert_ne!(syncpt, 0, "zero is never a channel's syncpoint");
        assert_eq!(out.u32(16), 0, "nothing has run yet");
        let (e, _) = call(&mut nv, ch, ALLOC_GPFIFO_EX2_CODE, &ex2);
        assert_eq!(e, NvError::ALREADY_ALLOCATED);

        for (i, class) in [0xB197u32, 0xB1C0, 0xA140, 0x902D, 0xB0B5]
            .iter()
            .enumerate()
        {
            let (e, out) = call(&mut nv, ch, ALLOC_OBJ_CTX_CODE, &u32s(&[*class, 0, 0, 0]));
            assert_eq!(e, NvError::SUCCESS);
            assert_eq!(out.u64(8), i as u64 + 1);
        }
        let err = nv
            .ioctl(ch, ALLOC_OBJ_CTX_CODE, &u32s(&[0xC397, 0, 0, 0]), &[])
            .unwrap_err();
        assert!(err.contains("0xc397"), "{err}");
    }

    #[test]
    fn user_data_round_trips() {
        let (mut nv, _, _, ch) = channel();
        let (e, _) = call(
            &mut nv,
            ch,
            SET_USER_DATA_CODE,
            &0xABCD_EF01_2345u64.to_le_bytes(),
        );
        assert_eq!(e, NvError::SUCCESS);
        let (_, out) = call(&mut nv, ch, GET_USER_DATA_CODE, &[]);
        assert_eq!(out.u64(0), 0xABCD_EF01_2345);
    }

    #[test]
    fn closing_a_channel_gives_its_syncpoint_back() {
        let (mut nv, _, _, ch) = channel();
        let mut ex2 = vec![0u8; 32];
        ex2[..4].copy_from_slice(&64u32.to_le_bytes());
        let (_, out) = call(&mut nv, ch, ALLOC_GPFIFO_EX2_CODE, &ex2);
        let syncpt = out.u32(12);
        nv.close(ch);
        assert_eq!(nv.gpu.syncpoints.allocate(), Some(syncpt));
    }

    /// A channel ready to take work: an nvmap handle of 0x1000 bytes at CPU address 0x2000_0000
    /// mapped at GPU address 0x500_0000, an address space bound, a queue, and a 3D object.
    /// Returns the driver, the memory, the channel descriptor and the channel's syncpoint.
    fn ready() -> (
        Nvdrv,
        std::sync::Arc<crate::mem::testing::TestMemory>,
        u32,
        u32,
    ) {
        let (nv, mem, _, ch, syncpt) = ready_with_space();
        (nv, mem, ch, syncpt)
    }

    /// Another channel in the address space `as_fd`, with a queue and a 3D object: its
    /// descriptor and syncpoint.
    fn another_channel(nv: &mut Nvdrv, as_fd: u32) -> (u32, u32) {
        let ch = nv.open("/dev/nvhost-gpu").unwrap();
        call(nv, as_fd, BIND_CHANNEL_CODE, &ch.to_le_bytes());
        let mut ex2 = vec![0u8; 32];
        ex2[..4].copy_from_slice(&0x800u32.to_le_bytes());
        let (_, out) = call(nv, ch, ALLOC_GPFIFO_EX2_CODE, &ex2);
        let (e, _) = call(nv, ch, ALLOC_OBJ_CTX_CODE, &u32s(&[0xB197, 0, 0, 0]));
        assert_eq!(e, NvError::SUCCESS);
        (ch, out.u32(12))
    }

    /// [`ready`], with the address space's descriptor too.
    fn ready_with_space() -> (
        Nvdrv,
        std::sync::Arc<crate::mem::testing::TestMemory>,
        u32,
        u32,
        u32,
    ) {
        let (mut nv, mem) = nvdrv();
        mem.map(0x2000_0000, 0x1000);
        let nvmap = nv.open("/dev/nvmap").unwrap();
        let as_fd = nv.open("/dev/nvhost-as-gpu").unwrap();
        let ch = nv.open("/dev/nvhost-gpu").unwrap();
        let (_, out) = call(&mut nv, nvmap, 0xC008_0101, &u32s(&[0x1000, 0]));
        let handle = out.u32(4);
        let mut alloc = vec![0u8; 32];
        alloc[..4].copy_from_slice(&handle.to_le_bytes());
        alloc[12..16].copy_from_slice(&0x1000u32.to_le_bytes());
        alloc[24..].copy_from_slice(&0x2000_0000u64.to_le_bytes());
        call(&mut nv, nvmap, 0xC020_0104, &alloc);
        call(&mut nv, as_fd, 0x4028_4109, &[0u8; 40]);
        let mut map = vec![0u8; 40];
        map[0..4].copy_from_slice(&1u32.to_le_bytes());
        map[8..12].copy_from_slice(&handle.to_le_bytes());
        map[32..40].copy_from_slice(&0x500_0000u64.to_le_bytes());
        let (e, _) = call(&mut nv, as_fd, 0xC028_4106, &map);
        assert_eq!(e, NvError::SUCCESS);
        call(&mut nv, as_fd, BIND_CHANNEL_CODE, &ch.to_le_bytes());
        let mut ex2 = vec![0u8; 32];
        ex2[..4].copy_from_slice(&0x800u32.to_le_bytes());
        let (_, out) = call(&mut nv, ch, ALLOC_GPFIFO_EX2_CODE, &ex2);
        let syncpt = out.u32(12);
        let (e, _) = call(&mut nv, ch, ALLOC_OBJ_CTX_CODE, &u32s(&[0xB197, 0, 0, 0]));
        assert_eq!(e, NvError::SUCCESS);
        (nv, mem, as_fd, ch, syncpt)
    }

    #[test]
    fn a_channel_waits_at_an_acquire_until_another_channel_releases() {
        let (mut nv, mem, as_fd, first, first_syncpt) = ready_with_space();
        let (second, second_syncpt) = another_channel(&mut nv, as_fd);
        // SEMAPHOREA..D as one increasing command: address 0x500_0200, payload, operation.
        let semaphore = |at: u32, payload: u32, op: u32| [0x2004_0004, 0, at, payload, op];
        let release = 2 | 1 << 24;
        // The first channel acquires 1 at 0x500_0200, then releases 0xA at 0x500_0300.
        let mut pb = semaphore(0x500_0200, 1, 1).to_vec();
        pb.extend(semaphore(0x500_0300, 0xA, release));
        assert!(mem.write(0x2000_0000, &words(&pb)));
        let (e, out) = submit(
            &mut nv,
            first,
            &[entry(0x500_0000, 10)],
            0x104,
            (first_syncpt, 1),
        );
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(
            (out.u32(16), out.u32(20)),
            (first_syncpt, 1),
            "the fence is promised"
        );
        assert_eq!(mem.read_u32(0x2000_0300), Some(0), "the channel waits");
        assert_eq!(nv.gpu.syncpoints.value(first_syncpt), Some(0));
        // A later submission of the same channel waits behind it.
        assert!(mem.write(0x2000_0040, &words(&semaphore(0x500_0304, 0xB, release))));
        let (e, _) = submit(
            &mut nv,
            first,
            &[entry(0x500_0040, 5)],
            0x104,
            (first_syncpt, 1),
        );
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(mem.read_u32(0x2000_0304), Some(0));
        assert_eq!(nv.gpu.parked().len(), 2, "{:?}", nv.gpu.parked());
        assert!(
            nv.gpu.parked()[0].contains("holds 0x0"),
            "{:?}",
            nv.gpu.parked()
        );
        // The second channel releases 1 there: the first goes on, both of its submissions end.
        assert!(mem.write(0x2000_0080, &words(&semaphore(0x500_0200, 1, release))));
        let (e, _) = submit(
            &mut nv,
            second,
            &[entry(0x500_0080, 5)],
            0x104,
            (second_syncpt, 1),
        );
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(mem.read_u32(0x2000_0300), Some(0xA));
        assert_eq!(mem.read_u32(0x2000_0304), Some(0xB));
        assert_eq!(nv.gpu.syncpoints.value(first_syncpt), Some(2));
        assert_eq!(nv.gpu.syncpoints.value(second_syncpt), Some(1));
        assert!(nv.gpu.parked().is_empty());
        assert_eq!(nv.gpu.stats.words, 20, "each word once");
    }

    /// A pushbuffer entry for `words` words at GPU address `va`.
    fn entry(va: u64, words: u64) -> [u8; 8] {
        ((va & 0xFFFF_FFFF) | ((va >> 32) << 32) | (words << (32 + 10))).to_le_bytes()
    }

    /// `SUBMIT_GPFIFO` with entries inline.
    fn submit(
        nv: &mut Nvdrv,
        ch: u32,
        entries: &[[u8; 8]],
        flags: u32,
        fence: (u32, u32),
    ) -> (NvError, Args) {
        let mut a = vec![0u8; 24];
        a[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        a[12..16].copy_from_slice(&flags.to_le_bytes());
        a[16..20].copy_from_slice(&fence.0.to_le_bytes());
        a[20..24].copy_from_slice(&fence.1.to_le_bytes());
        for e in entries {
            a.extend_from_slice(e);
        }
        let code = 0xC000_4808 | ((a.len() as u32) << 16);
        call(nv, ch, code, &a)
    }

    fn words(ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A queue that keeps what it is given, for the test to run.
    #[derive(Default)]
    struct Kept(std::sync::Mutex<Vec<crate::gpu::Task>>);

    impl crate::gpu::Queue for Kept {
        fn push(&self, task: crate::gpu::Task) {
            self.0.lock().unwrap().push(task);
        }
    }

    #[test]
    fn a_deferred_submission_gives_its_fence_at_once_and_runs_later_the_same() {
        let (mut nv, mem, ch, syncpt) = ready();
        let kept = std::sync::Arc::new(Kept::default());
        nv.gpu.queue = Some(kept.clone());
        let pb = words(&[
            0x2001_0000,
            0xB197,
            0x2004_06C0,
            0,
            0x500_0100,
            0xFEED_F00D,
            0,
        ]);
        assert!(mem.write(0x2000_0000, &pb));
        // INCREMENT (bit 8) with the one increment the semaphore release stands for.
        let (e, out) = submit(&mut nv, ch, &[entry(0x500_0000, 7)], 0x104, (syncpt, 1));
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(
            (out.u32(16), out.u32(20)),
            (syncpt, 1),
            "the fence is at once"
        );
        assert_eq!(mem.read_u32(0x2000_0100), Some(0), "nothing has run");
        assert_eq!(nv.gpu.syncpoints.value(syncpt), Some(0));
        assert_eq!(
            nv.gpu.syncpoints.max(syncpt),
            Some(1),
            "the increment is promised"
        );
        // The program uses the memory of its commands again: the submission has its own copy.
        assert!(mem.write(0x2000_0000, &words(&[0; 7])));
        let mut tasks = std::mem::take(&mut *kept.0.lock().unwrap());
        assert_eq!(tasks.len(), 1);
        let mut steps = 0;
        let done = loop {
            steps += 1;
            if let Some(done) = nv.gpu.run_task(&mut tasks[0]).unwrap() {
                break done;
            }
        };
        assert_eq!(steps, 2, "an entry, then the end of the submission");
        assert_eq!(done.fence, (syncpt, 1));
        assert_eq!(mem.read_u32(0x2000_0100), Some(0xFEED_F00D));
        assert_eq!(nv.gpu.syncpoints.value(syncpt), Some(1));
    }

    #[test]
    fn a_fence_that_running_makes_cannot_be_given_before_the_run() {
        let (mut nv, _mem, ch, _) = ready();
        nv.gpu.queue = Some(std::sync::Arc::new(Kept::default()));
        let err = nv
            .ioctl(
                ch,
                0xC018_4808,
                &{
                    let mut a = vec![0u8; 24];
                    a[12..16].copy_from_slice(&2u32.to_le_bytes());
                    a
                },
                &[],
            )
            .unwrap_err();
        assert!(err.contains("FENCE_GET"), "{err}");
    }

    #[test]
    fn a_pushbuffer_binds_an_engine_and_releases_a_semaphore() {
        let (mut nv, mem, ch, syncpt) = ready();
        // SET_OBJECT(3D) on subchannel 0; then SET_REPORT_SEMAPHORE_A..D as one increasing
        // command: upper address 0, lower address, payload, operation RELEASE with four words.
        let pb = words(&[
            0x2001_0000,
            0xB197,
            0x2004_06C0,
            0,
            0x500_0100,
            0xFEED_F00D,
            0,
        ]);
        assert!(mem.write(0x2000_0000, &pb));
        let (e, out) = submit(&mut nv, ch, &[entry(0x500_0000, 7)], 2, (0, 0));
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(mem.read_u32(0x2000_0100), Some(0xFEED_F00D));
        // FENCE_GET incremented the channel's syncpoint once and returned the fence.
        assert_eq!((out.u32(16), out.u32(20)), (syncpt, 1));
        assert_eq!(nv.gpu.syncpoints.value(syncpt), Some(1));
        assert_eq!(nv.gpu.stats.submits, 1);
        assert_eq!(nv.gpu.stats.words, 7);
        assert_eq!(nv.gpu.stats.semaphore_releases, 1);
        assert_eq!(
            nv.gpu.channels.values().next().unwrap().subchannels[0],
            0xB197
        );
    }

    #[test]
    fn promised_increments_are_completed_and_wake_the_waiters() {
        let (mut nv, _mem, ch, syncpt) = ready();
        let ctrl = nv.open("/dev/nvhost-ctrl").unwrap();
        call(&mut nv, ctrl, 0xC004_001F, &u32s(&[2]));
        let token = nv.query_event(ctrl, 0x1000_0002).unwrap();
        // Wait for the second increment (with a timeout: a wait of 0 only asks, A-055).
        let wait = u32s(&[syncpt, 2, 1000, 2]);
        let (e, _) = call(&mut nv, ctrl, 0xC010_001E, &wait);
        assert_eq!(e, NvError::TIMEOUT);
        // Two increments promised (flag 0x100: fence.value is the count): no commands at all.
        let (e, out) = submit(&mut nv, ch, &[], 0x100, (syncpt, 2));
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!((out.u32(16), out.u32(20)), (syncpt, 2));
        assert_eq!(nv.gpu.syncpoints.value(syncpt), Some(2));
        let signals = nv.take_signals();
        assert!(signals.iter().any(|s| s.token == token && s.signaled));
    }

    #[test]
    fn work_that_cannot_run_stops_the_run_and_says_why() {
        let (mut nv, mem, ch, _) = ready();
        // An entry at an address nothing is mapped at.
        let err = nv
            .ioctl(
                ch,
                0xC020_4808,
                &{
                    let mut a = vec![0u8; 24];
                    a[8] = 1;
                    a.extend_from_slice(&entry(0x900_0000, 4));
                    a
                },
                &[],
            )
            .unwrap_err();
        assert!(err.contains("not mapped"), "{err}");
        // A command nothing implements.
        mem.write(0x2000_0000, &words(&[0xC000_0000]));
        let err = nv
            .ioctl(
                ch,
                0xC020_4808,
                &{
                    let mut a = vec![0u8; 24];
                    a[8] = 1;
                    a.extend_from_slice(&entry(0x500_0000, 1));
                    a
                },
                &[],
            )
            .unwrap_err();
        assert!(err.contains("operation 6"), "{err}");
        // A wait for a fence nothing brings about.
        let err = nv
            .ioctl(
                ch,
                0xC018_4808,
                &{
                    let mut a = vec![0u8; 24];
                    a[12] = 1;
                    a[16] = 9;
                    a[20] = 5;
                    a
                },
                &[],
            )
            .unwrap_err();
        assert!(err.contains("waits for syncpoint 9"), "{err}");
    }

    #[test]
    fn entries_can_come_in_the_second_buffer() {
        let (mut nv, mem, ch, _) = ready();
        mem.write(0x2000_0000, &words(&[0x2001_0000, 0xB197]));
        let mut header = vec![0u8; 24];
        header[8] = 1;
        let r = nv
            .ioctl(ch, 0xC018_481B, &header, &entry(0x500_0000, 2))
            .unwrap();
        assert_eq!(r.error, NvError::SUCCESS);
        assert_eq!(nv.gpu.stats.words, 2);
        // Too few bytes for the entries announced.
        let r = nv.ioctl(ch, 0xC018_481B, &header, &[0; 4]).unwrap();
        assert_eq!(r.error, NvError::INVALID_SIZE);
    }

    #[test]
    fn queries_about_errors_answer_that_there_are_none() {
        let (mut nv, _, _, ch) = channel();
        let (e, out) = call(&mut nv, ch, 0x8080_4816, &[]);
        assert_eq!((e, out.u32(0)), (NvError::SUCCESS, 0));
    }
}
