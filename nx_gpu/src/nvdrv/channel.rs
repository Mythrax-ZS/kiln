//! `/dev/nvhost-gpu` ioctls (Switchbrew, "NV services", section 9): the command queue of the GPU.

use super::args::{unimplemented, Args, Fail, IoResult, Ioctl};
use super::{Node, Nvdrv};
use crate::error::NvError;
use crate::gpu::class;

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
        _ => return None,
    })
}

impl Nvdrv {
    pub(super) fn channel_ioctl(
        &mut self,
        id: u32,
        code: Ioctl,
        args: &mut Args,
        _extra_in: &[u8],
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
                let known = [
                    class::THREED,
                    class::COMPUTE,
                    class::TWOD,
                    class::INLINE_TO_MEMORY,
                    class::DMA_COPY,
                ];
                if !known.contains(&class_num) {
                    return unimplemented(format!("an object context of class {class_num:#x}"));
                }
                let ch = self.channel(id)?;
                if ch.space.is_none() || ch.syncpt.is_none() {
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
            (TYPE, SUBMIT_GPFIFO | SUBMIT_GPFIFO2, _) => {
                return unimplemented(
                    "a submission to the command processor, which is not built yet",
                )
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

    #[test]
    fn a_submission_is_not_answered_before_the_processor_exists() {
        let (mut nv, _, _, ch) = channel();
        let err = nv.ioctl(ch, 0xC018_4808, &[0; 24], &[]).unwrap_err();
        assert!(err.contains("SUBMIT_GPFIFO"), "{err}");
    }

    #[test]
    fn queries_about_errors_answer_that_there_are_none() {
        let (mut nv, _, _, ch) = channel();
        let (e, out) = call(&mut nv, ch, 0x8080_4816, &[]);
        assert_eq!((e, out.u32(0)), (NvError::SUCCESS, 0));
    }
}
