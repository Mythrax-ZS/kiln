//! `/dev/nvmap` ioctls (Switchbrew, "NV services", section 8.2).

use super::args::{unimplemented, Args, IoResult, Ioctl};
use super::Nvdrv;

pub const CREATE: u32 = 0xC008_0101;
pub const FROM_ID: u32 = 0xC008_0103;
pub const ALLOC: u32 = 0xC020_0104;
pub const FREE: u32 = 0xC018_0105;
pub const PARAM: u32 = 0xC00C_0109;
pub const GET_ID: u32 = 0xC008_010E;

pub fn name(code: Ioctl) -> Option<&'static str> {
    Some(match code.0 {
        CREATE => "NVMAP_IOC_CREATE",
        FROM_ID => "NVMAP_IOC_FROM_ID",
        ALLOC => "NVMAP_IOC_ALLOC",
        FREE => "NVMAP_IOC_FREE",
        PARAM => "NVMAP_IOC_PARAM",
        GET_ID => "NVMAP_IOC_GET_ID",
        _ => return None,
    })
}

impl Nvdrv {
    pub(super) fn nvmap_ioctl(&mut self, code: Ioctl, args: &mut Args) -> IoResult {
        match code.0 {
            // u32 size (in), u32 handle (out)
            CREATE => {
                let handle = self.gpu.nvmap.create(args.u32(0))?;
                args.set_u32(4, handle);
            }
            // u32 id (in), u32 handle (out)
            FROM_ID => {
                let handle = self.gpu.nvmap.from_id(args.u32(0))?;
                args.set_u32(4, handle);
            }
            // u32 handle, u32 heapmask, u32 flags, u32 align, u8 kind, u8 pad[7], u64 addr
            ALLOC => {
                self.gpu.nvmap.alloc(
                    args.u32(0),
                    args.u32(4),
                    args.u32(8),
                    args.u32(12),
                    args.data[16],
                    args.u64(24),
                )?;
            }
            // u32 handle, u32 pad, u64 address (out), u32 size (out), u32 flags (out)
            FREE => {
                let freed = self.gpu.nvmap.free(args.u32(0))?;
                if freed.addr != 0 {
                    // The decoder's frames in that memory go with it.
                    self.media
                        .forget(freed.addr..freed.addr + u64::from(freed.size));
                }
                args.set_u64(8, freed.addr);
                args.set_u32(16, freed.size);
                args.set_u32(20, freed.flags);
            }
            // u32 handle, u32 param, u32 result (out)
            PARAM => {
                let v = self.gpu.nvmap.param(args.u32(0), args.u32(4))?;
                args.set_u32(8, v);
            }
            // u32 id (out), u32 handle (in)
            GET_ID => {
                let id = self.gpu.nvmap.get_id(args.u32(4))?;
                args.set_u32(0, id);
            }
            other => {
                return unimplemented(format!(
                    "nvmap command {:#x} (type {:#x}, size {})",
                    other,
                    code.nr(),
                    code.size()
                ))
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::error::NvError;

    #[test]
    fn create_alloc_param_and_free_round_trip() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvmap").unwrap();
        let (e, out) = call(&mut nv, fd, CREATE, &u32s(&[0x3000, 0]));
        assert_eq!(e, NvError::SUCCESS);
        let handle = out.u32(4);
        assert_eq!(handle, 1);

        // kind sits in byte 16; the u64 address at 24.
        let mut buf = vec![0u8; 32];
        buf[..4].copy_from_slice(&handle.to_le_bytes());
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        buf[12..16].copy_from_slice(&0x1000u32.to_le_bytes());
        buf[16] = 6;
        buf[24..].copy_from_slice(&0x2_0000_0000u64.to_le_bytes());
        let (e, _) = call(&mut nv, fd, ALLOC, &buf);
        assert_eq!(e, NvError::SUCCESS);

        let (e, out) = call(&mut nv, fd, PARAM, &u32s(&[handle, 1, 0]));
        assert_eq!((e, out.u32(8)), (NvError::SUCCESS, 0x3000));
        let (_, out) = call(&mut nv, fd, PARAM, &u32s(&[handle, 5, 0]));
        assert_eq!(out.u32(8), 6, "kind");
        let (e, out) = call(&mut nv, fd, GET_ID, &u32s(&[0, handle]));
        assert_eq!((e, out.u32(0)), (NvError::SUCCESS, handle));

        let (e, out) = call(&mut nv, fd, FREE, &u32s(&[handle, 0]));
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(out.u64(8), 0x2_0000_0000);
        assert_eq!((out.u32(16), out.u32(20)), (0x3000, 0));
        let (e, _) = call(&mut nv, fd, FREE, &u32s(&[handle, 0]));
        assert_eq!(e, NvError::BAD_VALUE, "gone");
    }

    #[test]
    fn errors_come_back_as_codes_with_the_struct_intact() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvmap").unwrap();
        let (e, out) = call(&mut nv, fd, CREATE, &u32s(&[0, 77]));
        assert_eq!(e, NvError::BAD_PARAMETER);
        assert_eq!(out.u32(4), 77, "untouched on failure");
        let (e, _) = call(&mut nv, fd, PARAM, &u32s(&[5, 1, 0]));
        assert_eq!(e, NvError::BAD_VALUE);
    }
}
