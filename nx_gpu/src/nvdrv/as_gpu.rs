//! `/dev/nvhost-as-gpu` ioctls (Switchbrew, "NV services", section 8.10): one GPU address space
//! per descriptor.

use super::args::{unimplemented, Args, Fail, IoResult, Ioctl};
use super::{Node, Nvdrv};
use crate::error::NvError;
use crate::vm::{MapRequest, Remap};

/// The type byte of the address space ioctls (`'A'`).
pub const TYPE: u32 = 0x41;

pub const BIND_CHANNEL: u32 = 1;
pub const ALLOC_SPACE: u32 = 2;
pub const FREE_SPACE: u32 = 3;
pub const UNMAP_BUFFER: u32 = 5;
pub const MAP_BUFFER_EX: u32 = 6;
pub const ALLOC_AS: u32 = 7;
pub const GET_VA_REGIONS: u32 = 8;
pub const ALLOC_AS_EX: u32 = 9;
pub const REMAP: u32 = 0x14;

pub fn name(code: Ioctl) -> Option<&'static str> {
    if code.ty() != TYPE {
        return None;
    }
    Some(match code.nr() {
        BIND_CHANNEL => "NVGPU_AS_IOCTL_BIND_CHANNEL",
        ALLOC_SPACE => "NVGPU_AS_IOCTL_ALLOC_SPACE",
        FREE_SPACE => "NVGPU_AS_IOCTL_FREE_SPACE",
        UNMAP_BUFFER => "NVGPU_AS_IOCTL_UNMAP_BUFFER",
        MAP_BUFFER_EX => "NVGPU_AS_IOCTL_MAP_BUFFER_EX",
        ALLOC_AS => "NVGPU_AS_IOCTL_ALLOC_AS",
        GET_VA_REGIONS => "NVGPU_AS_IOCTL_GET_VA_REGIONS",
        ALLOC_AS_EX => "NVGPU_AS_IOCTL_ALLOC_AS_EX",
        REMAP => "NVGPU_AS_IOCTL_REMAP",
        _ => return None,
    })
}

impl Nvdrv {
    pub(super) fn as_gpu_ioctl(
        &mut self,
        space: u32,
        code: Ioctl,
        args: &mut Args,
        extra_out: &mut Vec<u8>,
    ) -> IoResult {
        if code.ty() != TYPE {
            return unimplemented(format!(
                "a type {:#x} command on an address space",
                code.ty()
            ));
        }
        let (nr, size) = (code.nr(), code.size());
        match (nr, size) {
            // u32 channel_fd
            (BIND_CHANNEL, 4) => {
                let fd = args.u32(0);
                let Some(Node::Gpu(channel)) = self.nodes.get(&fd) else {
                    return Err(NvError::BAD_VALUE.into());
                };
                let channel = *channel;
                let ch = self
                    .gpu
                    .channels
                    .get_mut(&channel)
                    .ok_or(NvError::BAD_VALUE)?;
                match ch.space {
                    Some(s) if s != space => return Err(NvError::INVALID_STATE.into()),
                    _ => ch.space = Some(space),
                }
            }
            // u32 pages, u32 page_size, u32 flags, u32 pad, u64 align (in) / offset (out)
            (ALLOC_SPACE, 24) => {
                let a = self.space_mut(space)?;
                let at = a.alloc_space(args.u32(0), args.u32(4), args.u32(8), args.u64(16))?;
                args.set_u64(16, at);
            }
            // u64 offset, u32 pages, u32 page_size
            (FREE_SPACE, 16) => {
                let a = self.space_mut(space)?;
                a.free_space(args.u64(0), args.u32(8), args.u32(12))?;
            }
            // u64 offset
            (UNMAP_BUFFER, 8) => {
                self.space_mut(space)?.unmap_buffer(args.u64(0))?;
            }
            // u32 flags, u32 kind, u32 nvmap handle, u32 page_size, u64 buffer_offset,
            // u64 mapping_size, u64 align (in) / offset (out)
            (MAP_BUFFER_EX, 40) => {
                let handle = args.u32(8);
                let h = self.gpu.nvmap.get(handle).ok_or(NvError::BAD_VALUE)?;
                let cpu = h.addr.ok_or(NvError::INVALID_STATE)?;
                let req = MapRequest {
                    flags: args.u32(0),
                    kind: args.u32(4),
                    handle,
                    handle_cpu: cpu,
                    handle_size: u64::from(h.size),
                    page_size: args.u32(12),
                    buffer_offset: args.u64(16),
                    mapping_size: args.u64(24),
                    offset: args.u64(32),
                };
                let at = self.space_mut(space)?.map_buffer(&req)?;
                args.set_u64(32, at);
            }
            // u32 flags, s32 as_fd, u32 big_page_size, u32 reserved, u64 va_range_start,
            // u64 va_range_end, u64 va_range_split. This is the order a title sends (libnx's
            // `nvioctlNvhostAsGpu_InitializeEx`; Switchbrew lists big_page_size and flags the
            // other way round, docs/ASSUMPTIONS.md). A zero range keeps the default one.
            (ALLOC_AS_EX, 40) => {
                let (flags, big_page_size) = (args.u32(0), args.u32(8));
                let (start, end, split) = (args.u64(16), args.u64(24), args.u64(32));
                self.space_mut(space)?
                    .initialize(flags, big_page_size, start, end, split)?;
            }
            // An array of operations, 20 bytes each: u16 flags (bit 2 cacheable), u16 kind,
            // u32 nvmap handle (0 unmaps), u32 offset in the handle in big pages, u32 offset in
            // the address space in big pages, u32 number of big pages.
            (REMAP, n) if n > 0 && n % 20 == 0 => {
                for i in 0..n / 20 {
                    let op = args.bytes(i * 20, 20).to_vec();
                    let word = |o: usize| u32::from_le_bytes(op[o..o + 4].try_into().unwrap());
                    let (flags, kind) = (
                        u32::from(u16::from_le_bytes([op[0], op[1]])),
                        u32::from(u16::from_le_bytes([op[2], op[3]])),
                    );
                    let (handle, from, to, count) = (word(4), word(8), word(12), word(16));
                    let (start, len) = (u64::from(to) << 16, u64::from(count) << 16);
                    let target = if handle == 0 {
                        None
                    } else {
                        let h = self.gpu.nvmap.get(handle).ok_or(NvError::BAD_VALUE)?;
                        let cpu = h.addr.ok_or(NvError::INVALID_STATE)?;
                        let begin = u64::from(from) << 16;
                        if begin + len > u64::from(h.size) {
                            return Err(NvError::BAD_VALUE.into());
                        }
                        Some(Remap {
                            handle,
                            cpu: cpu + begin,
                            kind,
                            cacheable: flags & 4 != 0,
                        })
                    };
                    self.space_mut(space)?.remap(start, len, target)?;
                }
            }
            // u64 buf_addr, u32 buf_size, u32 reserved, va_region regions[2] (24 bytes each)
            (GET_VA_REGIONS, 64) => {
                let regions = self.space_mut(space)?.va_regions();
                args.set_u32(8, 48);
                extra_out.clear();
                for (i, (offset, page_size, pages)) in regions.iter().enumerate() {
                    let mut r = [0u8; 24];
                    r[..8].copy_from_slice(&offset.to_le_bytes());
                    r[8..12].copy_from_slice(&page_size.to_le_bytes());
                    r[16..].copy_from_slice(&pages.to_le_bytes());
                    args.put(16 + 24 * i, &r);
                    extra_out.extend_from_slice(&r);
                }
            }
            _ => return unimplemented(format!("address space command {nr:#x} of size {size}")),
        }
        Ok(())
    }

    fn space_mut(&mut self, id: u32) -> Result<&mut crate::vm::AddressSpace, Fail> {
        self.gpu
            .spaces
            .get_mut(&id)
            .ok_or(Fail::Nv(NvError::INVALID_STATE))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    const ALLOC_AS_EX_CODE: u32 = 0x4028_4109;
    const ALLOC_SPACE_CODE: u32 = 0xC018_4102;
    const MAP_EX_CODE: u32 = 0xC028_4106;
    const UNMAP_CODE: u32 = 0xC008_4105;
    const VA_REGIONS_CODE: u32 = 0xC040_4108;
    const BIND_CODE: u32 = 0x4004_4101;

    fn setup() -> (Nvdrv, u32, u32) {
        let (mut nv, _) = nvdrv();
        let nvmap = nv.open("/dev/nvmap").unwrap();
        let as_fd = nv.open("/dev/nvhost-as-gpu").unwrap();
        (nv, nvmap, as_fd)
    }

    fn backed_handle(nv: &mut Nvdrv, nvmap: u32, size: u32, cpu: u64) -> u32 {
        let (_, out) = call(nv, nvmap, 0xC008_0101, &u32s(&[size, 0]));
        let handle = out.u32(4);
        let mut a = vec![0u8; 32];
        a[..4].copy_from_slice(&handle.to_le_bytes());
        a[8..12].copy_from_slice(&1u32.to_le_bytes());
        a[12..16].copy_from_slice(&0x1000u32.to_le_bytes());
        a[24..].copy_from_slice(&cpu.to_le_bytes());
        let (e, _) = call(nv, nvmap, 0xC020_0104, &a);
        assert_eq!(e, NvError::SUCCESS);
        handle
    }

    fn alloc_as_ex(nv: &mut Nvdrv, fd: u32) {
        let mut a = vec![0u8; 40];
        a[0..4].copy_from_slice(&1u32.to_le_bytes()); // flags: the program manages addresses
        a[8..12].copy_from_slice(&0x20000u32.to_le_bytes()); // big page size
        a[16..24].copy_from_slice(&0x0400_0000u64.to_le_bytes());
        a[24..32].copy_from_slice(&0x100_0000_0000u64.to_le_bytes());
        a[32..40].copy_from_slice(&0x1_0000_0000u64.to_le_bytes());
        let (e, _) = call(nv, fd, ALLOC_AS_EX_CODE, &a);
        assert_eq!(e, NvError::SUCCESS);
    }

    #[test]
    fn remap_maps_big_pages_of_a_handle_and_unmaps_with_handle_zero() {
        let (mut nv, nvmap, fd) = setup();
        call(&mut nv, fd, ALLOC_AS_EX_CODE, &[0u8; 40]);
        let handle = backed_handle(&mut nv, nvmap, 0x40000, 0x4000_0000);
        // flags 4 (cacheable), kind 0xFE, the handle, 1 page into it, to page 0x500000 of the
        // address space, 2 pages; the ioctl number carries the size of one operation (20).
        let mut op = vec![0u8; 20];
        op[0..2].copy_from_slice(&4u16.to_le_bytes());
        op[2..4].copy_from_slice(&0xFEu16.to_le_bytes());
        op[4..8].copy_from_slice(&handle.to_le_bytes());
        op[8..12].copy_from_slice(&1u32.to_le_bytes());
        op[12..16].copy_from_slice(&0x50_0000u32.to_le_bytes());
        op[16..20].copy_from_slice(&2u32.to_le_bytes());
        let (e, _) = call(&mut nv, fd, 0xC014_4114, &op);
        assert_eq!(e, NvError::SUCCESS);
        let space = nv.gpu.spaces.values().next().unwrap();
        let va = 0x50_0000u64 << 16;
        assert_eq!(space.translate(va), Some((0x4001_0000, 0x20000)));
        assert!(space.mappings().next().unwrap().cacheable);
        // Handle 0 takes it away again; two operations in one call work too.
        let mut two = op.clone();
        two[4..8].copy_from_slice(&0u32.to_le_bytes());
        two.extend_from_slice(&op);
        let (e, _) = call(&mut nv, fd, 0xC028_4114, &two);
        assert_eq!(e, NvError::SUCCESS);
        let space = nv.gpu.spaces.values().next().unwrap();
        assert_eq!(
            space.translate(va),
            Some((0x4001_0000, 0x20000)),
            "mapped by the second"
        );
        // A handle that does not exist, and a range past the end of the handle.
        let mut bad = op.clone();
        bad[4..8].copy_from_slice(&99u32.to_le_bytes());
        let (e, _) = call(&mut nv, fd, 0xC014_4114, &bad);
        assert_eq!(e, NvError::BAD_VALUE);
        let mut long = op.clone();
        long[16..20].copy_from_slice(&4u32.to_le_bytes());
        let (e, _) = call(&mut nv, fd, 0xC014_4114, &long);
        assert_eq!(e, NvError::BAD_VALUE, "1 + 4 pages of a 4-page handle");
    }

    #[test]
    fn alloc_as_ex_in_the_order_libnx_sends_it_with_no_range() {
        let (mut nv, _, fd) = setup();
        // flags 1, as_fd 0, big page size 0x10000, reserved 0, then three zero range words.
        let mut a = vec![0u8; 40];
        a[0..4].copy_from_slice(&1u32.to_le_bytes());
        a[8..12].copy_from_slice(&0x10000u32.to_le_bytes());
        let (e, _) = call(&mut nv, fd, ALLOC_AS_EX_CODE, &a);
        assert_eq!(e, NvError::SUCCESS);
        let space = nv.gpu.spaces.values().next().unwrap();
        assert!(space.userspace_managed);
        assert_eq!(
            space.range(),
            (crate::vm::DEFAULT_START, crate::vm::DEFAULT_END)
        );
        let r = nv.ioctl(fd, VA_REGIONS_CODE, &[0u8; 64], &[]).unwrap();
        assert_eq!(r.args.u64(16), crate::vm::DEFAULT_START);
        assert_eq!(r.args.u32(48), 0x10000);
    }

    #[test]
    fn va_regions_follow_the_range_the_program_set() {
        let (mut nv, _, fd) = setup();
        alloc_as_ex(&mut nv, fd);
        let r = nv.ioctl(fd, VA_REGIONS_CODE, &[0u8; 64], &[]).unwrap();
        assert_eq!(r.error, NvError::SUCCESS);
        assert_eq!(r.args.u32(8), 48);
        assert_eq!(r.args.u64(16), 0x0400_0000);
        assert_eq!(r.args.u32(24), 0x1000);
        assert_eq!(r.args.u64(40), 0x1_0000_0000);
        assert_eq!(r.args.u32(48), 0x20000);
        assert_eq!(r.extra_out.len(), 48);
        assert_eq!(&r.extra_out[..], r.args.bytes(16, 48));
        // The range can be set only once.
        let (e, _) = call(&mut nv, fd, ALLOC_AS_EX_CODE, &[0u8; 40]);
        assert_eq!(e, NvError::INVALID_STATE);
    }

    #[test]
    fn reserve_map_translate_and_unmap() {
        let (mut nv, nvmap, fd) = setup();
        alloc_as_ex(&mut nv, fd);
        let cpu = 0x3000_0000u64;
        let handle = backed_handle(&mut nv, nvmap, 0x20000, cpu);

        // Reserve 4 big pages, then map the handle at a fixed address inside the reservation.
        let mut a = vec![0u8; 24];
        a[0..4].copy_from_slice(&4u32.to_le_bytes());
        a[4..8].copy_from_slice(&0x10000u32.to_le_bytes());
        let (e, out) = call(&mut nv, fd, ALLOC_SPACE_CODE, &a);
        assert_eq!(e, NvError::SUCCESS);
        let base = out.u64(16);
        assert_eq!(base, 0x1_0000_0000);

        let mut m = vec![0u8; 40];
        m[0..4].copy_from_slice(&1u32.to_le_bytes()); // fixed
        m[4..8].copy_from_slice(&u32::MAX.to_le_bytes()); // kind -1
        m[8..12].copy_from_slice(&handle.to_le_bytes());
        m[32..40].copy_from_slice(&(base + 0x10000).to_le_bytes());
        let (e, out) = call(&mut nv, fd, MAP_EX_CODE, &m);
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(out.u64(32), base + 0x10000);
        let space = nv.gpu.spaces.values().next().unwrap();
        assert_eq!(
            space.translate(base + 0x10010),
            Some((cpu + 0x10, 0x20000 - 0x10))
        );

        let (e, _) = call(&mut nv, fd, UNMAP_CODE, &(base + 0x10000).to_le_bytes());
        assert_eq!(e, NvError::SUCCESS);
        let (e, _) = call(&mut nv, fd, UNMAP_CODE, &(base + 0x10000).to_le_bytes());
        assert_eq!(e, NvError::BAD_VALUE);
    }

    #[test]
    fn mapping_a_handle_without_memory_fails() {
        let (mut nv, nvmap, fd) = setup();
        let (_, out) = call(&mut nv, nvmap, 0xC008_0101, &u32s(&[0x1000, 0]));
        let mut m = vec![0u8; 40];
        m[8..12].copy_from_slice(&out.u32(4).to_le_bytes());
        let (e, _) = call(&mut nv, fd, MAP_EX_CODE, &m);
        assert_eq!(e, NvError::INVALID_STATE);
        m[8..12].copy_from_slice(&99u32.to_le_bytes());
        let (e, _) = call(&mut nv, fd, MAP_EX_CODE, &m);
        assert_eq!(e, NvError::BAD_VALUE);
    }

    #[test]
    fn a_channel_binds_to_one_address_space() {
        let (mut nv, _, as1) = setup();
        let as2 = nv.open("/dev/nvhost-as-gpu").unwrap();
        let ch = nv.open("/dev/nvhost-gpu").unwrap();
        let (e, _) = call(&mut nv, as1, BIND_CODE, &ch.to_le_bytes());
        assert_eq!(e, NvError::SUCCESS);
        let (e, _) = call(&mut nv, as1, BIND_CODE, &ch.to_le_bytes());
        assert_eq!(
            e,
            NvError::SUCCESS,
            "binding twice to the same space is harmless"
        );
        let (e, _) = call(&mut nv, as2, BIND_CODE, &ch.to_le_bytes());
        assert_eq!(e, NvError::INVALID_STATE, "never unbound, never moved");
        let (e, _) = call(&mut nv, as1, BIND_CODE, &77u32.to_le_bytes());
        assert_eq!(e, NvError::BAD_VALUE, "not a channel");
    }
}
