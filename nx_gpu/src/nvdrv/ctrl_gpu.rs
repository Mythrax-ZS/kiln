//! `/dev/nvhost-ctrl-gpu` ioctls (Switchbrew, "NV services", section 8.13): what the GPU is.
//!
//! The answers describe the Tegra X1's GM20B as Switchbrew's page lists them (docs/ASSUMPTIONS.md
//! for the values the page does not give).

use super::args::{unimplemented, Args, IoResult, Ioctl};
use super::Nvdrv;
use crate::error::NvError;

pub const ZCULL_GET_CTX_SIZE: u32 = 1;
pub const ZCULL_GET_INFO: u32 = 2;
pub const ZBC_SET_TABLE: u32 = 3;
pub const GET_CHARACTERISTICS: u32 = 5;
pub const GET_TPC_MASKS: u32 = 6;
pub const FLUSH_L2: u32 = 7;
pub const ZBC_GET_ACTIVE_SLOT_MASK: u32 = 0x14;
pub const GET_GPU_TIME: u32 = 0x1C;

/// The type byte of the `nvgpu` GPU ioctls (`'G'`).
pub const TYPE: u32 = 0x47;

/// The context size `ZCULL_GET_CTX_SIZE` answers (docs/ASSUMPTIONS.md).
pub const ZCULL_CTX_SIZE: u32 = 0x1000;

pub fn name(code: Ioctl) -> Option<&'static str> {
    if code.ty() != TYPE {
        return None;
    }
    Some(match code.nr() {
        ZCULL_GET_CTX_SIZE => "NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE",
        ZCULL_GET_INFO => "NVGPU_GPU_IOCTL_ZCULL_GET_INFO",
        ZBC_SET_TABLE => "NVGPU_GPU_IOCTL_ZBC_SET_TABLE",
        GET_CHARACTERISTICS => "NVGPU_GPU_IOCTL_GET_CHARACTERISTICS",
        GET_TPC_MASKS => "NVGPU_GPU_IOCTL_GET_TPC_MASKS",
        FLUSH_L2 => "NVGPU_GPU_IOCTL_FLUSH_L2",
        ZBC_GET_ACTIVE_SLOT_MASK => "NVGPU_GPU_IOCTL_ZBC_GET_ACTIVE_SLOT_MASK",
        GET_GPU_TIME => "NVGPU_GPU_IOCTL_GET_GPU_TIME",
        _ => return None,
    })
}

/// `struct nvgpu_gpu_characteristics` for a GM20B: 160 bytes (Switchbrew, section 8.13).
pub fn characteristics() -> [u8; 160] {
    let mut b = [0u8; 160];
    let mut put32 = |off: usize, v: u32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
    put32(0, 0x120); // arch: GM200
    put32(4, 0xB); // impl: GM20B
    put32(8, 0xA1); // rev
    put32(12, 1); // num_gpc
    put32(32, 2); // num_tpc_per_gpc
    put32(36, 0x20); // bus_type: AXI
    put32(40, 0x20000); // big_page_size
    put32(44, 0x20000); // compression_page_size
    put32(48, 0x1B); // pde_coverage_bit_count
    put32(52, 0x30000); // available_big_page_sizes
    put32(56, 1); // gpc_mask
    put32(60, 0x503); // sm_arch_sm_version
    put32(64, 0x503); // sm_arch_spa_version
    put32(68, 0x80); // sm_arch_warp_count
    put32(72, 0x28); // gpu_va_bit_count
    put32(88, super::super::gpu::class::TWOD);
    put32(92, super::super::gpu::class::THREED);
    put32(96, super::super::gpu::class::COMPUTE);
    put32(100, super::super::gpu::class::GPFIFO);
    put32(104, super::super::gpu::class::INLINE_TO_MEMORY);
    put32(108, super::super::gpu::class::DMA_COPY);
    put32(112, 1); // max_fbps_count
    put32(120, 2); // max_ltc_per_fbp
    put32(124, 1); // max_lts_per_ltc
    put32(132, 1); // max_gpc_count
    put32(136, 0x21D70); // rop_l2_en_mask_0
                         // l2_cache_size at 16, flags at 80, chipname at 144 are 64-bit.
    b[16..24].copy_from_slice(&0x40000u64.to_le_bytes());
    b[80..88].copy_from_slice(&0x55u64.to_le_bytes());
    b[144..149].copy_from_slice(b"gm20b"); // chipname, zero padded
    b
}

/// `struct nvgpu_gpu_zcull_info`: 40 bytes.
fn zcull_info() -> [u8; 40] {
    let words: [u32; 10] = [0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xC0, 0x20, 0x40, 0x10];
    let mut b = [0u8; 40];
    for (i, w) in words.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    b
}

/// The GPU's timer ticks (about 1.625 ns each) since the first call.
fn gpu_ticks() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let ns = START.get_or_init(Instant::now).elapsed().as_nanos() as u64;
    ns * 8 / 13
}

impl Nvdrv {
    pub(super) fn ctrl_gpu_ioctl(
        &mut self,
        code: Ioctl,
        args: &mut Args,
        extra_out: &mut Vec<u8>,
    ) -> IoResult {
        if code.ty() != TYPE {
            return unimplemented(format!(
                "a type {:#x} command on the GPU control node",
                code.ty()
            ));
        }
        match (code.nr(), code.size()) {
            (ZCULL_GET_CTX_SIZE, 4) => args.set_u32(0, ZCULL_CTX_SIZE),
            (ZCULL_GET_INFO, 40) => args.put(0, &zcull_info()),
            // The compression-bit tables only matter to rendering that uses them; the entries are
            // accepted and not kept.
            (ZBC_SET_TABLE, 44) => {}
            (GET_CHARACTERISTICS, 176) => {
                // The first word is the size of the buffer, which the driver overwrites with the
                // size it filled. The second is the address of a buffer, which the characteristics
                // do not use: they are returned inline, after these two words, and a title's call
                // passes a null address (docs/ASSUMPTIONS.md A-139).
                if args.u64(0) == 0 {
                    return Err(NvError::BAD_PARAMETER.into());
                }
                args.set_u64(0, 160);
                args.put(16, &characteristics());
                // `Ioctl3` has the program give the buffer as its second output buffer.
                extra_out.clear();
                extra_out.extend_from_slice(&characteristics());
            }
            (GET_TPC_MASKS, 24) => {
                // One 32-bit mask per GPC: the GM20B has one GPC with two TPCs.
                args.set_u64(16, 0b11);
            }
            (FLUSH_L2, 8) => {}
            (ZBC_GET_ACTIVE_SLOT_MASK, 8) => {
                args.set_u32(0, 7);
                args.set_u32(4, 1);
            }
            (GET_GPU_TIME, 16) => args.set_u64(0, gpu_ticks()),
            _ => {
                return unimplemented(format!(
                    "GPU control command {:#x} of size {}",
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

    const CHARACTERISTICS: u32 = 0xC0B0_4705;

    fn gpu_fd(nv: &mut Nvdrv) -> u32 {
        nv.open("/dev/nvhost-ctrl-gpu").unwrap()
    }

    #[test]
    fn characteristics_describe_a_gm20b() {
        let (mut nv, _) = nvdrv();
        let fd = gpu_fd(&mut nv);
        let mut input = vec![0u8; 176];
        input[..8].copy_from_slice(&160u64.to_le_bytes());
        input[8..16].copy_from_slice(&1u64.to_le_bytes());
        let (e, out) = call(&mut nv, fd, CHARACTERISTICS, &input);
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(out.u64(0), 160);
        let gc = |off: usize| out.u32(16 + off);
        assert_eq!((gc(0), gc(4), gc(8)), (0x120, 0xB, 0xA1));
        assert_eq!(gc(92), 0xB197, "3D class");
        assert_eq!(gc(96), 0xB1C0, "compute class");
        assert_eq!(gc(88), 0x902D);
        assert_eq!(gc(100), 0xB06F);
        assert_eq!(gc(104), 0xA140);
        assert_eq!(gc(108), 0xB0B5);
        assert_eq!(out.u64(16 + 16), 0x40000);
        assert_eq!(out.u64(16 + 80), 0x55);
        assert_eq!(&out.bytes(16 + 144, 5), b"gm20b");
        // A size of zero is refused; a null address is not (a title passes one).
        let (e, _) = call(&mut nv, fd, CHARACTERISTICS, &[0u8; 176]);
        assert_eq!(e, NvError::BAD_PARAMETER);
        input[8..16].copy_from_slice(&0u64.to_le_bytes());
        let (e, out) = call(&mut nv, fd, CHARACTERISTICS, &input);
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!((out.u64(0), out.u32(16)), (160, 0x120));
        // `Ioctl3` has the characteristics come back in its second output buffer too.
        let r = nv.ioctl(fd, CHARACTERISTICS, &input, &[]).unwrap();
        assert_eq!(r.extra_out.len(), 160);
        assert_eq!(r.extra_out, characteristics());
    }

    #[test]
    fn small_answers_have_their_documented_shape() {
        let (mut nv, _) = nvdrv();
        let fd = gpu_fd(&mut nv);
        let (_, out) = call(&mut nv, fd, 0x8004_4701, &[]);
        assert_eq!(out.u32(0), ZCULL_CTX_SIZE);
        let (_, out) = call(&mut nv, fd, 0x8028_4702, &[]);
        assert_eq!((out.u32(0), out.u32(8), out.u32(36)), (0x20, 0x400, 0x10));
        let (_, out) = call(&mut nv, fd, 0xC018_4706, &u32s(&[4, 0, 0, 0, 0, 0]));
        assert_eq!(out.u64(16), 3);
        let (_, out) = call(&mut nv, fd, 0x8008_4714, &[]);
        assert_eq!(out.u32(0), 7);
        let (e, _) = call(&mut nv, fd, 0x4008_4707, &u32s(&[1, 0]));
        assert_eq!(e, NvError::SUCCESS);
    }

    #[test]
    fn the_gpu_clock_counts_up() {
        let (mut nv, _) = nvdrv();
        let fd = gpu_fd(&mut nv);
        let (_, a) = call(&mut nv, fd, 0xC010_471C, &[]);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let (_, b) = call(&mut nv, fd, 0xC010_471C, &[]);
        assert!(b.u64(0) > a.u64(0));
    }

    #[test]
    fn other_sizes_of_a_command_are_not_guessed() {
        let (mut nv, _) = nvdrv();
        let fd = gpu_fd(&mut nv);
        // The Switch 2 layout of GET_CHARACTERISTICS has another size.
        let err = nv.ioctl(fd, 0xC0E0_4705, &[], &[]).unwrap_err();
        assert!(err.contains("GET_CHARACTERISTICS"), "{err}");
    }
}
