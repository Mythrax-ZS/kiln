//! A small world to render in: guest memory with a GPU address space over it, and a register file
//! of the 3D engine to set the state with.

#![allow(dead_code)]

use nx_gpu::mem::GuestMemory;
use nx_gpu::render::Scene;
use nx_gpu::state::{CbufBindings, Regs};
use nx_gpu::vm::{map_flags, AddressSpace, MapRequest};
use nx_vulkan::Vk;
use std::sync::Mutex;

/// Where the test memory is in the guest, and where in the GPU's address space.
pub const CPU_BASE: u64 = 0x4000_0000;
pub const GPU_BASE: u64 = 0x1_0000_0000;
pub const SIZE: u64 = 64 << 20;

pub fn vulkan() -> Option<Vk> {
    match Vk::new() {
        Ok(v) => Some(v),
        Err(e) => {
            if std::env::var("RECOMP_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
                panic!("no Vulkan device and RECOMP_REQUIRE_EXTERNAL=1: {e}");
            }
            eprintln!("SKIPPED: no Vulkan device: {e}");
            None
        }
    }
}

pub struct Memory(Mutex<Vec<u8>>);

impl GuestMemory for Memory {
    fn read(&self, addr: u64, out: &mut [u8]) -> bool {
        let m = self.0.lock().unwrap();
        let Some(off) = addr.checked_sub(CPU_BASE) else {
            return false;
        };
        let Some(src) = m.get(off as usize..off as usize + out.len()) else {
            return false;
        };
        out.copy_from_slice(src);
        true
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        let mut m = self.0.lock().unwrap();
        let Some(off) = addr.checked_sub(CPU_BASE) else {
            return false;
        };
        let Some(dst) = m.get_mut(off as usize..off as usize + data.len()) else {
            return false;
        };
        dst.copy_from_slice(data);
        true
    }
}

pub struct World {
    pub mem: Memory,
    pub space: AddressSpace,
    pub regs: Vec<u32>,
    pub cbufs: CbufBindings,
}

impl World {
    pub fn new() -> World {
        let mut space = AddressSpace::new();
        space
            .map_buffer(&MapRequest {
                flags: map_flags::FIXED_OFFSET,
                kind: 0,
                handle: 1,
                handle_cpu: CPU_BASE,
                handle_size: SIZE,
                page_size: 0,
                buffer_offset: 0,
                mapping_size: 0,
                offset: GPU_BASE,
            })
            .unwrap();
        World {
            mem: Memory(Mutex::new(vec![0; SIZE as usize])),
            space,
            regs: vec![0; 0x1000],
            cbufs: CbufBindings::default(),
        }
    }

    pub fn scene(&self) -> Scene<'_> {
        Scene {
            regs: Regs(&self.regs),
            cbufs: &self.cbufs,
            mem: &self.mem,
            space: &self.space,
        }
    }

    /// Write a register by its byte offset.
    pub fn set(&mut self, offset: u32, value: u32) {
        self.regs[offset as usize / 4] = value;
    }

    /// Write a 40-bit address into two registers (upper then lower) at `offset`.
    pub fn set_address(&mut self, offset: u32, va: u64) {
        self.set(offset, (va >> 32) as u32);
        self.set(offset + 4, va as u32);
    }

    /// Bind color target `j` as the only output: an address, a size and a format.
    pub fn color_target(&mut self, j: u32, va: u64, width: u32, height: u32, format: u32) {
        let b = 0x800 + j * 64;
        self.set_address(b, va);
        self.set(b + 8, width);
        self.set(b + 0xC, height);
        self.set(b + 0x10, format);
        // Block linear, a block of 16 GOBs; it does not matter to the renderer.
        self.set(b + 0x14, 4 << 4);
        self.set(b + 0x18, 1);
        self.set(0x121c, 1 | j << 4);
    }

    /// Write bytes into the guest memory behind GPU address `va`.
    pub fn put(&self, va: u64, data: &[u8]) {
        assert!(self.mem.write(CPU_BASE + (va - GPU_BASE), data));
    }

    pub fn cpu(&self, va: u64) -> u64 {
        CPU_BASE + (va - GPU_BASE)
    }
}
