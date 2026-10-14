//! A small world to render in: guest memory with a GPU address space over it, and a register file
//! of the 3D engine to set the state with.

#![allow(dead_code)]

use nx_gpu::mem::GuestMemory;
use nx_gpu::render::Scene;
use nx_gpu::state::{CbufBindings, Regs};
use nx_gpu::vm::{map_flags, AddressSpace, MapRequest};
use nx_shader::asm::{assemble, exit, Enc};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_shader::sph::SPH_WORDS;
use nx_vulkan::Vk;

/// Where the test memory is in the guest, and where in the GPU's address space.
pub const CPU_BASE: u64 = 0x4000_0000;
pub const GPU_BASE: u64 = 0x1_0000_0000;
pub const SIZE: u64 = 64 << 20;

pub fn vulkan() -> Option<Vk> {
    match Vk::new() {
        Ok(v) => Some(v),
        Err(e) => {
            if std::env::var("KILN_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
                panic!("no Vulkan device and KILN_REQUIRE_EXTERNAL=1: {e}");
            }
            eprintln!("SKIPPED: no Vulkan device: {e}");
            None
        }
    }
}

/// Guest memory for tests: one page-aligned block, so a device can be given its address.
pub struct Memory {
    ptr: *mut u8,
}

// SAFETY: the block is owned by this value and only reached through its methods or the device.
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

impl Memory {
    fn new() -> Memory {
        let layout = std::alloc::Layout::from_size_align(SIZE as usize, 4096).unwrap();
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Memory { ptr }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(SIZE as usize, 4096).unwrap();
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

impl GuestMemory for Memory {
    fn read(&self, addr: u64, out: &mut [u8]) -> bool {
        let Some(off) = addr.checked_sub(CPU_BASE) else {
            return false;
        };
        if off + out.len() as u64 > SIZE {
            return false;
        }
        // SAFETY: the range is inside the block.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.add(off as usize), out.as_mut_ptr(), out.len())
        };
        true
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        let Some(off) = addr.checked_sub(CPU_BASE) else {
            return false;
        };
        if off + data.len() as u64 > SIZE {
            return false;
        }
        // SAFETY: the range is inside the block.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(off as usize), data.len())
        };
        true
    }

    fn host_run(&self, addr: u64, max: u64) -> Option<(*mut u8, u64)> {
        let off = addr.checked_sub(CPU_BASE)?;
        if off >= SIZE {
            return None;
        }
        // SAFETY: inside the block.
        Some((unsafe { self.ptr.add(off as usize) }, (SIZE - off).min(max)))
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
            mem: Memory::new(),
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
        // Block linear, a block of 16 GOBs; it does not matter to the renderer unless the memory
        // is read as a texture.
        self.color_target_blocks(j, va, width, height, format, 4);
    }

    /// As [`World::color_target`] with a block of `1 << block_log2` GOBs.
    pub fn color_target_blocks(
        &mut self,
        j: u32,
        va: u64,
        width: u32,
        height: u32,
        format: u32,
        block_log2: u32,
    ) {
        let b = 0x800 + j * 64;
        self.set_address(b, va);
        self.set(b + 8, width);
        self.set(b + 0xC, height);
        self.set(b + 0x10, format);
        self.set(b + 0x14, block_log2 << 4);
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

// ---- programs and the state to run them with -----------------------------------------

/// A vertex program header followed by `code`.
pub fn vertex_program(code: &[u64]) -> Vec<u8> {
    let mut words = [0u32; SPH_WORDS];
    words[0] = 1 | 3 << 5 | 1 << 10;
    program(&words, code)
}

/// A pixel program header with the interpolation `kind` (1 flat, 2 perspective, 3 screen linear)
/// of the listed `(attribute, component)` inputs, target 0 written, and code.
pub fn pixel_program_with(inputs: &[(usize, usize)], kind: u32, code: &[u64]) -> Vec<u8> {
    let mut words = [0u32; SPH_WORDS];
    words[0] = 2 | 3 << 5 | 5 << 10;
    let set = |words: &mut [u32; SPH_WORDS], bit: usize, value: u32| {
        words[bit / 32] |= value << (bit % 32);
    };
    for &(attr, comp) in inputs {
        set(&mut words, 192 + (attr * 4 + comp) * 2, kind);
    }
    for comp in 0..4 {
        set(&mut words, 576 + comp, 1);
    }
    program(&words, code)
}

/// A pixel program whose inputs `flat` are read flat.
pub fn pixel_program(flat: &[(usize, usize)], code: &[u64]) -> Vec<u8> {
    pixel_program_with(flat, 1, code)
}

pub fn program(words: &[u32; SPH_WORDS], code: &[u64]) -> Vec<u8> {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.extend_from_slice(&assemble(code));
    bytes
}

pub fn ald(rd: u8, addr: u32, words: u64) -> u64 {
    Enc::op(Op::Ald)
        .rd(rd)
        .ra(RZ)
        .field(20, 10, u64::from(addr))
        .field(47, 2, words - 1)
        .build()
}

pub fn ast(rd: u8, addr: u32, words: u64) -> u64 {
    Enc::op(Op::Ast)
        .rd(rd)
        .ra(RZ)
        .field(20, 10, u64::from(addr))
        .field(47, 2, words - 1)
        .build()
}

pub fn ipa(rd: u8, addr: u32) -> u64 {
    Enc::op(Op::Ipa)
        .rd(rd)
        .ra(RZ)
        .rb(RZ)
        .field(28, 10, u64::from(addr))
        .build()
}

/// A vertex program that passes the position (location 0) and the color (location 1) through.
pub fn pass_through_vertex() -> Vec<u8> {
    vertex_program(&[
        ald(0, 0x80, 4),
        ald(4, 0x90, 4),
        ast(0, 0x70, 4),
        ast(4, 0x80, 4),
        exit(Guard::ALWAYS),
    ])
}

/// A pixel program that writes the flat color of location 0.
pub fn flat_color_pixel() -> Vec<u8> {
    let flat: Vec<(usize, usize)> = (0..4).map(|c| (0, c)).collect();
    pixel_program(
        &flat,
        &[
            ipa(0, 0x80),
            ipa(1, 0x84),
            ipa(2, 0x88),
            ipa(3, 0x8C),
            exit(Guard::ALWAYS),
        ],
    )
}

pub const REGION: u64 = GPU_BASE + 0x10_0000;
pub const VERTEX_PROGRAM: u64 = 0;
pub const PIXEL_PROGRAM: u64 = 0x1000;

/// A world with the programs placed, a `size` x `size` target at `target` and the pipeline state
/// a title would set: viewport over the target, writes enabled, vertices from stream 0.
pub fn scene(vertex: &[u8], pixel: &[u8], size: u32, target: u64) -> World {
    let mut w = World::new();
    w.put(REGION + VERTEX_PROGRAM, vertex);
    w.put(REGION + PIXEL_PROGRAM, pixel);
    w.set_address(0x1608, REGION);
    // Slot 1 is the vertex program, slot 5 the pixel program.
    w.set(0x2044, VERTEX_PROGRAM as u32);
    w.set(0x2140, 1 | 5 << 4);
    w.set(0x2144, PIXEL_PROGRAM as u32);
    w.set(0x2150, 4);
    w.color_target(0, target, size, size, 0xD5);
    w.set(0x037c, 1);
    w.set(0x1a00, 0x1111);
    // The viewport transform over the whole target; depth 0..1. The register that looks like its
    // enable is 0, as a title's render target macro leaves it (docs/ASSUMPTIONS.md A-112).
    w.set(0x192c, 0);
    let half = size as f32 / 2.0;
    for (i, v) in [half, half, 0.5, half, half, 0.5].iter().enumerate() {
        w.set(0xa00 + 4 * i as u32, v.to_bits());
    }
    w.set(0xc08, 0.0f32.to_bits());
    w.set(0xc0c, 1.0f32.to_bits());
    // Attributes: 0 is the position and 1 the color, both four floats in stream 0.
    w.set(0x1160, 7 << 27 | 1 << 21);
    w.set(0x1164, 16 << 7 | 7 << 27 | 1 << 21);
    w
}

/// Four floats of position and four of color per vertex, as stream 0 at `va`.
pub fn vertices(w: &mut World, va: u64, list: &[([f32; 4], [f32; 4])]) {
    let bytes: Vec<u8> = list
        .iter()
        .flat_map(|(p, c)| p.iter().chain(c.iter()).flat_map(|f| f.to_le_bytes()))
        .collect();
    w.put(va, &bytes);
    w.set(0x1c00, 32 | 1 << 12);
    w.set_address(0x1c04, va);
    w.set_address(0x1f00, va + bytes.len() as u64 - 1);
}
