//! Global memory through the whole path: a vertex program that reads its vertices (and writes
//! results) by GPU address, with the guest's memory imported into the device in place.

mod render_common;

use nx_gpu::mem::GuestMemory;
use nx_gpu::render::{Backend, Draw, Source};
use nx_shader::asm::{exit, mov32i, Enc};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_vulkan::{RenderConfig, Renderer};
use render_common::*;

fn ldg(rd: u8, ra: u8, offset: i32, size: u64) -> u64 {
    Enc::op(Op::Ldg)
        .rd(rd)
        .ra(ra)
        .field(20, 24, offset as u32 as u64)
        .bit(45, true)
        .field(48, 3, size)
        .build()
}

fn stg(rd: u8, ra: u8, offset: i32, size: u64) -> u64 {
    Enc::op(Op::Stg)
        .rd(rd)
        .ra(ra)
        .field(20, 24, offset as u32 as u64)
        .bit(45, true)
        .field(48, 3, size)
        .build()
}

/// `LDC.64 rd, c[0][offset]`.
fn ldc64(rd: u8, offset: i32) -> u64 {
    Enc::op(Op::Ldc)
        .rd(rd)
        .ra(RZ)
        .field(20, 16, offset as u32 as u64)
        .field(36, 5, 0)
        .field(48, 3, 5)
        .build()
}

/// R8:R9 = the 64-bit address in `c[0][0]` plus `R10 * 2^shift` (R10 is the vertex id).
fn address_of_vertex(shift: i32) -> Vec<u64> {
    vec![
        ald(10, 0x2FC, 1),
        ldc64(8, 0),
        Enc::op(Op::ShlImm20).rd(11).ra(10).imm20(shift).build(),
        Enc::op(Op::IaddReg)
            .rd(8)
            .ra(8)
            .rb(11)
            .bit(47, true)
            .build(),
        Enc::op(Op::IaddReg)
            .rd(9)
            .ra(9)
            .rb(RZ)
            .bit(43, true)
            .build(),
    ]
}

const BASE: u64 = GPU_BASE + 0x30_0000;
const OUT: u64 = GPU_BASE + 0x31_0000;
const CBUF: u64 = GPU_BASE + 0x32_0000;

/// A vertex program that pulls position and color from memory at `c[0][0]`, 32 bytes a vertex.
fn pulling_vertex() -> Vec<u8> {
    let mut code = address_of_vertex(5);
    code.extend([
        ldg(0, 8, 0, 6),
        ldg(4, 8, 16, 6),
        ast(0, 0x70, 4),
        ast(4, 0x80, 4),
        exit(Guard::ALWAYS),
    ]);
    vertex_program(&code)
}

fn bind_base(w: &mut World, address: u64) {
    let mut bytes = address.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0u8; 8]);
    w.put(CBUF, &bytes);
    w.cbufs.bind(0, 1, (CBUF, 16));
}

const DRAW: Draw = Draw {
    source: Source::Arrays { first: 0, count: 3 },
    topology: 4,
    instance: 0,
    instances: 1,
};

#[test]
fn a_vertex_program_reads_its_vertices_from_guest_memory_by_address() {
    let Some(vk) = vulkan() else { return };
    if !vk.caps.external_memory_host || !vk.caps.buffer_device_address {
        eprintln!("SKIPPED: the device cannot import host memory");
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    let mut w = scene(&pulling_vertex(), &flat_color_pixel(), 32, target);
    // No vertex attributes at all: the program fetches its own.
    w.set(0x1160, 0);
    w.set(0x1164, 0);
    let magenta = [1.0f32, 0.0, 1.0, 1.0];
    let list = [
        [-1.0f32, -1.0, 0.5, 1.0],
        [3.0, -1.0, 0.5, 1.0],
        [-1.0, 3.0, 0.5, 1.0],
    ];
    let bytes: Vec<u8> = list
        .iter()
        .flat_map(|p| p.iter().chain(magenta.iter()).flat_map(|f| f.to_le_bytes()))
        .collect();
    w.put(BASE, &bytes);
    bind_base(&mut w, BASE);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(&w.scene(), &DRAW).unwrap();
    let (rgba, _, _) = r.picture_at(w.cpu(target)).unwrap().expect("a target");
    for px in rgba.as_chunks::<4>().0 {
        assert_eq!(px, &[255, 0, 255, 255]);
    }
}

#[test]
fn a_program_that_stores_to_global_memory_changes_the_guests_memory() {
    let Some(vk) = vulkan() else { return };
    if !vk.caps.external_memory_host || !vk.caps.buffer_device_address {
        eprintln!("SKIPPED: the device cannot import host memory");
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    // Each vertex stores 100 + its id at OUT + 4 * id, then outputs a degenerate position.
    let mut code = address_of_vertex(2);
    code.extend([
        Enc::op(Op::Iadd32iImm32).rd(12).ra(10).imm32(100).build(),
        stg(12, 8, 0, 4),
        mov32i(0, 0),
        mov32i(1, 0),
        mov32i(2, 0),
        mov32i(3, 0x3F80_0000),
        mov32i(4, 0),
        mov32i(5, 0),
        mov32i(6, 0),
        mov32i(7, 0x3F80_0000),
        ast(0, 0x70, 4),
        ast(4, 0x80, 4),
        exit(Guard::ALWAYS),
    ]);
    let mut w = scene(&vertex_program(&code), &flat_color_pixel(), 8, target);
    w.set(0x1160, 0);
    w.set(0x1164, 0);
    bind_base(&mut w, OUT);
    let mut r = Renderer::new(RenderConfig::default());
    r.draw(&w.scene(), &DRAW).unwrap();
    r.finish().unwrap();
    for id in 0..3u64 {
        let mut word = [0u8; 4];
        assert!(w.mem.read(w.cpu(OUT) + 4 * id, &mut word));
        assert_eq!(u32::from_le_bytes(word), 100 + id as u32, "word {id}");
    }
}

#[test]
fn a_vertex_program_sees_the_instance_counter_of_its_draw() {
    let Some(vk) = vulkan() else { return };
    if !vk.caps.external_memory_host || !vk.caps.buffer_device_address {
        eprintln!("SKIPPED: the device cannot import host memory");
        return;
    }
    let target = GPU_BASE + 0x20_0000;
    // Each draw stores 1000 + its instance id at OUT + 4 * instance id, so the program can only
    // get the address right if it sees the counter of the draw.
    let mut code = vec![
        ald(10, 0x2F8, 1),
        ldc64(8, 0),
        Enc::op(Op::ShlImm20).rd(11).ra(10).imm20(2).build(),
        Enc::op(Op::IaddReg)
            .rd(8)
            .ra(8)
            .rb(11)
            .bit(47, true)
            .build(),
        Enc::op(Op::IaddReg)
            .rd(9)
            .ra(9)
            .rb(RZ)
            .bit(43, true)
            .build(),
        Enc::op(Op::Iadd32iImm32).rd(12).ra(10).imm32(1000).build(),
        stg(12, 8, 0, 4),
    ];
    code.extend([
        mov32i(0, 0),
        mov32i(1, 0),
        mov32i(2, 0),
        mov32i(3, 0x3F80_0000),
        mov32i(4, 0),
        mov32i(5, 0),
        mov32i(6, 0),
        mov32i(7, 0x3F80_0000),
        ast(0, 0x70, 4),
        ast(4, 0x80, 4),
        exit(Guard::ALWAYS),
    ]);
    let mut w = scene(&vertex_program(&code), &flat_color_pixel(), 8, target);
    w.set(0x1160, 0);
    w.set(0x1164, 0);
    bind_base(&mut w, OUT);
    let mut r = Renderer::new(RenderConfig::default());
    for instance in [0u32, 3, 1, 7] {
        r.draw(
            &w.scene(),
            &Draw {
                source: Source::Arrays { first: 0, count: 3 },
                topology: 4,
                instance,
                instances: 1,
            },
        )
        .unwrap();
    }
    r.finish().unwrap();
    for instance in [0u64, 1, 3, 7] {
        let mut word = [0u8; 4];
        assert!(w.mem.read(w.cpu(OUT) + 4 * instance, &mut word));
        assert_eq!(
            u32::from_le_bytes(word),
            1000 + instance as u32,
            "instance {instance}"
        );
    }
    let mut word = [0u8; 4];
    assert!(w.mem.read(w.cpu(OUT) + 8, &mut word));
    assert_eq!(u32::from_le_bytes(word), 0, "instance 2 was never drawn");
}

/// The word of guest memory at GPU address `va`.
fn word_at(w: &World, va: u64) -> u32 {
    let mut word = [0u8; 4];
    assert!(w.mem.read(w.cpu(va), &mut word));
    u32::from_le_bytes(word)
}

/// A launch of `code` (Maxwell code at `at`, one invocation) whose constant buffer 0 is the 16
/// bytes at `cbuf`.
fn launch(w: &mut World, at: u64, code: &[u64], cbuf: u64) -> nx_gpu::compute::Dispatch {
    w.put(at, &nx_shader::asm::assemble(code));
    let mut cbufs = [None; 8];
    cbufs[0] = Some(nx_gpu::compute::CbufBinding {
        address: cbuf,
        size: 16,
    });
    nx_gpu::compute::Dispatch {
        qmd: 0,
        program: at,
        grid: [1, 1, 1],
        block: [1, 1, 1],
        shared_bytes: 0,
        cbufs,
        sm86: false,
    }
}

#[test]
fn a_launch_is_waited_for_when_the_gpu_front_needs_what_it_wrote() {
    let Some(vk) = vulkan() else { return };
    if !vk.caps.external_memory_host || !vk.caps.buffer_device_address {
        eprintln!("SKIPPED: the device cannot import host memory");
        return;
    }
    let mut w = World::new();
    bind_base(&mut w, OUT);
    // The program stores a word at the address in c[0][0].
    let code = [
        ldc64(8, 0),
        mov32i(12, 0x1234_5678),
        stg(12, 8, 0, 4),
        exit(Guard::ALWAYS),
    ];
    let d = launch(&mut w, GPU_BASE + 0x33_0000, &code, CBUF);
    let mut r = Renderer::new(RenderConfig::default());
    r.dispatch(&w.scene(), &d).unwrap();
    // The release that follows the launch is made once the launch is waited for (A-274).
    let flag = OUT + 0x100;
    let release = nx_gpu::qmd3::Release {
        va: flag,
        op: nx_gpu::qmd3::ReleaseOp::Set,
        payload: 7,
    };
    r.release(&w.space, &w.mem, release).unwrap();
    assert_eq!(word_at(&w, flag), 0, "released before the launch had run");
    // So is a semaphore release of the GPU front.
    let semaphore = OUT + 0x200;
    r.write_after(&w.space, &w.mem, semaphore, &9u32.to_le_bytes())
        .unwrap();
    assert_eq!(word_at(&w, semaphore), 0, "a semaphore released too early");
    // An acquire may not compare the word as the releases will leave it: this launch's accesses
    // are not attributed to buffers, so it may write any word (A-274).
    assert_eq!(r.released_value(&w.space, &w.mem, flag), None);
    assert_eq!(
        r.launch_waits(),
        [0; 5],
        "the launch was waited for at once"
    );
    r.settle(&w.space, &w.mem, None).unwrap();
    assert_eq!(
        r.launch_waits(),
        [1, 0, 0, 0, 0],
        "one wait, for the GPU front"
    );
    assert_eq!(word_at(&w, flag), 7);
    assert_eq!(word_at(&w, semaphore), 9);
    assert_eq!(word_at(&w, OUT), 0x1234_5678);
}

#[test]
fn a_launch_that_reads_what_the_one_before_it_wrote_waits_for_it() {
    let Some(vk) = vulkan() else { return };
    if !vk.caps.external_memory_host || !vk.caps.buffer_device_address {
        eprintln!("SKIPPED: the device cannot import host memory");
        return;
    }
    let mut w = World::new();
    bind_base(&mut w, OUT);
    let first = [
        ldc64(8, 0),
        mov32i(12, 0xCAFE_F00D),
        stg(12, 8, 0, 4),
        exit(Guard::ALWAYS),
    ];
    // The second reads its constant buffer 0, which is that word, and stores it 8 bytes on.
    let to = OUT + 8;
    let second = [
        ldc64(12, 0),
        mov32i(8, to as u32),
        mov32i(9, (to >> 32) as u32),
        stg(12, 8, 0, 4),
        exit(Guard::ALWAYS),
    ];
    let a = launch(&mut w, GPU_BASE + 0x33_0000, &first, CBUF);
    let b = launch(&mut w, GPU_BASE + 0x34_0000, &second, OUT);
    let mut r = Renderer::new(RenderConfig::default());
    r.dispatch(&w.scene(), &a).unwrap();
    r.dispatch(&w.scene(), &b).unwrap();
    // The device may have run the first launch before the second was recorded; the wait is
    // what makes sure it has.
    assert_eq!(
        r.launch_waits()[2],
        1,
        "the second launch did not wait for the first"
    );
    r.settle(&w.space, &w.mem, None).unwrap();
    assert_eq!(word_at(&w, OUT), 0xCAFE_F00D);
    assert_eq!(
        word_at(&w, to),
        0xCAFE_F00D,
        "the second launch's constant buffer was copied before the first had run"
    );
}
