//! The kernel stage against the GPU: a compute program as a title launches it, with the thread
//! and block numbers from `S2R`, a workgroup size from the configuration, constant buffer 0 as the
//! parameters and memory reached through the table of GPU address ranges.

mod common;

use common::vulkan;
use nx_shader::asm::{assemble, exit, Enc};
use nx_shader::cfg::Code;
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_shader::translate::{translate, Config, Stage};
use nx_vulkan::AddressPatch;

const TABLE: u32 = 64 * 5 + 20;
const PARAMETERS: u32 = 64 * 5;
const START: u64 = 0x1_0000_0000;
const END: u64 = START + 0x1000;

fn s2r(rd: u8, register: u64) -> u64 {
    Enc::op(Op::S2r).rd(rd).field(20, 8, register).build()
}

/// `MOV Rd, c[0][offset]`.
fn mov_cbuf(rd: u8, offset: u32) -> u64 {
    Enc::op(Op::MovCbuf)
        .rd(rd)
        .ra(RZ)
        .cbuf(0, offset)
        .field(39, 4, 0xF)
        .build()
}

/// `STG.E.32`: memory at `(Ra+1 : Ra) + offset` from `rd`.
fn stg(rd: u8, ra: u8, offset: i32) -> u64 {
    Enc::op(Op::Stg)
        .rd(rd)
        .ra(ra)
        .field(20, 24, offset as u32 as u64)
        .bit(45, true)
        .field(48, 3, 4)
        .build()
}

/// The table of one range, `START..END`, whose device address is patched in.
fn table() -> Vec<u8> {
    let mut table = vec![0u8; 256];
    let mut put =
        |word: usize, v: u32| table[word * 4..word * 4 + 4].copy_from_slice(&v.to_le_bytes());
    put(0, 1);
    put(4, START as u32);
    put(5, (START >> 32) as u32);
    put(6, END as u32);
    put(7, (END >> 32) as u32);
    table
}

#[test]
fn each_thread_knows_its_place_in_its_block_and_the_block_its_place_in_the_grid() {
    let Some(vk) = vulkan() else { return };
    // Thread t of block b (8 threads to a block) writes (t, b, 8 b + t, c[0][8]) at
    // base + 16 (8 b + t), where the base is the GPU address in c[0][0..2].
    let program = [
        s2r(0, 0x21),
        s2r(1, 0x25),
        Enc::op(Op::IscaddReg)
            .rd(2)
            .ra(1)
            .rb(0)
            .field(39, 5, 3)
            .build(),
        mov_cbuf(4, 0),
        mov_cbuf(5, 4),
        mov_cbuf(7, 8),
        Enc::op(Op::ShlImm20).rd(6).ra(2).imm20(4).build(),
        Enc::op(Op::IaddReg).rd(4).ra(4).rb(6).build(),
        stg(0, 4, 0),
        stg(1, 4, 4),
        stg(2, 4, 8),
        stg(7, 4, 12),
        exit(Guard::ALWAYS),
    ];
    let code = assemble(&program);
    let mut config = Config::new(Stage::Kernel);
    config.block = [8, 1, 1];
    let t = translate(Code(&code), None, &config).unwrap();
    assert!(t.info.global_memory);
    nx_shader::validate::assert_valid(&t.spirv);

    let mut parameters = [0u8; 16];
    parameters[0..4].copy_from_slice(&(START as u32).to_le_bytes());
    parameters[4..8].copy_from_slice(&((START >> 32) as u32).to_le_bytes());
    parameters[8..12].copy_from_slice(&0xABCD_u32.to_le_bytes());
    let mut table = table();
    let mut memory = vec![0xEEu8; 0x1000];
    let mut scratch = vec![0u8; 4096];
    // Buffer 0 is the parameters, 1 the table, 2 the memory, 3 the scratch page.
    let mut buffers: Vec<(u32, &mut [u8])> = vec![
        (PARAMETERS, &mut parameters[..]),
        (TABLE, &mut table[..]),
        (200, &mut memory[..]),
        (201, &mut scratch[..]),
    ];
    let patches = [
        AddressPatch {
            into: 1,
            offset: 8,
            of: 3,
            addend: 0,
        },
        AddressPatch {
            into: 1,
            offset: 32,
            of: 2,
            addend: 0,
        },
    ];
    vk.run_compute_with(&t.spirv, &mut buffers, &[], &patches, [3, 1, 1])
        .unwrap();
    for block in 0..3u32 {
        for thread in 0..8u32 {
            let at = 16 * (8 * block + thread) as usize;
            let word = |k: usize| {
                u32::from_le_bytes(memory[at + 4 * k..at + 4 * k + 4].try_into().unwrap())
            };
            assert_eq!(
                [word(0), word(1), word(2), word(3)],
                [thread, block, 8 * block + thread, 0xABCD],
                "block {block} thread {thread}"
            );
        }
    }
    // Nothing beyond the 24 threads was touched.
    assert!(memory[16 * 24..].iter().all(|b| *b == 0xEE));
}

#[test]
fn a_kernel_cannot_read_an_attribute() {
    let ald = Enc::op(Op::Ald)
        .rd(0)
        .ra(RZ)
        .field(20, 10, 0x80)
        .field(47, 2, 0)
        .build();
    let code = assemble(&[ald, exit(Guard::ALWAYS)]);
    let e = translate(Code(&code), None, &Config::new(Stage::Kernel))
        .expect_err("a compute program has no vertex attributes")
        .message;
    assert!(e.contains("not a vertex shader"), "{e}");
}
