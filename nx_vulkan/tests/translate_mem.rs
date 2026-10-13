//! Global memory and textures: the translator's `LDG`/`STG` and `TEXS` against the GPU, through
//! the table of memory ranges and a sampled image the host binds.

mod common;

use ash::vk;
use common::{run_with, vulkan, Extras, Rng, State};
use nx_shader::asm::{exit, Enc};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_vulkan::{AddressPatch, SampledImage, SampledKind};

/// `LDG.E`: `rd` from memory at `(Ra+1 : Ra) + offset`; `size` is 0 u8, 1 s8, 2 u16, 3 s16,
/// 4 b32, 5 b64 and 6 b128.
fn ldg(rd: u8, ra: u8, offset: i32, size: u64) -> Enc {
    Enc::op(Op::Ldg)
        .rd(rd)
        .ra(ra)
        .field(20, 24, offset as u32 as u64)
        .bit(45, true)
        .field(48, 3, size)
}

/// `STG.E`: memory at `(Ra+1 : Ra) + offset` from `rd`.
fn stg(rd: u8, ra: u8, offset: i32, size: u64) -> Enc {
    Enc::op(Op::Stg)
        .rd(rd)
        .ra(ra)
        .field(20, 24, offset as u32 as u64)
        .bit(45, true)
        .field(48, 3, size)
}

const TABLE: u32 = 64 * 5 + 20;
const A_START: u64 = 0x1_0000_0000;
const A_END: u64 = A_START + 0x1000;
const B_START: u64 = 0x7_0000_0000;
const B_END: u64 = B_START + 0x400;
/// Where in the second buffer the second range starts.
const B_SKIP: u64 = 0x100;

/// The memory the table describes: two buffers (the second entered at an offset) and a scratch
/// page, with their contents.
struct World {
    mem_a: Vec<u8>,
    mem_b: Vec<u8>,
}

impl World {
    fn new(seed: u64) -> World {
        let mut rng = Rng(seed);
        let mut fill = |n: usize| (0..n / 4).flat_map(|_| rng.next().to_le_bytes()).collect();
        World {
            mem_a: fill(0x1000),
            mem_b: fill(0x800),
        }
    }

    fn extras(&self) -> Extras<'static> {
        let mut table = vec![0u8; 256];
        let mut put =
            |word: usize, v: u32| table[word * 4..word * 4 + 4].copy_from_slice(&v.to_le_bytes());
        put(0, 2);
        // Entry 0 starts at word 4: start, end, device address (the last is patched in).
        put(4, A_START as u32);
        put(5, (A_START >> 32) as u32);
        put(6, A_END as u32);
        put(7, (A_END >> 32) as u32);
        put(10, B_START as u32);
        put(11, (B_START >> 32) as u32);
        put(12, B_END as u32);
        put(13, (B_END >> 32) as u32);
        Extras {
            buffers: vec![
                (TABLE, table),
                (200, self.mem_a.clone()),
                (201, self.mem_b.clone()),
                (202, vec![0u8; 4096]),
            ],
            images: Vec::new(),
            patches: vec![
                // Buffers 0..5 are the harness's own; the table is 5, the memories 6 and 7, the
                // scratch page 8.
                AddressPatch {
                    into: 5,
                    offset: 8,
                    of: 8,
                    addend: 0,
                },
                AddressPatch {
                    into: 5,
                    offset: 32,
                    of: 6,
                    addend: 0,
                },
                AddressPatch {
                    into: 5,
                    offset: 56,
                    of: 7,
                    addend: B_SKIP,
                },
            ],
            ..Extras::default()
        }
    }

    /// The bytes behind a GPU address, if a range covers it.
    fn at(&mut self, gpu: u64) -> Option<&mut [u8]> {
        if (A_START..A_END).contains(&gpu) {
            Some(&mut self.mem_a[(gpu - A_START) as usize..])
        } else if (B_START..B_END).contains(&gpu) {
            Some(&mut self.mem_b[(gpu - B_START + B_SKIP) as usize..])
        } else {
            None
        }
    }

    fn word(&mut self, gpu: u64) -> u32 {
        self.at(gpu)
            .map_or(0, |m| u32::from_le_bytes(m[..4].try_into().unwrap()))
    }

    fn set_word(&mut self, gpu: u64, v: u32) {
        if let Some(m) = self.at(gpu) {
            m[..4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

#[test]
fn global_loads_and_stores_follow_the_table() {
    let Some(vk) = vulkan() else { return };
    // Every invocation has its own 0x100 bytes; R1/R2 hold the address of byte 0x20 of them.
    let program = [
        ldg(3, 1, -0x10, 4).build(),
        ldg(4, 1, 0, 5).build(),
        ldg(6, 1, 0x10, 6).build(),
        stg(10, 1, 0x20, 4).build(),
        stg(11, 1, 0x28, 5).build(),
        stg(10, 1, 0x30, 4).guard(nx_shader::asm::P0).build(),
        ldg(13, 1, -0x1D, 0).build(),
        ldg(14, 1, -0x0E, 3).build(),
        exit(Guard::ALWAYS),
    ];
    let mut world = World::new(5);
    let mut rng = Rng(77);
    let mut bases: Vec<u64> = (0..8).map(|i| A_START + 0x100 * i).collect();
    bases.extend((0..4).map(|i| B_START + 0x100 * i));
    let inputs: Vec<State> = bases
        .iter()
        .map(|b| {
            let mut s = State::default();
            let a = b + 0x20;
            s.regs[1] = a as u32;
            s.regs[2] = (a >> 32) as u32;
            for r in [10, 11, 12] {
                s.regs[r] = rng.next();
            }
            s.preds = rng.next() & 1;
            s
        })
        .collect();
    let out = run_with(&vk, &program, &inputs, &[], world.extras());
    for (i, (base, (inp, got))) in bases.iter().zip(inputs.iter().zip(&out.states)).enumerate() {
        let w = |world: &mut World, off: u64| world.word(base + off);
        assert_eq!(got.regs[3], w(&mut world, 0x10), "b32 load, invocation {i}");
        assert_eq!(
            got.regs[4],
            w(&mut world, 0x20),
            "b64 load, low word, invocation {i}"
        );
        assert_eq!(
            got.regs[5],
            w(&mut world, 0x24),
            "b64 load, high word, invocation {i}"
        );
        for k in 0..4 {
            assert_eq!(
                got.regs[6 + k],
                w(&mut world, 0x30 + 4 * k as u64),
                "b128 load, word {k}, invocation {i}"
            );
        }
        let byte = (w(&mut world, 0) >> 24) as u8;
        assert_eq!(got.regs[13], u32::from(byte), "u8 load, invocation {i}");
        let half = (w(&mut world, 0x10) >> 16) as u16 as i16;
        assert_eq!(got.regs[14], half as i32 as u32, "s16 load, invocation {i}");
        // The stores, in the model.
        world.set_word(base + 0x40, inp.regs[10]);
        world.set_word(base + 0x48, inp.regs[11]);
        world.set_word(base + 0x4C, inp.regs[12]);
        if inp.preds & 1 != 0 {
            world.set_word(base + 0x50, inp.regs[10]);
        }
    }
    assert_eq!(
        out.buffers[1], world.mem_a,
        "the first range after the stores"
    );
    assert_eq!(
        out.buffers[2], world.mem_b,
        "the second range after the stores"
    );
}

#[test]
fn addresses_outside_the_ranges_read_the_zeroed_scratch_page() {
    let Some(vk) = vulkan() else { return };
    let program = [ldg(3, 1, 0, 4).build(), exit(Guard::ALWAYS)];
    let mut world = World::new(9);
    let probes: [(u64, bool); 8] = [
        (A_START, true),
        (A_START + 4, true),
        (A_END - 4, true),
        (A_END, false),
        (A_START - 4, false),
        (B_END - 4, true),
        (B_END, false),
        (0x9_0000_0000, false),
    ];
    let inputs: Vec<State> = probes
        .iter()
        .map(|(a, _)| {
            let mut s = State::default();
            s.regs[1] = *a as u32;
            s.regs[2] = (*a >> 32) as u32;
            s.regs[3] = 0xDEAD_BEEF;
            s
        })
        .collect();
    let out = run_with(&vk, &program, &inputs, &[], world.extras());
    for ((addr, mapped), got) in probes.iter().zip(&out.states) {
        let expected = if *mapped { world.word(*addr) } else { 0 };
        assert_eq!(got.regs[3], expected, "load at {addr:#x}");
    }
    assert_eq!(out.buffers[1], world.mem_a, "loads change nothing");
}

#[test]
fn a_store_outside_the_ranges_lands_in_the_scratch_page() {
    let Some(vk) = vulkan() else { return };
    let program = [stg(4, 1, 0, 4).build(), exit(Guard::ALWAYS)];
    let mut world = World::new(11);
    let mut s = State::default();
    s.regs[1] = 0x8000_0000;
    s.regs[2] = 0x2;
    s.regs[4] = 0x1234_5678;
    let out = run_with(&vk, &program, &[s], &[], world.extras());
    assert_eq!(out.buffers[1], world.mem_a, "no mapped memory changed");
    assert_eq!(out.buffers[2], world.mem_b, "no mapped memory changed");
    assert_eq!(
        u32::from_le_bytes(out.buffers[3][..4].try_into().unwrap()),
        0x1234_5678,
        "the scratch page took the store"
    );
    let _ = world.word(0);
}

/// The texel `(x, y)` of the plain test texture, as four floats.
fn texel(x: u32, y: u32) -> [f32; 4] {
    [
        x as f32 + 1.0,
        y as f32 + 10.0,
        (x * 4 + y) as f32,
        0.5 + x as f32 / 8.0,
    ]
}

fn texel_bytes(texels: &[[f32; 4]]) -> Vec<u8> {
    texels
        .iter()
        .flat_map(|t| t.iter().flat_map(|c| c.to_le_bytes()))
        .collect()
}

/// `TEXS`: `rd`/`r28` receive the components `mask` selects from texture `index`, sampled with
/// kind `kind` at the coordinates in `ra` and `rb`; 32-bit results.
fn texs(rd: u8, ra: u8, rb: u8, r28: u8, index: u64, mask: u64, kind: u64) -> Enc {
    Enc::op(Op::Texs)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .field(28, 8, u64::from(r28))
        .field(36, 13, index)
        .field(50, 3, mask)
        .field(53, 4, kind)
        .bit(59, true)
}

#[test]
fn texs_samples_with_nearest_filtering_and_scatters_the_components() {
    let Some(vk) = vulkan() else { return };
    // The first texture the program names (5, here) is binding 64*5 + 24 of the compute stage.
    // R1/R2 are the coordinates.
    let binding = 64 * 5 + 24;
    let texels: Vec<[f32; 4]> = (0..4)
        .flat_map(|y| (0..4).map(move |x| texel(x, y)))
        .collect();
    let bytes = texel_bytes(&texels);
    let program = [
        // Kind 2: 2D, level of detail zero. Four components: R4..R7.
        texs(4, 1, 2, 6, 5, 4, 2).build(),
        // One destination: green and alpha (mask 6) to R8, R9.
        texs(8, 1, 2, RZ, 5, 6, 2).build(),
        // Alpha alone (mask 3) to R10.
        texs(10, 1, 2, RZ, 5, 3, 2).build(),
        // Kind 3: x and y in R1 and R2, the level in R3 (zero); red and green (mask 4) to R11,
        // R12.
        texs(11, 1, 3, RZ, 5, 4, 3).build(),
        exit(Guard::ALWAYS),
    ];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for y in 0..4u32 {
        for x in 0..4u32 {
            let mut s = State::default();
            s.regs[1] = ((x as f32 + 0.5) / 4.0).to_bits();
            s.regs[2] = ((y as f32 + 0.5) / 4.0).to_bits();
            s.regs[3] = 0;
            inputs.push(s);
            expected.push(texel(x, y));
        }
    }
    // Outside the texture the edge texel is used.
    for (u, v, x, y) in [
        (-0.3f32, 0.1f32, 0u32, 0u32),
        (1.7, 0.9, 3, 3),
        (0.6, -2.0, 2, 0),
    ] {
        let mut s = State::default();
        s.regs[1] = u.to_bits();
        s.regs[2] = v.to_bits();
        inputs.push(s);
        expected.push(texel(x, y));
    }
    let extras = Extras {
        images: vec![SampledImage {
            binding,
            format: vk::Format::R32G32B32A32_SFLOAT,
            width: 4,
            height: 4,
            depth: 1,
            layers: 1,
            kind: SampledKind::D2,
            compare: false,
            texels: &bytes,
        }],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        let f = |r: usize| f32::from_bits(got.regs[r]);
        assert_eq!(
            [f(4), f(5), f(6), f(7)],
            *want,
            "all four components, sample {i}"
        );
        assert_eq!(
            [f(8), f(9)],
            [want[1], want[3]],
            "green and alpha, sample {i}"
        );
        assert_eq!(f(10), want[3], "alpha, sample {i}");
        assert_eq!(
            [f(11), f(12)],
            [want[0], want[1]],
            "red and green with a level, sample {i}"
        );
    }
}

#[test]
fn texs_of_an_array_takes_the_layer_from_the_first_register() {
    let Some(vk) = vulkan() else { return };
    let binding = 64 * 5 + 24;
    // Three layers of 2x2; each texel says where it is.
    let layer_texel = |l: u32, x: u32, y: u32| {
        [
            l as f32,
            x as f32,
            y as f32,
            7.0 + (l * 4 + y * 2 + x) as f32,
        ]
    };
    let texels: Vec<[f32; 4]> = (0..3)
        .flat_map(|l| (0..2).flat_map(move |y| (0..2).map(move |x| layer_texel(l, x, y))))
        .collect();
    let bytes = texel_bytes(&texels);
    // Kind 8: 2D array, level zero. The layer is the integer in R1, x in R2 (the register after
    // it), y in R3.
    let program = [texs(4, 1, 3, 6, 2, 4, 8).build(), exit(Guard::ALWAYS)];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for l in 0..3u32 {
        for y in 0..2u32 {
            for x in 0..2u32 {
                let mut s = State::default();
                s.regs[1] = l;
                s.regs[2] = ((x as f32 + 0.5) / 2.0).to_bits();
                s.regs[3] = ((y as f32 + 0.5) / 2.0).to_bits();
                inputs.push(s);
                expected.push(layer_texel(l, x, y));
            }
        }
    }
    let extras = Extras {
        images: vec![SampledImage {
            binding,
            format: vk::Format::R32G32B32A32_SFLOAT,
            width: 2,
            height: 2,
            depth: 1,
            layers: 3,
            kind: SampledKind::D2Array,
            compare: false,
            texels: &bytes,
        }],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        let f = |r: usize| f32::from_bits(got.regs[r]);
        assert_eq!([f(4), f(5), f(6), f(7)], *want, "sample {i}");
    }
}

#[test]
fn a_table_of_many_ranges_is_searched_for_the_one_that_holds_an_address() {
    let Some(vk) = vulkan() else { return };
    // Seven ranges of 0x400 bytes at addresses 0x1000 apart (in order of their start, as the
    // renderer's table is), the i-th at offset 0x400 * i of one device memory of 0x1c00 bytes.
    const START: u64 = 0x3_0000_0000;
    const RANGES: u64 = 7;
    let mut rng = Rng(1234);
    let memory: Vec<u8> = (0..0x1c00 / 4)
        .flat_map(|_| rng.next().to_le_bytes())
        .collect();
    let mut table = vec![0u8; 4 * (4 + 6 * RANGES as usize)];
    let mut put =
        |word: usize, v: u32| table[word * 4..word * 4 + 4].copy_from_slice(&v.to_le_bytes());
    put(0, RANGES as u32);
    let mut patches = vec![AddressPatch {
        into: 5,
        offset: 8,
        of: 7,
        addend: 0,
    }];
    for i in 0..RANGES {
        let (start, end) = (START + 0x1000 * i, START + 0x1000 * i + 0x400);
        let at = 4 + 6 * i as usize;
        put(at, start as u32);
        put(at + 1, (start >> 32) as u32);
        put(at + 2, end as u32);
        put(at + 3, (end >> 32) as u32);
        patches.push(AddressPatch {
            into: 5,
            offset: 4 * (at + 4),
            of: 6,
            addend: 0x400 * i,
        });
    }
    // The address each invocation loads from: the edges of every range, the gaps, and below and
    // above all of them.
    let mut probes: Vec<u64> = vec![0, 4, START - 4, START + 0x1000 * RANGES, u64::MAX - 3];
    for i in 0..RANGES {
        let s = START + 0x1000 * i;
        probes.extend([s - 4, s, s + 4, s + 0x3F8, s + 0x3FC, s + 0x400, s + 0x800]);
    }
    let model = |gpu: u64| -> u32 {
        for i in 0..RANGES {
            let s = START + 0x1000 * i;
            if (s..s + 0x400).contains(&gpu) {
                let at = (0x400 * i + gpu - s) as usize;
                return u32::from_le_bytes(memory[at..at + 4].try_into().unwrap());
            }
        }
        0
    };
    let program = [ldg(3, 1, 0, 4).build(), exit(Guard::ALWAYS)];
    let inputs: Vec<State> = probes
        .iter()
        .map(|a| {
            let mut s = State::default();
            s.regs[1] = *a as u32;
            s.regs[2] = (*a >> 32) as u32;
            s
        })
        .collect();
    let extras = Extras {
        buffers: vec![
            (TABLE, table),
            (200, memory.clone()),
            (201, vec![0u8; 4096]),
        ],
        patches,
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (probe, got) in probes.iter().zip(&out.states) {
        assert_eq!(got.regs[3], model(*probe), "a load from {probe:#x}");
    }
}
