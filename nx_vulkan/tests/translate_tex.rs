//! The texture instructions of the translator against the GPU: the kinds of `TEXS`, `TEX`,
//! `TLD4S`, `TLD4` and `TXQ`, with cube maps, volumes, depth comparisons and packed halves. The
//! register layouts are the ones the encodings have (docs/ASSUMPTIONS.md A-106).

mod common;

use ash::vk;
use common::{run_with, vulkan, Extras, State};
use nx_shader::asm::{exit, Enc};
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_vulkan::{SampledImage, SampledKind};

const TEXTURE_INDEX: u64 = 5;
/// The first texture a program names is binding `64 * 5 + 24` of the compute stage.
const BINDING: u32 = 64 * 5 + 24;

/// `TEXS`: `rd`/`r28` receive the components `mask` selects from texture 5, sampled with kind
/// `kind`; `wide` is bit 59 (32-bit results; clear is packed halves).
fn texs(rd: u8, ra: u8, rb: u8, r28: u8, mask: u64, kind: u64, wide: bool) -> Enc {
    Enc::op(Op::Texs)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .field(28, 8, u64::from(r28))
        .field(36, 13, TEXTURE_INDEX)
        .field(50, 3, mask)
        .field(53, 4, kind)
        .bit(59, wide)
}

/// `TEX`: the components of `mask` go to consecutive registers from `rd`. `dim` is bits 28..30
/// (2 2D, 3 2D array, 4 3D, 6 cube), `lod` bits 55..57 (0 implicit, 1 zero, 2 bias, 3 level).
fn tex(rd: u8, ra: u8, rb: u8, dim: u64, mask: u64, lod: u64, dc: bool) -> Enc {
    Enc::op(Op::TexBound)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .field(28, 3, dim)
        .field(31, 4, mask)
        .field(36, 13, TEXTURE_INDEX)
        .bit(50, dc)
        .field(55, 3, lod)
}

fn floats(texels: &[[f32; 4]]) -> Vec<u8> {
    texels
        .iter()
        .flat_map(|t| t.iter().flat_map(|c| c.to_le_bytes()))
        .collect()
}

fn image(
    kind: SampledKind,
    format: vk::Format,
    size: (u32, u32, u32),
    layers: u32,
    compare: bool,
    texels: &[u8],
) -> SampledImage<'_> {
    SampledImage {
        binding: BINDING,
        format,
        width: size.0,
        height: size.1,
        depth: size.2,
        layers,
        kind,
        compare,
        texels,
    }
}

fn regs(values: &[(usize, f32)]) -> State {
    let mut s = State::default();
    for (r, v) in values {
        s.regs[*r] = v.to_bits();
    }
    s
}

fn f(s: &State, r: usize) -> f32 {
    f32::from_bits(s.regs[r])
}

/// A cube map's face, column and row of the texel a direction reaches (Vulkan's cube mapping).
fn cube_texel(d: [f32; 3], size: u32) -> (u32, u32, u32) {
    let (x, y, z) = (d[0], d[1], d[2]);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if x > 0.0 {
            (0, -z, -y, ax)
        } else {
            (1, z, -y, ax)
        }
    } else if ay >= az {
        if y > 0.0 {
            (2, x, z, ay)
        } else {
            (3, x, -z, ay)
        }
    } else if z > 0.0 {
        (4, x, -y, az)
    } else {
        (5, -x, -y, az)
    };
    let to_texel = |c: f32| (((c / ma + 1.0) * 0.5 * size as f32) as u32).min(size - 1);
    (face, to_texel(sc), to_texel(tc))
}

#[test]
fn texs_of_a_cube_picks_the_face_by_the_major_axis() {
    let Some(vk) = vulkan() else { return };
    // Six faces of 2x2; a texel says which face, column and row it is.
    let texels: Vec<[f32; 4]> = (0..6u32)
        .flat_map(|face| {
            (0..2u32).flat_map(move |y| {
                (0..2u32).map(move |x| [face as f32, x as f32, y as f32, 9.0 + face as f32])
            })
        })
        .collect();
    let bytes = floats(&texels);
    // Kind 12: the direction in R1, R2 (`Ra`, `Ra+1`) and R3 (`Rb`). Kind 13 adds the level of
    // detail in the register after `Rb` (R4, zero).
    let program = [
        texs(8, 1, 3, 10, 4, 12, true).build(),
        texs(5, 1, 3, RZ, 4, 13, true).build(),
        exit(Guard::ALWAYS),
    ];
    let directions: Vec<[f32; 3]> = vec![
        [1.0, 0.25, -0.5],
        [-1.0, 0.25, 0.5],
        [0.25, 1.0, 0.5],
        [-0.25, -1.0, 0.5],
        [0.5, -0.25, 1.0],
        [-0.5, 0.25, -1.0],
        [3.0, -1.0, 1.0],
        [-0.3, 2.0, -1.5],
    ];
    let inputs: Vec<State> = directions
        .iter()
        .map(|d| regs(&[(1, d[0]), (2, d[1]), (3, d[2]), (4, 0.0)]))
        .collect();
    let extras = Extras {
        images: vec![image(
            SampledKind::Cube,
            vk::Format::R32G32B32A32_SFLOAT,
            (2, 2, 1),
            6,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (d, got) in directions.iter().zip(&out.states) {
        let (face, x, y) = cube_texel(*d, 2);
        let want = [face as f32, x as f32, y as f32, 9.0 + face as f32];
        assert_eq!(
            [f(got, 8), f(got, 9), f(got, 10), f(got, 11)],
            want,
            "kind 12, direction {d:?}"
        );
        assert_eq!(f(got, 5), want[0], "kind 13, direction {d:?}");
    }
}

#[test]
fn texs_of_a_volume_reads_three_coordinates() {
    let Some(vk) = vulkan() else { return };
    // 2x2x2: a texel says where it is.
    let texel = |x: u32, y: u32, z: u32| {
        [
            x as f32,
            y as f32,
            z as f32,
            100.0 + (x + 2 * y + 4 * z) as f32,
        ]
    };
    let texels: Vec<[f32; 4]> = (0..2)
        .flat_map(|z| (0..2).flat_map(move |y| (0..2).map(move |x| texel(x, y, z))))
        .collect();
    let bytes = floats(&texels);
    // Kinds 10 and 11: x, y in R1, R2 (`Ra`, `Ra+1`), z in R3 (`Rb`).
    let program = [
        texs(8, 1, 3, 10, 4, 10, true).build(),
        texs(5, 1, 3, RZ, 3, 11, true).build(),
        exit(Guard::ALWAYS),
    ];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for z in 0..2u32 {
        for y in 0..2u32 {
            for x in 0..2u32 {
                inputs.push(regs(&[
                    (1, (x as f32 + 0.5) / 2.0),
                    (2, (y as f32 + 0.5) / 2.0),
                    (3, (z as f32 + 0.5) / 2.0),
                ]));
                expected.push(texel(x, y, z));
            }
        }
    }
    let extras = Extras {
        images: vec![image(
            SampledKind::D3,
            vk::Format::R32G32B32A32_SFLOAT,
            (2, 2, 2),
            1,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        assert_eq!(
            [f(got, 8), f(got, 9), f(got, 10), f(got, 11)],
            *want,
            "sample {i}"
        );
        assert_eq!(f(got, 5), want[3], "alpha alone, sample {i}");
    }
}

#[test]
fn texs_with_a_depth_comparison_gives_the_result_in_every_component() {
    let Some(vk) = vulkan() else { return };
    // 2x2 depths, compared "reference <= depth".
    let depths: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    let bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
    let reference = 0.6f32;
    // Kind 4 (implicit level): x, y in R1, R2 (`Ra`, `Ra+1`), the reference in R3 (`Rb`).
    // Kind 5 (a level): the level in R3 (`Rb`), the reference in R4. Kind 6 (level zero): the
    // reference in R3.
    let program = [
        texs(8, 1, 3, 10, 4, 4, true).build(),
        texs(5, 1, 3, RZ, 3, 5, true).build(),
        texs(6, 1, 3, RZ, 3, 6, true).build(),
        exit(Guard::ALWAYS),
    ];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for y in 0..2usize {
        for x in 0..2usize {
            // Kinds 4 and 6 read the reference from R3; kind 5 reads its level from R3 and its
            // reference from R4. Both hold the reference (a level of 0.6 is clamped to the only
            // level there is).
            inputs.push(regs(&[
                (1, (x as f32 + 0.5) / 2.0),
                (2, (y as f32 + 0.5) / 2.0),
                (3, reference),
                (4, reference),
            ]));
            expected.push(if reference <= depths[y * 2 + x] {
                1.0f32
            } else {
                0.0
            });
        }
    }
    let extras = Extras {
        images: vec![image(
            SampledKind::D2,
            vk::Format::D32_SFLOAT,
            (2, 2, 1),
            1,
            true,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        // The same result in all four components (kind 4: R8..R11), or in the one asked for.
        assert_eq!(
            [f(got, 8), f(got, 9), f(got, 10), f(got, 11)],
            [*want; 4],
            "kind 4, sample {i}"
        );
        assert_eq!(f(got, 5), *want, "kind 5, sample {i}");
        assert_eq!(f(got, 6), *want, "kind 6, sample {i}");
    }
}

#[test]
fn texs_of_an_array_with_a_depth_comparison_takes_the_layer_first() {
    let Some(vk) = vulkan() else { return };
    // Two layers of 1x1 depth: 0.3 and 0.8.
    let depths: [f32; 2] = [0.3, 0.8];
    let bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
    // Kind 9: the layer (an integer) in R1 (`Ra`), x in R2, y in R3 (`Rb`), the reference in R4.
    let program = [texs(8, 1, 3, RZ, 4, 9, true).build(), exit(Guard::ALWAYS)];
    let mut inputs = Vec::new();
    for layer in 0..2u32 {
        for reference in [0.2f32, 0.5, 0.9] {
            let mut s = regs(&[(2, 0.5), (3, 0.5), (4, reference)]);
            s.regs[1] = layer;
            inputs.push((layer, reference, s));
        }
    }
    let states: Vec<State> = inputs.iter().map(|(_, _, s)| *s).collect();
    let extras = Extras {
        images: vec![image(
            SampledKind::D2Array,
            vk::Format::D32_SFLOAT,
            (1, 1, 1),
            2,
            true,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &states, &[], extras);
    for ((layer, reference, _), got) in inputs.iter().zip(&out.states) {
        let want = if *reference <= depths[*layer as usize] {
            1.0
        } else {
            0.0
        };
        assert_eq!(f(got, 8), want, "layer {layer}, reference {reference}");
    }
}

/// A half of `v`, exact for the values the tests use.
fn half_bits(v: f32) -> u32 {
    let b = v.to_bits();
    let sign = (b >> 16) & 0x8000;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (b >> 13) & 0x3FF;
    assert!((1..31).contains(&exp), "{v} is not a plain half");
    sign | (exp as u32) << 10 | frac
}

#[test]
fn texs_without_bit_59_packs_halves_two_to_a_register() {
    let Some(vk) = vulkan() else { return };
    let texel = [1.0f32, 2.5, -3.0, 0.5];
    let bytes = floats(&[texel]);
    let program = [
        // All four components (selector 4) to two destinations: R8 = (x, y), R9 = (z, w).
        texs(8, 1, 2, 9, 4, 2, false).build(),
        // Two components, one destination: green and alpha (selector 6) to R5.
        texs(5, 1, 2, RZ, 6, 2, false).build(),
        // One component, packed with a zero: blue (selector 2) to R6.
        texs(6, 1, 2, RZ, 2, 2, false).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = vec![regs(&[(1, 0.5), (2, 0.5)])];
    let extras = Extras {
        images: vec![image(
            SampledKind::D2,
            vk::Format::R32G32B32A32_SFLOAT,
            (1, 1, 1),
            1,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    let got = &out.states[0];
    let pair = |a: f32, b: f32| half_bits(a) | half_bits(b) << 16;
    assert_eq!(got.regs[8], pair(texel[0], texel[1]));
    assert_eq!(got.regs[9], pair(texel[2], texel[3]));
    assert_eq!(got.regs[5], pair(texel[1], texel[3]));
    assert_eq!(got.regs[6], half_bits(texel[2]), "the second half is zero");
}

#[test]
fn tex_reads_its_operands_from_ra_and_the_register_after_rb() {
    let Some(vk) = vulkan() else { return };
    let texel = |x: u32, y: u32| [x as f32 + 1.0, y as f32 + 10.0, (x * 4 + y) as f32, 0.5];
    let texels: Vec<[f32; 4]> = (0..4)
        .flat_map(|y| (0..4).map(move |x| texel(x, y)))
        .collect();
    let bytes = floats(&texels);
    // The coordinates in R1, R2 (`Ra`), the level of detail in R3 (`Rb`).
    let program = [
        // Level of detail zero (mode 1): all four components to R5..R8.
        tex(5, 1, 3, 2, 0xF, 1, false).build(),
        // The level in a register (mode 3, zero here): red and blue (mask 5) to R9, R10.
        tex(9, 1, 3, 2, 0x5, 3, false).build(),
        // No level (mode 0): alpha alone (mask 8) to R11.
        tex(11, 1, 3, 2, 0x8, 0, false).build(),
        // A bias (mode 2; none to apply in a compute shader): green (mask 2) to R12.
        tex(12, 1, 3, 2, 0x2, 2, false).build(),
        exit(Guard::ALWAYS),
    ];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for y in 0..4u32 {
        for x in 0..4u32 {
            inputs.push(regs(&[
                (1, (x as f32 + 0.5) / 4.0),
                (2, (y as f32 + 0.5) / 4.0),
                (3, 0.0),
            ]));
            expected.push(texel(x, y));
        }
    }
    let extras = Extras {
        images: vec![image(
            SampledKind::D2,
            vk::Format::R32G32B32A32_SFLOAT,
            (4, 4, 1),
            1,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        assert_eq!(
            [f(got, 5), f(got, 6), f(got, 7), f(got, 8)],
            *want,
            "all components, sample {i}"
        );
        assert_eq!(
            [f(got, 9), f(got, 10)],
            [want[0], want[2]],
            "red and blue, sample {i}"
        );
        assert_eq!(f(got, 11), want[3], "alpha, sample {i}");
        assert_eq!(f(got, 12), want[1], "green, sample {i}");
    }
}

#[test]
fn tex_of_an_array_a_volume_and_a_cube() {
    let Some(vk) = vulkan() else { return };
    // An array of three 2x2 layers.
    let layer_texel = |l: u32, x: u32, y: u32| [l as f32, x as f32, y as f32, 7.0];
    let array: Vec<[f32; 4]> = (0..3)
        .flat_map(|l| (0..2).flat_map(move |y| (0..2).map(move |x| layer_texel(l, x, y))))
        .collect();
    let bytes = floats(&array);
    // The layer (an integer) in R1 (`Ra`), then x, y in R2, R3; level zero (mode 1).
    let program = [tex(8, 1, RZ, 3, 0xF, 1, false).build(), exit(Guard::ALWAYS)];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for l in 0..3u32 {
        for y in 0..2u32 {
            for x in 0..2u32 {
                let mut s = regs(&[(2, (x as f32 + 0.5) / 2.0), (3, (y as f32 + 0.5) / 2.0)]);
                s.regs[1] = l;
                inputs.push(s);
                expected.push(layer_texel(l, x, y));
            }
        }
    }
    let extras = Extras {
        images: vec![image(
            SampledKind::D2Array,
            vk::Format::R32G32B32A32_SFLOAT,
            (2, 2, 1),
            3,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        assert_eq!(
            [f(got, 8), f(got, 9), f(got, 10), f(got, 11)],
            *want,
            "array sample {i}"
        );
    }

    // A 2x2x2 volume: x, y, z in R1, R2, R3.
    let voxel = |x: u32, y: u32, z: u32| [x as f32, y as f32, z as f32, 3.0];
    let volume: Vec<[f32; 4]> = (0..2)
        .flat_map(|z| (0..2).flat_map(move |y| (0..2).map(move |x| voxel(x, y, z))))
        .collect();
    let bytes = floats(&volume);
    let program = [tex(8, 1, RZ, 4, 0xF, 1, false).build(), exit(Guard::ALWAYS)];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for z in 0..2u32 {
        for y in 0..2u32 {
            for x in 0..2u32 {
                inputs.push(regs(&[
                    (1, (x as f32 + 0.5) / 2.0),
                    (2, (y as f32 + 0.5) / 2.0),
                    (3, (z as f32 + 0.5) / 2.0),
                ]));
                expected.push(voxel(x, y, z));
            }
        }
    }
    let extras = Extras {
        images: vec![image(
            SampledKind::D3,
            vk::Format::R32G32B32A32_SFLOAT,
            (2, 2, 2),
            1,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        assert_eq!(
            [f(got, 8), f(got, 9), f(got, 10), f(got, 11)],
            *want,
            "volume sample {i}"
        );
    }

    // A cube of 2x2 faces: a direction in R1..R3.
    let faces: Vec<[f32; 4]> = (0..6u32)
        .flat_map(|face| {
            (0..2u32)
                .flat_map(move |y| (0..2u32).map(move |x| [face as f32, x as f32, y as f32, 5.0]))
        })
        .collect();
    let bytes = floats(&faces);
    let program = [tex(8, 1, RZ, 6, 0xF, 1, false).build(), exit(Guard::ALWAYS)];
    let directions: Vec<[f32; 3]> = vec![
        [1.0, 0.25, -0.5],
        [-0.5, -1.0, 0.25],
        [0.25, 0.5, 1.0],
        [-1.0, 0.5, 0.5],
    ];
    let inputs: Vec<State> = directions
        .iter()
        .map(|d| regs(&[(1, d[0]), (2, d[1]), (3, d[2])]))
        .collect();
    let extras = Extras {
        images: vec![image(
            SampledKind::Cube,
            vk::Format::R32G32B32A32_SFLOAT,
            (2, 2, 1),
            6,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (d, got) in directions.iter().zip(&out.states) {
        let (face, x, y) = cube_texel(*d, 2);
        assert_eq!(
            [f(got, 8), f(got, 9), f(got, 10), f(got, 11)],
            [face as f32, x as f32, y as f32, 5.0],
            "cube direction {d:?}"
        );
    }
}

#[test]
fn tex_with_a_depth_comparison_takes_the_reference_after_the_level() {
    let Some(vk) = vulkan() else { return };
    let depths: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    let bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
    // The coordinates in R1, R2; the level (mode 3) in R3 and the reference in R4.
    let program = [tex(8, 1, 3, 2, 0x1, 3, true).build(), exit(Guard::ALWAYS)];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for y in 0..2usize {
        for x in 0..2usize {
            for reference in [0.4f32, 0.9] {
                inputs.push(regs(&[
                    (1, (x as f32 + 0.5) / 2.0),
                    (2, (y as f32 + 0.5) / 2.0),
                    (3, 0.0),
                    (4, reference),
                ]));
                expected.push(if reference <= depths[y * 2 + x] {
                    1.0f32
                } else {
                    0.0
                });
            }
        }
    }
    let extras = Extras {
        images: vec![image(
            SampledKind::D2,
            vk::Format::D32_SFLOAT,
            (2, 2, 1),
            1,
            true,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    for (i, (got, want)) in out.states.iter().zip(&expected).enumerate() {
        assert_eq!(f(got, 8), *want, "sample {i}");
    }
}

/// `TLD4S`: the four texels' component `component` to R`rd`.. (the gather order of Vulkan:
/// left-bottom-up, right-bottom-up... the order of the texels around the point).
fn tld4s(rd: u8, ra: u8, rb: u8, r28: u8, dc: bool, component: u64, halves: bool) -> Enc {
    Enc::op(Op::Tld4s)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .field(28, 8, u64::from(r28))
        .field(36, 13, TEXTURE_INDEX)
        .bit(50, dc)
        .field(52, 2, component)
        .bit(55, halves)
}

fn tld4(rd: u8, ra: u8, rb: u8, mask: u64, dc: bool, component: u64) -> Enc {
    Enc::op(Op::Tld4Bound)
        .rd(rd)
        .ra(ra)
        .rb(rb)
        .field(28, 3, 2)
        .field(31, 4, mask)
        .field(36, 13, TEXTURE_INDEX)
        .bit(50, dc)
        .field(56, 2, component)
}

#[test]
fn tld4_gathers_one_component_of_the_four_texels_around_the_point() {
    let Some(vk) = vulkan() else { return };
    // A 2x2 texture; the point at its center has all four texels around it. Vulkan's gather
    // order is (0, 1), (1, 1), (1, 0), (0, 0) as (x, y).
    let texel = |x: u32, y: u32| {
        [
            x as f32 + 1.0,
            y as f32 + 10.0,
            (x + 2 * y) as f32 + 20.0,
            0.5,
        ]
    };
    let texels: Vec<[f32; 4]> = (0..2)
        .flat_map(|y| (0..2).map(move |x| texel(x, y)))
        .collect();
    let bytes = floats(&texels);
    let order = [(0u32, 1u32), (1, 1), (1, 0), (0, 0)];
    let program = [
        // Component 1 (green) of the long form: all four to R5..R8.
        tld4(5, 1, RZ, 0xF, false, 1).build(),
        // Component 2 (blue) of the short form: four values to R9, R10 and R11, R12.
        tld4s(9, 1, 2, 11, false, 2, false).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = vec![regs(&[(1, 0.5), (2, 0.5)])];
    let extras = Extras {
        images: vec![image(
            SampledKind::D2,
            vk::Format::R32G32B32A32_SFLOAT,
            (2, 2, 1),
            1,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    let got = &out.states[0];
    for (i, (x, y)) in order.iter().enumerate() {
        assert_eq!(f(got, 5 + i), texel(*x, *y)[1], "TLD4, texel {i}");
    }
    // `TLD4S` with two destinations writes (Rd, Rd+1) and (R28, R28+1).
    let short = [f(got, 9), f(got, 10), f(got, 11), f(got, 12)];
    for (i, (x, y)) in order.iter().enumerate() {
        assert_eq!(short[i], texel(*x, *y)[2], "TLD4S, texel {i}");
    }
}

#[test]
fn tld4s_with_a_depth_reference_gathers_four_comparisons() {
    let Some(vk) = vulkan() else { return };
    let depths: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    let bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
    // With a reference the coordinates are R1, R2 (`Ra`, `Ra+1`) and the reference is R3 (`Rb`).
    let program = [
        tld4s(9, 1, 3, 11, true, 0, false).build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = vec![regs(&[(1, 0.5), (2, 0.5), (3, 0.6)])];
    let extras = Extras {
        images: vec![image(
            SampledKind::D2,
            vk::Format::D32_SFLOAT,
            (2, 2, 1),
            1,
            true,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    let got = &out.states[0];
    // The texels in gather order: (0,1) = 0.75, (1,1) = 1.0, (1,0) = 0.5, (0,0) = 0.25.
    let want: Vec<f32> = [0.75f32, 1.0, 0.5, 0.25]
        .iter()
        .map(|d| if 0.6 <= *d { 1.0 } else { 0.0 })
        .collect();
    let have = [f(got, 9), f(got, 10), f(got, 11), f(got, 12)];
    assert_eq!(have.to_vec(), want);
}

#[test]
fn txq_reports_the_size_and_the_levels() {
    let Some(vk) = vulkan() else { return };
    let texels = vec![[0.0f32; 4]; 4 * 4 * 3];
    let bytes = floats(&texels);
    // `TXQ` names no dimension: the array sample before it says the texture is a 2D array. The
    // level asked about is in R1 (zero).
    let program = [
        texs(5, 2, 3, RZ, 0, 8, true).build(),
        Enc::op(Op::TxqBound)
            .rd(8)
            .ra(1)
            .field(22, 6, 1)
            .field(31, 4, 0xF)
            .field(36, 13, TEXTURE_INDEX)
            .build(),
        exit(Guard::ALWAYS),
    ];
    let inputs = vec![State::default()];
    let extras = Extras {
        images: vec![image(
            SampledKind::D2Array,
            vk::Format::R32G32B32A32_SFLOAT,
            (4, 4, 1),
            3,
            false,
            &bytes,
        )],
        ..Extras::default()
    };
    let out = run_with(&vk, &program, &inputs, &[], extras);
    let got = &out.states[0];
    assert_eq!(
        [got.regs[8], got.regs[9], got.regs[10], got.regs[11]],
        [4, 4, 3, 1],
        "width, height, layers, levels"
    );
}
