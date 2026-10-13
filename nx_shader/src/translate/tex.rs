//! Texture sampling and queries: `TEXS`, `TEX`, `TLD4S`, `TLD4` and `TXQ`.
//!
//! The texture an instruction names is the 13-bit index in bits 36..48, which picks a slot of the
//! table of handles the host resolves at draw time (so the translation only declares "sampler
//! number `index`"). The coordinates and the other operands come from consecutive registers; where
//! they start, and in which order, is the encoding of each form (docs/ASSUMPTIONS.md A-106):
//!
//! * `TEXS` (the short form of the compiler's common samples): bits 53..56 pick the kind
//!   (dimension, how the level of detail is chosen, whether the sample compares against a depth);
//!   the operands are `Ra` and `Rb` and the registers after them; the results go to `Rd`, `Rd+1`
//!   and, when bits 28..35 name a register, `R28`, `R28+1`, the components in the order the
//!   selector in bits 50..52 gives, as floats or (bit 59 clear) as packed halves;
//! * `TEX`: the dimension is in bits 28..30, the components wanted in bits 31..34 and the way the
//!   level of detail is chosen in bits 55..57. The coordinates start at `Ra` (with an array index
//!   first), the level of detail or bias and then the depth reference at `Rb`; the results go to
//!   consecutive registers from `Rd`, one for each component wanted;
//! * `TLD4S` and `TLD4` gather the same component of the four texels a bilinear sample would use;
//! * `TXQ` asks for the size of the texture (and the number of its mip levels).

use super::emit::{reg_at, Tx, R};
use super::{Dim, Stage, TextureUse};
use crate::cfg::{Cfg, Code};
use crate::inst::{Inst, RZ};
use crate::ops::Op;
use crate::spirv::{cap, deco, glsl, op, storage, Id};
use std::collections::HashMap;

/// How the level of detail of a sample is chosen.
#[derive(Clone, Copy, Debug)]
enum Lod {
    /// From the screen-space derivatives of the coordinates (the base level in a stage that has
    /// none).
    Implicit,
    /// Zero.
    Zero,
    /// This level.
    Level(Id),
    /// The implicit level plus this bias.
    Bias(Id),
}

/// The texture an instruction names: its number in the table of handles, or (a bindless
/// instruction) the constant buffer slot whose word `index` holds the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tex {
    index: u32,
    source: Option<u32>,
}

impl Tex {
    fn table(index: u32) -> Tex {
        Tex {
            index,
            source: None,
        }
    }
}

/// A sample to be made.
struct Sample {
    tex: Tex,
    dim: Dim,
    arrayed: bool,
    shadow: bool,
    /// The coordinates as floats, one for each dimension (a cube takes three).
    coords: Vec<Id>,
    /// The array layer, as a float.
    layer: Option<Id>,
    lod: Lod,
    /// The value a shadow sample compares against.
    dref: Option<Id>,
}

/// The dimension of the `TEX` family: bits 28..30 (0 1D, 1 1D array, 2 2D, 3 2D array, 4 3D,
/// 6 cube, 7 cube array).
fn tex_dim(code: u64) -> Option<(Dim, bool)> {
    Some(match code {
        0 => (Dim::D1, false),
        1 => (Dim::D1, true),
        2 => (Dim::D2, false),
        3 => (Dim::D2, true),
        4 => (Dim::D3, false),
        6 => (Dim::Cube, false),
        7 => (Dim::Cube, true),
        _ => return None,
    })
}

/// The kinds of `TEXS`: dimension, array, level of detail, depth compare (bits 53..56).
fn kind(code: u64) -> Option<(Dim, bool, KindLod, bool)> {
    use KindLod::{Implicit, Level, Zero};
    Some(match code {
        0 => (Dim::D1, false, Zero, false),
        1 => (Dim::D2, false, Implicit, false),
        2 => (Dim::D2, false, Zero, false),
        3 => (Dim::D2, false, Level, false),
        4 => (Dim::D2, false, Implicit, true),
        5 => (Dim::D2, false, Level, true),
        6 => (Dim::D2, false, Zero, true),
        7 => (Dim::D2, true, Implicit, false),
        8 => (Dim::D2, true, Zero, false),
        9 => (Dim::D2, true, Zero, true),
        10 => (Dim::D3, false, Implicit, false),
        11 => (Dim::D3, false, Zero, false),
        12 => (Dim::Cube, false, Implicit, false),
        13 => (Dim::Cube, false, Level, false),
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KindLod {
    Implicit,
    Zero,
    Level,
}

/// What the texture instructions of a program say about each texture slot's shape: the first
/// instruction that names a slot decides. `TXQ` has no dimension of its own and takes the one the
/// other instructions use.
pub(super) fn scan_dims(code: Code<'_>, graph: &Cfg) -> HashMap<u32, (Dim, bool)> {
    // The dimension codes of `TLDS` (bits 53..56) as the `TEX` family's.
    const TLDS: [i8; 16] = [0, 0, 2, 2, 2, 2, 2, 4, 3, -1, -1, -1, 2, -1, -1, -1];
    let mut dims = HashMap::new();
    for block in graph.blocks.values() {
        for (_, inst) in graph.instructions(code, block) {
            let Some(opc) = inst.op() else { continue };
            let found = match opc {
                Op::Texs => kind(inst.bits(53, 4)).map(|(d, a, _, _)| (d, a)),
                Op::Tlds => match TLDS[inst.bits(53, 4) as usize] {
                    -1 => None,
                    c => tex_dim(c as u64),
                },
                Op::Tld4s => Some((Dim::D2, false)),
                Op::TexBound | Op::TldBound | Op::Tld4Bound | Op::TmmlBound | Op::TxdBound => {
                    tex_dim(inst.bits(28, 3))
                }
                _ => None,
            };
            if let Some(shape) = found {
                dims.entry(inst.bits(36, 13) as u32).or_insert(shape);
            }
        }
    }
    dims
}

/// The combined image sampler variable of a texture, made on first use.
fn sampler(tx: &mut Tx<'_>, tex: Tex, dim: Dim, arrayed: bool, shadow: bool) -> R<(Id, Id)> {
    if dim == Dim::Cube && arrayed {
        return tx.fail("a cube map array is not translated yet");
    }
    if dim == Dim::D3 && (arrayed || shadow) {
        return tx.fail("a 3D texture with an array or a depth comparison");
    }
    let f = tx.t.f32;
    let dim_code = match dim {
        Dim::D1 => {
            tx.m.capability(cap::SAMPLED_1D);
            0
        }
        Dim::D2 => 1,
        Dim::D3 => 2,
        Dim::Cube => 3,
    };
    let image =
        tx.m.ty_image(f, dim_code, u32::from(shadow), u32::from(arrayed), 0, 1, 0);
    let sampled = tx.m.ty_sampled_image(image);
    if let Some(var) = tx.textures.get(&(tex.index, tex.source)) {
        let used = tx
            .info
            .textures
            .iter()
            .find(|t| (t.index, t.source) == (tex.index, tex.source));
        if used.is_some_and(|t| (t.dim, t.arrayed, t.shadow) != (dim, arrayed, shadow)) {
            return tx.fail(format!(
                "texture {} is sampled as two different kinds of image",
                tex.index
            ));
        }
        return Ok((*var, sampled));
    }
    let ptr = tx.m.ty_pointer(storage::UNIFORM_CONSTANT, sampled);
    let var = tx.m.global_variable(ptr, storage::UNIFORM_CONSTANT);
    tx.m.decorate(var, deco::DESCRIPTOR_SET, &[0]);
    let slot = tx.info.textures.len() as u32;
    if slot >= 40 {
        return tx.fail("more than 40 textures in one program");
    }
    let binding = 64 * tx.cfg.stage.index() + 24 + slot;
    tx.m.decorate(var, deco::BINDING, &[binding]);
    tx.interface.push(var);
    tx.textures.insert((tex.index, tex.source), var);
    tx.info.textures.push(TextureUse {
        index: tex.index,
        source: tex.source,
        slot,
        dim,
        arrayed,
        shadow,
    });
    Ok((var, sampled))
}

/// The coordinate operand: the coordinates in one vector (or a scalar for a 1D texture), the array
/// layer last.
fn coordinate(tx: &mut Tx<'_>, s: &Sample) -> Id {
    let mut parts = s.coords.clone();
    parts.extend(s.layer);
    if parts.len() == 1 {
        return parts[0];
    }
    let f = tx.t.f32;
    let ty = tx.m.ty_vector(f, parts.len() as u32);
    tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, ty, &parts)
}

/// Sample: the four components of the texel (a depth comparison gives its one result in all four).
fn sample(tx: &mut Tx<'_>, s: &Sample) -> R<Id> {
    let (var, sampled_ty) = sampler(tx, s.tex, s.dim, s.arrayed, s.shadow)?;
    let coord = coordinate(tx, s);
    let image = tx.f.load(&mut tx.m, sampled_ty, var);
    let f = tx.t.f32;
    // Only a pixel program has the derivatives an implicit level comes from.
    let lod = match s.lod {
        Lod::Implicit | Lod::Bias(_) if tx.cfg.stage != Stage::Pixel => Lod::Zero,
        other => other,
    };
    // The image operands that follow the coordinate (and the depth reference).
    let zero;
    let operands: Vec<u32> = match lod {
        Lod::Implicit => Vec::new(),
        Lod::Bias(b) => vec![0x1, b],
        Lod::Level(l) => vec![0x2, l],
        Lod::Zero => {
            zero = tx.cf32(0.0);
            vec![0x2, zero]
        }
    };
    let explicit = matches!(lod, Lod::Level(_) | Lod::Zero);
    if s.shadow {
        let Some(dref) = s.dref else {
            return tx.fail("a depth comparison without a reference");
        };
        let opcode = if explicit {
            op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD
        } else {
            op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD
        };
        let mut args = vec![image, coord, dref];
        args.extend(&operands);
        let scalar = tx.f.emit_result(&mut tx.m, opcode, f, &args);
        let v4 = tx.t.v4f32;
        return Ok(tx.f.emit_result(
            &mut tx.m,
            op::COMPOSITE_CONSTRUCT,
            v4,
            &[scalar, scalar, scalar, scalar],
        ));
    }
    let opcode = if explicit {
        op::IMAGE_SAMPLE_EXPLICIT_LOD
    } else {
        op::IMAGE_SAMPLE_IMPLICIT_LOD
    };
    let mut args = vec![image, coord];
    args.extend(&operands);
    let v4 = tx.t.v4f32;
    Ok(tx.f.emit_result(&mut tx.m, opcode, v4, &args))
}

/// `Rd`, `Rd+1`, `R28`, `R28+1` receive the components the selector names (see the module's
/// documentation); packed in halves when bit 59 is clear.
fn write_short_results(tx: &mut Tx<'_>, inst: Inst, result: Id) -> R<()> {
    let f = tx.t.f32;
    let two = inst.r28() != RZ;
    let sel = inst.bits(50, 3);
    let comps: &[u32] = if two {
        match sel {
            0 => &[0, 1, 2],
            1 => &[0, 1, 3],
            2 => &[0, 2, 3],
            3 => &[1, 2, 3],
            4 => &[0, 1, 2, 3],
            _ => {
                return tx.fail(format!(
                    "a short texture instruction's component selector {sel} with two destinations"
                ))
            }
        }
    } else {
        match sel {
            0 => &[0],
            1 => &[1],
            2 => &[2],
            3 => &[3],
            4 => &[0, 1],
            5 => &[0, 3],
            6 => &[1, 3],
            _ => &[2, 3],
        }
    };
    let values: Vec<Id> = comps
        .iter()
        .map(|c| {
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[result, *c])
        })
        .collect();
    if inst.bit(59) {
        for (i, v) in values.iter().enumerate() {
            let reg = if i < 2 {
                reg_at(inst.rd(), i as u32)
            } else {
                reg_at(inst.r28(), i as u32 - 2)
            };
            tx.write_f(inst, reg, *v);
        }
        return Ok(());
    }
    // Packed halves: two components to a register, the first in the low half.
    let zero = tx.cf32(0.0);
    for (pair, reg) in values.chunks(2).zip([inst.rd(), inst.r28()]) {
        let v2 = tx.t.v2f32;
        let both = tx.f.emit_result(
            &mut tx.m,
            op::COMPOSITE_CONSTRUCT,
            v2,
            &[pair[0], *pair.get(1).unwrap_or(&zero)],
        );
        let packed = tx.ext(tx.t.u32, glsl::PACK_HALF_2X16, &[both]);
        tx.write(inst, reg, packed);
    }
    Ok(())
}

/// The components `mask` (bits 31..34 of the long forms) selects, written to consecutive registers
/// from `Rd`; a mask of zero means all four.
fn write_masked_results(tx: &mut Tx<'_>, inst: Inst, result: Id) {
    let f = tx.t.f32;
    let mask = match inst.bits(31, 4) {
        0 => 0xF,
        m => m,
    };
    let mut at = 0;
    for c in 0..4u32 {
        if mask >> c & 1 == 0 {
            continue;
        }
        let v =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[result, c]);
        tx.write_f(inst, reg_at(inst.rd(), at), v);
        at += 1;
    }
}

pub(super) fn texs(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let code = inst.bits(53, 4);
    let Some((dim, arrayed, lod, shadow)) = kind(code) else {
        return tx.fail(format!("TEXS of kind {code}"));
    };
    let index = inst.bits(36, 13) as u32;
    let (ra, rb) = (inst.ra(), inst.rb());
    // Where the operands are: the plain 2D forms take x and y from `Ra` and `Rb`; the others keep
    // the coordinates together from `Ra` and put the level and the reference from `Rb` on. An
    // array takes its layer (an integer) from `Ra`, then the coordinates.
    let f = tx.t.f32;
    let (coords, layer, next): (Vec<Id>, Option<Id>, u8) = match (dim, arrayed) {
        (Dim::D1, _) => (vec![tx.read_f(ra)], None, rb),
        (Dim::D2, false) if lod != KindLod::Level && !shadow => {
            (vec![tx.read_f(ra), tx.read_f(rb)], None, reg_at(rb, 1))
        }
        (Dim::D2, false) => (vec![tx.read_f(ra), tx.read_f(reg_at(ra, 1))], None, rb),
        (Dim::D2, true) => {
            let layer = tx.read(ra);
            let layer = tx.op1(op::CONVERT_U_TO_F, f, layer);
            (
                vec![tx.read_f(reg_at(ra, 1)), tx.read_f(rb)],
                Some(layer),
                reg_at(rb, 1),
            )
        }
        (Dim::D3 | Dim::Cube, _) => (
            vec![tx.read_f(ra), tx.read_f(reg_at(ra, 1)), tx.read_f(rb)],
            None,
            reg_at(rb, 1),
        ),
    };
    // The plain forms with a level or a depth reference name them at `Rb`; for the forms whose
    // coordinates take `Rb` too, they follow it.
    let (level, dref) = match (lod == KindLod::Level, shadow) {
        (false, false) => (None, None),
        (true, false) => (Some(next), None),
        (false, true) => (None, Some(next)),
        (true, true) => (Some(next), Some(reg_at(next, 1))),
    };
    let lod = match lod {
        KindLod::Implicit => Lod::Implicit,
        KindLod::Zero => Lod::Zero,
        KindLod::Level => Lod::Level(tx.read_f(level.unwrap_or(next))),
    };
    let dref = dref.map(|r| tx.read_f(r));
    let result = sample(
        tx,
        &Sample {
            tex: Tex::table(index),
            dim,
            arrayed,
            shadow,
            coords,
            layer,
            lod,
            dref,
        },
    )?;
    write_short_results(tx, inst, result)
}

pub(super) fn tex(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let Some((dim, arrayed)) = tex_dim(inst.bits(28, 3)) else {
        return tx.fail(format!("TEX of dimension {}", inst.bits(28, 3)));
    };
    if arrayed && dim == Dim::D1 {
        return tx.fail("TEX of a 1D array");
    }
    if inst.bit(54) {
        return tx.fail("TEX with texel offsets (AOFFI) is not translated yet");
    }
    let shadow = inst.bit(50);
    let index = inst.bits(36, 13) as u32;
    let f = tx.t.f32;
    let mut r = inst.ra();
    let layer = if arrayed {
        let raw = tx.read(r);
        r = reg_at(r, 1);
        Some(tx.op1(op::CONVERT_U_TO_F, f, raw))
    } else {
        None
    };
    let n = match dim {
        Dim::D1 => 1,
        Dim::D2 => 2,
        Dim::D3 | Dim::Cube => 3,
    };
    let coords: Vec<Id> = (0..n).map(|k| tx.read_f(reg_at(r, k))).collect();
    let mut m = inst.rb();
    let lod = match inst.bits(55, 3) {
        0 => Lod::Implicit,
        1 => Lod::Zero,
        2 | 6 => {
            let bias = tx.read_f(m);
            m = reg_at(m, 1);
            Lod::Bias(bias)
        }
        3 | 7 => {
            let level = tx.read_f(m);
            m = reg_at(m, 1);
            Lod::Level(level)
        }
        mode => return tx.fail(format!("TEX with level of detail mode {mode}")),
    };
    let dref = shadow.then(|| tx.read_f(m));
    let result = sample(
        tx,
        &Sample {
            tex: Tex::table(index),
            dim,
            arrayed,
            shadow,
            coords,
            layer,
            lod,
            dref,
        },
    )?;
    write_masked_results(tx, inst, result);
    Ok(())
}

/// Gather the component `component` of the four texels of a sample; with a reference, the results
/// of four depth comparisons instead.
fn gather(
    tx: &mut Tx<'_>,
    index: u32,
    coords: Vec<Id>,
    shadow: bool,
    component: u32,
    dref: Option<Id>,
) -> R<Id> {
    let s = Sample {
        tex: Tex::table(index),
        dim: Dim::D2,
        arrayed: false,
        shadow,
        coords,
        layer: None,
        lod: Lod::Zero,
        dref,
    };
    let (var, sampled_ty) = sampler(tx, Tex::table(index), Dim::D2, false, shadow)?;
    let coord = coordinate(tx, &s);
    let image = tx.f.load(&mut tx.m, sampled_ty, var);
    let v4 = tx.t.v4f32;
    if shadow {
        let Some(dref) = dref else {
            return tx.fail("a depth comparison without a reference");
        };
        return Ok(tx
            .f
            .emit_result(&mut tx.m, op::IMAGE_DREF_GATHER, v4, &[image, coord, dref]));
    }
    let comp = tx.c32(component);
    Ok(tx
        .f
        .emit_result(&mut tx.m, op::IMAGE_GATHER, v4, &[image, coord, comp]))
}

pub(super) fn tld4s(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let (dc, aoffi) = (inst.bit(50), inst.bit(51));
    if aoffi {
        return tx.fail("TLD4S with texel offsets is not translated yet");
    }
    let index = inst.bits(36, 13) as u32;
    let (ra, rb) = (inst.ra(), inst.rb());
    // Without a reference the coordinates are `Ra` and `Rb`; with one, `Ra` and the register after
    // it, and the reference is `Rb`.
    let (coords, dref) = if dc {
        (
            vec![tx.read_f(ra), tx.read_f(reg_at(ra, 1))],
            Some(tx.read_f(rb)),
        )
    } else {
        (vec![tx.read_f(ra), tx.read_f(rb)], None)
    };
    let result = gather(tx, index, coords, dc, inst.bits(52, 2) as u32, dref)?;
    let f = tx.t.f32;
    let values: Vec<Id> = (0..4)
        .map(|c| {
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[result, c])
        })
        .collect();
    if inst.bit(55) {
        // Packed halves: x and y in `Rd`, z and w in `R28`.
        for (pair, reg) in values.chunks(2).zip([inst.rd(), inst.r28()]) {
            let v2 = tx.t.v2f32;
            let both =
                tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v2, &[pair[0], pair[1]]);
            let packed = tx.ext(tx.t.u32, glsl::PACK_HALF_2X16, &[both]);
            tx.write(inst, reg, packed);
        }
        return Ok(());
    }
    for (i, v) in values.iter().enumerate() {
        let reg = if i < 2 {
            reg_at(inst.rd(), i as u32)
        } else {
            reg_at(inst.r28(), i as u32 - 2)
        };
        tx.write_f(inst, reg, *v);
    }
    Ok(())
}

pub(super) fn tld4(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let Some((dim, arrayed)) = tex_dim(inst.bits(28, 3)) else {
        return tx.fail(format!("TLD4 of dimension {}", inst.bits(28, 3)));
    };
    if dim != Dim::D2 || arrayed {
        return tx.fail(format!(
            "TLD4 of a {dim:?}{} texture is not translated yet",
            if arrayed { " array" } else { "" }
        ));
    }
    let offsets = inst.bits(54, 2);
    if offsets != 0 {
        return tx.fail("TLD4 with texel offsets is not translated yet");
    }
    let shadow = inst.bit(50);
    let index = inst.bits(36, 13) as u32;
    let ra = inst.ra();
    let coords = vec![tx.read_f(ra), tx.read_f(reg_at(ra, 1))];
    let dref = shadow.then(|| tx.read_f(inst.rb()));
    let result = gather(tx, index, coords, shadow, inst.bits(56, 2) as u32, dref)?;
    write_masked_results(tx, inst, result);
    Ok(())
}

pub(super) fn txq(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let query = inst.bits(22, 6);
    let mask = inst.bits(31, 4);
    let index = inst.bits(36, 13) as u32;
    if query != 1 {
        return tx.fail(format!("TXQ of kind {query}"));
    }
    // The texture's shape is what the program's other instructions say it is; a 2D texture when
    // none does.
    let (dim, arrayed) = tx
        .texture_dims
        .get(&index)
        .copied()
        .unwrap_or((Dim::D2, false));
    let (var, sampled_ty) = sampler(tx, Tex::table(index), dim, arrayed, false)?;
    tx.m.capability(cap::IMAGE_QUERY);
    let sampled = tx.f.load(&mut tx.m, sampled_ty, var);
    // The image itself, for the query.
    let f = tx.t.f32;
    let image_ty = tx.m.ty_image(
        f,
        match dim {
            Dim::D1 => 0,
            Dim::D2 => 1,
            Dim::D3 => 2,
            Dim::Cube => 3,
        },
        0,
        u32::from(arrayed),
        0,
        1,
        0,
    );
    let image = tx.f.emit_result(&mut tx.m, op::IMAGE, image_ty, &[sampled]);
    let level = tx.read(inst.ra());
    let int = tx.t.i32;
    let size_components = match dim {
        Dim::D1 => 1 + u32::from(arrayed),
        Dim::D2 | Dim::Cube => 2 + u32::from(arrayed),
        Dim::D3 => 3,
    };
    let size_ty = if size_components == 1 {
        int
    } else {
        tx.m.ty_vector(int, size_components)
    };
    let level_i = tx.as_i32(level);
    let size = tx.f.emit_result(
        &mut tx.m,
        op::IMAGE_QUERY_SIZE_LOD,
        size_ty,
        &[image, level_i],
    );
    let levels =
        tx.f.emit_result(&mut tx.m, op::IMAGE_QUERY_LEVELS, int, &[image]);
    let u = tx.t.u32;
    let part = |tx: &mut Tx<'_>, i: u32| -> Id {
        if size_components == 1 {
            tx.as_u32(size)
        } else {
            let c =
                tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, int, &[size, i]);
            tx.as_u32(c)
        }
    };
    let one = tx.c32(1);
    // Width, height, depth or layers, levels.
    let mut values: Vec<Id> = Vec::new();
    values.push(part(tx, 0));
    values.push(if size_components >= 2 {
        part(tx, 1)
    } else {
        one
    });
    values.push(if size_components >= 3 {
        part(tx, 2)
    } else {
        one
    });
    values.push(tx.op1(op::BITCAST, u, levels));
    let mut at = 0;
    for (c, v) in values.iter().enumerate() {
        if mask >> c & 1 != 0 {
            tx.write(inst, reg_at(inst.rd(), at), *v);
            at += 1;
        }
    }
    Ok(())
}

/// `TMML` and `TMML.B`: the level of detail a sample at the coordinates would use. The two results
/// are fixed point numbers with 8 fractional bits: the first the level without the sampler's
/// limits, as a signed integer, the second the level that would be read, as an unsigned one
/// (docs/ASSUMPTIONS.md A-106). Only a pixel program has the derivatives it takes.
pub(super) fn tmml(tx: &mut Tx<'_>, inst: Inst, bindless: bool) -> R<()> {
    if tx.cfg.stage != Stage::Pixel {
        return tx
            .fail("TMML outside a pixel program (it needs the derivatives of the coordinates)");
    }
    let Some((dim, arrayed)) = tex_dim(inst.bits(28, 3)) else {
        return tx.fail(format!("TMML of dimension {}", inst.bits(28, 3)));
    };
    if arrayed {
        return tx.fail("TMML of an array texture is not translated yet");
    }
    let mask = match inst.bits(31, 4) {
        0 => 0b11,
        m if m < 4 => m,
        m => return tx.fail(format!("TMML with the component mask {m:#x}")),
    };
    let tex = if bindless {
        let Some(&(slot, word)) = tx.reg_origin.get(&inst.rb()) else {
            return tx.fail(
                "a bindless texture instruction whose handle does not come straight from a word of a constant buffer",
            );
        };
        Tex {
            index: word,
            source: Some(slot),
        }
    } else {
        Tex::table(inst.bits(36, 13) as u32)
    };
    let n = match dim {
        Dim::D1 => 1,
        Dim::D2 => 2,
        Dim::D3 | Dim::Cube => 3,
    };
    let coords: Vec<Id> = (0..n).map(|k| tx.read_f(reg_at(inst.ra(), k))).collect();
    let s = Sample {
        tex,
        dim,
        arrayed: false,
        shadow: false,
        coords,
        layer: None,
        lod: Lod::Implicit,
        dref: None,
    };
    let (var, sampled_ty) = sampler(tx, tex, dim, false, false)?;
    let coord = coordinate(tx, &s);
    let sampled = tx.f.load(&mut tx.m, sampled_ty, var);
    tx.m.capability(cap::IMAGE_QUERY);
    let v2 = tx.t.v2f32;
    let lod =
        tx.f.emit_result(&mut tx.m, op::IMAGE_QUERY_LOD, v2, &[sampled, coord]);
    let f = tx.t.f32;
    // SPIR-V gives the level that would be read first and the unclamped one second.
    let read =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[lod, 0]);
    let unclamped =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[lod, 1]);
    let scale = tx.cf32(256.0);
    let (i, u) = (tx.t.i32, tx.t.u32);
    let unclamped = tx.fmul(unclamped, scale);
    let read = tx.fmul(read, scale);
    let unclamped = tx.op1(op::CONVERT_F_TO_S, i, unclamped);
    let unclamped = tx.as_u32(unclamped);
    let read = tx.op1(op::CONVERT_F_TO_U, u, read);
    let mut at = 0;
    for (c, value) in [unclamped, read].into_iter().enumerate() {
        if mask >> c & 1 != 0 {
            tx.write(inst, reg_at(inst.rd(), at), value);
            at += 1;
        }
    }
    Ok(())
}
