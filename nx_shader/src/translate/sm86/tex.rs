//! Textures: `TEX`, `TLD`, `TLD4`, `TMML` and `TXQ`.
//!
//! The bound forms name the texture by a word of a constant buffer that holds its handle (bits
//! 40..53 the word, 54..58 the slot), which is what [`super::super::tex::Tex`] with a `source`
//! stands for. The operands come from two runs of registers, from bits 24..31 and 32..39: the
//! coordinates (an array index first) from the first and the rest (the level of detail or bias,
//! then the texel offsets, then the depth reference) from the second, or, with bit 60, half of
//! all of them (rounded up) from the first and the rest from the second (NeXium; A-244). The
//! results are the components bits 72..75 select, in order, into `Rd`, `Rd+1` and then the
//! register in bits 64..71 and the one after it; with `.F16` (bits 79..80) two components share a
//! register as halves.
//!
//! Field positions are NAK's: the dimension in bits 61..63 (0 1D, 4 1D array, 1 2D, 5 2D array,
//! 2 3D, 3 cube, 7 cube array), texel offsets in bit 76, the depth comparison in bit 78, the level
//! of detail in bits 87..89 (0 implicit, 1 zero, 2 bias, 3 level).

use super::super::emit::{Tx, R};
use super::super::tex::{coordinate, sample, sampler, texel_offsets, Lod, Sample, Tex};
use super::super::Dim;
use super::{reg_at, write, write_f, I};
use crate::sm86::{op, Inst, RZ};
use crate::spirv::{cap, glsl, op as sop, Id};

/// The dimension and arrayness bits 61..63 name.
fn dimension(code: u64) -> Option<(Dim, bool)> {
    Some(match code {
        0 => (Dim::D1, false),
        4 => (Dim::D1, true),
        1 => (Dim::D2, false),
        5 => (Dim::D2, true),
        2 => (Dim::D3, false),
        3 => (Dim::Cube, false),
        7 => (Dim::Cube, true),
        _ => return None,
    })
}

/// The shape a texture instruction says its texture has (for the translation's scan).
pub(super) fn shape(inst: Inst) -> Option<(Dim, bool)> {
    match inst.opcode() {
        op::TEX | op::TLD | op::TLD4 | op::TMML => dimension(inst.field(61, 3)),
        _ => None,
    }
}

/// The constant buffer word of a bound texture instruction's handle.
pub(super) fn handle_word(inst: Inst) -> u32 {
    inst.field(40, 14) as u32
}

/// The texture a bound instruction names.
fn texture(tx: &Tx<'_>, i: &I) -> R<Tex> {
    if matches!(
        i.inst.opcode(),
        op::TEX_B | op::TLD_B | op::TLD4_B | op::TMML_B | op::TXQ_B
    ) {
        return tx.fail("a bindless texture instruction");
    }
    Ok(Tex {
        index: handle_word(i.inst),
        source: Some(super::slot_of_texture(i.inst)),
    })
}

/// The registers of `count` operands: `coordinates` of them from the first run, the rest from the
/// second (or half and half with bit 60).
fn operands(tx: &Tx<'_>, i: &I, coordinates: u32, extras: u32) -> R<Vec<u8>> {
    let (first, second) = (i.inst.reg(24), i.inst.reg(32));
    let total = coordinates + extras;
    if i.inst.bit(60) {
        if total > 4 {
            return tx.fail("a texture instruction with more than four operands split in two");
        }
        let split = total.div_ceil(2);
        Ok((0..split)
            .map(|k| reg_at(i, first, k))
            .chain((0..total - split).map(|k| reg_at(i, second, k)))
            .collect())
    } else {
        Ok((0..coordinates)
            .map(|k| reg_at(i, first, k))
            .chain((0..extras).map(|k| reg_at(i, second, k)))
            .collect())
    }
}

/// Write the components of `result` the mask selects.
fn results(tx: &mut Tx<'_>, i: &I, result: Id, mask: u64) {
    let f = tx.t.f32;
    let values: Vec<Id> = (0..4u32)
        .filter(|c| mask >> c & 1 != 0)
        .map(|c| {
            tx.f.emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, f, &[result, c])
        })
        .collect();
    let (d0, d1) = (i.inst.dst(), i.inst.reg(64));
    if i.inst.field(79, 2) != 0 {
        // Two components to a register as halves, the first in the low half.
        let zero = tx.cf32(0.0);
        for (k, pair) in values.chunks(2).enumerate() {
            let reg = match k {
                0 => d0,
                _ if d1 != RZ => d1,
                _ => reg_at(i, d0, 1),
            };
            let v2 = tx.t.v2f32;
            let both = tx.f.emit_result(
                &mut tx.m,
                sop::COMPOSITE_CONSTRUCT,
                v2,
                &[pair[0], *pair.get(1).unwrap_or(&zero)],
            );
            let u = tx.t.u32;
            let packed = tx.ext(u, glsl::PACK_HALF_2X16, &[both]);
            write(tx, i, reg, packed);
        }
        return;
    }
    for (k, v) in values.into_iter().enumerate() {
        let k = k as u32;
        let reg = if k < 2 || d1 == RZ {
            reg_at(i, d0, k)
        } else {
            reg_at(i, d1, k - 2)
        };
        write_f(tx, i, reg, v);
    }
}

/// The number of coordinates of a dimension (not counting an array index).
fn coordinates_of(dim: Dim) -> u32 {
    match dim {
        Dim::D1 => 1,
        Dim::D2 => 2,
        Dim::D3 | Dim::Cube => 3,
    }
}

/// The texel offsets of the register `reg`, which an instruction of the block made a constant.
fn offsets(tx: &Tx<'_>, reg: u8) -> R<u32> {
    if reg == RZ {
        return Ok(0);
    }
    match tx.reg_const.get(&reg) {
        Some(v) => Ok(*v),
        None => tx.fail(format!(
            "texel offsets in R{reg}, which no instruction of the block made a constant"
        )),
    }
}

/// `TEX`: a sample.
pub(super) fn tex(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let Some((dim, arrayed)) = dimension(inst.field(61, 3)) else {
        return tx.fail(format!("TEX of dimension code {}", inst.field(61, 3)));
    };
    let mask = inst.field(72, 4);
    if mask == 0 {
        return tx.fail("TEX with no component selected");
    }
    let tex_ref = texture(tx, i)?;
    let lod_mode = inst.field(87, 3);
    let aoffi = inst.bit(76);
    let shadow = inst.bit(78);
    if aoffi && dim == Dim::Cube {
        return tx.fail("TEX of a cube map with texel offsets");
    }
    let lod_operand = match lod_mode {
        0 | 1 => false,
        2 | 3 => true,
        m => return tx.fail(format!("TEX with level of detail mode {m}")),
    };
    let n = coordinates_of(dim);
    let coordinate_count = n + u32::from(arrayed);
    let extras = u32::from(lod_operand) + u32::from(aoffi) + u32::from(shadow);
    let regs = operands(tx, i, coordinate_count, extras)?;
    let f = tx.t.f32;
    let mut at = 0usize;
    let layer = if arrayed {
        let raw = tx.read(regs[0]);
        at = 1;
        Some(tx.op1(sop::CONVERT_U_TO_F, f, raw))
    } else {
        None
    };
    let coords: Vec<Id> = (0..n as usize).map(|k| tx.read_f(regs[at + k])).collect();
    let mut rest = regs[coordinate_count as usize..].iter().copied();
    let lod = match lod_mode {
        0 => Lod::Implicit,
        1 => Lod::Zero,
        2 => Lod::Bias(tx.read_f(rest.next().unwrap_or(RZ))),
        _ => Lod::Level(tx.read_f(rest.next().unwrap_or(RZ))),
    };
    let offset = if aoffi {
        let reg = rest.next().unwrap_or(RZ);
        let packed = offsets(tx, reg)?;
        Some(texel_offsets(tx, packed, n as usize, 4, 4))
    } else {
        None
    };
    let dref = if shadow {
        Some(tx.read_f(rest.next().unwrap_or(RZ)))
    } else {
        None
    };
    let result = sample(
        tx,
        &Sample {
            tex: tex_ref,
            dim,
            arrayed,
            shadow,
            coords,
            layer,
            lod,
            dref,
            offset,
        },
    )?;
    results(tx, i, result, mask);
    Ok(())
}

/// `TLD`: one texel, by integer coordinates, at level zero (`.LZ`) or at a level.
pub(super) fn tld(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let Some((dim, arrayed)) = dimension(inst.field(61, 3)) else {
        return tx.fail(format!("TLD of dimension code {}", inst.field(61, 3)));
    };
    if dim == Dim::Cube || (arrayed && dim == Dim::D1) {
        return tx.fail("TLD of a cube map or a 1D array");
    }
    let mask = inst.field(72, 4);
    if mask == 0 {
        return tx.fail("TLD with no component selected");
    }
    if inst.bit(76) || inst.bit(78) {
        return tx.fail("TLD with texel offsets or of a multisampled texture");
    }
    let level_operand = match inst.field(87, 3) {
        1 => false,
        3 => true,
        m => return tx.fail(format!("TLD with level of detail mode {m}")),
    };
    let tex_ref = texture(tx, i)?;
    let n = coordinates_of(dim);
    let coordinate_count = n + u32::from(arrayed);
    let regs = operands(tx, i, coordinate_count, u32::from(level_operand))?;
    let (var, sampled_ty) = sampler(tx, tex_ref, dim, arrayed, false)?;
    let sampled = tx.f.load(&mut tx.m, sampled_ty, var);
    let f = tx.t.f32;
    let dim_code = match dim {
        Dim::D1 => 0,
        Dim::D2 => 1,
        _ => 2,
    };
    let image_ty = tx.m.ty_image(f, dim_code, 0, u32::from(arrayed), 0, 1, 0);
    let image =
        tx.f.emit_result(&mut tx.m, sop::IMAGE, image_ty, &[sampled]);
    let int = tx.t.i32;
    // The coordinates, then the layer (which comes first in the registers).
    let mut parts: Vec<Id> = Vec::new();
    let base = usize::from(arrayed);
    for k in 0..n as usize {
        let v = tx.read(regs[base + k]);
        parts.push(tx.as_i32(v));
    }
    if arrayed {
        let v = tx.read(regs[0]);
        parts.push(tx.as_i32(v));
    }
    let coord = if parts.len() == 1 {
        parts[0]
    } else {
        let ty = tx.m.ty_vector(int, parts.len() as u32);
        tx.f.emit_result(&mut tx.m, sop::COMPOSITE_CONSTRUCT, ty, &parts)
    };
    let level = if level_operand {
        let v = tx.read(regs[coordinate_count as usize]);
        tx.as_i32(v)
    } else {
        tx.ci32(0)
    };
    let v4 = tx.t.v4f32;
    let result =
        tx.f.emit_result(&mut tx.m, sop::IMAGE_FETCH, v4, &[image, coord, 0x2, level]);
    results(tx, i, result, mask);
    Ok(())
}

/// `TLD4`: the component bits 87..88 name of the four texels a bilinear sample would read (or four
/// depth comparisons).
pub(super) fn tld4(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let Some((dim, arrayed)) = dimension(inst.field(61, 3)) else {
        return tx.fail(format!("TLD4 of dimension code {}", inst.field(61, 3)));
    };
    if dim != Dim::D2 {
        return tx.fail(format!("TLD4 of a {dim:?} texture"));
    }
    if inst.field(76, 2) != 0 {
        return tx.fail("TLD4 with texel offsets");
    }
    let mask = inst.field(72, 4);
    let shadow = inst.bit(78);
    let tex_ref = texture(tx, i)?;
    let coordinate_count = 2 + u32::from(arrayed);
    let regs = operands(tx, i, coordinate_count, u32::from(shadow))?;
    let f = tx.t.f32;
    let base = usize::from(arrayed);
    let coords = vec![tx.read_f(regs[base]), tx.read_f(regs[base + 1])];
    let layer = if arrayed {
        let raw = tx.read(regs[0]);
        Some(tx.op1(sop::CONVERT_U_TO_F, f, raw))
    } else {
        None
    };
    let dref = shadow.then(|| tx.read_f(regs[coordinate_count as usize]));
    let s = Sample {
        tex: tex_ref,
        dim,
        arrayed,
        shadow,
        coords,
        layer,
        lod: Lod::Zero,
        dref,
        offset: None,
    };
    let (var, sampled_ty) = sampler(tx, tex_ref, dim, arrayed, shadow)?;
    let coord = coordinate(tx, &s);
    let image = tx.f.load(&mut tx.m, sampled_ty, var);
    let v4 = tx.t.v4f32;
    let result = if let Some(dref) = dref {
        tx.f.emit_result(&mut tx.m, sop::IMAGE_DREF_GATHER, v4, &[image, coord, dref])
    } else {
        let comp = tx.c32(inst.field(87, 2) as u32);
        tx.f.emit_result(&mut tx.m, sop::IMAGE_GATHER, v4, &[image, coord, comp])
    };
    results(tx, i, result, if mask == 0 { 0xF } else { mask });
    Ok(())
}

/// `TMML`: the level of detail a sample at the coordinates would use, as fixed point numbers with
/// eight fractional bits: the level before the sampler's limits (signed) and the one that would be
/// read (as Maxwell's, A-106). Only a pixel program has the derivatives it takes.
pub(super) fn tmml(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if tx.cfg.stage != super::super::Stage::Pixel {
        return tx.fail("TMML outside a pixel program");
    }
    let Some((dim, arrayed)) = dimension(inst.field(61, 3)) else {
        return tx.fail(format!("TMML of dimension code {}", inst.field(61, 3)));
    };
    let mask = match inst.field(72, 4) {
        0 => 0b11,
        m if m < 4 => m,
        m => return tx.fail(format!("TMML with the component mask {m:#x}")),
    };
    let tex_ref = texture(tx, i)?;
    // The coordinates are the ones its own dimension says; the texture is what the program's
    // samples say it is (a 2D array for a TMML that says 2D, A-244).
    let n = coordinates_of(dim);
    let regs = operands(tx, i, n + u32::from(arrayed), 0)?;
    let base = usize::from(arrayed);
    let coords: Vec<Id> = (0..n as usize).map(|k| tx.read_f(regs[base + k])).collect();
    let (dim, arrayed) = tx
        .texture_dims
        .get(&tex_ref.index)
        .copied()
        .unwrap_or((dim, arrayed));
    let s = Sample {
        tex: tex_ref,
        dim,
        arrayed: false,
        shadow: false,
        coords,
        layer: None,
        lod: Lod::Implicit,
        dref: None,
        offset: None,
    };
    // The layer does not change the level of detail: the query takes the coordinates alone.
    let (var, sampled_ty) = sampler(tx, tex_ref, dim, arrayed, false)?;
    let coord = coordinate(tx, &s);
    let sampled = tx.f.load(&mut tx.m, sampled_ty, var);
    tx.m.capability(cap::IMAGE_QUERY);
    let v2 = tx.t.v2f32;
    let lod =
        tx.f.emit_result(&mut tx.m, sop::IMAGE_QUERY_LOD, v2, &[sampled, coord]);
    let f = tx.t.f32;
    let read =
        tx.f.emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, f, &[lod, 0]);
    let unclamped =
        tx.f.emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, f, &[lod, 1]);
    let scale = tx.cf32(256.0);
    let (it, u) = (tx.t.i32, tx.t.u32);
    let unclamped = tx.fmul(unclamped, scale);
    let read = tx.fmul(read, scale);
    let unclamped = tx.op1(sop::CONVERT_F_TO_S, it, unclamped);
    let unclamped = tx.as_u32(unclamped);
    let read = tx.op1(sop::CONVERT_F_TO_U, u, read);
    let mut at = 0;
    for (c, value) in [unclamped, read].into_iter().enumerate() {
        if mask >> c & 1 != 0 {
            write(tx, i, reg_at(i, inst.dst(), at), value);
            at += 1;
        }
    }
    Ok(())
}

/// `TXQ`: the size of a level of the texture (width, height, depth or layers) and its number of
/// levels, the components bits 72..75 select.
pub(super) fn txq(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.field(62, 2) != 0 {
        return tx.fail(format!("TXQ of kind {}", inst.field(62, 2)));
    }
    let mask = inst.field(72, 4);
    let tex_ref = texture(tx, i)?;
    let (dim, arrayed) = tx
        .texture_dims
        .get(&tex_ref.index)
        .copied()
        .unwrap_or((Dim::D2, false));
    let (var, sampled_ty) = sampler(tx, tex_ref, dim, arrayed, false)?;
    tx.m.capability(cap::IMAGE_QUERY);
    let sampled = tx.f.load(&mut tx.m, sampled_ty, var);
    let f = tx.t.f32;
    let dim_code = match dim {
        Dim::D1 => 0,
        Dim::D2 => 1,
        Dim::D3 => 2,
        Dim::Cube => 3,
    };
    let image_ty = tx.m.ty_image(f, dim_code, 0, u32::from(arrayed), 0, 1, 0);
    let image =
        tx.f.emit_result(&mut tx.m, sop::IMAGE, image_ty, &[sampled]);
    let level = tx.read(inst.reg(24));
    let int = tx.t.i32;
    let components = match dim {
        Dim::D1 => 1 + u32::from(arrayed),
        Dim::D2 | Dim::Cube => 2 + u32::from(arrayed),
        Dim::D3 => 3,
    };
    let size_ty = if components == 1 {
        int
    } else {
        tx.m.ty_vector(int, components)
    };
    let level = tx.as_i32(level);
    let size = tx.f.emit_result(
        &mut tx.m,
        sop::IMAGE_QUERY_SIZE_LOD,
        size_ty,
        &[image, level],
    );
    let levels =
        tx.f.emit_result(&mut tx.m, sop::IMAGE_QUERY_LEVELS, int, &[image]);
    let one = tx.c32(1);
    let mut values = Vec::new();
    for k in 0..3u32 {
        values.push(if k < components {
            if components == 1 {
                tx.as_u32(size)
            } else {
                let c =
                    tx.f.emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, int, &[size, k]);
                tx.as_u32(c)
            }
        } else {
            one
        });
    }
    values.push(tx.as_u32(levels));
    let mut at = 0;
    for (c, v) in values.into_iter().enumerate() {
        if mask >> c & 1 != 0 {
            write(tx, i, reg_at(i, inst.dst(), at), v);
            at += 1;
        }
    }
    Ok(())
}
