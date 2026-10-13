//! Texture sampling.
//!
//! `TEXS` is the short form NVIDIA's compiler uses for the common samples. Bits 53..56 pick the
//! kind (dimension, how the level of detail is chosen, whether it compares against a depth); the
//! texture is the 13-bit index in bits 36..48, which names a slot of the table of handles the
//! host resolves at draw time (so the translation only declares "sampler number `index`"). The
//! coordinates are in `Ra` (x) and `Rb` (y), with the level of detail or the depth reference in
//! the register after `Rb`. The result goes to `Rd`, `Rd+1` and, when bits 28..35 name a
//! register, `R28`, `R28+1`, the components in the order the mask in bits 50..52 gives.

use super::emit::{Tx, R};
use super::{Dim, TextureUse};
use crate::inst::{Inst, RZ};
use crate::spirv::{cap, deco, op, storage, Id};

/// How the level of detail of a sample is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lod {
    /// From the screen-space derivatives of the coordinates.
    Implicit,
    /// Zero.
    Zero,
    /// From a register.
    Level,
}

/// The kinds of `TEXS`: dimension, array, level of detail, depth compare (bits 53..56).
fn kind(code: u64) -> Option<(Dim, bool, Lod, bool)> {
    Some(match code {
        0 => (Dim::D1, false, Lod::Zero, false),
        1 => (Dim::D2, false, Lod::Implicit, false),
        2 => (Dim::D2, false, Lod::Zero, false),
        3 => (Dim::D2, false, Lod::Level, false),
        4 => (Dim::D2, false, Lod::Implicit, true),
        5 => (Dim::D2, false, Lod::Level, true),
        6 => (Dim::D2, false, Lod::Zero, true),
        7 => (Dim::D2, true, Lod::Implicit, false),
        8 => (Dim::D2, true, Lod::Zero, false),
        9 => (Dim::D2, true, Lod::Zero, true),
        10 => (Dim::D3, false, Lod::Implicit, false),
        11 => (Dim::D3, false, Lod::Zero, false),
        12 => (Dim::Cube, false, Lod::Implicit, false),
        13 => (Dim::Cube, false, Lod::Level, false),
        _ => return None,
    })
}

/// The combined image sampler variable of texture slot `index`, made on first use.
fn sampler(tx: &mut Tx<'_>, index: u32, dim: Dim, arrayed: bool, shadow: bool) -> R<(Id, Id)> {
    if index >= 40 {
        return tx.fail(format!("texture slot {index} (there are 40)"));
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
    if let Some(var) = tx.textures.get(&index) {
        let used = tx.info.textures.iter().find(|t| t.index == index);
        if used.is_some_and(|t| (t.dim, t.arrayed, t.shadow) != (dim, arrayed, shadow)) {
            return tx.fail(format!(
                "texture slot {index} is sampled as two different kinds of image"
            ));
        }
        return Ok((*var, sampled));
    }
    let ptr = tx.m.ty_pointer(storage::UNIFORM_CONSTANT, sampled);
    let var = tx.m.global_variable(ptr, storage::UNIFORM_CONSTANT);
    tx.m.decorate(var, deco::DESCRIPTOR_SET, &[0]);
    let binding = 64 * tx.cfg.stage.index() + 24 + index;
    tx.m.decorate(var, deco::BINDING, &[binding]);
    tx.interface.push(var);
    tx.textures.insert(index, var);
    tx.info.textures.push(TextureUse {
        index,
        dim,
        arrayed,
        shadow,
    });
    Ok((var, sampled))
}

pub(super) fn texs(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    // Bit 59 set means 32-bit results (clear means packed 16-bit ones).
    if !inst.bit(59) {
        return tx.fail("TEXS with 16-bit results");
    }
    let code = inst.bits(53, 4);
    let Some((dim, arrayed, lod, depth)) = kind(code) else {
        return tx.fail(format!("TEXS of kind {code}"));
    };
    if depth || dim == Dim::D3 || dim == Dim::Cube || (arrayed && dim != Dim::D2) {
        return tx.fail(format!(
            "TEXS of kind {code} ({dim:?}{}{}) is not translated yet",
            if arrayed { " array" } else { "" },
            if depth { " with depth compare" } else { "" }
        ));
    }
    let index = inst.bits(36, 13) as u32;
    let (var, sampled_ty) = sampler(tx, index, dim, arrayed, depth)?;
    let f = tx.t.f32;
    let v4 = tx.t.v4f32;
    // The coordinates: x in Ra and y in Rb; for an array, the layer (an integer) is in Ra and x
    // in the register after it.
    let coord = match (dim, arrayed) {
        (Dim::D1, _) => tx.read_f(inst.ra()),
        (_, false) => {
            let x = tx.read_f(inst.ra());
            let y = tx.read_f(inst.rb());
            let v2 = tx.t.v2f32;
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v2, &[x, y])
        }
        (_, true) => {
            let layer = tx.read(inst.ra());
            let layer = tx.op1(op::CONVERT_U_TO_F, f, layer);
            let x = tx.read_f(inst.ra().wrapping_add(1));
            let y = tx.read_f(inst.rb());
            let v3 = tx.t.v3f32;
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v3, &[x, y, layer])
        }
    };
    let image = tx.f.load(&mut tx.m, sampled_ty, var);
    let result = match lod {
        Lod::Implicit => tx.f.emit_result(
            &mut tx.m,
            op::IMAGE_SAMPLE_IMPLICIT_LOD,
            v4,
            &[image, coord],
        ),
        Lod::Zero => {
            let zero = tx.cf32(0.0);
            // ImageOperands Lod (mask 0x2) then the level.
            tx.f.emit_result(
                &mut tx.m,
                op::IMAGE_SAMPLE_EXPLICIT_LOD,
                v4,
                &[image, coord, 0x2, zero],
            )
        }
        Lod::Level => {
            let level = tx.read_f(inst.rb().wrapping_add(1));
            tx.f.emit_result(
                &mut tx.m,
                op::IMAGE_SAMPLE_EXPLICIT_LOD,
                v4,
                &[image, coord, 0x2, level],
            )
        }
    };
    // Which components, and where they go.
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
                    "TEXS component selector {sel} with two destinations"
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
    for (i, c) in comps.iter().enumerate() {
        let value =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[result, *c]);
        let reg = if i < 2 {
            inst.rd().wrapping_add(i as u8)
        } else {
            inst.r28().wrapping_add((i - 2) as u8)
        };
        tx.write_f(inst, reg, value);
    }
    Ok(())
}
