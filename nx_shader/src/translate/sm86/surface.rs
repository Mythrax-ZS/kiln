//! Surfaces, the images a program loads and stores texels of: `SULD.P` and `SUST.P`.
//!
//! The bound forms (opcodes 0xb97 and 0xb9b, the bindless ones less one with bit 11 set, as the
//! texture instructions' are) name the image by a word of a constant buffer that holds its handle:
//! bits 40..53 the word, 54..58 the slot. The rest are NAK's fields of `SULD` and `SUST`: the
//! destination (`SULD`) in bits 16..23, the integer coordinates from the register of 24..31 (the
//! array layer after them), the data (`SUST`) from 32..39, the dimension in 61..63 (0 1D, 1 1D
//! buffer, 2 1D array, 3 2D, 4 2D array, 5 3D), the components in 72..75 (R, RG or RGBA: they go
//! to and come from consecutive registers) and, for `SULD`, the predicate a fault sets in 81..83
//! (none is: an access outside the image reads zeros and writes nothing on the host).
//!
//! A formatted access converts between the image's format and 32-bit components; the components
//! are floats here (docs/ASSUMPTIONS.md A-252): the renderer gives such a program images of
//! float, unsigned-normalized or signed-normalized formats only.

use super::super::emit::{Tx, R};
use super::super::{Dim, ImageUse};
use super::{reg_at, write, I};
use crate::sm86::{op, PT, RZ};
use crate::spirv::{cap, deco, op as sop, storage, Id};

/// The dimension and arrayness of bits 61..63; a buffer is not translated.
fn dimension(code: u64) -> Option<(Dim, bool)> {
    Some(match code {
        0 => (Dim::D1, false),
        2 => (Dim::D1, true),
        3 => (Dim::D2, false),
        4 => (Dim::D2, true),
        5 => (Dim::D3, false),
        _ => return None,
    })
}

/// The number of components bits 72..75 select.
fn components(tx: &Tx<'_>, mask: u64) -> R<u32> {
    match mask {
        0x1 => Ok(1),
        0x3 => Ok(2),
        0xF => Ok(4),
        m => tx.fail(format!("a surface access of the components {m:#x}")),
    }
}

/// How an instruction uses a surface.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Load,
    Store,
    /// An atomic operation on its 32-bit unsigned texels.
    Atomic,
}

/// The storage image variable of the image a handle names, made on first use: its type and the
/// variable.
/// The numbers a surface's texels are: floats, or (A-272) unsigned or signed integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Numbers {
    Float,
    Unsigned,
    Signed,
}

fn image(tx: &mut Tx<'_>, i: &I, access: Access) -> R<(Id, Id, Dim, bool, Numbers)> {
    let inst = i.inst;
    if !matches!(inst.opcode(), op::SULD_BOUND | op::SUST_BOUND | op::SURED) {
        return tx.fail("a bindless surface instruction");
    }
    let read = access == Access::Load;
    let atomic = access == Access::Atomic;
    let Some((dim, arrayed)) = dimension(inst.field(61, 3)) else {
        return tx.fail(format!("a surface of dimension code {}", inst.field(61, 3)));
    };
    let index = inst.field(40, 14) as u32;
    let source = Some(super::slot_of_texture(inst));
    let dim_code = match dim {
        Dim::D1 => {
            tx.m.capability(cap::IMAGE_1D);
            0
        }
        Dim::D2 => 1,
        Dim::D3 => 2,
        Dim::Cube => 3,
    };
    let known = tx
        .info
        .images
        .iter()
        .position(|u| (u.index, u.source) == (index, source));
    let slot = known.map_or(tx.info.images.len(), |k| tx.info.images[k].slot as usize);
    let numbers = if slot < 16 && tx.cfg.uint_images >> slot & 1 != 0 {
        Numbers::Unsigned
    } else if slot < 16 && tx.cfg.sint_images >> slot & 1 != 0 {
        Numbers::Signed
    } else {
        Numbers::Float
    };
    // An atomic operation needs the image's format, R32ui (33); the loads and stores have none.
    let (sampled, format) = if atomic {
        (tx.t.u32, 33)
    } else {
        tx.m.capability(if read {
            cap::STORAGE_IMAGE_READ_WITHOUT_FORMAT
        } else {
            cap::STORAGE_IMAGE_WRITE_WITHOUT_FORMAT
        });
        let sampled = match numbers {
            Numbers::Float => tx.t.f32,
            Numbers::Unsigned => tx.t.u32,
            Numbers::Signed => tx.t.i32,
        };
        (sampled, 0)
    };
    let ty =
        tx.m.ty_image(sampled, dim_code, 0, u32::from(arrayed), 0, 2, format);
    if let Some(k) = known {
        let used = &mut tx.info.images[k];
        if (used.dim, used.arrayed, used.atomic) != (dim, arrayed, atomic) {
            return tx.fail(format!(
                "surface {index} is used as two different kinds of image"
            ));
        }
        if read {
            used.read = true;
        } else {
            used.written = true;
        }
        let var = tx.images[&(index, source)];
        return Ok((ty, var, dim, arrayed, numbers));
    }
    let slot = tx.info.images.len() as u32;
    if slot >= 16 {
        return tx.fail("more than 16 surfaces in one program");
    }
    let ptr = tx.m.ty_pointer(storage::UNIFORM_CONSTANT, ty);
    let var = tx.m.global_variable(ptr, storage::UNIFORM_CONSTANT);
    tx.m.decorate(var, deco::DESCRIPTOR_SET, &[0]);
    tx.m.decorate(var, deco::BINDING, &[ImageUse::binding(tx.cfg.stage, slot)]);
    tx.interface.push(var);
    tx.images.insert((index, source), var);
    tx.info.images.push(ImageUse {
        index,
        source,
        slot,
        dim,
        arrayed,
        read: read || atomic,
        written: !read,
        atomic,
    });
    Ok((ty, var, dim, arrayed, numbers))
}

/// The vector type of a texel of `numbers`.
fn texel_type(tx: &mut Tx<'_>, numbers: Numbers) -> Id {
    match numbers {
        Numbers::Float => tx.t.v4f32,
        Numbers::Unsigned => tx.m.ty_vector(tx.t.u32, 4),
        Numbers::Signed => tx.m.ty_vector(tx.t.i32, 4),
    }
}

/// The coordinates: the integers of the register run from bits 24..31, the layer last.
fn coordinates(tx: &mut Tx<'_>, i: &I, dim: Dim, arrayed: bool) -> Id {
    let n = match dim {
        Dim::D1 => 1,
        Dim::D2 => 2,
        Dim::D3 | Dim::Cube => 3,
    } + u32::from(arrayed);
    let first = i.inst.reg(24);
    let parts: Vec<Id> = (0..n).map(|k| tx.read(reg_at(i, first, k))).collect();
    if n == 1 {
        return parts[0];
    }
    let u = tx.t.u32;
    let ty = tx.m.ty_vector(u, n);
    tx.f.emit_result(&mut tx.m, sop::COMPOSITE_CONSTRUCT, ty, &parts)
}

/// `SULD.P.dim.components Rd, [Ra], c[slot][word]`: load a texel's components.
pub(super) fn suld(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let n = components(tx, i.inst.field(72, 4))?;
    let (ty, var, dim, arrayed, numbers) = image(tx, i, Access::Load)?;
    let coord = coordinates(tx, i, dim, arrayed);
    let handle = tx.f.load(&mut tx.m, ty, var);
    let v4 = texel_type(tx, numbers);
    let texel =
        tx.f.emit_result(&mut tx.m, sop::IMAGE_READ, v4, &[handle, coord]);
    let scalar = match numbers {
        Numbers::Float => tx.t.f32,
        Numbers::Unsigned => tx.t.u32,
        Numbers::Signed => tx.t.i32,
    };
    let u = tx.t.u32;
    let d = i.inst.dst();
    for c in 0..n {
        let v =
            tx.f.emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, scalar, &[texel, c]);
        let bits = if numbers == Numbers::Unsigned {
            v
        } else {
            tx.op1(sop::BITCAST, u, v)
        };
        if d != RZ {
            write(tx, i, reg_at(i, d, c), bits);
        }
    }
    // No access faults on the host.
    let fault = i.inst.field(81, 3) as u8;
    if fault != PT {
        let no = tx.m.const_bool(false);
        super::write_pred(tx, i, fault, no);
    }
    Ok(())
}

/// `SUST.P.dim.components [Ra], Rb, c[slot][word]`: store a texel's components (those not given
/// are zero).
pub(super) fn sust(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let n = components(tx, i.inst.field(72, 4))?;
    let (ty, var, dim, arrayed, numbers) = image(tx, i, Access::Store)?;
    let coord = coordinates(tx, i, dim, arrayed);
    let data = i.inst.reg(32);
    let scalar = match numbers {
        Numbers::Float => tx.t.f32,
        Numbers::Unsigned => tx.t.u32,
        Numbers::Signed => tx.t.i32,
    };
    let zero_bits = tx.c32(0);
    let zero = if numbers == Numbers::Unsigned {
        zero_bits
    } else {
        tx.op1(sop::BITCAST, scalar, zero_bits)
    };
    let parts: Vec<Id> = (0..4)
        .map(|c| {
            if c < n {
                let v = tx.read(reg_at(i, data, c));
                if numbers == Numbers::Unsigned {
                    v
                } else {
                    tx.op1(sop::BITCAST, scalar, v)
                }
            } else {
                zero
            }
        })
        .collect();
    let v4 = texel_type(tx, numbers);
    let texel =
        tx.f.emit_result(&mut tx.m, sop::COMPOSITE_CONSTRUCT, v4, &parts);
    super::mem::guarded(tx, i.cond, |tx| {
        let handle = tx.f.load(&mut tx.m, ty, var);
        tx.f.emit(sop::IMAGE_WRITE, &[handle, coord, texel]);
        Ok(())
    })
}

/// `SURED.op.type [Ra], Rb, c[slot][word]`: an atomic operation on a texel, with no result (NAK's
/// fields as `SUATOM`'s: the operation in bits 87..89, the type in 73..75; the title has only
/// `.ADD.U32`, NVN2 guide 9.2). The image's texels are 32-bit unsigned integers.
pub(super) fn sured(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.field(87, 3) != 0 || inst.field(73, 3) != 0 {
        return tx.fail(format!(
            "SURED of operation {} on type {}",
            inst.field(87, 3),
            inst.field(73, 3)
        ));
    }
    let (_, var, dim, arrayed, _) = image(tx, i, Access::Atomic)?;
    let coord = coordinates(tx, i, dim, arrayed);
    let value = tx.read(inst.reg(32));
    super::mem::guarded(tx, i.cond, |tx| {
        let u = tx.t.u32;
        let ptr_ty = tx.m.ty_pointer(storage::IMAGE, u);
        let zero = tx.c32(0);
        let texel = tx.f.emit_result(
            &mut tx.m,
            sop::IMAGE_TEXEL_POINTER,
            ptr_ty,
            &[var, coord, zero],
        );
        let scope = tx.c32(1);
        let relaxed = tx.c32(0);
        tx.f.emit_result(
            &mut tx.m,
            sop::ATOMIC_I_ADD,
            u,
            &[texel, scope, relaxed, value],
        );
        Ok(())
    })
}
