//! What a program reads and writes besides registers: attributes, constant buffers, special
//! registers, and (in the harness mode) the buffers registers go in and out through.
//!
//! The attribute address space is the one of the hardware: byte `0x80 + 16 i + 4 c` is component `c`
//! of generic attribute `i`; `0x70..0x80` is the position; `0x6c` the point size; `0x64` the layer
//! and `0x68` the viewport index; `0x2c0..0x2e0` the clip distances; `0x2f8` and `0x2fc` the
//! instance and vertex ids; `0x3fc` whether the fragment faces the viewer.

use super::emit::{err, Tx, R};
use super::{Class, Stage};
use crate::inst::{Inst, RZ};
use crate::ops::Op;
use crate::sph::PixelImap;
use crate::spirv::{builtin, cap, deco, op, storage, Id};
use std::collections::BTreeMap;

/// Variables of the interface, made on first use.
#[derive(Default)]
pub(super) struct Io {
    /// Vertex stage: the input `vec4` of each generic attribute location.
    inputs: [Option<Id>; 32],
    /// Pixel stage: the scalar input of each (location, component).
    frag_inputs: BTreeMap<(u32, u32), Id>,
    vertex_index: Option<Id>,
    instance_index: Option<Id>,
    /// The push constant block holding the instance counter of the draw.
    instance_counter: Option<Id>,
    base_vertex: Option<Id>,
    base_instance: Option<Id>,
    frag_coord: Option<Id>,
    front_facing: Option<Id>,
    point_coord: Option<Id>,
    /// Shadow variables of the words an `AST` wrote, by word address (byte address / 4).
    outs: BTreeMap<u32, Id>,
    /// The harness mode's buffers.
    test: Option<TestBuffers>,
}

struct TestBuffers {
    gid: Id,
    regs_out: Id,
    attr_in: Id,
    attr_out: Id,
}

fn class_type(tx: &Tx<'_>, class: Class) -> Id {
    match class {
        Class::Float => tx.t.v4f32,
        Class::Sint => tx.t.v4i32,
        Class::Uint => tx.t.v4u32,
    }
}

fn scalar_type(tx: &Tx<'_>, class: Class) -> Id {
    match class {
        Class::Float => tx.t.f32,
        Class::Sint => tx.t.i32,
        Class::Uint => tx.t.u32,
    }
}

/// A read-only or writable `uint[]` storage buffer at a binding.
pub(super) fn storage_buffer(tx: &mut Tx<'_>, binding: u32, writable: bool) -> Id {
    let u = tx.t.u32;
    let rt = tx.m.ty_runtime_array(u, 4);
    let block = tx.m.ty_struct(&[rt]);
    tx.m.decorate(block, deco::BLOCK, &[]);
    tx.m.member_decorate(block, 0, deco::OFFSET, &[0]);
    if !writable {
        tx.m.member_decorate(block, 0, deco::NON_WRITABLE, &[]);
    }
    let ptr = tx.m.ty_pointer(storage::STORAGE_BUFFER, block);
    let var = tx.m.global_variable(ptr, storage::STORAGE_BUFFER);
    tx.m.decorate(var, deco::DESCRIPTOR_SET, &[0]);
    tx.m.decorate(var, deco::BINDING, &[binding]);
    tx.interface.push(var);
    var
}

/// A pointer to word `index` of a `uint[]` storage buffer made by [`storage_buffer`].
pub(super) fn word_ptr(tx: &mut Tx<'_>, buffer: Id, index: Id) -> Id {
    let zero = tx.c32(0);
    let ptr_ty = tx.m.ty_pointer(storage::STORAGE_BUFFER, tx.t.u32);
    tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[buffer, zero, index])
}

/// Start of the program: in the harness mode, the registers come from the input buffer.
pub(super) fn begin(tx: &mut Tx<'_>) -> R<()> {
    if tx.cfg.stage != Stage::Compute {
        return Ok(());
    }
    tx.m.capability(cap::SHADER);
    let uvec3 = tx.m.ty_vector(tx.t.u32, 3);
    let in_ptr = tx.m.ty_pointer(storage::INPUT, uvec3);
    let gid = tx.m.global_variable(in_ptr, storage::INPUT);
    tx.m.decorate(gid, deco::BUILT_IN, &[builtin::GLOBAL_INVOCATION_ID]);
    tx.interface.push(gid);
    let b = super::test_bindings::REGS_IN;
    let regs_in = storage_buffer(tx, b, false);
    let regs_out = storage_buffer(tx, super::test_bindings::REGS_OUT, true);
    let attr_in = storage_buffer(tx, super::test_bindings::ATTR_IN, false);
    let attr_out = storage_buffer(tx, super::test_bindings::ATTR_OUT, true);
    tx.io.test = Some(TestBuffers {
        gid,
        regs_out,
        attr_in,
        attr_out,
    });
    let base = invocation_base(tx, super::test_bindings::REGS);
    for r in 0..15u8 {
        let index = tx.c32(u32::from(r));
        let at = tx.iadd(base, index);
        let ptr = word_ptr(tx, regs_in, at);
        let u = tx.t.u32;
        let value = tx.f.load(&mut tx.m, u, ptr);
        tx.write_raw(r, value);
    }
    let preds_ix = tx.c32(15);
    let at = tx.iadd(base, preds_ix);
    let ptr = word_ptr(tx, regs_in, at);
    let u = tx.t.u32;
    let bits = tx.f.load(&mut tx.m, u, ptr);
    for p in 0..7u8 {
        let mask = tx.c32(1 << p);
        let masked = tx.and(bits, mask);
        let zero = tx.c32(0);
        let set = tx.cmp(op::I_NOT_EQUAL, masked, zero);
        tx.write_pred_raw(p, set);
    }
    Ok(())
}

/// `gid.x * words_per_invocation`.
fn invocation_base(tx: &mut Tx<'_>, words: u32) -> Id {
    let gid = tx.io.test.as_ref().expect("harness mode").gid;
    let uvec3 = tx.m.ty_vector(tx.t.u32, 3);
    let g = tx.f.load(&mut tx.m, uvec3, gid);
    let u = tx.t.u32;
    let x =
        tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[g, 0]);
    let n = tx.c32(words);
    tx.imul(x, n)
}

/// End of the program: the outputs of the stage are written from registers and shadow variables.
pub(super) fn epilogue(tx: &mut Tx<'_>) -> R<()> {
    match tx.cfg.stage {
        Stage::Compute => epilogue_test(tx),
        Stage::Vertex => epilogue_vertex(tx),
        Stage::Pixel => epilogue_pixel(tx),
    }
}

fn epilogue_test(tx: &mut Tx<'_>) -> R<()> {
    let regs_out = tx.io.test.as_ref().expect("harness mode").regs_out;
    let base = invocation_base(tx, super::test_bindings::REGS);
    for r in 0..15u8 {
        let value = tx.read(r);
        let index = tx.c32(u32::from(r));
        let at = tx.iadd(base, index);
        let ptr = word_ptr(tx, regs_out, at);
        tx.f.store(ptr, value);
    }
    let mut bits = tx.c32(0);
    for p in 0..7u8 {
        let set = tx.read_pred(p, false);
        let one = tx.c32(1 << p);
        let zero = tx.c32(0);
        let v = tx.select_u32(set, one, zero);
        bits = tx.or(bits, v);
    }
    let index = tx.c32(15);
    let at = tx.iadd(base, index);
    let ptr = word_ptr(tx, regs_out, at);
    tx.f.store(ptr, bits);
    Ok(())
}

/// An `Output` variable.
fn output_variable(tx: &mut Tx<'_>, ty: Id) -> Id {
    let ptr = tx.m.ty_pointer(storage::OUTPUT, ty);
    let var = tx.m.global_variable(ptr, storage::OUTPUT);
    tx.interface.push(var);
    var
}

fn epilogue_vertex(tx: &mut Tx<'_>) -> R<()> {
    let outs: Vec<(u32, Id)> = tx.io.outs.iter().map(|(w, v)| (*w, *v)).collect();
    let word =
        |outs: &[(u32, Id)], addr: u32| outs.iter().find(|(w, _)| *w == addr / 4).map(|(_, v)| *v);
    let f = tx.t.f32;
    let u = tx.t.u32;
    // Position, with the depth range moved from -w..w to 0..w when asked.
    let pos: [Option<Id>; 4] = [0x70, 0x74, 0x78, 0x7c].map(|a| word(&outs, a));
    if pos.iter().any(Option::is_some) {
        let zero = tx.cf32(0.0);
        let one = tx.cf32(1.0);
        let mut c = [zero, zero, zero, one];
        for (i, v) in pos.iter().enumerate() {
            if let Some(v) = v {
                let raw = tx.f.load(&mut tx.m, u, *v);
                c[i] = tx.as_f32(raw);
            }
        }
        if tx.cfg.depth_minus_one_to_one {
            let sum = tx.fadd(c[2], c[3]);
            let half = tx.cf32(0.5);
            c[2] = tx.fmul(sum, half);
        }
        let v4 = tx.t.v4f32;
        let value = tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v4, &c);
        let var = output_variable(tx, v4);
        tx.m.decorate(var, deco::BUILT_IN, &[builtin::POSITION]);
        tx.f.store(var, value);
    }
    if let Some(v) = word(&outs, 0x6c) {
        tx.info.writes_point_size = true;
        let raw = tx.f.load(&mut tx.m, u, v);
        let value = tx.as_f32(raw);
        let var = output_variable(tx, f);
        tx.m.decorate(var, deco::BUILT_IN, &[builtin::POINT_SIZE]);
        tx.f.store(var, value);
    }
    for (addr, builtin_id, flag) in [
        (0x64, builtin::LAYER, 0),
        (0x68, builtin::VIEWPORT_INDEX, 1),
    ] {
        if let Some(v) = word(&outs, addr) {
            tx.m.capability(cap::SHADER_VIEWPORT_INDEX_LAYER);
            tx.m.extension("SPV_EXT_shader_viewport_index_layer");
            if flag == 0 {
                tx.info.writes_layer = true;
            } else {
                tx.info.writes_viewport_index = true;
            }
            let i = tx.t.i32;
            let raw = tx.f.load(&mut tx.m, u, v);
            let value = tx.as_i32(raw);
            let var = output_variable(tx, i);
            tx.m.decorate(var, deco::BUILT_IN, &[builtin_id]);
            tx.f.store(var, value);
        }
    }
    // Generic attributes: a vec4 of floats per location with anything written.
    for loc in 0..32u32 {
        let base = 0x80 + loc * 16;
        let comps: [Option<Id>; 4] = [0, 4, 8, 12].map(|d| word(&outs, base + d));
        if comps.iter().all(Option::is_none) {
            continue;
        }
        let zero = tx.cf32(0.0);
        let mut c = [zero; 4];
        let mut mask = 0u8;
        for (i, v) in comps.iter().enumerate() {
            if let Some(v) = v {
                let raw = tx.f.load(&mut tx.m, u, *v);
                c[i] = tx.as_f32(raw);
                mask |= 1 << i;
            }
        }
        tx.info.outputs_written[loc as usize] = mask;
        let v4 = tx.t.v4f32;
        let value = tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v4, &c);
        let var = output_variable(tx, v4);
        tx.m.decorate(var, deco::LOCATION, &[loc]);
        tx.f.store(var, value);
    }
    Ok(())
}

fn epilogue_pixel(tx: &mut Tx<'_>) -> R<()> {
    let Some(sph) = tx.sph else {
        return err("a pixel shader without its header");
    };
    // Registers are assigned to the enabled components of the enabled targets in order, then the
    // sample mask and the depth.
    let mut next = 0u32;
    for target in 0..8usize {
        let enabled: [bool; 4] = [0, 1, 2, 3].map(|c| sph.omap_target(target, c));
        if !enabled.iter().any(|e| *e) {
            continue;
        }
        let class = tx.cfg.target_classes[target];
        let scalar = scalar_type(tx, class);
        let zero_bits = tx.c32(0);
        let mut comps = [zero_bits; 4];
        for c in 0..4 {
            if enabled[c] {
                comps[c] = tx.read(next as u8);
                next += 1;
            }
        }
        let converted: Vec<Id> = comps
            .iter()
            .map(|bits| match class {
                Class::Uint => *bits,
                Class::Sint => tx.as_i32(*bits),
                Class::Float => tx.as_f32(*bits),
            })
            .collect();
        let ty = class_type(tx, class);
        let value =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, ty, &converted);
        let _ = scalar;
        let var = output_variable(tx, ty);
        tx.m.decorate(var, deco::LOCATION, &[target as u32]);
        tx.f.store(var, value);
        tx.info.targets_written |= 1 << target;
    }
    if sph.omap_sample_mask() {
        // Not translated yet: the register is skipped so the depth is still found.
        next += 1;
    }
    if sph.omap_depth() {
        tx.info.writes_depth = true;
        let bits = tx.read(next as u8);
        let value = tx.as_f32(bits);
        let var = output_variable(tx, tx.t.f32);
        tx.m.decorate(var, deco::BUILT_IN, &[builtin::FRAG_DEPTH]);
        tx.f.store(var, value);
        let id = tx.f.id;
        tx.m.execution_mode(id, crate::spirv::exec::DEPTH_REPLACING, &[]);
    }
    Ok(())
}

// ---- instructions --------------------------------------------------------------------------

/// Translate an instruction that reads or writes something outside the registers. `false` when
/// the instruction is not one of these.
pub(super) fn translate(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<bool> {
    match opc {
        Op::Ald => ald(tx, inst)?,
        Op::Ast => ast(tx, inst)?,
        Op::Ipa | Op::IpaIdx => ipa(tx, inst, opc)?,
        Op::Ldc => ldc(tx, inst)?,
        Op::S2r => s2r(tx, inst)?,
        Op::Texs => super::tex::texs(tx, inst)?,
        Op::Ldg => super::global::ldg(tx, inst)?,
        Op::Stg => super::global::stg(tx, inst)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// How many words an attribute access moves (bits 47..48: 32, 64, 96 or 128 bits).
fn attribute_words(inst: Inst) -> u32 {
    inst.bits(47, 2) as u32 + 1
}

fn ald(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.ra() != RZ {
        return tx.fail("ALD with a register in the address (indirect attribute access)");
    }
    if inst.bit(31) || inst.bit(32) {
        return tx.fail("ALD of a patch or output attribute");
    }
    let base = inst.bits(20, 10) as u32;
    for i in 0..attribute_words(inst) {
        let value = attribute_in(tx, base + 4 * i)?;
        tx.write(inst, inst.rd().wrapping_add(i as u8), value);
    }
    Ok(())
}

/// The 32 bits of the input attribute at byte address `addr`.
fn attribute_in(tx: &mut Tx<'_>, addr: u32) -> R<Id> {
    if tx.cfg.stage == Stage::Compute {
        let attr_in = tx.io.test.as_ref().expect("harness mode").attr_in;
        let base = invocation_base(tx, super::test_bindings::ATTRS);
        let index = tx.c32(addr / 4);
        let at = tx.iadd(base, index);
        let ptr = word_ptr(tx, attr_in, at);
        let u = tx.t.u32;
        return Ok(tx.f.load(&mut tx.m, u, ptr));
    }
    if tx.cfg.stage != Stage::Vertex {
        return tx.fail("ALD in a stage that is not a vertex shader");
    }
    match addr {
        0x80..=0x27f => {
            let loc = (addr - 0x80) / 16;
            let comp = (addr - 0x80) % 16 / 4;
            let class = tx.cfg.attribute_classes[loc as usize];
            let var = match tx.io.inputs[loc as usize] {
                Some(v) => v,
                None => {
                    let ty = class_type(tx, class);
                    let ptr = tx.m.ty_pointer(storage::INPUT, ty);
                    let v = tx.m.global_variable(ptr, storage::INPUT);
                    tx.m.decorate(v, deco::LOCATION, &[loc]);
                    tx.interface.push(v);
                    tx.io.inputs[loc as usize] = Some(v);
                    v
                }
            };
            tx.info.attributes_read[loc as usize] |= 1 << comp;
            let ty = class_type(tx, class);
            let vec = tx.f.load(&mut tx.m, ty, var);
            let scalar = scalar_type(tx, class);
            let c =
                tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, scalar, &[vec, comp]);
            Ok(match class {
                Class::Uint => c,
                Class::Sint => {
                    let u = tx.t.u32;
                    tx.op1(op::BITCAST, u, c)
                }
                Class::Float => tx.as_u32(c),
            })
        }
        0x2f8 => {
            // The instance id: the counter of the draw (the hardware runs the instances of a draw
            // as a series of draws that count up) plus, for a draw that really is instanced, the
            // instance within it without the base instance.
            let counter = instance_counter(tx);
            let idx = builtin_input(tx, builtin::INSTANCE_INDEX, true, 0);
            let base = builtin_input(tx, builtin::BASE_INSTANCE, true, 1);
            let within = tx.isub(idx, base);
            Ok(tx.iadd(counter, within))
        }
        0x2fc => {
            let idx = builtin_input(tx, builtin::VERTEX_INDEX, true, 2);
            let base = builtin_input(tx, builtin::BASE_VERTEX, true, 3);
            Ok(tx.isub(idx, base))
        }
        _ => tx.fail(format!("ALD of attribute address {addr:#x}")),
    }
}

/// The instance counter of the draw: the 32-bit push constant at offset 0 (see
/// `Info::instance_counter`).
fn instance_counter(tx: &mut Tx<'_>) -> Id {
    let u = tx.t.u32;
    let var = match tx.io.instance_counter {
        Some(v) => v,
        None => {
            let block = tx.m.ty_struct(&[u]);
            tx.m.decorate(block, deco::BLOCK, &[]);
            tx.m.member_decorate(block, 0, deco::OFFSET, &[0]);
            let ptr = tx.m.ty_pointer(storage::PUSH_CONSTANT, block);
            let v = tx.m.global_variable(ptr, storage::PUSH_CONSTANT);
            tx.interface.push(v);
            tx.io.instance_counter = Some(v);
            v
        }
    };
    tx.info.instance_counter = true;
    let zero = tx.c32(0);
    let ptr_ty = tx.m.ty_pointer(storage::PUSH_CONSTANT, u);
    let ptr =
        tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[var, zero]);
    tx.f.load(&mut tx.m, u, ptr)
}

/// Load a scalar `int` built-in input (`which` picks the cache slot of the variable).
fn builtin_input(tx: &mut Tx<'_>, builtin_id: u32, signed: bool, which: usize) -> Id {
    let slot = match which {
        0 => &mut tx.io.instance_index,
        1 => &mut tx.io.base_instance,
        2 => &mut tx.io.vertex_index,
        _ => &mut tx.io.base_vertex,
    };
    let ty = if signed { tx.t.i32 } else { tx.t.u32 };
    let var = match *slot {
        Some(v) => v,
        None => {
            let ptr = tx.m.ty_pointer(storage::INPUT, ty);
            let v = tx.m.global_variable(ptr, storage::INPUT);
            tx.m.decorate(v, deco::BUILT_IN, &[builtin_id]);
            if builtin_id == builtin::BASE_VERTEX || builtin_id == builtin::BASE_INSTANCE {
                tx.m.capability(cap::DRAW_PARAMETERS);
                tx.m.extension("SPV_KHR_shader_draw_parameters");
            }
            tx.interface.push(v);
            match which {
                0 => tx.io.instance_index = Some(v),
                1 => tx.io.base_instance = Some(v),
                2 => tx.io.vertex_index = Some(v),
                _ => tx.io.base_vertex = Some(v),
            }
            v
        }
    };
    let loaded = tx.f.load(&mut tx.m, ty, var);
    let u = tx.t.u32;
    tx.op1(op::BITCAST, u, loaded)
}

fn ast(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.ra() != RZ {
        return tx.fail("AST with a register in the address (indirect attribute access)");
    }
    if inst.bit(31) {
        return tx.fail("AST of a patch attribute");
    }
    let base = inst.bits(20, 10) as u32;
    for i in 0..attribute_words(inst) {
        let value = tx.read(inst.rd().wrapping_add(i as u8));
        attribute_out(tx, inst, base + 4 * i, value)?;
    }
    Ok(())
}

fn attribute_out(tx: &mut Tx<'_>, inst: Inst, addr: u32, value: Id) -> R<()> {
    if tx.cfg.stage == Stage::Compute {
        let attr_out = tx.io.test.as_ref().expect("harness mode").attr_out;
        let base = invocation_base(tx, super::test_bindings::ATTRS);
        let index = tx.c32(addr / 4);
        let at = tx.iadd(base, index);
        let ptr = word_ptr(tx, attr_out, at);
        match tx.guard_condition(inst.guard()) {
            None => tx.f.store(ptr, value),
            Some(cond) => {
                let u = tx.t.u32;
                let old = tx.f.load(&mut tx.m, u, ptr);
                let merged = tx.select_u32(cond, value, old);
                tx.f.store(ptr, merged);
            }
        }
        return Ok(());
    }
    if tx.cfg.stage != Stage::Vertex {
        return tx.fail("AST in a stage that is not a vertex shader");
    }
    let known = matches!(addr, 0x64 | 0x68 | 0x6c | 0x70..=0x7c | 0x80..=0x27f | 0x2c0..=0x2dc);
    if !known || !addr.is_multiple_of(4) {
        return tx.fail(format!("AST to attribute address {addr:#x}"));
    }
    if (0x2c0..=0x2dc).contains(&addr) {
        return tx.fail("AST of a clip distance");
    }
    let word = addr / 4;
    let var = match tx.io.outs.get(&word) {
        Some(v) => *v,
        None => {
            let zero = tx.c32(0);
            let ptr = tx.m.ty_pointer(storage::FUNCTION, tx.t.u32);
            let v = tx.m.id();
            tx.f.local_with_init(ptr, v, zero);
            tx.io.outs.insert(word, v);
            v
        }
    };
    match tx.guard_condition(inst.guard()) {
        None => tx.f.store(var, value),
        Some(cond) => {
            let u = tx.t.u32;
            let old = tx.f.load(&mut tx.m, u, var);
            let merged = tx.select_u32(cond, value, old);
            tx.f.store(var, merged);
        }
    }
    Ok(())
}

/// Attribute interpolation: bits 54..55 pick the mode (0 pass, 1 multiply by a register, 2
/// constant, 3 screen linear), bits 28..37 the attribute address, `Rb` the multiplier.
fn ipa(tx: &mut Tx<'_>, inst: Inst, opc: Op) -> R<()> {
    if opc == Op::IpaIdx {
        return tx.fail("IPA with the attribute address in a register");
    }
    if inst.bits(52, 2) != 0 {
        return tx.fail("IPA with centroid or offset sampling");
    }
    let addr = inst.bits(28, 10) as u32;
    let mode = inst.bits(54, 2);
    let mut value = match tx.cfg.stage {
        Stage::Compute => {
            let bits = attribute_in(tx, addr)?;
            tx.as_f32(bits)
        }
        Stage::Pixel => interpolated(tx, addr)?,
        Stage::Vertex => return tx.fail("IPA in a vertex shader"),
    };
    if mode == 1 {
        let m = tx.read_f(inst.rb());
        value = tx.fmul(value, m);
    }
    if inst.bit(51) {
        let zero = tx.cf32(0.0);
        let one = tx.cf32(1.0);
        let t = tx.t.f32;
        value = tx.ext(t, crate::spirv::glsl::N_CLAMP, &[value, zero, one]);
    }
    tx.write_f(inst, inst.rd(), value);
    Ok(())
}

/// The value the pixel shader's interpolation unit gives for attribute address `addr`.
fn interpolated(tx: &mut Tx<'_>, addr: u32) -> R<Id> {
    match addr {
        0x70 | 0x74 | 0x78 | 0x7c => {
            let fc = frag_coord(tx);
            let v4 = tx.t.v4f32;
            let vec = tx.f.load(&mut tx.m, v4, fc);
            let f = tx.t.f32;
            let comp = (addr - 0x70) / 4;
            Ok(tx
                .f
                .emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[vec, comp]))
        }
        0x80..=0x27f => {
            let loc = (addr - 0x80) / 16;
            let comp = (addr - 0x80) % 16 / 4;
            let Some(sph) = tx.sph else {
                return tx.fail("a pixel shader without its header");
            };
            let imap = sph.pixel_imap(loc as usize, comp as usize);
            let var = match tx.io.frag_inputs.get(&(loc, comp)) {
                Some(v) => *v,
                None => {
                    let f = tx.t.f32;
                    let ptr = tx.m.ty_pointer(storage::INPUT, f);
                    let v = tx.m.global_variable(ptr, storage::INPUT);
                    tx.m.decorate(v, deco::LOCATION, &[loc]);
                    tx.m.decorate(v, deco::COMPONENT, &[comp]);
                    match imap {
                        PixelImap::Constant => tx.m.decorate(v, deco::FLAT, &[]),
                        PixelImap::ScreenLinear => tx.m.decorate(v, deco::NO_PERSPECTIVE, &[]),
                        PixelImap::Perspective | PixelImap::Unused => {}
                    }
                    tx.interface.push(v);
                    tx.io.frag_inputs.insert((loc, comp), v);
                    v
                }
            };
            tx.info.attributes_read[loc as usize] |= 1 << comp;
            let f = tx.t.f32;
            let v = tx.f.load(&mut tx.m, f, var);
            // The interpolation unit gives the plane equation's value, which for a perspective
            // attribute is the attribute divided by w. Vulkan's already divided it back, so
            // undo that with 1/w (FragCoord.w): the program multiplies by w itself.
            if imap == PixelImap::Perspective {
                let fc = frag_coord(tx);
                let v4 = tx.t.v4f32;
                let vec = tx.f.load(&mut tx.m, v4, fc);
                let w =
                    tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[vec, 3]);
                Ok(tx.fmul(v, w))
            } else {
                Ok(v)
            }
        }
        0x2e0 | 0x2e4 => {
            let pc = match tx.io.point_coord {
                Some(v) => v,
                None => {
                    let v2 = tx.t.v2f32;
                    let ptr = tx.m.ty_pointer(storage::INPUT, v2);
                    let v = tx.m.global_variable(ptr, storage::INPUT);
                    tx.m.decorate(v, deco::BUILT_IN, &[builtin::POINT_COORD]);
                    tx.interface.push(v);
                    tx.io.point_coord = Some(v);
                    v
                }
            };
            let v2 = tx.t.v2f32;
            let vec = tx.f.load(&mut tx.m, v2, pc);
            let f = tx.t.f32;
            let comp = (addr - 0x2e0) / 4;
            Ok(tx
                .f
                .emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, f, &[vec, comp]))
        }
        0x3fc => {
            let ff = match tx.io.front_facing {
                Some(v) => v,
                None => {
                    let b = tx.t.bool_;
                    let ptr = tx.m.ty_pointer(storage::INPUT, b);
                    let v = tx.m.global_variable(ptr, storage::INPUT);
                    tx.m.decorate(v, deco::BUILT_IN, &[builtin::FRONT_FACING]);
                    tx.interface.push(v);
                    tx.io.front_facing = Some(v);
                    v
                }
            };
            let b = tx.t.bool_;
            let facing = tx.f.load(&mut tx.m, b, ff);
            let all = tx.c32(0xFFFF_FFFF);
            let none = tx.c32(0);
            let bits = tx.select_u32(facing, all, none);
            Ok(tx.as_f32(bits))
        }
        _ => tx.fail(format!("IPA of attribute address {addr:#x}")),
    }
}

fn frag_coord(tx: &mut Tx<'_>) -> Id {
    if let Some(v) = tx.io.frag_coord {
        return v;
    }
    let v4 = tx.t.v4f32;
    let ptr = tx.m.ty_pointer(storage::INPUT, v4);
    let v = tx.m.global_variable(ptr, storage::INPUT);
    tx.m.decorate(v, deco::BUILT_IN, &[builtin::FRAG_COORD]);
    tx.interface.push(v);
    tx.io.frag_coord = Some(v);
    v
}

/// `LDC.size Rd, c[slot][Ra + imm16]`: a load from a constant buffer, with a register in the
/// address when `Ra` is not `RZ`.
fn ldc(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    if inst.bits(44, 2) != 0 {
        return tx.fail("LDC with the constant buffer index in a register");
    }
    let slot = inst.bits(36, 5) as u32;
    let imm = inst.sbits(20, 16) as i32;
    let size = inst.bits(48, 3);
    // Sizes: 0 u8, 1 s8, 2 u16, 3 s16, 4 b32, 5 b64.
    let words = match size {
        0..=4 => 1,
        5 => 2,
        _ => return tx.fail(format!("LDC of size code {size}")),
    };
    if inst.ra() == RZ {
        if imm < 0 {
            return tx.fail("LDC at a negative offset");
        }
        let off = imm as u32;
        for w in 0..words {
            let word_off = (off & !3) + 4 * w;
            let mut v = tx.cbuf_word(slot, word_off)?;
            if size < 4 {
                v = extract_small(tx, v, off & 3, size);
            }
            tx.write(inst, inst.rd().wrapping_add(w as u8), v);
        }
    } else {
        let base = tx.read(inst.ra());
        let imm_u = tx.c32(imm as u32);
        let addr = tx.iadd(base, imm_u);
        for w in 0..words {
            let add = tx.c32(4 * w);
            let a = tx.iadd(addr, add);
            let aligned = {
                let mask = tx.c32(!3u32);
                tx.and(a, mask)
            };
            let mut v = tx.cbuf_word_dynamic(slot, aligned)?;
            if size < 4 {
                let three = tx.c32(3);
                let byte = tx.and(a, three);
                v = extract_small_dynamic(tx, v, byte, size);
            }
            tx.write(inst, inst.rd().wrapping_add(w as u8), v);
        }
    }
    Ok(())
}

fn extract_small(tx: &mut Tx<'_>, word: Id, byte: u32, size: u64) -> Id {
    let b = tx.c32(byte);
    let eight = tx.c32(8);
    let shift = tx.imul(b, eight);
    extract_small_dynamic_shift(tx, word, shift, size)
}

fn extract_small_dynamic(tx: &mut Tx<'_>, word: Id, byte: Id, size: u64) -> Id {
    let eight = tx.c32(8);
    let shift = tx.imul(byte, eight);
    extract_small_dynamic_shift(tx, word, shift, size)
}

fn extract_small_dynamic_shift(tx: &mut Tx<'_>, word: Id, shift: Id, size: u64) -> Id {
    let shifted = tx.shr(word, shift);
    let (bits, signed) = match size {
        0 => (8, false),
        1 => (8, true),
        2 => (16, false),
        _ => (16, true),
    };
    let zero = tx.c32(0);
    let len = tx.c32(bits);
    let t = tx.t.u32;
    if signed {
        let s = tx.as_i32(shifted);
        let e = tx.op3(op::BIT_FIELD_S_EXTRACT, tx.t.i32, s, zero, len);
        tx.as_u32(e)
    } else {
        tx.op3(op::BIT_FIELD_U_EXTRACT, t, shifted, zero, len)
    }
}

/// `S2R Rd, SR_x`: special registers. The harness mode has the invocation's thread id.
fn s2r(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    let sr = inst.bits(20, 8);
    match (tx.cfg.stage, sr) {
        // SR_TID.X (0x21): the invocation index; Y and Z are zero.
        (Stage::Compute, 0x21) => {
            let gid = tx.io.test.as_ref().expect("harness mode").gid;
            let uvec3 = tx.m.ty_vector(tx.t.u32, 3);
            let g = tx.f.load(&mut tx.m, uvec3, gid);
            let u = tx.t.u32;
            let x =
                tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[g, 0]);
            tx.write(inst, inst.rd(), x);
            Ok(())
        }
        (Stage::Compute, 0x22 | 0x23) => {
            let zero = tx.c32(0);
            tx.write(inst, inst.rd(), zero);
            Ok(())
        }
        _ => tx.fail(format!("S2R of special register {sr:#x}")),
    }
}
