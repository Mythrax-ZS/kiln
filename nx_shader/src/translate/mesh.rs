//! The task and mesh programs of the Switch 2 (docs/ASSUMPTIONS.md A-270), as `TaskEXT` and
//! `MeshEXT` shaders (`SPV_EXT_mesh_shader`).
//!
//! They speak through the stage buffer: `ISBERD` reads it and `ISBEWR` writes it, at plain byte
//! addresses. A task workgroup reads its own number (`ISBERD.ATTR.SKEW [RZ]`) and writes the
//! region it hands its mesh workgroups: their number at byte 4, the payload from byte 16. Here
//! that region is the task payload. A mesh workgroup reads its own number at byte 0 and the
//! payload past it, and writes two regions:
//! - the attribute region: vertices, then primitives, in groups of 32, one line of 32 words per
//!   attribute word the program writes (in the order of its header's output map, a primitive's
//!   in the order of the pipeline's pass-through header), element `i` at word `i % 32` of its
//!   group's line;
//! - the map: the primitive count (a 32-bit write named by its last byte, 3) and the vertex
//!   numbers of each primitive, a byte each, from byte 4.
//!
//! Both live in workgroup memory, and the epilogue, which every invocation reaches, hands them to
//! Vulkan: `SetMeshOutputsEXT` and copies, or `EmitMeshTasksEXT`.

use super::emit::{Tx, R};
use super::Stage;
use crate::spirv::{builtin, deco, op, storage, Id};
use std::collections::BTreeMap;

pub(super) const EXEC_TASK: u32 = 5364;
pub(super) const EXEC_MESH: u32 = 5365;
pub(super) const CAP_MESH_SHADING: u32 = 5283;
pub(super) const DECO_PER_PRIMITIVE: u32 = 5271;
const STORAGE_TASK_PAYLOAD: u32 = 5402;
const BUILTIN_PRIMITIVE_TRIANGLE_INDICES: u32 = 5296;
const OP_EMIT_MESH_TASKS: u16 = 5294;
const OP_SET_MESH_OUTPUTS: u16 = 5295;
pub(super) const MODE_OUTPUT_VERTICES: u32 = 26;
pub(super) const MODE_OUTPUT_PRIMITIVES: u32 = 5270;
pub(super) const MODE_OUTPUT_TRIANGLES: u32 = 5298;
/// `Workgroup` scope, and acquire-release of workgroup memory.
const SCOPE_WORKGROUP: u32 = 2;
const SEMANTICS_WORKGROUP: u32 = 0x8 | 0x100;

/// The variables of the stage buffer and of the outputs, made on first use.
#[derive(Debug, Default)]
pub(super) struct Vars {
    payload: Option<(Id, u32)>,
    attributes: Option<(Id, u32)>,
    map: Option<(Id, u32)>,
    positions: Option<Id>,
    /// The output arrays of attribute words, a vertex's and a primitive's.
    vertex_outputs: BTreeMap<u32, Id>,
    primitive_outputs: BTreeMap<u32, Id>,
    indices: Option<Id>,
}

/// The extension and capability the stages need, said once.
pub(super) fn declare(tx: &mut Tx<'_>) {
    tx.m.extension("SPV_EXT_mesh_shader");
    tx.m.capability(CAP_MESH_SHADING);
}

fn config(tx: &Tx<'_>) -> R<super::MeshConfig> {
    match &tx.cfg.mesh {
        Some(m) => Ok(m.clone()),
        None => Err(super::Error {
            message: "a task or mesh program without its pipeline".into(),
            at: None,
        }),
    }
}

/// A module-scope array of `words` 32-bit words in storage class `class`.
fn words_variable(tx: &mut Tx<'_>, class: u32, words: u32) -> Id {
    let u = tx.t.u32;
    let length = tx.c32(words.max(1));
    let array = tx.m.ty_array(u, length);
    let ptr = tx.m.ty_pointer(class, array);
    let var = tx.m.global_variable(ptr, class);
    tx.interface.push(var);
    var
}

/// A pointer to word `index` of an array of `words`, kept inside it.
fn word(tx: &mut Tx<'_>, class: u32, var: Id, words: u32, index: Id) -> Id {
    let u = tx.t.u32;
    let last = tx.c32(words.max(1) - 1);
    let inside = tx.ext(u, crate::spirv::glsl::U_MIN, &[index, last]);
    let ptr_ty = tx.m.ty_pointer(class, u);
    tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[var, inside])
}

fn payload(tx: &mut Tx<'_>) -> R<(Id, u32)> {
    if let Some(p) = tx.mesh.payload {
        return Ok(p);
    }
    let words = config(tx)?.payload_words;
    declare(tx);
    let var = words_variable(tx, STORAGE_TASK_PAYLOAD, words);
    tx.mesh.payload = Some((var, words.max(1)));
    Ok((var, words.max(1)))
}

/// The attribute words a vertex's lines hold: the bits of the header's output map from bit 400,
/// one per attribute word from address 0.
fn vertex_words(tx: &Tx<'_>) -> Vec<u32> {
    match tx.sph {
        Some(sph) => (400..608)
            .filter(|b| sph.bit(*b))
            .map(|b| b as u32 - 400)
            .collect(),
        None => Vec::new(),
    }
}

/// Where the primitives' lines begin, in words: after every group of vertices.
fn primitive_base(tx: &Tx<'_>, m: &super::MeshConfig) -> u32 {
    m.max_vertices.div_ceil(32) * vertex_words(tx).len() as u32 * 32
}

fn attributes(tx: &mut Tx<'_>) -> R<(Id, u32)> {
    if let Some(a) = tx.mesh.attributes {
        return Ok(a);
    }
    let m = config(tx)?;
    let words = primitive_base(tx, &m)
        + m.max_primitives.div_ceil(32) * m.primitive_words.len() as u32 * 32;
    let var = words_variable(tx, storage::WORKGROUP, words);
    tx.mesh.attributes = Some((var, words.max(1)));
    Ok((var, words.max(1)))
}

/// The map, one word per byte (a byte written alone does not touch its neighbours).
fn map(tx: &mut Tx<'_>) -> R<(Id, u32)> {
    if let Some(m) = tx.mesh.map {
        return Ok(m);
    }
    let bytes = 4 + 3 * config(tx)?.max_primitives + 4;
    let var = words_variable(tx, storage::WORKGROUP, bytes);
    tx.mesh.map = Some((var, bytes));
    Ok((var, bytes))
}

fn workgroup_number(tx: &mut Tx<'_>) -> Id {
    let u = tx.t.u32;
    let uvec3 = tx.m.ty_vector(u, 3);
    let all = super::sm86::builtin_value(tx, builtin::WORKGROUP_ID, uvec3);
    tx.f.emit_result(&mut tx.m, op::COMPOSITE_EXTRACT, u, &[all, 0])
}

fn fail<T>(why: String) -> R<T> {
    Err(super::Error {
        message: why,
        at: None,
    })
}

/// `ISBERD`: the word the stage buffer holds at byte `addr` of its input (region 3, 32 bits).
pub(super) fn read(tx: &mut Tx<'_>, addr: Id, mode: Mode) -> R<Id> {
    if mode.output || mode.region != 3 || !mode.wide {
        return fail(format!("ISBERD of the stage buffer in mode {mode:?}"));
    }
    match tx.cfg.stage {
        // Its own number (`ISBERD.ATTR.SKEW [RZ]`).
        Stage::Task => Ok(workgroup_number(tx)),
        // Its own number at byte 0, the payload past it.
        Stage::Mesh => {
            let number = workgroup_number(tx);
            let two = tx.c32(2);
            let index = tx.shr(addr, two);
            let value = if config(tx)?.task {
                let (var, words) = payload(tx)?;
                let ptr = word(tx, STORAGE_TASK_PAYLOAD, var, words, index);
                let u = tx.t.u32;
                tx.f.load(&mut tx.m, u, ptr)
            } else {
                tx.c32(0)
            };
            let zero = tx.c32(0);
            let b = tx.t.bool_;
            let first = tx.op2(op::I_EQUAL, b, index, zero);
            Ok(tx.select_u32(first, number, value))
        }
        _ => fail("ISBERD in a program that is not a task or mesh program".into()),
    }
}

/// The mode bits of `ISBERD` and `ISBEWR`: 32 bits or a byte, the region, the output buffer.
#[derive(Clone, Copy, Debug)]
pub(super) struct Mode {
    pub wide: bool,
    pub region: u32,
    pub output: bool,
}

/// `ISBEWR`: `value` to byte `addr` of the output of the stage.
pub(super) fn write(tx: &mut Tx<'_>, addr: Id, value: Id, mode: Mode) -> R<()> {
    if !mode.output {
        return fail(format!(
            "ISBEWR to the input of the stage buffer ({mode:?})"
        ));
    }
    let two = tx.c32(2);
    match (tx.cfg.stage, mode.region, mode.wide) {
        (Stage::Task, 3, true) => {
            let (var, words) = payload(tx)?;
            let index = tx.shr(addr, two);
            let ptr = word(tx, STORAGE_TASK_PAYLOAD, var, words, index);
            tx.f.store(ptr, value);
            Ok(())
        }
        (Stage::Mesh, 3, true) => {
            let (var, words) = attributes(tx)?;
            let index = tx.shr(addr, two);
            let ptr = word(tx, storage::WORKGROUP, var, words, index);
            tx.f.store(ptr, value);
            Ok(())
        }
        // A word of the map, named by its last byte: its bytes, the lowest first.
        (Stage::Mesh, 0, true) => {
            let (var, bytes) = map(tx)?;
            let three = tx.c32(3);
            let first = tx.isub(addr, three);
            let mask = tx.c32(0xFF);
            for k in 0..4u32 {
                let kk = tx.c32(k);
                let at = tx.iadd(first, kk);
                let shift = tx.c32(8 * k);
                let shifted = tx.shr(value, shift);
                let byte = tx.and(shifted, mask);
                let ptr = word(tx, storage::WORKGROUP, var, bytes, at);
                tx.f.store(ptr, byte);
            }
            Ok(())
        }
        (Stage::Mesh, 0, false) => {
            let (var, bytes) = map(tx)?;
            let mask = tx.c32(0xFF);
            let byte = tx.and(value, mask);
            let ptr = word(tx, storage::WORKGROUP, var, bytes, addr);
            tx.f.store(ptr, byte);
            Ok(())
        }
        (stage, region, wide) => fail(format!(
            "ISBEWR of {} bits to region {region} in a {stage:?} program",
            if wide { 32 } else { 8 }
        )),
    }
}

fn barrier(tx: &mut Tx<'_>) {
    let scope = tx.c32(SCOPE_WORKGROUP);
    let semantics = tx.c32(SEMANTICS_WORKGROUP);
    tx.f.emit(op::CONTROL_BARRIER, &[scope, scope, semantics]);
}

/// The end of a task program: the mesh workgroups are launched, as many as it wrote at byte 4.
/// This ends the invocation (no return follows).
pub(super) fn epilogue_task(tx: &mut Tx<'_>) -> R<()> {
    let (var, words) = payload(tx)?;
    barrier(tx);
    let one = tx.c32(1);
    let ptr = word(tx, STORAGE_TASK_PAYLOAD, var, words, one);
    let u = tx.t.u32;
    let count = tx.f.load(&mut tx.m, u, ptr);
    tx.f.emit(OP_EMIT_MESH_TASKS, &[count, one, one, var]);
    Ok(())
}

/// `for (i = start; i < end; i += step) body(i)`, as a structured loop.
fn each(
    tx: &mut Tx<'_>,
    start: Id,
    end: Id,
    step: u32,
    mut body: impl FnMut(&mut Tx<'_>, Id) -> R<()>,
) -> R<()> {
    let u = tx.t.u32;
    let var = tx.f.local(&mut tx.m, u);
    tx.f.store(var, start);
    let (header, check, inside, cont, merge) =
        (tx.m.id(), tx.m.id(), tx.m.id(), tx.m.id(), tx.m.id());
    tx.f.branch(header);
    tx.f.label(header);
    tx.f.loop_merge(merge, cont);
    tx.f.branch(check);
    tx.f.label(check);
    let i = tx.f.load(&mut tx.m, u, var);
    let b = tx.t.bool_;
    let go = tx.op2(op::U_LESS_THAN, b, i, end);
    tx.f.branch_conditional(go, inside, merge);
    tx.f.label(inside);
    body(tx, i)?;
    tx.f.branch(cont);
    tx.f.label(cont);
    let s = tx.c32(step);
    let next = tx.iadd(i, s);
    tx.f.store(var, next);
    tx.f.branch(header);
    tx.f.label(merge);
    Ok(())
}

/// An output array of `n` floats at the location and component of attribute word `w`.
fn attribute_output(tx: &mut Tx<'_>, w: u32, n: u32, per_primitive: bool) -> Id {
    let made = if per_primitive {
        tx.mesh.primitive_outputs.get(&w)
    } else {
        tx.mesh.vertex_outputs.get(&w)
    };
    if let Some(v) = made {
        return *v;
    }
    let f = tx.t.f32;
    let length = tx.c32(n.max(1));
    let array = tx.m.ty_array(f, length);
    let ptr = tx.m.ty_pointer(storage::OUTPUT, array);
    let var = tx.m.global_variable(ptr, storage::OUTPUT);
    let (loc, comp) = ((w - 0x20) / 4, (w - 0x20) % 4);
    tx.m.decorate(var, deco::LOCATION, &[loc]);
    tx.m.decorate(var, deco::COMPONENT, &[comp]);
    if per_primitive {
        tx.m.decorate(var, DECO_PER_PRIMITIVE, &[]);
    }
    tx.interface.push(var);
    if per_primitive {
        tx.mesh.primitive_outputs.insert(w, var);
    } else {
        tx.mesh.vertex_outputs.insert(w, var);
    }
    var
}

/// The positions: an array of blocks whose one member is `Position`.
fn positions(tx: &mut Tx<'_>, n: u32) -> Id {
    if let Some(v) = tx.mesh.positions {
        return v;
    }
    let v4 = tx.t.v4f32;
    let block = tx.m.ty_struct(&[v4]);
    tx.m.decorate(block, deco::BLOCK, &[]);
    tx.m.member_decorate(block, 0, deco::BUILT_IN, &[builtin::POSITION]);
    let length = tx.c32(n.max(1));
    let array = tx.m.ty_array(block, length);
    let ptr = tx.m.ty_pointer(storage::OUTPUT, array);
    let var = tx.m.global_variable(ptr, storage::OUTPUT);
    tx.interface.push(var);
    tx.mesh.positions = Some(var);
    var
}

fn indices(tx: &mut Tx<'_>, n: u32) -> Id {
    if let Some(v) = tx.mesh.indices {
        return v;
    }
    let u = tx.t.u32;
    let uvec3 = tx.m.ty_vector(u, 3);
    let length = tx.c32(n.max(1));
    let array = tx.m.ty_array(uvec3, length);
    let ptr = tx.m.ty_pointer(storage::OUTPUT, array);
    let var = tx.m.global_variable(ptr, storage::OUTPUT);
    tx.m.decorate(var, deco::BUILT_IN, &[BUILTIN_PRIMITIVE_TRIANGLE_INDICES]);
    tx.interface.push(var);
    tx.mesh.indices = Some(var);
    var
}

/// The word of line `line` of element `index` whose lines start at word `base` and number
/// `lines` per group of 32.
fn line_word(tx: &mut Tx<'_>, base: u32, lines: u32, index: Id, line: u32) -> Id {
    let five = tx.c32(5);
    let group = tx.shr(index, five);
    let mask = tx.c32(31);
    let lane = tx.and(index, mask);
    let stride = tx.c32(lines * 32);
    let u = tx.t.u32;
    let at = tx.op2(op::I_MUL, u, group, stride);
    let offset = tx.c32(base + line * 32);
    let at = tx.iadd(at, offset);
    tx.iadd(at, lane)
}

/// The end of a mesh program: the vertices and primitives it wrote go out.
pub(super) fn epilogue_mesh(tx: &mut Tx<'_>) -> R<()> {
    let m = config(tx)?;
    if m.topology != 4 {
        return fail(format!("a mesh program of output topology {}", m.topology));
    }
    let vertex_words = vertex_words(tx);
    let base = primitive_base(tx, &m);
    let (attrs, attr_words) = attributes(tx)?;
    let (map_var, map_bytes) = map(tx)?;
    barrier(tx);
    let u = tx.t.u32;
    // The primitive count, from the map's first four bytes; no more than the most there may be.
    let mut count = tx.c32(0);
    for k in 0..4u32 {
        let at = tx.c32(k);
        let ptr = word(tx, storage::WORKGROUP, map_var, map_bytes, at);
        let byte = tx.f.load(&mut tx.m, u, ptr);
        let shift = tx.c32(8 * k);
        let part = tx.shl(byte, shift);
        count = tx.or(count, part);
    }
    let most = tx.c32(m.max_primitives);
    let count = tx.ext(u, crate::spirv::glsl::U_MIN, &[count, most]);
    let vertices = tx.c32(m.max_vertices);
    tx.f.emit(OP_SET_MESH_OUTPUTS, &[vertices, count]);
    let me = super::sm86::builtin_value(tx, builtin::LOCAL_INVOCATION_INDEX, u);
    let threads = m.threads.max(1);
    let lines = vertex_words.len() as u32;
    let max_vertices = m.max_vertices;
    each(tx, me, vertices, threads, |tx, v| {
        let zero = tx.cf32(0.0);
        let one = tx.cf32(1.0);
        let mut position = [zero, zero, zero, one];
        let mut has_position = false;
        for (line, w) in vertex_words.iter().enumerate() {
            let at = line_word(tx, 0, lines, v, line as u32);
            let ptr = word(tx, storage::WORKGROUP, attrs, attr_words, at);
            let raw = tx.f.load(&mut tx.m, u, ptr);
            let value = tx.as_f32(raw);
            match *w {
                0x1C..=0x1F => {
                    position[(*w - 0x1C) as usize] = value;
                    has_position = true;
                }
                0x20..=0x9F => {
                    let var = attribute_output(tx, *w, max_vertices, false);
                    let f = tx.t.f32;
                    let ptr_ty = tx.m.ty_pointer(storage::OUTPUT, f);
                    let ptr =
                        tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[var, v]);
                    tx.f.store(ptr, value);
                }
                _ => {}
            }
        }
        if has_position {
            if tx.cfg.depth_minus_one_to_one {
                let sum = tx.fadd(position[2], position[3]);
                let half = tx.cf32(0.5);
                position[2] = tx.fmul(sum, half);
            }
            let v4 = tx.t.v4f32;
            let value =
                tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, v4, &position);
            let var = positions(tx, max_vertices);
            let ptr_ty = tx.m.ty_pointer(storage::OUTPUT, v4);
            let member = tx.c32(0);
            let ptr =
                tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[var, v, member]);
            tx.f.store(ptr, value);
        }
        Ok(())
    })?;
    let primitive_words = m.primitive_words.clone();
    let plines = primitive_words.len() as u32;
    let max_primitives = m.max_primitives;
    each(tx, me, count, threads, |tx, p| {
        let uvec3 = tx.m.ty_vector(u, 3);
        let three = tx.c32(3);
        let four = tx.c32(4);
        let at = tx.op2(op::I_MUL, u, p, three);
        let first = tx.iadd(at, four);
        let mut corner = [first; 3];
        for (k, c) in corner.iter_mut().enumerate() {
            let kk = tx.c32(k as u32);
            let at = tx.iadd(first, kk);
            let ptr = word(tx, storage::WORKGROUP, map_var, map_bytes, at);
            *c = tx.f.load(&mut tx.m, u, ptr);
        }
        let value =
            tx.f.emit_result(&mut tx.m, op::COMPOSITE_CONSTRUCT, uvec3, &corner);
        let var = indices(tx, max_primitives);
        let ptr_ty = tx.m.ty_pointer(storage::OUTPUT, uvec3);
        let ptr =
            tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[var, p]);
        tx.f.store(ptr, value);
        for (line, w) in primitive_words.iter().enumerate() {
            if !(0x20..=0x9F).contains(w) {
                continue;
            }
            let at = line_word(tx, base, plines, p, line as u32);
            let ptr = word(tx, storage::WORKGROUP, attrs, attr_words, at);
            let raw = tx.f.load(&mut tx.m, u, ptr);
            let value = tx.as_f32(raw);
            let var = attribute_output(tx, *w, max_primitives, true);
            let f = tx.t.f32;
            let ptr_ty = tx.m.ty_pointer(storage::OUTPUT, f);
            let ptr =
                tx.f.emit_result(&mut tx.m, op::ACCESS_CHAIN, ptr_ty, &[var, p]);
            tx.f.store(ptr, value);
        }
        Ok(())
    })?;
    Ok(())
}
