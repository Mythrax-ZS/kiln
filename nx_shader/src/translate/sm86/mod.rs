//! Translation of sm_86 instructions (the Switch 2's, see [`crate::sm86`]) to SPIR-V, on the same
//! model as the Maxwell ones: registers, uniform registers and predicates are function-local
//! variables, and every instruction becomes a few operations on them.
//!
//! An instruction's guard is read once, before it writes anything: an instruction that writes the
//! predicate that guards it (`@P0 ISETP P0, ...`) is guarded by the old value, as on the hardware.
//! A uniform instruction (`UIADD3` and the rest) reads and writes uniform registers; the same
//! translation serves both, through [`I::uniform`].
//!
//! The scheduling control of an instruction (its barriers and stall counts) is not looked at: the
//! translation runs each instruction to completion before the next.

mod float;
mod int;
mod mem;
mod surface;
mod tex;
mod warp;

use super::emit::{Tx, R};
use super::local::Space;
use super::Dim;
use crate::cfg::Cfg;
use crate::inst::Guard;
use crate::sm86::{self, op, Inst, Operand, Source, PT, RZ, URZ};
use crate::spirv::Id;
use std::collections::HashMap;

/// The constant buffer slot NVN2's code names for "no buffer bound" (A-240): read as the spare slot.
const UNBOUND_SLOT: u32 = 24;
const SPARE_SLOT: u32 = 17;

/// One instruction being translated.
#[derive(Clone, Copy)]
pub(super) struct I {
    pub inst: Inst,
    /// The guard's value, read before the instruction writes anything (`None`: always).
    pub cond: Option<Id>,
    /// It runs on the uniform datapath.
    pub uniform: bool,
}

/// Translate one instruction that is not a terminator.
pub(super) fn instruction(tx: &mut Tx<'_>, inst: Inst) -> R<()> {
    // The scoreboard (A-256): the writes the instruction waits for land first (bits 116..121, a
    // barrier each; `DEPBAR.LE SBn, count`: the oldest of barrier n until at most `count` are out),
    // and the writes of an instruction that sets a write barrier (bits 110..112) are held.
    let wait = inst.field(116, 6);
    for b in 0..6u8 {
        if wait >> b & 1 != 0 {
            tx.wait_for(b, 0);
        }
    }
    if inst.opcode() == op::DEPBAR {
        let (barrier, keep) = depbar(inst);
        tx.wait_for(barrier, keep);
    }
    let guard = inst.guard();
    if guard.is_never() {
        return Ok(());
    }
    let cond = tx.guard_condition(guard);
    let i = I {
        inst,
        cond,
        uniform: inst.is_uniform(),
    };
    let barrier = inst.field(110, 3) as u8;
    if barrier < 6 {
        tx.scoreboard.holding = Some(barrier);
    }
    let done = dispatch(tx, &i);
    if barrier < 6 {
        tx.scoreboard.holding = None;
        let writes = std::mem::take(&mut tx.scoreboard.current);
        tx.scoreboard.groups[usize::from(barrier)].push_back(writes);
    }
    done
}

/// `DEPBAR.LE SBn, count`: the barrier (bits 44..46) and the count (bits 38..43) of writes that
/// may still be out.
fn depbar(inst: Inst) -> (u8, usize) {
    (inst.field(44, 3) as u8, inst.field(38, 6) as usize)
}

fn dispatch(tx: &mut Tx<'_>, i: &I) -> R<()> {
    use op::alu;
    let inst = i.inst;
    match inst.opcode() {
        // Scheduling, convergence and the bookkeeping of control flow the graph has used already:
        // nothing a single thread does.
        op::NOP | op::DEPBAR | op::BSSY | op::BSYNC | op::BREAK => return Ok(()),
        op::S2R => return mem::s2r(tx, i),
        op::CS2R => return mem::cs2r(tx, i),
        op::S2UR => return mem::s2ur(tx, i),
        op::LDC => return mem::ldc(tx, i),
        op::ULDC | op::ULDC_INDEXED => return mem::uldc(tx, i),
        op::ALD => return mem::ald(tx, i),
        op::AST => return mem::ast(tx, i),
        op::IPA => return mem::ipa(tx, i),
        op::LDG | op::LDG_U => return mem::ldg(tx, i),
        op::STG | op::STG_U => return mem::stg(tx, i),
        op::ATOMG | op::ATOMG_U | op::RED => return mem::atomg(tx, i),
        op::LDL => return mem::load_local(tx, i, Space::Local),
        op::LDS => return mem::load_local(tx, i, Space::Shared),
        op::STL | op::STL_U => return mem::store_local(tx, i, Space::Local),
        op::STS | op::STS_U => return mem::store_local(tx, i, Space::Shared),
        op::BAR => return mem::bar(tx, i),
        op::MEMBAR | op::ERRBAR => return mem::membar(tx, i),
        op::TEX | op::TEX_B => return tex::tex(tx, i),
        op::TLD | op::TLD_B => return tex::tld(tx, i),
        op::TLD4 | op::TLD4_B => return tex::tld4(tx, i),
        op::TMML | op::TMML_B => return tex::tmml(tx, i),
        op::TXQ | op::TXQ_B => return tex::txq(tx, i),
        op::SULD_BOUND | op::SULD => return surface::suld(tx, i),
        op::SUST_BOUND | op::SUST => return surface::sust(tx, i),
        op::SURED => return surface::sured(tx, i),
        op::FSWZADD => return warp::fswzadd(tx, i),
        op::MATCH => return warp::match_(tx, i),
        op::PLOP3 | op::UPLOP3 => return plop3(tx, i),
        op::R2UR => return r2ur(tx, i),
        op::REDUX => return warp::redux(tx, i),
        _ => {}
    }
    // The ALU instructions, by their 9-bit opcode; the uniform ones as the vector ones but where
    // they differ.
    let base = inst.base();
    let vector = match base {
        alu::UMOV => alu::MOV,
        alu::UP2UR => return up2ur(tx, i),
        alu::VOTEU => return warp::vote(tx, i, true),
        alu::UF2FP => alu::F2FP,
        alu::UFLO => alu::FLO,
        alu::UBREV => alu::BREV,
        alu::UPOPC => alu::POPC,
        b if i.uniform => b - 0x80,
        b => b,
    };
    match vector {
        alu::MOV => mov(tx, i),
        alu::SEL => sel(tx, i),
        alu::P2R => p2r(tx, i),
        alu::R2P => r2p(tx, i),
        alu::VOTE => warp::vote(tx, i, false),
        alu::WARPSYNC => Ok(()),
        alu::SHFL => warp::shfl(tx, i),
        alu::FSEL => float::fsel(tx, i),
        alu::FMNMX => float::fmnmx(tx, i),
        alu::FSET => float::fset(tx, i),
        alu::FSETP => float::fsetp(tx, i),
        alu::FMUL => float::fmul(tx, i),
        alu::FADD => float::fadd(tx, i),
        alu::FFMA => float::ffma(tx, i),
        alu::MUFU => float::mufu(tx, i),
        alu::FRND => float::frnd(tx, i),
        alu::F2I => float::f2i(tx, i),
        alu::I2F | alu::I2FP => float::i2f(tx, i),
        alu::F2F => float::f2f(tx, i),
        alu::F2FP => float::f2fp(tx, i),
        alu::HADD2 | alu::HMUL2 | alu::HFMA2 => float::half_arith(tx, i),
        alu::HSETP2 => float::hsetp2(tx, i),
        alu::ISETP => int::isetp(tx, i),
        alu::IADD3 => int::iadd3(tx, i),
        alu::LEA => int::lea(tx, i),
        alu::LOP3 => int::lop3(tx, i),
        alu::IABS => int::iabs(tx, i),
        alu::PRMT => int::prmt(tx, i),
        alu::IMNMX => int::imnmx(tx, i),
        alu::SHF => int::shf(tx, i),
        alu::SGXT => int::sgxt(tx, i),
        alu::BMSK => int::bmsk(tx, i),
        alu::IMAD => int::imad(tx, i),
        alu::IMAD_WIDE => int::imad_wide(tx, i),
        alu::IMAD_HI => int::imad_hi(tx, i),
        alu::FLO => int::flo(tx, i),
        alu::BREV => int::brev(tx, i),
        alu::POPC => int::popc(tx, i),
        _ => tx.fail(format!(
            "{} ({:032x}) is not translated",
            sm86::name(inst),
            inst.0
        )),
    }
}

// ---- operands ------------------------------------------------------------------------------

/// The raw 32 bits of an operand.
pub(super) fn operand(tx: &mut Tx<'_>, o: Operand) -> R<Id> {
    Ok(match o {
        Operand::Reg(r) => tx.read(r),
        Operand::UReg(r) => tx.read_u(r),
        Operand::Imm(v) => tx.c32(v),
        Operand::Cbuf { slot, offset } => cbuf(tx, slot, offset)?,
        Operand::CbufUReg { .. } => {
            return tx.fail("an operand from the constant buffer a uniform register names")
        }
    })
}

/// A word of constant buffer `slot` at byte `offset`.
pub(super) fn cbuf(tx: &mut Tx<'_>, slot: u32, offset: u32) -> R<Id> {
    if !offset.is_multiple_of(4) {
        return tx.fail(format!(
            "a constant buffer operand at {offset:#x}, not a multiple of four"
        ));
    }
    let slot = if slot == UNBOUND_SLOT {
        SPARE_SLOT
    } else {
        slot
    };
    tx.cbuf_word(slot, offset)
}

/// An operand as a float, with its absolute value and negation applied.
pub(super) fn float(tx: &mut Tx<'_>, s: Source) -> R<Id> {
    let raw = operand(tx, s.operand)?;
    let f = tx.as_f32(raw);
    Ok(super::alu::fmods(tx, f, s.neg, s.abs))
}

/// The first operand.
pub(super) fn a(i: &I) -> Source {
    i.inst.src_a()
}

/// The operand in bits 32..63 (the second of a two-operand instruction, whatever the form).
pub(super) fn b(tx: &Tx<'_>, i: &I) -> R<Source> {
    match i.inst.slot32() {
        Some(s) => Ok(s),
        None => tx.fail(format!(
            "{} in operand form {}",
            sm86::name(i.inst),
            i.inst.form()
        )),
    }
}

/// The second and third operands.
pub(super) fn bc(tx: &Tx<'_>, i: &I) -> R<(Source, Source)> {
    match i.inst.src_bc() {
        Some(s) => Ok(s),
        None => tx.fail(format!(
            "{} in operand form {}",
            sm86::name(i.inst),
            i.inst.form()
        )),
    }
}

/// The destination register (a uniform one for a uniform instruction).
pub(super) fn dst(i: &I) -> u8 {
    if i.uniform {
        i.inst.ureg(16)
    } else {
        i.inst.dst()
    }
}

/// Write the destination register `r` (or `r` places after it) of the instruction.
pub(super) fn write(tx: &mut Tx<'_>, i: &I, r: u8, value: Id) {
    if i.uniform {
        tx.write_u_if(i.cond, r, value);
    } else {
        tx.write_if(i.cond, r, value);
    }
}

pub(super) fn write_f(tx: &mut Tx<'_>, i: &I, r: u8, value: Id) {
    let u = tx.as_u32(value);
    write(tx, i, r, u);
}

/// The register `k` places after `base` (`RZ` and `URZ` stay themselves).
pub(super) fn reg_at(i: &I, base: u8, k: u32) -> u8 {
    let zero = if i.uniform { URZ } else { RZ };
    if base == zero {
        zero
    } else {
        base.wrapping_add(k as u8)
    }
}

/// Read register `r` of the instruction's register file.
pub(super) fn read(tx: &mut Tx<'_>, i: &I, r: u8) -> Id {
    if i.uniform {
        tx.read_u(r)
    } else {
        tx.read(r)
    }
}

/// A predicate operand's value.
pub(super) fn pred(tx: &mut Tx<'_>, g: Guard) -> Id {
    tx.read_pred(g.index, g.negated)
}

/// Write a predicate destination (in the translation's numbering, see [`crate::sm86::UP_BASE`]).
pub(super) fn write_pred(tx: &mut Tx<'_>, i: &I, p: u8, value: Id) {
    tx.write_pred_if(i.cond, p, value);
}

// ---- moves and predicates -------------------------------------------------------------------

/// `MOV Rd, B` (and `UMOV`): the lane mask of bits 72..75 is the whole quad for anything but a
/// quad move.
fn mov(tx: &mut Tx<'_>, i: &I) -> R<()> {
    if !i.uniform && i.inst.field(72, 4) != 0xF {
        return tx.fail(format!(
            "MOV with the quad lane mask {:#x}",
            i.inst.field(72, 4)
        ));
    }
    let src = b(tx, i)?;
    let v = operand(tx, src.operand)?;
    let d = dst(i);
    write(tx, i, d, v);
    // The immediate a register is known to hold, for a texture instruction's texel offsets.
    if let (Operand::Imm(k), false, None) = (src.operand, i.uniform, i.cond) {
        if d != RZ {
            tx.reg_const.insert(d, k);
        }
    }
    Ok(())
}

/// The constant buffer slot that holds a bound texture instruction's handle (bits 54..58).
pub(super) fn slot_of_texture(inst: Inst) -> u32 {
    inst.field(54, 5) as u32
}

/// `SEL Rd, Ra, B, Pc`: `Ra` where the predicate holds, else `B`.
fn sel(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let a = operand(tx, a(i).operand)?;
    let bsrc = b(tx, i)?;
    let b = operand(tx, bsrc.operand)?;
    let p = pred(tx, i.inst.pred(87, 90, i.uniform));
    let v = tx.select_u32(p, a, b);
    write(tx, i, dst(i), v);
    Ok(())
}

/// `P2R Rd, PR, Ra, mask`: `Ra` with the bits of the mask (shifted to the byte bits 76..77 name)
/// replaced by the predicates.
fn p2r(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.form() != 4 {
        return tx.fail("P2R with a mask that is not an immediate");
    }
    let mask = inst.field(32, 32) as u32 & 0x7F;
    let shift = inst.field(76, 2) as u32 * 8;
    let merged = read(tx, i, inst.reg(24));
    let keep = tx.c32(!(mask << shift));
    let mut v = tx.and(merged, keep);
    for k in 0..7u8 {
        if mask >> k & 1 == 0 {
            continue;
        }
        let index = if i.uniform { sm86::UP_BASE + k } else { k };
        let p = tx.read_pred(index, false);
        let bit = tx.c32(1 << (u32::from(k) + shift));
        let zero = tx.c32(0);
        let set = tx.select_u32(p, bit, zero);
        v = tx.or(v, set);
    }
    write(tx, i, dst(i), v);
    Ok(())
}

/// `R2P PR, Ra, mask`: the predicates of the mask take the bits of `Ra` (at the byte bits 76..77
/// name).
fn r2p(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.form() != 4 {
        return tx.fail("R2P with a mask that is not an immediate");
    }
    let mask = inst.field(32, 32) as u32 & 0x7F;
    let shift = inst.field(76, 2) as u32 * 8;
    let source = read(tx, i, inst.reg(24));
    for k in 0..7u8 {
        if mask >> k & 1 == 0 {
            continue;
        }
        let bit = tx.c32(1 << (u32::from(k) + shift));
        let masked = tx.and(source, bit);
        let zero = tx.c32(0);
        let set = tx.cmp(crate::spirv::op::I_NOT_EQUAL, masked, zero);
        let index = if i.uniform { sm86::UP_BASE + k } else { k };
        write_pred(tx, i, index, set);
    }
    Ok(())
}

/// `PLOP3 Pu, Pv, Pa, Pb, Pc, lut0, lut1` (and `UPLOP3`): two three-input logic operations of the
/// same predicates (the third a uniform one with bit 67).
fn plop3(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let uniform = inst.opcode() == op::UPLOP3;
    let s0 = pred(tx, inst.pred(87, 90, uniform));
    let s1 = pred(tx, inst.pred(77, 80, uniform));
    let s2 = pred(tx, inst.pred(68, 71, uniform || inst.bit(67)));
    let luts = [
        inst.field(64, 3) | inst.field(72, 5) << 3,
        inst.field(16, 8),
    ];
    for (at, lut) in [81u32, 84].into_iter().zip(luts) {
        let p = inst.pred_dst(at, uniform);
        if p == PT {
            continue;
        }
        let v = logic3(tx, lut, s0, s1, s2);
        write_pred(tx, i, p, v);
    }
    Ok(())
}

/// The three-input logic operation of `lut` on booleans: true where bit `4a + 2b + c` is set.
fn logic3(tx: &mut Tx<'_>, lut: u64, a: Id, b: Id, c: Id) -> Id {
    match lut & 0xFF {
        0x00 => return tx.cfalse(),
        0xFF => return tx.ctrue(),
        0xF0 => return a,
        0xCC => return b,
        0xAA => return c,
        _ => {}
    }
    let mut r = tx.cfalse();
    for k in 0..8u64 {
        if lut >> k & 1 == 0 {
            continue;
        }
        let ta = if k & 4 != 0 { a } else { tx.lnot(a) };
        let tb = if k & 2 != 0 { b } else { tx.lnot(b) };
        let tc = if k & 1 != 0 { c } else { tx.lnot(c) };
        let ab = tx.land(ta, tb);
        let term = tx.land(ab, tc);
        r = tx.lor(r, term);
    }
    r
}

/// `R2UR URd, Ra`: a register into a uniform register (the value is the same in every thread).
fn r2ur(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let v = tx.read(i.inst.reg(24));
    tx.write_u_if(i.cond, i.inst.ureg(16), v);
    Ok(())
}

/// `UP2UR URd, UPR, URa, mask`: the uniform predicates of the mask into the bits of a uniform
/// register.
fn up2ur(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    if inst.form() != 4 {
        return tx.fail("UP2UR with a mask that is not an immediate");
    }
    let mask = inst.field(32, 32) as u32 & 0x7F;
    let merged = tx.read_u(inst.ureg(24));
    let keep = tx.c32(!mask);
    let mut v = tx.and(merged, keep);
    for k in 0..7u8 {
        if mask >> k & 1 == 0 {
            continue;
        }
        let p = tx.read_pred(sm86::UP_BASE + k, false);
        let bit = tx.c32(1 << k);
        let zero = tx.c32(0);
        let set = tx.select_u32(p, bit, zero);
        v = tx.or(v, set);
    }
    tx.write_u_if(i.cond, inst.ureg(16), v);
    Ok(())
}

// ---- what the translation needs to know before it starts ----------------------------------------

/// What the texture instructions of a program say about the shape of each texture they name (the
/// constant buffer word of its handle): the first sample, fetch or gather that names one decides,
/// and `TMML` only where none does (it says 2D for a 2D array, A-244).
pub(super) fn scan_dims(code: &[u8], graph: &Cfg) -> HashMap<u32, (Dim, bool)> {
    let mut dims = HashMap::new();
    for pass in [false, true] {
        for block in graph.blocks.values() {
            for (_, inst) in sm86::cfg::instructions(code, block) {
                if (inst.opcode() == op::TMML) != pass {
                    continue;
                }
                if let Some(shape) = tex::shape(inst) {
                    dims.entry(tex::handle_word(inst)).or_insert(shape);
                }
            }
        }
    }
    dims
}

/// How many words of local (or shared) memory the constant-offset accesses of a program reach.
pub(super) fn scan_memory(code: &[u8], graph: &Cfg, space: Space) -> u32 {
    let mut words = 0;
    for block in graph.blocks.values() {
        for (_, inst) in sm86::cfg::instructions(code, block) {
            if let Some(end) = mem::constant_extent(inst, space) {
                words = words.max(end.div_ceil(4));
            }
        }
    }
    words
}

#[cfg(test)]
mod tests {
    use crate::sm86::tests::word;
    use crate::sm86::{Inst, PT, RZ, URZ};
    use crate::translate::{translate_sm86, Config, Dim, Stage};

    /// The bytes of a program: its instructions, then `EXIT` and the branch to itself that ends
    /// it.
    fn program(insts: &[Inst]) -> Vec<u8> {
        let exit = word(&[(0, 12, 0x94d), (12, 4, 7), (87, 3, 7)]);
        let end = word(&[
            (0, 12, 0x947),
            (12, 4, 7),
            (87, 3, 7),
            (34, 48, (-4i64 as u64) & ((1 << 48) - 1)),
        ]);
        insts
            .iter()
            .chain([&exit, &end])
            .flat_map(|i| i.0.to_le_bytes())
            .collect()
    }

    fn kernel(insts: &[Inst]) -> crate::translate::Translation {
        let mut cfg = Config::new(Stage::Kernel);
        cfg.block = [64, 1, 1];
        let t = translate_sm86(&program(insts), None, &cfg).expect("translates");
        match crate::validate::validate(&t.spirv) {
            Ok(Some(())) => {}
            Ok(None) => {
                eprintln!("SKIPPED: spirv-val is not installed, the SPIR-V is not validated")
            }
            Err(e) => panic!("invalid SPIR-V: {e}"),
        }
        t
    }

    #[test]
    fn surfaces_are_loaded_and_stored_as_storage_images() {
        // SULD.P.2D.RGBA R4, [R0], c[0][0x20]; SUST.P.2D.R [R0], R4, c[0][0x24]
        let suld = word(&[
            (0, 12, 0xb97),
            (12, 4, 7),
            (16, 8, 4),
            (24, 8, 0),
            (40, 14, 8),
            (54, 5, 0),
            (61, 3, 3),
            (72, 4, 0xF),
            (81, 3, u64::from(PT)),
        ]);
        let sust = word(&[
            (0, 12, 0xb9b),
            (12, 4, 7),
            (24, 8, 0),
            (32, 8, 4),
            (40, 14, 9),
            (54, 5, 0),
            (61, 3, 3),
            (72, 4, 0x1),
        ]);
        let t = kernel(&[suld, sust]);
        let images = &t.info.images;
        assert_eq!(images.len(), 2);
        assert_eq!((images[0].index, images[0].source), (8, Some(0)));
        assert!(images[0].read && !images[0].written);
        assert_eq!(
            (images[1].index, images[1].dim, images[1].arrayed),
            (9, Dim::D2, false)
        );
        assert!(images[1].written && !images[1].read);
        // A 2D array: the layer is the third coordinate.
        let array = word(&[
            (0, 12, 0xb97),
            (12, 4, 7),
            (16, 8, 4),
            (40, 14, 8),
            (61, 3, 4),
            (72, 4, 0x3),
            (81, 3, u64::from(PT)),
        ]);
        assert!(kernel(&[array]).info.images[0].arrayed);
    }

    #[test]
    fn global_atomics_carries_and_matches_translate() {
        // IMAD.X R5, RZ, RZ, R6, P0 (form 1: registers)
        let imad_x = word(&[
            (0, 12, 0x224),
            (12, 4, 7),
            (16, 8, 5),
            (24, 8, u64::from(RZ)),
            (32, 8, u64::from(RZ)),
            (64, 8, 6),
            (73, 1, 1),
            (74, 1, 1),
            (81, 3, u64::from(PT)),
            (87, 4, 0),
        ]);
        // ATOMG.E.ADD.U32 R8, [R2.64 + URZ], R3
        let atomg = word(&[
            (0, 12, 0x9a8),
            (12, 4, 7),
            (16, 8, 8),
            (24, 8, 2),
            (32, 8, 3),
            (64, 6, u64::from(URZ)),
            (70, 1, 1),
            (72, 1, 1),
            (81, 3, u64::from(PT)),
            (87, 4, 0),
            (91, 1, 1),
        ]);
        // RED.E.OR [R2.64 + URZ], R3 and ATOMG.E.EXCH R8, [R2.64], R3 (no uniform register)
        let red = word(&[
            (0, 12, 0x98e),
            (12, 4, 7),
            (24, 8, 2),
            (32, 8, 3),
            (64, 6, u64::from(URZ)),
            (72, 1, 1),
            (87, 3, 6),
            (90, 1, 1),
            (91, 1, 1),
        ]);
        let exch = word(&[
            (0, 12, 0x3a8),
            (12, 4, 7),
            (16, 8, 8),
            (24, 8, 2),
            (32, 8, 3),
            (72, 1, 1),
            (81, 3, u64::from(PT)),
            (87, 4, 8),
        ]);
        // MATCH.ALL.U64 P1, R9, R2
        let matching = word(&[
            (0, 12, 0x3a1),
            (12, 4, 7),
            (16, 8, 9),
            (24, 8, 2),
            (73, 1, 1),
            (79, 1, 0),
            (81, 3, 1),
        ]);
        let t = kernel(&[imad_x, atomg, red, exch, matching]);
        assert!(t.info.global_memory);
        // Inc and dec have no SPIR-V operation of their own: they are refused.
        let inc = word(&[
            (0, 12, 0x9a8),
            (12, 4, 7),
            (16, 8, 8),
            (24, 8, 2),
            (32, 8, 3),
            (64, 6, u64::from(URZ)),
            (87, 4, 3),
            (91, 1, 1),
        ]);
        let mut cfg = Config::new(Stage::Kernel);
        cfg.block = [64, 1, 1];
        assert!(translate_sm86(&program(&[inc]), None, &cfg).is_err());
    }
}
