//! The threads of a warp working together: `VOTE`, `VOTEU`, `SHFL`, `REDUX`, as subgroup
//! operations of the host (its subgroup is the warp: 32 invocations on the GPUs this runs on,
//! docs/ASSUMPTIONS.md A-243). An invocation whose guard is false takes no part, as a thread whose
//! guard is false does not run the instruction.

use super::super::emit::{Tx, R};
use super::mem::builtin_value;
use super::{pred, reg_at, write, write_f, write_pred, I};
use crate::sm86::{op, PT};
use crate::spirv::{builtin, cap, glsl, op as sop, Id};

/// The subgroup scope.
const SUBGROUP: u32 = 3;

/// The lane number of the invocation.
fn lane(tx: &mut Tx<'_>) -> Id {
    tx.m.capability(cap::GROUP_NON_UNIFORM);
    let u = tx.t.u32;
    builtin_value(tx, builtin::SUBGROUP_LOCAL_INVOCATION_ID, u)
}

/// The ballot of `p`: bit `n` set where lane `n` has it.
fn ballot(tx: &mut Tx<'_>, p: Id) -> Id {
    tx.m.capability(cap::GROUP_NON_UNIFORM_BALLOT);
    let scope = tx.c32(SUBGROUP);
    let v4 = tx.t.v4u32;
    let all =
        tx.f.emit_result(&mut tx.m, sop::GROUP_NON_UNIFORM_BALLOT, v4, &[scope, p]);
    let u = tx.t.u32;
    tx.f.emit_result(&mut tx.m, sop::COMPOSITE_EXTRACT, u, &[all, 0])
}

/// `VOTE.mode Rd, Pu, Pc` (and `VOTEU` with a uniform register and predicate): `Rd` the ballot of
/// the predicate, `Pu` whether it holds in all, in any, or in all or none of the threads (bits
/// 72..73: 0, 1, 2).
pub(super) fn vote(tx: &mut Tx<'_>, i: &I, uniform: bool) -> R<()> {
    let inst = i.inst;
    let mode = inst.field(72, 2);
    if mode == 3 {
        return tx.fail("VOTE in mode 3");
    }
    let p = pred(tx, inst.pred(87, 90, false));
    let taking_part = i.cond.unwrap_or_else(|| tx.ctrue());
    let voting = tx.land(p, taking_part);
    let bits = ballot(tx, voting);
    if uniform {
        tx.write_u_if(i.cond, inst.ureg(16), bits);
    } else {
        write(tx, i, inst.dst(), bits);
    }
    let target = inst.pred_dst(81, uniform);
    if target != PT {
        tx.m.capability(cap::GROUP_NON_UNIFORM_VOTE);
        let scope = tx.c32(SUBGROUP);
        let b = tx.t.bool_;
        let value = match mode {
            0 => {
                // A thread that takes no part does not make it false.
                let not_part = tx.lnot(taking_part);
                let either = tx.lor(p, not_part);
                tx.f.emit_result(&mut tx.m, sop::GROUP_NON_UNIFORM_ALL, b, &[scope, either])
            }
            1 => {
                tx.f.emit_result(&mut tx.m, sop::GROUP_NON_UNIFORM_ANY, b, &[scope, voting])
            }
            _ => {
                tx.f.emit_result(&mut tx.m, sop::GROUP_NON_UNIFORM_ALL_EQUAL, b, &[scope, p])
            }
        };
        write_pred(tx, i, target, value);
    }
    Ok(())
}

/// `SHFL.mode Pu, Rd, Ra, b, c`: the value of `Ra` in another lane. The lane and the clamp (`c`'s
/// low five bits) with its segment mask (bits 8..12) as PTX's `shfl` defines them; `Pu` says
/// whether the lane was in range (else the thread gets its own value).
pub(super) fn shfl(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let (b, c) = match inst.opcode() {
        0x389 => (tx.read(inst.reg(32)), tx.read(inst.reg(64))),
        0x589 => (tx.read(inst.reg(32)), tx.c32(inst.field(40, 13) as u32)),
        0x989 => (tx.c32(inst.field(53, 5) as u32), tx.read(inst.reg(64))),
        0xf89 => (
            tx.c32(inst.field(53, 5) as u32),
            tx.c32(inst.field(40, 13) as u32),
        ),
        _ => return tx.fail(format!("SHFL form {:#05x}", inst.opcode())),
    };
    let value = tx.read(inst.reg(24));
    let me = lane(tx);
    let thirty_one = tx.c32(31);
    let bval = tx.and(b, thirty_one);
    let cval = tx.and(c, thirty_one);
    let eight = tx.c32(8);
    let seg = tx.shr(c, eight);
    let segmask = tx.and(seg, thirty_one);
    let not_seg = tx.not(segmask);
    let lane_seg = tx.and(me, segmask);
    let clamp = tx.and(cval, not_seg);
    let max_lane = tx.or(lane_seg, clamp);
    let min_lane = lane_seg;
    let (j, in_range) = match inst.field(58, 2) {
        // IDX
        0 => {
            let pick = tx.and(bval, not_seg);
            let j = tx.or(min_lane, pick);
            let ok = tx.cmp(sop::U_LESS_THAN_EQUAL, j, max_lane);
            (j, ok)
        }
        // UP
        1 => {
            let j = tx.isub(me, bval);
            let js = tx.as_i32(j);
            let ms = tx.as_i32(max_lane);
            let ok = tx.cmp(sop::S_GREATER_THAN_EQUAL, js, ms);
            (j, ok)
        }
        // DOWN
        2 => {
            let j = tx.iadd(me, bval);
            let ok = tx.cmp(sop::U_LESS_THAN_EQUAL, j, max_lane);
            (j, ok)
        }
        // BFLY
        _ => {
            let j = tx.xor(me, bval);
            let ok = tx.cmp(sop::U_LESS_THAN_EQUAL, j, max_lane);
            (j, ok)
        }
    };
    let source = tx.select_u32(in_range, j, me);
    tx.m.capability(cap::GROUP_NON_UNIFORM_SHUFFLE);
    let scope = tx.c32(SUBGROUP);
    let u = tx.t.u32;
    let r = tx.f.emit_result(
        &mut tx.m,
        sop::GROUP_NON_UNIFORM_SHUFFLE,
        u,
        &[scope, value, source],
    );
    write(tx, i, inst.dst(), r);
    let p = inst.pred_dst(81, false);
    if p != PT {
        write_pred(tx, i, p, in_range);
    }
    Ok(())
}

/// `REDUX.op URd, Ra`: `Ra` of every thread combined (bits 78..80: 0 and, 1 or, 2 xor, 3 sum,
/// 4 minimum, 5 maximum; signed with bit 73) into a uniform register.
pub(super) fn redux(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let signed = inst.bit(73);
    let opcode = match inst.field(78, 3) {
        0 => sop::GROUP_NON_UNIFORM_BITWISE_AND,
        1 => sop::GROUP_NON_UNIFORM_BITWISE_OR,
        2 => sop::GROUP_NON_UNIFORM_BITWISE_XOR,
        3 => sop::GROUP_NON_UNIFORM_I_ADD,
        4 if signed => sop::GROUP_NON_UNIFORM_S_MIN,
        4 => sop::GROUP_NON_UNIFORM_U_MIN,
        5 if signed => sop::GROUP_NON_UNIFORM_S_MAX,
        5 => sop::GROUP_NON_UNIFORM_U_MAX,
        k => return tx.fail(format!("REDUX operation {k}")),
    };
    if i.cond.is_some() {
        return tx.fail("a guarded REDUX");
    }
    tx.m.capability(cap::GROUP_NON_UNIFORM_ARITHMETIC);
    let value = tx.read(inst.reg(24));
    let scope = tx.c32(SUBGROUP);
    let u = tx.t.u32;
    // Group operation 0: the reduction of all of them.
    let r = tx.f.emit_result(&mut tx.m, opcode, u, &[scope, 0, value]);
    tx.write_u_if(None, inst.ureg(16), r);
    Ok(())
}

/// `FSWZADD`: not yet.
/// `FSWZADD Rd, Ra, Rb, ops`: each thread of a quad (its lane's low two bits) combines its own
/// `Ra` and `Rb` by its own operation, two bits of bits 32..39 with the first lane's highest: 0
/// `a + b`, 1 `a - b`, 2 `b - a`, 3 `a` (NAK's encoding; the senses as the title's derivatives
/// need them: `Ra` the quad's first value, `Rb` the thread's own, and lanes 1 and 2 subtract;
/// docs/ASSUMPTIONS.md A-254). The rounding mode is to nearest.
pub(super) fn fswzadd(tx: &mut Tx<'_>, i: &I) -> R<()> {
    let inst = i.inst;
    let a = tx.read_f(inst.reg(24));
    let b = tx.read_f(inst.reg(64));
    let ops = tx.c32(inst.field(32, 8) as u32);
    let me = lane(tx);
    let three = tx.c32(3);
    let quad_lane = tx.and(me, three);
    let back = tx.isub(three, quad_lane);
    let one = tx.c32(1);
    let shift = tx.shl(back, one);
    let shifted = tx.shr(ops, shift);
    let mode = tx.and(shifted, three);
    let f = tx.t.f32;
    let sum = tx.op2(sop::F_ADD, f, a, b);
    let a_less_b = tx.op2(sop::F_SUB, f, a, b);
    let b_less_a = tx.op2(sop::F_SUB, f, b, a);
    let is = |tx: &mut Tx<'_>, m: u32| {
        let c = tx.c32(m);
        tx.cmp(sop::I_EQUAL, mode, c)
    };
    let (m0, m1, m2) = (is(tx, 0), is(tx, 1), is(tx, 2));
    let r = tx.select(f, m2, b_less_a, a);
    let r = tx.select(f, m1, a_less_b, r);
    let r = tx.select(f, m0, sum, r);
    write_f(tx, i, inst.dst(), r);
    Ok(())
}

/// `MATCH`: not yet.
pub(super) fn match_(tx: &mut Tx<'_>, i: &I) -> R<()> {
    // `MATCH.ALL[.U64] Pu, Rd, Ra` (NAK: bit 79 clear, a 64-bit value with bit 73): `Pu` whether
    // every thread that takes part holds the same value, `Rd` their mask then (zero otherwise).
    // Each compares its value with the first such thread's. `MATCH.ANY`, the mask of the threads
    // with the same value as each, is not translated.
    let inst = i.inst;
    debug_assert_eq!(inst.opcode(), op::MATCH);
    if inst.bit(79) {
        return tx.fail("MATCH.ANY");
    }
    let taking_part = i.cond.unwrap_or_else(|| tx.ctrue());
    let bits = ballot(tx, taking_part);
    let u = tx.t.u32;
    let first = tx.ext(u, glsl::FIND_I_LSB, &[bits]);
    tx.m.capability(cap::GROUP_NON_UNIFORM_SHUFFLE);
    let scope = tx.c32(SUBGROUP);
    let src = inst.reg(24);
    let mut same = tx.ctrue();
    for w in 0..1 + u32::from(inst.bit(73)) {
        let v = tx.read(reg_at(i, src, w));
        let theirs = tx.f.emit_result(
            &mut tx.m,
            sop::GROUP_NON_UNIFORM_SHUFFLE,
            u,
            &[scope, v, first],
        );
        let eq = tx.cmp(sop::I_EQUAL, v, theirs);
        same = tx.land(same, eq);
    }
    let not_part = tx.lnot(taking_part);
    let either = tx.lor(same, not_part);
    tx.m.capability(cap::GROUP_NON_UNIFORM_VOTE);
    let b = tx.t.bool_;
    let all =
        tx.f.emit_result(&mut tx.m, sop::GROUP_NON_UNIFORM_ALL, b, &[scope, either]);
    let zero = tx.c32(0);
    let mask = tx.select(u, all, bits, zero);
    write(tx, i, inst.dst(), mask);
    write_pred(tx, i, inst.pred_dst(81, false), all);
    Ok(())
}
