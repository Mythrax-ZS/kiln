//! Instruction semantics: each decoded instruction becomes IR operations.
//!
//! Every arm follows the matching arm of the interpreter (`nx_runtime::a64::interp`), which in turn
//! follows the Arm pseudocode, including the order in which registers are written when an
//! encoding is CONSTRAINED UNPREDICTABLE (docs/ASSUMPTIONS.md A-017). The flag formulas are the
//! textbook ones over explicit `i1` values; the differential tests are what establish that they
//! equal `AddWithCarry`.

use super::builder::Builder;
use super::ir::*;
use nx_runtime::a64::bits::decode_bit_masks;
use nx_runtime::a64::bus::{COUNTER_FREQUENCY, CTR_EL0, DCZID_EL0};
use nx_runtime::a64::inst::*;
use nx_runtime::a64::interp::{FPCR_MASK, FPSR_MASK};

#[derive(Clone, Copy)]
pub struct Flags {
    pub n: ValueId,
    pub z: ValueId,
    pub c: ValueId,
    pub v: ValueId,
}

pub struct Lifter<'a> {
    pub b: &'a mut Builder,
    pub blk: BlockId,
}

fn ty_of(sf: bool) -> Ty {
    if sf {
        Ty::I64
    } else {
        Ty::I32
    }
}

fn width(sf: bool) -> u32 {
    if sf {
        64
    } else {
        32
    }
}

impl<'a> Lifter<'a> {
    pub fn new(b: &'a mut Builder, blk: BlockId) -> Lifter<'a> {
        Lifter { b, blk }
    }

    // ---- small helpers ----

    fn k(&mut self, ty: Ty, v: u128) -> ValueId {
        self.b.konst(ty, v)
    }

    pub fn k64(&mut self, v: u64) -> ValueId {
        self.b.konst(Ty::I64, u128::from(v))
    }

    fn kb(&mut self, v: bool) -> ValueId {
        self.b.konst(Ty::I1, u128::from(v))
    }

    fn add(&mut self, x: ValueId, y: ValueId) -> ValueId {
        self.b.add(self.blk, x, y)
    }
    fn sub(&mut self, x: ValueId, y: ValueId) -> ValueId {
        self.b.sub(self.blk, x, y)
    }
    fn and(&mut self, x: ValueId, y: ValueId) -> ValueId {
        self.b.and(self.blk, x, y)
    }
    fn or(&mut self, x: ValueId, y: ValueId) -> ValueId {
        self.b.or(self.blk, x, y)
    }
    fn xor(&mut self, x: ValueId, y: ValueId) -> ValueId {
        self.b.xor(self.blk, x, y)
    }
    fn not(&mut self, x: ValueId) -> ValueId {
        self.b.not(self.blk, x)
    }
    fn icmp(&mut self, p: Pred, x: ValueId, y: ValueId) -> ValueId {
        self.b.icmp(self.blk, p, x, y)
    }
    fn select(&mut self, c: ValueId, x: ValueId, y: ValueId) -> ValueId {
        self.b.select(self.blk, c, x, y)
    }
    fn zext(&mut self, x: ValueId, to: Ty) -> ValueId {
        self.b.zext(self.blk, x, to)
    }
    fn sext(&mut self, x: ValueId, to: Ty) -> ValueId {
        self.b.sext(self.blk, x, to)
    }
    fn trunc(&mut self, x: ValueId, to: Ty) -> ValueId {
        self.b.trunc(self.blk, x, to)
    }
    fn shl_c(&mut self, x: ValueId, n: u32) -> ValueId {
        self.b.shl_c(self.blk, x, n)
    }
    fn lshr_c(&mut self, x: ValueId, n: u32) -> ValueId {
        self.b.lshr_c(self.blk, x, n)
    }
    fn ashr_c(&mut self, x: ValueId, n: u32) -> ValueId {
        self.b.ashr_c(self.blk, x, n)
    }

    // ---- registers ----

    /// X register `n` as 64 bits; 31 is the stack pointer when `sp`, else zero.
    pub fn x(&mut self, n: u8, sp: bool) -> ValueId {
        if n == 31 {
            if sp {
                self.b.read(Var::Sp, self.blk)
            } else {
                self.k64(0)
            }
        } else {
            self.b.read(Var::X(n), self.blk)
        }
    }

    /// The register at the instruction's width: `i64` for X, `i32` (the low half) for W.
    fn xw(&mut self, sf: bool, n: u8, sp: bool) -> ValueId {
        let v = self.x(n, sp);
        if sf {
            v
        } else {
            self.trunc(v, Ty::I32)
        }
    }

    pub fn set_x(&mut self, n: u8, v: ValueId, sp: bool) {
        debug_assert_eq!(self.b.ty(v), Ty::I64);
        if n == 31 {
            if sp {
                self.b.write(Var::Sp, self.blk, v);
            }
        } else {
            self.b.write(Var::X(n), self.blk, v);
        }
    }

    /// Write a result of the instruction's width; a W result zeroes the upper half.
    fn set_xw(&mut self, sf: bool, n: u8, v: ValueId, sp: bool) {
        let v = if sf { v } else { self.zext(v, Ty::I64) };
        self.set_x(n, v, sp);
    }

    fn flags(&mut self) -> Flags {
        Flags {
            n: self.b.read(Var::N, self.blk),
            z: self.b.read(Var::Z, self.blk),
            c: self.b.read(Var::C, self.blk),
            v: self.b.read(Var::V, self.blk),
        }
    }

    fn set_flags(&mut self, f: Flags) {
        self.b.write(Var::N, self.blk, f.n);
        self.b.write(Var::Z, self.blk, f.z);
        self.b.write(Var::C, self.blk, f.c);
        self.b.write(Var::V, self.blk, f.v);
    }

    /// `ConditionHolds`.
    pub fn condition(&mut self, cond: u8) -> ValueId {
        let f = self.flags();
        let r = match cond >> 1 {
            0 => f.z,
            1 => f.c,
            2 => f.n,
            3 => f.v,
            4 => {
                let nz = self.not(f.z);
                self.and(f.c, nz)
            }
            5 => self.icmp(Pred::Eq, f.n, f.v),
            6 => {
                let ge = self.icmp(Pred::Eq, f.n, f.v);
                let nz = self.not(f.z);
                self.and(ge, nz)
            }
            _ => self.kb(true),
        };
        if cond & 1 == 1 && cond != 0b1111 {
            self.not(r)
        } else {
            r
        }
    }

    // ---- arithmetic with flags ----

    fn nz(&mut self, res: ValueId) -> (ValueId, ValueId) {
        let zero = self.k(self.b.ty(res), 0);
        (
            self.icmp(Pred::Slt, res, zero),
            self.icmp(Pred::Eq, res, zero),
        )
    }

    /// `x + y`, with the flags `AddWithCarry(x, y, 0)` gives.
    fn add_flags(&mut self, x: ValueId, y: ValueId) -> (ValueId, Flags) {
        let res = self.add(x, y);
        let (n, z) = self.nz(res);
        let c = self.icmp(Pred::Ult, res, x);
        let (a, b) = (self.xor(x, res), self.xor(y, res));
        let both = self.and(a, b);
        let zero = self.k(self.b.ty(res), 0);
        let v = self.icmp(Pred::Slt, both, zero);
        (res, Flags { n, z, c, v })
    }

    /// `x - y`, with the flags `AddWithCarry(x, NOT(y), 1)` gives.
    fn sub_flags(&mut self, x: ValueId, y: ValueId) -> (ValueId, Flags) {
        let res = self.sub(x, y);
        let (n, z) = self.nz(res);
        let c = self.icmp(Pred::Uge, x, y);
        let (a, b) = (self.xor(x, y), self.xor(x, res));
        let both = self.and(a, b);
        let zero = self.k(self.b.ty(res), 0);
        let v = self.icmp(Pred::Slt, both, zero);
        (res, Flags { n, z, c, v })
    }

    /// `AddWithCarry(x, y, cin)` for an arbitrary carry-in `cin` (`i1`); `y` is the operand as
    /// added, already inverted for a subtraction.
    fn adc_flags(&mut self, x: ValueId, y: ValueId, cin: ValueId) -> (ValueId, Flags) {
        let ty = self.b.ty(x);
        let cin_w = self.zext(cin, ty);
        let partial = self.add(x, y);
        let res = self.add(partial, cin_w);
        let (n, z) = self.nz(res);
        let wrapped = self.icmp(Pred::Ult, res, x);
        let same = self.icmp(Pred::Eq, res, x);
        let carried = self.and(cin, same);
        let c = self.or(wrapped, carried);
        let (a, b) = (self.xor(x, res), self.xor(y, res));
        let both = self.and(a, b);
        let zero = self.k(ty, 0);
        let v = self.icmp(Pred::Slt, both, zero);
        (res, Flags { n, z, c, v })
    }

    fn logic_flags(&mut self, res: ValueId) -> Flags {
        let (n, z) = self.nz(res);
        let off = self.kb(false);
        Flags {
            n,
            z,
            c: off,
            v: off,
        }
    }

    // ---- operands ----

    /// `ShiftReg`: shift an already-read value by an immediate amount below the width.
    fn shift(&mut self, v: ValueId, ty: ShiftType, amount: u8) -> ValueId {
        let amount = u32::from(amount);
        if amount == 0 {
            return v;
        }
        match ty {
            ShiftType::Lsl => self.shl_c(v, amount),
            ShiftType::Lsr => self.lshr_c(v, amount),
            ShiftType::Asr => self.ashr_c(v, amount),
            ShiftType::Ror => {
                let k = self.k(self.b.ty(v), u128::from(amount));
                self.b.fshr(self.blk, v, v, k)
            }
        }
    }

    /// `ExtendReg` before the final truncation: extend the low bits of a 64-bit register value to
    /// 64 bits and shift left by `shift`.
    fn extend64(&mut self, v: ValueId, ext: ExtendType, shift: u8) -> ValueId {
        let e = match ext {
            ExtendType::Uxtx | ExtendType::Sxtx => v,
            ExtendType::Uxtb => self.and_const(v, 0xFF),
            ExtendType::Uxth => self.and_const(v, 0xFFFF),
            ExtendType::Uxtw => self.and_const(v, 0xFFFF_FFFF),
            ExtendType::Sxtb => {
                let t = self.trunc(v, Ty::I8);
                self.sext(t, Ty::I64)
            }
            ExtendType::Sxth => {
                let t = self.trunc(v, Ty::I16);
                self.sext(t, Ty::I64)
            }
            ExtendType::Sxtw => {
                let t = self.trunc(v, Ty::I32);
                self.sext(t, Ty::I64)
            }
        };
        if shift == 0 {
            e
        } else {
            self.shl_c(e, u32::from(shift))
        }
    }

    fn and_const(&mut self, v: ValueId, c: u64) -> ValueId {
        let k = self.k(self.b.ty(v), u128::from(c));
        self.and(v, k)
    }

    // ---- addressing ----

    /// The access address of a load/store and the value to write back to `Rn`, if any.
    fn address(&mut self, pc: u64, rn: u8, mode: AddrMode) -> (ValueId, Option<ValueId>) {
        match mode {
            AddrMode::Offset(imm) => {
                let base = self.x(rn, true);
                let k = self.k64(imm as u64);
                (self.add(base, k), None)
            }
            AddrMode::PreIndex(imm) => {
                let base = self.x(rn, true);
                let k = self.k64(imm as u64);
                let a = self.add(base, k);
                (a, Some(a))
            }
            AddrMode::PostIndex(imm) => {
                let base = self.x(rn, true);
                let k = self.k64(imm as u64);
                let wb = self.add(base, k);
                (base, Some(wb))
            }
            AddrMode::PostIndexReg(rm) => {
                let base = self.x(rn, true);
                let m = self.x(rm, false);
                let wb = self.add(base, m);
                (base, Some(wb))
            }
            AddrMode::Register { rm, extend, amount } => {
                let m = self.x(rm, false);
                let off = self.extend64(m, extend, amount);
                let base = self.x(rn, true);
                (self.add(base, off), None)
            }
            AddrMode::Literal(imm) => (self.k64(pc.wrapping_add(imm as u64)), None),
        }
    }

    // ---- the instructions ----

    /// Lift an instruction that falls through to the next one. Branches, calls, traps and the
    /// instructions the interpreter runs are the block driver's business.
    pub fn straight(&mut self, pc: u64, inst: &Inst) {
        let blk = self.blk;
        match *inst {
            Inst::Adr { rd, imm, page } => {
                let base = if page { pc & !0xFFF } else { pc };
                let v = self.k64(base.wrapping_add(imm as u64));
                self.set_x(rd, v, false);
            }
            Inst::AddSubImm {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                let ty = ty_of(sf);
                let a = self.xw(sf, rn, true);
                let imm = self.k(ty, u128::from(imm));
                let res = if set_flags {
                    let (res, f) = if sub {
                        self.sub_flags(a, imm)
                    } else {
                        self.add_flags(a, imm)
                    };
                    self.set_flags(f);
                    res
                } else if sub {
                    self.sub(a, imm)
                } else {
                    self.add(a, imm)
                };
                self.set_xw(sf, rd, res, !set_flags);
            }
            Inst::LogicalImm {
                sf,
                op,
                rd,
                rn,
                imm,
            } => {
                let ty = ty_of(sf);
                let a = self.xw(sf, rn, false);
                let k = self.k(ty, u128::from(imm));
                let res = match op {
                    LogicOp::And | LogicOp::Ands => self.and(a, k),
                    LogicOp::Orr => self.or(a, k),
                    LogicOp::Eor => self.xor(a, k),
                };
                if op == LogicOp::Ands {
                    let f = self.logic_flags(res);
                    self.set_flags(f);
                }
                self.set_xw(sf, rd, res, op != LogicOp::Ands);
            }
            Inst::MovWide {
                sf,
                op,
                rd,
                imm16,
                shift,
            } => {
                let ty = ty_of(sf);
                let imm = u128::from(imm16) << shift;
                let res = match op {
                    MovOp::Movn => self.k(ty, !imm),
                    MovOp::Movz => self.k(ty, imm),
                    MovOp::Movk => {
                        let old = self.xw(sf, rd, false);
                        let keep = self.k(ty, !(0xFFFFu128 << shift));
                        let kept = self.and(old, keep);
                        let new = self.k(ty, imm);
                        self.or(kept, new)
                    }
                };
                self.set_xw(sf, rd, res, false);
            }
            Inst::Bitfield {
                sf,
                op,
                rd,
                rn,
                immr,
                imms,
            } => {
                let ty = ty_of(sf);
                let w = width(sf);
                let (wmask, tmask) =
                    decode_bit_masks(sf, imms, immr, false, w).expect("validated by decode");
                let (inzero, extend) = match op {
                    BitfieldOp::Sbfm => (true, true),
                    BitfieldOp::Bfm => (false, false),
                    BitfieldOp::Ubfm => (true, false),
                };
                let dst = if inzero {
                    self.k(ty, 0)
                } else {
                    self.xw(sf, rd, false)
                };
                let src = self.xw(sf, rn, false);
                let amt = self.k(ty, u128::from(immr));
                let rotated = self.b.fshr(blk, src, src, amt);
                let wk = self.k(ty, u128::from(wmask));
                let nwk = self.k(ty, u128::from(!wmask));
                let kept = self.and(dst, nwk);
                let moved = self.and(rotated, wk);
                let bot = self.or(kept, moved);
                let top = if extend {
                    // Replicate the source bit `imms` across the width.
                    let up = self.shl_c(src, w - 1 - u32::from(imms));
                    self.ashr_c(up, w - 1)
                } else {
                    dst
                };
                let tk = self.k(ty, u128::from(tmask));
                let ntk = self.k(ty, u128::from(!tmask));
                let hi = self.and(top, ntk);
                let lo = self.and(bot, tk);
                let res = self.or(hi, lo);
                self.set_xw(sf, rd, res, false);
            }
            Inst::Extr {
                sf,
                rd,
                rn,
                rm,
                lsb,
            } => {
                let ty = ty_of(sf);
                let hi = self.xw(sf, rn, false);
                let lo = self.xw(sf, rm, false);
                let amt = self.k(ty, u128::from(lsb));
                let res = self.b.fshr(blk, hi, lo, amt);
                self.set_xw(sf, rd, res, false);
            }

            Inst::Hint { .. } => {}
            Inst::Barrier { kind, crm } => {
                let fence = match kind {
                    BarrierKind::Isb => Fence::Instruction,
                    BarrierKind::Dsb => Fence::Full,
                    BarrierKind::Dmb => match crm & 3 {
                        1 => Fence::Acquire,
                        2 => Fence::Release,
                        _ => Fence::Full,
                    },
                };
                self.b.fence(blk, fence);
            }
            Inst::Clrex => self.b.clrex(blk),
            Inst::Mrs { rt, reg } => {
                let v = match reg {
                    SysReg::Nzcv => {
                        let p = self.pack_nzcv();
                        self.zext(p, Ty::I64)
                    }
                    SysReg::Fpcr => {
                        let f = self.b.load_field(blk, Field::Fpcr);
                        self.zext(f, Ty::I64)
                    }
                    SysReg::Fpsr => {
                        let f = self.b.load_field(blk, Field::Fpsr);
                        self.zext(f, Ty::I64)
                    }
                    SysReg::TpidrEl0 => self.b.load_field(blk, Field::TpidrEl0),
                    SysReg::TpidrroEl0 => self.b.load_field(blk, Field::TpidrroEl0),
                    SysReg::CntfrqEl0 => self.k64(COUNTER_FREQUENCY),
                    SysReg::CntpctEl0 | SysReg::CntvctEl0 => self.b.counter(blk),
                    SysReg::CtrEl0 => self.k64(CTR_EL0),
                    SysReg::DczidEl0 => self.k64(DCZID_EL0),
                    SysReg::Other(..) => unreachable!("traps"),
                };
                self.set_x(rt, v, false);
            }
            Inst::Msr { rt, reg } => {
                let v = self.x(rt, false);
                match reg {
                    SysReg::Nzcv => {
                        let one = self.k64(1);
                        let bit = |l: &mut Self, pos: u32| {
                            let s = l.lshr_c(v, pos);
                            let m = l.and(s, one);
                            let z = l.k64(0);
                            l.icmp(Pred::Ne, m, z)
                        };
                        let f = Flags {
                            n: bit(self, 31),
                            z: bit(self, 30),
                            c: bit(self, 29),
                            v: bit(self, 28),
                        };
                        self.set_flags(f);
                    }
                    SysReg::Fpcr => {
                        let t = self.trunc(v, Ty::I32);
                        let masked = self.and_const(t, u64::from(FPCR_MASK));
                        self.b.store_field(blk, Field::Fpcr, masked);
                    }
                    SysReg::Fpsr => {
                        let t = self.trunc(v, Ty::I32);
                        let masked = self.and_const(t, u64::from(FPSR_MASK));
                        self.b.store_field(blk, Field::Fpsr, masked);
                    }
                    SysReg::TpidrEl0 => self.b.store_field(blk, Field::TpidrEl0, v),
                    _ => unreachable!("traps"),
                }
            }
            Inst::Sys {
                load: false,
                op1: 3,
                crn: 7,
                crm: 4,
                op2: 1,
                rt,
            } => {
                let a = self.x(rt, false);
                self.b.dc_zva(blk, pc, a);
            }
            // The other implemented cache maintenance operations have no architectural effect.
            Inst::Sys { .. } => {}

            Inst::LdSt {
                kind,
                size,
                rt,
                rn,
                mode,
            } => {
                let bytes = 1u8 << size;
                if kind == MemKind::Prefetch {
                    return;
                }
                let (addr, wb) = self.address(pc, rn, mode);
                match kind {
                    MemKind::Store => {
                        let v = self.x(rt, false);
                        let v = self.narrow(v, bytes);
                        self.b.store(blk, bytes, Order::Plain, pc, addr, v);
                    }
                    MemKind::Load | MemKind::LoadSigned32 | MemKind::LoadSigned64 => {
                        let raw = self.b.load(blk, bytes, Order::Plain, pc, addr);
                        let v = match kind {
                            MemKind::Load => self.zext(raw, Ty::I64),
                            MemKind::LoadSigned32 => {
                                let s = self.sext(raw, Ty::I32);
                                self.zext(s, Ty::I64)
                            }
                            _ => self.sext(raw, Ty::I64),
                        };
                        self.set_x(rt, v, false);
                    }
                    MemKind::Prefetch => unreachable!(),
                }
                if let Some(wb) = wb {
                    self.set_x(rn, wb, true);
                }
            }
            Inst::LdStPair {
                load,
                signed_word,
                size,
                rt,
                rt2,
                rn,
                imm,
                writeback,
                post_index,
            } => {
                let esize = 1u8 << size;
                let base = self.x(rn, true);
                let off = self.k64(imm as u64);
                let addr = if post_index {
                    base
                } else {
                    self.add(base, off)
                };
                let second = self.k64(u64::from(esize));
                let addr2 = self.add(addr, second);
                if load {
                    let a = self.b.load(blk, esize, Order::Plain, pc, addr);
                    let b2 = self.b.load(blk, esize, Order::Plain, pc, addr2);
                    let (a, b2) = if signed_word {
                        (self.sext(a, Ty::I64), self.sext(b2, Ty::I64))
                    } else {
                        (self.zext(a, Ty::I64), self.zext(b2, Ty::I64))
                    };
                    self.set_x(rt, a, false);
                    self.set_x(rt2, b2, false);
                } else {
                    let a = self.x(rt, false);
                    let a = self.narrow(a, esize);
                    let b2 = self.x(rt2, false);
                    let b2 = self.narrow(b2, esize);
                    self.b.store(blk, esize, Order::Plain, pc, addr, a);
                    self.b.store(blk, esize, Order::Plain, pc, addr2, b2);
                }
                if writeback {
                    let wb = self.add(base, off);
                    self.set_x(rn, wb, true);
                }
            }
            Inst::Exclusive {
                load,
                pair,
                size,
                acquire,
                release,
                rs,
                rt,
                rt2,
                rn,
            } => {
                let esize = 1u8 << size;
                let bytes = if pair { esize * 2 } else { esize };
                let ebits = 8 * u32::from(esize);
                let addr = self.x(rn, true);
                if load {
                    let v = self.b.load_excl(blk, bytes, acquire, pc, addr);
                    if pair {
                        // Two elements packed little-endian into one value.
                        let (lo, hi) = if bytes == 16 {
                            let lo = self.trunc(v, Ty::I64);
                            let sh = self.lshr_c(v, 64);
                            (lo, self.trunc(sh, Ty::I64))
                        } else {
                            let lo = self.and_const(v, (1u64 << ebits) - 1);
                            (lo, self.lshr_c(v, ebits))
                        };
                        self.set_x(rt, lo, false);
                        self.set_x(rt2, hi, false);
                    } else {
                        self.set_x(rt, v, false);
                    }
                } else {
                    let lo = self.x(rt, false);
                    let data = if bytes == 16 {
                        let lo = self.zext(lo, Ty::I128);
                        let hi = self.x(rt2, false);
                        let hi = self.zext(hi, Ty::I128);
                        let hi = self.shl_c(hi, 64);
                        self.or(lo, hi)
                    } else if pair {
                        let lo = self.and_const(lo, (1u64 << ebits) - 1);
                        let hi = self.x(rt2, false);
                        let hi = self.and_const(hi, (1u64 << ebits) - 1);
                        let hi = self.shl_c(hi, ebits);
                        self.or(lo, hi)
                    } else if ebits == 64 {
                        lo
                    } else {
                        self.and_const(lo, (1u64 << ebits) - 1)
                    };
                    let status = self.b.store_excl(blk, bytes, release, pc, addr, data);
                    let status = self.zext(status, Ty::I64);
                    self.set_x(rs, status, false);
                }
            }
            Inst::AcqRel { load, size, rt, rn } => {
                let bytes = 1u8 << size;
                let addr = self.x(rn, true);
                if load {
                    let raw = self.b.load(blk, bytes, Order::Acquire, pc, addr);
                    let v = self.zext(raw, Ty::I64);
                    self.set_x(rt, v, false);
                } else {
                    let v = self.x(rt, false);
                    let v = self.narrow(v, bytes);
                    self.b.store(blk, bytes, Order::Release, pc, addr, v);
                }
            }

            Inst::LogicalShifted {
                sf,
                op,
                invert,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let a = self.xw(sf, rn, false);
                let m = self.xw(sf, rm, false);
                let mut b2 = self.shift(m, shift, amount);
                if invert {
                    b2 = self.not(b2);
                }
                let res = match op {
                    LogicOp::And | LogicOp::Ands => self.and(a, b2),
                    LogicOp::Orr => self.or(a, b2),
                    LogicOp::Eor => self.xor(a, b2),
                };
                if op == LogicOp::Ands {
                    let f = self.logic_flags(res);
                    self.set_flags(f);
                }
                self.set_xw(sf, rd, res, false);
            }
            Inst::AddSubShifted {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let a = self.xw(sf, rn, false);
                let m = self.xw(sf, rm, false);
                let b2 = self.shift(m, shift, amount);
                let res = self.add_sub(a, b2, sub, set_flags);
                self.set_xw(sf, rd, res, false);
            }
            Inst::AddSubExtended {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let a = self.xw(sf, rn, true);
                let m = self.x(rm, false);
                let e = self.extend64(m, extend, amount);
                let b2 = if sf { e } else { self.trunc(e, Ty::I32) };
                let res = self.add_sub(a, b2, sub, set_flags);
                self.set_xw(sf, rd, res, !set_flags);
            }
            Inst::AddSubCarry {
                sf,
                sub,
                set_flags,
                rd,
                rn,
                rm,
            } => {
                let a = self.xw(sf, rn, false);
                let mut b2 = self.xw(sf, rm, false);
                if sub {
                    b2 = self.not(b2);
                }
                let cin = self.b.read(Var::C, blk);
                let res = if set_flags {
                    let (res, f) = self.adc_flags(a, b2, cin);
                    self.set_flags(f);
                    res
                } else {
                    let cin_w = self.zext(cin, ty_of(sf));
                    let partial = self.add(a, b2);
                    self.add(partial, cin_w)
                };
                self.set_xw(sf, rd, res, false);
            }
            Inst::CondCmp {
                sf,
                negative,
                imm_operand,
                rn,
                rm_or_imm,
                cond,
                nzcv,
            } => {
                let ty = ty_of(sf);
                let holds = self.condition(cond);
                let a = self.xw(sf, rn, false);
                let b2 = if imm_operand {
                    self.k(ty, u128::from(rm_or_imm))
                } else {
                    self.xw(sf, rm_or_imm, false)
                };
                let (_, computed) = if negative {
                    self.sub_flags(a, b2)
                } else {
                    self.add_flags(a, b2)
                };
                let bit = |i: u32| nzcv & (1 << i) != 0;
                let kn = self.kb(bit(3));
                let kz = self.kb(bit(2));
                let kc = self.kb(bit(1));
                let kv = self.kb(bit(0));
                let f = Flags {
                    n: self.select(holds, computed.n, kn),
                    z: self.select(holds, computed.z, kz),
                    c: self.select(holds, computed.c, kc),
                    v: self.select(holds, computed.v, kv),
                };
                self.set_flags(f);
            }
            Inst::CondSel {
                sf,
                op,
                rd,
                rn,
                rm,
                cond,
            } => {
                let ty = ty_of(sf);
                let holds = self.condition(cond);
                let a = self.xw(sf, rn, false);
                let m = self.xw(sf, rm, false);
                let other = match op {
                    CondSelOp::Csel => m,
                    CondSelOp::Csinc => {
                        let one = self.k(ty, 1);
                        self.add(m, one)
                    }
                    CondSelOp::Csinv => self.not(m),
                    CondSelOp::Csneg => {
                        let zero = self.k(ty, 0);
                        self.sub(zero, m)
                    }
                };
                let res = self.select(holds, a, other);
                self.set_xw(sf, rd, res, false);
            }
            Inst::DataProc1 { sf, op, rd, rn } => {
                let ty = ty_of(sf);
                let w = width(sf);
                let v = self.xw(sf, rn, false);
                let res = match op {
                    Dp1Op::Rbit => self.b.un(blk, UnOp::Rbit, v),
                    Dp1Op::Rev => self.b.un(blk, UnOp::Bswap, v),
                    Dp1Op::Rev16 => {
                        let lanes = if sf {
                            0x00FF_00FF_00FF_00FFu64
                        } else {
                            0x00FF_00FF
                        };
                        let lo = self.and_const(v, lanes);
                        let lo = self.shl_c(lo, 8);
                        let hi = self.lshr_c(v, 8);
                        let hi = self.and_const(hi, lanes);
                        self.or(lo, hi)
                    }
                    Dp1Op::Rev32 => {
                        let swapped = self.b.un(blk, UnOp::Bswap, v);
                        let k = self.k(ty, 32);
                        self.b.fshr(blk, swapped, swapped, k)
                    }
                    Dp1Op::Clz => self.b.un(blk, UnOp::Clz, v),
                    Dp1Op::Cls => {
                        // CLZ(v<w-1:1> EOR v<w-2:0>) over w-1 bits, as w-bit arithmetic.
                        let shifted = self.lshr_c(v, 1);
                        let e = self.xor(shifted, v);
                        let low = self.k(ty, (1u128 << (w - 1)) - 1);
                        let e = self.and(e, low);
                        let z = self.b.un(blk, UnOp::Clz, e);
                        let one = self.k(ty, 1);
                        self.sub(z, one)
                    }
                };
                self.set_xw(sf, rd, res, false);
            }
            Inst::DataProc2 { sf, op, rd, rn, rm } => {
                let w = width(sf);
                let a = self.xw(sf, rn, false);
                let m = self.xw(sf, rm, false);
                let res = match op {
                    Dp2Op::Udiv => self.b.bin(blk, BinOp::UDiv, a, m),
                    Dp2Op::Sdiv => self.b.bin(blk, BinOp::SDiv, a, m),
                    Dp2Op::Lslv | Dp2Op::Lsrv | Dp2Op::Asrv => {
                        let amt = self.and_const(m, u64::from(w - 1));
                        let o = match op {
                            Dp2Op::Lslv => BinOp::Shl,
                            Dp2Op::Lsrv => BinOp::LShr,
                            _ => BinOp::AShr,
                        };
                        self.b.bin(blk, o, a, amt)
                    }
                    Dp2Op::Rorv => self.b.fshr(blk, a, a, m),
                    Dp2Op::Crc32 { size, c } => {
                        let acc = self.trunc(a, Ty::I32);
                        let data = if sf { m } else { self.zext(m, Ty::I64) };
                        let r = self.b.crc32(blk, size, c, acc, data);
                        // The result is 32 bits wide whatever the operand width.
                        let r = self.zext(r, Ty::I64);
                        self.set_x(rd, r, false);
                        return;
                    }
                };
                self.set_xw(sf, rd, res, false);
            }
            Inst::DataProc3 {
                sf,
                op,
                rd,
                rn,
                rm,
                ra,
            } => match op {
                Dp3Op::Madd | Dp3Op::Msub => {
                    let a = self.xw(sf, rn, false);
                    let m = self.xw(sf, rm, false);
                    let acc = self.xw(sf, ra, false);
                    let prod = self.b.mul(blk, a, m);
                    let res = if op == Dp3Op::Madd {
                        self.add(acc, prod)
                    } else {
                        self.sub(acc, prod)
                    };
                    self.set_xw(sf, rd, res, false);
                }
                Dp3Op::Smaddl | Dp3Op::Smsubl | Dp3Op::Umaddl | Dp3Op::Umsubl => {
                    let signed = matches!(op, Dp3Op::Smaddl | Dp3Op::Smsubl);
                    let a = self.x(rn, false);
                    let m = self.x(rm, false);
                    let a = self.trunc(a, Ty::I32);
                    let m = self.trunc(m, Ty::I32);
                    let (a, m) = if signed {
                        (self.sext(a, Ty::I64), self.sext(m, Ty::I64))
                    } else {
                        (self.zext(a, Ty::I64), self.zext(m, Ty::I64))
                    };
                    let prod = self.b.mul(blk, a, m);
                    let acc = self.x(ra, false);
                    let res = if matches!(op, Dp3Op::Smaddl | Dp3Op::Umaddl) {
                        self.add(acc, prod)
                    } else {
                        self.sub(acc, prod)
                    };
                    self.set_x(rd, res, false);
                }
                Dp3Op::Smulh | Dp3Op::Umulh => {
                    let a = self.x(rn, false);
                    let m = self.x(rm, false);
                    let res = self.b.mulhi(blk, op == Dp3Op::Smulh, a, m);
                    self.set_x(rd, res, false);
                }
            },

            Inst::Vec { word, op } => {
                let src = op.int_source().map(|r| self.x(r, false));
                let nzcv = if op.reads_nzcv() {
                    let packed = self.pack_nzcv();
                    let nibble = self.lshr_c(packed, 28);
                    Some(self.zext(nibble, Ty::I64))
                } else {
                    None
                };
                let result = self.b.vec_op(blk, word, src, nzcv);
                if let Some(rd) = op.int_dest() {
                    self.set_x(rd, result, false);
                } else if op.writes_nzcv() {
                    let flag = |l: &mut Self, bit: u32| {
                        let shifted = l.lshr_c(result, bit);
                        l.trunc(shifted, Ty::I1)
                    };
                    let f = Flags {
                        n: flag(self, 3),
                        z: flag(self, 2),
                        c: flag(self, 1),
                        v: flag(self, 0),
                    };
                    self.set_flags(f);
                }
            }
            Inst::VecMem { word, addr, .. } => {
                let (a, wb) = self.address(pc, addr.rn, addr.mode);
                self.b.vec_mem(blk, word, pc, a);
                if let Some(wb) = wb {
                    self.set_x(addr.rn, wb, true);
                }
            }

            Inst::B { .. }
            | Inst::BCond { .. }
            | Inst::Cbz { .. }
            | Inst::Tbz { .. }
            | Inst::BranchReg { .. }
            | Inst::Svc { .. }
            | Inst::Brk { .. }
            | Inst::Unallocated(_)
            | Inst::Unsupported { .. } => {
                unreachable!("{inst:?} is a terminator, handled by the block driver")
            }
        }
    }

    /// Truncate a 64-bit register value to the size of a memory access.
    fn narrow(&mut self, v: ValueId, bytes: u8) -> ValueId {
        if bytes == 8 {
            v
        } else {
            self.trunc(v, Ty::of_bytes(bytes))
        }
    }

    /// `a + b` or `a - b` (with `b` as written), setting the flags when asked.
    fn add_sub(&mut self, a: ValueId, b2: ValueId, sub: bool, set_flags: bool) -> ValueId {
        if set_flags {
            let (res, f) = if sub {
                self.sub_flags(a, b2)
            } else {
                self.add_flags(a, b2)
            };
            self.set_flags(f);
            res
        } else if sub {
            self.sub(a, b2)
        } else {
            self.add(a, b2)
        }
    }

    /// `N:Z:C:V` as bits 31 to 28 of an `i32`.
    pub fn pack_nzcv(&mut self) -> ValueId {
        let f = self.flags();
        let mut out = self.k(Ty::I32, 0);
        for (flag, pos) in [(f.n, 31), (f.z, 30), (f.c, 29), (f.v, 28)] {
            let z = self.zext(flag, Ty::I32);
            let s = self.shl_c(z, pos);
            out = self.or(out, s);
        }
        out
    }

    /// The condition of a conditional branch instruction, `true` when it is taken.
    pub fn branch_condition(&mut self, inst: &Inst) -> ValueId {
        match *inst {
            Inst::BCond { cond, .. } => self.condition(cond),
            Inst::Cbz {
                sf, nonzero, rt, ..
            } => {
                let v = self.xw(sf, rt, false);
                let zero = self.b.is_zero(self.blk, v);
                if nonzero {
                    self.not(zero)
                } else {
                    zero
                }
            }
            Inst::Tbz {
                nonzero, rt, bit, ..
            } => {
                let v = self.x(rt, false);
                let mask = self.k64(1u64 << bit);
                let masked = self.and(v, mask);
                let zero = self.k64(0);
                let set = self.icmp(Pred::Ne, masked, zero);
                if nonzero {
                    set
                } else {
                    self.not(set)
                }
            }
            _ => unreachable!("not a conditional branch"),
        }
    }
}
