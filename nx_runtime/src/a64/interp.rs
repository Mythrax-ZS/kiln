//! A64 interpreter for the base (integer) instruction set.
//!
//! This is the oracle every other component is tested against, so each arm follows the Arm
//! pseudocode as literally as the language allows. It is also what runs any code the lifter does
//! not translate.
//!
//! Precise exceptions: a trap leaves `pc` at the trapping instruction (except `SVC`, which resumes
//! after itself) and no register changed; a store that faults halfway through a pair may have
//! completed its first half.

use super::bits::*;
use super::bus::{Bus, Fault, FaultKind, ZVA_BLOCK};
use super::decode::decode;
use super::inst::*;
use super::state::CpuState;

/// Why `step` stopped short of finishing an instruction normally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    /// `SVC #imm`; `pc` already points after it.
    Svc(u16),
    /// `BRK #imm`; `pc` points at it.
    Brk(u16),
    /// An unallocated encoding.
    Undefined(u32),
    /// A valid encoding this project does not implement.
    Unsupported { word: u32, what: &'static str },
    /// A memory access failed.
    Fault(Fault),
    /// A system register access the model does not implement or EL0 may not perform.
    SysReg { reg: SysReg, write: bool },
    /// A `SYS`/`SYSL` operation the model does not implement.
    SysInst {
        load: bool,
        op1: u8,
        crn: u8,
        crm: u8,
        op2: u8,
    },
}

impl From<Fault> for Trap {
    fn from(f: Fault) -> Trap {
        Trap::Fault(f)
    }
}

fn width(sf: bool) -> u32 {
    if sf {
        64
    } else {
        32
    }
}

fn check_align(addr: u64, bytes: u64, write: bool) -> Result<(), Trap> {
    if !addr.is_multiple_of(bytes) {
        Err(Trap::Fault(Fault {
            addr,
            write,
            kind: FaultKind::Alignment,
        }))
    } else {
        Ok(())
    }
}

/// Writable FPCR bits for an ARMv8.0 core (docs/ASSUMPTIONS.md A-013).
pub const FPCR_MASK: u32 = 0x07C0_9F00;
/// Writable FPSR bits: QC and the cumulative exception flags.
pub const FPSR_MASK: u32 = 0x0800_009F;

/// Fetch, decode and execute one instruction.
pub fn step<B: Bus>(cpu: &mut CpuState, bus: &mut B) -> Result<(), Trap> {
    let word = bus.fetch(cpu.pc)?;
    execute(cpu, bus, decode(word))
}

fn set_logic_flags(cpu: &mut CpuState, res: u64, w: u32) {
    let n = (res >> (w - 1)) & 1 == 1;
    let z = res == 0;
    cpu.set_nzcv4((u8::from(n) << 3) | (u8::from(z) << 2));
}

/// Execute one decoded instruction at `cpu.pc`.
pub fn execute<B: Bus>(cpu: &mut CpuState, bus: &mut B, inst: Inst) -> Result<(), Trap> {
    let pc = cpu.pc;
    let mut next = pc.wrapping_add(4);
    match inst {
        Inst::Adr { rd, imm, page } => {
            let base = if page { pc & !0xFFF } else { pc };
            cpu.set_x_zr(rd, base.wrapping_add(imm as u64));
        }
        Inst::AddSubImm {
            sf,
            sub,
            set_flags,
            rd,
            rn,
            imm,
        } => {
            let w = width(sf);
            let m = mask(w);
            let op1 = cpu.x_sp(rn) & m;
            let imm = u64::from(imm);
            let (op2, cin) = if sub { (!imm & m, true) } else { (imm, false) };
            let (res, flags) = add_with_carry(op1, op2, cin, w);
            if set_flags {
                cpu.set_nzcv4(flags);
                cpu.set_x_zr(rd, res);
            } else {
                cpu.set_x_sp(rd, res);
            }
        }
        Inst::LogicalImm {
            sf,
            op,
            rd,
            rn,
            imm,
        } => {
            let w = width(sf);
            let m = mask(w);
            let op1 = cpu.x_zr(rn) & m;
            let res = match op {
                LogicOp::And | LogicOp::Ands => op1 & imm,
                LogicOp::Orr => op1 | imm,
                LogicOp::Eor => op1 ^ imm,
            } & m;
            if op == LogicOp::Ands {
                set_logic_flags(cpu, res, w);
                cpu.set_x_zr(rd, res);
            } else {
                cpu.set_x_sp(rd, res);
            }
        }
        Inst::MovWide {
            sf,
            op,
            rd,
            imm16,
            shift,
        } => {
            let m = mask(width(sf));
            let imm = u64::from(imm16) << shift;
            let res = match op {
                MovOp::Movn => !imm & m,
                MovOp::Movz => imm,
                MovOp::Movk => (cpu.x_zr(rd) & m & !(0xFFFFu64 << shift)) | imm,
            };
            cpu.set_x_zr(rd, res);
        }
        Inst::Bitfield {
            sf,
            op,
            rd,
            rn,
            immr,
            imms,
        } => {
            let w = width(sf);
            let m = mask(w);
            // The decoder rejected every encoding DecodeBitMasks would call UNDEFINED.
            let (wmask, tmask) =
                decode_bit_masks(sf, imms, immr, false, w).expect("validated by decode");
            let (inzero, extend) = match op {
                BitfieldOp::Sbfm => (true, true),
                BitfieldOp::Bfm => (false, false),
                BitfieldOp::Ubfm => (true, false),
            };
            let dst = if inzero { 0 } else { cpu.x_zr(rd) & m };
            let src = cpu.x_zr(rn) & m;
            let bot = (dst & !wmask) | (ror(src, u32::from(immr), w) & wmask);
            let top = if extend {
                if (src >> imms) & 1 == 1 {
                    m
                } else {
                    0
                }
            } else {
                dst
            };
            cpu.set_x_zr(rd, ((top & !tmask) | (bot & tmask)) & m);
        }
        Inst::Extr {
            sf,
            rd,
            rn,
            rm,
            lsb,
        } => {
            let w = width(sf);
            let m = mask(w);
            let hi = u128::from(cpu.x_zr(rn) & m);
            let lo = u128::from(cpu.x_zr(rm) & m);
            let concat = (hi << w) | lo;
            cpu.set_x_zr(rd, ((concat >> lsb) as u64) & m);
        }

        Inst::B { link, imm } => {
            if link {
                cpu.x[30] = pc.wrapping_add(4);
            }
            next = pc.wrapping_add(imm as u64);
        }
        Inst::BCond { cond, imm } => {
            if condition_holds(cond, cpu.nzcv4()) {
                next = pc.wrapping_add(imm as u64);
            }
        }
        Inst::Cbz {
            sf,
            nonzero,
            rt,
            imm,
        } => {
            let zero = cpu.x_zr(rt) & mask(width(sf)) == 0;
            if zero != nonzero {
                next = pc.wrapping_add(imm as u64);
            }
        }
        Inst::Tbz {
            nonzero,
            rt,
            bit,
            imm,
        } => {
            let set = (cpu.x_zr(rt) >> bit) & 1 == 1;
            if set == nonzero {
                next = pc.wrapping_add(imm as u64);
            }
        }
        Inst::BranchReg { kind, rn } => {
            let target = cpu.x_zr(rn);
            if kind == BranchReg::Blr {
                cpu.x[30] = pc.wrapping_add(4);
            }
            next = target;
        }
        Inst::Svc { imm } => {
            cpu.pc = next;
            return Err(Trap::Svc(imm));
        }
        Inst::Brk { imm } => return Err(Trap::Brk(imm)),
        Inst::Hint { .. } => {}
        Inst::Barrier { .. } => std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst),
        Inst::Clrex => bus.clear_exclusive(cpu),
        Inst::Mrs { rt, reg } => {
            let v = match reg {
                SysReg::Nzcv => u64::from(cpu.nzcv & 0xF000_0000),
                SysReg::Fpcr => u64::from(cpu.fpcr),
                SysReg::Fpsr => u64::from(cpu.fpsr),
                SysReg::TpidrEl0 => cpu.tpidr_el0,
                SysReg::TpidrroEl0 => cpu.tpidrro_el0,
                other => bus.system_register(other).ok_or(Trap::SysReg {
                    reg: other,
                    write: false,
                })?,
            };
            cpu.set_x_zr(rt, v);
        }
        Inst::Msr { rt, reg } => {
            let v = cpu.x_zr(rt);
            match reg {
                SysReg::Nzcv => cpu.nzcv = (v as u32) & 0xF000_0000,
                SysReg::Fpcr => cpu.fpcr = (v as u32) & FPCR_MASK,
                SysReg::Fpsr => cpu.fpsr = (v as u32) & FPSR_MASK,
                SysReg::TpidrEl0 => cpu.tpidr_el0 = v,
                other => {
                    return Err(Trap::SysReg {
                        reg: other,
                        write: true,
                    })
                }
            }
        }
        Inst::Sys {
            load,
            op1,
            crn,
            crm,
            op2,
            rt,
        } => match (load, op1, crn, crm, op2) {
            // DC ZVA
            (false, 3, 7, 4, 1) => {
                let base = cpu.x_zr(rt) & !(ZVA_BLOCK - 1);
                for i in 0..ZVA_BLOCK / 8 {
                    bus.write(base + i * 8, 8, 0)?;
                }
            }
            // DC CVAC, DC CVAU, DC CIVAC, IC IVAU: cache maintenance has no architectural effect here.
            (false, 3, 7, 10, 1)
            | (false, 3, 7, 11, 1)
            | (false, 3, 7, 14, 1)
            | (false, 3, 7, 5, 1) => {}
            _ => {
                return Err(Trap::SysInst {
                    load,
                    op1,
                    crn,
                    crm,
                    op2,
                })
            }
        },

        Inst::LdSt {
            kind,
            size,
            rt,
            rn,
            mode,
        } => {
            let bytes = 1u8 << size;
            let (addr, writeback) = match mode {
                AddrMode::Offset(imm) => (cpu.x_sp(rn).wrapping_add(imm as u64), None),
                AddrMode::PreIndex(imm) => {
                    let a = cpu.x_sp(rn).wrapping_add(imm as u64);
                    (a, Some(a))
                }
                AddrMode::PostIndex(imm) => {
                    let base = cpu.x_sp(rn);
                    (base, Some(base.wrapping_add(imm as u64)))
                }
                AddrMode::Register { rm, extend, amount } => {
                    let off = extend_value(cpu.x_zr(rm), extend, u32::from(amount), 64);
                    (cpu.x_sp(rn).wrapping_add(off), None)
                }
                AddrMode::Literal(imm) => (pc.wrapping_add(imm as u64), None),
            };
            match kind {
                MemKind::Store => {
                    let v = u128::from(cpu.x_zr(rt) & mask(8 * u32::from(bytes)));
                    bus.write(addr, bytes, v)?;
                }
                MemKind::Load | MemKind::LoadSigned32 | MemKind::LoadSigned64 => {
                    let raw = bus.read(addr, bytes)? as u64;
                    let bits = 8 * u32::from(bytes);
                    let v = match kind {
                        MemKind::Load => raw,
                        MemKind::LoadSigned32 => sign_extend(raw, bits) & 0xFFFF_FFFF,
                        _ => sign_extend(raw, bits),
                    };
                    cpu.set_x_zr(rt, v);
                }
                MemKind::Prefetch => {}
            }
            if let Some(wb) = writeback {
                cpu.set_x_sp(rn, wb);
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
            let base = cpu.x_sp(rn);
            let addr = if post_index {
                base
            } else {
                base.wrapping_add(imm as u64)
            };
            if load {
                let a = bus.read(addr, esize)? as u64;
                let b = bus.read(addr.wrapping_add(u64::from(esize)), esize)? as u64;
                let (a, b) = if signed_word {
                    (sign_extend(a, 32), sign_extend(b, 32))
                } else {
                    (a, b)
                };
                cpu.set_x_zr(rt, a);
                cpu.set_x_zr(rt2, b);
            } else {
                let m = mask(8 * u32::from(esize));
                let a = u128::from(cpu.x_zr(rt) & m);
                let b = u128::from(cpu.x_zr(rt2) & m);
                bus.write(addr, esize, a)?;
                bus.write(addr.wrapping_add(u64::from(esize)), esize, b)?;
            }
            if writeback {
                cpu.set_x_sp(rn, base.wrapping_add(imm as u64));
            }
        }
        Inst::Exclusive {
            load,
            pair,
            size,
            rs,
            rt,
            rt2,
            rn,
            ..
        } => {
            let esize = 1u8 << size;
            let bytes = if pair { esize * 2 } else { esize };
            let addr = cpu.x_sp(rn);
            check_align(addr, u64::from(bytes), !load)?;
            let ebits = 8 * u32::from(esize);
            if load {
                let v = bus.load_exclusive(cpu, addr, bytes)?;
                if pair {
                    cpu.set_x_zr(rt, (v as u64) & mask(ebits));
                    cpu.set_x_zr(rt2, ((v >> ebits) as u64) & mask(ebits));
                } else {
                    cpu.set_x_zr(rt, v as u64);
                }
            } else {
                let lo = u128::from(cpu.x_zr(rt) & mask(ebits));
                let data = if pair {
                    lo | (u128::from(cpu.x_zr(rt2) & mask(ebits)) << ebits)
                } else {
                    lo
                };
                let ok = bus.store_exclusive(cpu, addr, bytes, data)?;
                cpu.set_x_zr(rs, u64::from(!ok));
            }
        }
        Inst::AcqRel { load, size, rt, rn } => {
            let bytes = 1u8 << size;
            let addr = cpu.x_sp(rn);
            check_align(addr, u64::from(bytes), !load)?;
            if load {
                let v = bus.read(addr, bytes)? as u64;
                cpu.set_x_zr(rt, v);
            } else {
                let v = u128::from(cpu.x_zr(rt) & mask(8 * u32::from(bytes)));
                bus.write(addr, bytes, v)?;
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
            let w = width(sf);
            let m = mask(w);
            let op1 = cpu.x_zr(rn) & m;
            let mut op2 = shift_value(cpu.x_zr(rm), shift, u32::from(amount), w);
            if invert {
                op2 = !op2 & m;
            }
            let res = match op {
                LogicOp::And | LogicOp::Ands => op1 & op2,
                LogicOp::Orr => op1 | op2,
                LogicOp::Eor => op1 ^ op2,
            };
            if op == LogicOp::Ands {
                set_logic_flags(cpu, res, w);
            }
            cpu.set_x_zr(rd, res);
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
            let w = width(sf);
            let m = mask(w);
            let op1 = cpu.x_zr(rn) & m;
            let mut op2 = shift_value(cpu.x_zr(rm), shift, u32::from(amount), w);
            if sub {
                op2 = !op2 & m;
            }
            let (res, flags) = add_with_carry(op1, op2, sub, w);
            if set_flags {
                cpu.set_nzcv4(flags);
            }
            cpu.set_x_zr(rd, res);
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
            let w = width(sf);
            let m = mask(w);
            let op1 = cpu.x_sp(rn) & m;
            let mut op2 = extend_value(cpu.x_zr(rm), extend, u32::from(amount), w);
            if sub {
                op2 = !op2 & m;
            }
            let (res, flags) = add_with_carry(op1, op2, sub, w);
            if set_flags {
                cpu.set_nzcv4(flags);
                cpu.set_x_zr(rd, res);
            } else {
                cpu.set_x_sp(rd, res);
            }
        }
        Inst::AddSubCarry {
            sf,
            sub,
            set_flags,
            rd,
            rn,
            rm,
        } => {
            let w = width(sf);
            let m = mask(w);
            let op1 = cpu.x_zr(rn) & m;
            let mut op2 = cpu.x_zr(rm) & m;
            if sub {
                op2 = !op2 & m;
            }
            let (res, flags) = add_with_carry(op1, op2, cpu.carry(), w);
            if set_flags {
                cpu.set_nzcv4(flags);
            }
            cpu.set_x_zr(rd, res);
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
            let w = width(sf);
            let m = mask(w);
            if condition_holds(cond, cpu.nzcv4()) {
                let op1 = cpu.x_zr(rn) & m;
                let op2 = if imm_operand {
                    u64::from(rm_or_imm)
                } else {
                    cpu.x_zr(rm_or_imm) & m
                };
                let (_, flags) = if negative {
                    add_with_carry(op1, !op2 & m, true, w)
                } else {
                    add_with_carry(op1, op2, false, w)
                };
                cpu.set_nzcv4(flags);
            } else {
                cpu.set_nzcv4(nzcv);
            }
        }
        Inst::CondSel {
            sf,
            op,
            rd,
            rn,
            rm,
            cond,
        } => {
            let m = mask(width(sf));
            let res = if condition_holds(cond, cpu.nzcv4()) {
                cpu.x_zr(rn) & m
            } else {
                let v = cpu.x_zr(rm) & m;
                match op {
                    CondSelOp::Csel => v,
                    CondSelOp::Csinc => v.wrapping_add(1) & m,
                    CondSelOp::Csinv => !v & m,
                    CondSelOp::Csneg => (!v).wrapping_add(1) & m,
                }
            };
            cpu.set_x_zr(rd, res);
        }
        Inst::DataProc1 { sf, op, rd, rn } => {
            let w = width(sf);
            let m = mask(w);
            let v = cpu.x_zr(rn) & m;
            let res = match op {
                Dp1Op::Rbit => v.reverse_bits() >> (64 - w),
                Dp1Op::Rev16 => {
                    let mut out = 0u64;
                    for i in 0..w / 16 {
                        let half = ((v >> (16 * i)) & 0xFFFF) as u16;
                        out |= u64::from(half.swap_bytes()) << (16 * i);
                    }
                    out
                }
                Dp1Op::Rev32 => {
                    let lo = (v as u32).swap_bytes();
                    let hi = ((v >> 32) as u32).swap_bytes();
                    u64::from(lo) | (u64::from(hi) << 32)
                }
                Dp1Op::Rev => v.swap_bytes() >> (64 - w),
                Dp1Op::Clz => u64::from(v.leading_zeros() - (64 - w)),
                Dp1Op::Cls => u64::from(count_leading_sign_bits(v, w)),
            };
            cpu.set_x_zr(rd, res);
        }
        Inst::DataProc2 { sf, op, rd, rn, rm } => {
            let w = width(sf);
            let m = mask(w);
            let a = cpu.x_zr(rn) & m;
            let b = cpu.x_zr(rm) & m;
            let res = match op {
                Dp2Op::Udiv => a.checked_div(b).unwrap_or(0),
                Dp2Op::Sdiv => {
                    let (sa, sb) = (sign_extend(a, w) as i64, sign_extend(b, w) as i64);
                    // Division by zero gives zero; MIN / -1 wraps to MIN (RoundTowardsZero).
                    (if sb == 0 { 0 } else { sa.wrapping_div(sb) }) as u64 & m
                }
                Dp2Op::Lslv => shift_value(a, ShiftType::Lsl, (b % u64::from(w)) as u32, w),
                Dp2Op::Lsrv => shift_value(a, ShiftType::Lsr, (b % u64::from(w)) as u32, w),
                Dp2Op::Asrv => shift_value(a, ShiftType::Asr, (b % u64::from(w)) as u32, w),
                Dp2Op::Rorv => shift_value(a, ShiftType::Ror, (b % u64::from(w)) as u32, w),
                Dp2Op::Crc32 { size, c } => u64::from(crc32_update(a as u32, b, size, c)),
            };
            cpu.set_x_zr(rd, res);
        }
        Inst::DataProc3 {
            sf,
            op,
            rd,
            rn,
            rm,
            ra,
        } => {
            let w = width(sf);
            let m = mask(w);
            let (n, mm, acc) = (cpu.x_zr(rn), cpu.x_zr(rm), cpu.x_zr(ra));
            let res = match op {
                Dp3Op::Madd => acc.wrapping_add(n.wrapping_mul(mm)) & m,
                Dp3Op::Msub => acc.wrapping_sub(n.wrapping_mul(mm)) & m,
                Dp3Op::Smaddl => {
                    acc.wrapping_add(((n as i32 as i64).wrapping_mul(mm as i32 as i64)) as u64)
                }
                Dp3Op::Smsubl => {
                    acc.wrapping_sub(((n as i32 as i64).wrapping_mul(mm as i32 as i64)) as u64)
                }
                Dp3Op::Umaddl => acc.wrapping_add((n & 0xFFFF_FFFF).wrapping_mul(mm & 0xFFFF_FFFF)),
                Dp3Op::Umsubl => acc.wrapping_sub((n & 0xFFFF_FFFF).wrapping_mul(mm & 0xFFFF_FFFF)),
                Dp3Op::Smulh => (((n as i64 as i128) * (mm as i64 as i128)) >> 64) as u64,
                Dp3Op::Umulh => ((u128::from(n) * u128::from(mm)) >> 64) as u64,
            };
            cpu.set_x_zr(rd, res);
        }

        Inst::Unallocated(word) => return Err(Trap::Undefined(word)),
        Inst::Unsupported { word, what } => return Err(Trap::Unsupported { word, what }),
    }
    cpu.pc = next;
    Ok(())
}
