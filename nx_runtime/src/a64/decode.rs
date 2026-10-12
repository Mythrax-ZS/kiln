//! A64 decoder for the base (integer) instruction set, following the top-level encoding index of
//! the Arm A-profile architecture.
//!
//! Encodings the architecture leaves unallocated decode to [`Inst::Unallocated`]. Valid encodings
//! this project does not implement (FP and SIMD before milestone M5, optional extensions, privileged
//! instructions) decode to [`Inst::Unsupported`] so that nothing is silently ignored.

use super::bits::{decode_bit_masks, sign_extend};
use super::inst::*;

fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & (u32::MAX >> (31 - (hi - lo)))
}

fn bit(w: u32, n: u32) -> bool {
    (w >> n) & 1 == 1
}

fn simm(w: u32, hi: u32, lo: u32) -> i64 {
    sign_extend(u64::from(bits(w, hi, lo)), hi - lo + 1) as i64
}

fn unsupported(word: u32, what: &'static str) -> Inst {
    Inst::Unsupported { word, what }
}

pub fn decode(w: u32) -> Inst {
    match bits(w, 28, 25) {
        0b1000 | 0b1001 => decode_dp_imm(w),
        0b1010 | 0b1011 => decode_branch(w),
        0b0100 | 0b0110 | 0b1100 | 0b1110 => decode_ldst(w),
        0b0101 | 0b1101 => decode_dp_reg(w),
        0b0111 | 0b1111 => unsupported(w, "scalar floating point and Advanced SIMD (milestone M5)"),
        0b0010 => unsupported(w, "SVE"),
        _ => Inst::Unallocated(w),
    }
}

// ---------------------------------------------------------------------------------------------
// Data processing, immediate
// ---------------------------------------------------------------------------------------------

fn decode_dp_imm(w: u32) -> Inst {
    let sf = bit(w, 31);
    let rd = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    match bits(w, 25, 23) {
        0b000 | 0b001 => {
            let imm = (simm(w, 23, 5) << 2) | i64::from(bits(w, 30, 29));
            let page = bit(w, 31);
            Inst::Adr {
                rd,
                imm: if page { imm << 12 } else { imm },
                page,
            }
        }
        0b010 => {
            let shifted = bit(w, 22);
            let imm12 = bits(w, 21, 10);
            Inst::AddSubImm {
                sf,
                sub: bit(w, 30),
                set_flags: bit(w, 29),
                rd,
                rn,
                imm: if shifted { imm12 << 12 } else { imm12 },
            }
        }
        0b011 => unsupported(w, "add/subtract immediate with tags (MTE)"),
        0b100 => {
            let n = bit(w, 22);
            if !sf && n {
                return Inst::Unallocated(w);
            }
            let size = if sf { 64 } else { 32 };
            let Some((wmask, _)) =
                decode_bit_masks(n, bits(w, 15, 10) as u8, bits(w, 21, 16) as u8, true, size)
            else {
                return Inst::Unallocated(w);
            };
            let op = match bits(w, 30, 29) {
                0 => LogicOp::And,
                1 => LogicOp::Orr,
                2 => LogicOp::Eor,
                _ => LogicOp::Ands,
            };
            Inst::LogicalImm {
                sf,
                op,
                rd,
                rn,
                imm: wmask,
            }
        }
        0b101 => {
            let hw = bits(w, 22, 21);
            if (!sf && hw >= 2) || bits(w, 30, 29) == 1 {
                return Inst::Unallocated(w);
            }
            let op = match bits(w, 30, 29) {
                0 => MovOp::Movn,
                2 => MovOp::Movz,
                _ => MovOp::Movk,
            };
            Inst::MovWide {
                sf,
                op,
                rd,
                imm16: bits(w, 20, 5) as u16,
                shift: (hw * 16) as u8,
            }
        }
        0b110 => {
            let opc = bits(w, 30, 29);
            let n = bit(w, 22);
            let immr = bits(w, 21, 16) as u8;
            let imms = bits(w, 15, 10) as u8;
            if opc == 3 || n != sf || (!sf && (immr & 0x20 != 0 || imms & 0x20 != 0)) {
                return Inst::Unallocated(w);
            }
            let op = match opc {
                0 => BitfieldOp::Sbfm,
                1 => BitfieldOp::Bfm,
                _ => BitfieldOp::Ubfm,
            };
            Inst::Bitfield {
                sf,
                op,
                rd,
                rn,
                immr,
                imms,
            }
        }
        _ => {
            // extract
            let imms = bits(w, 15, 10) as u8;
            if bits(w, 30, 29) != 0 || bit(w, 21) || bit(w, 22) != sf || (!sf && imms & 0x20 != 0) {
                return Inst::Unallocated(w);
            }
            Inst::Extr {
                sf,
                rd,
                rn,
                rm: bits(w, 20, 16) as u8,
                lsb: imms,
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Branches, exception generation, system
// ---------------------------------------------------------------------------------------------

fn decode_branch(w: u32) -> Inst {
    if bits(w, 30, 26) == 0b00101 {
        return Inst::B {
            link: bit(w, 31),
            imm: simm(w, 25, 0) << 2,
        };
    }
    if bits(w, 30, 25) == 0b011010 {
        return Inst::Cbz {
            sf: bit(w, 31),
            nonzero: bit(w, 24),
            rt: bits(w, 4, 0) as u8,
            imm: simm(w, 23, 5) << 2,
        };
    }
    if bits(w, 30, 25) == 0b011011 {
        return Inst::Tbz {
            nonzero: bit(w, 24),
            rt: bits(w, 4, 0) as u8,
            bit: ((bits(w, 31, 31) << 5) | bits(w, 23, 19)) as u8,
            imm: simm(w, 18, 5) << 2,
        };
    }
    if bits(w, 31, 25) == 0b0101010 {
        if bit(w, 24) || bit(w, 4) {
            return unsupported(w, "BC.cond (FEAT_HBC)");
        }
        return Inst::BCond {
            cond: bits(w, 3, 0) as u8,
            imm: simm(w, 23, 5) << 2,
        };
    }
    if bits(w, 31, 24) == 0xD4 {
        return decode_exception(w);
    }
    if bits(w, 31, 22) == 0b1101010100 {
        return decode_system(w);
    }
    if bits(w, 31, 25) == 0b1101011 {
        return decode_branch_reg(w);
    }
    Inst::Unallocated(w)
}

fn decode_exception(w: u32) -> Inst {
    let opc = bits(w, 23, 21);
    let op2 = bits(w, 4, 2);
    let ll = bits(w, 1, 0);
    let imm = bits(w, 20, 5) as u16;
    match (opc, op2, ll) {
        (0b000, 0, 0b01) => Inst::Svc { imm },
        (0b001, 0, 0b00) => Inst::Brk { imm },
        (0b000, 0, 0b10 | 0b11) => unsupported(w, "HVC/SMC (not available to EL0)"),
        (0b010, 0, 0b00) => unsupported(w, "HLT"),
        (0b101, 0, 0b01..=0b11) => unsupported(w, "DCPS"),
        _ => Inst::Unallocated(w),
    }
}

fn decode_branch_reg(w: u32) -> Inst {
    let opc = bits(w, 24, 21);
    let op2 = bits(w, 20, 16);
    let op3 = bits(w, 15, 10);
    let rn = bits(w, 9, 5) as u8;
    let op4 = bits(w, 4, 0);
    if op2 != 0b11111 {
        return Inst::Unallocated(w);
    }
    if op3 != 0 || op4 != 0 {
        return if op3 >> 1 == 1 {
            unsupported(w, "pointer authentication (FEAT_PAuth)")
        } else {
            Inst::Unallocated(w)
        };
    }
    match opc {
        0b0000 => Inst::BranchReg {
            kind: BranchReg::Br,
            rn,
        },
        0b0001 => Inst::BranchReg {
            kind: BranchReg::Blr,
            rn,
        },
        0b0010 => Inst::BranchReg {
            kind: BranchReg::Ret,
            rn,
        },
        0b0100 | 0b0101 => unsupported(w, "ERET/DRPS (not available to EL0)"),
        _ => Inst::Unallocated(w),
    }
}

fn sysreg_of(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> SysReg {
    match (op0, op1, crn, crm, op2) {
        (3, 3, 4, 2, 0) => SysReg::Nzcv,
        (3, 3, 4, 4, 0) => SysReg::Fpcr,
        (3, 3, 4, 4, 1) => SysReg::Fpsr,
        (3, 3, 13, 0, 2) => SysReg::TpidrEl0,
        (3, 3, 13, 0, 3) => SysReg::TpidrroEl0,
        (3, 3, 14, 0, 0) => SysReg::CntfrqEl0,
        (3, 3, 14, 0, 1) => SysReg::CntpctEl0,
        (3, 3, 14, 0, 2) => SysReg::CntvctEl0,
        (3, 3, 0, 0, 1) => SysReg::CtrEl0,
        (3, 3, 0, 0, 7) => SysReg::DczidEl0,
        _ => SysReg::Other(op0 as u8, op1 as u8, crn as u8, crm as u8, op2 as u8),
    }
}

fn decode_system(w: u32) -> Inst {
    let l = bit(w, 21);
    let op0 = bits(w, 20, 19);
    let op1 = bits(w, 18, 16);
    let crn = bits(w, 15, 12);
    let crm = bits(w, 11, 8);
    let op2 = bits(w, 7, 5);
    let rt = bits(w, 4, 0) as u8;
    match (l, op0) {
        (false, 0) => match crn {
            0b0010 if rt == 31 && op1 == 0b011 => Inst::Hint {
                imm: ((crm << 3) | op2) as u8,
            },
            0b0011 if rt == 31 && op1 == 0b011 => match op2 {
                0b010 => Inst::Clrex,
                0b100 => Inst::Barrier {
                    kind: BarrierKind::Dsb,
                    crm: crm as u8,
                },
                0b101 => Inst::Barrier {
                    kind: BarrierKind::Dmb,
                    crm: crm as u8,
                },
                0b110 => Inst::Barrier {
                    kind: BarrierKind::Isb,
                    crm: crm as u8,
                },
                0b111 => unsupported(w, "SB (FEAT_SB)"),
                _ => Inst::Unallocated(w),
            },
            0b0100 => unsupported(w, "MSR (immediate) to PSTATE"),
            _ => Inst::Unallocated(w),
        },
        (_, 1) => Inst::Sys {
            load: l,
            op1: op1 as u8,
            crn: crn as u8,
            crm: crm as u8,
            op2: op2 as u8,
            rt,
        },
        (false, 2 | 3) => Inst::Msr {
            rt,
            reg: sysreg_of(op0, op1, crn, crm, op2),
        },
        (true, 2 | 3) => Inst::Mrs {
            rt,
            reg: sysreg_of(op0, op1, crn, crm, op2),
        },
        _ => Inst::Unallocated(w),
    }
}

// ---------------------------------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------------------------------

fn decode_ldst(w: u32) -> Inst {
    if bit(w, 26) {
        return unsupported(w, "SIMD&FP load/store (milestone M5)");
    }
    match bits(w, 29, 27) {
        0b001 => decode_exclusive(w),
        0b011 => {
            if bits(w, 25, 24) != 0 {
                return Inst::Unallocated(w);
            }
            let rt = bits(w, 4, 0) as u8;
            let imm = simm(w, 23, 5) << 2;
            let (kind, size) = match bits(w, 31, 30) {
                0 => (MemKind::Load, 2),
                1 => (MemKind::Load, 3),
                2 => (MemKind::LoadSigned64, 2),
                _ => (MemKind::Prefetch, 3),
            };
            Inst::LdSt {
                kind,
                size,
                rt,
                rn: 0,
                mode: AddrMode::Literal(imm),
            }
        }
        0b101 => decode_pair(w),
        0b111 => decode_single(w),
        _ => Inst::Unallocated(w),
    }
}

fn decode_exclusive(w: u32) -> Inst {
    if bits(w, 29, 24) != 0b001000 {
        return Inst::Unallocated(w);
    }
    let size = bits(w, 31, 30) as u8;
    let o2 = bit(w, 23);
    let load = bit(w, 22);
    let o1 = bit(w, 21);
    let o0 = bit(w, 15);
    let rs = bits(w, 20, 16) as u8;
    let rt2 = bits(w, 14, 10) as u8;
    let rn = bits(w, 9, 5) as u8;
    let rt = bits(w, 4, 0) as u8;
    match (o2, o1) {
        (false, false) => Inst::Exclusive {
            load,
            pair: false,
            size,
            acquire: load && o0,
            release: !load && o0,
            rs,
            rt,
            rt2,
            rn,
        },
        (false, true) => {
            if size < 2 {
                return Inst::Unallocated(w);
            }
            Inst::Exclusive {
                load,
                pair: true,
                size,
                acquire: load && o0,
                release: !load && o0,
                rs,
                rt,
                rt2,
                rn,
            }
        }
        (true, false) => {
            if !o0 {
                return unsupported(w, "LDLAR/STLLR (FEAT_LOR)");
            }
            if rs != 31 || rt2 != 31 {
                return Inst::Unallocated(w);
            }
            Inst::AcqRel { load, size, rt, rn }
        }
        (true, true) => unsupported(w, "compare and swap (FEAT_LSE)"),
    }
}

fn decode_pair(w: u32) -> Inst {
    let opc = bits(w, 31, 30);
    let kind = bits(w, 24, 23);
    let load = bit(w, 22);
    let imm7 = simm(w, 21, 15);
    let rt2 = bits(w, 14, 10) as u8;
    let rn = bits(w, 9, 5) as u8;
    let rt = bits(w, 4, 0) as u8;
    let (size, signed_word) = match opc {
        0 => (2u8, false),
        1 => {
            if !load || kind == 0 {
                return unsupported(w, "STGP (MTE) or no-allocate LDPSW");
            }
            (2, true)
        }
        2 => (3, false),
        _ => return Inst::Unallocated(w),
    };
    Inst::LdStPair {
        load,
        signed_word,
        size,
        rt,
        rt2,
        rn,
        imm: imm7 << size,
        writeback: kind == 1 || kind == 3,
        post_index: kind == 1,
    }
}

fn decode_single(w: u32) -> Inst {
    let size = bits(w, 31, 30) as u8;
    let opc = bits(w, 23, 22);
    let rn = bits(w, 9, 5) as u8;
    let rt = bits(w, 4, 0) as u8;
    let kind = match (size, opc) {
        (_, 0) => MemKind::Store,
        (_, 1) => MemKind::Load,
        (0 | 1, 2) => MemKind::LoadSigned64,
        (0 | 1, 3) => MemKind::LoadSigned32,
        (2, 2) => MemKind::LoadSigned64,
        (3, 2) => MemKind::Prefetch,
        _ => return Inst::Unallocated(w), // (2, 3) and (3, 3)
    };
    let mode = if bit(w, 24) {
        AddrMode::Offset(i64::from(bits(w, 21, 10)) << size)
    } else if !bit(w, 21) {
        let imm9 = simm(w, 20, 12);
        match bits(w, 11, 10) {
            0b00 => AddrMode::Offset(imm9),
            0b01 => AddrMode::PostIndex(imm9),
            0b10 => {
                // LDTR/STTR behave as plain accesses at EL0; there is no unprivileged prefetch.
                if kind == MemKind::Prefetch {
                    return Inst::Unallocated(w);
                }
                AddrMode::Offset(imm9)
            }
            _ => AddrMode::PreIndex(imm9),
        }
    } else {
        match bits(w, 11, 10) {
            0b10 => {
                let option = bits(w, 15, 13);
                if option & 0b010 == 0 {
                    return Inst::Unallocated(w);
                }
                let extend = match option {
                    0b010 => ExtendType::Uxtw,
                    0b011 => ExtendType::Uxtx,
                    0b110 => ExtendType::Sxtw,
                    _ => ExtendType::Sxtx,
                };
                AddrMode::Register {
                    rm: bits(w, 20, 16) as u8,
                    extend,
                    amount: if bit(w, 12) { size } else { 0 },
                }
            }
            0b00 => return unsupported(w, "atomic memory operations (FEAT_LSE)"),
            _ => return unsupported(w, "load/store with pointer authentication"),
        }
    };
    // Prefetch with a pre/post-indexed form is unallocated.
    if kind == MemKind::Prefetch && matches!(mode, AddrMode::PreIndex(_) | AddrMode::PostIndex(_)) {
        return Inst::Unallocated(w);
    }
    Inst::LdSt {
        kind,
        size,
        rt,
        rn,
        mode,
    }
}

// ---------------------------------------------------------------------------------------------
// Data processing, register
// ---------------------------------------------------------------------------------------------

fn shift_of(v: u32) -> ShiftType {
    match v & 3 {
        0 => ShiftType::Lsl,
        1 => ShiftType::Lsr,
        2 => ShiftType::Asr,
        _ => ShiftType::Ror,
    }
}

fn extend_of(v: u32) -> ExtendType {
    match v & 7 {
        0 => ExtendType::Uxtb,
        1 => ExtendType::Uxth,
        2 => ExtendType::Uxtw,
        3 => ExtendType::Uxtx,
        4 => ExtendType::Sxtb,
        5 => ExtendType::Sxth,
        6 => ExtendType::Sxtw,
        _ => ExtendType::Sxtx,
    }
}

fn decode_dp_reg(w: u32) -> Inst {
    let sf = bit(w, 31);
    let rd = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    let rm = bits(w, 20, 16) as u8;
    let op2 = bits(w, 24, 21);
    if !bit(w, 28) {
        // For the shifted-register forms bits 15:10 are the shift amount, and a 32-bit operation
        // cannot shift by 32 or more. The extended-register form reads them as option:imm3.
        let amount = bits(w, 15, 10) as u8;
        let shifted_form = op2 & 0b1000 == 0 || op2 & 1 == 0;
        if shifted_form && !sf && amount & 0x20 != 0 {
            return Inst::Unallocated(w);
        }
        if op2 & 0b1000 == 0 {
            let op = match bits(w, 30, 29) {
                0 => LogicOp::And,
                1 => LogicOp::Orr,
                2 => LogicOp::Eor,
                _ => LogicOp::Ands,
            };
            return Inst::LogicalShifted {
                sf,
                op,
                invert: bit(w, 21),
                rd,
                rn,
                rm,
                shift: shift_of(bits(w, 23, 22)),
                amount,
            };
        }
        if op2 & 1 == 0 {
            if bits(w, 23, 22) == 3 {
                return Inst::Unallocated(w);
            }
            return Inst::AddSubShifted {
                sf,
                sub: bit(w, 30),
                set_flags: bit(w, 29),
                rd,
                rn,
                rm,
                shift: shift_of(bits(w, 23, 22)),
                amount,
            };
        }
        let imm3 = bits(w, 12, 10);
        if bits(w, 23, 22) != 0 || imm3 > 4 {
            return Inst::Unallocated(w);
        }
        return Inst::AddSubExtended {
            sf,
            sub: bit(w, 30),
            set_flags: bit(w, 29),
            rd,
            rn,
            rm,
            extend: extend_of(bits(w, 15, 13)),
            amount: imm3 as u8,
        };
    }
    match op2 {
        0b0000 => {
            if bits(w, 15, 10) != 0 {
                return unsupported(w, "flag manipulation (FEAT_FlagM) or reserved");
            }
            Inst::AddSubCarry {
                sf,
                sub: bit(w, 30),
                set_flags: bit(w, 29),
                rd,
                rn,
                rm,
            }
        }
        0b0010 => {
            if !bit(w, 29) || bit(w, 10) || bit(w, 4) {
                return Inst::Unallocated(w);
            }
            Inst::CondCmp {
                sf,
                negative: bit(w, 30),
                imm_operand: bit(w, 11),
                rn,
                rm_or_imm: bits(w, 20, 16) as u8,
                cond: bits(w, 15, 12) as u8,
                nzcv: bits(w, 3, 0) as u8,
            }
        }
        0b0100 => {
            if bit(w, 29) {
                return Inst::Unallocated(w);
            }
            let op2 = bits(w, 11, 10);
            if op2 > 1 {
                return Inst::Unallocated(w);
            }
            let op = match (bit(w, 30), op2) {
                (false, 0) => CondSelOp::Csel,
                (false, _) => CondSelOp::Csinc,
                (true, 0) => CondSelOp::Csinv,
                (true, _) => CondSelOp::Csneg,
            };
            Inst::CondSel {
                sf,
                op,
                rd,
                rn,
                rm,
                cond: bits(w, 15, 12) as u8,
            }
        }
        0b0110 => {
            if bit(w, 29) {
                return unsupported(w, "SUBPS (MTE)");
            }
            if bit(w, 30) {
                decode_dp1(w, sf, rd, rn)
            } else {
                decode_dp2(w, sf, rd, rn, rm)
            }
        }
        0b1000..=0b1111 => decode_dp3(w, sf, rd, rn, rm),
        _ => Inst::Unallocated(w),
    }
}

fn decode_dp1(w: u32, sf: bool, rd: u8, rn: u8) -> Inst {
    if bits(w, 20, 16) != 0 {
        return unsupported(w, "pointer authentication (FEAT_PAuth)");
    }
    let op = match (bits(w, 15, 10), sf) {
        (0b000000, _) => Dp1Op::Rbit,
        (0b000001, _) => Dp1Op::Rev16,
        (0b000010, false) => Dp1Op::Rev,
        (0b000010, true) => Dp1Op::Rev32,
        (0b000011, true) => Dp1Op::Rev,
        (0b000100, _) => Dp1Op::Clz,
        (0b000101, _) => Dp1Op::Cls,
        _ => return Inst::Unallocated(w),
    };
    Inst::DataProc1 { sf, op, rd, rn }
}

fn decode_dp2(w: u32, sf: bool, rd: u8, rn: u8, rm: u8) -> Inst {
    let op = match bits(w, 15, 10) {
        0b000010 => Dp2Op::Udiv,
        0b000011 => Dp2Op::Sdiv,
        0b001000 => Dp2Op::Lslv,
        0b001001 => Dp2Op::Lsrv,
        0b001010 => Dp2Op::Asrv,
        0b001011 => Dp2Op::Rorv,
        v @ 0b010000..=0b010111 => {
            let size = (v & 3) as u8;
            if (size == 3) != sf {
                return Inst::Unallocated(w);
            }
            Dp2Op::Crc32 {
                size,
                c: v & 4 != 0,
            }
        }
        0b000000 | 0b000100 | 0b000101 | 0b001100 => {
            return unsupported(w, "memory tagging or pointer authentication")
        }
        _ => return Inst::Unallocated(w),
    };
    Inst::DataProc2 { sf, op, rd, rn, rm }
}

fn decode_dp3(w: u32, sf: bool, rd: u8, rn: u8, rm: u8) -> Inst {
    let ra = bits(w, 14, 10) as u8;
    let o0 = bit(w, 15);
    if bits(w, 30, 29) != 0 {
        return Inst::Unallocated(w);
    }
    let op = match (bits(w, 23, 21), o0) {
        (0b000, false) => Dp3Op::Madd,
        (0b000, true) => Dp3Op::Msub,
        (0b001, false) if sf => Dp3Op::Smaddl,
        (0b001, true) if sf => Dp3Op::Smsubl,
        (0b010, false) if sf && ra == 31 => Dp3Op::Smulh,
        (0b101, false) if sf => Dp3Op::Umaddl,
        (0b101, true) if sf => Dp3Op::Umsubl,
        (0b110, false) if sf && ra == 31 => Dp3Op::Umulh,
        _ => return Inst::Unallocated(w),
    };
    Inst::DataProc3 {
        sf,
        op,
        rd,
        rn,
        rm,
        ra,
    }
}
