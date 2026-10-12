//! Decoder tests. The positive table holds encodings assembled by LLVM (clang, AArch64 backend),
//! an assembler that shares no code or assumptions with this project.

use super::decode::decode;
use super::inst::*;
use crate::testing::asm;

fn asi(sf: bool, sub: bool, s: bool, rd: u8, rn: u8, imm: u32) -> Inst {
    Inst::AddSubImm {
        sf,
        sub,
        set_flags: s,
        rd,
        rn,
        imm,
    }
}

fn limm(sf: bool, op: LogicOp, rd: u8, rn: u8, imm: u64) -> Inst {
    Inst::LogicalImm {
        sf,
        op,
        rd,
        rn,
        imm,
    }
}

fn mw(sf: bool, op: MovOp, rd: u8, imm16: u16, shift: u8) -> Inst {
    Inst::MovWide {
        sf,
        op,
        rd,
        imm16,
        shift,
    }
}

fn bf(sf: bool, op: BitfieldOp, rd: u8, rn: u8, immr: u8, imms: u8) -> Inst {
    Inst::Bitfield {
        sf,
        op,
        rd,
        rn,
        immr,
        imms,
    }
}

fn ld(kind: MemKind, size: u8, rt: u8, rn: u8, mode: AddrMode) -> Inst {
    Inst::LdSt {
        kind,
        size,
        rt,
        rn,
        mode,
    }
}

#[allow(clippy::too_many_arguments)]
fn pair(
    load: bool,
    signed_word: bool,
    size: u8,
    rt: u8,
    rt2: u8,
    rn: u8,
    imm: i64,
    writeback: bool,
    post_index: bool,
) -> Inst {
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
    }
}

#[allow(clippy::too_many_arguments)]
fn ex(
    load: bool,
    pair: bool,
    size: u8,
    acquire: bool,
    release: bool,
    rs: u8,
    rt: u8,
    rt2: u8,
    rn: u8,
) -> Inst {
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
    }
}

#[allow(clippy::too_many_arguments)]
fn lsh(
    sf: bool,
    op: LogicOp,
    invert: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: ShiftType,
    amount: u8,
) -> Inst {
    Inst::LogicalShifted {
        sf,
        op,
        invert,
        rd,
        rn,
        rm,
        shift,
        amount,
    }
}

#[allow(clippy::too_many_arguments)]
fn ash(sf: bool, sub: bool, s: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8) -> Inst {
    Inst::AddSubShifted {
        sf,
        sub,
        set_flags: s,
        rd,
        rn,
        rm,
        shift,
        amount,
    }
}

#[allow(clippy::too_many_arguments)]
fn aex(
    sf: bool,
    sub: bool,
    s: bool,
    rd: u8,
    rn: u8,
    rm: u8,
    extend: ExtendType,
    amount: u8,
) -> Inst {
    Inst::AddSubExtended {
        sf,
        sub,
        set_flags: s,
        rd,
        rn,
        rm,
        extend,
        amount,
    }
}

fn carry(sf: bool, sub: bool, s: bool, rd: u8, rn: u8, rm: u8) -> Inst {
    Inst::AddSubCarry {
        sf,
        sub,
        set_flags: s,
        rd,
        rn,
        rm,
    }
}

fn ccmp(
    sf: bool,
    negative: bool,
    imm_operand: bool,
    rn: u8,
    rm_or_imm: u8,
    cond: u8,
    nzcv: u8,
) -> Inst {
    Inst::CondCmp {
        sf,
        negative,
        imm_operand,
        rn,
        rm_or_imm,
        cond,
        nzcv,
    }
}

fn csel(sf: bool, op: CondSelOp, rd: u8, rn: u8, rm: u8, cond: u8) -> Inst {
    Inst::CondSel {
        sf,
        op,
        rd,
        rn,
        rm,
        cond,
    }
}

fn dp1(sf: bool, op: Dp1Op, rd: u8, rn: u8) -> Inst {
    Inst::DataProc1 { sf, op, rd, rn }
}

fn dp2(sf: bool, op: Dp2Op, rd: u8, rn: u8, rm: u8) -> Inst {
    Inst::DataProc2 { sf, op, rd, rn, rm }
}

fn dp3(sf: bool, op: Dp3Op, rd: u8, rn: u8, rm: u8, ra: u8) -> Inst {
    Inst::DataProc3 {
        sf,
        op,
        rd,
        rn,
        rm,
        ra,
    }
}

fn sys(op1: u8, crn: u8, crm: u8, op2: u8, rt: u8) -> Inst {
    Inst::Sys {
        load: false,
        op1,
        crn,
        crm,
        op2,
        rt,
    }
}

/// Encodings assembled by LLVM from the instruction named in the comment, paired with what the
/// decoder must produce.
fn reference() -> Vec<(u32, Inst)> {
    use AddrMode::*;
    use BitfieldOp::*;
    use CondSelOp::*;
    use Dp1Op::*;
    use Dp2Op::*;
    use Dp3Op::*;
    use ExtendType::*;
    use LogicOp::*;
    use MemKind::*;
    use MovOp::*;
    use ShiftType as S;
    vec![
        (0x9100_0420, asi(true, false, false, 0, 1, 1)), // add x0, x1, #1
        (0x117F_FC20, asi(false, false, false, 0, 1, 0xFFF000)), // add w0, w1, #0xfff, lsl #12
        (0xB100_43E2, asi(true, false, true, 2, 31, 16)), // adds x2, sp, #16
        (0xD100_83FF, asi(true, true, false, 31, 31, 32)), // sub sp, sp, #32
        (0x7100_1483, asi(false, true, true, 3, 4, 5)),  // subs w3, w4, #5
        (0xF100_001F, asi(true, true, true, 31, 0, 0)),  // cmp x0, #0
        (0x3100_043F, asi(false, false, true, 31, 1, 1)), // cmn w1, #1
        (
            0x10FF_FF20,
            Inst::Adr {
                rd: 0,
                imm: -28,
                page: false,
            },
        ), // adr x0, start (at 0x1c)
        (
            0x9000_0001,
            Inst::Adr {
                rd: 1,
                imm: 0,
                page: true,
            },
        ), // adrp x1, start
        (0x9240_1C20, limm(true, And, 0, 1, 0xFF)),      // and x0, x1, #0xff
        (0x3201_F062, limm(false, Orr, 2, 3, 0xAAAA_AAAA)), // orr w2, w3, #0xaaaaaaaa
        (0xD200_F0A4, limm(true, Eor, 4, 5, 0x5555_5555_5555_5555)), // eor x4, x5, #0x5555..
        (0xF240_0CE6, limm(true, Ands, 6, 7, 0xF)),      // ands x6, x7, #0xf
        (0xF240_001F, limm(true, Ands, 31, 0, 1)),       // tst x0, #1
        (0xD280_0020, mw(true, Movz, 0, 1, 0)),          // mov x0, #1
        (0xD2A2_4681, mw(true, Movz, 1, 0x1234, 16)),    // movz x1, #0x1234, lsl #16
        (0x1280_0002, mw(false, Movn, 2, 0, 0)),         // movn w2, #0
        (0xF2FF_FFE3, mw(true, Movk, 3, 0xFFFF, 48)),    // movk x3, #0xffff, lsl #48
        (0x9280_0004, mw(true, Movn, 4, 0, 0)),          // mov x4, #-1
        (0x9344_1C20, bf(true, Sbfm, 0, 1, 4, 7)),       // sbfx x0, x1, #4, #4
        (0x5308_3C62, bf(false, Ubfm, 2, 3, 8, 15)),     // ubfx w2, w3, #8, #8
        (0xB378_3CA4, bf(true, Bfm, 4, 5, 56, 15)),      // bfi x4, x5, #8, #16
        (0x3304_1CE6, bf(false, Bfm, 6, 7, 4, 7)),       // bfxil w6, w7, #4, #4
        (0x9340_1C20, bf(true, Sbfm, 0, 1, 0, 7)),       // sxtb x0, w1
        (0x5300_3C62, bf(false, Ubfm, 2, 3, 0, 15)),     // uxth w2, w3
        (0xD37C_EC20, bf(true, Ubfm, 0, 1, 60, 59)),     // lsl x0, x1, #4
        (0x5303_7C62, bf(false, Ubfm, 2, 3, 3, 31)),     // lsr w2, w3, #3
        (0x937F_FCA4, bf(true, Sbfm, 4, 5, 63, 63)),     // asr x4, x5, #63
        (
            0x93C2_3020,
            Inst::Extr {
                sf: true,
                rd: 0,
                rn: 1,
                rm: 2,
                lsb: 12,
            },
        ), // extr x0, x1, x2, #12
        (
            0x1384_1C83,
            Inst::Extr {
                sf: false,
                rd: 3,
                rn: 4,
                rm: 4,
                lsb: 7,
            },
        ), // ror w3, w4, #7
        (
            0x17FF_FFE2,
            Inst::B {
                link: false,
                imm: -120,
            },
        ), // b start (at 0x78)
        (
            0x97FF_FFE1,
            Inst::B {
                link: true,
                imm: -124,
            },
        ), // bl start
        (0x54FF_FC00, Inst::BCond { cond: 0, imm: -128 }), // b.eq start
        (0x54FF_FBE8, Inst::BCond { cond: 8, imm: -132 }), // b.hi start
        (
            0xB4FF_FBC0,
            Inst::Cbz {
                sf: true,
                nonzero: false,
                rt: 0,
                imm: -136,
            },
        ), // cbz x0
        (
            0x35FF_FBA1,
            Inst::Cbz {
                sf: false,
                nonzero: true,
                rt: 1,
                imm: -140,
            },
        ), // cbnz w1
        (
            0x362F_FB82,
            Inst::Tbz {
                nonzero: false,
                rt: 2,
                bit: 5,
                imm: -144,
            },
        ), // tbz w2, #5
        (
            0x37FF_FB63,
            Inst::Tbz {
                nonzero: true,
                rt: 3,
                bit: 31,
                imm: -148,
            },
        ), // tbnz w3, #31
        (
            0xB6FF_FB44,
            Inst::Tbz {
                nonzero: false,
                rt: 4,
                bit: 63,
                imm: -152,
            },
        ), // tbz x4, #63
        (
            0xD61F_0200,
            Inst::BranchReg {
                kind: BranchReg::Br,
                rn: 16,
            },
        ), // br x16
        (
            0xD63F_0100,
            Inst::BranchReg {
                kind: BranchReg::Blr,
                rn: 8,
            },
        ), // blr x8
        (
            0xD65F_03C0,
            Inst::BranchReg {
                kind: BranchReg::Ret,
                rn: 30,
            },
        ), // ret
        (
            0xD65F_0020,
            Inst::BranchReg {
                kind: BranchReg::Ret,
                rn: 1,
            },
        ), // ret x1
        (0xD402_4681, Inst::Svc { imm: 0x1234 }),
        (0xD43E_0000, Inst::Brk { imm: 0xF000 }),
        (0xD503_201F, Inst::Hint { imm: 0 }), // nop
        (0xD503_203F, Inst::Hint { imm: 1 }), // yield
        (0xD503_205F, Inst::Hint { imm: 2 }), // wfe
        (0xD503_207F, Inst::Hint { imm: 3 }), // wfi
        (0xD503_209F, Inst::Hint { imm: 4 }), // sev
        (0xD503_20BF, Inst::Hint { imm: 5 }), // sevl
        (0xD503_2FFF, Inst::Hint { imm: 127 }),
        (
            0xD503_3F9F,
            Inst::Barrier {
                kind: BarrierKind::Dsb,
                crm: 15,
            },
        ), // dsb sy
        (
            0xD503_3BBF,
            Inst::Barrier {
                kind: BarrierKind::Dmb,
                crm: 11,
            },
        ), // dmb ish
        (
            0xD503_3FDF,
            Inst::Barrier {
                kind: BarrierKind::Isb,
                crm: 15,
            },
        ),
        (
            0xD503_39BF,
            Inst::Barrier {
                kind: BarrierKind::Dmb,
                crm: 9,
            },
        ), // dmb ishld
        (
            0xD503_3A9F,
            Inst::Barrier {
                kind: BarrierKind::Dsb,
                crm: 10,
            },
        ), // dsb ishst
        (0xD503_3F5F, Inst::Clrex),
        (0xD503_355F, Inst::Clrex), // clrex #5
        (
            0xD53B_4200,
            Inst::Mrs {
                rt: 0,
                reg: SysReg::Nzcv,
            },
        ),
        (
            0xD51B_4201,
            Inst::Msr {
                rt: 1,
                reg: SysReg::Nzcv,
            },
        ),
        (
            0xD53B_4402,
            Inst::Mrs {
                rt: 2,
                reg: SysReg::Fpcr,
            },
        ),
        (
            0xD51B_4423,
            Inst::Msr {
                rt: 3,
                reg: SysReg::Fpsr,
            },
        ),
        (
            0xD53B_D044,
            Inst::Mrs {
                rt: 4,
                reg: SysReg::TpidrEl0,
            },
        ),
        (
            0xD51B_D045,
            Inst::Msr {
                rt: 5,
                reg: SysReg::TpidrEl0,
            },
        ),
        (
            0xD53B_D066,
            Inst::Mrs {
                rt: 6,
                reg: SysReg::TpidrroEl0,
            },
        ),
        (
            0xD53B_E007,
            Inst::Mrs {
                rt: 7,
                reg: SysReg::CntfrqEl0,
            },
        ),
        (
            0xD53B_E028,
            Inst::Mrs {
                rt: 8,
                reg: SysReg::CntpctEl0,
            },
        ),
        (
            0xD53B_E049,
            Inst::Mrs {
                rt: 9,
                reg: SysReg::CntvctEl0,
            },
        ),
        (
            0xD53B_002A,
            Inst::Mrs {
                rt: 10,
                reg: SysReg::CtrEl0,
            },
        ),
        (
            0xD53B_00EB,
            Inst::Mrs {
                rt: 11,
                reg: SysReg::DczidEl0,
            },
        ),
        (0xD50B_7420, sys(3, 7, 4, 1, 0)),            // dc zva, x0
        (0xD50B_7E21, sys(3, 7, 14, 1, 1)),           // dc civac, x1
        (0xD50B_7A22, sys(3, 7, 10, 1, 2)),           // dc cvac, x2
        (0xD50B_7B23, sys(3, 7, 11, 1, 3)),           // dc cvau, x3
        (0xD50B_7524, sys(3, 7, 5, 1, 4)),            // ic ivau, x4
        (0xF940_0020, ld(Load, 3, 0, 1, Offset(0))),  // ldr x0, [x1]
        (0xB940_0462, ld(Load, 2, 2, 3, Offset(4))),  // ldr w2, [x3, #4]
        (0xF940_07E4, ld(Load, 3, 4, 31, Offset(8))), // ldr x4, [sp, #8]
        (0x3940_0420, ld(Load, 0, 0, 1, Offset(1))),  // ldrb w0, [x1, #1]
        (0x7940_0462, ld(Load, 1, 2, 3, Offset(2))),  // ldrh w2, [x3, #2]
        (0x3980_0020, ld(LoadSigned64, 0, 0, 1, Offset(0))), // ldrsb x0, [x1]
        (0x39C0_0462, ld(LoadSigned32, 0, 2, 3, Offset(1))), // ldrsb w2, [x3, #1]
        (0x7980_00A4, ld(LoadSigned64, 1, 4, 5, Offset(0))), // ldrsh x4, [x5]
        (0x79C0_04E6, ld(LoadSigned32, 1, 6, 7, Offset(2))), // ldrsh w6, [x7, #2]
        (0xB980_0420, ld(LoadSigned64, 2, 0, 1, Offset(4))), // ldrsw x0, [x1, #4]
        (0xF900_0420, ld(Store, 3, 0, 1, Offset(8))), // str x0, [x1, #8]
        (0x3900_0062, ld(Store, 0, 2, 3, Offset(0))), // strb w2, [x3]
        (0x7900_04A4, ld(Store, 1, 4, 5, Offset(2))), // strh w4, [x5, #2]
        (0xB900_13E6, ld(Store, 2, 6, 31, Offset(16))), // str w6, [sp, #16]
        (0xF840_8C20, ld(Load, 3, 0, 1, PreIndex(8))), // ldr x0, [x1, #8]!
        (0xB840_4462, ld(Load, 2, 2, 3, PostIndex(4))), // ldr w2, [x3], #4
        (0xF81F_0CA4, ld(Store, 3, 4, 5, PreIndex(-16))), // str x4, [x5, #-16]!
        (0x381F_F4E6, ld(Store, 0, 6, 7, PostIndex(-1))), // strb w6, [x7], #-1
        (0xF85F_8020, ld(Load, 3, 0, 1, Offset(-8))), // ldur x0, [x1, #-8]
        (0xB800_3062, ld(Store, 2, 2, 3, Offset(3))), // stur w2, [x3, #3]
        (0x389F_F0A4, ld(LoadSigned64, 0, 4, 5, Offset(-1))), // ldursb x4, [x5, #-1]
        (0xB880_10E6, ld(LoadSigned64, 2, 6, 7, Offset(1))), // ldursw x6, [x7, #1]
        (
            0xF862_6820,
            ld(
                Load,
                3,
                0,
                1,
                Register {
                    rm: 2,
                    extend: Uxtx,
                    amount: 0,
                },
            ),
        ), // ldr x0, [x1, x2]
        (
            0xF865_7883,
            ld(
                Load,
                3,
                3,
                4,
                Register {
                    rm: 5,
                    extend: Uxtx,
                    amount: 3,
                },
            ),
        ), // ldr x3, [x4, x5, lsl #3]
        (
            0xB868_48E6,
            ld(
                Load,
                2,
                6,
                7,
                Register {
                    rm: 8,
                    extend: Uxtw,
                    amount: 0,
                },
            ),
        ), // ldr w6, [x7, w8, uxtw]
        (
            0xB86B_D949,
            ld(
                Load,
                2,
                9,
                10,
                Register {
                    rm: 11,
                    extend: Sxtw,
                    amount: 2,
                },
            ),
        ), // ldr w9, [x10, w11, sxtw #2]
        (
            0x386E_E9AC,
            ld(
                Load,
                0,
                12,
                13,
                Register {
                    rm: 14,
                    extend: Sxtx,
                    amount: 0,
                },
            ),
        ), // ldrb w12, [x13, x14, sxtx]
        (
            0x7822_7820,
            ld(
                Store,
                1,
                0,
                1,
                Register {
                    rm: 2,
                    extend: Uxtx,
                    amount: 1,
                },
            ),
        ), // strh w0, [x1, x2, lsl #1]
        (
            0xF862_C820,
            ld(
                Load,
                3,
                0,
                1,
                Register {
                    rm: 2,
                    extend: Sxtw,
                    amount: 0,
                },
            ),
        ), // ldr x0, [x1, w2, sxtw]
        (0x58FF_F2E0, ld(Load, 3, 0, 0, Literal(-420))), // ldr x0, start (at 0x1a4)
        (0x18FF_F2C1, ld(Load, 2, 1, 0, Literal(-424))), // ldr w1, start
        (0x98FF_F2A2, ld(LoadSigned64, 2, 2, 0, Literal(-428))), // ldrsw x2, start
        (0xF980_0000, ld(Prefetch, 3, 0, 0, Offset(0))), // prfm pldl1keep, [x0]
        (0xD8FF_F273, ld(Prefetch, 3, 19, 0, Literal(-436))), // prfm pstl2strm, start
        (0xA940_0440, pair(true, false, 3, 0, 1, 2, 0, false, false)), // ldp x0, x1, [x2]
        (0x2941_10A3, pair(true, false, 2, 3, 4, 5, 8, false, false)), // ldp w3, w4, [x5, #8]
        (
            0xA9BF_7BFD,
            pair(false, false, 3, 29, 30, 31, -16, true, false),
        ), // stp x29, x30, [sp, #-16]!
        (
            0xA8C1_7BFD,
            pair(true, false, 3, 29, 30, 31, 16, true, true),
        ), // ldp x29, x30, [sp], #16
        (0x6941_1D06, pair(true, true, 2, 6, 7, 8, 8, false, false)), // ldpsw x6, x7, [x8, #8]
        (
            0xA801_0440,
            pair(false, false, 3, 0, 1, 2, 16, false, false),
        ), // stnp x0, x1, [x2, #16]
        (0x2840_10A3, pair(true, false, 2, 3, 4, 5, 0, false, false)), // ldnp w3, w4, [x5]
        (0xC85F_7C20, ex(true, false, 3, false, false, 31, 0, 31, 1)), // ldxr x0, [x1]
        (0x885F_7C62, ex(true, false, 2, false, false, 31, 2, 31, 3)), // ldxr w2, [x3]
        (0x085F_7CA4, ex(true, false, 0, false, false, 31, 4, 31, 5)), // ldxrb w4, [x5]
        (0x485F_7CE6, ex(true, false, 1, false, false, 31, 6, 31, 7)), // ldxrh w6, [x7]
        (0xC85F_FD28, ex(true, false, 3, true, false, 31, 8, 31, 9)), // ldaxr x8, [x9]
        (
            0xC80A_7D8B,
            ex(false, false, 3, false, false, 10, 11, 31, 12),
        ), // stxr w10, x11, [x12]
        (
            0x880D_FDEE,
            ex(false, false, 2, false, true, 13, 14, 31, 15),
        ), // stlxr w13, w14, [x15]
        (
            0x0810_7E51,
            ex(false, false, 0, false, false, 16, 17, 31, 18),
        ), // stxrb w16, w17, [x18]
        (0xC87F_0440, ex(true, true, 3, false, false, 31, 0, 1, 2)), // ldxp x0, x1, [x2]
        (0x887F_90A3, ex(true, true, 2, true, false, 31, 3, 4, 5)), // ldaxp w3, w4, [x5]
        (0xC826_2127, ex(false, true, 3, false, false, 6, 7, 8, 9)), // stxp w6, x7, x8, [x9]
        (0x882A_B1AB, ex(false, true, 2, false, true, 10, 11, 12, 13)), // stlxp w10, w11, w12, [x13]
        (
            0xC8DF_FC20,
            Inst::AcqRel {
                load: true,
                size: 3,
                rt: 0,
                rn: 1,
            },
        ), // ldar x0, [x1]
        (
            0x08DF_FC62,
            Inst::AcqRel {
                load: true,
                size: 0,
                rt: 2,
                rn: 3,
            },
        ), // ldarb w2, [x3]
        (
            0x48DF_FCA4,
            Inst::AcqRel {
                load: true,
                size: 1,
                rt: 4,
                rn: 5,
            },
        ), // ldarh w4, [x5]
        (
            0xC89F_FCE6,
            Inst::AcqRel {
                load: false,
                size: 3,
                rt: 6,
                rn: 7,
            },
        ), // stlr x6, [x7]
        (
            0x089F_FD28,
            Inst::AcqRel {
                load: false,
                size: 0,
                rt: 8,
                rn: 9,
            },
        ), // stlrb w8, [x9]
        (
            0x489F_FD6A,
            Inst::AcqRel {
                load: false,
                size: 1,
                rt: 10,
                rn: 11,
            },
        ), // stlrh w10, [x11]
        (0x8A02_0020, lsh(true, And, false, 0, 1, 2, S::Lsl, 0)),       // and x0, x1, x2
        (0x0A05_0C83, lsh(false, And, false, 3, 4, 5, S::Lsl, 3)),      // and w3, w4, w5, lsl #3
        (0x8A68_10E6, lsh(true, And, true, 6, 7, 8, S::Lsr, 4)),        // bic x6, x7, x8, lsr #4
        (0xAA8B_1549, lsh(true, Orr, false, 9, 10, 11, S::Asr, 5)),     // orr x9, x10, x11, asr #5
        (0x2AEE_19AC, lsh(false, Orr, true, 12, 13, 14, S::Ror, 6)),    // orn w12, w13, w14, ror #6
        (0xCA11_020F, lsh(true, Eor, false, 15, 16, 17, S::Lsl, 0)),    // eor x15, x16, x17
        (0x4A34_0272, lsh(false, Eor, true, 18, 19, 20, S::Lsl, 0)),    // eon w18, w19, w20
        (0xEA17_02D5, lsh(true, Ands, false, 21, 22, 23, S::Lsl, 0)),   // ands x21, x22, x23
        (0x6A3A_0338, lsh(false, Ands, true, 24, 25, 26, S::Lsl, 0)),   // bics w24, w25, w26
        (0xEA01_001F, lsh(true, Ands, false, 31, 0, 1, S::Lsl, 0)),     // tst x0, x1
        (0xAA23_03E2, lsh(true, Orr, true, 2, 31, 3, S::Lsl, 0)),       // mvn x2, x3
        (0xAA05_03E4, lsh(true, Orr, false, 4, 31, 5, S::Lsl, 0)),      // mov x4, x5
        (0x8B02_0020, ash(true, false, false, 0, 1, 2, S::Lsl, 0)),     // add x0, x1, x2
        (0x0B05_0883, ash(false, false, false, 3, 4, 5, S::Lsl, 2)),    // add w3, w4, w5, lsl #2
        (0xAB48_0CE6, ash(true, false, true, 6, 7, 8, S::Lsr, 3)),      // adds x6, x7, x8, lsr #3
        (0xCB8B_1149, ash(true, true, false, 9, 10, 11, S::Asr, 4)),    // sub x9, x10, x11, asr #4
        (0x6B0E_01AC, ash(false, true, true, 12, 13, 14, S::Lsl, 0)),   // subs w12, w13, w14
        (0xEB10_01FF, ash(true, true, true, 31, 15, 16, S::Lsl, 0)),    // cmp x15, x16
        (0x2B12_023F, ash(false, false, true, 31, 17, 18, S::Lsl, 0)),  // cmn w17, w18
        (0xCB14_03F3, ash(true, true, false, 19, 31, 20, S::Lsl, 0)),   // neg x19, x20
        (0x6B16_03F5, ash(false, true, true, 21, 31, 22, S::Lsl, 0)),   // negs w21, w22
        (0x8B21_6BE0, aex(true, false, false, 0, 31, 1, Uxtx, 2)),      // add x0, sp, x1, lsl #2
        (0x8B22_0020, aex(true, false, false, 0, 1, 2, Uxtb, 0)),       // add x0, x1, w2, uxtb
        (0x8B25_C883, aex(true, false, false, 3, 4, 5, Sxtw, 2)),       // add x3, x4, w5, sxtw #2
        (0xAB28_24E6, aex(true, false, true, 6, 7, 8, Uxth, 1)),        // adds x6, x7, w8, uxth #1
        (0xCB29_63FF, aex(true, true, false, 31, 31, 9, Uxtx, 0)),      // sub sp, sp, x9
        (0x6B2C_816A, aex(false, true, true, 10, 11, 12, Sxtb, 0)),     // subs w10, w11, w12, sxtb
        (0xEB2E_41BF, aex(true, true, true, 31, 13, 14, Uxtw, 0)),      // cmp x13, w14, uxtw
        (0x9A02_0020, carry(true, false, false, 0, 1, 2)),              // adc x0, x1, x2
        (0x3A05_0083, carry(false, false, true, 3, 4, 5)),              // adcs w3, w4, w5
        (0xDA08_00E6, carry(true, true, false, 6, 7, 8)),               // sbc x6, x7, x8
        (0x7A0B_0149, carry(false, true, true, 9, 10, 11)),             // sbcs w9, w10, w11
        (0xDA0D_03EC, carry(true, true, false, 12, 31, 13)),            // ngc x12, x13
        (0x7A0F_03EE, carry(false, true, true, 14, 31, 15)),            // ngcs w14, w15
        (0xFA41_0004, ccmp(true, true, false, 0, 1, 0, 4)),             // ccmp x0, x1, #4, eq
        (0x3A43_1040, ccmp(false, false, false, 2, 3, 1, 0)),           // ccmn w2, w3, #0, ne
        (0xFA45_888F, ccmp(true, true, true, 4, 5, 8, 15)),             // ccmp x4, #5, #15, hi
        (0x3A5F_B8C8, ccmp(false, false, true, 6, 31, 11, 8)),          // ccmn w6, #31, #8, lt
        (0x9A82_0020, csel(true, Csel, 0, 1, 2, 0)),                    // csel x0, x1, x2, eq
        (0x1A85_1483, csel(false, Csinc, 3, 4, 5, 1)),                  // csinc w3, w4, w5, ne
        (0xDA88_80E6, csel(true, Csinv, 6, 7, 8, 8)),                   // csinv x6, x7, x8, hi
        (0x5A8B_B549, csel(false, Csneg, 9, 10, 11, 11)),               // csneg w9, w10, w11, lt
        (0x9A9F_17EC, csel(true, Csinc, 12, 31, 31, 1)),                // cset x12, eq
        (0x5A9F_03ED, csel(false, Csinv, 13, 31, 31, 0)),               // csetm w13, ne
        (0x9A8F_35EE, csel(true, Csinc, 14, 15, 15, 3)),                // cinc x14, x15, hs
        (0x5A91_2230, csel(false, Csinv, 16, 17, 17, 2)),               // cinv w16, w17, lo
        (0xDA93_5672, csel(true, Csneg, 18, 19, 19, 5)),                // cneg x18, x19, mi
        (0xDAC0_0020, dp1(true, Rbit, 0, 1)),
        (0x5AC0_0462, dp1(false, Rev16, 2, 3)),
        (0x5AC0_08A4, dp1(false, Rev, 4, 5)),
        (0xDAC0_08E6, dp1(true, Rev32, 6, 7)),
        (0xDAC0_0D28, dp1(true, Rev, 8, 9)),
        (0xDAC0_116A, dp1(true, Clz, 10, 11)),
        (0x5AC0_15AC, dp1(false, Cls, 12, 13)),
        (0x9AC2_0820, dp2(true, Udiv, 0, 1, 2)),
        (0x1AC5_0C83, dp2(false, Sdiv, 3, 4, 5)),
        (0x9AC8_20E6, dp2(true, Lslv, 6, 7, 8)),
        (0x1ACB_2549, dp2(false, Lsrv, 9, 10, 11)),
        (0x9ACE_29AC, dp2(true, Asrv, 12, 13, 14)),
        (0x1AD1_2E0F, dp2(false, Rorv, 15, 16, 17)),
        (
            0x1AC2_4020,
            dp2(false, Crc32 { size: 0, c: false }, 0, 1, 2),
        ),
        (
            0x1AC5_4483,
            dp2(false, Crc32 { size: 1, c: false }, 3, 4, 5),
        ),
        (
            0x1AC8_48E6,
            dp2(false, Crc32 { size: 2, c: false }, 6, 7, 8),
        ),
        (
            0x9ACB_4D49,
            dp2(true, Crc32 { size: 3, c: false }, 9, 10, 11),
        ),
        (
            0x1ACE_51AC,
            dp2(false, Crc32 { size: 0, c: true }, 12, 13, 14),
        ),
        (
            0x1AD1_560F,
            dp2(false, Crc32 { size: 1, c: true }, 15, 16, 17),
        ),
        (
            0x1AD4_5A72,
            dp2(false, Crc32 { size: 2, c: true }, 18, 19, 20),
        ),
        (
            0x9AD7_5ED5,
            dp2(true, Crc32 { size: 3, c: true }, 21, 22, 23),
        ),
        (0x9B02_0C20, dp3(true, Madd, 0, 1, 2, 3)),
        (0x1B06_9CA4, dp3(false, Msub, 4, 5, 6, 7)),
        (0x9B0A_7D28, dp3(true, Madd, 8, 9, 10, 31)), // mul
        (0x1B0D_FD8B, dp3(false, Msub, 11, 12, 13, 31)), // mneg
        (0x9B22_0C20, dp3(true, Smaddl, 0, 1, 2, 3)),
        (0x9B26_9CA4, dp3(true, Smsubl, 4, 5, 6, 7)),
        (0x9B2A_7D28, dp3(true, Smaddl, 8, 9, 10, 31)), // smull
        (0x9B2D_FD8B, dp3(true, Smsubl, 11, 12, 13, 31)), // smnegl
        (0x9BA2_0C20, dp3(true, Umaddl, 0, 1, 2, 3)),
        (0x9BA6_9CA4, dp3(true, Umsubl, 4, 5, 6, 7)),
        (0x9BAA_7D28, dp3(true, Umaddl, 8, 9, 10, 31)), // umull
        (0x9BAD_FD8B, dp3(true, Umsubl, 11, 12, 13, 31)), // umnegl
        (0x9B42_7C20, dp3(true, Smulh, 0, 1, 2, 31)),
        (0x9BC5_7C83, dp3(true, Umulh, 3, 4, 5, 31)),
    ]
}

#[test]
fn decodes_every_llvm_reference_instruction() {
    let table = reference();
    assert_eq!(
        table.len(),
        217,
        "the reference listing has 217 instructions"
    );
    for (word, expected) in table {
        assert_eq!(decode(word), expected, "{word:#010x}");
    }
}

#[test]
fn reserved_and_unallocated_encodings() {
    let undefined = [
        0x0000_0000u32,                                            // udf #0
        0x0000_FFFF,                                               // udf #0xffff
        asm::logical_imm(false, 0, true, 0, 0, 0, 0), // 32-bit logical immediate with N=1
        asm::logical_imm(true, 0, true, 0, 63, 0, 0), // all-ones S is reserved
        asm::logical_imm(false, 0, false, 0, 62, 0, 0), // no element size
        asm::mov_wide(false, 2, 2, 1, 0),             // 32-bit movz with hw=2
        asm::mov_wide(true, 1, 0, 1, 0),              // opc=01 is unallocated
        asm::bitfield(false, 3, false, 0, 0, 0, 0),   // opc=11
        asm::bitfield(true, 0, false, 0, 0, 0, 0),    // N != sf
        asm::bitfield(false, 0, false, 32, 0, 0, 0),  // 32-bit with immr<5>
        asm::extr(true, false, 0, 0, 0, 0),           // N != sf
        asm::extr(false, false, 0, 32, 0, 0),         // 32-bit with imms<5>
        asm::add_sub_shifted(true, false, false, 3, 0, 0, 0, 0), // reserved shift
        asm::add_sub_shifted(false, false, false, 0, 0, 32, 0, 0), // 32-bit with imm6<5>
        asm::logical_shifted(false, 0, 0, false, 0, 32, 0, 0), // 32-bit with imm6<5>
        asm::add_sub_ext(true, false, false, 0, 0, 5, 0, 0), // extend shift above 4
        asm::cond_cmp(true, true, false, 0, 0, 0, 0) & !(1 << 29), // CCMP with S=0
        asm::cond_cmp(true, true, false, 0, 0, 0, 0) | (1 << 10), // o2 set
        asm::cond_sel(true, 0, 0, 0, 0, 0) | (1 << 29), // S=1
        asm::cond_sel(true, 0, 0, 0, 0, 0) | (2 << 10), // op2 = 1x
        asm::dp1(true, 0b000110, 0, 0),               // unallocated DP1 opcode
        asm::dp2(true, 0b000001, 0, 0, 0),            // unallocated DP2 opcode
        asm::dp2(false, 0b010011, 0, 0, 0),           // CRC32X needs sf=1
        asm::dp2(true, 0b010000, 0, 0, 0),            // CRC32B needs sf=0
        asm::dp3(false, 1, false, 0, 0, 0, 0),        // SMADDL needs sf=1
        asm::dp3(true, 2, true, 0, 31, 0, 0),         // SMULH with o0=1
        asm::dp3(true, 2, false, 0, 5, 0, 0),         // SMULH with Ra != 31
        asm::ldst_unsigned(2, 3, 0, 0, 0),            // size=10 opc=11
        asm::ldst_unsigned(3, 3, 0, 0, 0),            // size=11 opc=11
        asm::ldst_reg(3, 1, 0, 0, false, 0, 0),       // register offset with option<1>=0
        asm::ldst_pair(3, 2, true, 0, 0, 0, 0),       // opc=11
        asm::ldst_imm9(3, 2, 0, 1, 0, 0),             // PRFM with post-index
        asm::branch_reg(3, 0),                        // opc=0011
        asm::branch_reg(0, 0) | 1,                    // op4 != 0 without a PAC pattern
        asm::sysreg(true, 0, 0, 2, 0, 0, 0),          // L=1 with op0=0
        asm::hint(0) & !(1 << 16),                    // hint space with op1 != 011
    ];
    for w in undefined {
        assert!(
            matches!(decode(w), Inst::Unallocated(x) if x == w),
            "{w:#010x} decoded as {:?}",
            decode(w)
        );
    }
}

#[test]
fn valid_but_unimplemented_encodings_are_unsupported_not_ignored() {
    // Valid in the architecture, outside this project's scope: must stop the run, never vanish.
    for w in [
        0x1E22_2820u32, // fadd s0, s1, s2
        0x4C40_7000,    // ld1 {v0.16b}, [x0]
        0xBD40_0000,    // ldr s0, [x0]
        0xC8A0_7C41,    // cas x0, x1, [x2]
        0xD4A0_0001,    // dcps1
        0xD400_0002,    // hvc #0
        0xD440_0000,    // hlt #0
        0xD69F_03E0,    // eret
    ] {
        assert!(
            matches!(decode(w), Inst::Unsupported { word, .. } if word == w),
            "{w:#010x} -> {:?}",
            decode(w)
        );
    }
    // PAC hints are HINT instructions and execute as no-ops on a core without PAuth.
    assert_eq!(decode(0xD503_233F), Inst::Hint { imm: 25 }); // paciasp
}

#[test]
fn decoding_never_panics() {
    let mut rng = crate::testing::Rng::new(0xDEC0DE);
    for _ in 0..200_000 {
        let _ = decode(rng.next_u32());
    }
    for top in 0..16u32 {
        for payload in [0u32, 0x0FFF_FFFF, 0x0155_5555, 0x00AA_AAAA] {
            let _ = decode((top << 28) | payload);
        }
    }
}
