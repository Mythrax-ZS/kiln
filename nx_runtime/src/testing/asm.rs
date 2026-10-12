//! Raw-field A64 encoders for tests. Each function packs the fields of one encoding group exactly
//! as the Arm encoding tables lay them out, so tests can build valid instructions, boundary cases
//! and deliberately invalid patterns.

fn r(n: u8) -> u32 {
    u32::from(n & 31)
}

fn b(v: bool) -> u32 {
    u32::from(v)
}

// ---- data processing, immediate ----

/// ADR (`page == false`) and ADRP (`page == true`); `imm21` is the raw 21-bit immediate
/// (bytes for ADR, pages for ADRP).
pub fn pc_rel(page: bool, imm21: i32, rd: u8) -> u32 {
    let imm = imm21 as u32 & 0x1F_FFFF;
    (b(page) << 31) | ((imm & 3) << 29) | (0b10000 << 24) | ((imm >> 2) << 5) | r(rd)
}

pub fn add_sub_imm(sf: bool, sub: bool, s: bool, shift12: bool, imm12: u32, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | (b(sub) << 30)
        | (b(s) << 29)
        | (0b100010 << 23)
        | (b(shift12) << 22)
        | ((imm12 & 0xFFF) << 10)
        | (r(rn) << 5)
        | r(rd)
}

pub fn logical_imm(sf: bool, opc: u32, n: bool, immr: u8, imms: u8, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | ((opc & 3) << 29)
        | (0b100100 << 23)
        | (b(n) << 22)
        | (u32::from(immr & 63) << 16)
        | (u32::from(imms & 63) << 10)
        | (r(rn) << 5)
        | r(rd)
}

pub fn mov_wide(sf: bool, opc: u32, hw: u32, imm16: u32, rd: u8) -> u32 {
    (b(sf) << 31)
        | ((opc & 3) << 29)
        | (0b100101 << 23)
        | ((hw & 3) << 21)
        | ((imm16 & 0xFFFF) << 5)
        | r(rd)
}

pub fn bitfield(sf: bool, opc: u32, n: bool, immr: u8, imms: u8, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | ((opc & 3) << 29)
        | (0b100110 << 23)
        | (b(n) << 22)
        | (u32::from(immr & 63) << 16)
        | (u32::from(imms & 63) << 10)
        | (r(rn) << 5)
        | r(rd)
}

pub fn extr(sf: bool, n: bool, rm: u8, imms: u8, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | (0b100111 << 23)
        | (b(n) << 22)
        | (r(rm) << 16)
        | (u32::from(imms & 63) << 10)
        | (r(rn) << 5)
        | r(rd)
}

// ---- branches, exceptions, system ----

pub fn b_imm(link: bool, words: i32) -> u32 {
    (b(link) << 31) | (0b00101 << 26) | (words as u32 & 0x03FF_FFFF)
}

pub fn b_cond(cond: u8, words: i32) -> u32 {
    (0b0101010 << 25) | ((words as u32 & 0x7_FFFF) << 5) | u32::from(cond & 15)
}

pub fn cbz(sf: bool, nonzero: bool, words: i32, rt: u8) -> u32 {
    (b(sf) << 31) | (0b011010 << 25) | (b(nonzero) << 24) | ((words as u32 & 0x7_FFFF) << 5) | r(rt)
}

pub fn tbz(nonzero: bool, bit: u8, words: i32, rt: u8) -> u32 {
    (u32::from(bit >> 5 & 1) << 31)
        | (0b011011 << 25)
        | (b(nonzero) << 24)
        | (u32::from(bit & 31) << 19)
        | ((words as u32 & 0x3FFF) << 5)
        | r(rt)
}

/// BR (`opc` 0), BLR (1), RET (2).
pub fn branch_reg(opc: u32, rn: u8) -> u32 {
    (0b1101011 << 25) | ((opc & 15) << 21) | (0b11111 << 16) | (r(rn) << 5)
}

pub fn svc(imm: u16) -> u32 {
    0xD400_0001 | (u32::from(imm) << 5)
}

pub fn brk(imm: u16) -> u32 {
    0xD420_0000 | (u32::from(imm) << 5)
}

pub fn hint(imm: u8) -> u32 {
    0xD503_201F | (u32::from(imm & 127) << 5)
}

pub fn nop() -> u32 {
    hint(0)
}

pub fn barrier(op2: u32, crm: u8) -> u32 {
    0xD503_301F | (u32::from(crm & 15) << 8) | ((op2 & 7) << 5)
}

pub fn dsb(crm: u8) -> u32 {
    barrier(0b100, crm)
}

pub fn dmb(crm: u8) -> u32 {
    barrier(0b101, crm)
}

pub fn isb(crm: u8) -> u32 {
    barrier(0b110, crm)
}

pub fn clrex(crm: u8) -> u32 {
    barrier(0b010, crm)
}

/// MRS (`read`) or MSR (register) for the encoding `(op0, op1, crn, crm, op2)`.
pub fn sysreg(read: bool, op0: u8, op1: u8, crn: u8, crm: u8, op2: u8, rt: u8) -> u32 {
    0xD500_0000
        | (b(read) << 21)
        | (u32::from(op0 & 3) << 19)
        | (u32::from(op1 & 7) << 16)
        | (u32::from(crn & 15) << 12)
        | (u32::from(crm & 15) << 8)
        | (u32::from(op2 & 7) << 5)
        | r(rt)
}

/// SYS (`load == false`) or SYSL.
pub fn sys(load: bool, op1: u8, crn: u8, crm: u8, op2: u8, rt: u8) -> u32 {
    sysreg(load, 1, op1, crn, crm, op2, rt)
}

// ---- loads and stores (integer registers) ----

pub fn ldst_unsigned(size: u32, opc: u32, imm12: u32, rn: u8, rt: u8) -> u32 {
    ((size & 3) << 30)
        | (0b111 << 27)
        | (0b01 << 24)
        | ((opc & 3) << 22)
        | ((imm12 & 0xFFF) << 10)
        | (r(rn) << 5)
        | r(rt)
}

/// `mode`: 0 unscaled, 1 post-index, 2 unprivileged, 3 pre-index.
pub fn ldst_imm9(size: u32, opc: u32, imm9: i32, mode: u32, rn: u8, rt: u8) -> u32 {
    ((size & 3) << 30)
        | (0b111 << 27)
        | ((opc & 3) << 22)
        | ((imm9 as u32 & 0x1FF) << 12)
        | ((mode & 3) << 10)
        | (r(rn) << 5)
        | r(rt)
}

pub fn ldst_reg(size: u32, opc: u32, rm: u8, option: u32, s: bool, rn: u8, rt: u8) -> u32 {
    ((size & 3) << 30)
        | (0b111 << 27)
        | ((opc & 3) << 22)
        | (1 << 21)
        | (r(rm) << 16)
        | ((option & 7) << 13)
        | (b(s) << 12)
        | (0b10 << 10)
        | (r(rn) << 5)
        | r(rt)
}

pub fn ldr_literal(opc: u32, words: i32, rt: u8) -> u32 {
    ((opc & 3) << 30) | (0b011 << 27) | ((words as u32 & 0x7_FFFF) << 5) | r(rt)
}

/// `kind`: 0 no-allocate, 1 post-index, 2 offset, 3 pre-index.
pub fn ldst_pair(opc: u32, kind: u32, load: bool, imm7: i32, rt2: u8, rn: u8, rt: u8) -> u32 {
    ((opc & 3) << 30)
        | (0b101 << 27)
        | ((kind & 3) << 23)
        | (b(load) << 22)
        | ((imm7 as u32 & 0x7F) << 15)
        | (r(rt2) << 10)
        | (r(rn) << 5)
        | r(rt)
}

#[allow(clippy::too_many_arguments)]
pub fn exclusive(
    size: u32,
    o2: bool,
    load: bool,
    o1: bool,
    rs: u8,
    o0: bool,
    rt2: u8,
    rn: u8,
    rt: u8,
) -> u32 {
    ((size & 3) << 30)
        | (0b001000 << 24)
        | (b(o2) << 23)
        | (b(load) << 22)
        | (b(o1) << 21)
        | (r(rs) << 16)
        | (b(o0) << 15)
        | (r(rt2) << 10)
        | (r(rn) << 5)
        | r(rt)
}

// ---- data processing, register ----

#[allow(clippy::too_many_arguments)]
pub fn logical_shifted(
    sf: bool,
    opc: u32,
    shift: u32,
    n: bool,
    rm: u8,
    imm6: u8,
    rn: u8,
    rd: u8,
) -> u32 {
    (b(sf) << 31)
        | ((opc & 3) << 29)
        | (0b01010 << 24)
        | ((shift & 3) << 22)
        | (b(n) << 21)
        | (r(rm) << 16)
        | (u32::from(imm6 & 63) << 10)
        | (r(rn) << 5)
        | r(rd)
}

#[allow(clippy::too_many_arguments)]
pub fn add_sub_shifted(
    sf: bool,
    sub: bool,
    s: bool,
    shift: u32,
    rm: u8,
    imm6: u8,
    rn: u8,
    rd: u8,
) -> u32 {
    (b(sf) << 31)
        | (b(sub) << 30)
        | (b(s) << 29)
        | (0b01011 << 24)
        | ((shift & 3) << 22)
        | (r(rm) << 16)
        | (u32::from(imm6 & 63) << 10)
        | (r(rn) << 5)
        | r(rd)
}

#[allow(clippy::too_many_arguments)]
pub fn add_sub_ext(
    sf: bool,
    sub: bool,
    s: bool,
    rm: u8,
    option: u32,
    imm3: u32,
    rn: u8,
    rd: u8,
) -> u32 {
    (b(sf) << 31)
        | (b(sub) << 30)
        | (b(s) << 29)
        | (0b01011001 << 21)
        | (r(rm) << 16)
        | ((option & 7) << 13)
        | ((imm3 & 7) << 10)
        | (r(rn) << 5)
        | r(rd)
}

pub fn add_sub_carry(sf: bool, sub: bool, s: bool, rm: u8, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | (b(sub) << 30)
        | (b(s) << 29)
        | (0b11010000 << 21)
        | (r(rm) << 16)
        | (r(rn) << 5)
        | r(rd)
}

#[allow(clippy::too_many_arguments)]
pub fn cond_cmp(
    sf: bool,
    negative: bool,
    imm: bool,
    rm_or_imm: u8,
    cond: u8,
    rn: u8,
    nzcv: u8,
) -> u32 {
    (b(sf) << 31)
        | (b(negative) << 30)
        | (1 << 29)
        | (0b11010010 << 21)
        | (r(rm_or_imm) << 16)
        | (u32::from(cond & 15) << 12)
        | (b(imm) << 11)
        | (r(rn) << 5)
        | u32::from(nzcv & 15)
}

/// `op` 0 CSEL, 1 CSINC, 2 CSINV, 3 CSNEG.
pub fn cond_sel(sf: bool, op: u32, rm: u8, cond: u8, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | (((op >> 1) & 1) << 30)
        | (0b11010100 << 21)
        | (r(rm) << 16)
        | (u32::from(cond & 15) << 12)
        | ((op & 1) << 10)
        | (r(rn) << 5)
        | r(rd)
}

pub fn dp1(sf: bool, opcode: u32, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31) | (1 << 30) | (0b11010110 << 21) | ((opcode & 63) << 10) | (r(rn) << 5) | r(rd)
}

pub fn dp2(sf: bool, opcode: u32, rm: u8, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | (0b11010110 << 21)
        | (r(rm) << 16)
        | ((opcode & 63) << 10)
        | (r(rn) << 5)
        | r(rd)
}

#[allow(clippy::too_many_arguments)]
pub fn dp3(sf: bool, op31: u32, o0: bool, rm: u8, ra: u8, rn: u8, rd: u8) -> u32 {
    (b(sf) << 31)
        | (0b11011 << 24)
        | ((op31 & 7) << 21)
        | (r(rm) << 16)
        | (b(o0) << 15)
        | (r(ra) << 10)
        | (r(rn) << 5)
        | r(rd)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words produced by LLVM's assembler for the same instructions.
    #[test]
    fn encoders_match_llvm_for_representative_instructions() {
        assert_eq!(add_sub_imm(true, false, false, false, 1, 1, 0), 0x9100_0420); // add x0, x1, #1
        assert_eq!(add_sub_imm(true, true, true, false, 0, 0, 31), 0xF100_001F); // cmp x0, #0
        assert_eq!(pc_rel(false, -28, 0), 0x10FF_FF20); // adr x0, start (at 0x1c)
        assert_eq!(pc_rel(true, 0, 1), 0x9000_0001); // adrp x1, start
        assert_eq!(logical_imm(true, 0, true, 0, 7, 1, 0), 0x9240_1C20); // and x0, x1, #0xff
        assert_eq!(mov_wide(true, 2, 0, 1, 0), 0xD280_0020); // mov x0, #1
        assert_eq!(mov_wide(true, 3, 3, 0xFFFF, 3), 0xF2FF_FFE3); // movk x3, #0xffff, lsl #48
        assert_eq!(bitfield(true, 0, true, 4, 7, 1, 0), 0x9344_1C20); // sbfx x0, x1, #4, #4
        assert_eq!(extr(true, true, 2, 12, 1, 0), 0x93C2_3020);
        assert_eq!(b_imm(false, -30), 0x17FF_FFE2);
        assert_eq!(b_imm(true, -31), 0x97FF_FFE1);
        assert_eq!(b_cond(0, -32), 0x54FF_FC00);
        assert_eq!(cbz(true, false, -34, 0), 0xB4FF_FBC0);
        assert_eq!(tbz(false, 5, -36, 2), 0x362F_FB82);
        assert_eq!(tbz(false, 63, -38, 4), 0xB6FF_FB44);
        assert_eq!(branch_reg(0, 16), 0xD61F_0200);
        assert_eq!(branch_reg(2, 30), 0xD65F_03C0);
        assert_eq!(svc(0x1234), 0xD402_4681);
        assert_eq!(brk(0xF000), 0xD43E_0000);
        assert_eq!(nop(), 0xD503_201F);
        assert_eq!(hint(127), 0xD503_2FFF);
        assert_eq!(dsb(15), 0xD503_3F9F);
        assert_eq!(dmb(11), 0xD503_3BBF);
        assert_eq!(isb(15), 0xD503_3FDF);
        assert_eq!(clrex(5), 0xD503_355F);
        assert_eq!(sysreg(true, 3, 3, 4, 2, 0, 0), 0xD53B_4200); // mrs x0, nzcv
        assert_eq!(sysreg(false, 3, 3, 13, 0, 2, 5), 0xD51B_D045); // msr tpidr_el0, x5
        assert_eq!(sys(false, 3, 7, 4, 1, 0), 0xD50B_7420); // dc zva, x0
        assert_eq!(ldst_unsigned(3, 1, 0, 1, 0), 0xF940_0020); // ldr x0, [x1]
        assert_eq!(ldst_unsigned(2, 0, 4, 31, 6), 0xB900_13E6); // str w6, [sp, #16]
        assert_eq!(ldst_imm9(3, 1, 8, 3, 1, 0), 0xF840_8C20); // ldr x0, [x1, #8]!
        assert_eq!(ldst_imm9(0, 0, -1, 1, 7, 6), 0x381F_F4E6); // strb w6, [x7], #-1
        assert_eq!(ldst_imm9(3, 1, -8, 0, 1, 0), 0xF85F_8020); // ldur x0, [x1, #-8]
        assert_eq!(ldst_reg(3, 1, 2, 3, false, 1, 0), 0xF862_6820); // ldr x0, [x1, x2]
        assert_eq!(ldst_reg(3, 1, 5, 3, true, 4, 3), 0xF865_7883); // ldr x3, [x4, x5, lsl #3]
        assert_eq!(ldr_literal(1, -105, 0), 0x58FF_F2E0);
        assert_eq!(ldst_pair(2, 3, false, -2, 30, 31, 29), 0xA9BF_7BFD); // stp x29, x30, [sp, #-16]!
        assert_eq!(ldst_pair(2, 1, true, 2, 30, 31, 29), 0xA8C1_7BFD); // ldp x29, x30, [sp], #16
        assert_eq!(
            exclusive(3, false, true, false, 31, false, 31, 1, 0),
            0xC85F_7C20
        ); // ldxr x0, [x1]
        assert_eq!(
            exclusive(3, false, false, false, 10, false, 31, 12, 11),
            0xC80A_7D8B
        ); // stxr w10, x11, [x12]
        assert_eq!(
            exclusive(3, true, true, false, 31, true, 31, 1, 0),
            0xC8DF_FC20
        ); // ldar x0, [x1]
        assert_eq!(logical_shifted(true, 0, 0, false, 2, 0, 1, 0), 0x8A02_0020);
        assert_eq!(logical_shifted(true, 1, 0, true, 3, 0, 31, 2), 0xAA23_03E2); // mvn x2, x3
        assert_eq!(
            add_sub_shifted(true, false, false, 0, 2, 0, 1, 0),
            0x8B02_0020
        );
        assert_eq!(add_sub_ext(true, false, false, 1, 3, 2, 31, 0), 0x8B21_6BE0);
        assert_eq!(add_sub_carry(true, false, false, 2, 1, 0), 0x9A02_0020);
        assert_eq!(cond_cmp(true, true, false, 1, 0, 0, 4), 0xFA41_0004);
        assert_eq!(cond_cmp(true, true, true, 5, 8, 4, 15), 0xFA45_888F);
        assert_eq!(cond_sel(true, 0, 2, 0, 1, 0), 0x9A82_0020);
        assert_eq!(cond_sel(false, 3, 11, 11, 10, 9), 0x5A8B_B549);
        assert_eq!(dp1(true, 0, 1, 0), 0xDAC0_0020); // rbit x0, x1
        assert_eq!(dp2(true, 2, 2, 1, 0), 0x9AC2_0820); // udiv x0, x1, x2
        assert_eq!(dp3(true, 0, false, 2, 3, 1, 0), 0x9B02_0C20); // madd x0, x1, x2, x3
        assert_eq!(dp3(true, 6, false, 5, 31, 4, 3), 0x9BC5_7C83); // umulh x3, x4, x5
    }
}
