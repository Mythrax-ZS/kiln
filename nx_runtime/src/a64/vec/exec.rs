//! Execution of the decoded floating-point and SIMD operations.

use super::{FpBinary, FpUnary, VMem, VOp};
use crate::a64::bits::condition_holds;
use crate::a64::bus::{Bus, Fault};
use crate::a64::fp::{self, Env, Fmt};
use crate::a64::state::CpuState;

/// `VFPExpandImm`.
pub fn expand_imm(fmt: Fmt, imm8: u8) -> u64 {
    let imm = u64::from(imm8);
    let sign = (imm >> 7) & 1;
    let b6 = (imm >> 6) & 1;
    let e = fmt.exp_bits();
    // exp = NOT(b6) : Replicate(b6, E-3) : imm8<5:4>
    let rep = if b6 == 1 { (1u64 << (e - 3)) - 1 } else { 0 };
    let exp = ((1 - b6) << (e - 1)) | (rep << 2) | ((imm >> 4) & 3);
    let frac = (imm & 0xF) << (fmt.frac() - 4);
    (sign << (fmt.bits() - 1)) | (exp << fmt.frac()) | frac
}

/// Execute one operation. `src` is the general source register's value (0 when the operation has
/// none), `nzcv` the current flags in bits 3 to 0. Returns the general destination's value, or the
/// new flags for the compare instructions.
pub fn exec(cpu: &mut CpuState, op: &VOp, src: u64, nzcv: u8) -> u64 {
    let mut env = Env::new(cpu.fpcr, cpu.fpsr);
    let result = run(cpu, &mut env, op, src, nzcv);
    cpu.fpsr = env.fpsr;
    result
}

fn run(cpu: &mut CpuState, env: &mut Env, op: &VOp, src: u64, nzcv: u8) -> u64 {
    match *op {
        VOp::FpMov { fmt, rd, rn } => {
            let v = cpu.scalar(rn, fmt);
            cpu.set_scalar(rd, v);
            0
        }
        VOp::FpUnary { op, fmt, rd, rn } => {
            let a = cpu.scalar(rn, fmt);
            let r = match op {
                FpUnary::Abs => fp::abs(fmt, a),
                FpUnary::Neg => fp::neg(fmt, a),
                FpUnary::Sqrt => fp::sqrt(env, fmt, a),
                FpUnary::Round { mode, exact } => {
                    let mode = mode.unwrap_or_else(|| env.rounding());
                    fp::round_int(env, fmt, a, mode, exact)
                }
            };
            cpu.set_scalar(rd, r);
            0
        }
        VOp::FpConvert { from, to, rd, rn } => {
            let a = cpu.scalar(rn, from);
            let r = fp::convert(env, from, to, a);
            cpu.set_scalar(rd, r);
            0
        }
        VOp::FpBinary {
            op,
            fmt,
            rd,
            rn,
            rm,
        } => {
            let (a, b) = (cpu.scalar(rn, fmt), cpu.scalar(rm, fmt));
            let r = match op {
                FpBinary::Mul => fp::mul(env, fmt, a, b),
                FpBinary::Div => fp::div(env, fmt, a, b),
                FpBinary::Add => fp::add(env, fmt, a, b),
                FpBinary::Sub => fp::sub(env, fmt, a, b),
                FpBinary::MinMax(m) => fp::min_max(env, fmt, m, a, b),
                FpBinary::Nmul => {
                    let p = fp::mul(env, fmt, a, b);
                    fp::neg(fmt, p)
                }
            };
            cpu.set_scalar(rd, r);
            0
        }
        VOp::FpFused {
            neg_addend,
            neg_product,
            fmt,
            rd,
            rn,
            rm,
            ra,
        } => {
            let mut addend = cpu.scalar(ra, fmt);
            let mut op1 = cpu.scalar(rn, fmt);
            let op2 = cpu.scalar(rm, fmt);
            if neg_addend {
                addend = fp::neg(fmt, addend);
            }
            if neg_product {
                op1 = fp::neg(fmt, op1);
            }
            let r = fp::mul_add(env, fmt, addend, op1, op2);
            cpu.set_scalar(rd, r);
            0
        }
        VOp::FpCompare {
            fmt,
            rn,
            rm,
            signal,
        } => {
            let a = cpu.scalar(rn, fmt);
            let b = rm.map_or(0, |m| cpu.scalar(m, fmt));
            u64::from(fp::compare(env, fmt, a, b, signal))
        }
        VOp::FpCondCompare {
            fmt,
            rn,
            rm,
            cond,
            nzcv: imm,
            signal,
        } => {
            if condition_holds(cond, nzcv) {
                let (a, b) = (cpu.scalar(rn, fmt), cpu.scalar(rm, fmt));
                u64::from(fp::compare(env, fmt, a, b, signal))
            } else {
                u64::from(imm)
            }
        }
        VOp::FpSelect {
            fmt,
            rd,
            rn,
            rm,
            cond,
        } => {
            let pick = if condition_holds(cond, nzcv) { rn } else { rm };
            let v = cpu.scalar(pick, fmt);
            cpu.set_scalar(rd, v);
            0
        }
        VOp::FpImmediate { fmt, rd, imm8 } => {
            cpu.set_scalar(rd, expand_imm(fmt, imm8));
            0
        }
        VOp::FpToInt {
            fmt,
            wide,
            signed,
            mode,
            fbits,
            rn,
            ..
        } => {
            let a = cpu.scalar(rn, fmt);
            fp::to_int(env, fmt, a, if wide { 64 } else { 32 }, signed, fbits, mode)
        }
        VOp::IntToFp {
            fmt,
            wide,
            signed,
            fbits,
            rd,
            ..
        } => {
            let r = fp::from_int(env, fmt, src, if wide { 64 } else { 32 }, signed, fbits);
            cpu.set_scalar(rd, r);
            0
        }
        VOp::FmovToGeneral { wide, high, rn, .. } => {
            if high {
                cpu.v[usize::from(rn)][1]
            } else if wide {
                cpu.v[usize::from(rn)][0]
            } else {
                cpu.v[usize::from(rn)][0] & 0xFFFF_FFFF
            }
        }
        VOp::FmovFromGeneral { wide, high, rd, .. } => {
            if high {
                cpu.v[usize::from(rd)][1] = src;
            } else if wide {
                cpu.set_scalar(rd, src);
            } else {
                cpu.set_scalar(rd, src & 0xFFFF_FFFF);
            }
            0
        }
        VOp::Simd(simd) => super::simd_exec::exec(cpu, env, &simd, src),
    }
}

/// Execute a vector load or store at `addr`. Loads write the register file only when every access
/// succeeded; a store that faults partway may have stored the first half of a pair.
pub fn exec_mem<B: Bus + ?Sized>(
    cpu: &mut CpuState,
    bus: &mut B,
    mem: &VMem,
    addr: u64,
) -> Result<(), Fault> {
    match *mem {
        VMem::Single { load, bytes, rt } => {
            if load {
                let v = bus.read(addr, bytes)?;
                cpu.set_vreg(rt, v);
            } else {
                let v = cpu.vreg(rt) & mask_bytes(bytes);
                bus.write(addr, bytes, v)?;
            }
        }
        VMem::Pair {
            load,
            bytes,
            rt,
            rt2,
        } => {
            let second = addr.wrapping_add(u64::from(bytes));
            if load {
                let a = bus.read(addr, bytes)?;
                let b = bus.read(second, bytes)?;
                cpu.set_vreg(rt, a);
                cpu.set_vreg(rt2, b);
            } else {
                let a = cpu.vreg(rt) & mask_bytes(bytes);
                let b = cpu.vreg(rt2) & mask_bytes(bytes);
                bus.write(addr, bytes, a)?;
                bus.write(second, bytes, b)?;
            }
        }
        VMem::Multi {
            load,
            q,
            size,
            rpt,
            selem,
            rt,
        } => {
            let ebytes = 1u32 << size;
            let elements = (if q { 16 } else { 8 }) / ebytes;
            let ebits = 8 * ebytes;
            let mut regs = [0u128; 4];
            let mut offs = 0u64;
            for r in 0..u32::from(rpt) {
                for e in 0..elements {
                    for s in 0..u32::from(selem) {
                        let idx = (r + s) as usize;
                        let reg = ((u32::from(rt) + r + s) % 32) as u8;
                        let at = addr.wrapping_add(offs);
                        if load {
                            let v = bus.read(at, ebytes as u8)?;
                            let m = mask_bytes(ebytes as u8) << (e * ebits);
                            regs[idx] = (regs[idx] & !m) | (v << (e * ebits));
                        } else {
                            let v = (cpu.vreg(reg) >> (e * ebits)) & mask_bytes(ebytes as u8);
                            bus.write(at, ebytes as u8, v)?;
                        }
                        offs += u64::from(ebytes);
                    }
                }
            }
            if load {
                for (i, v) in regs.iter().enumerate().take((rpt * selem) as usize) {
                    cpu.set_vreg(((u32::from(rt) + i as u32) % 32) as u8, *v);
                }
            }
        }
        VMem::Lane {
            load,
            size,
            selem,
            index,
            rt,
        } => {
            let ebytes = 1u32 << size;
            let ebits = 8 * ebytes;
            let mut loaded = [0u128; 4];
            for s in 0..u32::from(selem) {
                let at = addr.wrapping_add(u64::from(s * ebytes));
                if load {
                    loaded[s as usize] = bus.read(at, ebytes as u8)?;
                } else {
                    let reg = ((u32::from(rt) + s) % 32) as u8;
                    let v =
                        (cpu.vreg(reg) >> (u32::from(index) * ebits)) & mask_bytes(ebytes as u8);
                    bus.write(at, ebytes as u8, v)?;
                }
            }
            if load {
                for s in 0..u32::from(selem) {
                    let reg = ((u32::from(rt) + s) % 32) as u8;
                    let m = mask_bytes(ebytes as u8) << (u32::from(index) * ebits);
                    let old = cpu.vreg(reg);
                    cpu.set_vreg(
                        reg,
                        (old & !m) | (loaded[s as usize] << (u32::from(index) * ebits)),
                    );
                }
            }
        }
        VMem::Replicate { q, size, selem, rt } => {
            let ebytes = 1u32 << size;
            let ebits = 8 * ebytes;
            let mut loaded = [0u128; 4];
            for s in 0..u32::from(selem) {
                loaded[s as usize] =
                    bus.read(addr.wrapping_add(u64::from(s * ebytes)), ebytes as u8)?;
            }
            for s in 0..u32::from(selem) {
                let mut v = 0u128;
                for e in 0..(if q { 16 } else { 8 }) / ebytes {
                    v |= loaded[s as usize] << (e * ebits);
                }
                cpu.set_vreg(((u32::from(rt) + s) % 32) as u8, v);
            }
        }
    }
    Ok(())
}

fn mask_bytes(bytes: u8) -> u128 {
    if bytes == 16 {
        u128::MAX
    } else {
        (1u128 << (8 * u32::from(bytes))) - 1
    }
}
