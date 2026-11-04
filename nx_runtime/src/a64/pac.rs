//! `ComputePAC` of the Arm pseudocode: the QARMA5 block cipher the architecture's pointer
//! authentication uses (FEAT_PACQARMA5), for `PACGA`.
//!
//! The other keys are off (docs/ASSUMPTIONS.md A-189): adding or checking a code returns the
//! pointer unchanged and needs no cipher. `PACGA` is not switched off with them, so it computes,
//! with the generic key this runtime gives every process, zero.

/// The generic key, `APGAKeyHi_EL1:APGAKeyLo_EL1` (A-189).
pub const GENERIC_KEY: (u64, u64) = (0, 0);

const RC: [u64; 5] = [
    0x0000_0000_0000_0000,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
    0x4528_21E6_38D0_1377,
];
const ALPHA: u64 = 0xC0AC_29B7_C97C_50DD;

/// Output nibble `i` of `PACCellShuffle` is input nibble `SHUFFLE[i]` (nibble 0 is bits 3:0).
const SHUFFLE: [u8; 16] = [13, 6, 11, 0, 7, 12, 1, 10, 8, 3, 14, 5, 2, 9, 4, 15];
const SUB: [u8; 16] = [
    0xB, 0x6, 0x8, 0xF, 0xC, 0x0, 0x9, 0xE, 0x3, 0x7, 0x4, 0x5, 0xD, 0x2, 0x1, 0xA,
];
/// Output nibble `k` of `TweakShuffle` is input nibble `TWEAK[k]`, rotated when bit `k` of
/// `TWEAK_ROT` is set.
const TWEAK: [u8; 16] = [4, 5, 6, 7, 11, 2, 3, 8, 12, 13, 14, 15, 0, 1, 10, 9];
const TWEAK_ROT: u16 = 1 << 2 | 1 << 4 | 1 << 7 | 1 << 11 | 1 << 12 | 1 << 14 | 1 << 15;

fn nibble(x: u64, i: usize) -> u64 {
    x >> (4 * i) & 0xF
}

fn shuffle(x: u64) -> u64 {
    (0..16).fold(0, |o, i| o | nibble(x, SHUFFLE[i] as usize) << (4 * i))
}

fn inv_shuffle(x: u64) -> u64 {
    (0..16).fold(0, |o, i| o | nibble(x, i) << (4 * SHUFFLE[i] as usize))
}

fn sub(x: u64, table: &[u8; 16]) -> u64 {
    (0..16).fold(0, |o, i| {
        o | u64::from(table[nibble(x, i) as usize]) << (4 * i)
    })
}

fn inv_sub_table() -> [u8; 16] {
    let mut t = [0u8; 16];
    for (i, &s) in SUB.iter().enumerate() {
        t[s as usize] = i as u8;
    }
    t
}

fn rot(cell: u64, n: u32) -> u64 {
    (cell << n | cell >> (4 - n)) & 0xF
}

/// `PACMult`, an involution.
fn mult(x: u64) -> u64 {
    let mut o = 0;
    for i in 0..4 {
        let (c0, c4, c8, c12) = (
            nibble(x, i),
            nibble(x, i + 4),
            nibble(x, i + 8),
            nibble(x, i + 12),
        );
        let t0 = rot(c8, 1) ^ rot(c4, 2) ^ rot(c0, 1);
        let t1 = rot(c12, 1) ^ rot(c4, 1) ^ rot(c0, 2);
        let t2 = rot(c12, 2) ^ rot(c8, 1) ^ rot(c0, 1);
        let t3 = rot(c12, 1) ^ rot(c8, 2) ^ rot(c4, 1);
        o |= t3 << (4 * i) | t2 << (4 * (i + 4)) | t1 << (4 * (i + 8)) | t0 << (4 * (i + 12));
    }
    o
}

fn tweak_cell_rot(c: u64) -> u64 {
    c >> 1 | ((c ^ c >> 1) & 1) << 3
}

fn tweak_cell_inv_rot(c: u64) -> u64 {
    (c << 1 & 0xF) | ((c & 1) ^ c >> 3)
}

fn tweak_shuffle(x: u64) -> u64 {
    (0..16).fold(0, |o, k| {
        let c = nibble(x, TWEAK[k] as usize);
        let c = if TWEAK_ROT >> k & 1 == 1 {
            tweak_cell_rot(c)
        } else {
            c
        };
        o | c << (4 * k)
    })
}

fn tweak_inv_shuffle(x: u64) -> u64 {
    (0..16).fold(0, |o, k| {
        let c = nibble(x, k);
        let c = if TWEAK_ROT >> k & 1 == 1 {
            tweak_cell_inv_rot(c)
        } else {
            c
        };
        o | c << (4 * TWEAK[k] as usize)
    })
}

/// `ComputePAC(data, modifier, key0, key1)`, where `key0` is bits 127:64 of the key.
pub fn compute_pac(data: u64, modifier: u64, key0: u64, key1: u64) -> u64 {
    let inv_sub = inv_sub_table();
    let modk0 = key0 << 63 | (key0 >> 1 ^ key0 >> 63);
    let mut runningmod = modifier;
    let mut w = data ^ key0;
    for (i, rc) in RC.iter().enumerate() {
        w ^= key1 ^ runningmod;
        w ^= rc;
        if i > 0 {
            w = mult(shuffle(w));
        }
        w = sub(w, &SUB);
        runningmod = tweak_shuffle(runningmod);
    }
    w ^= modk0 ^ runningmod;
    w = sub(mult(shuffle(w)), &SUB);
    w = mult(shuffle(w));
    w ^= key1;
    w = inv_shuffle(w);
    w = mult(sub(w, &inv_sub));
    w = inv_shuffle(w);
    w ^= key0 ^ runningmod;
    for i in 0..5 {
        w = sub(w, &inv_sub);
        if i < 4 {
            w = inv_shuffle(mult(w));
        }
        runningmod = tweak_inv_shuffle(runningmod);
        w ^= RC[4 - i] ^ key1 ^ runningmod ^ ALPHA;
    }
    w ^ modk0
}

/// `PACGA Xd, Xn, Xm|SP`: the top 32 bits of the code of `data` under `modifier`, the low 32 zero.
pub fn pacga(data: u64, modifier: u64) -> u64 {
    let (hi, lo) = GENERIC_KEY;
    compute_pac(data, modifier, hi, lo) & 0xFFFF_FFFF_0000_0000
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from Unicorn 2.1.4's `PACGA` (QEMU's QARMA5) on its `max` core, with the generic
    /// key set to `key0:key1`; 300 random cases agreed when this was written.
    #[test]
    fn matches_the_reference_values() {
        for (data, modifier, key0, key1, pac) in [
            (
                0xf2a7_4de4_52e6_b438u64,
                0x6513_270e_269e_0d37u64,
                0xd23f_0824_128b_2f33u64,
                0x0c5c_7fd0_a6a3_a450u64,
                0xaff1_3c6f_2a4e_9d00u64,
            ),
            (
                0x1818_e811_892f_902b,
                0x9531_985d_5d9d_c9f8,
                0x36f6_75cc_81e7_4ef5,
                0xe8e2_5d94_0ed9_0475,
                0xbd8c_7542_aed9_7e43,
            ),
            (
                0x1600_a35a_0999_50d8,
                0x6b0d_549b_6f03_675a,
                0x8d11_6ece_1738_f7d9,
                0x3d9c_1724_11e2_0b8f,
                0x83e5_10f7_4947_7a42,
            ),
        ] {
            assert_eq!(compute_pac(data, modifier, key0, key1), pac, "{data:#x}");
        }
        assert_eq!(pacga(1, 2), 0x8d6a_3922_0000_0000);
        assert_eq!(pacga(0, 0), 0x7624_3b95_0000_0000);
    }

    #[test]
    fn the_inverse_steps_undo_the_forward_ones() {
        let inv_sub = inv_sub_table();
        for x in [0u64, u64::MAX, 0x0123_4567_89AB_CDEF, 0xF00D_CAFE_DEAD_BEEF] {
            assert_eq!(inv_shuffle(shuffle(x)), x);
            assert_eq!(sub(sub(x, &SUB), &inv_sub), x);
            assert_eq!(mult(mult(x)), x, "PACMult is an involution");
            assert_eq!(tweak_inv_shuffle(tweak_shuffle(x)), x);
        }
    }
}
