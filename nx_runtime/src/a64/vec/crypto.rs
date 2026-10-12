//! The cryptographic extension of Armv8.0: AES and SHA-1 / SHA-256 round operations.
//!
//! Every function is the Arm pseudocode of its instruction on 128-bit values (lane 0 in the low
//! bits, as the vector registers are stored). The AES S-box is computed at compile time from its
//! definition (the multiplicative inverse in GF(2^8) followed by the affine transformation of
//! FIPS 197), so there is no table to mistype. The tests check the whole algorithms against the
//! published examples (FIPS 197 for AES, FIPS 180 for SHA-1 and SHA-256).

// ---------------------------------------------------------------------------------------------
// AES
// ---------------------------------------------------------------------------------------------

/// Multiplication in GF(2^8) modulo `x^8 + x^4 + x^3 + x + 1`.
const fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut r = 0u8;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            r ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= 0x1B;
        }
        b >>= 1;
        i += 1;
    }
    r
}

const fn make_sbox() -> [u8; 256] {
    // Powers of the generator 3 and their logarithms give the inverses: 1/b = 3^(255 - log b).
    let mut exp = [0u8; 255];
    let mut log = [0u8; 256];
    let mut p = 1u8;
    let mut i = 0usize;
    while i < 255 {
        exp[i] = p;
        log[p as usize] = i as u8;
        p = gmul(p, 3);
        i += 1;
    }
    let mut s = [0u8; 256];
    let mut x = 0usize;
    while x < 256 {
        // Zero has no inverse and maps to zero.
        let inv = if x == 0 {
            0
        } else {
            exp[(255 - log[x] as usize) % 255]
        };
        s[x] = inv
            ^ inv.rotate_left(1)
            ^ inv.rotate_left(2)
            ^ inv.rotate_left(3)
            ^ inv.rotate_left(4)
            ^ 0x63;
        x += 1;
    }
    s
}

const fn invert(table: [u8; 256]) -> [u8; 256] {
    let mut out = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        out[table[i] as usize] = i as u8;
        i += 1;
    }
    out
}

const SBOX: [u8; 256] = make_sbox();
const INV_SBOX: [u8; 256] = invert(SBOX);

/// `AESShiftRows`: output byte `i` is input byte `5 i mod 16`.
fn shift_rows(b: [u8; 16]) -> [u8; 16] {
    std::array::from_fn(|i| b[(i * 5) % 16])
}

/// `AESInvShiftRows`: output byte `i` is input byte `13 i mod 16`.
fn inv_shift_rows(b: [u8; 16]) -> [u8; 16] {
    std::array::from_fn(|i| b[(i * 13) % 16])
}

fn mix_column(c: [u8; 4], m: [u8; 4]) -> [u8; 4] {
    // Row `r` of the circulant matrix with first row `m`.
    std::array::from_fn(|r| (0..4).fold(0u8, |acc, k| acc ^ gmul(c[k], m[(k + 4 - r) % 4])))
}

fn mix_columns(b: [u8; 16], m: [u8; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for col in 0..4 {
        let c = [b[4 * col], b[4 * col + 1], b[4 * col + 2], b[4 * col + 3]];
        out[4 * col..4 * col + 4].copy_from_slice(&mix_column(c, m));
    }
    out
}

/// `AESE Vd, Vn`: `SubBytes(ShiftRows(Vd ^ Vn))`.
pub fn aese(d: u128, n: u128) -> u128 {
    let s = shift_rows((d ^ n).to_le_bytes());
    u128::from_le_bytes(s.map(|x| SBOX[usize::from(x)]))
}

/// `AESD Vd, Vn`: `InvSubBytes(InvShiftRows(Vd ^ Vn))`.
pub fn aesd(d: u128, n: u128) -> u128 {
    let s = inv_shift_rows((d ^ n).to_le_bytes());
    u128::from_le_bytes(s.map(|x| INV_SBOX[usize::from(x)]))
}

/// `AESMC Vd, Vn`: `MixColumns(Vn)`.
pub fn aesmc(n: u128) -> u128 {
    u128::from_le_bytes(mix_columns(n.to_le_bytes(), [2, 3, 1, 1]))
}

/// `AESIMC Vd, Vn`: `InvMixColumns(Vn)`.
pub fn aesimc(n: u128) -> u128 {
    u128::from_le_bytes(mix_columns(n.to_le_bytes(), [14, 11, 13, 9]))
}

// ---------------------------------------------------------------------------------------------
// SHA-1
// ---------------------------------------------------------------------------------------------

fn word(v: u128, i: u32) -> u32 {
    (v >> (32 * i)) as u32
}

fn with_word(v: u128, i: u32, x: u32) -> u128 {
    (v & !(u128::from(u32::MAX) << (32 * i))) | (u128::from(x) << (32 * i))
}

fn choose(x: u32, y: u32, z: u32) -> u32 {
    ((y ^ z) & x) ^ z
}

fn majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}

fn parity(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

/// The three SHA-1 round functions the instructions apply four rounds of.
#[derive(Clone, Copy)]
pub enum Sha1Round {
    Choose,
    Parity,
    Majority,
}

/// `SHA1C`, `SHA1P`, `SHA1M`: four rounds on `abcd` (`Vd`), `e` (`Sn`) and the four words
/// `w + k` (`Vm`). Returns the new `abcd`.
pub fn sha1_rounds(round: Sha1Round, abcd: u128, e: u32, wk: u128) -> u128 {
    let (mut x, mut y) = (abcd, e);
    for i in 0..4 {
        let t = match round {
            Sha1Round::Choose => choose(word(x, 1), word(x, 2), word(x, 3)),
            Sha1Round::Parity => parity(word(x, 1), word(x, 2), word(x, 3)),
            Sha1Round::Majority => majority(word(x, 1), word(x, 2), word(x, 3)),
        };
        y = y
            .wrapping_add(word(x, 0).rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(word(wk, i));
        x = with_word(x, 1, word(x, 1).rotate_left(30));
        // <Y, X> = ROL(Y:X, 32): the 160-bit value Y:X rotated left by one word.
        let top = word(x, 3);
        x = (x << 32) | u128::from(y);
        y = top;
    }
    x
}

/// `SHA1H Sd, Sn`.
pub fn sha1h(e: u32) -> u32 {
    e.rotate_left(30)
}

/// `SHA1SU0 Vd.4S, Vn.4S, Vm.4S`.
pub fn sha1su0(d: u128, n: u128, m: u128) -> u128 {
    // t = Vn<63:0> : Vd<127:64>
    let t = (n << 64) | (d >> 64);
    d ^ t ^ m
}

/// `SHA1SU1 Vd.4S, Vn.4S`.
pub fn sha1su1(d: u128, n: u128) -> u128 {
    let t = d ^ (n >> 32);
    let mut r = 0u128;
    for i in 0..3 {
        r = with_word(r, i, word(t, i).rotate_left(1));
    }
    with_word(r, 3, word(t, 3).rotate_left(1) ^ word(t, 0).rotate_left(2))
}

// ---------------------------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------------------------

fn sigma0_hash(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

fn sigma1_hash(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

/// The shared body of `SHA256H` (`part1`) and `SHA256H2`: four rounds on the state halves `x`
/// and `y` with the four words `w + k`.
fn sha256_hash(mut x: u128, mut y: u128, wk: u128, part1: bool) -> u128 {
    for i in 0..4 {
        let chs = choose(word(y, 0), word(y, 1), word(y, 2));
        let maj = majority(word(x, 0), word(x, 1), word(x, 2));
        let t = word(y, 3)
            .wrapping_add(sigma1_hash(word(y, 0)))
            .wrapping_add(chs)
            .wrapping_add(word(wk, i));
        x = with_word(x, 3, t.wrapping_add(word(x, 3)));
        y = with_word(
            y,
            3,
            t.wrapping_add(sigma0_hash(word(x, 0))).wrapping_add(maj),
        );
        // <Y, X> = ROL(Y:X, 32): the 256-bit value Y:X rotated left by one word.
        let (new_x, new_y) = ((x << 32) | (y >> 96), (y << 32) | (x >> 96));
        x = new_x;
        y = new_y;
    }
    if part1 {
        x
    } else {
        y
    }
}

/// `SHA256H Qd, Qn, Vm.4S`: `Vd` is `abcd`, `Vn` is `efgh`.
pub fn sha256h(d: u128, n: u128, m: u128) -> u128 {
    sha256_hash(d, n, m, true)
}

/// `SHA256H2 Qd, Qn, Vm.4S`: `Vd` is `efgh`, `Vn` is `abcd`.
pub fn sha256h2(d: u128, n: u128, m: u128) -> u128 {
    sha256_hash(n, d, m, false)
}

/// `SHA256SU0 Vd.4S, Vn.4S`.
pub fn sha256su0(d: u128, n: u128) -> u128 {
    // T = Vn<31:0> : Vd<127:32>
    let t = (n << 96) | (d >> 32);
    let mut r = d;
    for i in 0..4 {
        let e = word(t, i);
        let s = e.rotate_right(7) ^ e.rotate_right(18) ^ (e >> 3);
        r = with_word(r, i, s.wrapping_add(word(d, i)));
    }
    r
}

/// `SHA256SU1 Vd.4S, Vn.4S, Vm.4S`.
pub fn sha256su1(d: u128, n: u128, m: u128) -> u128 {
    // T0 = Vm<31:0> : Vn<127:32>
    let t0 = (m << 96) | (n >> 32);
    let small = |e: u32| e.rotate_right(17) ^ e.rotate_right(19) ^ (e >> 10);
    let mut r = 0u128;
    // The two low words use the high half of Vm, the two high words the words just computed.
    let mut t1 = [word(m, 2), word(m, 3)];
    for e in 0..2u32 {
        let v = small(t1[e as usize])
            .wrapping_add(word(d, e))
            .wrapping_add(word(t0, e));
        r = with_word(r, e, v);
    }
    t1 = [word(r, 0), word(r, 1)];
    for e in 2..4u32 {
        let v = small(t1[(e - 2) as usize])
            .wrapping_add(word(d, e))
            .wrapping_add(word(t0, e));
        r = with_word(r, e, v);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex128(s: &str) -> u128 {
        // Written as the bytes of the block, first byte first.
        let bytes: Vec<u8> = (0..16)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        u128::from_le_bytes(bytes.try_into().unwrap())
    }

    #[test]
    fn the_sbox_has_the_published_values() {
        // FIPS 197, figure 7 and 14: a few entries and the first row.
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x01], 0x7C);
        assert_eq!(SBOX[0x53], 0xED);
        assert_eq!(SBOX[0xFF], 0x16);
        assert_eq!(
            &SBOX[..16],
            &[
                0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7,
                0xAB, 0x76
            ]
        );
        assert_eq!(INV_SBOX[0x63], 0x00);
        assert_eq!(INV_SBOX[0xED], 0x53);
        for i in 0..=255u8 {
            assert_eq!(INV_SBOX[usize::from(SBOX[usize::from(i)])], i);
        }
    }

    #[test]
    fn mix_columns_matches_the_fips_example_and_inverts() {
        // FIPS 197, section 5.1.3 and its test vectors: db 13 53 45 becomes 8e 4d a1 bc, f2 0a
        // 22 5c becomes 9f dc 58 9d, and a column of equal bytes is unchanged.
        let col = hex128("db135345f20a225c01010101c6c6c6c6");
        let want = hex128("8e4da1bc9fdc589d01010101c6c6c6c6");
        assert_eq!(aesmc(col), want);
        assert_eq!(aesimc(want), col);
        for seed in 1..50u128 {
            let v = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835);
            assert_eq!(aesimc(aesmc(v)), v);
            assert_eq!(aesd(aese(v, 0), 0), v);
        }
    }

    /// The AES-128 key schedule, written independently of the instructions.
    fn key_schedule(key: [u8; 16]) -> [u128; 11] {
        let mut w = [[0u8; 4]; 44];
        for i in 0..4 {
            w[i] = [key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]];
        }
        let mut rcon = 1u8;
        for i in 4..44 {
            let mut t = w[i - 1];
            if i % 4 == 0 {
                t = [t[1], t[2], t[3], t[0]].map(|b| SBOX[usize::from(b)]);
                t[0] ^= rcon;
                rcon = gmul(rcon, 2);
            }
            w[i] = std::array::from_fn(|k| w[i - 4][k] ^ t[k]);
        }
        std::array::from_fn(|r| {
            let mut b = [0u8; 16];
            for c in 0..4 {
                b[4 * c..4 * c + 4].copy_from_slice(&w[4 * r + c]);
            }
            u128::from_le_bytes(b)
        })
    }

    #[test]
    fn aes_128_encrypts_and_decrypts_the_fips_197_example() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let rk = key_schedule(key);
        let plain = hex128("3243f6a8885a308d313198a2e0370734");
        let cipher = hex128("3925841d02dc09fbdc118597196a0b32");
        // The instruction sequence compilers emit: nine AESE+AESMC rounds, AESE, then the last key.
        let mut s = plain;
        for k in &rk[..9] {
            s = aesmc(aese(s, *k));
        }
        s = aese(s, rk[9]) ^ rk[10];
        assert_eq!(s, cipher);
        // Decryption: AESD with the last key, then AESIMC after adding each earlier key.
        let mut t = aesd(cipher, rk[10]);
        for r in (1..=9).rev() {
            t = aesd(aesimc(t ^ rk[r]), 0);
        }
        assert_eq!(t ^ rk[0], plain);
    }

    fn be_words(block: &[u8]) -> [u32; 4] {
        std::array::from_fn(|i| u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap()))
    }

    fn pack(w: [u32; 4]) -> u128 {
        w.iter()
            .enumerate()
            .fold(0u128, |acc, (i, &x)| acc | (u128::from(x) << (32 * i)))
    }

    /// One padded block for the message "abc".
    fn abc_block() -> [u8; 64] {
        let mut b = [0u8; 64];
        b[..3].copy_from_slice(b"abc");
        b[3] = 0x80;
        b[63] = 24;
        b
    }

    #[test]
    fn sha_1_of_abc_with_the_instructions() {
        let block = abc_block();
        let mut w = [0u32; 80];
        for (i, x) in w.iter_mut().take(16).enumerate() {
            *x = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        // Message schedule four words at a time.
        for t in (16..80).step_by(4) {
            let v = |i: usize| pack([w[i], w[i + 1], w[i + 2], w[i + 3]]);
            let r = sha1su1(sha1su0(v(t - 16), v(t - 12), v(t - 8)), v(t - 4));
            for k in 0..4 {
                w[t + k] = word(r, k as u32);
            }
        }
        let h0 = [
            0x6745_2301u32,
            0xEFCD_AB89,
            0x98BA_DCFE,
            0x1032_5476,
            0xC3D2_E1F0,
        ];
        let k = [0x5A82_7999u32, 0x6ED9_EBA1, 0x8F1B_BCDC, 0xCA62_C1D6];
        let mut abcd = pack([h0[0], h0[1], h0[2], h0[3]]);
        let mut e = h0[4];
        for g in 0..20 {
            let kk = k[g / 5];
            let wk = pack(std::array::from_fn(|i| w[4 * g + i].wrapping_add(kk)));
            let e1 = sha1h(word(abcd, 0));
            let round = match g / 5 {
                0 => Sha1Round::Choose,
                2 => Sha1Round::Majority,
                _ => Sha1Round::Parity,
            };
            abcd = sha1_rounds(round, abcd, e, wk);
            e = e1;
        }
        let digest = [
            h0[0].wrapping_add(word(abcd, 0)),
            h0[1].wrapping_add(word(abcd, 1)),
            h0[2].wrapping_add(word(abcd, 2)),
            h0[3].wrapping_add(word(abcd, 3)),
            h0[4].wrapping_add(e),
        ];
        assert_eq!(
            digest,
            [
                0xa999_3e36,
                0x4706_816a,
                0xba3e_2571,
                0x7850_c26c,
                0x9cd0_d89d
            ]
        );
    }

    /// The first `n` primes.
    fn primes(n: usize) -> Vec<u128> {
        let mut v = Vec::new();
        let mut c = 2u128;
        while v.len() < n {
            if v.iter().all(|p| !c.is_multiple_of(*p)) {
                v.push(c);
            }
            c += 1;
        }
        v
    }

    /// floor(root(x)) by bisection.
    fn iroot(x: u128, degree: u32) -> u128 {
        let (mut lo, mut hi) = (0u128, 1u128 << (128 / degree).min(60));
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            let mut p = 1u128;
            let mut over = false;
            for _ in 0..degree {
                match p.checked_mul(mid) {
                    Some(q) if q <= x => p = q,
                    _ => {
                        over = true;
                        break;
                    }
                }
            }
            if over {
                hi = mid - 1;
            } else {
                lo = mid;
            }
        }
        lo
    }

    #[test]
    fn sha_256_of_abc_with_the_instructions() {
        // The constants are defined as fractional parts of roots of primes; compute them.
        let k: Vec<u32> = primes(64)
            .iter()
            .map(|&p| iroot(p << 96, 3) as u32)
            .collect();
        let h0: Vec<u32> = primes(8)
            .iter()
            .map(|&p| iroot(p << 64, 2) as u32)
            .collect();
        assert_eq!(k[0], 0x428a_2f98);
        assert_eq!(k[63], 0xc671_78f2);
        assert_eq!(h0[0], 0x6a09_e667);
        let block = abc_block();
        let mut m: Vec<u128> = (0..4).map(|i| pack(be_words(&block[16 * i..]))).collect();
        let mut abcd = pack([h0[0], h0[1], h0[2], h0[3]]);
        let mut efgh = pack([h0[4], h0[5], h0[6], h0[7]]);
        for g in 0..16 {
            if g >= 4 {
                // W[4g..4g+4] from the previous sixteen words.
                let next = sha256su1(sha256su0(m[g - 4], m[g - 3]), m[g - 2], m[g - 1]);
                m.push(next);
            }
            let kv = pack(std::array::from_fn(|i| k[4 * g + i]));
            let wk = pack(std::array::from_fn(|i| {
                word(m[g], i as u32).wrapping_add(word(kv, i as u32))
            }));
            let old = abcd;
            abcd = sha256h(abcd, efgh, wk);
            efgh = sha256h2(efgh, old, wk);
        }
        let digest: Vec<u32> = (0..4)
            .map(|i| h0[i].wrapping_add(word(abcd, i as u32)))
            .chain((0..4).map(|i| h0[4 + i].wrapping_add(word(efgh, i as u32))))
            .collect();
        assert_eq!(
            digest,
            [
                0xba78_16bf,
                0x8f01_cfea,
                0x4141_40de,
                0x5dae_2223,
                0xb003_61a3,
                0x9617_7a9c,
                0xb410_ff61,
                0xf200_15ad
            ]
        );
    }
}
