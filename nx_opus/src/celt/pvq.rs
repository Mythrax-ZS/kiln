//! Pyramid vector quantization (RFC 6716 sections 4.3.4.2 and 4.3.4.3): a band's shape is `K`
//! unit pulses over its `N` bins, coded as one index into the V(N, K) possible vectors, then
//! normalized and rotated to spread its energy.

use crate::range::RangeDecoder;

/// V(n, k) for n = 0..=N and k = 0..=K, row by row (the recurrence of section 4.3.4.2:
/// V(N, K) = V(N-1, K) + V(N, K-1) + V(N-1, K-1), V(N, 0) = 1, V(0, K) = 0).
fn counts(n: usize, k: usize) -> Vec<u64> {
    let stride = k + 1;
    let mut v = vec![0u64; (n + 1) * stride];
    v[0] = 1;
    for row in 1..=n {
        v[row * stride] = 1;
        for col in 1..=k {
            v[row * stride + col] = v[(row - 1) * stride + col]
                + v[row * stride + col - 1]
                + v[(row - 1) * stride + col - 1];
        }
    }
    v
}

/// Decode `K` pulses over `y.len()` bins (section 4.3.4.2); returns the vector's squared norm.
pub fn decode_pulses(y: &mut [i32], k: usize, dec: &mut RangeDecoder) -> f32 {
    let n = y.len();
    let v = counts(n, k);
    let at = |row: usize, col: usize| v[row * (k + 1) + col];
    let total = at(n, k);
    let mut i = u64::from(dec.uint(u32::try_from(total).expect("a codebook of at most 32 bits")));
    let mut k = k;
    for (j, out) in y.iter_mut().enumerate() {
        let rest = n - j;
        let mut p = (at(rest - 1, k) + at(rest, k)) / 2;
        let negative = i >= p;
        if negative {
            i -= p;
        }
        let k0 = k;
        p -= at(rest - 1, k);
        while p > i {
            k -= 1;
            p -= at(rest - 1, k);
        }
        i -= p;
        let mag = (k0 - k) as i32;
        *out = if negative { -mag } else { mag };
    }
    y.iter().map(|&x| (x * x) as f32).sum()
}

/// One pass of the spreading rotation (`exp_rotation1`): a cascade of 2-D rotations of the
/// pairs `stride` apart, forward then backward.
fn rotate_pass(x: &mut [f32], stride: usize, c: f32, s: f32) {
    let len = x.len();
    if len <= stride {
        return;
    }
    for i in 0..len - stride {
        let (x1, x2) = (x[i], x[i + stride]);
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    if len < 2 * stride + 1 {
        return;
    }
    for i in (0..len - 2 * stride).rev() {
        let (x1, x2) = (x[i], x[i + stride]);
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
}

/// The spreading rotation (section 4.3.4.3), undone (`dir` < 0) as the decoder does, for a
/// band of `blocks` interleaved blocks holding `k` pulses. `spread` 0 is no rotation.
pub fn spread_rotation(x: &mut [f32], dir: i32, blocks: usize, k: usize, spread: usize) {
    const FACTOR: [usize; 3] = [15, 10, 5];
    let len = x.len();
    if 2 * k >= len || spread == 0 {
        return;
    }
    let gain = len as f32 / (len + FACTOR[spread - 1] * k) as f32;
    let theta = 0.5 * gain * gain;
    let c = (0.5 * std::f32::consts::PI * theta).cos();
    let s = (0.5 * std::f32::consts::PI * (1.0 - theta)).cos();
    let mut stride2 = 0;
    if len >= 8 * blocks {
        stride2 = 1;
        // About sqrt(len / blocks), rounded.
        while (stride2 * stride2 + stride2) * blocks + (blocks >> 2) < len {
            stride2 += 1;
        }
    }
    let sub = len / blocks;
    for block in x.chunks_exact_mut(sub) {
        if dir < 0 {
            if stride2 != 0 {
                rotate_pass(block, stride2, s, c);
            }
            rotate_pass(block, 1, c, s);
        } else {
            rotate_pass(block, 1, c, -s);
            if stride2 != 0 {
                rotate_pass(block, stride2, s, -c);
            }
        }
    }
}

/// Which of the `blocks` interleaved blocks received a pulse (`extract_collapse_mask`).
fn collapse_mask(y: &[i32], blocks: usize) -> u32 {
    if blocks <= 1 {
        return 1;
    }
    let n0 = y.len() / blocks;
    let mut mask = 0;
    for (b, block) in y.chunks_exact(n0).enumerate() {
        if block.iter().any(|&v| v != 0) {
            mask |= 1 << b;
        }
    }
    mask
}

/// Decode a band's shape (`alg_unquant`): `k` pulses, normalized to `gain`, then the spreading
/// undone. Returns the collapse mask of its blocks.
pub fn decode_shape(
    x: &mut [f32],
    k: usize,
    spread: usize,
    blocks: usize,
    gain: f32,
    dec: &mut RangeDecoder,
) -> u32 {
    let mut y = vec![0i32; x.len()];
    let ryy = decode_pulses(&mut y, k, dec);
    let g = gain / ryy.sqrt();
    for (o, &v) in x.iter_mut().zip(&y) {
        *o = g * v as f32;
    }
    spread_rotation(x, -1, blocks, k, spread);
    collapse_mask(&y, blocks)
}

/// Scale a vector to the norm `gain` (`renormalise_vector`).
pub fn renormalize(x: &mut [f32], gain: f32) {
    let e = 1e-15 + x.iter().map(|v| v * v).sum::<f32>();
    let g = gain / e.sqrt();
    for v in x {
        *v *= g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_sizes_follow_the_recurrence() {
        let v = counts(4, 3);
        // V(1, K) = 2 for K > 0; V(2, K) = 4K; V(N, 1) = 2N.
        assert_eq!(v[5 + 2], 2);
        assert_eq!(v[2 * 4 + 3], 12);
        assert_eq!(v[4 * 4 + 1], 8);
        // V(3, 2) = 18 (the 3 positions times 4 sign pairs of two pulses... by the recurrence).
        assert_eq!(v[3 * 4 + 2], v[2 * 4 + 2] + v[3 * 4 + 1] + v[2 * 4 + 1]);
    }

    #[test]
    fn every_index_decodes_to_a_distinct_vector_of_k_pulses() {
        // Enumerate the vectors of N = 3, K = 2 by the procedure of section 4.3.4.2 directly.
        let (n, k) = (3usize, 2usize);
        let v = counts(n, k);
        let total = v[n * (k + 1) + k] as usize;
        let mut seen = std::collections::HashSet::new();
        for index in 0..total {
            let mut i = index as u64;
            let mut kk = k;
            let mut y = vec![0i32; n];
            for (j, out) in y.iter_mut().enumerate() {
                let at = |row: usize, col: usize| v[row * (k + 1) + col];
                let rest = n - j;
                let mut p = (at(rest - 1, kk) + at(rest, kk)) / 2;
                let neg = i >= p;
                if neg {
                    i -= p;
                }
                let k0 = kk;
                p -= at(rest - 1, kk);
                while p > i {
                    kk -= 1;
                    p -= at(rest - 1, kk);
                }
                i -= p;
                *out = if neg {
                    -((k0 - kk) as i32)
                } else {
                    (k0 - kk) as i32
                };
            }
            assert_eq!(y.iter().map(|v| v.abs()).sum::<i32>(), k as i32, "{y:?}");
            assert!(seen.insert(y));
        }
        assert_eq!(seen.len(), 18);
    }

    #[test]
    fn spreading_preserves_the_norm_and_inverts() {
        let mut x: Vec<f32> = (0..16).map(|i| ((i * 7 % 5) as f32) - 2.0).collect();
        let orig = x.clone();
        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        spread_rotation(&mut x, 1, 2, 3, 2);
        assert!((norm(&x) - norm(&orig)).abs() < 1e-4);
        spread_rotation(&mut x, -1, 2, 3, 2);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
