//! What the 48 kHz CELT mode derives from its tables: the width of each band in log units, the
//! MDCT window, and the cache of what PVQ codebooks cost (section 4.3.3; the reference keeps
//! these precomputed in `celt/static_modes_float.h` and computes them in `celt/rate.c`'s
//! `compute_pulse_cache`, which this follows). They are computed once, at first use.

use std::sync::OnceLock;

use super::tables::{BANDS, EBANDS};
use crate::range::ilog;

/// The largest frame is `1 << MAX_LM` short blocks (20 ms).
pub const MAX_LM: usize = 3;
/// The overlap of two MDCT windows, in samples (2.5 ms).
pub const OVERLAP: usize = 120;
/// The size of a short block's MDCT, in bins.
pub const SHORT_MDCT: usize = 120;
/// Bits are counted in eighths in the allocation (`BITRES`).
pub const BITRES: i32 = 3;
pub const MAX_FINE_BITS: i32 = 8;
pub const FINE_OFFSET: i32 = 21;
pub const QTHETA_OFFSET: i32 = 4;
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const MAX_PSEUDO: i32 = 40;
const LOG_MAX_PSEUDO: usize = 6;

pub struct Mode {
    /// log2 of each band's width at 2.5 ms, in eighths of a bit.
    pub log_n: [i32; BANDS],
    /// The rising half of the low-overlap window (section 4.3.7).
    pub window: [f32; OVERLAP],
    /// For each band and split depth (`lm + 1`, -1 to 3), where its row of `cache_bits` starts.
    cache_index: [i16; BANDS * (MAX_LM + 2)],
    /// Each row: its largest pseudo-pulse count, then what 1, 2... pseudo-pulses cost, in
    /// eighths of a bit, less one.
    cache_bits: Vec<u8>,
    /// The most each band can use, by frame size and channel count, in 1/4 bit per bin less 64.
    caps: [u8; BANDS * 2 * (MAX_LM + 1)],
}

pub fn mode() -> &'static Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    MODE.get_or_init(Mode::build)
}

/// The number of pulses a pseudo-pulse count stands for: 0..7 exactly, then 8 steps per octave.
pub fn pulses_of(q: i32) -> i32 {
    if q < 8 {
        q
    } else {
        (8 + (q & 7)) << ((q >> 3) - 1)
    }
}

/// log2 of `val` with `frac` fractional bits, rounded up (`celt/cwrs.c`, `log2_frac`).
fn log2_frac(mut val: u32, mut frac: u32) -> i32 {
    let l = ilog(val) as i32;
    if val & (val - 1) == 0 {
        return (l - 1) << frac;
    }
    if l > 16 {
        let sh = (l - 16) as u32;
        val = (val >> sh) + (((val & ((1 << sh) - 1)) + (1 << sh) - 1) >> sh);
    } else {
        val <<= 16 - l;
    }
    let mut out = (l - 1) << frac;
    loop {
        let b = (val >> 16) as i32;
        out += b << frac;
        val = (val + b as u32) >> b;
        val = (val * val + 0x7FFF) >> 15;
        if frac == 0 {
            break;
        }
        frac -= 1;
    }
    out + i32::from(val > 0x8000)
}

/// V(N, K) for K = 0..=kmax (section 4.3.4.2), saturating where it passes 64 bits.
fn pvq_counts(n: usize, kmax: usize) -> Vec<u64> {
    let mut row = vec![0u64; kmax + 1];
    row[0] = 1; // V(0, 0) = 1, V(0, K) = 0.
    for _ in 0..n {
        let mut prev_diag = row[0]; // V(n-1, k-1)
        for k in 1..=kmax {
            let up = row[k]; // V(n-1, k)
            row[k] = up.saturating_add(row[k - 1]).saturating_add(prev_diag);
            prev_diag = up;
        }
    }
    row
}

/// Whether V(N, K) fits in 32 bits, as the reference's limits say (`celt/rate.c`, `fits_in32`).
fn fits_in_32(n: i32, k: i32) -> bool {
    const MAX_N: [i32; 15] = [
        32767, 32767, 32767, 1476, 283, 109, 60, 40, 29, 24, 20, 18, 16, 14, 13,
    ];
    const MAX_K: [i32; 15] = [
        32767, 32767, 32767, 32767, 1172, 238, 95, 53, 36, 27, 22, 18, 16, 15, 13,
    ];
    if n >= 14 {
        k < 14 && n <= MAX_N[k as usize]
    } else {
        k <= MAX_K[n as usize]
    }
}

impl Mode {
    fn build() -> Mode {
        let mut log_n = [0; BANDS];
        for (i, l) in log_n.iter_mut().enumerate() {
            *l = log2_frac((EBANDS[i + 1] - EBANDS[i]) as u32, BITRES as u32);
        }
        let mut window = [0f32; OVERLAP];
        for (i, w) in window.iter_mut().enumerate() {
            let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / OVERLAP as f64).sin();
            *w = (0.5 * std::f64::consts::PI * s * s).sin() as f32;
        }

        // The cache: one row per distinct band size at each split depth.
        let width = |j: usize| (EBANDS[j + 1] - EBANDS[j]) as i32;
        let mut cache_index = [-1i16; BANDS * (MAX_LM + 2)];
        let mut entries: Vec<(i32, i32, usize)> = Vec::new(); // (N, K, start)
        let mut size = 0usize;
        for i in 0..=MAX_LM + 1 {
            for j in 0..BANDS {
                let n = width(j) << i >> 1;
                let same = (0..=i).find_map(|k| {
                    (0..BANDS)
                        .take_while(|&m| k != i || m < j)
                        .find(|&m| n == width(m) << k >> 1)
                        .map(|m| cache_index[k * BANDS + m])
                });
                if let Some(index) = same {
                    cache_index[i * BANDS + j] = index;
                }
                if cache_index[i * BANDS + j] == -1 && n != 0 {
                    let mut k = 0;
                    while fits_in_32(n, pulses_of(k + 1)) && k < MAX_PSEUDO {
                        k += 1;
                    }
                    cache_index[i * BANDS + j] = size as i16;
                    entries.push((n, k, size));
                    size += k as usize + 1;
                }
            }
        }
        let mut cache_bits = vec![0u8; size];
        for &(n, k, start) in &entries {
            // A band of one bin costs one sign bit whatever its pulses (`get_required_bits`).
            let counts = if n == 1 {
                Vec::new()
            } else {
                pvq_counts(n as usize, pulses_of(k) as usize)
            };
            cache_bits[start] = k as u8;
            for q in 1..=k {
                let bits = if n == 1 {
                    1 << BITRES
                } else {
                    let v = counts[pulses_of(q) as usize];
                    log2_frac(
                        u32::try_from(v).expect("a cached codebook fits in 32 bits"),
                        BITRES as u32,
                    )
                };
                cache_bits[start + q as usize] = (bits - 1) as u8;
            }
        }

        let mut mode = Mode {
            log_n,
            window,
            cache_index,
            cache_bits,
            caps: [0; BANDS * 2 * (MAX_LM + 1)],
        };
        mode.caps = mode.compute_caps();
        mode
    }

    /// The most bits each band can use reliably (`compute_pulse_cache`'s second half).
    fn compute_caps(&self) -> [u8; BANDS * 2 * (MAX_LM + 1)] {
        let mut caps = [0u8; BANDS * 2 * (MAX_LM + 1)];
        let mut at = 0;
        for i in 0..=MAX_LM as i32 {
            for c in 1..=2i32 {
                for j in 0..BANDS {
                    let w = (EBANDS[j + 1] - EBANDS[j]) as i32;
                    let mut n0 = w;
                    let mut max_bits;
                    if n0 << i == 1 {
                        max_bits = (c * (1 + MAX_FINE_BITS)) << BITRES;
                    } else {
                        let mut lm0 = 0;
                        if n0 > 2 {
                            n0 >>= 1;
                            lm0 -= 1;
                        } else if n0 <= 1 {
                            lm0 = i.min(1);
                            n0 <<= lm0;
                        }
                        let row = self.cache(j, lm0);
                        max_bits = i32::from(row[row[0] as usize]) + 1;
                        let mut n = n0;
                        for k in 0..i - lm0 {
                            max_bits <<= 1;
                            let offset =
                                ((self.log_n[j] + ((lm0 + k) << BITRES)) >> 1) - QTHETA_OFFSET;
                            let num = 459 * ((2 * n - 1) * offset + max_bits);
                            let den = ((2 * n - 1) << 9) - 459;
                            max_bits += ((num + (den >> 1)) / den).min(57);
                            n <<= 1;
                        }
                        if c == 2 {
                            max_bits <<= 1;
                            let two_phase = n == 2;
                            let offset = ((self.log_n[j] + (i << BITRES)) >> 1)
                                - if two_phase {
                                    QTHETA_OFFSET_TWOPHASE
                                } else {
                                    QTHETA_OFFSET
                                };
                            let ndof = 2 * n - 1 - i32::from(two_phase);
                            let p = if two_phase { 512 } else { 487 };
                            let num = p * (max_bits + ndof * offset);
                            let den = (ndof << 9) - p;
                            max_bits +=
                                ((num + (den >> 1)) / den).min(if two_phase { 64 } else { 61 });
                        }
                        let ndof = c * n + i32::from(c == 2 && n > 2);
                        let mut offset = ((self.log_n[j] + (i << BITRES)) >> 1) - FINE_OFFSET;
                        if n == 2 {
                            offset += 1 << BITRES >> 2;
                        }
                        let num = max_bits + ndof * offset;
                        let den = (ndof - 1) << BITRES;
                        max_bits += (c * ((num + (den >> 1)) / den).min(MAX_FINE_BITS)) << BITRES;
                    }
                    let cap = 4 * max_bits / (c * (w << i)) - 64;
                    caps[at] = u8::try_from(cap).expect("a cap within 0..256");
                    at += 1;
                }
            }
        }
        caps
    }

    /// The cost row of band `band` split to depth `lm` (-1 to 3).
    pub fn cache(&self, band: usize, lm: i32) -> &[u8] {
        let start = self.cache_index[(lm + 1) as usize * BANDS + band];
        &self.cache_bits[start as usize..]
    }

    /// The cap of a band for a frame of `1 << lm` blocks and `channels` channels (the
    /// decoder's `init_caps`), in eighths of a bit.
    pub fn cap(&self, band: usize, lm: usize, channels: usize) -> i32 {
        let n = ((EBANDS[band + 1] - EBANDS[band]) << lm) as i32;
        ((i32::from(self.caps[BANDS * (2 * lm + channels - 1) + band]) + 64) * channels as i32 * n)
            >> 2
    }

    /// The pseudo-pulse count whose cost is nearest `bits` (`celt/rate.h`, `bits2pulses`).
    pub fn bits_to_pulses(&self, band: usize, lm: i32, bits: i32) -> i32 {
        let cache = self.cache(band, lm);
        let (mut lo, mut hi) = (0usize, cache[0] as usize);
        let bits = bits - 1;
        for _ in 0..LOG_MAX_PSEUDO {
            let mid = (lo + hi + 1) >> 1;
            if i32::from(cache[mid]) >= bits {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let below = if lo == 0 { -1 } else { i32::from(cache[lo]) };
        if bits - below <= i32::from(cache[hi]) - bits {
            lo as i32
        } else {
            hi as i32
        }
    }

    /// What `q` pseudo-pulses cost in band `band` at depth `lm`, in eighths of a bit.
    pub fn pulses_to_bits(&self, band: usize, lm: i32, q: i32) -> i32 {
        if q == 0 {
            0
        } else {
            i32::from(self.cache(band, lm)[q as usize]) + 1
        }
    }

    /// The largest cost in the row (`cache[cache[0]]`), which decides whether a band splits.
    pub fn max_cost(&self, band: usize, lm: i32) -> i32 {
        let cache = self.cache(band, lm);
        i32::from(cache[cache[0] as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_widths_in_log_units_match_the_reference() {
        // `logN400` of the reference's static mode.
        let expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36,
        ];
        assert_eq!(mode().log_n, expected);
    }

    #[test]
    fn the_pulse_cache_matches_the_reference() {
        let m = mode();
        // The reference's static mode: 392 entries, `cache_index50` and `cache_caps50`.
        assert_eq!(m.cache_bits.len(), 392);
        let index_rows: [[i16; BANDS]; 2] = [
            [
                -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200, 222,
            ],
            [
                240, 240, 240, 240, 240, 240, 240, 240, 305, 305, 305, 305, 343, 343, 343, 351,
                351, 370, 376, 382, 387,
            ],
        ];
        assert_eq!(m.cache_index[..BANDS], index_rows[0]);
        assert_eq!(m.cache_index[4 * BANDS..], index_rows[1]);
        let caps_first: [u8; BANDS] = [
            224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178,
            168, 134, 61, 37,
        ];
        let caps_last: [u8; BANDS] = [
            204, 204, 204, 204, 204, 204, 204, 204, 201, 201, 201, 201, 198, 198, 198, 187, 187,
            175, 140, 66, 40,
        ];
        assert_eq!(m.caps[..BANDS], caps_first);
        assert_eq!(m.caps[7 * BANDS..], caps_last);
    }

    #[test]
    fn pseudo_pulses_grow_by_octaves() {
        assert_eq!(
            (0..10).map(pulses_of).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(pulses_of(16), 16);
        assert_eq!(pulses_of(40), 128);
    }

    #[test]
    fn the_window_is_power_complementary() {
        let w = &mode().window;
        for i in 0..OVERLAP {
            let sum = w[i] * w[i] + w[OVERLAP - 1 - i] * w[OVERLAP - 1 - i];
            assert!((sum - 1.0).abs() < 1e-6, "{i}: {sum}");
        }
    }
}
