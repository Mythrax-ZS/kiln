//! LZ4 block decompression, the compression used by NSO segments.
//!
//! This is the raw block format (no frame header): a series of sequences, each a token byte whose
//! high nibble is the literal count and low nibble the match length minus four (a nibble of 15 is
//! extended by following bytes that add up until one is not 255), the literals, then a two-byte
//! little-endian match offset. The final sequence has literals only.

use crate::{Error, Result};

fn read_len(src: &[u8], i: &mut usize, mut len: usize) -> Result<usize> {
    if len == 15 {
        loop {
            let b = *src
                .get(*i)
                .ok_or_else(|| Error::format("LZ4 block", "length runs past the input"))?;
            *i += 1;
            len = len
                .checked_add(usize::from(b))
                .ok_or_else(|| Error::format("LZ4 block", "length overflows"))?;
            if b != 255 {
                break;
            }
        }
    }
    Ok(len)
}

/// Decompress `src`, which must expand to exactly `out_len` bytes.
pub fn decompress(src: &[u8], out_len: usize) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(out_len);
    let mut i = 0usize;
    while i < src.len() {
        let token = src[i];
        i += 1;
        let lits = read_len(src, &mut i, usize::from(token >> 4))?;
        let lit_end = i
            .checked_add(lits)
            .filter(|&e| e <= src.len())
            .ok_or_else(|| Error::format("LZ4 block", "literals run past the input"))?;
        if out.len() + lits > out_len {
            return Err(Error::format(
                "LZ4 block",
                "output exceeds the declared size",
            ));
        }
        out.extend_from_slice(&src[i..lit_end]);
        i = lit_end;
        if i == src.len() {
            break;
        }
        if i + 2 > src.len() {
            return Err(Error::format(
                "LZ4 block",
                "match offset runs past the input",
            ));
        }
        let offset = usize::from(u16::from_le_bytes([src[i], src[i + 1]]));
        i += 2;
        if offset == 0 || offset > out.len() {
            return Err(Error::format(
                "LZ4 block",
                "match offset points before the output",
            ));
        }
        let len = read_len(src, &mut i, usize::from(token & 15))? + 4;
        if out.len() + len > out_len {
            return Err(Error::format(
                "LZ4 block",
                "output exceeds the declared size",
            ));
        }
        let start = out.len() - offset;
        if offset >= len {
            out.extend_from_within(start..start + len);
        } else {
            for k in 0..len {
                let b = out[start + k];
                out.push(b);
            }
        }
    }
    if out.len() != out_len {
        return Err(Error::format(
            "LZ4 block",
            format!("expanded to {} bytes, expected {out_len}", out.len()),
        ));
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod testing {
    fn push_len(out: &mut Vec<u8>, mut rem: usize) {
        while rem >= 255 {
            out.push(255);
            rem -= 255;
        }
        out.push(rem as u8);
    }

    fn emit(out: &mut Vec<u8>, literals: &[u8], matched: Option<(usize, usize)>) {
        let lit_nib = literals.len().min(15) as u8;
        let match_nib = matched.map_or(0, |(_, len)| (len - 4).min(15) as u8);
        out.push(lit_nib << 4 | match_nib);
        if literals.len() >= 15 {
            push_len(out, literals.len() - 15);
        }
        out.extend_from_slice(literals);
        if let Some((offset, len)) = matched {
            out.extend_from_slice(&(offset as u16).to_le_bytes());
            if len - 4 >= 15 {
                push_len(out, len - 4 - 15);
            }
        }
    }

    /// Greedy LZ4 block compressor for tests. Follows the end-of-block rules (the last five bytes
    /// are literals and no match starts in the last twelve).
    pub fn compress(data: &[u8]) -> Vec<u8> {
        let n = data.len();
        let mut out = Vec::new();
        let mut table = vec![usize::MAX; 1 << 12];
        let (mut anchor, mut i) = (0usize, 0usize);
        while i + 12 <= n {
            let seq = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            let h = (seq.wrapping_mul(2_654_435_761) >> 20) as usize;
            let cand = table[h];
            table[h] = i;
            if cand != usize::MAX && i - cand <= 65535 && data[cand..cand + 4] == data[i..i + 4] {
                let mut len = 4;
                while i + len < n - 5 && data[cand + len] == data[i + len] {
                    len += 1;
                }
                emit(&mut out, &data[anchor..i], Some((i - cand, len)));
                i += len;
                anchor = i;
            } else {
                i += 1;
            }
        }
        emit(&mut out, &data[anchor..], None);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testing::compress;
    use super::*;
    use crate::loader::lz4_vectors::{lcg, F_HC};
    use crate::util::hex_decode;
    use nx_runtime::testing::Rng;

    fn h(s: &str) -> Vec<u8> {
        hex_decode(s).unwrap()
    }

    /// Blocks produced by the reference implementation (the `lz4` Python package).
    #[test]
    fn decodes_reference_blocks() {
        let a = b"hello hello hello hello hello hello hello";
        assert_eq!(
            decompress(&h("6f68656c6c6f2006000b5068656c6c6f"), a.len()).unwrap(),
            a
        );

        let b = [
            [
                b"0123456789ABCDEF".repeat(20),
                b"tail-bytes-not-repeated-xyz".to_vec(),
            ]
            .concat(),
            b"0123456789ABCDEF".repeat(20),
        ]
        .concat();
        for hex in [
            "ff01303132333435363738394142434445461000ff1eff0c7461696c2d62797465732d6e6f742d72657065617465642d78797a4b01ff1e073001504243444546",
            "ff01303132333435363738394142434445461000ff1eff0c7461696c2d62797465732d6e6f742d72657065617465642d78797a5b01ff29504243444546",
        ] {
            assert_eq!(decompress(&h(hex), b.len()).unwrap(), b);
        }

        let c: Vec<u8> = (0..40).collect();
        assert_eq!(
            decompress(&h("f019000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f2021222324252627"), 40).unwrap(),
            c
        );
        // run of one byte: overlapping match with offset 1
        assert_eq!(
            decompress(&h("1f6101004b506161616161"), 100).unwrap(),
            vec![b'a'; 100]
        );
        assert_eq!(decompress(&h("00"), 0).unwrap(), Vec::<u8>::new());
        assert_eq!(decompress(&h("1078"), 1).unwrap(), b"x");
        // 6000 bytes: a 3000 byte random-looking run followed by an exact repeat (offset 3000)
        let f = [lcg(3000), lcg(3000)].concat();
        assert_eq!(decompress(&h(F_HC), 6000).unwrap(), f);
    }

    #[test]
    fn long_literal_runs_use_extension_bytes() {
        let e: Vec<u8> = (0..600).map(|i| ((i * 131 + 7) % 256) as u8).collect();
        let packed = compress(&e);
        assert_eq!(decompress(&packed, e.len()).unwrap(), e);
    }

    #[test]
    fn test_compressor_round_trips_random_shapes() {
        let mut rng = Rng::new(99);
        for round in 0..200 {
            let len = rng.below(3000) as usize;
            let mut data = vec![0u8; len];
            let mode = round % 4;
            for (i, b) in data.iter_mut().enumerate() {
                *b = match mode {
                    0 => rng.next_u32() as u8,
                    1 => (i % 7) as u8,
                    2 => {
                        if rng.chance(20) {
                            rng.next_u32() as u8
                        } else {
                            b'z'
                        }
                    }
                    _ => ((i / 3) % 251) as u8,
                };
            }
            let packed = compress(&data);
            assert_eq!(
                decompress(&packed, len).unwrap(),
                data,
                "round {round} len {len}"
            );
        }
    }

    #[test]
    fn rejects_malformed_blocks() {
        let good = h("6f68656c6c6f2006000b5068656c6c6f");
        let n = 41;
        assert!(
            decompress(&good, n - 1).is_err(),
            "output larger than declared"
        );
        assert!(
            decompress(&good, n + 1).is_err(),
            "output smaller than declared"
        );
        assert!(
            decompress(&good[..good.len() - 1], n).is_err(),
            "truncated literals"
        );
        assert!(decompress(&good[..7], n).is_err(), "truncated offset");
        // zero offset
        assert!(decompress(&h("1f6100004b506161616161"), 100).is_err());
        // offset beyond what has been produced
        assert!(decompress(&h("1f6101094b506161616161"), 100).is_err());
        // length extension runs off the end of the input
        assert!(decompress(&h("f0ff"), 300).is_err());
        // literal count claims more than the input holds
        assert!(decompress(&h("50616263"), 5).is_err());
        // empty input cannot produce output
        assert!(decompress(&[], 1).is_err());
        assert_eq!(decompress(&[], 0).unwrap(), Vec::<u8>::new());
    }
}
