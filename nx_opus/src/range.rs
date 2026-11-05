//! The range decoder (RFC 6716 section 4.1).
//!
//! The decoder keeps the pair (`val`, `rng`): `val` is the distance from the top of the current
//! range down to the coded value, less one, and `rng` the size of the range. Symbols are read
//! from the front of the frame; "raw bits" (used by CELT) are read from its back, least
//! significant bit first. The two may overlap (section 4.1.2.1), and past the end of the frame
//! every byte reads as zero. All arithmetic is bit-exact, as the encoder's must match it.

/// The number of bits the range is kept above (section 4.1.2.1: renormalize until rng > 2^23).
const RANGE_FLOOR: u32 = 1 << 23;

/// The number of bits needed to store `n` (section 1.1.10, `ilog`): 0 for 0.
pub fn ilog(n: u32) -> u32 {
    32 - n.leading_zeros()
}

#[derive(Clone, Debug)]
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    /// The next byte the range coder reads, from the front.
    front: usize,
    /// How many bytes the raw bits have read, from the back.
    back: usize,
    /// Raw bits read from the back and not used yet, least significant first.
    window: u32,
    window_bits: u32,
    /// The low bit of the last byte read from the front, the high bit of the next symbol byte.
    leftover: u32,
    /// Whole bits read so far, including those buffered in the range (section 4.1.6).
    total_bits: u32,
    rng: u32,
    val: u32,
    /// `rng / ft` from the last `decode`, which `update` uses.
    ext: u32,
    /// A value outside its range was decoded: the frame is corrupt (section 4.1.5).
    corrupt: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Start decoding a frame (section 4.1.1).
    pub fn new(buf: &'a [u8]) -> RangeDecoder<'a> {
        let b0 = u32::from(buf.first().copied().unwrap_or(0));
        let mut d = RangeDecoder {
            buf,
            front: 1,
            back: 0,
            window: 0,
            window_bits: 0,
            leftover: b0 & 1,
            // 9 before the first renormalization, which reads three more bytes: 33 after it.
            total_bits: 9,
            rng: 128,
            val: 127 - (b0 >> 1),
            ext: 0,
            corrupt: false,
        };
        d.normalize();
        d
    }

    fn next_front(&mut self) -> u32 {
        let b = self.buf.get(self.front).copied().unwrap_or(0);
        self.front += 1;
        u32::from(b)
    }

    fn next_back(&mut self) -> u32 {
        if self.back < self.buf.len() {
            self.back += 1;
            u32::from(self.buf[self.buf.len() - self.back])
        } else {
            0
        }
    }

    /// Renormalization (section 4.1.2.1): read bytes until the range is above 2^23.
    fn normalize(&mut self) {
        while self.rng <= RANGE_FLOOR {
            self.total_bits += 8;
            self.rng <<= 8;
            let b = self.next_front();
            let sym = (self.leftover << 7) | (b >> 1);
            self.leftover = b & 1;
            self.val = ((self.val << 8).wrapping_add(255 - sym)) & 0x7FFF_FFFF;
        }
    }

    /// The first step of decoding a symbol of a context of total `ft` (section 4.1.2): a value
    /// that lies within the range of the coded symbol.
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// `decode` with `ft = 1 << bits` (section 4.1.3.1).
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        let ft = 1 << bits;
        ft - (s + 1).min(ft)
    }

    /// The second step: the symbol `[fl, fh)` of total `ft` was decoded (section 4.1.2).
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext * (ft - fh);
        self.val -= s;
        self.rng = if fl > 0 {
            self.ext * (fh - fl)
        } else {
            self.rng - s
        };
        self.normalize();
    }

    /// One binary symbol whose "1" has the probability `1 / 2^logp` (section 4.1.3.2).
    pub fn bit_logp(&mut self, logp: u32) -> bool {
        let s = self.rng >> logp;
        let one = self.val < s;
        if one {
            self.rng = s;
        } else {
            self.val -= s;
            self.rng -= s;
        }
        self.normalize();
        one
    }

    /// A symbol of an inverse-CDF table of `1 << ftb` (section 4.1.3.3): `icdf[k]` is
    /// `(1 << ftb) - fh[k]`, the last entry 0.
    pub fn icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let r = self.rng >> ftb;
        let (mut high, mut low) = (self.rng, 0u32);
        for (k, &v) in icdf.iter().enumerate() {
            low = r * u32::from(v);
            if self.val >= low {
                self.val -= low;
                self.rng = high - low;
                self.normalize();
                return k;
            }
            high = low;
        }
        // A well-formed table ends with 0, which `val >= 0` always takes.
        let _ = low;
        unreachable!("an icdf table that does not end with 0")
    }

    /// `bits` raw bits from the back of the frame (section 4.1.4), at most 25.
    pub fn bits(&mut self, bits: u32) -> u32 {
        debug_assert!(bits <= 25);
        if bits == 0 {
            return 0;
        }
        while self.window_bits < bits {
            self.window |= self.next_back() << self.window_bits;
            self.window_bits += 8;
        }
        let value = self.window & ((1u32 << bits) - 1);
        self.window = if bits == 32 { 0 } else { self.window >> bits };
        self.window_bits -= bits;
        self.total_bits += bits;
        value
    }

    /// One of `ft` equiprobable values, 0 to `ft - 1` (section 4.1.5). A value past the range
    /// marks the frame corrupt and reads as `ft - 1`.
    pub fn uint(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 1);
        let top = ft - 1;
        let ftb = ilog(top);
        if ftb > 8 {
            let extra = ftb - 8;
            let ft1 = (top >> extra) + 1;
            let s = self.decode(ft1);
            self.update(s, s + 1, ft1);
            let t = (s << extra) | self.bits(extra);
            if t <= top {
                t
            } else {
                self.corrupt = true;
                top
            }
        } else {
            let s = self.decode(ft);
            self.update(s, s + 1, ft);
            s
        }
    }

    /// Whole bits used so far, an upper bound (section 4.1.6.1).
    pub fn tell(&self) -> u32 {
        self.total_bits - ilog(self.rng)
    }

    /// Bits used so far in eighths of a bit, an upper bound (section 4.1.6.2).
    pub fn tell_frac(&self) -> u32 {
        let mut lg = ilog(self.rng);
        let mut r = self.rng >> (lg - 16);
        for _ in 0..3 {
            r = (r * r) >> 15;
            let bit = r >> 16;
            lg = 2 * lg + bit;
            if bit != 0 {
                r >>= 1;
            }
        }
        self.total_bits * 8 - lg
    }

    /// Take `bytes` off the end of the frame (section 4.5.1.2: a redundant CELT frame at the end
    /// of a SILK or hybrid frame). Raw bits are read from the new end; past it everything reads
    /// as zero, as past the end of a frame.
    pub fn shrink(&mut self, bytes: usize) {
        debug_assert!(
            self.back == 0,
            "raw bits were read before the frame was shortened"
        );
        self.buf = &self.buf[..self.buf.len().saturating_sub(bytes)];
    }

    /// Count every remaining bit of the frame as used (a CELT frame flagged silent, section
    /// 4.3: the decoder acts as though it had read them all).
    pub fn skip_to_end(&mut self) {
        // `tell` becomes exactly the frame's size, down as well as up.
        self.total_bits = self.buf.len() as u32 * 8 + ilog(self.rng);
    }

    /// The size of the frame, in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The state of the range when the frame is done, which encoders report so that a decoder
    /// can be checked (the test vectors carry it for every packet).
    pub fn range(&self) -> u32 {
        self.rng
    }

    /// Whether a value outside its range was decoded.
    pub fn is_corrupt(&self) -> bool {
        self.corrupt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_decoder_has_used_one_bit() {
        // Section 4.1.6.1: before any symbol, one bit (the encoder's termination bit) is used.
        for frame in [&[][..], &[0x00], &[0xFF, 0x12, 0x34], &[0x80; 40]] {
            let d = RangeDecoder::new(frame);
            assert_eq!(d.tell(), 1, "{frame:x?}");
            assert!(d.rng > RANGE_FLOOR);
            assert!(d.tell_frac() <= 8 && d.tell_frac() > 0);
        }
    }

    #[test]
    fn tell_is_the_ceiling_of_tell_frac() {
        let data: Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8).collect();
        let mut d = RangeDecoder::new(&data);
        for i in 0..200 {
            match i % 4 {
                0 => {
                    d.bit_logp(1 + i % 15);
                }
                1 => {
                    d.uint(3 + i * 7);
                }
                2 => {
                    d.icdf(&[200, 100, 50, 0], 8);
                }
                _ => {
                    d.bits(1 + i % 9);
                }
            }
            assert_eq!(d.tell(), d.tell_frac().div_ceil(8), "after {i} symbols");
            assert!(d.rng > RANGE_FLOOR);
        }
    }

    #[test]
    fn raw_bits_come_from_the_back_least_significant_first() {
        // The last byte 0b1011_0010: its low bits first, then the byte before.
        let data = [0x00, 0x00, 0xA5, 0b1011_0010];
        let mut d = RangeDecoder::new(&data);
        assert_eq!(d.bits(2), 0b10);
        assert_eq!(d.bits(3), 0b100);
        assert_eq!(d.bits(3), 0b101);
        assert_eq!(d.bits(8), 0xA5);
        // Past the start, zeros.
        assert_eq!(d.bits(16), 0);
        assert_eq!(d.bits(5), 0);
    }

    #[test]
    fn decode_and_update_agree_with_the_special_methods() {
        // ec_decode_bin and ec_dec_bit_logp are exact shortcuts of ec_decode with a power of
        // two (section 4.1.3); the same input must give the same symbols and the same state.
        let data: Vec<u8> = (0..32u32).map(|i| (i * 91 + 7) as u8).collect();
        let (mut a, mut b) = (RangeDecoder::new(&data), RangeDecoder::new(&data));
        for i in 0..60u32 {
            let logp = 1 + i % 8;
            let ft = 1u32 << logp;
            let one = a.bit_logp(logp);
            let fs = b.decode(ft);
            let b_one = fs >= ft - 1;
            if b_one {
                b.update(ft - 1, ft, ft);
            } else {
                b.update(0, ft - 1, ft);
            }
            assert_eq!(one, b_one, "symbol {i}");
            assert_eq!((a.val, a.rng), (b.val, b.rng), "state after {i}");
            let s = a.decode_bin(5);
            let t = b.decode(32);
            assert_eq!(s, t);
            a.update(s, s + 1, 32);
            b.update(t, t + 1, 32);
        }
    }

    #[test]
    fn the_icdf_shortcut_matches_decode_with_the_frequencies() {
        // {3, 2, 2, 1}/8 as an icdf of ftb 3: 8 - fh = {5, 3, 1, 0}.
        let icdf = [5u8, 3, 1, 0];
        let fl = [0u32, 3, 5, 7];
        let fh = [3u32, 5, 7, 8];
        let data: Vec<u8> = (0..48u32).map(|i| (i * 53 + 29) as u8).collect();
        let (mut a, mut b) = (RangeDecoder::new(&data), RangeDecoder::new(&data));
        for i in 0..100 {
            let k = a.icdf(&icdf, 3);
            let fs = b.decode(8);
            let j = (0..4).find(|&j| fl[j] <= fs && fs < fh[j]).unwrap();
            b.update(fl[j], fh[j], 8);
            assert_eq!(k, j, "symbol {i}");
            assert_eq!((a.val, a.rng), (b.val, b.rng));
        }
    }

    #[test]
    fn a_uniform_integer_stays_in_its_range() {
        let data: Vec<u8> = (0..64u32).map(|i| (i * 13 + 200) as u8).collect();
        let mut d = RangeDecoder::new(&data);
        for ft in [2u32, 3, 7, 255, 256, 257, 1000, 65_536, 1 << 20, u32::MAX] {
            let v = d.uint(ft);
            assert!(v < ft, "{v} of {ft}");
        }
    }

    #[test]
    fn ilog_counts_bits() {
        assert_eq!([0, 1, 2, 3, 4, 255, 256].map(ilog), [0, 1, 2, 2, 3, 8, 9]);
        assert_eq!(ilog(u32::MAX), 32);
    }
}
