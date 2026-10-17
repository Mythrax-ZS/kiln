//! The boolean decoder (VP9 specification section 9.2) and the plain bit reader of the
//! uncompressed header (section 9.1).
//!
//! The specification reads one bit at a time into an 8-bit value. This reader keeps up to 64 bits
//! ahead in a register whose top byte is that value, which gives the same results: comparing the
//! register with the split shifted to the top byte is comparing the value with the split, since
//! the bits below it are less than one unit of the top byte.

use super::Error;

/// The arithmetic decoder of a tile or of the compressed header.
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    /// The next byte of `data` to load (it can pass the end: what lies past it reads as zeros).
    pos: usize,
    /// The bits read ahead, from the top: the specification's `BoolValue` in the top byte, then
    /// `count` more bits.
    value: u64,
    count: i32,
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    /// `init_bool( sz )` over all of `data`: the first byte, and the marker bit, which must be 0.
    pub fn new(data: &'a [u8]) -> Result<BoolDecoder<'a>, Error> {
        if data.is_empty() {
            return Err(Error::Stream("a boolean decoder over no bytes"));
        }
        let mut d = BoolDecoder {
            data,
            pos: 0,
            value: 0,
            count: -8,
            range: 255,
        };
        d.fill();
        if d.read(128) {
            return Err(Error::Stream("the marker bit of a boolean decoder is 1"));
        }
        Ok(d)
    }

    /// Load whole bytes below the bits held while they fit in the register.
    fn fill(&mut self) {
        let mut shift = 48 - self.count;
        while shift >= 0 {
            let byte = self.data.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.value |= u64::from(byte) << shift;
            self.count += 8;
            shift -= 8;
        }
    }

    /// `read_bool( p )`: a bool that is 0 with probability `p` / 256.
    #[inline]
    pub fn read(&mut self, p: u8) -> bool {
        let split = 1 + (((self.range - 1) * u32::from(p)) >> 8);
        if self.count < 0 {
            self.fill();
        }
        let big = u64::from(split) << 56;
        let bit = if self.value >= big {
            self.range -= split;
            self.value -= big;
            true
        } else {
            self.range = split;
            false
        };
        let shift = (self.range as u8).leading_zeros();
        self.range <<= shift;
        self.value <<= shift;
        self.count -= shift as i32;
        bit
    }

    /// `read_literal( n )`: `n` bits of probability one half, the first the most significant.
    pub fn literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = (x << 1) | u32::from(self.read(128));
        }
        x
    }

    /// Decode a value with a tree (section 9.3.3): `prob(node)` gives the probability of each
    /// node visited.
    #[inline]
    pub fn tree(&mut self, tree: &[i8], mut prob: impl FnMut(usize) -> u8) -> u8 {
        let mut n = 0i8;
        loop {
            let at = n as usize;
            n = tree[at + usize::from(self.read(prob(at >> 1)))];
            if n <= 0 {
                return (-n) as u8;
            }
        }
    }

    /// `exit_bool( )`: the bits the decoder did not use must all be zero, and it must not have
    /// used bits past the end. A decoder that follows a stream wrongly almost never ends exactly
    /// there, which makes this the cheapest check that a tile was decoded as its encoder meant.
    pub fn finish(&self) -> Result<(), Error> {
        let used = self.pos as i64 * 8 - i64::from(self.count);
        if used > self.data.len() as i64 * 8 {
            return Err(Error::Stream(
                "a boolean decoder read past the end of its data",
            ));
        }
        let ahead = self.count > 0 && (self.value << 8) != 0;
        let rest = self
            .data
            .get(self.pos.min(self.data.len())..)
            .is_some_and(|r| r.iter().any(|&b| b != 0));
        if ahead || rest {
            return Err(Error::Stream(
                "the padding after a boolean-coded part is not zero",
            ));
        }
        Ok(())
    }
}

/// The reader of the uncompressed header: bits from the most significant of each byte.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> BitReader<'a> {
        BitReader { data, pos: 0 }
    }

    /// `f( n )`: an unsigned value of `n` bits.
    pub fn f(&mut self, n: u32) -> Result<u32, Error> {
        let mut x = 0u32;
        for _ in 0..n {
            let byte = *self
                .data
                .get(self.pos >> 3)
                .ok_or(Error::Stream("the uncompressed header is cut short"))?;
            x = (x << 1) | u32::from((byte >> (7 - (self.pos & 7))) & 1);
            self.pos += 1;
        }
        Ok(x)
    }

    pub fn bit(&mut self) -> Result<bool, Error> {
        Ok(self.f(1)? == 1)
    }

    /// `s( n )`: a value of `n` bits, then its sign.
    pub fn s(&mut self, n: u32) -> Result<i32, Error> {
        let v = self.f(n)? as i32;
        Ok(if self.bit()? { -v } else { v })
    }

    /// The position in bits.
    pub fn position(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An encoder for the tests: the counterpart of the decoder (libvpx's writer, as the
    /// specification's decoder implies it).
    pub struct BoolEncoder {
        low: u64,
        range: u32,
        count: i32,
        out: Vec<u8>,
    }

    impl BoolEncoder {
        pub fn new() -> BoolEncoder {
            let mut e = BoolEncoder {
                low: 0,
                range: 255,
                count: -24,
                out: Vec::new(),
            };
            e.write(false, 128); // the marker bit
            e
        }

        pub fn write(&mut self, bit: bool, p: u8) {
            let split = 1 + (((self.range - 1) * u32::from(p)) >> 8);
            let (mut range, mut low) = (split, self.low);
            if bit {
                low += u64::from(split);
                range = self.range - split;
            }
            let mut shift = (range as u8).leading_zeros() as i32;
            range <<= shift;
            self.count += shift;
            if self.count >= 0 {
                let offset = shift - self.count;
                if (low << (offset - 1)) & 0x8000_0000 != 0 {
                    // Carry into the bytes written.
                    let mut x = self.out.len();
                    while x > 0 && self.out[x - 1] == 0xFF {
                        self.out[x - 1] = 0;
                        x -= 1;
                    }
                    self.out[x - 1] += 1;
                }
                self.out.push(((low >> (24 - offset)) & 0xFF) as u8);
                low <<= offset;
                shift = self.count;
                low &= 0xFF_FFFF;
                self.count -= 8;
            }
            low <<= shift;
            self.range = range;
            self.low = low;
        }

        pub fn finish(mut self) -> Vec<u8> {
            for _ in 0..32 {
                self.write(false, 128);
            }
            self.out
        }
    }

    #[test]
    fn bools_written_with_their_probabilities_read_back() {
        let mut rng = 0x1234_5678u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        let bits: Vec<(bool, u8)> = (0..5000)
            .map(|_| {
                let p = (next() % 255 + 1) as u8;
                let bit = (next() % 256) as u8 >= p;
                (bit, p)
            })
            .collect();
        let mut e = BoolEncoder::new();
        for &(bit, p) in &bits {
            e.write(bit, p);
        }
        let data = e.finish();
        let mut d = BoolDecoder::new(&data).unwrap();
        for (i, &(bit, p)) in bits.iter().enumerate() {
            assert_eq!(d.read(p), bit, "bit {i}");
        }
        d.finish().unwrap();
    }

    #[test]
    fn padding_that_is_not_zero_or_reading_past_the_end_is_refused() {
        let mut e = BoolEncoder::new();
        for i in 0..100 {
            e.write(i % 3 == 0, 200);
        }
        let mut data = e.finish();
        let mut d = BoolDecoder::new(&data).unwrap();
        for i in 0..100 {
            assert_eq!(d.read(200), i % 3 == 0);
        }
        d.finish().unwrap();
        // A stray bit in the padding.
        data.push(0x01);
        let mut d = BoolDecoder::new(&data).unwrap();
        for _ in 0..100 {
            d.read(200);
        }
        assert!(d.finish().is_err());
        // Reading on past the end.
        data.pop();
        let mut d = BoolDecoder::new(&data).unwrap();
        for _ in 0..(data.len() * 8 + 64) {
            d.read(128);
        }
        assert!(d.finish().is_err());
        // The marker bit.
        assert!(BoolDecoder::new(&[0xFF, 0xFF]).is_err());
        assert!(BoolDecoder::new(&[]).is_err());
    }

    #[test]
    fn the_bit_reader_reads_from_the_top_of_each_byte() {
        let mut r = BitReader::new(&[0b1010_0000, 0xFF]);
        assert_eq!(r.f(3).unwrap(), 0b101);
        assert_eq!(r.f(5).unwrap(), 0);
        assert_eq!(r.s(4).unwrap(), -15, "15, then a sign bit of 1");
        assert!(r.f(8).is_err(), "only 3 bits are left");
    }
}
