//! Small helpers shared by the parsers.

/// Decode a hexadecimal string. Returns `None` for odd length or non-hex characters.
pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.as_bytes();
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let nibble = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    s.chunks(2)
        .map(|p| Some((nibble(p[0])? << 4) | nibble(p[1])?))
        .collect()
}

/// Encode bytes as lowercase hexadecimal.
pub fn hex_encode(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}

pub fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

pub fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

pub fn u64_at(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

/// Copy `N` bytes starting at `off`.
pub fn array_at<const N: usize>(b: &[u8], off: usize) -> [u8; N] {
    let mut a = [0u8; N];
    a.copy_from_slice(&b[off..off + N]);
    a
}

/// Round `v` up to a multiple of `align`, which must be a power of two.
pub fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        let bytes = [0x00, 0x01, 0xab, 0xff, 0x7e];
        assert_eq!(hex_encode(&bytes), "0001abff7e");
        assert_eq!(hex_decode("0001abff7e").unwrap(), bytes);
        assert_eq!(hex_decode("0001ABFF7E").unwrap(), bytes);
    }

    #[test]
    fn hex_rejects_bad_input() {
        assert!(hex_decode("abc").is_none());
        assert!(hex_decode("zz").is_none());
        assert!(hex_decode("0x12").is_none());
        assert_eq!(hex_decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn little_endian_readers() {
        let b = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(u16_at(&b, 1), 0x0302);
        assert_eq!(u32_at(&b, 0), 0x0403_0201);
        assert_eq!(u64_at(&b, 1), 0x0908_0706_0504_0302);
        assert_eq!(array_at::<3>(&b, 2), [3, 4, 5]);
    }

    #[test]
    fn alignment() {
        assert_eq!(align_up(0, 0x200), 0);
        assert_eq!(align_up(1, 0x200), 0x200);
        assert_eq!(align_up(0x200, 0x200), 0x200);
        assert_eq!(align_up(0x201, 0x200), 0x400);
    }
}
