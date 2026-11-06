//! BC7 blocks of mode 6 decoded on the host, as the device's sampler decodes them: what the
//! device's encoder (`shaders/bc7_encode.comp`, A-291) is tested with.

/// The weights of BC7's four-bit indices.
const WEIGHTS: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// The sixteen RGBA texels of a BC7 block of mode 6, in rows; `None` for a block of another mode.
pub fn decode_mode6(block: &[u8; 16]) -> Option<[[u8; 4]; 16]> {
    let bits = u128::from_le_bytes(*block);
    if bits & 0x7F != 0x40 {
        return None;
    }
    let mut at = 7;
    let mut take = |n: u32| {
        let v = (bits >> at) as u32 & ((1 << n) - 1);
        at += n;
        v
    };
    let (mut q0, mut q1) = ([0u32; 4], [0u32; 4]);
    for (a, b) in q0.iter_mut().zip(q1.iter_mut()) {
        *a = take(7);
        *b = take(7);
    }
    let (p0, p1) = (take(1), take(1));
    let mut out = [[0u8; 4]; 16];
    for (i, texel) in out.iter_mut().enumerate() {
        let w = WEIGHTS[take(if i == 0 { 3 } else { 4 }) as usize];
        for ((t, a), b) in texel.iter_mut().zip(q0).zip(q1) {
            let (e0, e1) = (a << 1 | p0, b << 1 | p1);
            *t = (((64 - w) * e0 + w * e1 + 32) >> 6) as u8;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mode_6_block_of_one_color_decodes_to_it() {
        // Mode 6, both endpoints (100, 50, 25, 127) << 1 | 1, every index 0.
        let mut bits: u128 = 0x40;
        let mut at = 7;
        let mut put = |v: u128, n: u32| {
            bits |= v << at;
            at += n;
        };
        for c in [100u128, 50, 25, 127] {
            put(c, 7);
            put(c, 7);
        }
        put(1, 1);
        put(1, 1);
        let texels = decode_mode6(&bits.to_le_bytes()).unwrap();
        assert!(texels.iter().all(|t| *t == [201, 101, 51, 255]));
        assert_eq!(decode_mode6(&[0; 16]), None);
    }
}
