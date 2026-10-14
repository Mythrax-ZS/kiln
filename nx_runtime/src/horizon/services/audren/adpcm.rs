//! The 4-bit ADPCM of the console's audio renderer: frames of 8 bytes that hold 14 samples
//! (docs/ASSUMPTIONS.md A-125).
//!
//! A frame starts with a header byte: the low nibble is a shift (the step of a nibble is
//! `1 << shift`) and bits 4 to 6 pick one of eight pairs of predictor coefficients; the seven bytes
//! after it hold two signed nibbles each, the high one first. A sample is the nibble times the
//! step, plus the two samples before it weighted by the pair, in fixed point with 11 fractional
//! bits.

/// Bytes of a frame.
pub const FRAME_BYTES: usize = 8;

/// Samples of a frame.
pub const FRAME_SAMPLES: usize = 14;

/// The two samples before the one being decoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct History {
    /// The most recent sample.
    pub y1: i16,
    /// The one before it.
    pub y2: i16,
}

/// The coefficients of a voice: eight pairs, as the program stores them (16 values of 16 bits).
pub type Coefficients = [i16; 16];

/// Decode one frame, advancing `history` past its last sample.
pub fn decode_frame(
    frame: &[u8],
    coefficients: &Coefficients,
    history: &mut History,
    out: &mut [i16; FRAME_SAMPLES],
) {
    let header = frame[0];
    let step = 1i32 << (header & 0x0F);
    let pair = usize::from((header >> 4) & 7);
    let (c1, c2) = (
        i32::from(coefficients[pair * 2]),
        i32::from(coefficients[pair * 2 + 1]),
    );
    for (i, slot) in out.iter_mut().enumerate() {
        let byte = frame[1 + i / 2];
        let nibble = i32::from(if i % 2 == 0 { byte >> 4 } else { byte & 0x0F });
        let nibble = if nibble >= 8 { nibble - 16 } else { nibble };
        let predicted = c1 * i32::from(history.y1) + c2 * i32::from(history.y2);
        let sample = (((nibble * step) << 11) + 0x400 + predicted) >> 11;
        let sample = sample.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        history.y2 = history.y1;
        history.y1 = sample;
        *slot = sample;
    }
}

/// Which frame holds sample `index` and where in that frame it is.
pub fn locate(index: u64) -> (u64, usize) {
    (
        index / FRAME_SAMPLES as u64,
        (index % FRAME_SAMPLES as u64) as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame with the given header and nibbles (two per byte, the first in the high half).
    fn frame(header: u8, nibbles: [u8; FRAME_SAMPLES]) -> [u8; FRAME_BYTES] {
        let mut f = [0u8; FRAME_BYTES];
        f[0] = header;
        for (i, n) in nibbles.iter().enumerate() {
            let byte = &mut f[1 + i / 2];
            if i % 2 == 0 {
                *byte |= (n & 0x0F) << 4;
            } else {
                *byte |= n & 0x0F;
            }
        }
        f
    }

    #[test]
    fn without_prediction_a_nibble_is_the_step_times_its_value() {
        let coefficients = [0i16; 16];
        // shift 3 (step 8), the nibbles 1, 2, 7, 8 (that is -8), 15 (-1), 0 and then zeros.
        let f = frame(0x03, [1, 2, 7, 8, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut history = History::default();
        let mut out = [0i16; FRAME_SAMPLES];
        decode_frame(&f, &coefficients, &mut history, &mut out);
        assert_eq!(&out[..6], &[8, 16, 56, -64, -8, 0]);
        assert_eq!(history, History { y1: 0, y2: 0 });
    }

    #[test]
    fn the_prediction_carries_the_samples_forward() {
        // Pair 0 is (2048, 0): the sample before it, unchanged (a coefficient of 1.0).
        let mut coefficients = [0i16; 16];
        coefficients[0] = 2048;
        let f = frame(0x00, [3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut history = History::default();
        let mut out = [0i16; FRAME_SAMPLES];
        decode_frame(&f, &coefficients, &mut history, &mut out);
        assert_eq!(&out[..7], &[3, 3, 3, 3, 4, 4, 4]);
        assert_eq!(history.y1, 4);
        assert_eq!(history.y2, 4);
        // The history of one frame is the start of the next.
        let g = frame(0x00, [0; FRAME_SAMPLES]);
        decode_frame(&g, &coefficients, &mut history, &mut out);
        assert_eq!(out, [4; FRAME_SAMPLES]);
    }

    #[test]
    fn the_second_coefficient_weighs_the_older_sample() {
        // Pair 1 is (0, 2048): the sample two before.
        let mut coefficients = [0i16; 16];
        coefficients[3] = 2048;
        let f = frame(0x10, [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut history = History::default();
        let mut out = [0i16; FRAME_SAMPLES];
        decode_frame(&f, &coefficients, &mut history, &mut out);
        // 5, 6, then the sample two back each time: 5, 6, 5, 6 ...
        assert_eq!(&out[..6], &[5, 6, 5, 6, 5, 6]);
    }

    #[test]
    fn samples_are_clamped_to_sixteen_bits() {
        let mut coefficients = [0i16; 16];
        coefficients[0] = 2048;
        // Step 2^15 with the nibble 7 is far past the range, and so is the negative one.
        let f = frame(0x0F, [7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut history = History::default();
        let mut out = [0i16; FRAME_SAMPLES];
        decode_frame(&f, &coefficients, &mut history, &mut out);
        assert_eq!(out[0], i16::MAX);
        assert_eq!(out[1], i16::MIN);
    }

    #[test]
    fn the_position_of_a_sample_is_found_by_the_frame() {
        assert_eq!(locate(0), (0, 0));
        assert_eq!(locate(13), (0, 13));
        assert_eq!(locate(14), (1, 0));
        assert_eq!(locate(100), (7, 2));
    }
}
