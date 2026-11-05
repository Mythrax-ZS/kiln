//! The constant tables of the CELT layer. They are data of the specification: RFC 6716 gives
//! some in its prose (tables 55 to 63) and the rest in its normative reference implementation
//! (appendix A, `celt/modes.c`, `celt/quant_bands.c`, `celt/celt.c`, `celt/rate.c`; Simplified
//! BSD licence, copyright Xiph.Org Foundation, Skype Limited, Octasic, Jean-Marc Valin, Timothy
//! B. Terriberry, CSIRO, Gregory Maxwell, Mark Borgerding and Erik de Castro Lopo).

/// The number of energy bands of the 48 kHz mode (table 55).
pub const BANDS: usize = 21;

/// The first MDCT bin of each band for a 2.5 ms frame (table 55); a frame of `1 << lm` short
/// blocks scales these by `1 << lm`.
pub const EBANDS: [usize; BANDS + 1] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// The static bit allocation (table 57), in 1/32 bit per MDCT bin, eleven quality steps.
pub const BAND_ALLOCATION: [[u8; BANDS]; 11] = [
    [0; BANDS],
    [
        90, 80, 75, 69, 63, 56, 49, 40, 34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32, 26, 20, 12, 0, 0, 0, 0, 0, 0,
    ],
    [
        118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23, 15, 4, 0, 0, 0, 0,
    ],
    [
        126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25, 17, 12, 1, 0, 0,
    ],
    [
        134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41, 35, 29, 23, 16, 10, 1,
    ],
    [
        144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57, 51, 45, 39, 33, 26, 15, 1,
    ],
    [
        152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74, 67, 61, 55, 49, 43, 36, 20, 1,
    ],
    [
        162, 155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90, 84, 77, 71, 65, 59, 53, 46, 30, 1,
    ],
    [
        172, 165, 158, 152, 143, 137, 131, 125, 118, 112, 106, 100, 94, 87, 81, 75, 69, 63, 56, 45,
        20,
    ],
    [
        200, 200, 200, 200, 200, 200, 200, 200, 198, 193, 188, 183, 178, 173, 168, 163, 158, 153,
        148, 129, 104,
    ],
];

/// The parameters of the Laplace model of the coarse energy (section 4.3.2.1), for each frame
/// size (2.5, 5, 10 and 20 ms), inter then intra prediction: for each band the probability of
/// zero and the decay, both in Q8.
pub const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

/// The mean energy of each band (section 4.3.2.1), subtracted before coding, in sixteenths of
/// the base-2 log units (the reference's float table holds these over 16: 6.4375 for 103).
const E_MEANS_Q4: [u8; BANDS] = [
    103, 100, 92, 85, 81, 77, 72, 70, 78, 75, 73, 71, 78, 74, 69, 72, 70, 74, 76, 71, 60,
];

/// The mean energy of a band, in base-2 log units.
pub fn e_mean(band: usize) -> f32 {
    f32::from(E_MEANS_Q4[band]) / 16.0
}

/// The inter-frame prediction coefficient (alpha) and the inter-band one (beta) of the coarse
/// energy for each frame size (section 4.3.2.1), in Q15.
pub const PRED_COEF: [f32; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];
pub const BETA_COEF: [f32; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];
/// Beta of an intra frame, whose alpha is 0.
pub const BETA_INTRA: f32 = 4915.0 / 32768.0;

/// The coarse energy delta when only a few bits are left (section 4.3.2.1): 0, -1, 1.
pub const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];
/// The allocation trim (table 58).
pub const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
/// The spreading decision (table 56).
pub const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];
/// The post-filter's tap set (table 56).
pub const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// The time-frequency resolution changes (tables 60 to 63): for each frame size, indexed by
/// `4 * transient + 2 * tf_select + tf_res`.
pub const TF_SELECT: [[i8; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

/// log2 of 1 to 24 in eighths of a bit, rounded up (`celt/rate.c`): what coding a choice of
/// that many intensity bands costs.
pub const LOG2_FRAC: [i32; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// The three taps of each of the post-filter's tap sets (section 4.3.7.1).
pub const COMB_GAINS: [[f32; 3]; 3] = [
    [0.306_640_63, 0.217_041_02, 0.129_638_67],
    [0.463_867_2, 0.268_066_4, 0.0],
    [0.799_804_7, 0.100_097_656, 0.0],
];

/// The de-emphasis filter's coefficient (section 4.3.7.2).
pub const PREEMPHASIS: f32 = 0.850_006_1;
