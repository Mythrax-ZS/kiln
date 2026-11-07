//! API format geometry. Hardware component identifiers follow the existing TIC decoder.

#[derive(Clone, Copy, Debug)]
pub(super) struct Format {
    pub bytes: u32,
    pub block: [u32; 2],
    pub depth: bool,
}

pub(super) fn get(code: u32) -> Option<Format> {
    let bytes = match code {
        1..=4 | 49 | 81..=82 => 1,
        5..=9 | 13..=16 | 50 | 57..=60 | 83..=84 | 87..=88 => 2,
        10..=12
        | 17..=21
        | 25..=28
        | 37..=40
        | 51..=53
        | 55..=56
        | 61..=64
        | 85..=86
        | 89..=90
        | 93..=94
        | 99..=100
        | 105..=112 => 4,
        22..=24 | 29..=33 | 41..=45 | 54 | 91..=92 | 95..=96 | 101..=102 | 113..=117 => 8,
        34..=36 | 97..=98 => 12,
        46..=48 | 103..=104 | 118..=120 => 16,
        65..=80 => {
            let bytes = if matches!(code, 65 | 66 | 69 | 70 | 73 | 74) {
                8
            } else {
                16
            };
            return Some(Format {
                bytes,
                block: [4, 4],
                depth: false,
            });
        }
        121..=148 => {
            let block = [
                [4, 4],
                [5, 4],
                [5, 5],
                [6, 5],
                [6, 6],
                [8, 5],
                [8, 6],
                [8, 8],
                [10, 5],
                [10, 6],
                [10, 8],
                [10, 10],
                [12, 10],
                [12, 12],
            ][((code - 121) % 14) as usize];
            return Some(Format {
                bytes: 16,
                block,
                depth: false,
            });
        }
        _ => return None,
    };
    Some(Format {
        bytes,
        block: [1, 1],
        depth: (49..=54).contains(&code),
    })
}

pub(super) struct Encoding {
    pub component: u32,
    pub types: [u32; 4],
    pub sources: [u32; 4],
    pub srgb: bool,
}

pub(super) fn encoding(code: u32, stencil: bool) -> Option<Encoding> {
    let (component, data, channels, srgb) = match code {
        1..=4 => (0x1D, [2, 1, 4, 3][(code - 1) as usize], 1, false),
        5..=9 => (0x1B, [7, 2, 1, 4, 3][(code - 5) as usize], 1, false),
        10..=12 => (0x0F, [7, 4, 3][(code - 10) as usize], 1, false),
        13..=16 => (0x18, [2, 1, 4, 3][(code - 13) as usize], 2, false),
        17..=21 => (0x0C, [7, 2, 1, 4, 3][(code - 17) as usize], 2, false),
        22..=24 => (0x04, [7, 4, 3][(code - 22) as usize], 2, false),
        34..=36 => (0x02, [7, 4, 3][(code - 34) as usize], 3, false),
        37..=40 => (0x08, [2, 1, 4, 3][(code - 37) as usize], 4, false),
        41..=45 => (0x03, [7, 2, 1, 4, 3][(code - 41) as usize], 4, false),
        46..=48 => (0x01, [7, 4, 3][(code - 46) as usize], 4, false),
        49 => (0x1D, 4, 1, false),
        50 => (0x3A, 2, 1, false),
        51 => (0x2A, 2, 1, false),
        52 => (0x2F, 7, 1, false),
        53 => {
            return Some(Encoding {
                component: 0x2B,
                types: [2, 4, 2, 2],
                sources: [
                    if stencil { 3 } else { 2 },
                    0,
                    0,
                    if stencil { 6 } else { 7 },
                ],
                srgb: false,
            });
        }
        54 => {
            return Some(Encoding {
                component: 0x30,
                types: [7, 4, 2, 2],
                sources: [
                    if stencil { 3 } else { 2 },
                    0,
                    0,
                    if stencil { 6 } else { 7 },
                ],
                srgb: false,
            });
        }
        55 => (0x07, 2, 3, true),
        56 => (0x08, 2, 4, true),
        57 => (0x12, 2, 4, false),
        58 | 59 => (0x14, 2, if code == 58 { 3 } else { 4 }, false),
        60 => (0x15, 2, 3, false),
        61 => (0x09, 2, 4, false),
        62 => (0x09, 4, 4, false),
        63 => (0x21, 7, 3, false),
        64 => (0x20, 7, 3, false),
        65 | 66 | 69 | 70 => (
            0x24,
            2,
            if matches!(code, 65 | 69) { 3 } else { 4 },
            code >= 69,
        ),
        67 | 71 => (0x25, 2, 4, code == 71),
        68 | 72 => (0x26, 2, 4, code == 72),
        73 | 74 => (0x27, if code == 74 { 1 } else { 2 }, 1, false),
        75 | 76 => (0x28, if code == 76 { 1 } else { 2 }, 2, false),
        77 | 78 => (0x17, 2, 4, code == 78),
        79 => (0x10, 7, 3, false),
        80 => (0x11, 7, 3, false),
        109..=112 => (0x07, [2, 1, 4, 3][(code - 109) as usize], 3, false),
        113..=117 => (0x03, [7, 2, 1, 4, 3][(code - 113) as usize], 3, false),
        118..=120 => (0x01, [7, 4, 3][(code - 118) as usize], 3, false),
        121..=148 => {
            let components = [
                0x40, 0x50, 0x41, 0x51, 0x42, 0x55, 0x52, 0x44, 0x56, 0x57, 0x53, 0x45, 0x54, 0x46,
            ];
            (components[((code - 121) % 14) as usize], 2, 4, code >= 135)
        }
        _ => return None,
    };
    let one = if matches!(data, 3 | 4) { 6 } else { 7 };
    Some(Encoding {
        component,
        types: [data; 4],
        sources: [
            2,
            if channels > 1 { 3 } else { 0 },
            if channels > 2 { 4 } else { 0 },
            if channels > 3 { 5 } else { one },
        ],
        srgb,
    })
}
