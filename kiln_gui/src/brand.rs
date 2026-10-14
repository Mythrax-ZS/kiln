//! The look of the program: the icon, the mascot (Cinder, a fire salamander who lives in the
//! kiln) and the lettering of the name, as pixel art drawn for this project.
//!
//! The art is text, one character per pixel, so that it can be edited by hand. `cargo run -p
//! kiln_gui --example brand` writes every picture the repository keeps under `assets/brand/` (the
//! PNG files, the icon file, the vector files and the banners) from what is here, and the program
//! itself draws its icon and its mascot from the same rows.
#![allow(dead_code)] // The program uses part of this; the example that writes the pictures, the rest.

/// What each character of the art stands for (red, green, blue). `.` is transparent.
pub const PALETTE: &[(char, u32)] = &[
    ('#', 0x0d0910), // ink, the outline
    ('L', 0x161228), // the line between two parts of Cinder
    ('B', 0x2e2a4c), // Cinder's body
    ('H', 0x574d8c), // the light on its upper edges
    ('D', 0x1d1a34), // its shadow
    ('S', 0x25213f), // the shaded lower body
    ('f', 0x211d34), // a leg on the far side
    ('F', 0x272340), // the upper edge of a far leg
    ('b', 0xf0b868), // the belly
    ('c', 0xd08a40), // the belly's lower row
    ('s', 0xffc233), // a spot
    ('o', 0xff8f1f), // a spot's lower edge
    ('e', 0xfff8e6), // an eye
    ('w', 0xffffff), // the shine in it
    ('1', 0xfff4bd), // white-hot
    ('2', 0xffd558), // yellow fire
    ('3', 0xff9a2e), // orange fire
    ('4', 0xf0601f), // red fire
    ('5', 0xc03c18), // deep red fire
    ('k', 0xff8a5c), // a cheek
    ('A', 0xc4603a), // kiln brick
    ('Z', 0xa54a2c), // darker kiln brick
    ('M', 0x5a2418), // mortar
    ('T', 0x7a3220), // the slab the kiln stands on
    ('t', 0x492019), // the slab's shadow
];

/// The icon at 32 by 32 pixels, without the tile it stands on: the mouth of a brick kiln with the
/// fire inside, Cinder in front of it and a few sparks.
pub const ICON32: &[&str] = &[
    "................................",
    "................................",
    ".............######.....2.......",
    "..........###MMZZZM###..........",
    ".......2.#AAAAMZZZMAAA#.........",
    "........#MMAAAMZZZMAAAM#...3....",
    ".......#ZZMMA555555AAMZZ#.......",
    "......#ZZZZM55444545MZZZZ#......",
    "......#ZZZZ4544444545ZZZZ#......",
    "......#MMZ444444444444ZMM#......",
    ".....#AAM44444444444444AAA#.....",
    ".....#AAA44434333334444AAA#.....",
    ".....#AAA34343333333434AAA#.....",
    ".....#MMM33333333333333MMM#.....",
    ".....#AAA33333323333333ZZZ#.....",
    ".....#AAA33322222223333ZZZ#.....",
    ".....#MMM23232222222323MMM#.....",
    ".....#AAA22222222222222ZZZ#.....",
    ".....#AAA22222212122HH2ZZZ#.....",
    ".....#MMM2221111111HBBBMMM#.....",
    ".....#AAA1H11111111BeeBZZZ#.....",
    ".....#AAAHB1HHHHH1HBe#BZZZ#.....",
    ".....#MMMBBHBBBBBBBBBBBMMM#.....",
    ".....#AAA1BBsBBsBBBBBB1ZZZ#.....",
    ".....#AAA11BBBBBBBBBB11ZZZ#.....",
    ".....####11B1BB11BB1B11####.....",
    "....ZZZZZZZZZZZZZZZZZZZZZZZZ....",
    "....tttttttttttttttttttttttt....",
    "....tttttttttttttttttttttttt....",
    "................................",
    "................................",
    "................................",
];

/// The icon at 16 by 16 pixels, drawn again by hand (the small sizes are not a shrunken big one).
pub const ICON16: &[&str] = &[
    "................",
    "................",
    "......AZAZ......",
    "....ZAZAZAZA....",
    "...AAA5555AAA...",
    "...ZZ444444ZZ...",
    "..AA44444444AA..",
    "..ZZ33333333ZZ..",
    "..AA333333BBAA..",
    "..ZZH2222BeBZZ..",
    "..AABHHHBBBBAA..",
    "..ZZBBsBBsBBZZ..",
    "..AA1B1BB1B1AA..",
    ".ZZZZZZZZZZZZZZ.",
    ".tttttttttttttt.",
    "................",
];

/// Cinder, standing, 55 by 32 pixels, the tip of the tail glowing.
pub const CINDER: &[&str] = &[
    ".......................................................",
    "...................3...................................",
    "............3..........................................",
    "................2.....................#####............",
    "....................................##HHHHs##..........",
    "........#######...................##HHsBBB#####........",
    ".......#HH43222#.4...............#HHBBBBB#eeee##.......",
    "......#HBB43212#................#HBBBBBB#eeeeee##......",
    "....##HBBB43222#...............#HBBBBBBB#eee#we#H#.....",
    "...#HHBBBBD43##................#BBBBBBBB#ee#####B#.....",
    "..#HBBBBBD###.................#HBBBBBBBB#ee#####BH#....",
    "..#BBBBss#....................#BBBBBBBBBB#ee###BBBH#...",
    "..#BBBBoo#....................#BBBBBBBBBBB####BBBBBH#..",
    ".#HBBBBB#......################BBBBBBBBBBBBBBBBBBBB#H#.",
    ".#BBBBBB#.....#HHHHssHHHssHHHHLBBBBBBBBBBBBBBSSSSSSSS#.",
    ".HBsssBB#...##HSSSssssBBsssBBssLsBBBBBBBBBBSSSkkSSSSb#.",
    ".BBBooBB#.##LHSSSSSooBBBBooBBBoLBBBBSSSSSSSSSSSSSSSbc#.",
    ".BBBBBBBH#HHLSSSSSSSBBBBBBBBBBBBLBBBSSSSSSSSSbbbbb#c#..",
    ".DBBBBBBBHBLHSSSSSSSBBBBBBBBBBBBBLBBSSSSSSSbbcc###c#...",
    ".#BBBBBBBBBLbssLLLSSBBsBBBBBBBBBBBLLSSSSbbbcc######....",
    ".#BBBBBBBBBLcSLoHHLSSBBBBBBBBBBBBBBBLLLLccc##..........",
    ".#DBBBBBBBBBLLHBBBHLSSSSSSSSSSSSSSSLsHHH###............",
    "..#BBBBssBBBLHBBBBBHLSSSSSSSSSSSSSLHBBBBH#.............",
    "..#DBBBooBBBLBBBBBBBLSSSSSSSSSSSSSLBBBBBB#.............",
    "...#DDBBBBBBLsBBBBBDLbbbbbbbbbbbbbLBBBBBB#.............",
    "....##DBBBBBLoBBBBB##ccccccccccccccLBBBBB#.............",
    "......#DDDBLLBBBBBB#.#LLLLL###LLLLL#BBBBBH##...........",
    ".......###LHHBBBBBB##Ffffff#..#ffffLBBBBBBHH#..........",
    "..........#DBBBBBBD#FfffffD#..#DffffLBBBBBBD#..........",
    "...........#DDLDLD#.#DDLDL#....#DLDLLDDLDLD#...........",
    "............######...#####......###########............",
    ".......................................................",
];

/// The name in the project's own dot-matrix letters (the stand-in fonts' K, I, L and N), one
/// blank column between them.
pub const WORDMARK: &[&str] = &[
    "#...#..###..#.....#...#",
    "#..#....#...#.....##..#",
    "#.#.....#...#.....#.#.#",
    "##......#...#.....#..##",
    "#.#.....#...#.....#...#",
    "#..#....#...#.....#...#",
    "#...#..###..#####.#...#",
];

/// The colours of the tile the icon stands on: the top, the bottom and the lit edge on top.
const TILE_TOP: [u8; 3] = [0x33, 0x23, 0x1c];
const TILE_BOTTOM: [u8; 3] = [0x1a, 0x11, 0x10];
const TILE_RIM: [u8; 3] = [0x4a, 0x35, 0x2c];

/// RGBA pixels, row by row, 8 bits each; a pixel is either opaque or fully transparent in the art.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Image {
    /// A transparent image.
    pub fn new(width: usize, height: usize) -> Image {
        Image {
            width,
            height,
            rgba: vec![0; width * height * 4],
        }
    }

    /// The picture the rows draw. A character that is not in the palette is a mistake in the art.
    pub fn from_art(rows: &[&str]) -> Image {
        let width = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
        let mut image = Image::new(width, rows.len());
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '.' || ch == ' ' {
                    continue;
                }
                let rgb = PALETTE
                    .iter()
                    .find(|(c, _)| *c == ch)
                    .unwrap_or_else(|| panic!("{ch:?} is not in the palette"))
                    .1;
                image.set(x, y, [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255]);
            }
        }
        image
    }

    pub fn get(&self, x: usize, y: usize) -> [u8; 4] {
        let i = (y * self.width + x) * 4;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }

    pub fn set(&mut self, x: usize, y: usize, pixel: [u8; 4]) {
        let i = (y * self.width + x) * 4;
        self.rgba[i..i + 4].copy_from_slice(&pixel);
    }

    /// Lay `other` over this image with its corner at (`x`, `y`); what falls outside is dropped.
    pub fn paste(&mut self, other: &Image, x: usize, y: usize) {
        for oy in 0..other.height {
            for ox in 0..other.width {
                let p = other.get(ox, oy);
                if p[3] != 0 && x + ox < self.width && y + oy < self.height {
                    self.set(x + ox, y + oy, p);
                }
            }
        }
    }

    /// Every pixel as a square of `k` by `k` pixels.
    pub fn scaled(&self, k: usize) -> Image {
        let mut out = Image::new(self.width * k, self.height * k);
        for y in 0..self.height {
            for x in 0..self.width {
                let p = self.get(x, y);
                if p[3] == 0 {
                    continue;
                }
                for dy in 0..k {
                    for dx in 0..k {
                        out.set(x * k + dx, y * k + dy, p);
                    }
                }
            }
        }
        out
    }
}

/// A rounded square `size` pixels wide with a vertical gradient and a lit edge on top: what the
/// icon's picture stands on.
pub fn tile(size: usize, radius: f32) -> Image {
    let s = size as f32;
    let inside = |fx: f32, fy: f32| {
        let cx = fx.clamp(radius, s - radius);
        let cy = fy.clamp(radius, s - radius);
        (fx - cx).hypot(fy - cy) <= radius
    };
    // A pixel is in when most of nine samples are.
    let covered = |x: usize, y: usize| {
        let mut n = 0;
        for sy in 0..3 {
            for sx in 0..3 {
                if inside(
                    x as f32 + (sx as f32 + 0.5) / 3.0,
                    y as f32 + (sy as f32 + 0.5) / 3.0,
                ) {
                    n += 1;
                }
            }
        }
        n * 2 > 9
    };
    let mut image = Image::new(size, size);
    for y in 0..size {
        for x in 0..size {
            if !covered(x, y) {
                continue;
            }
            let t = y as f32 / (size - 1) as f32;
            let lit = y == 0 || !covered(x, y - 1);
            let channel = |i: usize| -> u8 {
                if lit {
                    TILE_RIM[i]
                } else {
                    (TILE_TOP[i] as f32 + (TILE_BOTTOM[i] as f32 - TILE_TOP[i] as f32) * t).round()
                        as u8
                }
            };
            image.set(x, y, [channel(0), channel(1), channel(2), 255]);
        }
    }
    image
}

/// The icon at 32 by 32 pixels.
pub fn icon32() -> Image {
    let mut image = tile(32, 7.2);
    image.paste(&Image::from_art(ICON32), 0, 0);
    image
}

/// The icon at 16 by 16 pixels.
pub fn icon16() -> Image {
    let mut image = tile(16, 3.8);
    image.paste(&Image::from_art(ICON16), 0, 0);
    image
}

/// Cinder on a transparent ground.
pub fn cinder() -> Image {
    Image::from_art(CINDER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_of_the_art_is_as_wide_as_the_others_and_uses_the_palette() {
        for (name, rows, width, height) in [
            ("ICON32", ICON32, 32, 32),
            ("ICON16", ICON16, 16, 16),
            ("CINDER", CINDER, 55, 32),
        ] {
            assert_eq!(rows.len(), height, "{name}: rows");
            for (y, row) in rows.iter().enumerate() {
                assert_eq!(row.chars().count(), width, "{name}: row {y}");
            }
            // from_art panics on a character that is not in the palette
            let image = Image::from_art(rows);
            assert!(
                image.rgba.chunks(4).any(|p| p[3] == 255),
                "{name} draws something"
            );
        }
        for row in WORDMARK {
            assert_eq!(row.chars().count(), 23, "the wordmark's rows are 23 wide");
        }
        assert_eq!(WORDMARK.len(), 7);
    }

    #[test]
    fn the_palette_has_no_character_twice_and_no_colour_twice() {
        for (i, (c, rgb)) in PALETTE.iter().enumerate() {
            assert!(*c != '.' && *c != ' ');
            for (d, other) in &PALETTE[i + 1..] {
                assert_ne!(c, d, "{c:?} twice");
                assert_ne!(rgb, other, "{c:?} and {d:?} are the same colour");
            }
        }
    }

    #[test]
    fn the_icons_are_opaque_where_the_tile_is_and_their_corners_are_clear() {
        for image in [icon32(), icon16()] {
            assert_eq!(image.get(0, 0)[3], 0, "the corner is transparent");
            let (w, h) = (image.width, image.height);
            assert_eq!(image.get(w / 2, h / 2)[3], 255);
            assert_eq!(image.get(w - 1, h - 1)[3], 0);
        }
        let big = icon32().scaled(4);
        assert_eq!((big.width, big.height), (128, 128));
    }
}
