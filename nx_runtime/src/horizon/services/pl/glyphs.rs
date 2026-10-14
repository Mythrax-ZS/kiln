//! The letters of the stand-in fonts: a dot-matrix alphabet drawn for this project (the shapes are
//! ours and come from no font), turned into the rectangles a TrueType glyph is made of
//! (docs/ASSUMPTIONS.md A-073).

use super::be16;

/// The side of one square of the grid in font units (1024 to the em): a capital letter is seven
/// squares tall (700), the body of a small letter five (500) and what goes below the baseline two
/// (200), which is the descent the fonts declare.
pub const UNIT: i16 = 100;

/// The advance of a glyph that has none of its own: six squares, which is a five wide letter and
/// the space after it.
pub const DEFAULT_ADVANCE: u16 = 6 * UNIT as u16;

/// The advance of the space.
const SPACE_ADVANCE: u16 = 4 * UNIT as u16;

/// A letter on the grid: `rows` run down from the row `top` (row 0 is the top of a capital letter,
/// row 6 the one that stands on the baseline, rows 7 and 8 go below it) and `#` is ink.
struct Bitmap {
    code: char,
    top: u8,
    rows: &'static [&'static str],
}

const fn bitmap(code: char, top: u8, rows: &'static [&'static str]) -> Bitmap {
    Bitmap { code, top, rows }
}

#[rustfmt::skip]
const ALPHABET: &[Bitmap] = &[
    bitmap('!', 0, &["#", "#", "#", "#", "#", ".", "#"]),
    bitmap('"', 0, &["#.#", "#.#"]),
    bitmap('#', 0, &[".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."]),
    bitmap('$', 0, &["..#..", ".####", "#.#..", ".###.", "..#.#", "####.", "..#.."]),
    bitmap('%', 0, &["##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"]),
    bitmap('&', 0, &[".##..", "#..#.", "#.#..", ".#...", "#.#.#", "#..#.", ".##.#"]),
    bitmap('\'', 0, &["#", "#"]),
    bitmap('(', 0, &["..#", ".#.", "#..", "#..", "#..", ".#.", "..#"]),
    bitmap(')', 0, &["#..", ".#.", "..#", "..#", "..#", ".#.", "#.."]),
    bitmap('*', 1, &["..#..", "#.#.#", ".###.", "#.#.#", "..#.."]),
    bitmap('+', 2, &["..#..", "..#..", "#####", "..#..", "..#.."]),
    bitmap(',', 6, &["##", ".#", "#."]),
    bitmap('-', 4, &["####"]),
    bitmap('.', 6, &["#"]),
    bitmap('/', 0, &["....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."]),
    bitmap('0', 0, &[".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."]),
    bitmap('1', 0, &["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."]),
    bitmap('2', 0, &[".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"]),
    bitmap('3', 0, &[".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."]),
    bitmap('4', 0, &["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."]),
    bitmap('5', 0, &["#####", "#....", "####.", "....#", "....#", "#...#", ".###."]),
    bitmap('6', 0, &["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."]),
    bitmap('7', 0, &["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."]),
    bitmap('8', 0, &[".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."]),
    bitmap('9', 0, &[".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."]),
    bitmap(':', 2, &["#", ".", ".", ".", "#"]),
    bitmap(';', 2, &[".#", "..", "..", "..", ".#", ".#", "#."]),
    bitmap('<', 2, &["..#", ".#.", "#..", ".#.", "..#"]),
    bitmap('=', 3, &["#####", ".....", "#####"]),
    bitmap('>', 2, &["#..", ".#.", "..#", ".#.", "#.."]),
    bitmap('?', 0, &[".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."]),
    bitmap('@', 0, &[".###.", "#...#", "#.###", "#.#.#", "#.###", "#....", ".####"]),
    bitmap('A', 0, &[".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"]),
    bitmap('B', 0, &["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."]),
    bitmap('C', 0, &[".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."]),
    bitmap('D', 0, &["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."]),
    bitmap('E', 0, &["#####", "#....", "#....", "####.", "#....", "#....", "#####"]),
    bitmap('F', 0, &["#####", "#....", "#....", "####.", "#....", "#....", "#...."]),
    bitmap('G', 0, &[".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"]),
    bitmap('H', 0, &["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"]),
    bitmap('I', 0, &[".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."]),
    bitmap('J', 0, &["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."]),
    bitmap('K', 0, &["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"]),
    bitmap('L', 0, &["#....", "#....", "#....", "#....", "#....", "#....", "#####"]),
    bitmap('M', 0, &["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"]),
    bitmap('N', 0, &["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"]),
    bitmap('O', 0, &[".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."]),
    bitmap('P', 0, &["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."]),
    bitmap('Q', 0, &[".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"]),
    bitmap('R', 0, &["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"]),
    bitmap('S', 0, &[".####", "#....", "#....", ".###.", "....#", "....#", "####."]),
    bitmap('T', 0, &["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."]),
    bitmap('U', 0, &["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."]),
    bitmap('V', 0, &["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."]),
    bitmap('W', 0, &["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"]),
    bitmap('X', 0, &["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"]),
    bitmap('Y', 0, &["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."]),
    bitmap('Z', 0, &["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"]),
    bitmap('[', 0, &["##", "#.", "#.", "#.", "#.", "#.", "##"]),
    bitmap('\\', 0, &["#....", "#....", ".#...", "..#..", "...#.", "....#", "....#"]),
    bitmap(']', 0, &["##", ".#", ".#", ".#", ".#", ".#", "##"]),
    bitmap('^', 0, &["..#..", ".#.#.", "#...#"]),
    bitmap('_', 8, &["#####"]),
    bitmap('`', 0, &["#.", ".#"]),
    bitmap('a', 2, &[".###.", "....#", ".####", "#...#", ".####"]),
    bitmap('b', 0, &["#....", "#....", "####.", "#...#", "#...#", "#...#", "####."]),
    bitmap('c', 2, &[".###.", "#...#", "#....", "#...#", ".###."]),
    bitmap('d', 0, &["....#", "....#", ".####", "#...#", "#...#", "#...#", ".####"]),
    bitmap('e', 2, &[".###.", "#...#", "#####", "#....", ".###."]),
    bitmap('f', 0, &["..##", ".#..", "###.", ".#..", ".#..", ".#..", ".#.."]),
    bitmap('g', 2, &[".####", "#...#", "#...#", "#...#", ".####", "....#", ".###."]),
    bitmap('h', 0, &["#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#"]),
    bitmap('i', 0, &[".#.", "...", "##.", ".#.", ".#.", ".#.", "###"]),
    bitmap('j', 0, &[".#.", "...", "##.", ".#.", ".#.", ".#.", ".#.", ".#.", "##."]),
    bitmap('k', 0, &["#...", "#...", "#..#", "#.#.", "##..", "#.#.", "#..#"]),
    bitmap('l', 0, &["##.", ".#.", ".#.", ".#.", ".#.", ".#.", "###"]),
    bitmap('m', 2, &["##.#.", "#.#.#", "#.#.#", "#.#.#", "#.#.#"]),
    bitmap('n', 2, &["####.", "#...#", "#...#", "#...#", "#...#"]),
    bitmap('o', 2, &[".###.", "#...#", "#...#", "#...#", ".###."]),
    bitmap('p', 2, &["####.", "#...#", "#...#", "#...#", "####.", "#....", "#...."]),
    bitmap('q', 2, &[".####", "#...#", "#...#", "#...#", ".####", "....#", "....#"]),
    bitmap('r', 2, &["#.##.", "##..#", "#....", "#....", "#...."]),
    bitmap('s', 2, &[".####", "#....", ".###.", "....#", "####."]),
    bitmap('t', 1, &[".#..", "###.", ".#..", ".#..", ".#..", "..##"]),
    bitmap('u', 2, &["#...#", "#...#", "#...#", "#..##", ".##.#"]),
    bitmap('v', 2, &["#...#", "#...#", "#...#", ".#.#.", "..#.."]),
    bitmap('w', 2, &["#...#", "#...#", "#.#.#", "#.#.#", ".#.#."]),
    bitmap('x', 2, &["#...#", ".#.#.", "..#..", ".#.#.", "#...#"]),
    bitmap('y', 2, &["#...#", "#...#", "#...#", ".####", "....#", "....#", ".###."]),
    bitmap('z', 2, &["#####", "...#.", "..#..", ".#...", "#####"]),
    bitmap('{', 0, &["..#", ".#.", ".#.", "#..", ".#.", ".#.", "..#"]),
    bitmap('|', 0, &["#", "#", "#", "#", "#", "#", "#", "#", "#"]),
    bitmap('}', 0, &["#..", ".#.", ".#.", "..#", ".#.", ".#.", "#.."]),
    bitmap('~', 3, &[".##.#", "#.##."]),
    bitmap('\u{00B7}', 4, &["#"]),
    bitmap('\u{00D7}', 2, &["#...#", ".#.#.", "..#..", ".#.#.", "#...#"]),
    bitmap('\u{2013}', 4, &["#####"]),
    bitmap('\u{2014}', 4, &["#######"]),
    bitmap('\u{2022}', 3, &[".#.", "###", ".#."]),
    bitmap('\u{2026}', 6, &["#.#.#"]),
];

/// A glyph: how far it moves the pen, and its ink as rectangles `[x0, y0, x1, y1]` in font units
/// (y up from the baseline).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glyph {
    pub advance: u16,
    pub rects: Vec<[i16; 4]>,
}

impl Glyph {
    /// A glyph with no ink.
    pub fn empty(advance: u16) -> Glyph {
        Glyph {
            advance,
            rects: Vec::new(),
        }
    }

    /// The left edge of the ink, which is the glyph's left side bearing.
    pub fn x_min(&self) -> i16 {
        self.rects.iter().map(|r| r[0]).min().unwrap_or(0)
    }

    /// The edges of the ink: `[x min, y min, x max, y max]`.
    pub fn bounds(&self) -> [i16; 4] {
        let edge = |i: usize, pick: fn(i16, i16) -> i16, from: i16| {
            self.rects.iter().map(|r| r[i]).fold(from, pick)
        };
        if self.rects.is_empty() {
            return [0; 4];
        }
        [
            edge(0, i16::min, i16::MAX),
            edge(1, i16::min, i16::MAX),
            edge(2, i16::max, i16::MIN),
            edge(3, i16::max, i16::MIN),
        ]
    }

    /// The glyph as a `glyf` table entry: one clockwise contour for each rectangle, with
    /// every point on the curve and no instructions, padded to an even length.
    pub fn record(&self) -> Vec<u8> {
        let [x_min, y_min, x_max, y_max] = self.bounds();
        let mut t = Vec::new();
        t.extend(be16(self.rects.len() as u16));
        for v in [x_min, y_min, x_max, y_max] {
            t.extend(be16(v as u16));
        }
        for i in 1..=self.rects.len() {
            t.extend(be16(4 * i as u16 - 1));
        }
        t.extend(be16(0)); // no instructions
        let points: Vec<(i16, i16)> = self
            .rects
            .iter()
            .flat_map(|&[x0, y0, x1, y1]| [(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
            .collect();
        // Flags: on the curve, with x and y given as 16-bit differences from the point before.
        t.extend(std::iter::repeat_n(1u8, points.len()));
        let mut last = (0i16, 0i16);
        for &(x, _) in &points {
            t.extend(be16(x.wrapping_sub(last.0) as u16));
            last.0 = x;
        }
        for &(_, y) in &points {
            t.extend(be16(y.wrapping_sub(last.1) as u16));
            last.1 = y;
        }
        if t.len() % 2 != 0 {
            t.push(0);
        }
        t
    }
}

/// The character whose shape a character is drawn with: the typographic spaces and quotes use
/// those of the keyboard.
fn drawn_as(code: u16) -> char {
    match code {
        0x00A0 => ' ',
        0x2018 | 0x2019 => '\'',
        0x201C | 0x201D => '"',
        c => char::from_u32(u32::from(c)).unwrap_or('\u{FFFD}'),
    }
}

/// The rectangles of a letter: runs of ink in a row, joined to the run above them when they have
/// the same columns, so that a stem is one rectangle.
fn rectangles(letter: &Bitmap) -> Vec<[i16; 4]> {
    // (first column, column after the last, first row, row after the last)
    let mut done: Vec<(usize, usize, usize, usize)> = Vec::new();
    // The same, with the last row not known yet.
    let mut open: Vec<(usize, usize, usize)> = Vec::new();
    for (r, row) in letter.rows.iter().enumerate() {
        let ink = row.as_bytes();
        let mut next = Vec::new();
        let mut c = 0;
        while c < ink.len() {
            if ink[c] != b'#' {
                c += 1;
                continue;
            }
            let start = c;
            while c < ink.len() && ink[c] == b'#' {
                c += 1;
            }
            match open.iter().position(|o| (o.0, o.1) == (start, c)) {
                Some(i) => next.push(open.remove(i)),
                None => next.push((start, c, r)),
            }
        }
        done.extend(open.drain(..).map(|(a, b, from)| (a, b, from, r)));
        open = next;
    }
    done.extend(
        open.into_iter()
            .map(|(a, b, from)| (a, b, from, letter.rows.len())),
    );
    done.into_iter()
        .map(|(a, b, from, to)| {
            let top = i16::from(letter.top) + from as i16;
            let bottom = i16::from(letter.top) + to as i16;
            [
                a as i16 * UNIT,
                (7 - bottom) * UNIT,
                b as i16 * UNIT,
                (7 - top) * UNIT,
            ]
        })
        .collect()
}

/// The glyph of a character: drawn if the alphabet has a shape for it, and without ink if not.
pub fn glyph(code: u16) -> Glyph {
    let letter = drawn_as(code);
    if letter == ' ' {
        return Glyph::empty(SPACE_ADVANCE);
    }
    match ALPHABET.iter().find(|b| b.code == letter) {
        Some(b) => {
            let width = b.rows.iter().map(|r| r.len()).max().unwrap_or(0) as u16;
            Glyph {
                // The room after the letter is one square, and a glyph is never narrower than
                // three.
                advance: ((width + 1).max(3)) * UNIT as u16,
                rects: rectangles(b),
            }
        }
        None => Glyph::empty(DEFAULT_ADVANCE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The squares of ink in a bitmap.
    fn ink(b: &Bitmap) -> i64 {
        b.rows
            .iter()
            .map(|r| r.bytes().filter(|&c| c == b'#').count() as i64)
            .sum()
    }

    #[test]
    fn every_key_on_the_keyboard_has_a_shape_and_the_space_has_none() {
        for code in 0x21u16..=0x7E {
            let g = glyph(code);
            assert!(
                !g.rects.is_empty(),
                "{:?} has no ink",
                char::from_u32(u32::from(code))
            );
        }
        let space = glyph(0x20);
        assert!(space.rects.is_empty());
        assert_eq!(space.advance, SPACE_ADVANCE);
        assert_eq!(glyph(0x00A0), space, "the no-break space is a space");
        assert_eq!(
            glyph(0x2019),
            glyph(0x27),
            "a curly quote is a straight one"
        );
    }

    #[test]
    fn bitmaps_are_drawn_on_the_grid() {
        let mut seen = std::collections::HashSet::new();
        for b in ALPHABET {
            assert!(seen.insert(b.code), "{:?} is drawn twice", b.code);
            assert!(!b.rows.is_empty(), "{:?}", b.code);
            // The rows stay within the nine rows of the grid and are at most seven squares wide.
            assert!(b.top as usize + b.rows.len() <= 9, "{:?}", b.code);
            for row in b.rows {
                assert!(row.len() <= 7, "{:?}", b.code);
                assert!(row.bytes().all(|c| c == b'#' || c == b'.'), "{:?}", b.code);
            }
            // All the rows of a letter are as long as each other: a typo shows as a ragged row.
            assert!(
                b.rows.iter().all(|r| r.len() == b.rows[0].len()),
                "{:?} has rows of different lengths",
                b.code
            );
            assert!(b.rows.iter().any(|r| r.contains('#')), "{:?}", b.code);
        }
    }

    #[test]
    fn rectangles_cover_the_squares_of_ink_exactly_once() {
        for b in ALPHABET {
            let area: i64 = rectangles(b)
                .iter()
                .map(|&[x0, y0, x1, y1]| i64::from(x1 - x0) * i64::from(y1 - y0))
                .sum();
            assert_eq!(
                area,
                ink(b) * i64::from(UNIT) * i64::from(UNIT),
                "{:?}",
                b.code
            );
            // No two overlap: every square belongs to one rectangle.
            let rects = rectangles(b);
            for (i, a) in rects.iter().enumerate() {
                for c in &rects[i + 1..] {
                    let apart = a[2] <= c[0] || c[2] <= a[0] || a[3] <= c[1] || c[3] <= a[1];
                    assert!(apart, "{:?}: {a:?} and {c:?} overlap", b.code);
                }
            }
        }
    }

    #[test]
    fn letters_stand_on_the_baseline_and_have_the_heights_of_their_kind() {
        // A capital: seven squares from the baseline up. A small letter without a tail: five.
        // The tail of a g goes two squares below.
        assert_eq!(glyph(u16::from(b'H')).bounds(), [0, 0, 5 * UNIT, 7 * UNIT]);
        assert_eq!(glyph(u16::from(b'x')).bounds(), [0, 0, 5 * UNIT, 5 * UNIT]);
        assert_eq!(glyph(u16::from(b'g')).bounds()[1], -2 * UNIT);
        assert_eq!(glyph(u16::from(b'g')).bounds()[3], 5 * UNIT);
        assert_eq!(glyph(u16::from(b'b')).bounds()[3], 7 * UNIT);
        assert_eq!(glyph(u16::from(b'.')).bounds(), [0, 0, UNIT, UNIT]);
        assert_eq!(glyph(u16::from(b'H')).advance, 6 * UNIT as u16);
        assert_eq!(glyph(u16::from(b'i')).advance, 4 * UNIT as u16);
    }

    #[test]
    fn a_stem_is_one_rectangle() {
        // The I is a bar on top, a stem and a bar at the bottom, in the order they are closed.
        let u = UNIT;
        assert_eq!(
            glyph(u16::from(b'I')).rects,
            [
                [u, 6 * u, 4 * u, 7 * u],
                [2 * u, u, 3 * u, 6 * u],
                [u, 0, 4 * u, u]
            ]
        );
    }

    #[test]
    fn a_record_has_one_contour_of_four_points_for_each_rectangle() {
        let g = glyph(u16::from(b'A'));
        let rec = g.record();
        let n = g.rects.len();
        assert_eq!(u16::from_be_bytes([rec[0], rec[1]]) as usize, n);
        assert_eq!(rec.len() % 2, 0);
        // Header, end points, instruction length, flags, x and y differences (padded).
        assert_eq!(
            rec.len(),
            (10 + 2 * n + 2 + 4 * n + 2 * 4 * n + 2 * 4 * n).next_multiple_of(2)
        );
        let last_end = u16::from_be_bytes([rec[10 + 2 * (n - 1)], rec[11 + 2 * (n - 1)]]);
        assert_eq!(last_end as usize, 4 * n - 1);
    }
}
