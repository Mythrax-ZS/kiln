//! `pl:u`, the platform service that shares the system fonts (Switchbrew, "Shared Database
//! services", the font section).
//!
//! The console's system fonts are firmware data, which this project does not have and does not
//! contain. What the service shares here is a stand-in made by this code: six TrueType fonts (one
//! per `SharedFontType`) that have a name, metrics, and a glyph for every character of the Basic
//! Multilingual Plane. Only the characters of an English keyboard (and a few typographic ones)
//! have an outline, in a dot-matrix alphabet drawn for this project; text in any other script is
//! blank (docs/ASSUMPTIONS.md A-073). They sit in a shared memory of 0x1100000 bytes at the
//! offsets `GetSharedMemoryAddressOffset` reports.

mod glyphs;

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// The size of the font shared memory (Switchbrew: `0x1100000`).
const FONT_MEMORY_SIZE: u64 = 0x110_0000;

/// How many `SharedFontType`s there are (Switchbrew: Standard, Chinese Simplified, Extended
/// Chinese Simplified, Chinese Traditional, Korean, Nintendo Extended).
const FONT_TYPES: usize = 6;

/// The room each font has in the shared memory.
const SLOT: u32 = 0x5_0000;

/// Where font `i` starts in the shared memory: one slot each after a first empty page.
fn font_offset(i: usize) -> u32 {
    0x1000 + SLOT * i as u32
}

/// The characters that have a glyph: U+0020 up to U+FFFE. Glyph 0 is `.notdef`; the glyph of a
/// character is its code minus `FIRST_CHAR - 1`.
const FIRST_CHAR: u16 = 0x0020;
const LAST_CHAR: u16 = 0xFFFE;
const GLYPHS: usize = (LAST_CHAR - FIRST_CHAR) as usize + 2;

/// The design grid of the stand-ins and the metrics derived from it.
const UNITS_PER_EM: u16 = 1024;
const ASCENT: i16 = 800;
const DESCENT: i16 = -200;

/// The last character that has a metric of its own in `hmtx` (the last one the alphabet draws, the
/// ellipsis): the glyphs after it all have the advance of the last one, which is the default.
const LAST_METRIC_CHAR: u16 = 0x2026;

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// The sum of the big-endian 32-bit words of a table, zero padded (the OpenType checksum).
fn checksum(table: &[u8]) -> u32 {
    let mut sum = 0u32;
    for chunk in table.chunks(4) {
        let mut w = [0u8; 4];
        w[..chunk.len()].copy_from_slice(chunk);
        sum = sum.wrapping_add(u32::from_be_bytes(w));
    }
    sum
}

/// A TrueType font named `name` with `GLYPHS` glyphs, the Latin ones drawn and all the others
/// empty, and a format 4 `cmap` that gives every character from U+0020 to U+FFFE its own glyph.
pub fn stand_in_font(name: &str) -> Vec<u8> {
    // The glyphs that have a metric of their own: `.notdef`, then U+0020 on.
    let drawn: Vec<glyphs::Glyph> = std::iter::once(glyphs::Glyph::empty(glyphs::DEFAULT_ADVANCE))
        .chain((FIRST_CHAR..=LAST_METRIC_CHAR).map(glyphs::glyph))
        .collect();
    debug_assert_eq!(
        drawn.last().map(|g| g.advance),
        Some(glyphs::DEFAULT_ADVANCE)
    );
    let inked = || drawn.iter().filter(|g| !g.rects.is_empty());
    // The edges of all the ink, and the widest advance.
    let (x_min, y_min, x_max, y_max) = inked().map(|g| g.bounds()).fold(
        (i16::MAX, i16::MAX, i16::MIN, i16::MIN),
        |(a, b, c, d), [x0, y0, x1, y1]| (a.min(x0), b.min(y0), c.max(x1), d.max(y1)),
    );
    let widest = drawn.iter().map(|g| g.advance).max().unwrap_or(0);
    let min_right_bearing = inked()
        .map(|g| g.advance as i16 - g.bounds()[2])
        .min()
        .unwrap_or(0);
    let head = {
        let mut t = Vec::new();
        t.extend(be32(0x0001_0000)); // version
        t.extend(be32(0x0001_0000)); // font revision
        t.extend(be32(0)); // checksum adjustment, filled in below
        t.extend(be32(0x5F0F_3CF5)); // magic
        t.extend(be16(0x000B)); // flags
        t.extend(be16(UNITS_PER_EM));
        t.extend([0u8; 16]); // created, modified
        t.extend(be16(x_min as u16));
        t.extend(be16(y_min as u16));
        t.extend(be16(x_max as u16));
        t.extend(be16(y_max as u16));
        t.extend(be16(0)); // mac style
        t.extend(be16(8)); // lowest recommended size
        t.extend(be16(2)); // direction hint
        t.extend(be16(0)); // short offsets in loca
        t.extend(be16(0)); // glyph data format
        t
    };
    let hhea = {
        let mut t = Vec::new();
        t.extend(be32(0x0001_0000));
        t.extend(be16(ASCENT as u16));
        t.extend(be16(DESCENT as u16));
        t.extend(be16(0)); // line gap
        t.extend(be16(widest)); // widest advance
        t.extend(be16(x_min as u16)); // min left side bearing
        t.extend(be16(min_right_bearing as u16)); // min right side bearing
        t.extend(be16(x_max as u16)); // largest extent
        t.extend(be16(1)); // caret slope rise
        t.extend(be16(0)); // caret slope run
        t.extend(be16(0)); // caret offset
        t.extend([0u8; 8]); // reserved
        t.extend(be16(0)); // metric data format
                           // The glyphs after the last full metric only have a left side bearing.
        t.extend(be16(drawn.len() as u16));
        t
    };
    let maxp = {
        let mut t = Vec::new();
        t.extend(be32(0x0001_0000));
        t.extend(be16(GLYPHS as u16));
        let most_rects = drawn.iter().map(|g| g.rects.len()).max().unwrap_or(0) as u16;
        t.extend(be16(4 * most_rects)); // max points
        t.extend(be16(most_rects)); // max contours
        t.extend(be16(0)); // max composite points
        t.extend(be16(0)); // max composite contours
        t.extend(be16(1)); // zones
                           // twilight points, storage, function defs, instruction defs, stack elements, size of
                           // instructions, component elements, component depth
        t.extend([0u8; 16]);
        t
    };
    // An advance and a left side bearing (where the ink starts) for each drawn glyph, then a left
    // side bearing (zero) for each of the others, which have the last advance.
    let mut hmtx: Vec<u8> = drawn
        .iter()
        .flat_map(|g| [be16(g.advance), be16(g.x_min() as u16)].concat())
        .collect();
    hmtx.extend(std::iter::repeat_n(0u8, 2 * (GLYPHS - drawn.len())));
    // Each glyph starts where the one before it ends (offsets are in units of two bytes): the
    // glyphs without ink take no room.
    let mut glyf: Vec<u8> = Vec::new();
    let mut loca: Vec<u8> = Vec::with_capacity(2 * (GLYPHS + 1));
    for gid in 0..GLYPHS {
        loca.extend(be16((glyf.len() / 2) as u16));
        if let Some(g) = drawn.get(gid).filter(|g| !g.rects.is_empty()) {
            glyf.extend(g.record());
        }
    }
    debug_assert!(glyf.len() < 2 * 0x1_0000, "short offsets reach 128 KiB");
    loca.extend(be16((glyf.len() / 2) as u16));
    if glyf.is_empty() {
        glyf = vec![0u8; 4];
    }
    let cmap = {
        let mut t = Vec::new();
        t.extend(be16(0)); // version
        t.extend(be16(1)); // one encoding record
        t.extend(be16(3)); // Windows
        t.extend(be16(1)); // Unicode BMP
        t.extend(be32(12)); // offset of the subtable
        t.extend(be16(4)); // format 4
        t.extend(be16(32)); // length
        t.extend(be16(0)); // language
        t.extend(be16(4)); // segment count times two
        t.extend(be16(4)); // search range
        t.extend(be16(1)); // entry selector
        t.extend(be16(0)); // range shift
        t.extend([be16(LAST_CHAR), be16(0xFFFF)].concat()); // end codes
        t.extend(be16(0)); // reserved
        t.extend([be16(FIRST_CHAR), be16(0xFFFF)].concat()); // start codes
                                                             // U+0020 is glyph 1; the last segment only ends the table (U+FFFF is glyph 0)
        t.extend([be16(1u16.wrapping_sub(FIRST_CHAR)), be16(1)].concat());
        t.extend([be16(0), be16(0)].concat()); // range offsets
        t
    };
    let name_table = {
        let utf16: Vec<u8> = name.encode_utf16().flat_map(|u| u.to_be_bytes()).collect();
        let mut t = Vec::new();
        t.extend(be16(0)); // format
        t.extend(be16(1)); // one record
        t.extend(be16(18)); // string storage offset
        t.extend(be16(3)); // Windows
        t.extend(be16(1)); // Unicode BMP
        t.extend(be16(0x0409)); // American English
        t.extend(be16(1)); // family name
        t.extend(be16(utf16.len() as u16));
        t.extend(be16(0));
        t.extend(utf16);
        t
    };
    let post = {
        let mut t = Vec::new();
        t.extend(be32(0x0003_0000)); // no glyph names
        t.extend(be32(0)); // italic angle
        t.extend(be16((-100i16) as u16)); // underline position
        t.extend(be16(50)); // underline thickness
        t.extend([0u8; 20]); // fixed pitch and memory hints
        t
    };
    // The OS/2 table, version 4 (96 bytes). A text engine takes the ascent and the height of a
    // line from its Windows metrics (and the typographic ones), not only from `hhea`: with this
    // table missing it computes a height of zero.
    let os2 = {
        let mut t = Vec::new();
        t.extend(be16(4)); // version
        t.extend(be16(glyphs::DEFAULT_ADVANCE)); // average character width
        t.extend(be16(400)); // weight class: regular
        t.extend(be16(5)); // width class: medium
        t.extend(be16(0)); // type flags: embedding allowed
                           // subscript and superscript: size x, size y, offset x, offset y
        for v in [650i16, 600, 0, 75, 650, 600, 0, 350] {
            t.extend(be16(v as u16));
        }
        t.extend(be16(50)); // strikeout size
        t.extend(be16(300)); // strikeout position
        t.extend(be16(0)); // family class
        t.extend([0u8; 10]); // panose
        t.extend(be32(1)); // unicode ranges: Basic Latin
        t.extend([0u8; 12]);
        t.extend(*b"NONE"); // vendor
        t.extend(be16(0x00C0)); // selection: regular, use the typographic metrics
        t.extend(be16(FIRST_CHAR));
        t.extend(be16(LAST_CHAR));
        t.extend(be16(ASCENT as u16)); // typographic ascender
        t.extend(be16(DESCENT as u16)); // typographic descender
        t.extend(be16(0)); // typographic line gap
        t.extend(be16(ASCENT as u16)); // Windows ascent
        t.extend(be16(DESCENT.unsigned_abs())); // Windows descent, a positive number
        t.extend(be32(1)); // code pages: Latin 1
        t.extend(be32(0));
        t.extend(be16(500)); // x height
        t.extend(be16(700)); // cap height
        t.extend(be16(0)); // default character: .notdef
        t.extend(be16(0x0020)); // break character: the space
        t.extend(be16(1)); // maximum context
        t
    };

    // Tables in tag order, each padded to four bytes.
    let tables: [(&[u8; 4], &Vec<u8>); 10] = [
        (b"OS/2", &os2),
        (b"cmap", &cmap),
        (b"glyf", &glyf),
        (b"head", &head),
        (b"hhea", &hhea),
        (b"hmtx", &hmtx),
        (b"loca", &loca),
        (b"maxp", &maxp),
        (b"name", &name_table),
        (b"post", &post),
    ];
    let count = tables.len();
    let header = 12 + 16 * count;
    let mut out = Vec::new();
    out.extend(be32(0x0001_0000));
    out.extend(be16(count as u16));
    out.extend(be16(128)); // search range: 16 times the largest power of two <= count (8)
    out.extend(be16(3)); // entry selector: log2 of that power
    out.extend(be16((count * 16 - 128) as u16)); // range shift
    let mut offset = header;
    let mut body = Vec::new();
    let mut head_at = 0;
    for (tag, table) in &tables {
        out.extend_from_slice(*tag);
        out.extend(be32(checksum(table)));
        out.extend(be32(offset as u32));
        out.extend(be32(table.len() as u32));
        if *tag == b"head" {
            head_at = offset;
        }
        body.extend_from_slice(table);
        let pad = table.len().next_multiple_of(4) - table.len();
        body.extend(std::iter::repeat_n(0u8, pad));
        offset += table.len() + pad;
    }
    out.extend(body);
    // The whole file sums to 0xB1B0AFBA once the head table's adjustment is set.
    let adjustment = 0xB1B0_AFBAu32.wrapping_sub(checksum(&out));
    out[head_at + 8..head_at + 12].copy_from_slice(&be32(adjustment));
    out
}

/// The names the six stand-ins carry.
const NAMES: [&str; FONT_TYPES] = [
    "kiln standard",
    "kiln chinese simplified",
    "kiln extended chinese simplified",
    "kiln chinese traditional",
    "kiln korean",
    "kiln nintendo extended",
];

/// The six fonts, made once.
fn fonts() -> &'static [Vec<u8>; FONT_TYPES] {
    static FONTS: OnceLock<[Vec<u8>; FONT_TYPES]> = OnceLock::new();
    FONTS.get_or_init(|| NAMES.map(stand_in_font))
}

/// What every session shares: the shared memory object.
#[derive(Default)]
pub struct Shared {
    memory: Mutex<Option<KObject>>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shared")
    }
}

/// `nn::pl::detail::ISharedResourceManager`.
#[derive(Debug)]
pub struct SharedResourceManager {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl SharedResourceManager {
    pub fn new(kernel: Weak<Kernel>, shared: Arc<Shared>) -> SharedResourceManager {
        SharedResourceManager { kernel, shared }
    }

    /// The shared memory, made and filled with the fonts on first use.
    fn memory(&self, kernel: &Kernel) -> Result<KObject, Stop> {
        let mut slot = self.shared.memory.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(o) = slot.as_ref() {
            return Ok(o.clone());
        }
        let (object, addr) = kernel.create_shared_memory(FONT_MEMORY_SIZE).map_err(|e| {
            Stop(format!(
                "the font shared memory cannot be made ({:#x})",
                e.0
            ))
        })?;
        for (i, font) in fonts().iter().enumerate() {
            if !kernel.write_memory(addr + u64::from(font_offset(i)), font) {
                return Err(Stop("the font shared memory cannot be written".into()));
            }
        }
        *slot = Some(object.clone());
        Ok(object)
    }
}

impl Service for SharedResourceManager {
    fn name(&self) -> &str {
        "nn::pl::detail::ISharedResourceManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        let font = |req: &Request| -> Result<usize, Stop> {
            let t = req.u32_at(0) as usize;
            if t < FONT_TYPES {
                Ok(t)
            } else {
                Err(Stop(format!(
                    "a shared font of type {t}, which does not exist"
                )))
            }
        };
        match req.command {
            // RequestLoad(u32 SharedFontType): they are all loaded
            0 => font(req).map(|_| Response::ok()),
            // GetLoadState(u32 type) -> u32: 1 is loaded
            1 => font(req).map(|_| Response::ok().with_u32(1)),
            // GetSize(u32 type) -> u32
            2 => {
                let i = font(req)?;
                Ok(Response::ok().with_u32(fonts()[i].len() as u32))
            }
            // GetSharedMemoryAddressOffset(u32 type) -> u32
            3 => {
                let i = font(req)?;
                Ok(Response::ok().with_u32(font_offset(i)))
            }
            // GetSharedMemoryNativeHandle() -> the shared memory
            4 => {
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".to_string()))?;
                Ok(Response::ok().with_copy_handle(self.memory(&kernel)?))
            }
            other => Err(Stop::unknown_command("pl:u", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(b: &[u8], o: usize) -> u16 {
        u16::from_be_bytes([b[o], b[o + 1]])
    }

    fn u32_at(b: &[u8], o: usize) -> u32 {
        u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }

    /// The offset and length of the table `tag`.
    fn table(f: &[u8], tag: &[u8; 4]) -> (usize, usize) {
        (0..u16_at(f, 4) as usize)
            .map(|i| &f[12 + 16 * i..28 + 16 * i])
            .find(|rec| &rec[..4] == tag)
            .map(|rec| (u32_at(rec, 8) as usize, u32_at(rec, 12) as usize))
            .unwrap()
    }

    #[test]
    fn the_stand_in_is_a_well_formed_truetype_file() {
        let f = stand_in_font("kiln standard");
        assert_eq!(u32_at(&f, 0), 0x0001_0000);
        let n = u16_at(&f, 4) as usize;
        assert_eq!(n, 10);
        let mut previous = [0u8; 4];
        let mut head = None;
        for i in 0..n {
            let rec = &f[12 + 16 * i..28 + 16 * i];
            let tag = [rec[0], rec[1], rec[2], rec[3]];
            assert!(tag > previous, "tables are in tag order");
            previous = tag;
            let (sum, offset, len) = (
                u32_at(rec, 4),
                u32_at(rec, 8) as usize,
                u32_at(rec, 12) as usize,
            );
            assert_eq!(offset % 4, 0);
            let table = &f[offset..offset + len];
            if &tag == b"head" {
                head = Some(offset);
                // The stored sum was taken with the adjustment still zero.
                let mut zeroed = table.to_vec();
                zeroed[8..12].copy_from_slice(&[0; 4]);
                assert_eq!(sum, checksum(&zeroed));
            } else {
                assert_eq!(sum, checksum(table), "{}", String::from_utf8_lossy(&tag));
            }
        }
        // The whole file sums to the magic number the head table is chosen to produce.
        assert_eq!(checksum(&f), 0xB1B0_AFBA);
        let head = head.unwrap();
        assert_eq!(u32_at(&f, head + 12), 0x5F0F_3CF5);
        assert_eq!(u16_at(&f, head + 18), UNITS_PER_EM, "units per em");
    }

    #[test]
    fn the_tables_have_the_sizes_the_format_gives_them() {
        let f = stand_in_font("x");
        // Fixed sizes of version 1.0 tables.
        assert_eq!(table(&f, b"head").1, 54);
        assert_eq!(table(&f, b"hhea").1, 36);
        assert_eq!(table(&f, b"maxp").1, 32);
        assert_eq!(table(&f, b"post").1, 32);
        // Sizes that follow from the glyph count: short offsets, a full metric for `.notdef` and
        // every character up to the end of Latin-1, a left side bearing for each of the others.
        let (maxp, _) = table(&f, b"maxp");
        assert_eq!(u16_at(&f, maxp + 4) as usize, GLYPHS);
        assert_eq!(table(&f, b"loca").1, 2 * (GLYPHS + 1));
        let (hhea, _) = table(&f, b"hhea");
        let full = u16_at(&f, hhea + 34) as usize;
        assert_eq!(full, 1 + (LAST_METRIC_CHAR - FIRST_CHAR) as usize + 1);
        assert_eq!(table(&f, b"hmtx").1, 4 * full + 2 * (GLYPHS - full));
        let (head, _) = table(&f, b"head");
        assert_eq!(u16_at(&f, head + 50), 0, "short offsets in loca");
    }

    #[test]
    fn the_os2_table_gives_the_same_metrics_as_the_horizontal_header() {
        let f = stand_in_font("x");
        let (os2, len) = table(&f, b"OS/2");
        assert_eq!((u16_at(&f, os2), len), (4, 96), "version 4 is 96 bytes");
        let (hhea, _) = table(&f, b"hhea");
        let (hhea_ascent, hhea_descent) =
            (u16_at(&f, hhea + 4) as i16, u16_at(&f, hhea + 6) as i16);
        // Typographic ascender, descender and line gap, then the Windows ascent and descent.
        assert_eq!(u16_at(&f, os2 + 68) as i16, hhea_ascent);
        assert_eq!(u16_at(&f, os2 + 70) as i16, hhea_descent);
        assert_eq!(u16_at(&f, os2 + 72), 0);
        assert_eq!(u16_at(&f, os2 + 74) as i16, hhea_ascent);
        assert_eq!(u16_at(&f, os2 + 76) as i16, -hhea_descent);
        // The first and last characters it says it covers are the ones the cmap maps.
        assert_eq!(
            (u16_at(&f, os2 + 64), u16_at(&f, os2 + 66)),
            (FIRST_CHAR, LAST_CHAR)
        );
        // The metrics are not zero: a text engine divides by the height they give.
        assert!(hhea_ascent > 0 && hhea_descent < 0);
    }

    #[test]
    fn every_character_has_a_glyph_of_its_own_and_only_the_latin_ones_are_drawn() {
        let f = stand_in_font("x");
        let (cmap, _) = table(&f, b"cmap");
        let sub = cmap + u32_at(&f, cmap + 8) as usize;
        assert_eq!(u16_at(&f, sub), 4);
        let segs = u16_at(&f, sub + 6) as usize / 2;
        let lookup = |c: u16| -> u16 {
            for s in 0..segs {
                let end = u16_at(&f, sub + 14 + 2 * s);
                let start = u16_at(&f, sub + 16 + 2 * segs + 2 * s);
                let delta = u16_at(&f, sub + 16 + 4 * segs + 2 * s);
                if c <= end && c >= start {
                    return c.wrapping_add(delta);
                }
            }
            0
        };
        assert_eq!(lookup(0x0020), 1);
        assert_eq!(lookup(u16::from(b'A')), u16::from(b'A') - 0x1F);
        assert_eq!(lookup(0x3042), 0x3042 - 0x1F, "hiragana");
        assert_eq!(
            lookup(0xFF21),
            0xFF21 - 0x1F,
            "a fullwidth letter, high in the plane"
        );
        assert_eq!(lookup(0xFFFD), 0xFFFD - 0x1F, "the replacement character");
        assert_eq!(lookup(LAST_CHAR) as usize, GLYPHS - 1, "the last glyph");
        assert_eq!(lookup(0x001F), 0, "control characters have none");
        assert_eq!(lookup(0xFFFF), 0);
        // Every glyph starts where the last one ended; the ones of the keyboard have ink and the
        // space and the other scripts have none.
        let (loca, len) = table(&f, b"loca");
        let offset = |gid: usize| u16_at(&f, loca + 2 * gid);
        assert!((0..len / 2 - 1).all(|g| offset(g) <= offset(g + 1)));
        let drawn = |c: u16| offset(usize::from(c - 0x1F) + 1) > offset(usize::from(c - 0x1F));
        assert!((0x21..=0x7E).all(&drawn));
        assert!(!drawn(0x20) && !drawn(0x3042) && !drawn(0xFF21));
        // The name table holds the family name in UTF-16.
        let (name, _) = table(&f, b"name");
        let len = u16_at(&f, name + 14) as usize;
        assert_eq!(&f[name + 18..name + 18 + len], &[0, b'x']);
    }

    /// The glyph `gid` of the font `f`: its outline as contours of points, and its box.
    fn outline(f: &[u8], gid: usize) -> (Vec<Vec<(i32, i32)>>, [i16; 4]) {
        let (loca, _) = table(f, b"loca");
        let (glyf, _) = table(f, b"glyf");
        let start = glyf + 2 * u16_at(f, loca + 2 * gid) as usize;
        let end = glyf + 2 * u16_at(f, loca + 2 * gid + 2) as usize;
        assert!(end > start, "glyph {gid} has no outline");
        let g = &f[start..end];
        let n = u16_at(g, 0) as usize;
        let bounds = [2, 4, 6, 8].map(|o| u16_at(g, o) as i16);
        let ends: Vec<usize> = (0..n).map(|i| u16_at(g, 10 + 2 * i) as usize).collect();
        let points = ends.last().map_or(0, |e| e + 1);
        let mut at = 10 + 2 * n;
        assert_eq!(u16_at(g, at), 0, "no instructions");
        at += 2;
        assert!(
            g[at..at + points].iter().all(|&b| b == 1),
            "every point is on the curve"
        );
        at += points;
        let mut coords = [Vec::new(), Vec::new()];
        for axis in &mut coords {
            let mut v = 0i32;
            for _ in 0..points {
                v += i32::from(u16_at(g, at) as i16);
                at += 2;
                axis.push(v);
            }
        }
        assert!(g.len() - at < 2, "nothing follows but the padding");
        let mut first = 0;
        let contours = ends
            .iter()
            .map(|&e| {
                let c = (first..=e).map(|i| (coords[0][i], coords[1][i])).collect();
                first = e + 1;
                c
            })
            .collect();
        (contours, bounds)
    }

    /// Twice the signed area of a contour (negative when it runs clockwise).
    fn area2(c: &[(i32, i32)]) -> i64 {
        (0..c.len())
            .map(|i| {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                i64::from(a.0) * i64::from(b.1) - i64::from(b.0) * i64::from(a.1)
            })
            .sum()
    }

    #[test]
    fn a_drawn_glyph_is_a_set_of_clockwise_rectangles_with_its_box() {
        let f = stand_in_font("x");
        let (contours, bounds) = outline(&f, usize::from(b'A') - 0x1F);
        assert_eq!(bounds, [0, 0, 500, 700]);
        assert!(contours.iter().all(|c| c.len() == 4 && area2(c) < 0));
        // Eighteen squares of ink, a hundred font units on a side.
        let ink: i64 = contours.iter().map(|c| -area2(c) / 2).sum();
        assert_eq!(ink, 18 * 100 * 100);
        // The points of the contours are inside the box the glyph declares.
        for &(x, y) in contours.iter().flatten() {
            assert!((0..=500).contains(&x) && (0..=700).contains(&y));
        }
    }

    #[test]
    fn the_metrics_follow_the_drawn_glyphs() {
        let f = stand_in_font("x");
        let (hmtx, _) = table(&f, b"hmtx");
        let metric = |c: u16| {
            let at = hmtx + 4 * usize::from(c - 0x1F);
            (u16_at(&f, at), u16_at(&f, at + 2))
        };
        assert_eq!(metric(u16::from(b'A')), (600, 0));
        assert_eq!(
            metric(u16::from(b'I')),
            (600, 100),
            "an I has a bearing: it is centered"
        );
        assert_eq!(metric(u16::from(b'i')), (400, 0));
        assert_eq!(metric(0x20), (400, 0), "the space");
        // The last full metric is the advance every later glyph has.
        assert_eq!(metric(LAST_METRIC_CHAR).0, glyphs::DEFAULT_ADVANCE);
        // The box and the extremes of the horizontal header cover the ink: the widest glyph is the
        // em dash (seven squares and the room after it).
        let (head, _) = table(&f, b"head");
        assert_eq!(
            [36, 38, 40, 42].map(|o| u16_at(&f, head + o) as i16),
            [0, -200, 700, 700]
        );
        let (hhea, _) = table(&f, b"hhea");
        assert_eq!(u16_at(&f, hhea + 10), 800, "widest advance");
        assert_eq!(u16_at(&f, hhea + 16), 700, "largest extent");
        // The profile gives the most points and contours any glyph needs.
        let (maxp, _) = table(&f, b"maxp");
        let (points, contours) = (u16_at(&f, maxp + 6), u16_at(&f, maxp + 8));
        assert_eq!(points, 4 * contours);
        assert!((1..=20).contains(&contours), "{contours}");
        assert!((0x21u16..=0x7E).all(|c| {
            let (cs, _) = outline(&f, usize::from(c - 0x1F));
            cs.len() <= usize::from(contours)
        }));
    }

    #[test]
    fn every_drawn_character_has_its_outline_and_its_metric_in_the_file() {
        let f = stand_in_font("x");
        let (hmtx, _) = table(&f, b"hmtx");
        let mut drawn = 0;
        for code in FIRST_CHAR..=LAST_METRIC_CHAR {
            let g = glyphs::glyph(code);
            let gid = usize::from(code - 0x1F);
            assert_eq!(
                u16_at(&f, hmtx + 4 * gid),
                g.advance,
                "advance of {code:#x}"
            );
            if !g.rects.is_empty() {
                let (contours, bounds) = outline(&f, gid);
                assert_eq!(contours.len(), g.rects.len(), "{code:#x}");
                assert_eq!(bounds, g.bounds(), "{code:#x}");
                assert_eq!(
                    i16::from_be_bytes([f[hmtx + 4 * gid + 2], f[hmtx + 4 * gid + 3]]),
                    bounds[0]
                );
                drawn += 1;
            }
        }
        // The keyboard's characters, the curly quotes, the dashes, the bullet and the ellipsis.
        assert!(drawn >= 94 + 7, "{drawn}");
        assert!(drawn <= 200, "{drawn}");
    }

    #[test]
    fn each_type_has_its_own_slot() {
        let end = |i: usize| font_offset(i) as usize + fonts()[i].len();
        for i in 0..FONT_TYPES - 1 {
            assert!(end(i) <= font_offset(i + 1) as usize, "font {i}");
        }
        assert!((end(FONT_TYPES - 1) as u64) < FONT_MEMORY_SIZE);
    }
}
