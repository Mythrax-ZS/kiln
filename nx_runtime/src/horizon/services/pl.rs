//! `pl:u`, the platform service that shares the system fonts (Switchbrew, "Shared Database
//! services", the font section).
//!
//! The console's system fonts are firmware data, which this project does not have and does not
//! contain. What the service shares here is a stand-in made by this code: six TrueType fonts (one
//! per `SharedFontType`) that have a name, metrics, and a glyph for every character of the Basic
//! Multilingual Plane, and no outline for any of them, so text drawn with them is blank
//! (docs/ASSUMPTIONS.md A-073). They sit in a shared memory of 0x1100000 bytes at the offsets
//! `GetSharedMemoryAddressOffset` reports.

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
const ADVANCE: u16 = 512;

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

/// A TrueType font named `name` with `GLYPHS` glyphs, all of them empty, and a format 4 `cmap`
/// that gives every character from U+0020 to U+FFFE its own glyph.
pub fn stand_in_font(name: &str) -> Vec<u8> {
    let head = {
        let mut t = Vec::new();
        t.extend(be32(0x0001_0000)); // version
        t.extend(be32(0x0001_0000)); // font revision
        t.extend(be32(0)); // checksum adjustment, filled in below
        t.extend(be32(0x5F0F_3CF5)); // magic
        t.extend(be16(0x000B)); // flags
        t.extend(be16(UNITS_PER_EM));
        t.extend([0u8; 16]); // created, modified
        t.extend(be16(0)); // x min
        t.extend(be16(DESCENT as u16)); // y min
        t.extend(be16(ADVANCE)); // x max
        t.extend(be16(ASCENT as u16)); // y max
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
        t.extend(be16(ADVANCE)); // widest advance
        t.extend(be16(0)); // min left side bearing
        t.extend(be16(0)); // min right side bearing
        t.extend(be16(ADVANCE)); // largest extent
        t.extend(be16(1)); // caret slope rise
        t.extend(be16(0)); // caret slope run
        t.extend(be16(0)); // caret offset
        t.extend([0u8; 8]); // reserved
        t.extend(be16(0)); // metric data format
        t.extend(be16(1)); // one full metric, the rest only repeat the left side bearing
        t
    };
    let maxp = {
        let mut t = Vec::new();
        t.extend(be32(0x0001_0000));
        t.extend(be16(GLYPHS as u16));
        t.extend(be16(0)); // max points
        t.extend(be16(0)); // max contours
        t.extend(be16(0)); // max composite points
        t.extend(be16(0)); // max composite contours
        t.extend(be16(1)); // zones
                           // twilight points, storage, function defs, instruction defs, stack elements, size of
                           // instructions, component elements, component depth
        t.extend([0u8; 16]);
        t
    };
    // One advance for all glyphs, then a left side bearing (zero) for each of the others.
    let hmtx = [be16(ADVANCE), be16(0)]
        .concat()
        .into_iter()
        .chain(std::iter::repeat_n(0u8, 2 * (GLYPHS - 1)))
        .collect::<Vec<u8>>();
    // Every glyph starts and ends at offset 0 of the glyph table: they have no data.
    let loca = vec![0u8; 2 * (GLYPHS + 1)];
    let glyf = vec![0u8; 4];
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

    // Tables in tag order, each padded to four bytes.
    let tables: [(&[u8; 4], &Vec<u8>); 9] = [
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
    out.extend(be16(128)); // search range: 16 times the largest power of two <= count
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
    "recomp standard",
    "recomp chinese simplified",
    "recomp extended chinese simplified",
    "recomp chinese traditional",
    "recomp korean",
    "recomp nintendo extended",
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
        let f = stand_in_font("recomp standard");
        assert_eq!(u32_at(&f, 0), 0x0001_0000);
        let n = u16_at(&f, 4) as usize;
        assert_eq!(n, 9);
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
        // Sizes that follow from the glyph count: short offsets, one full metric.
        let (maxp, _) = table(&f, b"maxp");
        assert_eq!(u16_at(&f, maxp + 4) as usize, GLYPHS);
        assert_eq!(table(&f, b"loca").1, 2 * (GLYPHS + 1));
        let (hhea, _) = table(&f, b"hhea");
        assert_eq!(u16_at(&f, hhea + 34), 1, "number of full metrics");
        assert_eq!(table(&f, b"hmtx").1, 4 + 2 * (GLYPHS - 1));
        let (head, _) = table(&f, b"head");
        assert_eq!(u16_at(&f, head + 50), 0, "short offsets in loca");
    }

    #[test]
    fn every_character_has_a_glyph_of_its_own_and_none_has_an_outline() {
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
        // The glyph table is empty and every glyph starts where the last one ended.
        let (loca, len) = table(&f, b"loca");
        assert!(f[loca..loca + len].iter().all(|&b| b == 0));
        // The name table holds the family name in UTF-16.
        let (name, _) = table(&f, b"name");
        let len = u16_at(&f, name + 14) as usize;
        assert_eq!(&f[name + 18..name + 18 + len], &[0, b'x']);
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
