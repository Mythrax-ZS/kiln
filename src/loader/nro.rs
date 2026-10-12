//! NRO0 executables (Switchbrew NRO0 page): a flat image whose first 0x10 bytes are the module
//! header location, followed by the 0x70-byte NRO header, text, rodata and data at their memory
//! offsets. Homebrew may append an `ASET` section holding an icon, NACP and an embedded RomFS.

use super::module::{Module, ModuleKind, ModuleParts, Perm, Segment, MAX_IMAGE, PAGE};
use crate::{Error, Result};
use nx_runtime::util::{align_up, array_at, u32_at, u64_at};

pub const MAGIC: &[u8; 4] = b"NRO0";
const MAGIC_AT: usize = 0x10;
const HEADER_END: usize = 0x80;
const NAMES: [&str; 3] = ["text", "rodata", "data"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NroHeader {
    pub version: u32,
    /// Size of the NRO proper; the optional asset section starts here.
    pub size: u32,
    pub flags: u32,
    /// (memory offset, size) of text, rodata and data.
    pub segs: [(u32, u32); 3],
    pub bss_size: u32,
    pub build_id: [u8; 32],
}

/// (file offset, size) of each asset, or `None` when the section is empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assets {
    pub icon: Option<(u64, u64)>,
    pub nacp: Option<(u64, u64)>,
    pub romfs: Option<(u64, u64)>,
}

#[derive(Debug)]
pub struct Loaded {
    pub module: Module,
    pub header: NroHeader,
    pub assets: Option<Assets>,
}

impl NroHeader {
    pub fn parse(bytes: &[u8]) -> Result<NroHeader> {
        if bytes.len() < HEADER_END {
            return Err(Error::format("NRO", "shorter than its header"));
        }
        if &bytes[MAGIC_AT..MAGIC_AT + 4] != MAGIC {
            return Err(Error::format("NRO", "bad magic"));
        }
        let seg = |at: usize| (u32_at(bytes, at), u32_at(bytes, at + 4));
        Ok(NroHeader {
            version: u32_at(bytes, 0x14),
            size: u32_at(bytes, 0x18),
            flags: u32_at(bytes, 0x1C),
            segs: [seg(0x20), seg(0x28), seg(0x30)],
            bss_size: u32_at(bytes, 0x38),
            build_id: array_at(bytes, 0x40),
        })
    }
}

fn parse_assets(bytes: &[u8], at: usize) -> Result<Option<Assets>> {
    if bytes.len() < at + 0x38 || &bytes[at..at + 4] != b"ASET" {
        return Ok(None);
    }
    let entry = |rel: usize, what: &str| -> Result<Option<(u64, u64)>> {
        let off = u64_at(bytes, at + rel);
        let size = u64_at(bytes, at + rel + 8);
        if size == 0 {
            return Ok(None);
        }
        let start = (at as u64).checked_add(off);
        match start.and_then(|s| s.checked_add(size).map(|e| (s, e))) {
            Some((s, e)) if e <= bytes.len() as u64 => Ok(Some((s, size))),
            _ => Err(Error::format(
                "NRO asset section",
                format!("{what} lies outside the file"),
            )),
        }
    };
    Ok(Some(Assets {
        icon: entry(0x08, "icon")?,
        nacp: entry(0x18, "NACP")?,
        romfs: entry(0x28, "RomFS")?,
    }))
}

/// Parse and relocate an NRO mapped at `base`.
pub fn load(bytes: &[u8], name: &str, base: u64) -> Result<Loaded> {
    let h = NroHeader::parse(bytes)?;
    let total = h.size as usize;
    if total < HEADER_END || total > bytes.len() {
        return Err(Error::format(
            "NRO",
            "header size is inconsistent with the file",
        ));
    }
    let end = |i: usize| u64::from(h.segs[i].0) + u64::from(h.segs[i].1);
    if h.segs[0].0 != 0 || (h.segs[0].1 as usize) < HEADER_END {
        return Err(Error::format(
            "NRO",
            "text must start at offset 0 and hold the header",
        ));
    }
    for (i, name) in NAMES.iter().enumerate() {
        if !u64::from(h.segs[i].0).is_multiple_of(PAGE) {
            return Err(Error::format(
                "NRO",
                format!("{name} segment is not page aligned"),
            ));
        }
        if end(i) > total as u64 {
            return Err(Error::format(
                "NRO",
                format!("{name} segment runs past the NRO"),
            ));
        }
    }
    for i in 0..2 {
        if end(i) > u64::from(h.segs[i + 1].0) {
            return Err(Error::format(
                "NRO",
                format!("{} segment overlaps the {} segment", NAMES[i], NAMES[i + 1]),
            ));
        }
    }
    let data_end = end(2);
    let mapped = align_up(data_end + u64::from(h.bss_size), PAGE);
    if mapped > MAX_IMAGE {
        return Err(Error::format("NRO", "mapped size is implausibly large"));
    }
    let mut image = vec![0u8; mapped as usize];
    for i in 0..3 {
        let (off, size) = (h.segs[i].0 as usize, h.segs[i].1 as usize);
        image[off..off + size].copy_from_slice(&bytes[off..off + size]);
    }
    let perms = [Perm::RX, Perm::R, Perm::RW];
    let mut segments: Vec<Segment> = (0..3)
        .map(|i| Segment {
            name: NAMES[i],
            offset: u64::from(h.segs[i].0),
            size: align_up(u64::from(h.segs[i].1), PAGE),
            perm: perms[i],
        })
        .collect();
    let bss_start = align_up(data_end, PAGE);
    if mapped > bss_start {
        segments.push(Segment {
            name: "bss",
            offset: bss_start,
            size: mapped - bss_start,
            perm: Perm::RW,
        });
    }
    let assets = parse_assets(bytes, total)?;
    let module = Module::from_parts(ModuleParts {
        kind: ModuleKind::Nro,
        name: name.to_string(),
        base,
        image,
        segments,
        build_id: h.build_id,
        dynsym: None,
    })?;
    Ok(Loaded {
        module,
        header: h,
        assets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::module::{GUEST_BASE, R_AARCH64_ABS64};
    use crate::loader::testing::*;

    fn spec() -> Spec {
        Spec {
            code: vec![0xD65F03C0; 3],
            with_mod0: true,
            relatives: vec![(SLOTS_OFF as u64, 0x4242)],
            symbol_relocs: vec![(SLOTS_OFF as u64 + 8, "entry", R_AARCH64_ABS64, 4)],
            exports: vec![("entry", 0x180)],
            init_array: vec![0x190],
            ..Spec::default()
        }
    }

    #[test]
    fn loads_relocates_and_lays_out() {
        let file = build_nro(&build_parts(&spec()), BSS_SIZE, None);
        let l = load(&file, "hello", GUEST_BASE).unwrap();
        let m = &l.module;
        assert_eq!(m.kind, ModuleKind::Nro);
        assert_eq!(m.image.len(), 0x5000);
        assert_eq!(l.header.size, 0x3000);
        assert_eq!(l.header.bss_size, BSS_SIZE);
        assert_eq!(m.build_id[1], 0xB2);
        let slot = |o: usize| u64::from_le_bytes(m.image[o..o + 8].try_into().unwrap());
        assert_eq!(slot(SLOTS_OFF), GUEST_BASE + 0x4242);
        assert_eq!(slot(SLOTS_OFF + 8), GUEST_BASE + 0x180 + 4);
        assert_eq!(m.init_array, [GUEST_BASE + 0x190]);
        assert!(m.image[0x3000..].iter().all(|&b| b == 0));
        assert!(l.assets.is_none());
        let segs: Vec<_> = m
            .segments
            .iter()
            .map(|s| (s.name, s.offset, s.size))
            .collect();
        assert_eq!(segs[3], ("bss", 0x3000, 0x2000));
        // the header bytes stay in the text segment, as the console maps them
        assert_eq!(&m.image[0x10..0x14], b"NRO0");
    }

    #[test]
    fn parses_the_asset_section() {
        let assets = AssetSpec {
            icon: b"JPEGDATA".to_vec(),
            nacp: vec![7u8; 0x40],
            romfs: b"ROMFSIMAGE".to_vec(),
        };
        let file = build_nro(&build_parts(&spec()), BSS_SIZE, Some(&assets));
        let l = load(&file, "hello", GUEST_BASE).unwrap();
        let a = l.assets.unwrap();
        let take = |r: Option<(u64, u64)>| {
            let (o, s) = r.unwrap();
            file[o as usize..(o + s) as usize].to_vec()
        };
        assert_eq!(take(a.icon), b"JPEGDATA");
        assert_eq!(take(a.nacp), vec![7u8; 0x40]);
        assert_eq!(take(a.romfs), b"ROMFSIMAGE");

        let empty = AssetSpec {
            icon: Vec::new(),
            nacp: Vec::new(),
            romfs: b"x".to_vec(),
        };
        let file = build_nro(&build_parts(&spec()), BSS_SIZE, Some(&empty));
        let a = load(&file, "hello", GUEST_BASE).unwrap().assets.unwrap();
        assert!(a.icon.is_none() && a.nacp.is_none() && a.romfs.is_some());

        // an asset that runs past the file is rejected
        let mut bad = build_nro(&build_parts(&spec()), BSS_SIZE, Some(&assets));
        let aset = 0x3000;
        bad[aset + 0x28 + 8..aset + 0x28 + 16].copy_from_slice(&0x10_0000u64.to_le_bytes());
        assert!(load(&bad, "hello", GUEST_BASE).is_err());
    }

    fn put32(file: &mut [u8], at: usize, v: u32) {
        file[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn rejects_malformed_files() {
        let good = build_nro(&build_parts(&spec()), BSS_SIZE, None);
        let try_load = |f: &[u8]| load(f, "x", GUEST_BASE);
        assert!(try_load(&good[..0x40]).is_err());
        let mut f = good.clone();
        f[0x10] = b'X';
        assert!(try_load(&f).unwrap_err().to_string().contains("bad magic"));

        let mut f = good.clone();
        put32(&mut f, 0x18, 0x100_0000); // size larger than the file
        assert!(try_load(&f).is_err());
        let mut f = good.clone();
        put32(&mut f, 0x18, 0x40); // size smaller than the header
        assert!(try_load(&f).is_err());

        let mut f = good.clone();
        put32(&mut f, 0x20, 0x1000); // text not at offset 0
        assert!(try_load(&f).is_err());
        let mut f = good.clone();
        put32(&mut f, 0x28, 0x1010); // unaligned rodata
        assert!(try_load(&f)
            .unwrap_err()
            .to_string()
            .contains("page aligned"));
        let mut f = good.clone();
        put32(&mut f, 0x30, 0x1000); // data overlaps rodata
        assert!(try_load(&f).unwrap_err().to_string().contains("overlaps"));
        let mut f = good.clone();
        put32(&mut f, 0x34, 0x10_0000); // data past the NRO
        assert!(try_load(&f).is_err());
        let mut f = good.clone();
        put32(&mut f, 0x38, u32::MAX); // absurd bss
        assert!(try_load(&f).is_err());
    }

    #[test]
    fn aarch32_modules_are_rejected() {
        let mut parts = build_parts(&spec());
        parts.text[0..4].copy_from_slice(&0xEB00_0003u32.to_le_bytes()); // A32 bl
        let f = build_nro(&parts, BSS_SIZE, None);
        assert!(matches!(load(&f, "x", GUEST_BASE), Err(Error::Aarch32(_))));
    }
}
