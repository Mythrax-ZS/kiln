//! NSO0 executables (Switchbrew NSO0 page): a 0x100-byte header describing `.text`, `.rodata` and
//! `.data`, each optionally LZ4-compressed and optionally checked against a SHA-256 of the
//! decompressed bytes, followed by the segments themselves.

use super::lz4;
use super::module::{Module, ModuleKind, ModuleParts, Perm, Segment, MAX_IMAGE, PAGE};
use crate::{Error, Result};
use nx_runtime::util::{align_up, array_at, u32_at};
use sha2::{Digest, Sha256};

pub const MAGIC: &[u8; 4] = b"NSO0";
pub const HEADER_SIZE: usize = 0x100;
const NAMES: [&str; 3] = ["text", "rodata", "data"];
const FLAG_COMPRESS: [u32; 3] = [1 << 0, 1 << 1, 1 << 2];
const FLAG_HASH: [u32; 3] = [1 << 3, 1 << 4, 1 << 5];
/// `[22.0.0+] UseZbicCompression`: zstd-based compression instead of LZ4.
const FLAG_ZBIC: u32 = 1 << 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegHeader {
    pub file_offset: u32,
    pub mem_offset: u32,
    pub size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsoHeader {
    pub version: u32,
    pub flags: u32,
    pub segs: [SegHeader; 3],
    pub module_name_offset: u32,
    pub module_name_size: u32,
    pub bss_size: u32,
    pub module_id: [u8; 32],
    /// Stored (compressed) sizes; meaningful only for segments with the compress flag.
    pub file_sizes: [u32; 3],
    pub embedded: (u32, u32),
    pub dynstr: (u32, u32),
    pub dynsym: (u32, u32),
    pub hashes: [[u8; 32]; 3],
}

impl NsoHeader {
    pub fn compressed(&self, i: usize) -> bool {
        self.flags & FLAG_COMPRESS[i] != 0
    }

    pub fn hash_checked(&self, i: usize) -> bool {
        self.flags & FLAG_HASH[i] != 0
    }

    pub fn parse(bytes: &[u8]) -> Result<NsoHeader> {
        if bytes.len() < HEADER_SIZE {
            return Err(Error::format("NSO", "shorter than its header"));
        }
        if &bytes[..4] != MAGIC {
            return Err(Error::format("NSO", "bad magic"));
        }
        let seg = |at: usize| SegHeader {
            file_offset: u32_at(bytes, at),
            mem_offset: u32_at(bytes, at + 4),
            size: u32_at(bytes, at + 8),
        };
        // The text, rodata and data records are interleaved with the module name fields.
        let segs = [seg(0x10), seg(0x20), seg(0x30)];
        Ok(NsoHeader {
            version: u32_at(bytes, 4),
            flags: u32_at(bytes, 0x0C),
            segs: [
                SegHeader {
                    size: u32_at(bytes, 0x18),
                    ..segs[0]
                },
                SegHeader {
                    size: u32_at(bytes, 0x28),
                    ..segs[1]
                },
                SegHeader {
                    size: u32_at(bytes, 0x38),
                    ..segs[2]
                },
            ],
            module_name_offset: u32_at(bytes, 0x1C),
            module_name_size: u32_at(bytes, 0x2C),
            bss_size: u32_at(bytes, 0x3C),
            module_id: array_at(bytes, 0x40),
            file_sizes: [
                u32_at(bytes, 0x60),
                u32_at(bytes, 0x64),
                u32_at(bytes, 0x68),
            ],
            embedded: (u32_at(bytes, 0x88), u32_at(bytes, 0x8C)),
            dynstr: (u32_at(bytes, 0x90), u32_at(bytes, 0x94)),
            dynsym: (u32_at(bytes, 0x98), u32_at(bytes, 0x9C)),
            hashes: [
                array_at(bytes, 0xA0),
                array_at(bytes, 0xC0),
                array_at(bytes, 0xE0),
            ],
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Check the SHA-256 of every segment whose header flag asks for it.
    pub verify_hashes: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            verify_hashes: true,
        }
    }
}

/// A loaded NSO together with what the header said.
#[derive(Debug)]
pub struct Loaded {
    pub module: Module,
    pub header: NsoHeader,
    /// Which segments had their SHA-256 checked and matched.
    pub hash_verified: [bool; 3],
}

fn segment_bytes(bytes: &[u8], h: &NsoHeader, i: usize) -> Result<Vec<u8>> {
    let seg = h.segs[i];
    let stored = if h.compressed(i) {
        h.file_sizes[i]
    } else {
        seg.size
    };
    let start = seg.file_offset as usize;
    let end = start
        .checked_add(stored as usize)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| {
            Error::format("NSO", format!("{} segment lies outside the file", NAMES[i]))
        })?;
    let raw = &bytes[start..end];
    if h.compressed(i) {
        lz4::decompress(raw, seg.size as usize)
            .map_err(|e| Error::format("NSO", format!("{} segment: {e}", NAMES[i])))
    } else {
        Ok(raw.to_vec())
    }
}

/// Parse, decompress, verify, lay out and relocate an NSO mapped at `base`.
pub fn load(bytes: &[u8], name: &str, base: u64, opts: &LoadOptions) -> Result<Loaded> {
    let h = NsoHeader::parse(bytes)?;
    if h.flags & FLAG_ZBIC != 0 {
        return Err(Error::unsupported(
            "NSO",
            "zstd-based (ZBIC) segment compression",
        ));
    }
    let mut data: Vec<Vec<u8>> = Vec::with_capacity(3);
    let mut verified = [false; 3];
    for i in 0..3 {
        let seg = h.segs[i];
        if !u64::from(seg.mem_offset).is_multiple_of(PAGE) {
            return Err(Error::format(
                "NSO",
                format!("{} segment is not page aligned", NAMES[i]),
            ));
        }
        if u64::from(seg.size) > MAX_IMAGE {
            return Err(Error::format(
                "NSO",
                format!("{} segment is implausibly large", NAMES[i]),
            ));
        }
        let bytes_i = segment_bytes(bytes, &h, i)?;
        if h.hash_checked(i) && opts.verify_hashes {
            if Sha256::digest(&bytes_i).as_slice() != h.hashes[i] {
                return Err(Error::integrity(
                    "NSO",
                    format!(
                        "SHA-256 of the {} segment does not match the header",
                        NAMES[i]
                    ),
                ));
            }
            verified[i] = true;
        }
        data.push(bytes_i);
    }
    let end = |i: usize| u64::from(h.segs[i].mem_offset) + u64::from(h.segs[i].size);
    for i in 0..2 {
        if end(i) > u64::from(h.segs[i + 1].mem_offset) {
            return Err(Error::format(
                "NSO",
                format!("{} segment overlaps the {} segment", NAMES[i], NAMES[i + 1]),
            ));
        }
    }
    let data_end = end(2);
    let mapped = align_up(data_end + u64::from(h.bss_size), PAGE);
    if mapped > MAX_IMAGE {
        return Err(Error::format("NSO", "mapped size is implausibly large"));
    }
    let mut image = vec![0u8; mapped as usize];
    for i in 0..3 {
        let at = h.segs[i].mem_offset as usize;
        image[at..at + data[i].len()].copy_from_slice(&data[i]);
    }
    let perms = [Perm::RX, Perm::R, Perm::RW];
    let mut segments: Vec<Segment> = (0..3)
        .map(|i| Segment {
            name: NAMES[i],
            offset: u64::from(h.segs[i].mem_offset),
            size: align_up(u64::from(h.segs[i].size), PAGE),
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
    let dynsym = (h.dynsym.1 > 0).then(|| {
        (
            u64::from(h.segs[1].mem_offset) + u64::from(h.dynsym.0),
            u64::from(h.dynsym.1),
        )
    });
    let module = Module::from_parts(ModuleParts {
        kind: ModuleKind::Nso,
        name: name.to_string(),
        base,
        image,
        segments,
        build_id: h.module_id,
        dynsym,
    })?;
    Ok(Loaded {
        module,
        header: h,
        hash_verified: verified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::module::{
        UnresolvedReloc, UnsupportedReloc, GUEST_BASE, R_AARCH64_ABS64, R_AARCH64_GLOB_DAT,
        R_AARCH64_JUMP_SLOT,
    };
    use crate::loader::testing::*;

    fn simple_spec() -> Spec {
        Spec {
            code: vec![0xD65F03C0; 4], // ret
            with_mod0: true,
            ..Spec::default()
        }
    }

    fn load_ok(bytes: &[u8]) -> Loaded {
        load(bytes, "main", GUEST_BASE, &LoadOptions::default()).unwrap()
    }

    #[test]
    fn segments_decode_identically_with_and_without_compression() {
        let parts = build_parts(&simple_spec());
        let mut reference: Option<Vec<u8>> = None;
        for compress in 0..8u32 {
            for hash in [0u32, 7] {
                let flags = NsoFlags {
                    compress: [compress & 1 != 0, compress & 2 != 0, compress & 4 != 0],
                    hash: [hash & 1 != 0, hash & 2 != 0, hash & 4 != 0],
                };
                let file = build_nso(&parts, flags);
                let l = load_ok(&file);
                assert_eq!(
                    l.hash_verified, flags.hash,
                    "compress {compress} hash {hash}"
                );
                match &reference {
                    None => reference = Some(l.module.image.clone()),
                    Some(r) => assert_eq!(&l.module.image, r, "compress {compress} hash {hash}"),
                }
            }
        }
        let img = reference.unwrap();
        assert_eq!(&img[CODE_OFF..CODE_OFF + 4], &0xD65F03C0u32.to_le_bytes());
    }

    #[test]
    fn layout_segments_and_bss() {
        let l = load_ok(&build_nso(
            &build_parts(&simple_spec()),
            NsoFlags::default(),
        ));
        let m = &l.module;
        assert_eq!(m.kind, ModuleKind::Nso);
        assert_eq!(m.image.len(), 0x5000);
        let names: Vec<_> = m
            .segments
            .iter()
            .map(|s| (s.name, s.offset, s.size))
            .collect();
        assert_eq!(
            names,
            [
                ("text", 0, 0x1000),
                ("rodata", 0x1000, 0x1000),
                ("data", 0x2000, 0x1000),
                ("bss", 0x3000, 0x2000)
            ]
        );
        assert_eq!(m.segment("text").unwrap().perm, Perm::RX);
        assert_eq!(m.segment("rodata").unwrap().perm, Perm::R);
        assert_eq!(m.segment("bss").unwrap().perm, Perm::RW);
        assert!(
            m.image[0x3000..].iter().all(|&b| b == 0),
            "bss must be zero"
        );
        assert_eq!(m.entry(), GUEST_BASE);
        assert_eq!(m.offset_of(GUEST_BASE + 0x4FFF), Some(0x4FFF));
        assert_eq!(m.offset_of(GUEST_BASE + 0x5000), None);
        assert_eq!(m.offset_of(GUEST_BASE - 1), None);
        assert_eq!(m.segment_at(0x2800).unwrap().name, "data");
        assert_eq!(m.segment_bytes("text").unwrap().len(), 0x1000);
        assert_eq!(l.header.bss_size, BSS_SIZE);
        assert_eq!(m.build_id[0], 0xB1);
        let mod0 = m.mod0.as_ref().unwrap();
        assert_eq!(mod0.header_offset, MOD0_OFF as u64);
        assert_eq!(mod0.dynamic, DYNAMIC_OFF as u64);
        assert_eq!(mod0.bss_start, 0x3000);
        assert_eq!(mod0.bss_end, 0x3000 + u64::from(BSS_SIZE));
    }

    fn slot(img: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(img[off..off + 8].try_into().unwrap())
    }

    #[test]
    fn relocations_are_applied_relative_to_the_fixed_base() {
        let spec = Spec {
            relatives: vec![(SLOTS_OFF as u64, 0x1234), (SLOTS_OFF as u64 + 8, -8)],
            symbol_relocs: vec![
                (SLOTS_OFF as u64 + 16, "foo", R_AARCH64_ABS64, 8),
                (SLOTS_OFF as u64 + 24, "foo", R_AARCH64_GLOB_DAT, 0),
            ],
            jmprel: vec![(SLOTS_OFF as u64 + 32, "foo")],
            exports: vec![("foo", 0x400), ("bar", 0x500)],
            ..simple_spec()
        };
        for base in [GUEST_BASE, 0x1_0000_0000] {
            let file = build_nso(&build_parts(&spec), NsoFlags::default());
            let l = load(&file, "main", base, &LoadOptions::default()).unwrap();
            let img = &l.module.image;
            assert_eq!(slot(img, SLOTS_OFF), base + 0x1234);
            assert_eq!(slot(img, SLOTS_OFF + 8), base - 8);
            assert_eq!(slot(img, SLOTS_OFF + 16), base + 0x400 + 8);
            assert_eq!(slot(img, SLOTS_OFF + 24), base + 0x400);
            assert_eq!(slot(img, SLOTS_OFF + 32), base + 0x400);
            let r = &l.module.relocs;
            assert_eq!((r.relative, r.abs64, r.glob_dat, r.jump_slot), (2, 1, 1, 1));
            assert_eq!(r.applied(), 5);
            assert!(r.unresolved.is_empty() && r.unsupported.is_empty());
        }
        // Different bases produce different relocated images but the same unrelocated layout.
        let _ = R_AARCH64_JUMP_SLOT;
    }

    #[test]
    fn the_unrelocated_image_is_what_the_file_holds() {
        let spec = Spec {
            relatives: vec![(SLOTS_OFF as u64, 0x1234)],
            symbol_relocs: vec![(SLOTS_OFF as u64 + 8, "foo", R_AARCH64_ABS64, 8)],
            jmprel: vec![(SLOTS_OFF as u64 + 16, "foo")],
            imports: vec![("memcpy", false)],
            exports: vec![("foo", 0x400)],
            ..simple_spec()
        };
        let mut parts = build_parts(&spec);
        // Slots hold junk in the file; one more relocation (an unresolved import) zeroes its slot.
        let junk = SLOTS_OFF - DATA_OFF;
        parts.data[junk..junk + 32].fill(0xEE);
        let l = load_ok(&build_nso(&parts, NsoFlags::default()));
        let m = &l.module;
        assert_ne!(slot(&m.image, SLOTS_OFF), 0xEEEE_EEEE_EEEE_EEEE);
        let raw = m.unrelocated_image();
        assert_eq!(raw.len(), m.image.len());
        for i in 0..3 {
            assert_eq!(slot(&raw, SLOTS_OFF + 8 * i), 0xEEEE_EEEE_EEEE_EEEE, "{i}");
        }
        // Everywhere else the two images agree.
        let relocated = [SLOTS_OFF, SLOTS_OFF + 8, SLOTS_OFF + 16];
        for (at, (a, b)) in m.image.iter().zip(&raw).enumerate() {
            if !relocated.iter().any(|&s| (s..s + 8).contains(&at)) {
                assert_eq!(a, b, "byte {at:#x}");
            }
        }
        // The text the analysis reads is the same in both.
        assert_eq!(&m.image[..0x1000], &raw[..0x1000]);
    }

    #[test]
    fn init_fini_arrays_exports_and_known_entries() {
        let spec = Spec {
            exports: vec![("nnMain", 0x200), ("helper", 0x240)],
            init_array: vec![0x300, 0x310],
            fini_array: vec![0x320],
            init_fn: Some(0x330),
            fini_fn: Some(0x340),
            ..simple_spec()
        };
        let l = load_ok(&build_nso(&build_parts(&spec), NsoFlags::default()));
        let m = &l.module;
        let names: Vec<(&str, u64)> = m
            .exports
            .iter()
            .map(|e| (e.name.as_str(), e.offset))
            .collect();
        assert_eq!(names, [("nnMain", 0x200), ("helper", 0x240)]);
        assert_eq!(m.init_array, [GUEST_BASE + 0x300, GUEST_BASE + 0x310]);
        assert_eq!(m.fini_array, [GUEST_BASE + 0x320]);
        assert_eq!(m.init_fn, Some(GUEST_BASE + 0x330));
        assert_eq!(m.fini_fn, Some(GUEST_BASE + 0x340));
        let entries = m.known_entries();
        for off in [0u64, 0x200, 0x240, 0x300, 0x310, 0x320, 0x330, 0x340] {
            assert!(entries.contains(&(GUEST_BASE + off)), "{off:#x}");
        }
        assert!(entries.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
    }

    #[test]
    fn unresolved_imports_are_reported_and_weak_ones_resolve_to_zero() {
        let spec = Spec {
            imports: vec![("memcpy", false), ("__gmon_start__", true)],
            symbol_relocs: vec![
                (SLOTS_OFF as u64, "memcpy", R_AARCH64_GLOB_DAT, 0),
                (
                    SLOTS_OFF as u64 + 8,
                    "__gmon_start__",
                    R_AARCH64_GLOB_DAT,
                    0,
                ),
            ],
            ..simple_spec()
        };
        let mut file = build_parts(&spec);
        // pre-fill both slots with junk to prove the loader zeroes them
        file.data[SLOTS_OFF - DATA_OFF..SLOTS_OFF - DATA_OFF + 16].fill(0xEE);
        let l = load_ok(&build_nso(&file, NsoFlags::default()));
        let r = &l.module.relocs;
        assert_eq!(
            r.unresolved,
            [UnresolvedReloc {
                offset: SLOTS_OFF as u64,
                symbol: "memcpy".to_string()
            }]
        );
        assert_eq!(slot(&l.module.image, SLOTS_OFF), 0);
        assert_eq!(slot(&l.module.image, SLOTS_OFF + 8), 0);
    }

    #[test]
    fn unsupported_relocation_types_are_listed_not_applied() {
        let spec = Spec {
            symbol_relocs: vec![(SLOTS_OFF as u64, "", 1030, 0)], // R_AARCH64_TLS_TPREL64
            ..simple_spec()
        };
        let mut parts = build_parts(&spec);
        parts.data[SLOTS_OFF - DATA_OFF..SLOTS_OFF - DATA_OFF + 8].fill(0x77);
        let l = load_ok(&build_nso(&parts, NsoFlags::default()));
        assert_eq!(
            l.module.relocs.unsupported,
            [UnsupportedReloc {
                offset: SLOTS_OFF as u64,
                r_type: 1030
            }]
        );
        assert_eq!(slot(&l.module.image, SLOTS_OFF), 0x7777_7777_7777_7777);
    }

    #[test]
    fn module_without_mod0_loads_unrelocated() {
        let spec = Spec {
            with_mod0: false,
            ..simple_spec()
        };
        let l = load_ok(&build_nso(&build_parts(&spec), NsoFlags::default()));
        assert!(l.module.mod0.is_none());
        assert_eq!(l.module.relocs.applied(), 0);
        assert_eq!(l.module.known_entries(), [GUEST_BASE]);
    }

    #[test]
    fn hash_mismatch_is_an_integrity_error_only_when_checked() {
        let parts = build_parts(&simple_spec());
        let flags = NsoFlags {
            compress: [false; 3],
            hash: [true; 3],
        };
        let mut file = build_nso(&parts, flags);
        // flip one byte of the rodata payload
        let ro_file_off = u32_at(&file, 0x20) as usize;
        file[ro_file_off + 5] ^= 0xFF;
        let err = load(&file, "main", GUEST_BASE, &LoadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Integrity { .. }), "{err}");
        assert!(err.to_string().contains("rodata"));
        // verification can be switched off
        assert!(load(
            &file,
            "main",
            GUEST_BASE,
            &LoadOptions {
                verify_hashes: false
            }
        )
        .is_ok());
        // and the flag decides: with the hash flags clear the same bytes load fine
        let mut unflagged = build_nso(&parts, NsoFlags::default());
        let ro_file_off = u32_at(&unflagged, 0x20) as usize;
        unflagged[ro_file_off + 5] ^= 0xFF;
        let l = load_ok(&unflagged);
        assert_eq!(l.hash_verified, [false; 3]);
        // tampering with the text hash itself is also caught
        let mut file = build_nso(&parts, flags);
        file[0xA0] ^= 1;
        assert!(matches!(
            load(&file, "main", GUEST_BASE, &LoadOptions::default()),
            Err(Error::Integrity { .. })
        ));
    }

    fn put32(file: &mut [u8], at: usize, v: u32) {
        file[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn rejects_malformed_headers_and_segments() {
        let parts = build_parts(&simple_spec());
        let good = build_nso(&parts, NsoFlags::default());
        let try_load = |f: &[u8]| load(f, "main", GUEST_BASE, &LoadOptions::default());

        assert!(try_load(&good[..0x80]).is_err());
        let mut f = good.clone();
        f[0] = b'X';
        assert!(try_load(&f).unwrap_err().to_string().contains("bad magic"));

        let mut f = good.clone();
        put32(&mut f, 0x0C, 1 << 7);
        assert!(matches!(try_load(&f), Err(Error::Unsupported { .. })));

        // segment file range outside the file
        let mut f = good.clone();
        put32(&mut f, 0x18, 0x10_0000);
        assert!(try_load(&f).is_err());

        // unaligned memory offsets
        for at in [0x14usize, 0x24, 0x34] {
            let mut f = good.clone();
            put32(&mut f, at, u32_at(&good, at) + 0x10);
            assert!(
                try_load(&f)
                    .unwrap_err()
                    .to_string()
                    .contains("page aligned"),
                "{at:#x}"
            );
        }

        // overlapping segments: move rodata onto text
        let mut f = good.clone();
        put32(&mut f, 0x24, 0);
        assert!(try_load(&f).unwrap_err().to_string().contains("overlaps"));

        // implausible size
        let mut f = good.clone();
        put32(&mut f, 0x3C, u32::MAX);
        assert!(try_load(&f).is_err());
        let mut f = good.clone();
        put32(&mut f, 0x38, u32::MAX);
        assert!(try_load(&f).is_err());

        // compressed segment whose declared decompressed size is wrong
        let flags = NsoFlags {
            compress: [true, false, false],
            hash: [false; 3],
        };
        let mut f = build_nso(&parts, flags);
        put32(&mut f, 0x18, 0x900);
        assert!(try_load(&f)
            .unwrap_err()
            .to_string()
            .contains("text segment"));

        // compressed segment with garbage
        let mut f = build_nso(&parts, flags);
        let at = u32_at(&f, 0x10) as usize;
        f[at..at + 8].fill(0xFF);
        assert!(try_load(&f).is_err());
    }

    #[test]
    fn eh_frame_functions_and_code_pointers_come_through_mod0() {
        use crate::loader::ehframe::FuncRange;
        let spec = Spec {
            eh_funcs: vec![(0x1C0, 0x20), (0x180, 0x40)],
            relatives: vec![
                (SLOTS_OFF as u64, 0x200),       // text, aligned: a code pointer
                (SLOTS_OFF as u64 + 8, 0x203),   // text but unaligned: not a function entry
                (SLOTS_OFF as u64 + 16, 0x2800), // data, not code
                (SLOTS_OFF as u64 + 24, 0x200),  // duplicate
                (SLOTS_OFF as u64 + 32, -4),     // negative addend is ignored
            ],
            init_array: vec![0x300],
            ..simple_spec()
        };
        let l = load_ok(&build_nso(&build_parts(&spec), NsoFlags::default()));
        let m = &l.module;
        assert_eq!(
            m.eh_functions().unwrap(),
            [
                FuncRange {
                    start: 0x180,
                    len: 0x40
                },
                FuncRange {
                    start: 0x1C0,
                    len: 0x20
                }
            ]
        );
        assert_eq!(m.code_pointers(), [0x200, 0x300]);
        // a module without eh data or without MOD0 reports none
        let plain = load_ok(&build_nso(
            &build_parts(&simple_spec()),
            NsoFlags::default(),
        ));
        assert!(plain.module.eh_functions().unwrap().is_empty());
        let bare = Spec {
            with_mod0: false,
            ..simple_spec()
        };
        let bare = load_ok(&build_nso(&build_parts(&bare), NsoFlags::default()));
        assert!(bare.module.eh_functions().unwrap().is_empty());
        assert!(bare.module.code_pointers().is_empty());
    }

    #[test]
    fn aarch32_modules_are_rejected() {
        let mut parts = build_parts(&simple_spec());
        parts.text[0..4].copy_from_slice(&0xEA00_0010u32.to_le_bytes()); // A32: b <label>
        let f = build_nso(&parts, NsoFlags::default());
        assert!(matches!(
            load(&f, "main", GUEST_BASE, &LoadOptions::default()),
            Err(Error::Aarch32(_))
        ));
    }

    #[test]
    fn corrupt_mod0_structures_are_errors() {
        let parts = build_parts(&simple_spec());
        let rel = |mut p: Parts, f: &dyn Fn(&mut Parts)| {
            f(&mut p);
            build_nso(&p, NsoFlags::default())
        };
        let try_load = |f: &[u8]| load(f, "main", GUEST_BASE, &LoadOptions::default());

        // dynamic offset points outside the image
        let f = rel(parts.clone(), &|p| {
            p.text[MOD0_OFF + 4..MOD0_OFF + 8].copy_from_slice(&0x40_0000i32.to_le_bytes())
        });
        assert!(try_load(&f).is_err());

        // dynamic section that starts too close to the end of the image to hold one entry
        let f = rel(parts.clone(), &|p| {
            let near_end = (0x5000 - 8 - MOD0_OFF) as i32;
            p.text[MOD0_OFF + 4..MOD0_OFF + 8].copy_from_slice(&near_end.to_le_bytes())
        });
        assert!(try_load(&f)
            .unwrap_err()
            .to_string()
            .contains("runs past the module"));

        // DT_REL tables are not supported
        let f = rel(parts.clone(), &|p| {
            let at = DYNAMIC_OFF - DATA_OFF;
            p.data[at..at + 8].copy_from_slice(&17u64.to_le_bytes()); // DT_REL
            p.data[at + 8..at + 16].copy_from_slice(&0x1000u64.to_le_bytes());
            p.data[at + 16..at + 24].copy_from_slice(&18u64.to_le_bytes()); // DT_RELSZ
            p.data[at + 24..at + 32].copy_from_slice(&16u64.to_le_bytes());
            p.data[at + 32..at + 40].copy_from_slice(&0u64.to_le_bytes()); // DT_NULL
        });
        assert!(matches!(try_load(&f), Err(Error::Unsupported { .. })));

        // relocation target outside the module
        let spec = Spec {
            relatives: vec![(0x10_0000, 0)],
            ..simple_spec()
        };
        let f = build_nso(&build_parts(&spec), NsoFlags::default());
        assert!(try_load(&f)
            .unwrap_err()
            .to_string()
            .contains("outside the module"));

        // relocation refers to a symbol past the symbol table
        let spec = Spec {
            symbol_relocs: vec![(SLOTS_OFF as u64, "real", R_AARCH64_ABS64, 0)],
            exports: vec![("real", 0x200)],
            ..simple_spec()
        };
        let mut p = build_parts(&spec);
        // patch the symbol index of the single symbol relocation to something huge
        let rela = RELA_OFF - RO_OFF;
        let info_at = rela + 24 + 8; // second entry (first is none): find by scanning instead
        let _ = info_at;
        for e in 0..21 {
            let at = rela + e * 24 + 8;
            let info = u64::from_le_bytes(p.ro[at..at + 8].try_into().unwrap());
            if info & 0xFFFF_FFFF == u64::from(R_AARCH64_ABS64) {
                p.ro[at..at + 8].copy_from_slice(
                    &((0x10_0000u64 << 32) | u64::from(R_AARCH64_ABS64)).to_le_bytes(),
                );
            }
        }
        let f = build_nso(&p, NsoFlags::default());
        assert!(try_load(&f).is_err());
    }
}
