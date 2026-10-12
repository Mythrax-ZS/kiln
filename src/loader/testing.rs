//! Synthetic NSO and NRO writers for tests. They build the files byte by byte from the Switchbrew
//! layouts and share no code with the loaders, so a mistake in a loader cannot hide behind a
//! matching mistake here.

use super::lz4::testing::compress;
use sha2::{Digest, Sha256};

pub const SEG_SIZE: usize = 0x1000;
pub const RO_OFF: usize = 0x1000;
pub const DATA_OFF: usize = 0x2000;
pub const MOD0_OFF: usize = 0x100;
pub const CODE_OFF: usize = 0x180;
/// RELA table: room for 21 entries.
pub const RELA_OFF: usize = 0x1000;
/// PLT relocation table: room for 21 entries.
pub const JMPREL_OFF: usize = 0x1200;
pub const DYNSYM_OFF: usize = 0x1400;
pub const DYNSTR_OFF: usize = 0x1800;
pub const DYNAMIC_OFF: usize = 0x2000;
pub const INIT_ARRAY_OFF: usize = 0x2100;
pub const FINI_ARRAY_OFF: usize = 0x2180;
/// Scratch slots that relocations write to.
pub const SLOTS_OFF: usize = 0x2200;
pub const BSS_SIZE: u32 = 0x1800;
/// `.eh_frame_hdr` and `.eh_frame` live at the end of rodata.
pub const EH_HDR_OFF: usize = 0x1C00;
pub const EH_OFF: usize = 0x1C80;

/// What the synthetic module contains.
#[derive(Clone, Default)]
pub struct Spec {
    /// Instruction words placed at `CODE_OFF`.
    pub code: Vec<u32>,
    pub with_mod0: bool,
    /// `R_AARCH64_RELATIVE`: (target offset, addend).
    pub relatives: Vec<(u64, i64)>,
    /// Relocations against a symbol: (target offset, symbol name or "" for none, type, addend).
    pub symbol_relocs: Vec<(u64, &'static str, u32, i64)>,
    /// Defined function symbols: (name, offset).
    pub exports: Vec<(&'static str, u64)>,
    /// Undefined symbols: (name, weak).
    pub imports: Vec<(&'static str, bool)>,
    pub init_array: Vec<u64>,
    pub fini_array: Vec<u64>,
    pub init_fn: Option<u64>,
    pub fini_fn: Option<u64>,
    /// `R_AARCH64_JUMP_SLOT` entries in the PLT table: (target offset, symbol name).
    pub jmprel: Vec<(u64, &'static str)>,
    /// Functions with unwind info: (start offset, length). Written as an `.eh_frame_hdr`.
    pub eh_funcs: Vec<(u32, u32)>,
}

#[derive(Clone)]
pub struct Parts {
    pub text: Vec<u8>,
    pub ro: Vec<u8>,
    pub data: Vec<u8>,
    pub nsyms: usize,
    pub dynstr_len: usize,
}

fn put32(v: &mut [u8], at: usize, x: u32) {
    v[at..at + 4].copy_from_slice(&x.to_le_bytes());
}

fn put64(v: &mut [u8], at: usize, x: u64) {
    v[at..at + 8].copy_from_slice(&x.to_le_bytes());
}

/// Write an `.eh_frame_hdr` (pcrel sdata4 frame pointer, udata4 count, datarel sdata4 table) and
/// an `.eh_frame` with one `zR` CIE and one FDE per function. Returns the header length, or 0
/// when there are no functions.
fn write_eh(ro: &mut [u8], funcs: &[(u32, u32)]) -> usize {
    if funcs.is_empty() {
        return 0;
    }
    let padded = |mut body: Vec<u8>| -> Vec<u8> {
        while !(body.len() + 4).is_multiple_of(8) {
            body.push(0);
        }
        let mut out = (body.len() as u32).to_le_bytes().to_vec();
        out.extend(body);
        out
    };
    // CIE: id 0, version 1, "zR", code align 4, data align -8, return register 30, one byte of
    // augmentation data holding the FDE encoding (pcrel | sdata4).
    let cie = padded(vec![0, 0, 0, 0, 1, b'z', b'R', 0, 4, 0x78, 30, 1, 0x1b]);
    let mut eh = cie.clone();
    let mut table: Vec<(u32, u32)> = Vec::new();
    for &(start, len) in funcs {
        let fde_off = EH_OFF + eh.len();
        let mut body = Vec::new();
        body.extend_from_slice(&((fde_off + 4 - EH_OFF) as u32).to_le_bytes());
        body.extend_from_slice(&((start as i64 - (fde_off + 8) as i64) as i32).to_le_bytes());
        body.extend_from_slice(&(len as i32).to_le_bytes());
        body.push(0); // augmentation data length
        eh.extend(padded(body));
        table.push((start, fde_off as u32));
    }
    table.sort();
    ro[EH_OFF - RO_OFF..EH_OFF - RO_OFF + eh.len()].copy_from_slice(&eh);
    let mut hdr = vec![1u8, 0x1b, 0x03, 0x3b];
    hdr.extend_from_slice(&((EH_OFF as i64 - (EH_HDR_OFF as i64 + 4)) as i32).to_le_bytes());
    hdr.extend_from_slice(&(table.len() as u32).to_le_bytes());
    for (start, fde) in table {
        hdr.extend_from_slice(&((start as i64 - EH_HDR_OFF as i64) as i32).to_le_bytes());
        hdr.extend_from_slice(&((fde as i64 - EH_HDR_OFF as i64) as i32).to_le_bytes());
    }
    ro[EH_HDR_OFF - RO_OFF..EH_HDR_OFF - RO_OFF + hdr.len()].copy_from_slice(&hdr);
    hdr.len()
}

pub fn build_parts(spec: &Spec) -> Parts {
    let mut text = vec![0u8; SEG_SIZE];
    let mut ro = vec![0u8; SEG_SIZE];
    let mut data = vec![0u8; SEG_SIZE];

    // text: `b CODE_OFF`, the MOD0 offset, code, and the MOD0 header
    put32(&mut text, 0, 0x1400_0000 | (CODE_OFF as u32 / 4));
    put32(
        &mut text,
        4,
        if spec.with_mod0 { MOD0_OFF as u32 } else { 0 },
    );
    for (i, w) in spec.code.iter().enumerate() {
        put32(&mut text, CODE_OFF + 4 * i, *w);
    }
    let eh_hdr_len = write_eh(&mut ro, &spec.eh_funcs);
    if spec.with_mod0 {
        let rel = |target: usize| (target as i64 - MOD0_OFF as i64) as i32 as u32;
        let bss_start = DATA_OFF + SEG_SIZE;
        text[MOD0_OFF..MOD0_OFF + 4].copy_from_slice(b"MOD0");
        put32(&mut text, MOD0_OFF + 4, rel(DYNAMIC_OFF));
        put32(&mut text, MOD0_OFF + 8, rel(bss_start));
        put32(&mut text, MOD0_OFF + 12, rel(bss_start + BSS_SIZE as usize));
        if eh_hdr_len > 0 {
            put32(&mut text, MOD0_OFF + 16, rel(EH_HDR_OFF));
            put32(&mut text, MOD0_OFF + 20, rel(EH_HDR_OFF + eh_hdr_len));
        } else {
            put32(&mut text, MOD0_OFF + 16, 0);
            put32(&mut text, MOD0_OFF + 20, 0);
        }
        put32(&mut text, MOD0_OFF + 24, rel(bss_start));
    }

    // dynamic symbols: null, imports, exports
    let mut dynstr = vec![0u8];
    let name_off = |name: &str, dynstr: &mut Vec<u8>| -> u32 {
        if name.is_empty() {
            return 0;
        }
        let at = dynstr.len() as u32;
        dynstr.extend_from_slice(name.as_bytes());
        dynstr.push(0);
        at
    };
    let mut symbols: Vec<[u8; 24]> = vec![[0u8; 24]];
    let mut index_of = std::collections::HashMap::new();
    for (name, weak) in &spec.imports {
        let mut s = [0u8; 24];
        s[0..4].copy_from_slice(&name_off(name, &mut dynstr).to_le_bytes());
        s[4] = (if *weak { 2 } else { 1 }) << 4; // binding, NOTYPE
        index_of.insert(*name, symbols.len() as u64);
        symbols.push(s);
    }
    for (name, off) in &spec.exports {
        let mut s = [0u8; 24];
        s[0..4].copy_from_slice(&name_off(name, &mut dynstr).to_le_bytes());
        s[4] = (1 << 4) | 2; // GLOBAL FUNC
        s[6..8].copy_from_slice(&1u16.to_le_bytes());
        s[8..16].copy_from_slice(&off.to_le_bytes());
        s[16..24].copy_from_slice(&0x10u64.to_le_bytes());
        index_of.insert(*name, symbols.len() as u64);
        symbols.push(s);
    }
    for (i, s) in symbols.iter().enumerate() {
        ro[DYNSYM_OFF - RO_OFF + 24 * i..DYNSYM_OFF - RO_OFF + 24 * (i + 1)].copy_from_slice(s);
    }
    ro[DYNSTR_OFF - RO_OFF..DYNSTR_OFF - RO_OFF + dynstr.len()].copy_from_slice(&dynstr);

    // relocations
    let sym_index = |name: &str| -> u64 {
        if name.is_empty() {
            0
        } else {
            index_of[name]
        }
    };
    let mut rela: Vec<(u64, u64, i64)> = Vec::new();
    for (off, addend) in &spec.relatives {
        rela.push((*off, 1027, *addend));
    }
    for (off, name, ty, addend) in &spec.symbol_relocs {
        rela.push((*off, u64::from(*ty) | sym_index(name) << 32, *addend));
    }
    for (i, f) in spec.init_array.iter().enumerate() {
        rela.push(((INIT_ARRAY_OFF + 8 * i) as u64, 1027, *f as i64));
    }
    for (i, f) in spec.fini_array.iter().enumerate() {
        rela.push(((FINI_ARRAY_OFF + 8 * i) as u64, 1027, *f as i64));
    }
    assert!(
        rela.len() <= 21,
        "too many relocations for the synthetic layout"
    );
    for (i, (off, info, addend)) in rela.iter().enumerate() {
        let at = RELA_OFF - RO_OFF + 24 * i;
        put64(&mut ro, at, *off);
        put64(&mut ro, at + 8, *info);
        put64(&mut ro, at + 16, *addend as u64);
    }
    assert!(spec.jmprel.len() <= 21);
    for (i, (off, name)) in spec.jmprel.iter().enumerate() {
        let at = JMPREL_OFF - RO_OFF + 24 * i;
        put64(&mut ro, at, *off);
        put64(&mut ro, at + 8, 1026 | sym_index(name) << 32);
        put64(&mut ro, at + 16, 0);
    }

    // dynamic section
    let mut dynamic: Vec<(i64, u64)> = vec![
        (7, RELA_OFF as u64),
        (8, 24 * rela.len() as u64),
        (9, 24),
        (6, DYNSYM_OFF as u64),
        (11, 24),
        (5, DYNSTR_OFF as u64),
        (10, dynstr.len() as u64),
    ];
    if !spec.init_array.is_empty() {
        dynamic.push((25, INIT_ARRAY_OFF as u64));
        dynamic.push((27, 8 * spec.init_array.len() as u64));
    }
    if !spec.fini_array.is_empty() {
        dynamic.push((26, FINI_ARRAY_OFF as u64));
        dynamic.push((28, 8 * spec.fini_array.len() as u64));
    }
    if let Some(v) = spec.init_fn {
        dynamic.push((12, v));
    }
    if let Some(v) = spec.fini_fn {
        dynamic.push((13, v));
    }
    if !spec.jmprel.is_empty() {
        dynamic.push((23, JMPREL_OFF as u64));
        dynamic.push((2, 24 * spec.jmprel.len() as u64));
        dynamic.push((20, 7));
    }
    dynamic.push((0, 0));
    assert!(dynamic.len() * 16 <= INIT_ARRAY_OFF - DYNAMIC_OFF);
    for (i, (tag, val)) in dynamic.iter().enumerate() {
        let at = DYNAMIC_OFF - DATA_OFF + 16 * i;
        put64(&mut data, at, *tag as u64);
        put64(&mut data, at + 8, *val);
    }

    Parts {
        text,
        ro,
        data,
        nsyms: symbols.len(),
        dynstr_len: dynstr.len(),
    }
}

fn module_id() -> [u8; 32] {
    core::array::from_fn(|i| 0xB1u8.wrapping_add(i as u8))
}

#[derive(Clone, Copy, Default)]
pub struct NsoFlags {
    pub compress: [bool; 3],
    pub hash: [bool; 3],
}

/// Write an NSO with the given per-segment compression and hash-check flags.
pub fn build_nso(parts: &Parts, flags: NsoFlags) -> Vec<u8> {
    let segs = [&parts.text, &parts.ro, &parts.data];
    let mut header = vec![0u8; 0x100];
    header[0..4].copy_from_slice(b"NSO0");
    let mut flag_bits = 0u32;
    let mut payload: Vec<u8> = Vec::new();
    for i in 0..3 {
        let stored = if flags.compress[i] {
            compress(segs[i])
        } else {
            segs[i].clone()
        };
        let base = 0x10 + 0x10 * i;
        put32(&mut header, base, (0x100 + payload.len()) as u32); // file offset
        put32(&mut header, base + 4, (i * SEG_SIZE) as u32); // memory offset
        put32(&mut header, base + 8, segs[i].len() as u32); // decompressed size
        put32(&mut header, 0x60 + 4 * i, stored.len() as u32);
        header[0xA0 + 0x20 * i..0xC0 + 0x20 * i].copy_from_slice(&Sha256::digest(segs[i]));
        if flags.compress[i] {
            flag_bits |= 1 << i;
        }
        if flags.hash[i] {
            flag_bits |= 8 << i;
        }
        payload.extend_from_slice(&stored);
    }
    put32(&mut header, 0x0C, flag_bits);
    put32(&mut header, 0x1C, 0x100); // module name offset
    put32(&mut header, 0x3C, BSS_SIZE);
    header[0x40..0x60].copy_from_slice(&module_id());
    put32(&mut header, 0x90, (DYNSTR_OFF - RO_OFF) as u32);
    put32(&mut header, 0x94, parts.dynstr_len as u32);
    put32(&mut header, 0x98, (DYNSYM_OFF - RO_OFF) as u32);
    put32(&mut header, 0x9C, 24 * parts.nsyms as u32);
    [header, payload].concat()
}

pub struct AssetSpec {
    pub icon: Vec<u8>,
    pub nacp: Vec<u8>,
    pub romfs: Vec<u8>,
}

/// Write an NRO: Rocrt header, NRO header, then the three segments at their memory offsets,
/// optionally followed by an `ASET` section.
pub fn build_nro(parts: &Parts, bss: u32, assets: Option<&AssetSpec>) -> Vec<u8> {
    let mut file = [parts.text.clone(), parts.ro.clone(), parts.data.clone()].concat();
    let total = file.len();
    file[0x10..0x14].copy_from_slice(b"NRO0");
    put32(&mut file, 0x18, total as u32);
    for i in 0..3 {
        put32(&mut file, 0x20 + 8 * i, (i * SEG_SIZE) as u32);
        put32(&mut file, 0x24 + 8 * i, SEG_SIZE as u32);
    }
    put32(&mut file, 0x38, bss);
    file[0x40..0x60].copy_from_slice(&module_id());
    put32(&mut file, 0x70, (DYNSTR_OFF - RO_OFF) as u32);
    put32(&mut file, 0x74, parts.dynstr_len as u32);
    put32(&mut file, 0x78, (DYNSYM_OFF - RO_OFF) as u32);
    put32(&mut file, 0x7C, 24 * parts.nsyms as u32);
    if let Some(a) = assets {
        let mut aset = vec![0u8; 0x38];
        aset[0..4].copy_from_slice(b"ASET");
        let mut body: Vec<u8> = Vec::new();
        for (slot, blob) in [(0x08usize, &a.icon), (0x18, &a.nacp), (0x28, &a.romfs)] {
            put64(&mut aset, slot, (0x38 + body.len()) as u64);
            put64(&mut aset, slot + 8, blob.len() as u64);
            body.extend_from_slice(blob);
        }
        file.extend_from_slice(&aset);
        file.extend_from_slice(&body);
    }
    file
}
