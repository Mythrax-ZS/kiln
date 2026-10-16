//! A loaded module: its segments mapped at a fixed guest base and relocated in place.
//!
//! All addresses inside the module are kept as offsets from the module start so that the guest
//! base is a single constant; pointer values stored in guest memory are `base + offset`, which
//! keeps pointer arithmetic identical to the original program.
//!
//! Relocation follows the MOD0 header (Switchbrew MOD page) to the `.dynamic` section and applies
//! the RELA tables the way the Horizon runtime linker does for a module linked at address zero.

use super::ehframe;
use crate::{Error, Result};
use nx_runtime::util::{u32_at, u64_at};

/// Guest address the first module is mapped at (docs/ASSUMPTIONS.md A-006).
pub const GUEST_BASE: u64 = 0x0800_0000;
pub const PAGE: u64 = 0x1000;
/// Sanity cap on a module's mapped size (a title may ask for gigabytes of `.bss`).
pub const MAX_IMAGE: u64 = 4 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Perm {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
}

impl Perm {
    pub const RX: Perm = Perm {
        read: true,
        write: false,
        exec: true,
    };
    pub const R: Perm = Perm {
        read: true,
        write: false,
        exec: false,
    };
    pub const RW: Perm = Perm {
        read: true,
        write: true,
        exec: false,
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub name: &'static str,
    /// Offset from the module start.
    pub offset: u64,
    pub size: u64,
    pub perm: Perm,
}

impl Segment {
    pub fn end(&self) -> u64 {
        self.offset + self.size
    }

    pub fn contains(&self, offset: u64) -> bool {
        offset >= self.offset && offset < self.end()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleKind {
    Nso,
    Nro,
}

/// The MOD0 header, with every field converted to a module offset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mod0 {
    pub header_offset: u64,
    pub dynamic: u64,
    pub bss_start: u64,
    pub bss_end: u64,
    pub eh_frame_hdr_start: u64,
    pub eh_frame_hdr_end: u64,
    pub module_object: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnresolvedReloc {
    pub offset: u64,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedReloc {
    pub offset: u64,
    pub r_type: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelocReport {
    pub relative: u32,
    pub abs64: u32,
    pub glob_dat: u32,
    pub jump_slot: u32,
    /// Imports no loaded module defines; the slot is left zero.
    pub unresolved: Vec<UnresolvedReloc>,
    /// Relocation types this loader does not implement; the slot is left untouched.
    pub unsupported: Vec<UnsupportedReloc>,
    /// Addends of the RELATIVE relocations: module offsets that pointers in memory refer to.
    pub relative_targets: Vec<u64>,
    /// Module offset and previous contents of every slot a relocation wrote, in the order they
    /// were written: what the kernel's loader leaves in memory for the runtime linker.
    pub originals: Vec<(u64, [u8; 8])>,
}

impl RelocReport {
    pub fn applied(&self) -> u32 {
        self.relative + self.abs64 + self.glob_dat + self.jump_slot
    }
}

#[derive(Clone, Debug)]
pub struct Module {
    pub kind: ModuleKind,
    pub name: String,
    pub base: u64,
    /// Every mapped byte from the module start: text, rodata, data, then zeroed bss.
    pub image: Vec<u8>,
    pub segments: Vec<Segment>,
    pub build_id: [u8; 32],
    pub mod0: Option<Mod0>,
    pub exports: Vec<Export>,
    /// Function pointers from `.init_array` / `.fini_array`, as guest addresses.
    pub init_array: Vec<u64>,
    pub fini_array: Vec<u64>,
    pub init_fn: Option<u64>,
    pub fini_fn: Option<u64>,
    pub relocs: RelocReport,
}

pub(crate) struct ModuleParts {
    pub kind: ModuleKind,
    pub name: String,
    pub base: u64,
    pub image: Vec<u8>,
    pub segments: Vec<Segment>,
    pub build_id: [u8; 32],
    /// Module offset and size of the dynamic symbol table when the container header says so.
    pub dynsym: Option<(u64, u64)>,
}

/// An A32 branch-always instruction (`0xEA......` or `0xEB......`) as the first word is how an
/// AArch32 module's entry looks. See docs/ASSUMPTIONS.md A-007: this is a heuristic for modules
/// that come without a `main.npdm`.
pub fn looks_like_aarch32(first_word: u32) -> bool {
    matches!(first_word >> 24, 0xEA | 0xEB)
}

const DT_NULL: i64 = 0;
const DT_PLTRELSZ: i64 = 2;
const DT_HASH: i64 = 4;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_RELAENT: i64 = 9;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_INIT: i64 = 12;
const DT_FINI: i64 = 13;
const DT_REL: i64 = 17;
const DT_RELSZ: i64 = 18;
const DT_PLTREL: i64 = 20;
const DT_JMPREL: i64 = 23;
const DT_INIT_ARRAY: i64 = 25;
const DT_FINI_ARRAY: i64 = 26;
const DT_INIT_ARRAYSZ: i64 = 27;
const DT_FINI_ARRAYSZ: i64 = 28;

pub const R_AARCH64_NONE: u32 = 0;
pub const R_AARCH64_ABS64: u32 = 257;
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
pub const R_AARCH64_RELATIVE: u32 = 1027;

const RELA_SIZE: u64 = 24;
const SYM_SIZE: u64 = 24;

#[derive(Default)]
struct Dynamic {
    pltrelsz: u64,
    pltrel: u64,
    hash: Option<u64>,
    strtab: Option<u64>,
    strsz: u64,
    symtab: Option<u64>,
    syment: u64,
    rela: Option<u64>,
    relasz: u64,
    relaent: u64,
    rel: Option<u64>,
    relsz: u64,
    jmprel: Option<u64>,
    init: Option<u64>,
    fini: Option<u64>,
    init_array: Option<u64>,
    init_arraysz: u64,
    fini_array: Option<u64>,
    fini_arraysz: u64,
}

fn find_mod0(image: &[u8]) -> Result<Option<Mod0>> {
    if image.len() < 8 {
        return Ok(None);
    }
    let loc = u64::from(u32_at(image, 4));
    let in_range = loc
        .checked_add(0x1C)
        .map(|e| e <= image.len() as u64)
        .unwrap_or(false);
    if !in_range || &image[loc as usize..loc as usize + 4] != b"MOD0" {
        return Ok(None);
    }
    let field = |at: u64| -> Result<u64> {
        let rel = i64::from(u32_at(image, (loc + at) as usize) as i32);
        let abs = loc as i64 + rel;
        if abs < 0 || abs as u64 > image.len() as u64 {
            return Err(Error::format(
                "MOD0 header",
                format!("field at +{at:#x} points outside the module"),
            ));
        }
        Ok(abs as u64)
    };
    Ok(Some(Mod0 {
        header_offset: loc,
        dynamic: field(4)?,
        bss_start: field(8)?,
        bss_end: field(0xC)?,
        eh_frame_hdr_start: field(0x10)?,
        eh_frame_hdr_end: field(0x14)?,
        module_object: field(0x18)?,
    }))
}

fn read_dynamic(image: &[u8], start: u64) -> Result<Dynamic> {
    let mut d = Dynamic::default();
    let mut off = start;
    for _ in 0..4096 {
        if off
            .checked_add(16)
            .map(|e| e > image.len() as u64)
            .unwrap_or(true)
        {
            return Err(Error::format(
                "MOD0 dynamic section",
                "runs past the module",
            ));
        }
        let tag = u64_at(image, off as usize) as i64;
        let val = u64_at(image, off as usize + 8);
        match tag {
            DT_NULL => return Ok(d),
            DT_PLTRELSZ => d.pltrelsz = val,
            DT_HASH => d.hash = Some(val),
            DT_STRTAB => d.strtab = Some(val),
            DT_SYMTAB => d.symtab = Some(val),
            DT_RELA => d.rela = Some(val),
            DT_RELASZ => d.relasz = val,
            DT_RELAENT => d.relaent = val,
            DT_STRSZ => d.strsz = val,
            DT_SYMENT => d.syment = val,
            DT_INIT => d.init = Some(val),
            DT_FINI => d.fini = Some(val),
            DT_REL => d.rel = Some(val),
            DT_RELSZ => d.relsz = val,
            DT_PLTREL => d.pltrel = val,
            DT_JMPREL => d.jmprel = Some(val),
            DT_INIT_ARRAY => d.init_array = Some(val),
            DT_INIT_ARRAYSZ => d.init_arraysz = val,
            DT_FINI_ARRAY => d.fini_array = Some(val),
            DT_FINI_ARRAYSZ => d.fini_arraysz = val,
            _ => {}
        }
        off += 16;
    }
    Err(Error::format(
        "MOD0 dynamic section",
        "has no DT_NULL terminator",
    ))
}

struct Symbol {
    name_off: u32,
    info: u8,
    shndx: u16,
    value: u64,
    size: u64,
}

fn read_symbol(image: &[u8], symtab: u64, index: u64) -> Result<Symbol> {
    let at = symtab
        .checked_add(index.saturating_mul(SYM_SIZE))
        .filter(|&a| {
            a.checked_add(SYM_SIZE)
                .map(|e| e <= image.len() as u64)
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            Error::format(
                "dynamic symbol table",
                format!("symbol {index} is outside the module"),
            )
        })? as usize;
    Ok(Symbol {
        name_off: u32_at(image, at),
        info: image[at + 4],
        shndx: u16::from_le_bytes([image[at + 6], image[at + 7]]),
        value: u64_at(image, at + 8),
        size: u64_at(image, at + 16),
    })
}

fn symbol_name(image: &[u8], strtab: Option<u64>, strsz: u64, name_off: u32) -> String {
    let Some(strtab) = strtab else {
        return format!("#{name_off}");
    };
    let start = strtab + u64::from(name_off);
    let limit = if strsz > 0 {
        (strtab + strsz).min(image.len() as u64)
    } else {
        image.len() as u64
    };
    if start >= limit {
        return format!("#{name_off}");
    }
    let bytes = &image[start as usize..limit as usize];
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}

fn symbol_count(image: &[u8], d: &Dynamic, hint: Option<(u64, u64)>) -> u64 {
    let Some(symtab) = d.symtab else { return 0 };
    if let Some((off, size)) = hint {
        if off == symtab && size >= SYM_SIZE {
            return size / SYM_SIZE;
        }
    }
    if let Some(h) = d.hash {
        if h + 8 <= image.len() as u64 {
            return u64::from(u32_at(image, h as usize + 4));
        }
    }
    match d.strtab {
        Some(s) if s > symtab => (s - symtab) / SYM_SIZE,
        _ => 0,
    }
}

fn apply_table(
    image: &mut [u8],
    base: u64,
    d: &Dynamic,
    table: u64,
    size: u64,
    report: &mut RelocReport,
) -> Result<()> {
    if table
        .checked_add(size)
        .map(|e| e > image.len() as u64)
        .unwrap_or(true)
    {
        return Err(Error::format("relocation table", "lies outside the module"));
    }
    let image_len = image.len() as u64;
    for i in 0..size / RELA_SIZE {
        let at = (table + i * RELA_SIZE) as usize;
        let r_offset = u64_at(image, at);
        let r_info = u64_at(image, at + 8);
        let r_addend = u64_at(image, at + 16) as i64;
        let r_type = (r_info & 0xFFFF_FFFF) as u32;
        let r_sym = r_info >> 32;
        if r_type == R_AARCH64_NONE {
            continue;
        }
        if r_offset
            .checked_add(8)
            .map(|e| e > image_len)
            .unwrap_or(true)
        {
            return Err(Error::format(
                "relocation",
                format!("target {r_offset:#x} is outside the module"),
            ));
        }
        let value = match r_type {
            R_AARCH64_RELATIVE => {
                report.relative += 1;
                if r_addend >= 0 {
                    report.relative_targets.push(r_addend as u64);
                }
                base.wrapping_add(r_addend as u64)
            }
            R_AARCH64_ABS64 | R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT => {
                match r_type {
                    R_AARCH64_ABS64 => report.abs64 += 1,
                    R_AARCH64_GLOB_DAT => report.glob_dat += 1,
                    _ => report.jump_slot += 1,
                }
                let sym_value = if r_sym == 0 {
                    0
                } else {
                    let symtab = d.symtab.ok_or_else(|| {
                        Error::format(
                            "relocation",
                            "refers to a symbol but the module has no DT_SYMTAB",
                        )
                    })?;
                    let sym = read_symbol(image, symtab, r_sym)?;
                    if sym.shndx == 0 {
                        // Undefined: weak imports resolve to zero, others are reported.
                        let weak = sym.info >> 4 == 2;
                        if !weak {
                            report.unresolved.push(UnresolvedReloc {
                                offset: r_offset,
                                symbol: symbol_name(image, d.strtab, d.strsz, sym.name_off),
                            });
                        }
                        // Leave the slot zero.
                        report.originals.push((r_offset, slot_at(image, r_offset)));
                        image[r_offset as usize..r_offset as usize + 8].fill(0);
                        continue;
                    }
                    base.wrapping_add(sym.value)
                };
                sym_value.wrapping_add(r_addend as u64)
            }
            other => {
                report.unsupported.push(UnsupportedReloc {
                    offset: r_offset,
                    r_type: other,
                });
                continue;
            }
        };
        report.originals.push((r_offset, slot_at(image, r_offset)));
        image[r_offset as usize..r_offset as usize + 8].copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn slot_at(image: &[u8], at: u64) -> [u8; 8] {
    let mut b = [0u8; 8];
    b.copy_from_slice(&image[at as usize..at as usize + 8]);
    b
}

fn read_pointer_array(image: &[u8], at: Option<u64>, size: u64) -> Result<Vec<u64>> {
    let Some(at) = at else { return Ok(Vec::new()) };
    if size == 0 {
        return Ok(Vec::new());
    }
    if at
        .checked_add(size)
        .map(|e| e > image.len() as u64)
        .unwrap_or(true)
    {
        return Err(Error::format("init/fini array", "lies outside the module"));
    }
    Ok((0..size / 8)
        .map(|i| u64_at(image, (at + i * 8) as usize))
        .collect())
}

impl Module {
    pub(crate) fn from_parts(p: ModuleParts) -> Result<Module> {
        let ModuleParts {
            kind,
            name,
            base,
            mut image,
            segments,
            build_id,
            dynsym,
        } = p;
        if image.len() >= 4 && looks_like_aarch32(u32_at(&image, 0)) {
            return Err(Error::Aarch32(format!(
                "first instruction of `{name}` is an A32 branch"
            )));
        }
        let mut module = Module {
            kind,
            name,
            base,
            image: Vec::new(),
            segments,
            build_id,
            mod0: find_mod0(&image)?,
            exports: Vec::new(),
            init_array: Vec::new(),
            fini_array: Vec::new(),
            init_fn: None,
            fini_fn: None,
            relocs: RelocReport::default(),
        };
        if let Some(mod0) = module.mod0.clone() {
            let d = read_dynamic(&image, mod0.dynamic)?;
            if d.rel.is_some() && d.relsz > 0 {
                return Err(Error::unsupported(
                    "relocations",
                    "DT_REL tables without addends are not used by Horizon toolchains",
                ));
            }
            let ent = if d.relaent == 0 { RELA_SIZE } else { d.relaent };
            if ent != RELA_SIZE || (d.symtab.is_some() && d.syment != 0 && d.syment != SYM_SIZE) {
                return Err(Error::format("dynamic section", "unexpected entry size"));
            }
            let mut report = RelocReport::default();
            if let Some(rela) = d.rela {
                apply_table(&mut image, base, &d, rela, d.relasz, &mut report)?;
            }
            if let Some(jmprel) = d.jmprel {
                if d.pltrel != 0 && d.pltrel != 7 {
                    return Err(Error::unsupported(
                        "PLT relocations",
                        "only RELA tables are supported",
                    ));
                }
                apply_table(&mut image, base, &d, jmprel, d.pltrelsz, &mut report)?;
            }
            module.relocs = report;
            module.init_array = read_pointer_array(&image, d.init_array, d.init_arraysz)?;
            module.fini_array = read_pointer_array(&image, d.fini_array, d.fini_arraysz)?;
            module.init_fn = d.init.map(|v| base + v);
            module.fini_fn = d.fini.map(|v| base + v);
            let count = symbol_count(&image, &d, dynsym);
            if let Some(symtab) = d.symtab {
                for index in 1..count {
                    let Ok(sym) = read_symbol(&image, symtab, index) else {
                        break;
                    };
                    let is_func = sym.info & 0xF == 2;
                    if sym.shndx != 0 && is_func && sym.value != 0 {
                        module.exports.push(Export {
                            name: symbol_name(&image, d.strtab, d.strsz, sym.name_off),
                            offset: sym.value,
                            size: sym.size,
                        });
                    }
                }
            }
        }
        module.image = image;
        Ok(module)
    }

    /// Guest address of the module start, where execution begins.
    pub fn entry(&self) -> u64 {
        self.base
    }

    /// The mapped bytes as the file holds them, before this loader applied any relocation. That
    /// is what the kernel's loader leaves in memory: the title's own runtime linker (`rtld`)
    /// relocates the modules when the program runs, so a produced program starts from this image.
    /// The relocated `image` is for the analysis, which reads jump tables and pointers.
    pub fn unrelocated_image(&self) -> Vec<u8> {
        let mut image = self.image.clone();
        for &(off, bytes) in self.relocs.originals.iter().rev() {
            image[off as usize..off as usize + 8].copy_from_slice(&bytes);
        }
        image
    }

    pub fn addr(&self, offset: u64) -> u64 {
        self.base + offset
    }

    /// Module offset of a guest address, if it lies inside the mapped image.
    pub fn offset_of(&self, addr: u64) -> Option<u64> {
        addr.checked_sub(self.base)
            .filter(|&o| o < self.image.len() as u64)
    }

    pub fn segment(&self, name: &str) -> Option<&Segment> {
        self.segments.iter().find(|s| s.name == name)
    }

    pub fn segment_at(&self, offset: u64) -> Option<&Segment> {
        self.segments.iter().find(|s| s.contains(offset))
    }

    /// Bytes of the named segment as currently relocated.
    pub fn segment_bytes(&self, name: &str) -> Option<&[u8]> {
        let s = self.segment(name)?;
        self.image.get(s.offset as usize..s.end() as usize)
    }

    /// Function ranges the toolchain recorded in `.eh_frame_hdr`, found through MOD0. Empty when
    /// the module has no MOD0 or no such header.
    pub fn eh_functions(&self) -> Result<Vec<ehframe::FuncRange>> {
        match &self.mod0 {
            Some(m) => {
                ehframe::function_ranges(&self.image, m.eh_frame_hdr_start, m.eh_frame_hdr_end)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Module offsets inside the text segment that RELATIVE relocations point at: vtable entries
    /// and callbacks, so very likely function entries reached through indirect calls.
    pub fn code_pointers(&self) -> Vec<u64> {
        let Some(text) = self.segment("text") else {
            return Vec::new();
        };
        let mut v: Vec<u64> = self
            .relocs
            .relative_targets
            .iter()
            .copied()
            .filter(|&t| text.contains(t) && t % 4 == 0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Every address the loader knows to be a function entry: the module start, exported
    /// functions, init/fini functions and the init/fini arrays.
    pub fn known_entries(&self) -> Vec<u64> {
        let mut v = vec![self.base];
        v.extend(self.exports.iter().map(|e| self.base + e.offset));
        v.extend(self.init_array.iter().chain(&self.fini_array));
        v.extend(self.init_fn.iter().chain(self.fini_fn.iter()));
        v.retain(|&a| self.offset_of(a).is_some());
        v.sort_unstable();
        v.dedup();
        v
    }
}
