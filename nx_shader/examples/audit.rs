//! What the programs given need that the translator cannot do yet: the instructions it has no
//! translation for (counted by programs and by instructions), and how each texture instruction is
//! used (its dimension, level of detail and so on, from the fields of the encoding).
//! usage: audit <program.bin | directory>...

use nx_shader::cfg::{Cfg, Code};
use nx_shader::inst::{Guard, Inst, PT};
use nx_shader::ops::{Op, TABLE};
use nx_shader::sph::{Sph, SPH_BYTES};
use nx_shader::translate::{translate, Config, Stage};
use std::collections::{BTreeMap, BTreeSet};

/// The instructions of the table that have no translation at all (as in the `coverage` example).
fn untranslated() -> BTreeSet<Op> {
    let mut missing = BTreeSet::new();
    for row in TABLE.iter() {
        let word = row.value | u64::from(PT) << 16 | 0x7 << 48 | 0x7 << 3;
        let word = word & !(row.mask) | row.value;
        let code = nx_shader::asm::assemble(&[word, nx_shader::asm::exit(Guard::ALWAYS)]);
        if let Err(e) = translate(Code(&code), None, &Config::new(Stage::Compute)) {
            if e.message.contains("is not translated") || e.message.contains("is not known") {
                missing.insert(row.op);
            }
        }
    }
    missing
}

/// The dimension of `TEX`-family instructions: bits 28..30 (0 1D, 1 1D array, 2 2D, 3 2D array,
/// 4 3D, 6 cube, 7 cube array).
fn dim_name(code: u64) -> &'static str {
    match code {
        0 => "1D",
        1 => "1D array",
        2 => "2D",
        3 => "2D array",
        4 => "3D",
        5 => "3D array",
        6 => "cube",
        _ => "cube array",
    }
}

/// How an instruction of the texture family is used, for the table of variants.
fn describe(op: Op, i: Inst) -> Option<String> {
    Some(match op {
        Op::Texs => format!(
            "TEXS kind {} results {} nodep {}",
            i.bits(53, 4),
            if i.bit(59) { "f32" } else { "f16" },
            i.bit(49)
        ),
        Op::Tlds => format!(
            "TLDS kind {} results {}",
            i.bits(53, 4),
            if i.bit(59) { "f32" } else { "f16" }
        ),
        Op::Tld4s => format!(
            "TLD4S dc {} aoffi {} component {} results {}",
            i.bit(50),
            i.bit(51),
            i.bits(52, 2),
            if i.bit(55) { "f16" } else { "f32" }
        ),
        Op::TexBound => format!(
            "TEX {} lod mode {} dc {} aoffi {} mask {:#x}",
            dim_name(i.bits(28, 3)),
            i.bits(55, 3),
            i.bit(50),
            i.bit(54),
            i.bits(31, 4)
        ),
        Op::TldBound => format!(
            "TLD {} lod {} toff {} ms {} mask {:#x}",
            dim_name(i.bits(28, 3)),
            i.bit(55),
            i.bit(35),
            i.bit(50),
            i.bits(31, 4)
        ),
        Op::Tld4Bound => format!(
            "TLD4 {} dc {} toff {} component {} mask {:#x}",
            dim_name(i.bits(28, 3)),
            i.bit(50),
            i.bits(54, 2),
            i.bits(56, 2),
            i.bits(31, 4)
        ),
        Op::TxqBound => format!("TXQ query {} mask {:#x}", i.bits(22, 6), i.bits(31, 4)),
        Op::TmmlBound => format!("TMML {} mask {:#x}", dim_name(i.bits(28, 3)), i.bits(31, 4)),
        Op::TxdBound => format!(
            "TXD {} toff {} mask {:#x}",
            dim_name(i.bits(28, 3)),
            i.bit(35),
            i.bits(31, 4)
        ),
        Op::TexBindless
        | Op::TldBindless
        | Op::Tld4Bindless
        | Op::TxqBindless
        | Op::TxdBindless
        | Op::TmmlBindless => format!("{op:?} (bindless)"),
        _ => return None,
    })
}

fn files_of(arg: &str, into: &mut Vec<std::path::PathBuf>) {
    let path = std::path::Path::new(arg);
    if path.is_dir() {
        let mut names: Vec<_> = std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .collect();
        names.sort();
        into.extend(names);
    } else {
        into.push(path.to_path_buf());
    }
}

fn main() {
    let mut filter: Option<String> = None;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--where" {
            filter = args.next();
            continue;
        }
        files_of(&a, &mut files);
    }
    let missing = untranslated();
    // op -> (programs, instructions); variant -> (programs, instructions)
    let mut by_op: BTreeMap<String, (BTreeSet<usize>, u64)> = BTreeMap::new();
    let mut by_variant: BTreeMap<String, (BTreeSet<usize>, u64)> = BTreeMap::new();
    let mut programs = 0;
    for (n, path) in files.iter().enumerate() {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        // A compute program has no header: its dump is the code alone (a name that starts COMPUTE).
        let headerless = path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("COMPUTE"));
        if !headerless {
            let Ok(sph) = Sph::parse(&bytes) else {
                continue;
            };
            if sph.stage().is_none() {
                continue;
            }
        }
        programs += 1;
        let code = Code(&bytes[if headerless { 0 } else { SPH_BYTES }..]);
        let cfg = Cfg::discover(code);
        for b in cfg.blocks.values() {
            for (pc, inst) in cfg.instructions(code, b) {
                let Some(op) = inst.op() else {
                    let e = by_op.entry("(unknown encoding)".to_string()).or_default();
                    e.0.insert(n);
                    e.1 += 1;
                    continue;
                };
                if missing.contains(&op) {
                    let e = by_op.entry(format!("{op:?}")).or_default();
                    e.0.insert(n);
                    e.1 += 1;
                }
                if let Some(v) = describe(op, inst) {
                    if filter.as_ref().is_some_and(|f| v.contains(f.as_str())) {
                        println!(
                            "{}:{pc:#x}: {v}; rd {} ra {} rb {} r28 {} r39 {}",
                            path.display(),
                            inst.rd(),
                            inst.ra(),
                            inst.rb(),
                            inst.bits(28, 8),
                            inst.bits(39, 8)
                        );
                    }
                    let e = by_variant.entry(v).or_default();
                    e.0.insert(n);
                    e.1 += 1;
                }
            }
        }
    }
    println!("{programs} programs");
    println!("instructions with no translation:");
    let mut rows: Vec<_> = by_op.into_iter().collect();
    rows.sort_by_key(|(_, (p, i))| (std::cmp::Reverse(p.len()), std::cmp::Reverse(*i)));
    for (name, (progs, count)) in rows {
        println!(
            "  {name:<24} {:>4} programs {count:>7} instructions",
            progs.len()
        );
    }
    println!("texture instructions by variant:");
    let mut rows: Vec<_> = by_variant.into_iter().collect();
    rows.sort_by_key(|(_, (p, i))| (std::cmp::Reverse(p.len()), std::cmp::Reverse(*i)));
    for (name, (progs, count)) in rows {
        println!(
            "  {name:<60} {:>4} programs {count:>7} instructions",
            progs.len()
        );
    }
}
