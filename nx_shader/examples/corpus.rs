//! Translates every shader program dump in a directory (as `NX_SHADER_DUMP` writes them: a header
//! and its code, or the code of a compute program), validates the results with `spirv-val` when
//! it is installed, and says how many of each stage translated and why the others did not, most
//! frequent reason first. Nothing is written.
//! usage: corpus <dir> [<how many examples of each reason>]

use nx_shader::cfg::Code;
use nx_shader::sph::{Sph, Stage as HeaderStage, SPH_BYTES};
use nx_shader::translate::{translate, translate_sm86, Config, Stage};
use nx_shader::validate;
use std::collections::BTreeMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("a directory of program dumps");
    let examples: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(2);
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("a readable directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    files.sort();
    let mut ok: BTreeMap<String, usize> = BTreeMap::new();
    let mut total: BTreeMap<String, usize> = BTreeMap::new();
    let mut reasons: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut invalid = 0;
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(&path).expect("readable file");
        let kind = name.split('_').next().unwrap_or("").to_string();
        *total.entry(kind.clone()).or_default() += 1;
        let result = if kind == "COMPUTE" {
            let mut cfg = Config::new(Stage::Kernel);
            cfg.block = [64, 1, 1];
            translate_sm86(&bytes, None, &cfg)
        } else {
            let sph = match Sph::parse(&bytes) {
                Ok(s) => s,
                Err(e) => {
                    reasons
                        .entry(format!("header: {e}"))
                        .or_default()
                        .push(name);
                    continue;
                }
            };
            let stage = match sph.stage() {
                Some(HeaderStage::Vertex) => Stage::Vertex,
                Some(HeaderStage::Pixel) => Stage::Pixel,
                other => {
                    reasons
                        .entry(format!("the stage {other:?}"))
                        .or_default()
                        .push(name);
                    continue;
                }
            };
            let cfg = Config::new(stage);
            if sph.version() == 4 {
                translate_sm86(&bytes[sph.code_offset()..], Some(&sph), &cfg)
            } else {
                translate(Code(&bytes[SPH_BYTES..]), Some(&sph), &cfg)
            }
        };
        match result {
            Ok(t) => match validate::validate(&t.spirv) {
                Ok(_) => *ok.entry(kind).or_default() += 1,
                Err(e) => {
                    invalid += 1;
                    let first = e.lines().next().unwrap_or("").to_string();
                    reasons
                        .entry(format!("INVALID: {first}"))
                        .or_default()
                        .push(name);
                }
            },
            Err(e) => {
                // The reason without the place, so the same reason counts once.
                let message = e.message.clone();
                reasons.entry(message).or_default().push(name);
            }
        }
    }
    for (kind, n) in &total {
        println!(
            "{kind}: {} of {n} translated and valid",
            ok.get(kind).copied().unwrap_or(0)
        );
    }
    println!("{invalid} translated to invalid SPIR-V");
    let mut sorted: Vec<_> = reasons.into_iter().collect();
    sorted.sort_by_key(|(_, names)| std::cmp::Reverse(names.len()));
    for (reason, names) in sorted {
        println!("{:>5}  {reason}", names.len());
        for n in names.iter().take(examples) {
            println!("         {n}");
        }
    }
}
