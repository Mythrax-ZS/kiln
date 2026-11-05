//! Translates every shader program dump in a directory (as `NX_SHADER_DUMP` writes them: a header
//! and its code, or the code of a compute program), validates the results with `spirv-val` when
//! it is installed, and says how many of each stage translated and why the others did not, most
//! frequent reason first. Nothing is written.
//! usage: corpus <dir> [<how many examples of each reason>]

use nx_shader::cfg::Code;
use nx_shader::sph::{Sph, Stage as HeaderStage, SPH_BYTES};
use nx_shader::translate::{translate, translate_sm86, Config, MeshConfig, Stage};
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
            // The largest pipeline the title sets up (A-270), for the programs of mesh pipelines.
            let mesh = MeshConfig {
                threads: 32,
                max_vertices: 256,
                max_primitives: 192,
                topology: 4,
                payload_words: 11 * 32,
                task: true,
                primitive_words: vec![0x20],
            };
            let stage = match sph.stage() {
                Some(HeaderStage::Vertex) => Stage::Vertex,
                Some(HeaderStage::Pixel) => Stage::Pixel,
                // A program in the tessellation slot of a mesh pipeline is its mesh program.
                _ if name.starts_with("TESSELLATION") => Stage::Mesh,
                other => {
                    reasons
                        .entry(format!("the stage {other:?}"))
                        .or_default()
                        .push(name);
                    continue;
                }
            };
            let mut cfg = Config::new(stage);
            if stage == Stage::Mesh {
                cfg.mesh = Some(mesh.clone());
            }
            let first = if sph.version() == 4 {
                translate_sm86(&bytes[sph.code_offset()..], Some(&sph), &cfg)
            } else {
                translate(Code(&bytes[SPH_BYTES..]), Some(&sph), &cfg)
            };
            match first {
                // A vertex program that reads the stage buffer is the task program of a mesh
                // pipeline.
                Err(e) if stage == Stage::Vertex && e.message.contains("ISBE") => {
                    cfg.stage = Stage::Task;
                    cfg.mesh = Some(mesh.clone());
                    match translate_sm86(&bytes[sph.code_offset()..], Some(&sph), &cfg) {
                        // One that writes the map is the mesh program of a pipeline without a
                        // task, which the vertex slot holds then.
                        Err(e) if e.message.contains("region 0") => {
                            cfg.stage = Stage::Mesh;
                            cfg.mesh = Some(MeshConfig {
                                task: false,
                                ..mesh
                            });
                            translate_sm86(&bytes[sph.code_offset()..], Some(&sph), &cfg)
                        }
                        other => other,
                    }
                }
                other => other,
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
