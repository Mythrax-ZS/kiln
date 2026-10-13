//! Translates shader program files (a header and its code) the way a draw would, validates the
//! result with `spirv-val` when it is installed, and says what happened.
//! usage: translate [--dis] [--spv <dir>] <program.bin>...

use nx_shader::cfg::Code;
use nx_shader::sph::{Sph, Stage as HeaderStage, SPH_BYTES};
use nx_shader::translate::{translate, Config, Stage};
use nx_shader::validate;

fn main() {
    let mut dis = false;
    let mut spv_dir: Option<std::path::PathBuf> = None;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dis" => dis = true,
            "--spv" => spv_dir = args.next().map(Into::into),
            _ => files.push(a),
        }
    }
    for path in files {
        let bytes = std::fs::read(&path).expect("readable file");
        let name = std::path::Path::new(&path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let sph = match Sph::parse(&bytes) {
            Ok(s) => s,
            Err(e) => {
                println!("{name}: header: {e}");
                continue;
            }
        };
        let stage = match sph.stage() {
            Some(HeaderStage::Vertex) => Stage::Vertex,
            Some(HeaderStage::Pixel) => Stage::Pixel,
            other => {
                println!("{name}: stage {other:?} is not translated");
                continue;
            }
        };
        let mut cfg = Config::new(stage);
        cfg.debug_names = true;
        match translate(Code(&bytes[SPH_BYTES..]), Some(&sph), &cfg) {
            Ok(t) => {
                let verdict = match validate::validate(&t.spirv) {
                    Ok(Some(())) => "valid".to_string(),
                    Ok(None) => "not validated (no spirv-val)".to_string(),
                    Err(e) => format!("INVALID: {}", e.lines().next().unwrap_or("")),
                };
                println!(
                    "{name}: {stage:?}, {} instructions, {} words of SPIR-V, {verdict}; blocks {}; \
                     cbufs {:?}; textures {}; global memory {}; kills {}",
                    t.info.instructions,
                    t.spirv.len(),
                    if t.info.dispatch_loop { "dispatch loop" } else { "straight line" },
                    t.info
                        .cbuf_extent
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| **e > 0)
                        .map(|(i, e)| format!("c{i}:{e}"))
                        .collect::<Vec<_>>(),
                    t.info.textures.len(),
                    t.info.global_memory,
                    t.info.kills
                );
                if let Some(dir) = &spv_dir {
                    let _ = std::fs::create_dir_all(dir);
                    let bytes: Vec<u8> = t.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let _ = std::fs::write(dir.join(format!("{name}.spv")), bytes);
                }
                if dis {
                    println!("{}", validate::disassemble(&t.spirv).unwrap_or_default());
                }
            }
            Err(e) => println!("{name}: {stage:?}: {e}"),
        }
    }
}
