//! Replay jobs of the console's video engines that a run wrote (`NX_MEDIA_DUMP`): decode each
//! decoder job with `nx_gpu::media::vp9::nvdec::Engine`, and write the frames each converter job
//! converts, in order, as raw I420, to compare with another decoder's output.
//!
//! usage: cargo run --release -p nx_gpu --example nvdec_replay -- <dump dir> <out.yuv>

use nx_gpu::media::vp9::nvdec::Engine;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
    let mut engine = Engine::new();
    let (mut decoded, mut converted) = (0, 0);
    for n in 0.. {
        let Ok(regs) = std::fs::read_to_string(dir.join(format!("job_{n:04}.txt"))) else {
            break;
        };
        let mut r = std::collections::HashMap::new();
        for line in regs.lines() {
            let v: Vec<u32> = line
                .split(' ')
                .map(|x| u32::from_str_radix(x, 16).unwrap())
                .collect();
            r.insert((v[0], v[1]), u64::from(v[2]) << 8);
        }
        if r.contains_key(&(0xF0, 0x300)) {
            let read = |s: &str| std::fs::read(dir.join(format!("job_{n:04}_{s}.bin"))).unwrap();
            let refs = [r[&(0xF0, 0x430)], r[&(0xF0, 0x434)], r[&(0xF0, 0x438)]];
            match engine.decode(
                &read("pic"),
                &read("probs"),
                &read("bits"),
                refs,
                r[&(0xF0, 0x43C)],
            ) {
                Ok(_) => decoded += 1,
                Err(e) => {
                    eprintln!("job {n}: {e}");
                    std::process::exit(1);
                }
            }
        } else if let Some(&luma) = r.get(&(0x5D, 0x400)) {
            let Some(f) = engine.frame(luma) else {
                eprintln!("job {n}: converts a surface nothing was decoded into");
                continue;
            };
            for p in &f.planes {
                out.write_all(&p.visible()).unwrap();
            }
            converted += 1;
        }
    }
    eprintln!("{decoded} frames decoded, {converted} converted");
}
