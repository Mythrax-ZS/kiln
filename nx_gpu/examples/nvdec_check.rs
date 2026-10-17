//! Compare the picture setups and probability tables a run wrote (`NX_MEDIA_DUMP`) with what the
//! headers of the stream they came from say, frame by frame (decoding order).
//!
//! usage: cargo run --release -p nx_gpu --example nvdec_check -- <dump dir> <in.webm>

use nx_gpu::media::vp9::nvdec::parse_picture;
use nx_gpu::media::vp9::probs::Probs;
use nx_gpu::media::vp9::Decoder;

#[path = "vp9_frames.rs"]
#[allow(dead_code)]
mod frames;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let data = std::fs::read(&args[2]).unwrap();
    let mut dec = Decoder::new();
    let mut headers = Vec::new();
    for chunk in frames::webm_frames(&data) {
        dec.decode(chunk).unwrap();
        headers.extend(dec.take_headers());
    }
    let mut job = 0;
    let mut shown = 0;
    for (i, (h, p)) in headers.iter().enumerate() {
        // The next decoder job.
        let pic = loop {
            if let Ok(pic) = std::fs::read(dir.join(format!("job_{job:04}_pic.bin"))) {
                break Some(pic);
            }
            if !dir.join(format!("job_{job:04}.txt")).exists() {
                break None;
            }
            job += 1;
        };
        let Some(pic) = pic else { break };
        let table = std::fs::read(dir.join(format!("job_{job:04}_probs.bin"))).unwrap();
        job += 1;
        let (probs, tree, pred) = Probs::from_nvdec(&table).unwrap();
        let n = parse_picture(&pic, tree, pred).unwrap();
        let g = &n.header;
        let mut diffs = Vec::new();
        macro_rules! cmp {
            ($($f:ident).+) => {
                if format!("{:?}", h.$($f).+) != format!("{:?}", g.$($f).+) {
                    diffs.push(format!("{}: stream {:?} setup {:?}", stringify!($($f).+), h.$($f).+, g.$($f).+));
                }
            };
        }
        cmp!(key_frame);
        cmp!(intra_only);
        cmp!(error_resilient);
        cmp!(width);
        cmp!(height);
        cmp!(sign_bias);
        cmp!(base_q_idx);
        cmp!(delta_q_y_dc);
        cmp!(delta_q_uv_dc);
        cmp!(delta_q_uv_ac);
        cmp!(lossless);
        cmp!(tx_mode);
        cmp!(allow_high_precision_mv);
        cmp!(interp_filter);
        cmp!(reference_mode);
        cmp!(tile_cols_log2);
        cmp!(tile_rows_log2);
        cmp!(lf.level);
        cmp!(lf.sharpness);
        cmp!(lf.delta_enabled);
        cmp!(lf.ref_deltas);
        cmp!(lf.mode_deltas);
        cmp!(seg.enabled);
        if probs != *p {
            diffs.push("probabilities differ".into());
        }
        if h.show_frame {
            shown += 1;
        }
        if !diffs.is_empty() || i < 3 {
            println!(
                "frame {i} (shown {shown}, show {}, flags {:#x}): {}",
                h.show_frame,
                n.flags,
                if diffs.is_empty() {
                    "same".to_string()
                } else {
                    diffs.join("; ")
                }
            );
        }
    }
}
