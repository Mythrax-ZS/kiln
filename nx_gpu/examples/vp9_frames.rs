//! Decode a VP9 stream (an IVF or a WebM file) with the decoder of `nx_gpu::media::vp9` and
//! write the frames it shows, as raw I420 (Y, then U, then V of each frame), to compare with
//! another decoder.
//!
//! usage: cargo run --release -p nx_gpu --example vp9_frames -- <in.ivf|in.webm> <out.yuv> [frames]

use nx_gpu::media::vp9::Decoder;
use std::io::Write;

pub fn ivf_frames(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let header = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut at = header;
    while at + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
        at += 12;
        if at + size > data.len() {
            break;
        }
        out.push(&data[at..at + size]);
        at += size;
    }
    out
}

/// An EBML variable-size integer: (value, length); `marker` keeps the length bits (element ids).
fn vint(b: &[u8], at: usize, marker: bool) -> (u64, usize) {
    let first = b[at];
    let len = first.leading_zeros() as usize + 1;
    let mut v = if marker {
        u64::from(first)
    } else {
        u64::from(first) & ((1u64 << (8 - len)) - 1)
    };
    for k in 1..len {
        v = (v << 8) | u64::from(b[at + k]);
    }
    (v, len)
}

pub fn webm_frames(data: &[u8]) -> Vec<&[u8]> {
    // Segment, Cluster, BlockGroup hold the blocks; SimpleBlock and Block are the frames.
    const CONTAINERS: [u64; 3] = [0x1853_8067, 0x1F43_B675, 0xA0];
    let mut out = Vec::new();
    let mut stack = vec![(0usize, data.len())];
    while let Some((mut at, end)) = stack.pop() {
        let mut children = Vec::new();
        while at + 2 <= end {
            let (id, n) = vint(data, at, true);
            let (mut size, m) = vint(data, at + n, false);
            let body = at + n + m;
            if size == (1u64 << (7 * m)) - 1 {
                size = (end - body) as u64;
            }
            let stop = (body + size as usize).min(end);
            if CONTAINERS.contains(&id) {
                children.push((body, stop));
            } else if id == 0xA3 || id == 0xA1 {
                let (_, k) = vint(data, body, false);
                out.push((body, &data[body + k + 3..stop]));
            }
            at = stop;
        }
        stack.extend(children.into_iter().rev());
    }
    out.sort_by_key(|(at, _)| *at);
    out.into_iter().map(|(_, f)| f).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: vp9_frames <in.ivf|in.webm> <out.yuv> [frames]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read the stream");
    let limit: usize = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let chunks = if data.starts_with(b"DKIF") {
        ivf_frames(&data)
    } else {
        webm_frames(&data)
    };
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("create the output"));
    let mut dec = Decoder::new();
    let mut shown = 0usize;
    let start = std::time::Instant::now();
    for (i, chunk) in chunks.iter().enumerate() {
        if shown >= limit {
            break;
        }
        match dec.decode(chunk) {
            Ok(frames) => {
                if std::env::var_os("VP9_HEADERS").is_some() {
                    for (h, _) in dec.take_headers() {
                        eprintln!(
                            "chunk {i}: {}x{} key {} show {} tile columns log2 {} rows log2 {}",
                            h.width,
                            h.height,
                            h.key_frame,
                            h.show_frame,
                            h.tile_cols_log2,
                            h.tile_rows_log2
                        );
                    }
                }
                for f in frames {
                    for p in &f.planes {
                        out.write_all(&p.visible()).unwrap();
                    }
                    shown += 1;
                }
            }
            Err(e) => {
                eprintln!("chunk {i}: {e}");
                std::process::exit(1);
            }
        }
    }
    eprintln!(
        "{} chunks, {shown} frames shown, {:.1} ms a frame",
        chunks.len(),
        start.elapsed().as_secs_f64() * 1000.0 / shown.max(1) as f64
    );
}
