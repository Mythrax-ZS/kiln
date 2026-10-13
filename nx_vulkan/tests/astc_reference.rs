//! The ASTC decoder against ARM's reference decoder (`astcenc`), on random blocks of every
//! footprint. The tool is external: `RECOMP_ASTCENC` names it (else it is looked for on the
//! `PATH`); without it the test says so and passes.

use nx_vulkan::astc::{decode_image, inspect, ERROR_COLOR};
use std::path::{Path, PathBuf};
use std::process::Command;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

const FOOTPRINTS: [(u32, u32); 14] = [
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
];

fn random_u128(rng: &mut Rng) -> u128 {
    (0..4).fold(0u128, |acc, _| (acc << 32) | u128::from(rng.next()))
}

fn set(v: &mut u128, lo: usize, len: usize, value: u32) {
    let mask = ((1u128 << len) - 1) << lo;
    *v = (*v & !mask) | ((u128::from(value) << lo) & mask);
}

/// A random block: mostly random bits (many are legal, many are not), with void-extent blocks and
/// blocks of low dynamic range endpoint modes mixed in.
fn random_block(rng: &mut Rng, i: usize) -> [u8; 16] {
    let mut v = random_u128(rng);
    let modes = [0, 1, 4, 5, 6, 8, 9, 10, 12, 13];
    match i % 16 {
        0 => {
            set(&mut v, 0, 9, 0x1FC);
            set(&mut v, 10, 2, 3);
            if i.is_multiple_of(32) {
                for lo in [12, 25, 38, 51] {
                    set(&mut v, lo, 13, 0x1FFF);
                }
            } else {
                set(&mut v, 12, 13, 100);
                set(&mut v, 25, 13, 200 + (rng.next() & 0x3FF));
                set(&mut v, 38, 13, 7);
                set(&mut v, 51, 13, 900);
            }
        }
        1..=3 => {
            set(&mut v, 11, 2, 0);
            set(&mut v, 13, 4, modes[(rng.next() % 10) as usize]);
        }
        4..=6 => {
            set(&mut v, 11, 2, 1 + rng.next() % 3);
            set(&mut v, 23, 2, 0);
            set(&mut v, 25, 4, modes[(rng.next() % 10) as usize]);
        }
        _ => {}
    }
    v.to_le_bytes()
}

fn tool() -> Option<PathBuf> {
    let path = std::env::var_os("RECOMP_ASTCENC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("astcenc"));
    Command::new(&path)
        .arg("-version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| path)
}

/// The RGBA8 texels of an uncompressed KTX file (as `astcenc` writes them), top row first. The
/// tool writes only the components a picture needs: gray, gray with alpha, color, or color with
/// alpha; the others read as 0 for color and 255 for alpha.
fn read_ktx(path: &Path, width: usize, height: usize) -> Vec<u8> {
    let b = std::fs::read(path).expect("the reference decoder wrote its picture");
    assert_eq!(&b[1..7], b"KTX 11", "a KTX file");
    let word = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
    assert_eq!((word(36), word(40)), (width, height), "the picture size");
    let components = match word(24) {
        0x1903 | 0x1909 => 1,
        0x190A | 0x8227 => 2,
        0x1907 => 3,
        0x1908 => 4,
        other => panic!("KTX format {other:#x}"),
    };
    let kv = word(60);
    let start = 64 + kv + 4;
    let data = &b[start..start + width * height * components];
    let mut out = vec![255u8; width * height * 4];
    for i in 0..width * height {
        let p = &data[i * components..(i + 1) * components];
        let rgba = match components {
            1 => [p[0], p[0], p[0], 255],
            2 => [p[0], p[0], p[0], p[1]],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        };
        out[i * 4..i * 4 + 4].copy_from_slice(&rgba);
    }
    out
}

fn write_astc(path: &Path, bw: u32, bh: u32, width: u32, height: u32, blocks: &[u8]) {
    let mut f = vec![0x13, 0xAB, 0xA1, 0x5C, bw as u8, bh as u8, 1];
    for v in [width, height, 1] {
        f.extend_from_slice(&v.to_le_bytes()[..3]);
    }
    f.extend_from_slice(blocks);
    std::fs::write(path, f).unwrap();
}

/// How the blocks of a run compare.
#[derive(Default, Debug)]
struct Tally {
    both_refused: usize,
    only_we_refused: usize,
    only_they_refused: usize,
    same: usize,
    different: usize,
    worst: i32,
    /// Legal blocks by what they are made of: (same, different).
    kinds: std::collections::BTreeMap<String, (usize, usize)>,
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        self.both_refused += o.both_refused;
        self.only_we_refused += o.only_we_refused;
        self.only_they_refused += o.only_they_refused;
        self.same += o.same;
        self.different += o.different;
        self.worst = self.worst.max(o.worst);
        for (k, (s, d)) in &o.kinds {
            let e = self.kinds.entry(k.clone()).or_default();
            e.0 += s;
            e.1 += d;
        }
    }

    fn disagreements(&self) -> usize {
        self.only_we_refused + self.only_they_refused + self.different
    }
}

/// The two decodings of a run of blocks (RGBA8, `width` texels per row).
struct Pictures<'a> {
    blocks: &'a [u8],
    ours: &'a [u8],
    theirs: &'a [u8],
    width: usize,
    block: (usize, usize),
}

/// Compare the two decodings block by block, saying the first few disagreements under `name`.
fn compare(name: &str, pictures: &Pictures<'_>, shown: &mut usize) -> Tally {
    let Pictures {
        blocks,
        ours,
        theirs,
        width,
        block: (bw, bh),
    } = *pictures;
    let across = width / bw;
    let mut tally = Tally::default();
    for b in 0..blocks.len() / 16 {
        let (bx, by) = (b % across, b / across);
        let texel = |image: &[u8], tx: usize, ty: usize| -> [u8; 4] {
            let i = ((by * bh + ty) * width + bx * bw + tx) * 4;
            [image[i], image[i + 1], image[i + 2], image[i + 3]]
        };
        let mut we_refused = true;
        let mut they_refused = true;
        let mut first_difference = None;
        for ty in 0..bh {
            for tx in 0..bw {
                let (o, t) = (texel(ours, tx, ty), texel(theirs, tx, ty));
                we_refused &= o == ERROR_COLOR;
                they_refused &= t == ERROR_COLOR;
                for c in 0..4 {
                    let d = (i32::from(o[c]) - i32::from(t[c])).abs();
                    tally.worst = tally.worst.max(d);
                    // The sRGB color channels are cut where the reference rounds.
                    if d > 0 && first_difference.is_none() {
                        first_difference = Some((tx, ty, o, t));
                    }
                }
            }
        }
        let mut one = [0u8; 16];
        one.copy_from_slice(&blocks[b * 16..b * 16 + 16]);
        if !(we_refused && they_refused) {
            // What this legal block is made of, and whether it came out right.
            let key = match inspect(&one, bw, bh) {
                Ok(info) if info.void_extent => "void extent".to_string(),
                Ok(info) => format!(
                    "{} partitions {:?}, {} plane(s), {} weight levels, {} endpoint levels, {} hdr",
                    info.partitions,
                    &info.modes[..info.partitions],
                    1 + usize::from(info.dual_plane),
                    info.weight_levels,
                    info.endpoint_levels,
                    info.hdr_partitions
                ),
                Err(why) => format!("refused: {why:?}"),
            };
            let e = tally.kinds.entry(key).or_default();
            if we_refused != they_refused || first_difference.is_some() {
                e.1 += 1;
            } else {
                e.0 += 1;
            }
        }
        let kind = match (we_refused, they_refused, first_difference) {
            (true, true, _) => {
                tally.both_refused += 1;
                continue;
            }
            (true, false, _) => {
                tally.only_we_refused += 1;
                "we refused, they decoded"
            }
            (false, true, _) => {
                tally.only_they_refused += 1;
                "they refused, we decoded"
            }
            (false, false, Some(_)) => {
                tally.different += 1;
                "both decoded, differently"
            }
            (false, false, None) => {
                tally.same += 1;
                continue;
            }
        };
        if *shown < 4 {
            *shown += 1;
            eprintln!(
                "{name}: block {b} {:02x?}: {kind} ({:?}); first difference {:?}",
                one,
                inspect(&one, bw, bh),
                first_difference
            );
        }
    }
    tally
}

#[test]
fn random_blocks_decode_as_in_the_reference_decoder() {
    let Some(tool) = tool() else {
        if std::env::var("RECOMP_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
            panic!("no astcenc (set RECOMP_ASTCENC) and RECOMP_REQUIRE_EXTERNAL=1");
        }
        eprintln!("SKIPPED: no astcenc (set RECOMP_ASTCENC to the reference decoder)");
        return;
    };
    let batches: usize = std::env::var("RECOMP_ASTC_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let dir = std::env::temp_dir().join(format!("nx_astc_reference_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut failures = Vec::new();
    let only: Option<usize> = std::env::var("RECOMP_ASTC_ONLY")
        .ok()
        .and_then(|v| v.parse().ok());
    for (n, (bw, bh)) in FOOTPRINTS.iter().enumerate() {
        if only.is_some_and(|o| o != n) {
            continue;
        }
        for srgb in [false, true] {
            if srgb && n % 3 != 0 && only.is_none() {
                continue;
            }
            let name = format!("{bw}x{bh}{}", if srgb { " sRGB" } else { "" });
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (n as u64 * 7919 + u64::from(srgb)));
            let mut total = Tally::default();
            let mut shown = 0;
            for batch in 0..batches {
                let across = 64usize;
                let count = 4096usize;
                let blocks: Vec<u8> = (0..count)
                    .flat_map(|i| random_block(&mut rng, i + 5 * batch))
                    .collect();
                let (width, height) = (across as u32 * bw, (count / across) as u32 * bh);
                let input = dir.join("in.astc");
                let output = dir.join("out.ktx");
                write_astc(&input, *bw, *bh, width, height, &blocks);
                let run = Command::new(&tool)
                    .arg(if srgb { "-ds" } else { "-dl" })
                    .arg(&input)
                    .arg(&output)
                    .output()
                    .expect("runs the reference decoder");
                assert!(
                    run.status.success(),
                    "astcenc failed: {}",
                    String::from_utf8_lossy(&run.stderr)
                );
                let theirs = read_ktx(&output, width as usize, height as usize);
                let ours = decode_image(&blocks, width, height, *bw, *bh, srgb);
                let t = compare(
                    &name,
                    &Pictures {
                        blocks: &blocks,
                        ours: &ours,
                        theirs: &theirs,
                        width: width as usize,
                        block: (*bw as usize, *bh as usize),
                    },
                    &mut shown,
                );
                total.add(&t);
            }
            eprintln!(
                "ASTC {name}: {} same, {} different, {} refused by both, we refused {}, they refused {}, worst difference {}",
                total.same, total.different, total.both_refused, total.only_we_refused, total.only_they_refused, total.worst
            );
            for (k, (s, d)) in &total.kinds {
                if *d > 0 {
                    eprintln!("    {k}: {s} same, {d} different");
                }
            }
            if total.disagreements() > 0 {
                failures.push(format!("{name}: {} blocks differ", total.disagreements()));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{failures:?}");
}

/// A 4x2 grid (weights `w0` and `w1` for the two planes, 2 bits each) in a 4x4 block, mode 12
/// endpoints stored as 8 bits, color component `ccs` on the second plane.
fn dual_plane_block(w0: [u8; 8], w1: [u8; 8], ccs: u32, e0: [u8; 4], e1: [u8; 4]) -> [u8; 16] {
    let mut v = 0u128;
    set(&mut v, 0, 11, 2 | (1 << 10)); // 4x2 grid, 2-bit weights, two planes
    set(&mut v, 11, 2, 0);
    set(&mut v, 13, 4, 12);
    let values = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2], e0[3], e1[3]];
    for (i, x) in values.iter().enumerate() {
        set(&mut v, 17 + 8 * i, 8, u32::from(*x));
    }
    for i in 0..8 {
        for (plane, w) in [(0, w0[i]), (1, w1[i])] {
            let p = 2 * (2 * i + plane); // the stream position of the weight's low bit
            set(&mut v, 127 - p, 1, u32::from(w & 1));
            set(&mut v, 126 - p, 1, u32::from(w >> 1 & 1));
        }
    }
    set(&mut v, 128 - 32 - 2, 2, ccs);
    v.to_le_bytes()
}

#[test]
fn dual_plane_blocks_decode_as_in_the_reference_decoder() {
    let Some(tool) = tool() else {
        eprintln!("SKIPPED: no astcenc (set RECOMP_ASTCENC to the reference decoder)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("nx_astc_dual_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng(12345);
    let mut blocks = Vec::new();
    for ccs in 0..4u32 {
        for _ in 0..8 {
            let w0: [u8; 8] = std::array::from_fn(|_| (rng.next() % 4) as u8);
            let w1: [u8; 8] = std::array::from_fn(|_| (rng.next() % 4) as u8);
            let e0: [u8; 4] = std::array::from_fn(|_| (rng.next() % 100) as u8);
            let e1: [u8; 4] = std::array::from_fn(|_| (150 + rng.next() % 100) as u8);
            blocks.extend_from_slice(&dual_plane_block(w0, w1, ccs, e0, e1));
        }
    }
    let count = blocks.len() / 16;
    let (width, height) = (count as u32 * 4, 4u32);
    let input = dir.join("in.astc");
    let output = dir.join("out.ktx");
    write_astc(&input, 4, 4, width, height, &blocks);
    let run = Command::new(&tool)
        .arg("-dl")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(run.status.success());
    let theirs = read_ktx(&output, width as usize, height as usize);
    let ours = decode_image(&blocks, width, height, 4, 4, false);
    let mut shown = 0;
    let t = compare(
        "dual",
        &Pictures {
            blocks: &blocks,
            ours: &ours,
            theirs: &theirs,
            width: width as usize,
            block: (4, 4),
        },
        &mut shown,
    );
    eprintln!("dual plane: {t:?}");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(t.disagreements(), 0);
}

/// A 4x4 grid of 2-bit weights, one plane, one partition of `mode` with its values stored as
/// 8 bits each.
fn single_plane_block(weights: [u8; 16], mode: u32, values: &[u8]) -> [u8; 16] {
    let mut v = 0u128;
    set(&mut v, 0, 11, (2 << 5) | 2);
    set(&mut v, 11, 2, 0);
    set(&mut v, 13, 4, mode);
    for (i, x) in values.iter().enumerate() {
        set(&mut v, 17 + 8 * i, 8, u32::from(*x));
    }
    for (i, w) in weights.iter().enumerate() {
        set(&mut v, 127 - 2 * i, 1, u32::from(w & 1));
        set(&mut v, 126 - 2 * i, 1, u32::from(w >> 1 & 1));
    }
    v.to_le_bytes()
}

#[test]
fn endpoint_modes_decode_as_in_the_reference_decoder() {
    let Some(tool) = tool() else {
        eprintln!("SKIPPED: no astcenc (set RECOMP_ASTCENC to the reference decoder)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("nx_astc_modes_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng(777);
    let mut failures = Vec::new();
    for mode in [0u32, 1, 4, 5, 6, 8, 9, 10, 12, 13] {
        let n = 2 * ((mode as usize >> 2) + 1);
        let mut blocks = Vec::new();
        let mut inputs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..512 {
            let w: [u8; 16] = std::array::from_fn(|_| (rng.next() % 4) as u8);
            let values: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            blocks.extend_from_slice(&single_plane_block(w, mode, &values));
            inputs.push(values);
        }
        let count = blocks.len() / 16;
        let (width, height) = (count as u32 * 4, 4u32);
        let input = dir.join("in.astc");
        let output = dir.join("out.ktx");
        write_astc(&input, 4, 4, width, height, &blocks);
        let run = Command::new(&tool)
            .arg("-dl")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(run.status.success());
        let theirs = read_ktx(&output, width as usize, height as usize);
        let ours = decode_image(&blocks, width, height, 4, 4, false);
        let mut bad = 0;
        for (b, values) in inputs.iter().enumerate() {
            let differs = (0..16).any(|t| {
                let (tx, ty) = (t % 4, t / 4);
                let i = (ty * width as usize + b * 4 + tx) * 4;
                ours[i..i + 4] != theirs[i..i + 4]
            });
            if differs {
                bad += 1;
                if bad <= 2 {
                    let i = (b * 4) * 4;
                    eprintln!(
                        "mode {mode}: values {:?}: texel 0 ours {:?} theirs {:?}",
                        values,
                        &ours[i..i + 4],
                        &theirs[i..i + 4]
                    );
                }
            }
        }
        eprintln!("mode {mode}: {bad} of {count} blocks differ");
        if bad > 0 {
            failures.push(mode);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "modes that differ: {failures:?}");
}
