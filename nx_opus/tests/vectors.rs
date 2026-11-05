//! Conformance with the Opus test vectors (RFC 6716 appendix A.4, renewed by RFC 8251 section
//! 11): every packet's final range must match the encoder's, and the decoded audio must pass the
//! reference's quality metric (`opus_compare`, re-implemented below) against the reference
//! output, decoding to stereo (`testvectorNN.dec`) and to mono (`.dec` or `m.dec`).
//!
//! `KILN_OPUS_VECTORS` names the directory of `testvector01.bit` and the rest; without it the
//! test says SKIPPED and passes (`KILN_REQUIRE_EXTERNAL=1` makes that a failure). Nothing from
//! the vectors is stored.

use std::path::Path;

use nx_opus::Decoder;

/// The packets of a `.bit` file: a big-endian length, the encoder's final range, the packet.
fn packets(data: &[u8]) -> Vec<(&[u8], u32)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 8 <= data.len() {
        let len = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        let range = u32::from_be_bytes(data[at + 4..at + 8].try_into().unwrap());
        out.push((&data[at + 8..at + 8 + len], range));
        at += 8 + len;
    }
    out
}

/// Decode as `opus_demo -d 48000 <channels>` does; returns the samples and the packets whose
/// final range differed.
fn decode(bits: &[u8], channels: usize) -> (Vec<i16>, Vec<usize>) {
    let mut dec = Decoder::new(48000, channels).unwrap();
    let mut out = Vec::new();
    let mut bad = Vec::new();
    let mut buf = vec![0i16; 5760 * channels];
    let mut lost_prev = true;
    for (i, (pkt, range)) in packets(bits).into_iter().enumerate() {
        let lost = pkt.is_empty();
        let n = dec
            .decode_i16(if lost { None } else { Some(pkt) }, &mut buf, false)
            .unwrap_or_else(|e| panic!("packet {i}: {e:?}"));
        out.extend_from_slice(&buf[..n * channels]);
        if range != 0 && !lost && !lost_prev && dec.final_range() != range {
            bad.push(i);
        }
        lost_prev = lost;
    }
    (out, bad)
}

#[derive(Clone, Copy)]
struct C {
    re: f32,
    im: f32,
}

/// A forward DFT of `out.len()` points of `input[start + k * stride]` (decimation in time,
/// mixed radix), with the twiddles `tw[k] = e^(-2 pi i k / N)` of the full size.
fn fft(input: &[C], out: &mut [C], start: usize, stride: usize, factors: &[usize], tw: &[C]) {
    let n = out.len();
    let p = factors[0];
    let m = n / p;
    if m == 1 {
        for (j, o) in out.iter_mut().enumerate() {
            *o = input[start + j * stride];
        }
    } else {
        for j in 0..p {
            fft(
                input,
                &mut out[j * m..(j + 1) * m],
                start + j * stride,
                stride * p,
                &factors[1..],
                tw,
            );
        }
    }
    let total = tw.len();
    let step = total / n;
    let mul = |a: C, b: C| C {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    };
    let mut tmp = [C { re: 0.0, im: 0.0 }; 5];
    for k in 0..m {
        for j in 0..p {
            tmp[j] = mul(out[j * m + k], tw[(j * k * step) % total]);
        }
        for q in 0..p {
            let mut acc = tmp[0];
            for (j, t) in tmp.iter().enumerate().take(p).skip(1) {
                let v = mul(*t, tw[(j * q * m * step) % total]);
                acc.re += v.re;
                acc.im += v.im;
            }
            out[q * m + k] = acc;
        }
    }
}

const NBANDS: usize = 21;
const NFREQS: usize = 240;
const BANDS: [usize; NBANDS + 1] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 200,
];
const WIN: usize = 480;
const STEP: usize = 120;

/// The power spectrum of each windowed frame (plus a floor), and each band's mean (`band_energy`).
fn band_energy(
    input: &[f32],
    channels: usize,
    frames: usize,
    want_bands: bool,
) -> (Vec<f32>, Vec<f32>) {
    let window: Vec<f32> = (0..WIN)
        .map(|j| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI / (WIN - 1) as f32) * j as f32).cos())
        .collect();
    let tw: Vec<C> = (0..WIN)
        .map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / WIN as f64;
            C {
                re: a.cos() as f32,
                im: a.sin() as f32,
            }
        })
        .collect();
    let factors = [4, 4, 2, 3, 5];
    let mut ps = vec![0f32; frames * NFREQS * channels];
    let mut bands = vec![
        0f32;
        if want_bands {
            frames * NBANDS * channels
        } else {
            0
        }
    ];
    let mut x = vec![C { re: 0.0, im: 0.0 }; WIN];
    let mut spec = vec![C { re: 0.0, im: 0.0 }; WIN];
    for f in 0..frames {
        for c in 0..channels {
            for k in 0..WIN {
                x[k] = C {
                    re: window[k] * input[(f * STEP + k) * channels + c],
                    im: 0.0,
                };
            }
            fft(&x, &mut spec, 0, 1, &factors, &tw);
            for (j, s) in spec.iter().enumerate().take(BANDS[NBANDS]) {
                ps[(f * NFREQS + j) * channels + c] = s.re * s.re + s.im * s.im + 100000.0;
            }
        }
        if want_bands {
            for b in 0..NBANDS {
                for c in 0..channels {
                    let p: f32 = (BANDS[b]..BANDS[b + 1])
                        .map(|j| ps[(f * NFREQS + j) * channels + c])
                        .sum();
                    bands[(f * NBANDS + b) * channels + c] = p / (BANDS[b + 1] - BANDS[b]) as f32;
                }
            }
        }
    }
    (ps, bands)
}

/// The quality metric of `opus_compare` at 48 kHz: `reference` is stereo, `decoded` has
/// `channels`; returns Q (a vector passes when Q >= 0).
fn quality(reference: &[i16], decoded: &[i16], channels: usize) -> f64 {
    let mut x: Vec<f32> = reference.iter().map(|&v| f32::from(v)).collect();
    if channels == 1 {
        x = x
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| 0.5 * (p[0] + p[1]))
            .collect();
    }
    let y: Vec<f32> = decoded.iter().map(|&v| f32::from(v)).collect();
    let len = x.len() / channels;
    assert_eq!(len, y.len() / channels, "sample counts differ");
    assert!(len >= WIN);
    let frames = (len - WIN + STEP) / STEP;
    let (mut xs, mut xb) = band_energy(&x, channels, frames, true);
    let (mut ys, _) = band_energy(&y, channels, frames, false);
    let at = |f: usize, b: usize, c: usize| (f * NBANDS + b) * channels + c;
    for f in 0..frames {
        // Frequency masking, low to high (10 dB/Bark) and high to low (15 dB/Bark).
        for b in 1..NBANDS {
            for c in 0..channels {
                xb[at(f, b, c)] += 0.1 * xb[at(f, b - 1, c)];
            }
        }
        for b in (0..NBANDS - 1).rev() {
            for c in 0..channels {
                xb[at(f, b, c)] += 0.03 * xb[at(f, b + 1, c)];
            }
        }
        // Temporal masking: -3 dB per 2.5 ms.
        if f > 0 {
            for b in 0..NBANDS {
                for c in 0..channels {
                    xb[at(f, b, c)] += 0.5 * xb[at(f - 1, b, c)];
                }
            }
        }
        // Some cross-talk.
        if channels == 2 {
            for b in 0..NBANDS {
                let (l, r) = (xb[at(f, b, 0)], xb[at(f, b, 1)]);
                xb[at(f, b, 0)] += 0.01 * r;
                xb[at(f, b, 1)] += 0.01 * l;
            }
        }
        for b in 0..NBANDS {
            for j in BANDS[b]..BANDS[b + 1] {
                for c in 0..channels {
                    let m = 0.1 * xb[at(f, b, c)];
                    xs[(f * NFREQS + j) * channels + c] += m;
                    ys[(f * NFREQS + j) * channels + c] += m;
                }
            }
        }
    }
    // Average consecutive frames.
    for j in 0..BANDS[NBANDS] {
        for c in 0..channels {
            let (mut xt, mut yt) = (xs[j * channels + c], ys[j * channels + c]);
            for f in 1..frames {
                let k = (f * NFREQS + j) * channels + c;
                let (x2, y2) = (xs[k], ys[k]);
                xs[k] += xt;
                ys[k] += yt;
                xt = x2;
                yt = y2;
            }
        }
    }
    let mut err = 0f64;
    for f in 0..frames {
        let mut ef = 0f64;
        for b in 0..NBANDS {
            let mut eb = 0f64;
            for j in BANDS[b]..BANDS[b + 1] {
                for c in 0..channels {
                    let k = (f * NFREQS + j) * channels + c;
                    let re = ys[k] / xs[k];
                    let mut im = re - re.ln() - 1.0;
                    // Less sensitive around the SILK/CELT cross-over.
                    if (79..=81).contains(&j) {
                        im *= 0.1;
                    }
                    if j == 80 {
                        im *= 0.1;
                    }
                    eb += f64::from(im);
                }
            }
            eb /= ((BANDS[b + 1] - BANDS[b]) * channels) as f64;
            ef += eb * eb;
        }
        ef /= NBANDS as f64;
        ef *= ef;
        err += ef * ef;
    }
    let err = (err / frames as f64).powf(1.0 / 16.0);
    100.0 * (1.0 - 0.5 * (1.0 + err).ln() / 1.13f64.ln())
}

fn read_pcm(path: &Path) -> Vec<i16> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

#[test]
fn the_test_vectors_decode_within_the_reference_tolerance() {
    let Some(dir) = std::env::var_os("KILN_OPUS_VECTORS") else {
        assert!(
            std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
            "KILN_OPUS_VECTORS is not set and KILN_REQUIRE_EXTERNAL is"
        );
        println!("SKIPPED: the Opus test vectors (set KILN_OPUS_VECTORS to their directory)");
        return;
    };
    let dir = Path::new(&dir);
    let only: Option<Vec<String>> = std::env::var("KILN_OPUS_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_owned).collect());
    let mut failures = Vec::new();
    for n in 1..=12 {
        let name = format!("testvector{n:02}");
        if only
            .as_ref()
            .is_some_and(|o| !o.iter().any(|v| *v == format!("{n:02}")))
        {
            continue;
        }
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).unwrap();
        let reference = read_pcm(&dir.join(format!("{name}.dec")));
        let reference_m = read_pcm(&dir.join(format!("{name}m.dec")));
        let (stereo, bad_stereo) = decode(&bits, 2);
        let q_stereo = quality(&reference, &stereo, 2);
        let (mono, bad_mono) = decode(&bits, 1);
        let q_mono = quality(&reference, &mono, 1).max(quality(&reference_m, &mono, 1));
        println!(
            "{name}: stereo Q {q_stereo:.1}, mono Q {q_mono:.1}, final range differs in {} and {} packets",
            bad_stereo.len(),
            bad_mono.len()
        );
        if !bad_stereo.is_empty() || !bad_mono.is_empty() {
            failures.push(format!(
                "{name}: final range (first at packet {:?})",
                bad_stereo.first().or(bad_mono.first())
            ));
        }
        if q_stereo < 0.0 || q_mono < 0.0 {
            failures.push(format!("{name}: quality {q_stereo:.1} / {q_mono:.1}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
