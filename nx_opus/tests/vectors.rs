//! Conformance with the Opus test vectors (RFC 6716 appendix A.4, renewed by RFC 8251 section
//! 11): every packet's final range must match the encoder's, and the decoded audio must pass the
//! reference's quality metric (`opus_compare`, re-implemented below) against the reference
//! output, decoding to stereo (`testvectorNN.dec`) and to mono (`.dec` or `m.dec`), at each
//! output rate (`opus_compare -r`).
//!
//! `KILN_OPUS_VECTORS` names the directory of `testvector01.bit` and the rest; without it the
//! test says SKIPPED and passes (`KILN_REQUIRE_EXTERNAL=1` makes that a failure).
//! `KILN_OPUS_ONLY=01,07` keeps to some vectors and `KILN_OPUS_RATES=48000,8000` to some rates.
//! Nothing from the vectors is stored.

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

/// Decode as `opus_demo -d <rate> <channels>` does; returns the samples and the packets whose
/// final range differed.
fn decode(bits: &[u8], rate: u32, channels: usize) -> (Vec<i16>, Vec<usize>) {
    let mut dec = Decoder::new(rate, channels).unwrap();
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

/// The power spectrum of each windowed frame plus a floor, `freqs` bins apart (`band_energy`),
/// and the mean of the first `nbands` bands when asked for.
#[allow(clippy::too_many_arguments)]
fn band_energy(
    input: &[f32],
    channels: usize,
    frames: usize,
    window_size: usize,
    step: usize,
    downsample: usize,
    nbands: usize,
    want_bands: bool,
) -> (Vec<f32>, Vec<f32>) {
    let freqs = window_size / 2;
    let window: Vec<f32> = (0..window_size)
        .map(|j| {
            0.5 - 0.5 * ((2.0 * std::f32::consts::PI / (window_size - 1) as f32) * j as f32).cos()
        })
        .collect();
    let tw: Vec<C> = (0..window_size)
        .map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / window_size as f64;
            C {
                re: a.cos() as f32,
                im: a.sin() as f32,
            }
        })
        .collect();
    let mut factors = Vec::new();
    let mut rest = window_size;
    for p in [4, 2, 3, 5] {
        while rest.is_multiple_of(p) {
            factors.push(p);
            rest /= p;
        }
    }
    assert_eq!(rest, 1);
    let mut ps = vec![0f32; frames * freqs * channels];
    let mut bands = vec![
        0f32;
        if want_bands {
            frames * NBANDS * channels
        } else {
            0
        }
    ];
    let mut x = vec![C { re: 0.0, im: 0.0 }; window_size];
    let mut spec = vec![C { re: 0.0, im: 0.0 }; window_size];
    for f in 0..frames {
        for c in 0..channels {
            for k in 0..window_size {
                x[k] = C {
                    re: window[k] * input[(f * step + k) * channels + c],
                    im: 0.0,
                };
            }
            fft(&x, &mut spec, 0, 1, &factors, &tw);
            for (j, s) in spec.iter().enumerate().take(BANDS[nbands]) {
                let (re, im) = (s.re * downsample as f32, s.im * downsample as f32);
                ps[(f * freqs + j) * channels + c] = re * re + im * im + 100000.0;
            }
        }
        if want_bands {
            for b in 0..nbands {
                for c in 0..channels {
                    let p: f32 = (BANDS[b]..BANDS[b + 1])
                        .map(|j| ps[(f * freqs + j) * channels + c])
                        .sum();
                    bands[(f * NBANDS + b) * channels + c] = p / (BANDS[b + 1] - BANDS[b]) as f32;
                }
            }
        }
    }
    (ps, bands)
}

/// The quality metric of `opus_compare -r <rate>`: `reference` is stereo at 48 kHz, `decoded`
/// has `channels` at `rate`; returns Q (a vector passes when Q >= 0).
fn quality(reference: &[i16], decoded: &[i16], channels: usize, rate: u32) -> f64 {
    let downsample = (48000 / rate) as usize;
    let ybands = match rate {
        8000 => 13,
        12000 => 15,
        16000 => 17,
        24000 => 19,
        _ => NBANDS,
    };
    let yfreqs = NFREQS / downsample;
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
    assert_eq!(len, y.len() / channels * downsample, "sample counts differ");
    assert!(len >= WIN);
    let frames = (len - WIN + STEP) / STEP;
    let (mut xs, mut xb) = band_energy(&x, channels, frames, WIN, STEP, 1, NBANDS, true);
    let (mut ys, _) = band_energy(
        &y,
        channels,
        frames,
        WIN / downsample,
        STEP / downsample,
        downsample,
        ybands,
        false,
    );
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
        for b in 0..ybands {
            for j in BANDS[b]..BANDS[b + 1] {
                for c in 0..channels {
                    let m = 0.1 * xb[at(f, b, c)];
                    xs[(f * NFREQS + j) * channels + c] += m;
                    ys[(f * yfreqs + j) * channels + c] += m;
                }
            }
        }
    }
    // Average consecutive frames.
    for j in 0..BANDS[ybands] {
        for c in 0..channels {
            let (mut xt, mut yt) = (xs[j * channels + c], ys[j * channels + c]);
            for f in 1..frames {
                let kx = (f * NFREQS + j) * channels + c;
                let ky = (f * yfreqs + j) * channels + c;
                let (x2, y2) = (xs[kx], ys[ky]);
                xs[kx] += xt;
                ys[ky] += yt;
                xt = x2;
                yt = y2;
            }
        }
    }
    // Below 48 kHz the last 300 Hz are left out (all of the last band at 12 kHz).
    let max_compare = match rate {
        48000 => BANDS[NBANDS],
        12000 => BANDS[ybands],
        _ => BANDS[ybands] - 3,
    };
    let mut err = 0f64;
    for f in 0..frames {
        let mut ef = 0f64;
        for b in 0..ybands {
            let mut eb = 0f64;
            for j in BANDS[b]..BANDS[b + 1].min(max_compare) {
                for c in 0..channels {
                    let re =
                        ys[(f * yfreqs + j) * channels + c] / xs[(f * NFREQS + j) * channels + c];
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

fn list(var: &str, default: &str) -> Vec<String> {
    std::env::var(var)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|s| s.trim().to_owned())
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
    let only = list("KILN_OPUS_ONLY", "01,02,03,04,05,06,07,08,09,10,11,12");
    let rates: Vec<u32> = list("KILN_OPUS_RATES", "48000,24000,16000,12000,8000")
        .iter()
        .map(|r| r.parse().unwrap())
        .collect();
    let mut failures = Vec::new();
    for n in &only {
        let name = format!("testvector{n}");
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).unwrap();
        let reference = read_pcm(&dir.join(format!("{name}.dec")));
        let reference_m = read_pcm(&dir.join(format!("{name}m.dec")));
        for &rate in &rates {
            let (stereo, bad_stereo) = decode(&bits, rate, 2);
            let q_stereo = quality(&reference, &stereo, 2, rate);
            let (mono, bad_mono) = decode(&bits, rate, 1);
            let q_mono =
                quality(&reference, &mono, 1, rate).max(quality(&reference_m, &mono, 1, rate));
            println!(
                "{name} at {rate} Hz: stereo Q {q_stereo:.1}, mono Q {q_mono:.1}, final range differs in {} and {} packets",
                bad_stereo.len(),
                bad_mono.len()
            );
            if !bad_stereo.is_empty() || !bad_mono.is_empty() {
                failures.push(format!(
                    "{name} at {rate} Hz: final range (first at packet {:?})",
                    bad_stereo.first().or(bad_mono.first())
                ));
            }
            if q_stereo < 0.0 || q_mono < 0.0 {
                failures.push(format!(
                    "{name} at {rate} Hz: quality {q_stereo:.1} / {q_mono:.1}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
