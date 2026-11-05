//! The inverse MDCT of the low-overlap window (RFC 6716 section 4.3.7): an N/4-point complex
//! FFT between two rotations, the computation of the reference's `clt_mdct_backward`
//! (`celt/mdct.c`), followed by the unfolding of the time-domain aliasing with the window.
//! The FFT is a plain mixed-radix one (sizes 60 to 480: radices 2 to 5).

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default)]
struct Cpx {
    re: f32,
    im: f32,
}

impl Cpx {
    fn mul(self, o: Cpx) -> Cpx {
        Cpx {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn add(self, o: Cpx) -> Cpx {
        Cpx {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

/// An unscaled inverse DFT, `out[k] = sum x[n] e^(2 pi i n k / N)`.
struct InverseFft {
    n: usize,
    factors: Vec<usize>,
    /// e^(2 pi i k / N).
    twiddles: Vec<Cpx>,
}

impl InverseFft {
    fn new(n: usize) -> InverseFft {
        let mut factors = Vec::new();
        let mut rest = n;
        for p in [4, 2, 3, 5] {
            while rest.is_multiple_of(p) {
                factors.push(p);
                rest /= p;
            }
        }
        assert_eq!(rest, 1, "an FFT size of radices 2, 3 and 5");
        let twiddles = (0..n)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                Cpx {
                    re: a.cos() as f32,
                    im: a.sin() as f32,
                }
            })
            .collect();
        InverseFft {
            n,
            factors,
            twiddles,
        }
    }

    fn run(&self, input: &[Cpx], out: &mut [Cpx]) {
        self.stage(out, input, 0, 1, &self.factors);
    }

    /// Decimation in time: `out` (length n') receives the DFT of `input[start + k * stride]`.
    fn stage(
        &self,
        out: &mut [Cpx],
        input: &[Cpx],
        start: usize,
        stride: usize,
        factors: &[usize],
    ) {
        let n = out.len();
        let p = factors[0];
        let m = n / p;
        if m == 1 {
            for (j, o) in out.iter_mut().enumerate() {
                *o = input[start + j * stride];
            }
        } else {
            for j in 0..p {
                self.stage(
                    &mut out[j * m..(j + 1) * m],
                    input,
                    start + j * stride,
                    stride * p,
                    &factors[1..],
                );
            }
        }
        // X[k + q m] = sum_j W_n^(j k) W_p^(j q) Y_j[k].
        let step = self.n / n;
        let mut tmp = [Cpx::default(); 5];
        for k in 0..m {
            for (j, t) in tmp.iter_mut().enumerate().take(p) {
                *t = out[j * m + k].mul(self.twiddles[(j * k * step) % self.n]);
            }
            for q in 0..p {
                let mut acc = tmp[0];
                for (j, t) in tmp.iter().enumerate().take(p).skip(1) {
                    acc = acc.add(t.mul(self.twiddles[(j * q * m * step) % self.n]));
                }
                out[q * m + k] = acc;
            }
        }
    }
}

/// The inverse MDCT of one size: `n` output samples before folding, `n / 2` bins.
pub struct Imdct {
    n: usize,
    /// cos(2 pi i / n) for i = 0..=n/4.
    trig: Vec<f32>,
    fft: InverseFft,
}

impl Imdct {
    fn new(n: usize) -> Imdct {
        let trig = (0..=n / 4)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos() as f32)
            .collect();
        Imdct {
            n,
            trig,
            fft: InverseFft::new(n / 4),
        }
    }

    /// The inverse MDCT for the bins of 120 << `shift_lm` (0 to 3).
    pub fn for_bins(bins_log: usize) -> &'static Imdct {
        static SIZES: OnceLock<[Imdct; 4]> = OnceLock::new();
        let sizes = SIZES.get_or_init(|| {
            [
                Imdct::new(240),
                Imdct::new(480),
                Imdct::new(960),
                Imdct::new(1920),
            ]
        });
        &sizes[bins_log]
    }

    /// Transform the bins `input[k * stride]` and add the windowed result into `out`: the first
    /// `window.len()` samples are added (they overlap the previous block's tail), the next
    /// `n / 2` are written. `out` holds `n / 2 + window.len()` samples.
    pub fn backward(&self, input: &[f32], stride: usize, out: &mut [f32], window: &[f32]) {
        let n2 = self.n / 2;
        let n4 = self.n / 4;
        let t = &self.trig;
        // The small rotation by pi / (4 n), with sin(x) = x and cos(x) = 1 as the reference has.
        let sine = 2.0 * std::f32::consts::PI * 0.125 / self.n as f32;
        let mut f2 = vec![Cpx::default(); n4];
        for (i, v) in f2.iter_mut().enumerate() {
            let x1 = input[2 * i * stride];
            let x2 = input[(n2 - 1 - 2 * i) * stride];
            let yr = -x2 * t[i] + x1 * t[n4 - i];
            let yi = -x2 * t[n4 - i] - x1 * t[i];
            *v = Cpx {
                re: yr - yi * sine,
                im: yi + yr * sine,
            };
        }
        let mut f = vec![Cpx::default(); n4];
        self.fft.run(&f2, &mut f);
        for (i, v) in f.iter_mut().enumerate() {
            let (re, im) = (v.re, v.im);
            let yr = re * t[i] - im * t[n4 - i];
            let yi = im * t[i] + re * t[n4 - i];
            *v = Cpx {
                re: yr - yi * sine,
                im: yi + yr * sine,
            };
        }
        // The folded signal: the real parts forward, the imaginary parts backward.
        let mut y = vec![0f32; n2];
        for i in 0..n4 {
            y[2 * i] = -f[i].re;
            y[2 * i + 1] = f[n4 - 1 - i].im;
        }
        // Unfold: odd symmetry at the start, even at the end, windowed where blocks overlap.
        let overlap = window.len();
        let h = overlap / 2;
        for j in 0..h {
            out[h + j] += window[h + j] * y[j];
            out[h - 1 - j] -= window[h - 1 - j] * y[j];
        }
        out[2 * h..n2].copy_from_slice(&y[h..n2 - h]);
        for r in 0..h {
            let v = y[n2 - h + r];
            out[n2 + r] = window[overlap - 1 - r] * v;
            out[n2 + overlap - 1 - r] = window[r] * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fft_matches_a_direct_dft() {
        for n in [60usize, 120, 240, 480] {
            let fft = InverseFft::new(n);
            let input: Vec<Cpx> = (0..n)
                .map(|i| Cpx {
                    re: ((i * 37 % 11) as f32) - 5.0,
                    im: ((i * 13 % 7) as f32) - 3.0,
                })
                .collect();
            let mut out = vec![Cpx::default(); n];
            fft.run(&input, &mut out);
            for (k, o) in out.iter().enumerate().step_by(7) {
                let (mut re, mut im) = (0f64, 0f64);
                for (j, x) in input.iter().enumerate() {
                    let a = 2.0 * std::f64::consts::PI * (j * k % n) as f64 / n as f64;
                    re += x.re as f64 * a.cos() - x.im as f64 * a.sin();
                    im += x.re as f64 * a.sin() + x.im as f64 * a.cos();
                }
                assert!(
                    (o.re as f64 - re).abs() < 1e-2 && (o.im as f64 - im).abs() < 1e-2,
                    "n {n} k {k}"
                );
            }
        }
    }

    /// The forward MDCT of the RFC's definition, for the test only.
    fn forward(x: &[f32], n: usize) -> Vec<f32> {
        let n2 = n / 2;
        (0..n2)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        let a = 2.0 * std::f64::consts::PI / n as f64
                            * (i as f64 + 0.5 + n as f64 / 4.0)
                            * (k as f64 + 0.5);
                        x[i] as f64 * a.cos()
                    })
                    .sum::<f64>() as f32
            })
            .collect()
    }

    #[test]
    fn the_inverse_is_an_mdct_up_to_scale() {
        // A full-overlap check of the transform kernel: with a window of the whole half block,
        // IMDCT(MDCT(x)) of two overlapping blocks adds back to x times a constant.
        let n = 240usize;
        let n2 = n / 2;
        let window: Vec<f32> = (0..n2)
            .map(|i| ((std::f64::consts::PI / 2.0) * ((i as f64 + 0.5) / n2 as f64)).sin() as f32)
            .collect();
        let signal: Vec<f32> = (0..3 * n2)
            .map(|i| ((i as f32) * 0.37).sin() + 0.5 * ((i as f32) * 0.11).cos())
            .collect();
        let imdct = Imdct::new(n);
        let mut out = vec![0f32; 3 * n2 + n2];
        for b in 0..2 {
            // Window the block (rising, falling) as an encoder would.
            let block: Vec<f32> = (0..n)
                .map(|i| {
                    let w = if i < n2 { window[i] } else { window[n - 1 - i] };
                    signal[b * n2 + i] * w
                })
                .collect();
            let bins = forward(&block, n);
            imdct.backward(&bins, 1, &mut out[b * n2..], &window);
        }
        // The middle half block is complete: it is the signal scaled by the same factor.
        let scale = out[n2 + 10] / signal[n2 + 10];
        for i in n2..2 * n2 {
            assert!(
                (out[i] - scale * signal[i]).abs() < 1e-2 * scale.abs(),
                "{i}: {} vs {}",
                out[i],
                scale * signal[i]
            );
        }
    }
}
