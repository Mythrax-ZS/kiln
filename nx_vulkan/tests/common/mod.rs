//! What the translator tests share: a Vulkan device (or a loud skip), a small random generator and a
//! runner that executes a program of the harness mode once per input state.

#![allow(dead_code)]

use nx_shader::asm::assemble;
use nx_shader::cfg::Code;
use nx_shader::translate::{test_bindings as tb, translate, Config, Stage};
use nx_vulkan::{AddressPatch, SampledImage, Vk};

pub fn vulkan() -> Option<Vk> {
    match Vk::new() {
        Ok(v) => Some(v),
        Err(e) => {
            if std::env::var("KILN_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
                panic!("no Vulkan device and KILN_REQUIRE_EXTERNAL=1: {e}");
            }
            eprintln!("SKIPPED: no Vulkan device: {e}");
            None
        }
    }
}

/// A small deterministic generator for test inputs.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// The registers (and predicates) of one invocation.
#[derive(Clone, Copy, Default)]
pub struct State {
    pub regs: [u32; 15],
    pub preds: u32,
}

/// Things a test binds besides the registers and constant buffer 0. Their buffers are numbered
/// from 5 for [`AddressPatch`] (0..5 are the registers in and out, the attributes in and out and
/// constant buffer 0).
#[derive(Default)]
pub struct Extras<'a> {
    /// `(binding, contents)`; the contents after the run are returned in [`Outcome`].
    pub buffers: Vec<(u32, Vec<u8>)>,
    pub images: Vec<SampledImage<'a>>,
    pub patches: Vec<AddressPatch>,
    /// The attribute words each invocation reads with `ALD`/`IPA` (zeros where absent).
    pub attrs_in: Vec<[u32; 64]>,
    /// Translate with the loop around a `switch` even for a program whose jumps all go forward.
    pub force_loop: bool,
    /// The words of constant buffers an indirect jump is resolved with.
    pub constants: Vec<nx_shader::cfg::TableWord>,
}

pub struct Outcome {
    pub states: Vec<State>,
    pub buffers: Vec<Vec<u8>>,
    /// The attribute words each invocation wrote with `AST`.
    pub attrs_out: Vec<[u32; 64]>,
}

/// Run `program` once per element of `inputs`. `cbuf0` is bound as constant buffer 0.
pub fn run_with(
    vk: &Vk,
    program: &[u64],
    inputs: &[State],
    cbuf0: &[u32],
    mut extras: Extras<'_>,
) -> Outcome {
    let bytes = assemble(program);
    let mut config = Config::new(Stage::Compute);
    config.force_loop = extras.force_loop;
    config.constants = extras.constants.clone();
    let t = translate(Code(&bytes), None, &config)
        .unwrap_or_else(|e| panic!("translation failed: {e}"));
    nx_shader::validate::assert_valid(&t.spirv);
    let n = inputs.len();
    let mut regs_in = Vec::with_capacity(n * 16 * 4);
    for s in inputs {
        for r in s.regs {
            regs_in.extend_from_slice(&r.to_le_bytes());
        }
        regs_in.extend_from_slice(&s.preds.to_le_bytes());
    }
    let mut regs_out = vec![0u8; n * 16 * 4];
    let mut attr_in = vec![0u8; n * tb::ATTRS as usize * 4];
    for (i, words) in extras.attrs_in.iter().enumerate() {
        for (j, w) in words.iter().enumerate() {
            let at = (i * tb::ATTRS as usize + j) * 4;
            attr_in[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    let mut attr_out = vec![0u8; n * tb::ATTRS as usize * 4];
    let mut c0: Vec<u8> = cbuf0.iter().flat_map(|w| w.to_le_bytes()).collect();
    c0.resize(c0.len().max(4096), 0);
    let cbuf_binding = 64 * Stage::Compute.index();
    let mut list: Vec<(u32, &mut [u8])> = vec![
        (tb::REGS_IN, &mut regs_in[..]),
        (tb::REGS_OUT, &mut regs_out[..]),
        (tb::ATTR_IN, &mut attr_in[..]),
        (tb::ATTR_OUT, &mut attr_out[..]),
        (cbuf_binding, &mut c0[..]),
    ];
    for (binding, data) in extras.buffers.iter_mut() {
        list.push((*binding, &mut data[..]));
    }
    vk.run_compute_with(
        &t.spirv,
        &mut list,
        &extras.images,
        &extras.patches,
        [n as u32, 1, 1],
    )
    .unwrap();
    let states = (0..n)
        .map(|i| {
            let word = |j: usize| {
                u32::from_le_bytes(
                    regs_out[(i * 16 + j) * 4..(i * 16 + j) * 4 + 4]
                        .try_into()
                        .unwrap(),
                )
            };
            let mut s = State::default();
            for (j, r) in s.regs.iter_mut().enumerate() {
                *r = word(j);
            }
            s.preds = word(15);
            s
        })
        .collect();
    let attrs_out = (0..n)
        .map(|i| {
            let mut words = [0u32; 64];
            for (j, w) in words.iter_mut().enumerate() {
                let at = (i * tb::ATTRS as usize + j) * 4;
                *w = u32::from_le_bytes(attr_out[at..at + 4].try_into().unwrap());
            }
            words
        })
        .collect();
    Outcome {
        states,
        buffers: extras.buffers.into_iter().map(|(_, d)| d).collect(),
        attrs_out,
    }
}

/// Run `program` once per element of `inputs` and return the final states.
pub fn run(vk: &Vk, program: &[u64], inputs: &[State], cbuf0: &[u32]) -> Vec<State> {
    run_with(vk, program, inputs, cbuf0, Extras::default()).states
}

pub fn random_states(n: usize, seed: u64) -> Vec<State> {
    let mut rng = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let mut s = State::default();
            for r in s.regs.iter_mut() {
                *r = rng.next();
            }
            s.preds = rng.next() & 0x7F;
            s
        })
        .collect()
}
