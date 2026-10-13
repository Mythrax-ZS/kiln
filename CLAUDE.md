# recomp: working rules

Static recompiler that turns Horizon OS (Nintendo Switch) AArch64 programs into native x86_64
executables and shared libraries. LLVM IR is emitted as text and compiled with clang.

Crates:
- `recomp` (root): library + CLI. Input layer, loader, analysis, lifter, codegen.
- `nx_runtime` (`nx_runtime/`): everything shared between compile time and run time: storage, RomFS,
  NPDM, the A64 decoder and interpreter, guest memory and the Horizon emulation layer. Built as
  `rlib` and `staticlib`; every produced binary links against it.
- `nx_gpu` (`nx_gpu/`): the GPU front, with no knowledge of Horizon: the driver service's device nodes,
  the display buffer queue and a Maxwell command processor with a macro engine. `nx_runtime`
  depends on it. It draws nothing itself: draws, clears and displayed buffers go to a `Backend`.
- `nx_shader` (`nx_shader/`): Maxwell shader programs: header, decoder, control flow, a SPIR-V builder
  and the translation to SPIR-V. No dependencies and no knowledge of the GPU model that feeds it.
- `nx_vulkan` (`nx_vulkan/`): the Vulkan device, the compute harness the translator is tested
  with, and the renderer (a `Backend`) with the ASTC decoder (`ash`, loaded at run time).
- `recomp_gui` (`recomp_gui/`): the desktop front end (`egui`); it runs `recomp` and the games as
  processes. Run `cargo run -p recomp_gui`; `RECOMP_GUI_SCREENSHOT=<png>` saves its window.

## Inputs (owner instruction, 2026-10-12)

The tool takes the **already extracted ExeFS and RomFS** of a game (a directory or an image for
each), or a homebrew NRO. It does **not use `prod.keys` at all** and has no NSP, NCA, ticket or
crypto code. Do not add any. The original brief's NCA/IVFC/key milestone M1 was replaced by an
input layer (ExeFS directory or PFS0, RomFS image parser, RomFS builder for extracted directories).

## Gate before every commit

```
cargo fmt --all --check
cargo lint            # alias for: cargo clippy --all-targets -- -D warnings
cargo test
```

## Commits

Small chunks. Title at most 50 characters. No body. No attribution lines of any kind.

## Hard rules

- No firmware, keys or title data in the repository or in any committed output. Fixtures are
  synthetic or generated inside the test itself.
- Do not describe the output as "1:1 accurate". Describe measured test coverage instead.
- Every undocumented or assumed behavior goes in `docs/ASSUMPTIONS.md` with its source.
- An unknown SVC or service command stops the run and reports its address and number. Never
  return a made-up success.
- Where Unicorn disagrees with the Arm pseudocode, the pseudocode wins and the disagreement is
  recorded in `docs/UNICORN_DISAGREEMENTS.md`.
- Milestones M1..M10 (see `docs/reports/`): stop after each one, run the tests, write the report,
  then move on. M8 passed (2026-10-12): the owner approved the GPU front, the `nx_gpu` crate
  (`nvdrv` and its device nodes, the display and buffer-queue path, a tracing command processor
  with a macro engine). The owner approved shader translation (Maxwell SASS to SPIR-V, crate
  `nx_shader`, M9) and the Vulkan backend (`ash`, crate `nx_vulkan`, M10) on 2026-10-13; M9
  passed the same day, and M10 draws the title's first screens (frames dumped and looked at) and is
  open at the window. Never describe the output as playable before frames are measured; saying
  what a frame shows needs the frame itself (a dumped image looked at), not a count.
- Out of scope: online services, account checks, licensing checks, AArch32 (detect and error).

## External tools used by tests

Tests that need an external tool print a loud `SKIPPED` line and pass when the tool is missing.
Set `RECOMP_REQUIRE_EXTERNAL=1` to turn those skips into failures.

- clang with the x86-64 target: `RECOMP_CLANG`, then `PATH`, then the LLVM bundled with Visual Studio.
  Produced programs link `nx_runtime.lib`/`libnx_runtime.a`: `RECOMP_RUNTIME_LIB`, else it is built
  with `cargo build -p nx_runtime` (the native tests do this themselves).
- Native differential test sizes: `RECOMP_NATIVE_WORDS` (instructions per encoding group, default
  40), `RECOMP_NATIVE_STATES` (random states per instruction, 4), `RECOMP_NATIVE_PROGRAMS` (random
  control-flow programs, 150). The IR-level tests scale with `RECOMP_LIFT_CASES` and
  `RECOMP_LIFT_PROGRAMS`.
- Real title (opt-in, `RECOMP_TITLE_DIR` = a folder holding `exefs/`): `tests/real_title.rs` checks
  the loader; `tests/real_code.rs` recompiles `main` and compares native with interpreted runs of
  sampled functions (`RECOMP_REAL_FUNCTIONS`, `RECOMP_REAL_STATES`, `RECOMP_REAL_ALL=1` for every
  function). Only aggregate numbers are printed; nothing from the title is stored.
- Unicorn differential test sizes: `RECOMP_DIFF_CASES` replaces the default number of cases of every
  group in `unicorn_diff`, `unicorn_fp` and `unicorn_simd` (the random-words test draws that many
  words).
- Kernel layer: `nx_runtime/src/horizon/tests.rs` drives the SVC interface with real host threads
  and needs no external tool. The homebrew tests in `src/build/e2e_tests.rs` build small NROs
  from assembled words and run them (clang needed). `NX_TRACE_SVC=1` logs every SVC and IPC
  request of a produced program; `NX_TRACE_LOG=1` prints what it logs through `lm`;
  `NX_RANDOM_SEED=<n>` makes the kernel's entropy reproducible; `NX_ROMFS=<dir|image>` names the
  RomFS it reads at run time.
- GPU front: `nx_gpu`'s unit tests need no external tool. `NX_TRACE_GPU=1` prints every submission
  of a produced program and, when it ends, what the command processor counted (`NX_GPU_TOP=<n>` is
  how many of the busiest methods it lists, 0 for all); `NX_TRACE_GPU=2` prints every method write
  and macro call; `NX_MAX_FRAMES=<n>` ends the run after `n` presented frames; `NX_SAVE_DIR=<dir>`
  is where save data goes (default `./nx_save`). `recomp symbols --exefs <dir> [--match <text>]
  [<address>...]` names the function around an address, for reading the return addresses a stop
  report ends with.
- Renderer: `nx_vulkan`'s tests need a Vulkan device (`NX_VULKAN_DEVICE` picks one; they print
  `SKIPPED` without one) and `spirv-val` for the translator's. `NX_RENDERER=vulkan` makes a
  produced program draw; `NX_DUMP_FRAMES=<dir>` writes each displayed frame as a PNG (opaque, cropped
  to what is shown; `NX_DUMP_EVERY=<n>`, `NX_DUMP_COUNT=<n>`, `NX_DUMP_TARGETS=1` for every render
  target too: a PNG is uncompressed, so dump every n-th frame); `NX_TRACE_DRAWS=<frame>` lists the
  draws and clears of one frame (with `NX_TRACE_STATE=1` their state and vertices, `NX_DUMP_DRAWS=<dir>`
  and `NX_DUMP_FROM=<n>` a picture after each); `NX_TRACE_TEXTURES=1`, `NX_TRACE_CBUFS=1`,
  `NX_DEBUG_NO_CULL=1`, `NX_DEBUG_FULL_VIEWPORT=1`, `NX_DEBUG_SOLID_PIXEL=1` and
  `NX_SPIRV_CHECK=1` look at one thing each; `NX_VULKAN_VALIDATE=1` turns the validation layer on;
  `NX_SHADER_DUMP=<dir>` writes the programs draws use (read them with `cargo run --release -p
  nx_shader --example sass -- <file>`). Never `NX_DUMP_FRAMES` without `NX_DUMP_EVERY` for a long run.
- ASTC decoder: `nx_vulkan/tests/astc_reference.rs` compares it with ARM's `astcenc`
  (`RECOMP_ASTCENC=<path>`, or `astcenc` on the `PATH`; the tool may be unpacked from a distribution
  package without installing it); `RECOMP_ASTC_BATCHES` (default 4, 4,096 blocks each) scales it and
  `RECOMP_ASTC_ONLY=<n>` runs one footprint.
- A real title through the whole pipeline (opt-in, by hand; nothing from it is stored): build with
  `recomp build --exefs <exefs> --romfs <romfs> --out <dir>`, run `<dir>/<name>` with
  `NX_TRACE_SVC=1`, and use `recomp relink` / `--reuse-objects` to iterate on the runtime.
- Unicorn (AArch64 oracle): `UNICORN_DLL`, else `.venv` (`py -m venv .venv`, then
  `.venv\Scripts\pip install unicorn`). The shared library is loaded with `libloading`.
