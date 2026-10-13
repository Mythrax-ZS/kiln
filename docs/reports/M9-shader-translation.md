# M9: shader translation (`nx_shader`)

Status: complete as far as it can be checked without drawing. Date: 2026-10-13.

The task (owner approval of 2026-10-13, "begin the shader translation and Vulkan backend"): turn
the Maxwell machine code a draw uses into SPIR-V, check the result on a real GPU, and do it well
enough that the programs the title binds in its first frames translate. The title is the owner's
own dump; this report gives aggregate numbers only and does not name it.

**Nothing is drawn yet.** The translation is tested in a compute harness on the host GPU (an
NVIDIA RTX 5070 Ti, Vulkan 1.4); the vertex and pixel stage inputs and outputs are checked only
for being valid SPIR-V until the Vulkan backend (M10) runs draws. No picture exists.

## Result

- **All 8 programs the title binds in its first frames translate and pass `spirv-val`**: 4 pixel
  programs (7, 15, 21 and 48 instructions) and 4 vertex programs (34, 121, 140 and 283
  instructions). The 48-instruction pixel program uses the dispatch loop, `KIL` and two constant
  buffers; the 121-instruction vertex program reads global memory (`LDG`) and three constant
  buffers. `cargo run --release -p nx_shader --example translate -- <program.bin>...` prints, for
  each program, its stage, instruction count, SPIR-V size, validation verdict, constant buffer
  extents, textures and whether it kills pixels or touches global memory.
- **36 GPU tests run the translated programs on the device** and compare with Rust arithmetic
  (`cargo test -p nx_vulkan`), plus 23 unit tests of the decoder, control flow graph, header
  parser and SPIR-V builder. A test prints `SKIPPED` and passes when there is no Vulkan device or
  no `spirv-val`; `RECOMP_REQUIRE_EXTERNAL=1` makes either a failure.
- **149 of the 296 rows of the instruction table have a translation** (`example coverage`).
  The other 147 stop the translation with the name of the instruction and its code offset (an
  unknown instruction never becomes a made-up result).

## What was built

| Part | What it does |
|------|--------------|
| `nx_shader::ops`, `inst` | The 296-row opcode table (from envytools' GM107 tables, MIT licence) and typed access to the fields of a 64-bit instruction. |
| `sph` | The 20-word version 3 program header of NVIDIA's `cla097.h` (A-081). |
| `cfg` | The control flow graph of a program, with `SSY`/`PBK`/`PCNT` stacks resolved per path (A-079). |
| `spirv` | A SPIR-V module builder (types, constants, functions, structured control flow, decorations). |
| `translate` | Instruction translation: `alu` (float, integer, comparison, conversion), `int` (`XMAD`, `IMAD`, `LEA`, `LOP3`, `SHF`, `PRMT`), `io` (attributes, constant buffers, special registers, stage prologues and epilogues), `tex` (`TEXS`), `global` (`LDG`/`STG`), `flow` (blocks and the dispatch loop). |
| `validate` | Finds `spirv-val` and `spirv-dis` (`RECOMP_SPIRV_TOOLS`, `PATH`, `VULKAN_SDK`, `C:\VulkanSDK`) and runs them. |
| `nx_vulkan` | The Vulkan device (extension and feature selection, the discrete GPU first), host-visible buffers, images with staging upload, and a one-shot compute dispatch with textures and device-address patching: the harness. |

## What the GPU tests cover

Each test runs a program of the harness mode once per input state (registers and predicates in
and out through buffers) and compares every output with Rust. Counts are states per run.

| Family | Instructions | Compared on |
|--------|--------------|-------------|
| Float arithmetic | `FADD`, `FMUL`, `FFMA` with negation and absolute value | 256 random states |
| Float comparison | `FSETP`, `FSET` (all 16 comparisons, all 3 boolean operations, predicate true and false), `FMNMX` | 144 pairs from a pool with NaN, infinities and equal values, per case |
| Special functions | `MUFU` rcp, rsq, sqrt, lg2, ex2 (relative error 1e-5), sin and cos (1e-3 inside [-3.1, 3.1]) | 209 and 64 values |
| Conversions | `F2I` to 8, 16 and 32 bits (saturation, NaN, ties to even), `I2F` from 8, 16, 32 bits (both signs, absolute value, negation), `F2F` rounding modes, `I2I` saturating narrowing | 230 values, 310 states |
| Integer arithmetic | `IADD`, `ISCADD`, `LOP`, `LOP32I`, `IMUL` (low, high, signed and unsigned), `IMAD` (with negations and the immediate form), `SHL`, `SHR` (wrapped or clamped, signed, bit-reversed), `BFE`, `POPC`, `FLO` (all forms) | 256 to 430 states each |
| Multiply-add family | `XMAD` (three-instruction 32-bit product, halves, types, modes, `.PSL`, `.MRG`, immediate, constant buffer forms), `LEA`, `LEA.HI` with the carry (a 64-bit sum), `LOP3` (all 256 tables, register and immediate forms, the predicate test), `SHF` (left, right, wrapped or clamped, 32 and 64-bit types, immediates), `PRMT` (4 modes, 3 forms) | 40 to 505 states each |
| Comparison and selection | `ISETP`, `ISET` (signed and unsigned, 8 comparisons, 3 operations), `SEL`, `IMNMX`, `PSETP`, `PSET` | 306 states; all 128 predicate combinations |
| Memory | `LDG`/`STG` (8 to 128 bits, through a table of two ranges, one entered at an offset), loads from outside the ranges read a zeroed page and stores land in it, `LDC` (sizes 8 to 64 bits, immediate and register-indexed), `ALD`/`AST` (1, 2 and 4 words, guarded), `S2R` | 1 to 100 invocations; the memory is compared whole after the stores |
| Textures | `TEXS` 2D and 2D array with nearest sampling: component masks and register scatter, clamping outside the texture, the layer taken from the first register | 16 texels and 3 outside points; 12 array texels |
| Control flow | guards, a forward `BRA`, a backward `BRA` loop (sum of 1 to n), `SSY`/`SYNC` reconvergence, `PBK`/`BRK`, a guarded `EXIT` | 40 to 64 states each |

The tests are not toothless: replacing one expected value makes the matching test fail (checked
by hand for the memory and texture tests).

## Bugs the tests found

- `MUFU` read its operand from `Rb`; the operand is `Ra` and bits 20..23 are the function code. A
  test that put the function in those bits and the operand in the same field read register
  number 4 for `RCP` and returned infinity. Every `MUFU.RCP` of the title's pixel programs
  would have been wrong. Fixed (A-083).
- The `LOP`/`LOP3` predicate field with test mode 0 left the predicate alone instead of setting
  it false. Fixed.
- A model of `BFE` in the test (not the translator) disagreed with the GPU for a field that runs
  past bit 31: bits past the end take the sign of the field, as PTX defines it. The translator was
  right.

## What is refused, and what is not checked

- **Not translated** (stops with the instruction and its address): the other texture
  instructions (`TEX`, `TLD`, `TLDS`, `TLD4`, `TXQ`, `TMML`, `TXD`), half-precision arithmetic
  (`HADD2`, `HMUL2`, `HFMA2`, `HSET2`, `HSETP2`), double precision, local and shared memory
  (`LDL`, `STL`, `LDS`, `STS`), generic `LD`/`ST`, atomics and reductions, surfaces,
  `FSWZADD`, `SHFL`, `VOTE`, `BAR`, `PIXLD`, `P2R`/`R2P`, `BFI`, `ICMP`/`FCMP`, `IADD3`, `IMAD32I`
  (the full list: `example coverage -- --list`). Several of these are certain to appear in a
  title that goes beyond its first frames; they are added when a real program needs them, each
  with a GPU test.
- **Flags that are refused**: condition code writes other than the carry, `.X` on `SHF`/`XMAD`,
  `XMAD.CSFU`, `IPA` with centroid or offset sampling, `TEXS` with 16-bit results or depth
  compare, `LEA` with a predicate destination, `PRMT` edge-clamp and replicate-16 modes
  (A-084, A-088, A-089, A-090).
- **Not modelled**: rounding modes and flush-to-zero of float instructions (A-082), the
  hardware's NaN and denormal behavior, `MUFU` precision beyond the host's.
- **Checked only for validity until M10**: attribute interpolation (`IPA`), vertex outputs and the
  depth remap, pixel outputs, `KIL`, implicit level of detail and the dispatch loop in a pixel
  program, where SPIR-V leaves derivatives undefined in divergent control flow (A-091).
- **Every semantic here comes from public descriptions** (envytools, Mesa NAK, the PTX
  instruction set, other emulators' behavior); none was compared with a Switch. That is the
  `UNVERIFIED on hardware` status in the assumptions. The real check is the picture, which M10
  makes possible.

## Files

`nx_shader/` (decoder, graph, translator, builder, examples `sass`, `translate`, `coverage`),
`nx_vulkan/` (device, harness, `tests/translate_*.rs`), `docs/ASSUMPTIONS.md` A-078 to A-091.
