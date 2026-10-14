# M5: floating point and Advanced SIMD

Status: complete. Date: 2026-10-12.

## What was implemented

Scalar floating point, the whole Advanced SIMD instruction set of an Armv8.0 core (vector and
scalar forms, loads and stores), and the polynomial multiply `PMULL` in the decoder, the
interpreter, the lifter and the code generator. One implementation of the semantics serves the
interpreter and the generated code (`docs/ARCHITECTURE.md`, "Floating point and SIMD").

- **Soft-float** (`nx_runtime/src/a64/fp.rs`): exact arithmetic on a 128-bit significand frame with a
  sticky bit and one rounding per result (`FPRound`), independent of the host FPU. Add, subtract,
  multiply, `FMULX`, divide, square root, fused multiply-add (and its halved form for `FRSQRTS`),
  compares (quiet and signalling), `FMAX`/`FMIN`/`FMAXNM`/`FMINNM`, conversions between half,
  single and double (including the alternative half format and round-to-odd), integer and fixed
  point to and from floating point (every rounding mode, saturating, NaN gives zero), `FRINT*`,
  the reciprocal and reciprocal-square-root estimates and steps, `FRECPX`, `URECPE`, `URSQRTE`.
  `FPCR` rounding mode, flush-to-zero, default NaN and `AHP` are honoured; `FPSR` collects
  `IOC DZC OFC UFC IXC IDC` and `QC`. NaN handling follows the pseudocode (signalling NaNs are
  quieted and signal; with `FPCR.DN` the result is the default NaN `0x7FC00000`).
- **Scalar floating point** (`vec/decode.rs`, `vec/exec.rs`): data processing with one, two and
  three sources, compares, conditional compare and select, immediates, conversions to and from
  integers and fixed point, every `FMOV` form, and loads and stores of `B`/`H`/`S`/`D`/`Q`
  registers (offset, pre/post index, register offset, literal, pair).
- **Advanced SIMD** (`vec/simd.rs`, `decode_simd.rs`, `decode_simd_imm.rs`, `simd_exec.rs`): three
  registers of the same size (integer, bitwise, floating point, pairwise), three registers of
  different sizes (long, wide, narrow, `PMULL`), two-register miscellaneous (including every
  `FRINT*`, `FCVT*`, estimate and comparison with zero), across lanes, permutes, `EXT`, `TBL`/`TBX`,
  element copies (`DUP`, `INS`, `SMOV`, `UMOV`), modified immediates, shifts by immediate (plain,
  rounding, saturating, narrowing, widening, fixed point conversions), operations by element, all
  the scalar forms of those groups, and the structure loads and stores `LD1`-`LD4`, `ST1`-`ST4`,
  `LD1R`-`LD4R`, one lane or whole registers, post-indexed by an immediate or a register.
- **Lifter and code generator**: two IR operations, `Vec` and `VecMem`, lowered to runtime calls
  `nx_vec` and `nx_vecmem`. The vector registers, `FPCR` and `FPSR` stay in the CPU structure
  (A-026, a deliberate departure from keeping `FPCR`/`FPSR` as SSA values); only the integer operand
  and result of an instruction and the `FCMP` flags travel as values. Vector memory accesses go
  through the checked guest memory like every other access, with the same fault check.
- **Decoder strictness**: encodings the architecture reserves or leaves unallocated, and the
  encodings only later architecture versions allocate (half-precision arithmetic, dot product,
  rounding multiply-accumulate, complex numbers, `FJCVTZS`, `FRINT32/64`...) are *unallocated*,
  which is what the Cortex-A57 does (A-030, A-033). The cryptographic extension is valid on that
  core and is reported as unsupported.

### Not implemented or not done

- The cryptographic extension (`AESE`/`AESD`/`AESMC`/`AESIMC`, `SHA1*`, `SHA256*`): decoded as
  unsupported, so a program that uses them stops with the instruction named. The title used for the
  measurements contains none.
- No inline lowering: every floating-point and SIMD instruction in generated code is a call into
  the runtime that decodes the word again. This keeps flags and rounding exact; it is not fast and
  nothing about speed was measured or is claimed.
- Floating-point exceptions never trap (A-031), as on the target core.
- Half-precision arithmetic (`FEAT_FP16`), which the target core lacks.

## Tests

`cargo test --workspace` (fmt and clippy clean): 183 `nx_runtime` unit tests, 29 Unicorn
differential tests (14 integer, 7 floating point, 8 SIMD), 75 `kiln` tests, plus the opt-in
real-title tests.

| Layer | Test | Result |
|-------|------|--------|
| Soft-float | 11 tests: finite results and exception flags of add, subtract, multiply, divide, square root, fused multiply-add, precision conversions, integer conversions and rounding to integral against the x86 SSE unit (inline assembly) in every rounding mode, with one stated exception (x86 detects tininess after rounding, Arm before); NaN rules, flush-to-zero, saturating conversions, half precision, compares, minimum/maximum and the estimates with directed vectors | identical |
| Hand-computed | 46 tests with words assembled by LLVM and results worked out from the pseudocode: every SIMD group, the scalar forms, structure loads and stores, `FPCR` modes, saturation and `QC`, the decoder's refusals | pass |
| Interpreter vs Unicorn, by encoding group | scalar floating point (9 groups), vector loads and stores (5), Advanced SIMD (20 groups: every vector and scalar group, structure loads and stores), words drawn from the group's fixed bits whether or not the decoder accepts them, random registers built from edge values (zeros, denormals, infinities, NaNs of both kinds, ties, integer lane limits) and random `FPCR`/`FPSR`; V0-V31, `FPCR`, `FPSR`, memory and how the run ended are compared | 1.75 million cases by default and **10.2 million (34 groups x 300,000) in one run: 0 differences**; unallocated encodings agree too |
| Interpreter vs Unicorn, random words | 8,000,000 random words over the whole instruction space, with random vector state, everything compared | 2,389,070 executed on both and equal, 4,727,623 unallocated on both; 854,148 in groups the project does not implement (433 of them cryptographic; the rest are extensions the A57 lacks, such as SVE, MTE, pointer authentication and atomics, and instructions EL0 cannot use); the remaining differences are understood and recorded (U-5, U-9, U-10, U-11 in `docs/UNICORN_DISAGREEMENTS.md`) |
| IR evaluator vs interpreter | 61 encoding groups (34 new), 983 random instructions each, random vector state | identical |
| Native (clang) vs interpreter | the same groups: 500 instructions x 8 states each (30,500 programs, 244,000 runs) on Windows and again on Linux (WSL, Ubuntu, clang 21) | identical |
| Native vs interpreter on a real title | `main` recompiled as a shared library (74,820 functions, 3,123,640 instructions, 21 s at `-O1`); 3,000 sampled functions x 3 states, then every function x 2 states (149,640 runs, 552 s): 22,214 returned, 107,218 faulted, 19,480 reached non-code, 661 stopped at an instruction that stops the run, 67 exceeded the interpreter budget and were not compared | **0 differences** |
| Mutation checks | 4 bugs injected into the scalar floating-point executor and 6 into the SIMD executor (wrong operand order, inverted condition, wrong rounding constant, wrong lane order, wrong register stride, `TBX` keeping the wrong byte...) | all 10 caught by the differential tests |

### A real title

On the owner's extracted title (aggregate numbers only; nothing from it is stored): the 330,823
instructions that were interpreter fallbacks after M4 (64 distinct mnemonics, all floating point and
Advanced SIMD) are now lifted, and **the analysis reports 0 interpreter fallbacks** among its
3,123,640 reachable instructions. The 8 instructions that stop the run are the title's own markers
(`0xE7FFDEFE` seven times, and the zero word at the module start).

## Defects the tests found

- `FRSQRTS` rounded twice (the fused multiply-subtract, then the halving), which overflows when the
  halved value is representable. It now rounds once, after halving (found by the first SIMD run).
- `RBIT` named a 16-bit lane in the `size` field and was decoded with 16-bit lanes; it always
  reverses the bits of bytes (found by the first SIMD run).
- A structure load or store with bit 31 set was accepted (found by 3 million random words).
- A real runtime bug from M4: a load that straddles two pages did not check the read permission
  (the byte copy was called with permission checking off). Found when the native group test was
  scaled from 40 to 500 instructions per group, because a register-offset load computed an address
  next to another program's inaccessible region. Fixed, with a unit test (`rt::mem`).
- Three Unicorn behaviors, none about floating-point values, were recorded rather than copied:
  `FCMP`/`FCMPE` with `#0.0` and a non-zero `Rm` executes (U-9), a misaligned ordered or exclusive
  access to an unmapped address reports the translation fault instead of the alignment fault
  (U-10), and Unicorn runs the guest at EL1 so it executes system registers and `SYS` operations
  the EL0 model refuses (U-11). No value computed by Unicorn's softfloat disagreed with the
  pseudocode.

## Not verified

- Nothing was run on a Switch. The oracles are the pseudocode, the host FPU (for IEEE-exact
  operations), and Unicorn's QEMU 5 softfloat (for everything). Choices the architecture leaves to
  the implementation follow Unicorn's Cortex-A57 model (A-013, A-030).
- The cryptographic extension.
- Speed, and the cost of one runtime call per floating-point or SIMD instruction.
- Atomicity of 16-byte vector accesses (two 64-bit halves; the architecture promises single-copy
  atomicity per element only) and the register state after a vector store that faults part-way
  (A-032).
- Behavior that needs the Horizon kernel: threads each own their vector state by construction, but
  no multi-threaded floating-point program was run.

## Next

M6: the Horizon kernel layer in `nx_runtime` (SVC interface per Switchbrew, memory, threads,
synchronization, IPC, the `sm:` service gated by the NPDM access list), the homebrew start-up
convention, and a synthetic NRO that runs natively and prints output.
