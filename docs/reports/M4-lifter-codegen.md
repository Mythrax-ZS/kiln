# M4: lifter and LLVM code generation (integer ISA)

Status: complete. Date: 2026-10-12.

## What was implemented

The pipeline from a loaded module to a native executable and shared library, for the integer
instruction set the interpreter already runs. See `docs/ARCHITECTURE.md` for the calling protocol.

- **IR and SSA** (`src/lift/ir.rs`, `builder.rs`, `sem.rs`, `eval.rs`): typed integer values, the
  four condition flags as separate `i1` values, memory and system operations in program order,
  calls and exits as terminators that spill the state that differs from the CPU structure. SSA
  is built incrementally (Braun et al. 2013). A verifier checks types, phi/predecessor
  consistency and dominance. The pure operations have one definition, shared by constant folding
  and by the reference evaluator.
- **Lifter** (`inst.rs`, `func.rs`): every instruction the interpreter executes, arm for arm,
  including the order of register writes in CONSTRAINED UNPREDICTABLE encodings. `UDiv`/`SDiv`
  are total in the IR (zero divisor gives zero, `MIN / -1` wraps) and guarded in the lowering.
  Instructions with no translation (all FP and Advanced SIMD until M5) become interpreter
  fallbacks that stop the run with the instruction named; traps become trap exits.
- **Discovery** (`discover.rs`, `jumptable.rs`): recursive descent from the module start, exports,
  init/fini arrays, `.eh_frame` function starts, code pointers in relocations and a hints file.
  A function is whatever is reachable from its entry; a `B` that leaves the function's unwind
  extent (or reaches another entry) is a tail call. `BR` through a jump table is resolved by
  *running* the code in front of the branch for every index the bounds check admits and reading
  the table from the image (A-027); the answer is only a candidate set, so a miss costs speed,
  not correctness.
- **Code generation** (`src/codegen`): LLVM IR text. Divides are guarded; every memory access is a
  call into the runtime followed by a test of the fault flag; barriers become fences (A-025);
  calls check that control returned to the expected address; tail calls and the dispatcher use
  `musttail`. Output is split into translation units, plus a support unit (dispatcher, sorted
  function table) and a module descriptor unit (segments, embedded image via `.incbin`, NPDM,
  exports, `main` or `recomp_main`).
- **Runtime** (`nx_runtime/src/rt`): guest memory behind one checked interface (39-bit address
  space, 4 KiB pages, atomic page entries), the global exclusive monitor with spurious-failure
  injection (A-021), the function table, the interpreter slow path (`nx_runtime/src/slow.rs`) that
  logs every address it is entered at, the run loop, process start-up and stop reporting.
- **Build** (`src/build`): clang discovery, parallel compile, response-file link against the
  `nx_runtime` static library, and the commands `recomp analyze` and `recomp build`
  (`--shared`, `--hints`, `--opt`, `--keep-ir`). Each worker thread lifts, emits and compiles one
  batch, so the IR of a whole program never exists at once. `build` writes a report of everything
  the analysis could not settle.

### Not implemented or not done

- All scalar floating point and Advanced SIMD (milestone M5): they are fallbacks that stop the run.
- Registers are not rebuilt at a memory fault in generated code (A-023); the faulting access is
  reported exactly. The interpreter and the slow path are precise.
- No inline fast path for memory: every access is a call. Nothing here is a statement about speed.
- Cross-module linking (`rtld`, `sdk`, `subsdk0`) and the Horizon process start-up (A-028):
  phase one translates `main` only, and the entry of such a title is in `rtld`, not in `main`.
- LSE atomics, MTE, SVE (decoded as unsupported, as in M3).

## Tests

`cargo test --workspace` (fmt and clippy clean): 125 tests in `nx_runtime`, 14 Unicorn
differential tests, 75 in `recomp`, plus the opt-in real-title tests. Highlights:

| Layer | Test | Result |
|-------|------|--------|
| SSA builder | loop, diamond, folding and sharing | pass; only loop-carried or merged variables get phis, unchanged ones are never spilled |
| IR evaluator vs interpreter | 27 encoding groups, 2,222 random instructions each (60,000), 30,000 plain random words, 30,000 bitfield and logical-immediate field combinations | identical: 31 X registers, SP, PC, NZCV, FPCR/FPSR, exclusive monitor, 80 KiB of memory, and how the run ended |
| | 15 hand-written programs (loops, recursion, tail calls, conditional tail calls, flags across joins, indirect calls with and without hints, four jump-table shapes, `SVC`, an unsupported instruction, a non-local return, an unexplained `BR`) at three load addresses, 24 states each | identical; programs that must use the slow path do, and only those |
| | 1,500 random control-flow programs (forward branches, a counted loop, a call), 4 states each | identical |
| | divide edge cases (zero divisor, `MIN / -1`, both widths) | identical |
| Native vs interpreter (clang, in-process DLL) | the same groups: 1,080 instructions x 4 states by default; run once at 10,800 x 8 (86,400 cases) | identical |
| | the same programs, 150 random programs by default; run once at 4,000 | identical |
| Native, Linux | the whole suite in WSL (Ubuntu, clang 21), including 10,800 x 8 and 2,000 programs, and the end-to-end builds of an ELF executable and shared object | identical |
| End to end | synthetic NSO to executable (exit status 55 = fib(10)); jump table, unwind extent and export to executable and library, library called through `nx_rt_find_export`/`nx_rt_call`; a program that stops reports the `SVC` number and instruction, or the fault address, and exits 101 | pass |
| Runtime | memory (straddling pages, permissions, the top of the address space), the exclusive monitor (own stores, another thread's store, `CLREX`, wrong address and size, spurious failure, pairs), `DC ZVA`, CRC32 reference values, dispatch, slow path, stop records | 26 + 3 + 6 tests |
| Codegen text | fences per barrier kind, ordered accesses use the ordered runtime calls, guarded divides, the list of runtime calls equals the runtime's exports, sorted function table | pass |

### A real title

On the owner's extracted title (aggregate numbers only; nothing from it is stored; opt-in,
`RECOMP_TITLE_DIR`), phase one translates `main`:

- Discovery: 74,820 functions, 1,728,527 guest blocks and 3,123,137 guest instructions, which is
  99.7% of the 3,131,392 instructions in its text segment, in 0.9 s. Lifting and emitting all of it
  takes 1.7 s on 32 threads and produces 12,954,008 IR instructions and 1.12 GB of LLVM IR.
- Indirect branches: 367 jump tables resolved. 4,916 `BR` instructions have no table; the sampled
  ones are virtual tail calls (`ldr x8, [x0]; ldr x1, [x8, #n]; ... br x1`) and tail calls through
  function pointers, which go through the function table, plus a few dozen tables whose bounds
  check is in another block or that are not bounds-checked.
- Clang compiles the whole of it (511 translation units): 6 s at `-O0`, 23 s at `-O1`, and the
  linked executable is 530 MB at `-O0`. Run as it is, it stops at its first instruction because the
  first word of this title's `main` text is zero (its entry is in `rtld`, which phase one does not
  load); the stop is reported, not hidden.
- Interpreter fallbacks: 330,823 instructions in 176,147 ranges, all floating point and Advanced
  SIMD (64 distinct mnemonics, led by loads, stores, `fmul`, `fadd`, `fsub`, `fmov`, `fcmp`). This
  is the M5 work list.
- Differential test on real code (`tests/real_code.rs`): the title is recompiled as a shared library;
  sampled functions run twice from the same random registers and memory (a pointer graph, the
  title's own writable segments, a stack, a thread-local block), in the interpreter and as native
  code, and the end state is compared (how the run ended, the registers, every writable byte).
  6,000 runs (1,500 functions x 4 states): 801 returned, 3,732 faulted, 663 reached non-code, 805
  stopped at an unimplemented instruction, 2 exceeded the interpreter budget and were not compared;
  **0 differences**. A sweep of every discovered function (74,820 functions x 2 states, 149,640 runs, 500 s): 19,852 returned, 93,016 faulted, 17,254 reached non-code, 19,468 stopped at an unimplemented instruction, 50 exceeded the interpreter budget and were not compared; again **0 differences**.

## Defects the tests found

- Purely random instruction words almost never reach narrow groups (`REV`, `ADC`, `CCMP`...);
  injecting bugs showed the first test missed them. The template-per-group sweep replaced it, and
  five injected lifter bugs and three injected code generator bugs are caught (the remaining one,
  dropping an acquire fence, cannot show in a single-threaded comparison and is covered by a text
  test instead).
- Scaling the native test to 10x found a real runtime bug: a page-straddling write near the top of
  the address space overflowed its end address (a panic in debug builds, silent wrap in release).
  Addresses are now checked, and the top of the space has a test.
- The end-to-end library test found `slice::from_raw_parts` called with a null pointer and length
  zero in the runtime's C entry points.
- The real title showed that the jump table resolver missed standard tables whose compared register
  is a copy of, or computed from, the register the table is indexed by (87 of the first 4,983 `BR`s).
  The resolver now solves for the starting value that makes the compared register equal each index;
  367 tables resolve.
- Test-harness faults that looked like product bugs and were not: stale code-region contents between
  runs of one program, and several programs' memory being visible at once in one library.

## Not verified

- Memory ordering in multi-threaded programs: the barrier mapping is checked structurally and the
  monitor with two thread contexts, but there is no concurrent litmus test, and the monitor's
  residual race (A-021) is unmeasured.
- Speed: nothing was measured and nothing is claimed.
- Registers at a fault in generated code (A-023).
- Behavior of anything past the first stop of a real title (needs `rtld`, the kernel layer and
  services).
- Windows and Linux (WSL, Ubuntu, clang 21) were exercised; macOS, other architectures and other
  clang versions were not.
- Unicorn does not take part in M4: the IR and native code are compared with the interpreter, which
  M3 compared with Unicorn.

## Next

M5: floating point and Advanced SIMD in the decoder, interpreter, lifter and code generator, with
the mismatch rules (default NaN, saturating conversions, FPSR flags), differential tests against
Unicorn and against the interpreter, starting from the 64 mnemonics the real title needs.
