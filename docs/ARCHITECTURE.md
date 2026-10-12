# Architecture

How a Switch program becomes a native one, and where each step lives.

```
ExeFS / NRO ──loader──▶ Module ──discover──▶ per-function CFGs ──lift──▶ SSA IR
                                                                       │
          nx_runtime (staticlib) ◀──link── objects ◀──clang── LLVM IR text ◀──codegen
```

| Step | Code | What it does |
|------|------|--------------|
| Input | `src/container`, `nx_runtime/src/{storage,romfs,npdm}.rs` | ExeFS directory or PFS0, RomFS directory or image, NPDM. No keys. |
| Load | `src/loader` | NSO0 / NRO0: LZ4, SHA-256, MOD0 relocations, exports, `.eh_frame_hdr` function extents. Fixed guest base `0x0800_0000`. |
| Decode | `nx_runtime/src/a64/decode.rs` | 32-bit word to `Inst`; strict (unallocated is distinct from unsupported). |
| Interpret | `nx_runtime/src/a64/interp.rs` | The oracle, and the fallback for anything not translated. |
| Discover | `src/lift/discover.rs`, `jumptable.rs` | Recursive descent from roots; functions, blocks, jump tables. |
| Lift | `src/lift/{builder,inst,func}.rs`, IR in `ir.rs` | SSA (Braun et al.), explicit `N Z C V`, one IR instruction per pseudocode operation. |
| Emit | `src/codegen` | LLVM IR text, one unit per batch of functions, plus the support unit and the module descriptor. |
| Build | `src/build` | clang discovery, parallel compile, link against `nx_runtime`. |
| Run | `nx_runtime/src/rt` | Guest memory, exclusive monitor, function table, slow path, process start-up. |

## Calling protocol (generated code)

Every function is `void f(NativeCpu *cpu)` (`nx_runtime/src/rt/abi.rs`). The guest registers live in
SSA values inside a function and in `cpu->state` between functions.

- **Entry**: `cpu->state` holds the guest state at the first instruction.
- **Exit**: `cpu->state` holds the full state and `cpu->state.pc` is where execution continues.
- **Call**: spill, set `pc` to the target, call, then compare `pc` with the expected return address.
  Equal: reload the state and go on. Different: return immediately. A `longjmp`-like transfer
  therefore unwinds native frames until a frame can take it or the run loop dispatches it.
- **Tail call** (`B` to another function, `BR` to a table-resolved or computed target): a
  `musttail` call, so loops built from jumps do not grow the native stack.
- **Indirect target**: `nx_dispatch` looks the address up in the sorted function table and tail
  calls the function, or tail calls the slow path. The slow path interprets from that address,
  logs the address (`NX_UNKNOWN_LOG`), and hands control back to native code at the next known
  entry. The log is a hints file (`recomp build --hints`).
- **Stop**: a memory fault, an instruction that always stops (`BRK`, undefined, unimplemented), or
  an unhandled `SVC` records its details in `cpu->ctl` and sets `pc` to `STOP_PC`, which no call
  site expects, so the whole stack unwinds to the run loop.

## Guest memory

One checked interface (`nx_runtime/src/rt/mem.rs`): a 39-bit address space of 4 KiB pages through a
two-level table of atomic entries. Generated code calls `nx_ld*`/`nx_st*` (and the acquire/release
and exclusive variants); each is followed by a test of the fault flag. A test pins the list of
runtime functions the emitter calls against the runtime's exports.

## Where tests compare what

| Layer | Against | Where |
|-------|---------|-------|
| Decoder | LLVM's assembler (217 encodings) | `nx_runtime/src/a64/decode_tests.rs` |
| Interpreter | Unicorn (Cortex-A57 model), pseudocode wins | `nx_runtime/tests/unicorn_diff.rs` |
| IR evaluator | the interpreter | `src/lift/tests.rs`, `flow_tests.rs` |
| Native code | the interpreter | `src/native_tests.rs` (synthetic), `tests/real_code.rs` (a real title, opt-in) |
