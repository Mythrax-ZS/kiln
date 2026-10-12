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
| Floating point and SIMD | `nx_runtime/src/a64/{fp.rs,vec/}` | Exact soft-float (`fp.rs`), the decoder for the scalar FP and Advanced SIMD space and the executor (`vec/`). One implementation, used by the interpreter and by generated code. |
| Interpret | `nx_runtime/src/a64/interp.rs` | The oracle, and the fallback for anything not translated. |
| Discover | `src/lift/discover.rs`, `jumptable.rs` | Recursive descent from roots; functions, blocks, jump tables. |
| Lift | `src/lift/{builder,inst,func}.rs`, IR in `ir.rs` | SSA (Braun et al.), explicit `N Z C V`, one IR instruction per pseudocode operation. |
| Emit | `src/codegen` | LLVM IR text, one unit per batch of functions, plus the support unit and the module descriptor. |
| Build | `src/build` | clang discovery, parallel compile, link against `nx_runtime`. |
| Run | `nx_runtime/src/rt` | Guest memory, exclusive monitor, function table, slow path, process start-up. |
| Horizon | `nx_runtime/src/horizon` | The kernel a produced NSO or NRO runs on: SVCs, threads, synchronization, IPC, `sm:`. |

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

## Floating point and SIMD

The vector registers, `FPCR` and `FPSR` live in the CPU structure, not in SSA values (docs/ASSUMPTIONS.md
A-026). An FP or SIMD instruction becomes one IR operation, `Vec { word, src, nzcv }`, that lowers
to a call of `nx_vec(cpu, word, src, nzcv)`: the runtime decodes the word, runs the same `vec::exec`
the interpreter runs, and returns the integer result or the new flags. Only what crosses the
integer register file travels as an SSA value: the source of `SCVTF`/`DUP`/`FMOV Dn, Xn`, the result
of `FCVTZS`/`UMOV`/`FMOV Xd, Dn`, and the `NZCV` nibble of `FCMP`/`FCCMP`/`FCSEL`. A vector load or
store is `VecMem { word, pc, addr }`, lowered to `nx_vecmem` followed by the usual fault check; the
base register's address computation and writeback are ordinary integer IR. Nothing is inlined, so
the flags and rounding come out of the exact soft-float whatever the host FPU does.

## The Horizon layer

A module descriptor carries a kind (`rt::host::kind`). A plain module (hosts and tests) is started
as a function; an NSO or an NRO goes to `horizon::process::run_module`, which builds one `Kernel`
for the process, maps the main thread's stack and thread-local region, and runs every guest thread
on its own host thread (`Kernel::run_process`). Generated code reaches the kernel only through
`nx_svc(cpu, imm)`: it spills the registers, `svc_entry` finds the thread's `Kernel` through
`cpu.ctl.user` and calls `Kernel::svc`, which takes its arguments from `X0..X7` and leaves its
results there as the Switchbrew SVC pages lay them out, and the generated code reloads the state.

- **State**: one `Mutex` and one `Condvar` guard everything the kernel knows (handles, memory
  blocks, threads, events, waiter lists). A blocking call sleeps in `wait_until`, which re-checks
  its predicate whenever anything changed.
- **Memory**: `memory.rs` is a block map over the guest address space (type, permission, attribute),
  which `QueryMemory` reads and `MapMemory`, `SetMemoryPermission` and `SetHeapSize` change; the data
  itself stays in `rt::mem`. Where the regions sit is in `layout.rs` (docs/ASSUMPTIONS.md A-034).
- **Synchronization**: mutex words with the wait bit, condition variables and the address arbiter
  work on the guest's own memory words (`Runtime::compare_exchange`), so a guest lock and a guest
  atomic see the same bytes.
- **IPC**: `ipc.rs` parses and builds HIPC messages and the CMIF headers, handles domains and the
  control commands; a `Service` (`service.rs`) answers a request with a `Response`, whose objects
  become move handles or domain object ids. `sm:` (`sm.rs`) is the only named port; it hands out a
  service only if the program's NPDM service access list allows the name. `services.rs` registers
  the services; none besides `sm:` exists yet.
- **Stops**: an SVC, a named port, a service or a command nobody implemented does not get an
  answer. The kernel records a report with the guest address and the number, ends the process and
  the produced program exits with status 101.

## Guest memory

One checked interface (`nx_runtime/src/rt/mem.rs`): a 39-bit address space of 4 KiB pages through a
two-level table of atomic entries. Generated code calls `nx_ld*`/`nx_st*` (and the acquire/release
and exclusive variants); each is followed by a test of the fault flag. A test pins the list of
runtime functions the emitter calls against the runtime's exports.

## Where tests compare what

| Layer | Against | Where |
|-------|---------|-------|
| Decoder | LLVM's assembler (217 encodings) | `nx_runtime/src/a64/decode_tests.rs` |
| Soft-float | the host FPU (x86 SSE, by inline assembly) | `nx_runtime/src/a64/fp_tests.rs` |
| Interpreter | Unicorn (Cortex-A57 model), pseudocode wins | `nx_runtime/tests/unicorn_diff.rs` (integer, random words), `unicorn_fp.rs`, `unicorn_simd.rs` |
| FP and SIMD | hand-computed results | `nx_runtime/src/a64/vec/{tests,scalar_tests}.rs` |
| IR evaluator | the interpreter | `src/lift/tests.rs`, `flow_tests.rs` |
| Native code | the interpreter | `src/native_tests.rs` (synthetic), `tests/real_code.rs` (a real title, opt-in) |
| Kernel layer | the Switchbrew contract of each call, with real host threads | `nx_runtime/src/horizon/tests.rs` and the unit tests next to each file |
| Homebrew NRO and NSO | expected output and exit status of built programs | `src/build/e2e_tests.rs` (`mod homebrew`) |
