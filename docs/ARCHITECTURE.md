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
| Load | `src/loader`, `src/program.rs` | NSO0 / NRO0: LZ4, SHA-256, MOD0 relocations, exports, `.eh_frame_hdr` function extents. Fixed guest base `0x0800_0000`. All NSOs of an ExeFS are mapped back to back (`rtld`, `main`, `subsdk0-9`, `sdk`) and the program starts in the first; the embedded image is the unrelocated one, as the kernel's loader leaves it (A-041). |
| Decode | `nx_runtime/src/a64/decode.rs` | 32-bit word to `Inst`; strict (unallocated is distinct from unsupported). |
| Floating point and SIMD | `nx_runtime/src/a64/{fp.rs,vec/}` | Exact soft-float (`fp.rs`), the decoder for the scalar FP and Advanced SIMD space and the executor (`vec/`). One implementation, used by the interpreter and by generated code. |
| Interpret | `nx_runtime/src/a64/interp.rs` | The oracle, and the fallback for anything not translated. |
| Discover | `src/lift/discover.rs`, `jumptable.rs` | Recursive descent from roots; functions, blocks, jump tables. |
| Lift | `src/lift/{builder,inst,func}.rs`, IR in `ir.rs` | SSA (Braun et al.), explicit `N Z C V`, one IR instruction per pseudocode operation. |
| Emit | `src/codegen` | LLVM IR text, one unit per batch of functions, plus the support unit and the module descriptor. |
| Build | `src/build` | clang discovery, parallel compile, link against `nx_runtime`. |
| Run | `nx_runtime/src/rt` | Guest memory, exclusive monitor, function table, slow path, process start-up. |
| Horizon | `nx_runtime/src/horizon` | The kernel a produced NSO or NRO runs on: SVCs, threads, synchronization, IPC, `sm:`, and the services. |
| GPU | `nx_gpu` | The driver service and its device nodes, the display and buffer queue, and a command processor with a macro engine (nothing is drawn). It decodes the 3D state (`state.rs`), keeps the constant buffer bindings, and counts the shader programs draws use (`shaders.rs`). |
| Shaders | `nx_shader` | The Maxwell program header, the instruction decoder, the control-flow graph (with the `SSY`/`PBK` stacks), a SPIR-V module builder, and the translation of a program to SPIR-V (`translate/`): registers and predicates as function variables, a dispatch loop over the blocks. |
| Vulkan | `nx_vulkan` | The Vulkan device (loaded at run time) and a compute harness; the translator's tests run its shaders here against plain Rust arithmetic. |
| Front end | `recomp_gui` | The desktop window around the command line: pick a game, recompile, keep a library of games and the settings each runs with. It runs `recomp` and the games as processes. |

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
  service only if the program's NPDM service access list allows the name. `services/` holds the
  services (each a file, registered in `services/mod.rs`): `lm`, `apm`, `appletOE`, `aoc:u`,
  `pctl`, `fsp-srv` (with save data on the host), `nvdrv`, `vi:u`, `acc:u0`, `set`, `hid`, `time:u`,
  `nifm:u`, `bsd:u`, `ssl`, `prepo:u`, `nfp:user`, `friend:u`, `pl:u` and `audren:u`; a service is
  handed a weak reference to its kernel for events, transfer memory, shared memory and guest
  buffers, and may reply through the caller's receive list (pointer buffers). The services that
  need a host thread (the vsync event of `vi:u`, the input samples of `hid`, the frame event of
  `audren:u`) start it on first use and stop it when the process ends.
- **Shared memory**: a service creates a shared memory object (`shmem.rs`) in a region of the
  address space the program cannot see, writes its pages there, and gives the program a handle;
  `MapSharedMemory` puts the same pages into the program (`hid` publishes input this way, `pl:u`
  the fonts).
- **Lent memory**: `CreateTransferMemory` reprotects a range of the program's heap and marks it
  locked until the handle is closed (`memobj.rs`); the services, which run in the process, read
  and write it through the same addresses.
- **RomFS**: `fsp-srv` serves the directory or image given with `--romfs` (or `NX_ROMFS`) as a
  virtual image (`nx_runtime/src/romfs.rs`), read on demand, so a multi-gigabyte RomFS costs no
  memory.
- **Stops**: an SVC, a named port, a service or a command nobody implemented does not get an
  answer. The kernel records a report with the guest address and the number, ends the process and
  the produced program exits with status 101.

## The GPU front (`nx_gpu`)

`nx_gpu/` is a separate crate with no knowledge of Horizon: it depends on nothing, and `nx_runtime`
depends on it. It models what the NVIDIA driver service and the display stack do, and a command
processor for the Maxwell engines that decodes and counts what a title sends. Nothing is drawn
(`docs/reports/M8-gpu-front.md`).

```
guest ──SendSyncRequest──▶ nvdrv service ──▶ nx_gpu::nvdrv ──▶ nvmap, address space, syncpoints
                                                          └──▶ channel ──▶ Gpu::submit ──▶ pushbuffer
guest ──IHOSBinderDriver──▶ vi service ──▶ nx_gpu::display (parcels, buffer queue, compositor)     │
                                                                                                   ▼
                                    engines (3D, compute, 2D, copy, I2M): register files, MME ◀── methods
```

- `nvdrv/` is a byte-level state machine: `open`, `close`, `ioctl`, `query_event`, `take_signals`.
  The service (`nx_runtime/src/horizon/services/nvdrv.rs`) copies the argument buffers and turns the
  event tokens the core returns into kernel events. The core reads and writes guest memory only
  through the `GuestMemory` trait, which `nx_runtime` implements over the guest address space.
- `gpu.rs`, `channel.rs`, `pushbuf.rs` and `engine.rs` are the command processor: a GPFIFO entry is
  decoded, its pushbuffer is run through the pusher state machine, and each method write goes to the
  host methods or to the engine bound to its subchannel. `mme.rs` is the 3D engine's macro
  processor (the Fermi macro instruction set): a macro call is run when its command ends, and its
  writes go through the same path.
- `methods.rs` is generated (`tools/gen_methods.sh`) from NVIDIA's open-gpu-doc headers and only
  gives names to offsets, for traces.
- `parcel.rs` and `display.rs` are the Android `Parcel` codec and the buffer queue of a layer.
- Time: syncpoint fences complete at once; the vsync is a 60 Hz host thread that sets the display's
  event (`vi.rs`), and the audio renderer's frame event a 200 Hz one (`audren.rs`).

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
| GPU front | hand-built ioctl argument bytes, pushbuffers and assembled macros, with the layouts of NVIDIA's headers and `libnx` | `nx_gpu` unit tests and the service tests in `nx_runtime/src/horizon/tests.rs` |
