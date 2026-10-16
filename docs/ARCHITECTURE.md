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
| GPU | `nx_gpu` | The driver service and its device nodes, the display and buffer queue, and a command processor with a macro engine. It decodes the 3D state (`state.rs`, `draw_state.rs`), keeps the constant buffer bindings, counts the shader programs draws use (`shaders.rs`), and hands each draw, clear, compute launch (`compute.rs` decodes its descriptor) and displayed buffer to a `Backend` (`render.rs`) when there is one; the inline-to-memory engine (`i2m.rs`) writes data sent in the command stream to memory. |
| Shaders | `nx_shader` | The Maxwell program header, the instruction decoder, the control-flow graph (with the `SSY`/`PBK` stacks), a SPIR-V module builder, and the translation of a program to SPIR-V (`translate/`): registers and predicates as function variables, the blocks in address order each under a flag of its own when every jump goes forward and a dispatch loop over the blocks when one goes back (A-114), for the vertex and pixel stages and for a compute program as the title launches it (the kernel stage); `alu`, `int`, `io`, `tex`, `global` and `flow` hold the instructions (149 of 296 table rows, `example coverage`). |
| Vulkan | `nx_vulkan` | The Vulkan device (loaded at run time), buffers, images with staging upload, a compute harness with textures and device-address patching (the translator's tests run its shaders here against plain Rust arithmetic), and the renderer, a `Backend` that draws with dynamic rendering, cached pipelines, textures (detiled, ASTC decoded by `astc.rs`), render targets that can be sampled or written back to the memory a texture reads, compute launches that run to the end before the next command, and guest memory imported for global access. |
| Front end | `kiln_gui` | The desktop window around the command line: pick a game, recompile, keep a library of games and the settings each runs with. It runs `kiln` and the games as processes. |

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
- **Indirect target**: `nx_dispatch` looks the address up in the function table (open addressing,
  built from the registered functions the first time a lookup follows a registration, read
  without a lock) and tail calls the function, or tail calls the slow path. The slow path interprets from that address,
  logs the address (`NX_UNKNOWN_LOG`), and hands control back to native code at the next known
  entry. The log is a hints file (`kiln build --hints`).
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
base register's address computation and writeback are ordinary integer IR.

The hot forms are lowered to LLVM IR instead (`src/codegen/llvm/vecs.rs`: the emitter decodes the
word with the runtime's own decoder, and the call stays as the slow form of anything that is not
on the fast path):

- *Exact by construction, inline*: register moves, the bitwise operations, `MOVI`, `MVNI`,
  `ORR`/`BIC` with an immediate, `FMOV`, `FABS`, `FNEG`, `FCSEL`, the immediate and
  general-register forms of `FMOV`, `DUP`/`INS` of an element, `ZIP`, `UZP`, `TRN`, `EXT`.
- *Guarded*: the single-precision `FADD`, `FSUB`, `FMUL`, `FDIV`, `FCMP`, `FCCMP`, `FMADD`, `FMSUB`,
  `FNMADD`, `FNMSUB`, the vector `FADD`, `FSUB`, `FMUL`, `FMLA`, `FMLS` (also by element, and the
  scalar forms by element), `FADDP`, `SCVTF`/`UCVTF` of 32-bit integers and `FCVTZS`/`FCVTZU` to a
  32-bit register. The fast form runs when `FPCR` is zero (or has only flush-to-zero, default
  NaN and alternative half precision set, and then no operand is a denormal: the architecture would
  read it as zero and raise `IDC`) and every result is finite and not tiny (zero results only when
  exact); then the host's answer is the architecture's, and the one flag it can raise, inexact, is
  worked out exactly (a sum by error-free transformation, a product, a quotient or a fused sum in
  double precision, which holds a product of two singles exactly, leaving to the call the one case
  where rounding twice could differ) so that `FPSR` ends as the soft-float leaves it. NaNs,
  infinities, denormals, overflow and other rounding modes take the call. The conversions to an
  integer keep the strict test (`FPCR` zero).
- *Memory*: the loads and stores of one or two registers of 1 to 16 bytes are 64-bit (or narrower)
  accesses through the same `nx_ld*`/`nx_st*` as the integer ones.

The native differential tests (`src/native_tests.rs`) compare all of it with the interpreter over
random operands (ordinary numbers, exact ones and the special values) and states, `FPSR` and the
flags included; a mutation of any of these lowerings fails them. Everything else (double
precision, the other conversions and the rest of SIMD) is the call.

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
- **Cores**: by default the guest's threads run at once on host threads. `NX_CORES=per-core|one`
  (`cores.rs`, A-115) makes a thread hold a core, a ticket queue, to run guest code and give it up at the
  supervisor calls that can block, which is a model of one thread at a time on each of the console's
  cores (without priorities).
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
- **Input** (`horizon/input.rs`, `horizon/gamepad.rs`): the keys of the window (named by physical
  key, bound to buttons and stick directions by a keymap) and an XInput gamepad are merged into one
  state of the controls, which the sampling thread of `hid` writes into the handheld Npad every
  5 ms (docs/ASSUMPTIONS.md A-104).
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
processor for the Maxwell engines that decodes and counts what a title sends. It draws nothing itself: pictures are the business of a `Backend`, the renderer below
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

## The renderer (`nx_vulkan`)

`nx_vulkan::Renderer` is a `nx_gpu::render::Backend`. The command processor calls `draw` and
`clear` as commands arrive, `finish` when a GPFIFO submission ends and `present` when the title
queues a buffer; the Vulkan device and everything on it are made at the first of those calls
(`NX_RENDERER=vulkan` turns it on in a produced program; without it the processor only counts).

```
registers ─▶ nx_gpu::draw_state ─▶ pass (targets) ─▶ pipeline key ─▶ cached pipeline
program slots ─▶ shaders (by address and hash) ─▶ nx_shader::translate ─▶ SPIR-V module
texture handles ─▶ header pools ─▶ textures (detile, ASTC decode, hash) or a render target's view
constant buffers, vertices, indices ─▶ upload ring ─▶ one command buffer per submission
```

- **Passes and targets** (`render/mod.rs`, `targets.rs`): a color or depth target is an image found
  again by the guest address of its memory (and made, zero-filled, the first time); a rendering is
  begun with the attachments the registers name and ended when they change, when a texture needs a
  render target in a sampling layout, or when the submission ends.
- **Textures** (`textures.rs`, `astc.rs`): a draw's handles are looked up in the header and sampler
  pools, the memory is read and detiled (block linear, with mip levels), ASTC blocks are decoded to
  plain texels on the host (the device usually cannot sample them), and the image is cached by its
  header and the hash of its bytes (taken in place, a word at a time, and compared again at most
  once a frame; the memory is copied only when the hash changed). A texture whose memory is the
  address of a render target of the same size and format is that target's image; one made from
  memory that targets drawn to cover gets the targets written to that memory first (a cube map
  the title renders face by face), `Targets::write_back`.
- **Pipelines** (`draw.rs`, `shaders.rs`): the program of each stage is translated once per
  variation (attribute classes, target classes, depth range) and the pipeline is cached by the
  programs, vertex layout, topology, formats, blend, depth clamp and descriptor layout; viewport,
  scissor, culling, depth and stencil state and the instance counter are set per draw.
- **Memory a program reads by address** (`global.rs`): the guest's memory is imported into the
  device where it is, and a table of GPU address ranges lets the translated program find it.
- **Compute launches** (`render/compute.rs`, `nx_gpu/src/compute.rs`, `i2m.rs`): the title uploads
  a launch descriptor with the inline-to-memory engine and signals the launch; the descriptor is
  decoded (grid, workgroup, constant buffers, program), the program is translated as a kernel and
  run at once with the commands before it flushed and the launch waited for, so that what it
  writes is in memory before a later command reads it.
- **Looking** (`debug.rs`, `png.rs`): frames, every render target, and the target after each
  command of one frame can be written as PNG, with a line of statistics; the draws, state, texture
  headers and constant buffers of a frame can be traced (`docs/reports/M10-vulkan-backend.md`). With
  `NX_TRACE_GPU` the renderer reports every 100 frames where the time of the draws went, step by
  step. `NX_WATCHDOG=<seconds>` (`nx_runtime/src/horizon/watchdog.rs`) says, when no frame has
  been presented for that long, what every guest thread is in (the supervisor call, the service
  and command of a request, the return addresses as of the last call) and what the GPU side was
  last doing (`nx_gpu::progress`). A run that stops prints the same for every thread, then the
  registers of the thread that stopped and what they point at. For a lost Vulkan device,
  `NX_DEBUG_CHECKPOINTS` (NVIDIA) marks the commands so that the report names the draw it was in,
  and `NX_DEBUG_SYNC_FROM` waits for the device after every draw so that the draw is named with its
  state, textures and the time the device took to be lost.
- **Showing** (`window.rs`, `render/display.rs`, `render/blit.rs`): with `NX_WINDOW=1` a window on
  a thread of its own (winit) gets the shown part of each displayed buffer, blitted into a swapchain
  image with black bars to keep its shape (FIFO presenting, so the guest runs at the display's
  refresh rate); the same window hands the keys and the keyboard focus to the runtime
  (`InputSink`, `input.rs`). Closing it ends the process.

## Guest memory

One checked interface (`nx_runtime/src/rt/mem.rs`): a 39-bit address space of 4 KiB pages through a
two-level table of atomic entries. Generated code calls `nx_ld*`/`nx_st*` (and the acquire/release
and exclusive variants); each is followed by a test of the fault flag. A test pins the list of
runtime functions the emitter calls against the runtime's exports.

The plain and the acquire/release accesses of 1, 2, 4 and 8 bytes first try a short path in the
export itself (`fast_ptr`): an aligned address below 2^39 whose page is mapped with the permission
needed is read or written with one atomic access, through the table's first level (published as
`nx_mem_l1` when the runtime is made, so no `rt()` call is needed). A store takes this path only
while no reservation is outstanding anywhere (`nx_mem_armed` is zero, the same test the general
path makes). Everything else (unaligned or page-straddling addresses, unmapped pages, missing
permissions, 16-byte accesses, stores while a reservation exists) takes the general path, which
reports the fault or does the copy; a test compares the two over the edges of mapped, read-only
and unmapped pages and the end of the address space.

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
| Shader translation | Rust arithmetic, run on a GPU in a compute harness, per instruction family | `nx_vulkan/tests/translate_*.rs` |
| Renderer | the pictures expected of hand-built scenes (clears, draws, blending, textures, render targets, global memory, instances), compared pixel by pixel on a GPU | `nx_vulkan/tests/render_*.rs` |
| ASTC decoder | hand-built blocks, and ARM's reference decoder (`astcenc`) on random blocks of all footprints, opt-in | `nx_vulkan/src/astc.rs`, `nx_vulkan/tests/astc_reference.rs` |
