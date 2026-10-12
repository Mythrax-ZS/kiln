# recomp: final report

Eight milestones, written down in `docs/reports/`. This page collects what the brief asks the final
report to say: what is implemented, every interpreter fallback, every assumption, what is untested,
and, plainly, what was not verified. The detail and the numbers are in the milestone reports; the
numbers here are the ones measured at the end of M8.

## Where it stands

`recomp` reads the **already extracted** ExeFS and RomFS of a Nintendo Switch title (or a homebrew
NRO) and produces a native x86-64 executable (PE on Windows, ELF on Linux) and, on request, a shared
library, both linked against `nx_runtime`, which holds the guest CPU model and the Horizon emulation
layer. It never touches keys, NSPs or NCAs.

The owner's title, the only real one tried, **runs its main loop**: all four of its NSOs are
recompiled, its own `rtld` links them at run time, its SDK starts up, it talks to the 20 services
listed below, mounts its RomFS and reads from it, initializes the GPU through `nvdrv` and the
display through `vi:u`, and then draws: in 600 presented frames it submits 1,201 GPFIFO
batches, 30,641 macro calls of 11 macros and 138,125 indexed draws into a 1600x900 target, all of
which the command processor of `nx_gpu` decodes and counts (M8; M7 had stopped at the first call
to `nvdrv`). **Nothing renders**: no shader is read, no picture is made, the fonts are blank
stand-ins, the sound is not played and nobody's input reaches the title. The next work is shader
translation and a Vulkan backend, and nothing here is a claim that a game is playable.

## What each milestone delivered

| | Delivered | Measured |
|--|-----------|----------|
| M1 input layer | ExeFS (directory, PFS0), RomFS (image parser, builder for a directory), NPDM, storage layers | `docs/reports/M1-input-layer.md` |
| M2 loader | NSO0 and NRO0 (LZ4, SHA-256, MOD0 relocations, exports, `.eh_frame_hdr`) | the real title's `main` loads with 0 unsupported relocations |
| M3 decoder and interpreter | the integer A64 ISA, an interpreter that follows the Arm pseudocode | decoder against LLVM's assembler (217 encodings); interpreter against Unicorn |
| M4 lifter and code generation | SSA IR, discovery with jump tables, LLVM IR text compiled by clang, build driver, runtime | IR and native code against the interpreter; real code, 149,640 runs, 0 differences |
| M5 floating point and SIMD | exact soft-float, scalar FP and the whole of Advanced SIMD (non-crypto) in all four layers | 10.2 million Unicorn cases per group sweep, 8 million random words, 244,000 native runs |
| M6 kernel layer | 38 SVCs, handles, memory, threads, synchronization, IPC, `sm:` gated by the NPDM, homebrew start-up | 46 kernel tests, four native programs, 680 repeated runs |
| M7 title boot | several modules in one program, six services, the RomFS, transfer memory, the cryptographic extension | the title boots to `nvdrv` on Windows and on Linux |
| M8 GPU front | `nx_gpu`: `nvdrv` and its nodes, the display and buffer queue, a command processor with a macro engine; 14 more services; shared memory; the title's main loop | 600 frames, 1,201 submissions, 138,125 draws, 0 undecoded methods or macros; `docs/reports/M8-gpu-front.md` |

## Instructions

Everything is decoded strictly: an encoding the architecture leaves unallocated, or that a
Cortex-A57 lacks, is *unallocated* (executing it stops the run), never a guess.

- **Implemented in the decoder, interpreter, lifter and code generator**: the integer A64 ISA
  (data processing in all forms, branches, every load and store form including pairs, literals,
  exclusives and ordered accesses, the system instructions an EL0 program uses, barriers, hints,
  CRC32); scalar floating point (half, single, double, every rounding mode, flush-to-zero, default
  NaN, the exception flags), the whole of Advanced SIMD (vector and scalar forms, structure loads
  and stores, table lookups, permutes, shifts, saturating and rounding forms, the by-element forms)
  and, since M7, the cryptographic extension (`AESE`, `AESD`, `AESMC`, `AESIMC`, `SHA1C/P/M/H/SU0/SU1`,
  `SHA256H/H2/SU0/SU1`, and the 64-bit `PMULL`). Per-instruction tables: M3 for the integer set.
- **Valid but not implemented, stopping the run with the instruction named**: the LSE atomics
  (`CAS`, `SWP`, `LD<op>`), `MSR` (immediate), `SB`, flag manipulation (FlagM), MTE, SVE,
  pointer-authenticated loads and stores (the PAC hints run as no-ops), and the privileged `HVC`,
  `SMC`, `HLT`, `DCPS`, `ERET`.
- **Unallocated on the modelled core, so refused**: half-precision arithmetic (`FEAT_FP16`),
  `SQRDMLAH`/`SQRDMLSH`, the dot product, `FCMLA`/`FCADD`, `FMLAL`, `BF16`, `FJCVTZS`, `FRINT32/64`
  (A-030).
- **Implementation choices the architecture leaves open** follow Unicorn's Cortex-A57 where it
  agrees with the pseudocode, and the pseudocode where it does not (`docs/UNICORN_DISAGREEMENTS.md`,
  U-1 to U-11).

## Supervisor calls (41)

| Area | SVCs |
|------|------|
| Memory | `SetHeapSize` 0x01, `SetMemoryPermission` 0x02, `SetMemoryAttribute` 0x03, `MapMemory` 0x04, `UnmapMemory` 0x05, `QueryMemory` 0x06, `MapSharedMemory` 0x13, `UnmapSharedMemory` 0x14, `CreateTransferMemory` 0x15 |
| Process | `ExitProcess` 0x07, `GetProcessId` 0x24, `Break` 0x26, `OutputDebugString` 0x27, `GetInfo` 0x29 |
| Threads | `CreateThread` 0x08, `StartThread` 0x09, `ExitThread` 0x0A, `SleepThread` 0x0B, `GetThreadPriority` 0x0C, `SetThreadPriority` 0x0D, `GetThreadCoreMask` 0x0E, `SetThreadCoreMask` 0x0F, `GetCurrentProcessorNumber` 0x10, `GetThreadId` 0x25, `SynchronizePreemptionState` 0x36 |
| Handles and waiting | `CreateEvent` 0x45, `SignalEvent` 0x11, `ClearEvent` 0x12, `CloseHandle` 0x16, `ResetSignal` 0x17, `WaitSynchronization` 0x18, `CancelSynchronization` 0x19 |
| User-mode synchronization | `ArbitrateLock` 0x1A, `ArbitrateUnlock` 0x1B, `WaitProcessWideKeyAtomic` 0x1C, `SignalProcessWideKey` 0x1D, `WaitForAddress` 0x34, `SignalToAddress` 0x35 (types 0 and 1) |
| Time | `GetSystemTick` 0x1E |
| IPC | `ConnectToNamedPort` 0x1F (`sm:` only), `SendSyncRequest` 0x21 |

A call the program's NPDM does not enable stops the run as if it were unimplemented. The title's
NPDM enables 47 SVCs; **the ones among them that are not implemented** are 0x20, 0x22 and 0x23 (the light
and user-buffer IPC calls), 0x28 (`ReturnFromException`),
0x2C and 0x2D (physical memory), 0x2F, 0x32, 0x33 (thread activity and context) and 0x5F (cache
flush). `recomp inspect` prints this list, and the services not implemented, for any title. Also not implemented: the server side of IPC (`CreateSession`, `AcceptSession`,
`ReplyAndReceive*`: 0x40-0x44), `SignalToAddress` type 2, and `GetInfo` types 9, 19, 20, 24 to 28.

## Services

`sm:` is the one named port; every other service is reached through it and only if the program's
NPDM service list allows the name (`0x1015` otherwise). Implemented, with the commands used so far:

| Service | Commands |
|---------|----------|
| `sm:` | `RegisterClient`, `GetServiceHandle`; the control commands `ConvertCurrentObjectToDomain`, `CloneCurrentObject`, `QueryPointerBufferSize`, `CloneCurrentObjectEx` |
| `lm` | `OpenLogger`; `Log`, `SetDestination` |
| `apm` | `OpenSession`, `GetPerformanceMode`, `IsCpuOverclockEnabled`; session `Set/GetPerformanceConfiguration`, `SetCpuOverclockEnabled` |
| `appletOE` | `OpenApplicationProxy` and, behind it, the common state getter, self controller, window controller, audio controller, application functions, library applet creator (storages only) with the commands in `docs/reports/M7-title-boot.md`; three interfaces exist and implement nothing |
| `aoc:u` | `CountAddOnContent`, `ListAddOnContent` (nothing installed) |
| `pctl` | factory `CreateService`, `CreateServiceWithoutInitialize`; `Initialize`, `CheckFreeCommunicationPermission`, `IsRestrictionEnabled` |
| `fsp-srv` | `SetCurrentProcess`, `GetGlobalAccessLogMode`, `OpenDataStorageByCurrentProcess`, and (M8) save data on the host: `OpenSaveDataFileSystem` and the file system, file and directory interfaces |
| `nvdrv` | `Open`, `Ioctl`, `Close`, `Initialize`, `QueryEvent`, `SetAruid`, `Ioctl2`, `Ioctl3` over `nx_gpu`'s five device nodes |
| `vi:u` | the root service, the application display service (displays, layers, the vsync event), `IHOSBinderDriver` |
| `acc:u0`, `set` | one local user with a profile; the language and region settings |
| `hid` | the handheld Npad connected with idle input; six-axis start and stop; vibration devices |
| `time:u` | the user, network and local clocks, the steady clock, a UTC time zone |
| `nifm:u`, `bsd:u`, `ssl`, `prepo:u`, `nfp:user`, `friend:u` | offline: no network, no sockets or TLS contexts, reports discarded, no amiibo, no friends |
| `pl:u` | six stand-in system fonts (blank glyphs) in a shared memory |
| `audren:u` | a silent audio renderer and an audio device |

20 of the 42 services in the title's list are implemented; the other 22 (`audin:u`, `audout:u`,
`hwopus`, `ldr:ro`, `mii:u`, `nfc:user`, `bcat:u`, `csrng`, ...) are not, nor is any command of the
implemented ones that the title has not used. Each stops the run with its name. The offline and
stand-in services answer in ways that are assumptions (A-066 to A-076), not what a console says.

## Interpreter fallbacks

A fallback is an instruction the decoder recognizes as valid and the lifter does not translate; it
stops the run when reached. On the four modules of the owner's title (4,869,883 reachable
instructions): **none**. Before M7 there were 1,322 instructions in 421 ranges, every one of them
cryptographic (`AES*`, `SHA*`), all inside one library of the SDK module, none reached by the
start-up; they are now translated. The build report (`<name>.report.txt`) lists the ranges of any
fallback a program has, with their reasons. The 25 instructions of that title that *stop the run* if
reached are not fallbacks but its own markers and data: twelve `0xE7FFDEFE` words, the zero word
that starts three of the modules, and ten data words inside two modules' text.

## Assumed behavior

Every assumed or undocumented behavior is a row of `docs/ASSUMPTIONS.md` (A-001 to A-077) with its
source and its status. The ones with the widest effect:

- Memory: the module base and layout (A-006, A-041), the process memory map (A-034), 39-bit address
  space with 4 KiB pages (A-022), no ASLR.
- Time: the counter at 19.2 MHz (A-001); no cycle timing anywhere.
- The CPU: the Cortex-A57 identification values (A-013, A-014), no alignment check on plain
  accesses (A-015), the exclusive monitors (A-016, A-021), memory ordering mapped by barrier kind
  (A-025), floating-point control state in memory (A-026).
- The kernel: no scheduler (A-037), the mutex and condition-variable protocol as remembered from
  Atmosphère (A-038), IPC details (A-040), `sm:` result codes (A-035), firmware version, total
  memory and a few `GetInfo` values (A-036).
- The services: handheld, in focus, American English, no add-on content, no parental control, a
  fixed pseudo device id (A-042 to A-046); one local user (A-066); an idle handheld controller
  (A-070); the clocks and a UTC zone (A-071); an offline console (A-072); stand-in fonts (A-073);
  a silent audio renderer (A-075, A-076).
- The GPU front: the driver's ioctls, error codes, memory handles, address space and syncpoints
  (A-050 to A-056, A-065, A-069), the command processor and its fences (A-060 to A-063), the display
  and its 60 Hz vsync (A-057 to A-059), and the macro engine's instruction set and its exit-flag
  rules (A-064, A-077): none verified on hardware.
- The RomFS the program reads is a virtual image whose byte layout is this tool's (A-048); shared
  memory is placed by this runtime (A-074).

## Tests, in short

- Windows: 125 `nx_gpu` unit tests, 298 `nx_runtime` unit tests, 30 Unicorn differential tests, 85
  `recomp` tests, and the opt-in real-title tests (`RECOMP_TITLE_DIR`). Linux (WSL, Ubuntu, clang
  21): see `docs/reports/M8-gpu-front.md`.
- Layers compared with an independent oracle: the decoder with LLVM's assembler; the soft-float with
  the host FPU; the interpreter with Unicorn (tens of millions of cases and millions of random words,
  every difference classified); the IR evaluator and native code with the interpreter (thousands of
  instructions per encoding group, random programs, and every function of a real title, 149,640
  runs); the cryptographic instructions with FIPS 197 and FIPS 180 examples and with Unicorn.
- The kernel layer with its own Switchbrew contract and real threads; the services with real IPC
  messages built byte by byte; native programs built from assembled words and run as processes.
- The GPU front by hand-built argument bytes, pushbuffers and assembled macros, with the layouts of
  NVIDIA's headers and `libnx`; no independent oracle exists for it here, so its end-to-end check is
  that a real title's whole command stream (600 frames) is decoded with nothing left over.
- Mutations were injected into the lifter, the code generator, the floating-point and SIMD executors
  and into the service code of M7, and into the M8 code (about 55 mutations; seven were missed at first and led to stronger tests), and every one was caught in the end.

## Untested

- Guest-implemented servers, `SendSyncRequestWithUserBuffer`, user exception handlers and the SVCs
  listed above (stop the run, by design).
- Programs built with libnx or the Nintendo SDK other than the one title: every NRO and NSO the
  tests run was assembled from words.
- Behavior under heavy multithreading, including memory ordering (there is no concurrent litmus
  test) and the many-thread limit; priorities and core affinity do nothing (A-037).
- Speed beyond one number: the title's first 600 frames run at the 60 Hz the vsync event gives.
- The GPU front with any other title; compute dispatches, 2D blits, the copy engine and
  `nvhost-ctrl`, which the title does not use.
- Platforms other than Windows (x86-64, clang from Visual Studio) and Linux (WSL, Ubuntu 26.04,
  clang 21).

## Not verified

Plainly:

- **Nothing here was run on a Switch.** The oracles are the Arm pseudocode, Unicorn's QEMU model of
  a Cortex-A57, the host FPU, the published AES and SHA examples, the Switchbrew pages, `libnx`'s
  headers, NVIDIA's class headers and Mesa's description of the macro engine. Where a console may
  differ (a result code, a scheduling order, an undocumented value) it is an assumption, and it is
  listed.
- The services answer the way a freshly started console plausibly would. None of the answers was
  compared with a console or with Atmosphère's output; where the answer is not documented the run
  stops instead, except where a stand-in was needed to go on (the fonts, the idle controller, the
  offline answers), which are listed as assumptions.
- **One title.** Its first 600 frames are what was measured. What a different title asks for is
  unknown.
- Cycle timing, GPU timing, audio timing and any real-time behavior beyond the 60 Hz and 200 Hz
  threads.
- **Whether a title would ever be *playable***: nothing renders, plays sound or reads a person's
  input. That the title submits draws is a count, not a picture.

## The next work

`docs/reports/M8-gpu-front.md` ends with the proposal: interpret the state the draws read, then
shaders (SASS to SPIR-V, tested against a reference interpreter), then a Vulkan backend with `ash`.
Shaders and the backend need the owner's go-ahead; nothing of them is started. Independent of
them: input from a real device, audio output, and the 22 services of the title's list that are
not implemented.

## Repository state

The author of this work pushed once (after the M1 report) and never again; pushing waits for the
owner's request. `git log origin/main..` lists what the remote does not have. (The reflog of
`origin/main` shows later pushes made under the owner's identity by something other than the
author, so the remote may be ahead of what the first push suggests.)
