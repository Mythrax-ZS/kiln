# recomp: final report

Seven milestones, written down in `docs/reports/`. This page collects what the brief asks the final
report to say: what is implemented, every interpreter fallback, every assumption, what is untested,
and, plainly, what was not verified. The detail and the numbers are in the milestone reports; the
numbers here are the ones measured at the end of M7.

## Where it stands

`recomp` reads the **already extracted** ExeFS and RomFS of a Nintendo Switch title (or a homebrew
NRO) and produces a native x86-64 executable (PE on Windows, ELF on Linux) and, on request, a shared
library, both linked against `nx_runtime`, which holds the guest CPU model and the Horizon emulation
layer. It never touches keys, NSPs or NCAs.

The owner's title, the only real one tried, **boots**: all four of its NSOs are recompiled, its own
`rtld` links them at run time, its SDK starts up, it talks to the services listed below, mounts its
RomFS and reads from it, and then asks for `nvdrv`, the GPU driver, which is the first call this
runtime does not implement (`GetServiceHandle("nvdrv")` from the guest address `0xad10d94`). That is
M7's result and the **GPU is the next blocker**. Nothing renders and nothing here is a claim that a
game is playable.

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

## Supervisor calls (39)

| Area | SVCs |
|------|------|
| Memory | `SetHeapSize` 0x01, `SetMemoryPermission` 0x02, `SetMemoryAttribute` 0x03, `MapMemory` 0x04, `UnmapMemory` 0x05, `QueryMemory` 0x06, `CreateTransferMemory` 0x15 |
| Process | `ExitProcess` 0x07, `GetProcessId` 0x24, `Break` 0x26, `OutputDebugString` 0x27, `GetInfo` 0x29 |
| Threads | `CreateThread` 0x08, `StartThread` 0x09, `ExitThread` 0x0A, `SleepThread` 0x0B, `GetThreadPriority` 0x0C, `SetThreadPriority` 0x0D, `GetThreadCoreMask` 0x0E, `SetThreadCoreMask` 0x0F, `GetCurrentProcessorNumber` 0x10, `GetThreadId` 0x25, `SynchronizePreemptionState` 0x36 |
| Handles and waiting | `CreateEvent` 0x45, `SignalEvent` 0x11, `ClearEvent` 0x12, `CloseHandle` 0x16, `ResetSignal` 0x17, `WaitSynchronization` 0x18, `CancelSynchronization` 0x19 |
| User-mode synchronization | `ArbitrateLock` 0x1A, `ArbitrateUnlock` 0x1B, `WaitProcessWideKeyAtomic` 0x1C, `SignalProcessWideKey` 0x1D, `WaitForAddress` 0x34, `SignalToAddress` 0x35 (types 0 and 1) |
| Time | `GetSystemTick` 0x1E |
| IPC | `ConnectToNamedPort` 0x1F (`sm:` only), `SendSyncRequest` 0x21 |

A call the program's NPDM does not enable stops the run as if it were unimplemented. The title's
NPDM enables 47 SVCs; **the ones among them that are not implemented** are 0x13 and 0x14 (shared
memory), 0x20, 0x22 and 0x23 (the light and user-buffer IPC calls), 0x28 (`ReturnFromException`),
0x2C and 0x2D (physical memory), 0x2F, 0x32, 0x33 (thread activity and context) and 0x5F (cache
flush). Also not implemented: the server side of IPC (`CreateSession`, `AcceptSession`,
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
| `fsp-srv` | `SetCurrentProcess`, `GetGlobalAccessLogMode`, `OpenDataStorageByCurrentProcess`; the storage's `Read` and `GetSize` |

Not implemented: the other 36 services in the title's list (`nvdrv`, `vi:u`, `hid`, `acc:u0`,
`time:u`, `set`, `nifm:u`, `bsd:u`, `audout:u`, `audren:u`, `hwopus`, `friend:u`, ...), and every
command of the implemented ones that the title has not yet used. Each stops the run with its name.

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

Every assumed or undocumented behavior is a row of `docs/ASSUMPTIONS.md` (A-001 to A-049) with its
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
- The services: handheld, in focus, American English, no add-on content, no parental control, no
  launch parameter, a fixed pseudo device id (A-042 to A-046).
- The RomFS the program reads is a virtual image whose byte layout is this tool's (A-048).

## Tests, in short

- Windows: 248 `nx_runtime` unit tests, 30 Unicorn differential tests, 84 `recomp` tests, and the
  opt-in real-title tests (`RECOMP_TITLE_DIR`). Linux (WSL, Ubuntu, clang 21): the same suite with
  every external tool required except Unicorn, and the title built and run to the same stop.
- Layers compared with an independent oracle: the decoder with LLVM's assembler; the soft-float with
  the host FPU; the interpreter with Unicorn (tens of millions of cases and millions of random words,
  every difference classified); the IR evaluator and native code with the interpreter (thousands of
  instructions per encoding group, random programs, and every function of a real title, 149,640
  runs); the cryptographic instructions with FIPS 197 and FIPS 180 examples and with Unicorn.
- The kernel layer with its own Switchbrew contract and real threads; the services with real IPC
  messages built byte by byte; native programs built from assembled words and run as processes.
- Mutations were injected into the lifter, the code generator, the floating-point and SIMD executors
  and into the new service code, and every one was caught.

## Untested

- Guest-implemented servers, shared memory, `SendSyncRequestWithUserBuffer`, user exception
  handlers and the SVCs listed above (stop the run, by design).
- Programs built with libnx or the Nintendo SDK other than the one title: every NRO and NSO the
  tests run was assembled from words.
- Behavior under heavy multithreading, including memory ordering (there is no concurrent litmus
  test) and the many-thread limit; priorities and core affinity do nothing (A-037).
- Speed. Nothing was measured and nothing is claimed.
- Platforms other than Windows (x86-64, clang from Visual Studio) and Linux (WSL, Ubuntu 26.04,
  clang 21).

## Not verified

Plainly:

- **Nothing here was run on a Switch.** The oracles are the Arm pseudocode, Unicorn's QEMU model of
  a Cortex-A57, the host FPU, the published AES and SHA examples and the Switchbrew pages. Where a
  console may differ (a result code, a scheduling order, an undocumented value) it is an
  assumption, and it is listed.
- The services answer the way a freshly started console plausibly would. None of the answers was
  compared with a console or with Atmosphère's output; where the answer is not documented the run
  stops instead.
- **One title.** Its start-up ends at `nvdrv`. What it, or any other title, does after that, and
  what a different title asks for first, is unknown.
- Cycle timing, vertical sync, audio timing and any real-time behavior.
- Whether a title would ever be *playable*: nothing renders, plays sound or reads input.

## The GPU, the next blocker

`docs/reports/M7-title-boot.md` ends with a proposal for the milestone that follows (`nx_gpu`):
`nvdrv` and its device nodes, the display path with `vi` and its buffer queue, a Maxwell command
processor that first only traces what the title sends, a shader translator from Maxwell programs to
SPIR-V tested against a reference interpreter, and a Vulkan backend with `ash`. It is a proposal:
nothing of it is started.

## Repository state

The commits since the report of M1 are local; only the work up to that commit was pushed to
`https://github.com/Mythrax-ZS/recomp.git`. Pushing again waits for the owner's request.
