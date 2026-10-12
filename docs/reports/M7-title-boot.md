# M7: booting a title with a RomFS

Status: complete. Date: 2026-10-12.

The task was to boot a title from its extracted ExeFS and RomFS until the first call this runtime
does not implement, and to report it. The title used is the owner's own dump; this report gives
aggregate numbers only and does not name it.

## Result

The title is recompiled whole (all four of its NSOs) and runs natively, on the project's own kernel
layer, through the same steps a console takes:

1. The program starts in `rtld`, the title's own runtime linker. It finds the other three modules
   with `QueryMemory`, relocates them in guest memory and enters the SDK.
2. The SDK start-up runs: `GetInfo` for the address-space layout and entropy, thread and TLS set-up,
   the service manager (`sm:`), the log manager (`lm`), the applet manager (`appletOE` and the
   eight interfaces the title takes from it), the parental-control service (`pctl`, after the variants `pctl:a`,
   `pctl:s` and `pctl:r` are refused because they are not in the title's service list), the
   add-on-content service (`aoc:u`), a 96 MiB transfer memory for game-play recording, the heap (a
   3.1 GiB `SetHeapSize`), and the file system proxy (`fsp-srv`).
3. The title mounts its RomFS: it asks for the data storage of the current process and reads from
   it through `IStorage::Read` (seven reads, served from the extracted directory).
4. It lends 8.6 MB to a transfer memory and asks the service manager for **`nvdrv`**.

**The first unimplemented call** is `GetServiceHandle("nvdrv")` (`sm:` command 1) sent by
`SendSyncRequest` at guest address `0xad10d94`, in the SDK module (offset `0xa9d94`). The produced
program stops there and says, on standard error and with exit status 101:

```
recomp runtime: the program asked the service manager (sm: command 1, GetServiceHandle) for
`nvdrv`, which it may use and which this runtime does not implement (SVC at 0xad10d94)
```

**The next blocker is the GPU.** `nvdrv` is the NVIDIA driver service, and the transfer memory the
title lent just before asking is the memory that service's `Initialize` takes. Everything after
this point in a title's start-up (display layers and buffer queues through `vi:u`, GPU memory and
command submission through `nvdrv`'s device nodes, the NVN library's command streams and Maxwell
shaders) is graphics. It is out of scope until this milestone passes, and it is the proposal at the
end of this report.

The run takes about a second. The trace has 165 supervisor calls, all on the main thread (66
`SignalProcessWideKey`, 52 `SendSyncRequest`, 21 `QueryMemory`, 14 `GetInfo`, and a few others);
the 52 `SendSyncRequest` calls carry the IPC requests to the services listed below.

## What was implemented

- **Several modules in one program** (`src/program.rs`, A-041). `Program` holds the modules in the
  order the kernel's loader maps them (`rtld`, `main`, `subsdk0` to `subsdk9`, `sdk`) back to back
  from `0x0800_0000`; the program starts in the first. The analysis reads all of them as one
  address space (`ProcessCode`), the image the produced program embeds is the *unrelocated* one
  that the title's `rtld` then relocates (`Module::unrelocated_image`, built from what every
  relocation overwrote), and the descriptor lists every module's segments. `recomp build` and
  `analyze` take all NSOs of an ExeFS; `--main-only` restores the single-module behavior that the
  earlier reports measured.
- **Services** (`nx_runtime/src/horizon/services/`): `lm`, `apm`, `appletOE`, `aoc:u`, `pctl`,
  `fsp-srv`, each with the commands the title uses and an honest stop for the rest:

  | Service | Interfaces and commands |
  |---------|-------------------------|
  | `sm:` | `RegisterClient` (0), `GetServiceHandle` (1); control commands 0, 2, 3, 4 |
  | `lm` | `OpenLogger` (0); `Log` (0), `SetDestination` (1) |
  | `apm` | `OpenSession` (0), `GetPerformanceMode` (1), `IsCpuOverclockEnabled` (6); session 0, 1, 2 |
  | `appletOE` | `OpenApplicationProxy` (0); proxy 0, 1, 2, 3, 4, 10, 11, 20, 1000; common state getter 0, 1, 5, 6, 9, 60, 61; self controller 1, 2, 9, 10-14, 16; window controller 1, 10; audio controller 0, 1, 2; application functions 1, 21, 30-33, 40, 50, 60, 65, 66, 67, 90, 124, 130, 160; library applet creator 10; storage 0; storage accessor 0, 10, 11. The display controller, process winding controller and debug functions exist and implement nothing. |
  | `aoc:u` | `CountAddOnContent` (2), `ListAddOnContent` (3): nothing is installed |
  | `pctl` | factory 0, 1; service 1, 1001, 1031: nothing is restricted |
  | `fsp-srv` | `SetCurrentProcess` (1), `GetGlobalAccessLogMode` (1005), `OpenDataStorageByCurrentProcess` (200); storage `Read` (0), `GetSize` (4) |

  The values these report (handheld, in focus, American English, no add-on content, a fixed
  pseudo-device id, ...) are A-042 to A-049 in `docs/ASSUMPTIONS.md`, each with its source.
- **Supervisor calls**: `CreateTransferMemory` (0x15) and `memobj.rs` (the range is reprotected and
  locked until the handle is closed, then it is the program's again); `GetInfo` reports the
  application memory pool a 4 GiB console has (`0xCD500000`), because the title sets a 3.1 GiB heap.
  39 SVCs are implemented now.
- **The RomFS** reaches the produced program through `--romfs` at build time (its absolute path is
  stored in the module descriptor, ABI version 3) or `NX_ROMFS` at run time, and is served as a
  virtual image read on demand from the extracted directory: the 5.5 GB tree costs no memory.
- **The cryptographic extension** (`AESE`, `AESD`, `AESMC`, `AESIMC`, `SHA1C/P/M/H/SU0/SU1`,
  `SHA256H/H2/SU0/SU1`; the 64-bit `PMULL` was done in M5). The title contains 1,322 of these
  instructions in 421 ranges of its SDK module (in one library, about 20 KB of code, not reached by
  the start-up) and they were the only interpreter fallbacks of the whole program. Now the analysis
  reports **0 interpreter fallbacks** among the 4,869,883 instructions it reaches.
- **Tools for iterating on a title**: `NX_TRACE_SVC=1` now also logs every IPC request with the
  service name, the command and the result, the service names asked of `sm:` and whether the NPDM
  allowed them, and marks calls that may block when they start (so a hang shows where it waits);
  `NX_TRACE_LOG=1` prints what the program logs through `lm`; `recomp relink` links the objects of
  an earlier build against the current runtime in seconds, and `--reuse-objects` rebuilds the
  descriptor and the image without recompiling the code.
- A **latent bug in the jump-table analysis**: a vector load (`LDR Q`) in the straight-line code in
  front of an indirect branch asked the image for a 16-byte value through an interface that reads
  1 to 8 bytes, and the analysis panicked (found by the first analysis of the SDK module).

## The analysis of the whole program (aggregate)

| | |
|--|--|
| Modules (`rtld`, `main`, `subsdk0`, `sdk`) | 4, 57 MB mapped in all (zeroed bss included) |
| Functions found | 115,538 (74,820 in `main` alone) |
| Guest instructions reached | 4,869,883 (3,123,640 in `main` alone) |
| Jump tables resolved / indirect `BR` left to the function table | 1,075 / 15,695 |
| Calls or branches outside executable memory | 0 |
| Interpreter fallbacks | **0** (1,322 before the cryptographic extension, all of it) |
| Instructions that stop the run if reached | 25: 12 `0xE7FFDEFE` markers, the zero word that starts three modules, and 10 data words inside the text of `rtld` and `sdk` that the analysis reaches as code |
| Time | discovery 1.5 s, lifting and emitting 1.9 s, compiling at `-O1` 36.7 s (790 units), linking 1.7 s, on 32 threads |
| Produced program | 300 MB (57 MB of it the embedded image) |

## Tests

`cargo test --workspace` on Windows (fmt and clippy clean, exit status of each step checked): 249
`nx_runtime` unit tests, 30 Unicorn differential tests (14 integer, 7 floating point, 9 SIMD and crypto), 84 `recomp` tests.

| Layer | Test | Result |
|-------|------|--------|
| Loader | the unrelocated image differs from the relocated one exactly at the slots a relocation wrote (a mutation that skips the undo is caught) | pass |
| Several modules | two synthetic NSOs mapped back to back, the first calls a function in the second; the program returns 42 when its data slot is still unrelocated, 41 when the relocated image is embedded | pass |
| Descriptor | the structure the runtime reads is 96 bytes and carries the RomFS path | pass |
| Tools | `relink` links the objects of an earlier build into a working program | pass |
| Jump-table bus | a 16-byte read is two reads | pass |
| Transfer memory | argument checks, reprotection, the locked attribute, the lent range visible to a service, release on close, memory restored with its contents (a mutation that skips the release is caught) | pass |
| Applet manager | through real IPC messages: the process handle copied in, the event handle copied out, the event signalled once, `ReceiveMessage` gives 15 and then `0x680` (a mutation that drops the message is caught) | pass |
| `fsp-srv` | the data storage is read through real receive descriptors and compared byte for byte with the image: header, an inner range, the tail, the whole image; a read past the end (or one whose end overflows) answers 2002-3005 and a program without a RomFS stops the run (a mutation of the offset is caught) | pass |
| Storages | a storage created through the library applet creator: size, a write from a send buffer, a read into a receive buffer, writes and reads outside it (2128-0503, nothing changed), `PopLaunchParameter` (2128-0002) | pass |
| Each service | unit tests on the commands, the refusals and the stops (`lm` decodes log packets; `apm` remembers configurations per mode; ...) | pass |
| Cryptographic extension, algorithms | the S-box against FIPS 197; MixColumns against its published example; a whole AES-128 encryption and decryption against the FIPS 197 example (the round keys come from an independent key schedule); SHA-1 and SHA-256 of "abc" computed with the instructions against the known digests (the round constants are derived from primes in the test) | pass |
| Cryptographic extension vs Unicorn | the three encoding groups, 60,000 to 120,000 cases each: 60,000 AES, 104,914 SHA-1/SHA-256 three-register and 44,872 two-register instructions executed on both sides with random registers, plus the reserved neighbours both sides reject | 0 differences |
| Random words vs Unicorn, again | the 8,000,000-word run of M5 with the new decoder: 2,389,543 executed on both and equal (473 more than before, the cryptographic words that used to be "unsupported"), 4,727,577 unallocated on both, 853,722 in groups still not implemented (all atomics, SVE, MTE, pointer authentication, FlagM and the privileged forms), and the same documented strictness differences as in M5 | 0 unexplained |
| Lifted and native vs interpreter | the three crypto groups join the 61 groups of M5 (937 instructions per group in the IR evaluator test; 25,600 functions x 8 states = 204,800 native runs in a release run) | identical |
| The title | analysis and build of all four modules; the run to the stop above; each service request checked against the Switchbrew command tables | as described |

### Linux

The same checks on Linux (WSL, Ubuntu, clang 21), on the commit that adds the cryptographic
extension: fmt and clippy clean, 248 `nx_runtime` unit tests and 84 `recomp` tests pass, and the 84
`recomp` library tests also pass with `RECOMP_REQUIRE_EXTERNAL=1` (so every end-to-end program,
the two-module program included, was built and run there). The Unicorn tests skip there (the
library is not installed in that environment), so those results come from Windows.

The title itself, built and run on Linux from the same extracted directories (read through
`/mnt/c`): 115,538 functions and 4,869,883 instructions (the same numbers as on Windows), 0
interpreter fallbacks, discovery 1.6 s, lifting and compiling 46.2 s, linking 0.8 s, and the run
ends the same way: the same stop at `nvdrv`, at the same guest address, with the same count of
supervisor calls in the trace.

## Defects the work found

- The jump-table panic above.
- The lifted-versus-interpreter test could not tell a store into the program's own code from a
  difference: the test memory is flat and writable, and three more groups in the table changed
  which random registers pointed into the code. The test memory now refuses stores to the code
  (as the guest memory of a produced program does, whose code pages are read/execute), on both
  sides.
- `GetInfo` reported 2 GiB of memory for a title that sets a 3.1 GiB heap; a real console's pool is
  larger, and the value is now that pool's size (A-036).

## Not verified

- Nothing was run on a Switch. Every service answer is Switchbrew's command table plus choices of
  ours (A-042 to A-049), and the `libnx` layouts of the commands Switchbrew leaves open; none was
  compared with a console. Where the real answer is not documented the run stops instead of guessing
  (`GetPerformanceConfiguration` before any `Set`, a RomFS the program does not have). The result
  codes the services answer with come from Switchbrew's error code list (2128-0002, 2128-0003,
  2128-0503, 2002-3005), which names them but does not say which command gives which, and from
  Atmosphère's `sm_results.hpp` (the `sm:` codes); `PopLaunchParameter` answering 2128-0002 is the
  one I am least sure of, and the title did not call it before it stopped (A-043).
- One title. Its start-up ends at `nvdrv`; what a different title asks for first is unknown.
- Everything after `nvdrv`: `hid`, `time:u`, `set`, `acc:u0`, `nifm:u`, `bsd:u`, the audio services,
  `friend:u`, `ldr:ro` and the rest of the 42 services the title may use, which it has not asked for
  yet, and whether it would wait for something the applet manager never sends (for example the
  home menu's messages) later on.
- Threads: this start-up uses one thread. The multi-threaded behavior is tested (M6) but no title
  workload was.
- The RomFS image's byte layout differs from the console's (A-048); a title that read the RomFS
  header and checked offsets or hashes of the original would behave differently. This one parses
  the tables it is given.
- Timing of any kind.

## Next: the GPU (`nx_gpu`)

What the title will ask for next, in the order its start-up goes, and a way to build it. None of
this is started, and nothing here promises that a game will be playable.

1. **`nvdrv` and its device nodes** (`nvmap`, `nvhost-ctrl`, `nvhost-ctrl-gpu`, `nvhost-as-gpu`,
   `nvhost-gpu`): the `Initialize` that takes the transfer memory just seen, `Open` of a node by
   path, `Ioctl` on a file descriptor with an input and output buffer (a few dozen different ones, as far as the public
   documentation says), `QueryEvent` for the syncpoint and error events, `Close`. This is a service
   like the others, with a real state machine (memory handles, an address space of GPU virtual
   addresses, channels, syncpoints and fences).
2. **The display path** (`vi:u`/`vi:m`, the `IHOSBinderDriver` with the Android-style `Parcel`s and
   a buffer queue, `ISelfController::CreateManagedDisplayLayer`): a window and swap chain on the
   host (Vulkan through `ash`), layers, the fence handshake between producer and consumer.
3. **The command processor**: the GPFIFO and pushbuffers `nvhost-gpu` submits, the Maxwell method
   interface of its engines (3D, compute, 2D, copy, inline-to-memory), and the state they set.
   First as a tracer that decodes and prints what the title sends, so the real workload is known
   before anything is drawn.
4. **Shaders**: the title's Maxwell programs (SASS with its header) translated to SPIR-V. This is
   the largest piece. It can be tested without the title: assemble small programs, translate, run
   both the translation (on a software Vulkan implementation) and a reference interpreter of the
   instruction set on random inputs, and compare, the way the CPU side is tested.
5. **The Vulkan backend** (`ash`): memory and resource management, texture formats and the
   block-linear layout, render targets, pipeline and descriptor caches, synchronization against the
   syncpoints.
6. Presentation, frame pacing and resolution, last.

Work from step 1 to step 3 alone would let the title run its start-up and main loop up to its
first draw call, with every GPU method it uses listed. That is the measurement the rest depends on.
The clean-room rule stays: Switchbrew and the public hardware documentation for the formats; the
GPL emulators are read for architecture only and no code is taken from them.
