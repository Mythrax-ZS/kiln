# M6: the Horizon kernel layer and a homebrew NRO

Status: complete. Date: 2026-10-12.

## What was implemented

A kernel layer in `nx_runtime/src/horizon` (about 3,700 lines, and 1,370 lines of tests) that a
produced NSO or NRO runs on. `docs/ARCHITECTURE.md` ("The Horizon layer") explains how the parts
fit; the assumptions behind it are A-034 to A-040 in `docs/ASSUMPTIONS.md`.

- **Start-up.** The module descriptor (ABI version 2) carries a kind: plain (the provisional start
  of M4, kept for hosts and tests), NSO or NRO. An NSO or an NRO goes to the kernel layer. An NSO is
  entered the way the loader enters one (`X0` = 0, `X1` = the main thread's handle); an NRO the way
  the homebrew loader does (Switchbrew, "Homebrew ABI": `X0` points at a list of loader
  configuration entries, `X1` = all ones). The list holds the main thread handle, the applet type,
  `SyscallAvailableHint`/`SyscallAvailableHint2` (computed from the SVCs implemented), the process
  handle, a random seed and the firmware version. The main thread gets a stack (8 MiB for an NRO; the
  NPDM's size for an NSO) and a thread-local region (`TPIDRRO_EL0`); returning from the entry
  function ends the process with `X0` as the status.
- **Supervisor calls (38).** Arguments in `X0`-`X7`, results in `X0`-`X7`, as the Switchbrew SVC
  pages lay them out:

  | Area | SVCs |
  |------|------|
  | Memory | `SetHeapSize` 0x01, `SetMemoryPermission` 0x02, `SetMemoryAttribute` 0x03, `MapMemory` 0x04, `UnmapMemory` 0x05, `QueryMemory` 0x06 |
  | Process | `ExitProcess` 0x07, `GetProcessId` 0x24, `Break` 0x26, `OutputDebugString` 0x27, `GetInfo` 0x29 |
  | Threads | `CreateThread` 0x08, `StartThread` 0x09, `ExitThread` 0x0A, `SleepThread` 0x0B, `GetThreadPriority` 0x0C, `SetThreadPriority` 0x0D, `GetThreadCoreMask` 0x0E, `SetThreadCoreMask` 0x0F, `GetCurrentProcessorNumber` 0x10, `GetThreadId` 0x25, `SynchronizePreemptionState` 0x36 |
  | Handles and waiting | `CreateEvent` 0x45, `SignalEvent` 0x11, `ClearEvent` 0x12, `CloseHandle` 0x16, `ResetSignal` 0x17, `WaitSynchronization` 0x18, `CancelSynchronization` 0x19 |
  | User-mode synchronization | `ArbitrateLock` 0x1A, `ArbitrateUnlock` 0x1B, `WaitProcessWideKeyAtomic` 0x1C, `SignalProcessWideKey` 0x1D, `WaitForAddress` 0x34, `SignalToAddress` 0x35 (types 0 and 1) |
  | Time | `GetSystemTick` 0x1E (the runtime's counter at 19.2 MHz, A-001) |
  | IPC | `ConnectToNamedPort` 0x1F (`sm:` only), `SendSyncRequest` 0x21 |

  `GetInfo` answers types 0-8, 10-18, 21-23 (the rest per A-036). The program's NPDM decides which
  SVCs exist: a call its kernel capabilities do not enable stops the run like an unimplemented one.
- **Handles.** A table of 15-bit index plus 15-bit generation, so a closed handle never names the
  next object in its slot; the pseudo handles `0xFFFF8000` (current process) and `0xFFFF8001`
  (current thread).
- **Memory.** A block map over the guest address space records the type, permission and attribute
  that `QueryMemory` reports (types per Switchbrew; neighbours that are equal are merged). The
  regions `GetInfo` reports and where the heap, stack, alias, thread-local and loader
  configuration areas live are in A-034.
- **Threads.** Each guest thread runs generated code on its own host thread with its own CPU
  structure. `CreateThread` validates its arguments (priority 0-63, core, stack and entry
  addresses) and `StartThread` starts it; `ExitThread` and `ExitProcess` end a thread or the process
  and wake everything that waits. A fault, an unimplemented call or a `Break` in any thread ends the
  process with a report.
- **Synchronization.** Events (a writable and a readable handle), `WaitSynchronization` over events,
  threads and the process (first signalled index, timeout, `CancelSynchronization`), the mutex word
  protocol and condition variables of `ArbitrateLock`/`ArbitrateUnlock`/`WaitProcessWideKeyAtomic`/
  `SignalProcessWideKey`, and the address arbiter, all working on the guest's own words with
  compare-and-swap (`Runtime::compare_exchange` records the store for the exclusive monitors).
  Waiters are served by priority, then arrival (A-038).
- **IPC.** HIPC messages in the thread-local buffer are parsed and answered in place: header, pid,
  copy and move handles, pointer/send/receive/exchange descriptors, the CMIF header; domains
  (object id 1 is the root, objects from 2); the control commands `ConvertCurrentObjectToDomain`,
  `CloneCurrentObject`, `QueryPointerBufferSize`, `CloneCurrentObjectEx` (A-040). A service is a
  Rust `Service` that answers a `Request` with a `Response`, whose objects become move handles or
  domain object ids.
- **`sm:`.** `RegisterClient` and `GetServiceHandle`, gated by the title's NPDM service access list
  (a program without an NPDM, a homebrew NRO, may use whatever is implemented). A name outside the
  list answers `0x1015`; a name inside it that nothing implements stops the run (A-035). No other
  service is implemented.
- **Stops.** Nothing is answered with an invented result. An SVC, a named port, a service, an IPC
  control command or message type, or a `GetInfo`/`SignalToAddress` variant that is not implemented
  ends the run: the report on standard error names it and the guest address of the `SVC`, and the
  exit status is 101. `NX_TRACE_SVC=1` logs every call; `NX_RANDOM_SEED=<n>` makes the kernel's
  entropy reproducible.
- **Runtime changes this needed.** The stop reasons `EXIT_THREAD`, `EXIT_PROCESS` and
  `UNIMPLEMENTED`; the module descriptor's `kind`; `nx_svc` handing the call to the thread's
  kernel; `Runtime::compare_exchange`; an `alias` mapping in guest memory for `MapMemory`.

### Not implemented or not done

- Every service except `sm:`. Asking for any other stops the run with its name.
- SVCs: `MapSharedMemory`/`UnmapSharedMemory`/`CreateTransferMemory` (0x13-0x15),
  `SendSyncRequestLight` (0x20), `SendSyncRequestWithUserBuffer` (0x22),
  `SendAsyncRequestWithUserBuffer` (0x23), `ReturnFromException` (0x28), `CreateSession` and the
  server side (`AcceptSession`, `ReplyAndReceive*`: 0x40-0x44), `SignalToAddress` type 2, and all
  other SVCs the Switchbrew list has. A program that implements a server of its own therefore
  stops at its first `CreateSession`.
- A scheduler: threads run in parallel on host threads, priorities and affinity are stored and
  returned but do nothing (A-037). No user exception handling (a fault ends the process).
- Several modules in one process (rtld plus main plus subsdk, M7) and the guest's own `fsp-srv`.
- Cycle timing and anything about speed.

## Tests

`cargo test --workspace` (fmt and clippy clean, exit status of each step checked): 229 `nx_runtime`
unit tests (46 of them in the kernel layer), 29 Unicorn differential tests, 80 `recomp` tests (5 of
them build and run homebrew programs), plus the opt-in real-title tests.

| Layer | Test | Result |
|-------|------|--------|
| Handles, memory blocks, IPC bytes, results | 15 unit tests next to the code: generation counters, block splitting and merging, descriptor and handle decoding, reply layout, result code values, `sm:` name decoding | pass |
| Memory SVCs | the heap grows and shrinks and is queryable; permissions follow the state rules; `MapMemory` makes a second view and `UnmapMemory` gives the pages back; `QueryMemory` on modules, gaps and the ends; `GetInfo` reports the regions and refuses what it does not know | pass |
| Process and diagnostics | debug output and the tick; an unimplemented SVC stops and names itself; the NPDM decides which SVCs exist; `Break` with the notification bit returns and any other stops | pass |
| Events and waiting | signal/clear/reset; argument checks and the first signalled index; a blocked wait wakes on a signal and times out; `CancelSynchronization` interrupts a wait or the next one; ending the process wakes every waiter with the terminating result | pass |
| Threads | a created thread runs guest code and can be waited for; its arguments are validated; the main thread starts another and waits for it; a fault in any thread ends the process with a report | pass |
| Mutex, condition variable, arbiter | a waiter gets the mutex when the owner unlocks; the highest priority gets it first and the next still waits; a condition variable releases and retakes the mutex; a signalled waiter queues behind the current owner; a timeout returns without the mutex; `WaitForAddress`/`SignalToAddress` in the three wait types and two signal types; sleeping yields or waits and the end of the process interrupts it | pass |
| IPC and `sm:` | a program asks `sm:` for a service and calls it through the thread-local buffer; a domain holds many objects behind one handle; services outside the NPDM list are refused and unimplemented ones stop the run; unknown commands and ports stop with their numbers | pass |
| Start-up | an NRO is entered with the loader configuration and its return value is the exit status | pass |
| Native, end to end | five programs assembled from words, written into NRO/NSO files, built by the real pipeline with clang and run as separate processes (below) | pass |
| Kernel tests repeated | the 46 kernel tests 200 times in a row, then 480 times as 16 parallel loops on a 32-thread machine | 0 failures in 680 runs |
| Hello program repeated | the first two programs below, built once each by the command line and run 1,000 times each (16 at a time), standard output and exit status compared | 0 differences in 2,000 runs |

The five native programs (`src/build/e2e_tests.rs`, `mod homebrew`):

1. **Hello.** Prints through `OutputDebugString`; grows the heap by 2 MiB; creates an event and a
   thread whose stack is in the heap; both threads do 1,000 exclusive-load/store increments of one
   counter while racing; the main thread waits on the event the second one signals, checks the
   counter is exactly 2,000, connects to `sm:` and performs `RegisterClient` through the
   thread-local buffer, and returns 0. Expected standard output
   `hello from a homebrew NRO / counter ok / sm ok / done`, exit status 0. Only `ADR` refers to the
   worker's entry, so the analysis does not find it; this program is built with the entry given as
   a root (`--hints`), and both threads run compiled code.
2. **The same without the hint.** The worker starts in the slow path and runs interpreted while the
   main thread runs compiled code; same output, same counter.
3. **Unimplemented service.** `GetServiceHandle("hid")`: exit status 101 and a report that names
   `hid` and the SVC address.
4. **NSO entry.** `X0` must be 0 and `X1` a handle; returns 42.
5. **`ExitProcess` with a live thread.** A second thread sleeps forever; the main thread calls
   `ExitProcess`; the program prints once and exits 0 well inside the time limit.

### Validation on Linux

The same checks on Linux (WSL, Ubuntu, clang 21), on the final commit: fmt and clippy clean, 229
`nx_runtime` unit tests and 80 `recomp` tests pass, and the 80 `recomp` library tests also pass with
`RECOMP_REQUIRE_EXTERNAL=1` (a missing tool becomes a failure), so the five native programs were
built and run there. The Unicorn tests do not run there: the library is not installed in that
environment, they skip and pass (with `RECOMP_REQUIRE_EXTERNAL=1` they fail), so the
interpreter-versus-Unicorn results in this and the earlier reports come from Windows. The opt-in
real-title tests need a title and were not run on Linux.

## Notes from writing the tests

- The kernel tests share one runtime, so each one builds its environment inside a block and drops
  it before the next; a second environment created while the first was alive deadlocked the test
  harness (a harness fault, not a kernel one).
- A fake main thread in a test must be marked as started, or the rule "the process ends when the
  last started thread has finished" ends it at once. An NPDM without enable-SVC descriptors enables
  no SVC at all, so the tests that use an NPDM list the calls they need.
- Commit `84981e6` was committed with a clippy error in a test (`manual_is_multiple_of`) because the
  gate command that produced it hid clippy's exit status; `e9ce42b` fixes it. The gate now runs
  each step on its own and checks its own exit status.

## Not verified

- Nothing was run on a Switch. The SVC register layouts, memory types, enums and the homebrew loader
  configuration are Switchbrew's. The result values and the check order of `sm:` (A-035) and the
  mutex/condition-variable word protocol (A-038) are from memory of Atmosphère and are not on the
  Switchbrew pages; they are exercised by this project's tests but not compared with a console or
  with Atmosphère itself.
- No program built by libnx (the library homebrew is linked with) was run; every NRO and NSO used here
  was assembled from words in the tests. How libnx's start-up, its heap, its threads and its
  `sm:` calls behave on this kernel layer is therefore not known.
- Guest-implemented servers, shared memory and transfer memory, and the `*WithUserBuffer` calls.
- Correct behavior of a guest that depends on priorities or on which core a thread runs on (A-037).
- Many threads (thousands): each guest thread reserves 512 MiB of host stack address space; the
  limit was not found.
- Timeouts are measured on the host clock; their accuracy was not measured.

## Next

M7: boot a title that has a RomFS, from its extracted ExeFS, until the first call this runtime does
not implement; report it with its guest address and number; and name the next blocker. That needs
several modules loaded into one process (rtld, main, subsdk, sdk), a file system service backed by
the RomFS, and whatever services the title asks for before it reaches the GPU.
