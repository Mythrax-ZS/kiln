# M12: the Switch 2 GPU front

Status: open. Started 2026-11-04.

The task (part of the owner's request of 2026-11-04): what a Switch 2 program's graphics need below
NVN2, at the driver level, until the title submits its first frames and they are traced. The plan
is in `docs/reports/M11-switch2-boot.md`. References: the NVN2 guide and the NVN guide the owner
supplied, NeXium (the owner's emulator) and Eglantine (a Switch 2 emulator derived from Eden,
GPL-3.0: used for what the services do, none of its code is taken), and the title's own SDK,
disassembled where the references disagree or say nothing; each fact used is in
docs/ASSUMPTIONS.md with its source.

## Where the title is

The title now sets up its graphics system, its display and its GPU driver, and makes its first GPU
submission. In order:

1. `ISelfController` 2000 hands over the supervisor, `nx2:gsv`, a TIPC session (A-194).
2. `nn::gsv::BootWithRemoteMemory`: commands 4, 11 (`RegisterClientHandle`), 101, and the
   permission checks 1501 and 1502.
3. `nn::nvsrv::BootWithRemoteMemory`: command 1402 opens `nx2:nvsrv`, the GPU driver's session,
   which speaks CMIF and is answered by the existing `nvdrv` service.
4. `nn::syncpt::sdk::InitializeEx`: the permission check 201, the syncpoint session (202, A-195),
   and the shared syncpoint pages (204, A-196).
5. `nn::gs::sdk::BootNativeMemoryManagement`: `QueryGrantedFeatures` (115), and the permission
   checks 1101 and 1102.
6. `nx2:gpu` (497, A-197): its event ids and a sub-session.
7. `nx2:vi2` (305, A-198): the display `default`, a layer and its history (seven timestamp
   recorders), the native event slots and their binding to the vsync and to the display's change
   sources, and the display's refresh rate, HDR status and largest buffer resolution, in the
   values the SDK's `nn::vi2` functions convert.
8. NvRm's device: `GET_CHARACTERISTICS` in its Switch 2 form (an Ampere T239, A-199),
   `GET_TPC_MASKS`, `VSMS_MAPPING`, `GET_ENGINE_INFO`, a graphics device (`nx2:gpu` 41), a GPU
   address space (`nx2:gpu` 33, 202, 203 and the supervisor's 434), the address space node with
   NVN2's unified partition, the syncpoint shim (A-200) and its VA regions.
9. GPU memory (A-201): `nx2:gpu` 101 and 211 map the program's memory into the Switch 2's one GPU
   address space, at its own address and where the program puts it, and set the jobs' events,
   which the program waits on.
10. A channel: a timeslice group (A-202, A-207), the channel's runlist (A-203), its binding to the
    group, its object context of the Ampere 3D class (A-204), its event 7 (A-205), its preemption
    modes (A-206), the address space, its GPFIFO, and the zcull binding.
11. **The first submission** (`SUBMIT_GPFIFO2`) reaches the command processor, which stops at
    `SET_OBJECT` of the Ampere 3D class, 0xC797: it has no engine for the Ampere classes yet.

Two findings changed what other emulators do:

- A TIPC reply's data follows the header and the handles with no padding, and a command keeps its
  layout when it fails, the handle slots included (A-129). The 16-byte alignment the runtime used
  had every `vi2` count, partition and flag read as 0.
- NvRm takes its 2D, 3D, compute and inline-to-memory classes from its table of the T239's classes
  only when the characteristics' flags 40 to 43 (nvgpu's `SUPPORT_2D`, `_3D`, `_COMPUTE`, `_I2M`)
  are set; without them it asked for an object context of class 0, which the other emulators map
  onto a class of their own choosing (A-199).

## What is left in this milestone

- The command processor for the Ampere classes (3D 0xC797, compute 0xC7C0 with QMD version 3,
  copy 0xC7B5, GPFIFO 0xC56F), MME64 macros, methods split across GPFIFO entries.
- `nx2:gpu`'s private regions (214, 215) and sparse mappings, when the title asks.
- The rest of `vi2` (228 and 229, presenting a layer's buffer) and the supervisor's native memory
  and image commands (111 to 114, 423 to 435, 1510), when the title asks.

## Tests so far

The supervisor's, `nx2:gpu`'s and `vi2`'s answers, the TIPC reply layout, the Switch 2's
characteristics, TPC masks, SM mapping, engine list, unified address space, syncpoint shim (read
by a semaphore acquire), timeslice groups, channel runlists and preemption modes are unit tested,
as are the kernel's Switch 2 values and the CPU forms.
