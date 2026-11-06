# M12: the Switch 2 GPU front

Status: complete. Started 2026-11-04, passed 2026-11-05: the title submits its frames, and they
are traced and presented (900 in a run stopped by `NX_MAX_FRAMES`). Nothing is drawn yet: the
renderer refuses every draw at its programs, which are sm_86 (M13).

The task (part of the owner's request of 2026-11-04): what a Switch 2 program's graphics need below
NVN2, at the driver level, until the title submits its first frames and they are traced. The plan
is in `docs/reports/M11-switch2-boot.md`. References:
NeXium (the owner's emulator) and Eglantine (a Switch 2 emulator derived from Eden,
GPL-3.0: used for what the services do, none of its code is taken), and the title's own SDK,
disassembled where the references disagree or say nothing; each fact used is in
docs/ASSUMPTIONS.md with its source.

## Where the title is

The title sets up its graphics system, its display and its GPU driver, submits its commands on four
channels and presents its frames. In order:

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
11. The first submission (`SUBMIT_GPFIFO2`) reaches the command processor. The Ampere classes
    run on the Maxwell engines, whose methods they keep, with MME64 for their macros (A-208) and
    QMD version 3 for compute launches (A-210).
12. Native memory and images (A-212), GPU-only regions backed by the graphics server (214 and
    215, A-215), GameChat and the camera (A-220 to A-222, A-234).
13. The macros. NVN2's macros fall through into the next and read more parameters than a call
    has; a missing one reads 0 (A-237). The macro code is each channel's: the title's two compute
    channels load their own code at instruction 0 after the 3D channel has loaded its macros there,
    and with one store for the process the 3D macros were replaced and drew nothing. With a store
    per channel the same start makes 1,141 draws and 290 clears in its first 644 submissions.
14. Synchronisation across channels: the compute channel acquires counters that the 3D channel
    releases in commands it submits later. A channel now waits at such an acquire, and goes on once
    another channel's commands have released it (A-238): about 2,000 waits in a run of 300 s, none
    of them for good.
15. The shared memory the title maps at an address it picks at random could fall where the
    graphics server later backs GPU-only memory; that range is now "in use" for a Switch 2 program
    (A-164). Before, most runs stopped there, each at a different point.
16. **Presenting**: `vi2` 351, 352 (A-235) and 229, `PresentClientBuffer` (A-236), which shows a
    native image from its memory, paced by the swap interval. The title presents with swap interval
    1, and with 2 from about frame 345.
17. The programs of a draw: the Ampere 3D class gives each program's whole address
    (`SET_PIPELINE_PROGRAM_ADDRESS_A/B`, A-239). Each draw's programs then start with a program
    header of type 1, version 4, which the translator does not read: the end of this milestone.

Two findings changed what other emulators do:

- A TIPC reply's data follows the header and the handles with no padding, and a command keeps its
  layout when it fails, the handle slots included (A-129). The 16-byte alignment the runtime used
  had every `vi2` count, partition and flag read as 0.
- NvRm takes its 2D, 3D, compute and inline-to-memory classes from its table of the T239's classes
  only when the characteristics' flags 40 to 43 (nvgpu's `SUPPORT_2D`, `_3D`, `_COMPUTE`, `_I2M`)
  are set; without them it asked for an object context of class 0, which the other emulators map
  onto a class of their own choosing (A-199).

## Measured

One run with the Vulkan renderer (`NX_RENDERER=vulkan NX_GPU_LENIENT=1 NX_MAX_FRAMES=900`), on one
RTX 5070 Ti:

- 900 frames presented, then the run ended by itself (exit 0).
- From frame 400 on, about 1,200 draws and 30 clears a frame. Every draw is refused at its programs
  (header version 4); the clears are made. 100 frames took about 20 s, almost all of it in refusing
  draws, so the rate says nothing of what the title will run at.
- The presented image is 3840x2160. No render target is drawn at its address, as the draws that
  would make it are refused, so there is no frame to dump and look at yet.
- What else the renderer cannot do yet, said once each: compute launches of QMD version 3 (sm_86
  programs), 2D copies from memory that no render target stands for, a clear of a color target
  of format 0, and draws with a tessellation program.

Without a renderer the run went 300 s without a stop.

## What M13 and M14 take on

- M13: the version 4 program header and the sm_86 instruction set, for the draws' and the compute
  launches' programs.
- M14: what the renderer then refuses: the 2D copies above, the formats, tessellation, the
  presented image (a render target at its address, or the pass that makes it), and frames dumped
  and looked at.
- `vi2` 353 (when its three values are turned off) and 228, and the 3D class's report-semaphore
  acquire across channels, when the title asks.

## Tests

The supervisor's, `nx2:gpu`'s and `vi2`'s answers, the TIPC reply layout, the Switch 2's
characteristics, TPC masks, SM mapping, engine list, unified address space, syncpoint shim (read
by a semaphore acquire), timeslice groups, channel runlists and preemption modes are unit tested,
as are the kernel's Switch 2 values and the CPU forms. New with this part: a macro's short read
(0, and counted); the macro code of two channels kept apart; a channel that waits at an acquire,
a later submission of it waiting behind, and both going on when another channel releases (each
word run once, the fences reached); the Ampere program address and the Maxwell region plus
offset; shared memory refused in the graphics server's range for a Switch 2 program only.
