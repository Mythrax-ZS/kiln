# M12: the Switch 2 GPU front

Status: open. Started 2026-11-04.

The task (part of the owner's request of 2026-11-04): what a Switch 2 program's graphics need below
NVN2, at the driver level, until the title submits its first frames and they are traced. The plan
is in `docs/reports/M11-switch2-boot.md`. References: the NVN2 guide and the NVN guide the owner
supplied, NeXium (the owner's emulator) and Eglantine (a Switch 2 emulator derived from Eden,
GPL-3.0: used for what the services do, none of its code is taken); each fact used is in
docs/ASSUMPTIONS.md with its source.

## Where the title is

After M11 the title asked the applet for its graphics supervisor. It now gets it and boots its
graphics system through it:

1. `ISelfController` 2000 hands over the supervisor, `nx2:gsv`, a TIPC session (A-194).
2. `nn::gsv::BootWithRemoteMemory`: commands 4, 11 (`RegisterClientHandle`), 101, and the
   permission checks 1501 and 1502.
3. `nn::nvsrv::BootWithRemoteMemory`: command 1402 opens `nx2:nvsrv`, the GPU driver's session,
   which speaks CMIF and is answered by the existing `nvdrv` service (its command 13 already ran).
4. `nn::syncpt::sdk::InitializeEx`: the permission check 201, the syncpoint session (202, A-195),
   and the shared syncpoint pages (204, A-196), which a thread of the runtime keeps current every
   millisecond from the driver's syncpoints.
5. `nn::gs::sdk::BootNativeMemoryManagement`: `QueryGrantedFeatures` (115), and the permission
   checks 1101 and 1102.
6. **The first unimplemented call** is supervisor command 497 with one u64 (4): the request for
   `nx2:gpu`, the session that creates GPU memory and maps it into the 0x800_0000 to
   0xC0_0000_0000 range (the NVN2 guide, section 5.2).

## What is left in this milestone

- `nx2:gpu`: GPU memory (create, map, unmap, the private regions) on `nx_gpu`'s address spaces,
  and its completion events.
- The Switch 2's `nvdrv` ioctls: an Ampere `GetCharacteristics`, the NVN2 address-space partition
  (`AllocAsEx`), the syncpoint shim (0x410B), the engine table (0x471F), TSG and subcontext
  ioctls, submissions that increment syncpoints.
- `vi2` (presentation), and the supervisor's native memory and image commands (111 to 114, 423 to
  435, 1510).
- The command processor for the Ampere classes (3D 0xC797, compute 0xC7C0 with QMD version 3,
  copy 0xC7B5, GPFIFO 0xC56F), MME64 macros, methods split across GPFIFO entries.

## Tests so far

The supervisor's answers (`gsv.rs`), the kernel's Switch 2 values and the CPU forms are unit
tested; the gate passes.
