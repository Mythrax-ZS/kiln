# M11: booting a Switch 2 title

Status: complete. Date: 2026-11-04.

The task (owner request of 2026-11-04, to make the owner's Switch 2 title run): recognise a
Switch 2 (NX2) program and boot it, from its extracted ExeFS and RomFS, until the first call this
runtime does not implement, as M7 did for a Switch title. The title is the owner's own dump; this
report gives aggregate numbers only and does not name it. Its graphics are not part of this
milestone.

## Result

The title is recompiled whole and runs natively on the project's kernel layer, through its own
start-up:

1. **Recognised as a Switch 2 program** by its `main.npdm` (A-188): the ACID header's magic is
   0x300 bytes in, behind a 3072-bit signature and key, where a Switch NPDM has it at 0x200.
2. **Recompiled**: its five NSOs (`rtld`, `main`, `subsdk0`, `subsdk1`, `sdk`), 251,435 functions,
   4,064,105 blocks and 13,806,607 instructions, in 2,190 translation units; on the development
   machine (32 threads) lifting and compiling at `-O2` takes 184 s and linking 1 s, and the
   executable is 757 MB. 58 instructions are left to the interpreter (`PACGA`, A-189); 7,568 stop
   the run when reached, statically almost all of them zero words after calls that never return.
3. **Boots**: `rtld` relocates the other four modules and enters the SDK; the SDK reads the address
   space layout (with the Switch 2's `GetInfo` 29 and 30, A-190, and type 28, A-191), opens `sm:`
   over TIPC and the log service, maps physical memory, starts the applet (`appletOE` command 10,
   the application proxy's interfaces 20, 40 and 60, `IApplicationFunctions` 320 and 321,
   `ISelfController` 2400, A-192), asks the parental controls and the time service, sets a heap of
   8,226 MiB (A-190), and initialises the modules, locking the permission of each one's relocated
   read-only data (A-193).
4. **The first unimplemented call** is `ISelfController` command 2000, which
   `nn::gsv::BootWithRemoteMemory` sends for `nn::gs::sdk::BootGraphicsSystem`: the request for the
   graphics supervisor (`nx2:gsv`), through which a Switch 2 program reaches its
   GPU. The run stops there with status 101:

   ```
   kiln runtime: service `ISelfController` has no implementation of command 2000 (SVC at 0xc384500)
   ```

The run takes under a second: 177 supervisor calls (70 `SendSyncRequest`, 38 `GetInfo`, 34
`QueryMemory`, and others) carrying 53 IPC requests to `sm:`, `lm`, `appletOE`, `pctl`, `time:u`,
`aoc:u`, `ectx:aw` and `fsp-srv`.

## What was implemented

| Part | What it does |
|------|--------------|
| Platform (`npdm.rs`, `horizon/platform.rs`) | `Npdm::platform` tells NX from NX2 (A-188); the kernel takes the counter's rate, the heap region, the program's memory, the core count, the Switch 2's information types and SVC 0x80 from it (A-190). |
| Function discovery (`lift/discover.rs`) | Straight-line code that leaves its function's extent or runs into data now ends a block of its own; it panicked on this title, the first that has such code. |
| LSE atomics (`a64`, `lift`, `codegen`, `rt`) | `LD<op>`, `ST<op>`, `SWP`, `CAS` and `CASP` in the decoder, the interpreter, the lifter and generated code (`nx_atomic`, `nx_cas`, `nx_cas128`): one host atomic operation on the guest's page, aligned, a write for its faults (A-189). |
| `LDAPR` | An `LDAR`. |
| Pointer authentication | Modelled as the kernel leaving the instruction and data keys off: `PAC*`/`AUT*` are no-ops, the authenticated branches and returns go to the register, `LDRAA`/`LDRAB` are plain loads, `XPACI`/`XPACD` strip as the pseudocode does; `PACGA` is the pseudocode's `ComputePAC` (QARMA5, `a64/pac.rs`) with the generic key zero, run by the interpreter (A-189). |
| Counter | `CNTFRQ_EL0` comes from the runtime (`nx_counter_frequency`), and the counter runs at the platform's rate, 31.25 MHz for NX2 (A-190). |
| Kernel | `GetInfo` 28 to 33, a 10 GiB heap region and `SetHeapSize` past 4 GiB, 9,809 MiB of memory, six cores, `SetGpuMemoryAttribute` (A-190, A-191), the permission lock of `SetMemoryAttribute` (A-193). |
| Services | The applet commands of A-192. |
| `kiln analyze` | Counts the interpreter fallbacks and the stops by reason, not only the first hundred of each. |

## What the tests cover

- **Decoder**: every form above, the unallocated neighbours (odd `CASP` registers, `XPACI` with a
  base, a 32-bit `PACIA`, `LDRAA` of 32 bits, `BRAAZ` with a modifier field).
- **Interpreter**: each atomic operation at each size against values that wrap and that sign
  decides, the zero-register alias, alignment and permission faults, `CAS`/`CASP` success and
  failure, pointer authentication with the keys off, `XPAC` on both halves, `PACGA` values.
- **QARMA5**: three vectors with random keys and two with the zero key from Unicorn 2.1.4's `max`
  core (QEMU's implementation); 300 random cases agreed when it was written.
- **Lifted against the interpreter**: five new encoding groups of 869 cases each, in the IR
  evaluator and natively through clang, and a control-flow program of atomics and compares that
  succeed at the stack pointer, ending in `RETAA`.
- **Unicorn**: the Switch 2 forms are compared on the `max` core set up as A-189 models the
  console (60,000 cases: 375 accepted as U-12, QEMU's unsigned narrow signed maximum, and 3 as
  U-13, QEMU's `XPAC` without keys, each only when Unicorn's value is exactly that behaviour's);
  the random-word test on the Cortex-A57 model leaves them out (650 of 300,000 words).
- **Runtime**: four threads adding 20,000 times each to one word through `nx_atomic` lose nothing;
  `CAS`, `CASP` and the narrow signed maximum through the C ABI.
- **Kernel**: a Switch 2 process gets the Switch 2 answers (the regions, the heap of more than
  4 GiB, `GetInfo` 29 to 33, SVC 0x80, core 5) and a Switch process does not.
- The gate (`cargo fmt --all --check`, `cargo lint`, `cargo test`) passes; nothing is skipped on the
  development machine (clang, Unicorn and a Vulkan device are present).

## What is not verified

- The console's kernel: whether it enables the pointer authentication keys (no program that only
  signs and checks its own pointers can tell), the real heap region and remote region, and the
  counter's rate. The values are NeXium's or chosen so that the title's requests fit (A-190).
- Half-precision arithmetic (FEAT_FP16) is not implemented; static analysis finds three such
  words in the title, and a stop names any that runs.
- Nothing past the request for the graphics supervisor ran: no GPU, display, input, audio or file
  access beyond what start-up does.

## Next: the Switch 2 GPU front (proposal)

The title is at the point where the Switch title of M7 was when it asked for `nvdrv`. A Switch 2
program compiles NVN2 into its own `sdk` module, so the recompiled code issues the GPU's commands
itself, and the work is at the driver level, as in M8 and as NeXium (the owner's emulator, MIT)
does it. In order, with NeXium's implementation as the reference:

1. **M12, the GPU front for NX2**: the graphics supervisor and its sessions (`nx2:syncpt` with its
   shared page, `nx2:gpu` memory, `vi2` presentation, `nx2:nvsrv` for the `nvdrv` nodes), the
   Switch 2's `nvdrv` ioctls (an Ampere `GetCharacteristics`, the NVN2 address-space partition,
   the syncpoint shim, the engine table), and a command processor for the Ampere classes (3D
   0xC797, compute 0xC7C0 with QMD version 3, copy 0xC7B5, GPFIFO 0xC56F), MME64 macros and
   methods split across GPFIFO entries. Exit: the title submits its first frames and they are
   traced.
2. **M13, sm_86 shaders**: the Ampere instruction set (128-bit instructions, uniform registers,
   convergence barriers, the version 4 header) translated to SPIR-V, compared on the GPU.
3. **M14, the renderer for NX2**: the swapped GOB sector order, Ampere texture headers, render
   targets sampled through depth views, task and mesh stages, the video decoder's NX2 path. Exit:
   frames dumped and looked at.

The owner's request of 2026-11-04 covers all three.
