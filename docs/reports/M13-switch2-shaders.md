# M13: sm_86 shaders

Status: complete. Started 2026-11-05, passed 2026-11-06: the title's draws and compute launches run
with their sm_86 programs translated to SPIR-V as they are (not lowered to Maxwell code first), and
the title reaches its title screen, drawn (frame 400 dumped and looked at), in a run stopped by
`NX_MAX_FRAMES` without a lost device.

The task (part of the owner's request of 2026-11-04, laid out in `docs/reports/M12-switch2-gpu-front.md`):
the version 4 program header and the sm_86 instruction set, for the draws' and the compute launches'
programs. References: Mesa's NAK (`sm70_encode.rs`, MIT licence) for the fields of each instruction,
NeXium (the owner's emulator) for the semantics that differ from Maxwell, and the NVN2 guide; each
fact used is in docs/ASSUMPTIONS.md (A-240 to A-258) with its source. The title is the owner's own
dump; this report gives aggregate numbers only and does not name it.

## What was built

| Part | What it does |
|------|--------------|
| `nx_shader::sm86` | The decoder of 128-bit sm_86 instructions (opcodes, guards, the operand forms and their exchanges, the uniform datapath), the end of a program, and the control flow graph (`BSSY`/`BSYNC` regions, `CALL`/`RET` as jumps, `BRX` jump tables from the constant buffers). |
| `nx_shader::translate::sm86` | The translation to SPIR-V: floats (`FADD` to `F2FP`, packed halves), integers (`IADD3`, `LEA`, `LOP3`, `SHF`, `IMAD` with `.WIDE` and `.X`, `PRMT`, ...), predicates, constant buffers (`LDC`, `ULDC`), attributes (`ALD`, `AST`, `IPA`), global, local and shared memory with the atomics, textures (`TEX`, `TLD`, `TLD4`, `TMML`, `TXQ`), surfaces (`SULD`, `SUST`, `SURED`), the warp (`VOTE`, `SHFL`, `REDUX`, `MATCH.ALL`, `FSWZADD`). |
| The scoreboard | A write of a variable-latency instruction lands when the code waits for its barrier (a wait mask, or `DEPBAR.LE SBn, count`), the register holding its old value until then (A-256). |
| Storage buffers | The base of every global access is followed back to the constant buffer word it is a pointer from; only those buffers are made reachable (A-258). |
| `nx_shader::sph` | The version 4 header (0x80 bytes, the first 0x50 as version 3 has them). |
| `nx_gpu` | QMD version 3 launches; the GOB sector order of the Switch 2; 2D copies made in guest memory. |
| `nx_vulkan` | Draws and launches with sm_86 programs; surfaces as storage images; a budget for textures too large for the device; the Switch 2's texture header address; 24-bit depth read from memory; the layers of a texture that changed uploaded alone; device memory and heap budgets in the report. |
| examples | `sm86` lists a program; `corpus` translates a directory of dumps and counts the reasons of those that do not translate; `replay` makes the pipeline of one vertex program away from the title and draws it once, the test `spirv-reduce` needs with `--crash-test`. |

## Two causes of a lost device

- **The driver's compiler (A-257).** The first draw of a particle vertex program lost the device
  after the operating system's 2 s watchdog. `VK_EXT_device_fault` named an invalid instruction.
  `replay` lost the device with every buffer zero and every texture a texel of zeros, so the data
  played no part. The same module through `spirv-cross` and `glslang` drew; that build lost the
  device again once only its saturating float-to-integer conversions were written as selects.
  `F2I` is now made as branches, with the conversion made only of a number inside the range, and the
  draw finishes. Variants of the select form crash the compiler in `vkCreateGraphicsPipelines`, not
  always the same ones from one run to the next.
- **Pinned guest memory (A-258).** Every mapping of the GPU address space was imported for global
  memory: by frame 200, 34.9 GiB in 5,504 runs (25 GiB of it distinct), with the device's heap at its
  budget, and a submission then lost the device. The storage buffers alone are 95 MiB at frame 400.

## Measured

- **Translation of the title's 1,191 program dumps:**

  | Stage | Translated and valid |
  |-------|----------------------|
  | Pixel | 625 of 627 |
  | Vertex | 426 of 452 |
  | Compute | 78 of 83 |
  | Geometry | 0 of 15 |
  | Tessellation | 0 of 14 |

  None translates to invalid SPIR-V. Those that do not translate:
  - geometry and tessellation, stages the renderer does not have yet;
  - 21 vertex programs that read or write the stage buffer of a mesh pipeline (`ISBERD`, `ISBEWR`);
  - 7 programs whose code has no end marker or whose jump table is in a constant buffer the dump does
    not have (it does at run time).
- **The title:** a run of 400 frames with `NX_GPU_LENIENT=1` ends at `NX_MAX_FRAMES` without a stop.
  - Frame 200, looked at: the loading screen, the illustrated world map with the logo and a spinner.
  - Frame 400, looked at: the title screen, with the logo, "Press L + R to start" and the version.
    Behind them the 3D scene is nearly black: tree outlines, a road and green specks show faintly.
- **Rate:** frames 300 to 400 took 49.7 s, about 2 frames a second.
  - 387,202 draws took 19.9 s, about 51 µs each.
  - 10,912 compute launches took 25.2 s, 13.9 s of it in the launches' constant buffer step.
- **Memory:** at frame 400 the renderer held 8,968 MiB in images and buffers. The device heap's
  budget on this machine was 5,003 MiB, because other programs held 12.5 GiB of the 16 GiB; the
  driver pages the rest.

## What M14 takes on

- What the renderer still refuses:
  - geometry and tessellation programs, and the vertex programs of mesh pipelines (`ISBERD`, `ISBEWR`);
  - surfaces of integer formats;
  - a clear of color format 0.
- The nearly black 3D scene of the title screen.
- The rate:
  - the descriptors and textures of a draw;
  - the constant buffer step of a launch;
  - 2D copies made in guest memory, 2,441 every 100 frames.
- The device memory budget.
- Past the title screen: the press of L and R with `NX_INPUT_SCRIPT`, then the menus, with frames
  dumped and looked at.

## Tests

New with this milestone:

- **Decoder:** the decoder's fields and forms (3 tests), the control flow of sm_86 code (4), and
  the Switch 2's GOB sector order.
- **On the device** (`translate_sm86`):
  - the scoreboard: a register keeps its old value until its barrier is waited for, and a counted
    wait lands the oldest writes first;
  - `F2I` to 8, 16 and 32 bits, signed and unsigned, of NaN, both infinities, the ends of each
    range and values between;
  - every form of `SHF`, at 12 amounts from 0 to past 64, against a 64-bit reference.
- **Unit tests:** surfaces made storage images, and global atomics.
- **QMD version 3:** the program address and constant buffers of a descriptor.
- **2D copies in guest memory:**
  - pixels moved between layouts;
  - the rectangle and clamp kept;
  - what the device cannot copy is copied in memory, and what no copy can make stops.
- **Textures:**
  - a texture too large for the device leaves out its largest levels;
  - 24-bit depth words read where their format has them;
  - a layer that changes is uploaded again while the others keep their texels.
- **The gate:** 1,113 tests pass, none skipped, one ignored.
