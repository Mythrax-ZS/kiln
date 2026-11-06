# The Switch 2 title after M14: what it uses, and its rate

Measured 2026-11-06, at commit 18dee73. The title is the owner's own dump; this report gives
aggregate numbers only and does not name it. Each run goes from the title screen through the menus
(an input script) and the cup's video into a race, and stops at `NX_MAX_FRAMES`.

The title's own graphics library (its NVN2 driver code) runs recompiled like the rest of its code.
What it asks of the system goes through the services below, and what it asks of the GPU goes
through the driver's channels to the GPU front. Coverage is therefore measured at those two
boundaries: which requests come and whether each is served, and which methods, macros and
programs come and whether each is carried out.

## NVN2

- **733 entry points named.** The read-only data of the title's `main` holds the names of 733
  distinct NVN2 entry points (`nvn…`). That is the API the title fetches by name through
  `nvnBootstrapLoader`.
- **Served by the title's own driver.** The loader is exported by the SDK's module (the module
  also holds `nn::os` and the other SDK libraries), and that driver runs recompiled like the rest
  of the title.
- **Reached in a run:** `NX_CENSUS` counts the targets of indirect branches, which is how the
  title calls NVN2. In a run of 3,500 frames, from the menus into the race:
  - 306 distinct functions of the driver's code were entered through pointers, 48,969,462 times
    in all (about 11% of the 458,889,924 indirect branches). The driver's code is taken to be the
    targets whose nearest export before them is an `nvn…` symbol.
  - That count covers the entry points the title calls and the driver's own calls through
    pointers.
- **Called by the title itself:** with `NX_CENSUS_FROM` set to the range of the title's own
  modules (`main`, `subsdk0` and `subsdk1`), the census also counts the targets whose return
  address is in the title's own code. In another run to the same point, the census listed every
  target it counted:
  - 234 distinct driver functions were called from those modules, 52,228,872 times. That is 234
    of the 733 entry points the title names, about 32%.
  - 45 of the 234 were called 10 times or fewer (the setting up).
- **By name:** the driver's `nvnDeviceGetProcAddress` (traced with `kiln disasm`) finds a name by
  binary search over 731 sorted names. With the title's device, it gives a different function for
  each name than without one, and those are the functions the title calls. Asking it for every
  name once the title had a device, then matching the census of a run into the race:
  - **246 NVN2 functions** were called from the title's code, 53,266,180 times, 39 of them 10
    times or fewer. This count is exact, by name, and replaces the estimate of 234 above.
  - **By class:**

    | Class | Functions |
    |-------|-----------|
    | `CommandBuffer` | 72 |
    | `Texture` | 48 |
    | `Sampler` | 22 |
    | `Queue` | 15 |
    | `MemoryPool` | 14 |
    | `Device` | 11 |
    | `Window` | 10 |
    | Others (programs, buffers, render state) | 54 |

    Each class counts its builders, views and pools with it.

  - **Most called:**
    - `nvnCommandBufferBindTexture`: 6,088,354
    - `nvnDeviceGetTextureHandle`: 5,801,644
    - `nvnChannelMaskStateSetChannelMask`: 3,513,941
    - `nvnBufferMap`: 2,151,952
  - These 246 functions are what an implementation of NVN2 in place of the driver would have to
    provide for this title, as far as the race.
- What those calls produce is counted below, at the services and the GPU front.

## The system

From one run with `NX_TRACE_SVC=1` (into the menus, past frame 500; the trace slows the run too
much to go further):

- **33 different supervisor calls**, all served. The most frequent: `SleepThread` (265,930),
  `WaitForAddress` (149,527), `WaitSynchronization` (82,741), `SignalToAddress` (74,500) and
  `SendSyncRequest` (49,149).
- **44 service interfaces and 218 of their commands**, all served. Seven of the interfaces are
  the Switch 2's own, with 88 of the commands:

  | Service | Commands used |
  |---------|---------------|
  | `nx2:gsv` | 33 |
  | `nx2:vi2` | 20 |
  | `nx2:gpu` | 16 |
  | `nx2:nvsrv` | 7 |
  | `nx2:syncpt` | 5 |
  | `nx2:chat:gsv` | 4 |
  | `nx2:camera` | 3 |

No request stopped a run: an unknown supervisor call or command would have stopped it and said its
number.

## The GPU

From one run to frame 3,600 with `NX_TRACE_GPU_WINDOW=3400:3430:1` (the counts of methods, macros
and programs are of those 31 frames of the race; the others are of the whole run):

- **The whole run:** 14,623 submissions, 6,563,321 pushbuffer entries, 363,990,930 words and
  306,477,241 method writes.
- **31 frames of the race:**
  - 739 different methods are written: 507 of the 3D class, 117 of the compute class and 10 of
    the host's.
  - 42 different macros are called 1,736,734 times. The most frequent methods are macro calls
    and constant buffer loads.
  - 143,047 draws use 562 different programs. Most use a vertex and a pixel program. 2,263 are
    task and mesh draws with four stages, and 186 more are mesh draws without a task.
- **The renderer refused about 16 commands a run**, with `NX_GPU_LENIENT=1`:
  - 15 draws sample a 3D texture with a 2D instruction;
  - 1 surface has format 0.

**Shader translation:** the 1,191 program dumps of a race run (`nx_shader`'s `corpus` example):

| Stage | Translated and valid |
|-------|----------------------|
| Pixel | 625 of 627 |
| Vertex | 447 of 452 |
| Compute | 78 of 83 |
| Tessellation (mesh programs) | 14 of 14 |
| Geometry (headers of mesh pipelines) | 0 of 15 |

None translates to invalid SPIR-V. The geometry dumps have no code to translate.

The rest fail only away from the title, and at run time they translate:

- 6 dumps end before the program's end marker;
- 6 programs' jump tables are in a constant buffer that the dump does not have.

## The rate

The race runs at about 8 to 9.6 frames a second. At the end of M14 it was about 5. The title asks
for 60.

| 100 frames of the race | End of M14 | Now |
|------------------------|------------|-----|
| Wall time | 19.7 to 21.9 s | 10.4 to 13 s |
| Renderer | 13.4 to 16.1 s | about 7 to 8 s |
| Render passes begun | about 135,000 | about 13,000 |

What changed:

- **Fewer draws:** a series of instances is folded into one draw (A-281).
- **Less texture work:**
  - texture bindings are kept for the rest of a frame (A-282);
  - layers are hashed in parts of 1 MiB (A-279);
  - memory is tracked in parts of 2 MB (A-277);
  - unchanged textures are found with one question to the tracking;
  - ASTC is decoded on the device (A-286).
- **Fewer waits for the device:**
  - draws and clears wait only when they read on the host what launches wrote (A-284);
  - textures sample a launch's image directly (A-283) and the images kept on the device (A-285).
- **A cheaper GPU front:**
  - plain register writes are stored at once;
  - methods are counted only when traced;
  - the last GPU mapping is remembered.

Where 100 frames of the race go now (about 272,000 draws and 12,500 compute launches; the run
varies by about 15%):

- **The renderer: about 8 s.**
  - Draws: about 5.5 s. Of that, textures take 2.3 s, descriptors 1.2 s (0.4 s of it copying
    7 GB of constant buffers) and passes 0.7 s.
  - Compute launches: about 1.2 s, plus about 2.2 s waiting for the device where the host must
    read what the device wrote.
- **The GPU front's interpreter of methods and macros: about 3 s.** That is about 28 million
  methods and 5.6 million macro calls.
- **The harvest of tracked memory: about 5 ms a frame.**

### What limits the rate (2026-11-07)

Measured in the race, at 223229c:

- **One thread sets the pace.** The thread that runs the GPU front and the renderer is busy
  about 93% of the time. The device is about 20% busy, and the title's own threads mostly wait for
  the fences of their submissions. Replacing the title's NVN2 driver with native code would not
  raise the rate: that code runs on the title's threads, which are not the limit.
- **The loop is bound by latency.** The title waits for the fence of each submission before it
  goes far with the next. A fence can be given only once the commands before it are recorded,
  so the front and the renderer cannot overlap across submissions.
- **The renderer on a thread of its own** (`NX_GPU_SPLIT=1`, A-288) overlaps the front with the
  renderer within a submission:
  - The front waits for the renderer's thread about 45% of the time; the renderer's thread is
    about 65% busy.
  - The waits are where the front reads memory the renderer's work may change (macro reads of
    memory, compute descriptors, acquires that stop a channel), and at the ends of submissions.
- **Constant buffers** are copied once per batch (A-289): 93% of the bytes repeat a copy made
  earlier in the batch, and the descriptors of the draws take 0.74 to 0.83 s instead of 1.16 to
  1.28 s per 100 frames.

Rate, without `NX_TRACE_PERF` (from the times of the frames dumped at 3,500 and 4,000):

| Mode | 100 frames of the race | Frames a second |
|------|------------------------|-----------------|
| Default | 10.5 s | about 9.5 |
| `NX_GPU_SPLIT=1` | 10.2 s | about 9.8 |

Where 100 frames of the race go now, on that one thread (with `NX_TRACE_PERF`):

| Part | Time |
|------|------|
| The GPU front (methods, macros) | about 4 s |
| Draws: textures | 1.1 to 1.6 s |
| Draws: descriptors | 0.74 to 0.83 s |
| Draws: the pass | about 0.6 s |
| Draws: vertices and recording | about 0.8 s |
| Compute launches | about 1.1 s |
| Settles of launches (waiting for the device, surfaces put into memory) | 1.7 to 1.9 s |
| The harvest of write tracking | about 0.45 s |

**Device memory ran out as the race loaded.** About 3,300 textures held about 12.5 GB, 7.6 to
8 GB of it ASTC textures decoded to four bytes a texel, and 70 to 90 draws a run were skipped when
an image could not be allocated. ASTC textures are now encoded as BC7 on the device (A-291): they
hold about 4.7 GB, all the textures 9.4 GB, and no allocation failed in a run into the race.

## Open

- **The road was sometimes black.** In about a quarter of the runs that reached the race (7 of
  about 30 on 2026-11-06), the road surface was black from the race's start to the end of the
  run, while its markings, the curbs and the rest of the scene were drawn. It was seen before the
  work on the rate too.
  - The likely cause: a macro's `MME_DMA_READ_FIFOED` (how an indirect draw takes its
    arguments) read memory without waiting for the work the renderer still held. In the race
    those reads cover words that a held release writes, so they took the old value. Fixed in
    dd0f378 (A-287).
  - Since then, 4 runs of 4 draw the road, with its lane lines (frame 3,500 looked at), at 10.5
    to 11.1 s per 100 frames of the race. With black in one run of four before, that is not yet
    proof.
- **The rate:**
  - the waits for the device (the host reading what launches wrote, and the ends of
    submissions);
  - the cost of a draw's textures and descriptors;
  - the GPU front's interpreter on the renderer's thread.
- **The two refused kinds of command.**
