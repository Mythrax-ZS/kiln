# M14: the renderer's Switch 2 path

Status: complete. Started and passed 2026-11-06: the title runs from its title screen through its
menus and its cup's video into a race, and the race is drawn (frames dumped and looked at), in runs
of 3,700 frames stopped by `NX_MAX_FRAMES` without a lost device. It runs at about 5 frames a second
there, not at full speed.

The task (the last part of the owner's request of 2026-11-04, laid out in
`docs/reports/M11-switch2-boot.md`): what the renderer lacked for the Switch 2's draws once their
sm_86 programs translate (M13). References:
NeXium (the owner's emulator, for field layouts), NVIDIA's class headers, and the title's own SDK
where they say nothing; each fact used is in docs/ASSUMPTIONS.md (A-259 to A-279) with its source.
The title is the owner's own dump; this report gives aggregate numbers only and does not name it.

## What was built

| Part | What it does |
|------|--------------|
| Video | `nx2:gsv` and the decoder service of `nn::codec::video` decode VP9 streams into the title's images (A-261 to A-264); the library's context ends (A-269); `NX_VDEC_DUMP` writes the frames. |
| Copy engine | The copy classes' `LAUNCH_DMA`: pitch and block-linear copies with their semaphore (A-268). |
| Render targets | A target keeps its block depth: the slices of a 3D texture are drawn as targets (A-266). A target of format 0 is off, and one between two that are on is a hole in the attachments (A-273). |
| Task and mesh pipelines | The mesh registers of the 3D class, the stage buffer (`ISBERD`, `ISBEWR`) and the task and mesh stages of `VK_EXT_mesh_shader`; the road and the scenery the title draws with them (A-270). |
| Translation | `IMAD.HI` (A-271), surfaces of integer formats (A-272), `HSETP2` (A-276), and which programs write global memory. |
| Depth | Depth targets copied into the depth textures whose layers or parts they are: the shadow map's cascades sampled as one array (A-275). |
| Compute launches | Waited for a run at a time, not one by one; their surfaces copied out for the host and put into memory when something needs them, their QMD releases and the GPU front's semaphore releases made after them, and an acquire compared with what the releases will leave (A-274). |
| Speed | Idle parts of tracked memory are not harvested (A-277), surface images and staging buffers are used again (A-278), a write of the renderer has only the layers it reaches read again (A-279), a target's layer is found from its offset, and the programs a draw uses are counted only for the trace. |
| Diagnosis | `VK_EXT_device_fault` is on for every run, so a lost device prints its fault; the report of every 100 frames counts the commands refused by kind, the launches by what they write, and why they were waited for. |

## Three causes of a wrong picture

- **The shadow map read from memory.** The race's road was black once its shadow pass was drawn
  (the cascades are depth-only passes, refused until A-273). The pass samples the three cascades as
  one depth array, and no depth target reached a texture: the array was what memory held. A-275
  copies the cascades into the array's image; the road, its markings and the cast shadows came back,
  and so did the bottom rows of the screen, which a texture a few texels narrower than its depth
  target had left dark.
- **Draws refused at a hole.** Commands whose color target 0 is off and target 1 on stopped at the
  attachments (16 in the first 250 frames); they are drawn with a hole now (A-273).
- **Programs refused.** `HSETP2` in a vertex program refused 808 draws of a race (A-276).

## Measured

- **Translation of the title's 2,031 program dumps** (the dumps of M13 and those of a race):

  | Stage | Translated and valid |
  |-------|----------------------|
  | Pixel | 1,123 of 1,125 |
  | Vertex | 733 of 756 |
  | Compute | 114 of 120 |
  | Tessellation | 14 of 14 |
  | Geometry | 0 of 16 |

  None translates to invalid SPIR-V. The tessellation dumps are the mesh programs of mesh pipelines
  and translate as such; the geometry dumps are their headers, which say what each primitive carries
  and have no code to translate. The rest fail only away from the title: 7 dumps end before the
  program's end marker, and 23 programs' jump tables are in a constant buffer the dump does not have
  (at run time they translate).
- **The title:** runs of 3,700 frames with `NX_GPU_LENIENT=1` and the input script of the menus end
  at `NX_MAX_FRAMES`. The last refused 19 commands in all: 18 draws that sample a 3D texture with a
  2D instruction, and one surface of a format the renderer does not know.
  - Frame 3550, looked at: the race's start, the kart from behind on the road with its lane
    markings and the starting grid, the other racers ahead, the stands and the gantry, cast shadows,
    the coin and lap counters and the map of the course.
  - Frame 3700, looked at: the race under way, the racers gone ahead, the position shown as 24th.
  - The gantry and the buildings beside the road are nearly black; whether they should be is not
    known.
- **No lost device** in the 9 runs of the title since the hole of A-273. Before it a sporadic loss
  while loading (frames 200 to 300, at a compute launch) came in 3 runs, and not in the 2 runs made
  with `NX_DEBUG_CHECKPOINTS`; draws refused at the hole were the change between them.
  `VK_EXT_device_fault` now prints the fault of any loss.
- **Rate in the race** (frames 3,500 to 3,700, about 4,600 draws and 125 compute launches a frame):

  | | Before the work on speed | At the end |
  |-|-|-|
  | 100 frames | 23.7 to 25.5 s | 19.7 to 21.9 s |
  | of it in the renderer | 19.7 to 21.1 s | 13.4 to 16.1 s |
  | waits for compute launches, a frame | 125 | 32 |
  | memory hashed, a frame | 187 MB | 15 MB |

  About 5 frames a second. The waits left are launches that read what the launch before them wrote
  (18 a frame), the GPU front (12) and draws (2); each drains the device's queue. A draw costs 22 to
  29 µs of the renderer's thread.
- **Memory:** at frame 3,700 the renderer held 10,125 MiB in images and buffers, against a device
  heap budget of 4,990 MiB on this machine (other programs hold about 12.5 GiB of the 16); the driver
  pages the rest.

## What comes next

- The rate, toward the 60 frames a second the title asks for:
  - the launches that read what the launch before them wrote: their surfaces on the device instead of
    copied through memory;
  - the cost of a draw: texture binding, the pass, the descriptors;
  - the GPU front's method and macro interpreter, about a fifth of the thread;
  - the harvest of tracked memory (8 ms a frame) and the walk of host pages per question.
- 3D textures sampled with 2D instructions, and `HSET2`.
- Whether the gantry's darkness is the title's.

## Tests

New with this milestone:

- **Translation:** the stage buffer of task and mesh programs (2), `IMAD.HI` on the device, surfaces
  of integer formats, `HSETP2` on the device (lanes alone and together), which programs write global
  memory (Maxwell and sm_86).
- **The GPU front:** the mesh registers of the 3D class; the copy engine's copies; QMD releases
  handed back with their launch; an acquire of a word still to be released waits for the release,
  and one compares what held releases will leave without waiting.
- **The renderer** on the device: color targets of format 0 cleared and drawn around (the draw was
  also run once under the validation layer, which found nothing but the unused write); a launch's
  release made only once the launch is waited for; a launch that reads what the launch before it
  stores waits for it. Unit tests: a depth target found as a layer of a depth array, and releases
  folded into a word.
- **Video:** the decoder service's requests, and a VP9 key frame decoded into an image.
- **Memory:** a part of tracked memory nobody asks about is let go, and its writes are still found.
- **The gate:** 1,140 tests pass, none skipped, one ignored.
