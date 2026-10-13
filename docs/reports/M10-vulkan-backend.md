# M10: the Vulkan backend (`nx_vulkan` renderer)

Status: the title's first screens and its opening cutscene (after the first menu) are rendered,
shown in a window, and the keyboard reaches the title; the milestone is open at speed (about 15
frames a second in the cutscene), at what comes after the cutscene, at an early stop that happens
in about one run in 30, and at sound (see "What is not done"). Date: 2026-10-14.

The task (owner approval of 2026-10-13, "begin the shader translation and Vulkan backend"): turn
what the command processor of `nx_gpu` decodes into pictures with the translated programs, on a
Vulkan device, and look at them. The title is the owner's own dump; this report gives aggregate
numbers and says in general terms what the pictures show, and does not name it.

**What a picture shows was checked by looking at it** (frames dumped as PNG and viewed), not from
counts. Counts alone had said "all white" for a frame that was in fact empty; see the bugs below.

## Result

- **The title's first screens render.** Over 1,500 presented frames the renderer ran 380,376
  draws and 1,499 clears and **none failed or was skipped** (the first run of the same boot had
  four draws a frame stopping on a texture format). The first frame is black; around frame 100 a
  full-screen introduction screen fades in (photographs, a logo, three small pictures and lines of
  text in two colors); frames 450 to 1,350 are that screen unchanged; at 1,500 a menu screen with
  large artwork and text is fading in over it. All text, icons and photographs are drawn.
- **Speed (a measurement of this boot only):** 1,500 frames take 23.0 s of wall clock with the
  renderer and 14.5 s without it (the command processor alone), on an NVIDIA RTX 5070 Ti with the
  title unthrottled, so about 65 frames a second including start-up. That is a number for these
  screens (370,000 of the draws are text glyphs, one draw each), not for a game being played.
- **What the title needed** to get there: 10 different programs (5 vertex, 5 pixel; 46 different
  instructions between them), three texture formats (BC4 for the glyph atlas, ASTC 4x4 for the
  pictures, and its own render target, 1280 x 720, for the last pass), instanced draws, one global
  memory program, alpha blending, 1600 x 900 swapchain-size targets presented with a 1280 x 720
  crop.
- **A window, and the keyboard.** With `NX_WINDOW=1` the frames appear in a window (letterboxed
  to its shape, vertical sync), and keys reach the title through `hid` (A-104, A-105). Looked at in
  captures of the window: the menu the title stops at moved its selection down and up again with
  the keys that stand for the directional pad, and again with the keys that stand for the left
  stick (a button appears in the samples within 5 ms of its key; the picture followed within
  300 ms); confirming it made the screen fade and the title's introduction text appear. (After
  that the picture stayed black when this was first tried: see the next bullet.) The window was
  resized to a wide, a tall and a normal shape (equal black bars on the two sides that need them)
  and minimized and restored (the title kept running and answered the next key).
- **The opening cutscene renders.** After the first menu is confirmed the title shows a story
  card (a line of white text on black) and then its opening cutscene: characters lit and in
  color, a flame effect, motion blur and subtitle text, upright and with the faces culled the right
  way round. This was looked at in dumped frames (every 100th frame of one run from 3,000 to
  5,400, the passes of one frame from the shadow maps to the final copy, and every 500th frame of
  a run to frame 13,000, whose last frames show two characters and a line of subtitle), after the
  two fixes below that turned the picture from black into this. A scene frame has 555 to 620 draws (shadow,
  depth prepass, base, mask and quad passes into a 640 x 720 HDR target, a luminance pyramid, a
  composite into 800 x 900 and a pass to 1280 x 720, then the text) and takes about 65 ms: about
  15 frames a second (it was 6 before the changes below), most of the time in the renderer's draw
  path (see "What is not done").
  The scene is drawn at half the width of the picture (a 640 wide HDR target for 1280 wide
  output), so the edges are blocky; whether that is the title's own dynamic resolution reacting
  to the slow frames has not been checked.
- **No sound, and no gamepad was tried** (the conversion of XInput states is unit-tested; no
  controller was connected). Nothing here is a claim that the title is playable.

## What was built

| Part | What it does |
|------|--------------|
| `nx_gpu::render` | The `Backend` trait (draw, clear, finish, present) and the `Scene` a draw is given: registers, constant buffers, memory, address space. The command processor calls it as commands arrive. |
| `nx_gpu::draw_state` | Decodes the state of a draw from the registers: targets, viewports, scissors, raster, depth and stencil, blend, vertex streams and attributes, index buffer, texture pools, clip control. |
| `nx_vulkan::render` | The renderer. Dynamic rendering with the attachments the registers name; pipelines cached by programs, vertex layout, topology, formats, blend and layout; descriptor sets per draw from per-frame pools (constant buffers as read-only storage buffers, textures as combined image samplers); an upload ring for vertex, index and constant data; the commands of a GPFIFO submission run when it ends. |
| `render::textures` | Texture and sampler headers, block-linear detiling with mip levels, the Vulkan formats of the header codes, host decoding of ASTC, a cache that compares texture memory with its hash once a frame. |
| `render::targets` | Render targets found again by the guest address of their memory, zero-filled when made, read back as pictures, a render target read as a texture without a copy. |
| `render::global` | Global memory: the table of GPU address ranges, and the guest's own memory imported into the device in place (`VK_EXT_external_memory_host`). |
| `astc` | An ASTC decoder (low dynamic range profile, all 14 footprints, multi-partition, dual plane, void extent), threaded for large textures. |
| `render::compute`, `nx_gpu::{compute, i2m}` | Compute launches (A-110): the descriptor of the launch decoded (grid, workgroup, constant buffers, program), the program translated as a kernel and run when it is signalled, its textures taken from constant buffer 2 of the descriptor; and the inline-to-memory engine (A-108), which the title uses to upload those descriptors and their parameters. |
| `render::debug` | The switches for looking: `NX_TRACE_DRAWS=<frame>`, `NX_TRACE_STATE`, `NX_TRACE_TEXTURES`, `NX_TRACE_CBUFS`, `NX_DUMP_FRAMES` (with `NX_DUMP_EVERY`, `NX_DUMP_COUNT`, `NX_DUMP_TARGETS`), `NX_DUMP_DRAWS=<dir>` with `NX_DUMP_FROM` (a picture of the target after each command of the traced frame), `NX_DEBUG_NO_CULL`, `NX_DEBUG_FULL_VIEWPORT`, `NX_DEBUG_SOLID_PIXEL`, `NX_SPIRV_CHECK`, `NX_VULKAN_VALIDATE`, and those the black-scene hunt added (`NX_DUMP_BUSY` and `NX_DUMP_PASSES`: a picture of every pass of the first busy frame; `NX_TRACE_REGISTER`, `NX_TRACE_MACRO`, `NX_TRACE_TARGETS`, `NX_DEBUG_FLIP_FRONT`, `NX_DEBUG_NO_DEPTH_TEST`, `NX_DEBUG_SYNC_DRAWS`; and those the hunt for a lost device added: `NX_DEBUG_CHECKPOINTS`, `NX_DEBUG_SYNC_FROM`, `NX_DEBUG_SKIP_PROGRAMS`, `NX_DEBUG_DISCARD_PROGRAMS`, `NX_DEBUG_TRIVIAL_VERTEX`, `NX_DEBUG_SOLID_ONLY`, `NX_SPIRV_OPT`, `NX_DEBUG_DISPATCH_LOOP`; CLAUDE.md lists them). With `NX_TRACE_GPU` the renderer says every 100 frames where the time went (draw, clear, finish, present, compute, and the steps of a draw). `NX_RENDERER=vulkan` turns the renderer on in a produced program. |
| `window`, `render::display`, `render::blit` | A winit window on a thread of its own (feature `window`, on by default in the runtime): the raw handles for a surface, its size, close-to-exit, and the keys and focus handed to an `InputSink`. The display makes the swapchain (sRGB when the buffers are, FIFO unless `NX_PRESENT_MODE` says otherwise), and the blit copies the shown crop of a buffer into the swapchain image, fitted with black bars. |
| `nx_runtime` `horizon::input`, `horizon::gamepad` | Keys by physical name, the key map (standard, or `NX_KEYMAP`), stick directions from keys, the XInput gamepad (loaded at run time, positional face buttons, dead zones), merging, and the trace (`NX_TRACE_KEYS`). `hid` writes the merged state into its samples every 5 ms. |

## What the tests cover

- **26 renderer tests on the device** (`tests/render_*.rs`): clears (the whole target, a rectangle,
  BGRA targets, refusals), flat and blended draws, indexed quads, triangle strips and fans
  rewritten for Vulkan, the viewport (a negative scale y puts clip y up in the picture, the front
  face is judged on the picture for both signs of the scale, a viewport without width draws
  nothing; the shared scene leaves `SET_VIEWPORT_SCALE_OFFSET` 0 as the title's macro does, A-112),
  textured draws (block-linear texels, the header swizzle, compressed blocks, ASTC blocks decoded
  on the host, once-a-frame comparison with memory, a handle that names no header), a render target
  sampled as a texture (with the header's swizzle, drawn to again afterwards, not read while it is
  being drawn to), a texture made from memory that a render target covers (the write-back of A-111:
  the sizes differ, so the texture cannot be the target's picture; it sees what was drawn, and
  again after the target is drawn to once more), a texture one of whose levels is a render
  target on the device (the target is wider than the texture, so its rows are the texture's
  with padding: the picture is right after a draw and after the target is drawn to again, and
  memory is not touched; A-113), global memory read and written in place by a vertex program,
  and the instance counter of each draw. Each picture is compared pixel by pixel
  with the expected one.
- **The translator on the device** (`tests/translate_*.rs`, 54 tests): programs assembled from
  words run as compute programs and compared with Rust arithmetic, now including the texture
  instructions (`TEXS` of cube, volume, depth-compare and array images and with packed halves,
  `TEX`, `TLD4S`, `TLD4` and `TXQ`), the condition code flags (A-107), 64-bit sums by `LEA`, and
  two tests of kernels (each invocation knows its place in its workgroup and the workgroup's in
  the grid, writing through global memory; a kernel cannot read a vertex attribute). `TMML` and
  what the translator refuses (a handle that was computed, a handle loaded in another block,
  derivatives outside a pixel program) are tested in `nx_shader/tests/tmml.rs` (5 tests, the
  SPIR-V validated), not run on the device. How control flow is translated (the loop-free form for
  jumps that only go forward, the loop for a jump back, the forced loop, a kill in a loop-free
  program) is tested in `nx_shader/tests/control_flow.rs` (4 tests, the SPIR-V validated), and the
  device test `the_loop_free_form_agrees_with_the_loop_on_every_state` runs both forms of one
  program for 256 random states.
- **The command processor** (`nx_gpu`, 159 unit tests): the hash of GPU memory taken in place
  (the same by address and by copy, changed by a change of any byte, covering what is mapped and
  saying how much that is), the inline-to-memory engine (lines a pitch
  apart, a launch with nothing to send, the refusal of block-linear and semaphore launches, data
  written once all of it has come, an upload to unmapped memory stopping the run; A-108), the
  decoding of a compute descriptor (grid, workgroup, constant buffers) and its refusals (an empty
  grid, a workgroup of more than 1,024 invocations), and a launch that reaches the backend with
  what its descriptor says or stops the run when the descriptor is not mapped (A-110).
- **The ASTC decoder against ARM's reference decoder** (`tests/astc_reference.rs`, needs `astcenc`
  named by `RECOMP_ASTCENC`; the decoder was written from the Khronos specification and not from
  the reference code): 16,384 random blocks for each of the 14 footprints in linear mode and for
  five of them in sRGB mode, 45,197 legal blocks in all, **0 texels differ and every block the
  reference refuses is refused here** (A-099). A second test builds dual-plane blocks by hand and
  a third every low dynamic range endpoint mode on its own (512 blocks each). Without the tool the
  tests print `SKIPPED`.
- **Unit tests** of the decoder (trit and quint decoding of all 243 and 125 combinations, the
  evenness of every unquantization table, void extent, a hand-built block mixed by its weights,
  blue contraction, sRGB, the reserved encodings, threads against one thread), of the state
  decode (the clip control register field by field), of the formats, mip sizes and sampler
  headers, and of which layer and level of a texture a render target is (a mip chain and the
  layers of an array, rows padded to a whole GOB, strips a block of rows down a level, and what
  is refused: another format, block height, tiling, layer count, an address between levels).
- **Mutation checks done by hand**: with the render-target path switched off the two matching
  tests fail with the memory's zeros; with the instance counter fixed at 0 the instance test fails
  at the second draw.
- **The window and its blit**: three unit tests of the blit (the shown part of a buffer, the fitted
  rectangle with its bars for several shapes, and one on the device that blits a cropped, fitted
  picture and reads it back pixel by pixel), the packing of the window's size and the names of the
  keys. The whole `nx_vulkan` suite was run with the validation layer on (`NX_VULKAN_VALIDATE=1`),
  which found three mistakes in the new code and its tests (the depth offsets of a blit region, an image view
  of an image made for transfers only, a test that tore down while a command was still being
  recorded), all fixed. No test opens a window; it was run by hand (below).
- **Input**: eight unit tests of the key map (the standard keys, no key bound twice, stick keys
  and their diagonals, the direction buttons of a tilted stick, merging, a keymap file and its
  mistakes, the gamepad counting only while the window has the focus), five of the XInput
  conversion (buttons by place and by name, triggers, dead zones and the rim, the layout of its
  state) and one that asks for a gamepad where there may be none; `hid` has a test that a sample
  carries the buttons and both sticks (and that releasing is the next sample), and the existing
  test of the sampling thread now presses keys through the same entry the window uses, sees A and
  the stick appear in the shared memory, and sees a lost focus let go of them.
- **By hand, on the title** (frames of the window captured with `PrintWindow`, only the game's
  own window; keys posted to that window; a keymap file using only keys nobody types, so that the
  person at the keyboard could not disturb the run): the menu selection moves with the
  directional pad keys and with the stick keys, as above. A first set of captures showed part of
  the window only, because the capturing program was not DPI aware and the display was scaled;
  the widths of the black bars matched the full window exactly, and a second set taken DPI aware
  shows the whole picture.

## Bugs that looking at the frames found

- **The "white" frames were empty.** A viewer shows transparent pixels as its background, and the
  dumps kept the target's alpha, so frames in which nothing had been drawn looked white. Pictures
  are now written opaque, and the log says how many pixels have color and alpha.
- **The clip control register was read as a flag.** `SET_VIEWPORT_CLIP_CONTROL` holds, besides
  "clip z runs from 0 to 1" (bit 0), a guard band and a clip mode; any non-zero value was taken as
  the first. The title's final pass (z at -1, as OpenGL conventions have it) was clipped away
  entirely, and so was most of the geometry of its other draws (A-096).
- **Every glyph was the first glyph.** The hardware runs the instances of a draw as a series of
  draws that count up, and the title issues them that way; the renderer made each draw one
  Vulkan instance whose id read as 0. The vertex stage now gets the counter in a push constant
  (A-098), and the text appears.
- **Reset values**: a title that never writes `SET_RASTER_ENABLE` or the binding groups still
  draws and binds on a real engine; without those values every draw was dropped (A-092).
- **ASTC, found by random blocks**: the offset endpoint modes clamp after the blue contraction,
  not before; endpoint modes of high dynamic range spoil only their own partition; an unsigned
  subtraction in the bilinear weights underflowed (in a debug build, a panic); in sRGB mode the
  reference widens alpha like the colors.
- **Mip levels of compressed textures** were sized by halving the number of blocks, not the number
  of texels (a 100 texel texture has 13 blocks at level 1, not 12). No title texture had hit it yet.
- **The picture after the first menu was black, for two reasons found by dumping the passes of
  one scene frame and looking at each.** (1) *Cube maps and 3D color lookup tables read as zeros
  (A-111).* In a scene frame 236 of the 957 texture bindings were all-zero cube maps and lookup
  tables: the title renders them face by face and level by level as 2D render targets, and those
  pixels lived only in the renderer's images, never in the memory the textures are made from.
  Targets are now written back to that memory when a texture is made from memory they cover.
  After it the HDR target of the scene held a lit picture, and the composite was still black.
  (2) *The viewport (A-112).* The macro that sets a render target writes 0 to
  `SET_VIEWPORT_SCALE_OFFSET` in every pass of the scene, and the renderer took that for "the
  transform is off": it drew with a full-target viewport that was not turned over, so the scene was
  upside down with its faces culled the wrong way round (inside out). The title's vertex programs
  output clip space and want the registers' viewport (negative scale y), whatever that register
  holds. With it every pass, the luminance chain and the composite show the scene.
- **The device was lost about two minutes into the cutscene, on an NVIDIA device only (A-114).**
  The first draw of a group of seven instanced draws (a vertex program of 840 instructions in 32
  blocks, a pixel program of 47) hung the GPU: `VK_ERROR_DEVICE_LOST` 2.5 s after the submission,
  the time of the operating system's 2 s watchdog, so a hang and not a fault. It was found by
  checkpoints in the command buffer (`NX_DEBUG_CHECKPOINTS`, which name the commands the device
  reached), then by waiting for the device after every draw from a frame near the loss
  (`NX_DEBUG_SYNC_FROM`, which names the draw with its state and textures). Taking the draw apart
  (each run is the whole boot, about four minutes): it still hung with the pixel program replaced
  by a constant color, without the depth test, with the rasterizer discarding everything, without
  its texture fetch, and after `spirv-opt -O`; it did not hang with its vertex program replaced by
  one that writes a constant position, or skipped, or on an AMD device. The program is valid
  SPIR-V and every jump in it goes forward, so the loop that the translator put around a `switch`
  on a program counter (A-091) cannot run for ever, yet the translation without that loop did not
  hang: a program whose jumps all go forward is now translated as its blocks in address order,
  each under a flag of its own that earlier blocks set (A-114). Over the same run, frames to
  13,000 were drawn with no loss, and a device test shows both forms give the same registers
  for 256 random states. Why the driver hangs on the loop is not known; the loop was also tried
  with its test in the loop header and with blocks numbered 0, 1, 2, and hung in both.
- **Things the scene needed that the first screens did not** (all found by the draws that
  failed, each with its program's address): texture instructions beyond `TEXS` (`TEX`, `TLD4S`,
  `TLD4`, `TXQ`, `TMML`; A-106), the four condition code flags where only the carry was modelled
  (A-107), sums and products that the host's compiler fused into multiply-adds and a position that
  was not invariant, so that a depth test for equality after a depth prepass rejected pixels
  (A-109), inline-to-memory uploads and compute launches (a launch or two a frame; A-108,
  A-110).
- **A frame was copied and hashed byte by byte.** Every frame each distinct texture was copied out
  of guest memory and hashed one byte at a time (FNV) to see whether it had changed: about 100 MB
  a frame in the scene, 90% of the draw time. Textures and programs are now hashed in place, a
  word at a time (`AddressSpace::hash`), and copied only when they changed: about 11 frames a
  second instead of 6.
- **Mip chains drawn as targets went through memory 22 times a frame.** What was left of the
  draw time was the write-back of A-111: each frame 22 targets (the levels of the title's
  luminance chain and of a bloom chain, which it draws as targets and samples as one texture)
  were read out of the device, which waited for everything recorded so far, swizzled into memory,
  read back from it and uploaded again: 40 ms of a 90 ms frame, 2 MB a frame. A target that is
  exactly a layer and level of the texture, or a strip of one, is now copied into the texture's
  image on the device (A-113), without waiting; the lookups of targets by address that this
  needed (they were a scan of all targets for each of about 10 textures a draw) are indexed.

## What is not done, and what is not checked

- **What follows the opening cutscene.** Nothing past it has been seen: the runs that were
  watched went to frame 13,000 (the cutscene starts at about frame 9,000 when the first menu is
  confirmed at frame 5,000 in a run without a person; the frames before it are the introduction
  screens, the first menu and the story card).
- **Speed.** The cutscene runs at about 15 frames a second (it was 6 before the textures were
  hashed in place and the mip chains copied on the device): about 600 draws a frame, most of the
  time in the renderer's draw path. An AMD integrated GPU of the same machine ran the same
  cutscene at about 26 frames a second, so the RTX card is slowed by something of the discrete
  device (the uploads through the ring in host memory, the waits for the device) and not by the
  draws themselves; that has not been looked into.
  A step-by-step account is printed every 100 frames with `NX_TRACE_GPU`. Nothing here is a claim
  about speed on other hardware or other scenes (A-102: every submission still waits for the
  device).
- **A run that stops answering.** `NX_WATCHDOG` caught one in about four runs of the cutscene:
  the main thread waiting for a mutex that another thread held while it was inside a request to
  a service (the command submission), with the command processor handling no methods and the GPU
  idle; it was not seen again after the watchdog learned to say more, and its cause is not known.
  An intermittent early stop of the title is also open: about one run in 30 with the renderer
  and none headless (0 in 88), more often while the machine is busy, it ends in one of two ways
  while the title loads resources on several threads, both in `al::ResourceSystem::createResource`:
  a jump into the heap from the destructor of a temporary heap, or the heap's allocation-failed
  callback for a 136-byte block of an ExpHeap of 755 MB that has one free block left (the dump the
  stop report now prints shows the heap's own fields). It is not tied to `NX_RANDOM_SEED`. The
  guest's threads run at once on host threads with no model of the console's cores and
  priorities, which the console's scheduler does not do (a thread keeps its core until it blocks,
  and equal-priority threads on a core never overlap); `NX_CORES=per-core` (A-115) is a first
  model of the cores, and whether it removes the stop has not been measured.
- **Sound** does not exist yet. **The gamepad** was never tried with a controller. Only the
  handheld controls exist (one player; touch, mouse, motion and rumble are absent), so a title
  that does not support the handheld style sees no controller (A-104). There is no gamepad on
  Linux (only the keyboard).
- **The window was run on one machine** (Windows 11, one NVIDIA GPU): it was compiled and its
  blit tested on Linux, but no window was opened there, and neither a display that is not 60 Hz,
  nor several monitors, nor fullscreen were tried. FIFO presenting and the guest's own 60 Hz
  thread both pace the title; how they interact on other refresh rates is unmeasured (A-105).
- **Shader instructions not translated** still stop the draw with the instruction and its
  address (`nx_shader --example audit` lists what the translator lacks: `TLD`, `TLDS`, `TXD`, the
  bindless forms of `TEX`, `TLD4` and `TXQ`, half-precision arithmetic, local and shared memory, atomics
  and others); the programs of the screens and the cutscene seen so far did not need them.
- **Textures:** 2D, 2D array, cube, 3D and depth textures are sampled now, with BC, ASTC LDR and
  the plain color formats; 1D and stencil textures, texture views of other formats (an sRGB view
  of a linear target) and mip chains of render targets other than through the write-back are
  refused or not modelled. ASTC textures of high dynamic range blocks are magenta.
- **The device's pictures reach guest memory only through the write-back (A-111).** A texture that
  is a render target of its own size is read from the device (A-100); a texture made from memory
  that color targets cover gets them written first; depth targets are never written back, and a
  title that reads a target with the CPU, or copies it with the copy or 2D engine, would see old
  memory. The copy and 2D engines are not driven; the inline-to-memory and compute engines are
  (A-108, A-110).
- **State not applied:** the window origin and `FLIP_Y` bits, per-viewport scissor and viewport
  arrays beyond the first, depth bounds, multisampling, point and line smoothing, logic ops beyond
  the blend state, and transform feedback.
- **Face culling and winding** are right for the title's quads and for its scene, which was
  looked at (A-103, A-112); unchecked beyond them.
- **Only one title, and only its first 7,000 frames.** What a gameplay frame needs is unknown; the
  next draw that fails will say why.

## Files

`nx_vulkan/src/astc.rs`, `nx_vulkan/src/render/` (`mod`, `draw`, `frame`, `shaders`, `textures`,
`targets`, `global`, `compute`, `convert`, `debug`, `display`, `blit`),
`nx_vulkan/src/{window,input}.rs`, `nx_vulkan/tests/render_*.rs`, `translate_*.rs` and
`astc_reference.rs`, `nx_gpu/src/{render,draw_state,compute,i2m,vm,lib}.rs` (`lib` has the progress counters),
`nx_shader/src/translate/` and `tests/{tmml,control_flow}.rs`, `nx_runtime/src/horizon/{input,gamepad,watchdog,cores}.rs`
and `services/hid.rs`, `docs/ASSUMPTIONS.md` A-092 to A-115.
