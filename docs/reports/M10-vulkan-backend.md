# M10: the Vulkan backend (`nx_vulkan` renderer)

Status: the title's first screens are rendered, shown in a window, and the keyboard reaches the
title; the milestone is open at what comes after its first menu, and at sound (see "What is not
done"). Date: 2026-10-13.

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
  300 ms); confirming it made the screen fade and the title's introduction text appear, after
  which the picture stayed black for the 45 s that were watched, at about 10 to 13 frames a second
  (the menu itself ran at 57 to 71, by the dump times of every 120th frame), and why has not been
  looked into. The window was resized to a wide, a tall and a normal shape (equal black bars on
  the two sides that need them) and minimized and restored (the title kept running and answered
  the next key).
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
| `render::debug` | The switches for looking: `NX_TRACE_DRAWS=<frame>`, `NX_TRACE_STATE`, `NX_TRACE_TEXTURES`, `NX_TRACE_CBUFS`, `NX_DUMP_FRAMES` (with `NX_DUMP_EVERY`, `NX_DUMP_COUNT`, `NX_DUMP_TARGETS`), `NX_DUMP_DRAWS=<dir>` with `NX_DUMP_FROM` (a picture of the target after each command of the traced frame), `NX_DEBUG_NO_CULL`, `NX_DEBUG_FULL_VIEWPORT`, `NX_DEBUG_SOLID_PIXEL`, `NX_SPIRV_CHECK`, `NX_VULKAN_VALIDATE`. `NX_RENDERER=vulkan` turns the renderer on in a produced program. |
| `window`, `render::display`, `render::blit` | A winit window on a thread of its own (feature `window`, on by default in the runtime): the raw handles for a surface, its size, close-to-exit, and the keys and focus handed to an `InputSink`. The display makes the swapchain (sRGB when the buffers are, FIFO unless `NX_PRESENT_MODE` says otherwise), and the blit copies the shown crop of a buffer into the swapchain image, fitted with black bars. |
| `nx_runtime` `horizon::input`, `horizon::gamepad` | Keys by physical name, the key map (standard, or `NX_KEYMAP`), stick directions from keys, the XInput gamepad (loaded at run time, positional face buttons, dead zones), merging, and the trace (`NX_TRACE_KEYS`). `hid` writes the merged state into its samples every 5 ms. |

## What the tests cover

- **21 renderer tests on the device** (`tests/render_*.rs`): clears (the whole target, a rectangle,
  BGRA targets, refusals), flat and blended draws, indexed quads, triangle strips and fans
  rewritten for Vulkan, textured draws (block-linear texels, the header swizzle, compressed
  blocks, ASTC blocks decoded on the host, once-a-frame comparison with memory, a handle that
  names no header), a render target sampled as a texture (with the header's swizzle, drawn to
  again afterwards, not read while it is being drawn to), global memory read and written in place
  by a vertex program, and the instance counter of each draw. Each picture is compared pixel by
  pixel with the expected one.
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
  headers.
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

## What is not done, and what is not checked

- **What follows the first menu.** After the menu is confirmed the title shows its introduction
  text and then a black picture for as long as it was watched (45 s, at about 10 to 13 frames a
  second). Whether that is a scene that does not draw yet (an untranslated instruction would stop
  the run and say so; none did), a load that takes longer, or something else has not been
  looked into. Nothing past it has been seen.
- **Sound** does not exist yet. **The gamepad** was never tried with a controller. Only the
  handheld controls exist (one player; touch, mouse, motion and rumble are absent), so a title
  that does not support the handheld style sees no controller (A-104). There is no gamepad on
  Linux (only the keyboard).
- **The window was run on one machine** (Windows 11, one NVIDIA GPU): it was compiled and its
  blit tested on Linux, but no window was opened there, and neither a display that is not 60 Hz,
  nor several monitors, nor fullscreen were tried. FIFO presenting and the guest's own 60 Hz
  thread both pace the title; how they interact on other refresh rates is unmeasured (A-105).
- **Shader instructions not translated** still stop the draw with the instruction and its
  address, as in M9 (`TEX`, `TLD`, `TLD4`, `TXQ`, half-precision, local and shared memory,
  atomics and others): the 10 programs of these screens did not need them. The next screens will.
- **Textures:** only 2D and 2D array (and BC, ASTC LDR, the plain color formats) are sampled;
  1D, 3D, cube, depth and stencil textures, texture views of other formats (an sRGB view of a
  linear target), and mip chains of render targets are refused or not modelled. ASTC textures of
  high dynamic range blocks are magenta.
- **The device's pictures never reach guest memory.** A texture that is a render target is read
  from the device (A-100); a title that reads a target with the CPU, or copies it with the copy or
  2D engine, would see old memory. The compute, copy, 2D and I2M engines are not driven; none of
  their methods was among the busiest the processor listed in these frames.
- **State not applied:** the window origin and `FLIP_Y` bits, per-viewport scissor and viewport
  arrays beyond the first, depth bounds, multisampling, point and line smoothing, logic ops beyond
  the blend state, and transform feedback.
- **Face culling and winding** are right for the title's quads and unchecked beyond them (A-103).
- **Only one title, and only its first 1,500 frames.** What a gameplay frame needs is unknown; the
  next draw that fails will say why.
- **Timing** is the measurement above; there is no per-frame profile, and every submission waits
  for the device (A-102).

## Files

`nx_vulkan/src/astc.rs`, `nx_vulkan/src/render/` (`mod`, `draw`, `frame`, `shaders`, `textures`,
`targets`, `global`, `convert`, `debug`, `display`, `blit`), `nx_vulkan/src/{window,input}.rs`,
`nx_vulkan/tests/render_*.rs` and `astc_reference.rs`, `nx_gpu/src/{render,draw_state}.rs`,
`nx_runtime/src/horizon/{input,gamepad}.rs` and `services/hid.rs`, `docs/ASSUMPTIONS.md` A-092 to
A-105.
