# M8: the GPU front (`nx_gpu`)

Status: complete, as far as the owner approved it. Date: 2026-10-12.

The task was steps 1 to 3 of the proposal that ends `M7-title-boot.md`: the driver service and its
device nodes, the display path with its buffer queue, and a Maxwell command processor that traces
what a title sends *before* anything is drawn. The measurement it was to make possible: let the
title run its start-up and its main loop up to its first draw call, with every GPU method it uses
listed. The title is the owner's own dump; this report gives aggregate numbers only and does not
name it.

**Nothing is drawn, no shader is read and no picture exists.** Shader translation and the Vulkan
backend are not started; they need the owner's go-ahead again (`CLAUDE.md`).

## Result

The title runs. It goes through the SDK start-up it did in M7, then through the GPU, display,
account, input, time, audio, network and font services listed below, and enters its main loop
(the game framework's frame loop, a frame's update and its rendering commands). It presents
frames to our display layer until the run is stopped on purpose:

```
$ NX_MAX_FRAMES=600 NX_TRACE_GPU=1 ./title
recomp runtime: NX_MAX_FRAMES reached after 600 presented frames
GPU commands: 1201 submissions, 6003 pushbuffer entries, 749770 words, 630488 method writes, 4228 semaphore releases
  macro calls: 30641 calls of 11 different macros
  macros ran 1616820 instructions and wrote 606861 methods
  draws: 138125 (828750 elements named), clears: 600
    138125 x DRAW_INDEX_BUFFER
  first draws: #1 DRAW_INDEX_BUFFER, triangles first 0, count 6, color target 0 at 0x8103530000 1600x900, from macro 27
```

(The addresses are GPU virtual addresses of this run.) 600 frames take 11.4 s on this machine,
about what the 60 Hz vsync event we set (A-059) gives plus the start-up: the run is paced by that
event, not by how fast the GPU front is. Three runs of 30 frames gave the same counts, down to the
addresses.

- **The first draw call** is a `DRAW_INDEX_BUFFER` of 6 indices as triangles into a 1600x900 color
  target, written by macro 27. Every frame has one `CLEAR_SURFACE` (600 in 600 frames). All the
  draws are indexed with 6 indices each (a quad): one per frame in the first 30 frames, 230 per
  frame on average over the 600.
- **Everything is decoded.** In 600 frames the command processor read 1,201 `SUBMIT_GPFIFO` calls
  (alternately a 7-entry and a 3-entry submission), ran 30,641 macro calls of 11 macros
  (1,616,820 macro instructions, 606,861 method writes) and met no method, class, pushbuffer
  format or macro instruction it could not handle. The distinct methods are listed in
  `docs/reports/M8-methods-used.md`: 519 written by the method stream (489 on the 3D class, 21 on
  compute, 3 on 2D, 6 host methods; array registers counted per element) and 53 written by macros
  (all on the 3D class).
- **The driver used 4 nodes and 22 different ioctls** (`NVMAP_IOC_CREATE`, `ALLOC`, `GET_ID`;
  `NVGPU_AS_IOCTL_ALLOC_AS_EX`, `ALLOC_SPACE`, `BIND_CHANNEL`, `MAP_BUFFER_EX`, `REMAP`,
  `GET_VA_REGIONS`; `NVGPU_IOCTL_CHANNEL_*`: `SET_NVMAP_FD`, `SET_PRIORITY`, `ALLOC_GPFIFO_EX2`,
  `ALLOC_OBJ_CTX`, `SUBMIT_GPFIFO`, `SET_ERROR_NOTIFIER`, `ZCULL_BIND`, `SET_USER_DATA`;
  `NVGPU_GPU_IOCTL_GET_CHARACTERISTICS`, `GET_TPC_MASKS`, `ZCULL_GET_CTX_SIZE`, `ZCULL_GET_INFO`,
  `ZBC_GET_ACTIVE_SLOT_MASK`). In 600 frames: 169 memory handles created and allocated, 335 buffers
  mapped into the GPU address space, 10 reservations, 9 remaps. It never opened `nvhost-ctrl`.
- The reports at the end of a run, and the stop for the run itself, are for measuring:
  `NX_TRACE_GPU=1` prints each submission and the report above (`NX_GPU_TOP=0` lists every method,
  the default is the busiest 25), `NX_TRACE_GPU=2` every method write and every macro call and
  write, `NX_MAX_FRAMES=<n>` ends the run after `n` presented frames.

## What was built

### `nx_gpu`, a crate with no knowledge of Horizon

`nx_gpu/` depends on nothing; `nx_runtime` depends on it and puts it behind services. 125 unit tests.

- **`nvdrv`** (`nvdrv/`): a byte-level state machine. `open`/`close`/`ioctl`/`query_event`/
  `take_signals` over five nodes (`nvmap`, `nvhost-ctrl`, `nvhost-ctrl-gpu`, `nvhost-as-gpu`,
  `nvhost-gpu`; any other path stops the run). An ioctl number is decoded as a Linux `_IOC` word
  and dispatched by (type, number, size). Descriptors start at 1. Error codes are the `NvError`
  values (A-051).
- **`nvmap`**: handles with ids from 1, page-rounded sizes, allocation with alignment checks,
  `GET_ID`, reference counts (A-052).
- **The GPU address space** (`vm.rs`): `ALLOC_AS_EX` with the default range and the split between
  small and big pages, `ALLOC_SPACE`/`FREE_SPACE`, `MAP_BUFFER_EX` (fixed and free placement, the
  over-reach to the next big page the title relies on), `UNMAP`, `REMAP`, translation, and GPU
  reads and writes that cross mappings (A-053, A-065, A-069).
- **Syncpoints** (`syncpt.rs`): 192, id 0 reserved, promises, increments, waiters woken when a
  threshold is reached, wrap-around comparison (A-055).
- **Channels and the command processor** (`nvdrv/channel.rs`, `gpu.rs`, `channel.rs`, `pushbuf.rs`,
  `engine.rs`): `ALLOC_GPFIFO_EX2`, `ALLOC_OBJ_CTX`, `SUBMIT_GPFIFO` with fences; GPFIFO entry
  decoding; the pusher state machine for the increasing, non-increasing, immediate, increase-once
  and end-of-segment commands; the host methods (`SET_OBJECT`, semaphores, `MEM_OP`, `WFI`, ...);
  five engine classes bound to subchannels; a 0x1000-word register file for each engine; the
  report semaphore of the 3D engine; method name tables generated from NVIDIA's open-gpu-doc
  headers by `tools/gen_methods.sh` (MIT; only offsets and names are used); a `Stats` report
  (A-056, A-060 to A-063).
- **The macro engine** (`mme.rs`, new in this milestone): the 3D engine's small processor.
  `LOAD_MME_INSTRUCTION_RAM` and `LOAD_MME_START_ADDRESS_RAM` fill its code and its table of start
  addresses; a `CALL_MME_MACRO`/`CALL_MME_DATA` command is a call whose parameters are its words;
  the interpreter (the Fermi macro instruction set, from Mesa's description, A-077) runs it and
  sends every write the macro makes through the same path as a write from the method stream.
  Draws (`DRAW_INDEX_BUFFER`, `DRAW_VERTEX_ARRAY`, the combined begin-draw-end forms, inline
  vertices and indices closed by `END`, `DRAW_AUTO`) and clears are counted and the first three are
  described. A fault in a macro stops the run with the macro number and, with `NX_TRACE_GPU`, a
  listing of its code. 18 tests, including a small assembler.
- **The display** (`parcel.rs`, `display.rs`): the Android `Parcel` codec, the graphic buffer
  (`GBFR`, its nvmap id), the 84-byte `QueueBuffer` input, a 64-slot `BufferQueue` (free, dequeued,
  acquired), a compositor with displays, layers, binder ids and the buffer-available and vsync
  events, and the `IGraphicBufferProducer` transactions (request, dequeue, queue, cancel, connect,
  disconnect, get native handle, ...) (A-057 to A-059).

### Services in `nx_runtime`

New in this milestone (`nx_runtime/src/horizon/services/`), each with the commands the title used
and an honest stop for the rest. The bracketed numbers are the assumptions in
`docs/ASSUMPTIONS.md`.

| Service | What it does |
|---------|--------------|
| `nvdrv` | `Open`, `Ioctl`, `Close`, `Initialize`, `QueryEvent`, `SetAruid`, `Ioctl2`, `Ioctl3` over `nx_gpu` (A-050) |
| `vi:u` | the root service, the application display service (`OpenDisplay`, `OpenLayer`, `SetLayerScalingMode`, `GetDisplayVsyncEvent`, the relay service), `IHOSBinderDriver` (A-057) |
| `acc:u0`, `set` | one local user and the language/region settings (A-066); the launch parameter that names the pre-selected user |
| `hid` | the handheld Npad connected with idle input, a new sample every 5 ms; six-axis start/stop; vibration devices that never vibrate (A-070) |
| `time:u` | the clocks and a UTC time zone (A-071) |
| `nifm:u`, `bsd:u`, `ssl`, `prepo:u`, `nfp:user`, `friend:u` | the console is offline: no network, no sockets, no contexts, reports discarded, no amiibo, no friends (A-072) |
| `pl:u` | six stand-in system fonts with an empty glyph for every character of the Basic Multilingual Plane (A-073) |
| `audren:u` | a silent audio renderer with the layouts `libnx` documents, an audio device, a 5 ms frame event (A-075, A-076) |
| `fsp-srv` | save data on the host: `OpenSaveDataFileSystem` and the file system, file and directory interfaces (A-068) |
| `appletOE` | the library applet launchable event, the managed display layer, the pre-selected user's launch parameter (A-043, A-066) |

And in the kernel layer: receive-list (C) and pointer (X) descriptors in both directions
(`Request::pointer_output`, `Response::with_pointer_reply`, the pointer buffer size a service
reports, A-067), shared memory objects with `MapSharedMemory` and `UnmapSharedMemory` (SVCs 0x13
and 0x14: 41 SVCs are implemented, A-074), the `on_finish` hooks, and in a stop report the
return addresses of the stopped thread's frame-pointer chain.

### Tools for iterating on a title

- `recomp symbols --exefs <dir> [--match <text>] [<address>...]` names the function around an
  address in any module with a symbol table (the title's SDK has one), which turns the return
  addresses of a stop report into a call stack.
- `NX_TRACE_SVC=1` now also lists what each `WaitSynchronization` waits on and when an event is
  made; stop reports end with the return addresses up the stack.
- `NX_TRACE_GPU`, `NX_GPU_TOP`, `NX_MAX_FRAMES`, `NX_SAVE_DIR` (default `./nx_save/<program id>/<kind>`).

## How the title got here: what each stop was

The method was the one of M7: run, read the stop, implement exactly what was asked, run again. The
stops, in order, with what the title was doing and what the answer was:

1. `nvdrv`, `vi:u` and the display path: a full NVN device initialization, a layer, a swap chain.
2. `acc:u0`, `set`, `fsp-srv` save data, `ISelfController::CreateManagedDisplayLayer`,
   `EnsureSaveData`, the pre-selected user's launch parameter (the title aborted in its account
   code without it), pointer-buffer sizes (a size of zero made the SDK answer 0x11A0B).
3. `hid`, `time:u`, `nifm:u`, `bsd:u`, `nfp:user`, `friend:u`.
4. **`pl:u`**: a size-0 answer made the title fault in its font code, and so did a first stand-in.
   The cause was in the stand-in, not in the title: its `maxp` table was 30 bytes where version 1.0
   needs 32, and its character map had only the space, so the title's glyph cache registered
   nothing and the code that used the result faulted on a null node. Both are fixed and tested.
5. A thread of the network library aborted inside `nn::socket::Initialize`: our `RegisterClient`
   answered the process id, and the SDK treats a nonzero client id as a failure (it now answers 0).
6. `ssl`, then `nifm`'s `GetResult` (2110-0300, no internet connection), `audren:u`, `apm`'s
   `GetPerformanceConfiguration` (a new session now reports `0x00020003` for the normal mode),
   `prepo:u`, and the library applet launchable event, which the title waits for.
7. **The controller applet**: the title's pad system found no controller and launched the
   `controller` library applet (`CreateLibraryApplet(0xC)`). A handheld Npad with idle input makes
   it carry on. Library applets are firmware, so a request for one still stops the run, naming the
   applet.
8. The macro engine: the first macro the title ran ended its loop with an exit flag in a branch
   delay slot (see A-077); the semantics that make all its macros end sensibly are implemented.

The last stop was none: the title runs.

## Tests

`cargo test` on Windows (fmt and clippy clean, the exit status of each step checked): 125 `nx_gpu`
unit tests, 298 `nx_runtime` unit tests (of which 110 are in the kernel layer and its services, driven through real IPC
messages and real host threads), 30 Unicorn differential tests, 85 `recomp` tests, and the opt-in
real-title tests (skipped without `RECOMP_TITLE_DIR`).

| Layer | Test | Result |
|-------|------|--------|
| `nvdrv` | every node and ioctl by hand-built argument bytes; error codes; descriptors; the over-reach of a mapping; remap | pass |
| Address space | reservation, fixed and free mapping, unmap, remap, translation, reads that cross mappings | pass |
| Pushbuffers | each command format, a command continued in the next entry, `ONE_INC` | pass |
| Engines | register files, the report semaphore (release and acquire), macro upload through its pointers | pass |
| Macro engine | each ALU operation and its carry, the bit-field forms, state reads, every assignment form, branches with and without delay slots, loops, the exit flag in each position, parameters (too few, too many), the runaway and the stray macro, a start address | 18 pass |
| Draws | a macro that draws (its primitive, first, count and where it came from), the combined draw methods in the stream, inline indices closed by `END`, clears | pass |
| `display` | the parcel codec round trips; buffer queue states; every transaction; vsync events | pass |
| Services | each command by IPC messages built byte by byte, with the stops for the commands nobody implemented | pass |
| `hid` | the Npad publishes only when the program has asked for it; the lifo ring fills and wraps; the entry fields; the same through the shared memory a program maps | pass |
| `pl:u` | the font is a well-formed TrueType file (table order, checksums, sizes, the head adjustment, the character map) | pass |
| `audren:u` | each section of the answer; memory pool states; wave buffers consumed in virtual time (several frames, pitch and rate, looping, stopped and new voices); a layout nobody documented stops | pass |
| The title | built whole, run to 30 and to 600 presented frames on Windows (three runs of 30 frames identical in every count; two runs of 600 identical in theirs) | as described |

### Mutations

About 55 single-line mutations were injected into the new code and the tests were run after each:
the macro engine (the ALU operations, the bit-field forms, the method address and its step, the
immediate's sign, the exit-flag and delay-slot rules), the draw decoding, the Npad publisher (the
attribute bits, the ring, the conditions for being connected), the audio renderer (pool states, the
voice clock, looping, stopped and new voices, the answer's sections), the font generator (table
sizes, the glyph count, the highest characters), the answers of the offline services, and, of the
code written earlier in this milestone, the pushbuffer decoder, the syncpoint comparison, `nvmap`
rounding and alignment checks, parcel padding, the address space's placement and page-size choice
and the buffer queue's states. Seven were not caught at first (the merge not clearing its
destination field, a one-element inline draw, the attribute constant compared with itself, the
highest characters of the font's map, the exact 64 KiB page-size boundary, and the two alignments
of free placement); each showed a missing check, the check was added, and all are caught now.
One mutation did not compile and three did not apply.

### Linux

See the end of this file for the run on Linux.

## Defects the work found

In the code written earlier in this milestone, each found by running the title:

- `ALLOC_AS_EX` had its fields in the wrong order; `nvmap` `ALLOC` checked an alignment a title
  need not give; `MAP_BUFFER_EX` refused a mapping that reaches past the end of its handle (which
  the console accepts), and the title died in a "pure virtual function called" because of it.
- Receive-list and pointer descriptors were not supported; two services answered with a pointer
  buffer size of zero.
- A host method (`MEM_OP_B`, 0x2c) and the binding of classes that need no separate context were
  refused; an `ONE_INC` pushbuffer command lost its state between words.

In this part:

- The stand-in font's `maxp` table was two bytes short and its character map covered one character.
- `RegisterClient` answered a nonzero client id, which makes the SDK abort.
- In the macro engine, the reference simulator the instruction set was taken from runs the delay
  slot of a branch that is not taken twice and stores the wrong value for one of the assignment
  forms; both were found by reading the title's own macros and are fixed here (A-077).

## Not verified

- **Nothing was run on a Switch.** The layouts are `libnx`'s (audio, HID, the nvdrv structures,
  the Android parcels), NVIDIA's public class headers (methods and their fields), Switchbrew's
  command tables, and Mesa's description of the macro instruction set. The behavior of the
  answers (what an idle controller publishes, what an offline console says, the silent renderer,
  the stand-in fonts, every value in A-050 to A-077) is ours and is marked UNVERIFIED there.
- **The pictures.** What the draws would show is unknown: nothing was drawn, the blank fonts would
  show nothing for text, and the audio is silent. "The title runs 600 frames" means it submits
  work and presents buffers to a layer that discards them; it does not mean it is playable, or
  that the work is what the console would receive.
- **One title.** Its first frames are what was measured. Another title uses other ioctls, other
  macros and other methods; the runtime stops with the name of the first one it has not seen.
- **The macro engine on hardware.** All 30,641 macro calls ran to the end with every parameter
  read, which fits the instruction set but cannot prove it. The exit-flag and delay-slot rules are
  chosen to make the title's own macros finish (A-077). Instancing, split begin/end and the shadow
  RAM modes are not modelled.
- **State the draws would need**: render targets, textures, vertex and index buffers, constant
  buffers, shader programs and blend state are written by the title and recorded only as register
  values; none is interpreted yet.
- **Timing**: fences and syncpoints complete at once; the vsync is a 60 Hz thread; the audio frame
  is a 200 Hz thread.
- Not reached by this title: `nvhost-ctrl`, compute dispatches, 2D blits, the copy engine, the
  other layers and displays, `nvdrv` `Ioctl2`/`Ioctl3` with extra buffers.

## Next

None of this is started. The two largest pieces need the owner's go-ahead.

1. **State tracking before shaders.** Interpret the register state the draws read (render target
   and depth target, viewports and scissors, blend and depth state, vertex formats and buffers,
   index buffers, constant buffer bindings, texture and sampler tables, the shader program table)
   into one structure per draw, traced the way the commands are. It needs no new service and shows
   what a Vulkan backend would be asked for. The title's 1600x900 target and indexed quads are the
   first thing it would describe.
2. **Shaders** (needs the go-ahead): the title's programs are Maxwell SASS behind a 0x50-byte
   header. Translate to SPIR-V, tested without the title by assembling programs and comparing the
   translation (run on a software Vulkan implementation) with a reference interpreter of the
   instruction set, the way the CPU side is tested.
3. **The Vulkan backend** (`ash`, needs the go-ahead): memory and resources, the block-linear
   texture layout and its formats, render targets, pipeline and descriptor caches, synchronization
   against the syncpoints, presentation.
4. In parallel and independent of the GPU: real input (a keyboard and gamepad mapped to the Npad),
   audio output for the renderer's voices, the next services the title will ask for as it goes
   past the first frames (`hwopus`, `audout:u`, `ldr:ro`, `mii:u`, ... : 22 of its 42 services are
   still not implemented), and a look at what it does after the opening frames.

The clean-room rule stays: Switchbrew and the public hardware documentation for the formats; the
GPL emulators are read for architecture only and no code is taken from them.
