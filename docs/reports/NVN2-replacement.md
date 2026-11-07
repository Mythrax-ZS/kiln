# NVN2 replacement: textures and descriptor views

Updated 2026-11-07. The objective is to replace the guest graphics driver and remove the GPU
front's command decoding from its draws. The optional API boundary now owns native devices,
memory pools, buffers, descriptor pools, samplers, textures/views, queues and syncs. The owner's
NX2 program creates its display textures and stops at window construction. HLE does not yet
render a frame or improve the race's rate. The default remains off.

## What runs

- The exported `nvnBootstrapLoader` has an optional native wrapper and a preserved guest body.
  Direct calls and calls through the generated function table reach the same wrapper.
- `NX_NVN2=off`, or leaving it unset, runs the guest body. This remains the default.
- `NX_NVN2=trace` forwards calls to the guest driver and follows the procedure pointers returned
  by its loader and `nvnDeviceGetProcAddress`. It needs no title-specific addresses, procedure
  table layout, or injected calls into the driver. Each `(name, guest entry)` has a stable
  dispatcher token, and each call uses the caller's actual CPU, stack and TLS. Null results,
  faults and unwinds are preserved. Tokens change procedure pointer values, so trace is a
  diagnostic mode with overhead.
- `NX_NVN2=hle` owns procedure resolution and the native objects described below. An
  unimplemented entry stops at its first call, with its name and calling address. No API falls
  back to the guest driver in this mode: their state cannot be mixed. Native sm_86 translation
  remains the intended shader path.

Enabled modes require `Npdm::platform == Nx2`. The module descriptor advertises whether its
loader stub was built; an older descriptor refuses an enabled mode. Building once with
`--reuse-objects` recompiles the loader's unit and refreshes the descriptor, retaining the other
units. A runtime-only relink is sufficient for subsequent implementation changes.

`NX_NVN2_REPORT=<file>` writes a text report when the process ends: loader calls, resolved
procedures, and each procedure's count, original guest address (zero for HLE), and API name.
Trace now also records integer query results as `# integer <selector> <signed value> <count>`,
using arguments captured before the guest call even if that call overwrites the registers.
Sampler registrations also report how many descriptors matched the native encoder, differed,
or could not be observed. Unreadable observations leave forwarding and guest faults/unwinds intact.
Texture observations additionally compare storage sizes, alignments, initialized block geometry
and descriptor fields, grouped by flags/format. Descriptor checks exclude hardware cache,
compression and sample-location policy bits; they use the guest's block choice, while the
separate initialization check validates native automatic layout selection.
Keep reports from real programs outside the repository. A killed process leaves the initial
`running` marker instead of a completed report.

## Native devices and storage

Device initialization creates and retains the renderer's real Vulkan context. Unsupported
device flags, a missing Vulkan backend or an incompatible uniform-buffer alignment return
false. Each device has separate origin/depth state and a generation identity. Device, pool and
buffer calls validate native ownership and checked guest storage; retirement invalidates their
identities, and parents cannot retire while children remain live. No native host pointer is
placed in a guest object.

The initial integer profile follows 18 selectors observed through trace mode before the first
presented frame:

| Selectors | Meaning and value |
|---|---|
| `0x00`, `0x01` | API version 231, 39 |
| `0x04`, `0x0D`, `0x14` | Uniform, indirect-draw and control-memory alignments: 256, 4, 8 bytes |
| `0x0F`, `0x10`; `0x11`, `0x12` | Texture/sampler descriptors: 32 bytes each; 256 reserved entries each |
| `0x18`, `0x19`, `0x1A` | Shader scratch scale 576; alignment 4,096; granularity 16,384 |
| `0x2A`, `0x39` | ASTC support through the renderer's decoder; linear render-target stride alignment 128 |
| `0x51`, `0x55`, `0x5B` | Queue command minimum 65,536; compute default 262,144; control minimum 4,096 |
| `0x5A` | No NVN debug layer |

These values describe the emulated API, including its scratch-allocation policy; they do not
describe the host GPU's physical layout. The native virtual-pool model additionally reports
64-KiB pages through `0x3A`. Unknown selectors stop before writing the output. Only API version
queries accept a null device. Texture/image handles use the observed 20-bit texture and 12-bit
sampler packing. Explicit fast-clear registration returns false.

Pool and buffer builders preserve their observed fields and padding. Native object headers
retain the fields used by SDK accessors inlined into the caller. Ordinary pools map checked
guest backing at the same GPU address. Physical pools supply backing for 64-KiB virtual-page
mappings; virtual pools reserve GPU addresses outside CPU memory, independently for each device.
Mapping requests validate all owners and ranges before applying changes in order, including
unmapping. Buffers are bounded views of ordinary or virtual pools; physical pools cannot be
used directly as buffers. CPU mapping respects the pool's access flags. Labels are stored
without logging their contents.

Flush/invalidate validate ranges and fence coherent host storage. Native queues submit
completion markers; no accepted draw/copy/dispatch payload can write guest storage yet.
Completion/writeback, resource lifetimes during execution and sparse GPU access semantics must
be connected before payload submissions are accepted. Native-handle and allocator-backed pools
remain unsupported. The owned renderer is initialized; HLE currently submits only completion
markers through its shared device.

Sampler builders and objects now retain their full observed state, including FP arguments and
returns, raw border words, labels and immutable object snapshots. Sampler registration encodes
one 32-byte descriptor, writes its backing storage and retains the encoded entry in the native
pool. Both sampler and texture descriptor pools have checked ranges and parent lifetimes,
preserve fields read by inline SDK accessors, and install their reserved entries. Partial
descriptor-table writes stage the source before changing storage and refresh the native entries.
Texture registration now copies native resource/view snapshots alongside their encoded
descriptors; partial raw table writes retire the affected snapshots. Views select mip/layer
ranges and compatible format/swizzle/depth-stencil overrides. Layer views rebase their address
using the texture's layer stride. The renderer does not yet consume these entries through HLE.

Texture builders and 16-byte views preserve observed SDK fields and optional getter outputs.
Native textures own immutable state and checked ordinary or virtual pool storage, preserve
inline object accessors, and prevent their parents from retiring. Packaged uploads stage the
checked source before writing backing, including overlapping ranges and split virtual pages.
Native storage classes are opaque tags interpreted by the native address-space model. Labels
and debug identities are retained without exposing host pointers.

Storage sizing models buffer, pitched, block-linear, array/cube, volume, mipmapped, multisample
and BC/ASTC geometry. Packaged layouts override the block choice; automatic layout uses bounded
GOB dimensions, and minimal layout selects smaller blocks only when that saves storage. Sparse,
video/external textures and depth ZCull storage still stop explicitly. Vulkan image
materialization, texel transfers, resident-image updates/readback and in-flight lifetimes must
be connected before graphics payloads can consume these resources.

Queue builders now preserve the observed NX2 ABI, including boolean size getters with output
pointers. Native queues own Vulkan command pools on the device already created by HLE. They
keep separate pending markers and checked command/control budgets; secondary queues hold
their parent live. Supplied queue storage is leased, otherwise a host arena is allocated.
Normal priority and the observed base/parent/secondary flags are supported; other priorities,
unknown flags and nonzero ray-tracing budgets are refused.

FenceSync queues a fresh GPU completion without flushing. Flush submits the pending markers;
Finish appends a completion and waits. A sync created without a fence reports FAILED, an
unflushed fence can time out, and re-fencing keeps an older waiter's completion alive. CPU waits
release the driver lock and modeled core, and notice process termination. Renderer and native
queues share the actual Vulkan device and serialize access to its physical queue. Command
recording, draw/copy/dispatch payloads and presentation remain unimplemented; completion tests
exercise marker submissions. Graphics submission must also implement the distinct vertex-stage
completion condition and connect guest resource visibility before those payloads are accepted.

The layouts, ownership policy and remaining limits are recorded in A-299 through A-314.

## Validation

The runtime tests check forwarding of integer and FP/vector arguments and results, flags, stack
and TLS; stable tokens, different device entries, null resolution, guest unwinds and stops;
checked names and builder memory faults; and a missing HLE entry stopping without a fake result.
The new tests cover four-byte capability outputs and guards, cross-page faults, device
creation against a real headless Vulkan context, independent device state, reuse and stale
identities, and handle argument truncation. Storage tests need no graphics device: they check
pool/buffer aliasing, inline accessor fields, empty views, bounds, access flags and retirement;
map two physical pages into one virtual buffer and read across their boundary; and reject bad
mapping arrays and stale owners. Vulkan-dependent tests retain the usual external-tool skip
policy when no device is available.
Descriptor tests additionally cover sampler FP state and snapshots, pool counts/alignment and
inline fields, reserved and out-of-range indices, same-device registration, copied entries
after sampler retirement, partial table writes, stale identities and parent-finalization guards.
Queue tests cover the NX2 builder ABI, native parent/secondary creation, pending syncs,
cross-thread flushing, fence replacement and resource retirement. A Vulkan backend test checks
real GPU completion, cancellation, refusal to reuse a fence and device lifetime after the
original queue/renderer retire. Builds also pass without the Vulkan feature.
Texture tests check builder/view padding and output faults, mip/layer/volume/pitch geometry,
minimal layouts, immutable pool-backed objects, inline address/size fields, copied registrations,
overlapping packaged uploads and uploads across remapped virtual pages. Unreadable trace
observations preserve both normal returns and guest unwinds.

Synthetic NSO executables exercise direct loader calls and indirect procedure calls in all
three modes. A guest setter writes 7 into a builder; off and trace return that value, while HLE
clears it and returns 0. Another executable calls an unimplemented entry and must report its
name and stop. Corrupting the loader's old object before `--reuse-objects` verifies it is rebuilt.

The owner's NX2 program resolves 731 procedures in HLE mode, creates a real native Vulkan
device with flags zero, queries its profile, sets its origin/depth modes, and creates the first
native memory pool, initializes both descriptor pools, and constructs/registers a native sampler.
It creates a native queue with flags `0x9`, 1 MiB command and control budgets and a 256 KiB compute
budget and constructs three display textures, then stops at `nvnWindowBuilderSetDefaults`, with
the missing entry named. This validates progress through initialization; the buffer and
virtual-mapping paths have synthetic
coverage and have not yet been reached by this real-program HLE probe.

The texture validation trace completed 3,600 frames through the race. These comparisons all
had zero differences:

| Observation | Matched | Unavailable |
|---|---:|---:|
| Storage size | 18,358 | 284 |
| Storage alignment | 6,561 | 142 |
| Initialized size/block layout | 80,560 | 258 |
| Consumed texture descriptor fields | 203,123 | 258 |

The unavailable observations are sparse layouts, which HLE still refuses. Descriptor field
matches do not establish full descriptor-byte equality or native GPU consumption. Sampler
registration also compared 551 complete descriptors: all matched. Frame 3,500 was inspected:
road, starting grid and HUD drawn, with dark lighting/characters. An earlier diagnostic capture
also showed a white rectangle behind the countdown; that rendering issue remains unisolated.
The final trace refused 18 texture-header/type draws and one unsupported-format compute launch.
These are guest-driver trace frames. They do not establish native HLE rendering, a rendering
regression pass, or an I/O/performance improvement.

A trace through the race to frame 3,600 compares the native sampler encoder against the 32-byte
descriptors the guest driver actually writes: 888 matched, zero differed and zero were
unavailable. Frame 3,500 was inspected: the race road and HUD are drawn, with the existing
lighting/texture limitations. The run refused 18 texture-header/type mismatches and one
unsupported surface format. The full test gate ran alongside this diagnostic capture, so its
timing is not used as a performance comparison. This validates the observed sampler states and
their bytes; native queue consumption and wider state coverage remain open.

At entry-point revision `f8acfed`, trace mode completed the scripted route through the race to
frame 3,600, with three loader calls, 733 resolved procedure names and 59,857,523 calls to 247
distinct APIs. It sees every name
in the earlier 246-function census, plus 15 calls to `nvnMemoryPoolBuilderSetNativeHandleNVX`.
That extension is resolved after device initialization; the new boundary captures it without a
predefined procedure map. The off-mode and trace-mode dumps at frame 3,500 were inspected: both
show the road drawn. This covers this route through the race.

The same device, input script, frame dumps every 250 frames and `NX_TRACE_PERF=1`, seconds per
100 frames:

| Build and mode | At frame 3,400 | At frame 3,500 |
|---|---:|---:|
| Original revision `72bf797` | 10.82 | 12.94 |
| Entry-point revision `f8acfed`, off | 10.66 | 10.93 |
| Entry-point revision `f8acfed`, trace | 10.71 | 11.20 |
| Devices and storage, off | 10.18 | 10.12 |
| Queues and completion, off | 12.25 | 11.21 |

These are single runs with substantial timing variation. The original runtime was built from
that revision and linked with the original objects; its embedded guest image was checked to be
byte-identical to the new build's. These samples do not establish a performance change. The
devices/storage off run also completed 3,600 frames; its frame 3,500 was inspected and
shows the race starting grid, road and HUD drawn. No speed-up is attributed to the native
initialization work; HLE still stops before rendering. This off run still refused 16 commands
with a texture-header/type mismatch and one unsupported surface format; those cases remain open.

The queues/completion off run completed 3,600 frames. Frame 3,500 was inspected: road and HUD
drawn, black lighting regions/characters and large foreground geometry remain. Its countdown
phase differs from earlier captures, so those images are not a matched visual comparison.
This run refused 16 texture-header/type mismatches and one unsupported surface format. Tests
overlapped early frames but had ended before the measured race samples.

A further off-mode comparison used the previous descriptor revision `392394e` and the queue
implementation, unchanged guest objects, the same script and random seed 79,831. Both completed
3,600 frames. Seconds per 100 frames at 3,400/3,500 were 12.52/11.25 before and 10.20/10.65 after.
Tests overlapped the baseline's early frames, and shader-cache/startup conditions and countdown
phases differ, so this pair does not establish a speed-up. Both frame 3,500 images were inspected.
The baseline draws the road and HUD; the new capture draws the HUD and part of the road, with
a large black road region under/left of the kart. The intermittent road issue described in
[the coverage report](NX2-coverage-and-rate.md#open) therefore remains open; this comparison
does not isolate its cause or establish a rendering-regression pass. The baseline/new runs
refused 20/17 texture-header/type draws and one unsupported-format compute launch each.

On Windows, `cargo fmt --all --check`, `cargo lint` and `cargo test` pass: 1,191 tests passed and
one was ignored.

## Next

1. Window ownership, programs and render state, plus remaining texture modes and image/texel
   operations. Program creation must retain sm_86 programs for the existing native translator.
2. Command buffers that record draw-level operations and preserve API ordering, callbacks,
   indirect arguments, memory visibility and events.
3. Connect queued payloads and window presentation to the renderer and owned address
   spaces without replaying method streams. Complete GPU visibility, sparse access and in-flight
   resource lifetime handling here. Validate frames and time the race before enabling HLE by default.

The traced route reaches 247 API functions on the path into the race. That is an observed minimum,
not complete coverage of NVN2 or of later gameplay.
