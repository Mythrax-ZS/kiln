# NVN2 replacement: shared events and deferred queues

Updated 2026-11-07. The objective is to replace the guest graphics driver and remove the GPU
front's command decoding from its draws. The optional API boundary now owns native devices,
memory pools, buffers, descriptor pools, samplers, textures/views, queues, syncs, windows,
command buffers, programs and events. Typed command sets execute native GPU color clears and barriers,
with allocation callbacks, shader bindings and ordered fences. Programs retain sm_86 binaries
and feed native Vulkan module/pipeline caches. Typed render state drives scissored clears and
a limited single-target array-draw path; watched immutable vertex buffers reuse device memory
until their CPU backing changes. Event signals update coherent guest storage on the GPU;
logical queue workers wait without blocking the API caller or the physical device queue.
Vertex data is resolved after waits, including direct reads of single-page event aliases.
The owner's NX2 program creates five events and three queues, then stops at
`nvnTextureWriteTexels`. Synthetic tests execute translated graphics shaders,
draw instances, check fixed state against pixels and dispatch a translated compute module.
The title has not rendered an HLE frame.
No race-rate improvement is established. The default
remains off.

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
  back to the guest driver in this mode: their state cannot be mixed. Native binaries use the
  existing sm_86 translator; shader-resource binding and wider draw/dispatch execution remain open.

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
HLE reports also count owned native textures/windows and display GPU clears, presentations and
pixel-readback bytes and draws across live windows. These counters cover the explicit display-image
boundary, excluding GPU-to-GPU display blits and other resource transfers.
They also count owned command buffers, recorded command sets, GPU submissions across live
native queues and the scratch-binding byte ranges retained in those queues' state. Scratch
binding metadata alone does not implement shader execution or allocate shader scratch storage.
Program counters describe live program objects, their retained binaries and their cached
Vulkan modules; code-input bytes count successful copies at program linking. Queue binding
counters describe retained stage/sample-shading state. Vertex-input counters separate logical
CPU bytes copied, padded GPU upload bytes and unchanged-input cache hits; queue submissions
include those input uploads and ready-prefix splits. Shared vertex binding bytes count logical
ranges bound directly from event imports, not measured GPU bus reads. State updates count
expanded typed state operations. Event counts describe live native objects. These are
scoped counters, not a measure
of total process CPU/GPU traffic; object retirement can reduce the live-resource counts.

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
queries accept a null device. Native command recording additionally reports alignment four
through `0x13` and minimum command/control allocation callbacks of 256 bytes through
`0x15`/`0x16`. These are native memory-budget choices, not guest hardware-stream sizes.
Texture/image handles use the observed 20-bit texture and 12-bit
sampler packing. Explicit fast-clear registration returns false.

Pool and buffer builders preserve their observed fields and padding. Native object headers
retain the fields used by SDK accessors inlined into the caller. Ordinary pools map checked
guest backing at the same GPU address. Physical pools supply backing for 64-KiB virtual-page
mappings; virtual pools reserve GPU addresses outside CPU memory, independently for each device.
Mapping requests validate all owners and ranges before applying changes in order, including
unmapping. Buffers are bounded views of ordinary or virtual pools; physical pools cannot be
used directly as buffers. CPU mapping respects the pool's access flags. Labels are stored
without logging their contents.

Flush/invalidate validate ranges and fence coherent host storage. Native queues execute clears,
limited array draws and completion markers. Display-image writes stay in their Vulkan images;
general writeback to guest pool bytes, aliasing, sparse GPU access and copy/dispatch visibility
remain open. Watched vertex inputs retain immutable GPU versions through real submission
fences. Native-handle and allocator-backed pools remain unsupported. The title does not yet
submit draw/copy/dispatch payloads through HLE.

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
video/external textures and depth ZCull storage still stop explicitly. Window display textures
now materialize retained Vulkan images. Other texture residency, texel transfers, general
resident-image updates/readback and command-resource lifetimes must be connected before
graphics payloads can consume these resources.

Queue builders now preserve the observed NX2 ABI, including boolean size getters with output
pointers. Native queues own Vulkan command pools on the device already created by HLE. They
keep ordered pending command sets/markers and checked command/control budgets; secondary queues hold
their parent live. Supplied queue storage is leased, otherwise a host arena is allocated.
Normal priority and the observed base/parent/secondary flags are supported; other priorities,
unknown flags and nonzero ray-tracing budgets are refused.

FenceSync queues a fresh GPU completion without flushing. Flush submits the pending prefix;
Finish appends a completion and waits. A sync created without a fence reports FAILED, an
unflushed fence can time out, and re-fencing keeps an older waiter's completion alive. CPU waits
release the driver lock and modeled core, and notice process termination. Renderer and native
queues share the actual Vulkan device and serialize access to its physical queue. Typed clear
and limited array-draw payloads execute alongside those markers. Copies, dispatch and general
guest resource visibility remain open. FenceSync condition zero supports full GPU completion;
vertex-only condition one stops explicitly. It previously used the same full completion and
could not establish the requested release point. Unsupported flags stop explicitly.

Each logical queue now has a submission worker. Flush validates and enqueues a ready prefix,
then returns. The worker submits operations before an event wait, observes the actual shared
word and resumes when its predicate passes. Another logical queue can submit its signal while
the first waits; no spin shader occupies the physical queue. CPU API signals notify waiters,
while GPU and inline CPU stores are observed with one-millisecond polling timeouts. Finish and
finalization wait outside the driver mutex and guest core. Presentation waits for its prefix
to reach Vulkan submission before consulting the image's completion. QueueWaitSync retains
the source completion, including a flushed prefix that has not yet reached physical submission.
Unflushed cross-queue waits remain refused. Worker errors cancel dependent completions and
become fatal diagnostics on subsequent calls. Native budgets cap deferred work at 4,096 queued
jobs and one million queued operations; this excludes a currently executing job and GPU flights.
GPU-written private completion cells order a wait after earlier writes to the same event word.
An old matching value cannot satisfy the wait before a queued reset executes. These cells are
reused only after GPU/CPU owners release their import; private storage is capped at 16 MiB and
4,096 imports per logical queue. They add internal GPU writes, excluded from vertex-upload and
pixel-readback counters. The tests do not measure total event bus traffic or release latency.

## Native windows

Window builders preserve the observed 96-byte layout, device field, interval encoding and
HDR metadata argument/output ABI. Native windows preserve inline SDK accessors in a 536-byte
header and retain their non-null opaque guest window handle. The current application display supports
two to four distinct, equally sized 2D display textures, one acquired texture at a time, swap
intervals 0 through 5, active-buffer changes and checked crops. HDR/color-space conversion and
other platform windowing modes remain open; metadata storage alone does not implement HDR.

Each display texture retains a real Vulkan image on its native device, independently of the
old command processor's target cache and guest-memory hashes. Initial contents are transparent
black by native policy, written by an asynchronous GPU clear. Acquisition waits for the image's
actual prior-use fence, then installs its availability sync and returns the texture index.
Acquisition can construct the sync in uninitialized caller storage, matching the observed
SDK copy-out behavior. Presentation requires the acquired texture and a queue from the same
device, flushes that queue's pending prefix and presents the retained image. Reuse is tracked
against a fresh GPU fence covering the presentation blit. Window ownership prevents texture
retirement; images and fences retain their actual Vulkan device.

Acquisition waits and presentation release the driver mutex and modeled core. The existing
display pacing, input frame counters and `NX_MAX_FRAMES` apply to the native renderer through
a shared presentation callback. A visible window receives a GPU blit. Headless presentation
performs no pixel readback unless a frame dump is requested. Explicit diagnostics/dumps still
copy pixels out; the display counters record those bytes. Image destruction currently waits
for the device as a conservative lifetime barrier; normal presentation does not call device
idle. More granular retirement remains an optimization to complete with command execution.

## Command recording and submission

Native command buffers preserve the observed 176-byte SDK header and supported inline
bookkeeping. Command blocks use checked ordinary CPU/GPU-accessible pools; control blocks use
mapped CPU storage. A replacement block retains earlier allocations already consumed by the
current recording. Used/free queries describe native recording budgets, not encoded GPU method
bytes. Inlined BeginRecording bookkeeping is adopted; unaccounted inline command words stop
explicitly instead of being omitted or replayed.

Allocation callbacks run on the caller's guest core with the driver mutex released. They can
reenter AddCommandMemory/AddControlMemory. Normal returns retry the pending operation;
insufficient allocation, disabled callbacks, faults and nonlocal unwinds are handled explicitly.
Callback changes to FP status, TLS and exclusive state survive the callback boundary.

EndRecording produces an immutable typed command set with an opaque native token and a small
control-storage stamp. Sets survive recorder finalization. Reusing their command/control
storage invalidates old handles through interval indexes; pool/device generations are checked
again at submission and flush. CallCommands retains referenced sets; CopyCommands copies their
typed operations independently. Called graphs are limited to 64 levels and one million expanded
operations. No host pointers or guest GPU method streams are stored in the command handles.

The payload subset binds display render targets, program stages, vertex buffers and typed
render state, records full-channel RGBA clears with optional scissors and records array draws
and supported ordering/cache barriers. Views, depth/stencil targets, partial-channel clears,
indexed/indirect draws, copies and dispatch remain unsupported and stop explicitly. Scratch
bindings retain validated state for future shader execution; they do not allocate scratch storage.

QueueSubmitCommands snapshots and validates the complete handle array before enqueueing any
prefix, including device identities and cross-page faults. Queued sets retain their resources
and block premature command-pool retirement. Flush expands the typed sets in order, batching
image clears/draws/barriers into a Vulkan command buffer between fence markers. Image locks remain
held through GPU submission, and real completion fences retain their image lifetimes. Submitted
recordings also guard their pools while a deferred prefix or GPU completion remains pending. Recording,
clear execution and headless presentation upload/read back no pixel buffers. Explicit diagnostic
reads and configured dumps still copy pixels out and are counted separately.

Native display pixels currently live in retained Vulkan images. GPU image writes are not yet
made coherent with raw mapped guest-pool bytes; general CPU image access, aliasing and texture
flush/invalidate/read/write operations remain open. The synthetic clear validation reads the
actual Vulkan image through an explicit diagnostic helper. It does not validate mapped texture
storage coherence, which must be completed with general resource execution.

## Program binaries and translation

Programs own an empty or linked native object. Builders preserve the observed device, padding,
shader/hash arrays and sample-shading fields; ray-tracing modules/flags and subroutine linkage
remain unsupported. SetShaders, SetShadersExt and InitializeFromBuilder validate all inputs
before installing new binaries. A bad last stage or cross-page fault preserves the previous
link. Debug-hash overrides and the consumed control metadata are copied, with no retained
caller pointers or embedded compiler source/intermediate dumps.

The current binary subset accepts control versions 201.3 through 201.12, bounded GPU data in
owned executable pools, vertex/fragment headers of version 4, and headerless compute code.
Other stages/formats stop explicitly. Each successful link captures code once. Native module
preparation uses the existing sm_86 control-flow discovery and translator with the actual
graphics variation or compute workgroup/shared-memory/image classes. Cache hits check only
constant words used to resolve indirect branches; they do not reread or hash guest code. A
changed branch table or variation creates another module. Failed translations never become
cached successes. Modules retain their Vulkan device through their last owner.

BindProgram records the current linked stages and sample-shading mode, retaining their pool
generations. Relinking/finalizing the original program does not change existing recordings.
Stage masks support partial replacement and null-program unbinding. Active recordings and
queued sets guard their pool lifetimes; pool retirement invalidates cached command handles.
Queue flush updates bindings without submitting GPU work for a bind alone. Shader-resource
binding and wider graphics/dispatch execution remain to complete. Dynamic shader-code invalidation
barriers stop explicitly until writable shader storage is connected to execution; they cannot
silently reuse the immutable link snapshot after a caller requests invalidation.

The layouts, ownership policy and remaining limits are recorded in A-299 through A-332.

## Render state and array execution

Vertex attributes/streams, blend, channel-mask, color, depth/stencil, polygon and multisample
objects preserve their observed packed layouts and output widths. Recording copies the
consumed state and arrays, including viewport/scissor/depth-range slots, blend constants,
stencil masks/reference and primitive restart. Unsupported execution modes are refused;
owning their state does not establish execution support. FP depth-range arguments come from
FP registers and clamp to [0, 1]. Clears obey scissor zero, ignore the viewport and capture the
device origin at recording. Negative scissor origins clip to the image using wide arithmetic.

DrawArrays/Instanced currently support points, lines, line strips, triangles, triangle strips
and fans, with one single-sample display color target and vertex/fragment stages. Native
Vulkan pipelines consume the retained sm_86 modules and cache the consumed vertex/fixed state.
Basic blending, constants, logic, write masks, face culling, viewport/depth range and scissor
are connected. Lower-left viewports reverse height; culling and fragment-coordinate Y follow
that origin, including pixel centers. Instance IDs include the base and advance within one
Vulkan instanced draw. Vertex formats 1..48 and divisors zero/one are supported. Descriptor or
global-memory shaders, depth/stencil testing, multiple targets, packed/scaled formats, larger
divisors, advanced blending and multisample execution remain explicit stops.

Vertex bindings retain owned pool/device generations. Flush captures checked backing spans;
the worker resolves their data after earlier waits and refuses a changed host mapping.
It copies only the required prefix on a miss. The cache arms existing CPU page-write tracking
before its snapshot; unchanged mappings/pages reuse the GPU buffer without hashing, copying or
uploading the bytes again. A rewrite creates a fresh device-local version, leaving earlier GPU
submissions' buffers alive through their fences. Staging uploads are asynchronous; completed
staging resources are reaped without a draw-path wait. Native budgets cap each input at 16 MiB
and each queue cache at 64 MiB/4,096 entries; retained in-flight versions can outlive eviction.
Unbound fixed state uses the API defaults, and unbound viewport/scissor use the full target.
These are native initialization policies, not wider hardware-state coverage.

## Shared events and buffer visibility

Event builders preserve the consumed 16-byte layout. Native objects preserve the consumed
40-byte header, using its unused last word as an identity cookie. Initialization validates an
aligned four-byte range in an ordinary owned pool and leaves the stored value intact. CPU
reads and writes are naturally aligned atomic word accesses; CPU ADD follows the observed
load/add/store sequence, with wrapping arithmetic and a final memory fence. It is not an
atomic read-modify-write operation. Builder output aliasing and object/pool generations are checked.

Event storage imports its stable guest host page with HOST_VISIBLE and HOST_COHERENT required.
No shadow value or readback copy supplies event status. GPU WRITE/ADD run small internal atomic
kernels against that actual word, followed by a host visibility dependency. Default signals
release preceding memory writes with an ALL_COMMANDS dependency; NO_WAIT_MEMORY uses the
requested top, vertex or bottom source stage. Current vertex execution covers the supported
vertex-only graphics pipeline. Tests establish storage updates, wrapping and cross-queue
dataflow; precise pipeline-release timing against the original driver remains unverified.
Imports requiring alignment above a guest page and virtual/physical event pools remain refused.

Recording snapshots event bindings, so finalizing the event/recorder leaves queued work intact.
A ledger retains words exposed to GPU event operations until their pool retires. A vertex read
aliasing one such word binds the coherent import directly; it cannot reuse a stale device-local
snapshot after an untracked GPU write. Aliases must fit one shared page, and fragmented/larger
aliases stop explicitly. Ordinary inputs retain their device-local cache. GPU shader resources,
general buffer/image aliases, broader CPU alias tracking and long-run write-epoch rollover still
need completion/validation; this event subset does not establish general storage coherence.

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
budget. It constructs three display textures and a native window with three 3840×2160 Vulkan
images, initializes its first command buffer and callbacks, creates a native program, and captures
its vertex/fragment binaries and passes the render-state constructors before stopping at
`nvnTextureWriteTexels`. Its report
confirms five events, four textures, one window, one command buffer, one program, two binaries (1,536 bytes)
and three GPU clears, with zero translated title modules, recorded sets, native queue submissions,
presentations or display readback bytes. Vertex copies/uploads and executed state updates are
also zero at this boundary. It also constructs/maps its first native buffer. Recording, callback
execution and queued payloads currently have synthetic evidence; the real program stops before
using them. Virtual mappings have not yet been reached by this real-program HLE probe.

The synthetic window test checks builder padding/interval encoding, FP/stack metadata arguments,
checked outputs, GPU-backed acquisition, repeated presentation/reuse, active-buffer changes,
crop reset, uninitialized availability-sync construction, device mismatches, invalid ownership
and lifetime guards. It clears three retained images red/green/blue on the GPU, checks every
read-back pixel and inspects saved PNGs. Per-image counters verify that headless presentation
adds zero pixel-readback bytes; the subsequent diagnostic reads are counted separately. A
display-service test checks shared frame numbering and process frame limits. These synthetic
clears do not establish native title rendering or a gameplay I/O/performance improvement. The
window tests also pass with Vulkan required; opt-in real-title differential tests are outside
the standard gate used here.

All 56 NVN2-focused runtime tests pass with Vulkan required. Six command tests cover callback
reentry, preserved caller state and callback side effects, disabled/insufficient allocation,
faults/unwinds, inline begin/queries, recorder and handle lifetimes, called versus copied sets,
memory reuse, cross-device rejection, atomic handle-array validation and cancellation of
fences whose control storage was overwritten. The GPU test records thirty color clears into
one command set, finalizes the recorder, submits the set and a fence, then flushes. It verifies
two native queue GPU submissions: one batch of thirty clears, followed by the fence marker.
It presents the acquired image, checks every diagnostic pixel is green and inspects its PNG.
The caller's color/target arrays are overwritten before submission to verify recording copied
their values; upper integer argument bits are also checked through actual GPU execution.
Readback counters remain zero until that explicit diagnostic. This is native synthetic command
execution, not evidence of a native title frame or a measured gameplay speed-up.

Four further runtime tests check builder padding, shader/hash capture, atomic relinking,
cross-page faults, executable storage, unsupported controls/stages, retained code after caller
memory changes, cached graphics modules and recorded bindings surviving relink/finalization.
Three native backend tests check variation reuse, branch-table-dependent invalidation, failed
translation rejection and real dispatch of a translated compute module. The dispatch uses a
synthetic EXIT kernel and its real fence; it does not establish resource-bearing title compute
or draw execution. All seven new tests run with Vulkan required.

Eleven further runtime tests cover the packed state ABI, output widths/faults, FP depth-range
arguments, incomplete arrays, native array graphics, fixed state and input residency. Six use
real Vulkan execution with SPIR-V validation required. Synthetic vertex and fragment binaries
draw a green triangle over blue; diagnostic pixels and its external PNG were checked. Replaying
the unchanged input keeps cumulative CPU copies/GPU uploads at 48 bytes and adds one cache hit.
Rewriting the watched vertices changes the pixels and raises both totals to 96 bytes. Program
and recorder finalization do not invalidate the retained execution; headless presentation adds
no readback. Other GPU tests check culling in both origins, blend constants, write masks, logic
operations, nonzero-base instancing, fragment Y pixel centers, out-of-range inputs and every
pixel of a clipped scissored clear. That clear ignores a 1×1 viewport and preserves recorded
state after caller memory changes. These are synthetic results, not native title rendering or
a gameplay I/O/performance improvement.

Six further runtime tests and four backend tests cover events/deferred execution with Vulkan
and SPIR-V validation required. They check the CPU ABI, preserved initial values, wrapping GPU
ADD, actual storage writes, cross-queue GPU wakeups, inline CPU stores without notifications,
finalization while another thread signals, in-flight pool guards, failed initialization and
cancelled preparations. QueueWaitSync is tested with a flushed source still blocked at an event.
Duplicate enqueue rejection preserves the recording ticket; a queued reset followed by an
initially matching wait is checked across all locations/flags. One draw test first caches a
48-byte triangle input, flushes a wait plus another draw, then creates a new event pointing at
the first vertex. Another queue changes that vertex on the GPU and releases the wait. Pixels
verify the changed geometry; cumulative CPU copies/GPU uploads stay at 48 bytes. This covers
the shared-page alias path without CPU copyback, not general shader/image coherence.

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
| Windows and display images, off | 10.83 | 11.62 |
| Typed command recording, off | 11.53 | 11.66 |

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

The windows/display-image off-mode check completed 3,600 frames in 369 seconds, exit zero.
Frame 3,500 was inspected: road, starting grid, countdown 2 and HUD drawn, with dark lighting,
black characters and large foreground geometry remaining. The earlier intermittent black-road
issue remains open; this one capture does not isolate it. The renderer refused 15 texture-type
draws and one unsupported-format compute launch. Builds/tests overlapped this diagnostic run,
so its timing does not establish a performance change or a full rendering-regression pass.

The typed-command off-mode check completed 3,600 frames in 361 seconds, exit zero. Frame 3,500
was inspected: the road and HUD are drawn, the start lights are green, and a large black band
crosses the middle of the road. Black road markings, dark structures/characters and foreground
geometry remain. This confirms the intermittent road issue is still present; the different
start phase prevents a matched visual comparison with the prior capture. The renderer refused
15 texture-type draws and one unsupported-format compute launch. Builds/tests overlapped this
diagnostic too; these samples establish neither a performance gain nor a rendering-regression
pass.

The program-binary off-mode check completed 3,600 frames in 399 seconds, exit zero. Frame 3,500
was inspected: the start's GO overlay, road and HUD are visible, with dark structures/characters
and large foreground geometry still present. This different start phase and overlay do not
isolate the earlier road issue. Builds/tests overlapped the capture; its timing establishes
neither a performance gain nor a rendering-regression pass.

The state/array-draw off-mode check completed 3,600 frames in 370 seconds, exit zero, without
concurrent builds/tests. Frame 3,500 was inspected: race road and HUD are visible at countdown
2, with dark character/structure regions still present. The last 100 frames at 3,400, 3,500 and
3,600 took 12.572, 12.378 and 11.854 seconds respectively. This checks the rebuilt default path
through the race start; the different start phase prevents a matched rendering/performance
comparison. It establishes no native gameplay I/O or rate improvement.

The event/deferred-queue off-mode check completed 3,600 frames in 341 seconds, exit zero, with
no concurrent builds/tests. Frame 3,500 was inspected: GO overlay and HUD are visible, together
with a large black road region and dark character/structure regions. The road issue remains
unisolated; the capture does not establish a rendering-regression pass. The last 100 frames at
3,400/3,500/3,600 took 10.187/10.541/11.850 seconds. These timings are for the default path and
different race-start phases, and establish no native gameplay I/O or rate improvement.

On Windows, `cargo fmt --all --check`, `cargo lint` and `cargo test` pass: 1,229 tests passed and
one was ignored.

## Next

1. Implement TextureWriteTexels and general image residency/coherence at the next initialization
   boundary. Complete shader resource bindings, remaining texture modes, program control/stage
   coverage and dynamic code invalidation.
2. Extend typed recording/execution to depth/MRT/indexed/indirect draws, copies/dispatch and
   views. Complete resource visibility beyond retained display images and shared event pages.
3. Implement vertex-only sync completion and validate precise event release points. Complete
   broader buffer/CPU aliases, write-epoch rollover, sparse access and in-flight resource
   lifetimes. Validate native title frames and compare race I/O/rate before enabling HLE by default.

The traced route reaches 247 API functions on the path into the race. That is an observed minimum,
not complete coverage of NVN2 or of later gameplay.
