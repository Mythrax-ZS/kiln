# NVN2 replacement: devices and storage

Updated 2026-11-07. The objective is to replace the guest graphics driver and remove the GPU
front's command decoding from its draws. The optional API boundary now owns native devices,
memory pools and buffers. The owner's NX2 program creates its first native device and pool,
then stops at sampler-pool initialization. HLE does not yet render a frame or improve the race's
rate. The default remains off.

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

Flush/invalidate validate ranges and fence coherent host storage. There is no native queue
submission yet, so no native GPU write can be pending. Completion/writeback, resource lifetimes
during execution and sparse GPU access semantics must be connected before submissions are
accepted. Native-handle and allocator-backed pools remain unsupported. The owned renderer is
initialized but has no native API commands submitted to it yet.

The layouts, ownership policy and remaining limits are recorded in A-299 through A-303.

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

Synthetic NSO executables exercise direct loader calls and indirect procedure calls in all
three modes. A guest setter writes 7 into a builder; off and trace return that value, while HLE
clears it and returns 0. Another executable calls an unimplemented entry and must report its
name and stop. Corrupting the loader's old object before `--reuse-objects` verifies it is rebuilt.

The owner's NX2 program resolves 731 procedures in HLE mode, creates a real native Vulkan
device with flags zero, queries its profile, sets its origin/depth modes, and creates the first
native memory pool. It stops at `nvnSamplerPoolInitialize`, with the missing entry named. This
validates progress through initialization; the buffer and virtual-mapping paths have synthetic
coverage and have not yet been reached by this real-program HLE probe.

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

These are single runs with substantial timing variation. The original runtime was built from
that revision and linked with the original objects; its embedded guest image was checked to be
byte-identical to the new build's. The samples show no slowdown in the new off path in these
runs. The devices/storage off run also completed 3,600 frames; its frame 3,500 was inspected and
shows the race starting grid, road and HUD drawn. No speed-up is attributed to the native
initialization work; HLE still stops before rendering. This off run still refused 16 commands
with a texture-header/type mismatch and one unsupported surface format; those cases remain open.

On Windows, `cargo fmt --all --check`, `cargo lint` and `cargo test` pass: 1,178 tests passed and
one was ignored.

## Next

1. Programs, textures, samplers, descriptor pools and render state. Program creation must retain
   sm_86 programs for the existing native translator.
2. Command buffers that record draw-level operations and preserve API ordering, callbacks,
   indirect arguments, memory visibility and events.
3. Native queues, syncs and window presentation, connected to the renderer and owned address
   spaces without replaying method streams. Complete GPU visibility, sparse access and in-flight
   resource lifetime handling here. Validate frames and time the race before enabling HLE by default.

The traced route reaches 247 API functions on the path into the race. That is an observed minimum,
not complete coverage of NVN2 or of later gameplay.
