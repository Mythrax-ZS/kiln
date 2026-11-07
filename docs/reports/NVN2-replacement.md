# NVN2 replacement: entry point

Started 2026-11-07. The objective is to replace the guest graphics driver and remove the GPU
front's command decoding from its draws. This first step establishes the API boundary. HLE does
not yet create a device or render a frame, and does not improve the race's rate.

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
- `NX_NVN2=hle` owns procedure resolution and implements `nvnDeviceBuilderSetDefaults` and
  `nvnDeviceBuilderSetFlags`. An unimplemented entry stops at its first call, with its name and
  calling address. No API falls back to the guest driver in this mode: their state cannot be
  mixed. Native sm_86 translation remains the intended shader path.

Enabled modes require `Npdm::platform == Nx2`. The module descriptor advertises whether its
loader stub was built; an older descriptor refuses an enabled mode. Building once with
`--reuse-objects` recompiles the loader's unit and refreshes the descriptor, retaining the other
units. A runtime-only relink is sufficient for subsequent implementation changes.

`NX_NVN2_REPORT=<file>` writes a text report when the process ends: loader calls, resolved
procedures, and each procedure's count, original guest address (zero for HLE), and API name.
Keep reports from real programs outside the repository. A killed process leaves the initial
`running` marker instead of a completed report.

## Validation

The runtime tests check forwarding of integer and FP/vector arguments and results, flags, stack
and TLS; stable tokens, different device entries, null resolution, guest unwinds and stops;
checked names and builder memory faults; and a missing HLE entry stopping without a fake result.

Synthetic NSO executables exercise direct loader calls and indirect procedure calls in all
three modes. A guest setter writes 7 into a builder; off and trace return that value, while HLE
clears it and returns 0. Another executable calls an unimplemented entry and must report its
name and stop. Corrupting the loader's old object before `--reuse-objects` verifies it is rebuilt.

The owner's NX2 program resolves 731 procedures in HLE mode, runs both builder setters once,
and stops at `nvnDeviceInitialize`. This confirms the boundary is reached before device
creation. It is not a rendering or performance result.

Trace mode completes the same scripted route through the race to frame 3,600, with three loader
calls, 733 resolved procedure names and 59,857,523 calls to 247 distinct APIs. It sees every name
in the earlier 246-function census, plus 15 calls to `nvnMemoryPoolBuilderSetNativeHandleNVX`.
That extension is resolved after device initialization; the new boundary captures it without a
predefined procedure map. The off-mode and trace-mode dumps at frame 3,500 were inspected: both
show the road drawn. This covers this route through the race.

The same device, input script, frame dumps every 250 frames and `NX_TRACE_PERF=1`, seconds per
100 frames:

| Build and mode | At frame 3,400 | At frame 3,500 |
|---|---:|---:|
| Original revision `72bf797` | 10.82 | 12.94 |
| New build, off | 10.66 | 10.93 |
| New build, trace | 10.71 | 11.20 |

These are single runs with substantial timing variation. The original runtime was built from
that revision and linked with the original objects; its embedded guest image was checked to be
byte-identical to the new build's. The samples show no slowdown in the new off path in these
runs. No speed-up is attributed to this entry-point work; HLE still stops before rendering.

On Windows, `cargo fmt --all --check`, `cargo lint` and `cargo test` pass: 1,170 tests passed and
one was ignored.

## Next

1. Device creation, capabilities and object sizes; memory-pool and buffer lifetimes, with checked
   guest mappings and a single native owner of device state.
2. Programs, textures, samplers, descriptor pools and render state. Program creation must retain
   sm_86 programs for the existing native translator.
3. Command buffers that record draw-level operations and preserve API ordering, callbacks,
   indirect arguments, memory visibility and events.
4. Native queues, syncs and window presentation, connected to the renderer without replaying
   method streams. Validate frames and time the race before enabling HLE by default.

The new trace reaches 247 API functions on the path into the race. That is an observed minimum,
not complete coverage of NVN2 or of later gameplay.
