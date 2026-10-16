<p align="center">
  <img src="assets/brand/kiln-banner.png" width="100%"
       alt="Kiln: a brick kiln with a fire in it, the name in fire colors, and Cinder, a black salamander with yellow spots">
</p>

# Kiln

A static recompiler for Horizon OS (Nintendo Switch) AArch64 programs. It reads the **already
extracted ExeFS and RomFS** of a title the owner dumped from their own console (or a homebrew NRO)
and produces:

- a native executable (PE on Windows, ELF on Linux) for the guest program, and
- a shared library exposing the same entry points for embedding.

Both link against `nx_runtime`, the crate that holds the guest CPU model and the Horizon emulation
layer. The tool never reads `prod.keys` and contains no NSP, NCA or decryption code. No firmware,
keys or title data are stored in this repository.

## Status

Work proceeds in milestones M1 to M10. Each milestone has a report under `docs/reports/` that states
what was implemented, what was tested and how, and what was not verified. `docs/ARCHITECTURE.md`
explains how the pieces fit together.

Done: M1 (input layer), M2 (loader), M3 (decoder and interpreter, integer ISA), M4 (lifter and
LLVM code generation, integer ISA), M5 (floating point and Advanced SIMD in all four layers), M6
(the Horizon kernel layer: a homebrew NRO built by `kiln build` runs natively, prints through
`svcOutputDebugString`, starts threads and talks to `sm:`), M7 (a title with a RomFS boots on that
layer, through its own `rtld` and SDK start-up, until it asks for the GPU driver; the
cryptographic extension was added on the way), M8 (the GPU front, `nx_gpu`: the driver service and
its device nodes, the display and buffer queue, a Maxwell command processor with a macro engine
that traces what a title sends; the same title now runs its main loop and submits draw calls), M9
(shader translation, `nx_shader`: the Maxwell programs the title binds translate to SPIR-V and are
compared with Rust arithmetic by tests on a GPU) and M10 (the Vulkan backend: `nx_vulkan` draws
what the command processor decodes, with ASTC textures decoded on the host, shows the frames in a
window, and takes keyboard and gamepad input).

**First frames.** Over its first 1,500 presented frames the title's 380,376 draws and
1,499 clears run on a Vulkan device and none fails; the frames, dumped as PNG files and looked at,
show its first screens (an introduction screen with photographs, a logo and text, then a menu with
artwork), in 23 s of wall clock on an RTX 5070 Ti (`docs/reports/M10-vulkan-backend.md`). With
`NX_WINDOW=1` the same frames appear in a window, and the keyboard (or an XInput gamepad) reaches
the title: the keys that stand for the directional pad and for the left stick moved the selection
of its menu, and confirming it started the title's introduction: a story card and then its opening
cutscene (characters lit and in color, a flame effect, subtitles), which was looked at in dumped
frames. After the cutscene the title loads its first area, and scripted input (`NX_INPUT_SCRIPT`)
turns its camera and makes its character stand up and run, seen in dumped frames, with its interface
(the counters with their numbers, the life gauge, the compass and the prompts of the controls); a scripted run
with random inputs went through the menu and the first area for 120,000 presented frames (about
30 minutes, flat memory), through the character's death (the screen closes on it, loads for a
second and the game goes on) and ended at its frame limit. On that card the title runs at 40 to 60 frames
a second in its first area (49 in the heaviest scene seen, about 27 in the cutscene, whose 2,800
draws a frame are the limit of the renderer thread). Its music and sounds are decoded, mixed and
played (voices, ADPCM, mixes and the device sink; no reverb or other effects), but only measured
(`NX_AUDIO_DUMP=<file>` writes what is played to a WAV file), never listened to by a person. The
title sometimes stops early: rare, and the report says what is known. Nothing here is a claim that
a game is playable; what a person finds past the first area has not been tried.

**A second title.** A second title (a platformer with 3D characters in 2D courses) boots on the same
runtime. Frames that were dumped and looked at show its start-up screens, the screen where a user is
chosen, a story narration, its first plaza with the whole cast (a castle, hills, clouds, flowers, the
characters lit and in color), a cutscene of a giant face above the cast, a golden transition effect
and its first course, with scripted input (`NX_INPUT_SCRIPT`). A run of 7,200 presented frames ended at
its frame limit with no draw and no compute launch failed, and the 100 frames from frame 4,800 on took
2.9 to 3.5 s on the RTX 5070 Ti (3,300 draws a frame in the plaza). With a game that had progress
saved, a run of 15,900 presented frames (7 minutes, scripted input) went from its title through the
map of its first world and a course card into a course (a tree, a pipe, blocks, enemies, the
counters) and back to the map after the lives ran out, with no draw failed and nothing stalled
(frames dumped every 500 were looked at). That run first stopped twice at the course, and both causes
are fixed: the audio renderer answered an effect with the layout of an older revision, so the title
waited for ever for effects that were gone (A-127), and the pixels of a render target drawn to a
few frames earlier were put over data that the title had stored in the same memory (A-171).
Looking at its frames found what
the code had wrong: the range of clip-space depth is the register that says so (docs/ASSUMPTIONS.md
A-096), the copies of the 2D engine were dropped (A-165), programs in the last megabyte of a mapping
were not read (A-168), some compute and pixel programs branch on the condition code or use shared memory
(A-169, A-170), and a render target that nothing drew to for a while lost its pixels before they reached
the memory that a later draw samples (A-171), which left the text of an interface blank (also in the
first title). What is not right: the leaves that fell in the course after the plaza had black blobs
in one run (a lit material that came out black, cause not found; the frames of the saved game's
course that were looked at show none) and its sound was not checked. Every program cut out
of the shader archives of the two titles that looks like one translates to SPIR-V that `spirv-val`
accepts (23,341 of 23,348 candidates; five are not programs and two use an attribute pointer form that
is not translated). Nothing here is a claim that a game is playable.

### Keys

With the window focused, the keyboard is the console's handheld controls (the keys are the
physical ones, so the layout of the keyboard does not matter):

| Control | Key |
|---|---|
| Left stick | W A S D (click: Left Shift) |
| Right stick | Arrow keys (click: Right Shift) |
| A, B, X, Y | L, K, I, J (X above, Y left, B below, A right) |
| Directional pad | T up, F left, G down, H right |
| L, R | Q, E |
| ZL, ZR | Z, C |
| Plus, Minus | Enter, Backspace |

`NX_KEYMAP=<file>` rebinds them: one `Action=Key` line per binding (`A=Space`, `LeftStickUp=KeyW`;
more lines for one action add keys; the actions are `A B X Y L3 R3 L R ZL ZR Plus Minus DpadLeft
DpadUp DpadRight DpadDown` and `LeftStick`/`RightStick` with `Left Right Up Down`; keys are named
as `KeyW`, `Digit1`, `ArrowUp`, `Space`, `Numpad4`, `F5`, in any case). `NX_TRACE_KEYS=1` prints
the name of each key pressed, which is how to find the names. An Xbox-style (XInput) gamepad on Windows works
without setup, with its face buttons by position (the bottom one is B, as on the console;
`NX_GAMEPAD_LABELS=1` keeps the names instead), and `NX_GAMEPAD=0` leaves it out. Only the
handheld controls exist, so a title that does not support the handheld mode sees no controller.
For a run without a person, `NX_INPUT_SCRIPT="<frame> <Action> [<frames>];..."` (or the name of a file
holding that) presses the actions named above (`A`, `DpadDown`, ...) at presented frame numbers, for
that many frames (default 6): `NX_INPUT_SCRIPT="2500 A 6"` presses A at frame 2500.

## Using it

```
kiln inspect --exefs <dir|pfs0> [--romfs <dir|image>]
kiln analyze (--exefs <dir|pfs0> | --nro <file>) [--hints <file>] [--main-only]
kiln build   (--exefs <dir|pfs0> | --nro <file>) --out <dir> [--name <name>] [--opt <0-3>]
               [--hints <file>] [--shared] [--keep-ir] [--main-only] [--romfs <dir|image>]
               [--reuse-objects]
kiln relink  --out <dir> --name <name>
kiln symbols (--exefs <dir|pfs0> | --nro <file>) [--match <text>] [<address>...]
kiln doctor
```

`inspect` also says which of the SVCs and services a program may use this runtime does not implement.
`doctor` says what the tool can use on this machine (its version, the C compiler it will run, the
runtime library programs link) as `key: value` lines, and builds nothing.

### The desktop front end

`kiln-gui` (the `kiln_gui` crate, built with `cargo build --release -p kiln_gui`) wraps the
command line in a window: pick a game's already extracted folder (the one that holds `exefs` and
`romfs`, or a homebrew `.nro`) or drop it on the window, choose how hard the compiler works, and
press Recompile. The output of `kiln` streams into the window; the finished game appears in the
Library, which keeps one folder per game and the settings each is run with (window and sound, keys, frame limit, save data
and RomFS locations, the logging and GPU trace switches, extra environment variables), and starts
the game (drawn in a window with the sound on, unless the settings say otherwise) with the output shown beside them. The Settings screen names the `kiln` program, the C
compiler and the runtime library to use; the header shows whether each was found. The GUI finds
`kiln` beside itself or on the `PATH`. It reads no keys and opens no NSP, XCI or NCA files, as
the command line does not. `KILN_GUI_CONFIG_DIR` moves its settings folder, and
`KILN_GUI_SCREENSHOT=<file.png>` makes it save a picture of its window and exit (with
`KILN_GUI_TAB` and `KILN_GUI_GAME` choosing what is shown), which is how its looks are checked.

`build` writes `<name>.exe` (or `<name>`), optionally `<name>.dll` with `--shared`, the embedded
image, and `<name>.report.txt` listing everything the analysis could not settle: indirect branches
no jump table explains, interpreter fallbacks by address range, and instructions that stop the
run. A produced program that stops says why on stderr and exits with status 101. Addresses it
reached that no function covers can be logged with `NX_UNKNOWN_LOG=<file>` and fed back with
`--hints`.

A produced NSO or NRO runs on the project's own Horizon kernel layer. What the guest sends to
`svcOutputDebugString` goes to the program's standard output. An SVC, a service or a service
command nobody implemented stops the run: the report on standard error names the guest address and
the number, and the exit status is 101.

From an ExeFS, `analyze` and `build` take every NSO (`rtld`, `main`, `subsdk0-9`, `sdk`) and the
produced program starts in the first, the way the console starts a title; `--main-only` takes `main`
alone. A title's RomFS is given with `--romfs` when building (the produced program remembers the
path) or with `NX_ROMFS=<dir|image>` when running. Environment variables that help while
debugging: `NX_TRACE_SVC=1` logs every supervisor call and every IPC request with its result on
standard error, `NX_TRACE_LOG=1` prints what the program logs through the log manager, and
`NX_RANDOM_SEED=<decimal number>` makes the kernel's pseudo-random values (loader configuration,
`GetInfo` entropy) reproducible, and `NX_WATCHDOG=<seconds>` says what every guest thread is doing
when no frame has been presented for that long. `kiln relink` links the objects of an earlier build against the
current runtime and `--reuse-objects` skips recompiling unchanged code, which keeps the loop of
trying a title short. A relink keeps the code the earlier build generated: a change of the code
generator or of the analysis (a floating-point form lowered to native code, say) shows only after
a new build.

For the GPU: `NX_TRACE_GPU=1` prints every submission and, when the run ends, what the command
processor counted (`NX_GPU_TOP=<n>` sets how many of the busiest methods it lists, 0 lists all);
`NX_TRACE_GPU=2` also prints every method write and macro call; `NX_MAX_FRAMES=<n>` ends the run
after `n` presented frames; `NX_SAVE_DIR=<dir>` is where a program's save data goes (default
`./nx_save`). `kiln symbols --exefs <dir> [--match <text>] [<address>...]` names the function
around an address (the return addresses a stop report ends with), which is how a stop is read.

To draw, run a produced program with `NX_RENDERER=vulkan` (it needs a Vulkan 1.1 device).
`NX_WINDOW=1` shows the frames in a window (`NX_WINDOW_SIZE=<width>x<height>` sets its size,
1280x720 by default, `NX_WINDOW_TITLE=<text>` its title, and `NX_PRESENT_MODE=mailbox` or
`immediate` replaces the vertical sync that paces the program); closing the window ends the
program. `NX_DUMP_FRAMES=<dir>` with `NX_DUMP_EVERY=<n>` writes every n-th frame as a PNG file
instead of, or as well. `NX_VULKAN_VALIDATE=1` turns the validation layer on. The keys are listed
above, under "Keys".

## What "accuracy" means here

Static recompilation cannot reproduce cycle timing or GPU output, and this project does not claim
either. Instruction behavior is checked against the Arm A-profile pseudocode with randomized
differential tests (lifted native code against the interpreter, and the interpreter against
Unicorn). The measured coverage is reported in the milestone reports, and every assumed or
undocumented behavior is listed in `docs/ASSUMPTIONS.md` with its source.

## Building

```
cargo build --release
cargo fmt --all --check
cargo lint
cargo test
```

Generating native outputs needs clang with the x86-64 target. See `CLAUDE.md` for how tests find
clang and Unicorn.

### Continuous integration and the bundle

`.github/workflows/ci.yml` runs on every push to `main`, on pull requests and by hand:

- **windows**: `cargo fmt --all --check`, `cargo lint` and `cargo test` (with Unicorn 2.1.4 as the
  oracle of the instruction tests and the runner's clang for the programs the tests build), then
  the library tests again with `KILN_REQUIRE_EXTERNAL=1`, so that a missing tool fails the run
  instead of skipping.
- **linux**: the same lint and tests for everything but `kiln_gui` (the ELF output and the
  runtime on Linux).
- **bundle** (after **windows**): `cargo build --release --workspace`, then `tools/bundle.ps1`
  makes `kiln-<version>-<commit>-windows-x64.zip` from `kiln-gui.exe`, `kiln.exe` and
  `nx_runtime.lib` with a README, `doctor.cmd`, `selftest.cmd` (it compiles and runs a tiny
  synthetic program, no game needed) and `setup-toolchain.cmd` (it starts Microsoft's installer
  for the clang and the Windows libraries a build needs; clang is not in the bundle), and the job unpacks the zip somewhere else, with no `cargo`
  on the machine, and runs the self-test. A push to `main` that passes keeps the zip (with
  `SHA256SUMS.txt`) for good: the release makes a tag `build-<run number>-<commit>` on that commit
  and a pre-release (never marked "latest"); pushing a tag `v*` by hand makes a normal release of
  it instead. A pull request or a manual run keeps it as the artifact `kiln-windows-x64` for 30
  days.

- **bundle-linux** (after the three above): the same for Linux, `tools/bundle.sh`: a
  `kiln-<version>-<commit>-linux-x86_64.tar.gz` (`kiln-gui`, `kiln`, `libnx_runtime.a`, a
  README, `doctor.sh`, `selftest.sh`, `setup-toolchain.sh`) and an AppImage
  `kiln-<version>-<commit>-x86_64.AppImage` of the desktop front end (`appimagetool` and the
  AppImage runtime are downloaded at pinned versions and checked against their SHA-256; the
  AppImage runs the command line tool when its first argument is `kiln`). The job unpacks the
  tarball with no `cargo` on the machine and runs the self-test, and builds and runs the
  self-test program with the `kiln` inside the AppImage. Both files, with `SHA256SUMS-linux.txt`,
  go to the same tag and pre-release as the Windows bundle of that build. Clang is in neither
  bundle (a person installs it: `dist/linux/setup-toolchain.sh`); a game has not been run on
  Linux, and Linux has no sound output or gamepad.

What no runner has: a Vulkan device
 (the renderer's tests print `SKIPPED` and pass), `spirv-val`
and `astcenc`, and any game. The bundle holds no game, key or firmware; a person who gets it
brings the extracted ExeFS and RomFS of a game of their own (`dist/README.txt` is what they read).
To make the bundle by hand: `cargo build --release --workspace`, then
`KILN_SELFTEST_FIXTURE=<folder> cargo test -p kiln --lib -- --ignored --exact
build::e2e_tests::writes_the_selftest_program`, then
`pwsh -File tools/bundle.ps1 -Out <folder> -Selftest <the first folder>`.

## The look

The icon, the mascot and the lettering are pixel art drawn for this project and kept as text in
`kiln_gui/src/brand.rs`. Cinder is a fire salamander who lives in the kiln: black, with yellow
spots and a tail that glows at the tip. `cargo run -p kiln_gui --example brand` writes the pictures
under `assets/brand/` from that text (the icon as PNG files from 16 to 512 pixels and as `kiln.ico`,
the icon and Cinder as vector files, the logo, the banner above and a 1280 by 640 card for the
repository's social preview) and the icon of the AppImage, `dist/linux/kiln.png`;
`kiln_gui/build.rs` puts the icon into `kiln-gui.exe`.

## License

Not chosen yet; that is the owner's decision. yuzu is GPL and no code was copied from it; the
implementation is written from the public specifications and from its own tests.
