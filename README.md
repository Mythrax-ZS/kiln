# recomp

A static recompiler for Horizon OS (Nintendo Switch) AArch64 programs. It reads the **already
extracted ExeFS and RomFS** of a title the owner dumped from their own console (or a homebrew NRO)
and produces:

- a native executable (PE on Windows, ELF on Linux) for the guest program, and
- a shared library exposing the same entry points for embedding.

Both link against `nx_runtime`, the crate that holds the guest CPU model and the Horizon emulation
layer. The tool never reads `prod.keys` and contains no NSP, NCA or decryption code. No firmware,
keys or title data are stored in this repository.

## Status

Work proceeds in milestones M1 to M8. Each milestone has a report under `docs/reports/` that states
what was implemented, what was tested and how, and what was not verified. `docs/ARCHITECTURE.md`
explains how the pieces fit together.

Done: M1 (input layer), M2 (loader), M3 (decoder and interpreter, integer ISA), M4 (lifter and
LLVM code generation, integer ISA), M5 (floating point and Advanced SIMD in all four layers), M6
(the Horizon kernel layer: a homebrew NRO built by `recomp build` runs natively, prints through
`svcOutputDebugString`, starts threads and talks to `sm:`), M7 (a title with a RomFS boots on that
layer, through its own `rtld` and SDK start-up, until it asks for the GPU driver; the
cryptographic extension was added on the way), M8 (the GPU front, `nx_gpu`: the driver service and
its device nodes, the display and buffer queue, a Maxwell command processor with a macro engine
that traces what a title sends; the same title now runs its main loop and submits draw calls).

**Nothing renders.** The command processor decodes and counts what the title sends (in 600 frames:
1,201 submissions, 30,641 macro calls, 138,125 draws), but no shader is read, no picture is made
and no sound or input reaches the title from a person. Shader translation and a Vulkan backend are
the next, separate work. Nothing here is a claim that a game is playable.

## Using it

```
recomp inspect --exefs <dir|pfs0> [--romfs <dir|image>]
recomp analyze (--exefs <dir|pfs0> | --nro <file>) [--hints <file>] [--main-only]
recomp build   (--exefs <dir|pfs0> | --nro <file>) --out <dir> [--name <name>] [--opt <0-3>]
               [--hints <file>] [--shared] [--keep-ir] [--main-only] [--romfs <dir|image>]
               [--reuse-objects]
recomp relink  --out <dir> --name <name>
recomp symbols (--exefs <dir|pfs0> | --nro <file>) [--match <text>] [<address>...]
recomp doctor
```

`inspect` also says which of the SVCs and services a program may use this runtime does not implement.
`doctor` says what the tool can use on this machine (its version, the C compiler it will run, the
runtime library programs link) as `key: value` lines, and builds nothing.

### The desktop front end

`recomp-gui` (the `recomp_gui` crate, built with `cargo build --release -p recomp_gui`) wraps the
command line in a window: pick a game's already extracted folder (the one that holds `exefs` and
`romfs`, or a homebrew `.nro`) or drop it on the window, choose how hard the compiler works, and
press Recompile. The output of `recomp` streams into the window; the finished game appears in the
Library, which keeps one folder per game and the settings each is run with (frame limit, save data
and RomFS locations, the logging and GPU trace switches, extra environment variables), and starts
the game with the output shown beside them. The Settings screen names the `recomp` program, the C
compiler and the runtime library to use; the header shows whether each was found. The GUI finds
`recomp` beside itself or on the `PATH`. It reads no keys and opens no NSP, XCI or NCA files, as
the command line does not. `RECOMP_GUI_CONFIG_DIR` moves its settings folder, and
`RECOMP_GUI_SCREENSHOT=<file.png>` makes it save a picture of its window and exit (with
`RECOMP_GUI_TAB` and `RECOMP_GUI_GAME` choosing what is shown), which is how its looks are checked.

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
`GetInfo` entropy) reproducible. `recomp relink` links the objects of an earlier build against the
current runtime and `--reuse-objects` skips recompiling unchanged code, which keeps the loop of
trying a title short.

For the GPU: `NX_TRACE_GPU=1` prints every submission and, when the run ends, what the command
processor counted (`NX_GPU_TOP=<n>` sets how many of the busiest methods it lists, 0 lists all);
`NX_TRACE_GPU=2` also prints every method write and macro call; `NX_MAX_FRAMES=<n>` ends the run
after `n` presented frames; `NX_SAVE_DIR=<dir>` is where a program's save data goes (default
`./nx_save`). `recomp symbols --exefs <dir> [--match <text>] [<address>...]` names the function
around an address (the return addresses a stop report ends with), which is how a stop is read.

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

## License

Not chosen yet; that is the owner's decision. yuzu is GPL and no code was copied from it; the
implementation is written from the public specifications and from its own tests.
