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

Work proceeds in milestones M1 to M7. Each milestone has a report under `docs/reports/` that states
what was implemented, what was tested and how, and what was not verified. `docs/ARCHITECTURE.md`
explains how the pieces fit together.

Done: M1 (input layer), M2 (loader), M3 (decoder and interpreter, integer ISA), M4 (lifter and
LLVM code generation, integer ISA), M5 (floating point and Advanced SIMD in all four layers; the
cryptographic extension is the one thing left out). Nothing renders: GPU work is out of scope until
M7 passes.

## Using it

```
recomp inspect --exefs <dir|pfs0> [--romfs <dir|image>]
recomp analyze (--exefs <dir|pfs0> | --nro <file>) [--hints <file>]
recomp build   (--exefs <dir|pfs0> | --nro <file>) --out <dir> [--name <name>] [--opt <0-3>]
               [--hints <file>] [--shared] [--keep-ir]
```

`build` writes `<name>.exe` (or `<name>`), optionally `<name>.dll` with `--shared`, the embedded
image, and `<name>.report.txt` listing everything the analysis could not settle: indirect branches
no jump table explains, interpreter fallbacks by address range, and instructions that stop the
run. A produced program that stops says why on stderr and exits with status 101. Addresses it
reached that no function covers can be logged with `NX_UNKNOWN_LOG=<file>` and fed back with
`--hints`.

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
