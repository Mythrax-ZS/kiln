# recomp

A static recompiler for Horizon OS (Nintendo Switch) AArch64 programs. It reads a title that the
owner dumped from their own console (NSP or NCA files) plus the owner's `prod.keys`, and produces:

- a native executable (PE on Windows, ELF on Linux) for the guest program, and
- a shared library exposing the same entry points for embedding.

Both link against `nx_runtime`, the crate that holds the guest CPU model and the Horizon emulation
layer. Keys, firmware and title data are never bundled, printed or logged, and none are stored in
this repository.

## Status

Work proceeds in milestones M1 to M7. Each milestone has a report under `docs/reports/` that states
what was implemented, what was tested and how, and what was not verified.

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
