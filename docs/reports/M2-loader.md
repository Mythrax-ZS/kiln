# M2: loader

Status: complete. Date: 2026-10-12.

## What was implemented

| Piece | Where | Notes |
|-------|-------|-------|
| LZ4 block decoder | `kiln::loader::lz4` | Bounds-checked; rejects zero or out-of-range offsets, runaway lengths and size mismatches. |
| NSO0 loader | `kiln::loader::nso` | Header, per-segment LZ4, SHA-256 check of each segment whose header flag asks for it, layout of text, rodata, data and bss at a fixed guest base. The zstd-based (ZBIC) flag is reported as unsupported. |
| NRO0 loader | `kiln::loader::nro` | Header at 0x10, flat segments, bss, optional `ASET` asset section (icon, NACP, embedded RomFS offsets). |
| MOD0 and relocations | `kiln::loader::module` | Follows MOD0 to `.dynamic`; applies RELA and PLT RELA tables (RELATIVE, ABS64, GLOB_DAT, JUMP_SLOT); collects exports, init and fini arrays, `DT_INIT`/`DT_FINI`. Imports no loaded module defines are zeroed and listed. |
| `.eh_frame_hdr` reader | `kiln::loader::ehframe` | Exact function starts and lengths from the unwind table; also RELATIVE-relocation targets in text as likely indirect-call entries. These feed function discovery in M4. |
| NPDM parser | `nx_runtime::npdm` | META, ACID, ACI0; service access list with glob matching and the server bit; kernel capabilities including the enabled-SVC mask. The runtime will enforce the service list and SVC mask. |
| AArch32 detection | NPDM `Is64BitInstruction` and address space (authoritative), plus a first-instruction heuristic | Stops with `Error::Aarch32`. |
| CLI | `kiln inspect` | Now describes every NSO, MOD0, relocation counts, exports, `.eh_frame` coverage and the NPDM. |

## Tests

`cargo test`: 31 tests in `nx_runtime`, 42 in `kiln`, plus 4 opt-in real-title tests; all pass.
`cargo fmt --all --check` and `cargo clippy --all-targets -- -D warnings` are clean.

- LZ4: eight blocks produced by the reference implementation (the `lz4` Python package,
  default and high-compression modes, including overlapping matches, extension bytes and a
  6000-byte block with a 3000-byte offset), a test compressor round-tripped over 200 random
  shapes, and malformed-block cases.
- NSO: all eight compress-flag combinations with and without hash flags decode to the same image;
  layout and segment table; relocations at two different bases; init/fini arrays, exports and
  known entries; unresolved strong and weak imports; unsupported relocation types; a module
  without MOD0; hash mismatch (caught only when flagged and enabled); malformed headers (short
  file, bad magic, ZBIC, segment outside the file, unaligned offsets, overlap, absurd sizes, wrong
  decompressed size, garbage compressed data); AArch32 first instruction; corrupt MOD0
  structures (dynamic section near the image end, `DT_REL`, relocation target outside the module,
  symbol index outside the table); `.eh_frame` ranges and code pointers through MOD0.
- NRO: layout and relocation, asset section, empty assets, malformed headers, AArch32.
- `.eh_frame_hdr`: several FDE encodings and augmentations, unsorted input, absent or omitted
  tables, zero-length functions, malformed headers, LEB128.
- NPDM: META fields, service gate (exact, wildcard, server-bit), kernel capability decoding of every
  descriptor type, SVC mask, AArch32 detection, malformed files and capability lists.
- The synthetic NSO and NRO writers in `loader/testing.rs` and the NPDM writer are independent of
  the parsers.

## Checked against a real title

The owner supplied one extracted retail 64-bit application title. It is read in place; nothing
from it is stored in the repository, and it is not named here. `tests/real_title.rs` runs when
`KILN_TITLE_DIR` points at a folder holding `exefs/` and `romfs/`, and otherwise skips loudly.
Measured results:

- `main`: all three segments LZ4-compressed; all three SHA-256 hashes match the header, which
  checks the decompressor and the header parse against Nintendo's own data.
- MOD0 found and followed; about 190,000 relocations applied (174,723 RELATIVE, 14,817 ABS64,
  33 GLOB_DAT, 639 JUMP_SLOT); none of an unsupported type.
- `.eh_frame_hdr` lists 74,181 functions covering 99.9% of the text segment; 70,141 exported
  function symbols; 26,513 RELATIVE pointers into text.
- **15,477 imports remain unresolved** because they refer to symbols defined in the other NSOs of
  the ExeFS (`sdk`, `subsdk0`). This is the cost of the "main only" first phase and bears directly
  on M7: calls through those slots cannot run until the modules are linked to each other as the
  runtime linker would. M7 will need multi-module loading and cross-module symbol resolution.
- `main.npdm`: 64-bit, 39-bit address space, 47 SVCs enabled, 42 services in the access list.
- RomFS: the extracted directory (4,636 files, 52 directories, about 5.5 GB) becomes a valid image
  in about a second; the first and last files read through the image are byte-identical to the
  host files.

## Not verified

- NRO loading has only been exercised with synthetic files; no real homebrew NRO was available.
- `.eh_frame` handling covers the common encodings (pcrel/datarel with sdata4, absolute 8-byte).
  Other encodings are reported as unsupported rather than guessed.
- Relocation types beyond RELATIVE/ABS64/GLOB_DAT/JUMP_SLOT (TLS, IRELATIVE) are reported and not
  applied; the real title used none.
- MOD0 fields added in firmware 17 and 19 (version offset, RELRO and build-id ranges) are not read.
- Symbols are resolved only inside one module. Cross-module resolution (A-009) is not implemented.
- The AArch32 first-word heuristic (A-007) can miss modules; with an NPDM the check is exact.
- Behavior on other titles, other firmware generations and NSOs with `ExecuteOnlyMemory` is
  untested.

## Next

M3: the A64 decoder and interpreter for the integer ISA, differential-tested against Unicorn.
