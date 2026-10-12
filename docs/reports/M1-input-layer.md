# M1: input layer

Status: complete for the amended scope. Date: 2026-10-12.

## Scope amendment

The original brief defined M1 as the NSP/NCA container layer: PFS0, NCA header XTS decryption,
section keys, AES-CTR/XTS, and the IVFC hash tree. The owner then instructed (2026-10-12) that
**`prod.keys` is not used at all** and that the tool works from the **extracted ExeFS and RomFS**
of a game. M1 was therefore replaced by an input layer for those extracted forms. There is no key
parser, no AES, no NSP/NCA/ticket code in the repository. A keys-file reader and AES modes were
written first and deleted before any commit.

## What was implemented

| Piece | Where | Notes |
|-------|-------|-------|
| Storage abstraction (`Storage`, in-memory, file, sub-range) | `nx_runtime::storage` | Positioned reads, bounds checked. |
| PFS0 reader | `recomp::container::pfs0` | For an ExeFS given as a PFS0 image. |
| ExeFS | `recomp::container::exefs` | Directory (or a game folder containing `exefs/`) or PFS0 image. Lists NSOs in loader order: rtld, main, subsdk0-9, sdk. Requires `main`; `main.npdm` is optional at this layer. |
| RomFS image parser | `nx_runtime::romfs::RomFs` | Walks the directory and file tables, validates every link, parent, name, and data range. Hash tables are not used for parsing. |
| RomFS builder | `nx_runtime::romfs::{plan, build_image}` | Lays out header, hash tables, entry tables, and file data for a list of paths and sizes. |
| Directory image | `nx_runtime::romfs::DirImage` | Presents an extracted RomFS directory as a flat image through `Storage`. Metadata is in memory and file data is read from the host files on demand, so multi-gigabyte trees cost no memory. The runtime serves this to the guest. |
| CLI | `recomp inspect --exefs <dir\|pfs0> [--romfs <dir\|image>]` | Prints the file list, NSO load order, and RomFS totals. |

## Tests

`cargo test`: 22 tests in `nx_runtime`, 14 in `recomp`; all pass. `cargo fmt --all --check` and
`cargo clippy --all-targets -- -D warnings` are clean.

- PFS0: valid and empty partitions, padded and unpadded string tables, bad magic, truncated tables,
  implausible entry count, entries outside the data area, offset+size overflow, name offset outside
  the string table, unterminated names, non-UTF-8 names, duplicate names, and names with path
  separators or `..`.
- ExeFS: directory form, game-folder form, PFS0 form, missing `main`, missing `main.npdm`,
  unsafe names, a non-PFS0 file, load order.
- RomFS parser: a hand-assembled image written from the libnx structure definitions (independent
  of the builder); short and bad headers, absurd table sizes, tables outside the image, sibling
  loops, parent mismatches, directory cycles, unaligned and out-of-range links, bad name lengths,
  non-UTF-8 names, `/` in names, data outside the image, offset overflow, duplicate paths.
- RomFS builder: round trip through the parser including a zero-length file, non-ASCII names and
  4 KiB+ files; a separate test looks up every file through the hash tables the way a guest-side
  reader would; layout properties (alignments, sizes); rejected paths; empty image.
- Directory image: 300 random partial reads (spanning metadata, padding, and several files)
  agree with a full read; a directory with an empty directory and an empty file parses back
  correctly; a missing directory is reported.
- CLI: argument parsing, version/help/unknown command, `inspect` output.

## Not verified

- No real ExeFS or RomFS was available, so none has been read. Every fixture is synthetic.
- The RomFS layout comes from the libnx header and the `switch-tools` builder source (see
  `docs/ASSUMPTIONS.md` A-002, A-004, A-005); the Switchbrew wiki has no reachable RomFS page.
- The hash-table check proves the builder is self-consistent under that reading of the hash
  function. It has not been checked against Nintendo's own images.
- The builder places the tables before the file data (the `switch-tools` builder puts the file data
  first). The parser follows the header offsets, so either order reads correctly; whether a
  particular game's own RomFS reader depends on the order is unknown.
- `DirImage` re-reads host files on each request and keeps 32 handles open; it has not been
  measured on a multi-gigabyte tree.

## Next

M2: NSO0, NRO0, and NPDM parsing with synthetic files; LZ4 decompression, segment hash
verification, MOD0 relocations, and the AArch32 check.
