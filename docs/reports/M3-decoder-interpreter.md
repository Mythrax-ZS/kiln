# M3: A64 decoder and interpreter (integer ISA)

Status: complete. Date: 2026-10-12.

## What was implemented

`nx_runtime::a64`: a decoder (`decode`), a shared instruction type (`Inst`), the CPU state with a
pinned `repr(C)` layout (`CpuState`), a `Bus` trait for memory and system services, and an
interpreter (`step`, `execute`) written to follow the Arm pseudocode. The interpreter is the
oracle for every later test and runs whatever the lifter cannot translate.

### Instructions implemented

| Group | Instructions |
|-------|--------------|
| Data processing, immediate | `ADR`, `ADRP`; `ADD`, `ADDS`, `SUB`, `SUBS` (immediate, optional shift by 12); `AND`, `ORR`, `EOR`, `ANDS` (bitmask immediate); `MOVN`, `MOVZ`, `MOVK`; `SBFM`, `BFM`, `UBFM` with every alias (`SBFX`, `UBFX`, `BFI`, `BFXIL`, `SXTB/H/W`, `UXTB/H`, `LSL/LSR/ASR` immediate); `EXTR` (and `ROR` immediate) |
| Branches | `B`, `BL`, `B.cond`, `CBZ`, `CBNZ`, `TBZ`, `TBNZ`, `BR`, `BLR`, `RET` |
| Exceptions and system | `SVC` and `BRK` (reported as traps); every `HINT` encoding (no-op, including the PAC and BTI hints); `DSB`, `DMB`, `ISB`, `CLREX`; `MRS`/`MSR` for `NZCV`, `FPCR`, `FPSR`, `TPIDR_EL0`, `TPIDRRO_EL0` (read), `CNTFRQ_EL0`, `CNTPCT_EL0`, `CNTVCT_EL0`, `CTR_EL0`, `DCZID_EL0`; `SYS` for `DC ZVA`, `DC CVAC`, `DC CVAU`, `DC CIVAC`, `IC IVAU` |
| Loads and stores | `LDR`/`STR` of 8, 16, 32 and 64 bits with unsigned offset, unscaled, pre-index, post-index, unprivileged and register-offset (`UXTW`, `LSL`, `SXTW`, `SXTX`) addressing; `LDRSB`, `LDRSH`, `LDRSW` into W or X; PC-relative literal loads (`LDR`, `LDRSW`); `PRFM` and `PRFUM` (no-op); `LDP`, `STP`, `LDNP`, `STNP`, `LDPSW` with offset, pre- and post-index; `LDXR`, `LDAXR`, `STXR`, `STLXR` in all four sizes; `LDXP`, `LDAXP`, `STXP`, `STLXP`; `LDAR`, `STLR` in all sizes |
| Data processing, register | `AND`, `BIC`, `ORR`, `ORN`, `EOR`, `EON`, `ANDS`, `BICS` (shifted register); `ADD`, `ADDS`, `SUB`, `SUBS` (shifted and extended register); `ADC`, `ADCS`, `SBC`, `SBCS`; `CCMN`, `CCMP` (register and immediate); `CSEL`, `CSINC`, `CSINV`, `CSNEG` and their aliases; `RBIT`, `REV16`, `REV32`, `REV`, `CLZ`, `CLS`; `UDIV`, `SDIV`; `LSLV`, `LSRV`, `ASRV`, `RORV`; `CRC32B/H/W/X` and `CRC32CB/CH/CW/CX`; `MADD`, `MSUB`, `SMADDL`, `SMSUBL`, `SMULH`, `UMADDL`, `UMSUBL`, `UMULH` and aliases |

### Not implemented

Decoded as `Unsupported` and stopping the run with a report, never ignored: scalar floating point
and Advanced SIMD loads, stores and arithmetic (milestone M5), LSE atomics, `MSR` (immediate),
`SB`, flag-manipulation instructions, MTE, pointer-authentication instructions (the PAC hints
run as no-ops), SVE, and the privileged `HVC`, `SMC`, `HLT`, `DCPS`, `ERET`.

## Tests

`cargo test`: 102 tests in `nx_runtime`, all passing, of which 57 are unit tests of this layer and
14 are Unicorn differential tests. `cargo fmt --all --check` and
`cargo clippy --all-targets -- -D warnings` are clean.

- **Decoder**: 217 encodings assembled by LLVM (clang's AArch64 backend, an assembler that shares
  nothing with this project) must decode to the expected operands; 36 reserved and unallocated
  patterns must decode as such; valid-but-unimplemented encodings must decode as `Unsupported`;
  200,000 random words must not panic. The test assembler was itself checked against LLVM.
- **Interpreter**: 40 hand-computed tests: flags on every arithmetic form, SP versus ZR in each
  operand position, carry chains, every divide edge case (zero divisor, `MIN / -1`, 32-bit
  forms), multiply-long and high-half forms, bit manipulation, CRC32, all addressing modes and
  writeback, pairs, literal loads, the exclusive monitor (success, `CLREX`, wrong address, wrong
  size, second store, pairs, alignment faults), system registers, `DC ZVA`, traps and precise
  faults.
- **Differential against Unicorn 2.1 (Cortex-A57 model), compared on all 31 X registers, SP, PC,
  NZCV, FPCR/FPSR where relevant, and 16 KiB of memory**:

  | Group | Cases | Result |
  |-------|-------|--------|
  | Data processing, immediate | 40,000 | identical |
  | Data processing, register | 80,000 | identical |
  | Branches | 40,000 | identical (2,636 branches to unmapped targets: harness artifact H-1) |
  | Hints, barriers, `NZCV` | 20,000 | identical |
  | `DC ZVA` | 5,000 | identical |
  | `FPCR`, `FPSR`, `TPIDR_EL0`, `TPIDRRO_EL0` | 10,000 | identical |
  | Single-register loads and stores | 80,000 | identical (6,051 unmapped and 729 undefined cases agree) |
  | `LDP`/`STP` family | 40,000 | identical |
  | Literal loads | 10,000 | identical |
  | Exclusive sequences | 40,000 | identical except accepted H-2 (5,411), U-3 (720), U-6 (195) |
  | `LDAR`/`STLR` | 20,000 | identical except accepted U-4 (2,609) |
  | Random 32-bit words | 300,000 drawn | 70,169 executed identically on both; 121,861 undefined on both; 390 CONSTRAINED UNPREDICTABLE not compared; 106,881 in unsupported groups not executed; 699 strict-decoder rejections that Unicorn executes, every one in two documented families (U-5) |
  | `CTR_EL0`, `DCZID_EL0`, `CNTFRQ_EL0` | 3 | identical; `CNTFRQ_EL0` differs by design (U-8) |

  About 385,000 targeted cases plus the random-word run. Every accepted difference is explained in
  `docs/UNICORN_DISAGREEMENTS.md`, with the pseudocode ruling in each case.

### Defects the tests found

- The decoder applied the 32-bit "shift amount below 32" reservation to extended-register forms,
  where those bits are the `option` field, so valid `ADD/SUB ... , sxtb`-style instructions decoded
  as unallocated.
- The decoder accepted `PRFM` in the unprivileged form, which does not exist.
- `FPCR` and `FPSR` writable-bit masks were wrong for a core that implements AArch32.
- The harness itself had three problems (stale Unicorn translations, branch-target fetch, stores into
  the code region); each is recorded as an H entry.

## Not verified

- Nothing here has run on real hardware; Unicorn (QEMU 5) is a model, and where it was the only
  oracle its behavior was weighed against the pseudocode. Values that depend on the real chip
  (`CTR_EL0`, the `FPCR`/`FPSR` masks, SP alignment checking, the counter frequency) are
  assumptions A-001 and A-013 to A-015.
- Unicorn runs the guest at EL1; the interpreter models EL0, so EL0-only restrictions (for example
  that `TPIDRRO_EL0` is read-only) are checked by unit tests only.
- `SVC` and `BRK` are checked as "an exception in both" and by unit tests; their register effects
  belong to the runtime.
- Exclusives are checked on one thread. The cross-thread monitor and spurious store-exclusive
  failure belong to the runtime memory model (M4/M6).
- Partial completion of a faulting `STP` is not compared with hardware.
- Cycle timing is not modeled and is not claimed.

## Next

M4: lifter and LLVM codegen for the integer ISA, with native code compared against this
interpreter.
