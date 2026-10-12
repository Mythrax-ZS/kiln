# Unicorn disagreements and harness artifacts

Differential tests (`nx_runtime/tests/unicorn_diff.rs`) run the same instruction from the same
random state on Unicorn 2.1 (QEMU 5 TCG, Cortex-A57 model) and on the project's interpreter and
compare every register, the flags and memory. **Where Unicorn disagrees with the Arm pseudocode,
the pseudocode wins** and the disagreement is recorded here. Entries tagged H are artifacts of how
the harness drives Unicorn, not differences in the CPU model.

Unicorn starts the guest at EL1, while the interpreter models EL0. Only registers and operations
that exist at EL0 are compared; an EL1-only access stops the interpreter with a report instead of
returning a value.

## Harness artifacts

| ID | What happens | Handling |
|----|--------------|----------|
| H-1 | After a taken branch Unicorn fetches the target before honouring the instruction count, so a branch to an unmapped address reports a fetch fault. | The code region is filled with NOPs and branch targets are kept inside it. A deliberately wild target is accepted when Unicorn reports `Unmapped` and the interpreter stopped after the branch. |
| H-2 | A misaligned exclusive access is reported as a generic exception, not as an alignment error code. | Compared as the same stop. |
| H-3 | Unicorn keeps stale translations after guest code is rewritten with a host write, and a taken branch pre-translates its target. A later case at that address then ran the stale block (it looked like the instruction did nothing). | The translation cache for the case's code slot is invalidated before every run. |
| H-4 | A store executed by Unicorn into the code region changed later cases. | The code region is read-only for guest stores in both backends. |

## Where Unicorn deviates from the pseudocode (the pseudocode wins)

| ID | Unicorn | Pseudocode and this project | Seen in |
|----|---------|-----------------------------|---------|
| U-3 | A store-exclusive whose monitor check fails never reaches the alignment check. | `AArch64.ExclusiveMonitorsPass` checks alignment first. The architecture says the order of the monitor check and memory aborts is IMPLEMENTATION DEFINED, so this is tolerated rather than a bug. | 720 of 40,000 exclusive sequences |
| U-4 | A misaligned `LDAR`/`STLR` does not fault. | Ordered accesses must be aligned (`CheckAlignment`, `AccType_ORDERED`): alignment fault. | 2,609 of 20,000 |
| U-5 | Executes `LDAR`/`STLR` and `SMULH`/`UMULH` encodings whose should-be-one fields are not all ones (`Rs`/`Rt2`, `Ra`). | These fields are `(1)`: not-ones is CONSTRAINED UNPREDICTABLE. The decoder is strict and reports them as unallocated, which also keeps data from decoding as code. | 560 and 139 of 300,000 random words |
| U-6 | `LDXR`/`STXR` are a compare-and-swap of the value `LDXR` observed. An intervening plain store that changes the monitored location makes the store-exclusive fail, and an ABA write would wrongly let it succeed. | The monitor is a state machine: the PE's own plain stores do not affect it. This is the "plain compare-and-swap is not equivalent" point. | 195 of 40,000 |
| U-7 | Uses -1 as its "monitor open" marker, so a store-exclusive to address `0xFFFF_FFFF_FFFF_FFFF` looks armed and faults on the unmapped address. | An unarmed monitor fails the store with no memory access. | not drawn in the last run; rule kept |
| U-8 | `CNTFRQ_EL0` reads 62,500,000 (QEMU's generic timer default). | The Switch counter runs at 19,200,000 Hz (docs/ASSUMPTIONS.md A-001). A platform property, not a CPU behavior. | `cache_identification_registers_match_the_a57` |
| U-9 | Executes `FCMP`/`FCMPE` against `#0.0` when the `Rm` field, which the zero form fixes at zero, is not zero. | No encoding matches: the decoder reports it as unallocated. | 8 of 8,000,000 random words |
| U-10 | A misaligned `LDAR`/`STLR`/`LDXR`/`STXR` to an unmapped or read-only address reports the translation or permission fault. | `AArch64.CheckAlignment` runs before the address is translated, so the access is an alignment fault. Both stop the run; only the reported cause differs. | 1 of 8,000,000 random words |
| U-11 | Executes system registers and `SYS` operations that are not EL0 operations on the Switch's core or that the model does not provide (debug and PMU registers, the physical timer, `TLBI` and `DC` operations), because it runs the guest at EL1. | The interpreter stops with a report naming the encoding and never invents a value; a title that needs the register shows up as a stop with its address. | 16 of 8,000,000 random words |

## Not compared (CONSTRAINED UNPREDICTABLE)

Both results are architecturally allowed, so the random-word test skips these encodings (390 of
300,000): `LDP` with `Rt == Rt2`; writeback with `Rt == Rn` or `Rt2 == Rn` (`Rn != 31`);
store-exclusive with `Rs` equal to `Rt`, `Rt2` or `Rn`; `LDXP` with `Rt == Rt2`. The interpreter's
choices are in docs/ASSUMPTIONS.md A-017.

## Resolved, not a disagreement

The first run disagreed on the writable bits of `FPCR` and `FPSR`. The Cortex-A57 implements
AArch32, so `FPSR[31:28]` are storage, and it cannot trap floating-point exceptions, so the trap
enable bits are RAZ/WI. The interpreter now uses those masks (docs/ASSUMPTIONS.md A-013).
