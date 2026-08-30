# `NEVPT2(source, **options)`

Strongly contracted NEVPT2 [[143]](../../references.md#r143)[[144]](../../references.md#r144)
on the Dyall zeroth-order Hamiltonian [[145]](../../references.md#r145) — dynamic
correlation as **post-processing** on a converged reference, per state, decomposed by
excitation class, changing no wavefunction. `source` is a finished [`CASSCF`](CASSCF.md) or
[`CASCI`](CASCI.md). ⚠ On a `CASCI` the total is `E(CASCI) + E2` — a different reference
from `E(CASSCF) + E2` and not comparable with it.

Works on **both** solver routes: a conventional-CI source supplies its stored CI vectors; a
`solver="dmrg"` CASSCF goes through the network-backed contraction provider
[[139]](../../references.md#r139) — ⚠ which serves **six of the eight classes** today, so
that `E2` is a loud **partial** sum (`result.complete` is `False`, the report prints
PARTIAL) and is not comparable with a complete NEVPT2.

**After `.run()`:** `.e2` (Eh, per state), `.total_energies`, `.class_energies`,
`.multiplets()`, and the full result as `.result`.

## Quick reference

| option | default | meaning |
|---|---|---|
| `classes` | all eight | restrict the excitation classes — ⚠ the eight are a *partition* of the first-order interacting space, so a restricted sum is a **partial** `E2` and says so |
| `fock` | `"state-averaged"` | the Dyall H0's Fock: the block-equalized state-averaged density. `"state-specific"` exists to *measure* the alternative, warns, and is never production |
| `frozen_core` | `None` (off) | freeze core spinors — stated as an **orbital energy** on the pseudo-canonical spectrum (e.g. `-10.0` Eh), never as a count |
| `deleted_virtual` | `None` (off) | delete high virtuals — an orbital energy, same rule |
| `shift` | `0.0` | real level shift for intruders [[148]](../../references.md#r148); any applied shift warns |
| `imaginary_shift` | `False` | the parameter-free imaginary shift [[149]](../../references.md#r149); warns when applied |
| `norm_cutoff` | `1e-14` | perturber-norm floor |
| `degeneracy_tol` | `1e-6` | the degenerate-`eps` grouping tolerance — ⚠ every threshold in the module acts on **whole degenerate groups**, never single spinors |
| `on_split` | `"raise"` | what a cut through a degenerate group does |
| `checkpoint`, `restart` | `None` | the per-`(state, class)` table; separate arguments — a resumed run that should keep protecting itself passes **both** |
| `checkpoint_options` | `{}` | `min_interval=` only — the file is kilobytes whatever the active space, so the knob rations filesystem calls, not bytes |
| `deadline`, `signals` | `None` | stop **between classes**, finished ones written; resolved at construction ([clusters](../../guide/clusters.md)) |
| `report` | `True` | the standard output blocks |

## Reading the result

```python
pt = kuiva.NEVPT2(cas, frozen_core=-10.0).run()
pt.multiplets()                     # barycentres BESIDE the per-state energies
```

- ⚠ **Inside a degenerate CI manifold the individual per-state `E2` depend on the
  eigensolver's arbitrary basis; the barycentre does not** (see
  [notation](../../notation.md#degenerate-blocks-and-arbitrary-phases)). No contraction
  fixes this, so the treatment is reporting rather than repair: `multiplets()` gives
  barycentres beside the per-state energies with the member spread visible — never instead
  of them.
- ⚠ **The intruder diagnostic warns on its own.** Each class's smallest energy denominator
  appears in its table as `min |dE|`: below 0.1 Eh the class's `E2` leans on a perturber out
  of proportion to its physical weight (the warning names the class and points at
  `imaginary_shift`); below 1e-6 Eh, or at a **negative** denominator — a perturber at or
  below the reference — the warning is louder, because the class energy is then divergent
  rather than ill-conditioned, and wrong in sign. A shift bounds the damage; it does not
  remove the intruder. The fix is the reference: the active space, or which states it
  averages over.
- The corrected energies reach the property dump **only** through
  `PropertyDump(pt, ...)` — never as a flag — so the hybrid protocol (`H` from perturbation
  theory, `mu` from the CASSCF states) is recorded in the file with the substitution
  ([PropertyDump](PropertyDump.md)).

## Checkpointing

The stage checkpoints per `(state, class)` pair — the only boundary it has; a class result
is a handful of scalars, so the whole table is kilobytes whatever the active space. ⚠ **It
stores no reference**: the orbitals and CI vectors belong to the CASSCF checkpoint, and this
file keeps a *digest* — a restart against a different reference is refused, because inside a
degenerate manifold the CI basis is arbitrary, and resuming across a re-solved reference
would compute some members of a manifold in one basis and some in another. The practical
consequence: `CASSCF.from_checkpoint` reproduces the vectors exactly only when they survived
the CASSCF file's thinning, and only then do the two restarts compose. ⚠ A stop between
classes **raises** (`StopRequested`) rather than returning a partial `E2` — a correction
that stopped is not a smaller correction; the file is what finishes it.

`NEVPT2.from_checkpoint(path, source)` rebuilds a **finished** correction from a complete
table — a file read, no computation; an incomplete table is refused with the state it
stopped at (pass it to `restart=` and finish it instead). ⚠ The file's digest is *not*
checked against `source` here — a stage materialized from a thinned CASSCF checkpoint
legitimately differs inside a degenerate manifold — so pairing a table with the wrong source
is the one mistake this cannot see: pass the stage the correction was computed on.

## Not implemented, on a measurement

FIC-NEVPT2 and the quasi-degenerate variants
[[150]](../../references.md#r150)[[151]](../../references.md#r151)[[153]](../../references.md#r153)
are not implemented and not planned: the artificial multiplet splitting they would cure was
measured four orders of magnitude below the size at which a splitting means different
physics. ⚠ The published working equations [[146]](../../references.md#r146) are spin-free
and real; the ones Kuiva implements were re-derived in spinor second quantization for a
complex two-component Hamiltonian with 4-fold integral symmetry only. No 3- or 4-RDM is ever
stored [[147]](../../references.md#r147): the integrals are contracted into one perturber
vector per external label.
