# `CASCI(upstream, ...)` — a spectrum at fixed orbitals

A full CI at orbitals that are **not** re-optimized — the scan primitive of this API.
`upstream` is any finished stage that carries orbitals: a [`CASSCF`](CASSCF.md) (the usual
one — spend one converged orbital set on a second spectrum without paying for a second
optimization), a [`CheapCI`](CheapCI.md), or a plain [`Reference`](Reference.md) (the CI at
the SCF guess). The orbitals come from that stage and so does the active space; what varies
here is everything else.

```python
cas      = kuiva.CASSCF(ref, character=("Ti", "d"), n_active=10,
                        n_active_elec=1, n_states=2).run()   # orbitals for the ground doublet
spectrum = kuiva.CASCI(cas, n_states=10).run()               # all ten, at those orbitals
spectrum = kuiva.CASCI(cas, n_states=10,
                       solver_options=dict(kramers="restricted")).run()   # same ten, cheaper
```

**After `.run()`:** `.energy` (state-averaged — ⚠ variational at these orbitals and nothing
more), `.energies`, `.coeff`, `.active`, the full result as `.result`, and
`.spin_analysis()` / `.assign()` exactly as on [`CASSCF`](CASSCF.md).

## Quick reference

| option | default | meaning |
|---|---|---|
| `active`, `character`, `avas`, `n_active`, `n_active_elec`, `threshold` | inherited | the active space — but see the inheritance rules below; on a `CheapCI`/`CASSCF` upstream only `active=` may restate it |
| `n_states` | `1` | a count, or a per-irrep mapping wherever `CASSCF` accepts one |
| `weights` | equal | the averaging weights (for the RDM gate; equalized inside degenerate blocks, splits refused) |
| `coeff` | `None` | orbitals from elsewhere (a checkpoint, another program) — `Reference` upstream only, with `active=` |
| `solver_options` | `{}` | the full-CI solver's options, as on `CASSCF` (`kramers="restricted"`, `conv_tol`, `enforce_kramers`, `degeneracy_tol`, …) |
| `classify` | `True` | non-abelian classification of converged blocks |
| `report` | `True` | the standard output blocks |

## The inheritance rules

⚠ Two rules, and they are one rule stated twice: **a statement about orbitals belongs to the
orbitals it was made against.**

- `character=` and `avas=` read atomic populations off the *reference's own SCF orbitals*,
  so they may be stated only on a `Reference` upstream — where those are also the orbitals
  the CI runs at. On a `CheapCI` or `CASSCF` upstream the space is **inherited**, and an
  active space varied at fixed orbitals is stated as `active=` (spinor indices into the
  orbital set at hand). The orbitals have moved, so re-running a character selection against
  them may legitimately return a *different* set — and the spectrum would then not be the
  one belonging to those orbitals, with nothing in the output saying so. Restating is
  refused rather than reconciled, and so are stray selection tuners (`n_active=`, …) with
  nothing to tune.
- `coeff=` is accepted only where there is nothing to inherit — on a `Reference` upstream,
  for orbitals that came from somewhere else — and then together with `active=`, for the
  same reason. Elsewhere the chain already answers "which orbitals", and two answers to that
  is how a state set and an orbital set stop matching.

## What runs and what does not

The state-averaging gate applies exactly as on a CASSCF: weights are equalized inside a
degenerate block and a count that splits one is refused. What does **not** run is the
state-average boundary *diagnostic* — that is a statement about an orbital trajectory, and
there is none here. ⚠ A CASCI energy is variational at those orbitals and nothing more: over
a state average the upstream CASSCF did not optimize, the orbitals are not stationary, and
the levels can order themselves differently from a CASSCF over the same states.

A `solver="dmrg"` CASSCF is a legal upstream — an exact CI at network-converged orbitals is
a real check on a truncated result — but this is the conventional CI, so its determinant
ceiling applies and past it the memory ledger refuses before it allocates.

`CASCI` feeds [`NEVPT2`](NEVPT2.md) and [`PropertyDump`](PropertyDump.md) exactly as a
`CASSCF` does — but not [`PseudospinExport`](PseudospinExport.md), which consumes converged
*orbitals* rather than states and therefore belongs on the stage that optimized them.
