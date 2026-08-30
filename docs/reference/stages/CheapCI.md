# `CheapCI(reference, ...)` — the optional pre-optimization

A cheap selected CI [[90]](../../references.md#r90)[[91]](../../references.md#r91) inside
the shared orbital optimizer: the raw spinor guess in, physical active orbitals out. Its two
products feed the stages after it — the rotated orbitals (natural in the active space) start
the [`CASSCF`](CASSCF.md), and the entanglement data
[[109]](../../references.md#r109)[[110]](../../references.md#r110) seeds a tensor-network
topology. A `CASSCF` built on this stage inherits both, plus the active space stated here,
unless told otherwise.

⚠ **The pre-optimizer's total energy means nothing** and is deliberately not an attribute.
What it claims — and what is asserted — is that the *occupations* converge long before the
energy does, which is what makes it useful for selecting orbitals.

**After `.run()`:** `.orbitals`, `.occupations`, `.natural_occupation`, `.entropy`,
`.mutual_information`, `.suggested_active()`, `.dmrg_ordering()`, the full result as
`.result`, and (with `avas=`) the AVAS result as `.avas`.

## Quick reference

**The active space** — exactly one of `active=`, `character=`, `avas=`, as on
[`CASSCF`](CASSCF.md), with `n_active=`, `n_active_elec=`, `threshold=` where they apply.
⚠ This is the stage `avas=` belongs on when a pre-optimization is wanted: AVAS works from
the reference's integer occupations, and is refused on a `CheapCI` *upstream* of a CASSCF
(the cheap CI's natural occupations are all distinct, so the rotation would be the
identity).

**The optimization** (from `kuiva.mcscf.preopt.preoptimize`)

| option | default | meaning |
|---|---|---|
| `n_states` | `1` | states in the cheap average |
| `max_iter` | `20` | macro-iteration budget |
| `mode` | `"quasi-newton"` | the orbital step engine |
| `conv_grad` | `1e-3` | gradient convergence — deliberately loose; occupations are what must converge |
| `natural_spinors` | `True` | return natural spinors in the active space |
| `space_policy` | `"event"` | how determinant re-selection meets the optimizer: event-gated by default |
| `freeze_determinants` | `None` | pin the determinant space across iterations |
| `tau`, `event_interval` | `1e-6`, `1` | the event-gating controls |

**The selected CI** (from `kuiva.mcscf.preopt.cheap_ci`)

| option | default | meaning |
|---|---|---|
| `max_determinants` | `6000` | ceiling on the selected space |
| `max_reference` | `500` | ceiling on the reference set |
| `max_excitation` | `2` | excitation rank out of the reference set |
| `selection_rounds` | `2` | selection/diagonalization rounds |
| `max_generators` | `200` | bounded generator set (the ASCI-style selection [[91]](../../references.md#r91)) |
| `ensemble_selection` | `True` | select for the state ensemble rather than the ground state |
| `state_weights` | equal | the ensemble weights |
| `with_2rdm` | `True` | build the 2-RDM (needed by the orbital step) |

## What comes out

```python
pre = kuiva.CheapCI(ref, character=("Ti", "d"), n_active=10, n_active_elec=1).run()
pre.suggested_active()        # fractionally occupied spinors — a LOWER BOUND, not an answer
pre.dmrg_ordering()           # Fiedler ordering for a path network
```

- ⚠ **`suggested_active()` is a lower bound by construction**: occupation-based selection
  [[108]](../../references.md#r108)[[113]](../../references.md#r113) cannot flag an empty
  orbital that a better treatment would populate. Combine it with orbital character and
  near-degeneracy. The concrete case: it will **never** suggest a double shell — the
  correlating shell is empty at this level of treatment (~1e-4 occupations) — so a double
  shell has to be asked for (`avas=dict(..., n_shells=2)`).
- `dmrg_ordering()` is the Fiedler ordering [[111]](../../references.md#r111)
  [[112]](../../references.md#r112) of the active spinors by mutual information;
  `graph="mutual-information"` / `"fiedler"` on a downstream `CASSCF(solver="dmrg")` builds
  the network topology from the same data.
- **Exact Kramers pairing is restored on the orbitals before they leave this stage**: the
  truncated cheap CI's determinant space is not closed under time reversal, so its orbitals
  legitimately drift off pairing — while everything downstream (the state-averaging gate, a
  contiguous-pair active space) assumes the pairing convention exactly. Chaining this stage
  into a CASSCF therefore needs no repair of its own.
