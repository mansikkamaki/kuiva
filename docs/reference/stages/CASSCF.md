# `CASSCF(upstream, ...)`

State-averaged two-component CASSCF [[93]](../../references.md#r93)[[106]](../../references.md#r106)[[107]](../../references.md#r107)
— the calculation this program exists for. `upstream` is a finished
[`Reference`](Reference.md) or [`CheapCI`](CheapCI.md); built on a `CheapCI`, the stage
starts from its rotated orbitals and — when no space is requested here — inherits its active
space unchanged. Its CI roots *are* the spin–orbit eigenstates; there is no separate
spin–orbit mixing step afterwards.

**After `.run()`:** `.energy` (state-averaged, Eh), `.energies` (total state energies,
ascending), `.coeff` (converged spinors, AO basis), `.converged`, `.active`, `.solver`, and
per-solver results — `solver="ci"`: `.outcome`, `.boundary`, `.boundary_initial`;
`solver="dmrg"`: `.orbital`, `.events`, `.graph`, `.max_discarded` (the largest ensemble
truncation weight of the final sweep — the network's primary quality number, without which
its energies are not quotable). With `project_from=`: `.projection`. Plus
`.spin_analysis()` and `.assign()` (below).

## Quick reference

**The active space** — exactly one of the three selection forms; there is no default,
because an active space is a physical statement (elaboration:
[workflows](../../guide/workflows.md#active-spaces-beyond-the-simple-case))

| option | meaning |
|---|---|
| `character=` | `(atom, l)` with `n_active=` (and `n_active_elec=`): the lowest Kramers pairs of that character; a list of `(atom, l, n_spinors[, skip_pairs])` fragments unions selections; `atom` may be a sequence of pooled centres; `threshold=` is the Löwdin cut |
| `active=` | explicit spinor indices (or a resolved `ActiveSpace`) |
| `avas=` | `dict(atom=..., l=..., threshold=0.2, n_shells=1, max_pairs=None)`: the AVAS projection [[89]](../../references.md#r89); needs `atomic_reference=True` on the front end and a `Reference` upstream |

**The states**

| option | default | meaning |
|---|---|---|
| `n_states` | `1` | a count, or — with point-group labels present — a per-irrep mapping `{irrep: n}` |
| `weights` | equal | state-average weights; ⚠ equalized inside a degenerate block by the gate, and a count that splits a Kramers pair is refused |
| `boundary_check` | `8` | extra roots solved (and discarded) to measure the state-average boundary gap, at the starting **and** converged orbitals; `0` switches the diagnostic off. Advisory — it never kills a run |
| `preserve_symmetry` | `False` | mask inter-irrep orbital rotations, so the labels stay exact at convergence. ⚠ A constraint: converges to the lowest *symmetric* solution |
| `classify` | `True` | the non-abelian classification of converged degenerate blocks (where active) |

**The solver**

| option | default | meaning |
|---|---|---|
| `solver` | `"ci"` | `"ci"` (conventional complex determinant CI) or `"dmrg"` (the tree tensor network [[117]](../../references.md#r117)[[125]](../../references.md#r125)) |
| `solver_options` | `{}` | per-solver, two tables below |
| `graph` | `None` | `"dmrg"` only: a `NetworkGraph`, or `"mutual-information"` / `"fiedler"` to build one from a `CheapCI` upstream |

**The orbital optimizer** (remaining keywords pass through to it)

| option | default | meaning |
|---|---|---|
| `mode` | `"auto"` | `"quasi-newton"` (L-BFGS [[104]](../../references.md#r104)[[105]](../../references.md#r105)), `"second-order"` (augmented Hessian [[96]](../../references.md#r96)[[97]](../../references.md#r97)[[98]](../../references.md#r98)), or `"auto"` — the robust default, escalating on the **gradient** trajectory |
| `max_iter` | `50` | total macro-iterations (counted **across** a restart) |
| `conv_grad` | `1e-4` | orbital gradient convergence |
| `conv_energy` | `1e-8` | energy-change convergence (both must be met) |
| `max_step` | `0.20` | trust-region step bound [[102]](../../references.md#r102) |
| `memory` | `10` | L-BFGS history length |
| `active_active` | `False` | active–active rotations (redundant for a full CI; opt-in) |
| `callback` | `None` | `callback(info)` after every macro-iteration; returning `False` stops the run |
| event driver only | | `tau=1e-6`, `event_interval=1`, `max_event_interval`, `trust_floor`, `keep_memory_on_adopt` — the event-gating controls of the adaptive (DMRG) route |

⚠ **It is the orbital problem that decides `mode`, not the CI cost**: `mode="second-order"`
is the right explicit choice for a heavy element, a large state average or a DMRG solver —
a caller decision, deliberately not inferred. If you set it, give `max_iter` room to survive
the escalation delay.

**Lifecycle** (full treatment: [running on clusters](../../guide/clusters.md))

| option | default | meaning |
|---|---|---|
| `checkpoint` | `None` | schema-versioned HDF5, written every macro-iteration under the adaptive budget; the converged write is unconditional |
| `restart` | `None` | resume a run; the file **checks** the restated settings — a different active space or state average is refused |
| `checkpoint_options` | `{}` | the cadence knobs (`budget_gb`, `min_interval_seconds`, …) |
| `deadline` | `None` | `None` / a duration / `"slurm"` / `"queue"` / `"auto"`; resolved at construction |
| `signals` | `None` | `True` (TERM/USR1/USR2) or a sequence of names; resolved at construction |
| `project_from` | `None` | start from a finished stage of the same molecule in a **different basis**; `projection=dict(carry=, scheme=, repair_pairing=)` configures it ([workflows](../../guide/workflows.md#a-casscf-from-a-different-basis-set-project_from)) |
| `report` | `True` | the standard output blocks |

## `solver="ci"`: the conventional CI

`solver_options` go to the full-CI solver:

| option | default | meaning |
|---|---|---|
| `conv_tol` | `1e-8` | Davidson convergence [[192]](../../references.md#r192)[[193]](../../references.md#r193) |
| `max_iter` | `300` | Davidson iteration budget |
| `max_subspace`, `block` | auto | subspace and block-size controls |
| `backend` | `None` | kernel-backend override for this solve |
| `kramers` | `"general"` | `"restricted"`: the time-reversal-adapted eigensolver [[58]](../../references.md#r58)[[60]](../../references.md#r60) — odd electron counts only, ~1.9× less CPU from three averaged Kramers pairs up (⚠ measurably *slower* at two), refuses non-Kramers-paired integrals, and does **not** raise the memory ceiling |
| `enforce_kramers` | `True` | check (never assume) the time-reversal symmetry of the active integrals at every solve |
| `degeneracy_tol` | `1e-6` | the degenerate-block grouping tolerance the averaging gate uses |
| `on_split` | `"raise"` | what a count that splits a degenerate block does |
| `warm_start` | `True` | reuse the previous macro-iteration's vectors |

The conventional-CI active-space ceiling is a **memory** bound on the determinant count
(20–22 half-filled spinors at an 8 GB limit; it moves with the limit, and dilute or
nearly-full spaces run well past it), enforced before the first allocation; the hard limit
is 64 spinors. Past it, `solver="dmrg"`.

## `solver="dmrg"`: the tree tensor network

`solver_options` go to the network solver — `max_bond` is **required** (an uncapped tree
state allocates charge-sector-maximal bonds):

| option | default | meaning |
|---|---|---|
| `max_bond` | ⚠ required | the bond-dimension cap |
| `max_sweeps` | `30` | sweeps per solve |
| `conv_tol`, `davidson_tol` | `1e-9`, `1e-8` | sweep-energy and local-eigensolver convergence |
| `trunc_tol` | `0.0` | discard below this weight (degenerate Schmidt groups kept whole, always) |
| `bond_schedule` | `None` | e.g. `[16, 32, 64]`: ramp the cap per sweep inside the first solve (convergence is declared only at the final cap) |
| `bond_steps` | `None` | e.g. `[64, 128, 256]`: a per-macro-iteration cap ladder — each rung a chart change adopted only when it lowers the energy at fixed integrals; giving it selects the event-gated driver automatically |
| `expansion` | `0.0` | the deterministic subspace expansion [[130]](../../references.md#r130)[[129]](../../references.md#r129) for escaping local minima — no RNG enters the trajectory, energies stay variational, degenerate Schmidt groups stay exactly degenerate |
| `expansion_sweeps` | `6` | sweeps over which the expansion decays off |
| `adaptive` | `False` | adaptive network topology [[133]](../../references.md#r133) through the propose/adopt seam (the event-gated driver) |
| `policy`, `propose_sweeps` | — | the reconnection policy and its proposal budget |
| `symmetry`, `sector` | `None` | conserved symmetry labels: a labelled sweep cannot leave the sector it targets |
| `enforce_kramers`, `on_split`, `boundary_check`, `seed` | — | as on the CI route |

⚠ **Read `w_disc`.** The iteration table carries the largest discarded weight of each
macro-iteration's sweeps, and `cas.max_discarded` is the final value: every energy from this
solver has to be quoted with it, and truncation *growing* as the orbitals move is the signal
that `max_bond` is too small. The `E(w_disc → 0)` extrapolation
[[131]](../../references.md#r131)[[132]](../../references.md#r132) is a separate driver over
a converged problem, `kuiva.dmrg.bond_series`, which reports the extrapolate with the series
and its fit residual beside it, never alone.

`checkpoint=`/`restart=` work on this route too and write **two** files: the ordinary
trajectory checkpoint, and a sibling `*.network.h5` with the network state, rolling at the
end of each completed sweep. A restart resumes the trajectory exactly and warm-starts the
network from the sibling; an absent or unfitting sibling warns and starts the network cold —
time, never correctness. ⚠ `restart=` needs the frozen-chart driver: `adaptive=True` or a
`bond_steps=` ladder re-derives its space by proposals and does not resume an optimizer
state. The environment cache pages its coldest entries to scratch when a reservation would
otherwise refuse (default on; `environment_resident_gb` on the sweep caps the resident set)
— bitwise-inert, and reverting to the honest memory refusal where scratch is unconfigured.

## Restart, materialization, and the state average

`restart=` resumes a *running* optimization: the file's system fingerprint must match, the
active space may be omitted (it comes from the file) but a restated one that disagrees is
refused, and ⚠ **a different `n_states`/`weights` is refused** — a different state average
is a different calculation, not a different chart of this one. Starting a *new* average from
converged orbitals is `coeff=`, not `restart=`.

`CASSCF.from_checkpoint(path, reference, *, n_states=None, weights=None, solver_options=None,
boundary_check=None, require_converged=True)` materializes a **finished** run — no
macro-iterations; the states are re-solved at the stored orbitals, seeded by the stored CI
vectors; `n_states`/`weights` default **from** the file. The result is an ordinary `CASSCF`,
so every downstream stage takes it unchanged. ⚠ Only the converged-orbital boundary
diagnostic comes back — the starting-orbital one belongs to the run that wrote the file.
Details: [clusters](../../guide/clusters.md#restarting).

## After the run: `spin_analysis()` and `assign()`

```python
cas.spin_analysis().report()      # <S^2> per degenerate block
cas.assign()                      # a term-label OFFER: label, evidence, fit residual
```

Both work on **either** solver route through one implementation (the network route contracts
the same quantities through per-root and transition densities). `spin_analysis()` reports
`<S^2>` **per degenerate block and never per state** — inside a block the eigensolver may
return any mixture, so a single state's value belongs to that arbitrary choice while the
block trace does not — and states how much of the value came from `S` reaching *out* of the
active space (computed, never assumed away). ⚠ With spin–orbit coupling on, do not expect an
exactly integral `2S+1` even for a one-electron active space: a converged general-complex
CASSCF has orbitals that are not spin-pure — mixing spin is what SOC does — and the excess
over `S(S+1)` is a *measurement* of that mixing.

`assign()` offers a `^{2S+1}L_J` label per block, inferred from the block dimension
(`2J+1`), `<S^2>` and the measured isotropic g with the Landé formula inverted for `L`
[[181]](../../references.md#r181). ⚠ **Every label it prints is an inference** — it is its
own report with the evidence and a fit residual beside each row, never a column of the state
table, and never written to a stored file. A block whose evidence does not add up is
labelled `?`, which is the normal and correct outcome for the crystal-field levels of a
complex (those are not `2J+1` manifolds). Pass `matrices=` (from a finished
[`PropertyDump`](PropertyDump.md)) to reuse moment matrices already built.

## Per-irrep selection

With `point_group=` on the front end, `n_states={"1E1/2": 2, "2E1/2": 2}` solves each irrep
in its own sector of the determinant space — a request the plain "lowest n" form cannot
express. The states come back merged and ascending in the ordinary convention, so the
averaging gate, the boundary diagnostic, the property dump and NEVPT2 see exactly what they
always did; the boundary diagnostic runs per selected sector and reports the tightest. A
per-irrep spectrum is only meaningful while the orbitals are symmetry-pure, which is checked
at every solve; `preserve_symmetry=True` keeps them so by construction. ⚠ A per-irrep count
is **not** a safety mechanism — where the abelian group is smaller than the molecule's real
one it can cut a physically degenerate manifold exactly as a plain count can (see
[workflows](../../guide/workflows.md#designing-a-state-average)). The tensor-network solver
takes the same labels as a conserved quantum number instead
(`solver_options=dict(symmetry=..., sector=...)`).
