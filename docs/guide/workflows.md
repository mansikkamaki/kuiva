# Workflows

The harder real cases, in the order they tend to show up: designing a state average, active
spaces that a one-line selection cannot state, antiferromagnetically coupled centres,
carrying orbitals between calculations, spectra at fixed orbitals, embedding in a crystal,
spin-free reference runs, dynamic correlation, and the tensor-network route. Each section is
self-contained; all of them assume [a first calculation](first-calculation.md).

## Designing a state average

The state average is the single most common way to get a plausible wrong answer out of this
program. Four questions, asked before setting `n_states`:

1. **Does the count land on a manifold boundary?** A count that stops *inside* a
   near-degenerate manifold makes the averaged density non-invariant; the Fock operator
   built from it splits the shell, and the selection keeps cutting the same way. Kuiva
   solves a few roots the average does *not* use and reports the gap at the starting
   orbitals and at the converged ones, warning below 50 cm⁻¹. Read those two lines — they
   are the only evidence there is. (50 cm⁻¹ is not a physical tolerance; it states that the
   boundary is *unambiguous*.)
2. **Is it the count for *this* spectrum?** A count that bounds the `2S+1` terms of a
   spin-free calculation generally cuts the `2J+1` multiplets of the same system with
   spin–orbit coupling on, and the converse bites equally. Carry the two counts separately;
   do not reuse one.
3. **Is the average one the symmetry leaves invariant?** Landing on a boundary is necessary
   and not sufficient. The converged boundary report also states the averaged density's
   **spin non-invariance**: near zero — a whole term, or the whole space — is safe by
   symmetry and the gap stops mattering; large — a single J, a single Kramers doublet —
   means the average *leans* on the spin–orbit structure and is protected only by its gap
   and by where it started. Measured across free ions and ligand-field complexes: a leaning
   average over a ~30 cm⁻¹ gap is a saddle — a 10⁻⁸-sized Kramers defect grows by orders of
   magnitude per iteration until the run is refused — while leaning averages over gaps of a
   few hundred cm⁻¹ and up were locally stable, ligand-field ground-doublet averages among
   them. ⚠ That growth was measured with the rotation *unconstrained*. Kuiva now constrains
   the rotation by default wherever the orbitals are Kramers paired
   ([casscf](../methods/casscf.md#keeping-the-orbitals-kramers-paired)), and a constrained
   rotation cannot amplify a Kramers-breaking seed at all — so what warns you on a leaning
   average is the **gap and the non-invariance**, not a refusal. From a start that is already
   time-reversal broken (a restart on unconstrained orbitals, an unrestricted reference, a
   deliberately perturbed guess) the constraint does not apply and the amplification is what
   it always was. Either way the gap has to be read at the *symmetric* orbitals to mean
   anything.
4. **If it leans, where did the orbitals come from?** The case to know about is the free ion
   with several open-shell electrons: a J-only average that is locally stable at the
   symmetric orbitals still converged *from the scalar guess* to a solution with the `2J+1`
   manifold split by ~2 cm⁻¹, **every diagnostic clean** — a wrong basin, which no check can
   see from inside the run. So: average whole terms; if a sub-manifold average is what you
   want, converge the whole-term (or full-space) average first, start the sub-manifold run
   from those orbitals (`coeff=`), and check the manifold splitting of the result directly.

`boundary_check=0` switches the diagnostic off; `boundary_check=n` sets how many extra roots
it solves. It never kills a run: a failure to *measure* the gap is a warning and **no**
report, which is a weaker statement than a clean one, not a substitute for it.

Two refusals run unconditionally: a count that would split a Kramers pair, and — with
`point_group=` on — a count that cuts a multiplet whose dimension the full double group
fixes. ⚠ Neither catches a manifold cut in half (an even remainder passes both), which is
why the questions above are yours and not the program's. And ⚠ a **per-irrep** count
(`n_states={"1E1/2": 2, ...}`) is not a safety mechanism either: where the abelian group
used in the mathematics is smaller than the molecule's real one, two members of a physically
degenerate manifold can carry different labels, and a per-irrep count can split the manifold
exactly as a plain count can.

## Active spaces beyond the simple case

`character=(atom, l)` with `n_active=` states the simple case: the lowest Kramers pairs of a
given angular-momentum character on a centre. `atom` may be a sequence of centres whose
populations are pooled — the right form for equivalent centres whose canonical orbitals
delocalize. There is no default active space: it is a physical statement. Three harder cases
follow.

### Two shells in one active space

The union form takes a list of fragments, each `(atom, l, n_spinors)` in whole Kramers
pairs:

```python
character=[("Dy", "f", 14), ("Dy", "d", 10)]     # 4f + 5d, one active space
```

The fragments are resolved independently and unioned, and a pair claimed by two of them is
**refused rather than shared** — a pair clearing two thresholds at once means the fragments
were not the disjoint physical statement they were written as.

⚠ A second shell of the **same** `l` is a different case, because two fragments with the
same `(atom, l)` select the same lowest pairs and are refused for overlapping. Either pool
them into one selection, or offset the second fragment with an **ordinal window** — the
optional fourth element, `skip_pairs`:

```python
character=("Ti", "d"), n_active=20                    # the lowest TEN d pairs: 3d + 4d
character=[("Ti", "d", 10), ("Ti", "d", 10, 5)]       # the same twenty, named as two shells
```

Neither form names a *principal quantum number*, and that refusal is not softened: an `n`
label counts shells within the basis, while an ordinal within a stated
character-and-threshold ordering is something another program can reproduce.

### The filled-shell trap

The same window is **required, not optional**, wherever filled shells of the same `l` lie
below the valence one — and forgetting it fails silently, twice over. `character=(atom, l)`
takes the **lowest** qualifying pairs, which is the valence shell only when nothing of that
`l` is filled beneath: true for 3d and 4f, false for every p shell above the second row.
`character=("Ga", "p")` selects gallium's **2p core**, not its 4p valence electron, and the
calculation converges and reports an entirely ordinary spectrum. A g value cannot detect it
(a p¹ shell is Landé 2/3 whichever shell it occupies), and whether the CASSCF repairs the
wrong starting guess depends on the Hamiltonian rather than the active space — measured, the
same Ga selection lands on the valence answer with `screening="none"` and on a ²P splitting
of 249 400 cm⁻¹ (against an experimental 826) with the default screening. **Count the filled
shells of that `l` and skip them** — `("Ga", "p", 6, 6)`: 2p and 3p are six pairs — and
check a computed splitting against a published one, which is the cheapest check that catches
this.

One case of it Kuiva *can* see, and warns about: a selection whose orbitals are **entirely
occupied** in the reference (or entirely empty) while the active electron count you asked for
is a different number. That is a shell taken whole and then given the wrong occupation —
`character=("U", "f")` with three electrons takes uranium's *filled* 4f shell, which holds
fourteen. The warning names both counts. It is a warning rather than a refusal, because an
intended non-aufbau configuration is a legitimate request; but a partly filled valence shell
does not produce it, so if you see it and did not mean it, the window is what you are
missing. Note the converse is not detectable at all: a selection that is *partly* occupied
and still on the wrong shell passes silently, which is why the counting rule above stands.

### AVAS: when no single orbital *is* the target shell

A character selection can only pick orbitals that already carry the character. Where the
metal–ligand bond is covalent, the d (or f) weight is spread over several bonding and
antibonding combinations, none of which clears any threshold, and the active space you want
is a **rotation** of them. That is what AVAS constructs [[89]](../references.md#r89):

```python
scf = kuiva.ScalarSCF(mol, atomic_reference=True).run()      # AVAS projects onto these
cas = kuiva.CASSCF(kuiva.Reference(scf).run(),
                   avas=dict(atom="Ti", l="d")).run()
cas.avas.report()          # the projection eigenvalues, and the gap at the cut
```

Every orbital is projected onto the free-atom valence orbitals of `(atom, l)`, the projector
is diagonalized within the occupied and within the virtual space, and the combinations of
large eigenvalue become the active space. `active=`, `character=` and `avas=` are three ways
of answering one question, and exactly one may be given.

- `threshold=` (default 0.2) is the projection eigenvalue a Kramers pair must carry. ⚠ It is
  a **selection knob, not a tolerance**: the right value is the one that falls in the gap of
  the eigenvalue spectrum, which the report prints for exactly that reason. A small gap is
  warned about, because it means the threshold and not the electronic structure chose the
  space.
- `n_shells=2` is the **double shell** — the target shell plus its correlating partner. This
  is the case a character threshold cannot find at all, the correlating shell being diffuse
  and covalent. (⚠ The cheap CI's `suggested_active()` will never find it either: it selects
  on fractional occupation, and a correlating shell is empty at that level — a structural
  blindness, not a threshold to lower. A double shell has to be asked for.)
- `max_pairs=` refuses rather than returning more than you expected — worth setting, since a
  threshold slightly too low gives a perfectly plausible space one or two pairs too large,
  discovered only when the CI runs.

⚠ Three things to know. Kuiva projects onto the free-atom reference orbitals that
`atomic_reference=True` computed, at the same per-element reference state the atomic mean
field uses — **not** onto the published method's minimal MINAO basis: the orbitals selected
agree, but the eigenvalues are not numerically comparable with another program's AVAS. The
rotation stays inside groups of equal occupation, so the SCF density and energy do not move.
And an AVAS space carries **no symmetry labels** — they belong to the guess spinors and AVAS
has rotated them, so per-irrep `n_states` is unavailable after it.

### Which centre an active orbital belongs to

Selection by character says *which* orbitals are active; for two equivalent centres it
cannot say *which centre*, and the honest answer for the canonical orbitals is "both".
Localization rotates inside the active space so that every orbital sits on one site —
sequential fragment projection (SPADE [[115]](../references.md#r115)):

```python
from kuiva.interface.api import active_space_for, localize_active_space

space = active_space_for(ref, character=([0, 1], "d"), n_active=20, n_active_elec=2)
local = localize_active_space(ref, space, [0, 1])          # one site per metal
cas   = kuiva.CASSCF(ref, active=space, coeff=local.coeff).run()
```

⚠ It changes no number — the rotation is active-active, so the CI energy is invariant to
machine precision (measured 5e-15 Eh). What it changes is what the orbitals *mean*: it is
what a broken-symmetry guess flips, what a multi-centre pseudospin export partitions, and
what a tensor network wants its modes ordered by. A site may be a whole fragment
(`[["Fe", 3, 4], ...]`) and `counts=` states an uneven split. A set that does not localize
is **refused** with its populations printed, since a site partition that is half delocalized
is one in name only.

## Antiferromagnetically coupled centres: broken symmetry

⚠ **An unrestricted SCF started the ordinary way stays symmetric.** The closed-shell density
is a stationary point of the energy, so nothing in the iteration pushes off it: you ask for
UHF on a coupled pair of metals, get the restricted answer, and nothing in the output says
so. The polarization has to be put in by the starting density
[[196]](../references.md#r196)[[197]](../references.md#r197):

```python
scf = kuiva.ScalarSCF(dimer, reference="uhf",
                      broken_symmetry={"Fe1": +5, "Fe2": -5}).run()   # signed, per centre
```

The values are **signed counts of unpaired electrons per centre**, addressed the same way a
per-atom basis or reference configuration is (`"Fe1"`, `"Fe"`, or a 1-based atom number).
Kuiva converges the **high-spin** state — the easy, unambiguous one — localizes its singly
occupied orbitals onto the centres you named (the SPADE partition above; for two equivalent
metals that localization is what makes the flip expressible at all), flips the ones you
asked to be spin-down into the beta set, and runs the SCF from that density.

Two things are then reported, and **both** matter:

- **`<S^2>` between the low-spin and high-spin values** says the determinant really is
  broken-symmetry — that the solution is spin-contaminated **on purpose**. Coming back at
  the low-spin value means the polarization did not survive the iteration, and Kuiva warns
  rather than letting it pass as a converged UHF.
- **The spin populations must carry the signs you asked for**
  (`kuiva.props.population.scalar_spin_populations`). A solution with the two centres
  swapped has the same energy and the same `<S^2>`, and is a different state; nothing but
  the signs can tell them apart.

Measured on two Ti(3+) ions 4 Å apart: the ordinary UHF gives `<S^2> = 0` and zero spin on
both metals; the broken-symmetry guess gives `<S^2> = 1.00`, `+1.00 / −1.00` electrons of
spin, and an energy 0.29 Eh lower. ⚠ It is one of three mutually exclusive ways to start an
SCF (with `guess_from=` and `init_guess=`), needs `reference="uhf"`, and refuses when the
magnetic orbitals do not localize (`bs_min_population=` is the knob, and the refusal prints
the populations). ⚠ One consequence downstream: an unrestricted spinor set is orthonormal
but **not Kramers paired**, so an active space on a UHF reference may not be chosen as a
contiguous spinor range — select by orbital character per spin set.

## Carrying orbitals between calculations

### An SCF from another SCF: `guess_from=`

`guess_from=` takes a finished `ScalarSCF` (or its data) and starts the SCF from its
orbitals — used as they are over the same AO basis (the potential-energy-surface case), or
projected onto a different basis through the same projector `project_from=` uses
[[50]](../references.md#r50)[[47]](../references.md#r47).

```python
small = kuiva.ScalarSCF(mol_svp).run()
big   = kuiva.ScalarSCF(mol_tzvp, guess_from=small).run()   # projected onto the larger basis
```

⚠ Two measured caveats. It buys nothing on a closed shell (Fock-build counts move by ±2,
which is noise); the saving is real only where the SCF is hard. And there it decides *which*
stationary point you find: on TiCl₃ the projection cut 35 Fock builds to 21 **and landed on
a different SCF solution**, 9.7 mEh above the cold start's — internally unstable, and
`stability="follow"` walks back to the cold-start solution to every printed digit. **So pair
`guess_from=` with `stability=` on anything open-shell.** It cannot be combined with
`init_guess=`, and the two calculations must be the same molecule (elements and order are
checked; the geometry is not — carrying orbitals along a scan is the ordinary use).

### A CASSCF from a different basis set: `project_from=`

A CASSCF costs what its basis costs, and almost all of that expense buys the *orbitals* —
while the active space, a statement about chemistry, is very nearly the same in a small
basis as in a large one. `project_from=` converges the calculation where it is cheap and
continues it where you want it (the `EXPBAS` workflow of OpenMolcas
[[51]](../references.md#r51), through Kuiva's own projector):

```python
cheap = kuiva.CASSCF(kuiva.Reference(scf_small).run(),
                     character=("Ti", "d"), n_active=10, n_active_elec=1).run()

cas   = kuiva.CASSCF(kuiva.Reference(scf_large).run(), project_from=cheap).run()
```

The converged spinors are re-expressed over the target AO basis, made orthonormal again, and
the dimensions the larger basis adds are completed from its own SCF. The source may be a
`CASSCF`, a `CheapCI` or a plain `Reference`, and it may be in a *larger* basis than the
target — projecting downwards is the same call.

⚠ **The active space comes across with the orbitals and may not be restated.** It was chosen
once, against the orbitals being carried; re-selecting it here would resolve it against the
target reference's own guess orbitals — a different calculation wearing the same name. (A
projection from a bare `Reference` has no space to inherit, so there it *is* stated, and is
resolved against the source.) A projection replaces a `CheapCI` pre-optimization rather than
following it, and does not combine with `restart=`.

`projection=dict(...)` configures it — `carry` (`"active"` default: the inactive and virtual
orbitals come from the *target's own SCF*, because the source's are eigenvectors of nothing
in the target basis and carrying them costs roughly twice the macro-iterations, measured),
`scheme` (`"blocked"` default: re-orthonormalization space by space, so the CAS partition
stays exact; `"symmetric"`; `"gram-schmidt"`), and `repair_pairing` (on by default: a
converged general-complex CASSCF is entitled to leave its active orbitals far from
Kramers-pair-aligned, and everything downstream needs pairs).

⚠ The projection reports **invariants**, not coefficient comparisons: the retained norm of
each source orbital, the principal overlaps [[54]](../references.md#r54) between the source
active space and the one handed on, and the complement separation. Read them — every way of
getting a basis projection wrong still produces an orthonormal orbital set of the right
shape that starts a calculation which converges; the failure shows up only as a run that
takes longer than it should.

## A spectrum at fixed orbitals: `CASCI`

Optimizing the orbitals for the states you want and then reading the full spectrum off them
is the ordinary shape of a ligand-field calculation — one CI solve instead of a second
orbital optimization:

```python
cas      = kuiva.CASSCF(ref, character=("Ti", "d"), n_active=10,
                        n_active_elec=1, n_states=2).run()   # orbitals for the ground doublet
spectrum = kuiva.CASCI(cas, n_states=10).run()               # all ten, at those orbitals
```

`CASCI` is where a scan is written: what varies is the state count, a per-irrep request, the
CI symmetry mode, or the active space restated against those same orbitals. ⚠ One rule,
stated twice: **a statement about orbitals belongs to the orbitals it was made against.**
`character=` and `avas=` read populations off the reference's own SCF orbitals, so on a
`CheapCI` or `CASSCF` upstream — where the orbitals have moved — the space is inherited, and
a variation is stated as `active=[spinor indices]`; restating a character selection there is
refused rather than reconciled. Likewise `coeff=` is accepted only on a `Reference`
upstream, together with `active=`.

The state-averaging gate applies as in a CASSCF; the boundary *diagnostic* does not run
(it is a statement about an orbital trajectory, and there is none). ⚠ A CASCI energy is
variational at those orbitals and nothing more: over a state average the upstream CASSCF did
not optimize, the levels can order themselves differently from a CASSCF over the same
states.

## Embedding in a crystal: point charges

A single-molecule magnet is measured in a crystal, and a bare gas-phase 3+ ion has a
qualitatively different ligand field from the one in the lattice. Point charges are the
smallest honest way to say what surrounds the molecule:

```python
env = kuiva.Environment(
    point_charges=[(+2.0, (0.0, 0.0, 5.4)), (-1.0, (0.0, 3.1, -2.2))],  # (q, (x, y, z))
    label="two shells of the lattice, Ewald-fitted elsewhere")
mol = kuiva.Molecule(atoms=[...], basis="x2c-TZVPall-2c", environment=env)
```

A whole lattice is given as arrays: `Environment(point_charges=(charges, coords))`.
Generating the field — an Ewald fit from a crystal structure — is *not* part of Kuiva; the
list is the input.

- **Coordinates are in the molecule's own unit** unless the environment states its own
  (`unit="bohr"`). ⚠ A charge field copied from a crystallographic file is in Angstrom; the
  same numbers read as bohr put the lattice 1.9× too far away and produce a perfectly
  plausible ligand field for the wrong crystal.
- **What it changes is exactly two things**: the one-electron Hamiltonian gains the charges'
  potential, and the classical charge–nucleus energy is added — on its own output line, not
  folded into the nuclear repulsion, so an embedded total stays separable. Everything after
  the front end is untouched, which is what keeps an embedded calculation the *same*
  calculation. The gauge origin does not move, so the vacuum and embedded property files
  stay comparable; the field is recorded in the provenance by count, net charge, extent and
  a digest.
- **Symmetry sees the field**: a field of lower symmetry than the nuclei restricts the
  labels and switches the non-abelian classification off. A charge closer than 0.5 bohr to
  a nucleus is refused (the density polarizes onto it without bound).
- ⚠ **How big the field must be is a property of the cut, not of the radius.** Truncating an
  ionic lattice on a **sphere never converges** — measured, ±100 cm⁻¹ out to 2900 charges
  with no trend, because the surface multipoles do not vanish with radius. An
  **Evjen-weighted cube** (face charges ×1/2, edge ×1/4, corner ×1/8) is converged to a few
  cm⁻¹ at ~300 charges. Check the convergence of the lattice's own potential and field
  gradient at the metal before trusting a spectrum: it is free and it shows the same split.
- ⚠ **What an embedding is worth differs by shell**, measured: a 3d complex's ligand-field
  levels move by up to ~900 cm⁻¹ with a modest g shift; a 4f complex's levels move ten times
  less — the shielded shell — but the field **rewrites the ground doublet's g tensor**
  (CeCl₃: (0.880, 2.531, 2.531) → (0.709, 1.361, 3.501)). A lanthanide doublet's composition
  inside its `2J+1` manifold is what the crystal field fixes, so **a gas-phase Ln g tensor
  is a different quantity, not an approximate one**.
- Cost is not a consideration at any realistic size: the bare potential is one batched grid
  integral (~41 ms per 1000 charges). The potential is added bare by default;
  `picture_change=True` transforms it through the Hamiltonian's own X2C decoupling instead
  (measured ~1e-5 relative, not growing with Z).

## Turning spin-orbit coupling off

Two options are easy to confuse, and they answer different questions:

- **`screening="none"`** drops the *two-electron* picture change and keeps the one-electron
  spin–orbit operator: still two-component, still spin–orbit coupled, with j-splittings
  5–30% too large. It is a statement about **cost** (it skips the four-component atomic
  solves), the right choice for anything that is not about spin–orbit coupling.
- **`with_soc=False`** ingests no two-component Hamiltonian at all: the correlated step runs
  on the spin-free X2C operator, and what comes out is a scalar-relativistic spectrum whose
  degeneracies are `2S+1` rather than `2J+1`. The front end warns every time, because a
  scalar answer read as a relativistic one is the whole failure mode.

`with_soc=False` is the right setting for a spin-free reference number: a term energy to
compare against a non-relativistic code, a `2S+1` term count to check a state average
against before turning coupling on. Three things to know:

- **It costs what the two-component calculation costs.** The determinant space is still
  built over spinors — the full two-component determinant count, not the smaller α×β
  product a spin-free code would use. Switching coupling off simplifies the answer, not the
  calculation.
- ⚠ **`Sz` is conserved here and Kuiva does not exploit it**: the determinants fall into
  sectors, and every root is solved in the one full space.
- ⚠ **Which is exactly why the eigensolver's guess is *generic* rather than merely good.**
  Those sectors are invariant subspaces, and a Krylov method can never leave the ones its
  starting vectors lie in; a guess from the lowest-diagonal determinants can lie entirely
  inside a few of them, and the solver then converges every residual, reports success, and
  returns states that are **not the lowest**. Measured on a spin-free Dy(3+) CAS(9, 14): a
  66-root solve missed 22 of the true lowest 66 and came back 6766 cm⁻¹ too high — in a
  *third* of the iterations of the correct solve, so a fast solve is evidence of nothing.
  Generic vectors are prepended to every cold start, making that failure structurally
  unavailable. ⚠ A spectrum that changes qualitatively with the root count is this, not
  physics — and the definitive check is a dense diagonalization of the active-space
  Hamiltonian (`kuiva.ci.strings.hamiltonian_matrix`), cheap enough to reach for whenever a
  spin-free spectrum surprises you.

## Dynamic correlation: NEVPT2

Strongly contracted NEVPT2 [[143]](../references.md#r143)[[144]](../references.md#r144) on
the Dyall Hamiltonian [[145]](../references.md#r145), as post-processing on a converged
`CASSCF` or `CASCI` — per state, decomposed by excitation class, changing no wavefunction:

```python
pt = kuiva.NEVPT2(cas, frozen_core=-10.0).run()   # an orbital ENERGY, never a count
pt.multiplets()                                   # barycentres beside the per-state energies
```

- **Frozen core and deleted virtuals are off by default**, and both are stated as an orbital
  energy on the pseudo-canonical spectrum, never as a count — an energy survives a basis
  change; an index does not.
- **Intruder-state level shifts** (real [[148]](../references.md#r148) and imaginary
  [[149]](../references.md#r149)) exist and are parameter-free by default; any applied shift
  warns. ⚠ The intruder diagnostic warns on its own: each class's smallest energy
  denominator is printed, and below 0.1 Eh the class leans on a perturber out of proportion
  to its weight; below 1e-6 Eh, or at a *negative* denominator, the warning is louder,
  because the class energy is then divergent or wrong in sign. A shift bounds the damage; it
  does not remove the intruder — the fix is the reference (the active space, or which states
  it averages).
- ⚠ **Inside a degenerate CI manifold the individual per-state `E2` depend on the
  eigensolver's arbitrary basis; the barycentre does not.** No contraction fixes this, so
  the treatment is reporting rather than repair: `multiplets()` gives barycentres beside the
  per-state energies, with the member spread visible.
- All eight excitation classes are a *partition* of the first-order interacting space, so a
  restricted `classes=` gives a **partial** `E2` and says so. ⚠ A tensor-network reference
  reaches NEVPT2 with six of the eight classes served; that `E2` is a loud **partial** sum,
  not comparable with a complete NEVPT2.
- ⚠ On a `CASCI` source the total is `E(CASCI) + E2` — a different reference from
  `E(CASSCF) + E2` and not comparable with it.

## Beyond the CI ceiling: the tensor-network solver

The conventional CI reaches 20–22 half-filled spinors at an 8 GB memory limit (a bound on
the determinant count, so it moves with the limit; dilute or nearly-full spaces run well
past it). Beyond that, the same CASSCF runs on the in-house tree tensor network
[[117]](../references.md#r117)[[125]](../references.md#r125) through the same orbital
optimizer:

```python
pre = kuiva.CheapCI(ref, character=..., n_active=..., n_active_elec=...).run()
cas = kuiva.CASSCF(pre, solver="dmrg", n_states=4, graph="mutual-information",
                   solver_options=dict(max_bond=128, adaptive=True)).run()
```

`solver_options` must carry `max_bond` and may carry `adaptive=True` (network-topology
changes adopted only when they lower the energy at fixed integrals), a `bond_schedule` ramp,
and `expansion=` (the deterministic subspace expansion
[[130]](../references.md#r130)[[129]](../references.md#r129), for escaping local minima
without an RNG in the trajectory). `graph=` seeds the topology from the cheap CI's
entanglement [[111]](../references.md#r111). ⚠ **Read `w_disc`** — the largest discarded
weight, in the iteration table and on the finished stage as `cas.max_discarded`: it is the
network's primary quality number, and every energy from this solver has to be quoted with
it. Truncation *growing* as the orbitals move is the signal that `max_bond` is too small.
`kuiva.dmrg.bond_series` runs the ascending-bond-dimension series behind an
`E(w_disc → 0)` extrapolation [[131]](../references.md#r131), reported with the series and
its fit residual beside it, never alone.

## Non-Kramers ions: when the ground "doublet" is two singlets

An **integer**-spin ion — Tb(3+), Ho(3+), and the lanthanide single-molecule magnets this
code exists for — has no Kramers protection: its ground doublet is two *singlets* split by a
tunnelling gap Δ [[182]](../references.md#r182)[[180]](../references.md#r180), and each
arrives as a block of one state, which carries no magnetic moment (the moment belongs to the
pair). The multiplet table prints `nan` for them — never 0, because "not defined here" is a
different statement from "measured zero" — and the run warns, naming Δ. Group them
explicitly:

```python
for m in matrices.analyse(pseudo_doublet_tol_cm=50.0):
    if m.non_kramers:
        print(m.g_z, m.tunnelling_gap_cm, m.g_transverse_residual)
```

Grouping is **opt-in and never inferred** — whether two nearby singlets are one
tunnelling-split doublet is physics their energies cannot settle. `g_z` and Δ are the
numbers to quote (the transverse components vanish identically for a true non-Kramers
doublet), and ⚠ `g_transverse_residual` is the check that can fail: a value comparable to
`g_z` says these two states are not one doublet and the grouping request was wrong.
