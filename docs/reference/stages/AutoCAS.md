# `AutoCAS(reference, targets=None, ...)` — the active space, chosen for you

```python
auto = kuiva.AutoCAS(ref).run()                 # the valence shells of every open d/f centre
cas  = kuiva.CASSCF(auto).run()                 # space, orbitals AND the proposed state count
```

A stage between [`Reference`](Reference.md) and the production one — [`CASSCF`](CASSCF.md),
[`CASCI`](CASCI.md) or [`CheapCI`](CheapCI.md), all of which take it wherever they take a
`CheapCI`. It assembles an active space from **stated targets** (physical statements of what
the calculation has to describe), decides each feature class by probing the cheap CI, and
hands on what a `CheapCI` hands on — orbitals, a stated space, a spectrum, an ordering —
plus a proposed number of states.

⚠ **It needs `atomic_reference=True` on [`ScalarSCF`](ScalarSCF.md)** (every shell it builds
is an AVAS projection onto the free-atom orbitals) **and a restricted or ROHF reference**.
Both are refused at construction, naming the knob.

**After `.run()`:** `.space`, `.orbitals`, `.n_states`, `.window`, `.floor`,
`.product_floor`, `.spectrum_cm`, `.rounds`, `.sites`, `.entropy`, `.mutual_information`,
`.occupations`, `.dmrg_ordering()`, `.fiedler_ordering()`, `.proposal`, `.assembly`,
`.result`.

## Quick reference

| option | default | meaning |
|---|---|---|
| `targets` | `None` | one statement or a list of them; `None` is `"shells"` — the valence shell of every detected open d/f centre. The five classes are in the table below |
| `solver` | `"ci"` | which production solver the size budget is resolved for: the conventional CI's ceiling is a memory bound on the determinant count, a network's is not |
| `max_spinors` | from the memory limit | largest active space allowed. Unstated: the largest whose conventional-CI residency fits the free memory at the floor root count, or a provisional cap for `solver="dmrg"` — printed either way |
| `max_determinants` | `None` | the same bound in the honest quantity for the CI; exclusive with `max_spinors` |
| `max_states` | `64` | cap on the proposed count. A manifold boundary above it proposes **no count** — only the window |
| `spectrum_tol` | `0.05` | a class stays if it moves the target manifold by more than this fraction of the manifold's width … |
| `spectrum_tol_cm` | `50.0` | … or by more than this many cm⁻¹, whichever is larger |
| `probe_noise_cm` | `10.0` | below this a difference is not a measurement and the class is dropped; between it and the tolerance the verdict is **"inconclusive, kept"**. Measured at fixed orbitals by adding up to seven pairs that describe nothing: under 0.2 cm⁻¹ on the single-ion systems, 8 on a coupled dimer |
| `prune_rel` | `0.1` | keep a candidate pair whose single-orbital entropy is at least this fraction of the largest in its own class |
| `manifold_gap_cm` | `50.0` | consecutive states closer than this are one manifold and are never separated |
| `require` | `()` | statements pinning orbitals into the core: `("character", atom, l, n_spinors[, skip_pairs])`, or `("avas", atom, l, n_spinors)` — the pairs of that free-atom reference shell, selected by the **same** union projection as the shells (never a second one, which would scramble them) |
| `exclude` | `()` | a character statement, banning orbitals from every class |
| `probe` | `dict(max_iter=8, max_determinants=6000, margin=8)` | the cheap-CI budget, held constant across the whole protocol: `max_determinants` for every measurement (for a trial, what it may add to the accepted space's determinants), `max_iter` for the one pre-optimization at the end, `margin` the roots solved above the floor so the boundary can be seen from the state above it. ⚠ `max_determinants` must hold the product of the sites' Hund configurations (32 768 for three d⁵ ions) or the stage refuses |
| `localize` | `True` | localize the shells onto the individual centres afterwards, so the space has **sites** as well as orbitals |
| `report` | `True` | print the `[automatic active space]` block |

## The target vocabulary

Atoms are addressed as everywhere else — an element symbol, a label `"Dy2"`, a 1-based atom
number, or a sequence of those for a fragment — and an element symbol **pools** every atom of
that element.

| spelling | what it selects | count | priority |
|---|---|---|---|
| `"shells"`, `("shell", "Dy")`, `("shell", ("Ti1","Ti2"), "d")` | the whole valence `l` shell of the centre(s), by count-stated AVAS. Centres of different `l` (`[("shell", "Cu", "d"), ("shell", "Tb", "f")]`) are **one** projection onto the union of their shells, each pair attributed to the shell it projects onto most | fixed: `2l+1` pairs per centre; **never pruned, never cut** | 1 |
| `("frontier", atoms)`, `("frontier", atoms, n_occ, n_vir)` | a fragment's singly occupied pairs — the radical HOMO — plus neighbours of largest population there | stated | 2 |
| `("bridge", (site_a, site_b))`, `("bridge", (site_a, site_b), atoms, n_pairs)` | ligand pairs on the bridging atoms (named, or detected: every **ligand** — a fragment connected by covalent contact [[200]](../../references.md#r200), other centres not conducting — that touches both sites, so a μ-carboxylate is found as well as a μ-oxo) that **mix with the shell**. A site may be a sublattice, `((1, 3), (2,))` | bounded, pruned | 3 |
| `("bonding", atoms)`, `("bonding", atoms, n_pairs)` | the metal–ligand bonding combinations just below the shell's projection cut | bounded, pruned | 4 |
| `("double", atoms)` | the correlating shell of the same `l`, built in what the shell left over — asking for it changes neither the shell nor the spectrum the other classes read | fixed; accepted or dropped **whole** | 5 |

⚠ **Priority is fixed and is not a knob.** It is the order the classes are *added* in — so a
bonding partner is tested in the presence of the bridge and never the other way round — and
the reverse of the order they are *dropped* in when the space is over budget. A class asked
for by name is still dropped over budget, with the drop printed; what "requested" changes is
that the class is offered at all.

```python
kuiva.AutoCAS(ref)                                                   # level 0
kuiva.AutoCAS(ref, targets=["shells", ("bridge", ("Dy1", "Dy2"))])   # level 1
kuiva.AutoCAS(ref, targets=[("shell", "Fe", "d"), ("bonding", "Fe", 3),
                            ("frontier", ("N1", "N2"), 1, 1)],
              max_spinors=24, spectrum_tol_cm=20.0,
              require=[("character", "Fe", "d", 10)])                # level 2
```

## How a class is decided

⚠ **The cheap CI measures, the spectrum decides, and entropy only prunes.** The shells are the
core; each further class is offered in its own round, its candidates pruned by *relative*
single-orbital entropy [[113]](../../references.md#r113)[[198]](../../references.md#r198),
and the class kept only if its **target manifold** — the ground manifold's relative
energies and the gap above it — moved by more than the tolerance. That is the exchange
splitting for coupled shells, the ligand-field pattern for a single ion, the radical–ion
coupling for a radical bridge: the quantity the active space is being chosen to describe.

The entropy criterion is a ranking *inside one class* and never a verdict, for two measured
reasons: it is blind to a correlating shell and to the empty members of a d manifold (both
carry ~1e-4 occupations at this level of correlation), and a low-dimensional bridge shares at
most `ln 2` nats with either ion, so in absolute terms it ranks below every metal orbital.

⚠ **Every verdict is a cheap CI at fixed orbitals** — the orbitals the candidates were
constructed in — with each trial's determinants nested in the accepted space's; only the
accepted space is pre-optimized, once, at the end. Two pre-optimizations differ by what the
addition did to the optimizer's path, which moved the spectrum by up to 100 cm⁻¹ for pairs
that describe nothing ([active spaces](../../methods/active-spaces.md#what-decides-whether-a-class-stays)).
What a verdict includes is therefore correlation and the state-specific relaxation a larger
space allows: a double shell is kept on a one-electron ion for the second reason alone.

**Three verdicts, not two.** Above the tolerance is *kept*; below the noise floor is
*dropped*; in between is **"inconclusive, kept"** — a larger space is the safe error and the
budget bounds it, whereas dropping a class the probe could not resolve is a silent claim that
it does not matter. Lanthanide exchange is the case this exists for: 256 states split by a few
cm⁻¹ is below what any cheap CI resolves, and the honest outcome is that the bridge orbitals
ride on a *requested* class.

⚠ **A truncated core of coupled centres is not measured at all.** Where the shells belong to
more than one atom and their determinant space exceeds the budget, every requested class that
fits is **"kept (not measurable)"**, with the reason printed: a selected CI of three coupled
high-spin Mn(II) ions was measured not to be a spin manifold at 6 000 or 40 000 determinants,
and seeded with the sites' Hund configurations it put the ferromagnetic level lowest and split
its components by more than the exchange. On such a system a bridge is in the space because it
was asked for, never because a number said so.

⚠ **A measurement that failed is not one that measured nothing.** A solver failure at a trial
space marks the round "not measured", keeps the class **out**, and the run continues on the
last accepted space.

## The size budget

⚠ **A shell is never cut to fit.** Over budget, whole classes are dropped in reverse priority
with each drop printed and its cost named; the core alone over budget **refuses**, with the
two ways out stated (the tensor-network solver, whose ceiling is not the determinant count,
or fewer/pooled centres). A shell with pairs missing is a different physical statement wearing
the shell's name.

A resolved budget is re-evaluated at the filling it is asked about, because the
conventional-CI ceiling bounds the **determinant count** and not the spinor count: a dilute
space well past 22 spinors runs where a half-filled one of 24 refuses
([ci](../../methods/ci.md)). The headline number in the output is the spinor count at the
core's own filling and at the floor root count.

## The proposal

The count is the first manifold boundary of the final probe's spectrum **at or above the
theoretical floor** — the Hund [[199]](../../references.md#r199) ground manifold of each
centre's shell: the level `2J+1` on the f block, where spin–orbit coupling dominates the
ligand field, and the spin multiplicity `2S+1` on the d block, where it does not and the
orbital part is the field's to decide. The *product* over centres is the dimension of the
exchange manifold and is printed even where no cap can reach it (two Dy(3+) is 256), because
that is what a whole-manifold average would cost.

`.window` is the equivalent [`EnergyWindow`](CASSCF.md#choosing-the-states-by-an-energy-cutoff),
whose cutoff sits in the middle of the gap the count was read at, so a production ladder
re-resolves the same boundary against its own spectrum.

⚠ **It is a proposal and nothing downstream trusts it.** The count is read off a *qualitative*
probe; what makes it a state count is the production stage's own machinery — the
state-averaging gate, the boundary diagnostic at both ends of the optimization, the window's
ladder with its converged witness roots ([casscf](../../methods/casscf.md)). Two outcomes are
stated rather than rounded away: a boundary above the cap proposes **no count at all**, and a
spectrum that never gapped proposes the floor marked *boundary not found*, a lower bound.

**`n_states` unstated on the [`CASSCF`](CASSCF.md) or [`CASCI`](CASCI.md) after this one
takes the proposal**, and says so in the output — the one place a stage default changes on
its upstream's account. An explicit `n_states=` is a request and stays one, including
`n_states=1`. (A [`CheapCI`](CheapCI.md) built on an `AutoCAS` keeps its own default of one
state: its average is its own, and the proposal is about the production calculation.)

## What comes out

```python
auto = kuiva.AutoCAS(ref, targets=["shells", ("bridge", ("Ti1", "Ti2"))]).run()
auto.space.description     # the assembled physical statement — never an index list
auto.n_states, auto.window # the proposal, both forms
auto.floor                 # the theoretical ground manifold; .product_floor for the coupled one
auto.rounds                # the rounds table: class, candidates, pruned, d, verdict, size, cpu
auto.sites                 # per-site active columns, once the shells localized
print(auto.summary())

cas = kuiva.CASSCF(auto).run()                                   # inherits all three
cas = kuiva.CASSCF(auto, n_states=16).run()                      # an explicit count wins
cas = kuiva.CASSCF(auto, n_states=auto.window).run()             # the window instead
cas = kuiva.CASSCF(auto, solver="dmrg", graph="site-blocked",
                   solver_options=dict(max_bond=128)).run()      # the site-blocked topology
```

- `dmrg_ordering()` is **site-blocked** where the sites are known, unlike `CheapCI`'s, which
  is always the Fiedler order: the topology of a polynuclear space comes from which centre an
  orbital sits on and never from entanglement. `fiedler_ordering()` is the other one, by name,
  and `graph="fiedler"` asks for exactly that. ⚠ `graph="site-blocked"` **refuses** where
  there is no site partition rather than quietly handing back an entanglement order.
- **A restated `character=` is refused** on the stage after this one: a selection is resolved
  against the orbitals it was stated on, and AVAS has rotated these. State `active=` — a
  statement about the orbitals at hand — or inherit the space.
- ⚠ **The space carries no symmetry labels**, as any AVAS space does: the labels belong to the
  guess spinors and the projection has rotated them, so a per-irrep `n_states` is unavailable
  downstream ([symmetry](../../methods/symmetry.md)).
- ⚠ **The orbitals are a by-product and are not always an improvement.** What this stage is
  for is the space and the proposal; the orbitals it hands on are a pre-optimized starting
  guess at a deliberately small budget, and pre-optimized orbitals are a better starting
  *density* and a worse-conditioned starting *point* for the orbital problem — the same
  trade-off a [`CheapCI`](CheapCI.md) upstream has, measured there too. On an easy ionic
  system a CASSCF started from them can therefore cost *more* than one started from the plain
  SCF guess (measured on TiCl3: four minutes against thirty seconds, the second-order
  escalation paying for a pre-optimization far from stationary). Raise
  `probe=dict(max_iter=...)` where that matters, or state the space and let the production
  stage start from the reference.
- ⚠ **The cost is one fixed-orbital CI per round** (plus one for every prune that removed
  something) **and one pre-optimization**. The rounds table records the CPU seconds of each;
  the budget is an explicit argument. On a coupled polynuclear system the determinant budget
  the stage requires is large — tens of thousands of determinants — and so is each
  measurement.

## Cross-links

[Active spaces](../../methods/active-spaces.md#automatic-selection-targets-a-probe-and-a-proposal)
for the construction behind each class and what it is measured to do ·
[workflows](../../guide/workflows.md#active-spaces-beyond-the-simple-case) for choosing a
space by hand · [`CheapCI`](CheapCI.md), which this stage's probe is ·
[`CASSCF`](CASSCF.md) · [limitations](../../limitations.md).
