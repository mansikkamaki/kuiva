# Active spaces

An active space is a **physical statement** — "the seven lowest Kramers pairs of f
character on the dysprosium, nine electrons" — never an orbital-index window: only the
former can be reproduced by an independent implementation, and every committed reference
calculation in this project is defined that way. This page covers the constructions behind
the statement; the practical selection surfaces and their traps are in
[workflows](../guide/workflows.md#active-spaces-beyond-the-simple-case), the option syntax
on [`CASSCF`](../reference/stages/CASSCF.md).

Whatever the construction, the space is **whole Kramers pairs, always**: every orbital
space is defined on spatial orbitals, so a scalar selection maps to contiguous spinor pairs
and a pair is never split across a space boundary
([notation](../notation.md#the-spinor-basis)). The inactive count is fixed by the electron
count — $`n_{\mathrm{inactive}} = (N - n_{\mathrm{active\,elec}})/2`$ pairs, never by
counting occupied orbitals (an ROHF singly occupied orbital has occupation $`\gt 0`$ while
holding one electron, and that miscount has produced plausible wrong answers here before);
an odd inactive spinor count is refused outright.

## Selection by orbital character

The reference form: project each guess orbital's Löwdin reduced populations
[[44]](../references.md#r44) onto a stated `(atom, l)` character and take the **lowest**
pairs clearing a threshold. It takes `(atom, l)` and ⚠ **refuses a principal quantum
number**, because the integral library's principal-quantum AO labels count shells *within
the basis* — a live instance of the contraction-type trap, and not the same thing as the
physical shell. Two extensions keep the statement reproducible from what it prints:

- an **ordinal window** (`skip_pairs`) names a second shell of the same $`l`$ — "skip the
  first six p pairs, take the next" — an ordinal within a stated
  character-and-threshold ordering, which another program can reproduce where an $`n`$ label
  cannot. ⚠ It is *required*, not optional, wherever filled shells of the same $`l`$ lie
  below the valence one, and forgetting it fails silently twice over
  ([workflows](../guide/workflows.md#the-filled-shell-trap));
- pooled centres: for equivalent atoms whose canonical orbitals delocalize, the populations
  of the named centres are summed before thresholding.

Fragments are resolved independently and unioned; a pair claimed by two fragments is
refused rather than shared.

## AVAS: the covalent case

Where the metal–ligand bond is covalent, no single canonical orbital *is* the target shell
— the d or f weight is spread over bonding and antibonding combinations, none clearing any
threshold — and the space wanted is a **rotation** of them. AVAS
[[89]](../references.md#r89) constructs it: project every orbital onto a set of reference
atomic valence orbitals, diagonalize the projector **separately within the occupied and
within the virtual space**, and take the eigenvectors of large eigenvalue — the
combinations carrying the character, with the eigenvalue saying how much each carries.
Because the rotation stays inside the occupied and inside the virtual space, the reference
density and the SCF energy do not move at all.

Three properties of Kuiva's implementation, the first a **stated departure** from the
published method:

- ⚠ **The reference set is not MINAO.** Kuiva projects onto the free-atom orbitals it
  already computes for the atomic-reference charges — in the calculation's *own* basis, at
  the same per-element reference state the atomic mean field uses (neutral; M(3+) on the f
  block), so one element has one reference across the whole program, and for an ion the
  reference is better than a neutral minimal basis. The orbitals selected agree with a
  MINAO AVAS in every case tested; the *eigenvalues* are not numerically comparable with
  another program's.
- **Whole Kramers pairs, by construction.** The projector is spin-free and says nothing
  distinguishing a pair's members, so it is folded onto the pair space, diagonalized there,
  and the rotation lifted back with the barred partners taking the **conjugate** rotation —
  the output is Kramers-paired exactly, and a set that cannot be folded is refused rather
  than averaged.
- **The rotation is within groups of equal occupation**, not within "occupied": mixing a
  doubly with a singly occupied orbital would change the density, the one thing this
  transformation must not do. Every group is offered to the threshold, so a singly occupied
  orbital of the right character is selected on its merits.

The threshold is a **selection knob, not a tolerance** — the right value falls in the gap
of the printed eigenvalue spectrum, and a small gap is warned about because it means the
threshold and not the electronic structure chose the space. `n_shells=2` is the double
shell, the case a character threshold cannot find at all. ⚠ An AVAS space carries no
symmetry labels: they belong to the guess spinors and AVAS has rotated them
([symmetry](symmetry.md)).

## The cheap-CI pre-optimization

`CheapCI` runs a **selected multireference CISD** in the candidate space — CIPSI-style
perturbative selection [[90]](../references.md#r90) against a bounded set of generators, as
in ASCI [[91]](../references.md#r91), with heat-bath selection [[92]](../references.md#r92)
in the family — inside the *same* shared orbital optimizer a full CASSCF uses, as its
`ci_solver` callback. Its claim, asserted rather than assumed, is that the **natural
occupations converge long before the energy does**
[[108]](../references.md#r108)[[61]](../references.md#r61): the total energy of the
pre-optimizer means nothing and is deliberately not exposed, while the rotated
natural-orbital set is a good CASSCF start and the occupations are good evidence about the
space. ⚠ `suggested_active()` is a **lower bound by construction** — occupation-based
selection cannot flag an empty orbital a better treatment would populate, the correlating
shell of a double-shell space being the standing example (~1e-4 occupations, structurally
invisible).

The second product is **orbital entanglement**: single-orbital entropies and mutual
information [[109]](../references.md#r109)[[110]](../references.md#r110)
[[114]](../references.md#r114), used to seed a tensor-network topology and orbital ordering
(Fiedler vector of the mutual-information graph
[[111]](../references.md#r111)[[112]](../references.md#r112)) and available as
active-space evidence in the spirit of automated selection [[113]](../references.md#r113).

## Fragment localization: which centre

Selection by character says which orbitals are active; for two equivalent centres it cannot
say **which centre** — the canonical orbitals are the symmetric and antisymmetric
combinations, and the honest answer is "both". The site partition is defined **once**, by
**sequential fragment projection** — the SPADE construction [[115]](../references.md#r115),
published for one fragment and its environment, applied here sequentially so several sites
partition one active space: for each site in turn, the singular value decomposition of the
site-projected orbital block yields the combinations most localized on that site, which are
assigned to it and projected out before the next site is treated. Exact, non-iterative, and
complex-safe on spinors unchanged — the reasons it was chosen over the classical iterative
Pipek–Mezey localization [[116]](../references.md#r116), a recorded departure from the
literature default.

- ⚠ **The rotation is active–active, so it changes no energy** — asserted to machine
  precision (measured 5e-15 Eh), not to a tolerance. What it changes is what the orbitals
  *mean*: the site identity a broken-symmetry guess flips
  ([scf-reference](scf-reference.md#broken-symmetry-the-antiferromagnetic-starting-density)),
  a multi-centre pseudospin export partitions ([properties](properties.md)), and a tensor
  network orders its modes by ([dmrg](dmrg.md)).
- ⚠ **A localization that did not localize is refused**, with the populations printed:
  below the population floor the orbitals are not site orbitals, and anything built on them
  is a site partition in name only.
- The localized set is not Kramers paired (like any active–active rotation), but each
  site's span is time-reversal closed, so the pairs are rebuilt per site before the set is
  handed on.
