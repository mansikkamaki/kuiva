# Spin-orbit coupling

Spin–orbit coupling enters Kuiva through the ingested two-component Hamiltonian — never
through the SCF, whose orbitals stay scalar and real — and it enters **twice**: the
one-electron spin–orbit operator arrives with the X2C decoupling ([x2c](x2c.md)), and the
**two-electron screening** of that operator arrives through the atomic mean-field picture
change, X2CAMF [[16]](../references.md#r16), **on by default**. The screening is what makes
fine structure quantitative: without it, atomic j-splittings come out 5–30% too large.
Options are on the [`ScalarSCF`](../reference/stages/ScalarSCF.md) page.

## The spinor basis, briefly

Each real scalar orbital $\phi_p$ becomes the Kramers pair $(\psi_{2p}, \hat T\psi_{2p+1})$
with $\hat T = -i\sigma_y K$ — the conventions, fixed once and relied on by the CI
addressing, are in [notation](../notation.md#the-spinor-basis)
[[20]](../references.md#r20)[[57]](../references.md#r57). The guess carries no spin–orbit
coupling; the coupling is introduced when the two-component wavefunction is built and
optimized, and the complex part of the orbital rotation is what lets the CASSCF respond to
it ([casscf](casscf.md)).

## X2CAMF: the two-electron picture change

Exact two-component theory would transform the two-electron operator with the same $X$ and
$R$ as the one-electron one, which is prohibitive. X2CAMF applies the transformation to the
converged **four-component atomic mean field** instead — the atomic version of the
mean-field spin–orbit idea [[17]](../references.md#r17), done within X2C. Per unique
element, a spherically averaged four-component Dirac–Hartree–Fock atom
[[19]](../references.md#r19)[[20]](../references.md#r20)[[21]](../references.md#r21) is
converged, its mean field $G[D_{4c}]$ is picture-changed with the identical expression the
one-electron operator uses,

$$
\tilde G = R^\dagger \left( G_{LL} + G_{LS} X + X^\dagger G_{SL}
           + X^\dagger G_{SS} X \right) R ,
$$

and what is **added** to the molecular Hamiltonian is the difference between that and what
the molecular Hamiltonian already contains — the untransformed non-relativistic Coulomb
operator, used unmodified by the CI:

$$
\Delta G \;=\; \tilde G \;-\; G_{nr}[\tilde D]
\;+\; \left( h^{1e}(X_{2e}) - h^{1e}(X_{1e}) \right),
$$

with $G_{nr}$ the ordinary non-relativistic $J - K$ built from the **two-component density**
$\tilde D$ behind the four-component one. Three parts of this expression are load-bearing:

- **The subtraction is the whole ballgame.** Adding all of $\tilde G$ double-counts the
  Coulomb interaction massively; a wrong subtracted term is Hermitian, time-reversal even,
  of plausible magnitude, and wrong — an error class that once needed an external
  four-component reference to find. The subtraction is implemented **once** and shared by
  the atomic (X2CAMF) and molecular (mmf) routes, so the atomic path's committed reference
  numbers validate the molecular one.
- ⚠ **The density is $\tilde D = R^{-1} D_{LL} R^{-\dagger}$, not $R^\dagger D_{LL} R$.**
  Coefficients and densities transform oppositely
  ([notation](../notation.md#density-matrices)), and $R$ is Hermitian positive definite but
  not unitary, so the two differ substantially — while both are Hermitian with plausible
  traces, and their spin–orbit *splittings* agree to 0.2%. What separates them, by five to
  six orders of magnitude, is the X2CAMF total-energy functional against four-component
  Dirac–Coulomb in the same basis (the $c \to \infty$ limit does **not** separate them —
  every plausible variant passes it).
- **The decoupling inside the correction is the converged Fock's**, following the reference
  implementation [[18]](../references.md#r18); the third term compensates for the one- and
  two-electron decouplings being defined by different operators.

The correction is applied in **one place, in the AO basis, before any change of basis** —
so no caller can transform one part and forget the other — and the Hamiltonian carries a
screening *record* stating what it already contains: adding a correction to a Hamiltonian
whose record is not `"none"` double-counts it. ⚠ The correction has a **spin-free part as
well, and it is the larger one**; the two are reported separately and never summed, and
**no reported total energy contains the mean-field double-counting term** — an absolute
Kuiva total is not directly comparable with a four-component total until that term is
accounted for. Relative energies are unaffected.

### The atomic solves, their reference states, and the cache

The price of the default is one four-component atomic SCF per unique element — sub-second
for a light atom, tens of minutes for a lanthanide. Three things about those solves:

- ⚠ **The atomic SCF is constrained to spherical solutions, and fractional occupation is
  not that constraint.** The spherically symmetric point of a fractionally occupied
  Hartree–Fock functional is an *unstable* fixed point — broken-symmetry solutions lie
  below it, and the anisotropy grows roughly an order of magnitude per cycle from roundoff.
  Every atomic SCF (the four-component backend and the scalar AOC reference alike) projects
  the Fock onto its rank-zero spherical part each cycle, through one shared implementation;
  a mean field from an unconstrained solve would carry an arbitrary spatial orientation
  baked in.
- **Open shells are occupied by average of configuration**
  [[21]](../references.md#r21)[[23]](../references.md#r23), not aufbau; the reference
  configuration defaults to the neutral atom — except the f block, which defaults to M(3+),
  on chemistry. `configuration=` overrides per atom
  ([ScalarSCF](../reference/stages/ScalarSCF.md#reference-configurations-configuration)).
- **The solve is geometry-independent and cached persistently**, keyed on
  `(element, basis content, configuration, interaction, nuclear model, c)` — the basis by
  its parsed content rather than its name, the configuration canonicalized, and the speed
  of light included so the $c \to \infty$ tests cannot be fooled by a cache hit
  ([configuration](../guide/configuration.md#the-atomic-mean-field-cache)).

### The interaction: Coulomb, Gaunt, Breit

The four-component reference runs with the Dirac–Coulomb operator by default;
`screening_options={"interaction": "gaunt"}` or `"breit"` selects the magnetic and retarded
two-electron terms [[20]](../references.md#r20). ⚠ The interaction is not part of the
Hamiltonian's *name* — the screening record's `interaction` field, carried in every stored
product's header, is what records it.

### The asymmetry in X

Kuiva's one-electron path uses the exact **molecular** decoupling while the AMF correction
is atomic by construction — its $X$ and $R$ come from the atomic problem. This is standard
for X2CAMF and is not a bug: the atomic approximation is applied to the two-electron
picture change, the term that would otherwise be missing entirely, while the one-electron
part stays exact. It is recorded in the provenance, and `x2c_approx="atom1e"` makes the two
consistent for anyone measuring the difference ([x2c](x2c.md#local-decoupling-dlu)).

## The molecular mean field: `screening="mmf"`

X2C-mmf takes the same subtraction from a full four-component SCF **on the whole molecule**
— the same physics with no atomic approximation and no assembly, through the same
implementation. It exists to measure what X2CAMF's atomic approximation is worth, warns at
the point of selection, and is never production: its cost grows as the fourth power of the
basis, so anything with a heavy element is out of reach — which is why X2CAMF exists. The
two are values of one axis and cannot be combined (combining would double-count the
identical correction); on an isolated closed-shell atom they solve the same problem and
agree to 1e-13 Eh.

`screening="x2camf-external"` routes the correction through the original authors' plugin
[[18]](../references.md#r18) instead — a bisection tool for three-way disagreements, never
a default (it always uses the neutral-atom reference, cannot decouple in a contracted
basis, and refuses a finite nucleus).

Empirical screening alternatives — SNSO/Boettger factors and a Breit–Pauli AMFI
[[28]](../references.md#r28)[[29]](../references.md#r29)[[30]](../references.md#r30) — were
considered and **rejected**, not deferred: their parameters are fitted to neutral atoms and
applied to ions in ligand fields, which is precisely this program's regime.

## Comparing with other codes

Two rules bind any spin–orbit splitting quoted anywhere, both having produced
plausible-looking "method errors": **state which construction produced it** — a
self-consistent two-component calculation and a frozen-orbital diagonalization of the same
operator differ by tens of percent — and **never compare against a reference in a different
basis**: no picture-change correction can recover a basis truncation, and in a small basis
the two errors partly cancel, so a strictly better Hamiltonian can move a total *away* from
experiment. The screening record and the nuclear model are the header fields to match
first. `screening="none"` (keep SOC, drop the screening — a statement about cost) and
`with_soc=False` (a genuinely scalar calculation) are different things:
[workflows](../guide/workflows.md#turning-spin-orbit-coupling-off).
