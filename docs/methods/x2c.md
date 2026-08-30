# The X2C decoupling

Exact two-component (X2C) theory
[[13]](../references.md#r13)[[14]](../references.md#r14)[[15]](../references.md#r15)[[11]](../references.md#r11)
replaces the four-component Dirac problem by a two-component one that reproduces its
positive-energy spectrum exactly, at the one-electron level. Kuiva is an X2C code
throughout: the SCF runs on the spin-free X2C operator, the correlated Hamiltonian on the
full two-component one, and the same transformation machinery serves the molecular
Hamiltonian, the atomic mean field ([soc](soc.md)) and the picture change of property
operators ([properties](properties.md)). Options are on the
[`ScalarSCF`](../reference/stages/ScalarSCF.md) page; this page is the mathematics.

## The transformation

In a restricted kinetically balanced (RKB) basis
[[25]](../references.md#r25)[[26]](../references.md#r26), with the small-component basis
$|S\rangle = (\boldsymbol{\sigma}\cdot\mathbf{p}/2c)\,|L\rangle$, the four-component
one-electron operator and metric are blocked as

$$
h_{4c} = \begin{pmatrix} V & T \\ T & W/4c^2 - T \end{pmatrix},
\qquad
S_{4c} = \begin{pmatrix} S & 0 \\ 0 & T/2c^2 \end{pmatrix},
$$

with $T$ the non-relativistic kinetic energy, $V$ the nuclear attraction, and
$W = \langle \boldsymbol{\sigma}\cdot\mathbf{p}\, V\, \boldsymbol{\sigma}\cdot\mathbf{p} \rangle$
(the operator whose spin-dependent part carries the one-electron spin–orbit coupling). This
is the convention PySCF's four-component and X2C modules use, and Kuiva fixes it once so
every backend must match it rather than invent its own.

X2C finds the **decoupling matrix** $X$ relating small to large components of the
positive-energy solutions, $C_S = X\, C_L$, from the eigenvectors of the four-component
eigenproblem, and the **renormalization** $R$ from

$$
R^\dagger \tilde{S} R = S, \qquad \tilde{S} = S + X^\dagger \frac{T}{2c^2} X .
$$

Any four-component operator with blocks $A_{LL}, A_{LS}, A_{SL}, A_{SS}$ then maps to the
two-component picture as

$$
A_{X2C} \;=\; R^\dagger \left( A_{LL} + A_{LS} X + X^\dagger A_{SL}
              + X^\dagger A_{SS} X \right) R .
$$

Three facts about this expression organize the whole relativistic layer:

- Applied to $h_{4c}$ it gives the **one-electron X2C Hamiltonian** — the correlated
  Hamiltonian's one-electron part, spin–orbit coupling included.
- Applied to the spin-free part only (PySCF's `sfx2c1e` [[10]](../references.md#r10)) it
  gives the **scalar** X2C operator the SCF runs on.
- Applied to a converged four-component **mean field** $G$ it gives the two-electron
  picture change — the content of X2CAMF [[16]](../references.md#r16), covered in
  [soc](soc.md). That the *same* function transforms $h$ and $G$ is the method, not a
  coincidence.

The decoupling is performed **in the decontracted basis** and the result contracted back —
as PySCF's X2C helper also does; a code that decouples in the contracted basis gets a
different (worse) operator. Linear dependence is projected out of the four-component metric
at 1e-7 before decoupling, keeping whole degenerate groups
([integrals](integrals.md#the-degenerate-group-rule)).

⚠ **Two implementations of the exact decoupling exist and they are not bitwise equal.** The
default (`x2c_approx="1e"`) uses PySCF's; the DLU path below is built from Kuiva's own,
which additionally projects the four-component metric. They agree to ~1e-13 on light
molecules and differ by up to ~2.4e-7 relative on a heavy element. A DLU error measurement
therefore uses `decoupling_options={"partition": "single"}` — one fragment covering the
whole basis, which *is* Kuiva's exact decoupling through identical code — as its
like-for-like reference, never `"1e"`.

## Local decoupling: DLU

The exact decoupling costs a dense four-component eigenproblem of dimension $4 n_{ao}$. The
**diagonal local approximation to the unitary decoupling transformation** (DLU
[[27]](../references.md#r27)) replaces the single global transformation by a block-diagonal
one over atoms: $X = \bigoplus_A X_A$, $R = \bigoplus_A R_A$, so that

$$
A^{DLU}_{AB} = R_A^\dagger \left( A_{LL,AB} + A_{LS,AB} X_B + X_A^\dagger A_{SL,AB}
               + X_A^\dagger A_{SS,AB} X_B \right) R_B .
$$

⚠ **Every molecular block is still transformed, including the off-diagonal ones** — that is
the whole difference from the cruder DLH (diagonal local approximation to the
*Hamiltonian*), which is not implemented and not wanted: it costs the same, is
substantially worse, and breaks translational invariance, which DLU does not. Because
block-diagonal $X$ and $R$ make the DLU expression *be* the exact `picture_change` applied
to them, there is no second transformation code path that could drift: DLU differs from
exact X2C only in how $X$ and $R$ are obtained.

Where the local problem comes from is a real convention, recorded in the provenance:

- `source="diagonal"` (default): the diagonal block of the *molecular* matrices — the
  fragment sees every nucleus's attraction, needs no extra integrals, and cannot be
  misassembled (it is a slice).
- `source="isolated"`: a separate isolated-fragment problem — transferable and cacheable
  across a potential-energy surface, with a failure mode the diagonal source cannot have (a
  permuted AO block would be Hermitian, plausible and wrong; the overlap comparison refuses
  it).

Selecting DLU **warns**: it is the escape hatch where the exact decoupling is prohibitive,
never a cheaper default. Measured state-level accuracy against the single-fragment exact
reference (d¹ and f¹ ligand fields, state-averaged CASSCF): splittings within 0.6 cm⁻¹ and
0.1%, principal g values within 2e-4 relative — ⚠ but the near-zero transverse g of a
strongly axial doublet moved by ~6% of itself, which is exactly the number a tunnelling
analysis reads; check it against the exact decoupling before quoting it. The DLU-transformed
property operators remain unmeasured.

`x2c_approx="atom1e"` is a third construction — PySCF's block-diagonal isolated-atom $X$
with the full **molecular** $R$. ⚠ It is not DLU and not cheaper in scaling (the
$O(n_{ao}^3)$ work stays in $R$); it exists because it makes the one-electron decoupling
consistent with the atomic mean field's, for anyone measuring that difference
([soc](soc.md#the-asymmetry-in-x)).

## The nuclear charge model

The third axis of the Hamiltonian, beside decoupling and screening — deliberately absent
from every method *name*, because it is a property of the potential the integrals were
evaluated over. The nucleus is a **point charge by default** (what every committed
reference number was produced with); `nuclear_model="gaussian"` selects the finite Gaussian
distribution of Visscher and Dyall [[24]](../references.md#r24),

$$
\rho(r) = Z\, N\, e^{-\zeta r^2}, \qquad \zeta = \frac{3}{2\langle r^2 \rangle},
$$

with the rms radius parametrized from the mass number of the most abundant isotope. One
statement per molecule, inherited by **every** consumer — the molecular integrals, the
atomic mean-field solves, the free-atom reference orbitals, the DLU fragments — because an
atomic mean field over a different nucleus from the molecular integrals is Hermitian,
plausible and wrong; a mixed molecule is refused rather than summarized.

The effect is concentrated at the nucleus and grows steeply with Z: on core-region
j-splittings, under 1e-6 relative at neon, ~5e-4 at krypton, ~3e-3 at mercury. ⚠ A
*valence* property moves far less (measured ~40× less on Tl's valence ²P splitting; a
ground-doublet g factor never moved beyond 1.2e-5 anywhere measured), so the option matters
for one identified case — a heavy-element ligand-field spectrum quoted to better than ~0.1
cm⁻¹ absolute. It is part of the Hamiltonian, recorded in every provenance header, and the
mean-field cache keys on it. Several four-component programs (DIRAC among them) default to
the Gaussian nucleus: the first thing to match in a cross-code comparison.

## The speed of light

$c$ belongs to whatever produced the integrals — PySCF's $c = 137.03599967994$ on this
front end — never to a constant chosen elsewhere; the CODATA 2018 value
[[184]](../references.md#r184) is used for reporting only. The distinction is measurable:
building the four-component blocks at the CODATA value against PySCF integrals degrades the
X2C agreement from ~6e-13 to ~1e-11 relative — far below physics, far above the validation
level, and a numerical discrepancy with no physical cause. See
[notation](../notation.md#units-and-physical-constants).
