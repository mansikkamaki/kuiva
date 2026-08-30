# Point-group and double-group symmetry

Symmetry in Kuiva is **opt-in, abelian, and honest about its limits**: an abelian
double-group label vocabulary that the front end assigns to orbitals, the CI blocks its
determinants by, the orbital optimizer masks rotations by, and the tensor network conserves
— plus a second, non-abelian layer that **classifies** converged states and never adapts
anything. Labels alone change nothing; what they enable (per-irrep state selection, the
symmetry-preserving rotation mask, a conserved network sector) is asked for separately.
Options: [`ScalarSCF`](../reference/stages/ScalarSCF.md) and
[`CASSCF`](../reference/stages/CASSCF.md#per-irrep-selection).

## What a label is

An abelian group's irreps are one-dimensional, so an orbital that is an eigenvector of
every operation carries one character per operation. Writing the group as a product of
cyclic factors $\langle g_1\rangle \times \cdots \times \langle g_k\rangle$ of orders
$n_1 \ldots n_k$,

$$
\chi(g_j) = e^{2\pi i\, m_j / n_j}, \qquad m_j \in [0, n_j),
$$

so **a label is a tuple of integers with per-component moduli**: composition is
componentwise addition modulo $n_j$, and complex conjugation — what time reversal does to a
character — is componentwise negation. A determinant's label is the group sum of its
occupied spinors' labels, and the Hamiltonian is exactly block-diagonal over the sectors
whenever the orbitals are symmetry-pure (checked at every solve, never assumed).

⚠ **The group is the DOUBLE group
[[65]](../references.md#r65)[[64]](../references.md#r64), and its generators are not the
spatial operations**: a spatial $C_2$ squares not to the identity but to the $2\pi$
rotation $\bar E$, which acts on a spinor as $-1$ — so $C_2$ has order four, the moduli are
the point of the vocabulary, and a fermion (double-valued) irrep is recognized by
$\chi(\bar E) = -1$. Only groups whose *double* group is abelian have one-dimensional
fermion irreps: in the $D_{2h}$ chain that is $C_1, C_i, C_2, C_s, C_{2h}$ — the double
groups of $C_{2v}$, $D_2$ and $D_{2h}$ carry a two-dimensional fermion irrep, so those
requests are **reduced to the largest subgroup that can label a spinor with a number**,
reported, never silently. Working in the abelian subgroup of a relativistic code is the
standard design [[21]](../references.md#r21)[[66]](../references.md#r66); adapting to a
two-dimensional fermion irrep would be non-abelian adaptation, which is out of scope.

Two consequences of the spinor conventions:

- ⚠ **The spin-active generator must be about z** — spin is quantized along z, and a
  rotation about any other axis mixes the members of a Kramers pair. The molecule is
  **never reoriented** (that would move the gauge origin and every property operator fixed
  at ingestion), so a symmetry axis that is not z is reported and **not used**; orienting
  the input is the fix, and the message says so.
- **Degenerate orbitals are symmetry-adapted, not refused**: the SCF may return any mixture
  of two degenerate orbitals, and a mixture is an eigenvector of nothing. The adapting
  rotation stays inside the degenerate block, so no density, energy or observable changes;
  a residual that survives it is refused with the orbital and operation named.

**The character table of the group used in the mathematics is printed whenever symmetry is
on** — boson and fermion irreps, every operation named by its lab-frame geometry
(`C2(z)`, `sigma(xy)`), never by a Schoenflies label whose orientation the reader must
guess: two programs agreeing on "$C_{2h}$, $B_u$" and disagreeing on which axis is z
produce different numbers and no error message. The printed characters are checked against
$\mathrm{tr}\, U(g)$ computed from the run's own operator matrices.

## What abelian symmetry cannot promise

Wherever the molecule's real group is bigger than the abelian one used — every atom, and
every group reduced above — **physically degenerate partners can carry different labels**.
A per-irrep state count can therefore split a degenerate manifold exactly as a plain count
can, and ⚠ **a per-irrep count is not a safety mechanism**: the state-averaging gate, the
boundary-gap diagnostic and the spin-non-invariance report stay load-bearing *with*
symmetry on ([casscf](casscf.md#the-state-average)). "Safe by construction" is claimed only
where the abelian group is the whole story. A single state inside a degenerate block has no
irrep of its own — the eigensolver may return any rotation of the block — so
classification, like every other per-state quantity, is **per degenerate block**
([notation](../notation.md#degenerate-blocks-and-arbitrary-phases)).

## The non-abelian layer: classification, never adaptation

The second layer names each converged degenerate block by the irreps of the molecule's
**full point double group** — so a multiplet the abelian labels cannot see gets a name and
a theory-fixed dimension — and changes no number: no symmetry-adapted many-particle basis,
no double-group coupling coefficients, nothing that could grow into adaptation.

- **The character tables are computed, not transcribed.** Stored are only the *generators*
  as explicit $(3{\times}3\ \text{spatial}, 2{\times}2\ \text{spin})$ matrices — correct by
  inspection — and the group is their closure in $SO(3) \times SU(2)$, the double group
  arising by construction ($C_2(z)^2 = \bar E$). Characters follow by the class-sum
  construction [[68]](../references.md#r68) in Dixon's numerical form
  [[69]](../references.md#r69); irrep names are assigned by rule
  [[70]](../references.md#r70) and checked against published tables in the *tests* — where
  transcription belongs. The group operators on the AO basis are built from real
  solid-harmonic rotation matrices by recursion [[67]](../references.md#r67).
- **Three tables print together or the labels are folklore**: the abelian table used in the
  mathematics, the full table used in the labelling, and the **computed** subduction
  between them (each non-abelian irrep decomposed into abelian sectors by the ordinary
  projection formula [[71]](../references.md#r71)) — which is what connects a per-irrep
  request to the physical multiplet it selects.
- **How $U(g)$ reaches a CI vector.** A non-abelian element mixes partner spinors, so its
  Fock-space action is not a signed permutation. Every unitary factors as
  $u = G_1 G_2 \cdots G_m D$ with each $G$ a rotation of two **adjacent** modes and $D$
  diagonal (the Givens QR restricted to adjacent rows
  [[73]](../references.md#r73)), and the Fock-space representation is a homomorphism
  [[72]](../references.md#r72), so $U(u)$ is the same product of elementary factors, each
  one pass over the determinant coefficients. ⚠ Adjacency is the point: the fermionic phase
  of the two-mode mix is $(-1)^{\#\{\text{occupied between}\}}$, which is $+1$ for
  neighbours — a general pair would need that popcount, and a sign error there is
  norm-preserving, Hermitian-looking and wrong. The block traces are then projected onto
  the group's characters, and the reported leakage is the block's distance from whole
  numbers — the part that is a statement about the *orbitals*.
- **What it refuses** is a state count that cuts a multiplet whose dimension theory fixes.
  ⚠ **What it does not protect**: *near*-degeneracies of different irreps — two multiplets
  a few wavenumbers apart are a different problem, and the boundary diagnostics remain the
  only evidence there. Every failure to classify degrades to a warning, never an error.

The tensor-network solver takes the same labels one step further, as a **conserved quantum
number** on the block-sparse tensors — a labelled sweep cannot leave its sector
([dmrg](dmrg.md#conventions-that-bind-everything)). An AVAS-rotated space carries no labels
at all ([active-spaces](active-spaces.md#avas-the-covalent-case)), and a lower-symmetry
embedding field restricts them and switches classification off
([Molecule](../reference/stages/Molecule.md#environmentpoint_charges-)).
