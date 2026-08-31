# The working basis and the two-electron integrals

Everything downstream of ingestion is built on the **orthonormal working basis** — not the
raw AO basis — with near-linear dependence explicitly removed, and on a **three-index
factorization** of the two-electron integrals from which four-index blocks are assembled on
demand and never stored whole. This page is the mathematics of both, plus the projector
that carries orbitals between basis sets. Options are on
[`ScalarSCF`](../reference/stages/ScalarSCF.md#the-two-electron-route) and
[`Reference`](../reference/stages/Reference.md).

## Canonical orthogonalization

Diagonalize the AO overlap, $`S U = U s`$ with $`s`$ sorted **descending**, and keep the $`k`$
columns above the threshold (default $`10^{-7}`$):

```math
X = U_k\, s_k^{-1/2}, \qquad X^{\mathsf T} S X = \mathbf{1},
```

so a truncation is a slice, never a scatter
[[44]](../references.md#r44)[[45]](../references.md#r45)[[47]](../references.md#r47)[[46]](../references.md#r46).
Columns with small overlap eigenvalues are the near-linearly-dependent combinations;
keeping them multiplies numerical noise by $`s^{-1/2}`$, the classic route to a silently
wrong correlated calculation — and near-linear dependence is the *expected* case here, not
an exotic one, since uncontracted and mixed heavy-element bases are deliberately supported.
Dropping a vector is a deliberate, **reported** reduction of the variational space.

Symmetric (Löwdin) orthogonalization ($`X = S^{-1/2}`$, the orthonormal basis closest to the
AOs [[53]](../references.md#r53)) is available but is square by construction — it cannot
remove a dimension — so it **raises** rather than return a rank-deficient basis wearing the
name "orthonormal"; the Cholesky scheme [[49]](../references.md#r49) performs no
linear-dependence detection at all. Column phases are fixed (largest element positive), so
restarts and run-to-run comparisons do not inherit LAPACK's unguaranteed sign convention.

### The degenerate-group rule

⚠ **A basis that drops directions drops whole degenerate groups, and a group straddling the
threshold is dropped whole rather than kept whole** — a retained direction below the
threshold amplifies noise by $`s^{-1/2}`$, and a cut *through* an exact degeneracy breaks it
by the truncation threshold. One implementation of this rule (a shared grouping-and-cutting
utility) serves the working basis and the four-component metric projection; the same
discipline recurs, in its own terms, in the Cholesky pivoting below, the state-averaging
gate ([casscf](casscf.md)), the network truncations ([dmrg](dmrg.md)) and the NEVPT2
contraction ([nevpt2](nevpt2.md)). Inside an exactly degenerate block the eigenvectors
remain defined only up to a rotation — no convention can fix that, and nothing downstream
may depend on it.

## The three-index factorization

Both factorization routes produce the *same* object: real AO factors $`L^P_{\mu\nu}`$ with

```math
(\mu\nu\,|\,\lambda\kappa) \;=\; \sum_P L^P_{\mu\nu}\, L^P_{\lambda\kappa},
```

so downstream code never asks whether density fitting or Cholesky produced them.
Four-index integrals are assembled on demand for the blocks that need them (the active
space, the gradient blocks, the NEVPT2 classes) — the difference between an $`N^4`$ array and
an $`N^2 K`$ one.

**Cholesky is the default** [[74]](../references.md#r74)[[75]](../references.md#r75): its
threshold is a rigorous error bound the user sets — every integral error is bounded by
$`\sqrt{d_i d_j} \le \tau`$ — where a fitting error is not bounded at all. Density fitting
[[79]](../references.md#r79)[[80]](../references.md#r80)[[81]](../references.md#r81) stays
fully supported on request, with the auxiliary's accuracy the user's responsibility; ⚠ a
**Coulomb-fitting (J) auxiliary is not accurate enough for correlated integrals**, by
orders of magnitude — J-fitting sets are calibrated for the Coulomb *matrix*, where errors
cancel against the density, and an individual $`(pq|rs)`$ gets no such cancellation — so
Kuiva warns whenever the auxiliary is Coulomb-only.

### One-centre pivoting: symmetry enforced structurally

⚠ **A pivoted Cholesky error is anisotropic**: plain column pivoting treats the $`m`$
components of one shell inequivalently and **splits degeneracies that symmetry makes
exact** — a qualitative failure landing squarely in the 1e-8…1e-6 Eh band reserved for
genuine numerical splittings. The default is therefore the atomic (one-centre)
decomposition of Aquilante, Lindh and Pedersen [[78]](../references.md#r78): pivots are
**complete shell-pair orbits**, so rotational invariance of the factorization holds
*exactly, independent of the threshold* — and it is faster. Plain pivoting remains
available (`orbit_pivots=False`) for measuring exactly that, not for production.

### The default threshold, and what it does and does not bound

The default $`\tau = 10^{-8}`$ was decided on **relative** energies, where the factorization
error cancels by a measured ~500×. An absolute total is bounded by the *basis*, not by
$`\tau`$: one step up a basis series moves a total by six to eight orders of magnitude more
than tightening $`\tau`$ to $`10^{-10}`$ does. Tighten it only to compare two totals in the
same basis.

### Three routes to the decomposition, one result

How the integrals reach the decomposition is a value on the `fitting` axis, resolved by the
memory plan by default: the **stored** route materializes the conventional ERI array and
releases it the moment the factors exist; the **direct** route
[[76]](../references.md#r76)[[77]](../references.md#r77) evaluates each column as the
pivoting asks for it and never forms the array (measured within a few percent of the stored
route in CPU from ~160 AOs up — the array is the entire decision). Where the finished
factors live is a second axis: spilled to scratch and streamed back (bitwise identical), or
the decomposition itself run out of core (⚠ not bitwise — the same subtractions summed in a
different order, agreeing to rounding, ties between symmetry-equivalent columns possibly
broken the other way: a different but equally valid factorization with the same error
bound). Consequences and knobs:
[ScalarSCF](../reference/stages/ScalarSCF.md#the-two-electron-route).

## The spinor-MO transformation

The AO integrals are **spin-free** — the Coulomb operator does not act on spin — so in the
spinor basis the spin sum collapses into the transformation matrices
[[84]](../references.md#r84)[[21]](../references.md#r21)[[82]](../references.md#r82)[[83]](../references.md#r83):

```math
B^P_{pq} \;=\; \sum_{\sigma\in\{\alpha,\beta\}} \sum_{\mu\nu}
   C^{\sigma *}_{\mu p}\; L^P_{\mu\nu}\; C^{\sigma}_{\nu q},
\qquad
(pq\,|\,rs) \;=\; \sum_P B^P_{pq}\, B^P_{rs}.
```

⚠ **The assembly carries no complex conjugate**: the conjugation lives inside $`B`$, on the
bra index of each charge-cloud pair, and applying another one in the assembly is a silent,
plausible-looking error that survives every hermiticity check. The cost is twice a scalar
transformation of the same AO dimension — not the sixteen times a naive $`2n_{ao}`$
"AO-spinor" treatment would cost — and the resulting integrals have **4-fold permutational
symmetry only** ([notation](../notation.md#second-quantization-and-determinants)).

⚠ **No path transforms the square $`B^P_{pq}`$ over all orbitals.** The optimizer holds
$`B^P_{p,\mathrm{active}}`$ (one active index — a fraction of the array), the perturbation
one block per space pattern, and the memory pre-flight plans those blocks: a budget that
refused on an array nobody allocates would teach the user to raise the limit blindly.

## Carrying orbitals between basis sets

The one projector, with two consumers (`CASSCF(project_from=)` and the SCF's
`guess_from=`). An orbital over the source AO basis is carried onto the target by the
orthogonal projector onto the target's **working basis**:

```math
c_{\mathrm{work}} \;=\; X^{\mathsf T}\, \langle \mathrm{AO}(t) | \mathrm{AO}(s) \rangle\, c ,
```

and no linear system is solved at all. The textbook form
$`S_{tt}^{-1}\langle t|s\rangle\, c`$ [[50]](../references.md#r50) is identical whenever
$`S_{tt}`$ is invertible — and exactly the wrong thing to write down when it is nearly not,
which for large heavy-element bases is the normal case: inverting $`S_{tt}`$ amplifies noise
by the same $`s^{-1/2}`$ the working basis was built to discard, while the form above
*inherits* the linear-dependence removal for free. The projector is real and
spin-independent, hence commutes with $`\hat T = -i\sigma_y K`$: **a Kramers-paired set
projects to a Kramers-paired set**, and a pair cannot be split by it.

A projector is not unitary, so the projected set is re-orthonormalized — by default space
by space (`scheme="blocked"`), keeping the CAS partition exact rather than taking the
least-squares-optimal global Löwdin [[53]](../references.md#r53); and by default only the
**active** orbitals cross (`carry="active"`), the inactive and virtual ones coming from the
target's own SCF — the source's inactive orbitals are eigenvectors of nothing in the target
basis, and carrying them re-introduces a core–virtual gradient the target SCF had already
removed (measured at up to twice the macro-iterations). ⚠ A converged general-complex
CASSCF's active orbitals are *not* pair-aligned — active–active rotations are redundant, so
nothing pushes back — and the carried blocks are rebuilt as explicit Kramers pairs.

⚠ **Every way of getting a projection wrong still yields an orthonormal set of the right
shape that converges**, so the reported diagnostics are load-bearing, not decoration: the
retained norm of each source orbital, the principal overlaps
[[54]](../references.md#r54) between the source active space and the space handed on
(invariant under rotations inside either), and the complement separation. The active space
follows the orbitals and is never re-selected against the target's guess — re-selecting
would define a different calculation wearing the same name, and is refused. The workflow
precedent is OpenMolcas' `EXPBAS` [[51]](../references.md#r51); usage:
[workflows](../guide/workflows.md#a-casscf-from-a-different-basis-set-project_from).
