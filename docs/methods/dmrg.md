# The tensor-network solver (DMRG/TTNS)

The multireference solver beyond the conventional CI's determinant ceiling, and the route
to multi-site magnetic systems: an in-house, **tree-native** implementation of the density
matrix renormalization group
[[117]](../references.md#r117)[[118]](../references.md#r118) as tree tensor network states
[[123]](../references.md#r123)[[124]](../references.md#r124)[[125]](../references.md#r125)[[126]](../references.md#r126),
Kramers-**un**restricted and `complex128` throughout — the relativistic (general-spinor)
DMRG setting [[135]](../references.md#r135)[[136]](../references.md#r136). The matrix
product state is the path special case: one topology owner, and no sweep, operator or
environment code knows "left" from "right". It plugs into the shared orbital optimizer as
one more CI solver ([casscf](casscf.md)), through the adaptive-solver protocol. Options:
[`CASSCF`](../reference/stages/CASSCF.md#solverdmrg-the-tree-tensor-network).

## Conventions that bind everything

- **Jordan–Wigner order is the global ascending mode index**
  [[137]](../references.md#r137) — the *same* convention as the determinant machinery, which
  is what makes the dense TTNO equal the CI Hamiltonian matrix element for element with no
  phase fixups. The JW order is a property of the modes, not the tree: correctness never
  depends on the topology, only the compressed bond dimensions do. ⚠ Consequence: a leaf
  swap is *not* a reordering — "tree moves subsume orbital ordering" holds for spin models
  only.
- **Quantum numbers are abelian, tuple-valued, particle number first**, on block-sparse
  tensors [[134]](../references.md#r134) with a signed flux convention; abelian double-group
  irreps widen the tuple with no rewrite ([symmetry](symmetry.md)); non-abelian adaptation
  stays out of scope.
- The TTNO and the sweep energies **exclude the core energy**, exactly as the CI layer's
  sigma operator does; the caller adds it when reporting molecular energies.

## The operator: a compiled TTNO

The Hamiltonian (chemists'-notation integrals, 4-fold symmetry only) is compiled
symbolically into one tree tensor network operator: every term is reduced to a product of
one local matrix per mode (the JW mapping absorbing all signs), and for each tree bond a
term is assigned a **content label** — identity, completed, *normal* (labelled by the
inside factors) or *complementary* (labelled by the outside factors, coefficients folded
into the flowing sum: the complementary-operator idea
[[119]](../references.md#r119)[[120]](../references.md#r120)). Labels are complete
descriptions of operator content, so distinct terms sharing a label genuinely share the
state — the entire compression, taking the ab initio Hamiltonian to the classic $`O(n^2)`$
operator bond dimension [[121]](../references.md#r121)[[122]](../references.md#r122)
instead of the $`O(n^4)`$ term count, with each coefficient attached exactly once. The
compiler's input is a generic operator-sum, which is a deliberate test seam: the structure
machinery can be driven by model spin Hamiltonians with *known* exchange graphs, no
integrals involved — and structure claims are validated against theorems and those models,
never against the network's own entanglement output.

## The sweep

A state-averaged **two-site** update over the Euler tour of the tree, with the local
eigenproblem solved by the same block Davidson the CI uses ([ci](ci.md#the-eigensolver)) —
reused, never duplicated — and environments cached per directed bond, refreshed right
after the center crosses (the tour's depth-first structure keeps every cached environment
current with no invalidation logic; a topology change keys the cache by subtree content
instead).

- **The state average is a shared-basis average** [[127]](../references.md#r127): one tree
  of isometries, one center tensor per root. The ensemble truncation stacks the roots on an
  auxiliary leg weighted by $`\sqrt{w_r}`$, so **one** SVD performs the weighted
  density-matrix truncation.
- ⚠ **Every truncation keeps degenerate groups whole, through exactly one code path**: the
  SVD truncates on the *merged* singular-value spectrum across all symmetry sectors (a
  degenerate group can straddle sectors — a Kramers partner carries the same particle
  number, an orbital multiplet need not sit in one block), detects degenerate groups at a
  relative tolerance, and applies its accuracy and stability floors to whole groups. A cut
  it cannot make is **refused, not rounded**. A degenerate Schmidt pair — Kramers — is kept
  or dropped whole across the whole ensemble.
- The value threshold plus the cap give dynamic-block-selection behaviour
  [[128]](../references.md#r128); the default is exact-within-the-cap. `expansion=` is the
  **deterministic subspace expansion**
  [[130]](../references.md#r130) — White's density-matrix noise
  [[129]](../references.md#r129) evaluated instead of sampled — chosen because the
  deterministic term keeps degenerate Schmidt groups exactly degenerate and no RNG enters
  any trajectory; it decays off over the first sweeps and energies stay variational at
  every strength.
- **The state-averaging discipline is split**: mid-sweep, weights are equalized within
  *observed* degenerate blocks of the local spectrum and nothing more (unconverged
  environments do not span time-reversal-closed spaces, so the structural odd-electron
  argument does not yet apply); the full averaging gate — with its theorem-backed refusal —
  is applied to the **converged** spectrum, where the RDM-consuming weights come from. A
  local two-site analogue of the boundary diagnostic reports the gap above the averaged set
  at convergence (warning below the same 50 cm⁻¹) — necessary, cheaper and **weaker** than
  the full-CI one: it cannot see a state the current bond dimension cannot represent.
- ⚠ **Kramers degeneracy between paired roots comes out around 1e-9 Eh** even where nothing
  is truncated — the sweeps converge roots separately — larger than the conventional CI's
  figure and for a different reason; both are far below physics, and neither bounds the
  other.

⚠ **Read `w_disc`.** The largest discarded ensemble weight is the network's primary quality
number, and every energy from this solver is quoted with it; truncation *growing* as the
orbitals move is the signal the cap is too small. The $`E(w_{\mathrm{disc}} \to 0)`$
extrapolation [[131]](../references.md#r131)[[132]](../references.md#r132) is a separate
driver over a converged problem (`kuiva.dmrg.bond_series`), reporting the extrapolate with
the series and its fit residual beside it, never alone.

⚠ **Threaded BLAS buys this layer nothing** — a sweep takes the same wall time at 1 and 8
threads while spending several times the CPU on spin-wait — so the network solver runs at a
small thread width, and its CPU-second figures are read accordingly
([configuration](../guide/configuration.md#threads-one-number)). The environment cache
pages its coldest entries to a configured scratch directory under memory pressure,
bitwise-inertly, instead of refusing.

## Topology, adaptivity, and the optimizer

The topology is seeded from the cheap CI's entanglement (mutual-information graph, Fiedler
ordering — [active-spaces](active-spaces.md#the-cheap-ci-pre-optimization)). Adaptive
topology — local reconnection moves in the spirit of automatic structural optimization
[[133]](../references.md#r133) — and the per-macro-iteration bond-cap ladder are **chart
changes** offered through the propose/adopt seam of the event-gated optimizer
([casscf](casscf.md#adaptive-solvers-the-optimizer-owns-the-space)): evaluated at fixed
integrals, adopted only when they lower the energy, with curvature memory cleared on
adoption. RDMs come back in the same objects and conventions as the CI's, through the same
state-averaging gate, so the orbital optimizer is untouched.

## Densities and the local-multiplet model

**Ranks 1–2, the production path**: one backward pass computes every node's operator
environment $`\partial E / \partial W_u`$, and each elementary operator's expectation is read
out of its coefficient-attachment slot — about two sweeps' worth of environment builds per
macro-iteration, independent of $`n^4`$. **Ranks 1–4, the direct-contraction path**
[[138]](../references.md#r138)[[139]](../references.md#r139): Gram matrices of *annihilated
states* $`\chi_A = a_{A_1} a_{A_2}\cdots|\psi\rangle`$ — a JW string changes node tensors
without touching any bond, so every $`\chi_A`$ shares the state's bond dimensions and each
overlap is one tree contraction. Exact and cumulant-free (the chosen route for the NEVPT2
densities — cumulant approximations are known intruder generators), and scope-limited by
design: the dense $`n^{2k}`$ result array is refused by the memory budget long before the
contraction count matters, which is why the perturbation is served by contractions instead
([nevpt2](nevpt2.md#the-network-route)).

For a polymetallic system the low-energy manifold is a **product of local multiplet
spaces** — thousands of states for three f ions, not viable as individual roots — and the
manifold layer inverts the problem: each site's ensemble RDM is diagonalized (same SVD,
same group discipline; ⚠ the multiplet cut must land on a reported spectral gap, refused
through a degenerate group), its dominant eigenspace becomes the site multiplet, and

```math
H_{\mathrm{eff}} = \Big(\textstyle\bigotimes_k V_k\Big)^\dagger H
                   \Big(\textstyle\bigotimes_k V_k\Big)
```

is contracted with **open multiplet indices** — the whole product family costs one
contraction, never $`\prod_k d_k`$ solves. $`H_{\mathrm{eff}}`$ is a Rayleigh–Ritz projection
onto an orthonormal product basis, so every model eigenvalue bounds its exact counterpart
from above (Cauchy interlacing — asserted exactly in the tests, independent of how good the
product approximation is); the construction is the zeroth step of the CORE program
[[142]](../references.md#r142), with the Bloch/des Cloizeaux improvements
[[140]](../references.md#r140)[[141]](../references.md#r141) deliberately not implemented.
The same open-index contraction serves the moment operators, and the result feeds the
pseudospin export ([properties](properties.md#the-pseudospin-export)).
