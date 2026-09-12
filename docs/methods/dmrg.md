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
never against the network's own entanglement output. The suite drives every polynuclear
validation graph (a chain, an ion–radical–ion chain with local dimensions 16/2/16, a
frustrated triangle, a star, a cubane face with a pendant, a complete graph and two
eight-site rings) through that seam as an effective Heisenberg model, one mode per centre,
and checks the *committed* state's energy and $`\langle S^2\rangle`$ against dense or
sparse exact diagonalization and the Lieb–Mattis ground spin. Two things that measurement
settled: the star's tree network is exact at a bond dimension of six where a chain needs
eleven, and an entanglement-driven ordering cannot see a low-dimensional bridge — a radical
between two large-J ions shares at most $`\ln 2`$ nats with either — so a multi-site
topology is taken from the exchange graph through the localization's site partition, not
from the mutual information of a converged state.

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
- ⚠ **Kramers degeneracy between paired roots is machine zero at a saturating cap** (1e-16 Eh
  at fixed orbitals, 1e-12 Eh through a full CASSCF, where the residual is the orbital
  optimizer's convergence) — and where a residual does exist it is the **truncation error**,
  equal to the state-averaged energy error within a factor of two over five orders of
  magnitude: a network run whose paired states sit *X* apart has an energy roughly *X*/2 from
  the exact one. It is neither a symmetry defect nor a convergence artefact, and the CI path's
  figure is not a bound on it.

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

**The effective Hamiltonian is applied to an iteration's new Davidson directions in
batches.** On a state average that is roughly one direction per root, and applying them one
at a time paid the contraction chain's bookkeeping — and a full set of small block GEMMs —
once per root: a 25-root FeCl2 bond tour made 1575 single-vector applications for 59
Davidson iterations. Batched, the vectors ride on one extra leg of every intermediate and
the bookkeeping is paid per batch. The width is a **measured default of four**
(`solve_ttn(batch=)`): two to four vectors pay the bookkeeping off, and a batch of every
root of a large problem grows every intermediate out of the cache and runs slower than none.
It is further bounded by the resource budget's transient allowance from the single-vector
intermediate the memory plan sizes, so batching never enlarges what the plan promised
beyond that allowance. The pair tables every contraction needs are built once per operand
structure and memoized (they hold indices only, so the memo is bitwise-inert), and the
operand matricizations run through a compiled kernel where one is built.

### What a sweep costs in memory

⚠ **The largest array is not one that is stored** — it is a single intermediate inside one
application of the effective Hamiltonian, built and freed once per Davidson matrix-vector
product, and its size is a *contraction order*. An intermediate carrying two operator legs at
once costs the product of two operator bond dimensions: on a 20-spinor active space
partitioned over four five-mode nodes it reached **4.3 GB at a bond dimension of 4** and
about 11 GB at 16, while everything the run stored came to 0.13 GB, and the run was killed by
the operating system rather than refused.

Kuiva's chain keeps **one operator leg open at a time**: the first side contracts its
environments before its operator tensor, the second side its operator tensor before its
environments. On a chain that is a rule and the same intermediate is 8 MB. A node with several
branches opens one leg per branch, and there the order — which environments are folded into
their operator tensor once, ahead of the solve, and reused across every Davidson iteration —
is chosen by walking every candidate over the contraction's own block structure, with no data
and nothing allocated, and taking the smallest peak. ⚠ A changed contraction order sums the
same products in a different order: energies agree with the previous order to about
$`10^{-13}`$ Eh, not bitwise. The reorder is also cheaper in CPU, by 4× on thin chains and
8–30× on fat nodes and trees, because the old third step contracted the two-leg
intermediate itself.

The sizing runs through the *same* chain — spaces, signs and sector tables, walked in
structure-space — for every bond of the sweep, and the resulting plan is printed before the
first bond is solved. A solve that cannot fit is **refused with the bond named**, and the knobs
are stated in the order they matter:

1. the **node partition** — the local dimension is $`2^{\text{modes at the node}}`$ and enters
   squared, so more nodes with fewer modes each is the largest single lever;
2. the **active-space size** — the operator bond dimension grows as its square, and the
   intermediate carries one of them;
3. `max_bond` — quadratic through the two-site dimension.

The RDM contraction after the sweep is planned the same way: its per-node operator
environments are dense in the node's local dimension squared with one operator leg per
neighbour, so on fat or branching nodes they, not the sweep, are the largest term, and a cap
whose environments do not fit is refused with the node named. A fixed-orbital network CASCI
that only needs energies and transition densities skips that extraction with
`DMRGSolver(rdms=False)`; an orbital optimization cannot, since the RDMs are its input.
The environment builds and the RDM path's messages run the same one-leg chain, with the
same structural choice of which environments to fold into the operator on a branching node,
and their transient is in the plan. The Davidson preconditioner's diagonal is dense over every operator leg of a node at once,
so on a node with many neighbours it is sized first and dropped where it would exceed the
kernels' transient budget: the solve then runs unpreconditioned at that bond, slower but
identical in what it converges to.

⚠ The estimate describes Kuiva's arrays. What remains above it is the operator compile's own
transient and the arena it leaves behind — a process that has freed a multi-gigabyte array does
not usually return it to the operating system — so leave a factor of about two of headroom in
the configured limit.

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

## Resolving a state count on the network

The energy-window form of `n_states` ([casscf](casscf.md#resolving-the-count-from-an-energy-cutoff))
works here through the same rule and the same round loop; what differs is the cost and the
kind of evidence a rung can produce, and both are stated in the output rather than assumed.

**A rung is a whole sweep campaign.** The ladder asks the network for the lowest $`n`$ roots,
which is a state-averaged solve to convergence on a cold state — so the rung table prints the
sweeps and the CPU seconds each rung took, and an expensive ladder is visible rather than
inferred. Growth is by a factor of 1.5 rather than the CI's doubling for the same reason, and
a grown rung starts **cold**: warm-starting across a root-count change by padding the
incumbent shared basis with random centers is the obvious variant and is an unmeasured one,
so it stays out until it is measured — what it would buy is time, never correctness.

**The witness roots are converged network roots.** A rung solves the count *plus a whole
Kramers pair* and reads the gap between the last state inside the window and the first
outside it. The cheaper alternative — one extra root of a single two-site problem, which is
what the sweep's own boundary diagnostic measures — bounds the next eigenvalue **from above**,
so it can prove a window *incomplete* and never that it is complete. ⚠ The resolution
therefore states which kind of witness it used, every time, and the network's local
diagnostic is never silently substituted for it.

**The topology decides whether a window can be answered at all.** The two-site space at the
narrowest bond of the tour bounds the whole ensemble no matter how large `max_bond` is: for
a one-electron active space a tree of one-spinor nodes holds three roots, two in whole
Kramers pairs, which leaves no room for a witness above a two-state average. ⚠ A window that
cannot be given a witness is **refused**, with the capacity, the particle-number sector and
the fix named — truncating the ensemble to the roots the network happens to be able to hold
would be exactly the arbitrary cut a window exists to forbid. The fix is a coarser node
partition (fewer nodes, more spinors each), which is also what raises the capacity fastest.
That bound is measured from the *full* sector set a freshly canonicalized state carries, so
a cap tight enough to truncate a bond can put a rung past what that bond actually holds; the
sweep refuses that too, and the ladder re-raises it saying the roots were a rung's rather
than a stated count, because the two have different fixes.

**The first rung comes from a pilot** when nothing upstream supplies one: a short campaign
(four sweeps) at a small bond dimension (8) over a generous root count, whose spectrum the
rule reads for the first rung only. A pilot is variational from above per root and its
splittings are rough — which is what a first rung is for, and why the rule is re-run on the
production spectrum rather than trusted there. A [`CheapCI`](../reference/stages/CheapCI.md)
upstream supplies the estimate instead and the pilot is not paid for at all; `initial=` on
the window overrides both. The pilot runs once per calculation, not once per round.

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
