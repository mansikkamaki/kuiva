# The CASSCF orbital optimization

One shared, state-averaged orbital optimizer, consumed identically by the cheap CI, the
conventional-CI CASSCF and the DMRG-CASSCF: **reduced density matrices in, orbital rotation
out**, with the CI method entering only as a callback returning
$`(E, \gamma, \Gamma)`$ at the current integrals. Nothing in the optimizer knows whether the
RDMs came from a determinant list, a full CI or a tensor network — that contract is what
lets every solver plug in unchanged, and it is never bypassed. Options are on
[`CASSCF`](../reference/stages/CASSCF.md); the state-average design questions are in
[workflows](../guide/workflows.md#designing-a-state-average).

## The energy and its gradient

With inactive spinors $`i`$, active $`t,u,v,w`$, virtual $`a`$
[[93]](../references.md#r93)[[83]](../references.md#r83):

```math
F^{I}_{pq} = h_{pq} + \sum_i \left[ (pq|ii) - (pi|iq) \right],
\qquad
F^{A}_{pq} = \sum_{tu} \gamma_{tu} \left[ (pq|tu) - (pu|tq) \right],
```

```math
E = E_{\mathrm{nuc}} + \tfrac12 \sum_i \left( h_{ii} + F^{I}_{ii} \right)
  + \sum_{tu} F^{I}_{tu}\, \gamma_{tu}
  + \tfrac12 \sum_{tuvw} (tu|vw)\, \Gamma_{tuvw}.
```

Orbitals rotate as $`C \to C\, e^{\kappa}`$ with $`\kappa`$ **anti-Hermitian and complex
throughout** [[106]](../references.md#r106)[[107]](../references.md#r107): the imaginary
part is not decoration — it is what lets the optimizer mix Kramers partners, i.e. what
makes the wavefunction respond to spin–orbit coupling at all. To first order
$`dE = 2\,\mathrm{Re} \sum_{p\gt q} G_{pq} \kappa_{pq}`$ with the anti-Hermitian gradient
$`G = F^\dagger - F`$ built from the generalized Fock matrix

```math
F_{pq} = \sum_r D_{pr}\, h_{qr} + \sum_{rst} \Gamma_{prst}\, (qr|st),
```

which specializes to

```math
F_{iq} = (F^{I} + F^{A})_{qi}, \qquad
F_{tq} = \sum_u \gamma_{tu} F^{I}_{qu} + \sum_{uvw} \Gamma_{tuvw}\, (qu|vw), \qquad
F_{aq} = 0 .
```

⚠ **The gradient is verified redundantly** — against finite differences *and* against the
general formula evaluated from explicit total $`D`$ and $`\Gamma`$ — because the CASSCF fixed
point is *defined* by its vanishing: an error there yields a confidently converged wrong
answer, with no residual anywhere to notice. Redundant rotations (inactive–inactive,
virtual–virtual) are never parameters; active–active rotations are redundant for a full CI
and are opt-in, off by default.

**The Fock matrices are built in the AO basis** from factorized densities, and nothing of
size $`n_{\mathrm{param}}^2`$ is ever formed: the only three-index quantity transformed to MO
is $`B^P_{p,t}`$ with one active index ([integrals](integrals.md#the-spinor-mo-transformation))
— gigabytes against megabytes, every macro-iteration. This is the single most important
performance decision in the module.

## The step, and the three ways to take it

All step types share the exact gradient, the trust region [[102]](../references.md#r102)
and the accept/reject logic; they differ only in the curvature model:

- **`"quasi-newton"`** — self-scaled L-BFGS
  [[104]](../references.md#r104)[[105]](../references.md#r105) preconditioned by an
  approximate diagonal Hessian (a *preconditioner*, not a curvature claim: floored,
  anisotropy-clamped, and a persistently negative diagonal is reported rather than damped).
  Measured 3–4× less total work than the second-order step on problems that converge.
- **`"second-order"`** — the exact orbital Hessian through an augmented-Hessian
  eigenproblem
  [[96]](../references.md#r96)[[97]](../references.md#r97)[[98]](../references.md#r98),
  with Hessian-vector products at gradient cost via one-index-transformed Fock matrices
  [[99]](../references.md#r99) and inexact-Newton forcing
  [[100]](../references.md#r100)[[101]](../references.md#r101). Converges cases the
  quasi-Newton step does not converge at all — an answer versus none, not a speed
  difference.
- **`"auto"`** (default) — start cheap, escalate when the **gradient trajectory** shows the
  cheap step is going nowhere. The robust choice, not the cheapest; ⚠ it is the *orbital
  problem* that decides the mode, never the CI cost — a heavy element, a large state
  average or a DMRG solver wants `"second-order"` explicitly, and that is a caller decision
  deliberately not inferred.

⚠ **The scheme is two-step, and the wall is real**: the orbital–CI coupling is not in the
Hessian, so the asymptotic rate is linear rather than quadratic — demonstrated cleanly by
freezing the RDMs (the same optimizer then converges to $`|g| = 10^{-8}`$) versus re-solving
the CI each macro-iteration (it settles at $`\sim 6 \times 10^{-4}`$; that gap *is* the
neglected coupling). Including the coupling would break the callback contract that lets a
truncated CI, a full CI and a DMRG plug in identically, and is deliberately not done.
Convergence therefore requires **both** $`|g|`$ below `conv_grad` and a converged energy,
because a truncated CI can stall the energy while the orbitals still move.

## The state average

State averaging is **imposed where the RDMs are built** ([ci](ci.md#the-sigma-vector)),
because nothing downstream can recover it: weights equalized inside every degenerate block,
a count that splits a block refused, and the *same* equalized weights producing the
reported state-averaged energy. Around that gate sit the diagnostics, because the gate
alone cannot see a near-degenerate manifold cut in half:

- **The boundary-gap diagnostic** solves a few roots the average does *not* use and reports
  the gap between the averaged set and the first discarded root — at the **starting**
  orbitals (what says the trajectory was safe) and at the **converged** ones — warning
  below 50 cm⁻¹. The threshold is a statement that the boundary is *unambiguous*, not a
  physical tolerance. It is advisory and never kills a run: a failure to measure is a
  warning and **no** report, a weaker statement than a clean one.
- **The spin non-invariance report** states whether the averaged density is an ensemble the
  symmetry leaves invariant at all — near zero for every term-complete average (safe by
  construction), large for a single-J or single-doublet average, which *leans* on the
  spin–orbit structure and is protected only by its gap and its starting orbitals. The
  measured stability map, and the wrong-basin case no diagnostic can see from inside a run,
  are in [workflows](../guide/workflows.md#designing-a-state-average).
- A state count is a property of **one spectrum**: a count bounding the $`2S{+}1`$ terms of a
  spin-free run generally cuts the $`2J{+}1`$ multiplets of the same system with coupling on,
  and the converse equally. The two counts are carried separately.

Per-irrep selection and the symmetry-preserving rotation mask change what is *counted*, not
what the gate does ([symmetry](symmetry.md)); a degenerate block spanning two sectors is
one block to the gate — which it must be, or a per-irrep request would be a way to walk
past the refusal.

## Adaptive solvers: the optimizer owns the space

A solver that re-chooses its internal space per set of integrals — a selected CI
re-selecting determinants, a DMRG re-distributing bond dimensions — makes $`E(\kappa)`$ not
one surface but a **family** indexed by that space, and every mechanism above then
misfires: the quadratic model is trusted across a hop, accept/reject reads a hop as a
failed step, curvature memory mixes surfaces. The remedy is a four-method protocol —
`solve` on the incumbent space (which must be smooth and deterministic; the promise
everything rests on), `propose` a fresh space (evaluated, **not** adopted), `adopt`,
`space_key` — driven by an **event-gated sibling driver**, not a mode of the smooth one:

- **adoption is variational, at fixed integrals**, keeping the trajectory monotone across
  space changes and making the method a descent method again;
- **curvature memory is chart-scoped** — an adoption clears the L-BFGS pairs rather than
  transporting them across surfaces (and the same scoping holds across a process boundary:
  a restart whose `space_key` mismatches clears, never restores);
- **convergence means $`|g|`$ below `conv_grad` *and* a proposal that refuses** — a
  gradient norm alone is a statement about one chart;
- ⚠ event gating recovers *frozen-chart* quality on an adaptive surface; it does not exceed
  the two-step wall.

The full CI satisfies the protocol trivially (`propose` returns nothing), so the plain
smooth-surface driver runs it; the DMRG solver is the real adaptive client
([dmrg](dmrg.md)). Checkpointing, deadlines and restart semantics are shared machinery:
[clusters](../guide/clusters.md).
