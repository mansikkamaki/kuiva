# SC-NEVPT2

Strongly contracted n-electron valence state perturbation theory
[[143]](../references.md#r143)[[144]](../references.md#r144) on a converged two-component
CASCI/CASSCF: the second-order dynamic-correlation correction, per state, decomposed by
excitation class — a pure **post-processing** stage that consumes converged orbitals, CI
vectors and integral factors and changes no wavefunction. ⚠ **Every published working
equation of SC-NEVPT2 is spin-free and real
[[146]](../references.md#r146); the equations Kuiva implements were re-derived in spinor
second quantization** for a complex two-component Hamiltonian: no spin-traced excitation
operators, the antisymmetrized integral $`(ai|bj) - (aj|bi)`$ appearing directly rather than
as a $`2J - K`$ combination, and 4-fold integral symmetry only
([notation](../notation.md#second-quantization-and-determinants)) — the relation
$`(pq|rs) = (rq|ps)`$ is false here and nothing may rely on it. Options:
[`NEVPT2`](../reference/stages/NEVPT2.md).

## The zeroth-order Hamiltonian

Dyall's [[145]](../references.md#r145), in the spinor basis:

```math
\hat H_D = C + \sum_i \varepsilon_i\, a_i^\dagger a_i
             + \sum_a \varepsilon_a\, a_a^\dagger a_a + \hat H_{\mathrm{act}},
\qquad
\hat H_{\mathrm{act}} = \sum_{tu} f^{I}_{tu}\, a_t^\dagger a_u
   + \tfrac12 \sum_{tuvw} (tu|vw)\, a_t^\dagger a_v^\dagger a_w a_u ,
```

with $`i`$ inactive, $`a`$ virtual, $`t\ldots w`$ active. $`\hat H_{\mathrm{act}}`$ is the
**exact** active-space Hamiltonian — the full two-component one, complex, spin–orbit
coupling and the mean-field screening already inside $`f^I`$ (any Hermitian operator is a
legal $`\hat H_0`$, so the mean field inside it is a choice of partitioning, not a double
counting; no reported total contains the mean-field double-counting term either way,
[soc](soc.md#x2camf-the-two-electron-picture-change)). $`C`$ is fixed by
$`\hat H_D|\Psi_0\rangle = E_0|\Psi_0\rangle`$, making every denominator an energy difference
from the reference. The $`\varepsilon`$ are eigenvalues of the state-averaged generalized
Fock over the inactive and virtual blocks (pseudo-canonicalization); ⚠ **the active block
is never rotated** — the CI vectors are expressed in the active basis as converged, and the
active Hamiltonian, core energy and CI eigenvalues are asserted invariant under the
canonicalization.

⚠ **The Fock is built from the block-equalized state-averaged density, by default and for
production always.** A single state's density is not time-reversal even, so a
state-specific Fock breaks the symmetry *inside $`\hat H_0`$ itself* and splits a degenerate
manifold's $`E_2`$ artificially — measured at 0.7 to 340 cm⁻¹ against the averaged Fock's
1e-6…1e-4 cm⁻¹, five to six orders, and far above the 0.1 cm⁻¹ at which a splitting already
implies different physics. `fock="state-specific"` exists to *measure* that mechanism,
warns, and is never production. (What it destroys is a manifold **larger than a Kramers
pair**: a doublet's members are time reverses of each other whatever basis the eigensolver
returned, so their Fock matrices, $`\varepsilon`$ and class energies are identical —
measured.) Per-state RDMs still drive each state's norms and Koopmans matrices, which is
the state-specific content of SC-NEVPT2.

## Perturbers, norms, and the group-complete contraction

The eight excitation classes are a **partition** of the first-order interacting space —
all eight are always registered, and a sum that is not over all eight is reported as
partial, never silently as $`E_2`$. For class $`k`$ and one set of external labels $`l`$, the
strongly contracted perturber is the single function
$`|\Psi_l\rangle = \hat P_l \hat H |\Psi_0\rangle`$, and

```math
N_l = \langle \Psi_0 | \hat H \hat P_l \hat H | \Psi_0 \rangle, \qquad
\Delta E_l = \frac{\langle \Psi_l | \hat H_D | \Psi_l \rangle}{N_l} - E_0, \qquad
E^{(k)}_2 = -\sum_l \frac{N_l}{\Delta E_l} .
```

$`N_l`$ and $`\Delta E_l`$ are scale-invariant, convention-free numbers any correct
implementation must reproduce — which is what makes the per-class comparison against an
independent program a real check.

⚠ **The contraction is over whole degenerate-$`\varepsilon`$ groups, never single spinors,
and this is a correctness requirement, not a knob.** Strong contraction fixes one perturber
per external *label*, so the label resolution is part of the method — and a label finer
than the arbitrariness in the orbitals is not a label at all: pseudo-canonicalization
leaves an arbitrary unitary inside every degenerate $`\varepsilon`$ block (Kramers pairs
among them), so a per-spinor contraction makes $`E_2`$ depend on a choice the eigensolver
made. Within a group the perturbers transform among themselves unitarily, so the group's
accumulated quantities are traces, which no unitary can move:

```math
N_G = \sum_{l \in G} N_l, \qquad
D_G = \sum_{l \in G} N_l\, \Delta E_l, \qquad
E^{(G)}_2 = -\,\frac{N_G^2}{D_G} .
```

Lumping restores exact invariance as a theorem. On a real system it is a no-op for every
class whose degeneracy comes from a symmetry the integrals share — which is why the rule is
easy to miss — and load-bearing for the one class whose same-spin and spin-flip perturbers
are not related by any symmetry. ⚠ **In the scalar limit the implementation reproduces the
published spin-free SC-NEVPT2 class by class** (measured to 1e-14 relative against PySCF on
every class), even though Kuiva's grouping is the coarser partition wherever two spatial
orbitals are symmetry-degenerate; with spin–orbit coupling on, the groups are Kramers pairs
— the same rule with the only symmetry that survives. Every other threshold in the module
drops whole degenerate groups too.

## Degenerate manifolds: reporting, not repair

⚠ **Nothing makes the per-state $`E_2`$ of a degenerate manifold's members well defined, and
that is a property of the method, not of this code**: the correction is state-specific
through the per-state RDMs, and the CI basis inside a degenerate manifold is arbitrary, so
the members carry an arbitrary share of the manifold's internal spread (measured at
1e-6…1e-4 cm⁻¹ on stationary references; distinct from the degenerate-$`\varepsilon`$
freedom, which the group-complete contraction *does* remove exactly). The barycentre is
four to seven orders more stable, and the treatment is reporting: barycentres **beside**
the per-state energies, member spread visible, never instead of them.

## What is never stored, and the intruder handling

No 3- or 4-RDM is ever formed: the stored 4-RDM is $`n_{\mathrm{act}}^8`$ (impossible at
multi-site sizes), cumulant approximations are rejected as known intruder generators, and —
one step past the matrix-free precedents [[147]](../references.md#r147) — the integrals are
contracted into **one perturber vector per external label**, so no rank-3 or rank-4 object
exists at all.

Intruder states are handled by real [[148]](../references.md#r148) and imaginary
[[149]](../references.md#r149) level shifts, **parameter-free by default**, warning when
applied; the per-class smallest denominator is printed and warned on (below 0.1 Eh:
leaning; below 1e-6 Eh or negative: divergent or sign-wrong). A shift bounds the damage —
the fix is the reference. Frozen core and deleted virtuals are off by default and stated as
**orbital energies** on the pseudo-canonical spectrum, never as counts.

## The network route

A `solver="dmrg"` reference reaches the same driver through a second contraction provider
(the seam is primitives, never SC-assembled quantities — which is also what keeps
FIC-NEVPT2 implementable later without touching the driver): ranks 1–2 per state from the
network's operator environments, the rank-3 Gram kernels as Gram matrices of applied
Jordan–Wigner strings with $`\hat H_{\mathrm{act}} - E`$ between
([dmrg](dmrg.md#densities-and-the-local-multiplet-model)), following the DMRG-NEVPT2
precedent [[139]](../references.md#r139) but via applied-string Grams instead of stored
higher densities — a recorded departure. ⚠ **The two primed single-external classes are not
served**: their per-label perturber vectors have no cheap network form, and every dense
workaround dies at exactly the sizes DMRG exists for. Their scalable route (a per-label
perturber TTNO, applied and compressed) is open work; until it lands, a network-reference
$`E_2`$ is **PARTIAL** — six of eight classes, warned per skipped class, marked incomplete —
and is not comparable with a complete NEVPT2. At a saturating bond dimension the provider
reproduces the CI provider to rounding, which is how it is validated.

## Not implemented, on a measurement

FIC-NEVPT2 and the quasi-degenerate variants
[[150]](../references.md#r150)[[151]](../references.md#r151)[[152]](../references.md#r152)
— including the spin–orbit QD-NEVPT2 that is the closest prior art
[[153]](../references.md#r153) — are not planned: the artificial multiplet splitting they
would cure was measured four orders of magnitude below the 0.1 cm⁻¹ at which a splitting
means different physics.
