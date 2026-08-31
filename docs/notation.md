# Notation and conventions

The conventions every equation in this documentation assumes, stated once. They are the
program's own conventions — fixed in the code and, where they are part of an interface,
asserted by its tests — so an equation written in them describes what the code computes, not
a textbook's version of it. Pages elsewhere in `docs/` do not restate these; where a page
needs a convention beyond this file, it defines it locally and says so.

## Units and physical constants

Kuiva computes in **Hartree atomic units** throughout. The user-facing surfaces convert:

| quantity | unit | notes |
|---|---|---|
| geometry input | Angstrom | the `Molecule` default (`unit="Bohr"` to override) |
| total and relative energies | hartree (Eh) | printed at 1e-8 Eh, matching the meaningful tolerance |
| spectroscopic gaps and splittings | cm⁻¹ | 1 Eh = 219 474.631 363 2 cm⁻¹ |
| magnetic moments | Bohr magneton (μ_B) | printed at 1e-5 μ_B |
| electric dipoles | atomic units (e·a₀) | |

⚠ **One deliberate exception to "input is Angstrom": a bare `(x, y, z)` tuple given as a
gauge origin is in bohr.** That is the historical meaning and it is kept so that no stored
number silently moves; the bare form warns, and the unambiguous spellings
`("angstrom", x, y, z)` and `("bohr", x, y, z)` say what they mean.

**The speed of light.** Two values coexist, deliberately, and confusing them shows up as a
numerical discrepancy with no physical cause:

- Every calculation takes $`c`$ from whatever produced its integrals. For the PySCF front end
  that is PySCF's own $`c = 137.03599967994`$ a.u.
- The CODATA 2018 value $`c = 1/\alpha = 137.035999084`$ [[184]](references.md#r184) is used
  for **reporting only**, never in a Hamiltonian built on PySCF integrals.

The relative difference is 4.3e-9 — far below any physical tolerance and far above the
1e-13-level agreement at which the X2C machinery is validated against its inputs.

## Orbitals, spinors, and index letters

Kuiva distinguishes **scalar (spatial) orbitals** — real, spin-free, the product of the
scalar-relativistic SCF — from **spinors**, the complex two-component one-particle functions
the multireference stages work in. Unless a page says otherwise:

| letters | run over |
|---|---|
| $`\mu, \nu, \lambda, \sigma`$ | atomic-orbital (AO) basis functions |
| $`p, q, r, s`$ | general spinors (or general scalar MOs, where the context is scalar) |
| $`i, j`$ | inactive (doubly occupied core) orbitals |
| $`t, u, v, w`$ | active orbitals |
| $`a, b`$ | virtual (secondary) orbitals |
| $`P, Q`$ | auxiliary functions: Cholesky vectors or density-fitting functions |
| $`I, J, K`$ | determinants, or many-electron states |

Every orbital *space* (inactive, active, virtual) is defined on **spatial** orbitals; the
spinor spaces follow from them by the pairing below, so a Kramers pair is never split across
a space boundary.

## The spinor basis

These conventions are part of the program's interface — CI addressing depends on them
[[20]](references.md#r20)[[57]](references.md#r57)[[58]](references.md#r58)[[59]](references.md#r59).

**Rows are spin-blocked.** A spinor coefficient array has shape
$`(2 n_{\mathrm{bas}}, n_{\mathrm{spinor}})`$ over one underlying real scalar basis (the AO
basis, or the orthonormal working basis):

```math
C = \begin{pmatrix} C^\alpha \\ C^\beta \end{pmatrix},
\qquad C^\alpha, C^\beta \in \mathbb{C}^{\,n_{\mathrm{bas}} \times n_{\mathrm{spinor}}}.
```

**Columns are interleaved Kramers pairs.** Spinor $`2p`$ is the *unbarred* partner built from
scalar orbital $`\phi_p`$, and spinor $`2p+1`$ its *barred* (time-reversed) partner
$`\overline{p} = \hat{T}(2p)`$. A scalar space $`[m, n)`$ therefore maps to the contiguous
spinor range $`[2m, 2n)`$.

**Time reversal** is

```math
\hat{T} = -\,i\,\sigma_y K, \qquad
\hat{T}\,(C^\alpha, C^\beta) = \left(-\,\overline{C^\beta},\; \overline{C^\alpha}\right),
```

where $`K`$ conjugates the *coefficients only* — which presumes the scalar basis functions are
real, true for the real solid-harmonic Gaussians used throughout. On any single spinor
$`\hat{T}^2 = -1`$, the origin of Kramers degeneracy in odd-electron systems
[[55]](references.md#r55)[[56]](references.md#r56). The scalar-to-spinor guess is the
trivial pairing $`\psi_{2p} = (\phi_p, 0)`$, $`\psi_{2p+1} = (0, \phi_p)`$; spin–orbit coupling
enters when the two-component wavefunction is built and optimized, not in the guess.

Storage is `complex128`, C-contiguous; complex arithmetic is first-class everywhere in the
multireference layer.

## Second quantization and determinants

A determinant is an ordered product with **ascending** spinor index,

```math
|I\rangle = a^\dagger_{k_1} a^\dagger_{k_2} \cdots a^\dagger_{k_N} |\mathrm{vac}\rangle,
\qquad k_1 \lt k_2 \lt \cdots \lt k_N,
```

so that both $`a_p|I\rangle`$ and $`a^\dagger_p|I\rangle`$ carry the sign
$`(-1)^{\#\{\text{occupied } k \lt p\}}`$ — the standard Slater–Condon phase convention
[[85]](references.md#r85)[[86]](references.md#r86)[[83]](references.md#r83). Spinors are the
elementary fermionic modes: there is no $`\alpha/\beta`$ factorization, because spin–orbit
coupling breaks spin symmetry.

The Hamiltonian in the spinor-MO basis is

```math
\hat{H} = \sum_{pq} h_{pq}\, a^\dagger_p a_q
\;+\; \tfrac{1}{2} \sum_{pqrs} (pq|rs)\; a^\dagger_p a^\dagger_r a_s a_q ,
```

with $`(pq|rs)`$ in **chemists' (charge-cloud) notation**,

```math
(pq|rs) = \iint \psi^*_p(1)\,\psi_q(1)\; r_{12}^{-1}\; \psi^*_r(2)\,\psi_s(2)\; d1\, d2 .
```

⚠ **The two-electron integrals have only 4-fold permutational symmetry**, not the 8-fold of
a real spin-free code: with complex spinors the surviving relations are

```math
(pq|rs) = (rs|pq), \qquad (pq|rs)^* = (qp|sr),
```

and nothing else. Formulas transcribed from spin-free literature that assume 8-fold symmetry
are wrong here; the method pages point this out wherever it bites.

## Density matrices

The one- and two-particle reduced density matrices of a state (or state average)
$`|\Psi\rangle`$ are

```math
\gamma_{pq} = \langle \Psi | a^\dagger_p a_q | \Psi \rangle, \qquad
\Gamma_{pqrs} = \langle \Psi | a^\dagger_p a^\dagger_r a_s a_q | \Psi \rangle ,
```

paired with the integral convention so that the energy is

```math
E = \sum_{pq} h_{pq}\, \gamma_{pq}
\;+\; \tfrac{1}{2} \sum_{pqrs} (pq|rs)\, \Gamma_{pqrs}
\;+\; E_{\mathrm{nuc}} .
```

The 2-RDM obeys $`\Gamma_{pqrs} = \Gamma_{rspq}`$, $`\Gamma_{pqrs}^* = \Gamma_{qpsr}`$,
$`\Gamma_{pqrs} = -\Gamma_{psrq}`$, and the trace condition
$`\sum_r \Gamma_{pqrr} = (N-1)\,\gamma_{pq}`$.

⚠ **Coefficients and density matrices transform oppositely.** Under an orbital rotation
$`|p'\rangle = \sum_p |p\rangle\, U_{pp'}`$ (coefficients $`C \to C U`$), the 1-RDM transforms as

```math
\gamma \;\to\; U^{\mathsf T}\, \gamma\, U^{*},
```

*not* as $`U^\dagger \gamma U`$ (which is how an operator matrix such as $`h`$ transforms). The
two are indistinguishable for real quantities and produce plausible, wrong complex results —
occupations stay bounded, traces stay right. Anyone post-processing Kuiva's stored matrices
should check a transformation through a known spectrum, not through the algebra alone.

## Degenerate blocks and arbitrary phases

Two facts shape how Kuiva reports nearly everything, and how its output can — and cannot —
be compared with other programs':

1. **Every eigenvector carries an arbitrary phase**, and inside a degenerate manifold the
   eigensolver may return *any* unitary mixture of the members. Kuiva fixes no phase
   convention in its stored operator matrices; a consumer fixes phases itself.
2. **Consequently, single states inside a degenerate block have no individual identity.**
   Populations, spin expectation values, irrep labels and per-state properties are reported
   **per degenerate block** — block traces are invariant where single-state values are
   basis-dependent noise.

Cross-code comparison of operator matrices therefore goes through **phase-invariant
reductions** only: degeneracy patterns, relative energies, the block invariants
$`\mathrm{Tr}_{\mathrm{block}}(\mu_i \mu_j)`$ with their principal g values
[[179]](references.md#r179), and, for electric dipoles,
$`\mathrm{Tr}_{\mathrm{block}}(d_i d_j)`$ and block-to-block line strengths
$`\sum |d_{IJ}|^2`$. Element-by-element comparison of
moment matrices between programs is meaningless, and no page in this documentation asks for
it.

A related convention used throughout: a **degenerate group is truncated, kept, or dropped
whole** — in basis-set truncation, integral factorization, tensor-network truncation and
perturbation-theory cutoffs alike — because cutting through an exact degeneracy produces
Hermitian, plausible, wrong results. Method pages state where this rule acts in each
context.

## Miscellaneous

- **"Active space" is a physical statement.** Throughout these pages an active space is
  specified by orbital character and electron count ("the seven lowest Kramers pairs of f
  character on the dysprosium, nine electrons"), never by orbital index windows, because
  only the former is reproducible in another program. The reference pages describe the
  concrete selection surfaces.
- **Kramers pair counting**: sizes of spinor spaces are always even, and quoted either as
  spinors ($`n_{\mathrm{active}} = 14`$) or as Kramers pairs (7 pairs); the pages say which.
- $`g_e = 2.002\,319\,304\,362\,56`$ (CODATA 2018 [[184]](references.md#r184)) wherever the
  free-electron g factor enters an operator or a Landé formula.
