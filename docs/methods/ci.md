# The conventional CI

Complex determinant full CI in the spinor basis — the reference solver of the program, and
the path every committed reference number was produced on. Spin–orbit coupling breaks spin
symmetry, so there is no α/β factorization anywhere: spinors are the elementary fermionic
modes, a determinant is a single occupation string, and everything is `complex128`. Options
are on [`CASSCF`](../reference/stages/CASSCF.md#solverci-the-conventional-ci) and
[`CASCI`](../reference/stages/CASCI.md); conventions (determinant ordering, phases,
integral symmetry, RDMs) are fixed in [notation](../notation.md).

## Representation and addressing

A determinant is a **bitmask** — bit $`k`$ set means spinor $`k`$ occupied — with excitation
analysis by XOR/popcount, as in modern selected-CI codes
[[87]](../references.md#r87)[[88]](../references.md#r88). Masks are single 64-bit words,
capping the active space at **64 spinors** (four d shells or four f shells — comfortably
past where this algorithm is right at all). Two addressing modes exist: an explicit list
with pairwise connection search for the selected CI of the pre-optimizer, and, for the
complete CAS space the full CI runs on, **lexicographic combinatorial rank**
[[188]](../references.md#r188)[[83]](../references.md#r83) — table-driven integer
arithmetic, never a hash table (hundreds of MB at $`10^6`$ determinants, slow, and it would
put the address map where a compiled backend cannot reach it).

The dense active-space Hamiltonian (`kuiva.ci.strings.hamiltonian_matrix`, by Slater–Condon
rules [[85]](../references.md#r85)[[86]](../references.md#r86)) is kept as a genuinely
independent implementation — different algorithm, different sign bookkeeping — and is what
the production sigma vector is validated against, as well as the definitive check whenever
a spectrum surprises you.

## The sigma vector

With $`E_{pq} = a_p^\dagger a_q`$ (spinor, not spin-summed) and
$`a_p^\dagger a_r^\dagger a_s a_q = E_{pq}E_{rs} - \delta_{qr}E_{ps}`$, the Hamiltonian takes
the one-particle-excitation form

```math
\hat H = \sum_{pq} \tilde h_{pq}\, E_{pq}
   + \tfrac12 \sum_{pqrs} (pq|rs)\, E_{pq} E_{rs},
\qquad
\tilde h_{pq} = h_{pq} - \tfrac12 \sum_r (pr|rq),
```

and the sigma vector is the two-step $`E_{pq}`$ resolution
[[185]](../references.md#r185)[[186]](../references.md#r186)[[187]](../references.md#r187)[[183]](../references.md#r183):

```math
\begin{aligned}
\text{1 (gather):}\quad & F[K, pq] = \sum_J \langle K|E_{pq}|J\rangle\, c_J
\\ \text{2 (GEMM):}\quad & G[K, pq] = \sum_{rs} (pq|rs)\, F[K, rs]
\\ \text{3 (gather):}\quad & \sigma_I = \sum_{pq} \tilde h_{pq} F[I, pq]
   + \tfrac12 \sum_{K,pq} \langle I|E_{pq}|K\rangle\, G[K, pq].
\end{aligned}
```

Step 2 is one dense complex GEMM of shape $`(N \times n^2)\cdot(n^2 \times n^2)`$ and holds
the great majority of the arithmetic; the $`n^2 \times n^2`$ integral matrix respects the
4-fold symmetry only, asserted on construction. ⚠ Two quiet traps are fixed by convention
and tested against the independent dense implementation: the $`\tilde h`$ folding (an error
there survives hermiticity, Kramers degeneracy and the RDM trace conditions, surfacing
later as an energy offset that looks like a basis effect), and the index sense of $`F`$
($`p`$ annihilated, $`q`$ created — a transposition leaves a Hermitian, plausible, wrong
operator).

**Residency, not flops, sets the ceiling**: the step-3 gather needs all of $`G`$ —
$`\binom{n}{k}\, n^2`$ complex numbers ($`F`$ needs no second buffer; $`G`$ overwrites it in row
tiles) — which with the Davidson stacks gives the practical bound of 20–22 half-filled
spinors at an 8 GB limit, enforced by exact sizing before the first allocation. It is a
bound on the determinant count, so dilute or nearly-full spaces run well past it; beyond
it, the tensor network takes over ([dmrg](dmrg.md)).

**One excitation map, one intermediate, three consumers.** $`F`$ is also how the density
matrices and the transition densities are built [[183]](../references.md#r183): with
$`E_{pq}^\dagger = E_{qp}`$,

```math
\langle c | E_{pq} E_{rs} | c \rangle = \sum_K \overline{F[K, qp]}\; F[K, rs],
\qquad
\Gamma_{pqrs} = \langle E_{pq}E_{rs}\rangle - \delta_{qr}\gamma_{ps},
```

so the 2-RDM is a single $`F^{\mathsf H}F`$ GEMM over the intermediate the sigma vector
already owns — one kernel contract to get right instead of three, and no second residency.
⚠ The $`\overline{F[K,qp]}`$ transposition is the conjugation trap in its RDM guise (a wrong
2-RDM that is still Hermitian, positive, correctly traced), caught by the trace condition
$`\sum_r \Gamma_{pqrr} = (N-1)\gamma_{pq}`$ and the energy-closure check against the Davidson
eigenvalue.

**State averaging is imposed where the RDMs are built**, because nothing downstream can
recover it: weights are equalized inside every degenerate block (inside a manifold the
individual roots are defined only up to a rotation, so unequal weights would make the
converged orbitals depend on the eigensolver's arbitrary choice), and a state count that
splits a block is refused — rigorously for odd electron counts, where Kramers' theorem
[[55]](../references.md#r55) makes it a theorem; empirically for even ones
([casscf](casscf.md#the-state-average) carries the diagnostics this cannot cover).

## The eigensolver

A **block** complex-Hermitian Davidson [[192]](../references.md#r192) with simultaneous
expansion of all roots [[193]](../references.md#r193)[[194]](../references.md#r194),
re-orthogonalized Gram–Schmidt [[195]](../references.md#r195), and a dense `eigh` fallback
for small spaces. Block by necessity, not preference: with an odd electron count every root
is at least doubly degenerate, and a solver expanding roots one at a time cannot separate a
Kramers cluster — an ARPACK failure on exactly that case is this solver's specification.
The subspace floor is $`2 n_{\mathrm{roots}} + \mathrm{margin}`$, warm-started across
macro-iterations.

⚠ **The convergence tolerance is set by the RDMs, not the energy.** The Ritz value error is
second order in the residual while the **density matrices are first order** — and the
densities are what the orbital gradient is built from. Measured: at residual 1e-6 the
energy is exact to 1e-14 while the 1-RDM error (3e-8) sits at the optimizer's gradient
tolerance; the default 1e-8 costs ~25% more applications of $`\hat H`$ and buys two and a
half orders of RDM accuracy.

⚠ **"Converged" is not "lowest", and the guess decides which.** A Krylov method never
leaves the invariant subspaces its starting vectors lie in; wherever a conserved operator
is diagonal in the determinant basis (spin-free runs, symmetry sectors), a biased
lowest-diagonal guess can miss whole sectors, and the solver then converges every residual
and returns states that are *not* the lowest — in fewer iterations. Generic vectors are
prepended to every cold start, making that failure structurally unavailable; a spectrum
that changes qualitatively with the root count is this effect, not physics
([workflows](../guide/workflows.md#turning-spin-orbit-coupling-off)). With symmetry labels
on, each sector is solved on its own — the structural form of the same fix
([symmetry](symmetry.md)).

## The Kramers-restricted mode

The default path is **general complex (Kramers-unrestricted), forever**: Kramers degeneracy
emerges numerically there, in the 1e-8…1e-6 Eh band reserved for genuine numerical
splittings, and every committed reference is one of its results.
`solver_options=dict(kramers="restricted")` selects the time-reversal-adapted mode
[[58]](../references.md#r58)[[59]](../references.md#r59): the *eigensolver's* expansion
subspace is kept closed under $`\hat T`$, so one stored direction spans a whole Kramers pair
— the quaternionic structure of the subspace problem [[60]](../references.md#r60) halving
the work. Kept **for the factor of two, not for the degeneracy**: measured 1.8–1.9× less
CPU from three averaged Kramers pairs up, and ~8% *slower* at two pairs.

The restriction lives in the eigensolver only — the sigma vector, the excitation map and
every kernel are untouched, so the memory ceiling does not move — the states come back
**pair-expanded in the general convention**, so nothing downstream learns anything new, and
it requires time-reversal-symmetric active integrals, a property of the *orbitals* that is
**checked at every solve, never assumed**. Odd electron counts only: for even $`N`$,
$`\hat T^2 = +1`$ makes the Hamiltonian real rather than the subspace halvable — a different
theorem, and a recorded descope rather than an omission.
