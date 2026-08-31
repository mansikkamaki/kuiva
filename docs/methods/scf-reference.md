# The scalar SCF reference

Every Kuiva calculation starts from a **scalar-relativistic (spin-free) X2C SCF**
[[10]](../references.md#r10), run through PySCF [[1]](../references.md#r1) as a thin front
end. The orbitals are real and comparatively easy to converge; spin–orbit coupling is
deliberately absent from them and enters with the two-component wavefunction
([soc](soc.md)). This page covers what the reference *is* and the constructions around it;
the option surface is on [`ScalarSCF`](../reference/stages/ScalarSCF.md).

## Two operators, one calculation

The SCF and the correlated Hamiltonian use **different one-electron operators, on
purpose**: the SCF stays on the spin-free X2C operator, while the multireference step uses
the full two-component one ([x2c](x2c.md)). The scalar orbitals are a *guess* — a basis the
CASSCF re-optimizes — so the operator whose expectation value is the energy is one
well-defined Hamiltonian, and the difference between the two (a picture-change effect) is
reported rather than hidden. The consequence for the user: an SCF that fails to converge
**refuses** to hand its orbitals on — "the CASSCF re-optimizes them" is a hope, not a
property — and the stability analysis exists because a *converged* SCF can still be a
saddle point with a lower solution one rotation away.

## Restricted and unrestricted references

`reference="auto"` picks RHF for a closed shell and ROHF otherwise; UHF is explicit, and
RHF on an open shell is refused rather than silently promoted. Two structural facts bind
everything downstream:

- The orbitals are consumed as one set (restricted) or two (unrestricted) — never by
  guessing from array shapes, which is how a `(2, nao, nmo)` array eventually gets treated
  as a different basis.
- ⚠ **An unrestricted spinor set is orthonormal but *not* Kramers paired**: spinors $`2p`$ and
  $`2p+1`$ are then the $`p`$-th α and $`p`$-th β orbital, which need not describe the same
  spatial function. An active space on a UHF reference therefore may not be chosen as a
  contiguous spinor range — select by orbital character per spin set.

## Broken symmetry: the antiferromagnetic starting density

⚠ **A UHF started from a closed-shell density stays closed-shell**: the symmetric point is a
stationary point of the energy, so nothing in the iteration pushes off it — you ask for UHF
on a coupled pair of metals, get the restricted answer, and nothing in the output says so.
The polarization must be put in by the starting density. Kuiva implements the
broken-symmetry construction of Noodleman
[[196]](../references.md#r196)[[197]](../references.md#r197), in the practical form:

1. converge the **high-spin** state — the easy, unambiguous one;
2. **localize its singly occupied orbitals** onto the named centres with the SPADE fragment
   partition ([active-spaces](active-spaces.md#fragment-localization-which-centre)) — for
   two equivalent metals this is what makes the flip expressible at all, the canonical
   orbitals being the symmetric and antisymmetric combinations, half on each metal (an
   atom-blocked density would be a second, worse definition of "this centre");
3. flip the orbitals assigned negative signs into the β set and start the SCF from that
   density.

The converged solution is **spin-contaminated on purpose**, and the two diagnostics are the
point rather than caveats:

- $`\langle S^2 \rangle`$ **between the low-spin and high-spin values** is what says the
  determinant really is broken-symmetry — coming back at the low-spin value means the
  polarization did not survive, and Kuiva warns rather than passing it off as a converged
  UHF;
- **the spin populations must carry the assigned signs**, which is the check no energy can
  make: a solution with the centres swapped has the same energy and the same
  $`\langle S^2 \rangle`$ and is a different state.

A localization below the population floor is refused with the populations printed — a guess
built from half-delocalized orbitals is a symmetric guess wearing a fragment label, and it
converges straight back to the closed-shell solution. Usage:
[workflows](../guide/workflows.md#antiferromagnetically-coupled-centres-broken-symmetry).

## Warm starts and basis projection

`guess_from=` starts an SCF from another calculation's orbitals — used directly over the
same AO basis, or projected through the **same projector** the CASSCF's `project_from=`
uses ([integrals](integrals.md#carrying-orbitals-between-basis-sets)): one expression, two
callers, never a second implementation
[[50]](../references.md#r50)[[47]](../references.md#r47). ⚠ A warm start decides *which*
stationary point the SCF finds — measured, a projected guess on TiCl₃ converged faster to a
*different, internally unstable* solution — so on anything open-shell it is paired with the
stability analysis ([workflows](../guide/workflows.md#an-scf-from-another-scf-guess_from)).

## Average-of-configuration atoms

Two consumers need a spherically averaged free atom rather than a molecule: the atomic mean
field ([soc](soc.md#the-atomic-solves-their-reference-states-and-the-cache)) and the
free-atom reference orbitals behind AVAS and the atomic-reference charges. Both are
**average-of-configuration** (AOC) SCF solutions
[[23]](../references.md#r23)[[21]](../references.md#r21)[[181]](../references.md#r181):
open shells carry fractional, equal occupations over a whole shell, and the Fock is
projected onto its spherical part every cycle — the same spherical constraint, one
implementation, for the four-component and scalar paths alike
([soc](soc.md#the-atomic-solves-their-reference-states-and-the-cache) explains why the
constraint is structural rather than an occupation choice).

The scalar AOC entry point is public
(`kuiva.interface.pyscf_bridge.run_scalar_aoc(element, configuration, basis=...)`) and is
what the Slater–Condon extras are built on
[[189]](../references.md#r189)[[181]](../references.md#r181). ⚠ Its occupations are
fractional and its electron-count bookkeeping formal: feeding an AOC reference into the
molecular pipeline stages is not validated.

## The environment

Point charges enter the reference exactly here — a one-electron potential added in the AO
basis at ingestion, plus a separately reported classical charge–nucleus energy — and the
multireference layer never learns the charges exist, which is what keeps an embedded
calculation the *same* calculation
([workflows](../guide/workflows.md#embedding-in-a-crystal-point-charges),
[Molecule](../reference/stages/Molecule.md#environmentpoint_charges-)).
