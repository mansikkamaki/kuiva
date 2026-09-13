# Limitations — what not to trust

Kuiva is usable for production work **with care**, and this page is what the care is about:
every approximation of measured size, every ceiling, and every number that must be read
with its context. Each entry links to the page that explains the mechanism.

## Property operators

- **No picture-change correction is applied to the property operators by default, and the
  approximation has been measured** ([properties](methods/properties.md#the-operator-matrices)).
  `L` and `S` are the bare non-relativistic AO operators, used unchanged in the
  two-component basis — what OpenMolcas RASSI does, which makes cross-code comparison
  like-for-like. Every property dump warns and records which operators it used. The
  correction (`property_picture_change=True`) is worth, measured: 1e-4 relative on a
  free-ion g at Z = 5 rising smoothly to 1e-3 at Z = 81 (2.6e-3 on Yb³⁺), 1.9e-4 on a 3d
  complex's ground doublet, and **exactly zero** on every degeneracy. ⚠ It grows with Z, so
  those figures bound the elements measured and not a heavier one; on a level whose g
  approaches zero the relative shift is inflated by its own denominator — quote an absolute
  shift too. ⚠ Turning it on changes what a stored `mu` means while `format_version` stays
  put: the header field is what distinguishes the files, so reading the header is
  obligatory ([files](reference/files.md#the-property-dump)).
- **The same flag corrects the electric dipole, and there is no way to correct one operator
  and not the other.** For the dipole the correction is measured at 2e-6 relative on the
  hydrogen halides HF→HI and does **not** grow with Z (a near-nucleus effect on a valence
  operator); ⚠ those figures bound a valence dipole of a light diatomic — an f→d transition
  of a lanthanide, what the operator exists for, has not been measured.
- ⚠ **The electric dipole is length gauge only, and Kuiva computes no oscillator
  strengths.** There is no velocity-gauge second estimate, so no in-code number says how
  converged a transition dipole is — judge by enlarging the active space. The file carries
  the operator and its invariants; f values, Einstein coefficients and radiative rates are
  the external property code's job, as the crystal-field analysis is.

## The Hamiltonian

- ⚠ **DLU is measured at the state level: safe for splittings, not for the transverse g of
  an axial doublet** ([x2c](methods/x2c.md#local-decoupling-dlu)). Splittings within
  0.6 cm⁻¹ / 0.1%, principal g within 2e-4 relative — but the near-zero transverse g of a
  strongly axial doublet moves by ~6% of itself, exactly the number a tunnelling analysis
  reads. Check it against `decoupling_options={"partition": "single"}` before quoting. The
  DLU-transformed *property operators* remain unmeasured.
- **Absolute total energies are bounded by the basis, not by the code's thresholds**
  ([integrals](methods/integrals.md#the-default-threshold-and-what-it-does-and-does-not-bound)):
  the Cholesky default was decided on relative energies (error cancellation measured
  ~500×); walking up a basis series moves an absolute total by six to eight orders more
  than tightening the threshold does. ⚠ A Kuiva total also omits the mean-field
  double-counting term ([soc](methods/soc.md#x2camf-the-two-electron-picture-change)), so
  it is not directly comparable with a four-component total in any case. Relative energies
  — what everything here is about — are unaffected.
- ⚠ **No actinide system is validated at any tier.** The committed cross-checks against
  DIRAC and OpenMolcas stop at Bi (Z = 83); the heaviest f element in them is Dy. The basis
  sets are registered, the Hamiltonian has no element cutoff, and nothing in the path is
  element-specific — but "in scope" is not "tested", and an actinide number out of this
  code currently has no external reference behind it.

## Sizes and ceilings

- **The conventional CI reaches 20–22 half-filled spinors at an 8 GB memory limit**
  ([ci](methods/ci.md#the-sigma-vector)) — a memory bound on the *determinant count*, so it
  moves with the limit and dilute or nearly-full spaces run well past it; past 20 spinors
  the state count co-decides, and the hard limit is 64 spinors. Refused before the first
  allocation. Beyond the ceiling, the tensor-network solver takes over.
- **The Kramers-restricted CI covers odd electron counts only, and is worth its factor of
  two only above two Kramers pairs** ([ci](methods/ci.md#the-kramers-restricted-mode)):
  measured 1.8–1.9× less CPU from three averaged pairs up, ~8% *slower* at two. It does not
  raise the memory ceiling. Even electron counts are a different theorem, not implemented.
- **The integral factorization is memory-bound and the memory plan picks the route**
  ([ScalarSCF](reference/stages/ScalarSCF.md#the-two-electron-route)); the direct and
  streamed routes exist so a large system starts at all, not to be faster. ⚠ The scalar SCF
  is PySCF's and makes its own in-core/direct decision within the memory it is given: the
  direct route removes Kuiva's copy of the integral array, not necessarily every copy.
- **A tensor-network solve's memory is planned, not discovered**
  ([dmrg](methods/dmrg.md#what-a-sweep-costs-in-memory)): the largest array a sweep holds is a
  single intermediate inside an effective-Hamiltonian application, kept to one open operator
  leg by the contraction order and sized from the contraction's own structure before the first
  bond is solved. The plan prints at the start of the solve and names the bond that peaks; the
  three knobs that move it are, in order of effect, the **node partition** (the local dimension
  enters squared), the **active-space size** (one operator bond dimension) and `max_bond`. On
  fat nodes the RDM contraction's per-node operator environments, not the sweep, are the
  largest term, and they are planned and refused the same way. ⚠
  The estimate is of Kuiva's arrays: the operator compile's transient and the arena it leaves
  put the process about 2× above the plan on a fat-node system, so leave that much headroom
  in the limit.
- **One node, shared memory.** There is no MPI and no distributed tensor layer; memory, not
  core count, is the scaling limit.

## Degeneracies and state averages

- **Kramers degeneracy in the general two-component CI emerges numerically, not by
  construction** — measured far below the 1e-8…1e-6 Eh band reserved for genuine numerical
  splittings, but not zero by symmetry. The tensor-network solver's figure is machine zero
  at a saturating cap and, at a truncating cap, is the truncation error itself (equal to the
  energy error within a factor of two); neither bounds the other.
- **The rotation is Kramers constrained by default, and the constrained answer is then
  tested rather than assumed**
  ([casscf](methods/casscf.md#keeping-the-orbitals-kramers-paired)). The constraint holds
  every orbital space time-reversal closed exactly, which stops a drift that otherwise grows
  by a factor of ten every few macro-iterations until the state-average gate refuses; at the
  converged point Kuiva measures the curvature of the orbital Hessian along the time-odd
  rotations the constraint forbids, and where that is negative it releases the constraint and
  follows the instability. ⚠ **The release happens at an even active electron count only** —
  at an odd count a time-reversal-broken solution has no Kramers degeneracy and is refused
  downstream anyway — so at an odd count a negative curvature is reported and the symmetric
  solution kept. ⚠ **A verdict needs a converged run**: a run stopped by `max_iter` or by a
  deadline reports the curvature as not measured, which is a weaker statement than "stable".
  ⚠ **The event-gated driver** (an adaptive DMRG space) measures and never releases, and says
  so. An **unrestricted** reference, and orbitals carried from a previous unconstrained run,
  have no pairing to preserve and are never constrained; the output says which case it was.
- **The state average is the single most common way to get a plausible wrong answer out of
  this program**, and its diagnostics are advisory: a count landing inside a near-degenerate
  manifold self-reinforces with every check clean, and one measured failure mode (a leaning
  average converging from the scalar guess into a wrong basin) is invisible to any static
  check. Read [workflows](guide/workflows.md#designing-a-state-average) before setting
  `n_states` on anything harder than a ground doublet.
- **An energy window resolves a count, and what it guarantees is a clean *cut*, not a
  meaningful ensemble** ([casscf](methods/casscf.md#resolving-the-count-from-an-energy-cutoff)).
  The manifold rule cannot split a numerically degenerate block and the resolved boundary gap
  is wider than the unambiguity threshold by construction — but a window says nothing about
  whether the ensemble is one the symmetry leaves invariant, and the spin non-invariance
  report and the four questions of
  [workflows](guide/workflows.md#designing-a-state-average) apply to a resolved count exactly
  as they do to a stated one. ⚠ A window whose verdict is still moving when `max_rounds` runs
  out keeps the last converged round and is reported as **ambiguous**: both counts are
  self-consistent fixed points, and which side of the straddling state the calculation is
  about is a question about the request. On the tensor-network route a window can also be
  refused outright, because the narrowest two-site window of the topology bounds the ensemble
  and therefore the witness root the verdict needs
  ([dmrg](methods/dmrg.md#resolving-a-state-count-on-the-network)).
- ⚠ **A window resolved on a truncating tensor network is a statement about the network's
  spectrum, not the exact one** — and a truncating cap *splits degenerate manifolds*:
  measured, a far trimer's eight exactly degenerate product states came back spread over
  13 096 cm⁻¹ at a bond dimension of 8, from a **converged** sweep. The rule then reads a cut
  that can sit inside a manifold. Where the electron count is odd the resolution measures the
  splitting inside each Kramers pair (it is the truncation's, by theorem) and warns when it
  exceeds the manifold gap; where it is even there is no theorem to measure against, and the
  bond-dimension series is the only evidence. The pilot that supplies the ladder's first rung
  is subject to the same effect and its count is a **lower bound** on a multi-site system.
- **Point-group symmetry is abelian double groups only, opt-in, and does not make a state
  average safe** ([symmetry](methods/symmetry.md#what-abelian-symmetry-cannot-promise)): a
  per-irrep count can split a physically degenerate manifold exactly as a plain count can.
  The non-abelian layer classifies and never adapts; it refuses a count cutting a
  theory-fixed multiplet and protects nothing across near-degeneracies.

## Methods that are partial or inferential

- **NEVPT2 is strongly contracted only** — FIC and quasi-degenerate variants are not
  implemented and not planned, on a measurement
  ([nevpt2](methods/nevpt2.md#not-implemented-on-a-measurement)). ⚠ A tensor-network
  reference reaches NEVPT2 with **six of the eight classes**: that `E2` is a loud
  **partial** sum and is not comparable with a complete NEVPT2
  ([nevpt2](methods/nevpt2.md#the-network-route)).
- ⚠ **A term label is an inference and is printed as one**
  ([properties](methods/properties.md#the-spin-analysis-and-the-term-assignment-offer)):
  its own report, evidence and fit residual beside every label, `?` where the evidence does
  not add up (the normal outcome for crystal-field levels). Do not quote a label without
  the residual next to it. `<S²>` is a measurement and trustworthy — per degenerate block
  only.
- ⚠ **An automatically chosen active space is a proposal made by a *qualitative* probe**
  ([AutoCAS](reference/stages/AutoCAS.md)), and three of its limits bite in practice. The
  keep/drop verdict is taken on a cheap CI's state spectrum at the reference orbitals —
  comparable only with another at the same orbitals and budget, which is what the protocol
  does, and never with a CASSCF's. A class whose effect is below what it resolves comes back
  **"inconclusive, kept"** rather than decided: that is the normal outcome for lanthanide
  exchange, where the manifold is split by a few cm⁻¹, and it means the bridge orbitals are
  in the space because they were asked for. ⚠ A "kept" means the class changes the CI at
  **fixed orbitals** — correlation, or the state-specific relaxation a larger space allows —
  and the second alone keeps a **double shell** on a one-electron ion (Ti(3+), Ce(3+)), where
  there is no correlation for it to describe. ⚠ On a coupled polynuclear system the stage
  refuses a determinant budget below the product of the sites' Hund configurations (32 768
  for three high-spin d⁵ ions), and ⚠ **where the coupled core is larger than the budget no
  class is decided at all**: requested classes come back "kept (not measurable)", because a
  truncated cheap CI of coupled centres was measured not to represent their exchange
  manifold. An automatically assembled polynuclear space beyond the complete-CI size is
  therefore exactly the classes that were requested. And the proposed state
  count is a boundary read off a pre-optimized probe's spectrum — what makes it a count is
  the state-averaging gate, the boundary diagnostic and the window's ladder downstream, none
  of which this stage replaces.
- ⚠ **Shells of two different `l` (a 3d and a 4f centre) are one AVAS projection onto the
  union of their reference shells**, with each selected pair attributed to the shell it
  projects onto most. The attribution is a comparison of projections, not a rotation, so
  where two shells' orbitals genuinely mix it can come out other than `2l+1` pairs per
  centre; that **warns**, and the per-centre electron counts and floors built on it are then
  not reliable, although the space is still the union asked for.
- ⚠ **Löwdin atomic charges are withdrawn from every printed report** (measured sign-wrong
  on three of five characterized systems, unrescued by basis); the supported charge is the
  atomic-reference partition
  ([properties](methods/properties.md#populations-and-charges)).

## Scope boundaries

- **Magnetic properties themselves are out of scope**: Kuiva writes operator matrices and
  their invariants; the ITO/Stevens/crystal-field decomposition — and every intensity — is
  an external code's job.
- **Four-component methods are out of scope**: four-component machinery exists only as an
  ingredient of the two-electron picture change.
- **An unrestricted reference gives spinors that are not Kramers paired**, so an active
  space cannot be a contiguous spinor range there
  ([scf-reference](methods/scf-reference.md#restricted-and-unrestricted-references)).
