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
- **One node, shared memory.** There is no MPI and no distributed tensor layer; memory, not
  core count, is the scaling limit.

## Degeneracies and state averages

- **Kramers degeneracy in the general two-component CI emerges numerically, not by
  construction** — measured far below the 1e-8…1e-6 Eh band reserved for genuine numerical
  splittings, but not zero by symmetry. The tensor-network solver's figure is larger
  (~1e-9 Eh) and for a different reason (roots converged separately); both are far below
  physics, and neither bounds the other.
- **The state average is the single most common way to get a plausible wrong answer out of
  this program**, and its diagnostics are advisory: a count landing inside a near-degenerate
  manifold self-reinforces with every check clean, and one measured failure mode (a leaning
  average converging from the scalar guess into a wrong basin) is invisible to any static
  check. Read [workflows](guide/workflows.md#designing-a-state-average) before setting
  `n_states` on anything harder than a ground doublet.
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
