"""Strongly contracted NEVPT2 on a two-component multireference reference.

**Orchestration.** The driver, the zeroth-order Hamiltonian and the reporting live here; the
per-class working equations are in :mod:`kuiva.pt.classes`, the active-space contractions in
:mod:`kuiva.pt.contractions`, and the integral blocks in :mod:`kuiva.pt.blocks`. Nothing in
this module is a registered kernel.

What it computes
----------------
The second-order correction to a converged CASCI/CASSCF, per state, decomposed by excitation
class. It is a **post-processing stage**: it consumes converged orbitals, CI vectors and
integral factors and changes no wavefunction. ``E2`` is reported beside ``E_CASSCF``; the
states remain the CASSCF states.

**All eight classes produce an energy**, so ``E2`` is the complete strongly contracted
correction. :attr:`NEVPT2Result.complete` still exists and the driver still refuses to call an
incomplete sum ``E2`` — a class restricted away by ``classes=`` or added later must not be able
to shrink the total quietly. The class table in :mod:`kuiva.pt.classes` is the authority on
what needs what, including ⚠ the finding that a Kuiva external label is a **degenerate-``eps``
group of spinors**, which is a *coarser* contraction than the published spin-free one wherever
two spatial orbitals are symmetry degenerate.

The zeroth-order Hamiltonian
----------------------------
Dyall's, in the two-component spinor basis::

    H_D = C + sum_i eps_i a+_i a_i + sum_a eps_a a+_a a_a + H_act
    H_act = sum_tu f^I_tu a+_t a_u + 1/2 sum_tuvw (tu|vw) a+_t a+_v a_w a_u

with ``i`` inactive, ``a`` virtual, ``t..w`` active, and ``H_act`` the **exact** active-space
Hamiltonian — in Kuiva the full two-component one, complex, carrying spin-orbit coupling and
the X2CAMF ``Dh_sf``/``Dw`` already inside ``f^I``. ``C`` is fixed by
``H_D |Psi_0> = E_0 |Psi_0>``, which is what makes every ``dE_l`` an energy difference from the
reference and not from an arbitrary origin.

A standing instruction — *the zeroth-order splitting is defined on the spinor
Fock matrix* — is the ``eps`` above: eigenvalues of the state-averaged generalized Fock
``F^I + F^A`` over the inactive and virtual blocks (:func:`pseudo_canonicalize`).

⚠ **Any Hermitian operator is a legal ``H0``, so the atomic mean field being inside ``f^I`` is
a choice of partitioning and not a double counting.** the anchor-admissibility rule that no reported total
contains ``DG`` is untouched: what is reported here is ``E2`` and ``E_CASSCF + E2``.

Four decisions that bind the rest of the calculation
-----------------------------------------------------
* **The Fock is built from the block-equalized state-averaged density by default**. A
  single state's density is not time-reversal even, so a state-specific Fock breaks the
  symmetry *in ``H0`` itself* and splits a degenerate manifold's ``E2`` by a purely artificial
  amount — **measured at 0.7 to 340 cm^-1 against the averaged Fock's 1e-6..1e-4**, i.e. five
  to six orders, and far above the 0.1 cm^-1 at which a splitting already implies different
  physics. ``fock="state-specific"`` exists so that that mechanism can be *measured*
  (``tests/generate/nevpt2_multiplet.py``), never for production. The density goes through
  :func:`kuiva.rdm.rdm.state_average_weights`, the same state-averaging gate every solver uses. Per-state
  RDMs still drive each state's norms and Koopmans matrices — that is the state-specific
  content of SC-NEVPT2.

  ⚠ **What it destroys is a manifold LARGER than a Kramers pair, not a Kramers pair**, and the
  distinction was measured rather than assumed. A doublet's two members are time reverses of
  each other whatever basis the eigensolver returned, so the time-reversed density builds the
  time-reversed Fock, every ``eps`` is identical and so is every class energy: a ligand-field
  system with nothing but doublets gives 4e-07 cm^-1 where a free ion gives tens.
* ⚠ **Nothing makes the per-state ``E2`` of a degenerate manifold's members well defined, and
  that is a property of the method rather than of this code.** The correction is state specific
  through the per-state RDMs, and the CI's basis inside a degenerate manifold is arbitrary, so
  the members carry an arbitrary share of the manifold's internal spread. It is **not** the
  degenerate-``eps`` freedom, which the group-complete contraction does remove — the
  discriminator is that the orbital one shows up with a single state and a fixed CI vector.
  Measured at 1e-6..1e-4 cm^-1 on state-averaged CASSCF references and larger on a
  non-stationary one, as ``S^(0')`` is. The treatment is reporting, not repair:
  :meth:`NEVPT2Result.multiplets` and :func:`_report_multiplets` give each manifold's
  barycentre — measurably four to seven orders more stable — **beside** the member energies and
  never instead of them.
* **The active block is never rotated.** ``H_D`` needs no active canonicalization and the CI
  vectors are expressed in the active basis as converged, so rotating it would invalidate them.
  A consequence worth stating: the active Hamiltonian, the core energy and hence every CI
  eigenvalue are **invariant** under this canonicalization, and that is asserted rather than
  assumed (:func:`pseudo_canonicalize`, and a Tier-0 test).
* **Every threshold drops whole degenerate groups** (the whole-degenerate-group rule in its PT2 instance).
  See :func:`kuiva.pt.classes.group_complete_mask`.

⚠ The conjugation trap applies to the canonical rotation like every other:
coefficients transform as ``C -> C U`` while a density transforms oppositely. The test for it
goes through the object that is used downstream — rotate, rebuild the Fock, assert the block is
diagonal with the expected spectrum — never through the algebra in isolation.

Dependency direction (asserted from the sources, as for ``kuiva/x2c/`` and ``kuiva/qc/``)
------------------------------------------------------------------------------------------
``kuiva.pt`` imports from ``ci/``, ``mcscf/``, ``rdm/``, ``integrals/`` and — for the optional
dump seam of :func:`corrected_property_matrices` — ``props/``; **nothing in the calculation path
imports** ``kuiva.pt``. Corrected energies reach the property dump because this module hands them
*to* it, on request; ``props/`` does not learn that ``pt/`` exists.

References
----------
* K. G. Dyall, J. Chem. Phys. 102, 4909 (1995), doi:10.1063/1.469539 — the zeroth-order
  Hamiltonian.
* C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu, J. Chem. Phys. 114,
  10252 (2001), doi:10.1063/1.1361246; C. Angeli, R. Cimiraglia, J.-P. Malrieu, J. Chem. Phys.
  117, 9138 (2002), doi:10.1063/1.1515317; C. Angeli, M. Pastore, R. Cimiraglia, Theor. Chem.
  Acc. 117, 743 (2007), doi:10.1007/s00214-006-0207-0 — NEVPT2.
* Level shifts for intruder states: B. O. Roos, K. Andersson, Chem. Phys. Lett. 245, 215
  (1995), doi:10.1016/0009-2614(95)01010-7 (real); N. Forsberg, P.-A. Malmqvist, Chem. Phys.
  Lett. 274, 196 (1997), doi:10.1016/S0009-2614(97)00669-6 (imaginary).
* Q. Sun et al., WIREs Comput. Mol. Sci. 8, e1340 (2018), doi:10.1002/wcms.1340 — PySCF, the
  Tier-1 scalar reference implementation this is checked against class by class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ci.strings import CASSpace
from ..integrals.transform import ThreeIndexAO
from ..mcscf.orbopt import CASIntegrals, OrbitalSpaces, averaged_fock
from ..rdm.rdm import DEFAULT_DEGENERACY_TOL, RDMBuilder, state_average_weights
from ..util import output as out
from ..util.logging import get_logger
from ..util.timing import timer
from .blocks import IntegralBlocks
from .classes import (DEFAULT_NORM_CUTOFF, EPS_DEGENERACY_RTOL, ClassContext, ClassResult,
                      available_classes, degeneracy_groups, excitation_class)
from .contractions import (CIContractionProvider, ShiftedSpaces,
                           active_hamiltonian_check)

log = get_logger(__name__)

#: How the Dyall Fock may be built. ``"state-averaged"`` is the default and the only
#: production choice; ``"state-specific"`` exists to measure what it costs.
FOCK_MODES = ("state-averaged", "state-specific")

#: Largest change in the active-space Hamiltonian the pseudo-canonicalization may cause before
#: it is warned about [Eh]. Zero in exact arithmetic — the active block is not rotated — so a
#: nonzero value means the inactive/virtual rotation leaked into the active space.
ACTIVE_INVARIANCE_TOL = 1.0e-10

#: ⚠ **The intruder band [Eh], and both bounds are FIXED IN ADVANCE.** ``min_denominator`` was
#: computed and printed for every class from the beginning and compared against nothing, so
#: the one number that says whether a class's ``E2`` can be believed sat in a table with no
#: verdict attached to it.
#:
#: Two tiers, because there are two distinct failures and they deserve different volumes:
#:
#: * :data:`INTRUDER_WARN_EH` — the conventional intruder band. ``E2`` is still a number, but a
#:   perturber this close to the reference contributes out of proportion to its physical
#:   weight and the class is sensitive to everything upstream of it. The level shifts exist
#:   for exactly this, and the message names them.
#: * :data:`INTRUDER_SEVERE_EH` — a denominator that is essentially zero, or (the signed
#:   check) **negative**: a perturber has fallen to or below the reference. That is not a
#:   badly conditioned sum, it is a divergent term, and the class energy is meaningless rather
#:   than merely suspect. A negative denominator additionally makes it wrong in *sign*, which
#:   no amount of looking at the magnitude would reveal.
#:
#: ⚠ Both were chosen **before** measuring the shipped systems, which is the project's
#: pre-registration discipline: a threshold set to whatever the current test set happens to
#: produce is a threshold that fires on the next one and cannot fail on this one. What the
#: shipped systems actually show is recorded in this package's validation notes, so the band
#: re-opens on data rather than on impression.
INTRUDER_WARN_EH = 0.1
INTRUDER_SEVERE_EH = 1.0e-6


# --- pseudo-canonicalization ----------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalOrbitals:
    """Orbitals in which the inactive and virtual blocks of the Dyall Fock are diagonal."""

    coeff: np.ndarray                 #: (2*nao, n_orb) rotated spinor coefficients
    spaces: OrbitalSpaces
    ints: CASIntegrals                #: rebuilt at :attr:`coeff`
    eps_inactive: np.ndarray          #: ascending inactive orbital energies [Eh]
    eps_virtual: np.ndarray           #: ascending virtual orbital energies [Eh]
    u_inactive: np.ndarray            #: the applied rotation, for provenance and tests
    u_virtual: np.ndarray
    #: How far the active-space Hamiltonian moved. Should be rounding; see
    #: :data:`ACTIVE_INVARIANCE_TOL`.
    active_drift: float = 0.0

    def __repr__(self) -> str:
        return "CanonicalOrbitals(n_inactive={}, n_virtual={}, active_drift={:.2e})".format(
            self.eps_inactive.size, self.eps_virtual.size, self.active_drift)


def pseudo_canonicalize(factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
                        spaces: OrbitalSpaces, gamma: np.ndarray, *,
                        e_nuc: float = 0.0,
                        ints: Optional[CASIntegrals] = None) -> CanonicalOrbitals:
    """Diagonalize the inactive and virtual blocks of ``F^I + F^A``; leave active alone.

    Converged CASSCF orbitals diagonalize no Fock matrix — :func:`kuiva.mcscf.orbopt
    .fock_diagonal`'s docstring already warns that its diagonal is a label — so the ``eps`` of
    ``H_D`` do not exist until this runs. The rotation is block diagonal, unitary, and applied
    to the *coefficients*; the active columns are untouched.

    ⚠ **Inside a degenerate ``eps`` group the canonical spinors are defined only up to a
    unitary rotation, and nothing downstream may depend on the choice.** With the time-even
    state-averaged Dyall density the ``eps`` come in exactly degenerate Kramers pairs, so this
    is not a corner case but the generic situation. Both SC and FIC energies contract over
    complete label sets and are therefore exactly invariant under it — a Tier-0 assertion,
    and the PT2 instance of the whole-degenerate-group rule.

    Parameters
    ----------
    gamma : ``(n_act, n_act)``
        The active 1-RDM the Fock is built from. ⚠ The **state-averaged, block-equalized** one
        for a production run; the driver applies the state-averaging gate before calling.
    ints : :class:`~kuiva.mcscf.orbopt.CASIntegrals`, optional
        Already built at ``c_spinor``, to save one Fock build.
    """
    c_spinor = np.ascontiguousarray(c_spinor)
    with timer("NEVPT2 pseudo-canonicalization"):
        ints = ints if ints is not None else CASIntegrals.build(
            factors, h_ao, c_spinor, spaces, e_nuc=e_nuc)
        fock = averaged_fock(ints, factors, c_spinor, gamma)
        # F^A is a Coulomb/exchange build on a factorized density, so its Hermiticity is
        # accurate to rounding, not exact; eigh would silently read only one triangle.
        fock = 0.5 * (fock + fock.conj().T)
        eps_i, u_i = _diagonalize_block(fock, spaces.inactive)
        eps_v, u_v = _diagonalize_block(fock, spaces.virtual)

        rotated = c_spinor.copy()
        if spaces.n_inactive:
            rotated[:, spaces.inactive] = c_spinor[:, spaces.inactive] @ u_i
        if spaces.n_virtual:
            rotated[:, spaces.virtual] = c_spinor[:, spaces.virtual] @ u_v
        new_ints = CASIntegrals.build(factors, h_ao, rotated, spaces, e_nuc=e_nuc)

    drift = max(
        float(np.max(np.abs(new_ints.h_active_effective() - ints.h_active_effective())))
        if spaces.n_active else 0.0,
        abs(new_ints.e_core - ints.e_core))
    if drift > ACTIVE_INVARIANCE_TOL:
        log.warning("the pseudo-canonicalization moved the active-space Hamiltonian by %.3e "
                    "Eh; it rotates only the inactive and virtual blocks, so the CI vectors "
                    "solved before it should still be exact eigenvectors and now are not",
                    drift)
    log.debug("pseudo-canonical eps: %d inactive in [%.4f, %.4f], %d virtual in [%.4f, %.4f] "
              "Eh; active drift %.2e", eps_i.size, _lo(eps_i), _hi(eps_i),
              eps_v.size, _lo(eps_v), _hi(eps_v), drift)
    return CanonicalOrbitals(coeff=rotated, spaces=spaces, ints=new_ints,
                             eps_inactive=eps_i, eps_virtual=eps_v,
                             u_inactive=u_i, u_virtual=u_v, active_drift=drift)


def _diagonalize_block(fock: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        return np.zeros(0), np.zeros((0, 0), dtype=np.complex128)
    eps, u = np.linalg.eigh(fock[np.ix_(idx, idx)])
    return np.ascontiguousarray(eps.real), np.ascontiguousarray(u)


def _lo(a):
    return float(a[0]) if a.size else float("nan")


def _hi(a):
    return float(a[-1]) if a.size else float("nan")


# --- frozen core and deleted virtuals ----------------------------------------------------

@dataclass(frozen=True)
class CorrelatedSpaces:
    """Which inactive and virtual spinors the perturbation may use as external labels.

    Positions **within** ``spaces.inactive`` / ``spaces.virtual``, in the pseudo-canonical
    order, so they are directly usable as :class:`~kuiva.pt.blocks.IntegralBlocks` selections.
    """

    inactive: np.ndarray
    virtual: np.ndarray
    n_frozen: int = 0
    n_deleted: int = 0

    @property
    def restricted(self) -> bool:
        return bool(self.n_frozen or self.n_deleted)

    def __repr__(self) -> str:
        return "CorrelatedSpaces({} core frozen, {} virtuals deleted)".format(
            self.n_frozen, self.n_deleted)


def select_correlated(eps_inactive: np.ndarray, eps_virtual: np.ndarray, *,
                      frozen_core: Optional[float] = None,
                      deleted_virtual: Optional[float] = None,
                      rtol: float = EPS_DEGENERACY_RTOL) -> CorrelatedSpaces:
    """Resolve the frozen-core / deleted-virtual thresholds into label selections.

    ⚠ **The selection is stated as an energy, never as a count of orbitals**. After the
    pseudo-canonicalization the inactive and virtual spinors *are* eigenvectors of the Dyall
    Fock, so their ``eps`` is the only label that means the same thing in another program or
    another basis; "the lowest four" does not. The number itself is chosen by looking at the
    orbital energies and the reduced orbital populations of the shells in question — the physical
    statement is "the 1s..(n-2) shells", and the threshold is how it is expressed.

    ⚠ **A threshold that would split a degenerate ``eps`` group is refused, not rounded.**
    Freezing half a shell is the whole-degenerate-group rule broken at the orbital level: it
    removes some members of a Kramers or symmetry orbit from every class's label range and
    keeps the others, which manufactures exactly the splitting reserved the 1e-8..1e-6 Eh
    band for. The refusal names the group and the two thresholds that would not split it.

    Parameters
    ----------
    eps_inactive, eps_virtual : ascending pseudo-canonical orbital energies [Eh].
    frozen_core : float, optional
        Freeze every inactive spinor with ``eps < frozen_core``. ``None`` correlates all.
    deleted_virtual : float, optional
        Delete every virtual spinor with ``eps > deleted_virtual``. ``None`` keeps all.
    """
    keep_i = _threshold_selection("inactive", eps_inactive, frozen_core, below=True, rtol=rtol)
    keep_v = _threshold_selection("virtual", eps_virtual, deleted_virtual, below=False,
                                  rtol=rtol)
    return CorrelatedSpaces(inactive=keep_i, virtual=keep_v,
                            n_frozen=int(eps_inactive.size - keep_i.size),
                            n_deleted=int(eps_virtual.size - keep_v.size))


def _threshold_selection(space: str, eps: np.ndarray, threshold: Optional[float],
                         *, below: bool, rtol: float) -> np.ndarray:
    """Positions kept; refuse if the cut falls inside a degenerate group."""
    eps = np.asarray(eps, dtype=float).ravel()
    if threshold is None:
        return np.arange(eps.size)
    dropped = eps < float(threshold) if below else eps > float(threshold)
    groups = degeneracy_groups(eps)
    for group in np.unique(groups):
        member = groups == group
        if dropped[member].any() and not dropped[member].all():
            values = eps[member]
            raise ValueError(
                "the {} threshold {:.6f} Eh cuts through a degenerate group of {} spinors at "
                "{:.6f} Eh: freezing or deleting part of a shell removes some members of a "
                "Kramers or symmetry orbit from every class label range and keeps the others, "
                "which manufactures a splitting no later check can tell from a physical one "
                ". Move the threshold {} {:.6f} Eh or {} "
                "{:.6f} Eh".format(space, float(threshold), int(member.sum()),
                                   float(values[0]),
                                   "below" if below else "above", float(values.min()),
                                   "above" if below else "below", float(values.max())))
    kept = np.nonzero(~dropped)[0]
    if kept.size == 0:
        log.warning("the %s threshold %.6f Eh removes every spinor of that space, so every "
                    "excitation class with a %s label contributes exactly zero", space,
                    float(threshold), space)
    return kept


# --- degenerate manifolds (C1) -----------------------------------------------------------------

#: How wide a window counts as one degenerate manifold of the reference spectrum [cm^-1].
#: Far above the 1e-6..1e-5 cm^-1 a converged CI carries on a degenerate block and far below any
#: physical splitting — a ligand field's smallest is hundreds of cm^-1.
DEFAULT_MULTIPLET_TOL_CM = 1.0

#: The corrected spread at which a degenerate manifold stops meaning one level [cm^-1], and the
#: threshold :func:`NEVPT2Result.multiplets` warns above. ⚠ **Not a numerical tolerance**: physics
#: fixes 0.1 cm^-1 as the size at which a splitting already implies different physics, so a
#: manifold spread above it is a statement about the result, not about rounding.
MULTIPLET_TOL_CM = 0.1


@dataclass(frozen=True)
class MultipletCorrection:
    """One degenerate manifold of the **reference** spectrum, after correction.

    ⚠ **The manifolds are read off the reference spectrum and applied unchanged to the corrected
    one.** Re-grouping the corrected energies would let a manifold the perturbation split
    *become* two manifolds, and the diagnostic would report zero spread for exactly the split it
    exists to find.
    """

    start: int                      #: first state of the manifold, in ascending reference order
    size: int
    e_reference: float              #: barycentre of ``E_CASSCF`` over the manifold [Eh]
    e_corrected: float              #: barycentre of ``E_CASSCF + E2`` [Eh]
    reference_spread_cm: float      #: what the reference already carried [cm^-1]
    corrected_spread_cm: float      #: what the corrected members carry [cm^-1]

    @property
    def e2(self) -> float:
        """The manifold's correction [Eh] — the barycentre one, which is the invariant one."""
        return self.e_corrected - self.e_reference

    @property
    def degenerate(self) -> bool:
        """Whether the corrected members still mean one level (the 0.1 cm^-1 physical-degeneracy)."""
        return self.corrected_spread_cm <= MULTIPLET_TOL_CM

    def __repr__(self) -> str:
        return ("MultipletCorrection(start={}, size={}, E2={:.10f} Eh, spread={:.3e} cm^-1)"
                .format(self.start, self.size, self.e2, self.corrected_spread_cm))


# --- the result ------------------------------------------------------------------------------

@dataclass
class NEVPT2Result:
    """Per-state SC-NEVPT2 corrections and their class decomposition."""

    e_casscf: np.ndarray                       #: (n_states,) reference total energies [Eh]
    e2: np.ndarray                             #: (n_states,) second-order correction [Eh]
    #: ``name -> (n_states,)``. ⚠ ``nan`` where the class cannot yet form an energy — never
    #: zero, so that a caller summing the dictionary gets ``nan`` rather than a wrong total.
    class_energies: Dict[str, np.ndarray] = field(default_factory=dict)
    class_norms: Dict[str, np.ndarray] = field(default_factory=dict)
    eps_inactive: Optional[np.ndarray] = None
    eps_virtual: Optional[np.ndarray] = None
    #: True only when every one of the eight classes contributed an energy.
    complete: bool = False
    missing: Tuple[str, ...] = ()
    fock: str = "state-averaged"
    shift: float = 0.0
    imaginary_shift: bool = False
    #: How many inactive spinors kept their mean field but were removed from the class label
    #: ranges, and how many virtuals were dropped. Both zero by default.
    n_frozen: int = 0
    n_deleted: int = 0
    #: The raw per-state, per-class results, for diagnostics and for the multiplet protocol
    #: of ``tests/generate/nevpt2_multiplet.py``.
    per_state: List[Dict[str, ClassResult]] = field(default_factory=list)

    @property
    def total_energies(self) -> np.ndarray:
        """``E_CASSCF + E2`` [Eh]. ⚠ A **partial** total while :attr:`complete` is false."""
        return self.e_casscf + self.e2

    def excitation_energies_cm(self) -> np.ndarray:
        """Corrected relative state energies [cm^-1], from the lowest corrected root."""
        from ..props.multiplet import HARTREE_TO_CM
        total = self.total_energies
        return (total - total.min()) * HARTREE_TO_CM

    def multiplets(self, tol_cm: float = DEFAULT_MULTIPLET_TOL_CM
                   ) -> List[MultipletCorrection]:
        """The reference's degenerate manifolds, with the correction's barycentre and spread.

        **Why a barycentre is worth reporting at all.** SC-NEVPT2 is state specific through the
        per-state RDMs that build every norm and every Koopmans matrix. Inside a degenerate
        manifold the CI returns an *arbitrary* unitary mix of the members, those RDMs are not
        invariant under it, and so the individual ``E2`` are not either — while the manifold's
        **barycentre is**, by five orders of magnitude on the measured systems
        (measured). The barycentre is therefore the quantity that means
        something about the physics; a member energy is that plus an arbitrary share of the
        manifold's internal spread.

        ⚠ **This is a reporting discipline and nothing more. It is offered alongside the member
        energies and never instead of them**, and :attr:`total_energies` is untouched. Averaging
        the members would *hide* the spread, which is the only observable of the mechanism, and
        a barycentre is legitimate only where the degeneracy is exact — which is why the
        manifolds come from the reference spectrum and :attr:`MultipletCorrection
        .reference_spread_cm` is reported beside the corrected one. A manifold that is not
        degenerate in the reference has no barycentre worth taking.

        ⚠ **A spread of the order of the reference's own is a statement about the CI solver, not
        about the perturbation.** A converged Davidson leaves 1e-6..1e-5 cm^-1 on a degenerate
        block; reporting the corrected spread alone would credit the perturbation with it.

        Parameters
        ----------
        tol_cm : float
            Manifold grouping width on the **reference** spectrum. See
            :data:`DEFAULT_MULTIPLET_TOL_CM`.

        Raises
        ------
        ValueError
            If the reference energies are unavailable — the driver was not given ``energies=``,
            so there is no spectrum to find manifolds in.
        """
        from ..props.multiplet import HARTREE_TO_CM, degenerate_blocks

        e_ref = np.asarray(self.e_casscf, dtype=float)
        if e_ref.size and not np.all(np.isfinite(e_ref)):
            raise ValueError(
                "the reference state energies are not available, so the degenerate manifolds "
                "cannot be identified. Pass energies= to sc_nevpt2(); grouping the *corrected* "
                "spectrum instead would let a manifold this correction split become two "
                "manifolds and report no spread for it")
        if e_ref.size > 1 and np.any(np.diff(e_ref) < 0.0):
            # `degenerate_blocks` sorts defensively and returns positions in the sorted order,
            # so on an unsorted spectrum its (start, size) would address the wrong states.
            raise ValueError(
                "the reference state energies are not ascending, so a manifold's (start, size) "
                "would not address the states it names. Pass the spectrum in energy order")
        total = np.asarray(self.total_energies, dtype=float)
        rel = (e_ref - e_ref.min()) * HARTREE_TO_CM if e_ref.size else e_ref
        out_blocks: List[MultipletCorrection] = []
        for start, size in degenerate_blocks(rel, tol_cm=float(tol_cm)):
            sl = slice(start, start + size)
            out_blocks.append(MultipletCorrection(
                start=int(start), size=int(size),
                e_reference=float(np.mean(e_ref[sl])),
                e_corrected=float(np.mean(total[sl])),
                reference_spread_cm=float(np.ptp(rel[sl])),
                corrected_spread_cm=float(np.ptp(total[sl]) * HARTREE_TO_CM)))
        return out_blocks

    def __repr__(self) -> str:
        return "NEVPT2Result(n_states={}, E2[0]={:.10f} Eh, complete={})".format(
            self.e2.size, float(self.e2[0]) if self.e2.size else float("nan"), self.complete)


# --- the driver --------------------------------------------------------------------------------

def sc_nevpt2(factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
              spaces: OrbitalSpaces, civecs: np.ndarray, n_elec: int, *,
              energies: Optional[Sequence[float]] = None,
              weights: Optional[Sequence[float]] = None,
              e_nuc: float = 0.0,
              classes: Optional[Sequence[str]] = None,
              fock: str = "state-averaged",
              frozen_core: Optional[float] = None,
              deleted_virtual: Optional[float] = None,
              shift: float = 0.0,
              imaginary_shift: bool = False,
              norm_cutoff: float = DEFAULT_NORM_CUTOFF,
              degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
              on_split: str = "raise",
              report: bool = True) -> NEVPT2Result:
    """SC-NEVPT2 on converged orbitals and CI vectors.

    Parameters
    ----------
    factors, h_ao, c_spinor, spaces, e_nuc
        The same objects a CASCI is run with (:func:`kuiva.mcscf.casci.casci`). ``c_spinor``
        is the **converged** set; this function canonicalizes it internally and does not
        modify the caller's array.
    civecs : ``(ndet,)`` or ``(n_states, ndet)``
        Rows are states, as :attr:`kuiva.mcscf.casci.CASCIResult.vectors`.
    n_elec : int
        Active electrons.
    energies : sequence, optional
        Active-space eigenvalues [Eh] (``CASCIResult.energies``). Required for more than one
        state, because the state average's block-completeness rule cannot be applied without them.
    classes : sequence of str, optional
        Restrict evaluation to these class names. Default: every registered class, with any
        whose status is not ``"energy"`` reported and skipped.
    frozen_core, deleted_virtual : float, optional
        Orbital-energy thresholds [Eh] on the pseudo-canonical ``eps``: inactive spinors below
        ``frozen_core`` keep their mean field but are removed from every class's label range,
        and virtual spinors above ``deleted_virtual`` are dropped. ⚠ **The default
        correlates everything**, and a threshold that would split a degenerate ``eps`` group is
        refused — see :func:`select_correlated`.
    shift, imaginary_shift : float, bool
        Level shift on every denominator. ⚠ **The default is parameter-free**, and any
        nonzero shift emits a ``WARNING`` and is recorded on the result — a shifted energy is
        not the same quantity as an unshifted one.
    fock : ``"state-averaged"`` or ``"state-specific"``
        See the module docstring. ⚠ ``"state-specific"`` is a measurement tool, not a
        production option, and warns.

    Returns
    -------
    :class:`NEVPT2Result`
    """
    engine = _CIEngine(civecs, spaces.n_active, int(n_elec))
    return _drive(engine, factors, h_ao, c_spinor, spaces, int(n_elec),
                  energies=energies, weights=weights, e_nuc=e_nuc, classes=classes,
                  fock=fock, frozen_core=frozen_core, deleted_virtual=deleted_virtual,
                  shift=shift, imaginary_shift=imaginary_shift, norm_cutoff=norm_cutoff,
                  degeneracy_tol=degeneracy_tol, on_split=on_split, report=report)


class _CIEngine:
    """The conventional-CI half of the driver: CI vectors in, densities and providers out.

    The reference-specific seam the shared loop (:func:`_drive`) runs on. Its network
    twin lives in :mod:`kuiva.pt.network`; both serve the same four things — a state
    count, the averaged density that defines ``H0``, a per-state density for the
    measurement-only state-specific Fock, and one contraction provider per state — and
    the loop learns nothing else about where the reference came from.
    """

    kind = "conventional CI"

    def __init__(self, civecs: np.ndarray, n_active: int, n_elec: int) -> None:
        self.vectors = np.atleast_2d(np.ascontiguousarray(civecs, dtype=np.complex128))
        self.n_states = int(self.vectors.shape[0])
        self.space = CASSpace(int(n_active), int(n_elec))
        if self.vectors.shape[1] != self.space.ndet:
            raise ValueError("CI vectors have length {}, but CAS({}, {}) has {} "
                             "determinants".format(self.vectors.shape[1], n_elec,
                                                   n_active, self.space.ndet))
        self.builder = RDMBuilder(self.space)
        self._shifted: Optional[ShiftedSpaces] = None

    def averaged_gamma(self, equalized: np.ndarray) -> np.ndarray:
        return self.builder(self.vectors, equalized, enforce_kramers=False)[0]

    def state_gamma(self, state: int) -> np.ndarray:
        return self.builder(self.vectors[state], enforce_kramers=False)[0]

    def provider(self, state: int, h_act: np.ndarray, eri_act: np.ndarray):
        if self._shifted is None:
            # Built once for the run, not per state: the active Hamiltonian is the same
            # for every state (the canonicalization never rotates the active block) and
            # a sigma workspace per state would charge the memory budget n_states times
            # for one array. See contractions.ShiftedSpaces.
            self._shifted = ShiftedSpaces(self.space.n_spinor, self.space.n_elec,
                                          h_act, eri_act)
        return CIContractionProvider(self.space, self.vectors[state], h_act, eri_act,
                                     builder=self.builder, spaces=self._shifted)

    def release(self) -> None:
        if self._shifted is not None:
            self._shifted.release()


def _drive(engine, factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
           spaces: OrbitalSpaces, n_elec: int, *,
           energies=None, weights=None, e_nuc: float = 0.0, classes=None,
           fock: str = "state-averaged", frozen_core=None, deleted_virtual=None,
           shift: float = 0.0, imaginary_shift: bool = False,
           norm_cutoff: float = DEFAULT_NORM_CUTOFF,
           degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
           on_split: str = "raise", report: bool = True) -> NEVPT2Result:
    """The shared SC-NEVPT2 loop over one reference-specific ``engine``.

    Everything reference-independent lives here — the pseudo-canonicalization, the
    frozen/deleted selection, the class loop with its intruder and imaginary-part
    diagnostics, the assembly and the reporting — and the engine supplies only what
    :class:`_CIEngine`'s docstring lists. ⚠ This is :func:`sc_nevpt2`'s validated body,
    factored so a second reference type is a second engine and not a second loop.
    """
    if fock not in FOCK_MODES:
        raise ValueError("fock must be one of {}, got {!r}".format(list(FOCK_MODES), fock))
    if fock == "state-specific":
        log.warning("the Dyall Fock is being built per state. A single state's density is not "
                    "time-reversal even in general, so H0 itself breaks Kramers symmetry and "
                    "the E2 of a Kramers pair splits by an artificial amount. This option "
                    "exists to measure that mechanism, not to run calculations with")
    if shift != 0.0:
        log.warning("%s level shift of %.4e Eh is applied to every NEVPT2 denominator; the "
                    "resulting E2 is not the parameter-free one and the two may not be mixed "
                    "in one comparison", "an imaginary" if imaginary_shift else "a real", shift)

    n_states = int(engine.n_states)
    names = tuple(classes) if classes is not None else available_classes()
    for name in names:
        excitation_class(name)                              # refuse an unknown name up front

    if report:
        out.subsection(log, "SC-NEVPT2")
        out.entries(log, [
            ("active space", "CAS({}, {})".format(n_elec, spaces.n_active)),
            ("inactive / virtual spinors",
             "{} / {}".format(spaces.n_inactive, spaces.n_virtual)),
            ("states", n_states),
            ("reference", engine.kind),
            ("zeroth-order Fock", fock),
            ("level shift [Eh]", "{:.3e}{}".format(shift, " (imaginary)" if imaginary_shift
                                                   else "") if shift else "none"),
        ])

    # The averaged density that defines H0, through the same state-averaging gate as every solver.
    # ⚠ The gate runs whatever `fock` is: a state count that splits a degenerate block is a
    # defect in the *state selection*, not in this stage's choice of H0, and it is as fatal to
    # a state-specific run as to an averaged one.
    equalized = (state_average_weights(_require_energies(energies, n_states), n_elec, weights,
                                       tol=degeneracy_tol, on_split=on_split)
                 if n_states > 1 else np.ones(1))
    averaged = engine.averaged_gamma(equalized) if fock == "state-averaged" else None

    e_casscf = np.zeros(n_states)
    per_state: List[Dict[str, ClassResult]] = []
    canonical: Optional[CanonicalOrbitals] = None
    blocks: Optional[IntegralBlocks] = None
    correlated: Optional[CorrelatedSpaces] = None
    # State-independent class results, reused across states. ⚠ Only populated with the
    # state-averaged Fock, where every state shares one set of canonical orbitals and one
    # `eps`; see `ExcitationClass.state_independent`.
    shared: Optional[Dict[str, ClassResult]] = {} if fock == "state-averaged" else None
    try:
        for state in range(n_states):
            if canonical is None or fock == "state-specific":
                if blocks is not None:
                    blocks.release()
                gamma_h0 = (averaged if fock == "state-averaged"
                            else engine.state_gamma(state))
                canonical = pseudo_canonicalize(factors, h_ao, c_spinor, spaces, gamma_h0,
                                                e_nuc=e_nuc)
                correlated = select_correlated(canonical.eps_inactive, canonical.eps_virtual,
                                               frozen_core=frozen_core,
                                               deleted_virtual=deleted_virtual)
                if correlated.restricted and state == 0 and report:
                    out.entries(log, [
                        ("frozen core spinors", correlated.n_frozen),
                        ("deleted virtual spinors", correlated.n_deleted),
                    ])
                blocks = IntegralBlocks(factors, canonical.coeff, spaces,
                                        inactive_keep=correlated.inactive,
                                        virtual_keep=correlated.virtual)
            ints = canonical.ints
            provider = engine.provider(state, ints.h_active_effective(),
                                       ints.active_eri())
            if energies is not None:
                active_hamiltonian_check(provider, float(np.asarray(energies)[state]))
            e_casscf[state] = (float(np.asarray(energies)[state]) + ints.e_core
                               if energies is not None else np.nan)
            core = spaces.inactive[correlated.inactive]
            virt = spaces.virtual[correlated.virtual]
            ctx = ClassContext(blocks=blocks, provider=provider,
                               eps_inactive=canonical.eps_inactive[correlated.inactive],
                               eps_virtual=canonical.eps_virtual[correlated.virtual],
                               fock_vi=_fock_block(ints, virt, core),
                               fock_va=_fock_block(ints, virt, spaces.active),
                               fock_ai=_fock_block(ints, spaces.active, core),
                               shift=float(shift), imaginary_shift=bool(imaginary_shift),
                               norm_cutoff=float(norm_cutoff))
            try:
                per_state.append(_evaluate_classes(names, ctx, state, shared))
            finally:
                # The cached ladder-string vector sets are the largest per-state arrays in
                # the run; a state's are useless to the next one.
                provider.release()
    finally:
        if blocks is not None:
            blocks.release()
        engine.release()

    result = _assemble(per_state, names, e_casscf, canonical, fock, shift, imaginary_shift,
                       correlated)
    if report:
        _report(result, names)
    return result


#: What the property dump's header must say when its diagonal has been NEVPT2 corrected. ⚠ A file
#: whose energies and whose wavefunctions come from different levels of theory has to say so:
#: it outlives the session, and nothing in it would otherwise reveal that ``mu`` belongs to the
#: CASSCF states while ``H`` does not.
DUMP_PROTOCOL = ("diagonal energies are CASSCF + SC-NEVPT2 (strongly contracted, Dyall H0); "
                 "the states, and therefore every moment matrix element, are CASSCF")


def corrected_property_matrices(matrices, result: NEVPT2Result):
    """A copy of a :class:`~kuiva.props.dump.PropertyMatrices` with ``E2`` on its diagonal.

    ⚠ **On request, never by default**. the property-dump file is a contract with an external
    crystal-field code, and substituting
    corrected energies makes it a *hybrid*: `H` from second-order perturbation theory, `mu`
    from the CASSCF states it was built on. That is a legitimate and common protocol — it is
    what a two-step treatment does — but it is a choice, and the header has to carry it.

    This function exists so that the substitution and the record cannot be separated. Doing it
    by hand is one line and leaves a file that lies by omission.

    ⚠ **The dependency still runs one way.** ``kuiva.pt`` reaches into ``kuiva.props``; nothing
    in ``props/`` learns that ``pt/`` exists (asserted from the sources).
    """
    from dataclasses import replace

    energies = np.asarray(matrices.energies, dtype=float)
    if energies.size != result.e2.size:
        raise ValueError("the dump carries {} states and this correction {}; they must be the "
                         "same spectrum in the same order"
                         .format(energies.size, result.e2.size))
    if not result.complete:
        log.warning("the property dump is being given a PARTIAL E2 (missing %s); the file will "
                    "record that, but a partial correction is not comparable with another "
                    "program's NEVPT2", ", ".join(result.missing))
    provenance = dict(matrices.provenance)
    provenance["nevpt2"] = {
        "flavor": "SC-NEVPT2", "complete": bool(result.complete),
        "missing": list(result.missing), "fock": result.fock,
        "shift": result.shift, "imaginary_shift": bool(result.imaginary_shift),
        "n_frozen": int(result.n_frozen), "n_deleted": int(result.n_deleted),
    }
    return replace(matrices, energies=result.total_energies.copy(), provenance=provenance,
                   comments=tuple(matrices.comments) + (DUMP_PROTOCOL,))


def _fock_block(ints: CASIntegrals, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """A C-contiguous block of the **inactive** Fock, for the classes with a one-body part.

    ⚠ ``F^I``, never ``F^I + F^A``: the active mean field is carried by the two-body
    coefficients of those perturbers, and adding it here would double-count it.
    """
    return np.ascontiguousarray(ints.f_inactive[np.ix_(rows, cols)])


def _require_energies(energies, n_states: int):
    """the state-averaging gate cannot run without them, and it is not optional above one state."""
    if energies is None:
        raise ValueError(
            "SC-NEVPT2 over {} states needs the state energies, so that degenerate blocks are "
            "weighted equally and a state count that splits one is refused. "
            "Pass energies=".format(n_states))
    return energies


def _evaluate_classes(names: Sequence[str], ctx: ClassContext, state: int,
                      shared: Optional[Dict[str, ClassResult]] = None
                      ) -> Dict[str, ClassResult]:
    results: Dict[str, ClassResult] = {}
    for name in names:
        spec = excitation_class(name)
        if spec.status == "planned":
            continue
        # ⚠ The declared-capability seam: a class asks the provider only for the
        # primitives it registered in `requires`, so a provider that does not serve one
        # of them (the network provider and the primed single-external classes) skips
        # the class here — visibly, and the assembly then refuses to call the sum E2.
        unsupported = [p for p in spec.requires if not hasattr(ctx.provider, p)]
        if unsupported:
            if state == 0:
                log.warning(
                    "class %s is SKIPPED: this contraction provider (%s) does not serve "
                    "%s. The sum below is a PARTIAL E2 and the result says so",
                    name, type(ctx.provider).__name__, ", ".join(unsupported))
            continue
        reuse = shared is not None and spec.state_independent
        if reuse and name in shared:
            # ⚠ Not an optimization to be clever about: `Sijrs` is the MP2-like bulk and the
            # most expensive class on any real system (n_c^2 n_v^2), and recomputing an
            # identical answer once per state is the difference between a state average being
            # affordable and not.
            results[name] = shared[name]
            continue
        with timer("NEVPT2 class {}".format(name)):
            results[name] = spec.evaluate(ctx)
        if reuse:
            shared[name] = results[name]
        entry = results[name]
        if entry.max_imaginary > 1e-8 * max(abs(entry.norm), 1.0):
            log.warning("state %d, class %s: a quantity that is real by construction carried "
                        "an imaginary part of %.3e. That is a conjugation error somewhere in "
                        "the contraction chain, not rounding",
                        state, name, entry.max_imaginary)
        _warn_on_intruder(state, name, entry, shifted=ctx.shift != 0.0)
        log.debug("state %d %s: norm %.6e, E %s, %d perturbers (%d dropped), min |dE| %s",
                  state, name, entry.norm,
                  "n/a" if entry.energy is None else "{:.10f}".format(entry.energy),
                  entry.n_perturbers, entry.n_dropped,
                  "n/a" if entry.min_denominator is None
                  else "{:.3e}".format(entry.min_denominator))
    return results


def _warn_on_intruder(state: int, name: str, entry, *, shifted: bool) -> None:
    """Compare a class's smallest denominator against the pre-registered band.

    ⚠ The whole point of this function is that :attr:`ClassResult.min_denominator` used to be
    computed, printed in the per-class table, and compared to **nothing** — so an intruder
    announced itself only to a reader who already knew what number to be alarmed by.

    A level shift already applied does not remove an intruder; it bounds the damage. The
    warning therefore still fires, and says so, rather than going quiet because the symptom
    was treated.
    """
    signed = entry.min_signed_denominator
    absolute = entry.min_denominator
    if absolute is None:
        return                                    # a class that formed no denominators
    treated = " (a level shift is applied, which bounds this rather than removing it)" \
        if shifted else ""
    # ⚠ The signed test first, and it is NOT "signed < severe": a denominator of +1e-9 is
    # essentially zero (the tier below), while a negative one is a qualitatively different
    # failure -- a perturber genuinely underneath the reference, which the absolute value
    # cannot see at all. Wording them alike would blur the two.
    if signed is not None and signed <= 0.0:
        log.warning(
            "state %d, class %s: the smallest energy denominator is %.3e Eh -- a perturber "
            "has fallen to or below the reference, so this class's E2 is divergent and wrong "
            "in sign, not merely ill-conditioned. Do not quote this correction: the reference "
            "is the thing to fix (the active space, or which states it is averaged over)%s",
            state, name, signed, treated)
    elif absolute < INTRUDER_SEVERE_EH:
        log.warning(
            "state %d, class %s: the smallest energy denominator is %.3e Eh, essentially "
            "zero, so this class's E2 is meaningless rather than merely suspect%s",
            state, name, absolute, treated)
    elif absolute < INTRUDER_WARN_EH:
        log.warning(
            "state %d, class %s: the smallest energy denominator is %.3e Eh, inside the "
            "%.2g Eh intruder band -- E2 for this class is sensitive to a perturber that "
            "contributes out of proportion to its physical weight. Check it against a run "
            "with imaginary_shift=, and against a larger active space%s",
            state, name, absolute, INTRUDER_WARN_EH, treated)


def _assemble(per_state, names, e_casscf, canonical, fock, shift,
              imaginary_shift, correlated) -> NEVPT2Result:
    n_states = len(per_state)
    energies: Dict[str, np.ndarray] = {}
    norms: Dict[str, np.ndarray] = {}
    # Every registered class appears in the tables, evaluated or not, so that a caller reading
    # the dictionaries always sees the whole partition and gets `nan` for what was skipped.
    for name in tuple(available_classes()) + tuple(n for n in names
                                                   if n not in available_classes()):
        energies[name] = np.full(n_states, np.nan)
        norms[name] = np.full(n_states, np.nan)
        for state, table in enumerate(per_state):
            entry = table.get(name)
            if entry is None:
                continue
            norms[name][state] = entry.norm
            if entry.energy is not None:
                energies[name][state] = entry.energy
    contributing = [n for n in energies if np.all(np.isfinite(energies[n]))]
    # ⚠ **A class the caller restricted away counts as missing too.** The eight are a partition
    # of the first-order interacting space, so a sum over a subset is a different quantity from
    # ``E2`` however cleanly each term converged, and `classes=` is the easiest way to produce
    # one by accident.
    missing = tuple(n for n in available_classes() if n not in contributing)
    e2 = (np.sum([energies[n] for n in contributing], axis=0) if contributing
          else np.zeros(n_states))
    if missing:
        log.warning("E2 is a PARTIAL sum: the class(es) %s produced no energy, so the total "
                    "below is a lower bound on the magnitude of the correction and is not "
                    "comparable with another program's NEVPT2 ",
                    ", ".join(missing))
    return NEVPT2Result(e_casscf=np.asarray(e_casscf, dtype=float), e2=np.asarray(e2),
                        class_energies=energies, class_norms=norms,
                        eps_inactive=None if canonical is None else canonical.eps_inactive,
                        eps_virtual=None if canonical is None else canonical.eps_virtual,
                        complete=not missing, missing=missing, fock=fock, shift=float(shift),
                        imaginary_shift=bool(imaginary_shift), per_state=per_state,
                        n_frozen=0 if correlated is None else correlated.n_frozen,
                        n_deleted=0 if correlated is None else correlated.n_deleted)


def _report(result: NEVPT2Result, names: Sequence[str]) -> None:
    """One per-class table per state, then the state summary.

    Deliberately shaped like PySCF's ``mrpt.NEVPT`` per-class print, so that the Tier-1 scalar
    comparison can be read line for line.
    """
    for state in range(result.e2.size):
        out.blank(log)
        out.note(log, "state {}".format(state))
        table = out.Table(log, [
            out.Column("class", "{}", 13, align="<"),
            out.Column("dN_act", "{:+d}", 8),
            out.col_sci("norm"),
            out.col_energy("E2 [Eh]"),
            out.col_count("labels", 12),
            out.col_sci("min |dE|"),
        ])
        table.start()
        for name in names:
            spec = excitation_class(name)
            entry = result.per_state[state].get(name)
            if entry is None:
                table.row(spec.label, spec.delta_n_act, "-", "not implemented", "-", "-")
                continue
            table.row(spec.label, spec.delta_n_act, entry.norm,
                      "-" if entry.energy is None else entry.energy,
                      entry.n_perturbers,
                      "-" if entry.min_denominator is None else entry.min_denominator)
        table.end()
        out.entries(log, [
            ("E(CASSCF)", float(result.e_casscf[state]), "Eh", "", out.E_FMT),
            ("E2" if result.complete else "E2 (PARTIAL)",
             float(result.e2[state]), "Eh", "", out.E_FMT),
            ("total" if result.complete else "total (PARTIAL)",
             float(result.total_energies[state]), "Eh", "", out.E_FMT),
        ])
    _report_multiplets(result)


def _report_multiplets(result: NEVPT2Result) -> None:
    """The degenerate-manifold table (C1), printed **beside** the per-state one, never instead.

    The correction is state specific through the per-state RDMs, and inside a degenerate
    manifold the CI's basis is arbitrary — so the member energies carry an arbitrary share of
    the manifold's internal spread while the barycentre does not. Both are printed: the
    barycentre because it is what means something, the spread because it is the only observable
    of the mechanism and averaging it away would hide it.
    """
    try:
        blocks = result.multiplets()
    except ValueError:
        return                                  # no reference spectrum; nothing to group
    if not any(b.size > 1 for b in blocks):
        return
    out.blank(log)
    out.note(log, "degenerate manifolds of the reference spectrum")
    table = out.Table(log, [
        out.Column("states", "{}", 10, align="<"),
        out.col_count("size", 6),
        out.col_energy("E2 (bary) [Eh]"),
        out.Column("dE [cm^-1]", out.CM_FMT, 16),
        out.col_sci("ref spread"),
        out.col_sci("spread [cm^-1]"),
    ])
    table.start()
    base = min(b.e_corrected for b in blocks)
    from ..props.multiplet import HARTREE_TO_CM
    for block in blocks:
        table.row("{}-{}".format(block.start, block.start + block.size - 1)
                  if block.size > 1 else str(block.start),
                  block.size, block.e2, (block.e_corrected - base) * HARTREE_TO_CM,
                  block.reference_spread_cm, block.corrected_spread_cm)
    table.end()
    split = [b for b in blocks if b.size > 1 and not b.degenerate]
    if split:
        worst = max(split, key=lambda b: b.corrected_spread_cm)
        log.warning("%d degenerate manifold(s) of the reference come out split by more than "
                    "%.2f cm^-1 after the correction, the worst by %.4f cm^-1 over %d states "
                    "(the reference itself carried %.2e cm^-1 there). At that size a splitting "
                    "already implies different physics, so the members are not one level and "
                    "the barycentre above is not a summary of them",
                    len(split), MULTIPLET_TOL_CM, worst.corrected_spread_cm, worst.size,
                    worst.reference_spread_cm)


__all__ = ["CanonicalOrbitals", "CorrelatedSpaces", "MultipletCorrection", "NEVPT2Result",
           "FOCK_MODES", "ACTIVE_INVARIANCE_TOL", "DEFAULT_MULTIPLET_TOL_CM",
           "MULTIPLET_TOL_CM", "DUMP_PROTOCOL", "corrected_property_matrices",
           "pseudo_canonicalize", "sc_nevpt2", "select_correlated"]
