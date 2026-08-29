"""SC-NEVPT2 on a tensor-network reference: the network-backed contraction provider.

The second implementation of the provider seam of :mod:`kuiva.pt.contractions`, exactly as
that module's docstring foresaw: the same primitives, served by network contractions
instead of determinant-space vectors, consumed by the same driver loop
(:func:`kuiva.pt.nevpt2._drive`) through the engine seam. Nothing in ``classes.py`` or the
loop changes; a class reads only the primitives it registered in ``requires``, and the
driver skips — loudly — a class whose primitives this provider does not serve.

What is served, and how
------------------------
* **Ranks 1–2 per state** — the production environments route with one-hot weights
  (:func:`kuiva.dmrg.density.state_rdms`).
* **The derived density kernels and the rank-2 Koopmans matrices** — the same pure
  functions the CI provider uses (:func:`~kuiva.pt.contractions.hole_rdm1`,
  :func:`~kuiva.pt.contractions.pair_matrix`, :func:`~kuiva.pt.contractions.hole_pair_matrix`,
  :func:`~kuiva.pt.contractions.koopmans_annihilation`, ``koopmans_creation``): one
  implementation of the algebra, two sources of densities.
* **The rank-3 Gram kernels** (the ``Srs (-2)``, ``Sij (+2)`` and ``Sir (0')``
  denominators) — Gram matrices of applied ladder-string states with ``H_act - E``
  between, contracted through the network (:func:`kuiva.dmrg.density.koopmans_gram`):
  the same Gram-of-explicit-vectors route the CI provider takes, with the shifted
  determinant space replaced by Jordan–Wigner strings applied to the tree. The overlap
  partners come from the density algebra, so the two constructions cross-check each
  other exactly as they do on the CI side (asserted in the tests, to rounding).

⚠ What is deliberately NOT served, and why (recorded scope)
------------------------------------------------------------
``annihilation_perturbers`` / ``creation_perturbers`` — the per-external-label perturber
vectors of the primed single-external classes ``Sr (-1')`` and ``Si (+1')``. On the CI
side these are formed as one *vector* per label in a shifted determinant space; on a
network no such vector can be formed cheaply, and every dense workaround dies at exactly
the sizes DMRG exists for: the Gram over the ``n_act^3`` ladder strings is an
``n_act^6`` object with ``O(n^6)`` tree contractions behind it, and a stored 4-RDM is
refused by the resource budget from ~12 active spinors (both rejected in
:mod:`kuiva.pt.contractions` and :mod:`kuiva.dmrg.density` for the same reasons). The
scalable route is a **per-label perturber network** — the label's operator compiled as a
TTNO, applied and variationally compressed — which is a separate piece of work, tracked
where gaps are tracked. Until it lands, an ``E2`` from this provider is **PARTIAL**
(six of eight classes), the driver warns per skipped class, and
``NEVPT2Result.complete`` is ``False`` — machinery that predates this module and exists
for exactly this.

⚠ What a truncated reference means here
-----------------------------------------
The CI provider's reference is an exact eigenvector of ``H_act``; a network reference is
converged to a cap and a tolerance. The Gram kernels are Hermitian by construction
regardless, but the rank-2 Koopmans matrices are Hermitian only *because* the reference
is an eigenvector — so their Hermiticity residual, checked as on the CI side, becomes a
diagnostic of the reference's own convergence, and the ``excitation_koopmans`` rows
involving the reference are only as zero as ``|H|psi> - E|psi>|`` is. At a saturating
cap all of it is exact and the provider reproduces the CI provider to rounding.

Dependency note: ``kuiva.pt`` already sits above the calculation path; this module
additionally imports ``kuiva.dmrg`` (lazily, inside the classes that need it, so
``import kuiva.pt`` stays free of the network layer).

References
----------
* C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu, J. Chem. Phys.
  114, 10252 (2001), doi:10.1063/1.1361246; C. Angeli, R. Cimiraglia, J.-P. Malrieu,
  J. Chem. Phys. 117, 9138 (2002), doi:10.1063/1.1515317 — SC-NEVPT2.
* S. Guo, M. A. Watson, W. Hu, Q. Sun, G. K.-L. Chan, J. Chem. Theory Comput. 12, 1583
  (2016), doi:10.1021/acs.jctc.6b00118 — DMRG-SC-NEVPT2, the precedent for serving the
  perturbation from a matrix-product reference (via stored higher densities there; via
  applied-string Grams here, the departure recorded above).
"""
from __future__ import annotations

from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..rdm.rdm import active_space_energy
from ..util.logging import get_logger
from ..util.timing import timer
from .contractions import (hole_pair_matrix, hole_rdm1, koopmans_annihilation,
                           koopmans_creation, pair_matrix)
from .nevpt2 import NEVPT2Result, _drive

log = get_logger(__name__)


class NetworkContractionProvider:
    """Network-backed contraction provider for **one** root of a converged network.

    Duck-typed against :class:`~kuiva.pt.contractions.CIContractionProvider`: the classes
    call only primitive methods, and the two providers are asserted equal on a reference
    both can represent exactly (the cross-implementation check this seam was designed
    for). See the module docstring for what is served and what is deliberately not.

    Parameters
    ----------
    template : :class:`~kuiva.dmrg.ttno.TTNOTemplate` on the state's topology.
    state : the converged :class:`~kuiva.dmrg.sweep.TTNState` (all roots).
    root : which root this provider describes.
    h_act, eri_act : the Dyall active Hamiltonian's one- and two-electron parts — the
        pair the reference was converged on (the canonicalization never rotates the
        active block, so they are the CASSCF's own active integrals).
    gamma, gamma2 : this root's 1- and 2-RDM, from
        :func:`kuiva.dmrg.density.state_rdms` — passed in so the engine computes each
        root's densities once for both ``H0`` and the provider.
    """

    def __init__(self, template, state, root: int, h_act: np.ndarray,
                 eri_act: np.ndarray, *, gamma: np.ndarray, gamma2: np.ndarray,
                 n_elec: int, check: bool = True) -> None:
        self.template = template
        self.state = state
        self.root = int(root)
        self.n_active = int(np.shape(h_act)[0])
        self.n_elec = int(n_elec)
        self.h_act = np.ascontiguousarray(h_act, dtype=np.complex128)
        self.eri_act = np.ascontiguousarray(eri_act, dtype=np.complex128)
        self._check = bool(check)
        self._gamma = np.ascontiguousarray(gamma)
        self._gamma2 = np.ascontiguousarray(gamma2)
        self._derived: Dict[str, object] = {}
        #: The root's active-space energy from its OWN densities — the ``E`` of every
        #: ``(H_act - E)`` kernel, exactly as on the CI side: the value that makes the
        #: kernels consistent with the densities they sit beside.
        self.e_active = active_space_energy(self.h_act, self.eri_act,
                                            self._gamma, self._gamma2)
        #: The Dyall active Hamiltonian as a TTNO on the state's topology, filled once
        #: and shared by every Gram kernel of this provider.
        self._ttno_h = template.fill(self.h_act, self.eri_act)

    # -- stored ranks ---------------------------------------------------------------------
    def rdm1(self) -> np.ndarray:
        return self._gamma

    def rdm2(self) -> np.ndarray:
        return self._gamma2

    def rdm3(self) -> np.ndarray:
        raise NotImplementedError(
            "no 3-RDM is built on the network path either: every rank-3 requirement of "
            "SC-NEVPT2 is served as a Gram matrix of applied ladder-string states with "
            "H_act between (koopmans_gram); see kuiva.pt.contractions for the shared "
            "reasoning")

    def contract_rdm4(self, active_eri: np.ndarray):
        raise NotImplementedError(
            "no rank-4 quantity is contracted on the network path: the primed "
            "single-external classes are not served by this provider (module docstring "
            "-- their scalable route is a per-label perturber network, not built), and "
            "a stored or contracted 4-RDM dies at exactly the sizes DMRG exists for")

    # -- derived primitives (one implementation of the algebra, shared with the CI) --------
    def hole_rdm1(self) -> np.ndarray:
        return self._cached("hole1", lambda: hole_rdm1(self._gamma))

    def pair_matrix(self) -> np.ndarray:
        return self._cached("pair", lambda: pair_matrix(self._gamma2))

    def hole_pair_matrix(self) -> np.ndarray:
        return self._cached("hole_pair", lambda: hole_pair_matrix(self._gamma,
                                                                  self._gamma2))

    def koopmans_annihilation(self) -> np.ndarray:
        return self._cached("koopmans_a", lambda: koopmans_annihilation(
            self.h_act, self.eri_act, self._gamma, self._gamma2, check=self._check))

    def koopmans_creation(self) -> np.ndarray:
        return self._cached("koopmans_c", lambda: koopmans_creation(
            self.h_act, self.eri_act, self._gamma, self._gamma2, check=self._check))

    # -- the Gram route, through the network ------------------------------------------------
    def pair_koopmans(self) -> np.ndarray:
        """``<psi| a+_t a+_u (H_act - E) a_w a_v |psi>`` as ``M[(t,u), (v,w)]``."""
        return self._cached("pair_koopmans", lambda: self._pair_gram()[1])

    def hole_pair_koopmans(self) -> np.ndarray:
        """``<psi| a_u a_t (H_act - E) a+_v a+_w |psi>`` as ``M[(t,u), (v,w)]``."""
        return self._cached("hole_pair_koopmans", lambda: self._hole_pair_gram()[1])

    def excitation_overlap(self) -> np.ndarray:
        """``S`` over the augmented set ``[|psi>, E_00|psi>, E_01|psi>, ...]``."""
        return self._cached("exc_overlap", lambda: self._excitation_gram()[0])

    def excitation_koopmans(self) -> np.ndarray:
        """``M`` over the same augmented set, with ``H_act - E`` between.

        ⚠ Its first row and column are zero only to the extent the reference is an
        eigenvector — for a truncated network state they measure the residual, which is
        the honest statement (module docstring).
        """
        return self._cached("exc_koopmans", lambda: self._excitation_gram()[1])

    def shifted_ndet(self, delta: int) -> int:
        """Determinant count of the ``n_elec + delta`` space; zero when it does not exist.

        Pure counting — no shifted space is ever built here.
        """
        target = self.n_elec + int(delta)
        if not 0 <= target <= self.n_active:
            return 0
        return comb(self.n_active, target)

    def release(self) -> None:
        """Drop the cached Gram kernels (the densities are small and stay)."""
        for key in ("pair_koopmans_pair", "hole_pair_grams", "exc_grams"):
            self._derived.pop(key, None)

    # -- construction ------------------------------------------------------------------------
    def _grams(self, key: str, terms) -> Tuple[np.ndarray, np.ndarray]:
        from ..dmrg.density import koopmans_gram
        value = self._derived.get(key)
        if value is None:
            with timer("NEVPT2 network Gram ({})".format(key)):
                value = koopmans_gram(self._ttno_h, self.state, self.root, terms,
                                      float(self.e_active))
            self._derived[key] = value
        return value

    def _pair_gram(self) -> Tuple[np.ndarray, np.ndarray]:
        from ..dmrg.ttno import fermion_term
        n = self.n_active
        # |D_tu> = a_u a_t |psi>, row t*n + u — the CI provider's flattening exactly.
        terms = [fermion_term(1.0, [(u, False), (t, False)])
                 for t in range(n) for u in range(n)]
        return self._grams("pair_koopmans_pair", terms)

    def _hole_pair_gram(self) -> Tuple[np.ndarray, np.ndarray]:
        from ..dmrg.ttno import fermion_term
        n = self.n_active
        # |C_tu> = a+_t a+_u |psi>, row t*n + u.
        terms = [fermion_term(1.0, [(t, True), (u, True)])
                 for t in range(n) for u in range(n)]
        return self._grams("hole_pair_grams", terms)

    def _excitation_gram(self) -> Tuple[np.ndarray, np.ndarray]:
        from ..dmrg.density import IDENTITY_TERM
        from ..dmrg.ttno import fermion_term
        n = self.n_active
        terms = [IDENTITY_TERM] + [fermion_term(1.0, [(t, True), (u, False)])
                                   for t in range(n) for u in range(n)]
        return self._grams("exc_grams", terms)

    def _cached(self, key: str, build):
        value = self._derived.get(key)
        if value is None:
            value = build()
            self._derived[key] = value
        return value

    def __repr__(self) -> str:
        return "NetworkContractionProvider(n_active={}, root={})".format(
            self.n_active, self.root)


class _NetworkEngine:
    """The tensor-network half of the driver — :class:`kuiva.pt.nevpt2._CIEngine`'s twin.

    Holds the converged network and serves per-root densities and providers. Densities
    are computed once per root and shared between ``H0`` and that root's provider.
    """

    kind = "tensor network (DMRG)"

    def __init__(self, template, state, n_elec: int, *, check: bool = True) -> None:
        self.template = template
        self.state = state
        self.n_states = int(state.n_roots)
        self.n_elec = int(n_elec)
        self._check = bool(check)
        self._pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def _root_rdms(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self._pairs is None:
            from ..dmrg.density import state_rdms
            with timer("NEVPT2 network per-root RDMs"):
                self._pairs = state_rdms(self.template, self.state)
        return self._pairs

    def averaged_gamma(self, equalized: np.ndarray) -> np.ndarray:
        # The equalized weights come from the driver's own state-averaging gate; the
        # per-root densities are already in hand for the providers, so the average is
        # their weighted sum rather than a second network pass.
        pairs = self._root_rdms()
        return sum(float(w) * gamma for w, (gamma, _) in zip(equalized, pairs))

    def state_gamma(self, state: int) -> np.ndarray:
        return self._root_rdms()[state][0]

    def provider(self, state: int, h_act: np.ndarray,
                 eri_act: np.ndarray) -> NetworkContractionProvider:
        gamma, gamma2 = self._root_rdms()[state]
        return NetworkContractionProvider(self.template, self.state, state,
                                          h_act, eri_act, gamma=gamma, gamma2=gamma2,
                                          n_elec=self.n_elec, check=self._check)

    def release(self) -> None:
        self._pairs = None


def sc_nevpt2_dmrg(factors, h_ao: np.ndarray, c_spinor: np.ndarray, spaces,
                   solver, n_elec: int, *, energies=None, weights=None,
                   e_nuc: float = 0.0, classes=None, fock: str = "state-averaged",
                   frozen_core=None, deleted_virtual=None, shift: float = 0.0,
                   imaginary_shift: bool = False,
                   norm_cutoff: Optional[float] = None,
                   degeneracy_tol: Optional[float] = None,
                   on_split: str = "raise", report: bool = True) -> NEVPT2Result:
    """SC-NEVPT2 on a converged DMRG-CASSCF — :func:`kuiva.pt.nevpt2.sc_nevpt2`'s
    network sibling, sharing its whole loop through the engine seam.

    Parameters differ in one place: instead of CI vectors it takes the **converged**
    :class:`~kuiva.dmrg.DMRGSolver` (after its finishing solve at the converged
    orbitals), whose stored state and last spectrum are the reference. ``energies``
    defaults to that spectrum (active-space eigenvalues, no ``e_core`` — the sweep
    convention, which is also what the CI route passes).

    ⚠ **The result is a PARTIAL E2 — six of the eight classes** — until the per-label
    perturber network lands (module docstring): the primed single-external classes are
    skipped with a warning, ``NEVPT2Result.complete`` is ``False``, and the report
    prints the total as PARTIAL. That is the driver's standing machinery, not a special
    case of this route.
    """
    from ..pt.classes import DEFAULT_NORM_CUTOFF
    from ..rdm.rdm import DEFAULT_DEGENERACY_TOL

    state = getattr(solver, "_state", None)
    last = getattr(solver, "last", None)
    if state is None or last is None:
        raise ValueError(
            "sc_nevpt2_dmrg needs a solver that has converged a state: run the "
            "DMRG-CASSCF (or one finishing solve at the converged orbitals) first")
    if energies is None:
        energies = [float(e) for e in last.energies]
    template = solver._template(solver.graph)
    engine = _NetworkEngine(template, state, int(n_elec))
    return _drive(engine, factors, h_ao, c_spinor, spaces, int(n_elec),
                  energies=energies, weights=weights, e_nuc=e_nuc, classes=classes,
                  fock=fock, frozen_core=frozen_core, deleted_virtual=deleted_virtual,
                  shift=shift, imaginary_shift=imaginary_shift,
                  norm_cutoff=DEFAULT_NORM_CUTOFF if norm_cutoff is None
                  else float(norm_cutoff),
                  degeneracy_tol=DEFAULT_DEGENERACY_TOL if degeneracy_tol is None
                  else float(degeneracy_tol),
                  on_split=on_split, report=report)


__all__ = ["NetworkContractionProvider", "sc_nevpt2_dmrg"]
