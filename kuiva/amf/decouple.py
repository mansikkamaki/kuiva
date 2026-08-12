"""X2C decoupling of the two-electron mean field, and the subtraction.

This module is the physics core of X2CAMF, and this docstring is the load-bearing part of it:
the accounting below is what the whole method reduces to, and getting the *subtracted* term
wrong produces an answer that is Hermitian, time-reversal even, of a plausible magnitude, and
wrong. Everything here is pure linear algebra on the arrays a
:class:`~kuiva.amf.backend.AtomicDiracSolution` carries — no integral library, no ``Mole``.

Primary reference: J. Liu, L. Cheng, J. Chem. Phys. **148**, 144108 (2018),
doi:10.1063/1.5023750 — "An atomic mean-field spin-orbit approach within exact two-component
theory for a non-perturbative treatment of spin-orbit coupling".

The accounting
--------------
**1. The one-electron X2C transformation, restated.** In a restricted kinetically balanced
basis the four-component one-electron problem is blocked ``LL/LS/SL/SS``. X2C finds the
decoupling matrix ``X`` relating small to large components (``C_S = X C_L``) and the
renormalization ``R``, and the two-component Hamiltonian is::

    h_X2C = R^dag ( h_LL + h_LS X + X^dag h_SL + X^dag h_SS X ) R

``X`` is obtained from the eigenvectors of a four-component eigenproblem, ``C_S = X C_L``, and
``R`` from ``R^dag S~ R = S`` with ``S~ = S_LL + X^dag S_SS X``. Both are built here, from the
blocks the backend returned, so that they are by construction the matrices belonging to *that*
problem: a mismatch between the two would be silent and fatal.

**2. The two-electron picture change, done atomically.** Exact two-component theory would
transform the two-electron operator with the same transformation, which is prohibitive.
X2CAMF applies it to the converged **four-component atomic mean field** ``G[D_4c]`` instead::

    G~ = R^dag ( G_LL + G_LS X + X^dag G_SL + X^dag G_SS X ) R

— the identical expression, on ``G`` instead of ``h``. This is where the two-electron
spin-orbit coupling comes from: ``G_SS`` and the off-diagonal blocks carry the spin-dependent
terms that the large-component Coulomb operator does not.

⚠ **Which ``X``: the converged Fock's, not the bare one-electron problem's.** The decoupling
here is defined by ``F = h + G[D]`` (:func:`x2c_decoupling`, ``source="fock"``), following the
reference implementation of Liu & Cheng 2018. The molecular Hamiltonian this correction is
added to carries the *one-electron* X2C Hamiltonian, so the mismatch is paid for explicitly
rather than ignored — see point 5. Kuiva used the one-electron ``X`` at first and changed on
measurement, not preference; the evidence, and what it does and does not move, is in
:func:`x2c_decoupling`.

**3. The subtraction, which is the whole ballgame.** Kuiva's molecular Hamiltonian
already carries the **untransformed** non-relativistic Coulomb operator — it is
``mol.intor("int2e")``, used unmodified by the CI. So what must be *added* to the one-electron
Hamiltonian is only the difference between what the picture change would have contributed and
what is already there::

    dG_AMF = G~ - G_nr[ D~ ]

with ``G_nr`` the ordinary non-relativistic ``J - K`` and ``D~`` the **two-component** density
that corresponds to the four-component one. Adding the whole of ``G~`` double-counts massively;
using the wrong density is wrong by an amount that looks physically plausible.

**4. Which density, and why it is not ``D_LL`` (the conjugation trap: coefficients and densities transform oppositely).**
The two-component X2C orbitals relate to the four-component large component by
``C_L = R C~``, because ``h_X2C = R^dag h_FW R`` and ``R^dag S~ R = S`` together turn the
two-component eigenproblem into the four-component one. Coefficients and densities transform
oppositely, so::

    D_LL = R D~ R^dag        ==>        D~ = R^-1 D_LL R^-dag

**not** ``R^dag D_LL R``. ``R`` is Hermitian positive definite but **not** unitary, so the two
differ substantially, and both are Hermitian, both have plausible traces, and both give a
correction of roughly the right size — and their spin-orbit splittings agree to 0.2%.

⚠ **The ``c -> inf`` test does not distinguish them**, contrary to what one would expect and to
what this module's docstring once claimed: every plausible variant (``R^-1 D R^-dag``,
``R^dag D R``, ``D_LL``, ``R D R^dag``) reduces to ``D_LL`` as ``R -> 1``, so all of them pass
it. That test settles the subtracted *operator*. What settles the *density* is the **X2CAMF
energy functional against four-component Dirac-Coulomb in the same basis**, which separates the
variants by five to six orders of magnitude (Ne: 3.4e-07 Eh for the correct one against
2.8e-02 to 5.7e-02 Eh for the others — worse than applying no correction at all).

(Equivalently, and this is what the reference implementation does: ``D~`` is the density of
the two-component orbitals obtained by diagonalizing the picture-changed Fock in the
large-component metric. X2C is constructed so that those orbitals are exactly ``R^-1 C_L``, so
the two routes are the same object; the closed form is used here because it needs no second
eigenproblem and no occupation bookkeeping.)

**5. The compensating one-electron term, which point 2's choice of ``X`` obliges.** With ``X``
taken from the Fock, ``h`` picture-changed with *this* decoupling is no longer the
one-electron X2C Hamiltonian the molecule carries. The difference is therefore added to the
correction::

    dG_AMF = G~ - G_nr[ D~ ]  +  ( h1e(X_2e) - h1e(X_1e) )

so that adding ``dG_AMF`` to the molecular one-electron X2C Hamiltonian gives the Hamiltonian
this convention actually implies, rather than a mixture of the two. Both bracketed terms use
the ``R`` belonging to their own ``X``. The term vanishes identically whenever the two
decouplings coincide — a vanishing mean field, a one-electron system, and the ``c -> inf``
limit — which is what leaves the exact-limit tests exact.

What the non-relativistic limit does settle
-------------------------------------------
As ``c -> inf``: ``X -> 0`` and ``R -> 1``, so ``G~ -> G_LL`` and ``D~ -> D_LL``; and the
four-component ``G_LL`` reduces to exactly the non-relativistic ``J - K`` over ``D_LL``, since
the ``(SS|LL)`` and ``(SS|SS)`` contributions are ``O(1/c^2)``. The two terms therefore cancel
**identically**, not approximately, and the point-5 compensation goes to zero with them. The
correction dies as exactly ``1/c^2``, which is the statement that the subtracted *operator* is
the right one — a test worth writing before the physics it checks, and the one that fails
loudest on a mis-scaled or mis-subtracted mean field. It is silent about the density (above).

What the correction contains — and it is not only spin-orbit coupling
---------------------------------------------------------------------
``dG_AMF`` contributes to **both** parts of the spin-free + spin-orbit decomposition: a spin-free
``delta h_sf`` and a spin-orbit ``delta w``. The two-electron *scalar* picture change is a real
effect that Breit-Pauli AMFI and SNSO screening factors do not capture at all, and it is
typically an order of magnitude larger than the spin-orbit part (measured on Ne:
``max |dh_sf| = 6.3e-3 Eh`` against ``max |dw| = 5.3e-4 Eh``). Both are reported separately
rather than summed.

References
----------
* X2CAMF: J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018), doi:10.1063/1.5023750;
  and the atomic mean-field idea it builds on, B. A. Hess, C. M. Marian, U. Wahlgren,
  O. Gropen, Chem. Phys. Lett. 251, 365 (1996), doi:10.1016/0009-2614(96)00119-4 (AMFI).
* Exact two-component decoupling: see :mod:`kuiva.x2c.decouple`, which implements it and
  carries the full reference list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..spinor.expand import decompose_two_component, time_reversal_residual
from ..util.logging import get_logger
# The decoupling itself is not an atomic-mean-field concept: it is shared with the molecular
# one-electron path, so it lives in kuiva.x2c and is re-exported here (see __all__) for the
# callers and tests that have always reached for it through this module.
from ..x2c.decouple import (FourComponentBlocks, decoupling_matrices, picture_change,
                            renormalization, two_component_density)
from ..x2c.mean_field import TIME_REVERSAL_LIMIT, mean_field_picture_change
from .backend import AtomicDiracSolution
from .configuration import AtomicConfiguration

log = get_logger(__name__)


# --- The X2C decoupling matrices ----------------------------------------------------------

def one_electron_integrals(solution: AtomicDiracSolution
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover ``(t, v, w, s)`` from a solution's four-component blocks.

    In the modified small-component normalization the backend fixes (see
    :mod:`kuiva.amf.backend`)::

        hcore.ll = v            hcore.ls = hcore.sl = t          hcore.ss = w/(4c^2) - t
        overlap.ll = s                                           overlap.ss = t/(2c^2)

    so all four one-electron matrices are already present and no integral library is needed.
    Deriving them from the *returned blocks* rather than recomputing them is the point: it
    makes "the same ``X`` and ``R`` as the one-electron Hamiltonian" structural rather than a
    convention two modules have to agree on.
    """
    c = solution.light_speed
    v = solution.hcore.ll
    t = solution.hcore.ls
    w = 4.0 * c * c * (solution.hcore.ss + t)
    s = solution.overlap.ll
    return t, v, w, s


#: The four-component operator whose positive-energy branch defines ``X``. See
#: :func:`x2c_decoupling`; ``"fock"`` is what the correction uses and ``"one-electron"`` is
#: what reproduces the molecular one-electron Hamiltonian.
DECOUPLING_SOURCES = ("fock", "one-electron")


def four_component_fock(solution: AtomicDiracSolution) -> FourComponentBlocks:
    """``F = h + G[D]``, the converged four-component Fock operator, block by block.

    ⚠ For an **open shell** this is the closed-shell-form Fock ``F_c``, not the shell-dependent
    effective operator the average-of-configuration SCF actually diagonalizes (the SCF's Fock and the stored ``veff`` are deliberately no longer the same object).
    That is the right choice and not a compromise — ``X`` describes the *decoupling* of the
    small component from the large one, which is a property of the operator's relativistic
    structure and not of how a degenerate manifold was averaged, and it is what the reference
    implementation uses. It also keeps ``X`` independent of the configuration-average coupling
    ``alpha``, which has no counterpart in the picture change.
    """
    return FourComponentBlocks(
        ll=solution.hcore.ll + solution.veff.ll, ls=solution.hcore.ls + solution.veff.ls,
        sl=solution.hcore.sl + solution.veff.sl, ss=solution.hcore.ss + solution.veff.ss)


def x2c_decoupling(solution: AtomicDiracSolution,
                   source: str = "fock") -> Tuple[np.ndarray, np.ndarray]:
    """The X2C decoupling matrix ``X`` and renormalization ``R`` for an atomic solution.

    ``X`` comes from the positive-energy eigenvectors of a four-component eigenproblem,
    ``C_S = X C_L``; ``R`` from ``R^dag S~ R = S`` with ``S~ = S_LL + X^dag S_SS X``. Both are
    ``(2*nao, 2*nao)`` in the spin-blocked spin-orbital basis.

    Parameters
    ----------
    source : {"fock", "one-electron"}
        Which four-component operator defines the decoupling.

        ``"fock"`` (the default, and what :func:`amf_atomic_correction` uses) takes the
        **converged four-component Fock** ``h + G[D]``, so the small component is decoupled in
        the presence of the very mean field being transformed. ``"one-electron"`` takes the
        bare ``h``, giving the same ``X`` the molecular one-electron X2C Hamiltonian is built
        with.

    ⚠ **The two are different conventions, not a right and a wrong one, and the choice is
    recorded rather than defaulted into.** Kuiva used ``"one-electron"`` at first, on
    the argument that "the same ``X`` as the one-electron Hamiltonian the molecule carries" is
    then structural rather than a convention two modules must agree on. Measurement against
    the reference implementation of Liu & Cheng 2018 (the ``x2camf`` plugin) settled it the other way:

    * the two differ by **exactly** this choice plus its compensating term — rebuilding
      Kuiva's correction the plugin's way reproduced its matrix to **2.0e-10 Eh on 3.0e-02**;
    * the Fock convention is **5-35x sharper on the energy functional** against four-component
      Dirac-Coulomb (Ne −1.1e-07 Eh against −4.0e-06; Ar −1.1e-06 against −6.8e-05), which is
      the one in-house check that discriminates the subtraction at all;
    * and it changes **no splitting** (0.011% on ``dw``, <= 0.42 cm^-1), so every committed
      four-component comparison stands either way.

    With the Fock convention the one-electron Hamiltonian no longer decouples with the same
    ``X`` as the two-electron mean field, and the difference is **not** dropped: it is added
    back as ``h1e(X_2e) - h1e(X_1e)``, exactly as in ``dhf_sph_pcc.cpp::x2c2ePCC``. See
    :func:`amf_atomic_correction`.

    The eigenproblem is solved in a canonically orthogonalized basis rather than by
    ``scipy.linalg.eigh(h, m)`` directly; that, and the linear-dependence threshold it
    applies, are :func:`kuiva.x2c.decouple.decoupling_matrices`'s business. **This function
    is only the choice of which four-component operator to hand it**, which is the one part
    of the decoupling that is specific to the atomic mean field.
    """
    if source not in DECOUPLING_SOURCES:
        raise ValueError("unknown decoupling source {!r}; expected one of {}"
                         .format(source, DECOUPLING_SOURCES))
    blocks = four_component_fock(solution) if source == "fock" else solution.hcore
    return decoupling_matrices(blocks, solution.overlap, solution.light_speed)


# --- The correction -----------------------------------------------------------------------

@dataclass(frozen=True)
class AtomicAMF:
    """The atomic mean-field correction for one element, in the spin-free + spin-orbit decomposition.

    ``h_sf`` and ``w`` are in the **target** (molecular) basis; the diagnostics are measured
    before contraction, in the basis the physics was done in.
    """

    h_sf: np.ndarray                  # (nao_target, nao_target) real symmetric
    w: np.ndarray                     # (3, nao_target, nao_target) real antisymmetric
    #: The reference configuration the mean field was taken over. Carried on the result and
    #: not only on the solution, because a correction that has been detached from the state it
    #: describes is provenance-free — and for an open shell that state is a genuine choice
    #:, not a formality.
    configuration: "AtomicConfiguration"
    #: ``max |dG|`` of the raw two-component correction, before decomposition [Eh].
    scale: float
    #: Absolute size of the time-reversal-**odd** part that was projected out [Eh], and its
    #: size relative to the *terms* of the subtraction rather than to their difference (see
    #: :func:`amf_atomic_correction` for why that distinction matters).
    tr_residual: float
    tr_residual_rel: float
    #: ``max |G~|`` and ``max |G_nr|``, the two terms of the subtraction [Eh]. Their ratio to
    #: ``scale`` says how much cancellation the correction rests on, which is the number that
    #: decides whether the result is meaningful at the precision claimed.
    transformed_scale: float
    subtracted_scale: float
    #: ``max |h1e(X_2e) - h1e(X_1e)|``, the compensating one-electron term the Fock-based
    #: decoupling brings with it [Eh]. Recorded separately because it is the *entire* visible
    #: consequence of the convention (:func:`x2c_decoupling`) and because it must go to zero in
    #: both exact limits — a non-vanishing value there would be the first symptom of the two
    #: decouplings having drifted apart.
    compensation_scale: float = 0.0

    @property
    def spin_free_scale(self) -> float:
        return float(np.max(np.abs(self.h_sf))) if self.h_sf.size else 0.0

    @property
    def spin_orbit_scale(self) -> float:
        return float(np.max(np.abs(self.w))) if self.w.size else 0.0

    @property
    def cancellation(self) -> float:
        """How many orders of magnitude of cancellation the subtraction involves.

        ``max(|G~|, |G_nr|) / |dG|``. A value of 1 means no cancellation; a value of 1e12
        would mean the correction is entirely rounding error. Measured on the default
        basis it sits at order 10, i.e. the correction is a genuine fraction of the terms it
        is built from.
        """
        biggest = max(self.transformed_scale, self.subtracted_scale)
        return biggest / self.scale if self.scale > 0.0 else float("inf")


def amf_atomic_correction(solution: AtomicDiracSolution,
                          coulomb_mean_field) -> AtomicAMF:
    """The atomic mean-field two-electron correction ``(delta h_sf, delta w)`` for one element.

    Parameters
    ----------
    solution : AtomicDiracSolution
        A converged four-component atomic calculation.
    coulomb_mean_field : callable
        ``dm -> J - K``: the **non-relativistic** two-electron mean field of a two-component
        density, over the solver's basis. Supplied by the backend (see
        :meth:`kuiva.amf.backend.AtomicDiracBackend.coulomb_mean_field`) because it needs the
        two-electron integrals of exactly that basis.

    Returns
    -------
    AtomicAMF in the solution's **target** basis (contracted, if the solve was uncontracted).
    """
    if not solution.converged:
        log.warning("building an atomic mean-field correction for %s from a four-component "
                    "SCF that did not converge. The correction is returned so the failure is "
                    "visible downstream, but nothing built on it is trustworthy.",
                    solution.element)

    # ⚠ The formula lives in :mod:`kuiva.x2c.mean_field`, not here, because X2C-mmf is
    # the identical subtraction on a *molecular* four-component solve. Two copies of it would
    # be the most dangerous duplication in the project: a wrong subtracted term is Hermitian,
    # time-reversal even, of plausible magnitude, and wrong. What remains atomic — and what
    # this function is — is the choice of *what was solved* and the contraction back.
    change = mean_field_picture_change(
        solution.hcore, solution.overlap, solution.veff, solution.density.ll,
        solution.light_speed, coulomb_mean_field, label=solution.element)

    dg_target = solution.contract(change.dg)
    h_sf, w = decompose_two_component(dg_target)
    return AtomicAMF(
        h_sf=h_sf, w=w, configuration=solution.configuration,
        scale=change.scale,
        tr_residual=change.tr_residual, tr_residual_rel=change.tr_residual_rel,
        transformed_scale=change.transformed_scale,
        subtracted_scale=change.subtracted_scale,
        compensation_scale=change.compensation_scale)


def x2c_one_electron(solution: AtomicDiracSolution) -> np.ndarray:
    """The one-electron X2C Hamiltonian of the atomic problem, in the target basis.

    Not used by the correction — the molecular one-electron Hamiltonian comes from the
    front-end — but it is what makes the decoupling consistency requirement testable:
    the ``X`` and ``R`` built here must reproduce, on the one-electron Hamiltonian, the same
    matrix PySCF's ``SpinOrbitalX2CHelper`` produces for the same atom. If they do not, the
    two-electron picture change is being done with the wrong decoupling and nothing downstream
    would notice.

    ⚠ Explicitly ``source="one-electron"``: this function's whole purpose is to reproduce the
    *molecular* one-electron Hamiltonian, which is built with the one-electron ``X``. The
    correction's own decoupling is the Fock one (:func:`x2c_decoupling`), and the difference
    between the two is carried by the compensating term inside
    :func:`amf_atomic_correction` — not by silently using one where the other is meant.
    """
    x, r = x2c_decoupling(solution, source="one-electron")
    return solution.contract(picture_change(solution.hcore, x, r))


# ``picture_change``, ``two_component_density`` and ``renormalization`` are re-exports from
# :mod:`kuiva.x2c.decouple`, kept here because they have always been reachable through this
# module. They are defined there, not here — this module must never grow a second definition.
__all__ = ["AtomicAMF", "TIME_REVERSAL_LIMIT", "amf_atomic_correction", "x2c_decoupling",
           "picture_change", "two_component_density", "renormalization",
           "one_electron_integrals", "x2c_one_electron"]
