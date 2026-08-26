"""Active-space contraction primitives for NEVPT2.

**Orchestration plus small dense algebra, not a registered kernel.** Everything here is
``n_active^k`` work on arrays that are small by construction — the conventional-CI ceiling is
20 spinors — and the one expensive piece, the ``F`` intermediate the RDMs are built
from, belongs to :mod:`kuiva.ci.sigma` and is *borrowed*, never duplicated.

The seam
--------
:class:`ContractionProvider` is what a perturbation class is allowed to ask the active space
for. It serves **primitives** — density matrices, and the overlap/Koopmans pairs below — never
an SC-specific assembled quantity, because the same primitives have to build a future FIC
metric and per-class Hamiltonian (the FIC-ready rule (primitives, never SC-assembled quantities)). Indeed each pair *is* that metric and
that Hamiltonian; strong contraction is the special case of taking one vector out of the
space. Two implementations are foreseen and only the first exists:

* :class:`CIContractionProvider` — conventional CI.
* a network-backed provider — :mod:`kuiva.dmrg.density` already produces ranks 1-4 by direct
  contraction in this module's convention and already uses the *same* Gram-of-annihilated
  states route as the second half of this module, so it is a second implementation of this
  protocol and not a new seam. It is not this plan's work.

⚠ **No 3-RDM is built, and that is a departure from what the method is usually written
with** (a recorded decision). The original plan was "stored ranks 1-3, rank 4 only matrix-free". The derivation showed that every
rank-3 requirement of SC-NEVPT2 has the form

::

    K[(t,u),(v,w)] = <Psi| O+_tu (H_act - E) O_vw |Psi>

for a one- or two-body ladder string ``O``, i.e. a **Gram matrix of explicitly constructed
vectors** with ``H_act`` between them. Building it that way instead of contracting a stored
``Gamma^3``: costs ``n_act^2`` applications of the active Hamiltonian (the same order as the
``n_act^6`` contraction it replaces), removes the ``n_act^6`` residency from the CI path
entirely, and — the reason it is the right call rather than merely a cheaper one — replaces a
five-term ``delta`` expansion, every term of which is invisible to a Hermiticity or trace
check, with two matrix products. It is also the same route the network side already takes for
its high-rank densities. ``rdm3`` therefore raises, and says this.

The rank-4 requirement, and why it is *not* a bigger Gram matrix
-----------------------------------------------------------------
⚠ **The primed single-external classes are contracted BEFORE the Gram, not after it, and that
is the second such departure** (the planning-level guess was the other route
and said so). Their perturber is ``O_L|Psi>`` for a *three*-operator ladder string, so the
naive extension of the paragraph above is a Gram matrix over the ``n_act^3`` strings — an
``n_act^6`` array again, built at the cost of ``n_act^3`` applications of ``H_act``. That is
the "contracted 4-RDM" of the literature and it is the wrong trade here, because **strong
contraction only ever needs one vector per external label**:

::

    |Psi_a> = a+_a [ sum_t f^I_at a_t + sum_tuv (at|uv) a+_u a_v a_t ] |Psi>      (Sr)

so the perturber is a *linear combination* of the ``n_act^3`` strings with coefficients the
caller already holds. Forming the combination first
(:meth:`CIContractionProvider.annihilation_perturbers`) costs one ``H_act`` application per
**external label** and stores nothing of size ``n_act^6``:

======================================  ==========================  =====================
route                                   ``H_act`` applications      stored
======================================  ==========================  =====================
Gram over the ``n_act^3`` strings       ``n_act^3``                 ``n_act^6``
one vector per external label           ``n_ext``                   ``n_ext * ndet``
======================================  ==========================  =====================

⚠ **The crossover is ``n_ext`` against ``n_act^3``, and it is not always on this side.** With
20 active spinors the string route costs 8000 sigma applications and a 1 GB matrix, so a
virtual space of any realistic size wins; with a *small* active space and a large basis it
would not, and the honest statement is that the label route is chosen because it also removes
the ``n_act^6`` residency — the thing the resource pre-flight was worried about — not because it is
cheaper in every regime. ⚠ It is **FIC-ready in the same way the rest of this module is**: the
primitive takes an arbitrary coefficient matrix and ``full=True`` returns the whole
``(n_rows, n_rows)`` Gram pair, which with the identity as coefficients *is* the FIC metric and
per-class Hamiltonian over the string basis. SC and FIC are two callers of one primitive.

⚠ **Everything is per state.** A Koopmans matrix is built from one state's density or vectors
and is only Hermitian because that state is an eigenvector of the active Hamiltonian (see
:func:`koopmans_annihilation`). Handing a state-averaged density to these functions produces a
plausible non-Hermitian matrix, so the driver builds one provider per state.

Conventions, all inherited and none re-defined here
----------------------------------------------------
``gamma_pq = <a+_p a_q>`` and ``Gamma_pqrs = <a+_p a+_r a_s a_q>`` — :mod:`kuiva.rdm.rdm`'s,
which is also :mod:`kuiva.dmrg.density`'s. Chemists' notation ``(pq|rs)`` with **4-fold**
symmetry only: ``(pq|rs) = (rs|pq) = (qp|sr)*``. Nothing here may assume more.

References
----------
* C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu, J. Chem. Phys. 114,
  10252 (2001), doi:10.1063/1.1361246; C. Angeli, R. Cimiraglia, J.-P. Malrieu, J. Chem. Phys.
  117, 9138 (2002), doi:10.1063/1.1515317 — NEVPT2 and the contracted perturber norms.
* K. G. Dyall, J. Chem. Phys. 102, 4909 (1995), doi:10.1063/1.469539 — the zeroth-order
  Hamiltonian whose active part these commutators are taken with respect to.
* Operator algebra and the ``E_pq`` identities: T. Helgaker, P. Jorgensen, J. Olsen,
  "Molecular Electronic-Structure Theory", Wiley (2000), ch. 1-2.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..ci.sigma import SigmaOperator
from ..ci.strings import CASSpace, apply_ladder
from ..rdm.rdm import RDMBuilder, active_space_energy
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

#: Relative tolerance on the Hermiticity of a Koopmans matrix before it is warned about. It is
#: **not** a rounding tolerance on an algebraic identity: Hermiticity here is a *consequence*
#: of the CI vector being an exact eigenvector of the active Hamiltonian (see
#: :func:`koopmans_annihilation`), so a violation says the RDMs and the active integrals do
#: not belong to each other. Sized like :func:`kuiva.rdm.rdm.active_space_energy`'s closure
#: check, which fails on exactly the same inconsistency.
KOOPMANS_HERMITICITY_RTOL = 1.0e-8


# --- pure density algebra ------------------------------------------------------------------

def hole_rdm1(gamma: np.ndarray) -> np.ndarray:
    """``<a_t a+_u> = delta_tu - gamma_ut``, as a matrix indexed ``[t, u]``.

    The norm kernel of the ``Sijr (+1)`` class. Hermitian, and positive semidefinite for a
    physical ``gamma`` (its eigenvalues are the hole occupations ``1 - n``).
    """
    gamma = np.asarray(gamma)
    return np.eye(gamma.shape[0], dtype=np.complex128) - gamma.conj()


def pair_matrix(gamma2: np.ndarray) -> np.ndarray:
    """``<a+_t a+_u a_w a_v>`` as the ``(n^2, n^2)`` matrix ``M[(t,u), (v,w)]``.

    The norm kernel of the ``Srs (-2)`` class: with the perturber
    ``a+_a a+_b sum_tu (at|bu) a_u a_t |Psi>`` the norm is a plain quadratic form in the
    integral pair ``(t,u)`` against this matrix. In the ``Gamma_pqrs = <a+_p a+_r a_s a_q>``
    convention it is ``Gamma[t, v, u, w]`` — ⚠ the middle pair of indices is interleaved, not
    blocked, and writing ``gamma2.reshape(n**2, n**2)`` instead gives a Hermitian,
    correctly-traced, wrong matrix.
    """
    return np.ascontiguousarray(
        np.einsum("tvuw->tuvw", np.asarray(gamma2)).reshape(gamma2.shape[0] ** 2, -1))


def hole_pair_matrix(gamma: np.ndarray, gamma2: np.ndarray) -> np.ndarray:
    """``<a_u a_t a+_v a+_w>`` as the ``(n^2, n^2)`` matrix ``M[(t,u), (v,w)]``.

    The norm kernel of the ``Sij (+2)`` class. Obtained by anticommuting the four operators
    into normal order,

    ::

        <a_u a_t a+_v a+_w> =  d_tv d_uw - d_uv d_tw
                             - d_tv gamma_wu + d_uv gamma_wt
                             + d_tw gamma_vu - d_uw gamma_vt
                             + Gamma_{v t w u}

    ⚠ Six ``delta`` terms with three sign patterns, and every one of them is invisible to a
    Hermiticity or trace check on the result. The test that catches an error here builds the
    same object by brute force in a tiny Fock space (``tests/fockspace.py``), which shares no
    code with this expression.
    """
    gamma = np.asarray(gamma)
    gamma2 = np.asarray(gamma2)
    n = gamma.shape[0]
    eye = np.eye(n, dtype=np.complex128)
    m = (np.einsum("tv,uw->tuvw", eye, eye)
         - np.einsum("uv,tw->tuvw", eye, eye)
         - np.einsum("tv,wu->tuvw", eye, gamma)
         + np.einsum("uv,wt->tuvw", eye, gamma)
         + np.einsum("tw,vu->tuvw", eye, gamma)
         - np.einsum("uw,vt->tuvw", eye, gamma)
         + np.einsum("vtwu->tuvw", gamma2))
    return np.ascontiguousarray(m.reshape(n * n, n * n))


# --- Koopmans matrices (rank <= 2) ----------------------------------------------------------

def koopmans_annihilation(h_act: np.ndarray, eri_act: np.ndarray, gamma: np.ndarray,
                          gamma2: np.ndarray, *, check: bool = True) -> np.ndarray:
    """``K_tu = <Psi| a+_t (H_act - E) a_u |Psi>``, indexed ``[t, u]``.

    The active-space part of the energy denominator of every class whose perturber removes one
    active electron (``Srsi (-1)``). Since ``H_act|Psi> = E|Psi>``,

    ::

        K_tu = <a+_t [H_act, a_u]>,   [H_act, a_u] = - sum_q f_uq a_q
                                                     - sum_qrs (uq|rs) a+_r a_s a_q

    so ``K_tu = - sum_q f_uq gamma_tq - sum_qrs (uq|rs) Gamma_tqrs`` — **rank 2 only**, which
    is what puts this class in the first implementation stage rather than the third.

    ⚠ ``K`` is Hermitian, but **not manifestly so**: the expression above is Hermitian only
    because ``gamma`` and ``Gamma`` come from an exact eigenvector of the very ``(f, eri)``
    pair passed in. That makes the check worth running — it fails on a truncated CI vector, on
    a state-averaged density used where a state-specific one belongs, and on any transposition
    in either argument. The matrix is symmetrized after the check, because the residue is
    rounding once the check has passed.

    Parameters
    ----------
    h_act : ``(n, n)``
        The **inactive Fock restricted to the active space** — ``CASIntegrals``'s
        ``h_active_effective()``, i.e. the one-electron operator of the Dyall active
        Hamiltonian. Not the bare ``h``.
    eri_act : ``(n, n, n, n)``
        Active ``(tu|vw)``, chemists' notation.
    """
    h_act = np.asarray(h_act)
    eri_act = np.asarray(eri_act)
    gamma = np.asarray(gamma)
    gamma2 = np.asarray(gamma2)
    n = gamma.shape[0]
    # - sum_q f_uq gamma_tq  ->  - gamma @ f^T
    k = -(gamma @ h_act.T)
    # - sum_qrs (uq|rs) Gamma_tqrs: a GEMM over the fused (q, r, s) index, never an einsum on
    # a hot path. (uq|rs) is (n, n^3) with u leading; Gamma_tqrs likewise with t.
    k -= (gamma2.reshape(n, n ** 3) @ eri_act.reshape(n, n ** 3).T)
    return _hermitize(k, "annihilation Koopmans matrix K", check)


def koopmans_creation(h_act: np.ndarray, eri_act: np.ndarray, gamma: np.ndarray,
                      gamma2: np.ndarray, *, check: bool = True) -> np.ndarray:
    """``K'_tu = <Psi| a_t (H_act - E) a+_u |Psi>``, indexed ``[t, u]``.

    The counterpart of :func:`koopmans_annihilation` for the classes whose perturber *adds* an
    active electron (``Sijr (+1)``). With ``[H_act, a+_u] = sum_p f_pu a+_p +
    sum_pqr (pu|qr) a+_p a+_q a_r``,

    ::

        K'_tu = sum_p f_pu (d_tp - gamma_pt)
              + sum_pqr (pu|qr) [ d_tp gamma_qr - d_tq gamma_pr + Gamma_{p r q t} ]

    Rank 2 only, and Hermitian for the same non-manifest reason as ``K``.
    """
    h_act = np.asarray(h_act)
    eri_act = np.asarray(eri_act)
    gamma = np.asarray(gamma)
    gamma2 = np.asarray(gamma2)
    n = gamma.shape[0]
    #   sum_p f_pu d_tp  -  sum_p f_pu gamma_pt
    kp = np.ascontiguousarray(h_act - gamma.T @ h_act, dtype=np.complex128)
    #  + sum_qr (tu|qr) gamma_qr        (the d_tp term of the two-electron part)
    kp += (eri_act.reshape(n * n, n * n) @ gamma.reshape(n * n)).reshape(n, n)
    #  - sum_pr (pu|tr) gamma_pr        (the d_tq term)
    kp -= np.einsum("putr,pr->tu", eri_act, gamma, optimize=True)
    #  + sum_pqr (pu|qr) Gamma_{p r q t}
    kp += np.einsum("puqr,prqt->tu", eri_act, gamma2, optimize=True)
    return _hermitize(kp, "creation Koopmans matrix K'", check)


# --- the Gram route: vectors in a shifted electron-number space ------------------------------

def ladder_vector_gb(n_active: int, ndet: int, n_vectors: Optional[int] = None) -> float:
    """[GB] of a set of ``n_active**2`` (by default) vectors over ``ndet`` determinants."""
    count = n_active ** 2 if n_vectors is None else int(n_vectors)
    return res.array_gb((count, int(ndet)), np.complex128)


def perturber_vector_gb(ndet: int, n_labels: int = 1) -> float:
    """[GB] of ``n_labels`` single-external perturber vectors over ``ndet`` determinants.

    The per-label cost the primed classes batch against (kernel rule B7): one complex
    vector in the shifted electron-number space per external label, plus the same again for
    ``H_act`` applied to it.
    """
    return 2.0 * res.array_gb((int(n_labels), int(ndet)), np.complex128)


class ShiftedSpace:
    """The active space at ``n_elec + delta`` electrons, and ``H_act``'s action on it.

    The perturbers of the ``(+-2)`` and ``(0')`` classes are ladder strings applied to the
    reference, so their overlap and Koopmans kernels are Gram matrices over a determinant space
    with a *different* electron count. This owns that space and the sigma operator on it.

    ⚠ **Three edge cases, all reachable from ordinary small active spaces**, and each is a
    silent wrong answer if it is met by an exception handler instead of by arithmetic:

    * ``n_elec + delta`` outside ``[0, n_spinor]`` — the ladder string annihilates the
      reference, every vector is zero, and the class has no perturbers. Reported as
      :attr:`empty`; the kernels come back as zeros of the right shape.
    * ``n_elec + delta == 0`` — the space is the **vacuum**, one determinant, and ``H_act`` is
      identically zero on it. :class:`~kuiva.ci.strings.CASSpace` cannot represent it (it
      requires at least one electron), so the masks are built directly. Reached by any
      two-electron active space, ``CAS(2, 2)`` included, which is why it is handled rather
      than refused.
    * ``n_elec + delta == n_spinor`` — one determinant again, and ``CASSpace`` *does* represent
      it; the sigma operator on a one-determinant space returns its diagonal element.
    """

    def __init__(self, n_spinor: int, n_elec: int, h_act: np.ndarray, eri_act: np.ndarray,
                 *, label: str = "", family: Optional["ShiftedSpaces"] = None) -> None:
        self.n_spinor = int(n_spinor)
        self.n_elec = int(n_elec)
        self.label = label
        #: The :class:`ShiftedSpaces` this space belongs to, if any — consulted before a
        #: sigma operator is built, so the family holds at most one workspace at a time.
        self._family = family
        self.h_act = h_act
        self.eri_act = eri_act
        self.empty = not (0 <= self.n_elec <= self.n_spinor)
        self._sigma: Optional[SigmaOperator] = None
        self.space: Optional[CASSpace] = None
        if self.empty:
            self.masks = np.zeros(0, dtype=np.uint64)
            self.ndet = 0
        elif self.n_elec == 0:
            self.masks = np.zeros(1, dtype=np.uint64)
            self.ndet = 1
        else:
            self.space = CASSpace(self.n_spinor, self.n_elec, build_map=False)
            self.masks = self.space.masks
            self.ndet = int(self.space.ndet)

    def apply_h(self, vectors: np.ndarray) -> np.ndarray:
        """``H_act |v>`` for every row of ``vectors``; zero on the vacuum."""
        vectors = np.atleast_2d(np.ascontiguousarray(vectors, dtype=np.complex128))
        if self.empty or self.n_elec == 0:
            return np.zeros_like(vectors)
        if self._sigma is None:
            # ⚠ One live workspace per family: the sibling spaces' operators are dropped —
            # and with them their ledger reservations, which die with the buffers — before
            # this one allocates, so the family's peak is the *largest* shifted workspace,
            # never the sum over the five shifts (measured at 4.6 GB against 1.15 at a
            # 20-spinor half-filled active space). What survives the drop is what is worth
            # keeping: the CASSpace with its excitation map; a rebuilt operator is one
            # buffer allocation and two small integral reshapes.
            if self._family is not None:
                self._family.make_room(self)
            # check_symmetry is off: the same integrals were validated when the reference CI
            # was built, and re-checking an n^4 array once per shifted space is pure cost.
            self._sigma = SigmaOperator(self.space, self.h_act, self.eri_act,
                                        check_symmetry=False)
        out = np.empty_like(vectors)
        with timer("NEVPT2 H_act on the {} space".format(self.label or "shifted")):
            for row in range(vectors.shape[0]):
                self._sigma(np.ascontiguousarray(vectors[row]), out=out[row])
        return out

    def gram(self, vectors: np.ndarray, energy: float) -> Tuple[np.ndarray, np.ndarray]:
        """``(S, K)`` with ``S_ij = <v_i|v_j>`` and ``K_ij = <v_i|(H_act - E)|v_j>``.

        Both are Hermitian by construction here — unlike the RDM-derived Koopmans matrices of
        :func:`koopmans_annihilation`, whose Hermiticity is a *consequence* of the reference
        being an eigenvector. That is not a reason to skip the check: it is a reason the check
        moved to the *cross-comparison* between the two routes, which is what the tests do.
        """
        vectors = np.atleast_2d(np.ascontiguousarray(vectors, dtype=np.complex128))
        if self.empty or vectors.size == 0:
            n = vectors.shape[0]
            return (np.zeros((n, n), dtype=np.complex128),
                    np.zeros((n, n), dtype=np.complex128))
        applied = self.apply_h(vectors) - float(energy) * vectors
        overlap = vectors.conj() @ vectors.T
        koopmans = vectors.conj() @ applied.T
        return (0.5 * (overlap + overlap.conj().T),
                0.5 * (koopmans + koopmans.conj().T))

    def release(self) -> None:
        """Drop the sigma operator — and, through its buffer, its ledger reservation.

        The :class:`~kuiva.ci.strings.CASSpace` (masks, excitation map) is kept: it is what
        cost something to build and it is small; the workspace is the large part and is
        rebuilt in one allocation when this space is next applied.
        """
        self._sigma = None

    def __repr__(self) -> str:
        return "ShiftedSpace({} electrons, {} determinants{})".format(
            self.n_elec, self.ndet, ", empty" if self.empty else "")


class ShiftedSpaces:
    """The ``0, +-1, +-2`` electron spaces of one active space, built once and **shared**.

    **Shared across states** because there is nothing state-specific in them: the
    pseudo-canonicalization never rotates the active block, so ``h_act`` and ``eri_act`` do
    not depend on the state or on which Fock built them, and the determinant machinery (the
    masks and the excitation maps, which are what cost something to build) serves every
    state alike.

    ⚠ **At most one sigma workspace is live across the whole family** (:meth:`make_room`,
    consulted by every member before it builds one). The workspaces are the family's only
    large arrays — ``C(n, k+delta) * n^2`` complex each — and their reservations die with
    the operators' buffers, so dropping a sibling really returns its memory to the ledger.
    Holding all five at once was measured at 4.6 GB against the largest single one's 1.15 at
    a 20-spinor half-filled active space, for operators the class loop only ever applies one
    at a time; the price of the policy is one buffer allocation and two small reshapes per
    shift *switch*, of which a state's evaluation has about a dozen — class boundaries, not
    inner loops, because each cached vector set and each class works a single shift.
    """

    def __init__(self, n_spinor: int, n_elec: int, h_act: np.ndarray,
                 eri_act: np.ndarray) -> None:
        self.n_spinor = int(n_spinor)
        self.n_elec = int(n_elec)
        self.h_act = h_act
        self.eri_act = eri_act
        self._spaces = {}
        #: Sigma-operator builds across the family's lifetime — the observable the one-live
        #: policy is judged by: it must stay near (shifts touched) x (state count), and a
        #: test pins that it does not explode into the inner loops.
        self.n_sigma_builds = 0

    def make_room(self, keep: "ShiftedSpace") -> None:
        """Drop every sibling's sigma operator before ``keep`` builds its own."""
        for space in self._spaces.values():
            if space is not keep:
                space.release()
        self.n_sigma_builds += 1

    def get(self, delta: int) -> ShiftedSpace:
        space = self._spaces.get(int(delta))
        if space is None:
            space = ShiftedSpace(self.n_spinor, self.n_elec + int(delta),
                                 self.h_act, self.eri_act,
                                 label="N{:+d}".format(delta) if delta else "N",
                                 family=self)
            self._spaces[int(delta)] = space
        return space

    def release(self) -> None:
        """Drop every space, operator and workspace; the ledger follows the buffers."""
        for space in self._spaces.values():
            space.release()
        self._spaces.clear()

    def __repr__(self) -> str:
        return "ShiftedSpaces(n_elec={}, built={})".format(
            self.n_elec, sorted(self._spaces))


def _hermitize(matrix: np.ndarray, what: str, check: bool) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
    if check:
        scale = max(float(np.max(np.abs(matrix))), 1.0)
        err = float(np.max(np.abs(matrix - matrix.conj().T)))
        if err > KOOPMANS_HERMITICITY_RTOL * scale:
            log.warning(
                "the %s is not Hermitian (max |K - K^dag| = %.3e, scale %.3e). It is Hermitian "
                "only when the density matrices come from an exact eigenvector of the active "
                "Hamiltonian they are contracted with, so this says the CI vector, the RDMs "
                "and the integrals do not belong to each other -- not that the algebra is "
                "rounding-limited", what, err, scale)
    return 0.5 * (matrix + matrix.conj().T)


# --- the provider ----------------------------------------------------------------------------

class CIContractionProvider:
    """Conventional-CI contraction provider for **one** state.

    Holds the state's 1- and 2-RDM and derives everything else on request, caching each
    derived object because a class loop asks for the same one repeatedly. Rank 3 and the
    matrix-free rank-4 contractions are the next stages and raise.

    Parameters
    ----------
    space : :class:`~kuiva.ci.strings.CASSpace`
    civec : ``(ndet,)`` complex, normalized — **one** state, not the ensemble.
    h_act, eri_act : the Dyall active Hamiltonian's one- and two-electron parts.
    builder : :class:`~kuiva.rdm.rdm.RDMBuilder`, optional
        Pass the driver's, so that the ``F`` intermediate is allocated once for the whole run
        rather than once per state (1.1 GB at 20 spinors).
    """

    def __init__(self, space: CASSpace, civec: np.ndarray, h_act: np.ndarray,
                 eri_act: np.ndarray, *, builder: Optional[RDMBuilder] = None,
                 spaces: Optional[ShiftedSpaces] = None, check: bool = True) -> None:
        self.space = space
        self.n_active = int(space.n_spinor)
        self.h_act = np.ascontiguousarray(h_act, dtype=np.complex128)
        self.eri_act = np.ascontiguousarray(eri_act, dtype=np.complex128)
        self._check = bool(check)
        #: The shifted electron-number spaces. Pass the driver's, shared across states, so
        #: the determinant machinery is built once — see :class:`ShiftedSpaces`, which also
        #: holds at most one sigma workspace across the family at any moment.
        self.spaces = spaces if spaces is not None else ShiftedSpaces(
            self.n_active, space.n_elec, self.h_act, self.eri_act)
        self.civec = np.ascontiguousarray(civec, dtype=np.complex128).ravel()
        builder = builder if builder is not None else RDMBuilder(space)
        with timer("NEVPT2 state RDMs"):
            # enforce_kramers is off *because this is one state*: the state-averaging gate is a statement
            # about an ensemble, and the driver has already applied it to the density that
            # defines H0. A single state's own RDMs are what SC-NEVPT2's norms are made of.
            gamma, gamma2 = builder(self.civec, enforce_kramers=False)
        self._gamma = gamma
        self._gamma2 = gamma2
        self._derived: dict = {}
        #: The state's own active-space eigenvalue, from its own RDMs. ⚠ Not taken from the
        #: caller: every Koopmans matrix here is ``<.. (H_act - E) ..>`` and the ``E`` that
        #: makes it Hermitian is the expectation value of the *same* Hamiltonian these RDMs
        #: were contracted with, not whatever number a CI solver reported.
        self.e_active = active_space_energy(self.h_act, self.eri_act, gamma, gamma2)

    # -- stored ranks ---------------------------------------------------------------------
    def rdm1(self) -> np.ndarray:
        return self._gamma

    def rdm2(self) -> np.ndarray:
        return self._gamma2

    def rdm3(self) -> np.ndarray:
        raise NotImplementedError(
            "no 3-RDM is built on the conventional-CI path, and none is needed: every rank-3 "
            "requirement of SC-NEVPT2 is a Gram matrix of ladder-string vectors with H_act "
            "between them, served by pair_koopmans / hole_pair_koopmans / "
            "excitation_koopmans. See this module's docstring for why "
            "that replaced the stored n_act^6 object")

    def contract_rdm4(self, active_eri: np.ndarray):
        raise NotImplementedError(
            "no rank-4 quantity is contracted on the conventional-CI path, and none is "
            "needed: the primed single-external classes contract the active integrals into "
            "ONE perturber vector per external label before any Gram is taken, which is what "
            "annihilation_perturbers / creation_perturbers do. A stored or contracted 4-RDM "
            "is the n_act^3-string route this module's docstring rejects; see it and "
            "this module's docstring for the crossover")

    # -- derived primitives ------------------------------------------------------------------
    def hole_rdm1(self) -> np.ndarray:
        return self._cached("hole1", lambda: hole_rdm1(self._gamma))

    def pair_matrix(self) -> np.ndarray:
        return self._cached("pair", lambda: pair_matrix(self._gamma2))

    def hole_pair_matrix(self) -> np.ndarray:
        return self._cached("hole_pair", lambda: hole_pair_matrix(self._gamma, self._gamma2))

    def koopmans_annihilation(self) -> np.ndarray:
        return self._cached("koopmans_a", lambda: koopmans_annihilation(
            self.h_act, self.eri_act, self._gamma, self._gamma2, check=self._check))

    def koopmans_creation(self) -> np.ndarray:
        return self._cached("koopmans_c", lambda: koopmans_creation(
            self.h_act, self.eri_act, self._gamma, self._gamma2, check=self._check))

    # -- the Gram route: rank-3 requirements without a 3-RDM -----------------------------------
    def annihilated(self) -> np.ndarray:
        """``A_t = a_t |Psi>`` over the active modes, as ``(n_act, ndet_-1)``. Cached.

        Shared by the pair vectors and the excitation vectors, which is worth one cache entry:
        ``E_tu|Psi> = a+_t (a_u|Psi>)`` and ``a_u a_t|Psi> = a_u (a_t|Psi>)``.
        """
        return self._cached("annihilated", self._build_annihilated)

    def pair_koopmans(self) -> np.ndarray:
        """``<Psi| a+_t a+_u (H_act - E) a_w a_v |Psi>`` as ``M[(t,u), (v,w)]``.

        The denominator kernel of ``Srs (-2)``, and the Gram matrix of
        ``|D_tu> = a_u a_t |Psi>`` with ``H_act - E`` between. ⚠ Its **overlap** partner is
        :meth:`pair_matrix`, built from the 2-RDM by a completely different route — the two
        are asserted equal in the tests, which is what validates the ladder-string signs
        against the density algebra.
        """
        return self._cached("pair_koopmans", lambda: self._pair_gram()[1])

    def hole_pair_koopmans(self) -> np.ndarray:
        """``<Psi| a_u a_t (H_act - E) a+_v a+_w |Psi>`` as ``M[(t,u), (v,w)]``.

        The denominator kernel of ``Sij (+2)``; overlap partner :meth:`hole_pair_matrix`.
        """
        return self._cached("hole_pair_koopmans", lambda: self._hole_pair_gram()[1])

    def excitation_overlap(self) -> np.ndarray:
        """``S`` over the **augmented** set ``[|Psi>, E_00|Psi>, E_01|Psi>, ...]``.

        ``(1 + n_act^2)`` square. The ``Sir (0')`` perturber is
        ``a+_a a_i (f^I_ai + sum_tu g_tu E_tu)|Psi>``, i.e. a vector in exactly this span with
        the reference itself as its zeroth component — hence the augmentation rather than a
        separate scalar term bolted onto an ``n_act^2`` form. Row and column zero are
        ``<Psi|Psi> = 1`` and ``<Psi|E_tu|Psi> = gamma_tu``.
        """
        return self._cached("exc_overlap", lambda: self._excitation_gram()[0])

    def excitation_koopmans(self) -> np.ndarray:
        """``M`` over the same augmented set, with ``H_act - E`` between.

        ⚠ Its first row and column are **exactly zero**, because ``(H_act - E)|Psi> = 0``. That
        is not a special case to code around — it falls out — and it is a free check on the
        state actually being an eigenvector.
        """
        return self._cached("exc_koopmans", lambda: self._excitation_gram()[1])

    # -- the ladder-string vector sets, cached and shared --------------------------------------
    #
    # ⚠ Cached rather than transient because each is used by **two** classes — the rank-3 one
    # that motivated it and the rank-4 one of the same electron-number shift — and rebuilding
    # costs the same ladder applications a second time. The driver drops the whole provider
    # between states, and :meth:`release` exists for a caller that wants the memory back sooner.

    def created(self) -> np.ndarray:
        """``B_t = a+_t |Psi>`` over the active modes, as ``(n_act, ndet_+1)``. Cached.

        The mirror of :meth:`annihilated`, shared by the ``Sij (+2)`` pair vectors and by the
        one-body half of the ``Si (+1')`` perturber.
        """
        return self._cached("created", self._build_created)

    def pair_annihilated(self) -> np.ndarray:
        """``|D_tu> = a_u a_t |Psi>`` as ``(n_act^2, ndet_-2)``, row index ``t*n + u``. Cached.

        ⚠ The row flattening is ``t`` outer, ``u`` inner, and it is the same convention
        :func:`pair_matrix` uses for its ``[(t,u), (v,w)]`` indexing — the two are asserted
        equal in the tests, which is what validates the ladder signs against the density
        algebra. Also the string set the ``Sr (-1')`` perturber is a linear combination of,
        one creation operator further on.
        """
        return self._cached("pair_annihilated", self._build_pair_annihilated)

    def excitation_vectors(self) -> np.ndarray:
        """``[|Psi>, E_00|Psi>, E_01|Psi>, ...]`` as ``(1 + n_act^2, ndet)``. Cached.

        Row ``1 + t*n + u`` is ``E_tu|Psi>``. Row zero is the reference itself, because the
        ``Sir (0')`` perturber lives in exactly this augmented span; the ``Si (+1')`` perturber
        uses rows ``1:`` and its own creation operator.
        """
        return self._cached("excitation_vectors", self._build_excitation_vectors)

    # -- the single-external perturbers: one vector per external label -------------------------

    def annihilation_perturbers(self, w_one: np.ndarray, w_three: Optional[np.ndarray],
                                *, full: bool = False):
        """Norms and Koopmans values of ``Sr (-1')``-shaped perturbers, one per label row.

        For each row ``L`` of the coefficients, the vector in the ``N-1`` electron space is

        ::

            |P_L> = sum_t w1[L,t] a_t|Psi>  +  sum_tuv w3[L,t,u,v] a+_u a_v a_t |Psi>

        and what comes back is ``N_L = <P_L|P_L>`` and ``K_L = <P_L|(H_act - E)|P_L>``, both
        real by construction. See the module docstring for why the combination is formed
        *before* the Gram rather than after it.

        Parameters
        ----------
        w_one : ``(n_labels, n_act)`` complex
        w_three : ``(n_labels, n_act, n_act, n_act)`` complex or ``None``
            Indexed ``[L, t, u, v]``, matching the ``(at|uv)`` of the working equation. ``None``
            means the class has no two-body part.
        full : bool
            Return the whole ``(n_labels, n_labels)`` overlap and Koopmans matrices instead of
            their diagonals. ⚠ Not used by strong contraction, and the reason it exists is
            FIC: with the identity as coefficients these two matrices **are** the internally
            contracted metric and per-class Hamiltonian over the string basis.

        Returns
        -------
        ``(norm, koopmans, max_imaginary)`` with the first two ``(n_labels,)`` real arrays, or
        ``(overlap, koopmans, max_imaginary)`` with the first two ``(n_labels, n_labels)``
        complex Hermitian matrices when ``full``.
        """
        n = self.n_active
        minus1 = self._shifted(-1)
        vectors = self._empty_perturbers(w_one, minus1, "annihilation")
        if vectors is None:
            return self._degenerate_forms(w_one.shape[0], full)
        annihilated = self.annihilated()
        if annihilated.size:
            vectors += np.ascontiguousarray(w_one) @ annihilated
        minus2 = self._shifted(-2)
        if w_three is not None and not minus2.empty:
            pairs = self.pair_annihilated()                     # rows (t, v)
            rows = w_one.shape[0]
            with timer("NEVPT2 single-external annihilation perturbers"):
                for u in range(n):
                    # a+_u |D_tv> = a+_u a_v a_t |Psi>, so one creation operator turns the
                    # whole cached pair set into the u-th slice of the string basis at once —
                    # n ladder applications and n GEMMs, not n^3 of either.
                    lifted = apply_ladder(minus2.masks, minus1.masks, u, pairs, dagger=True)
                    vectors += (np.ascontiguousarray(w_three[:, :, u, :]).reshape(rows, n * n)
                                @ lifted)
        return self._forms(minus1, vectors, full)

    def creation_perturbers(self, w_one: np.ndarray, w_three: Optional[np.ndarray],
                            *, full: bool = False):
        """Norms and Koopmans values of ``Si (+1')``-shaped perturbers, one per label row.

        The particle-hole mirror of :meth:`annihilation_perturbers`, in the ``N+1`` space:

        ::

            |P_L> = sum_t w1[L,t] a+_t|Psi>  +  sum_tuv w3[L,t,u,v] a+_t a+_u a_v |Psi>

        ``w_three`` is indexed ``[L, t, u, v]``, matching ``(ti|uv)``.
        """
        n = self.n_active
        plus1 = self._shifted(+1)
        vectors = self._empty_perturbers(w_one, plus1, "creation")
        if vectors is None:
            return self._degenerate_forms(w_one.shape[0], full)
        created = self.created()
        if created.size:
            vectors += np.ascontiguousarray(w_one) @ created
        if w_three is not None and self._shifted(-1).ndet:
            excited = self.excitation_vectors()[1:]             # rows (u, v) = E_uv|Psi>
            rows = w_one.shape[0]
            with timer("NEVPT2 single-external creation perturbers"):
                for t in range(n):
                    # a+_t E_uv|Psi> = a+_t a+_u a_v |Psi>: the same one-creation lift as
                    # above, applied to the cached excitation set.
                    lifted = apply_ladder(self.space.masks, plus1.masks, t, excited,
                                          dagger=True)
                    vectors += (np.ascontiguousarray(w_three[:, t, :, :]).reshape(rows, n * n)
                                @ lifted)
        return self._forms(plus1, vectors, full)

    def release(self) -> None:
        """Drop the cached ladder-string vector sets (not the RDMs, which are small)."""
        for key in ("annihilated", "created", "pair_annihilated", "excitation_vectors"):
            self._derived.pop(key, None)

    def shifted_ndet(self, delta: int) -> int:
        """Determinant count of the ``n_elec + delta`` space; zero when it does not exist.

        Exposed so a class can size its own label batches (:func:`perturber_vector_gb`)
        without reaching into :class:`ShiftedSpaces`.
        """
        space = self._shifted(int(delta))
        return 0 if space.empty else int(space.ndet)

    # -- construction --------------------------------------------------------------------------
    def _shifted(self, delta: int) -> ShiftedSpace:
        return self.spaces.get(delta)

    def _empty_perturbers(self, w_one: np.ndarray, space: ShiftedSpace,
                          what: str) -> Optional[np.ndarray]:
        """The zeroed accumulator, or ``None`` when the target space does not exist.

        ⚠ An absent ``N-1`` (or ``N+1``) space is not an error and not a corner case to guard
        with an exception: the ladder string annihilates the reference, every perturber of the
        class vanishes identically, and the class contributes exactly zero. A full active space
        (the complete-active-space null test) reaches it by construction.
        """
        rows = int(np.shape(w_one)[0])
        if space.empty or space.ndet == 0 or rows == 0:
            return None
        # ⚠ Once per *batch* of external labels, which is the granularity at which this array
        # is actually allocated — not inside any kernel loop. A class calls this a
        # handful of times per state, matching the budget's "order 10-100 calls per run" for a
        # resident declaration.
        res.require("NEVPT2 {} perturber vectors".format(what),
                    ladder_vector_gb(self.n_active, space.ndet, n_vectors=rows),
                    note="{} external labels x {} determinants".format(rows, space.ndet),
                    advice=["freeze core spinors or delete high virtuals: "
                            "this set is one vector per external label",
                            "lower the class batch size, which is what bounds this array"])
        return np.zeros((rows, space.ndet), dtype=np.complex128)

    @staticmethod
    def _degenerate_forms(rows: int, full: bool):
        shape = (rows, rows) if full else (rows,)
        dtype = np.complex128 if full else np.float64
        return np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype), 0.0

    def _forms(self, space: ShiftedSpace, vectors: np.ndarray, full: bool):
        """``(N, K)`` for a set of perturber vectors: diagonals, or the whole Gram pair."""
        if full:
            overlap, koopmans = space.gram(vectors, self.e_active)
            return overlap, koopmans, 0.0
        applied = space.apply_h(vectors) - float(self.e_active) * vectors
        # The norm is manifestly real; the Koopmans value is real only because H_act is
        # Hermitian, so its imaginary part is the conjugation-trap diagnostic.
        norm = np.sum(vectors.real ** 2 + vectors.imag ** 2, axis=1)
        koopmans = np.einsum("Ld,Ld->L", vectors.conj(), applied)
        return (np.ascontiguousarray(norm), np.ascontiguousarray(koopmans.real),
                float(np.max(np.abs(koopmans.imag))) if koopmans.size else 0.0)

    def _build_annihilated(self) -> np.ndarray:
        minus = self._shifted(-1)
        n = self.n_active
        if minus.empty:
            return np.zeros((n, 0), dtype=np.complex128)
        res.require("NEVPT2 annihilated reference vectors",
                    ladder_vector_gb(n, minus.ndet, n_vectors=n),
                    note="{} modes x {} determinants".format(n, minus.ndet))
        out = np.empty((n, minus.ndet), dtype=np.complex128)
        for mode in range(n):
            out[mode] = apply_ladder(self.space.masks, minus.masks, mode, self.civec)
        return out

    def _build_created(self) -> np.ndarray:
        plus = self._shifted(+1)
        n = self.n_active
        if plus.empty:
            return np.zeros((n, 0), dtype=np.complex128)
        res.require("NEVPT2 created reference vectors",
                    ladder_vector_gb(n, plus.ndet, n_vectors=n),
                    note="{} modes x {} determinants".format(n, plus.ndet))
        out = np.empty((n, plus.ndet), dtype=np.complex128)
        for mode in range(n):
            out[mode] = apply_ladder(self.space.masks, plus.masks, mode, self.civec,
                                     dagger=True)
        return out

    def _build_pair_annihilated(self) -> np.ndarray:
        n = self.n_active
        minus2 = self._shifted(-2)
        if minus2.empty:
            return np.zeros((n * n, 0), dtype=np.complex128)
        annihilated = self.annihilated()
        res.require("NEVPT2 pair-annihilated vectors", ladder_vector_gb(n, minus2.ndet),
                    note="{} orbital pairs x {} determinants".format(n * n, minus2.ndet),
                    advice=["reduce the active space: this set is n_act^2 x C(n_act, N-2)"])
        vectors = np.empty((n * n, minus2.ndet), dtype=np.complex128)
        minus1 = self._shifted(-1)
        with timer("NEVPT2 pair-annihilated vectors"):
            for u in range(n):
                # |D_tu> = a_u a_t |Psi> = a_u A_t, so the inner index of the row block is t.
                vectors[u::n] = apply_ladder(minus1.masks, minus2.masks, u, annihilated)
        return vectors

    def _build_excitation_vectors(self) -> np.ndarray:
        n = self.n_active
        here = self._shifted(0)
        annihilated = self.annihilated()
        res.require("NEVPT2 excitation vectors", ladder_vector_gb(n, here.ndet, n * n + 1),
                    note="1 + {} orbital pairs x {} determinants".format(n * n, here.ndet),
                    advice=["reduce the active space: this set is (1 + n_act^2) x ndet"])
        vectors = np.empty((n * n + 1, here.ndet), dtype=np.complex128)
        vectors[0] = self.civec
        minus1 = self._shifted(-1)
        with timer("NEVPT2 excitation vectors"):
            if annihilated.size:
                for t in range(n):
                    # |X_tu> = E_tu|Psi> = a+_t (a_u|Psi>) = a+_t A_u: the row block index is t
                    # and the inner index u, matching the (t*n + u) flattening the classes use.
                    vectors[1 + t * n:1 + (t + 1) * n] = apply_ladder(
                        minus1.masks, here.masks, t, annihilated, dagger=True)
            else:
                vectors[1:] = 0.0
        return vectors

    def _pair_gram(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_active
        minus2 = self._shifted(-2)
        if minus2.empty:
            zero = np.zeros((n * n, n * n), dtype=np.complex128)
            return zero, zero.copy()
        return minus2.gram(self.pair_annihilated(), self.e_active)

    def _hole_pair_gram(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_active
        plus2 = self._shifted(+2)
        if plus2.empty:
            zero = np.zeros((n * n, n * n), dtype=np.complex128)
            return zero, zero.copy()
        plus1 = self._shifted(+1)
        created = self.created()
        res.require("NEVPT2 pair-created vectors", ladder_vector_gb(n, plus2.ndet),
                    note="{} orbital pairs x {} determinants".format(n * n, plus2.ndet),
                    advice=["reduce the active space: this set is n_act^2 x C(n_act, N+2)"])
        vectors = np.empty((n * n, plus2.ndet), dtype=np.complex128)
        with timer("NEVPT2 pair-created vectors"):
            for t in range(n):
                # |C_tu> = a+_t a+_u |Psi> = a+_t B_u, so the outer index of the row block is t.
                vectors[t * n:(t + 1) * n] = apply_ladder(plus1.masks, plus2.masks, t, created,
                                                          dagger=True)
        return plus2.gram(vectors, self.e_active)

    def _excitation_gram(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._shifted(0).gram(self.excitation_vectors(), self.e_active)

    def _cached(self, key: str, build):
        value = self._derived.get(key)
        if value is None:
            value = build()
            self._derived[key] = value
        return value

    def __repr__(self) -> str:
        return "CIContractionProvider(n_active={}, ndet={})".format(
            self.n_active, self.space.ndet)


def active_hamiltonian_check(provider: "CIContractionProvider", energy: float,
                             tol: float = 1e-8) -> float:
    """``|E - (sum h gamma + 1/2 sum (pq|rs) Gamma)|`` — the closure check, returned.

    The cheapest statement that the provider's RDMs and the Dyall active Hamiltonian are the
    matching pair. Called by the driver at DEBUG; ``tol`` is only used for the message.
    """
    from ..rdm.rdm import active_space_energy
    err = abs(active_space_energy(provider.h_act, provider.eri_act,
                                  provider.rdm1(), provider.rdm2()) - float(energy))
    if err > tol:
        log.warning("the active-space energy rebuilt from this state's RDMs differs from its "
                    "CI eigenvalue by %.3e Eh; the perturbation is being built on a vector "
                    "that is not an eigenvector of the active Hamiltonian", err)
    return err


__all__ = ["CIContractionProvider", "ShiftedSpace", "ShiftedSpaces",
           "perturber_vector_gb",
           "active_hamiltonian_check", "ladder_vector_gb", "hole_pair_matrix",
           "hole_rdm1", "koopmans_annihilation", "koopmans_creation", "pair_matrix",
           "KOOPMANS_HERMITICITY_RTOL"]
