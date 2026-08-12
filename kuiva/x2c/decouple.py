"""The X2C decoupling transformation: block container, decoupling matrices, picture change.

Everything here is pure linear algebra on four-component blocks. No integral library, no
``Mole``, no four-component solver — which is what lets the same three functions serve the
atomic mean field of :mod:`kuiva.amf` and the molecular one-electron path of
:mod:`kuiva.interface.pyscf_bridge` without either knowing about the other.

The transformation
------------------
In a restricted kinetically balanced basis the four-component one-electron problem is blocked
``LL/LS/SL/SS``. X2C finds the decoupling matrix ``X`` relating small to large components
(``C_S = X C_L``) and the renormalization ``R``, and the two-component operator is::

    A_X2C = R^dag ( A_LL + A_LS X + X^dag A_SL + X^dag A_SS X ) R

``X`` comes from the positive-energy eigenvectors of a four-component eigenproblem
(:func:`decoupling_matrices`), ``R`` from ``R^dag S~ R = S`` with ``S~ = S_LL + X^dag S_SS X``
(:func:`renormalization`), and the transformation itself is :func:`picture_change`. That the
*same* function transforms a one-electron Hamiltonian and a two-electron mean field is the
content of X2CAMF, not a coincidence worth abstracting away.

Block conventions
-----------------
The four-component one-electron operator is stored in the **modified (restricted kinetically
balanced) small-component normalization**, in which the small-component basis is
``|S> = (sigma.p / 2c) |L>``. In that convention::

    hcore.ll = V                       overlap.ll = S
    hcore.ls = hcore.sl = T            overlap.ls = overlap.sl = 0
    hcore.ss = W / (4 c^2) - T         overlap.ss = T / (2 c^2)

with ``T`` the nonrelativistic kinetic energy, ``V`` the nuclear attraction and
``W = <sigma.p V sigma.p>``. This is the convention PySCF's ``scf.dhf`` and ``x2c`` both use
and the one the X2C equations are written in; fixing it here means a future four-component
backend must match it rather than invent its own.

All matrices are ``(2*nao, 2*nao)`` in the spin-blocked ``[alpha; beta]`` row layout of
the fixed spin-blocked basis — *not* the j-adapted 2-spinor basis a four-component code naturally works in.

References
----------
* Exact two-component decoupling: W. Kutzelnigg, W. Liu, J. Chem. Phys. 123, 241102 (2005),
  doi:10.1063/1.2137315; W. Liu, D. Peng, J. Chem. Phys. 125, 044102 (2006),
  doi:10.1063/1.2222365; M. Ilias, T. Saue, J. Chem. Phys. 126, 064102 (2007),
  doi:10.1063/1.2436882; D. Peng, M. Reiher, Theor. Chem. Acc. 131, 1081 (2012),
  doi:10.1007/s00214-011-1081-y.
* Local (atom-blocked) exact decoupling, i.e. the DLU approximation: D. Peng, M. Reiher,
  "Local relativistic exact decoupling", J. Chem. Phys. 136, 244108 (2012),
  doi:10.1063/1.4729788.
* Restricted kinetic balance: R. E. Stanton, S. Havriliak, J. Chem. Phys. 81, 1910 (1984),
  doi:10.1063/1.447865; K. G. Dyall, K. Faegri, Chem. Phys. Lett. 174, 25 (1990),
  doi:10.1016/0009-2614(90)85321-3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.linalg

from ..util import resources as res
from ..util.degeneracy import DEFAULT_GROUP_RTOL, cut_at_group_boundary
from ..util.logging import get_logger

log = get_logger(__name__)

#: Speed of light in atomic units, ``1 / alpha``: the CODATA 2018 fine-structure constant.
#: Used for **reporting only** — every calculation takes its ``c`` from the object that
#: recorded it, never from here (the ``c -> inf`` limit is a first-class test).
#:
#: ⚠ **This is NOT PySCF's value, and must never be used as the ``c`` of a calculation whose
#: integrals came from PySCF.** PySCF ships ``lib.param.LIGHT_SPEED = 137.03599967994``, an
#: older determination, against this CODATA 2018 ``137.035999084`` — a relative difference of
#: **4.3e-09** (8.7e-09 in ``c^2``, which is how it enters). That is far below any physical
#: tolerance and far *above* the level at which
#: Kuiva reproduces PySCF's X2C Hamiltonian, so mixing the two shows up immediately as a
#: numerical discrepancy with no physical cause: building the molecular four-component blocks
#: at this constant instead of PySCF's degrades agreement from **6e-13 to 1e-11 relative**.
#: This docstring previously claimed the two matched. They do not, and the rule that follows
#: is that ``c`` belongs to whatever produced the integrals
#: (:func:`kuiva.interface.pyscf_bridge.four_component_one_electron`,
#: :func:`kuiva.amf.pyscf_dhf.current_light_speed`), never to a constant chosen elsewhere.
LIGHT_SPEED = 137.035999084

#: Eigenvalues of the (diagonal-normalized) four-component metric below this are dropped
#: before anything is solved in it. **It lives here, in the module that fixes the block
#: conventions, because every operation on that metric must use the same value** — a
#: four-component SCF and the X2C decoupling of its mean field are two operations on one
#: metric, and if the decoupling represents directions the SCF projected away, ``X`` and ``R``
#: describe a space the density and mean field they are applied to do not live in.
#:
#: ⚠ **That inconsistency is not hypothetical; it was measured, and it is silent.** The
#: decoupling used 1e-14 at first, on the argument that discarding a direction the decoupling
#: cannot represent is worse than keeping a nearly redundant one. The argument is wrong for a
#: decontracted Karlsruhe lanthanide set, whose metric is numerically singular (Ce: smallest
#: eigenvalue **6.1e-14** over 1004 directions). Keeping those directions filled ``X`` with
#: noise — ``max |X| = 5.5e+03`` against **7.7** for a well-conditioned atom — and produced a
#: two-electron correction that was **96% time-reversal odd**, i.e. entirely meaningless.
#:
#: ⚠ **And the per-family check that was supposed to catch this cannot.** Reproducing PySCF's
#: one-electron X2C stayed at 9e-8 relative for Ce at *every* threshold, because ``h`` is
#: dominated by well-conditioned directions while the two-electron mean field is not. The
#: diagnostic that works is ``max |X|`` itself, and it is asserted in ``tests/test_amf_basis.py``.
#:
#: The same effect is visible on **molecules**, not only isolated atoms: decontracted TlH in
#: ``x2c-SVPall-2c`` has ``max |X| = 988`` with nothing dropped and **13.8** at this
#: threshold, where 62 of 960 directions go (``tests/test_x2c_decouple.py``).
#:
#: The value matches :mod:`kuiva.orth.canonical`'s default, so the same
#: redundancy is judged the same way wherever it appears. Measured effect of adopting it in
#: the decoupling: Ne, Ar and Kr drop **nothing** and are bitwise unchanged — which is what
#: keeps every committed DIRAC reference number valid — while Ce, Dy, Yb and Bi drop 42-66 of
#: ~1000 directions and their ``max |X|`` falls to 3-10.
METRIC_LINDEP_THRESHOLD = 1e-7

#: Overlap eigenvalues below this fraction are dropped when inverting a metric square root.
#: Only reached for a genuinely singular basis; the supported uncontracted sets sit far
#: above it. ⚠ This is a *relative* cut inside :func:`renormalization`, and is unrelated to
#: :data:`METRIC_LINDEP_THRESHOLD`, which is the absolute one applied to the four-component
#: metric before anything is solved in it.
_METRIC_TOL = 1e-14


# --- Sizing functions ----------------------------------------------------

def blocks_memory_gb(nao: int) -> float:
    """Size [GB] of one :class:`FourComponentBlocks` over ``nao`` scalar basis functions.

    Four ``(2*nao, 2*nao)`` ``complex128`` matrices, i.e. ``4 * (2 nao)^2 * 16`` bytes. The
    atomic problem is small — 0.5 MB for Ne's uncontracted basis, 6 MB for Xe's — but the accounting rule
    admits no exceptions to the accounting rule, and the point of an exact sizing function is
    that it is exact whatever the size (``tests/test_amf_backend.py`` pins it two-sidedly).
    """
    return 4.0 * res.array_gb((2 * nao, 2 * nao), np.complex128)


def decoupling_memory_gb(nao: int) -> float:
    """Size [GB] of the decoupling matrices ``X`` and ``R`` (exact sizing function).

    Two ``(2*nao, 2*nao)`` ``complex128`` matrices. Both the exact and the local (DLU) paths
    produce exactly these, so the *result* of a decoupling costs the same either way; what
    differs is the workspace used to get there (:func:`exact_decoupling_workspace_gb`).
    """
    return 2.0 * res.array_gb((2 * nao, 2 * nao), np.complex128)


def exact_decoupling_workspace_gb(nao: int) -> float:
    """Size [GB] of the workspace an **exact** molecular decoupling needs.

    The four-component eigenproblem is assembled as two ``(4*nao, 4*nao)`` ``complex128``
    matrices — the operator and the metric — and the canonical orthogonalizer plus the
    eigenvector matrix are of the same order. Counted as **four** such matrices.

    ⚠ **This is the number that says whether the exact path is affordable, and therefore the
    number that makes DLU discoverable**: it grows as ``nao^2`` with a large
    constant — 0.24 GB at nao = 500, 6.0 GB at nao = 2500, 23.8 GB at nao = 5000 — and the local
    path replaces it with one such workspace **per fragment**, which for atom-sized fragments
    is negligible. A refusal here should name DLU as the knob.
    """
    return 4.0 * res.array_gb((4 * nao, 4 * nao), np.complex128)


# --- The block container ------------------------------------------------------------------

@dataclass(frozen=True)
class FourComponentBlocks:
    """A four-component operator in ``LL / LS / SL / SS`` blocks (see the module docstring).

    Each block is ``(2*nao, 2*nao)`` ``complex128`` in the spin-blocked ``[alpha; beta]`` row
    layout of kuiva/spinor/expand.py.
    """

    ll: np.ndarray
    ls: np.ndarray
    sl: np.ndarray
    ss: np.ndarray

    def __post_init__(self) -> None:
        shapes = {b.shape for b in (self.ll, self.ls, self.sl, self.ss)}
        if len(shapes) != 1:
            raise ValueError("the four blocks must have the same shape, got {}".format(
                sorted(shapes)))
        shape = shapes.pop()
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] % 2:
            raise ValueError("each block must be square with an even dimension (alpha block "
                             "then beta block), got {}".format(shape))

    @property
    def nao(self) -> int:
        """Size of the underlying **scalar** basis (half a block dimension)."""
        return int(self.ll.shape[0] // 2)

    @property
    def n2c(self) -> int:
        """Block dimension, ``2 * nao``."""
        return int(self.ll.shape[0])

    def assemble(self) -> np.ndarray:
        """The full ``(4*nao, 4*nao)`` matrix, large component first."""
        n = self.n2c
        big = np.empty((2 * n, 2 * n), dtype=np.complex128)
        big[:n, :n], big[:n, n:] = self.ll, self.ls
        big[n:, :n], big[n:, n:] = self.sl, self.ss
        return big

    def hermiticity(self) -> float:
        """``max |A - A^dag|`` of the assembled matrix — zero for a Hamiltonian, a density or
        a mean field, and a structural check that costs nothing."""
        a = self.assemble()
        return float(np.max(np.abs(a - a.conj().T)))

    def scale(self) -> float:
        """``max |A|`` over all four blocks."""
        return max(float(np.max(np.abs(b))) if b.size else 0.0
                   for b in (self.ll, self.ls, self.sl, self.ss))

    @classmethod
    def from_matrix(cls, a: np.ndarray) -> "FourComponentBlocks":
        """Split a ``(4*nao, 4*nao)`` matrix into its four blocks."""
        a = np.ascontiguousarray(a, dtype=np.complex128)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] % 4:
            raise ValueError("a four-component matrix must be square with a dimension "
                             "divisible by 4, got shape {}".format(a.shape))
        n = a.shape[0] // 2
        return cls(ll=np.ascontiguousarray(a[:n, :n]), ls=np.ascontiguousarray(a[:n, n:]),
                   sl=np.ascontiguousarray(a[n:, :n]), ss=np.ascontiguousarray(a[n:, n:]))


# --- The decoupling matrices --------------------------------------------------------------

def decoupling_matrices(blocks: FourComponentBlocks, overlap: FourComponentBlocks,
                        light_speed: float) -> Tuple[np.ndarray, np.ndarray]:
    """The X2C decoupling matrix ``X`` and renormalization ``R`` for a four-component operator.

    ``X`` comes from the positive-energy eigenvectors of the four-component eigenproblem
    defined by ``blocks`` in the metric ``overlap``, via ``C_S = X C_L``; ``R`` from
    ``R^dag S~ R = S`` with ``S~ = S_LL + X^dag S_SS X``. Both are ``(2*nao, 2*nao)`` in the
    spin-blocked basis.

    ``blocks`` may be a bare one-electron Hamiltonian or a converged four-component Fock — the
    choice is the caller's and is a genuine convention, not a right and a wrong one. See
    :func:`kuiva.amf.decouple.x2c_decoupling`, which makes it explicit and records it.

    The eigenproblem is solved in a canonically orthogonalized basis rather than by
    ``scipy.linalg.eigh(h, m)`` directly: the four-component metric's small-component block
    carries a ``1/(2c^2)`` factor, so the generalized problem is badly conditioned for exactly
    the same reason a four-component SCF is (:mod:`kuiva.amf.pyscf_dhf`, point 2).

    ⚠ **This is why Kuiva's exact molecular X2C-1e Hamiltonian is not bitwise PySCF's, and
    the difference grows with the element.** PySCF solves the generalized problem directly and
    removes no linear dependence; Kuiva projects the metric at
    :data:`METRIC_LINDEP_THRESHOLD` first. Where nothing is dropped the two agree to
    **1e-13 relative** (Ne, H2O, and TiCl3 once the threshold is loosened to 1e-9). Where
    directions do go the difference is a real, deliberate one: TiCl3 drops 6 of 1008 and
    agrees to 6.7e-09, decontracted TlH drops 62 of 960 and agrees to **2.4e-07**. Neither
    number is an error bar on the physics — the projected operator is the better-conditioned
    one, and ``max |X|`` is the diagnostic that says so (13.8 against 988 for TlH).
    """
    c = float(light_speed)
    s, s_ss = overlap.ll, overlap.ss
    n = s.shape[0]

    nao = n // 2
    res.reserve("exact X2C decoupling workspace",
                exact_decoupling_workspace_gb(nao) + decoupling_memory_gb(nao),
                note="nao = {}; one dense 4c eigenproblem of dimension {}".format(nao, 2 * n),
                advice=["use the local (DLU) decoupling, which solves one small problem per "
                        "fragment instead of one large one ",
                        "use a smaller basis set; this workspace grows as nao^2"])

    h = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    m = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    h[:n, :n], h[:n, n:] = blocks.ll, blocks.ls
    h[n:, :n], h[n:, n:] = blocks.sl, blocks.ss
    m[:n, :n] = s
    m[n:, n:] = s_ss

    x_orth = canonical_orth(m)
    e, a = np.linalg.eigh(x_orth.conj().T @ h @ x_orth)
    a = x_orth @ a
    # Positive-energy branch: the electronic states sit above -c^2, the positronic below.
    positive = e > -c * c
    cl, cs = a[:n, positive], a[n:, positive]
    # X = C_S C_L^-1, as a least-squares solve rather than an explicit inverse.
    x = scipy.linalg.lstsq(cl.conj().T, cs.conj().T)[0].conj().T

    s_nesc = s + x.conj().T @ s_ss @ x
    r = renormalization(s, s_nesc)
    return np.ascontiguousarray(x), np.ascontiguousarray(r)


def renormalization(s: np.ndarray, s_nesc: np.ndarray) -> np.ndarray:
    """``R`` with ``R^dag S~ R = S``: ``R = S^-1/2 (S^-1/2 S~ S^-1/2)^-1/2 S^1/2``.

    Written out rather than taken from a library because it is the matrix the density
    transformation of :func:`two_component_density` depends on, and because the
    ``S^1/2 ... S^-1/2`` sandwich is what makes ``R`` *not* Hermitian in a non-orthogonal
    basis — the property that makes ``R^-1 D R^-dag`` and ``R^dag D R`` genuinely different.
    """
    val, vec = np.linalg.eigh(s)
    keep = val > _METRIC_TOL * float(val.max())
    vec, val = vec[:, keep], val[keep]
    root, inv_root = np.sqrt(val), 1.0 / np.sqrt(val)
    mid = (vec.conj().T @ s_nesc @ vec) * inv_root[:, None] * inv_root[None, :]
    val1, vec1 = np.linalg.eigh(mid)
    keep1 = val1 > _METRIC_TOL * float(val1.max())
    mid_inv_root = (vec1[:, keep1] / np.sqrt(val1[keep1])) @ vec1[:, keep1].conj().T
    r = inv_root[:, None] * mid_inv_root * root[None, :]
    return vec @ r @ vec.conj().T


def canonical_orth(m: np.ndarray) -> np.ndarray:
    """``X`` with ``X^dag M X = 1``, dropping near-null directions of the 4c metric.

    Same scheme as :func:`kuiva.amf.pyscf_dhf._canonical_orthogonalization`, and **the same
    threshold**, :data:`METRIC_LINDEP_THRESHOLD`. They have to agree: they are two operations
    on the same metric, and a decoupling that represents directions a four-component SCF
    projected away describes a space the density and mean field it is applied to do not live
    in.

    ⚠ **This used to be 1e-14, with a docstring arguing that discarding more than strictly
    necessary was the worse error. That argument was wrong, and the way it failed is worth
    keeping.** A decontracted Karlsruhe lanthanide set is numerically singular — Ce's metric
    has a smallest eigenvalue of **6.1e-14** over 1004 directions — so a 1e-14 cut keeps the
    null space and ``X`` fills up with noise:

    ==== ============ ============ =====================
    atom max abs X    TR residual  one-electron check
    ==== ============ ============ =====================
    Ti      7.7          1.7e-09    5.2e-11 (fine)
    Ce   **5.5e+03**  **1.0e-03**   9.2e-08 (fine!)
    ==== ============ ============ =====================

    The two-electron correction built from that ``X`` came out **96% time-reversal odd**, i.e.
    meaningless. ⚠ **And the third column is the point**: reproducing PySCF's one-electron X2C
    — the per-family basis check — is at 9e-08 for Ce *whatever* the threshold, because ``h``
    is dominated by well-conditioned directions and the two-electron mean field is not. It
    cannot see this failure. ``max |X|`` can, and is asserted in ``tests/test_amf_basis.py``.

    At the shared 1e-7 value, Ne, Ar and Kr drop **nothing** and are bitwise unchanged (which
    is what keeps the committed reference numbers valid), while Ce, Dy, Yb and Bi drop
    42-66 of ~1000 directions and their ``max |X|`` falls to 3-10.
    """
    d = np.real(np.diag(m))
    norm = 1.0 / np.sqrt(np.where(d > 0.0, d, 1.0))
    mn = norm[:, None] * m * norm[None, :]
    val, vec = np.linalg.eigh(mn)
    keep = metric_keep_mask(val, METRIC_LINDEP_THRESHOLD)
    return norm[:, None] * (vec[:, keep] / np.sqrt(val[keep]))


def metric_keep_mask(val_ascending: np.ndarray, threshold: float,
                     degeneracy_rtol: float = DEFAULT_GROUP_RTOL) -> np.ndarray:
    """Boolean mask over an **ascending** metric spectrum, cut at a degenerate-group boundary.

    The shared cut of :func:`canonical_orth` and
    :func:`kuiva.amf.pyscf_dhf._canonical_orthogonalization` — the single-definition rule requires those two to
    agree, so they call one function rather than repeat one line.

    ⚠ **Whole groups, never single vectors**. A four-component metric of a
    decontracted heavy-element basis is spherically degenerate in ``(2l+1)``-plets, so keeping
    part of one would represent some ``m`` components of a shell and not others — an
    anisotropic projection, exactly the failure the orbit-complete Cholesky avoids. The
    whole straddling group is dropped rather than kept, because the threshold is there to
    exclude directions whose ``s^{-1/2}`` amplifies noise (see :mod:`kuiva.util.degeneracy`).

    ⚠ Measured over 84 (system, basis) pairs of the project's own systems and bases, at every
    threshold from 1e-5 to 1e-8, **no cut straddles a group** — Ce³⁺, Dy³⁺, Yb³⁺, Bi, Lu³⁺,
    Th⁴⁺ and TlH all cut cleanly between the atomic ``(2l+1)`` groups. This is a guard, and it
    reproduces the previous ``val >= threshold`` bitwise on every one of them
    (measured).

    ⚠ One deliberate unification: the comparison is now **strict** (``> threshold``) here as
    it always was in :mod:`kuiva.orth.canonical`, where it used to be ``>=``. The shared-primitive rule requires
    the two cuts to agree and they did not, on a boundary case no float metric reaches; they
    do now.
    """
    val = np.asarray(val_ascending, dtype=float)
    n_keep, straddle = cut_at_group_boundary(val[::-1], threshold, rtol=degeneracy_rtol)
    if straddle is not None:
        log.warning("the 4c metric projection at %.2e fell inside a degenerate group of %d "
                    "eigenvalues spanning %.6e..%.6e (relative spread %.2e); keeping %d of "
                    "them would project the metric anisotropically, so the whole group is "
                    "dropped. %d directions removed instead of %d.",
                    threshold, straddle.size, straddle.largest, straddle.smallest,
                    straddle.relative_spread, straddle.kept_by_threshold,
                    int(val.size - n_keep),
                    int(val.size - straddle.start - straddle.kept_by_threshold))
    mask = np.zeros(val.size, dtype=bool)
    if n_keep:
        mask[val.size - n_keep:] = True          # ascending: the kept ones are the largest
    return mask


# --- The picture change of an operator and of the density ---------------------------------

def picture_change(blocks: FourComponentBlocks, x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``R^dag ( A_LL + A_LS X + X^dag A_SL + X^dag A_SS X ) R`` — the X2C transformation of a
    four-component **operator**.

    Applied to the one-electron Hamiltonian this reproduces ``h_X2C``; applied to a
    two-electron mean field it is the picture change X2CAMF is built on.
    That it is literally the same function in both cases is the content of the method.
    """
    fw = blocks.ll + blocks.ls @ x + x.conj().T @ blocks.sl + x.conj().T @ blocks.ss @ x
    return np.ascontiguousarray(r.conj().T @ fw @ r)


def two_component_density(d_ll: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``D~ = R^+ D_LL R^+dag`` — the two-component density behind a four-component one.

    ⚠ **Not** ``R^dag D_LL R``. Coefficients and densities transform oppositely (a trap
    that has bitten this project twice), and ``R`` is not unitary, so the two
    expressions differ by a lot while both remain Hermitian with plausible traces. The
    accounting is in :mod:`kuiva.amf.decouple`, point 4; the energy functional against
    four-component theory is what tells the variants apart.

    ⚠ **A pseudo-inverse, not a solve, and this is not fastidiousness — ``R`` is genuinely
    singular for a linearly dependent basis.** :func:`renormalization` builds ``R`` from the
    eigenvectors of the overlap it keeps, so it is **exactly zero** on any direction dropped
    there, and a decontracted Karlsruhe lanthanide set has several: measured on neutral Ce,
    ``R`` has rank **496 of 502** with a condition number of **8.7e+17**. An LU solve against
    that divides by zero, and quietly:

    ==== ================= ================== ==================
    atom max abs D~ (solve) max abs D~ (pinv)  4c max abs D_LL
    ==== ================= ================== ==================
    Ne        4.648e-01         4.648e-01         4.648e-01
    Ti(3+)    6.723e-01         6.723e-01         6.722e-01
    Ce    **2.383e+16**         4.570e+00         4.538e+00
    ==== ================= ================== ==================

    The resulting correction was 1.1e-02 time-reversal odd and ``max abs dG`` was 1.6e+04 Eh
    against a physical 8.5 Eh. Where ``R`` is full rank the pseudo-inverse **is** the inverse,
    and Ne and Ti(3+) reproduce the previous values to the last printed digit — so nothing
    that ever worked changes.

    The null directions carry no density: they are basis directions the metric projection
    already removed, so inverting on the range of ``R`` is the physically meaningful operation
    rather than a numerical patch. The cut matches :data:`_METRIC_TOL`, the one
    :func:`renormalization` used to create the null space in the first place.
    """
    r_inv = np.linalg.pinv(np.asarray(r), rcond=_METRIC_TOL)
    return np.ascontiguousarray(r_inv @ np.asarray(d_ll) @ r_inv.conj().T)


__all__ = ["FourComponentBlocks", "LIGHT_SPEED", "METRIC_LINDEP_THRESHOLD", "metric_keep_mask",
           "blocks_memory_gb", "canonical_orth", "decoupling_matrices", "picture_change",
           "renormalization", "two_component_density"]
