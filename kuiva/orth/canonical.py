"""Canonical orthogonalization and linear-dependence removal.

The orthonormal working basis is the fundamental object of the code: it is what the
spinor expansion, the integral transformation, CI, DMRG and NEVPT2 are all built on, and it
is where the contraction-type complexity of the front-end stops. Producing it is one explicit, tested,
threshold-controlled step, performed here and nowhere else.

The construction
----------------
Diagonalize the AO overlap, ``S U = U s`` with ``s`` sorted descending, and set

    X = U_k s_k^{-1/2}          (nao x nwork),   keeping the k columns with s_k > threshold

so that ``X^T S X = 1``. Columns with a small overlap eigenvalue are the near-linearly-
dependent combinations of the AO basis: keeping them multiplies numerical noise by
``s^{-1/2}`` and is the classic route to a silently wrong correlated calculation. Dropping
them is a *deliberate, reported* reduction of the variational space — hence the WARNING at
warning severity, the recorded count, and the diagnostics on this object.

This matters more here than in a typical code: uncontracted (Dyall) sets are deliberately supported
and mixed general/uncontracted (Peterson cc-pVnZ-X2C) sets on heavy elements, in molecules
where a metal's diffuse functions overlap a ligand's. Near-linear dependence is the expected
case, not an exotic one.

Why canonical and not symmetric (Loewdin) orthogonalization by default
----------------------------------------------------------------------
Symmetric orthogonalization ``X = S^{-1/2}`` is the unique orthonormal basis closest to the
original AOs in a least-squares sense, which makes it the right choice for population analysis
and for keeping orbitals atom-like. But it is square by construction: it cannot *remove* a
dimension, only produce an ill-conditioned basis when ``S`` is nearly singular. Canonical
orthogonalization discards a dimension cleanly, so it is the default for the working basis.
Both schemes are available; :func:`orthogonalize` refuses to return a symmetric basis when
vectors would have to be dropped rather than silently returning a rank-deficient one.

Conventions fixed here (they propagate downstream, so they are part of the interface)
------------------------------------------------------------------------------------
* Eigenvalues, and therefore working-basis functions, are ordered by **descending** overlap
  eigenvalue. The best-conditioned combinations come first and the dropped ones are exactly
  the tail; a truncation is then a slice, never a scatter.
* Column phases are fixed by making the largest-magnitude element of each column of ``U``
  positive. LAPACK's sign convention is not guaranteed across versions or vendors, and an
  unfixed phase makes checkpoint restarts and run-to-run comparisons needlessly
  irreproducible. Within an exactly degenerate block of ``s`` the eigenvectors are still only
  defined up to a rotation — no convention can fix that, and nothing downstream may depend on
  it (physical results must be invariant; that is a Tier-0 test).

References
----------
* Canonical and symmetric orthogonalization: P.-O. Loewdin, "On the Non-Orthogonality Problem
  Connected with the Use of Atomic Wave Functions in the Theory of Molecules and Crystals",
  J. Chem. Phys. 18, 365 (1950), doi:10.1063/1.1747632; and P.-O. Loewdin, "On the
  Non-Orthogonality Problem", Adv. Quantum Chem. 5, 185-199 (1970),
  doi:10.1016/S0065-3276(08)60339-1.
* Textbook treatment and the canonical/symmetric distinction: A. Szabo, N. S. Ostlund,
  "Modern Quantum Chemistry", Dover (1996), section 3.4.5.
* Linear-dependence removal by overlap-eigenvalue truncation in practical codes:
  J. Almloef, K. Faegri, K. Korsell, J. Comput. Chem. 3, 385 (1982),
  doi:10.1002/jcc.540030314.
* Cholesky orthogonalization (the ``cholesky`` scheme): F. Aquilante, T. B. Pedersen,
  V. Veryazov, R. Lindh, "MOLCAS - a software for multiconfigurational quantum chemistry
  calculations", WIREs Comput. Mol. Sci. 3, 143 (2013), doi:10.1002/wcms.1117, and the
  Cholesky-basis discussion in F. Aquilante et al., J. Chem. Phys. 125, 174101 (2006),
  doi:10.1063/1.2360264.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..util import output as out
from ..util.degeneracy import DEFAULT_GROUP_RTOL, cut_at_group_boundary
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

#: Overlap eigenvalues below this are treated as linear dependence and dropped. 1e-7 is the
#: conventional value (it is what PySCF, OpenMolcas and Turbomole use to within an order of
#: magnitude): comfortably above double-precision noise in ``S`` (~1e-14 relative) and below
#: the smallest eigenvalue a well-conditioned valence basis produces.
DEFAULT_THRESHOLD = 1.0e-7

#: Condition number of ``S`` above which the basis is reported as ill-conditioned even if
#: nothing was dropped: ``cond ~ 1e10`` costs about half the available digits in anything
#: built on it, and is worth knowing about before a correlated calculation, not after.
CONDITION_WARN = 1.0e10


@dataclass(frozen=True)
class OrthonormalBasis:
    """The orthonormal working basis and its conditioning diagnostics.

    Attributes
    ----------
    x : ndarray (nao, nwork)
        AO -> working-basis transformation, ``X^T S X = 1``.
    x_dag : ndarray (nwork, nao)
        Its metric adjoint ``X^T S``, i.e. the map that takes AO-basis *coefficient* columns
        into the working basis. (Not ``X^T``: coefficients and operators transform
        differently in a non-orthogonal basis, and mixing the two up is the standard bug
        this attribute exists to prevent.)
    s_eigenvalues : ndarray (nao,)
        All overlap eigenvalues, descending. The last ``n_dropped`` were discarded.
    n_dropped : int
    threshold : float
    scheme : str
        ``"canonical"``, ``"symmetric"`` or ``"cholesky"``.
    """

    x: np.ndarray
    x_dag: np.ndarray
    s_eigenvalues: np.ndarray
    n_dropped: int
    threshold: float
    scheme: str

    # -- dimensions ---------------------------------------------------------------------
    @property
    def nao(self) -> int:
        return int(self.x.shape[0])

    @property
    def nwork(self) -> int:
        """Number of working-basis functions; ``nao - n_dropped``."""
        return int(self.x.shape[1])

    # -- conditioning -------------------------------------------------------------------
    @property
    def smallest_kept(self) -> float:
        return float(self.s_eigenvalues[self.nwork - 1])

    @property
    def largest_dropped(self) -> Optional[float]:
        return float(self.s_eigenvalues[self.nwork]) if self.n_dropped else None

    @property
    def condition_number(self) -> float:
        """Condition number of ``S`` restricted to the retained space."""
        return float(self.s_eigenvalues[0]) / self.smallest_kept

    # -- transformations ----------------------------------------------------------------
    def to_working(self, c_ao: np.ndarray) -> np.ndarray:
        """Express AO-basis orbital coefficients in the working basis (``X^T S C``)."""
        return self.x_dag @ c_ao

    def to_ao(self, c_work: np.ndarray) -> np.ndarray:
        """Express working-basis orbital coefficients in the AO basis (``X C``)."""
        return self.x @ c_work

    def transform_operator(self, a_ao: np.ndarray) -> np.ndarray:
        """One-electron operator AO -> working basis: ``X^dag A X``."""
        return self.x.conj().T @ a_ao @ self.x

    def transform_density(self, d_work: np.ndarray) -> np.ndarray:
        """Density-like (contravariant) quantity working -> AO: ``X D X^dag``."""
        return self.x @ d_work @ self.x.conj().T

    def outside_norm(self, c_ao: np.ndarray, s_ao: np.ndarray) -> np.ndarray:
        """Norm of each AO-basis column lying *outside* the working space, in the S metric.

        The projector onto the working space is ``P = X X^T S``; this returns
        ``||(1 - P) c||_S`` per column. A value comparable to 1 means the orbital being
        projected lives in the discarded space and the projection is not faithful — which is
        why :func:`project_orbitals` checks it instead of trusting the projection blindly.
        """
        c_ao = np.atleast_2d(c_ao.T).T
        resid = c_ao - self.x @ (self.x_dag @ c_ao)
        return np.sqrt(np.abs(np.einsum("mi,mn,ni->i", resid.conj(), s_ao, resid)))

    def report(self, logger=None) -> None:
        """Log the standard working-basis block (the standard output grammar)."""
        logger = logger or log
        out.entry(logger, "orthogonalization scheme", self.scheme)
        out.entry(logger, "AO basis functions", self.nao)
        out.entry(logger, "working-basis functions", self.nwork,
                  note="{} dropped".format(self.n_dropped) if self.n_dropped else "")
        out.entry(logger, "linear-dependence threshold", self.threshold, fmt="{:.2e}")
        out.entry(logger, "largest overlap eigenvalue", float(self.s_eigenvalues[0]),
                  fmt="{:.6e}")
        out.entry(logger, "smallest retained overlap eigenvalue", self.smallest_kept,
                  fmt="{:.6e}")
        if self.n_dropped:
            out.entry(logger, "largest discarded overlap eigenvalue", self.largest_dropped,
                      fmt="{:.6e}")
        out.entry(logger, "condition number of S (retained)", self.condition_number,
                  fmt="{:.3e}")

    def __repr__(self) -> str:
        return ("OrthonormalBasis(scheme={}, nao={}, nwork={}, dropped={}, cond={:.2e})"
                .format(self.scheme, self.nao, self.nwork, self.n_dropped,
                        self.condition_number))


def _fix_column_phases(u: np.ndarray) -> np.ndarray:
    """Make the largest-magnitude element of each column positive (see module docstring)."""
    idx = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[idx, np.arange(u.shape[1])].real)
    signs[signs == 0.0] = 1.0
    return u * signs


def canonical_orthogonalization(s_ao: np.ndarray,
                                threshold: float = DEFAULT_THRESHOLD,
                                *, report: bool = False,
                                degeneracy_rtol: float = DEFAULT_GROUP_RTOL
                                ) -> OrthonormalBasis:
    """Canonical orthogonalization of the AO overlap with linear-dependence removal.

    Parameters
    ----------
    s_ao : ndarray (nao, nao)
        AO overlap matrix; symmetric positive definite up to the linear dependence being
        removed here.
    threshold : float
        Overlap eigenvalues at or below this are discarded (:data:`DEFAULT_THRESHOLD`).
    report : bool
        Log the standard working-basis block at INFO.
    degeneracy_rtol : float
        Relative gap below which neighbouring overlap eigenvalues count as degenerate, for
        the group-completeness rule below. See
        :mod:`kuiva.util.degeneracy` for why this is a parameter and not a constant.

    Notes
    -----
    ⚠ **The cut drops whole degenerate groups** (:func:`~kuiva.util.degeneracy.
    cut_at_group_boundary`). Keeping part of a degenerate group makes the working basis
    non-invariant under whatever symmetry made those directions degenerate, which is an
    *anisotropic* basis truncation and splits degeneracies the symmetry makes exact — the same
    failure the pivoted Cholesky avoids structurally. ⚠ On every system and basis this
    project has measured, no cut in the whole 1e-5..1e-8 range straddles a group and this rule
    changes nothing; it is a guard for the case that has not happened yet, not a correction to
    the ones that have (measured).

    Cost is one ``dsyev`` on ``S``: O(nao^3) with a small prefactor, run once per calculation,
    and negligible against the integral transformation. There is nothing to optimize here and
    no reason to trade accuracy for speed — use the most stable driver available.
    """
    s_ao = np.asarray(s_ao)
    if s_ao.ndim != 2 or s_ao.shape[0] != s_ao.shape[1]:
        raise ValueError("overlap must be square, got shape {}".format(s_ao.shape))
    asym = float(np.max(np.abs(s_ao - s_ao.conj().T))) if s_ao.size else 0.0
    if asym > 1e-10:
        log.warning("overlap matrix is not Hermitian (max asymmetry %.2e); symmetrizing", asym)
        s_ao = 0.5 * (s_ao + s_ao.conj().T)

    # eigh returns ascending eigenvalues; reverse for the descending convention (see above).
    evals, evecs = np.linalg.eigh(s_ao)
    evals = evals[::-1]
    evecs = _fix_column_phases(np.ascontiguousarray(evecs[:, ::-1]))

    keep, straddle = cut_at_group_boundary(evals, threshold, rtol=degeneracy_rtol)
    if straddle is not None:
        log.warning("the linear-dependence cut at %.2e fell inside a degenerate group of %d "
                    "overlap eigenvalues spanning %.6e..%.6e (relative spread %.2e); keeping "
                    "%d of them would make the working basis non-invariant, so the whole "
                    "group is dropped. %d functions removed instead of %d.",
                    threshold, straddle.size, straddle.largest, straddle.smallest,
                    straddle.relative_spread, straddle.kept_by_threshold,
                    int(evals.size - keep),
                    int(evals.size - straddle.start - straddle.kept_by_threshold))
    n_dropped = int(evals.size - keep)
    if keep == 0:
        if straddle is not None:
            raise ValueError(
                "the whole spectrum is one degenerate group straddling the threshold {:.2e} "
                "(relative spread {:.2e} <= rtol {:.1e}), so there is no cut that keeps whole "
                "groups; the overlap is wrong, or degeneracy_rtol is far too loose"
                .format(threshold, straddle.relative_spread, degeneracy_rtol))
        raise ValueError("every overlap eigenvalue is below the threshold {:.2e}; the basis "
                         "is empty or the overlap matrix is wrong".format(threshold))
    if np.any(evals < -1e-10):
        log.error("overlap matrix has negative eigenvalues (min %.3e); it is not a valid "
                  "overlap", float(evals.min()))

    x = evecs[:, :keep] / np.sqrt(evals[:keep])
    basis = OrthonormalBasis(
        x=np.ascontiguousarray(x),
        x_dag=np.ascontiguousarray(x.T @ s_ao),
        s_eigenvalues=evals,
        n_dropped=n_dropped,
        threshold=float(threshold),
        scheme="canonical",
    )
    _warn_conditioning(basis)
    if report:
        basis.report()
    return basis


def _warn_conditioning(basis: OrthonormalBasis) -> None:
    """WARNING: proceeds, but the user must know the variational space changed."""
    if basis.n_dropped:
        log.warning("dropped %d of %d basis functions as near-linearly-dependent "
                    "(overlap eigenvalue <= %.2e; largest discarded %.3e). The variational "
                    "space is reduced; energies are not comparable with a run at a different "
                    "threshold.", basis.n_dropped, basis.nao, basis.threshold,
                    basis.largest_dropped)
    if basis.condition_number > CONDITION_WARN:
        log.warning("working basis is ill-conditioned (cond(S) = %.2e). Expect a loss of "
                    "roughly %d significant digits in quantities built on it; consider "
                    "raising the linear-dependence threshold.",
                    basis.condition_number, int(np.log10(basis.condition_number)))


def sqrt_overlap(s_ao: np.ndarray) -> np.ndarray:
    """``S^{1/2}``, the matrix Loewdin population analysis is built on.

    ⚠ **Deliberately not** :func:`symmetric_orthogonalization`, which raises on a linearly
    dependent basis and is right to: it computes ``S^{-1/2}``, and a rank-deficient square
    ``X`` masquerading as an orthonormal basis is a singular matrix pushed downstream. The
    **square root** has no such problem — it is well defined for a singular ``S``, with the
    dependent directions simply carrying eigenvalue zero — so population analysis must not be
    the reason that guard gets weakened. Negative eigenvalues (rounding, order 1e-16) are
    clamped to zero rather than producing a complex square root.

    Returns a real symmetric ``(nao, nao)`` array.
    """
    s_ao = np.asarray(s_ao)
    if s_ao.ndim != 2 or s_ao.shape[0] != s_ao.shape[1]:
        raise ValueError("overlap must be square, got shape {}".format(s_ao.shape))
    evals, evecs = np.linalg.eigh(s_ao)
    if np.any(evals < -1e-10):
        log.error("overlap matrix has negative eigenvalues (min %.3e); it is not a valid "
                  "overlap", float(evals.min()))
    root = (evecs * np.sqrt(np.clip(evals, 0.0, None))) @ evecs.conj().T
    return np.ascontiguousarray(np.real_if_close(root))


def symmetric_orthogonalization(s_ao: np.ndarray,
                                threshold: float = DEFAULT_THRESHOLD,
                                *, report: bool = False) -> OrthonormalBasis:
    """Symmetric (Loewdin) orthogonalization ``X = S^{-1/2}``.

    Raises if any overlap eigenvalue is below ``threshold``: a symmetric basis cannot drop a
    dimension, and returning a rank-deficient square ``X`` would push a singular matrix
    downstream under the name "orthonormal basis". Use it for population analysis and for
    keeping orbitals atom-like, not as the working basis of an ill-conditioned set.
    """
    s_ao = np.asarray(s_ao)
    evals, evecs = np.linalg.eigh(s_ao)
    evals = evals[::-1]
    evecs = _fix_column_phases(np.ascontiguousarray(evecs[:, ::-1]))
    if evals[-1] <= threshold:
        raise ValueError(
            "symmetric orthogonalization is not defined for a linearly dependent basis "
            "(smallest overlap eigenvalue {:.3e} <= threshold {:.2e}); use the canonical "
            "scheme, which removes the dependent combinations".format(evals[-1], threshold))
    x = (evecs / np.sqrt(evals)) @ evecs.T
    basis = OrthonormalBasis(x=np.ascontiguousarray(x),
                             x_dag=np.ascontiguousarray(x.T @ s_ao),
                             s_eigenvalues=evals, n_dropped=0, threshold=float(threshold),
                             scheme="symmetric")
    _warn_conditioning(basis)
    if report:
        basis.report()
    return basis


def cholesky_orthogonalization(s_ao: np.ndarray, *, report: bool = False) -> OrthonormalBasis:
    """Cholesky orthogonalization ``X = (L^T)^{-1}`` from ``S = L L^T``.

    Cheapest of the three (one ``dpotrf`` plus a triangular inverse, no eigendecomposition)
    and it preserves locality, but it performs **no** linear-dependence detection and is
    numerically unsafe for a near-singular overlap. Offered for the well-conditioned case;
    the overlap eigenvalues are still computed for the diagnostics, so the conditioning
    warning is not lost.
    """
    s_ao = np.asarray(s_ao)
    from scipy.linalg import cholesky, solve_triangular

    ell = cholesky(s_ao, lower=True)
    x = solve_triangular(ell, np.eye(s_ao.shape[0]), lower=True, trans="T")
    evals = np.linalg.eigvalsh(s_ao)[::-1]
    basis = OrthonormalBasis(x=np.ascontiguousarray(x),
                             x_dag=np.ascontiguousarray(x.T @ s_ao),
                             s_eigenvalues=evals, n_dropped=0,
                             threshold=0.0, scheme="cholesky")
    if evals[-1] <= DEFAULT_THRESHOLD:
        log.warning("Cholesky orthogonalization applied to a near-singular overlap "
                    "(smallest eigenvalue %.3e): the resulting basis is orthonormal only to "
                    "within the conditioning of S. The canonical scheme is the safe choice.",
                    float(evals[-1]))
    _warn_conditioning(basis)
    if report:
        basis.report()
    return basis


def orthogonalize(s_ao: np.ndarray, scheme: str = "canonical",
                  threshold: float = DEFAULT_THRESHOLD, *,
                  report: bool = False,
                  degeneracy_rtol: float = DEFAULT_GROUP_RTOL) -> OrthonormalBasis:
    """Build the orthonormal working basis by ``scheme``; the single public entry point.

    ``degeneracy_rtol`` reaches only the canonical scheme, which is the only one that cuts.
    """
    schemes = {"canonical": canonical_orthogonalization,
               "symmetric": symmetric_orthogonalization,
               "cholesky": cholesky_orthogonalization}
    if scheme not in schemes:
        raise ValueError("unknown orthogonalization scheme {!r}; expected one of {}".format(
            scheme, sorted(schemes)))
    with timer("orthogonalization ({})".format(scheme)):
        if scheme == "cholesky":
            return cholesky_orthogonalization(s_ao, report=report)
        if scheme == "canonical":
            return canonical_orthogonalization(s_ao, threshold, report=report,
                                               degeneracy_rtol=degeneracy_rtol)
        return schemes[scheme](s_ao, threshold, report=report)


def project_orbitals(basis: OrthonormalBasis, c_ao: np.ndarray, s_ao: np.ndarray,
                     *, tol: float = 1e-6) -> np.ndarray:
    """Project AO-basis orbitals into the working basis, checking they actually fit in it.

    The scalar-X2C MOs arrive spanning the full AO space. If linear dependence was
    removed, some of that space no longer exists, and an orbital with a significant component
    in the discarded space cannot be represented. Silently projecting it is exactly the kind
    of error that surfaces 200 lines of output later as a CASSCF that will not converge, so
    the loss is measured and reported here.

    Returns the ``(nwork, nmo)`` working-basis coefficients.
    """
    c_work = basis.to_working(c_ao)
    if basis.n_dropped:
        lost = basis.outside_norm(c_ao, s_ao)
        worst = float(np.max(lost)) if lost.size else 0.0
        if worst > tol:
            log.warning("%d orbital(s) have a component outside the working basis above "
                        "%.1e (worst %.3e, orbital %d): they cannot be represented after "
                        "linear-dependence removal and have been projected.",
                        int(np.count_nonzero(lost > tol)), tol, worst, int(np.argmax(lost)))
        else:
            log.debug("orbital projection loss max %.3e (below %.1e)", worst, tol)
    return c_work


__all__ = ["OrthonormalBasis", "canonical_orthogonalization", "symmetric_orthogonalization",
           "cholesky_orthogonalization", "orthogonalize", "project_orbitals", "sqrt_overlap",
           "DEFAULT_THRESHOLD", "CONDITION_WARN"]
