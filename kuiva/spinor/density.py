"""Exact real decomposition of a spinor density.

The problem
-----------
A two-component spinor has no isosurface. Its density

    rho_i(r) = |psi_i^alpha(r)|^2 + |psi_i^beta(r)|^2

is a perfectly good real, non-negative function and is what one wants to *look at* when
checking an active space — but ``sqrt(rho_i)`` is **not** in the span of the AO basis, so it
cannot be written into a molden file as an orbital. The standard counter-example is any
``m_j`` eigenfunction of a d shell: ``|l=2, m=2>`` is ``(d_{x2-y2} + i d_{xy})/sqrt(2)``,
whose density is a torus, and no single real orbital expandable in that basis squares to a
torus. ⚠ Writing ``sqrt(|c^alpha|^2 + |c^beta|^2)`` **per coefficient** — the tempting reading
— is not the square root of the density at all; it is a plausible-looking, wrong picture, and
nothing downstream can detect it.

The way out, and it is exact
----------------------------
The AO functions are **real** (time reversal conjugates coefficients only, which presumes it), so writing
``c^sigma = Re c^sigma + i Im c^sigma`` gives, with no approximation,

    rho_i(r) = sum_{k=1..4} (v_k . chi(r))^2 ,
    V = [ Re c^alpha ; Im c^alpha ; Re c^beta ; Im c^beta ]        (4, nao) real

i.e. ``rho_i = chi^T D_i chi`` with ``D_i = V^T V`` real symmetric, positive semidefinite,
and of rank **at most 4**. Diagonalizing ``D_i`` in the metric ``S`` gives at most four
S-orthonormal *real* orbitals ``u_k`` and weights ``lambda_k`` summing to 1 with

    rho_i(r) = sum_k lambda_k (u_k . chi(r))^2 .

Each ``u_k`` is an ordinary real orbital: it goes into a molden file unchanged, and a
visualizer draws it with the usual isosurfaces. The set of them reproduces the spinor density
**exactly**, which is the property that makes this honest rather than a fit.

The diagonalization is done through the ``(4, 4)`` Gram matrix ``G = V S V^T``, never an
``(nao, nao)`` eigenproblem — the rank is 4 whatever the basis size, so the cost is one
``(4, nao) x (nao, nao)`` product per spinor and nothing scales badly.

Reading the weights
-------------------
``n_components`` (the count of ``lambda_k`` above a threshold) is the diagnostic that says how
much of the story one picture tells:

* **1** — the spinor is essentially a real orbital. The SOC-free guess is exactly
  this, and its single component is the scalar MO it was expanded from.
* **2** — the normal case for a spin-orbit-coupled ``m_j`` function. The two components carry
  ``lambda = 0.5, 0.5`` for a pure ``|l, m>`` and are related by a rotation about the
  symmetry axis; both must be looked at, and neither alone is the density.
* **3-4** — genuinely low symmetry, or an unconverged/contaminated spinor.

Invariance, and what it means for what to plot
----------------------------------------------
⚠ **A single spinor's density is only defined up to the arbitrary basis choice inside a
degenerate manifold.** This is the same statement :mod:`kuiva.props.multiplet` makes about
moment matrices, and it has the same resolution: sums over a degenerate block are invariant,
because ``sum_{i in block} c_i c_i^dag`` is the block projector.

Two consequences, both acted on by :mod:`kuiva.props.molden`:

* **Kramers partners have identical densities.** ``T psi = (-conj(psi^beta),
  conj(psi^alpha))`` has ``|T psi^alpha|^2 + |T psi^beta|^2 = rho_psi``. So one picture per
  Kramers *pair* is not a simplification, it is the complete information — and it halves the
  file. :func:`kramers_pair_density` is the version that says so.
* For a manifold larger than a Kramers pair, plot the manifold, not its members.

References
----------
* Spinor densities and the loss of a real orbital picture under spin-orbit coupling:
  K. G. Dyall, K. Faegri, "Introduction to Relativistic Quantum Chemistry", Oxford University
  Press (2007), ch. 6 and 10.
* Natural-orbital decomposition of a one-particle density matrix (what the eigenproblem here
  is, applied to a rank-<=4 density): P.-O. Loewdin, Phys. Rev. 97, 1474 (1955),
  doi:10.1103/PhysRev.97.1474.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..util.logging import get_logger

log = get_logger(__name__)

#: Weights below this are dropped from a decomposition: they contribute less than this
#: fraction of the density and would be plotted as numerical noise.
DEFAULT_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class DensityComponents:
    """A spinor density as ``rho(r) = sum_k weights[k] * (components[:, k] . chi(r))^2``.

    Attributes
    ----------
    components : ndarray (nao, k) real
        S-orthonormal real orbitals. Ordered by descending weight.
    weights : ndarray (k,) real
        Non-negative, summing to the norm of the spinor (1 for a normalized one).
    """

    components: np.ndarray
    weights: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.weights.size)

    @property
    def leading_weight(self) -> float:
        """Weight of the dominant component: 1 means a real orbital, 0.5 a typical m_j pair."""
        return float(self.weights[0]) if self.weights.size else 0.0

    def density_matrix(self) -> np.ndarray:
        """The real symmetric AO density ``D`` with ``rho(r) = chi^T D chi``."""
        return (self.components * self.weights) @ self.components.T

    def __repr__(self) -> str:
        return "DensityComponents(k={}, leading={:.4f})".format(
            self.n_components, self.leading_weight)


def spinor_density_matrix(c: np.ndarray) -> np.ndarray:
    """The real symmetric AO-basis density matrix of one spinor column ``c`` ``(2*nao,)``.

    ``D = Re(c^alpha c^alpha^dag) + Re(c^beta c^beta^dag)``, which is ``V^T V`` for the ``V``
    of the module docstring. Provided for testing the decomposition against the object it is
    supposed to represent; the decomposition itself never forms it.
    """
    c = np.asarray(c).reshape(-1, 1)
    nao = c.shape[0] // 2
    v = _real_components(c, nao)
    return v.T @ v


def _real_components(c: np.ndarray, nao: int) -> np.ndarray:
    """``V`` of the module docstring: ``(4*k, nao)`` real for ``c`` of shape ``(2*nao, k)``."""
    ca, cb = c[:nao], c[nao:]
    return np.ascontiguousarray(
        np.concatenate([ca.real.T, ca.imag.T, cb.real.T, cb.imag.T]))


def decompose_density(c: np.ndarray, weights: Optional[np.ndarray] = None,
                      s_ao: Optional[np.ndarray] = None, *,
                      tolerance: float = DEFAULT_WEIGHT_TOLERANCE) -> DensityComponents:
    """Decompose the density of a **set** of spinors into real orbitals (module docstring).

    ``rho(r) = sum_i weights[i] * rho_i(r)`` for ``c`` of shape ``(2*nao, k)``, exactly, in at
    most ``4k`` real components. The general form exists because a degenerate manifold —
    where the individual spinor densities are *not* well defined — must be decomposed as a
    block; see the module docstring.

    Rank matters and is exploited: a Kramers pair has two identical spinor densities, so its
    ``(8, 8)`` Gram matrix has rank at most 4 and the extra eigenvalues come out zero and are
    dropped. The eigenproblem is always over the small index, never over ``nao``.

    Parameters
    ----------
    c : ndarray (2*nao, k) complex
        Spinors in the spin-blocked row layout ``[alpha; beta]``, over a **real** scalar basis.
    weights : ndarray (k,), optional
        Occupation numbers; ones by default. The returned weights sum to ``sum(weights)``.
    s_ao : ndarray (nao, nao), optional
        Overlap of the scalar basis. ``None`` means it is orthonormal (the working basis).
    tolerance : float
        Weights below this fraction of the total are discarded. ⚠ The decomposition is an
        identity, but this truncation is not part of it: with a nonzero ``tolerance`` the
        components reproduce the density to about ``tolerance``, not to machine precision.
        Pass ~0 where exactness is the point and leave the default where a viewer would
        otherwise be handed noise.
    """
    c = np.atleast_2d(np.asarray(c))
    if c.shape[0] % 2:
        raise ValueError("spinors have an even number of rows (alpha block then beta "
                         "block), got {}".format(c.shape[0]))
    nao = c.shape[0] // 2
    k = c.shape[1]
    w = np.ones(k) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != k:
        raise ValueError("{} weights for {} spinors".format(w.size, k))
    if np.any(w < 0.0):
        raise ValueError("occupation weights must be non-negative")

    # sqrt(w) folded into V, so the Gram matrix is the weighted density directly.
    v = _real_components(c, nao) * np.sqrt(np.tile(w, 4))[:, None]
    sv = v if s_ao is None else v @ np.asarray(s_ao)
    gram = v @ sv.T                                   # (4k, 4k), real symmetric PSD

    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    total = float(evals.sum())
    keep = evals > tolerance * max(total, 1.0)
    if not np.any(keep):                              # a zero column: return nothing, not junk
        return DensityComponents(components=np.zeros((nao, 0)), weights=np.zeros(0))
    evals, evecs = evals[keep], evecs[:, keep]

    # u_k = sum_j evecs[j, k] v_j / sqrt(lambda_k): S-orthonormal by construction.
    comp = (v.T @ evecs) / np.sqrt(evals)
    return DensityComponents(components=np.ascontiguousarray(comp),
                             weights=np.ascontiguousarray(evals))


def decompose_spinor_density(c: np.ndarray, s_ao: Optional[np.ndarray] = None, *,
                             tolerance: float = DEFAULT_WEIGHT_TOLERANCE
                             ) -> DensityComponents:
    """Decompose **one** spinor's density into at most four real orbitals.

    ``c`` is ``(2*nao,)`` or ``(2*nao, 1)``. The weights sum to the spinor's norm, 1 for a
    normalized one. See :func:`decompose_density` for the general case and the module
    docstring for what the components mean.
    """
    c = np.asarray(c)
    return decompose_density(c.reshape(-1, 1) if c.ndim == 1 else c, None, s_ao,
                             tolerance=tolerance)


def kramers_pair_density(c: np.ndarray, s_ao: Optional[np.ndarray] = None, *,
                         tolerance: float = DEFAULT_WEIGHT_TOLERANCE) -> DensityComponents:
    """Decomposition of the density shared by a spinor and its Kramers partner.

    Identical to :func:`decompose_spinor_density` on either partner — that is the point (see
    the module docstring) — and this name is how a caller records that it plotted the pair
    rather than an arbitrary member of it. ``c`` is one spinor; the weights sum to 1, i.e.
    they describe one member. The pair's *total* density is twice this.
    """
    return decompose_spinor_density(c, s_ao, tolerance=tolerance)


__all__ = ["DEFAULT_WEIGHT_TOLERANCE", "DensityComponents", "decompose_density",
           "decompose_spinor_density", "kramers_pair_density", "spinor_density_matrix"]
