"""Orbital entanglement measures.

These are the quantities the cheap CI exists to produce: the single-orbital
entropy says *which* spinors are correlated, and the mutual information says *how they are
connected*, which is what fixes the ordering — and eventually the topology — of the tensor
network.

Spinors are two-state modes, which makes this much cheaper than the spatial-orbital case
usually described in the DMRG literature. The one-mode reduced density matrix is

    rho_p = diag(1 - n_p, n_p),          n_p = gamma_pp

so the single-orbital entropy is a function of one occupation number. The two-mode density
matrix is 4x4 and, for a particle-number-conserving state, block diagonal in the occupation
sectors {00}, {10, 01}, {11}::

    rho_pq = diag_blocks[ 1 - n_p - n_q + <n_p n_q> ,
                          [[n_p - <n_p n_q>, gamma_pq], [gamma_qp, n_q - <n_p n_q>]] ,
                          <n_p n_q> ]

so **everything needed is the 1-RDM and the occupation correlations `<n_p n_q>`** — never the
full ``n^4`` 2-RDM. ``<n_p n_q>`` is diagonal in a determinant basis
(:func:`kuiva.ci.strings.occupation_correlations`), so the entanglement analysis of a
determinant CI is essentially free. That is why a cheap CI can serve a DMRG that it could
never afford to solve.

Only ``|gamma_pq|`` enters the eigenvalues of the one-particle block, so the result is
independent of the arbitrary phase of each spinor — as any entanglement measure must be.

References
----------
* Orbital entanglement entropies and mutual information as DMRG diagnostics: J. Rissler,
  R. M. Noack, S. R. White, "Measuring orbital interaction using quantum information theory",
  Chem. Phys. 323, 519 (2006), doi:10.1016/j.chemphys.2005.10.018; O. Legeza, J. Solyom,
  Phys. Rev. B 68, 195116 (2003), doi:10.1103/PhysRevB.68.195116.
* Entanglement-based orbital ordering and the Fiedler vector: G. Barcza, O. Legeza,
  K. H. Marti, M. Reiher, "Quantum-information analysis of electronic states of different
  molecular structures", Phys. Rev. A 83, 012508 (2011), doi:10.1103/PhysRevA.83.012508;
  M. Fiedler, Czechoslovak Math. J. 23, 298 (1973).
* Entanglement-driven active-space selection: C. J. Stein, M. Reiher, "Automated Selection of
  Active Orbital Spaces", J. Chem. Theory Comput. 12, 1760 (2016),
  doi:10.1021/acs.jctc.6b00156.
* Review of the measures and their conventions: K. Boguslawski, P. Tecmer, "Orbital
  entanglement in quantum chemistry", Int. J. Quantum Chem. 115, 1289 (2015),
  doi:10.1002/qua.24832.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger

log = get_logger(__name__)

#: Occupations closer than this to 0 or 1 contribute no entropy and are clamped, so that
#: ``0 * log(0)`` is 0 rather than a NaN that would propagate into every ordering decision.
_EPS = 1e-12


def _plogp(x: np.ndarray) -> np.ndarray:
    """``-x ln x``, with the ``x -> 0`` limit taken as 0."""
    x = np.clip(np.asarray(x, dtype=float).real, 0.0, 1.0)
    out_ = np.zeros_like(x)
    m = x > _EPS
    out_[m] = -x[m] * np.log(x[m])
    return out_


def single_orbital_entropy(gamma: np.ndarray) -> np.ndarray:
    """Single-orbital (von Neumann) entropy ``s(1)_p`` for every spinor [nats].

    Ranges from 0 (empty or fully occupied — an uncorrelated spinor) to ``ln 2`` (occupation
    1/2 — maximally entangled with the rest). This is the quantity active-space selection
    thresholds on: a spinor with ``s(1) ~ 0`` carries no correlation and belongs in the
    inactive or virtual space.
    """
    n = np.clip(np.real(np.diag(gamma)), 0.0, 1.0)
    return _plogp(n) + _plogp(1.0 - n)


def two_orbital_entropy(gamma: np.ndarray, nn: np.ndarray) -> np.ndarray:
    """Two-orbital entropy ``s(2)_pq`` [nats] from the 1-RDM and ``<n_p n_q>``.

    ``nn`` is the occupation correlation matrix (:func:`kuiva.ci.strings.
    occupation_correlations`). The diagonal is set to ``s(1)``, since a mode with itself is
    just the one-mode state.
    """
    gamma = np.asarray(gamma)
    nn = np.asarray(nn, dtype=float).real
    n = np.clip(np.real(np.diag(gamma)), 0.0, 1.0)
    npq = nn                                        # <n_p n_q>
    a = 1.0 - n[:, None] - n[None, :] + npq         # both empty
    d = npq                                         # both occupied
    b = n[:, None] - npq                            # p occupied, q empty
    c = n[None, :] - npq                            # q occupied, p empty
    # one-particle sector: 2x2 Hermitian block [[b, gamma_pq], [conj, c]]
    off = np.abs(gamma) ** 2
    half = 0.5 * (b + c)
    disc = np.sqrt(np.maximum(0.25 * (b - c) ** 2 + off, 0.0))
    lam1, lam2 = half + disc, half - disc
    s2 = _plogp(a) + _plogp(d) + _plogp(lam1) + _plogp(lam2)
    np.fill_diagonal(s2, single_orbital_entropy(gamma))
    return s2


def mutual_information(gamma: np.ndarray, nn: np.ndarray) -> np.ndarray:
    """Orbital mutual information ``I_pq = s(1)_p + s(1)_q - s(2)_pq`` [nats], zero diagonal.

    Non-negative by subadditivity of the von Neumann entropy; a small negative value from
    rounding is clipped, and a large one is reported because it means the input 1-RDM and
    ``<n_p n_q>`` are not from the same state.
    """
    s1 = single_orbital_entropy(gamma)
    s2 = two_orbital_entropy(gamma, nn)
    info = s1[:, None] + s1[None, :] - s2
    np.fill_diagonal(info, 0.0)
    worst = float(info.min())
    if worst < -1e-6:
        log.warning("mutual information has a significantly negative entry (%.2e); the "
                    "1-RDM and the occupation correlations are inconsistent, which usually "
                    "means they came from different states or different weights", worst)
    return np.clip(info, 0.0, None)


def fiedler_order(info: np.ndarray) -> np.ndarray:
    """Orbital ordering from the Fiedler vector of the mutual-information graph.

    Returns the permutation that sorts spinors so strongly entangled ones sit close together —
    which is what a matrix-product state needs, since its cost is set by how far entanglement
    has to travel along the chain. Minimizes ``sum_pq I_pq (p - q)^2`` in the relaxed sense
    (Barcza, Legeza, Marti & Reiher 2011): the Fiedler vector is the eigenvector of the graph
    Laplacian ``L = D - I`` belonging to its smallest non-zero eigenvalue, and sorting by it
    is the standard spectral seriation.

    A disconnected entanglement graph (several non-interacting fragments) gives a degenerate
    zero eigenvalue; the ordering is then arbitrary *between* fragments, which is correct — no
    ordering can help, and the topology should be a tree, not a chain.
    """
    info = np.asarray(info, dtype=float)
    n = info.shape[0]
    if n < 3:
        return np.arange(n)
    lap = np.diag(info.sum(axis=1)) - info
    w, v = np.linalg.eigh(lap)
    n_zero = int(np.count_nonzero(w < 1e-10))
    if n_zero > 1:
        log.info("entanglement graph has %d disconnected components; the Fiedler ordering "
                 "is only defined within each", n_zero)
    return np.argsort(v[:, min(n_zero, n - 1)])


def entanglement_report(gamma: np.ndarray, nn: np.ndarray, *, logger=None,
                        labels: Optional[np.ndarray] = None,
                        threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Log the standard entanglement block and return ``(s1, I)``."""
    logger = logger or log
    s1 = single_orbital_entropy(gamma)
    info = mutual_information(gamma, nn)
    occ = np.real(np.diag(gamma))
    strong = int(np.count_nonzero(s1 > threshold))
    out.entries(logger, [
        ("spinors analysed", int(s1.size)),
        ("max single-orbital entropy", float(s1.max()) if s1.size else 0.0, "nats", "",
         "{:.4f}"),
        ("strongly entangled spinors", strong, "", "s(1) > {:.2f}".format(threshold)),
        ("total correlation, sum I_pq / 2", 0.5 * float(info.sum()), "nats", "", "{:.4f}"),
        ("max mutual information", float(info.max()) if info.size else 0.0, "nats", "",
         "{:.4f}"),
    ])
    order = np.argsort(-s1)[:min(12, s1.size)]
    table = out.Table(logger, [out.Column("spinor", "{:d}", 7),
                               out.Column("occupation", "{:.6f}", 12),
                               out.Column("s(1) [nats]", "{:.6f}", 12),
                               out.Column("max I_pq", "{:.6f}", 11),
                               out.Column("partner", "{:d}", 8)])
    def _label(i):
        return int(labels[i]) if labels is not None else int(i)

    table.start("most entangled spinors")
    for p in order:
        partner = int(np.argmax(info[p])) if info.shape[0] > 1 else int(p)
        # Both columns must use the same numbering: printing a global spinor index next to a
        # local partner index is the kind of table that gets misread for months.
        table.row(_label(p), float(occ[p]), float(s1[p]), float(info[p, partner]),
                  _label(partner))
    table.end()
    return s1, info


__all__ = ["single_orbital_entropy", "two_orbital_entropy", "mutual_information",
           "fiedler_order", "entanglement_report"]
