"""The bond-dimension series and the ``E(w_disc -> 0)`` extrapolation.

The number a tensor-network paper quotes is not the energy at the largest affordable bond
dimension but the extrapolate of a **series**: solve at ascending caps, record each cap's
discarded weight, and fit the near-linear relation ``E(w) = E0 + c w`` — the standard
quantum-chemistry DMRG practice (Chan & Head-Gordon, J. Chem. Phys. 116, 4462 (2002);
reviewed in Olivares-Amaya et al., J. Chem. Phys. 142, 034102 (2015)). This module is that
protocol as a driver: one call runs the series at fixed integrals, warm-starting each cap
from the previous one, and returns the fit **with the series beside it**.

⚠ Three rules, all reporting discipline:

* **The extrapolate is never presented without its series.** The fit residual and the raw
  ``(D, w_disc, E)`` rows are part of the result and of the report; an extrapolate quoted
  alone is a number whose error bar has been thrown away.
* **A series the truncation never bit is reported as exact, not extrapolated.** With every
  discarded weight zero there is nothing to fit and the largest-cap energies *are* the
  answer; fitting rounding noise would manufacture a spurious correction.
* **Energies exclude ``e_core``**, like every energy in this layer (the sweep convention);
  the caller adds it when reporting molecular energies, and the extrapolation is
  indifferent to the constant.

⚠ What the fit means, and what it cannot: the linear relation holds asymptotically for
converged sweeps at each cap, and the residual measures how linear this series actually
was. A residual comparable to the correction itself says the caps are too small to
extrapolate from — the honest reading is "run a larger series", not "quote the fit".
Per-root fits are provided because the roots of a state average converge at different
rates; the state-average energy's fit is not the weighted sum of the per-root fits unless
the weights are cap-independent, which equalized weights on a stable spectrum are.

Everything here is orchestration over :func:`kuiva.dmrg.sweep.solve_ttn`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from ..util.timing import timer
from .sweep import SweepResult, TTNState, random_state, solve_ttn
from .ttno import TTNO

log = get_logger(__name__)


@dataclass
class BondSeriesResult:
    """A bond-dimension series and its ``w_disc -> 0`` extrapolation (module docstring)."""

    bonds: np.ndarray                #: (n_caps,) the requested caps, ascending
    discarded: np.ndarray            #: (n_caps,) largest ensemble w_disc, final sweep each
    energies: np.ndarray             #: (n_caps, n_roots) per-root energies [Eh], no e_core
    weights: np.ndarray              #: (n_roots,) equalized weights at the largest cap
    extrapolated: np.ndarray         #: (n_roots,) fitted E(w_disc -> 0) [Eh]
    residuals: np.ndarray            #: (n_roots,) max |fit - E| over the series [Eh]
    slopes: np.ndarray               #: (n_roots,) dE/dw of the fit [Eh]
    #: The truncation never bit (every w_disc is zero): :attr:`extrapolated` is then the
    #: largest cap's energies verbatim and the fit fields are zero — exact, not fitted.
    exact: bool = False
    #: The converged state at the largest cap, for downstream reuse (RDMs, properties).
    state: Optional[TTNState] = None
    #: The last cap's full sweep result.
    final: Optional[SweepResult] = None

    @property
    def sa_energies(self) -> np.ndarray:
        """(n_caps,) state-averaged energy per cap [Eh]."""
        return self.energies @ self.weights

    @property
    def sa_extrapolated(self) -> float:
        """The state-averaged extrapolate [Eh]."""
        return float(np.dot(self.extrapolated, self.weights))

    def report(self, logger=None) -> None:
        """The series table with the extrapolate beside it — never the latter alone."""
        logger = logger or log
        out.subsection(logger, "Bond-dimension series")
        table = out.Table(logger, [
            out.col_count("cap D", 8), out.col_count("used", 6),
            out.col_sci("w_disc"), out.col_energy("E_SA [Eh]"),
        ])
        table.start()
        used = getattr(self, "_used_dims", [0] * self.bonds.size)
        for i, cap in enumerate(self.bonds):
            table.row(int(cap), int(used[i]), float(self.discarded[i]),
                      float(self.sa_energies[i]))
        table.end("energies exclude e_core (the sweep convention)")
        if self.exact:
            out.note(logger, "the truncation never bit (every w_disc is zero): the "
                             "largest cap is exact and there is nothing to extrapolate")
            return
        table = out.Table(logger, [
            out.col_count("root", 6), out.col_energy("E(w->0) [Eh]"),
            out.col_sci("fit residual"), out.col_sci("dE/dw"),
        ])
        table.start()
        for r in range(self.extrapolated.size):
            table.row(r, float(self.extrapolated[r]), float(self.residuals[r]),
                      float(self.slopes[r]))
        table.end()
        worst = float(np.max(self.residuals))
        correction = float(np.max(np.abs(self.extrapolated - self.energies[-1])))
        if correction > 0.0 and worst > 0.5 * correction:
            log.warning("the extrapolation residual (%.3e Eh) is comparable to the "
                        "correction it claims (%.3e Eh): this series is not in the "
                        "linear regime, so run larger caps rather than quoting the fit",
                        worst, correction)

    def __repr__(self) -> str:
        return "BondSeriesResult(D={}, E_SA(w->0)={:.10f} Eh, exact={})".format(
            self.bonds.tolist(), self.sa_extrapolated, self.exact)


def bond_series(ttno: TTNO, n_elec: int, bonds: Sequence[int], *,
                n_roots: int = 1, weights: Optional[Sequence[float]] = None,
                initial_state: Optional[TTNState] = None,
                max_sweeps: int = 30, conv_tol: float = 1e-9,
                davidson_tol: float = 1e-8, trunc_tol: float = 0.0,
                expansion: float = 0.0, expansion_sweeps: int = 6,
                on_split: str = "raise", seed: int = 0,
                report: bool = True) -> BondSeriesResult:
    """Solve the same problem at every cap in ``bonds`` (ascending) and fit the series.

    Each cap's solve warm-starts from the previous cap's converged state — ascending
    order is required, since a descending step would truncate a converged state and call
    the damage a data point. ``expansion`` applies to the **first** (cold) solve only,
    for the same reason it does on the solver. A solve that does not converge fails the
    series: an unconverged energy is not a point on the ``E(w)`` curve.

    Returns a :class:`BondSeriesResult`; per-root linear fits of ``E`` against the
    largest final-sweep discarded weight of each cap's solve.
    """
    caps = [int(d) for d in bonds]
    if len(caps) < 2:
        raise ValueError("a series needs at least two caps, got {}".format(bonds))
    if any(b <= a for a, b in zip(caps, caps[1:])):
        raise ValueError("bonds must be strictly ascending (each solve warm-starts the "
                         "next; a descending step would truncate a converged state), "
                         "got {}".format(bonds))
    state = initial_state
    if state is None:
        state = random_state(ttno, int(n_elec), caps[0], n_roots=int(n_roots),
                             rng=np.random.default_rng(seed))
    energies: List[np.ndarray] = []
    discarded: List[float] = []
    used_dims: List[int] = []
    result = None
    with timer("bond-dimension series"):
        for i, cap in enumerate(caps):
            result = solve_ttn(ttno, state, max_sweeps=max_sweeps, conv_tol=conv_tol,
                               trunc_tol=trunc_tol, max_bond=cap, weights=weights,
                               n_elec=int(n_elec), boundary_check=0,
                               davidson_tol=davidson_tol, on_split=on_split,
                               expansion=expansion if i == 0 else 0.0,
                               expansion_sweeps=expansion_sweeps, report=False)
            if not result.converged:
                raise RuntimeError(
                    "the D = {} member of the series did not converge in {} sweeps; an "
                    "unconverged energy is not a point on the E(w_disc) curve, so the "
                    "series stops rather than fitting it".format(cap, max_sweeps))
            energies.append(np.asarray(result.energies, dtype=float).copy())
            discarded.append(float(result.max_discarded))
            used_dims.append(int(result.max_bond_dim))
            log.debug("bond series D=%d: E_SA=%.12f, w_disc=%.3e", cap,
                      float(np.dot(result.weights, result.energies)), discarded[-1])

    e = np.asarray(energies)                                   # (n_caps, n_roots)
    w_disc = np.asarray(discarded)
    w_final = np.asarray(result.weights, dtype=float)
    n_r = e.shape[1]
    exact = bool(np.all(w_disc == 0.0))
    if exact or float(np.ptp(w_disc)) == 0.0:
        extrapolated = e[-1].copy()
        residuals = np.zeros(n_r)
        slopes = np.zeros(n_r)
        if not exact:
            log.warning("every cap in the series produced the same discarded weight "
                        "(%.3e); there is no abscissa to extrapolate along, so the "
                        "largest cap's energies are returned unfitted", float(w_disc[0]))
    else:
        extrapolated = np.empty(n_r)
        residuals = np.empty(n_r)
        slopes = np.empty(n_r)
        for r in range(n_r):
            slope, intercept = np.polyfit(w_disc, e[:, r], 1)
            extrapolated[r] = intercept
            slopes[r] = slope
            residuals[r] = float(np.max(np.abs(slope * w_disc + intercept - e[:, r])))
    series = BondSeriesResult(bonds=np.asarray(caps), discarded=w_disc, energies=e,
                              weights=w_final, extrapolated=extrapolated,
                              residuals=residuals, slopes=slopes, exact=exact,
                              state=state, final=result)
    series._used_dims = used_dims
    if report:
        series.report(log)
    return series


__all__ = ["BondSeriesResult", "bond_series"]
