"""The two-electron picture change of a converged four-component mean field.

This is the subtraction that both mean-field X2C methods reduce to, and it is implemented
**once** here because the two differ only in what the four-component calculation was run on:

* :mod:`kuiva.amf` runs it on an **isolated atom**, once per element, and assembles the
  atom-diagonal blocks into a molecule — X2CAMF;
* :func:`kuiva.interface.pyscf_bridge.molecular_mean_field` runs it on the **whole molecule**
  — X2C-mmf, which is the same physics with no atomic approximation and no assembly.

⚠ **Two copies of this formula would be the single most dangerous duplication in the project.**
Getting the *subtracted* term wrong produces an answer that is Hermitian, time-reversal even,
of a plausible magnitude, and wrong; it took an external four-component reference to find such
an error once already. One implementation, two callers, and the atomic path's committed
reference numbers therefore also validate the molecular one.

The accounting
--------------
``G~`` is the picture-changed four-component mean field, ``D~`` the two-component density
behind the four-component one, and ``G_nr`` the ordinary non-relativistic ``J - K``::

    dG = G~ - G_nr[ D~ ]  +  ( h1e(X_2e) - h1e(X_1e) )

The subtraction is there because the Hamiltonian this is added to already carries the
**untransformed** non-relativistic Coulomb operator; adding the whole of ``G~`` double-counts
massively. The third term compensates for the decoupling being defined by the converged Fock
rather than the bare one-electron problem, so that the total is the Hamiltonian that convention
implies rather than a mixture of two. The full derivation, the density convention (``R^-1 D
R^-dag``, *not* ``R^dag D R``) and the evidence for each choice are in
:mod:`kuiva.amf.decouple`, which is where they were established.

⚠ **``dG`` is a mean field: it enters a Fock operator whole and an energy with a ½.** Adding it
to ``hcore`` and reading an SCF total energy double-counts it exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spinor.expand import time_reversal_residual
from ..util.logging import get_logger
from .decouple import (FourComponentBlocks, decoupling_matrices, picture_change,
                       two_component_density)

log = get_logger(__name__)

#: Largest time-reversal-odd fraction of a correction that is still usable. Above this the
#: correction is **refused** rather than warned about: projecting the odd part out keeps Kramers
#: degeneracy exact and makes the result *look* fine, but the even half — the half that gets
#: used — is contaminated by the same amount. Well-conditioned problems sit at 1e-9 or below, so
#: the limit is six orders above anything healthy and is a statement about validity, not a
#: tolerance to tune.
TIME_REVERSAL_LIMIT = 1e-3


@dataclass(frozen=True)
class MeanFieldPictureChange:
    """The raw two-component correction ``dG``, with the diagnostics needed to judge it."""

    dg: np.ndarray
    #: ``max |dG|`` [Eh].
    scale: float
    #: Absolute size of the time-reversal-**odd** part that will be projected out [Eh], and its
    #: size relative to the *terms* of the subtraction rather than to their difference.
    tr_residual: float
    tr_residual_rel: float
    #: ``max |G~|`` and ``max |G_nr|``. Their ratio to :attr:`scale` says how much cancellation
    #: the correction rests on, which decides whether it is meaningful at the precision claimed.
    transformed_scale: float
    subtracted_scale: float
    #: ``max |h1e(X_2e) - h1e(X_1e)|`` [Eh] — the entire visible consequence of defining the
    #: decoupling by the Fock. It must vanish in both exact limits.
    compensation_scale: float

    @property
    def cancellation(self) -> float:
        """``max(|G~|, |G_nr|) / |dG|`` — 1 means no cancellation, 1e12 means rounding error."""
        biggest = max(self.transformed_scale, self.subtracted_scale)
        return biggest / self.scale if self.scale > 0.0 else float("inf")


def mean_field_picture_change(hcore: FourComponentBlocks, overlap: FourComponentBlocks,
                              veff: FourComponentBlocks, density_ll: np.ndarray,
                              light_speed: float, coulomb_mean_field,
                              *, label: str = "the system") -> MeanFieldPictureChange:
    """``dG`` for one converged four-component calculation, atomic or molecular.

    Parameters
    ----------
    hcore, overlap, veff : FourComponentBlocks
        The converged four-component core Hamiltonian, metric and two-electron mean field.
    density_ll : ndarray
        Large-component block of the converged four-component density.
    coulomb_mean_field : callable
        ``dm -> J - K``: the **non-relativistic** two-electron mean field of a two-component
        density, over **exactly** the basis the four-component solve used. ⚠ Supplied by the
        caller rather than built here, because a separately constructed basis would make the
        subtraction a difference between two slightly different things.
    label : str
        What this is, for diagnostics — an element symbol or a molecular formula.

    Raises ``RuntimeError`` if the result is more than :data:`TIME_REVERSAL_LIMIT` odd.
    """
    fock = FourComponentBlocks(ll=hcore.ll + veff.ll, ls=hcore.ls + veff.ls,
                               sl=hcore.sl + veff.sl, ss=hcore.ss + veff.ss)
    x, r = decoupling_matrices(fock, overlap, light_speed)
    g_transformed = picture_change(veff, x, r)
    d_2c = two_component_density(density_ll, r)
    g_subtracted = np.asarray(coulomb_mean_field(d_2c))

    # ⚠ The compensating one-electron term, and it is not optional. With ``X`` taken from the
    # Fock, the one-electron Hamiltonian this correction is added to — built with the
    # one-electron ``X`` — is no longer the picture-changed ``h`` belonging to this decoupling.
    # It vanishes identically when the two decouplings coincide (a vanishing mean field, and
    # the ``c -> inf`` limit), which is what keeps the exact-limit tests exact.
    x_1e, r_1e = decoupling_matrices(hcore, overlap, light_speed)
    h_compensation = picture_change(hcore, x, r) - picture_change(hcore, x_1e, r_1e)
    dg = g_transformed - g_subtracted + h_compensation

    transformed_scale = float(np.max(np.abs(g_transformed))) if g_transformed.size else 0.0
    subtracted_scale = float(np.max(np.abs(g_subtracted))) if g_subtracted.size else 0.0
    compensation_scale = float(np.max(np.abs(h_compensation))) if h_compensation.size else 0.0

    # ⚠ Measured against the **terms**, not against their difference. Symmetry breaking enters
    # through the matrix square roots inside ``R``, so it scales with ``G~``; dividing it by
    # ``|dG|`` would make the ``c -> inf`` limit — where ``dG`` correctly goes to zero — report
    # a *growing* relative asymmetry and warn about a test that is passing perfectly.
    residual, _ = time_reversal_residual(dg)
    residual_rel = residual / (max(transformed_scale, subtracted_scale,
                                   compensation_scale) or 1.0)
    if residual_rel > TIME_REVERSAL_LIMIT:
        raise RuntimeError(
            "the mean-field picture change for {} is {:.1e} time-reversal odd relative to the "
            "terms it is built from ({:.2e} Eh of {:.2e} Eh), against a limit of {:.0e}. The "
            "odd part would be projected out and the result would look entirely plausible, but "
            "the even part is contaminated by the same amount, so this correction is not "
            "usable. The cause is almost always a numerically singular metric in a decontracted "
            "basis: check max|X| from decoupling_matrices (order 10 is healthy, 1e3 is not) and "
            "see METRIC_LINDEP_THRESHOLD in kuiva.x2c.decouple.".format(
                label, residual_rel, residual, max(transformed_scale, subtracted_scale),
                TIME_REVERSAL_LIMIT))
    if residual_rel > 1e-8:
        log.warning("the mean-field picture change for %s deviates from time-reversal symmetry "
                    "by %.2e Eh (%.1e relative). The odd part has been projected out so Kramers "
                    "degeneracy stays exact, but a large value here means the decoupling is "
                    "poorly conditioned in this basis.", label, residual, residual_rel)

    return MeanFieldPictureChange(
        dg=dg, scale=float(np.max(np.abs(dg))) if dg.size else 0.0,
        tr_residual=residual, tr_residual_rel=residual_rel,
        transformed_scale=transformed_scale, subtracted_scale=subtracted_scale,
        compensation_scale=compensation_scale)


__all__ = ["TIME_REVERSAL_LIMIT", "MeanFieldPictureChange", "mean_field_picture_change"]
