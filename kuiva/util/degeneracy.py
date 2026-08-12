"""Cutting a spectrum without splitting a degenerate group.

One rule, stated once for the whole project: *any truncation of a symmetric operator
truncates whole degenerate groups, never individual vectors*, because every violation produces
a Hermitian, plausible, wrong result. This module is that rule as arithmetic, so the
orthogonalization and the X2C metric projection share one implementation
instead of two that can drift.

What the rule protects against
------------------------------
A threshold cut ``keep = values > threshold`` treats the members of a degenerate group
inequivalently the moment the group straddles the threshold: some directions of an invariant
subspace survive and the rest do not, so the retained space is no longer invariant under
whatever symmetry made them degenerate. In an overlap metric that is an anisotropic basis, and
it splits degeneracies the symmetry makes exact — the same failure the pivoted Cholesky avoids
structurally (:func:`kuiva.integrals.transform.pivoted_cholesky`'s orbit path) and the network
SVD avoids at every bond.

What is done about it, and why in that direction
-------------------------------------------------
When a group straddles the cut the **whole group is dropped**, not kept, and the reduction is
reported. The asymmetry is deliberate: the threshold exists because a direction below it
amplifies noise by ``s^{-1/2}``, so keeping the below-threshold members to complete the group
would reintroduce exactly what the threshold was protecting against, while dropping the
above-threshold members costs a few directions of a variational space that is already being
truncated. ⚠ This is *not* the network truncation's "a cut it cannot make is refused, not rounded": there the cut
size is a hard constraint (a bond dimension) and rounding it would be silent, whereas here the
constraint is a threshold and moving to the nearest boundary is a reportable choice.
``policy="raise"`` is offered for callers and tests that want the refusal instead.

⚠ **The grouping tolerance has no universal valley**, measured — `kuiva/integrals/
Measurement records relative eigenvalue gaps at *every* decade for Lu³⁺ against a
clean four-decade valley for Ar. :data:`DEFAULT_GROUP_RTOL` therefore matches
``ORBIT_DEGENERACY_RTOL``, so the two places the project groups a spectrum group it the same
way, and it is a parameter rather than a constant at both.

⚠ **Grouping is by consecutive relative gap, so it can chain**: a spectrum with no gap wider
than ``rtol`` anywhere is one group. That is correct behaviour (such a spectrum has no safe
cut) but it means a caller must handle "everything was dropped", which is why
:func:`cut_at_group_boundary` returns the count and lets the caller raise its own error.

References
----------
* The invariance argument for truncating on complete symmetry orbits, in its factorization
  form: F. Aquilante, R. Lindh, T. B. Pedersen, "Unbiased auxiliary basis sets for accurate
  two-electron integral approximations", J. Chem. Phys. 127, 114107 (2007),
  doi:10.1063/1.2777146.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: Relative gap below which two neighbouring eigenvalues are treated as degenerate. The same
#: value as :data:`kuiva.integrals.transform.ORBIT_DEGENERACY_RTOL`, deliberately (see above).
DEFAULT_GROUP_RTOL = 1.0e-6


@dataclass(frozen=True)
class GroupStraddle:
    """The group a threshold cut fell inside, and what was done about it.

    Attributes
    ----------
    start, stop : int
        Half-open index range of the group in the **descending** spectrum.
    kept_by_threshold : int
        How many of its members the bare threshold would have kept — always
        ``0 < kept_by_threshold < stop - start``, since that is what makes it a straddle.
    largest, smallest : float
        The group's extreme values, so a log line can say how tight the group was.
    """

    start: int
    stop: int
    kept_by_threshold: int
    largest: float
    smallest: float

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def relative_spread(self) -> float:
        scale = max(abs(self.largest), abs(self.smallest), 1e-300)
        return (self.largest - self.smallest) / scale


def relative_gap(a: float, b: float) -> float:
    """Relative separation of two neighbouring eigenvalues, scale-free and sign-safe."""
    scale = max(abs(float(a)), abs(float(b)), 1e-300)
    return (float(a) - float(b)) / scale


def group_bounds(values_desc: np.ndarray, index: int,
                 rtol: float = DEFAULT_GROUP_RTOL) -> Tuple[int, int]:
    """Half-open range of the degenerate group containing ``index``.

    ``values_desc`` must be sorted descending. Membership is by *consecutive* relative gap, so
    the group is the maximal run around ``index`` whose neighbours are within ``rtol``.
    """
    v = np.asarray(values_desc, dtype=float)
    n = v.size
    if not 0 <= index < n:
        raise IndexError("index {} outside a spectrum of {}".format(index, n))
    start = index
    while start > 0 and relative_gap(v[start - 1], v[start]) <= rtol:
        start -= 1
    stop = index + 1
    while stop < n and relative_gap(v[stop - 1], v[stop]) <= rtol:
        stop += 1
    return start, stop


def cut_at_group_boundary(values_desc: np.ndarray, threshold: float, *,
                          rtol: float = DEFAULT_GROUP_RTOL,
                          policy: str = "drop") -> Tuple[int, Optional[GroupStraddle]]:
    """Number of leading values to keep, moved to a degenerate-group boundary.

    Parameters
    ----------
    values_desc : ndarray
        Eigenvalues sorted **descending**.
    threshold : float
        Values at or below it are candidates for dropping — the caller's own convention,
        reproduced here exactly: ``values > threshold`` is what a bare cut would keep.
    rtol : float
        Relative gap below which neighbours are degenerate (:data:`DEFAULT_GROUP_RTOL`).
    policy : ``"drop"`` or ``"raise"``
        What to do when a group straddles the cut. ``"drop"`` removes the whole group (see
        the module docstring for why that direction); ``"raise"`` refuses.

    Returns
    -------
    (n_keep, straddle)
        ``straddle`` is ``None`` whenever the bare cut already landed on a boundary — which,
        measured over 84 (system, basis) pairs of the project's own systems and bases at every
        threshold from 1e-5 to 1e-8, is every case seen so far. **A caller therefore gets
        bitwise-unchanged behaviour except on a spectrum that actually straddles.**
    """
    if policy not in ("drop", "raise"):
        raise ValueError("unknown policy {!r}; expected 'drop' or 'raise'".format(policy))
    v = np.asarray(values_desc, dtype=float)
    keep = int(np.count_nonzero(v > threshold))
    if keep == 0 or keep == v.size:
        return keep, None
    if relative_gap(v[keep - 1], v[keep]) > rtol:
        return keep, None

    start, stop = group_bounds(v, keep - 1, rtol)
    straddle = GroupStraddle(start=start, stop=stop, kept_by_threshold=keep - start,
                             largest=float(v[start]), smallest=float(v[stop - 1]))
    if policy == "raise":
        raise ValueError(
            "the cut at {:.3e} falls inside a degenerate group of {} values spanning "
            "{:.6e}..{:.6e} (relative spread {:.2e} <= rtol {:.1e}); it would keep {} of them "
            "and drop {}, which breaks the invariance of the retained space (a truncation "
            "keeps whole degenerate groups only)".format(threshold, straddle.size, straddle.largest,
                                   straddle.smallest, straddle.relative_spread, rtol,
                                   straddle.kept_by_threshold,
                                   straddle.size - straddle.kept_by_threshold))
    return start, straddle


__all__ = ["DEFAULT_GROUP_RTOL", "GroupStraddle", "cut_at_group_boundary", "group_bounds",
           "relative_gap"]
