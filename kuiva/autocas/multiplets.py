"""Hund ground terms of an ``l^n`` shell, and the theoretical floor of a state count.

What this is for
----------------
An automatic active space has to propose how many states to average over, and a proposal
needs a *floor*: a number that comes from theory rather than from the probe, so that a
spectrum the probe resolved badly cannot propose an obviously incomplete average. The floor
is the dimension of the ground manifold Hund's rules give the shell, and the proposal is the
first manifold boundary of the probe's spectrum at or above it.

⚠ **The floor is a floor and not an answer.** Where the ligand field is strong the ground
manifold is not the free-ion level at all, and only the probe's own spectrum can say where
the boundary is. Where the shells are coupled the honest manifold is the *product* of the
per-centre floors -- 256 states for two Dy(3+) -- which is printed even when no cap can
reach it, because that number is what a whole-manifold average would cost.

⚠ Two regimes, because ``2S+1`` and ``2J+1`` are different counts
-----------------------------------------------------------------
* **f block: the level, ``2J+1``.** Spin-orbit coupling dominates the ligand field, so the
  ground *level* is the manifold a state average has to complete.
* **d block: the spin multiplicity, ``2S+1``.** There the ligand field dominates and the
  orbital part of the free-ion term is the field's to decide, so ``2J+1`` is not a manifold
  of the molecule at all; what survives as a lower bound is the spin degeneracy. It is
  rounded up to an even number for an odd electron count, where Kramers degeneracy makes
  every level at least two-fold.
* **A radical is 2** -- one unpaired electron in one orbital, a Kramers doublet.

This is the same distinction the state-selection rule makes: a count that bounds ``2S+1``
terms generally cuts the ``2J+1`` multiplets of the same system with spin-orbit coupling on,
and the converse bites equally, so the two counts are carried separately and never
substituted for each other.

References
----------
* F. Hund, "Zur Deutung verwickelter Spektren, insbesondere der Elemente Scandium bis
  Nickel", Z. Phys. 33, 345 (1925), doi:10.1007/BF01328319 -- the maximum-``S``,
  maximum-``L`` rules and the ``J = |L - S|`` / ``L + S`` alternative below and above half
  filling.
* E. U. Condon, G. H. Shortley, "The Theory of Atomic Spectra", Cambridge University Press
  (1935), chapters on equivalent electrons -- the ground-level dimensions this reproduces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = ["HundTerm", "RADICAL_FLOOR", "coupled_floor", "hund_configuration_dimension",
           "hund_ground_term", "shell_floor",
           "term_letter"]

#: Total-orbital-angular-momentum letters. ⚠ ``J`` is skipped, by spectroscopic convention.
_LETTERS = "SPDFGHIKLMNOQRTUVWXYZ"

#: The floor of a radical centre: one unpaired electron, a Kramers doublet.
RADICAL_FLOOR = 2


def term_letter(l_total: int) -> str:
    """``5 -> "H"``. Raises above the letters the convention defines."""
    l_total = int(l_total)
    if not 0 <= l_total < len(_LETTERS):
        raise ValueError("no spectroscopic letter for L = {}".format(l_total))
    return _LETTERS[l_total]


def _half(two_x: int) -> str:
    """``5 -> "5/2"``, ``4 -> "2"`` -- a possibly half-integer quantum number, as text."""
    return "{}/2".format(two_x) if two_x % 2 else str(two_x // 2)


@dataclass(frozen=True)
class HundTerm:
    """The Hund ground term and level of ``l^n``, with both degeneracies it defines.

    ``two_s`` and ``two_j`` are **twice** ``S`` and ``J``, so everything here is integer
    arithmetic and no half-integer ever passes through a float comparison.
    """

    l: int
    n: int
    two_s: int
    l_total: int
    two_j: int

    @property
    def spin_multiplicity(self) -> int:
        """``2S+1`` -- the d-block floor."""
        return self.two_s + 1

    @property
    def level_dimension(self) -> int:
        """``2J+1`` -- the f-block floor, i.e. the dimension of the ground level."""
        return self.two_j + 1

    @property
    def term_dimension(self) -> int:
        """``(2S+1)(2L+1)`` -- the whole term, i.e. the spin-free manifold."""
        return (self.two_s + 1) * (2 * self.l_total + 1)

    @property
    def term(self) -> str:
        """``"6H"`` -- the term symbol without the level."""
        return "{}{}".format(self.spin_multiplicity, term_letter(self.l_total))

    @property
    def symbol(self) -> str:
        """``"6H15/2"`` -- term and level, the spelling a report prints."""
        return "{}{}".format(self.term, _half(self.two_j))

    def __repr__(self) -> str:
        return "HundTerm({}^{} -> {}, 2J+1 = {}, 2S+1 = {})".format(
            "spdfghi"[self.l], self.n, self.symbol, self.level_dimension,
            self.spin_multiplicity)


def hund_ground_term(l: int, n: int) -> HundTerm:
    """The ground term and level of ``n`` equivalent electrons in a shell of ``l``.

    ``S`` is maximal (``min(n, 4l+2-n)/2``), ``L`` is maximal at that ``S`` -- the electrons
    fill the ``m_l`` ladder from the top, so ``L = k*l - k(k-1)/2`` for the ``k`` unpaired
    electrons -- and ``J = |L - S|`` below half filling, ``L + S`` above it, ``S`` at exactly
    half filling (where ``L = 0``). A closed or empty shell gives ``1S0``, dimension one,
    which is the correct statement that it contributes no manifold at all.
    """
    l = int(l)
    n = int(n)
    if l < 0:
        raise ValueError("angular momentum cannot be negative; got {}".format(l))
    capacity = 4 * l + 2
    if not 0 <= n <= capacity:
        raise ValueError("a shell of l = {} holds 0..{} electrons; got {}"
                         .format(l, capacity, n))
    unpaired = min(n, capacity - n)                     # electrons or holes, whichever fewer
    two_s = unpaired
    # Maximum L at maximum S: the unpaired electrons occupy m_l = l, l-1, ..., and below
    # half filling that is all of them. Above half filling the shell is a hole shell of the
    # same count, so the same expression in the hole number gives the same L -- which is the
    # particle-hole symmetry of the term structure, not a separate rule.
    k = unpaired
    l_total = k * l - (k * (k - 1)) // 2
    if n == 0 or n == capacity:
        l_total, two_s = 0, 0
    if 2 * n == capacity:
        two_j = two_s                                   # L = 0 exactly at half filling
    elif 2 * n < capacity:
        two_j = abs(2 * l_total - two_s)
    else:
        two_j = 2 * l_total + two_s
    return HundTerm(l=l, n=n, two_s=two_s, l_total=l_total, two_j=two_j)


def shell_floor(l: int, n: int, *, regime: Optional[str] = None) -> int:
    """The floor a shell of ``n`` electrons in ``l`` contributes to a state count.

    ``regime`` is ``"level"`` (``2J+1``), ``"spin"`` (``2S+1``, rounded up to even for an odd
    electron count) or ``None`` to take the one the module docstring assigns to the shell:
    the level for ``f`` and above, the spin multiplicity for ``d`` and below.
    """
    term = hund_ground_term(l, n)
    if regime is None:
        regime = "level" if int(l) >= 3 else "spin"
    if regime == "level":
        return term.level_dimension
    if regime == "spin":
        floor = term.spin_multiplicity
        # An odd electron count has no non-degenerate level: Kramers makes every one of them
        # at least two-fold, so an odd floor there would be a count that cannot be a manifold
        # boundary. (For an odd n the spin multiplicity is already even; the rounding is what
        # states the rule rather than relying on that.)
        if int(n) % 2 and floor % 2:
            floor += 1
        return floor
    raise ValueError("regime is 'level', 'spin' or None; got {!r}".format(regime))


def hund_configuration_dimension(l: int, n: int) -> int:
    """Determinants of the configurations that hold the Hund ground term of ``l^n``.

    The high-spin term needs the largest possible number of singly occupied orbitals,
    ``u = min(n, 2(2l+1) - n)``, with the remaining ``(n - u) / 2`` orbitals doubly occupied:
    ``C(2l+1, u) C(2l+1-u, (n-u)/2) 2^u`` determinants. ``d^5`` is 32, ``d^1`` and ``d^9`` are
    10, ``f^9`` is 672, a closed shell is 1.

    ⚠ **The product over the sites is a LOWER bound on what a determinant list must hold to
    carry a coupled system's exchange manifold** -- every state of it has weight in that
    product space, and the charge-transfer determinants that mediate the coupling come on top.
    A selected CI capped below it cannot represent the manifold at all, and returns a spectrum
    that is an artefact of which determinants it happened to select.
    """
    from math import comb

    l, n = int(l), int(n)
    m = 2 * l + 1
    if not 0 <= n <= 2 * m:
        raise ValueError("{} electrons do not fit a shell of l = {}".format(n, l))
    u = min(n, 2 * m - n)
    return int(comb(m, u) * comb(m - u, (n - u) // 2) * 2 ** u)


def coupled_floor(floors: Iterable[int]) -> int:
    """The product of per-centre floors -- the dimension of the exchange manifold.

    ⚠ It grows as fast as it sounds (two Dy(3+): ``16 x 16 = 256``) and is reported even
    where it exceeds every practical cap, because it is the honest statement of what a
    whole-manifold average would be. What to do about a number no solver can reach is a
    decision for whoever reads it, not a number to quietly reduce.
    """
    product = 1
    empty = True
    for f in floors:
        f = int(f)
        if f < 1:
            raise ValueError("a floor is at least one state; got {}".format(f))
        product *= f
        empty = False
    if empty:
        raise ValueError("no centres: there is no manifold to bound")
    return product
