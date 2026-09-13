"""Reading a state count off a probe's spectrum, and the window that says the same thing.

Two things live here, and the first is the one other modules use
-----------------------------------------------------------------
:func:`manifold_boundary` chains a spectrum at the manifold gap and returns the first block
boundary **at or above a floor**. It is the one definition of "where does the ground manifold
end" in this package: the round loop reads its target manifold through it (the states inside
the ground manifold, plus the gap to the next one, are what a feature class has to move), and
the proposal below reads its count through it. Two definitions of a manifold boundary would
be two answers to the only question the whole protocol asks.

⚠ **The chaining is not a fourth implementation.** :func:`kuiva.util.window.chain_blocks` is
the project's one consecutive-gap grouping -- the energy window's manifold rule, the RDM
gate's degenerate blocks and this are the same function at three tolerances -- and the
default gap here is :data:`~kuiva.util.window.DEFAULT_MANIFOLD_GAP_CM`, which is also the
state-average boundary diagnostic's threshold. A proposal is therefore a clean boundary by
that diagnostic's own standard, by construction rather than by luck.

What a proposal is, and what it is not
--------------------------------------
:func:`propose_roots` offers a count and an equivalent :class:`~kuiva.util.window.EnergyWindow`
whose cutoff sits in the middle of the gap the count was read at, so the production stage's
ladder re-resolves the same boundary against **its own** spectrum. Both are offered because
they fail differently: a count is what the state-averaging gate consumes and works on every
route today, while a window is what survives the production solver moving the spectrum.

⚠ **It is a proposal and nothing downstream trusts it.** The count is read off a *qualitative*
probe -- a truncated CI in a truncated space -- so what makes it a state count is the
production stage's own machinery: the state-averaging gate's refusal to split a degenerate
block, the boundary diagnostic at both ends of the optimization, and the window's ladder with
its converged witness roots. This module's job is to arrive at those checks with a number
that is not obviously wrong.

Three outcomes, all stated
--------------------------
* **A boundary was found** at or above the floor: the count, its gap, and the window.
* **The count exceeds the cap** (:data:`~kuiva.util.window.DEFAULT_MAX_STATES` by default):
  no count is proposed. The window is offered with the cap on it and the report says what the
  whole manifold would have been -- a coupled pair of Dy(3+) is 256 states, and quietly
  proposing 64 of them would be proposing a cut inside an exchange manifold.
* **No boundary in the roots solved**: the floor is proposed and marked
  ``boundary not found``. ⚠ Never a rounded count: the probe did not see where the manifold
  ends, and a number invented at that point is exactly the self-reinforcing cut inside a
  near-degenerate manifold that the state-selection rule exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from ..util.units import HARTREE_TO_CM
from ..util.window import (DEFAULT_MANIFOLD_GAP_CM, DEFAULT_MAX_STATES, EnergyWindow,
                           chain_blocks)

log = get_logger(__name__)

__all__ = ["ManifoldBoundary", "RootProposal", "manifold_boundary", "propose_roots",
           "target_spectrum"]


@dataclass(frozen=True)
class ManifoldBoundary:
    """Where a spectrum's ground manifold ends, at or above a floor.

    ``count`` is the boundary when :attr:`found`; otherwise it is the number of roots solved,
    which is a *lower* bound and is labelled as one everywhere it is printed.
    """

    count: int
    found: bool
    floor: int
    #: ``E[count] - E[count-1]`` [cm^-1] -- the gap the boundary was read at. ``None`` when no
    #: boundary was found, and when the count is the whole solved spectrum.
    gap_cm: Optional[float]
    #: ``(start, stop)`` blocks of the chained spectrum, for the report.
    blocks: Tuple[Tuple[int, int], ...] = ()
    manifold_gap_cm: float = DEFAULT_MANIFOLD_GAP_CM

    @property
    def spans_solved(self) -> bool:
        return not self.found

    def describe(self) -> str:
        if not self.found:
            return ("boundary not found in {} roots (the floor of {} is a lower bound)"
                    .format(self.count, self.floor))
        return ("{} states, gap {:.2f} cm^-1 at the cut".format(self.count, self.gap_cm)
                if self.gap_cm is not None else "{} states, the whole solved spectrum"
                .format(self.count))


def manifold_boundary(spectrum_cm: Sequence[float], floor: int, *,
                      gap_cm: float = DEFAULT_MANIFOLD_GAP_CM) -> ManifoldBoundary:
    """The first manifold boundary of ``spectrum_cm`` at or above ``floor``.

    ``spectrum_cm`` is ascending relative energies [cm^-1] (a probe's
    :attr:`~kuiva.autocas.probe.ProbeResult.spectrum_cm`). Consecutive states closer than
    ``gap_cm`` are one manifold and are never separated, so the returned count is a cut the
    state-average boundary diagnostic calls unambiguous.

    ⚠ **A floor inside a manifold is extended to the top of it, never truncated to below it.**
    The floor is theory's statement of the *smallest* honest average (a Hund ground level, a
    spin multiplicity), and a count below it would average over part of a term; a count that
    ends inside a near-degenerate manifold makes the averaged density non-invariant, the Fock
    operator built from it splits the shell, and the selection keeps cutting the same way.
    Both failures are one-sided, so the rule extends outward and only outward.
    """
    energies = np.asarray(spectrum_cm, dtype=float).ravel()
    floor = int(max(1, floor))
    if energies.size == 0:
        raise ValueError("a manifold boundary cannot be read off an empty spectrum")
    blocks = tuple(chain_blocks(energies / HARTREE_TO_CM, float(gap_cm) / HARTREE_TO_CM))
    for start, stop in blocks:
        if stop >= floor:
            if stop >= energies.size:
                break                       # the last block: no state above it was solved
            return ManifoldBoundary(count=int(stop), found=True, floor=floor,
                                    gap_cm=float(energies[stop] - energies[stop - 1]),
                                    blocks=blocks, manifold_gap_cm=float(gap_cm))
    return ManifoldBoundary(count=int(energies.size), found=False, floor=floor, gap_cm=None,
                            blocks=blocks, manifold_gap_cm=float(gap_cm))


def target_spectrum(spectrum_cm: Sequence[float], floor: int, *,
                    gap_cm: float = DEFAULT_MANIFOLD_GAP_CM) -> np.ndarray:
    """The numbers a feature class is judged by: the ground manifold, plus its gap.

    The relative energies inside the theoretical ground manifold -- the floor, chained outward
    to the boundary the probe shows -- followed by **the gap to the next manifold**. That is
    the exchange splitting for coupled shells, the ligand-field pattern for a single ion and
    the radical-ion coupling for a radical bridge: the quantity an active space is chosen to
    describe, and therefore the quantity a class is kept or dropped on.

    ⚠ The gap is the last entry rather than a separate number because a class that leaves the
    pattern alone and moves the manifold's *isolation* has changed the calculation just as
    much -- and a class that was accepted for moving a gap the next round then inherits is
    exactly the attribution the round loop exists to give.
    """
    energies = np.asarray(spectrum_cm, dtype=float).ravel()
    boundary = manifold_boundary(energies, floor, gap_cm=gap_cm)
    values: List[float] = list(energies[:boundary.count])
    if boundary.gap_cm is not None:
        values.append(float(boundary.gap_cm))
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class RootProposal:
    """A proposed state count, the window that says the same thing, and the evidence."""

    #: The proposed count, or ``None`` when the boundary is above the cap.
    n_states: Optional[int]
    #: The equivalent request. ⚠ Its cutoff is the **midpoint of the witness gap**, so the
    #: production ladder re-resolves the same boundary against a spectrum that has moved.
    window: Optional[EnergyWindow]
    floor: int
    #: The product of the per-centre floors -- the dimension of the exchange manifold, printed
    #: even where no cap can reach it, because it is what a whole-manifold average would cost.
    product_floor: int
    boundary: ManifoldBoundary
    max_states: int = DEFAULT_MAX_STATES
    notes: Tuple[str, ...] = ()

    @property
    def capped(self) -> bool:
        return self.n_states is None

    def describe(self) -> str:
        if self.capped:
            return ("no count: the boundary is at {} states, above the cap of {}"
                    .format(self.boundary.count, self.max_states))
        return "n_states {}  ({})".format(self.n_states, self.boundary.describe())

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger, "floor (theoretical ground manifold)", self.floor, "",
                  "product over centres {}".format(self.product_floor)
                  if self.product_floor != self.floor else "")
        out.entry(logger, "proposal", self.describe())
        if self.window is not None:
            edge = ""
            if self.boundary.found and self.boundary.gap_cm is not None:
                edge = "gap at the cut {:.2f} cm^-1".format(self.boundary.gap_cm)
            out.entry(logger, "equivalent window", "{:g} cm^-1".format(self.window.cutoff),
                      "", edge)
        for note in self.notes:
            out.note(logger, note)


def propose_roots(spectrum_cm: Sequence[float], floor: int, *, product_floor: Optional[int] = None,
                  gap_cm: float = DEFAULT_MANIFOLD_GAP_CM,
                  max_states: int = DEFAULT_MAX_STATES) -> RootProposal:
    """Propose a state count and an equivalent window from one probe's spectrum.

    ``floor`` is the theoretical ground-manifold dimension the count may not fall below
    (:mod:`kuiva.autocas.multiplets`); ``product_floor`` is the coupled-centre product, which
    is reported and never proposed.

    The three outcomes are the module docstring's, and each one is a different *statement*
    rather than a different number: a found boundary, a boundary above the cap (no count, the
    window with the cap), and no boundary in the roots solved (the floor, marked).
    """
    energies = np.asarray(spectrum_cm, dtype=float).ravel()
    boundary = manifold_boundary(energies, floor, gap_cm=gap_cm)
    product_floor = int(product_floor if product_floor is not None else floor)
    notes: List[str] = []

    if not boundary.found:
        window = EnergyWindow(cutoff=float(max(energies[-1], gap_cm)), unit="cm^-1",
                              manifold_gap=float(gap_cm), max_states=int(max_states))
        notes.append(
            "boundary not found in the {} roots the probe solved: the chained spectrum runs "
            "to the top without a gap wider than {:.0f} cm^-1 above the floor, so the floor "
            "of {} is proposed as a LOWER BOUND and not as a boundary. Solve more roots, or "
            "state a count.".format(int(energies.size), gap_cm, boundary.floor))
        return RootProposal(n_states=int(boundary.floor), window=window, floor=int(floor),
                            product_floor=product_floor, boundary=boundary,
                            max_states=int(max_states), notes=tuple(notes))

    cutoff = 0.5 * (float(energies[boundary.count - 1]) + float(energies[boundary.count]))
    window = EnergyWindow(cutoff=float(cutoff), unit="cm^-1", manifold_gap=float(gap_cm),
                          max_states=int(max_states))
    if boundary.count > int(max_states):
        notes.append(
            "the manifold boundary is at {} states, above the cap of {}: no count is "
            "proposed, because a capped count would be a cut INSIDE the manifold. The window "
            "is offered with the cap on it, where it will refuse rather than round. The whole "
            "coupled manifold is {} states.".format(boundary.count, max_states, product_floor))
        log.warning("%s", notes[-1])
        return RootProposal(n_states=None, window=window, floor=int(floor),
                            product_floor=product_floor, boundary=boundary,
                            max_states=int(max_states), notes=tuple(notes))
    if product_floor > boundary.count:
        notes.append(
            "the proposed count of {} is below the coupled manifold's {} states: the probe's "
            "boundary is a ligand-field one and the exchange manifold above it is not "
            "averaged whole".format(boundary.count, product_floor))
    return RootProposal(n_states=int(boundary.count), window=window, floor=int(floor),
                        product_floor=product_floor, boundary=boundary,
                        max_states=int(max_states), notes=tuple(notes))
