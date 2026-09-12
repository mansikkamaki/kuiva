"""State selection by an energy cutoff: the rule, the request object and the ladder.

A state-averaged calculation whose number of states is not stated as a count but **resolved
from an energy cutoff**: every state below the cutoff above the lowest one is solved and
averaged over, extended to the top of any manifold the cutoff falls inside. This module holds
the three things every CI route shares — the request (:class:`EnergyWindow`), the pure rule
(:func:`resolve_window`) and the ladder that drives a solver until the rule returns a verdict
(:func:`resolve_states`) — and knows nothing about any solver. It sits in ``util/`` because
``kuiva.dmrg`` may not import ``kuiva.mcscf`` and ``kuiva.ci`` may not import either; the
rule has to live below all three.

What a window resolves to is **a count**, and after that the calculation is the one the code
already runs: the state-averaging gate (equalized weights inside a degenerate block, an odd
count on an odd-electron system refused), the boundary and spin-invariance diagnostics and
every downstream consumer see an ordinary state count and learn nothing new.

The rule
--------
Inputs: ascending energies ``E[0..m)`` of the ``m`` lowest states solved at one set of
integrals, the size of the space they live in (``None`` when unknown), the cutoff ``Delta``
and the manifold gap ``g`` (both in Eh), and optionally the previous round's count.

* **Verdict.** ``k`` is the smallest index ``k >= 1`` with ``E[k] - E[0] > Delta`` **and**
  ``E[k] - E[k-1] > g``. If it exists the count is ``k`` and the verdict *complete*, with the
  witness gap ``E[k] - E[k-1]`` and the *spill* ``max(0, E[k-1] - E[0] - Delta)`` — how far
  the manifold rule extended the window past the cutoff. If no such ``k`` exists and the
  solved roots are the whole space, the count is the space size and the verdict is complete
  (no root is left out, no witness is needed — the same "pass, not a gap of zero" reading the
  boundary diagnostic has). Otherwise the verdict is *incomplete*: every solved root is inside
  the window or chained to it, ``m`` is a lower bound, and the caller must solve more.
* **The manifold rule is consecutive-gap chaining**, the way every other grouping in the
  project defines a group: a state above the cutoff whose predecessor is within ``g`` of it
  belongs to the predecessor's manifold and is included, and so on until a gap wider than
  ``g``. ⚠ Chaining can run: a spectrum with no gap wider than ``g`` anywhere chains to the
  cap. That is correct behaviour — such a spectrum has no unambiguous cut at this cutoff — and
  it is **refused, never rounded**: a cap that truncated the chain would be a cut inside a
  manifold, the one thing a window exists to forbid. The refusal prints the relative spectrum
  and the counts at which a gap wider than ``g`` *does* occur under the cap.
* **Hysteresis across rounds (dead band).** With the previous round's count given, the
  verdict is held at it if that count is still a gap boundary **and** the state that would
  move the count lies within ``g`` of the cutoff on its side — a count changes between rounds
  only when a state crosses the cutoff by more than one manifold tolerance. Without it a state
  sitting at the cutoff would flip the count on every round as the orbitals relax by a few
  wavenumbers; with it the first round's semantics are exactly the plain rule (there is no
  previous count) and a later round can only change the count for a reason the output names.
* **What the rule guarantees.** With ``g`` far above the degeneracy tolerance the RDM gate
  uses (50 cm^-1 against 1e-6 Eh, about 0.2 cm^-1) a resolved count can never split a
  numerically degenerate block, and the boundary gap at the resolved count is ``> g`` by
  construction — :data:`DEFAULT_MANIFOLD_GAP_CM` **is** the boundary-unambiguity threshold
  of the CASSCF driver, which is the whole reason the two numbers are one number. What it does
  **not** promise: that the window is a term-complete ensemble (the spin-non-invariance
  diagnostic stays), that a multiplet the ligand field splits by *more* than ``g`` is kept
  whole (it is genuinely split), or anything about states between the cutoff and the first
  gap when the witness roots were not converged (a one-sided witness, see below).

The ladder
----------
:func:`resolve_states` drives a :class:`SpectrumOracle` — anything that answers "the lowest
``n`` eigenvalues at these integrals", keeping its own warm start between calls — through
rungs of growing root count until the rule returns a verdict::

    n_1 = initial (user) | estimate (an upstream cheap CI, a pilot sweep) | 8
    each rung: energies = oracle.spectrum(n_r)      # warm from rung r-1, generic vectors added
               verdict  = resolve_window(energies, ..., space_size, n_prev)
               complete -> stop;  incomplete -> n_{r+1} = grow(n_r)
    grow: x2 by default (x1.5 for a solver whose rung is a sweep campaign); rounded to an
          even count for an odd electron count; never above min(max_states, space_size);
          reaching the cap without a verdict refuses

⚠ **The witness must be generic.** A rung supplies the previous rung's converged vectors and
asks for more roots than it has vectors; the new roots would otherwise be seeded from that
biased set alone, and a Krylov method cannot leave the invariant subspaces its starting
vectors lie in. The oracle is therefore asked with ``generic=True`` at every rung — the
eigensolver's generic starting vectors go in front of the warm start — which is the
Rayleigh-Ritz argument that a converged subspace with a component on every eigenvector cannot
skip a lower one. The oracle's spectrum must also be **converged**: on the tensor network a
*local* extra root bounds the true next eigenvalue from above and can prove a window
incomplete but never complete, and :attr:`WindowResolution.witness` states which kind was
used, every time.

Per-irrep form
--------------
``{irrep: EnergyWindow}`` — one window shared by every windowed sector, an ``int`` beside it a
fixed count for that sector — is resolved by :func:`resolve_states_per_sector`: each sector
is solved on its own ladder, the windowed sectors' spectra are **merged** and the rule is
applied to the merged spectrum against the **global** lowest root (the minimum over every
selected sector), so a degenerate block spanning two sectors is one block to the rule exactly
as it is one block to the gate. Only the part of the merged spectrum every unexhausted
sector has solved *past* is trusted, and the sectors limiting that bound are the ones grown.
A windowed sector whose lowest state lies above the window ends with a count of zero and is
dropped from the selection, which the output states.

References
----------
* The Rayleigh-Ritz interlacing argument behind "a converged generic subspace cannot skip a
  lower eigenvalue": B. N. Parlett, "The Symmetric Eigenvalue Problem", SIAM Classics (1998),
  ch. 10-11; E. R. Davidson, J. Comput. Phys. 17, 87 (1975), doi:10.1016/0021-9991(75)90065-0.
* Kramers' theorem, the reason an odd count is refused downstream and a rung is even on an
  odd-electron system: H. A. Kramers, Proc. Amsterdam Acad. 33, 959 (1930).
* CODATA 2018, the unit table the cutoff is converted through: E. Tiesinga, P. J. Mohr,
  D. B. Newell, B. N. Taylor, Rev. Mod. Phys. 93, 025010 (2021),
  doi:10.1103/RevModPhys.93.025010.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import (Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, TypeVar,
                    Union)

import numpy as np

from . import output as out
from .logging import get_logger
from .units import HARTREE_TO_CM, canonical_unit, to_hartree

log = get_logger(__name__)

#: Default manifold gap [cm^-1]: two consecutive states closer than this belong to one
#: manifold and are never separated by a window. ⚠ **Deliberately the boundary-unambiguity
#: threshold of the CASSCF driver** (its ``BOUNDARY_WARN_CM`` is this constant), so that a
#: resolved cut is clean by that diagnostic's own standard by construction. Not a physical
#: tolerance: it states that the cut is *unambiguous*, nothing more.
DEFAULT_MANIFOLD_GAP_CM = 50.0
#: Default cap on the count a window may resolve to. Reaching it without a verdict refuses.
DEFAULT_MAX_STATES = 64
#: Default cap on the number of resolve-optimize rounds a windowed CASSCF may take.
DEFAULT_MAX_ROUNDS = 4
#: The first rung of the ladder when nothing better is known — no user ``initial``, no
#: upstream estimate. A measured default in waiting: the rung cost decides whether it moves.
DEFAULT_FIRST_RUNG = 8
#: Growth factor of the ladder for a solver whose rung is one eigensolve.
DEFAULT_GROWTH = 2.0


# --- The request ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyWindow:
    """A state selection stated as an energy cutoff — the third form of ``n_states``.

    ``kuiva.EnergyWindow(1000)`` selects every state within 1000 cm^-1 of the lowest one,
    extended to the top of any manifold the cutoff falls inside (:data:`manifold_gap`, default
    :data:`DEFAULT_MANIFOLD_GAP_CM`). Validated eagerly; the cutoff and the gap are converted
    to Hartree once, here, and every consumer reads :attr:`cutoff_eh` and :attr:`gap_eh`.

    Parameters
    ----------
    cutoff : float
        The window above the lowest state, in ``unit``. Positive.
    unit : str
        ``"cm^-1"`` by default; any name :mod:`kuiva.util.units` accepts.
    manifold_gap : float
        Consecutive states closer than this are one manifold. In ``gap_unit`` (``unit`` when
        omitted). Positive.
    initial : int, optional
        The first rung of the ladder — how many roots to solve first. Overrides every estimate
        the code would make (an upstream cheap CI's count, a pilot sweep). ``None`` lets the
        code decide.
    max_states : int
        The cap on the resolved count. ⚠ Reaching it without a verdict **refuses**; it never
        truncates a chain.
    max_rounds : int
        The cap on the resolve-optimize rounds of a windowed CASSCF.
    """

    cutoff: float
    unit: str = "cm^-1"
    manifold_gap: float = DEFAULT_MANIFOLD_GAP_CM
    gap_unit: Optional[str] = None
    initial: Optional[int] = None
    max_states: int = DEFAULT_MAX_STATES
    max_rounds: int = DEFAULT_MAX_ROUNDS

    def __post_init__(self) -> None:
        unit = canonical_unit(self.unit)
        gap_unit = unit if self.gap_unit is None else canonical_unit(self.gap_unit)
        cutoff = float(self.cutoff)
        gap = float(self.manifold_gap)
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("an energy window needs a positive, finite cutoff; got {!r} {}"
                             .format(self.cutoff, unit))
        if not math.isfinite(gap) or gap <= 0.0:
            raise ValueError("the manifold gap must be positive and finite; got {!r} {}"
                             .format(self.manifold_gap, gap_unit))
        if self.initial is not None:
            if isinstance(self.initial, bool) or int(self.initial) != self.initial \
                    or int(self.initial) < 1:
                raise ValueError("initial= is the first number of roots to solve and must be "
                                 "a positive integer or None; got {!r}".format(self.initial))
        for name in ("max_states", "max_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("{}= must be a positive integer; got {!r}".format(name, value))
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "manifold_gap", gap)
        object.__setattr__(self, "gap_unit", gap_unit)
        object.__setattr__(self, "initial", None if self.initial is None else int(self.initial))
        object.__setattr__(self, "max_states", int(self.max_states))
        object.__setattr__(self, "max_rounds", int(self.max_rounds))

    @property
    def cutoff_eh(self) -> float:
        return to_hartree(self.cutoff, self.unit)

    @property
    def gap_eh(self) -> float:
        return to_hartree(self.manifold_gap, self.gap_unit)

    @property
    def cutoff_cm(self) -> float:
        return self.cutoff_eh * HARTREE_TO_CM

    @property
    def gap_cm(self) -> float:
        return self.gap_eh * HARTREE_TO_CM

    def describe(self) -> str:
        """One line: the cutoff as stated and in cm^-1, the gap, the caps."""
        text = "{:g} {}".format(self.cutoff, self.unit)
        if self.unit != "cm^-1":
            text += " ({:.2f} cm^-1)".format(self.cutoff_cm)
        text += ", manifold gap {:g} {}".format(self.manifold_gap, self.gap_unit)
        if self.gap_unit != "cm^-1":
            text += " ({:.2f} cm^-1)".format(self.gap_cm)
        text += ", at most {} states".format(self.max_states)
        if self.initial is not None:
            text += ", first rung {}".format(self.initial)
        return text

    def to_json(self) -> str:
        """A stable JSON form, for checkpoint metadata and file headers."""
        return json.dumps({"cutoff": self.cutoff, "unit": self.unit,
                           "manifold_gap": self.manifold_gap, "gap_unit": self.gap_unit,
                           "initial": self.initial, "max_states": self.max_states,
                           "max_rounds": self.max_rounds}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "EnergyWindow":
        return cls(**json.loads(text))

    def __repr__(self) -> str:
        parts = ["{:g}".format(self.cutoff)]
        if self.unit != "cm^-1":
            parts.append("unit={!r}".format(self.unit))
        if self.manifold_gap != DEFAULT_MANIFOLD_GAP_CM or self.gap_unit != self.unit:
            parts.append("manifold_gap={:g}".format(self.manifold_gap))
        if self.gap_unit != self.unit:
            parts.append("gap_unit={!r}".format(self.gap_unit))
        if self.initial is not None:
            parts.append("initial={}".format(self.initial))
        if self.max_states != DEFAULT_MAX_STATES:
            parts.append("max_states={}".format(self.max_states))
        if self.max_rounds != DEFAULT_MAX_ROUNDS:
            parts.append("max_rounds={}".format(self.max_rounds))
        return "EnergyWindow({})".format(", ".join(parts))


def is_window_request(n_states) -> bool:
    """Whether ``n_states`` is an :class:`EnergyWindow` or a per-irrep mapping holding one."""
    if isinstance(n_states, EnergyWindow):
        return True
    return isinstance(n_states, dict) and any(isinstance(v, EnergyWindow)
                                              for v in n_states.values())


def shared_window(request: Mapping[Hashable, Union[EnergyWindow, int]]) -> EnergyWindow:
    """The one window of a per-irrep request, refusing two different ones.

    A window is a statement about *the* spectrum — its cutoff is measured from the global
    lowest state and its manifold rule chains across sectors — so a per-irrep request holds one
    window, possibly beside fixed counts; two different cutoffs in one request would be two
    different statements about where one spectrum ends.
    """
    windows = [v for v in request.values() if isinstance(v, EnergyWindow)]
    if not windows:
        raise ValueError("the per-irrep request holds no EnergyWindow")
    for other in windows[1:]:
        if other != windows[0]:
            raise ValueError(
                "a per-irrep request holds ONE energy window (the cutoff is measured from the "
                "global lowest state and the manifold rule chains across sectors), possibly "
                "beside fixed counts; got {!r} and {!r}".format(windows[0], other))
    return windows[0]


# --- The rule -------------------------------------------------------------------------------

def chain_blocks(energies_asc: Sequence[float], tol_eh: float) -> List[Tuple[int, int]]:
    """Group ascending ``energies_asc`` into ``(start, stop)`` blocks, split where a consecutive
    gap exceeds ``tol_eh``.

    The one chaining implementation: :func:`kuiva.rdm.rdm.degenerate_blocks` (the RDM gate's
    degenerate blocks) is this at the degeneracy tolerance, and the window's manifold rule is
    this at the manifold gap. Refuses a descending input rather than sorting it, because a
    caller that hands a spectrum in the wrong order has a bug somewhere else.
    """
    values = np.asarray(energies_asc, dtype=float)
    if values.size == 0:
        return []
    if np.any(np.diff(values) < -float(tol_eh)):
        raise ValueError("state energies must be given in ascending order")
    edges = np.nonzero(np.diff(values) > float(tol_eh))[0] + 1
    bounds = [0] + edges.tolist() + [int(values.size)]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


@dataclass
class WindowVerdict:
    """What the rule says about one solved spectrum."""

    #: The resolved count when complete; the lower bound (every solved root) when not.
    count: int
    complete: bool
    #: Roots the verdict was read from.
    n_solved: int
    #: ``E[count] - E[count-1]`` [Eh] when a witness root exists; ``None`` when the average
    #: spans the whole space (a pass, not a gap of zero) or the verdict is incomplete.
    witness_gap_eh: Optional[float] = None
    #: How far the manifold rule extended the window past the cutoff [Eh].
    spill_eh: float = 0.0
    #: The last selected state and the ones beyond it, relative to the reference [cm^-1].
    relative_cm: Tuple[float, ...] = ()
    #: Whether hysteresis held the count at the previous round's value.
    held: bool = False
    #: Whether the count is the whole space.
    spans_space: bool = False
    #: The rule's own count before hysteresis, when it differs from :attr:`count`.
    unheld_count: Optional[int] = None

    @property
    def witness_gap_cm(self) -> Optional[float]:
        return None if self.witness_gap_eh is None else self.witness_gap_eh * HARTREE_TO_CM

    @property
    def spill_cm(self) -> float:
        return self.spill_eh * HARTREE_TO_CM

    @property
    def edge_cm(self) -> Optional[Tuple[float, float]]:
        """``(last inside, first outside)`` relative energies [cm^-1], when both exist."""
        if not self.complete or self.spans_space or self.count == 0 \
                or len(self.relative_cm) < 2:
            return None
        return (self.relative_cm[0], self.relative_cm[1])


def resolve_window(energies_asc: Sequence[float], window: EnergyWindow, *,
                   space_size: Optional[int] = None, n_prev: Optional[int] = None,
                   reference_eh: Optional[float] = None) -> WindowVerdict:
    """The rule of the module docstring, applied to one ascending spectrum. Pure.

    ``reference_eh`` is the energy the cutoff is measured from — ``E[0]`` by default, and the
    *global* lowest root when ``energies_asc`` is one sector's spectrum of a per-irrep request.
    """
    energies = np.asarray(energies_asc, dtype=float).ravel()
    if energies.size == 0:
        raise ValueError("a window cannot be resolved on an empty spectrum")
    if np.any(np.diff(energies) < -window.gap_eh):
        raise ValueError("the spectrum handed to resolve_window is not ascending")
    if space_size is not None and energies.size > int(space_size):
        raise ValueError("{} roots solved in a space of {} states"
                         .format(energies.size, space_size))
    m = int(energies.size)
    delta, gap = window.cutoff_eh, window.gap_eh
    reference = float(energies[0]) if reference_eh is None else float(reference_eh)
    rel = energies - reference
    steps = np.diff(energies)

    k: Optional[int] = None
    # ⚠ The loop starts at 0 only in the per-sector form (reference below E[0]): a sector
    # whose lowest state is already above the cutoff by more than one gap contributes nothing.
    start = 0 if reference_eh is not None else 1
    for i in range(start, m):
        above = rel[i] > delta
        gapped = (i == 0) or steps[i - 1] > gap
        if above and gapped:
            k = i
            break

    if k is not None:
        verdict = WindowVerdict(
            count=k, complete=True, n_solved=m,
            witness_gap_eh=None if k == 0 else float(steps[k - 1]),
            spill_eh=float(max(0.0, rel[k - 1] - delta)) if k > 0 else 0.0,
            relative_cm=tuple(float(x) * HARTREE_TO_CM for x in rel[max(k - 1, 0):]))
    elif space_size is not None and m >= int(space_size):
        verdict = WindowVerdict(
            count=m, complete=True, n_solved=m, witness_gap_eh=None,
            spill_eh=float(max(0.0, rel[-1] - delta)), spans_space=True,
            relative_cm=(float(rel[-1]) * HARTREE_TO_CM,))
    else:
        verdict = WindowVerdict(count=m, complete=False, n_solved=m,
                                relative_cm=tuple(float(x) * HARTREE_TO_CM for x in rel[-2:]))

    if n_prev is None or not 1 <= int(n_prev) < m or verdict.count == int(n_prev):
        return verdict
    n_prev = int(n_prev)
    # The dead band. Held only where n_prev is still a gap boundary AND the state that would
    # move the count sits within one manifold gap of the cutoff on its side.
    if steps[n_prev - 1] <= gap:
        return verdict
    # ⚠ "Near" is **within one manifold gap of the cutoff**, on whichever side the mover is.
    # Written as ``mover - delta <= gap`` the entering branch is true for every state below
    # the cutoff, however far below, so any count that grew between rounds would be held at
    # the old one for ever — found by a two-round CASSCF whose second round never ran.
    if verdict.count > n_prev:
        mover = rel[n_prev]                  # the first state that entered the window
        near = mover <= delta and (delta - mover) <= gap
    else:
        mover = rel[n_prev - 1]              # the last state that left the window
        near = mover > delta and (mover - delta) <= gap
    if not near:
        return verdict
    return WindowVerdict(
        count=n_prev, complete=True, n_solved=m, witness_gap_eh=float(steps[n_prev - 1]),
        spill_eh=float(max(0.0, rel[n_prev - 1] - delta)),
        relative_cm=tuple(float(x) * HARTREE_TO_CM for x in rel[n_prev - 1:]),
        held=True, unheld_count=verdict.count)


class WindowCapReached(ValueError):
    """The ladder reached ``max_states`` without a verdict — a chain the cap would have cut."""


def clean_counts(energies_asc: Sequence[float], window: EnergyWindow, *,
                 reference_eh: Optional[float] = None) -> List[int]:
    """Counts ``k`` at which a gap wider than the manifold gap occurs (``E[k]-E[k-1] > g``)."""
    energies = np.asarray(energies_asc, dtype=float).ravel()
    return [int(i) for i in np.nonzero(np.diff(energies) > window.gap_eh)[0] + 1]


def refuse_at_cap(energies_asc: Sequence[float], window: EnergyWindow, *,
                  reference_eh: Optional[float] = None) -> None:
    """Raise :class:`WindowCapReached` with the spectrum and the clean counts under the cap."""
    energies = np.asarray(energies_asc, dtype=float).ravel()
    reference = float(energies[0]) if reference_eh is None else float(reference_eh)
    rel_cm = (energies - reference) * HARTREE_TO_CM
    clean = clean_counts(energies, window)
    largest = ("the largest unambiguous count under the cap is {}".format(clean[-1])
               if clean else "no gap wider than the manifold gap occurs under the cap at all")
    raise WindowCapReached(
        "the energy window {} chains past its cap of {} states: {} roots were solved and the "
        "manifold rule (consecutive gaps under {:.2f} cm^-1) never found a cut above the "
        "cutoff. A cap that truncated the chain would cut inside a manifold, so this is "
        "refused rather than rounded. Relative energies [cm^-1]: {}. Counts at which a gap "
        "wider than the manifold gap does occur: {} -- {}. State a count, lower the cutoff, "
        "or raise max_states".format(
            repr(window), window.max_states, energies.size, window.gap_cm,
            ", ".join("{:.2f}".format(x) for x in rel_cm),
            ", ".join(map(str, clean)) if clean else "none", largest))


# --- The ladder ------------------------------------------------------------------------------

class SpectrumOracle:
    """What the ladder drives: ``spectrum(n_roots) -> ascending energies [Eh]``.

    A structural protocol (any object with the method qualifies). The oracle keeps its own
    warm start between calls and **adds the eigensolver's generic vectors in front of it** at
    every call — a rung supplies fewer converged vectors than the roots it asks for, and the
    new roots would otherwise be seeded from a biased set alone. It may expose ``n_apply``
    (cumulative applications of ``H``) for the rung table.
    """

    def spectrum(self, n_roots: int) -> np.ndarray:            # pragma: no cover - protocol
        raise NotImplementedError


@dataclass
class Rung:
    """One rung of the ladder: what was asked, what it cost, what the rule said."""

    index: int
    n_roots: int
    verdict: WindowVerdict
    n_apply: Optional[int] = None
    cpu_seconds: float = 0.0
    #: Per-sector root counts on a per-irrep ladder (names -> roots asked).
    sectors: Optional[Dict[str, int]] = None


@dataclass
class WindowResolution:
    """A resolved window: the count, the ladder that produced it, and what to report."""

    window: EnergyWindow
    count: int
    verdict: WindowVerdict
    rungs: List[Rung]
    #: The last rung's spectrum [Eh] (merged, for a per-irrep ladder).
    energies: np.ndarray
    #: Where the first rung came from (``"initial="``, ``"upstream CheapCI"``, ``"default"``,
    #: ``"dense space"``, ...).
    first_rung_source: str = "default"
    space_size: Optional[int] = None
    #: Which kind of witness the rule read: ``"converged roots"`` (an eigensolve to
    #: convergence, which can prove completeness) or ``"local, one-sided"`` (a network's
    #: local extra root, which cannot).
    witness: str = "converged roots"
    #: Per-sector resolved counts for a per-irrep request (sector name -> count).
    counts: Optional[Dict[str, int]] = None
    #: For a per-irrep request: the sector whose edge is the tightest, and its gap [cm^-1].
    sector: Optional[str] = None
    #: Whether the rounds of a windowed CASSCF ended with the verdict still changing.
    ambiguous: bool = False
    #: The previous round's count the resolution was measured against, when there was one.
    n_prev: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.verdict.complete

    @property
    def boundary_gap_cm(self) -> Optional[float]:
        """The gap from the last selected state to the first left out [cm^-1]; ``None`` when
        the selection spans the space."""
        return self.verdict.witness_gap_cm

    @property
    def n_apply(self) -> int:
        return int(sum(r.n_apply or 0 for r in self.rungs))

    @property
    def cpu_seconds(self) -> float:
        return float(sum(r.cpu_seconds for r in self.rungs))

    def to_json(self) -> str:
        return json.dumps({
            "window": json.loads(self.window.to_json()), "count": int(self.count),
            "complete": bool(self.complete), "spans_space": bool(self.verdict.spans_space),
            "witness_gap_cm": self.boundary_gap_cm, "spill_cm": self.verdict.spill_cm,
            "held": bool(self.verdict.held), "witness": self.witness,
            "first_rung_source": self.first_rung_source, "ambiguous": bool(self.ambiguous),
            "rungs": [{"roots": r.n_roots, "n_apply": r.n_apply,
                       "cpu_seconds": round(r.cpu_seconds, 3),
                       "complete": bool(r.verdict.complete), "count": int(r.verdict.count)}
                      for r in self.rungs],
            "counts": self.counts, "sector": self.sector}, sort_keys=True)

    def report(self, logger=None, *, level: int = logging.INFO, where: str = "") -> None:
        """The ``[state window]`` block, through the output grammar only."""
        logger = logger or log
        w = self.window
        out.subsection(logger, "state window" + (" at the {}".format(where) if where else ""))
        note = "{:.8f} Eh".format(w.cutoff_eh)
        if w.unit != "cm^-1":
            note += "; stated as {:g} {}".format(w.cutoff, w.unit)
        out.entry(logger, "cutoff", w.cutoff_cm, "cm^-1", note, fmt=out.CM_FMT, level=level)
        out.entry(logger, "manifold gap", w.gap_cm, "cm^-1",
                  "" if w.gap_unit == "cm^-1" else "stated as {:g} {}".format(w.manifold_gap,
                                                                             w.gap_unit),
                  fmt=out.CM_FMT, level=level)
        if self.rungs:
            out.entry(logger, "first rung", self.rungs[0].n_roots, "",
                      "from {}".format(self.first_rung_source), level=level)
        out.entry(logger, "witness", self.witness, "", level=level)
        columns = [out.col_count("rung", 5), out.col_count("roots", 6),
                   out.col_count("n_apply", 8), out.Column("cpu [s]", out.TIME_FMT, 8),
                   out.Column("verdict", "{}", 44, align="<")]
        table = out.Table(logger, columns, level=level)
        table.start()
        for rung in self.rungs:
            v = rung.verdict
            if v.complete and v.spans_space:
                text = "complete: {} states, the whole space".format(v.count)
            elif v.complete:
                text = "complete: {} states, witness gap {:.2f} cm^-1".format(
                    v.count, v.witness_gap_cm)
                if v.held:
                    text += " (held at the previous count; the rule alone says {})".format(
                        v.unheld_count)
            else:
                text = "incomplete ({} of {} inside)".format(v.count, rung.n_roots)
            roots = rung.n_roots
            if rung.sectors:
                text += "  [" + ", ".join("{}: {}".format(k, n)
                                          for k, n in rung.sectors.items()) + "]"
            table.row(rung.index, roots, rung.n_apply, rung.cpu_seconds, text)
        table.end()
        out.entry(logger, "resolved", self.count, "",
                  "spill {:.2f} cm^-1 above the cutoff".format(self.verdict.spill_cm),
                  level=level)
        if self.counts is not None:
            out.entry(logger, "per irrep", "", "",
                      ", ".join("{}: {}".format(k, n) for k, n in self.counts.items()),
                      level=level)
        edge = self.verdict.edge_cm
        if edge is not None:
            out.entry(logger, "edge [cm^-1]", "{:.2f} | {:.2f}".format(*edge), "",
                      "tightest irrep {}".format(self.sector) if self.sector else
                      "last selected | first left out", level=level)
        elif self.verdict.spans_space:
            out.entry(logger, "edge", "none", "", "the selection spans the whole space",
                      level=level)


def _even_up(n: int, n_elec: Optional[int]) -> int:
    """Round a rung up to an even count on an odd-electron system (whole Kramers pairs)."""
    if n_elec is not None and int(n_elec) % 2 == 1 and n % 2 == 1:
        return n + 1
    return n


def _clamp(n: int, window: EnergyWindow, space_size: Optional[int]) -> int:
    ceiling = window.max_states if space_size is None else min(window.max_states,
                                                                 int(space_size))
    return max(1, min(int(n), int(ceiling)))


def first_rung(window: EnergyWindow, *, estimate: Optional[int] = None,
               n_elec: Optional[int] = None, space_size: Optional[int] = None,
               n_prev: Optional[int] = None) -> Tuple[int, str]:
    """``(n_1, source)``: the ladder's first rung and where it came from.

    The user's ``initial`` overrides an ``estimate`` (an upstream cheap CI's resolved count,
    a pilot sweep's), which overrides :data:`DEFAULT_FIRST_RUNG`. With a previous round's
    count the first rung is at least that count plus one witness pair, so the re-resolution can
    see a verdict at it — and where that is what decided the rung, the source says so rather
    than naming an estimate that did not.
    """
    if window.initial is not None:
        n, source = int(window.initial), "initial="
    elif estimate is not None:
        n, source = int(estimate), "the upstream estimate"
    else:
        n, source = DEFAULT_FIRST_RUNG, "the default first rung"
    if n_prev is not None and int(n_prev) + 2 > n:
        # ⚠ And it says so: a later round's first rung is set by the count the previous one
        # ran at, not by whatever estimate started the calculation, and a source line naming
        # that estimate would make the handoff unreadable in an output with several rounds.
        n = int(n_prev) + 2
        source = "the previous round's count plus a witness pair"
    n = _even_up(n, n_elec)
    return _clamp(n, window, space_size), source


def next_rung(n: int, window: EnergyWindow, *, n_elec: Optional[int] = None,
              space_size: Optional[int] = None, growth: float = DEFAULT_GROWTH) -> int:
    """The rung after ``n``: grown, rounded to a pair on an odd-electron system, clamped."""
    grown = max(n + 1, int(math.ceil(n * float(growth))))
    return _clamp(_even_up(grown, n_elec), window, space_size)


def _cost(oracle) -> Optional[int]:
    value = getattr(oracle, "n_apply", None)
    return None if value is None else int(value)


def resolve_states(oracle: SpectrumOracle, window: EnergyWindow, *,
                   n_elec: Optional[int] = None, space_size: Optional[int] = None,
                   estimate: Optional[int] = None, n_prev: Optional[int] = None,
                   first: Optional[Tuple[int, str]] = None,
                   growth: float = DEFAULT_GROWTH,
                   witness: str = "converged roots") -> WindowResolution:
    """Drive ``oracle`` up the ladder until the rule returns a verdict.

    ``first`` overrides the first-rung choice with an explicit ``(n_1, source)`` — the CI
    solver uses it to ask for the whole space in one rung where the space is small enough for
    a dense solve. A rung whose oracle fails propagates the failure: the resolution runs at an
    accepted point (starting or converged orbitals), where there is nothing to reject. Reaching
    the cap without a verdict raises :class:`WindowCapReached`.
    """
    if first is None:
        n, source = first_rung(window, estimate=estimate, n_elec=n_elec,
                               space_size=space_size, n_prev=n_prev)
    else:
        n, source = _clamp(int(first[0]), window, space_size), str(first[1])
    rungs: List[Rung] = []
    energies = np.zeros(0)
    ceiling = window.max_states if space_size is None else min(window.max_states,
                                                                 int(space_size))
    while True:
        before, tic = _cost(oracle), time.process_time()
        energies = np.asarray(oracle.spectrum(int(n)), dtype=float).ravel()
        cpu = time.process_time() - tic
        after = _cost(oracle)
        if energies.size < n:
            raise RuntimeError("the spectrum oracle returned {} roots for a request of {}"
                               .format(energies.size, n))
        verdict = resolve_window(energies, window, space_size=space_size, n_prev=n_prev)
        rungs.append(Rung(index=len(rungs) + 1, n_roots=int(n), verdict=verdict,
                          n_apply=None if before is None or after is None else after - before,
                          cpu_seconds=float(cpu)))
        if verdict.complete:
            break
        if n >= ceiling:
            if space_size is not None and n >= int(space_size):    # pragma: no cover
                raise RuntimeError("the rule read an incomplete verdict on the whole space")
            refuse_at_cap(energies, window)
        n = next_rung(n, window, n_elec=n_elec, space_size=space_size, growth=growth)
    return WindowResolution(window=window, count=int(verdict.count), verdict=verdict,
                            rungs=rungs, energies=energies, first_rung_source=source,
                            space_size=space_size, witness=witness, n_prev=n_prev)


# --- The per-irrep form -------------------------------------------------------------------

K = TypeVar("K", bound=Hashable)


def resolve_states_per_sector(oracles: Mapping[K, SpectrumOracle],
                              request: Mapping[K, Union[EnergyWindow, int]], *,
                              sizes: Mapping[K, int], names: Mapping[K, str],
                              n_elec: Optional[int] = None,
                              estimate: Optional[int] = None, n_prev: Optional[int] = None,
                              growth: float = DEFAULT_GROWTH,
                              witness: str = "converged roots") -> WindowResolution:
    """Resolve ``{irrep: EnergyWindow | int}`` over per-sector oracles (module docstring).

    Fixed-count sectors are solved at their count plus one witness pair, so the report can
    state their edge too, and are never re-solved. Windowed sectors share one window
    (:func:`shared_window`) and are grown until the merged rule is complete and every
    unexhausted windowed sector has solved past the merged edge. Returns a
    :class:`WindowResolution` whose :attr:`~WindowResolution.counts` holds each sector's count
    (a windowed sector may end at zero and is then absent) and whose ``count`` is the total.
    """
    window = shared_window(request)
    windowed = [k for k, v in request.items() if isinstance(v, EnergyWindow)]
    fixed = {k: int(v) for k, v in request.items() if not isinstance(v, EnergyWindow)}
    for k, v in fixed.items():
        if v <= 0:
            raise ValueError("asked for {} states of {}; a sector is either requested or left "
                             "out".format(v, names[k]))
    spectra: Dict[K, np.ndarray] = {}
    asked: Dict[K, int] = {}
    rungs: List[Rung] = []

    def solve(k: K, n: int) -> Tuple[Optional[int], float]:
        n = max(1, min(int(n), int(sizes[k])))
        before, tic = _cost(oracles[k]), time.process_time()
        spectra[k] = np.asarray(oracles[k].spectrum(n), dtype=float).ravel()
        asked[k] = n
        after = _cost(oracles[k])
        return (None if before is None or after is None else after - before,
                time.process_time() - tic)

    # Fixed sectors first: their counts are the request, their extra roots are the witness.
    for k, count in fixed.items():
        if count > int(sizes[k]):
            raise ValueError("asked for {} states of {}, which holds {} determinants"
                             .format(count, names[k], sizes[k]))
        solve(k, count + 2)

    to_grow = list(windowed)
    n_first, source = first_rung(window, estimate=estimate, n_elec=None, n_prev=n_prev)
    next_n = {k: max(1, min(n_first, int(sizes[k]))) for k in windowed}
    ceiling = window.max_states
    while True:
        n_apply: Optional[int] = 0
        cpu = 0.0
        for k in to_grow:
            cost, seconds = solve(k, next_n[k])
            n_apply = None if (n_apply is None or cost is None) else n_apply + cost
            cpu += seconds
        reference = min(float(e[0]) for e in spectra.values())
        merged, owner = _merge(spectra, windowed)
        # The trusted part of the merged spectrum: everything at or below the lowest
        # top-solved root of an unexhausted windowed sector.
        limiting = [k for k in windowed if asked[k] < int(sizes[k])]
        bound = min((float(spectra[k][-1]) for k in limiting), default=np.inf)
        trusted = int(np.count_nonzero(merged <= bound + 1e-14 * max(1.0, abs(bound))))
        exhausted = int(merged.size) if not limiting else None
        verdict = resolve_window(merged[:trusted], window,
                                 space_size=exhausted if exhausted is not None else None,
                                 n_prev=n_prev, reference_eh=reference)
        rungs.append(Rung(index=len(rungs) + 1,
                          n_roots=int(sum(asked[k] for k in windowed)), verdict=verdict,
                          n_apply=n_apply, cpu_seconds=cpu,
                          sectors={names[k]: asked[k] for k in windowed}))
        if verdict.complete:
            break
        total_asked = sum(asked[k] for k in windowed)
        if total_asked >= ceiling:
            refuse_at_cap(merged[:trusted], window, reference_eh=reference)
        # Grow the sectors that bound the trusted spectrum (the ones with no root above it).
        to_grow = [k for k in limiting if float(spectra[k][-1]) <= bound + 1e-14]
        if not to_grow:                                        # pragma: no cover - defensive
            to_grow = limiting
        for k in to_grow:
            next_n[k] = max(asked[k] + 1, min(next_rung(asked[k], window, growth=growth),
                                              int(sizes[k])))
            next_n[k] = min(next_n[k], int(sizes[k]))

    counts: Dict[str, int] = {}
    tightest: Optional[Tuple[float, str]] = None
    selected = owner[:verdict.count]
    for k in windowed:
        n_k = int(np.count_nonzero(selected == windowed.index(k)))
        if n_k:
            counts[names[k]] = n_k
        # this sector's own edge, for the report's "tightest irrep"
        if 0 < n_k < spectra[k].size:
            gap_k = float(spectra[k][n_k] - spectra[k][n_k - 1]) * HARTREE_TO_CM
            if tightest is None or gap_k < tightest[0]:
                tightest = (gap_k, names[k])
    for k, count in fixed.items():
        counts[names[k]] = count
        if count < spectra[k].size:
            gap_k = float(spectra[k][count] - spectra[k][count - 1]) * HARTREE_TO_CM
            if tightest is None or gap_k < tightest[0]:
                tightest = (gap_k, names[k])
    total = int(sum(counts.values()))
    return WindowResolution(window=window, count=total, verdict=verdict, rungs=rungs,
                            energies=merged, first_rung_source=source, witness=witness,
                            counts=counts, sector=None if tightest is None else tightest[1],
                            n_prev=n_prev,
                            extra={"tightest_gap_cm": None if tightest is None
                                   else tightest[0]})


def _merge(spectra: Mapping[K, np.ndarray], keys: Sequence[K]) -> Tuple[np.ndarray, np.ndarray]:
    """Merged ascending energies of ``keys`` and, per merged state, the index into ``keys``."""
    parts = [np.asarray(spectra[k], dtype=float) for k in keys]
    owners = [np.full(p.size, i, dtype=int) for i, p in enumerate(parts)]
    merged = np.concatenate(parts) if parts else np.zeros(0)
    owner = np.concatenate(owners) if owners else np.zeros(0, dtype=int)
    order = np.argsort(merged, kind="stable")
    return merged[order], owner[order]


__all__ = ["EnergyWindow", "WindowVerdict", "WindowResolution", "WindowCapReached", "Rung",
           "SpectrumOracle", "chain_blocks", "resolve_window", "resolve_states",
           "resolve_states_per_sector", "first_rung", "next_rung", "clean_counts",
           "refuse_at_cap", "is_window_request", "shared_window",
           "DEFAULT_MANIFOLD_GAP_CM", "DEFAULT_MAX_STATES", "DEFAULT_MAX_ROUNDS",
           "DEFAULT_FIRST_RUNG", "DEFAULT_GROWTH"]
