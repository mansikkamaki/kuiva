"""Resolving a state count from an energy cutoff on the tensor network.

The network side of :mod:`kuiva.util.window`: the pilot that gives the ladder its first
rung, the :class:`~kuiva.util.window.SpectrumOracle` whose every rung is a whole sweep
campaign, and the truncation that hands the resolved ensemble on as a warm start. The rule
itself is not here and is not restated — it is one implementation in ``util/window.py``,
shared with the conventional CI and the cheap CI, and this module only drives it.

Three things make a network rung different from a CI rung, and all three are decisions
rather than details.

**The witness roots are network roots.** A rung solves ``n + 2`` roots (a whole pair) to
convergence and hands the rule the converged spectrum, so the verdict has a witness above
the count that was asked for. The cheaper alternative — the *local* extra root of the
boundary sweep (:func:`kuiva.dmrg.sweep._boundary_sweep`) — cannot serve here: an extra
root of one two-site problem bounds the true next eigenvalue **from above**, so it can prove
a state lies *inside* the window and never that the window is complete. The kind of witness
is carried on the resolution (:attr:`~kuiva.util.window.WindowResolution.witness`) and
stated in the output every time, because the two are not the same claim.

**The first rung comes from a pilot.** An upstream cheap CI supplies the estimate when there
is one and ``EnergyWindow(initial=)`` overrides everything; failing both,
:func:`pilot_estimate` runs a short campaign at a small cap (:data:`PILOT_CAP`) over a
generous root count and reads the rule on *its* spectrum. A pilot is variational from above
per root and its splittings are rough — which is exactly what a first rung is, and why the
rule is re-run on the production spectrum afterwards rather than trusted here.

**Growth is cold.** An incomplete verdict rebuilds the state at the grown count from
:func:`kuiva.dmrg.sweep.random_state`, as :func:`kuiva.dmrg.manifold.solve_manifold` does for
the same reason: adding random centers to the incumbent shared basis is the obvious warm
variant and warm-starting across a root-count change is an **unmeasured** optimization. The
variant exists here as :func:`pad_roots`, reached by ``resolve_network_window(warm_growth=
True)`` and **off**: what it can buy is time, never correctness, and what it must be shown not
to move is the resolved count. Until that measurement exists on a real multi-site system the
default is the cold rebuild.

⚠ **A root count the network cannot represent is refused, not truncated.** The two-site
window at some bond of the tour can be smaller than the root count no matter how large the
cap is (:func:`kuiva.dmrg.plan.two_site_capacity`), so the ladder's ceiling is that capacity
as well as the cutoff's own cap: the rungs stop below the refusal instead of walking into
it, and a window that would need more roots than the topology can hold is refused with the
capacity, the sector and the fix named. The same bound applies to the pilot. That ceiling is
the **guaranteed** dimension — one per reachable charge sector — while a state at a tight cap
may carry fewer of the sectors than the bond allows, so a rung can still land past a bond's
actual dimension; the sweep's own :class:`~kuiva.dmrg.sweep.TwoSiteCapacityError` catches
that one and the ladder re-raises it saying the roots were a rung's rather than a stated
count.

References
----------
* The shared-basis multi-root network the witness roots join: J. J. Dorando, J. Hachmann,
  G. K.-L. Chan, J. Chem. Phys. 127, 084109 (2007), doi:10.1063/1.2768360.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from ..util.units import HARTREE_TO_CM
from ..util.window import (EnergyWindow, WindowResolution, resolve_states,
                          resolve_window)
from .plan import two_site_capacity
from .block import BlockTensor
from .sweep import (TTNState, TwoSiteCapacityError, _normalized, random_state,
                    solve_ttn)
from .ttno import TTNO

log = get_logger(__name__)

#: Bond-dimension cap of the pilot campaign. Small on purpose: the pilot is asked how many
#: roots lie under the cutoff, not what their energies are.
PILOT_CAP = 8
#: Sweeps the pilot is allowed. A bounded budget, never a convergence criterion.
PILOT_SWEEPS = 4
#: Roots the pilot solves when nothing upstream suggests a number.
PILOT_ROOTS = 16
#: Witness roots a rung carries above the count it is testing — a whole Kramers pair.
WITNESS_ROOTS = 2
#: Rung growth on this route. Lower than the CI's doubling because a rung is a campaign.
NETWORK_GROWTH = 1.5
#: What :attr:`~kuiva.util.window.WindowResolution.witness` says for a network resolution.
WITNESS_KIND = "converged network roots"


def sector_upper_bound(n_modes: int, n_elec: int) -> int:
    """``C(n_modes, n_elec)`` — states in the particle-number sector, an **upper** bound.

    With irrep labels on, the sector the solve targets is finer than this; the binding
    ceiling in that case is :func:`~kuiva.dmrg.plan.two_site_capacity`, which is computed
    from the charge sectors the state actually carries.
    """
    return int(math.comb(int(n_modes), int(n_elec)))


def truncate_roots(state: TTNState, count: int) -> TTNState:
    """The ensemble with its first ``count`` roots kept — the production state of a rung.

    ⚠ The witness centers are **dropped, never averaged**: they exist to prove there is a
    gap above the count, exactly as the extra roots of the conventional CI's boundary solve
    do. What is kept is a warm start and not an answer — the shared basis was optimized for
    a larger ensemble, so it is a superset of the one the production solve needs.
    """
    count = int(count)
    if count > state.n_roots:
        raise ValueError("cannot keep {} roots of a {}-root ensemble"
                         .format(count, state.n_roots))
    return TTNState(graph=state.graph, center=state.center,
                    tensors=[None if t is None else t.copy() for t in state.tensors],
                    centers=[c.copy() for c in state.centers[:count]], charge=state.charge)


def pad_roots(state: TTNState, count: int, rng) -> TTNState:
    """The ensemble grown to ``count`` roots by appending random centers — **warm growth**.

    The isometries and every bond space are the incumbent's, so what the grown rung starts
    from is the shared basis the previous rung converged; the new centers are random tensors
    of the same block structure, which is what :func:`~kuiva.dmrg.sweep.random_state` builds
    every center as anyway (they are not mutually orthogonalized there either — the local
    eigensolver orthogonalizes its own guess).

    ⚠ **Measured-only, and off by default** (see :func:`resolve_network_window`): whether this
    beats a cold rebuild at the grown count is a cost question on a real multi-site system, and
    the resolved count must be shown not to move.
    """
    count = int(count)
    if count <= state.n_roots:
        return truncate_roots(state, count)
    seed = state.centers[0]
    extra = [_normalized(BlockTensor.random(seed.spaces, seed.signs, seed.charge, rng=rng))
             for _ in range(count - state.n_roots)]
    return TTNState(graph=state.graph, center=state.center,
                    tensors=[None if t is None else t.copy() for t in state.tensors],
                    centers=[c.copy() for c in state.centers] + extra, charge=state.charge)


def kramers_split_cm(energies: Sequence[float], count: int, n_elec: int) -> Optional[float]:
    """The worst splitting inside a Kramers pair of the selected roots [cm^-1], or ``None``.

    ⚠ **What a window's verdict on a truncating network actually rests on.** With an odd
    electron count every level is at least doubly degenerate as a matter of theorem, so any
    splitting between roots ``2i`` and ``2i+1`` of a converged ensemble is the *truncation's*,
    not the physics'. When it exceeds the manifold gap the rule is reading a spectrum whose
    own degeneracies are broken, and the count it returns can cut a manifold — which is the
    one thing a window exists to prevent. It is reported rather than acted on: the remedy is
    a larger bond dimension, and only the caller can decide to pay for it.
    """
    if int(n_elec) % 2 == 0:
        return None
    e = np.asarray(energies, dtype=float)[:int(count)]
    if e.size < 2:
        return None
    pairs = e[1:e.size - e.size % 2:2] - e[0:e.size - e.size % 2:2]
    return float(np.max(pairs)) * HARTREE_TO_CM if pairs.size else None


@dataclass
class PilotEstimate:
    """What the pilot campaign read, and what it cost."""

    n_roots: int
    cap: int
    n_sweeps: int
    cpu_seconds: float
    #: The count the rule read on the pilot's spectrum — the ladder's first rung.
    count: int
    #: Whether the rule reached a verdict at all on the pilot's roots.
    complete: bool
    converged: bool
    relative_cm: Tuple[float, ...] = ()

    def report(self, logger=None, *, level: int = logging.INFO) -> None:
        logger = logger or log
        out.subsection(logger, "state window: pilot sweep")
        out.entry(logger, "bond dimension", self.cap, "", "a rough spectrum, deliberately",
                  level=level)
        out.entry(logger, "roots", self.n_roots, "", level=level)
        out.entry(logger, "sweeps", self.n_sweeps, "",
                  "converged" if self.converged else "budget spent (a pilot need not "
                                                     "converge)", level=level)
        out.entry(logger, "cpu", self.cpu_seconds, "s", fmt=out.TIME_FMT, level=level)
        out.entry(logger, "first rung it suggests", self.count, "",
                  "the rule found a cut on the pilot spectrum" if self.complete
                  else "every pilot root is inside the window; a lower bound", level=level)


def _solve_roots(ttno: TTNO, n_elec: int, n_roots: int, *, max_bond: int, rng, charge,
                 solve_kwargs: Dict[str, Any], memory_plan: bool, max_sweeps: int,
                 start: Optional[TTNState] = None
                 ) -> Tuple[np.ndarray, TTNState, int, bool]:
    """One campaign at ``n_roots`` roots: ``(energies, state, n_sweeps, converged)``.

    Cold from :func:`~kuiva.dmrg.sweep.random_state` unless ``start`` supplies an ensemble
    already at that root count (warm growth, measured-only).

    ⚠ **A rung is a spectrum probe and passes ``on_split="warn"``.** The state-averaging
    gate refuses a count that splits a degenerate block because a state-averaged **RDM**
    built from one depends on an arbitrary rotation inside the split pair — and a rung builds
    no RDMs, keeps no state the calculation uses and is thrown away the moment the rule has
    read its energies. Leaving the gate on "raise" here refuses the *ladder* for a property
    of the trial ensemble: measured on the first Tier-3 system, where a 16-root rung at
    D = 8 has every Kramers pair split by the truncation and the run died before the window
    could be resolved at all. What the splitting means for the verdict is reported instead
    (:func:`kramers_split_cm`), which is the statement a user can act on; the production
    solve at the resolved count meets the gate unchanged.
    """
    state = start if start is not None else random_state(
        ttno, int(n_elec), int(max_bond), n_roots=int(n_roots), rng=rng, charge=charge)
    kwargs = dict(solve_kwargs)
    kwargs.setdefault("on_split", "warn")
    result = solve_ttn(ttno, state, max_bond=int(max_bond), n_elec=int(n_elec),
                       max_sweeps=int(max_sweeps), boundary_check=0, weights=None,
                       memory_plan=memory_plan, plan_rdms=False, report=False,
                       **kwargs)
    return (np.asarray(result.energies, dtype=float), result.state, int(result.n_sweeps),
            bool(result.converged))


def pilot_estimate(ttno: TTNO, n_elec: int, window: EnergyWindow, *,
                   n_roots: Optional[int] = None, cap: int = PILOT_CAP,
                   sweeps: int = PILOT_SWEEPS, ceiling: Optional[int] = None,
                   rng=None, charge=None, solve_kwargs: Optional[Dict[str, Any]] = None,
                   report: bool = True, level: int = logging.INFO) -> PilotEstimate:
    """A short campaign at a small cap, read by the rule for the ladder's first rung only.

    ``n_roots`` defaults to :data:`PILOT_ROOTS` (twice an upstream estimate is the caller's
    business), rounded to a whole pair on an odd-electron system and clamped by ``ceiling``
    — the window's cap and the topology's two-site capacity. The campaign is bounded by
    ``sweeps`` and **need not converge**: what is read off it is how many roots lie under the
    cutoff, and a pilot that stopped on its budget still answers that to the accuracy a first
    rung deserves.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if ceiling is None:
        # The same bound the ladder uses: a pilot asked for more roots than the tour's
        # narrowest two-site window can hold is refused by the sweep, and a *pilot* has no
        # business raising that.
        ceiling = two_site_capacity(ttno.graph, n_elec)
    asked = min(PILOT_ROOTS if n_roots is None else int(n_roots),
                int(ceiling), window.max_states)
    # ⚠ Whole Kramers pairs, and the clamp is what decides the direction: rounding *up*
    # past a ceiling that is itself odd would ask for an ensemble the tour cannot hold.
    if int(n_elec) % 2 == 1 and asked % 2 == 1:
        asked -= 1
    asked = max(1, asked)
    tic = time.process_time()
    energies, _, n_sweeps, converged = _solve_roots(
        ttno, n_elec, asked, max_bond=int(cap), rng=rng, charge=charge,
        solve_kwargs=dict(solve_kwargs or {}), memory_plan=True, max_sweeps=int(sweeps))
    cpu = time.process_time() - tic
    verdict = resolve_window(energies, window)
    estimate = PilotEstimate(n_roots=asked, cap=int(cap), n_sweeps=n_sweeps,
                             cpu_seconds=float(cpu), count=max(1, int(verdict.count)),
                             complete=bool(verdict.complete), converged=converged,
                             relative_cm=verdict.relative_cm)
    if report:
        estimate.report(log, level=level)
    return estimate


class NetworkSpectrumOracle:
    """:class:`~kuiva.util.window.SpectrumOracle` over one TTNO at fixed integrals.

    A rung at ``n`` roots solves ``min(n + WITNESS_ROOTS, ceiling)`` of them to convergence
    on a cold state and returns the whole converged spectrum, so the rule reads a witness
    above the count it is testing (module docstring). The last rung's state is kept on
    :attr:`state` for :func:`truncate_roots` to hand on as the production warm start.

    The memory plan is printed for a root count larger than any already planned and not
    again: every rung refuses through :func:`kuiva.dmrg.sweep.solve_ttn`'s own per-bond
    requirement whether or not the table was printed, and a table per rung per round would
    be noise in a file that *is* the output.
    """

    def __init__(self, ttno: TTNO, n_elec: int, *, max_bond: int, ceiling: int,
                 e_core: float = 0.0, rng=None, charge=None,
                 max_sweeps: int = 30, planned: int = 0, warm: bool = False,
                 solve_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.ttno, self.n_elec = ttno, int(n_elec)
        self.max_bond, self.ceiling = int(max_bond), int(ceiling)
        self.e_core = float(e_core)
        self.max_sweeps = int(max_sweeps)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.charge = charge
        self.solve_kwargs = dict(solve_kwargs or {})
        #: The last rung's converged ensemble (witness roots included).
        self.state: Optional[TTNState] = None
        #: Sweeps spent per rung, in order — the network's cost unit for the rung table.
        self.sweeps: List[int] = []
        self.n_apply: Optional[int] = None
        self._planned = int(planned)
        #: Grow a rung from the previous one's ensemble (:func:`pad_roots`) instead of
        #: rebuilding it cold. ⚠ Measured-only and off by default (module docstring).
        self.warm = bool(warm)

    def spectrum(self, n_roots: int) -> np.ndarray:
        asked = min(int(n_roots) + WITNESS_ROOTS, self.ceiling)
        if asked < int(n_roots):                        # pragma: no cover - defensive
            raise RuntimeError("the ladder asked for {} roots above the ceiling {}"
                               .format(n_roots, self.ceiling))
        plan = asked > self._planned
        self._planned = max(self._planned, asked)
        start = (pad_roots(self.state, asked, self.rng)
                 if self.warm and self.state is not None else None)
        try:
            energies, state, n_sweeps, converged = _solve_roots(
                self.ttno, self.n_elec, asked, max_bond=self.max_bond, rng=self.rng,
                charge=self.charge, solve_kwargs=self.solve_kwargs, memory_plan=plan,
                max_sweeps=self.max_sweeps, start=start)
        except TwoSiteCapacityError as exc:
            # ⚠ The structural ceiling is computed over the FULL allowed sector set, which a
            # capped state need not have kept — so a cap tight enough to truncate a bond can
            # still put a rung past what that bond holds. Refused, never truncated, and the
            # refusal says which of the two it was.
            raise TwoSiteCapacityError(
                "{}\n   These are the {} roots of a rung of an energy window's ladder (the "
                "count under test plus a witness pair), not a count anybody stated: the "
                "window cannot be resolved on this topology at this bond dimension. Give the "
                "network a coarser node partition, raise max_bond, lower the cutoff, or "
                "state a count".format(exc, asked)) from exc
        self.state = state
        self.sweeps.append(n_sweeps)
        log.debug("window rung: %d roots (%d asked for) in %d sweeps, converged=%s",
                  asked, n_roots, n_sweeps, converged)
        if not converged:
            # ⚠ A rung is read by the rule as if it were the spectrum, so an unconverged one
            # is a verdict about a state that is not stationary. The resolution runs at
            # accepted points only (starting or converged orbitals), where there is nothing
            # to reject, so this propagates rather than becoming a rejected step.
            from ..util.errors import SolverFailure
            raise SolverFailure(
                "a rung of the energy window's ladder ({} roots, cap {}) did not converge in "
                "{} sweeps, so the spectrum the cutoff would be read from is not stationary. "
                "Raise max_sweeps, loosen conv_tol, or state a count"
                .format(asked, self.max_bond, self.max_sweeps))
        return energies + self.e_core


@dataclass
class NetworkResolution:
    """A resolved window on the network: the rule's answer and the ensemble it was read on."""

    resolution: WindowResolution
    #: The converged ensemble truncated to the resolved count — the production warm start.
    state: TTNState
    pilot: Optional[PilotEstimate] = None
    #: Sweeps per rung, in order.
    sweeps: List[int] = field(default_factory=list)
    #: The largest root count whose memory plan has been printed — carried back so a later
    #: round does not print the same table again (INFO is the output file).
    planned: int = 0


def resolve_network_window(ttno: TTNO, n_elec: int, window: EnergyWindow, *,
                           max_bond: int, e_core: float = 0.0,
                           estimate: Optional[int] = None, n_prev: Optional[int] = None,
                           rng=None, charge=None, max_sweeps: int = 30,
                           solve_kwargs: Optional[Dict[str, Any]] = None,
                           pilot: bool = True, pilot_cap: int = PILOT_CAP,
                           pilot_sweeps: int = PILOT_SWEEPS,
                           pilot_roots: Optional[int] = None, planned: int = 0,
                           warm_growth: bool = False,
                           estimate_source: str = "the upstream estimate",
                           where: str = "", report: bool = True,
                           level: int = logging.INFO) -> NetworkResolution:
    """Drive the ladder of :func:`kuiva.util.window.resolve_states` on this network.

    ``estimate`` is a first rung from upstream (a cheap CI's resolved count); with neither it
    nor ``EnergyWindow(initial=)`` a pilot campaign supplies one. ``n_prev`` is the previous
    round's count, which engages the rule's dead band and makes the first rung at least that
    count plus a witness pair.

    ⚠ The ladder's ceiling is the smaller of the window's own cap, the particle-number
    sector and the topology's two-site capacity. Reaching the capacity without a verdict is
    **refused** — a window that needs more roots than the network can represent is not a
    window this topology can answer, and truncating the ensemble to the ones it can hold
    would be the cut a window exists to forbid.
    """
    n_modes = sum(len(c) for c in ttno.graph.contents)
    sector = sector_upper_bound(n_modes, n_elec)
    capacity = two_site_capacity(ttno.graph, n_elec)
    ceiling = max(1, min(sector, capacity))
    # ⚠ Whole Kramers pairs, at the ceiling as well as at every rung: with an odd electron
    # count an odd ensemble splits a degenerate pair and the state-averaging gate refuses the
    # solve — so a capacity that happens to be odd buys one root that can never be asked for.
    if int(n_elec) % 2 == 1 and ceiling % 2 == 1:
        ceiling -= 1
    if ceiling < 1 or (int(n_elec) % 2 == 1 and ceiling < 2):
        raise ValueError(
            "this network cannot hold an ensemble at all: the narrowest two-site window of "
            "the tour has dimension {}, and with {} active electrons a state average is "
            "whole Kramers pairs. Give the network a topology whose bonds separate more "
            "orbitals (a coarser node partition)".format(capacity, n_elec))
    if pilot and estimate is None and window.initial is None:
        measured = pilot_estimate(ttno, n_elec, window, n_roots=pilot_roots, cap=pilot_cap,
                                  sweeps=pilot_sweeps, ceiling=ceiling, rng=rng,
                                  charge=charge, solve_kwargs=solve_kwargs, report=report,
                                  level=level)
        estimate = measured.count
    else:
        measured = None
    oracle = NetworkSpectrumOracle(ttno, n_elec, max_bond=max_bond, ceiling=ceiling,
                                   e_core=e_core, rng=rng, charge=charge,
                                   max_sweeps=max_sweeps, planned=planned,
                                   warm=warm_growth, solve_kwargs=solve_kwargs)
    resolution = resolve_states(oracle, window, n_elec=n_elec, space_size=ceiling,
                                estimate=estimate, n_prev=n_prev, growth=NETWORK_GROWTH,
                                witness=WITNESS_KIND)
    if measured is None and estimate is not None \
            and resolution.first_rung_source == "the upstream estimate":
        # ⚠ Say *which* estimate. A later round's first rung comes from what this
        # calculation already resolved, not from anything upstream of it, and a source line
        # that claimed otherwise would make the handoff unreadable in the output.
        resolution.first_rung_source = estimate_source
    if resolution.verdict.spans_space and ceiling < sector:
        raise ValueError(
            "the energy window {!r} could not be resolved on this network: the ladder "
            "reached {} roots, which is every state the tour's narrowest two-site window can "
            "hold ({} states, whole Kramers pairs; the particle-number sector holds {}), and "
            "the manifold rule still found no cut above the cutoff. A window is resolved "
            "against a converged witness root ABOVE the count -- a local extra root would "
            "prove a window incomplete and never complete -- so an ensemble the "
            "network cannot represent is refused rather than truncated to one it can. Give "
            "the network a topology whose bonds separate more orbitals (a coarser node "
            "partition), lower the cutoff, or state a count"
            .format(window, ceiling, capacity, sector))
    if measured is not None:
        resolution.first_rung_source = "the pilot sweep at D = {}".format(pilot_cap)
        resolution.extra["pilot"] = {"roots": measured.n_roots, "cap": measured.cap,
                                     "sweeps": measured.n_sweeps,
                                     "cpu_seconds": round(measured.cpu_seconds, 3),
                                     "count": measured.count}
    resolution.extra["sweeps"] = list(oracle.sweeps)
    split = kramers_split_cm(resolution.energies, resolution.count, n_elec)
    if split is not None:
        resolution.extra["kramers_split_cm"] = round(float(split), 4)
        if split > window.gap_cm:
            # ⚠ Loud, because it is the one thing that can make a clean-looking verdict wrong
            # on this route: the rule read a spectrum whose own Kramers degeneracies the
            # truncation had broken, so the "cut" it found may sit inside a manifold.
            log.warning(
                "the resolved ensemble's Kramers pairs are split by up to %.2f cm^-1 at "
                "max_bond = %d, which is more than the window's manifold gap (%.2f cm^-1): "
                "with an odd electron count every level is at least doubly degenerate by "
                "theorem, so that splitting is the TRUNCATION's and the count %d was read "
                "from a spectrum whose degeneracies are already broken. Raise max_bond until "
                "it falls below the gap before trusting this cut",
                split, max_bond, window.gap_cm, resolution.count)
    if report:
        resolution.report(log, level=level, where=where)
        out.entry(log, "sweeps per rung", ", ".join(str(s) for s in oracle.sweeps), "",
                  "a rung is a whole sweep campaign on this route", level=level)
    if oracle.state is None:                             # pragma: no cover - defensive
        raise RuntimeError("the ladder returned without solving anything")
    return NetworkResolution(resolution=resolution,
                             state=truncate_roots(oracle.state, resolution.count),
                             pilot=measured, sweeps=list(oracle.sweeps),
                             planned=int(oracle._planned))


__all__ = ["NetworkResolution", "NetworkSpectrumOracle", "PilotEstimate",
           "kramers_split_cm", "pad_roots",
           "pilot_estimate", "resolve_network_window", "sector_upper_bound",
           "truncate_roots", "NETWORK_GROWTH", "PILOT_CAP", "PILOT_ROOTS",
           "PILOT_SWEEPS", "WITNESS_KIND", "WITNESS_ROOTS"]
