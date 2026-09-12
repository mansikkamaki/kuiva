"""The round loop of a windowed CASSCF: resolve at fixed orbitals, optimize at a fixed count.

A state count resolved from an energy cutoff (:mod:`kuiva.util.window`) is a statement about
*a spectrum*, and a CASSCF moves the spectrum. This module is the outer fixed-point loop that
reconciles the two, and its whole design is one decision:

⚠ **The count is resolved at fixed orbitals and held for a whole orbital optimization; it may
change only between rounds.** The optimizer never sees a window. Every inner run is the
validated driver (:func:`kuiva.mcscf.orbopt.optimize_orbitals`, or the event-gated sibling)
with an ordinary fixed-count solver, one smooth surface and nothing about its contract
touched — so a windowed CASSCF that resolves to ``n`` on its first round and stays there
reproduces the fixed-count run at ``n`` **bitwise**.

The alternative — re-resolving inside the optimizer's loop — was rejected for a structural
reason rather than a practical one: a count change is not a variational adoption (a larger
average is *higher* by construction), so the event-gated driver's adoption rule cannot govern
it, and every mechanism of the optimizer — the quadratic model, accept/reject, the curvature
memory, the convergence test — would be trusted across a change of energy functional.

The loop
--------
::

    round 1: resolve at the starting orbitals   -> count n_1 (and the initial boundary report)
             optimize at n_1                    -> converged orbitals
             re-resolve there (n_prev = n_1)    -> verdict
             verdict == n_1  -> done
    round r: a sibling solver at n_r, warm from the re-resolution's vectors,
             a fresh optimization from round r-1's orbitals with fresh curvature,
             re-resolve, ... up to the window's max_rounds

Three rules bind what happens at the edges, and none of them silently picks a number:

* **The budget is total.** ``max_iter`` counts macro-iterations *across* rounds — the same
  reading a restart gives it — so a windowed run costs what a fixed-count run would plus the
  rounds it actually needed. A run that stops on its budget (or on a deadline, a signal or a
  callback) mid-round keeps that round's result and **reports** the verdict at the stopped
  orbitals as a warning rather than adopting it: a count change costs a whole further
  optimization, and there is no budget left to pay for it.
* **The round cap is not an error.** On reaching ``max_rounds`` with the verdict still
  changing, the last round's converged result is kept — it *is* a converged CASSCF at its own
  count — and the resolution is marked :attr:`~kuiva.util.window.WindowResolution.ambiguous`
  with the counts seen and the state that straddles the cutoff named. ⚠ **Never silently take
  the larger count**: both counts are self-consistent fixed points, the cutoff falls on a
  state the orbitals move across, and only the user can say which side they meant.
* **Curvature is chart-scoped.** A new count is a new energy functional, so each round starts
  the optimizer cold rather than transporting L-BFGS pairs across two surfaces. That is the
  same rule a checkpoint applies across a process boundary.

This module knows nothing about CI, tensor networks or integrals: the caller supplies a
``resolve`` and an ``optimize`` closure and the loop drives them, so the conventional CI and
the tensor-network route share one implementation of the loop and differ only in what a rung
and a macro-iteration cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from ..util.window import EnergyWindow, WindowResolution

log = get_logger(__name__)


@dataclass
class Round:
    """One resolve-optimize-re-resolve round."""

    index: int
    #: The count this round's orbital optimization ran at.
    n_states: int
    #: Macro-iterations spent in this round (not the running total).
    n_iterations: int
    energy: float
    grad_norm: float
    converged: bool
    #: The re-resolution at this round's converged orbitals.
    resolution: WindowResolution
    #: Whatever ``optimize`` returned — a :class:`~kuiva.mcscf.orbopt.CASSCFResult`.
    orbital: Any = None

    @property
    def verdict_count(self) -> int:
        return int(self.resolution.count)

    @property
    def changed(self) -> bool:
        return self.verdict_count != int(self.n_states)


@dataclass
class RoundsResult:
    """What the loop produced: the last round's orbitals, and how the count got there."""

    #: The solver at the count the result was optimized at — an ordinary fixed-count solver.
    solver: Any
    #: ``optimize``'s return value from the last round.
    orbital: Any
    rounds: List[Round]
    #: The resolution at the **starting** orbitals of round 1. ⚠ The one that says whether
    #: the trajectory was safe; ``None`` on a restart, where the trajectory (and the
    #: starting-orbital measurement) belongs to the run that wrote the file.
    initial: Optional[WindowResolution]
    #: The resolution at the converged orbitals of the last round.
    final: WindowResolution
    #: True when the loop ran out of rounds with the verdict still changing.
    ambiguous: bool = False
    #: True when the loop stopped because the macro-iteration budget was spent.
    budget_stopped: bool = False
    history: List[float] = field(default_factory=list)

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def n_states(self) -> int:
        return int(self.solver.n_states)

    def report(self, logger=None, *, level: int = logging.INFO) -> None:
        """The rounds table: one row per round, through the output grammar only."""
        logger = logger or log
        if not self.rounds:                                # pragma: no cover - defensive
            return
        out.subsection(logger, "state window: rounds")
        columns = [out.col_count("round", 6), out.col_count("states", 7),
                   out.col_count("iters", 6), out.col_energy("E_SA [Eh]"),
                   out.col_resid("|grad|"), out.Column("verdict", "{}", 32, align="<")]
        table = out.Table(logger, columns, level=level)
        table.start()
        for r in self.rounds:
            if not r.changed:
                verdict = "count unchanged"
            else:
                verdict = "count -> {}".format(r.verdict_count)
            if not r.converged:
                verdict += " (not converged)"
            table.row(r.index, r.n_states, r.n_iterations, r.energy, r.grad_norm, verdict)
        table.end("{} round{}, {} macro-iterations".format(
            self.n_rounds, "" if self.n_rounds == 1 else "s",
            sum(r.n_iterations for r in self.rounds)))
        if self.ambiguous:
            note = "ambiguous: the verdict was still moving when the rounds ran out"
        elif self.budget_stopped:
            note = ("not confirmed: the run stopped before the count could be re-tested at "
                    "a converged point")
        else:
            note = "stable across the last round"
        out.entry(logger, "resolved state count", self.n_states, "", note, level=level)


def optimize_rounds(window: EnergyWindow, solver: Any, coeff: np.ndarray, *,
                    resolve: Callable[..., Tuple[WindowResolution, Any]],
                    optimize: Callable[..., Any],
                    max_iter: int, start_iteration: int = 0,
                    resolved: Optional[Tuple[WindowResolution, Any]] = None,
                    report: bool = True,
                    level: int = logging.INFO) -> RoundsResult:
    """Drive the resolve-optimize loop of the module docstring.

    Parameters
    ----------
    window : EnergyWindow
        The request. Only :attr:`~kuiva.util.window.EnergyWindow.max_rounds` is read here;
        the cutoff and the manifold rule live inside ``resolve``.
    solver : object
        The solver as the caller built it — unresolved (``n_states is None``) unless
        ``resolved`` supplies the first resolution.
    coeff : ndarray
        The starting orbitals.
    resolve : callable
        ``resolve(coeff, solver, n_prev=..., where=...) -> (resolution, solver_at_count)``.
        Runs the ladder at *fixed* orbitals and returns both the verdict and a solver built
        at its count, warm-started from the ladder's own vectors. ⚠ It is called only at
        **accepted** points (the starting orbitals and a converged set), where there is
        nothing to reject, so a failure propagates rather than becoming a rejected step.
    optimize : callable
        ``optimize(coeff, solver, round_index=..., start_iteration=..., max_iter=...) ->
        result``. One whole orbital optimization at a fixed count; the result must carry
        ``coeff``, ``energy``, ``grad_norm``, ``converged``, ``n_iterations`` and ``history``.
        ⚠ ``n_iterations`` is read as the **running total** (the optimizer counts from
        ``start_iteration``), which is what makes ``max_iter`` a budget across rounds.
    start_iteration : int
        Macro-iterations already spent before this loop — a restart's own count, which the
        budget continues from.
    resolved : tuple, optional
        ``(resolution, solver_at_count)`` for round 1, supplied instead of resolving at the
        starting orbitals — a restart, where the count comes from the checkpoint and the
        starting-orbital measurement belongs to the run that wrote it.
    """
    if resolved is None:
        initial, solver = resolve(coeff, solver, n_prev=None, where="starting orbitals")
    else:
        initial, solver = None, resolved[1]
    final: Optional[WindowResolution] = initial if resolved is None else resolved[0]
    rounds: List[Round] = []
    history: List[float] = []
    used = int(start_iteration)
    ambiguous = False
    budget_stopped = False
    orbital = None

    for index in range(1, int(window.max_rounds) + 1):
        count = int(solver.n_states)
        if rounds and used >= int(max_iter):
            # ⚠ Not silent: a round that never ran is not a round whose count was confirmed.
            # ⚠ Round 1 always runs, however little budget is left, so that a windowed run
            # whose budget is already spent returns the same thing a fixed-count one does
            # (an iterate, reported as unconverged) rather than nothing at all.
            log.warning("the macro-iteration budget (max_iter = %d) was spent before round "
                        "%d could start, so the state count %d is the last one that was "
                        "optimized at and the verdict below was never re-tested. max_iter "
                        "counts macro-iterations across rounds", int(max_iter), index, count)
            budget_stopped = True
            break
        if report:
            out.entry(log, "state window: round", index, "",
                      "{} states, macro-iterations {}..{}".format(count, used + 1,
                                                                  int(max_iter)),
                      level=level)
        orbital = optimize(coeff, solver, round_index=index, start_iteration=used,
                           max_iter=int(max_iter))
        spent = int(orbital.n_iterations) - used
        used = int(orbital.n_iterations)
        coeff = orbital.coeff
        history.extend(float(e) for e in getattr(orbital, "history", ()))
        final, grown = resolve(coeff, solver, n_prev=count, where="converged orbitals")
        rounds.append(Round(index=index, n_states=count, n_iterations=spent,
                            energy=float(orbital.energy),
                            grad_norm=float(orbital.grad_norm),
                            converged=bool(orbital.converged), resolution=final,
                            orbital=orbital))
        if int(final.count) == count:
            break
        if not orbital.converged:
            # ⚠ Reported, never adopted. A count change costs a whole further optimization
            # and the run has already stopped paying; adopting the verdict here would label
            # an unconverged iterate with a count nothing optimized at.
            log.warning("the orbital optimization of round %d stopped without converging "
                        "(|g| = %.3e), and the energy window reads %d states at the "
                        "orbitals it stopped on against the %d it ran at. The verdict is "
                        "reported and NOT adopted: it is a statement about orbitals that "
                        "are an iterate rather than a result",
                        index, float(orbital.grad_norm), int(final.count), count)
            budget_stopped = True
            break
        if index >= int(window.max_rounds):
            ambiguous = True
            _warn_ambiguous(window, rounds, final)
            break
        solver = grown

    if final is None:                                      # pragma: no cover - defensive
        raise RuntimeError("the window round loop ended without a resolution")
    if orbital is None:                                    # pragma: no cover - defensive
        raise RuntimeError("the window round loop ended without optimizing anything")
    if ambiguous:
        final.ambiguous = True
    result = RoundsResult(solver=solver, orbital=orbital, rounds=rounds, initial=initial,
                          final=final, ambiguous=ambiguous, budget_stopped=budget_stopped,
                          history=history)
    if report:
        result.report(log, level=level)
    return result


def _warn_ambiguous(window: EnergyWindow, rounds: List[Round],
                    final: WindowResolution) -> None:
    """Name the counts, the straddling state and its distance from the cutoff.

    ⚠ Both counts are converged fixed points of their own state average, so this is a
    statement about the *request*, not a failure of the calculation: the cutoff falls on a
    state the orbitals carry back and forth across it, and which side was meant is the one
    thing nothing here can decide.
    """
    counts = [r.n_states for r in rounds] + [int(final.count)]
    edge = final.verdict.edge_cm
    where = ""
    if edge is not None:
        where = (" The states around the cut sit at {:.2f} and {:.2f} cm^-1 above the "
                 "lowest, against a cutoff of {:.2f} cm^-1.".format(edge[0], edge[1],
                                                                    window.cutoff_cm))
    log.warning(
        "the energy window %r did not settle in %d rounds: the resolved count went %s, and "
        "the result kept is the converged CASSCF at %d states -- the last one that was "
        "actually optimized.%s Neither count is wrong; a state crosses the cutoff as the "
        "orbitals relax, and only you can say which side of it the calculation is about. "
        "Move the cutoff clear of that state, state a count, or raise max_rounds",
        window, int(window.max_rounds), " -> ".join(str(c) for c in counts),
        int(rounds[-1].n_states), where)


__all__ = ["Round", "RoundsResult", "optimize_rounds"]
