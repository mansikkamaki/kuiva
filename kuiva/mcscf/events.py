"""Event-gated orbital optimization for adaptive CI and DMRG surfaces.

The problem
-----------
:func:`kuiva.mcscf.orbopt.optimize_orbitals` is a trust-region method on **one** smooth
surface. Hand it a solver that re-chooses its own internal space — a selected CI, later a DMRG
that re-distributes bond dimensions — and it is no longer optimizing a function: the objective
is a family of smooth surfaces indexed by that space, and the iterate hops between them. Every
mechanism in the optimizer then misfires in a characteristic way. The quadratic model is
trusted across a hop. The accept/reject test reads a hop as a failed step and shrinks the
trust radius for it. L-BFGS pairs mix two surfaces. Convergence is judged against noise.

Measured on a Ti(2+) CAS(10,18) proxy with a 2000-determinant selected CI (the record is in
measured): the second-order optimizer stalls at ``|g| = 1.8e-1``,
which *equals* the measured surface-to-surface gradient noise. In the tail the hop is a
**single determinant** out of 2000 moving the state-averaged energy by ``±9.8e-4 Eh``, while
genuine per-step progress has decayed to ``1e-6 Eh`` — the optimizer is riding noise. It then
rejects sixteen consecutive steps with the space perfectly stable, because the trust radius
has collapsed and the curvature memory is poisoned.

The mechanism
-------------
The controller here is **not** a new step engine: the quasi-Newton and augmented-Hessian steps
of :class:`~kuiva.mcscf.orbopt.OrbitalOptimizer` are used unchanged, and no linear algebra in
this module is new. What is new is who owns the space.

1. **The optimizer owns space changes; the solver never re-selects on its own.** The inner
   loop runs entirely on the incumbent surface, which is smooth and deterministic — exactly
   what the exact gradient, the chart-corrected Hessian, the augmented-Hessian solve and the
   warm starts are validated for. This makes the accept/reject test chart-consistent *by
   construction*: a trial step is evaluated on the same surface the model was built from, so
   a hop can never masquerade as a failed step.
2. **Space changes are proposed, compared variationally, and adopted or refused.** At an
   *event* the solver runs a fresh selection at the current integrals; the candidate is
   adopted only if it lowers the energy — at those same integrals — by more than
   :data:`DEFAULT_TAU`. The trajectory is therefore monotone in energy across events, which
   is the property that makes the whole thing a descent method again. On the benchmark above,
   the same optimizer with the same settings reaches ``|g| = 1.2e-4`` and 35 mEh lower energy,
   and the fresh selection wins **once in thirty calls**.
3. **Events are cheap early and rare later.** One after every accepted macro-iteration to
   begin with, backing off exponentially after each refusal, and — mandatory — one at inner
   convergence. That last one is the termination test: converged means ``|g| < conv_grad`` on
   the incumbent surface *and* a fresh selection cannot improve it. That is the honest fixed
   point for an adaptive-space MCSCF; a gradient norm alone is a statement about one chart.
4. **Curvature memory is scoped to the surface that produced it**
   (:meth:`~kuiva.mcscf.orbopt.OrbitalOptimizer.reset_chart`), and the trust radius is
   restored to a floor on adoption rather than inherited from a collapsed state.
5. **A failed solve is an event, not an exception.** ``SolverFailure`` at a trial point
   rejects the step; at an event it refuses the adoption. Only at the starting point, where
   there is nothing to fall back to, does it propagate.

The open question this leaves
-----------------------------
A proposal costs whatever a solve costs, and for a DMRG that is the whole calculation. The
backoff makes proposals rare and the measurement says the trade is affordable (roughly 3.5×
rarer for 2.7× in the converged gradient), but it is still a *linear* trade — thinning proposals
costs quality in proportion. The way out, if one is needed, is a **cheaper "is the space still
stable?" predicate than a full proposal**, supplied by the solver as an optional fifth method:
a DMRG can inspect its discarded weight, and a selected CI its perturbative residual, without
re-solving anything. That is deliberately not designed here — there is nothing to measure it on
until the network solver is the consumer, and a predicate tuned against the cheap CI would be tuned against the case
that does not need it.

⚠ What this does **not** fix
----------------------------
The neglected orbital–CI coupling of the two-step scheme is untouched, and it is what sets the
achievable gradient on stiff cases (``|g| ~ 1e-3``–``1e-4`` on a *frozen* chart). Event gating
recovers that quality on an adaptive surface; it does not exceed it. Do not read a converged
``|g| = 1e-4`` here as evidence that ``conv_grad = 1e-5`` is reachable.

⚠ A solver that re-selects inside ``solve`` defeats all of this silently. See
:mod:`kuiva.mcscf.adaptive`.

References
----------
* Trust-region globalization, accept/reject, and the level-shift trust radius the inner
  engines use: J. Nocedal, S. J. Wright, "Numerical Optimization", 2nd ed., Springer (2006);
  A. R. Conn, N. I. M. Gould, P. L. Toint, "Trust-Region Methods", SIAM (2000).
* Optimization under inexact or adaptive function information, the closest existing framing —
  though those methods *model* the noise where this one removes it by taking control of its
  source: R. G. Carter, SIAM J. Numer. Anal. 28, 251 (1991), doi:10.1137/0728014;
  A. S. Berahas, R. H. Byrd, J. Nocedal, SIAM J. Optim. 29, 965 (2019),
  doi:10.1137/18M1190164.
* The MCSCF machinery this drives is unchanged; its references are in
  :mod:`kuiva.mcscf.orbopt`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..integrals.transform import ThreeIndexAO
from ..util import output as out
from ..util import threads
from ..util.logging import get_logger
from ..util.timing import timer
from .adaptive import AdaptiveCISolver, Proposal, SolverFailure, as_adaptive_solver
from .orbopt import (CASIntegrals, CASSCFResult, DEFAULT_MAX_STEP, OrbitalOptimizer,
                     OrbitalSpaces, SECOND_ORDER_START, kramers_rotation_note,
                     measure_time_odd_curvature, resolve_kramers_rotation)

log = get_logger(__name__)

#: Variational improvement [Eh] a candidate space must show before it is adopted. It admits
#: the rare real improvement and refuses the knife-edge flips, whose amplitude on the Ti(2+)
#: benchmark is ~1e-3 Eh but whose *sign* is what matters — a flip is adopted only when it
#: genuinely lowers the energy at this point. **Measured insensitive over two decades below
#: this** (1e-8 and 1e-6 agree to 1e-9 Eh in the final energy) and catastrophic above it: at
#: 1e-4 the one winning proposal of the benchmark run is refused and the calculation silently
#: degenerates to a frozen space. ⚠ It is *absolute* rather than scaled to ``conv_energy``;
#: whether it should scale is untested, and the flat two decades mean there is no pressure to
#: find out.
DEFAULT_TAU = 1.0e-6
#: Accepted macro-iterations between events, at the start. **One**, and the measurement is
#: emphatic: on the Ti(2+) benchmark the entire gain rests on catching one early proposal, and
#: starting at 2 or 4 steps over it — recovering 0.5 mEh of the available 7.4 mEh while saving
#: one or two selections. A proposal is cheap relative to a macro-iteration for a selected CI.
DEFAULT_EVENT_INTERVAL = 1
#: Cap on the exponential backoff. ⚠ Where a proposal is expensive (a DMRG, where it costs a
#: full solve) the knob is :data:`DEFAULT_EVENT_INTERVAL`, not this: raise it so proposals
#: start rare. Measured on the cost proxy, proposals can be made 3.5x rarer for about a 2.7x
#: cost in the converged gradient, and the termination guarantee is untouched either way
#: because the mandatory event at inner convergence is not on the cadence.
DEFAULT_MAX_EVENT_INTERVAL = 16
#: Trust radius [rad] restored on adoption, as a fraction of ``max_step``.
DEFAULT_TRUST_FLOOR_FRACTION = 0.25


@dataclass
class EventRecord:
    """One proposal and what was done with it — the audit trail of the space trajectory."""

    iteration: int
    incumbent: float                     # [Eh] at the current integrals, incumbent space
    candidate: Optional[float]           # [Eh] at the same integrals, candidate space
    adopted: bool
    reason: str

    @property
    def gain(self) -> float:
        """Energy lowering offered by the candidate [Eh]; 0 when there was none."""
        return 0.0 if self.candidate is None else self.incumbent - self.candidate


@dataclass
class EventCASSCFResult(CASSCFResult):
    """:class:`~kuiva.mcscf.orbopt.CASSCFResult` plus the space-trajectory bookkeeping.

    ⚠ ``converged`` here means more than it does on the plain driver: the gradient is below
    threshold **and** the last event refused to improve the space (:attr:`event_stable`). A
    run that hits ``max_iter`` with a small gradient but an unstable space is not converged,
    and saying so is the point.

    ``n_solver_failures`` is **not** declared here: it belongs to
    :class:`~kuiva.mcscf.orbopt.CASSCFResult`, since the plain driver rejects a refused trial
    point too and one count is better than two spellings of the same number.
    """

    n_events: int = 0
    n_adoptions: int = 0
    n_refusals: int = 0
    event_stable: bool = False
    events: List[EventRecord] = field(default_factory=list)


def _run_event(solver: AdaptiveCISolver, ints: CASIntegrals, energy: float, tau: float,
               iteration: int) -> Tuple[Optional[Proposal], EventRecord]:
    """Propose a space at the current point and decide whether to adopt it.

    The comparison is variational and local: both energies are evaluated at ``ints``, so the
    only difference between them is the space. Adoption therefore lowers the energy by
    construction, which is what keeps the trajectory monotone across events.
    """
    try:
        proposal = solver.propose(ints)
    except SolverFailure as exc:
        log.debug("the proposal at iteration %d failed to solve (%s); keeping the incumbent "
                  "space", iteration, exc)
        return None, EventRecord(iteration, energy, None, False, "solve failed")
    if proposal is None:
        return None, EventRecord(iteration, energy, None, False, "same space")
    gain = energy - float(proposal.energy)
    if gain > tau:
        solver.adopt(proposal.key)
        log.debug("adopted a new space at iteration %d: %.3e Eh lower (%s)",
                  iteration, gain, proposal.label)
        return proposal, EventRecord(iteration, energy, float(proposal.energy), True,
                                     proposal.label or "adopted")
    return None, EventRecord(iteration, energy, float(proposal.energy), False,
                             "gain {:.1e} < tau".format(gain))


@threads.blas_stage
def optimize_orbitals_events(
        factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
        spaces: OrbitalSpaces, ci_solver: Any, *,
        e_nuc: float = 0.0, max_iter: int = 50, conv_grad: float = 1e-4,
        conv_energy: float = 1e-8, max_step: float = DEFAULT_MAX_STEP, memory: int = 10,
        active_active: bool = False, mode: str = "auto",
        kramers_rotation: Any = "auto", n_active_elec: Optional[int] = None,
        kramers_stability: Any = "auto", second_order_start: float = SECOND_ORDER_START,
        tau: float = DEFAULT_TAU, event_interval: int = DEFAULT_EVENT_INTERVAL,
        max_event_interval: int = DEFAULT_MAX_EVENT_INTERVAL,
        trust_floor: Optional[float] = None, keep_memory_on_adopt: bool = False,
        callback: Optional[Callable[[dict], Optional[bool]]] = None,
        report: bool = True,
        extra_columns: Sequence[Tuple[Any, Callable[[], Any]]] = ()
        ) -> EventCASSCFResult:
    """Event-gated MCSCF driver for a solver whose internal space adapts.

    A **sibling** of :func:`kuiva.mcscf.orbopt.optimize_orbitals`, not a mode of it. The two
    axes are independent: ``mode`` still selects the inner step engine (``"auto"``,
    ``"quasi-newton"``, ``"second-order"``), while event gating is the outer control of the
    surface. Folding the second axis into ``mode`` would make ``mode="event"`` silently also a
    choice of step engine, and would put an outer loop inside a signature documented as one
    smooth surface. Keeping them separate leaves ``optimize_orbitals`` untouched — it remains
    the validated smooth-surface driver, and it is what a plain callable should still use.

    ``ci_solver`` is an :class:`~kuiva.mcscf.adaptive.AdaptiveCISolver`. A plain
    ``ci_solver(ints) -> (energy, gamma, gamma2)`` callable is accepted and wrapped, in which
    case every proposal comes back empty, nothing is ever adopted, and the trajectory is
    **bitwise** that of the plain driver — correct for an exact CI,
    and ⚠ **not** a way to gate a callable that re-selects internally.

    Knobs beyond the plain driver's:

    ``tau``
        Variational improvement required of a candidate space (:data:`DEFAULT_TAU`).
    ``event_interval`` / ``max_event_interval``
        Accepted macro-iterations between proposals, and the cap the exponential backoff
        climbs to. A refusal doubles the interval; an adoption resets it.
    ``trust_floor``
        Trust radius restored on adoption; ``None`` means ``max_step / 4``.
    ``keep_memory_on_adopt``
        Transport the L-BFGS pairs across an adoption instead of clearing them. Off by
        default — see :meth:`~kuiva.mcscf.orbopt.OrbitalOptimizer.reset_chart`.
    ``kramers_rotation`` / ``n_active_elec`` / ``kramers_stability``
        As in the plain driver: ``"auto"`` constrains the rotation to keep Kramers-paired
        orbitals paired where the incoming ones are
        (:func:`~kuiva.mcscf.orbopt.resolve_kramers_rotation`). ⚠ A chart change moves the
        *space*, never the orbitals or the electron count, so the decision made at entry
        holds for the whole run — an adopted space cannot make a paired orbital set unpaired.
        ⚠ **The time-odd stability test here REPORTS and never releases**, which is the one
        place this driver deliberately does less than the plain one: releasing the constraint
        means continuing an unconstrained optimization from a displaced point, and on an
        adaptive surface that is a chart change whose incumbent space was chosen for the
        symmetric solution. What a saddle verdict means here is *re-run this unconstrained*,
        and the warning says so. ``kramers_stability=False`` switches the measurement off.

    ``extra_columns``
        Solver-specific ``(Column, zero-argument getter)`` pairs appended to the iteration
        table, exactly as in :func:`~kuiva.mcscf.orbopt.optimize_orbitals` — the two drivers
        take the same argument so a caller does not have to know which one it reached.

    ``callback(info)`` behaves as in the plain driver (returning ``False`` stops the run), with
    ``event`` added to the info mapping.
    """
    solver = as_adaptive_solver(ci_solver)
    c = np.ascontiguousarray(c_spinor, dtype=np.complex128)
    kramers, pairing = resolve_kramers_rotation(kramers_rotation, c, spaces,
                                                active_active=active_active)
    opt = OrbitalOptimizer(spaces, max_step=max_step, memory=memory,
                           active_active=active_active, mode=mode,
                           second_order_start=second_order_start, conv_grad=conv_grad,
                           kramers=kramers)
    floor = (DEFAULT_TRUST_FLOOR_FRACTION * max_step if trust_floor is None
             else float(trust_floor))
    interval = max(1, int(event_interval))
    max_interval = max(interval, int(max_event_interval))

    if report:
        out.subsection(log, "Orbital optimization (event-gated)")
        out.entries(log, [
            ("inactive / active / virtual spinors",
             "{} / {} / {}".format(spaces.n_inactive, spaces.n_active, spaces.n_virtual)),
            ("orbital rotation parameters", opt.n_parameters, "", "complex"),
            ("rotation", "Kramers constrained" if kramers else "general complex", "",
             kramers_rotation_note(kramers, pairing)),
            ("step engine", mode),
            ("adoption threshold tau", tau, "Eh", "variational, at fixed integrals",
             out.SCI_FMT),
            ("event interval", interval, "", "accepted iterations, doubling on refusal"),
            ("gradient convergence", conv_grad, "", "and a stable space", "{:.1e}"),
            ("maximum rotation per step", max_step, "rad", "", "{:.2f}"),
        ])
        table = out.Table(log, [out.col_iter(), out.col_energy("E [Eh]"), out.col_delta(),
                                out.col_resid("|g|"), out.Column("max rot", "{:.4f}", 8),
                                out.Column("step", "{}", 6, align="<"),
                                out.Column("event", "{}", 18, align="<"), out.col_time()]
                          + [column for column, _ in extra_columns])
        table.start()

    # The starting point establishes the incumbent surface. A failure here has nothing to
    # fall back on, so it propagates (D5).
    ints = CASIntegrals.build(factors, h_ao, c, spaces, e_nuc=e_nuc)
    energy, gamma, gamma2 = solver.solve(ints)
    history: List[float] = [energy]
    events: List[EventRecord] = []
    n_adoptions = n_refusals = n_failures = 0
    since_event = 0
    converged = False
    event_stable = False
    gnorm = np.inf
    it = 0

    for it in range(1, max_iter + 1):
        tag = ""
        de = 0.0
        with timer("macro-iteration") as t_it:
            step = opt.step(ints, gamma, gamma2, factors, c, energy=energy, h_ao=h_ao)
            gnorm = step.grad_norm

            if gnorm < conv_grad:
                # Inner convergence on this chart. The mandatory event is the termination
                # test: a fixed point of the orbitals that a better space would move is not
                # a fixed point of the calculation.
                #
                # The step just built is discarded here, and when the event adopts, so is the
                # augmented-Hessian solve that produced it. That is deliberate: the gradient
                # is only available *from* the step, and computing it separately every
                # iteration would add a J/K build to all of them to save one solve on the at
                # most (n_adoptions + 1) iterations that reach this branch.
                proposal, record = _run_event(solver, ints, energy, tau, it)
                events.append(record)
                if proposal is None:
                    converged = True
                    event_stable = True
                    n_refusals += 1
                    tag = "stable"
                else:
                    energy, gamma, gamma2 = (float(proposal.energy), proposal.gamma,
                                             proposal.gamma2)
                    history.append(energy)
                    opt.reset_chart(trust_floor=floor, keep_memory=keep_memory_on_adopt)
                    n_adoptions += 1
                    since_event = 0
                    interval = max(1, int(event_interval))
                    de = -record.gain
                    tag = "adopt"
            else:
                c_try = np.ascontiguousarray(c @ step.unitary)
                ints_try = CASIntegrals.build(factors, h_ao, c_try, spaces, e_nuc=e_nuc)
                try:
                    e_try, g_try, g2_try = solver.solve(ints_try)
                except SolverFailure as exc:
                    # A point the solver cannot evaluate is a point the optimizer must not
                    # move to. Rejecting shrinks the trust radius, which is the right
                    # response: the next trial point is closer to one that did work.
                    log.warning("the CI failed at the trial point of iteration %d (%s); the "
                                "step is rejected", it, exc)
                    opt.reject()
                    n_failures += 1
                    tag = "solve fail"
                else:
                    de = e_try - energy
                    if de <= conv_energy:                  # accept (a tiny rise is noise)
                        grad_new, _ = opt.gradient(ints_try, g_try, g2_try, factors, c_try)
                        opt.accept(grad_new)
                        c, ints, energy = c_try, ints_try, e_try
                        gamma, gamma2 = g_try, g2_try
                        history.append(energy)
                        since_event += 1
                        tag = ""
                        if since_event >= interval:
                            proposal, record = _run_event(solver, ints, energy, tau, it)
                            events.append(record)
                            since_event = 0
                            if proposal is None:
                                n_refusals += 1
                                interval = min(2 * interval, max_interval)
                                tag = "no change"
                            else:
                                energy, gamma, gamma2 = (float(proposal.energy),
                                                         proposal.gamma, proposal.gamma2)
                                history.append(energy)
                                opt.reset_chart(trust_floor=floor,
                                                keep_memory=keep_memory_on_adopt)
                                n_adoptions += 1
                                interval = max(1, int(event_interval))
                                de += -record.gain
                                tag = "adopt"
                    else:
                        opt.reject()
                        tag = "reject"
                        log.debug("rejected a step that raised the energy by %.3e Eh; trust "
                                  "radius now %.2e", de, opt.trust)

        if report:
            table.row(it, energy, de, gnorm, step.max_rotation,
                      "2nd" if step.second_order else "qn", tag, t_it.wall,
                      *[getter() for _, getter in extra_columns])
        if callback is not None:
            # Extended additively with the same keys the plain driver carries — the
            # orbitals, the RDMs, the spaces and the optimizer are what a checkpoint
            # writer (kuiva.io.checkpoint.CheckpointPolicy) needs, and they are already
            # in hand here. A callback that ignores them is unaffected.
            info = {"iteration": it, "energy": energy, "grad_norm": gnorm, "de": de,
                    "second_order": step.second_order, "converged": converged,
                    "n_hessian_matvec": opt.n_hessian_matvec, "wall": t_it.wall,
                    "event": tag, "n_adoptions": n_adoptions,
                    "space_key": solver.space_key(),
                    "coeff": c, "gamma": gamma, "gamma2": gamma2, "spaces": spaces,
                    "optimizer": opt, "trust": opt.trust, "history": history,
                    "ci_solver": solver, "e_nuc": e_nuc}
            if callback(info) is False:
                log.warning("orbital optimization stopped by callback at iteration %d "
                            "(|g| = %.3e); the result is the last iterate, not a converged "
                            "one", it, gnorm)
                break
        if converged:
            break

    n_events = len(events)
    # The time-odd stability of the constrained solution: measured here, never acted on
    # (see the docstring). It is meaningful at a converged point and nowhere else.
    curvature = None
    if converged and opt.kmap is not None and kramers_stability is not False and (
            kramers_stability is True
            or (kramers_rotation == "auto" and n_active_elec is not None
                and int(n_active_elec) % 2 == 0)):
        # ⚠ The same condition the plain driver releases on, minus the release: measuring
        # where a broken solution could not be an answer anyway (an odd count, an explicit
        # kramers_rotation=True) would spend Hessian-vector products on a verdict nobody
        # can act on. `True` is the diagnostic setting and measures regardless.
        curvature = measure_time_odd_curvature(opt, ints, factors, h_ao, c, gamma, gamma2)
    if report:
        table.end("converged" if converged else
                  "NOT converged in {} macro-iterations".format(max_iter))
        entries = [
            ("second-order steps taken", opt.n_second_order_steps),
            ("Hessian-vector products", opt.n_hessian_matvec),
            ("rejected steps", opt.n_rejected),
            ("space proposals", n_events, "", "{} adopted".format(n_adoptions)),
            ("failed CI solves", n_failures),
        ]
        if curvature is not None:
            entries.append(("lowest time-odd curvature", curvature.value, "Eh/rad^2",
                            "{}; {} products".format(curvature.verdict, curvature.n_matvec),
                            out.SCI_FMT))
        out.entries(log, entries)
    if curvature is not None and curvature.unstable:
        log.warning("the Kramers-constrained solution is a SADDLE: the orbital Hessian has "
                    "curvature %.3e Eh/rad^2 along a time-reversal-odd rotation, so a "
                    "time-reversal-broken solution lies below it. This driver does not "
                    "release the constraint on an adaptive surface; re-run with "
                    "kramers_rotation=False to follow it", curvature.value)
    if not converged:
        log.warning("orbital optimization did not converge in %d macro-iterations "
                    "(|g| = %.3e, target %.1e); the orbitals are the last iterate and may "
                    "not be stationary", max_iter, gnorm, conv_grad)
    return EventCASSCFResult(
        energy=energy, coeff=c, gamma=gamma, gamma2=gamma2, converged=converged,
        n_iterations=it, grad_norm=gnorm, history=history,
        n_hessian_matvec=opt.n_hessian_matvec,
        n_second_order_steps=opt.n_second_order_steps, n_rejected=opt.n_rejected,
        time_odd_curvature=None if curvature is None else curvature.value,
        n_events=n_events, n_adoptions=n_adoptions, n_refusals=n_refusals,
        n_solver_failures=n_failures, event_stable=event_stable, events=events)


__all__ = ["optimize_orbitals_events", "EventCASSCFResult", "EventRecord",
           "DEFAULT_TAU", "DEFAULT_EVENT_INTERVAL", "DEFAULT_MAX_EVENT_INTERVAL"]
