"""The tensor-network CI solver behind the adaptive-solver contract.

:class:`DMRGSolver` is the DMRG counterpart of :class:`kuiva.mcscf.casci.FullCISolver` and
:class:`kuiva.mcscf.preopt.CheapCISolver`: the same four-method contract
(``solve`` / ``propose`` / ``adopt`` / ``space_key``), the same
``ci_solver(ints) -> (energy, gamma, gamma2)`` facade via ``__call__``, so it plugs into
:func:`kuiva.mcscf.orbopt.optimize_orbitals` (frozen chart) and
:func:`kuiva.mcscf.events.optimize_orbitals_events` (event-gated topology adaptation)
without either driver changing.

⚠ ``kuiva.dmrg`` never imports ``kuiva.mcscf`` (the dependency rule: ``mcscf`` is
the *consumer* of this layer). The contract is structural — a ``runtime_checkable``
protocol on the mcscf side — so this module defines its own :class:`NetworkProposal`
carrying the same attributes rather than importing ``mcscf.adaptive.Proposal``, and the
integrals object is duck-typed on ``h_active_effective()`` / ``active_eri()`` / ``e_core``
exactly as :func:`kuiva.dmrg.ttno.ttno_from_cas_integrals` established.

What the chart is (and why the bond-dimension *distribution* is not in the key)
-------------------------------------------------------------------------------
``solve`` runs the fixed-topology two-site sweep: the variational manifold is "every TTN
on this tree within ``max_bond``", and that manifold — not the particular per-sector
dimensions the last SVD happened to produce — is what makes ``E(kappa)`` one smooth
surface. A two-site update re-derives the distribution inside the cap at every bond it
touches, deterministically from the integrals; putting the observed distribution into
``space_key`` would declare a chart change at nearly every macro-iteration and reset the
optimizer's curvature memory each time, which is precisely the adaptive-solver machinery misfiring.
The key therefore carries the **topology digest and the manifold parameters**
(``max_bond``, ``trunc_tol``, root count) — the "bond-dimension distribution"
wording is realised as the cap that defines it.

``propose`` runs :func:`kuiva.dmrg.reconnect.solve_adaptive` on a **copy** of the
incumbent state at the current integrals: the incumbent is never touched (the adopt-is-explicit rule:
contract), a proposal that reproduces the incumbent topology returns ``None``, and a
changed topology comes back energy-and-RDM evaluated at the same integrals as a
:class:`NetworkProposal`. A proposal here costs a full adaptive solve — "where a
proposal is expensive, raise ``event_interval``" applies verbatim and is the caller's
knob, not this class's.

Warm starts and determinism: like :class:`FullCISolver`'s Davidson warm start, ``solve``
keeps the converged network as the next solve's starting point. The converged answer at
fixed integrals is history-independent to within ``conv_tol``; the ci_solver contract's
"deterministic, smooth" is met in the same sense and to the same degree as the CI
precedent. A solve that does not converge raises
:class:`~kuiva.util.errors.SolverFailure` — at a trial point the event controller turns
that into a rejected step, which is the adaptive-protocol semantics.

Everything here is orchestration.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util import resources as res
from ..util import threads
from ..util.errors import SolverFailure
from ..util.logging import get_logger
from .graph import NetworkGraph
from .density import network_rdms
from .reconnect import ReconnectionPolicy, solve_adaptive
from .sweep import SweepResult, TTNState, random_state, solve_ttn
from .ttno import TTNO, TTNOTemplate, hamiltonian_product_terms

log = get_logger(__name__)


@dataclass(eq=False)
class NetworkProposal:
    """Duck-typed twin of ``kuiva.mcscf.adaptive.Proposal`` (module docstring)."""

    energy: float
    gamma: np.ndarray
    gamma2: Optional[np.ndarray]
    key: Hashable
    label: str = ""


def _graph_digest(graph: NetworkGraph) -> str:
    """Stable 128-bit identity of (edges, node contents) — checkpoint/log-safe, like
    :func:`kuiva.mcscf.adaptive.array_key`."""
    payload = repr((tuple(sorted(tuple(sorted(e)) for e in graph.edges)),
                    tuple(tuple(sorted(c)) for c in graph.contents))).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _copy_state(state: TTNState) -> TTNState:
    return TTNState(graph=state.graph, center=state.center,
                    tensors=[None if t is None else t.copy() for t in state.tensors],
                    centers=[c.copy() for c in state.centers], charge=state.charge)


class DMRGSolver:
    """State-averaged two-site DMRG as a ``ci_solver`` and an adaptive solver.

    Parameters
    ----------
    n_elec : int — electrons in the active space.
    max_bond : int
        The bond-dimension cap — part of the chart (module docstring), so it is fixed at
        construction, not per call. ⚠ Required: an uncapped tree state allocates
        charge-sector-maximal bond dimensions (measured).
    n_roots, weights : the state average; weights are re-equalized inside degenerate
        blocks at every solve and a count that splits a Kramers pair is refused
        (``on_split``), exactly as in :class:`FullCISolver`.
    graph : optional seed topology (default: a path over the modes in label order —
        the cheap-CI seeding hands a better one when available).
    initial_state : optional seeded :class:`TTNState` (e.g.
        :func:`kuiva.dmrg.guess.expansion_to_ttn`); must live on ``graph``.
    adaptive : bool
        With ``False``, :meth:`propose` always returns ``None`` and the solver is a
        static (frozen-chart) DMRG — the right thing under plain ``optimize_orbitals``.
    policy : :class:`~kuiva.dmrg.reconnect.ReconnectionPolicy` for proposals.
        ⚠ Default here is ``rule="weight"``, **not** the reconnect module's ``entropy``
        default, by measurement (the wall-time experiment): a
        CASSCF solver always runs with a binding cap, and there discarded weight is the
        direct error proxy — the weight-gated run converged 30x closer to the exact-CI
        CASSCF than the entropy-gated one on the same system. Entropy remains the right
        default for uncapped structure *discovery*, which is a different job.
    propose_sweeps : adaptive-sweep budget per proposal (a proposal costs a full solve).
    rdms : extract the state-averaged ``(gamma, Gamma)`` in :meth:`solve` (default). ``False``
        returns ``(energy, None, None)`` for a fixed-orbital use that only needs energies and
        transition densities: the extraction's per-node ``dE/dW`` tensors are dense in the
        local dimension squared with one operator leg per neighbour, and on a fat or
        branching node they, not the sweep, are what the memory plan refuses.
    boundary_check : extra local roots for the state-average boundary diagnostic at convergence of
        each solve. Default 0 inside an optimization loop — the diagnostic costs one full
        extra sweep per solve; run the final converged orbitals through a solve with it
        on (or a finishing :func:`solve_ttn`) before trusting a state average.
    checkpoint : path or :class:`~kuiva.dmrg.checkpoint.NetworkCheckpointPolicy`
        Rolling network-state checkpointing: the state is written at the end of each
        completed sweep of every incumbent solve (never of a trial proposal), one file,
        replaced atomically, under the checkpoint cadence rules. The file records the
        chart's ``space_key``, refreshed per solve so an adopted topology is recorded
        under its own key.
    restart : path
        Warm-start the network from a state written by ``checkpoint=``. ⚠ A warm start
        only: a state that does not fit this solver (topology, root count) warns and
        starts cold — the loss is time, never correctness — while a different electron
        count is a different calculation and is refused. See :mod:`kuiva.dmrg.checkpoint`.
    bond_steps : ascending sequence of caps, last equal to ``max_bond``
        The per-**macro-iteration** cap ladder. The incumbent chart starts at the first
        rung; each later rung is offered through :meth:`propose` — a bond-dimension
        change is a chart change (``D`` is in :meth:`space_key`), so it must be an
        event under the adaptive-solver rules, never a silent drift — and the
        event-gated driver adopts it variationally at fixed integrals. ⚠ Under the
        plain (frozen-chart) driver nothing ever proposes, so the ladder needs
        :func:`~kuiva.mcscf.events.optimize_orbitals_events`; the class API selects it
        automatically. A rung refused for gaining less than the driver's ``tau`` is a
        measurement that the current cap suffices, not a failure.
    bond_schedule, expansion, expansion_sweeps
        The per-**sweep** controls of :func:`~kuiva.dmrg.sweep.solve_ttn`, applied to
        the FIRST (cold) solve only: a warm-started solve re-derives its fixed point in
        a few sweeps, and re-ramping the cap below the state's own bonds — or
        re-perturbing a converged truncation — would destroy exactly the warm start it
        was handed. ``bond_schedule`` must end at the first solve's cap.
    """

    KEY_PREFIX = "dmrg"

    def __init__(self, n_elec: int, *, max_bond: int, n_roots: int = 1,
                 weights: Optional[Sequence[float]] = None,
                 graph: Optional[NetworkGraph] = None,
                 initial_state: Optional[TTNState] = None,
                 adaptive: bool = False,
                 policy: Optional[ReconnectionPolicy] = None,
                 propose_sweeps: int = 12,
                 max_sweeps: int = 30, conv_tol: float = 1e-9,
                 davidson_tol: float = 1e-8, trunc_tol: float = 0.0,
                 boundary_check: int = 0, on_split: str = "raise",
                 enforce_kramers: bool = True, rdms: bool = True,
                 symmetry: Optional[object] = None,
                 sector=None, seed: int = 0,
                 checkpoint=None, restart=None,
                 bond_steps: Optional[Sequence[int]] = None,
                 bond_schedule: Optional[Sequence[int]] = None,
                 expansion: float = 0.0, expansion_sweeps: int = 6) -> None:
        self.n_elec = int(n_elec)
        self.max_bond = int(max_bond)
        #: The per-macro-iteration cap ladder (module docstring). ``self._cap`` is the
        #: incumbent chart's cap — it starts at the first rung and moves only through
        #: :meth:`adopt`, because a mid-run cap change is a chart change and must be an
        #: event, never a silent drift.
        self.bond_steps = None if bond_steps is None else [int(d) for d in bond_steps]
        if self.bond_steps is not None:
            if not self.bond_steps or any(b <= a for a, b in zip(self.bond_steps,
                                                                 self.bond_steps[1:])):
                raise ValueError("bond_steps must be strictly ascending, got {}"
                                 .format(bond_steps))
            if self.bond_steps[-1] != self.max_bond:
                raise ValueError(
                    "bond_steps ends at {} but max_bond is {}; the last rung is the "
                    "manifold the calculation promises, so the two must be one number"
                    .format(self.bond_steps[-1], self.max_bond))
        self._cap = self.max_bond if self.bond_steps is None else self.bond_steps[0]
        self._pending_caps = [] if self.bond_steps is None else list(self.bond_steps[1:])
        #: Intra-solve controls, passed through to :func:`~kuiva.dmrg.sweep.solve_ttn`
        #: on the FIRST (cold) solve only: a warm-started solve re-derives its fixed
        #: point in a few sweeps, and re-ramping the cap below the state's own bond
        #: dimensions — or re-perturbing a converged truncation — would destroy exactly
        #: the warm start it is handed.
        self.bond_schedule = None if bond_schedule is None \
            else [int(d) for d in bond_schedule]
        self.expansion = float(expansion)
        self.expansion_sweeps = int(expansion_sweeps)
        self._first_solve_done = False
        if self.bond_schedule is not None and self.bond_schedule[-1] != self._cap:
            raise ValueError(
                "bond_schedule (the per-sweep ramp of the first solve) ends at {} but "
                "the first solve's cap is {}; the two must agree — the ramp is an "
                "iteration strategy inside one manifold, never a way to exceed it"
                .format(self.bond_schedule[-1], self._cap))
        self.n_roots = int(n_roots)
        self.requested_weights = None if weights is None \
            else np.asarray(weights, dtype=float)
        self.adaptive = bool(adaptive)
        self.policy = policy if policy is not None \
            else ReconnectionPolicy(rule="weight")
        self.propose_sweeps = int(propose_sweeps)
        self.max_sweeps = int(max_sweeps)
        self.conv_tol = float(conv_tol)
        self.davidson_tol = float(davidson_tol)
        self.trunc_tol = float(trunc_tol)
        self.boundary_check = int(boundary_check)
        self.on_split = on_split
        self.enforce_kramers = bool(enforce_kramers)
        #: Whether :meth:`solve` extracts the state-averaged ``(gamma, Gamma)`` — the
        #: ``ci_solver`` contract an orbital optimizer needs. A fixed-orbital use (a network
        #: CASCI whose products are energies and the transition densities of
        #: :meth:`transition_densities`) passes ``False`` and gets ``(energy, None, None)``:
        #: the extraction forms every node's ``dE/dW``, dense in the node's local dimension
        #: squared and with one operator leg per neighbour, which on a fat or branching
        #: node is refused by the memory plan where the sweep itself fits comfortably.
        self.rdms = bool(rdms)
        #: Irrep labels of the active spinors (:class:`kuiva.symm.OrbitalLabels`), widening
        #: the network's conserved quantum number from ``(N,)`` to ``(N, irrep)``. ⚠ With
        #: labels on, ``N`` alone is no longer a sector, so ``sector`` names the irrep the
        #: solve targets — defaulted to the label of the aufbau determinant, which is what
        #: an unlabelled run implicitly solved for and keeps the default honest.
        self.symmetry = symmetry
        self.sector = sector
        self._bases = None
        if symmetry is not None:
            from ..symm.sectors import mode_bases
            self._bases = mode_bases(symmetry, symmetry.group)
            if len(symmetry) < self.n_elec:
                raise ValueError("{} labelled spinors cannot hold {} electrons"
                                 .format(len(symmetry), self.n_elec))
            out.entry(log, "network quantum number", "(N, irrep)", "",
                      "{}; sector {}".format(
                          symmetry.group.name,
                          symmetry.group.irrep_name(self._target_label())))
        self._rng = np.random.default_rng(seed)

        self._graph = graph
        self._state = initial_state
        if initial_state is not None:
            if graph is None:
                self._graph = initial_state.graph
            elif initial_state.graph != graph:
                raise ValueError("initial_state lives on a different topology than the "
                                 "given graph")
            self.n_roots = initial_state.n_roots
        #: Rolling network-state checkpointing (a path, or a ready
        #: :class:`~kuiva.dmrg.checkpoint.NetworkCheckpointPolicy`); threaded into every
        #: incumbent :meth:`solve`, never into a trial proposal.
        self.checkpoint = None
        if checkpoint is not None:
            from .checkpoint import NetworkCheckpointPolicy
            self.checkpoint = (checkpoint if isinstance(checkpoint, NetworkCheckpointPolicy)
                               else NetworkCheckpointPolicy(checkpoint))
        if restart is not None:
            if initial_state is not None:
                raise ValueError("restart= and initial_state= both supply the starting "
                                 "network; give one or the other")
            if self.bond_steps is not None or self.bond_schedule is not None:
                raise ValueError(
                    "restart= does not combine with bond_steps= or bond_schedule=: a "
                    "restart continues at the cap the interrupted run had reached, and "
                    "re-ramping from below it would truncate the restored state — "
                    "destroying the warm start the file exists to provide")
            self._restart_state(restart)
        self._templates: Dict[NetworkGraph, TTNOTemplate] = {}
        self._candidate: Optional[Tuple[Hashable, NetworkGraph, TTNState]] = None
        #: The most recent converged solve (energies exclude ``e_core``), for callers
        #: that want the spectrum or the state rather than the RDMs.
        self.last: Optional[SweepResult] = None
        self.n_solves = 0
        self.n_proposals = 0
        self.n_adoptions = 0

    def _restart_state(self, path) -> None:
        """Adopt a checkpointed network state as the warm start (module rules in
        :mod:`kuiva.dmrg.checkpoint`).

        ⚠ A read failure **propagates** — the restart was explicitly requested — but a
        state that merely spoils the warm start (a different topology or root count than
        this solver was built for) **warns and starts cold**: the sweep re-derives its
        fixed point from the integrals, so the loss is time, never correctness. A
        different electron count is a different calculation and is refused.
        """
        from .checkpoint import read_network_state
        state, meta = read_network_state(path)
        if int(meta["n_elec"]) != self.n_elec:
            raise ValueError(
                "the network state at {} holds {} active electrons and this solver is "
                "built for {}; a restart continues the calculation that was interrupted"
                .format(path, meta["n_elec"], self.n_elec))
        if self._graph is not None and state.graph != self._graph:
            log.warning("the network state at %s lives on a different topology than the "
                        "given graph; the warm start is discarded and the solve starts "
                        "cold on the given topology (time, not correctness)", path)
            return
        if state.n_roots != self.n_roots:
            log.warning("the network state at %s carries %d roots and this solver "
                        "averages %d; the warm start is discarded and the solve starts "
                        "cold (time, not correctness)", path, state.n_roots, self.n_roots)
            return
        self._graph = state.graph
        self._state = state
        log.debug("network state restored from %s (sweep %d, converged=%s)", path,
                  meta["sweep"], meta["converged"])

    def _target_label(self):
        """The irrep the solve targets: the request, or the aufbau determinant's own label.

        ⚠ The default is a **choice**, not a neutral one: the totally symmetric sector would
        be the obvious guess and is routinely the wrong one for an odd electron count, where
        every determinant carries a fermion label. Taking it from the lowest ``N`` spinors
        reproduces what an unlabelled run solved for.
        """
        group = self.symmetry.group
        if self.sector is not None:
            return group.label_of(self.sector)
        total = group.identity()
        for row in self.symmetry.labels[:self.n_elec]:
            total = group.compose(total, row)
        return total

    def _charge(self):
        from ..symm.sectors import sector_charge
        return sector_charge(self._target_label(), self.symmetry.group, self.n_elec)

    @property
    def graph(self) -> Optional[NetworkGraph]:
        """The incumbent topology (adoptions included); ``None`` before the first solve
        when the graph was defaulted."""
        return self._graph

    # -- internals ---------------------------------------------------------------------
    def _active(self, ints) -> Tuple[np.ndarray, np.ndarray, float]:
        h = np.ascontiguousarray(ints.h_active_effective())
        eri = np.asarray(ints.active_eri())
        return h, eri, float(getattr(ints, "e_core", 0.0))

    def _ensure_chart(self, n: int) -> None:
        if self._graph is None:
            self._graph = NetworkGraph.path(n)
        modes = sorted(m for c in self._graph.contents for m in c)
        if modes != list(range(n)):
            raise ValueError("the topology carries modes {} but the integrals have {} "
                             "spinors".format(modes, n))

    def _template(self, graph: NetworkGraph) -> TTNOTemplate:
        tpl = self._templates.get(graph)
        if tpl is None:
            tpl = TTNOTemplate(graph, bases=self._bases)
            self._templates[graph] = tpl
        return tpl

    def _weights_for(self) -> Optional[np.ndarray]:
        return self.requested_weights

    def _evaluate(self, template: TTNOTemplate, ttno: TTNO, state: TTNState,
                  result: SweepResult, e_core: float):
        energy = float(np.dot(result.weights, result.energies)) + e_core
        if not self.rdms:
            return energy, None, None
        gamma, gamma2 = network_rdms(
            template, state, energies=[float(e) for e in result.energies],
            n_elec=self.n_elec, weights=self.requested_weights,
            enforce_kramers=self.enforce_kramers, on_split=self.on_split, ttno=ttno)
        return energy, gamma, gamma2

    # -- the AdaptiveCISolver contract ---------------------------------------------------
    @threads.kernel_stage
    def solve(self, ints) -> Tuple[float, np.ndarray, np.ndarray]:
        """``(energy, gamma, Gamma)`` on the incumbent chart — the ci_solver contract.

        The whole solve is a kernel region (threaded BLAS buys this layer nothing, measured), which is what puts
        the *RDM* contractions of :meth:`_evaluate` under the same policy as the sweeps
        inside :func:`solve_ttn` — they are the same block-tensor kernels, and this method
        is the outermost point at which the network layer owns the process.
        """
        h, eri, e_core = self._active(ints)
        n = h.shape[0]
        self._ensure_chart(n)
        template = self._template(self._graph)
        ttno = template.fill(h, eri)
        first = not self._first_solve_done
        if self._state is None:
            self._state = random_state(ttno, self.n_elec, self._cap,
                                       n_roots=self.n_roots, rng=self._rng,
                                       charge=None if self._bases is None else self._charge())
        if self.checkpoint is not None:
            # Refreshed per solve, not set once: an adopted topology or cap moves the
            # key, and the file must say which chart the state it holds belongs to.
            self.checkpoint.space_key = self.space_key()
        # ⚠ The memory plan is printed on the FIRST solve only — sixty identical tables in
        # an output file are noise, not output. What runs on every solve is the exact
        # per-bond requirement inside the sweep, which is what actually refuses; the plan
        # is the statement a user gets to read before the sweep starts.
        result = solve_ttn(ttno, self._state, max_sweeps=self.max_sweeps,
                           conv_tol=self.conv_tol, trunc_tol=self.trunc_tol,
                           max_bond=self._cap, weights=self.requested_weights,
                           n_elec=self.n_elec, boundary_check=self.boundary_check,
                           davidson_tol=self.davidson_tol, on_split=self.on_split,
                           checkpoint=self.checkpoint,
                           bond_schedule=self.bond_schedule if first else None,
                           expansion=self.expansion if first else 0.0,
                           expansion_sweeps=self.expansion_sweeps,
                           memory_plan=first, plan_rdms=self.rdms, report=False)
        self._first_solve_done = True
        self.n_solves += 1
        log.debug("DMRG solve %d: E_SA = %.12f Eh (+e_core %.6f) in %d sweeps, max D %d",
                  self.n_solves, float(np.dot(result.weights, result.energies)), e_core,
                  result.n_sweeps, result.max_bond_dim)
        if not result.converged:
            raise SolverFailure(
                "two-site DMRG did not converge in {} sweeps (last dE {:.1e} Eh) at "
                "these integrals".format(self.max_sweeps,
                                         abs(result.history[-1] - result.history[-2])
                                         if len(result.history) > 1 else float("nan")))
        self.last = result
        return self._evaluate(template, ttno, self._state, result, e_core)

    __call__ = solve

    def propose(self, ints) -> Optional[NetworkProposal]:
        """The next chart at these integrals, or ``None`` (module docstring).

        Two kinds of chart change come through this one seam, and a pending **bond
        step** takes precedence over a topology re-adaptation: the cap ladder is a
        stated schedule, the reconnection an opportunistic search, and the search is
        better run on the manifold the schedule was heading for.
        """
        if self._state is None:
            return None
        if self._pending_caps:
            return self._propose_bond_step(ints)
        if not self.adaptive:
            return None
        h, eri, e_core = self._active(ints)
        terms = hamiltonian_product_terms(h, eri)
        trial = _copy_state(self._state)
        self.n_proposals += 1
        result = solve_adaptive(
            terms, self._graph, self.n_elec, state=trial, max_bond=self._cap,
            policy=self.policy, max_sweeps=self.propose_sweeps, conv_tol=self.conv_tol,
            trunc_tol=self.trunc_tol, weights=self.requested_weights,
            davidson_tol=self.davidson_tol, boundary_check=0, on_split=self.on_split,
            report_structure=False, report=False)
        if result.graph == self._graph:
            return None
        if not result.final.converged:
            raise SolverFailure("the reconnected candidate's finishing solve did not "
                                "converge in {} sweeps".format(self.propose_sweeps))
        template = self._template(result.graph)
        ttno = template.fill(h, eri)
        energy, gamma, gamma2 = self._evaluate(template, ttno, result.state,
                                               result.final, e_core)
        key = self._key_for(result.graph)
        self._candidate = (key, result.graph, result.state, None)
        moved = len(result.moves)
        return NetworkProposal(energy=energy, gamma=gamma, gamma2=gamma2, key=key,
                               label="reconnected ({} move{})".format(
                                   moved, "" if moved == 1 else "s"))

    def _propose_bond_step(self, ints) -> NetworkProposal:
        """The next rung of the cap ladder, solved on a copy at these integrals.

        A mid-run bond-dimension change is a chart change (``D`` is in ``space_key``),
        so it goes through the propose/adopt seam like any other: the candidate is
        evaluated at fixed integrals and adopted only variationally by the event-gated
        driver. ⚠ A rung whose gain falls below the driver's ``tau`` is *refused*, and
        that refusal is a measurement — the current cap already carries the state to the
        driver's own resolution — not a failure of the schedule.
        """
        h, eri, e_core = self._active(ints)
        next_cap = self._pending_caps[0]
        template = self._template(self._graph)
        ttno = template.fill(h, eri)
        trial = _copy_state(self._state)
        self.n_proposals += 1
        result = solve_ttn(ttno, trial, max_sweeps=self.max_sweeps,
                           conv_tol=self.conv_tol, trunc_tol=self.trunc_tol,
                           max_bond=next_cap, weights=self.requested_weights,
                           n_elec=self.n_elec, boundary_check=0,
                           davidson_tol=self.davidson_tol, on_split=self.on_split,
                           report=False)
        if not result.converged:
            raise SolverFailure("the bond-step candidate (D {} -> {}) did not converge "
                                "in {} sweeps".format(self._cap, next_cap,
                                                      self.max_sweeps))
        energy, gamma, gamma2 = self._evaluate(template, ttno, trial, result, e_core)
        key = self._key_for(self._graph, cap=next_cap)
        self._candidate = (key, self._graph, trial, next_cap)
        return NetworkProposal(energy=energy, gamma=gamma, gamma2=gamma2, key=key,
                               label="bond step D {} -> {}".format(self._cap, next_cap))

    def adopt(self, key: Hashable) -> None:
        if self._candidate is None or self._candidate[0] != key:
            raise ValueError("no pending proposal carries key {!r}".format(key))
        _, graph, state, cap = self._candidate
        self._graph = graph
        self._state = state
        if cap is not None:
            self._cap = int(cap)
            self._pending_caps.pop(0)
        self._candidate = None
        self.n_adoptions += 1
        log.debug("adopted network chart %s", self._key_for(graph))

    # -- the analysis contract (props duck-types these; it never imports this layer) -------

    def _analysis_state(self) -> Tuple[TTNOTemplate, TTNState]:
        if self._state is None or self._graph is None:
            raise RuntimeError("solve first: the moments are network contractions over "
                               "the converged state")
        return self._template(self._graph), self._state

    @threads.kernel_stage
    def one_body_moments(self, ops: np.ndarray, vectors=None):
        """``<I|A_k|I>``, ``<I|A_k A_k|I>`` and the per-root 1-RDMs — the
        :meth:`kuiva.mcscf.casci.FullCISolver.one_body_moments` contract, network side.

        This is what lets :func:`kuiva.props.spin.spin_squared_states` — and through it
        ``<S^2>`` and the term assignment — run on a tensor-network reference: ``props``
        duck-types the solver on this method and never imports this layer.

        The square goes through each root's **own** 1- and 2-RDM
        (:func:`kuiva.dmrg.density.state_rdms` — the production environments route with
        one-hot weights): for a Hermitian one-body ``A``,

        ::

            <A A> = sum_ps (A^2)_ps gamma_ps  -  sum_pqrs A_pq A_rs Gamma_{p s r q}

        (``a_q a+_r = delta_qr - a+_r a_q`` and the ``Gamma_pqrs = <a+_p a+_r a_s a_q>``
        convention). The CI route computes the same number as ``||A|I>||^2`` through the
        excitation map; the two are asserted equal in the tests, which is what validates
        this contraction against an implementation it shares nothing with.

        ⚠ The out-of-active-space families (``B``, ``C``, ``D`` of
        :mod:`kuiva.props.spin`) are a property of the *orbitals*, not of the CI method:
        they are built by the caller from ``rdm1`` exactly as on the CI route, and
        nothing here re-derives them.

        ``vectors`` must be ``None``: a tensor-network solver holds no CI vectors and the
        moments are evaluated on the stored roots.
        """
        if vectors is not None:
            raise ValueError("a tensor-network solver holds no CI vectors; the moments "
                             "are evaluated on the stored roots (pass vectors=None)")
        template, state = self._analysis_state()
        a = np.ascontiguousarray(ops, dtype=np.complex128)
        if a.ndim == 2:
            a = a[None]
        n = sum(len(c) for c in state.graph.contents)
        if a.ndim != 3 or a.shape[1:] != (n, n):
            raise ValueError("the operators must be (n_ops, {0}, {0}); got {1}"
                             .format(n, np.shape(ops)))
        worst = float(np.max(np.abs(a - a.conj().transpose(0, 2, 1)))) if a.size else 0.0
        if worst > 1e-10:
            raise ValueError(
                "one_body_moments evaluates <A^2> through <a+ a a+ a> expectation values "
                "reduced with A Hermitian; the operators are non-Hermitian by "
                "{:.3e}".format(worst))
        from .density import state_rdms
        pairs = state_rdms(template, state)
        n_states, n_ops = len(pairs), a.shape[0]
        expect = np.empty((n_ops, n_states))
        square = np.empty((n_ops, n_states))
        rdm1 = np.stack([gamma for gamma, _ in pairs])
        for r, (gamma, gamma2) in enumerate(pairs):
            for k in range(n_ops):
                ak = a[k]
                expect[k, r] = float(np.real(np.sum(ak * gamma)))
                square[k, r] = float(np.real(
                    np.sum((ak @ ak) * gamma)
                    - np.einsum("pq,rs,psrq->", ak, ak, gamma2, optimize=True)))
        return expect, square, rdm1

    @threads.kernel_stage
    def transition_densities(self, vectors=None) -> np.ndarray:
        """``gamma^{IJ}_pq = <I|E_pq|J>`` over the stored roots, by network contraction.

        Same convention and shape as
        :meth:`kuiva.mcscf.casci.FullCISolver.transition_densities`, so the property
        layer's moment-matrix assembly consumes it unchanged. ⚠ Phases are per root and
        arbitrary (they are on the CI route too); consume only through phase-invariant
        reductions. See :func:`kuiva.dmrg.density.transition_rdm1s`.
        """
        if vectors is not None:
            raise ValueError("a tensor-network solver holds no CI vectors; the "
                             "transition densities are contracted over the stored roots "
                             "(pass vectors=None)")
        template, state = self._analysis_state()
        from .density import transition_rdm1s
        return transition_rdm1s(template.ttno, state)

    def space_key(self) -> str:
        return self._key_for(self._graph) if self._graph is not None \
            else "{}:unset".format(self.KEY_PREFIX)

    def _key_for(self, graph: NetworkGraph, cap: Optional[int] = None) -> str:
        key = "{}:e{}:roots{}:D{}:tol{:.1e}:{}".format(
            self.KEY_PREFIX, self.n_elec, self.n_roots,
            self._cap if cap is None else int(cap),
            self.trunc_tol, _graph_digest(graph))
        if self._bases is None:
            return key
        # Same rule as the CI solver's symmetry mode: the key moves only for a solver that
        # actually carries labels, so every state and checkpoint written without them still
        # matches its own and keeps its warm start. A labelled network is a *different*
        # surface -- its sectors are finer and its target is one of them.
        return "{}:{}[{}]".format(key, self.symmetry.group.name,
                                  self.symmetry.group.irrep_name(self._target_label()))

    def reset_state(self) -> None:
        """Forget the warm-start network (a fresh problem, or a restart from disk)."""
        self._state = None

    def __repr__(self) -> str:
        return ("DMRGSolver(n_elec={}, roots={}, max_bond={}, adaptive={}, {} solves, "
                "{} proposals, {} adoptions)").format(
            self.n_elec, self.n_roots, self.max_bond, self.adaptive, self.n_solves,
            self.n_proposals, self.n_adoptions)


__all__ = ["DMRGSolver", "NetworkProposal"]
