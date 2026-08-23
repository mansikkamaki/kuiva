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
    boundary_check : extra local roots for the state-average boundary diagnostic at convergence of
        each solve. Default 0 inside an optimization loop — the diagnostic costs one full
        extra sweep per solve; run the final converged orbitals through a solve with it
        on (or a finishing :func:`solve_ttn`) before trusting a state average.
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
                 enforce_kramers: bool = True, symmetry: Optional[object] = None,
                 sector=None, seed: int = 0) -> None:
        self.n_elec = int(n_elec)
        self.max_bond = int(max_bond)
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
        self._templates: Dict[NetworkGraph, TTNOTemplate] = {}
        self._candidate: Optional[Tuple[Hashable, NetworkGraph, TTNState]] = None
        #: The most recent converged solve (energies exclude ``e_core``), for callers
        #: that want the spectrum or the state rather than the RDMs.
        self.last: Optional[SweepResult] = None
        self.n_solves = 0
        self.n_proposals = 0
        self.n_adoptions = 0

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
        gamma, gamma2 = network_rdms(
            template, state, energies=[float(e) for e in result.energies],
            n_elec=self.n_elec, weights=self.requested_weights,
            enforce_kramers=self.enforce_kramers, on_split=self.on_split, ttno=ttno)
        energy = float(np.dot(result.weights, result.energies)) + e_core
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
        if self._state is None:
            self._state = random_state(ttno, self.n_elec, self.max_bond,
                                       n_roots=self.n_roots, rng=self._rng,
                                       charge=None if self._bases is None else self._charge())
        result = solve_ttn(ttno, self._state, max_sweeps=self.max_sweeps,
                           conv_tol=self.conv_tol, trunc_tol=self.trunc_tol,
                           max_bond=self.max_bond, weights=self.requested_weights,
                           n_elec=self.n_elec, boundary_check=self.boundary_check,
                           davidson_tol=self.davidson_tol, on_split=self.on_split,
                           report=False)
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
        """A re-adapted network at these integrals, or ``None`` (module docstring)."""
        if not self.adaptive or self._state is None:
            return None
        h, eri, e_core = self._active(ints)
        terms = hamiltonian_product_terms(h, eri)
        trial = _copy_state(self._state)
        self.n_proposals += 1
        result = solve_adaptive(
            terms, self._graph, self.n_elec, state=trial, max_bond=self.max_bond,
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
        self._candidate = (key, result.graph, result.state)
        moved = len(result.moves)
        return NetworkProposal(energy=energy, gamma=gamma, gamma2=gamma2, key=key,
                               label="reconnected ({} move{})".format(
                                   moved, "" if moved == 1 else "s"))

    def adopt(self, key: Hashable) -> None:
        if self._candidate is None or self._candidate[0] != key:
            raise ValueError("no pending proposal carries key {!r}".format(key))
        _, graph, state = self._candidate
        self._graph = graph
        self._state = state
        self._candidate = None
        self.n_adoptions += 1
        log.debug("adopted network topology %s", self._key_for(graph))

    def space_key(self) -> str:
        return self._key_for(self._graph) if self._graph is not None \
            else "{}:unset".format(self.KEY_PREFIX)

    def _key_for(self, graph: NetworkGraph) -> str:
        key = "{}:e{}:roots{}:D{}:tol{:.1e}:{}".format(
            self.KEY_PREFIX, self.n_elec, self.n_roots, self.max_bond,
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
