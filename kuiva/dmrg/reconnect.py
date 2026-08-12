"""Adaptive topology optimization by local reconnection.

The novel core of the tensor-network layer. A two-site update on tree bond ``(u, v)``
merges the two node tensors into one object whose external legs are ``u``'s branch bonds,
``u``'s physical leg, ``v``'s branch bonds and ``v``'s physical leg. The ordinary update
splits that object back along the incumbent assignment; *every other* two-sided assignment
of the same legs is a different local tree — moving a branch re-hangs its whole subtree on
the other node, exchanging the physical legs relocates the nodes' orbital content (leaf
swap / orbital reordering as the degenerate case of the same move). Computing the split
for candidate assignments and keeping the best one makes the sweep itself optimize the
topology, at the cost of extra SVDs of a tensor the update has already built — the
mechanism of Hikihara, Ueda, Okunishi, Harada & Nishino (automatic structural optimization
of tree tensor networks), applied here to ab initio spinor Hamiltonians through the
compiled TTNO.

Decisions this module realises (the tensor-network design)
-------------------------------------------------------------------------------
* **The move set fixes the node <-> mode-set assignment up to permutation.** Every
  candidate keeps exactly one physical leg per node, so branches move and whole mode sets
  swap between the two nodes, but mode sets are never merged or split. Merging would
  require re-fusing physical legs across the TTNO's per-node kron conventions — pure
  bookkeeping cost with no new reachable physics for single-mode-per-node networks, where
  branch moves + swaps already reach every tree over the given node contents.
* ⚠ **A leaf swap moves a tree position, not a Jordan-Wigner string.** The JW ordering is
  the global mode index (a [FIRM] decision, :mod:`kuiva.dmrg.ttno`), so the
  network's kron basis carries crossing parities between interleaved mode *labels*: a
  fermionic product state over interleaved label sets is genuinely entangled in this
  representation (a 2+2-mode example has Schmidt rank 2 where the fermionic rank is 1),
  and no tree move can remove that entanglement. The "ordering optimization is
  subsumed" holds for spin models only. Consequence: **mode-label choice is a seed-time
  seed-time decision** (the Fiedler/cluster ordering fixes the labels before the
  integrals enter the TTNO); reconnection optimizes the tree at fixed labels. A future
  JW-relabeling move class (permute term mode indices, recompile, re-evaluate) is
  possible but is a different, full-energy-evaluation move — recorded, not planned.
* **Acceptance is gated with a margin and hysteresis**: a candidate must beat the
  incumbent metric by ``max(rel_margin * |incumbent|, abs_floor)``, and an adopted bond is
  not re-examined for ``cooldown_sweeps`` sweeps, so greedy moves cannot cycle. A tie is a
  tie and keeps the incumbent. Both acceptance rules run through
  :func:`kuiva.dmrg.block.svd`, so every compared spectrum is **degenerate-group-complete**
  — a Kramers pair is kept or dropped whole on every side of every candidate, which is
  what makes "smaller truncated weight" a well-posed comparison at the exact degeneracies
  this code targets.
* **Candidates with a vacuous side are excluded**: a side whose dense dimension is 1 (a
  bare dim-1 physical leg) minimizes any entanglement metric trivially by hanging a dead
  leaf on the tree. A genuine rank-1 cut through a product state remains a legal winner —
  that is precisely the structure discovery this module reports.
* **A topology change is a chart change**: the adoption recompiles the TTNO for
  the new graph (cached per graph — the compiler is deterministic), aborts the current
  Euler tour, and restarts from the new center. Convergence therefore means what the adaptive-solver protocol
  says it must: the sweep energy is stationary **and** a full sweep of reconnection
  attempts refused.
* **Environments survive what they may and nothing more.**
  :class:`HashedEnvironmentCache` keys every environment by directed bond **plus** the
  ``NetworkGraph.subtree_hash`` of the content behind it (the extension
  :class:`~kuiva.dmrg.sweep.EnvironmentCache` was designed for). On adoption,
  :meth:`~HashedEnvironmentCache.rebind` keeps an entry only if a directed bond with the
  same subtree content still exists *and* its operator leg matches the recompiled TTNO —
  re-keying entries whose anchor node changed (a moved branch keeps its environment) and
  releasing the rest. Reuse across the recompile is sound because the TTNO compiler is
  deterministic per (terms, inside-mode-set, orientation): an unchanged subtree gets
  bitwise the same W tensors, which a test asserts by comparing an adopted run against a
  from-scratch solve on the final topology.
* **The discovered structure is a reported result**::func:`discovered_structure`
  walks the converged state, records the ensemble Schmidt spectrum of every bond, cuts the
  tree at weak bonds, and reports each resulting site's orbital content, its backbone
  entropy and its effective (group-complete) Schmidt dimension — the number that seeds the
  the local-multiplet spaces of `manifold.py`.

References
----------
* Automatic structural optimization of tree tensor networks (the local reconnection move
  and entanglement-based acceptance): T. Hikihara, H. Ueda, K. Okunishi, K. Harada,
  T. Nishino, Phys. Rev. Research 5, 013031 (2023), doi:10.1103/PhysRevResearch.5.013031
  (arXiv:2209.03196).
* Entanglement measures the acceptance rules optimize: O. Legeza, J. Solyom, Phys. Rev. B
  68, 195116 (2003), doi:10.1103/PhysRevB.68.195116; J. Rissler, R. M. Noack,
  S. R. White, Chem. Phys. 323, 519 (2006), doi:10.1016/j.chemphys.2005.10.018.
* Shared-basis state averaging (the ensemble spectra all comparisons run on): J. J.
  Dorando, J. Hachmann, G. K.-L. Chan, J. Chem. Phys. 127, 084109 (2007),
  doi:10.1063/1.2768360.
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from .block import BlockTensor, svd, tensordot, _degenerate_groups
from .graph import NetworkGraph
from .sweep import (EnvironmentCache, SweepResult, TTNState, _bond_axis, _commit_split,
                    _solve_local, _stack_roots, random_state, solve_ttn)
from .ttno import TTNO, compile_ttno

log = get_logger(__name__)

#: Per-rule absolute floors below which a metric improvement is noise, not a win.
_ABS_FLOORS = {"weight": 1e-12, "entropy": 1e-3}
#: Per-rule relative margins a candidate must clear to win (hysteresis).
_REL_MARGINS = {"weight": 0.2, "entropy": 0.05}


@dataclass(frozen=True)
class ReconnectionPolicy:
    """The reconnection acceptance rule and its hysteresis, as explicit knobs.

    ``rule``: ``"entropy"`` (least bipartite entanglement entropy of the untruncated
    ensemble spectrum — the Hikihara criterion) or ``"weight"`` (least ensemble discarded
    weight at the shared truncation parameters). ⚠ Entropy is the default **by
    measurement**, not preference (measured): whenever the
    bond dimension can represent the state exactly, *every* candidate discards nothing,
    the weight comparison is all ties, and a weight-based default sits inert
    while the entropy rule discovers the known product structure in the same runs. The
    weight rule stays available for capped production runs, where discarded weight is
    the direct error proxy — re-measure there before trusting either as final.
    ``rel_margin``/``abs_floor`` define a *win* (per-rule defaults); ``cooldown_sweeps``
    keeps a just-reconnected bond untouched; ``attempt_interval`` rate-limits attempts
    (attempt every k-th sweep); ``max_degree`` bounds the node degree a move may create
    (update cost grows exponentially with degree); ``max_enumerated_branches`` caps the
    exhaustive ``2^nb`` enumeration, beyond which only single-branch moves and the plain
    swap are tried.
    """

    rule: str = "entropy"
    rel_margin: Optional[float] = None
    abs_floor: Optional[float] = None
    cooldown_sweeps: int = 1
    attempt_interval: int = 1
    max_degree: int = 4
    max_enumerated_branches: int = 5

    def floor(self) -> float:
        if self.abs_floor is not None:
            return float(self.abs_floor)
        try:
            return _ABS_FLOORS[self.rule]
        except KeyError:
            raise ValueError("unknown acceptance rule {!r}; use 'weight' or 'entropy'"
                             .format(self.rule))

    def margin(self) -> float:
        return float(self.rel_margin) if self.rel_margin is not None \
            else _REL_MARGINS[self.rule]


@dataclass(frozen=True)
class Move:
    """One adopted reconnection — what changed and by how much it won."""

    sweep: int
    bond: Tuple[int, int]
    moved_branches: Tuple[int, ...]        #: branch nodes whose anchor changed
    phys_swapped: bool                     #: the two nodes exchanged their mode sets
    rule: str
    metric_before: float
    metric_after: float


@dataclass(eq=False)
class AdaptiveResult(object):
    """Outcome of :func:`solve_adaptive`. Energies exclude ``e_core``."""

    energies: np.ndarray
    weights: np.ndarray
    state: TTNState
    graph: NetworkGraph                    #: the discovered topology
    converged: bool                        #: stationary energy AND a refusing sweep AND final solve converged
    n_sweeps: int                          #: adaptive sweeps (the finishing solve adds its own)
    moves: List[Move]
    history: List[float]
    final: SweepResult                     #: the finishing fixed-topology solve (the state-averaging gate lives there)
    structure: Optional["StructureReport"]


# --- environments across topology changes -------------------------------------------------

class HashedEnvironmentCache(EnvironmentCache):
    """Environment cache keyed by directed bond **and** subtree content hash.

    At fixed topology this behaves exactly like the base class (the hash never changes,
    so keys are effectively directed bonds). Across a reconnection, :meth:`rebind` is the
    single sanctioned survival path — see the module docstring for what may be kept and
    why that is sound.
    """

    def _key(self, u: int, v: int) -> tuple:
        return (u, v, self.state.graph.subtree_hash(u, v))

    def _op_matches(self, env: BlockTensor, a: int, b: int) -> bool:
        tt = self.ttno
        if int(tt.parent[a]) == b:
            sp, sg = tt.bond_space[a], -1
        elif int(tt.parent[b]) == a:
            sp, sg = tt.bond_space[b], 1
        else:
            return False
        return env.spaces[1] == sp and env.signs[1] == sg

    def rebind(self, ttno: TTNO, graph: NetworkGraph) -> None:
        """Adopt a recompiled TTNO and new topology, keeping every still-valid entry."""
        self.ttno = ttno
        old_cache, old_held = self._cache, self._held
        self._cache, self._held = {}, {}
        edges = set(graph.directed_bonds())
        kept = 0
        for (a, b, h), env in old_cache.items():
            target = None
            if (a, b) in edges and graph.subtree_hash(a, b) == h:
                target = (a, b, h)
            else:
                for b2 in graph.neighbors(a):
                    if (a, b2) in edges and graph.subtree_hash(a, b2) == h:
                        target = (a, b2, h)
                        break
            held = old_held.pop((a, b, h), None)
            if target is not None and target not in self._cache \
                    and self._op_matches(env, target[0], target[1]):
                self._cache[target] = env
                if held is not None:
                    self._held[target] = held
                kept += 1
            else:
                res.BUDGET.release(held)
        for held in old_held.values():                  # entries with no surviving env
            res.BUDGET.release(held)
        log.debug("environment cache rebind: kept %d of %d entries", kept,
                  len(old_cache))


# --- candidate enumeration and acceptance -------------------------------------------------

@dataclass(frozen=True)
class _Choice:
    left_labels: Tuple[tuple, ...]
    left_branches: Tuple[int, ...]
    phys_owner: int                        #: node whose modes end up at u
    metric: float
    incumbent_metric: float


def _spectrum_entropy(s_sectors) -> float:
    s = np.concatenate([np.asarray(x, dtype=float) for x in s_sectors if x is not None])
    p = s ** 2
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    p = p / total
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


def _metric(stacked, left_axes, trunc_tol, max_bond, rule):
    """The acceptance metric of one bipartition, or ``None`` if the split is refused
    (e.g. the cap would cut the leading degenerate group — refusal, not rounding)."""
    try:
        if rule == "entropy":
            _, s_sectors, _, _ = svd(stacked, left_axes, tol=0.0, max_bond=None)
            return _spectrum_entropy(s_sectors)
        _, _, _, info = svd(stacked, left_axes, tol=trunc_tol, max_bond=max_bond)
        return float(info.discarded_weight)
    except ValueError:
        return None


def _enumerate_assignments(branches_u, branches_v, u, v, policy):
    """Candidate (left branch set, phys owner) pairs, incumbent excluded."""
    branches = list(branches_u) + list(branches_v)
    nb = len(branches)
    incumbent = (frozenset(branches_u), u)
    if nb <= policy.max_enumerated_branches:
        subsets = [frozenset(c) for r in range(nb + 1)
                   for c in itertools.combinations(branches, r)]
    else:                                              # degree too high: single-leg moves
        base = frozenset(branches_u)
        subsets = [base]
        subsets += [base - {x} for x in branches_u]
        subsets += [base | {y} for y in branches_v]
    seen = set()
    for sub in subsets:
        for owner in (u, v):
            key = (sub, owner)
            if key == incumbent or key in seen:
                continue
            seen.add(key)
            deg_u = len(sub) + 1
            deg_v = nb - len(sub) + 1
            if max(deg_u, deg_v) > policy.max_degree:
                continue
            yield sub, owner


def _choose_split(prob, roots, w_used, trunc_tol, max_bond, policy) -> Optional[_Choice]:
    """Evaluate every candidate bipartition of the merged ensemble; return a winning one.

    All candidates and the incumbent are measured with the same rule on the same stacked
    ensemble tensor — "at fixed everything else". Returns ``None`` when nothing
    wins by the margin (a tie is a tie)."""
    labels = list(prob.labels)
    branch_labels = {lab[2]: lab for lab in labels if lab[0] == "b"}
    phys_labels = {lab[1]: lab for lab in labels if lab[0] == "p"}
    dims = {lab: sp.total_dim for lab, sp in zip(labels, prob.template.spaces)}
    stacked = _stack_roots(roots, w_used)

    def axes_for(left_labels):
        return tuple(sorted(labels.index(l) for l in left_labels))

    incumbent_left = labels[:prob.n_left]
    inc = _metric(stacked, axes_for(incumbent_left), trunc_tol, max_bond, policy.rule)
    if inc is None:
        return None
    best = None
    for sub, owner in _enumerate_assignments(prob.branches_u, prob.branches_v,
                                             prob.u, prob.v, policy):
        left = tuple(branch_labels[x] for x in sorted(sub)) + (phys_labels[owner],)
        d_left = int(np.prod([dims[l] for l in left]))
        d_right = int(np.prod([dims[l] for l in labels if l not in left]))
        if d_left < 2 or d_right < 2:                  # vacuous side (module docstring)
            continue
        m = _metric(stacked, axes_for(left), trunc_tol, max_bond, policy.rule)
        if m is None:
            continue
        if best is None or m < best.metric:
            best = _Choice(left_labels=left, left_branches=tuple(sorted(sub)),
                           phys_owner=owner, metric=m, incumbent_metric=inc)
    if best is None:
        return None
    if best.metric < inc - max(policy.margin() * abs(inc), policy.floor()):
        return best
    return None


def _reorder_branch_legs(state: TTNState, old_graph: NetworkGraph,
                         new_graph: NetworkGraph, moved: Sequence[int],
                         u: int, v: int) -> None:
    """Re-permute the legs of re-anchored branch nodes to the new sorted-neighbor order.

    A node tensor's legs are its bonds in ascending *neighbor id* order (sweep module
    conventions). A moved branch keeps the same physical bond — same space, same content
    — but its anchor's node id changed from one of ``(u, v)`` to the other, which can
    move that leg's position in the sorted order. ⚠ Skipping this leaves a tensor whose
    leg order disagrees with the graph: every later contraction either raises on a space
    mismatch or, worse, matches two same-shaped legs wrongly. Pure relabeling; no
    arithmetic."""
    for x in moved:
        old_nbrs = sorted(old_graph.neighbors(x))
        new_nbrs = sorted(new_graph.neighbors(x))
        a_old = u if u in old_nbrs else v
        a_new = v if a_old == u else u
        perm = [old_nbrs.index(a_old if y == a_new else y) for y in new_nbrs]
        perm.append(len(old_nbrs))                     # the physical leg stays last
        if perm != list(range(len(perm))):
            state.tensors[x] = state.tensors[x].transpose(perm)


def _reconnected_graph(graph: NetworkGraph, u: int, v: int,
                       left_branches: Sequence[int], phys_owner: int) -> NetworkGraph:
    """The tree after re-anchoring branches and (possibly) swapping the mode sets."""
    left = set(left_branches)
    edges = []
    for a, b in graph.edges:
        if {a, b} == {u, v}:
            edges.append((a, b))
        elif a in (u, v) or b in (u, v):
            x = b if a in (u, v) else a
            edges.append((x, u if x in left else v))
        else:
            edges.append((a, b))
    contents = list(graph.contents)
    contents[u] = graph.contents[phys_owner]
    contents[v] = graph.contents[v if phys_owner == u else u]
    return NetworkGraph(graph.n_nodes, edges, contents)


# --- the adaptive driver ------------------------------------------------------------------

def solve_adaptive(terms, graph: NetworkGraph, n_elec: int, *, bases=None,
                   state: Optional[TTNState] = None, n_roots: int = 1,
                   max_bond: Optional[int] = None,
                   policy: Optional[ReconnectionPolicy] = None,
                   max_sweeps: int = 50, conv_tol: float = 1e-9,
                   trunc_tol: float = 0.0, weights: Optional[Sequence[float]] = None,
                   davidson_tol: float = 1e-8, boundary_check: int = 4,
                   on_split: str = "raise", rng: Optional[np.random.Generator] = None,
                   ttno_root: int = 0, report_structure: bool = True,
                   report: bool = True) -> AdaptiveResult:
    """Adaptive-topology state-averaged two-site DMRG (module docstring).

    Takes the Hamiltonian as **product terms** (``hamiltonian_product_terms`` output or
    the model-Hamiltonian seam), because every adopted topology needs its own compiled TTNO;
    compiles are cached per graph. ``state`` may come from the cheap-CI seeding
    (:func:`kuiva.dmrg.guess.expansion_to_ttn`); without one, a random state at
    ``max_bond`` is built (which then must be given).

    The adaptive phase optimizes topology; a finishing :func:`~kuiva.dmrg.sweep.solve_ttn`
    on the discovered topology then applies the full state-averaging discipline (theorem-backed
    weight gate, boundary diagnostic), so ``result.final`` is a validated fixed-topology
    result and ``result.converged`` requires *both* phases: a stationary energy with a
    full sweep of refused reconnection attempts, and the finishing solve's own
    convergence. Adaptive never claims what fixed topology has not confirmed.
    """
    policy = policy if policy is not None else ReconnectionPolicy()
    policy.floor()                                     # validates the rule early
    terms = [t for t in terms if t is not None]
    ttnos: Dict[NetworkGraph, TTNO] = {}

    def compiled(g: NetworkGraph) -> TTNO:
        op = ttnos.get(g)
        if op is None:
            op = compile_ttno(g, terms, bases=bases, root=ttno_root)
            ttnos[g] = op
        return op

    ttno = compiled(graph)
    if state is None:
        if max_bond is None:
            raise ValueError("give either a seeded state or max_bond for a random one")
        state = random_state(ttno, n_elec, max_bond, n_roots=n_roots, rng=rng)
    else:
        if state.graph != graph:
            raise ValueError("the state's topology differs from the given graph")
        n_roots = state.n_roots
    requested = np.full(n_roots, 1.0 / n_roots) if weights is None \
        else np.asarray(weights, dtype=float) / float(np.sum(weights))

    cache = HashedEnvironmentCache(ttno, state)
    table = out.Table(log, [out.col_iter("sweep"), out.col_energy("E_SA [Eh]"),
                            out.col_delta(), out.col_sci("w_disc"),
                            out.col_count("max D", 6), out.col_count("moves", 6),
                            out.col_time()],
                      level=logging.INFO if report else logging.DEBUG)
    table.start("adaptive two-site DMRG ({} roots, {} nodes, rule '{}')".format(
        n_roots, graph.n_nodes, policy.rule))

    cooldown: Dict[Tuple[int, int], int] = {}
    moves: List[Move] = []
    history: List[float] = []
    e_prev = None
    converged_adaptive = False
    with timer("adaptive DMRG sweeps"):
        for sweep in range(1, max_sweeps + 1):
            t0 = time.perf_counter()
            attempts = policy.attempt_interval > 0 \
                and sweep % policy.attempt_interval == 0
            adopted = None
            suppressed = False
            max_disc = 0.0
            max_dim = 0
            energies = w_used = None
            for u, v in state.graph.sweep_schedule(state.center):
                prob, energies, w_used, roots, _ = _solve_local(
                    ttno, state, cache, u, v, requested, davidson_tol)
                choice = None
                edge = (min(u, v), max(u, v))
                if attempts:
                    if cooldown.get(edge, 0) >= sweep:
                        suppressed = True
                    else:
                        choice = _choose_split(prob, roots, w_used, trunc_tol,
                                               max_bond, policy)
                if choice is not None:
                    old_graph = state.graph
                    new_graph = _reconnected_graph(old_graph, u, v,
                                                   choice.left_branches,
                                                   choice.phys_owner)
                    moved = tuple(sorted(set(prob.branches_u).symmetric_difference(
                        choice.left_branches)))
                    ttno = compiled(new_graph)
                    cache.rebind(ttno, new_graph)
                    _reorder_branch_legs(state, old_graph, new_graph, moved, u, v)
                    info = _commit_split(state, cache, prob, roots, w_used, trunc_tol,
                                         max_bond, graph=new_graph,
                                         left_labels=list(choice.left_labels))
                    adopted = Move(sweep=sweep, bond=edge, moved_branches=moved,
                                   phys_swapped=choice.phys_owner != prob.u,
                                   rule=policy.rule, metric_before=choice.incumbent_metric,
                                   metric_after=choice.metric)
                    moves.append(adopted)
                    cooldown[edge] = sweep + policy.cooldown_sweeps
                    log.debug("reconnection adopted on bond %s: moved branches %s, "
                              "phys swap %s, %s %.3e -> %.3e", edge, moved,
                              adopted.phys_swapped, policy.rule,
                              choice.incumbent_metric, choice.metric)
                    max_disc = max(max_disc, info.discarded_weight)
                    max_dim = max(max_dim, info.bond_dim)
                    break                              # chart change: restart the tour
                info = _commit_split(state, cache, prob, roots, w_used, trunc_tol,
                                     max_bond)
                max_disc = max(max_disc, info.discarded_weight)
                max_dim = max(max_dim, info.bond_dim)
            e_sa = float(np.dot(w_used, energies))
            de = None if e_prev is None else e_sa - e_prev
            history.append(e_sa)
            table.row(sweep, e_sa, de, max_disc, max_dim,
                      0 if adopted is None else 1, time.perf_counter() - t0)
            e_prev = e_sa
            if adopted is not None:
                continue
            if de is not None and abs(de) < conv_tol and attempts and not suppressed:
                converged_adaptive = True              # stationary AND a refusing sweep
                break
    table.end("topology {} after {} moves".format(
        "stationary" if converged_adaptive else "NOT stationary", len(moves)))
    if not converged_adaptive:
        log.warning("adaptive DMRG did not reach a stationary topology in %d sweeps "
                    "(%d moves adopted); the finishing solve runs on the last topology",
                    max_sweeps, len(moves))

    cache.release_all()
    final = solve_ttn(ttno, state, max_sweeps=max_sweeps, conv_tol=conv_tol,
                      trunc_tol=trunc_tol, max_bond=max_bond, weights=requested,
                      n_elec=n_elec, boundary_check=boundary_check,
                      davidson_tol=davidson_tol, on_split=on_split, report=report)

    structure = None
    if report_structure and state.graph.n_nodes > 1:
        structure = discovered_structure(state, weights=final.weights)
    return AdaptiveResult(energies=final.energies, weights=final.weights, state=state,
                          graph=state.graph,
                          converged=converged_adaptive and final.converged,
                          n_sweeps=len(history), moves=moves, history=history,
                          final=final, structure=structure)


# --- the discovered-structure report ----------------------------------------------

@dataclass(frozen=True)
class BondReport:
    """Ensemble entanglement across one tree bond."""

    edge: Tuple[int, int]
    entropy_nats: float
    eff_dim: int                           #: group-complete Schmidt count holding 1 - eff_weight
    bond_dim: int                          #: current state bond dimension


@dataclass(frozen=True)
class SiteReport:
    """One discovered site: a subtree whose internal bonds are strong."""

    nodes: Tuple[int, ...]
    orbitals: Tuple[int, ...]
    internal_max_entropy: float            #: 0 for a single-node site


@dataclass(frozen=True)
class StructureReport:
    """The structure-discovery deliverable: sites, backbone bonds, and the inter-site graph."""

    sites: Tuple[SiteReport, ...]
    bonds: Dict[Tuple[int, int], BondReport]
    weak_edges: Tuple[Tuple[int, int], ...]
    site_cut_nats: float


def _move_center(state: TTNState, u: int, v: int) -> None:
    """Move the canonical center across ``(u, v)`` exactly (gauge only, no solve).

    The shared basis is re-derived from the exact ensemble SVD, so the move never
    truncates (the stability floor only removes rounding noise). ⚠ Gauge moves refresh
    no environment cache; callers holding one must not reuse it across this."""
    graph = state.graph
    ax = _bond_axis(graph, u, v)
    n_roots = len(state.centers)
    stacked = _stack_roots(state.centers, np.full(n_roots, 1.0 / n_roots))
    left = tuple(i for i in range(stacked.ndim - 1) if i != ax)
    u_iso, _, _, _ = svd(stacked, left, tol=0.0)

    def left_pos(j):
        return j if j < ax else j - 1

    nbrs = sorted(graph.neighbors(u))
    perm = [len(left) if x == v else left_pos(idx) for idx, x in enumerate(nbrs)]
    perm.append(left_pos(state.centers[0].ndim - 1))
    new_u = u_iso.transpose(perm)

    tv = state.tensors[v]
    pax = _bond_axis(graph, v, u)
    nbrs_v = sorted(graph.neighbors(v))
    perm2 = [0 if y == u else (idx if idx < pax else idx - 1) + 1
             for idx, y in enumerate(nbrs_v)]
    perm2.append(tv.ndim - 1)
    centers = []
    for c in state.centers:
        rho = tensordot(c, u_iso.conj(), axes=(list(left), list(range(len(left)))))
        m = tensordot(rho, tv, axes=([0], [pax]))
        centers.append(m.transpose(perm2))
    state.tensors[u] = new_u
    state.centers = centers
    state.tensors[v] = None
    state.center = v


def _edge_spectrum(state: TTNState, u: int, v: int,
                   weights: np.ndarray) -> np.ndarray:
    """Descending ensemble Schmidt values across edge ``(u, v)``, center at ``u``.

    The root leg belongs on the ``u`` side: the spectrum is that of the weighted ensemble
    density matrix ``sum_r w_r Tr_u |psi_r><psi_r|`` on the far side of the bond."""
    ax = _bond_axis(state.graph, u, v)
    stacked = _stack_roots(state.centers, weights)
    left = tuple(i for i in range(stacked.ndim) if i != ax)
    _, s_sectors, _, _ = svd(stacked, left, tol=0.0)
    s = np.concatenate([np.asarray(x, dtype=float) for x in s_sectors if x is not None])
    return np.sort(s)[::-1]


def _effective_dimension(s: np.ndarray, eff_weight: float) -> int:
    """Smallest group-complete Schmidt count carrying weight ``1 - eff_weight``."""
    p = s ** 2
    total = float(p.sum())
    if total <= 0.0:
        return 0
    cum = np.cumsum(p) / total
    m = int(np.searchsorted(cum, 1.0 - eff_weight) + 1)
    for group in _degenerate_groups(s, 1e-6):
        if m - 1 in group:
            return int(group[-1]) + 1                  # never split a degenerate group
    return min(m, s.size)


def discovered_structure(state: TTNState, *, weights: Optional[Sequence[float]] = None,
                         site_cut_nats: float = 0.1, eff_weight: float = 1e-6,
                         logger=None) -> StructureReport:
    """Read the site decomposition off a converged state and report it.

    Walks the Euler tour with exact gauge moves, recording each bond's weighted ensemble
    Schmidt spectrum; bonds below ``site_cut_nats`` of entanglement entropy are *weak*,
    the connected components of the strong bonds are the **sites**, and the weak edges
    with their effective Schmidt dimensions form the inter-site graph — for a magnetic
    site, that dimension is the local multiplet dimension `manifold.py` truncates to. The
    state
    is re-gauged in place (exactly); no environment cache survives this (see
    :func:`_move_center`)."""
    logger = logger or log
    graph = state.graph
    n_roots = len(state.centers)
    w = np.full(n_roots, 1.0 / n_roots) if weights is None \
        else np.asarray(weights, dtype=float) / float(np.sum(weights))

    bonds: Dict[Tuple[int, int], BondReport] = {}
    dims = state.bond_dimensions()
    for u, v in graph.sweep_schedule(state.center):
        edge = (min(u, v), max(u, v))
        if edge not in bonds:
            s = _edge_spectrum(state, u, v, w)
            bonds[edge] = BondReport(edge=edge, entropy_nats=_entropy(s),
                                     eff_dim=_effective_dimension(s, eff_weight),
                                     bond_dim=dims[edge])
        _move_center(state, u, v)

    weak = tuple(sorted(e for e, b in bonds.items()
                        if b.entropy_nats < site_cut_nats))
    strong = [e for e in graph.edges if e not in set(weak)]
    root_of = list(range(graph.n_nodes))

    def find(x):
        while root_of[x] != x:
            root_of[x] = root_of[root_of[x]]
            x = root_of[x]
        return x

    for a, b in strong:
        root_of[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for u in range(graph.n_nodes):
        groups.setdefault(find(u), []).append(u)
    sites = []
    for members in sorted(groups.values()):
        internal = [bonds[e].entropy_nats for e in graph.edges
                    if e[0] in members and e[1] in members]
        orbitals = tuple(sorted(x for m in members for x in graph.contents[m]))
        sites.append(SiteReport(nodes=tuple(sorted(members)), orbitals=orbitals,
                                internal_max_entropy=max(internal) if internal else 0.0))
    report = StructureReport(sites=tuple(sites), bonds=bonds, weak_edges=weak,
                             site_cut_nats=site_cut_nats)

    out.subsection(logger, "discovered network structure")
    out.entries(logger, [
        ("sites", len(sites), "", "connected components above {:.2f} nats"
         .format(site_cut_nats)),
        ("inter-site bonds", len(weak)),
    ])
    table = out.Table(logger, [out.Column("site", "{:d}", 5),
                               out.Column("orbitals", "{:s}", 30),
                               out.Column("S_int max", "{:.4f}", 10)])
    table.start()
    for i, site in enumerate(sites):
        table.row(i, " ".join(str(x) for x in site.orbitals),
                  site.internal_max_entropy)
    table.end()
    if weak:
        table = out.Table(logger, [out.Column("bond", "{:s}", 10),
                                   out.Column("S [nats]", "{:.5f}", 10),
                                   out.Column("eff dim", "{:d}", 8),
                                   out.Column("D", "{:d}", 6)])
        table.start("inter-site bonds")
        for e in weak:
            b = bonds[e]
            table.row("{}-{}".format(*e), b.entropy_nats, b.eff_dim, b.bond_dim)
        table.end()
    return report


def _entropy(s: np.ndarray) -> float:
    p = s ** 2
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    p = p / total
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


__all__ = ["ReconnectionPolicy", "Move", "AdaptiveResult", "HashedEnvironmentCache",
           "solve_adaptive", "discovered_structure", "StructureReport", "SiteReport",
           "BondReport"]
