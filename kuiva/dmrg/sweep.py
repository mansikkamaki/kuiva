"""Two-site DMRG sweep on an arbitrary tree.

The fixed-topology solver: a state-averaged two-site sweep over the Euler tour of
:meth:`~kuiva.dmrg.graph.NetworkGraph.sweep_schedule`, with the local eigenproblem solved
by :func:`kuiva.ci.davidson.davidson` (reused, never duplicated), environments cached
per directed bond, and every truncation running through :func:`kuiva.dmrg.block.svd`'s
degenerate-group-complete rule. Path topology is nothing special: the first *validated
runs* are paths, but the code is the tree code.

Conventions (fixed here; the seeding and the reconnection layers build on them)
-------------------------------------------------------------------------------
* **A node tensor's legs are its bonds in ascending neighbor order, physical leg last** —
  always, with a dim-1 physical space on a mode-less branching node (the TTNO fixes the
  physical spaces; the state uses the same objects).
* **Signs: the away-from-center side of every edge carries -1, the center side +1**; the
  physical leg +1. Isometries carry charge 0; the center tensors carry the state's total
  quantum number. QR and SVD produce exactly these signs, so a canonicalization move never
  re-signs anything.
* **The state is a shared-basis state average** (Dorando, Hachmann & Chan 2007): one tree
  of isometries, ``n_roots`` center tensors. The ensemble truncation stacks the roots on an
  auxiliary charge-0 leg weighted by ``sqrt(w_r)``, so one :func:`svd` call performs the
  weighted density-matrix truncation with the degenerate-group discipline intact — a degenerate
  Schmidt pair (Kramers!) is kept or dropped whole across the whole ensemble.
* **State-averaging discipline, split between sweeping and finishing.** During sweeps the averaging
  weights are equalized within *observed* degenerate blocks of the local spectrum — and
  nothing more, deliberately: mid-sweep the environments are unconverged truncations that
  do not span time-reversal-closed spaces, so the local spectrum of a Kramers-symmetric
  Hamiltonian is *not* yet doubled and the structural odd-electron argument does not
  apply to it (a random start would be refused at its very first update). The full
  state-averaging gate — :func:`kuiva.rdm.rdm.state_average_weights` with its theorem-backed
  refusal of a count that splits a Kramers pair — is applied to the **converged**
  spectrum, which is where the RDM-consuming weights come from (imposed where the
  RDMs are built). The full CI's boundary-gap diagnostic has its sweep analogue: one extra
  local solve at convergence reports the gap above the averaged set, warning below
  ``BOUNDARY_GAP_WARN_CM`` (the same 50 cm^-1, restated because ``kuiva.dmrg`` does not
  import ``kuiva.mcscf``). ⚠ It is a *local two-site* diagnostic — necessary, cheaper and
  weaker than the full-CI ``state_average_boundary``; it cannot see a state that
  the current bond dimension cannot represent.
* **Environments** ``env(u -> v)`` (subtree at ``u``, contracted against the TTNO and the
  bra) have legs ``[bra, op, ket]`` and are refreshed **right after the center crosses**
  ``u -> v`` — the Euler tour's DFS structure guarantees a subtree is finished before the
  tour leaves it, so this single refresh point keeps every cached environment current with
  no invalidation logic. ⚠ That argument is exactly what adaptive reconnection breaks: a
  topology change must key the cache by subtree content hash
  (``NetworkGraph.subtree_hash``) — do not reuse this cache unmodified.
* ⚠ **Energies exclude ``e_core``**, like the TTNO itself (the CI-layer convention); the caller
  adds it when reporting molecular energies.

Dynamic bond dimension: :func:`kuiva.dmrg.block.svd`'s value threshold (``trunc_tol``)
plus the ``max_bond`` cap give DBSS-like behaviour (Legeza, Roeder & Hess 2003). The
default is ``trunc_tol = 0`` — exact within the cap, noise removed by the stability floor;
whether a nonzero threshold should be the default is an open measurement question.

Everything here is orchestration. The arithmetic lives in
:func:`kuiva.dmrg.block.tensordot` calls; the named port candidate is a batched block-GEMM
driver below that interface, decided by a CPU-seconds profile and nothing else.

References
----------
* Two-site DMRG: S. R. White, Phys. Rev. Lett. 69, 2863 (1992),
  doi:10.1103/PhysRevLett.69.2863; U. Schollwoeck, Ann. Phys. 326, 96 (2011),
  doi:10.1016/j.aop.2010.09.012.
* Tree sweeps: N. Nakatani, G. K.-L. Chan, J. Chem. Phys. 138, 134113 (2013),
  doi:10.1063/1.4798639; K. Gunst, F. Verstraete, S. Wouters, O. Legeza, D. Van Neck,
  J. Chem. Theory Comput. 14, 2026 (2018), doi:10.1021/acs.jctc.8b00098.
* State-averaged DMRG in a shared basis: J. J. Dorando, J. Hachmann, G. K.-L. Chan,
  J. Chem. Phys. 127, 084109 (2007), doi:10.1063/1.2768360.
* Dynamic block selection: O. Legeza, J. Roeder, B. A. Hess, Phys. Rev. B 67, 125114
  (2003), doi:10.1103/PhysRevB.67.125114.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ci.davidson import davidson, davidson_workspace_gb, subspace_cap
from ..props.multiplet import HARTREE_TO_CM
from ..rdm.rdm import DEFAULT_DEGENERACY_TOL, degenerate_blocks, state_average_weights
from ..util import output as out
from ..util import resources as res
from ..util import threads
from ..util.logging import get_logger
from ..util.timing import timer
from .block import (BlockShape, BlockTensor, QuantumNumber, Space, TruncationInfo,
                    block_tensor_gb, fuse, qr, svd, tensordot)
from .graph import NetworkGraph
from .sparse import SparseW, dot_sparse
from .ttno import TTNO

log = get_logger(__name__)

#: The boundary-gap warning threshold [cm^-1], restated here (module docstring).
BOUNDARY_GAP_WARN_CM = 50.0


# --- labelled-tensor helpers --------------------------------------------------------------

class _Lab(object):
    """A BlockTensor with named legs, so multi-step contractions stay readable.

    Pure bookkeeping over :func:`tensordot`/``transpose`` — and the reason the environment
    and effective-Hamiltonian code below contains no bare axis integers to get wrong.
    """

    __slots__ = ("t", "labels")

    def __init__(self, t: BlockTensor, labels: Sequence[tuple]):
        if len(labels) != t.ndim:
            raise ValueError("{} labels for {} legs".format(len(labels), t.ndim))
        self.t = t
        self.labels = list(labels)

    def dot(self, other: "_Lab", pairs: Sequence[Tuple[tuple, tuple]]) -> "_Lab":
        """Contract on named legs. A sparse operand on the right takes the sparse
        contraction (:func:`kuiva.dmrg.sparse.dot_sparse`) — this is the *only* dispatch
        point between the two, which is why the TTNO could become sparse without any
        caller learning about it. A sparse operand on the **left** is an accumulator, not
        a hot path (only :mod:`kuiva.dmrg.manifold`'s site contraction does it), and is
        densified once.

        ⚠ A third operand kind, and it is what lets a chain be *sized* before it is run:
        with :class:`~kuiva.dmrg.block.BlockShape` operands the same chain propagates
        structure alone and allocates nothing, so every contraction here has exactly one
        description — the one below — and a sizing function cannot drift from the code it
        sizes. The extra ``isinstance`` costs nothing beside the GEMMs this dispatches to.
        """
        ia = [self.labels.index(a) for a, _ in pairs]
        ib = [other.labels.index(b) for _, b in pairs]
        if isinstance(self.t, BlockShape):
            return _Lab(self.t.dot(other.t, (ia, ib)),
                        [l for i, l in enumerate(self.labels) if i not in ia]
                        + [l for i, l in enumerate(other.labels) if i not in ib])
        left = self.t.to_block_tensor() if isinstance(self.t, SparseW) else self.t
        if isinstance(other.t, SparseW):
            t = dot_sparse(left, other.t, (ia, ib))
        else:
            t = tensordot(left, other.t, (ia, ib))
        labels = [l for i, l in enumerate(self.labels) if i not in ia] \
            + [l for i, l in enumerate(other.labels) if i not in ib]
        return _Lab(t, labels)

    def to(self, order: Sequence[tuple]) -> BlockTensor:
        t = self.t.to_block_tensor() if isinstance(self.t, SparseW) else self.t
        return t.transpose([self.labels.index(l) for l in order])


class _DenseLab(object):
    """The same idea for plain ndarrays (used only by the diagonal assembly)."""

    __slots__ = ("a", "labels")

    def __init__(self, a: np.ndarray, labels: Sequence[tuple]):
        self.a = a
        self.labels = list(labels)

    def dot(self, other: "_DenseLab", pairs) -> "_DenseLab":
        ia = [self.labels.index(x) for x, _ in pairs]
        ib = [other.labels.index(y) for _, y in pairs]
        a = np.tensordot(self.a, other.a, axes=(ia, ib))
        labels = [l for i, l in enumerate(self.labels) if i not in ia] \
            + [l for i, l in enumerate(other.labels) if i not in ib]
        return _DenseLab(a, labels)

    def to(self, order) -> np.ndarray:
        return self.a.transpose([self.labels.index(l) for l in order])


# --- the state ----------------------------------------------------------------------------

def _bond_axis(graph: NetworkGraph, u: int, v: int) -> int:
    return sorted(graph.neighbors(u)).index(v)


@dataclass(eq=False)
class TTNState(object):
    """A canonical tree tensor network state with a shared-basis root set.

    ``tensors[u]`` are the isometries (``None`` at the center); ``centers`` holds one
    tensor per root, all at ``center`` with identical block structure. Conventions in the
    module docstring.
    """

    graph: NetworkGraph
    center: int
    tensors: List[Optional[BlockTensor]]
    centers: List[BlockTensor]
    charge: QuantumNumber

    @property
    def n_roots(self) -> int:
        return len(self.centers)

    @property
    def nbytes(self) -> int:
        return int(sum(t.nbytes for t in self.tensors if t is not None)
                   + sum(c.nbytes for c in self.centers))

    def bond_space(self, u: int, v: int) -> Space:
        holder, other = (u, v) if u != self.center else (v, u)
        return self.tensors[holder].spaces[_bond_axis(self.graph, holder, other)]

    def bond_dimensions(self) -> Dict[Tuple[int, int], int]:
        return {(u, v): self.bond_space(u, v).total_dim for u, v in self.graph.edges}


def state_gb(state: TTNState) -> float:
    """Exact current size [GB] of the state (pinned against ``nbytes`` in a test)."""
    return state.nbytes / 1024.0 ** 3


def state_to_dense(state: TTNState, ttno: TTNO, root: int = 0,
                   max_dim: int = 4096) -> np.ndarray:
    """One root of the state as a dense vector in the **global mode-ascending kron basis**.

    The Tier-0 oracle for the state, mirroring :meth:`TTNO.to_dense` (same index
    convention: plain C-order kron over all modes ascending, first mode slowest — the
    basis in which a determinant of ``ci/strings.py`` is the unit vector at the
    bit-reversed mask index). Validation only, never a compute path; refuses above
    ``max_dim``.
    """
    graph = state.graph
    total = 1
    for dims in ttno.mode_dims:
        for d in dims:
            total *= d
    if total > max_dim:
        raise ValueError("dense state would have dimension {}; raise max_dim if you "
                         "really mean it".format(total))

    def msg(u, parent):
        t = state.centers[root] if u == state.center else state.tensors[u]
        a = t.to_dense()
        nbrs = sorted(graph.neighbors(u))
        inv = np.argsort(ttno.phys_perm[u])            # sector order -> kron order
        a = np.take(a, inv, axis=len(nbrs))
        a = a.reshape(a.shape[:len(nbrs)] + tuple(ttno.mode_dims[u]))
        labels = [("b", x) for x in nbrs] + [("m", m) for m in ttno.node_modes[u]]
        for x in nbrs:
            if x == parent:
                continue
            child, cmodes = msg(x, u)
            ax = labels.index(("b", x))
            a = np.tensordot(a, child, axes=([ax], [0]))
            labels = labels[:ax] + labels[ax + 1:] + [("m", m) for m in cmodes]
        modes = [lab[1] for lab in labels if lab[0] == "m"]
        if parent is None:
            order = [labels.index(("m", m)) for m in sorted(modes)]
            return np.ascontiguousarray(a.transpose(order)).reshape(-1), sorted(modes)
        pax = labels.index(("b", parent))
        order = [pax] + [i for i in range(len(labels)) if i != pax]
        return np.ascontiguousarray(a.transpose(order)), modes

    vec, _ = msg(state.center, None)
    return vec


def target_charge(ttno: TTNO, n_elec: int, sector=None) -> QuantumNumber:
    """The quantum number of the sector a solve targets: ``N`` plus, with labels, an irrep.

    ``sector`` is the irrep component as a plain integer tuple. Omitted, it is the identity —
    correct and complete when the quantum number is particle number alone, and a **choice**
    once labels widen it, which is why every caller that has a sector passes one.
    """
    width = ttno.charge.width
    values = [int(n_elec)] + list(sector if sector is not None else [0] * (width - 1))
    if len(values) != width:
        raise ValueError("sector {} has {} components; the quantum number is {} wide "
                         "(one of them is N)".format(sector, len(values) - 1, width))
    return QuantumNumber(*values, moduli=ttno.charge.moduli)


def node_layouts(graph: NetworkGraph, bond_spaces: Dict[Tuple[int, int], Space],
                 phys_space: Sequence[Space], center: int,
                 charge: QuantumNumber) -> List[Tuple[Tuple[Space, ...],
                                                      Tuple[int, ...], QuantumNumber]]:
    """``(spaces, signs, charge)`` of every node tensor for a state centred at ``center``.

    The module docstring's conventions as arithmetic: legs are the bonds in ascending
    neighbor order with the physical leg last, the away-from-center side of every edge
    carries ``-1``, and only the center tensor carries the state's charge.

    ⚠ Shared with the memory plan, which sizes the two-site problem at every bond of the
    tour and therefore has to know what the node tensors look like once the center has
    moved there. A second statement of the sign rule would be a second convention.
    """
    zero = charge.zero_like()
    parent, _ = graph.parents(center)
    layouts = []
    for u in range(graph.n_nodes):
        spaces, signs = [], []
        for x in sorted(graph.neighbors(u)):
            spaces.append(bond_spaces[(min(u, x), max(u, x))])
            signs.append(1 if (u == center or x != int(parent[u])) else -1)
        spaces.append(phys_space[u])
        signs.append(1)
        layouts.append((tuple(spaces), tuple(signs), charge if u == center else zero))
    return layouts


def random_state(ttno: TTNO, n_elec: int, max_bond: int, n_roots: int = 1,
                 center: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None,
                 charge: Optional[QuantumNumber] = None) -> TTNState:
    """A canonicalized random state with charge-consistent bond spaces.

    Bond sectors are the subtree-reachable charges (folded from the TTNO's physical
    spaces) with dimensions ``min(count_inside, count_outside)``, scaled down
    proportionally toward ``max_bond`` — a *guess* builder: the cheap-CI seeding
    replaces it, and nothing downstream depends on how these dimensions were chosen. With a cap at or
    above the full Schmidt rank the state spans the whole charge sector, which is what the
    exactness validation uses.
    """
    graph = ttno.graph
    if graph.n_nodes < 2:
        raise ValueError("a single-node network has no bonds to sweep; use the "
                         "conventional CI for a one-node problem")
    rng = rng if rng is not None else np.random.default_rng()
    center = ttno.root if center is None else int(center)
    width = ttno.charge.width
    # ⚠ With irrep labels widening the quantum number, "N electrons" is no longer a sector:
    # the target has to name the irrep too, and a zero there would silently ask for the
    # totally symmetric one. The caller supplies it; without labels the two agree.
    charge = (target_charge(ttno, n_elec) if charge is None
              else QuantumNumber(*charge, moduli=ttno.charge.moduli))
    zero = ttno.charge.zero_like()

    def fold(nodes) -> Dict[QuantumNumber, int]:
        counts = {zero: 1}
        for w in nodes:
            sp = ttno.phys_space[w]
            new: Dict[QuantumNumber, int] = {}
            for qn, c in counts.items():
                for q, d in zip(sp.qns, sp.dims):
                    key = qn + q
                    new[key] = min(new.get(key, 0) + c * int(d), 10 ** 9)
            counts = new
        return counts

    parent, preorder = graph.parents(center)
    bond_spaces: Dict[Tuple[int, int], Space] = {}
    for u, v in graph.edges:
        away, near = (u, v) if int(parent[u]) == v else (v, u)
        inside = fold(graph.subtree_nodes(away, near))
        outside = fold(graph.subtree_nodes(near, away))
        sectors = []
        for qn, c_in in sorted(inside.items()):
            c_out = outside.get(charge - qn, 0)
            if c_in and c_out:
                sectors.append((qn, min(c_in, c_out)))
        if not sectors:
            raise ValueError("no charge-consistent sectors on bond ({}, {}) for N = {}"
                             .format(u, v, n_elec))
        total = sum(d for _, d in sectors)
        if total > max_bond:
            sectors = [(qn, max(1, (d * max_bond) // total)) for qn, d in sectors]
        bond_spaces[(u, v)] = Space(sectors)

    layouts = node_layouts(graph, bond_spaces, ttno.phys_space, center, charge)
    # Sized and reserved before the first tensor allocates. ⚠ Found by the
    # ab initio ladder: at max_bond=None the charge-sector-
    # maximal bond dimensions of a many-node tree reach hundreds, and the un-reserved
    # random tensors were multi-GB before the first truncation could trim them.
    per_node = [block_tensor_gb(sp, sg, ch) for sp, sg, ch in layouts]
    res.reserve("DMRG random state ({} nodes, {} roots)".format(graph.n_nodes, n_roots),
                sum(per_node) + (n_roots - 1) * per_node[center],
                note="bond dims {}".format(sorted(
                    sp.total_dim for sp in bond_spaces.values())[-3:]),
                advice=["set max_bond deliberately: an uncapped random tree state "
                        "allocates charge-sector-maximal bond dimensions"])
    tensors: List[Optional[BlockTensor]] = [None] * graph.n_nodes
    for u, (spaces, signs, ch) in enumerate(layouts):
        tensors[u] = BlockTensor.random(spaces, signs, ch, rng=rng)

    # canonicalize: children first, QR toward the parent, absorb R into the parent
    for u in [int(x) for x in reversed(preorder)]:
        if u == center:
            continue
        p = int(parent[u])
        ax = _bond_axis(graph, u, p)
        q, r = qr(tensors[u], tuple(i for i in range(tensors[u].ndim) if i != ax))
        perm = list(range(q.ndim - 1))
        perm.insert(ax, q.ndim - 1)
        tensors[u] = q.transpose(perm)
        pax = _bond_axis(graph, p, u)
        t = tensordot(tensors[p], r, axes=([pax], [1]))
        perm = list(range(t.ndim - 1))
        perm.insert(pax, t.ndim - 1)
        tensors[p] = t.transpose(perm)

    seed = tensors[center]
    tensors[center] = None
    centers = [_normalized(BlockTensor.random(seed.spaces, seed.signs, seed.charge,
                                              rng=rng)) for _ in range(n_roots)]
    return TTNState(graph=graph, center=center, tensors=tensors, centers=centers,
                    charge=charge)


def _normalized(t: BlockTensor) -> BlockTensor:
    n = t.norm()
    if n <= 0.0:
        raise ValueError("cannot normalize a zero tensor")
    scaled = t.copy()
    for b in scaled.blocks:
        b /= n
    return scaled


# --- environments -------------------------------------------------------------------------

class EnvironmentCache(object):
    """Renormalized operator blocks per directed bond (module docstring: legs, refresh).

    ⚠ Keys are directed bonds only (:meth:`_key`), which is sufficient **at fixed
    topology** (the Euler tour refresh argument). The adaptive driver uses
    :class:`kuiva.dmrg.reconnect.HashedEnvironmentCache`, which overrides :meth:`_key`
    to include ``NetworkGraph.subtree_hash`` — the extension this class was designed
    to take, and the only sanctioned way to reuse environments across a topology change.
    """

    def __init__(self, ttno: TTNO, state: TTNState, pager=None,
                 resident_cap_gb: Optional[float] = None):
        self.ttno = ttno
        self.state = state
        self._cache: Dict[tuple, BlockTensor] = {}
        self._held: Dict[tuple, object] = {}
        #: Scratch backend for cold environments (kuiva.dmrg.paging.EnvironmentPager), or
        #: ``None`` for the historical always-resident behaviour. With a pager, an
        #: environment whose reservation cannot fit is paged out coldest-first instead of
        #: refusing — and ``resident_cap_gb`` additionally bounds the resident set below
        #: the hard limit, which is what keeps headroom for the two-site solve transients.
        self._pager = pager
        self._resident_cap_gb = resident_cap_gb
        self._recency: Dict[tuple, None] = {}      # insertion-ordered; oldest first

    def _key(self, u: int, v: int) -> tuple:
        return (u, v)

    def _touch(self, key: tuple) -> None:
        self._recency.pop(key, None)
        self._recency[key] = None

    def get(self, u: int, v: int) -> BlockTensor:
        key = self._key(u, v)
        env = self._cache.get(key)
        if env is None and self._pager is not None and self._pager.has(key):
            env = self._pager.load(key)
            self._admit(key, u, v, env, note="paged back in")
        if env is None:
            env = self._build(u, v)
            self._admit(key, u, v, env)
        self._touch(key)
        return env

    def refresh(self, u: int, v: int) -> None:
        """Rebuild ``env(u -> v)`` after the tensor at ``u`` changed (center crossed)."""
        key = self._key(u, v)
        res.BUDGET.release(self._held.pop(key, None))
        self._cache.pop(key, None)
        self._recency.pop(key, None)
        if self._pager is not None:
            self._pager.discard(key)               # a paged copy is equally stale
        env = self._build(u, v)
        self._admit(key, u, v, env)
        self._touch(key)

    def release_all(self) -> None:
        """Drop every entry and give back its memory reservation.

        A cache that outlives its solve would otherwise keep its environments counted as
        resident forever — across the dozens of solves of a CASSCF loop that inflates the
        accounting until a fit calculation is refused (an estimate nobody checks is
        decoration, and one that only grows is worse)."""
        for held in self._held.values():
            res.BUDGET.release(held)
        self._held.clear()
        self._cache.clear()
        self._recency.clear()
        if self._pager is not None:
            self._pager.close()

    # -- admission and eviction ------------------------------------------------------------

    def _admit(self, key: tuple, u: int, v: int, env: BlockTensor, note: str = "") -> None:
        """Put ``env`` on the ledger and in the resident set, paging others out to fit.

        ⚠ Correctness never depends on eviction timing: a consumer that already holds a
        reference (a local problem pinning its window, a build mid-contraction) keeps the
        object alive through that reference; eviction only writes the payload to scratch and
        drops the cache's own reference, so the next ``get`` pages it back in.
        """
        gb = env.nbytes / 1024.0 ** 3
        label = "DMRG environment ({} -> {})".format(u, v)
        note = note or "subtree of {} nodes".format(
            len(self.state.graph.subtree_nodes(u, v)))
        advice = ["reduce max_bond: an environment scales as D^2 * D_op"]
        try:
            held = res.reserve(label, gb, note=note, advice=advice)
        except res.MemoryLimitError:
            if self._pager is None or not self._evict_until(skip=key, need_gb=gb):
                raise
            held = res.reserve(label, gb, note=note, advice=advice)
        self._held[key] = held
        self._cache[key] = env
        if self._resident_cap_gb is not None:
            self._evict_over_cap(skip=key)

    def _resident_env_gb(self) -> float:
        return sum(a.gb for a in self._held.values())

    def _evict_over_cap(self, skip: tuple) -> None:
        while self._resident_env_gb() > self._resident_cap_gb:
            if not self._evict_one(skip):
                break

    def _evict_until(self, skip: tuple, need_gb: float) -> bool:
        """Evict coldest-first until ``need_gb`` fits the budget; False if nothing left.

        A pager failure here — no scratch directory configured, or no space — must not
        replace the refusal the caller was about to raise: paging is an escape hatch, and
        an escape hatch that fails reverts to the honest refusal, with the knob named.
        """
        while res.BUDGET.available_gb() < need_gb:
            try:
                if not self._evict_one(skip):
                    return False
            except (res.ConfigurationError, res.ScratchLimitError) as exc:
                log.warning("the environment cache would page to scratch here, but cannot "
                            "(%s); the memory refusal below stands.",
                            type(exc).__name__)
                return False
        return True

    def _evict_one(self, skip: tuple) -> bool:
        """Page out the least recently used resident environment. False if none qualify."""
        assert self._pager is not None
        for key in self._recency:
            if key != skip and key in self._cache:
                env = self._cache.pop(key)
                self._pager.store(key, env)
                res.BUDGET.release(self._held.pop(key, None))
                self._recency.pop(key, None)
                return True
        return False

    def _w(self, u: int, v: int) -> "_Lab":
        """The node's W tensor, as a hook rather than a direct call, so
        :class:`ShapeEnvironments` can run this exact :meth:`_build` over structure alone
        (:class:`~kuiva.dmrg.block.BlockShape`) instead of over data. One attribute lookup
        per environment build, of which a sweep does a handful."""
        return _w_lab(self.ttno, u, v)

    def _build(self, u: int, v: int) -> BlockTensor:
        graph = self.state.graph
        a = self.state.tensors[u]
        if a is None:
            raise RuntimeError("environment requested for the center node {}".format(u))
        nbrs = sorted(graph.neighbors(u))
        t = _Lab(a, [("b", u, x) for x in nbrs] + [("p", u)])
        for x in nbrs:
            if x == v:
                continue
            env = _Lab(self.get(x, u), [("bra", x), ("op", x), ("ket", x)])
            t = t.dot(env, [(("b", u, x), ("ket", x))])
        t = t.dot(self._w(u, v),
                  [(("op", x), ("op", x)) for x in nbrs if x != v]
                  + [(("p", u), ("pi",))])
        c = _Lab(a.conj(), [("cb", u, x) for x in nbrs] + [("cp",)])
        t = t.dot(c, [(("bra", x), ("cb", u, x)) for x in nbrs if x != v]
                  + [(("po",), ("cp",))])
        return t.to([("cb", u, v), ("op_out",), ("b", u, v)])


def _w_lab(ttno: TTNO, u: int, v: Optional[int]) -> _Lab:
    """The node's W tensor with op legs labelled by neighbor; the root's dim-1
    completed-channel leg is closed here, so no caller ever sees it. The leg toward ``v``
    is labelled ``("op_out",)``."""
    w = ttno.tensors[u]
    labels: List[tuple] = []
    if u == ttno.root:
        w = w.close_leading_leg()
    else:
        p = int(ttno.parent[u])
        labels.append(("op_out",) if p == v else ("op", p))
    for c in ttno.children[u]:
        labels.append(("op_out",) if c == v else ("op", c))
    labels += [("po",), ("pi",)]
    return _Lab(w, labels)


def shape_state(state: TTNState, fill_center: bool = False) -> TTNState:
    """The state with every tensor replaced by its structure — legs, sectors, no data.

    ⚠ ``fill_center`` puts the first center tensor's structure at the center node, so that
    *every* directed environment is buildable — which is what sizing the cache a whole sweep
    fills needs, since the center visits every node in turn. The substitute carries the
    state's total charge where an isometry carries zero, so one bond's figure can differ a
    little from what the cache holds after the center has actually crossed it; the sum is
    the honest statement and the per-bond number is not claimed.
    """
    tensors = [None if t is None else BlockShape.of(t) for t in state.tensors]
    if fill_center:
        tensors[state.center] = BlockShape.of(state.centers[0])
    return TTNState(graph=state.graph, center=state.center, tensors=tensors,
                    centers=[BlockShape.of(c) for c in state.centers],
                    charge=state.charge)


class ShapeEnvironments(EnvironmentCache):
    """:class:`EnvironmentCache` over structure alone: sizes environments without building.

    ⚠ It inherits :meth:`EnvironmentCache._build` **unchanged** — that contraction has one
    description, and a second one written to size it is exactly the drift a memory estimate
    cannot afford. What is overridden is only what touches the world: the ledger (nothing is
    allocated, so nothing is reserved) and where the W tensors come from.

    Construct it on a :func:`shape_state`; ``get(u, v)`` then returns a
    :class:`~kuiva.dmrg.block.BlockShape` whose ``nbytes`` is exactly what the real cache
    would hold at that bond.
    """

    def _w(self, u: int, v: int) -> _Lab:
        lab = _w_lab(self.ttno, u, v)
        return _Lab(BlockShape.of(lab.t), lab.labels)

    def _admit(self, key, u, v, env, note: str = "") -> None:
        self._cache[key] = env


def environment_gb(ttno: TTNO, state: TTNState) -> float:
    """Size [GB] of the whole environment set a sweep caches — every directed bond.

    Entries are added as the center moves and removed only by :meth:`release_all` at the
    end of the solve, so a full sweep leaves all ``2 * (n_nodes - 1)`` of them resident:
    this is what a sweep carries, not a per-bond figure. Structure only — nothing is built,
    and this runs before the first environment exists.
    """
    shapes = ShapeEnvironments(ttno, shape_state(state, fill_center=True))
    total = 0
    for a, b in state.graph.edges:
        total += shapes.get(a, b).nbytes + shapes.get(b, a).nbytes
    return total / 1024.0 ** 3


# --- the local (two-site) problem ---------------------------------------------------------

class _Order(object):
    """One way of applying ``H_eff``: which side goes first and which environments each
    side folds into its operator ahead of the solve (the *halves*)."""

    __slots__ = ("first", "pre")

    def __init__(self, first: int, pre: Dict[int, Tuple[int, ...]]):
        self.first = int(first)
        self.pre = {int(n): tuple(int(x) for x in xs) for n, xs in pre.items()}

    def key(self) -> tuple:
        return (self.first, tuple(sorted(self.pre.items())))

    def __eq__(self, other) -> bool:
        return isinstance(other, _Order) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:                                   # pragma: no cover - debug
        return "_Order(first={}, pre={})".format(self.first, self.pre)


class _LocalProblem(object):
    """One two-site eigenproblem on bond ``(u, v)`` (center at ``u``).

    The variational space is **all** symmetry-allowed blocks of the merged spaces — not
    just the blocks the incoming tensors happen to populate — so the template is built
    with :meth:`BlockTensor.zeros` and every vector is packed against it.

    The order of the effective-Hamiltonian contraction
    --------------------------------------------------
    ``H_eff`` is the merged tensor contracted with each side's environments and W tensor,
    and the *order* of those contractions — not the algorithm — decides the largest array a
    network run ever holds. An intermediate carrying two operator legs at once costs the
    **product** of two operator bond dimensions; measured on a 20-spinor, four-node network
    that was 4.3 GB at a bond dimension of 4, against 0.13 GB for everything the run stored,
    and it grew to 11 GB at D = 16 — which is why network runs were killed by the kernel
    rather than refused. Two rules keep one leg open at a time on a path:

    * the **first** side contracts its environments *before* its W tensor (each environment
      opens one operator leg, W closes them all and opens the single leg toward the other
      node);
    * the **second** side contracts its W tensor *before* its environments (W closes the
      leg from the first side and opens one leg per branch, each environment closes one).

    The mirror orderings are dominated (they hold one leg more at the same step), so this is
    a rule rather than a choice. ⚠ **A node with several branches still opens one leg per
    branch**, and there the choice is real: folding an environment into its W tensor ahead
    of the solve (a *half*, built once per two-site problem and reused across every
    Davidson iteration) closes that leg at the price of a dense ``d^2`` payload — larger
    than the open-leg intermediate for a fat node at small bond dimension, smaller by about
    a factor ``D`` otherwise. Which, and for which side, is a **structural search**
    (:meth:`_choose_order`): every candidate order is walked over
    :class:`~kuiva.dmrg.block.BlockShape` operands, allocating nothing, and the one with the
    smallest peak is taken. A side with a single branch never takes a half — the reorder
    already leaves it one open leg, so a half there trades a sparse product per
    matrix-vector product for a dense one and buys no memory.

    ⚠ **The halves are built by :meth:`prepare`, not by the constructor**, so the memory
    check in :func:`_solve_local` still precedes every allocation this object makes; the
    plan sizes them through the same chain that builds them. ⚠ **A different contraction
    order is not bitwise**: the same products are summed in a different order, so this
    agrees with the previous order to rounding and every committed network reference was
    re-checked against its tolerance when it landed, never assumed.
    """

    def __init__(self, ttno: TTNO, state: TTNState, cache: EnvironmentCache,
                 u: int, v: int):
        graph = state.graph
        self.u, self.v = u, v
        self.branches_u = [x for x in sorted(graph.neighbors(u)) if x != v]
        self.branches_v = [y for y in sorted(graph.neighbors(v)) if y != u]
        # ⚠ Both operands may be structures rather than tensors
        # (:class:`~kuiva.dmrg.block.BlockShape`), which is how the memory plan sizes every
        # bond of a sweep before a single one has been solved. Nothing else in the class
        # branches on it: the chain, the sizing and the template all follow from here.
        merge = ([_bond_axis(graph, u, v)], [_bond_axis(graph, v, u)])
        centre = state.centers[0]
        m0 = centre.dot(state.tensors[v], merge) if isinstance(centre, BlockShape) \
            else tensordot(centre, state.tensors[v], axes=merge)
        self.structural = isinstance(m0, BlockShape)
        self.labels = [("b", u, x) for x in self.branches_u] + [("p", u)] \
            + [("b", v, y) for y in self.branches_v] + [("p", v)]
        self.n_left = len(self.branches_u) + 1
        if self.structural:
            self.template = BlockShape.allowed(m0.spaces, m0.signs, m0.charge)
            self.sizes = [int(x) for x in self.template.block_sizes]
        else:
            self.template = BlockTensor.zeros(m0.spaces, m0.signs, m0.charge)
            self.sizes = [b.size for b in self.template.blocks]
        self.dim = int(sum(self.sizes))
        #: Exact size of one merged root tensor, kept from the one that was built anyway —
        #: :meth:`solve_workspace_gb` needs it and rebuilding it to size it would be absurd.
        self.merged_bytes = int(m0.nbytes)
        self.envs_u = [(x, _Lab(cache.get(x, u), [("bra", x), ("op", x), ("ket", x)]))
                       for x in self.branches_u]
        self.envs_v = [(y, _Lab(cache.get(y, v), [("bra", y), ("op", y), ("ket", y)]))
                       for y in self.branches_v]
        self.w_u = _w_lab(ttno, u, v)
        self.w_v = _w_lab(ttno, v, u)
        # the per-side operands the chain works with: environments keyed by branch, and each
        # W with its physical legs named by node so the two sides' never collide
        self._envs = {x: env for x, env in self.envs_u + self.envs_v}
        self._w = {u: self._w_side(self.w_u, u), v: self._w_side(self.w_v, v)}
        self._branches = {u: list(self.branches_u), v: list(self.branches_v)}
        self.order, self.candidates = self._choose_order()
        #: The pre-contracted halves, ``node -> _Lab``; ``None`` until :meth:`prepare`.
        self.halves = None

    @staticmethod
    def _w_side(w: _Lab, node: int) -> _Lab:
        return _Lab(w.t, [("po", node) if l == ("po",) else ("pi", node) if l == ("pi",)
                          else l for l in w.labels])

    def pack(self, t: BlockTensor) -> np.ndarray:
        lookup = {tuple(int(i) for i in r): b for r, b in zip(t.sectors, t.blocks)}
        vec = np.zeros(self.dim, dtype=np.complex128)
        pos = 0
        for row, size in zip(self.template.sectors, self.sizes):
            b = lookup.pop(tuple(int(i) for i in row), None)
            if b is not None:
                vec[pos:pos + size] = b.ravel()
            pos += size
        if lookup:
            raise ValueError("tensor carries blocks outside the merged template")
        return vec

    def unpack(self, vec: np.ndarray) -> BlockTensor:
        blocks = []
        pos = 0
        for size, tmpl in zip(self.sizes, self.template.blocks):
            blocks.append(np.ascontiguousarray(vec[pos:pos + size].reshape(tmpl.shape)))
            pos += size
        # trusted: identical structure to the (validated) template, once per matvec
        return BlockTensor._trusted(self.template.spaces, self.template.signs,
                                    self.template.charge, self.template.sectors,
                                    self.template._keys, blocks)

    # -- the chain -------------------------------------------------------------------------

    @staticmethod
    def _half(w: _Lab, envs: Dict[int, _Lab], pre: Sequence[int],
              sizes: Optional[List[int]] = None) -> _Lab:
        """One side's operator with the environments ``pre`` folded in, or the bare W.

        The environment is the *left* operand at every step so a sparse W stays on the
        right, where :meth:`_Lab.dot` takes the sparse contraction; the result is a dense
        block tensor with legs ``[bra, ket]`` per folded branch ahead of W's own.
        """
        h = w
        for x in pre:
            h = envs[x].dot(h, [(("op", x), ("op", x))])
            if sizes is not None:
                sizes.append(h.t.nbytes)
        return h

    def _chain(self, t: _Lab, envs: Dict[int, _Lab], halves: Dict[int, _Lab],
               order: _Order, sizes: Optional[List[int]] = None) -> _Lab:
        """The effective-Hamiltonian contraction, over data or over structure.

        ⚠ **One description of the chain, run twice.** With ``_Lab``s over
        :class:`~kuiva.dmrg.block.BlockTensor` this is the matrix-vector product Davidson
        calls; with ``_Lab``s over :class:`~kuiva.dmrg.block.BlockShape` it propagates
        sector tables and allocates nothing, which is how :meth:`apply_peak_gb` knows the
        size of an intermediate before the first one is built and how :meth:`_choose_order`
        compares orders. A sizing function written beside the chain instead of *through*
        it would drift the first time the order changed — and the order is now chosen.

        ``sizes`` collects every intermediate's byte count in order, when asked.
        """
        first = order.first
        second = self.v if first == self.u else self.u
        # first side: environments open one operator leg each, the half closes them and
        # opens the single leg toward the second node
        n, pre = first, order.pre[first]
        loose = [x for x in self._branches[n] if x not in pre]
        for x in loose:
            t = t.dot(envs[x], [(("b", n, x), ("ket", x))])
            if sizes is not None:
                sizes.append(t.t.nbytes)
        t = t.dot(halves[n], [(("op", x), ("op", x)) for x in loose]
                  + [(("b", n, x), ("ket", x)) for x in pre]
                  + [(("p", n), ("pi", n))])
        if sizes is not None:
            sizes.append(t.t.nbytes)
        # second side: the half closes that leg and opens one per loose branch, each
        # environment closes one
        n, pre = second, order.pre[second]
        loose = [x for x in self._branches[n] if x not in pre]
        t = t.dot(halves[n], [(("op_out",), ("op_out",))]
                  + [(("b", n, x), ("ket", x)) for x in pre]
                  + [(("p", n), ("pi", n))])
        if sizes is not None:
            sizes.append(t.t.nbytes)
        for x in loose:
            t = t.dot(envs[x], [(("b", n, x), ("ket", x)), (("op", x), ("op", x))])
            if sizes is not None:
                sizes.append(t.t.nbytes)
        return t

    # -- choosing the order (structure only) -----------------------------------------------

    def _shape_operands(self):
        """The chain's operands as structures: the template, environments and W tensors."""
        start = _Lab(BlockShape.of(self.template), list(self.labels))
        envs = {x: _Lab(BlockShape.of(e.t), e.labels) for x, e in self._envs.items()}
        ws = {n: _Lab(BlockShape.of(w.t), w.labels) for n, w in self._w.items()}
        return start, envs, ws

    def _size_order(self, order: _Order, start: _Lab, envs: Dict[int, _Lab],
                    ws: Dict[int, _Lab]):
        """``(peak_bytes, halves_bytes, build_sizes, apply_sizes)`` of one order.

        Peak model: the halves are resident for the whole solve; on top of them the largest
        step of the application holds its input and its output together; building a half
        holds the halves already finished, the partial one and the new one. The bare W of
        a side with nothing folded in is the operator's own tensor and costs nothing new.
        """
        halves: Dict[int, _Lab] = {}
        build: List[int] = []
        resident = 0
        build_peak = 0
        for n in (order.first, self.v if order.first == self.u else self.u):
            steps: List[int] = []
            halves[n] = self._half(ws[n], envs, order.pre[n], sizes=steps)
            prev = 0
            for out in steps:
                build_peak = max(build_peak, resident + prev + out)
                prev = out
            if order.pre[n]:
                resident += halves[n].t.nbytes
            build.extend(steps)
        sizes: List[int] = []
        self._chain(start, envs, halves, order, sizes=sizes)
        apply_peak = max(a + b for a, b in zip([start.t.nbytes] + sizes[:-1], sizes))
        return max(build_peak, resident + apply_peak), resident, build, sizes

    def _choose_order(self):
        """The order with the smallest structural peak, and every candidate beside it.

        Candidates: which side goes first, and for each side with **two or more** branches
        every subset of them to fold into its half (a single-branch side never takes one —
        the class docstring says why). On a path that is one candidate and no search; on a
        tree it is ``2 * 2^k`` per branching side, walked over structure in milliseconds.
        Ties go to fewer halves, then to ``u`` first, so the choice is deterministic.
        """
        from itertools import combinations

        start, envs, ws = self._shape_operands()

        def subsets(node):
            branches = self._branches[node]
            if len(branches) < 2:
                return [()]
            return [c for r in range(len(branches) + 1)
                    for c in combinations(branches, r)]

        branching = max(len(self.branches_u), len(self.branches_v)) >= 2
        firsts = (self.u, self.v) if branching else (self.u,)
        candidates = []
        for first in firsts:
            for pre_u in subsets(self.u):
                for pre_v in subsets(self.v):
                    order = _Order(first, {self.u: pre_u, self.v: pre_v})
                    peak, resident, build, sizes = self._size_order(order, start, envs, ws)
                    candidates.append((order, peak, resident, build, sizes))
        best = min(candidates, key=lambda c: (c[1], len(c[0].pre[self.u])
                                              + len(c[0].pre[self.v]),
                                              c[0].first != self.u))
        return best[0], candidates

    def _chosen(self):
        for c in self.candidates:
            if c[0] == self.order:
                return c
        raise RuntimeError("the chosen order is not among the candidates")  # pragma: no cover

    # -- sizes -----------------------------------------------------------------------------

    def intermediate_bytes(self) -> List[int]:
        """Byte count of every intermediate one :meth:`apply` builds, in order.

        Structure only: no operand data is touched and nothing of the size being reported
        is allocated. The input is the *template*'s structure — every symmetry-allowed
        block — which is exactly what :meth:`unpack` produces for any vector.
        """
        return list(self._chosen()[4])

    def half_build_bytes(self) -> List[int]:
        """Byte count of every intermediate :meth:`prepare` builds, in order (empty on a
        path, where no environment is folded into a W tensor)."""
        return list(self._chosen()[3])

    def halves_gb(self) -> float:
        """Resident [GB] of the pre-contracted halves, held for the whole solve."""
        return self._chosen()[2] / 1024.0 ** 3

    def apply_peak_gb(self) -> float:
        """Peak [GB] one application of ``H_eff`` holds beyond what the solve keeps resident
        — the layer's largest allocation.

        The model is the largest step of the chain, an intermediate and the one it is built
        from being live together: ``max over steps of (input + output)``. It is **measured**
        rather than argued: against the process's resident set across one application it is
        right to 0.02% on every bond of a 20-spinor network, at three bond dimensions.

        ⚠ This is a **transient**, built and freed once per Davidson matrix-vector product,
        which is why no ledger entry ever saw it and why the process oscillated between 4
        and 11 GB under a declared 0.13 GB. ⚠ And it is a statement about Kuiva's *arrays*,
        not about the process: an allocator that keeps a freed arena rather than returning
        it to the operating system is what the budget's ``warn_fraction`` is for, and it
        cannot be sized from here. The halves are not in it — they are resident across the
        solve and :meth:`solve_workspace_gb` carries them.
        """
        sizes = self.intermediate_bytes()
        if not sizes:                                    # pragma: no cover - a bond acts
            return 0.0
        peak = max(a + b for a, b in zip([self.template.nbytes] + sizes[:-1], sizes))
        return peak / 1024.0 ** 3

    def solve_workspace_gb(self, n_roots: int, n_solve: int) -> float:
        """Resident [GB] of one two-site solve, exclusive of :meth:`apply_peak_gb`.

        Exactly the five things :func:`_solve_local` holds across the Davidson call: the
        ``n_roots`` merged root tensors, their packed copies (the guess), the eigensolver's
        three ``(subspace cap, dim)`` stacks and its converged vectors, the unpacked
        roots handed back, and the pre-contracted operator halves (:meth:`halves_gb`).

        ⚠ The stacks are the term the old estimate missed and they are the term that scales
        with the **root count**: the subspace cap is a multiple of it, so a 10-root solve
        holds ten times what a one-root solve does where the previous ``(n_roots + 2) *
        dim`` line grew by barely a factor of four.
        """
        vec = 16.0 * self.dim
        return ((n_roots * (self.merged_bytes + vec + vec) + n_solve * vec)
                / 1024.0 ** 3) \
            + davidson_workspace_gb(self.dim, subspace_cap(n_solve, self.dim)) \
            + self.halves_gb()

    # -- the application -------------------------------------------------------------------

    def prepare(self) -> "_LocalProblem":
        """Build the halves the chosen order needs (a no-op on a path). Called by
        :func:`_solve_local` after its memory check, and by :meth:`apply` if nobody did."""
        if self.halves is None:
            if self.structural:
                raise TypeError("a structural local problem sizes its halves; it cannot "
                                "build them")
            self.halves = {n: self._half(self._w[n], self._envs, self.order.pre[n])
                           for n in (self.u, self.v)}
        return self

    def apply(self, vec: np.ndarray) -> np.ndarray:
        u, v = self.u, self.v
        if self.halves is None:
            self.prepare()
        t = self._chain(_Lab(self.unpack(vec), list(self.labels)), self._envs,
                        self.halves, self.order)
        labels = []
        for lab in t.labels:
            if lab[0] == "bra":
                x = lab[1]
                labels.append(("b", u, x) if x in self.branches_u else ("b", v, x))
            elif lab[0] == "po":
                labels.append(("p", lab[1]))
            else:
                labels.append(lab)
        return self.pack(_Lab(t.t, labels).to(self.labels))

    def diagonal(self) -> np.ndarray:
        """``<I|H_eff|I>`` over the packed space, exactly (Davidson preconditioner).

        Only charge-0 operator sectors can contribute to a diagonal element (a diagonal
        element conserves every subtree charge), which keeps this one small dense
        contraction per template block. A wrong diagonal can only slow Davidson down,
        never corrupt an answer — it is tested against a dense build regardless.
        """
        halves = []
        for node, branches, envs, w in ((self.u, self.branches_u, self.envs_u, self.w_u),
                                        (self.v, self.branches_v, self.envs_v, self.w_v)):
            wd = _w_diag(w)
            if wd is None:
                return np.zeros(self.dim, dtype=np.float64)
            envd = {}
            for x, env in envs:
                d = _env_diag(env.t)
                if d is None:
                    return np.zeros(self.dim, dtype=np.float64)
                envd[x] = d
            halves.append((branches, envd, wd))

        diag = np.empty(self.dim, dtype=np.float64)
        pos = 0
        k = len(self.branches_u)
        for row, tmpl in zip(self.template.sectors, self.template.blocks):
            parts = []
            for half, (branches, envd, wd) in enumerate(halves):
                off = 0 if half == 0 else k + 1
                t = _DenseLab(wd.a, list(wd.labels))
                p_sec = int(row[off + len(branches)])
                p_space = self.template.spaces[off + len(branches)]
                psl = slice(int(p_space.offsets[p_sec]), int(p_space.offsets[p_sec + 1]))
                t = _DenseLab(t.a[..., psl], t.labels)          # phys axis is last
                for j, x in enumerate(branches):
                    b_space = self.template.spaces[off + j]
                    b_sec = int(row[off + j])
                    bsl = slice(int(b_space.offsets[b_sec]),
                                int(b_space.offsets[b_sec + 1]))
                    e = _DenseLab(envd[x][:, bsl], [("op", x), ("m", x)])
                    t = t.dot(e, [(("op", x), ("op", x))])
                parts.append(t.to([("op_out",)] + [("m", x) for x in branches]
                                  + [("phys",)]))
            block = np.real(np.tensordot(parts[0], parts[1], axes=([0], [0])))
            diag[pos:pos + tmpl.size] = block.reshape(-1)
            pos += tmpl.size
        return diag


def _charge_zero_slice(space: Space) -> Optional[slice]:
    zero = space.qns[0].zero_like()
    for i, qn in enumerate(space.qns):
        if qn == zero:
            return slice(int(space.offsets[i]), int(space.offsets[i + 1]))
    return None


def _env_diag(env: BlockTensor) -> Optional[np.ndarray]:
    """Dense ``(d_op0, D_bond)`` diagonal of an environment (charge-0 op sector only)."""
    op_space, bond_space = env.spaces[1], env.spaces[2]
    sl = _charge_zero_slice(op_space)
    if sl is None:
        return None
    arr = np.zeros((sl.stop - sl.start, bond_space.total_dim), dtype=np.complex128)
    zero = op_space.qns[0].zero_like()
    for row, block in zip(env.sectors, env.blocks):
        sb, so, sk = (int(i) for i in row)
        if sb != sk or op_space.qns[so] != zero:
            continue
        piece = np.einsum("mom->om", block)
        b0 = int(bond_space.offsets[sb])
        arr[:, b0:b0 + piece.shape[1]] = piece
    return arr


def _w_diag(w: _Lab) -> Optional[_DenseLab]:
    """Physical-diagonal of a W tensor with every op axis restricted to charge 0.

    Returned as a labelled dense array ``[op labels.., ("phys",)]`` so the caller pairs op
    axes with environments by *name*, exactly as the sparse contraction does. ``None``
    when some op axis has no charge-0 sector (no diagonal contribution at this node).
    """
    t = w.t
    n_op = t.ndim - 2
    slices = [_charge_zero_slice(t.spaces[i]) for i in range(n_op)]
    if any(s is None for s in slices):
        return None
    phys = t.spaces[-1]
    zero = t.charge.zero_like()
    arr = np.zeros(tuple(s.stop - s.start for s in slices) + (phys.total_dim,),
                   dtype=np.complex128)
    for b, row in enumerate(t.sectors):
        if int(row[-2]) != int(row[-1]):
            continue
        if any(t.spaces[i].qns[int(row[i])] != zero for i in range(n_op)):
            continue
        idx, val = t.block_entries(b)
        shape = tuple(int(sp.dims[int(i)]) for sp, i in zip(t.spaces, row))
        multi = np.unravel_index(idx, shape)
        keep = multi[-2] == multi[-1]                  # the physical diagonal only
        if not keep.any():
            continue
        p0 = int(phys.offsets[int(row[-1])])
        target = tuple(m[keep] for m in multi[:n_op]) + (p0 + multi[-1][keep],)
        np.add.at(arr, target, val[keep])
    return _DenseLab(arr, list(w.labels[:-2]) + [("phys",)])


# --- the sweep ----------------------------------------------------------------------------

@dataclass(eq=False)
class SweepResult(object):
    """What a DMRG run produced. Energies exclude ``e_core`` (module docstring)."""

    energies: np.ndarray                 # (n_roots,) final local eigenvalues, ascending
    weights: np.ndarray                  # (n_roots,) equalized averaging weights
    state: TTNState
    converged: bool
    n_sweeps: int
    max_discarded: float                 # largest ensemble truncation weight, final sweep
    max_bond_dim: int
    boundary_gap_cm: Optional[float]     # None: not checked, or the average is complete
    history: List[float]                 # SA energy after each sweep


def solve_ttn(ttno: TTNO, state: TTNState, *, max_sweeps: int = 25,
              conv_tol: float = 1e-9, trunc_tol: float = 0.0,
              max_bond: Optional[int] = None, weights: Optional[Sequence[float]] = None,
              n_elec: Optional[int] = None, boundary_check: int = 4,
              davidson_tol: float = 1e-8, on_split: str = "raise",
              page_environments: bool = True,
              environment_resident_gb: Optional[float] = None,
              checkpoint=None,
              bond_schedule: Optional[Sequence[int]] = None,
              expansion: float = 0.0, expansion_sweeps: int = 6,
              memory_plan: bool = True,
              report: bool = True) -> SweepResult:
    """State-averaged two-site DMRG to energy stationarity (module docstring).

    ``state`` is updated in place and returned inside the result. ``conv_tol`` is on the
    change of the state-averaged energy between sweeps; ``davidson_tol`` on the local
    residual norms. ``weights`` are the *requested* averaging weights — they are
    re-equalized inside degenerate blocks at every update, and the equalized
    weights are what both the truncation and the reported SA energy use.

    ``page_environments`` (default on) lets the environment cache page its coldest entries
    to scratch **when a reservation would otherwise refuse** — a two-site window touches
    ``O(degree)`` of the ``2(n-1)`` cached environments, so the cold tail is exactly what a
    memory refusal does not need resident. Strictly an escape hatch: a solve whose
    environments fit never touches scratch, a solve that pages produces bit-identical
    energies (the paged tensor is the written tensor), and a machine with no scratch
    directory configured gets the same refusal it always got, with the knob named.
    ``environment_resident_gb`` additionally caps the resident environment set below the
    hard limit — which is what keeps headroom for the two-site solve's own transients on a
    machine where the environments *barely* fit.

    ``memory_plan`` (default on) prints the sweep's memory plan and **refuses before the
    first bond** if it cannot fit (:func:`kuiva.dmrg.plan.network_memory_plan`). ⚠ It is the
    only thing that makes the memory limit mean anything here: the layer's largest array is
    a transient inside one effective-Hamiltonian application, so without this a run that
    does not fit is not refused but killed by the kernel, with no message. Turn it off only
    where a driver plans for itself once per chart — :class:`kuiva.dmrg.solver.DMRGSolver`
    does exactly that, so a CASSCF does not print the same table sixty times.

    ``report=False`` moves the sweep table to DEBUG — for a solver called once per
    CASSCF macro-iteration, whose driver already owns the INFO table (INFO is the
    output file, and sixty inner tables in it are noise, not output).

    ``checkpoint`` is a callable ``checkpoint(state, sweep=, energies=, converged=)``
    invoked at the end of **each completed sweep** — rolling network-state checkpointing
    (:class:`kuiva.dmrg.checkpoint.NetworkCheckpointPolicy`), which owns its own cadence
    and failure semantics; this loop only reports the sweep. It is called once more at
    convergence with ``converged=True`` (that write is unconditional by the policy's own
    rule), and never during the boundary diagnostic, whose extra roots are discarded.

    Production controls
    -------------------
    ``bond_schedule`` ramps the cap **within this solve**: entry ``s`` caps sweep
    ``s + 1``, the last entry holds from there on, and it must equal ``max_bond`` (or
    supplies it when ``max_bond`` is ``None``). The variational manifold is defined by
    the final cap alone, so the ramp is an iteration strategy, not a chart change —
    convergence is therefore only *declared* on sweeps running at the final cap.

    ``expansion`` (``alpha``) enriches every truncation with the ``alpha``-scaled
    operator-channel columns of :func:`_expansion_columns` — the deterministic
    two-site, ensemble form of the subspace expansion (Hubig et al. 2015), which is
    White's density-matrix perturbation (2005) evaluated instead of sampled. The choice
    of the deterministic form over the sampled one is deliberate and load-bearing for
    the degenerate-group truncation rule: the ensemble density plus the H-channel term
    commutes with time reversal whenever the ensemble and ``H`` do, so degenerate
    Schmidt groups stay degenerate to rounding and the group-complete cut keeps meaning
    what it says — a *sampled* perturbation would split them by ``O(alpha)`` and turn
    the group rule into a coin toss (it would also put an RNG into an example's
    trajectory, which the reproducibility rules forbid). ``alpha`` decays by 4x each
    sweep and is off after ``expansion_sweeps``; convergence is only declared on
    unperturbed sweeps. The energies are Davidson eigenvalues and stay variational at
    every ``alpha``; the reported ``w_disc`` is always the **ensemble's own** discarded
    weight, recomputed from the committed projections when the expansion is active,
    never the augmented density's. ⚠ Cost: the truncation SVD's column count grows by
    the operator bond dimension while ``alpha`` is on — pay it in the first sweeps,
    where it buys escape from a local minimum, not at convergence.
    """
    graph = state.graph
    n_roots = state.n_roots
    n_elec = state.charge.n if n_elec is None else int(n_elec)
    requested = np.full(n_roots, 1.0 / n_roots) if weights is None \
        else np.asarray(weights, dtype=float) / float(np.sum(weights))
    if requested.size != n_roots:
        raise ValueError("{} weights for {} roots".format(requested.size, n_roots))
    schedule = None
    if bond_schedule is not None:
        schedule = [int(d) for d in bond_schedule]
        if not schedule or any(d <= 0 for d in schedule) \
                or any(b < a for a, b in zip(schedule, schedule[1:])):
            raise ValueError("bond_schedule must be a non-empty ascending sequence of "
                             "positive caps, got {}".format(bond_schedule))
        if max_bond is None:
            max_bond = schedule[-1]
        elif schedule[-1] != int(max_bond):
            raise ValueError(
                "bond_schedule ends at {} but max_bond is {}. The final cap defines the "
                "variational manifold (it is what a solver's space_key names), so the "
                "two must be one number — drop max_bond or make them agree"
                .format(schedule[-1], max_bond))
    alpha0 = float(expansion)
    if alpha0 < 0.0:
        raise ValueError("expansion must be non-negative, got {}".format(expansion))

    if memory_plan:
        # Before the first environment and the first two-site problem, which is the earliest
        # point every network dimension is known and the last point before the layer's
        # transients start. ⚠ ``begin=False``: this is a phase of a calculation already
        # under way (the CASSCF that called it reserved the integrals and the operator),
        # not the start of one, and stamping a new generation here would report those as
        # leftovers from somebody else's run.
        from .plan import network_memory_plan
        res.preflight(network_memory_plan(ttno, state, n_roots=n_roots,
                                          extra_roots=max(0, int(boundary_check)),
                                          max_bond=max_bond),
                      title="Memory plan: two-site DMRG", begin=False)

    pager = None
    if page_environments or environment_resident_gb is not None:
        from .paging import EnvironmentPager
        pager = EnvironmentPager()
    cache = EnvironmentCache(ttno, state, pager=pager,
                             resident_cap_gb=environment_resident_gb)
    table = out.Table(log, [out.col_iter("sweep"), out.col_energy("E_SA [Eh]"),
                            out.col_delta(), out.col_sci("w_disc"),
                            out.col_count("max D", 6), out.col_time()],
                      level=logging.INFO if report else logging.DEBUG)
    table.start("state-averaged two-site DMRG ({} roots, {} nodes)".format(
        n_roots, graph.n_nodes))

    e_prev = None
    de = None
    converged = False
    energies = w_used = None
    max_disc = 0.0
    max_dim = 0
    history: List[float] = []
    # ⚠ The sweep is a KERNEL region, not a BLAS one, and that is measured rather than
    # assumed: measurement found the threaded BLAS buying this layer nothing
    # (identical wall at 1 and 8 threads) while spending four CPU seconds per wall second
    # in spin-wait, while the pair-table kernels scale. So MKL is
    # clamped to one thread for the whole sweep and the budget goes to the kernels.
    # Entered once per solve, never per bond (B8).
    with threads.kernel_region(), timer("DMRG sweeps"):
        for sweep in range(1, max_sweeps + 1):
            t0 = time.perf_counter()
            max_disc = 0.0
            max_dim = 0
            cap = max_bond if schedule is None \
                else schedule[min(sweep - 1, len(schedule) - 1)]
            alpha = alpha0 * 4.0 ** (1 - sweep) if sweep <= int(expansion_sweeps) else 0.0
            for u, v in graph.sweep_schedule(state.center):
                energies, w_used, info, _ = _update_bond(
                    ttno, state, cache, u, v, requested, n_elec, trunc_tol, cap,
                    davidson_tol, on_split, expansion=alpha)
                max_disc = max(max_disc, info.discarded_weight)
                max_dim = max(max_dim, info.bond_dim)
            e_sa = float(np.dot(w_used, energies))
            de = None if e_prev is None else e_sa - e_prev
            history.append(e_sa)
            table.row(sweep, e_sa, de, max_disc, max_dim, time.perf_counter() - t0)
            # ⚠ Only a sweep at the final cap and without the expansion may declare
            # convergence: a ramp sweep is stationary on a smaller manifold than the one
            # the solve promises, and a perturbed truncation is not the fixed point.
            final_form = (schedule is None or sweep >= len(schedule)) and alpha == 0.0
            if final_form and de is not None and abs(de) < conv_tol:
                converged = True
            if checkpoint is not None:
                checkpoint(state, sweep=sweep, energies=[float(e) for e in energies],
                           converged=converged)
            if converged:
                break
            e_prev = e_sa
    table.end("converged" if converged else "NOT converged in {} sweeps".format(
        max_sweeps))
    if not converged:
        log.warning("two-site DMRG did not reach dE < %.1e in %d sweeps (last dE %s)",
                    conv_tol, max_sweeps,
                    "n/a" if de is None else "{:.1e}".format(de))

    # the state-averaging gate, on the converged spectrum (module docstring): equalized weights with
    # the theorem-backed refusal of a count that splits a Kramers pair — this is where the
    # RDM-consuming weights come from, so this is where the discipline is imposed
    w_final = state_average_weights([float(e) for e in energies], n_elec, requested,
                                    on_split=on_split)

    gap_cm = None
    if converged and boundary_check > 0:
        with threads.kernel_region():                     # same policy as the sweep above
            gap_cm = _boundary_sweep(ttno, state, cache, requested, n_elec, trunc_tol,
                                     max_bond, davidson_tol, on_split, boundary_check)
    cache.release_all()
    return SweepResult(energies=np.asarray(energies), weights=np.asarray(w_final),
                       state=state, converged=converged, n_sweeps=len(history),
                       max_discarded=max_disc, max_bond_dim=max_dim,
                       boundary_gap_cm=gap_cm, history=history)


def _sweep_weights(energies: np.ndarray, requested: np.ndarray,
                   tol: float = DEFAULT_DEGENERACY_TOL) -> np.ndarray:
    """Equalize weights within observed degenerate blocks — the sweep-time half of state averaging.

    No structural (odd-electron) refusal here, on purpose: see the module docstring. The
    theorem-backed gate runs once, on the converged spectrum, in :func:`solve_ttn`.
    """
    w = requested.copy()
    for a, b in degenerate_blocks([float(e) for e in energies], tol):
        w[a:b] = float(np.mean(requested[a:b]))
    return w / float(np.sum(w))


def _update_bond(ttno, state, cache, u, v, requested, n_elec, trunc_tol, max_bond,
                 davidson_tol, on_split, extra_roots=0, expansion=0.0):
    """One two-site update; with ``extra_roots`` it also returns the state-average boundary gap.

    ⚠ Extra roots are solved and *discarded* — they never enter the average or the
    truncation (the extra roots are discarded, never averaged over).
    """
    prob, energies, w_used, roots, gap = _solve_local(ttno, state, cache, u, v,
                                                      requested, davidson_tol,
                                                      extra_roots=extra_roots)
    info = _commit_split(state, cache, prob, roots, w_used, trunc_tol, max_bond,
                         expansion=expansion)
    return energies, w_used, info, gap


def _solve_local(ttno, state, cache, u, v, requested, davidson_tol, extra_roots=0):
    """Solve the two-site eigenproblem on bond ``(u, v)`` without committing anything.

    Returns ``(prob, energies, w_used, roots, gap)`` — the local problem, the lowest
    ``n_roots`` energies, the sweep-equalized weights, the merged two-site root tensors,
    and (with ``extra_roots``) the state-average boundary gap. The state is untouched; the caller
    decides how to split (the seam adaptive reconnection uses to evaluate candidate
    topologies on the *same* merged ensemble before committing one).
    """
    graph = state.graph
    prob = _LocalProblem(ttno, state, cache, u, v)
    n_roots = state.n_roots
    if prob.dim < n_roots:
        # ⚠ Not necessarily a bond-dimension shortage: the two-site space on a bond that
        # separates very few spinors from the rest can be smaller than the root count no
        # matter what the cap is, because the charge sectors reachable there simply do not
        # hold that many states. Refusing is right — an ensemble that cannot be represented
        # must not be silently truncated to one that can.
        raise ValueError("the two-site space on bond ({}, {}) has dimension {}, below the {} "
                         "roots being averaged; ask for fewer roots, raise max_bond, or give "
                         "the network a topology whose bonds separate more orbitals"
                         .format(u, v, prob.dim, n_roots))
    n_solve = min(n_roots + int(extra_roots), prob.dim)
    # ⚠ The whole of what this solve holds, and the second term is the one that was missing
    # for the layer's entire history: the effective-Hamiltonian application's intermediate
    # is a *transient*, built and freed once per Davidson matrix-vector product, and with
    # two operator legs open it was two orders of magnitude larger than everything declared
    # around it — a 20-spinor network measured 10.8 GB of it against a 0.13 GB ledger, and
    # the run died as an OOM kill rather than as a refusal. The chain now keeps one leg open
    # (the _LocalProblem docstring), but the term stays: it is what refuses a branching
    # node's order that does not fit. Both terms are exact and neither is padded.
    workspace = prob.solve_workspace_gb(n_roots, n_solve)
    transient = prob.apply_peak_gb()
    res.require("DMRG two-site problem ({}-{})".format(u, v), workspace + transient,
                note="{} determinants; {:.3f} GB workspace + {:.3f} GB per H_eff "
                     "application".format(prob.dim, workspace, transient),
                advice=[
                    "reduce max_bond: the local dimension scales as D^2 d^2 and the "
                    "application's intermediate carries one operator bond dimension on "
                    "top of that",
                    "reduce n_states: the Davidson stacks scale with the subspace cap, "
                    "which is a multiple of the root count",
                    "a finer node partition (more nodes, fewer modes each) shrinks the "
                    "application's intermediate quadratically in the local dimension"])
    # ⚠ Only now: the pre-contracted halves are this object's one allocation of its own,
    # and they are built after the check so that a refusal precedes every byte of them.
    prob.prepare()
    merged = [tensordot(c, state.tensors[v],
                        axes=([_bond_axis(graph, u, v)], [_bond_axis(graph, v, u)]))
              for c in state.centers]
    guess = np.stack([prob.pack(m) for m in merged])
    # ⚠ davidson's dense fallback applies H `ndet` times, which is the right trade for a
    # cheap CI sigma and exactly wrong here, where one H_eff application is the expensive
    # object — measured 40x on a D = 16 sweep. Only genuinely tiny problems go dense.
    result = davidson(prob.apply, prob.diagonal(), n_solve, guess=guess,
                      conv_tol=davidson_tol, dense_max_det=4 * n_solve + 2,
                      label="DMRG bond ({}, {})".format(u, v))
    gap = None if n_solve == n_roots \
        else float(result.energies[n_roots] - result.energies[n_roots - 1])
    w_used = _sweep_weights(result.energies[:n_roots], requested)
    roots = [prob.unpack(vec) for vec in result.vectors[:n_roots]]
    return prob, result.energies[:n_roots], w_used, roots, gap


def _branch_axis(labels, x):
    """Position of the bond label toward node ``x`` in a merged-leg label list."""
    for i, lab in enumerate(labels):
        if lab[0] == "b" and lab[2] == x:
            return i
    raise ValueError("no bond leg toward node {} among {}".format(x, labels))


def _phys_axis(labels):
    for i, lab in enumerate(labels):
        if lab[0] == "p":
            return i
    raise ValueError("no physical leg among {}".format(labels))


def _commit_split(state, cache, prob, roots, w_used, trunc_tol, max_bond, *,
                  graph=None, left_labels=None, expansion=0.0):
    """Split the merged ensemble across ``(u, v)`` and move the center to ``v``.

    ``graph`` is the topology to commit under (default: the incumbent ``state.graph``)
    and ``left_labels`` the merged legs assigned to node ``u`` (default: ``u``'s own
    legs — the ordinary two-site update). With a different assignment this **is** the
    adaptive reconnection commit: the same SVD machinery, a different bipartition, a new
    graph. The caller guarantees ``graph`` is consistent with the assignment (each side
    exactly one physical leg; branch legs wired to their new owner); nothing here can
    check the physics of that choice, only its bookkeeping.
    """
    u, v = prob.u, prob.v
    graph = state.graph if graph is None else graph
    labels = list(prob.labels)
    if left_labels is None:
        left_axes = tuple(range(prob.n_left))
    else:
        left_axes = tuple(sorted(labels.index(l) for l in left_labels))
    stacked = _stack_roots(roots, w_used)
    if expansion > 0.0:
        # The augmented ensemble: same left legs, extra alpha-scaled columns, ONE svd —
        # the single truncation path, degenerate-group rule included, is untouched.
        stacked = _concat_aux(stacked, _expansion_columns(prob, stacked),
                              float(expansion))
    u_iso, _, _, info = svd(stacked, left_axes, tol=trunc_tol, max_bond=max_bond)

    # node u: [bonds ascending, phys last]; the new bond is u_iso's last leg
    left_labs = [labels[i] for i in left_axes]
    perm = []
    for x in sorted(graph.neighbors(u)):
        perm.append(len(left_labs) if x == v else _branch_axis(left_labs, x))
    perm.append(_phys_axis(left_labs))
    state.tensors[u] = u_iso.transpose(perm)

    # new center tensors at v: project each root on the kept basis
    u_conj = u_iso.conj()
    rest_labs = [labels[i] for i in range(len(labels)) if i not in left_axes]
    centers = []
    for m in roots:
        c = tensordot(m, u_conj, axes=(list(left_axes), list(range(len(left_axes)))))
        perm = []
        for y in sorted(graph.neighbors(v)):
            perm.append(len(rest_labs) if y == u else _branch_axis(rest_labs, y))
        perm.append(_phys_axis(rest_labs))
        centers.append(c.transpose(perm))
    state.graph = graph
    state.centers = centers
    state.tensors[v] = None
    state.center = v
    cache.refresh(u, v)
    if expansion > 0.0:
        # ⚠ The svd's discarded weight belongs to the AUGMENTED density (ensemble plus
        # alpha-scaled expansion columns), which is not the quality number w_disc means
        # everywhere else. The ensemble's own is recovered exactly from the projections
        # just computed: the roots are unit vectors, so 1 - sum_r w_r ||U^+ m_r||^2.
        kept = sum(float(w) * float(c.norm()) ** 2 for c, w in zip(centers, w_used))
        info = TruncationInfo(bond_dim=info.bond_dim, n_discarded=info.n_discarded,
                              discarded_weight=max(0.0, 1.0 - kept),
                              smallest_kept=info.smallest_kept,
                              largest_discarded=info.largest_discarded)
    return info


def _expansion_columns(prob: _LocalProblem, stacked: BlockTensor) -> BlockTensor:
    """The subspace-expansion enrichment: the ``u``-side half of ``H_eff`` applied to the
    ensemble, operator channel left open.

    Contract the ``sqrt(w)``-stacked ensemble with node ``u``'s environments and W tensor
    only — the ``v`` side untouched — and leave the operator-channel leg toward ``v``
    open. The result spans, on the ``u``-side (row) legs, exactly the directions the
    Hamiltonian connects the current ensemble to: appending its columns to the
    truncation's input is the two-site, ensemble form of the subspace expansion of Hubig,
    McCulloch, Schollwoeck & Wolf, Phys. Rev. B 91, 155115 (2015), and — because the SVD
    of the augmented matrix diagonalizes ``rho + alpha^2 sum_b T_b rho T_b^+`` — it is at
    the same time the deterministic density-matrix perturbation of White, J. Chem. Phys.
    122, 084108 (2005), evaluated instead of sampled.

    Returned with the same leg order and spaces as ``stacked`` except the trailing
    auxiliary leg, which fuses the ensemble leg with the open operator channel.
    """
    u = prob.u
    t = _Lab(stacked, list(prob.labels) + [("r",)])
    for x, env in prob.envs_u:
        t = t.dot(env, [(("b", u, x), ("ket", x))])
    t = t.dot(prob.w_u, [(("op", x), ("op", x)) for x in prob.branches_u]
              + [(("p", u), ("pi",))])
    relabel = []
    for lab in t.labels:
        if lab[0] == "bra":
            relabel.append(("b", u, lab[1]))
        elif lab == ("po",):
            relabel.append(("p", u))
        else:
            relabel.append(lab)
    ordered = _Lab(t.t, relabel).to(list(prob.labels) + [("r",), ("op_out",)])
    fused, _ = fuse(ordered, (ordered.ndim - 2, ordered.ndim - 1))
    return fused.transpose(list(range(1, fused.ndim)) + [0])


def _concat_aux(a: BlockTensor, b: BlockTensor, scale: float) -> BlockTensor:
    """``[a | scale * b]`` along the trailing auxiliary leg; every other leg shared.

    The augmented ensemble the expansion feeds to the **one** truncation path: one
    :func:`svd` call on the result performs the perturbed weighted-density-matrix
    truncation with the degenerate-group discipline intact. The two operands must agree
    on everything but the auxiliary leg — asserted, because a mismatch here would be a
    sign or flux bookkeeping error upstream that the concatenation must not paper over.
    """
    ax = a.ndim - 1
    if b.ndim != a.ndim or a.spaces[:ax] != b.spaces[:ax] \
            or a.signs[:ax] != b.signs[:ax] or a.signs[ax] != b.signs[ax] \
            or a.charge != b.charge:
        raise ValueError("expansion columns do not share the ensemble's legs; this is a "
                         "sign or flux bookkeeping error upstream")
    sa, sb = a.spaces[ax], b.spaces[ax]

    def dim_of(space: Space, qn) -> int:
        return int(space.dims[space.qns.index(qn)]) if qn in space.qns else 0

    qns = sorted(set(sa.qns) | set(sb.qns))
    combined = Space([(qn, dim_of(sa, qn) + dim_of(sb, qn)) for qn in qns])
    sector_of = {qn: k for k, qn in enumerate(qns)}
    merged: Dict[tuple, np.ndarray] = {}
    for tensor, offset_fn, factor in ((a, lambda q: 0, 1.0),
                                      (b, lambda q: dim_of(sa, q), float(scale))):
        for row, block in zip(tensor.sectors, tensor.blocks):
            qn = tensor.spaces[ax].qns[int(row[ax])]
            key = tuple(int(i) for i in row[:ax]) + (sector_of[qn],)
            out = merged.get(key)
            if out is None:
                out = np.zeros(block.shape[:-1] + (int(combined.dims[sector_of[qn]]),),
                               dtype=np.complex128)
                merged[key] = out
            start = offset_fn(qn)
            out[..., start:start + block.shape[-1]] += factor * block
    rows = np.array(sorted(merged), dtype=np.int64).reshape(len(merged), a.ndim)
    blocks = [np.ascontiguousarray(merged[tuple(int(i) for i in row)]) for row in rows]
    return BlockTensor(a.spaces[:ax] + (combined,), a.signs, a.charge, rows, blocks)


def _stack_roots(roots: List[BlockTensor], weights: np.ndarray) -> BlockTensor:
    """The ensemble tensor: roots on an auxiliary charge-0 leg, scaled by ``sqrt(w)``.

    One SVD of this equals the weighted-density-matrix truncation of SA-DMRG (module
    docstring), with the degenerate-group rule applied to the *ensemble* spectrum.
    """
    t0 = roots[0]
    aux = Space([(t0.charge.zero_like(), len(roots))])
    spaces = t0.spaces + (aux,)
    signs = t0.signs + (1,)
    blocks = []
    for i in range(t0.nblocks):
        stacked = np.empty(t0.blocks[i].shape + (len(roots),), dtype=np.complex128)
        for r, (m, w) in enumerate(zip(roots, weights)):
            stacked[..., r] = np.sqrt(w) * m.blocks[i]
        blocks.append(stacked)
    rows = np.hstack([t0.sectors, np.zeros((t0.nblocks, 1), dtype=np.int64)])
    return BlockTensor(spaces, signs, t0.charge, rows, blocks)


def _boundary_sweep(ttno, state, cache, requested, n_elec, trunc_tol, max_bond,
                    davidson_tol, on_split, boundary_check):
    """One extra sweep solving ``boundary_check`` extra local roots at every bond: the
    State-average boundary diagnostic, sweep flavour.

    The reported gap is the **minimum over bonds** of the gap between root ``n_roots`` and
    root ``n_roots - 1``. One bond is not enough: an end bond's two-site space is too
    small to represent the state the average may be cutting (measured: a split Kramers
    partner invisible at an end bond, 339 cm^-1 instead of ~0, while a middle bond sees
    it) — the minimum over the tour is the honest local statement. Each extra root is
    variational, so the reported gap still bounds the true gap from above; a warning here
    is trustworthy, silence is necessary-not-sufficient (module docstring). The sweep
    re-derives the converged fixed point, so the state is unchanged apart from rounding.
    """
    n_roots = state.n_roots
    gaps = []
    for u, v in state.graph.sweep_schedule(state.center):
        _, _, _, gap = _update_bond(ttno, state, cache, u, v, requested, n_elec,
                                    trunc_tol, max_bond, davidson_tol, on_split,
                                    extra_roots=boundary_check)
        if gap is not None:
            gaps.append(gap)
    if not gaps:
        out.entry(log, "state-average boundary", "complete",
                  note="every local space spans the whole average")
        return None
    gap_cm = float(min(gaps)) * HARTREE_TO_CM
    out.entry(log, "state-average boundary gap", gap_cm, unit="cm^-1", fmt="{:.2f}",
              note="min over the tour; local, variational from above")
    if gap_cm < BOUNDARY_GAP_WARN_CM:
        log.warning("the state average ends %.2f cm^-1 below the next local root — the "
                    "averaged set may cut a degenerate manifold; change "
                    "the root count", gap_cm)
    return gap_cm


__all__ = ["TTNState", "EnvironmentCache", "SweepResult", "random_state", "solve_ttn",
           "state_gb", "state_to_dense", "BOUNDARY_GAP_WARN_CM"]
