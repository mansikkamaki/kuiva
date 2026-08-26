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

from ..ci.davidson import davidson
from ..props.multiplet import HARTREE_TO_CM
from ..rdm.rdm import DEFAULT_DEGENERACY_TOL, degenerate_blocks, state_average_weights
from ..util import output as out
from ..util import resources as res
from ..util import threads
from ..util.logging import get_logger
from ..util.timing import timer
from .block import (BlockTensor, QuantumNumber, Space, block_tensor_gb, qr, svd,
                    tensordot)
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
        densified once."""
        ia = [self.labels.index(a) for a, _ in pairs]
        ib = [other.labels.index(b) for _, b in pairs]
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

    layouts = []
    for u in range(graph.n_nodes):
        spaces, signs = [], []
        for x in sorted(graph.neighbors(u)):
            spaces.append(bond_spaces[(min(u, x), max(u, x))])
            signs.append(1 if (u == center or x != int(parent[u])) else -1)
        spaces.append(ttno.phys_space[u])
        signs.append(1)
        layouts.append((tuple(spaces), tuple(signs), charge if u == center else zero))
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
        t = t.dot(_w_lab(self.ttno, u, v),
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


# --- the local (two-site) problem ---------------------------------------------------------

class _LocalProblem(object):
    """One two-site eigenproblem on bond ``(u, v)`` (center at ``u``).

    The variational space is **all** symmetry-allowed blocks of the merged spaces — not
    just the blocks the incoming tensors happen to populate — so the template is built
    with :meth:`BlockTensor.zeros` and every vector is packed against it.
    """

    def __init__(self, ttno: TTNO, state: TTNState, cache: EnvironmentCache,
                 u: int, v: int):
        graph = state.graph
        self.u, self.v = u, v
        self.branches_u = [x for x in sorted(graph.neighbors(u)) if x != v]
        self.branches_v = [y for y in sorted(graph.neighbors(v)) if y != u]
        m0 = tensordot(state.centers[0], state.tensors[v],
                       axes=([_bond_axis(graph, u, v)], [_bond_axis(graph, v, u)]))
        self.labels = [("b", u, x) for x in self.branches_u] + [("p", u)] \
            + [("b", v, y) for y in self.branches_v] + [("p", v)]
        self.n_left = len(self.branches_u) + 1
        self.template = BlockTensor.zeros(m0.spaces, m0.signs, m0.charge)
        self.sizes = [b.size for b in self.template.blocks]
        self.dim = int(sum(self.sizes))
        self.envs_u = [(x, _Lab(cache.get(x, u), [("bra", x), ("op", x), ("ket", x)]))
                       for x in self.branches_u]
        self.envs_v = [(y, _Lab(cache.get(y, v), [("bra", y), ("op", y), ("ket", y)]))
                       for y in self.branches_v]
        self.w_u = _w_lab(ttno, u, v)
        self.w_v = _w_lab(ttno, v, u)

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

    def apply(self, vec: np.ndarray) -> np.ndarray:
        u, v = self.u, self.v
        t = _Lab(self.unpack(vec), list(self.labels))
        for x, env in self.envs_u:
            t = t.dot(env, [(("b", u, x), ("ket", x))])
        t = t.dot(self.w_u, [(("op", x), ("op", x)) for x in self.branches_u]
                  + [(("p", u), ("pi",))])
        for y, env in self.envs_v:
            t = t.dot(env, [(("b", v, y), ("ket", y))])
        t = t.dot(self.w_v, [(("op_out",), ("op_out",))]
                  + [(("op", y), ("op", y)) for y in self.branches_v]
                  + [(("p", v), ("pi",))])
        labels = []
        seen_po = 0
        for lab in t.labels:
            if lab[0] == "bra":
                x = lab[1]
                labels.append(("b", u, x) if x in self.branches_u else ("b", v, x))
            elif lab == ("po",):
                labels.append(("p", u) if seen_po == 0 else ("p", v))
                seen_po += 1
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

    ``report=False`` moves the sweep table to DEBUG — for a solver called once per
    CASSCF macro-iteration, whose driver already owns the INFO table (INFO is the
    output file, and sixty inner tables in it are noise, not output).
    """
    graph = state.graph
    n_roots = state.n_roots
    n_elec = state.charge.n if n_elec is None else int(n_elec)
    requested = np.full(n_roots, 1.0 / n_roots) if weights is None \
        else np.asarray(weights, dtype=float) / float(np.sum(weights))
    if requested.size != n_roots:
        raise ValueError("{} weights for {} roots".format(requested.size, n_roots))

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
            for u, v in graph.sweep_schedule(state.center):
                energies, w_used, info, _ = _update_bond(
                    ttno, state, cache, u, v, requested, n_elec, trunc_tol, max_bond,
                    davidson_tol, on_split)
                max_disc = max(max_disc, info.discarded_weight)
                max_dim = max(max_dim, info.bond_dim)
            e_sa = float(np.dot(w_used, energies))
            de = None if e_prev is None else e_sa - e_prev
            history.append(e_sa)
            table.row(sweep, e_sa, de, max_disc, max_dim, time.perf_counter() - t0)
            if de is not None and abs(de) < conv_tol:
                converged = True
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
                 davidson_tol, on_split, extra_roots=0):
    """One two-site update; with ``extra_roots`` it also returns the state-average boundary gap.

    ⚠ Extra roots are solved and *discarded* — they never enter the average or the
    truncation (the extra roots are discarded, never averaged over).
    """
    prob, energies, w_used, roots, gap = _solve_local(ttno, state, cache, u, v,
                                                      requested, davidson_tol,
                                                      extra_roots=extra_roots)
    info = _commit_split(state, cache, prob, roots, w_used, trunc_tol, max_bond)
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
    res.require("DMRG two-site problem ({}-{})".format(u, v),
                (n_roots + 2) * prob.dim * 16.0 / 1024.0 ** 3,
                note="{} packed vectors of {} elements".format(n_roots + 2, prob.dim),
                advice=["reduce max_bond: the local dimension scales as D^2 d^2"])

    n_solve = min(n_roots + int(extra_roots), prob.dim)
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
                  graph=None, left_labels=None):
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
    return info


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
