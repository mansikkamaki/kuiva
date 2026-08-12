"""NetworkGraph: the tree topology of the tensor network.

The network machinery is **tree-native from day one**: every routine here works on an
arbitrary tree, and the matrix-product chain is the path-graph special case rather than a
separate code path. The graph is the *single owner of topology* — sweep code asks it for
schedules and orientations and never assumes "left/right" — and it is **immutable**: an
adaptive topology change builds a new graph, which is what makes topology part of a
solver's ``space_key`` for free (chart changes are explicit objects, never mutations).

Nodes and contents
------------------
Nodes are integers ``0..n_nodes-1``; each node carries a (possibly empty) tuple of **orbital
labels** — the physical/spinor indices whose tensor lives at that node. A plain MPS has one
orbital per node; a branching TTNS node with no physical leg carries the empty tuple. Labels
must be disjoint across nodes: an orbital that lives at two sites is a bookkeeping error
nothing downstream could detect.

Directed bonds and environment keying
-------------------------------------
A directed bond ``(u, v)`` means "looking from ``u`` toward ``v``"; the environment
(renormalized operator block) attached to it summarizes the subtree on the ``u`` side of the
edge. Its cache key is the directed bond **plus** :meth:`NetworkGraph.subtree_hash` — a
digest of the orbital content of that subtree — so a topology change invalidates exactly the
environments whose subtree content changed and nothing else. This keying is also what keeps
restart and event-gating bookkeeping honest. The digest is BLAKE2b over the
sorted labels (the ``array_key`` convention of :mod:`kuiva.mcscf.adaptive`): stable across
processes, safe to write to a checkpoint or a log.

Sweep schedules
---------------
:meth:`NetworkGraph.sweep_schedule` returns the two-site update sequence as directed bonds
``(u, v)``: update the merged tensor on edge ``{u, v}``, then move the canonical center from
``u`` to ``v``. The schedule is the depth-first Euler tour from the chosen center (children
in ascending node order — deterministic): every edge is visited exactly once in each
direction, consecutive updates share a node (the center walks, never jumps), and the walk
returns to the center. On a path with the center at one end this reduces to the conventional
left-to-right-and-back DMRG sweep.

Everything here is orchestration and stays Python: pure bookkeeping on graphs
with at most a few hundred nodes, no kernel content.

References
----------
* Tree tensor networks: Y.-Y. Shi, L.-M. Duan, G. Vidal, Phys. Rev. A 74, 022320 (2006),
  doi:10.1103/PhysRevA.74.022320; V. Murg, F. Verstraete, O. Legeza, R. M. Noack, Phys.
  Rev. B 82, 205105 (2010), doi:10.1103/PhysRevB.82.205105.
* Tree sweeps and canonical centers for ab initio DMRG on trees: N. Nakatani, G. K.-L. Chan,
  J. Chem. Phys. 138, 134113 (2013), doi:10.1063/1.4798639; K. Gunst, F. Verstraete,
  S. Wouters, O. Legeza, D. Van Neck (T3NS), J. Chem. Theory Comput. 14, 2026 (2018),
  doi:10.1021/acs.jctc.8b00098.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util.logging import get_logger

log = get_logger(__name__)


class NetworkGraph(object):
    """An immutable labelled tree (module docstring). Validated on construction."""

    __slots__ = ("_n", "_edges", "_adjacency", "_contents")

    def __init__(self, n_nodes: int, edges: Sequence[Tuple[int, int]],
                 contents: Optional[Sequence[Sequence[int]]] = None):
        n = int(n_nodes)
        if n < 1:
            raise ValueError("a network needs at least one node")
        canon = []
        seen = set()
        for u, v in edges:
            u, v = int(u), int(v)
            if u == v:
                raise ValueError("self-loop at node {}".format(u))
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("edge ({}, {}) out of range for {} nodes".format(u, v, n))
            key = (min(u, v), max(u, v))
            if key in seen:
                raise ValueError("duplicate edge {}".format(key))
            seen.add(key)
            canon.append(key)
        if len(canon) != n - 1:
            raise ValueError("a tree on {} nodes has {} edges, got {}".format(n, n - 1,
                                                                              len(canon)))
        adjacency: List[List[int]] = [[] for _ in range(n)]
        for u, v in canon:
            adjacency[u].append(v)
            adjacency[v].append(u)

        # connectivity: |E| = n-1 plus connected <=> tree
        if n > 1:
            reached = self._component(0, None, adjacency)
            if len(reached) != n:
                raise ValueError("the edge list is not connected: node 0 reaches only {} of "
                                 "{} nodes".format(len(reached), n))

        if contents is None:
            contents = [(i,) for i in range(n)]
        if len(contents) != n:
            raise ValueError("{} content tuples for {} nodes".format(len(contents), n))
        cleaned = tuple(tuple(int(x) for x in c) for c in contents)
        flat = [x for c in cleaned for x in c]
        if len(flat) != len(set(flat)):
            raise ValueError("orbital labels are not disjoint across nodes")

        self._n = n
        self._edges = tuple(sorted(canon))
        self._adjacency = tuple(tuple(sorted(a)) for a in adjacency)
        self._contents = cleaned

    @classmethod
    def path(cls, n_nodes: int,
             contents: Optional[Sequence[Sequence[int]]] = None) -> "NetworkGraph":
        """The path graph ``0 - 1 - ... - n-1`` (the MPS chain)."""
        return cls(n_nodes, [(i, i + 1) for i in range(int(n_nodes) - 1)], contents)

    @staticmethod
    def _component(start: int, blocked: Optional[int],
                   adjacency: Sequence[Sequence[int]]) -> List[int]:
        """Nodes reachable from ``start`` without stepping onto ``blocked``."""
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for w in adjacency[u]:
                if w != blocked and w not in seen:
                    seen.add(w)
                    stack.append(w)
        return sorted(seen)

    # --- basic queries --------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self._n

    @property
    def edges(self) -> Tuple[Tuple[int, int], ...]:
        """Undirected bonds as sorted ``(min, max)`` pairs, sorted."""
        return self._edges

    @property
    def contents(self) -> Tuple[Tuple[int, ...], ...]:
        return self._contents

    def neighbors(self, u: int) -> Tuple[int, ...]:
        return self._adjacency[int(u)]

    def degree(self, u: int) -> int:
        return len(self._adjacency[int(u)])

    def leaves(self) -> Tuple[int, ...]:
        """Degree-1 nodes (for a single-node graph, the node itself)."""
        if self._n == 1:
            return (0,)
        return tuple(u for u in range(self._n) if self.degree(u) == 1)

    def directed_bonds(self) -> Tuple[Tuple[int, int], ...]:
        """Every bond in both directions."""
        return tuple((u, v) for u, v in self._edges) \
            + tuple((v, u) for u, v in self._edges)

    def _check_edge(self, u: int, v: int) -> Tuple[int, int]:
        u, v = int(u), int(v)
        if (min(u, v), max(u, v)) not in set(self._edges):
            raise ValueError("({}, {}) is not an edge of this tree".format(u, v))
        return u, v

    # --- rooted orientation ---------------------------------------------------------------

    def parents(self, center: int) -> Tuple[np.ndarray, np.ndarray]:
        """Rooted orientation from ``center``: ``(parent, preorder)``.

        ``parent[center] == -1``; ``preorder`` is the deterministic DFS order (children
        ascending) with each node after its parent — reverse it for a children-first
        (environment-building) order.
        """
        center = int(center)
        if not (0 <= center < self._n):
            raise ValueError("center {} out of range".format(center))
        parent = np.full(self._n, -1, dtype=np.int64)
        preorder = np.empty(self._n, dtype=np.int64)
        stack = [center]
        seen = {center}
        k = 0
        while stack:
            u = stack.pop()
            preorder[k] = u
            k += 1
            # push descending so ascending neighbors pop first (deterministic DFS)
            for w in reversed(self._adjacency[u]):
                if w not in seen:
                    seen.add(w)
                    parent[w] = u
                    stack.append(w)
        return parent, preorder

    # --- sweep schedule ---------------------------------------------------------------------

    def sweep_schedule(self, center: int) -> List[Tuple[int, int]]:
        """The two-site Euler-tour schedule from ``center`` (module docstring).

        Each directed bond appears exactly once; consecutive bonds share a node; the tour
        starts and ends at ``center``. Empty for a single-node graph.
        """
        center = int(center)
        if not (0 <= center < self._n):
            raise ValueError("center {} out of range".format(center))
        schedule: List[Tuple[int, int]] = []

        def _tour(u: int, parent: int) -> None:
            for w in self._adjacency[u]:
                if w != parent:
                    schedule.append((u, w))
                    _tour(w, u)
                    schedule.append((w, u))

        _tour(center, -1)
        return schedule

    # --- subtrees and environment keys ------------------------------------------------------

    def subtree_nodes(self, u: int, v: int) -> Tuple[int, ...]:
        """Nodes on the ``u`` side of edge ``{u, v}`` (sorted). ``(u,v)`` must be an edge."""
        u, v = self._check_edge(u, v)
        return tuple(self._component(u, v, self._adjacency))

    def subtree_contents(self, u: int, v: int) -> Tuple[int, ...]:
        """Sorted orbital labels on the ``u`` side of edge ``{u, v}``."""
        return tuple(sorted(x for w in self.subtree_nodes(u, v)
                            for x in self._contents[w]))

    def subtree_hash(self, u: int, v: int) -> str:
        """Digest of the orbital content behind directed bond ``(u, v)`` — the environment
        cache key ingredient (module docstring). Depends on the *content set only*: stable
        under internal renumbering of nodes, changed by any orbital entering or leaving the
        subtree."""
        labels = np.asarray(self.subtree_contents(u, v), dtype=np.int64)
        return hashlib.blake2b(labels.tobytes(), digest_size=16).hexdigest()

    # --- identity ---------------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        return (isinstance(other, NetworkGraph) and self._n == other._n
                and self._edges == other._edges and self._contents == other._contents)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._n, self._edges, self._contents))

    def __repr__(self) -> str:
        return "NetworkGraph(n_nodes={}, n_edges={})".format(self._n, len(self._edges))


__all__ = ["NetworkGraph"]
