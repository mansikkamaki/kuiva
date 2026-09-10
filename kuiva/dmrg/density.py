"""Reduced density matrices by direct network contraction.

The network-side half of the design's "1–4 RDMs: CI-exact (small CAS) and MPS-based direct
contraction". The CI-exact path (:mod:`kuiva.rdm.rdm`) is untouched; both return the same
objects in the same convention, so either plugs into the shared orbital optimizer contract:

    gamma_pq            = <a+_p a_q>
    Gamma_pqrs          = <a+_p a+_r a_s a_q>
    Gamma3[p,q,r,s,t,u] = <a+_p a+_r a+_t a_u a_s a_q>
    Gamma4[...]         = the same pattern, one pair further

(pairs ``(p_i, q_i)`` interleaved, creations as written, annihilations reversed — exactly
the ``Gamma_pqrs = <E_pq E_rs> - delta_qr gamma_ps`` object of :mod:`kuiva.rdm.rdm`).

⚠ The module is named ``density``, not ``rdm``, so ``kuiva.dmrg.rdm`` never exists to
shadow ``kuiva.rdm`` in a relative import — the same never-shadow-a-package rule as ``kuiva/io``.

Two routes, deliberately different
----------------------------------
**Ranks 1–2, the production path** (:func:`network_rdms`): one backward pass over the
network builds every node's environments, and each elementary operator's expectation is
read at its coefficient-attachment slot of the node's operator environment
``G_u = dE/dW_u`` — the contraction of bra, ket and every *other* node's W tensor around
node ``u`` (:meth:`kuiva.dmrg.ttno.TTNOTemplate.rdms_from_slot_values`). This costs about
two sweeps' worth of environment builds per call, independent of ``n^4``, which is what a
CASSCF macro-iteration needs. Correctness rests on the TTNO compiler's label-completeness
(a channel's operator content is uniquely determined by its label, so the read is
uncontaminated by other terms' coefficients — see the template's docstring), and is
asserted by the energy-closure and CI-parity tests rather than trusted.

⚠ **``G_u`` is never formed.** It has every leg of ``W_u`` open — two operator bond
dimensions times the local dimension squared, dense — and on a five-mode node of a
20-spinor space that was 5.75 GB for one node while the sweep's largest array was 8 MB; the
RDM phase, not the sweep, bounded the reachable node partition. What the template reads
is a sparse subset of it: the slots, one ``(channel tuple, i, j)`` per (term, matrix
element). :func:`slot_values` evaluates exactly those, grouped by the channel values on
all but one operator leg: the environments of the grouped legs are *sliced* at their
channel (a bra/ket matrix each), contracted into the ket, closed by the bra, and the one
remaining environment opens its operator leg last — ``D^k d r``, ``D^2 d^2`` and
``w d^2`` per group, never ``w^2 d^2``. The dense ``G_u`` survives as the validation
oracle (:func:`node_environments`, :meth:`~kuiva.dmrg.ttno.TTNOTemplate.rdms_from_environments`)
the production route is pinned against.

**Ranks 1–4, the direct-contraction path** (:func:`network_rdm`): the Gram matrix
``M[A, B] = <chi_A | chi_B>`` of *annihilated states* ``chi_A = a_{A1} a_{A2} .. |psi>``.
A Jordan–Wigner string is a product of local matrices, so applying it to a TTN changes
node tensors without touching any bond — every ``chi_A`` shares the state's bond
dimensions, and each overlap is one tree contraction with identity shortcuts over
untouched subtrees. Exact, cumulant-free, and how the 3-/4-RDMs are produced.
⚠ **Scope-limited by design**: the cost is ``C(n, k)^2`` tree contractions
and the result is the dense ``n^(2k)`` array — 6.4 GB at 12 active spinors for the 4-RDM,
22 GB at 14 (:func:`kuiva.util.resources.rdm_gb`), refused by the resource budget long before the
contraction count matters. For genuinely multi-site actives a stored 4-RDM is impossible
and SC-NEVPT2 needs a contraction-on-demand formulation that is its own future plan; this
module's deliverable is parity with the conventional-CI path on small actives.

⚠ Odd-rank strings carry a Jordan–Wigner tail (:func:`annihilation_term`): the compiler's
term machinery truncates a string's support at its lowest operator because for the *even*
strings of a Hamiltonian the parity factors below cancel in pairs. For an odd string they
do not, and both sides of every overlap must carry the explicit ``Z`` tail down to mode 0
or the Gram matrix is wrong by exactly a fermionic sign pattern that no Hermiticity or
trace check can see.

State averaging follows exactly :class:`kuiva.rdm.rdm.RDMBuilder` does: imposed
where the RDMs are built, through the same :func:`kuiva.rdm.rdm.state_average_weights`
gate, with the same escape hatch and the same refusal semantics.

Everything here is orchestration: the arithmetic is
:func:`kuiva.dmrg.block.tensordot`, and the per-term extraction is two ``bincount`` calls
in the template.

References
----------
* One- and two-particle density matrices in DMRG sweeps: S. R. White, Phys. Rev. B 48,
  10345 (1993), doi:10.1103/PhysRevB.48.10345; U. Schollwoeck, Ann. Phys. 326, 96 (2011),
  doi:10.1016/j.aop.2010.09.012.
* Higher-order RDMs from matrix-product states for multireference perturbation theory
  (the chosen no-cumulant route): Y. Kurashige, T. Yanai, J. Chem. Phys. 135, 094104
  (2011), doi:10.1063/1.3629454; S. Guo, M. A. Watson, W. Hu, Q. Sun, G. K.-L. Chan,
  J. Chem. Theory Comput. 12, 1583 (2016), doi:10.1021/acs.jctc.5b01225.
* Jordan-Wigner transformation: P. Jordan, E. Wigner, Z. Phys. 47, 631 (1928),
  doi:10.1007/BF01331938.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..rdm.rdm import DEFAULT_DEGENERACY_TOL, state_average_weights
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from .block import BlockShape, BlockTensor, Space, _row_keys, tensordot
from .sweep import (EnvironmentCache, TTNState, _Lab, _stack_roots, _w_lab, choose_fold,
                    renormalize)
from .ttno import (FERMION_MODE, ProductTerm, TTNO, TTNOTemplate, _I2, _Z,
                   _charge_shift, fermion_term)

log = get_logger(__name__)


# --- per-node operator environments (G_u = dE/dW_u) ----------------------------------------

def _down_message(ttno: TTNO, state: TTNState, cache: EnvironmentCache,
                  down: Dict[Tuple[int, int], BlockTensor], center_side: Dict[int, int],
                  stacked: BlockTensor, v: int, u: int) -> BlockTensor:
    """The center-side message ``env(v -> u)``: the whole network beyond ``v`` as seen
    from ``u``, with the ensemble roots (weighted) summed in at the center.

    The mirror image of :meth:`EnvironmentCache._build`, with two substitutions: at the
    center the node tensor is the ``sqrt(w)``-stacked root ensemble (contracting the
    auxiliary root legs of bra and ket against each other is what performs
    ``sum_r w_r <psi_r|..|psi_r>``), and the input on ``v``'s own center side is a
    previously built down message instead of a subtree environment.
    """
    graph = state.graph
    nbrs = sorted(graph.neighbors(v))
    at_center = v == state.center
    ket_t = stacked if at_center else state.tensors[v]
    labels = [("b", v, x) for x in nbrs] + [("p", v)] + ([("r",)] if at_center else [])
    bra_labels = [("cb", v, x) for x in nbrs] + [("cp",)] + ([("cr",)] if at_center else [])
    ket = _Lab(ket_t, labels)
    bra = _Lab(ket_t.conj(), bra_labels)
    envs = {x: _Lab(down[(x, v)] if (not at_center and x == center_side[v])
                    else cache.get(x, v), [("bra", x), ("op", x), ("ket", x)])
            for x in nbrs if x != u}
    w = _w_lab(ttno, v, u)
    root_pair = [(("r",), ("cr",))] if at_center else []
    # the sweep's renormalization chain and its fold search: one open operator leg where
    # the shape allows it, the fold decided over structure (see renormalize)
    pre = choose_fold(ket, envs, w, bra, v, root_pair)
    t = renormalize(ket, envs, w, bra, v, pre, root_pair)
    return t.to([("cb", v, u), ("op_out",), ("b", v, u)])


def node_environments(ttno: TTNO, state: TTNState,
                      weights: Sequence[float]) -> List[BlockTensor]:
    """``G_u = dE/dW_u`` for every node: bra, ket and every other node's W contracted.

    ⚠ The validation oracle, not the production path (module docstring): it forms every
    node's dense ``G_u``, which :func:`slot_values` exists to avoid, and it is what the
    slot extraction is pinned against.

    Legs in W-tensor order ``[parent-op, child-ops ascending, phys-bra, phys-ket]`` over
    the same spaces, so :meth:`~kuiva.dmrg.ttno.TTNOTemplate.expectations` can read
    entries positionally. The root's dim-1 completed channel is kept as an explicit leg.
    The identity ``sum_entries W_u[e] G_u[e] = <H>`` holds for **every** node — the test
    the whole construction is pinned by.

    ⚠ The values read from ``G_u`` at label-determined channels are independent of the
    coefficients the ``ttno`` was filled with (template docstring) — any fill on the same
    topology serves, and a test asserts that invariance rather than leaving it as an
    argument.
    """
    graph = state.graph
    w = np.asarray(weights, dtype=float)
    stacked = _stack_roots(state.centers, w / float(np.sum(w)))
    # Deliberately unpaged: this phase materializes essentially the whole environment set
    # at once (every neighbor of every node), which is the one access pattern paging cannot
    # help — an LRU under that load would thrash the scratch disk for no resident saving.
    # It is also one-shot and released below, so its residency is a transient of this call.
    cache = EnvironmentCache(ttno, state)
    parent, preorder = graph.parents(state.center)
    center_side = {int(x): int(parent[x]) for x in preorder[1:]}

    down: Dict[Tuple[int, int], BlockTensor] = {}
    for x in [int(i) for i in preorder[1:]]:
        v = center_side[x]
        down[(v, x)] = _down_message(ttno, state, cache, down, center_side, stacked,
                                     v, x)

    envs: List[BlockTensor] = []
    held = 0
    with timer("network RDM environments"):
        for u in range(graph.n_nodes):
            nbrs = sorted(graph.neighbors(u))
            at_center = u == state.center
            ket = stacked if at_center else state.tensors[u]
            env_of = {x: (down[(x, u)] if (not at_center and x == center_side[u])
                          else cache.get(x, u)) for x in nbrs}
            # ⚠ Sized through the same chain before it is built, per node: the transient
            # of this contraction is what killed a 20-spinor run at D = 8 with a 0.7 GB
            # plan, and the finished G tensors stay resident until every node is done.
            shape, steps = _node_environment(
                u, BlockShape.of(ket), nbrs,
                {x: BlockShape.of(e) for x, e in env_of.items()}, at_center, ttno,
                sizes=[])
            transient = max(a + b for a, b in zip([ket.nbytes] + steps[:-1], steps)) \
                if steps else 0
            res.require("network RDM environment (node {})".format(u),
                        (held + shape.nbytes + transient) / 1024.0 ** 3,
                        note="dE/dW_u over {} operator leg(s); {:.3f} GB held by the "
                             "nodes before it".format(len(nbrs), held / 1024.0 ** 3),
                        advice=["a finer node partition: the environment is dense in "
                                "the node's local dimension squared",
                                "reduce max_bond: the contraction's intermediate carries "
                                "two bond legs and one operator leg"])
            g, _ = _node_environment(u, ket, nbrs, env_of, at_center, ttno)
            held += g.nbytes
            envs.append(g)
    cache.release_all()
    return envs


def _node_environment(u: int, ket, nbrs: Sequence[int], env_of: Dict[int, object],
                      at_center: bool, ttno: TTNO, sizes: Optional[List[int]] = None):
    """``G_u`` from the node's ket, its neighbours' environments and its own bra — over
    data (:class:`~kuiva.dmrg.block.BlockTensor`) or over structure
    (:class:`~kuiva.dmrg.block.BlockShape`), one description of the chain for both.

    ⚠ **The order is the memory.** Contracting every neighbour's environment before the
    bra leaves every operator leg open at once on top of the bond legs — ``D^2 d w^2``
    for an interior node of a chain, times the root count at the center, which is what
    the sweep's own reorder had just removed and what killed a 20-spinor ladder at D = 8
    with a 0.7 GB plan. Here the **bra is contracted right after the first environment**,
    closing that bra leg and the ensemble legs, and every further environment closes a
    bra/ket bond pair as it opens its operator leg: the intermediates are ``D^2 d w r``,
    ``D^2 d^2 w`` and the output itself, ``w^2 d^2`` — which is dense in the node's local
    dimension squared by definition (``dE/dW_u`` has every leg of ``W_u`` open) and is
    the term the plan now carries. ``sizes`` collects the intermediates' byte counts.
    """
    labels = [("b", u, x) for x in nbrs] + [("p", u)] + ([("r",)] if at_center else [])
    t = _Lab(ket, labels)
    bra_labels = [("cb", u, x) for x in nbrs] + [("cp",)] + ([("cr",)] if at_center else [])
    bra = _Lab(ket.conj(), bra_labels)
    root_pair = [(("r",), ("cr",))] if at_center else []
    if not nbrs:                                   # a one-node network: ket against bra
        t = t.dot(bra, root_pair)
        if sizes is not None:
            sizes.append(t.t.nbytes)
    for i, x in enumerate(nbrs):
        env = _Lab(env_of[x], [("bra", x), ("op", x), ("ket", x)])
        if i == 0:
            t = t.dot(env, [(("b", u, x), ("ket", x))])
            if sizes is not None:
                sizes.append(t.t.nbytes)
            t = t.dot(bra, [(("bra", x), ("cb", u, x))] + root_pair)
        else:
            t = t.dot(env, [(("b", u, x), ("ket", x)), (("cb", u, x), ("bra", x))])
        if sizes is not None:
            sizes.append(t.t.nbytes)
    order = []
    if u != ttno.root:
        order.append(("op", int(ttno.parent[u])))
    order += [("op", c) for c in ttno.children[u]]
    order += [("cp",), ("p", u)]
    g = t.to(order)
    if u == ttno.root:
        g = _prepend_root_channel(g, ttno)
    return g, (sizes if sizes is not None else [])


def _stacked_shape(center: BlockShape, n_roots: int) -> BlockShape:
    """The structure :func:`~kuiva.dmrg.sweep._stack_roots` gives ``n_roots`` centers."""
    aux = Space([(center.charge.zero_like(), int(n_roots))])
    rows = np.hstack([center.sectors, np.zeros((center.nblocks, 1), dtype=np.int64)])
    return BlockShape(center.spaces + (aux,), center.signs + (1,), center.charge, rows)


def node_environments_gb(ttno: TTNO, state: TTNState, n_roots: int) -> Tuple[float, float]:
    """``(resident_gb, transient_gb)`` of :func:`node_environments`, from structure alone.

    The resident part is every node's ``G_u`` together (they are all held until the last
    is built); the transient is the largest step of any node's chain, an intermediate and
    its input live together. Environments come from :class:`~kuiva.dmrg.sweep.ShapeEnvironments`
    on the centre-filled shape state, which stands in for the down messages the real
    contraction uses on the centre side — same spaces, the same sector sets up to the
    centre charge, and a forecast rather than the exact per-node ``require`` above.
    """
    from .sweep import ShapeEnvironments, shape_state

    shapes = shape_state(state, fill_center=True)
    cache = ShapeEnvironments(ttno, shapes)
    graph = state.graph
    resident = 0
    transient = 0
    for u in range(graph.n_nodes):
        nbrs = sorted(graph.neighbors(u))
        at_center = u == state.center
        ket = _stacked_shape(shapes.tensors[u], n_roots) if at_center else shapes.tensors[u]
        env_of = {x: cache.get(x, u) for x in nbrs}
        g, steps = _node_environment(u, ket, nbrs, env_of, at_center, ttno, sizes=[])
        resident += g.nbytes
        if steps:
            transient = max(transient, max(
                a + b for a, b in zip([ket.nbytes] + steps[:-1], steps)))
    return resident / 1024.0 ** 3, transient / 1024.0 ** 3


# --- slot extraction: G_u at the template's slots, never the whole of it --------------------

def _take_op(env, sector: int, offset: int):
    """The environment sliced at one operator channel: legs ``[bra, ket]``, the operator
    leg's quantum number folded into the charge — over data or over structure."""
    spaces, signs = env.spaces, env.signs
    qn = spaces[1].qns[int(sector)]
    charge = env.charge - qn if signs[1] > 0 else env.charge + qn
    keep = env.sectors[:, 1] == int(sector)
    rows = np.ascontiguousarray(env.sectors[keep][:, [0, 2]])
    out_spaces = (spaces[0], spaces[2])
    out_signs = (signs[0], signs[2])
    if isinstance(env, BlockShape):
        return BlockShape(out_spaces, out_signs, charge, rows)
    blocks = [np.ascontiguousarray(b[:, int(offset), :])
              for b, k in zip(env.blocks, keep) if k]
    # correct by construction (the rows keep their order once the middle leg is fixed,
    # and the flux moved into the charge): the validating constructor's per-block Python
    # arithmetic was most of a slot group's cost on a fat node
    keys = _row_keys(rows, [sp.nsectors for sp in out_spaces])
    return BlockTensor._trusted(out_spaces, out_signs, charge, rows, keys, blocks)


def _leg_positions(ttno: TTNO, u: int, nbrs: Sequence[int]) -> List[int]:
    """W-leg position of every neighbour's operator leg: the parent's is 0, child ``j``'s
    is ``1 + j``; the root's leg 0 is its dim-1 completed channel and belongs to no
    neighbour."""
    out = []
    for x in nbrs:
        if u != ttno.root and x == int(ttno.parent[u]):
            out.append(0)
        else:
            out.append(1 + ttno.children[u].index(x))
    return out


def _slot_group_chain(u: int, ket, nbrs: Sequence[int], env_of: Dict[int, object],
                      at_center: bool, open_leg: Optional[int],
                      prefix: Sequence[Tuple[int, int, int]],
                      sizes: Optional[List[int]] = None):
    """The contraction that yields ``G_u`` restricted to one channel group: every prefix
    leg ``(x, sector, offset)`` sliced, the bra closed early, the open leg's environment
    last — over data (:class:`~kuiva.dmrg.block.BlockTensor`) or over structure
    (:class:`~kuiva.dmrg.block.BlockShape`), one description of the chain for both.

    Returns the labelled result with legs ``[op(open), cp, p]`` (``[cp, p]`` for a node
    with no neighbours); ``sizes`` collects every intermediate's byte count.
    """
    labels = [("b", u, x) for x in nbrs] + [("p", u)] + ([("r",)] if at_center else [])
    t = _Lab(ket, labels)
    bra_labels = [("cb", u, x) for x in nbrs] + [("cp",)] + ([("cr",)] if at_center else [])
    bra = _Lab(ket.conj(), bra_labels)
    root_pair = [(("r",), ("cr",))] if at_center else []
    for x, sec, off in prefix:
        env = _Lab(_take_op(env_of[x], sec, off), [("bra", x), ("ket", x)])
        t = t.dot(env, [(("b", u, x), ("ket", x))])
        if sizes is not None:
            sizes.append(t.t.nbytes)
    early = [(("bra", x), ("cb", u, x)) for x, _, _ in prefix] + root_pair
    if open_leg is None:
        t = t.dot(bra, early)
        if sizes is not None:
            sizes.append(t.t.nbytes)
        return t
    env = _Lab(env_of[open_leg], [("bra", open_leg), ("op", open_leg), ("ket", open_leg)])
    if early:
        t = t.dot(bra, early)
        if sizes is not None:
            sizes.append(t.t.nbytes)
        t = t.dot(env, [(("b", u, open_leg), ("ket", open_leg)),
                        (("cb", u, open_leg), ("bra", open_leg))])
    else:
        t = t.dot(env, [(("b", u, open_leg), ("ket", open_leg))])
        if sizes is not None:
            sizes.append(t.t.nbytes)
        t = t.dot(bra, [(("bra", open_leg), ("cb", u, open_leg))])
    if sizes is not None:
        sizes.append(t.t.nbytes)
    return t


def _chain_peak(ket_bytes: int, steps: Sequence[int]) -> int:
    return max(a + b for a, b in zip([ket_bytes] + list(steps[:-1]), steps)) if steps else 0


def _composite(cols: np.ndarray, widths: Sequence[int]) -> np.ndarray:
    """One int64 key per row of ``cols`` (mixed radix over ``widths``) — the row-wise
    unique the grouping needs, at the cost of a one-dimensional sort instead of a
    structured one (measured 30x on half a million slots)."""
    key = np.zeros(cols.shape[0], dtype=np.int64)
    for j, w in enumerate(widths):
        key = key * int(w) + cols[:, j]
    return key


class _SlotGroups(object):
    """The channel grouping of one node's slots: which operator leg stays open, and the
    slots sorted by their prefix-channel group (:func:`_node_slot_values` runs one chain
    per group, over data or over structure, from this one record)."""

    __slots__ = ("open_leg", "pos", "prefix_legs", "order", "bounds", "widths")

    def __init__(self, ttno: TTNO, u: int, nbrs: Sequence[int], rows: np.ndarray,
                 index: np.ndarray):
        self.pos = _leg_positions(ttno, u, nbrs)
        n_slots = rows.shape[0]
        channel = np.zeros((n_slots, len(nbrs)), dtype=np.int64)
        widths = []
        for k, (x, pk) in enumerate(zip(nbrs, self.pos)):
            space = ttno.bond_space[u] if pk == 0 \
                else ttno.bond_space[ttno.children[u][pk - 1]]
            channel[:, k] = space.offsets[rows[:, pk]] + index[:, pk]
            widths.append(space.total_dim)
        self.widths = widths
        # the open leg: fewest groups times its width, the groups being the distinct
        # channel tuples on the other legs — decided over the slots actually read
        best = None
        for k, x in enumerate(nbrs):
            others = [j for j in range(len(nbrs)) if j != k]
            n_groups = np.unique(_composite(channel[:, others],
                                            [widths[j] for j in others])).size \
                if others else 1
            cost = (n_groups * widths[k], x)
            if best is None or cost < best[0]:
                best = (cost, x)
        self.open_leg = None if best is None else best[1]
        self.prefix_legs = [(k, x) for k, x in enumerate(nbrs) if x != self.open_leg]
        if self.prefix_legs:
            pre = [k for k, _ in self.prefix_legs]
            key = _composite(channel[:, pre], [widths[k] for k in pre])
            self.order = np.argsort(key, kind="stable")
            sorted_key = key[self.order]
            starts = np.nonzero(np.concatenate([[True], sorted_key[1:] != sorted_key[:-1]]))[0]
            self.bounds = np.concatenate([starts, [n_slots]])
        else:
            self.order = np.arange(n_slots, dtype=np.int64)
            self.bounds = np.array([0, n_slots], dtype=np.int64)

    @property
    def n_groups(self) -> int:
        return int(self.bounds.size - 1)

    def members(self, g: int) -> np.ndarray:
        return self.order[int(self.bounds[g]):int(self.bounds[g + 1])]


def _node_slot_values(u: int, ket, nbrs: Sequence[int], env_of: Dict[int, object],
                      at_center: bool, ttno: TTNO, rows: np.ndarray, index: np.ndarray,
                      groups: _SlotGroups,
                      sizes: Optional[Dict[tuple, List[int]]] = None) -> np.ndarray:
    """``G_u`` at the slots ``(rows, index)`` — W-leg order ``[parent-op, child-ops,
    phys-bra, phys-ket]`` — by :func:`_slot_group_chain` per channel group.

    Over structure (``BlockShape`` operands) it runs every distinct prefix *sector*
    tuple once and fills ``sizes`` with each chain's intermediates, returning nothing
    numerical; over data it returns one complex value per slot.
    """
    structural = isinstance(ket, BlockShape)
    open_leg, pos = groups.open_leg, groups.pos
    n_slots = rows.shape[0]
    values = np.zeros(n_slots, dtype=np.complex128)
    if open_leg is None:
        read_cols = [rows.shape[1] - 2, rows.shape[1] - 1]
    else:
        po = pos[nbrs.index(open_leg)]
        read_cols = [po, rows.shape[1] - 2, rows.shape[1] - 1]
    seen_sector_tuples = set()
    for g in range(groups.n_groups):
        members = groups.members(g)
        first = int(members[0])
        prefix = [(x, int(rows[first, pos[k]]), int(index[first, pos[k]]))
                  for k, x in groups.prefix_legs]
        if structural:
            key = tuple((x, sec) for x, sec, _ in prefix)
            if key in seen_sector_tuples:
                continue
            seen_sector_tuples.add(key)
            steps: List[int] = []
            _slot_group_chain(u, ket, nbrs, env_of, at_center, open_leg, prefix, sizes=steps)
            sizes[key] = steps
            continue
        t = _slot_group_chain(u, ket, nbrs, env_of, at_center, open_leg, prefix)
        m = t.to([("cp",), ("p", u)] if open_leg is None
                 else [("op", open_leg), ("cp",), ("p", u)])
        sub_rows = rows[members][:, read_cols]
        sub_idx = index[members][:, read_cols]
        keys = _row_keys(sub_rows, [sp.nsectors for sp in m.spaces])
        # one sort of the group's slots by block, then one gather per block
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.nonzero(np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]]))[0]
        bounds = np.concatenate([starts, [keys.size]])
        for b in range(starts.size):
            sel = order[int(bounds[b]):int(bounds[b + 1])]
            blk = m.find(sub_rows[sel[0]])
            if blk is None:
                continue
            values[members[sel]] = blk[tuple(sub_idx[sel].T)]
    return values


def _node_slot_transient(u: int, ket_shape: BlockShape, nbrs: Sequence[int],
                         env_shapes: Dict[int, BlockShape], at_center: bool, ttno: TTNO,
                         rows: np.ndarray, index: np.ndarray, groups: _SlotGroups) -> int:
    """Largest step [bytes] any group's chain at node ``u`` holds, over structure."""
    sizes: Dict[tuple, List[int]] = {}
    _node_slot_values(u, ket_shape, nbrs, env_shapes, at_center, ttno, rows, index, groups,
                      sizes=sizes)
    return max((_chain_peak(ket_shape.nbytes, steps) for steps in sizes.values()),
               default=0)


def slot_values(template: TTNOTemplate, ttno: TTNO, state: TTNState,
                weights: Sequence[float]) -> List[np.ndarray]:
    """The value of ``G_u`` at every slot the template reads, node by node — the
    production RDM extraction (module docstring), one array per
    :attr:`~kuiva.dmrg.ttno.TTNOTemplate.slot_nodes` entry.

    The environments are the ones :func:`node_environments` builds (the down messages
    included); what differs is that no node's ``G_u`` is ever assembled. Each node's
    chains are sized over structure and ``require``d before they run.
    """
    graph = state.graph
    w = np.asarray(weights, dtype=float)
    stacked = _stack_roots(state.centers, w / float(np.sum(w)))
    cache = EnvironmentCache(ttno, state)
    parent, preorder = graph.parents(state.center)
    center_side = {int(x): int(parent[x]) for x in preorder[1:]}
    down: Dict[Tuple[int, int], BlockTensor] = {}
    for x in [int(i) for i in preorder[1:]]:
        v = center_side[x]
        down[(v, x)] = _down_message(ttno, state, cache, down, center_side, stacked, v, x)
    values: List[np.ndarray] = []
    with timer("network RDM slot extraction"):
        for rec in template.slot_nodes:
            u = rec.node
            nbrs = sorted(graph.neighbors(u))
            at_center = u == state.center
            ket = stacked if at_center else state.tensors[u]
            env_of = {x: (down[(x, u)] if (not at_center and x == center_side[u])
                          else cache.get(x, u)) for x in nbrs}
            groups = _SlotGroups(ttno, u, nbrs, rec.slot_rows, rec.slot_index)
            transient = _node_slot_transient(
                u, BlockShape.of(ket), nbrs, {x: BlockShape.of(e) for x, e in env_of.items()},
                at_center, ttno, rec.slot_rows, rec.slot_index, groups)
            res.require("network RDM slot extraction (node {})".format(u),
                        transient / 1024.0 ** 3,
                        note="{} slots over {} operator leg(s)".format(rec.slot.size,
                                                                       len(nbrs)),
                        advice=["reduce max_bond: a group's chain carries two bond legs "
                                "and the node's local dimension",
                                "a finer node partition: the closing step is dense in the "
                                "local dimension squared times one operator bond"])
            values.append(_node_slot_values(u, ket, nbrs, env_of, at_center, ttno,
                                            rec.slot_rows, rec.slot_index, groups))
    cache.release_all()
    return values


def slot_extraction_gb(ttno: TTNO, state: TTNState, n_roots: int) -> float:
    """Transient [GB] of :func:`slot_values` from structure alone — a bound, since the
    template's slots are not in hand at plan time: for every node, the largest chain over
    every choice of open leg and every prefix sector tuple.

    Environments come from :class:`~kuiva.dmrg.sweep.ShapeEnvironments` on the
    centre-filled shape state, standing in for the down messages the real contraction
    uses on the centre side (same spaces, the same sector sets up to the centre charge).
    """
    from .sweep import ShapeEnvironments, shape_state

    shapes = shape_state(state, fill_center=True)
    cache = ShapeEnvironments(ttno, shapes)
    graph = state.graph
    peak = 0
    for u in range(graph.n_nodes):
        nbrs = sorted(graph.neighbors(u))
        at_center = u == state.center
        ket = _stacked_shape(shapes.tensors[u], n_roots) if at_center else shapes.tensors[u]
        env_of = {x: cache.get(x, u) for x in nbrs}
        candidates = [None] if not nbrs else list(nbrs)
        for open_leg in candidates:
            prefix_legs = [x for x in nbrs if x != open_leg]
            sector_sets = [range(env_of[x].spaces[1].nsectors) for x in prefix_legs]
            for secs in itertools.product(*sector_sets):
                prefix = [(x, int(sec), 0) for x, sec in zip(prefix_legs, secs)]
                if any(not np.any(env_of[x].sectors[:, 1] == sec) for x, sec, _ in prefix):
                    continue
                steps: List[int] = []
                _slot_group_chain(u, ket, nbrs, env_of, at_center, open_leg, prefix,
                                  sizes=steps)
                peak = max(peak, _chain_peak(ket.nbytes, steps))
    return peak / 1024.0 ** 3


def _prepend_root_channel(g, ttno: TTNO):
    """Add the root W tensor's dim-1 completed-channel leg back onto ``G_root``.

    In the full contraction that leg is closed by a unit vector, so ``dE/dW_root`` is the
    closed contraction broadcast over a dim-1 leg — a reshape, not arithmetic.
    """
    space = ttno.bond_space[ttno.root]
    spaces = (space,) + g.spaces
    signs = (1,) + g.signs
    charge = g.charge + space.qns[0]
    rows = np.hstack([np.zeros((g.nblocks, 1), dtype=np.int64), g.sectors])
    if isinstance(g, BlockShape):
        return BlockShape(spaces, signs, charge, rows)
    blocks = [np.ascontiguousarray(b.reshape((1,) + b.shape)) for b in g.blocks]
    return BlockTensor(spaces, signs, charge, rows, blocks)


def network_rdms(template: TTNOTemplate, state: TTNState, *,
                 energies: Optional[Sequence[float]] = None,
                 n_elec: Optional[int] = None,
                 weights: Optional[Sequence[float]] = None,
                 enforce_kramers: bool = True,
                 degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
                 on_split: str = "raise",
                 ttno: Optional[TTNO] = None) -> Tuple[np.ndarray, np.ndarray]:
    """State-averaged ``(gamma, Gamma)`` from a converged network — the production path.

    ``template`` must be built on ``state.graph``; ``ttno`` is any fill on that topology
    (default: the template's own compile — the extracted values are fill-independent, see
    :func:`node_environments`). The state-averaging discipline mirrors
    :class:`kuiva.rdm.rdm.RDMBuilder`: with more than one root and ``enforce_kramers``,
    ``energies`` and ``n_elec`` are required and the weights are equalized within
    degenerate blocks, refusing a count that splits a Kramers pair.
    """
    if template.graph != state.graph:
        raise ValueError("the template was compiled on a different topology than the "
                         "state's — RDM extraction would read wrong channels")
    n_roots = state.n_roots
    if enforce_kramers and n_roots > 1:
        if energies is None or n_elec is None:
            raise ValueError(
                "state-averaged network RDMs over {} roots need the state energies and "
                "the electron count, so degenerate blocks can be weighted equally and an "
                "incomplete block refused. Pass energies= and n_elec=, "
                "or enforce_kramers=False to state deliberately that the weights are to "
                "be used as given".format(n_roots))
        w = state_average_weights([float(e) for e in energies], int(n_elec), weights,
                                  tol=degeneracy_tol, on_split=on_split)
    elif weights is None:
        w = np.full(n_roots, 1.0 / n_roots)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
    values = slot_values(template, template.ttno if ttno is None else ttno, state, w)
    return template.rdms_from_slot_values(values)


# --- annihilated states and their Gram matrix (ranks 1-4) ----------------------------------

def annihilation_term(modes: Sequence[int]) -> ProductTerm:
    """The Jordan-Wigner string of ``a_{m1} a_{m2} .. a_{mk}`` (ascending), tail included.

    For an odd operator count the parity factors below the lowest mode do not pair-cancel
    (module docstring), so the explicit ``Z`` tail down to mode 0 is part of the string.
    """
    modes = tuple(int(m) for m in modes)
    if any(b <= a for a, b in zip(modes, modes[1:])):
        raise ValueError("annihilation modes must be strictly ascending, got {}"
                         .format(modes))
    term = fermion_term(1.0, [(m, False) for m in modes])
    if term is None:                                   # pragma: no cover - distinct modes
        raise AssertionError("a string of distinct annihilations cannot vanish")
    if len(modes) % 2 == 0:
        return term
    lo = modes[0]
    tail_modes = tuple(range(lo)) + term.modes
    tail_mats = (np.ascontiguousarray(_Z),) * lo + term.mats
    return ProductTerm(1.0, tail_modes, tail_mats)


def _apply_term(ttno: TTNO, state: TTNState,
                term: ProductTerm) -> Tuple[List[Optional[BlockTensor]],
                                            List[BlockTensor], frozenset]:
    """Apply a product term to every root of a state: local matrices only, bonds untouched.

    Returns ``(tensors, centers, modified_nodes)`` — untouched tensors are shared with
    the input state, which is what keeps ``C(n, k)`` applied states affordable.
    """
    graph = state.graph
    mat_at = dict(zip(term.modes, term.mats))
    tensors: List[Optional[BlockTensor]] = list(state.tensors)
    centers = list(state.centers)
    modified = []
    for u in range(graph.n_nodes):
        touched = [m for m in ttno.node_modes[u] if m in mat_at]
        if not touched:
            continue
        modified.append(u)
        local = np.eye(1, dtype=np.complex128)
        shift = ttno.charge.zero_like()
        for m in ttno.node_modes[u]:
            mat = mat_at.get(m)
            if mat is None:
                local = np.kron(local, _I2)
            else:
                local = np.kron(local, mat)
                shift = shift + _charge_shift(mat, FERMION_MODE)
        perm = ttno.phys_perm[u]
        matp = local[np.ix_(perm, perm)]
        sp = ttno.phys_space[u]
        op = BlockTensor.from_dense(matp, (sp, sp), (1, -1), charge=shift)
        ax = (state.centers[0] if u == state.center else state.tensors[u]).ndim - 1
        if u == state.center:
            centers = [tensordot(c, op, axes=([ax], [1])) for c in centers]
        else:
            tensors[u] = tensordot(tensors[u], op, axes=([ax], [1]))
    return tensors, centers, frozenset(modified)


def _bond_identity(space, width: int) -> BlockTensor:
    rows = np.array([[i, i] for i in range(space.nsectors)], dtype=np.int64)
    blocks = [np.eye(int(space.dims[i]), dtype=np.complex128)
              for i in range(space.nsectors)]
    return BlockTensor((space, space), (1, -1), space.qns[0].zero_like(), rows,
                       blocks)


def _overlap(graph, center: int, parent: np.ndarray, order_leafward: Sequence[int],
             bra, ket, weights: np.ndarray) -> complex:
    """``sum_r w_r <bra_r|ket_r>`` of two applied states sharing graph, center and bonds.

    ``bra``/``ket`` are ``(tensors, centers, modified)`` triples from :func:`_apply_term`.
    Subtrees untouched by both sides contract to the identity (canonical isometries) and
    are skipped — the shortcut that keeps the Gram matrix affordable.
    """
    bra_t, bra_c, bra_mod = bra
    ket_t, ket_c, ket_mod = ket
    touched = bra_mod | ket_mod
    dirty: Dict[int, bool] = {}
    msgs: Dict[int, Optional[BlockTensor]] = {}
    for u in order_leafward:
        if u == center:
            continue
        p = int(parent[u])
        children = [x for x in sorted(graph.neighbors(u)) if x != p]
        is_dirty = (u in touched) or any(dirty[c] for c in children)
        dirty[u] = is_dirty
        if not is_dirty:
            msgs[u] = None
            continue
        nbrs = sorted(graph.neighbors(u))
        t = _Lab(ket_t[u], [("b", u, x) for x in nbrs] + [("p", u)])
        for c in children:
            if msgs[c] is not None:
                t = t.dot(_Lab(msgs[c], [("bra", c), ("ket", c)]),
                          [(("b", u, c), ("ket", c))])
        bra_lab = _Lab(bra_t[u].conj(), [("cb", u, x) for x in nbrs] + [("cp",)])
        pairs = [(("p", u), ("cp",))]
        for c in children:
            pairs.append(((("bra", c) if msgs[c] is not None else ("b", u, c)),
                          ("cb", u, c)))
        t = t.dot(bra_lab, pairs)
        msgs[u] = t.to([("cb", u, p), ("b", u, p)])

    nbrs = sorted(graph.neighbors(center))
    total = 0.0 + 0.0j
    for r, w in enumerate(weights):
        if w == 0.0:
            continue
        t = _Lab(ket_c[r], [("b", center, x) for x in nbrs] + [("p", center)])
        labels = []
        for x in nbrs:
            if msgs.get(x) is not None:
                t = t.dot(_Lab(msgs[x], [("bra", x), ("ket", x)]),
                          [(("b", center, x), ("ket", x))])
                labels.append(("bra", x))
            else:
                labels.append(("b", center, x))
        kt = t.to(labels + [("p", center)])
        val = 0.0 + 0.0j
        for row, blk in zip(kt.sectors, kt.blocks):
            b = bra_c[r].find(row)
            if b is not None:
                val += np.vdot(b, blk)                 # vdot conjugates the bra
        total += w * val
    return complex(total)


def _perm_inversions(p: Sequence[int]) -> int:
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])


def _overlap_pairs(graph, center: int, parent: np.ndarray,
                   order_leafward: Sequence[int], bra, ket) -> np.ndarray:
    """``M[I, J] = <bra_I|ket_J>`` over every root pair of two applied states.

    The cross-root generalization of :func:`_overlap`: the subtree messages depend only
    on the isometries (which every root shares), so they are built once and the root
    pairing happens only at the center closure — one contraction per ket root, one
    ``vdot`` per pair. Same dirty-subtree shortcut, same conventions.
    """
    bra_t, bra_c, bra_mod = bra
    ket_t, ket_c, ket_mod = ket
    touched = bra_mod | ket_mod
    dirty: Dict[int, bool] = {}
    msgs: Dict[int, Optional[BlockTensor]] = {}
    for u in order_leafward:
        if u == center:
            continue
        p = int(parent[u])
        children = [x for x in sorted(graph.neighbors(u)) if x != p]
        is_dirty = (u in touched) or any(dirty[c] for c in children)
        dirty[u] = is_dirty
        if not is_dirty:
            msgs[u] = None
            continue
        nbrs = sorted(graph.neighbors(u))
        t = _Lab(ket_t[u], [("b", u, x) for x in nbrs] + [("p", u)])
        for c in children:
            if msgs[c] is not None:
                t = t.dot(_Lab(msgs[c], [("bra", c), ("ket", c)]),
                          [(("b", u, c), ("ket", c))])
        bra_lab = _Lab(bra_t[u].conj(), [("cb", u, x) for x in nbrs] + [("cp",)])
        pairs = [(("p", u), ("cp",))]
        for c in children:
            pairs.append(((("bra", c) if msgs[c] is not None else ("b", u, c)),
                          ("cb", u, c)))
        t = t.dot(bra_lab, pairs)
        msgs[u] = t.to([("cb", u, p), ("b", u, p)])

    nbrs = sorted(graph.neighbors(center))
    # ⚠ Every root's center tensor has the same block structure (one shared basis, one
    # local operator applied to all of them), so the roots ride on a trailing leg: the
    # messages are contracted ONCE for the whole ensemble and the closure is one GEMM per
    # block. Measured motive: the per-root form — a block lookup and a ``vdot`` per root
    # pair per block — was half the CPU of a 25-root ladder point after the sweep itself
    # had been made cheap (165 000 lookups for one FeCl2 transition-density set).
    ket_all = _stacked_centers(ket_c)
    bra_all = _stacked_centers(bra_c)
    n_roots = int(ket_all.spaces[-1].total_dim)
    t = _Lab(ket_all, [("b", center, x) for x in nbrs] + [("p", center), ("r",)])
    labels = []
    for x in nbrs:
        if msgs.get(x) is not None:
            t = t.dot(_Lab(msgs[x], [("bra", x), ("ket", x)]),
                      [(("b", center, x), ("ket", x))])
            labels.append(("bra", x))
        else:
            labels.append(("b", center, x))
    kt = t.to(labels + [("p", center), ("r",)])
    m = np.zeros((n_roots, n_roots), dtype=np.complex128)
    for row, blk in zip(kt.sectors, kt.blocks):
        b = bra_all.find(row)
        if b is not None:
            # M[i, j] += sum_k conj(bra_i[k]) ket_j[k]
            m += b.reshape(-1, n_roots).T.conj() @ blk.reshape(-1, n_roots)
    return m


def _stacked_centers(centers) -> BlockTensor:
    """The roots of an applied state on one trailing leg, unit weights — or the tensor
    itself when the caller stacked it already (:func:`transition_rdm1s` does, once per
    applied state instead of twice per mode pair)."""
    if isinstance(centers, BlockTensor):
        return centers
    return _stack_roots(list(centers), np.ones(len(centers)))


def transition_rdm1s(ttno: TTNO, state: TTNState) -> np.ndarray:
    """``gamma^{IJ}_pq = <psi_I| a+_p a_q |psi_J>`` over every root pair.

    Returns ``(n_roots, n_roots, n, n)`` — the network counterpart of
    :meth:`kuiva.mcscf.casci.FullCISolver.transition_densities`, in the same
    ``<I|E_pq|J>`` convention, so :func:`kuiva.props.dump.property_matrices` consumes it
    unchanged. The diagonal blocks reproduce the per-root 1-RDMs exactly (asserted in the
    tests against :func:`network_rdms` with one-hot weights).

    Route: ``<psi_I|a+_p a_q|psi_J> = <a_p psi_I | a_q psi_J>`` — the same applied-string
    Gram as :func:`network_rdm` at rank 1, with bra and ket taken from **different**
    roots (:func:`_overlap_pairs`). ⚠ The single-annihilation string is odd, so both
    sides carry the explicit Jordan-Wigner ``Z`` tail (:func:`annihilation_term`); the
    tails cancel in the overlap only because *both* sides carry them.

    ⚠ **Phases are per state and arbitrary**: an off-diagonal block is defined up to the
    phase difference of its two roots, exactly as the CI route's is up to the
    eigensolver's phases. Everything downstream (the moment matrices of the property
    layer) is consumed through phase-invariant reductions only.
    """
    graph = state.graph
    n = sum(len(c) for c in graph.contents)
    n_roots = state.n_roots
    parent, preorder = graph.parents(state.center)
    order_leafward = [int(x) for x in reversed(preorder)]
    with timer("network transition densities: applied states"):
        applied = []
        for m in range(n):
            tensors, centers, modified = _apply_term(ttno, state, annihilation_term((m,)))
            applied.append((tensors, _stacked_centers(centers), modified))
    gamma = np.empty((n_roots, n_roots, n, n), dtype=np.complex128)
    with timer("network transition densities: overlaps"):
        for p in range(n):
            for q in range(n):
                gamma[:, :, p, q] = _overlap_pairs(graph, state.center, parent,
                                                   order_leafward, applied[p], applied[q])
    return gamma


#: Sentinel for :func:`koopmans_gram`: the row is the unmodified reference itself.
IDENTITY_TERM = "identity"


def _applied_or_none(ttno: TTNO, state: TTNState, term):
    if term is None:
        return None
    if term == IDENTITY_TERM:
        return (list(state.tensors), list(state.centers), frozenset())
    return _apply_term(ttno, state, term)


def _sandwich_pair(graph, center: int, parent: np.ndarray,
                   order_leafward: Sequence[int], cache: EnvironmentCache,
                   ttno: TTNO, bra, ket, root: int) -> complex:
    """``<bra_root| H |ket_root>`` for two applied states sharing the state's bonds.

    ⚠ The dirty-path messages below contract the environments into the ket before the
    W, one operator leg per child open at once — the order the sweep's
    :func:`~kuiva.dmrg.sweep.renormalize` no longer uses. It is fine on a chain and costs
    the product of the children's operator bond dimensions on a branching node; the
    perturbation has not run on a tree, and when it does this is the chain to move onto
    ``renormalize``.

    The three-layer (bra, TTNO, ket) tree contraction with the clean-subtree shortcut:
    a subtree neither side modified contributes exactly the standard environment of the
    unmodified state, served by ``cache``; only the dirty paths are recomputed per pair,
    which is what keeps a Gram over ``n^2`` strings affordable.
    """
    bra_t, bra_c, bra_mod = bra
    ket_t, ket_c, ket_mod = ket
    touched = bra_mod | ket_mod
    dirty: Dict[int, bool] = {}
    msgs: Dict[int, Optional[BlockTensor]] = {}
    for u in order_leafward:
        if u == center:
            continue
        p = int(parent[u])
        children = [x for x in sorted(graph.neighbors(u)) if x != p]
        is_dirty = (u in touched) or any(dirty[c] for c in children)
        dirty[u] = is_dirty
        if not is_dirty:
            msgs[u] = None
            continue
        nbrs = sorted(graph.neighbors(u))
        t = _Lab(ket_t[u], [("b", u, x) for x in nbrs] + [("p", u)])
        for c in children:
            env = msgs[c] if msgs[c] is not None else cache.get(c, u)
            t = t.dot(_Lab(env, [("bra", c), ("op", c), ("ket", c)]),
                      [(("b", u, c), ("ket", c))])
        t = t.dot(_w_lab(ttno, u, p),
                  [(("op", c), ("op", c)) for c in children] + [(("p", u), ("pi",))])
        t = t.dot(_Lab(bra_t[u].conj(), [("cb", u, x) for x in nbrs] + [("cp",)]),
                  [(("bra", c), ("cb", u, c)) for c in children] + [(("po",), ("cp",))])
        msgs[u] = t.to([("cb", u, p), ("op_out",), ("b", u, p)])

    nbrs = sorted(graph.neighbors(center))
    t = _Lab(ket_c[root], [("b", center, x) for x in nbrs] + [("p", center)])
    for x in nbrs:
        env = msgs[x] if msgs.get(x) is not None else cache.get(x, center)
        t = t.dot(_Lab(env, [("bra", x), ("op", x), ("ket", x)]),
                  [(("b", center, x), ("ket", x))])
    t = t.dot(_w_lab(ttno, center, None),
              [(("op", x), ("op", x)) for x in nbrs] + [(("p", center), ("pi",))])
    # Close against the bra center by block matching (vdot conjugates the bra) — a full
    # tensordot to a scalar is not a BlockTensor, and this is _overlap's own idiom.
    kt = t.to([("bra", x) for x in nbrs] + [("po",)])
    total = 0.0 + 0.0j
    for row, blk in zip(kt.sectors, kt.blocks):
        b = bra_c[root].find(row)
        if b is not None:
            total += np.vdot(b, blk)
    return complex(total)


def koopmans_gram(ttno_h: TTNO, state: TTNState, root: int, terms,
                  energy: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(S, K)`` over applied ladder-string states of **one** root: ``S_ab =
    <chi_a|chi_b>`` and ``K_ab = <chi_a|(H_act - E)|chi_b>``.

    The network counterpart of :meth:`kuiva.pt.contractions.ShiftedSpace.gram` — the
    denominator kernels of the SC-NEVPT2 ``(+-2)`` and ``(0')`` classes, contracted
    through the network instead of through a shifted determinant space. ``terms`` is a
    sequence of :class:`~kuiva.dmrg.ttno.ProductTerm` (**even** strings only — an odd
    string needs the explicit Jordan-Wigner tail of :func:`annihilation_term`, and
    nothing here adds one), ``None`` for an identically vanishing row (a coincident
    label), or :data:`IDENTITY_TERM` for the reference itself. ``ttno_h`` must be the
    active Hamiltonian the reference was converged on — for a CASSCF reference, the
    Dyall active Hamiltonian, which the pseudo-canonicalization leaves untouched.

    Both matrices are Hermitian by construction (only the upper triangle is contracted)
    whatever the reference's convergence; what a *truncated* reference costs is that
    ``K`` rows involving the reference are only approximately what an eigenvector would
    give, which is the same statement the provider's Hermiticity diagnostics make.
    """
    for term in terms:
        if term is None or term == IDENTITY_TERM:
            continue
        # Parity of a pure fermionic string equals its particle-number change mod 2, and
        # the compiler truncates the JW tail below the string's lowest mode — sound for
        # even strings only (module docstring). Refuse rather than return a Gram that is
        # wrong by exactly a fermionic sign pattern no Hermiticity check can see.
        delta_n = sum(_charge_shift(mat, FERMION_MODE).n for mat in term.mats)
        if delta_n % 2 != 0:
            raise ValueError(
                "koopmans_gram takes even ladder strings only: an odd string needs the "
                "explicit Jordan-Wigner tail (annihilation_term), which these terms do "
                "not carry")
    graph = state.graph
    parent, preorder = graph.parents(state.center)
    order_leafward = [int(x) for x in reversed(preorder)]
    with timer("network Koopmans Gram: applied states"):
        applied = [_applied_or_none(ttno_h, state, t) for t in terms]
    m = len(applied)
    overlap = np.zeros((m, m), dtype=np.complex128)
    koopmans = np.zeros((m, m), dtype=np.complex128)
    cache = EnvironmentCache(ttno_h, state)
    try:
        with timer("network Koopmans Gram: sandwiches"):
            for a in range(m):
                if applied[a] is None:
                    continue
                for b in range(a, m):
                    if applied[b] is None:
                        continue
                    s_ab = _overlap_pairs(graph, state.center, parent, order_leafward,
                                          applied[a], applied[b])[root, root]
                    h_ab = _sandwich_pair(graph, state.center, parent, order_leafward,
                                          cache, ttno_h, applied[a], applied[b], root)
                    k_ab = h_ab - float(energy) * s_ab
                    overlap[a, b] = s_ab
                    koopmans[a, b] = k_ab
                    if b != a:
                        overlap[b, a] = np.conj(s_ab)
                        koopmans[b, a] = np.conj(k_ab)
    finally:
        cache.release_all()
    return overlap, koopmans


def state_rdms(template: TTNOTemplate, state: TTNState, *,
               ttno: Optional[TTNO] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per-root ``(gamma, Gamma)`` — one one-hot pass of the production path per root.

    The per-state counterpart of :func:`network_rdms`, for the analysis layer
    (``<S^2>``, term assignment): each root's own densities, **no** state averaging and
    therefore no averaging gate — the same deliberate exemption the CI route's per-state
    RDMs carry (a single state's densities are what a per-state diagnostic is made of;
    the gate is a statement about an ensemble).
    """
    op = template.ttno if ttno is None else ttno
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(state.n_roots):
        weights = np.zeros(state.n_roots)
        weights[r] = 1.0
        out.append(template.rdms_from_slot_values(slot_values(template, op, state, weights)))
    return out


def network_rdm(ttno: TTNO, state: TTNState, rank: int, *,
                energies: Optional[Sequence[float]] = None,
                n_elec: Optional[int] = None,
                weights: Optional[Sequence[float]] = None,
                enforce_kramers: bool = True,
                degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
                on_split: str = "raise") -> np.ndarray:
    """The dense rank-``k`` RDM by direct network contraction (module docstring).

    Returns the ``n^(2k)`` array in the interleaved pair convention above. Exact —
    no cumulant enters anywhere — and refused by the resource budget before the dense array
    or the applied-state set would not fit. Ranks 1 and 2 are supported as the
    independent cross-check of :func:`network_rdms`; ranks 3 and 4 are the product.
    """
    rank = int(rank)
    if not 1 <= rank <= 4:
        raise ValueError("rank must be 1..4, got {}".format(rank))
    for dims in ttno.mode_dims:
        if any(d != 2 for d in dims):
            raise ValueError("RDMs are fermionic; every mode must have local dimension 2")
    graph = state.graph
    n = sum(len(c) for c in graph.contents)
    n_roots = state.n_roots
    if enforce_kramers and n_roots > 1:
        if energies is None or n_elec is None:
            raise ValueError("state-averaged network RDMs need energies= and n_elec= "
                             ", or enforce_kramers=False")
        w = state_average_weights([float(e) for e in energies], int(n_elec), weights,
                                  tol=degeneracy_tol, on_split=on_split)
    elif weights is None:
        w = np.full(n_roots, 1.0 / n_roots)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)

    combos = list(itertools.combinations(range(n), rank))
    nc = len(combos)
    gamma_gb = res.rdm_gb(n, rank)
    res.require(
        "network {}-RDM ({} spinors)".format(rank, n),
        gamma_gb + res.array_gb((nc, nc), np.complex128),
        note="the dense n^{} array plus a {}x{} Gram matrix".format(2 * rank, nc, nc),
        advice=["the stored rank-{} RDM is n^{}; only a smaller active space changes "
                "it. For multi-site actives a stored 4-RDM is impossible and needs the "
                "contraction-on-demand NEVPT2 of its own future plan "
                "".format(rank, 2 * rank)])

    parent, preorder = graph.parents(state.center)
    order_leafward = [int(x) for x in reversed(preorder)]
    applied = []
    applied_bytes = 0
    with timer("network {}-RDM: applied states".format(rank)):
        for combo in combos:
            triple = _apply_term(ttno, state, annihilation_term(combo))
            tensors, centers, modified = triple
            applied_bytes += sum(tensors[u].nbytes for u in modified
                                 if u != state.center)
            if state.center in modified:
                applied_bytes += sum(c.nbytes for c in centers)
            applied.append(triple)
    # exact-at-build reservation, the environment precedent in `sweep.py`: the applied
    # tensors share every untouched block with the state, so their size is only known
    # once the touched set is
    alloc = res.reserve("network {}-RDM applied states ({} strings)".format(rank, nc),
                        applied_bytes / 1024.0 ** 3,
                        note="modified node tensors only; untouched blocks are shared",
                        advice=["a smaller active space or rank is the only knob"])

    m = np.zeros((nc, nc), dtype=np.complex128)
    with timer("network {}-RDM: Gram matrix".format(rank)):
        for a in range(nc):
            for b in range(a, nc):
                val = _overlap(graph, state.center, parent, order_leafward,
                               applied[a], applied[b], w)
                m[a, b] = val
                if b != a:
                    m[b, a] = np.conj(val)
    res.BUDGET.release(alloc)

    combo_arr = np.asarray(combos, dtype=np.int64)
    gamma = np.zeros((n,) * (2 * rank), dtype=np.complex128)
    k = rank
    rev_sign = (-1) ** (k * (k - 1) // 2)
    with timer("network {}-RDM: antisymmetry fill".format(rank)):
        for pi in itertools.permutations(range(k)):
            sgn_c = rev_sign * (-1) ** _perm_inversions(pi)
            p_cols = [combo_arr[:, j] for j in pi]      # creations as written
            for rho in itertools.permutations(range(k)):
                sgn = sgn_c * (-1) ** _perm_inversions(rho)
                w_cols = [combo_arr[:, j] for j in rho]  # annihilations as written
                index: List[np.ndarray] = []
                for j in range(k):
                    index.append(p_cols[j][:, None])          # creation j -> axis 2j
                    index.append(w_cols[k - 1 - j][None, :])  # annihilation -> axis 2j+1
                gamma[tuple(index)] = sgn * m
    return gamma


__all__ = ["network_rdms", "network_rdm", "node_environments", "slot_values",
           "slot_extraction_gb", "annihilation_term", "state_rdms", "transition_rdm1s",
           "koopmans_gram", "IDENTITY_TERM"]
