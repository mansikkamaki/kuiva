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
network computes every node's operator environment ``G_u = dE/dW_u`` — the contraction of
bra, ket and every *other* node's W tensor around node ``u`` — and
:meth:`kuiva.dmrg.ttno.TTNOTemplate.rdms_from_environments` reads each elementary
operator's expectation out of its coefficient-attachment slot. This costs about two sweeps'
worth of environment builds per call, independent of ``n^4``, which is what a CASSCF
macro-iteration needs. Correctness rests on the TTNO compiler's label-completeness (a
channel's operator content is uniquely determined by its label, so the read is
uncontaminated by other terms' coefficients — see the template's docstring), and is
asserted by the energy-closure and CI-parity tests rather than trusted.

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
  J. Chem. Theory Comput. 12, 1583 (2016), doi:10.1021/acs.jctc.6b00118.
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
from .block import BlockTensor, QuantumNumber, tensordot
from .sweep import EnvironmentCache, TTNState, _Lab, _stack_roots, _w_lab
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
    ket = stacked if at_center else state.tensors[v]
    labels = [("b", v, x) for x in nbrs] + [("p", v)]
    if at_center:
        labels.append(("r",))
    t = _Lab(ket, labels)
    for x in nbrs:
        if x == u:
            continue
        env = down[(x, v)] if (not at_center and x == center_side[v]) \
            else cache.get(x, v)
        t = t.dot(_Lab(env, [("bra", x), ("op", x), ("ket", x)]),
                  [(("b", v, x), ("ket", x))])
    t = t.dot(_w_lab(ttno, v, u),
              [(("op", x), ("op", x)) for x in nbrs if x != u] + [(("p", v), ("pi",))])
    bra_labels = [("cb", v, x) for x in nbrs] + [("cp",)]
    pairs = [(("bra", x), ("cb", v, x)) for x in nbrs if x != u] + [(("po",), ("cp",))]
    if at_center:
        bra_labels.append(("cr",))
        pairs.append((("r",), ("cr",)))
    t = t.dot(_Lab(ket.conj(), bra_labels), pairs)
    return t.to([("cb", v, u), ("op_out",), ("b", v, u)])


def node_environments(ttno: TTNO, state: TTNState,
                      weights: Sequence[float]) -> List[BlockTensor]:
    """``G_u = dE/dW_u`` for every node: bra, ket and every other node's W contracted.

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
    cache = EnvironmentCache(ttno, state)
    parent, preorder = graph.parents(state.center)
    center_side = {int(x): int(parent[x]) for x in preorder[1:]}

    down: Dict[Tuple[int, int], BlockTensor] = {}
    for x in [int(i) for i in preorder[1:]]:
        v = center_side[x]
        down[(v, x)] = _down_message(ttno, state, cache, down, center_side, stacked,
                                     v, x)

    envs: List[BlockTensor] = []
    with timer("network RDM environments"):
        for u in range(graph.n_nodes):
            nbrs = sorted(graph.neighbors(u))
            at_center = u == state.center
            ket = stacked if at_center else state.tensors[u]
            labels = [("b", u, x) for x in nbrs] + [("p", u)]
            if at_center:
                labels.append(("r",))
            t = _Lab(ket, labels)
            for x in nbrs:
                env = down[(x, u)] if (not at_center and x == center_side[u]) \
                    else cache.get(x, u)
                t = t.dot(_Lab(env, [("bra", x), ("op", x), ("ket", x)]),
                          [(("b", u, x), ("ket", x))])
            bra_labels = [("cb", u, x) for x in nbrs] + [("cp",)]
            pairs = [(("bra", x), ("cb", u, x)) for x in nbrs]
            if at_center:
                bra_labels.append(("cr",))
                pairs.append((("r",), ("cr",)))
            t = t.dot(_Lab(ket.conj(), bra_labels), pairs)

            order = []
            if u != ttno.root:
                order.append(("op", int(ttno.parent[u])))
            order += [("op", c) for c in ttno.children[u]]
            order += [("cp",), ("p", u)]
            g = t.to(order)
            if u == ttno.root:
                g = _prepend_root_channel(g, ttno)
            envs.append(g)
    cache.release_all()
    return envs


def _prepend_root_channel(g: BlockTensor, ttno: TTNO) -> BlockTensor:
    """Add the root W tensor's dim-1 completed-channel leg back onto ``G_root``.

    In the full contraction that leg is closed by a unit vector, so ``dE/dW_root`` is the
    closed contraction broadcast over a dim-1 leg — a reshape, not arithmetic.
    """
    space = ttno.bond_space[ttno.root]
    spaces = (space,) + g.spaces
    signs = (1,) + g.signs
    charge = g.charge + space.qns[0]
    rows = np.hstack([np.zeros((g.nblocks, 1), dtype=np.int64), g.sectors])
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
    envs = node_environments(template.ttno if ttno is None else ttno, state, w)
    return template.rdms_from_environments(envs)


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
        shift = QuantumNumber.zero(ttno.charge.width)
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
    return BlockTensor((space, space), (1, -1), QuantumNumber.zero(width), rows, blocks)


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


__all__ = ["network_rdms", "network_rdm", "node_environments", "annihilation_term"]
