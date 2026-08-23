"""Seeding the tensor network from the cheap CI.

Two seeds, both computed from data the cheap-CI preoptimization already produces and both
consumed here as **plain arrays** — ``kuiva.dmrg`` never imports ``kuiva.mcscf`` (the
dependency runs the other way, exactly as for the TTNO compiler):

* :func:`topology_from_mutual_information` — a tree topology guessed from the orbital
  mutual-information matrix ``I_pq`` by agglomerative clustering: strongly entangled
  orbitals cluster into candidate *sites*, each site is laid out as a Fiedler-ordered
  chain, and sites are joined by single edges at their most-entangled orbital pair. For a
  single cluster this reduces exactly to the Fiedler-ordered MPS chain — the configuration
  every fixed-topology run validated — and for separated magnetic sites it produces the
  chains-joined-by-weak-bonds shape that adaptive reconnection refines. The
  guess does not need to be right; it needs to be *cheap and unbiased*, because
  reconnection exists to fix it.
* :func:`expansion_to_ttn` — the selected-CI determinant expansion converted to a tree
  tensor network state by sequential Schmidt splits along the tree, **exact within the
  CI's determinant space** when untruncated. The conversion works on the sparse
  determinant list directly (never a ``2^n`` dense vector): for each subtree, the
  ensemble's Schmidt basis is computed from the matrix of CI coefficients indexed by
  (inside occupation pattern) x (outside pattern, root), and every SVD runs through
  :func:`kuiva.dmrg.block.svd`, so the degenerate-group truncation discipline (at
  every bond) applies to the seed exactly as it does to the sweep. The resulting Schmidt
  spectra *are* the seeded bond dimensions.

Why no fermionic signs appear in the conversion
-----------------------------------------------
The TTNO compiler Jordan-Wigner-transforms the *operators* against the global ascending
mode order (:mod:`kuiva.dmrg.ttno`), which makes the network's kron basis coincide with
the determinant basis of :mod:`kuiva.ci.strings` with unit coefficients. A state tensor
network is a plain product-basis object in that convention: splitting the mode set into
subtrees is an index permutation of tensor factors, never an operator reordering, so the
conversion moves coefficients and takes SVDs — no sign bookkeeping anywhere. (This is the
same statement as "correctness is topology-independent" in the TTNO docstring, applied to
states.)

Everything here is orchestration and stays Python: the conversion runs once
per seed, and its cost is a loop over the determinant list with small dense contractions.

References
----------
* Exact MPS decomposition of a state by successive SVDs: U. Schollwoeck, Ann. Phys. 326,
  96 (2011), doi:10.1016/j.aop.2010.09.012 (Sec. 4.1.3), generalized here to trees and to
  a sparse determinant list with a shared-basis root ensemble (Dorando, Hachmann & Chan,
  J. Chem. Phys. 127, 084109 (2007), doi:10.1063/1.2768360).
* Mutual-information-driven orbital ordering and clustering for DMRG/TTNS: J. Rissler,
  R. M. Noack, S. R. White, Chem. Phys. 323, 519 (2006), doi:10.1016/j.chemphys.2005.10.018;
  G. Barcza, O. Legeza, K. H. Marti, M. Reiher, Phys. Rev. A 83, 012508 (2011),
  doi:10.1103/PhysRevA.83.012508; trees laid out along entanglement structure:
  V. Murg, F. Verstraete, O. Legeza, R. M. Noack, Phys. Rev. B 82, 205105 (2010),
  doi:10.1103/PhysRevB.82.205105; N. Nakatani, G. K.-L. Chan, J. Chem. Phys. 138, 134113
  (2013), doi:10.1063/1.4798639.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..rdm.entropy import fiedler_order
from ..util import resources as res
from ..util.logging import get_logger
from .block import BlockTensor, QuantumNumber, Space, svd
from .graph import NetworkGraph
from .sweep import TTNState, _bond_axis
from .ttno import TTNO

log = get_logger(__name__)

#: Relative linkage (against the largest mutual-information entry) below which an
#: agglomerative merge is *inter-site* rather than intra-site. A pure guess-shaping knob:
#: any value produces a valid tree, and adaptive reconnection is what corrects a wrong
#: guess. 0 makes the guess a single Fiedler chain; large values make every orbital its
#: own site.
DEFAULT_SITE_SPLIT = 0.25


# --- topology guess -----------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyGuess:
    """What the clustering produced: the tree, and the candidate site decomposition."""

    graph: NetworkGraph
    sites: Tuple[Tuple[int, ...], ...]     #: orbital clusters, each sorted, order stable
    cut: float                             #: the absolute linkage threshold used


def _linkage(info: np.ndarray, a: Sequence[int], b: Sequence[int], kind: str) -> float:
    block = info[np.ix_(list(a), list(b))]
    if kind == "average":
        return float(block.mean())
    if kind == "single":
        return float(block.max())
    if kind == "complete":
        return float(block.min())
    raise ValueError("unknown linkage {!r}; use 'average', 'single' or 'complete'"
                     .format(kind))


def topology_from_mutual_information(info: np.ndarray, *,
                                     site_split: float = DEFAULT_SITE_SPLIT,
                                     linkage: str = "average") -> TopologyGuess:
    """A tree topology from the mutual-information matrix (module docstring).

    Nodes are the orbitals themselves (node ``i`` carries orbital label ``i``): within
    each cluster the orbitals form a Fiedler-ordered chain, and clusters are joined —
    in agglomerative merge order — by one edge between their most-entangled orbital
    pair. Exactly ``n - 1`` edges, so the result is always a tree, and with a single
    cluster it is exactly the Fiedler chain.

    ``site_split`` is relative to ``info.max()``; ``linkage`` selects the cluster
    similarity (average by default, on measurement; single/complete kept
    so the choice can be measured rather than argued).
    """
    info = np.asarray(info, dtype=float)
    n = info.shape[0]
    if info.shape != (n, n):
        raise ValueError("mutual information must be square, got {}".format(info.shape))
    if n == 1:
        return TopologyGuess(NetworkGraph(1, [], [(0,)]), ((0,),), 0.0)

    scale = float(info.max())
    cut = site_split * scale
    clusters: List[List[int]] = [[i] for i in range(n)]
    merges: List[Tuple[float, List[int], List[int]]] = []
    while len(clusters) > 1:
        best, bi, bj = -1.0, 0, 1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = _linkage(info, clusters[i], clusters[j], linkage)
                if s > best:
                    best, bi, bj = s, i, j
        a, b = clusters[bi], clusters[bj]
        merges.append((best, list(a), list(b)))
        clusters = [c for k, c in enumerate(clusters) if k not in (bi, bj)] + [a + b]

    # sites: connected components over the intra-site merges (linkage >= cut)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, a, b in merges:
        if s >= cut:
            parent[find(a[0])] = find(b[0])
    site_of = [find(i) for i in range(n)]
    site_roots = sorted(set(site_of))
    sites = tuple(tuple(sorted(i for i in range(n) if site_of[i] == r))
                  for r in site_roots)

    edges: List[Tuple[int, int]] = []
    for members in sites:                              # Fiedler chain within each site
        if len(members) > 1:
            sub = info[np.ix_(members, members)]
            order = [members[k] for k in fiedler_order(sub)]
            edges.extend((order[i], order[i + 1]) for i in range(len(order) - 1))
    joined = {r: {r} for r in site_roots}              # inter-site: merge order, max-I pair
    for s, a, b in merges:
        ra, rb = find(a[0]), find(b[0])
        if ra == rb:
            continue                                   # intra-site merge, already chained
        block = info[np.ix_(a, b)]
        ia, ib = np.unravel_index(int(np.argmax(block)), block.shape)
        edges.append((a[int(ia)], b[int(ib)]))
        parent[ra] = rb

    graph = NetworkGraph(n, edges, [(i,) for i in range(n)])
    log.debug("topology guess: %d orbitals -> %d sites (cut %.3e nats, linkage %s)",
              n, len(sites), cut, linkage)
    return TopologyGuess(graph, sites, cut)


# --- determinant expansion -> TTN ---------------------------------------------------------

def _kron_index(masks: np.ndarray, modes: Sequence[int]) -> np.ndarray:
    """Local kron index of each determinant at a node (first mode slowest, C order)."""
    idx = np.zeros(masks.shape, dtype=np.int64)
    for m in modes:
        idx = (idx << 1) | ((masks >> np.uint64(m)) & np.uint64(1)).astype(np.int64)
    return idx


def _popcount(masks: np.ndarray) -> np.ndarray:
    counts = np.zeros(masks.shape, dtype=np.int64)
    work = masks.copy()
    while work.any():
        counts += (work & np.uint64(1)).astype(np.int64)
        work >>= np.uint64(1)
    return counts


def _mode_quantum_numbers(ttno, n_modes: int):
    """The occupied-mode quantum numbers, in mode order, defaulting to particle number.

    A TTNO compiled without irrep labels carries none; particle number alone is then the whole
    quantum number and ``(1,)`` per mode is exactly right.
    """
    if getattr(ttno, "mode_charges", ()):
        return list(ttno.mode_charges)
    one = ttno.charge.like([1] + [0] * (ttno.charge.width - 1))
    return [one] * int(n_modes)


def _mask_charge(masks: np.ndarray, mode_qn, reference):
    """Quantum number of each determinant bitmask: the group sum over its occupied modes."""
    zero = reference.zero_like()
    out = []
    for m in np.asarray(masks, dtype=np.uint64):
        total = zero
        bits = int(m)
        for mode in range(len(mode_qn)):
            if (bits >> mode) & 1:
                total = total + mode_qn[mode]
        out.append(total)
    return out


def expansion_to_ttn(ttno: TTNO, masks: np.ndarray, civecs: np.ndarray, *,
                     weights: Optional[Sequence[float]] = None,
                     center: Optional[int] = None, max_bond: Optional[int] = None,
                     tol: float = 0.0) -> TTNState:
    """Convert a determinant expansion into a canonical shared-basis :class:`TTNState`.

    ``masks`` are ``ci/strings.py`` occupation bitmasks (bit ``p`` = mode ``p``),
    ``civecs`` the ``(ndet, n_roots)`` coefficients; roots become the shared-basis center
    set, with the ensemble Schmidt bases computed from the ``sqrt(w)``-weighted stack
    (the same weighting the sweep's truncation uses). Untruncated (``tol = 0``,
    ``max_bond = None``, the default) the conversion is **exact within the span of the
    determinant list**: each returned center reproduces its CI vector up to
    normalization, which a Tier-0 test asserts through :func:`~kuiva.dmrg.sweep.
    state_to_dense`. With a cap or tolerance, every truncation is a
    :func:`~kuiva.dmrg.block.svd` group-complete truncation of the ensemble spectrum.

    ⚠ Determinant expansions are fermionic: every mode must be a dim-2
    :data:`~kuiva.dmrg.ttno.FERMION_MODE`-like basis (checked). The graph's mode labels
    must be exactly ``0..n_modes-1``, matching the bitmask convention.
    """
    graph = ttno.graph
    center = ttno.root if center is None else int(center)
    masks = np.ascontiguousarray(np.asarray(masks, dtype=np.uint64))
    civecs = np.asarray(civecs, dtype=np.complex128)
    if civecs.ndim == 1:
        civecs = civecs[:, None]
    ndet, n_roots = civecs.shape
    if masks.shape != (ndet,):
        raise ValueError("{} masks for {} CI columns".format(masks.shape, ndet))
    if ndet == 0:
        raise ValueError("cannot seed a network from an empty determinant list")
    for dims in ttno.mode_dims:
        if any(d != 2 for d in dims):
            raise ValueError("determinant expansions are fermionic; every mode must "
                             "have local dimension 2")
    all_modes = sorted(m for c in graph.contents for m in c)
    if all_modes != list(range(len(all_modes))):
        raise ValueError("graph mode labels must be exactly 0..n-1 to match determinant "
                         "bitmasks, got {}".format(all_modes))

    counts = _popcount(masks)
    n_elec = int(counts[0])
    if not np.all(counts == n_elec):
        raise ValueError("determinants carry different electron counts")
    width = ttno.charge.width
    # The subtree quantum numbers below are built the way the total is: particle number plus,
    # where the modes carry them, the group sum of the occupied modes' irrep labels.
    # ⚠ Without that second half a labelled network would be seeded entirely in the totally
    # symmetric sector — a perfectly normalized state of the wrong symmetry.
    mode_qn = _mode_quantum_numbers(ttno, len(all_modes))
    det_charges = _mask_charge(masks, mode_qn, ttno.charge)
    charge = det_charges[0]
    if int(charge.n) != n_elec:                       # pragma: no cover - defensive
        raise AssertionError("determinant charge disagrees with its electron count")
    if any(q != charge for q in det_charges[1:]):
        raise ValueError(
            "the determinants span more than one symmetry sector ({}), so they do not "
            "describe one state of the labelled network; expand one sector at a time"
            .format(sorted({tuple(q) for q in det_charges})))
    zero = ttno.charge.zero_like()
    w = np.full(n_roots, 1.0 / n_roots) if weights is None \
        else np.asarray(weights, dtype=float) / float(np.sum(weights))
    sqrtw = np.sqrt(w)

    parent, preorder = graph.parents(center)
    node_bits = [np.uint64(sum(1 << m for m in ttno.node_modes[u]))
                 for u in range(graph.n_nodes)]
    inside_bits = [np.uint64(0)] * graph.n_nodes

    # per-node physical placement of every determinant (sector index + offset)
    inv_perm = [np.argsort(p) for p in ttno.phys_perm]
    phys_pos: List[Tuple[np.ndarray, np.ndarray]] = []
    for u in range(graph.n_nodes):
        pos = inv_perm[u][_kron_index(masks, ttno.node_modes[u])]
        off = ttno.phys_space[u].offsets
        sec = np.searchsorted(off, pos, side="right") - 1
        phys_pos.append((sec.astype(np.int64), (pos - off[sec]).astype(np.int64)))

    bond_space: List[Optional[Space]] = [None] * graph.n_nodes
    r_sec: List[Optional[np.ndarray]] = [None] * graph.n_nodes
    r_vec: List[Optional[list]] = [None] * graph.n_nodes
    tensors: List[Optional[BlockTensor]] = [None] * graph.n_nodes
    max_dim = 0
    worst_disc = 0.0

    for u in [int(x) for x in reversed(preorder)]:
        children = [x for x in sorted(graph.neighbors(u)) if x != int(parent[u])] \
            if u != center else sorted(graph.neighbors(u))
        bits = node_bits[u]
        for c in children:
            bits |= inside_bits[c]
        inside_bits[u] = bits
        phys_sp = ttno.phys_space[u]
        p_sec, p_off = phys_pos[u]
        child_spaces = [bond_space[c] for c in children]
        k = len(children)

        # each det's row: outer product of its children's Schmidt rows, memoized by the
        # inside occupation pattern (dets sharing a pattern share the row exactly)
        pat = (masks & bits).astype(np.uint64)
        row_cache: Dict[int, np.ndarray] = {}

        def det_row(i):
            key = int(pat[i])
            vec = row_cache.get(key)
            if vec is None:
                vec = np.ones((), dtype=np.complex128)
                for c in children:
                    vec = np.multiply.outer(vec, r_vec[c][i])
                row_cache[key] = vec
            return vec

        alive = np.ones(ndet, dtype=bool)
        for c in children:
            alive &= np.array([r_vec[c][i] is not None for i in range(ndet)])

        if u == center:
            centers = []
            for r in range(n_roots):
                blocks: Dict[tuple, np.ndarray] = {}
                for i in range(ndet):
                    if not alive[i]:
                        continue
                    row = tuple(int(r_sec[c][i]) for c in children) + (int(p_sec[i]),)
                    block = blocks.get(row)
                    if block is None:
                        shape = tuple(int(sp.dims[j]) for sp, j in zip(child_spaces,
                                                                       row[:-1])) \
                            + (int(phys_sp.dims[row[-1]]),)
                        block = blocks[row] = np.zeros(shape, dtype=np.complex128)
                    block[(Ellipsis, int(p_off[i]))] += civecs[i, r] * det_row(i)
                rows = np.array(sorted(blocks), dtype=np.int64).reshape(len(blocks),
                                                                        k + 1)
                payload = [np.ascontiguousarray(blocks[tuple(int(x) for x in rr)])
                           for rr in rows]
                spaces = tuple(child_spaces) + (phys_sp,)
                signs = tuple([1] * (k + 1))
                c_t = BlockTensor(spaces, signs, charge, rows, payload)
                nrm = c_t.norm()
                if nrm <= 0.0:
                    raise ValueError("root {} vanished in the conversion — the "
                                     "determinant list does not support it".format(r))
                for b in c_t.blocks:
                    b /= nrm
                if abs(nrm - 1.0) > 1e-8:
                    log.debug("seed root %d norm %.6f after truncation", r, nrm)
                centers.append(c_t)
            state = TTNState(graph=graph, center=center, tensors=tensors,
                             centers=centers, charge=charge)
            log.debug("determinant expansion -> TTN: %d dets, %d roots, max bond "
                      "dimension %d, worst discarded weight %.2e", ndet, n_roots,
                      max_dim, worst_disc)
            return state

        # non-center node: build T[(child bonds.., phys), (outside pattern, root)]
        out_key = (masks & ~bits).astype(np.uint64)
        uniq, col_of = np.unique(out_key, return_inverse=True)
        # The subtree label of a column is the total minus the label of everything outside it
        # — the same subtraction that gives the subtree electron count, done in the group.
        in_qn = [charge - q for q in _mask_charge(uniq, mode_qn, ttno.charge)]
        qns = sorted(set(in_qn))
        col_counts = {q: sum(1 for x in in_qn if x == q) for q in qns}
        col_space = Space([(q, col_counts[q] * n_roots) for q in qns])
        # position of each unique column inside its sector: rank within its equal-label group
        col_sec = np.zeros(len(uniq), dtype=np.int64)
        col_base = np.zeros(len(uniq), dtype=np.int64)
        rank = {q: 0 for q in qns}
        for j in range(len(uniq)):
            q = in_qn[j]
            col_sec[j] = col_space.sector_index(q)
            col_base[j] = rank[q] * n_roots
            rank[q] += 1

        # The subtree matrix is the one potentially large allocation of the
        # conversion — size it exactly from the block shapes before allocating anything
        def t_shape(row):
            return tuple(int(sp.dims[jj]) for sp, jj in zip(child_spaces, row[:-2])) \
                + (int(phys_sp.dims[row[-2]]), int(col_space.dims[row[-1]]))

        rows_needed = {}
        for i in range(ndet):
            if alive[i]:
                row = tuple(int(r_sec[c][i]) for c in children) \
                    + (int(p_sec[i]), int(col_sec[int(col_of[i])]))
                if row not in rows_needed:
                    rows_needed[row] = t_shape(row)
        t_bytes = 16.0 * sum(int(np.prod(s)) for s in rows_needed.values())
        res.require("determinant->TTN subtree matrix (node {})".format(u),
                    t_bytes / 1024.0 ** 3,
                    note="{} blocks over {} determinants".format(len(rows_needed), ndet),
                    advice=["lower max_bond or the cheap CI's determinant count: the "
                            "matrix scales with (bond dims) x (outside patterns)"])

        blocks = {}
        for i in range(ndet):
            if not alive[i]:
                continue
            j = int(col_of[i])
            row = tuple(int(r_sec[c][i]) for c in children) \
                + (int(p_sec[i]), int(col_sec[j]))
            block = blocks.get(row)
            if block is None:
                block = blocks[row] = np.zeros(rows_needed[row], dtype=np.complex128)
            amp = civecs[i, :] * sqrtw
            base = int(col_base[j])
            block[(Ellipsis, int(p_off[i]),
                   slice(base, base + n_roots))] += np.multiply.outer(det_row(i), amp)
        rows = np.array(sorted(blocks), dtype=np.int64).reshape(len(blocks), k + 2)
        payload = [np.ascontiguousarray(blocks[tuple(int(x) for x in rr)])
                   for rr in rows]
        spaces = tuple(child_spaces) + (phys_sp, col_space)
        signs = tuple([1] * (k + 1)) + (-1,)
        t = BlockTensor(spaces, signs, zero, rows, payload)

        u_t, _, _, info = svd(t, tuple(range(k + 1)), tol=tol, max_bond=max_bond)
        bs = u_t.spaces[-1]
        bond_space[u] = bs
        max_dim = max(max_dim, bs.total_dim)
        worst_disc = max(worst_disc, info.discarded_weight)

        # Schmidt rows for the parent: project each det's pattern on the kept basis
        sec = np.full(ndet, -1, dtype=np.int64)
        vecs: list = [None] * ndet
        proj_cache: Dict[int, Optional[Tuple[int, np.ndarray]]] = {}
        n_in_det = _popcount((masks & bits).astype(np.uint64))
        for i in range(ndet):
            if not alive[i]:
                continue
            key = int(pat[i])
            got = proj_cache.get(key, False)
            if got is False:
                q = _mask_charge(np.asarray([masks[i] & bits], dtype=np.uint64),
                                 mode_qn, ttno.charge)[0]
                if q not in bs.qns:
                    got = None                          # pattern truncated away entirely
                else:
                    sb = bs.sector_index(q)
                    ub = u_t.find(tuple(int(r_sec[c][i]) for c in children)
                                  + (int(p_sec[i]), sb))
                    if ub is None:
                        got = None
                    else:
                        piece = np.conj(ub[(Ellipsis, int(p_off[i]), slice(None))])
                        vec = det_row(i)
                        got = (sb, np.tensordot(vec, piece, axes=(tuple(range(k)),
                                                                  tuple(range(k)))))
                proj_cache[key] = got
            if got is not None:
                sec[i], vecs[i] = got[0], got[1]
        r_sec[u] = sec
        r_vec[u] = vecs

        # the node tensor: bond into the parent's sorted-neighbor position, phys last
        perm = list(range(u_t.ndim - 1))
        perm.insert(_bond_axis(graph, u, int(parent[u])), u_t.ndim - 1)
        tensors[u] = u_t.transpose(perm)

    raise AssertionError("unreachable: the center is always processed last")


__all__ = ["TopologyGuess", "topology_from_mutual_information", "expansion_to_ttn",
           "DEFAULT_SITE_SPLIT"]
