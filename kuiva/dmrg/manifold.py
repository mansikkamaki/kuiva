"""Local multiplets and the effective Hamiltonian on the model space.

The state-count answer of the tensor-network layer. A polymetallic system's low-energy
manifold is a *product* of local multiplet spaces — thousands of states for three f ions —
and solving those as individual roots is not viable. This module inverts the problem:

1. **Local multiplet spaces.** A *site* is a connected subtree of the converged network
   (given explicitly, or read off :func:`~kuiva.dmrg.reconnect.discovered_structure`). Each
   site's ensemble reduced density matrix is diagonalized through one
   :func:`~kuiva.dmrg.block.svd` of the site's merged tensor — the same call, and therefore
   the same degenerate-group discipline, as every other truncation in this package — and its
   dominant eigenspace, cut by a **pluggable rule** (the multiplet definition is deliberately
   empirical), becomes the site's multiplet space, carried as an isometry
   ``V_k : C^{d_k} -> H_site``.
2. **The model space** is the tensor product of the site spaces, dimension ``prod d_k``.
   Model basis states are products over sites of RDM eigenvectors *in the network's global
   kron basis* (`ttno.py`'s JW convention), so every matrix element below is exact — no sign
   bookkeeping appears anywhere in this module, for the same reason as in ``guess.py``.
3. **Open-index contraction.** ``H_eff = (kron_k V_k)^dag H (kron_k V_k)`` is computed by
   sandwiching the operator with the bra and ket isometries site by site and joining the
   sites over the quotient tree — the shared-basis idea of SA-DMRG applied to a *family* of
   product states: every state shares every tensor except the open multiplet indices, so the
   whole ``prod d_k``-dimensional family costs one contraction, never ``prod d_k`` solves.
   Any operator with a term representation takes the same route
   (:func:`effective_operator`); the magnetic moments enter through
   :func:`~kuiva.dmrg.ttno.one_electron_product_terms`.

   ⚠ **The operator is re-labelled on the quotient tree, one node per site, and never
   contracted node by node through a site.** Threading the TTNO through a five-node site
   opened every operator leg of the site at once — a 44 GB dense intermediate on the
   30-spinor three-site operator, allocated raw by the block layer with no plan to refuse
   it. On the quotient tree a site *is* one node: the compiler's label pass
   (:func:`kuiva.dmrg.ttno.compile_labels`) says which operator channel every term
   flows through on each inter-site bond and which local pattern it carries on the
   site, and the site block ``V^dag W_site V`` is then one small sandwich
   ``V^dag L V`` per distinct local pattern, summed into its channel tuple with the
   term coefficients — ``n_ops x d_k^2`` complex, never ``w^2 d_site^2`` and never the
   site's merged transition tables either (the 30-spinor ten-mode node's are 13 GB;
   nothing here needs them). The join then folds leaves upward with the model legs
   dense and one parent leg open, rooted at the site that keeps the largest folded
   environment smallest; every array in it is a plain product of known dimensions,
   reserved before it exists.
4. **Diagonalize** ``H_eff`` densely — trivial at model dimensions — for the manifold
   spectrum, and hand the same matrices to the pseudospin export (kuiva/props/pseudospin.py,
   :mod:`kuiva.props.pseudospin`).

``H_eff`` is a Rayleigh–Ritz (variational) projection onto an orthonormal product basis:
every model eigenvalue bounds its exact counterpart from above (Cauchy interlacing), which
is what the tests assert *exactly*, independent of how good the product approximation is.
The construction is the zeroth (Rayleigh–Ritz) step of CORE (Morningstar & Weinstein 1996);
the perturbative/Bloch improvements are deliberately not implemented.

⚠ The multiplet cut must land on a spectral gap of the local RDM
-------------------------------------------------------------------------
The direct generalization of the state average's manifold-boundary rule, with the same
self-reinforcing failure mode: a cut through a degenerate group is refused, never rounded (Kramers pairs and
orbital multiplets are exactly what these spectra contain), and the gap at every accepted
cut is *reported*, with a warning when the singular-value ratio across it is below
:data:`MULTIPLET_GAP_RATIO_WARN`. Like the boundary diagnostic's 50 cm^-1, that threshold is a statement that
the cut is *unambiguous*, not a physical tolerance.

The ensemble loop (realised as root growth)
----------------------------------------------------------------------
"How does the ensemble target a manifold it has not solved?" Two mechanisms, both live:

* **Inter-site entanglement resolves site spaces for free.** A small root set over a
  *coupled* system spreads each site over its local multiplet (exchange mixes the local
  states), so the site RDM supports the full multiplet at small weight — the model space
  then contains states the ensemble never solved as roots. This is the production
  mechanism and the reason the construction is cheap.
* **Root growth is the fallback** for weak coupling (the ``ti2cl6_far`` limit, where
  nothing spreads the sites): when a site space cannot be resolved to the rule's
  satisfaction (:class:`UnderResolved`), :func:`solve_manifold` grows the root count —
  rounded to even for an odd electron count — and re-solves, up to ``max_roots``.
  Each outer iteration restarts from a fresh random state; warm-starting across growth is
  an unmeasured optimization, deliberately left out until it is measured.

Convergence of the loop is *stability of the model spectrum across ensembles* (two
successive outer iterations within ``outer_tol``), plus, for an explicit-dimension rule,
the requested dimensions being resolved. Non-uniform ensemble re-weighting remains open
(the ensemble-size question) and is to be settled on the ab initio ladder, not guessed here.

⚠ Site-local operators require label-contiguous sites
-----------------------------------------------------
A one-electron operator restricted to one site's modes acts on the model space as
``1 (x) A_k (x) 1`` **only when the site's mode labels are contiguous**: the JW string of
``a+_p a_q`` covers every mode between ``p`` and ``q``, and an interleaved label set makes
that string act as a non-constant parity on *other* sites (the reconnection lesson, in
operator form). The cheap-CI seeding of :mod:`kuiva.dmrg.guess` orders labels by cluster, so
physical runs satisfy this by construction; :func:`effective_model` verifies the
factorization numerically and refuses with the cause named rather than returning a plausible
wrong site operator.

Everything here is orchestration and stays Python: a handful of small
contractions per site once per model build. The arithmetic lives in
:func:`~kuiva.dmrg.block.tensordot`.

References
----------
* Effective Hamiltonians on model spaces: C. Bloch, Nucl. Phys. 6, 329 (1958),
  doi:10.1016/0029-5582(58)90116-0; J. des Cloizeaux, Nucl. Phys. 20, 321 (1960),
  doi:10.1016/0029-5582(60)90177-2. The variational product-block construction realised
  here: C. J. Morningstar, M. Weinstein (CORE), Phys. Rev. D 54, 4131 (1996),
  doi:10.1103/PhysRevD.54.4131.
* Shared-basis state ensembles: J. J. Dorando, J. Hachmann, G. K.-L. Chan, J. Chem. Phys.
  127, 084109 (2007), doi:10.1063/1.2768360.
* The multi-site anisotropic exchange picture this derives ab initio: L. F. Chibotaru,
  L. Ungur, J. Chem. Phys. 137, 064112 (2012), doi:10.1063/1.4739763.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from .block import (BlockTensor, QuantumNumber, svd, _degenerate_groups,
                    SCHMIDT_DEGENERACY_RTOL, SCHMIDT_STABILITY_RTOL)
from .graph import NetworkGraph
from .reconnect import _move_center, discovered_structure
from .sweep import TTNState, _Lab, _stack_roots, random_state, solve_ttn, SweepResult
from .ttno import (TTNO, TermTable, compile_labels, compile_ttno, node_transitions,
                   pattern_tables)

#: The local Z matrix a fermionic Jordan-Wigner string is made of (ttno.py convention).
_Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

log = get_logger(__name__)

#: Warn when the singular-value ratio across an accepted multiplet cut,
#: ``s[keep-1] / s[keep]``, is below this. Like the state-average boundary threshold, it is a
#: statement about *unambiguity*, not physics: a clean local multiplet sits orders of
#: magnitude above what it excludes, and a cut with less than one order of separation is
#: a cut whose position an equally converged run could move.
MULTIPLET_GAP_RATIO_WARN = 10.0

#: Default discarded-weight tolerance of the ``"weight"`` multiplet rule.
DEFAULT_MULTIPLET_WEIGHT_TOL = 1.0e-6

#: Relative singular value below which a site-RDM direction counts as *unresolved* for
#: the refuse-vs-grow decision: a group at this scale carries under 1e-12 of the ensemble
#: weight and is typically the eigensolver's residual, not a multiplet member. A cut
#: landing inside such a group is curable by a richer ensemble (:class:`UnderResolved`),
#: where a cut inside a *significant* degenerate group (a Kramers pair, an orbital
#: multiplet) stays a hard refusal. Found the way most numerical floors are: Davidson
#: dust at ~1e-9 of the leading value formed a six-fold "degenerate group" and tripped
#: the refusal meant for real multiplets.
MULTIPLET_RESOLUTION_RTOL = 1.0e-6


class UnderResolved(ValueError):
    """The ensemble does not support the requested multiplet space.

    Raised when a site's RDM spectrum carries fewer resolved directions (whole degenerate
    groups above the stability floor) than the cut rule needs. Distinct from a cut through
    a degenerate group — that is a *wrong request* and stays a plain ``ValueError`` —
    because this one is curable by a richer ensemble, which is exactly what
    :func:`solve_manifold`'s root growth does with it.
    """


# --- site multiplet spaces ------------------------------------------------------------------

@dataclass(eq=False)
class SiteSpace:
    """One site's local multiplet space: an isometry plus the spectrum that chose it.

    ``isometry`` has legs ``[phys(u) for u in nodes ascending, multiplet]`` with the
    multiplet leg an isometric bond (sign -1, charge 0); its bond :class:`Space` orders the
    basis states ascending by quantum number sector — ``charges`` and ``populations`` are
    aligned with that order, which is also the dense-embedding order every model matrix
    below uses. ``spectrum`` is the full merged RDM spectrum (descending singular values;
    populations are their squares), kept so a later reader can judge the cut.
    """

    nodes: Tuple[int, ...]
    orbitals: Tuple[int, ...]
    isometry: BlockTensor
    dim: int
    charges: Tuple[QuantumNumber, ...]
    populations: np.ndarray
    spectrum: np.ndarray
    gap_ratio: float

    @property
    def n_electrons(self) -> Optional[int]:
        """The site's electron count if the space is N-pure, else ``None``."""
        counts = {qn.n for qn in self.charges}
        return counts.pop() if len(counts) == 1 else None


def _walk_center(state: TTNState, target: int) -> None:
    """Gauge the canonical center to ``target`` by exact edge moves (no truncation)."""
    if state.center == target:
        return
    parent, _ = state.graph.parents(target)
    u = state.center
    while u != target:
        v = int(parent[u])
        _move_center(state, u, v)
        u = v


def _merge_site(state: TTNState, nodes: Sequence[int], weights: np.ndarray) -> "_Lab":
    """Contract a site subtree (center inside, roots stacked) into one labelled tensor.

    Result legs: boundary bonds ``("b", u, x)`` (``x`` outside), one physical leg
    ``("p", u)`` per node, and the ensemble root leg ``("r",)``.
    """
    graph = state.graph
    inside = set(int(u) for u in nodes)
    if state.center not in inside:
        raise RuntimeError("the canonical center must be inside the site")
    order = [sorted(inside)[0]]
    parent_of = {order[0]: None}
    seen = {order[0]}
    qi = 0
    while qi < len(order):
        u = order[qi]
        qi += 1
        for x in sorted(graph.neighbors(u)):
            if x in inside and x not in seen:
                seen.add(x)
                parent_of[x] = u
                order.append(x)
    if seen != inside:
        raise ValueError("site nodes {} are not connected in the tree"
                         .format(sorted(inside)))

    def labelled(u: int) -> _Lab:
        labels = [("b", u, x) for x in sorted(graph.neighbors(u))]
        if u == state.center:
            return _Lab(_stack_roots(state.centers, weights),
                        labels + [("p", u), ("r",)])
        return _Lab(state.tensors[u], labels + [("p", u)])

    merged = labelled(order[0])
    for u in order[1:]:
        p = parent_of[u]
        merged = merged.dot(labelled(u), [(("b", p, u), ("b", u, p))])
    return merged


def _choose_cut(s: np.ndarray, rule: str, *, dim: Optional[int], weight_tol: float,
                min_dim: int, max_dim: Optional[int], degeneracy_rtol: float,
                stability_rtol: float, label: str) -> int:
    """The kept dimension of one site spectrum under the pluggable rule (module docstring).

    ``s`` is the merged descending spectrum. Every candidate cut is a whole-group
    boundary; a request that would cut inside a degenerate group is refused (plain
    ``ValueError``), and one the resolved spectrum cannot support raises
    :class:`UnderResolved` so the ensemble loop can grow.
    """
    groups = _degenerate_groups(s, degeneracy_rtol)
    floor = stability_rtol * float(s[0])
    boundaries: List[int] = []
    for g in groups:
        if float(s[g].min()) <= floor:
            break
        boundaries.append(int(g[-1]) + 1)
    resolved = boundaries[-1] if boundaries else 0
    if resolved == 0:
        raise UnderResolved("site {}: no RDM eigenvalue clears the stability floor — the "
                            "ensemble does not populate this site at all".format(label))

    if rule == "dimension":
        d = int(dim)
        if d in boundaries:
            return d
        if d > resolved:
            raise UnderResolved(
                "site {}: the ensemble resolves only {} RDM directions (whole degenerate "
                "groups above the stability floor) but the rule asks for {}. A richer "
                "ensemble (more averaged roots, or coupling that spreads the site) is "
                "needed".format(label, resolved, d))
        holder = next(g for g in groups if int(g[0]) < d <= int(g[-1]))
        if float(s[holder].max()) <= MULTIPLET_RESOLUTION_RTOL * float(s[0]):
            # the straddling group is dust, not a multiplet (module constant): curable
            raise UnderResolved(
                "site {}: the cut at dimension {} lands inside a group of {} RDM "
                "directions at relative weight below {:.0e} — eigensolver residue, not "
                "resolved multiplet members. A richer ensemble is needed"
                .format(label, d, len(holder), MULTIPLET_RESOLUTION_RTOL ** 2))
        raise ValueError(
            "site {}: a multiplet space of dimension {} would cut inside a degenerate "
            "group of {} RDM eigenvalues (indices {}..{}, rtol {:.1e}) — refused, not "
            "rounded (a cut through a degenerate group is never rounded). Ask for {} or {} states"
            .format(label, d, len(holder), int(holder[0]), int(holder[-1]),
                    degeneracy_rtol, int(holder[0]), int(holder[-1]) + 1))

    if rule == "weight":
        total = float(np.sum(s ** 2))
        for b in boundaries:
            if float(np.sum(s[b:] ** 2)) <= weight_tol * total:
                return b
        raise UnderResolved(
            "site {}: even keeping all {} resolved directions leaves a discarded weight "
            "above {:.1e} — the ensemble carries weight the resolution cannot represent"
            .format(label, resolved, weight_tol))

    if rule == "gap":
        candidates = [b for b in boundaries
                      if b >= int(min_dim) and (max_dim is None or b <= int(max_dim))]
        if not candidates:
            raise UnderResolved(
                "site {}: no group-complete cut inside [{}, {}] is resolved (resolved "
                "dimension {})".format(label, min_dim,
                                       max_dim if max_dim is not None else "inf",
                                       resolved))

        def ratio(b: int) -> float:
            if b >= s.size or s[b] <= floor:
                return np.inf
            return float(s[b - 1] / s[b])

        best = max(candidates, key=lambda b: (ratio(b), -b))
        return best

    raise ValueError("unknown multiplet rule {!r}; use 'dimension', 'weight' or 'gap'"
                     .format(rule))


def site_spaces(state: TTNState, sites: Sequence[Sequence[int]], *,
                weights: Optional[Sequence[float]] = None, rule: str = "gap",
                dims=None, weight_tol: float = DEFAULT_MULTIPLET_WEIGHT_TOL,
                min_dim: int = 1, max_dim: Optional[int] = None,
                degeneracy_rtol: float = SCHMIDT_DEGENERACY_RTOL,
                stability_rtol: float = SCHMIDT_STABILITY_RTOL) -> List[SiteSpace]:
    """Extract every site's local multiplet space from a converged state.

    ``sites`` are node groups partitioning the tree (tuples of node ids, or objects with a
    ``nodes`` attribute such as :class:`~kuiva.dmrg.reconnect.SiteReport`); ``weights`` the
    equalized ensemble weights (uniform if omitted — pass ``SweepResult.weights``). ``dims`` is
    a per-site sequence or one integer, consumed by ``rule="dimension"``.

    ⚠ The state is re-gauged **in place** (exactly); no environment cache survives this,
    same as :func:`~kuiva.dmrg.reconnect.discovered_structure`.
    """
    graph = state.graph
    groups = [tuple(int(u) for u in getattr(s, "nodes", s)) for s in sites]
    flat = [u for g in groups for u in g]
    if sorted(flat) != list(range(graph.n_nodes)):
        raise ValueError("sites must partition the {} tree nodes; got {}"
                         .format(graph.n_nodes, groups))
    n_roots = len(state.centers)
    w = np.full(n_roots, 1.0 / n_roots) if weights is None \
        else np.asarray(weights, dtype=float) / float(np.sum(weights))
    if dims is not None and np.ndim(dims) == 0:
        dims = [int(dims)] * len(groups)

    spaces: List[SiteSpace] = []
    for k, nodes in enumerate(groups):
        _walk_center(state, nodes[0])
        merged = _merge_site(state, nodes, w)
        phys = [("p", u) for u in sorted(nodes)]
        rest = [l for l in merged.labels if l[0] != "p"]
        t = merged.to(phys + rest)

        _, s_sectors, _, _ = svd(t, tuple(range(len(phys))), tol=0.0,
                                 degeneracy_rtol=degeneracy_rtol,
                                 stability_rtol=stability_rtol)
        spectrum = np.sort(np.concatenate(
            [np.asarray(x, dtype=float) for x in s_sectors if x is not None]))[::-1]
        label = "{} (nodes {})".format(k, nodes)
        keep = _choose_cut(spectrum, rule, dim=None if dims is None else dims[k],
                           weight_tol=weight_tol, min_dim=min_dim, max_dim=max_dim,
                           degeneracy_rtol=degeneracy_rtol,
                           stability_rtol=stability_rtol, label=label)
        gap_ratio = np.inf if keep >= spectrum.size or spectrum[keep] <= 0.0 \
            else float(spectrum[keep - 1] / spectrum[keep])
        if np.isfinite(gap_ratio) and gap_ratio < MULTIPLET_GAP_RATIO_WARN:
            log.warning("site %s: the multiplet cut at dimension %d sits on a spectral "
                        "gap of only %.2f (singular-value ratio; populations %.1f x). "
                        "The cut is ambiguous — an equally converged ensemble could move "
                        "it ", label, keep, gap_ratio,
                        gap_ratio ** 2)

        iso, s_kept, _, info = svd(t, tuple(range(len(phys))), tol=0.0, max_bond=keep,
                                   degeneracy_rtol=degeneracy_rtol,
                                   stability_rtol=stability_rtol)
        if info.bond_dim != keep:                       # pragma: no cover - guarded above
            raise AssertionError("multiplet cut moved between spectra ({} vs {})"
                                 .format(info.bond_dim, keep))
        bond = iso.spaces[-1]
        charges: List[QuantumNumber] = []
        pops: List[float] = []
        for j, qn in enumerate(bond.qns):
            vals = s_kept[j]
            charges.extend([qn] * int(bond.dims[j]))
            pops.extend(float(x) ** 2 for x in vals)
        orbitals = tuple(sorted(x for u in nodes for x in graph.contents[u]))
        spaces.append(SiteSpace(nodes=tuple(sorted(nodes)), orbitals=orbitals,
                                isometry=iso, dim=keep, charges=tuple(charges),
                                populations=np.asarray(pops), spectrum=spectrum,
                                gap_ratio=gap_ratio))
    return spaces


# --- the open-index contraction -------------------------------------------------------------

def _quotient_graph(graph: NetworkGraph, spaces: Sequence[SiteSpace]) -> NetworkGraph:
    """The tree of sites: one node per site carrying the site's modes, one edge per tree
    edge that crosses a site boundary."""
    site_of: Dict[int, int] = {}
    for k, sp in enumerate(spaces):
        for u in sp.nodes:
            site_of[int(u)] = k
    edges = sorted({(min(site_of[a], site_of[b]), max(site_of[a], site_of[b]))
                    for a, b in graph.edges if site_of[a] != site_of[b]})
    return NetworkGraph(len(spaces), edges, [tuple(sp.orbitals) for sp in spaces])


def _quotient_root(qgraph: NetworkGraph, dims: Sequence[int]) -> int:
    """The site whose rooting keeps the largest folded environment smallest: the model
    dimension of every subtree hanging off it is minimized over the choice of root."""
    best = None
    for r in range(qgraph.n_nodes):
        parent, preorder = qgraph.parents(r)
        sub = [1.0] * qgraph.n_nodes
        for u in reversed([int(x) for x in preorder]):
            sub[u] *= float(dims[u])
            if u != r:
                sub[int(parent[u])] *= sub[u]
        worst = max((sub[u] for u in range(qgraph.n_nodes) if u != r), default=1.0)
        if best is None or (worst, r) < best:
            best = (worst, r)
    return best[1]


def _site_vector(space: SiteSpace, ttno: TTNO, ctx, k: int) -> np.ndarray:
    """The site isometry as a dense ``(d_site, d_k)`` matrix over the quotient node's
    sector-sorted physical basis.

    The isometry's legs are the site's nodes' physical spaces (each in that node's sector
    order); the quotient node's basis is the kron over *all* the site's modes ascending,
    re-sorted into sectors. Both permutations are known, so this is index bookkeeping
    and no arithmetic.
    """
    nodes = [int(u) for u in sorted(space.nodes)]
    q_modes = list(ctx.node_modes[k])
    q_dims = [ctx.bases[m].dim for m in q_modes]
    stride = {}
    acc = 1
    for m, d in zip(reversed(q_modes), reversed(q_dims)):
        stride[m] = acc
        acc *= d
    inv_q = np.argsort(ctx.phys[k][1])                     # site kron index -> sector pos
    partial = []
    for u in nodes:
        modes_u = list(ttno.node_modes[u])
        dims_u = tuple(int(d) for d in ttno.mode_dims[u])
        perm_u = ttno.phys_perm[u]                          # sector pos -> node kron idx
        part = np.zeros(perm_u.size, dtype=np.int64)
        if modes_u:
            digits = np.unravel_index(perm_u, dims_u)
            for m, dig in zip(modes_u, digits):
                part += dig.astype(np.int64) * stride[m]
        partial.append(part)
    site_kron = np.zeros((1,), dtype=np.int64)
    for part in partial:
        site_kron = (site_kron[:, None] + part[None, :]).reshape(-1)
    dense = space.isometry.to_dense().reshape(-1, space.dim)
    d_site = int(np.prod(q_dims)) if q_dims else 1
    v = np.zeros((d_site, space.dim), dtype=np.complex128)
    v[inv_q[site_kron], :] = dense
    return v


def _site_payload(ctx, k: int, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``V^dag W_site V`` per operator-channel tuple of quotient node ``k``:
    ``(ops, payload)`` with ``ops`` the ``(n_ops, n_legs)`` global channel indices in W
    leg order (the root's dim-1 completed channel dropped) and ``payload[e, mc, m]`` the
    ``(d_k, d_k)`` block.

    One sandwich per distinct local pattern (``L V`` is a gather, since a pattern has at
    most one nonzero per column), then each unit transition takes its pattern's block
    and each coefficient transition the coefficient-weighted sum of its terms' — a sparse
    ``(transitions x patterns)`` coefficient matrix times the pattern blocks. Nothing of
    size ``w^2``, ``d_site^2`` or the node's merged transition tables is formed.
    """
    from scipy import sparse as sp

    tr = node_transitions(k, ctx)
    pat_ptr, pat_rows, pat_cols, pat_vals, alloc = pattern_tables(k, ctx, tr)
    try:
        d_site, d_k = v.shape
        n_pat = tr.nnz_pat.size
        res.require("site {} pattern sandwiches".format(k),
                    res.array_gb((n_pat, d_k, d_k), np.complex128)
                    + res.array_gb((d_site, d_k), np.complex128),
                    note="{} local patterns x ({}, {}) plus one (d_site, d_k) product".format(
                        n_pat, d_k, d_k),
                    advice=["a smaller site, or a lower multiplet dimension"])
        m_pat = np.empty((n_pat, d_k, d_k), dtype=np.complex128)
        lv = np.empty((d_site, d_k), dtype=np.complex128)
        for p in range(n_pat):
            lo, hi = int(pat_ptr[p]), int(pat_ptr[p + 1])
            lv[:] = 0.0
            np.add.at(lv, pat_rows[lo:hi], pat_vals[lo:hi, None] * v[pat_cols[lo:hi]])
            m_pat[p] = v.conj().T @ lv
        n_unit = tr.unit_keys.shape[0]
        n_coeff = tr.coeff_keys.shape[0]
        keys = np.vstack([tr.unit_keys, tr.coeff_keys])
        n_keys = keys.shape[0]
        res.require("site {} block payload".format(k),
                    res.array_gb((n_keys, d_k, d_k), np.complex128),
                    note="{} operator-channel tuples x ({}, {})".format(n_keys, d_k, d_k),
                    advice=["reduce the site's multiplet dimension"])
        payload = np.empty((n_keys, d_k, d_k), dtype=np.complex128)
        payload[:n_unit] = m_pat[tr.unit_pat]
        if n_coeff:
            c_terms = tr.coeff_idx
            weights = sp.csr_matrix((ctx.table.coeff[c_terms],
                                     (tr.coeff_inv, tr.pat_of_term[c_terms])),
                                    shape=(n_coeff, n_pat))
            payload[n_unit:] = np.asarray(weights @ m_pat.reshape(n_pat, -1)).reshape(
                n_coeff, d_k, d_k)
    finally:
        res.BUDGET.release(alloc)
        res.BUDGET.release(tr.alloc)
    # channel tuples: global bond-space index per leg, in W leg order
    is_root = k == ctx.root
    legs = []
    if not is_root:
        legs.append((ctx.bond_space[k], ctx.pos_sector[k], ctx.pos_offset[k], 0))
    for j, c in enumerate(ctx.children[k]):
        legs.append((ctx.bond_space[c], ctx.pos_sector[c], ctx.pos_offset[c], 1 + j))
    ops = np.zeros((n_keys, len(legs)), dtype=np.int64)
    for i, (space, psec, poff, col) in enumerate(legs):
        lab = keys[:, col]
        ops[:, i] = space.offsets[psec[lab]] + poff[lab]
    return ops, payload


def _fold_sites(ctx, qgraph: NetworkGraph, blocks: List[Tuple[np.ndarray, np.ndarray]],
                dims: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
    """Join the site blocks leaves-upward over the quotient tree: each site's folded
    environment is dense in its subtree's model legs and open on its parent channel,
    ``(w_parent, D_sub, D_sub)`` with ``D_sub`` the subtree's model dimension; the root's
    fold is the model matrix itself. Returns it with the product index in fold order
    (a site, then its children's subtrees), plus that order."""
    _, preorder = qgraph.parents(ctx.root)
    children = ctx.children
    env: Dict[int, np.ndarray] = {}
    order_of: Dict[int, List[int]] = {}
    for s in reversed([int(x) for x in preorder]):
        ops, payload = blocks[s]
        kids = list(children[s])
        my_order = [s]
        for c in kids:
            my_order += order_of[c]
        order_of[s] = my_order
        d_sub = int(np.prod([dims[t] for t in my_order]))
        is_root = s == ctx.root
        col = 0
        if not is_root:
            w_par = ctx.bond_space[s].total_dim
            gb = res.array_gb((w_par, d_sub, d_sub), np.complex128)
            res.require("site {} folded environment".format(s), gb,
                        note="({}, {}, {}): the parent channel x the subtree's model "
                             "dimension squared".format(w_par, d_sub, d_sub),
                        advice=["reduce a site's multiplet dimension: the subtree's model "
                                "dimension is their product",
                                "the join is rooted at the site keeping this smallest"])
            out = np.zeros((w_par, d_sub, d_sub), dtype=np.complex128)
            col = 1
            groups = ops[:, 0]
        else:
            out = np.zeros((1, d_sub, d_sub), dtype=np.complex128)
            groups = np.zeros(ops.shape[0], dtype=np.int64)
        # the einsum over one parent-channel group: payload (e a b), each child's
        # environment at the entry's channel (e c d), output (a c.. b d..) — one direct
        # loop into the output slice, no intermediate of the product's size
        letters = "cdfghijklmnopqrstuvwxyz"
        subs = ["eab"]
        bra_out, ket_out = "a", "b"
        for j in range(len(kids)):
            lc, lk = letters[2 * j], letters[2 * j + 1]
            subs.append("e" + lc + lk)
            bra_out += lc
            ket_out += lk
        expr = ",".join(subs) + "->" + bra_out + ket_out
        for g in np.unique(groups):
            sel = np.nonzero(groups == g)[0]
            operands = [payload[sel]]
            for j, c in enumerate(kids):
                operands.append(env[c][ops[sel, col + j]])
            block = np.einsum(expr, *operands, optimize=False)
            out[int(g)] = block.reshape(d_sub, d_sub)
        for c in kids:
            del env[c]
        env[s] = out
    root_env = env[ctx.root]
    return root_env[0], order_of[ctx.root]


def model_gb(spaces: Sequence[SiteSpace]) -> float:
    """Exact size [GB] of one dense model matrix (exact sizing function).

    ``(prod d_k)^2`` complex entries — what :func:`effective_operator` materializes. Pinned
    two-sided against a real array's ``nbytes`` in the tests.
    """
    d = 1.0
    for sp in spaces:
        d *= float(sp.dim)
    return res.array_gb((d, d), np.complex128)


def effective_operator_from_terms(terms, spaces: Sequence[SiteSpace], ttno: TTNO, *,
                                  bases=None) -> np.ndarray:
    """:func:`effective_operator` for an operator given as product terms (a
    :class:`~kuiva.dmrg.ttno.TermTable` or a term sequence): compiled on the quotient
    tree of the sites, sandwiched per site, folded over the tree (module docstring).
    ``ttno`` supplies the graph and the node conventions the isometries were extracted
    in; ``bases`` defaults to its."""
    covered = sorted(u for sp in spaces for u in sp.nodes)
    if covered != list(range(ttno.graph.n_nodes)):
        raise ValueError("site spaces must cover every node of the operator's tree")
    gb = model_gb(spaces)
    res.require("effective model operator ({} sites)".format(len(spaces)), gb,
                note="dense ({0}, {0}) over the multiplet product basis".format(
                    int(np.prod([sp.dim for sp in spaces]))),
                advice=["reduce a site's multiplet dimension: the matrix scales as "
                        "(prod d_k)^2"])
    table = TermTable.coerce(terms)
    dims = [int(sp.dim) for sp in spaces]
    qgraph = _quotient_graph(ttno.graph, spaces)
    root = _quotient_root(qgraph, dims)
    ctx = compile_labels(qgraph, table, bases=ttno.bases if bases is None else bases,
                         root=root)
    blocks = []
    for k, sp in enumerate(spaces):
        blocks.append(_site_payload(ctx, k, _site_vector(sp, ttno, ctx, k)))
    folded, order = _fold_sites(ctx, qgraph, blocks, dims)
    n = len(spaces)
    fold_dims = [dims[t] for t in order]
    t = folded.reshape(fold_dims + fold_dims)
    inv = [order.index(k) for k in range(n)]
    dense = t.transpose(inv + [n + i for i in inv])
    d = int(np.prod(dims))
    return np.ascontiguousarray(dense.reshape(d, d))


def effective_operator(ttno: TTNO, spaces: Sequence[SiteSpace]) -> np.ndarray:
    """Contract an operator with the multiplet indices left open (the open-index contraction).

    Returns the dense ``(prod d_k, prod d_k)`` matrix of the operator over the model
    product basis, sites in the order given (site 0 slowest, C order; within a site the
    :class:`SiteSpace` basis order). ``ttno`` must be compiled on the graph the isometries
    were extracted from and carry its term table (every compiled or template-filled
    operator does): the contraction recompiles it on the quotient tree of the sites.
    """
    if ttno.terms is None:
        raise ValueError("the operator carries no term table; the model-space contraction "
                         "compiles it on the quotient tree of the sites and needs the "
                         "terms it was built from (compile_ttno and TTNOTemplate.fill "
                         "both record them)")
    return effective_operator_from_terms(ttno.terms, spaces, ttno, bases=ttno.bases)


# --- the effective model --------------------------------------------------------------------

@dataclass(eq=False)
class EffectiveModel:
    """``H_eff`` and companion operators on the local-multiplet product space.

    ``product_charges`` carries each product basis state's total particle number; when site
    spaces mix N sectors the model space contains unphysical total-N combinations, and
    :meth:`spectrum`/:meth:`eigenstates` restrict to the physical sector by default.
    Energies exclude ``e_core``, like everything in this package.
    """

    sites: Tuple[SiteSpace, ...]
    h_eff: np.ndarray
    operators: Dict[str, np.ndarray]
    site_operators: Dict[str, List[np.ndarray]]
    n_elec: int
    product_charges: np.ndarray

    @property
    def dims(self) -> Tuple[int, ...]:
        return tuple(sp.dim for sp in self.sites)

    @property
    def model_dim(self) -> int:
        return int(np.prod(self.dims))

    def sector(self, sector=None) -> np.ndarray:
        """Product-state indices of a total-N sector.

        ``None`` (default) is the physical sector ``N = n_elec``; ``"all"`` is the whole
        model space — the right choice for a model Hamiltonian whose conserved ``N`` is a
        magnetization rather than an electron count; an integer selects that ``N``.
        """
        if sector == "all":
            return np.arange(self.model_dim)
        n = self.n_elec if sector is None else int(sector)
        return np.nonzero(self.product_charges == n)[0]

    def hermiticity_error(self) -> float:
        return float(np.max(np.abs(self.h_eff - self.h_eff.conj().T)))

    def eigenstates(self, sector=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(energies, vectors, indices)`` of one sector's effective spectrum.

        ``vectors[:, i]`` is the i-th eigenstate over the product basis rows ``indices``.
        """
        idx = self.sector(sector)
        if idx.size == 0:
            raise ValueError("no product state carries N = {}; the site spaces do not "
                             "reach the physical sector".format(
                                 self.n_elec if sector is None else sector))
        h = self.h_eff[np.ix_(idx, idx)]
        e, v = np.linalg.eigh(0.5 * (h + h.conj().T))
        return e, v, idx

    def spectrum(self, sector=None) -> np.ndarray:
        return self.eigenstates(sector)[0]

    def operator_in_eigenbasis(self, name: str, sector=None) -> np.ndarray:
        """A model operator transformed to the effective eigenstate basis."""
        _, v, idx = self.eigenstates(sector)
        a = self.operators[name][np.ix_(idx, idx)]
        return v.conj().T @ a @ v

    def report(self, logger=None) -> None:
        logger = logger or log
        out.subsection(logger, "local-multiplet model space")
        out.entries(logger, [
            ("sites", len(self.sites)),
            ("model dimension", self.model_dim,
             "", " x ".join(str(d) for d in self.dims)),
            ("physical sector (N = {})".format(self.n_elec), int(self.sector().size)),
            ("H_eff hermiticity", self.hermiticity_error(), "Eh", "", "{:.2e}"),
        ])
        table = out.Table(logger, [out.Column("site", "{:d}", 5),
                                   out.Column("dim", "{:d}", 5),
                                   out.Column("N", "{:s}", 6),
                                   out.Column("gap ratio", "{:s}", 10),
                                   out.Column("orbitals", "{:s}", 28)])
        table.start()
        for k, sp in enumerate(self.sites):
            n = sp.n_electrons
            gap = "exact" if not np.isfinite(sp.gap_ratio) \
                else "{:.1f}".format(sp.gap_ratio)
            table.row(k, sp.dim, "mixed" if n is None else str(n), gap,
                      " ".join(str(x) for x in sp.orbitals))
        table.end("gap ratio: singular-value separation at the multiplet cut; "
                  "'exact' means nothing was discarded")


def _site_factor(a: np.ndarray, dims: Sequence[int], k: int, name: str) -> np.ndarray:
    """Extract ``A_k`` from a model matrix known to be ``1 (x) A_k (x) 1`` — verified.

    The verification is the point (module docstring): with non-contiguous site labels the
    JW strings make the ``1``s false, and returning the slice anyway would be a plausible
    wrong site operator.
    """
    dims = tuple(int(d) for d in dims)
    d_k = dims[k]
    t = a.reshape(dims + dims)
    sl = [0] * len(dims) + [0] * len(dims)
    sl[k] = slice(None)
    sl[len(dims) + k] = slice(None)
    factor = np.ascontiguousarray(t[tuple(sl)])
    other = 1
    for i, d in enumerate(dims):
        if i != k:
            other *= d
    norm_ok = abs(float(np.linalg.norm(a)) ** 2
                  - other * float(np.linalg.norm(factor)) ** 2) \
        <= 1e-10 * max(float(np.linalg.norm(a)) ** 2, 1e-300)
    probe_ok = True
    if other > 1:
        rng = np.random.default_rng(0)
        idx = [int(rng.integers(d)) for d in dims]
        sl2 = list(idx) + list(idx)
        sl2[k] = slice(None)
        sl2[len(dims) + k] = slice(None)
        probe_ok = np.allclose(t[tuple(sl2)], factor, atol=1e-10)
    if not (norm_ok and probe_ok):
        raise ValueError(
            "operator {!r} restricted to site {} does not factor as 1 (x) A (x) 1 over "
            "the model basis. The usual cause is a site whose mode labels are not "
            "contiguous: the Jordan-Wigner string of a site-local a+_p a_q then acts as a "
            "non-constant parity on other sites (the reconnection JW lesson). Relabel the "
            "modes so each site is a contiguous label range (the cheap-CI seeding "
            "this by construction)".format(name, k))
    return factor


def effective_model(ttno: TTNO, state: TTNState, sites=None, *,
                    weights: Optional[Sequence[float]] = None,
                    rule: str = "gap", dims=None,
                    weight_tol: float = DEFAULT_MULTIPLET_WEIGHT_TOL,
                    min_dim: int = 1, max_dim: Optional[int] = None,
                    operators: Optional[Dict[str, Sequence]] = None,
                    site_local: Optional[Sequence[str]] = None,
                    bases=None, n_elec: Optional[int] = None,
                    report: bool = True) -> EffectiveModel:
    """Site spaces + open-index contraction in one call (site spaces, then the open-index contraction).

    ``sites`` defaults to the discovered structure of the converged state.
    ``operators`` maps names to *term lists* (:func:`~kuiva.dmrg.ttno.
    one_electron_product_terms` output, or model-Hamiltonian terms); each is compiled on the
    state's graph and contracted with the same isometries, and its **site-local part**
    (the terms supported inside one site) is additionally reduced to a per-site
    ``(d_k, d_k)`` matrix — what the pseudospin labelling consumes.

    ``site_local`` names which operators get that per-site reduction; ``None`` (the default)
    is all of them, which is what every caller before it existed got. It costs one further
    model-space contraction per operator **per site**, so an operator nothing reduces per site
    — a delocalized one-electron operator written whole, such as a nucleus's hyperfine field —
    is cheaper and clearer named out. An unknown name is refused rather than ignored: a typo
    would otherwise silently restore the default for the operator it was meant to exclude.

    ⚠ **What the reduction is, and when it is complete.** It is a *projection*: only the terms
    supported inside one site survive it. Where every site space sits in **one particle-number
    sector** the dropped inter-site terms of a one-electron operator have identically zero
    matrix elements on the model space — they move an electron between sites, and no pair of
    charge-pure product states differs that way — so the per-site parts sum back to the model
    operator exactly (measured to 1e-15) and skipping the reduction loses nothing but the
    table. Where a site space **mixes** sectors they do not vanish, and the per-site matrices
    are then an incomplete account of the operator that is still Hermitian and still plausible.
    """
    if ttno.graph != state.graph:
        raise ValueError("the TTNO and the state live on different trees")
    if sites is None:
        sites = [s.nodes for s in discovered_structure(state, weights=weights,
                                                       logger=log).sites]
    with timer("multiplet site spaces"):
        spaces = site_spaces(state, sites, weights=weights, rule=rule, dims=dims,
                             weight_tol=weight_tol, min_dim=min_dim, max_dim=max_dim)
    with timer("effective Hamiltonian contraction"):
        h_eff = effective_operator(ttno, spaces)

    site_charges = [np.array([qn.n for qn in sp.charges], dtype=np.int64)
                    for sp in spaces]
    total = np.zeros((1,), dtype=np.int64)
    for c in site_charges:
        total = (total[:, None] + c[None, :]).reshape(-1)

    model_ops: Dict[str, np.ndarray] = {}
    site_ops: Dict[str, List[np.ndarray]] = {}
    dims_t = tuple(sp.dim for sp in spaces)
    op_bases = ttno.bases if bases is None else bases
    wanted = None if site_local is None else set(str(n) for n in site_local)
    if wanted is not None:
        unknown = sorted(wanted - set(operators or {}))
        if unknown:
            raise ValueError(
                "site_local names {} which is not an operator of this model ({}); a name "
                "that is ignored would silently give the operator it was meant to exclude "
                "the default per-site reduction".format(unknown,
                                                        ", ".join(sorted(operators or {}))
                                                        or "none"))
    for name, terms in (operators or {}).items():
        table = TermTable.coerce(terms)
        with timer("effective operator {}".format(name)):
            model_ops[name] = effective_operator_from_terms(table, spaces, ttno,
                                                            bases=op_bases)
        if wanted is not None and name not in wanted:
            continue
        locals_: List[np.ndarray] = []
        for k, sp in enumerate(spaces):
            inside = set(sp.orbitals)
            # ⚠ a term whose *operator content* sits inside this site but whose support
            # leaks outside is a Jordan-Wigner string crossing another site (module
            # docstring): the site-local operator then cannot be represented on the site
            # factor at all, and silently dropping the term would corrupt the M
            # labelling downstream. Refused, with the cause named.
            crossing = table.crossing_terms(inside)
            if crossing.size:
                t = table[int(crossing[0])]
                core = sorted(m for m, mat in zip(t.modes, t.mats)
                              if not (mat.shape == (2, 2) and np.array_equal(mat, _Z2)))
                raise ValueError(
                    "operator {!r} has a term acting on site {} (modes {}) whose "
                    "Jordan-Wigner string crosses other sites (support {}): the "
                    "site-local operator cannot be represented on the site factor. "
                    "Relabel the modes so each site is a contiguous label range "
                    "(the cheap-CI seeding does this by construction)"
                    .format(name, k, core, sorted(t.modes)))
            mine = table.restricted_to(inside)
            if len(mine) == 0:
                locals_.append(np.zeros((sp.dim, sp.dim), dtype=np.complex128))
                continue
            locals_.append(_site_factor(
                effective_operator_from_terms(mine, spaces, ttno, bases=op_bases),
                dims_t, k, name))
        site_ops[name] = locals_

    model = EffectiveModel(sites=tuple(spaces), h_eff=h_eff, operators=model_ops,
                           site_operators=site_ops,
                           n_elec=state.charge.n if n_elec is None else int(n_elec),
                           product_charges=total)
    if report:
        model.report()
    return model


# --- the ensemble loop ----------------------------------------------------------------------

@dataclass(eq=False)
class ManifoldResult:
    """Outcome of :func:`solve_manifold`. Energies exclude ``e_core``."""

    model: EffectiveModel
    sweep: SweepResult
    converged: bool                        #: model spectrum stable across ensembles
    n_outer: int
    n_roots: int                           #: roots the final network solve averaged
    history: List[dict]                    #: per outer iteration: roots, action, spectrum


def _next_roots(n_roots: int, n_elec: int, max_roots: int, grow_factor: float) -> int:
    grown = max(n_roots + 1, int(np.ceil(n_roots * float(grow_factor))))
    if n_elec % 2 == 1 and grown % 2 == 1:
        grown += 1                          # an odd count splits a Kramers pair
    return min(int(max_roots), grown)


def solve_manifold(terms, graph: NetworkGraph, n_elec: int, *, bases=None,
                   sites=None, rule: str = "gap", dims=None,
                   weight_tol: float = DEFAULT_MULTIPLET_WEIGHT_TOL,
                   min_dim: int = 1, max_dim: Optional[int] = None,
                   operators: Optional[Dict[str, Sequence]] = None,
                   site_local: Optional[Sequence[str]] = None,
                   n_roots: int = 2, max_roots: int = 64, grow_factor: float = 2.0,
                   max_outer: int = 6, outer_tol: float = 1.0e-6,
                   max_bond: Optional[int] = None, trunc_tol: float = 0.0,
                   conv_tol: float = 1.0e-9, davidson_tol: float = 1.0e-8,
                   max_sweeps: int = 25, boundary_check: int = 0,
                   on_split: str = "raise", ttno_root: int = 0,
                   rng: Optional[np.random.Generator] = None) -> ManifoldResult:
    """The ensemble self-consistency loop (module docstring: the realised form).

    Solves the network state-averaged over ``n_roots`` roots, extracts the site multiplet
    spaces, builds and diagonalizes ``H_eff``, and iterates — growing the root count when
    a site space is :class:`UnderResolved` — until the model spectrum is stable across two
    successive ensembles within ``outer_tol`` (and, for ``rule="dimension"``, the
    requested dimensions are resolved, which then suffices on its own). Fixed topology by
    design: run :func:`~kuiva.dmrg.reconnect.solve_adaptive` first and pass the
    discovered graph if the topology is in question.
    """
    terms = TermTable.coerce(terms)
    rng = rng if rng is not None else np.random.default_rng()
    ttno = compile_ttno(graph, terms, bases=bases, root=ttno_root)
    cap = 10 ** 9 if max_bond is None else int(max_bond)

    table = out.Table(log, [out.col_iter("outer"), out.col_count("roots", 6),
                            out.col_count("model", 6), out.col_energy("E0_eff [Eh]"),
                            out.col_sci("drift"), out.Column("action", "{:s}", 14)])
    table.start("local-multiplet ensemble loop ({} rule)".format(rule))

    history: List[dict] = []
    prev: Optional[np.ndarray] = None
    model = sweep = None
    roots = int(n_roots)
    converged = False
    with timer("manifold ensemble loop"):
        for outer in range(1, max_outer + 1):
            state = random_state(ttno, n_elec, cap, n_roots=roots, rng=rng)
            # The memory plan on the first pass only: the ensemble loop re-solves the
            # same manifold at the same cap, so every later table would be the first one
            # again, and INFO is the output file.
            sweep = solve_ttn(ttno, state, max_sweeps=max_sweeps, conv_tol=conv_tol,
                              trunc_tol=trunc_tol, max_bond=max_bond, n_elec=n_elec,
                              boundary_check=boundary_check,
                              davidson_tol=davidson_tol, on_split=on_split,
                              memory_plan=outer == 1)
            try:
                model = effective_model(ttno, state, sites, weights=sweep.weights,
                                        rule=rule, dims=dims, weight_tol=weight_tol,
                                        min_dim=min_dim, max_dim=max_dim,
                                        operators=operators, site_local=site_local,
                                        bases=bases, n_elec=n_elec, report=False)
            except UnderResolved as exc:
                if roots >= max_roots:
                    table.end("under-resolved at the root cap")
                    raise UnderResolved(
                        "{} — and the root count is already at max_roots = {}. Raise "
                        "max_roots, lower the requested dimensions, or loosen the rule"
                        .format(exc, max_roots))
                history.append({"n_roots": roots, "action": "grow", "reason": str(exc)})
                table.row(outer, roots, 0, float("nan"), float("nan"), "grow roots")
                log.debug("manifold iteration %d under-resolved: %s", outer, exc)
                roots = _next_roots(roots, n_elec, max_roots, grow_factor)
                continue

            spec = model.spectrum()
            rel = spec - spec[0]
            drift = float("nan")
            action = "stability check"
            if prev is not None:
                m = min(prev.size, rel.size)
                drift = float(np.max(np.abs(rel[:m] - prev[:m]))) if m else 0.0
                if drift < outer_tol:
                    converged = True
                    action = "converged"
            elif rule == "dimension":
                # the spaces are exactly what was asked for; stability iterations are
                # the gap/weight rules' mechanism (module docstring)
                converged = True
                action = "resolved"
            history.append({"n_roots": roots, "action": action,
                            "spectrum": [float(x) for x in rel]})
            table.row(outer, roots, model.model_dim, float(spec[0]), drift, action)
            if converged:
                break
            prev = rel
            if roots < max_roots:
                roots = _next_roots(roots, n_elec, max_roots, grow_factor)
    table.end("model spectrum {} across ensembles".format(
        "stable" if converged else "NOT stable"))
    if not converged:
        log.warning("the manifold loop did not reach a stable model spectrum in %d outer "
                    "iterations (last drift vs outer_tol %.1e); the returned model is "
                    "the last iterate", max_outer, outer_tol)
    model.report()
    return ManifoldResult(model=model, sweep=sweep, converged=converged,
                          n_outer=len(history), n_roots=roots, history=history)


__all__ = ["SiteSpace", "EffectiveModel", "ManifoldResult", "UnderResolved",
           "site_spaces", "effective_operator", "effective_operator_from_terms",
           "effective_model", "solve_manifold",
           "model_gb", "MULTIPLET_GAP_RATIO_WARN", "DEFAULT_MULTIPLET_WEIGHT_TOL",
           "MULTIPLET_RESOLUTION_RTOL"]
