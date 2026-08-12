"""The Hamiltonian as a tree tensor network operator.

One operator object for the whole network: a TTNO whose node tensors are
:class:`~kuiva.dmrg.sparse.SparseW`\\ s, built by a symbolic compiler from a sum of
operator-product terms. The ab initio Hamiltonian enters through
:func:`hamiltonian_product_terms` (complex integrals with **4-fold permutational symmetry
only**, never assume 8-fold); Tier-3 model spin Hamiltonians enter through the same
compiler as generic :class:`ProductTerm` sums — a deliberate test seam (the structure
machinery can be driven by Hamiltonians with *known* exchange graphs, no integrals
involved), documented as such rather than hidden.

Conventions (fixed here; the sweep and every environment depend on them)
------------------------------------------------------------------------
* **Fermions are Jordan-Wigner transformed with the global ascending mode order** — the
  *same* convention as :mod:`kuiva.ci.strings` (determinants are ordered products with
  ascending spinor index, sign ``(-1)^(occupied below p)``), which is what makes the dense
  TTNO equal ``ci.strings.hamiltonian_matrix`` element for element with no phase fixups.
  The JW ordering is a property of the *modes*, not of the tree: correctness never depends
  on the topology; only the compressed bond dimensions do (a tree laid out against the mode
  order pays in operator bond dimension, never in errors).
* **The Hamiltonian convention** is ``H = sum_pq h_pq a+_p a_q + 1/2 sum_pqrs (pq|rs)
  a+_p a+_r a_s a_q`` with ``(pq|rs)`` in chemists' notation (``ci/strings.py``).
  :func:`ttno_from_cas_integrals` consumes the active-space quantities of ``CASIntegrals``
  duck-typed (``h_active_effective()``, ``active_eri()``) — ⚠ ``kuiva.dmrg`` never imports
  ``kuiva.mcscf``, because ``mcscf`` is the *consumer* of this layer and the
  dependency must run one way. ⚠ The TTNO excludes ``e_core``, exactly as ``SigmaOperator``
  does; the solver adds it to reported energies.
* **W-tensor leg order** is ``[parent-op, child-ops (children ascending), phys-out,
  phys-in]`` with signs ``(-1, +1.., +1, -1)`` and charge 0; an operator-leg state's
  quantum number is the particle number its flowing operator adds to the subtree. The root
  tensor's parent leg has dimension 1 (the completed-Hamiltonian channel), so one code path
  serves every node.
* **A node's physical space is the kron product of its modes in ascending mode order**
  (first mode slowest, C order), re-sorted into quantum-number sectors for the block
  structure; the kron<->sector permutation is stored on the TTNO. A node with no modes has
  a trivial dim-1 physical space, so branching tensors are nothing special.

How the compiler works (and why it is exact)
--------------------------------------------
Every term is first reduced to a **product of one local matrix per mode** (for fermions the
JW mapping does this, absorbing all signs; ``Z`` factors on in-between modes are part of the
term's support). For each tree bond, a term is assigned a **content label**: the identity
channel (no support inside the subtree), the completed channel ``H`` (all support inside), a
*normal* state (the flowing operator is the product of the inside factors, labelled by
them), or a *complementary* state (labelled by the **outside** factors, with the coefficient
folded into the flowing sum — the complementary-operator idea of White & Martin). Labels are
complete descriptions of operator content, so distinct terms sharing a label genuinely share
the state, which is the entire compression: the ab initio Hamiltonian compiles to the
classic ``O(n^2)`` operator bond dimension instead of the ``O(n^4)`` term count. The
normal/complementary choice is per term per bond — fewer non-``Z`` factors wins, ties go to
the smaller subtree — and the label kind changes **monotonically** along the root-ward path
(the inside support only grows), which is what makes the coefficient rule below well
defined.

A term's coefficient is attached **exactly once**: at the unique node where its outgoing
state first becomes coefficient-carrying (``H`` or complementary) while no incoming state
is. All other transitions have unit weight and are label-determined, so they are written
once and shared by every term that flows through them. Cross-talk between terms sharing
states is impossible *because* labels are content-complete: any path through the state
graph reconstructs a well-defined operator product, and each completion entry sums exactly
the coefficients of the terms whose content it completes.

Everything here is orchestration and stays Python: the compiler runs once per
integral set, and the hot object it produces is consumed by the sweep, whose contraction
driver is the named port candidate — not this.

References
----------
* Complementary operators for the ab initio Hamiltonian: T. Xiang, Phys. Rev. B 53, R10445
  (1996), doi:10.1103/PhysRevB.53.R10445; S. R. White, R. L. Martin, J. Chem. Phys. 110,
  4127 (1999), doi:10.1063/1.478522.
* MPO/TTNO construction by symbolic state compression (the finite-state-machine view this
  compiler implements, generalized to trees): C. Hubig, I. P. McCulloch, U. Schollwoeck,
  Phys. Rev. B 95, 035129 (2017), doi:10.1103/PhysRevB.95.035129; G. K.-L. Chan,
  A. Keselman, N. Nakatani, Z. Li, S. R. White, J. Chem. Phys. 145, 014102 (2016),
  doi:10.1063/1.4955108.
* Jordan-Wigner transformation: P. Jordan, E. Wigner, Z. Phys. 47, 631 (1928),
  doi:10.1007/BF01331938.
* Tree operators and sweeps: N. Nakatani, G. K.-L. Chan, J. Chem. Phys. 138, 134113
  (2013), doi:10.1063/1.4798639; K. Gunst, F. Verstraete, S. Wouters, O. Legeza,
  D. Van Neck, J. Chem. Theory Comput. 14, 2026 (2018), doi:10.1021/acs.jctc.8b00098.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import resources as res
from ..util.logging import get_logger
from .block import BlockTensor, QuantumNumber, Space
from .graph import NetworkGraph
from .sparse import SparseW, sparse_w_gb

log = get_logger(__name__)

# Local fermionic mode operators in the (|0>, |1>) basis. `a` annihilates: a|1> = |0>.
_I2 = np.eye(2, dtype=np.complex128)
_A = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
_ADAG = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


@dataclass(frozen=True)
class ModeBasis:
    """The local basis of one physical mode: dimension and per-state quantum numbers."""

    dim: int
    charges: Tuple[QuantumNumber, ...]

    def __post_init__(self):
        if len(self.charges) != self.dim:
            raise ValueError("{} charges for a dim-{} mode".format(len(self.charges),
                                                                   self.dim))


#: A spinor orbital: empty (N = 0) or occupied (N = 1).
FERMION_MODE = ModeBasis(2, (QuantumNumber(0), QuantumNumber(1)))


@dataclass(frozen=True)
class ProductTerm:
    """One term ``coeff * O_{m1} O_{m2} ...`` with exactly one local matrix per mode.

    ``modes`` is strictly ascending and aligned with ``mats``. For fermionic strings this is
    the *output* of :func:`fermion_term` — never build fermionic terms by hand, the JW signs
    live there. Generic (spin-model) terms are built directly; their matrices must commute
    across sites, which is the user's statement, not something the compiler can check.
    """

    coeff: complex
    modes: Tuple[int, ...]
    mats: Tuple[np.ndarray, ...]

    def __post_init__(self):
        if len(self.modes) != len(self.mats):
            raise ValueError("modes and mats must align")
        if any(b <= a for a, b in zip(self.modes, self.modes[1:])):
            raise ValueError("term modes must be strictly ascending, got {}"
                             .format(self.modes))


def fermion_term(coeff: complex, ops: Sequence[Tuple[int, bool]]) -> Optional[ProductTerm]:
    """Jordan-Wigner a fermionic operator string into a :class:`ProductTerm`.

    ``ops`` is the string **as written** (leftmost operator acts last on a ket):
    ``[(p, True), (q, False)]`` is ``a+_p a_q``. Mode ``m``'s local matrix is the
    left-to-right product of the string's contributions at ``m`` — the operator itself where
    it acts, ``Z`` from every operator on a *higher* mode — so all fermionic signs are
    absorbed with no global prefactor (module docstring; this is what matches the
    ``ci/strings.py`` determinant convention). Returns ``None`` when the product vanishes
    identically (``a a`` on one mode) or the coefficient is zero. Identity factors are
    dropped, so the support is ``[min, max]`` of the string minus even-``Z`` gaps.
    """
    if coeff == 0.0:
        return None
    ops = [(int(m), bool(dag)) for m, dag in ops]
    if not ops:
        raise ValueError("an operator string needs at least one operator")
    lo, hi = min(m for m, _ in ops), max(m for m, _ in ops)
    modes, mats = [], []
    for m in range(lo, hi + 1):
        mat = _I2
        for m_i, dag in ops:
            if m_i == m:
                mat = mat @ (_ADAG if dag else _A)
            elif m_i > m:
                mat = mat @ _Z
        if not mat.any():
            return None
        if not np.array_equal(mat, _I2):
            modes.append(m)
            mats.append(np.ascontiguousarray(mat))
    if not modes:
        # the whole string collapsed to the identity; carry it on the lowest mode so the
        # term still has a well-defined (trivial) support
        modes, mats = [lo], [np.ascontiguousarray(_I2.copy())]
    return ProductTerm(complex(coeff), tuple(modes), tuple(mats))


def consolidate(terms: Sequence[Optional[ProductTerm]],
                tol: float = 0.0) -> List[ProductTerm]:
    """Merge terms with identical support and matrices; drop merged coefficients ``<= tol``.

    Exact merging: the key is the byte content of the matrices, so only genuinely identical
    operator products merge.
    """
    merged: Dict[tuple, list] = {}
    for t in terms:
        if t is None:
            continue
        key = (t.modes, tuple(m.tobytes() for m in t.mats))
        if key in merged:
            merged[key][0] += t.coeff
        else:
            merged[key] = [t.coeff, t]
    return [ProductTerm(complex(c), t.modes, t.mats)
            for c, t in merged.values() if abs(c) > tol]


def hamiltonian_product_terms(h: np.ndarray, eri: np.ndarray,
                              tol: float = 0.0) -> List[ProductTerm]:
    """The ab initio Hamiltonian as consolidated product terms (module docstring convention).

    ``h`` Hermitian ``(n, n)``; ``eri`` ``(pq|rs)`` in chemists' notation, 4-fold symmetry
    only. ``tol`` screens *merged* coefficients — 0 by default: a dropped term is an
    approximation the caller must choose, never a default.
    """
    h = np.asarray(h)
    eri = np.asarray(eri)
    n = h.shape[0]
    if h.shape != (n, n) or eri.shape != (n, n, n, n):
        raise ValueError("h must be (n, n) and eri (n, n, n, n); got {} and {}"
                         .format(h.shape, eri.shape))
    terms: List[Optional[ProductTerm]] = []
    for p in range(n):
        for q in range(n):
            terms.append(fermion_term(h[p, q], [(p, True), (q, False)]))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    terms.append(fermion_term(0.5 * eri[p, q, r, s],
                                              [(p, True), (r, True),
                                               (s, False), (q, False)]))
    return consolidate(terms, tol)


def one_electron_product_terms(a: np.ndarray, tol: float = 0.0) -> List[ProductTerm]:
    """A one-electron operator ``sum_pq A_pq a+_p a_q`` as consolidated product terms.

    This is how the property operators reach the network layer: a
    magnetic-moment component over the active spinors is a one-electron matrix, and lifting
    it onto the local-multiplet model space needs nothing beyond these terms and
    :func:`compile_ttno` — the same route the Hamiltonian takes, so there is no second sign
    convention to get wrong. ``tol`` screens merged coefficients; 0 by default (the rule:
    a dropped term is an approximation the caller must choose).
    """
    a = np.asarray(a)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("a one-electron operator must be (n, n), got {}".format(a.shape))
    terms: List[Optional[ProductTerm]] = []
    for p in range(n):
        for q in range(n):
            terms.append(fermion_term(a[p, q], [(p, True), (q, False)]))
    return consolidate(terms, tol)


def ttno_from_cas_integrals(ints, graph: NetworkGraph, root: int = 0,
                            tol: float = 0.0) -> "TTNO":
    """Compile the active-space Hamiltonian of a ``CASIntegrals``-like object.

    Duck-typed on purpose (module docstring): needs ``h_active_effective()`` and
    ``active_eri()`` and nothing else, so ``kuiva.dmrg`` stays import-free of
    ``kuiva.mcscf``. ⚠ ``e_core`` is *not* in the operator; the solver adds it.
    """
    return compile_ttno(graph, hamiltonian_product_terms(ints.h_active_effective(),
                                                         ints.active_eri(), tol),
                        root=root)


# --- the compiler -------------------------------------------------------------------------

def _charge_shift(mat: np.ndarray, basis: ModeBasis) -> QuantumNumber:
    """The (unique) charge shift of a local matrix; refuses inhomogeneous matrices.

    A matrix on a charge-labelled mode must shift every state it connects by the same
    quantum number, or it cannot live on one operator-leg state. With all-zero charges (a
    symmetry-free model) every matrix passes trivially.
    """
    rows, cols = np.nonzero(mat)
    if rows.size == 0:
        raise ValueError("a zero local matrix cannot enter a term")
    shifts = {basis.charges[int(i)] - basis.charges[int(j)] for i, j in zip(rows, cols)}
    if len(shifts) != 1:
        raise ValueError("local matrix is not charge-homogeneous: shifts {}"
                         .format(sorted(shifts)))
    return shifts.pop()


def _phys_space(modes: Sequence[int], bases: Dict[int, ModeBasis],
                width: int) -> Tuple[Space, np.ndarray]:
    """A node's physical space and the sector<->kron permutation.

    ``perm[s]`` is the kron index of sector-sorted position ``s`` (stable within a sector,
    so equal-charge states keep their kron order).
    """
    if not modes:
        return Space([(QuantumNumber.zero(width), 1)]), np.array([0], dtype=np.int64)
    charge_lists = [bases[m].charges for m in modes]
    states = list(itertools.product(*[range(bases[m].dim) for m in modes]))
    qns = [sum((cl[i] for cl, i in zip(charge_lists, st)), QuantumNumber.zero(width))
           for st in states]
    order = sorted(range(len(states)), key=lambda k: (qns[k], k))
    sectors: List[Tuple[QuantumNumber, int]] = []
    for k in order:
        if sectors and sectors[-1][0] == qns[k]:
            sectors[-1] = (qns[k], sectors[-1][1] + 1)
        else:
            sectors.append((qns[k], 1))
    return Space(sectors), np.array(order, dtype=np.int64)


@dataclass(eq=False)
class TTNO:
    """The compiled operator: one sparse W tensor per node plus bond state spaces.

    ``tensors[u]`` is a :class:`~kuiva.dmrg.sparse.SparseW` with legs ``[parent-op,
    child-ops (children[u] order), phys-out, phys-in]`` — sparse because a compiled
    operator node is a list of transitions, not a dense array (that module's docstring
    carries the measurement); ``bond_space[u]``/``bond_labels[u]`` describe the operator
    states on the bond from ``u`` toward its parent (the root's is the trivial completed
    channel).
    ``node_modes[u]`` is the node's ascending mode tuple, ``mode_dims[u]`` the matching
    local dimensions, and ``phys_perm[u]`` the sector->kron permutation of
    :func:`_phys_space`.
    """

    graph: NetworkGraph
    root: int
    parent: Tuple[int, ...]
    children: Tuple[Tuple[int, ...], ...]
    tensors: List[SparseW]
    bond_space: Tuple[Space, ...]
    bond_labels: Tuple[Tuple[tuple, ...], ...]
    phys_space: Tuple[Space, ...]
    phys_perm: Tuple[np.ndarray, ...]
    node_modes: Tuple[Tuple[int, ...], ...]
    mode_dims: Tuple[Tuple[int, ...], ...]
    charge: QuantumNumber
    #: memory reservations backing ``tensors`` (one per node). An owner that drops the TTNO
    #: in a limit-configured run releases these via ``res.BUDGET.release``; a cached TTNO
    #: (the reconnection compile cache) is genuinely resident and keeps them.
    allocations: List[object] = field(default_factory=list)

    def bond_dimensions(self) -> Dict[Tuple[int, int], int]:
        """Operator bond dimension per directed bond ``(u, parent(u))`` — a diagnostic."""
        return {(u, int(self.parent[u])): self.bond_space[u].total_dim
                for u in range(self.graph.n_nodes) if u != self.root}

    @property
    def nbytes(self) -> int:
        return int(sum(t.nbytes for t in self.tensors))

    @property
    def dense_nbytes(self) -> int:
        """What the same operator would cost with dense sector blocks — the diagnostic
        behind the sparse storage decision, reported rather than claimed."""
        return int(sum(t.dense_nbytes for t in self.tensors))

    @property
    def nnz(self) -> int:
        return int(sum(t.nnz for t in self.tensors))

    # --- dense oracle (validation only) ---------------------------------------------------

    def to_dense(self, max_dim: int = 4096) -> np.ndarray:
        """The full operator in the **global mode-ascending kron basis**.

        Index convention: plain C-order kron over all modes ascending (lowest mode
        slowest). This is the Tier-0 oracle path, never a compute path — it refuses above
        ``max_dim`` total dimension.
        """
        total = 1
        for dims in self.mode_dims:
            for d in dims:
                total *= d
        if total > max_dim:
            raise ValueError("dense TTNO would be {0} x {0}; raise max_dim if you really "
                             "mean it".format(total))
        msg, modes = self._dense_message(self.root)
        msg = msg[0]                                       # close the root (H) leg
        k = len(modes)
        order = np.argsort(np.asarray(modes, dtype=np.int64), kind="stable")
        msg = msg.transpose(tuple(order) + tuple(order + k))
        return np.ascontiguousarray(msg.reshape(total, total))

    def _dense_message(self, u: int) -> Tuple[np.ndarray, List[int]]:
        """Contract the subtree at ``u``: axes ``(parent-op, out per mode.., in per
        mode..)``, modes in the returned order."""
        w = self.tensors[u].to_dense()                     # (par, ch.., pout, pin)
        inv = np.argsort(self.phys_perm[u])                # kron idx -> sector position
        w = w[..., inv, :][..., :, inv]                    # phys axes back to kron order
        n_ch = len(self.children[u])
        dims = self.mode_dims[u]
        w = w.reshape(w.shape[:1 + n_ch] + dims + dims)
        # label every axis, contract children, then regroup by labels
        labels: List[tuple] = [("par",)] + [("ch", c) for c in self.children[u]] \
            + [("out", m) for m in self.node_modes[u]] \
            + [("in", m) for m in self.node_modes[u]]
        m = w
        for c in self.children[u]:
            child_msg, child_modes = self._dense_message(c)
            axis = labels.index(("ch", c))
            m = np.tensordot(m, child_msg, axes=([axis], [0]))
            labels = labels[:axis] + labels[axis + 1:] \
                + [("out", cm) for cm in child_modes] + [("in", cm) for cm in child_modes]
        modes = [lab[1] for lab in labels if lab[0] == "out"]
        perm = ([labels.index(("par",))]
                + [labels.index(("out", mm)) for mm in modes]
                + [labels.index(("in", mm)) for mm in modes])
        return m.transpose(perm), modes


def compile_ttno(graph: NetworkGraph, terms: Sequence[Optional[ProductTerm]],
                 bases=None, root: int = 0,
                 attachments: Optional[list] = None) -> TTNO:
    """Compile a term sum into a TTNO on ``graph`` (module docstring: how and why exact).

    ``bases``: per-mode :class:`ModeBasis` mapping, or one basis for every mode; default
    :data:`FERMION_MODE`. ``root`` fixes the rooted orientation — any node works; the
    choice affects operator bond dimensions, never the operator.

    ``attachments``, if a list, is filled with one record per (non-``None``) input term:
    ``(node, (out_sector, out_offset), ((in_sector, in_offset), ...))`` — the node where
    the term's coefficient is attached and the bond-space positions of its transition
    channels, in the W-tensor leg order ``[parent, children ascending]``. This is the seam
    :class:`TTNOTemplate` builds its refill/extraction tables on; the compiler invariant
    that every term attaches **exactly once** is asserted here rather than trusted.
    """
    terms = [t for t in terms if t is not None]
    if not terms:
        raise ValueError("no terms to compile")
    all_modes = sorted(m for c in graph.contents for m in c)
    mode_set = set(all_modes)
    if bases is None:
        bases = {m: FERMION_MODE for m in all_modes}
    elif isinstance(bases, ModeBasis):
        bases = {m: bases for m in all_modes}
    width = bases[all_modes[0]].charges[0].width if all_modes else 1
    zero = QuantumNumber.zero(width)

    # --- intern matrices, derive charge shifts, validate the terms ------------------------
    mat_ids: Dict[bytes, int] = {}
    mats: List[np.ndarray] = []
    z_id = -1
    shifts: Dict[Tuple[int, int], QuantumNumber] = {}
    term_pairs: List[Tuple[Tuple[int, int], ...]] = []    # ((mode, mat_id), ...) per term
    charge = None
    for t in terms:
        pairs = []
        total = zero
        for m, mat in zip(t.modes, t.mats):
            if m not in mode_set:
                raise ValueError("term touches mode {} which no node carries".format(m))
            mat = np.ascontiguousarray(mat, dtype=np.complex128)
            key = mat.tobytes()
            if key not in mat_ids:
                mat_ids[key] = len(mats)
                mats.append(mat)
                if mat.shape == (2, 2) and np.array_equal(mat, _Z):
                    z_id = mat_ids[key]
            mid = mat_ids[key]
            if (m, mid) not in shifts:
                shifts[(m, mid)] = _charge_shift(mat, bases[m])
            pairs.append((m, mid))
            total = total + shifts[(m, mid)]
        term_pairs.append(tuple(pairs))
        if charge is None:
            charge = total
        elif total != charge:
            raise ValueError("terms carry different total charge shifts ({} vs {}): they "
                             "cannot share one TTNO".format(charge, total))

    # --- rooted orientation and closed subtree mode sets ----------------------------------
    parent_arr, preorder = graph.parents(root)
    parent = tuple(int(x) for x in parent_arr)
    child_lists: List[List[int]] = [[] for _ in range(graph.n_nodes)]
    for u in range(graph.n_nodes):
        if u != root:
            child_lists[parent[u]].append(u)
    children = tuple(tuple(sorted(c)) for c in child_lists)
    subtree: List[set] = [set() for _ in range(graph.n_nodes)]
    for u in reversed([int(x) for x in preorder]):
        s = set(graph.contents[u])
        for c in children[u]:
            s |= subtree[c]
        subtree[u] = s
    n_total_modes = len(all_modes)

    # --- pass 1: content label of every term on every bond --------------------------------
    ID, H = ("1",), ("H",)
    registries: List[Dict[tuple, int]] = [dict() for _ in range(graph.n_nodes)]
    lab_idx = np.zeros((len(terms), graph.n_nodes), dtype=np.int64)
    for ti, pairs in enumerate(term_pairs):
        for u in range(graph.n_nodes):
            if u == root:
                continue
            ins = subtree[u]
            inside = tuple(p for p in pairs if p[0] in ins)
            if not inside:
                label = ID
            elif len(inside) == len(pairs):
                label = H
            else:
                outside = tuple(p for p in pairs if p[0] not in ins)
                nzi = sum(1 for _, mid in inside if mid != z_id)
                nzo = sum(1 for _, mid in outside if mid != z_id)
                if nzi != nzo:
                    kind = "L" if nzi < nzo else "R"
                else:
                    kind = "L" if 2 * len(ins) <= n_total_modes else "R"
                if kind == "L":
                    label = ("L", inside)
                else:
                    dn = charge
                    for mm, mid in outside:
                        dn = dn - shifts[(mm, mid)]
                    label = ("R", outside, dn)
            reg = registries[u]
            idx = reg.get(label)
            if idx is None:
                idx = reg[label] = len(reg)
            lab_idx[ti, u] = idx

    # --- bond spaces: states sorted by (charge sector, label) -----------------------------
    def label_charge(label: tuple) -> QuantumNumber:
        if label == ID:
            return zero
        if label == H:
            return charge
        if label[0] == "L":
            dn = zero
            for mm, mid in label[1]:
                dn = dn + shifts[(mm, mid)]
            return dn
        return label[2]

    bond_space: List[Space] = [None] * graph.n_nodes
    bond_labels: List[Tuple[tuple, ...]] = [None] * graph.n_nodes
    positions: List[Dict[int, Tuple[int, int]]] = [None] * graph.n_nodes
    for u in range(graph.n_nodes):
        if u == root:
            bond_space[u] = Space([(charge, 1)])
            bond_labels[u] = (H,)
            positions[u] = {0: (0, 0)}
            continue
        labels = list(registries[u])
        order = sorted(range(len(labels)), key=lambda i: (label_charge(labels[i]),
                                                          labels[i]))
        sectors: List[Tuple[QuantumNumber, int]] = []
        pos: Dict[int, Tuple[int, int]] = {}
        for i in order:
            qn = label_charge(labels[i])
            if sectors and sectors[-1][0] == qn:
                sectors[-1] = (qn, sectors[-1][1] + 1)
            else:
                sectors.append((qn, 1))
            pos[registries[u][labels[i]]] = (len(sectors) - 1, sectors[-1][1] - 1)
        bond_space[u] = Space(sectors)
        bond_labels[u] = tuple(labels[i] for i in order)
        positions[u] = pos

    # --- physical spaces ------------------------------------------------------------------
    node_modes = tuple(tuple(sorted(graph.contents[u])) for u in range(graph.n_nodes))
    mode_dims = tuple(tuple(bases[m].dim for m in node_modes[u])
                      for u in range(graph.n_nodes))
    phys: List[Tuple[Space, np.ndarray]] = [
        _phys_space(node_modes[u], bases, width) for u in range(graph.n_nodes)]

    # --- pass 2: transitions and W tensors ------------------------------------------------
    carrying = [np.zeros(len(reg) if u != root else 1, dtype=bool)
                for u, reg in enumerate(registries)]
    for u in range(graph.n_nodes):
        if u == root:
            carrying[u][0] = True
            continue
        for label, idx in registries[u].items():
            carrying[u][idx] = label[0] in ("H", "R")

    attach_slots: Optional[list] = [None] * len(terms) if attachments is not None else None
    tensors: List[SparseW] = [None] * graph.n_nodes
    allocations: List[object] = []
    for u in range(graph.n_nodes):
        tensors[u] = _node_tensor(u, root, children[u], registries, positions, bond_space,
                                  bond_labels, carrying, lab_idx, term_pairs, terms,
                                  node_modes[u], mode_dims[u], phys[u], bases, mats, zero,
                                  attach=attach_slots, allocations=allocations)
    if attach_slots is not None:
        missing = [i for i, a in enumerate(attach_slots) if a is None]
        if missing:                                   # pragma: no cover - compiler invariant
            raise AssertionError("terms {} were never coefficient-attached — compiler "
                                 "invariant broken".format(missing[:5]))
        attachments[:] = attach_slots

    ttno = TTNO(graph=graph, root=root, parent=parent, children=children, tensors=tensors,
                bond_space=tuple(bond_space), bond_labels=tuple(bond_labels),
                phys_space=tuple(p[0] for p in phys),
                phys_perm=tuple(p[1] for p in phys),
                node_modes=node_modes, mode_dims=mode_dims, charge=charge,
                allocations=allocations)
    dims = ttno.bond_dimensions()
    if dims:
        log.debug("TTNO compiled: %d terms, max operator bond dimension %d",
                  len(terms), max(dims.values()))
    return ttno


def _node_tensor(u, root, children_u, registries, positions, bond_space, bond_labels,
                 carrying, lab_idx, term_pairs, terms, modes_u, dims_u, phys_u, bases,
                 mats, zero, attach=None, allocations=None) -> SparseW:
    """Assemble one W tensor from the label bookkeeping (see :func:`compile_ttno`)."""
    phys_sp, perm = phys_u
    d = phys_sp.total_dim
    mode_pos = {m: i for i, m in enumerate(modes_u)}

    # local-matrix cache: key = per-node-mode mat id (-1 = identity)
    local_cache: Dict[tuple, np.ndarray] = {}

    def local_matrix(key: tuple) -> np.ndarray:
        mat = local_cache.get(key)
        if mat is None:
            mat = np.eye(1, dtype=np.complex128)
            for mm, mid in zip(modes_u, key):
                factor = np.eye(bases[mm].dim, dtype=np.complex128) if mid < 0 else mats[mid]
                mat = np.kron(mat, factor)
            local_cache[key] = mat
        return mat

    unit_trans: Dict[tuple, np.ndarray] = {}
    coeff_trans: Dict[tuple, np.ndarray] = {}
    identity_needed = False

    id_index = -1 if u == root else registries[u].get(("1",), -1)
    for ti, pairs in enumerate(term_pairs):
        out_idx = 0 if u == root else int(lab_idx[ti, u])
        if out_idx == id_index and u != root:
            identity_needed = True
            continue
        in_idx = tuple(int(lab_idx[ti, c]) for c in children_u)
        key = (out_idx, in_idx)
        lkey = tuple(_term_matid_at(pairs, m) for m in modes_u)
        local = local_matrix(lkey)
        out_carries = carrying[u][out_idx] if u != root else True
        in_carries = any(bool(carrying[c][i]) for c, i in zip(children_u, in_idx))
        if out_carries and not in_carries:
            if attach is not None:
                if attach[ti] is not None:            # pragma: no cover - compiler invariant
                    raise AssertionError("term {} attached at nodes {} and {} — the "
                                         "monotonicity invariant is broken"
                                         .format(ti, attach[ti][0], u))
                attach[ti] = (u, positions[u][out_idx],
                              tuple(positions[c][i] for c, i in zip(children_u, in_idx)))
            acc = coeff_trans.get(key)
            if acc is None:
                coeff_trans[key] = terms[ti].coeff * local
            else:
                acc += terms[ti].coeff * local
        else:
            existing = unit_trans.get(key)
            if existing is None:
                unit_trans[key] = local
            elif existing is not local:                    # cache gives object identity
                raise AssertionError("unit transition at node {} is not label-determined "
                                     "— compiler invariant broken".format(u))

    if identity_needed:
        id_out = registries[u][("1",)]
        id_in = tuple(registries[c][("1",)] for c in children_u)
        unit_trans[(id_out, id_in)] = local_matrix(tuple(-1 for _ in modes_u))

    # --- scatter transitions into blocks --------------------------------------------------
    # sector-sorted physical matrix: sector position s <-> kron index perm[s]
    par_space = bond_space[u]
    child_spaces = [bond_space[c] for c in children_u]
    spaces = (par_space,) + tuple(child_spaces) + (phys_sp, phys_sp)
    signs = (-1,) + tuple(1 for _ in children_u) + (1, -1)
    p_off = phys_sp.offsets

    # Pass A enumerates every scatter target and the block set WITHOUT allocating a block,
    # so the memory reservation below fires before the first large allocation — the fix for
    # the multi-site OOM kills: a W tensor stores each symmetry
    # sector as a dense block scaled by the operator bond dimension squared, and that is
    # the allocation that must be refused, not discovered by the kernel OOM killer. The
    # held ``sub`` views reference only the d x d ``matp`` matrices, never a block.
    entries: List[tuple] = []
    shapes: Dict[tuple, tuple] = {}
    for (out_idx, in_idx), mat in itertools.chain(unit_trans.items(),
                                                  coeff_trans.items()):
        osec, ooff = positions[u][out_idx]
        cpos = [positions[c][i] for c, i in zip(children_u, in_idx)]
        matp = mat[np.ix_(perm, perm)]
        for i in range(phys_sp.nsectors):
            ri = slice(int(p_off[i]), int(p_off[i + 1]))
            for j in range(phys_sp.nsectors):
                rj = slice(int(p_off[j]), int(p_off[j + 1]))
                sub = matp[ri, rj]
                if not sub.any():
                    continue
                row = (osec,) + tuple(cs for cs, _ in cpos) + (i, j)
                if row not in shapes:
                    shapes[row] = (int(par_space.dims[osec]),) \
                        + tuple(int(sp.dims[cs]) for sp, (cs, _) in zip(child_spaces, cpos)) \
                        + (int(phys_sp.dims[i]), int(phys_sp.dims[j]))
                index = (ooff,) + tuple(co for _, co in cpos)
                entries.append((row, index, sub))

    # Pass B turns the scatter targets into sparse entries — one (row, flat index, value)
    # triple per genuine nonzero, never a dense sector block. ⚠ The local matrix ``sub`` is
    # a Kronecker product of I/a/a+/Z, so it carries at most one nonzero per column: the
    # dense payload would pay ``d^2`` for content of size ``d``, and that factor is what
    # scales with node fatness (:mod:`kuiva.dmrg.sparse`, measured).
    rows_list, flat_list, val_list = [], [], []
    for row, index, sub in entries:
        aa, bb = np.nonzero(sub)
        if aa.size == 0:                               # pragma: no cover - pass A screens
            continue
        head = tuple(np.full(aa.size, int(i), dtype=np.int64) for i in index)
        flat_list.append(np.ravel_multi_index(head + (aa, bb), shapes[row]))
        rows_list.append(np.tile(np.asarray(row, dtype=np.int64), (aa.size, 1)))
        val_list.append(sub[aa, bb])
    if not rows_list:                                  # pragma: no cover - a node always acts
        raise AssertionError("node {} compiled to an empty operator".format(u))
    nnz = int(sum(a.size for a in flat_list))
    dense_elems = sum(int(np.prod(shape)) for shape in shapes.values())
    # ⚠ The reservation covers the tensor **and its contraction-pattern caches**: a node is
    # contracted once per direction (one op leg left open) plus once with every op leg
    # contracted, and each pattern caches a CSR of the same nonzeros; the root additionally
    # carries the memoized copy with its dim-1 completed channel closed. Bounding all of it
    # here, at compile time, is what keeps a lazily built cache from being
    # resident-but-unaccounted (an estimate nobody checks is decoration). One "unit"
    # is the tensor's own exact size; a CSR of the same nonzeros is slightly smaller, so
    # this is an upper bound on residency and never an under-estimate.
    n_units = 1 + (len(spaces) - 2) + (1 if u == root else 0)
    alloc = res.reserve(
        "TTNO node {} ({} blocks, {} nonzeros)".format(u, len(shapes), nnz),
        n_units * sparse_w_gb(spaces, nnz, len(shapes)),
        note="operator bond dims {} x phys dim {}; dense form would be {:.2f} GB".format(
            [sp.total_dim for sp in (par_space,) + tuple(child_spaces)], d,
            16.0 * dense_elems / 1024.0 ** 3),
        advice=["the transition count grows with the operator bond dimension, i.e. as the "
                "square of the active-space size; a different topology (or fewer modes per "
                "node) changes the bond dimensions it is built from"])
    if allocations is not None:
        allocations.append(alloc)

    return SparseW.from_entries(spaces, signs, zero, np.concatenate(rows_list),
                                np.concatenate(flat_list), np.concatenate(val_list))


def _term_matid_at(pairs: Tuple[Tuple[int, int], ...], mode: int) -> int:
    for m, mid in pairs:
        if m == mode:
            return mid
    return -1


# --- the reusable template: one compile per topology, refill per integral set ---------------

@dataclass(eq=False)
class _RefillNode:
    """Per-node refill/extraction tables of a :class:`TTNOTemplate` (internal)."""

    node: int
    skeleton: SparseW            #: the union entry set: compiled transitions + every slot
    base: np.ndarray             #: unit-transition content, coefficient slots at zero
    gidx: np.ndarray             #: skeleton entry position per (term, matrix element)
    tid: np.ndarray              #: term id per entry
    val: np.ndarray              #: local-matrix value per entry (JW signs included)


class TTNOTemplate:
    """The Hamiltonian TTNO compiled **once per topology**, refilled per integral set.

    The compiler's label structure is coefficient-independent (a measurement this
    class exists to exploit): the template enumerates **every** one- and two-electron index
    tuple — including those whose integral happens to be zero — compiles the label
    structure once, and records where each term's coefficient lands
    (:func:`compile_ttno`'s ``attachments``). Two consumers, sharing one table:

    * :meth:`fill` produces the TTNO at a given ``(h, eri)`` by scattering coefficients
      into the recorded slots — no recompilation inside a CASSCF macro-iteration.
    * :meth:`rdms_from_environments` reads the **same slots** out of the per-node operator
      environments ``G_u = dE/dW_u`` (:mod:`kuiva.dmrg.density`): the expectation value of
      one term's operator content is ``sum_ij local[i,j] G_u[channels, i, j]`` at its
      attachment, because every transition above and below the attachment is
      label-determined and unit. Refill and extraction are transposes of each other, so a
      defect in the table breaks the energy closure test rather than hiding.

    ⚠ The full index enumeration is what makes extraction complete: a template built from
    one integral set's *nonzero* terms could not report a 2-RDM element whose integral is
    a symmetry zero, and a Γ silently missing such elements looks entirely plausible.

    Orchestration: built once per topology; ``fill`` is two vectorized
    scatters per refilled node.
    """

    def __init__(self, graph: NetworkGraph, root: int = 0):
        all_modes = sorted(m for c in graph.contents for m in c)
        if all_modes != list(range(len(all_modes))):
            raise ValueError("graph mode labels must be exactly 0..n-1, got {}"
                             .format(all_modes))
        n = len(all_modes)
        self.graph = graph
        self.root = int(root)
        self.n_modes = n

        terms: List[ProductTerm] = []
        kinds: List[int] = []
        idxs: List[Tuple[int, int, int, int]] = []
        for p in range(n):
            for q in range(n):
                t = fermion_term(1.0, [(p, True), (q, False)])
                if t is not None:
                    terms.append(t)
                    kinds.append(0)
                    idxs.append((p, q, 0, 0))
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        t = fermion_term(1.0, [(p, True), (r, True),
                                               (s, False), (q, False)])
                        if t is not None:
                            terms.append(t)
                            kinds.append(1)
                            idxs.append((p, q, r, s))
        self._kinds = np.asarray(kinds, dtype=np.int64)
        self._idxs = np.asarray(idxs, dtype=np.int64)
        self.n_terms = len(terms)

        attachments: list = []
        ttno = compile_ttno(graph, terms, root=self.root, attachments=attachments)
        self.ttno = ttno

        # -- per-node entry tables ---------------------------------------------------------
        raw: Dict[int, List[tuple]] = {}
        for tid, (term, (u, opos, cpos)) in enumerate(zip(terms, attachments)):
            modes_u = ttno.node_modes[u]
            mat_at = dict(zip(term.modes, term.mats))
            local = np.eye(1, dtype=np.complex128)
            for m in modes_u:
                local = np.kron(local, mat_at.get(m, _I2))
            perm = ttno.phys_perm[u]
            matp = local[np.ix_(perm, perm)]
            offs = ttno.phys_space[u].offsets
            osec, ooff = opos
            csecs = tuple(cs for cs, _ in cpos)
            coffs = tuple(co for _, co in cpos)
            aa, bb = np.nonzero(matp)
            for a, b in zip(aa, bb):
                sa = int(np.searchsorted(offs, a, side="right") - 1)
                sb = int(np.searchsorted(offs, b, side="right") - 1)
                row = (osec,) + csecs + (sa, sb)
                index = (ooff,) + coffs + (int(a - offs[sa]), int(b - offs[sb]))
                raw.setdefault(u, []).append((row, index, tid, matp[a, b]))

        self._nodes: List[_RefillNode] = []
        base_bytes = 0
        for u in sorted(raw):
            w = ttno.tensors[u]
            # the skeleton is the union of the compiled transitions and every attachment
            # slot: a term whose integral is a symmetry zero still needs its slot, or the
            # extraction below could not report that element of Gamma (class docstring)
            slot_rows = np.asarray([e[0] for e in raw[u]], dtype=np.int64)
            slot_flat = np.asarray(
                [np.ravel_multi_index(e[1], tuple(int(sp.dims[i]) for sp, i
                                                  in zip(w.spaces, e[0])))
                 for e in raw[u]], dtype=np.int64)
            block_of = np.repeat(np.arange(w.nblocks, dtype=np.int64),
                                 np.diff(w.indptr))
            skeleton = SparseW.from_entries(
                w.spaces, w.signs, w.charge,
                np.concatenate([w.sectors[block_of], slot_rows]),
                np.concatenate([w.flat, slot_flat]),
                np.concatenate([w.values, np.zeros(slot_flat.size, dtype=np.complex128)]))
            gidx = skeleton.entry_positions(slot_rows, slot_flat)
            tid_arr = np.asarray([e[2] for e in raw[u]], dtype=np.int64)
            val = np.asarray([e[3] for e in raw[u]], dtype=np.complex128)
            # remove the compile's coefficient-1 content: base then carries the unit
            # transitions only, and fill() adds true coefficients into clean slots
            base = skeleton.values.copy()
            np.subtract.at(base, gidx, val)
            base_bytes += base.nbytes
            self._nodes.append(_RefillNode(node=u, skeleton=skeleton, base=base,
                                           gidx=gidx, tid=tid_arr, val=val))
        res.reserve("TTNO template base ({} nodes refilled)".format(len(self._nodes)),
                    base_bytes / 1024.0 ** 3,
                    note="unit-transition copy of every coefficient-carrying W tensor",
                    advice=["one template per topology; it replaces a recompile per "
                            "CASSCF macro-iteration"])

    def fill(self, h: np.ndarray, eri: np.ndarray) -> TTNO:
        """The TTNO at these integrals (``0.5 * (pq|rs)`` per term,
        ``e_core`` excluded). Nodes carrying no coefficient share the template's tensors."""
        n = self.n_modes
        h = np.asarray(h)
        eri = np.asarray(eri)
        if h.shape != (n, n) or eri.shape != (n, n, n, n):
            raise ValueError("h must be ({0}, {0}) and eri ({0}, {0}, {0}, {0}); got {1} "
                             "and {2}".format(n, h.shape, eri.shape))
        i = self._idxs
        one = self._kinds == 0
        coeffs = np.where(one, h[i[:, 0], i[:, 1]],
                          0.5 * eri[i[:, 0], i[:, 1], i[:, 2], i[:, 3]])
        tensors = list(self.ttno.tensors)
        for rec in self._nodes:
            res.require("TTNO refill node {}".format(rec.node),
                        rec.base.nbytes / 1024.0 ** 3,
                        note="coefficient scatter into the template's slot table")
            buf = rec.base.copy()
            np.add.at(buf, rec.gidx, coeffs[rec.tid] * rec.val)
            tensors[rec.node] = rec.skeleton.with_values(buf)
        t = self.ttno
        return TTNO(graph=t.graph, root=t.root, parent=t.parent, children=t.children,
                    tensors=tensors, bond_space=t.bond_space, bond_labels=t.bond_labels,
                    phys_space=t.phys_space, phys_perm=t.phys_perm,
                    node_modes=t.node_modes, mode_dims=t.mode_dims, charge=t.charge)

    def expectations(self, environments: Sequence[Optional[BlockTensor]]) -> np.ndarray:
        """Per-term operator expectations from per-node environments ``G_u = dE/dW_u``.

        ``environments[u]`` must carry W-tensor leg order ``[parent-op, child-ops
        ascending, phys-bra, phys-ket]`` over the same spaces
        (:func:`kuiva.dmrg.density.node_environments` produces exactly this). A block the
        state does not populate is simply absent and contributes zero.
        """
        exp = np.zeros(self.n_terms, dtype=np.complex128)
        for rec in self._nodes:
            g_t = environments[rec.node]
            if g_t is None:
                raise ValueError("no environment supplied for node {}, which carries "
                                 "coefficient attachments".format(rec.node))
            skel = rec.skeleton
            gbuf = np.zeros(skel.nnz, dtype=np.complex128)
            for b, row in enumerate(skel.sectors):
                blk = g_t.find(row)
                if blk is not None:
                    lo, hi = int(skel.indptr[b]), int(skel.indptr[b + 1])
                    gbuf[lo:hi] = blk.ravel()[skel.flat[lo:hi]]
            prod = rec.val * gbuf[rec.gidx]
            exp += np.bincount(rec.tid, weights=prod.real, minlength=self.n_terms) \
                + 1j * np.bincount(rec.tid, weights=prod.imag, minlength=self.n_terms)
        return exp

    def rdms_from_environments(self, environments: Sequence[Optional[BlockTensor]]
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """State-averaged ``(gamma, Gamma)`` in the :mod:`kuiva.rdm.rdm` convention.

        ``gamma_pq = <a+_p a_q>`` and ``Gamma_pqrs = <a+_p a+_r a_s a_q>`` — the same
        objects :class:`kuiva.rdm.rdm.RDMBuilder` returns, so the two paths are directly
        comparable and either plugs into the shared orbital optimizer contract. Index tuples whose
        operator string vanishes identically (``p = r`` or ``q = s``) are exact zeros.
        """
        n = self.n_modes
        res.require("network 2-RDM ({} spinors)".format(n), res.rdm_gb(n, 2),
                    note="the dense n^4 Gamma array",
                    advice=["the 2-RDM itself is n^4; a smaller active space is the only "
                            "knob"])
        exp = self.expectations(environments)
        i = self._idxs
        one = self._kinds == 0
        gamma = np.zeros((n, n), dtype=np.complex128)
        gamma[i[one, 0], i[one, 1]] = exp[one]
        gamma = 0.5 * (gamma + gamma.conj().T)
        gamma2 = np.zeros((n, n, n, n), dtype=np.complex128)
        two = ~one
        gamma2[i[two, 0], i[two, 1], i[two, 2], i[two, 3]] = exp[two]
        return gamma, gamma2


__all__ = ["ModeBasis", "FERMION_MODE", "ProductTerm", "TTNO", "TTNOTemplate",
           "fermion_term", "consolidate", "hamiltonian_product_terms",
           "one_electron_product_terms", "compile_ttno", "ttno_from_cas_integrals"]
