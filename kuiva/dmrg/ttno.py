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

The term table is arrays, and the local matrices are sparse
------------------------------------------------------------
Two things the ledger could not see were measured on the 30-spinor three-site operator, and
both are representation choices rather than algorithms:

* **The term list.** ``n^4`` :class:`ProductTerm` objects — 740 690 of them for 30 spinors —
  were 3.1 GB of Python objects, invisible to every sizing function. The compiler's native
  input is now :class:`TermTable`: coefficients, a CSR of ``(mode, matrix-id)`` pairs and
  the interned distinct local matrices, tens of megabytes for the same operator and
  reserved on the ledger. A :class:`TermTable` still reads as a sequence of
  :class:`ProductTerm` (indexing materializes one), so the model-Hamiltonian seam and every
  consumer that only iterates are unchanged; a consumer that would build a list of all of
  them must use the table's own array methods instead.
* **The transition tables.** A transition's local matrix is a Kronecker product of
  ``I``/``a``/``a+``/``Z``, so it carries at most one nonzero per column — and the compiler
  stored it as a dense ``d x d`` array per pattern, ``2^(2 modes)`` for content of size
  ``2^modes``, which at five modes per node was 5.3 GB against a 2.3 GB reservation (the
  reservation missed the per-transition permuted copies) and at ten modes per node — the
  site-sized nodes the local-multiplet extraction wants — 186 GB. Patterns are now stored as
  their nonzeros, coefficient-carrying transitions as merged nonzero lists accumulated in
  chunks bounded by the resource budget's transient allowance, and every table is reserved
  at its exact size before it is built (:func:`pattern_table_gb`, :func:`assembly_gb`).

Everything here is orchestration and stays Python: the compiler runs once per
integral set, and the hot object it produces is consumed by the sweep, whose contraction
driver is the named port candidate — not this. ⚠ Sums of coefficients into one transition
are accumulated by sorted reduction, so a recompile agrees with the previous per-term
accumulation to rounding, not bitwise.

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
from .block import BlockTensor, QuantumNumber, Space, _flux, _row_keys
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


# --- the local fermionic monoid ------------------------------------------------------------
#
# Every local matrix a Jordan-Wigner string can produce is a signed product of I, a, a+ and
# Z: the closure is twelve matrices plus zero, so a whole batch of strings is reduced with
# one integer product table instead of one matrix product per (term, mode).

def _monoid() -> Tuple[List[np.ndarray], np.ndarray]:
    mats = [_I2, _A, _ADAG, _Z]
    ids = {m.tobytes(): i for i, m in enumerate(mats)}
    table: Dict[Tuple[int, int], int] = {}
    frontier = list(range(len(mats)))
    while frontier:
        new = []
        for i in range(len(mats)):
            for j in range(len(mats)):
                if (i, j) in table:
                    continue
                prod = np.ascontiguousarray(mats[i] @ mats[j])
                if not prod.any():
                    table[(i, j)] = -1
                    continue
                key = prod.tobytes()
                if key not in ids:
                    ids[key] = len(mats)
                    mats.append(prod)
                    new.append(ids[key])
                table[(i, j)] = ids[key]
        frontier = new
    n = len(mats)
    arr = np.full((n + 1, n + 1), -1, dtype=np.int64)     # index n stands for zero
    for (i, j), k in table.items():
        arr[i, j] = k
    return mats, arr


_MONO_MATS, _MONO_TABLE = _monoid()
_MONO_ZERO = len(_MONO_MATS)                              # the absorbing element's index
_MONO_I, _MONO_A, _MONO_ADAG, _MONO_Z = 0, 1, 2, 3


def _string_local_ids(op_modes: np.ndarray, dag: Sequence[bool], n_modes: int) -> np.ndarray:
    """Monoid id of every string's local matrix on every mode: ``(n_strings, n_modes)``.

    ``op_modes`` is ``(n_strings, L)`` — the strings as written, leftmost acting last —
    and ``dag`` says which of the ``L`` positions create. Mode ``m``'s matrix is the
    left-to-right product of the string's contributions at ``m`` (the operator itself
    where it acts, ``Z`` from every operator on a *higher* mode), exactly
    :func:`fermion_term`'s rule; ``_MONO_ZERO`` marks a vanishing product.
    """
    n_str, length = op_modes.shape
    out = np.empty((n_str, n_modes), dtype=np.int8)
    for m in range(n_modes):
        acc = np.full(n_str, _MONO_I, dtype=np.int64)
        for k in range(length):
            x = op_modes[:, k]
            factor = np.where(x == m, _MONO_ADAG if dag[k] else _MONO_A,
                              np.where(x > m, _MONO_Z, _MONO_I))
            acc = _MONO_TABLE[acc, factor]
            acc = np.where(acc < 0, _MONO_ZERO, acc)
        out[:, m] = acc
    return out


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


# --- the term table -------------------------------------------------------------------------

class TermTable(object):
    """Product terms as arrays: coefficients, a CSR of ``(mode, matrix id)`` pairs and the
    interned distinct local matrices (module docstring: why this is the compiler's input).

    Reads as a sequence of :class:`ProductTerm` — ``len``, integer indexing, iteration —
    so the model-Hamiltonian seam and every consumer that only iterates see no change.
    ⚠ Iterating materializes one object per term; a consumer that keeps all of them has
    rebuilt the 3 GB this class removes, so the drivers use :meth:`coerce` and the array
    methods (:meth:`restricted_to`, :meth:`with_coefficients`) instead.

    The table's arrays are reserved on the ledger by the builders that create one
    (:func:`hamiltonian_product_terms`, :meth:`from_terms`), released with the object.
    """

    __slots__ = ("coeff", "ptr", "modes", "matid", "mats", "_z_id", "_alloc",
                 "__weakref__")

    def __init__(self, coeff: np.ndarray, ptr: np.ndarray, modes: np.ndarray,
                 matid: np.ndarray, mats: Sequence[np.ndarray], *, reserve: bool = True):
        self.coeff = np.ascontiguousarray(coeff, dtype=np.complex128)
        self.ptr = np.ascontiguousarray(ptr, dtype=np.int64)
        self.modes = np.ascontiguousarray(modes, dtype=np.int32)
        self.matid = np.ascontiguousarray(matid, dtype=np.int32)
        self.mats = [np.ascontiguousarray(m, dtype=np.complex128) for m in mats]
        if self.ptr.size != self.coeff.size + 1 or self.modes.size != self.matid.size \
                or (self.ptr.size and int(self.ptr[-1]) != self.modes.size):
            raise ValueError("term table arrays do not align")
        self._z_id = -1
        for i, m in enumerate(self.mats):
            if m.shape == (2, 2) and np.array_equal(m, _Z):
                self._z_id = i
        self._alloc = None
        if reserve:
            self._alloc = res.reserve_owned(
                self, "product-term table ({} terms)".format(len(self)),
                self.nbytes / 1024.0 ** 3,
                note="{} (mode, matrix) pairs, {} distinct local matrices".format(
                    self.modes.size, len(self.mats)),
                advice=["the term count grows as the fourth power of the active-space "
                        "size; nothing but a smaller space changes it"])

    # -- construction ----------------------------------------------------------------------

    @classmethod
    def from_terms(cls, terms: Sequence[Optional[ProductTerm]], *,
                   reserve: bool = True) -> "TermTable":
        """The array form of a sequence of :class:`ProductTerm` (``None`` entries skipped),
        matrices interned by byte content."""
        ids: Dict[bytes, int] = {}
        mats: List[np.ndarray] = []
        coeff, ptr, modes, matid = [], [0], [], []
        for t in terms:
            if t is None:
                continue
            coeff.append(complex(t.coeff))
            for m, mat in zip(t.modes, t.mats):
                mat = np.ascontiguousarray(mat, dtype=np.complex128)
                key = mat.tobytes()
                k = ids.get(key)
                if k is None:
                    k = ids[key] = len(mats)
                    mats.append(mat)
                modes.append(int(m))
                matid.append(k)
            ptr.append(len(modes))
        return cls(np.asarray(coeff, dtype=np.complex128), np.asarray(ptr, dtype=np.int64),
                   np.asarray(modes, dtype=np.int32), np.asarray(matid, dtype=np.int32),
                   mats, reserve=reserve)

    @classmethod
    def coerce(cls, terms, *, reserve: bool = True) -> "TermTable":
        """``terms`` as a table: itself if it already is one, else :meth:`from_terms`."""
        if isinstance(terms, TermTable):
            return terms
        return cls.from_terms(terms, reserve=reserve)

    # -- the sequence view -----------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.coeff.size)

    def __getitem__(self, i) -> ProductTerm:
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        lo, hi = int(self.ptr[i]), int(self.ptr[i + 1])
        return ProductTerm(complex(self.coeff[i]),
                           tuple(int(m) for m in self.modes[lo:hi]),
                           tuple(self.mats[int(k)] for k in self.matid[lo:hi]))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return "TermTable({} terms, {} pairs, {} matrices)".format(
            len(self), self.modes.size, len(self.mats))

    # -- array queries ---------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """Exact bytes of the arrays (the matrices are a handful of small ones)."""
        return int(self.coeff.nbytes + self.ptr.nbytes + self.modes.nbytes
                   + self.matid.nbytes + sum(m.nbytes for m in self.mats))

    @property
    def z_id(self) -> int:
        """The id of the fermionic ``Z`` matrix among :attr:`mats`, or -1."""
        return self._z_id

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.ptr)

    @property
    def term_of_pair(self) -> np.ndarray:
        """Term index of every pair — the segment map every vectorized pass uses."""
        return np.repeat(np.arange(len(self), dtype=np.int64), self.counts)

    def all_modes(self) -> np.ndarray:
        return np.unique(self.modes)

    def subset(self, mask: np.ndarray, *, reserve: bool = False) -> "TermTable":
        """The terms selected by a boolean mask, sharing the matrix list."""
        mask = np.asarray(mask, dtype=bool)
        if mask.size != len(self):
            raise ValueError("mask length {} for {} terms".format(mask.size, len(self)))
        keep_pair = np.repeat(mask, self.counts)
        counts = self.counts[mask]
        ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return TermTable(self.coeff[mask], ptr, self.modes[keep_pair],
                         self.matid[keep_pair], self.mats, reserve=reserve)

    def restricted_to(self, modes: Sequence[int]) -> "TermTable":
        """The terms whose whole support lies inside ``modes``."""
        inside = self._inside_mask(modes)
        n_in = np.bincount(self.term_of_pair, weights=inside[self.modes],
                           minlength=len(self))
        return self.subset(n_in == self.counts)

    def _inside_mask(self, modes: Sequence[int]) -> np.ndarray:
        modes = np.asarray(list(modes), dtype=np.int64)
        top = max(int(self.modes.max()) if self.modes.size else 0,
                  int(modes.max()) if modes.size else 0)
        inside = np.zeros(top + 1, dtype=bool)
        inside[modes] = True
        return inside

    def crossing_terms(self, modes: Sequence[int]) -> np.ndarray:
        """Indices of terms whose *operator content* sits inside ``modes`` while their
        support leaks outside — a Jordan-Wigner string crossing another site's modes.
        ``Z`` factors are not content; anything else is."""
        inside = self._inside_mask(modes)
        term_of = self.term_of_pair
        core = self.matid != self._z_id
        n_in = np.bincount(term_of, weights=inside[self.modes], minlength=len(self))
        n_core = np.bincount(term_of, weights=core, minlength=len(self))
        n_core_in = np.bincount(term_of, weights=core & inside[self.modes],
                                minlength=len(self))
        bad = (n_in > 0) & (n_in < self.counts) & (n_core > 0) & (n_core_in == n_core)
        return np.nonzero(bad)[0]

    def with_coefficients(self, coeff: np.ndarray) -> "TermTable":
        """The same terms with new coefficients (the template's refill), sharing every
        other array."""
        coeff = np.ascontiguousarray(coeff, dtype=np.complex128)
        if coeff.size != len(self):
            raise ValueError("{} coefficients for {} terms".format(coeff.size, len(self)))
        return TermTable(coeff, self.ptr, self.modes, self.matid, self.mats, reserve=False)


def _dedupe(coeff: np.ndarray, ptr: np.ndarray, modes: np.ndarray, matid: np.ndarray,
            n_mats: int, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge terms with identical ``(modes, matid)`` sequences in first-occurrence order,
    summing coefficients in term order; drop merged coefficients ``<= tol``."""
    n = coeff.size
    if n == 0:
        return coeff, ptr, modes, matid
    counts = np.diff(ptr)
    width = int(counts.max()) if n else 0
    padded = np.full((n, width), -1, dtype=np.int64)
    term_of = np.repeat(np.arange(n, dtype=np.int64), counts)
    pos = np.arange(modes.size, dtype=np.int64) - np.repeat(ptr[:-1], counts)
    padded[term_of, pos] = modes.astype(np.int64) * int(n_mats) + matid
    _, first, inverse = np.unique(padded, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    merged = np.zeros(first.size, dtype=np.complex128)
    np.add.at(merged, inverse, coeff)
    order = np.argsort(first, kind="stable")             # first-occurrence order
    keep = order[np.abs(merged[order]) > tol]
    rep = first[keep]                                    # representative term per group
    sel = np.zeros(n, dtype=bool)
    sel[rep] = True
    keep_pair = np.repeat(sel, counts)
    new_counts = counts[rep]
    new_ptr = np.concatenate([[0], np.cumsum(new_counts)]).astype(np.int64)
    return merged[keep], new_ptr, modes[keep_pair], matid[keep_pair]


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


def term_build_gb(n_strings: int, n_modes: int, dedupe: bool) -> float:
    """Transient [GB] :func:`_fermion_table` holds at its peak for ``n_strings`` strings
    over ``n_modes`` modes: the local-id and support tables (a byte each per string and
    mode), the string and coefficient copies, the support's nonzero lists with the
    matrix ids gathered from them (three int64 per pair, a pair being a string and a mode
    of its support, at most half the modes on average), and — when the strings are
    consolidated — the dedupe's key table with its copies (:func:`dedupe_gb`). A model of
    the arrays live together, pinned against the measured peak in the tests."""
    build = (n_strings * n_modes * 2 + n_strings * (8 * 4 + 16 + 8 * 2)
             + n_strings * n_modes * 8 * 3 / 2) / 1024.0 ** 3
    return build + (dedupe_gb(n_strings, n_modes) if dedupe else 0.0)


def dedupe_gb(n_terms: int, width: int) -> float:
    """Transient [GB] :func:`_dedupe` holds at its peak — the padded ``(n_terms, width)``
    int64 key table and what the unique-rows call makes of it: measured at seven copies
    of the table (its structured view, the sort's buffer, the sorted copy, the mask
    gather, the unique rows and their reshaped view), plus the per-term sort order,
    first-occurrence index, inverse and counts."""
    return (n_terms * width * 8 * 7 + n_terms * 8 * 5) / 1024.0 ** 3


def _fermion_table(strings: np.ndarray, dag: Sequence[bool], coeff: np.ndarray, n: int,
                   *, tol: Optional[float], reserve: bool = True
                   ) -> Tuple[TermTable, np.ndarray]:
    """The table of a batch of fermionic strings with coefficients (module docstring:
    the vectorized :func:`fermion_term`, term by term identical to it), and the mask of
    the input strings that survived (nonzero coefficient, non-vanishing product).

    ``tol=None`` keeps every non-vanishing string as its own term (the template's full
    enumeration); a float consolidates exactly as :func:`consolidate` does.
    """
    coeff = np.asarray(coeff, dtype=np.complex128).reshape(-1)
    survived = coeff != 0.0
    strings = strings[survived]
    coeff = coeff[survived]
    res.require("product-term construction ({} strings)".format(strings.shape[0]),
                term_build_gb(strings.shape[0], n, tol is not None),
                note="per-string local matrix ids and support over {} modes".format(n),
                advice=["the string count grows as the fourth power of the active-space "
                        "size"])
    if strings.shape[0] == 0:
        return TermTable(np.zeros(0, dtype=np.complex128), np.zeros(1, dtype=np.int64),
                         np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                         list(_MONO_MATS), reserve=False), survived
    ids = _string_local_ids(strings, dag, n)
    alive = ~np.any(ids == _MONO_ZERO, axis=1)
    survived[np.nonzero(survived)[0][~alive]] = False
    ids = ids[alive]
    coeff = coeff[alive]
    strings = strings[alive]
    support = ids != _MONO_I
    counts = support.sum(axis=1)
    # a string that collapsed to the identity keeps a trivial support on its lowest mode
    collapsed = counts == 0
    if collapsed.any():
        lo = strings[collapsed].min(axis=1)
        support[np.nonzero(collapsed)[0], lo] = True
        counts = support.sum(axis=1)
    term_idx, mode_idx = np.nonzero(support)           # row-major: modes ascending per term
    matid = ids[term_idx, mode_idx]
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if tol is not None:
        coeff, ptr, mode_idx, matid = _dedupe(coeff, ptr, mode_idx, matid,
                                              len(_MONO_MATS), float(tol))
    return TermTable(coeff, ptr, mode_idx.astype(np.int32), matid.astype(np.int32),
                     list(_MONO_MATS), reserve=reserve), survived


def hamiltonian_product_terms(h: np.ndarray, eri: np.ndarray,
                              tol: float = 0.0) -> TermTable:
    """The ab initio Hamiltonian as consolidated product terms (module docstring convention).

    ``h`` Hermitian ``(n, n)``; ``eri`` ``(pq|rs)`` in chemists' notation, 4-fold symmetry
    only. ``tol`` screens *merged* coefficients — 0 by default: a dropped term is an
    approximation the caller must choose, never a default. Returns a :class:`TermTable`;
    term by term it is what :func:`fermion_term` + :func:`consolidate` produce, in the
    same order, built as arrays.
    """
    h = np.asarray(h)
    eri = np.asarray(eri)
    n = h.shape[0]
    if h.shape != (n, n) or eri.shape != (n, n, n, n):
        raise ValueError("h must be (n, n) and eri (n, n, n, n); got {} and {}"
                         .format(h.shape, eri.shape))
    one, _ = _fermion_table(_index_strings(n, 2), (True, False), h.reshape(-1), n,
                            tol=None, reserve=False)
    p, q, r, s = np.meshgrid(*[np.arange(n)] * 4, indexing="ij")
    two_strings = np.stack([p.ravel(), r.ravel(), s.ravel(), q.ravel()], axis=1)
    two, _ = _fermion_table(two_strings, (True, True, False, False),
                            0.5 * eri.reshape(-1), n, tol=None, reserve=False)
    coeff = np.concatenate([one.coeff, two.coeff])
    ptr = np.concatenate([one.ptr, one.ptr[-1] + two.ptr[1:]])
    modes = np.concatenate([one.modes, two.modes])
    matid = np.concatenate([one.matid, two.matid])
    width = int(np.diff(ptr).max()) if coeff.size else 0
    res.require("product-term consolidation ({} terms)".format(coeff.size),
                dedupe_gb(coeff.size, width) + (modes.size * 32 + coeff.size * 48)
                / 1024.0 ** 3,
                note="the tables of both electron counts, their concatenation and int64 "
                     "copies, and the dedupe's key table padded to {} pairs".format(width),
                advice=["the term count grows as the fourth power of the active-space "
                        "size"])
    coeff, ptr, modes, matid = _dedupe(coeff, ptr, modes.astype(np.int64),
                                       matid.astype(np.int64), len(_MONO_MATS), float(tol))
    return TermTable(coeff, ptr, modes, matid, list(_MONO_MATS))


def _index_strings(n: int, length: int) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(n)] * length, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def one_electron_product_terms(a: np.ndarray, tol: float = 0.0) -> TermTable:
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
    return _fermion_table(_index_strings(n, 2), (True, False), a.reshape(-1), n,
                          tol=float(tol))[0]


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
    zero = charge_lists[0][0].zero_like()
    states = list(itertools.product(*[range(bases[m].dim) for m in modes]))
    qns = [sum((cl[i] for cl, i in zip(charge_lists, st)), zero)
           for st in states]
    order = sorted(range(len(states)), key=lambda k: (qns[k], k))
    sectors: List[Tuple[QuantumNumber, int]] = []
    for k in order:
        if sectors and sectors[-1][0] == qns[k]:
            sectors[-1] = (qns[k], sectors[-1][1] + 1)
        else:
            sectors.append((qns[k], 1))
    return Space(sectors), np.array(order, dtype=np.int64)


def _resolve_bases(bases, all_modes: Sequence[int]) -> Dict[int, ModeBasis]:
    if bases is None:
        return {m: FERMION_MODE for m in all_modes}
    if isinstance(bases, ModeBasis):
        return {m: bases for m in all_modes}
    return {m: bases[m] for m in all_modes}


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
    :func:`_phys_space`. ``terms`` is the :class:`TermTable` it was compiled from and
    ``bases`` the per-mode bases, so a consumer that needs the *same operator on another
    partition of the same modes* — the local-multiplet extraction compiles it on the
    quotient tree of the sites — can, without a second statement of the operator.
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
    #: The quantum number each mode carries when **occupied**, in mode order. With particle
    #: number alone this is ``(1,)`` everywhere and says nothing; with irrep labels widening
    #: the quantum number it is what lets a consumer compute the sector of a determinant —
    #: which is how a guess and a target charge are built without re-deriving the labels.
    mode_charges: Tuple[QuantumNumber, ...] = ()
    #: memory reservations backing ``tensors`` (one per node). An owner that drops the TTNO
    #: in a limit-configured run releases these via ``res.BUDGET.release``; a cached TTNO
    #: (the reconnection compile cache) is genuinely resident and keeps them.
    allocations: List[object] = field(default_factory=list)
    terms: Optional[TermTable] = None
    bases: Optional[Dict[int, ModeBasis]] = None

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


def _sector_filter(bases, all_modes):
    """``allowed(p, q, r, s)`` — whether an excitation string conserves the group label.

    ``E_pq`` shifts the label by ``chg(p) - chg(q)`` and the two-electron string by
    ``chg(p) + chg(r) - chg(q) - chg(s)``; a nonzero shift is a term the group forbids, and
    every operator in one TTNO must carry the same total charge shift. Without widened labels
    every string conserves the particle number by construction and this is the constant
    ``True`` — so an unlabelled template enumerates exactly what it always did.
    """
    if bases is None:
        return lambda p, q, r, s: True
    if isinstance(bases, ModeBasis):
        bases = {m: bases for m in all_modes}
    chg = {m: bases[m].charges[1] for m in all_modes}
    zero = next(iter(chg.values())).zero_like() if chg else None
    if zero is None or zero.width <= 1:
        return lambda p, q, r, s: True

    def allowed(p, q, r, s):
        shift = chg[p] - chg[q]
        if r is not None:
            shift = shift + chg[r] - chg[s]
        return shift == zero

    return allowed


@dataclass(eq=False)
class NodeSlots:
    """Where each coefficient-carrying term lands at one node — the compiler's record of
    its own attachments, in arrays (:func:`compile_ttno`'s ``slots`` seam).

    ``tid`` are the terms attached here, ``key`` their coefficient transition (an index
    into ``out_sector``/``out_offset``/``in_sector``/``in_offset``, which give the bond-space
    positions of the transition's channels in W leg order ``[parent, children ascending]``)
    and ``pattern`` their local matrix pattern (an index into the node's pattern table:
    ``pat_ptr``/``pat_rows``/``pat_cols``/``pat_vals``, entries in **sector-sorted**
    physical indices).
    """

    node: int
    tid: np.ndarray
    key: np.ndarray
    pattern: np.ndarray
    out_sector: np.ndarray
    out_offset: np.ndarray
    in_sector: np.ndarray          #: (n_keys, n_children)
    in_offset: np.ndarray
    pat_ptr: np.ndarray
    pat_rows: np.ndarray
    pat_cols: np.ndarray
    pat_vals: np.ndarray


def compile_ttno(graph: NetworkGraph, terms, bases=None, root: int = 0,
                 slots: Optional[list] = None, attachments: Optional[list] = None) -> TTNO:
    """Compile a term sum into a TTNO on ``graph`` (module docstring: how and why exact).

    ``terms``: a :class:`TermTable`, or any sequence of :class:`ProductTerm` (``None``
    entries skipped), which is coerced into one. ``bases``: per-mode :class:`ModeBasis`
    mapping, or one basis for every mode; default :data:`FERMION_MODE`. ``root`` fixes the
    rooted orientation — any node works; the choice affects operator bond dimensions,
    never the operator.

    ``slots``, if a list, is filled with one :class:`NodeSlots` per node that attaches a
    coefficient. This is the seam :class:`TTNOTemplate` builds its refill/extraction tables
    on; the compiler invariant that every term attaches **exactly once** is asserted here
    rather than trusted. ``attachments`` is the same record in the older per-term tuple
    form ``(node, (out_sector, out_offset), ((in_sector, in_offset), ...))`` — kept for
    the tests that read it, never for a large operator.
    """
    ctx = compile_labels(graph, terms, bases=bases, root=root)
    table = ctx.table
    n_terms = len(table)
    attached = np.full(n_terms, -1, dtype=np.int64)
    tensors: List[SparseW] = [None] * graph.n_nodes
    allocations: List[object] = []
    for u in range(graph.n_nodes):
        tensors[u], node_slots = _node_tensor(u, ctx, allocations)
        if node_slots is not None:
            if np.any(attached[node_slots.tid] >= 0):     # pragma: no cover - invariant
                dup = int(node_slots.tid[attached[node_slots.tid] >= 0][0])
                raise AssertionError("term {} attached at nodes {} and {} — the "
                                     "monotonicity invariant is broken"
                                     .format(dup, attached[dup], u))
            attached[node_slots.tid] = u
            if slots is not None:
                slots.append(node_slots)
            if attachments is not None:
                for t, k in zip(node_slots.tid, node_slots.key):
                    attachments_entry = (u, (int(node_slots.out_sector[k]),
                                             int(node_slots.out_offset[k])),
                                         tuple((int(a), int(b)) for a, b in zip(
                                             node_slots.in_sector[k],
                                             node_slots.in_offset[k])))
                    while len(attachments) <= int(t):
                        attachments.append(None)
                    attachments[int(t)] = attachments_entry
    missing = np.nonzero(attached < 0)[0]
    if missing.size:                                       # pragma: no cover - invariant
        raise AssertionError("terms {} were never coefficient-attached — compiler "
                             "invariant broken".format(missing[:5].tolist()))

    ttno = TTNO(graph=graph, root=root, parent=ctx.parent, children=ctx.children,
                tensors=tensors, bond_space=tuple(ctx.bond_space),
                bond_labels=tuple(ctx.bond_labels),
                phys_space=tuple(p[0] for p in ctx.phys),
                phys_perm=tuple(p[1] for p in ctx.phys),
                node_modes=ctx.node_modes, mode_dims=ctx.mode_dims, charge=ctx.charge,
                mode_charges=tuple(ctx.bases[m].charges[1] if ctx.bases[m].dim > 1
                                   else ctx.zero for m in ctx.all_modes),
                allocations=allocations, terms=table, bases=ctx.bases)
    dims = ttno.bond_dimensions()
    if dims:
        log.debug("TTNO compiled: %d terms, max operator bond dimension %d",
                  len(table), max(dims.values()))
    return ttno


def context_gb(n_terms: int, n_pairs: int, n_nodes: int) -> float:
    """Exact size [GB] of the arrays :func:`compile_labels` keeps for the whole compile:
    the int64 mode, matrix-id, pair-code and term-of-pair maps and the int32 label
    table."""
    return (n_pairs * 8 * 4 + n_terms * 4 * n_nodes) / 1024.0 ** 3


def label_pass_gb(n_terms: int, n_pairs: int, width: int) -> float:
    """Transient [GB] one node of :func:`compile_labels` holds at its peak beyond the
    context: the per-pair masks and the selected-pair index, the per-term counts and
    kind masks, the padded label-key table, its sorted copy and its unique rows with the
    sort order and the inverse — the arrays live together in the unique-rows call. A
    model of the dominant arrays, pinned against the measured peak in the tests."""
    return (n_pairs * (3 + 8 + 8) + n_terms * (8 * 6 + 5 + 8 * 4)
            + n_terms * (width + 1) * 8 * 3) / 1024.0 ** 3


def node_transitions_gb(n_terms: int, n_modes: int, n_children: int
                        ) -> Tuple[float, float]:
    """``(resident, transient)`` [GB] of :func:`node_transitions`: what the record keeps
    for the node's build (the per-term channel indices, masks, pattern ids and the
    coefficient terms' key), and what building it holds beside that at its peak (the
    per-term pattern rows with their unique copy, sort order and inverse, and the
    per-term key rows with theirs). A model, pinned in the tests."""
    nm = max(n_modes, 1)
    resident = n_terms * (8 * (1 + n_children) + 2 + 8 + 16)
    transient = n_terms * (8 * nm * 3 + 8 * 2 + 8 * (1 + n_children) * 3 + 8 * 2)
    return resident / 1024.0 ** 3, transient / 1024.0 ** 3


def compile_labels(graph: NetworkGraph, terms, bases=None, root: int = 0) -> "_CompileContext":
    """The compiler's first pass: every term's content label on every bond, the bond
    spaces and the physical spaces — everything :func:`_node_tensor` builds a tensor
    from, and everything a consumer that never needs the tensors (the local-multiplet
    site blocks, :mod:`kuiva.dmrg.manifold`) needs instead."""
    table = TermTable.coerce(terms)
    if len(table) == 0:
        raise ValueError("no terms to compile")
    all_modes = sorted(m for c in graph.contents for m in c)
    mode_set = set(all_modes)
    n_total_modes = len(all_modes)
    bases = _resolve_bases(bases, all_modes)
    width = bases[all_modes[0]].charges[0].width if all_modes else 1
    # ⚠ The identity is taken **from the bases**, not built from the width alone: a cyclic
    # component carries its modulus on the label, and an identity without it would poison
    # every sum it takes part in with plain integer arithmetic (:class:`QuantumNumber`).
    zero = (bases[all_modes[0]].charges[0].zero_like() if all_modes
            else QuantumNumber.zero(width))

    # --- validate the modes, derive per-(mode, matrix) charge shifts ----------------------
    modes = table.modes.astype(np.int64)
    matid = table.matid.astype(np.int64)
    if modes.size and (np.any(modes < 0) or not set(np.unique(modes).tolist()) <= mode_set):
        bad = sorted(set(np.unique(modes).tolist()) - mode_set)
        raise ValueError("term touches mode {} which no node carries".format(bad[0]))
    mats = table.mats
    n_mats = len(mats)
    pair_kind = np.unique(np.stack([modes, matid], axis=1), axis=0) if modes.size \
        else np.zeros((0, 2), dtype=np.int64)
    shifts: Dict[Tuple[int, int], QuantumNumber] = {}
    shift_rows = np.zeros((pair_kind.shape[0], width), dtype=np.int64)
    for k, (m, mid) in enumerate(pair_kind):
        qn = _charge_shift(mats[int(mid)], bases[int(m)])
        shifts[(int(m), int(mid))] = qn
        shift_rows[k] = np.asarray(tuple(qn), dtype=np.int64)
    # every term's total shift must be one quantum number
    pair_code = modes * n_mats + matid
    kind_code = pair_kind[:, 0] * n_mats + pair_kind[:, 1]
    kind_of_pair = np.searchsorted(kind_code, pair_code)
    term_of = table.term_of_pair
    n_terms = len(table)
    total = np.zeros((n_terms, width), dtype=np.int64)
    for c in range(width):
        total[:, c] = np.bincount(term_of, weights=shift_rows[kind_of_pair, c],
                                  minlength=n_terms).astype(np.int64)
    del kind_of_pair
    charge = zero.like(tuple(int(x) for x in total[0]))
    distinct = sorted({zero.like(tuple(int(x) for x in row))
                       for row in np.unique(total, axis=0)})
    if len(distinct) != 1:
        other = distinct[1] if distinct[0] == charge else distinct[0]
        raise ValueError("terms carry different total charge shifts ({} vs {}): they "
                         "cannot share one TTNO".format(charge, other))
    z_id = table.z_id

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
    mode_index = np.full(max(all_modes) + 1 if all_modes else 1, -1, dtype=np.int64)
    mode_index[np.asarray(all_modes, dtype=np.int64)] = np.arange(n_total_modes)

    # --- pass 1: content label of every term on every bond --------------------------------
    ID, H = ("1",), ("H",)
    registries: List[Dict[tuple, int]] = [dict() for _ in range(graph.n_nodes)]
    lab_idx = np.zeros((n_terms, graph.n_nodes), dtype=np.int32)
    counts = table.counts
    nz_pair = matid != z_id
    max_width = int(counts.max()) if n_terms else 0
    alloc = res.reserve("TTNO compile context ({} terms)".format(n_terms),
                        context_gb(n_terms, modes.size, graph.n_nodes),
                        note="mode, matrix and term maps of {} pairs and the label table "
                             "over {} nodes".format(modes.size, graph.n_nodes),
                        advice=["the term count grows as the fourth power of the "
                                "active-space size"])
    res.require("TTNO label pass ({} terms)".format(n_terms),
                label_pass_gb(n_terms, modes.size, max_width),
                note="per-term label keys of one bond, padded to {} pairs".format(
                    max_width),
                advice=["the term count grows as the fourth power of the active-space "
                        "size"])
    for u in range(graph.n_nodes):
        if u == root:
            continue
        ins = np.zeros(n_total_modes, dtype=bool)
        ins[mode_index[np.asarray(sorted(subtree[u]), dtype=np.int64)]] = True
        pair_in = ins[mode_index[modes]]
        n_in = np.bincount(term_of, weights=pair_in, minlength=n_terms).astype(np.int64)
        nzi = np.bincount(term_of, weights=pair_in & nz_pair, minlength=n_terms)
        nzo = np.bincount(term_of, weights=(~pair_in) & nz_pair, minlength=n_terms)
        is_id = n_in == 0
        is_h = n_in == counts
        is_l = ~is_id & ~is_h & ((nzi < nzo) | ((nzi == nzo)
                                                & (2 * len(subtree[u]) <= n_total_modes)))
        is_r = ~is_id & ~is_h & ~is_l
        reg = registries[u]
        if is_id.any():
            reg[ID] = len(reg)
            lab_idx[is_id, u] = reg[ID]
        if is_h.any():
            reg[H] = len(reg)
            lab_idx[is_h, u] = reg[H]
        lr = is_l | is_r
        if lr.any():
            sel = (is_l[term_of] & pair_in) | (is_r[term_of] & ~pair_in)
            sel_idx = np.nonzero(sel)[0]
            cnt = np.bincount(term_of[sel_idx], minlength=n_terms)
            width_lr = int(cnt.max())
            lr_idx = np.nonzero(lr)[0]
            rows = np.full((lr_idx.size, width_lr + 1), -1, dtype=np.int64)
            rows[:, 0] = np.where(is_r[lr_idx], 1, 0)
            row_of_term = np.full(n_terms, -1, dtype=np.int64)
            row_of_term[lr_idx] = np.arange(lr_idx.size)
            starts = np.cumsum(cnt) - cnt
            pos = np.arange(sel_idx.size, dtype=np.int64) - np.repeat(starts, cnt)
            rows[row_of_term[term_of[sel_idx]], pos + 1] = pair_code[sel_idx]
            del sel, pos
            uniq, inverse = np.unique(rows, axis=0, return_inverse=True)
            del rows
            inverse = np.asarray(inverse).reshape(-1)
            base = len(reg)
            for k, row in enumerate(uniq):
                codes = row[1:][row[1:] >= 0]
                pairs = tuple((int(c // n_mats), int(c % n_mats)) for c in codes)
                if row[0] == 0:
                    label = ("L", pairs)
                else:
                    dn = charge
                    for mm, mid in pairs:
                        dn = dn - shifts[(mm, mid)]
                    label = ("R", pairs, dn)
                reg[label] = base + k
            lab_idx[lr_idx, u] = (base + inverse).astype(np.int32)

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
    pos_sector: List[np.ndarray] = [None] * graph.n_nodes
    pos_offset: List[np.ndarray] = [None] * graph.n_nodes
    for u in range(graph.n_nodes):
        if u == root:
            bond_space[u] = Space([(charge, 1)])
            bond_labels[u] = (H,)
            pos_sector[u] = np.zeros(1, dtype=np.int64)
            pos_offset[u] = np.zeros(1, dtype=np.int64)
            continue
        labels = list(registries[u])
        order = sorted(range(len(labels)), key=lambda i: (label_charge(labels[i]),
                                                          labels[i]))
        sectors: List[Tuple[QuantumNumber, int]] = []
        psec = np.zeros(len(labels), dtype=np.int64)
        poff = np.zeros(len(labels), dtype=np.int64)
        for i in order:
            qn = label_charge(labels[i])
            if sectors and sectors[-1][0] == qn:
                sectors[-1] = (qn, sectors[-1][1] + 1)
            else:
                sectors.append((qn, 1))
            idx = registries[u][labels[i]]
            psec[idx] = len(sectors) - 1
            poff[idx] = sectors[-1][1] - 1
        bond_space[u] = Space(sectors)
        bond_labels[u] = tuple(labels[i] for i in order)
        pos_sector[u] = psec
        pos_offset[u] = poff

    # --- physical spaces ------------------------------------------------------------------
    node_modes = tuple(tuple(sorted(graph.contents[u])) for u in range(graph.n_nodes))
    mode_dims = tuple(tuple(bases[m].dim for m in node_modes[u])
                      for u in range(graph.n_nodes))
    phys: List[Tuple[Space, np.ndarray]] = [
        _phys_space(node_modes[u], bases, width) for u in range(graph.n_nodes)]

    carrying = [np.zeros(len(reg) if u != root else 1, dtype=bool)
                for u, reg in enumerate(registries)]
    for u in range(graph.n_nodes):
        if u == root:
            carrying[u][0] = True
            continue
        for label, idx in registries[u].items():
            carrying[u][idx] = label[0] in ("H", "R")

    ctx = _CompileContext(table=table, mats=mats, bases=bases, zero=zero, root=root,
                          parent=parent, children=children, registries=registries,
                          carrying=carrying, lab_idx=lab_idx, pos_sector=pos_sector,
                          pos_offset=pos_offset, bond_space=bond_space,
                          bond_labels=bond_labels, node_modes=node_modes,
                          mode_dims=mode_dims, phys=phys, term_of=term_of, modes=modes,
                          matid=matid, charge=charge, all_modes=all_modes)
    # the context's arrays live as long as the context: released with it
    res.owned_by(ctx, alloc)
    return ctx


@dataclass(eq=False)
class _CompileContext:
    """Everything the label pass produced: what :func:`_node_tensor` reads, and what a
    consumer of the labels alone (the site blocks of :mod:`kuiva.dmrg.manifold`) reads."""

    table: TermTable
    mats: List[np.ndarray]
    bases: Dict[int, ModeBasis]
    zero: QuantumNumber
    root: int
    parent: Tuple[int, ...]
    children: Tuple[Tuple[int, ...], ...]
    registries: List[Dict[tuple, int]]
    carrying: List[np.ndarray]
    lab_idx: np.ndarray
    pos_sector: List[np.ndarray]
    pos_offset: List[np.ndarray]
    bond_space: List[Space]
    bond_labels: List[Tuple[tuple, ...]]
    node_modes: Tuple[Tuple[int, ...], ...]
    mode_dims: Tuple[Tuple[int, ...], ...]
    phys: List[Tuple[Space, np.ndarray]]
    term_of: np.ndarray
    modes: np.ndarray
    matid: np.ndarray
    charge: QuantumNumber
    all_modes: List[int]


@dataclass(eq=False)
class NodeTransitions:
    """One node's transitions and local patterns, before any table is built
    (:func:`node_transitions`): the per-term channel indices, which terms carry a
    coefficient here, the distinct local patterns (``-1`` = identity per mode) with the
    pattern of every kept term, and the distinct unit and coefficient transition keys
    ``(out label, in labels...)`` with their members."""

    out_idx: np.ndarray
    in_idx: np.ndarray               #: (n_terms, n_children)
    keep: np.ndarray
    coeff_mask: np.ndarray
    patterns: np.ndarray             #: (n_patterns, n_modes)
    pat_of_term: np.ndarray          #: -1 where the term skips the node
    nnz_pat: np.ndarray
    identity_pattern: int            #: index, or -1 when no term skips the node
    unit_keys: np.ndarray            #: (n_unit, 1 + n_children), the identity key last if any
    unit_pat: np.ndarray             #: pattern per unit key
    coeff_keys: np.ndarray           #: (n_coeff, 1 + n_children)
    coeff_idx: np.ndarray            #: the coefficient terms
    coeff_inv: np.ndarray            #: key per coefficient term
    #: the ledger entry of the arrays above, released by whoever consumes the record
    alloc: object = None


#: Bytes one nonzero of a pattern or merged coefficient table costs: int64 row, int64
#: column, complex128 value.
_ENTRY_BYTES = 32


def pattern_table_gb(nnz_per_pattern: Sequence[int]) -> float:
    """Exact size [GB] of a node's local-pattern tables — one ``(row, column, value)``
    triple per nonzero of every distinct pattern, plus the pattern pointer (exact sizing
    function; pinned two-sided against the built arrays in the tests)."""
    nnz = np.asarray(nnz_per_pattern, dtype=np.int64)
    return (_ENTRY_BYTES * float(nnz.sum()) + 8.0 * (nnz.size + 1)) / 1024.0 ** 3


def coefficient_chunk_gb(n_entries: int) -> float:
    """Transient [GB] one chunk of ``n_entries`` unmerged coefficient contributions holds
    at its peak, in :func:`_node_tensor` and :func:`_merge_coefficients`: the gather
    index and the pattern gathers while the composite key and value are built (two
    int64 and one complex per entry beside the two being made), then the composite key
    and value with the sort order, the sort's own buffer and the sorted copies. A model
    of the arrays live together, pinned against the measured peak in the tests."""
    return (n_entries * (8 + 8 + 8 + 16 + 16 + 8 + 4 + 8 + 16)) / 1024.0 ** 3


def assembly_gb(nnz: int, ndim: int) -> float:
    """Exact transient [GB] :func:`_assemble` holds while turning a node's transitions
    into one sorted entry set: the row table, flat index and value of every entry, its
    block key and the sort order, the sorted copies of all four — the unsorted ones
    stay referenced by the caller until the sort has produced the tensor's arrays — and
    the sort's own key buffer with the duplicate-check masks. A model of the arrays
    live together, pinned against the measured peak in the tests."""
    return (nnz * (2 * (8 * ndim + 8 + 16 + 8) + 24)) / 1024.0 ** 3


def _pattern_entries(pattern_row: np.ndarray, modes_u: Sequence[int], bases, mats,
                     inv_perm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The nonzeros of one local pattern (``-1`` = identity per mode) as
    ``(rows, cols, vals)`` in **sector-sorted** physical indices."""
    r = np.zeros(1, dtype=np.int64)
    c = np.zeros(1, dtype=np.int64)
    v = np.ones(1, dtype=np.complex128)
    for m, mid in zip(modes_u, pattern_row):
        d = bases[m].dim
        if mid < 0:
            fr = np.arange(d, dtype=np.int64)
            fc = fr
            fv = np.ones(d, dtype=np.complex128)
        else:
            fr, fc = np.nonzero(mats[int(mid)])
            fv = mats[int(mid)][fr, fc]
        r = (r[:, None] * d + fr[None, :]).reshape(-1)
        c = (c[:, None] * d + fc[None, :]).reshape(-1)
        v = (v[:, None] * fv[None, :]).reshape(-1)
    return inv_perm[r], inv_perm[c], v


def _merge_coefficients(comp: np.ndarray, vals: np.ndarray, d: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum contributions to the same composite ``(key * d + row) * d + col`` — a stable
    sorted reduction, so the terms of one transition add in term order — returning
    ``(key, rows, cols, merged values)`` (:func:`coefficient_chunk_gb` is this routine's
    peak, written to match it)."""
    order = np.argsort(comp, kind="stable")
    comp = comp[order]
    vals = vals[order]
    del order
    if comp.size == 0:
        return (np.zeros(0, dtype=np.int64),) * 3 + (vals,)
    new = np.ones(comp.size, dtype=bool)
    new[1:] = comp[1:] != comp[:-1]
    starts = np.nonzero(new)[0]
    merged = np.add.reduceat(vals, starts)
    comp_u = comp[starts]
    cols_u = comp_u % d
    rest = comp_u // d
    rows_u = rest % d
    key_u = rest // d
    return key_u, rows_u, cols_u, merged


def node_transitions(u: int, ctx: _CompileContext) -> NodeTransitions:
    """The transitions and local patterns of one node from the label pass, vectorized
    over the term table (:class:`NodeTransitions`)."""
    table, mats, bases = ctx.table, ctx.mats, ctx.bases
    root, children_u = ctx.root, ctx.children[u]
    n_terms = len(table)
    modes_u = ctx.node_modes[u]
    nm = len(modes_u)
    nc = len(children_u)
    kept_gb, build_gb = node_transitions_gb(n_terms, nm, nc)
    alloc = res.reserve("TTNO node {} transitions".format(u), kept_gb,
                        note="per-term channels, masks and patterns over {} terms".format(
                            n_terms),
                        advice=["the term count grows as the fourth power of the "
                                "active-space size"])
    res.require("TTNO node {} transition build".format(u), build_gb,
                note="per-term pattern and key rows with their unique copies",
                advice=["the term count grows as the fourth power of the active-space "
                        "size"])
    if u == root:
        out_idx = np.zeros(n_terms, dtype=np.int64)
        keep = np.ones(n_terms, dtype=bool)
        out_car = np.ones(n_terms, dtype=bool)
    else:
        out_idx = ctx.lab_idx[:, u].astype(np.int64)
        id_index = ctx.registries[u].get(("1",), -1)
        keep = out_idx != id_index
        out_car = ctx.carrying[u][out_idx]
    in_idx = np.zeros((n_terms, nc), dtype=np.int64)
    in_car = np.zeros(n_terms, dtype=bool)
    for j, c in enumerate(children_u):
        in_idx[:, j] = ctx.lab_idx[:, c]
        in_car |= ctx.carrying[c][in_idx[:, j]]
    coeff_mask = keep & out_car & ~in_car
    unit_mask = keep & ~coeff_mask
    identity_needed = not bool(keep.all())

    # --- local patterns: the matrix id per node mode, per term ----------------------------
    top = max(int(ctx.modes.max()) if ctx.modes.size else 0,
              max(modes_u) if modes_u else 0)
    pos_of_mode = np.full(top + 1, -1, dtype=np.int64)
    pos_of_mode[np.asarray(modes_u, dtype=np.int64)] = np.arange(nm)
    pat = np.full((n_terms, max(nm, 1)), -1, dtype=np.int64)
    sel = pos_of_mode[ctx.modes] >= 0
    pat[ctx.term_of[sel], pos_of_mode[ctx.modes[sel]]] = ctx.matid[sel]
    kept_idx = np.nonzero(keep)[0]
    rows_in = pat[kept_idx]
    if identity_needed:
        rows_in = np.vstack([rows_in, np.full((1, rows_in.shape[1]), -1, dtype=np.int64)])
    patterns, pat_inv = np.unique(rows_in, axis=0, return_inverse=True)
    pat_inv = np.asarray(pat_inv).reshape(-1)
    pat_of_term = np.full(n_terms, -1, dtype=np.int64)
    pat_of_term[kept_idx] = pat_inv[:kept_idx.size]
    identity_pattern = int(pat_inv[-1]) if identity_needed else -1
    n_pat = patterns.shape[0]
    nnz_mat = np.array([int(np.count_nonzero(m)) for m in mats] + [0], dtype=np.int64)
    dims_u = np.array([bases[m].dim for m in modes_u], dtype=np.int64)
    nnz_pat = np.ones(n_pat, dtype=np.int64)
    for j in range(nm):
        col = patterns[:, j]
        nnz_pat *= np.where(col < 0, dims_u[j], nnz_mat[np.where(col < 0, -1, col)])
    if nm == 0:
        nnz_pat[:] = 1

    # --- transitions: unique (out, in) rows, unit and coefficient -------------------------
    key_rows = np.column_stack([out_idx, in_idx]) if nc else out_idx.reshape(-1, 1)
    unit_idx = np.nonzero(unit_mask)[0]
    if unit_idx.size:
        unit_keys, unit_first, unit_inv = np.unique(key_rows[unit_idx], axis=0,
                                                    return_index=True, return_inverse=True)
        unit_inv = np.asarray(unit_inv).reshape(-1)
        unit_pat = pat_of_term[unit_idx[unit_first]]
        if np.any(pat_of_term[unit_idx] != unit_pat[unit_inv]):
            raise AssertionError("unit transition at node {} is not label-determined "
                                 "— compiler invariant broken".format(u))
    else:
        unit_keys = np.zeros((0, 1 + nc), dtype=np.int64)
        unit_pat = np.zeros(0, dtype=np.int64)
    if identity_needed:
        id_row = np.array([[ctx.registries[u][("1",)]]
                           + [ctx.registries[c][("1",)] for c in children_u]],
                          dtype=np.int64)
        unit_keys = np.vstack([unit_keys, id_row])
        unit_pat = np.concatenate([unit_pat, [identity_pattern]])
    coeff_idx = np.nonzero(coeff_mask)[0]
    if coeff_idx.size:
        coeff_keys, coeff_inv = np.unique(key_rows[coeff_idx], axis=0, return_inverse=True)
        coeff_inv = np.asarray(coeff_inv).reshape(-1)
    else:
        coeff_keys = np.zeros((0, 1 + nc), dtype=np.int64)
        coeff_inv = np.zeros(0, dtype=np.int64)
    return NodeTransitions(out_idx=out_idx, in_idx=in_idx, keep=keep,
                           coeff_mask=coeff_mask, patterns=patterns,
                           pat_of_term=pat_of_term, nnz_pat=nnz_pat,
                           identity_pattern=identity_pattern, unit_keys=unit_keys,
                           unit_pat=unit_pat, coeff_keys=coeff_keys, coeff_idx=coeff_idx,
                           coeff_inv=coeff_inv, alloc=alloc)


def pattern_tables(u: int, ctx: _CompileContext, tr: NodeTransitions
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """The node's local patterns as sector-sorted nonzero lists
    ``(pat_ptr, pat_rows, pat_cols, pat_vals)``, reserved at their exact size before they
    are built; the reservation is returned for the caller to release."""
    modes_u = ctx.node_modes[u]
    _, perm = ctx.phys[u]
    inv_perm = np.argsort(perm)
    nnz_pat = tr.nnz_pat
    alloc = res.reserve(
        "TTNO node {} local patterns".format(u), pattern_table_gb(nnz_pat),
        note="{} distinct patterns over {} modes, {} nonzeros".format(
            nnz_pat.size, len(modes_u), int(nnz_pat.sum())),
        advice=["a finer node partition (fewer modes per node): a pattern's nonzeros "
                "grow as 2^(modes at the node)"])
    pat_ptr = np.concatenate([[0], np.cumsum(nnz_pat)]).astype(np.int64)
    pat_rows = np.empty(int(pat_ptr[-1]), dtype=np.int64)
    pat_cols = np.empty(int(pat_ptr[-1]), dtype=np.int64)
    pat_vals = np.empty(int(pat_ptr[-1]), dtype=np.complex128)
    for k in range(nnz_pat.size):
        r, c, v = _pattern_entries(tr.patterns[k], modes_u, ctx.bases, ctx.mats, inv_perm)
        if r.size != nnz_pat[k]:                          # pragma: no cover - arithmetic
            raise AssertionError("pattern nonzero count mismatch at node {}".format(u))
        pat_rows[pat_ptr[k]:pat_ptr[k + 1]] = r
        pat_cols[pat_ptr[k]:pat_ptr[k + 1]] = c
        pat_vals[pat_ptr[k]:pat_ptr[k + 1]] = v
    return pat_ptr, pat_rows, pat_cols, pat_vals, alloc


def _node_tensor(u: int, ctx: _CompileContext, allocations: List[object]
                 ) -> Tuple[SparseW, Optional[NodeSlots]]:
    """Assemble one W tensor from the label bookkeeping (see :func:`compile_ttno`).

    Vectorized over the term table (:func:`node_transitions`), and every table is
    reserved at its exact size before it is built — the patterns as nonzero lists
    (:func:`pattern_table_gb`), the coefficient sums in chunks bounded by the transient
    allowance (:func:`coefficient_chunk_gb`), the assembled entry set
    (:func:`assembly_gb`) and the sparse tensor itself (:func:`sparse_w_gb`). Every
    reservation of a table that does not outlive the compile is released on the way
    out, on success and on a refusal alike.
    """
    table = ctx.table
    children_u = ctx.children[u]
    phys_sp, _ = ctx.phys[u]
    d = phys_sp.total_dim
    nc = len(children_u)
    tr = node_transitions(u, ctx)
    p_off = phys_sp.offsets
    sec_of = np.repeat(np.arange(phys_sp.nsectors, dtype=np.int64), phys_sp.dims)
    loc_of = np.arange(d, dtype=np.int64) - p_off[sec_of]
    held: List[object] = [tr.alloc]
    try:
        pat_ptr, pat_rows, pat_cols, pat_vals, pat_alloc = pattern_tables(u, ctx, tr)
        held.append(pat_alloc)
        nnz_pat = tr.nnz_pat

        # --- coefficient transitions, merged in chunks under the transient allowance ------
        c_key = np.zeros(0, dtype=np.int64)
        c_rows = np.zeros(0, dtype=np.int64)
        c_cols = np.zeros(0, dtype=np.int64)
        c_vals = np.zeros(0, dtype=np.complex128)
        coeff_idx, coeff_inv, coeff_keys = tr.coeff_idx, tr.coeff_inv, tr.coeff_keys
        n_coeff = coeff_keys.shape[0]
        if coeff_idx.size:
            order = np.argsort(coeff_inv, kind="stable")  # by key, term order inside a key
            t_sorted = coeff_idx[order]
            k_sorted = coeff_inv[order]
            p_sorted = tr.pat_of_term[t_sorted]
            n_each = nnz_pat[p_sorted]
            # Blocking is decided once from the budget, outside the loop, and never on a
            # per-chunk check: a chunk holds whole keys, so a key whose own contributions
            # exceed the allowance is one chunk by itself and is required explicitly.
            budget_entries = max(1, int(res.transient_gb() / coefficient_chunk_gb(1)))
            key_bounds = np.concatenate([[0], np.nonzero(np.diff(k_sorted))[0] + 1,
                                         [k_sorted.size]])
            key_cost = np.add.reduceat(n_each, key_bounds[:-1])
            chunks: List[Tuple[int, int]] = []
            start = 0
            acc = 0
            for i, cost in enumerate(key_cost):
                if acc and acc + cost > budget_entries:
                    chunks.append((start, i))
                    start, acc = i, 0
                acc += int(cost)
            chunks.append((start, len(key_cost)))
            parts = []
            for a, b in chunks:
                lo, hi = int(key_bounds[a]), int(key_bounds[b])
                n_entries = int(n_each[lo:hi].sum())
                res.require("TTNO node {} coefficient accumulation".format(u),
                            coefficient_chunk_gb(n_entries),
                            note="{} contributions over {} transitions in one chunk".format(
                                n_entries, b - a),
                            advice=["a finer node partition: contributions per transition "
                                    "grow with the local dimension"])
                reps = n_each[lo:hi]
                starts = pat_ptr[p_sorted[lo:hi]]
                idx = np.repeat(starts - (np.cumsum(reps) - reps), reps)
                idx += np.arange(n_entries, dtype=np.int64)
                comp = pat_rows[idx]
                comp *= d
                comp += pat_cols[idx]
                comp += np.repeat(k_sorted[lo:hi] * (d * d), reps)
                vals = pat_vals[idx]
                vals *= np.repeat(table.coeff[t_sorted[lo:hi]], reps)
                del idx
                part = _merge_coefficients(comp, vals, d)
                del comp, vals
                # the merged chunk is within the transient just required; it goes on the
                # ledger now, before the next chunk's transient is checked against it
                held.append(res.reserve(
                    "TTNO node {} coefficient transitions (chunk)".format(u),
                    (_ENTRY_BYTES + 8.0) * part[0].size / 1024.0 ** 3,
                    note="{} merged nonzeros".format(part[0].size)))
                parts.append(part)
            total_merged = int(sum(p[0].size for p in parts))
            if len(parts) > 1:
                res.require("TTNO node {} coefficient transitions (join)".format(u),
                            (_ENTRY_BYTES + 8.0) * total_merged / 1024.0 ** 3,
                            note="the chunks concatenated into one table",
                            advice=["a finer node partition: a node's merged coefficient "
                                    "transitions grow with its local dimension squared"])
            c_key = np.concatenate([p[0] for p in parts])
            c_rows = np.concatenate([p[1] for p in parts])
            c_cols = np.concatenate([p[2] for p in parts])
            c_vals = np.concatenate([p[3] for p in parts])
            del parts
            log.debug("TTNO node %d: %d merged coefficient nonzeros over %d transitions",
                      u, c_key.size, n_coeff)

        # --- assemble the sparse entry set ------------------------------------------------
        par_space = ctx.bond_space[u]
        child_spaces = [ctx.bond_space[c] for c in children_u]
        spaces = (par_space,) + tuple(child_spaces) + (phys_sp, phys_sp)
        signs = (-1,) + tuple(1 for _ in children_u) + (1, -1)
        ndim = len(spaces)
        psec_u, poff_u = ctx.pos_sector[u], ctx.pos_offset[u]
        psec_c = [ctx.pos_sector[c] for c in children_u]
        poff_c = [ctx.pos_offset[c] for c in children_u]
        unit_keys, unit_pat = tr.unit_keys, tr.unit_pat
        n_unit = unit_keys.shape[0]
        unit_nnz = nnz_pat[unit_pat] if n_unit else np.zeros(0, dtype=np.int64)
        nnz = int(unit_nnz.sum()) + int(c_key.size)
        if nnz == 0:                                      # pragma: no cover - a node acts
            raise AssertionError("node {} compiled to an empty operator".format(u))
        res.require("TTNO node {} entry assembly".format(u), assembly_gb(nnz, ndim),
                    note="{} entries, {} legs".format(nnz, ndim),
                    advice=["a finer node partition (fewer modes per node)"])

        def channel_arrays(keys: np.ndarray, reps: np.ndarray):
            """Per-entry sector rows and head index of the op legs, for keys repeated."""
            osec = np.repeat(psec_u[keys[:, 0]], reps)
            head = np.repeat(poff_u[keys[:, 0]], reps)
            sec_cols = [osec]
            for j in range(nc):
                cs = np.repeat(psec_c[j][keys[:, 1 + j]], reps)
                co = np.repeat(poff_c[j][keys[:, 1 + j]], reps)
                head = head * child_spaces[j].dims[cs] + co
                sec_cols.append(cs)
            return sec_cols, head

        if n_unit:
            starts = pat_ptr[unit_pat]
            idx = np.repeat(starts - (np.cumsum(unit_nnz) - unit_nnz), unit_nnz) \
                + np.arange(int(unit_nnz.sum()), dtype=np.int64)
            u_r, u_c, u_v = pat_rows[idx], pat_cols[idx], pat_vals[idx]
            u_secs, u_head = channel_arrays(unit_keys, unit_nnz)
        else:
            u_r = u_c = np.zeros(0, dtype=np.int64)
            u_v = np.zeros(0, dtype=np.complex128)
            u_secs, u_head = [np.zeros(0, dtype=np.int64)] * (1 + nc), \
                np.zeros(0, dtype=np.int64)
        if c_key.size:
            c_secs, c_head = channel_arrays(coeff_keys[c_key],
                                            np.ones(c_key.size, dtype=np.int64))
        else:
            c_secs, c_head = [np.zeros(0, dtype=np.int64)] * (1 + nc), \
                np.zeros(0, dtype=np.int64)
        a_r = np.concatenate([u_r, c_rows])
        a_c = np.concatenate([u_c, c_cols])
        a_v = np.concatenate([u_v, c_vals])
        head = np.concatenate([u_head, c_head])
        rows = np.empty((nnz, ndim), dtype=np.int64)
        for j in range(1 + nc):
            rows[:, j] = np.concatenate([u_secs[j], c_secs[j]])
        del u_r, u_c, u_v, c_rows, c_cols, c_vals, c_key, u_secs, c_secs, u_head, c_head
        rows[:, ndim - 2] = sec_of[a_r]
        rows[:, ndim - 1] = sec_of[a_c]
        flat = head * phys_sp.dims[rows[:, ndim - 2]]
        flat += loc_of[a_r]
        flat *= phys_sp.dims[rows[:, ndim - 1]]
        flat += loc_of[a_c]
        del head, a_r, a_c
        tensor = _assemble(spaces, signs, ctx.zero, rows, flat, a_v, u, allocations)
        del rows, flat, a_v
    finally:
        # the tables were transients of this compile: the tensor holds what the operator
        # needs, and the template copies what the slot record refers to
        for alloc in held:
            res.BUDGET.release(alloc)

    node_slots = None
    if coeff_idx.size:
        node_slots = NodeSlots(node=u, tid=coeff_idx, key=coeff_inv,
                               pattern=tr.pat_of_term[coeff_idx],
                               out_sector=psec_u[coeff_keys[:, 0]],
                               out_offset=poff_u[coeff_keys[:, 0]],
                               in_sector=np.column_stack(
                                   [psec_c[j][coeff_keys[:, 1 + j]] for j in range(nc)])
                               if nc else np.zeros((n_coeff, 0), dtype=np.int64),
                               in_offset=np.column_stack(
                                   [poff_c[j][coeff_keys[:, 1 + j]] for j in range(nc)])
                               if nc else np.zeros((n_coeff, 0), dtype=np.int64),
                               pat_ptr=pat_ptr, pat_rows=pat_rows, pat_cols=pat_cols,
                               pat_vals=pat_vals)
    return tensor, node_slots


def _assemble(spaces, signs, zero, rows: np.ndarray, flat: np.ndarray, vals: np.ndarray,
              u: int, allocations: List[object]) -> SparseW:
    """One sorted, duplicate-free entry set into a :class:`SparseW`, with the tensor's
    reservation (its own size plus its contraction-pattern caches) taken first."""
    nsec = [sp.nsectors for sp in spaces]
    bkey = _row_keys(rows, nsec)
    order = np.lexsort((flat, bkey))
    bkey = bkey[order]
    flat = flat[order]
    vals = vals[order]
    rows = rows[order]
    del order
    if bkey.size > 1 and np.any((bkey[1:] == bkey[:-1]) & (flat[1:] == flat[:-1])):
        raise AssertionError("duplicate transition entry at node {} — compiler invariant "
                             "broken".format(u))                # pragma: no cover
    new = np.ones(bkey.size, dtype=bool)
    new[1:] = bkey[1:] != bkey[:-1]
    starts = np.nonzero(new)[0]
    sectors = np.ascontiguousarray(rows[starts])
    for row in sectors:
        if _flux(spaces, signs, row) != zero:               # pragma: no cover - invariant
            raise AssertionError("node {} block {} violates the flux rule".format(u, row))
    counts = np.diff(np.concatenate([starts, [bkey.size]]))
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    nnz = int(flat.size)
    dense_elems = 0
    for row in sectors:
        size = 1
        for sp, i in zip(spaces, row):
            size *= int(sp.dims[int(i)])
        dense_elems += size
    d = spaces[-1].total_dim
    # ⚠ The reservation covers the tensor **and its contraction-pattern caches**: a node is
    # contracted once per direction (one op leg left open) plus once with every op leg
    # contracted, and each pattern caches a CSR of the same nonzeros; the root additionally
    # carries the memoized copy with its dim-1 completed channel closed. Bounding all of it
    # here, at compile time, is what keeps a lazily built cache from being
    # resident-but-unaccounted (an estimate nobody checks is decoration). One "unit"
    # is the tensor's own exact size; a CSR of the same nonzeros is slightly smaller, so
    # this is an upper bound on residency and never an under-estimate.
    is_root = spaces[0].total_dim == 1
    n_units = 1 + (len(spaces) - 2) + (1 if is_root else 0)
    alloc = res.reserve(
        "TTNO node {} ({} blocks, {} nonzeros)".format(u, sectors.shape[0], nnz),
        n_units * sparse_w_gb(spaces, nnz, sectors.shape[0]),
        note="operator bond dims {} x phys dim {}; dense form would be {:.2f} GB".format(
            [sp.total_dim for sp in spaces[:-2]], d, 16.0 * dense_elems / 1024.0 ** 3),
        advice=["the transition count grows with the operator bond dimension, i.e. as the "
                "square of the active-space size; a different topology (or fewer modes per "
                "node) changes the bond dimensions it is built from"])
    allocations.append(alloc)
    return SparseW(spaces, signs, zero, sectors, indptr, np.ascontiguousarray(flat),
                   np.ascontiguousarray(vals))


# --- the reusable template: one compile per topology, refill per integral set ---------------

@dataclass(eq=False)
class _RefillNode:
    """Per-node refill/extraction tables of a :class:`TTNOTemplate` (internal).

    ``gidx`` is the skeleton entry of every (term, matrix element) pair, ``tid`` the term
    and ``val`` the local-matrix value (JW signs included). ``slot`` are the **distinct**
    skeleton entries those pairs read (``gidx = slot[slot_of]``), which is what the RDM
    extraction computes directly: their block rows and in-block multi-indices are
    ``slot_rows``/``slot_index`` in W leg order.
    """

    node: int
    skeleton: SparseW            #: the union entry set: compiled transitions + every slot
    base: np.ndarray             #: unit-transition content, coefficient slots at zero
    gidx: np.ndarray             #: skeleton entry position per (term, matrix element)
    tid: np.ndarray              #: term id per entry
    val: np.ndarray              #: local-matrix value per entry (JW signs included)
    slot: np.ndarray             #: distinct skeleton positions read
    slot_of: np.ndarray          #: gidx = slot[slot_of]
    slot_rows: np.ndarray        #: (n_slots, ndim) sector row of each slot
    slot_index: np.ndarray       #: (n_slots, ndim) in-block multi-index of each slot


class TTNOTemplate:
    """The Hamiltonian TTNO compiled **once per topology**, refilled per integral set.

    The compiler's label structure is coefficient-independent (a measurement this
    class exists to exploit): the template enumerates **every** one- and two-electron index
    tuple — including those whose integral happens to be zero — compiles the label
    structure once, and records where each term's coefficient lands
    (:func:`compile_ttno`'s ``slots``). Two consumers, sharing one table:

    * :meth:`fill` produces the TTNO at a given ``(h, eri)`` by scattering coefficients
      into the recorded slots — no recompilation inside a CASSCF macro-iteration.
    * :meth:`rdms_from_environments` reads the **same slots** out of the per-node operator
      environments ``G_u = dE/dW_u`` (:mod:`kuiva.dmrg.density`): the expectation value of
      one term's operator content is ``sum_ij local[i,j] G_u[channels, i, j]`` at its
      attachment, because every transition above and below the attachment is
      label-determined and unit. Refill and extraction are transposes of each other, so a
      defect in the table breaks the energy closure test rather than hiding. The
      production extraction never forms ``G_u``: it evaluates the distinct slots
      (:attr:`_RefillNode.slot`) by contracting the state around the node at exactly
      those channels (:func:`kuiva.dmrg.density.slot_values`), and
      :meth:`rdms_from_slot_values` reads them.

    ⚠ The full index enumeration is what makes extraction complete: a template built from
    one integral set's *nonzero* terms could not report a 2-RDM element whose integral is
    a symmetry zero, and a Γ silently missing such elements looks entirely plausible.

    Orchestration: built once per topology; ``fill`` is two vectorized
    scatters per refilled node. Its tables are reserved on the ledger at their exact size
    (they are resident for the whole optimization the template serves).
    """

    def __init__(self, graph: NetworkGraph, root: int = 0, bases=None):
        """``bases``: per-mode :class:`ModeBasis`, widening the quantum number with irrep
        labels (:func:`kuiva.symm.mode_bases`). Default: particle number only."""
        all_modes = sorted(m for c in graph.contents for m in c)
        if all_modes != list(range(len(all_modes))):
            raise ValueError("graph mode labels must be exactly 0..n-1, got {}"
                             .format(all_modes))
        n = len(all_modes)
        self.graph = graph
        self.root = int(root)
        self.n_modes = n

        # ⚠ **Every** index tuple is enumerated, including those whose integral is a
        # symmetry zero, because the same table is read backwards to extract Gamma and a
        # template built from one integral set's nonzero terms could not report an element
        # whose integral happened to vanish. The one exception is a tuple the *group* forbids
        # (below): its operator string changes the sector, so no state in a definite sector
        # has any expectation of it at all, and Gamma's zero there is exact rather than
        # circumstantial.
        one_str = _index_strings(n, 2)
        one_ok = _allowed_mask(bases, all_modes, one_str)
        p, q, r, s = np.meshgrid(*[np.arange(n)] * 4, indexing="ij")
        two_idx = np.stack([p.ravel(), q.ravel(), r.ravel(), s.ravel()], axis=1)
        two_ok = _allowed_mask(bases, all_modes, two_idx)
        two_str = two_idx[:, [0, 2, 3, 1]]                 # (p, r, s, q) as written
        one_tab, one_alive = _fermion_table(one_str[one_ok], (True, False),
                                            np.ones(int(one_ok.sum())), n, tol=None,
                                            reserve=False)
        two_tab, two_alive = _fermion_table(two_str[two_ok], (True, True, False, False),
                                            np.ones(int(two_ok.sum())), n, tol=None,
                                            reserve=False)
        # the tables dropped the vanishing strings; keep the index tuples that survived
        idxs = np.vstack([np.column_stack([one_str[one_ok][one_alive],
                                           np.zeros((int(one_alive.sum()), 2),
                                                    dtype=np.int64)]),
                          two_idx[two_ok][two_alive]])
        kinds = np.concatenate([np.zeros(int(one_alive.sum()), dtype=np.int64),
                                np.ones(int(two_alive.sum()), dtype=np.int64)])
        coeff = np.concatenate([one_tab.coeff, two_tab.coeff])
        ptr = np.concatenate([one_tab.ptr, one_tab.ptr[-1] + two_tab.ptr[1:]])
        modes = np.concatenate([one_tab.modes, two_tab.modes])
        matid = np.concatenate([one_tab.matid, two_tab.matid])
        table = TermTable(coeff, ptr, modes, matid, one_tab.mats)
        self._kinds = kinds
        self._idxs = np.ascontiguousarray(idxs, dtype=np.int64)
        self.n_terms = len(table)
        self._terms = table

        slots: list = []
        ttno = compile_ttno(graph, table, bases=bases, root=self.root, slots=slots)
        self.ttno = ttno
        self._bases = ttno.bases

        # -- per-node entry tables ---------------------------------------------------------
        self._nodes: List[_RefillNode] = []
        for rec in sorted(slots, key=lambda r: r.node):
            u = rec.node
            w = ttno.tensors[u]
            self._nodes.append(_build_refill_node(u, w, rec, ttno))

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
            prod = coeffs[rec.tid] * rec.val
            buf = rec.base + np.bincount(rec.gidx, weights=prod.real,
                                         minlength=rec.base.size) \
                + 1j * np.bincount(rec.gidx, weights=prod.imag, minlength=rec.base.size)
            tensors[rec.node] = rec.skeleton.with_values(buf)
        t = self.ttno
        return TTNO(graph=t.graph, root=t.root, parent=t.parent, children=t.children,
                    tensors=tensors, bond_space=t.bond_space, bond_labels=t.bond_labels,
                    phys_space=t.phys_space, phys_perm=t.phys_perm,
                    node_modes=t.node_modes, mode_dims=t.mode_dims, charge=t.charge,
                    mode_charges=t.mode_charges,
                    terms=self._terms.with_coefficients(coeffs), bases=t.bases)

    @property
    def slot_nodes(self) -> List[_RefillNode]:
        """The nodes that carry coefficient attachments, with their slot tables."""
        return list(self._nodes)

    def expectations(self, environments: Sequence[Optional[BlockTensor]]) -> np.ndarray:
        """Per-term operator expectations from per-node environments ``G_u = dE/dW_u``.

        ``environments[u]`` must carry W-tensor leg order ``[parent-op, child-ops
        ascending, phys-bra, phys-ket]`` over the same spaces
        (:func:`kuiva.dmrg.density.node_environments` produces exactly this). A block the
        state does not populate is simply absent and contributes zero. The validation
        path: production goes through :meth:`expectations_from_slot_values`.
        """
        values = []
        for rec in self._nodes:
            g_t = environments[rec.node]
            if g_t is None:
                raise ValueError("no environment supplied for node {}, which carries "
                                 "coefficient attachments".format(rec.node))
            vals = np.zeros(rec.slot.size, dtype=np.complex128)
            keys = _row_keys(rec.slot_rows, [sp.nsectors for sp in rec.skeleton.spaces])
            for key in np.unique(keys):
                sel = np.nonzero(keys == key)[0]
                blk = g_t.find(rec.slot_rows[sel[0]])
                if blk is None:
                    continue
                vals[sel] = blk[tuple(rec.slot_index[sel].T)]
            values.append(vals)
        return self.expectations_from_slot_values(values)

    def expectations_from_slot_values(self, values: Sequence[np.ndarray]) -> np.ndarray:
        """Per-term expectations from the value of ``G_u`` at each node's distinct slots
        (``values[k]`` aligned with ``slot_nodes[k].slot``)."""
        exp = np.zeros(self.n_terms, dtype=np.complex128)
        for rec, vals in zip(self._nodes, values):
            prod = rec.val * np.asarray(vals, dtype=np.complex128)[rec.slot_of]
            exp += np.bincount(rec.tid, weights=prod.real, minlength=self.n_terms) \
                + 1j * np.bincount(rec.tid, weights=prod.imag, minlength=self.n_terms)
        return exp

    def _rdms(self, exp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_modes
        res.require("network 2-RDM ({} spinors)".format(n), res.rdm_gb(n, 2),
                    note="the dense n^4 Gamma array",
                    advice=["the 2-RDM itself is n^4; a smaller active space is the only "
                            "knob"])
        i = self._idxs
        one = self._kinds == 0
        gamma = np.zeros((n, n), dtype=np.complex128)
        gamma[i[one, 0], i[one, 1]] = exp[one]
        gamma = 0.5 * (gamma + gamma.conj().T)
        gamma2 = np.zeros((n, n, n, n), dtype=np.complex128)
        two = ~one
        gamma2[i[two, 0], i[two, 1], i[two, 2], i[two, 3]] = exp[two]
        return gamma, gamma2

    def rdms_from_environments(self, environments: Sequence[Optional[BlockTensor]]
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """State-averaged ``(gamma, Gamma)`` in the :mod:`kuiva.rdm.rdm` convention.

        ``gamma_pq = <a+_p a_q>`` and ``Gamma_pqrs = <a+_p a+_r a_s a_q>`` — the same
        objects :class:`kuiva.rdm.rdm.RDMBuilder` returns, so the two paths are directly
        comparable and either plugs into the shared orbital optimizer contract. Index tuples whose
        operator string vanishes identically (``p = r`` or ``q = s``) are exact zeros.
        """
        return self._rdms(self.expectations(environments))

    def rdms_from_slot_values(self, values: Sequence[np.ndarray]
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """:meth:`rdms_from_environments` from slot values — the production route."""
        return self._rdms(self.expectations_from_slot_values(values))


def _allowed_mask(bases, all_modes: Sequence[int], idx: np.ndarray) -> np.ndarray:
    """Which index tuples (``(p, q)`` or ``(p, q, r, s)``) conserve the group label —
    :func:`_sector_filter`'s rule over a whole batch. Without widened labels every
    string conserves the particle number by construction and this is all ``True``."""
    n_str = idx.shape[0]
    if bases is None:
        return np.ones(n_str, dtype=bool)
    b = _resolve_bases(bases, all_modes)
    zero = b[all_modes[0]].charges[0].zero_like() if all_modes else None
    if zero is None or zero.width <= 1:
        return np.ones(n_str, dtype=bool)
    width = zero.width
    chg = np.zeros((int(max(all_modes)) + 1, width), dtype=np.int64)
    for m in all_modes:
        chg[m] = np.asarray(tuple(b[m].charges[1]), dtype=np.int64)
    shift = chg[idx[:, 0]] - chg[idx[:, 1]]
    if idx.shape[1] == 4:
        shift = shift + chg[idx[:, 2]] - chg[idx[:, 3]]
    moduli = zero.moduli or (None,) * width
    for c, m in enumerate(moduli):
        if m is not None:
            shift[:, c] %= int(m)
    return np.all(shift == 0, axis=1)


def template_slot_transient_gb(n_entries: int, nnz_w: int, ndim: int) -> float:
    """Transient [GB] :func:`_build_refill_node` holds at its peak — the slot entries'
    row and index tables, flat index, term id, key, pattern entries and value, plus the
    skeleton's assembly (the compiled entries and the slots concatenated, their unique
    row table, the sort order and the sorted copies). A model of the arrays that are
    live together, pinned against the measured peak in the tests."""
    per_slot = 16 * ndim + 8 + 8 + 8 + 16 + 8 + 8 + 8
    n_all = n_entries + nnz_w
    per_all = 8 * ndim + 8 + 16 + 8 * ndim + 8 + 8 + 8 + 8 + 16
    return (n_entries * per_slot + n_all * per_all) / 1024.0 ** 3


def _build_refill_node(u: int, w: SparseW, rec: NodeSlots, ttno: TTNO) -> _RefillNode:
    """The slot tables of one node from the compiler's :class:`NodeSlots` record."""
    phys_sp = ttno.phys_space[u]
    nc = len(ttno.children[u])
    ndim = 3 + nc
    child_spaces = [ttno.bond_space[c] for c in ttno.children[u]]
    sec_of = np.repeat(np.arange(phys_sp.nsectors, dtype=np.int64), phys_sp.dims)
    loc_of = np.arange(phys_sp.total_dim, dtype=np.int64) - phys_sp.offsets[sec_of]
    nnz_pat = np.diff(rec.pat_ptr)
    reps = nnz_pat[rec.pattern]
    n_entries = int(reps.sum())
    res.require("TTNO template slot tables (node {})".format(u),
                template_slot_transient_gb(n_entries, w.nnz, ndim),
                note="{} (term, matrix element) pairs".format(n_entries),
                advice=["the slot count is the term count times the local dimension: a "
                        "finer node partition shrinks it"])
    starts = rec.pat_ptr[rec.pattern]
    idx = np.repeat(starts - (np.cumsum(reps) - reps), reps) \
        + np.arange(n_entries, dtype=np.int64)
    r = rec.pat_rows[idx]
    c = rec.pat_cols[idx]
    val = rec.pat_vals[idx]
    tid = np.repeat(rec.tid, reps)
    key = np.repeat(rec.key, reps)
    rows = np.empty((n_entries, ndim), dtype=np.int64)
    index = np.empty((n_entries, ndim), dtype=np.int64)
    rows[:, 0] = rec.out_sector[key]
    index[:, 0] = rec.out_offset[key]
    for j in range(nc):
        rows[:, 1 + j] = rec.in_sector[key, j]
        index[:, 1 + j] = rec.in_offset[key, j]
    rows[:, ndim - 2] = sec_of[r]
    rows[:, ndim - 1] = sec_of[c]
    index[:, ndim - 2] = loc_of[r]
    index[:, ndim - 1] = loc_of[c]
    shape_cols = [ttno.bond_space[u].dims[rows[:, 0]]] \
        + [child_spaces[j].dims[rows[:, 1 + j]] for j in range(nc)] \
        + [phys_sp.dims[rows[:, ndim - 2]], phys_sp.dims[rows[:, ndim - 1]]]
    flat = np.zeros(n_entries, dtype=np.int64)
    for j in range(ndim):
        flat = flat * shape_cols[j] + index[:, j]
    # the skeleton is the union of the compiled transitions and every attachment slot: a
    # term whose integral is a symmetry zero still needs its slot, or the extraction
    # could not report that element of Gamma (class docstring)
    block_of = np.repeat(np.arange(w.nblocks, dtype=np.int64), np.diff(w.indptr))
    skeleton = SparseW.from_entries(
        w.spaces, w.signs, w.charge,
        np.concatenate([w.sectors[block_of], rows]),
        np.concatenate([w.flat, flat]),
        np.concatenate([w.values, np.zeros(n_entries, dtype=np.complex128)]))
    gidx = skeleton.entry_positions(rows, flat)
    slot, slot_of = np.unique(gidx, return_inverse=True)
    slot_of = np.asarray(slot_of).reshape(-1)
    first = np.zeros(slot.size, dtype=np.int64)
    first[slot_of] = np.arange(gidx.size, dtype=np.int64)     # some representative
    slot_rows = np.ascontiguousarray(rows[first])
    slot_index = np.ascontiguousarray(index[first])
    del rows, index, flat
    # remove the compile's coefficient-1 content: base then carries the unit
    # transitions only, and fill() adds true coefficients into clean slots
    base = skeleton.values.copy()
    base -= np.bincount(gidx, weights=val.real, minlength=base.size) \
        + 1j * np.bincount(gidx, weights=val.imag, minlength=base.size)
    node = _RefillNode(node=u, skeleton=skeleton, base=base, gidx=gidx, tid=tid, val=val,
                       slot=slot, slot_of=slot_of, slot_rows=slot_rows,
                       slot_index=slot_index)
    res.reserve_owned(node, "TTNO template node {} tables".format(u),
                      (skeleton.nbytes + base.nbytes + gidx.nbytes + tid.nbytes
                       + val.nbytes + slot.nbytes + slot_of.nbytes + slot_rows.nbytes
                       + slot_index.nbytes) / 1024.0 ** 3,
                      note="skeleton, unit-transition copy and the slot tables",
                      advice=["one template per topology; it replaces a recompile per "
                              "CASSCF macro-iteration"])
    return node


__all__ = ["ModeBasis", "FERMION_MODE", "ProductTerm", "TermTable", "TTNO", "TTNOTemplate",
           "NodeSlots", "NodeTransitions", "fermion_term", "consolidate",
           "hamiltonian_product_terms", "one_electron_product_terms", "compile_ttno",
           "compile_labels", "node_transitions", "pattern_tables",
           "ttno_from_cas_integrals", "pattern_table_gb", "coefficient_chunk_gb",
           "assembly_gb", "template_slot_transient_gb", "label_pass_gb", "term_build_gb",
           "context_gb", "node_transitions_gb", "dedupe_gb"]
