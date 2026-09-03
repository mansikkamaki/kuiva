"""Quantum numbers and block-sparse tensors for the tensor-network layer.

This is the core the whole in-house DMRG/TTNS engine is built on: the extensible
:class:`QuantumNumber` label, the sector-structured :class:`Space` a tensor leg lives in, and
the :class:`BlockTensor` whose nonzero blocks are exactly the symmetry-allowed ones. The
factorizations every sweep needs — :func:`fuse`/:func:`split`, :func:`qr`, :func:`svd` — live
here too, because they are operations *on the block structure*, not on the network.

Conventions (fixed here, relied on everywhere downstream)
---------------------------------------------------------
* **Quantum numbers are abelian, tuple-valued, particle number ``N`` first**. Adding a
  later label (abelian double-group irrep, an omega-like axial label) widens the tuple and
  rewrites nothing. Non-abelian adaptation is out of scope.
* **Flux convention.** Every leg carries a sign, +1 (outgoing) or -1 (incoming). A block at
  sectors ``(q_1 .. q_d)`` is symmetry-allowed iff ``sum_l sign_l * q_l == charge``, where
  ``charge`` is the tensor's total quantum number. A wavefunction network puts the total
  ``N`` on one tensor (the canonical center) and ``charge = 0`` elsewhere.
* **Dense embedding.** ``to_dense``/``from_dense`` order each leg's sectors ascending by
  quantum number (:class:`Space` sorts on construction); the dense array is the block
  tensor with symmetry-forbidden entries exactly zero. Round trips are bitwise: the
  embedding moves data, it never does arithmetic.
* **Block tables are sorted-key rectangular arrays** (plain sorted-key rectangular arrays a compiled backend can consume — B2/B3 by design): the sector
  table is an int64 ``(nblocks, ndim)`` array sorted lexicographically, addressed by
  mixed-radix keys and ``searchsorted`` — never by a dict keyed on quantum numbers. Block
  payloads are ``complex128``, C-contiguous (complex is first-class; a real path is
  never assumed).
* **Everything in this module is orchestration and stays Python**: it is the
  data model and the *correct* reference implementation of the factorizations. The hot
  kernels — the block-sparse two-site contraction driver, the environment update — arrive
  with the sweep and register in ``ci/kernels.py``; nothing here is a port candidate.

⚠ Truncation keeps degenerate groups whole — the load-bearing rule
------------------------------------------------------------------
Every truncation anywhere in the network inherits the discipline the Cholesky factorization and the state average established: **a cut through a degenerate
group breaks an exact degeneracy by the truncation threshold**, and for this code's targets
the degenerate groups are Kramers pairs and free-ion multiplets — precisely the physics the
calculation exists to resolve. :func:`svd` therefore truncates on the *merged* singular-value
spectrum across all symmetry sectors (a degenerate group can straddle sectors: a Kramers
partner carries the same ``N``, an orbital multiplet need not sit in one block), detects
degenerate groups at :data:`SCHMIDT_DEGENERACY_RTOL`, and applies **two floors, both on the
whole group**:

* *accuracy* — a group whose largest member is at or below ``tol`` is dropped whole, with
  everything after it (the spectrum is descending);
* *stability* — a group whose smallest member is at or below
  :data:`SCHMIDT_STABILITY_RTOL` times the largest singular value is numerical noise and is
  dropped whole. Without this, a loose degeneracy tolerance that merges a real value with a
  rounding-level one would keep noise directions, and later gauge moves that divide by
  singular values amplify them without bound.

``max_bond`` is enforced at **group granularity**: a group that does not fit entirely is not
started, so the kept dimension stops short of the cap rather than splitting a group. A cap
that cannot even hold the first group is **refused, not rounded** (the state-averaging discipline): the
error names the knob.

Phase conventions
-----------------
:func:`qr` fixes the phases so each diagonal element of ``R`` is real and non-negative —
LAPACK's sign convention is not guaranteed across vendors, and an unfixed phase makes
checkpoint restarts needlessly irreproducible (the working-basis rule, applied
here for the same reason). :func:`svd` fixes nothing beyond what LAPACK returns: inside a
degenerate group the singular vectors are defined only up to a rotation, no convention can
fix that, and nothing downstream may depend on it.

References
----------
* Symmetry-blocked DMRG tensors and the truncated-SVD sweep: S. R. White, Phys. Rev. Lett.
  69, 2863 (1992), doi:10.1103/PhysRevLett.69.2863; U. Schollwoeck, "The density-matrix
  renormalization group in the age of matrix product states", Ann. Phys. 326, 96 (2011),
  doi:10.1016/j.aop.2010.09.012 (canonical forms, QR/SVD gauges, truncation).
* Abelian-symmetry block bookkeeping in tensor networks: S. Singh, R. N. C. Pfeifer,
  G. Vidal, Phys. Rev. B 83, 115125 (2011), doi:10.1103/PhysRevB.83.115125.
* General-spinor (complex, no spin symmetry) DMRG, the regime this layer serves:
  S. Knecht, O. Legeza, M. Reiher, J. Chem. Phys. 140, 041101 (2014),
  doi:10.1063/1.4862495; H. Zhai et al., J. Chem. Phys. 159, 234801 (2023),
  doi:10.1063/5.0180424.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg.blas import zgemm as _blas_zgemm

from ..ci import kernels
from ..util import threads
from ..util.logging import get_logger

log = get_logger(__name__)

#: Relative width (against the largest singular value of the whole spectrum) within which
#: adjacent singular values count as **degenerate** for group-complete truncation. The
#: degeneracies this protects are exact in theory and numerical in practice: Kramers pairs in
#: the general (Kramers-unrestricted) network agree to roughly the 1e-8..1e-6 relative band
#: the general-complex CI path reserves for them, so the tolerance sits at the loose end of that band. Same role as
#: ``ORBIT_DEGENERACY_RTOL`` in :mod:`kuiva.integrals.transform`, and the same genuine
#: trade-off: too tight splits a physical group, too loose merges distinct values and
#: over-truncates. Exposed as a parameter everywhere it is used.
SCHMIDT_DEGENERACY_RTOL = 1.0e-6

#: Relative floor (against the largest singular value) below which a singular value is
#: numerical noise rather than a Schmidt direction. Its job is **stability, not accuracy**
#: (``tol`` does accuracy): a group is kept only if its *smallest* member clears this floor,
#: so a degeneracy tolerance loose enough to merge a real value with a rounding-level one
#: drops the group whole instead of keeping a noise direction. Mirrors
#: ``ORBIT_STABILITY_RTOL`` in :mod:`kuiva.integrals.transform`.
SCHMIDT_STABILITY_RTOL = 1.0e-13


class QuantumNumber(tuple):
    """An abelian quantum-number label: a tuple of integers, particle number ``N`` first.

    Extensible by design: the tuple starts as ``(N,)`` and abelian double-group irrep (and
    omega-like) labels widen it with no rewrite of anything that consumes it. Arithmetic is
    the componentwise group operation.

    ⚠ Subclasses :class:`tuple` for hashability, total ordering (lexicographic — what sorts
    :class:`Space` sectors and block-table rows) and immutability, but ``+`` and ``-`` are
    **componentwise**, not concatenation. Mixing a :class:`QuantumNumber` with a plain tuple
    in arithmetic raises, so the redefinition cannot be triggered by accident.

    Cyclic components
    -----------------
    ``moduli`` gives a per-component modulus, ``None`` where the component is unbounded
    (``N`` always is). A component with a modulus is **reduced on construction**, so the
    stored tuple is the canonical representative and hashing, ordering and equality — the
    three things the block machinery is built on — need no change at all.

    ⚠ **The modulus has to live here rather than in the layer that builds the labels**, and
    that is a correction to the obvious design. A finite cyclic group is not a subgroup of the
    integers: a double-group label has terms that conserve it and do *not* conserve the plain
    integer sum (``l_p - l_q + l_r - l_s = 4`` in a ``Z4`` component, which is the identity),
    so integer-only arithmetic would silently drop legal Hamiltonian terms. Every one of them
    is Hermitian, correctly labelled and missing, which is exactly the class of error that
    produces a converged, plausible, wrong energy.

    Moduli **propagate** through arithmetic: an operand that carries them wins, and two
    operands carrying different ones is an error rather than a merge. That is what lets an
    unlabelled ``QuantumNumber.zero(width)`` — of which there are many, and all of them are
    genuinely the identity — go on working unchanged.
    """

    def __new__(cls, *labels: int, moduli=None):
        if len(labels) == 1 and isinstance(labels[0], (tuple, list)):
            labels = tuple(labels[0])
        if not labels:
            raise ValueError("a QuantumNumber needs at least the particle-number label N")
        if not all(isinstance(x, (int, np.integer)) for x in labels):
            raise TypeError("QuantumNumber labels must be integers, got {!r}".format(labels))
        values = tuple(int(x) for x in labels)
        if moduli is None and isinstance(labels[0], QuantumNumber):
            moduli = labels[0].moduli
        if moduli is not None:
            moduli = tuple(None if m is None else int(m) for m in moduli)
            if len(moduli) != len(values):
                raise ValueError("{} moduli for a width-{} quantum number"
                                 .format(len(moduli), len(values)))
            if any(m is not None and m <= 0 for m in moduli):
                raise ValueError("a cyclic component needs a positive modulus, got {}"
                                 .format(moduli))
            values = tuple(v if m is None else v % m for v, m in zip(values, moduli))
        obj = tuple.__new__(cls, values)
        obj._moduli = moduli
        return obj

    @property
    def moduli(self):
        """Per-component moduli, or ``None`` when every component is unbounded."""
        return getattr(self, "_moduli", None)

    @classmethod
    def zero(cls, width: int = 1, moduli=None) -> "QuantumNumber":
        """The identity element with ``width`` labels."""
        return cls(*([0] * int(width)), moduli=moduli)

    def zero_like(self) -> "QuantumNumber":
        """The identity element of *this* label's group — moduli included."""
        return QuantumNumber(*([0] * len(self)), moduli=self.moduli)

    def like(self, values) -> "QuantumNumber":
        """A label of this group with the given components."""
        return QuantumNumber(*values, moduli=self.moduli)

    @property
    def n(self) -> int:
        """The particle number — the mandatory first label."""
        return self[0]

    @property
    def width(self) -> int:
        return len(self)

    def _check(self, other) -> "QuantumNumber":
        if not isinstance(other, QuantumNumber):
            raise TypeError("QuantumNumber arithmetic needs a QuantumNumber, got {!r}"
                            .format(other))
        if len(other) != len(self):
            raise ValueError("QuantumNumber widths differ: {} vs {}".format(len(self),
                                                                            len(other)))
        return other

    def _moduli_with(self, other):
        mine, theirs = self.moduli, other.moduli
        if mine is None or theirs is None or mine == theirs:
            return mine if mine is not None else theirs
        raise ValueError("QuantumNumber moduli differ: {} vs {}; two different groups are "
                         "being added".format(mine, theirs))

    def __add__(self, other):
        other = self._check(other)
        return QuantumNumber(*(a + b for a, b in zip(self, other)),
                             moduli=self._moduli_with(other))

    def __sub__(self, other):
        other = self._check(other)
        return QuantumNumber(*(a - b for a, b in zip(self, other)),
                             moduli=self._moduli_with(other))

    def __neg__(self):
        return QuantumNumber(*(-a for a in self), moduli=self.moduli)

    def __radd__(self, other):
        # sum() starts from int 0; map it to the identity so sum(qns) works.
        if isinstance(other, int) and other == 0:
            return self
        return self._check(other).__add__(self)

    def __repr__(self) -> str:
        return "QN" + tuple.__repr__(self)


class Space(object):
    """An ordered direct sum of symmetry sectors: the structure of one tensor leg.

    Sectors are ``(QuantumNumber, dimension)`` pairs, stored sorted ascending by quantum
    number — that ordering **is** the dense-embedding convention, so it is fixed here and
    nowhere else. Duplicate quantum numbers are refused rather than merged: two sectors with
    the same label is a bookkeeping error upstream, and merging would hide it.
    """

    __slots__ = ("qns", "dims", "offsets")

    def __init__(self, sectors: Sequence[Tuple[QuantumNumber, int]]):
        pairs = []
        for qn, dim in sectors:
            if not isinstance(qn, QuantumNumber):
                qn = QuantumNumber(*qn) if isinstance(qn, (tuple, list)) else QuantumNumber(qn)
            dim = int(dim)
            if dim <= 0:
                raise ValueError("sector {} has non-positive dimension {}".format(qn, dim))
            pairs.append((qn, dim))
        if not pairs:
            raise ValueError("a Space needs at least one sector")
        widths = {q.width for q, _ in pairs}
        if len(widths) != 1:
            raise ValueError("sectors carry QuantumNumbers of different widths: {}"
                             .format(sorted(widths)))
        pairs.sort(key=lambda p: p[0])
        qns = tuple(q for q, _ in pairs)
        if len(set(qns)) != len(qns):
            raise ValueError("duplicate sector quantum numbers: {}".format(qns))
        self.qns: Tuple[QuantumNumber, ...] = qns
        self.dims = np.array([d for _, d in pairs], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.dims)]).astype(np.int64)

    @property
    def nsectors(self) -> int:
        return len(self.qns)

    @property
    def total_dim(self) -> int:
        return int(self.offsets[-1])

    @property
    def width(self) -> int:
        return self.qns[0].width

    def sector_index(self, qn: QuantumNumber) -> int:
        try:
            return self.qns.index(qn)
        except ValueError:
            raise KeyError("no sector {} in {}".format(qn, self))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Space) and self.qns == other.qns
                and np.array_equal(self.dims, other.dims))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.qns, tuple(int(d) for d in self.dims)))

    def __repr__(self) -> str:
        inner = ", ".join("{}:{}".format(q, d) for q, d in zip(self.qns, self.dims))
        return "Space({})".format(inner)


# --- internal helpers ---------------------------------------------------------------------

def _flux(spaces: Sequence[Space], signs: Sequence[int], row: Sequence[int]) -> QuantumNumber:
    """The signed quantum-number sum of one sector combination."""
    total = QuantumNumber.zero(spaces[0].width)
    for space, sign, i in zip(spaces, signs, row):
        qn = space.qns[int(i)]
        total = total + qn if sign > 0 else total - qn
    return total


def _allowed_rows(spaces: Sequence[Space], signs: Sequence[int],
                  charge: QuantumNumber) -> np.ndarray:
    """All symmetry-allowed sector combinations, lexicographically sorted (int64 rows).

    ``itertools.product`` over ascending ranges yields lexicographic order by construction,
    so no sort is needed.
    """
    rows = [row for row in itertools.product(*[range(s.nsectors) for s in spaces])
            if _flux(spaces, signs, row) == charge]
    return np.array(rows, dtype=np.int64).reshape(len(rows), len(spaces))


def _row_keys(sectors: np.ndarray, nsec: Sequence[int]) -> np.ndarray:
    """Mixed-radix (C-order lexicographic) scalar key per row of the sector table."""
    total = 1
    for n in nsec:
        total *= int(n)                        # Python ints: no silent overflow
    if total > 2 ** 62:
        raise OverflowError("sector-index key overflows int64")   # pragma: no cover
    strides = np.ones(len(nsec), dtype=np.int64)
    for l in range(len(nsec) - 2, -1, -1):
        strides[l] = strides[l + 1] * int(nsec[l + 1])
    return sectors @ strides


class BlockTensor(object):
    """A block-sparse tensor whose nonzero blocks are the symmetry-allowed ones.

    ``spaces``/``signs`` describe the legs, ``charge`` the total quantum number (module
    docstring: flux convention). ``sectors`` is the sorted int64 block table, one row of
    per-leg sector indices per block; ``blocks`` is the aligned list of C-contiguous
    ``complex128`` payloads.

    The constructor validates everything — this class is orchestration (module docstring)
    and correctness beats the few microseconds the checks cost. Use the classmethods
    (:meth:`zeros`, :meth:`random`, :meth:`from_dense`) rather than assembling by hand.
    """

    __slots__ = ("spaces", "signs", "charge", "sectors", "blocks", "_keys")

    def __init__(self, spaces: Sequence[Space], signs: Sequence[int], charge: QuantumNumber,
                 sectors: np.ndarray, blocks: List[np.ndarray]):
        spaces = tuple(spaces)
        signs = tuple(int(s) for s in signs)
        if not spaces:
            raise ValueError("a BlockTensor needs at least one leg")
        if len(signs) != len(spaces):
            raise ValueError("{} legs but {} signs".format(len(spaces), len(signs)))
        if any(s not in (-1, 1) for s in signs):
            raise ValueError("leg signs must be +1 or -1, got {}".format(signs))
        if not isinstance(charge, QuantumNumber):
            raise TypeError("charge must be a QuantumNumber, got {!r}".format(charge))
        widths = {sp.width for sp in spaces} | {charge.width}
        if len(widths) != 1:
            raise ValueError("leg/charge QuantumNumber widths differ: {}".format(sorted(widths)))

        sectors = np.asarray(sectors, dtype=np.int64)
        if sectors.ndim != 2 or sectors.shape[1] != len(spaces):
            raise ValueError("sector table must be (nblocks, {}), got {}"
                             .format(len(spaces), sectors.shape))
        if len(blocks) != sectors.shape[0]:
            raise ValueError("{} blocks for {} table rows".format(len(blocks),
                                                                  sectors.shape[0]))
        nsec = [sp.nsectors for sp in spaces]
        keys = _row_keys(sectors, nsec)
        if np.any(np.diff(keys) <= 0):
            raise ValueError("sector table rows must be sorted and unique")
        for row, block in zip(sectors, blocks):
            if np.any(row < 0) or np.any(row >= np.array(nsec)):
                raise ValueError("sector index out of range in row {}".format(row))
            if _flux(spaces, signs, row) != charge:
                raise ValueError("block at {} violates the flux rule (charge {})"
                                 .format(row, charge))
            shape = tuple(int(sp.dims[i]) for sp, i in zip(spaces, row))
            if not isinstance(block, np.ndarray) or block.shape != shape:
                raise ValueError("block at {} has shape {}, expected {}"
                                 .format(row, getattr(block, "shape", None), shape))
            if block.dtype != np.complex128 or not block.flags.c_contiguous:
                raise ValueError("blocks must be C-contiguous complex128")

        self.spaces = spaces
        self.signs = signs
        self.charge = charge
        self.sectors = sectors
        self.blocks = blocks
        self._keys = keys

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def _trusted(cls, spaces: Tuple[Space, ...], signs: Tuple[int, ...],
                 charge: QuantumNumber, sectors: np.ndarray, keys: np.ndarray,
                 blocks: List[np.ndarray]) -> "BlockTensor":
        """Constructor without validation, for tensors correct **by construction**.

        ⚠ Internal use only (:func:`tensordot`, :meth:`transpose`, :meth:`conj`, and the
        sweep's pack/unpack): every invariant the public constructor checks — sorted-unique
        rows, flux conservation, shapes, dtype, contiguity — must already hold. Measured
        motive: the public constructor's per-row flux validation in Python quantum-number
        arithmetic was **half the cost of a DMRG sweep** (14.1 of 29.5 CPU s on a
        10-mode, D = 16 profile), all of it re-checking tensors that could not be wrong.
        The public constructor keeps validating; anything user-facing goes through it.
        """
        t = object.__new__(cls)
        t.spaces = spaces
        t.signs = signs
        t.charge = charge
        t.sectors = sectors
        t.blocks = blocks
        t._keys = keys
        return t

    @classmethod
    def zeros(cls, spaces: Sequence[Space], signs: Sequence[int],
              charge: Optional[QuantumNumber] = None) -> "BlockTensor":
        """All symmetry-allowed blocks, zero-filled. ``charge`` defaults to the identity."""
        spaces = tuple(spaces)
        if charge is None:
            charge = QuantumNumber.zero(spaces[0].width)
        rows = _allowed_rows(spaces, signs, charge)
        blocks = [np.zeros(tuple(int(sp.dims[i]) for sp, i in zip(spaces, row)),
                           dtype=np.complex128) for row in rows]
        return cls(spaces, signs, charge, rows, blocks)

    @classmethod
    def random(cls, spaces: Sequence[Space], signs: Sequence[int],
               charge: Optional[QuantumNumber] = None,
               rng: Optional[np.random.Generator] = None) -> "BlockTensor":
        """All allowed blocks filled with standard-normal complex entries (tests, guesses)."""
        rng = rng if rng is not None else np.random.default_rng()
        t = cls.zeros(spaces, signs, charge)
        for block in t.blocks:
            block[...] = rng.standard_normal(block.shape) \
                + 1j * rng.standard_normal(block.shape)
        return t

    @classmethod
    def from_dense(cls, dense: np.ndarray, spaces: Sequence[Space], signs: Sequence[int],
                   charge: Optional[QuantumNumber] = None,
                   forbidden_rtol: float = 1.0e-12) -> "BlockTensor":
        """Extract the allowed blocks of a dense array (module docstring: embedding order).

        ⚠ Weight on symmetry-forbidden entries above ``forbidden_rtol`` (relative to the
        total norm) **raises**: it means the dense object does not have the symmetry this
        tensor claims, and silently projecting it would launder a symmetry-broken state into
        a symmetric-looking one.
        """
        spaces = tuple(spaces)
        if charge is None:
            charge = QuantumNumber.zero(spaces[0].width)
        shape = tuple(sp.total_dim for sp in spaces)
        dense = np.asarray(dense)
        if dense.shape != shape:
            raise ValueError("dense shape {} does not match spaces {}".format(dense.shape,
                                                                              shape))
        rows = _allowed_rows(spaces, signs, charge)
        blocks = []
        kept = 0.0
        for row in rows:
            slices = tuple(slice(int(sp.offsets[i]), int(sp.offsets[i + 1]))
                           for sp, i in zip(spaces, row))
            block = np.ascontiguousarray(dense[slices], dtype=np.complex128)
            kept += float(np.vdot(block, block).real)
            blocks.append(block)
        total = float(np.vdot(dense, dense).real)
        forbidden = max(total - kept, 0.0)
        if total > 0.0 and forbidden > forbidden_rtol * total:
            raise ValueError("dense tensor carries relative weight {:.3e} on symmetry-"
                             "forbidden entries (allowed {:.1e}): it does not have the "
                             "declared symmetry".format(forbidden / total, forbidden_rtol))
        return cls(spaces, signs, charge, rows, blocks)

    # --- basic queries --------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.spaces)

    @property
    def nblocks(self) -> int:
        return int(self.sectors.shape[0])

    @property
    def nbytes(self) -> int:
        """Exact payload + table bytes (sizing is exact, never padded)."""
        return int(sum(b.nbytes for b in self.blocks) + self.sectors.nbytes)

    def find(self, row: Sequence[int]) -> Optional[np.ndarray]:
        """The block at sector-index ``row``, or ``None`` if absent."""
        key = _row_keys(np.asarray([row], dtype=np.int64),
                        [sp.nsectors for sp in self.spaces])[0]
        pos = int(np.searchsorted(self._keys, key))
        if pos < len(self._keys) and self._keys[pos] == key:
            return self.blocks[pos]
        return None

    def norm(self) -> float:
        return float(np.sqrt(sum(np.vdot(b, b).real for b in self.blocks)))

    def copy(self) -> "BlockTensor":
        return BlockTensor(self.spaces, self.signs, self.charge, self.sectors.copy(),
                           [b.copy() for b in self.blocks])

    # --- structural operations ------------------------------------------------------------

    def to_dense(self) -> np.ndarray:
        """The dense embedding (module docstring: sector order). Bitwise scatter."""
        dense = np.zeros(tuple(sp.total_dim for sp in self.spaces), dtype=np.complex128)
        for row, block in zip(self.sectors, self.blocks):
            slices = tuple(slice(int(sp.offsets[i]), int(sp.offsets[i + 1]))
                           for sp, i in zip(self.spaces, row))
            dense[slices] = block
        return dense

    def transpose(self, perm: Sequence[int]) -> "BlockTensor":
        """Permute legs; the block table is re-sorted to keep the sorted-key invariant."""
        perm = tuple(int(p) for p in perm)
        if sorted(perm) != list(range(self.ndim)):
            raise ValueError("not a permutation of {} legs: {}".format(self.ndim, perm))
        spaces = tuple(self.spaces[p] for p in perm)
        signs = tuple(self.signs[p] for p in perm)
        sectors = self.sectors[:, perm]
        keys = _row_keys(sectors, [sp.nsectors for sp in spaces])
        order = np.argsort(keys)
        return BlockTensor._trusted(
            spaces, signs, self.charge, np.ascontiguousarray(sectors[order]),
            keys[order],
            [np.ascontiguousarray(self.blocks[i].transpose(perm)) for i in order])

    def conj(self) -> "BlockTensor":
        """Complex conjugate. Legs flip direction and the charge negates, so that
        contracting a tensor with its conjugate over matching legs conserves flux."""
        return BlockTensor._trusted(self.spaces, tuple(-s for s in self.signs),
                                    -self.charge, self.sectors, self._keys,
                                    [np.conj(b) for b in self.blocks])

    def __repr__(self) -> str:
        return "BlockTensor(ndim={}, nblocks={}, charge={}, dims={})".format(
            self.ndim, self.nblocks, self.charge,
            tuple(sp.total_dim for sp in self.spaces))


# --- fuse / split -------------------------------------------------------------------------

@dataclass(frozen=True)
class FuseRecord:
    """Everything needed to split a fused leg back **exactly**.

    ``combos`` enumerates every sector combination of the fused legs (lexicographic rows),
    ``combo_sector``/``combo_offset``/``combo_dim`` place each combination inside its flux
    sector of ``fused_space``. Rectangular arrays throughout — this record is what a future
    compiled kernel would consume, so it is shaped like one (B3).
    """

    spaces: Tuple[Space, ...]          #: the original fused legs, in fused order
    signs: Tuple[int, ...]             #: their signs
    fused_space: Space
    combos: np.ndarray                 #: (ncombo, nfused) int64, lexicographic
    combo_sector: np.ndarray           #: (ncombo,) index into fused_space sectors
    combo_offset: np.ndarray           #: (ncombo,) offset inside the fused sector
    combo_dim: np.ndarray              #: (ncombo,) product of member dims


def fuse(tensor: BlockTensor, axes: Sequence[int]) -> Tuple[BlockTensor, FuseRecord]:
    """Fuse the legs ``axes`` (in the order given) into one leg, placed first.

    The fused leg carries sign +1 and sector quantum numbers equal to the *flux* of each
    member combination, so conservation is preserved verbatim. Within a fused sector,
    combinations are packed in lexicographic order of their sector-index tuples — a fixed,
    documented convention, because :func:`split` and every environment built against a fused
    bond depend on it.

    Returns ``(fused_tensor, record)``; the remaining legs keep their relative order after
    the fused one. Pure data movement — bitwise invertible by :func:`split`.
    """
    axes = tuple(int(a) for a in axes)
    if not axes or len(set(axes)) != len(axes) \
            or any(a < 0 or a >= tensor.ndim for a in axes):
        raise ValueError("axes must be a non-empty subset of 0..{}, got {}"
                         .format(tensor.ndim - 1, axes))
    rest = tuple(a for a in range(tensor.ndim) if a not in axes)
    fspaces = tuple(tensor.spaces[a] for a in axes)
    fsigns = tuple(tensor.signs[a] for a in axes)

    combos = np.array(list(itertools.product(*[range(sp.nsectors) for sp in fspaces])),
                      dtype=np.int64).reshape(-1, len(axes))
    fluxes = [_flux(fspaces, fsigns, row) for row in combos]
    combo_dim = np.array([int(np.prod([sp.dims[i] for sp, i in zip(fspaces, row)]))
                          for row in combos], dtype=np.int64)

    sector_qns = sorted(set(fluxes))
    sector_of = {qn: k for k, qn in enumerate(sector_qns)}
    combo_sector = np.array([sector_of[f] for f in fluxes], dtype=np.int64)
    combo_offset = np.zeros(len(combos), dtype=np.int64)
    running = np.zeros(len(sector_qns), dtype=np.int64)
    for c in range(len(combos)):
        combo_offset[c] = running[combo_sector[c]]
        running[combo_sector[c]] += combo_dim[c]
    fused_space = Space([(qn, int(running[k])) for k, qn in enumerate(sector_qns)])

    record = FuseRecord(fspaces, fsigns, fused_space, combos, combo_sector, combo_offset,
                        combo_dim)

    # index of each old block's fused combination (mixed radix over the fused legs)
    fused_nsec = [sp.nsectors for sp in fspaces]
    combo_keys = _row_keys(combos, fused_nsec)          # == arange, but stated not assumed
    old_combo_keys = _row_keys(tensor.sectors[:, list(axes)], fused_nsec)
    combo_index = np.searchsorted(combo_keys, old_combo_keys)

    new_spaces = (fused_space,) + tuple(tensor.spaces[a] for a in rest)
    new_signs = (1,) + tuple(tensor.signs[a] for a in rest)

    # accumulate: several old blocks (different combos, same flux) land in one fused block
    assembled = {}
    for b, (row, block) in enumerate(zip(tensor.sectors, tensor.blocks)):
        ci = int(combo_index[b])
        new_row = (int(combo_sector[ci]),) + tuple(int(row[a]) for a in rest)
        if new_row not in assembled:
            shape = (int(fused_space.dims[combo_sector[ci]]),) \
                + tuple(int(tensor.spaces[a].dims[row[a]]) for a in rest)
            assembled[new_row] = np.zeros(shape, dtype=np.complex128)
        moved = np.ascontiguousarray(block.transpose(axes + rest))
        flat = moved.reshape((int(combo_dim[ci]),) + moved.shape[len(axes):])
        off = int(combo_offset[ci])
        assembled[new_row][off:off + int(combo_dim[ci])] = flat

    rows = np.array(sorted(assembled), dtype=np.int64).reshape(len(assembled),
                                                               len(new_spaces))
    blocks = [assembled[tuple(int(i) for i in row)] for row in rows]
    return BlockTensor(new_spaces, new_signs, tensor.charge, rows, blocks), record


def split(tensor: BlockTensor, record: FuseRecord, axis: int = 0) -> BlockTensor:
    """Invert :func:`fuse`: expand leg ``axis`` back into the recorded legs, in place of it.

    The leg being split must match the record's ``fused_space`` (and carry sign +1) — a
    mismatch means the record belongs to a different fusion and the result would be
    plausible-looking garbage, so it is refused. Every recorded combination is emitted
    (zero blocks included), so the result's block table is deterministic.
    """
    axis = int(axis)
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError("axis {} out of range".format(axis))
    if tensor.spaces[axis] != record.fused_space or tensor.signs[axis] != 1:
        raise ValueError("leg {} does not match the fuse record (space or sign differs)"
                         .format(axis))
    before = tuple(range(axis))
    after = tuple(range(axis + 1, tensor.ndim))
    new_spaces = tuple(tensor.spaces[a] for a in before) + record.spaces \
        + tuple(tensor.spaces[a] for a in after)
    new_signs = tuple(tensor.signs[a] for a in before) + record.signs \
        + tuple(tensor.signs[a] for a in after)

    assembled = {}
    for row, block in zip(tensor.sectors, tensor.blocks):
        fs = int(row[axis])
        for ci in np.nonzero(record.combo_sector == fs)[0]:
            ci = int(ci)
            off, cd = int(record.combo_offset[ci]), int(record.combo_dim[ci])
            piece = np.take(block, np.arange(off, off + cd), axis=axis)
            member_dims = tuple(int(sp.dims[i]) for sp, i in zip(record.spaces,
                                                                 record.combos[ci]))
            shape = piece.shape[:axis] + member_dims + piece.shape[axis + 1:]
            new_row = tuple(int(row[a]) for a in before) \
                + tuple(int(i) for i in record.combos[ci]) \
                + tuple(int(row[a]) for a in after)
            assembled[new_row] = np.ascontiguousarray(piece.reshape(shape),
                                                      dtype=np.complex128)
    rows = np.array(sorted(assembled), dtype=np.int64).reshape(len(assembled),
                                                               len(new_spaces))
    blocks = [assembled[tuple(int(i) for i in row)] for row in rows]
    return BlockTensor(new_spaces, new_signs, tensor.charge, rows, blocks)


# --- matricization (shared by qr and svd) -------------------------------------------------

def _as_matrix(tensor: BlockTensor, left_axes: Sequence[int]):
    """Fuse ``tensor`` into a two-leg matrix ``[fused_left, fused_right]``.

    Returns ``(matrix, left_record, right_record, right_axes)``. Because sector quantum
    numbers are unique within a leg, charge conservation pairs each left sector with at most
    one right sector: every matrix block is indexed by its left (bond) sector alone.
    """
    left_axes = tuple(int(a) for a in left_axes)
    if not (0 < len(left_axes) < tensor.ndim) or len(set(left_axes)) != len(left_axes) \
            or any(a < 0 or a >= tensor.ndim for a in left_axes):
        raise ValueError("left_axes must be a proper non-empty subset of the {} legs, got {}"
                         .format(tensor.ndim, left_axes))
    right_axes = tuple(a for a in range(tensor.ndim) if a not in left_axes)
    t1, rec_r = fuse(tensor, right_axes)                  # [fused_right, left...]
    t2, rec_l = fuse(t1, tuple(range(1, t1.ndim)))        # [fused_left, fused_right]
    return t2, rec_l, rec_r, right_axes


def qr(tensor: BlockTensor, left_axes: Sequence[int]) -> Tuple[BlockTensor, BlockTensor]:
    """Blockwise thin QR across the bipartition ``left_axes`` | rest.

    Returns ``(Q, R)`` with ``Q`` legs ``[left..., bond]`` (bond sign -1, charge 0 — an
    isometry: ``Q+ Q = 1`` on the bond space) and ``R`` legs ``[bond, right...]`` carrying
    the tensor's charge. Phases are fixed so ``diag(R)`` is real and non-negative per block
    (module docstring). The bond space keeps one sector per matrix block, dimension
    ``min(rows, cols)`` — no truncation ever happens in QR.
    """
    m, rec_l, rec_r, _ = _as_matrix(tensor, left_axes)

    bond_sectors = []
    q_blocks, r_blocks = [], []
    for row, block in zip(m.sectors, m.blocks):
        qb, rb = np.linalg.qr(block)                      # thin
        d = np.diagonal(rb).copy()
        w = np.where(np.abs(d) > 0.0, np.conj(d) / np.where(np.abs(d) > 0.0, np.abs(d), 1.0),
                     1.0)
        qb = np.ascontiguousarray(qb * np.conj(w)[None, :])
        rb = np.ascontiguousarray(rb * w[:, None])
        bond_sectors.append((m.spaces[0].qns[int(row[0])], qb.shape[1], int(row[0]),
                             int(row[1])))
        q_blocks.append(qb)
        r_blocks.append(rb)

    bond_space = Space([(qn, k) for qn, k, _, _ in bond_sectors])
    q_rows, r_rows = [], []
    for b, (qn, _, i_left, i_right) in enumerate(bond_sectors):
        q_rows.append((i_left, bond_space.sector_index(qn)))
        r_rows.append((bond_space.sector_index(qn), i_right))
    order_q = np.argsort(_row_keys(np.asarray(q_rows, dtype=np.int64),
                                   [m.spaces[0].nsectors, bond_space.nsectors]))
    order_r = np.argsort(_row_keys(np.asarray(r_rows, dtype=np.int64),
                                   [bond_space.nsectors, m.spaces[1].nsectors]))

    q_m = BlockTensor((m.spaces[0], bond_space), (1, -1), QuantumNumber.zero(tensor.charge.width),
                      np.asarray(q_rows, dtype=np.int64)[order_q],
                      [q_blocks[i] for i in order_q])
    r_m = BlockTensor((bond_space, m.spaces[1]), (1, 1), tensor.charge,
                      np.asarray(r_rows, dtype=np.int64)[order_r],
                      [r_blocks[i] for i in order_r])
    return split(q_m, rec_l, axis=0), split(r_m, rec_r, axis=1)


# --- SVD with degenerate-group-complete truncation ----------------------------------------

@dataclass(frozen=True)
class TruncationInfo:
    """What a truncation did — reported, never silent (reported, part of the output contract)."""

    bond_dim: int                #: kept Schmidt vectors, all sectors together
    n_discarded: int             #: discarded Schmidt vectors
    discarded_weight: float      #: sum of discarded s^2 over sum of all s^2 (0 if exact)
    smallest_kept: float
    largest_discarded: float     #: 0.0 when nothing was discarded


def svd(tensor: BlockTensor, left_axes: Sequence[int], tol: float = 0.0,
        max_bond: Optional[int] = None,
        degeneracy_rtol: float = SCHMIDT_DEGENERACY_RTOL,
        stability_rtol: float = SCHMIDT_STABILITY_RTOL
        ) -> Tuple[BlockTensor, List[np.ndarray], BlockTensor, TruncationInfo]:
    """Blockwise SVD across ``left_axes`` | rest, truncated in whole degenerate groups.

    Returns ``(U, s_sectors, Vh, info)``: ``U`` legs ``[left..., bond]`` (isometry, charge
    0), ``Vh`` legs ``[bond, right...]`` (carries the charge), and ``s_sectors`` the
    singular values as one descending float64 array per bond sector, aligned with the bond
    :class:`Space` of ``U``'s last / ``Vh``'s first leg.

    Truncation (module docstring — the load-bearing rule): the merged spectrum across all
    sectors is grouped at ``degeneracy_rtol`` (relative to the largest value), and groups
    are kept in descending order while the group clears **both** floors whole (accuracy:
    largest member > ``tol``; stability: smallest member > ``stability_rtol *
    s_max``) and, if ``max_bond`` is set, fits entirely within it. The first failing group
    stops the truncation; a cap or tolerance that cannot keep even the first group is
    refused with the knob named.

    ⚠ A degenerate group may straddle bond sectors (Kramers partners share ``N``; orbital
    multiplets need not). Per-sector truncation cannot see that — this global comparison is
    the point of the routine.
    """
    m, rec_l, rec_r, _ = _as_matrix(tensor, left_axes)

    per_block = []
    for row, block in zip(m.sectors, m.blocks):
        u, s, vh = np.linalg.svd(block, full_matrices=False)
        per_block.append((row, u, s, vh))

    all_s = np.concatenate([s for _, _, s, _ in per_block]) if per_block \
        else np.zeros(0, dtype=np.float64)
    all_block = np.concatenate([np.full(s.size, b, dtype=np.int64)
                                for b, (_, _, s, _) in enumerate(per_block)]) \
        if per_block else np.zeros(0, dtype=np.int64)
    order = np.argsort(-all_s, kind="stable")
    s_sorted = all_s[order]

    if s_sorted.size == 0 or s_sorted[0] <= 0.0:
        raise ValueError("SVD truncation of a tensor with no nonzero singular values is "
                         "meaningless — the tensor is zero")

    floor = stability_rtol * float(s_sorted[0])
    keep = 0
    for group in _degenerate_groups(s_sorted, degeneracy_rtol):
        if float(s_sorted[group].max()) <= tol or float(s_sorted[group].min()) <= floor:
            break
        if max_bond is not None and int(group[-1]) + 1 > int(max_bond):
            break
        keep = int(group[-1]) + 1
    if keep == 0:
        first = _degenerate_groups(s_sorted, degeneracy_rtol)[0]
        raise ValueError(
            "truncation would cut inside the leading degenerate group ({} values within "
            "rtol {:.1e}): raise max_bond (>= {}) or loosen tol ({:.3e} vs largest value "
            "{:.3e}) — a cut through a degenerate group is refused, not rounded"
            .format(len(first), degeneracy_rtol, len(first), tol, float(s_sorted[0])))

    total_w = float(np.sum(s_sorted ** 2))
    disc_w = float(np.sum(s_sorted[keep:] ** 2))
    info = TruncationInfo(bond_dim=keep, n_discarded=int(s_sorted.size - keep),
                          discarded_weight=disc_w / total_w if total_w > 0.0 else 0.0,
                          smallest_kept=float(s_sorted[keep - 1]),
                          largest_discarded=float(s_sorted[keep]) if keep < s_sorted.size
                          else 0.0)

    kept_per_block = np.zeros(len(per_block), dtype=np.int64)
    for g in all_block[order[:keep]]:
        kept_per_block[int(g)] += 1

    bond_entries = []           # (qn, k, block_index)
    for b, (row, _, _, _) in enumerate(per_block):
        k = int(kept_per_block[b])
        if k:
            bond_entries.append((m.spaces[0].qns[int(row[0])], k, b))
    bond_space = Space([(qn, k) for qn, k, _ in bond_entries])

    u_rows, u_blocks, s_sectors, vh_rows, vh_blocks = [], [], [], [], []
    for qn, k, b in bond_entries:
        row, u, s, vh = per_block[b]
        j = bond_space.sector_index(qn)
        u_rows.append((int(row[0]), j))
        u_blocks.append(np.ascontiguousarray(u[:, :k]))
        s_sectors.append(np.ascontiguousarray(s[:k], dtype=np.float64))
        vh_rows.append((j, int(row[1])))
        vh_blocks.append(np.ascontiguousarray(vh[:k, :]))

    order_u = np.argsort(_row_keys(np.asarray(u_rows, dtype=np.int64),
                                   [m.spaces[0].nsectors, bond_space.nsectors]))
    order_v = np.argsort(_row_keys(np.asarray(vh_rows, dtype=np.int64),
                                   [bond_space.nsectors, m.spaces[1].nsectors]))
    u_m = BlockTensor((m.spaces[0], bond_space), (1, -1),
                      QuantumNumber.zero(tensor.charge.width),
                      np.asarray(u_rows, dtype=np.int64)[order_u],
                      [u_blocks[i] for i in order_u])
    vh_m = BlockTensor((bond_space, m.spaces[1]), (1, 1), tensor.charge,
                       np.asarray(vh_rows, dtype=np.int64)[order_v],
                       [vh_blocks[i] for i in order_v])
    # s_sectors aligned to bond_space sector order (ascending qn), matching U's bond leg
    s_by_sector = [None] * bond_space.nsectors
    for (qn, k, b), s_arr in zip(bond_entries, s_sectors):
        s_by_sector[bond_space.sector_index(qn)] = s_arr

    return (split(u_m, rec_l, axis=0), s_by_sector, split(vh_m, rec_r, axis=1), info)


@kernels.kernel("block_pair_gemm")
def block_pair_gemm_numpy(a_data: np.ndarray, a_offset: np.ndarray, b_data: np.ndarray,
                          b_offset: np.ndarray, pairs: np.ndarray, dims: np.ndarray,
                          out_data: np.ndarray, out_offset: np.ndarray,
                          n_threads: int) -> np.ndarray:
    """Accumulate ``out[io] += A[ia] @ B[ib]`` over a table of matched block pairs.

    **The tensor-network hot kernel**: every environment build, every
    effective-Hamiltonian application and every RDM contraction of :mod:`kuiva.dmrg`
    reduces to this loop. The block structure is flattened away before the call, exactly so
    a compiled backend can own it:

    * ``a_data``/``b_data``/``out_data`` are flat ``complex128`` buffers of row-major
      matricized blocks; ``a_offset``/``b_offset``/``out_offset`` are their ``int64`` start
      indices, one per block plus a final total (B1/B3);
    * ``pairs`` is ``(npair, 4)`` ``int64`` — ``(block in A, block in B, block in out,
      beta)`` — and ``dims`` is ``(npair, 3)`` ``int64`` — ``(m, k, n)`` (B3, rectangular,
      never ragged, never hash-addressed: B2). ``beta`` is 0 on the first pair targeting an
      output block and 1 afterwards, exactly the BLAS/batched-GEMM parameter, so the caller
      need not pre-zero and no pass over the output is wasted;
    * ``out_data`` is caller-provided and may not alias either input (B6).

    * ``n_threads`` is the explicit thread budget over the pair table (B7 applied to
      threads) — never a global, never an environment read. This
      NumPy implementation is serial (its GEMMs thread through whatever BLAS NumPy binds)
      and ignores it; the compiled backend parallelizes across the table with it.

    ⚠ **Every pair is one Fortran ``zgemm`` call, degenerate shapes included, on both
    backends** — this kernel goes through SciPy's direct BLAS binding, never through
    ``numpy.dot``. Deliberate, and load-bearing for the bitwise kernel-parity claim:
    ``numpy.dot`` silently serves a product whose operand has size 1 through its own
    build-dependent SIMD multiply loop (measured 2026-08-09: ~1 ulp from every BLAS
    routine, different again between NumPy builds), so a compiled backend can match it on
    those shapes only by reproducing a specific NumPy binary. One shared BLAS entry point
    makes parity a property of the *contract* instead. The call is column-major on the
    transposed views (``C^T = B^T A^T``), which is a plain view of the same row-major
    buffers — no copy — and the pair's ``beta`` goes straight to the routine.

    ⚠ **Reduction order (B10):** the accumulation into an output block runs in pair-table
    order, and several pairs may target the same block. A threaded backend that splits the
    table or batches the ``beta = 1`` accumulations will reorder that sum; parity is then
    1e-13 relative, not bitwise (the tolerance fixed in advance for B10 kernels). The registered native backend
    deliberately avoids both by owner-computes over output blocks, and is bitwise at any
    fixed BLAS thread width; MKL's *own* in-GEMM reduction order changes with that width,
    which is a property of the BLAS, not of either backend.

    ⚠ **Measured, and it is why this boundary is worth having** (measured): the
    blocks here are small enough that intra-GEMM BLAS threading buys **nothing** — a sweep
    takes the same wall time on 1 and on 8 threads while spending 4x the CPU on spin-wait.
    The parallelism a port must claim is *across* the pair table, which is this signature
    and not a translation of it.
    """
    if a_data.dtype != np.complex128 or b_data.dtype != np.complex128:
        raise TypeError("operand buffers must be complex128, got {} and {}"
                        .format(a_data.dtype, b_data.dtype))
    if out_data.dtype != np.complex128:
        raise TypeError("the output buffer must be complex128, got {}".format(out_data.dtype))
    if not (a_data.flags.c_contiguous and b_data.flags.c_contiguous
            and out_data.flags.c_contiguous):
        raise ValueError("operand and output buffers must be C-contiguous")
    if np.shares_memory(out_data, a_data) or np.shares_memory(out_data, b_data):
        raise ValueError("the output buffer may not alias an operand")
    if n_threads < 1:
        raise ValueError("the thread count must be a positive integer")
    # The index tables are hoisted to Python integers **once, outside the loop**: NumPy
    # scalar indexing costs a measured 23% of this loop's time, and nothing per pair may be
    # spent on bookkeeping (B8). A compiled backend reads the int64 arrays directly.
    ia_all = pairs[:, 0].tolist()
    ib_all = pairs[:, 1].tolist()
    io_all = pairs[:, 2].tolist()
    beta_all = pairs[:, 3].tolist()
    m_all = dims[:, 0].tolist()
    k_all = dims[:, 1].tolist()
    n_all = dims[:, 2].tolist()
    a_start = a_offset.tolist()
    b_start = b_offset.tolist()
    o_start = out_offset.tolist()
    for p in range(pairs.shape[0]):
        m = m_all[p]
        k = k_all[p]
        n = n_all[p]
        s = a_start[ia_all[p]]
        t = b_start[ib_all[p]]
        u = o_start[io_all[p]]
        am = a_data[s:s + m * k].reshape(m, k)
        bm = b_data[t:t + k * n].reshape(k, n)
        om = out_data[u:u + m * n].reshape(m, n)
        # C^T = B^T A^T on F-contiguous views of the same buffers (docstring: the one
        # shared zgemm entry point both backends dispatch to, beta included).
        _blas_zgemm(1.0, bm.T, am.T, beta=float(beta_all[p]), c=om.T, overwrite_c=1)
    return out_data


def tensordot(a: BlockTensor, b: BlockTensor,
              axes: Tuple[Sequence[int], Sequence[int]]) -> BlockTensor:
    """Contract ``a``'s legs ``axes[0]`` with ``b``'s legs ``axes[1]``, pairwise.

    Contracted leg pairs must carry **equal spaces and opposite signs** — anything else is a
    bookkeeping error upstream and is refused, because contracting mismatched legs produces
    a plausible-shaped wrong tensor. Output legs are ``a``'s uncontracted followed by
    ``b``'s, in order (NumPy's ``tensordot`` layout); the output charge is the sum.

    ⚠ Orchestration, but *hot-path* orchestration: this is the arithmetic core of every
    environment build and every effective-Hamiltonian application in the sweep. It
    matricizes each block once into a flat buffer, builds the pair table, and hands the
    arithmetic to :func:`block_pair_gemm_numpy` through the ``ci/kernels.py`` registry — so
    a compiled backend registers itself and this function does not change. ⚠ Measured
    motive for matricizing once (n = 12, D = 32 sweep profile): ``np.tensordot`` per pair
    re-derives axes and re-matricizes both operands through the generic dispatch machinery
    — 70 of 85 CPU s, ~105 us per call on blocks whose arithmetic is worth ~5.
    """
    axes_a = tuple(int(x) for x in axes[0])
    axes_b = tuple(int(x) for x in axes[1])
    if len(axes_a) != len(axes_b):
        raise ValueError("axes lists differ in length")
    for ia, ib in zip(axes_a, axes_b):
        if a.spaces[ia] != b.spaces[ib]:
            raise ValueError("contracted legs {}<->{} carry different spaces".format(ia, ib))
        if a.signs[ia] != -b.signs[ib]:
            raise ValueError("contracted legs {}<->{} carry equal signs; flux would not "
                             "cancel".format(ia, ib))
    rest_a = tuple(i for i in range(a.ndim) if i not in axes_a)
    rest_b = tuple(i for i in range(b.ndim) if i not in axes_b)
    spaces = tuple(a.spaces[i] for i in rest_a) + tuple(b.spaces[i] for i in rest_b)
    signs = tuple(a.signs[i] for i in rest_a) + tuple(b.signs[i] for i in rest_b)
    charge = a.charge + b.charge
    if not spaces:
        raise ValueError("full contraction to a scalar is not represented as a BlockTensor;"
                         " keep at least one leg (use a dim-1 leg for scalars)")

    perm_a = rest_a + axes_a
    perm_b = axes_b + rest_b
    b_index: Dict[tuple, List[int]] = {}
    b_dims = np.empty((b.nblocks, 2), dtype=np.int64)
    for jb, row in enumerate(b.sectors):
        block = b.blocks[jb]
        cols = 1
        for i in rest_b:
            cols *= block.shape[i]
        b_dims[jb] = (block.size // max(cols, 1), cols)
        b_index.setdefault(tuple(int(row[i]) for i in axes_b), []).append(jb)

    a_dims = np.empty((a.nblocks, 2), dtype=np.int64)
    for ja in range(a.nblocks):
        block = a.blocks[ja]
        rows_dim = 1
        for i in rest_a:
            rows_dim *= block.shape[i]
        a_dims[ja] = (rows_dim, block.size // max(rows_dim, 1))

    pair_rows: List[tuple] = []
    pair_dims: List[tuple] = []
    out_index: Dict[tuple, int] = {}
    for ja, row_a in enumerate(a.sectors):
        partners = b_index.get(tuple(int(row_a[i]) for i in axes_a))
        if not partners:
            continue
        head = tuple(int(row_a[i]) for i in rest_a)
        for jb in partners:
            new_row = head + tuple(int(b.sectors[jb][i]) for i in rest_b)
            io = out_index.get(new_row)
            beta = 1
            if io is None:
                io = out_index[new_row] = len(out_index)
                beta = 0                          # first pair on this block: overwrite
            pair_rows.append((ja, jb, io, beta))
            pair_dims.append((int(a_dims[ja, 0]), int(a_dims[ja, 1]), int(b_dims[jb, 1])))

    rows = np.array(sorted(out_index), dtype=np.int64).reshape(len(out_index), len(spaces))
    if not pair_rows:
        return BlockTensor._trusted(spaces, signs, charge, rows,
                                    _row_keys(rows, [sp.nsectors for sp in spaces]), [])
    # renumber output blocks into the sorted row order, so the buffer is laid out as the
    # block table is and the payloads below are plain views into it
    order = {tuple(int(i) for i in r): k for k, r in enumerate(rows)}
    remap = np.empty(len(out_index), dtype=np.int64)
    for row_key, io in out_index.items():
        remap[io] = order[row_key]
    pairs = np.asarray(pair_rows, dtype=np.int64)
    pairs[:, 2] = remap[pairs[:, 2]]
    dims = np.asarray(pair_dims, dtype=np.int64)

    a_data, a_offset = _matricized_buffer(a.blocks, perm_a, a_dims)
    b_data, b_offset = _matricized_buffer(b.blocks, perm_b, b_dims)
    out_sizes = np.zeros(rows.shape[0], dtype=np.int64)
    out_sizes[pairs[:, 2]] = dims[:, 0] * dims[:, 2]
    out_offset = np.concatenate([[0], np.cumsum(out_sizes)]).astype(np.int64)
    # not zeroed: every output block's first pair carries beta = 0 and overwrites it
    out_data = np.empty(int(out_offset[-1]), dtype=np.complex128)
    kernels.resolve("block_pair_gemm")(a_data, a_offset, b_data, b_offset, pairs, dims,
                                       out_data, out_offset, threads.thread_count())

    blocks = []
    for k, r in enumerate(rows):
        shape = tuple(int(sp.dims[i]) for sp, i in zip(spaces, r))
        blocks.append(out_data[int(out_offset[k]):int(out_offset[k + 1])].reshape(shape))
    # correct by construction: operand rows were valid and flux adds — the trusted path
    # exists because re-validating here was half the cost of a DMRG sweep (see _trusted)
    return BlockTensor._trusted(spaces, signs, charge, rows,
                                _row_keys(rows, [sp.nsectors for sp in spaces]), blocks)


def _matricized_buffer(blocks: List[np.ndarray], perm: Tuple[int, ...],
                       dims: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pack every block, transposed to ``perm`` and matricized, into one flat buffer.

    ⚠ ``reshape(-1)`` of a transposed block materializes a contiguous temporary and the
    assignment then copies it again — two passes. Assigning a *permuted view* of the
    destination instead does one strided pass, and is **slower** (measured: +5% of a whole
    sweep): NumPy's strided copy loses more than the extra memcpy costs. Recorded because
    the flop/pass count says the opposite.
    """
    sizes = dims[:, 0] * dims[:, 1]
    offset = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    data = np.empty(int(offset[-1]), dtype=np.complex128)
    for j, block in enumerate(blocks):
        data[int(offset[j]):int(offset[j + 1])] = block.transpose(perm).reshape(-1)
    return data, offset


def _joined_rows(rows_a: np.ndarray, rows_b: np.ndarray,
                 axes_a: Sequence[int], axes_b: Sequence[int],
                 rest_a: Sequence[int], rest_b: Sequence[int],
                 nsec_a: Sequence[int], nsec_b: Sequence[int]) -> np.ndarray:
    """The sector table :func:`tensordot` would produce, from the operands' tables alone.

    A relational join on the contracted sector indices: every pair of rows that agree there
    contributes one output row (uncontracted of ``a`` then of ``b``), and duplicates
    collapse because several pairs write the same block.

    ⚠ **A second implementation of the rule** :func:`tensordot` applies, and deliberately
    so: that one is fused into the pair-table loop it has to build anyway (hot path), while
    this one must run with no operand data in memory at all — which is the whole point of
    sizing a chain before allocating it. The two are pinned against each other by a test on
    every contraction a sweep performs, so a drift fails rather than mis-sizing silently.
    """
    axes_a, axes_b = list(axes_a), list(axes_b)
    rest_a, rest_b = list(rest_a), list(rest_b)
    width = len(rest_a) + len(rest_b)
    if rows_a.shape[0] == 0 or rows_b.shape[0] == 0:
        return np.zeros((0, width), dtype=np.int64)
    key_a = _row_keys(rows_a[:, axes_a], [nsec_a[i] for i in axes_a])
    key_b = _row_keys(rows_b[:, axes_b], [nsec_b[i] for i in axes_b])
    order = np.argsort(key_b, kind="stable")
    sorted_b = key_b[order]
    lo = np.searchsorted(sorted_b, key_a, side="left")
    hi = np.searchsorted(sorted_b, key_a, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0, width), dtype=np.int64)
    idx_a = np.repeat(np.arange(rows_a.shape[0], dtype=np.int64), counts)
    # position within each matched run, without a Python loop over the runs
    offsets = np.arange(total, dtype=np.int64) \
        - np.repeat(np.cumsum(counts) - counts, counts)
    idx_b = order[np.repeat(lo, counts) + offsets]
    out = np.concatenate([rows_a[idx_a][:, rest_a], rows_b[idx_b][:, rest_b]], axis=1)
    return np.unique(out, axis=0)


class BlockShape(object):
    """A block tensor's structure — spaces, signs, charge, sector table — without its data.

    Why it exists: in this layer the array that decides whether a run fits in memory is not
    a declared object but a **contraction intermediate**, produced and freed inside one
    application of the effective Hamiltonian. Its size is fixed by the operands' structure
    alone, so a whole contraction chain can be walked in structure-space and sized exactly
    before a byte of it is allocated — which is what a memory limit needs if it is to refuse
    rather than be discovered by the kernel's OOM killer.

    ⚠ **Exact, not bounding.** :func:`block_tensor_gb` sizes the *allowed* sector set, which
    a contracted tensor rarely fills: measured on a two-site application of a 20-spinor
    network, the allowed set was 4.7x the tensor that was actually built. Sizing a
    reservation that way would refuse calculations that run comfortably, so this class
    propagates the real sector table instead.
    """

    __slots__ = ("spaces", "signs", "charge", "sectors")

    def __init__(self, spaces: Sequence[Space], signs: Sequence[int],
                 charge: QuantumNumber, sectors: np.ndarray) -> None:
        self.spaces = tuple(spaces)
        self.signs = tuple(int(s) for s in signs)
        self.charge = charge
        self.sectors = np.asarray(sectors, dtype=np.int64).reshape(-1, len(self.spaces))

    @classmethod
    def of(cls, tensor) -> "BlockShape":
        """The structure of a live tensor — a :class:`BlockTensor` or a ``SparseW``."""
        return cls(tensor.spaces, tensor.signs, tensor.charge, tensor.sectors)

    @classmethod
    def allowed(cls, spaces: Sequence[Space], signs: Sequence[int],
                charge: Optional[QuantumNumber] = None) -> "BlockShape":
        """Every symmetry-allowed sector — the structure :meth:`BlockTensor.zeros` builds."""
        spaces = tuple(spaces)
        if charge is None:
            charge = QuantumNumber.zero(spaces[0].width)
        return cls(spaces, signs, charge, _allowed_rows(spaces, signs, charge))

    @property
    def ndim(self) -> int:
        return len(self.spaces)

    @property
    def nblocks(self) -> int:
        return int(self.sectors.shape[0])

    @property
    def size(self) -> int:
        """Number of complex entries across every block."""
        if self.nblocks == 0:
            return 0
        return int(self.block_sizes.sum())

    @property
    def block_sizes(self) -> np.ndarray:
        """Element count of each block, in sector-table order."""
        dims = np.ones(self.nblocks, dtype=np.int64)
        for j, space in enumerate(self.spaces):
            dims = dims * np.asarray(space.dims, dtype=np.int64)[self.sectors[:, j]]
        return dims

    @property
    def nbytes(self) -> int:
        """Exactly what :attr:`BlockTensor.nbytes` reports for a tensor of this structure."""
        return int(16 * self.size + self.sectors.nbytes)

    @property
    def gb(self) -> float:
        return self.nbytes / 1024.0 ** 3

    def dot(self, other: "BlockShape",
            axes: Tuple[Sequence[int], Sequence[int]]) -> "BlockShape":
        """The structure :func:`tensordot` would return for these operands and axes."""
        axes_a = tuple(int(x) for x in axes[0])
        axes_b = tuple(int(x) for x in axes[1])
        if len(axes_a) != len(axes_b):
            raise ValueError("axes lists differ in length")
        for ia, ib in zip(axes_a, axes_b):
            if self.spaces[ia] != other.spaces[ib]:
                raise ValueError("contracted legs {}<->{} carry different spaces"
                                 .format(ia, ib))
            if self.signs[ia] != -other.signs[ib]:
                raise ValueError("contracted legs {}<->{} carry equal signs; flux would "
                                 "not cancel".format(ia, ib))
        rest_a = tuple(i for i in range(self.ndim) if i not in axes_a)
        rest_b = tuple(i for i in range(other.ndim) if i not in axes_b)
        spaces = tuple(self.spaces[i] for i in rest_a) \
            + tuple(other.spaces[i] for i in rest_b)
        if not spaces:
            raise ValueError("full contraction to a scalar is not represented as a "
                             "BlockTensor; keep at least one leg (use a dim-1 leg)")
        signs = tuple(self.signs[i] for i in rest_a) + tuple(other.signs[i] for i in rest_b)
        rows = _joined_rows(self.sectors, other.sectors, axes_a, axes_b, rest_a, rest_b,
                            [sp.nsectors for sp in self.spaces],
                            [sp.nsectors for sp in other.spaces])
        return BlockShape(spaces, signs, self.charge + other.charge, rows)

    def transpose(self, perm: Sequence[int]) -> "BlockShape":
        """Legs permuted; the block table is re-sorted, as :meth:`BlockTensor.transpose`
        does, so the two tables stay comparable row for row."""
        perm = [int(i) for i in perm]
        spaces = tuple(self.spaces[i] for i in perm)
        sectors = self.sectors[:, perm]
        order = np.argsort(_row_keys(sectors, [sp.nsectors for sp in spaces]))
        return BlockShape(spaces, tuple(self.signs[i] for i in perm), self.charge,
                          np.ascontiguousarray(sectors[order]))

    def conj(self) -> "BlockShape":
        """Signs flipped and charge negated, exactly as :meth:`BlockTensor.conj` does."""
        return BlockShape(self.spaces, tuple(-s for s in self.signs), -self.charge,
                          self.sectors)

    def __repr__(self) -> str:                                   # pragma: no cover - debug
        return "BlockShape({} legs, {} blocks, {:.4f} GB)".format(
            self.ndim, self.nblocks, self.gb)


def block_tensor_gb(spaces: Sequence[Space], signs: Sequence[int],
                    charge: Optional[QuantumNumber] = None) -> float:
    """Exact size [GB] of the fully-allocated block tensor (exact sizing function).

    Matches ``BlockTensor.zeros(...).nbytes`` exactly — payload of every symmetry-allowed
    block plus the sector table — and is pinned two-sided against it in the tests. A tensor
    produced by contraction can only be smaller (some allowed blocks absent), so as a
    reservation this is exact for what :meth:`BlockTensor.zeros` allocates and an upper
    bound for everything else with the same spaces.
    """
    spaces = tuple(spaces)
    if charge is None:
        charge = QuantumNumber.zero(spaces[0].width)
    rows = _allowed_rows(spaces, signs, charge)
    payload = 0
    for row in rows:
        size = 1
        for sp, i in zip(spaces, row):
            size *= int(sp.dims[i])
        payload += size
    return (16.0 * payload + 8.0 * rows.size) / 1024.0 ** 3


def _degenerate_groups(s: np.ndarray, rtol: float) -> List[np.ndarray]:
    """Split a **descending** value array into degenerate groups (index arrays).

    Same construction as the Cholesky orbit grouping in
    :mod:`kuiva.integrals.transform`: a new group starts wherever the gap to the previous
    value exceeds ``rtol`` times the largest value. Chained near-degeneracies merge — the
    conservative direction, since keeping a group whole is always allowed.
    """
    if s.size == 0:
        return []
    cuts = np.nonzero(np.diff(s) < -rtol * abs(float(s[0])))[0] + 1
    return np.split(np.arange(s.size), cuts)


__all__ = ["QuantumNumber", "Space", "BlockTensor", "BlockShape", "FuseRecord",
           "TruncationInfo",
           "fuse", "split", "qr", "svd", "tensordot", "block_tensor_gb",
           "block_pair_gemm_numpy",
           "SCHMIDT_DEGENERACY_RTOL", "SCHMIDT_STABILITY_RTOL"]
