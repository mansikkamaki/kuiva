"""Sparse-payload block tensors: the TTNO's W storage and its contraction.

A :class:`~kuiva.dmrg.block.BlockTensor` stores every symmetry-allowed sector as a **dense**
block. That is right for a wavefunction tensor, whose blocks are genuinely full, and wrong
for a compiled operator: a TTNO node's W tensor has legs
``[parent-op, child-ops, phys-out, phys-in]``, and its content is a list of *transitions*
— one operator-index tuple carrying one local matrix — so a dense sector block is mostly
zeros. :class:`SparseW` stores the nonzeros only, and :func:`dot_sparse` contracts it with a
dense :class:`~kuiva.dmrg.block.BlockTensor` directly, never materializing the dense form.

Why this exists — the measurement, not the intuition
-----------------------------------------------------
Two independent factors make a W tensor sparse, and they are worth separating because the
second is the one that killed a real run (measured):

* **Operator-index sparsity.** The charge-sector blocking already removes ~93% of the naive
  ``D_op^deg`` area; what survives is a further ~5-10% dense in the transition indices,
  except at the one node where the completed channel collects the complementary
  contractions (~40-50% there). Net: about 6x.
* **Local-matrix sparsity, which grows with node fatness.** A transition's local matrix is a
  Kronecker product of ``I``/``a``/``a+``/``Z`` over the node's modes, so it carries **at
  most one nonzero per column** — measured 1.5-1.9 nonzeros out of ``d^2 = 16`` for a
  two-mode node, and ~2 out of 1024 for a five-mode node. A dense payload pays ``d^2`` for
  content of size ``d``.

The product is what matters: at one mode per node the whole W of a 20-spinor path is 24 MB
dense and sparsity buys ~6x, while the five-mode nodes of a multi-site trimer paid the
``d^2`` factor on top and put the operator at ~13 GB — an OOM kill with no diagnosis. ⚠ So
the original diagnosis ("the ``D_op^2`` sector-block scaling is the structural cost")
is only half right, and the half it missed is the one that binds: **node fatness is not a
constant factor on a small term, it is the dominant term.**

Conventions
-----------
* Legs, signs, charge and the sorted ``sectors`` table are **exactly**
  :class:`~kuiva.dmrg.block.BlockTensor`'s (that module's flux convention, verbatim), so a
  sparse and a dense W are interchangeable descriptions of the same operator and
  :meth:`SparseW.to_block_tensor` is a pure re-embedding.
* Entries are stored as one flat C-order index into the block's shape plus a value, grouped
  by block through an ``indptr`` and **sorted ascending within a block** — a rectangular,
  hash-free layout (B2/B3) a compiled backend can consume unchanged.
* :func:`dot_sparse` is the **hot path**: it is the sparse counterpart of
  :func:`kuiva.dmrg.block.tensordot` and is reached from exactly one place, the sweep's
  ``_Lab.dot``. Its inner work is one ``scipy.sparse`` CSR times dense product per matched
  block pair, with the CSR built once per contraction pattern and cached on the operator —
  every node is contracted in the same few patterns for the whole run.

⚠ Sparse and dense contraction sum in different orders, so agreement between the two paths
is to rounding (~1e-14 relative), **not bitwise**. The tests assert the tolerance rather
than equality, and the reduction order is named here because a threaded port would reorder
it again (a B10 reduction-order note).

References
----------
* Sparse (list-of-transitions) MPO storage and its contraction: C. Hubig, I. P. McCulloch,
  U. Schollwoeck, Phys. Rev. B 95, 035129 (2017), doi:10.1103/PhysRevB.95.035129;
  G. K.-L. Chan, A. Keselman, N. Nakatani, Z. Li, S. R. White, J. Chem. Phys. 145, 014102
  (2016), doi:10.1063/1.4955108.
* Complementary operators, whose label structure is what makes the transition list short:
  S. R. White, R. L. Martin, J. Chem. Phys. 110, 4127 (1999), doi:10.1063/1.478522.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp
# The kernel's engine, bound once: SciPy's canonical CSR-times-dense-block C loop. Going
# through ``sp.csr_matrix.dot`` instead costs a measured ~12% of a whole sweep in dispatch
# and per-call allocation (measured), and the raw routine accumulates into a
# caller-provided buffer, which is what the kernel contract needs.
from scipy.sparse._sparsetools import csr_matvecs as _csr_matvecs

from ..ci import kernels
from ..util import threads
from .block import (BlockTensor, QuantumNumber, Space, _flux, _matricized_buffer,
                    _row_keys)


def _block_shape(spaces: Sequence[Space], row: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(s.dims[int(i)]) for s, i in zip(spaces, row))


#: Reused operand-pack scratch for :func:`dot_sparse` (grown geometrically, never shrunk).
#: A *transient* buffer in the budget's sense — kernel input only, dead after the call — kept
#: alive across calls purely to avoid per-call mmap page faults (see the pack comment in
#: :func:`dot_sparse`). Safe because sweeps drive contractions from the main thread only
#: (`util/timing.py` records the same assumption).
_PACK_SCRATCH = np.empty(0, dtype=np.complex128)


def _pack_scratch(n: int) -> np.ndarray:
    global _PACK_SCRATCH
    if _PACK_SCRATCH.size < n:
        _PACK_SCRATCH = np.empty(n + (n >> 1) + 16, dtype=np.complex128)
    return _PACK_SCRATCH[:n]


class SparseW(object):
    """A block tensor whose sector blocks carry only their nonzero entries.

    ``sectors`` is the sorted int64 ``(nblocks, ndim)`` table of
    :class:`~kuiva.dmrg.block.BlockTensor`; block ``b``'s entries are
    ``flat[indptr[b]:indptr[b + 1]]`` (C-order indices into the block's shape) with the
    matching ``values``. Construct through :meth:`from_entries`, which does the sorting and
    the validation.
    """

    __slots__ = ("spaces", "signs", "charge", "sectors", "indptr", "flat", "values",
                 "_keys", "_csr_cache", "_closed")

    def __init__(self, spaces, signs, charge, sectors, indptr, flat, values, keys=None):
        self._closed: Optional["SparseW"] = None
        self.spaces = tuple(spaces)
        self.signs = tuple(int(s) for s in signs)
        self.charge = charge
        self.sectors = sectors
        self.indptr = indptr
        self.flat = flat
        self.values = values
        self._keys = _row_keys(sectors, [s.nsectors for s in self.spaces]) \
            if keys is None else keys
        self._csr_cache: Dict[tuple, dict] = {}

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def from_entries(cls, spaces: Sequence[Space], signs: Sequence[int],
                     charge: QuantumNumber, rows: Sequence[Sequence[int]],
                     flat: Sequence[int], values: Sequence[complex]) -> "SparseW":
        """Build from one ``(row, flat-index, value)`` triple per nonzero.

        Duplicate ``(row, flat)`` pairs are **summed**, which is what an operator compiler
        produces when several terms share a transition. Rows are validated against the flux
        rule and the block shapes exactly as the dense constructor does — this is the
        public path, and it is not on the hot loop (one call per node per compile).
        """
        spaces = tuple(spaces)
        signs = tuple(int(s) for s in signs)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, len(spaces))
        flat = np.asarray(flat, dtype=np.int64)
        values = np.asarray(values, dtype=np.complex128)
        if rows.shape[0] != flat.size or flat.size != values.size:
            raise ValueError("rows, flat indices and values must align")
        nsec = [s.nsectors for s in spaces]
        uniq, inverse = np.unique(rows, axis=0, return_inverse=True) if rows.size \
            else (np.zeros((0, len(spaces)), dtype=np.int64), np.zeros(0, dtype=np.int64))
        for row in uniq:
            if np.any(row < 0) or np.any(row >= np.asarray(nsec)):
                raise ValueError("sector index out of range in row {}".format(row))
            if _flux(spaces, signs, row) != charge:
                raise ValueError("block at {} violates the flux rule (charge {})"
                                 .format(row, charge))
        sizes = np.array([int(np.prod(_block_shape(spaces, r))) for r in uniq],
                         dtype=np.int64)
        inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
        if flat.size and (np.any(flat < 0) or np.any(flat >= sizes[inverse])):
            raise ValueError("a flat entry index lies outside its block")
        order = np.lexsort((flat, inverse))
        inverse, flat, values = inverse[order], flat[order], values[order]
        # merge duplicates (same block, same position): several terms share a transition
        if flat.size:
            new = np.ones(flat.size, dtype=bool)
            new[1:] = (inverse[1:] != inverse[:-1]) | (flat[1:] != flat[:-1])
            group = np.cumsum(new) - 1
            merged = np.zeros(int(group[-1]) + 1, dtype=np.complex128)
            np.add.at(merged, group, values)
            keep = np.nonzero(new)[0]
            inverse, flat, values = inverse[keep], flat[keep], merged
        counts = np.bincount(inverse, minlength=uniq.shape[0]).astype(np.int64)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(spaces, signs, charge, np.ascontiguousarray(uniq), indptr,
                   np.ascontiguousarray(flat), np.ascontiguousarray(values))

    @classmethod
    def from_block_tensor(cls, t: BlockTensor) -> "SparseW":
        """The sparse form of a dense block tensor (tests and the dense-path comparison)."""
        rows, flat, vals = [], [], []
        for row, block in zip(t.sectors, t.blocks):
            idx = np.nonzero(block.ravel())[0]
            rows.extend([tuple(int(i) for i in row)] * idx.size)
            flat.append(idx)
            vals.append(block.ravel()[idx])
        if not rows:
            raise ValueError("a zero operator has no transitions")
        return cls.from_entries(t.spaces, t.signs, t.charge, rows,
                                np.concatenate(flat), np.concatenate(vals))

    def with_values(self, values: np.ndarray) -> "SparseW":
        """The same structure with new entry values — the template's refill path."""
        values = np.ascontiguousarray(values, dtype=np.complex128)
        if values.size != self.values.size:
            raise ValueError("{} values for {} entries".format(values.size,
                                                               self.values.size))
        return SparseW(self.spaces, self.signs, self.charge, self.sectors, self.indptr,
                       self.flat, values, keys=self._keys)

    # --- queries --------------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.spaces)

    @property
    def nblocks(self) -> int:
        return int(self.sectors.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    @property
    def nbytes(self) -> int:
        """Exact stored bytes (sizing is exact, never padded)."""
        return int(self.values.nbytes + self.flat.nbytes + self.indptr.nbytes
                   + self.sectors.nbytes)

    @property
    def dense_nbytes(self) -> int:
        """What the same operator would cost as a dense :class:`BlockTensor` — a
        diagnostic, reported so the sparsity is a measured number and not a claim."""
        payload = sum(int(np.prod(_block_shape(self.spaces, r))) for r in self.sectors)
        return int(16 * payload + self.sectors.nbytes)

    def block_entries(self, b: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(flat indices, values)`` of block ``b`` — the sparse iteration seam."""
        lo, hi = int(self.indptr[b]), int(self.indptr[b + 1])
        return self.flat[lo:hi], self.values[lo:hi]

    def entry_positions(self, rows: Sequence[Sequence[int]],
                        flat: Sequence[int]) -> np.ndarray:
        """Positions of ``(row, flat)`` pairs in :attr:`values`; refuses an absent entry.

        The seam :class:`kuiva.dmrg.ttno.TTNOTemplate` refills through — a slot the
        skeleton does not carry would silently drop a term's coefficient, so it raises.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, self.ndim)
        flat = np.asarray(flat, dtype=np.int64)
        keys = _row_keys(rows, [s.nsectors for s in self.spaces])
        b = np.searchsorted(self._keys, keys)
        if np.any(b >= self._keys.size) or np.any(self._keys[np.minimum(
                b, self._keys.size - 1)] != keys):
            raise KeyError("a requested entry lies in a block this operator does not carry")
        lo, hi = self.indptr[b], self.indptr[b + 1]
        pos = lo + np.array([int(np.searchsorted(self.flat[l:h], f))
                             for l, h, f in zip(lo, hi, flat)], dtype=np.int64)
        if np.any(pos >= hi) or np.any(self.flat[np.minimum(pos, self.nnz - 1)] != flat):
            raise KeyError("a requested entry is not present in its block")
        return pos

    def dense_block(self, b: int) -> np.ndarray:
        """Block ``b`` materialized (one block only — validation and the diagonal)."""
        shape = _block_shape(self.spaces, self.sectors[b])
        out = np.zeros(int(np.prod(shape)), dtype=np.complex128)
        idx, val = self.block_entries(b)
        out[idx] = val
        return out.reshape(shape)

    def to_block_tensor(self) -> BlockTensor:
        """The dense embedding. ⚠ Costs :attr:`dense_nbytes`; validation and the
        model-space contractions of :mod:`kuiva.dmrg.manifold` only."""
        return BlockTensor(self.spaces, self.signs, self.charge, self.sectors,
                           [self.dense_block(b) for b in range(self.nblocks)])

    def to_dense(self) -> np.ndarray:
        return self.to_block_tensor().to_dense()

    def close_leading_leg(self) -> "SparseW":
        """Drop a dimension-1 leading leg, folding its quantum number into the charge.

        The TTNO root's parent leg is the dim-1 completed-Hamiltonian channel; closing it
        with the unit vector is a reshape, and doing it here keeps the sweep from needing a
        dense contraction just to remove a leg of size one. Memoized, because the result
        carries the contraction-pattern cache the sweep depends on.
        """
        if self._closed is not None:
            return self._closed
        if self.spaces[0].total_dim != 1:
            raise ValueError("the leading leg has dimension {}, not 1"
                             .format(self.spaces[0].total_dim))
        qn = self.spaces[0].qns[0]
        charge = self.charge + qn if self.signs[0] < 0 else self.charge - qn
        shapes = [_block_shape(self.spaces, r) for r in self.sectors]
        # dropping a leading axis of extent 1 leaves the C-order flat index unchanged
        assert all(s[0] == 1 for s in shapes)
        block_of = np.repeat(np.arange(self.nblocks, dtype=np.int64),
                             np.diff(self.indptr))
        self._closed = SparseW.from_entries(
            self.spaces[1:], self.signs[1:], charge,
            self.sectors[block_of][:, 1:], self.flat, self.values)
        return self._closed

    def __repr__(self) -> str:
        return "SparseW(ndim={}, nblocks={}, nnz={}, dims={})".format(
            self.ndim, self.nblocks, self.nnz,
            tuple(s.total_dim for s in self.spaces))

    # --- the contraction pattern cache -----------------------------------------------------

    def _pattern(self, axes_w: Tuple[int, ...]) -> dict:
        """CSR blocks for contracting legs ``axes_w``, built once and cached.

        For each block: rows = the free legs' combined index, columns = the contracted
        legs' combined index, so one CSR times the matricized operand does the whole
        block pair. Every node is contracted in the same handful of patterns for a whole
        run, so this is built ``deg`` times per node per solve, never per matvec.

        The pattern carries two views of the same content: ``groups`` (block key ->
        partner list, what :func:`dot_sparse`'s pair enumeration walks) and the **flat
        arrays** the ``sparse_pair_dot`` kernel consumes — every block CSR concatenated
        into one ``(indptr, indices, values)`` triple with an int64 ``(n_csr, 3)`` meta
        table ``(indptr start, entry start, n_rows)``. SciPy builds and canonicalizes each
        CSR (duplicate-free, sorted within a row); the flat copy is made once per pattern
        here, never per matvec, and its within-row entry order **is** the kernel's
        reduction order (B10 — see :func:`sparse_pair_dot_numpy`).
        """
        cached = self._csr_cache.get(axes_w)
        if cached is not None:
            return cached
        rest_w = tuple(i for i in range(self.ndim) if i not in axes_w)
        groups: Dict[tuple, List[tuple]] = {}
        csr_list: List[sp.csr_matrix] = []
        for b in range(self.nblocks):
            row = self.sectors[b]
            shape = _block_shape(self.spaces, row)
            idx, val = self.block_entries(b)
            if idx.size == 0:
                continue
            multi = np.unravel_index(idx, shape)
            in_dims = tuple(shape[i] for i in axes_w)
            out_dims = tuple(shape[i] for i in rest_w)
            in_idx = np.ravel_multi_index(tuple(multi[i] for i in axes_w), in_dims) \
                if axes_w else np.zeros(idx.size, dtype=np.int64)
            out_idx = np.ravel_multi_index(tuple(multi[i] for i in rest_w), out_dims) \
                if rest_w else np.zeros(idx.size, dtype=np.int64)
            n_in = int(np.prod(in_dims)) if in_dims else 1
            n_out = int(np.prod(out_dims)) if out_dims else 1
            csr = sp.csr_matrix((val, (out_idx, in_idx)), shape=(n_out, n_in))
            key = tuple(int(row[i]) for i in axes_w)
            groups.setdefault(key, []).append(
                (tuple(int(row[i]) for i in rest_w), out_dims, len(csr_list),
                 n_out, n_in))
            csr_list.append(csr)
        meta = np.empty((len(csr_list), 3), dtype=np.int64)
        ip_parts: List[np.ndarray] = []
        ix_parts: List[np.ndarray] = []
        val_parts: List[np.ndarray] = []
        ip0 = nz0 = 0
        for j, csr in enumerate(csr_list):
            meta[j] = (ip0, nz0, csr.shape[0])
            ip_parts.append(csr.indptr.astype(np.int64))
            ix_parts.append(csr.indices.astype(np.int64))
            val_parts.append(csr.data.astype(np.complex128, copy=False))
            ip0 += csr.shape[0] + 1
            nz0 += int(csr.nnz)
        pattern = {"rest": rest_w, "groups": groups, "meta": meta,
                   "indptr": (np.concatenate(ip_parts) if ip_parts
                              else np.zeros(0, dtype=np.int64)),
                   "indices": (np.concatenate(ix_parts) if ix_parts
                               else np.zeros(0, dtype=np.int64)),
                   "values": (np.concatenate(val_parts) if val_parts
                              else np.zeros(0, dtype=np.complex128))}
        self._csr_cache[axes_w] = pattern
        return pattern


def dot_sparse(a: BlockTensor, w: SparseW,
               axes: Tuple[Sequence[int], Sequence[int]]) -> BlockTensor:
    """Contract dense ``a`` with sparse ``w`` — the sparse :func:`kuiva.dmrg.block.tensordot`.

    Same contract as the dense routine: paired legs must carry equal spaces and opposite
    signs, output legs are ``a``'s uncontracted followed by ``w``'s uncontracted in order,
    and the output charge is the sum. The result is an ordinary dense
    :class:`~kuiva.dmrg.block.BlockTensor` — only the *operator* is sparse, because only
    the operator is.

    ⚠ Hot-path orchestration, restructured exactly as :func:`kuiva.dmrg.block.tensordot`
    was for its kernel (the post-port re-profile): this wrapper validates, walks the cached
    pattern to build the pair table, packs the matched ``a`` blocks once into a reused
    scratch, and hands the arithmetic to the registered ``sparse_pair_dot`` kernel — so a
    compiled backend registers itself and this function does not change. Measured motive
    (measured): after the `block_pair_gemm` port this loop was 48% of a sweep's
    CPU, serial; the per-pair unpack copy moved into the kernel, and the operand pack
    stayed here with two measured fixes (the reused scratch and the one-pass copy).

    ⚠ B10: the reduction over ``w``'s contracted indices runs in CSR entry order,
    which is neither the dense routine's order nor a naive threaded port's. Parity with
    the **dense** path is ~1e-14 relative, asserted as such; parity between backends of
    ``sparse_pair_dot`` is the kernel's own contract (bitwise — see its docstring).
    """
    axes_a = tuple(int(x) for x in axes[0])
    axes_w = tuple(int(x) for x in axes[1])
    if len(axes_a) != len(axes_w):
        raise ValueError("axes lists differ in length")
    for ia, iw in zip(axes_a, axes_w):
        if a.spaces[ia] != w.spaces[iw]:
            raise ValueError("contracted legs {}<->{} carry different spaces".format(ia, iw))
        if a.signs[ia] != -w.signs[iw]:
            raise ValueError("contracted legs {}<->{} carry equal signs; flux would not "
                             "cancel".format(ia, iw))
    pattern = w._pattern(axes_w)
    rest_w = pattern["rest"]
    rest_a = tuple(i for i in range(a.ndim) if i not in axes_a)
    spaces = tuple(a.spaces[i] for i in rest_a) + tuple(w.spaces[i] for i in rest_w)
    signs = tuple(a.signs[i] for i in rest_a) + tuple(w.signs[i] for i in rest_w)
    if not spaces:
        raise ValueError("full contraction to a scalar is not represented as a BlockTensor;"
                         " keep at least one leg (use a dim-1 leg for scalars)")
    charge = a.charge + w.charge
    groups = pattern["groups"]

    perm_a = axes_a + rest_a                     # contracted first: (in_size, rest_dim)
    matched: List[int] = []
    am_dims: List[tuple] = []
    pair_rows: List[tuple] = []
    pair_dims: List[tuple] = []
    out_index: Dict[tuple, int] = {}
    out_shapes: List[tuple] = []
    for ja, row_a in enumerate(a.sectors):
        key = tuple(int(row_a[i]) for i in axes_a)
        partners = groups.get(key)
        if not partners:
            continue
        block = a.blocks[ja]
        rest_dim = 1
        for i in rest_a:
            rest_dim *= block.shape[i]
        in_size = block.size // max(rest_dim, 1)
        iam = len(matched)
        matched.append(ja)
        am_dims.append((in_size, rest_dim))
        head = tuple(int(row_a[i]) for i in rest_a)
        rest_dims = tuple(block.shape[i] for i in rest_a)
        for tail, out_dims, csr_id, n_rows, _n_cols in partners:
            new_row = head + tail
            io = out_index.get(new_row)
            if io is None:
                io = out_index[new_row] = len(out_index)
                out_shapes.append(rest_dims + out_dims)
            pair_rows.append((iam, csr_id, io))
            pair_dims.append((in_size, rest_dim, n_rows))

    rows = np.array(sorted(out_index), dtype=np.int64).reshape(len(out_index), len(spaces))
    if not pair_rows:
        return BlockTensor._trusted(spaces, signs, charge, rows,
                                    _row_keys(rows, [s.nsectors for s in spaces]), [])
    # renumber output blocks into the sorted row order, so the buffer is laid out as the
    # block table is and the views below are plain slices of it (as block.tensordot does)
    order = {tuple(int(i) for i in r): k for k, r in enumerate(rows)}
    remap = np.empty(len(out_index), dtype=np.int64)
    for row_key, io in out_index.items():
        remap[io] = order[row_key]
    pairs = np.asarray(pair_rows, dtype=np.int64)
    dims = np.asarray(pair_dims, dtype=np.int64)
    pairs[:, 2] = remap[pairs[:, 2]]
    shapes_sorted: List[tuple] = [()] * len(out_shapes)
    for io, shape in enumerate(out_shapes):
        shapes_sorted[int(remap[io])] = shape
    out_sizes = np.zeros(rows.shape[0], dtype=np.int64)
    out_sizes[pairs[:, 2]] = dims[:, 1] * dims[:, 2]
    out_offset = np.concatenate([[0], np.cumsum(out_sizes)]).astype(np.int64)
    # zero-filled, not empty: the kernel only ever accumulates, and a CSR row with no
    # entries must leave exact zeros in the elements it never touches
    out_data = np.zeros(int(out_offset[-1]), dtype=np.complex128)
    am_dims_arr = np.asarray(am_dims, dtype=np.int64)
    am_offset = np.concatenate([[0], np.cumsum(am_dims_arr[:, 0]
                                               * am_dims_arr[:, 1])]).astype(np.int64)
    # ⚠ The operand pack reuses one module-level scratch and copies in ONE strided pass,
    # and both halves are measured, not stylistic (measured): a fresh
    # tens-of-MB buffer per call is served by mmap and pays its page faults every call
    # (6.0 -> 2.8 s on the reference workload), and `dest[:] = t.reshape(-1)` is two
    # passes where `np.copyto` through a permuted destination view is one (2.8 -> 2.2 s).
    # The opposite measured finding for the GEMM operands stands — these are different shapes,
    # measured separately; _matricized_buffer is deliberately not changed.
    am_data = _pack_scratch(int(am_offset[-1]))
    for i, j in enumerate(matched):
        block = a.blocks[j]
        view = am_data[int(am_offset[i]):int(am_offset[i + 1])]
        source = block.transpose(perm_a)
        np.copyto(view.reshape(source.shape), source)
    kernels.resolve("sparse_pair_dot")(
        am_data, am_offset, pattern["values"], pattern["indices"], pattern["indptr"],
        pattern["meta"], pairs, dims, out_data, out_offset, threads.thread_count())
    blocks = [out_data[int(out_offset[k]):int(out_offset[k + 1])].reshape(shapes_sorted[k])
              for k in range(rows.shape[0])]
    return BlockTensor._trusted(spaces, signs, charge, rows,
                                _row_keys(rows, [s.nsectors for s in spaces]), blocks)


@kernels.kernel("sparse_pair_dot")
def sparse_pair_dot_numpy(am_data: np.ndarray, am_offset: np.ndarray,
                          csr_values: np.ndarray, csr_indices: np.ndarray,
                          csr_indptr: np.ndarray, csr_meta: np.ndarray,
                          pairs: np.ndarray, dims: np.ndarray,
                          out_data: np.ndarray, out_offset: np.ndarray,
                          n_threads: int) -> np.ndarray:
    """Accumulate ``out[io] += (csr[ic] @ A[ia])^T`` over a table of matched block pairs.

    The sparse-W application — the second tensor-network hot kernel (the first is
    :func:`kuiva.dmrg.block.block_pair_gemm_numpy`, whose buffer
    conventions this follows):

    * ``am_data`` holds the matched ``a`` blocks, each matricized to ``(in_size,
      rest_dim)`` row-major (contracted legs first); ``am_offset`` is int64, one start
      per block plus the total (B1/B3);
    * the operator is a flat concatenation of canonical CSR blocks —
      ``csr_values``/``csr_indices``/``csr_indptr`` — addressed through ``csr_meta``
      ``(n_csr, 3)`` int64 rows ``(indptr start, entry start, n_rows)`` (B2/B3: flat,
      rectangular, no hash anywhere);
    * ``pairs`` is ``(npair, 3)`` int64 ``(a block, csr block, out block)`` and ``dims``
      is ``(npair, 3)`` int64 ``(in_size, rest_dim, out_size)``;
    * ``out_data`` is caller-provided, **zero-filled**, and may not alias an operand
      (B6): the kernel only accumulates, so elements no CSR entry touches stay exactly
      zero. Each output block is stored ``(rest_dim, out_size)`` row-major — the layout
      the block table needs — so the kernel owns the per-pair transposed accumulation
      the pre-kernel implementation paid as a separate copy-then-add (measured;
      the compiled backend fuses it, this implementation keeps the measured
      two-pass form);
    * ``n_threads`` is the explicit thread budget over the pair table (B7 applied to
      threads). This NumPy implementation is serial and ignores it; the compiled backend
      parallelizes across the table with it.

    ⚠ **Reduction order (B10), and it is what the parity claim rests on.** For one output
    element the sum runs: pairs targeting its block in **table order**; within a pair, the
    CSR row's entries in **ascending stored order**, accumulated into a zeroed per-pair
    scratch row which is then added to the output element (one add per pair). A threaded
    backend that splits the table arbitrarily or accumulates entries straight into the
    output would reorder both sums; the registered native backend preserves them by
    owner-computes over output blocks and by reproducing the per-pair scratch, and is
    **bitwise** against this implementation at every thread count. The engine here is
    SciPy's ``csr_matvecs`` (its C loop is exactly the entry-order axpy stated above), so
    the bitwise claim is asserted against the running SciPy build by
    ``tests/test_native_backend.py`` rather than assumed.
    """
    if am_data.dtype != np.complex128 or csr_values.dtype != np.complex128:
        raise TypeError("operand buffers must be complex128, got {} and {}"
                        .format(am_data.dtype, csr_values.dtype))
    if out_data.dtype != np.complex128:
        raise TypeError("the output buffer must be complex128, got {}".format(out_data.dtype))
    if (am_offset.dtype != np.int64 or csr_indices.dtype != np.int64
            or csr_indptr.dtype != np.int64 or csr_meta.dtype != np.int64
            or pairs.dtype != np.int64 or dims.dtype != np.int64
            or out_offset.dtype != np.int64):
        raise TypeError("index tables must be int64")
    if not (am_data.flags.c_contiguous and csr_values.flags.c_contiguous
            and csr_indices.flags.c_contiguous and csr_indptr.flags.c_contiguous
            and out_data.flags.c_contiguous):
        raise ValueError("operand and output buffers must be C-contiguous")
    if np.shares_memory(out_data, am_data) or np.shares_memory(out_data, csr_values):
        raise ValueError("the output buffer may not alias an operand")
    if pairs.ndim != 2 or pairs.shape[1] != 3:
        raise ValueError("pairs must have shape (npair, 3)")
    if dims.ndim != 2 or dims.shape[1] != 3 or dims.shape[0] != pairs.shape[0]:
        raise ValueError("dims must have shape (npair, 3)")
    if csr_meta.ndim != 2 or csr_meta.shape[1] != 3:
        raise ValueError("csr_meta must have shape (n_csr, 3)")
    if n_threads < 1:
        raise ValueError("the thread count must be a positive integer")
    # Index tables hoisted to Python integers once, outside the loop, exactly as
    # block_pair_gemm_numpy does and for the same measured reason.
    ia_all = pairs[:, 0].tolist()
    ic_all = pairs[:, 1].tolist()
    io_all = pairs[:, 2].tolist()
    in_all = dims[:, 0].tolist()
    rest_all = dims[:, 1].tolist()
    out_all = dims[:, 2].tolist()
    a_start = am_offset.tolist()
    o_start = out_offset.tolist()
    ip_start = csr_meta[:, 0].tolist() if csr_meta.size else []
    nz_start = csr_meta[:, 1].tolist() if csr_meta.size else []
    matvecs = _csr_matvecs
    # One scratch for the per-pair piece, sized to the largest pair and zeroed per use:
    # csr_matvecs accumulates, and a fresh np.zeros per pair costs an allocation per pair.
    max_piece = int(np.max(dims[:, 1] * dims[:, 2])) if pairs.shape[0] else 0
    scratch = np.empty(max_piece, dtype=np.complex128)
    for p in range(pairs.shape[0]):
        in_size = in_all[p]
        rest_dim = rest_all[p]
        out_size = out_all[p]
        ic = ic_all[p]
        ip0 = ip_start[ic]
        nz0 = nz_start[ic]
        indptr = csr_indptr[ip0:ip0 + out_size + 1]
        nnz = int(indptr[out_size])
        s = a_start[ia_all[p]]
        u = o_start[io_all[p]]
        am = am_data[s:s + in_size * rest_dim].reshape(in_size, rest_dim)
        piece = scratch[:out_size * rest_dim]
        piece[:] = 0.0
        matvecs(out_size, in_size, rest_dim, indptr, csr_indices[nz0:nz0 + nnz],
                csr_values[nz0:nz0 + nnz], am.reshape(-1), piece)
        out = out_data[u:u + rest_dim * out_size].reshape(rest_dim, out_size)
        # ⚠ ascontiguousarray-then-add, not `out += piece.T`: a counterintuitive, measured
        # strided-copy result holds here too — two fast passes beat one strided pass
        # (measured on the reference workload; same values either way).
        out += np.ascontiguousarray(piece.reshape(out_size, rest_dim).T)
    return out_data


def sparse_w_gb(spaces: Sequence[Space], nnz: int, nblocks: int) -> float:
    """Exact size [GB] of a :class:`SparseW` (exact sizing function; never padded).

    Values plus flat indices plus the block pointer plus the sector table — pinned
    two-sided against :attr:`SparseW.nbytes` in the tests. Both counts are known before
    the arrays are allocated (the compiler enumerates its scatter targets first), so this
    is exact rather than a bound.
    """
    ndim = len(tuple(spaces))
    return (16.0 * int(nnz) + 8.0 * int(nnz) + 8.0 * (int(nblocks) + 1)
            + 8.0 * ndim * int(nblocks)) / 1024.0 ** 3


__all__ = ["SparseW", "dot_sparse", "sparse_w_gb"]
