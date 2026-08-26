"""Occupation strings, Slater-Condon rules and RDMs for complex spinor determinants.

Representation
--------------
A determinant is a **bitmask**: bit ``k`` set means spinor ``k`` is occupied. Spinors are
already the elementary fermionic modes — there is no alpha/beta factorization to
exploit, because spin-orbit coupling breaks spin symmetry — so a single integer *is*
the occupation string, and excitation analysis is bit arithmetic.

Masks are ``uint64``, capping the active space at **64 spinors**. That is a deliberate,
documented limit for the truncated CI this module currently serves: 64 spinors is four
transition-metal d shells or four lanthanide f shells, comfortably past the point where the
determinant-list approach below is the right algorithm anyway. Going further needs multi-word
masks, which is a mechanical change confined to :func:`_popcount` and the bit helpers.

Two addressing modes, both here and nowhere else
-------------------------------------------------------
* **Arbitrary list** (:class:`Determinants`) — an explicit array of masks plus a hash lookup.
  A truncated or selected CI needs it: its space is an arbitrary adaptively chosen subset. The
  cost is that finding which determinant pairs interact is a pairwise search, ``O(N^2)`` — see
  :func:`connections`, a fully vectorized blocked XOR/popcount. For the qualitative
  pre-optimizer (N of order 10^3-10^4) that is the right trade: a few seconds of
  BLAS-free NumPy against a great deal of machinery, and exact within the space.
* **Complete CAS space** (:class:`CASSpace`) — all ``C(n, k)`` determinants, addressed by
  **combinatorial rank**: table-driven, ``O(k)``, pure integer arithmetic, no list lookup at
  all. This is what the full CI runs on. ⚠ A hash table would be wrong here on three
  counts — hundreds of MB at 10^6 determinants, slow, and it puts the address map in Python,
  which blocks a compiled backend outright. No profile could rescue that later, so it is
  decided here rather than deferred.

The ``O(N^2)`` sparse-matrix path and the ``O(N * n^2)`` string-driven sigma vector of
``ci/sigma.py`` are different algorithms for different regimes, and both live on this
representation. Keeping them side by side is deliberate: :func:`hamiltonian_matrix` is what
the sigma vector is validated against, and it is a genuinely independent implementation
(different algorithm, different code path, different sign bookkeeping), which is the only kind
of check that can actually fail.

Conventions (fixed here; the RDMs and the orbital optimizer depend on them)
--------------------------------------------------------------------------
* Determinants are ordered products with **ascending** spinor index::

      |I> = a_{k1}^dag a_{k2}^dag ... |vac>,    k1 < k2 < ... < kn

  so both ``a_p |I>`` and ``a_p^dag |K>`` carry the sign ``(-1)^(occupied below p)``.
* The Hamiltonian is ``H = sum_pq h_pq a_p^dag a_q + 1/2 sum_pqrs (pq|rs) a_p^dag a_r^dag a_s
  a_q`` with ``(pq|rs)`` in chemists' notation, as produced by
  :func:`kuiva.integrals.transform.assemble_4c`.
* The two-particle density matrix is paired to match::

      Gamma_pqrs = <a_p^dag a_r^dag a_s a_q>,     E_2 = 1/2 sum_pqrs (pq|rs) Gamma_pqrs

  with symmetries ``Gamma_pqrs = Gamma_rspq``, ``Gamma_pqrs* = Gamma_qpsr``,
  ``Gamma_pqrs = -Gamma_psrq``, and the trace condition ``sum_r Gamma_pqrr = (N-1) gamma_pq``.
  The last is asserted in the tests: it is a cheap, strong check on the phase bookkeeping.
* **Time reversal on determinants** — the mask (:func:`kramers_partner`), the sign
  (:func:`kramers_sign`), and the two combined as an antiunitary operation on CI vectors
  (:class:`KramersMap`). Defined here and nowhere else, for the same reason as everything
  above: a second derivation of the sign would agree on every test until one of them changed.
  The derivation and the ``T^2 = (-1)^N`` identity are with the functions.

References
----------
* Slater-Condon rules: J. C. Slater, Phys. Rev. 34, 1293 (1929), doi:10.1103/PhysRev.34.1293;
  E. U. Condon, Phys. Rev. 36, 1121 (1930), doi:10.1103/PhysRev.36.1121. Modern treatment
  including the phase conventions used here: T. Helgaker, P. Jorgensen, J. Olsen, "Molecular
  Electronic-Structure Theory", Wiley (2000), ch. 1-2.
* Bitmask determinant representation and excitation analysis by XOR/popcount, as used by
  modern selected-CI codes: A. Scemama, E. Giner, arXiv:1311.6244 (2013); Y. Garniron et al.,
  "Quantum Package 2.0", J. Chem. Theory Comput. 15, 3591 (2019),
  doi:10.1021/acs.jctc.9b00176.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..util import resources as res
from ..util import threads
from ..util.logging import get_logger
from ..util.timing import timer
from . import kernels

log = get_logger(__name__)

#: Maximum number of spinors addressable by a single ``uint64`` mask.
DEFAULT_MAX_SPINORS = 64

#: 16-bit popcount lookup table (64 kB). NumPy 1.x has no bit_count ufunc, and four table
#: lookups plus three adds beat any pure-Python alternative by orders of magnitude.
_POPCNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

_U64 = np.uint64
_MASK16 = _U64(0xFFFF)
_SHIFTS = (_U64(0), _U64(16), _U64(32), _U64(48))

#: ``_BELOW[p]`` has bits 0..p-1 set: the mask selecting spinors *below* ``p``.
_BELOW = np.array([(1 << p) - 1 for p in range(DEFAULT_MAX_SPINORS + 1)], dtype=_U64)

#: Connection batch size for the matrix-element and RDM passes. Bounds the temporaries at
#: ``batch * n_spinor`` rather than ``n_connections * n_spinor``, which is the difference
#: between 70 MB and several GB once the determinant space passes ~10^4.
_CONN_BATCH = 1 << 16


def _popcount(x: np.ndarray) -> np.ndarray:
    """Number of set bits, elementwise, for a ``uint64`` array."""
    x = np.asarray(x, dtype=_U64)
    out = _POPCNT16[np.asarray(x & _MASK16, dtype=np.uint16)].astype(np.int64)
    for s in _SHIFTS[1:]:
        out += _POPCNT16[np.asarray((x >> s) & _MASK16, dtype=np.uint16)]
    return out


def _lowest_bit(x: np.ndarray) -> np.ndarray:
    """Isolate the lowest set bit of each element (``x & -x`` in unsigned arithmetic)."""
    x = np.asarray(x, dtype=_U64)
    return x & (~x + _U64(1))


def _bit_index(single_bit: np.ndarray) -> np.ndarray:
    """Position of the single set bit: ``popcount(b - 1)``."""
    return _popcount(np.asarray(single_bit, dtype=_U64) - _U64(1))


def _below(det: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Number of occupied spinors of ``det`` with index below ``p`` — the fermionic sign
    exponent for both ``a_p`` and ``a_p^dag`` under the ascending-order convention."""
    return _popcount(np.asarray(det, dtype=_U64) & _BELOW[np.asarray(p)])


def _bit(p: np.ndarray) -> np.ndarray:
    return _U64(1) << np.asarray(p, dtype=_U64)


# --- Public aliases for the bit convention -------------------------------------------------
#
# ⚠ The determinant bit convention is defined **here and nowhere else**, and the
# three helpers above are the whole of it: which bit a mode owns, how many modes lie below one,
# and how many are occupied. :mod:`kuiva.qc.mapping` needs exactly these to write the
# Jordan-Wigner parity string, because a JW parity string *is* this module's fermionic sign
# rule spelled in Pauli operators. Re-deriving them there would produce a second convention
# that agrees on every test until the day one of them changes.

def popcount(x) -> np.ndarray:
    """Number of set bits, elementwise, for a ``uint64`` array. Public alias of the internal
    helper; see the note above for why it is exported rather than copied."""
    return _popcount(x)


def mode_bit(p) -> np.ndarray:
    """The ``uint64`` mask with only mode ``p``'s bit set."""
    return _bit(p)


def below_mask(p) -> np.ndarray:
    """The ``uint64`` mask with bits ``0 .. p-1`` set — the modes *below* ``p``.

    ⚠ Not to be confused with :func:`_below`, which counts the *occupied* modes below ``p`` in
    a given determinant. This is the static index mask that count is taken over.
    """
    return _BELOW[np.asarray(p)]


#: Bits of the unbarred (even) spinors of an interleaved Kramers-paired set, and of the barred
#: (odd) ones. kuiva/spinor/expand.py fixes that interleaving; these two constants are the whole of it in mask
#: form and are what :func:`kramers_partner` swaps.
_UNBARRED_BITS = _U64(0x5555555555555555)
_BARRED_BITS = _U64(0xAAAAAAAAAAAAAAAA)


def kramers_partner(masks) -> np.ndarray:
    """The time-reversed occupation pattern: swap each Kramers pair's two bits.

    Under the interleaved convention, spinor ``2p`` is the unbarred partner and ``2p+1`` its
    barred one, so time reversal ``T = -i sigma_y K`` maps the *occupation pattern* of a
    determinant to the pattern with every pair's bits exchanged. Applying it twice is the
    identity on masks.

    ⚠ **The mask, not the phase.** ``T|I>`` carries a sign from ``T^2 = -1`` on each pair and
    from reordering the creation operators back into ascending order; this function does not
    compute it, and no caller may read a returned mask as a *state*. What it is for is
    membership: whether a determinant **space** is closed under time reversal, which is a
    statement about the set of masks alone. The phase is :func:`kramers_sign` and the two are
    combined, once, by :class:`KramersMap`.

    ⚠ Presumes a Kramers-paired spinor set. With an unrestricted reference (``kramers_paired=False``) the spinors are not partnered and the swap means nothing — the
    caller has to know, because a mask array carries no such flag.
    """
    m = np.asarray(masks, dtype=_U64)
    return ((m & _UNBARRED_BITS) << _U64(1)) | ((m & _BARRED_BITS) >> _U64(1))


# --- Time reversal on determinants: the phase kramers_partner deliberately leaves out -------
#
# ⚠ **The whole of the time-reversal determinant algebra is here**, because this module is the
# one place the determinant conventions are defined and a second derivation of the sign would
# agree on every test until the day one of them changed.
#
# On creation operators the ``-i sigma_y K`` convention of kuiva/spinor/expand.py reads
#
#     T a+_{2p} T^-1 = +a+_{2p+1},      T a+_{2p+1} T^-1 = -a+_{2p}
#
# i.e. ``t_k = (-1)^k``. Applying that to the ascending reference string of a determinant and
# then restoring ascending order gives ``T|I> = s_I |Ibar>`` with **two** factors, both pure
# bit arithmetic:
#
#   * ``(-1)^(occupied barred spinors)`` from the ``t_k`` above;
#   * ``(-1)^(closed Kramers pairs)`` from the reordering. A *closed* site contributes the
#     adjacent operator pair ``(2p, 2p+1)``, which comes back as ``(2p+1, 2p)`` — exactly one
#     transposition — while a singly occupied site's single operator keeps its position in the
#     ordering, since the map ``k -> k^1`` never moves an operator past a different site.
#
# so
#
#     s_I = (-1)^(barred occupations of I + closed pairs of I)
#
# and therefore ``s_I s_Ibar = (-1)^N``: T^2 = -1 exactly for an odd electron count, which is
# Kramers' theorem in its determinant form and is asserted as such in the tests.


def kramers_sign(masks) -> np.ndarray:
    """``s_I`` in ``T|I> = s_I |Ibar>``, elementwise ``int8``.

    See the derivation above; ``kramers_partner`` supplies ``Ibar`` and this the phase.

    ⚠ Presumes a Kramers-paired spinor set, exactly as :func:`kramers_partner` does. A mask
    array carries no such flag, so the caller has to know.
    """
    m = np.asarray(masks, dtype=_U64)
    barred = _popcount(m & _BARRED_BITS)
    closed = _popcount((m & (m >> _U64(1))) & _UNBARRED_BITS)
    return (1 - 2 * ((barred + closed) & 1)).astype(np.int8)


def kramers_representative(masks) -> np.ndarray:
    """Boolean mask of the canonical member of each time-reversal pair.

    The canonical member is the one whose **lowest singly occupied Kramers pair is unbarred**,
    which — usefully — is the same thing as the numerically smaller mask: ``I`` and ``Ibar``
    agree on every closed and every empty pair and first differ at the lowest singly occupied
    one, where the unbarred member carries the lower bit. So the predicate is one comparison,
    no table, and it costs nothing to state either way round.

    A **self-conjugate** determinant (``Ibar == I``, all pairs closed or empty — possible only
    for an even electron count) is its own representative and is reported as ``True``.
    """
    m = np.asarray(masks, dtype=_U64)
    return m <= kramers_partner(m)


# --- Sizing for the time-reversal map ------------------------------------------------------

#: Bytes per determinant of a :class:`KramersMap`: an ``int64`` partner index, an ``int8``
#: sign, and the ``float64`` phase the vector operation multiplies by.
BYTES_PER_KRAMERS_ENTRY = 8 + 1 + 8


def kramers_map_gb(ndet: int) -> float:
    """Size [GB] of a :class:`KramersMap` over ``ndet`` determinants (exact sizing function).

    Small against everything it sits beside — 3 MB at the 184 756 determinants of a
    half-filled 20-spinor space, against 1.2 GB of sigma workspace — but it is an
    ``ndet``-sized resident array and is declared like one.
    """
    return float(ndet) * BYTES_PER_KRAMERS_ENTRY / res.BYTES_PER_GB


class KramersMap:
    """Time reversal over a complete CAS space: a permutation, a sign, and a conjugation.

    ``T`` is antiunitary, so on a CI vector it is *not* a matrix. It is exactly three cheap
    operations::

        (T v)_J = s_{Jbar} conj(v_{Jbar})

    — a gather through :attr:`partner`, a conjugation, and a multiply by :attr:`phase`. That
    is ``O(ndet)``, against the ``O(ndet * n_elec * n_empty)`` of one sigma-vector gather, so
    ⚠ **this is orchestration and not a kernel-port candidate**: it is roughly ``1/n^2`` of the
    work of the operator it accompanies and could not clear the port gate at any size.

    Attributes
    ----------
    partner : ``(ndet,)`` ``int64``
        Index of ``Ibar``. An involution: ``partner[partner[I]] == I``.
    sign : ``(ndet,)`` ``int8``
        ``s_I`` of :func:`kramers_sign`.
    phase : ``(ndet,)`` ``float64``
        ``s_{Ibar}``, i.e. ``sign[partner]`` — the factor :meth:`time_reverse` multiplies by,
        precomputed because it is the one that appears in the vector operation. It equals
        ``(-1)^N * sign``, and that identity is what ``T^2 = (-1)^N`` reduces to here.
    representatives : ``(n_pairs,)`` ``int64``
        Indices of the canonical member of each pair (:func:`kramers_representative`). For an
        odd electron count there are exactly ``ndet / 2`` of them and no determinant is
        self-conjugate.
    """

    def __init__(self, space: "CASSpace") -> None:
        n_elec = int(space.n_elec)
        if space.n_spinor % 2 != 0:
            raise ValueError(
                "time reversal pairs spinors 2p and 2p+1, so a Kramers-paired active space "
                "has an even number of spinors; got {}".format(space.n_spinor))
        res.reserve("CAS time-reversal map ({} spinors, {} electrons)"
                    .format(space.n_spinor, n_elec), kramers_map_gb(space.ndet),
                    note="{} determinants".format(space.ndet))
        partner_masks = kramers_partner(space.masks)
        self.partner = np.ascontiguousarray(space.rank(partner_masks), dtype=np.int64)
        self.sign = kramers_sign(space.masks)
        self.phase = np.ascontiguousarray(self.sign[self.partner], dtype=np.float64)
        self.n_elec = n_elec
        self.ndet = int(space.ndet)
        self.representatives = np.ascontiguousarray(
            np.nonzero(kramers_representative(space.masks))[0], dtype=np.int64)
        # Two checks that cost one pass each and fail loudly on the only two ways this can be
        # wrong: a mask array that is not closed under the pair swap (an active space that
        # splits a Kramers pair across its boundary, which the spinor conventions forbid), and
        # a sign convention that has drifted from T^2 = (-1)^N.
        if not np.array_equal(self.partner[self.partner], np.arange(self.ndet)):
            raise RuntimeError("the determinant space is not closed under time reversal: the "
                               "partner map is not an involution")
        expected = -1 if n_elec % 2 else 1
        if not np.all(self.sign[self.partner] * self.sign == expected):
            raise RuntimeError("the time-reversal signs do not satisfy T^2 = {} for {} "
                               "electrons".format(expected, n_elec))

    @property
    def parity(self) -> int:
        """``T^2`` on this space: ``-1`` for an odd electron count, ``+1`` for an even one.

        ⚠ The two are **different theorems and different code paths**. ``-1`` is Kramers'
        theorem — every level at least doubly degenerate, no self-conjugate determinant, and
        the quaternion structure the Kramers-restricted CI runs on. ``+1`` gives neither; what
        it gives instead is a basis in which ``H`` can be made real, which is a separate
        construction and not this one.
        """
        return -1 if self.n_elec % 2 else 1

    @property
    def n_pairs(self) -> int:
        return int(self.representatives.size)

    def time_reverse(self, vectors: np.ndarray,
                     out: Optional[np.ndarray] = None) -> np.ndarray:
        """``T|v>`` for one CI vector ``(ndet,)`` or a stack ``(n, ndet)``.

        ⚠ **Antilinear**: applied to a stack it time-reverses each row *independently*, so
        ``T(sum_i c_i v_i) = sum_i conj(c_i) T v_i`` has to be assembled by the caller with the
        conjugated coefficients. Every consumer in the Kramers-restricted solver does exactly
        that, and getting it wrong is a plausible, norm-preserving, wrong answer.
        """
        v = np.asarray(vectors, dtype=np.complex128)
        if v.shape[-1] != self.ndet:
            raise ValueError("a CI vector over this space has length {}, got {}"
                             .format(self.ndet, v.shape[-1]))
        result = np.take(v, self.partner, axis=-1, out=out)
        np.conjugate(result, out=result)
        result *= self.phase
        return result

    def restrict(self, mask) -> "KramersMap":
        """The same map on a **time-reversal-closed subset** of the determinants.

        ⚠ The subset must be closed under ``T``, and it is checked rather than assumed. A
        symmetry sector generally is **not**: time reversal conjugates an irrep label, so a
        sector maps onto its *conjugate* and only the union of the two is closed. That union
        is what a Kramers-restricted per-irrep solve runs in, and it is why a conjugate pair
        of sectors — not a sector — is the indivisible unit of such a selection.

        The returned map addresses the compressed space (position within ``mask``), so it
        pairs with a compressed operator; nothing about the convention changes.
        """
        mask = np.ascontiguousarray(mask, dtype=bool)
        if mask.shape != (self.ndet,):
            raise ValueError("the subset mask covers {} determinants and this map {}"
                             .format(mask.size, self.ndet))
        index = np.nonzero(mask)[0]
        position = np.full(self.ndet, -1, dtype=np.int64)
        position[index] = np.arange(index.size, dtype=np.int64)
        partner = position[self.partner[index]]
        if np.any(partner < 0):
            missing = int(np.count_nonzero(partner < 0))
            raise ValueError(
                "the subset is not closed under time reversal: {} of its {} determinants have "
                "their time-reversed partner outside it. A symmetry sector is closed only "
                "together with its conjugate sector".format(missing, index.size))
        out = object.__new__(KramersMap)
        out.partner = np.ascontiguousarray(partner, dtype=np.int64)
        out.sign = np.ascontiguousarray(self.sign[index])
        out.phase = np.ascontiguousarray(self.phase[index], dtype=np.float64)
        out.n_elec = self.n_elec
        out.ndet = int(index.size)
        # The canonical member of each pair is the one with the lower compressed index --
        # a choice, not a convention: the only requirement on the representatives is that they
        # take exactly one determinant from each pair, and the mask-based predicate the full
        # map uses does not survive compression.
        out.representatives = np.ascontiguousarray(
            np.nonzero(np.arange(out.ndet) < out.partner)[0], dtype=np.int64)
        return out

    def __repr__(self) -> str:
        return "KramersMap(ndet={}, n_elec={}, T^2={:+d})".format(
            self.ndet, self.n_elec, self.parity)


def ladder_map(masks_from, masks_to, mode: int, *, dagger: bool = False):
    """``(source, target, sign)``: the sparse action of ``a_p`` or ``a_p^dag`` between spaces.

    A single ladder operator maps the ``N``-electron determinant space onto the ``N-1`` (or
    ``N+1``) one, one determinant to at most one determinant. This returns that map as three
    equal-length arrays, so applying it is ``out[target] += sign * vec[source]`` and nothing
    else — see :func:`apply_ladder`, which is that line.

    ⚠ **The sign convention is this module's and is not re-derived anywhere else**.
    Both ``a_p`` and ``a_p^dag`` carry ``(-1)**(occupied modes below p in the SOURCE
    determinant)`` under the ascending-order reference string ``a+_0 a+_1 ... |vac>`` — the
    same rule :func:`_below` states and the same one ``kuiva.qc.mapping`` writes as a
    Jordan-Wigner parity string. A second copy would agree on every test until one changed.

    Parameters
    ----------
    masks_from, masks_to : ``uint64`` arrays, **ascending**
        The two determinant spaces. :attr:`CASSpace.masks` is in rank order, which *is*
        ascending mask order, so the two can be passed directly. A space of a different
        electron count than ``masks_from`` +/- 1 will simply produce no matches and is caught.
    mode : int
        The spinor the operator acts on.
    dagger : bool
        ``False`` for ``a_p`` (the default), ``True`` for ``a_p^dag``.

    Raises
    ------
    ValueError
        If a determinant the operator produces is absent from ``masks_to``. That is not a
        recoverable condition: it means the two spaces are not adjacent in the electron count,
        and silently dropping the terms would give a plausible, smaller, wrong result.
    """
    src = np.ascontiguousarray(masks_from, dtype=_U64)
    dst = np.ascontiguousarray(masks_to, dtype=_U64)
    bit = _bit(int(mode))
    occupied = (src & bit) != 0
    source = np.nonzero(~occupied if dagger else occupied)[0]
    if source.size == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, np.zeros(0, dtype=np.int64)
    kept = src[source]
    produced = kept ^ bit
    target = np.searchsorted(dst, produced)
    if target.size and (np.any(target >= dst.size) or np.any(dst[np.minimum(
            target, dst.size - 1)] != produced)):
        raise ValueError(
            "a_{}{} maps out of the target determinant space: the two spaces differ by "
            "something other than one electron in this mode".format(mode, "^dag" if dagger
                                                                     else ""))
    sign = 1 - 2 * (_below(kept, int(mode)) & 1)
    return source, target.astype(np.intp), sign.astype(np.int64)


def apply_ladder(masks_from, masks_to, mode: int, vectors, *, dagger: bool = False
                 ) -> np.ndarray:
    """``a_p |v>`` (or ``a_p^dag |v>``) for one vector or a stack of them.

    ``vectors`` is ``(ndet_from,)`` or ``(n_vec, ndet_from)``; the result has the same leading
    shape and ``len(masks_to)`` columns. The whole fermionic convention is in
    :func:`ladder_map`; this is the scatter.
    """
    vectors = np.asarray(vectors)
    flat = np.atleast_2d(vectors)
    out = np.zeros((flat.shape[0], len(masks_to)), dtype=np.result_type(vectors,
                                                                        np.complex128))
    source, target, sign = ladder_map(masks_from, masks_to, mode, dagger=dagger)
    if source.size:
        out[:, target] = flat[:, source] * sign
    return out[0] if vectors.ndim == 1 else out


def _check_array(kernel: str, name: str, arr, dtype, ndim: int) -> None:
    """Entry check for a registered kernel: exact dtype, rank and C-contiguity (B4/B5).

    ⚠ **Asserted, never inherited.** A kernel that accepts whatever NumPy hands it is a kernel
    whose compiled replacement needs transposes and dtype branches the NumPy one did not, and
    the "port" then changes its callers. Memory layout is part of the contract, so it is
    checked on entry — once per call, outside every loop (B8).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("{}: {} must be a numpy array, got {}"
                        .format(kernel, name, type(arr).__name__))
    if arr.dtype != np.dtype(dtype):
        raise TypeError("{}: {} must be {}, got {}"
                        .format(kernel, name, np.dtype(dtype), arr.dtype))
    if arr.ndim != ndim:
        raise ValueError("{}: {} must be {}-dimensional, got {}"
                         .format(kernel, name, ndim, arr.ndim))
    if not arr.flags.c_contiguous:
        raise ValueError("{}: {} must be C-contiguous".format(kernel, name))


def _check_output(kernel: str, name: str, arr, dtype, ndim: int, shape, *inputs) -> None:
    """:func:`_check_array` plus an exact shape and a non-aliasing check (B4/B5/B6)."""
    _check_array(kernel, name, arr, dtype, ndim)
    if tuple(arr.shape) != tuple(shape):
        raise ValueError("{}: {} has shape {}, expected {}"
                         .format(kernel, name, arr.shape, tuple(shape)))
    for other in inputs:
        if np.shares_memory(arr, other):
            raise ValueError("{}: {} aliases an input array".format(kernel, name))


# --- The determinant space ----------------------------------------------------------------

@dataclass
class Determinants:
    """An explicit list of determinant bitmasks over ``n_spinor`` spinors.

    ``index`` maps a mask to its position, which is what makes excitation generation usable:
    generate a candidate mask, look it up, keep it if it is in the space.
    """

    masks: np.ndarray                       # (ndet,) uint64
    n_spinor: int
    n_elec: int

    def __post_init__(self) -> None:
        if self.n_spinor > DEFAULT_MAX_SPINORS:
            raise ValueError(
                "active spaces beyond {} spinors need multi-word determinant masks, which are "
                "not implemented; got {}".format(DEFAULT_MAX_SPINORS, self.n_spinor))
        self.masks = np.ascontiguousarray(self.masks, dtype=_U64)
        counts = _popcount(self.masks)
        if self.masks.size and not np.all(counts == self.n_elec):
            raise ValueError("determinant list mixes particle numbers: found {}, expected {}"
                             .format(sorted(set(counts.tolist())), self.n_elec))
        self._index: Dict[int, int] = {int(m): i for i, m in enumerate(self.masks)}
        if len(self._index) != self.masks.size:
            raise ValueError("determinant list contains duplicates")

    def __len__(self) -> int:
        return int(self.masks.size)

    @property
    def ndet(self) -> int:
        return int(self.masks.size)

    def position(self, mask) -> int:
        """Index of ``mask`` in the space, or -1."""
        return self._index.get(int(mask), -1)

    def positions(self, masks) -> np.ndarray:
        """Vectorized :meth:`position` (``-1`` where absent)."""
        idx = self._index
        return np.fromiter((idx.get(int(m), -1) for m in np.asarray(masks, dtype=_U64)),
                           dtype=np.int64, count=len(np.asarray(masks)))

    def occupations(self) -> np.ndarray:
        """``(ndet, n_spinor)`` boolean occupation matrix (cached: every consumer wants it)."""
        cached = getattr(self, "_occ", None)
        if cached is None:
            cached = occupation_matrix(self.masks, self.n_spinor)
            self._occ = cached
        return cached

    @classmethod
    def from_occupations(cls, occ_lists: Iterable[Sequence[int]], n_spinor: int
                         ) -> "Determinants":
        masks, nelec = [], None
        for occ in occ_lists:
            m = _U64(0)
            for p in occ:
                m |= _U64(1) << _U64(int(p))
            masks.append(m)
            nelec = len(list(occ)) if nelec is None else nelec
        return cls(masks=np.array(masks, dtype=_U64), n_spinor=n_spinor, n_elec=int(nelec or 0))

    @classmethod
    def aufbau(cls, n_spinor: int, n_elec: int) -> "Determinants":
        """The single lowest-index determinant — the usual reference to excite from."""
        return cls.from_occupations([range(n_elec)], n_spinor)

    def __repr__(self) -> str:
        return "Determinants(ndet={}, n_spinor={}, n_elec={})".format(
            self.ndet, self.n_spinor, self.n_elec)


def occupation_matrix(masks: np.ndarray, n_spinor: int) -> np.ndarray:
    """``(ndet, n_spinor)`` boolean occupations from bitmasks."""
    masks = np.asarray(masks, dtype=_U64)
    bits = _U64(1) << np.arange(n_spinor, dtype=_U64)
    return (masks[:, None] & bits[None, :]) != 0


# --- Complete CAS spaces: combinatorial rank addressing -----------------------------------
#
# Ordering convention (fixed here; the excitation map, the sigma vector and every checkpoint
# depend on it)
# ---------------------------------------------------------------------------------------
# A determinant with occupied spinors c_0 < c_1 < ... < c_{k-1} has rank
#
#     rank = sum_{i=0}^{k-1} C(c_i, i+1)
#
# which is the **colexicographic** rank of the occupied set, and is a bijection onto
# ``[0, C(n,k))``. Colex is chosen over lex for one concrete reason: it orders determinants by
# **increasing bitmask value**, so ``rank`` is monotone in the mask and a ``CASSpace``'s mask
# array is exactly ``np.sort`` of every k-subset. That makes the two addressing modes of this
# module directly comparable — a :class:`Determinants` built from ``CASSpace.masks`` indexes
# determinants identically — which is what lets the sparse Hamiltonian validate the sigma
# vector element by element rather than only through an eigenvalue.
#
# Both directions are table-driven integer arithmetic over a Pascal table: ``rank`` is a loop
# over the k set bits, ``unrank`` a loop over k binary searches. Nothing here allocates per
# determinant, nothing hashes, and a C++ implementation is the same two loops.
#
# References: the combinatorial number system, D. E. Knuth, "The Art of Computer Programming"
# vol. 4A, sec. 7.2.1.3 (2011); its use for CI string addressing, P. J. Knowles, N. C. Handy,
# Chem. Phys. Lett. 111, 315 (1984), doi:10.1016/0009-2614(84)85513-X.


def binomial_table(n_spinor: int, n_elec: int) -> np.ndarray:
    """``(n_spinor + 1, n_elec + 1)`` Pascal table, ``binom[n, k] = C(n, k)``.

    ``int64`` throughout and exact: the largest entry reachable at the 64-spinor mask limit is
    ``C(64, 32) = 1.83e18``, comfortably inside ``int64``.
    """
    n_spinor, n_elec = int(n_spinor), int(n_elec)
    if n_spinor < 0 or n_elec < 0:
        raise ValueError("binomial_table needs non-negative dimensions")
    table = np.zeros((n_spinor + 1, n_elec + 1), dtype=np.int64)
    table[:, 0] = 1
    for k in range(1, n_elec + 1):
        table[k:, k] = np.cumsum(table[k - 1:-1, k - 1])
    return table


def _rank_masks(masks: np.ndarray, binom: np.ndarray, n_elec: int) -> np.ndarray:
    """Colexicographic rank of each ``uint64`` mask holding exactly ``n_elec`` bits.

    Shared by :func:`cas_rank_numpy` and :func:`excitation_map_numpy` so the address map has
    exactly one implementation; in C++ this is a static function, not a callback (B9).
    """
    work = np.array(masks, dtype=_U64, copy=True)
    out = np.zeros(work.shape, dtype=np.int64)
    for slot in range(int(n_elec)):
        low = work & (~work + _U64(1))
        out += binom[_bit_index(low), slot + 1]
        work ^= low
    return out


@kernels.kernel("cas_rank")
def cas_rank_numpy(masks: np.ndarray, binom: np.ndarray, n_elec: int,
                   out: np.ndarray) -> None:
    """Colexicographic rank of complete-CAS determinant masks (registered kernel).

    Parameters
    ----------
    masks : ``(ndet,)`` ``uint64``, C-contiguous
        Occupation bitmasks, each with exactly ``n_elec`` bits set (**not** checked: this is a
        kernel, and a popcount per determinant would cost as much as the rank itself).
    binom : ``(>= n_spinor + 1, >= n_elec + 1)`` ``int64``, C-contiguous
        Pascal table from :func:`binomial_table`.
    n_elec : int
        Number of electrons ``k``; the loop count.
    out : ``(ndet,)`` ``int64``, C-contiguous
        Caller-allocated output. Must not alias ``masks``.

    Notes
    -----
    **Portability.** Plain arrays and scalars only (B1); no hashing (B2); flat contiguous
    layout asserted on entry (B3/B4/B5); caller-provided, non-aliasing output (B6); nothing
    to block (B7); no logging, timing, resource check or raise inside the loop (B8); no
    callbacks (B9). The only globals read are the constant popcount lookup tables, which are
    compile-time constants in a compiled backend.

    **Reduction order (B10):** none. The per-determinant sum is over exactly ``n_elec`` terms
    in a fixed order that no parallelization would change, so any port must be **bitwise**
    identical.
    """
    _check_array("cas_rank", "masks", masks, np.uint64, 1)
    _check_array("cas_rank", "binom", binom, np.int64, 2)
    _check_array("cas_rank", "out", out, np.int64, 1)
    if out.shape != masks.shape:
        raise ValueError("cas_rank: out has shape {}, expected {}"
                         .format(out.shape, masks.shape))
    if np.shares_memory(out, masks) or np.shares_memory(out, binom):
        raise ValueError("cas_rank: out aliases an input array")
    out[:] = _rank_masks(masks, binom, n_elec)


@kernels.kernel("cas_unrank")
def cas_unrank_numpy(ranks: np.ndarray, binom: np.ndarray, n_spinor: int, n_elec: int,
                     out: np.ndarray) -> None:
    """Inverse of :func:`cas_rank_numpy`: rank -> occupation bitmask (registered kernel).

    Parameters
    ----------
    ranks : ``(ndet,)`` ``int64``, C-contiguous
        Ranks in ``[0, C(n_spinor, n_elec))`` (**not** range-checked, per B8).
    binom : ``(>= n_spinor + 1, >= n_elec + 1)`` ``int64``, C-contiguous
    n_spinor, n_elec : int
    out : ``(ndet,)`` ``uint64``, C-contiguous
        Caller-allocated output. Must not alias ``ranks``.

    Notes
    -----
    Greedy from the highest slot down: the orbital in slot ``i`` is the largest ``c`` with
    ``C(c, i+1) <= r``, found by binary search on a column of the Pascal table (which is
    non-decreasing in ``c``). ``np.searchsorted`` is a binary search and ports mechanically.

    **Portability:** as :func:`cas_rank_numpy`. **Reduction order (B10):** none — a port must
    be bitwise identical.
    """
    _check_array("cas_unrank", "ranks", ranks, np.int64, 1)
    _check_array("cas_unrank", "binom", binom, np.int64, 2)
    _check_array("cas_unrank", "out", out, np.uint64, 1)
    if out.shape != ranks.shape:
        raise ValueError("cas_unrank: out has shape {}, expected {}"
                         .format(out.shape, ranks.shape))
    if np.shares_memory(out, ranks) or np.shares_memory(out, binom):
        raise ValueError("cas_unrank: out aliases an input array")
    remaining = np.array(ranks, dtype=np.int64, copy=True)
    out[:] = _U64(0)
    for slot in range(int(n_elec) - 1, -1, -1):
        column = binom[:int(n_spinor), slot + 1]
        orb = np.searchsorted(column, remaining, side="right") - 1
        remaining -= column[orb]
        out |= _U64(1) << orb.astype(_U64)


@kernels.kernel("excitation_map")
def excitation_map_numpy(hole_masks: np.ndarray, binom: np.ndarray, n_spinor: int,
                         n_elec: int, block: int,
                         h2d_det: np.ndarray, h2d_orb: np.ndarray, h2d_sign: np.ndarray,
                         d2h_hole: np.ndarray, d2h_orb: np.ndarray, d2h_sign: np.ndarray
                         ) -> None:
    """Build the two rectangular single-excitation tables of a complete CAS space.

    The map is what every later kernel is written against, so its layout is a **contract**,
    fixed here (B3/B4/B5). A hole string carries ``k-1`` electrons over ``n`` spinors and
    therefore has **exactly** ``n-k+1`` empty orbitals — never a variable number — so both
    tables are dense 2-D arrays with no offsets and nothing ragged:

    ==============  ======================  ====================================
    table           shape                   entry ``[x, slot]``
    ==============  ======================  ====================================
    ``h2d_*``       ``(C(n,k-1), n-k+1)``   determinant ``J``, created orbital
                                            ``q`` (ascending in ``slot``), sign
    ``d2h_*``       ``(C(n,k), k)``         hole string ``h``, annihilated
                                            orbital ``p`` (ascending), sign
    ==============  ======================  ====================================

    They hold the same ``C(n,k)*k = C(n,k-1)*(n-k+1)`` incidences transposed (that identity is
    asserted by the caller), so **only one construction has to be right**: ``d2h_*`` is derived
    from ``h2d_*`` here, not built independently.

    Sign and slot are the *same* quantity, which is why the transposition is exact and free.
    Under the ascending-order convention of this module,
    ``a_q^dag |h> = (-1)^b |J>`` and ``a_p |J> = (-1)^b |h>`` with
    ``b = (occupied orbitals of h below the orbital)`` — and ``b`` is also the position of that
    orbital among ``J``'s occupied orbitals in ascending order, i.e. its ``d2h_*`` slot.

    Parameters
    ----------
    hole_masks : ``(C(n,k-1),)`` ``uint64``, C-contiguous
        Masks of the ``(k-1)``-electron strings, in rank order.
    binom : ``int64`` Pascal table, C-contiguous.
    n_spinor, n_elec, block : int
        ``block`` is the number of hole strings vectorized at once. ⚠ It is a **parameter**,
        never read from a config file or a resource budget (B7); the caller sizes it once,
        outside this call. A compiled backend has no temporaries and may ignore it.
    h2d_det, d2h_hole : ``int32`` output tables, C-contiguous.
    h2d_orb, d2h_orb, h2d_sign, d2h_sign : ``int8`` output tables, C-contiguous.

    Notes
    -----
    **Portability:** B1-B9 as :func:`cas_rank_numpy`; the six outputs are caller-allocated and
    must not alias the inputs (B6). **Reduction order (B10):** none — every output element is
    written exactly once, by construction, so a threaded port over hole-string blocks changes
    nothing and must be **bitwise** identical.
    """
    n, k = int(n_spinor), int(n_elec)
    m = n - k + 1
    _check_array("excitation_map", "hole_masks", hole_masks, np.uint64, 1)
    _check_array("excitation_map", "binom", binom, np.int64, 2)
    hole_shape = (hole_masks.size, m)
    det_shape = (int(binom[n, k]), k)
    _check_output("excitation_map", "h2d_det", h2d_det, np.int32, 2, hole_shape,
                  hole_masks, binom)
    _check_output("excitation_map", "h2d_orb", h2d_orb, np.int8, 2, hole_shape,
                  hole_masks, binom)
    _check_output("excitation_map", "h2d_sign", h2d_sign, np.int8, 2, hole_shape,
                  hole_masks, binom)
    _check_output("excitation_map", "d2h_hole", d2h_hole, np.int32, 2, det_shape,
                  hole_masks, binom)
    _check_output("excitation_map", "d2h_orb", d2h_orb, np.int8, 2, det_shape,
                  hole_masks, binom)
    _check_output("excitation_map", "d2h_sign", d2h_sign, np.int8, 2, det_shape,
                  hole_masks, binom)
    if block < 1:
        raise ValueError("excitation_map: block must be positive, got {}".format(block))

    bits = _U64(1) << np.arange(n, dtype=_U64)
    for lo in range(0, hole_masks.size, int(block)):
        hi = min(lo + int(block), hole_masks.size)
        holes = hole_masks[lo:hi]
        empty = (holes[:, None] & bits[None, :]) == 0                 # (B, n)
        # Exactly m empties per row, and np.nonzero walks rows in order, so the reshape puts
        # the created orbitals in ascending order within each row -- which is the layout the
        # table promises.
        orb = np.nonzero(empty)[1].reshape(hi - lo, m).astype(np.int64)
        dets = holes[:, None] | (_U64(1) << orb.astype(_U64))         # (B, m)
        rank = _rank_masks(dets.ravel(), binom, k).reshape(hi - lo, m)
        slot = _popcount(holes[:, None] & _BELOW[orb])                # (B, m) = sign exponent
        sign = (1 - 2 * (slot & 1)).astype(np.int8)

        h2d_det[lo:hi] = rank
        h2d_orb[lo:hi] = orb
        h2d_sign[lo:hi] = sign
        hole_index = np.arange(lo, hi, dtype=np.int64)[:, None]
        d2h_hole[rank, slot] = hole_index
        d2h_orb[rank, slot] = orb
        d2h_sign[rank, slot] = sign


# --- Sizing for complete CAS spaces ------------------------------------------------

#: Bytes per excitation-map incidence: ``int32`` index + ``int8`` orbital + ``int8`` sign, in
#: each of the two tables. The two hold the same ``C(n,k)*k`` incidences transposed.
BYTES_PER_INCIDENCE = 2 * (4 + 1 + 1)


def cas_dimension(n_spinor: int, n_elec: int) -> int:
    """``C(n_spinor, n_elec)`` — the determinant count of a complete CAS space."""
    if not 0 <= n_elec <= n_spinor:
        raise ValueError("a CAS space needs 0 <= n_elec <= n_spinor, got {} of {}"
                         .format(n_elec, n_spinor))
    return int(binomial_table(n_spinor, n_elec)[n_spinor, n_elec])


def excitation_map_gb(n_spinor: int, n_elec: int) -> float:
    """Size [GB] of the two rectangular excitation tables (exact sizing function).

    Exact and unpadded: ``2 * C(n,k) * k`` stored incidences at
    :data:`BYTES_PER_INCIDENCE` bytes each. 8.2 MB at 16 spinors half filled, 22 MB at 20,
    372 MB at 24 — small against the sigma workspace of ``ci/sigma.py``, which is what
    actually binds.
    """
    incidences = float(cas_dimension(n_spinor, n_elec)) * float(n_elec)
    return incidences * BYTES_PER_INCIDENCE / res.BYTES_PER_GB


def cas_vector_gb(n_spinor: int, n_elec: int, n_states: int = 1) -> float:
    """Size [GB] of ``n_states`` complex CI vectors over a complete CAS space."""
    return res.array_gb((cas_dimension(n_spinor, n_elec), n_states), np.complex128)


class CASSpace:
    """A complete active space of ``C(n_spinor, n_elec)`` determinants, rank-addressed.

    The Python-side owner of the addressing kernels: it holds the Pascal table, the
    determinant masks, the two excitation tables, and the resource reservations for them. All
    policy, logging and budgeting lives here; the kernels themselves see only arrays (B8).

    Attributes
    ----------
    masks : ``(ndet,)`` ``uint64``
        Determinant bitmasks **in rank order**, hence in ascending mask order.
    h2d_det, h2d_orb, h2d_sign : ``(n_hole, n_empty)``
    d2h_hole, d2h_orb, d2h_sign : ``(ndet, n_elec)``
        The excitation map of :func:`excitation_map_numpy`; see its docstring for the layout
        contract, which every later kernel is written against.
    """

    def __init__(self, n_spinor: int, n_elec: int, *, backend: Optional[str] = None,
                 build_map: bool = True) -> None:
        n_spinor, n_elec = int(n_spinor), int(n_elec)
        if n_spinor > DEFAULT_MAX_SPINORS:
            raise ValueError(
                "active spaces beyond {} spinors need multi-word determinant masks, which are "
                "not implemented; got {}".format(DEFAULT_MAX_SPINORS, n_spinor))
        if not 1 <= n_elec <= n_spinor:
            raise ValueError("a CAS space needs 1 <= n_elec <= n_spinor, got {} of {}"
                             .format(n_elec, n_spinor))
        self.n_spinor = n_spinor
        self.n_elec = n_elec
        self.backend = backend
        self.binom = binomial_table(n_spinor, n_elec)
        self.ndet = int(self.binom[n_spinor, n_elec])
        self.n_hole = int(self.binom[n_spinor, n_elec - 1])
        self.n_empty = n_spinor - n_elec + 1
        if self.ndet > np.iinfo(np.int32).max:
            raise ValueError(
                "{} determinants exceeds the int32 indexing of the excitation map; a "
                "conventional CI is far past its ceiling here — use DMRG"
                .format(self.ndet))

        res.reserve("CAS determinant masks ({} spinors, {} electrons)".format(n_spinor, n_elec),
                    res.array_gb((self.ndet,), np.uint64),
                    note="{} determinants".format(self.ndet))
        self.masks = np.empty(self.ndet, dtype=_U64)
        resolve = kernels.resolve("cas_unrank", backend)
        with timer("CAS determinant addressing"):
            resolve(np.arange(self.ndet, dtype=np.int64), self.binom, n_spinor, n_elec,
                    self.masks)
        self._hole_masks: Optional[np.ndarray] = None
        self._occ: Optional[np.ndarray] = None
        self._kramers: Optional["KramersMap"] = None
        self.h2d_det = self.h2d_orb = self.h2d_sign = None
        self.d2h_hole = self.d2h_orb = self.d2h_sign = None
        if build_map:
            self.build_excitation_map()

    # -- addressing --------------------------------------------------------------------
    def rank(self, masks) -> np.ndarray:
        """Rank of each mask (vectorized, ``O(k)`` per determinant, no lookup table)."""
        masks = np.ascontiguousarray(masks, dtype=_U64)
        out = np.empty(masks.shape, dtype=np.int64)
        kernels.resolve("cas_rank", self.backend)(masks, self.binom, self.n_elec, out)
        return out

    def unrank(self, ranks) -> np.ndarray:
        """Mask of each rank (vectorized inverse of :meth:`rank`)."""
        ranks = np.ascontiguousarray(ranks, dtype=np.int64)
        out = np.empty(ranks.shape, dtype=_U64)
        kernels.resolve("cas_unrank", self.backend)(ranks, self.binom, self.n_spinor,
                                                    self.n_elec, out)
        return out

    def hole_masks(self) -> np.ndarray:
        """Masks of the ``(k-1)``-electron hole strings, in rank order."""
        if self._hole_masks is None:
            binom = binomial_table(self.n_spinor, max(self.n_elec - 1, 0))
            out = np.empty(self.n_hole, dtype=_U64)
            kernels.resolve("cas_unrank", self.backend)(
                np.arange(self.n_hole, dtype=np.int64), binom, self.n_spinor,
                self.n_elec - 1, out)
            self._hole_masks = out
        return self._hole_masks

    def occupations(self) -> np.ndarray:
        """``(ndet, n_spinor)`` boolean occupation matrix (cached)."""
        if self._occ is None:
            self._occ = occupation_matrix(self.masks, self.n_spinor)
        return self._occ

    def kramers(self) -> "KramersMap":
        """The time-reversal map over this space (built once, cached).

        A complete CAS space over a Kramers-paired spinor set is closed under time reversal by
        construction — the pair swap permutes the ``k``-subsets among themselves — which is
        exactly what makes this a permutation rather than a projection.
        """
        if self._kramers is None:
            self._kramers = KramersMap(self)
        return self._kramers

    def determinants(self) -> "Determinants":
        """The same space as an arbitrary-list :class:`Determinants`.

        ⚠ **A bridge for validation, not a production path.** It builds the Python ``dict``
        that :class:`CASSpace` exists to avoid, so it costs hundreds of MB at 10^6
        determinants. Its value is that the two representations index determinants
        identically (colex rank is ascending mask order), so :func:`hamiltonian_matrix` and
        :func:`rdm12` can be compared element by element against the string-driven path.
        """
        return Determinants(masks=self.masks, n_spinor=self.n_spinor, n_elec=self.n_elec)

    # -- the excitation map ------------------------------------------------------------
    @property
    def has_excitation_map(self) -> bool:
        return self.d2h_hole is not None

    def build_excitation_map(self, *, block: Optional[int] = None) -> None:
        """Build the two rectangular excitation tables, with their memory reservation."""
        if self.has_excitation_map:
            return
        n, k = self.n_spinor, self.n_elec
        res.reserve("CAS single-excitation map ({} spinors, {} electrons)".format(n, k),
                    excitation_map_gb(n, k),
                    note="{} x {} + {} x {} incidences".format(
                        self.n_hole, self.n_empty, self.ndet, k),
                    advice=["reduce the active space: the map grows as C(n,k)*k",
                            "above the conventional-CI ceiling, use DMRG"])
        assert self.ndet * k == self.n_hole * self.n_empty, "incidence identity violated"

        self.h2d_det = np.empty((self.n_hole, self.n_empty), dtype=np.int32)
        self.h2d_orb = np.empty((self.n_hole, self.n_empty), dtype=np.int8)
        self.h2d_sign = np.empty((self.n_hole, self.n_empty), dtype=np.int8)
        self.d2h_hole = np.empty((self.ndet, k), dtype=np.int32)
        self.d2h_orb = np.full((self.ndet, k), -1, dtype=np.int8)
        self.d2h_sign = np.empty((self.ndet, k), dtype=np.int8)

        if block is None:
            # One number, once, outside the kernel (B7). The per-block temporaries are
            # a handful of (block, n_empty) int64/uint64 arrays; 64 B per entry is a generous
            # accounting of them.
            per_row = 64.0 * self.n_empty / res.BYTES_PER_GB
            block = int(max(1, min(self.n_hole, res.transient_gb() / max(per_row, 1e-12))))
        with timer("CAS excitation map"):
            kernels.resolve("excitation_map", self.backend)(
                self.hole_masks(), self.binom, n, k, int(block),
                self.h2d_det, self.h2d_orb, self.h2d_sign,
                self.d2h_hole, self.d2h_orb, self.d2h_sign)

        # The transposition writes every (determinant, slot) entry exactly once by
        # construction. Checking it costs one pass over an int8 array and is the only cheap
        # way to fail loudly if that ever stops being true -- a partially written map is
        # plausible, Hermitian and wrong everywhere downstream.
        if np.any(self.d2h_orb < 0):
            raise RuntimeError("excitation map transposition left {} entries unwritten"
                               .format(int(np.count_nonzero(self.d2h_orb < 0))))
        log.debug("CAS space %d spinors / %d electrons: %d determinants, %d incidences",
                  n, k, self.ndet, self.ndet * k)

    def excitation_arrays(self) -> Tuple[np.ndarray, ...]:
        """The six map arrays in the order every registered kernel takes them."""
        if not self.has_excitation_map:
            self.build_excitation_map()
        return (self.h2d_det, self.h2d_orb, self.h2d_sign,
                self.d2h_hole, self.d2h_orb, self.d2h_sign)

    def __len__(self) -> int:
        return self.ndet

    def __repr__(self) -> str:
        return "CASSpace(n_spinor={}, n_elec={}, ndet={})".format(
            self.n_spinor, self.n_elec, self.ndet)


# --- Connections --------------------------------------------------------------------------

@dataclass
class Connections:
    """Determinant pairs ``I < J`` that interact, split by excitation rank.

    Stored as flat arrays so every consumer (Hamiltonian, RDMs) is a vectorized pass. The
    ``phase`` arrays already contain the fermionic sign of the excitation ``J -> I``: the
    orbitals ``i`` (and ``j``) are occupied in ``J`` and empty in ``I``, ``a`` (and ``b``) the
    reverse, so a matrix element reads ``<I|H|J> = phase * (...)``.
    """

    single_i: np.ndarray                    # (n1,) int64 determinant indices I
    single_j: np.ndarray                    # (n1,) int64 determinant indices J
    single_from: np.ndarray                 # (n1,) annihilated in J
    single_to: np.ndarray                   # (n1,) created in I
    single_phase: np.ndarray                # (n1,) float64, +-1
    double_i: np.ndarray
    double_j: np.ndarray
    double_from: np.ndarray                 # (n2, 2) annihilated in J, ascending
    double_to: np.ndarray                   # (n2, 2) created in I, ascending
    double_phase: np.ndarray

    @property
    def n_single(self) -> int:
        return int(self.single_i.size)

    @property
    def n_double(self) -> int:
        return int(self.double_i.size)

    def __repr__(self) -> str:
        return "Connections(single={}, double={})".format(self.n_single, self.n_double)


# --- Sizing ------------------------------------------------------------------------
# Exact byte counts for the arrays below, so a determinant space that cannot fit is refused
# rather than discovered. They are stated per connection because that — not the determinant
# count — is what actually grows: a space of N determinants has O(N^2) interacting pairs in
# the worst case and a few percent of that in practice, which is precisely why it has to be
# checked as it is counted rather than bounded in advance (see :func:`connections`).

#: Bytes per stored single excitation: four int64 index arrays plus a float64 phase.
BYTES_PER_SINGLE = 4 * 8 + 8
#: Bytes per stored double excitation: two int64 indices, two (2,) int64 pairs, one phase.
BYTES_PER_DOUBLE = 2 * 8 + 2 * 2 * 8 + 8


def determinant_memory_gb(ndet: int, n_states: int = 1) -> float:
    """Size [GB] of a determinant space and its CI vectors (exact sizing function)."""
    return (res.array_gb((ndet,), np.uint64)
            + res.array_gb((ndet, n_states), np.complex128))


def connection_memory_gb(n_single: int, n_double: int) -> float:
    """Size [GB] of a :class:`Connections` object with the given counts."""
    return (n_single * BYTES_PER_SINGLE + n_double * BYTES_PER_DOUBLE) / res.BYTES_PER_GB


def hamiltonian_memory_gb(ndet: int, n_single: int, n_double: int) -> float:
    """Peak size [GB] of the sparse CI Hamiltonian (exact sizing function).

    ``nnz = ndet + 2 (n_single + n_double)`` after Hermitian completion. The peak counts the
    COO triplet arrays (two int64 indices and a complex value, 32 B) *and* the CSR that is
    built from them (complex data plus an int32 column index, 20 B), because both are live
    during the conversion.
    """
    nnz = float(ndet) + 2.0 * (float(n_single) + float(n_double))
    return nnz * (32.0 + 20.0) / res.BYTES_PER_GB


def _phase_single(det_j: np.ndarray, i: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Sign of ``a_a^dag a_i |J>``: annihilate ``i`` from ``J``, then create ``a``."""
    s = _below(det_j, i)
    det1 = det_j ^ _bit(i)
    s = s + _below(det1, a)
    return np.where(s % 2 == 0, 1.0, -1.0)


def _phase_double(det_j: np.ndarray, i: np.ndarray, j: np.ndarray,
                  a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign of ``a_a^dag a_b^dag a_j a_i |J>`` with ``i < j`` and ``a < b``.

    Applied strictly in that operator order so the result matches the two-electron matrix
    element ``(ai|bj) - (aj|bi)`` derived from the Hamiltonian's normal ordering.
    """
    s = _below(det_j, i)
    d1 = det_j ^ _bit(i)
    s = s + _below(d1, j)
    d2 = d1 ^ _bit(j)
    s = s + _below(d2, b)
    d3 = d2 | _bit(b)
    s = s + _below(d3, a)
    return np.where(s % 2 == 0, 1.0, -1.0)


@kernels.kernel("connections_scan")
def connections_scan_numpy(masks: np.ndarray, row_start: int, row_stop: int,
                           s_i: np.ndarray, s_j: np.ndarray, s_from: np.ndarray,
                           s_to: np.ndarray, s_phase: np.ndarray,
                           d_i: np.ndarray, d_j: np.ndarray, d_from: np.ndarray,
                           d_to: np.ndarray, d_phase: np.ndarray, n_threads: int):
    """Scan rows ``[row_start, row_stop)`` of the pairwise XOR/popcount search.

    **The registered kernel of the connection search** (the first port
    candidate). One call scans one row block of the ``O(N^2)`` search — blocking,
    budgeting, timing and concatenation belong to :func:`connections`, the orchestration
    wrapper, whose per-block call is the outermost (B9) callback point.

    Contract (B1-B10):

    * ``masks`` is the full ``(n,)`` ``uint64`` determinant list; pairs are emitted for
      ``row_start <= I < row_stop``, ``I < J < n``, in row-major ``(I, J)`` order, split
      by excitation rank into the single- and double-excitation buffers;
    * output buffers are caller-provided flat arrays (B6), all singles sharing one
      capacity and all doubles another; ``d_from``/``d_to`` are ``(capacity, 2)`` with
      the two orbitals ascending; phases are the fermionic signs of
      :func:`_phase_single`/:func:`_phase_double`, computed here so a compiled backend
      owns the whole per-pair loop;
    * the return value is the pair ``(n1, n2)`` actually found. ⚠ **A capacity miss is a
      protocol, not an error** (B8): when either count exceeds its buffer's capacity,
      nothing is written and the counts go back so the wrapper can reallocate and re-call.
      A caller must therefore never read the buffers past the returned counts, and must
      check them against the capacities it granted.
    * ``n_threads`` is the explicit thread budget (B7 applied to threads); this NumPy
      implementation is vectorized, not threaded, and ignores it.

    Reduction order (B10): none — the outputs are integer index arrays and +-1 phases,
    pure per-pair functions with no floating-point reduction anywhere. A threaded backend
    that emits per-row-chunk buffers concatenated in row order is bitwise identical.
    """
    if masks.dtype != np.uint64:
        raise TypeError("masks must be uint64, got {}".format(masks.dtype))
    if s_i.dtype != np.int64 or s_j.dtype != np.int64 or s_from.dtype != np.int64 \
            or s_to.dtype != np.int64 or d_i.dtype != np.int64 or d_j.dtype != np.int64 \
            or d_from.dtype != np.int64 or d_to.dtype != np.int64:
        raise TypeError("index buffers must be int64")
    if s_phase.dtype != np.float64 or d_phase.dtype != np.float64:
        raise TypeError("phase buffers must be float64")
    if not (masks.flags.c_contiguous and s_i.flags.c_contiguous and s_j.flags.c_contiguous
            and s_from.flags.c_contiguous and s_to.flags.c_contiguous
            and s_phase.flags.c_contiguous and d_i.flags.c_contiguous
            and d_j.flags.c_contiguous and d_from.flags.c_contiguous
            and d_to.flags.c_contiguous and d_phase.flags.c_contiguous):
        raise ValueError("mask and output buffers must be C-contiguous")
    if np.shares_memory(s_i, masks) or np.shares_memory(s_j, masks) \
            or np.shares_memory(s_from, masks) or np.shares_memory(s_to, masks) \
            or np.shares_memory(s_phase, masks) or np.shares_memory(d_i, masks) \
            or np.shares_memory(d_j, masks) or np.shares_memory(d_from, masks) \
            or np.shares_memory(d_to, masks) or np.shares_memory(d_phase, masks):
        raise ValueError("an output buffer may not alias the mask array")
    if d_from.ndim != 2 or d_from.shape[1] != 2 or d_to.ndim != 2 or d_to.shape[1] != 2:
        raise ValueError("d_from and d_to must have shape (capacity, 2)")
    if not (0 <= row_start <= row_stop <= masks.size):
        raise ValueError("row range must satisfy 0 <= row_start <= row_stop <= n")
    if n_threads < 1:
        raise ValueError("the thread count must be a positive integer")

    rows = masks[row_start:row_stop]
    diff = rows[:, None] ^ masks[None, :]
    rank2 = _popcount(diff)                           # 2 * excitation rank
    # upper triangle only (I < J), and interacting ranks only
    ri, rj = np.nonzero((rank2 <= 4) & (rank2 > 0))
    ri_abs = ri + row_start
    keep = ri_abs < rj
    ri, rj, ri_abs = ri[keep], rj[keep], ri_abs[keep]
    if ri.size == 0:
        return 0, 0
    r2 = rank2[ri, rj]
    dif = diff[ri, rj]
    det_i, det_j = masks[ri_abs], masks[rj]

    m1 = r2 == 2
    m2 = r2 == 4
    n1 = int(np.count_nonzero(m1))
    n2 = int(np.count_nonzero(m2))
    if n1 > s_i.shape[0] or n2 > d_i.shape[0]:
        return n1, n2                                 # capacity miss: counts only, no write

    if n1:
        d = dif[m1]
        to = _bit_index(d & det_i[m1])                # created in I
        fr = _bit_index(d & det_j[m1])                # annihilated in J
        s_i[:n1] = ri_abs[m1]
        s_j[:n1] = rj[m1]
        s_from[:n1] = fr
        s_to[:n1] = to
        s_phase[:n1] = _phase_single(det_j[m1], fr, to)
    if n2:
        d = dif[m2]
        bits_i = d & det_i[m2]
        bits_j = d & det_j[m2]
        lo_i = _lowest_bit(bits_i)
        lo_j = _lowest_bit(bits_j)
        d_i[:n2] = ri_abs[m2]
        d_j[:n2] = rj[m2]
        d_from[:n2, 0] = _bit_index(lo_j)
        d_from[:n2, 1] = _bit_index(bits_j ^ lo_j)
        d_to[:n2, 0] = _bit_index(lo_i)
        d_to[:n2, 1] = _bit_index(bits_i ^ lo_i)
        d_phase[:n2] = _phase_double(det_j[m2], d_from[:n2, 0], d_from[:n2, 1],
                                     d_to[:n2, 0], d_to[:n2, 1])
    return n1, n2


def connections(dets: Determinants, *, block: int = 512,
                row_limit: Optional[int] = None, n_threads: Optional[int] = None,
                backend: Optional[str] = None) -> Connections:
    """Find all interacting determinant pairs ``I < J`` (excitation rank 1 or 2).

    Blocked pairwise search: ``popcount(mask_I XOR mask_J) == 2*rank``. The ``O(N^2)``
    scan is the cost of supporting an arbitrary determinant space; ``block`` bounds the
    NumPy backend's temporary to ``block * ndet`` 64-bit words, so memory is flat in ``N``.

    ``row_limit`` restricts the scan to pairs whose *lower* index is below it, turning the
    quadratic search into a rectangular ``row_limit x ndet`` one. That is what makes
    perturbative selection affordable: put a small generator set first in the list and the
    cost becomes linear in the (large) candidate count instead of quadratic.

    ⚠ **Orchestration, and stays orchestration** (the profiling rule cuts both ways: nobody should port
    this wrapper by mistake). The scan itself is the registered kernel
    :func:`connections_scan_numpy` — resolved through :mod:`kuiva.ci.kernels`, so the
    compiled backend replaces it with no change here — and this wrapper keeps everything a
    kernel may not contain: the timer, the ``res.require`` cadence, the logging, the
    block loop (its per-block call is the B9 outermost callback point) and the
    concatenation. ``n_threads`` is resolved once by entering
    :func:`kuiva.util.threads.kernel_region` — this scan is the kernel-bound stage that
    policy is for — and passed as the kernel's explicit thread argument; the NumPy
    backend ignores it, the compiled one is bitwise at any value.
    """
    masks = dets.masks
    n = masks.size
    scan = kernels.resolve("connections_scan", backend)
    s_parts, d_parts = [], []

    n_rows = n if row_limit is None else min(int(row_limit), n)
    n1 = n2 = 0
    # Reused, grown-on-demand output buffers for the kernel. The pair count of a block is
    # unknowable in advance (see the budgeting note below); a capacity miss returns the needed
    # counts and the block is re-called once with exact room — rare after the first block,
    # since the buffers only ever grow.
    cap1 = cap2 = 1 << 12
    s_buf = [np.empty(cap1, dtype=np.int64) for _ in range(4)] \
        + [np.empty(cap1, dtype=np.float64)]
    d_buf = [np.empty(cap2, dtype=np.int64) for _ in range(2)] \
        + [np.empty((cap2, 2), dtype=np.int64) for _ in range(2)] \
        + [np.empty(cap2, dtype=np.float64)]
    # A kernel-bound stage: no BLAS runs inside the scan at all, so the thread budget
    # belongs to the compiled kernel and MKL is clamped for the duration. Entered once
    # around the whole block loop, never per block (B8), and an explicit ``n_threads``
    # still wins over the budget.
    with threads.kernel_region(n_threads) as nt, timer("determinant connections"):
        for start in range(0, n_rows, block):
            stop = min(start + block, n_rows)
            k1, k2 = scan(masks, start, stop, s_buf[0], s_buf[1], s_buf[2], s_buf[3],
                          s_buf[4], d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4], nt)
            if k1 > cap1 or k2 > cap2:
                if k1 > cap1:
                    cap1 = int(k1)
                    s_buf = [np.empty(cap1, dtype=np.int64) for _ in range(4)] \
                        + [np.empty(cap1, dtype=np.float64)]
                if k2 > cap2:
                    cap2 = int(k2)
                    d_buf = [np.empty(cap2, dtype=np.int64) for _ in range(2)] \
                        + [np.empty((cap2, 2), dtype=np.int64) for _ in range(2)] \
                        + [np.empty(cap2, dtype=np.float64)]
                k1, k2 = scan(masks, start, stop, s_buf[0], s_buf[1], s_buf[2], s_buf[3],
                              s_buf[4], d_buf[0], d_buf[1], d_buf[2], d_buf[3], d_buf[4],
                              nt)
            if k1:
                s_parts.append(tuple(b[:k1].copy() for b in s_buf))
            if k2:
                d_parts.append(tuple(b[:k2].copy() for b in d_buf))

            # The number of interacting pairs is not knowable before the search — it is
            # bounded by N^2/2 and is a few percent of that in practice — so it is checked as
            # it is counted, once per block (a dozen calls for a typical space, none of them
            # inside the kernel's work). The factor 2 covers the final concatenation, which
            # holds the per-block parts and the joined array at the same time.
            n1 += int(k1)
            n2 += int(k2)
            res.require("determinant connections", 2.0 * connection_memory_gb(n1, n2),
                        note="{} single + {} double excitations found after {} of {} rows"
                             .format(n1, n2, stop, n_rows),
                        advice=["reduce the determinant count (max_determinants= in "
                                "mcscf.preopt): the pair count grows as its square",
                                "use a smaller active space, or DMRG above the "
                                "conventional-CI ceiling"])

    def _cat(parts, index, shape_tail=(), dtype=np.int64):
        chosen = [p[index] for p in parts]
        if chosen:
            return np.concatenate(chosen)
        return np.zeros((0,) + shape_tail, dtype=dtype)

    conn = Connections(_cat(s_parts, 0), _cat(s_parts, 1), _cat(s_parts, 2),
                       _cat(s_parts, 3), _cat(s_parts, 4, dtype=np.float64),
                       _cat(d_parts, 0), _cat(d_parts, 1), _cat(d_parts, 2, (2,)),
                       _cat(d_parts, 3, (2,)), _cat(d_parts, 4, dtype=np.float64))
    log.debug("connections over %d determinants: %d single, %d double",
              n, conn.n_single, conn.n_double)
    return conn


# --- Matrix elements ------------------------------------------------------------------------

def diagonal_energies(dets, h: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """``<I|H|I>`` for every determinant: ``sum_k h_kk + 1/2 sum_kl [(kk|ll) - (kl|lk)]``.

    ``dets`` is anything carrying an ``occupations()`` matrix — a :class:`Determinants` or a
    :class:`CASSpace`. ⚠ Deliberately *not* annotated to the former: the full-CI Davidson
    preconditioner needs this over a complete CAS space, and routing it through
    :meth:`CASSpace.determinants` would build the very hash table :class:`CASSpace` exists to
    avoid (hundreds of MB at 10^6 determinants).
    """
    occ = dets.occupations().astype(np.float64)
    w = np.real(np.einsum("kkll->kl", eri) - np.einsum("kllk->kl", eri))
    return occ @ np.real(np.diag(h)) + 0.5 * np.einsum("nk,nk->n", occ @ w, occ)


def hamiltonian_matrix(dets: Determinants, h: np.ndarray, eri: np.ndarray,
                       conn: Optional[Connections] = None):
    """Assemble the CI Hamiltonian as a sparse Hermitian matrix over the determinant space.

    Returned as ``scipy.sparse.csr_matrix`` of ``complex128``. Building it once and reusing it
    across eigensolver iterations is the right trade at this scale: the connection search is
    the expensive part and it does not depend on the vector.
    """
    from scipy.sparse import coo_matrix

    conn = connections(dets) if conn is None else conn
    n = dets.ndet
    res.require("sparse CI Hamiltonian", hamiltonian_memory_gb(n, conn.n_single, conn.n_double),
                note="{} determinants, {} nonzeros".format(
                    n, n + 2 * (conn.n_single + conn.n_double)),
                advice=["reduce the determinant count (max_determinants= in mcscf.preopt)",
                        "a matrix-free sigma-vector CI (ci.sigma) never forms "
                        "this matrix; the cheap CI of 7.3 deliberately does"])
    diag = diagonal_energies(dets, h, eri).astype(np.complex128)
    rows = [np.arange(n)]
    cols = [np.arange(n)]
    vals = [diag]

    with timer("CI Hamiltonian assembly"):
        if conn.n_single:
            # (ai|kk) - (ak|ki) precomputed once as an n^3 array. Indexing the n^4 ERI per
            # connection instead would materialize (n_single, n, n) — gigabytes at 1e5 pairs.
            g3 = np.einsum("aikk->aik", eri) - np.einsum("akki->aik", eri)
            occ = dets.occupations()
            a, i = conn.single_to, conn.single_from
            v = np.empty(conn.n_single, dtype=np.complex128)
            for lo in range(0, conn.n_single, _CONN_BATCH):
                hi = min(lo + _CONN_BATCH, conn.n_single)
                sl = slice(lo, hi)
                common = (occ[conn.single_i[sl]] & occ[conn.single_j[sl]]).astype(np.float64)
                v[sl] = conn.single_phase[sl] * (
                    h[a[sl], i[sl]] + np.einsum("xk,xk->x", common, g3[a[sl], i[sl], :]))
            rows.append(conn.single_i); cols.append(conn.single_j); vals.append(v)
        if conn.n_double:
            i, j = conn.double_from[:, 0], conn.double_from[:, 1]
            a, b = conn.double_to[:, 0], conn.double_to[:, 1]
            v = conn.double_phase * (eri[a, i, b, j] - eri[a, j, b, i])
            rows.append(conn.double_i); cols.append(conn.double_j); vals.append(v)

    r = np.concatenate(rows); c = np.concatenate(cols); x = np.concatenate(vals)
    # Hermitian completion: the search returned only I < J.
    off = r != c
    r_full = np.concatenate([r, c[off]])
    c_full = np.concatenate([c, r[off]])
    x_full = np.concatenate([x, np.conj(x[off])])
    return coo_matrix((x_full, (r_full, c_full)), shape=(n, n)).tocsr()


def single_excitation_operator(dets: Determinants, p: int, q: int,
                               conn: Optional[Connections] = None):
    """Sparse matrix of ``a_p^dag a_q`` over the determinant space (diagnostics and tests)."""
    from scipy.sparse import coo_matrix

    conn = connections(dets) if conn is None else conn
    n = dets.ndet
    rows, cols, vals = [], [], []
    if p == q:
        occ = dets.occupations()[:, p]
        idx = np.nonzero(occ)[0]
        rows.append(idx); cols.append(idx); vals.append(np.ones(idx.size))
    m = (conn.single_to == p) & (conn.single_from == q)
    if np.any(m):
        rows.append(conn.single_i[m]); cols.append(conn.single_j[m])
        vals.append(conn.single_phase[m])
    m = (conn.single_to == q) & (conn.single_from == p)
    if np.any(m):                                   # the reverse direction of the same pair
        rows.append(conn.single_j[m]); cols.append(conn.single_i[m])
        vals.append(conn.single_phase[m])
    if not rows:
        return coo_matrix((n, n), dtype=np.complex128).tocsr()
    return coo_matrix((np.concatenate(vals).astype(np.complex128),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()


# --- Reduced density matrices ----------------------------------------------------------------

def _scatter(target: np.ndarray, flat_index: np.ndarray, values: np.ndarray) -> None:
    """Accumulate ``values`` into ``target.ravel()`` at ``flat_index`` (complex bincount).

    ``np.add.at`` is correct but roughly an order of magnitude slower than two real
    ``bincount`` calls, and this runs over every connected determinant pair.
    """
    size = target.size
    target.ravel().real += np.bincount(flat_index, weights=values.real, minlength=size)
    target.ravel().imag += np.bincount(flat_index, weights=values.imag, minlength=size)


def rdm12(dets: Determinants, civecs: np.ndarray, weights: Optional[np.ndarray] = None,
          conn: Optional[Connections] = None, *, with_2rdm: bool = True
          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """State-averaged 1- and 2-particle density matrices over the determinant space.

    Parameters
    ----------
    civecs : ndarray ``(ndet,)`` or ``(ndet, nstate)``
        Normalized CI vectors.
    weights : ndarray ``(nstate,)``, optional
        State-averaging weights (default: uniform). averaging must be imposed
        explicitly; this is where it happens for the cheap CI.

    Returns
    -------
    ``(gamma, Gamma)`` with ``gamma_pq = <a_p^dag a_q>`` and
    ``Gamma_pqrs = <a_p^dag a_r^dag a_s a_q>`` (``None`` if ``with_2rdm`` is False).

    Notes
    -----
    Every contribution is derived from the same Slater-Condon algebra as the Hamiltonian, by
    reading off which ``Gamma`` element each energy term contracts with. That is why the two
    cannot drift apart: a phase error would break the energy and the RDM identically, and the
    trace condition ``sum_r Gamma_pqrr = (N-1) gamma_pq`` (asserted in the tests) would fail.
    The 2-RDM is ``n^4`` complex — 50 MB at 42 spinors, 400 MB at 80 — so ``with_2rdm=False``
    is offered for the entanglement-only path, which needs only ``gamma`` and the diagonal
    ``<n_p n_q>`` (:func:`occupation_correlations`).
    """
    civecs = np.asarray(civecs, dtype=np.complex128)
    if civecs.ndim == 1:
        civecs = civecs[:, None]
    nstate = civecs.shape[1]
    weights = (np.full(nstate, 1.0 / nstate) if weights is None
               else np.asarray(weights, dtype=float) / np.sum(weights))
    conn = connections(dets) if conn is None else conn
    n = dets.n_spinor
    occ = dets.occupations()

    gamma = np.zeros((n, n), dtype=np.complex128)
    gam2 = np.zeros((n, n, n, n), dtype=np.complex128) if with_2rdm else None

    with timer("CI density matrices"):
        # --- diagonal (I = J) ---
        pop = np.einsum("s,ns->n", weights, np.abs(civecs) ** 2).real     # (ndet,)
        occf = occ.astype(np.float64)
        np.fill_diagonal(gamma, gamma.diagonal() + (pop @ occf))
        if with_2rdm:
            nn = np.einsum("n,nk,nl->kl", pop, occf, occf)                # <n_k n_l>
            np.fill_diagonal(nn, 0.0)                                     # k = l gives zero
            idx = np.arange(n)
            gam2[idx[:, None], idx[:, None], idx[None, :], idx[None, :]] += nn
            gam2[idx[:, None], idx[None, :], idx[None, :], idx[:, None]] -= nn

        # --- rank 1: J -> I excites i -> a, spectators k occupied in both ---
        if conn.n_single:
            i, a = conn.single_from, conn.single_to
            rho = conn.single_phase * np.einsum(
                "s,xs,xs->x", weights, np.conj(civecs[conn.single_i]), civecs[conn.single_j])
            _scatter(gamma, a * n + i, rho)
            if with_2rdm:
                for lo in range(0, conn.n_single, _CONN_BATCH):
                    hi = min(lo + _CONN_BATCH, conn.n_single)
                    sl = slice(lo, hi)
                    common = occ[conn.single_i[sl]] & occ[conn.single_j[sl]]
                    x, k = np.nonzero(common)
                    w = rho[sl][x]
                    ax, ix = a[sl][x], i[sl][x]
                    _scatter(gam2, ((ax * n + ix) * n + k) * n + k, w)    # Gamma_{ai,kk}
                    _scatter(gam2, ((k * n + k) * n + ax) * n + ix, w)    # Gamma_{kk,ai}
                    _scatter(gam2, ((ax * n + k) * n + k) * n + ix, -w)   # Gamma_{ak,ki}
                    _scatter(gam2, ((k * n + ix) * n + ax) * n + k, -w)   # Gamma_{ki,ak}

        # --- rank 2: J -> I excites i,j -> a,b ---
        if conn.n_double and with_2rdm:
            i, j = conn.double_from[:, 0], conn.double_from[:, 1]
            a, b = conn.double_to[:, 0], conn.double_to[:, 1]
            rho = conn.double_phase * np.einsum(
                "s,xs,xs->x", weights, np.conj(civecs[conn.double_i]), civecs[conn.double_j])
            _scatter(gam2, ((a * n + i) * n + b) * n + j, rho)
            _scatter(gam2, ((b * n + j) * n + a) * n + i, rho)
            _scatter(gam2, ((a * n + j) * n + b) * n + i, -rho)
            _scatter(gam2, ((b * n + i) * n + a) * n + j, -rho)

    # The scan covered only I < J; restore hermiticity.
    gamma = gamma + gamma.conj().T
    np.fill_diagonal(gamma, gamma.diagonal() - (pop @ occf))
    if with_2rdm:
        diag_part = np.zeros_like(gam2)
        idx = np.arange(n)
        diag_part[idx[:, None], idx[:, None], idx[None, :], idx[None, :]] = nn
        diag_part[idx[:, None], idx[None, :], idx[None, :], idx[:, None]] -= nn
        gam2 = gam2 + gam2.conj().transpose(1, 0, 3, 2) - diag_part
    return gamma, gam2


def occupation_correlations(dets: Determinants, civecs: np.ndarray,
                            weights: Optional[np.ndarray] = None) -> np.ndarray:
    """``<n_p n_q>`` for every spinor pair — diagonal in the determinant basis, hence free.

    This is all the two-particle information the entanglement measures of
    :mod:`kuiva.rdm.entropy` need, so orbital ordering for DMRG can be obtained without ever
    forming the ``n^4`` 2-RDM.
    """
    civecs = np.asarray(civecs, dtype=np.complex128)
    if civecs.ndim == 1:
        civecs = civecs[:, None]
    nstate = civecs.shape[1]
    weights = (np.full(nstate, 1.0 / nstate) if weights is None
               else np.asarray(weights, dtype=float) / np.sum(weights))
    pop = np.einsum("s,ns->n", weights, np.abs(civecs) ** 2).real
    occf = dets.occupations().astype(np.float64)
    return np.einsum("n,nk,nl->kl", pop, occf, occf)


__all__ = ["Determinants", "Connections", "connections", "connections_scan_numpy",
           "diagonal_energies",
           "hamiltonian_matrix", "occupation_matrix", "rdm12", "occupation_correlations",
           "single_excitation_operator", "determinant_memory_gb",
           "connection_memory_gb", "hamiltonian_memory_gb",
           "CASSpace", "binomial_table", "cas_dimension", "cas_vector_gb",
           "excitation_map_gb", "cas_rank_numpy", "cas_unrank_numpy",
           "excitation_map_numpy", "BYTES_PER_INCIDENCE",
           "kramers_partner", "kramers_sign", "kramers_representative", "KramersMap",
           "kramers_map_gb", "BYTES_PER_KRAMERS_ENTRY", "ladder_map", "apply_ladder",
           "DEFAULT_MAX_SPINORS"]
