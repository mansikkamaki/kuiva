"""AO -> spinor-MO two-electron integral transformation.

This is the most expensive step before the multireference layer starts, and the one whose
structure decides how expensive everything after it is. Three decisions shape it.

1. Everything is a three-index factorization
--------------------------------------------
Both factorization routes — density fitting where a good auxiliary exists, Cholesky decomposition
as the general fallback — produce the *same* object: real factors ``L^P_{mu nu}`` with

    (mu nu | la ka)  =  sum_P  L^P_{mu nu} L^P_{la ka} .

Downstream code therefore never asks which route was taken; it asks
:class:`ThreeIndexAO` for factors. That is what "DF and Cholesky paths behind one interface"
means here, and it is also what keeps the door open for integral-direct Cholesky later:
:func:`pivoted_cholesky` takes a *column callback*, so the matrix never has to exist.

The four-index integrals are never stored for the full orbital space. They are assembled on
demand from the factors for the small blocks that actually need them (the active space, the
CASSCF gradient blocks, the NEVPT2 classes), which is the difference between an N^4 array and
an N^2 K one.

2. The spin structure is exploited, the spinor structure is not
----------------------------------------------------------------
The AO integrals are **spin-free** — the Coulomb operator does not act on spin — so in the
two-component spinor basis the spin sum collapses into the transformation matrices:

    B^P_{pq}  =  sum_{sigma in {alpha,beta}}  sum_{mu nu} C^{sigma *}_{mu p} L^P_{mu nu}
                 C^{sigma}_{nu q}                                    (this module's kernel)
    (pq|rs)   =  sum_P B^P_{pq} B^P_{rs}                             (:func:`assemble_4c`)

Note carefully that the second line carries **no complex conjugate**: the conjugation lives
inside ``B`` (on the bra index of each pair) and applying another one here is a silent,
plausible-looking error that survives every hermiticity check. See :func:`assemble_4c`.

Cost: the transformation is two quarter-transforms per spin component, i.e. **twice** a
scalar transform of the same AO dimension, not the sixteen times a naive treatment of the
2*nao "AO spinor" basis would cost, and not the eight times that four-component work needs.
The Kramers-pair sparsity of the *guess* (each spinor has one nonzero spin block) is
deliberately not exploited — it does not survive the first CASSCF rotation. The one structure
that does survive, and is exploited, is that ``L`` stays real: a real x complex GEMM is done
as two real GEMMs rather than promoting to a complex GEMM (which would do four), and if the
coefficients happen to be exactly real the whole chain stays real automatically.

3. It is written to be replaced kernel-by-kernel
------------------------------------------------------
Every heavy operation here is a BLAS-3 call on a blocked buffer: :func:`transform_3c` loops
over auxiliary blocks and does two ``GEMM``s and one batched ``GEMM`` per block, with a
memory budget instead of a full-array allocation. There is no Python-level loop over orbital
indices anywhere in the hot path. That means (a) NumPy on MKL already reaches most of the
achievable performance, and (b) the C++ port is mechanical: the same blocking, the same
three calls, with the temporaries fused. Profile on **CPU** seconds before deciding it is hot.

.. warning::
   **Screening and thresholds are where speed can silently cost correctness**. The
   Cholesky threshold ``tol`` is not a convergence knob to be tightened only when something
   looks wrong: it bounds the error in *every* two-electron integral, and hence the error in
   correlation energies computed from them, at ``sqrt(d_i d_j) <= tol``. The default here is
   deliberately tight (1e-6 Eh worst-case element error, well below the 1e-8 Eh energy
   tolerance of the suite once errors average out over a correlation energy). Loosening it to make
   a calculation fit is a physics decision, not a performance one, and it is logged as such.

References
----------
* Cholesky decomposition of the two-electron integral matrix: N. H. F. Beebe, J. Linderberg,
  "Simplifications in the generation and transformation of two-electron integrals in
  molecular calculations", Int. J. Quantum Chem. 12, 683 (1977), doi:10.1002/qua.560120408.
* Error bounds, pivoting and modern practice: H. Koch, A. Sanchez de Meras, T. B. Pedersen,
  "Reduced scaling in electronic structure calculations using Cholesky decompositions",
  J. Chem. Phys. 118, 9481 (2003), doi:10.1063/1.1578621; F. Aquilante, T. B. Pedersen,
  R. Lindh, "Low-cost evaluation of the exchange Fock matrix from Cholesky and density
  fitting representations of the electron repulsion integrals", J. Chem. Phys. 126, 194106
  (2007), doi:10.1063/1.2736701; F. Aquilante et al., "Cholesky Decomposition Techniques in
  Electronic Structure Theory", in "Linear-Scaling Techniques in Computational Chemistry and
  Physics", Springer (2011), pp. 301-343, doi:10.1007/978-90-481-2853-2_13.
* Density fitting (resolution of the identity): J. L. Whitten, J. Chem. Phys. 58, 4496
  (1973), doi:10.1063/1.1679012; B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, J. Chem. Phys.
  71, 3396 (1979), doi:10.1063/1.438728; O. Vahtras, J. Almloef, M. W. Feyereisen, Chem. Phys.
  Lett. 213, 514 (1993), doi:10.1016/0009-2614(93)89151-7.
* Integral transformation by successive quarter transformations: M. Yoshimine, IBM Technical
  Report RJ-555 (1969); and the standard treatment in T. Helgaker, P. Jorgensen, J. Olsen,
  "Molecular Electronic-Structure Theory", Wiley (2000), ch. 9.
* Two-component/relativistic integral transformation with spin-free AO integrals and complex
  spinor coefficients: L. Visscher, "Approximate molecular relativistic Dirac-Coulomb
  calculations using a simple Coulombic correction", Theor. Chem. Acc. 98, 68 (1997),
  doi:10.1007/s002140050280; T. Saue, H. J. Aa. Jensen, J. Chem. Phys. 111, 6211 (1999),
  doi:10.1063/1.479958.
"""
from __future__ import annotations

import math
import os
import weakref
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util import resources as res
from ..util import scratch as scratch_io
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

#: Default Cholesky threshold [Eh]. Bounds the error in any single two-electron integral
#: (Koch et al. 2003). See the warning in the module docstring before changing it.
#:
#: ⚠ **This number is an ENERGY bound and nothing else, and that is a recent statement.** It
#: was 1e-6, then 1e-8 to suppress the *anisotropy* of column pivoting — a factorization error
#: that splits degeneracies symmetry makes exact (see :func:`shell_pair_orbits` and
#: :func:`pivoted_cholesky`). That is now structural: pivoting on complete symmetry orbits is
#: the default, so the threshold no longer doubles as a symmetry knob and was re-decided on
#: energy grounds alone.
#:
#: Measured with the orbit path, at **fixed orbitals** so that nothing but the factorization
#: varies (CASCI over the full SOC manifold, measured):
#:
#: ==========  ==================  ====================  =======================
#: threshold   naux (TiCl3/Ce3+)   worst |dE_total| Eh   worst relative [cm^-1]
#: ==========  ==================  ====================  =======================
#: 1e-4        484 / 552           9.3e-03               12.1
#: 1e-6        787 / 741           2.2e-05               **0.061**
#: 1e-8        1139 / 961          5.9e-07               **0.0016**
#: 1e-10       1522 / 1235         5.7e-09               <1e-4
#: ==========  ==================  ====================  =======================
#:
#: A *relative* energy is what a spectrum is read from, and errors cancel strongly in it: no
#: threshold here reaches the 1e-8 Eh of a total energy, so the relative column is the
#: criterion. 1e-6 leaves 0.06 cm^-1, within a factor of two of the ~0.1 cm^-1 at which a
#: splitting starts to mean different physics; 1e-8 leaves 0.0016 cm^-1 for 30-45% more
#: Cholesky vectors, which is what buys the margin. Hence 1e-8.
#:
#: ⚠ Loosening this is a physics decision. It also degrades **Kramers** degeneracy, and 1e-6 Eh
#: is the top of the 1e-8..1e-6 Eh band reserved for *genuine* numerical Kramers
#: splitting — so a factorization artifact would land exactly where a real effect is expected
#: to live.
DEFAULT_CHOLESKY_TOL = 1.0e-8

#: Cholesky vectors per AO function, used for **estimating** — the pre-flight's prediction of
#: the factors, and the capacity :func:`pivoted_cholesky` starts its output array at. It is
#: replaced by the true count as soon as the decomposition has run.
#:
#: ⚠ This is one of the few numbers that must err **high**: it multiplies the three-index
#: factors and every transformed block, so a pre-flight that under-estimates it lets a
#: calculation start that cannot finish — the one failure the whole budget exists to prevent.
#: Erring high costs a wider pre-flight table and, since the decomposition grows its array
#: when the estimate is exceeded, nothing at all beyond that.
#:
#: 12 rather than 8, because **the count is a function of the decomposition threshold and the
#: threshold was tightened after this constant was first set**: 8 was measured at 1e-6, where
#: 5.6 (Ne) and 7.4 (TiCl3) were the counts, while the default is now 1e-8 and the whole
#: committed cross-check suite lands at 8.6-11.5 vectors per AO. A stale estimate here is
#: silent: it makes every plan look 30% cheaper than the calculation is.
CHOLESKY_VECTORS_PER_AO = 12.0

#: Fallback working-buffer budget for the blocked transform [GB], used when no memory limit
#: is configured. With a limit in force the buffer comes from
#: :func:`kuiva.util.resources.transient_gb` instead, so the block size scales with the
#: machine rather than with a constant chosen on one workstation. It bounds temporaries only;
#: the result array is sized by the requested orbital block and checked separately.
DEFAULT_BUFFER_GB = res.FALLBACK_BUFFER_GB


def factor_memory_gb(nao: int, naux: int) -> float:
    """Size [GB] of the packed three-index AO factors ``L^P_{mu nu}`` (exact sizing function).

    Exact: ``naux * nao(nao+1)/2`` doubles. Used by the pre-flight before the decomposition
    has run, with ``naux`` estimated from the AO count (see
    :data:`kuiva.interface.api.CHOLESKY_VECTORS_PER_AO`).
    """
    return res.array_gb((naux, npair_of(nao)), np.float64)


def mo_block_memory_gb(naux: int, nbra: int, nket: int, dtype=np.complex128) -> float:
    """Size [GB] of a transformed three-index MO block ``B^P_{pq}`` (exact sizing function).

    This is the array that decides whether a system fits: it is ``naux * nbra * nket``
    complex, i.e. ``O(nao^3)`` with a large prefactor once every spinor is transformed.
    """
    return res.array_gb((naux, nbra, nket), dtype)


# --- Packed lower-triangular AO pair indexing -------------------------------------------
# The pair index is the row-major lower triangle, ij = i*(i+1)/2 + j for i >= j — the same
# ordering as numpy's tril_indices and as PySCF's pack_tril, so DF factors from the bridge
# need no repacking.

def _pair_rows(nao: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(nao)


def npair_of(nao: int) -> int:
    return nao * (nao + 1) // 2


def _s8_column(eri8: np.ndarray, npair: int, q: int) -> np.ndarray:
    """Column ``q`` of the ``(npair, npair)`` ERI matrix, from the 8-fold packed array."""
    ij = np.arange(npair)
    idx = np.where(ij >= q, ij * (ij + 1) // 2 + q, q * (q + 1) // 2 + ij)
    return eri8[idx]


def _s8_diagonal(eri8: np.ndarray, npair: int) -> np.ndarray:
    ij = np.arange(npair)
    return eri8[ij * (ij + 3) // 2]


def shell_pair_orbits(ao_shell: np.ndarray, ao_atom: np.ndarray,
                      one_centre: bool = True) -> np.ndarray:
    """Symmetry-orbit label for every packed AO pair, for :func:`pivoted_cholesky`.

    The density functions ``chi_mu chi_nu`` with ``mu`` in shell ``A`` and ``nu`` in shell
    ``B`` span a subspace closed under rotations of the atom the two shells sit on: a rotation
    mixes the ``m`` components within each shell and nothing else. The unordered pair
    ``(A, B)`` is therefore a complete orbit, and feeding these labels to
    :func:`pivoted_cholesky` makes the factorization's spherical symmetry exact by
    construction rather than approximate below the threshold.

    ``ao_shell`` and ``ao_atom`` are :class:`kuiva.basis.layout.AOLayout` columns — plain
    integer arrays, so this module keeps no dependency on the layout object. ⚠ ``ao_shell``
    must index **one contracted function** each, which is what ``AOLayout`` guarantees;
    a labelling built from raw ``libcint`` shells would merge the several radial functions of
    a general contraction into one orbit, and those are *not* related by symmetry.

    ⚠ **This is a one-centre (spherical) construction and deliberately not a point-group one**
    (a one-centre construction by decision). With ``one_centre`` — the default — only pairs
    whose two AOs sit on the same atom are grouped, and every other pair is its own singleton
    orbit, for which the block step *is* an ordinary rank-one update. So a molecule keeps
    plain pivoting between centres and gains the atomic-local invariance, which is where all
    the measured damage is. Setting ``one_centre=False`` groups every shell pair regardless of
    centre; that is still a valid partition (correctness never depends on the labelling) but
    it blocks pairs no symmetry relates, which costs vectors for nothing.

    References
    ----------
    F. Aquilante, R. Lindh, T. B. Pedersen, "Unbiased auxiliary basis sets for accurate
    two-electron integral approximations", J. Chem. Phys. 127, 114107 (2007),
    doi:10.1063/1.2777146 — the atomic / one-centre Cholesky decomposition.
    """
    ao_shell = np.asarray(ao_shell, dtype=np.int64)
    ao_atom = np.asarray(ao_atom, dtype=np.int64)
    if ao_shell.ndim != 1 or ao_shell.shape != ao_atom.shape:
        raise ValueError("ao_shell and ao_atom must be matching 1-D arrays, got {} and {}"
                         .format(ao_shell.shape, ao_atom.shape))
    nao = ao_shell.size
    i, j = _pair_rows(nao)
    nshell = int(ao_shell.max()) + 1 if nao else 1
    hi = np.maximum(ao_shell[i], ao_shell[j])
    lo = np.minimum(ao_shell[i], ao_shell[j])
    labels = hi * nshell + lo
    if one_centre:
        # a negative, per-pair-unique label can never collide with a shell-pair label above
        off = ao_atom[i] != ao_atom[j]
        labels = np.where(off, -(np.arange(labels.size, dtype=np.int64) + 1), labels)
    return labels


# --- Pivoted Cholesky --------------------------------------------------------------------

#: Relative width within which two eigenvalues of an orbit's Gram matrix count as
#: **degenerate**, measured against that block's largest eigenvalue. Degeneracies inside an
#: orbit are symmetry degeneracies, so they are exact up to rounding, and the residual Gram of
#: a run in which every previous orbit was complete is symmetric too. See
#: :func:`pivoted_cholesky`; the behaviour is measured, not assumed.
ORBIT_DEGENERACY_RTOL = 1.0e-6

#: Relative floor below which a direction inside an orbit's Gram matrix is numerical noise
#: rather than a direction, measured against that block's largest eigenvalue. Its job is
#: **stability, not accuracy** (``tol`` does accuracy): the block emits ``M_{:,P} U s^-1/2``,
#: so inverting the square root of a null — or, from rounding, slightly negative — eigenvalue
#: is unbounded amplification. A group is kept only if its *smallest* member clears this, so
#: a merged group containing a null direction is dropped **whole** and no ``degeneracy_rtol``
#: can produce a NaN. Without it, a grouping tolerance loose enough to hold Lu(3+)'s
#: degeneracies together made Ar's blocks produce ``sqrt`` of a negative number.
ORBIT_STABILITY_RTOL = 1.0e-13


def _initial_capacity(n: int, initial_vectors: Optional[int], limit: int) -> int:
    """Rows to allocate the factor array with before the decomposition has run.

    ⚠ **Not a limit and not an accuracy setting**: the array grows when the decomposition
    needs more, so an estimate that is too low costs one copy and nothing else. What it is
    for is the opposite failure — allocating for the worst case. The worst case is one vector
    per column, i.e. ``npair`` of them, and that array is *larger than the two-electron
    integrals themselves* (27.5 GB against 13.7 GB at nao = 348), which is what used to be
    allocated here on every route whether or not it was ever going to be filled.
    """
    if initial_vectors is None:
        # From the pair count, which is what this function is given: nao is its inverse.
        nao = int((math.sqrt(8.0 * n + 1.0) - 1.0) / 2.0)
        initial_vectors = int(CHOLESKY_VECTORS_PER_AO * max(nao, 1))
    return int(max(1, min(limit, initial_vectors)))


def _grow_factors(lvec: np.ndarray, needed: int, limit: int) -> np.ndarray:
    """Double the factor array's capacity, up to ``limit``. Copies once; the peak is the two
    arrays together, which is why the capacity starts from an estimate rather than at one."""
    capacity = int(min(limit, max(needed, 2 * lvec.shape[0], 1)))
    grown = np.zeros((capacity, lvec.shape[1]), dtype=lvec.dtype)
    grown[:lvec.shape[0]] = lvec
    log.debug("Cholesky factor array grown to %d vectors (%.3f GB)", capacity,
              grown.nbytes / res.BYTES_PER_GB)
    return grown


def _degenerate_groups(s: np.ndarray, rtol: float):
    """Split a **descending** eigenvalue array into degenerate groups (index arrays)."""
    if s.size == 0:
        return []
    cuts = np.nonzero(np.diff(s) < -rtol * abs(float(s[0])))[0] + 1
    return np.split(np.arange(s.size), cuts)


def pivoted_cholesky(diagonal: np.ndarray,
                     column: Callable[[int], np.ndarray],
                     tol: float = DEFAULT_CHOLESKY_TOL,
                     max_vectors: Optional[int] = None,
                     orbits: Optional[np.ndarray] = None,
                     degeneracy_rtol: float = ORBIT_DEGENERACY_RTOL,
                     initial_vectors: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pivoted Cholesky decomposition of a symmetric positive-semidefinite matrix ``M``.

    ``M`` is never required to exist: it is accessed through its ``diagonal`` and a
    ``column(q)`` callback. That is what makes the same routine serve the in-memory ERI
    matrix here and an integral-direct decomposition later, where ``column`` evaluates a
    shell-pair batch on demand (design the kernel so the port is mechanical). ⚠ The block
    path below calls ``column`` once per index of an orbit and needs no block callback, so
    that interface commitment is unchanged.

    Returns ``(L, pivots, residual)`` with ``L`` of shape ``(nvec, n)`` such that
    ``M ~= L^T L``, the pivot order, and the largest remaining diagonal element.

    **Plain path** (``orbits is None``) — the standard algorithm (Beebe & Linderberg 1977;
    Koch, Sanchez de Meras & Pedersen 2003): repeatedly pick the largest remaining diagonal,
    subtract the rank-one update, stop when the largest remaining diagonal falls below
    ``tol``. Because ``M`` is positive semidefinite, that residual bounds the error in every
    element by Cauchy-Schwarz, ``|M_ij - (L^T L)_ij| <= sqrt(d_i d_j) <= tol`` — the property
    that makes the truncation safe, and why pivoting is essential.

    **Block-orbit path** (``orbits`` given) — ``orbits[i]`` labels column ``i`` with the orbit
    of a symmetry group it belongs to; whole orbits are then taken as pivots at once.

    Why: the truncation is the projection ``M ~= M_{:,P} (M_{P,P})^-1 M_{P,:}`` onto the span
    of the selected columns. ``M`` commutes with the group, so if the pivot set ``P`` is a
    **union of complete orbits** the projector is group-invariant and the whole approximation
    commutes with the group — the symmetry is preserved *exactly*, independent of how many
    orbits are kept. Plain pivoting selects individual columns and therefore does not: it
    treats the ``m`` components of a shell inequivalently at the size of the threshold, which
    splits degeneracies symmetry makes exact (see :data:`DEFAULT_CHOLESKY_TOL`).

    .. warning::
       **This cannot be done as a sequence of rank-one updates, and doing so destroys the
       factorization.** Plain pivoting is stable *because* it always takes the largest
       remaining diagonal; forcing a whole orbit forces small-diagonal pivots, and
       ``col / sqrt(d)`` then amplifies rounding without bound. Measured on Lu(3+) 4f: the
       naive sequential version gives a factorization error of **251 Eh**, against 2.5e-08 Eh
       for plain pivoting. Ar's small, well-conditioned shells hide this completely.

       The orbit is therefore processed as a **block**: eigendecompose the residual Gram
       ``M_{P,P} = U s U^T`` and emit ``M_{:,P} U s^-1/2``, which is the same projector
       written stably.

    .. warning::
       ⚠ **Small eigenvalues are discarded in whole degenerate groups, never individually.**
       A per-eigenvector threshold is the obvious implementation and it silently breaks the
       invariance the block form exists to preserve, while looking like ordinary numerical
       hygiene. Two floors apply, and a group must clear **both** as a whole:

       * *accuracy* — the group's largest member must exceed ``tol``. A direction contributing
         less than ``tol`` to the residual is what the error bound already allows to be
         dropped.
       * *stability* — the group's smallest member must exceed
         :data:`ORBIT_STABILITY_RTOL` times the block's largest. See that constant.

    ⚠ ``degeneracy_rtol`` is a genuine trade-off with **no universal valley**, so it is
    exposed rather than hidden: too tight splits a symmetry-degenerate group and leaks the
    invariance, too loose merges distinct eigenvalues and discards a group that was carrying
    real weight, which costs accuracy. Measured gap spectra show
    Ar with a clean valley from 1e-10 to 1e-1 but Lu(3+) with gaps at *every* decade. The
    default is chosen to hold the accuracy contract (``residual <= tol``) everywhere measured.

    ⚠ ``max_vectors`` is enforced at **orbit granularity** on the block path — truncating
    inside an orbit is precisely what breaks the invariance, so an orbit that would not fit
    entirely is not started, and the decomposition stops short of the limit rather than
    splitting one.

    ⚠ ``initial_vectors`` is a **capacity**, not a limit: the output array starts at that many
    rows and is grown when the decomposition needs more, so an under-estimate costs one copy
    and never accuracy. It exists because the alternative — allocating for the worst case of
    one vector per column — is an array *larger than the two-electron integrals themselves*
    (27.5 GB against 13.7 GB at nao = 348) that is never more than a few per cent filled, and
    it was the real ceiling on system size long before the integrals were.

    References
    ----------
    The orbit-complete idea is the atomic / one-centre Cholesky decomposition of
    F. Aquilante, R. Lindh, T. B. Pedersen, "Unbiased auxiliary basis sets for accurate
    two-electron integral approximations", J. Chem. Phys. 127, 114107 (2007),
    doi:10.1063/1.2777146.
    """
    diag = np.array(diagonal, dtype=np.float64, copy=True)
    n = diag.size

    if np.any(diag < -abs(tol)):
        log.error("the matrix being Cholesky-decomposed has a negative diagonal element "
                  "(min %.3e): it is not positive semidefinite and the factors are "
                  "meaningless", float(diag.min()))

    if orbits is not None:
        return _blocked_cholesky(diag, column, tol, max_vectors, orbits, degeneracy_rtol,
                                 initial_vectors)

    limit = int(max_vectors) if max_vectors else n
    lvec = np.zeros((_initial_capacity(n, initial_vectors, limit), n), dtype=np.float64)
    pivots = np.zeros(limit, dtype=np.int64)

    nvec = 0
    while nvec < limit:
        q = int(np.argmax(diag))
        dmax = float(diag[q])
        if dmax <= tol:
            break
        if nvec == lvec.shape[0]:
            lvec = _grow_factors(lvec, nvec + 1, limit)
        col = np.asarray(column(q), dtype=np.float64)
        if nvec:
            # M_iq - sum_{P<nvec} L_P,i L_P,q : one GEMV, the whole cost of the step.
            col = col - lvec[:nvec].T @ lvec[:nvec, q]
        v = col / np.sqrt(dmax)
        lvec[nvec] = v
        diag -= v * v
        np.maximum(diag, 0.0, out=diag)      # rounding can push a converged diagonal below 0
        pivots[nvec] = q
        nvec += 1

    residual = float(diag.max()) if n else 0.0
    if nvec == limit and residual > tol:
        log.warning("Cholesky decomposition stopped at the vector limit (%d) with residual "
                    "%.3e > tol %.2e; the factorization is incomplete and two-electron "
                    "integrals carry an error of that order", limit, residual, tol)
    return lvec[:nvec], pivots[:nvec], residual


def _orbit_keep_count(s: np.ndarray, tol: float, degeneracy_rtol: float) -> int:
    """How many of an orbit block's **descending** eigenvalues to keep. The whole rule, once.

    ⚠ Whole degenerate groups only, on BOTH floors — splitting a group across either is
    exactly what breaks the invariance the block form exists to preserve, while looking like
    ordinary numerical hygiene:

    * *accuracy* — the group's **largest** member must exceed ``tol``. A direction
      contributing less than that to the residual is what the error bound already allows to
      be dropped.
    * *stability* — the group's **smallest** must exceed :data:`ORBIT_STABILITY_RTOL` times
      the block's largest, since the emitted vectors carry ``s^-1/2``. See that constant.

    Shared by the in-core and the out-of-core block paths: two copies of this rule would be
    two definitions of what the factorization keeps, and both would pass every test that
    compares a factorization with itself.
    """
    if s.size == 0:
        return 0
    floor = ORBIT_STABILITY_RTOL * max(float(s[0]), 0.0)
    keep = 0
    for group in _degenerate_groups(s, degeneracy_rtol):
        if float(s[group].max()) <= tol or float(s[group].min()) <= floor:
            break
        keep = int(group[-1]) + 1
    return keep


def _blocked_cholesky(diag: np.ndarray,
                      column: Callable[[int], np.ndarray],
                      tol: float,
                      max_vectors: Optional[int],
                      orbits: np.ndarray,
                      degeneracy_rtol: float,
                      initial_vectors: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """The orbit-complete block path of :func:`pivoted_cholesky`; see its docstring."""
    n = diag.size
    orbits = np.asarray(orbits)
    if orbits.shape != (n,):
        raise ValueError("orbits must have one label per column ({}), got {}"
                         .format(n, orbits.shape))
    members = {label: np.nonzero(orbits == label)[0] for label in np.unique(orbits)}

    limit = int(max_vectors) if max_vectors else n
    lvec = np.zeros((_initial_capacity(n, initial_vectors, limit), n), dtype=np.float64)
    pivots = np.zeros(limit, dtype=np.int64)
    nvec = 0
    # ⚠ an orbit is exhausted by the block that processes it: the projection removes the whole
    # span of its columns, so re-selecting it could only ever come from rounding.
    open_mask = np.ones(n, dtype=bool)

    while nvec < limit and open_mask.any():
        q = int(np.argmax(np.where(open_mask, diag, -np.inf)))
        if float(diag[q]) <= tol:
            break
        idx = members[orbits[q]]
        if nvec + idx.size > limit:
            break                          # never truncate an orbit -- stop short instead

        cols = np.empty((n, idx.size), dtype=np.float64)
        for k, p in enumerate(idx):
            cols[:, k] = np.asarray(column(int(p)), dtype=np.float64)
        if nvec:
            cols -= lvec[:nvec].T @ lvec[:nvec][:, idx]

        gram = cols[idx, :]
        gram = 0.5 * (gram + gram.T)
        s, u = np.linalg.eigh(gram)
        s, u = s[::-1], u[:, ::-1]                       # descending

        # Whole degenerate groups only, on both floors -- see _orbit_keep_count, which is
        # the single statement of that rule and is shared with the out-of-core path.
        keep = _orbit_keep_count(s, tol, degeneracy_rtol)
        if keep:
            v = (cols @ u[:, :keep]) / np.sqrt(s[:keep])
            if nvec + keep > lvec.shape[0]:
                lvec = _grow_factors(lvec, nvec + keep, limit)
            lvec[nvec:nvec + keep] = v.T
            pivots[nvec:nvec + keep] = idx[:keep]
            nvec += keep
            diag -= np.einsum("ij,ij->i", v, v)
            np.maximum(diag, 0.0, out=diag)              # rounding can push it below 0
        open_mask[idx] = False

    lvec, piv = lvec[:nvec], pivots[:nvec]
    residual = float(diag.max()) if n else 0.0
    if nvec >= limit and residual > tol:
        log.warning("orbit-complete Cholesky decomposition stopped at the vector limit (%d) "
                    "with residual %.3e > tol %.2e; the factorization is incomplete and "
                    "two-electron integrals carry an error of that order", limit, residual, tol)
    return lvec, piv, residual



# --- Out-of-core: the decomposition that never holds the factor array ---------------------

#: Candidate columns updated per GEMM in the streamed decomposition's rank-k updates. It
#: bounds the *temporary* those updates would otherwise make (one full candidate block), and
#: nothing else: blocking is a parameter here rather than a budget read, and the loop it
#: bounds contains no resource call of any kind (B7/B8).
STREAM_UPDATE_CHUNK = 256

#: Fraction of the streamed decomposition's working budget given to the history read buffer;
#: the rest holds the candidate columns. Half and half because the two are the same kind of
#: array read the same number of times per pass — the read buffer sets how many passes the
#: file costs, the candidate width sets how many pivots one pass can serve, and neither is
#: worth starving for the other. Measured alternatives are in the package validation notes.
STREAM_HISTORY_FRACTION = 0.5


class _StreamingFactorWriter:
    """The scratch file a decomposition appends its vectors to while it runs.

    The write half of the out-of-core route (:func:`streamed_cholesky`); the read half is
    :class:`_ScratchFactorStore`, which **adopts this file** when the decomposition is done,
    so the finished factors are a spilled store that was never in RAM to begin with.

    Two things it is deliberately not. It is not an :class:`kuiva.util.scratch.ExtentFile` —
    nothing here is ever freed or replaced, the file only grows, and an allocator for that is
    a liability rather than a service. And it is not buffered by us beyond one slab: the page
    cache is the buffer (see :mod:`kuiva.util.scratch`), so a machine with RAM to spare
    serves the update passes at memory speed and a tight one degrades to device speed.

    ⚠ The file is deleted with this object unless :meth:`adopt` hands it on. A decomposition
    that raises half way therefore leaves nothing behind, which is the only acceptable
    behaviour for a store whose contents are meaningless without the pivot bookkeeping.
    """

    #: Append slab [bytes]: rows are accumulated to about this much before one write.
    WRITE_SLAB = 64 * 1024 * 1024

    def __init__(self, path: str, npair: int) -> None:
        self.path = str(path)
        self.npair = int(npair)
        self.naux = 0
        self._row_bytes = self.npair * 8
        self._file = open(self.path, "w+b", buffering=0)
        self._pending: List[np.ndarray] = []
        self._pending_rows = 0
        self._slab_rows = max(1, self.WRITE_SLAB // max(self._row_bytes, 1))
        #: Bytes read back by the update passes — the cost figure of this route, reported.
        self.bytes_read = 0
        self.n_passes = 0
        self._finalizer = weakref.finalize(self, _scratch_cleanup, self._file, self.path,
                                           None)

    @property
    def size_gb(self) -> float:
        return self.naux * self._row_bytes / res.BYTES_PER_GB

    def append(self, rows: np.ndarray) -> None:
        """Add ``(k, npair)`` finished vectors. Copied, so the caller may reuse its buffer."""
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.npair:
            raise ValueError("streamed factor rows must be (k, {}), got {}"
                             .format(self.npair, rows.shape))
        self._pending.append(rows.copy())
        self._pending_rows += rows.shape[0]
        self.naux += rows.shape[0]
        if self._pending_rows >= self._slab_rows:
            self.flush()

    def flush(self) -> None:
        """Write out whatever is pending. Cheap and idempotent; no fsync (see the class)."""
        if not self._pending:
            return
        self._file.seek((self.naux - self._pending_rows) * self._row_bytes)
        for block in self._pending:
            _write_full(self._file, block)
        self._pending = []
        self._pending_rows = 0

    def passes(self, block_rows: int, buffer: np.ndarray):
        """Iterate the vectors written so far as sequential ``(nb, npair)`` blocks.

        ⚠ **Sequential and row-major, never a column gather.** The rows are the auxiliary
        index and the columns the AO pair index, so taking "the pivot columns of the whole
        history" would be one seek per row; a whole-file pass at streaming bandwidth beats
        that by orders of magnitude even when it reads a hundred times the bytes.
        """
        self.flush()
        if self.naux == 0:
            return
        self.n_passes += 1
        self._file.seek(0)
        for p0 in range(0, self.naux, block_rows):
            nb = min(block_rows, self.naux - p0)
            view = buffer[:nb]
            _readinto_full(self._file, view)
            self.bytes_read += view.nbytes
            yield p0, view

    def close(self) -> None:
        """Close and delete the file. Never raises; what a failed decomposition leaves."""
        self._finalizer()

    def adopt(self) -> "_ScratchFactorStore":
        """Hand the finished file to a reading store, which owns and deletes it from now on."""
        self.flush()
        naux, path = self.naux, self.path
        self._finalizer.detach()               # the store's finalizer replaces this one
        self._file.close()
        return _ScratchFactorStore(path, naux, self.npair)


def streamed_working_gb(nao: int, *, rows: int) -> float:
    """Size [GB] of the streamed decomposition's whole working set (exact sizing function).

    ``rows`` is the whole row budget: the history read block, the candidate columns and the
    update chunk are all ``(rows_i, npair)`` float64 and share it — everything this route
    holds is a whole number of packed AO-pair rows, which is what makes one number size it.
    The ``(npair,)`` diagonal is counted on top, so this is what the arrays actually cost
    and not a fraction of it.
    """
    n = npair_of(int(nao))
    return res.array_gb((int(rows), n), np.float64) + res.array_gb((n,), np.float64)


def streamed_cholesky(diagonal: np.ndarray,
                      column: Callable[[int], np.ndarray],
                      nao: int,
                      tol: float = DEFAULT_CHOLESKY_TOL,
                      *,
                      directory=None,
                      max_vectors: Optional[int] = None,
                      orbits: Optional[np.ndarray] = None,
                      degeneracy_rtol: float = ORBIT_DEGENERACY_RTOL,
                      budget_gb: Optional[float] = None
                      ) -> Tuple["_ScratchFactorStore", np.ndarray, float, dict]:
    """:func:`pivoted_cholesky` with the factor array on scratch instead of in RAM.

    Returns ``(store, pivots, residual, stats)``. The store is the finished factorization,
    already spilled; nothing of size ``(naux, npair)`` is ever allocated.

    **Why it exists.** The factor spill frees the rows *after* the decomposition has built
    them, so on a large system the peak that binds is the array being built — 45 GB at
    nao = 1000, 155 GB at a real single-molecule magnet — and a machine that cannot hold that
    transient is refused even though every consumer of the rows already streams them.

    **The algorithm, and what is exactly preserved.** The pivot sequence is the one
    :func:`pivoted_cholesky` would take, *unchanged*: the next pivot is always the largest
    element of the true updated diagonal, which is maintained in RAM (``diag -= v*v``) and
    needs no history at all. Only the **updated column** of a pivot needs the vectors already
    produced, and that is what is reorganized:

    * a **pass** qualifies the widest set of candidate columns the budget allows, reads the
      whole history file once (sequentially, in ``(nb, npair)`` blocks) and subtracts it from
      all of them in one rank-``nb`` update per block;
    * the pass then produces pivots from that set exactly as the in-core loop does, applying
      each new vector to the candidates it holds, and **ends the moment the true argmax lies
      outside the set** — so no pivot is ever taken out of order;
    * candidates the pass did not consume are **retained**: they are current (every in-pass
      update reached them), so the next pass re-reads the history only for the columns it
      adds. Without this a pass that ends after one pivot would re-evaluate everything it had
      qualified, which on the integral-direct route means re-evaluating integrals.

    Total IO is therefore ``O(n_passes x file)`` with ``n_passes ~ naux / rows``, and both
    numbers are measured and reported rather than assumed: ``stats`` carries the pass count,
    the bytes read and the columns evaluated, and :meth:`ThreeIndexAO.report` prints them.

    ⚠ **Bitwise identity with the in-core path is NOT claimed and cannot be.** The same
    subtraction is summed in a different order — history blocks and in-pass rank-*k* updates,
    against one GEMV over all previous vectors — so the two agree to rounding, not to the
    last bit. The vector *count* and the residual bound are the same, and the pivot sequence
    is the same wherever the diagonal has no ties; where two symmetry-equivalent columns are
    tied, the last bits decide which goes first and the result is a **different valid
    factorization of the same matrix**, with elements agreeing to ``tol``. Measured on a 3d
    complex: identical vector count and residual at every working-set size, reconstructed
    integrals agreeing to 4e-16 where the sequence matched and to the threshold where a tie
    broke the other way. Anything that compares two factorizations must therefore compare the
    **integrals they reconstruct**, never the factor rows.

    ⚠ The orbit-complete path (``orbits``) keeps its whole-block treatment: an orbit is
    qualified in full or not at all, its Gram matrix is eigendecomposed exactly as in core,
    and :func:`_orbit_keep_count` — one implementation, shared — decides what it emits. A
    budget too small to hold the largest orbit is refused rather than made to split one.

    References
    ----------
    The two-step / batched organization of an out-of-core Cholesky decomposition of the
    two-electron matrix follows F. Aquilante, T. B. Pedersen, R. Lindh, "Low-cost evaluation
    of the exchange Fock matrix from Cholesky and density fitting representations",
    J. Chem. Phys. 126, 194106 (2007), doi:10.1063/1.2736701, and the practice described in
    F. Aquilante et al., "Cholesky Decomposition Techniques in Electronic Structure Theory",
    Springer (2011), pp. 301-343, doi:10.1007/978-90-481-2853-2_13. The pivoting rule and the
    error bound are Beebe & Linderberg (1977) and Koch, Sanchez de Meras & Pedersen (2003) as
    in :func:`pivoted_cholesky`.
    """
    diag = np.array(diagonal, dtype=np.float64, copy=True)
    n = diag.size
    if n != npair_of(int(nao)):
        raise ValueError("the diagonal has {} elements, which is not the packed AO pair "
                         "count of nao = {}".format(n, nao))
    if np.any(diag < -abs(tol)):
        log.error("the matrix being Cholesky-decomposed has a negative diagonal element "
                  "(min %.3e): it is not positive semidefinite and the factors are "
                  "meaningless", float(diag.min()))
    limit = int(max_vectors) if max_vectors else n

    members = None
    widest = 1
    if orbits is not None:
        orbits = np.asarray(orbits)
        if orbits.shape != (n,):
            raise ValueError("orbits must have one label per column ({}), got {}"
                             .format(n, orbits.shape))
        members = {label: np.nonzero(orbits == label)[0] for label in np.unique(orbits)}
        widest = max(int(idx.size) for idx in members.values())

    # ⚠ One budget question, asked once and outside every loop (a kernel's blocking is a
    # parameter, never a budget read inside it): the working set is a transient, so it is
    # sized against what the limit allows rather than reserved, and the loops below contain
    # no resource call at all.
    naux_estimate = max(1, int(CHOLESKY_VECTORS_PER_AO * max(int(nao), 1)))
    budget = res.transient_gb() if budget_gb is None else float(budget_gb)
    row_gb = res.array_gb((n,), np.float64)
    # Everything the decomposition holds is a whole number of ``(npair,)`` rows: the history
    # block it reads into, the candidate columns it works on, and the update chunk. ⚠ Clamped
    # by what the problem can use as well as by what the budget allows — there are at most
    # ``n`` columns to qualify and at most one estimated factorization of history to read, and
    # a buffer past either is memory spent on nothing (measured: a 1 GB history buffer for a
    # 19-AO molecule before this clamp existed).
    rows = min(int(max(0.0, budget - row_gb) // row_gb),
               n + naux_estimate + STREAM_UPDATE_CHUNK)
    hist_rows = max(1, min(int(rows * STREAM_HISTORY_FRACTION), naux_estimate))
    rest = rows - hist_rows
    chunk = int(max(1, min(STREAM_UPDATE_CHUNK, rest // 2)))
    cand_max = int(min(rest - chunk, n))
    if cand_max < widest:
        # Refused, never made to fit by splitting an orbit or by shrinking the history block
        # until a pass reads one row at a time.
        res.require("streamed Cholesky working set",
                    streamed_working_gb(nao, rows=2 * max(widest, 1) + 2),
                    note="{} AO pairs; the widest symmetry orbit is {} columns"
                         .format(n, widest),
                    advice=["raise the memory limit: this route needs room for the widest "
                            "symmetry orbit and one history block, and nothing more",
                            "use a smaller basis set",
                            "factors=\"scratch\" decomposes in core and spills afterwards, "
                            "which needs the whole factor array in RAM but no working set"])
        cand_max = max(widest, 1)
        chunk = min(chunk, cand_max)

    directory = directory if directory is not None else res.require_scratch(
        "streamed Cholesky factors", factor_memory_gb(nao, naux_estimate),
        advice=["loosen the Cholesky threshold (fewer vectors)",
                "use a smaller basis set"])
    path = os.path.join(str(directory),
                        "kuiva-cholesky-{}-{:x}.bin".format(os.getpid(), id(diag)))
    writer = _StreamingFactorWriter(path, n)
    history = np.empty((hist_rows, n), dtype=np.float64)

    pivots = np.zeros(limit, dtype=np.int64)
    nvec = 0
    n_columns = 0
    open_mask = np.ones(n, dtype=bool)         # an index is a pivot (or an orbit) once
    # ⚠ Candidates are stored **one column per row**, ``(ncand, npair)``, the same way the
    # file is: every update then writes contiguous memory. Held the other way up — the
    # natural ``(npair, ncand)`` — each update writes one strided column at a time, which
    # measured 1.6-3.3x the CPU of the same decomposition for no other difference.
    cand_idx = np.zeros(0, dtype=np.int64)     # retained candidates, always fully updated
    cand_rows = np.zeros((0, n), dtype=np.float64)
    finished = False
    try:
        while nvec < limit and not finished:
            # -- qualify: the candidate set this pass will work from, chosen from the true
            # diagonal and ⚠ **always containing the next pivot's group first**. Choosing it
            # from scratch rather than filling the room left by the retained columns is what
            # makes that guarantee: a retained set that happened to fill the budget would
            # otherwise be able to lock the pivot the loop is about to need out of the pass,
            # and the decomposition would stop with a residual far above the threshold.
            selected: List[int] = []
            seen: set = set()
            for q in np.argsort(-np.where(open_mask, diag, -np.inf), kind="stable"):
                q = int(q)
                if not open_mask[q] or diag[q] <= tol:
                    break
                group = np.asarray([q]) if orbits is None else members[orbits[q]]
                if int(group[0]) in seen:
                    continue
                if len(selected) + group.size > cand_max:
                    break                      # never a partial orbit; the next pass has it
                selected.extend(int(i) for i in group)
                seen.update(int(i) for i in group)
            if not selected:
                break                          # nothing left above the threshold
            wanted = np.asarray(sorted(selected), dtype=np.int64)
            # Retained columns still wanted are already current — every in-pass update
            # reached them — so only the rest are fetched and taken through the history.
            if cand_idx.size:
                still = np.isin(cand_idx, wanted)
                cand_idx, cand_rows = cand_idx[still], np.ascontiguousarray(cand_rows[still])
            new_idx = np.setdiff1d(wanted, cand_idx)
            if new_idx.size:
                new_rows = np.empty((new_idx.size, n), dtype=np.float64)
                for k, p in enumerate(new_idx):
                    new_rows[k] = np.asarray(column(int(p)), dtype=np.float64)
                n_columns += int(new_idx.size)
                # -- one sequential pass over everything written so far, applied to the new
                # columns only.
                for _p0, block in writer.passes(hist_rows, history):
                    gathered = np.ascontiguousarray(block[:, new_idx].T)   # (n_new, nb)
                    for c0 in range(0, new_idx.size, chunk):
                        sl = slice(c0, c0 + chunk)
                        new_rows[sl] -= gathered[sl] @ block
                cand_idx = np.concatenate([cand_idx, new_idx])
                cand_rows = (np.vstack([cand_rows, new_rows]) if cand_rows.size
                             else new_rows)
            where = {int(p): k for k, p in enumerate(cand_idx)}
            consumed: List[int] = []

            # -- produce pivots, in exactly the order the in-core loop would take them
            while nvec < limit:
                q = int(np.argmax(np.where(open_mask, diag, -np.inf)))
                if float(diag[q]) <= tol:
                    finished = True
                    break
                if orbits is None:
                    if q not in where:
                        break                  # the argmax left the set: re-qualify
                    v = (cand_rows[where[q]] / math.sqrt(float(diag[q])))[None, :]
                    take = np.asarray([q], dtype=np.int64)
                    open_mask[q] = False
                    consumed.append(where[q])
                else:
                    idx = members[orbits[q]]
                    if int(idx[0]) not in where:
                        break
                    if nvec + idx.size > limit:
                        finished = True
                        break                  # never truncate an orbit -- stop short
                    pos = np.asarray([where[int(p)] for p in idx], dtype=np.int64)
                    block_rows = cand_rows[pos]                     # (m, n), contiguous
                    gram = block_rows[:, idx]
                    gram = 0.5 * (gram + gram.T)
                    sv, u = np.linalg.eigh(gram)
                    sv, u = sv[::-1], u[:, ::-1]
                    keep = _orbit_keep_count(sv, tol, degeneracy_rtol)
                    open_mask[idx] = False
                    consumed.extend(int(p) for p in pos)
                    if not keep:
                        continue
                    v = (u[:, :keep].T @ block_rows) / np.sqrt(sv[:keep])[:, None]
                    take = idx[:keep]
                writer.append(v)
                pivots[nvec:nvec + v.shape[0]] = take
                nvec += v.shape[0]
                diag -= np.einsum("kj,kj->j", v, v)
                np.maximum(diag, 0.0, out=diag)   # rounding can push a converged one below 0
                # The new vectors reach every candidate this pass holds, retained ones
                # included -- which is what keeps a retained column current across passes.
                gathered = np.ascontiguousarray(v[:, cand_idx].T)      # (ncand, k)
                for c0 in range(0, cand_idx.size, chunk):
                    sl = slice(c0, c0 + chunk)
                    cand_rows[sl] -= gathered[sl] @ v

            keepers = np.setdiff1d(np.arange(cand_idx.size), np.asarray(consumed, dtype=int))
            if keepers.size == cand_idx.size and not new_idx.size:
                break                          # no progress and nothing new: it would loop
            cand_idx = cand_idx[keepers]
            cand_rows = np.ascontiguousarray(cand_rows[keepers])
        store = writer.adopt()
    except BaseException:
        writer.close()
        raise

    residual = float(diag.max()) if n else 0.0
    if nvec >= limit and residual > tol:
        log.warning("out-of-core Cholesky decomposition stopped at the vector limit (%d) "
                    "with residual %.3e > tol %.2e; the factorization is incomplete and "
                    "two-electron integrals carry an error of that order", limit, residual,
                    tol)
    stats = {"passes": writer.n_passes, "gb_read": writer.bytes_read / res.BYTES_PER_GB,
             "columns": n_columns, "candidate_columns": cand_max,
             "history_rows": hist_rows, "update_chunk": chunk}
    log.debug("streamed Cholesky: %d vectors, %d update passes, %.2f GB read, %d columns "
              "evaluated", nvec, writer.n_passes, stats["gb_read"], n_columns)
    return store, pivots[:nvec], residual, stats


def _warn_if_coulomb_only(aux_name: Optional[str]) -> None:
    """Warn when a *Coulomb-fitting* auxiliary is about to be used for correlated integrals.

    An efficiency choice that can quietly cost accuracy must be flagged, and this is
    one. A J-fitting set (``x2c-JFIT``, ``def2/J``, ...) is optimized to reproduce the Coulomb
    *matrix* built from an SCF density, where fitting errors cancel between the numerator and
    the density. Individual transformed integrals ``(pq|rs)`` — which is what CI, CASSCF and
    NEVPT2 consume, one at a time and without that cancellation — are reproduced far less
    well. Measured on HF/x2c-SVPall-2c with the registry's recommended ``x2c-JFIT``: worst
    active-space integral error **1.7e-3 Eh**, i.e. five orders of magnitude above the 1e-8 Eh
    tolerance of the suite, and a hundred times the default Cholesky threshold.

    The fix is a JK- or correlation-fitting auxiliary, or the Cholesky route — whose error,
    unlike a fitting error, is bounded by a threshold the user sets.
    """
    if not aux_name:
        return
    role = ""
    try:
        from ..basis import registry as reg
        if reg.has_family(aux_name):
            role = reg.get_family(aux_name).role or ""
    except Exception:                                  # registry is metadata, never a blocker
        role = ""
    name = aux_name.lower()
    coulomb_only = ("coulomb (j)" in role.lower()) or (
        role == "" and ("jfit" in name or name.endswith("/j")) and "jk" not in name)
    if coulomb_only:
        log.warning("density fitting with a Coulomb-fitting auxiliary (%s). It is calibrated "
                    "for the Coulomb matrix, not for individual transformed integrals: "
                    "errors of order 1e-3 Eh per (pq|rs) are expected, against a 1e-8 Eh "
                    "target. Use a JK/correlation-fitting auxiliary or the Cholesky route "
                    "for correlated work.", aux_name)


# --- Scratch residence for the factor rows ------------------------------------------------

# The short-read/short-write loops live in kuiva.util.scratch, shared with every other
# out-of-core store, so no layer re-derives them.
_readinto_full = scratch_io.readinto_full
_write_full = scratch_io.write_full


def _scratch_cleanup(fileobj, path: str, executor) -> None:
    """Finalizer for a scratch store: stop the prefetcher, close, delete. Never raises."""
    try:
        if executor is not None:
            executor.shutdown(wait=True)
    except Exception:
        pass
    try:
        fileobj.close()
    except Exception:
        pass
    try:
        os.unlink(path)
    except OSError:
        pass


class _ScratchFactorStore:
    """The packed factor rows ``(naux, npair)`` in a raw scratch file, read in row blocks.

    IO design, each point deliberate:

    * **Raw row-major binary, no container format.** The file lives exactly as long as the
      process (a finalizer deletes it), is regenerated rather than restarted from (the
      checkpoint layer's never-checkpoint rule for integral factors), and is read only by this class — so a
      block read is one contiguous ``readinto`` at ``offset = p0 * npair * 8`` with no chunk
      layer, no metadata seeks and no decompression. Random factor data does not compress.
    * **Buffered by the page cache, not by us, and never ``O_DIRECT``.** The configured memory limit
      governs Kuiva's *own* arrays; whatever RAM the machine has beyond it is exactly
      what the OS page cache uses. On a node with spare RAM the repeated sequential passes a
      CASSCF makes over this file are then served at memory speed, and on a tight node the
      cache degrades gracefully to device speed — an adaptive cache nobody had to size.
    * **One prefetch worker, double-buffered.** Every consumer walks the auxiliary index in
      ascending contiguous blocks, so after serving ``[p0, p1)`` the next block is issued to
      a single background worker while the caller computes; ``readinto`` and BLAS both
      release the GIL, so read and compute overlap. All file access goes through that one
      worker (the serving thread waits on a future), so there is no seek race and no lock.
      This is an IO thread, not a compute thread — the calculation's threading budget is untouched.
    * **Sequential append on write**, with the file preallocated to its final size
      (``posix_fallocate``) so the extents are laid out contiguously; there is deliberately
      no fsync, because scratch outliving a crash protects nothing.

    A wrong-prediction read (a consumer jumping backwards, or two consumers interleaved on
    one store) is served correctly by a synchronous read through the same worker — slower,
    never wrong. The two packed buffers are the store's whole resident footprint;
    :attr:`ThreeIndexAO.stream_row_bytes` is how a consumer's block sizing accounts for them.

    ⚠ Failure semantics are a hard error, not a cache miss: by the time a block is read the
    in-RAM copy is gone, and the only recovery would be re-running the decomposition.
    """

    #: Write slab [bytes]. Large enough to reach device bandwidth, small enough to be noise
    #: against the transient budget.
    WRITE_SLAB = 64 * 1024 * 1024

    @classmethod
    def spill(cls, l_packed: np.ndarray, path: str) -> "_ScratchFactorStore":
        """Write a finished in-RAM factor array out, then read it back through this store."""
        naux, npair = (int(l_packed.shape[0]), int(l_packed.shape[1]))
        row_bytes = npair * 8
        rows_per_slab = max(1, cls.WRITE_SLAB // max(row_bytes, 1))
        with timer("factor spill to scratch"):
            with open(str(path), "wb", buffering=0) as f:
                try:
                    os.posix_fallocate(f.fileno(), 0, naux * row_bytes)
                except (AttributeError, OSError):
                    pass                       # preallocation is an optimization, not a need
                for r0 in range(0, naux, rows_per_slab):
                    _write_full(f, l_packed[r0:r0 + rows_per_slab])
        return cls(str(path), naux, npair)

    def __init__(self, path: str, naux: int, npair: int) -> None:
        """Adopt an existing row-major ``(naux, npair)`` float64 file at ``path``.

        ⚠ The store **owns** the file from here: its finalizer deletes it. Two producers
        hand it one — :meth:`spill` (a finished array written out) and the streamed
        decomposition, which wrote the rows as it made them and never held the array at all.
        """
        from concurrent.futures import ThreadPoolExecutor

        self.naux, self.npair = int(naux), int(npair)
        self.path = str(path)
        self._row_bytes = self.npair * 8
        self._file = open(self.path, "rb", buffering=0)
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="kuiva-factor-io")
        self._buffers = [np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)]
        self._active = 0
        self._pending = None                   # (future, p0, p1, slot) of the prefetched read
        self._finalizer = weakref.finalize(self, _scratch_cleanup, self._file, self.path,
                                           self._executor)

    def _do_read(self, p0: int, p1: int, slot: int) -> np.ndarray:
        """Runs on the worker thread — the only thread that touches the file."""
        need = (p1 - p0) * self.npair
        if self._buffers[slot].size < need:
            self._buffers[slot] = np.empty(need, dtype=np.float64)
        block = self._buffers[slot][:need]
        self._file.seek(p0 * self._row_bytes)
        _readinto_full(self._file, block)
        return block.reshape(p1 - p0, self.npair)

    def read(self, p0: int, p1: int) -> np.ndarray:
        """Rows ``[p0, p1)`` as ``(nb, npair)`` float64.

        ⚠ Returns a view of an internal buffer, valid until the next :meth:`read` — the one
        consumer (:meth:`ThreeIndexAO.unpack`) copies it into square form immediately.
        """
        pending, self._pending = self._pending, None
        if pending is not None and pending[1] == p0 and pending[2] == p1:
            block = pending[0].result()
            self._active = pending[3]
        else:
            if pending is not None:
                pending[0].result()            # drain the stale read; the worker is serial
            block = self._executor.submit(self._do_read, p0, p1, self._active).result()
        # Predict the next contiguous block of the same walk and start it now.
        q0, q1 = p1, min(p1 + (p1 - p0), self.naux)
        if q1 > q0:
            slot = 1 - self._active
            self._pending = (self._executor.submit(self._do_read, q0, q1, slot), q0, q1, slot)
        return block

    def close(self) -> None:
        self._finalizer()


# --- The three-index AO factors ----------------------------------------------------------

@dataclass
class ThreeIndexAO:
    """Real three-index factors ``L^P_{mu nu}`` with ``(mu nu|la ka) = sum_P L^P L^P``.

    Storage is ``(naux, npair)`` over the packed lower triangle of the AO pair index (see
    above): half the memory of the square form, the layout PySCF's DF factors already use,
    and contiguous in the pair index so an auxiliary block unpacks into a buffer with one
    fancy-index assignment. ``origin`` records how the factors were obtained, because the
    error they carry is a different kind in each case: a Cholesky factorization is exact to
    ``residual``, while a density fit carries an uncontrolled (if usually small) fitting error
    set by the auxiliary basis.
    """

    l_packed: np.ndarray
    nao: int
    origin: str                                   # "cholesky" | "df"
    tol: Optional[float] = None
    residual: Optional[float] = None
    aux_name: Optional[str] = None
    #: Whether the pivots were complete symmetry orbits, so that the factorization's spherical
    #: symmetry is exact by construction rather than accurate to ``tol``. Reported, because a
    #: free-ion multiplet splitting means something different in each case.
    orbit_complete: bool = False
    _rows: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)
    _store: Optional[_ScratchFactorStore] = field(default=None, repr=False)
    _shape: Optional[Tuple[int, int]] = field(default=None, repr=False)
    _reservation: Optional[object] = field(default=None, repr=False)

    #: Pass count, bytes read and columns evaluated, on the out-of-core route only
    #: (:func:`streamed_cholesky`). ``None`` everywhere else. Reported, because the cost of
    #: that route is IO and a route whose cost is invisible is a route nobody can judge.
    stream_stats: Optional[dict] = None

    def __post_init__(self) -> None:
        expected = npair_of(self.nao)
        if self.l_packed is None and self._store is not None:
            # Built straight into the spilled state by the out-of-core decomposition: the
            # rows were written as they were produced and the array never existed. No
            # reservation, for the same reason spill_to_scratch gives one back.
            if int(self._store.npair) != expected:
                raise ValueError("a streamed factor store of {} AO pairs does not match "
                                 "nao={} ({} expected)".format(self._store.npair, self.nao,
                                                               expected))
            self._shape = (int(self._store.naux), int(self._store.npair))
            self._rows = _pair_rows(self.nao)
            return
        if self.l_packed.ndim != 2 or self.l_packed.shape[1] != expected:
            raise ValueError("three-index factors must have shape (naux, {}) for nao={}, "
                             "got {}".format(expected, self.nao, self.l_packed.shape))
        self.l_packed = np.ascontiguousarray(self.l_packed, dtype=np.float64)
        self._shape = (int(self.l_packed.shape[0]), int(self.l_packed.shape[1]))
        self._rows = _pair_rows(self.nao)
        # the factors live for the whole calculation, so they are declared resident.
        # Declared *after* construction rather than predicted before it, because naux is not
        # known until the decomposition has run — the pre-flight makes the prediction, this
        # replaces it with the fact so that every later check starts from the truth.
        self._reservation = res.reserve(
            "three-index AO factors ({})".format(self.origin), self.memory_gb,
            note="naux={} x {} AO pairs".format(self.naux, self.npair),
            advice=["loosen the Cholesky threshold (fewer vectors, less accurate "
                    "integrals)",
                    "use density fitting with an explicit auxiliary basis, which is "
                    "usually a smaller factorization than Cholesky",
                    "spill the factors to scratch after the decomposition "
                    "(factors=\"scratch\" on the front end)",
                    "use a smaller basis set"])

    @property
    def naux(self) -> int:
        return int(self._shape[0])

    @property
    def npair(self) -> int:
        return int(self._shape[1])

    @property
    def memory_gb(self) -> float:
        """Size [GB] of the factor data, wherever it resides (RAM in-core, disk spilled)."""
        return self._shape[0] * self._shape[1] * 8.0 / 1024.0 ** 3

    @property
    def is_spilled(self) -> bool:
        """Whether the rows live in a scratch file rather than in RAM."""
        return self._store is not None

    @property
    def stream_row_bytes(self) -> float:
        """Bytes per auxiliary row a consumer's block sizing must add for the stream buffers.

        Zero in-core. Spilled, the store double-buffers packed rows (one being consumed, one
        being prefetched), so a consumer holding ``nb`` rows of temporaries makes the store
        hold ``2 nb`` packed rows beside them — unaccounted, that would be a systematic
        under-plan of every blocked loop over a spilled store.
        """
        return 2.0 * self.npair * 8.0 if self.is_spilled else 0.0

    def spill_to_scratch(self) -> "ThreeIndexAO":
        """Move the factor rows to a scratch file and free their RAM (in place).

        The trade, and when it is taken: every consumer walks the rows in sequential
        auxiliary blocks, so streaming them from disk costs bandwidth but no algorithm —
        while the freed RAM is what lets the CI workspace, the MO blocks and the Hessian
        transients of the later stages fit at all. ``factors="auto"`` on the front end takes
        it exactly when the in-core plan does not fit the configured memory limit and the spilled one
        does. On a node whose *machine* RAM exceeds the configured limit the OS page cache
        holds the file anyway and the streaming reads run at memory speed — the limit
        governs Kuiva's arrays, not the kernel's cache (see :class:`_ScratchFactorStore`
        for the whole IO design).

        ⚠ This does not reduce the decomposition-time peak: the rows are spilled *after*
        they exist. A factorization too large to build in RAM at all is what
        :func:`streamed_cholesky` is for (``factors="streamed"`` on the front end), which
        writes the rows as it produces them and never allocates the array — one rung further
        down the same ladder, and the one that costs passes over the file rather than
        nothing. Idempotent; returns ``self``.
        """
        if self.is_spilled:
            return self
        size_gb = self.memory_gb
        directory = res.require_scratch(
            "three-index factor spill", size_gb,
            advice=["loosen the Cholesky threshold (fewer vectors)",
                    "use a smaller basis set"])
        path = os.path.join(str(directory),
                            "kuiva-factors-{}-{:x}.bin".format(os.getpid(), id(self)))
        self._store = _ScratchFactorStore.spill(self.l_packed, path)
        self.l_packed = None
        # The RAM reservation goes back to the ledger; the stream buffers are the store's
        # only resident footprint and are charged to each consumer's block sizing instead
        # (stream_row_bytes), because their size is the consumer's block choice.
        res.release(self._reservation)
        self._reservation = None
        # DEBUG, not INFO: the output file reports the residence through its own structured
        # rows (the front end's residence line, report()'s entry), and this line carries a
        # per-process path that would make a committed reference output differ on every run.
        log.debug("three-index factors spilled to scratch: %.3f GB at %s", size_gb, path)
        return self

    def unpack(self, aux: slice) -> np.ndarray:
        """Unpack an auxiliary block to ``(nb, nao, nao)`` square form.

        Always a fresh array the caller owns, on either residence. On a spilled store the
        packed rows are read through the prefetching scratch reader; the sequential block
        walks every consumer already does are exactly what it is optimized for.
        """
        if self._store is not None:
            p0, p1, step = aux.indices(self.naux)
            if step != 1:
                raise ValueError("a spilled factor store reads contiguous auxiliary blocks; "
                                 "got a slice with step {}".format(step))
            blk = self._store.read(p0, p1)
        else:
            blk = self.l_packed[aux]
        i, j = self._rows
        full = np.zeros((blk.shape[0], self.nao, self.nao), dtype=np.float64)
        full[:, i, j] = blk
        full[:, j, i] = blk                        # symmetric; the diagonal is written twice
        return full

    # -- constructors -------------------------------------------------------------------
    @classmethod
    def from_df(cls, cderi: np.ndarray, nao: int, aux_name: Optional[str] = None
                ) -> "ThreeIndexAO":
        """Wrap density-fitting factors (PySCF ``_cderi``: ``(naux, npair)``, already
        contracted with the inverse-square-root metric, so no further metric work is needed).
        """
        obj = cls(l_packed=np.asarray(cderi), nao=nao, origin="df", aux_name=aux_name)
        _warn_if_coulomb_only(aux_name)
        return obj

    @classmethod
    def from_matrix(cls, diagonal: np.ndarray, column: Callable[[int], np.ndarray],
                    nao: int, tol: float = DEFAULT_CHOLESKY_TOL, *,
                    max_vectors: Optional[int] = None, orbits: Optional[np.ndarray] = None,
                    report: bool = True, note: str = "",
                    residence: str = "in-core") -> "ThreeIndexAO":
        """Cholesky-decompose the two-electron matrix reached through ``diagonal``/``column``.

        The engine both Cholesky constructors run on, and the reason
        :func:`pivoted_cholesky`'s contract never requires the matrix to exist: an in-memory
        ERI array and an integral-direct evaluator differ only in what these two callables do.
        ⚠ Feed it callables that are **consistent with each other** — ``diagonal()[q]`` must be
        the same number as ``column(q)[q]``, bit for bit. The plain path divides a column by
        the square root of a diagonal element, so two sources that disagree in the last bits
        make a factorization that is *nearly* the one asked for, which is the hardest kind of
        difference to trace back here from downstream.

        ``orbits`` — from :func:`shell_pair_orbits` — selects the orbit-complete path, which
        makes the factorization's spherical symmetry exact rather than threshold-dependent.
        ``note`` is a line for the report, saying where the matrix elements came from.

        ``residence="streamed"`` runs :func:`streamed_cholesky` instead: the vectors go to a
        scratch file as they are produced and the ``(naux, npair)`` array is never allocated,
        which is the difference between a large system running and being refused. It returns
        the same object, already spilled — see that function for what it does and does not
        preserve.
        """
        if residence == "streamed":
            with timer("Cholesky decomposition (out-of-core)") as t:
                store, _piv, residual, stats = streamed_cholesky(
                    diagonal, column, nao, tol, max_vectors=max_vectors, orbits=orbits)
            obj = cls(l_packed=None, nao=nao, origin="cholesky", tol=tol, residual=residual,
                      orbit_complete=orbits is not None, _store=store, stream_stats=stats)
            if report:
                out.subsection(log, "Cholesky decomposition of the AO two-electron "
                                    "integrals (out-of-core)")
                obj.report()
                if note:
                    out.entry(log, "matrix elements", note)
                out.entry(log, "decomposition time", t.wall, "s wall", fmt=out.TIME_FMT)
            return obj
        if residence not in ("in-core", "scratch"):
            raise ValueError("residence must be \"in-core\", \"scratch\" or \"streamed\", "
                             "got {!r}".format(residence))
        with timer("Cholesky decomposition") as t:
            # The capacity to start the factor array at — a prediction, grown if it is short.
            # Given here rather than left to be inferred, because this is the one caller that
            # knows the AO count exactly instead of having to invert the pair count for it.
            lvec, _, residual = pivoted_cholesky(
                diagonal, column, tol, max_vectors, orbits=orbits,
                initial_vectors=int(CHOLESKY_VECTORS_PER_AO * max(int(nao), 1)))
        obj = cls(l_packed=lvec, nao=nao, origin="cholesky", tol=tol, residual=residual,
                  orbit_complete=orbits is not None)
        if report:
            out.subsection(log, "Cholesky decomposition of the AO two-electron integrals")
            obj.report()
            if note:
                out.entry(log, "matrix elements", note)
            out.entry(log, "decomposition time", t.wall, "s wall", fmt=out.TIME_FMT)
        return obj

    @classmethod
    def from_eri(cls, eri: np.ndarray, nao: int, tol: float = DEFAULT_CHOLESKY_TOL,
                 *, max_vectors: Optional[int] = None, orbits: Optional[np.ndarray] = None,
                 report: bool = True, residence: str = "in-core") -> "ThreeIndexAO":
        """Cholesky-decompose conventional AO ERIs that are already in memory.

        ``eri`` may be 8-fold packed (1-D, as the bridge provides), 4-fold packed
        (``(npair, npair)``) or full (``(nao,)*4``). The 8-fold form is decomposed **without**
        ever forming the square matrix: columns are gathered from the packed array on demand,
        which halves the peak memory of the whole route.

        ⚠ **The array still has to exist**, which is the memory bound this route carries and
        the integral-direct route does not: it grows as the fourth power of the basis. Both
        routes run the same decomposition through :meth:`from_matrix` and produce the same
        object, so nothing downstream can tell them apart.
        """
        eri = np.asarray(eri)
        np_ = npair_of(nao)
        if eri.ndim == 1:
            diag = _s8_diagonal(eri, np_)
            column = lambda q: _s8_column(eri, np_, q)          # noqa: E731 (callback)
        elif eri.ndim == 2:
            diag = np.diag(eri).copy()
            column = lambda q: eri[:, q]                        # noqa: E731
        elif eri.ndim == 4:
            i, j = _pair_rows(nao)
            mat = eri[i[:, None], j[:, None], i[None, :], j[None, :]]
            diag = np.diag(mat).copy()
            column = lambda q: mat[:, q]                        # noqa: E731
        else:
            raise ValueError("unrecognised ERI array of shape {}".format(eri.shape))
        return cls.from_matrix(diag, column, nao, tol, max_vectors=max_vectors,
                               orbits=orbits, report=report, residence=residence,
                               note="stored AO integral array")

    @classmethod
    def from_scalar_data(cls, data, tol: float = DEFAULT_CHOLESKY_TOL, *,
                         report: bool = True, orbit_pivots: bool = True,
                         one_centre: bool = True,
                         release_eri: bool = True) -> "ThreeIndexAO":
        """Build the factors from an ingested :class:`~kuiva.interface.pyscf_bridge.
        ScalarX2CData`, taking whichever two-electron route the front-end chose.

        The Cholesky route uses **complete symmetry orbits as pivots by default**
        (:func:`shell_pair_orbits`), which is what makes the factorization's atomic spherical
        symmetry exact by construction rather than accurate to ``tol``. That needs the AO
        layout the front-end carries on ``data``; without it the decomposition falls back to
        plain column pivoting with a ``WARNING``, because the symmetry then depends on the
        threshold again and a splitting means something different in that case.

        ``one_centre=False`` groups every shell pair regardless of centre. It is measurably
        faster (2–3× on the decomposition) and costs ~3% more vectors, but ⚠ an orbit spanning
        two atoms is **not** an invariant subspace of anything, so it buys no symmetry beyond
        the one-centre grouping; it is a speed knob, not a stronger statement.

        ⚠ **On the integral-direct route the decomposition has already run**, in the front end
        while the integral evaluator was still alive, and the finished factors are what the
        container carries. This function then hands them back unchanged and ``tol``,
        ``orbit_pivots`` and ``one_centre`` here can no longer be applied — they belong to the
        front-end call on that route. A ``tol`` that disagrees with the one the factors were
        built at is reported rather than silently ignored.

        ⚠ **On the stored route the container's integral array is RELEASED here**, once the
        factors that replace it exist (``release_eri=False`` keeps it). It is the largest
        thing the container holds and nothing downstream reads it again, so carrying it is
        ``O(nao^4/8)`` of dead weight in every later phase — 7.7 GB at nao = 300 against
        factor rows of 1.2 GB, and enough to make the factor spill on this route free
        nothing that mattered. Pass ``release_eri=False`` to factorize the *same* container
        again (a second threshold, different pivots); a second call after a release is
        refused with that advice rather than served a container that looks empty.
        """
        prebuilt = getattr(data, "factors", None)
        if prebuilt is not None:
            if report:
                # The decomposition reported itself where it ran; this restates what the
                # calculation is now holding, so the section is not silent on this route.
                out.subsection(log, "Two-electron factors from the front end "
                                    "(integral-direct)")
                prebuilt.report()
            if prebuilt.tol is not None and float(prebuilt.tol) != float(tol):
                log.warning("the three-index factors were built in the front end at a "
                            "Cholesky threshold of %.2e and cannot be rebuilt here at the "
                            "%.2e asked for: the integral-direct route decomposes while the "
                            "integrals can still be evaluated, so the threshold is an "
                            "argument of the front-end call on that route.",
                            prebuilt.tol, tol)
            return prebuilt
        # The residence the front end resolved (its factors= axis, "in-core" by default) —
        # honoured here because the stored and DF routes are the ones whose decomposition
        # runs at this point; the integral-direct route spilled where it decomposed.
        residence = getattr(data, "factor_residence", "in-core")
        if getattr(data, "df_cderi", None) is not None:
            obj = cls.from_df(data.df_cderi, data.nao, aux_name=getattr(data, "aux_name", None))
            if residence in ("scratch", "streamed"):
                # ⚠ Not an oversight: the DF factors *are* the container's own array, and
                # the container keeps it (as it keeps the stored route's ERI array) so the
                # factorization can be rebuilt. Spilling this alias would free nothing
                # while telling the ledger it had — the accounting lie the budget exists
                # to prevent — so the request is declined out loud instead.
                log.warning("factors=%r has no effect on the density-fitting route: the "
                            "factor rows are the ingested container's own DF array, which "
                            "stays in memory either way, and there is no decomposition here "
                            "to run out of core. Staying in core.", residence)
            if report:
                out.subsection(log, "Density-fitted two-electron integrals")
                obj.report()
            return obj
        if getattr(data, "eri", None) is None:
            if getattr(data, "eri_released", False):
                raise ValueError(
                    "the stored AO integral array was released after the first "
                    "factorization of this container (it is dead weight once the factors "
                    "exist, and it is the largest array the container holds). To factorize "
                    "the same ingested data twice — at a second threshold, or with "
                    "different pivots — pass release_eri=False to the first call; "
                    "otherwise re-ingest with run_scalar_x2c(...).")
            raise ValueError("the ingested data carries neither conventional ERIs nor DF "
                             "factors; nothing to factorize")
        orbits = None
        if orbit_pivots:
            layout = getattr(data, "ao_layout", None)
            if layout is None:
                log.warning("no AO layout on the ingested data, so the Cholesky decomposition "
                            "falls back to plain column pivoting: the spherical symmetry of "
                            "the factorization is then only as good as the threshold "
                            "(%.0e Eh)", tol)
            else:
                orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom,
                                           one_centre=one_centre)
        obj = cls.from_eri(data.eri, data.nao, tol, orbits=orbits, report=report,
                           residence="streamed" if residence == "streamed" else "in-core")
        # ⚠ Before the spill, not after: the two are the same phase's memory, and giving
        # the larger array back first is what makes the spill's own reservation check see
        # the truth. An explicit step and a stated one — a released array is a change in
        # what the run holds, and the output file is where that is said.
        if release_eri:
            freed = data.release_eri()
            if report and freed:
                out.entry(log, "stored integral array released", freed, "GB",
                          fmt="{:.3f}",
                          note="dead weight once the factors exist; release_eri=False "
                               "keeps it for a second factorization")
        if residence == "scratch":
            obj.spill_to_scratch()
        return obj

    # -- reporting ----------------------------------------------------------------------
    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger, "factorization", self.origin)
        out.entry(logger, "AO basis functions", self.nao)
        out.entry(logger, "auxiliary / Cholesky vectors", self.naux,
                  note="{:.1f} per AO".format(self.naux / max(self.nao, 1)))
        if self.is_spilled:
            out.entry(logger, "factor residence",
                      "scratch (decomposed out of core)" if self.stream_stats else "scratch",
                      note="{:.3f} GB streamed per pass".format(self.memory_gb))
        if self.stream_stats:
            # The cost of the out-of-core route is IO, so it is stated rather than left to
            # be inferred from a wall time that is mostly the filesystem's.
            st = self.stream_stats
            out.entry(logger, "out-of-core update passes", st["passes"],
                      note="{} candidate columns, {} history rows a block".format(
                          st["candidate_columns"], st["history_rows"]))
            out.entry(logger, "out-of-core data read", st["gb_read"], "GB", fmt="{:.2f}")
        if self.aux_name:
            out.entry(logger, "auxiliary basis", self.aux_name)
        if self.tol is not None:
            out.entry(logger, "decomposition threshold", self.tol, "Eh", fmt="{:.2e}")
        if self.origin == "cholesky":
            # ⚠ Reported either way, because the *plain* path is now the notable case: the
            # orbit path is the default, and a record that does not say which was used cannot
            # be read — a free-ion multiplet splitting means something different in each.
            if self.orbit_complete:
                out.entry(logger, "pivot selection", "complete symmetry orbits",
                          note="spherical symmetry exact by construction, not to the "
                               "threshold")
            else:
                out.entry(logger, "pivot selection", "column pivoting",
                          note="spherical symmetry only as good as the threshold")
        if self.residual is not None:
            out.entry(logger, "largest neglected diagonal", self.residual, "Eh",
                      fmt="{:.3e}")
        out.entry(logger, "factor storage", self.memory_gb, "GB", fmt="{:.3f}")
        if self.origin == "df":
            # State the uncontrolled approximation rather than let it be assumed exact.
            out.note(logger, "note: the fitting error is set by the auxiliary basis;")
            out.note(logger, "      unlike a Cholesky residual it is not bounded by a")
            out.note(logger, "      threshold and is not reported per integral.")

    def __repr__(self) -> str:
        return "ThreeIndexAO(origin={}, nao={}, naux={}, residual={})".format(
            self.origin, self.nao, self.naux, self.residual)


# --- Kernels -----------------------------------------------------------------------------

def _real_or_complex(a: np.ndarray) -> np.ndarray:
    """Return a real view when the imaginary part is *exactly* zero, else the array itself.

    The point is not the memory: it lets the whole GEMM chain below stay in real arithmetic
    (4x fewer flops) whenever the coefficients happen to be real — the SOC-free guess, and
    every scalar cross-check in the test suite. Only an exact zero qualifies; discarding a
    small-but-nonzero imaginary part would be an approximation, and this module makes none.
    """
    if np.iscomplexobj(a) and not a.imag.any():
        return np.ascontiguousarray(a.real)
    return a


def _lgemm(lm: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``lm @ c`` with ``lm`` real and ``c`` possibly complex — the hottest kernel here.

    A complex ``c`` can be handled two ways, and the obvious one is a trap:

    * ``(lm @ c.real) + 1j * (lm @ c.imag)`` halves the flops on paper. **Do not do this.**
      ``c.real`` and ``c.imag`` are *strided views* of the complex array, and BLAS falls off a
      cliff on them — measured at **1.1 GFLOP/s against 36 for a plain complex GEMM**, i.e.
      the "optimization" was a 16x pessimization. It was in this file, documented as a
      speedup, until it was profiled: it accounted for **85% of a Hessian-vector product**.
    * Packing ``[Re c | Im c]`` into one *contiguous* real matrix and doing a **single** real
      GEMM keeps the halved flop count and full BLAS efficiency — 49 GFLOP/s.

    The packing costs an ``O(nao * k)`` copy and an ``O(m * k)`` strided recombination, which
    pays only while ``2k`` stays under the contraction length; past that the plain complex
    GEMM wins. Measured (m = 37380, nao = 105): k = 20 and 40 favour the packed form by 2-3x,
    k = 128 favours the complex GEMM by 1.5x. Hence the switch below.

    ⚠ This is exactly the case the stability rule warns about — an optimization that reduced
    the operation count while destroying performance, invisible until measured. Profile before
    believing any transformation of this kernel.
    """
    if not np.iscomplexobj(c):
        return lm @ c
    k = c.shape[1]
    if 2 * k > lm.shape[1]:
        return lm @ c
    both = np.empty((c.shape[0], 2 * k), dtype=np.float64)
    both[:, :k] = c.real
    both[:, k:] = c.imag
    out = lm @ both
    return out[:, :k] + 1j * out[:, k:]


def transform_buffer_gb(nao: int, nket: int, naux: int) -> float:
    """Buffer [GB] :func:`transform_3c` would use *unblocked* (exact sizing function).

    The unpacked ``L`` block plus one intermediate, for every auxiliary index at once. The
    kernel never needs more than this, so a pre-flight must plan for
    ``min(this, transient budget)`` — planning for the budget alone would report a Ne atom as
    needing gigabytes and refuse calculations that fit comfortably.
    """
    per_aux = nao * nao * 8.0 + nao * nket * 16.0
    return naux * per_aux / res.BYTES_PER_GB


def _buffer_gb(buffer_gb: Optional[float]) -> float:
    """Resolve a kernel's temporary-buffer budget.

    ``None`` — the default — asks the process budget for a share of what is not already
    reserved, so a fat node gets large blocks and a laptop gets small ones. One call per
    kernel invocation, outside the loop: memory management never touches a hot path.
    """
    return res.transient_gb() if buffer_gb is None else float(buffer_gb)


def _aux_blocksize(nao: int, nket: int, naux: int, buffer_gb: float,
                   stream_row_bytes: float = 0.0) -> int:
    """Auxiliary block size that keeps the transform's temporaries inside ``buffer_gb``.

    ``stream_row_bytes`` is :attr:`ThreeIndexAO.stream_row_bytes` — the packed read buffers
    a spilled store holds per row beside the consumer's own temporaries; zero in-core.
    """
    per_aux = nao * nao * 8.0 + nao * nket * 16.0        # unpacked L block + intermediate
    nb = int(buffer_gb * 1024.0 ** 3 / max(per_aux + stream_row_bytes, 1.0))
    return max(1, min(naux, nb))


def transform_3c(factors: ThreeIndexAO, c_bra: np.ndarray, c_ket: np.ndarray, *,
                 aux_blocksize: Optional[int] = None,
                 buffer_gb: Optional[float] = None,
                 dtype=np.complex128,
                 out_array: Optional[np.ndarray] = None) -> np.ndarray:
    """Transform the three-index factors into a spinor-MO block: ``B^P_{pq}``.

    Parameters
    ----------
    factors : ThreeIndexAO
        AO factors (DF or Cholesky — the caller does not care which).
    c_bra, c_ket : ndarray (2*nao, n), complex or real
        Spinor coefficient blocks **in the AO basis**, rows blocked ``[alpha; beta]``
        (:mod:`kuiva.spinor.expand`). Coefficients in the orthonormal working basis must be
        taken back to AO first (``SpinorBasis.transform_scalar_basis(orth.x)``), which costs
        one small GEMM and keeps the factors in their natural basis.

    dtype :
        Dtype of the **result**, ``complex128`` by default and independent of whether the
        coefficients happened to be real — a function whose output dtype depends on the
        numerical content of its input is a trap for every caller downstream (complex is
        first-class). The internal arithmetic still drops to real automatically when it can
        (see :func:`_real_or_complex`); only the final store is cast. Pass ``float64``
        deliberately when the caller knows the result is real, e.g. a scalar cross-check.

    Returns
    -------
    ndarray (naux, nbra, nket) of ``dtype``

    Notes
    -----
    Per auxiliary block and per spin component: one ``(nb*nao, nao) x (nao, nket)`` GEMM and
    one batched ``(nbra, nao) x (nao, nket)`` GEMM. Both are BLAS-3 on contiguous buffers,
    and the loop bound is memory, not the auxiliary dimension. Cost
    ``O(naux * nao * nket * (nao + nbra))`` — the transform is linear in the auxiliary
    dimension, which is the entire reason for factorizing.
    """
    nao = factors.nao
    if c_bra.shape[0] != 2 * nao or c_ket.shape[0] != 2 * nao:
        raise ValueError("spinor coefficients must have 2*nao = {} rows, got {} and {}"
                         .format(2 * nao, c_bra.shape[0], c_ket.shape[0]))
    nbra, nket = int(c_bra.shape[1]), int(c_ket.shape[1])
    naux = factors.naux

    bra_a = _real_or_complex(np.ascontiguousarray(c_bra[:nao]))
    bra_b = _real_or_complex(np.ascontiguousarray(c_bra[nao:]))
    ket_a = _real_or_complex(np.ascontiguousarray(c_ket[:nao]))
    ket_b = _real_or_complex(np.ascontiguousarray(c_ket[nao:]))
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.complexfloating) and \
            any(np.iscomplexobj(a) for a in (bra_a, bra_b, ket_a, ket_b)):
        raise ValueError("a real result dtype was requested but the spinor coefficients are "
                         "complex; the imaginary part would be discarded silently")

    size_gb = mo_block_memory_gb(naux, nbra, nket, dtype)
    if out_array is None:
        # checked before the allocation, with the knobs that change it. Only when the
        # array is ours — an out_array belongs to the caller and is already accounted for.
        res.require("three-index MO block B^P_pq", size_gb,
                    note="naux={} x {} x {} spinors, {}".format(naux, nbra, nket, dtype.name),
                    advice=["transform a smaller orbital block (one active index rather "
                            "than all spinors) — see mcscf.orbopt, which needs only B^P_pt",
                            "reduce the auxiliary/Cholesky dimension: a looser Cholesky "
                            "threshold trades integral accuracy for naux, and it is an "
                            "error bound rather than a knob - loosening it is a physics "
                            "decision, so read what the default is set from first",
                            "use a smaller basis set"])

    if out_array is None:
        b = np.empty((naux, nbra, nket), dtype=dtype)
    else:
        b = out_array
        if b.shape != (naux, nbra, nket) or b.dtype != dtype:
            raise ValueError("out_array has shape {}/{} but the result is {}/{}".format(
                b.shape, b.dtype, (naux, nbra, nket), dtype))
    nb = int(aux_blocksize or _aux_blocksize(nao, nket, naux, _buffer_gb(buffer_gb),
                                             factors.stream_row_bytes))
    log.debug("3-index transform: naux=%d nao=%d (%d x %d) block=%d dtype=%s %.3f GB",
              naux, nao, nbra, nket, nb, dtype.name, size_gb)

    with timer("3-index transform"):
        for p0 in range(0, naux, nb):
            p1 = min(p0 + nb, naux)
            lblk = factors.unpack(slice(p0, p1))                  # (nb, nao, nao) real
            lmat = lblk.reshape(-1, nao)
            acc = None
            for cb, ck in ((bra_a, ket_a), (bra_b, ket_b)):
                tmp = _lgemm(lmat, ck).reshape(p1 - p0, nao, -1)  # (nb, nao, nket)
                # Batched GEMM over the auxiliary index; broadcasting keeps it BLAS-3 with
                # no explicit copy of the (nb, nao, nket) intermediate.
                part = np.matmul(cb.conj().T, tmp)                # (nb, nbra, nket)
                acc = part if acc is None else acc + part
            b[p0:p1] = acc
    return b


def half_transform_memory_gb(naux: int, nao: int, k: int) -> float:
    """Size [GB] of one :func:`half_transform` result (exact sizing function).

    Two spin blocks of ``(naux, nao, k)`` complex. This is what a caller trades for skipping
    the right-hand half transform of every subsequent :func:`coulomb_exchange` call against
    the same factors — the orbital Hessian's response builds are the consumer.
    """
    return 2.0 * naux * nao * k * 16.0 / res.BYTES_PER_GB


def half_transform(factors: ThreeIndexAO, orbitals: np.ndarray, *,
                   buffer_gb: Optional[float] = None) -> List[np.ndarray]:
    """The occupied-side half transform ``W^P_{mi} = sum_l L^P_{ml} c_{li}``, per spin block.

    Returns ``[W_alpha, W_beta]``, each ``(naux, nao, k)`` and always ``complex128`` (the
    result dtype never depends on the numerical content of the input — see
    :func:`transform_3c`). This is the first stage of :func:`coulomb_exchange`, split out so
    a caller that runs **many** J/K builds against the *same* right-hand factors — one per
    Davidson expansion of an orbital Hessian-vector product, at fixed orbitals — can pay it
    once and pass the result back as ``right_half=``. The cache is a statement about one set
    of orbitals; it must be rebuilt whenever they change.
    """
    nao = factors.nao
    orbitals = np.asarray(orbitals)
    if orbitals.ndim != 2 or orbitals.shape[0] != 2 * nao:
        raise ValueError("orbitals must be ({}, k) in the two-component AO basis, got {}"
                         .format(2 * nao, orbitals.shape))
    k_orb = orbitals.shape[1]
    naux = factors.naux
    spin = [_real_or_complex(np.ascontiguousarray(orbitals[:nao])),
            _real_or_complex(np.ascontiguousarray(orbitals[nao:]))]
    out_w = [np.empty((naux, nao, k_orb), dtype=np.complex128) for _ in range(2)]
    nb = max(1, min(naux, int(_buffer_gb(buffer_gb) * 1024.0 ** 3 /
                              max(nao * nao * 8.0 + 2 * nao * k_orb * 16.0
                                  + factors.stream_row_bytes, 1.0))))
    with timer("J/K half transform"):
        for p0 in range(0, naux, nb):
            p1 = min(p0 + nb, naux)
            lmat = factors.unpack(slice(p0, p1)).reshape(-1, nao)
            for s in (0, 1):
                out_w[s][p0:p1] = _lgemm(lmat, spin[s]).reshape(p1 - p0, nao, k_orb)
    return out_w


def diagonal_pair_blocks(factors: ThreeIndexAO, c: np.ndarray, cols: np.ndarray, *,
                         rows: Optional[np.ndarray] = None,
                         half: Optional[Tuple[np.ndarray, Sequence[np.ndarray]]] = None,
                         buffer_gb: Optional[float] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """``(b_diag, s2)``: the diagonal of the MO three-index block, and exchange-type sums.

    * ``b_diag[P, p] = B^P_pp`` — real, because the AO factors are real symmetric; always
      over **every** orbital ``p``.
    * ``s2[k, j] = sum_P |B^P_{rows[k], cols[j]}|^2 = (p c_j | c_j p)`` — the
      exchange-type diagonal two-electron integrals of the ``rows`` orbitals (all of them
      when ``rows`` is None) against the ``cols`` orbitals; real and non-negative.

    Together with an active block ``B^P_{p,t}`` these are what an **exact orbital-Hessian
    diagonal** needs: ``(pp|qq) = sum_P b_diag[P,p] b_diag[P,q]`` and ``(pq|qp) = s2`` —
    and ``rows`` exists because that consumer needs the exchange sums only for the pair
    classes not already covered by its resident active block (the virtual rows), which is
    the larger of the two GEMM stages here.

    ``half = (indices, [W_alpha, W_beta])`` supplies :func:`half_transform` output for the
    ``indices`` columns of ``c`` — the orbital Hessian already holds it for the occupied
    columns — so the streamed stage only transforms the complement. ``cols`` must be a
    subset of ``indices`` when ``half`` is given: the s2 stage reads its ket columns from
    the cache. The arrays are trusted to be what they claim, exactly as
    :func:`coulomb_exchange`'s ``right_half`` is.

    One streamed pass over the auxiliary index — nothing of size ``(naux, n, n)`` is ever
    stored, so the square-block rule stands.
    """
    nao = factors.nao
    c = np.asarray(c)
    if c.ndim != 2 or c.shape[0] != 2 * nao:
        raise ValueError("coefficients must be ({}, n) in the two-component AO basis, got {}"
                         .format(2 * nao, c.shape))
    n = c.shape[1]
    cols = np.asarray(cols, dtype=int).ravel()
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=int).ravel()
    naux = factors.naux
    if half is not None:
        have = np.asarray(half[0], dtype=int).ravel()
        w_have = half[1]
        if len(w_have) != 2 or any(
                np.asarray(w).shape != (naux, nao, have.size) for w in w_have):
            raise ValueError("half must be (indices, [W_alpha, W_beta]) with W of shape "
                             "{} from half_transform over those columns"
                             .format((naux, nao, have.size)))
        missing = np.setdiff1d(cols, have)
        if missing.size:
            raise ValueError("cols must be a subset of half's columns; {} are not"
                             .format(missing.size))
        pos_in_have = {int(p): k for k, p in enumerate(have)}
        col_pos = np.array([pos_in_have[int(q)] for q in cols], dtype=int)
        todo = np.setdiff1d(np.arange(n), have)
    else:
        have = np.zeros(0, dtype=int)
        w_have = None
        col_pos = cols
        todo = np.arange(n)
    spin = [_real_or_complex(np.ascontiguousarray(c[:nao])),
            _real_or_complex(np.ascontiguousarray(c[nao:]))]
    conj_spin = [np.conj(s) for s in spin]
    conj_rows = [np.ascontiguousarray(cs[:, rows]) for cs in conj_spin]
    spin_todo = [np.ascontiguousarray(s[:, todo]) for s in spin]
    conj_todo = [np.ascontiguousarray(cs[:, todo]) for cs in conj_spin]
    conj_have = [np.ascontiguousarray(cs[:, have]) for cs in conj_spin]
    b_diag = np.zeros((naux, n), dtype=np.float64)
    s2 = np.zeros((rows.size, cols.size), dtype=np.float64)
    per_aux = (nao * nao * 8.0 + 2.0 * nao * max(todo.size, 1) * 16.0
               + 2.0 * rows.size * max(cols.size, 1) * 16.0 + factors.stream_row_bytes)
    nb = max(1, min(naux, int(_buffer_gb(buffer_gb) * 1024.0 ** 3 / max(per_aux, 1.0))))
    with timer("Hessian diagonal integrals"):
        for p0 in range(0, naux, nb):
            p1 = min(p0 + nb, naux)
            nblk = p1 - p0
            lmat = None
            if todo.size:
                lmat = factors.unpack(slice(p0, p1)).reshape(-1, nao)
            b_ket = np.zeros((nblk, rows.size, cols.size), dtype=np.complex128) \
                if cols.size and rows.size else None
            for s in (0, 1):
                if todo.size:
                    w = _lgemm(lmat, spin_todo[s]).reshape(nblk, nao, todo.size)
                    b_diag[p0:p1, todo] += np.real(
                        conj_todo[s][None, :, :] * w).sum(axis=1)
                if half is not None and have.size:
                    wc = w_have[s][p0:p1]
                    b_diag[p0:p1, have] += np.real(
                        conj_have[s][None, :, :] * wc).sum(axis=1)
                if b_ket is not None:
                    w_cols = (w_have[s][p0:p1][:, :, col_pos] if half is not None
                              else w[:, :, col_pos])
                    b_ket += np.matmul(conj_rows[s].T, w_cols)
            if b_ket is not None:
                s2 += (np.abs(b_ket) ** 2).sum(axis=0)
    return b_diag, s2


def coulomb_exchange(factors: ThreeIndexAO, orbitals: np.ndarray,
                     occupations: Optional[np.ndarray] = None, *,
                     orbitals_right: Optional[np.ndarray] = None,
                     right_half: Optional[Sequence[np.ndarray]] = None,
                     buffer_gb: Optional[float] = None,
                     with_k: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Coulomb and exchange matrices in the two-component AO basis.

    The density is given in **factorized** form,
    ``D = sum_i n_i c_i d_i^dag`` — ``orbitals`` supplies the ``c_i`` (shape ``(2*nao, k)``,
    rows blocked ``[alpha; beta]``), ``orbitals_right`` the ``d_i`` (default: the same,
    giving the Hermitian positive-semidefinite case), and ``occupations`` the ``n_i``.
    Returns ``(J, K)`` of shape ``(2*nao, 2*nao)`` with

        J^{st}_{mn} = delta_st sum_{lk} (mn|lk) sum_u D^{uu}_{kl}
        K^{st}_{mn} = sum_{lk} (ml|kn) D^{st}_{lk}

    Two things about this are load-bearing.

    **The spin structure.** The Coulomb operator is spin-free, so **J is spin-diagonal and
    sees only the spin-traced density**, while **K carries all four spin blocks** — including
    the off-diagonal ones, which is precisely how spin-orbit coupling in the orbitals feeds
    back into the mean field. Dropping them (the scalar-code habit) would silently make the
    Fock matrix spin-free again.

    **The occupied index is contracted first.** Building ``W^P_{mi} = sum_l L^P_{ml} c_{li}``
    and then ``K_{mn} = sum_{Pi} W^P_{mi} conj(W^P_{ni})`` costs ``O(naux * nao^2 * k)``
    where forming ``L D L`` per auxiliary vector costs ``O(naux * nao^3)`` — an order of
    magnitude for a typical inactive space, and the reason this takes a factorized density
    rather than a matrix. It is also why every contraction below is written as an explicit
    reshape-and-GEMM: ``einsum`` does **not** dispatch these batched contractions to BLAS, and
    left as einsum this single function measured **88% of a CASSCF macro-iteration, running
    single-threaded**.

    A density that is not idempotent (an active-space 1-RDM) still factorizes: diagonalize it
    and fold the occupations in, which is what :func:`kuiva.mcscf.orbopt._active_fock` does.
    A density that is not even *positive* — the transition densities of a Hessian-vector
    product, ``X C^dag + C X^dag`` — is why ``orbitals_right`` exists: pass
    ``[X | C]`` and ``[C | X]`` and the same builder handles it, with no eigendecomposition
    and no square roots of negative numbers. (The orbital Hessian in fact passes only the
    one-sided ``X`` / ``C`` pair and Hermitizes the result — J and K are linear in the
    density and map ``D^dag`` to the Hermitian adjoint over real AO integrals.)

    ``right_half`` supplies the half transform of ``orbitals_right`` precomputed by
    :func:`half_transform` — for a caller that builds J/K repeatedly against the same
    right-hand factors. It replaces only the K path's half transform; ``orbitals_right`` must
    still be given, because the Coulomb contraction uses the raw coefficients. The arrays are
    trusted to *be* the half transform of ``orbitals_right`` — only their shape is checked,
    since recomputing them for comparison would cost exactly what they exist to save.
    """
    nao = factors.nao
    orbitals = np.asarray(orbitals)
    if orbitals.ndim != 2 or orbitals.shape[0] != 2 * nao:
        raise ValueError("orbitals must be ({}, k) in the two-component AO basis, got {}"
                         .format(2 * nao, orbitals.shape))
    right = orbitals if orbitals_right is None else np.asarray(orbitals_right)
    if right.shape != orbitals.shape:
        raise ValueError("orbitals_right must match orbitals in shape, got {} and {}"
                         .format(right.shape, orbitals.shape))
    k_orb = orbitals.shape[1]
    if right_half is not None:
        if orbitals_right is None:
            raise ValueError("right_half needs orbitals_right too: the Coulomb contraction "
                             "uses the raw coefficients, only the exchange half transform "
                             "is replaced")
        expect = (factors.naux, nao, k_orb)
        if (len(right_half) != 2
                or any(np.asarray(w).shape != expect for w in right_half)):
            raise ValueError("right_half must be the two spin blocks of shape {} from "
                             "half_transform(factors, orbitals_right)".format(expect))
    # Fold the occupations into one factor rather than taking square roots: it is exact for
    # *any* sign of n_i, which matters because a transition density is indefinite.
    cw = orbitals if occupations is None else orbitals * np.asarray(occupations, dtype=float)
    dw = right
    same_factors = occupations is None and orbitals_right is None
    naux = factors.naux
    j = np.zeros((nao, nao), dtype=np.complex128)
    k = np.zeros((2 * nao, 2 * nao), dtype=np.complex128) if with_k else None
    if k_orb == 0:
        return np.zeros((2 * nao, 2 * nao), dtype=np.complex128), k

    cw_spin = [_real_or_complex(np.ascontiguousarray(cw[:nao])),
               _real_or_complex(np.ascontiguousarray(cw[nao:]))]
    dw_spin = cw_spin if same_factors else [
        _real_or_complex(np.ascontiguousarray(dw[:nao])),
        _real_or_complex(np.ascontiguousarray(dw[nao:]))]
    nb = max(1, min(naux, int(_buffer_gb(buffer_gb) * 1024.0 ** 3 /
                              max(nao * nao * 8.0 + 4 * nao * k_orb * 16.0
                                  + factors.stream_row_bytes, 1.0))))
    with timer("J/K build"):
        for p0 in range(0, naux, nb):
            p1 = min(p0 + nb, naux)
            nblk = p1 - p0
            lblk = factors.unpack(slice(p0, p1))                  # (nblk, nao, nao) real
            lmat = lblk.reshape(-1, nao)                          # (nblk*nao, nao)
            lflat = lblk.reshape(nblk, -1)                        # (nblk, nao*nao)
            # W^P_{mi} = sum_l L^P_{ml} c_{li}, per spin component (and the same for d).
            wc = [_lgemm(lmat, c).reshape(nblk, nao, k_orb) for c in cw_spin]
            if right_half is not None:
                wd = [w[p0:p1] for w in right_half]
            else:
                wd = wc if dw_spin is cw_spin else [
                    _lgemm(lmat, d).reshape(nblk, nao, k_orb) for d in dw_spin]
            # Coulomb: d_P = sum_{s,m,i} conj(d^s_{mi}) Wc^s_{Pmi}, then J = sum_P d_P L^P.
            d_p = np.zeros(nblk, dtype=np.complex128)
            for d, ws in zip(dw_spin, wc):
                d_p += ws.reshape(nblk, -1) @ np.conj(np.ascontiguousarray(d)).ravel()
            # Not projected onto the reals: d_P is real for a Hermitian density (every
            # physical use here), and for a general one the complex value is the right answer.
            j += (d_p @ lflat).reshape(nao, nao)
            if not with_k:
                continue
            fc = [np.ascontiguousarray(ws.transpose(1, 0, 2)).reshape(nao, -1) for ws in wc]
            fd = fc if wd is wc else [
                np.ascontiguousarray(ws.transpose(1, 0, 2)).reshape(nao, -1) for ws in wd]
            for s in (0, 1):
                for t in (0, 1):
                    k[s * nao:(s + 1) * nao, t * nao:(t + 1) * nao] += \
                        fc[s] @ fd[t].conj().T
    j_2c = np.zeros((2 * nao, 2 * nao), dtype=np.complex128)
    j_2c[:nao, :nao] = j
    j_2c[nao:, nao:] = j
    return j_2c, k


def assemble_4c(b_pq: np.ndarray, b_rs: Optional[np.ndarray] = None) -> np.ndarray:
    """Assemble four-index integrals ``(pq|rs) = sum_P B^P_{pq} B^P_{rs}`` (chemists' notation).

    .. warning::
       There is **no complex conjugation** in this contraction. The bra index of each electron
       pair was already conjugated when ``B`` was built, so writing this as an inner product
       (``einsum("Ppq,Prs->pqrs", B.conj(), B)`` or ``np.tensordot(B.conj(), B, ...)``) gives
       a wrong, Hermitian-looking, physically plausible answer. The symmetry that does hold
       is ``(pq|rs) = (qp|sr)^*`` and ``(pq|rs) = (rs|pq)`` — 4-fold, not 8-fold — and
       :func:`check_permutational_symmetry` verifies it.
    """
    b_rs = b_pq if b_rs is None else b_rs
    return np.tensordot(b_pq, b_rs, axes=([0], [0]))


def check_permutational_symmetry(eri: np.ndarray) -> Tuple[float, float]:
    """Return ``(max |(pq|rs) - (qp|sr)^*|, max |(pq|rs) - (rs|pq)|)``.

    The 4-fold symmetry of complex spinor two-electron integrals. A Tier-0 check, and
    cheap enough to run on active-space blocks in debug builds.
    """
    eri = np.asarray(eri)
    herm = float(np.max(np.abs(eri - np.conj(np.transpose(eri, (1, 0, 3, 2))))))
    swap = float(np.max(np.abs(eri - np.transpose(eri, (2, 3, 0, 1)))))
    return herm, swap


def transform_1e(h: np.ndarray, c_bra: np.ndarray, c_ket: Optional[np.ndarray] = None
                 ) -> np.ndarray:
    """One-electron AO -> spinor-MO transformation.

    ``h`` may be

    * ``(nao, nao)``       — a **spin-free** operator (the scalar X2C one-electron
      Hamiltonian, the overlap): implicitly ``1_2 (x) h``, contracted per spin component;
    * ``(2*nao, 2*nao)``   — a full two-component operator (e.g. spin-free plus spin-orbit,
      assembled by :func:`kuiva.spinor.expand.two_component_operator`).

    Handling both is what lets spin-orbit coupling be switched on at this level without
    touching any caller: the same call transforms the SOC-free and the SOC Hamiltonian.
    """
    h = np.asarray(h)
    c_ket = c_bra if c_ket is None else c_ket
    n2 = c_bra.shape[0]
    if h.shape[0] == n2:
        return c_bra.conj().T @ h @ c_ket
    nao = n2 // 2
    if h.shape[0] != nao:
        raise ValueError("one-electron operator of shape {} matches neither the scalar basis "
                         "({}) nor the two-component basis ({})".format(h.shape, nao, n2))
    return (c_bra[:nao].conj().T @ h @ c_ket[:nao] +
            c_bra[nao:].conj().T @ h @ c_ket[nao:])


# --- The assembled integral set ----------------------------------------------------------

@dataclass
class SpinorMOIntegrals:
    """One- and two-electron integrals over a chosen set of spinors.

    The two-electron integrals are kept in **factored** form (``b``): the four-index block for
    a subspace is assembled on demand by :meth:`eri`. For a CAS of 14 spinors the four-index
    array is 0.6 MB and could be stored, but the same object has to serve the CASSCF gradient
    blocks and the NEVPT2 classes over the full orbital space, where it cannot — so the
    factored form is the interface everywhere, and nothing downstream is written against a
    stored four-index array.

    ⚠ **This is the deliberately *unblocked* reference implementation, and it is not on any
    production path — a decision, not an oversight.** ``b`` is square, ``naux * n * n`` over
    whichever spinors it was handed, and both :meth:`eri` (which slices it in both indices)
    and :meth:`hermiticity_error` (which compares it against its own transpose) are written
    against that squareness. That is what makes the object worth having: it is the form in
    which the permutational and time-reversal invariants of a whole spinor set can be checked
    in one line, and those checks are what license every blocked path below it.

    What it is **not** is the way to hold the integrals of a real calculation. The cost is
    quadratic in the spinor count, so ``n`` here is a *chosen subset* — pass a block of
    orbitals, never the whole set of a heavy-element system, where the array reaches hundreds
    of gigabytes and the memory check will refuse it. Production holds the blocks it needs and
    nothing more: the orbital optimizer and the CI drivers hold ``B^P_{p,active}``
    (``mcscf.orbopt.CASIntegrals``), and the perturbation holds one block per space pair its
    classes ask for (``pt.blocks.IntegralBlocks``). Both are built by the same
    :func:`transform_3c` this class calls, with a narrower ket.
    """

    h: np.ndarray                                  # (n, n) complex, one-electron
    b: np.ndarray                                  # (naux, n, n) three-index two-electron
    e_nuc: float = 0.0
    label: str = ""

    @property
    def n(self) -> int:
        return int(self.h.shape[0])

    @property
    def naux(self) -> int:
        return int(self.b.shape[0])

    def eri(self, p=slice(None), q=None, r=None, s=None) -> np.ndarray:
        """Assemble ``(pq|rs)`` for the given index selections (default: everything)."""
        q = p if q is None else q
        r = p if r is None else r
        s = q if s is None else s
        return assemble_4c(self.b[:, p, :][:, :, q], self.b[:, r, :][:, :, s])

    @classmethod
    def build(cls, factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray, *,
              e_nuc: float = 0.0, label: str = "", report: bool = True,
              buffer_gb: Optional[float] = None) -> "SpinorMOIntegrals":
        """Transform both integral classes into the spinor space spanned by ``c_spinor``.

        ``c_spinor`` is ``(2*nao, n)`` in the **AO** basis (see :func:`transform_3c`).
        """
        with timer("integral transformation") as t:
            h_mo = transform_1e(h_ao, c_spinor)
            b = transform_3c(factors, c_spinor, c_spinor, buffer_gb=buffer_gb)
        obj = cls(h=h_mo, b=b, e_nuc=e_nuc, label=label)
        # transform_3c checked this before allocating; record it as resident now that
        # the object owns it, so what comes next is measured against the truth.
        res.reserve("spinor MO integrals B^P_pq" + (" ({})".format(label) if label else ""),
                    res.array_gb(b.shape, b.dtype),
                    note="naux={} x {} spinors".format(b.shape[0], b.shape[1]))
        if report:
            out.subsection(log, "Integral transformation to the spinor MO basis"
                           + (" ({})".format(label) if label else ""))
            obj.report()
            out.entry(log, "transformation time", t.wall, "s wall", fmt=out.TIME_FMT)
            out.entry(log, "transformation cost", t.cpu, "s cpu", fmt=out.TIME_FMT)
        return obj

    def hermiticity_error(self) -> float:
        """``max |h - h^dag|`` plus the three-index hermiticity ``max |B^P - (B^P)^dag|``.

        Both must vanish to machine precision for a Hermitian Hamiltonian in an orthonormal
        spinor basis; a nonzero value means the coefficients, the operator, or the spin
        blocking are inconsistent. Cheap, and worth asserting whenever this object is built
        in a test.
        """
        eh = float(np.max(np.abs(self.h - self.h.conj().T)))
        eb = float(np.max(np.abs(self.b - np.conj(np.transpose(self.b, (0, 2, 1))))))
        return max(eh, eb)

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger, "spinors transformed", self.n)
        out.entry(logger, "auxiliary dimension", self.naux)
        out.entry(logger, "three-index storage", self.b.nbytes / 1024.0 ** 3, "GB",
                  fmt="{:.3f}")
        out.entry(logger, "four-index storage if assembled",
                  self.n ** 4 * self.b.itemsize / 1024.0 ** 3, "GB", fmt="{:.3f}",
                  note="not allocated")
        out.entry(logger, "hermiticity error", self.hermiticity_error(), fmt="{:.2e}")

    def __repr__(self) -> str:
        return "SpinorMOIntegrals(n={}, naux={}, label={!r})".format(
            self.n, self.naux, self.label)


__all__ = ["ThreeIndexAO", "SpinorMOIntegrals", "pivoted_cholesky", "shell_pair_orbits",
           "transform_3c", "transform_1e", "assemble_4c", "coulomb_exchange",
           "check_permutational_symmetry", "npair_of", "factor_memory_gb",
           "half_transform", "half_transform_memory_gb", "diagonal_pair_blocks",
           "mo_block_memory_gb", "DEFAULT_CHOLESKY_TOL", "DEFAULT_BUFFER_GB",
           "ORBIT_DEGENERACY_RTOL", "ORBIT_STABILITY_RTOL"]
