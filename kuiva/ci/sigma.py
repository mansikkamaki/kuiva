"""Full-CI sigma vector over a complete CAS space of complex spinors.

The algorithm: two-step ``E_pq`` resolution with one dense GEMM
---------------------------------------------------------------
Write the Hamiltonian of ``ci/strings.py`` in one-particle-excitation form. With
``E_pq = a_p^dag a_q`` (spinor, **not** spin-summed — there is no alpha/beta factorization,
because spin-orbit coupling breaks spin symmetry) the normal-ordered two-electron
operator is ``a_p^dag a_r^dag a_s a_q = E_pq E_rs - delta_qr E_ps``, so::

    H = sum_pq  h~_pq E_pq  +  1/2 sum_pqrs (pq|rs) E_pq E_rs ,
    h~_pq = h_pq - 1/2 sum_r (pr|rq)

⚠ **The ``h~`` folding is the quiet trap of this module.** Getting ``-1/2 sum_r (pr|rq)``
wrong is invisible to every structural test — hermiticity, Kramers degeneracy and the RDM
trace conditions all survive it — and it surfaces later as an energy offset that looks like a
basis or threshold effect. It is caught only by comparison against a genuinely independent
implementation, which is why ``tests/test_ci_sigma.py`` validates against
:func:`kuiva.ci.strings.hamiltonian_matrix` (different algorithm, different sign bookkeeping)
rather than against the algebra in isolation.

Over the determinant space, with ``K`` an intermediate determinant index::

    step 1  (gather)  F[K, p, q] = sum_J <K|E_pq|J> c_J
    step 2  (GEMM)    G[K, p, q] = sum_rs (pq|rs) F[K, r, s]
    step 3  (gather)  sigma_I    = sum_pq h~_pq F[I, p, q]
                                 + 1/2 sum_K sum_pq <I|E_pq|K> G[K, p, q]

**Step 2 is one dense complex GEMM of shape ``(N x n^2) . (n^2 x n^2)``**, and it is where the
great majority of the arithmetic lives — already at MKL speed through NumPy. That is the whole
performance argument for this formulation.

Index convention for ``F`` and ``G``, fixed here
------------------------------------------------
⚠ ``F[K, p, q]`` is indexed by the operator ``E_pq``, so ``p`` is the orbital **annihilated
from K** on the way to the hole string and ``q`` the one **created** to reach ``J``. That
follows from ``<K|E_pq|J> = <J|E_qp|K>``: applying ``a_p`` to ``K`` and then ``a_q^dag``
reaches ``J`` with exactly that matrix element. Getting this transposed leaves a Hermitian,
plausible, wrong operator, so it is stated here and asserted in the tests against
:func:`kuiva.ci.strings.single_excitation_operator`.

Decisions that constrain the rest of the code
----------------------------------------------
* ⚠ **The one-electron term is row-local and stays out of the GEMM.** ``sum_pq h~_pq F[I,pq]``
  contracts ``F``'s own row ``I``; folding it into ``G`` would put it behind a second ``E_pq``
  and turn it into a double-excitation structure. Both terms read ``F``; only the second
  passes through the GEMM.
* ⚠ **Steps 1 and 3 are both gathers over the output row, using the same map in the same
  direction.** For output row ``I`` the loop runs over its ``k(n-k+1)`` excitations and reads:
  from ``c`` in step 1, from ``G`` in step 3. Every write is to the row the loop owns, so
  there is no accumulation conflict, no ``np.add.at``, and **no atomics or per-thread
  privatization in a threaded C++ port**. This is a portability decision as much as a NumPy
  one, and it is why the two steps are separate registered kernels rather than one.
* ⚠ **That costs full residency of ``G``, and residency — not flops — sets the ceiling.** The
  ``K`` connected to a given ``I`` are scattered across the whole space, so the step-3 gather
  needs all of ``G`` in memory. ⚠ ``F`` costs no *second* buffer: once the row-local
  one-electron term is banked, nothing needs ``F`` after the GEMM, so ``G`` overwrites it in
  row tiles (BLAS forbids an aliased product; see :func:`sigma_workspace_gb`).
  :func:`sigma_workspace_gb` is exact and is reserved before the first allocation; above
  roughly 24 spinors it refuses, and the refusal says why, because the alternative there is
  *a different kernel with a different parallelization* (batched over ``K``, hence scattering
  where this one gathers) and not a larger run of this one. Without that message the failure
  mode is an OOM that looks like a machine problem rather than an algorithmic ceiling.
* **``F`` is per CI vector.** A block Davidson expands several directions per iteration; they
  are applied one at a time, so the resident cost stays ``N n^2`` (plus the tile) and not
  ``n_roots`` times that.
* ⚠ **4-fold permutational symmetry only**. The ``n^2 x n^2`` reshape must respect
  ``(pq|rs) = (rs|pq) = (qp|sr)*`` and **must not assume 8-fold** — with complex spinors
  ``(pq|rs) != (rq|ps)`` in general. Both 4-fold relations are asserted on construction; a
  test feeds an integral set that has exactly 4-fold symmetry and no more, so an 8-fold
  assumption creeping in would fail rather than merely be unused.

Profiling
---------
Steps 1, 2 and 3 are timed as **separate regions** from the outset, because the measurement
that decides whether the gathers are worth porting to C++ is not obtainable if they are one
region — and the two gathers are not alike. Step 1 reads ``c``, a few MB that sits in cache;
**step 3 reads ``G`` fully randomly**, which at 20 spinors is 1.1 GB touched 16 bytes at a
time, one DRAM latency per excitation with only memory-level parallelism to hide it. Rank
them on **CPU seconds** (wall time on the development machine is partly a thermal
measurement).

⚠ Everything in this module outside the two registered kernels is **orchestration and stays
Python permanently** — the operator class, the symmetry assertions, the budgeting, the
timing. the unprofiled-optimization warning about unprofiled optimization cuts both ways: porting the wrapper
would buy nothing and cost the readability the kernels' correctness rests on.

References
----------
* Two-step ``E_pq`` resolution of the CI sigma vector: B. O. Roos, Chem. Phys. Lett. 15, 153
  (1972), doi:10.1016/0009-2614(72)80140-4; P. E. M. Siegbahn, J. Chem. Phys. 72, 1647 (1980),
  doi:10.1063/1.439365. String-driven organization and the ``h~`` one-electron folding:
  P. J. Knowles, N. C. Handy, Chem. Phys. Lett. 111, 315 (1984),
  doi:10.1016/0009-2614(84)85513-X; J. Olsen, B. O. Roos, P. Jorgensen, H. J. Aa. Jensen,
  J. Chem. Phys. 89, 2185 (1988), doi:10.1063/1.455063. Operator algebra and the
  ``E_pq E_rs - delta_qr E_ps`` identity: T. Helgaker, P. Jorgensen, J. Olsen, "Molecular
  Electronic-Structure Theory", Wiley (2000), ch. 1-2, 11.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from . import kernels
from .strings import CASSpace, _check_array

log = get_logger(__name__)

#: Generous byte accounting for one excitation incidence of a gather block's temporaries:
#: the gathered determinant index, orbital and sign, the flat index (``intp``) and the
#: gathered complex value, plus NumPy's own internal index array. Used once, outside the
#: loop, to turn a transient budget into a block size (B7).
BYTES_PER_GATHER_INCIDENCE = 64.0


#: Rows of the in-place GEMM tile (see :func:`sigma_workspace_gb`). A **fixed** count, so the
#: sizing function is deterministic across machines; large enough that a
#: ``(GEMM_TILE_ROWS, n^2) . (n^2, n^2)`` product is squarely BLAS-3, small enough that the
#: tile is ~50 MB at 20 spinors against the 1.1 GB second buffer it replaced.
GEMM_TILE_ROWS = 8192


def sigma_workspace_gb(n_spinor: int, n_elec: int) -> float:
    """Size [GB] of the resident sigma workspace (exact sizing function).

    ⚠ **This is the number that sets the conventional-CI ceiling**, not the flop count:
    ``C(n,k) * n^2`` complex numbers plus a fixed GEMM tile. 0.27 GB at 18 spinors half
    filled, **1.15 GB at 20**, 5.15 GB at 22, 23.3 GB at 24.

    ⚠ **``F`` and ``G`` share this one buffer, and that is a theorem about the data flow, not
    a squeeze**: once the row-local one-electron term is banked (its contraction reads
    ``F``'s own row and nothing else), no consumer needs ``F`` after the GEMM that turns it
    into ``G`` — the RDM builder and the perturbation regather ``F`` from scratch. BLAS
    forbids an aliased GEMM, so the product runs in row tiles through
    :data:`GEMM_TILE_ROWS` of scratch and is copied back; the copy is a few per cent of the
    GEMM (measured, in this package's validation record). Step 3's gather still needs **all
    of ``G``** resident — the ``K`` connected to an output row are scattered across the whole
    space — so the halving ends here: going further is the batched, scattering kernel that
    was measured and scratched, not a smaller run of this one.

    ⚠ **The Kramers-restricted symmetry mode does not move this**, measured rather than
    assumed and recorded in this package's validation record. That mode halves the *Davidson*
    subspace, which is a different term; the workspace is gathered over the whole determinant
    space in both modes. Switching symmetry mode is therefore never the answer to this
    refusal.
    """
    from .strings import cas_dimension
    ndet = cas_dimension(n_spinor, n_elec)
    tile = min(ndet, GEMM_TILE_ROWS)
    return res.array_gb((ndet + tile, n_spinor * n_spinor), np.complex128)


def eri_matrix_gb(n_spinor: int) -> float:
    """Size [GB] of the ``(n^2, n^2)`` reshaped two-electron integrals."""
    return res.array_gb((n_spinor * n_spinor, n_spinor * n_spinor), np.complex128)


# --- The two gather kernels --------------------------------------------------------

def gather_block_size(n_elec: int, n_empty: int, ndet: int,
                       block: Optional[int]) -> int:
    """Rows per vectorized chunk, from the transient budget — **once, outside the kernel**.

    A memory check may never sit inside a loop, and B7 generalizes that to all global state: the
    kernel receives a block size and never asks for one. This is where the asking happens.
    """
    if block is not None:
        return int(max(1, block))
    per_row = BYTES_PER_GATHER_INCIDENCE * n_elec * n_empty / res.BYTES_PER_GB
    return int(max(1, min(ndet, res.transient_gb() / max(per_row, 1e-12))))


def _check_map(kernel: str, h2d_det, h2d_orb, h2d_sign, d2h_hole, d2h_orb, d2h_sign) -> None:
    """Entry checks for the six excitation-map arrays (B4/B5)."""
    _check_array(kernel, "h2d_det", h2d_det, np.int32, 2)
    _check_array(kernel, "h2d_orb", h2d_orb, np.int8, 2)
    _check_array(kernel, "h2d_sign", h2d_sign, np.int8, 2)
    _check_array(kernel, "d2h_hole", d2h_hole, np.int32, 2)
    _check_array(kernel, "d2h_orb", d2h_orb, np.int8, 2)
    _check_array(kernel, "d2h_sign", d2h_sign, np.int8, 2)
    if not (h2d_det.shape == h2d_orb.shape == h2d_sign.shape):
        raise ValueError("{}: the hole->determinant tables disagree in shape".format(kernel))
    if not (d2h_hole.shape == d2h_orb.shape == d2h_sign.shape):
        raise ValueError("{}: the determinant->hole tables disagree in shape".format(kernel))
    if h2d_det.shape[0] * h2d_det.shape[1] != d2h_hole.shape[0] * d2h_hole.shape[1]:
        raise ValueError("{}: the two excitation tables hold different incidence counts "
                         "({} and {}); they must be transposes of one another"
                         .format(kernel, h2d_det.size, d2h_hole.size))


@kernels.kernel("sigma_gather_f")
def sigma_gather_f_numpy(c: np.ndarray,
                         h2d_det: np.ndarray, h2d_orb: np.ndarray, h2d_sign: np.ndarray,
                         d2h_hole: np.ndarray, d2h_orb: np.ndarray, d2h_sign: np.ndarray,
                         n_spinor: int, block: int, f_out: np.ndarray) -> None:
    """Step 1: ``F[K, p*n + q] = sum_J <K|E_pq|J> c_J`` (registered kernel).

    A gather over the output row: for each determinant ``K``, loop over its ``k`` occupied
    orbitals ``p`` (reaching hole string ``h``) and over the ``n-k+1`` empty orbitals ``q`` of
    ``h`` (reaching determinant ``J``), and write ``sign * c[J]``. Every write is to the row
    the loop owns and every ``(p, q)`` within a row is distinct, so nothing accumulates: a
    threaded port parallelizes over ``K`` with **no atomics and no privatization**.

    Parameters
    ----------
    c : ``(ndet,)`` ``complex128``, C-contiguous — the CI vector.
    h2d_det, h2d_orb, h2d_sign, d2h_hole, d2h_orb, d2h_sign
        The excitation map of :func:`kuiva.ci.strings.excitation_map_numpy`; that docstring is
        the layout contract.
    n_spinor : int
    block : int
        Output rows per vectorized chunk. ⚠ A **parameter**, never read from a budget or a
        config file (B7); a compiled backend has no temporaries and may ignore it.
    f_out : ``(ndet, n_spinor**2)`` ``complex128``, C-contiguous
        Caller-allocated. Zeroed here (only ``k(n-k+1)`` of its ``n^2`` columns are touched
        per row, so the rest must be cleared for the GEMM that follows). Must not alias ``c``.

    Notes
    -----
    **Portability:** plain arrays and scalars (B1), no hashing (B2), rectangular tables (B3),
    dtypes and C-contiguity asserted on entry (B4/B5), caller-provided non-aliasing output
    (B6), blocking as a parameter (B7), no logging/timing/resource call and no raise inside
    any loop (B8), no callbacks (B9).

    **Reduction order (B10): none.** Every element of ``f_out`` is written exactly once, so a
    threaded or reordered port must be **bitwise** identical.
    """
    _check_array("sigma_gather_f", "c", c, np.complex128, 1)
    _check_array("sigma_gather_f", "f_out", f_out, np.complex128, 2)
    _check_map("sigma_gather_f", h2d_det, h2d_orb, h2d_sign, d2h_hole, d2h_orb, d2h_sign)
    n = int(n_spinor)
    ndet = d2h_hole.shape[0]
    if f_out.shape != (ndet, n * n):
        raise ValueError("sigma_gather_f: f_out has shape {}, expected {}"
                         .format(f_out.shape, (ndet, n * n)))
    if c.shape != (ndet,):
        raise ValueError("sigma_gather_f: c has shape {}, expected {}"
                         .format(c.shape, (ndet,)))
    if np.shares_memory(f_out, c):
        raise ValueError("sigma_gather_f: f_out aliases c")
    if block < 1:
        raise ValueError("sigma_gather_f: block must be positive, got {}".format(block))

    f_flat = f_out.reshape(-1)
    f_out[...] = 0.0
    for lo in range(0, ndet, int(block)):
        hi = min(lo + int(block), ndet)
        hole = d2h_hole[lo:hi]                                     # (B, k)
        ann = d2h_orb[lo:hi].astype(np.intp)                       # (B, k)   annihilated p
        cre = h2d_orb[hole].astype(np.intp)                        # (B, k, m) created q
        sign = d2h_sign[lo:hi][:, :, None] * h2d_sign[hole]        # (B, k, m) int8
        row = np.arange(lo, hi, dtype=np.intp)[:, None, None] * (n * n)
        f_flat[row + ann[:, :, None] * n + cre] = c[h2d_det[hole]] * sign


@kernels.kernel("sigma_gather_out")
def sigma_gather_out_numpy(g: np.ndarray,
                           h2d_det: np.ndarray, h2d_orb: np.ndarray, h2d_sign: np.ndarray,
                           d2h_hole: np.ndarray, d2h_orb: np.ndarray, d2h_sign: np.ndarray,
                           n_spinor: int, block: int, out: np.ndarray) -> None:
    """Step 3's two-electron half: ``out_I = sum_K sum_pq <I|E_pq|K> G[K, p*n + q]``.

    The **same map in the same direction** as :func:`sigma_gather_f_numpy` — same loop, same
    signs — reading ``G`` at row ``K`` and column ``p*n + q`` instead of ``c`` at ``J``. The
    factor ``1/2`` and the one-electron term ``sum_pq h~_pq F[I,pq]`` are applied by the
    caller: the former is a scalar and the latter is a row-local BLAS-2 contraction of ``F``,
    and neither belongs in a gather whose CPU time is the port-gate measurement.

    ⚠ This is the expensive gather. ``G`` is 1.1 GB at 20 spinors and is read fully randomly.

    Parameters
    ----------
    g : ``(ndet, n_spinor**2)`` ``complex128``, C-contiguous.
    out : ``(ndet,)`` ``complex128``, C-contiguous. Caller-allocated; must not alias ``g``.
    Others as :func:`sigma_gather_f_numpy`.

    Notes
    -----
    **Portability:** B1-B9 as :func:`sigma_gather_f_numpy`.

    ⚠ **Reduction order (B10): yes.** Each output element sums ``k(n-k+1)`` gathered terms,
    and NumPy's pairwise summation is not the order a sequential or vectorized C++ loop would
    use. A port therefore gets the **1e-13 relative** parity tolerance rather than a bitwise
    one — the row ownership is unchanged, so this is floating-point re-association only, never
    a race.
    """
    _check_array("sigma_gather_out", "g", g, np.complex128, 2)
    _check_array("sigma_gather_out", "out", out, np.complex128, 1)
    _check_map("sigma_gather_out", h2d_det, h2d_orb, h2d_sign, d2h_hole, d2h_orb, d2h_sign)
    n = int(n_spinor)
    ndet = d2h_hole.shape[0]
    if g.shape != (ndet, n * n):
        raise ValueError("sigma_gather_out: g has shape {}, expected {}"
                         .format(g.shape, (ndet, n * n)))
    if out.shape != (ndet,):
        raise ValueError("sigma_gather_out: out has shape {}, expected {}"
                         .format(out.shape, (ndet,)))
    if np.shares_memory(out, g):
        raise ValueError("sigma_gather_out: out aliases g")
    if block < 1:
        raise ValueError("sigma_gather_out: block must be positive, got {}".format(block))

    g_flat = g.reshape(-1)
    for lo in range(0, ndet, int(block)):
        hi = min(lo + int(block), ndet)
        hole = d2h_hole[lo:hi]                                     # (B, k)
        ann = d2h_orb[lo:hi].astype(np.intp)                       # (B, k)
        cre = h2d_orb[hole].astype(np.intp)                        # (B, k, m)
        sign = d2h_sign[lo:hi][:, :, None] * h2d_sign[hole]        # (B, k, m) int8
        row = h2d_det[hole].astype(np.intp) * (n * n)              # (B, k, m) -> K
        out[lo:hi] = np.sum(g_flat[row + ann[:, :, None] * n + cre] * sign, axis=(1, 2))


# --- The operator (orchestration; stays Python) -------------------------------------------

class SigmaOperator:
    """``c -> H c`` over a complete CAS space, matrix-free.

    Holds the excitation map, the folded one-electron matrix, the reshaped integrals and the
    resident ``F``/``G`` workspaces, so that a Davidson iteration is a sequence of calls with
    no reallocation. Everything here is orchestration — policy, budgeting, assertions and
    timing — and none of it is a port candidate; the arithmetic is in the two registered
    kernels and in one BLAS call.

    Parameters
    ----------
    space : :class:`kuiva.ci.strings.CASSpace`
    h : ``(n, n)`` complex Hermitian one-electron integrals in the active spinor basis.
    eri : ``(n, n, n, n)`` complex two-electron integrals ``(pq|rs)`` in chemists' notation,
        as produced by :func:`kuiva.integrals.transform.assemble_4c`.
    backend : str, optional
        Kernel backend; ``None`` takes the preferred one. Callers do not learn which.
    block : int, optional
        Gather block size. ``None`` sizes it once from the transient budget.
    check_symmetry : bool
        Assert the 4-fold permutational symmetry of ``eri`` on construction. Costs one pass
        over an ``n^4`` array — 2.6 MB at 20 spinors — against the cost of discovering a
        transposed integral block through a wrong energy.
    """

    def __init__(self, space: CASSpace, h: np.ndarray, eri: np.ndarray, *,
                 backend: Optional[str] = None, block: Optional[int] = None,
                 check_symmetry: bool = True) -> None:
        n = space.n_spinor
        # Cheap shape guard *before* the reservation: the workspace is 1.2 GB at 20 spinors
        # and there is no reason to allocate it only to refuse the integrals afterwards.
        if np.shape(h) != (n, n):
            raise ValueError("h has shape {}, expected {}".format(np.shape(h), (n, n)))
        if np.shape(eri) != (n, n, n, n):
            raise ValueError("eri has shape {}, expected {}".format(np.shape(eri), (n,) * 4))
        space.build_excitation_map()

        self.space = space
        self.backend = backend
        self.n_spinor = n
        self.ndet = space.ndet

        alloc = res.reserve("full-CI sigma workspace ({} spinors, {} electrons)"
                            .format(n, space.n_elec),
                            sigma_workspace_gb(n, space.n_elec),
                            note="F/G (one shared buffer) of {} determinants x {} orbital "
                                 "pairs, plus the GEMM tile".format(self.ndet, n * n),
                            advice=[
                                "reduce the active space: the workspace grows as "
                                "C(n,k) * n^2",
                                "this kernel keeps the whole F/G intermediate resident so "
                                "that both gathers own their output row (no atomics in a "
                                "threaded port); above ~24 spinors that is no longer "
                                "possible and the answer is a batched, scattering kernel "
                                "that does not exist yet -- not a larger run of this one",
                                "the Kramers-restricted CI mode does NOT help here: it "
                                "halves the Davidson subspace, not this workspace, which is "
                                "gathered over the whole determinant space in either mode",
                                "above the conventional-CI ceiling, use DMRG"])
        self.f_buf = np.empty((self.ndet, n * n), dtype=np.complex128)
        self.tile_buf = np.empty((min(self.ndet, GEMM_TILE_ROWS), n * n),
                                 dtype=np.complex128)
        # The reservation dies with the buffer: an operator is long-lived in a CASSCF (one
        # per active space, its integrals swapped in place), but the perturbation builds one
        # per shifted electron space and keeps at most one alive — a reservation that
        # outlived the dropped operator would charge the budget for every shift at once,
        # which is exactly the aggregate the one-live policy exists to avoid.
        res.owned_by(self.f_buf, alloc)
        self.block = gather_block_size(space.n_elec, space.n_empty, self.ndet, block)
        self.n_apply = 0
        self.set_integrals(h, eri, check_symmetry=check_symmetry)
        log.debug("sigma operator: %d determinants, %d spinors, block %d, workspace %.3f GB",
                  self.ndet, n, self.block, sigma_workspace_gb(n, space.n_elec))

    def set_integrals(self, h: np.ndarray, eri: np.ndarray, *,
                      check_symmetry: bool = True) -> None:
        """Install a new one- and two-electron set over the **same** determinant space.

        ⚠ This is what makes a CASSCF macro-iteration cheap. The integrals change at every
        orbital rotation while the space, the excitation map and the ``F``/``G`` workspace do
        not — and that workspace is 1.2 GB at 20 spinors (:func:`sigma_workspace_gb`).
        Constructing a fresh operator per macro-iteration would free and re-allocate it every
        time; this replaces two small arrays instead. Nothing that has been handed out
        (``f_buf``, shared with :class:`~kuiva.rdm.rdm.RDMBuilder`) is invalidated.
        """
        n = self.n_spinor
        h = np.ascontiguousarray(h, dtype=np.complex128)
        eri = np.ascontiguousarray(eri, dtype=np.complex128)
        if h.shape != (n, n):
            raise ValueError("h has shape {}, expected {}".format(h.shape, (n, n)))
        if eri.shape != (n, n, n, n):
            raise ValueError("eri has shape {}, expected {}".format(eri.shape, (n,) * 4))
        if check_symmetry:
            _assert_hermitian(h, "one-electron integrals")
            _assert_four_fold(eri)
        # h~_pq = h_pq - 1/2 sum_r (pr|rq). See the module docstring: this folding is
        # invisible to every structural check, so it lives on one line, next to its equation.
        self.h_eff = h - 0.5 * np.einsum("prrq->pq", eri)
        self.h_eff_flat = np.ascontiguousarray(self.h_eff.reshape(n * n))
        # (pq|rs) as an (n^2, n^2) matrix. Symmetric (not Hermitian) under (pq) <-> (rs), so
        # the GEMM of step 2 needs no transpose -- asserted just above rather than assumed.
        self.eri_mat = np.ascontiguousarray(eri.reshape(n * n, n * n))

    def __call__(self, c: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """``H c`` for one CI vector.

        ``F`` is per vector: a block Davidson applies this to its expansion directions one at
        a time, so the resident cost stays ``N n^2`` plus the tile rather than ``n_roots`` times that.
        """
        c = np.ascontiguousarray(c, dtype=np.complex128)
        if c.shape != (self.ndet,):
            raise ValueError("CI vector has shape {}, expected {}"
                             .format(c.shape, (self.ndet,)))
        if out is None:
            out = np.empty(self.ndet, dtype=np.complex128)
        arrays = self.space.excitation_arrays()
        n = self.n_spinor

        # The three regions are timed separately on purpose: step 1 reads a cached
        # few MB, step 3 reads G at ~1.1 GB fully randomly, and a single "gather" number
        # would hide the difference that decides whether either is worth porting.
        with timer("sigma step 1: gather F"):
            kernels.resolve("sigma_gather_f", self.backend)(
                c, *arrays, n, self.block, self.f_buf)
        # The one-electron term is row-local — a BLAS-2 contraction of F's own row — and it
        # is banked *before* the GEMM, which is what lets F and G share one buffer: after
        # this line nothing needs F, so the GEMM may overwrite it. Kept out of the gathers'
        # timed regions so the number the C++ port gate is decided on is the gather alone.
        with timer("sigma one-electron term"):
            one_electron = self.f_buf @ self.h_eff_flat
        # ⚠ In place through a tile, not out= on the same array: BLAS forbids an aliased
        # GEMM. Row tiles keep every output row's contraction identical to the one-shot
        # product's (a GEMM row depends on its own A row only); the copy-back is a few per
        # cent of the GEMM and buys the second N x n^2 buffer this operator no longer holds.
        with timer("sigma step 2: GEMM"):
            nt = self.tile_buf.shape[0]
            for r0 in range(0, self.ndet, nt):
                r1 = min(r0 + nt, self.ndet)
                np.dot(self.f_buf[r0:r1], self.eri_mat, out=self.tile_buf[:r1 - r0])
                self.f_buf[r0:r1] = self.tile_buf[:r1 - r0]
        with timer("sigma step 3: gather sigma"):
            kernels.resolve("sigma_gather_out", self.backend)(
                self.f_buf, *arrays, n, self.block, out)
        out *= 0.5
        out += one_electron
        self.n_apply += 1
        return out

    def gather_f(self, c: np.ndarray) -> np.ndarray:
        """Fill the shared workspace with ``F[K, pq] = sum_J <K|E_pq|J> c_J`` and return it.

        ⚠ The returned array is the operator's own buffer, and it holds ``G`` after an
        application — ``F`` and ``G`` share it (:func:`sigma_workspace_gb`). Anything that
        needs ``F`` therefore gathers it afresh, which is what the RDM builder and the
        perturbation always did by driving the gather kernel themselves; this is the same
        gather with the entry checks in one place.
        """
        c = np.ascontiguousarray(c, dtype=np.complex128)
        if c.shape != (self.ndet,):
            raise ValueError("CI vector has shape {}, expected {}"
                             .format(c.shape, (self.ndet,)))
        kernels.resolve("sigma_gather_f", self.backend)(
            c, *self.space.excitation_arrays(), self.n_spinor, self.block, self.f_buf)
        return self.f_buf

    def matrix(self) -> np.ndarray:
        """The dense CI Hamiltonian, by applying to unit vectors (tests and tiny spaces).

        ``O(N)`` sigma calls and ``N^2`` storage — a diagnostic, never a solver path.
        """
        h = np.empty((self.ndet, self.ndet), dtype=np.complex128)
        e = np.zeros(self.ndet, dtype=np.complex128)
        for j in range(self.ndet):
            e[:] = 0.0
            e[j] = 1.0
            h[:, j] = self(e)
        return h

    def __repr__(self) -> str:
        return "SigmaOperator(ndet={}, n_spinor={}, applications={})".format(
            self.ndet, self.n_spinor, self.n_apply)


def _assert_hermitian(h: np.ndarray, what: str) -> None:
    scale = max(float(np.max(np.abs(h))), 1.0)
    err = float(np.max(np.abs(h - h.conj().T)))
    if err > 1e-10 * scale:
        raise ValueError("{} are not Hermitian: max |h - h^dag| = {:.3e}".format(what, err))


def _assert_four_fold(eri: np.ndarray) -> None:
    """Assert ``(pq|rs) = (rs|pq)`` and ``(pq|rs) = (qp|sr)*`` — and nothing more.

    ⚠ These are the **only** two symmetries complex spinor integrals have. The 8-fold
    relations of a real basis, ``(pq|rs) = (rq|ps)``, are false here, and an implementation
    that quietly assumed one would produce a Hermitian, plausible, wrong Hamiltonian — the
    same class of trap as ``assemble_4c``'s deliberately absent conjugation. Nothing
    downstream of this module uses more than what is checked here.
    """
    scale = max(float(np.max(np.abs(eri))), 1.0)
    err = float(np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))))
    if err > 1e-10 * scale:
        raise ValueError("two-electron integrals violate (pq|rs) = (rs|pq): "
                         "max deviation {:.3e}".format(err))
    err = float(np.max(np.abs(eri - eri.transpose(1, 0, 3, 2).conj())))
    if err > 1e-10 * scale:
        raise ValueError("two-electron integrals violate (pq|rs) = (qp|sr)*: "
                         "max deviation {:.3e}".format(err))


#: Relative tolerance on the time-reversal symmetry of a set of active-space integrals.
#: ⚠ Not a physical tolerance: the relations below are **exact** for a Kramers-paired orbital
#: set, so anything above roundoff means the orbitals have stopped being Kramers paired. Loose
#: enough (1e-9) to survive a CASSCF trajectory's accumulated rotation error, tight enough that
#: an unrestricted reference or a genuinely broken pairing fails by many orders.
TIME_REVERSAL_TOL = 1.0e-9


def time_reversal_violation(h: np.ndarray, eri: np.ndarray) -> Tuple[float, float]:
    """``(one-electron, two-electron)`` relative breach of time-reversal symmetry.

    In the interleaved Kramers-paired convention of ``kuiva/spinor/expand.py`` — spinor ``2p``
    unbarred, ``2p+1`` barred, ``T a+_k T^-1 = t_k a+_{kbar}`` with ``t_k = (-1)^k`` — an
    operator commutes with time reversal exactly when::

        h_pq     = t_p t_q         conj(h_{pbar,qbar})
        (pq|rs)  = t_p t_q t_r t_s conj((pbar qbar|rbar sbar))

    ⚠ **This is a genuine structural check and not a restatement of hermiticity.** It fails on
    a swapped spin block, on an unrestricted (un-Kramers-paired) orbital set, and on an active
    space that splits a Kramers pair across its boundary — none of which hermiticity, the
    4-fold permutational relations or any trace condition can see. It is what the
    Kramers-restricted CI stands on: that path assumes ``[H, T] = 0`` in the *operator*, and if
    it is false the solver returns a converged, degenerate, plausible, wrong spectrum.

    Both figures are relative to ``max|h|`` and ``max|eri|`` respectively, so they are
    comparable across systems. One pass over the ``n^4`` array — 2.6 MB at 20 spinors.
    """
    h = np.asarray(h)
    eri = np.asarray(eri)
    n = h.shape[0]
    if n % 2 != 0:
        raise ValueError("a Kramers-paired active space has an even number of spinors; got {}"
                         .format(n))
    swap = np.arange(n) ^ 1
    t = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

    scale_h = max(float(np.max(np.abs(h))), 1.0)
    err_h = float(np.max(np.abs(h - np.outer(t, t) * np.conj(h[np.ix_(swap, swap)]))))
    sign4 = (t[:, None, None, None] * t[None, :, None, None]
             * t[None, None, :, None] * t[None, None, None, :])
    scale_eri = max(float(np.max(np.abs(eri))), 1.0)
    err_eri = float(np.max(np.abs(
        eri - sign4 * np.conj(eri[np.ix_(swap, swap, swap, swap)]))))
    return err_h / scale_h, err_eri / scale_eri


def assert_time_reversal(h: np.ndarray, eri: np.ndarray, *,
                         tol: float = TIME_REVERSAL_TOL, what: str = "") -> None:
    """Raise unless ``h`` and ``eri`` commute with time reversal to ``tol``.

    See :func:`time_reversal_violation`. The message names the orbital set rather than the
    integrals, because that is what the caller can do something about.
    """
    err_h, err_eri = time_reversal_violation(h, eri)
    if max(err_h, err_eri) <= tol:
        return
    raise ValueError(
        "{}the active-space integrals are not time-reversal symmetric (relative breach "
        "{:.2e} one-electron, {:.2e} two-electron, against {:.1e}). The orbitals spanning "
        "this active space are not a Kramers-paired set: with an unrestricted reference they "
        "never are, and an active space that splits a Kramers pair across its boundary "
        "destroys the pairing too. Use the general complex CI path, which assumes none of "
        "this".format(what and (what + ": "), err_h, err_eri, tol))


def sigma_vector(space: CASSpace, c: np.ndarray, h: np.ndarray, eri: np.ndarray,
                 **kwargs) -> np.ndarray:
    """One-shot ``H c`` (tests and one-off use).

    Builds and discards the workspace, so it is the wrong entry point for an iterative solve —
    use :class:`SigmaOperator` there. Deliberately **not** called ``sigma``: that name belongs
    to this module, and a function shadowing its own module in ``kuiva.ci`` is a trap.
    """
    return SigmaOperator(space, h, eri, **kwargs)(c)


__all__ = ["SigmaOperator", "sigma_vector", "sigma_workspace_gb", "eri_matrix_gb",
           "gather_block_size", "time_reversal_violation", "assert_time_reversal",
           "TIME_REVERSAL_TOL",
           "sigma_gather_f_numpy", "sigma_gather_out_numpy", "BYTES_PER_GATHER_INCIDENCE"]
