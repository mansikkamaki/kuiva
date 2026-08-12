"""Full-CI 1- and 2-particle density matrices from the sigma vector's intermediate.

One excitation map, one intermediate, three consumers
------------------------------------------------------
``F[K, p, q] = sum_J <K|E_pq|J> c_J`` is not only the sigma vector's scratch. With
``E_pq^dag = E_qp``,

::

    <c|E_pq E_rs|c> = sum_K conj(F[K, q, p]) F[K, r, s]
    Gamma_pqrs      = <E_pq E_rs> - delta_qr gamma_ps

so the **2-RDM is a single GEMM ``F^H F`` over the same intermediate** that the sigma vector
already builds, and the transition densities ``props/dump.py`` will need come from the same
map. That is load-bearing rather than merely tidy: **one** kernel contract has to be got right
instead of three, and a caller that already has a
:class:`~kuiva.ci.sigma.SigmaOperator` can hand its ``f_buf`` straight to
:class:`RDMBuilder` and pay no second residency (1.1 GB at 20 spinors).

⚠ The index order is the trap, and it is the ``conj(F[K, q, p])`` above. ``F`` is indexed by
the operator ``E_pq``, so the bra side needs the *transposed* orbital pair — the accumulated
Hermitian product ``M[ab, rs] = sum_K conj(F[K,ab]) F[K,rs]`` is therefore
``D[pq, rs] = M[qp, rs]``, not ``M`` itself. Getting that wrong leaves a 2-RDM that is still
Hermitian, still positive, still has the right trace, and is wrong — the conjugation
trap in its RDM guise. It is caught by the trace condition
``sum_r Gamma_pqrr = (N-1) gamma_pq`` and by the closure check against the Davidson eigenvalue,
both of which fail immediately on a transposition.

State averaging and Kramers-block completeness happen **here**
---------------------------------------------------------------------------
State averaging happens "where the RDMs are built, because nothing downstream can recover
it", and the same is true of the block-completeness rule:

* Weights are **equal within a degenerate block**. Inside a degenerate manifold the individual
  roots are defined only up to a rotation, so an unequal weighting makes the RDMs — and hence
  the converged orbitals — depend on an arbitrary choice the eigensolver happened to make.
* ⚠ A requested state count that **splits** a degenerate block is refused. With an **odd**
  electron count Kramers' theorem makes every level at least doubly degenerate, so an odd
  state count necessarily splits a pair and the refusal is rigorous. With an even count there
  is no such theorem and the check is what the computed spectrum shows, which is honest but
  weaker: an accidental degeneracy straddling the last requested root cannot be seen without
  computing one more root.
* ⚠ **That policy lives in this module's Python wrapper, never in the kernel** (B8).

3- and 4-RDMs are deliberately absent HERE — the network path has them
-----------------------------------------------------------------------
They belong to SC-NEVPT2, which does not exist yet on either CI branch. The
**network-side** builders do exist (:mod:`kuiva.dmrg.density`, ``network_rdm`` ranks 1–4):
direct contraction, no cumulant — the chosen no-cumulant route — returning the same convention as
this module (``Gamma_pqrs = <a+_p a+_r a_s a_q>``, pairs interleaved at every rank). The
4-RDM is ``n_act^8`` — 6.4 GB at 12 active spinors, 22 GB at 14, 381 GB at 20
(:func:`kuiva.util.resources.rdm_gb`) — refused by the memory budget long before anything else binds.
Nothing in *this* module should be extended to rank 3 or 4 by analogy: the ``F^H F`` trick
does not generalize, since a rank-3 RDM needs a two-particle intermediate; a CI-exact
higher-rank path, if SC-NEVPT2 ever needs one on the conventional branch, is that plan's
problem.

References
----------
* Density matrices from the CI one-particle-excitation intermediate: J. Olsen, B. O. Roos,
  P. Jorgensen, H. J. Aa. Jensen, J. Chem. Phys. 89, 2185 (1988), doi:10.1063/1.455063;
  P. J. Knowles, N. C. Handy, Chem. Phys. Lett. 111, 315 (1984),
  doi:10.1016/0009-2614(84)85513-X. Operator algebra, the ``E_pq E_rs - delta_qr E_ps``
  identity and the trace conditions: T. Helgaker, P. Jorgensen, J. Olsen, "Molecular
  Electronic-Structure Theory", Wiley (2000), ch. 1-2.
* Kramers' theorem and the degeneracy it imposes on an odd-electron system: H. A. Kramers,
  Proc. Amsterdam Acad. 33, 959 (1930); E. Wigner, Nachr. Ges. Wiss. Gottingen, Math.-Phys.
  Kl. 1932, 546 (1932).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..ci import kernels
from ..ci.sigma import gather_block_size
from ..ci.strings import CASSpace, _check_array, _check_output
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

#: Energy separation [Eh] below which two CI roots are treated as one degenerate block.
#: ⚠ Sized for the general-complex path, where Kramers degeneracy is realized only
#: numerically, at 1e-8 to 1e-6 Eh. Tighter than 1e-6 and a genuine Kramers pair is seen as
#: two levels; much looser and physically distinct states get averaged together.
DEFAULT_DEGENERACY_TOL = 1.0e-6

#: Rows of ``F`` per block in the accumulation, as a byte budget for the conjugate temporary.
BYTES_PER_RDM_ROW = 32.0


def rdm_workspace_gb(n_spinor: int) -> float:
    """Size [GB] of the accumulator and the finished 2-RDM (exact sizing function).

    Two ``n^4`` complex arrays: the Hermitian product ``M`` and ``Gamma`` built from it. 5.1 MB
    at 20 active spinors, 100 MB at 42, 1.6 GB at 84 — which is well past the
    conventional-CI ceiling in any case.
    """
    return 2.0 * res.rdm_gb(n_spinor, 2)


def intermediate_gb(ndet: int, n_spinor: int) -> float:
    """Size [GB] of the ``F`` intermediate this module gathers, when it allocates its own."""
    return res.array_gb((ndet, n_spinor * n_spinor), np.complex128)


# --- The accumulation kernel -------------------------------------------------------

@kernels.kernel("rdm_accumulate")
def rdm_accumulate_numpy(f: np.ndarray, c: np.ndarray, weight: float, block: int,
                         m_out: np.ndarray, gamma_out: np.ndarray) -> None:
    """Accumulate ``weight * F^H F`` into ``m_out`` and ``weight * c^H F`` into ``gamma_out``.

    Parameters
    ----------
    f : ``(ndet, n**2)`` ``complex128``, C-contiguous
        The intermediate ``F[K, p*n + q] = sum_J <K|E_pq|J> c_J`` for **one** state, as
        produced by :func:`kuiva.ci.sigma.sigma_gather_f_numpy`.
    c : ``(ndet,)`` ``complex128``, C-contiguous — that state's CI vector.
    weight : float — its state-averaging weight.
    block : int
        Rows per chunk. ⚠ A **parameter** (B7): the caller sizes it once, outside this call.
        It exists because NumPy has no lazy conjugate and would otherwise copy the whole of
        ``F``; a compiled backend calls ``zgemm`` with ``transa='C'`` and ``beta=1`` and needs
        neither the temporary nor the block.
    m_out : ``(n**2, n**2)`` ``complex128``, C-contiguous
        Accumulator for ``M[ab, rs] = sum_K conj(F[K,ab]) F[K,rs]``. ⚠ **Not** the 2-RDM and
        not even ``<E_pq E_rs>``: the caller transposes the bra pair and subtracts the
        ``delta_qr gamma_ps`` term. Accumulated across states, so it is **not zeroed here**.
    gamma_out : ``(n, n)`` ``complex128``, C-contiguous
        Accumulator for ``gamma_pq = <c|E_pq|c> = sum_K conj(c_K) F[K, p*n+q]``. Not zeroed.

    Notes
    -----
    **Portability:** plain arrays and scalars (B1), no hashing (B2), rectangular (B3), dtypes
    and contiguity asserted on entry (B4/B5), caller-provided non-aliasing outputs that the
    caller also owns the zeroing of (B6), blocking as a parameter (B7), no logging, timing,
    resource check or raise inside the loop (B8), no callbacks (B9).

    ⚠ **Reduction order (B10): yes**, twice over — the sum over determinants is split across
    blocks here and across threads in any port, and BLAS chooses its own order inside each
    block. A port therefore gets the **1e-13 relative** parity tolerance, not a bitwise one.
    Note this is a *summation* over a shared accumulator: a threaded port needs per-thread
    accumulators and a final reduction, which for an ``n^2 x n^2`` matrix is cheap — unlike
    the sigma gathers, which were deliberately designed so that no such thing is needed.
    """
    _check_array("rdm_accumulate", "f", f, np.complex128, 2)
    _check_array("rdm_accumulate", "c", c, np.complex128, 1)
    ndet, npair = f.shape
    n = int(round(npair ** 0.5))
    if n * n != npair:
        raise ValueError("rdm_accumulate: f has {} orbital-pair columns, which is not a "
                         "square".format(npair))
    _check_output("rdm_accumulate", "m_out", m_out, np.complex128, 2, (npair, npair), f, c)
    _check_output("rdm_accumulate", "gamma_out", gamma_out, np.complex128, 2, (n, n), f, c)
    if c.shape != (ndet,):
        raise ValueError("rdm_accumulate: c has shape {}, expected {}"
                         .format(c.shape, (ndet,)))
    if block < 1:
        raise ValueError("rdm_accumulate: block must be positive, got {}".format(block))

    gamma_flat = gamma_out.reshape(npair)
    for lo in range(0, ndet, int(block)):
        hi = min(lo + int(block), ndet)
        rows = f[lo:hi]
        conjugated = np.conj(rows)
        m_out += weight * (conjugated.T @ rows)
        gamma_flat += weight * (np.conj(c[lo:hi]) @ rows)


# --- Degenerate blocks and state averaging -----------------------------------

def degenerate_blocks(energies: Sequence[float],
                      tol: float = DEFAULT_DEGENERACY_TOL) -> List[Tuple[int, int]]:
    """Group ascending ``energies`` into ``(start, stop)`` blocks, split where a gap exceeds
    ``tol``."""
    values = np.asarray(energies, dtype=float)
    if values.size == 0:
        return []
    if np.any(np.diff(values) < -tol):
        raise ValueError("state energies must be given in ascending order")
    edges = np.nonzero(np.diff(values) > tol)[0] + 1
    bounds = [0] + edges.tolist() + [values.size]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def state_average_weights(energies: Sequence[float], n_elec: int,
                          weights: Optional[Sequence[float]] = None, *,
                          tol: float = DEFAULT_DEGENERACY_TOL,
                          on_split: str = "raise") -> np.ndarray:
    """Normalized weights, **equalized within each degenerate block**.

    Parameters
    ----------
    energies : ascending state energies [Eh].
    n_elec : int
        Electron count. ⚠ Load-bearing: Kramers' theorem applies to an **odd** number of
        electrons, and only then is "every level is at least doubly degenerate" — and hence
        "an odd state count splits a pair" — a theorem rather than an observation.
    weights : optional
        Requested weights; uniform if omitted. They are equalized within blocks, never used
        as given inside one.
    on_split : ``"raise"`` or ``"warn"``
        What to do when the requested states cut a degenerate block in half.

    Raises
    ------
    ValueError
        On a split block with ``on_split="raise"``. ⚠ Proceeding produces converged orbitals
        that depend on an arbitrary rotation the eigensolver chose inside the pair — a result
        that is not reproducible and does not look wrong.
    """
    values = np.asarray(energies, dtype=float)
    n_states = values.size
    requested = (np.full(n_states, 1.0 / n_states) if weights is None
                 else np.asarray(weights, dtype=float))
    if requested.size != n_states:
        raise ValueError("{} weights for {} states".format(requested.size, n_states))
    if np.any(requested < 0.0) or not np.sum(requested) > 0.0:
        raise ValueError("state-averaging weights must be non-negative and not all zero")
    requested = requested / np.sum(requested)

    blocks = degenerate_blocks(values, tol)
    averaged = requested.copy()
    for start, stop in blocks:
        averaged[start:stop] = np.mean(requested[start:stop])
        if not np.allclose(requested[start:stop], averaged[start:stop], rtol=0, atol=1e-12):
            log.warning("states %d-%d are degenerate to within %.1e Eh; their weights are "
                        "equalized to %.6f, because inside a degenerate block the individual "
                        "roots are defined only up to a rotation ",
                        start, stop - 1, tol, averaged[start])

    if int(n_elec) % 2 == 1:
        # Kramers' theorem: with an odd electron count every level is at least doubly
        # degenerate, so an odd block cannot be complete.
        odd = [(a, b) for a, b in blocks if (b - a) % 2 == 1]
        if odd:
            last = blocks[-1]
            message = (
                "the requested {} states split a Kramers-degenerate block of an odd-electron "
                "({}) system: block(s) {} have an odd number of members. Every level of such "
                "a system is at least doubly degenerate, so a state-averaged RDM built from "
                "this set depends on an arbitrary rotation inside the split pair, and nothing "
                "downstream can recover it ".format(                    n_states, n_elec, ", ".join("{}-{}".format(a, b - 1) for a, b in odd)))
            if odd[-1] == last and (last[1] - last[0]) % 2 == 1:
                message += ". The last block is the truncated one: ask for {} or {} states"\
                    .format(n_states - 1, n_states + 1)
            else:
                message += (". No truncated block is at the end, so the degeneracy tolerance "
                            "({:.1e} Eh) is probably too tight for this spectrum".format(tol))
            if on_split == "raise":
                raise ValueError(message)
            if on_split != "warn":
                raise ValueError("on_split must be 'raise' or 'warn', got {!r}"
                                 .format(on_split))
            log.warning("%s", message)
    return averaged


# --- The builder (orchestration; stays Python) --------------------------------------------

class RDMBuilder:
    """State-averaged ``gamma`` and ``Gamma`` over a complete CAS space.

    Everything here is orchestration — the averaging policy, the budgeting, the transposition
    and the ``delta`` correction — and none of it is a port candidate. The arithmetic is one
    registered gather (shared with the sigma vector) and one registered GEMM.

    Parameters
    ----------
    space : :class:`kuiva.ci.strings.CASSpace`
    f_buf : ``(ndet, n**2)`` ``complex128``, optional
        The intermediate buffer. ⚠ Pass a
        :class:`~kuiva.ci.sigma.SigmaOperator`'s ``f_buf`` here: it is 1.1 GB at 20 spinors
        and there is no reason to hold two. Its contents are overwritten.
    """

    def __init__(self, space: CASSpace, *, backend: Optional[str] = None,
                 f_buf: Optional[np.ndarray] = None, block: Optional[int] = None) -> None:
        n = space.n_spinor
        space.build_excitation_map()
        self.space = space
        self.backend = backend
        self.n_spinor = n
        self.ndet = space.ndet
        if f_buf is None:
            res.reserve("full-CI RDM intermediate ({} spinors, {} electrons)"
                        .format(n, space.n_elec), intermediate_gb(self.ndet, n),
                        note="{} determinants x {} orbital pairs".format(self.ndet, n * n),
                        advice=["pass the sigma operator's f_buf instead of letting this "
                                "allocate its own -- they are the same object "])
            f_buf = np.empty((self.ndet, n * n), dtype=np.complex128)
        elif f_buf.shape != (self.ndet, n * n) or f_buf.dtype != np.complex128:
            raise ValueError("f_buf has shape {} dtype {}, expected {} complex128"
                             .format(f_buf.shape, f_buf.dtype, (self.ndet, n * n)))
        self.f_buf = f_buf
        res.reserve("full-CI density matrices ({} spinors)".format(n), rdm_workspace_gb(n),
                    note="the n^4 accumulator and the 2-RDM built from it")
        self.m_buf = np.zeros((n * n, n * n), dtype=np.complex128)
        self.gamma_buf = np.zeros((n, n), dtype=np.complex128)
        # One number, once, outside every loop (B7).
        per_row = BYTES_PER_RDM_ROW * n * n / res.BYTES_PER_GB
        self.block = (int(max(1, min(self.ndet, res.transient_gb() / max(per_row, 1e-12))))
                      if block is None else int(max(1, block)))
        # The gather's temporaries are different from the accumulation's, so it gets its own
        # block size -- from the same function the sigma vector uses, not a second policy.
        self.gather_block = gather_block_size(space.n_elec, space.n_empty, self.ndet, None)

    def __call__(self, civecs: np.ndarray, weights: Optional[Sequence[float]] = None, *,
                 energies: Optional[Sequence[float]] = None,
                 enforce_kramers: bool = True,
                 degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
                 on_split: str = "raise") -> Tuple[np.ndarray, np.ndarray]:
        """``(gamma, Gamma)``, state-averaged.

        Parameters
        ----------
        civecs : ``(ndet,)`` or ``(n_states, ndet)`` complex, normalized.
            ⚠ **Rows are states**, matching :attr:`kuiva.ci.davidson.DavidsonResult.vectors`.
        energies : sequence, optional
            Required when averaging over more than one state, because the block-completeness
            rule cannot be applied without them.
        enforce_kramers : bool
            The explicit escape hatch. ⚠ With it off, no degeneracy check runs and the
            requested weights are used as given — which is a statement that the caller has
            some other reason to want them, not a way to avoid supplying energies.
        """
        vectors = np.atleast_2d(np.ascontiguousarray(civecs, dtype=np.complex128))
        if vectors.shape[1] != self.ndet:
            raise ValueError("CI vectors have length {}, expected {}"
                             .format(vectors.shape[1], self.ndet))
        n_states = vectors.shape[0]
        if enforce_kramers and n_states > 1:
            if energies is None:
                raise ValueError(
                    "state-averaged RDMs over {} states need the state energies, so that "
                    "degenerate blocks can be weighted equally and an incomplete block "
                    "refused. Pass energies=, or enforce_kramers=False to "
                    "state deliberately that the weights are to be used as given"
                    .format(n_states))
            averaged = state_average_weights(energies, self.space.n_elec, weights,
                                             tol=degeneracy_tol, on_split=on_split)
        elif weights is None:
            averaged = np.full(n_states, 1.0 / n_states)
        else:
            averaged = np.asarray(weights, dtype=float)
            averaged = averaged / np.sum(averaged)

        n = self.n_spinor
        arrays = self.space.excitation_arrays()
        gather = kernels.resolve("sigma_gather_f", self.backend)
        accumulate = kernels.resolve("rdm_accumulate", self.backend)
        self.m_buf[...] = 0.0
        self.gamma_buf[...] = 0.0
        for state in range(n_states):
            vector = np.ascontiguousarray(vectors[state])
            with timer("RDM gather F"):
                gather(vector, *arrays, n, self.gather_block, self.f_buf)
            with timer("RDM accumulation F^H F"):
                accumulate(self.f_buf, vector, float(averaged[state]), self.block,
                           self.m_buf, self.gamma_buf)

        gamma = 0.5 * (self.gamma_buf + self.gamma_buf.conj().T)
        # <E_pq E_rs> = M[qp, rs]: the bra side carries the transposed orbital pair. See the
        # module docstring -- omitting this transposition leaves a Hermitian, positive,
        # correctly traced and wrong 2-RDM.
        gamma2 = np.ascontiguousarray(
            self.m_buf.reshape(n, n, n, n).transpose(1, 0, 2, 3))
        for orbital in range(n):                      # Gamma_pqrs -= delta_qr gamma_ps
            gamma2[:, orbital, orbital, :] -= gamma
        return gamma, gamma2

    def __repr__(self) -> str:
        return "RDMBuilder(ndet={}, n_spinor={})".format(self.ndet, self.n_spinor)


def cas_rdms(space: CASSpace, civecs: np.ndarray, weights: Optional[Sequence[float]] = None,
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """One-shot state-averaged ``(gamma, Gamma)`` (tests and one-off use).

    Builds and discards the intermediate, so it is the wrong entry point inside a CASSCF
    macro-iteration — use :class:`RDMBuilder` there, sharing the sigma operator's ``f_buf``.
    """
    builder_keys = ("backend", "f_buf", "block")
    builder = RDMBuilder(space, **{k: v for k, v in kwargs.items() if k in builder_keys})
    return builder(civecs, weights,
                   **{k: v for k, v in kwargs.items() if k not in builder_keys})


def active_space_energy(h: np.ndarray, eri: np.ndarray, gamma: np.ndarray,
                        gamma2: np.ndarray) -> float:
    """``E = sum_pq h_pq gamma_pq + 1/2 sum_pqrs (pq|rs) Gamma_pqrs`` [Eh].

    The **closure check** between the Hamiltonian and the density matrices: it must reproduce
    the CI eigenvalue exactly, and it fails on any phase or transposition error in either.
    Distinct from :func:`kuiva.mcscf.orbopt.cas_energy`, which does the same contraction with
    the *inactive Fock* and the core energy for an active space embedded in a larger orbital
    set; this one is the bare active-space Hamiltonian and takes no orbital spaces.
    """
    one = np.einsum("pq,pq->", h, gamma)
    two = 0.5 * np.einsum("pqrs,pqrs->", eri, gamma2)
    return float(np.real(one + two))


__all__ = ["RDMBuilder", "cas_rdms", "active_space_energy", "state_average_weights",
           "degenerate_blocks", "rdm_accumulate_numpy", "rdm_workspace_gb",
           "intermediate_gb", "DEFAULT_DEGENERACY_TOL"]
