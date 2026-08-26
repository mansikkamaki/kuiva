"""Block complex-Hermitian Davidson eigensolver for the full CI.

Matrix-free: the only thing it knows about the Hamiltonian is a callable ``apply_h(c) -> H c``
(:class:`kuiva.ci.sigma.SigmaOperator`) and its diagonal. Everything else is subspace algebra
on ``n_roots`` vectors.

⚠ Why this is a **block** solver, and why that is not a preference
------------------------------------------------------------------
With an odd number of electrons every CI root is **at least doubly degenerate** by Kramers'
theorem, and the general-complex path realizes that degeneracy only numerically, at
1e-8 to 1e-6 Eh. A solver that converges roots one at a time, or that carries a subspace
barely larger than the root count, cannot separate such a cluster: it either grinds or returns
two vectors spanning the pair badly.

**This project has already been bitten by exactly that.** The pre-optimization stage records an ARPACK Lanczos solve
failing on the near-degenerate low pair of a state-averaged Kramers-degenerate calculation at
as few as 1000 determinants, and the fix that did most of the work was *subspace size*. That
failure is the specification for this solver, so:

* all requested roots are expanded **simultaneously**, every iteration;
* the subspace floor is ``2 * n_roots + margin``, never ``n_roots + 1``;
* the initial guess is ``2 * n_roots`` unit vectors when nothing better is supplied, so a
  degenerate cluster starts with room around it;
* a dense ``eigh`` handles small spaces outright — exact, untunable, and it cannot fail.

Convergence, and why the tolerance is what it is
-------------------------------------------------
Convergence is on the **residual norm** of every requested root, and whatever it is set to
becomes a **noise floor on E(kappa)** — the same argument the pre-optimizer makes for its own tolerance: an
optimizer whose accept/reject test works at ``conv_energy = 1e-8`` Eh cannot be handed a
surface that wobbles at 1e-9.

⚠ **But the energy is the wrong quantity to size this against, and that is why the default is
1e-8 rather than the 1e-6 an energy argument would give.** The Ritz value error is
``O(|r|^2 / gap)`` — second order, because a Rayleigh quotient is variational — while the
**density matrices are first order in the eigenvector error**, and the density matrices are
what the orbital gradient is built from. Measured on LiH/STO-3G CAS(4,4) against PySCF's FCI,
with the energy agreeing to 1e-14 throughout:

===========  ==============  =================
``conv_tol``  energy error   1-RDM error
===========  ==============  =================
1e-6          1.2e-14        **3.0e-8**
1e-8          0              1.3e-10
1e-10         1.8e-15        1.2e-14
===========  ==============  =================

A 3e-8 RDM error sits right at the gradient tolerance the optimizer converges to, so the
energy-based reading of this tolerance would have put CI noise directly into the CASSCF
convergence test — while looking perfectly converged in every energy. Tightening to 1e-8 costs
**~25% more applications of H** (measured over four spaces from 252 to 3432 determinants) and
buys two and a half orders of RDM accuracy.

⚠ ``gap`` there means the gap to the *rest of the spectrum*, not to the neighbouring root. A
Kramers partner sits at zero gap and no tolerance can separate the two members of a pair —
nor should it, since inside a degenerate manifold the individual vectors are defined only up
to a rotation. What is well defined is the **cluster**, and that is what both the energies and
the state-averaged RDMs depend on.

⚠ "Converged" is not "lowest", and the guess is what decides which
------------------------------------------------------------------
A Krylov method can never leave the invariant subspaces its starting vectors lie in. The
natural CI guess — unit vectors on the lowest-diagonal determinants — is *biased*, and where
``H`` has a conserved quantum number the biased set can miss whole sectors; the solver then
converges every residual it is judged on and returns eigenpairs that are **not the lowest**,
with nothing anywhere to notice. This is not hypothetical: see :data:`N_GENERIC_GUESS`, which
is the fix and carries the measurement. Two consequences worth carrying elsewhere:

* ⚠ **A faster solve is not a better one.** The wrong answer there converged in 17 iterations
  and the right one in 36.
* ⚠ **The failure scales with the request in the wrong direction**: asking for *more* roots
  gives more guess vectors and can be right where fewer is wrong. A spectrum that changes
  qualitatively with ``n_roots`` is this, not physics.

⚠ What this solver depends on, and what it therefore cannot do
---------------------------------------------------------------
Davidson's convergence rests entirely on the **diagonal preconditioner**, i.e. on ``H_II``
being informative about where the low eigenvectors live. A real CI Hamiltonian in an MO basis
satisfies that comfortably. A Hamiltonian whose off-diagonal couplings are the size of its
diagonal spread does not, and there Davidson degenerates to a restarted Lanczos and may not
converge at all: measured on a 12 870-determinant space built from *random* one-electron
integrals, 4 roots stall at ``max|r| = 2.8e-1`` after 250 iterations, while the **same space
and the same two-electron integrals** with a Fock-like spread of orbital energies added to
``h`` converges in 161.

Two consequences. **A large-space test must use realistic integrals** — a random Hermitian
matrix is not a hard instance of this problem, it is a different problem. And a genuine
non-convergence in production is a statement about the *preconditioner*, so the answer is a
better one (a block-diagonal or Olsen preconditioner), not more iterations.

⚠ Non-convergence raises :class:`~kuiva.util.errors.SolverFailure`
-------------------------------------------------------------------
It never returns an unconverged vector. An unconverged vector is a plausible-looking answer
that poisons the RDMs, the orbital gradient and every macro-iteration after it; a raised
failure is information the event-gated optimizer already knows how to turn into a
rejected step.

⚠ Portability: this module is **orchestration, not a kernel**
-------------------------------------------------------------
Nothing here is a kernel-port candidate and nothing here should ever be ported. The subspace
operations are ``O(n_roots^2)`` on a handful of vectors; the entire cost is in the
``apply_h`` calls, i.e. in ``ci/sigma.py``'s kernels. Porting this module would move a few
percent of the arithmetic and lose the readability its correctness rests on — the unprofiled-optimization warning
about unprofiled optimization cutting both ways.

References
----------
* E. R. Davidson, J. Comput. Phys. 17, 87 (1975), doi:10.1016/0021-9991(75)90065-0 — the
  original method and the diagonal preconditioner.
* B. Liu, "The simultaneous expansion method", in *Numerical Algorithms in Chemistry:
  Algebraic Methods*, LBL-8158, Lawrence Berkeley Laboratory (1978), p. 49 — the block
  (simultaneous-expansion) generalization implemented here.
* M. Crouzeix, B. Philippe, M. Sadkane, SIAM J. Sci. Comput. 15, 62 (1994),
  doi:10.1137/0915004 — convergence analysis and the role of the restart subspace.
* Restart/collapse and the practice of complex-arithmetic CI eigensolvers: T. Helgaker,
  P. Jorgensen, J. Olsen, "Molecular Electronic-Structure Theory", Wiley (2000), ch. 11.7.
* Re-orthogonalized modified Gram-Schmidt: A. Bjorck, BIT 7, 1 (1967),
  doi:10.1007/BF01934122; G. H. Golub, C. F. Van Loan, "Matrix Computations", 4th ed.,
  Johns Hopkins (2013), sec. 5.2.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..util import output as out
from ..util import resources as res
from ..util.errors import SolverFailure
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

#: Residual-norm convergence threshold. ⚠ Set from the **1-RDM** error, not the energy error:
#: see the module docstring's table. An energy-based reading of this number would give 1e-6 and
#: put CI noise straight into the CASSCF gradient while every energy looked converged.
DEFAULT_CONV_TOL = 1.0e-8
#: Iteration cap. ⚠ Deliberately generous: a premature cap turns a solve that *would* have
#: converged into a :class:`SolverFailure`, which the event-gated optimizer reads as a rejected
#: step — an expensive and thoroughly confusing way to fail. A **cold** start on 12 870
#: determinants with 4 roots needs 115 iterations; the same solve warm-started from the
#: previous macro-iteration needs one. Grinding is bounded from the other side by the stall
#: detector, which raises as soon as no independent expansion direction is left.
DEFAULT_MAX_ITER = 300
#: Subspace cap as a multiple of the root count, and the additive margin that keeps a small
#: calculation from being starved. ⚠ The floor of ``2 * n_roots + SUBSPACE_MARGIN`` is the
#: lesson of the pre-optimizer's Lanczos failure and may not be lowered to save memory: room around a
#: degenerate cluster is what lets it separate.
#:
#: The *factor* is a measured performance knob rather than a correctness one. On a 12 870
#: determinant spin-orbit-coupled space with 4 roots, cold start:
#:
#: ====  ============  ============
#: cap   iterations    H applications
#: ====  ============  ============
#: 24    161           632
#: 48    115           447
#: 96    108           416
#: 160   105           405
#: ====  ============  ============
#:
#: The first doubling buys 29% and the rest saturates, so 12 is the knee, not the maximum.
#: The cost is ``3 * cap * ndet`` complex numbers (:func:`davidson_workspace_gb`), which at
#: production size is real — 1.2 GB for 10 roots of 184 756 determinants — so ``max_subspace``
#: is the knob a refusal names.
SUBSPACE_FACTOR = 12
SUBSPACE_MARGIN = 16
#: Determinant count below which the Hamiltonian is built densely and diagonalized outright.
#: ⚠ The cost is ``ndet`` applications of ``apply_h`` — unlike ``mcscf.preopt``, which has the
#: matrix in hand already — so this is deliberately small.
DENSE_SOLVE_MAX_DET = 250
#: Norm below which a preconditioned direction is judged to lie in the existing subspace and
#: is dropped, measured *after* normalization so it is a relative test.
LINDEP_TOL = 1.0e-6
#: ⚠ **Generic vectors prepended to every initial subspace, and they are a CORRECTNESS
#: requirement, not a convergence aid.** The rest of the guess is unit vectors on the
#: lowest-diagonal determinants, which is a *biased* set: it can lie entirely inside a few
#: invariant subspaces of ``H``, and the Krylov expansion can never leave the invariant
#: subspaces it starts in. The solver then converges every residual, reports success, and
#: returns eigenpairs that are **not the lowest** — no gate anywhere sees it.
#:
#: Measured on Dy(3+)'s spin-free CAS(9, 14): with SOC off the Hamiltonian conserves ``Sz``,
#: the determinants split into six ``2Sz`` sectors, and the 132 lowest-diagonal ones contain no
#: member of ``2Sz = +-1`` at all — so a 66-root solve missed 22 of the true lowest 66 and came
#: back "converged" 6766 cm^-1 too high. Asking for 134 roots, hence 268 guesses, happened to
#: reach every sector and was right; asking for 74 was wrong by 17590 cm^-1. ⚠ **The wrong
#: answer converged in a THIRD of the iterations of the right one**, so speed is evidence of
#: nothing here.
#:
#: One generic vector is enough and the argument is Rayleigh-Ritz, not luck: if the subspace
#: has a component on every eigenvector then the lowest Ritz values bound the true lowest
#: eigenvalues from above, so a converged set cannot skip one. Four are used because the margin
#: costs four applications of ``H`` and removes any reliance on a single draw.
#:
#: ⚠ **Stated honestly: this makes the failure structurally unavailable, not impossible.** The
#: interlacing argument is about the subspace, and the collapse keeps only the lowest
#: ``2 * n_roots`` Ritz vectors — a low eigenvector reached with a very small component is
#: retained because its Ritz value interlaces below the ones kept, but that is an argument
#: rather than a bound. The only definitive check on a suspect spectrum is still a dense
#: diagonalization of :func:`kuiva.ci.strings.hamiltonian_matrix`, which is what the test of
#: this behaviour uses and what is cheap enough to reach for whenever a spectrum surprises you.
N_GENERIC_GUESS = 4
#: ⚠ **Fixed**, because the event-gated optimizer needs ``solve`` at the same integrals to be deterministic — an
#: optimizer that reads solver noise as a surface change is the failure that section exists to
#: prevent. Do not make this a caller option without saying what breaks.
#:
#: ⚠ They are added on a **cold** start only. A supplied ``guess`` is a converged set from a
#: nearby Hamiltonian and already spans what its own cold solve reached, so the bias being
#: corrected is the fallback's, not the caller's — and ``dmrg/sweep.py`` supplies a guess at
#: every bond of every sweep, where an extra application of ``H_eff`` is the expensive object.
GUESS_SEED = 20260808
#: Floor on ``|theta - H_II|`` in the preconditioner. Without it a determinant whose diagonal
#: sits on the Ritz value produces an infinite correction: the classic Davidson blow-up.
PRECONDITIONER_FLOOR = 1.0e-8


@dataclass
class DavidsonResult:
    """Converged eigenpairs and what it cost to get them."""

    energies: np.ndarray                 # (n_roots,) real, ascending
    vectors: np.ndarray                  # (n_roots, ndet) complex, orthonormal rows
    residuals: np.ndarray                # (n_roots,) final residual norms
    n_iter: int
    n_apply: int                         # applications of H -- the only real cost
    converged: bool
    dense: bool = False                  # solved by dense eigh rather than by iterating

    def __repr__(self) -> str:
        return ("DavidsonResult(n_roots={}, E0={:.10f}, max|r|={:.2e}, iterations={}, "
                "H applications={})".format(self.energies.size, self.energies[0],
                                            float(np.max(self.residuals)), self.n_iter,
                                            self.n_apply))


def davidson_workspace_gb(ndet: int, max_subspace: int) -> float:
    """Size [GB] of the expansion, sigma-vector and conjugate stacks (exact sizing function).

    **Three** ``(max_subspace, ndet)`` complex arrays:

    * ``V``, the expansion subspace;
    * ``H V``, kept because a collapse rebuilds the Ritz vectors *and* their images from what
      is already stored, at no further cost in ``apply_h``;
    * ``conj(V)``, maintained one row at a time as vectors are added.

    ⚠ The third looks like waste and is not. Every subspace contraction here is of the form
    ``sum_K conj(V[i,K]) X[j,K]``, and NumPy has no lazy conjugate: written directly it forms
    a ``.conj()`` copy of the **whole stack** on every iteration — hundreds of MB, unaccounted,
    inside the loop. Keeping the conjugate makes each contraction a plain GEMM and each update
    a single 3 MB row. The alternative is blocking the contraction over determinants, which is
    more code for less clarity and the same memory.
    """
    return 3.0 * res.array_gb((max_subspace, ndet), np.complex128)


def subspace_cap(n_roots: int, ndet: int, max_subspace: Optional[int] = None) -> int:
    """The subspace size the solver will grow to before collapsing.

    ⚠ Floored at ``2 * n_roots + SUBSPACE_MARGIN`` whatever the caller asks for. A caller
    trying to save memory by asking for ``n_roots + 1`` would be reintroducing that failure,
    and the memory this costs is small against the ``F``/``G`` residency of ``ci/sigma.py``.
    """
    floor = 2 * n_roots + SUBSPACE_MARGIN
    requested = SUBSPACE_FACTOR * n_roots if max_subspace is None else int(max_subspace)
    return int(min(ndet, max(floor, requested)))


def _orthonormalize(vec: np.ndarray, basis: np.ndarray,
                    basis_conj: np.ndarray) -> Optional[np.ndarray]:
    """Project ``vec`` out of the orthonormal rows of ``basis``; ``None`` if it dissolves.

    ``basis_conj`` is ``conj(basis)``, maintained by the caller — see
    :func:`davidson_workspace_gb` for why it is kept rather than recomputed.

    Modified Gram-Schmidt with **one re-orthogonalization pass**. The second pass is not
    optional bookkeeping: in complex arithmetic on a near-degenerate cluster a single pass
    leaves a component of order ``eps * kappa`` behind, and that component lies along exactly
    the direction the solver is trying to resolve.
    """
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    work = vec / norm
    for _ in range(2):
        if basis.shape[0]:
            work -= basis.T @ (basis_conj @ work)
        remaining = float(np.linalg.norm(work))
        if remaining < LINDEP_TOL:
            return None
        work /= remaining
    return work


def _initial_subspace(diagonal: np.ndarray, n_roots: int, n_guess: int,
                      guess: Optional[np.ndarray], space: np.ndarray,
                      space_conj: np.ndarray) -> int:
    """Fill the preallocated buffers with orthonormal starting vectors; return how many.

    ⚠ On a **cold** start, :data:`N_GENERIC_GUESS` generic vectors go first — see that constant,
    which carries the whole argument. The unit vectors below are a biased set that can sit
    inside a few invariant subspaces of ``H``, and a Krylov method cannot leave the invariant
    subspaces it starts in; the generic vectors are what make the answer the *lowest* eigenpairs
    rather than the lowest of whichever sectors the guess happened to touch. They go first
    because the starting set is truncated at ``target``, so anything appended is what gets
    dropped — appending them measured as a null result and looked like evidence the remedy did
    not work.

    ⚠ **A supplied ``guess`` suppresses them, deliberately.** A warm start is a converged set
    from a nearby Hamiltonian, so it already spans the sectors its own cold solve reached, and
    the bias being corrected is the *fallback* below. The cost matters: ``dmrg/sweep.py`` calls
    this once per bond of every sweep with the current tensors as the guess, where one
    application of ``H_eff`` is the expensive object, and padding there would buy nothing. It is
    also what keeps every warm-started result bitwise unchanged.

    Then the supplied guess, padded with unit vectors on the **lowest diagonal elements** — for
    a CI Hamiltonian, the lowest-energy determinants. Twice the root count by default, because a
    Kramers-degenerate cluster wants room around it from the first iteration.
    """
    ndet = int(diagonal.size)
    target = min(space.shape[0], max(n_roots, n_guess))
    size = 0

    def accept(vec: np.ndarray) -> None:
        nonlocal size
        direction = _orthonormalize(vec, space[:size], space_conj[:size])
        if direction is not None:
            space[size] = direction
            space_conj[size] = np.conj(direction)
            size += 1

    if guess is None:
        rng = np.random.default_rng(GUESS_SEED)
        for _ in range(min(N_GENERIC_GUESS, target)):
            accept(rng.standard_normal(ndet) + 1j * rng.standard_normal(ndet))
    if guess is not None:
        supplied = np.atleast_2d(np.asarray(guess, dtype=np.complex128))
        if supplied.shape[1] != ndet:
            raise ValueError("guess vectors have length {}, expected {}"
                             .format(supplied.shape[1], ndet))
        for vec in supplied[:target]:
            accept(np.array(vec, dtype=np.complex128, copy=True))
    unit = np.zeros(ndet, dtype=np.complex128)
    for index in np.argsort(diagonal):
        if size >= target:
            break
        unit[:] = 0.0
        unit[index] = 1.0
        accept(unit.copy())
    if size < n_roots:
        raise SolverFailure("could not build {} independent starting vectors in a space of "
                            "{} determinants".format(n_roots, ndet))
    return size


def _dense_solve(apply_h: Callable[[np.ndarray], np.ndarray], ndet: int, n_roots: int
                 ) -> DavidsonResult:
    """Build ``H`` by applying it to every unit vector and diagonalize it exactly.

    Used below :data:`DENSE_SOLVE_MAX_DET` and whenever the requested roots approach the
    dimension of the space, where an iterative solver has nothing to iterate on. Exact, needs
    no tuning, and cannot fail to converge — which is what makes it the right answer for the
    small Kramers-degenerate cases the iterative path finds hardest.
    """
    res.require("dense CI Hamiltonian", 2.0 * res.array_gb((ndet, ndet), np.complex128),
                note="{} determinants; the matrix plus LAPACK's copy".format(ndet),
                advice=["this path is taken only below {} determinants or when the requested "
                        "root count approaches the space size".format(DENSE_SOLVE_MAX_DET)])
    matrix = np.empty((ndet, ndet), dtype=np.complex128)
    unit = np.zeros(ndet, dtype=np.complex128)
    with timer("dense CI diagonalization"):
        for column in range(ndet):
            unit[:] = 0.0
            unit[column] = 1.0
            matrix[:, column] = apply_h(unit)
        matrix = 0.5 * (matrix + matrix.conj().T)
        values, vectors = np.linalg.eigh(matrix)
    keep = min(n_roots, ndet)
    return DavidsonResult(energies=np.real(values[:keep]).copy(),
                          vectors=np.ascontiguousarray(vectors[:, :keep].T),
                          residuals=np.zeros(keep), n_iter=0, n_apply=ndet,
                          converged=True, dense=True)


def davidson(apply_h: Callable[[np.ndarray], np.ndarray], diagonal: np.ndarray,
             n_roots: int = 1, *, guess: Optional[np.ndarray] = None,
             conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
             max_subspace: Optional[int] = None, n_guess: Optional[int] = None,
             dense_max_det: int = DENSE_SOLVE_MAX_DET, label: str = "CI",
             level: int = logging.DEBUG) -> DavidsonResult:
    """Lowest ``n_roots`` eigenpairs of a complex-Hermitian matrix, matrix-free.

    Parameters
    ----------
    apply_h : callable
        ``c -> H c`` for a single ``(ndet,)`` complex vector. Applied once per new expansion
        direction per iteration; a collapse costs none, because the images of the Ritz vectors
        are recovered from the stored ones.
    diagonal : ``(ndet,)`` real
        ``<I|H|I>``, for the preconditioner and the initial guess. From
        :func:`kuiva.ci.strings.diagonal_energies`.
    n_roots : int
        ⚠ With an odd electron count every level is Kramers doubled, so an odd ``n_roots``
        necessarily splits a degenerate pair. That is not refused here — it is a well-posed
        eigenvalue request — but it *is* refused where the state-averaged RDMs are built
        (:mod:`kuiva.rdm.rdm`), which is the point at which it would silently make the
        converged orbitals depend on an arbitrary rotation inside the pair.
    guess : ``(m, ndet)`` complex, optional
        Warm start, typically the previous macro-iteration's vectors (checkpoints store exactly "what
        is needed to restart Davidson from a good guess"). Need not be orthonormal, need not
        number ``n_roots``: it is orthonormalized and padded here.
    conv_tol : float
        Residual-norm threshold, applied to **every** requested root.
    level : int
        Logging level of the iteration table. Davidson iterations are *micro*-iterations, so
        the default is DEBUG; a standalone CASCI driver may raise it to INFO.

    Returns
    -------
    :class:`DavidsonResult`

    Raises
    ------
    :class:`~kuiva.util.errors.SolverFailure`
        If ``max_iter`` is reached, or the subspace cannot be extended, without every root
        meeting ``conv_tol``. It never returns an unconverged vector.
    """
    diagonal = np.ascontiguousarray(np.real(diagonal), dtype=np.float64)
    ndet = int(diagonal.size)
    n_roots = int(n_roots)
    if not 1 <= n_roots <= ndet:
        raise ValueError("cannot ask for {} roots of a {}-determinant space"
                         .format(n_roots, ndet))
    if ndet <= dense_max_det or n_roots >= ndet - 1:
        return _dense_solve(apply_h, ndet, n_roots)

    cap = subspace_cap(n_roots, ndet, max_subspace)
    n_guess = min(cap, 2 * n_roots if n_guess is None else int(n_guess))
    # ⚠ Reserved before the stacks exist (the refusal must cost nothing), then owned by them:
    # this function runs once per CI solve — twice per CASSCF macro-iteration — and its
    # stacks die on return, so a reservation with no release would grow the ledger by one
    # subspace per solve and refuse a long optimization that fits at every instant
    # (invisible so far only because every committed CASSCF is tiny here or dense-solved).
    alloc = res.reserve("Davidson subspace ({} roots, {} determinants)".format(n_roots, ndet),
                        davidson_workspace_gb(ndet, cap),
                        note="{} expansion vectors and their images".format(cap),
                        advice=["reduce the number of states, or max_subspace (it is floored "
                                "at 2 * n_roots + {}, which is what separates a "
                                "Kramers-degenerate cluster and may not be lowered further)"
                                .format(SUBSPACE_MARGIN)])

    space = np.zeros((cap, ndet), dtype=np.complex128)
    res.owned_by(space, alloc)
    space_conj = np.zeros((cap, ndet), dtype=np.complex128)
    images = np.zeros((cap, ndet), dtype=np.complex128)
    size = _initial_subspace(diagonal, n_roots, n_guess, guess, space, space_conj)
    n_apply = 0
    with timer("Davidson: initial sigma vectors"):
        for i in range(size):
            images[i] = apply_h(space[i])
            n_apply += 1

    table = out.Table(log, [out.col_iter(), out.col_count("space", 7),
                            out.col_energy("E(lowest) [Eh]"), out.col_delta(),
                            out.col_resid("max|r|"), out.col_time()], level=level)
    table.start("{} Davidson, {} roots of {} determinants".format(label, n_roots, ndet))

    energies = np.zeros(n_roots)
    ritz = np.zeros((n_roots, ndet), dtype=np.complex128)
    norms = np.full(n_roots, np.inf)
    previous = np.nan
    converged = False
    iteration = 0
    tic = time.time()

    with timer("Davidson iterations"):
        for iteration in range(1, max_iter + 1):
            # Rayleigh-Ritz. The subspace matrix is Hermitian by construction; symmetrizing
            # it costs nothing and keeps eigh from seeing rounding asymmetry as structure.
            gram = space_conj[:size] @ images[:size].T
            gram = 0.5 * (gram + gram.conj().T)
            values, coefficients = np.linalg.eigh(gram)
            energies = np.ascontiguousarray(np.real(values[:n_roots]))
            ritz = coefficients[:, :n_roots].T @ space[:size]
            ritz_images = coefficients[:, :n_roots].T @ images[:size]
            residual = ritz_images - energies[:, None] * ritz
            norms = np.linalg.norm(residual, axis=1)

            delta = energies[0] - previous
            table.row(iteration, size, energies[0], 0.0 if np.isnan(delta) else delta,
                      float(np.max(norms)), time.time() - tic)
            previous = energies[0]
            if float(np.max(norms)) < conv_tol:
                converged = True
                break

            unconverged = np.nonzero(norms >= conv_tol)[0]
            # Collapse *before* extending, so the new directions are added to a fresh basis
            # rather than being the ones squeezed out by the cap. The images come along for
            # free -- H(V y) = (H V) y -- so a restart costs no applications of H at all.
            if size + unconverged.size > cap:
                n_keep = min(size, max(n_roots, min(2 * n_roots, cap - unconverged.size)))
                space[:n_keep] = coefficients[:, :n_keep].T @ space[:size]
                images[:n_keep] = coefficients[:, :n_keep].T @ images[:size]
                space_conj[:n_keep] = np.conj(space[:n_keep])
                size = n_keep
                log.debug("Davidson collapsed to %d vectors at iteration %d",
                          n_keep, iteration)

            added = 0
            for root in unconverged:
                if size >= cap:
                    break
                # Davidson's diagonal preconditioner, floored: without the floor a determinant
                # whose diagonal sits on the Ritz value gives an infinite correction.
                gap = energies[root] - diagonal
                gap[np.abs(gap) < PRECONDITIONER_FLOOR] = PRECONDITIONER_FLOOR
                direction = _orthonormalize(residual[root] / gap, space[:size],
                                            space_conj[:size])
                if direction is None:
                    continue
                space[size] = direction
                space_conj[size] = np.conj(direction)
                images[size] = apply_h(direction)
                n_apply += 1
                size += 1
                added += 1
            if added == 0:
                table.end("no independent direction left to expand")
                raise SolverFailure(
                    "{} Davidson stalled at iteration {}: every preconditioned direction "
                    "for the {} unconverged root(s) already lay in the {}-vector subspace "
                    "(max|r| = {:.2e} against conv_tol = {:.1e})".format(
                        label, iteration, unconverged.size, size, float(np.max(norms)),
                        conv_tol))

    if converged:
        table.end("converged in {} iterations, {} applications of H".format(
            iteration, n_apply))
    else:
        table.end("NOT converged")
        raise SolverFailure(
            "{} Davidson did not converge {} roots of {} determinants in {} iterations: "
            "max|r| = {:.2e} against conv_tol = {:.1e}".format(
                label, n_roots, ndet, max_iter, float(np.max(norms)), conv_tol))

    log.debug("%s Davidson: %d roots, %d iterations, %d H applications, max|r| = %.2e",
              label, n_roots, iteration, n_apply, float(np.max(norms)))
    return DavidsonResult(energies=energies, vectors=np.ascontiguousarray(ritz),
                          residuals=norms, n_iter=iteration, n_apply=n_apply,
                          converged=True)


# --- The Kramers-restricted (time-reversal adapted) path -----------------------------------
#
# ⚠ **Read this before changing anything below.** With an odd electron count ``T^2 = -1``, so
# ``T`` pairs every CI vector with an orthogonal partner and ``H`` — which commutes with ``T``
# — is *self-dual*: in the basis ``(v_1..v_k, T v_1..T v_k)`` of any T-closed subspace it reads
#
#     M = [[A, B], [-conj(B), conj(A)]],     A Hermitian,   B antisymmetric
#
# whose eigenvalues are **exactly** doubly degenerate. That is the quaternion structure of
# the second symmetry mode of the two-component CI, and this is where it lives.
#
# ⚠ **Where the factor of two comes from, stated so nobody looks for it in the wrong place.**
# It is *not* a shorter vector: one quaternion expansion direction is one full-length complex
# array, and applying ``H`` to it is one ordinary sigma-vector call — the operator below is
# the unchanged :class:`~kuiva.ci.sigma.SigmaOperator` and its ``F``/``G`` residency does not
# move. What halves is the **subspace**: storing ``v`` implicitly stores ``T v``, so ``k``
# stored vectors span ``2k`` complex dimensions and ``k`` applications of ``H`` deliver the
# images of all ``2k``. For a state average over whole Kramers pairs — the only kind the
# state-averaging gate accepts on an odd-electron system — that is half the storage and half
# the applications of ``H`` for the same spectrum. Whether it is half the *time* is a
# measurement, recorded in this package's validation record, not a claim.
#
# ⚠ **This is deliberately not where the obvious design would have put it.** The natural
# reading is to put the structure in the *operator*, on half-length vectors over canonical
# representative determinants, leaving this module untouched. It does not work: ``sigma_I`` for a
# representative ``I`` sums over intermediates ``K`` scattered across the *whole* space, so a
# half-space operator still needs the full ``F`` and buys nothing, while costing a second
# excitation-map contract and a new addressing scheme. The structure is a property of the
# invariant *subspace*, not of the operator, so it belongs to the eigensolver — and putting it
# here leaves ``ci/sigma.py``, ``ci/strings.py``'s excitation map and every registered kernel
# bitwise unchanged, which is worth more than the symmetry of the two code paths.
#
# ⚠ **The Kramers degeneracy this delivers exactly is a side effect and is never the argument
# for the mode.** The general path's measured splitting is 1e-15..1e-13 Eh, five orders below
# the band reserved for a genuine numerical splitting, so nothing here is sold on it.
#
# References for the self-dual/quaternion eigenvalue structure and its use in two-component
# CI: J. Thyssen, T. Fleig, H. J. Aa. Jensen, J. Chem. Phys. 129, 034109 (2008),
# doi:10.1063/1.2943670; T. Saue, H. J. Aa. Jensen, J. Chem. Phys. 111, 6211 (1999),
# doi:10.1063/1.479958; the quaternion (self-dual) diagonalization itself, A. Bunse-Gerstner,
# R. Byers, V. Mehrmann, SIAM J. Matrix Anal. Appl. 10, 419 (1989), doi:10.1137/0610030.

#: Norm below which a candidate Ritz vector is judged to lie in the span of the pairs already
#: selected, i.e. to be the *partner* of one of them rather than a new pair.
#:
#: ⚠ Deliberately a wide separator (0.5) rather than a tight tolerance, because the quantity it
#: cuts is bimodal by construction: the subspace matrix is assembled self-dual to roundoff, so
#: a partner's residual is ~1e-14 and a genuinely new pair's is ~1. Anything between the two
#: would mean the structure had been lost, and the run says so rather than rounding.
KRAMERS_PAIR_TOL = 0.5


# --- Sector-restricted solve --------------------------------------------------------------

def davidson_sector(apply_h: Callable[[np.ndarray], np.ndarray], diagonal: np.ndarray,
                    mask: np.ndarray, n_roots: int = 1, *,
                    guess: Optional[np.ndarray] = None,
                    label: str = "CI", **kwargs) -> DavidsonResult:
    """Lowest ``n_roots`` eigenpairs **within one symmetry sector** of the determinant space.

    ``mask`` is a boolean over determinants; ``H`` is assumed to preserve it, which it does
    exactly when the orbitals are symmetry-pure (:mod:`kuiva.symm.sectors` — the caller is
    what checks that, not this function).

    The solve runs in the **compressed** space: the eigenproblem, the subspace stacks, the
    preconditioner, the dense fallback and the guess padding all see only the sector's
    determinants, so a request for ``n`` roots of a sector costs ``n`` roots rather than
    however many roots of the full spectrum happen to lie below the sector's ``n``-th. The
    price is one gather and one scatter per application of ``H``, which is a copy against a
    contraction and does not show up in a profile.

    ⚠ **This is where the biased-guess failure is fixed structurally rather than mitigated.**
    A Krylov method cannot leave the invariant subspaces its starting vectors lie in, and a
    conserved label diagonal in the determinant basis is exactly such a subspace; solving each
    sector on its own means the guess *cannot* miss a sector, whereas the lowest-diagonal
    determinants of the full space can all lie in one of them.

    Returns a :class:`DavidsonResult` whose vectors are **full length**, zero outside the
    sector — the same object in the same convention as every other path here, so nothing
    downstream learns that a sector was solved.
    """
    diagonal = np.ascontiguousarray(np.real(diagonal), dtype=np.float64)
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.shape != diagonal.shape:
        raise ValueError("the sector mask covers {} determinants and the diagonal {}"
                         .format(mask.size, diagonal.size))
    index = np.nonzero(mask)[0]
    ndet_sector = int(index.size)
    n_roots = int(n_roots)
    if not 1 <= n_roots <= ndet_sector:
        raise ValueError("cannot ask for {} roots of a sector holding {} determinants"
                         .format(n_roots, ndet_sector))
    scratch = np.zeros(diagonal.size, dtype=np.complex128)

    def apply_compressed(c: np.ndarray) -> np.ndarray:
        scratch[:] = 0.0
        scratch[index] = c
        return np.ascontiguousarray(apply_h(scratch)[index])

    # ⚠ **The dense fallback is decided on the FULL determinant count, not on the sector's.**
    # A dense solve costs ``ndet`` applications of ``H``, and here an application costs the
    # full-space price whatever the sector's size — so a sector small enough to look "dense"
    # would be bought with hundreds of full-space sigma vectors. Measured on a 495-determinant
    # CAS split into four sectors: the dense route spent 375 applications against the
    # iterative path's 163 for the same states, and twice the CPU.
    kwargs.setdefault("dense_max_det",
                      DENSE_SOLVE_MAX_DET if diagonal.size <= DENSE_SOLVE_MAX_DET else 0)

    sub_guess = None
    if guess is not None:
        supplied = np.atleast_2d(np.asarray(guess, dtype=np.complex128))[:, index]
        keep = np.linalg.norm(supplied, axis=1) > 1e-10
        if np.any(keep):
            sub_guess = np.ascontiguousarray(supplied[keep])
    result = davidson(apply_compressed, diagonal[index], n_roots, guess=sub_guess,
                      label=label, **kwargs)
    vectors = np.zeros((n_roots, diagonal.size), dtype=np.complex128)
    vectors[:, index] = result.vectors
    return DavidsonResult(energies=result.energies, vectors=vectors,
                          residuals=result.residuals, n_iter=result.n_iter,
                          n_apply=result.n_apply, converged=result.converged,
                          dense=result.dense)


def davidson_kramers_sector(apply_h: Callable[[np.ndarray], np.ndarray], kramers,
                            diagonal: np.ndarray, mask: np.ndarray, n_pairs: int = 1, *,
                            guess: Optional[np.ndarray] = None,
                            label: str = "CI", **kwargs) -> DavidsonResult:
    """Lowest ``n_pairs`` Kramers pairs **within one time-reversal-closed symmetry subspace**.

    The two symmetries of this Hamiltonian commute, so they can be imposed together — but not
    on the same object. ⚠ **Time reversal conjugates an irrep label**, so a sector is mapped
    onto its *conjugate* sector and is not itself ``T``-closed unless it is self-conjugate;
    the union of a conjugate pair is. ``mask`` is therefore that union (or a self-conjugate
    sector), and :meth:`kuiva.ci.strings.KramersMap.restrict` refuses anything else rather
    than silently solving in a subspace the operator leaves.

    What comes back is the ordinary object every other path here returns: ``2 * n_pairs``
    **full-length** vectors, zero outside the subspace, pair-expanded and ascending. The
    caller cannot tell that two symmetries were used at once, which is the whole point —
    the state-averaging gate, the boundary diagnostic and the RDM builder are unchanged.

    ⚠ The states of a pair need **not** be sector-pure individually: a Kramers pair spanning
    two conjugate sectors is exactly degenerate, so any rotation inside it is as good an
    eigenvector, and half of it lands in each sector. That is the eigensolver's freedom and
    not a broken symmetry, which is why classification of such a solve is per degenerate
    block and never per state.
    """
    diagonal = np.ascontiguousarray(np.real(diagonal), dtype=np.float64)
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.shape != diagonal.shape:
        raise ValueError("the sector mask covers {} determinants and the diagonal {}"
                         .format(mask.size, diagonal.size))
    index = np.nonzero(mask)[0]
    restricted = kramers.restrict(mask)
    scratch = np.zeros(diagonal.size, dtype=np.complex128)

    def apply_compressed(c: np.ndarray) -> np.ndarray:
        scratch[:] = 0.0
        scratch[index] = c
        return np.ascontiguousarray(apply_h(scratch)[index])

    # ⚠ Decided on the FULL determinant count, exactly as in :func:`davidson_sector` and for
    # the same measured reason: an application of H costs the whole space whatever the
    # subspace's size, so a subspace small enough to look "dense" would be bought with
    # hundreds of full-space sigma vectors.
    kwargs.setdefault("dense_max_det",
                      DENSE_SOLVE_MAX_DET if diagonal.size <= DENSE_SOLVE_MAX_DET else 0)
    sub_guess = None
    if guess is not None:
        supplied = np.atleast_2d(np.asarray(guess, dtype=np.complex128))[:, index]
        keep = np.linalg.norm(supplied, axis=1) > 1e-10
        if np.any(keep):
            sub_guess = np.ascontiguousarray(supplied[keep])
    result = davidson_kramers(apply_compressed, restricted, diagonal[index], n_pairs,
                              guess=sub_guess, label=label, **kwargs)
    vectors = np.zeros((result.vectors.shape[0], diagonal.size), dtype=np.complex128)
    vectors[:, index] = result.vectors
    return DavidsonResult(energies=result.energies, vectors=vectors,
                          residuals=result.residuals, n_iter=result.n_iter,
                          n_apply=result.n_apply, converged=result.converged,
                          dense=result.dense)


def _kramers_dual(coefficients: np.ndarray) -> np.ndarray:
    """``(a, b) -> (-conj(b), conj(a))``: a subspace vector's time-reversed partner.

    In the ``(v_i, T v_i)`` basis, ``T(sum a_i v_i + sum b_i T v_i) = sum -conj(b_i) v_i +
    sum conj(a_i) T v_i``, using ``T^2 = -1`` and the antilinearity of ``T``. This map is
    antiunitary on the coefficient space, which is why a selected set closed under it stays
    orthonormal without a second Gram-Schmidt pass.
    """
    k = coefficients.shape[-1] // 2
    return np.concatenate([-np.conj(coefficients[..., k:]),
                           np.conj(coefficients[..., :k])], axis=-1)


def _orthonormalize_self_dual(vec: np.ndarray, basis: np.ndarray, basis_conj: np.ndarray,
                              time_reverse) -> Optional[np.ndarray]:
    """:func:`_orthonormalize`, but against ``basis`` **and its time reverses**.

    The stored basis is only half of the subspace it spans, so a new direction has to be
    projected out of both halves or the "orthonormal" expansion silently develops an overlap
    with the partners — which shows up as a subspace matrix that is no longer self-dual and
    hence as Kramers pairs that no longer come out degenerate.

    ``<T v_i|w> = -<T w|v_i>`` (the ``T^2 = -1`` identity), so the second projection needs one
    time reversal of ``w`` and no stored partner stack; and
    ``sum_i c_i T v_i = T(sum_i conj(c_i) v_i)``, so removing it needs one more.
    """
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    work = vec / norm
    for _ in range(2):
        if basis.shape[0]:
            work -= basis.T @ (basis_conj @ work)
            overlap = -(basis @ np.conj(time_reverse(work)))
            work -= time_reverse(np.conj(overlap) @ basis)
        remaining = float(np.linalg.norm(work))
        if remaining < LINDEP_TOL:
            return None
        work /= remaining
    return work


def _select_kramers_pairs(matrix: np.ndarray, values: np.ndarray, vectors: np.ndarray,
                          n_pairs: int, label: str):
    """One member of each of the lowest ``n_pairs`` Kramers pairs of a self-dual subspace.

    ``eigh`` returns the exactly-degenerate pairs in an arbitrary basis, so picking "every
    other eigenvector" is wrong the moment two pairs are degenerate with each other (a 4-fold
    level: eigenvectors 0 and 2 are orthogonal, but the *partner* of 0 need not be orthogonal
    to 2). Instead: walk the eigenvectors in ascending order, project out everything already
    selected **and its partners**, and take what survives. Inside an exactly degenerate
    eigenspace the survivor is still an eigenvector, so nothing is lost.

    Returns ``(energies, coefficients)`` with ``coefficients`` the ``(n_pairs, 2k)`` selected
    members, each one's partner being :func:`_kramers_dual` of it.
    """
    dimension = vectors.shape[0]
    chosen = np.zeros((2 * n_pairs, dimension), dtype=np.complex128)
    picked = np.zeros((n_pairs, dimension), dtype=np.complex128)
    energies = np.zeros(n_pairs)
    count = 0
    found = 0
    for column in range(vectors.shape[1]):
        if found == n_pairs:
            break
        work = np.array(vectors[:, column], dtype=np.complex128, copy=True)
        for _ in range(2):
            if count:
                work -= chosen[:count].T @ (np.conj(chosen[:count]) @ work)
        norm = float(np.linalg.norm(work))
        if norm < KRAMERS_PAIR_TOL:
            continue
        work /= norm
        picked[found] = work
        # The Rayleigh quotient of the vector actually used, not the eigenvalue it came from:
        # inside a degenerate eigenspace they agree, and where they do not, this is the honest
        # one and the difference is what the residual test will see.
        energies[found] = float(np.real(np.vdot(work, matrix @ work)))
        chosen[count] = work
        chosen[count + 1] = _kramers_dual(work)
        count += 2
        found += 1
    if found < n_pairs:
        raise SolverFailure(
            "{} Kramers-restricted Davidson: the {}-dimensional subspace yielded only {} of "
            "the {} time-reversal pairs asked for. The subspace matrix should be exactly "
            "self-dual, so this means the expansion is no longer closed under time reversal — "
            "check that the active-space integrals are time-reversal symmetric"
            .format(label, dimension, found, n_pairs))
    return energies, picked


def _dense_solve_kramers(apply_h, kramers, ndet: int, n_pairs: int) -> DavidsonResult:
    """Build the self-dual subspace matrix on the whole space and diagonalize it exactly.

    The complete basis is ``(e_r, T e_r)`` over the ``ndet/2`` representative determinants, so
    this costs **``ndet/2`` applications of ``H``** where :func:`_dense_solve` costs ``ndet`` —
    the same halving the iterative path gets, in the one place where it is a certainty rather
    than a measurement.
    """
    reps = kramers.representatives
    m = int(reps.size)
    res.require("dense Kramers-restricted CI Hamiltonian",
                2.0 * res.array_gb((2 * m, 2 * m), np.complex128),
                note="{} determinants; the self-dual matrix plus LAPACK's copy".format(ndet),
                advice=["this path is taken only below {} determinants or when the requested "
                        "pair count approaches half the space size"
                        .format(DENSE_SOLVE_MAX_DET)])
    a_block = np.empty((m, m), dtype=np.complex128)
    b_block = np.empty((m, m), dtype=np.complex128)
    unit = np.zeros(ndet, dtype=np.complex128)
    with timer("dense Kramers-restricted CI diagonalization"):
        for column in range(m):
            unit[:] = 0.0
            unit[reps[column]] = 1.0
            image = apply_h(unit)
            a_block[:, column] = image[reps]                      # <e_i|H|e_j>
            b_block[:, column] = kramers.time_reverse(image)[reps]  # <e_i|T H|e_j>
        matrix = _self_dual_matrix(a_block, b_block)
        values, vectors = np.linalg.eigh(matrix)
    energies, coefficients = _select_kramers_pairs(matrix, values, vectors, n_pairs, "CI")

    states = np.zeros((2 * n_pairs, ndet), dtype=np.complex128)
    scratch = np.zeros(ndet, dtype=np.complex128)
    for root in range(n_pairs):
        a_part, b_part = coefficients[root, :m], coefficients[root, m:]
        states[2 * root, reps] = a_part
        scratch[:] = 0.0
        scratch[reps] = np.conj(b_part)
        states[2 * root] += kramers.time_reverse(scratch)
        states[2 * root + 1] = kramers.time_reverse(states[2 * root])
    return DavidsonResult(energies=np.repeat(energies, 2),
                          vectors=np.ascontiguousarray(states),
                          residuals=np.zeros(2 * n_pairs), n_iter=0, n_apply=m,
                          converged=True, dense=True)


def _self_dual_matrix(a_block: np.ndarray, b_block: np.ndarray) -> np.ndarray:
    """``[[A, B], [-conj(B), conj(A)]]`` with ``A`` Hermitized and ``B`` antisymmetrized.

    ⚠ The projection is not cosmetic. Assembling the matrix from ``A`` and ``B`` rather than
    from all four blocks is what makes it self-dual to *roundoff in these two operations only*,
    and that is what makes ``eigh``'s eigenvalues degenerate to machine precision — which is in
    turn what :data:`KRAMERS_PAIR_TOL` relies on being able to cut on.
    """
    a_block = 0.5 * (a_block + a_block.conj().T)
    b_block = 0.5 * (b_block - b_block.T)
    return np.block([[a_block, b_block],
                     [-b_block.conj(), a_block.conj()]])


def _initial_subspace_kramers(diagonal: np.ndarray, n_pairs: int, n_guess: int,
                              guess: Optional[np.ndarray], space: np.ndarray,
                              space_conj: np.ndarray, time_reverse) -> int:
    """:func:`_initial_subspace` against a self-dual basis; returns the pair count accepted.

    ⚠ :data:`N_GENERIC_GUESS`'s argument survives unchanged and is if anything sharper here:
    a Krylov method cannot leave the invariant subspaces it starts in, and time reversal is
    not the only symmetry a CI Hamiltonian can have. The unit vectors below are still the
    biased fallback they always were — what the Kramers structure removes is one *particular*
    way of missing a sector (a guess whose partner sector is unreachable), not the general
    failure. A supplied guess suppresses them for the same reason as in the general path.

    A supplied guess may be the **pair-expanded** vectors of a previous solve: a partner
    dissolves against the direction it partners and is dropped, at the cost of one
    orthogonalization each.
    """
    ndet = int(diagonal.size)
    target = min(space.shape[0], max(n_pairs, n_guess))
    size = 0

    def accept(vec: np.ndarray) -> None:
        nonlocal size
        direction = _orthonormalize_self_dual(vec, space[:size], space_conj[:size],
                                              time_reverse)
        if direction is not None:
            space[size] = direction
            space_conj[size] = np.conj(direction)
            size += 1

    if guess is None:
        rng = np.random.default_rng(GUESS_SEED)
        for _ in range(min(N_GENERIC_GUESS, target)):
            accept(rng.standard_normal(ndet) + 1j * rng.standard_normal(ndet))
    else:
        supplied = np.atleast_2d(np.asarray(guess, dtype=np.complex128))
        if supplied.shape[1] != ndet:
            raise ValueError("guess vectors have length {}, expected {}"
                             .format(supplied.shape[1], ndet))
        for vec in supplied:
            if size >= target:
                break
            accept(np.array(vec, dtype=np.complex128, copy=True))
    unit = np.zeros(ndet, dtype=np.complex128)
    for index in np.argsort(diagonal):
        if size >= target:
            break
        unit[:] = 0.0
        unit[index] = 1.0
        accept(unit.copy())
    if size < n_pairs:
        raise SolverFailure("could not build {} independent time-reversal-closed starting "
                            "directions in a space of {} determinants".format(n_pairs, ndet))
    return size


def davidson_kramers(apply_h: Callable[[np.ndarray], np.ndarray], kramers,
                     diagonal: np.ndarray, n_pairs: int = 1, *,
                     guess: Optional[np.ndarray] = None,
                     conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                     max_subspace: Optional[int] = None, n_guess: Optional[int] = None,
                     dense_max_det: int = DENSE_SOLVE_MAX_DET, label: str = "CI",
                     level: int = logging.DEBUG) -> DavidsonResult:
    """Lowest ``n_pairs`` **Kramers pairs** of a time-reversal-symmetric complex-Hermitian
    matrix, matrix-free.

    Same solver as :func:`davidson` — same preconditioner, same collapse, same stall detector,
    same refusal to return an unconverged vector — over a subspace kept **closed under time
    reversal**. See the block comment above this function for why the structure sits here and
    not in the operator, and for what the factor of two is and is not.

    Parameters
    ----------
    apply_h : callable
        ``c -> H c``, exactly as in :func:`davidson`. ⚠ It **must** commute with ``T``. That is
        a property of the integrals, is not checked here, and is checked by the caller
        (:class:`~kuiva.mcscf.casci.FullCISolver`) against the integrals themselves, where the
        failure can be explained.
    kramers : :class:`~kuiva.ci.strings.KramersMap`
        Time reversal over this determinant space: ``time_reverse`` and ``representatives``.
    diagonal : ``(ndet,)`` real — ``<I|H|I>``, for the preconditioner and the guess.
    n_pairs : int
        Kramers pairs, **not** states. The result carries ``2 * n_pairs`` states.
    max_subspace : int, optional
        ⚠ Counted in **time-reversal-closed directions**, so it bounds a subspace of
        ``2 * max_subspace`` complex dimensions — the same memory as ``max_subspace`` vectors
        of :func:`davidson`, and twice the span. Given the same number in both modes, this one
        holds twice the states; the memory the refusal names is the same either way.
    guess : ``(m, ndet)`` complex, optional
        Warm start. May be the **pair-expanded** vectors of a previous solve: each partner
        dissolves against the direction it partners and is dropped, so a caller does not have
        to know which half to hand over.

    Returns
    -------
    :class:`DavidsonResult`, **pair-expanded**: ``2 * n_pairs`` vectors with state ``2r+1`` the
    time reverse of state ``2r`` and its energy an exact copy of ``2r``'s. ⚠ That is the whole
    of what downstream sees — the state-averaging gate, the boundary diagnostic, ``props/`` and
    the RDMs get the shape the general path produces and learn no new convention.
    """
    diagonal = np.ascontiguousarray(np.real(diagonal), dtype=np.float64)
    ndet = int(diagonal.size)
    n_pairs = int(n_pairs)
    if kramers.parity != -1:
        raise ValueError(
            "the Kramers-restricted path is the odd-electron theorem: T^2 = -1 pairs every "
            "determinant with an orthogonal partner and no determinant is self-conjugate. "
            "This space has {} electrons, where T^2 = +1 and neither holds"
            .format(kramers.n_elec))
    if kramers.ndet != ndet:
        raise ValueError("the time-reversal map covers {} determinants, the diagonal {}"
                         .format(kramers.ndet, ndet))
    if not 1 <= 2 * n_pairs <= ndet:
        raise ValueError("cannot ask for {} Kramers pairs ({} states) of a {}-determinant "
                         "space".format(n_pairs, 2 * n_pairs, ndet))
    time_reverse = kramers.time_reverse

    if ndet <= dense_max_det or 2 * n_pairs >= ndet - 1:
        return _dense_solve_kramers(apply_h, kramers, ndet, n_pairs)

    # The cap counts *quaternion* directions; the complex span is twice it, which is what the
    # floor of the general path has to be compared against — hence 2 * n_pairs = n_states here.
    cap = subspace_cap(n_pairs, ndet // 2, max_subspace)
    n_guess = min(cap, 2 * n_pairs if n_guess is None else int(n_guess))
    # Reserved before the stacks exist, then owned by them — same per-solve lifetime as the
    # general path above, and the same ledger leak without the release.
    alloc = res.reserve("Kramers-restricted Davidson subspace ({} pairs, {} determinants)"
                        .format(n_pairs, ndet), davidson_workspace_gb(ndet, cap),
                        note="{} time-reversal-closed directions (spanning {} states) and "
                             "their images".format(cap, 2 * cap),
                        advice=["reduce the number of states, or max_subspace (it is floored "
                                "at 2 * n_pairs + {})".format(SUBSPACE_MARGIN)])

    space = np.zeros((cap, ndet), dtype=np.complex128)
    res.owned_by(space, alloc)
    space_conj = np.zeros((cap, ndet), dtype=np.complex128)
    images = np.zeros((cap, ndet), dtype=np.complex128)
    size = _initial_subspace_kramers(diagonal, n_pairs, n_guess, guess, space, space_conj,
                                     time_reverse)
    n_apply = 0
    with timer("Kramers Davidson: initial sigma vectors"):
        for i in range(size):
            images[i] = apply_h(space[i])
            n_apply += 1

    table = out.Table(log, [out.col_iter(), out.col_count("space", 7),
                            out.col_energy("E(lowest) [Eh]"), out.col_delta(),
                            out.col_resid("max|r|"), out.col_time()], level=level)
    table.start("{} Kramers-restricted Davidson, {} pairs ({} states) of {} determinants"
                .format(label, n_pairs, 2 * n_pairs, ndet))

    energies = np.zeros(n_pairs)
    ritz = np.zeros((n_pairs, ndet), dtype=np.complex128)
    norms = np.full(n_pairs, np.inf)
    scratch = np.empty(ndet, dtype=np.complex128)
    previous = np.nan
    converged = False
    iteration = 0
    tic = time.time()

    def ritz_vectors(coefficients: np.ndarray, stack: np.ndarray) -> np.ndarray:
        """``sum_i a_i v_i + sum_i b_i T v_i`` for each selected pair member.

        ⚠ The second sum is ``T(sum_i conj(b_i) v_i)`` — ``T`` is antilinear, so the
        coefficients are conjugated *before* it is applied. The same expression with ``H V`` in
        place of ``V`` gives the image, because ``H`` commutes with ``T``: no second
        application of ``H`` anywhere in this function.
        """
        a_part = coefficients[:, :size]
        b_part = coefficients[:, size:]
        return (a_part @ stack[:size]) + time_reverse(np.conj(b_part) @ stack[:size])

    with timer("Kramers Davidson iterations"):
        for iteration in range(1, max_iter + 1):
            a_block = space_conj[:size] @ images[:size].T
            b_block = np.empty((size, size), dtype=np.complex128)
            for column in range(size):
                time_reverse(images[column], out=scratch)
                b_block[:, column] = space_conj[:size] @ scratch
            matrix = _self_dual_matrix(a_block, b_block)
            values, coefficients = np.linalg.eigh(matrix)
            energies, selected = _select_kramers_pairs(matrix, values, coefficients, n_pairs,
                                                       label)
            ritz = ritz_vectors(selected, space)
            residual = ritz_vectors(selected, images) - energies[:, None] * ritz
            norms = np.linalg.norm(residual, axis=1)

            delta = energies[0] - previous
            table.row(iteration, 2 * size, energies[0], 0.0 if np.isnan(delta) else delta,
                      float(np.max(norms)), time.time() - tic)
            previous = energies[0]
            if float(np.max(norms)) < conv_tol:
                converged = True
                break

            unconverged = np.nonzero(norms >= conv_tol)[0]
            if size + unconverged.size > cap:
                n_keep = min(size, max(n_pairs, min(2 * n_pairs, cap - unconverged.size)))
                _, kept = _select_kramers_pairs(matrix, values, coefficients, n_keep, label)
                new_space = ritz_vectors(kept, space)
                new_images = ritz_vectors(kept, images)
                space[:n_keep] = new_space
                images[:n_keep] = new_images
                space_conj[:n_keep] = np.conj(space[:n_keep])
                size = n_keep
                log.debug("Kramers Davidson collapsed to %d directions at iteration %d",
                          n_keep, iteration)

            added = 0
            for root in unconverged:
                if size >= cap:
                    break
                gap = energies[root] - diagonal
                gap[np.abs(gap) < PRECONDITIONER_FLOOR] = PRECONDITIONER_FLOOR
                direction = _orthonormalize_self_dual(residual[root] / gap, space[:size],
                                                      space_conj[:size], time_reverse)
                if direction is None:
                    continue
                space[size] = direction
                space_conj[size] = np.conj(direction)
                images[size] = apply_h(direction)
                n_apply += 1
                size += 1
                added += 1
            if added == 0:
                table.end("no independent direction left to expand")
                raise SolverFailure(
                    "{} Kramers-restricted Davidson stalled at iteration {}: every "
                    "preconditioned direction for the {} unconverged pair(s) already lay in "
                    "the {}-direction subspace (max|r| = {:.2e} against conv_tol = {:.1e})"
                    .format(label, iteration, unconverged.size, size, float(np.max(norms)),
                            conv_tol))

    if converged:
        table.end("converged in {} iterations, {} applications of H".format(
            iteration, n_apply))
    else:
        table.end("NOT converged")
        raise SolverFailure(
            "{} Kramers-restricted Davidson did not converge {} pairs of {} determinants in "
            "{} iterations: max|r| = {:.2e} against conv_tol = {:.1e}".format(
                label, n_pairs, ndet, max_iter, float(np.max(norms)), conv_tol))

    states = np.empty((2 * n_pairs, ndet), dtype=np.complex128)
    states[0::2] = ritz
    states[1::2] = time_reverse(ritz)
    log.debug("%s Kramers Davidson: %d pairs, %d iterations, %d H applications, "
              "max|r| = %.2e", label, n_pairs, iteration, n_apply, float(np.max(norms)))
    return DavidsonResult(energies=np.repeat(energies, 2), vectors=states,
                          residuals=np.repeat(norms, 2), n_iter=iteration, n_apply=n_apply,
                          converged=True)


__all__ = ["davidson", "davidson_kramers", "davidson_kramers_sector",
           "davidson_sector", "DavidsonResult", "davidson_workspace_gb",
           "subspace_cap",
           "DEFAULT_CONV_TOL", "DEFAULT_MAX_ITER", "DENSE_SOLVE_MAX_DET",
           "SUBSPACE_FACTOR", "SUBSPACE_MARGIN", "N_GENERIC_GUESS", "GUESS_SEED",
           "KRAMERS_PAIR_TOL"]
