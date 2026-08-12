"""Tier-0 tests for the block Davidson eigensolver.

⚠ **What each comparison here can and cannot fail on** matters more than usual, because the
obvious check — Davidson against a dense diagonalization of the *same* sigma vector — shares
an implementation with the thing it tests and therefore cannot see an error in it. So
the file is layered deliberately:

* against a dense ``eigh`` of :func:`kuiva.ci.strings.hamiltonian_matrix`, which is a
  genuinely independent Hamiltonian (``O(N^2)`` search, own phase routine). This one can fail
  on the Hamiltonian *and* on the solver;
* against a dense ``eigh`` of the sigma vector's own matrix, which tests **the solver alone**
  and is honest about that: Stage 1 already established the Hamiltonian independently;
* against **theorems** — Kramers doubling of every level of an odd-electron system, and
  invariance to a global phase — which need no reference calculation at all.

The Kramers model built here is the sharpest of the three. It is a time-reversal-symmetric
complex spinor Hamiltonian with genuine spin-orbit coupling in the one-electron part, so
every eigenvalue is **exactly** doubled while the spin degeneracy above that is broken. That
is precisely the near-degenerate cluster an ARPACK solve was recorded failing on, and the
reason this solver is a block solver at all.
"""
import numpy as np
import pytest

from kuiva.ci.davidson import (DENSE_SOLVE_MAX_DET, SUBSPACE_MARGIN, davidson,
                               davidson_workspace_gb, subspace_cap)
from kuiva.ci.sigma import SigmaOperator
from kuiva.ci.strings import CASSpace, diagonal_energies, hamiltonian_matrix
from kuiva.util.errors import SolverFailure
from test_ci_strings import random_spinor_integrals

#: The solver converges residuals to 1e-6, which bounds the Ritz-value error by |r|^2/gap.
ENERGY_TOL = 1e-9


def kramers_spinor_integrals(n_spatial, seed=0, soc=0.3):
    """A **time-reversal symmetric** spinor Hamiltonian with real spin-orbit coupling.

    In the interleaved Kramers convention (spinor ``2p + sigma``):

    * ``h = h0 (x) 1_2 + i sum_k W_k (x) sigma_k`` with ``h0`` real symmetric and ``W_k`` real
      antisymmetric — Hermitian, and time-reversal **even** because
      ``sigma_y sigma_k^* sigma_y = -sigma_k`` cancels the conjugation of the explicit ``i``.
      This is the ``A (x) 1_2 + sigma.W`` decomposition the X2C operators are stored in;
    * a **spin-free** two-electron part, ``(2p+sigma, 2q+sigma | 2r+mu, 2s+mu)``, which is
      time-reversal even trivially.

    So the whole Hamiltonian commutes with ``T``, and with an odd electron count every level
    is exactly doubled by Kramers' theorem — while the spin-orbit term breaks the higher spin
    degeneracy, leaving clusters of exactly two. Both halves of that are asserted below before
    the model is used to test anything, so a defect in this generator cannot masquerade as a
    solver result.
    """
    rng = np.random.default_rng(seed)
    h0 = rng.standard_normal((n_spatial, n_spatial))
    h0 = 0.5 * (h0 + h0.T)
    factors = rng.standard_normal((2 * n_spatial, n_spatial, n_spatial))
    factors = 0.5 * (factors + factors.transpose(0, 2, 1))
    eri0 = np.tensordot(factors, factors, axes=([0], [0]))

    pauli = (np.array([[0, 1], [1, 0]], dtype=complex),
             np.array([[0, -1j], [1j, 0]]),
             np.array([[1, 0], [0, -1]], dtype=complex))
    h = np.kron(h0, np.eye(2)).astype(np.complex128)
    for sigma in pauli:
        w = rng.standard_normal((n_spatial, n_spatial))
        h = h + 1j * soc * np.kron(0.5 * (w - w.T), sigma)

    n = 2 * n_spatial
    eri = np.zeros((n, n, n, n), dtype=np.complex128)
    for bra in (0, 1):
        for ket in (0, 1):
            eri[bra::2, bra::2, ket::2, ket::2] = eri0
    return h, eri


def _setup(n, k, seed=0, kramers=False, spread=0.0, scale=1.0):
    if kramers:
        h, eri = kramers_spinor_integrals(n // 2, seed=seed)
    else:
        h, eri = random_spinor_integrals(n, seed=seed, scale=scale)
    if spread:
        h = h + np.diag(np.arange(n) * spread)        # a Fock-like spread of orbital energies
    space = CASSpace(n, k)
    return space, h, eri, SigmaOperator(space, h, eri), diagonal_energies(space, h, eri)


# --- against a genuinely independent Hamiltonian ------------------------------------------

@pytest.mark.parametrize("n,k,n_roots", [(8, 4, 4), (9, 4, 6), (10, 5, 2), (12, 6, 6)])
def test_matches_dense_eigh_of_the_independent_hamiltonian(n, k, n_roots):
    space, h, eri, operator, diagonal = _setup(n, k, seed=n + k, spread=3.0, scale=0.1)
    result = davidson(operator, diagonal, n_roots, dense_max_det=0)
    reference = np.linalg.eigvalsh(
        hamiltonian_matrix(space.determinants(), h, eri).toarray())[:n_roots]
    assert result.converged
    assert np.max(np.abs(result.energies - reference)) < ENERGY_TOL


@pytest.mark.parametrize("n,k,n_roots", [(8, 4, 3), (10, 4, 5)])
def test_the_returned_vectors_are_orthonormal_eigenvectors(n, k, n_roots):
    space, h, eri, operator, diagonal = _setup(n, k, seed=n * k, spread=3.0, scale=0.1)
    result = davidson(operator, diagonal, n_roots, dense_max_det=0)
    overlap = result.vectors.conj() @ result.vectors.T
    assert np.max(np.abs(overlap - np.eye(n_roots))) < 1e-10
    for root in range(n_roots):
        residual = operator(result.vectors[root]) - result.energies[root] * result.vectors[root]
        assert np.linalg.norm(residual) < 1e-6


# --- theorems: no reference calculation involved ------------------------------------------

@pytest.mark.parametrize("n_spatial,k", [(3, 3), (4, 3), (4, 5)])
def test_every_level_of_an_odd_electron_system_is_kramers_doubled(n_spatial, k):
    """⚠ The model is validated first, then the solver — in that order and separately.

    If the doubling assertion on the *dense* spectrum fails, the integral generator is wrong
    and nothing about Davidson has been learnt. Only once it holds does reproducing those
    pairs iteratively mean the solver resolves an exactly degenerate cluster.
    """
    assert k % 2 == 1, "Kramers' theorem is about an odd electron count"
    n = 2 * n_spatial
    space, h, eri, operator, diagonal = _setup(n, k, seed=n_spatial + k, kramers=True)
    dense = np.linalg.eigvalsh(hamiltonian_matrix(space.determinants(), h, eri).toarray())
    assert np.max(np.abs(dense[0::2] - dense[1::2])) < 1e-11        # the model is TR-even
    gaps = np.abs(np.diff(dense)[1::2])
    assert gaps.min() > 1e-3, "spin-orbit coupling must break the degeneracy above the pair"

    n_roots = min(6, space.ndet - 2)
    result = davidson(operator, diagonal, n_roots, dense_max_det=0)
    assert result.converged
    assert np.max(np.abs(result.energies - dense[:n_roots])) < ENERGY_TOL
    # ... and the pairs come out as pairs, not as one member converged and one not.
    for pair in range(n_roots // 2):
        assert abs(result.energies[2 * pair] - result.energies[2 * pair + 1]) < 1e-9


def test_the_energies_are_invariant_to_a_global_phase_of_the_guess():
    """Tier 0: a CI vector is defined up to a phase, so the spectrum cannot depend on one."""
    space, h, eri, operator, diagonal = _setup(10, 5, seed=4, spread=3.0, scale=0.1)
    plain = davidson(operator, diagonal, 4, dense_max_det=0)
    phased = davidson(operator, diagonal, 4,
                      guess=np.exp(1.7j) * plain.vectors, dense_max_det=0)
    assert np.max(np.abs(plain.energies - phased.energies)) < ENERGY_TOL


def test_a_deliberately_degenerate_model_is_resolved():
    """A Hamiltonian with an exactly four-fold lowest level, and no spin structure at all.

    A solver that expanded roots one at a time, or carried a subspace of ``n_roots + 1``,
    fails here — which is why the Davidson subspace floor is a floor.
    """
    ndet = 400
    rng = np.random.default_rng(2)
    diagonal = np.concatenate([np.zeros(4), 1.0 + np.sort(rng.random(ndet - 4))])
    coupling = rng.standard_normal((ndet, ndet)) + 1j * rng.standard_normal((ndet, ndet))
    coupling = 1e-3 * (coupling + coupling.conj().T)
    np.fill_diagonal(coupling, 0.0)
    matrix = np.diag(diagonal).astype(np.complex128) + coupling
    reference = np.linalg.eigvalsh(matrix)

    result = davidson(lambda c: matrix @ c, np.real(np.diag(matrix)), 4, dense_max_det=0)
    assert result.converged
    assert np.max(np.abs(result.energies - reference[:4])) < ENERGY_TOL
    assert np.ptp(result.energies) < 1e-2                 # they really are one cluster


# --- warm start ----------------------------------------------------------------------

def test_a_warm_start_costs_almost_nothing():
    """Checkpoints store "what is needed to restart Davidson from a good guess"; this is why."""
    space, h, eri, operator, diagonal = _setup(12, 6, seed=1, spread=3.0, scale=0.1)
    cold = davidson(operator, diagonal, 4, dense_max_det=0)
    warm = davidson(operator, diagonal, 4, guess=cold.vectors, dense_max_det=0)
    assert warm.converged
    assert np.max(np.abs(warm.energies - cold.energies)) < ENERGY_TOL
    assert warm.n_apply < cold.n_apply / 4


def test_a_short_or_badly_sized_guess_is_accepted_and_padded():
    space, h, eri, operator, diagonal = _setup(10, 5, seed=5, spread=3.0, scale=0.1)
    full = davidson(operator, diagonal, 4, dense_max_det=0)
    partial = davidson(operator, diagonal, 4, guess=full.vectors[:1], dense_max_det=0)
    assert np.max(np.abs(partial.energies - full.energies)) < ENERGY_TOL


def test_a_guess_of_the_wrong_length_is_refused():
    space, h, eri, operator, diagonal = _setup(8, 4, seed=6)
    with pytest.raises(ValueError, match="guess vectors have length"):
        davidson(operator, diagonal, 2, guess=np.ones((1, 3), dtype=complex),
                 dense_max_det=0)


# --- failure semantics -----------------------------------------------------------

def test_non_convergence_raises_rather_than_returning_a_plausible_vector():
    """⚠ The whole point: an unconverged vector poisons every macro-iteration after it."""
    space, h, eri, operator, diagonal = _setup(10, 5, seed=7)
    with pytest.raises(SolverFailure, match="did not converge"):
        davidson(operator, diagonal, 4, max_iter=2, dense_max_det=0, conv_tol=1e-12)


def test_the_failure_message_carries_the_numbers_needed_to_act_on_it():
    space, h, eri, operator, diagonal = _setup(10, 5, seed=8)
    with pytest.raises(SolverFailure) as excinfo:
        davidson(operator, diagonal, 3, max_iter=1, dense_max_det=0, conv_tol=1e-12,
                 label="test")
    message = str(excinfo.value)
    assert "test" in message and "conv_tol" in message and "max|r|" in message


def test_an_impossible_root_count_is_refused_up_front():
    space, h, eri, operator, diagonal = _setup(6, 3, seed=9)
    with pytest.raises(ValueError, match="cannot ask for"):
        davidson(operator, diagonal, space.ndet + 1)
    with pytest.raises(ValueError, match="cannot ask for"):
        davidson(operator, diagonal, 0)


# --- the dense fallback -------------------------------------------------------------------

def test_the_dense_path_and_the_iterative_path_agree():
    space, h, eri, operator, diagonal = _setup(9, 4, seed=3, spread=3.0, scale=0.1)
    assert space.ndet <= DENSE_SOLVE_MAX_DET
    dense = davidson(operator, diagonal, 5)                      # takes the dense branch
    iterative = davidson(operator, diagonal, 5, dense_max_det=0)
    assert dense.dense and not iterative.dense
    assert dense.n_apply == space.ndet
    assert np.max(np.abs(dense.energies - iterative.energies)) < ENERGY_TOL


def test_asking_for_nearly_every_root_takes_the_dense_path():
    """An iterative solver has nothing to iterate on when the subspace is the whole space."""
    space, h, eri, operator, diagonal = _setup(8, 4, seed=2)
    assert davidson(operator, diagonal, space.ndet - 1, dense_max_det=0).dense


# --- the subspace floor and its cost -----------------------------------------------

# --- "Converged" is not "lowest": the biased-guess failure (N_GENERIC_GUESS) ---------------

def _open_shell_spin_free_integrals(n_spatial, seed, spread=0.05, scale=1.0):
    """A **spin-free, near-degenerate open shell** in the interleaved spinor basis.

    Two properties, and both are needed to exercise the failure:

    * **Spin-free**, so ``Sz`` is conserved and the determinant space splits into sectors the
      Krylov expansion cannot cross — the structure every ``with_soc=False`` calculation has;
    * **near-degenerate orbitals with real exchange**, so the determinant diagonals order by
      ``|Sz|`` (Hund: parallel spins gain exchange). That is what makes the lowest-diagonal
      determinants a *spin-polarized* set rather than a representative one, and it is why the
      failure showed up on an f shell rather than on a generic random Hamiltonian.

    ``L`` is symmetric in its orbital pair, so the spatial ERI has full 8-fold symmetry and is
    positive semidefinite; its exchange ``(pq|qp) = sum_P L_pq^2`` is then positive by
    construction, which is what fixes the sign of the Hund preference.
    """
    rng = np.random.default_rng(seed)
    h_s = np.diag(rng.standard_normal(n_spatial) * spread)      # a near-degenerate shell
    l = rng.standard_normal((2 * n_spatial, n_spatial, n_spatial)) * scale
    l = 0.5 * (l + np.transpose(l, (0, 2, 1)))
    eri_s = np.einsum("Ppq,Prs->pqrs", l, l)

    n = 2 * n_spatial
    h = np.zeros((n, n), dtype=np.complex128)
    eri = np.zeros((n, n, n, n), dtype=np.complex128)
    for s in (0, 1):
        h[s::2, s::2] = h_s
        for u in (0, 1):
            eri[s::2, s::2, u::2, u::2] = eri_s
    return h, eri


def _two_sz(space):
    """``2 Sz`` of every determinant: unbarred spinors are alpha, barred beta."""
    masks = np.asarray(space.determinants().masks)
    bits = np.arange(space.n_spinor, dtype=np.uint64)
    occ = ((masks[:, None] >> bits[None, :]) & np.uint64(1)).astype(np.int8)
    return occ[:, 0::2].sum(1) - occ[:, 1::2].sum(1)


def test_a_biased_guess_cannot_return_the_wrong_sectors_lowest_states():
    """⚠ **The mechanism, not the observable: "converged" must mean "lowest".**

    A Krylov method cannot leave the invariant subspaces its starting vectors lie in, and the
    natural CI guess — unit vectors on the lowest-diagonal determinants — is biased. On a
    spin-free Hamiltonian it can contain no member of some ``Sz`` sector at all; the solver
    then converges every residual it is judged on and returns eigenpairs that are not the
    lowest, with **nothing anywhere to notice**. Measured on Dy(3+)'s CAS(9, 14) with SOC off:
    a 66-root solve came back 6766 cm^-1 too high, in 17 iterations against the correct
    answer's 36. :data:`~kuiva.ci.davidson.N_GENERIC_GUESS` is the fix.

    The reference is a dense diagonalization of the **independently built**
    :func:`~kuiva.ci.strings.hamiltonian_matrix`, so this can fail on the guess and on the
    solver, and the control below is what makes it a measurement rather than a hope: it asserts
    that the biased guess on its own really does get this system wrong.
    """
    n_spatial, n_elec, n_roots = 6, 5, 8
    h, eri = _open_shell_spin_free_integrals(n_spatial, seed=1)
    space = CASSpace(2 * n_spatial, n_elec)
    operator = SigmaOperator(space, h, eri)
    diagonal = diagonal_energies(space, h, eri)
    exact = np.linalg.eigvalsh(np.asarray(hamiltonian_matrix(space, h, eri).todense()))

    # The premise, asserted rather than assumed: the guess the solver would build for itself
    # really does miss a sector here, which is what makes the control below meaningful.
    sectors = _two_sz(space)
    seen = set(sectors[np.argsort(diagonal)[:2 * n_roots]].tolist())
    assert seen < set(sectors.tolist()), \
        "the lowest-diagonal determinants span every Sz sector; nothing to test"

    got = davidson(operator, diagonal, n_roots, dense_max_det=0)
    assert np.allclose(got.energies, exact[:n_roots], atol=1e-8), \
        "the solver did not return the LOWEST {} roots: {} against {}".format(
            n_roots, got.energies, exact[:n_roots])

    # ⚠ The control. Reproduce the old behaviour exactly by *supplying* the biased guess the
    # solver used to build for itself: a supplied guess suppresses the generic vectors, so this
    # is the old code path and not an imitation of it. Then assert it gets a different, higher
    # answer. A guard that cannot fail proves nothing, and without this the assertion
    # above would pass on any correct solver whether or not the guess was ever the problem.
    biased_order = np.argsort(diagonal)[:2 * n_roots]
    biased = np.zeros((biased_order.size, space.ndet), dtype=np.complex128)
    biased[np.arange(biased_order.size), biased_order] = 1.0
    blind = davidson(operator, diagonal, n_roots, guess=biased, dense_max_det=0)
    assert not np.allclose(blind.energies, exact[:n_roots], atol=1e-8), \
        "this system no longer exercises the failure; pick another seed"
    assert blind.energies[-1] > exact[n_roots - 1] + 1e-6, \
        "a missed sector can only push the top root UP (Rayleigh-Ritz)"


def test_the_generic_guess_vectors_are_deterministic():
    """⚠ Event gating rests on ``solve`` at the same integrals being deterministic — an optimizer
    that reads solver noise as a surface change is the failure that section exists to prevent.
    So the generic vectors come from a fixed seed, and two solves must agree **bitwise**."""
    h, eri = _open_shell_spin_free_integrals(4, seed=6)
    space = CASSpace(8, 4)
    operator = SigmaOperator(space, h, eri)
    diagonal = diagonal_energies(space, h, eri)
    first = davidson(operator, diagonal, 5, dense_max_det=0)
    second = davidson(operator, diagonal, 5, dense_max_det=0)
    assert np.array_equal(first.energies, second.energies)
    assert first.n_iter == second.n_iter and first.n_apply == second.n_apply


def test_the_subspace_floor_cannot_be_argued_down():
    """⚠ The recorded Lanczos failure is the specification; a caller may not undercut it."""
    assert subspace_cap(4, 10_000, max_subspace=5) == 2 * 4 + SUBSPACE_MARGIN
    assert subspace_cap(4, 10_000) >= 2 * 4 + SUBSPACE_MARGIN
    assert subspace_cap(4, 10) == 10                       # ... but never exceeds the space


def test_workspace_sizing_is_exact_two_sided():
    ndet, cap = 1000, 20
    stack = np.zeros((cap, ndet), dtype=np.complex128)
    assert davidson_workspace_gb(ndet, cap) == pytest.approx(
        3 * stack.nbytes / 1024.0 ** 3, rel=1e-12)


def test_the_subspace_reservation_is_refused_when_it_cannot_fit(monkeypatch):
    from kuiva.util import resources as res

    space, h, eri, operator, diagonal = _setup(10, 5, seed=1)
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=1e-7, source="test"))
    with pytest.raises(res.MemoryLimitError, match="Davidson subspace"):
        davidson(operator, diagonal, 2, dense_max_det=0)
