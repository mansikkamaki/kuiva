"""Tier-0 tests for the full-CI sigma vector.

The reference is :func:`kuiva.ci.strings.hamiltonian_matrix`, and that choice is the point of
this file. It is a **genuinely independent implementation**: it finds interacting determinant
pairs by an ``O(N^2)`` XOR/popcount search, signs them with its own phase routine and applies
the Slater-Condon rules directly, where the sigma vector resolves the same operator through
one-particle excitations, an intermediate index and a GEMM. Different algorithm, different
code path, different sign bookkeeping — so the comparison **can fail**, which is the
measure of a check's worth. Two errors that survive every structural test are caught only
here:

* a transposed ``F`` index or a dropped fermionic phase (the conjugation-trap analogue);
* ⚠ the ``h~_pq = h_pq - 1/2 sum_r (pr|rq)`` folding, which is invisible to hermiticity,
  degeneracy and every trace condition and merely shifts the energy.

``test_the_h_folding_is_what_makes_this_agree`` deliberately breaks the folding to show the
comparison is sensitive to it, rather than leaving that as an assumption.
"""
import numpy as np
import pytest

from kuiva.ci import kernels
from kuiva.ci.sigma import (SigmaOperator, eri_matrix_gb, sigma_vector,
                            sigma_workspace_gb)
from kuiva.ci.strings import (CASSpace, cas_dimension, hamiltonian_matrix,
                              single_excitation_operator)
from test_ci_strings import random_spinor_integrals

#: Machine precision is the right tolerance here: the two implementations compute the same
#: sums of the same products, so anything above rounding is a bug, not a method difference.
EXACT = 1e-11

SPACES = [(4, 1), (5, 2), (6, 3), (6, 5), (7, 2), (8, 4), (9, 4)]


def _reference(space, h, eri):
    return hamiltonian_matrix(space.determinants(), h, eri).toarray()


def _random_vector(ndet, seed):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(ndet) + 1j * rng.standard_normal(ndet)
    return c / np.linalg.norm(c)


# --- the load-bearing comparison ---------------------------------------------------------

@pytest.mark.parametrize("n,k", SPACES)
def test_sigma_matches_the_independent_sparse_hamiltonian(n, k):
    h, eri = random_spinor_integrals(n, seed=10 * n + k)
    space = CASSpace(n, k)
    c = _random_vector(space.ndet, seed=n + k)
    reference = _reference(space, h, eri) @ c
    got = sigma_vector(space, c, h, eri)
    scale = max(float(np.max(np.abs(reference))), 1.0)
    assert np.max(np.abs(got - reference)) < EXACT * scale


@pytest.mark.parametrize("n,k", [(5, 2), (6, 3), (7, 3)])
def test_the_implied_matrix_is_the_hamiltonian_element_by_element(n, k):
    """Not just ``H c`` for one vector: every column, so no vector can accidentally agree."""
    h, eri = random_spinor_integrals(n, seed=n * k)
    space = CASSpace(n, k)
    built = SigmaOperator(space, h, eri).matrix()
    reference = _reference(space, h, eri)
    assert np.max(np.abs(built - reference)) < EXACT * max(
        float(np.max(np.abs(reference))), 1.0)
    # ... and Hermitian, which the reference enforces by construction and this one does not.
    assert np.max(np.abs(built - built.conj().T)) < EXACT


def test_the_h_folding_is_what_makes_this_agree():
    """⚠ Break ``h~`` and the comparison must fail — otherwise it proves nothing.

    The folded ``-1/2 sum_r (pr|rq)`` is the term the structural checks cannot see. This
    asserts that the check above is actually sensitive to it, by removing it.
    """
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=7)
    space = CASSpace(n, k)
    c = _random_vector(space.ndet, seed=1)
    reference = _reference(space, h, eri) @ c

    operator = SigmaOperator(space, h, eri)
    assert np.max(np.abs(operator(c) - reference)) < EXACT * np.max(np.abs(reference))

    operator.h_eff = h.copy()                                     # the unfolded operator
    operator.h_eff_flat = np.ascontiguousarray(operator.h_eff.reshape(n * n))
    broken = operator(c)
    assert np.max(np.abs(broken - reference)) > 1e-3 * np.max(np.abs(reference))


# --- the intermediate itself -------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(5, 2), (6, 3), (6, 4)])
def test_the_F_intermediate_is_E_pq_applied_to_the_vector(n, k):
    """``F[K, p*n+q] = sum_J <K|E_pq|J> c_J``, against the independent sparse operators.

    ⚠ This pins the **index order** stated in the module docstring: ``p`` is annihilated from
    ``K`` and ``q`` created. A transposed ``F`` still gives a Hermitian Hamiltonian and a
    plausible energy, so it has to be checked here and not inferred from the total.
    """
    h, eri = random_spinor_integrals(n, seed=n + 2 * k)
    space = CASSpace(n, k)
    operator = SigmaOperator(space, h, eri)
    c = _random_vector(space.ndet, seed=k)
    # ⚠ Through gather_f, not by peeking after an application: F and G share the buffer, so
    # after __call__ it holds G — reading it as F is exactly the mistake the shared-buffer
    # contract exists to name.
    f = operator.gather_f(c)
    dets = space.determinants()
    for p in range(n):
        for q in range(n):
            expected = single_excitation_operator(dets, p, q) @ c
            assert np.allclose(f[:, p * n + q], expected, atol=1e-13)


# --- permutational symmetry (4-fold, never 8-fold) ---------------------------------

def test_four_fold_symmetry_is_asserted():
    n = 5
    h, eri = random_spinor_integrals(n, seed=2)
    space = CASSpace(n, 2)
    broken = eri.copy()
    broken[0, 1, 2, 3] += 0.5                                     # kills (pq|rs) = (rs|pq)
    with pytest.raises(ValueError, match=r"\(rs\|pq\)"):
        SigmaOperator(space, h, broken)
    broken = eri.copy()
    broken[0, 1, 2, 3] += 0.5
    broken[2, 3, 0, 1] += 0.5                                     # kills (pq|rs) = (qp|sr)*
    with pytest.raises(ValueError, match=r"\(qp\|sr\)"):
        SigmaOperator(space, h, broken)


def test_eight_fold_symmetry_is_never_assumed():
    """⚠ The test integrals must genuinely lack 8-fold symmetry, or this proves nothing."""
    n = 6
    h, eri = random_spinor_integrals(n, seed=11)
    eight_fold_error = np.max(np.abs(eri - eri.transpose(2, 1, 0, 3)))
    assert eight_fold_error > 1e-3 * np.max(np.abs(eri))
    space = CASSpace(n, 3)
    c = _random_vector(space.ndet, seed=4)
    reference = _reference(space, h, eri) @ c
    assert np.max(np.abs(sigma_vector(space, c, h, eri) - reference)) < EXACT * np.max(
        np.abs(reference))


def test_non_hermitian_one_electron_integrals_are_refused():
    n = 4
    h, eri = random_spinor_integrals(n, seed=5)
    h[0, 1] += 1.0
    with pytest.raises(ValueError, match="Hermitian"):
        SigmaOperator(CASSpace(n, 2), h, eri)


# --- blocking is a parameter, and it changes nothing (B7) --------------------------------

def test_the_block_size_does_not_change_the_answer():
    """⚠ B7: the kernel takes its block size, so a different one must be a no-op.

    Step 1 writes every element once, so it is required to be **bitwise** invariant. Step 3
    sums ``k(n-k+1)`` gathered terms per row and carries a B10 reduction-order note, so it
    gets the 1e-13 relative parity tolerance fixed in advance.
    """
    n, k = 7, 3
    h, eri = random_spinor_integrals(n, seed=6)
    space = CASSpace(n, k)
    c = _random_vector(space.ndet, seed=9)
    whole = SigmaOperator(space, h, eri, block=space.ndet)
    chopped = SigmaOperator(space, h, eri, block=1)
    a, b = whole(c), chopped(c)
    assert np.array_equal(whole.f_buf, chopped.f_buf)
    assert np.allclose(a, b, rtol=1e-13, atol=1e-15)


# --- the dispatch shim ------------------------------------------------------------

@pytest.mark.parametrize("name", ["sigma_gather_f", "sigma_gather_out",
                                  "cas_rank", "cas_unrank", "excitation_map"])
def test_every_registered_backend_gives_the_same_answer(name):
    """Parametrized over :func:`kernels.backends_for`, so a second backend is exercised the
    day it is registered — with no test retrofit. That is what the shim buys now."""
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=8)
    reference = None
    for backend in kernels.backends_for(name):
        space = CASSpace(n, k, backend=backend)
        c = _random_vector(space.ndet, seed=2)
        got = SigmaOperator(space, h, eri, backend=backend)(c)
        if reference is None:
            reference = got
        else:
            assert np.allclose(got, reference, rtol=1e-13, atol=1e-15)


def test_callers_never_learn_which_backend_they_got():
    """The kernel-interface boundary, expressed as a test: resolution is by name.

    The preferred backend is a *preference*, not a guarantee: a kernel the compiled
    backend does not implement resolves to one that exists (KUIVA_KERNELS=auto semantics),
    so what is asserted is that resolution lands on a registered backend — not that every
    kernel has the preferred one.
    """
    assert kernels.spec("sigma_gather_f").backend in kernels.backends_for("sigma_gather_f")
    assert kernels.resolve("sigma_gather_f") is kernels.spec("sigma_gather_f").impl
    with pytest.raises(KeyError):
        kernels.resolve("sigma_gather_f", "fortran77")


# --- sizing and the residency ceiling ---------------------------------------------

@pytest.mark.parametrize("n,k", [(6, 3), (8, 4), (10, 5)])
def test_sigma_workspace_sizing_is_exact_two_sided(n, k):
    h, eri = random_spinor_integrals(n, seed=3)
    operator = SigmaOperator(CASSpace(n, k), h, eri)
    actual = operator.f_buf.nbytes + operator.tile_buf.nbytes
    assert sigma_workspace_gb(n, k) == pytest.approx(actual / 1024.0 ** 3, rel=1e-12)
    assert eri_matrix_gb(n) == pytest.approx(operator.eri_mat.nbytes / 1024.0 ** 3, rel=1e-12)


def test_workspace_sizing_reproduces_the_residency_table():
    """The numbers the conventional-CI ceiling is set by — memory, not flops.

    ⚠ Halved on 2026-08-26: F and G share one buffer (the one-electron term is banked before
    the GEMM overwrites F in tiles), so the former ``2 C(n,k) n^2`` — 2.2 GB at 20 spinors,
    10.2 at 22 — became ``C(n,k) n^2`` plus a fixed tile. The validation record carries the
    measurement; this table is the sizing function's current word.
    """
    assert sigma_workspace_gb(18, 9) == pytest.approx(0.274, abs=0.002)
    assert sigma_workspace_gb(20, 10) == pytest.approx(1.150, abs=0.002)
    assert sigma_workspace_gb(22, 11) == pytest.approx(5.147, abs=0.02)
    assert sigma_workspace_gb(24, 12) == pytest.approx(23.28, abs=0.1)
    assert cas_dimension(20, 10) == 184756


def test_the_refusal_names_the_algorithmic_ceiling(monkeypatch):
    """⚠ Risk 9: an OOM here looks like a machine problem, so the refusal must say it is not."""
    from kuiva.util import resources as res

    h, eri = random_spinor_integrals(8, seed=1)
    space = CASSpace(8, 4)
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=1e-5, source="test"))
    with pytest.raises(res.MemoryLimitError) as excinfo:
        SigmaOperator(space, h, eri)
    message = str(excinfo.value).lower()
    assert "sigma workspace" in message
    assert "batched" in message and "not a larger run" in message
    assert "dmrg" in message


def test_the_workspace_reservation_dies_with_the_operator():
    """⚠ The perturbation's one-live shifted family depends on this: dropping a
    SigmaOperator must return its workspace to the ledger, or a reservation per shift
    accumulates into exactly the aggregate the policy exists to avoid. The reservation is
    owned by the buffer, so a consumer still holding ``f_buf`` (the RDM builder) keeps the
    reservation too — the truthful accounting either way."""
    import gc

    from kuiva.util import resources as res

    space = CASSpace(8, 4)
    space.build_excitation_map()               # the space's own reservations, out of the way
    h, eri = random_spinor_integrals(8, seed=5)
    res.ensure_configured()
    base = res.BUDGET.resident_gb()
    operator = SigmaOperator(space, h, eri)
    assert res.BUDGET.resident_gb() == pytest.approx(base + sigma_workspace_gb(8, 4))
    del operator
    gc.collect()
    assert res.BUDGET.resident_gb() == pytest.approx(base)
