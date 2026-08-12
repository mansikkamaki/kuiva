"""Tier-0 tests for the full-CI density matrices.

Four independent kinds of check, because the failure mode here is a 2-RDM that is Hermitian,
positive and correctly traced while being wrong — the conjugation trap in its RDM
guise:

* against :func:`kuiva.ci.strings.rdm12`, a **genuinely independent implementation** (pairwise
  connection search, its own Slater-Condon algebra, a scatter rather than a GEMM);
* against **exact identities** — ``Tr gamma = N``, ``sum_r Gamma_pqrr = (N-1) gamma_pq``,
  hermiticity, the pair-exchange symmetries;
* the **closure check**: contracting the Hamiltonian with the RDMs must reproduce the Davidson
  eigenvalue. This ties the two together and fails on a phase error in either;
* the **state-averaging policy**, asserted as behaviour: equal weights inside a
  degenerate block, and a refusal when the requested state count splits one.

``test_the_bra_transposition_is_what_makes_this_right`` removes the ``M[qp, rs]``
transposition on purpose, to show the identities above are actually sensitive to it.
"""
import numpy as np
import pytest

from kuiva.ci.davidson import DEFAULT_CONV_TOL, davidson
from kuiva.ci.sigma import SigmaOperator
from kuiva.ci.strings import CASSpace, diagonal_energies, hamiltonian_matrix, rdm12
from kuiva.rdm.rdm import (DEFAULT_DEGENERACY_TOL, RDMBuilder, active_space_energy,
                           cas_rdms, degenerate_blocks, intermediate_gb, rdm_workspace_gb,
                           state_average_weights)
from test_ci_davidson import kramers_spinor_integrals
from test_ci_strings import random_spinor_integrals

EXACT = 1e-11


def _eigenstates(space, h, eri, n_states):
    """Lowest states by dense diagonalization of the *independent* Hamiltonian."""
    values, vectors = np.linalg.eigh(
        hamiltonian_matrix(space.determinants(), h, eri).toarray())
    return values[:n_states], np.ascontiguousarray(vectors[:, :n_states].T)


# --- against an independent implementation ------------------------------------------------

@pytest.mark.parametrize("n,k,n_states", [(6, 3, 1), (7, 3, 2), (8, 4, 3), (6, 4, 4),
                                          (7, 5, 2)])
def test_matches_the_independent_scatter_based_rdms(n, k, n_states):
    h, eri = random_spinor_integrals(n, seed=n * k)
    space = CASSpace(n, k)
    energies, vectors = _eigenstates(space, h, eri, n_states)
    gamma, gamma2 = cas_rdms(space, vectors, energies=energies, enforce_kramers=False)
    reference_gamma, reference_gamma2 = rdm12(space.determinants(), vectors.T)
    assert np.max(np.abs(gamma - reference_gamma)) < EXACT
    assert np.max(np.abs(gamma2 - reference_gamma2)) < EXACT


@pytest.mark.parametrize("n,k", [(6, 3), (7, 4)])
def test_non_uniform_weights_are_carried_through_identically(n, k):
    h, eri = random_spinor_integrals(n, seed=n + k)
    space = CASSpace(n, k)
    energies, vectors = _eigenstates(space, h, eri, 3)
    weights = np.array([0.5, 0.3, 0.2])
    gamma, gamma2 = cas_rdms(space, vectors, weights, enforce_kramers=False)
    reference_gamma, reference_gamma2 = rdm12(space.determinants(), vectors.T, weights)
    assert np.max(np.abs(gamma - reference_gamma)) < EXACT
    assert np.max(np.abs(gamma2 - reference_gamma2)) < EXACT


# --- exact identities ---------------------------------------------------------------------

@pytest.mark.parametrize("n,k,n_states", [(6, 3, 1), (8, 4, 3), (7, 5, 2), (9, 4, 4)])
def test_the_exact_trace_and_symmetry_conditions_hold(n, k, n_states):
    h, eri = random_spinor_integrals(n, seed=2 * n + k)
    space = CASSpace(n, k)
    energies, vectors = _eigenstates(space, h, eri, n_states)
    gamma, gamma2 = cas_rdms(space, vectors, energies=energies, enforce_kramers=False)

    assert abs(np.trace(gamma).real - k) < EXACT and abs(np.trace(gamma).imag) < EXACT
    assert np.max(np.abs(gamma - gamma.conj().T)) < EXACT
    # sum_r Gamma_pqrr = (N-1) gamma_pq -- the sharpest cheap check on the phase bookkeeping.
    assert np.max(np.abs(np.einsum("pqrr->pq", gamma2) - (k - 1) * gamma)) < EXACT
    assert np.max(np.abs(gamma2 - gamma2.transpose(2, 3, 0, 1))) < EXACT      # pair exchange
    assert np.max(np.abs(gamma2 - gamma2.conj().transpose(1, 0, 3, 2))) < EXACT
    assert np.max(np.abs(gamma2 + gamma2.transpose(0, 3, 2, 1))) < EXACT      # antisymmetry
    # A 1-RDM is positive semidefinite with occupations in [0, 1] for fermions.
    occupations = np.linalg.eigvalsh(gamma)
    assert occupations.min() > -EXACT and occupations.max() < 1.0 + EXACT


def test_the_bra_transposition_is_what_makes_this_right():
    """⚠ Drop ``D[pq,rs] = M[qp,rs]`` and the identities must fail — else they prove nothing.

    The un-transposed matrix is still Hermitian and still has a sensible-looking trace, which
    is exactly why this needs asserting rather than assuming.
    """
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=13)
    space = CASSpace(n, k)
    energies, vectors = _eigenstates(space, h, eri, 1)
    builder = RDMBuilder(space)
    gamma, gamma2 = builder(vectors, energies=energies, enforce_kramers=False)
    assert np.max(np.abs(np.einsum("pqrr->pq", gamma2) - (k - 1) * gamma)) < EXACT

    naive = np.ascontiguousarray(builder.m_buf.reshape(n, n, n, n))     # no transposition
    for orbital in range(n):
        naive[:, orbital, orbital, :] -= gamma
    assert np.max(np.abs(np.einsum("pqrr->pq", naive) - (k - 1) * gamma)) > 1e-3
    assert abs(active_space_energy(h, eri, gamma, naive) - float(energies[0])) > 1e-3


# --- the closure check --------------------------------------------------------------------

@pytest.mark.parametrize("n,k,n_states", [(6, 3, 1), (8, 4, 2), (9, 4, 4)])
def test_the_rdms_reproduce_the_davidson_eigenvalue(n, k, n_states):
    """⚠ The check that ties the Hamiltonian and the RDMs together.

    ``E = sum h_pq gamma_pq + 1/2 sum (pq|rs) Gamma_pqrs`` must equal the state-averaged CI
    eigenvalue. A phase or index error in either the sigma vector or the density matrices
    breaks it, and nothing else in this file covers both at once.
    """
    h, eri = random_spinor_integrals(n, seed=n + 3 * k, scale=0.1)
    h = h + np.diag(np.arange(n) * 3.0)
    space = CASSpace(n, k)
    operator = SigmaOperator(space, h, eri)
    result = davidson(operator, diagonal_energies(space, h, eri), n_states, dense_max_det=0)

    builder = RDMBuilder(space, f_buf=operator.f_buf)
    gamma, gamma2 = builder(result.vectors, energies=result.energies,
                            enforce_kramers=False)
    assert abs(active_space_energy(h, eri, gamma, gamma2)
               - float(np.mean(result.energies))) < 1e-10


def test_sharing_the_sigma_operators_buffer_changes_nothing():
    """One intermediate, three consumers. Sharing it is the point, not an optimization."""
    n, k = 7, 3
    h, eri = random_spinor_integrals(n, seed=21)
    space = CASSpace(n, k)
    operator = SigmaOperator(space, h, eri)
    energies, vectors = _eigenstates(space, h, eri, 2)
    shared = RDMBuilder(space, f_buf=operator.f_buf)(vectors, energies=energies,
                                                     enforce_kramers=False)
    own = RDMBuilder(space)(vectors, energies=energies, enforce_kramers=False)
    assert np.array_equal(shared[0], own[0]) and np.array_equal(shared[1], own[1])


# --- state averaging and Kramers-block completeness --------------------------

def test_degenerate_blocks_are_found_by_gap():
    assert degenerate_blocks([0.0, 1e-9, 1.0, 1.0 + 1e-9, 2.0]) == [(0, 2), (2, 4), (4, 5)]
    assert degenerate_blocks([0.0, 1.0, 2.0]) == [(0, 1), (1, 2), (2, 3)]
    assert degenerate_blocks([]) == []
    with pytest.raises(ValueError, match="ascending"):
        degenerate_blocks([1.0, 0.0])


def test_weights_are_equalized_inside_a_degenerate_block(kuiva_caplog):
    """⚠ Inside a degenerate manifold the individual roots are defined only up to a rotation,
    so unequal weights make the converged orbitals depend on an eigensolver's arbitrary
    choice. It is imposed where the RDMs are built; this is that."""
    energies = [0.0, 1e-9, 1.0, 1.0 + 1e-10]
    weights = state_average_weights(energies, n_elec=3, weights=[0.7, 0.1, 0.15, 0.05])
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] == pytest.approx(weights[3])
    assert weights.sum() == pytest.approx(1.0)
    assert "equalized" in kuiva_caplog.text


def test_uniform_weights_over_complete_blocks_produce_no_warning(kuiva_caplog):
    state_average_weights([0.0, 1e-9, 1.0, 1.0 + 1e-10], n_elec=3)
    assert "equalized" not in kuiva_caplog.text


def test_an_odd_state_count_on_an_odd_electron_system_is_refused():
    """⚠ Rigorous, not heuristic: Kramers' theorem makes every level at least doubly
    degenerate, so an odd count *must* cut a pair."""
    with pytest.raises(ValueError, match="split a Kramers-degenerate block"):
        state_average_weights([0.0, 1e-9, 1.0], n_elec=3)
    with pytest.raises(ValueError, match="ask for 2 or 4 states"):
        state_average_weights([0.0, 1e-9, 1.0], n_elec=3)


def test_the_same_count_on_an_even_electron_system_is_allowed():
    """No Kramers theorem for an even electron count, so three states is a legitimate ask."""
    weights = state_average_weights([0.0, 1e-9, 1.0], n_elec=4)
    assert weights[0] == pytest.approx(weights[1])
    assert weights.sum() == pytest.approx(1.0)


def test_a_split_block_can_be_downgraded_to_a_warning(kuiva_caplog):
    weights = state_average_weights([0.0, 1e-9, 1.0], n_elec=3, on_split="warn")
    assert weights.sum() == pytest.approx(1.0)
    assert "split a Kramers-degenerate block" in kuiva_caplog.text
    with pytest.raises(ValueError, match="on_split must be"):
        state_average_weights([0.0, 1e-9, 1.0], n_elec=3, on_split="ignore")


def test_an_interior_odd_block_blames_the_tolerance_not_the_state_count():
    """A truncation can only happen at the end; anywhere else it is the tolerance."""
    with pytest.raises(ValueError, match="tolerance"):
        state_average_weights([0.0, 1e-9, 2e-9, 1.0, 1.0 + 1e-10], n_elec=3,
                              tol=1.5e-9)


def test_the_builder_enforces_the_policy_and_needs_energies_to_do_it():
    n, k = 8, 5
    h, eri = random_spinor_integrals(n, seed=17)
    space = CASSpace(n, k)
    energies, vectors = _eigenstates(space, h, eri, 3)
    builder = RDMBuilder(space)
    with pytest.raises(ValueError, match="need the state energies"):
        builder(vectors)
    builder(vectors, enforce_kramers=False)                # the explicit escape hatch


def test_a_real_kramers_degenerate_spectrum_passes_the_policy_unchanged():
    """The physical case: an odd-electron Hamiltonian with genuine spin-orbit coupling."""
    n_spatial, k = 4, 3
    h, eri = kramers_spinor_integrals(n_spatial, seed=5)
    space = CASSpace(2 * n_spatial, k)
    energies, vectors = _eigenstates(space, h, eri, 4)
    assert abs(energies[0] - energies[1]) < 1e-11 and abs(energies[2] - energies[3]) < 1e-11
    gamma, gamma2 = cas_rdms(space, vectors, energies=energies)
    assert abs(np.trace(gamma).real - k) < EXACT
    assert np.max(np.abs(np.einsum("pqrr->pq", gamma2) - (k - 1) * gamma)) < EXACT
    with pytest.raises(ValueError, match="split a Kramers-degenerate block"):
        cas_rdms(space, vectors[:3], energies=energies[:3])


# --- sizing ------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(6, 3), (8, 4)])
def test_rdm_sizing_is_exact_two_sided(n, k):
    space = CASSpace(n, k)
    builder = RDMBuilder(space)
    gamma2 = np.zeros((n, n, n, n), dtype=np.complex128)
    assert rdm_workspace_gb(n) == pytest.approx(2 * gamma2.nbytes / 1024.0 ** 3, rel=1e-12)
    assert intermediate_gb(space.ndet, n) == pytest.approx(
        builder.f_buf.nbytes / 1024.0 ** 3, rel=1e-12)


def test_the_block_size_does_not_change_the_answer():
    """B7: blocking is a parameter, so it must be invisible in the result (to 1e-13, since
    the accumulation carries a B10 reduction-order note)."""
    n, k = 7, 3
    h, eri = random_spinor_integrals(n, seed=23)
    space = CASSpace(n, k)
    energies, vectors = _eigenstates(space, h, eri, 2)
    whole = RDMBuilder(space, block=space.ndet)(vectors, energies=energies,
                                                enforce_kramers=False)
    chopped = RDMBuilder(space, block=1)(vectors, energies=energies, enforce_kramers=False)
    assert np.allclose(whole[0], chopped[0], rtol=1e-13, atol=1e-15)
    assert np.allclose(whole[1], chopped[1], rtol=1e-13, atol=1e-15)


def test_the_degeneracy_tolerance_is_where_the_general_complex_path_needs_it():
    """Kramers degeneracy emerges numerically at 1e-8 to 1e-6 Eh in this path."""
    assert 1e-8 <= DEFAULT_DEGENERACY_TOL <= 1e-5


# --- Tier 1: the whole stack against PySCF ------------------------------------------------

def test_the_full_stack_reproduces_pyscf_fci():
    """CASSpace -> sigma -> Davidson -> RDMs on a real molecule, against PySCF (Tier 1).

    A **spin-free** Hamiltonian in a Kramers-paired spinor basis must give exactly the spatial
    answer, so this is the "same problem in a doubled basis" check applied to the whole
    full-CI stack at once — addressing, excitation map, sigma vector, eigensolver and density
    matrices. It catches convention errors that an in-house reference would share with the
    code under test, which is what a check is for.

    ⚠ The active space is CAS(4,4), so **eight** spinors hold four electrons: with spin-free
    integrals every level is at least doubly degenerate (here fourfold, since spin is a good
    quantum number), which is why the ground state is taken as a one-state average and the
    Kramers policy is switched off explicitly.
    """
    pyscf_fci = pytest.importorskip("pyscf.fci")
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.60", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    ncas, nelecas = 4, 4
    ncore = (mol.nelectron - nelecas) // 2
    mo = mf.mo_coeff[:, ncore:ncore + ncas]
    mo_core = mf.mo_coeff[:, :ncore]
    dm_core = 2 * mo_core @ mo_core.T
    vj, vk = mf.get_jk(mol, dm_core)
    h_core = mf.get_hcore()
    h_eff = mo.T @ (h_core + vj - 0.5 * vk) @ mo
    e_core = np.einsum("ij,ji->", h_core + 0.5 * (vj - 0.5 * vk), dm_core) + mol.energy_nuc()
    eri_cas = ao2mo.restore(1, ao2mo.kernel(mol, mo), ncas)
    e_reference, fci_vector = pyscf_fci.direct_spin1.kernel(h_eff, eri_cas, ncas, nelecas)
    e_reference += e_core

    # Kramers expansion: spinors 2p and 2p+1 carry the same spatial orbital.
    n = 2 * ncas
    h = np.zeros((n, n), dtype=np.complex128)
    eri = np.zeros((n,) * 4, dtype=np.complex128)
    for spin in (0, 1):
        h[spin::2, spin::2] = h_eff
    for bra in (0, 1):
        for ket in (0, 1):
            eri[bra::2, bra::2, ket::2, ket::2] = eri_cas

    space = CASSpace(n, nelecas)
    operator = SigmaOperator(space, h, eri)
    result = davidson(operator, diagonal_energies(space, h, eri), 1, dense_max_det=0)
    assert abs(result.energies[0] + e_core - e_reference) < 1e-9

    builder = RDMBuilder(space, f_buf=operator.f_buf)
    gamma, gamma2 = builder(result.vectors, enforce_kramers=False)
    assert abs(active_space_energy(h, eri, gamma, gamma2) + e_core - e_reference) < 1e-9

    # The spinor 1-RDM must trace back to PySCF's spin-summed spatial one.
    #
    # ⚠ The tolerance is the **solver's promise**, not a number read off one run, and this
    # test previously had the latter. A Rayleigh quotient is variational, so the energy error
    # is second order in the eigenvector error while a density matrix is first order — which
    # is the whole reason `DEFAULT_CONV_TOL` is set from the RDM (see `ci/davidson.py`). So
    # the bound here is the residual bound, and the density tracks it one-for-one; measured
    # on this system, with the energy exact to ~4e-15 at every rung:
    #
    #     conv_tol     |r|        max|gamma - PySCF|
    #     1e-8         4.1e-9     4.8e-9
    #     1e-10        8.4e-11    5.0e-11
    #     1e-12        8.2e-13    1.7e-13
    #
    # The old 1e-9 was where one particular starting subspace happened to land, four times
    # tighter than anything the solver undertakes to deliver, and a change of guess moved it
    # (a band set at the observed spread is a band that fails on the next system).
    spatial = gamma[0::2, 0::2] + gamma[1::2, 1::2]
    pyscf_gamma = pyscf_fci.direct_spin1.make_rdm1(fci_vector, ncas, nelecas)
    assert np.max(np.abs(spatial - pyscf_gamma.T)) < 2 * DEFAULT_CONV_TOL
    # ⚠ Same bound, and for the same reason: the exact spin-summed spatial 1-RDM of a real
    # spin-free problem *is* real, so its imaginary part is not a separate quantity — it is
    # another view of the same first-order eigenvector error, and it scales with |r| exactly
    # as the table above does.
    assert np.max(np.abs(spatial.imag)) < 2 * DEFAULT_CONV_TOL
