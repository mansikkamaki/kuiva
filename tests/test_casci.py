"""Tier-0/Tier-1 tests for the CASCI/CASSCF drivers and the active-space selection.

What each group can actually fail on, which is the test of a check's worth:

* the **solver** against :func:`kuiva.ci.strings.hamiltonian_matrix` plus a dense ``eigh`` — a
  different algorithm with different sign bookkeeping, so a phase or index error in the sigma
  vector's route to the RDMs shows up here and not only in ``test_ci_sigma``;
* the **transition densities** against :func:`kuiva.ci.strings.single_excitation_operator`,
  built element by element from the operator algebra. Their diagonal must reproduce the state
  1-RDMs, which is the cheapest statement that ``rdm/rdm.py`` and this module agree;
* the **CASSCF composition** against the invariances CASSCF actually has: the energy is
  invariant to a rotation *inside* the active space and to one inside the inactive space, and
  it must be ``<=`` the CASCI energy at the same starting orbitals;
* the **active-space selection** against its two traps — the electron count and the
  odd inactive count that would split a Kramers pair — asserted as refusals, because
  both of them otherwise produce a plausible number;
* a **Tier-1** check of the whole stack against PySCF's CASSCF with spin-orbit coupling
  switched off, where the two-component problem reduces to two copies of a scalar one.
"""
import numpy as np
import pytest

from kuiva.ci.strings import (CASSpace, hamiltonian_matrix, rdm12,
                              single_excitation_operator)
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.mcscf.casci import (BOUNDARY_WARN_CM, ActiveSpace, BoundaryReport, CASCIResult,
                               FullCISolver, active_space, active_space_by_character, casci,
                               casscf, state_average_boundary)
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces, cas_energy
from kuiva.util.errors import SolverFailure
from test_ci_strings import random_spinor_integrals

EXACT = 1e-11


@pytest.fixture(scope="module")
def system():
    """The synthetic two-component system of ``test_mcscf``, with a core-guess orbital set."""
    rng = np.random.default_rng(3)
    nao = 6
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=4, n_orb=n)
    return factors, h_ao, np.ascontiguousarray(c0), spaces, 2


def reference_states(space_size, n_elec, h, eri, n_states):
    """Dense diagonalization of the independently built Hamiltonian.

    ⚠ The determinant list comes from :meth:`CASSpace.determinants`, **not** from
    ``Determinants.from_occupations(itertools.combinations(...))``. The latter keeps whatever
    order it is handed — lexicographic in the occupied orbitals — while a
    :class:`~kuiva.ci.strings.CASSpace` is in colex rank order, and the two differ from six
    spinors upward. Energies and RDMs are insensitive to it and *transition densities are
    not*: mixing the orderings compares a CI vector against the wrong operator and produces a
    Hermitian, plausible, wrong matrix. (Found exactly that way while writing this file.)
    """
    dets = CASSpace(space_size, n_elec).determinants()
    values, vectors = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    return dets, values[:n_states], np.ascontiguousarray(vectors[:, :n_states])


# --- the solver ----------------------------------------------------------------------------

@pytest.mark.parametrize("n,k,n_states", [(6, 3, 2), (8, 4, 4), (7, 5, 2), (6, 2, 6)])
def test_the_solver_reproduces_a_dense_diagonalization(n, k, n_states):
    h, eri = random_spinor_integrals(n, seed=n * 13 + k)
    solver = FullCISolver(n, k, n_states=n_states, enforce_kramers=False)
    result = solver.solve_active(h, eri, e_core=-2.5)
    _, values, vectors = reference_states(n, k, h, eri, n_states)

    assert np.max(np.abs(result.energies - values)) < EXACT
    assert np.allclose(result.total_energies, values - 2.5, atol=EXACT)
    assert abs(result.energy - (float(np.dot(result.weights, values)) - 2.5)) < EXACT
    # The RDMs are the state-averaged ones with the weights actually used, not the requested
    # ones: inside a degenerate block they are equalized and the energy must agree.
    gamma, gamma2 = rdm12(CASSpace(n, k).determinants(), vectors, result.weights)
    assert np.max(np.abs(result.gamma - gamma)) < EXACT
    assert np.max(np.abs(result.gamma2 - gamma2)) < EXACT


def test_the_solver_is_a_static_adaptive_solver():
    """A complete CAS space has one surface, so the event controller degrades cleanly."""
    from kuiva.mcscf.adaptive import AdaptiveCISolver, as_adaptive_solver

    solver = FullCISolver(6, 3, n_states=2, enforce_kramers=False)
    assert isinstance(solver, AdaptiveCISolver)
    assert as_adaptive_solver(solver) is solver          # not wrapped in a StaticSolver
    key = solver.space_key()
    h, eri = random_spinor_integrals(6, seed=1)
    solver.solve_active(h, eri)
    assert solver.propose(None) is None
    assert solver.space_key() == key                      # the chart never moves
    with pytest.raises(ValueError, match="nothing to adopt"):
        solver.adopt(key)


def test_reusing_the_operator_gives_the_same_answer_as_a_fresh_one():
    """``set_integrals`` is what keeps a macro-iteration from reallocating F and G. It must
    be indistinguishable from building a new operator, or the saving is bought with a bug."""
    n, k = 7, 3
    solver = FullCISolver(n, k, n_states=2, warm_start=False, enforce_kramers=False)
    first = solver.solve_active(*random_spinor_integrals(n, seed=4))
    h2, eri2 = random_spinor_integrals(n, seed=5)
    second = solver.solve_active(h2, eri2)
    fresh = FullCISolver(n, k, n_states=2, warm_start=False,
                         enforce_kramers=False).solve_active(h2, eri2)
    assert np.max(np.abs(second.energies - fresh.energies)) < EXACT
    assert np.max(np.abs(second.gamma - fresh.gamma)) < EXACT
    assert not np.allclose(first.energies, second.energies)   # the control


def test_the_warm_start_does_not_change_the_answer():
    """A guess is a guess: it may not move the converged eigenvalues beyond the tolerance."""
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=9)
    cold = FullCISolver(n, k, n_states=3, warm_start=False,
                        enforce_kramers=False).solve_active(h, eri)
    warm = FullCISolver(n, k, n_states=3, warm_start=True, enforce_kramers=False)
    warm.solve_active(h, eri)
    again = warm.solve_active(h, eri)
    assert np.max(np.abs(again.energies - cold.energies)) < 1e-10


def test_an_impossible_state_count_is_refused():
    with pytest.raises(ValueError, match="states of a"):
        FullCISolver(4, 2, n_states=100)


def test_integrals_from_the_wrong_active_space_are_refused(system):
    factors, h_ao, c0, spaces, n_elec = system
    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    with pytest.raises(ValueError, match="active spinors"):
        FullCISolver(spaces.n_active + 2, n_elec).casci(ints)


# --- transition densities ------------------------------------------------------------------

@pytest.mark.parametrize("n,k,n_states", [(6, 3, 3), (7, 4, 2)])
def test_transition_densities_match_the_operator_algebra(n, k, n_states):
    """Against ``single_excitation_operator``: a sparse matrix built from Slater-Condon rules
    rather than from the excitation map, so it can genuinely disagree."""
    h, eri = random_spinor_integrals(n, seed=n + 3 * k)
    solver = FullCISolver(n, k, n_states=n_states, enforce_kramers=False)
    solver.solve_active(h, eri)
    dets, _, vectors = reference_states(n, k, h, eri, n_states)
    gamma_ij = solver.transition_densities(solver.last.vectors)

    assert gamma_ij.shape == (n_states, n_states, n, n)
    ours = solver.last.vectors
    for p, q in ((0, 0), (1, 4), (n - 1, 2)):
        operator = single_excitation_operator(dets, p, q).toarray()
        expected = np.conj(ours) @ operator @ ours.T
        assert np.max(np.abs(gamma_ij[:, :, p, q] - expected)) < 1e-10
    # Hermiticity of the operator: gamma^{IJ}_pq = conj(gamma^{JI}_qp).
    assert np.max(np.abs(gamma_ij - gamma_ij.transpose(1, 0, 3, 2).conj())) < 1e-12
    del vectors


def test_the_transition_density_diagonal_is_the_state_1_rdm():
    """⚠ The cheapest statement that this module and ``rdm/rdm.py`` agree — they share the
    map and the intermediate, so a transposition would move both together everywhere except
    here, where an independent scatter-based implementation is the reference."""
    n, k, n_states = 6, 3, 3
    h, eri = random_spinor_integrals(n, seed=21)
    solver = FullCISolver(n, k, n_states=n_states, enforce_kramers=False)
    solver.solve_active(h, eri)
    gamma_ij = solver.transition_densities()
    dets = solver.space.determinants()
    for state in range(n_states):
        reference, _ = rdm12(dets, solver.last.vectors[state], np.array([1.0]))
        assert np.max(np.abs(gamma_ij[state, state] - reference)) < 1e-10


def test_transition_densities_before_a_solve_are_refused():
    with pytest.raises(RuntimeError, match="solve first"):
        FullCISolver(6, 3).transition_densities(np.zeros((1, 20), dtype=np.complex128))


# --- the CASCI and CASSCF drivers ------------------------------------------------------------

def test_casci_reproduces_the_hand_built_energy(system):
    factors, h_ao, c0, spaces, n_elec = system
    result = casci(factors, h_ao, c0, spaces, n_elec, n_states=1, report=False)
    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    _, values, _ = reference_states(spaces.n_active, n_elec, ints.h_active_effective(),
                                    ints.active_eri(), 1)
    assert abs(result.energy - (values[0] + ints.e_core)) < EXACT
    # And the RDM closure: contracting the integrals with the RDMs gives the same number.
    assert abs(cas_energy(ints, result.gamma, result.gamma2) - result.energy) < 1e-10


def test_casscf_lowers_the_energy_and_is_stationary(system):
    factors, h_ao, c0, spaces, n_elec = system
    reference = casci(factors, h_ao, c0, spaces, n_elec, report=False)
    outcome = casscf(factors, h_ao, c0, spaces, n_elec, mode="second-order", max_iter=40,
                     conv_grad=1e-6, report=False)
    assert outcome.converged
    assert outcome.energy <= reference.energy + 1e-12
    assert outcome.orbital.grad_norm < 1e-6
    # The CI attached to the outcome is the one solved on the converged orbitals.
    assert abs(outcome.ci.energy - outcome.energy) < 1e-9
    assert outcome.state_energies.size == 1


def test_the_casscf_energy_is_invariant_to_an_active_space_rotation(system):
    """Tier 0. An exact CAS is invariant to a unitary mixing of the active orbitals, so a
    CASSCF started from a rotated set must land on the same energy."""
    factors, h_ao, c0, spaces, n_elec = system
    rng = np.random.default_rng(11)
    a = rng.standard_normal((spaces.n_active,) * 2) + 1j * rng.standard_normal(
        (spaces.n_active,) * 2)
    a = 0.2 * (a - a.conj().T)
    from kuiva.mcscf.orbopt import unitary_from_antihermitian
    rotated = c0.copy()
    rotated[:, spaces.active] = c0[:, spaces.active] @ unitary_from_antihermitian(a)

    plain = casci(factors, h_ao, c0, spaces, n_elec, report=False)
    turned = casci(factors, h_ao, np.ascontiguousarray(rotated), spaces, n_elec, report=False)
    assert abs(plain.energy - turned.energy) < 1e-9


def test_the_ci_energy_is_invariant_to_a_global_phase():
    """Tier 0: a CI vector is defined up to a phase, and nothing observable may see it."""
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=6)
    solver = FullCISolver(n, k, n_states=2, enforce_kramers=False)
    result = solver.solve_active(h, eri)
    phased = np.ascontiguousarray(result.vectors * np.exp(1j * np.array([[0.7], [-2.1]])))
    gamma, gamma2 = solver._builder()(phased, result.weights, enforce_kramers=False)
    assert np.max(np.abs(gamma - result.gamma)) < 1e-12
    assert np.max(np.abs(gamma2 - result.gamma2)) < 1e-12


def test_a_casci_over_every_spinor_with_no_inactive_space_is_the_full_space(system):
    """CAS over the whole orbital set: the "active space" is then the entire problem, and the
    lowest root is the exact ground state of the two-component Hamiltonian in this basis."""
    factors, h_ao, c0, _, _ = system
    n = c0.shape[1]
    spaces = OrbitalSpaces.from_counts(n_inactive=0, n_active=n, n_orb=n)
    result = casci(factors, h_ao, c0, spaces, 2, report=False)
    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    _, values, _ = reference_states(n, 2, ints.h_active_effective(), ints.active_eri(), 1)
    assert abs(result.total_energies[0] - values[0]) < EXACT


# --- active-space selection -------------------------------------------------------------------

def test_explicit_selection_derives_the_electron_count_by_aufbau():
    space = active_space([4, 5, 6, 7], n_orb=12, n_elec_total=6)
    assert space.n_elec == 2                       # spinors 0..3 hold four of the six
    assert space.spaces.n_inactive == 4
    assert list(space.spaces.inactive) == [0, 1, 2, 3]
    assert list(space.spaces.virtual) == [8, 9, 10, 11]


def test_an_explicit_electron_count_overrides_the_aufbau_reading():
    space = active_space([4, 5, 6, 7], n_orb=12, n_elec_total=6, n_active_elec=4)
    assert space.spaces.n_inactive == 2
    assert space.n_elec == 4


def test_an_odd_inactive_count_is_refused():
    """⚠ It would split a Kramers pair across the space boundary, and the answer would
    still look perfectly plausible."""
    with pytest.raises(ValueError, match="odd inactive count"):
        active_space([4, 5, 6, 7], n_orb=12, n_elec_total=6, n_active_elec=3)


def test_the_electron_count_trap_produces_a_different_space():
    """The occupation-count trap, as an assertion rather than a comment: ``2 * (mo_occ > 0).sum()`` over an
    ROHF set over-counts by one per open shell, and the two readings disagree."""
    # Six electrons, two of them in singly occupied MOs: occ > 0 on four spatial orbitals.
    n_elec_total, n_active_elec = 6, 2
    right = n_elec_total - n_active_elec                       # 4 inactive spinors
    wrong = 2 * 4                                              # 2 * (mo_occ > 0).sum() = 8
    assert right != wrong
    space = active_space([4, 5, 6, 7], n_orb=12, n_elec_total=n_elec_total,
                         n_active_elec=n_active_elec)
    assert space.spaces.n_inactive == right


def test_too_many_active_electrons_is_refused():
    with pytest.raises(ValueError, match="impossible"):
        active_space([4, 5], n_orb=12, n_elec_total=6, n_active_elec=4)


def test_an_out_of_range_active_index_is_refused():
    with pytest.raises(ValueError, match="must lie in"):
        active_space([10, 11, 12], n_orb=12, n_elec_total=6)


def test_a_contiguous_range_on_an_unrestricted_set_warns(kuiva_caplog):
    """⚠ With an unrestricted reference spinors 2p and 2p+1 are the p-th alpha and p-th
    beta orbital, and a contiguous range is the natural, wrong thing to write."""
    active_space([4, 5, 6, 7], n_orb=12, n_elec_total=6, kramers_paired=False)
    assert any("not Kramers paired" in r.message for r in kuiva_caplog.records)


def test_a_principal_quantum_number_is_refused_by_the_character_selector():
    """The AO-label trap: "6p" counts shells *within the basis*, so it means different orbitals in
    a segmented and a general set. Only (atom, l) is unambiguous."""
    from kuiva.mcscf.casci import _angular_momentum

    assert _angular_momentum("d") == 2
    assert _angular_momentum(3) == 3
    with pytest.raises(ValueError, match="basis-dependent"):
        _angular_momentum("6p")


# --- selection by angular-momentum character -------------------------------------------

@pytest.fixture(scope="module")
def character_system():
    """A Ti/Cl pair, an AO layout, and a spinor set whose pairs are single orthogonalized AOs.

    No SCF: taking the columns of ``S^{-1/2}`` as the spinors makes each Kramers pair carry
    **exactly** 100% of its Löwdin population on one AO, so what the selector returns is
    known in advance rather than inferred. That is what lets this test fail for the right
    reason.
    """
    from pyscf import gto

    from kuiva.interface.pyscf_bridge import ao_layout
    from kuiva.orth.canonical import symmetric_orthogonalization

    mol = gto.M(atom=[["Ti", (0, 0, 0)], ["Cl", (0, 0, 2.3)]], basis="x2c-SVPall-2c",
                spin=1, verbose=0)
    layout = ao_layout(mol)
    s_ao = mol.intor("int1e_ovlp")
    x = symmetric_orthogonalization(s_ao).x               # S^{-1/2}, so S^{1/2} x = 1
    ti_d = np.nonzero((layout.ao_atom == 0) & (layout.ao_l == 2))[0]
    cl_p = np.nonzero((layout.ao_atom == 1) & (layout.ao_l == 1))[0]
    # ⚠ Interleaved on purpose: "the lowest pairs of d character" must mean the lowest ones,
    # not the first block the loop happens to reach.
    order = np.array([cl_p[0], ti_d[0], ti_d[1], cl_p[1], ti_d[2], ti_d[3], ti_d[4],
                      cl_p[2]])
    nao = mol.nao
    coeff = np.zeros((2 * nao, 2 * order.size), dtype=np.complex128)
    for j, mu in enumerate(order):
        coeff[:nao, 2 * j] = x[:, mu]                  # unbarred
        coeff[nao:, 2 * j + 1] = x[:, mu]              # barred
    return coeff, s_ao, layout, order, ti_d


def test_character_selection_takes_the_lowest_pairs_of_that_character(character_system):
    coeff, s_ao, layout, order, ti_d = character_system
    space = active_space_by_character(coeff, s_ao, layout, n_elec_total=4,
                                      atom="Ti", l="d", n_pairs=3, n_active_elec=2)
    # Pairs 1, 2 and 4 are the three lowest Ti d ones (pair 0 is Cl p, pair 3 is Cl p).
    assert list(space.spaces.active) == [2, 3, 4, 5, 8, 9]
    assert space.n_elec == 2
    assert "d" in space.description or "l=2" in space.description
    del ti_d, order


def test_character_selection_returns_whole_kramers_pairs(character_system):
    coeff, s_ao, layout, _, _ = character_system
    space = active_space_by_character(coeff, s_ao, layout, n_elec_total=2,
                                      atom=0, l=2, n_pairs=2, n_active_elec=2)
    active = np.asarray(space.spaces.active)
    assert active.size % 2 == 0
    assert np.array_equal(active[0::2] + 1, active[1::2])       # 2p and 2p+1 together


def test_asking_for_more_pairs_than_exist_says_what_the_candidates_look_like(
        character_system):
    coeff, s_ao, layout, _, _ = character_system
    with pytest.raises(ValueError, match="best candidates"):
        active_space_by_character(coeff, s_ao, layout, n_elec_total=2, atom="Ti", l="d",
                                  n_pairs=6, n_active_elec=2)


def test_a_character_absent_from_the_basis_is_refused(character_system):
    coeff, s_ao, layout, _, _ = character_system
    with pytest.raises(ValueError, match="no l = 6"):
        active_space_by_character(coeff, s_ao, layout, n_elec_total=2, atom="Ti", l=6,
                                  n_pairs=1)


def test_a_repeated_element_symbol_demands_an_index():
    """⚠ Which centre the active space sits on is the whole point, so an ambiguous symbol is
    refused rather than resolved to the first match."""
    from pyscf import gto

    from kuiva.interface.pyscf_bridge import ao_layout
    from kuiva.mcscf.casci import _atom_index

    mol = gto.M(atom=[["Ti", (0, 0, 0)], ["Ti", (0, 0, 4.0)]], basis="x2c-SVPall-2c",
                spin=2, verbose=0)
    layout = ao_layout(mol)
    with pytest.raises(ValueError, match="appears 2 times"):
        _atom_index(layout, "Ti")
    assert _atom_index(layout, 1) == 1


def test_a_multi_centre_active_space_is_stated_as_a_list_of_centres(character_system):
    """A dimer's active space spans **both** metals, and that is one physical statement.

    ⚠ The scalar-symbol refusal above must not make a multi-site active space unstatable:
    ``ti2cl6``'s d^1 (x) d^1 manifold is ten Kramers pairs of d character across two titaniums,
    and reproducibility requires it be said that way rather than as an index window. Here the second
    centre is Cl, so asking for both centres' p+d population and getting the *union* is the
    same mechanism on a system whose answer is known by construction.
    """
    coeff, s_ao, layout, order, ti_d = character_system
    both = active_space_by_character(coeff, s_ao, layout, n_elec_total=4,
                                     atom=[0, 1], l="d", n_pairs=3, n_active_elec=2)
    only_ti = active_space_by_character(coeff, s_ao, layout, n_elec_total=4,
                                        atom="Ti", l="d", n_pairs=3, n_active_elec=2)
    # The Cl in this fixture carries no d population, so the two selections coincide -- but
    # the *statement* differs, and the description says so.
    assert list(both.spaces.active) == list(only_ti.spaces.active)
    assert "and" in both.description and both.description != only_ti.description
    del order, ti_d


def test_active_space_reports_itself_in_physical_terms():
    space = active_space([4, 5], n_orb=8, n_elec_total=6, description="the 2p pair on Ne")
    assert "2p pair" in repr(space) or "2p pair" in space.description
    assert isinstance(space, ActiveSpace)


# --- Tier 1: against PySCF, with spin-orbit coupling off ---------------------------------------

@pytest.mark.slow
def test_scalar_casscf_matches_pyscf():
    """⚠ The check that ties the whole stack to an external reference (Tier 1).

    With SOC switched off the two-component problem is two identical copies of a scalar one,
    so a spinor CASSCF must reproduce PySCF's scalar CASSCF **exactly** — not approximately:
    the same Hamiltonian, the same active space, the same variational principle. The target
    is 1e-8 Eh, the locked tolerance for a scalar energy.

    It runs on LiH/STO-3G with CAS(2,4 spinors) = CAS(2,2 spatial), which is a few seconds.
    """
    pyscf_mcscf = pytest.importorskip("pyscf.mcscf")
    from pyscf import ao2mo, gto, scf

    from kuiva.spinor.expand import spin_block_diagonal

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ncas, nelecas = 2, 2                                  # spatial orbitals / electrons
    reference = pyscf_mcscf.CASSCF(mf, ncas, nelecas)
    reference.conv_tol = 1e-12
    e_reference = reference.kernel()[0]

    # The same problem in the spinor basis. With no spin-orbit coupling the two-component
    # Hamiltonian is block diagonal in spin and the ERI are spin-free, so each scalar MO
    # becomes a Kramers pair (the interleaved ordering) and the spinor CASSCF is two
    # identical copies of the scalar one. Anything but exact agreement is a defect.
    nao = mol.nao
    factors = ThreeIndexAO.from_eri(ao2mo.restore(1, mf._eri, nao), nao, 1e-12, report=False)
    h_ao = spin_block_diagonal(mf.get_hcore())
    c_spinor = np.zeros((2 * nao, 2 * mf.mo_coeff.shape[1]), dtype=np.complex128)
    c_spinor[:nao, 0::2] = mf.mo_coeff                    # unbarred partners
    c_spinor[nao:, 1::2] = mf.mo_coeff                    # barred partners

    n_inactive = mol.nelectron - nelecas # ⚠ from the ELECTRON count
    assert n_inactive % 2 == 0
    spaces = OrbitalSpaces.from_counts(n_inactive, 2 * ncas, c_spinor.shape[1])
    outcome = casscf(factors, h_ao, c_spinor, spaces, nelecas, mode="second-order",
                     max_iter=60, conv_grad=1e-7, e_nuc=mol.energy_nuc(), report=False,
                     solver_options={"enforce_kramers": False})
    assert abs(outcome.energy - e_reference) < 1e-8


def test_casci_result_reports_relative_energies_in_wavenumbers():
    result = CASCIResult(energies=np.array([-1.0, -0.9]), vectors=np.zeros((2, 2), complex),
                         weights=np.array([0.5, 0.5]), gamma=np.zeros((2, 2), complex),
                         gamma2=np.zeros((2, 2, 2, 2), complex), e_core=3.0, n_apply=0,
                         n_iter=0)
    assert abs(result.excitation_energies_cm()[1] - 0.1 * 219474.6313632) < 1e-6
    assert abs(result.energy - (-0.95 + 3.0)) < 1e-12
    assert np.allclose(result.total_energies, np.array([2.0, 2.1]))


def test_a_non_converging_ci_raises_rather_than_returning_a_vector():
    """⚠ An unconverged eigenvector poisons the RDMs, the gradient and every macro-iteration
    after it; a raised :class:`SolverFailure` is something the optimizer can act on."""
    # ⚠ Above DENSE_SOLVE_MAX_DET, deliberately: below it the solver diagonalizes densely,
    # which cannot fail to converge and so cannot exercise this at all.
    n, k = 12, 6
    h, eri = random_spinor_integrals(n, seed=2)
    solver = FullCISolver(n, k, n_states=2, max_iter=2, conv_tol=1e-14,
                          enforce_kramers=False)
    assert solver.ndet > 250
    with pytest.raises(SolverFailure):
        solver.solve_active(h, eri)


# --- The state-average boundary (the C1 defect was this) ---------------------

class _ActiveHamiltonian:
    """The two methods :func:`state_average_boundary` actually consumes.

    A stub rather than a real :class:`CASIntegrals`, deliberately: building one needs Cholesky
    factors and an orbital set, none of which this is about. What it pins is the **contract** —
    the boundary measurement reads the effective one-electron matrix and the active ERIs and
    nothing else, so it can never depend on orbital-space bookkeeping it has no business
    seeing. The end-to-end path is covered by the Tier-2 records.
    """

    def __init__(self, h, eri):
        self._h, self._eri = h, eri

    def h_active_effective(self):
        return self._h

    def active_eri(self):
        return self._eri


def test_spectrum_does_not_disturb_the_states_the_caller_is_using():
    """⚠ The load-bearing property of :meth:`FullCISolver.spectrum`. It exists to look at roots
    the average does *not* take, so it must leave the ones it does take exactly alone — the
    stored result, the warm start and the state count."""
    n, k = 10, 4
    h, eri = random_spinor_integrals(n, seed=3)
    solver = FullCISolver(n, k, n_states=4, enforce_kramers=False)
    kept = solver.solve_active(h, eri)
    energies, vectors, n_states = kept.energies.copy(), kept.vectors.copy(), solver.n_states

    extra = solver.spectrum(h, eri, 8)

    assert solver.last is kept, "spectrum() replaced the caller's states"
    assert np.array_equal(solver.last.energies, energies)
    assert np.array_equal(solver.last.vectors, vectors)
    assert solver.n_states == n_states
    # It really did solve more roots, and they extend the same spectrum.
    assert extra.size == 8
    assert np.allclose(extra[:4], energies, atol=1e-7)


def test_a_full_space_average_reports_no_boundary_rather_than_a_zero_gap():
    """⚠ **A null gap is a pass, and the distinction is physics.** Averaging over the whole CI
    space leaves no root out, so the set is complete by construction — which is exactly why
    Bi, Ce(3+) and Yb(3+) were never at risk from C1 while Dy(3+) was."""
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=5)
    solver = FullCISolver(n, k, n_states=20, enforce_kramers=False)
    assert solver.ndet == 20
    report = state_average_boundary(solver, _ActiveHamiltonian(h, eri))
    assert report.spans_full_ci and report.gap_cm is None
    assert report.is_clean


def test_the_boundary_gap_is_measured_against_the_first_root_left_out():
    """The gap must be the distance from root ``n_states`` to root ``n_states + 1`` of the
    *same* spectrum, which a plain dense diagonalization can confirm independently."""
    n, k, n_states = 8, 4, 6
    h, eri = random_spinor_integrals(n, seed=7)
    solver = FullCISolver(n, k, n_states=n_states, enforce_kramers=False)
    solver.solve_active(h, eri)
    report = state_average_boundary(solver, _ActiveHamiltonian(h, eri))

    space = CASSpace(n, k)
    exact = np.linalg.eigvalsh(np.asarray(hamiltonian_matrix(space, h, eri).todense()))
    expected = (exact[n_states] - exact[n_states - 1]) * 219474.6313632
    assert report.gap_cm == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert report.ndet == space.ndet


def test_an_ambiguous_boundary_warns_and_a_clean_one_does_not(kuiva_caplog):
    """⚠ **The guard for C1's mechanism.** The failure it describes produced a converged
    calculation with plausible energies, plausible g values and a plausible spectrum; the only
    evidence was the root the average did not take. So the warning is the deliverable, and it
    is tested as behaviour."""
    import logging

    cut = BoundaryReport(n_states=126, ndet=2002, margin=8, gap_cm=3.94,
                         next_cm=(0.0, 3.94, 8.61))
    clean = BoundaryReport(n_states=134, ndet=2002, margin=8, gap_cm=2057.5,
                           next_cm=(0.0, 2057.5))
    assert not cut.is_clean and clean.is_clean

    with kuiva_caplog.at_level(logging.WARNING):
        cut.report()
    assert any("cutting a degenerate manifold" in r.message for r in kuiva_caplog.records)

    kuiva_caplog.clear()
    with kuiva_caplog.at_level(logging.WARNING):
        clean.report()
    assert not [r for r in kuiva_caplog.records if r.levelno >= logging.WARNING]


def test_the_warning_threshold_is_the_one_the_reference_suite_asserts():
    """The library warns and ``tests/test_tier2_soc.py`` asserts; they must agree, or a record
    can pass the suite while the calculation that produced it warned."""
    import test_tier2_soc                                                # noqa: PLC0415
    assert test_tier2_soc.BOUNDARY_GAP_MIN_CM == BOUNDARY_WARN_CM


# --- An EVEN cut through a degenerate block: the mechanism, not the observable -------------
#
# A fix must leave behind a test for the *mechanism*. The observable was
# dy3p's scalar CASSCF dying three macro-iterations in with a broken Kramers degeneracy; the
# mechanism is below, and it is cheap, synthetic and has nothing to do with dysprosium.

def _spin_free_spinor_integrals(n_spatial: int, seed: int):
    """A **spin-free** Hamiltonian in the interleaved spinor basis.

    Spin-free is the whole point: it is what gives the CI exact spin multiplets, so an
    odd-electron spectrum contains blocks of **four** (quartets) as well as of two. A block of
    four is the case the odd-block state-averaging gate is blind to, because half of four is even.

    ``L`` is symmetric in its orbital pair, so the spatial ERI has full 8-fold symmetry and is
    positive semidefinite — a real Hamiltonian, not a random Hermitian matrix (the Davidson
    needs an informative diagonal).
    """
    rng = np.random.default_rng(seed)
    h_s = rng.standard_normal((n_spatial, n_spatial))
    h_s = 0.5 * (h_s + h_s.T)
    l = rng.standard_normal((2 * n_spatial, n_spatial, n_spatial)) * 0.4
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


def _time_odd_fraction(a: np.ndarray) -> float:
    """``max |J conj(A) J^T - A| / max|A|`` in the interleaved pair order.

    ``J`` is time reversal on the MO index (``J c_2p = c_2p+1``, ``J c_2p+1 = -c_2p``), so an
    object built from a time-reversal-invariant ensemble satisfies ``J conj(A) J^T = A``
    exactly. ⚠ Written out rather than taken from
    :func:`kuiva.spinor.expand.time_reversal_residual`, which measures the residual of a
    projection onto the time-even **Hermitian** operators in the *blocked* layout — a different
    statement, and one that reports any anti-Hermitian matrix as 100% odd whatever its symmetry.
    """
    n = a.shape[0]
    j = np.zeros((n, n))
    j[1::2, 0::2] = np.eye(n // 2)
    j[0::2, 1::2] = -np.eye(n // 2)
    return float(np.max(np.abs(j @ np.conj(a) @ j.T - a))) / float(np.max(np.abs(a)))


def _first_quartet(energies: np.ndarray, tol: float = 1e-9):
    """``(start, stop)`` of the first block of four in an ascending spectrum."""
    from kuiva.rdm.rdm import degenerate_blocks                        # noqa: PLC0415
    for a, b in degenerate_blocks(energies, tol):
        if b - a == 4:
            return a, b
    raise AssertionError("no 4-fold block in this spectrum: {}".format(energies[:12]))


def test_an_even_cut_through_a_degenerate_block_passes_the_kramers_gate():
    """⚠ **The hole the odd-block state-averaging gate leaves open, asserted as a hole.**

    Kramers' theorem makes an *odd* block impossible for an odd electron count, which is what
    makes that refusal rigorous — and it is exactly why it cannot see a block of four cut in
    half. This is not a defect in the gate; it is the reason the boundary gap has to be
    measured separately, and the reason this test exists next to it.
    """
    from kuiva.rdm.rdm import state_average_weights                    # noqa: PLC0415

    h, eri = _spin_free_spinor_integrals(3, seed=11)
    solver = FullCISolver(6, 3, n_states=20, enforce_kramers=False)
    energies = solver.solve_active(h, eri).energies
    start, stop = _first_quartet(energies)
    cut = start + 2                                     # two of the four, an even remainder

    # No raise, no complaint: every block of the retained set has an even number of members.
    weights = state_average_weights(energies[:cut], 3)
    assert weights.size == cut
    # ... and it is a genuine cut, not a boundary that happens to be clean.
    assert stop - start == 4 and start < cut < stop
    assert energies[cut] - energies[cut - 1] < 1e-9, "the block really is degenerate here"


def test_an_even_cut_through_a_degenerate_block_breaks_the_averaged_DENSITY():
    """⚠ **The mechanism itself**: what an incomplete average actually damages.

    Inside a degenerate manifold the individual roots are defined only up to a rotation, so
    only a sum over the *whole* block is invariant. Take half of a quartet and the
    state-averaged 1-RDM stops being time-reversal invariant — by an amount that has nothing to
    do with any tolerance, because it is not an error but a different quantity. The Fock
    operator built from it is then not time-even either, the orbitals optimize on that, and the
    CI spectrum it produces splits Kramers pairs. Measured on dy3p's scalar CASSCF: 6.5e-3 of
    time-odd density at macro-iteration 2, from orbitals still paired to 1.3e-9.

    The complete average is the control, and it is what makes this a measurement rather than an
    assertion that a number is large.
    """
    h, eri = _spin_free_spinor_integrals(3, seed=11)
    solver = FullCISolver(6, 3, n_states=20, enforce_kramers=False)
    energies = solver.solve_active(h, eri).energies
    start, stop = _first_quartet(energies)
    # Both averages run from root 0; they differ only in whether the quartet is whole. Uniform
    # weights throughout, so what is measured is the manifold and not the weighting policy.
    complete = FullCISolver(6, 3, n_states=stop, enforce_kramers=False)
    whole = _time_odd_fraction(complete.solve_active(h, eri).gamma)
    cut_solver = FullCISolver(6, 3, n_states=start + 2, enforce_kramers=False)
    half = _time_odd_fraction(cut_solver.solve_active(h, eri).gamma)

    assert whole < 1e-9, "averaging over the whole block must leave gamma time-even"
    assert half > 1e-3, "half a quartet must not"
    assert half > 1e5 * max(whole, 1e-15), "and the two must not be the same number"


# --- The spin-invariance half of "is this ensemble safe" -----------------------------------
#
# A clean boundary gap says the cut is unambiguous; it deliberately does not say the ensemble
# is one the symmetry leaves invariant, and the measured counterexample is a complete 2J+1
# manifold whose average converged with that manifold split by 725 cm^-1. The detector for
# the second statement is the spin-rotation non-invariance of the averaged density: exactly
# zero for a complete spin multiplet (or a term-complete average), order one for any ensemble
# that weights the spin-orbit structure of its states.

def _interleaved_spin_operator(n_spatial: int) -> np.ndarray:
    """``S_k`` in the synthetic interleaved (spatial, spin) spinor basis."""
    sigma = np.array([[[0.0, 1.0], [1.0, 0.0]],
                      [[0.0, -1.0j], [1.0j, 0.0]],
                      [[1.0, 0.0], [0.0, -1.0]]], dtype=np.complex128)
    return np.stack([np.kron(np.eye(n_spatial), 0.5 * sigma[k]) for k in range(3)])


def test_spin_noninvariance_separates_complete_from_cut_spin_multiplets():
    """The detector itself, on the same synthetic system as the mechanism tests above:
    averaging a whole quartet leaves the density spin-rotation invariant, half of it does
    not — and the two differ by orders of magnitude, not by a tolerance."""
    from kuiva.mcscf.casci import ensemble_spin_noninvariance                # noqa: PLC0415

    h, eri = _spin_free_spinor_integrals(3, seed=11)
    s_mo = _interleaved_spin_operator(3)
    solver = FullCISolver(6, 3, n_states=20, enforce_kramers=False)
    energies = solver.solve_active(h, eri).energies
    start, stop = _first_quartet(energies)

    whole = FullCISolver(6, 3, n_states=stop, enforce_kramers=False)
    s_whole = ensemble_spin_noninvariance(whole.solve_active(h, eri).gamma, s_mo)
    cut = FullCISolver(6, 3, n_states=start + 2, enforce_kramers=False)
    s_cut = ensemble_spin_noninvariance(cut.solve_active(h, eri).gamma, s_mo)

    assert s_whole < 1e-9, "a complete spin-multiplet average must be spin-rotation invariant"
    assert s_cut > 0.05, "half a quartet must lean on the spin structure"
    assert s_cut > 1e5 * max(s_whole, 1e-15)


def test_boundary_report_carries_and_reports_the_spin_invariance(kuiva_caplog):
    """The field reaches the report, and an ambiguous boundary of a *leaning* ensemble warns
    with the mechanism named — the warning a user actually needs is the combination, because
    each half alone has a benign reading (a clean gap; a stable ligand-field doublet)."""
    import logging                                                     # noqa: PLC0415

    leaning = BoundaryReport(n_states=2, ndet=6, margin=4, gap_cm=10.0,
                             next_cm=(0.0, 10.0), spin_noninvariance=0.64)
    assert leaning.leaning is True
    with kuiva_caplog.at_level(logging.INFO):
        leaning.report()
    messages = [r.getMessage() for r in kuiva_caplog.records]
    assert any("spin invariance" in m and "leaning" in m for m in messages)
    warnings = [r.getMessage() for r in kuiva_caplog.records
                if r.levelno >= logging.WARNING]
    assert any("not spin-rotation invariant" in m for m in warnings)

    invariant = BoundaryReport(n_states=6, ndet=6, margin=0, gap_cm=None,
                               spin_noninvariance=3e-16)
    assert invariant.leaning is False
    unmeasured = BoundaryReport(n_states=2, ndet=6, margin=4, gap_cm=500.0)
    assert unmeasured.leaning is None


def test_casscf_measures_spin_invariance_when_given_the_spin_operator(system):
    """End to end through the mcscf driver: with ``spin_ao_2c`` the converged boundary
    report carries the number; without it the field stays ``None`` (the pre-flight one
    always does — no density exists yet where it runs)."""
    factors, h_ao, c0, spaces, n_elec = system
    n = h_ao.shape[0]
    rng = np.random.default_rng(7)
    s2c = rng.standard_normal((3, n, n)) + 1j * rng.standard_normal((3, n, n))
    s2c = 0.5 * (s2c + s2c.conj().transpose(0, 2, 1))

    outcome = casscf(factors, h_ao, c0, spaces, n_elec, n_states=2, max_iter=3,
                     report=False, spin_ao_2c=s2c)
    assert outcome.boundary is not None
    assert outcome.boundary.spin_noninvariance is not None
    assert outcome.boundary_initial is not None
    assert outcome.boundary_initial.spin_noninvariance is None

    bare = casscf(factors, h_ao, c0, spaces, n_elec, n_states=2, max_iter=3, report=False)
    assert bare.boundary.spin_noninvariance is None


def test_the_boundary_is_measured_before_the_optimization_and_not_only_after(system,
                                                                            kuiva_caplog):
    """⚠ **The check that can actually save a run.**

    An incomplete average does its damage *along* the trajectory, so by the time the converged
    check runs the orbitals are already wrong — and if the odd-block state-averaging gate fires on the way,
    there is no converged check at all. On dy3p that is exactly what happened: the run died at
    macro-iteration 3 with nothing before it to say why. So the boundary is measured at the
    starting orbitals too, and this asserts it reaches the log *even when the optimization
    afterwards blows up*, which is the only case where it is the difference between a diagnosis
    and none.
    """
    import logging                                                     # noqa: PLC0415

    factors, h_ao, c0, spaces, n_elec = system

    class Boom(RuntimeError):
        pass

    def explode(_info):
        raise Boom("the optimization dies here")

    with kuiva_caplog.at_level(logging.INFO):
        with pytest.raises(Boom):
            casscf(factors, h_ao, c0, spaces, n_elec, n_states=2, max_iter=4,
                   report=True, callback=explode)
    messages = [r.getMessage() for r in kuiva_caplog.records]
    assert any("state-average boundary" in m and "starting orbitals" in m for m in messages), \
        "the pre-flight boundary never reached the log: {}".format(messages)


def test_both_boundaries_are_reported_and_say_which_orbitals_they_are_about(system):
    """The two measurements are different statements and the report has to distinguish them —
    a converged boundary quoted as if it were the starting one is how a trajectory that was
    never safe gets read as one that was."""
    factors, h_ao, c0, spaces, n_elec = system
    outcome = casscf(factors, h_ao, c0, spaces, n_elec, n_states=2, max_iter=3, report=False)
    assert outcome.boundary_initial is not None and outcome.boundary is not None
    assert outcome.boundary_initial.where == "starting orbitals"
    assert outcome.boundary.where == "converged orbitals"
    # Switching the check off switches off both, and leaves neither behind.
    off = casscf(factors, h_ao, c0, spaces, n_elec, n_states=2, max_iter=3, report=False,
                 boundary_check=0)
    assert off.boundary_initial is None and off.boundary is None


def test_a_boundary_check_that_fails_to_converge_warns_and_does_not_kill_the_run(
        system, monkeypatch, kuiva_caplog):
    """⚠ **A diagnostic may not kill the calculation it is diagnosing.**

    The extra roots are harder than the ones the average uses — higher, so the diagonal
    preconditioner is worse there, and with no warm start of their own — so a Davidson that
    converges ``n_states`` roots can fail on ``n_states + margin``. Measured on dy3p: 142 roots
    stalled at ``max|r| = 1.8e-08`` against a 1e-8 tolerance, which threw away a whole CASSCF
    for the sake of an advisory check. ⚠ And the result is ``None``, never a clean report:
    *"the boundary could not be measured"* is a weaker statement than *"the boundary is
    clean"*, and a run must not be able to confuse them.
    """
    import logging                                                     # noqa: PLC0415

    factors, h_ao, c0, spaces, n_elec = system

    def stall(*_args, **_kwargs):
        raise SolverFailure("CAS boundary Davidson did not converge")

    monkeypatch.setattr(FullCISolver, "spectrum", stall)
    with kuiva_caplog.at_level(logging.WARNING):
        outcome = casscf(factors, h_ao, c0, spaces, n_elec, n_states=2, max_iter=3,
                         report=False)

    assert outcome.boundary is None and outcome.boundary_initial is None
    assert outcome.energy is not None and outcome.ci.energies.size == 2
    failures = [r.getMessage() for r in kuiva_caplog.records
                if "could not measure the state-average boundary" in r.getMessage()]
    assert len(failures) == 2, "both boundaries must report their own failure"
    assert any("starting orbitals" in m for m in failures)
    assert any("converged orbitals" in m for m in failures)
