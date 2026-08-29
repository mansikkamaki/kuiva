"""Tier-0/1 tests for the fixed-topology two-site DMRG solver.

The oracle is dense diagonalization of the same Hamiltonian through the *independent*
Slater-Condon machinery of ``ci/strings.py``: when the bond dimension saturates the Schmidt
rank, the sweep energies must equal the exact CI eigenvalues to 1e-8 Eh
— on paths and on trees, single-root and state-averaged. Kramers structure (the general path's
1e-8..1e-6 Eh band), the weight-equalization discipline and the boundary-gap diagnostic are asserted
as behaviour.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from kuiva.ci.strings import Determinants, hamiltonian_matrix
from kuiva.dmrg.block import BlockTensor, QuantumNumber, tensordot, block_tensor_gb
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.sweep import random_state, solve_ttn, state_gb
from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms

from test_ci_strings import random_spinor_integrals
from test_ci_davidson import kramers_spinor_integrals

E_TOL = 1e-8            # energies to 1e-8 Eh when the bond dimension saturates


def exact_energies(n, k, h, eri, n_roots):
    """Dense reference spectrum from the independent Slater-Condon implementation."""
    from itertools import combinations
    masks = np.sort(np.array([sum(1 << i for i in c)
                              for c in combinations(range(n), k)], dtype=np.uint64))
    dets = Determinants(masks, n_spinor=n, n_elec=k)
    ham = hamiltonian_matrix(dets, h, eri).toarray()
    return np.sort(np.linalg.eigvalsh(ham))[:n_roots]


def solve(graph, h, eri, k, n_roots=1, max_bond=200, seed=0, **kw):
    # boundary_check off by default in the tests (it costs one extra sweep per solve and
    # has its own dedicated test); production keeps solve_ttn's boundary-check default
    kw.setdefault("boundary_check", 0)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = random_state(op, k, max_bond, n_roots=n_roots,
                         rng=np.random.default_rng(seed))
    return solve_ttn(op, state, max_bond=max_bond, **kw)


# --- exactness at saturated bond dimension ------------------------------------------------

def test_ground_state_on_a_path_equals_exact_ci():
    # even k throughout the random-integral tests: these integrals are not time-reversal
    # symmetric, so the structural Kramers refusal (asserted in its own test below)
    # would rightly reject odd root counts of an odd-electron system
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=11)
    result = solve(NetworkGraph.path(n), h, eri, k)
    assert result.converged
    ref = exact_energies(n, k, h, eri, 1)
    assert abs(result.energies[0] - ref[0]) < E_TOL


def test_state_average_on_a_path_equals_exact_ci():
    n, k, n_roots = 6, 2, 3
    h, eri = random_spinor_integrals(n, seed=12)
    result = solve(NetworkGraph.path(n), h, eri, k, n_roots=n_roots)
    assert result.converged
    ref = exact_energies(n, k, h, eri, n_roots)
    assert np.max(np.abs(result.energies - ref)) < E_TOL


def test_ground_state_on_a_tree_equals_exact_ci():
    #    0 - 1 - 2       branched topology: the same code path, no chain assumptions
    #        |
    #        3 - 4
    #        |
    #        5
    graph = NetworkGraph(6, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=13)
    result = solve(graph, h, eri, k, n_roots=2)
    assert result.converged
    ref = exact_energies(n, k, h, eri, 2)
    assert np.max(np.abs(result.energies - ref)) < E_TOL


def test_multi_mode_nodes_equal_exact_ci():
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=14)
    graph = NetworkGraph(3, [(0, 1), (1, 2)], contents=[(0, 1), (2, 3), (4, 5)])
    result = solve(graph, h, eri, k)
    assert result.converged
    ref = exact_energies(n, k, h, eri, 1)
    assert abs(result.energies[0] - ref[0]) < E_TOL


# --- truncation is variational and reported -----------------------------------------------

def test_small_truncated_run_is_variational_and_reported():
    """Fast-suite version of the truncation contract; the n = 8 one is in the slow suite."""
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=15, scale=0.2)
    ref = exact_energies(n, k, h, eri, 1)
    loose = solve(NetworkGraph.path(n), h, eri, k, max_bond=4, seed=1, max_sweeps=6)
    assert loose.energies[0] >= ref[0] - E_TOL
    assert loose.max_bond_dim <= 4
    assert loose.max_discarded > 0.0


@pytest.mark.slow
def test_truncated_energy_is_variational_and_reported():
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=15, scale=0.2)
    ref = exact_energies(n, k, h, eri, 1)
    loose = solve(NetworkGraph.path(n), h, eri, k, max_bond=8, seed=1, max_sweeps=6)
    assert loose.energies[0] >= ref[0] - E_TOL          # variational
    assert loose.max_bond_dim <= 8                      # the cap held
    assert loose.max_discarded > 0.0                    # the truncation is reported


# --- Kramers structure (numerical Kramers band, state-averaging discipline) ---------------------------------------

def test_kramers_doublets_and_weight_discipline():
    """Time-reversal-symmetric H, odd electrons: levels come in exact Kramers pairs.

    The general (Kramers-unrestricted) network resolves the degeneracy only numerically —
    the general-complex path reserves 1e-8..1e-6 Eh for that — and the SA truncation must keep the pair whole.
    An odd root count over a Kramers-degenerate spectrum must be *refused*.
    """
    h, eri = kramers_spinor_integrals(3, seed=16)       # 6 spinors
    n, k = 6, 3
    ref = exact_energies(n, k, h, eri, 4)
    assert ref[1] - ref[0] < 1e-10                      # the model really is doubled
    assert ref[3] - ref[2] < 1e-10

    result = solve(NetworkGraph.path(n), h, eri, k, n_roots=4)
    assert result.converged
    assert np.max(np.abs(result.energies - ref)) < E_TOL
    assert abs(result.energies[1] - result.energies[0]) < 1e-6      # the numerical Kramers band
    assert np.allclose(result.weights, 0.25)            # equalized inside the pairs

    with pytest.raises(ValueError, match="degenerate"):
        solve(NetworkGraph.path(n), h, eri, k, n_roots=3)


def test_boundary_gap_is_reported_and_warns_when_small(kuiva_caplog):
    """The boundary diagnostic, sweep flavour. The extra local root is variational, so the
    reported gap bounds the exact one from above (the docstring's honest statement: a
    local diagnostic, weaker than the full-CI one) — and a count that ends inside a
    Kramers pair must trip the warning."""
    from kuiva.props.multiplet import HARTREE_TO_CM

    n, k = 5, 2
    h, eri = random_spinor_integrals(n, seed=17)
    result = solve(NetworkGraph.path(n), h, eri, k, n_roots=2, boundary_check=4)
    ref = exact_energies(n, k, h, eri, 3)
    exact_gap = (ref[2] - ref[1]) * HARTREE_TO_CM
    assert result.boundary_gap_cm is not None
    assert result.boundary_gap_cm >= exact_gap * (1.0 - 1e-9)      # variational bound
    assert not any("degenerate manifold" in r.message for r in kuiva_caplog.records)

    # end the average inside a Kramers doublet: the boundary gap collapses and warns
    kuiva_caplog.clear()
    h2, eri2 = kramers_spinor_integrals(3, seed=16)
    res2 = solve(NetworkGraph.path(6), h2, eri2, 3, n_roots=3, on_split="warn",
                 boundary_check=4)
    assert res2.boundary_gap_cm < 1.0                              # a split pair: ~0
    assert any("degenerate manifold" in r.message for r in kuiva_caplog.records)


# --- structural details -------------------------------------------------------------------

def test_energies_are_root_and_center_independent():
    n, k = 5, 2
    h, eri = random_spinor_integrals(n, seed=18)
    graph = NetworkGraph.path(n)
    op0 = compile_ttno(graph, hamiltonian_product_terms(h, eri), root=0)
    op2 = compile_ttno(graph, hamiltonian_product_terms(h, eri), root=2)
    e = []
    for op, center in ((op0, 0), (op0, 3), (op2, 2)):
        state = random_state(op, k, 100, n_roots=1, center=center,
                             rng=np.random.default_rng(5))
        e.append(solve_ttn(op, state).energies[0])
    assert np.max(np.abs(np.diff(e))) < E_TOL


def test_state_stays_charge_consistent_and_normalized():
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=19)
    result = solve(NetworkGraph.path(n), h, eri, k, n_roots=2)
    state = result.state
    for c in state.centers:
        assert c.charge == QuantumNumber(k)
        assert c.norm() == pytest.approx(1.0, abs=1e-10)
    for u, t in enumerate(state.tensors):
        if t is not None:
            assert t.charge == QuantumNumber(0)


def test_sizing_function_matches_real_allocation():
    """Two-sided pin of the sizing function against a real array's nbytes."""
    from kuiva.dmrg.block import Space
    sp = Space([(QuantumNumber(0), 3), (QuantumNumber(1), 4), (QuantumNumber(2), 2)])
    ph = Space([(QuantumNumber(0), 1), (QuantumNumber(1), 1)])
    t = BlockTensor.zeros((sp, ph, sp), (1, 1, -1), QuantumNumber(1))
    gb = block_tensor_gb((sp, ph, sp), (1, 1, -1), QuantumNumber(1))
    assert gb * 1024.0 ** 3 == t.nbytes                 # exact, both sides
    n, k = 4, 2
    h, eri = random_spinor_integrals(n, seed=20)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(op, k, 16)
    assert state_gb(state) * 1024.0 ** 3 == state.nbytes


def test_tensordot_matches_dense():
    rng = np.random.default_rng(21)
    from kuiva.dmrg.block import Space
    a_sp = Space([(QuantumNumber(0), 2), (QuantumNumber(1), 3)])
    b_sp = Space([(QuantumNumber(0), 2), (QuantumNumber(1), 2), (QuantumNumber(2), 1)])
    a = BlockTensor.random((a_sp, b_sp), (1, 1), QuantumNumber(1), rng=rng)
    b = BlockTensor.random((b_sp, a_sp), (-1, 1), QuantumNumber(1), rng=rng)
    c = tensordot(a, b, ([1], [0]))
    ref = np.tensordot(a.to_dense(), b.to_dense(), axes=([1], [0]))
    assert np.max(np.abs(c.to_dense() - ref)) < 1e-13
    with pytest.raises(ValueError, match="signs"):
        tensordot(a, b, ([0], [1]))                     # equal signs: flux cannot cancel


# --- production controls: the per-sweep bond schedule and the subspace expansion ------------

def test_bond_schedule_ramps_and_still_reaches_the_exact_answer():
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=31)
    result = solve(NetworkGraph.path(n), h, eri, k, n_roots=2,
                   bond_schedule=[4, 8, 200], max_bond=200)
    assert result.converged
    # convergence may only be declared at the final cap, so the ramp sweeps all ran
    assert result.n_sweeps >= 3
    ref = exact_energies(n, k, h, eri, 2)
    assert np.max(np.abs(result.energies - ref)) < E_TOL


def test_bond_schedule_supplies_max_bond_and_refuses_a_contradiction():
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=32)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(op, k, 200, rng=np.random.default_rng(32))
    result = solve_ttn(op, state, bond_schedule=[8, 200], max_bond=None,
                       boundary_check=0, report=False)
    assert result.converged                              # max_bond taken from the ramp
    assert abs(result.energies[0] - exact_energies(n, k, h, eri, 1)[0]) < E_TOL
    with pytest.raises(ValueError, match="one number"):
        solve(NetworkGraph.path(n), h, eri, k, bond_schedule=[8, 64], max_bond=200)
    with pytest.raises(ValueError, match="ascending"):
        solve(NetworkGraph.path(n), h, eri, k, bond_schedule=[64, 8], max_bond=8)


def test_expansion_changes_no_converged_number():
    # at a saturating cap the enrichment columns are redundant directions: the energies,
    # and the ENSEMBLE discarded weight the result reports, must be exactly the
    # unperturbed story
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=33)
    plain = solve(NetworkGraph.path(n), h, eri, k, n_roots=2)
    perturbed = solve(NetworkGraph.path(n), h, eri, k, n_roots=2, expansion=1e-3)
    assert perturbed.converged
    assert np.max(np.abs(perturbed.energies - plain.energies)) < E_TOL
    assert perturbed.max_discarded < 1e-12               # the ensemble's own, not the
    #                                                      augmented density's


def test_expansion_preserves_kramers_degeneracy():
    """⚠ The load-bearing property of the DETERMINISTIC expansion: the perturbed density
    commutes with time reversal whenever the ensemble and H do, so a Kramers pair's
    Schmidt values stay degenerate and the group-complete truncation keeps meaning what
    it says. A sampled (stochastic) perturbation would split them by O(alpha)."""
    n, k = 6, 3
    h, eri = kramers_spinor_integrals(3, seed=34)        # 3 pairs -> 6 spinors
    result = solve(NetworkGraph.path(n), h, eri, k, n_roots=2, expansion=1e-3)
    assert result.converged
    assert abs(result.energies[1] - result.energies[0]) < 1e-8
