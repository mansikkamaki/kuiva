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


# --- what a sweep will hold, before it holds it (the memory plan) --------------------------

def _local_problem(op, state, u=None, v=None):
    """The two-site problem at the state's center, with its environments built."""
    from kuiva.dmrg.sweep import EnvironmentCache, _LocalProblem

    u = state.center if u is None else u
    v = sorted(state.graph.neighbors(u))[0] if v is None else v
    return _LocalProblem(op, state, EnvironmentCache(op, state), u, v)


def _record_dots(fn):
    """Run ``fn`` with every ``_Lab.dot`` output recorded: ``(tensor, labels)`` pairs."""
    from kuiva.dmrg import sweep as sweep_mod

    real = []
    original = sweep_mod._Lab.dot

    def recording(self, other, pairs):
        out = original(self, other, pairs)
        if isinstance(out.t, BlockTensor):
            real.append((out.t, list(out.labels)))
        return out

    sweep_mod._Lab.dot = recording
    try:
        fn()
    finally:
        sweep_mod._Lab.dot = original
    return real


def _branching_case(seed):
    """A tree with a degree-3 node and two modes per node, fat enough that folding an
    environment into its W tensor is the smaller order at a bond next to that node."""
    contents = [list(range(i, i + 2)) for i in range(0, 12, 2)]
    graph = NetworkGraph(6, [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5)], contents)
    h, eri = random_spinor_integrals(12, seed=seed)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = random_state(op, 6, 16, n_roots=2, rng=np.random.default_rng(seed))
    return op, state


@pytest.mark.parametrize("case", ["path", "tree"])
def test_block_shape_reproduces_every_contraction_a_sweep_performs(case):
    """⚠ The pin that keeps the two descriptions of a contraction from drifting.

    ``BlockShape`` propagates sector tables so a chain can be sized before it is
    allocated; ``tensordot`` derives the same table fused into its pair-table loop. If
    they ever disagree the memory plan silently sizes a different calculation, so this
    asserts them equal — table and byte count — on every contraction of a real
    effective-Hamiltonian application, not on a constructed example. On the tree the
    application takes a pre-contracted half, so its build is pinned the same way.
    """
    from kuiva.dmrg.block import BlockShape

    if case == "path":
        h, eri = kramers_spinor_integrals(3, seed=40)
        op = compile_ttno(NetworkGraph.path(6), hamiltonian_product_terms(h, eri))
        state = random_state(op, 3, 8, n_roots=2, rng=np.random.default_rng(40))
        problem = _local_problem(op, state)
    else:
        op, state = _branching_case(40)
        problem = _local_problem(op, state, u=0, v=1)
        assert any(problem.order.pre.values())        # the case exists to take a half

    built = _record_dots(problem.prepare)
    predicted = problem.half_build_bytes()
    assert len(predicted) == len(built)
    for shape_bytes, (tensor, _) in zip(predicted, built):
        assert shape_bytes == tensor.nbytes

    vec = np.zeros(problem.dim, dtype=np.complex128)
    vec[0] = 1.0
    real = _record_dots(lambda: problem.apply(vec))
    predicted = problem.intermediate_bytes()
    assert len(predicted) == len(real)
    for shape_bytes, (tensor, _) in zip(predicted, real):
        assert shape_bytes == tensor.nbytes
    # and the starting structure itself: what the chain is walked from is exactly what
    # ``unpack`` produces for any vector
    assert BlockShape.of(problem.template).nbytes == problem.template.nbytes


def _open_operator_legs(labels):
    return sum(1 for lab in labels if lab[0] in ("op", "op_out"))


def test_one_operator_leg_is_open_at_a_time_on_a_path():
    """⚠ The mechanism behind the layer's largest array, pinned as a property of the chain
    rather than as a byte count: an intermediate carrying two operator legs costs the
    *product* of two operator bond dimensions (4.3 GB at D = 4 on a 20-spinor network),
    and the reorder — environments before W on the first side, W before environments on
    the second — leaves exactly one open at every step of a path. No half is taken on a
    path: a single-branch side has one leg open already, and a half there would only
    trade a sparse product for a dense one.
    """
    h, eri = kramers_spinor_integrals(4, seed=46)
    op = compile_ttno(NetworkGraph.path(8), hamiltonian_product_terms(h, eri))
    state = random_state(op, 3, 8, n_roots=2, rng=np.random.default_rng(46))
    problem = _local_problem(op, state)
    assert not any(problem.order.pre.values())
    assert problem.halves_gb() == 0.0
    assert len(problem.candidates) == 1                     # a rule, not a search
    vec = np.zeros(problem.dim, dtype=np.complex128)
    vec[0] = 1.0
    steps = _record_dots(lambda: problem.apply(vec))
    assert len(steps) == len(problem.branches_u) + len(problem.branches_v) + 2
    for _tensor, labels in steps:
        assert _open_operator_legs(labels) <= 1, labels


def test_a_branching_node_takes_the_smallest_order_and_every_order_agrees():
    """The tree case is a search: every candidate order is sized over structure, the
    smallest peak is taken, and the choice changes nothing but rounding — the product
    under the chosen order equals the product under the naive one (first side ``u``,
    nothing folded) to the tolerance a different summation order allows.
    """
    from kuiva.dmrg.sweep import EnvironmentCache, _LocalProblem

    op, state = _branching_case(47)
    cache = EnvironmentCache(op, state)
    problem = _LocalProblem(op, state, cache, 0, 1)
    peaks = {repr(c[0]): c[1] for c in problem.candidates}
    assert len(problem.candidates) == 8                     # 2 sides x 2^2 subsets at node 1
    chosen = peaks[repr(problem.order)]
    assert chosen == min(peaks.values())
    naive = [c for c in problem.candidates
             if c[0].first == 0 and not any(c[0].pre.values())][0]
    assert naive[1] > 2 * chosen                             # the search bought something
    assert problem.halves_gb() > 0.0
    assert problem.solve_workspace_gb(2, 2) > problem.halves_gb()
    assert problem.apply_peak_gb() >= max(problem.intermediate_bytes()) / 1024.0 ** 3

    other = _LocalProblem(op, state, cache, 0, 1)
    other.order = naive[0]                                   # before prepare(): no halves yet
    other.prepare()
    assert other.halves_gb() == 0.0
    rng = np.random.default_rng(47)
    vec = rng.standard_normal(problem.dim) + 1j * rng.standard_normal(problem.dim)
    a, b = problem.apply(vec), other.apply(vec)
    assert np.linalg.norm(a - b) <= 1e-13 * np.linalg.norm(b)


def test_the_diagonal_is_sized_and_dropped_past_the_cap(monkeypatch):
    """The preconditioner's dense W arrays are sized from the slices (pinned against the
    arrays actually built) and, past the kernels' transient budget, dropped rather than
    built — Davidson then runs unpreconditioned and still converges. On a degree-4 node
    that array was 2.77 GB outside every plan, and it killed a run planned at 1.0 GB.
    """
    from kuiva.dmrg import sweep as sweep_mod
    from kuiva.util import resources as res

    op, state = _branching_case(48)
    problem = _local_problem(op, state, u=0, v=1)
    sizes = [sweep_mod._w_diag_bytes(w) for w in (problem.w_u, problem.w_v)]
    built = [sweep_mod._w_diag(w) for w in (problem.w_u, problem.w_v)]
    for b, wd in zip(sizes, built):
        assert (wd is None and b == 0) or wd.a.nbytes == b
    assert problem.diagonal_gb() == sum(sizes) / 1024.0 ** 3
    exact = problem.diagonal()
    assert np.any(exact != 0.0)
    # a budget too small for either side: the diagonal is zero and costs nothing
    monkeypatch.setattr(res.BUDGET, "transient_gb", lambda **kw: 1e-9)
    assert problem.diagonal_gb() == 0.0
    assert not np.any(problem.diagonal())
    assert problem.transient_peak_gb() == problem.apply_peak_gb()
    h, eri = random_spinor_integrals(6, seed=48)
    result = solve(NetworkGraph.path(6), h, eri, 2, n_roots=2, max_bond=32)
    assert result.converged
    assert np.max(np.abs(result.energies - exact_energies(6, 2, h, eri, 2))) < E_TOL


def test_environment_builds_fold_by_structure_and_are_sized():
    """⚠ The fourth chain with the same defect: an environment build opened one operator
    leg per neighbour of the node before its W closed them — 3.8 GB at D = 2 on a
    degree-4 node while the plan, sizing only the output, said 1.2 GB. The build now
    folds environments into the W by a structural search; this pins that the fold chosen
    at the branching node is the smallest candidate and beats the unfolded order, that the
    chain over structure reports the real byte counts step for step, and that the plan's
    build transient covers what the real cache builds.
    """
    from kuiva.dmrg import sweep as sweep_mod
    from kuiva.dmrg.block import BlockShape
    from kuiva.dmrg.sweep import (EnvironmentCache, _Lab, _fold_peak, choose_fold,
                                  environment_build_peak_gb, renormalize)

    op, state = _branching_case(49)
    cache = EnvironmentCache(op, state)
    u, v = 1, 0                                   # node 1 has neighbours 0, 2, 3
    assert state.center not in state.graph.subtree_nodes(1, 0)
    nbrs = sorted(state.graph.neighbors(u))
    a = state.tensors[u]
    ket = _Lab(a, [("b", u, x) for x in nbrs] + [("p", u)])
    bra = _Lab(a.conj(), [("cb", u, x) for x in nbrs] + [("cp",)])
    envs = {x: _Lab(cache.get(x, u), [("bra", x), ("op", x), ("ket", x)])
            for x in nbrs if x != v}
    w = sweep_mod._w_lab(op, u, v)
    pre = choose_fold(ket, envs, w, bra, u)
    shape = lambda lab: _Lab(BlockShape.of(lab.t), lab.labels)      # noqa: E731
    sh = (shape(ket), {x: shape(e) for x, e in envs.items()}, shape(w), shape(bra))
    chosen = _fold_peak(*sh, u, pre)
    naive = _fold_peak(*sh, u, ())
    assert chosen <= naive
    assert chosen == min(_fold_peak(*sh, u, c) for c in [(), (2,), (3,), (2, 3)])
    # the same chain over structure and over data, step for step
    sizes = []
    renormalize(*sh, u, pre, sizes=sizes)
    real = _record_dots(lambda: renormalize(ket, envs, w, bra, u, pre))
    assert sizes == [t.nbytes for t, _ in real]
    # and the plan's transient is at least the peak of the build the real cache performs
    peak = max(a_ + b_ for a_, b_ in zip([a.nbytes] + sizes[len(pre):-1], sizes[len(pre):]))
    assert environment_build_peak_gb(op, state) * 1024.0 ** 3 >= min(peak, chosen)


def test_shape_environments_match_the_real_environment_cache():
    """The environment sizes the plan quotes are the ones the cache actually holds."""
    from kuiva.dmrg.sweep import (EnvironmentCache, ShapeEnvironments, environment_gb,
                                  shape_state)

    n, k = 6, 3
    h, eri = kramers_spinor_integrals(3, seed=41)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(op, k, 8, n_roots=2, rng=np.random.default_rng(41))
    real = EnvironmentCache(op, state)
    shapes = ShapeEnvironments(op, shape_state(state))
    checked = 0
    for a, b in state.graph.edges:
        for u, v in ((a, b), (b, a)):
            if state.center in state.graph.subtree_nodes(u, v):
                continue                    # not a well-defined environment at this center
            assert np.array_equal(real.get(u, v).sectors, shapes.get(u, v).sectors)
            assert real.get(u, v).nbytes == shapes.get(u, v).nbytes
            checked += 1
    assert checked > 0
    assert environment_gb(op, state) > 0.0


def test_two_site_requirement_covers_the_application_transient():
    """⚠ The regression this whole plan exists for: the ``require`` before a two-site
    solve must cover the effective-Hamiltonian application's intermediate, which is a
    *transient* no ledger entry ever saw. Before it did, a 20-spinor network held 10.8 GB
    against a 0.13 GB declaration and the run was killed by the kernel with no message.

    The pin is that the requirement exceeds the largest intermediate the chain builds —
    the term that was missing — and not merely the packed vectors it used to count.
    """
    n, k = 6, 3
    h, eri = kramers_spinor_integrals(3, seed=42)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(op, k, 8, n_roots=2, rng=np.random.default_rng(42))
    problem = _local_problem(op, state)
    largest = max(problem.intermediate_bytes()) / 1024.0 ** 3
    assert problem.apply_peak_gb() >= largest
    # the old estimate, kept here as the thing that must no longer bound the requirement
    old = (2 + 2) * problem.dim * 16.0 / 1024.0 ** 3
    assert problem.solve_workspace_gb(2, 2) + problem.apply_peak_gb() > old


def test_a_network_solve_is_refused_rather_than_killed(kuiva_caplog):
    """A limit the sweep cannot fit refuses *before* the first bond, naming the knob."""
    from kuiva.util import resources as res

    n, k = 6, 3
    h, eri = kramers_spinor_integrals(3, seed=43)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(op, k, 8, n_roots=2, rng=np.random.default_rng(43))
    budget = res.MemoryBudget()
    budget.configure(res.ResourceLimits(memory_gb=1e-9, source="test"))
    saved, res.BUDGET = res.BUDGET, budget
    try:
        with pytest.raises(res.MemoryLimitError) as excinfo:
            solve_ttn(op, state, max_bond=8, n_elec=k, boundary_check=0, max_sweeps=1)
    finally:
        res.BUDGET = saved
    assert "max_bond" in str(excinfo.value)


def test_the_per_bond_requirement_refuses_even_with_the_plan_off():
    """The backstop, and the thing that makes the guarantee hard: bond dimensions are
    re-derived by every truncation, so the solve that does not fit need not be the one the
    plan looked at. With the plan switched off the sweep must still refuse at the bond."""
    from kuiva.util import resources as res

    # Two modes per node: with one operator leg open at a time the two-site solve of a
    # one-mode node is smaller than an environment, and this test needs a bond where the
    # solve is the largest thing (the merged dimension grows as d^2, the environment as D^2).
    n, k = 8, 4
    h, eri = kramers_spinor_integrals(4, seed=45)
    graph = NetworkGraph.path(4, [[0, 1], [2, 3], [4, 5], [6, 7]])
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = random_state(op, k, 8, n_roots=2, rng=np.random.default_rng(45))
    # A limit that holds the environments a bond needs and not the two-site solve, so the
    # refusal has to come from the requirement under test rather than from an earlier one.
    from kuiva.dmrg.sweep import ShapeEnvironments, shape_state
    problem = _local_problem(op, state)
    need = problem.solve_workspace_gb(2, 2) + problem.apply_peak_gb()
    shapes = ShapeEnvironments(op, shape_state(state, fill_center=True))
    biggest_env = max(shapes.get(a, b).nbytes for a, b in
                      list(state.graph.edges) + [(b, a) for a, b in state.graph.edges])
    limit = 0.9 * need
    assert limit > biggest_env / 1024.0 ** 3            # an environment still fits
    budget = res.MemoryBudget()
    budget.configure(res.ResourceLimits(memory_gb=limit, source="test"))
    saved, res.BUDGET = res.BUDGET, budget
    try:
        with pytest.raises(res.MemoryLimitError) as excinfo:
            solve_ttn(op, state, max_bond=8, n_elec=k, boundary_check=0, max_sweeps=1,
                      memory_plan=False)
    finally:
        res.BUDGET = saved
    assert "two-site problem" in str(excinfo.value)


def test_the_memory_plan_can_be_switched_off_without_changing_the_answer():
    """``memory_plan=False`` is the driver's "I planned once per chart" switch and must
    move no number: the exact per-bond requirement inside the sweep is what refuses."""
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=44)
    a = solve(NetworkGraph.path(n), h, eri, k, n_roots=2)
    b = solve(NetworkGraph.path(n), h, eri, k, n_roots=2, memory_plan=False)
    assert np.array_equal(a.energies, b.energies)
