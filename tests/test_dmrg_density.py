"""Tier-0 tests for the network RDMs and the TTNO template.

Three independent oracles pin the same objects:

* the **CI-exact** path (:class:`kuiva.rdm.rdm.RDMBuilder` through ``FullCISolver``) for
  ranks 1-2 — the parity the whole stage exists to establish;
* the **dense Jordan-Wigner Fock oracle** (plain kron, built here, independent of the
  compiler's fermion machinery) for ranks 1-4;
* the **energy closure** ``E = sum h gamma + 1/2 sum eri Gamma``, which fails on any
  phase, transposition or convention error in either factor.

⚠ Every test runs on genuinely **complex** integrals: the conjugation trap is
invisible for real density matrices, and a transposed gamma passes every Hermiticity and
trace check.
"""
import numpy as np
import pytest

from kuiva.dmrg.block import BlockTensor
from kuiva.dmrg.density import (annihilation_term, network_rdm, network_rdms,
                                node_environments)
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.sparse import sparse_w_gb
from kuiva.dmrg.sweep import random_state, solve_ttn, state_to_dense
from kuiva.dmrg.ttno import TTNOTemplate, compile_ttno, hamiltonian_product_terms
from kuiva.mcscf.casci import FullCISolver
from kuiva.rdm.rdm import active_space_energy

from test_ci_strings import random_spinor_integrals
from test_ci_davidson import kramers_spinor_integrals

E_TOL = 1e-8            # energies to 1e-8 Eh at saturated bond dimension
RDM_TOL = 1e-9          # RDM elements at Davidson tolerance 1e-8 on toy systems

TREE = NetworkGraph(6, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])


def solved(n, k, h, eri, n_roots=1, graph=None, seed=0, **kw):
    graph = NetworkGraph.path(n) if graph is None else graph
    tpl = TTNOTemplate(graph)
    op = tpl.fill(h, eri)
    state = random_state(op, k, 200, n_roots=n_roots,
                         rng=np.random.default_rng(seed))
    result = solve_ttn(op, state, max_bond=200, boundary_check=0, report=False, **kw)
    assert result.converged
    return tpl, op, state, result


# --- the TTNO template ---------------------------------------------------------------------

def test_template_fill_equals_direct_compile_and_refills():
    n = 5
    g = NetworkGraph(n, [(0, 1), (1, 2), (2, 3), (2, 4)])
    tpl = TTNOTemplate(g)
    for seed in (3, 7):
        h, eri = random_spinor_integrals(n, seed=seed)
        filled = tpl.fill(h, eri).to_dense(max_dim=100)
        direct = compile_ttno(g, hamiltonian_product_terms(h, eri)).to_dense(max_dim=100)
        assert np.max(np.abs(filled - direct)) < 1e-12


def test_template_refuses_wrong_sized_integrals():
    tpl = TTNOTemplate(NetworkGraph.path(4))
    h, eri = random_spinor_integrals(5, seed=1)
    with pytest.raises(ValueError, match="must be"):
        tpl.fill(h, eri)


def test_node_reservations_cover_the_compiled_tensors_and_their_caches():
    # the per-node reserve is the exact tensor size times the number of contraction
    # patterns its CSR cache can hold — an *upper bound* on residency by construction, and
    # never below the tensors themselves. Both sides are asserted: a reservation that
    # stopped covering the caches, and one that started padding, each fail here.
    h, eri = random_spinor_integrals(5, seed=2)
    op = compile_ttno(NetworkGraph.path(5), hamiltonian_product_terms(h, eri))
    reserved = sum(a.gb for a in op.allocations)
    exact = sum(sparse_w_gb(w.spaces, w.nnz, w.nblocks) for w in op.tensors)
    assert exact == pytest.approx(op.nbytes / 1024.0 ** 3, rel=0, abs=0)
    patterns = sum((1 + w.ndim - 2 + (1 if u == op.root else 0))
                   * sparse_w_gb(w.spaces, w.nnz, w.nblocks)
                   for u, w in enumerate(op.tensors))
    assert reserved == pytest.approx(patterns, rel=0, abs=0)
    assert reserved > exact                        # the caches are genuinely accounted


# --- ranks 1-2: the production path against the CI-exact path ------------------------------

@pytest.mark.parametrize("graph", [None, TREE], ids=["path", "tree"])
def test_network_rdms_match_ci_exact_rdms(graph):
    n, k, n_roots = 6, 2, 3
    h, eri = random_spinor_integrals(n, seed=11)
    tpl, op, state, result = solved(n, k, h, eri, n_roots=n_roots, graph=graph)
    ci = FullCISolver(n, k, n_states=n_roots, enforce_kramers=False)
    ref = ci.solve_active(h, eri)
    assert np.max(np.abs(result.energies - ref.energies)) < E_TOL

    gamma, gamma2 = network_rdms(tpl, state, energies=result.energies, n_elec=k,
                                 enforce_kramers=False, ttno=op)
    assert np.max(np.abs(gamma - ref.gamma)) < RDM_TOL
    assert np.max(np.abs(gamma2 - ref.gamma2)) < RDM_TOL
    # closure: the same integrals contracted with the RDMs reproduce the SA energy
    e_net = active_space_energy(h, eri, gamma, gamma2)
    assert abs(e_net - float(np.dot(result.weights, result.energies))) < E_TOL


def test_state_rdms_match_ci_per_root():
    """Per-root (gamma, Gamma) against the CI-exact per-state RDMs, elementwise.

    A random spectrum is non-degenerate, so each eigenvector is unique up to a phase and
    a same-state density is phase-invariant — elementwise comparison is legitimate.
    """
    from kuiva.rdm.rdm import RDMBuilder

    n, k, n_roots = 6, 2, 3
    h, eri = random_spinor_integrals(n, seed=15)
    tpl, op, state, result = solved(n, k, h, eri, n_roots=n_roots)
    ci = FullCISolver(n, k, n_states=n_roots, enforce_kramers=False)
    ref = ci.solve_active(h, eri)
    assert np.max(np.abs(result.energies - ref.energies)) < E_TOL

    from kuiva.dmrg.density import state_rdms
    builder = RDMBuilder(ci.space)
    for r, (gamma, gamma2) in enumerate(state_rdms(tpl, state)):
        g_ci, g2_ci = builder(ref.vectors[r], enforce_kramers=False)
        assert np.max(np.abs(gamma - g_ci)) < RDM_TOL, "root {}".format(r)
        assert np.max(np.abs(gamma2 - g2_ci)) < RDM_TOL, "root {}".format(r)


@pytest.mark.parametrize("graph", [None, TREE], ids=["path", "tree"])
def test_transition_rdm1s_match_ci(graph):
    """gamma^{IJ} against the CI's transition densities, through the phase freedom.

    Each root carries an arbitrary phase on both routes, so an off-diagonal block is
    compared through |gamma^{IJ}_pq| elementwise (one global phase per (I, J) pair);
    the diagonal blocks are phase-invariant and compared directly. Hermiticity
    gamma^{IJ}_pq = conj(gamma^{JI}_qp) is asserted exactly as a structure check.
    """
    from kuiva.dmrg.density import transition_rdm1s

    n, k, n_roots = 6, 2, 3
    h, eri = random_spinor_integrals(n, seed=16)
    tpl, op, state, result = solved(n, k, h, eri, n_roots=n_roots, graph=graph)
    ci = FullCISolver(n, k, n_states=n_roots, enforce_kramers=False)
    ref = ci.solve_active(h, eri)
    assert np.max(np.abs(result.energies - ref.energies)) < E_TOL

    tdm_net = transition_rdm1s(op, state)
    tdm_ci = ci.transition_densities(ref.vectors)
    assert tdm_net.shape == tdm_ci.shape == (n_roots, n_roots, n, n)
    assert np.max(np.abs(tdm_net - np.conj(tdm_net.transpose(1, 0, 3, 2)))) < RDM_TOL
    for i in range(n_roots):
        assert np.max(np.abs(tdm_net[i, i] - tdm_ci[i, i])) < RDM_TOL
    assert np.max(np.abs(np.abs(tdm_net) - np.abs(tdm_ci))) < RDM_TOL


def test_transition_diagonal_reproduces_the_production_rdm1():
    # the diagonal blocks of the applied-string route against the environments route:
    # two contractions sharing no intermediate
    from kuiva.dmrg.density import state_rdms, transition_rdm1s

    n, k, n_roots = 6, 2, 2
    h, eri = random_spinor_integrals(n, seed=17)
    tpl, op, state, result = solved(n, k, h, eri, n_roots=n_roots)
    tdm = transition_rdm1s(op, state)
    for r, (gamma, _) in enumerate(state_rdms(tpl, state)):
        assert np.max(np.abs(tdm[r, r] - gamma)) < 1e-10


def test_extraction_is_fill_independent():
    # the channels the template reads are label-determined and unit, so the extracted
    # RDMs cannot depend on which coefficients the environment TTNO was filled with
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=12)
    tpl, op, state, result = solved(n, k, h, eri, n_roots=2)
    kw = dict(energies=result.energies, n_elec=k, enforce_kramers=False)
    g_a, g2_a = network_rdms(tpl, state, ttno=op, **kw)
    g_b, g2_b = network_rdms(tpl, state, **kw)          # the template's coeff=1 fill
    assert np.max(np.abs(g_a - g_b)) == 0.0
    assert np.max(np.abs(g2_a - g2_b)) == 0.0


def test_every_node_environment_closes_the_energy():
    # sum_entries W_u[e] G_u[e] = <H> for EVERY node u — the identity dE/dW rests on
    n, k = 5, 2
    h, eri = random_spinor_integrals(n, seed=13)
    tpl, op, state, result = solved(n, k, h, eri, graph=NetworkGraph(
        5, [(0, 1), (1, 2), (1, 3), (3, 4)]))
    envs = node_environments(op, state, [1.0])
    e_sa = float(result.energies[0])
    for u, (w, g) in enumerate(zip(op.tensors, envs)):
        total = 0.0 + 0.0j
        for b, row in enumerate(w.sectors):
            gb = g.find(row)
            if gb is not None:
                idx, val = w.block_entries(b)          # sparse W: entries, not a block
                total += np.sum(val * gb.ravel()[idx])
        assert abs(total - e_sa) < 1e-9, "node {}".format(u)


def test_node_environment_chain_is_sized_by_its_own_structure():
    """⚠ The pin for the RDM contraction's memory: ``_node_environment`` walked over
    ``BlockShape`` operands reports, step for step, the byte counts of the tensors the
    same chain builds over data — on a tree with a degree-3 node and a state-averaged
    center, so the ensemble leg and the branching case are both covered. The chain used
    to open every neighbour's operator leg before closing a single bond leg, and that
    intermediate — times the root count at the center — killed a 20-spinor ladder at
    D = 8 under a 0.7 GB plan; the sizing is what lets the plan refuse it instead.
    """
    from kuiva.dmrg import density as dens
    from kuiva.dmrg import sweep as sweep_mod
    from kuiva.dmrg.block import BlockShape
    from kuiva.dmrg.sweep import EnvironmentCache, _stack_roots

    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=21)
    graph = NetworkGraph(6, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    tpl, op, state, result = solved(n, k, h, eri, graph=graph, n_roots=2)
    w = np.array([0.5, 0.5])
    stacked = _stack_roots(state.centers, w)
    cache = EnvironmentCache(op, state)
    # the centre-side messages, exactly as node_environments builds them
    parent, preorder = graph.parents(state.center)
    center_side = {int(x): int(parent[x]) for x in preorder[1:]}
    down = {}
    for x in [int(i) for i in preorder[1:]]:
        v = center_side[x]
        down[(v, x)] = dens._down_message(op, state, cache, down, center_side, stacked,
                                          v, x)
    checked = 0
    for u in range(graph.n_nodes):
        if u == state.center:
            continue
        nbrs = sorted(graph.neighbors(u))
        env_of = {x: (down[(x, u)] if x == center_side[u] else cache.get(x, u))
                  for x in nbrs}
        real = []
        original = sweep_mod._Lab.dot

        def recording(self, other, pairs):
            out = original(self, other, pairs)
            real.append(out.t.nbytes)
            return out

        sweep_mod._Lab.dot = recording
        try:
            g, _ = dens._node_environment(u, state.tensors[u], nbrs, env_of, False, op)
        finally:
            sweep_mod._Lab.dot = original
        shape, steps = dens._node_environment(
            u, BlockShape.of(state.tensors[u]), nbrs,
            {x: BlockShape.of(e) for x, e in env_of.items()}, False, op, sizes=[])
        assert steps == real
        assert shape.nbytes == g.nbytes
        checked += 1
    assert checked > 0
    # the ensemble center, through the stacked shape
    u = state.center
    nbrs = sorted(graph.neighbors(u))
    env_of = {x: cache.get(x, u) for x in nbrs}
    g, _ = dens._node_environment(u, stacked, nbrs, env_of, True, op)
    shape, steps = dens._node_environment(
        u, dens._stacked_shape(BlockShape.of(state.centers[0]), 2), nbrs,
        {x: BlockShape.of(e) for x, e in env_of.items()}, True, op, sizes=[])
    assert shape.nbytes == g.nbytes
    resident, transient = dens.node_environments_gb(op, state, 2)
    assert resident > 0.0 and transient > 0.0


def test_state_average_weights_are_imposed_where_rdms_are_built():
    # State-averaging behaviour: on a Kramers-degenerate spectrum, deliberately unequal requested
    # weights are equalized inside the degenerate pair — the RDMs cannot depend on the
    # arbitrary rotation the eigensolver chose within it
    h, eri = kramers_spinor_integrals(3, seed=5)        # 6 spinors, T-symmetric
    n, k, n_roots = 6, 3, 2
    tpl, op, state, result = solved(n, k, h, eri, n_roots=n_roots, seed=3)
    kw = dict(energies=result.energies, n_elec=k)
    g_skew, g2_skew = network_rdms(tpl, state, weights=[0.9, 0.1], ttno=op, **kw)
    g_flat, g2_flat = network_rdms(tpl, state, weights=[0.5, 0.5], ttno=op, **kw)
    assert np.max(np.abs(g_skew - g_flat)) < 1e-12
    assert np.max(np.abs(g2_skew - g2_flat)) < 1e-12


def test_state_averaged_rdms_require_energies():
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=14)
    tpl, op, state, result = solved(n, k, h, eri, n_roots=2)
    with pytest.raises(ValueError, match="energies"):
        network_rdms(tpl, state, n_elec=k)


def test_template_topology_mismatch_is_refused():
    n, k = 5, 2
    h, eri = random_spinor_integrals(n, seed=15)
    tpl, op, state, result = solved(n, k, h, eri)
    other = TTNOTemplate(NetworkGraph(5, [(0, 1), (1, 2), (1, 3), (3, 4)]))
    with pytest.raises(ValueError, match="topology"):
        network_rdms(other, state, energies=result.energies, n_elec=k,
                     enforce_kramers=False)


# --- ranks 1-4: direct contraction against the dense Jordan-Wigner oracle ------------------

def dense_annihilators(n):
    """``a_p`` in the global mode-ascending kron basis (mode 0 slowest), built by plain
    kron — independent of ``fermion_term`` and of the compiler."""
    i2 = np.eye(2)
    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    z = np.diag([1.0, -1.0])
    ops = []
    for p in range(n):
        m = np.eye(1)
        for j in range(n):
            m = np.kron(m, z if j < p else (a if j == p else i2))
        ops.append(m)
    return ops


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_network_rdm_matches_the_dense_fock_oracle(rank):
    n, k, n_roots = 6, 3, 2
    h, eri = random_spinor_integrals(n, seed=21)
    graph = NetworkGraph(6, [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)])
    tpl, op, state, result = solved(n, k, h, eri, n_roots=n_roots, graph=graph,
                                    seed=1, on_split="warn")
    gam = network_rdm(op, state, rank, enforce_kramers=False)

    a = dense_annihilators(n)
    ad = [x.conj().T for x in a]
    psis = [state_to_dense(state, op, root=r) for r in range(n_roots)]
    rng = np.random.default_rng(7)
    for _ in range(120):
        idx = rng.integers(0, n, size=2 * rank)
        ps, qs = idx[0::2], idx[1::2]
        opm = np.eye(2 ** n, dtype=complex)
        for p in ps:
            opm = opm @ ad[p]
        for q in qs[::-1]:
            opm = opm @ a[q]
        ref = np.mean([np.vdot(psi, opm @ psi) for psi in psis])
        assert abs(gam[tuple(idx)] - ref) < 1e-10


def test_gram_ranks_1_and_2_cross_check_the_production_path():
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=22)
    tpl, op, state, result = solved(n, k, h, eri, n_roots=2, graph=TREE)
    gamma, gamma2 = network_rdms(tpl, state, energies=result.energies, n_elec=k,
                                 enforce_kramers=False, ttno=op)
    g1 = network_rdm(op, state, 1, enforce_kramers=False)
    g2 = network_rdm(op, state, 2, enforce_kramers=False)
    # two entirely different contractions of the same state; agreement is to rounding
    assert np.max(np.abs(g1 - gamma)) < 1e-12
    assert np.max(np.abs(g2 - gamma2)) < 1e-12


def test_partial_traces_chain_down_the_ranks():
    # sum_t Gamma^(m)[..., t, t] = (N - m + 1) Gamma^(m-1) — the factor is wrong on any
    # ordering or sign error in the antisymmetry fill
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=23)
    tpl, op, state, result = solved(n, k, h, eri, seed=2, on_split="warn")
    g1 = network_rdm(op, state, 1)
    g2 = network_rdm(op, state, 2)
    g3 = network_rdm(op, state, 3)
    g4 = network_rdm(op, state, 4)
    assert np.max(np.abs(np.einsum("pqtt->pq", g2) - (k - 1) * g1)) < 1e-10
    assert np.max(np.abs(np.einsum("pqrstt->pqrs", g3) - (k - 2) * g2)) < 1e-10
    assert np.max(np.abs(np.einsum("pqrstuvv->pqrstu", g4) - (k - 3) * g3)) < 1e-10


def test_odd_rank_strings_carry_the_jordan_wigner_tail():
    # the tail is load-bearing (module docstring): an odd string truncated at its lowest
    # operator differs by a Z on every mode below it
    term = annihilation_term((3, 4))
    assert term.modes[0] >= 3                            # even rank: no tail
    term = annihilation_term((3,))
    assert term.modes == (0, 1, 2, 3)                    # odd rank: explicit tail
    z = np.diag([1.0, -1.0])
    for mat in term.mats[:3]:
        assert np.array_equal(mat, z)


def test_rank_out_of_range_is_refused():
    n, k = 4, 2
    h, eri = random_spinor_integrals(n, seed=24)
    tpl, op, state, result = solved(n, k, h, eri)
    with pytest.raises(ValueError, match="rank"):
        network_rdm(op, state, 5)
