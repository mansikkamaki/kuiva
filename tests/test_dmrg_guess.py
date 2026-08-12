"""Tier-0 tests for the cheap-CI seeding of :mod:`kuiva.dmrg.guess`.

The conversion oracle is :func:`kuiva.dmrg.sweep.state_to_dense` against the CI vector in
the mode-ascending kron basis: untruncated, the determinant-expansion -> TTN conversion
must reproduce every root **exactly** (up to normalization and a global phase), on paths
and on trees, and a seeded solve must start already converged when the seed spans the
exact eigenvector. The topology guess is asserted on its contract: always a tree, a
Fiedler chain for one cluster, and a site-separating tree for block-structured mutual
information.
"""
import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix
from kuiva.dmrg.block import QuantumNumber
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.guess import expansion_to_ttn, topology_from_mutual_information
from kuiva.dmrg.sweep import solve_ttn, state_to_dense
from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms

from test_ci_strings import random_spinor_integrals

E_TOL = 1e-8


def full_ci(n, k, h, eri, n_roots):
    from itertools import combinations
    masks = np.sort(np.array([sum(1 << i for i in c)
                              for c in combinations(range(n), k)], dtype=np.uint64))
    dets = Determinants(masks, n_spinor=n, n_elec=k)
    ham = hamiltonian_matrix(dets, h, eri).toarray()
    w, v = np.linalg.eigh(ham)
    return masks, w[:n_roots], v[:, :n_roots]


def kron_index(mask, n):
    """Determinant bitmask -> mode-ascending kron index (first mode slowest)."""
    return sum(((int(mask) >> m) & 1) << (n - 1 - m) for m in range(n))


def dense_from_expansion(masks, coeff, n):
    vec = np.zeros(2 ** n, dtype=np.complex128)
    for m, c in zip(masks, coeff):
        vec[kron_index(m, n)] = c
    return vec


# --- the conversion is exact --------------------------------------------------------------

@pytest.mark.parametrize("graph", [
    NetworkGraph.path(6),
    NetworkGraph(6, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]),
])
def test_expansion_to_ttn_is_exact_within_the_ci_space(graph):
    n, k, n_roots = 6, 2, 3
    h, eri = random_spinor_integrals(n, seed=31)
    masks, energies, vecs = full_ci(n, k, h, eri, n_roots)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = expansion_to_ttn(op, masks, vecs)
    assert state.n_roots == n_roots
    assert state.charge == QuantumNumber(k)
    for r in range(n_roots):
        got = state_to_dense(state, op, root=r)
        ref = dense_from_expansion(masks, vecs[:, r], n)
        overlap = abs(np.vdot(got, ref))
        assert overlap == pytest.approx(1.0, abs=1e-10)


def test_expansion_from_a_truncated_list_is_exact_within_it():
    """A selected-CI-like sparse list: exact within its own span (exact within its own span)."""
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=32)
    masks, _, vecs = full_ci(n, k, h, eri, 1)
    keep = np.argsort(-np.abs(vecs[:, 0]))[:7]           # a ragged 7-determinant subset
    sub_masks = masks[keep]
    sub_vec = vecs[keep, 0] / np.linalg.norm(vecs[keep, 0])
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = expansion_to_ttn(op, sub_masks, sub_vec)
    got = state_to_dense(state, op)
    ref = dense_from_expansion(sub_masks, sub_vec, n)
    assert abs(np.vdot(got, ref)) == pytest.approx(1.0, abs=1e-10)


def test_multi_mode_nodes_convert_exactly():
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=33)
    graph = NetworkGraph(3, [(0, 1), (1, 2)], contents=[(0, 1), (2, 3), (4, 5)])
    masks, _, vecs = full_ci(n, k, h, eri, 2)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = expansion_to_ttn(op, masks, vecs)
    for r in range(2):
        got = state_to_dense(state, op, root=r)
        ref = dense_from_expansion(masks, vecs[:, r], n)
        assert abs(np.vdot(got, ref)) == pytest.approx(1.0, abs=1e-10)


def test_seeded_state_is_canonical_and_solves_immediately():
    """Seeding from the exact eigenvectors: the very first sweep must already sit at the
    exact energies (the seeding payoff, in its sharpest form)."""
    n, k, n_roots = 6, 2, 2
    h, eri = random_spinor_integrals(n, seed=34)
    masks, energies, vecs = full_ci(n, k, h, eri, n_roots)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = expansion_to_ttn(op, masks, vecs)
    # canonical: isometries carry charge 0, centers the total N, unit norm
    for t in state.tensors:
        if t is not None:
            assert t.charge == QuantumNumber(0)
    for c in state.centers:
        assert c.norm() == pytest.approx(1.0, abs=1e-10)
    result = solve_ttn(op, state, max_bond=64, boundary_check=0)
    assert np.max(np.abs(result.energies - energies)) < E_TOL
    assert result.n_sweeps <= 2                          # started converged
    assert abs(result.history[0] - float(np.mean(energies))) < 1e-7


def test_seed_truncation_is_reported_and_capped():
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=35)
    masks, _, vecs = full_ci(n, k, h, eri, 1)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = expansion_to_ttn(op, masks, vecs, max_bond=3)
    dims = state.bond_dimensions()
    assert max(dims.values()) <= 3
    # still a valid normalized state
    for c in state.centers:
        assert c.norm() == pytest.approx(1.0, abs=1e-10)


# --- the topology guess -------------------------------------------------------------------

def test_single_cluster_guess_is_a_fiedler_chain():
    rng = np.random.default_rng(36)
    n = 6
    # chain-structured MI, presented in a scrambled order
    order = rng.permutation(n)
    info = np.zeros((n, n))
    for a in range(n - 1):
        info[order[a], order[a + 1]] = info[order[a + 1], order[a]] = 1.0 - 0.01 * a
    guess = topology_from_mutual_information(info, site_split=0.05)
    assert len(guess.sites) == 1
    degrees = [guess.graph.degree(u) for u in range(n)]
    assert max(degrees) <= 2                             # a chain
    # strongly linked pairs sit adjacent
    edges = set(guess.graph.edges)
    for a in range(n - 1):
        pair = (min(order[a], order[a + 1]), max(order[a], order[a + 1]))
        assert pair in edges


def test_two_site_guess_separates_the_blocks():
    n = 6
    a_members, b_members = (0, 2, 4), (1, 3, 5)          # interleaved on purpose
    info = np.zeros((n, n))
    for grp in (a_members, b_members):
        for i in grp:
            for j in grp:
                if i != j:
                    info[i, j] = 1.0
    info[4, 5] = info[5, 4] = 0.05                       # weak inter-site link
    guess = topology_from_mutual_information(info, site_split=0.25)
    assert sorted(map(tuple, guess.sites)) == [tuple(a_members), tuple(b_members)]
    # exactly one inter-site edge, at the strongest cross pair
    cross = [e for e in guess.graph.edges
             if (e[0] in a_members) != (e[1] in a_members)]
    assert cross == [(4, 5)]


def test_guess_is_always_a_tree():
    rng = np.random.default_rng(37)
    for n in (2, 3, 5, 9):
        info = np.abs(rng.standard_normal((n, n)))
        info = info + info.T
        np.fill_diagonal(info, 0.0)
        guess = topology_from_mutual_information(info)
        assert guess.graph.n_nodes == n                  # NetworkGraph validates tree-ness
        assert sorted(x for c in guess.graph.contents for x in c) == list(range(n))
