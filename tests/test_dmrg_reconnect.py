"""Tier-0 tests for adaptive reconnection.

The sharp oracle is a Hamiltonian with **known** product structure: two non-interacting
fermionic fragments with *contiguous* mode labels, presented on a deliberately wrong seed
tree that interleaves them. Adaptive reconnection must (a) reach the exact energy, (b)
*discover* the fragment decomposition — the inter-fragment bond collapsing to an exact
product — and (c) refuse to move when started from the right topology (hysteresis: a tie
is a tie). Environment reuse across adoptions is validated by comparing an adopted run
against a from-scratch solve on the final topology.

⚠ Why the fragments carry contiguous labels: the Jordan-Wigner ordering is the global
mode index (a [FIRM] decision, `kuiva/dmrg/ttno.py`), so the network's kron basis
carries crossing parities between *interleaved* mode labels — a fermionic product state
over interleaved label sets is genuinely entangled in that representation, and no tree
move can undo it (a leaf swap moves a tree position, not a JW string). Mode-label choice
is therefore a seed-time decision; reconnection optimizes the *tree* at fixed
labels. The spin-model tests (no JW strings) have no such constraint and use interleaved
seeds freely.
"""
import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix
from kuiva.dmrg.block import QuantumNumber
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.reconnect import (ReconnectionPolicy, discovered_structure,
                                  solve_adaptive)
from kuiva.dmrg.sweep import random_state, solve_ttn
from kuiva.dmrg.ttno import (ModeBasis, ProductTerm, compile_ttno,
                             hamiltonian_product_terms)

from test_ci_strings import random_spinor_integrals

E_TOL = 1e-8


def exact_energies(n, k, h, eri, n_roots):
    from itertools import combinations
    masks = np.sort(np.array([sum(1 << i for i in c)
                              for c in combinations(range(n), k)], dtype=np.uint64))
    dets = Determinants(masks, n_spinor=n, n_elec=k)
    ham = hamiltonian_matrix(dets, h, eri).toarray()
    return np.sort(np.linalg.eigvalsh(ham))[:n_roots]


def two_fragments(u=6.0):
    """Two non-interacting 3-spinor fragments on modes (0, 1, 2) and (3, 4, 5).

    Each fragment is a complex-hopping triangle plus strong intra-fragment repulsion, so
    the 2-electron ground state is uniquely one delocalized electron per fragment: both
    fragments carry internal entanglement, the inter-fragment cut is an exact product,
    and the ground state is non-degenerate."""
    n = 6
    frag_a, frag_b = (0, 1, 2), (3, 4, 5)
    h = np.zeros((n, n), dtype=np.complex128)
    eri = np.zeros((n, n, n, n), dtype=np.complex128)
    for frag, eps in ((frag_a, 0.0), (frag_b, 0.05)):
        for i, p in enumerate(frag):
            h[p, p] = eps + 0.1 * i
            for q in frag:
                if p < q:
                    h[p, q] = -1.0 + 0.1j
                    h[q, p] = -1.0 - 0.1j
                if p != q:
                    eri[p, p, q, q] = u
    return n, h, eri, frag_a, frag_b


#: A wrong seed: the path tree whose *nodes* interleave the two fragments while the mode
#: labels stay contiguous (docstring: the JW constraint).
def wrong_seed_tree():
    return NetworkGraph(6, [(i, i + 1) for i in range(5)],
                        contents=[(0,), (3,), (1,), (4,), (2,), (5,)])


# --- (a) + (b): discovery of product structure from a wrong seed --------------------------

def test_product_structure_is_discovered_from_a_wrong_seed():
    n, h, eri, frag_a, frag_b = two_fragments()
    k = 2
    ref = exact_energies(n, k, h, eri, 1)
    result = solve_adaptive(hamiltonian_product_terms(h, eri), wrong_seed_tree(),
                            k, max_bond=16, policy=ReconnectionPolicy(rule="entropy"),
                            rng=np.random.default_rng(7), boundary_check=0)
    assert abs(result.energies[0] - ref[0]) < E_TOL
    assert len(result.moves) > 0                        # it actually moved something
    report = result.structure
    assert sorted(s.orbitals for s in report.sites) == [frag_a, frag_b]
    assert len(report.weak_edges) == 1
    inter = report.bonds[report.weak_edges[0]]
    assert inter.eff_dim == 1                           # non-interacting: exact product
    assert inter.entropy_nats < 1e-8
    for site in report.sites:
        assert site.internal_max_entropy > 0.1          # the sites are genuinely internal


def test_adaptive_matches_from_scratch_solve_on_final_topology():
    """Environment reuse across adoptions cannot change the answer: an adopted run and a
    fresh fixed-topology solve on the discovered tree must agree to the 1e-8 Eh tolerance."""
    n, h, eri, _, _ = two_fragments(u=4.0)
    k = 2
    terms = hamiltonian_product_terms(h, eri)
    result = solve_adaptive(terms, wrong_seed_tree(), k, max_bond=16,
                            policy=ReconnectionPolicy(rule="entropy"),
                            rng=np.random.default_rng(8), boundary_check=0,
                            report_structure=False)
    assert len(result.moves) > 0
    op = compile_ttno(result.graph, terms)
    fresh = solve_ttn(op, random_state(op, k, 32, rng=np.random.default_rng(9)),
                      max_bond=32, boundary_check=0)
    assert abs(result.energies[0] - fresh.energies[0]) < E_TOL


# --- (c): hysteresis and stability --------------------------------------------------------

def test_no_adoption_from_the_right_topology():
    """Started from the fragment-separating tree (the plain path over contiguous
    labels), every attempt must refuse — there is nothing to win."""
    n, h, eri, _, _ = two_fragments()
    result = solve_adaptive(hamiltonian_product_terms(h, eri), NetworkGraph.path(6),
                            2, max_bond=16, policy=ReconnectionPolicy(rule="entropy"),
                            rng=np.random.default_rng(10), boundary_check=0,
                            report_structure=False)
    assert result.moves == []
    assert result.converged


def test_adaptive_never_ends_above_fixed_topology():
    """Stage-5 validation (c): on a generic (non-product) Hamiltonian, adaptive must not
    end above the fixed-good-topology energy."""
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=46)
    terms = hamiltonian_product_terms(h, eri)
    graph = NetworkGraph.path(n)
    op = compile_ttno(graph, terms)
    fixed = solve_ttn(op, random_state(op, k, 64, rng=np.random.default_rng(11)),
                      max_bond=64, boundary_check=0)
    adaptive = solve_adaptive(terms, graph, k, max_bond=64,
                              rng=np.random.default_rng(12), boundary_check=0,
                              report_structure=False)
    assert adaptive.energies[0] <= fixed.energies[0] + E_TOL


def test_a_huge_margin_freezes_the_topology():
    n, h, eri, _, _ = two_fragments()
    result = solve_adaptive(hamiltonian_product_terms(h, eri), wrong_seed_tree(),
                            2, max_bond=16,
                            policy=ReconnectionPolicy(rule="entropy", abs_floor=1e9),
                            rng=np.random.default_rng(13), boundary_check=0,
                            report_structure=False)
    assert result.moves == []


# --- the model seam: structure tests with no integrals ----------------------------

def heisenberg_terms(bonds):
    sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    sm = sp.T.copy()
    terms = []
    for i, k, j in bonds:
        i, k = min(i, k), max(i, k)
        terms.append(ProductTerm(0.5 * j, (i, k), (sp, sm)))
        terms.append(ProductTerm(0.5 * j, (i, k), (sm, sp)))
        terms.append(ProductTerm(j, (i, k), (sz, sz)))
    return terms, ModeBasis(2, (QuantumNumber(0), QuantumNumber(1)))


def test_singlet_pairs_pop_out_as_sites():
    """Three uncoupled Heisenberg dimers, interleaved on the seed chain (spins carry no
    JW strings, so label interleaving is fair game here): the unique ground state is a
    product of three singlets, and the report must find exactly the three pairs."""
    pairs = ((0, 1), (2, 3), (4, 5))
    js = (1.0, 1.3, 0.7)
    terms, basis = heisenberg_terms([(a, b, j) for (a, b), j in zip(pairs, js)])
    seed = NetworkGraph(6, [(i, i + 1) for i in range(5)],
                        contents=[(0,), (2,), (4,), (1,), (3,), (5,)])
    # on_split="warn": the state-averaging gate reads N as an electron count, but here N counts up
    # spins and a "3-electron" Heisenberg sector carries no Kramers theorem
    result = solve_adaptive(terms, seed, 3, bases=basis, n_roots=1, max_bond=8,
                            policy=ReconnectionPolicy(rule="entropy"),
                            rng=np.random.default_rng(14), boundary_check=0,
                            on_split="warn")
    assert result.energies[0] == pytest.approx(-0.75 * sum(js), abs=1e-9)
    report = result.structure
    assert sorted(s.orbitals for s in report.sites) == [p for p in pairs]
    assert len(report.weak_edges) == 2
    for e in report.weak_edges:
        assert report.bonds[e].entropy_nats < 1e-8
        assert report.bonds[e].eff_dim == 1
    for site in report.sites:
        assert site.internal_max_entropy == pytest.approx(np.log(2.0), abs=1e-6)


def test_structure_report_is_purely_diagnostic():
    """The report re-gauges the state exactly: energies before and after agree."""
    n, k = 5, 2
    h, eri = random_spinor_integrals(n, seed=48)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(op, k, 32, rng=np.random.default_rng(15))
    first = solve_ttn(op, state, max_bond=32, boundary_check=0)
    discovered_structure(state, weights=first.weights)
    second = solve_ttn(op, state, max_bond=32, boundary_check=0)
    assert abs(first.energies[0] - second.energies[0]) < E_TOL
