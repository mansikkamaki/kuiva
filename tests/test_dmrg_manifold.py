"""Tier-0 tests for the local-multiplet manifold machinery.

Two sharp oracles, both independent of the machinery under test:

* **Non-interacting fragments with non-degenerate internal spectra** (the two-fragment oracle
  family — complex hopping breaks every accidental degeneracy on purpose, so root growth
  is deterministic): the effective spectrum must equal the *analytic* pairwise sums of the
  fragment levels exactly, including model states the network ensemble never solved as
  roots — which is the entire point of the construction.
* **Cauchy interlacing**: ``H_eff`` is a Rayleigh–Ritz compression, so its k-th eigenvalue
  bounds the k-th exact eigenvalue (from the independent Slater–Condon CI, or from a
  plain-kron dense ED for spin models) from above — an *exact* statement that holds at any
  coupling, independent of how good the product truncation is.

The gap discipline (refuse a cut through a degenerate group; warn on an ambiguous cut) is
asserted as behaviour, with the guard-must-be-able-to-fail pattern: the warning must fire on an ambiguous
cut *and* stay silent on a clean one.
"""
import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix
from kuiva.dmrg.block import QuantumNumber
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.manifold import (UnderResolved, effective_model, model_gb,
                                 solve_manifold)
from kuiva.dmrg.sweep import random_state, solve_ttn
from kuiva.dmrg.ttno import (ModeBasis, ProductTerm, compile_ttno,
                             hamiltonian_product_terms, one_electron_product_terms)

from test_ci_strings import random_spinor_integrals
from test_ci_davidson import kramers_spinor_integrals

E_TOL = 1e-8


def two_fragments(u=6.0, t_inter=0.0):
    """Two 3-spinor fragments (complex triangle hopping + intra-fragment U), optionally
    bridged by a weak inter-fragment hop. Non-degenerate fragment spectra by design."""
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
    if t_inter:
        h[2, 3] = h[3, 2] = t_inter
    return n, h, eri, frag_a, frag_b


def fragment_sums(h, frag_a, frag_b):
    """The analytic 9-state (1, 1)-electron manifold of the uncoupled fragments."""
    ea = np.linalg.eigvalsh(h[np.ix_(frag_a, frag_a)])
    eb = np.linalg.eigvalsh(h[np.ix_(frag_b, frag_b)])
    return np.sort((ea[:, None] + eb[None, :]).ravel())


def exact_energies(n, k, h, eri, n_roots=None):
    from itertools import combinations
    masks = np.sort(np.array([sum(1 << i for i in c)
                              for c in combinations(range(n), k)], dtype=np.uint64))
    dets = Determinants(masks, n_spinor=n, n_elec=k)
    e = np.sort(np.linalg.eigvalsh(hamiltonian_matrix(dets, h, eri).toarray()))
    return e if n_roots is None else e[:n_roots]


# --- exactness on the product oracle, with root growth ------------------------------------

def test_uncoupled_fragments_model_is_exact_and_ensemble_grows():
    """The headline local-multiplet claim: the 9-state model spectrum is exact although the network
    ensemble solved fewer roots, and the loop grew the ensemble to resolve the sites."""
    n, h, eri, fa, fb = two_fragments()
    result = solve_manifold(hamiltonian_product_terms(h, eri), NetworkGraph.path(n), 2,
                            sites=[fa, fb], rule="dimension", dims=3, n_roots=2,
                            max_roots=9, max_outer=8, rng=np.random.default_rng(1))
    assert result.converged
    spec = result.model.spectrum()
    assert spec.size == 9
    assert np.max(np.abs(spec - fragment_sums(h, fa, fb))) < E_TOL
    assert any(step["action"] == "grow" for step in result.history)
    assert result.n_roots < 9                    # states the ensemble never solved


def test_gap_rule_finds_the_exact_rank():
    n, h, eri, fa, fb = two_fragments()
    ttno = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(ttno, 2, 10 ** 9, n_roots=4, rng=np.random.default_rng(2))
    sweep = solve_ttn(ttno, state, boundary_check=0)
    model = effective_model(ttno, state, [fa, fb], weights=sweep.weights, rule="gap",
                            report=False)
    assert model.dims == (3, 3)                  # the exact-rank cut (ratio = inf)
    assert all(not np.isfinite(sp.gap_ratio) for sp in model.sites)
    assert np.max(np.abs(model.spectrum() - fragment_sums(h, fa, fb))) < E_TOL


def test_growth_cap_raises_under_resolved_with_the_knob_named():
    n, h, eri, fa, fb = two_fragments()
    with pytest.raises(UnderResolved, match="max_roots"):
        solve_manifold(hamiltonian_product_terms(h, eri), NetworkGraph.path(n), 2,
                       sites=[fa, fb], rule="dimension", dims=3, n_roots=2, max_roots=2,
                       rng=np.random.default_rng(3))


def test_three_fragment_model_is_exact_triple_product():
    """Three uncoupled fragments: the K = 3 quotient-tree contraction and the 27-state
    triple-product model, exact against the analytic level sums. This is the Tier-0
    stand-in for the ab initio trimer rung, whose TTNO does not fit the dev box yet
    (what the sparse W storage of :mod:`kuiva.dmrg.sparse` addressed)."""
    n = 9
    frags = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    h = np.zeros((n, n), dtype=np.complex128)
    eri = np.zeros((n, n, n, n), dtype=np.complex128)
    for frag, eps in zip(frags, (0.0, 0.05, 0.11)):
        for i, p in enumerate(frag):
            h[p, p] = eps + 0.1 * i
            for q in frag:
                if p < q:
                    h[p, q] = -1.0 + 0.1j
                    h[q, p] = -1.0 - 0.1j
                if p != q:
                    eri[p, p, q, q] = 6.0
    levels = [np.linalg.eigvalsh(h[np.ix_(f, f)]) for f in frags]
    sums = levels[0]
    for lv in levels[1:]:
        sums = (sums[:, None] + lv[None, :]).ravel()
    sums = np.sort(sums)

    # one 3-mode node per fragment: a single-mode path's END bond caps the shared-basis
    # ensemble at its own two-site dimension (6 here), which root growth must exceed
    graph = NetworkGraph(3, [(0, 1), (1, 2)], contents=list(frags))
    # on_split="warn": 3 electrons is odd, but these integrals are not time-reversal
    # symmetric, so the Kramers refusal would rightly reject the singleton blocks
    result = solve_manifold(hamiltonian_product_terms(h, eri), graph, 3,
                            sites=[(0,), (1,), (2,)], rule="dimension", dims=3,
                            n_roots=4, max_roots=27, max_outer=9, on_split="warn",
                            rng=np.random.default_rng(17))
    spec = result.model.spectrum()
    assert spec.size == 27
    assert np.max(np.abs(spec - sums)) < E_TOL
    assert result.n_roots < 27                   # states the ensemble never solved


# --- interlacing and mixed charge sectors at real coupling --------------------------------

def test_coupled_fragments_interlace_and_mix_charge_sectors():
    """With a real bridge the site spaces gain N = 0/2 sectors; the physical-sector
    machinery filters them, and every model eigenvalue bounds its exact counterpart from
    above (Cauchy interlacing — exact at any coupling)."""
    n, h, eri, fa, fb = two_fragments(t_inter=0.05)
    ttno = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(ttno, 2, 10 ** 9, n_roots=2, rng=np.random.default_rng(4))
    sweep = solve_ttn(ttno, state, boundary_check=0)
    model = effective_model(ttno, state, [fa, fb], weights=sweep.weights, rule="weight",
                            weight_tol=1e-9, report=False)
    assert any(sp.n_electrons is None for sp in model.sites)       # mixed N kept
    assert model.sector().size < model.model_dim                   # filtered out
    exact = exact_energies(n, 2, h, eri)
    spec = model.spectrum()
    assert np.all(spec >= exact[:spec.size] - E_TOL)               # interlacing
    # the 2-root ensemble resolves THREE low states — one more than it solved, through
    # the inter-site entanglement the bridge creates (the entanglement mechanism); the 4th exact
    # state involves a fragment level the ensemble never populated and is honestly absent
    assert np.max(np.abs(spec[:3] - exact[:3])) < 1e-5             # measured <= 5e-7


def test_untruncated_model_reproduces_the_exact_ci():
    """With nothing truncated the model space *is* the CI space: the sector spectrum must
    equal the independent Slater–Condon reference to the 1e-8 Eh tolerance."""
    n, k = 4, 2
    h, eri = random_spinor_integrals(n, seed=31)
    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1), (2, 3)])
    ttno = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = random_state(ttno, k, 10 ** 9, n_roots=6, rng=np.random.default_rng(5))
    sweep = solve_ttn(ttno, state, boundary_check=0)
    model = effective_model(ttno, state, [(0,), (1,)], weights=sweep.weights,
                            rule="weight", weight_tol=0.0, report=False)
    ref = exact_energies(n, k, h, eri)
    spec = model.spectrum()
    assert spec.size == ref.size
    assert np.max(np.abs(spec - ref)) < E_TOL


# --- the gap discipline (refuse, warn, stay silent) --------------------------------

def test_a_cut_through_a_kramers_pair_is_refused():
    h, eri = kramers_spinor_integrals(3, seed=16)                  # 6 spinors, TR symmetric
    ttno = compile_ttno(NetworkGraph.path(6), hamiltonian_product_terms(h, eri))
    state = random_state(ttno, 1, 10 ** 9, n_roots=2, rng=np.random.default_rng(6))
    sweep = solve_ttn(ttno, state, boundary_check=0)
    with pytest.raises(ValueError, match="degenerate group"):
        effective_model(ttno, state, [tuple(range(6))], weights=sweep.weights,
                        rule="dimension", dims=1, report=False)
    model = effective_model(ttno, state, [tuple(range(6))], weights=sweep.weights,
                            rule="dimension", dims=2, report=False)
    assert model.dims == (2,)


def heisenberg_pair_model(j_inter):
    """Four spins, sites (0, 1) and (2, 3): ferromagnetic intra-site coupling makes each
    site a triplet (S = 1); ``j_inter`` couples the sites antiferromagnetically."""
    sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    sm = sp.T.copy()
    basis = ModeBasis(2, (QuantumNumber(0), QuantumNumber(1)))
    terms = []
    for i, k, j in ((0, 1, -1.0), (2, 3, -1.0), (1, 2, j_inter)):
        terms.append(ProductTerm(0.5 * j, (i, k), (sp, sm)))
        terms.append(ProductTerm(0.5 * j, (i, k), (sm, sp)))
        terms.append(ProductTerm(j, (i, k), (sz, sz)))
    return terms, basis


def dense_heisenberg(bonds, n_spins=4):
    """Independent plain-kron dense ED of the same spin model (no TTNO involved)."""
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def site_op(op, i):
        m = np.eye(1, dtype=np.complex128)
        for s in range(n_spins):
            m = np.kron(m, op if s == i else np.eye(2))
        return m

    h = np.zeros((2 ** n_spins, 2 ** n_spins), dtype=np.complex128)
    for i, k, j in bonds:
        for op in (sx, sy, sz):
            h += j * (site_op(op, i) @ site_op(op, k))
    return np.sort(np.linalg.eigvalsh(h))


def _pair_model(j_inter, seed):
    terms, basis = heisenberg_pair_model(j_inter)
    graph = NetworkGraph.path(4)
    ttno = compile_ttno(graph, terms, bases=basis)
    state = random_state(ttno, 2, 10 ** 9, n_roots=1, rng=np.random.default_rng(seed))
    sweep = solve_ttn(ttno, state, boundary_check=0, on_split="warn")
    return ttno, state, sweep


def test_gap_warning_fires_on_an_ambiguous_cut_and_not_on_a_clean_one(kuiva_caplog):
    """Guard pattern: at strong inter-site coupling the singlet admixture sits close
    under the triplet and the d = 3 cut is ambiguous (warns); at weak coupling it is
    orders of magnitude down (silent)."""
    ttno, state, sweep = _pair_model(0.5, seed=7)
    effective_model(ttno, state, [(0, 1), (2, 3)], weights=sweep.weights,
                    rule="dimension", dims=3, report=False)
    assert any("ambiguous" in r.message for r in kuiva_caplog.records)

    kuiva_caplog.clear()
    ttno, state, sweep = _pair_model(0.02, seed=8)
    effective_model(ttno, state, [(0, 1), (2, 3)], weights=sweep.weights,
                    rule="dimension", dims=3, report=False)
    assert not any("ambiguous" in r.message for r in kuiva_caplog.records)


# --- composite spin sites: Clebsch-Gordan structure vs independent dense ED ----------------

def test_composite_spin_sites_reproduce_clebsch_gordan_structure():
    """One averaged root resolves both site triplets (inter-site exchange spreads each
    site over its multiplet — the production mechanism), and the 9-state model
    reproduces the S = 1 (x) S = 1 exchange ladder: degeneracies 1 + 3 + 5 and the
    Heisenberg ratio, cross-checked against a plain-kron dense ED."""
    from kuiva.props.multiplet import degenerate_blocks as blocks_cm

    j = 0.02
    ttno, state, sweep = _pair_model(j, seed=9)
    model = effective_model(ttno, state, [(0, 1), (2, 3)], weights=sweep.weights,
                            rule="dimension", dims=3, report=False)
    # each site space is the full triplet, necessarily mixing the magnetization label
    assert model.dims == (3, 3)
    assert all(sp.n_electrons is None for sp in model.sites)

    spec = np.sort(model.spectrum(sector="all"))
    rel = (spec - spec[0]) / j
    sizes = [b for _, b in blocks_cm(rel, tol_cm=1e-3 / j)]
    assert sizes == [1, 3, 5]                                      # 0 + 1 + 2 = 1 (x) 1
    e1 = float(np.mean(rel[1:4]))
    e2 = float(np.mean(rel[4:9]))
    assert e2 / e1 == pytest.approx(3.0, abs=0.05)                 # E ~ S(S+1)/2

    exact = dense_heisenberg(((0, 1, -1.0), (2, 3, -1.0), (1, 2, j)))
    assert np.all(spec >= exact[:9] - E_TOL)                       # interlacing
    assert np.max(np.abs(spec - exact[:9])) < 5e-4                 # low manifold, O(j^2/J)


# --- site-local operators and the JW contiguity trap ---------------------------------------

def test_site_operators_are_extracted_and_verified():
    n, h, eri, fa, fb = two_fragments()
    number_a = np.zeros((n, n), dtype=np.complex128)
    for p in fa:
        number_a[p, p] = 1.0
    ttno = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(ttno, 2, 10 ** 9, n_roots=4, rng=np.random.default_rng(10))
    sweep = solve_ttn(ttno, state, boundary_check=0)
    model = effective_model(ttno, state, [fa, fb], weights=sweep.weights,
                            rule="dimension", dims=3,
                            operators={"n_A": one_electron_product_terms(number_a)},
                            report=False)
    # every kept fragment-A state carries exactly one electron on A
    assert np.allclose(model.site_operators["n_A"][0], np.eye(3), atol=1e-10)
    assert np.allclose(model.site_operators["n_A"][1], 0.0, atol=1e-10)
    assert np.allclose(model.operators["n_A"], np.eye(9), atol=1e-10)


def test_site_operator_refuses_non_contiguous_site_labels():
    """The reconnection JW lesson in operator form: with interleaved mode labels a site-local
    ``a+_p a_q`` drags a Jordan-Wigner string through the other site, the model matrix is
    not ``1 (x) A (x) 1``, and the verification must refuse rather than return it."""
    n = 6
    frag_a, frag_b = (0, 2, 4), (1, 3, 5)                          # interleaved labels
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
                    eri[p, p, q, q] = 6.0
    hop_a = np.zeros((n, n), dtype=np.complex128)
    hop_a[0, 4] = hop_a[4, 0] = 1.0                                # JW string crosses B
    graph = NetworkGraph(n, [(i, i + 1) for i in range(n - 1)],
                         contents=[(0,), (2,), (4,), (1,), (3,), (5,)])
    ttno = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = random_state(ttno, 2, 10 ** 9, n_roots=4, rng=np.random.default_rng(11))
    sweep = solve_ttn(ttno, state, boundary_check=0)
    with pytest.raises(ValueError, match="Jordan-Wigner"):
        effective_model(ttno, state, [(0, 1, 2), (3, 4, 5)], weights=sweep.weights,
                        rule="dimension", dims=3,
                        operators={"hop": one_electron_product_terms(hop_a)},
                        report=False)


# --- structure ------------------------------------------------------------------------------

def test_sizing_function_matches_real_allocation():
    """Two-sided pin of the model sizing function against a real array's nbytes."""
    n, h, eri, fa, fb = two_fragments()
    ttno = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(ttno, 2, 10 ** 9, n_roots=4, rng=np.random.default_rng(12))
    sweep = solve_ttn(ttno, state, boundary_check=0)
    model = effective_model(ttno, state, [fa, fb], weights=sweep.weights,
                            rule="dimension", dims=3, report=False)
    assert model_gb(model.sites) * 1024.0 ** 3 == model.h_eff.nbytes


def test_sites_must_partition_the_tree():
    n, h, eri, fa, fb = two_fragments()
    ttno = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    state = random_state(ttno, 2, 10 ** 9, n_roots=2, rng=np.random.default_rng(13))
    solve_ttn(ttno, state, boundary_check=0)
    with pytest.raises(ValueError, match="partition"):
        effective_model(ttno, state, [(0, 1), (3, 4, 5)], report=False)
